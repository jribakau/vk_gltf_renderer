/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Vulkan ray tracing path tracer renderer. Builds and dispatches the
// ray generation / closest-hit / any-hit / miss shader pipeline for
// progressive path tracing of glTF scenes. Handles accumulation,
// depth-of-field, and denoiser integration (OptiX / OIDN / DLSS).
//

#include <fmt/format.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <span>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include <nvaftermath/aftermath.hpp>

#include <nvapp/elem_dbgprintf.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/specialization.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvgui/tooltip.hpp>

#include "renderer_pathtracer.hpp"
#include "sample_schedule.hpp"
#include "../shaders/adaptive_stop.h.slang"  // adaptiveSceneThreshold
#include "adaptive_relax.hpp"

#include "scene_shader_macros.hpp"
#include "pipeline_build_scope.hpp"
#include "ui_helpers.hpp"
#include "ui_linear_color.hpp"

// Pre-compiled shaders
#include "_autogen/gltf_pathtrace.slang.h"
#if defined(USE_DLSS)
// The same path tracer built with USE_DLSS_SHADER=1 (CMakeLists.txt): what a DLSS frame asks for.
#include "_autogen/gltf_pathtrace.slang_dlss.h"
#endif

// Toggle deferred host operations for RTX pipeline creation.
// 1 = use VkDeferredOperationKHR to parallelize SPIR-V->ISA compilation across worker threads.
// 0 = create the pipeline synchronously on the calling thread (useful for comparing compile
//     times or as a workaround if the driver misbehaves with deferred ops).
// NOTE: the NVIDIA driver currently does not populate VkPipelineCreationFeedback when a
// deferred op is used, so the cache-hit flag is only meaningful with USE_DEFERRED_RTX_COMPILE=0.
#ifndef USE_DEFERRED_RTX_COMPILE
#define USE_DEFERRED_RTX_COMPILE 0
#endif

namespace {
// The path-trace profiler sections. Named once: cmdFrameSection() and getFrameTimerInfo() take a
// std::string, and both are called every frame.
const std::string kTimerPathTraceRq  = "Path Trace (RQ)";
const std::string kTimerPathTraceRtx = "Path Trace (RTX)";

// The RTX pipeline's shader stages and groups. createRtxPipeline() and the warm-up both build from
// it, so they ask the driver for the same pipeline (a different one would only miss the cache).
struct RtxStages
{
  enum Stage
  {
    eRaygen,
    eMiss,
    eShadowMiss,
    eClosestHit,
    eAnyHit,
    eShadowAnyHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  std::vector<VkRayTracingShaderGroupCreateInfoKHR>             groups;

  explicit RtxStages(VkShaderModule module)
  {
    stages[eRaygen]     = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_RAYGEN_BIT_KHR, module, "rgenMain"};
    stages[eMiss]         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR, module, "rmissMain"};
    stages[eClosestHit]   = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, module, "rchitMain"};
    stages[eAnyHit]       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, module, "rahitMain"};
    stages[eShadowMiss]   = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR, module, "rmissShadow"};
    stages[eShadowAnyHit] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, module, "rahitShadow"};

    VkRayTracingShaderGroupCreateInfoKHR group{
        .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .generalShader      = VK_SHADER_UNUSED_KHR,
        .closestHitShader   = VK_SHADER_UNUSED_KHR,
        .anyHitShader       = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR,
    };
    // Raygen, Miss-0, Shadow Miss-1
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    for(uint32_t general : {uint32_t(eRaygen), uint32_t(eMiss), uint32_t(eShadowMiss)})
    {
      group.generalShader = general;
      groups.push_back(group);
    }
    // Hit Group-0
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.anyHitShader     = eAnyHit;
    groups.push_back(group);
    // Hit Group-1 (shadow): any-hit only -- shadow rays skip closest-hit (TraceShadow)
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader     = eShadowAnyHit;
    groups.push_back(group);
  }
};

VkPipelineCreateFlags rtxPipelineFlags(bool opacityMicromaps, bool captureStatistics)
{
  // Allow the pipeline to use opacity micromaps (EXT_mesh_opacity_micromap); harmless when
  // geometry is not carrying a micromap. Only set when the device supports VK_KHR_opacity_micromap.
  VkPipelineCreateFlags flags = opacityMicromaps ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_KHR) : VkPipelineCreateFlags(0);
  if(captureStatistics)
    flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  return flags;
}

// A path-tracer pipeline from `module`, as createRqPipeline() or createRtxPipeline() builds it (no
// deferred operation): any thread, as it touches only the device and the pipeline cache.
// VK_NULL_HANDLE when the creation failed (`result` says why).
VkPipeline createPathTracePipeline(VkDevice              device,
                                   VkPipelineCache       cache,
                                   VkPipelineLayout      layout,
                                   VkShaderModule        module,
                                   bool                  rayQuery,
                                   int32_t               useSER,
                                   int32_t               transparency,
                                   VkPipelineCreateFlags rtxFlags,
                                   VkPipelineCreateFlags rqFlags,
                                   VkResult&             result)
{
  nvvk::Specialization specialization;
  specialization.add(0, useSER);  // USE_SER
#if defined(USE_DLSS)
  specialization.add(1, transparency);  // USE_DLSS_TRANSP
#else
  (void)transparency;
#endif
  VkPipeline pipeline = VK_NULL_HANDLE;
  if(rayQuery)
  {
    const VkComputePipelineCreateInfo info{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags  = rqFlags,
        .stage  = {.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module              = module,
                   .pName               = "computeMain",
                   .pSpecializationInfo = specialization.getSpecializationInfo()},
        .layout = layout,
    };
    result = vkCreateComputePipelines(device, cache, 1, &info, nullptr, &pipeline);
  }
  else
  {
    RtxStages table(module);
    table.stages[RtxStages::eRaygen].pSpecializationInfo = specialization.getSpecializationInfo();
    const VkRayTracingPipelineCreateInfoKHR info{
        .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .flags                        = rtxFlags,
        .stageCount                   = static_cast<uint32_t>(table.stages.size()),
        .pStages                      = table.stages.data(),
        .groupCount                   = static_cast<uint32_t>(table.groups.size()),
        .pGroups                      = table.groups.data(),
        .maxPipelineRayRecursionDepth = 2,
        .layout                       = layout,
    };
    result = vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, cache, 1, &info, nullptr, &pipeline);
  }
  if(result != VK_SUCCESS)
  {
    vkDestroyPipeline(device, pipeline, nullptr);
    return VK_NULL_HANDLE;
  }
  return pipeline;
}
}  // namespace

PathTracer::PathTracer()
{
  // Default parameters for overall material
  m_maxDepth                        = 5;
  m_pushConst.frameCount            = 0;
  m_pushConst.fireflyClampThreshold = 10.;
  m_pushConst.numSamples            = 1;  // Number of samples per pixel

#if defined(USE_DLSS)
  m_dlss = std::make_unique<Dlss>(Dlss::Kind::RR);
#endif

#if defined(USE_OPTIX_DENOISER)
  m_optix = std::make_unique<OptiXDenoiser>();
#endif

#if defined(USE_OIDN_DENOISER)
  m_oidn = std::make_unique<OidnDenoiser>();
#endif
}

//--------------------------------------------------------------------------------------------------
// Constructor
// Initialize the device, the frame counter and the firefly clamp threshold
void PathTracer::onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler)
{
  ::BaseRenderer::onAttach(resources, profiler);
  m_device = resources.allocator.getDevice();

  // Pipeline cache and compiled SPIR-V. An empty --cacheDir keeps both next to the executable,
  // which is where a stand-alone run has always written pipeline_cache.bin.
  {
    const std::filesystem::path cacheRoot = m_cacheDir.empty() ? nvutils::getExecutablePath().parent_path() : m_cacheDir;
    std::error_code             cacheEc;
    std::filesystem::create_directories(cacheRoot, cacheEc);
    if(cacheEc)
      LOGW("PathTracer: cache directory '%s' was not created (%s); disk caches will fail to write\n",
           nvutils::utf8FromPath(cacheRoot).c_str(), cacheEc.message().c_str());
    m_pipelineCache.init(m_device, cacheRoot / "pipeline_cache.bin");
    m_buildRecord.load(cacheRoot / "pipeline_builds.txt");
    if(!cacheEc)
    {
      m_shaderCacheDir = cacheRoot / "shaders";
      std::error_code shaderEc;
      std::filesystem::create_directories(m_shaderCacheDir, shaderEc);
      if(shaderEc)
      {
        LOGW("PathTracer: shader cache directory '%s' was not created (%s); SPIR-V disk cache is off\n",
             nvutils::utf8FromPath(m_shaderCacheDir).c_str(), shaderEc.message().c_str());
        m_shaderCacheDir.clear();
      }
      pruneShaderCache();
    }
  }

  // Written by probeCameraMedium() before any path reads it
  NVVK_CHECK(resources.allocator.createBuffer(m_cameraMediumBuffer, sizeof(shaderio::CameraMedium),
                                              VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
  NVVK_DBG_NAME(m_cameraMediumBuffer.buffer);

  // The path tracer writes depth as a storage image. nvvk::RenderTarget only gives the G-buffer depth
  // STORAGE usage when the format supports it, which D32_SFLOAT does not on every GPU (AMD); there the
  // shader writes an R32_SFLOAT image instead, so the binding stays valid, and a copy brings it into the
  // depth buffer. That copy reinterprets 32-bit float depth only; other formats get no path-traced depth.
  {
    const VkFormat     depthFormat = resources.gBuffers.getDepthFormat();
    VkFormatProperties depthProps{};
    vkGetPhysicalDeviceFormatProperties(resources.allocator.getPhysicalDevice(), depthFormat, &depthProps);
    const bool storageDepth = (depthProps.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
    const bool floatDepth   = depthFormat == VK_FORMAT_D32_SFLOAT || depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT;
    m_depthViaStaging       = !storageDepth;
    m_depthCopyable         = floatDepth;
    if(!storageDepth && !floatDepth)
      LOGW("PathTracer: depth format %d is neither a storage image nor 32-bit float; path-traced depth is not written\n",
           int(depthFormat));
    if(m_depthViaStaging)
      NVVK_CHECK(m_depthStaging.init(
          {.device = m_device, .alloc = &resources.allocator, .colorFormats = {VK_FORMAT_R32_SFLOAT}, .debugName = "PathTracer-DepthStaging"}));
  }
  NVVK_CHECK(m_convergence.init(
      {.device = m_device, .alloc = &resources.allocator, .colorFormats = {VK_FORMAT_R32G32_SFLOAT}, .debugName = "PathTracer-Convergence"}));


  // Requesting ray tracing properties
  // The SER properties are only chained (and trusted) when the extension is enabled: a driver
  // without it may still report a reordering hint, and compiling SER in would then make every
  // path-tracer pipeline fail to create.
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext                  = &m_rtPipelineProperties;
  m_rtPipelineProperties.pNext = resources.settings.serHardwareAvailable ? &m_reorderProperties : nullptr;
  vkGetPhysicalDeviceProperties2(resources.allocator.getPhysicalDevice(), &prop2);
  m_driverVersion = prop2.properties.driverVersion;

  m_supportSER = resources.settings.serHardwareAvailable
                 && (m_reorderProperties.rayTracingInvocationReorderReorderingHint & VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT);

  // AMD's driver crashes creating the compute (ray-query) pipeline from this module, which also
  // holds the ray-tracing entry points; the ray-tracing pipeline works. Keep the path tracer on it.
  constexpr uint32_t kAmdVendorId = 0x1002;
  m_supportRayQuery               = prop2.properties.vendorID != kAmdVendorId;
  m_capturePipelineStats          = false;
  for(size_t i = 0; i < resources.deviceExtensions.size(); i++)
  {
    if(resources.deviceExtensions[i] == VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)
      m_capturePipelineStats = true;
  }
  if(!m_supportRayQuery)
    LOGI("PathTracer: the Ray Query technique is not supported with this driver; using the Ray Tracing Pipeline\n");
  // Keep whatever --ptUseSER asked for (the parameter is parsed before this runs); the device
  // capability only ever turns it off.
  m_useSER         = m_useSER && m_supportSER;
  m_pipelineUseSER = m_useSER;  // no pipelines exist yet; keep the first frame from invalidating

  // The embedded build has AVAILABLE_SER=1: a device without SER compiles from source instead
  compileShader(resources, !m_supportSER);

  // #DLSS - Fast initialization: create GBuffers if hardware available
#if defined(USE_DLSS)
  m_dlss->init(resources);
#endif

  // #OPTIX - Create the OptiX denoiser
#if defined(USE_OPTIX_DENOISER)
  m_optix->init(resources);
#endif

  // #OIDN - Create the vk_oidn context (compute pipelines); the network follows the size
#if defined(USE_OIDN_DENOISER)
  m_oidn->init(resources);
#endif

  m_handoff.init(resources);
}

//--------------------------------------------------------------------------------------------------
// Register command line parameters for the PathTracer
void PathTracer::registerParameters(SettingsRegistry* settings, std::function<void()> resetAccumulation)
{
  // A setting that changes what a frame renders must restart the accumulation when it is set from the
  // command line, a benchmark sequence or MCP: otherwise new frames are averaged into the old image, and
  // once the frame count has reached maxFrames nothing renders at all.
  const nvutils::ParameterBase::CallbackSuccess reset = [resetAccumulation](const nvutils::ParameterBase* const) {
    if(resetAccumulation)
      resetAccumulation();
  };
  // PathTracer-specific settings. Persist marks what survives a restart (see settings_registry.hpp).
  // Bounds mirror the UI sliders so CLI and benchmark sequences accept the same range the UI does:
  //   ptMaxDepth  -> "Max Depth" slider in showUI() (0..20)
  //   ptSamples   -> "Samples per Frame" slider in onUIRender() (MIN_SAMPLES_PER_PIXEL..MAX_SAMPLES_PER_PIXEL);
  //                  the adaptive-sampling autopilot also clamps to the same range at runtime.
  settings->add({.name = "ptMaxDepth", .help = "PathTracer: Maximum ray depth", .callbackSuccess = reset}, &m_maxDepth, Persist::eYes, 0, 20);
  settings->add({.name            = "ptMaxDepthDlss",
                 .help            = "PathTracer: DLSS frames: the bounces a path takes, when below ptMaxDepth (0: ptMaxDepth). "
                                    "Biased: the light past it is lost while the view moves; the still keeps ptMaxDepth",
                 .callbackSuccess = reset},
                &m_maxDepthDlss, Persist::eNo, 0, 20);
  settings->add({.name            = "ptSamples",
                 .help            = "PathTracer: Samples per pixel traced each frame (see ptMaxSamples for the total)",
                 .callbackSuccess = [this](const nvutils::ParameterBase* const) { m_adaptiveSampling = false; }},
                &m_pushConst.numSamples, Persist::eYes, MIN_SAMPLES_PER_PIXEL, MAX_SAMPLES_PER_PIXEL);
  // An accumulation's frames clamp at ptFireflyClampStill when it is set, so a new ptFireflyClamp (the
  // host sends one with every exposure change) restarts only an accumulation that reads it; DLSS frames
  // pick it up as they come (Unreal keeps a still across small exposure changes too)
  settings->add({.name            = "ptFireflyClamp",
                 .help            = "PathTracer: Firefly clamp threshold",
                 .callbackSuccess = [this, reset](const nvutils::ParameterBase* const p) {
                   if(m_pushConst.fireflyClampStill <= 0.0F)
                     reset(p);
                 }},
                &m_pushConst.fireflyClampThreshold, Persist::eYes);
  settings->add({.name            = "ptFireflyClampStill",
                 .help            = "PathTracer: Firefly clamp threshold for the frames that accumulate a still (0: ptFireflyClamp)",
                 .callbackSuccess = reset},
                &m_pushConst.fireflyClampStill, Persist::eYes);
  // Not persisted: a different value only changes the noise pattern, which nobody wants carried into the next session.
  settings->add({"ptSeed",
                 "PathTracer: Random sequence (0 = default); another value gives statistically independent noise, "
                 "e.g. for a reference image compared against a test render"},
                &m_pushConst.seed, Persist::eNo);
  settings->add({.name = "ptTexGradScale", .help = "PathTracer: Ray-footprint gradient scale (0=mip0, 1=physical)", .callbackSuccess = reset},
                &m_pushConst.texGradScale, Persist::eYes, 0.0F, 1.0F);
  settings->add({.name = "ptAperture", .help = "PathTracer: Camera aperture", .callbackSuccess = reset}, &m_pushConst.aperture, Persist::eYes, 0.0F, 1.0F);
  settings->add({.name = "ptFocalDistance", .help = "PathTracer: Focal distance", .callbackSuccess = reset}, &m_pushConst.focalDistance, Persist::eYes);
  settings->add({.name            = "ptShadowTransmission",
                 .help            = "PathTracer: Let shadow rays pass through transmissive surfaces (biased approximation)",
                 .callbackSuccess = reset},
                &m_shadowTransmission, Persist::eYes);
  // Not persisted: the viewer turns it on while the camera moves and off once the view is still.
  settings->add({"dlssHandoffDelayMs",
                 "Denoisers: With DLSS-RR and OptiX or OIDN, milliseconds the view must have been still before the "
                 "accumulation takes over from DLSS (0: the frame after it stops)"},
                &m_dlssHandoffDelayMs, Persist::eYes, 0, 5000);
  settings->add({.name            = "ptCameraMedium",
                 .help            = "PathTracer: Probe which medium the camera is in when an accumulation starts [Off:0, On:1]. "
                                    "Off, camera paths start in clear air",
                 .callbackSuccess = reset},
                &m_probeCameraMedium, Persist::eYes);
  settings->add({.name            = "scatterAsThin",
                 .help            = "PathTracer: Shade volumetric scatter as the thin-walled preview [Off:0, On:1, "
                                    "On the frames DLSS renders and off on the frames an accumulation traces:2]",
                 .callbackSuccess = reset},
                &m_scatterAsThin, Persist::eNo, 0, 2);
  // Dense subsurface (skin): next-event estimation at interior scatter points, straight through the
  // boundary, counts light that would be internally reflected and keeps counting after it is, so a
  // dense medium returns more light than it gets, from a handful of near-surface events (grainy).
  // With this flag a walk gathers nothing inside and leaves the surface as a Lambertian point,
  // shaded like any surface (the random walk of production renderers).
  settings->add({.name            = "ptVolumeDiffuseExit",
                 .help            = "PathTracer: Scatter walks gather light only where they leave the surface, as a diffuse exit (dense media)",
                 .callbackSuccess = reset},
                &m_volumeDiffuseExit, Persist::eYes);
  // The material model's closure, phase M2 of the app's docs/material-model-plan.md. A specialization
  // constant: the pipelines are built again, cached after the first time, and no shader recompiles.
  settings->add({.name            = "ptMaterialModel",
                 .help            = "PathTracer: The closure surfaces shade through. 0: today's glTF lobes, 1: OpenPBR",
                 .callbackSuccess = reset},
                &m_materialModel, Persist::eYes);
  // The walk's way in to match its way out: through a Lambertian lobe, as Unreal's path tracer
  // enters subsurface (and Iray's translucency is a diffuse transmission), not by a rough refraction
  // whose direction follows the view.
  settings->add({.name            = "ptDiffuseEntry",
                 .help            = "PathTracer: Light enters a scatter walk through a Lambertian lobe instead of by refraction",
                 .callbackSuccess = reset},
                &m_diffuseEntry, Persist::eYes);
  // The clamp scaled a camera sample's whole path, so one bright bounce cut the direct light of the
  // same path too and a tight clamp dulled every catchlight. Clamping each contribution on its own,
  // and only light that reached the path after a rough bounce, keeps the surfaces seen directly or
  // through a mirror or a cornea as they are (Unreal's path tracer: FPathState::AccumulateRadiance).
  settings->add({.name            = "ptFireflyClampMode",
                 .help            = "PathTracer: What the firefly clamp scales [Whole path:0, Each indirect contribution:1]",
                 .callbackSuccess = reset},
                &m_fireflyClampMode, Persist::eNo, 0, 1);
  // A light sample evaluated one BSDF lobe, picked at random like a BSDF sample: a cornea (transmission 1)
  // picks its transmission 97.5% of the time, which gives nothing for a light on the viewer's side, so
  // a point light's catchlight came in one sample in forty at forty times its value. Every lobe, each by
  // its weight, gives the same mean in every sample. A punctual light has no BSDF-sampled partner, so
  // that alone is unbiased (1); for the environment, the sky's sun and emitters both sides of MIS move
  // to the lobe mixture's pdf together (2; Unreal's path tracer, AddLobeWithMIS). 2 on every vertex cost
  // a third of a still's frame (2026-10-28: 312 -> 214 ms at 0, 278 at 1) for a difference that shows
  // where the path is direct, so 3 keeps it there and evaluates one lobe past a rough bounce.
  settings->add({.name            = "ptFullLobeEval",
                 .help            = "PathTracer: BSDF lobes a light sample evaluates [One, picked at random:0, "
                                    "All for punctual lights:1, All, with the lobe-mixture pdf in MIS:2, "
                                    "2 while the path is direct, else 1:3]",
                 .callbackSuccess = reset},
                &m_fullLobeEval, Persist::eNo, 0, 3);
  settings->add({.name            = "ptApproximateCaustics",
                 .help            = "PathTracer: After a diffuse or glossy bounce, every later lobe is at least as rough as the path "
                                    "so far (biased, fewer fireflies; off: as rough as the roughest material so far)",
                 .callbackSuccess = reset},
                &m_approximateCaustics, Persist::eNo);
  settings->add({.name            = "ptRegularize",
                 .help            = "PathTracer: Later lobes at least as rough as the roughest material so far, against fireflies "
                                    "(biased; never a clear refraction) [Never:0, DLSS frames:1, Always:2]",
                 .callbackSuccess = reset},
                &m_regularize, Persist::eNo, 0, 2);
  settings->add({.name            = "ptLightsVisible",
                 .help            = "PathTracer: The camera sees sized lights (a sphere light, a sun's disc); reflections always do",
                 .callbackSuccess = reset},
                &m_lightsVisible, Persist::eNo);
  settings->add({.name            = "ptExitFresnel",
                 .help            = "PathTracer: A diffuse exit (ptVolumeDiffuseExit) passes light by the boundary's Fresnel "
                                    "transmittance, less at grazing, more head-on, the same in all",
                 .callbackSuccess = reset},
                &m_exitFresnel, Persist::eNo);
  settings->add({.name            = "ptWalkGuiding",
                 .help            = "PathTracer: A scatter walk entered from a surface draws about half its steps back towards "
                                    "it (Dwivedi guiding): the same image, sooner",
                 .callbackSuccess = reset},
                &m_walkGuiding, Persist::eNo);
  settings->add({.name            = "ptLightTree",
                 .help            = "PathTracer: Emissive meshes are picked by how near and how bright they are (a light "
                                    "tree over them); off: by power alone, wherever they are",
                 .callbackSuccess = reset},
                &m_lightTree, Persist::eNo);
  settings->add({.name            = "ptHairBounces",
                 .help            = "PathTracer: Fibre events a path takes through hair before they count as surface bounces (stills)",
                 .callbackSuccess = reset},
                &m_hairBounces, Persist::eNo, 0, 64);
  settings->add({.name            = "ptHairBouncesDlss",
                 .help            = "PathTracer: Fibre events a path takes through hair before they count as surface bounces (DLSS frames)",
                 .callbackSuccess = reset},
                &m_hairBouncesDlss, Persist::eNo, 0, 64);
  settings->add({.name            = "ptHairRouletteDlss",
                 .help            = "PathTracer: DLSS frames: a path continues from a fibre at most with this probability, "
                                    "once it has taken ptHairRouletteFrom fibre events (1: off). Unbiased, noisier",
                 .callbackSuccess = reset},
                &m_hairRouletteDlss, Persist::eNo, 0.05F, 1.0F);
  settings->add({.name            = "ptHairRouletteFrom",
                 .help            = "PathTracer: the fibre events a DLSS path takes before ptHairRouletteDlss applies",
                 .callbackSuccess = reset},
                &m_hairRouletteFrom, Persist::eNo, 0, 64);
  settings->add({.name            = "ptHairCutDlss",
                 .help            = "PathTracer: DLSS frames: a path through hair ends at this fibre event, whose light sample "
                                    "is its last (0: off; negative: the still's paths too, to measure). Biased: the lock's "
                                    "scattered light past it is ptHairCutTail's",
                 .callbackSuccess = reset},
                &m_hairCutDlss, Persist::eNo, -64, 64);
  settings->add({.name            = "ptHairCutTail",
                 .help            = "PathTracer: the light a path cut by ptHairCutDlss would have scattered on, as this "
                                    "times the lock's colour over 4 pi in its last light sample",
                 .callbackSuccess = reset},
                &m_hairCutTail, Persist::eNo, 0.0F, 64.0F);
  settings->add({.name            = "ptHairResponsivity",
                 .help            = "PathTracer: DLSS frames: DLSS-RR's responsivity mask where the first hit is hair, -1 to 1 "
                                    "(0: no mask). How far those pixels trust their history",
                 .callbackSuccess = reset},
                &m_hairResponsivity, Persist::eNo, -1.0F, 1.0F);
  settings->add({.name            = "ptAdaptiveStop",
                 .help            = "PathTracer: a still's pixel stops sampling once its mean has converged (off: every pixel "
                                    "takes every sample, strictly unbiased)",
                 .callbackSuccess = reset},
                &m_adaptiveStop, Persist::eNo);
  settings->add({.name            = "ptAdaptiveThreshold",
                 .help            = "PathTracer: ptAdaptiveStop's bound on a pixel's standard error, times the root of its "
                                    "luminance, in displayed brightness (after the exposure)",
                 .callbackSuccess = reset},
                &m_adaptiveThreshold, Persist::eNo, 0.0001F, 1.0F);
  settings->add({.name            = "ptAdaptiveMin",
                 .help            = "PathTracer: the samples a pixel takes before ptAdaptiveStop can stop it",
                 .callbackSuccess = reset},
                &m_adaptiveMin, Persist::eNo, 2, 100000);
  settings->add({.name            = "ptAdaptiveRelax",
                 .help            = "PathTracer: ptAdaptiveStop lets highlights the tone mapper flattens keep more noise, as "
                                    "much more as the curve hides it (off: the plain rule everywhere)",
                 .callbackSuccess = reset},
                &m_adaptiveRelax, Persist::eNo);
  settings->add({.name            = "ptAdaptiveDone",
                 .help            = "PathTracer: with ptAdaptiveStop, the still ends once at most this share of its pixels "
                                    "still sample (0: only at ptMaxSamples)",
                 .callbackSuccess = reset},
                &m_adaptiveDone, Persist::eNo, 0.0F, 1.0F);
  // Light that reaches skin after a rough bounce is blurred by that bounce: a random walk there costs
  // tens of events for detail nothing shows, so the medium shades as the thin preview's diffuse lobes
  // (Unreal's bSimplifySSS, at path roughness 0.15). Skin seen directly, or through a mirror or a
  // cornea, still walks.
  settings->add({.name            = "ptSimplifyScatter",
                 .help            = "PathTracer: After a rough bounce, a scattering medium shades as its thin preview "
                                    "instead of a random walk",
                 .callbackSuccess = reset},
                &m_simplifyScatter, Persist::eNo);
  settings->add({.name            = "ptSampler",
                 .help            = "PathTracer: The accumulation's random numbers for the pixel, the lens and each "
                                    "vertex's light and BSDF samples: 0=PCG, 1=Owen-scrambled Sobol (default; DLSS frames keep PCG)",
                 .callbackSuccess = reset},
                &m_sampler, Persist::eNo, 0, 1);
  settings->add({"ptAutoFocus", "PathTracer: Enable auto focus"}, &m_autoFocus, Persist::eYes);
  settings->add({"ptTechnique", "PathTracer: Rendering technique of the accumulation [RayQuery:0, RayTracing:1]"},
                (int*)&m_techniqueStill, Persist::eYes, 0, 1);
  settings->add({"ptTechniqueDlss", "PathTracer: Rendering technique of DLSS frames [RayQuery:0, RayTracing:1]"},
                (int*)&m_techniqueDlss, Persist::eYes, 0, 1);
  // SER is a specialization constant of both path-tracing pipelines, so a change is picked up by
  // ensureShadersAndPipelines() on the next frame -- which is what makes it settable mid-run from a
  // benchmark sequence, not just on the command line. Silently ignored when the device lacks SER.
  settings->add({"ptUseSER", "PathTracer: Use Shader Execution Reordering (ignored if unsupported)"}, &m_useSER, Persist::eYes);
  settings->add({"ptAdaptiveSampling", "PathTracer: Enable adaptive sampling"}, &m_adaptiveSampling, Persist::eYes);
  settings->add({.name = "ptPerformanceTarget",
                 .help = "PathTracer: Auto SPP target [Interactive:0, Balanced:1, Quality:2, MaxQuality:3, Throughput:4]. "
                         "Default: Balanced, Throughput in headless runs",
                 .callbackSuccess = [this](const nvutils::ParameterBase* const) { restartAdaptiveSampling(); }},
                (int*)&m_performanceTarget, Persist::eYes, 0, 4);
#if defined(USE_DLSS)
  // dlssEnable switches between RR's one-sample frames and accumulation: the accumulation must not
  // start from the last DLSS input, weighted by every DLSS frame since the last reset.
  m_dlss->registerParameters(settings, resetAccumulation);
#endif

#if defined(USE_OPTIX_DENOISER)
  m_optix->registerParameters(settings);
#endif

#if defined(USE_OIDN_DENOISER)
  m_oidn->registerParameters(settings);
#endif
  m_handoff.registerParameters(settings);
}


//--------------------------------------------------------------------------------------------------
// Destroy the resources
void PathTracer::onDetach(Resources& resources)
{
  // Wait for any background compile to finish before destroying the Vulkan objects it owns.
  if(m_compileThread.joinable())
    m_compileThread.join();
  joinPipelineWarm();
  discardVariantJob();

  resources.allocator.destroyBuffer(m_sbtBuffer);
  resources.allocator.destroyBuffer(m_depthCopyBuffer);
  resources.allocator.destroyBuffer(m_cameraMediumBuffer);
  resources.allocator.destroyBuffer(m_adaptiveActiveBuffer);
  m_adaptiveCounts.clear();
  m_depthStaging.deinit();
  m_convergence.deinit();

#if defined(USE_DLSS)
  m_dlss->deinit(resources);
#endif

#if defined(USE_OPTIX_DENOISER)
  m_optix->deinit(resources);
#endif
#if defined(USE_OIDN_DENOISER)
  m_oidn->deinit(resources);
#endif
  m_handoff.deinit(resources);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyShaderModule(m_device, m_shaderModule, nullptr);
  vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
  vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
  destroyVariantCache(resources);
  m_pipelineCache.deinit();
}

//--------------------------------------------------------------------------------------------------
// Resize the G-Buffer and the renderers
void PathTracer::onResize(VkCommandBuffer cmd, const VkExtent2D& /*size*/, Resources& resources)
{
  updateDepthStaging(cmd, resources);
  updateConvergence(cmd, resources);
  updateDlssResources(cmd, resources);
  updateOptiXResources(cmd, resources);
#if defined(USE_OIDN_DENOISER)
  m_oidn->updateSize(cmd, resources.gBuffers.getSize());
#endif
  m_handoff.onResize(cmd, resources.gBuffers.getSize());
#if defined(USE_DLSS)
  m_dlss->notifyReset();  // viewport size changed -> reseed jitter sequence + drop NGX history
#endif
}

//--------------------------------------------------------------------------------------------------
// Scene was destroyed/replaced; reset per-scene transient state.
void PathTracer::onSceneInvalidated(Resources& /*resources*/)
{
  // a variant that failed to build off the frame may build for the next scene (it may have run out of memory)
  m_failedVariant.reset();
#if defined(USE_DLSS)
  m_dlss->notifyReset();
#endif
}

void PathTracer::notifyDlssContentReset(Resources& /*resources*/)
{
#if defined(USE_DLSS)
  // Material/light appearance change (e.g. KHR_interactivity texture-transform pointer/set) isn't
  // describable by motion vectors - discard DLSS's temporal history so the new content doesn't
  // blend with stale history at the same screen location. See docs/denoising.md.
  m_dlss->notifyReset();
#endif
}

void PathTracer::updateDlssResources(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_DLSS)
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  VkExtent2D size = resources.gBuffers.getSize();
  m_dlss->updateSize(cmd, resources, size);
  m_dlss->setResources();
  m_dlss->setOutputImage(resources.gBuffers.getColorImage(Resources::eImgRendered),
                         resources.gBuffers.getColorAttachmentView(Resources::eImgRendered),
                         resources.gBuffers.getColorFormat(Resources::eImgRendered));
#if defined(USE_DLSSNR)
  m_dlss->setNrImage(resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                     resources.gBuffers.getColorAttachmentView(Resources::eImgTonemapped),
                     resources.gBuffers.getColorFormat(Resources::eImgTonemapped));
#endif
#endif
}

void PathTracer::updateOptiXResources(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_OPTIX_DENOISER)
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  VkExtent2D size = resources.gBuffers.getSize();
  m_optix->updateSize(cmd, size);
#endif
}

bool PathTracer::onUIRender(Resources& resources)
{
  // Setting the aperture max slider value, based on the scene size
  float sceneRadius = resources.gpuView().hasContent() ? resources.gpuView().bounds().radius() : 1.0f;
  float scaleFactor = std::log(sceneRadius);
  scaleFactor       = std::max(scaleFactor, 0.0f);   // Prevent negative values when the scene is small
  float apertureMax = 0.0001f + scaleFactor * 5.0f;  // Minimum max aperture is 0.0001

  namespace PE = nvgui::PropertyEditor;
  bool changed = false;
  if(PE::begin())
  {
    // Rendering technique selector
    const char* techniques[] = {"Compute / Ray Query", "Ray Tracing Pipeline"};
    int         current      = static_cast<int>(m_techniqueStill);
    ImGui::BeginDisabled(!m_supportRayQuery);
    if(PE::Combo("Rendering Pipeline", &current, techniques, IM_ARRAYSIZE(techniques)))
    {
      m_techniqueStill = m_techniqueDlss = static_cast<RenderTechnique>(current);  // one choice for both views here
      changed                           = true;
    }
    ImGui::EndDisabled();
    if(!m_supportRayQuery)
    {
      ImGui::SameLine();
      ImGui::TextDisabled("(Ray Query unsupported on this driver)");
    }
    nvgui::tooltip(
        "Both Ray Query and Ray Tracing use hardware accelerated ray tracing."
        "Ray Query uses a compute shader interface, while Ray Tracing uses the dedicated RTX pipeline.");

    if(m_supportSER && (!m_supportRayQuery || m_techniqueStill == RenderTechnique::RayTracing || m_techniqueDlss == RenderTechnique::RayTracing))
    {
      // The toggle only records the wish; ensureShadersAndPipelines() drops the pipelines built with
      // the old value on the next frame. That path also covers --ptUseSER and the ray-query
      // pipeline, which this checkbox used to leave stale.
      changed |= PE::Checkbox("Use SER", &m_useSER, "Use shader execution reorder");
    }

    // Scene-aware shader optimization.
    PE::Checkbox("Optimize Shader for Current Scene", &resources.settings.optimalShader,
                 "Recompile the shader to match scene features for better performance.\nMay take a few seconds.");


#if !defined(NDEBUG)
    // Status line : which scene features are active (informational)
    if(resources.settings.optimalShader)
    {
      const auto&       fs           = resources.currentFeatureSet;
      const std::string usedFeatures = fs.toString();
      std::string sceneExtTooltip = fmt::format("Active scene feature(s): {}", usedFeatures.empty() ? "none" : usedFeatures);
      nvgui::tooltip(sceneExtTooltip.c_str(), true);
      std::string featureGatesTooltip = fmt::format("Feature gates: Transmission: {}, Volume: {}, Scatter: {}, DLSS Guide: {}",
                                                    fs.has(nvvkgltf::SceneFeatureSet::eTransmission) ? "ON" : "off",
                                                    fs.has(nvvkgltf::SceneFeatureSet::eVolume) ? "ON" : "off",
                                                    fs.has(nvvkgltf::SceneFeatureSet::eScatter) ? "ON" : "off",
                                                    fs.has(nvvkgltf::SceneFeatureSet::eDlssGuide) ? "ON" : "off");
      nvgui::tooltip(featureGatesTooltip.c_str(), true);
    }
#endif

    changed |= PE::SliderInt("Max Depth", &m_maxDepth, 0, 20, "%d", 0, "Maximum number of bounces");
    changed |= PE::SliderFloat("FireFly Clamp", &m_pushConst.fireflyClampThreshold, 0.0f, 100.0f, "%.4g", 0,
                               "Clamp threshold for fireflies. 0 = disabled.");
    changed |= PE::SliderFloat("Texture LOD", &m_pushConst.texGradScale, 0.0f, 1.0f, "%.2f", 0,
                               "Ray-footprint gradient scale for texture LOD.\n"
                               "0 = always mip 0 (default: sharpest, relies on MC accumulation for AA).\n"
                               "1 = full physically-derived LOD (may look soft at distance).");
    changed |= PE::Checkbox("Shadow Transmission", &m_shadowTransmission,
                            "Let shadow rays pass straight through transmissive surfaces (KHR_materials_transmission).\n"
                            "Biased approximation: brightens what lies behind glass but ignores refraction,\n"
                            "so caustics are not focused. When disabled, transmissive surfaces block shadow rays\n"
                            "and light through them comes only from BSDF sampling: no approximation, but caustics\n"
                            "from small or punctual lights (sun, point, spot, directional) are missing.");
    PE::end();
  }

  // Manual sampling controls
  if(PE::begin())
  {
    // Progress is counted in samples per pixel: a frame holds a varying number of them
    const int  maxSamples = std::max(resources.settings.maxSamples, 1);
    const auto progress   = fmt::format("{} / {} spp", m_totalSamplesAccumulated, maxSamples);
    PE::entry(
        "Accumulated",
        [&] {
          ImGui::ProgressBar(std::min(float(m_totalSamplesAccumulated) / float(maxSamples), 1.0F), ImVec2(-FLT_MIN, 0),
                             progress.c_str());
          return false;
        },
        "Samples per pixel in the image, out of Max Samples");
    // Bounds mirror the ptMaxSamples registration in GltfRenderer
    PE::SliderInt("Max Samples", &resources.settings.maxSamples, 1, 1000000, "%d spp", ImGuiSliderFlags_Logarithmic,
                  "Stop accumulating once the image holds this many samples per pixel");
    // Set by hand, or, with Auto SPP or DLSS, showing the count they trace (ptSamples keeps the hand-set one)
    const bool samplesByHand = !m_adaptiveSampling && !isDlssEveryFrame();
    int        shownSamples  = samplesPerFrame();
    ImGui::BeginDisabled(!samplesByHand);
    PE::SliderInt("Samples per Frame", samplesByHand ? &m_pushConst.numSamples : &shownSamples, MIN_SAMPLES_PER_PIXEL,
                  MAX_SAMPLES_PER_PIXEL, "%d", 0,
                  "Samples per pixel traced each frame. A frame traces fewer to land on Max Samples or an "
                  "--outputSamples count");
    ImGui::EndDisabled();
    if(isDlssEveryFrame())
    {
      ImGui::SameLine();
      ImGui::TextDisabled("(DLSS: 1 spp)");
    }

    // Adaptive sampling controls
    ImGui::BeginDisabled(isDlssEveryFrame());
    PE::Checkbox("Auto SPP", &m_adaptiveSampling, "Automatically adjust samples per pixel based on performance target");
    ImGui::EndDisabled();
    if(isDlssEveryFrame())
    {
      ImGui::SameLine();
      ImGui::TextDisabled("(DLSS disabled)");
    }
    if(m_adaptiveSampling)
    {
      ImGui::SameLine();
      if(m_performanceTarget == PerformanceTarget::eThroughput)
        ImGui::TextDisabled(m_throughputSearch.settled() ? "(held: fastest)" : "(searching)");
      else
        ImGui::TextDisabled(m_frameRateControl.held() ? "(held: at target)" : "(climbing)");
      nvgui::tooltip(
          "After each camera move Auto SPP adds one sample per frame until the path-trace GPU time "
          "reaches the target, then holds the count while the view is still. It steps down if frames "
          "run over the target.",
          true);

      // Performance target selection
      const char* targets[] = {"Interactive (60 FPS)", "Balanced (30 FPS)", "Quality (15 FPS)", "Max Quality (10 FPS)",
                               "Max Throughput (offline)"};
      int         currentTarget = static_cast<int>(m_performanceTarget);
      if(PE::Combo("Performance Target", &currentTarget, targets, IM_ARRAYSIZE(targets), -1,
                   "Frame rate Auto SPP aims at. Max Throughput finds the samples per frame with the lowest frame "
                   "time per sample, for renders nobody interacts with (the headless default)"))
      {
        m_performanceTarget = static_cast<PerformanceTarget>(currentTarget);
        restartAdaptiveSampling();
      }
    }
    // Performance info - always visible
    const int   frames      = resources.frameCount + 1;
    const float sppPerFrame = (frames > 0) ? float(m_totalSamplesAccumulated) / float(frames) : 0.f;
    ImGui::TextDisabled("Frames: %d (%.1f spp/frame)", frames, sppPerFrame);
    ImGui::TextDisabled("Throughput: %.2f MSPP/s", m_throughputRollingAvg.getAverage());
    nvgui::tooltip(fmt::format("Mega-sample-pixels per second (rolling average over last {} frames)",
                               m_throughputRollingAvg.SAMPLE_COUNT)
                       .c_str(),
                   true);

    PE::end();
  }

  // Camera controls
  if(PE::begin())
  {
    changed |= PE::SliderFloat("Aperture", &m_pushConst.aperture, 0.0f, apertureMax, "%5.9f",
                               ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat, "Out-of-focus effect");
    changed |= PE::Checkbox("Auto Focus", &m_autoFocus, "Use interest position");
    ImGui::BeginDisabled(m_autoFocus);
    changed |= PE::DragFloat("Focal Distance", &m_pushConst.focalDistance, 100.0f, 0.0f, 1000000.0f, "%5.9f",
                             ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat, "Distance to focal point");
    m_pushConst.focalDistance = std::max(0.000000001f, m_pushConst.focalDistance);
    ImGui::EndDisabled();
    PE::end();
  }

  // Infinite plane
  if(PE::begin())
  {
    changed |= PE::Checkbox("Infinite Plane", (bool*)&resources.settings.useInfinitePlane);
    if(resources.settings.useInfinitePlane)
    {
      changed |= PE::Checkbox("Shadow Catcher", (bool*)&resources.settings.isShadowCatcher);
      const float extentY = resources.gpuView().hasContent() ? resources.gpuView().bounds().extents().y : 10.0f;
      if(PE::treeNode("Infinite Plane Settings"))
      {
        changed |= PE::SliderFloat("Height", &resources.settings.infinitePlaneDistance, -extentY, extentY, "%5.9f",
                                   ImGuiSliderFlags_NoRoundToFormat, "Distance to infinite plane");
        changed |= uicolor::colorEdit3Linear("Color", glm::value_ptr(resources.settings.infinitePlaneBaseColor),
                                             "Infinite-plane base color (shown/edited in linear; swatch/wheel perceptual).");
        changed |= PE::SliderFloat("Metallic", &resources.settings.infinitePlaneMetallic, 0.0f, 1.0f);
        changed |= PE::SliderFloat("Roughness", &resources.settings.infinitePlaneRoughness, 0.0f, 1.0f);
        if(resources.settings.isShadowCatcher)
        {
          changed |= PE::SliderFloat("Shadow Darkness", &resources.settings.shadowCatcherDarkness, 0.0f, 1.0f, "%.2f",
                                     ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat,
                                     "Non-physical: 0=default, 1=deeper shadows");
        }
        PE::treePop();
      }
    }

    PE::end();
  }

  if(ImGui::CollapsingHeader("AI Denoisers", ImGuiTreeNodeFlags_DefaultOpen))
  {
#if defined(USE_DLSS)
    bool dlssChanged = m_dlss->onUiActivation(resources);
#if defined(USE_DLSSNR)
    dlssChanged |= m_dlss->onUiNrActivation();
#endif
#else
    ImGui::TextDisabled("DLSS is not enabled.");
    nvsamples::HelpMarker("Define USE_DLSS in CMake to enable DLSS support.");
#endif

#if defined(USE_OPTIX_DENOISER)
    changed |= m_optix->onUiActivation(resources);
#else
    ImGui::TextDisabled("OptiX Denoiser is not enabled.");
    nvsamples::HelpMarker("Define USE_OPTIX_DENOISER in CMake to enable OptiX denoiser support.");
#endif

#if defined(USE_OIDN_DENOISER)
    changed |= m_oidn->onUiActivation(resources);
#else
    ImGui::TextDisabled("OIDN Denoiser is not enabled.");
    nvsamples::HelpMarker("Set USE_OIDN_DENOISER in CMake to enable the vk_oidn denoiser.");
#endif

    // DLSS-RR with OptiX or OIDN: how DLSS's image hands off to theirs
    if(isDlssEnabled() && isAccumulationDenoiserEnabled())
      m_handoff.onUi();

    ImGui::Spacing();

#if defined(USE_DLSS)
    dlssChanged |= m_dlss->onUiSettings(resources);
#if defined(USE_DLSSNR)
    dlssChanged |= m_dlss->onUiNrSettings();
#endif
    changed |= dlssChanged;
#endif
#if defined(USE_OPTIX_DENOISER)
    changed |= m_optix->onUiSettings(resources);
#endif
#if defined(USE_OIDN_DENOISER)
    changed |= m_oidn->onUiSettings(resources);
#endif

    ImGui::Spacing();

#if defined(USE_OPTIX_DENOISER)
    changed |= m_optix->onUiPreview(resources);
#endif
#if defined(USE_OIDN_DENOISER)
    changed |= m_oidn->onUiPreview(resources);
#endif
#if defined(USE_DLSS)
    changed |= m_dlss->onUiGuideBuffers();
#endif
  }
  return changed;
}

//--------------------------------------------------------------------------------------------------
// What the live shader / pipeline lack for the current settings. Called every rendered frame, so it
// stays cheap: build the wanted key, then one lock and a handful of compares.
PathTracer::PendingCompileWork PathTracer::getPendingCompileWork(const Resources& resources)
{
  const VariantKey wanted = makeVariantKey(resources);
  const bool       useSER = m_useSER && m_supportSER;

  std::lock_guard<std::mutex> lock(m_compileMutex);
  const VkPipeline            live = (m_renderTechnique == RenderTechnique::RayQuery) ? m_rqPipeline : m_rtxPipeline;
  return {
      .recompile       = !wanted.sameShader(m_compiled),
      .stalePipeline   = useSER != m_pipelineUseSER || wanted.dlssTransparency != m_compiled.dlssTransparency,
      .missingPipeline = live == VK_NULL_HANDLE,
  };
}

std::optional<nvvkgltf::SceneFeatureSet> PathTracer::liveShaderFeatures()
{
  std::lock_guard<std::mutex> lock(m_compileMutex);
  if(!m_compiled.optimal || m_compiledIsFallback)
    return std::nullopt;
  return m_compiled.features;
}

bool PathTracer::wantOpacityMicromapShader(const Resources& resources)
{
  return resources.gpuView().hasOpacityMicromaps();
}

//--------------------------------------------------------------------------------------------------
// Ensure shader binaries and the selected technique pipeline are available. Runs on the compile
// worker (or inline when headless); getPendingCompileWork() says what is missing.
void PathTracer::ensureShadersAndPipelines(Resources& resources)
{
  const PipelineBuildScope building;  // /stats says so while this holds the application thread
  m_useSER = m_useSER && m_supportSER;  // a device without SER support can never turn it on

  // A shader macro changed (wireframe, visualization, opacity micromaps, optimal mode and the
  // scene's feature set, DLSS, denoiser guides). compileShader() also destroys the pipelines built from the old shader.
  PendingCompileWork work = getPendingCompileWork(resources);
  if(work.recompile)
  {
    VariantKey compiled;
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      compiled = m_compiled;
    }
    LOGI("[PathTracer] Shader recompile: %s\n", compiled.shaderDiff(makeVariantKey(resources)).c_str());
    if(m_busyWindow)
      m_busyWindow->setReason("Compiling Slang shaders...");
    compileShader(resources);
    work = getPendingCompileWork(resources);
  }

  // SER (--ptUseSER / the UI checkbox) and the DLSS transparency mode (--dlssTransparency / the
  // UI combo) are pipeline specialization constants, not shader macros, so they need no recompile --
  // only the pipelines built with the old values have to go. The SBT is generated from the RTX
  // pipeline, so it goes with it.
  if(work.stalePipeline)
  {
    // SYNC NOTE: the previous frame may still use these pipelines. prepareFrame() (or reloadShader())
    // drained the queue on the application thread before this ran, and no path-tracer frame is
    // recorded until it is done (the BusyWindow gates GltfRenderer::onRender). No wait here: this
    // runs on the compile worker while the application thread submits the BusyWindow's frames, and
    // host access to a queue must be externally synchronized.
    const VariantKey wantKey = makeVariantKey(resources);
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
      vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
      m_rqPipeline                = VK_NULL_HANDLE;
      m_rtxPipeline               = VK_NULL_HANDLE;
      m_pipelineUseSER            = m_useSER;
      m_compiled.dlssTransparency = wantKey.dlssTransparency;
      resources.allocator.destroyBuffer(m_sbtBuffer);
      m_sbtBuffer  = {};
      m_sbtRegions = {};
    }
    work = getPendingCompileWork(resources);
  }

  if(work.missingPipeline)
  {
    if(m_renderTechnique == RenderTechnique::RayQuery)
    {
      if(m_busyWindow)
        m_busyWindow->setReason("Creating Ray Query pipeline...");
      createRqPipeline(resources);
    }
    else
    {
      if(m_busyWindow)
        m_busyWindow->setReason("Creating RTX pipeline...");
      createRtxPipeline(resources);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Launches a background worker that (re)compiles the Slang shader and builds the ray
// tracing pipeline for the currently selected technique.
void PathTracer::startAsyncCompile(Resources& resources)
{
  if(m_compileThread.joinable())
    m_compileThread.join();

  // Without a BusyWindow nothing gates the next frame, which then joins this thread in
  // startAsyncCompile(): the compile still completes, synchronously.
  if(m_busyWindow)
    m_busyWindow->start("Preparing path tracer...");

  m_compileThread = std::thread([this, &resources]() {
    ensureShadersAndPipelines(resources);
    if(m_busyWindow)
      m_busyWindow->stop();
  });
}

//--------------------------------------------------------------------------------------------------
// Bring the live shader, pipeline, and denoiser targets in line with the settings, however those
// were changed (UI, command line, sequence script, MCP). Nothing but compares when they already
// match. Returns false when this frame must be skipped because an async build was started.
bool PathTracer::prepareFrame(Resources& resources)
{
  // A variant already in the cache is swapped in right here: only handles move (what the previous
  // frame may still use is released through the app's frame ring), so there is no queue wait, no
  // BusyWindow and no accumulation reset. Anything else goes to an async job that shows the BusyWindow.
  installBuiltVariant(resources);  // a variant built off the frame goes into the cache, swapped in below
  PendingCompileWork work = getPendingCompileWork(resources);
  if(work.recompile && tryRestoreCachedVariant(resources))
    work = getPendingCompileWork(resources);  // the cached pipelines may still need a rebuild (SER, technique)
  if(work.any() && buildsOffTheFrame(resources, work))
    work = {};  // the live variant, or a cached one standing in, draws this frame

  if(work.any())
  {
    // SYNC NOTE: a rebuild destroys live handles the previous frame may still use. The wait is here,
    // on the application thread, before the worker starts; the worker itself never touches the queue.
    NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
    if(!resources.app->isHeadless())
    {
      startAsyncCompile(resources);
      return false;
    }
    // In headless mode we must compile/build synchronously so frames are actually rendered before the application frame loop exits.
    ensureShadersAndPipelines(resources);
  }

#if defined(USE_OPTIX_DENOISER)
  m_optix->ensureTargets(resources);
#endif
#if defined(USE_OIDN_DENOISER)
  m_oidn->ensureTargets(resources);
#endif
  return true;
}

//--------------------------------------------------------------------------------------------------
// Render the scene
void PathTracer::onRender(VkCommandBuffer cmd, Resources& resources)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  // A new accumulation: whether DLSS renders its frame 0 is set where it does, and any hand-off is over
  if(resources.frameCount == 0)
  {
    m_dlssFrame0 = false;
    m_handoff.cancel();
    m_prevDlssFrame     = false;
    m_accumulationStart = std::chrono::steady_clock::now();
  }
  // Whether DLSS draws this frame, decided once: the shader variant prepareFrame() picks and the frame
  // must agree. The hand-off delay's frames are DLSS's with the count held at 1; on the regular variant
  // they wrote no guides, and DLSS reapplied the last moving frame's motion vectors to the still view,
  // smearing it along that motion until the accumulation took over.
  m_dlssShaderFrame = !isAccumulationDenoiserEnabled() || resources.frameCount == 0 || inHandoffDelay();
  // The technique follows the variant (makeVariantKey's dlss), so a view's pipeline is the cached one
  m_renderTechnique = (isDlssEnabled() && m_dlssShaderFrame) ? m_techniqueDlss : m_techniqueStill;
  if(!m_supportRayQuery)
    m_renderTechnique = RenderTechnique::RayTracing;

  if(!prepareFrame(resources))
    return;

#if defined(USE_DLSS)
  // Drive the DLSS state machine, then decide the frame: DLSS renders it (every frame without an
  // accumulation denoiser; frame 0 and the delay's frames with one), or the accumulation does, this
  // being the hand-off frame when DLSS rendered the one before.
  const bool dlssTicked = m_dlss->tick(resources);
  m_prevDlssFrame       = m_dlssFrame;
  {
    const bool dlssActive = m_dlss->isActive();
    m_dlssFrame = dlssActive && m_dlssShaderFrame;
  }
  m_handoffFrame = !m_dlssFrame && m_prevDlssFrame && resources.frameCount >= 1;
  if(dlssTicked)
    updateDlssResources(cmd, resources);
#else
  m_prevDlssFrame = m_dlssFrame = m_handoffFrame = false;
#endif

  // Reset display buffer to rendered on first frame: the denoised image is stale until denoised again
  if((resources.frameCount == 0)
     && (resources.settings.displayBuffer == DisplayBuffer::eOptixDenoised
         || resources.settings.displayBuffer == DisplayBuffer::eOidnDenoised))
  {
    resources.settings.displayBuffer = DisplayBuffer::eRendered;
  }

  // Handle adaptive sampling (SPP adjustment)
  updateAdaptiveSampling(resources);

  // Finding the rendering size (needed before setupPushConstant so pixelAngle can be derived).
  VkExtent2D renderingSize = resources.gBuffers.getSize();
  updateDepthStaging(cmd, resources);  // first frame: onResize may not have run yet
  updateConvergence(cmd, resources);
#if defined(USE_DLSS)
  // When DLSS is effectively enabled, use DLSS render size
  if(getEffectiveDlssEnabled(resources))
  {
    renderingSize = m_dlss->getRenderSize();
  }
#endif
#if defined(USE_OPTIX_DENOISER)
  if(getEffectiveOptixEnabled(resources) && m_optix->isUpscaleMode())
  {
    renderingSize = m_optix->getRenderSize();
  }
#endif

  // Setting up the push constant
  const shaderio::PathtracePushConstant pushConst = setupPushConstant(cmd, resources, renderingSize);

  // The hand-off frame, after DLSS's: keep DLSS's image for the hand-off before the path tracer
  // overwrites it. The barrier below orders that write after the copy.
  if(m_handoffFrame)
    if(const std::optional<DenoiserHandoff::Denoiser> denoiser = handoffDenoiser(resources))
      m_handoff.cmdCapture(cmd, resources, *denoiser);

  // Make sure buffer is ready to be used.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

  // The technique's pipeline and descriptors, bound once for the probe and the frame's paths
  if(!bindTechnique(cmd, resources))
    return;

  // Every change to the camera or the scene starts a new accumulation; a scene without a medium
  // has no camera medium to find (setupPushConstant leaves the pointer null)
  if(resources.frameCount == 0 && pushConst.cameraMedium != nullptr)
    probeCameraMedium(cmd, pushConst);

  // Tracing the rays: Ray Query or Ray Tracing
  {
    auto timerSection =
        m_profiler->cmdFrameSection(cmd, m_renderTechnique == RenderTechnique::RayQuery ? kTimerPathTraceRq : kTimerPathTraceRtx);
    dispatchRays(cmd, renderingSize);
  }

  // The adaptive stop's count, for the host once the frame is done (countAdaptivePixels)
  if(pushConst.adaptiveActive != nullptr)
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                           VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_HOST_READ_BIT);

  // Making sure the rendered image is ready to be used by tonemapper (written by ray query compute or the RTX pipeline)
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

  // Depth is only traced on the first frame (see processPixel); bring it into the depth buffer
  // before anything reads it (OptiX upscale below, overlays and picking after).
  if(m_depthViaStaging && m_depthCopyable && resources.frameCount == 0)
    cmdCopyStagedDepth(cmd, resources);

#if defined(USE_DLSS)
  // If DLSS is effectively enabled for this frame, perform denoising
  if(getEffectiveDlssEnabled(resources))
  {
    denoiseDlss(cmd, resources);
    m_dlssFrame0 = isAccumulationDenoiserEnabled();  // frame 0 and the delay's frames, then OptiX or OIDN takes over
  }
#endif

#if defined(USE_OPTIX_DENOISER)
  // Update OptiX auto-denoiser (only when effectively enabled)
  if(getEffectiveOptixEnabled(resources))
  {
    // OptiX runs on CUDA, outside this command buffer, where a Vulkan GPU timer reads ~0. The section's
    // GPU time is the denoiser invoke measured with CUDA events (the Vulkan copies around it excluded),
    // comparable to OIDN's; its CPU time is the host wall time, the wait for the frames in flight included.
    {
      const uint32_t denoisesBefore = m_optix->denoiseCount();
      const auto section = m_profilerTimeline ? m_profilerTimeline->frameBeginSection("Optix denoiser", &m_optixTimeProvider) :
                                                nvutils::ProfilerTimeline::FrameSectionID{};
      m_optix->updateDenoiser(resources, m_dlssFrame0, handoffEnd(resources));
      if(m_profilerTimeline)
      {
        m_optixGpuUs[section.subFrame] = (m_optix->denoiseCount() != denoisesBefore) ? 1000.0 * m_optix->lastInvokeGpuMs() : 0.0;
        m_profilerTimeline->frameEndSection(section);
      }
    }

    if(m_optix->isUpscaleMode() && resources.frameCount == 0)
    {
      auto timerSection = m_profiler->cmdFrameSection(cmd, "Optix upscale guides");
      upscaleSelectionAndDepth(cmd, resources);
    }
  }
#endif

#if defined(USE_OIDN_DENOISER)
  // OIDN auto-denoise, recorded in this frame's command buffer: no lag, no host wait
  if(getEffectiveOidnEnabled(resources))
  {
    // The image it reads includes the samples traced in this frame (not counted until updateStatistics)
    m_oidn->updateDenoiser(cmd, resources, m_profiler, m_totalSamplesAccumulated + m_frameSamples, handoffEnd(resources));
  }
#endif

  // DLSS hand-off: the image shown while OptiX or OIDN takes over from DLSS's frame 0
  {
    const std::optional<DenoiserHandoff::Denoiser> denoiser = handoffDenoiser(resources);
    m_handoff.cmdUpdate(cmd, resources,
                        {.denoiser   = denoiser ? &*denoiser : nullptr,
                         .samples    = m_totalSamplesAccumulated + m_frameSamples,
                         .handoffEnd = handoffEnd(resources)});
  }

  updateStatistics(resources);
}

//--------------------------------------------------------------------------------------------------
// Idle frames, once the accumulation is complete: the auto denoisers denoise its final image, which
// is shown from then on, if their last denoise was not of it. OptiX, one frame behind, needs this for
// every accumulation; OIDN only when the cap was lowered below it, or Auto turned on, since.
void PathTracer::onAccumulationComplete(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_OPTIX_DENOISER)
  if(getEffectiveOptixEnabled(resources))
    m_optix->denoiseFinal(resources);
#endif
#if defined(USE_OIDN_DENOISER)
  if(getEffectiveOidnEnabled(resources))
    m_oidn->denoiseFinal(cmd, resources, m_profiler);
#endif
}

//--------------------------------------------------------------------------------------------------
// Upscale selection ID and depth from render resolution (half) to display resolution (full).
// In OptiX 2x upscale mode the shader writes into the top-left corner of the full-res GBuffer.
// vkCmdBlitImage cannot blit an image to itself with overlapping regions, so we copy the
// half-res data to a staging image first, then blit it back at full resolution.
void PathTracer::upscaleSelectionAndDepth(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_OPTIX_DENOISER)
  VkExtent2D srcSize = m_optix->getRenderSize();
  VkExtent2D dstSize = resources.gBuffers.getSize();

  // Copy half-res region to staging, then blit staging back to full-res destination.
  auto copyAndBlit = [cmd, &srcSize, &dstSize](VkImage image, VkImage staging, VkImageAspectFlags aspect) {
    VkImageSubresourceLayers subresource = {.aspectMask = aspect, .layerCount = 1};

    VkImageCopy copy{
        .srcSubresource = subresource,
        .dstSubresource = subresource,
        .extent         = {srcSize.width, srcSize.height, 1},
    };
    vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, staging, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkOffset3D srcExtent = {int(srcSize.width), int(srcSize.height), 1};
    VkOffset3D dstExtent = {int(dstSize.width), int(dstSize.height), 1};

    VkImageBlit blit{
        .srcSubresource = subresource,
        .srcOffsets     = {{0, 0, 0}, srcExtent},
        .dstSubresource = subresource,
        .dstOffsets     = {{0, 0, 0}, dstExtent},
    };
    vkCmdBlitImage(cmd, staging, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
  };

  copyAndBlit(resources.gBuffers.getColorImage(Resources::eImgSelection), m_optix->getStagingSelectionImage(), VK_IMAGE_ASPECT_COLOR_BIT);
  copyAndBlit(resources.gBuffers.getDepthImage(), m_optix->getStagingDepthImage(), VK_IMAGE_ASPECT_DEPTH_BIT);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
#endif
}

//--------------------------------------------------------------------------------------------------
// Size the depth staging image and copy buffer to the G-buffer (path-traced depth on GPUs whose
// depth format cannot be a storage image).
void PathTracer::updateDepthStaging(VkCommandBuffer cmd, Resources& resources)
{
  if(!m_depthViaStaging)
    return;
  const VkExtent2D size = resources.gBuffers.getSize();
  const VkExtent2D cur  = m_depthStaging.getSize();
  if(size.width == cur.width && size.height == cur.height && m_depthCopyBuffer.buffer != VK_NULL_HANDLE)
    return;

  NVVK_CHECK(m_depthStaging.update(cmd, size));
  resources.allocator.destroyBuffer(m_depthCopyBuffer);
  NVVK_CHECK(resources.allocator.createBuffer(m_depthCopyBuffer, VkDeviceSize(size.width) * size.height * sizeof(float),
                                              VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
  NVVK_DBG_NAME(m_depthCopyBuffer.buffer);
}

uint32_t* PathTracer::countAdaptivePixels(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderingSize)
{
  if(!resources.app)
    return nullptr;
  // A slot per frame in flight: the application waited for the frame that last used this one
  const uint32_t slots = std::max(resources.app->getFrameCycleSize(), 1u);
  if(m_adaptiveCounts.size() != slots)
  {
    resources.allocator.destroyBuffer(m_adaptiveActiveBuffer);
    NVVK_CHECK(resources.allocator.createBuffer(m_adaptiveActiveBuffer, slots * sizeof(uint32_t),
                                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                    | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    NVVK_DBG_NAME(m_adaptiveActiveBuffer.buffer);
    m_adaptiveCounts.assign(slots, {});
  }
  const uint32_t slot = resources.app->getFrameCycleIndex() % slots;

  // That frame's count, if it was this accumulation's: once few enough pixels sample, the still is done.
  // It ends a frame or two after the count says so, the frames in flight.
  const AdaptiveCount& last = m_adaptiveCounts[slot];
  if(last.counted && last.accumulation == m_accumulationId)
  {
    uint32_t active = 0;
    std::memcpy(&active, m_adaptiveActiveBuffer.mapping + slot * sizeof(uint32_t), sizeof(active));
    if(double(active) <= double(m_adaptiveDone) * double(renderingSize.width) * double(renderingSize.height))
      resources.adaptiveDone = true;
  }
  m_adaptiveCounts[slot] = {m_accumulationId, true};

  // the barrier after setupPushConstant orders the clear before the trace
  vkCmdFillBuffer(cmd, m_adaptiveActiveBuffer.buffer, slot * sizeof(uint32_t), sizeof(uint32_t), 0);
  return reinterpret_cast<uint32_t*>(m_adaptiveActiveBuffer.address + slot * sizeof(uint32_t));
}

void PathTracer::updateConvergence(VkCommandBuffer cmd, Resources& resources)
{
  const VkExtent2D size = resources.gBuffers.getSize();
  const VkExtent2D cur  = m_convergence.getSize();
  if(size.width != cur.width || size.height != cur.height)
    NVVK_CHECK(m_convergence.update(cmd, size));  // an accumulation's first frame writes every pixel
}

//--------------------------------------------------------------------------------------------------
// Copy the depth the path tracer wrote into m_depthStaging (R32_SFLOAT) to the G-buffer depth image.
// Core Vulkan has no direct copy between a color and a depth image, so it goes through a buffer:
// both use the same 32-bit float texel layout.
void PathTracer::cmdCopyStagedDepth(VkCommandBuffer cmd, Resources& resources)
{
  NVVK_DBG_SCOPE(cmd);
  const VkExtent2D size = resources.gBuffers.getSize();

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

  VkBufferImageCopy region{
      .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
      .imageExtent      = {size.width, size.height, 1},
  };
  vkCmdCopyImageToBuffer(cmd, m_depthStaging.getColorImage(0), VK_IMAGE_LAYOUT_GENERAL, m_depthCopyBuffer.buffer, 1, &region);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  vkCmdCopyBufferToImage(cmd, m_depthCopyBuffer.buffer, resources.gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

//--------------------------------------------------------------------------------------------------
// Push the descriptor set
// This is making sure our shader has the latest TLAS, and the latest output images
void PathTracer::pushDescriptorSet(VkCommandBuffer cmd, Resources& resources, VkPipelineBindPoint bindPoint) const
{
  nvvk::WriteSetContainer write{};
  write.append(resources.descriptorBinding[1].getWriteSet(shaderio::BindingPoints::eTlas), resources.gpuView().tlas());

  // Output images by shaderio::OutputImage slot, the first outputCount of them bound. Fixed storage:
  // this runs every frame (twice on frame 0). ObjectID is written to the eSelectImage slot as float in .r.
  std::array<VkDescriptorImageInfo, shaderio::eConvergence + 1> outputImages{};
  uint32_t                                                 outputCount = 2;
  outputImages[shaderio::eResultImage] = resources.gBuffers.getColorStorageImageInfo(Resources::eImgRendered);
  outputImages[shaderio::eSelectImage] = resources.gBuffers.getColorStorageImageInfo(Resources::eImgSelection);

#if defined(USE_DLSS)
  if(getEffectiveDlssEnabled(resources))
  {
    using namespace shaderio;
    // Re-route every output image at the path tracer's RR guide-buffer slots. The Dlss adapter
    // hides the inner RenderTarget behind getRrAttachment() so we can iterate the OutputImage enum
    // values without touching m_innerGBuffer directly.
    static constexpr OutputImage kRrSlots[] = {
        eResultImage, eSelectImage, eDlssAlbedo,          eDlssSpecAlbedo, eDlssNormalRoughness,
        eDlssMotion,  eDlssDepth,   eDlssSpecularHitDist, eNrMask, eDlssResponsivity,
    };
    outputCount = static_cast<uint32_t>(kRrSlots[std::size(kRrSlots) - 1]) + 1;
    for(OutputImage slot : kRrSlots)
      outputImages[slot] = m_dlss->getRrAttachment(slot);
  }
#endif

  // The denoiser guide slots (ePtDenoiserGuides) get the images of the denoiser of the accumulation on
  // this frame: OptiX and OIDN are exclusive, and neither shares a frame with DLSS
#if defined(USE_OPTIX_DENOISER)
  if(getEffectiveOptixEnabled(resources))
  {
    using namespace shaderio;
    outputCount                   = std::max<uint32_t>(outputCount, eDenoiserNormal + 1);
    outputImages[eDenoiserAlbedo] = m_optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferAlbedo);
    outputImages[eDenoiserNormal] = m_optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferNormal);
  }
#endif
#if defined(USE_OIDN_DENOISER)
  if(getEffectiveOidnEnabled(resources))
  {
    using namespace shaderio;
    outputCount                   = std::max<uint32_t>(outputCount, eDenoiserNormal + 1);
    outputImages[eDenoiserAlbedo] = m_oidn->getDescriptorImageInfo(OidnDenoiser::eGBufferAlbedo);
    outputImages[eDenoiserNormal] = m_oidn->getDescriptorImageInfo(OidnDenoiser::eGBufferNormal);
  }
#endif


  // An accumulation's per-pixel convergence (ePtAdaptiveStop); DLSS frames never read it
  if(m_pushConst.flags & shaderio::ePtAdaptiveStop)
  {
    outputCount                          = shaderio::eConvergence + 1;
    outputImages[shaderio::eConvergence] = m_convergence.getColorStorageImageInfo(0);
  }

  VkWriteDescriptorSet allTextures = resources.descriptorBinding[1].getWriteSet(shaderio::BindingPoints::eOutImages);
  allTextures.descriptorCount      = outputCount;
  write.append(allTextures, outputImages.data());

  // Bind GBuffer depth as storage image for writing hardware depth from path tracer (frame 0)
  // (or the R32_SFLOAT staging image when the depth format cannot be a storage image; see onAttach)
  const VkDescriptorImageInfo depthStorageInfo = m_depthViaStaging ? m_depthStaging.getColorStorageImageInfo(0) :
                                                                     VkDescriptorImageInfo{
                                                                         .imageView = resources.gBuffers.getDepthImageView(),
                                                                         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                                     };
  write.append(resources.descriptorBinding[1].getWriteSet(shaderio::BindingPoints::eOutDepth), depthStorageInfo);

  vkCmdPushDescriptorSetKHR(cmd, bindPoint, m_pipelineLayout, 1, write.size(), write.data());
}

//--------------------------------------------------------------------------------------------------
// Create the pipeline
void PathTracer::createPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);
  // Set 3 is the physical sky's scattering LUTs, read by aerial perspective. Always in the layout,
  // even for a scene that never selects that sky: a pipeline layout is built once and the
  // environment changes at runtime, so making the layout depend on it would mean rebuilding the
  // pipeline every time someone switched to an HDR.
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts{resources.descriptorSetLayout[0], resources.descriptorSetLayout[1],
                                                          resources.hdrIbl.getDescriptorSetLayout(),
                                                          resources.skyLutDescriptorSetLayout};

  // Creating the pipeline layout
  static_assert(sizeof(shaderio::PathtracePushConstant) <= 256, "the push constants Vulkan guarantees on NVIDIA");
  static_assert(offsetof(shaderio::PathtracePushConstant, adaptiveRelax) % 16 == 0, "a float4 array's alignment in the shader");
  VkPushConstantRange        pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PathtracePushConstant)};
  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = uint32_t(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  discardVariantJob();  // it builds against the layout
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);
}

//--------------------------------------------------------------------------------------------------
// After a renderer update the driver has none of the path tracer's pipelines, and creating one takes
// 10-25 s (the ray-query one the longest). The first path-traced frame asks for the embedded DLSS
// build and a still for the embedded build itself (isEmbeddedBaseVariant), and everything their
// pipelines are made from is known at attach: the embedded SPIR-V, the technique, the layout, SER and
// the DLSS transparency mode. So a thread of its own per build creates each into the pipeline cache
// now, while the app starts and a scene loads, and createRtxPipeline() or createRqPipeline() waits
// for them and finds it there. The threads touch only the device, the pipeline cache (pipeline
// creation may share one between threads) and the values they were handed.
void PathTracer::startPipelineWarm(Resources& resources)
{
#if defined(USE_DLSS)
  if(!m_supportSER || m_pipelineLayout == VK_NULL_HANDLE)
    return;
  std::lock_guard lock(m_warmMutex);
  if(!m_warmThreads.empty())
    return;
  const VkPipelineLayout      layout       = m_pipelineLayout;
  const int32_t               useSER       = m_useSER ? 1 : 0;
  const int32_t               transparency = static_cast<int32_t>(m_dlss->getTransparencyMode());
  const VkPipelineCreateFlags flags        = rtxPipelineFlags(resources.settings.opacityMicromapSupported, m_capturePipelineStats);
  const VkPipelineCreateFlags rqFlags = m_capturePipelineStats ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) : 0u;
  auto warm = [this, layout, useSER, transparency, flags, rqFlags](std::span<const uint32_t> code, const char* build, bool rayQuery) {
    const auto               start = std::chrono::steady_clock::now();
    VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size_bytes(),
        .pCode    = code.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if(vkCreateShaderModule(m_device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
      return;
    VkResult         result   = VK_ERROR_UNKNOWN;
    const auto       built    = std::chrono::steady_clock::now();
    const VkPipeline pipeline = createPathTracePipeline(m_device, m_pipelineCache.getCache(), layout, module, rayQuery,
                                                        useSER, transparency, flags, rqFlags, result);
    const double     ms       = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - built).count();
    vkDestroyPipeline(m_device, pipeline, nullptr);
    vkDestroyShaderModule(m_device, module, nullptr);
    LOGI("%s pipeline warm-up (%s): %s (%.2f ms)\n", rayQuery ? "RQ" : "RTX", build, result == VK_SUCCESS ? "done" : "failed",
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    if(result == VK_SUCCESS)
      notePipelineBuild(nvvk::fnv1a64(code.data(), code.size_bytes()), rayQuery, useSER, transparency, rayQuery ? rqFlags : flags, ms);
    // the creation that follows is a cache hit and saves nothing, so the cache is saved here
    if(result == VK_SUCCESS)
      m_pipelineCache.saveAsync();
  };
  // each build with the technique its view renders with (onRender)
  const bool rqDlss  = m_supportRayQuery && m_techniqueDlss == RenderTechnique::RayQuery;
  const bool rqStill = m_supportRayQuery && m_techniqueStill == RenderTechnique::RayQuery;
  m_warmThreads.emplace_back(warm, std::span<const uint32_t>(gltf_pathtrace_slang_dlss, gltf_pathtrace_slang_dlss_sizeInBytes / sizeof(uint32_t)),
                             "DLSS", rqDlss);
  m_warmThreads.emplace_back(warm, std::span<const uint32_t>(gltf_pathtrace_slang, gltf_pathtrace_slang_sizeInBytes / sizeof(uint32_t)),
                             "still", rqStill);
#endif
}

int32_t PathTracer::transparencyMode() const
{
#if defined(USE_DLSS)
  return static_cast<int32_t>(m_dlss->getTransparencyMode());
#else
  return 0;
#endif
}

void PathTracer::rememberModuleCode(VkShaderModule module, const void* code, size_t bytes)
{
  std::lock_guard lock(m_moduleCodeMutex);
  m_moduleCode[module] = nvvk::fnv1a64(code, bytes);
}

uint64_t PathTracer::moduleCode(VkShaderModule module)
{
  std::lock_guard lock(m_moduleCodeMutex);
  const auto      it = m_moduleCode.find(module);
  return it != m_moduleCode.end() ? it->second : 0;
}

void PathTracer::notePipelineBuild(uint64_t code, bool rayQuery, int32_t useSER, int32_t transparency, VkPipelineCreateFlags flags, double ms)
{
  if(code == 0)
    return;  // a module made from code nobody hashed
  const uint64_t state[] = {code, rayQuery ? 1u : 0u, uint64_t(useSER), uint64_t(transparency), uint64_t(flags), m_driverVersion};
  if(m_buildRecord.note(nvvk::fnv1a64(state, sizeof(state)), ms))
    LOGW("[PathTracer] A pipeline built before compiled again (%.0f ms): the driver's shader cache is full\n", ms);
}

void PathTracer::joinPipelineWarm()
{
  std::vector<std::thread> warm;
  {
    std::lock_guard lock(m_warmMutex);
    warm.swap(m_warmThreads);
  }
  for(std::thread& t : warm)
    if(t.joinable())
      t.join();
}

//--------------------------------------------------------------------------------------------------
// Create the compute pipeline
void PathTracer::createRqPipeline(Resources& /*resources*/)
{
  SCOPED_TIMER(__FUNCTION__);
  // as createRtxPipeline(): a warm-up of this pipeline still running is waited for, not repeated
  joinPipelineWarm();

  nvvk::Specialization specialization;
  specialization.add(0, m_useSER ? 1 : 0);  // USE_SER
#if defined(USE_DLSS)
  specialization.add(1, static_cast<int32_t>(m_dlss->getTransparencyMode()));  // USE_DLSS_TRANSP
#endif

  VkPipelineShaderStageCreateInfo shaderStage{
      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
      .module              = m_shaderModule,
      .pName               = "computeMain",
      .pSpecializationInfo = specialization.getSpecializationInfo(),
  };

  // Query the driver about pipeline cache hit / compilation duration (core in Vulkan 1.3)
  VkPipelineCreationFeedback           feedback{};
  VkPipelineCreationFeedbackCreateInfo feedbackInfo{
      .sType                     = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
      .pPipelineCreationFeedback = &feedback,
  };

  VkComputePipelineCreateInfo cpCreateInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext  = &feedbackInfo,
      .flags  = m_capturePipelineStats ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) : 0u,
      .stage  = shaderStage,
      .layout = m_pipelineLayout,
  };

  // NOTE: if the creation is slow, disable the validation layers for faster creation (--vvl 0)
  VkPipeline rqPipeline = VK_NULL_HANDLE;
  const auto rqStart    = std::chrono::steady_clock::now();
  NVVK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache.getCache(), 1, &cpCreateInfo, nullptr, &rqPipeline));
  NVVK_DBG_NAME(rqPipeline);
  notePipelineBuild(moduleCode(m_shaderModule), true, m_useSER ? 1 : 0, transparencyMode(), cpCreateInfo.flags,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rqStart).count());

  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    m_rqPipeline = rqPipeline;
    ++m_pipelineGeneration;
  }

  const bool valid = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
  const bool hit   = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
  LOGI("RQ pipeline creation: %s, %s (%.2f ms)\n", valid ? "valid" : "invalid", hit ? "CACHE HIT" : "cache miss",
       feedback.duration / 1e6);
  logPipelineStatistics(rqPipeline, "RQ");
  if(!hit)
    m_pipelineCache.saveAsync();
}

//--------------------------------------------------------------------------------------------------
// One log line per pipeline executable with the statistics the driver reports for it (register
// count, spills, ...): what decides occupancy, and so the first thing to read after a shader edit
// that should change it. Only when VK_KHR_pipeline_executable_properties is enabled, which also
// makes the pipelines capture them (VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR). NVIDIA's driver
// reports the ray-query (compute) pipeline but no executables for the ray tracing pipeline.
void PathTracer::logPipelineStatistics(VkPipeline pipeline, const char* label) const
{
  if(!m_capturePipelineStats)
    return;
  const VkPipelineInfoKHR pipelineInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipeline};
  uint32_t                executableCount = 0;
  vkGetPipelineExecutablePropertiesKHR(m_device, &pipelineInfo, &executableCount, nullptr);
  std::vector<VkPipelineExecutablePropertiesKHR> executables(executableCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  vkGetPipelineExecutablePropertiesKHR(m_device, &pipelineInfo, &executableCount, executables.data());
  for(uint32_t i = 0; i < executableCount; ++i)
  {
    const VkPipelineExecutableInfoKHR executableInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipeline, .executableIndex = i};
    uint32_t statCount = 0;
    vkGetPipelineExecutableStatisticsKHR(m_device, &executableInfo, &statCount, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> stats(statCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    vkGetPipelineExecutableStatisticsKHR(m_device, &executableInfo, &statCount, stats.data());
    std::string line;
    for(size_t s = 0; s < stats.size(); s++)
    {
      const VkPipelineExecutableStatisticKHR& stat = stats[s];
      line += fmt::format("{}{}=", line.empty() ? "" : ", ", stat.name);
      switch(stat.format)
      {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
          line += stat.value.b32 ? "true" : "false";
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
          line += std::to_string(stat.value.i64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
          line += std::to_string(stat.value.u64);
          break;
        default:
          line += fmt::format("{:.3g}", stat.value.f64);
          break;
      }
    }
    LOGI("%s pipeline executable %u (%s): %s\n", label, i, executables[i].name, line.c_str());
  }
}


//--------------------------------------------------------------------------------------------------
// Create the RTX pipeline
void PathTracer::createRtxPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);
  // A warm-up of this pipeline still running: wait for it and find it in the cache, rather than
  // have the driver compile it a second time beside it
  joinPipelineWarm();
  // Creating all shaders
  RtxStages table(m_shaderModule);
  auto&     stages        = table.stages;
  auto&     shader_groups = table.groups;

  // Shader Execution Reorder (SER)
  nvvk::Specialization specialization;
  specialization.add(0, m_useSER ? 1 : 0);  // USE_SER
#if defined(USE_DLSS)
  specialization.add(1, static_cast<int32_t>(m_dlss->getTransparencyMode()));  // USE_DLSS_TRANSP
#endif
  stages[RtxStages::eRaygen].pSpecializationInfo = specialization.getSpecializationInfo();


  // Query the driver about pipeline cache hit / compilation duration (core in Vulkan 1.3)
  VkPipelineCreationFeedback           feedback{};
  VkPipelineCreationFeedbackCreateInfo feedbackInfo{
      .sType                     = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
      .pPipelineCreationFeedback = &feedback,
  };

  const VkPipelineCreateFlags flags = rtxPipelineFlags(resources.settings.opacityMicromapSupported, m_capturePipelineStats);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .pNext                        = &feedbackInfo,
      .flags                        = flags,
      .stageCount                   = static_cast<uint32_t>(stages.size()),  // Stages are shaders
      .pStages                      = stages.data(),
      .groupCount                   = static_cast<uint32_t>(shader_groups.size()),
      .pGroups                      = shader_groups.data(),
      .maxPipelineRayRecursionDepth = 2,  // Ray depth
      .layout                       = m_pipelineLayout,
  };
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
    m_rtxPipeline = VK_NULL_HANDLE;
  }


  // Time the create (+ join, when deferred) wall-clock ourselves. The NVIDIA driver does
  // not populate VkPipelineCreationFeedback when a VkDeferredOperationKHR is used, so
  // feedback.duration and the VALID_BIT cannot be relied on on the deferred path.
  // NOTE: if the creation is slow, disable the validation layers for faster creation (--vvl 0)
  const auto createStart = std::chrono::steady_clock::now();
  VkPipeline rtxPipeline = VK_NULL_HANDLE;

#if USE_DEFERRED_RTX_COMPILE
  // Deferred host operation: lets the driver split the SPIR-V->ISA compile across worker
  // threads. On a cache hit the call returns VK_SUCCESS immediately and the join loop is a
  // no-op, so this is safe to do unconditionally.
  VkDeferredOperationKHR deferredOp = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateDeferredOperationKHR(m_device, nullptr, &deferredOp));

  VkResult createResult = vkCreateRayTracingPipelinesKHR(m_device, deferredOp, m_pipelineCache.getCache(), 1,
                                                         &rtPipelineCreateInfo, nullptr, &rtxPipeline);

  if(createResult == VK_OPERATION_DEFERRED_KHR)
  {
    // Parallelize the compile across worker threads. We spawn (maxConcurrency - 1) workers
    // and have the calling thread also participate, so we use all available parallelism
    // reported by the driver without over-subscribing.
    const uint32_t maxConcurrency = vkGetDeferredOperationMaxConcurrencyKHR(m_device, deferredOp);
    const uint32_t hwConcurrency  = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t threadCount    = std::min(maxConcurrency, hwConcurrency);

    auto joinLoop = [this, deferredOp]() {
      VkResult r;
      do
      {
        r = vkDeferredOperationJoinKHR(m_device, deferredOp);
      } while(r == VK_THREAD_IDLE_KHR);
      // VK_THREAD_DONE_KHR: no more work available for this thread (others may still be running)
      // VK_SUCCESS: whole deferred op finished
    };

    std::vector<std::thread> workers;
    workers.reserve(threadCount > 0 ? threadCount - 1 : 0);
    for(uint32_t i = 1; i < threadCount; ++i)
      workers.push_back(std::thread(joinLoop));
    joinLoop();
    for(auto& t : workers)
      t.join();

    // All threads returned either VK_THREAD_DONE_KHR or VK_SUCCESS; poll the final result
    // (per spec, VK_NOT_READY means workers are done but some bookkeeping remains).
    VkResult finalResult;
    do
    {
      finalResult = vkGetDeferredOperationResultKHR(m_device, deferredOp);
    } while(finalResult == VK_NOT_READY);
    NVVK_CHECK(finalResult);
  }
  else
  {
    NVVK_CHECK(createResult);
  }

  vkDestroyDeferredOperationKHR(m_device, deferredOp, nullptr);
  const bool wasDeferred = (createResult == VK_OPERATION_DEFERRED_KHR);
#else
  // Synchronous path: no deferred op, driver is expected to populate VkPipelineCreationFeedback.
  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, m_pipelineCache.getCache(), 1,
                                            &rtPipelineCreateInfo, nullptr, &rtxPipeline));
  const bool wasDeferred = false;
#endif

  NVVK_DBG_NAME(rtxPipeline);

  const double elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - createStart).count();

  // Log pipeline creation status and whether deferred compilation was used. On the deferred
  // path the driver may skip filling in VkPipelineCreationFeedback, so we rely on our own
  // wall-clock timing and treat the cache-hit bit as a best-effort hint.
  const bool  valid = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
  const char* cacheStatus =
      valid ? ((feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) ? "CACHE HIT" : "cache miss") :
              "cache status unknown";
  LOGI("RTX pipeline creation: %s, deferred=%s (%.2f ms)\n", cacheStatus, wasDeferred ? "yes" : "no", elapsedMs);
  notePipelineBuild(moduleCode(m_shaderModule), false, m_useSER ? 1 : 0, transparencyMode(), flags, elapsedMs);
  logPipelineStatistics(rtxPipeline, "RTX");
  // Unknown feedback (the deferred path leaves it unset) is not a hit, so it still saves.
  if((feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) == 0)
    m_pipelineCache.saveAsync();

  // Create the Shading Binding Table
  {
    nvvk::Buffer                sbtBuffer;
    nvvk::SBTGenerator::Regions sbtRegions;
    createSbt(resources, rtxPipeline, m_shaderModule, sbtBuffer, sbtRegions);
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      resources.allocator.destroyBuffer(m_sbtBuffer);
      m_rtxPipeline = rtxPipeline;
      m_sbtBuffer   = sbtBuffer;
      m_sbtRegions  = sbtRegions;
      ++m_pipelineGeneration;
    }
  }
}


namespace {

constexpr uint64_t kSpirvCacheVersion   = 3;
constexpr size_t   kMaxCachedSpirvBytes = 64ULL * 1024 * 1024;
// Keys kept in shaders/ after pruning, least recently used dropped first. Each key is one
// variant (DLSS, wireframe, optimal-shader feature set, ...) of several MB.
constexpr size_t kMaxCachedSpirvKeys = 32;
// A source written this close to the compile's start may have been read half-saved; that compile
// is not cached. Also covers file systems with coarse timestamps.
constexpr auto kSourceSettleTime = std::chrono::seconds(2);
// Temporary files older than this belong to a process that was killed while writing.
constexpr auto kStaleTempAge = std::chrono::minutes(10);

uint64_t fnv1a(uint64_t hash, const void* data, size_t size)
{
  return nvvk::fnv1a64(data, size, hash);
}

uint64_t fnv1a(uint64_t hash, std::string_view text)
{
  hash = fnv1a(hash, text.data(), text.size());
  const char separator = 0;
  return fnv1a(hash, &separator, 1);
}

struct SourceStamp
{
  std::uintmax_t size    = 0;
  std::int64_t   mtimeNs = 0;
};

bool readStamp(const std::filesystem::path& path, SourceStamp& stamp)
{
  std::error_code ec;
  if(!std::filesystem::is_regular_file(path, ec))
    return false;
  stamp.size = std::filesystem::file_size(path, ec);
  if(ec)
    return false;
  const auto mtime = std::filesystem::last_write_time(path, ec);
  if(ec)
    return false;
  stamp.mtimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch()).count();
  return true;
}

std::filesystem::path canonicalDep(const std::filesystem::path& path)
{
  std::error_code       ec;
  std::filesystem::path canon = std::filesystem::weakly_canonical(path, ec);
  if(ec || canon.empty())
    return path;
  return canon;
}

void hashOption(uint64_t& hash, const slang::CompilerOptionEntry& option)
{
  const int name = static_cast<int>(option.name);
  const int kind = static_cast<int>(option.value.kind);
  hash           = fnv1a(hash, &name, sizeof(name));
  hash           = fnv1a(hash, &kind, sizeof(kind));
  hash           = fnv1a(hash, &option.value.intValue0, sizeof(option.value.intValue0));
  hash           = fnv1a(hash, &option.value.intValue1, sizeof(option.value.intValue1));
  hash           = fnv1a(hash, option.value.stringValue0 ? std::string_view(option.value.stringValue0) : std::string_view{});
  hash           = fnv1a(hash, option.value.stringValue1 ? std::string_view(option.value.stringValue1) : std::string_view{});
}

// Everything that changes the SPIR-V except the source text itself (that lives in .deps): macros,
// every compiler option (capabilities included), targets, the search paths, this executable, and
// the Slang build. Two trees or two builds therefore do not share a file.
uint64_t spirvCacheKey(nvslang::SlangCompiler& compiler)
{
  uint64_t hash = fnv1a(nvvk::kFnv1aOffset, std::string_view("pt-spirv"));
  hash          = fnv1a(hash, &kSpirvCacheVersion, sizeof(kSpirvCacheVersion));
  const char* tag = spGetBuildTagString();
  hash            = fnv1a(hash, tag ? std::string_view(tag) : std::string_view{});

  const std::filesystem::path executable = nvutils::getExecutablePath();
  hash                                   = fnv1a(hash, nvutils::utf8FromPath(executable));
  SourceStamp exeStamp{};
  if(readStamp(executable, exeStamp))
  {
    hash = fnv1a(hash, &exeStamp.size, sizeof(exeStamp.size));
    hash = fnv1a(hash, &exeStamp.mtimeNs, sizeof(exeStamp.mtimeNs));
  }

  for(const std::filesystem::path& searchPath : compiler.searchPaths())
    hash = fnv1a(hash, nvutils::utf8FromPath(canonicalDep(searchPath)));
  for(const slang::PreprocessorMacroDesc& macro : compiler.macros())
  {
    hash = fnv1a(hash, macro.name ? std::string_view(macro.name) : std::string_view{});
    hash = fnv1a(hash, macro.value ? std::string_view(macro.value) : std::string_view{});
  }
  for(const slang::CompilerOptionEntry& option : compiler.options())
    hashOption(hash, option);
  for(const slang::TargetDesc& target : compiler.targets())
  {
    const int format = static_cast<int>(target.format);
    const int floatMode = static_cast<int>(target.floatingPointMode);
    const int lineMode  = static_cast<int>(target.lineDirectiveMode);
    const int scalar    = target.forceGLSLScalarBufferLayout ? 1 : 0;
    hash                = fnv1a(hash, &format, sizeof(format));
    hash                = fnv1a(hash, &target.profile, sizeof(target.profile));
    hash                = fnv1a(hash, &target.flags, sizeof(target.flags));
    hash                = fnv1a(hash, &floatMode, sizeof(floatMode));
    hash                = fnv1a(hash, &lineMode, sizeof(lineMode));
    hash                = fnv1a(hash, &scalar, sizeof(scalar));
    for(uint32_t i = 0; i < target.compilerOptionEntryCount && target.compilerOptionEntries != nullptr; ++i)
      hashOption(hash, target.compilerOptionEntries[i]);
  }
  return hash;
}

constexpr std::string_view kSpvPrefix = "spv ";

// The blob a .deps names on its first line ("spv <hash>"), or an empty path.
std::filesystem::path blobOfDeps(std::istream& deps, const std::filesystem::path& dir, const std::string& stem)
{
  std::string line;
  if(!std::getline(deps, line))
    return {};
  if(!line.empty() && line.back() == '\r')
    line.pop_back();
  if(!line.starts_with(kSpvPrefix) || line.size() != kSpvPrefix.size() + 16)
    return {};
  return dir / (stem + "-" + line.substr(kSpvPrefix.size()) + ".spv");
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Run once from onAttach, before any compile.
// Keeps the shaders/ folder bounded: the kMaxCachedSpirvKeys most recently used keys (a hit
// touches its .deps), blobs no .deps names, and temporary files a killed writer left. Blobs and
// temporary files younger than kStaleTempAge are left alone: another renderer may be between
// writing a blob and publishing its .deps.
void PathTracer::pruneShaderCache() const
{
  if(m_shaderCacheDir.empty())
    return;
  // pipeline_cache.bin and its temporaries sit in the parent folder (the exe's without --cacheDir).
  nvvk::removeStaleTempFiles(m_shaderCacheDir.parent_path(), kStaleTempAge, "pipeline_cache.bin.");
  nvvk::removeStaleTempFiles(m_shaderCacheDir, kStaleTempAge, "pt-");

  struct Entry
  {
    std::filesystem::path                deps;
    std::filesystem::file_time_type      used;
  };
  std::vector<Entry>                    keys;
  std::vector<std::filesystem::path>    blobs;
  std::error_code                       ec;
  for(std::filesystem::directory_iterator it(m_shaderCacheDir, ec), end; !ec && it != end; it.increment(ec))
  {
    std::error_code entryEc;
    if(!it->is_regular_file(entryEc))
      continue;
    const auto extension = it->path().extension();
    if(extension == ".deps")
      keys.push_back({it->path(), it->last_write_time(entryEc)});
    else if(extension == ".spv")
      blobs.push_back(it->path());
  }

  std::sort(keys.begin(), keys.end(), [](const Entry& a, const Entry& b) { return a.used > b.used; });
  std::unordered_set<std::string> named;
  for(size_t i = 0; i < keys.size(); ++i)
  {
    std::error_code removeEc;
    if(i >= kMaxCachedSpirvKeys)
    {
      std::filesystem::remove(keys[i].deps, removeEc);
      continue;
    }
    std::ifstream               deps(keys[i].deps);
    const std::filesystem::path blob = blobOfDeps(deps, m_shaderCacheDir, nvutils::utf8FromPath(keys[i].deps.stem()));
    if(!blob.empty())
      named.insert(nvutils::utf8FromPath(blob.filename()));
  }

  const auto cutoff = std::filesystem::file_time_type::clock::now() - kStaleTempAge;
  for(const std::filesystem::path& blob : blobs)
  {
    std::error_code blobEc;
    if(named.contains(nvutils::utf8FromPath(blob.filename())))
      continue;
    const auto written = std::filesystem::last_write_time(blob, blobEc);
    if(!blobEc && written < cutoff)
      std::filesystem::remove(blob, blobEc);
  }
}

//--------------------------------------------------------------------------------------------------
// .deps starts with "spv <hash>" naming the blob written before it. The following lines are
// "size mtimeNs path" (canonical when written) and match only when each file is unchanged.
// A hit touches the .deps, which is what pruneShaderCache orders by.
bool PathTracer::loadCachedSpirv(uint64_t key, std::vector<uint32_t>& spirv) const
{
  spirv.clear();
  if(m_shaderCacheDir.empty())
    return false;

  const std::string           stem     = fmt::format("pt-{:016x}", key);
  const std::filesystem::path depsPath = m_shaderCacheDir / (stem + ".deps");

  std::ifstream deps(depsPath);
  if(!deps)
    return false;
  const std::filesystem::path spvPath = blobOfDeps(deps, m_shaderCacheDir, stem);
  if(spvPath.empty())
    return false;

  std::string line;
  bool        any = false;
  while(std::getline(deps, line))
  {
    if(!line.empty() && line.back() == '\r')
      line.pop_back();
    if(line.empty())
      continue;
    const auto sp1 = line.find(' ');
    const auto sp2 = sp1 == std::string::npos ? std::string::npos : line.find(' ', sp1 + 1);
    if(sp1 == std::string::npos || sp2 == std::string::npos)
      return false;
    SourceStamp expected{};
    try
    {
      expected.size    = static_cast<std::uintmax_t>(std::stoull(line.substr(0, sp1)));
      expected.mtimeNs = std::stoll(line.substr(sp1 + 1, sp2 - sp1 - 1));
    }
    catch(const std::exception&)
    {
      return false;
    }
    const std::filesystem::path path = nvutils::pathFromUtf8(line.substr(sp2 + 1));
    SourceStamp                 actual{};
    if(path.empty() || !readStamp(path, actual) || actual.size != expected.size || actual.mtimeNs != expected.mtimeNs)
      return false;
    any = true;
  }
  if(!any)
    return false;

  std::error_code ec;
  const auto      spvSize = std::filesystem::file_size(spvPath, ec);
  if(ec || spvSize == 0 || spvSize > kMaxCachedSpirvBytes || (spvSize % sizeof(uint32_t)) != 0)
    return false;

  std::ifstream spv(spvPath, std::ios::binary);
  if(!spv)
    return false;
  spirv.resize(static_cast<size_t>(spvSize / sizeof(uint32_t)));
  spv.read(reinterpret_cast<char*>(spirv.data()), static_cast<std::streamsize>(spvSize));
  if(!spv)
  {
    spirv.clear();
    return false;
  }
  deps.close();
  std::filesystem::last_write_time(depsPath, std::filesystem::file_time_type::clock::now(), ec);
  return true;
}

//--------------------------------------------------------------------------------------------------
// compileFile loads the entry as a source string, so the module's dependency list can omit
// gltf_pathtrace.slang itself. The resolved entry path is always recorded. A source written at
// or after `compileStart` (less kSourceSettleTime) may not be the text the compile read, so that
// compile is not cached; the next one is.
void PathTracer::storeCachedSpirv(uint64_t key, nvslang::SlangCompiler& compiler, std::filesystem::file_time_type compileStart)
{
  if(m_shaderCacheDir.empty())
    return;
  const uint32_t* spirv      = compiler.getSpirv();
  const size_t    spirvBytes = compiler.getSpirvSize();
  if(spirv == nullptr || spirvBytes == 0 || (spirvBytes % sizeof(uint32_t)) != 0 || spirvBytes > kMaxCachedSpirvBytes)
    return;

  std::vector<std::filesystem::path> deps;
  std::unordered_set<std::string>    seen;  // canonical paths already in deps
  auto                               addDep = [&](const std::filesystem::path& path) {
    if(path.empty())
      return;
    const std::filesystem::path canon = canonicalDep(path);
    const std::string           utf8  = nvutils::utf8FromPath(canon);
    if(utf8.empty() || !seen.insert(utf8).second)
      return;
    deps.push_back(canon);
  };

  addDep(nvutils::findFile("gltf_pathtrace.slang", compiler.searchPaths(), false));
  if(slang::IModule* module = compiler.getSlangModule())
  {
    const int32_t count = static_cast<int32_t>(module->getDependencyFileCount());
    for(int32_t i = 0; i < count; ++i)
    {
      const char* dep = module->getDependencyFilePath(i);
      if(dep != nullptr && dep[0] != '\0')
        addDep(nvutils::pathFromUtf8(dep));
    }
  }
  if(deps.empty())
  {
    LOGW("Slang SPIR-V cache: no source files to stamp; not writing the cache\n");
    return;
  }

  // loadModuleFromSourceString records the entry as a content hash, not a path. Those entries
  // are not files and are skipped.
  const std::int64_t settledNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>((compileStart - kSourceSettleTime).time_since_epoch()).count();
  std::ostringstream text;
  for(const std::filesystem::path& path : deps)
  {
    SourceStamp stamp{};
    if(!readStamp(path, stamp))
      continue;
    const std::string utf8 = nvutils::utf8FromPath(path);
    if(stamp.mtimeNs >= settledNs)
    {
      LOGW("Slang SPIR-V cache: '%s' changed around the compile; not writing the cache\n", utf8.c_str());
      return;
    }
    text << stamp.size << ' ' << stamp.mtimeNs << ' ' << utf8 << '\n';
  }
  const std::string sources = text.str();
  if(sources.empty())
  {
    LOGW("Slang SPIR-V cache: no source files to stamp; not writing the cache\n");
    return;
  }

  const uint64_t    spirvHash = fnv1a(nvvk::kFnv1aOffset, spirv, spirvBytes);
  const std::string spirvKey  = fmt::format("{:016x}", spirvHash);
  const std::string body      = std::string(kSpvPrefix) + spirvKey + "\n" + sources;

  std::error_code             ec;
  const std::string           stem     = fmt::format("pt-{:016x}", key);
  const std::filesystem::path spvPath  = m_shaderCacheDir / (stem + "-" + spirvKey + ".spv");
  const std::filesystem::path depsPath = m_shaderCacheDir / (stem + ".deps");
  std::filesystem::path       previousBlob;
  {
    std::ifstream previous(depsPath);
    if(previous)
      previousBlob = blobOfDeps(previous, m_shaderCacheDir, stem);
  }
  // The blob is named by its contents, so replacing .deps is the only publish step. A reader
  // keeps seeing the previous blob until that rename, and two writers do not share a .tmp name.
  std::string error;
  if(!std::filesystem::is_regular_file(spvPath, ec) && !nvvk::writeFileAtomic(spvPath, spirv, spirvBytes, &error))
  {
    LOGW("Slang SPIR-V cache: %s\n", error.c_str());
    return;
  }
  if(!nvvk::writeFileAtomic(depsPath, body.data(), body.size(), &error))
  {
    LOGW("Slang SPIR-V cache: %s\n", error.c_str());
    return;
  }
  // Only this key's .deps names its blobs. A reader that opened the old .deps just before the
  // replace finds the blob gone and compiles instead.
  if(!previousBlob.empty() && previousBlob != spvPath)
    std::filesystem::remove(previousBlob, ec);
}

//--------------------------------------------------------------------------------------------------
// Compile the shader
bool PathTracer::reloadShader(Resources& resources)
{
  if(m_compileThread.joinable())
    m_compileThread.join();

  // SYNC NOTE: User-initiated shader reload — cached pipelines may still be in flight from a
  // recent variant switch; live handles are destroyed in compileShader() below.
  NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
  discardVariantJob();  // built from the sources before the edit
  m_failedVariant.reset();
  destroyVariantCache(resources);
  m_skipVariantCache      = true;
  const bool compiledFile = compileShader(resources, true);
  m_skipVariantCache      = false;
  return compiledFile;
}

//--------------------------------------------------------------------------------------------------
// Compile the shader
bool PathTracer::compileShader(Resources& resources, bool fromFile)
{
  SCOPED_TIMER(__FUNCTION__);
  // Variant cache: try to find a previously-compiled shader + pipelines for the variant the
  // user wants this frame. On hit we skip both the Slang recompile and the RT pipeline link
  // (the slow part - the driver can take several seconds per RTX pipeline). On miss we
  // proceed with the normal compile path and the new entry will be stored on the next
  // variant switch.
  // The key is read once: it depends on the frame (see makeVariantKey()), and the macros below and
  // m_compiled must describe the same variant.
  const VariantKey key = makeVariantKey(resources);
  if(fromFile && !m_skipVariantCache)
  {
    // Every caller drained the queue first (prepareFrame, reloadShader) or has rendered nothing yet (onAttach)
    if(swapVariant(resources, key, /*gpuIdle=*/true))
    {
      LOGI("[PathTracer] Variant cache hit.\n");
      return true;
    }
    // Cache miss: handles were cleared; will rebuild below.
  }

  // The SPIR-V the module is created from: the embedded build unless the embedded DLSS build, the
  // disk cache or a from-file compile replaces it below
  std::span<const uint32_t> spirv(gltf_pathtrace_slang, gltf_pathtrace_slang_sizeInBytes / sizeof(uint32_t));

  // Compile from shader file if requested, used when reloading the shader
  bool                  compiledFromFile = false;
  std::vector<uint32_t> cachedSpirv;  // A disk-cache hit; spirv points here until the module exists
  if(fromFile)
  {
    SCOPED_TIMER("Slang compile from file");

    resources.slangCompiler.clearMacros();
    nvvkgltf::applyCommonShaderMacros(resources.slangCompiler);

    const std::vector<std::pair<std::string, std::string>> macros = variantMacros(key);
    if(key.optimal)
    {
      const auto& fs = key.features;
      LOGI(
          "[PathTracer] Optimal shader: scene uses [%s]; %d GLTF_USE_* gates off "
          "(each set explicitly to 0 or 1).\n",
          fs.toString().c_str(), fs.unusedExtensionCount());
    }

    for(const auto& [k, v] : macros)
      resources.slangCompiler.addMacro({k.c_str(), v.c_str()});

    // A user-forced reload (reloadShader) compiles from source even when the embedded DLSS build
    // or the disk cache would serve, and the result replaces that disk entry.
    const uint64_t spirvKey = spirvCacheKey(resources.slangCompiler);
    if(!m_skipVariantCache && isEmbeddedDlssVariant(resources))
    {
#if defined(USE_DLSS)
      spirv            = {gltf_pathtrace_slang_dlss, gltf_pathtrace_slang_dlss_sizeInBytes / sizeof(uint32_t)};
      compiledFromFile = true;
      LOGI("Slang: embedded DLSS path tracer, no compile\n");
#endif
    }
    else if(!m_skipVariantCache && isEmbeddedBaseVariant(resources))
    {
      // the variant a still hands off from (DLSS off, guides on) is the embedded build itself
      spirv            = {gltf_pathtrace_slang, gltf_pathtrace_slang_sizeInBytes / sizeof(uint32_t)};
      compiledFromFile = true;
      LOGI("Slang: embedded path tracer, no compile\n");
    }
    else if(!m_skipVariantCache && loadCachedSpirv(spirvKey, cachedSpirv))
    {
      spirv            = cachedSpirv;
      compiledFromFile = true;
      LOGI("Slang SPIR-V cache hit %016llx\n", static_cast<unsigned long long>(spirvKey));
#if defined(AFTERMATH_AVAILABLE)
      // The compile callback in GltfRenderer does not run when the SPIR-V is loaded from disk.
      AftermathCrashTracker::getInstance().addShaderBinary(std::span<const uint32_t>(cachedSpirv.data(), cachedSpirv.size()));
#endif
    }
    else if(const auto compileStart = std::filesystem::file_time_type::clock::now();
            resources.slangCompiler.compileFile("gltf_pathtrace.slang"))
    {
      LOGI("Slang SPIR-V cache %s %016llx\n", m_skipVariantCache ? "bypassed (reload)" : "miss",
           static_cast<unsigned long long>(spirvKey));
      spirv            = {resources.slangCompiler.getSpirv(), resources.slangCompiler.getSpirvSize() / sizeof(uint32_t)};
      compiledFromFile = true;
      storeCachedSpirv(spirvKey, resources.slangCompiler, compileStart);
    }
    else
    {
      LOGI("Slang SPIR-V cache miss %016llx\n", static_cast<unsigned long long>(spirvKey));
      // Falls back to the embedded SPIR-V, which is built with USE_OMM=0: in a scene with
      // micromaps its ray queries lack the OpacityMicromapIdKHR opt-in. The fallback stands in for
      // the wanted variant (m_compiledIsFallback) until a setting changes or the user reloads.
      LOGW("Error compiling gltf_pathtrace.slang; the embedded shader stands in until the next change or reload\n");
    }
  }

  // Create a shader module
  {
    SCOPED_TIMER("Create Shader Module");
    vkDestroyShaderModule(m_device, m_shaderModule, nullptr);

    VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size_bytes(),
        .pCode    = spirv.data(),
    };
    NVVK_CHECK(vkCreateShaderModule(m_device, &moduleInfo, nullptr, &m_shaderModule));
    NVVK_DBG_NAME(m_shaderModule);
    rememberModuleCode(m_shaderModule, spirv.data(), spirv.size_bytes());
  }

  // Record which variant this compile is, so getPendingCompileWork() can tell whether the settings
  // still match it. A from-file compile (or the embedded DLSS build standing in for one) follows
  // the live settings; so does a failed one, as the variant its fallback stands in for, or every
  // frame would retry the compile. The embedded SPIR-V at startup mirrors the CMake build (see
  // CMakeLists.txt): not optimal, all-on features, USE_GUIDE_SHADER=1 iff DLSS/OptiX/OIDN is
  // available at build time, and USE_DLSS_SHADER=0 (DLSS is never active at startup; when it
  // engages, the next compile takes the embedded DLSS build). Its pipelines are specialized with
  // the live transparency mode, which the key keeps.
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
#if defined(USE_DLSS) || defined(USE_OPTIX_DENOISER) || defined(USE_OIDN_DENOISER)
    constexpr bool kEmbeddedDlssGuide = true;
#else
    constexpr bool kEmbeddedDlssGuide = false;
#endif
    m_compiledIsFallback = fromFile && !compiledFromFile;
    if(compiledFromFile || m_compiledIsFallback)
      m_compiled = key;
    else
      m_compiled = VariantKey{.dlssGuide = kEmbeddedDlssGuide, .dlssTransparency = key.dlssTransparency};

    // Destroy pipeline since there is a new shader
    destroyPipelinesLocked();
  }

  return !fromFile || compiledFromFile;
}

//--------------------------------------------------------------------------------------------------
// The embedded DLSS build is CMake's path tracer flags plus USE_DLSS_SHADER=1: guide buffers on
// (USE_DLSS implies them), SER available, and wireframe, visualization, opacity micromaps and the
// optimal shader at their off defaults. Any other variant still compiles from source.
bool PathTracer::isEmbeddedDlssVariant(const Resources& resources) const
{
#if defined(USE_DLSS)
  if(!m_supportSER)
    return false;
  return makeVariantKey(resources).sameShader(VariantKey{.dlss = true, .dlssGuide = true});
#else
  return false;
#endif
}

//--------------------------------------------------------------------------------------------------
// The embedded build itself is the same flags without USE_DLSS_SHADER: DLSS off and guide buffers on
// when a denoiser is built in, which is what a still asks for before it hands off to OptiX. Taking
// it saves the Slang compile and, as its SPIR-V is the one the warm-up built a pipeline from, the
// pipeline build too.
bool PathTracer::isEmbeddedBaseVariant(const Resources& resources) const
{
  if(!m_supportSER)
    return false;
#if defined(USE_DLSS) || defined(USE_OPTIX_DENOISER) || defined(USE_OIDN_DENOISER)
  return makeVariantKey(resources).sameShader(VariantKey{.dlssGuide = true});
#else
  return makeVariantKey(resources).sameShader(VariantKey{});
#endif
}

//--------------------------------------------------------------------------------------------------
// The variant the current settings ask for -- the variant-cache key compileShader() looks up.
// DLSS's guide captures cost every sample, so they are compiled in only for the frames DLSS renders:
// with OptiX or OIDN also on, frame 0 and the hand-off delay's (m_dlssShaderFrame), while the
// accumulation's frames after them run the regular variant. Each camera start and stop is then a
// cached-variant swap in prepareFrame(); a moving camera stays on DLSS's.
// While DLSS is still loading, its variant is compiled ahead of its first frame.
PathTracer::VariantKey PathTracer::makeVariantKey(const Resources& resources) const
{
  return VariantKey{
      resources.settings.wireframe,
      resources.settings.visualization != shaderio::Visualization::eRendered,
      wantOpacityMicromapShader(resources),
      resources.settings.optimalShader,
      isDlssEnabled() && m_dlssShaderFrame,
      resources.currentFeatureSet.has(nvvkgltf::SceneFeatureSet::eDlssGuide),
      resources.settings.optimalShader ? resources.currentFeatureSet : nvvkgltf::SceneFeatureSet{},
#if defined(USE_DLSS)
      m_dlss ? static_cast<int>(m_dlss->getTransparencyMode()) : 0,
#else
      0,
#endif
      m_materialModel,
  };
}

std::string PathTracer::VariantKey::shaderDiff(const VariantKey& to) const
{
  std::string diff;
  auto        add = [&](const char* name, bool a, bool b) {
    if(a != b)
      diff += std::string(diff.empty() ? "" : ", ") + name + (a ? " 1->0" : " 0->1");
  };
  add("wireframe", wireframe, to.wireframe);
  add("visualize", visualize, to.visualize);
  add("omm", omm, to.omm);
  add("optimal", optimal, to.optimal);
  add("dlss", dlss, to.dlss);
  add("dlssGuide", dlssGuide, to.dlssGuide);
  add("openpbr", materialModel != 0, to.materialModel != 0);
  if(optimal && to.optimal && features != to.features)
    diff += std::string(diff.empty() ? "" : ", ") + "features [" + features.toString() + "] -> [" + to.features.toString() + "]";
  return diff.empty() ? "no shader field changed" : diff;
}

//--------------------------------------------------------------------------------------------------
// Swap in the wanted variant if the cache already holds it. A hit is only a handle swap, so
// onRender() does it inline instead of going through the async worker and the BusyWindow, whose
// completion resets the accumulation. On a miss nothing is touched: the live handles stay put for
// compileShader() to park.
bool PathTracer::tryRestoreCachedVariant(Resources& resources)
{
  if(m_skipVariantCache)
    return false;
  const VariantKey key = makeVariantKey(resources);
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    if(std::none_of(m_variantCache.begin(), m_variantCache.end(),
                    [&](const VariantCacheEntry& e) { return e.key == key; }))
      return false;
  }
  const bool hit = swapVariant(resources, key, /*gpuIdle=*/false);  // the previous frame may be in flight
  if(hit)  // debug level: with DLSS-RR and OptiX or OIDN, every camera start and stop swaps
    LOGD("[PathTracer] Variant cache hit (swapped in place).\n");
  return hit;
}

void PathTracer::destroyPipelinesLocked()
{
  vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
  m_rtxPipeline = VK_NULL_HANDLE;
  vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
  m_rqPipeline = VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Variant cache: park the active shader+pipelines+SBT under the variant they were built for, and
// try to restore handles for a target variant. Returns true on cache hit (caller can skip the
// expensive Slang recompile + RT pipeline link + SBT build).
//
// Reasoning for moving handles rather than copying: VkPipeline / VkShaderModule / nvvk::Buffer
// are not reference-counted across copies. We always have exactly one "live" instance per
// variant; the cache holds the others. On cache hit the current active handles are swapped with
// the cached ones (active handles for the OLD variant get parked, NEW handles become active).
//
// The SBT belongs to the RTX pipeline that built it (shader group handles are pipeline-specific)
// so we have to swap them together. Forgetting to swap the SBT causes a crash on the first
// dispatched ray because traversal jumps to handles from a different pipeline.
//
// Nothing is destroyed on a plain hit, which is why the per-frame swap needs no queue wait. What is
// destroyed (a duplicate parked entry, pipelines built with the other SER value, an LRU victim) may
// still be used by the frame in flight, so without gpuIdle it is released through the app's frame
// ring, which runs the callback once that frame's fence has been waited on.
bool PathTracer::swapVariant(Resources& resources, const VariantKey& newKey, bool gpuIdle)
{
  std::lock_guard<std::mutex> lock(m_compileMutex);

  auto release = [&](VkShaderModule module, VkPipeline rtx, VkPipeline rq, nvvk::Buffer sbt) {
    releaseVariant(resources, module, rtx, rq, sbt, gpuIdle);
  };

  // 1) Park the currently-active shader/pipelines/SBT so we don't leak them. The embedded fallback
  // standing in for a variant that failed to compile is not that variant: it is released instead,
  // so the next request for the variant compiles it again.
  const VariantKey currentKey = m_compiled;
  const bool       hasLive    = m_shaderModule != VK_NULL_HANDLE || m_rtxPipeline != VK_NULL_HANDLE
                       || m_rqPipeline != VK_NULL_HANDLE || m_sbtBuffer.buffer != VK_NULL_HANDLE;
  if(hasLive && m_compiledIsFallback)
  {
    release(m_shaderModule, m_rtxPipeline, m_rqPipeline, m_sbtBuffer);
  }
  else if(hasLive)
  {
    // If the current variant already exists in the cache (possible after a switch that
    // happened without our knowledge), update it in place rather than duplicating.
    bool foundExisting = false;
    for(auto& e : m_variantCache)
    {
      if(e.key == currentKey)
      {
        // Reclaim any stale handles already cached for this key before overwriting.
        release(e.shaderModule != m_shaderModule ? e.shaderModule : VK_NULL_HANDLE,
                e.rtxPipeline != m_rtxPipeline ? e.rtxPipeline : VK_NULL_HANDLE,
                e.rqPipeline != m_rqPipeline ? e.rqPipeline : VK_NULL_HANDLE,
                e.sbtBuffer.buffer != m_sbtBuffer.buffer ? e.sbtBuffer : nvvk::Buffer{});
        e.shaderModule = m_shaderModule;
        e.rtxPipeline  = m_rtxPipeline;
        e.rqPipeline   = m_rqPipeline;
        e.useSER       = m_pipelineUseSER;
        e.sbtBuffer    = m_sbtBuffer;
        e.sbtRegions   = m_sbtRegions;
        foundExisting  = true;
        break;
      }
    }
    if(!foundExisting)
    {
      // Front is most-recently used, so the LRU trim below evicts from the back.
      m_variantCache.insert(m_variantCache.begin(), VariantCacheEntry{currentKey, m_shaderModule, m_rtxPipeline, m_rqPipeline,
                                                                      m_pipelineUseSER, m_sbtBuffer, m_sbtRegions});
    }
  }

  // Detach handles from the live slots; if we find a hit we'll fill them from cache,
  // otherwise the caller will rebuild fresh.
  m_compiledIsFallback = false;
  m_shaderModule       = VK_NULL_HANDLE;
  m_rtxPipeline  = VK_NULL_HANDLE;
  m_rqPipeline   = VK_NULL_HANDLE;
  m_sbtBuffer    = {};
  m_sbtRegions   = {};

  // 2) Look up newKey. On hit, lift its handles out of the cache slot (so we don't
  // hold duplicate ownership) and remove the slot from the cache.
  for(auto it = m_variantCache.begin(); it != m_variantCache.end(); ++it)
  {
    if(it->key == newKey)
    {
      m_shaderModule = it->shaderModule;  // SER does not affect the SPIR-V, so this is always reusable
      if(it->useSER == m_pipelineUseSER)
      {
        m_rtxPipeline = it->rtxPipeline;
        m_rqPipeline  = it->rqPipeline;
        m_sbtBuffer   = it->sbtBuffer;
        m_sbtRegions  = it->sbtRegions;
        ++m_pipelineGeneration;
      }
      else
      {
        // Cached with the other SER setting: discard the pipelines (and the SBT built from the RTX
        // one) and let ensureShadersAndPipelines() rebuild them at the current setting.
        release(VK_NULL_HANDLE, it->rtxPipeline, it->rqPipeline, it->sbtBuffer);
        m_rtxPipeline = VK_NULL_HANDLE;
        m_rqPipeline  = VK_NULL_HANDLE;
        m_sbtBuffer   = {};
        m_sbtRegions  = {};
      }
      m_compiled = newKey;
      m_variantCache.erase(it);
      return true;
    }
  }

  // 3) Enforce LRU bound: drop oldest entries if over capacity.
  while(m_variantCache.size() > kVariantCacheMaxEntries)
  {
    const VariantCacheEntry& victim = m_variantCache.back();
    release(victim.shaderModule, victim.rtxPipeline, victim.rqPipeline, victim.sbtBuffer);
    m_variantCache.pop_back();
  }

  return false;  // miss: caller must (re)compile + (re)build pipelines + (re)build SBT
}

void PathTracer::releaseVariant(Resources& resources, VkShaderModule module, VkPipeline rtx, VkPipeline rq, nvvk::Buffer sbt, bool gpuIdle)
{
  if(module == VK_NULL_HANDLE && rtx == VK_NULL_HANDLE && rq == VK_NULL_HANDLE && sbt.buffer == VK_NULL_HANDLE)
    return;
  auto destroy = [device = m_device, alloc = &resources.allocator, module, rtx, rq, sbt]() mutable {
    vkDestroyShaderModule(device, module, nullptr);
    vkDestroyPipeline(device, rtx, nullptr);
    vkDestroyPipeline(device, rq, nullptr);
    if(sbt.buffer != VK_NULL_HANDLE)
      alloc->destroyBuffer(sbt);
  };
  if(gpuIdle || resources.app == nullptr)
    destroy();
  else
    resources.app->submitResourceFree(std::move(destroy));
}

//--------------------------------------------------------------------------------------------------
// The shader macros of a variant, after applyCommonShaderMacros()'s: what compileShader() compiles
// it with, and a variant job with it, so both find the same SPIR-V in the disk cache.
std::vector<std::pair<std::string, std::string>> PathTracer::variantMacros(const VariantKey& key) const
{
  std::vector<std::pair<std::string, std::string>> macros = {
      {"AVAILABLE_SER", std::to_string(m_supportSER)},
      {"WIREFRAME", std::to_string(int(key.wireframe))},
      // Debug-only visualization (incl. the OMM debug payload field). Compiled out of the normal
      // render path so it adds no payload/register cost; a mode switch triggers a recompile.
      {"USE_VISUALIZE", std::to_string(key.visualize ? 1 : 0)},
      // Ray queries against a BLAS with an opacity micromap need an explicit opt-in (see gltf_pathtrace.slang).
      {"USE_OMM", std::to_string(key.omm ? 1 : 0)},
      {"USE_OPENPBR", std::to_string(key.materialModel)},
  };
  nvvkgltf::SceneFeatureSet guide;
  guide.set(nvvkgltf::SceneFeatureSet::eDlssGuide, key.dlssGuide);
  nvvkgltf::appendPathTracerDlssShaderMacro(macros, guide, key.dlss);
  // Scene-aware optimal mode: every GLTF_USE_* set to 0 or 1 from the feature set. Never
  // MAT_EXT_X=0 (that would change GltfShadeMaterial layout while the host uploads the all-on struct).
  if(key.optimal)
    nvvkgltf::appendPathTracerOptimalMacros(macros, key.features);
  return macros;
}

//--------------------------------------------------------------------------------------------------
// The SBT of a ray tracing pipeline built from `module`
void PathTracer::createSbt(Resources& resources, VkPipeline pipeline, VkShaderModule module, nvvk::Buffer& sbtBuffer, nvvk::SBTGenerator::Regions& regions)
{
  RtxStages                               table(module);
  const VkRayTracingPipelineCreateInfoKHR info{
      .sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount = static_cast<uint32_t>(table.stages.size()),
      .pStages    = table.stages.data(),
      .groupCount = static_cast<uint32_t>(table.groups.size()),
      .pGroups    = table.groups.data(),
  };
  nvvk::SBTGenerator sbtGenerator;
  sbtGenerator.init(m_device, m_rtPipelineProperties);
  const size_t bufferSize = sbtGenerator.calculateSBTBufferSize(pipeline, info);
  NVVK_CHECK(resources.allocator.createBuffer(
      sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, sbtGenerator.getBufferAlignment()));
  NVVK_DBG_NAME(sbtBuffer.buffer);
  NVVK_CHECK(sbtGenerator.populateSBTBuffer(sbtBuffer.address, bufferSize, sbtBuffer.mapping));
  regions = sbtGenerator.getSBTRegions();
  sbtGenerator.deinit();
}

//--------------------------------------------------------------------------------------------------
// D15 (daz-viewer docs/retained-scene-plan.md, "Shader variants under incremental edits"): on the
// retained scene, a variant that only adds scene features to the optimal build is built off the
// frame. The live variant draws meanwhile, or a cached one with the same shader otherwise and
// features between the live one's and the wanted ones: the residency holds the edits that need
// more, so nothing drawn does. Anything else (a setting, the variant a scene replace wants, a
// first compile, a job that failed) is compiled before the frame, as before.
bool PathTracer::buildsOffTheFrame(Resources& resources, const PendingCompileWork& work)
{
  const VariantKey wanted = makeVariantKey(resources);
  if(!work.recompile || work.stalePipeline || !wanted.optimal || !resources.gpuView().variantMayBuildOffFrame()
     || (m_failedVariant && *m_failedVariant == wanted))
    return false;
  const bool rayQuery = m_renderTechnique == RenderTechnique::RayQuery;
  VariantKey live;
  bool       liveDraws = false;
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    live      = m_compiled;
    liveDraws = !m_compiledIsFallback && (rayQuery ? m_rqPipeline : m_rtxPipeline) != VK_NULL_HANDLE;
  }
  if(!live.optimal)
    return false;
  const auto standsIn = [&](const VariantKey& k) {
    VariantKey same = k;
    same.features   = wanted.features;
    return same == wanted && wanted.features.isSupersetOf(k.features) && k.features.isSupersetOf(live.features);
  };
  if(!liveDraws || !standsIn(live))
  {
    std::optional<VariantKey> cached;
    {
      std::lock_guard<std::mutex> lock(m_compileMutex);
      for(const VariantCacheEntry& e : m_variantCache)
        if(standsIn(e.key) && e.useSER == m_pipelineUseSER && (rayQuery ? e.rqPipeline : e.rtxPipeline) != VK_NULL_HANDLE)
        {
          cached = e.key;
          break;
        }
    }
    if(!cached || !swapVariant(resources, *cached, /*gpuIdle=*/false))
      return false;
  }
  if(!m_variantJobs.job())
    startVariantJob(resources, wanted);
  return true;
}

// The job: a snapshot of what the variant is built from, taken on the application thread
void PathTracer::startVariantJob(Resources& resources, const VariantKey& key)
{
  VariantCompileJob job{
      .key         = key,
      .macros      = variantMacros(key),
      .searchPaths = resources.slangCompiler.searchPaths(),
      .targets     = resources.slangCompiler.targets(),
      .options     = resources.slangCompiler.options(),
      // the technique of the view the variant draws (a sibling is built for the other view)
      .rayQuery    = m_supportRayQuery && (key.dlss ? m_techniqueDlss : m_techniqueStill) == RenderTechnique::RayQuery,
      .useSER      = m_useSER && m_supportSER,
      .layout      = m_pipelineLayout,
      .rtxFlags    = rtxPipelineFlags(resources.settings.opacityMicromapSupported, m_capturePipelineStats),
      .rqFlags     = m_capturePipelineStats ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) : 0u,
  };
  VariantKey live;
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    live = m_compiled;
  }
  LOGI("[PathTracer] Variant building off the frame: %s\n", live.shaderDiff(key).c_str());
  m_variantJobs.start(std::move(job), [this](const VariantCompileJob& j) { return buildVariant(j); });
}

// On the worker: the job's Slang compile (or the disk cache's SPIR-V) on a compiler of its own, then
// its module and pipeline. Nothing of Resources, and nothing destroyed but what it made and failed.
PathTracer::VariantBuilt PathTracer::buildVariant(const VariantCompileJob& job)
{
  const auto             start = std::chrono::steady_clock::now();
  nvslang::SlangCompiler compiler;  // a global session of its own: the application thread's is not shared
  compiler.targets() = job.targets;
  compiler.options() = job.options;
  compiler.addSearchPaths(job.searchPaths);
  nvvkgltf::applyCommonShaderMacros(compiler);
  for(const auto& [name, value] : job.macros)
    compiler.addMacro({name.c_str(), value.c_str()});
  const uint64_t        spirvKey = spirvCacheKey(compiler);
  std::vector<uint32_t> spirv;
  if(!loadCachedSpirv(spirvKey, spirv))
  {
    const auto compileStart = std::filesystem::file_time_type::clock::now();
    if(!compiler.compileFile("gltf_pathtrace.slang"))
    {
      LOGW("[PathTracer] Variant job: Slang compile failed\n%s\n", compiler.getLastDiagnosticMessage().c_str());
      return {};
    }
    spirv.assign(compiler.getSpirv(), compiler.getSpirv() + compiler.getSpirvSize() / sizeof(uint32_t));
    storeCachedSpirv(spirvKey, compiler, compileStart);
  }
  VariantBuilt                   built;
  const VkShaderModuleCreateInfo moduleInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size() * sizeof(uint32_t),
      .pCode    = spirv.data(),
  };
  if(vkCreateShaderModule(m_device, &moduleInfo, nullptr, &built.module) != VK_SUCCESS)
    return {};
  rememberModuleCode(built.module, spirv.data(), spirv.size() * sizeof(uint32_t));
  VkResult   result     = VK_ERROR_UNKNOWN;
  const auto buildStart = std::chrono::steady_clock::now();
  built.pipeline = createPathTracePipeline(m_device, m_pipelineCache.getCache(), job.layout, built.module, job.rayQuery,
                                           job.useSER ? 1 : 0, job.key.dlssTransparency, job.rtxFlags, job.rqFlags, result);
  if(built.pipeline)
    notePipelineBuild(moduleCode(built.module), job.rayQuery, job.useSER ? 1 : 0, job.key.dlssTransparency,
                      job.rayQuery ? job.rqFlags : job.rtxFlags,
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count());
  if(!built.pipeline)
  {
    LOGW("[PathTracer] Variant job: pipeline creation failed (%d)\n", int(result));
    vkDestroyShaderModule(m_device, built.module, nullptr);
    return {};
  }
  m_pipelineCache.saveAsync();
  built.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return built;
}

// At frame top: a finished job's variant into the cache, with its SBT, where the frame's
// tryRestoreCachedVariant() swaps it in. With DLSS on, the same features under the other DLSS
// state is built next, for the next camera start or stop.
void PathTracer::installBuiltVariant(Resources& resources)
{
  std::optional<std::pair<VariantCompileJob, VariantBuilt>> done = m_variantJobs.take();
  if(!done)
    return;
  const VariantCompileJob& job   = done->first;
  VariantBuilt&            built = done->second;
  if(!built.module)
  {
    LOGW("[PathTracer] The variant built off the frame failed; it is compiled before the frame instead\n");
    m_failedVariant = job.key;
    return;
  }
  if(job.layout != m_pipelineLayout)  // never used: no frame holds it
  {
    vkDestroyPipeline(m_device, built.pipeline, nullptr);
    vkDestroyShaderModule(m_device, built.module, nullptr);
    return;
  }
  VariantCacheEntry entry{.key = job.key, .shaderModule = built.module, .useSER = job.useSER};
  if(job.rayQuery)
    entry.rqPipeline = built.pipeline;
  else
  {
    entry.rtxPipeline = built.pipeline;
    createSbt(resources, built.pipeline, built.module, entry.sbtBuffer, entry.sbtRegions);
  }
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    for(auto it = m_variantCache.begin(); it != m_variantCache.end(); ++it)
      if(it->key == job.key)
      {
        releaseVariant(resources, it->shaderModule, it->rtxPipeline, it->rqPipeline, it->sbtBuffer, /*gpuIdle=*/false);
        m_variantCache.erase(it);
        break;
      }
    m_variantCache.insert(m_variantCache.begin(), entry);
    while(m_variantCache.size() > kVariantCacheMaxEntries)
    {
      const VariantCacheEntry& victim = m_variantCache.back();
      releaseVariant(resources, victim.shaderModule, victim.rtxPipeline, victim.rqPipeline, victim.sbtBuffer, /*gpuIdle=*/false);
      m_variantCache.pop_back();
    }
  }
  LOGI("[PathTracer] Variant built off the frame in %.0f ms: [%s]\n", built.ms, job.key.features.toString().c_str());

  if(!isDlssEnabled())
    return;
  VariantKey sibling = job.key;
  sibling.dlss       = !job.key.dlss;
  bool known         = false;
  {
    std::lock_guard<std::mutex> lock(m_compileMutex);
    known = m_compiled == sibling
            || std::any_of(m_variantCache.begin(), m_variantCache.end(),
                           [&](const VariantCacheEntry& e) { return e.key == sibling; });
  }
  if(!known)
    startVariantJob(resources, sibling);
}

// A job's work thrown away (teardown, a shader reload, a new pipeline layout): waited for, and what
// it built destroyed, which no frame has used
void PathTracer::discardVariantJob()
{
  m_variantJobs.join();
  if(std::optional<std::pair<VariantCompileJob, VariantBuilt>> done = m_variantJobs.take())
  {
    vkDestroyPipeline(m_device, done->second.pipeline, nullptr);
    vkDestroyShaderModule(m_device, done->second.module, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Destroy every cached variant (called from onDetach). Live handles are NOT touched here -
// onDetach's explicit vkDestroyPipeline / vkDestroyShaderModule cover the active slots; the active SBT
// is freed in onDetach via resources.allocator.destroyBuffer(m_sbtBuffer).
void PathTracer::destroyVariantCache(Resources& resources)
{
  for(auto& e : m_variantCache)
  {
    if(e.shaderModule != VK_NULL_HANDLE)
      vkDestroyShaderModule(m_device, e.shaderModule, nullptr);
    if(e.rtxPipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(m_device, e.rtxPipeline, nullptr);
    if(e.rqPipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(m_device, e.rqPipeline, nullptr);
    if(e.sbtBuffer.buffer != VK_NULL_HANDLE)
      resources.allocator.destroyBuffer(e.sbtBuffer);
  }
  m_variantCache.clear();
}


//--------------------------------------------------------------------------------------------------
// Auto SPP starts over from one sample per frame, e.g. on a new target: the throughput search measures again
void PathTracer::restartAdaptiveSampling()
{
  m_adaptiveSamples  = MIN_SAMPLES_PER_PIXEL;
  m_throughputSearch = {};
  m_frameRateControl.restart();
}

//--------------------------------------------------------------------------------------------------
// Update adaptive sampling based on frame timing
void PathTracer::updateAdaptiveSampling(Resources& resources)
{
  // DLSS alone traces one sample per frame. With OptiX or OIDN, the frames it renders do
  // (setupPushConstant), and Auto SPP climbs over the frames after them from the reset below.
  if(m_dlssFrame)
    return;

  if(!m_adaptiveSampling || !m_profilerTimeline)
    return;

  // Another pipeline renders the frames (Rendering Pipeline, SER, Optimize Shader, a shader reload):
  // its cost per sample differs, so the count is found again, which also shows the change in throughput
  const uint64_t pipelineKey = (uint64_t(m_pipelineGeneration.load()) << 1) | uint64_t(m_renderTechnique);
  if(pipelineKey != m_adaptivePipelineKey)
  {
    m_adaptivePipelineKey = pipelineKey;
    restartAdaptiveSampling();
  }

  // Wall time since the previous rendered frame: once the count is steady, what a frame costs end to
  // end, the path trace and everything else in the frame (tonemapper, submit, the CPU loop)
  const auto   now        = std::chrono::steady_clock::now();
  const double wallMs     = std::chrono::duration<double, std::milli>(now - m_lastAdaptiveFrameTime).count();
  m_lastAdaptiveFrameTime = now;

  // Reset samples when frame count resets to 0 (scene change, etc.), so the first frames after a change
  // stay responsive. The throughput search keeps what it found: it serves runs nobody interacts with.
  if(resources.frameCount == 0 || m_handoffFrame)
  {
    m_accumulationFrames = 0;
    if(m_performanceTarget != PerformanceTarget::eThroughput)
    {
      // Half the count the last view settled on: a frame of it still costs under the target, and the
      // climb from there to this view's count is half as long. A view much dearer than the last
      // steps down from it, one a frame.
      m_adaptiveSamples = std::max(MIN_SAMPLES_PER_PIXEL, m_adaptiveSamples / 2);
      m_frameRateControl.restart();  // a new view: climb again to its count
    }
    return;
  }

  // Don't adjust on the first few frames to allow for stabilization (the accumulation's frames: the
  // GPU time of a DLSS frame before them is no measure of a sample)
  if(++m_accumulationFrames < 5)
    return;

  // The profiler reports a GPU time frames after it was recorded. The throughput search compares counts,
  // so it waits until every frame the time can come from traced the current count (not a frame ended
  // early on a stop, nor the count before). The frame-rate targets step by one, where that lag is small.
  if(m_performanceTarget == PerformanceTarget::eThroughput
     && std::any_of(m_recentFrameSamples.begin(), m_recentFrameSamples.end(),
                    [&](int samples) { return samples != m_adaptiveSamples; }))
    return;

  // Get timing information for the path tracing section (the technique's timer)
  nvutils::ProfilerTimeline::TimerInfo timerInfo;
  static std::string apiName;  // an output nobody reads; kept so the per-frame call allocates nothing
  const std::string& timerName = (m_renderTechnique == RenderTechnique::RayQuery) ? kTimerPathTraceRq : kTimerPathTraceRtx;

  if(m_profilerTimeline->getFrameTimerInfo(timerName, timerInfo, apiName))
  {
    // Convert from microseconds to milliseconds
    double currentFrameTimeMs = timerInfo.gpu.last / 1000.0;

    // Throughput counts the whole frame: several samples per frame pay off by sharing the per-frame
    // work, which the path-trace timer does not see
    if(m_performanceTarget == PerformanceTarget::eThroughput)
      m_adaptiveSamples =
          m_throughputSearch.update(m_adaptiveSamples, wallMs, getTargetFrameTimeMs(), MAX_ADAPTIVE_SAMPLES_PER_PIXEL);
    else
      m_adaptiveSamples = m_frameRateControl.update(m_adaptiveSamples, currentFrameTimeMs, getTargetFrameTimeMs(),
                                                    MIN_SAMPLES_PER_PIXEL, MAX_ADAPTIVE_SAMPLES_PER_PIXEL);
  }
}


void PathTracer::updateStatistics(Resources& resources)
{


  // Update rolling average throughput calculation using wall-clock time
  {
    // Time elapsed for this frame (wall-clock time from user perspective)
    float wallClockFrameTime = std::max(ImGui::GetIO().DeltaTime, 1e-6f);

    // Total number of pixels in the image
    VkExtent2D imageSize   = resources.gBuffers.getSize();
    uint64_t   totalPixels = static_cast<uint64_t>(imageSize.width) * static_cast<uint64_t>(imageSize.height);

    // Calculate mega-sample-pixels per second of wall-clock time
    // This tells the user how much rendering work is being done per real-world second
    const float MEGA_SCALE_FACTOR = 1000000.0f;  // Convert to mega-sample-pixels
    float       megaSamplePixelsPerSecond =
        (static_cast<float>(m_frameSamples) * static_cast<float>(totalPixels) / MEGA_SCALE_FACTOR) / wallClockFrameTime;

    // Update rolling average with wall-clock throughput for this frame
    m_throughputRollingAvg.addValue(megaSamplePixelsPerSecond);
  }

  // Track total samples accumulated. DLSS's frames before a hand-off (frame 0, the delay's) add
  // nothing to it: the hand-off frame starts it over, and their count must not reach the cap first.
  if(!(m_dlssFrame && isAccumulationDenoiserEnabled()))
    m_totalSamplesAccumulated += m_frameSamples;
  std::rotate(m_recentFrameSamples.begin(), m_recentFrameSamples.begin() + 1, m_recentFrameSamples.end());
  m_recentFrameSamples.back()  = m_frameSamples;
  resources.accumulatedSamples = m_totalSamplesAccumulated;
}

//--------------------------------------------------------------------------------------------------
// Bind the selected technique: its pipeline, descriptor sets 0 (TLAS, output images, textures), 2 (the
// HDR environment) and 3 (the physical sky's LUTs for aerial perspective, bound whatever the
// environment is; the shader reads them only when the frame says the sky is the physical one), and
// the pushed set 1. Once per frame; dispatchRays() launches it, for the probe and for the paths.
// prepareFrame() builds the pipeline before a frame is recorded, so a missing one is a logic error,
// not something to build here, inside command recording and with no BusyWindow.
bool PathTracer::bindTechnique(VkCommandBuffer cmd, Resources& resources)
{
  const bool                rayQuery  = m_renderTechnique == RenderTechnique::RayQuery;
  const VkPipelineBindPoint bindPoint = rayQuery ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
  const VkPipeline          pipeline  = rayQuery ? m_rqPipeline : m_rtxPipeline;
  assert(pipeline != VK_NULL_HANDLE && "PathTracer: no pipeline at record time; prepareFrame() builds it first");
  if(pipeline == VK_NULL_HANDLE)
  {
    LOGW("PathTracer: no %s pipeline at record time; frame skipped\n", rayQuery ? "ray-query" : "ray-tracing");
    return false;
  }

  vkCmdBindPipeline(cmd, bindPoint, pipeline);
  vkCmdBindDescriptorSets(cmd, bindPoint, m_pipelineLayout, 0, 1, &resources.descriptorSet, 0, nullptr);
  const VkDescriptorSet hdrDescSet = resources.hdrIbl.getDescriptorSet();
  vkCmdBindDescriptorSets(cmd, bindPoint, m_pipelineLayout, 2, 1, &hdrDescSet, 0, nullptr);
  vkCmdBindDescriptorSets(cmd, bindPoint, m_pipelineLayout, 3, 1, &resources.skyLutDescriptorSet, 0, nullptr);
  pushDescriptorSet(cmd, resources, bindPoint);
  return true;
}

void PathTracer::dispatchRays(VkCommandBuffer cmd, VkExtent2D size) const
{
  if(m_renderTechnique == RenderTechnique::RayQuery)
  {
    const VkExtent2D numGroups = nvvk::getGroupCounts(size, VkExtent2D{PT_WORKGROUP_X, PT_WORKGROUP_Y});
    vkCmdDispatch(cmd, numGroups.width, numGroups.height, 1);
  }
  else
  {
    vkCmdTraceRaysKHR(cmd, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                      size.width, size.height, 1);
  }
}

//--------------------------------------------------------------------------------------------------
// The medium the camera is in, for every camera path to start in (probeCameraMedium in
// gltf_pathtrace.slang): one thread of the bound pipeline, before the frame's paths.
void PathTracer::probeCameraMedium(VkCommandBuffer cmd, const shaderio::PathtracePushConstant& pushConst)
{
  auto timerSection = m_profiler->cmdFrameSection(cmd, "Camera medium");

  // The previous frame's paths may still be reading it
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

  shaderio::PathtracePushConstant probe = pushConst;
  probe.flags |= shaderio::ePtProbeCameraMedium;
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(probe), &probe);
  dispatchRays(cmd, {1, 1});

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pushConst), &pushConst);
}

void PathTracer::denoiseDlss(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_DLSS)
  auto timerSection = m_profiler->cmdFrameSection(cmd, "DLSS");

  // #DLSS - Denoising.
  const glm::mat4& view = resources.cameraManip->getViewMatrix();
  const glm::mat4& proj = resources.cameraManip->getPerspectiveMatrix();

  m_dlss->setResponsivityMask(m_hairResponsivity != 0.0F);
  m_dlss->evaluate(cmd, view, proj);

  // The selection image the blit reads was written by the path tracer's ray-tracing or compute
  // stage (the barrier after traceRays only made it visible to compute)
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

  {
    // Blit the selection image from the DLSS GBuffer (different resolution) to the Renderer GBuffer Selection
    const VkExtent2D innerSz   = m_dlss->getRenderSize();
    VkOffset3D       minCorner = {0, 0, 0};
    VkImageBlit      blitRegions{
             .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
             .srcOffsets     = {minCorner, {int(innerSz.width), int(innerSz.height), 1}},
             .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
             .dstOffsets = {minCorner, {int(resources.gBuffers.getSize().width), int(resources.gBuffers.getSize().height), 1}},
    };
    vkCmdBlitImage(cmd, m_dlss->getRrImage(shaderio::OutputImage::eSelectImage), VK_IMAGE_LAYOUT_GENERAL,
                   resources.gBuffers.getColorImage(Resources::eImgSelection), VK_IMAGE_LAYOUT_GENERAL, 1, &blitRegions,
                   VK_FILTER_NEAREST);  // ObjectID in .r, no interpolation

    // The blit's write, and DLSS's compute write of the rendered image, before the compute passes
    // that read them (the hand-off, the tonemapper)
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_READ_BIT);
  }
#endif
}

shaderio::PathtracePushConstant PathTracer::setupPushConstant(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderingSize)
{
  // Current frame count, can be overridden by DLSS
  int frameCount = resources.frameCount;

  // Handle frame reset detection (needed for both adaptive and non-adaptive modes). The hand-off
  // frame after DLSS's starts over too: DLSS's denoised image is no sample of the accumulation.
  if(resources.frameCount == 0 || m_handoffFrame)
  {
    // Reset sample counter when scene/camera changes. Resources' copy too: what runs before
    // updateStatistics() this frame (the OptiX denoiser) must not see the previous accumulation's count.
    m_totalSamplesAccumulated    = 0;
    resources.accumulatedSamples = 0;
    resources.adaptiveDone       = false;
    m_accumulationId++;
  }

  // Adjust focal distance if auto-focus is enabled
  if(m_autoFocus)
  {
    m_pushConst.focalDistance =
        static_cast<float>(glm::length(resources.cameraManip->getEye() - resources.cameraManip->getCenter()));
  }

  // Samples per frame: the setting (ptSamples), Auto SPP's count, or 1 with DLSS
  int  samplesPerFrame = m_adaptiveSampling ? m_adaptiveSamples : m_pushConst.numSamples;
  bool useDlss         = false;
#if defined(USE_DLSS)
  useDlss = getEffectiveDlssEnabled(resources);
  if(useDlss)
    samplesPerFrame = 1;

  // Halton jitter and frame-index counter. GltfRenderer::renderFrame writes the frame info with a
  // zero jitter every frame; only a DLSS frame patches it.
  const Dlss::FrameContext fc = m_dlss->beginFrame();
  if(useDlss)
  {
    frameCount = static_cast<int>(fc.frameIndex);

    // That earlier full-struct write is a prior transfer write to the same buffer, so order this second
    // update after it (and after the compute-visibility barrier that followed it) to avoid a WRITE_AFTER_WRITE.
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Patch only the 8-byte jitter slot; the rest of SceneFrameInfo was already written this frame.
    vkCmdUpdateBuffer(cmd, resources.bFrameInfo.buffer, offsetof(shaderio::SceneFrameInfo, jitter), sizeof(glm::vec2), &fc.jitter);
  }
#endif

  m_pushConst.frameCount = frameCount;
  m_pushConst.renderSize = {renderingSize.width, renderingSize.height};
  // First-frame flag must use resources.frameCount so depth and ObjectID are written on app frame 0 and after
  // every reset (camera/scene change). The local frameCount can be overridden to a Halton index when DLSS
  // is on and never resets; using it for ePtFirstFrame would only write depth once and break selection/depth.
  m_pushConst.flags = (resources.frameCount == 0 ? shaderio::ePtFirstFrame : 0);
  // The selection image only when something reads it on a DLSS frame (SelectionWrite): the selection ray
  // per pixel no one read was 8% of the moving trace (30.2 -> 27.9 ms, the harness figure at Optimal input)
  const bool selectionRead = !resources.selectedRenderNodes.empty() || resources.hoverInViewport;
  m_pushConst.flags |= m_selectionWrite.due(resources.frameCount == 0, useDlss, selectionRead) ? shaderio::ePtWriteSelection : 0;
  m_pushConst.flags |= m_shadowTransmission ? shaderio::ePtShadowTransmission : 0;
  m_pushConst.flags |= (m_scatterAsThin == 1 || (m_scatterAsThin == 2 && useDlss)) ? shaderio::ePtScatterAsThin : 0;
  m_pushConst.flags |= m_volumeDiffuseExit ? shaderio::ePtVolumeDiffuseExit : 0;
  m_pushConst.flags |= m_diffuseEntry ? shaderio::ePtDiffuseEntry : 0;
  m_pushConst.flags |= m_fireflyClampMode == 1 ? shaderio::ePtClampIndirect : 0;
  m_pushConst.flags |= m_approximateCaustics ? shaderio::ePtApproximateCaustics : 0;
  m_pushConst.flags |= (m_regularize == 2 || (m_regularize == 1 && useDlss)) ? shaderio::ePtRegularize : 0;
  m_pushConst.maxDepth   = sample_schedule::frameMaxDepth(m_maxDepth, m_maxDepthDlss, useDlss);
  m_pushConst.hairBudget = useDlss ? m_hairBouncesDlss : m_hairBounces;
  m_pushConst.hairRouletteDlss = m_hairRouletteDlss;
  m_pushConst.hairRouletteFrom = m_hairRouletteFrom;
  m_pushConst.hairCutDlss      = m_hairCutDlss;
  m_pushConst.hairCutTail      = m_hairCutTail;
  m_pushConst.hairResponsivity = useDlss ? m_hairResponsivity : 0.0F;
  m_pushConst.flags |= (m_adaptiveStop && !useDlss) ? shaderio::ePtAdaptiveStop : 0;
  // in displayed brightness, with the exposure the host sets (auto exposure, which it never uses, has none)
  m_pushConst.adaptiveThreshold = resources.tonemapperData.autoExposure ?
                                      m_adaptiveThreshold :
                                      adaptiveSceneThreshold(m_adaptiveThreshold, resources.tonemapperData.exposure);
  m_pushConst.adaptiveMin       = m_adaptiveMin;
  const adaptive_relax::Table relax = m_adaptiveRelax ? adaptive_relax::build(resources.tonemapperData) : adaptive_relax::Table{};
  m_pushConst.adaptiveRelaxBase     = relax.base;
  std::memcpy(m_pushConst.adaptiveRelax, relax.relax.data(), sizeof(m_pushConst.adaptiveRelax));
  m_pushConst.adaptiveActive = (m_pushConst.flags & shaderio::ePtAdaptiveStop) && m_adaptiveDone > 0.0F ?
                                   countAdaptivePixels(cmd, resources, renderingSize) :
                                   nullptr;
  m_pushConst.flags |= m_lightsVisible ? shaderio::ePtLightsVisible : 0;
  m_pushConst.flags |= m_exitFresnel ? shaderio::ePtExitFresnel : 0;
  m_pushConst.flags |= m_walkGuiding ? shaderio::ePtWalkGuiding : 0;
  m_pushConst.flags |= m_lightTree ? shaderio::ePtLightTree : 0;
  m_pushConst.flags |= m_simplifyScatter ? shaderio::ePtSimplifyScatter : 0;
  m_pushConst.flags |= (m_sampler == 1 && !useDlss) ? shaderio::ePtSobol : 0;
  m_pushConst.flags |= m_fullLobeEval >= 1 ? shaderio::ePtLobeSumPunctual : 0;
  m_pushConst.flags |= m_fullLobeEval >= 2 ? shaderio::ePtLobeMixture : 0;
  m_pushConst.flags |= m_fullLobeEval >= 3 ? shaderio::ePtLobeMixtureDirect : 0;
#if defined(USE_DLSS)
  m_pushConst.flags |= useDlss ? shaderio::ePtUseDlss : 0;
#endif
  // Guides of the denoiser of the accumulation (OptiX or OIDN, one at a time, never with DLSS): accumulated
  // like the color into that denoiser's images, over the samples since they last restarted. Each
  // denoiser is told whether its guides are written, so a denoiser turned on restarts them.
  bool denoiserGuides      = false;
  m_pushConst.guideSamples = 0;
#if defined(USE_OPTIX_DENOISER)
  const bool useOptixDenoiser = getEffectiveOptixEnabled(resources);
  const int  optixGuides      = m_optix->updateGuideState(useOptixDenoiser, m_totalSamplesAccumulated);
  if(useOptixDenoiser)
    m_pushConst.guideSamples = optixGuides;
  denoiserGuides |= useOptixDenoiser;
#endif
#if defined(USE_OIDN_DENOISER)
  const bool useOidnDenoiser = getEffectiveOidnEnabled(resources);
  const int  oidnGuides      = m_oidn->updateGuideState(useOidnDenoiser, m_totalSamplesAccumulated);
  if(useOidnDenoiser)
    m_pushConst.guideSamples = oidnGuides;
  denoiserGuides |= useOidnDenoiser;
#endif
  m_pushConst.flags |= denoiserGuides ? shaderio::ePtDenoiserGuides : 0;
  // The frame ends exactly on Max Samples and on the counts --outputSamples saves, whatever the
  // samples per frame
  m_frameSamples           = sample_schedule::frameSamples(m_totalSamplesAccumulated, samplesPerFrame,
                                                           resources.settings.maxSamples, resources.sampleStops);
  m_pushConst.totalSamples = m_totalSamplesAccumulated;
  m_pushConst.frameInfo    = (shaderio::SceneFrameInfo*)resources.bFrameInfo.address;
  m_pushConst.gltfScene    = (shaderio::GltfScene*)resources.gpuView().sceneDesc();
  m_pushConst.mouseCoord   = nvapp::ElementDbgPrintf::getMouseCoord();  // Use for debugging: printf in shader
  // Only a scene with a KHR_materials_volume material has a medium the camera can be in
  m_pushConst.cameraMedium = (m_probeCameraMedium && resources.currentFeatureSet.has(nvvkgltf::SceneFeatureSet::eVolume)) ?
                                 (shaderio::CameraMedium*)m_cameraMediumBuffer.address :
                                 nullptr;

  // #DLSS instance motion vectors: previous-frame render-node transforms (snapshotted at frame start
  // in GltfRenderer::onRender, only on frames where transforms actually changed). Bound only when DLSS
  // runs this frame AND the snapshot ran (dlssInstanceMotionActive); null otherwise so the shader
  // falls back to exact camera-only motion. The shader null-checks the pointer, so this is crash-safe.
#if defined(USE_DLSS)
  // (a view that keeps its own tables, the retained scene, gives them; legacy's are the renderer's)
  const VkDeviceAddress prevO2W = resources.gpuView().prevObjectToWorldTable() ? resources.gpuView().prevObjectToWorldTable() :
                                                                                 resources.transformCompute.prevObjectToWorldAddress();
  const VkDeviceAddress prevPos = resources.gpuView().prevPositionTable() ? resources.gpuView().prevPositionTable() :
                                                                            resources.prevPositions.tableAddress();
  m_pushConst.prevRenderNodeObjectToWorld = (useDlss && resources.dlssInstanceMotionActive) ? (shaderio::float4x4*)prevO2W : nullptr;
  // #DLSS deformation motion: the positions from before this frame's skin / morph / vertex uploads,
  // bound only on frames where those ran (GltfRenderer::snapshotPrevPositions).
  m_pushConst.prevVertexPositions = (useDlss && resources.dlssDeformMotionActive) ? (shaderio::float3**)prevPos : nullptr;
#else
  m_pushConst.prevRenderNodeObjectToWorld = nullptr;
  m_pushConst.prevVertexPositions         = nullptr;
#endif

  // Angular size of one pixel (radians): 2 * tan(fovY/2) / viewportHeight, the shader's former
  // computePixelAngle(). The camera's projection is a symmetric perspective (or orthographic)
  // matrix with a Y flip, so tan(fovY/2) (the half height, orthographic) is 1 / |proj[1][1]| and no
  // inverse is needed. Constant across the dispatch, so it is computed once here instead of per ray.
  const double projYY    = resources.cameraManip->getPerspectiveMatrix()[1][1];
  m_pushConst.pixelAngle = static_cast<float>(2.0 / (std::max(std::abs(projYY), 1e-9) * std::max(double(renderingSize.height), 1.0)));

  // numSamples is the setting (ptSamples); the shader gets this frame's count
  shaderio::PathtracePushConstant pushConst = m_pushConst;
  pushConst.numSamples                      = m_frameSamples;
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PathtracePushConstant), &pushConst);
  return pushConst;
}

//--------------------------------------------------------------------------------------------------
// Returns true iff DLSS will actually denoise this frame (strict): NGX is up, the inner GBuffer
// is populated, the user enabled DLSS-RR, AND the OptiX/OIDN-exclusivity rule lets DLSS run.
bool PathTracer::getEffectiveDlssEnabled(const Resources& resources) const
{
#if defined(USE_DLSS)
  // decided at the frame's start (onRender); outside a frame, the last frame's
  (void)resources;
  return m_dlssFrame;
#else
  (void)resources;
  return false;
#endif
}

//--------------------------------------------------------------------------------------------------
bool PathTracer::isAccumulationDenoiserEnabled() const
{
  bool enabled = false;
#if defined(USE_OPTIX_DENOISER)
  enabled |= m_optix->isEnabled();
#endif
#if defined(USE_OIDN_DENOISER)
  enabled |= m_oidn->isEnabled();
#endif
  return enabled;
}

//--------------------------------------------------------------------------------------------------
// The DLSS hand-off: the accumulation denoiser it hands off to, the denoises so far, and when it ends
std::optional<DenoiserHandoff::Denoiser> PathTracer::handoffDenoiser(const Resources& resources) const
{
#if defined(USE_OPTIX_DENOISER)
  if(getEffectiveOptixEnabled(resources))
    return DenoiserHandoff::Denoiser{.view         = DisplayBuffer::eOptixDenoised,
                                     .image        = m_optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferDenoised),
                                     .autoDenoise  = m_optix->isAutoDenoiseActive(),
                                     .denoiseCount = accumulationDenoiseCount()};
#endif
#if defined(USE_OIDN_DENOISER)
  if(getEffectiveOidnEnabled(resources))
    return DenoiserHandoff::Denoiser{.view         = DisplayBuffer::eOidnDenoised,
                                     .image        = m_oidn->getDescriptorImageInfo(OidnDenoiser::eGBufferDenoised),
                                     .autoDenoise  = m_oidn->isAutoDenoiseActive(),
                                     .denoiseCount = accumulationDenoiseCount()};
#endif
  return std::nullopt;
}

uint32_t PathTracer::accumulationDenoiseCount() const
{
  uint32_t count = 0;
#if defined(USE_OPTIX_DENOISER)
  count += m_optix->denoiseCount();
#endif
#if defined(USE_OIDN_DENOISER)
  count += m_oidn->denoiseCount();
#endif
  return count;
}

int PathTracer::handoffEnd(const Resources& resources) const
{
  return (m_dlssFrame0 && m_handoff.spp() > 0) ? std::min(m_handoff.spp(), resources.settings.maxSamples) : 0;
}

const VkDescriptorImageInfo* PathTracer::handoffImage(const Resources& resources) const
{
  const DisplayBuffer view = resources.settings.displayBuffer;
  if(view != DisplayBuffer::eOptixDenoised && view != DisplayBuffer::eOidnDenoised)
    return nullptr;
  return m_handoff.displayImage();
}

//--------------------------------------------------------------------------------------------------
bool PathTracer::keepOneAccumulationDenoiser(Resources& resources)
{
  bool restart = false;
#if defined(USE_OPTIX_DENOISER) && defined(USE_OIDN_DENOISER)
  if(m_optix->isEnabled() && m_oidn->isEnabled())
  {
    if(!m_optixWasOn && m_oidnWasOn)  // OptiX was turned on while OIDN was
      m_oidn->disable(resources);
    else
      restart = m_optix->disable(resources);
  }
  m_optixWasOn = m_optix->isEnabled();
  m_oidnWasOn  = m_oidn->isEnabled();
#endif
  return restart;
}

//--------------------------------------------------------------------------------------------------
// Determine if OptiX should actively denoise this frame
bool PathTracer::getEffectiveOptixEnabled(const Resources& resources) const
{
#if defined(USE_OPTIX_DENOISER)
  bool optixEnabled = m_optix->isEnabled();
#if defined(USE_DLSS)
  // OptiX-exclusivity rule: when DLSS-RR is also currently engaged (loading or active), DLSS
  // owns frame 0 and OptiX takes 1+. We treat eLoading the same as eActive here so the rule
  // doesn't flip mid-init (otherwise OptiX would run on frame 0 during the DLSS-RR async-init
  // window, then yield once DLSS becomes Active -- visibly inconsistent).
  const auto s           = m_dlss->state();
  const bool dlssEngaged = (s == Dlss::State::eLoading || s == Dlss::State::eActive);
  if(dlssEngaged && optixEnabled)
  {
    return resources.frameCount > 0 && !m_dlssFrame;
  }
#endif

  // Normal behavior when DLSS is off
  return optixEnabled;
#else
  return false;
#endif
}

//--------------------------------------------------------------------------------------------------
// Determine if OIDN should write its guides and denoise this frame. Same exclusivity rule as OptiX:
// with DLSS-RR also engaged, DLSS owns frame 0 and OIDN takes 1+.
bool PathTracer::getEffectiveOidnEnabled(const Resources& resources) const
{
#if defined(USE_OIDN_DENOISER)
  const bool oidnEnabled = m_oidn->isEnabled();
#if defined(USE_DLSS)
  const auto s           = m_dlss->state();
  const bool dlssEngaged = (s == Dlss::State::eLoading || s == Dlss::State::eActive);
  if(dlssEngaged && oidnEnabled)
    return resources.frameCount > 0 && !m_dlssFrame;
#endif
  return oidnEnabled;
#else
  return false;
#endif
}
