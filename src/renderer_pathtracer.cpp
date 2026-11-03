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

#include "renderer_pathtracer.hpp"
#include "sample_schedule.hpp"
#include "../shaders/adaptive_stop.h.slang"  // adaptiveSceneThreshold
#include "../shaders/hit_signal.h.slang"
#include "adaptive_relax.hpp"

#include "scene_shader_macros.hpp"
#include "pipeline_build_scope.hpp"

namespace {
// The path-trace profiler sections. Named once: cmdFrameSection() and getFrameTimerInfo() take a
// std::string, and both are called every frame.
const std::string kTimerPathTraceRq  = "Path Trace (RQ)";
const std::string kTimerPathTraceRtx = "Path Trace (RTX)";
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
    m_buildRecord.load(cacheRoot / PipelineBuildRecord::kFileName);
    shader_work_detail::lastBuildMs() = int64_t(m_buildRecord.lastCompiledMs());
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
  // command line, the app or MCP: otherwise new frames are averaged into the old image, and
  // once the frame count has reached maxFrames nothing renders at all.
  const nvutils::ParameterBase::CallbackSuccess reset = [resetAccumulation](const nvutils::ParameterBase* const) {
    if(resetAccumulation)
      resetAccumulation();
  };
  // PathTracer-specific settings. Persist marks what survives a restart (see settings_registry.hpp).
  // ptSamples: the adaptive-sampling autopilot clamps to the same range at runtime.
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
  settings->add({.name            = "ptGlassBounces",
                 .help            = "PathTracer: Events on clear glass (smooth, transmissive, nothing scattering inside) a path "
                                    "takes before they count as surface bounces; 0: every one counts",
                 .callbackSuccess = reset},
                &m_pushConst.glassBudget, Persist::eNo, 0, 64);
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
  // ensureShadersAndPipelines() on the next frame -- which is what makes it settable mid-run, not
  // just on the command line. Silently ignored when the device lacks SER.
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
  resources.allocator.destroyBuffer(m_hitCounts);
  for(nvvk::Buffer& b : m_hitReadback)
    resources.allocator.destroyBuffer(b);
  m_hitReadback.clear();
  m_hitCopied.clear();
  m_hitFrames = 0;
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

//--------------------------------------------------------------------------------------------------
// Bring the live shader, pipeline, and denoiser targets in line with the settings, however those
// were changed (UI, command line, sequence script, MCP). Nothing but compares when they already
// match. Returns false when this frame must be skipped because an async build was started.
bool PathTracer::prepareFrame(Resources& resources)
{
  // A variant already in the cache is swapped in right here: only handles move (what the previous
  // frame may still use is released through the app's frame ring), so there is no queue wait, no
  // BusyState and no accumulation reset. Anything else goes to an async job that shows the BusyState.
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
    denoiseDlss(cmd, resources, (pushConst.flags & shaderio::ePtWriteSelection) != 0);
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
  // a view whose samples ended before its window did counted what it counted, and traces nothing from here
  if(m_hitFrames > 0)
    copyHitCounts(cmd, resources);
  takeHitCounts(resources);
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

uint32_t* PathTracer::countHits(VkCommandBuffer cmd, Resources& resources)
{
  static_assert(HIT_SIGNAL_BINS == rs::kHitBins);
  static_assert(sizeof(shaderio::PathtracePushConstant) <= 256, "the least a device has to offer for push constants");
  // far more materials than a scene that fits memory has; those past it just aren't counted
  constexpr uint32_t     kSlots  = 4096;
  constexpr VkDeviceSize kBytes  = kSlots * HIT_SIGNAL_BINS * sizeof(uint32_t);
  constexpr int          kFrames = 8;  // at one pixel in sixteen, half a million hits at 1080p
  if(!resources.app || !resources.retainedGpu || !resources.retainedScene || !m_countHits)
    return nullptr;
  const uint32_t frameSlots = std::max(resources.app->getFrameCycleSize(), 1u);
  if(m_hitReadback.size() != frameSlots)
  {
    for(nvvk::Buffer& b : m_hitReadback)
      resources.allocator.destroyBuffer(b);
    m_hitReadback.assign(frameSlots, {});
    for(nvvk::Buffer& b : m_hitReadback)
    {
      NVVK_CHECK(resources.allocator.createBuffer(b, kBytes, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
                                                  VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
      NVVK_DBG_NAME(b.buffer);
    }
    m_hitCopied.assign(frameSlots, 0);
  }
  if(!m_hitCounts.buffer)
  {
    NVVK_CHECK(resources.allocator.createBuffer(m_hitCounts, kBytes,
                                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                    | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
    NVVK_DBG_NAME(m_hitCounts.buffer);
  }
  takeHitCounts(resources);

  // Frame 0 is whatever moved it; what is at rest after that counts. A window a move cut short says nothing.
  if(resources.frameCount == 0)
  {
    m_hitFrames = 0;
    return nullptr;
  }
  if(m_hitFrames < 0)
    return nullptr;
  if(m_hitFrames == 0)
  {
    vkCmdFillBuffer(cmd, m_hitCounts.buffer, 0, VK_WHOLE_SIZE, 0);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  }
  if(m_hitFrames < kFrames)
  {
    m_hitFrames++;
    m_pushConst.hitSignalSlots = int(std::min(resources.retainedGpu->residency().materialSlotCount(), kSlots));
    return reinterpret_cast<uint32_t*>(m_hitCounts.address);
  }
  copyHitCounts(cmd, resources);
  return nullptr;
}

void PathTracer::takeHitCounts(Resources& resources)
{
  if(m_hitCopied.empty())
    return;
  // The application waited for the frame that last used this slot
  const uint32_t slot = resources.app->getFrameCycleIndex() % uint32_t(m_hitCopied.size());
  if(std::exchange(m_hitCopied[slot], uint8_t(0)))
    resources.retainedGpu->setMaterialHits(*resources.retainedScene, {reinterpret_cast<const uint32_t*>(m_hitReadback[slot].mapping), size_t(m_hitReadback[slot].bufferSize / sizeof(uint32_t))});
}

void PathTracer::copyHitCounts(VkCommandBuffer cmd, Resources& resources)
{
  const uint32_t slot = resources.app->getFrameCycleIndex() % uint32_t(m_hitCopied.size());
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
  const VkBufferCopy region{0, 0, m_hitReadback[slot].bufferSize};
  vkCmdCopyBuffer(cmd, m_hitCounts.buffer, m_hitReadback[slot].buffer, 1, &region);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_HOST_READ_BIT);
  m_hitCopied[slot] = 1;
  m_hitFrames       = -1;
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

  // A slot below the count that this frame's images leave empty (a still's DLSS guide slots under its
  // convergence, slot 9) gets the rendered image: a null view is invalid without nullDescriptor, and
  // the shader writes no slot the frame didn't fill
  for(uint32_t i = 0; i < outputCount; ++i)
    if(outputImages[i].imageView == VK_NULL_HANDLE)
      outputImages[i] = outputImages[shaderio::eResultImage];

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
// not something to build here, inside command recording and with no BusyState.
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

void PathTracer::denoiseDlss(VkCommandBuffer cmd, Resources& resources, bool selectionWritten)
{
#if defined(USE_DLSS)
  auto timerSection = m_profiler->cmdFrameSection(cmd, "DLSS");

  // #DLSS - Denoising.
  const glm::mat4& view = resources.cameraManip->getViewMatrix();
  const glm::mat4& proj = resources.cameraManip->getPerspectiveMatrix();

  m_dlss->setResponsivityMask(m_hairResponsivity != 0.0F);
  m_dlss->evaluate(cmd, view, proj);

  // The selection image goes to the full-size one only on a frame that traced it (SelectionWrite):
  // otherwise the full-size image keeps what it last got, which the inner one still holds too. A
  // blit every DLSS frame was 33 MB and two barriers a 4K frame for an image nothing read.
  if(selectionWritten)
  {
    // The selection image the blit reads was written by the path tracer's ray-tracing or compute
    // stage (the barrier after traceRays only made it visible to compute)
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                           VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

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
  }

  // DLSS's compute write of the rendered image, and the blit's, before the compute passes that read
  // them (the hand-off, the tonemapper, the silhouette)
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (selectionWritten ? VK_PIPELINE_STAGE_2_BLIT_BIT : 0),
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT | (selectionWritten ? VK_ACCESS_2_TRANSFER_WRITE_BIT : 0), VK_ACCESS_2_SHADER_READ_BIT);
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
  const bool selectionRead = !resources.selectedRenderNodes.empty();
  m_pushConst.flags |= m_selectionWrite.due(resources.frameCount == 0, useDlss, selectionRead) ? shaderio::ePtWriteSelection : 0;
#if defined(USE_DLSSNR)
  // The per-material NR mask only while DLSS-NR runs: nothing else reads it
  m_pushConst.flags |= (useDlss && m_dlss->isNrActive()) ? shaderio::ePtWriteNrMask : 0;
#endif
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
  m_pushConst.hitSignalSlots = 0;
  m_pushConst.hitSignal      = countHits(cmd, resources);
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
  const VkDeviceAddress prevO2W = resources.gpuView().prevObjectToWorldTable();
  const VkDeviceAddress prevPos = resources.gpuView().prevPositionTable();
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
