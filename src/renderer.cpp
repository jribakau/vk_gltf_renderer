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

// GltfRenderer: the application element, its attach and detach, and the frame.
//
// The rest of the class: renderer_settings.cpp, renderer_environment.cpp, renderer_scene.cpp and
// renderer_selection.cpp.

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>  // glm::rotation
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <optional>
#include <span>
#include <sstream>
#include <utility>
#include <vulkan/vulkan_core.h>

#include "GLFW/glfw3.h"
#undef APIENTRY

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

// Pre-compiled shaders
#include "_autogen/tonemapper.slang.h"
#include "_autogen/env_write_pdf.slang.h"
#include "_autogen/sky_bruneton_precompute.slang.h"
#include "_autogen/sky_clouds_bake.slang.h"
#include "_autogen/sky_clouds_noise.slang.h"

//
#include <nvaftermath/aftermath.hpp>
#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/mipmaps.hpp>

#include <glm/gtc/quaternion.hpp>

#include "renderer.hpp"
#include "gizmo_pivot.hpp"
#include "gpu_crash_trail.hpp"

#ifdef NVP_SUPPORTS_BASISU
// the EXR writer of basis_universal's tinyexr, which env_image_loader.cpp reads environments with
#include <3rdparty/tinyexr.h>
#endif
#include <nvvk/helpers.hpp>

#include <figure/io/timing.hpp>
#include "staging_config.hpp"
#include "tinygltf_utils.hpp"
#include "utils.hpp"
#include "animation_control.hpp"

extern nvutils::ProfilerManager g_profilerManager;  // #PROFILER

namespace {
// Background clear color used when no scene is loaded or to show DLSS render resolution borders
constexpr VkClearColorValue kBackgroundClearColor = {{0.17f, 0.21f, 0.25f, 1.f}};


}  // namespace



//--------------------------------------------------------------------------------------------------
// The onAttach method is called when the application is attached to the renderer
void GltfRenderer::onAttach(nvapp::Application* app)
{
  SCOPED_TIMER("GltfRenderer::onAttach");

  m_app                = app;
  m_device             = app->getDevice();
  m_resources.instance = app->getInstance();
  m_resources.app      = app;

  // VK_EXT_memory_budget is asked for as an optional extension (main.cpp), so a device that has it
  // has it enabled: deviceMemoryBudget() may chain its query then
  {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(app->getPhysicalDevice(), nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(app->getPhysicalDevice(), nullptr, &count, extensions.data());
    m_memoryBudgetExtension = std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& e) {
      return std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    });
  }

  // ===== Settings Handler (ImGui persistent) =====
  // The list lives with the declarations in the constructor; this replays the persisted ones now
  // that the handler exists. Nothing to keep in sync by hand.
  m_settingsHandler.setHandlerName("GltfRenderer");
  m_settings.applyPersistence(m_settingsHandler);
  m_settingsHandler.setLoadFilter([this](const std::string& key) {
    // Skip loading settings that were explicitly set via the command line
    return !(m_parameterParser && m_parameterParser->wasParsed(key));
  });
  m_settingsHandler.addImGuiHandler();

  // ===== Memory Allocation & Buffer Management =====
  m_resources.allocator.init({
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = app->getPhysicalDevice(),
      .device           = app->getDevice(),
      .instance         = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });  // Allocator

  // If there is a leak (nvvkAllocID -> ID)
  // m_resources.allocator.setLeakID(155);


  m_transientCmdPool = nvvk::createTransientCommandPool(m_device, app->getQueue(0).familyIndex);
  NVVK_DBG_NAME(m_transientCmdPool);

  m_loadPipeline.init(m_device, app->getQueue(0).queue, m_transientCmdPool);

  // FrameUploader records copies into the caller's command buffer and retires staging
  // once the frame timeline semaphore (or a load-pipeline wait) has signaled.
  // blockSize is the max size of a single append (circular allocator constraint) and
  // is larger than the library default so scene-create can stage large meshes/textures.
  NVVK_CHECK(m_resources.staging.init({
      .allocator      = &m_resources.allocator,
      .blockSize      = kFrameUploaderBlockSize,  // shared with createSampledImage's row-strip chunking
      .debugName      = "frameUploads",
      .keepBlockCount = kFrameUploaderKeepBlocks,
      .maxBlocks      = kFrameUploaderMaxBlocks,
  }));

  m_resources.commandPool = app->getCommandPool();


  // ===== Texture & Image Resources =====
  m_resources.samplerPool.init(m_device);
  VkSampler linearSampler{};
  NVVK_CHECK(m_resources.samplerPool.acquireSampler(linearSampler));
  NVVK_DBG_NAME(linearSampler);

  // IBL environment map
  m_resources.hdrIbl.init(&m_resources.allocator, &m_resources.samplerPool);
  // Required before HdrIbl::updateFromGpuImage: the file-load path writes the per-texel PDF into
  // the image alpha on the CPU, but a GPU-baked environment is never read back, so the PDF write
  // is a compute pass that has to be created up front. Without it EnvBaker::commit dispatches a
  // null pipeline -- caught only by an assert, so a release build simply crashes.
  m_resources.hdrIbl.initGpuWriter(std::span(env_write_pdf_slang));

  // Application-level memory tracker (frame targets, DLSS, OptiX images)
  m_resources.appMemoryTracker.init(&m_resources.allocator);

  // G-Buffer
  NVVK_CHECK(m_resources.gBuffers.init({.device = m_device,
                                        .alloc  = &m_resources.allocator,
                                        .colorFormats =
                                            {
                                                VK_FORMAT_R8G8B8A8_UNORM,       // Tonemapped (eImgTonemapped)
                                                VK_FORMAT_R32G32B32A32_SFLOAT,  // Rendered image (eImgRendered)
                                                VK_FORMAT_R32_SFLOAT,  // ObjectID for selection/silhouette (eImgSelection), .r = render node ID+1
                                            },
                                        .depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice()),
                                        .debugName   = "GBuffers"}));
  m_resources.linearSampler = linearSampler;
  {
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    NVVK_CHECK(m_resources.gBuffers.update(cmd, {100, 100}));
    m_resources.gBuffers.cmdClear(cmd);
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
    m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  }
  m_resources.tonemappedUi.init(m_resources.gBuffers.getUiImageView(Resources::eImgTonemapped));

  // ===== Rendering Utilities =====

  // Ray picker
  m_rayPicker.init(&m_resources.allocator);

  // Tonemapper
  m_resources.tonemapper.init(&m_resources.allocator, tonemapper_slang);

  // Silhouette renderer
  m_silhouette.init(m_resources);
  // The frame an HDR display is shown
  m_hdrOutput.init(m_resources);
  m_bloom.init(m_resources);
  m_localExposure.init(m_resources);


  // Producer for analytic skies. Its lat-long image feeds the same HdrIbl the HDR file path uses,
  // so the path tracer consumes one environment regardless of sky type.
  // Allocated here but only dispatched once a baked sky type is selected.

  // Bruneton atmosphere LUTs. Built once here from the Earth defaults; they depend only on the
  // atmosphere, never on the sun or the camera, so moving the sun later costs nothing.
  m_skyBruneton.init(m_resources, SkyBruneton::Shaders{.precompute = std::span(sky_bruneton_precompute_slang)});
  // Seed from the settings, not from the Earth defaults init() installs. Otherwise the first
  // refreshBakedEnvironment() finds the parameters changed and rebuilds the whole set immediately
  // -- two full precomputes, every startup, even when nothing was customised.
  m_skyBruneton.setParameters(buildAtmosphereParameters());
  m_skyBruneton.precompute(m_resources);

  // The clouds light themselves through the atmosphere's tables, so after SkyBruneton; their noise
  // is generated here, once.
  m_skyClouds.init(m_resources,
                   SkyClouds::Shaders{.noise = std::span(sky_clouds_noise_slang), .bake = std::span(sky_clouds_bake_slang)},
                   m_skyBruneton.getDescriptorSetLayout(), m_skyBruneton.getDescriptorSet());

  // After SkyBruneton and SkyClouds: EnvBaker takes their descriptor-set layouts to build a pipeline
  // layout that covers every producer.
  m_envBaker.init(m_resources, m_skyBruneton.getDescriptorSetLayout(), m_skyBruneton.getDescriptorSet(), m_skyClouds);
  m_envBaker.setResolution(m_resources, kEnvBakeSize);

  // Publish the same set to the two renderers, which read the tables directly for aerial
  // perspective. Here rather than in each of them: they build their pipeline layouts from
  // Resources, and neither needs to know which class owns the LUTs.
  m_resources.skyLutDescriptorSetLayout   = m_skyBruneton.getDescriptorSetLayout();
  m_resources.skyLutDescriptorSet         = m_skyBruneton.getDescriptorSet();
  m_resources.skyCloudDescriptorSetLayout = m_skyClouds.getDescriptorSetLayout();
  m_resources.skyCloudDescriptorSet       = m_skyClouds.getDescriptorSet();

  // ===== Profiling & Performance =====
  {
    SCOPED_TIMER("Profiler");
    m_profilerTimeline = g_profilerManager.createTimeline({.name = "Primary Timeline"});
    m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(),
                            int32_t(m_app->getQueue(0).familyIndex), false);
  }


  // ===== Shader Compilation =====
  {
    SCOPED_TIMER("Shader Slang");
    using namespace slang;
    m_resources.slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_resources.slangCompiler.defaultTarget();
    m_resources.slangCompiler.defaultOptions();

    // Specific options for this sample. Standard debug info (NonSemantic.Shader.DebugInfo, for
    // Nsight) only on NVIDIA: AMD's Vulkan driver crashes compiling SPIR-V that carries it, while
    // minimal debug info (OpLine) is accepted everywhere. Embedded shaders follow
    // SHADER_DEBUG_LEVEL in CMake instead.
    VkPhysicalDeviceProperties gpuProps{};
    vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &gpuProps);
    constexpr uint32_t kNvidiaVendorId = 0x10DE;
    int debugInfoLevel = (gpuProps.vendorID == kNvidiaVendorId) ? SLANG_DEBUG_INFO_LEVEL_STANDARD : SLANG_DEBUG_INFO_LEVEL_MINIMAL;
    int optimization   = SLANG_OPTIMIZATION_LEVEL_NONE;
    // FV_SLANG_DEBUG / FV_SLANG_OPT (0-3): other levels, to time the driver's build against them
    if(const char* v = std::getenv("FV_SLANG_DEBUG"); v != nullptr && v[0] >= '0' && v[0] <= '3')
      debugInfoLevel = v[0] - '0';
    if(const char* v = std::getenv("FV_SLANG_OPT"); v != nullptr && v[0] >= '0' && v[0] <= '3')
      optimization = v[0] - '0';
    LOGI("Slang run-time compile: debug info level %d, optimization level %d\n", debugInfoLevel, optimization);
    m_resources.slangCompiler.addOption({CompilerOptionName::DebugInformation, {CompilerOptionValueKind::Int, debugInfoLevel}});
    m_resources.slangCompiler.addOption({CompilerOptionName::Optimization, {CompilerOptionValueKind::Int, optimization}});

    // Enable specific capabilities for better performance and features.
    m_resources.slangCompiler.addCapability("spvShaderInvocationReorderEXT");  // SER via VK_EXT_ray_tracing_invocation_reorder
    m_resources.slangCompiler.addCapability("spvInt64Atomics");                // # 64-bit atomic operations
    m_resources.slangCompiler.addCapability("spvShaderClockKHR");              // # Shader clock for profiling
    m_resources.slangCompiler.addCapability("spvRayTracingMotionBlurNV");  // # Motion blur for ray tracing
    m_resources.slangCompiler.addCapability("spvRayQueryKHR");             // # Ray query operations
    m_resources.slangCompiler.addCapability("spvGroupNonUniformBallot");  // # Ballot operations for subgroup functionality
    m_resources.slangCompiler.addCapability("spvGroupNonUniformArithmetic");  // # Arithmetic operations across subgroups
    m_resources.slangCompiler.addCapability("nonuniformqualifier");  // # NonUniformResourceIndex on the bindless arrays (also in CMake)

#if defined(AFTERMATH_AVAILABLE)
    // This aftermath callback is used to report the shader hash (Spirv) to the Aftermath library.
    m_resources.slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
      std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
      AftermathCrashTracker::getInstance().addShaderBinary(data);
    });
#endif
  }

  // ===== Renderer Initialization =====

  // Create resources
  createDescriptorSets();
  createHDR("");  // Dummy HDR
  createResourceBuffers();

  // Initialize the renderers
  m_pathTracer.setCacheDir(m_cacheDir);
  m_pathTracer.onAttach(m_resources, &m_profilerGpuTimer);
  m_pathTracer.setProfilerTimeline(m_profilerTimeline);
  m_pathTracer.setBusyState(&m_busy);

  m_pathTracer.createPipeline(m_resources);
  // the RTX pipeline the first path-traced frame needs, built on a thread while the app starts
  m_pathTracer.startPipelineWarm(m_resources);

  // ===== Visual Helpers (Grid + Transform Gizmo) =====
  {
    VkFormat depthFormat = m_resources.gBuffers.getDepthFormat();
    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Matches eImgTonemapped

    VisualHelpers::Resources helperRes{
        .app           = m_app,
        .alloc         = &m_resources.allocator,
        .uploader      = &m_resources.staging,
        .device        = m_device,
        .sampler       = linearSampler,
        .slangCompiler = &m_resources.slangCompiler,
        .colorFormat   = colorFormat,
        .depthFormat   = depthFormat,
    };
    m_visualHelpers.init(helperRes);

    // the gizmo drags its pivot; the host gets the entity's own transform
    const auto sendGizmo = [this](bool done) {
      if(m_gizmoEntity.isNone() || !m_gizmoHooks.edit)
        return;
      const glm::quat r = glm::quat(glm::radians(m_gizmoRotation));
      m_gizmoHooks.edit(m_gizmoEntity, translationFromPivot(m_gizmoPosition, r, m_gizmoScale, m_gizmoPivot), r, m_gizmoScale, done);
    };
    m_visualHelpers.transform.setOnTransformChange([sendGizmo]() { sendGizmo(false); });
    m_visualHelpers.transform.setOnTransformEnd([sendGizmo]() { sendGizmo(true); });
  }

}


//--------------------------------------------------------------------------------------------------
// Detach the renderers and destroy the resources
void GltfRenderer::onDetach()
{
  // SYNC NOTE: Full device wait during shutdown is the standard Vulkan teardown pattern.
  vkDeviceWaitIdle(m_device);
  m_visualHelpers.deinit();
  m_pathTracer.onDetach(m_resources);
  destroyResources();
}

//--------------------------------------------------------------------------------------------------
// Resize the G-Buffer and the renderers
void GltfRenderer::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  NVVK_CHECK(m_resources.gBuffers.update(cmd, size));
  m_resources.gBuffers.cmdClear(cmd);
  m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  m_pathTracer.onResize(cmd, size, m_resources);
  m_hdrOutput.onResize(cmd, size);
  m_bloom.onResize(cmd, size);
  m_localExposure.onResize(cmd, size);
  m_resources.tonemappedUi.update(m_resources.gBuffers.getUiImageView(Resources::eImgTonemapped));

  // Resize visual helpers (depth buffer + scene depth descriptor set)
  m_visualHelpers.onResize(cmd, size, m_resources.gBuffers.getDepthImage(), m_resources.gBuffers.getDepthImageView(),
                           m_resources.linearSampler);

  // Camera (was handled by ElementCamera, now owned by the renderer)
  m_cameraManip->setWindowSize({size.width, size.height});
  m_cameraManip->adjustOrthographicAspect();

  resetFrame();  // Reset frame to restart the rendering
}

//--------------------------------------------------------------------------------------------------
bool GltfRenderer::isHeadlessMode() const
{
  return m_app != nullptr && m_app->isHeadless();
}

bool GltfRenderer::isAutomatedRun() const
{
  return isHeadlessMode();
}

void GltfRenderer::alignMaxSamplesForHeadless(uint32_t headlessFrames)
{
  // An explicit --ptMaxSamples is the point of the run (e.g. --frames 2000 --ptMaxSamples 1024)
  if(m_parameterParser && m_parameterParser->wasParsed("ptMaxSamples"))
    return;
  const int samplesPerFrame =
      m_pathTracer.m_adaptiveSampling ? PathTracer::MAX_ADAPTIVE_SAMPLES_PER_PIXEL : m_pathTracer.m_pushConst.numSamples;
  const int64_t needed = std::min<int64_t>(int64_t(headlessFrames) * std::max(samplesPerFrame, 1), INT_MAX);
  if(m_resources.settings.maxSamples < needed)
  {
    LOGW("ptMaxSamples (%d) is less than headless --frames (%u) x %d spp per frame; setting it to %lld\n",
         m_resources.settings.maxSamples, headlessFrames, samplesPerFrame, static_cast<long long>(needed));
    m_resources.settings.maxSamples = static_cast<int>(needed);
  }
}

void GltfRenderer::applyHeadlessDefaults()
{
  if(!(m_parameterParser && m_parameterParser->wasParsed("animPlay")))
    m_animPlay = false;
  m_resources.animationControl.play = m_animPlay;
  // Nobody watches a headless frame: Auto SPP aims at throughput rather than a frame rate
  if(!(m_parameterParser && m_parameterParser->wasParsed("ptPerformanceTarget")))
    m_pathTracer.m_performanceTarget = PathTracer::PerformanceTarget::eThroughput;
}

//--------------------------------------------------------------------------------------------------
// End of a headless run: the cost of the AI denoisers that ran. OIDN is GPU time inside the frame;
// OptiX runs outside it (queue drain, copies, CUDA), so its breakdown is host wall time.
void GltfRenderer::logDenoiserTimings() const
{
#if defined(USE_OIDN_DENOISER)
  if(const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser(); oidn && oidn->hasValidDenoisedOutput() && oidn->getGpuTimeMs() > 0.0)
    LOGI("DENOISER_TIMING oidn gpu_ms=%.3f\n", oidn->getGpuTimeMs());
#endif
#if defined(USE_OPTIX_DENOISER)
  if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser(); optix && optix->hasValidDenoisedOutput())
  {
    const OptiXDenoiser::Timings t = optix->getTimings();
    LOGI("DENOISER_TIMING optix total_ms=%.3f drain_ms=%.3f copy_in_ms=%.3f invoke_ms=%.3f invoke_gpu_ms=%.3f copy_out_ms=%.3f\n",
         t.totalMs, t.drainMs, t.copyInMs, t.invokeMs, t.invokeGpuMs, t.copyOutMs);
  }
#endif
}

std::filesystem::path GltfRenderer::headlessOutputPathOrDefault() const
{
  return m_resources.headlessOutputPath.empty() ? nvutils::getExecutablePath().replace_extension(".jpg") : m_resources.headlessOutputPath;
}

//--------------------------------------------------------------------------------------------------
// --outputSamples: the samples per pixel of the image the tonemapper shows. A denoised image counts
// the samples it was denoised from (OptiX runs one frame behind, OIDN in the same frame). Returns
// -1 for an image that is not a path-traced accumulation (the DLSS hand-off's blend included).
int GltfRenderer::displayedImageSamples(DisplayBuffer& source) const
{
  source = m_resources.settings.displayBuffer;
  if(m_pathTracer.handoffImage(m_resources))
    return -1;
#if defined(USE_OPTIX_DENOISER)
  if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser();
     source == DisplayBuffer::eOptixDenoised && optix && optix->hasValidDenoisedOutput())
    return optix->denoisedSamples();
#endif
#if defined(USE_OIDN_DENOISER)
  if(const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser();
     source == DisplayBuffer::eOidnDenoised && oidn && oidn->hasValidDenoisedOutput())
    return oidn->denoisedSamples();
#endif
  source = DisplayBuffer::eRendered;  // what runTonemapPass falls back to
  return m_resources.accumulatedSamples;
}

//--------------------------------------------------------------------------------------------------
// --outputSamples: parse the counts when the option changed. They are also stops for the path tracer
// (Resources::sampleStops): a frame ends exactly on each, so none is stepped over, whatever the samples
// per frame. Called before the frame renders.
void GltfRenderer::refreshSampleTargets()
{
  SampleCapture& capture = m_sampleCapture;
  if(capture.parsedFrom == m_outputSamples)
    return;
  capture          = {.parsedFrom = m_outputSamples};
  std::string list = m_outputSamples;
  std::replace(list.begin(), list.end(), ',', ' ');
  std::istringstream stream(list);
  for(int value; stream >> value;)
    if(value > 0)
      capture.targets.push_back(value);
  std::sort(capture.targets.begin(), capture.targets.end());
  capture.targets.erase(std::unique(capture.targets.begin(), capture.targets.end()), capture.targets.end());
  m_resources.sampleStops = capture.targets;
}

//--------------------------------------------------------------------------------------------------
// --outputSamples: one run saves a whole convergence series instead of one run per sample count.
// When the accumulation restarts, the counts from the current one on are captured again, so the saved
// files always come from the last accumulation. When another image is shown (e.g. the first denoised
// one), the current count is captured again from it, and the counts no image reached yet still are.
// A count skipped over (several samples per frame, a denoiser interval) is saved at the first image
// past it, named by its actual count.
void GltfRenderer::updateSampleCaptures()
{
  SampleCapture& capture = m_sampleCapture;
  refreshSampleTargets();
  if(capture.targets.empty())
    return;

  // A .hdr capture saves the rendered image, never a denoiser's (saveViewportImage): name it after
  // that image's samples, not after the denoised one on screen, which OptiX keeps a frame behind.
  const bool    linear = nvutils::extensionMatches(headlessOutputPathOrDefault(), ".hdr")
                      || nvutils::extensionMatches(headlessOutputPathOrDefault(), ".pfm");
  DisplayBuffer source{};
  const int     samples = linear ? m_resources.accumulatedSamples : displayedImageSamples(source);
  if(linear)
    source = DisplayBuffer::eRendered;
  if(samples <= 0)
    return;
  const size_t current = std::lower_bound(capture.targets.begin(), capture.targets.end(), samples) - capture.targets.begin();
  if(samples < capture.lastSamples)  // the accumulation restarted
    capture.next = current;
  else if(int(source) != capture.lastSource)  // another image: keep the counts not saved yet
    capture.next = std::min(capture.next, current);
  capture.lastSamples = samples;
  capture.lastSource  = int(source);

  if(capture.next < capture.targets.size() && capture.targets[capture.next] <= samples)
  {
    while(capture.next < capture.targets.size() && capture.targets[capture.next] <= samples)
      capture.next++;
    const std::filesystem::path output = headlessOutputPathOrDefault();
    capture.pending                    = output.parent_path()
                      / (output.stem().string() + "_" + std::to_string(samples) + "spp" + output.extension().string());
  }
}

//--------------------------------------------------------------------------------------------------
// The frame is submitted: save the image updateSampleCaptures() asked for. saveImageToFile records
// its copy on the same queue and waits for it, so it sees this frame's tonemapped image.
void GltfRenderer::onPostRender()
{
  if(!m_app)
    return;
  // the frame just submitted: had the GPU already finished the one before it, and waited on the CPU?
  if(fv::timing::enabled())
  {
    const nvvk::SemaphoreInfo frame = m_app->getFrameSignalSemaphore();
    uint64_t                  done  = 0;
    vkGetSemaphoreCounterValue(m_device, frame.semaphore, &done);
    m_lastFrameCpu.gpuIdleAtSubmit = m_lastFrameSignal != 0 && done >= m_lastFrameSignal;
    m_lastFrameSignal              = frame.value;
  }
  // A still's frames run one at a time. Queued, they gain a sample nothing can see, and an edit arriving
  // now would wait for all of them before its first moving frame, freezing and then jumping the image
  // at the start of every action. (A moving view's frames queue, for their rate; a window presents after this.)
  if(m_streamHost && !m_pathTracer.dlssFrame())
  {
    const nvvk::SemaphoreInfo frame = m_app->getFrameSignalSemaphore();
    const VkSemaphoreWaitInfo wait{.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .semaphoreCount = 1,
                                   .pSemaphores    = &frame.semaphore,
                                   .pValues        = &frame.value};
    NVVK_CHECK(vkWaitSemaphores(m_device, &wait, std::numeric_limits<uint64_t>::max()));
  }
  if(!m_screenshotPending.empty())
  {
    saveViewportImage(m_screenshotPending);
    m_screenshotPending.clear();
  }
  if(m_sampleCapture.pending.empty())
    return;
  saveViewportImage(m_sampleCapture.pending);
  m_sampleCapture.pending.clear();
}

// One rule for every way the viewport is saved (--output, --outputSamples, the screenshot
// parameter): a .hdr file gets the linear rendered image, so it can be measured (a .pfm the same, every
// bit of its floats, for a bit-exact comparison); an .exr the
// linear image the screen is made from (saveExr); any other extension gets what the screen shows,
// opaque as the screen and the EXR are: the path tracer's alpha is whether the camera ray hit
// something, and a PNG saved with it had the sky, the floor's fade and a cloud's edge see-through.
void GltfRenderer::saveViewportImage(const std::filesystem::path& filename)
{
  if(nvutils::extensionMatches(filename, ".exr"))
  {
    saveExr(filename);
    return;
  }
  const Resources::ImageType imageType =
      nvutils::extensionMatches(filename, ".hdr") || nvutils::extensionMatches(filename, ".pfm") ? Resources::eImgRendered :
                                                                                                   Resources::eImgTonemapped;
  m_app->saveImageToFile(m_resources.gBuffers.getColorImage(imageType), m_resources.gBuffers.getSize(), filename, 100,
                         VK_IMAGE_LAYOUT_GENERAL, /*opaque=*/imageType == Resources::eImgTonemapped);
}

// An EXR of what the frame was made from, for compositing and grading elsewhere: the denoised image the
// tonemapper took, with its exposure and white balance, linear Rec. 709 half floats, unclamped, before the
// tone curve and the grading. From the last frame drawn, its images as it left them: after a frame's
// submit (onPostRender) or at a headless run's end (--output).
void GltfRenderer::saveExr(const std::filesystem::path& filename)
{
  const TonemapInput& in = m_tonemapInput;
  if(in.size.width == 0 || in.size.height == 0)
  {
    LOGE("Saving %s: no frame was drawn to export\n", nvutils::utf8FromPath(filename).c_str());
    return;
  }
#ifdef NVP_SUPPORTS_BASISU
  // The frame's images still hold what it drew: the export pass from them, then its read back
  VkImage         linear{};
  VkDeviceMemory  linearMemory{};
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  m_hdrOutput.cmdExportLinear(cmd, m_resources, in.size, in.data, in.image);
  const VkExtent2D size   = m_hdrOutput.exportSize();
  VkResult         result = nvvk::imageToLinear(cmd, m_device, m_app->getPhysicalDevice(), m_hdrOutput.exportImage(), size,
                                                linear, linearMemory, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_GENERAL);
  m_app->submitAndWaitTempCmdBuffer(cmd);
  std::vector<float> pixels;
  if(result == VK_SUCCESS)
  {
    VkImageSubresource  subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout layout{};
    vkGetImageSubresourceLayout(m_device, linear, &subresource, &layout);
    const char* data = nullptr;
    result           = vkMapMemory(m_device, linearMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
    if(result == VK_SUCCESS)
    {
      pixels.resize(size_t(size.width) * size.height * 4);
      data += layout.offset;
      for(uint32_t y = 0; y < size.height; y++, data += layout.rowPitch)
        std::memcpy(pixels.data() + size_t(y) * size.width * 4, data, size_t(size.width) * 4 * sizeof(float));
      vkUnmapMemory(m_device, linearMemory);
    }
  }
  vkDeviceWaitIdle(m_device);  // nothing reads the export any more
  vkDestroyImage(m_device, linear, nullptr);
  vkFreeMemory(m_device, linearMemory, nullptr);
  m_hdrOutput.releaseExport();
  if(result != VK_SUCCESS)
  {
    LOGE("Saving %s: reading the image back failed (VkResult %d)\n", nvutils::utf8FromPath(filename).c_str(), int(result));
    return;
  }
  const char* err = nullptr;
  if(SaveEXR(pixels.data(), int(size.width), int(size.height), 4, /*save_as_fp16=*/1, nvutils::utf8FromPath(filename).c_str(), &err) != TINYEXR_SUCCESS)
  {
    LOGE("Saving %s: %s\n", nvutils::utf8FromPath(filename).c_str(), err ? err : "the EXR writer failed");
    FreeEXRErrorMessage(err);
    return;
  }
  LOGI("Saved %s (%ux%u, linear half floats)\n", nvutils::utf8FromPath(filename).c_str(), size.width, size.height);
#else
  m_hdrOutput.releaseExport();
  LOGE("Saving %s: this build has no EXR writer (basis_universal's tinyexr)\n", nvutils::utf8FromPath(filename).c_str());
#endif
}

void GltfRenderer::saveHeadlessOutputImage()
{
  std::string                 outputPath = headlessOutputPathOrDefault().string();
  const std::filesystem::path parentDir  = std::filesystem::path(outputPath).parent_path();
  if(!parentDir.empty() && !std::filesystem::exists(parentDir))
  {
    LOGW("GltfRenderer::saveHeadlessOutputImage(): output directory does not exist for path %s\n", outputPath.c_str());
  }
  if(!m_app)
  {
    LOGE("GltfRenderer::saveHeadlessOutputImage(): application is not set; cannot save image to %s\n", outputPath.c_str());
    return;
  }
  saveViewportImage(outputPath);
}

void GltfRenderer::onUIRender()
{
  // Run any post-restore hooks now that Application::run() has reloaded the ini (which happens
  // between onAttach and the first frame, in both windowed and headless paths). ImGui writes
  // restored values straight into storage without firing the parameter's callbackSuccess, so
  // derived state (e.g. Resources::sunDirection from the restored sunAzimuth/Elevation) would
  // otherwise stay stale until the user touched a UI slider.
  if(m_pendingRestoreCallbacks)
  {
    m_pendingRestoreCallbacks = false;
    m_settings.runPostRestoreHooks();
  }

  applyPendingSkyPresets();          // frame-top: start-up presets
  applyPendingEnvironmentRefresh();  // frame-top: re-bake / rebind the environment after a sky change (see method)

  // The viewport's per-frame camera step: camera transitions and camera.frame need it.
  m_cameraManip->updateAnim();
}

//--------------------------------------------------------------------------------------------------
// Render the scene
// What a pixel of the frame holds in GPU memory, for refuseFrameSize and for the room DLSS has to find
// when it comes on: the driver's usage between 1080p and 4K (scripts/memory-limit-check.py, RTX 4080,
// 2026-11-01, while a rasterizer still kept its targets beside the path tracer's, so an upper bound now).
// DLSS is its guides and the upscaler's history, which the driver holds.
constexpr uint32_t kPathTracerBytesPerPixel = 209;
constexpr uint32_t kDlssBytesPerPixel       = 143;
constexpr uint32_t kMostBytesPerPixel       = kPathTracerBytesPerPixel + kDlssBytesPerPixel;

GltfRenderer::MemoryBudget GltfRenderer::deviceMemoryBudget() const
{
  MemoryBudget b;
  if(!m_memoryBudgetExtension || !m_app)
    return b;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &budget};
  vkGetPhysicalDeviceMemoryProperties2(m_app->getPhysicalDevice(), &props);
  for(uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; i++)
    if(props.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
    {
      b.budget += budget.heapBudget[i];
      b.usage += budget.heapUsage[i];
    }
  b.known = true;
  const AllocatedMemory allocated = allocatedMemory();
  b.allocatorBlocks               = allocated.blocks;
  b.allocatorAllocations          = allocated.allocations;
  return b;
}

std::string GltfRenderer::frameProfileJson() const
{
  std::string out = "{\"sections\":[";
  if(m_profilerTimeline)
  {
    nvutils::ProfilerTimeline::Snapshot snapshot;
    m_profilerTimeline->getFrameSnapshot(snapshot);
    bool first = true;
    for(size_t i = 0; i < snapshot.timerInfos.size(); i++)
    {
      const auto& info = snapshot.timerInfos[i];
      if(info.numAveraged == 0)
        continue;
      std::string name;
      for(char c : snapshot.timerNames[i])
        if(c == '"' || c == '\\')
          name += {'\\', c};
        else if(uint8_t(c) >= 0x20)
          name += c;
      char buf[96];
      // the profiler keeps microseconds
      std::snprintf(buf, sizeof(buf), "\",\"level\":%u,\"cpu\":%.3f,\"gpu\":%.3f}", info.level, info.cpu.average * 1e-3,
                    info.gpu.average * 1e-3);
      out += (first ? "{\"name\":\"" : ",{\"name\":\"") + name + buf;
      first = false;
    }
  }
  out += "]";
  const MemoryBudget b = deviceMemoryBudget();
  if(b.known)
  {
    char buf[96];
    std::snprintf(buf, sizeof(buf), ",\"gpuMemoryMB\":%.0f,\"gpuBudgetMB\":%.0f,\"gpuAllocatedMB\":%.0f", double(b.usage) / (1 << 20),
                  double(b.budget) / (1 << 20), double(allocatedMemory().allocations) / (1 << 20));
    out += buf;
  }
  return out + "}";
}

double GltfRenderer::frameGpuMs() const
{
  nvutils::ProfilerTimeline::TimerInfo info;
  std::string                          api;
  // the profiler keeps microseconds
  return m_profilerTimeline && m_profilerTimeline->getFrameTimerInfo("Frame", info, api) ? info.gpu.last * 1e-3 : 0.0;
}

std::string GltfRenderer::refuseFrameSize(VkExtent2D size, double extraBytesPerPixel) const
{
  const MemoryBudget b = deviceMemoryBudget();
  if(!b.known)
    return {};
  const uint64_t packed = m_frameSizePacked.load();
  const int64_t  pixelsNow  = int64_t(packed >> 32) * int64_t(packed & 0xffffffffu);
  const int64_t  pixelsMore = int64_t(size.width) * size.height - pixelsNow;
  if(pixelsMore <= 0)
    return {};
  const double need = double(pixelsMore) * (frameBytesPerPixel() + extraBytesPerPixel);
  const double free = double(b.budget) - double(b.usage);
  // a tenth of what is free stays for the scene's own growth (a BLAS refit, a texture)
  if(need <= free * 0.9)
    return {};
  return fmt::format("a {}x{} frame needs about {:.1f} GB more of GPU memory and {:.1f} GB is free ({:.1f} of {:.1f} GB in use)",
                     size.width, size.height, need / 1e9, std::max(free, 0.0) / 1e9, b.usage / 1e9, b.budget / 1e9);
}

// What the allocator holds on the device and has handed out: the driver's usage counts whole blocks,
// so what was freed inside one still reads as used, though a new image can go there
GltfRenderer::AllocatedMemory GltfRenderer::allocatedMemory() const
{
  AllocatedMemory out;
  if(!m_app)
    return out;
  VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
  vmaGetHeapBudgets(m_resources.allocator, budgets);
  VkPhysicalDeviceMemoryProperties props{};
  vkGetPhysicalDeviceMemoryProperties(m_app->getPhysicalDevice(), &props);
  for(uint32_t i = 0; i < props.memoryHeapCount; i++)
    if(props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
    {
      out.blocks += budgets[i].statistics.blockBytes;
      out.allocations += budgets[i].statistics.allocationBytes;
    }
  return out;
}

uint64_t GltfRenderer::sceneBudget() const
{
  const uint64_t budget = deviceMemoryBudget().budget;
  const uint32_t limit  = m_resources.settings.sceneBudgetMB;
  return limit > 0 ? std::min<uint64_t>(budget, uint64_t(limit) << 20) : budget;
}

// A tenth of the budget stays free for what grows unasked (a BLAS refit, a denoiser's targets), and so
// does what DLSS would add to the frame, for it to find when it comes on.
std::optional<int64_t> GltfRenderer::sceneHeadroom() const
{
  const MemoryBudget b = deviceMemoryBudget();
  if(!b.known)
    return std::nullopt;
  const uint64_t packed = m_frameSizePacked.load();
  const uint64_t pixels = (packed >> 32) * (packed & 0xffffffffu);
  const uint32_t perPixel = frameBytesPerPixel();
  const uint64_t frameMore = perPixel < kMostBytesPerPixel ? pixels * (kMostBytesPerPixel - perPixel) : 0;
  const uint64_t budget = sceneBudget();
  const AllocatedMemory allocated = allocatedMemory();
  uint64_t              used      = b.usage - std::min<uint64_t>(b.usage, allocated.blocks - allocated.allocations);
  // memoryFillMB stands in for what the scene doesn't know of, so the scene comes in on top of it and spills
  used -= std::min<uint64_t>(used, uint64_t(m_memoryFill.size()) << 28);
  return int64_t(budget) - int64_t(budget / 10 + frameMore + used);
}

// A load or wear that can't fit even with every texture lowered would fall to system memory: on an RTX 4080
// about six times the frame time for what the view looks at (docs/scene-memory-plan.md). What can come in
// smaller does; the rest comes all the same, said so.
std::string GltfRenderer::geometryShortfall(uint64_t bytes, bool keepTextureDetail)
{
  const std::optional<int64_t> headroom = sceneHeadroom();
  if(!headroom)
    return {};
  // a 4K map at 2K hardly shows at a portrait's framing, and below that a skin's pores go before a
  // hair's strands do
  constexpr uint32_t kDetailExtent = 2048;
  const uint32_t     least         = keepTextureDetail ? kDetailExtent : rs::kLeastImageExtent;
  const int64_t      lowerable     = m_resources.retainedGpu ? int64_t(m_resources.retainedGpu->lowerableImageBytes(least)) : 0;
  const int64_t room      = *headroom - int64_t(m_pendingGeometry) + lowerable;
  if(int64_t(bytes) <= room)
    return {};
  const MemoryBudget b = deviceMemoryBudget();
  const auto size = [](double n) { return n < 1e9 ? fmt::format("{:.0f} MB", n / 1e6) : fmt::format("{:.1f} GB", n / 1e9); };
  return fmt::format("it needs about {} of GPU memory and {} can be freed ({:.1f} of {:.1f} GB in use), so part of the scene "
                     "runs from system memory and rendering is slower. Removing something or lowering Subdivision helps.",
                     size(double(bytes)), size(double(std::max<int64_t>(room, 0))), double(b.usage) / 1e9, double(b.budget) / 1e9);
}

uint64_t GltfRenderer::blasBytes(uint64_t triangles, uint64_t vertices, bool refittable) const
{
  if(triangles == 0 || !m_app)
    return 0;
  VkAccelerationStructureGeometryKHR geometry{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                              .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR};
  geometry.geometry.triangles = {.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                                 .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                                 .vertexStride = sizeof(glm::vec3),
                                 .maxVertex    = uint32_t(std::min<uint64_t>(vertices ? vertices - 1 : 0, UINT32_MAX)),
                                 .indexType    = VK_INDEX_TYPE_UINT32};
  const VkAccelerationStructureBuildGeometryInfoKHR info{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = VkBuildAccelerationStructureFlagsKHR(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                    | (refittable ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR :
                                                                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)),
      .mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = 1,
      .pGeometries   = &geometry};
  const uint32_t                           count = uint32_t(std::min<uint64_t>(triangles, UINT32_MAX));
  VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(m_app->getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &count, &sizes);
  return refittable ? sizes.accelerationStructureSize : uint64_t(double(sizes.accelerationStructureSize) * rs::kCompactedBlasShare);
}

rs::MemoryLedger GltfRenderer::memoryLedger() const
{
  rs::MemoryLedger ledger = m_resources.retainedGpu ? m_resources.retainedGpu->residency().ledger() : rs::MemoryLedger{};
  const nvvk::BufferCircularAllocator::Report staging = m_resources.staging.getStagingReport();
  ledger.stagingHeld                                  = staging.allocatedSize;
  ledger.stagingLive                                  = staging.requestedSize;
  return ledger;
}

//--------------------------------------------------------------------------------------------------
// The wipe against a kept frame (setWipe): the tone-mapped image as this frame left it is kept when
// asked, and the kept one copied over the live image left of the divider. Copies only: the tone
// mapper, DLSS-NR and the HDR output are as they were. A kept frame of another size (a resize) goes.
void GltfRenderer::cmdWipe(VkCommandBuffer cmd)
{
  if(!m_wipeKeep && m_wipeAt < 0.0F)
    return;
  const VkExtent2D size  = m_resources.gBuffers.getSize();
  const VkImage    shown = m_resources.gBuffers.getColorImage(Resources::eImgTonemapped);
  if(m_wipeFrame.image && (m_wipeExtent.width != size.width || m_wipeExtent.height != size.height))
    dropWipeFrame();
  const uint32_t width = m_wipeAt >= 0.0F ? std::min(size.width, uint32_t(std::lround(m_wipeAt * float(size.width)))) : 0u;
  if(!m_wipeKeep && (!m_wipeFrame.image || width == 0))
    return;
  // after the tone mapper (compute) and DLSS-NR's blit wrote the image, and the HDR output read it
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
  const VkImageSubresourceLayers color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  if(m_wipeKeep)
  {
    m_wipeKeep = false;
    if(!m_wipeFrame.image)
    {
      const VkImageCreateInfo info{.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                   .imageType     = VK_IMAGE_TYPE_2D,
                                   .format        = m_resources.gBuffers.getColorFormat(Resources::eImgTonemapped),
                                   .extent        = {size.width, size.height, 1},
                                   .mipLevels     = 1,
                                   .arrayLayers   = 1,
                                   .samples       = VK_SAMPLE_COUNT_1_BIT,
                                   .tiling        = VK_IMAGE_TILING_OPTIMAL,
                                   .usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
      if(m_resources.allocator.createImage(m_wipeFrame, info) != VK_SUCCESS)
      {
        m_wipeFrame = {};
        return;
      }
      NVVK_DBG_NAME(m_wipeFrame.image);
      m_wipeExtent = size;
      nvvk::cmdImageMemoryBarrier(cmd, {m_wipeFrame.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
    }
    const VkImageCopy whole{.srcSubresource = color, .dstSubresource = color, .extent = {size.width, size.height, 1}};
    vkCmdCopyImage(cmd, shown, VK_IMAGE_LAYOUT_GENERAL, m_wipeFrame.image, VK_IMAGE_LAYOUT_GENERAL, 1, &whole);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
  }
  if(width > 0 && m_wipeFrame.image)
  {
    const VkImageCopy left{.srcSubresource = color, .dstSubresource = color, .extent = {width, size.height, 1}};
    vkCmdCopyImage(cmd, m_wipeFrame.image, VK_IMAGE_LAYOUT_GENERAL, shown, VK_IMAGE_LAYOUT_GENERAL, 1, &left);
  }
  // the overlays draw into it next, the HDR output composes from it, the stream reads it
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

void GltfRenderer::onRender(VkCommandBuffer cmd)
{
  const bool timed = fv::timing::enabled();
  const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  renderFrame(cmd);
  // Whatever the frame did, or returned early from: the HDR image gets its overlays, or the SDR image whole
  m_hdrOutput.cmdFinish(cmd, m_resources);
  if(timed)
    m_lastFrameCpu.recordMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void GltfRenderer::renderFrame(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  m_profilerTimeline->frameAdvance();
  gpu_trail::frameStart(cmd, ++m_trailFrames);
  // for refuseFrameSize on the stream's thread
  m_frameBytesPerPixel.store(kPathTracerBytesPerPixel + (m_pathTracer.isDlssEnabled() ? kDlssBytesPerPixel : 0)
                              + (m_hdrOutput.active() ? HdrOutput::kBytesPerPixel : 0) + (m_bloom.active() ? Bloom::kBytesPerPixel : 0)
                              + (m_localExposure.active() ? LocalExposure::kBytesPerPixel : 0));
  {
    const VkExtent2D s = renderSize();
    m_frameSizePacked.store(uint64_t(s.width) << 32 | s.height);
  }
  // Before anything reads the sun: if the scene marks a light as the sky's sun, that light owns
  // its direction.
  m_skySun.syncFromMarkedLight();
  // Don't do anything if the busy window is open
  if(m_busy.isBusy())
  {
    return;
  }

  // Consume the done signal from the busy state.
  if(m_busy.isDone())
  {
    m_busy.consumeDone();
    resetFrame();
  }

  // Loading pipeline: submit queued work, poll completion, run callbacks
  if(isAutomatedRun())
  {
    // The GPU work a load or rebuild enqueued (uploads, BLAS and TLAS builds), which the compose
    // that enqueued it never waits for: timed here, as its own depth-0 scope.
    std::optional<fv::timing::Scope> wait;
    if(fv::timing::enabled() && !m_loadPipeline.idle())
      wait.emplace("drain load pipeline");
    m_loadPipeline.drain();
  }
  else if(m_loadPipeline.poll())
    return;  // Still loading

  // Preview re-bake for a control that is being dragged. Unlike the commit path this records into
  // the frame's own command buffer and waits on nothing, so it belongs here rather than in
  // onUIRender: it is per-frame work, not a frame-boundary rebuild.
  if(m_envPreviewPending)
  {
    m_envPreviewPending = false;
    if(isBakedEnvironment(m_resources.settings.envSystem))
    {
      // Nothing waits here, so order explicitly: the LUTs, the lat-long image and the sun-disk
      // buffer are still being read by frames in flight, and are read again by this frame's
      // ray-tracing and graphics passes after the rewrite.
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

      // Most of the atmosphere controls -- Rayleigh, Mie, ozone, the planet radii -- change
      // nothing except the LUTs, so without a preview rebuild they look like dead sliders until
      // the mouse comes up. Preview quality is what makes that affordable: a sixteenth of the
      // directions and two bounces instead of four (see SkyBruneton::Quality), and recorded into this
      // frame rather than submitted and waited on. The commit that follows restores full quality.
      if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky)
      {
        const shaderio::SkyAtmosphereParameters wanted = buildAtmosphereParameters();
        if(!m_skyBruneton.lutsSatisfy(wanted, SkyBruneton::kQualityPreview))
        {
          m_skyBruneton.setParameters(wanted);
          m_skyBruneton.recordPrecompute(cmd, SkyBruneton::kQualityPreview);
        }
      }

      // The baker holds the parameters from the last commit; refresh them or the preview re-bakes
      // the image it already has. Invisible for the authored skies -- their background is drawn
      // per-ray from settings, so only the lighting lagged -- but the Bruneton sky *is* its baked
      // image, so without this a drag showed nothing until the mouse came up.
      updateBakerParameters();
      m_envBaker.preview(cmd, m_resources);
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      resetFrame();
    }
  }
  else if(m_envCommitOwed)
  {
    // First frame since the edits stopped: pay for the full rebuild the previews skipped.
    m_envCommitOwed = false;
    onEnvironmentChanged();
  }

  // Recycle staging from completed frames, then tag new uploads with this frame's signal.
  m_resources.staging.releaseCompletedAllocations();
  m_resources.staging.updateFrameSemaphoreState(nvvk::SemaphoreState::makeFixed(m_app->getFrameSignalSemaphore()));

  // Empty scene, clear the G-Buffer
  if(!m_resources.gpuView().hasContent())
  {
    reportMemory({});
    clearGbuffer(cmd);
    return;
  }

  // Keep runtime denoiser guide usage synchronized with the optimal-shader feature set.
  // PathTracer::ensureShadersAndPipelines() watches currentFeatureSet and recompiles when this flips.
  m_resources.currentFeatureSet.set(nvvkgltf::SceneFeatureSet::eDlssGuide, dlssGuideRequired());


  // Start the profiler section for the GPU timer
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  updateClickPickState();   // Poll the previous click's async ray-pick, apply it once ready (see m_pendingClickPick)
  updateInspectState();     // ...and the pixel readout's
  // Check for changes
  bool changed{false};
  changed |= m_pathTracer.keepOneAccumulationDenoiser(m_resources);  // OptiX or OIDN, not both
  changed |= syncRetainedScene(cmd);  // the clip, and the residency's sync of the host's edits

  // #DLSS motion: the retained scene keeps both tables current every sync (a row at rest is no motion)
  const bool dlss = m_pathTracer.isDlssEnabled();
  m_resources.dlssInstanceMotionActive = dlss && m_resources.gpuView().prevObjectToWorldTable() != 0;
  m_resources.dlssDeformMotionActive   = dlss && m_resources.gpuView().prevPositionTable() != 0;

  if(changed)
  {
    resetFrame();
  }
  if(updateCloudView(cmd))
    resetFrame();  // the sky the still is of changed
  refreshSampleTargets();                    // --outputSamples: the stops this frame may end on
  bool frameChanged = updateFrameCounter();  // Check if the frame counter has changed
  // A shader variant built off the frame goes in only when a frame draws: one draws for it even with
  // the still converged, or the edits it holds back would wait for an unrelated change
  const bool variantWaiting = m_pathTracer.variantBuilt();

  const bool traced = changed || frameChanged || variantWaiting;
  if(traced)
  {
    if(m_resources.frameCount == 0)
    {
      m_cpuTimer.reset();
      m_cpuTimePrinted = false;  // Reset print flag when rendering starts
    }

    // Update the scene frame information uniform buffer
    const glm::mat4          viewProj = m_cameraManip->getPerspectiveMatrix() * m_cameraManip->getViewMatrix();
    const VkExtent2D         gbufSize = m_resources.gBuffers.getSize();
    const glm::quat          envRot   = m_resources.settings.envRotationQuat();  // normalised
    shaderio::SceneFrameInfo finfo{
        .viewMatrix     = m_cameraManip->getViewMatrix(),
        .projInv        = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
        .viewInv        = glm::inverse(m_cameraManip->getViewMatrix()),
        .viewProjMatrix = viewProj,
        .prevMVP        = m_prevMVP,
        .jitter         = {0.0f, 0.0f},
        .imageSize      = {float(gbufSize.width), float(gbufSize.height)},
        .flags =
            ((m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? shaderio::eSceneIsOrthographic : 0)
            | (m_resources.settings.useSolidBackground ? shaderio::eSceneUseSolidBackground : 0)
            // A baked analytic sky IS an environment image by the time the shaders see it:
            // same lat-long texture, same alias table. Flagging it as such is what keeps
            // sampling and MIS free of per-sky-type branches.
            | ((m_resources.settings.envSystem == shaderio::EnvSystem::eHdr || isBakedEnvironment(m_resources.settings.envSystem)) ?
                   shaderio::eSceneUseHdrEnvironment :
                   0)
            | ((m_resources.settings.envSystem == shaderio::EnvSystem::eNone) ? shaderio::eSceneUseNoEnvironment : 0)
            | (m_resources.settings.useInfinitePlane ? shaderio::eSceneUseInfinitePlane : 0)
            | ((m_resources.settings.useInfinitePlane && m_resources.settings.isShadowCatcher) ? shaderio::eSceneInfinitePlaneShadowCatcher : 0),
        .envRotation               = {envRot.x, envRot.y, envRot.z, envRot.w},
        .envBlur                   = m_resources.settings.hdrBlur,
        .envIntensity              = m_resources.settings.hdrEnvIntensity,
        .backgroundColor           = m_resources.settings.solidBackgroundColor,
        .visualization             = m_resources.settings.visualization,
        .infinitePlaneDistance     = m_resources.settings.infinitePlaneDistance,
        .infinitePlaneBaseColor    = m_resources.settings.infinitePlaneBaseColor,
        .infinitePlaneMetallic     = m_resources.settings.infinitePlaneMetallic,
        .infinitePlaneRoughness    = m_resources.settings.infinitePlaneRoughness,
        .shadowCatcherDarkenAmount = std::max(m_resources.settings.shadowCatcherDarkness, 0.0f),
        // Only the physical sky has a disk to draw; a null pointer is how every other environment
        // tells the background passes there is nothing to add. Declaration order matters here --
        // these sit before envSkyType in SceneFrameInfo.
        .sunDisk      = (m_resources.settings.envSystem == shaderio::EnvSystem::eSky) ?
                            reinterpret_cast<shaderio::SkySunDisk*>(m_envBaker.sunDiskAddress()) :
                            nullptr,
        .sunDirection = m_resources.sunDirection,
        // The sky's own sun, supplied only while the scene has no marked one. Selecting an
        // environment must not edit the user's scene, so nothing is added to the graph until a
        // save asks for it; from then on the marked light is the sun and this stays off.
        .rendererSunLight = (skyHasSun(m_resources.settings.envSystem) && m_skySun.lightIndex() < 0) ? 1 : 0,
        .sunLightIndex    = skyHasSun(m_resources.settings.envSystem) ? m_skySun.lightIndex() : -1,
        .envSkyType       = authoredSkyType(m_resources.settings.envSystem),
        .skyOmi           = buildSkyOmiParameters(),
        // Aerial perspective. Physical sky only (the sunDisk gate above says which), and the
        // altitude is the same fixed observer the bake uses -- metres in the settings because that
        // is what a person types, kilometres here because that is what the atmosphere works in.
        .aerialPerspectiveScale = m_resources.settings.atmoAerialPerspectiveScale,
        .observerAltitudeKm     = aerialPerspectiveEyeAltitudeKm(),
        // The gradient sky's sun. Zero for every other environment: the physical sky's sun comes
        // from `sunDisk` above, and no other sky has one.
        .skySunIlluminance = (m_resources.settings.envSystem == shaderio::EnvSystem::eGradient) ? gradientSunIlluminance() :
                                                                                                  glm::vec3(0.0F),
        .envSamplingGridWidth  = m_resources.hdrIbl.getSamplingGrid().width,
        .envSamplingGridHeight = m_resources.hdrIbl.getSamplingGrid().height,
        .skySunScale           = m_resources.settings.skySunScale,
        .envPickEstimate       = 0.25F * m_resources.hdrIbl.getIntegral(),
    };
    fillGround(finfo);
    fillHaze(finfo);
    // The clouds where the camera looks, while the view image holds the clouds of the moment: it
    // looks where it was baked, which the camera may since have turned from (updateCloudView).
    if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky && m_skyClouds.viewCurrent())
    {
      const SkyCloudView& view = m_skyClouds.view();
      finfo.cloudViewRight     = view.right;
      finfo.cloudViewUp        = view.up;
      finfo.cloudViewForward   = glm::vec4(view.forward, 1.0F);
      finfo.cloudViewUvScaleX  = float(view.size.x) / float(CLOUD_VIEW_CAPACITY);
      finfo.cloudViewUvScaleY  = float(view.size.y) / float(CLOUD_VIEW_CAPACITY);
    }
    // Update the camera information
    m_prevMVP = finfo.viewProjMatrix;

    vkCmdUpdateBuffer(cmd, m_resources.bFrameInfo.buffer, 0, sizeof(shaderio::SceneFrameInfo), &finfo);
    // Update the sky
    m_resources.sunYIsUp = m_cameraManip->getUp().y > 0.5f;
    // Make sure buffer is ready to be used
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    ++m_resources.drawnFrameCount;
    m_pathTracer.onRender(cmd, m_resources);
  }
  else
  {
    // Print CPU time only once after render completes
    if(!m_cpuTimePrinted)
    {
      LOGI("Rendering finished: %f ms\n", m_cpuTimer.getMilliseconds());
      m_cpuTimePrinted = true;
    }
    // Auto denoise the final image of the accumulation
    if(m_resources.accumulationComplete())
    {
      m_pathTracer.onAccumulationComplete(cmd, m_resources);
      // A headless run given --ptMaxSamples is done: this frame shows the final image (and its
      // denoise, which OptiX makes on this first idle frame), and --frames is only an upper bound.
      // Not while an image is still decoding: a large file's last texture can land after the first
      // samples are in, and the still begins again with it, so what is saved has every texture (two
      // runs of a 326 MB figure differed where its top's map had landed in one of them only).
      const bool decoding = m_resources.retainedGpu && m_resources.retainedGpu->residency().queuedDecodes() > 0;
      if(isHeadlessMode() && m_parameterParser && m_parameterParser->wasParsed("ptMaxSamples") && m_app && !decoding)
        m_app->close();
    }
  }

  // Apply the post-processing effects
  const bool nrAllowed = tonemap(cmd);
#if defined(USE_DLSS) && defined(USE_DLSSNR)
  // DLSS-NR: enhance eImgTonemapped (LDR) after tonemapping; NR expects display-range input.
  // Skip on guide buffers, visualization debug modes, and bypassed-tonemapper paths.
  if(nrAllowed)
    if(Dlss* dlss = activeDlss())
      if(dlss->isNrActive())
      {
        auto nrSection = m_profilerGpuTimer.cmdFrameSection(cmd, "DLSS-NR");
        dlss->evaluateNr(cmd);
      }
#endif
  // The SDR image is final: the HDR one keeps a copy to tell the overlays drawn next from it
  m_hdrOutput.cmdKeepSdr(cmd, m_resources);
  // The wipe's kept frame over its side, after that copy: the HDR image shows that side at SDR white,
  // as it shows an overlay. Not on a screenshot's frame, which is the live image whole.
  if(m_screenshotPending.empty())
    cmdWipe(cmd);
  // The overlays go into the tonemapped image; a frame a screenshot takes is drawn without them
  // (the stream shows that one frame bare, the next has them again)
  const bool overlays = m_screenshotPending.empty();
  if(overlays)
    silhouette(cmd);
  updateSampleCaptures();  // --outputSamples: the image is final for this frame
  // Async click ray-pick: record the pick dispatch on this frame's own command buffer - the TLAS is
  // already proven valid here (the path tracer just used it above) - then hand off a
  // frame-semaphore to poll instead of blocking. See m_pendingClickPick's comment (renderer.hpp).
  if(m_pendingClickPick)
  {
    m_rayPicker.run(cmd, {.modelViewInv   = m_pendingClickPick->modelViewInv,
                          .perspectiveInv = m_pendingClickPick->perspectiveInv,
                          .isOrthographic = m_pendingClickPick->isOrthographic,
                          .pickPos        = m_pendingClickPick->pickPos,
                          .tlas           = m_resources.gpuView().tlas()});
    m_pendingClickResult =
        PendingClickResult{.semaphoreState = nvvk::SemaphoreState::makeFixed(m_app->getFrameSignalSemaphore()),
                           .isDoubleClick  = m_pendingClickPick->isDoubleClick};
    m_pendingClickPick.reset();
  }
  // The pixel readout reads the frame's image as drawn, after every pass that writes it
  cmdInspect(cmd, traced);
  // The grid and transform gizmo are drawn into the frame: a headless --stream is the native
  // viewport, so the grid shows there.
  if(overlays)
  {
    renderVisualHelpers(cmd);
  }
}

//--------------------------------------------------------------------------------------------------
// Called with headless rendering, to save the final image
void GltfRenderer::onLastHeadlessFrame()
{
  logDenoiserTimings();
  if(m_sampleCapture.next < m_sampleCapture.targets.size())
    LOGW("--outputSamples: %d spp and above were not reached (the image has %d spp); raise --frames\n",
         m_sampleCapture.targets[m_sampleCapture.next], m_sampleCapture.lastSamples);
  saveHeadlessOutputImage();
}

#if defined(USE_DLSS)
//--------------------------------------------------------------------------------------------------
// The path tracer's DLSS (Ray Reconstruction).
Dlss* GltfRenderer::activeDlss()
{
  return m_pathTracer.getDlss();
}

const Dlss* GltfRenderer::activeDlss() const
{
  return m_pathTracer.getDlss();
}
#endif

//--------------------------------------------------------------------------------------------------
// Runtime gate for path-tracer guide-buffer shader code.
bool GltfRenderer::dlssGuideRequired() const
{
#if defined(USE_DLSS) || defined(USE_OPTIX_DENOISER) || defined(USE_OIDN_DENOISER)
  // The build with guides is the one in the exe, warmed at start, and it writes none unless a denoiser
  // asks (ePtDenoiserGuides). Without guides there's nothing prepared: turning the denoiser off froze
  // the renderer for 27 s the first time, to compile and build the pipeline of a shader that differs
  // only by code that doesn't run. The optimal shader is built per scene anyway, and leaves it out.
  if(!m_resources.settings.optimalShader)
    return true;
#endif

#if defined(USE_DLSS)
  if(m_pathTracer.isDlssEnabled())
    return true;
#endif

#if defined(USE_OPTIX_DENOISER)
  if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser(); optix && optix->isEnabled())
    return true;
#endif

#if defined(USE_OIDN_DENOISER)
  if(const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser(); oidn && oidn->isEnabled())
    return true;
#endif

  return false;
}

//--------------------------------------------------------------------------------------------------
// Apply the tonemapper on the rendered image
bool GltfRenderer::tonemap(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);
  return runTonemapPass(cmd);
}

VkImage GltfRenderer::presentImage() const
{
  const VkImage hdr = m_hdrOutput.active() ? m_hdrOutput.image() : VK_NULL_HANDLE;
  return hdr != VK_NULL_HANDLE ? hdr : tonemappedImage();
}

VkFormat GltfRenderer::presentFormat() const
{
  return m_hdrOutput.active() && m_hdrOutput.image() != VK_NULL_HANDLE ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
}

bool GltfRenderer::setDisplayHdr(const DisplayHdr& display)
{
  m_hdrOutput.setDisplay(display, m_presentHdr, m_presentHdrPeak);
  return m_hdrOutput.active();
}

bool GltfRenderer::runTonemapPass(VkCommandBuffer cmd)
{
  // Select which buffer to tonemap based on user selection
  VkDescriptorImageInfo inputBuffer =
      m_resources.gBuffers.getColorSampleDescriptorImageInfo(Resources::eImgRendered, m_resources.linearSampler);
  VkExtent2D gbufSize         = m_resources.gBuffers.getSize();
  bool       usingGuideBuffer = false;
  bool       bypassTonemapper = false;


  // OptiX / OIDN denoised output (path-tracer only). Routed via the global DisplayBuffer toggle the
  // denoisers write when they denoise or the user clicks their thumbnail.
#if defined(USE_OPTIX_DENOISER)
  if(m_resources.settings.displayBuffer == DisplayBuffer::eOptixDenoised)
  {
    const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser();
    if(optix && optix->hasValidDenoisedOutput())
    {
      inputBuffer         = optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferDenoised);
      inputBuffer.sampler = m_resources.linearSampler;
    }
  }
#endif
#if defined(USE_OIDN_DENOISER)
  if(m_resources.settings.displayBuffer == DisplayBuffer::eOidnDenoised)
  {
    const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser();
    if(oidn && oidn->hasValidDenoisedOutput())
    {
      inputBuffer         = oidn->getDescriptorImageInfo(OidnDenoiser::eGBufferDenoised);
      inputBuffer.sampler = m_resources.linearSampler;
    }
  }
#endif
  // The DLSS hand-off, shown in place of the denoiser's view while OptiX or OIDN takes over from DLSS
  if(const VkDescriptorImageInfo* handoff = m_pathTracer.handoffImage(m_resources))
  {
    inputBuffer         = *handoff;
    inputBuffer.sampler = m_resources.linearSampler;
  }

#if defined(USE_DLSS)
  // Displaying or not DLSS guide buffer.
  if(Dlss* dlss = activeDlss())
  {
    if(auto guide = dlss->activeGuideImage())
    {
      inputBuffer         = guide->image;
      inputBuffer.sampler = m_resources.linearSampler;
      gbufSize            = guide->extent;
      usingGuideBuffer    = true;

      // Clear the output image to a distinct color so the DLSS render-resolution borders are
      // visible when the guide buffer is smaller than the display.
      VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
      vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                           &kBackgroundClearColor, 1, &range);
      // Barrier: clear must complete before tonemapper compute shader runs
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    }
  }
#endif

  // Disable tonemapping for debug buffers or guide buffers (display raw values)
  shaderio::TonemapperData tonemapperData = m_resources.tonemapperData;
  const bool               vizNormal      = (m_resources.settings.visualization == shaderio::Visualization::eRendered
                          || m_resources.settings.visualization == shaderio::Visualization::eClay);
  if(!vizNormal || usingGuideBuffer)
    tonemapperData.isActive = 0;
  if(bypassTonemapper)
    tonemapperData.isActive = 0;
  tonemapperData.falseColor = m_falseColor ? 1 : 0;
  // the grain moves with each frame drawn, as film's does, and holds on a finished still
  tonemapperData.grainSeed = uint32_t(m_resources.drawnFrameCount);

  // The glow and the star (Bloom), then the shadows lifted and the highlights held (local exposure),
  // on what the tone mapper and the HDR output take, not on a debug or guide view; an .exr keeps the
  // image without them (m_tonemapInput below)
  const VkDescriptorImageInfo unbloomed = inputBuffer;
  const bool  picture  = vizNormal && !usingGuideBuffer && !bypassTonemapper && tonemapperData.isActive;
  const float exposure = tonemapperData.autoExposure ? 1.0F : tonemapperData.exposure;
  if(picture && m_bloom.active())
  {
    auto bloomSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Bloom");
    inputBuffer       = m_bloom.cmdApply(cmd, m_resources, inputBuffer, gbufSize, exposure);
  }
  if(picture && m_localExposure.active())
  {
    auto localSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Local exposure");
    inputBuffer       = m_localExposure.cmdApply(cmd, m_resources, inputBuffer, gbufSize, exposure, m_profilerGpuTimer);
  }

  m_resources.tonemapper.runCompute(cmd, gbufSize, tonemapperData, inputBuffer,
                                    m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgTonemapped));

  // The HDR image from the same input and settings, while frames are HDR. Not for a DLSS guide view,
  // drawn into a corner of the SDR image, nor with auto exposure, which only the SDR tonemapper
  // measures: those frames show the SDR image whole (HdrOutput::cmdFinish).
  if(!usingGuideBuffer && !bypassTonemapper && !tonemapperData.autoExposure)
    m_hdrOutput.cmdTonemap(cmd, m_resources, gbufSize, tonemapperData, inputBuffer);
  // What an .exr saved from this frame is made from (saveExr)
  m_tonemapInput = {.image = unbloomed, .size = gbufSize, .data = tonemapperData};

  // Barrier: tonemapper compute writes must be visible to both the DLSS-NR compute pass (which
  // reads eImgTonemapped immediately after tonemap()) and the silhouette fragment shader.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

  // NR post-process is only valid on the main rendered path (not guide buffers, viz modes, or bypassed tonemapper).
  return vizNormal && !usingGuideBuffer && !bypassTonemapper;
}

//--------------------------------------------------------------------------------------------------
// Render the silhouette of the selected object(s)
//
// Selection visualization uses an ObjectID buffer (R32_UINT, filled on frame 0 by the path tracer
// with the first-hit render node ID per pixel) and a selection bitmask (one bit per
// render node). The silhouette compute shader reads ObjectID per pixel, checks the bitmask, and
// applies a Sobel edge filter where the bit is set, then composites the outline onto the image.
void GltfRenderer::silhouette(VkCommandBuffer cmd)
{
  // Run the silhouette pass only when something is selected (one primitive or a node and its branch).
  // We rebuild the CPU bitmask from selectedRenderNodes, upload it, then dispatch the silhouette compute.
  if(!m_resources.selectedRenderNodes.empty())
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

    // Rebuild selection bitmask and ensure GPU buffer exists and is up to date.
    int numRenderNodes = static_cast<int>(m_resources.gpuView().renderNodeCapacity());
    m_resources.updateSelectionBitMask(numRenderNodes);
    const VkDeviceSize bitmaskBytes = m_resources.selectionBitMask.size() * sizeof(uint32_t);
    if(bitmaskBytes > 0)
    {
      // Build or update the selection bitmask buffer, a power of two of the bytes it needs (render
      // node capacity grows in powers of two); a smaller one goes once no frame in flight reads it
      if(m_resources.bSelectionBitMask.buffer == VK_NULL_HANDLE || m_resources.bSelectionBitMask.bufferSize < bitmaskBytes)
      {
        if(m_resources.bSelectionBitMask.buffer != VK_NULL_HANDLE)
          m_app->submitResourceFree([alloc = &m_resources.allocator, buffer = m_resources.bSelectionBitMask]() mutable {
            alloc->destroyBuffer(buffer);
          });
        m_resources.bSelectionBitMask = {};
        NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bSelectionBitMask, std::bit_ceil(bitmaskBytes),
                                                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                      VMA_MEMORY_USAGE_CPU_TO_GPU));
        NVVK_DBG_NAME(m_resources.bSelectionBitMask.buffer);
      }

      // In case the bitmask is too large (> 524,288 render nodes), we need to update it in chunks
      constexpr VkDeviceSize kMaxCmdUpdateSize = 65536;
      for(VkDeviceSize offset = 0; offset < bitmaskBytes; offset += kMaxCmdUpdateSize)
      {
        const VkDeviceSize chunkSize = std::min(kMaxCmdUpdateSize, bitmaskBytes - offset);
        vkCmdUpdateBuffer(cmd, m_resources.bSelectionBitMask.buffer, offset, chunkSize,
                          reinterpret_cast<const uint8_t*>(m_resources.selectionBitMask.data()) + offset);
      }
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    std::vector<VkDescriptorImageInfo> imageInfos = {
        m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgSelection),
        m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgTonemapped),
    };
    VkDescriptorBufferInfo bitmaskBufferInfo = {m_resources.bSelectionBitMask.buffer, 0, bitmaskBytes};
    m_silhouette.dispatch(cmd, m_resources.gBuffers.getSize(), imageInfos, bitmaskBufferInfo,
                          static_cast<uint32_t>(m_resources.selectionBitMask.size()));
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
  }
}

//--------------------------------------------------------------------------------------------------
// Render visual helpers (grid + transform gizmo) onto the tonemapped image
void GltfRenderer::renderVisualHelpers(VkCommandBuffer cmd)
{
  // Sync grid and snap settings to visual helpers
  m_visualHelpers.grid.setVisible(m_resources.settings.showGrid);
  m_visualHelpers.grid.style().baseUnit = m_resources.settings.gridUnit;
  m_visualHelpers.transform.setSnapEnabled(m_resources.settings.snapEnabled);
  m_visualHelpers.transform.setSnapValues(m_resources.settings.gridUnit, m_resources.settings.snapRotation,
                                          m_resources.settings.snapScale);

  // Sync gizmo attachment from selection state
  updateGizmoAttachment();

  if(!m_visualHelpers.shouldRender())
    return;

  NVVK_DBG_SCOPE(cmd);

  const VkExtent2D size = m_resources.gBuffers.getSize();
  glm::vec2        viewportSize(static_cast<float>(size.width), static_cast<float>(size.height));
  glm::vec2        depthBufferSize = viewportSize;
#if defined(USE_DLSS)
  // The path tracer writes depth at the DLSS render size into the corner of the G-buffer depth.
  // isActive() is strict (false during async init), so we never pick up a 0x0 inner extent.
  if(const Dlss* dlss = activeDlss(); dlss && dlss->isActive())
  {
    const VkExtent2D rs = dlss->getRenderSize();
    depthBufferSize     = {static_cast<float>(rs.width), static_cast<float>(rs.height)};
  }
#endif

  m_visualHelpers.render(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                         m_resources.gBuffers.getColorAttachmentView(Resources::eImgTonemapped), m_resources.descriptorSet,
                         m_cameraManip->getViewMatrix(), m_cameraManip->getPerspectiveMatrix(), viewportSize, depthBufferSize);
}

//--------------------------------------------------------------------------------------------------
// Set DLSS hardware/extension availability per NGX feature (Ray Reconstruction / Super Resolution).
// This should be called early, before any DLSS initialization occurs.
void GltfRenderer::setDlssHardwareAvailability(bool rrAvailable, bool srAvailable)
{
  m_resources.settings.dlssRrHardwareAvailable = rrAvailable;
  m_resources.settings.dlssSrHardwareAvailable = srAvailable;
}

//--------------------------------------------------------------------------------------------------
// Set Shader Execution Reordering (VK_EXT_ray_tracing_invocation_reorder) availability. Call
// before the path tracer is attached: it decides whether SER is compiled into the shader.
void GltfRenderer::setSerAvailable(bool available)
{
  m_resources.settings.serHardwareAvailable = available;
}

//--------------------------------------------------------------------------------------------------
// Names of the extensions enabled on the device. Call before the path tracer is attached: the
// vk_oidn denoiser picks its optional features (tensor cores, ...) from them.
void GltfRenderer::setDeviceExtensions(std::vector<std::string> extensions)
{
  m_resources.deviceExtensions = std::move(extensions);
}

//--------------------------------------------------------------------------------------------------
// Set Opacity Micromap (VK_KHR_opacity_micromap) availability: whether the device supports it, and
// whether the user enabled it (--useOpacityMicromap). When unavailable, the
// EXT_mesh_opacity_micromap glTF extension is ignored. Call early, before scene creation.
void GltfRenderer::setOpacityMicromapAvailable(bool hardware, bool enabled)
{
  m_resources.settings.opacityMicromapMissing   = enabled && !hardware;
  m_resources.settings.opacityMicromapSupported = hardware && enabled;
}

//--------------------------------------------------------------------------------------------------
// Clear the G-Buffer
void GltfRenderer::clearGbuffer(VkCommandBuffer cmd)
{
  VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
  vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                       &kBackgroundClearColor, 1, &range);

  // Ensure the clear operation completes before any subsequent reads from this image
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
}

//--------------------------------------------------------------------------------------------------
// Create the uniform buffer for frame-specific data: per-frame camera matrices, environment
// settings and debug information, updated each frame.
//
void GltfRenderer::createResourceBuffers()
{
  // Create the buffer of the current camera transformation, changing at each frame
  NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bFrameInfo, sizeof(shaderio::SceneFrameInfo),
                                                VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                VMA_MEMORY_USAGE_CPU_TO_GPU));
  NVVK_DBG_NAME(m_resources.bFrameInfo.buffer);
}

//--------------------------------------------------------------------------------------------------
// Create the descriptor set and the pipelines
// There are two descriptor: one for the textures (set) and one (push) for the top level acceleration structure and the default output image
void GltfRenderer::createDescriptorSets()
{
  // Reserve texture descriptors (m_maxTextures from renderer.hpp, clamped to device limits)
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &deviceProperties);
  m_maxTextures = std::min(m_maxTextures, deviceProperties.limits.maxDescriptorSetSampledImages - 1);  // Set limits of sample textures (defaut: 100 000)
  m_maxSamplers = std::min(m_maxTextures, deviceProperties.limits.maxDescriptorSetSamplers - 1);

  // 0: Descriptor SET: scene material images (SAMPLED_IMAGE) and samplers (SAMPLER) are now separate
  // bindless arrays instead of one combined-image-sampler array (see GltfTextureInfo.index/samplerIndex).
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTextures, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                              m_maxTextures, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eSamplers, VK_DESCRIPTOR_TYPE_SAMPLER,
                                              m_maxSamplers, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  // 2D environment textures (see shaderio.h HDR_* indices): the lat-long HDR environment image.
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTexturesHdr, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              HDR_TEXTURE_COUNT, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  NVVK_CHECK(m_resources.descriptorBinding[0].createDescriptorSetLayout(
      m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, &m_resources.descriptorSetLayout[0]));
  NVVK_DBG_NAME(m_resources.descriptorSetLayout[0]);

  // Pool sizes are derived from descriptorBinding[0] (now SAMPLED_IMAGE + SAMPLER for the scene, plus
  // COMBINED_IMAGE_SAMPLER for the HDR array). Each nvapp::ImTexture (viewport: 1, DLSS guides: 8,
  // OptiX denoised: 1 + margin) additionally uses one VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE descriptor.
  constexpr uint32_t                kImTextureMaxSets = 15;
  std::vector<VkDescriptorPoolSize> poolSize          = m_resources.descriptorBinding[0].calculatePoolSizes();
  poolSize.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kImTextureMaxSets});
  VkDescriptorPoolCreateInfo dpoolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  // allows descriptor sets to be updated after they have been bound to a command buffer
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // individual descriptor sets can be freed from the descriptor pool
      .maxSets       = kImTextureMaxSets + 1,                      // ImTexture sets + the scene texture set
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &dpoolInfo, nullptr, &m_resources.descriptorPool));
  NVVK_DBG_NAME(m_resources.descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_resources.descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_resources.descriptorSetLayout[0],
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_resources.descriptorSet));
  NVVK_DBG_NAME(m_resources.descriptorSet);


  // 1: Descriptor PUSH: top level acceleration structure and the output image
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eTlas,
                                              VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eOutImages, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10,
                                              VK_SHADER_STAGE_ALL);
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eOutDepth, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                              VK_SHADER_STAGE_ALL);

  NVVK_CHECK(m_resources.descriptorBinding[1].createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                                        &m_resources.descriptorSetLayout[1]));
  NVVK_DBG_NAME(m_resources.descriptorSetLayout[1]);
}

//--------------------------------------------------------------------------------------------------
// Recompile the path tracer's shaders and show the result.
// SYNC NOTE: the recompile destroys live pipelines, so the queue must be drained first.
bool GltfRenderer::reloadShaders()
{
  vkQueueWaitIdle(m_app->getQueue(0).queue);
  const bool compiled = compileShaders();
  resetFrame();
  return compiled;
}

//--------------------------------------------------------------------------------------------------
// Recompile the path tracer's shaders.
bool GltfRenderer::compileShaders()
{
  nvutils::ScopedTimer st(__FUNCTION__);
  return m_pathTracer.reloadShader(m_resources);
}

//--------------------------------------------------------------------------------------------------
// Reset the frame counter
void GltfRenderer::resetFrame()
{
  m_resources.frameCount = -1;
  m_framesSinceChange    = 0;
}

//--------------------------------------------------------------------------------------------------
// Update the frame counter
// This is called every frame to update the frame counter or to reset it if the camera has changed
// The frame counter is used to limit the number of frames rendered
// If the frame counter is greater than the maximum number of frames, the rendering stops
// Returns true if the frame counter is less than the maximum number of frames
bool GltfRenderer::updateFrameCounter()
{

  static nvutils::CameraManipulator::Camera ref_camera{};

  const auto currentCamera = m_cameraManip->getCamera();

  if(ref_camera != currentCamera)
  {
    resetFrame();
    ref_camera = currentCamera;
  }

  if(m_resources.accumulationComplete())
  {
    return false;
  }
  // a drape on its way (setStillPaused): what the still would refine is about to change
  if(m_stillPaused && m_framesSinceChange >= kPausedStillFrames)
    return false;
  m_framesSinceChange++;
  // The DLSS frames of a hand-off delay are no part of the accumulation: the count waits at 1 (frame
  // 0 was DLSS's) so the frame bound cannot end the still before it begins. They draw all the same.
  if(m_resources.frameCount >= 1 && m_pathTracer.dlssHandoffDelayPending())
    return true;
  m_resources.frameCount++;
  return true;
}

//--------------------------------------------------------------------------------------------------
// Destroy the resources
// Resource cleanup follows a specific order to prevent validation errors:
// 1. First flush any pending command buffers to ensure GPU work is complete
// 2. Then destroy higher-level objects before their dependencies
// 3. Finally clean up allocator after all resources using it are destroyed
// This ensures proper synchronization and prevents use-after-free errors
void GltfRenderer::destroyResources()
{

  m_loadPipeline.destroy();
  m_resources.retainedGpu.reset();  // while the allocator, the uploader and the sampler pool live
  m_resources.retainedSpare.reset();

  m_resources.allocator.destroyBuffer(m_resources.bFrameInfo);
  if(m_resources.bSelectionBitMask.buffer != VK_NULL_HANDLE)
    m_resources.allocator.destroyBuffer(m_resources.bSelectionBitMask);

  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[0], nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[1], nullptr);
  vkDestroyDescriptorPool(m_device, m_resources.descriptorPool, nullptr);
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);

  m_profilerGpuTimer.deinit();
  {
#ifdef USE_NVMCP
    // Automation (MCP) reads the timeline from a worker thread under this same lock, so take it
    // here: the release only happens once no read is in flight, and every later read sees nullptr.
    const std::lock_guard<std::mutex> lock(m_profilerTimelineMutex);
#endif
    g_profilerManager.destroyTimeline(m_profilerTimeline);
    m_profilerTimeline = nullptr;
  }
  m_silhouette.deinit(m_resources);
  m_hdrOutput.deinit(m_resources);
  m_bloom.deinit(m_resources);
  m_localExposure.deinit(m_resources);
  if(m_wipeFrame.image)
    m_resources.allocator.destroyImage(m_wipeFrame);
  m_wipeFrame = {};
  m_skyBruneton.deinit(m_resources);
  m_envBaker.deinit(m_resources);
  m_skyClouds.deinit(m_resources);

  m_resources.tonemapper.deinit();
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  m_resources.tonemappedUi.deinit();
  m_resources.gBuffers.deinit();
  m_resources.hdrIbl.deinit();
  m_resources.samplerPool.deinit();
  m_resources.staging.deinit();
  m_rayPicker.deinit();
  m_resources.allocator.deinit();
}



