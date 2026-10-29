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

#pragma once

#include "settings_registry.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

#include <nvvk/sbt_generator.hpp>
#include <nvutils/profiler.hpp>
#include "denoiser_handoff.hpp"
#include "renderer_base.hpp"
#include "sample_schedule.hpp"
#include "selection_write.hpp"
#include "utils.hpp"
#include "pipeline_cache_util.hpp"
#include "scene_feature_detection.hpp"
#include "ui_busy_window.hpp"
#include "variant_compile_job.hpp"

// #DLSS
#if defined(USE_DLSS)
#include "dlss.hpp"
#endif

// #OPTIX
#if defined(USE_OPTIX_DENOISER)
#include "optix_denoiser.hpp"
#endif

// #OIDN
#if defined(USE_OIDN_DENOISER)
#include "oidn_denoiser.hpp"
#endif


class PathTracer : public BaseRenderer
{
public:
  PathTracer();
  ~PathTracer() override = default;

  enum class RenderTechnique
  {
    RayQuery,
    RayTracing
  };

  void onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler) override;
  void setProfilerTimeline(nvutils::ProfilerTimeline* timeline)
  {
    m_profilerTimeline = timeline;
#if defined(USE_OIDN_DENOISER)
    m_oidn->setProfilerTimeline(timeline);
#endif
  }
  void onDetach(Resources& resources) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources) override;
  bool onUIRender(Resources& resources) override;
  void onRender(VkCommandBuffer cmd, Resources& resources) override;
  // Called instead of onRender() while the accumulation is complete
  void onAccumulationComplete(VkCommandBuffer cmd, Resources& resources);
  // The DLSS hand-off's image, to show instead of the denoiser's view (null when not shown)
  const VkDescriptorImageInfo* handoffImage(const Resources& resources) const;
  void                         onSceneInvalidated(Resources& resources) override;
  void                         notifyDlssContentReset(Resources& resources) override;

  void updateDlssResources(VkCommandBuffer cmd, Resources& resources);
  void updateOptiXResources(VkCommandBuffer cmd, Resources& resources);
  void pushDescriptorSet(VkCommandBuffer cmd, Resources& resources, VkPipelineBindPoint bindPoint) const;
  void createPipeline(Resources& resources) override;
  void createRqPipeline(Resources& resources);
  void createRtxPipeline(Resources& resources);
  // Creates the embedded DLSS path tracer's pipeline for the technique in use (ray tracing or ray
  // query) into the pipeline cache on a thread of its own, so the first path-traced frame after a
  // renderer update finds it there instead of building it on the application thread (10-25 s).
  // After createPipeline(); createRtxPipeline() and createRqPipeline() wait for it.
  void startPipelineWarm(Resources& resources);
  bool compileShader(Resources& resources, bool fromFile = true) override;
  // The scene features the live variant has compiled in: the optimal build's; none when every one
  // is (the optimal shader off, or the embedded build standing in for a failed compile)
  std::optional<nvvkgltf::SceneFeatureSet> liveShaderFeatures();
  // User-initiated hot reload: drop cached variants and recompile from Slang source.
  bool reloadShader(Resources& resources);
  void setBusyWindow(BusyWindow* busy) { m_busyWindow = busy; }
  // --cacheDir. Empty (the default) keeps the caches next to the executable. Set before onAttach.
  void setCacheDir(const std::filesystem::path& dir) { m_cacheDir = dir; }
  // Lens radius squared in metres (0: a pinhole). autoFocus uses the eye-to-target distance. Blades:
  // below 3 a round aperture, else a polygon with a corner at bladeRotation (radians).
  void setDepthOfField(float focalDistance, float aperture, bool autoFocus, int blades = 0, float bladeRotation = 0)
  {
    m_pushConst.focalDistance = std::max(1e-9f, focalDistance);
    m_pushConst.aperture      = std::max(0.f, aperture);
    m_pushConst.bladeCount    = blades;
    m_pushConst.bladeRotation = bladeRotation;
    m_autoFocus               = autoFocus;
  }

  // Register command line parameters
  /// Declare this subsystem's settings (command line + benchmark + MCP + persistence).
  /// resetAccumulation restarts the frame count: settings that change what a frame renders call it
  /// when they are set outside the UI (whose changes reset through onUI's return value).
  void registerParameters(SettingsRegistry* settings, std::function<void()> resetAccumulation);

  VkDevice                        m_device{};  // Vulkan device
  VkPipelineLayout                m_pipelineLayout{};
  VkPipeline                      m_rtxPipeline{};    // Ray tracing pipeline
  VkPipeline                      m_rqPipeline{};     // Ray tracing pipeline
  shaderio::PathtracePushConstant m_pushConst{};      // Information sent to the shader
  bool                            m_autoFocus{true};  // Enable auto-focus
  VkShaderModule                  m_shaderModule{};   // Shader module for RTX

  nvvk::PipelineCacheManager m_pipelineCache{};  // Pipeline cache for faster creation

  // Shader Binding Table (SBT)
  nvvk::Buffer                m_sbtBuffer{};   // Buffer for the Shader Binding Table
  nvvk::SBTGenerator::Regions m_sbtRegions{};  // The SBT regions (raygen, miss, chit, ahit)

  // Ray tracing properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtPipelineProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT m_reorderProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT};

  bool m_supportSER{false};      // True when the device supports SER (Shader Execution Reordering).
  bool m_supportRayQuery{true};  // False where the driver cannot build the ray-query pipeline (see onAttach).

  // Path-traced depth when the G-buffer depth format cannot be a storage image (D32_SFLOAT on AMD):
  // the shader writes it into m_depthStaging, and the first frame copies it into the depth buffer.
  bool               m_depthViaStaging{false};
  bool               m_depthCopyable{true};  // The depth format is 32-bit float, so the staged depth can be copied in
  nvvk::RenderTarget m_depthStaging;         // R32_SFLOAT, G-buffer size
  nvvk::RenderTarget m_convergence;          // RG32_SFLOAT, G-buffer size: shaderio::eConvergence
  nvvk::Buffer       m_depthCopyBuffer;      // Bridge for the color -> depth copy (no direct image copy between them)
  nvvk::Buffer       m_cameraMediumBuffer;   // shaderio::CameraMedium: the medium camera paths start in
  // ptCameraMedium: whether to probe it. The probe passes through surfaces that bound no medium, so
  // through an opening (an eye socket, a mouth) it meets a figure's skin from inside and every path
  // starts in the skin; a host whose camera is never inside a medium turns it off.
  bool m_probeCameraMedium = true;
  // scatterAsThin: 0 the volume walk, 1 the thin-walled preview, 2 the preview on the frames DLSS
  // renders (every frame while the camera moves) and the walk on the frames an accumulation traces,
  // so a host need not switch it (and reset the accumulation) when the view stops
  int m_scatterAsThin = 0;
  bool               m_useSER{true};         // Requested SER state; clamped to m_supportSER each frame.
  bool m_shadowTransmission{true};  // Shadow rays pass through transmissive surfaces (biased; see ePtShadowTransmission).
  bool m_volumeDiffuseExit{false};  // Scatter walks gather light only where they leave the surface (ePtVolumeDiffuseExit).
  bool m_diffuseEntry{false};       // Light enters a scatter walk through a Lambertian lobe (ePtDiffuseEntry). Off: the user preferred the refraction entry's look (2026-10-26).
  int  m_fireflyClampMode{1};       // 0: the whole path; 1: each indirect contribution (ePtClampIndirect).
  bool m_simplifyScatter{true};     // A rough path meets a scattering medium as its thin preview (ePtSimplifyScatter).
  int  m_sampler{1};                // The accumulation's sampler: 0 PCG, 1 Owen-scrambled Sobol (ePtSobol). DLSS frames: PCG.
  int  m_fullLobeEval{3};           // 0: one random lobe per light sample; 1: every lobe, punctual lights; 2: every lobe, mixture-pdf MIS; 3: 2 while the path is direct, else 1 (ePtLobe*).
  // Lobes after a rough bounce are as rough as the path (ePtApproximateCaustics). Off: on a DAZ figure
  // it darkens the irises, whose light comes through the cornea, itself a caustic.
  bool m_approximateCaustics{false};
  // Path regularisation (ePtRegularize): 0 never, 1 on DLSS frames, 2 always. The still has the indirect
  // clamp and the denoiser; regularising it blurred smooth objects in glossy floors and the eyes' corneas.
  int m_regularize{1};
  // Camera rays see sized lights (ePtLightsVisible). Off: a prop sphere lights and shows in reflections,
  // but is not a ball in the frame.
  bool m_lightsVisible{false};
  // The diffuse exit's light-side Fresnel (ePtExitFresnel): grazing light on skin as the boundary passes it
  bool m_exitFresnel{true};
  // Fibre events before they count as surface bounces (PathtracePushConstant::hairBudget): the still's,
  // and the DLSS frames', fewer, since white hair keeps its energy through every one.
  int m_hairBounces{16};
  int m_hairBouncesDlss{4};
  // DLSS frames: past ptHairRouletteFrom fibre events a path continues at most with ptHairRouletteDlss
  float m_hairRouletteDlss{1.0F};
  int   m_hairRouletteFrom{1};
  // DLSS frames: hair paths end at their third fibre event, whose light sample carries the rest of the
  // lock's light (ptHairCutDlss, ptHairCutTail; measured 2026-10-29: moving frame 29.5 -> 24.5 ms,
  // the hair's brightness against the still about as close as the uncut moving view's)
  int   m_hairCutDlss{3};
  float m_hairCutTail{4.0F};
  // An accumulation's pixel stops once its mean has converged (ptAdaptiveStop, ptAdaptiveThreshold,
  // ptAdaptiveMin; shaders/adaptive_stop.h.slang). Measured 2026-10-29 on the bench (ACES, so 0.13 in
  // displayed brightness is 0.1 in the scene's): a 128-sample still 7.96 -> 4.95 s, 0.57/255 from the
  // full one (two full ones differ by 0.18), 0.06/255 darker
  bool  m_adaptiveStop{true};
  float m_adaptiveThreshold{0.13F};
  int   m_adaptiveMin{32};
  SelectionWrite m_selectionWrite;  // when a frame traces the selection image's object IDs
  bool m_pipelineUseSER{false};     // SER value the currently-live pipelines were built with.

  // Variant pipeline cache: avoids slow pipeline (re)compilation by reusing previously built
  // VkShaderModule and pipelines for a given VariantKey. LRU-limited (see kVariantCacheMaxEntries).
  struct VariantKey
  {
    bool wireframe = false;  // True when the shader is the wireframe build.
    bool visualize = false;  // True when the debug-visualization code is compiled in (USE_VISUALIZE).
    bool omm       = false;  // True when the ray-query entry points opt in to opacity micromaps (USE_OMM).
    bool optimal   = false;  // True when the shader is the scene-aware optimized build.
    bool dlss      = false;  // True on the frames DLSS renders (drives USE_DLSS_SHADER: DLSS's guide captures).
    bool dlssGuide = false;  // True when guide-buffer capture is compiled in (USE_GUIDE_SHADER: DLSS, OptiX or OIDN).
    nvvkgltf::SceneFeatureSet features{};  // only meaningful when `optimal == true`
    int dlssTransparency = 0;  // USE_DLSS_TRANSP the pipeline was specialized with (Dlss::TransparencyMode).

    // Same SPIR-V: every shader macro matches. dlssTransparency is a pipeline specialization
    // constant, not a macro, so it is left out. dlss and dlssGuide are compared in every mode (they
    // drive USE_DLSS_SHADER / USE_GUIDE_SHADER independently of optimal); the full extension feature
    // set only matters for the optimal build.
    bool sameShader(const VariantKey& o) const
    {
      return wireframe == o.wireframe && visualize == o.visualize && omm == o.omm && optimal == o.optimal
             && dlss == o.dlss && dlssGuide == o.dlssGuide && (optimal ? (features == o.features) : true);
    }
    bool operator==(const VariantKey& o) const { return sameShader(o) && dlssTransparency == o.dlssTransparency; }
    // The fields sameShader() sees differ on, e.g. "omm 0->1, dlss 1->0", for the recompile log.
    std::string shaderDiff(const VariantKey& to) const;
  };
  VariantKey m_compiled{};  // The variant the live shader module and pipelines were built as. Guarded by m_compileMutex.
  // The live module is the embedded SPIR-V standing in for a variant whose compile failed. It is not
  // parked in the variant cache, so the variant compiles again when it is next asked for after a
  // change; until then no frame retries it. Guarded by m_compileMutex.
  bool m_compiledIsFallback{false};

  // Variant cache entry: stores a shader module, RTX/RQ pipelines, and SBT for a given VariantKey.
  struct VariantCacheEntry
  {
    VariantKey                  key;
    VkShaderModule              shaderModule = VK_NULL_HANDLE;
    VkPipeline                  rtxPipeline  = VK_NULL_HANDLE;
    VkPipeline                  rqPipeline   = VK_NULL_HANDLE;
    bool                        useSER       = false;  // SER value the cached pipelines were built with.
    nvvk::Buffer                sbtBuffer{};           // Buffer for the SBT (Shader Binding Table)
    nvvk::SBTGenerator::Regions sbtRegions{};          // The SBT regions (raygen, miss, chit, ahit)
  };
  std::vector<VariantCacheEntry> m_variantCache;  // MRU front, LRU back
  static constexpr size_t        kVariantCacheMaxEntries = 8;

  // Saves active shader/pipeline/SBT to cache by VariantKey; switches to `newKey`.
  // On hit, restores cached handles and returns true; on miss, clears handles for rebuild.
  // gpuIdle: the caller drained the queue, so whatever it destroys goes at once; otherwise handles a
  // frame in flight may still use are released through the app's frame ring (submitResourceFree).
  bool       swapVariant(Resources& resources, const VariantKey& newKey, bool gpuIdle);
  void       destroyVariantCache(Resources& resources);
  VariantKey makeVariantKey(const Resources& resources) const;  // The variant the current settings ask for.
  // True when that variant is the embedded DLSS build (gltf_pathtrace.slang_dlss.h): DLSS and the
  // guide buffers on, nothing else, on a GPU with SER (the embedded build has AVAILABLE_SER=1).
  bool isEmbeddedDlssVariant(const Resources& resources) const;
  bool isEmbeddedBaseVariant(const Resources& resources) const;
  // Swaps in the wanted variant if it is cached (cheap; no BusyWindow). Leaves live handles alone on a miss.
  bool tryRestoreCachedVariant(Resources& resources);
  // Destroys a variant's handles now (gpuIdle) or through the app's frame ring
  void releaseVariant(Resources& resources, VkShaderModule module, VkPipeline rtx, VkPipeline rq, nvvk::Buffer sbt, bool gpuIdle);
  std::vector<std::pair<std::string, std::string>> variantMacros(const VariantKey& key) const;
  void createSbt(Resources& resources, VkPipeline pipeline, VkShaderModule module, nvvk::Buffer& sbtBuffer, nvvk::SBTGenerator::Regions& regions);

  // D15: a variant that only adds scene features, built off the frame (see buildsOffTheFrame())
  struct VariantCompileJob
  {
    VariantKey                                       key;
    std::vector<std::pair<std::string, std::string>> macros;  // after applyCommonShaderMacros()'s
    std::vector<std::filesystem::path>               searchPaths;
    std::vector<slang::TargetDesc>                   targets;
    std::vector<slang::CompilerOptionEntry>          options;
    bool                                             rayQuery = false;
    bool                                             useSER   = false;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;  // the path tracer's: a job is discarded before it goes
    VkPipelineCreateFlags rtxFlags = 0, rqFlags = 0;
  };
  struct VariantBuilt
  {
    VkShaderModule module   = VK_NULL_HANDLE;  // none: the build failed
    VkPipeline     pipeline = VK_NULL_HANDLE;  // the job's technique's
    double         ms       = 0.0;
  };
  VariantJobs<VariantCompileJob, VariantBuilt> m_variantJobs;
  std::optional<VariantKey> m_failedVariant;  // a job that failed: that variant is compiled before the frame
  void                      startVariantJob(Resources& resources, const VariantKey& key);
  VariantBuilt              buildVariant(const VariantCompileJob& job);  // on the worker
  void                      installBuiltVariant(Resources& resources);
  // A variant built off the frame waits to be installed, which only a drawn frame does
  bool                      variantBuilt() const { return m_variantJobs.done(); }
  void                      discardVariantJob();

  BusyWindow* m_busyWindow{nullptr};  // Modal shown during async shader/pipeline compile.
  std::mutex  m_compileMutex;         // Guards compile metadata and live pipeline handles.
  std::thread m_compileThread;        // Joined in onDetach().
  std::mutex               m_warmMutex;
  std::vector<std::thread> m_warmThreads;  // startPipelineWarm(); joined before a creation and in onDetach()
  void        joinPipelineWarm();

  // The technique of the frame being rendered: m_techniqueDlss on the DLSS variant's frames,
  // m_techniqueStill on the others (onRender). Each variant keeps its own pipeline in the cache.
  RenderTechnique m_renderTechnique{RenderTechnique::RayQuery};
  // Measured 2026-10-28/29 at 4K on the dressed figure: the pipeline with SER took the moving trace
  // from 31 to 19 ms, and once Sobol read no table (sobol.h.slang) the still's frame from 162 to
  // 62 ms. Ray query was the default from upstream's merge (e33c30b), which made the still about 3x
  // dearer. Each view still has its own setting.
  RenderTechnique m_techniqueStill{RenderTechnique::RayTracing};
  RenderTechnique m_techniqueDlss{RenderTechnique::RayTracing};

  // Adaptive sampling for performance optimization
  void                                  updateAdaptiveSampling(Resources& resources);
  void                                  restartAdaptiveSampling();
  sample_schedule::ThroughputSearch     m_throughputSearch;       // Auto SPP's Throughput target
  sample_schedule::FrameRateControl     m_frameRateControl;       // Auto SPP's frame-rate targets
  std::chrono::steady_clock::time_point m_lastAdaptiveFrameTime;  // When the previous frame was rendered
#if defined(USE_OPTIX_DENOISER)
  // The "Optix denoiser" profiler section's GPU time: the CUDA-measured invoke, per in-flight frame slot
  std::array<double, nvutils::ProfilerTimeline::MAX_FRAME_DELAY> m_optixGpuUs{};
  nvutils::ProfilerTimeline::GpuTimeProvider                     m_optixTimeProvider{
                          .apiName       = "CUDA",
                          .frameFunction = [this](nvutils::ProfilerTimeline::FrameSectionID sec, double& gpuTime) {
        gpuTime = m_optixGpuUs[sec.subFrame];
        return true;
      }};
#endif
  bool m_optixWasOn{false};  // keepOneAccumulationDenoiser(): last frame's state
  bool m_oidnWasOn{false};
  // DLSS-RR rendered frame 0 of this accumulation, and an accumulation denoiser (OptiX, OIDN) takes the
  // frames after it: DLSS's image is no sample of the accumulation, and the hand-off blends from it
  bool                       m_dlssFrame0{false};
  DenoiserHandoff            m_handoff;
  // Whether DLSS renders this frame (dlssFrame(), set at the frame's start), whether it rendered the
  // one before, and whether this is the hand-off frame: the first the accumulation traces after DLSS's.
  // With dlssHandoffDelayMs DLSS renders on past frame 0 until the view has been still that long
  // (input arrives slower than frames render; a pause of a frame would otherwise start an
  // accumulation and a denoise that the next input throws away).
  bool                                  m_dlssFrame{false};
  bool                                  m_prevDlssFrame{false};
  bool                                  m_handoffFrame{false};
  int                                   m_dlssHandoffDelayMs{0};
  std::chrono::steady_clock::time_point m_accumulationStart{};
  // DLSS draws this frame (onRender decides it before the variant is picked: they must agree)
  bool m_dlssShaderFrame = true;
  int                                   m_accumulationFrames{0};  // frames the accumulation traced (not DLSS's)
  std::atomic<uint32_t>      m_pipelineGeneration{0};   // Bumped whenever a new or swapped pipeline is installed
  uint64_t                   m_adaptivePipelineKey{0};  // Pipeline (generation, technique) Auto SPP was found with
  nvutils::ProfilerTimeline* m_profilerTimeline{nullptr};
  bool                       m_adaptiveSampling{true};
  int m_adaptiveSamples{MIN_SAMPLES_PER_PIXEL};  // Auto SPP's samples per frame; ptSamples keeps the user's value
  int m_totalSamplesAccumulated{0};              // Track total samples separately
  int m_frameSamples{1};  // Samples per pixel traced this frame: ptSamples, fewer on a frame that ends on a stop
  // Samples traced by the last frames, as far back as the profiler's GPU times lag (Auto SPP)
  std::array<int, nvutils::ProfilerTimeline::MAX_FRAME_DELAY + 1> m_recentFrameSamples{};

  nvsamples::RollingAverage<float, 100> m_throughputRollingAvg;  // Rolling average of mega-sample-pixels per second (MSPP/s)

  // Adaptive performance targets
  enum class PerformanceTarget
  {
    eInteractive = 0,  // 60 FPS - for real-time interaction
    eBalanced    = 1,  // 30 FPS - good balance of responsiveness and quality
    eQuality     = 2,  // 15 FPS - prioritize quality convergence
    eMaxQuality  = 3,  // 10 FPS - maximum GPU utilization for fastest convergence
    eThroughput  = 4   // Lowest GPU time per sample, frames up to 10 FPS - offline renders (headless default)
  };

  PerformanceTarget    m_performanceTarget{PerformanceTarget::eBalanced};  // Default to balanced for path tracing
  static constexpr int MAX_SAMPLES_PER_PIXEL = 100;  // ptSamples, the samples per frame set by hand
  // Auto SPP's ceiling. Higher than the slider: the controller bounds the frame time, and small or
  // simple images need many samples per frame to keep the GPU busy (headless throughput).
  static constexpr int MAX_ADAPTIVE_SAMPLES_PER_PIXEL = 1024;
  static constexpr int MIN_SAMPLES_PER_PIXEL          = 1;

  double getTargetFrameTimeMs() const
  {
    switch(m_performanceTarget)
    {
      case PerformanceTarget::eInteractive:
        return 1000.0 / 60.0;  // 16.67ms
      case PerformanceTarget::eBalanced:
        return 1000.0 / 30.0;  // 33.33ms
      case PerformanceTarget::eQuality:
        return 1000.0 / 15.0;  // 66.67ms
      case PerformanceTarget::eMaxQuality:
      case PerformanceTarget::eThroughput:  // the longest frame the search may try
        return 1000.0 / 10.0;               // 100ms
      default:
        return 1000.0 / 30.0;
    }
  }

  // One accumulated-image denoiser at a time (OptiX or OIDN): both would denoise the same image, and
  // only one result can be shown. The one turned on last stays on; when both come on together
  // (ImGui.ini, the command line), OIDN does. Called every frame; returns true when the accumulation
  // must restart.
  bool keepOneAccumulationDenoiser(Resources& resources);

  // Samples per frame the next frame traces before it ends on a stop: Auto SPP's count, the
  // setting (ptSamples), or 1 when DLSS renders every frame
  int samplesPerFrame() const
  {
    return isDlssEveryFrame() ? 1 : (m_adaptiveSampling ? m_adaptiveSamples : m_pushConst.numSamples);
  }

  // DLSS-RR renders every frame, one sample each. Not with OptiX or OIDN also on: DLSS renders only
  // frame 0, and the frames they take trace the samples Auto SPP or Samples per Frame ask for.
  bool isDlssEveryFrame() const { return isDlssEnabled() && !isAccumulationDenoiserEnabled(); }
  // DLSS rendered the last frame (every frame, frame 0, or the frames of the hand-off delay)
  bool dlssFrame() const { return m_dlssFrame; }
  // DLSS renders on while the hand-off delay runs: those frames are no part of the accumulation,
  // so the renderer holds its frame count (GltfRenderer::updateFrameCounter) and no sample of
  // theirs is counted (updateStatistics); neither bound can end the still before it begins
  bool dlssHandoffDelayPending() const { return m_dlssFrame && isAccumulationDenoiserEnabled() && inHandoffDelay(); }
  // The view has been still for less than dlssHandoffDelayMs (false when there is no delay)
  bool inHandoffDelay() const
  {
    return m_dlssHandoffDelayMs > 0
           && std::chrono::steady_clock::now() - m_accumulationStart < std::chrono::milliseconds(m_dlssHandoffDelayMs);
  }

  // True when the user has DLSS-RR enabled
  bool isDlssEnabled() const
  {
#if defined(USE_DLSS)
    if(!m_dlss)
      return false;
    const auto s = m_dlss->state();
    return s == Dlss::State::eLoading || s == Dlss::State::eActive;
#else
    return false;
#endif
  }

  // #DLSS - Implementation of the DLSS denoiser (Ray Reconstruction).
#if defined(USE_DLSS)
  std::unique_ptr<Dlss> m_dlss;
  Dlss*                 getDlss() { return m_dlss.get(); }
  const Dlss*           getDlss() const { return m_dlss.get(); }
#endif


  // #OPTIX - Implementation of the OptiX denoiser
#if defined(USE_OPTIX_DENOISER)
  std::unique_ptr<OptiXDenoiser> m_optix;
  OptiXDenoiser*                 getOptiXDenoiser() { return m_optix.get(); }
  const OptiXDenoiser*           getOptiXDenoiser() const { return m_optix.get(); }
#endif

  // #OIDN - Implementation of the vk_oidn denoiser (Open Image Denoise on Vulkan)
#if defined(USE_OIDN_DENOISER)
  std::unique_ptr<OidnDenoiser> m_oidn;
  OidnDenoiser*                 getOidnDenoiser() { return m_oidn.get(); }
  const OidnDenoiser*           getOidnDenoiser() const { return m_oidn.get(); }
#endif


private:
  // VK_KHR_pipeline_executable_properties: log each pipeline build's statistics (register count, spills)
  void logPipelineStatistics(VkPipeline pipeline, const char* label) const;
  bool m_capturePipelineStats{false};

  // What the live shader / pipeline lack for the current settings (see getPendingCompileWork()).
  struct PendingCompileWork
  {
    bool recompile       = false;  // A shader macro changed: needs another shader variant.
    bool stalePipeline   = false;  // Pipelines built with stale specialization constants (SER, DLSS transparency).
    bool missingPipeline = false;  // No pipeline for the selected technique.
    bool any() const { return recompile || stalePipeline || missingPipeline; }
  };
  PendingCompileWork getPendingCompileWork(const Resources& resources);
  bool               buildsOffTheFrame(Resources& resources, const PendingCompileWork& work);  // D15
  bool               prepareFrame(Resources& resources);  // false: skip this frame (async build started)
  void               ensureShadersAndPipelines(Resources& resources);
  void               startAsyncCompile(Resources& resources);
  void               updateStatistics(Resources& resources);
  // Bind the selected technique's pipeline and descriptors once per frame (false: it has no
  // pipeline), then launch it over `size`, once per launch; the caller times each launch.
  bool bindTechnique(VkCommandBuffer cmd, Resources& resources);
  void dispatchRays(VkCommandBuffer cmd, VkExtent2D size) const;
  // Find the medium the camera is in (ePtProbeCameraMedium) on the bound pipeline, then push
  // `pushConst` back for the frame.
  void probeCameraMedium(VkCommandBuffer cmd, const shaderio::PathtracePushConstant& pushConst);
  void denoiseDlss(VkCommandBuffer cmd, Resources& resources);
  // Push this frame's constants; returns what was pushed.
  shaderio::PathtracePushConstant setupPushConstant(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderingSize);
  // Determine if DLSS should actively denoise this frame
  bool getEffectiveDlssEnabled(const Resources& resources) const;
  // A denoiser of the accumulated image (OptiX or OIDN) is on
  bool isAccumulationDenoiserEnabled() const;
  // The accumulation denoiser on this frame, for the DLSS hand-off; nullopt when none is
  std::optional<DenoiserHandoff::Denoiser> handoffDenoiser(const Resources& resources) const;
  // Denoises made so far by OptiX and OIDN together
  uint32_t accumulationDenoiseCount() const;
  // Samples per pixel the DLSS hand-off ends at (its setting, at most Max Samples); 0 without one
  int handoffEnd(const Resources& resources) const;
  // Determine if OptiX should actively denoise this frame
  bool getEffectiveOptixEnabled(const Resources& resources) const;
  // Determine if OIDN should write its guides and denoise this frame
  bool getEffectiveOidnEnabled(const Resources& resources) const;
  // Upscale selection ID and depth from render resolution to display resolution (OptiX 2x mode)
  void upscaleSelectionAndDepth(VkCommandBuffer cmd, Resources& resources);
  // Size m_depthStaging / m_depthCopyBuffer to the G-buffer (no-op unless m_depthViaStaging).
  void updateDepthStaging(VkCommandBuffer cmd, Resources& resources);
  // Size m_convergence to the G-buffer
  void updateConvergence(VkCommandBuffer cmd, Resources& resources);
  // Copy the path-traced depth from m_depthStaging into the G-buffer depth image.
  void cmdCopyStagedDepth(VkCommandBuffer cmd, Resources& resources);
  // Destroy the pipelines for both Ray Query and Ray Tracing
  void destroyPipelinesLocked();
  // True when the scene's BLAS carry opacity micromaps, so the shader must be built with USE_OMM.
  static bool wantOpacityMicromapShader(const Resources& resources);
  // Disk cache of the from-file path-tracer SPIR-V. A hit fills `spirv`.
  bool loadCachedSpirv(uint64_t key, std::vector<uint32_t>& spirv) const;
  // compileStart is when compileFile began; sources written after it are not trusted.
  void storeCachedSpirv(uint64_t key, nvslang::SlangCompiler& compiler, std::filesystem::file_time_type compileStart);
  // Drops old keys, orphaned blobs and stale temporary files from the cache folder.
  void pruneShaderCache() const;

  std::filesystem::path m_cacheDir;        // --cacheDir; empty means next to the executable
  std::filesystem::path m_shaderCacheDir;  // <cache>/shaders, empty when that directory could not be created
  bool m_skipVariantCache{false};  // Set during reloadShader(); bypasses the variant cache and the SPIR-V disk cache.
};
