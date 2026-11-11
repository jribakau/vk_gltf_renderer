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
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device
#include "shaders/pixel_time.h.slang"  // The Time view's slots

#include <nvvk/sbt_generator.hpp>
#include <nvutils/profiler.hpp>
#include "denoiser_handoff.hpp"
#include "invalid_samples.hpp"
#include "radiance_cache.hpp"
#include "renderer_base.hpp"
#include "pipeline_build_record.hpp"
#include "sample_schedule.hpp"
#include "selection_write.hpp"
#include "utils.hpp"
#include "pipeline_cache_util.hpp"
#include "pixel_time.hpp"
#include "scene_feature_detection.hpp"
#include "busy_state.hpp"
#include "variant_compile_job.hpp"

// #DLSS
#if defined(USE_DLSS)
#include "dlss.hpp"
#endif

// DLSS-RR over the still as it refines and its final image
#include "rr_still_denoiser.hpp"


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
  void setProfilerTimeline(nvutils::ProfilerTimeline* timeline) { m_profilerTimeline = timeline; }
  void onDetach(Resources& resources) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources) override;
  void onRender(VkCommandBuffer cmd, Resources& resources) override;
  // Called instead of onRender() while the accumulation is complete
  void onAccumulationComplete(VkCommandBuffer cmd, Resources& resources);
  // The DLSS hand-off's image, to show instead of the denoiser's view (null when not shown)
  const VkDescriptorImageInfo* handoffImage(const Resources& resources) const;
  void                         onSceneInvalidated(Resources& resources) override;
  void                         notifyDlssContentReset(Resources& resources) override;

  void updateDlssResources(VkCommandBuffer cmd, Resources& resources);
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
  // The part of the G-buffer's depth image, from its corner, that holds the view's depth: the size of
  // the frame that last started it (a DLSS frame's is DLSS's input size), the whole of `image` before any
  VkExtent2D depthExtent(VkExtent2D image) const
  {
    const bool none = m_depthExtent.width == 0 || m_depthExtent.height == 0;
    return none || m_depthExtent.width > image.width || m_depthExtent.height > image.height ? image : m_depthExtent;
  }

  // Whether what first hits want is counted (countHits): memory is near full
  void setCountHits(bool on) { m_countHits = on; }
  // The Time view (shaders/pixel_time.h.slang), for the frame info: where this frame sums its pixels'
  // time and how many slots it has (none while the view is off), and the scale its image is coloured by.
  // Before the frame info is written; takes what an earlier frame summed into this frame's slot.
  struct PixelTimeFrame
  {
    uint32_t* sums  = nullptr;
    int       slots = 0;
    float     scale = 0.0F;
  };
  PixelTimeFrame preparePixelTime(Resources& resources);
  // Where this frame counts what it throws away as not finite (shaders/invalid_samples.h.slang), for the
  // frame info. Before the frame info is written; adds what an earlier frame counted into this frame's
  // slot to the accumulation's tally, and logs a tally that has any once its accumulation is over.
  uint32_t* prepareInvalidSamples(Resources& resources);
  // The moving view's bounce cache (RadianceCache): where this frame finds its table, null without one.
  // Before the frame info is written.
  RadianceCache::Frame prepareRadianceCache(Resources& resources) { return m_radianceCache.prepare(resources, isDlssEnabled()); }
  // The latest frame's sums once it has come back, once
  std::optional<pixel_time::Reading> takePixelTimeReading() { return std::exchange(m_timeReading, std::nullopt); }
  // The view is on and the device has the clock
  static bool pixelTimeOn(const Resources& resources)
  {
    return resources.settings.pixelTime && resources.settings.shaderClockAvailable;
  }
  // The Samples view (SampleMap): the convergence image, while the last frame traced was a still's with
  // the adaptive stop on, so each pixel holds the samples it took (adaptiveStateValue); none otherwise
  // (a DLSS frame binds its responsivity there, an exact still or the Time view write no counts)
  struct SampleCounts
  {
    VkDescriptorImageInfo image{};
    int                   minSamples = 0;  // ptAdaptiveMin
  };
  std::optional<SampleCounts> sampleCounts() const
  {
    if(!m_sampleCountsLive)
      return std::nullopt;
    return SampleCounts{.image = m_convergence.getColorStorageImageInfo(0), .minSamples = m_adaptiveMin};
  }
  std::optional<nvvkgltf::SceneFeatureSet> liveShaderFeatures();
  // A pipeline built before compiled again this run: the driver's shader cache is full (PipelineBuildRecord)
  bool driverCacheFull() const { return m_buildRecord.driverDropped(); }
  // User-initiated hot reload: drop cached variants and recompile from Slang source.
  bool reloadShader(Resources& resources);
  void setBusyState(BusyState* busy) { m_busy = busy; }
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
  /// Declare this subsystem's settings (command line + MCP + persistence).
  /// resetAccumulation restarts the frame count: settings that change what a frame renders call it
  /// when they are set.
  void registerParameters(SettingsRegistry* settings, std::function<void()> resetAccumulation);

  VkDevice                        m_device{};  // Vulkan device
  VkPipelineLayout                m_pipelineLayout{};
  VkPipeline                      m_rtxPipeline{};    // Ray tracing pipeline
  VkPipeline                      m_rqPipeline{};     // Ray tracing pipeline
  shaderio::PathtracePushConstant m_pushConst{};      // Information sent to the shader
  bool                            m_autoFocus{true};  // Enable auto-focus
  VkShaderModule                  m_shaderModule{};   // Shader module for RTX

  nvvk::PipelineCacheManager m_pipelineCache{};  // Pipeline cache for faster creation
  // The pipelines built before (pipeline_builds.txt beside the cache), to tell a full driver cache
  // from a new shader; each build is noted by notePipelineBuild()
  PipelineBuildRecord m_buildRecord;
  uint32_t            m_driverVersion{0};  // in the build keys: a driver update empties its cache
  // The code each live shader module was made from, by its hash, for the build keys
  std::mutex                                     m_moduleCodeMutex;
  std::unordered_map<VkShaderModule, uint64_t>   m_moduleCode;
  void     rememberModuleCode(VkShaderModule module, const void* code, size_t bytes);
  uint64_t moduleCode(VkShaderModule module);
  // A pipeline from code of hash `code`, built with this state in `ms`, in the record; a warning when
  // the driver had dropped it since it was last built
  void notePipelineBuild(uint64_t code, bool rayQuery, int32_t useSER, int32_t transparency, int32_t inlineShadow,
                         VkPipelineCreateFlags flags, double ms);

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
  nvvk::Buffer       m_pixelInspectBuffer;   // shaderio::PixelInspect, host-mapped: the pixel readout (cmdInspectPixel)
  // ptCameraMedium: whether to probe it. The probe passes through surfaces that bound no medium, so
  // through an opening (an eye socket, a mouth) it meets a figure's skin from inside and every path
  // starts in the skin; a host whose camera is never inside a medium turns it off.
  bool m_probeCameraMedium = true;
  // scatterAsThin: 0 the volume walk, 1 the thin-walled preview, 2 the preview on the frames DLSS
  // renders (every frame while the camera moves) and the walk on the frames an accumulation traces,
  // so a host need not switch it (and reset the accumulation) when the view stops
  int m_scatterAsThin = 0;
  bool               m_useSER{true};         // Requested SER state; clamped to m_supportSER each frame.
  // ptInlineShadow: the RTX pipeline's shadow rays as inline ray queries in raygen (USE_INLINE_SHADOW)
  // rather than through rahitShadow: 15-18% off a moving frame's trace and 12-16% off a still's on
  // 1TEST, 7TEST and 10TEST (2026-11-08). Read when a pipeline is created, so set on the command line.
  bool m_inlineShadow{true};
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
  // Dwivedi guiding of scatter walks entered from a surface (ePtWalkGuiding, walk_guide.h.slang)
  bool m_walkGuiding{true};
  // Measured skin's map lit head-on, not averaged over directions (ePtSkinFaceOn, skin_albedo.h.slang)
  bool m_skinFaceOn{false};
  // The emitters' light tree in the light pick (ePtLightTree, light_tree.h.slang)
  bool m_lightTree{true};
  // A still's light draws pick as they score where the light tree is on (ePtStillPickAsScored); off,
  // by the walk (measured 2026-11-07: the walk's spread of the Sobol numbers is worth less than it costs)
  bool m_stillPickAsScored{true};
  // A vertex whose every lobe is a mirror leaves the lights its BSDF ray reaches to that ray, no shadow
  // ray (ePtSmoothBsdfOnly, smooth_vertex.h.slang); off, it samples them as any vertex
  bool m_smoothBsdfOnly{true};
  // Surface bounces (PathtracePushConstant::maxDepth): the still's, and the DLSS frames' when lower (0: the
  // still's), set each frame by sample_schedule::frameMaxDepth
  int m_maxDepth{5};
  int m_maxDepthDlss{0};
  // Fibre events before they count as surface bounces (PathtracePushConstant::hairBudget): the still's,
  // and the DLSS frames', fewer, since white hair keeps its energy through every one.
  int m_hairBounces{16};
  int m_hairBouncesDlss{4};
  // Light samples a surface vertex draws to keep one of (PathtracePushConstant::lightCandidates): the
  // still's, and the DLSS frames'. 1: the one sample it traces, as before
  int m_lightCandidates{1};
  int m_lightCandidatesDlss{1};
  // The first hit's light sample reused from frame to frame (ptLightReuse, ptLightReuseDlss,
  // ptLightReuseCap; updateLightReuse): the records, and what the last frame wrote into them
  bool         m_lightReuse{false};
  bool         m_lightReuseDlss{false};
  float        m_lightReuseCap{1.0F};  // longer histories sparkle and boil under Ray Reconstruction (ptLightReuseCap)
  nvvk::Buffer m_lightReservoirs;
  VkDeviceSize m_lightReservoirsSize{0};
  bool         m_lightReuseParity{false};
  bool         m_lightReuseLastOn{false};
  bool         m_lightReuseLastDlss{false};
  VkExtent2D   m_lightReuseLastSize{0, 0};
  // DLSS frames: past ptHairRouletteFrom fibre events a path continues at most with ptHairRouletteDlss
  float m_hairRouletteDlss{1.0F};
  int   m_hairRouletteFrom{1};
  // DLSS frames: hair paths end at their fourth fibre event, whose light sample carries the rest of the
  // lock's light (ptHairCutDlss, ptHairCutTail). Measured 2026-11-09 at 1080p against the converged
  // still: a sunlit figure's shadowed hair 12% dark cut at the third event, 6% at the fourth, 3% uncut,
  // for no trace time measured between the third and the fourth; the cut saves about 1 ms of a moving
  // trace where long hair fills the view (it saved 5 ms on 2026-10-29, before the strand split)
  int   m_hairCutDlss{4};
  float m_hairCutTail{4.0F};
  // DLSS frames: DLSS-RR's responsivity mask on hair (ptHairResponsivity; 0: no mask). Negative trusts
  // the history more: test4 held on DLSS, 12 frames, the hair's per-pixel std 2.32 -> 1.45 (top of the
  // head) and 2.19 -> 1.35 (long strands) at -1, -0.5 halfway, +1 worse (2.96); skin 1.04 -> 0.98
  float m_hairResponsivity{-1.0F};
  // An accumulation's pixel stops once its mean has converged (ptAdaptiveStop, ptAdaptiveThreshold,
  // ptAdaptiveMin; shaders/adaptive_stop.h.slang). 0.09: test4's still 12.7 -> 9.0 s with its skin's
  // pores as the full still's; at 0.11 (7.5 s) and 0.13 (6.5 s) the denoiser of the time (OptiX) smoothed
  // them away (2026-10-29)
  bool  m_adaptiveStop{true};
  bool  m_sampleCountsLive{false};  // the last frame traced wrote its pixels' sample counts (sampleCounts)
  float m_adaptiveThreshold{0.09F};
  int   m_adaptiveMin{32};
  // Highlights the tone mapper flattens keep more noise (ptAdaptiveRelax, adaptive_relax.hpp): the
  // brightest pixels of a still, lamps and lit walls, otherwise sampled to the cap
  bool m_adaptiveRelax{true};
  // The still ends once at most this share of its pixels still sample (ptAdaptiveDone; 0: only at the
  // cap). Counted by the GPU each frame (adaptiveActive), read back once the frame is done.
  float        m_adaptiveDone{0.001F};
  nvvk::Buffer m_adaptiveActiveBuffer;  // a uint per frame in flight, host-visible
  bool                      m_countHits{false};
  int                       m_hitFrames{0};         // this rest's frames that counted, -1 once they are copied; a move starts again
  nvvk::Buffer              m_hitCounts;            // on the device: every frame's first hits add up here
  std::vector<nvvk::Buffer> m_hitReadback;          // per frame in flight, host-visible
  std::vector<uint8_t>      m_hitCopied;            // per frame in flight: that frame copied a count there
  // The Time view's sums (preparePixelTime): on the device, cleared and summed by each frame; copied back
  // per frame in flight, read when the frame cycle reaches that slot again
  nvvk::Buffer                       m_timeSums;
  std::vector<nvvk::Buffer>          m_timeReadback;
  std::vector<uint32_t>              m_timeCopied;  // per frame in flight: the slots that frame copied there (0: none)
  std::vector<int>                   m_timeCounted;  // ...and what it summed (Settings::pixelTimeCount)
  int                                m_timeSlots{0};  // this frame's (0: the view is off)
  float                              m_timeScale{0.0F};  // the last reading's, kept while the view is off
  int                                m_timeScaleCounted{0};  // what that reading summed: its scale is for that alone
  std::optional<pixel_time::Reading> m_timeReading;
  struct AdaptiveCount
  {
    uint64_t accumulation{0};  // the accumulation (m_accumulationId) that counted into the slot
    bool     counted{false};
  };
  std::vector<AdaptiveCount> m_adaptiveCounts;  // per frame in flight: what the slot holds
  // What frames threw away as not finite (prepareInvalidSamples): a block per frame in flight,
  // host-visible, the slots' state as the adaptive count's, and the accumulation's tally
  nvvk::Buffer               m_invalidBuffer;
  std::vector<AdaptiveCount> m_invalidSlots;
  invalid_samples::Tally     m_invalidTally;
  uint64_t                   m_invalidTallyOf{0};  // the accumulation m_invalidTally sums
  uint64_t                   m_accumulationId{0};
  SelectionWrite m_selectionWrite;  // when a frame traces the selection image's object IDs
  // Whether the selection image's object IDs are not the picture's (no frame since the view last
  // changed traced them): nothing can be outlined from it
  bool selectionImageStale() const { return m_selectionWrite.stale; }
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
    bool dlssGuide = false;  // True when guide-buffer capture is compiled in (USE_GUIDE_SHADER: DLSS-RR, moving or still).
    nvvkgltf::SceneFeatureSet features{};  // only meaningful when `optimal == true`
    int dlssTransparency = 0;  // USE_DLSS_TRANSP the pipeline was specialized with (Dlss::TransparencyMode).
    // True when the light samples' resampling and reuse are compiled in (USE_LIGHT_RIS: ptLightCandidates,
    // ptLightReuse). Last, as makeVariantKey() fills the key in order.
    bool lightRis = false;
    // True when each pixel is timed (USE_PIXEL_TIME, the Time view: pixelTimeOn)
    bool pixelTime = false;
    // True when the moving view's bounce cache is compiled in (USE_RADIANCE_CACHE: ptRadianceCache), for
    // the frames DLSS draws
    bool radianceCache = false;
    // True when the path keeps the medium around the one it is in (USE_NESTED_MEDIA): the scene draws
    // a clear solid (SceneFeatureSet::eNestedMedia), optimal shader or not
    bool nestedMedia = false;

    // Same SPIR-V: every shader macro matches. dlssTransparency is a pipeline specialization
    // constant, not a macro, so it is left out. dlss and dlssGuide are compared in every mode (they
    // drive USE_DLSS_SHADER / USE_GUIDE_SHADER independently of optimal); the full extension feature
    // set only matters for the optimal build.
    bool sameShader(const VariantKey& o) const
    {
      return wireframe == o.wireframe && visualize == o.visualize && omm == o.omm && optimal == o.optimal
             && dlss == o.dlss && dlssGuide == o.dlssGuide && lightRis == o.lightRis && pixelTime == o.pixelTime
             && radianceCache == o.radianceCache && nestedMedia == o.nestedMedia
             && (optimal ? (features == o.features) : true);
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
  // Swaps in the wanted variant if it is cached (cheap; no BusyState). Leaves live handles alone on a miss.
  bool tryRestoreCachedVariant(Resources& resources);
  // Destroys a variant's handles now (gpuIdle) or through the app's frame ring
  void releaseVariant(Resources& resources, VkShaderModule module, VkPipeline rtx, VkPipeline rq, nvvk::Buffer sbt, bool gpuIdle);
  std::vector<std::pair<std::string, std::string>> variantMacros(const VariantKey& key) const;
  void createSbt(Resources& resources, VkPipeline pipeline, VkShaderModule module, nvvk::Buffer& sbtBuffer, nvvk::SBTGenerator::Regions& regions);

  // D15: a variant that only adds scene features, built off the frame (see buildsOffTheFrame())
  struct VariantCompileJob
  {
    VariantKey                                       key;
    std::vector<std::pair<std::string, std::string>> macros;
    std::vector<std::filesystem::path>               searchPaths;
    std::vector<slang::TargetDesc>                   targets;
    std::vector<slang::CompilerOptionEntry>          options;
    std::string                                      profile;       // by name, as the compiler keeps them
    std::vector<std::string>                         capabilities;  // until it compiles
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
  // A variant built off the frame waits to be taken in: by the frame that draws next (prepareFrame),
  // or, when none does (the still is complete), by takeBuiltVariant alone
  bool                      variantBuilt() const { return m_variantJobs.done(); }
  bool                      takeBuiltVariant(Resources& resources);
  void                      discardVariantJob();

  BusyState* m_busy{nullptr};  // Says what the async shader and pipeline compiles are doing
  std::mutex  m_compileMutex;         // Guards compile metadata and live pipeline handles.
  std::thread m_compileThread;        // Joined in onDetach().
  std::mutex               m_warmMutex;
  std::vector<std::thread> m_warmThreads;  // startPipelineWarm(); joined before a creation and in onDetach()
  void        joinPipelineWarm();
  // The DLSS transparency mode the pipelines are specialized on; 0 without DLSS
  int32_t transparencyMode() const;

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
  void                                  updateDlssSamples(const Resources& resources);
  int                                   dlssFrameSamples() const;
  void                                  restartAdaptiveSampling();
  sample_schedule::ThroughputSearch     m_throughputSearch;       // Auto SPP's Throughput target
  sample_schedule::FrameRateControl     m_frameRateControl;       // Auto SPP's frame-rate targets
  std::chrono::steady_clock::time_point m_lastAdaptiveFrameTime;  // When the previous frame was rendered
  bool m_rrStillWasOn{false};  // DLSS-RR's still on last frame (pollStillDenoiser)
  // DLSS-RR rendered frame 0 of this accumulation, and the still's denoiser takes the frames after it:
  // DLSS's image is no sample of the accumulation, and the hand-off blends from it
  bool                       m_dlssFrame0{false};
  DenoiserHandoff            m_handoff;
  RadianceCache              m_radianceCache;  // the moving view's bounce cache (ptRadianceCache)
  // Whether DLSS renders this frame (dlssFrame(), set at the frame's start), whether it rendered the
  // one before, and whether this is the hand-off frame: the first the accumulation traces after DLSS's.
  // With dlssHandoffDelayMs DLSS renders on past frame 0 until the view has been still that long
  // (input arrives slower than frames render; a pause of a frame would otherwise start an
  // accumulation and a denoise that the next input throws away).
  bool                                  m_dlssFrame{false};
  bool                                  m_prevDlssFrame{false};
  bool                                  m_handoffFrame{false};
  VkExtent2D m_depthExtent{};  // the size of the frame that last started the depth image (frameStartsDepth); none: not yet
  int                                   m_dlssHandoffDelayMs{0};
  bool                                  m_stillWaits{false};  // setStillWaits
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
  int m_samplesDlss{1};   // Samples per pixel a DLSS frame may trace (ptSamplesDlss; dlssFrameSamples)
  std::chrono::steady_clock::time_point m_dlssViewChanged{};  // The last reset (a frame of count 0)
  double m_dlssSampleMs{0.0};  // A sample's path-trace GPU time, from frames that all traced one count; 0: none yet
  // The DLSS refine's frames: its samples once the view has held this long, as many as fit this much path trace
  static constexpr double kDlssHoldMs        = 250.0;
  static constexpr double kDlssRefineTraceMs = 33.0;
  int m_samplesPerFrame{1};  // ptSamples: samples per pixel a frame traces, a pass of the kernel each
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

  // DLSS-RR's still came on or went off (its setting, or NGX coming up, which an idle still's frames
  // take here since they do not tick DLSS). Called every frame; returns true when the accumulation must
  // restart: its guides cover the still only when it began with them, and the shader variant changes.
  bool pollStillDenoiser();

  // A headless run's still is not done yet: DLSS-RR's still is its denoiser and NGX is still starting
  // (the still begins again once it is up), or its final pass has not run
  bool stillDenoisePending(const Resources& resources) const;

  // Samples per frame the next frame traces before it ends on a stop: Auto SPP's count, the
  // setting (ptSamples), or when DLSS renders every frame dlssFrameSamples()
  int samplesPerFrame() const
  {
    return isDlssEveryFrame() ? dlssFrameSamples() : (m_adaptiveSampling ? m_adaptiveSamples : m_samplesPerFrame);
  }

  // DLSS-RR renders every frame, one sample each. Not with the still's denoiser on: DLSS renders only
  // frame 0, and the frames it takes trace the samples Auto SPP or Samples per Frame ask for.
  bool isDlssEveryFrame() const { return isDlssEnabled() && !rrStillOn(); }
  // Whether the path tracer's variant for DLSS's frames, or the still's, has the light candidates and
  // reuse compiled in (VariantKey::lightRis): only for the frames whose settings take them
  bool lightRisFor(bool dlssFrame) const
  {
    return dlssFrame ? (m_lightCandidatesDlss > 1 || m_lightReuseDlss) : (m_lightCandidates > 1 || m_lightReuse);
  }
  // Whether the variant for DLSS's frames, or the still's, has the bounce cache compiled in
  // (VariantKey::radianceCache): DLSS's alone while the setting is on (the still's too for the debug
  // setting that measures the cache)
  bool radianceCacheFor(bool dlssFrame) const { return m_radianceCache.reads(dlssFrame); }
  // DLSS rendered the last frame (every frame, frame 0, or the frames of the hand-off delay)
  bool dlssFrame() const { return m_dlssFrame; }
  // DLSS renders on while the hand-off delay runs: those frames are no part of the accumulation,
  // so the renderer holds its frame count (GltfRenderer::updateFrameCounter) and no sample of
  // theirs is counted (updateStatistics); neither bound can end the still before it begins
  bool dlssHandoffDelayPending() const { return m_dlssFrame && rrStillOn() && inHandoffDelay(); }
  // The view has been still for less than dlssHandoffDelayMs (false when there is no delay), or the
  // still waits for textures still arriving (setStillWaits)
  bool inHandoffDelay() const
  {
    return m_stillWaits
           || (m_dlssHandoffDelayMs > 0
               && std::chrono::steady_clock::now() - m_accumulationStart < std::chrono::milliseconds(m_dlssHandoffDelayMs));
  }
  // Textures are still arriving: the hand-off waits for them, DLSS drawing the view, so the still
  // begins once instead of starting again as each lands. Needs DLSS and the still's denoiser,
  // as the delay does; without them the still restarts as it did.
  void setStillWaits(bool on) { m_stillWaits = on; }

  // DLSS-RR's still is the still's denoiser: its frames gather RR's guides on the DLSS shader variant
  bool rrStillOn() const;

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
  // DLSS-RR as the still's denoiser: its final image, once (rrStillEnable)
  std::unique_ptr<RrStillDenoiser> m_rrStill;
  const RrStillDenoiser*           getRrStillDenoiser() const { return m_rrStill.get(); }
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
  void               updateLightReuse(Resources& resources, VkExtent2D size, bool useDlss);
  void               updateStatistics(Resources& resources);
  // Bind the selected technique's pipeline and descriptors once per frame (false: it has no
  // pipeline), then launch it over `size`, once per launch; the caller times each launch.
  bool bindTechnique(VkCommandBuffer cmd, Resources& resources);
  void cmdTraceFrameSamples(VkCommandBuffer cmd, const shaderio::PathtracePushConstant& first, VkExtent2D size) const;
  void dispatchRays(VkCommandBuffer cmd, VkExtent2D size) const;
  // Find the medium the camera is in (ePtProbeCameraMedium) on the bound pipeline, then push
  // `pushConst` back for the frame.
  void probeCameraMedium(VkCommandBuffer cmd, const shaderio::PathtracePushConstant& pushConst);

public:
  // The pixel readout (view.inspect): record the camera ray through the render pixel at `uv` (a share
  // of the view, origin top-left) on the technique's pipeline (ePtInspectPixel), with the last traced
  // frame's constants, after whatever this frame drew; its PixelInspect is the host's once the frame is
  // done (inspectResult). One at a time: the caller waits for that frame before recording another.
  // False with no pipeline (one being rebuilt) or no frame traced yet.
  bool                   cmdInspectPixel(VkCommandBuffer cmd, Resources& resources, glm::vec2 uv);
  shaderio::PixelInspect inspectResult() const;

private:
  // `selectionWritten`: the frame traced the selection image (ePtWriteSelection), which then goes to full size
  void denoiseDlss(VkCommandBuffer cmd, Resources& resources, bool selectionWritten);
  // Push this frame's constants; returns what was pushed.
  shaderio::PathtracePushConstant setupPushConstant(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderingSize);
  // Determine if DLSS should actively denoise this frame
  bool getEffectiveDlssEnabled(const Resources& resources) const;
  // The still's denoiser on this frame, for the DLSS hand-off; nullopt when it is not
  std::optional<DenoiserHandoff::Denoiser> handoffDenoiser(const Resources& resources) const;
  // Denoises the still's denoiser has made so far
  uint32_t accumulationDenoiseCount() const;
  // Samples per pixel the DLSS hand-off ends at (its setting, at most Max Samples); 0 without one
  int handoffEnd(const Resources& resources) const;
  // DLSS-RR's still: on, and this an accumulation's frame (frame 0 and the hand-off delay's being
  // DLSS's while it is engaged)
  bool getEffectiveRrStillEnabled(const Resources& resources) const;
  // Its setting on, NGX up or not
  bool rrStillWanted() const;
  // The frame runs the DLSS shader variant: DLSS renders it, or DLSS-RR's still gathers its guides
  bool dlssVariantFrame() const;
  // Size m_depthStaging / m_depthCopyBuffer to the G-buffer (no-op unless m_depthViaStaging).
  void updateDepthStaging(VkCommandBuffer cmd, Resources& resources);
  // Size m_convergence to the G-buffer
  void updateConvergence(VkCommandBuffer cmd, Resources& resources);
  // The adaptive stop's count of the pixels still sampling: reads the one this frame's slot holds
  // (that frame is done), may end the still (Resources::adaptiveDone), and clears it for this frame.
  // Returns the counter for the push constant.
  uint32_t* countAdaptivePixels(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderingSize);
  // This frame's invalid-sample block zeroed before it traces, and the slot marked as counted (prepareInvalidSamples)
  void armInvalidSamples(VkCommandBuffer cmd, Resources& resources);
  void logInvalidSamples();
  // The hit signal: where this frame counts what first hits want (hit_signal.h.slang), null when it doesn't.
  // A few frames of each rest, from when memory is near full (setCountHits): the frame that counts nothing
  // pays one branch a pixel, and a scene that fits never counts.
  uint32_t* countHits(VkCommandBuffer cmd, Resources& resources);
  // What a frame copied has come back when the frame cycle reaches its slot again
  void takeHitCounts(Resources& resources);
  void copyHitCounts(VkCommandBuffer cmd, Resources& resources);
  // The Time view's: what came back to this frame's slot, and this frame's sums copied there
  void takePixelTimes(Resources& resources);
  void copyPixelTimes(VkCommandBuffer cmd, Resources& resources);
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
