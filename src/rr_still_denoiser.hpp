/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "settings_registry.hpp"

//-------------------------------------------------------------------------------------------------------
// RrStillDenoiser
//
// DLSS Ray Reconstruction as the denoiser of a still: the accumulation's image goes through DLSS-RR at
// native size (DLAA), its history reset, every `autoDenoiseSpp` samples per pixel as it refines (on
// AutoDenoiseSchedule, the DLSS hand-off included) and once it is complete; with 0, only the final image.
// The guides are RR's own (the DLSS shader variant's: diffuse and specular albedo, linear roughness,
// and through clear glass the surface behind it), averaged over the still's samples like the colour
// (ePtRrStillGuides, into this class's images), so their edges are anti-aliased as the colour's are.
// Motion is zero: nothing moved. The moving view's DLSS-RR feature is left alone: this one is its own
// (the shared NGX context, Dlss::ngxContext()), with its own network preset (rrStillPreset).
//
// Usage:
//   denoiser.init(resources);                        // once
//   denoiser.setContext(dlss.ngxContext());          // every frame: null until NGX is up with RR
//   denoiser.ensureTargets(resources);               // every frame: images and feature follow the settings
//   denoiser.update(cmd, resources, ...);            // every accumulation frame: a checkpoint or the final image
//   denoiser.denoiseFinal(cmd, resources, ...);      // idle frames: the final image, if not yet
//   denoiser.deinit(resources);                      // before the NGX context goes
//
// The images (5 RGBA32F/R32F and a motion image) and the feature exist only while it is enabled.
//-------------------------------------------------------------------------------------------------------

#if defined(USE_DLSS)

#include <glm/glm.hpp>

#include <nvutils/profiler.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/render_target.hpp>

#include "denoiser_handoff.hpp"
#include "denoiser_schedule.hpp"
#include "dlss_wrapper.hpp"
#include "gpu_memory_tracker.hpp"
#include "resources.hpp"
#include "rr_presets.hpp"
#include "still_keep.hpp"

class RrStillDenoiser
{
public:
  enum GBufferIndex
  {
    eGBufferDenoised   = 0,  // RR's output (RGBA32F)
    eGBufferAlbedo     = 1,  // Accumulated diffuse albedo guide (RGBA32F; shaderio eDenoiserAlbedo)
    eGBufferNormal     = 2,  // Accumulated world-space normal, linear roughness in w (RGBA32F; eDenoiserNormal)
    eGBufferSpecAlbedo = 3,  // Accumulated specular albedo (RGBA32F; eRrStillSpecAlbedo)
    eGBufferDepth      = 4,  // Nearest guide depth (R32F; eRrStillDepth)
    eGBufferMotion     = 5,  // Zero motion (R16G16F), cleared and never written
    eGBufferCount
  };

  struct Settings
  {
    bool  enable   = false;
    float blendMax = 0.0f;  // Keep detail: the raw accumulation mixed back in (sample_schedule::denoiseBlend)
    int   passes   = 1;     // Evaluations of an image, the first with the history reset
    int   autoDenoiseSpp = 0;  // Samples per pixel between the refine's evaluations (0 = the final image only)
    uint32_t preset = rr_presets::kStillDefault;  // rrStillPreset, numbered as dlssPreset (rr_presets.hpp)
    // rrStillKeepNoise: the noise a converged pixel keeps its raw value at, in displayed brightness
    // (StillKeep, still_keep.h.slang); 0 = off
    float keepNoise = 0.02f;
    bool  keepView  = false;  // rrStillKeepView: show the share kept as grey, for tuning (not saved)
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  void registerParameters(SettingsRegistry* settings);

  // The NGX context with DLSS-RR available, null without
  void setContext(NgxContext* ngx) { m_ngx = ngx; }
  // The accumulation's luminance moments (OutputImage::eConvergence), which the path tracer keeps while
  // this is on: the per-pixel Keep detail reads them
  void setMoments(const VkDescriptorImageInfo& moments) { m_moments = moments; }

  // The setting, whether NGX is up or not
  bool isWanted() const { return m_settings.enable; }
  // rrStillPreset, as set: a change is a new still (the feature is made again with it)
  uint32_t presetSetting() const { return m_settings.preset; }
  // On, and NGX can run it
  bool isEnabled() const { return m_settings.enable && m_ngx != nullptr; }
  // The final image of `samples` per pixel is still to be evaluated, and can be: the feature made, the
  // guides the path tracer's for this accumulation
  bool finalPending(int samples) const
  {
    return isEnabled() && m_featureCreated && m_guides.isCurrent() && samples > 0 && m_triedSamples != samples;
  }
  bool hasValidDenoisedOutput() const { return m_hasValidOutput; }
  bool isAutoDenoiseActive() const { return m_settings.autoDenoiseSpp > 0; }
  int  denoisedSamples() const { return m_denoisedSamples; }
  uint32_t denoiseCount() const { return m_denoiseCount; }

  VkDescriptorImageInfo getDescriptorImageInfo(GBufferIndex index) const;

  // Whether the path tracer writes the guides this frame; the samples already in them
  int updateGuideState(bool writtenThisFrame, int totalSamples) { return m_guides.update(writtenThisFrame, totalSamples); }

  // Images and feature follow `enable`, the size and the preset; released when off. Waits for the queue
  // when it (re)allocates. Returns true when it did.
  bool ensureTargets(Resources& resources);

  // An accumulation frame whose image has `samples` per pixel: evaluated at the refine's checkpoints and
  // on the frame that completes it (samples >= the cap). `handoffEnd` is DLSS's hand-off's (0: none).
  // view and proj are the camera's, unjittered.
  void update(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler, int samples, int handoffEnd,
              const glm::mat4& view, const glm::mat4& proj);
  // An idle frame, the accumulation complete (the adaptive stop may have ended it before the cap): its
  // final image of `samples`, once
  void denoiseFinal(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler, int samples,
                    const glm::mat4& view, const glm::mat4& proj);

  // A new accumulation: the output is of another image
  void forgetOutput()
  {
    m_hasValidOutput = false;
    m_triedSamples   = -1;
    m_schedule.forgetDenoised();
  }

  void disable(Resources& resources);

private:
  void release(Resources& resources);
  void bindResources(Resources& resources);
  bool canEvaluate(int samples) const { return isEnabled() && m_featureCreated && samples > 0 && m_guides.isCurrent(); }
  void evaluate(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler, int samples, bool finalImage,
                const glm::mat4& view, const glm::mat4& proj);

  Settings          m_settings{};
  NgxContext*       m_ngx    = nullptr;
  DlssFeature      m_feature{DlssFeature::Kind::RR};
  bool              m_featureCreated = false;
  VkExtent2D        m_featureSize{};
  uint32_t          m_featurePreset = 0;
  bool              m_targetInit    = false;
  bool              m_failed        = false;  // a feature create failed for this size and preset: not retried every frame
  bool              m_hasValidOutput  = false;
  int               m_denoisedSamples = 0;
  int               m_triedSamples    = -1;  // the image last evaluated (or tried), by its samples per pixel
  uint32_t          m_denoiseCount    = 0;
  GuideAccumulation m_guides;
  AutoDenoiseSchedule m_schedule;  // the refine's checkpoints

  VkDevice                    m_device           = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*    m_allocator        = nullptr;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvvk::RenderTarget          m_target;  // See GBufferIndex
  ImageBlend                  m_blend;   // Keep detail
  StillKeep                   m_keep;    // Keep detail per pixel
  VkDescriptorImageInfo       m_moments{};
};

#endif  // USE_DLSS
