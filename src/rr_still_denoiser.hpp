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
// DLSS Ray Reconstruction as the denoiser of a still: the accumulation refines undenoised, and once it
// is complete its final image goes through DLSS-RR once, at native size (DLAA), its history reset.
// The guides are RR's own (the DLSS shader variant's: diffuse and specular albedo, linear roughness,
// and through clear glass the surface behind it), averaged over the still's samples like the colour
// (ePtRrStillGuides, into this class's images), so their edges are anti-aliased as the colour's are.
// Motion is zero: nothing moved. The moving view's DLSS-RR feature is left alone: this one is its own
// (the shared NGX context, Dlss::ngxContext()).
//
// Usage, as OidnDenoiser's:
//   denoiser.init(resources);                        // once
//   denoiser.setContext(dlss.ngxContext(), preset);  // every frame: null until NGX is up with RR
//   denoiser.ensureTargets(resources);               // every frame: images and feature follow the settings
//   denoiser.denoiseFinal(cmd, resources, ...);      // the frame the accumulation completes, and idle frames
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
    int   passes   = 1;     // Evaluations of the final image, the first with the history reset
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  void registerParameters(SettingsRegistry* settings);

  // The NGX context with DLSS-RR available, null without; `preset`: the moving view's RR preset
  void setContext(NgxContext* ngx, uint32_t preset);

  // The setting, whether NGX is up or not
  bool isWanted() const { return m_settings.enable; }
  // On, and NGX can run it
  bool isEnabled() const { return m_settings.enable && m_ngx != nullptr; }
  // The final image of `samples` per pixel is still to be evaluated, and can be: the feature made, the
  // guides the path tracer's for this accumulation
  bool finalPending(int samples) const
  {
    return isEnabled() && m_featureCreated && m_guides.isCurrent() && samples > 0 && m_triedSamples != samples;
  }
  bool hasValidDenoisedOutput() const { return m_hasValidOutput; }
  int  denoisedSamples() const { return m_denoisedSamples; }
  uint32_t denoiseCount() const { return m_denoiseCount; }

  VkDescriptorImageInfo getDescriptorImageInfo(GBufferIndex index) const;

  // As OidnDenoiser's: whether the path tracer writes the guides this frame; the samples already in them
  int updateGuideState(bool writtenThisFrame, int totalSamples) { return m_guides.update(writtenThisFrame, totalSamples); }

  // Images and feature follow `enable`, the size and the preset; released when off. Waits for the queue
  // when it (re)allocates. Returns true when it did.
  bool ensureTargets(Resources& resources);

  // Denoises the accumulation's final image of `samples` per pixel, once: the frame that completes it
  // (`complete`), or an idle frame after. view and proj are the camera's, unjittered.
  void denoiseFinal(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler, int samples, bool complete,
                    const glm::mat4& view, const glm::mat4& proj);

  // A new accumulation: the output is of another image
  void forgetOutput()
  {
    m_hasValidOutput = false;
    m_triedSamples   = -1;
  }

  void disable(Resources& resources);

private:
  void release(Resources& resources);
  void bindResources(Resources& resources);

  Settings          m_settings{};
  NgxContext*       m_ngx    = nullptr;
  uint32_t          m_preset = 0;
  DlssFeature       m_feature{DlssFeature::Kind::RR};
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

  VkDevice                    m_device           = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*    m_allocator        = nullptr;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvvk::RenderTarget          m_target;  // See GBufferIndex
  ImageBlend                  m_blend;   // Keep detail
};

#endif  // USE_DLSS
