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

//
// DLSS Ray Reconstruction over a still's final image (see rr_still_denoiser.hpp).
//

#include "rr_still_denoiser.hpp"

#if defined(USE_DLSS)

#include <algorithm>
#include <limits>

#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "sample_schedule.hpp"

// Name of the GPU timer section around the evaluation (Profiler window)
static constexpr const char* kProfilerSection = "DLSS-RR still";
static constexpr const char* kMemoryCategory  = "DLSS-RR still/GBuffers";

void RrStillDenoiser::init(Resources& resources)
{
  m_device           = resources.allocator.getDevice();
  m_allocator        = &resources.allocator;
  m_appMemoryTracker = &resources.appMemoryTracker;
  m_blend.init(m_device);
}

void RrStillDenoiser::deinit(Resources& resources)
{
  release(resources);
  m_blend.deinit(m_device);
  m_ngx = nullptr;
}

void RrStillDenoiser::registerParameters(SettingsRegistry* settings)
{
  settings->add({"rrStillEnable",
                 "DLSS-RR still: denoise the still's final image with DLSS Ray Reconstruction, once the accumulation is complete "
                 "(it refines undenoised); OptiX and OIDN off"},
                &m_settings.enable, Persist::eYes);
  settings->add({"rrStillBlendMax", "DLSS-RR still: share of the raw accumulation mixed back into its output (0 = all RR's)"},
                &m_settings.blendMax, Persist::eYes, 0.0f, 1.0f);
  settings->add({"rrStillPasses", "DLSS-RR still: evaluations of the final image, the first with the history reset"},
                &m_settings.passes, Persist::eYes, 1, 16);
}

void RrStillDenoiser::setContext(NgxContext* ngx, uint32_t preset)
{
  m_ngx    = ngx;
  m_preset = preset;
}

VkDescriptorImageInfo RrStillDenoiser::getDescriptorImageInfo(GBufferIndex index) const
{
  return m_target.getColorStorageImageInfo(index);
}

void RrStillDenoiser::release(Resources& resources)
{
  if(!m_targetInit && !m_featureCreated)
    return;
  NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
  if(m_featureCreated)
    m_feature.deinit();
  m_featureCreated = false;
  m_featureSize    = {};
  if(m_targetInit)
  {
    if(m_appMemoryTracker)
      m_appMemoryTracker->untrack(kMemoryCategory, m_target, eGBufferCount);
    m_target.deinit();
  }
  m_targetInit     = false;
  m_hasValidOutput = false;
  m_guides.restart();
}

bool RrStillDenoiser::ensureTargets(Resources& resources)
{
  if(!isEnabled())
  {
    // off, or NGX gone: nothing kept (about 100 MB of images a megapixel, and the feature's own)
    release(resources);
    m_failed = false;
    return false;
  }
  const VkExtent2D size = resources.gBuffers.getSize();
  if(size.width == 0 || size.height == 0)
    return false;
  const VkExtent2D current = m_targetInit ? m_target.getSize() : VkExtent2D{};
  const bool       resize  = current.width != size.width || current.height != size.height;
  const bool recreate = !m_featureCreated || m_featureSize.width != size.width || m_featureSize.height != size.height || m_featurePreset != m_preset;
  if(!resize && !recreate)
    return false;
  if(!resize && m_failed && m_featurePreset == m_preset)
    return false;

  // SYNC NOTE: rare (enabled, resized, preset changed); the old images and feature may be in flight
  NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
  VkCommandBuffer cmd = resources.app->createTempCmdBuffer();
  if(!m_targetInit)
  {
    NVVK_CHECK(m_target.init({.device       = m_device,
                              .alloc        = m_allocator,
                              .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT,
                                               VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16_SFLOAT},
                              .debugName    = "DLSS-RR-still"}));
    m_targetInit = true;
  }
  if(resize)
  {
    if(m_appMemoryTracker && current.width != 0)
      m_appMemoryTracker->untrack(kMemoryCategory, m_target, eGBufferCount);
    NVVK_CHECK(m_target.update(cmd, size));
    m_target.cmdClear(cmd);  // the motion stays zero
    if(m_appMemoryTracker)
      m_appMemoryTracker->track(kMemoryCategory, m_target, eGBufferCount);
    m_guides.restart();
    m_hasValidOutput = false;
    m_failed         = false;
  }
  if(m_featureCreated)
    m_feature.deinit();
  m_featureCreated = false;
  // DLAA: the input is the accumulation, at the output's size
  const DlssFeature::InitInfo info{
      .packedNormalRoughness = true,
      .hardwareDepth         = true,
      .quality               = NVSDK_NGX_PerfQuality_Value_DLAA,
      .preset                = m_preset,
      .inputSize             = size,
      .outputSize            = size,
  };
  const NVSDK_NGX_Result result = m_feature.cmdInit(cmd, *m_ngx, info);
  m_featureSize                 = size;
  m_featurePreset               = m_preset;
  if(NVSDK_NGX_FAILED(result))
  {
    LOGE("DLSS-RR still: feature create failed: %d (%ux%u); the still stays undenoised until the size or preset changes\n",
         result, size.width, size.height);
    m_failed = true;
  }
  else
  {
    m_featureCreated = true;
    LOGI("DLSS-RR still: feature at %ux%u (DLAA)\n", size.width, size.height);
  }
  resources.app->submitAndWaitTempCmdBuffer(cmd);
  return true;
}

void RrStillDenoiser::bindResources(Resources& resources)
{
  using Type     = DlssFeature::ResourceType;
  const auto own = [&](Type type, GBufferIndex index) {
    m_feature.setResource({type, m_target.getColorImage(index), m_target.getColorAttachmentView(index), m_target.getColorFormat(index)});
  };
  // the G-buffer's image may have been reallocated since the last evaluation
  m_feature.setResource({Type::eColorIn, resources.gBuffers.getColorImage(Resources::eImgRendered),
                         resources.gBuffers.getColorAttachmentView(Resources::eImgRendered),
                         resources.gBuffers.getColorFormat(Resources::eImgRendered)});
  own(Type::eColorOut, eGBufferDenoised);
  own(Type::eDiffuseAlbedo, eGBufferAlbedo);
  own(Type::eSpecularAlbedo, eGBufferSpecAlbedo);
  own(Type::eNormalRoughness, eGBufferNormal);
  own(Type::eMotionVector, eGBufferMotion);
  own(Type::eDepth, eGBufferDepth);
}

void RrStillDenoiser::denoiseFinal(VkCommandBuffer       cmd,
                                   Resources&            resources,
                                   nvvk::ProfilerGpuTimer* profiler,
                                   int                   samples,
                                   bool                  complete,
                                   const glm::mat4&      view,
                                   const glm::mat4&      proj)
{
  if(!complete || !isEnabled() || !m_featureCreated || samples <= 0)
    return;
  // once an image, a failed evaluation too (not every idle frame); guides the path tracer has not
  // written for this accumulation would be garbage
  if(m_triedSamples == samples)
    return;
  if(!m_guides.isCurrent())
    return;

  NVVK_DBG_SCOPE(cmd);
  const nvutils::ProfilerTimeline::FrameSectionID section = profiler->cmdFrameBeginSection(cmd, kProfilerSection);

  // The colour and guides the path tracer wrote, before NGX reads them; its previous output no longer read
  nvvk::cmdMemoryBarrier(cmd,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  bindResources(resources);
  bool ok = true;
  for(int pass = 0; pass < std::max(m_settings.passes, 1) && ok; pass++)
  {
    if(pass > 0)
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    ok = NVSDK_NGX_SUCCEED(m_feature.cmdEvaluate(cmd, *m_ngx, {glm::vec2(0.0f), view, proj, /*reset=*/pass == 0, /*responsivity=*/false}));
  }
  if(ok)
  {
    // Keep detail: the raw accumulation mixed back in, as OIDN's (RGBA32F: no half-float clamp)
    const float blend = sample_schedule::denoiseBlend(m_settings.blendMax, samples, resources.settings.maxSamples, /*finalImage=*/true);
    if(blend > 0.0f)
    {
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
      const VkDescriptorImageInfo denoised = m_target.getColorStorageImageInfo(eGBufferDenoised);
      m_blend.cmdBlend(cmd, denoised, resources.gBuffers.getColorStorageImageInfo(Resources::eImgRendered), denoised,
                       m_target.getSize(), blend, std::numeric_limits<float>::max());
    }
  }
  // The output, before the tonemapper and the preview thumbnail read it
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  profiler->cmdFrameEndSection(cmd, section);

  m_triedSamples    = samples;
  m_denoisedSamples = samples;
  m_hasValidOutput  = ok;
  if(!ok)
  {
    LOGE("DLSS-RR still: evaluate failed at %d spp; the still shows the accumulation\n", samples);
    return;
  }
  ++m_denoiseCount;
  resources.settings.displayBuffer = DisplayBuffer::eRrStillDenoised;
}

void RrStillDenoiser::disable(Resources& resources)
{
  m_settings.enable = false;
  if(resources.settings.displayBuffer == DisplayBuffer::eRrStillDenoised)
    resources.settings.displayBuffer = DisplayBuffer::eRendered;
}

#endif  // USE_DLSS
