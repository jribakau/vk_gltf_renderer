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
// DLSS Ray Reconstruction over a still's image as it refines and its final one (see rr_still_denoiser.hpp).
//

#include "rr_still_denoiser.hpp"

#if defined(USE_DLSS)

#include <algorithm>
#include <cmath>
#include <limits>

#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "sample_schedule.hpp"
#include "../shaders/adaptive_stop.h.slang"  // adaptiveSceneThreshold

// Name of the GPU timer section around the evaluation (Profiler window)
static constexpr const char* kProfilerSection = "DLSS-RR still";
static constexpr const char* kMemoryCategory  = "DLSS-RR still/GBuffers";

void RrStillDenoiser::init(Resources& resources)
{
  m_device           = resources.allocator.getDevice();
  m_allocator        = &resources.allocator;
  m_appMemoryTracker = &resources.appMemoryTracker;
  m_blend.init(m_device);
  m_keep.init(m_device);
}

void RrStillDenoiser::deinit(Resources& resources)
{
  release(resources);
  m_blend.deinit(m_device);
  m_keep.deinit(m_device);
  m_ngx = nullptr;
}

void RrStillDenoiser::registerParameters(SettingsRegistry* settings)
{
  settings->add({"rrStillEnable",
                 "DLSS-RR still: denoise the still with DLSS Ray Reconstruction, every rrStillAutoDenoiseSpp samples per pixel "
                 "and its final image"},
                &m_settings.enable, Persist::eYes);
  settings->add({"rrStillBlendMax", "DLSS-RR still: share of the raw accumulation mixed back into its output (0 = all RR's)"},
                &m_settings.blendMax, Persist::eYes, 0.0f, 1.0f);
  settings->add({"rrStillPasses", "DLSS-RR still: evaluations of an image, the first with the history reset"},
                &m_settings.passes, Persist::eYes, 1, 16);
  settings->add({"rrStillAutoDenoiseSpp",
                 "DLSS-RR still: samples per pixel between its evaluations as the still refines (0 = the final image only)"},
                &m_settings.autoDenoiseSpp, Persist::eYes, 0, 1000);
  settings->add({"rrStillPreset",
                 "DLSS-RR still: its network preset, numbered as dlssPreset: 0=Default (NGX chooses), 1=D (transformer), "
                 "2=E (latest transformer), 3=F (RR2). E by default: F smooths away detail the still has resolved"},
                &m_settings.preset, Persist::eYes, 0u, rr_presets::kCount - 1);
  settings->add({"rrStillKeepNoise",
                 "DLSS-RR still: Keep detail per pixel, the noise in displayed brightness below which a pixel of the "
                 "accumulation is shown as it is, over the denoised image (the adaptive stop's measure; 0 = off)"},
                &m_settings.keepNoise, Persist::eYes, 0.0f, 1.0f);
  settings->add({"rrStillKeepView", "DLSS-RR still: show the share of each pixel Keep detail per pixel keeps, as grey"},
                &m_settings.keepView, Persist::eNo);
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
  const uint32_t   preset  = rr_presets::hint(m_settings.preset);
  const VkExtent2D current = m_targetInit ? m_target.getSize() : VkExtent2D{};
  const bool       resize  = current.width != size.width || current.height != size.height;
  const bool recreate = !m_featureCreated || m_featureSize.width != size.width || m_featureSize.height != size.height || m_featurePreset != preset;
  if(!resize && !recreate)
    return false;
  if(!resize && m_failed && m_featurePreset == preset)
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
      .preset                = preset,
      .inputSize             = size,
      .outputSize            = size,
  };
  const NVSDK_NGX_Result result = m_feature.cmdInit(cmd, *m_ngx, info);
  m_featureSize                 = size;
  m_featurePreset               = preset;
  if(NVSDK_NGX_FAILED(result))
  {
    LOGE("DLSS-RR still: feature create failed: %d (%ux%u); the still stays undenoised until the size or preset changes\n",
         result, size.width, size.height);
    m_failed = true;
  }
  else
  {
    m_featureCreated = true;
    LOGI("DLSS-RR still: feature at %ux%u (DLAA, preset hint %u)\n", size.width, size.height, preset);
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

void RrStillDenoiser::update(VkCommandBuffer         cmd,
                             Resources&              resources,
                             nvvk::ProfilerGpuTimer* profiler,
                             int                     samples,
                             int                     handoffEnd,
                             const glm::mat4&        view,
                             const glm::mat4&        proj)
{
  // guides the path tracer has not written for this accumulation would be garbage
  if(!canEvaluate(samples) || m_triedSamples == samples)
    return;
  const bool finalImage = samples >= resources.settings.maxSamples;
  if(!m_schedule.isDue(samples, m_settings.autoDenoiseSpp, finalImage, handoffEnd, resources.settings.maxSamples))
    return;
  evaluate(cmd, resources, profiler, samples, finalImage, view, proj);
}

void RrStillDenoiser::denoiseFinal(VkCommandBuffer         cmd,
                                   Resources&              resources,
                                   nvvk::ProfilerGpuTimer* profiler,
                                   int                     samples,
                                   const glm::mat4&        view,
                                   const glm::mat4&        proj)
{
  // once an image, a failed evaluation too (not every idle frame)
  if(!canEvaluate(samples) || m_triedSamples == samples)
    return;
  evaluate(cmd, resources, profiler, samples, /*finalImage=*/true, view, proj);
}

void RrStillDenoiser::evaluate(VkCommandBuffer         cmd,
                               Resources&              resources,
                               nvvk::ProfilerGpuTimer* profiler,
                               int                     samples,
                               bool                    finalImage,
                               const glm::mat4&        view,
                               const glm::mat4&        proj)
{
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
  if(ok && m_settings.keepNoise > 0.0f && m_moments.imageView != VK_NULL_HANDLE)
  {
    // Keep detail per pixel: the converged pixels of the accumulation over RR's image. The threshold in
    // the scene's units for the exposure the host sets, as the adaptive stop's (auto exposure: none)
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    const shaderio::StillKeepPushConstant keep{
        .threshold = resources.tonemapperData.autoExposure ?
                         m_settings.keepNoise :
                         adaptiveSceneThreshold(m_settings.keepNoise, resources.tonemapperData.exposure),
        .samples   = samples,
        .view      = m_settings.keepView ? 1 : 0,
    };
    m_keep.cmdKeep(cmd, m_target.getColorStorageImageInfo(eGBufferDenoised),
                   resources.gBuffers.getColorStorageImageInfo(Resources::eImgRendered), m_moments, m_target.getSize(), keep);
  }
  if(ok)
  {
    // Keep detail: the raw accumulation mixed back in (RGBA32F: no half-float clamp)
    const float blend = sample_schedule::denoiseBlend(m_settings.blendMax, samples, resources.settings.maxSamples, finalImage);
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
  m_schedule.markDenoised(samples);  // a failed one is not retried every frame
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
