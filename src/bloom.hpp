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

#include <array>

#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"
#include "shaders/bloom_io.h.slang"

//-------------------------------------------------------------------------------------------------------
// Bloom
//
// A lens's glow round bright light, and a star when the aperture has blades, added to the image the
// tone mapper takes (shaders/bloom.slang, the kernels in shaders/bloom.h.slang). The glow is a sum of
// Gaussians that keeps the image's light: `share` of it (bloomShare, the View tab's Bloom) moves from
// each pixel into a long-tailed glow, and of that, `star` (bloomStar) into spikes across the aperture's
// straight edges: as many as its blades when even, twice as many when odd, none for a round one. Off
// (share 0), the image goes to the tone mapper as it is and nothing is allocated.
//
//   bloom.cmdApply(cmd, resources, input, size, exposure)  // in runTonemapPass, the tone mapper's input
//
// It runs on every frame while on, the moving view's too, after DLSS, the denoisers and the hand-off,
// on the SDR and the HDR images alike. An .exr export keeps the image without it.
//-------------------------------------------------------------------------------------------------------
class Bloom
{
public:
  struct Settings
  {
    float share = 0.0f;  // of the image's light, into the glow (0: off)
    float star  = 0.0f;  // of the glow, into the star (only with aperture blades)
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  void registerParameters(SettingsRegistry* settings);
  // The device is idle: allocated images follow the new size
  void onResize(VkCommandBuffer cmd, VkExtent2D size);

  // The lens's blades (below 3 round) and its first corner's angle, radians (setDepthOfField's)
  void setAperture(int blades, float rotation)
  {
    m_blades   = blades;
    m_rotation = rotation;
  }

  bool active() const { return m_settings.share > 0.0f; }

  // The tone mapper's input with the glow and the star in it, sampled linearly; `input` itself when off.
  // `exposure`: the tone mapper's, so the light the glow takes in is held in units of its white.
  VkDescriptorImageInfo cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input,
                                 VkExtent2D size, float exposure);

  // GPU memory a pixel of the frame costs while on: the output, and the levels (a third of it each)
  static constexpr uint32_t kBytesPerPixel = 8 + 6;

private:
  enum LevelImage
  {
    eLevelDown,  // the level: the finer one blurred and halved
    eLevelUp,    // its glow: the level and the coarser glow mixed
    eLevelImageCount
  };

  void initTargets();
  bool ensureTargets(VkCommandBuffer cmd, VkExtent2D size);
  void releaseTargets();
  void dispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkExtent2D outSize, const shaderio::BloomPushConstant& push,
                const VkDescriptorImageInfo& a, const VkDescriptorImageInfo& b, const VkDescriptorImageInfo& c,
                const VkDescriptorImageInfo& out);
  VkDescriptorImageInfo sampled(const nvvk::RenderTarget& target, uint32_t index) const;

  Settings m_settings{};
  int      m_blades   = 0;
  float    m_rotation = 0.0f;

  VkDevice                        m_device = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*        m_alloc  = nullptr;
  nvvkgltf::GpuMemoryTracker*     m_appMemoryTracker = nullptr;
  nvvk::SamplerPool*              m_samplerPool      = nullptr;
  VkSampler                       m_sampler          = VK_NULL_HANDLE;  // linear, clamped to the edge
  VkExtent2D                      m_size{};                             // the image's, 0 when nothing is allocated
  std::array<nvvk::RenderTarget, 8> m_levels;                           // 1 to bloomLevels() (BLOOM_MAX_LEVELS): half, quarter, ...
  int                             m_levelCount = 0;                     // those allocated
  nvvk::RenderTarget              m_star;                               // on BLOOM_STAR_LEVEL's size
  nvvk::RenderTarget              m_out;                                // the image's size

  nvvk::DescriptorBindings m_bindings;
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipelineLayout         m_pipelineLayout{};
  VkPipeline               m_downPipeline{};
  VkPipeline               m_upPipeline{};
  VkPipeline               m_starPipeline{};
  VkPipeline               m_compositePipeline{};
};
