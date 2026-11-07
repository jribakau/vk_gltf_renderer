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

#include <glm/glm.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/resources.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"
#include "shaders/local_exposure_io.h.slang"

//-------------------------------------------------------------------------------------------------------
// LocalExposure
//
// Shadows lifted and highlights held, region by region, as a photographer's dodge and burn: the
// image's luminance averaged over the regions it falls into, without crossing an edge where it jumps
// (a bilateral grid, shaders/local_exposure.h.slang), and each region brought toward mid grey by a
// curve that leaves the mid-tones alone. A pixel gets its region's gain, so texture and detail keep
// their contrast and no halo forms round a dark figure against a bright wall. `shadows`
// (localExposureShadows, the View tab's Lift shadows) and `highlights` (localExposureHighlights, Hold
// highlights), 0 to 1 each; both 0, the image goes to the tone mapper as it is and nothing is
// allocated.
//
//   localExposure.cmdApply(cmd, resources, input, size, exposure)  // in runTonemapPass, after bloom
//
// It runs on every frame while on, after DLSS, the denoisers, the hand-off and bloom (a lens's glow
// is in the light the sensor takes; this is what is done with the picture), on the SDR and the HDR
// images alike. An .exr export keeps the image without it.
//-------------------------------------------------------------------------------------------------------
class LocalExposure
{
public:
  struct Settings
  {
    float shadows    = 0.0f;  // lift, 0 to 1 (0: none)
    float highlights = 0.0f;  // hold, 0 to 1
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  void registerParameters(SettingsRegistry* settings);
  // The device is idle: what is allocated follows the new size
  void onResize(VkCommandBuffer cmd, VkExtent2D size);

  bool active() const { return m_settings.shadows > 0.0f || m_settings.highlights > 0.0f; }

  // The tone mapper's input with each region's gain on it, sampled linearly; `input` itself when off.
  // `exposure`: the tone mapper's, so the regions' levels are counted from mid grey after it.
  // `timer`: each pass in a section of its own (the grid's, the gain's)
  VkDescriptorImageInfo cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input,
                                 VkExtent2D size, float exposure, nvvk::ProfilerGpuTimer& timer);

  // GPU memory a pixel of the frame costs while on: the output (the grid is a few hundred kB)
  static constexpr uint32_t kBytesPerPixel = 8;

private:
  bool ensureTargets(VkCommandBuffer cmd, VkExtent2D size);
  void releaseTargets();

  Settings m_settings{};

  VkDevice                    m_device           = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*    m_alloc            = nullptr;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvvk::SamplerPool*          m_samplerPool      = nullptr;
  VkSampler                   m_sampler          = VK_NULL_HANDLE;  // linear, clamped to the edge
  VkExtent2D                  m_size{};  // the image's, 0 when nothing is allocated
  int                         m_cell = 0;
  glm::ivec2                  m_cells{0};
  nvvk::Buffer                m_grid;     // the splat: (stops summed, pixels) a cell's bin
  nvvk::Buffer                m_blurred;  // the grid blurred
  nvvk::RenderTarget          m_out;      // the image's size

  nvvk::DescriptorBindings m_bindings;
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipelineLayout         m_pipelineLayout{};
  VkPipeline               m_splatPipeline{};
  VkPipeline               m_blurPipeline{};
  VkPipeline               m_applyPipeline{};
};
