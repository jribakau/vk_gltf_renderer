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
#include <nvvk/render_target.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"
#include "shaders/lens_io.h.slang"

//-------------------------------------------------------------------------------------------------------
// Lens
//
// A cheap wide lens's defects on the image the tone mapper takes (shaders/lens.slang, the mapping in
// shaders/lens.h.slang): `bend` (lensBend), barrel distortion that draws the image in toward its
// corners, and `fringe` (lensFringe), its colours split toward the corners. The app's Footage looks
// set them (fv_lighting.cpp grade). Both 0, the image goes on as it is and nothing is allocated.
//
//   lens.cmdApply(cmd, resources, input, size, straight)  // in runTonemapPass, before bloom
//
// It runs on every frame while on, the moving view's too, after DLSS, the denoisers and the hand-off
// and before bloom (a glow round a light is round in the picture), on the SDR and the HDR images
// alike. An .exr export keeps the image without it.
//
// The path tracer, the pickers and the overlays all work in the image as rendered. So a frame that
// draws an overlay (the selection's outline, the grid, the gizmo) is not bent (`straight`): they
// would sit beside what they mark. A frame that is bent says so (`bent()`), and `rendered(uv)` is
// where a point of the picture lies in the image as rendered, for a click.
//-------------------------------------------------------------------------------------------------------
class Lens
{
public:
  struct Settings
  {
    float bend   = 0.0f;  // barrel distortion, 0 none to 1
    float fringe = 0.0f;  // colour fringing, 0 none to 1
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  void registerParameters(SettingsRegistry* settings);
  // The device is idle: an allocated image follows the new size
  void onResize(VkCommandBuffer cmd, VkExtent2D size);

  bool active() const { return m_settings.bend > 0.0f || m_settings.fringe > 0.0f; }

  // The tone mapper's input through the lens, sampled linearly; `input` itself when off. `straight`:
  // without the bend (the fringing stays: it moves an edge by a few pixels at most).
  VkDescriptorImageInfo cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input,
                                 VkExtent2D size, bool straight);
  // A frame that went to the tone mapper without the pass (a debug view, off)
  void skipped() { m_bent = 0.0f; }

  // Whether the last frame drawn was bent, and where a point of its picture (uv) lies in the image as
  // rendered: `uv` itself when it was not
  bool      bent() const { return m_bent > 0.0f; }
  glm::vec2 rendered(glm::vec2 uv) const;

  // GPU memory a pixel of the frame costs while on: the output
  static constexpr uint32_t kBytesPerPixel = 8;

private:
  bool ensureTarget(VkCommandBuffer cmd, VkExtent2D size);
  void releaseTarget();

  Settings m_settings{};
  float    m_bent   = 0.0f;  // the last frame's bend, 0 when it had none
  float    m_aspect = 1.0f;  // that frame's width over its height

  VkDevice                    m_device           = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*    m_alloc            = nullptr;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvvk::SamplerPool*          m_samplerPool      = nullptr;
  VkSampler                   m_sampler          = VK_NULL_HANDLE;  // linear, clamped to the edge
  VkExtent2D                  m_size{};                             // the image's, 0 when nothing is allocated
  nvvk::RenderTarget          m_out;                                // the image's size

  nvvk::DescriptorBindings m_bindings;
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipelineLayout         m_pipelineLayout{};
  VkPipeline               m_pipeline{};
};
