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

#include <cassert>
#include <cstdint>
#include <span>

#include <glm/glm.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/resources.hpp>

struct Resources;

// The cloud image's size. Columns run round the horizon (0.18 degrees each), rows from the horizon
// to the zenith, crowded at the horizon (sky_clouds.h.slang). What the camera sees of the clouds is
// read from it, so this is their sharpness on screen.
static constexpr VkExtent2D kCloudImageSize{2048, 512};

// One cloud layer, as the panel sets it.
struct SkyCloudsParameters
{
  float    coverage{0.0F};     // 0 clear .. 1 overcast
  float    thicknessKm{1.2F};  // from the base to the highest tops
  uint32_t seed{0};            // which arrangement of the clouds

  bool operator==(const SkyCloudsParameters&) const = default;
};

// Where the noise is read for an arrangement: a stretch of the endless tiling far from the last.
glm::vec2 cloudSeedOffsetKm(uint32_t seed);

// The share of the distant ground the clouds leave sunlit, for the bake (cloudSunlitAir); 1 clear.
float cloudSunlitGround(const SkyCloudsParameters& parameters);

//--------------------------------------------------------------------------------------------------
// SkyClouds -- the physical sky's cloud layer.
//
// Owns the noise the clouds are cut from (generated once, at init), the cloud image (one bake of the
// layer as the fixed observer sees it, for this sun) and the pass that bakes it. EnvBaker records
// that pass ahead of its own, which composes the image over the sky into the lighting environment;
// the path tracer composes it over what the camera sees. Both read it through this class's runtime
// set. What the image holds is shaders/sky_clouds.h.slang.
//
// With no clouds the pass is not run: the image is cleared once to (0, 0, 0, 1), which composes to
// the sky exactly.
//--------------------------------------------------------------------------------------------------
class SkyClouds
{
public:
  SkyClouds() = default;
  ~SkyClouds() { assert(!m_initialized && "deinit must be called first"); }

  SkyClouds(const SkyClouds&)            = delete;
  SkyClouds& operator=(const SkyClouds&) = delete;

  // The noise module has one entry point per volume (sky_clouds_noise.slang); the bake one.
  struct Shaders
  {
    std::span<const uint32_t> noise;
    std::span<const uint32_t> bake;
  };

  // `brunetonSetLayout` / `brunetonSet` are SkyBruneton's runtime set: the bake lights the clouds
  // through its transmittance and irradiance tables. Submits and waits once, for the noise.
  void init(Resources& res, const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout, VkDescriptorSet brunetonSet);
  void deinit(Resources& res);

  void                       setParameters(const SkyCloudsParameters& parameters) { m_parameters = parameters; }
  const SkyCloudsParameters& parameters() const { return m_parameters; }
  bool                       hasClouds() const { return m_parameters.coverage > 0.0F; }

  // Bake the cloud image for this sun (toward it, the environment's frame) and observer, into the
  // caller's command buffer, behind a barrier for frames still reading the last one and ahead of one
  // for the passes that read it. `preview` takes fewer samples, for a drag.
  void record(VkCommandBuffer cmd, const glm::vec3& sunDirection, float observerAltitudeKm, bool preview);

  // The cloud image, as a combined image sampler at CLOUD_IMAGE_BINDING.
  VkDescriptorSetLayout getDescriptorSetLayout() const { return m_runtimePack.getLayout(); }
  VkDescriptorSet       getDescriptorSet() const { return m_runtimePack.getSet(0); }

private:
  void createImages(Resources& res);
  void createPipelines(const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout);
  void generateNoise(Resources& res);

  bool                m_initialized{false};
  VkDevice            m_device{VK_NULL_HANDLE};
  SkyCloudsParameters m_parameters{};
  bool                m_imageClear{false};  // the image holds (0, 0, 0, 1), and the bake has not run since

  nvvk::Image m_shapeNoise;    // 3D R8, repeating
  nvvk::Image m_detailNoise;   // 3D R8, repeating
  nvvk::Image m_weatherNoise;  // 2D R8, repeating
  nvvk::Image m_cloudImage;    // 2D RGBA16F, azimuth repeating; GENERAL for life

  VkDescriptorSet      m_brunetonSet{VK_NULL_HANDLE};  // SkyBruneton's, borrowed
  nvvk::DescriptorPack m_noisePack;                    // the volumes, as storage, for their one write
  nvvk::DescriptorPack m_bakePack;                     // the image out, the volumes in
  nvvk::DescriptorPack m_runtimePack;                  // the image, for its readers

  VkPipelineLayout m_noiseLayout{VK_NULL_HANDLE};
  VkPipeline       m_shapePipeline{VK_NULL_HANDLE};
  VkPipeline       m_detailPipeline{VK_NULL_HANDLE};
  VkPipeline       m_weatherPipeline{VK_NULL_HANDLE};
  VkPipelineLayout m_bakeLayout{VK_NULL_HANDLE};
  VkPipeline       m_bakePipeline{VK_NULL_HANDLE};
  VkCommandPool    m_transientCmdPool{VK_NULL_HANDLE};
};
