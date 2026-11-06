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

#include "shaders/sky_clouds_io.h.slang"

struct Resources;

// The cloud image's size. Columns run round the horizon (0.09 degrees each), rows from the horizon
// to the zenith, crowded at the horizon (sky_clouds.h.slang). What the camera sees of the clouds is
// read from it, so this is their sharpness on screen. A drag's preview bakes a quarter of each.
static constexpr VkExtent2D kCloudImageSize{4096, 1024};
static constexpr VkExtent2D kCloudPreviewSize{1024, 256};

// The clouds as the panel sets them.
struct SkyCloudsSettings
{
  float    coverage{0.0F};      // 0 clear .. 1 overcast: the share of the land they shade
  float    thicknessKm{1.45F};  // the type's depth, base to the highest tops (cloudDefaultThickness)
  uint32_t seed{0};             // which arrangement of the clouds
  int      type{0};             // CLOUD_TYPE_* (sky_clouds.h.slang)

  bool operator==(const SkyCloudsSettings&) const = default;
};

// Where the weather map and the shape are read for an arrangement: a stretch of their endless
// tiling far from the last one's.
glm::vec2 cloudSeedOffsetKm(uint32_t seed);

// The share of the distant ground the clouds leave sunlit, for the bake (cloudSunlitAir); 1 clear.
float cloudSunlitGround(const SkyCloudsSettings& settings);

// The bake's parameter block for these settings: takram's layers and medium, the arrangement's
// offsets, at the settled quality or a drag's.
shaderio::SkyCloudsParameters cloudBakeParameters(const SkyCloudsSettings& settings, bool preview);

//--------------------------------------------------------------------------------------------------
// SkyClouds -- the physical sky's cloud layers.
//
// Owns the noise the clouds are cut from (generated once, at init), the cloud image (one bake of the
// layers as the fixed observer sees them, for this sun), a quarter-size preview image for drags, and
// the pass that bakes them. EnvBaker records that pass ahead of its own, which composes the image
// over the sky into the lighting environment; the path tracer composes it over what the camera sees.
// Both read it through this class's runtime set, and SkySunDisk::cloudPreview says which image holds
// the clouds of the moment. The model is takram's (shaders/sky_clouds*.slang).
//
// With no clouds the pass is not run: the full image is cleared once to (0, 0, 0, 1), which composes
// to the sky exactly.
//--------------------------------------------------------------------------------------------------
class SkyClouds
{
public:
  SkyClouds() = default;
  ~SkyClouds() { assert(!m_initialized && "deinit must be called first"); }

  SkyClouds(const SkyClouds&)            = delete;
  SkyClouds& operator=(const SkyClouds&) = delete;

  // The noise module has one entry point per texture (sky_clouds_noise.slang); the bake one.
  struct Shaders
  {
    std::span<const uint32_t> noise;
    std::span<const uint32_t> bake;
  };

  // `brunetonSetLayout` / `brunetonSet` are SkyBruneton's runtime set: the bake lights the clouds
  // through its transmittance and irradiance tables. Submits and waits once, for the noise.
  void init(Resources& res, const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout, VkDescriptorSet brunetonSet);
  void deinit(Resources& res);

  void                     setSettings(const SkyCloudsSettings& settings) { m_settings = settings; }
  const SkyCloudsSettings& settings() const { return m_settings; }
  bool                     hasClouds() const { return m_settings.coverage > 0.0F; }

  // Bake the clouds for this sun (toward it, the environment's frame) and observer, into the
  // caller's command buffer, behind a barrier for frames still reading the last bake and ahead of
  // one for the passes that read this one. `preview` bakes the quarter-size image at low quality.
  void record(VkCommandBuffer cmd, const glm::vec3& sunDirection, float observerAltitudeKm, bool preview);

  // Whether the preview image holds the clouds of the moment (the last record was a preview).
  bool showingPreview() const { return m_showingPreview; }

  // The images, as combined image samplers at CLOUD_IMAGE_BINDING and CLOUD_PREVIEW_BINDING.
  VkDescriptorSetLayout getDescriptorSetLayout() const { return m_runtimePack.getLayout(); }
  VkDescriptorSet       getDescriptorSet() const { return m_runtimePack.getSet(0); }

private:
  void createImages(Resources& res);
  void createPipelines(const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout);
  void generateNoise(Resources& res);

  bool              m_initialized{false};
  VkDevice          m_device{VK_NULL_HANDLE};
  SkyCloudsSettings m_settings{};
  bool              m_imageClear{false};      // the full image holds (0, 0, 0, 1), and no bake has run since
  bool              m_showingPreview{false};  // the last bake went to the preview image
  uint32_t          m_imageMips{1};
  uint32_t          m_weatherMips{1};
  uint32_t          m_shapeMips{1};
  uint32_t          m_detailMips{1};

  nvvk::Image m_shapeNoise;       // 3D R8, repeating, mipmapped
  nvvk::Image m_detailNoise;      // 3D R8, repeating, mipmapped
  nvvk::Image m_weatherNoise;     // 2D RGBA8, repeating, mipmapped
  nvvk::Image m_turbulenceNoise;  // 2D RGBA8, repeating
  nvvk::Image m_cloudImage;       // 2D RGBA16F, azimuth repeating, mipmapped; GENERAL for life
  nvvk::Image m_previewImage;     // the same, a quarter the size, one level
  // Storage writes need a view of one level; the images' own views span their chains for sampling.
  VkImageView m_shapeStorageView{VK_NULL_HANDLE};
  VkImageView m_detailStorageView{VK_NULL_HANDLE};
  VkImageView m_weatherStorageView{VK_NULL_HANDLE};
  VkImageView m_cloudStorageView{VK_NULL_HANDLE};

  nvvk::Buffer m_paramsBuf;  // device-local SkyCloudsParameters, updated in the command buffer per bake

  VkDescriptorSet      m_brunetonSet{VK_NULL_HANDLE};  // SkyBruneton's, borrowed
  nvvk::DescriptorPack m_noisePack;                    // the noise textures, as storage, for their one write
  nvvk::DescriptorPack m_bakePack;                     // two sets: set 0 writes the full image, set 1 the preview
  nvvk::DescriptorPack m_runtimePack;                  // the images, for their readers

  VkPipelineLayout m_noiseLayout{VK_NULL_HANDLE};
  VkPipeline       m_shapePipeline{VK_NULL_HANDLE};
  VkPipeline       m_detailPipeline{VK_NULL_HANDLE};
  VkPipeline       m_weatherPipeline{VK_NULL_HANDLE};
  VkPipeline       m_turbulencePipeline{VK_NULL_HANDLE};
  VkPipelineLayout m_bakeLayout{VK_NULL_HANDLE};
  VkPipeline       m_bakePipeline{VK_NULL_HANDLE};
  VkCommandPool    m_transientCmdPool{VK_NULL_HANDLE};
};
