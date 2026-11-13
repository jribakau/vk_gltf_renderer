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
#include "sky_cloud_view.hpp"

struct Resources;

// The cloud image's size. Columns run round the horizon (0.09 degrees each), rows from the horizon
// to the zenith, crowded at the horizon (sky_clouds.h.slang). What the camera sees of the clouds
// outside the view image is read from it, at this sharpness. A drag's preview bakes a quarter of each.
static constexpr VkExtent2D kCloudImageSize{4096, 1024};
static constexpr VkExtent2D kCloudPreviewSize{1024, 256};
static_assert(kCloudImageSize.width == CLOUD_RAW_WIDTH, "a view bake runs its rows through the raw image's (rawTexel)");
static_assert(uint64_t(CLOUD_VIEW_CAPACITY) * CLOUD_VIEW_CAPACITY <= uint64_t(kCloudImageSize.width) * kCloudImageSize.height,
              "the raw image holds a full view bake");

// The clouds as the panel sets them: three tiers over one another (cloudSkyParameters).
struct SkyCloudsSettings
{
  float    coverage{0.0F};      // the low tier, 0 clear .. 1 overcast: the share of the land it shades
  float    thicknessKm{1.45F};  // the low tier's depth, base to the highest tops (cloudDefaultThickness)
  uint32_t seed{0};             // which arrangement of the clouds
  int      type{0};             // the low tier's kind, CLOUD_TYPE_* below CLOUD_LOW_TYPE_COUNT
  float    midCoverage{0.0F};   // the mid tier, altocumulus: the share of the land it shades
  float    highCoverage{0.0F};  // the high tier, cirrus: the share of the sky it veils

  bool operator==(const SkyCloudsSettings&) const = default;
};

// Where the weather map and the shape are read for an arrangement: a stretch of their endless
// tiling far from the last one's.
glm::vec2 cloudSeedOffsetKm(uint32_t seed);

// The share of the distant ground the clouds leave sunlit, for the bake (cloudSunlitAirOfTiers); 1 clear.
float cloudSunlitGround(const SkyCloudsSettings& settings);

// The bake's parameter block for these settings: the tiers' layers and takram's medium, the
// arrangement's offsets, at the settled quality or a drag's.
shaderio::SkyCloudsParameters cloudBakeParameters(const SkyCloudsSettings& settings, bool preview);

//--------------------------------------------------------------------------------------------------
// SkyClouds -- the physical sky's cloud layers.
//
// Owns the noise the clouds are cut from (generated once, at init), the cloud image (one bake of the
// layers as the scene's origin sees them, for this sun), a quarter-size preview image for drags, the
// images the camera sees them in from where it is -- the sky image that follows it (follow) and the
// view image, where it looks, at the screen's resolution (recordView) -- and the pass that bakes
// them. EnvBaker records that pass ahead of its own, which composes the cloud image over the sky into
// the lighting environment; the path tracer composes over what the camera sees the sky image that
// follows it, or the view image where it covers. Both read them through this class's runtime set;
// SkySunDisk::cloudPreview says whether the preview holds the clouds of the moment,
// SceneFrameInfo::cloudFront which sky image the camera takes, and cloudViewForward whether the
// view image serves. The model is takram's (shaders/sky_clouds*.slang).
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
  bool hasClouds() const { return m_settings.coverage > 0.0F || m_settings.midCoverage > 0.0F || m_settings.highCoverage > 0.0F; }
  // The top of the highest layer the settings turn on, km above the surface; 0 with none. The air
  // below it is under the clouds' shade (sky_clouds_air.h.slang).
  float topKm() const { return hasClouds() ? cloudBakeParameters(m_settings, false).maxHeight : 0.0F; }

  // How far the wind has carried the clouds over the land, km along the environment's x and z
  // (cloudDriftStep in sky_cloud_view.hpp). Every bake from here on is of the clouds there: the
  // anchors are places under them (cloudEyeOf), and the shadow image lies on the land under them.
  void             setDrift(const glm::vec2& driftKm) { m_drift = driftKm; }
  const glm::vec2& drift() const { return m_drift; }
  // How far the clouds have gone since the shadow image was baked, km: a reader takes the image that
  // much the other way (cloudSunShare in gltf_pathtrace.slang).
  glm::vec2 shadowShift() const { return m_drift - m_shadowDrift; }
  // Whether the clouds have gone far enough that the shadow image is baked again (kCloudShadowDrift),
  // by recordShadow, behind recordWait.
  bool shadowDue() const { return viewable() && glm::length(shadowShift()) > kCloudShadowDrift; }
  void recordShadow(VkCommandBuffer cmd);

  // Bake the clouds for this sun (toward it, the environment's frame) and observer, into the
  // caller's command buffer, behind a barrier for frames still reading the last bake and ahead of
  // one for the passes that read this one. `preview` bakes the quarter-size image at low quality.
  // A full bake is from the scene's origin under the clouds as they have drifted, and the readers
  // take it: the follow starts over.
  void record(VkCommandBuffer cmd, const glm::vec3& sunDirection, float observerAltitudeKm, bool preview);

  // The sky image as the camera at `eye` sees it: one frame of its follow (cloudFollowStep in
  // sky_cloud_view.hpp), which bakes a band of a follow image when the camera has gone too far from
  // where the image the readers take was baked. Into the caller's command buffer, behind recordWait
  // when followBakes says a band is due. True when the readers take another image from this frame
  // on. Only once the full image holds clouds (viewable).
  bool follow(VkCommandBuffer cmd, const SkyCloudAnchor& eye);
  // Whether follow would bake in this frame for a camera at `eye`.
  bool followBakes(const SkyCloudAnchor& eye) const;
  // Which image the readers take, and where it was baked from.
  const SkyCloudFollow& followed() const { return m_follow; }
  // The lowest of the baked clouds' bases, km above the surface, when the image the readers take is a
  // whole one (cloudImageWhole in sky_clouds.h.slang: a follow image baked from above those bases,
  // with the clouds under its horizon too); 0 when it holds what is above its horizon.
  float followedWholeBaseKm() const;
  // Whether a camera at `eye` is among the baked clouds, above their bases: its follow is the faster
  // one (cloudFollowStep's `among`).
  bool amongClouds(const SkyCloudAnchor& eye) const;

  // Whether the preview image holds the clouds of the moment (the last record was a preview).
  bool showingPreview() const { return m_showingPreview; }

  // Where the last bake measured the shares of the air's light the clouds leave, the sun's and the
  // sky's (a float2, airMain in sky_clouds_bake.slang), for the environment bake recorded after it.
  VkDeviceAddress airAddress() const { return m_airBuf.address; }

  // The clouds where the camera looks, from where it is, into the view image: the last full bake's
  // clouds and sun, marched again for the view's directions only, from where the camera stands
  // (SkyCloudView::eye). Into the caller's command buffer, behind a barrier for frames still reading
  // the view image (recordWait) and ahead of one for those that read this one. Only once the full
  // image holds clouds (viewable).
  void recordView(VkCommandBuffer cmd, const SkyCloudView& view);
  // The barrier a view bake or a follow's goes behind, which the caller records ahead of it: a
  // profiler section starts at the top of the pipe, so one round both would count the frames this
  // waits for.
  void recordWait(VkCommandBuffer cmd);
  // The view image no longer holds what the camera sees (a camera it does not serve): the sky image
  // alone draws the clouds until the next view bake.
  void dropView() { m_viewGeneration = ~uint64_t(0); }
  // Whether a view bake would show anything: the full image holds clouds, not the preview.
  bool viewable() const { return m_bakedClouds && !m_showingPreview; }
  // The view image holds the clouds of the moment, looking where `view()` says.
  bool viewCurrent() const { return viewable() && m_viewGeneration == m_generation; }
  bool viewHolds(const SkyCloudView& view) const { return viewCurrent() && m_view == view; }
  const SkyCloudView& view() const { return m_view; }

  // The images, as combined image samplers at CLOUD_IMAGE_BINDING, CLOUD_PREVIEW_BINDING,
  // CLOUD_VIEW_BINDING, CLOUD_FOLLOW_BINDING, CLOUD_DEPTH_BINDING, CLOUD_SHADOW_BINDING and
  // CLOUD_SHAFTS_BINDING.
  VkDescriptorSetLayout getDescriptorSetLayout() const { return m_runtimePack.getLayout(); }
  VkDescriptorSet       getDescriptorSet() const { return m_runtimePack.getSet(0); }

private:
  void createImages(Resources& res);
  void recordShafts(VkCommandBuffer cmd);
  void recordAirShade(VkCommandBuffer cmd, uint32_t width, uint32_t height);
  void recordDepth(VkCommandBuffer cmd);
  void recordFollow(VkCommandBuffer cmd, const SkyCloudFollowStep& step);
  void createPipelines(const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout);
  void generateNoise(Resources& res);

  bool              m_initialized{false};
  VkDevice          m_device{VK_NULL_HANDLE};
  SkyCloudsSettings m_settings{};
  bool              m_imageClear{false};      // the full image holds (0, 0, 0, 1), and no bake has run since
  bool              m_showingPreview{false};  // the last bake went to the preview image
  // What the last full bake was of, which a view bake repeats: the settings, the sun and observer it
  // took, and whether it had clouds. Each record moves the generation on; the view image holds the
  // clouds of the one it was baked in.
  SkyCloudsSettings m_bakedSettings{};
  glm::vec3         m_bakedSun{0.0F, 1.0F, 0.0F};
  float             m_bakedAltitude{0.0F};
  float             m_bakedBaseKm{0.0F};  // the lowest of the baked clouds' bases, km above the surface; 0 with none
  bool              m_bakedClouds{false};
  uint64_t          m_generation{0};
  uint64_t          m_viewGeneration{~uint64_t(0)};
  SkyCloudView      m_view{};
  SkyCloudFollow    m_follow{};  // the sky image the readers take, and the bake of the next one
  glm::vec2         m_drift{0.0F};        // how far the wind has carried the clouds, km (setDrift)
  glm::vec2         m_shadowDrift{0.0F};  // ... when the shadow image was baked
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
  nvvk::Image m_rawImage;         // the settled march before the filter (filterMain): full size, one level
  nvvk::Image m_viewImage;        // the clouds in view (recordView): CLOUD_VIEW_CAPACITY square, one level, clamped
  nvvk::Image m_followImages[2];  // the sky image from where the camera is (follow): the sky image's size, one level
  nvvk::Image m_rawDepth;         // how far the clouds are in each texel of a march, km: the sky image's size, R16F
  nvvk::Image m_depthImages[3];   // each sky image's depth image (depthMain): the sky image's, then the follow images'
  nvvk::Image m_shadowImage;      // the clouds' shadow on the land (shadowMain): CLOUD_SHADOW_SIZE square, R16F, clamped
  nvvk::Image m_shaftsImage;      // ... over a wider square, for the light shafts (shaftsMain): CLOUD_SHAFTS_SIZE square
  // Storage writes need a view of one level; the images' own views span their chains for sampling.
  VkImageView m_shapeStorageView{VK_NULL_HANDLE};
  VkImageView m_detailStorageView{VK_NULL_HANDLE};
  VkImageView m_weatherStorageView{VK_NULL_HANDLE};
  VkImageView m_cloudStorageView{VK_NULL_HANDLE};

  nvvk::Buffer m_paramsBuf;  // device-local SkyCloudsParameters, updated in the command buffer per bake
  nvvk::Buffer m_airBuf;     // device-local float2, the air's shares a bake measures (airMain)

  VkDescriptorSet      m_brunetonSet{VK_NULL_HANDLE};  // SkyBruneton's, borrowed
  nvvk::DescriptorPack m_noisePack;                    // the noise textures, as storage, for their one write
  nvvk::DescriptorPack m_bakePack;                     // five sets: set 0 writes the full image, 1 the preview, 2 the view image, 3 and 4 the follow images
  nvvk::DescriptorPack m_runtimePack;                  // the images, for their readers

  VkPipelineLayout m_noiseLayout{VK_NULL_HANDLE};
  VkPipeline       m_shapePipeline{VK_NULL_HANDLE};
  VkPipeline       m_detailPipeline{VK_NULL_HANDLE};
  VkPipeline       m_weatherPipeline{VK_NULL_HANDLE};
  VkPipeline       m_turbulencePipeline{VK_NULL_HANDLE};
  VkPipelineLayout m_bakeLayout{VK_NULL_HANDLE};
  VkPipeline       m_bakePipeline{VK_NULL_HANDLE};
  VkPipeline       m_filterPipeline{VK_NULL_HANDLE};
  VkPipeline       m_airPipeline{VK_NULL_HANDLE};
  VkPipeline       m_airShadePipeline{VK_NULL_HANDLE};
  VkPipeline       m_depthPipeline{VK_NULL_HANDLE};
  VkPipeline       m_shadowPipeline{VK_NULL_HANDLE};
  VkPipeline       m_shaftsPipeline{VK_NULL_HANDLE};
  VkCommandPool    m_transientCmdPool{VK_NULL_HANDLE};
};
