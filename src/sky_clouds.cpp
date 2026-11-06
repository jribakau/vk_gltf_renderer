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
// SkyClouds -- see sky_clouds.hpp for what this class is for, and shaders/sky_clouds.h.slang for
// what the clouds are.
//
// Memory: the noise is 2.4 MB (shape 128^3 and detail 32^3 of R8, the weather 512^2 and turbulence
// 128^2 of RGBA8), the cloud image 43 MB with its mips (kCloudImageSize of RGBA16F), the raw image
// the filter reads 32 MB, the view image 32 MB (CLOUD_VIEW_CAPACITY square) and the preview 2 MB. All
// of it is allocated at init, clouds or none, because the images are bound to the path tracer and the
// bake either way.
//

#include <algorithm>
#include <array>
#include <cmath>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/mipmaps.hpp>

#include "resources.hpp"
#include "sky_clouds.hpp"

// The cloud model's arithmetic, compiled here too for the parameter block it builds. It includes
// nothing and declares no namespace of its own, so it takes shaderio's, where the Slang-named vector
// types, lerp and SkyCloudsParameters live.
namespace shaderio {
#include "shaders/sky_clouds.h.slang"
}  // namespace shaderio

namespace {

nvvk::Image createImage(Resources&           res,
                        VkExtent3D           extent,
                        VkFormat             format,
                        uint32_t             mipLevels,
                        VkSamplerAddressMode addressV,
                        const char*          debugName,
                        VkSamplerAddressMode addressU = VK_SAMPLER_ADDRESS_MODE_REPEAT)
{
  const bool is3D = extent.depth > 1;

  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.imageType         = is3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  imageInfo.extent            = extent;
  imageInfo.format            = format;
  imageInfo.mipLevels         = mipLevels;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  VkImageViewCreateInfo viewInfo       = DEFAULT_VkImageViewCreateInfo;
  viewInfo.viewType                    = is3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.subresourceRange.levelCount = mipLevels;

  nvvk::Image image;
  NVVK_CHECK(res.allocator.createImage(image, imageInfo, viewInfo));
  nvvk::DebugUtil::getInstance().setObjectName(image.image, debugName);

  // The noise tiles on every axis; the cloud images wrap round the horizon and stop at the zenith
  // and the horizon rows, which is what `addressV` says for them. The view image stops on both.
  const VkSamplerCreateInfo samplerInfo{
      .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter    = VK_FILTER_LINEAR,
      .minFilter    = VK_FILTER_LINEAR,
      .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = addressU,
      .addressModeV = addressV,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .maxLod       = VK_LOD_CLAMP_NONE,
  };
  NVVK_CHECK(res.samplerPool.acquireSampler(image.descriptor.sampler, samplerInfo));

  // GENERAL for life: written as storage, sampled after, the one layout legal for both.
  image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  return image;
}

VkImageView createLevelView(VkDevice device, const nvvk::Image& image, VkFormat format, bool is3D = false)
{
  VkImageViewCreateInfo viewInfo = DEFAULT_VkImageViewCreateInfo;
  viewInfo.viewType              = is3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.image                 = image.image;
  viewInfo.format                = format;
  viewInfo.subresourceRange      = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view               = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &view));
  return view;
}

void destroyImage(Resources& res, nvvk::Image& image)
{
  if(image.descriptor.sampler != VK_NULL_HANDLE)
    res.samplerPool.releaseSampler(image.descriptor.sampler);
  res.allocator.destroyImage(image);
  image = {};
}

VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, std::span<const uint32_t> spirv, const char* entryPoint)
{
  VkShaderModuleCreateInfo moduleInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size_bytes(),
      .pCode    = spirv.data(),
  };
  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.pName = entryPoint;
  stageInfo.pNext = &moduleInfo;

  VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipelineInfo.layout = layout;
  pipelineInfo.stage  = stageInfo;
  VkPipeline pipeline = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateComputePipelines(device, {}, 1, &pipelineInfo, nullptr, &pipeline));
  return pipeline;
}

uint32_t groups(uint32_t size, uint32_t workgroup)
{
  return (size + workgroup - 1) / workgroup;
}

// A volume's mip chain, each level blitted from the one above: nvvk's helper is for 2D images. The
// volume stays in GENERAL, which a blit may read and write.
void generateMipmaps3D(VkCommandBuffer cmd, VkImage image, uint32_t size, uint32_t levels)
{
  for(uint32_t level = 1; level < levels; ++level)
  {
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    const int32_t     src = int32_t(std::max(size >> (level - 1), 1U));
    const int32_t     dst = int32_t(std::max(size >> level, 1U));
    const VkImageBlit blit{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1},
        .srcOffsets     = {{0, 0, 0}, {src, src, src}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
        .dstOffsets     = {{0, 0, 0}, {dst, dst, dst}},
    };
    vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
  }
}

void clearImage(VkCommandBuffer cmd, const nvvk::Image& image)
{
  const VkClearColorValue       clear{.float32 = {0.0F, 0.0F, 0.0F, 1.0F}};
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
  vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
}

}  // namespace

// Marches a texel of the view image. Its texels are a pixel each, where the sky image's are several,
// so fewer do: the noise they leave is a pixel's grain, which the filter averages as it does the sky
// image's.
constexpr int kViewSamples = 4;

glm::vec2 cloudSeedOffsetKm(uint32_t seed)
{
  // The golden ratio's and the silver ratio's fractions, so successive seeds land far apart and
  // never on a multiple of the noise's periods. Within 5000 km, where a float still resolves the
  // detail noise's texels.
  const double a = std::fmod(double(seed) * 0.6180339887498949, 1.0);
  const double b = std::fmod(double(seed) * 0.4142135623730950, 1.0);
  return {float(a * 5000.0), float(b * 5000.0)};
}

float cloudSunlitGround(const SkyCloudsSettings& settings)
{
  return shaderio::cloudSunlitAirOfTiers(settings.coverage, settings.midCoverage, settings.highCoverage);
}

shaderio::SkyCloudsParameters cloudBakeParameters(const SkyCloudsSettings& settings, bool preview)
{
  shaderio::SkyCloudsParameters p =
      shaderio::cloudWithQuality(shaderio::cloudSkyParameters(settings.coverage, settings.thicknessKm, settings.type,
                                                              settings.midCoverage, settings.highCoverage),
                                 preview);
  // The arrangement moves the weather map under the observer, and the billows within it.
  const glm::vec2 offset = cloudSeedOffsetKm(settings.seed);
  p.weatherOffset        = offset;
  p.shapeOffset          = glm::fract(glm::vec3(offset.x, 0.0F, offset.y) / shaderio::CLOUD_SHAPE_PERIOD);
  return p;
}

void SkyClouds::init(Resources& res, const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout, VkDescriptorSet brunetonSet)
{
  assert(!m_initialized && "init called twice");
  m_device      = res.allocator.getDevice();
  m_brunetonSet = brunetonSet;

  const VkCommandPoolCreateInfo poolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = res.app->getQueue(0).familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_transientCmdPool));
  NVVK_DBG_NAME(m_transientCmdPool);

  // Device-local and written by record() through the command buffer: a preview is recorded while
  // earlier frames may still be running an earlier bake that reads it.
  static_assert(sizeof(shaderio::SkyCloudsParameters) % 4 == 0 && sizeof(shaderio::SkyCloudsParameters) <= 65536, "vkCmdUpdateBuffer limits");
  NVVK_CHECK(res.allocator.createBuffer(m_paramsBuf, sizeof(shaderio::SkyCloudsParameters),
                                        VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
  NVVK_DBG_NAME(m_paramsBuf.buffer);
  // Written and read by the bakes alone: airMain measures, the bake's march and the environment
  // bake (through its address) read, each behind a barrier.
  NVVK_CHECK(res.allocator.createBuffer(m_airBuf, 2 * sizeof(float),
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
  NVVK_DBG_NAME(m_airBuf.buffer);

  createImages(res);
  createPipelines(shaders, brunetonSetLayout);
  generateNoise(res);
  m_initialized = true;
}

void SkyClouds::deinit(Resources& res)
{
  if(!m_initialized)
    return;
  vkDestroyImageView(m_device, m_shapeStorageView, nullptr);
  vkDestroyImageView(m_device, m_detailStorageView, nullptr);
  vkDestroyImageView(m_device, m_weatherStorageView, nullptr);
  vkDestroyImageView(m_device, m_cloudStorageView, nullptr);
  for(nvvk::Image* image :
      {&m_shapeNoise, &m_detailNoise, &m_weatherNoise, &m_turbulenceNoise, &m_cloudImage, &m_previewImage, &m_rawImage, &m_viewImage})
    destroyImage(res, *image);
  res.allocator.destroyBuffer(m_paramsBuf);
  res.allocator.destroyBuffer(m_airBuf);
  for(VkPipeline pipeline :
      {m_shapePipeline, m_detailPipeline, m_weatherPipeline, m_turbulencePipeline, m_bakePipeline, m_filterPipeline, m_airPipeline,
       m_airShadePipeline})
    vkDestroyPipeline(m_device, pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_noiseLayout, nullptr);
  vkDestroyPipelineLayout(m_device, m_bakeLayout, nullptr);
  m_noisePack.deinit();
  m_bakePack.deinit();
  m_runtimePack.deinit();
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);
  m_initialized = false;
}

void SkyClouds::createImages(Resources& res)
{
  constexpr VkSamplerAddressMode kRepeat = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  constexpr VkSamplerAddressMode kClamp  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  m_weatherMips = nvvk::mipLevels(VkExtent2D{CLOUD_WEATHER_SIZE, CLOUD_WEATHER_SIZE});
  m_imageMips   = nvvk::mipLevels(kCloudImageSize);
  m_shapeMips   = nvvk::mipLevels(uint32_t(CLOUD_SHAPE_NOISE_SIZE));
  m_detailMips  = nvvk::mipLevels(uint32_t(CLOUD_DETAIL_NOISE_SIZE));

  m_shapeNoise  = createImage(res, {CLOUD_SHAPE_NOISE_SIZE, CLOUD_SHAPE_NOISE_SIZE, CLOUD_SHAPE_NOISE_SIZE}, VK_FORMAT_R8_UNORM,
                              m_shapeMips, kRepeat, "SkyClouds::shapeNoise");
  m_detailNoise = createImage(res, {CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE}, VK_FORMAT_R8_UNORM,
                              m_detailMips, kRepeat, "SkyClouds::detailNoise");
  m_weatherNoise = createImage(res, {CLOUD_WEATHER_SIZE, CLOUD_WEATHER_SIZE, 1}, VK_FORMAT_R8G8B8A8_UNORM, m_weatherMips, kRepeat,
                               "SkyClouds::weatherNoise");
  m_turbulenceNoise = createImage(res, {CLOUD_TURBULENCE_SIZE, CLOUD_TURBULENCE_SIZE, 1}, VK_FORMAT_R8G8B8A8_UNORM, 1, kRepeat,
                                  "SkyClouds::turbulenceNoise");
  m_cloudImage = createImage(res, {kCloudImageSize.width, kCloudImageSize.height, 1}, VK_FORMAT_R16G16B16A16_SFLOAT, m_imageMips,
                             kClamp, "SkyClouds::image");
  m_previewImage = createImage(res, {kCloudPreviewSize.width, kCloudPreviewSize.height, 1}, VK_FORMAT_R16G16B16A16_SFLOAT, 1, kClamp,
                               "SkyClouds::preview");
  m_rawImage = createImage(res, {kCloudImageSize.width, kCloudImageSize.height, 1}, VK_FORMAT_R16G16B16A16_SFLOAT, 1, kClamp,
                           "SkyClouds::raw");
  m_viewImage = createImage(res, {CLOUD_VIEW_CAPACITY, CLOUD_VIEW_CAPACITY, 1}, VK_FORMAT_R16G16B16A16_SFLOAT, 1, kClamp,
                            "SkyClouds::view", kClamp);

  m_shapeStorageView   = createLevelView(m_device, m_shapeNoise, VK_FORMAT_R8_UNORM, true);
  m_detailStorageView  = createLevelView(m_device, m_detailNoise, VK_FORMAT_R8_UNORM, true);
  m_weatherStorageView = createLevelView(m_device, m_weatherNoise, VK_FORMAT_R8G8B8A8_UNORM);
  m_cloudStorageView   = createLevelView(m_device, m_cloudImage, VK_FORMAT_R16G16B16A16_SFLOAT);
}

void SkyClouds::createPipelines(const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout)
{
  // Noise: four storage textures, written once.
  {
    using B = shaderio::SkyCloudsNoiseBindings;
    nvvk::DescriptorBindings bindings;
    for(uint32_t b : {uint32_t(B::eCloudNoiseShape), uint32_t(B::eCloudNoiseDetail), uint32_t(B::eCloudNoiseWeather),
                      uint32_t(B::eCloudNoiseTurbulence)})
      bindings.addBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_noisePack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_noisePack.getLayout());
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_noiseLayout, {m_noisePack.getLayout()}, {}));
    NVVK_DBG_NAME(m_noiseLayout);
    m_shapePipeline      = createComputePipeline(m_device, m_noiseLayout, shaders.noise, "shapeMain");
    m_detailPipeline     = createComputePipeline(m_device, m_noiseLayout, shaders.noise, "detailMain");
    m_weatherPipeline    = createComputePipeline(m_device, m_noiseLayout, shaders.noise, "weatherMain");
    m_turbulencePipeline = createComputePipeline(m_device, m_noiseLayout, shaders.noise, "turbulenceMain");

    nvvk::WriteSetContainer writes;
    writes.append(m_noisePack.makeWrite(B::eCloudNoiseShape), m_shapeStorageView, VK_IMAGE_LAYOUT_GENERAL);
    writes.append(m_noisePack.makeWrite(B::eCloudNoiseDetail), m_detailStorageView, VK_IMAGE_LAYOUT_GENERAL);
    writes.append(m_noisePack.makeWrite(B::eCloudNoiseWeather), m_weatherStorageView, VK_IMAGE_LAYOUT_GENERAL);
    writes.append(m_noisePack.makeWrite(B::eCloudNoiseTurbulence), m_turbulenceNoise);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // Bake: the image out, the noise and the parameters in; set 1 is SkyBruneton's. Three sets of the
  // one layout, the full image's, the preview's and the view image's, which differ only in the
  // images written: the settled march writes the raw image, which the filter (the same layout) reads
  // into the cloud image's first level, or into the view image; a drag's preview is written as it is.
  {
    using B = shaderio::SkyCloudsBakeBindings;
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(B::eCloudBakeOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    for(uint32_t b : {uint32_t(B::eCloudBakeShape), uint32_t(B::eCloudBakeDetail), uint32_t(B::eCloudBakeWeather),
                      uint32_t(B::eCloudBakeTurbulence)})
      bindings.addBinding(b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(B::eCloudBakeParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(B::eCloudBakeFiltered, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(B::eCloudBakeAir, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_bakePack.init(bindings, m_device, 3));
    NVVK_DBG_NAME(m_bakePack.getLayout());

    const VkPushConstantRange pushRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::SkyCloudsPushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_bakeLayout, {m_bakePack.getLayout(), brunetonSetLayout}, {pushRange}));
    NVVK_DBG_NAME(m_bakeLayout);
    m_bakePipeline   = createComputePipeline(m_device, m_bakeLayout, shaders.bake, "main");
    m_filterPipeline = createComputePipeline(m_device, m_bakeLayout, shaders.bake, "filterMain");
    m_airPipeline      = createComputePipeline(m_device, m_bakeLayout, shaders.bake, "airMain");
    m_airShadePipeline = createComputePipeline(m_device, m_bakeLayout, shaders.bake, "airShadeMain");

    nvvk::WriteSetContainer writes;
    writes.append(m_bakePack.makeWrite(B::eCloudBakeOut, 0), m_rawImage);
    writes.append(m_bakePack.makeWrite(B::eCloudBakeOut, 1), m_previewImage);
    writes.append(m_bakePack.makeWrite(B::eCloudBakeOut, 2), m_rawImage);
    writes.append(m_bakePack.makeWrite(B::eCloudBakeFiltered, 2), m_viewImage);
    for(uint32_t set = 0; set < 3; ++set)
    {
      writes.append(m_bakePack.makeWrite(B::eCloudBakeShape, set), m_shapeNoise);
      writes.append(m_bakePack.makeWrite(B::eCloudBakeDetail, set), m_detailNoise);
      writes.append(m_bakePack.makeWrite(B::eCloudBakeWeather, set), m_weatherNoise);
      writes.append(m_bakePack.makeWrite(B::eCloudBakeTurbulence, set), m_turbulenceNoise);
      writes.append(m_bakePack.makeWrite(B::eCloudBakeParams, set), m_paramsBuf);
      writes.append(m_bakePack.makeWrite(B::eCloudBakeAir, set), m_airBuf);
      if(set != 2)
        writes.append(m_bakePack.makeWrite(B::eCloudBakeFiltered, set), m_cloudStorageView, VK_IMAGE_LAYOUT_GENERAL);
    }
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // Runtime: the images, for the environment bake and the path tracer.
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(CLOUD_IMAGE_BINDING, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(CLOUD_PREVIEW_BINDING, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(CLOUD_VIEW_BINDING, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    NVVK_CHECK(m_runtimePack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_runtimePack.getLayout());

    nvvk::WriteSetContainer writes;
    writes.append(m_runtimePack.makeWrite(CLOUD_IMAGE_BINDING), m_cloudImage);
    writes.append(m_runtimePack.makeWrite(CLOUD_PREVIEW_BINDING), m_previewImage);
    writes.append(m_runtimePack.makeWrite(CLOUD_VIEW_BINDING), m_viewImage);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }
}

void SkyClouds::generateNoise(Resources& res)
{
  nvutils::ScopedTimer st("SkyClouds::generateNoise");
  VkCommandBuffer      cmd = VK_NULL_HANDLE;
  NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));

  for(nvvk::Image* image :
      {&m_shapeNoise, &m_detailNoise, &m_weatherNoise, &m_turbulenceNoise, &m_cloudImage, &m_previewImage, &m_rawImage, &m_viewImage})
    nvvk::cmdImageMemoryBarrier(cmd, {image->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_noiseLayout, 0, 1, m_noisePack.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_shapePipeline);
  vkCmdDispatch(cmd, groups(CLOUD_SHAPE_NOISE_SIZE, 4), groups(CLOUD_SHAPE_NOISE_SIZE, 4), groups(CLOUD_SHAPE_NOISE_SIZE, 4));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_detailPipeline);
  vkCmdDispatch(cmd, groups(CLOUD_DETAIL_NOISE_SIZE, 4), groups(CLOUD_DETAIL_NOISE_SIZE, 4), groups(CLOUD_DETAIL_NOISE_SIZE, 4));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_weatherPipeline);
  vkCmdDispatch(cmd, groups(CLOUD_WEATHER_SIZE, CLOUD_WORKGROUP_SIZE), groups(CLOUD_WEATHER_SIZE, CLOUD_WORKGROUP_SIZE), 1);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_turbulencePipeline);
  vkCmdDispatch(cmd, groups(CLOUD_TURBULENCE_SIZE, CLOUD_WORKGROUP_SIZE), groups(CLOUD_TURBULENCE_SIZE, CLOUD_WORKGROUP_SIZE), 1);

  // takram samples the weather map at a level by distance, and the bake the volumes (noiseLod in
  // sky_clouds_bake.slang): their chains, from the levels just written.
  nvvk::cmdGenerateMipmaps(cmd, m_weatherNoise.image, {CLOUD_WEATHER_SIZE, CLOUD_WEATHER_SIZE}, m_weatherMips, 1, VK_IMAGE_LAYOUT_GENERAL);
  generateMipmaps3D(cmd, m_shapeNoise.image, CLOUD_SHAPE_NOISE_SIZE, m_shapeMips);
  generateMipmaps3D(cmd, m_detailNoise.image, CLOUD_DETAIL_NOISE_SIZE, m_detailMips);

  // The images start clear, so a sky with no clouds is the sky before any bake runs. The view image's
  // texels past a bake's part are read by the filtering at its edges, and must hold a number.
  clearImage(cmd, m_cloudImage);
  clearImage(cmd, m_previewImage);
  clearImage(cmd, m_viewImage);
  m_imageClear     = true;
  m_showingPreview = false;

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, res.app->getQueue(0).queue));
}

void SkyClouds::record(VkCommandBuffer cmd, const glm::vec3& sunDirection, float observerAltitudeKm, bool preview)
{
  assert(m_initialized && "init must be called first");
  if(!hasClouds() && m_imageClear && !m_showingPreview)
    return;  // nothing to bake, and the image the readers look at already says so

  // Frames still in flight may be reading the last bake, and its parameters; overwriting them needs
  // those frames done, which an execution dependency alone gives.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_NONE, VK_ACCESS_2_NONE);

  // What a view bake repeats: a full bake's clouds, sun and observer. Any bake leaves the view image
  // behind, until a view bake catches it up.
  ++m_generation;
  if(!preview)
  {
    m_bakedSettings = m_settings;
    m_bakedSun      = sunDirection;
    m_bakedAltitude = observerAltitudeKm;
    m_bakedClouds   = hasClouds();
  }
  if(!hasClouds())
  {
    clearImage(cmd, m_cloudImage);
    m_imageClear     = true;
    m_showingPreview = false;
    m_bakedClouds    = false;
  }
  else
  {
    const shaderio::SkyCloudsParameters params = cloudBakeParameters(m_settings, preview);
    vkCmdUpdateBuffer(cmd, m_paramsBuf.buffer, 0, sizeof(params), &params);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

    const VkExtent2D                      size = preview ? kCloudPreviewSize : kCloudImageSize;
    const shaderio::SkyCloudsPushConstant push{
        .imageSize        = {size.width, size.height},
        .observerAltitude = observerAltitudeKm,
        .sunlitAir        = cloudSunlitGround(m_settings),
        .sunDirection     = sunDirection,
        .texelAngle       = 6.2831853F / float(size.width),
    };
    const std::array<VkDescriptorSet, 2> sets{m_bakePack.getSet(preview ? 1 : 0), m_brunetonSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bakeLayout, 0, uint32_t(sets.size()), sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_bakeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    // First what the clouds leave of the sun's and the sky's light on the air, which the march reads
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_airPipeline);
    vkCmdDispatch(cmd, 1, 1, 1);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bakePipeline);
    vkCmdDispatch(cmd, groups(size.width, CLOUD_WORKGROUP_SIZE), groups(size.height, CLOUD_WORKGROUP_SIZE), 1);
    recordAirShade(cmd, size.width, size.height);

    if(!preview)
    {
      // The raw march through the filter into the cloud image (filterMain): its neighbours' marches
      // average out the noise of eight a texel. The preview's frames move on before it would show.
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_filterPipeline);
      vkCmdDispatch(cmd, groups(size.width, CLOUD_WORKGROUP_SIZE), groups(size.height, CLOUD_WORKGROUP_SIZE), 1);
      // The environment bake reads the full image two levels down, at its own size.
      nvvk::cmdGenerateMipmaps(cmd, m_cloudImage.image, kCloudImageSize, m_imageMips, 1, VK_IMAGE_LAYOUT_GENERAL);
    }
    m_imageClear     = m_imageClear && preview;
    m_showingPreview = preview;
  }

  // For the environment bake next in this command buffer, which also reads the air's shares, and the
  // frames that draw the sky.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}

void SkyClouds::recordAirShade(VkCommandBuffer cmd, uint32_t width, uint32_t height)
{
  // The march's texels, in place: the shade of the air seen through them (airShadeMain)
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_airShadePipeline);
  vkCmdDispatch(cmd, groups(width, CLOUD_WORKGROUP_SIZE), groups(height, CLOUD_WORKGROUP_SIZE), 1);
}

void SkyClouds::recordView(VkCommandBuffer cmd, const SkyCloudView& view)
{
  assert(m_initialized && "init must be called first");
  assert(viewable() && "a view bake repeats a full bake's clouds");
  assert(view.size.x <= CLOUD_VIEW_CAPACITY && view.size.y <= CLOUD_VIEW_CAPACITY);

  // Frames in flight may be reading the view image, and an earlier bake the raw image and the
  // parameters.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_NONE, VK_ACCESS_2_NONE);

  shaderio::SkyCloudsParameters params = cloudBakeParameters(m_bakedSettings, false);
  params.samples                       = kViewSamples;
  vkCmdUpdateBuffer(cmd, m_paramsBuf.buffer, 0, sizeof(params), &params);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

  // The noise's level follows a texel's angle (noiseLod): the view's are a pixel, so the eddies the
  // sky image averages away are marched here.
  const shaderio::SkyCloudsPushConstant push{
      .imageSize        = view.size,
      .observerAltitude = m_bakedAltitude,
      .sunlitAir        = cloudSunlitGround(m_bakedSettings),
      .sunDirection     = m_bakedSun,
      .texelAngle       = 2.0F * view.right.w / float(view.size.x),
      .viewRight        = view.right,
      .viewUp           = view.up,
      .viewForward      = glm::vec4(view.forward, 1.0F),
  };
  const std::array<VkDescriptorSet, 2> sets{m_bakePack.getSet(2), m_brunetonSet};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bakeLayout, 0, uint32_t(sets.size()), sets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, m_bakeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bakePipeline);
  vkCmdDispatch(cmd, groups(view.size.x, CLOUD_WORKGROUP_SIZE), groups(view.size.y, CLOUD_WORKGROUP_SIZE), 1);
  recordAirShade(cmd, view.size.x, view.size.y);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_filterPipeline);
  vkCmdDispatch(cmd, groups(view.size.x, CLOUD_WORKGROUP_SIZE), groups(view.size.y, CLOUD_WORKGROUP_SIZE), 1);

  // For the frames that draw the sky.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  m_view           = view;
  m_viewGeneration = m_generation;
}
