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
// Memory: the noise is 2.4 MB (shape 128^3 and detail 64^3 of R8, weather 512^2), the cloud image
// 8 MB (kCloudImageSize of RGBA16F). All of it is allocated at init, clouds or none, because the
// image is bound to the path tracer and the bake either way.
//

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

#include "resources.hpp"
#include "sky_clouds.hpp"

#include "shaders/sky_clouds_io.h.slang"
// The layer's arithmetic, compiled here too for the one number the host needs (cloudSunlitAir).
// It includes nothing and declares no namespace of its own, so it takes shaderio's, where the
// Slang-named vector types and lerp live.
namespace shaderio {
#include "shaders/sky_clouds.h.slang"
}  // namespace shaderio

namespace {

// Samples along a view ray through the layer and toward the sun from each: the settled image's, and a
// drag's. The settled ones put a step under 10 m through a kilometre-deep cloud overhead; the preview
// is a third as many, softer in the cloud edges and the shadowed sides.
constexpr int kViewSteps        = 128;
constexpr int kLightSteps       = 6;
constexpr int kPreviewViewSteps = 40;
constexpr int kPreviewLightSteps = 4;

nvvk::Image createImage(Resources& res, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkSamplerAddressMode addressV, const char* debugName)
{
  const bool is3D = extent.depth > 1;

  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.imageType         = is3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  imageInfo.extent            = extent;
  imageInfo.format            = format;
  imageInfo.usage             = usage;

  VkImageViewCreateInfo viewInfo = DEFAULT_VkImageViewCreateInfo;
  viewInfo.viewType              = is3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;

  nvvk::Image image;
  NVVK_CHECK(res.allocator.createImage(image, imageInfo, viewInfo));
  nvvk::DebugUtil::getInstance().setObjectName(image.image, debugName);

  // The noise tiles on every axis; the cloud image wraps round the horizon and stops at the zenith
  // and the horizon rows, which is what `addressV` says for it.
  const VkSamplerCreateInfo samplerInfo{
      .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter    = VK_FILTER_LINEAR,
      .minFilter    = VK_FILTER_LINEAR,
      .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = addressV,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
  };
  NVVK_CHECK(res.samplerPool.acquireSampler(image.descriptor.sampler, samplerInfo));

  // GENERAL for life: written as storage, sampled after, the one layout legal for both.
  image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  return image;
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

}  // namespace

glm::vec2 cloudSeedOffsetKm(uint32_t seed)
{
  // The golden ratio's and the silver ratio's fractions, so successive seeds land far apart and
  // never on a multiple of any of the noise's periods. Within 5000 km, where a float still resolves
  // the detail noise's texels.
  const double a = std::fmod(double(seed) * 0.6180339887498949, 1.0);
  const double b = std::fmod(double(seed) * 0.4142135623730950, 1.0);
  return {float(a * 5000.0), float(b * 5000.0)};
}

float cloudSunlitGround(const SkyCloudsParameters& parameters)
{
  return parameters.coverage > 0.0F ? shaderio::cloudSunlitAir(parameters.coverage) : 1.0F;
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

  createImages(res);
  createPipelines(shaders, brunetonSetLayout);
  generateNoise(res);
  m_initialized = true;
}

void SkyClouds::deinit(Resources& res)
{
  if(!m_initialized)
    return;
  destroyImage(res, m_shapeNoise);
  destroyImage(res, m_detailNoise);
  destroyImage(res, m_weatherNoise);
  destroyImage(res, m_cloudImage);
  for(VkPipeline pipeline : {m_shapePipeline, m_detailPipeline, m_weatherPipeline, m_bakePipeline})
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
  constexpr VkImageUsageFlags kNoiseUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  m_shapeNoise = createImage(res, {CLOUD_SHAPE_NOISE_SIZE, CLOUD_SHAPE_NOISE_SIZE, CLOUD_SHAPE_NOISE_SIZE}, VK_FORMAT_R8_UNORM,
                             kNoiseUsage, VK_SAMPLER_ADDRESS_MODE_REPEAT, "SkyClouds::shapeNoise");
  m_detailNoise = createImage(res, {CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE}, VK_FORMAT_R8_UNORM,
                              kNoiseUsage, VK_SAMPLER_ADDRESS_MODE_REPEAT, "SkyClouds::detailNoise");
  m_weatherNoise = createImage(res, {CLOUD_WEATHER_SIZE, CLOUD_WEATHER_SIZE, 1}, VK_FORMAT_R8_UNORM, kNoiseUsage,
                               VK_SAMPLER_ADDRESS_MODE_REPEAT, "SkyClouds::weatherNoise");
  m_cloudImage = createImage(res, {kCloudImageSize.width, kCloudImageSize.height, 1}, VK_FORMAT_R16G16B16A16_SFLOAT,
                             kNoiseUsage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, "SkyClouds::image");
}

void SkyClouds::createPipelines(const Shaders& shaders, VkDescriptorSetLayout brunetonSetLayout)
{
  // Noise: three storage volumes, written once.
  {
    using B = shaderio::SkyCloudsNoiseBindings;
    nvvk::DescriptorBindings bindings;
    for(uint32_t b : {uint32_t(B::eCloudNoiseShape), uint32_t(B::eCloudNoiseDetail), uint32_t(B::eCloudNoiseWeather)})
      bindings.addBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_noisePack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_noisePack.getLayout());
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_noiseLayout, {m_noisePack.getLayout()}, {}));
    NVVK_DBG_NAME(m_noiseLayout);
    m_shapePipeline   = createComputePipeline(m_device, m_noiseLayout, shaders.noise, "shapeMain");
    m_detailPipeline  = createComputePipeline(m_device, m_noiseLayout, shaders.noise, "detailMain");
    m_weatherPipeline = createComputePipeline(m_device, m_noiseLayout, shaders.noise, "weatherMain");

    nvvk::WriteSetContainer writes;
    writes.append(m_noisePack.makeWrite(B::eCloudNoiseShape), m_shapeNoise);
    writes.append(m_noisePack.makeWrite(B::eCloudNoiseDetail), m_detailNoise);
    writes.append(m_noisePack.makeWrite(B::eCloudNoiseWeather), m_weatherNoise);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // Bake: the image out, the volumes in; set 1 is SkyBruneton's.
  {
    using B = shaderio::SkyCloudsBakeBindings;
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(B::eCloudBakeOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    for(uint32_t b : {uint32_t(B::eCloudBakeShape), uint32_t(B::eCloudBakeDetail), uint32_t(B::eCloudBakeWeather)})
      bindings.addBinding(b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_bakePack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_bakePack.getLayout());

    const VkPushConstantRange pushRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::SkyCloudsPushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_bakeLayout, {m_bakePack.getLayout(), brunetonSetLayout}, {pushRange}));
    NVVK_DBG_NAME(m_bakeLayout);
    m_bakePipeline = createComputePipeline(m_device, m_bakeLayout, shaders.bake, "main");

    nvvk::WriteSetContainer writes;
    writes.append(m_bakePack.makeWrite(B::eCloudBakeOut), m_cloudImage);
    writes.append(m_bakePack.makeWrite(B::eCloudBakeShape), m_shapeNoise);
    writes.append(m_bakePack.makeWrite(B::eCloudBakeDetail), m_detailNoise);
    writes.append(m_bakePack.makeWrite(B::eCloudBakeWeather), m_weatherNoise);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // Runtime: the image, for the environment bake and the path tracer.
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(CLOUD_IMAGE_BINDING, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    NVVK_CHECK(m_runtimePack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_runtimePack.getLayout());

    nvvk::WriteSetContainer writes;
    writes.append(m_runtimePack.makeWrite(CLOUD_IMAGE_BINDING), m_cloudImage);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }
}

void SkyClouds::generateNoise(Resources& res)
{
  nvutils::ScopedTimer st("SkyClouds::generateNoise");
  VkCommandBuffer      cmd = VK_NULL_HANDLE;
  NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));

  for(nvvk::Image* image : {&m_shapeNoise, &m_detailNoise, &m_weatherNoise, &m_cloudImage})
    nvvk::cmdImageMemoryBarrier(cmd, {image->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_noiseLayout, 0, 1, m_noisePack.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_shapePipeline);
  vkCmdDispatch(cmd, groups(CLOUD_SHAPE_NOISE_SIZE, 4), groups(CLOUD_SHAPE_NOISE_SIZE, 4), groups(CLOUD_SHAPE_NOISE_SIZE, 4));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_detailPipeline);
  vkCmdDispatch(cmd, groups(CLOUD_DETAIL_NOISE_SIZE, 4), groups(CLOUD_DETAIL_NOISE_SIZE, 4), groups(CLOUD_DETAIL_NOISE_SIZE, 4));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_weatherPipeline);
  vkCmdDispatch(cmd, groups(CLOUD_WEATHER_SIZE, CLOUD_WORKGROUP_SIZE), groups(CLOUD_WEATHER_SIZE, CLOUD_WORKGROUP_SIZE), 1);

  // The image starts clear, so a sky with no clouds is the sky before any bake runs.
  const VkClearColorValue       clear{.float32 = {0.0F, 0.0F, 0.0F, 1.0F}};
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(cmd, m_cloudImage.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
  m_imageClear = true;

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, res.app->getQueue(0).queue));
}

void SkyClouds::record(VkCommandBuffer cmd, const glm::vec3& sunDirection, float observerAltitudeKm, bool preview)
{
  assert(m_initialized && "init must be called first");
  if(!hasClouds() && m_imageClear)
    return;  // nothing to bake, and the image already says so

  // Frames still in flight may be reading the last image; overwriting it needs them done, which an
  // execution dependency alone gives.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_NONE, VK_ACCESS_2_NONE);

  if(!hasClouds())
  {
    const VkClearColorValue       clear{.float32 = {0.0F, 0.0F, 0.0F, 1.0F}};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_cloudImage.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
    m_imageClear = true;
  }
  else
  {
    const glm::vec2                 offset = cloudSeedOffsetKm(m_parameters.seed);
    const shaderio::SkyCloudsPushConstant push{
        .imageSize        = {kCloudImageSize.width, kCloudImageSize.height},
        .level            = shaderio::cloudLevel(m_parameters.coverage),
        .bottom           = shaderio::CLOUD_BOTTOM_KM,
        .sunDirection     = sunDirection,
        .thickness        = m_parameters.thicknessKm,
        .offset           = offset,
        .observerAltitude = observerAltitudeKm,
        .sunlitAir        = cloudSunlitGround(m_parameters),
        .viewSteps        = preview ? kPreviewViewSteps : kViewSteps,
        .lightSteps       = preview ? kPreviewLightSteps : kLightSteps,
    };
    const std::array<VkDescriptorSet, 2> sets{m_bakePack.getSet(0), m_brunetonSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bakeLayout, 0, uint32_t(sets.size()), sets.data(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bakePipeline);
    vkCmdPushConstants(cmd, m_bakeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, groups(kCloudImageSize.width, CLOUD_WORKGROUP_SIZE), groups(kCloudImageSize.height, CLOUD_WORKGROUP_SIZE), 1);
    m_imageClear = false;
  }

  // For the environment bake next in this command buffer, and the frames that draw the sky.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}
