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

// Bloom: a glow and a star on the image the tone mapper takes (see bloom.hpp).

#include "bloom.hpp"

#include <string>

#include <glm/glm.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>

#include "_autogen/bloom.slang.h"

// The kernels' sizes and constants, as the shader has them
namespace bloom_kernels {
using namespace glm;
using float2 = glm::vec2;
#include "shaders/bloom.h.slang"
}  // namespace bloom_kernels

static constexpr const char* kMemoryCategory = "Bloom";
static constexpr VkFormat    kFormat         = VK_FORMAT_R16G16B16A16_SFLOAT;

void Bloom::init(Resources& resources)
{
  m_device           = resources.allocator.getDevice();
  m_alloc            = &resources.allocator;
  m_appMemoryTracker = &resources.appMemoryTracker;
  m_samplerPool      = &resources.samplerPool;

  // Linear, clamped to the edge: the shared linear sampler repeats, which would bring light in from
  // the frame's far side
  const VkSamplerCreateInfo samplerInfo{.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                        .magFilter    = VK_FILTER_LINEAR,
                                        .minFilter    = VK_FILTER_LINEAR,
                                        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .maxLod       = VK_LOD_CLAMP_NONE};
  NVVK_CHECK(m_samplerPool->acquireSampler(m_sampler, samplerInfo));
  NVVK_DBG_NAME(m_sampler);

  static_assert(bloom_kernels::BLOOM_MAX_LEVELS == 8, "m_levels holds BLOOM_MAX_LEVELS");
  initTargets();

  m_bindings.addBinding(shaderio::BloomBindings::eBloomA, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::BloomBindings::eBloomB, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::BloomBindings::eBloomC, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::BloomBindings::eBloomOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange        pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(shaderio::BloomPushConstant)};
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  const VkShaderModuleCreateInfo shaderInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bloom_slang_sizeInBytes,
      .pCode    = bloom_slang,
  };
  const auto create = [&](const char* entry, VkPipeline& pipeline) {
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .pNext = &shaderInfo,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                   .pName = entry},
        .layout = m_pipelineLayout,
    };
    NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &pipeline));
    NVVK_DBG_NAME(pipeline);
  };
  create("bloomDown", m_downPipeline);
  create("bloomUp", m_upPipeline);
  create("bloomStar", m_starPipeline);
  create("bloomComposite", m_compositePipeline);
}

void Bloom::deinit(Resources& /*resources*/)
{
  for(VkPipeline* p : {&m_downPipeline, &m_upPipeline, &m_starPipeline, &m_compositePipeline})
  {
    vkDestroyPipeline(m_device, *p, nullptr);
    *p = {};
  }
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};

  releaseTargets();
  m_out.deinit();
  m_star.deinit();
  for(nvvk::RenderTarget& level : m_levels)
    level.deinit();
  if(m_sampler != VK_NULL_HANDLE)
  {
    m_samplerPool->releaseSampler(m_sampler);
    m_sampler = VK_NULL_HANDLE;
  }
}

void Bloom::registerParameters(SettingsRegistry* settings)
{
  // Not persisted: the host sends both with the lighting (fv_lighting.cpp), after every load too
  settings->add({"bloomShare", "Bloom: share of the image's light moved into the glow round bright light (0 = off)"},
                &m_settings.share, Persist::eNo, 0.0f, 0.25f);
  settings->add({"bloomStar", "Bloom: share of the glow drawn as a star across the aperture's blades (none for a round one)"},
                &m_settings.star, Persist::eNo, 0.0f, 1.0f);
}

void Bloom::onResize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(m_size.width == 0 || m_size.height == 0)  // never allocated: the first frame with bloom does
    return;
  releaseTargets();
  ensureTargets(cmd, size);
}

// The targets set up with nothing allocated (update() allocates)
void Bloom::initTargets()
{
  NVVK_CHECK(m_out.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Bloom out"}));
  NVVK_CHECK(m_star.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Bloom star"}));
  for(size_t k = 0; k < m_levels.size(); k++)
  {
    const std::string name = "Bloom level " + std::to_string(k + 1);
    NVVK_CHECK(m_levels[k].init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat, kFormat}, .debugName = name}));
  }
}

// Every image let go (the device idle, or deinit)
void Bloom::releaseTargets()
{
  if(m_size.width == 0 || m_size.height == 0)
    return;
  if(m_appMemoryTracker)
  {
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
    m_appMemoryTracker->untrack(kMemoryCategory, m_star, 1);
    for(int k = 0; k < m_levelCount; k++)
      m_appMemoryTracker->untrack(kMemoryCategory, m_levels[k], eLevelImageCount);
  }
  m_out.deinit();
  m_star.deinit();
  for(nvvk::RenderTarget& level : m_levels)
    level.deinit();
  initTargets();
  m_levelCount = 0;
  m_size       = {};
}

// Allocates the output, the levels and the star for `size` (nothing in flight uses them: the first
// frame with bloom, or a resize with the device idle). False when the image is too small to bloom.
bool Bloom::ensureTargets(VkCommandBuffer cmd, VkExtent2D size)
{
  if(size.width == m_size.width && size.height == m_size.height)
    return m_levelCount > 0;
  const int levels = bloom_kernels::bloomLevels(int(size.width), int(size.height));
  if(levels < bloom_kernels::BLOOM_STAR_LEVEL)
    return false;

  NVVK_CHECK(m_out.update(cmd, size));
  for(int k = 0; k < levels; k++)
  {
    const VkExtent2D levelSize{uint32_t(bloom_kernels::bloomLevelSide(int(size.width), k + 1)),
                               uint32_t(bloom_kernels::bloomLevelSide(int(size.height), k + 1))};
    NVVK_CHECK(m_levels[k].update(cmd, levelSize));
  }
  NVVK_CHECK(m_star.update(cmd, m_levels[bloom_kernels::BLOOM_STAR_LEVEL - 1].getSize()));
  m_levelCount = levels;
  m_size       = size;
  if(m_appMemoryTracker)
  {
    m_appMemoryTracker->track(kMemoryCategory, m_out, 1);
    m_appMemoryTracker->track(kMemoryCategory, m_star, 1);
    for(int k = 0; k < levels; k++)
      m_appMemoryTracker->track(kMemoryCategory, m_levels[k], eLevelImageCount);
  }
  return true;
}

VkDescriptorImageInfo Bloom::sampled(const nvvk::RenderTarget& target, uint32_t index) const
{
  VkDescriptorImageInfo info = target.getColorStorageImageInfo(index);
  info.sampler               = m_sampler;
  return info;
}

void Bloom::dispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkExtent2D outSize, const shaderio::BloomPushConstant& push,
                     const VkDescriptorImageInfo& a, const VkDescriptorImageInfo& b, const VkDescriptorImageInfo& c,
                     const VkDescriptorImageInfo& out)
{
  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::BloomBindings::eBloomA), a);
  writes.append(m_bindings.getWriteSet(shaderio::BloomBindings::eBloomB), b);
  writes.append(m_bindings.getWriteSet(shaderio::BloomBindings::eBloomC), c);
  writes.append(m_bindings.getWriteSet(shaderio::BloomBindings::eBloomOut), out);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  const VkExtent2D groups = nvvk::getGroupCounts(outSize, VkExtent2D{BLOOM_WORKGROUP_SIZE, BLOOM_WORKGROUP_SIZE});
  vkCmdDispatch(cmd, groups.width, groups.height, 1);
  // Read by the next pass, or by the tone mapper and the HDR output
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}

VkDescriptorImageInfo Bloom::cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input,
                                      VkExtent2D size, float exposure)
{
  if(!active() || size.width == 0 || size.height == 0 || !ensureTargets(cmd, size))
    return input;
  NVVK_DBG_SCOPE(cmd);

  const float scale = exposure > 0.0f ? exposure : 1.0f;
  const auto  texel = [](VkExtent2D s) { return glm::vec2(1.0f / float(s.width), 1.0f / float(s.height)); };

  // The previous frame's tone mapper and HDR output may still read the output and the levels
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  VkDescriptorImageInfo image = input;
  image.sampler               = m_sampler;

  // Down: the image into level 1, held in the tone mapper's white under the limit; each level into the next
  const int levels = m_levelCount;
  for(int k = 0; k < levels; k++)
  {
    const VkExtent2D            srcSize = k == 0 ? size : m_levels[k - 1].getSize();
    const VkDescriptorImageInfo src     = k == 0 ? image : sampled(m_levels[k - 1], eLevelDown);
    shaderio::BloomPushConstant push{};
    push.outSize = {int(m_levels[k].getSize().width), int(m_levels[k].getSize().height)};
    push.aTexel  = texel(srcSize);
    push.scale   = k == 0 ? scale : 1.0f;
    push.limit   = k == 0 ? bloom_kernels::BLOOM_LIMIT : 0.0f;
    dispatch(cmd, m_downPipeline, m_levels[k].getSize(), push, src, src, src, m_levels[k].getColorStorageImageInfo(eLevelDown));
  }

  // The star, on its level
  const bool star = m_settings.star > 0.0f && bloom_kernels::bloomStarLines(m_blades) > 0;
  if(star)
  {
    const nvvk::RenderTarget&   level = m_levels[bloom_kernels::BLOOM_STAR_LEVEL - 1];
    const VkDescriptorImageInfo src   = sampled(level, eLevelDown);
    shaderio::BloomPushConstant push{};
    push.outSize  = {int(level.getSize().width), int(level.getSize().height)};
    push.aTexel   = texel(level.getSize());
    push.blades   = m_blades;
    push.rotation = m_rotation;
    push.starStep = bloom_kernels::BLOOM_STAR_LENGTH * float(level.getSize().height) / float(bloom_kernels::BLOOM_STAR_TAPS);
    dispatch(cmd, m_starPipeline, level.getSize(), push, src, src, src, m_star.getColorStorageImageInfo(0));
  }

  // Up: each level's glow from the coarser one's, the coarsest its own level
  for(int k = levels - 2; k >= 0; k--)
  {
    const nvvk::RenderTarget&   coarse    = m_levels[k + 1];
    const VkDescriptorImageInfo coarseUp  = k + 1 == levels - 1 ? sampled(coarse, eLevelDown) : sampled(coarse, eLevelUp);
    const VkDescriptorImageInfo fineLevel = sampled(m_levels[k], eLevelDown);
    shaderio::BloomPushConstant push{};
    push.outSize = {int(m_levels[k].getSize().width), int(m_levels[k].getSize().height)};
    push.aTexel  = texel(m_levels[k].getSize());
    push.bTexel  = texel(coarse.getSize());
    push.tail    = bloom_kernels::BLOOM_TAIL;
    dispatch(cmd, m_upPipeline, m_levels[k].getSize(), push, fineLevel, coarseUp, fineLevel, m_levels[k].getColorStorageImageInfo(eLevelUp));
  }

  // The image with its share of light in the glow, back in its own units
  const VkDescriptorImageInfo glow = levels > 1 ? sampled(m_levels[0], eLevelUp) : sampled(m_levels[0], eLevelDown);
  shaderio::BloomPushConstant push{};
  push.outSize = {int(size.width), int(size.height)};
  push.aTexel  = texel(m_levels[0].getSize());
  push.scale   = 1.0f / scale;
  push.share   = m_settings.share;
  push.star    = star ? m_settings.star : 0.0f;
  const VkDescriptorImageInfo starImage = star ? sampled(m_star, 0) : glow;
  dispatch(cmd, m_compositePipeline, size, push, glow, starImage, image, m_out.getColorStorageImageInfo(0));

  VkDescriptorImageInfo out = m_out.getColorStorageImageInfo(0);
  out.sampler               = resources.linearSampler;  // as the tone mapper reads its input
  return out;
}
