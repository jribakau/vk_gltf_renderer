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

// Lens: a cheap wide lens's distortion and colour fringing on the image the tone mapper takes (see lens.hpp).

#include "lens.hpp"

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>

#include "_autogen/lens.slang.h"

// The mapping, as the shader has it
namespace lens_map {
using namespace glm;
using float2 = glm::vec2;
#include "shaders/lens.h.slang"
}  // namespace lens_map

static constexpr const char* kMemoryCategory = "Lens";
static constexpr VkFormat    kFormat         = VK_FORMAT_R16G16B16A16_SFLOAT;

void Lens::init(Resources& resources)
{
  m_device           = resources.allocator.getDevice();
  m_alloc            = &resources.allocator;
  m_appMemoryTracker = &resources.appMemoryTracker;
  m_samplerPool      = &resources.samplerPool;

  // Linear, clamped to the edge: the shared linear sampler repeats, and the fringes read a little
  // past the corners
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

  // Set up with nothing allocated: the first frame through the lens allocates
  NVVK_CHECK(m_out.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Lens out"}));

  m_bindings.addBinding(shaderio::LensBindings::eLensIn, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::LensBindings::eLensOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange        pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(shaderio::LensPushConstant)};
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
      .codeSize = lens_slang_sizeInBytes,
      .pCode    = lens_slang,
  };
  const VkComputePipelineCreateInfo pipelineInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = &shaderInfo,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .pName = "lens"},
      .layout = m_pipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);
}

void Lens::deinit(Resources& /*resources*/)
{
  vkDestroyPipeline(m_device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_pipeline            = {};
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};

  if(m_appMemoryTracker && m_size.width != 0 && m_size.height != 0)
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
  m_out.deinit();
  m_size = {};
  if(m_sampler != VK_NULL_HANDLE)
  {
    m_samplerPool->releaseSampler(m_sampler);
    m_sampler = VK_NULL_HANDLE;
  }
}

void Lens::registerParameters(SettingsRegistry* settings)
{
  // Not persisted: the host sends both with the lighting (fv_lighting.cpp), after every load too
  settings->add({"lensBend", "Lens: barrel distortion, the image drawn in toward its corners (0 = none, 1 = a fisheye action camera's)"},
                &m_settings.bend, Persist::eNo, 0.0f, 1.0f);
  settings->add({"lensFringe", "Lens: colour fringing toward the corners (0 = none)"}, &m_settings.fringe, Persist::eNo, 0.0f, 1.0f);
}

void Lens::onResize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(m_size.width == 0 || m_size.height == 0)  // never allocated: the first frame through the lens does
    return;
  releaseTarget();
  ensureTarget(cmd, size);
}

// The output let go (the device idle)
void Lens::releaseTarget()
{
  if(m_size.width == 0 || m_size.height == 0)
    return;
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
  m_out.deinit();
  NVVK_CHECK(m_out.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Lens out"}));
  m_size = {};
}

// Allocates the output for `size` (nothing in flight uses it: the first frame through the lens, or a
// resize with the device idle)
bool Lens::ensureTarget(VkCommandBuffer cmd, VkExtent2D size)
{
  if(size.width == m_size.width && size.height == m_size.height)
    return true;
  if(m_appMemoryTracker && m_size.width != 0 && m_size.height != 0)
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
  NVVK_CHECK(m_out.update(cmd, size));
  m_size = size;
  if(m_appMemoryTracker)
    m_appMemoryTracker->track(kMemoryCategory, m_out, 1);
  return true;
}

VkDescriptorImageInfo Lens::cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input,
                                     VkExtent2D size, bool straight)
{
  m_bent = 0.0f;
  const shaderio::LensPushConstant push{.outSize = {int(size.width), int(size.height)},
                                        .bend    = straight ? 0.0f : m_settings.bend,
                                        .fringe  = m_settings.fringe};
  if((push.bend <= 0.0f && push.fringe <= 0.0f) || size.width == 0 || size.height == 0 || !ensureTarget(cmd, size))
    return input;
  NVVK_DBG_SCOPE(cmd);

  // The previous frame's tone mapper and HDR output may still read the output
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  VkDescriptorImageInfo image = input;
  image.sampler               = m_sampler;
  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::LensBindings::eLensIn), image);
  writes.append(m_bindings.getWriteSet(shaderio::LensBindings::eLensOut), m_out.getColorStorageImageInfo(0));
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  const VkExtent2D groups = nvvk::getGroupCounts(size, VkExtent2D{LENS_WORKGROUP_SIZE, LENS_WORKGROUP_SIZE});
  vkCmdDispatch(cmd, groups.width, groups.height, 1);
  // Read by bloom, local exposure, the tone mapper and the HDR output
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

  m_bent   = push.bend;
  m_aspect = float(size.width) / float(size.height);

  VkDescriptorImageInfo out = m_out.getColorStorageImageInfo(0);
  out.sampler               = resources.linearSampler;  // as the tone mapper reads its input
  return out;
}

glm::vec2 Lens::rendered(glm::vec2 uv) const
{
  return bent() ? lens_map::lensSource(uv, m_aspect, m_bent, 0.0f, 0.0f) : uv;
}
