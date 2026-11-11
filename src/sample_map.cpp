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

// SampleMap: the Samples view, each pixel of a still by the samples it took (see sample_map.hpp).

#include "sample_map.hpp"

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>

#include "_autogen/sample_map.slang.h"

void SampleMap::init(Resources& resources)
{
  m_device = resources.allocator.getDevice();

  m_bindings.addBinding(shaderio::SampleMapBindings::eSampleMapState, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::SampleMapBindings::eSampleMapOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(shaderio::SampleMapPushConstant)};
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
      .codeSize = sample_map_slang_sizeInBytes,
      .pCode    = sample_map_slang,
  };
  const VkComputePipelineCreateInfo pipelineInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = &shaderInfo,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .pName = "sampleMap"},
      .layout = m_pipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);
}

void SampleMap::deinit(Resources& /*resources*/)
{
  vkDestroyPipeline(m_device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_pipeline            = {};
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};
}

void SampleMap::cmdDraw(VkCommandBuffer cmd, const VkDescriptorImageInfo& counts, const VkDescriptorImageInfo& out,
                        VkExtent2D size, int minSamples, int cap)
{
  if(size.width == 0 || size.height == 0)
    return;
  NVVK_DBG_SCOPE(cmd);

  // The path tracer's last pass wrote the counts; the previous frame's readers of the output are done
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  const shaderio::SampleMapPushConstant push{.outSize = {int(size.width), int(size.height)}, .minSamples = minSamples, .cap = cap};
  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::SampleMapBindings::eSampleMapState), counts);
  writes.append(m_bindings.getWriteSet(shaderio::SampleMapBindings::eSampleMapOut), out);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  const VkExtent2D groups = nvvk::getGroupCounts(size, VkExtent2D{SAMPLE_MAP_WORKGROUP_SIZE, SAMPLE_MAP_WORKGROUP_SIZE});
  vkCmdDispatch(cmd, groups.width, groups.height, 1);
}
