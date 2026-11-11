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
// The still's per-pixel Keep detail (see still_keep.hpp).
//

#include "still_keep.hpp"

#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>

// Pre-compiled shader
#include "_autogen/still_keep.slang.h"

void StillKeep::init(VkDevice device)
{
  m_bindings.addBinding(shaderio::StillKeepBindings::eKeepDenoised, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::StillKeepBindings::eKeepRaw, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::StillKeepBindings::eKeepMoments, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange        pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .size       = sizeof(shaderio::StillKeepPushConstant)};
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  NVVK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  const VkShaderModuleCreateInfo shaderInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = still_keep_slang_sizeInBytes,
      .pCode    = still_keep_slang,
  };
  const VkComputePipelineCreateInfo pipelineInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = &shaderInfo,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .pName = "main"},
      .layout = m_pipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);
}

void StillKeep::deinit(VkDevice device)
{
  vkDestroyPipeline(device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_pipeline            = {};
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};
}

void StillKeep::cmdKeep(VkCommandBuffer                        cmd,
                        const VkDescriptorImageInfo&           denoised,
                        const VkDescriptorImageInfo&           raw,
                        const VkDescriptorImageInfo&           moments,
                        VkExtent2D                             size,
                        const shaderio::StillKeepPushConstant& pushConstant)
{
  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::StillKeepBindings::eKeepDenoised), denoised);
  writes.append(m_bindings.getWriteSet(shaderio::StillKeepBindings::eKeepRaw), raw);
  writes.append(m_bindings.getWriteSet(shaderio::StillKeepBindings::eKeepMoments), moments);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstant), &pushConstant);
  const VkExtent2D groups = nvvk::getGroupCounts(size, WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groups.width, groups.height, 1);
}
