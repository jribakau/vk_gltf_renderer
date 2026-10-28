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

#include "gizmo_lines_vk.hpp"

#include <algorithm>
#include <cstring>

#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvutils/logger.hpp>

#include "_autogen/gizmo_lines.slang.h"

void LineHelperVk::init(const Resources& res)
{
  if(m_initialized)
    deinit();
  m_app                       = res.app;
  m_alloc                     = res.alloc;
  m_device                    = res.device;
  m_helperDescriptorSetLayout = res.helperDescriptorSetLayout;
  const size_t frames         = std::max<size_t>(1, m_app ? m_app->getFrameCycleSize() : 3);
  m_buffers.assign(frames, {});
  m_capacity.assign(frames, 0);
  createPipeline();
  m_initialized = true;
}

void LineHelperVk::deinit()
{
  if(!m_initialized)
    return;
  vkDeviceWaitIdle(m_device);
  for(nvvk::Buffer& b : m_buffers)
    if(b.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(b);
  for(Retired& r : m_retired)
    m_alloc->destroyBuffer(r.buffer);
  m_retired.clear();
  m_buffers.clear();
  m_capacity.clear();
  vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
  m_vertexShader = VK_NULL_HANDLE;
  vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
  m_fragmentShader = VK_NULL_HANDLE;
  if(m_pipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_pipelineLayout = VK_NULL_HANDLE;
  }
  m_helperDescriptorSetLayout = VK_NULL_HANDLE;
  m_initialized               = false;
}

void LineHelperVk::createPipeline()
{
  using shaderio::lines::LinePushConstants;
  VkPushConstantRange pushRange = {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .size = sizeof(LinePushConstants)};

  VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount             = 1;
  layoutInfo.pSetLayouts                = &m_helperDescriptorSetLayout;
  layoutInfo.pushConstantRangeCount     = 1;
  layoutInfo.pPushConstantRanges        = &pushRange;
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
      .nextStage              = VK_SHADER_STAGE_FRAGMENT_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = gizmo_lines_slang_sizeInBytes,
      .pCode                  = gizmo_lines_slang,
      .pName                  = "vertmain",
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_helperDescriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushRange,
  };
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_vertexShader));
  NVVK_DBG_NAME(m_vertexShader);
  shaderInfo.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
  shaderInfo.nextStage = 0;
  shaderInfo.pName     = "fragmain";
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_fragmentShader));
  NVVK_DBG_NAME(m_fragmentShader);

  m_dynamicPipeline.inputAssemblyState.topology                  = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  m_dynamicPipeline.rasterizationState.cullMode                  = VK_CULL_MODE_NONE;
  m_dynamicPipeline.rasterizationState.lineWidth                 = 1.5f;
  m_dynamicPipeline.rasterizationLineState.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH;
  m_dynamicPipeline.depthStencilState.depthTestEnable            = VK_FALSE;
  m_dynamicPipeline.depthStencilState.depthWriteEnable           = VK_FALSE;
  m_dynamicPipeline.colorBlendEnables[0]                         = VK_TRUE;
  m_dynamicPipeline.colorBlendEquations[0].srcColorBlendFactor   = VK_BLEND_FACTOR_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].dstColorBlendFactor   = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].colorBlendOp          = VK_BLEND_OP_ADD;
  m_dynamicPipeline.colorBlendEquations[0].srcAlphaBlendFactor   = VK_BLEND_FACTOR_ONE;
  m_dynamicPipeline.colorBlendEquations[0].dstAlphaBlendFactor   = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].alphaBlendOp          = VK_BLEND_OP_ADD;
}

void LineHelperVk::renderRaster(VkCommandBuffer  cmd,
                                VkDescriptorSet  helperDescriptorSet,
                                const glm::mat4& viewProj,
                                const glm::mat4& sceneViewProj,
                                bool             isPerspective,
                                const glm::vec2& viewportSize,
                                const glm::vec2& depthBufferSize)
{
  if(!m_initialized)
    return;
  // The frame ring the app waits on before it records a slot again: init() can run before the app
  // has its ring (headless), so the ring is read here, every frame.
  const size_t ring = std::max<size_t>(1, m_app ? m_app->getFrameCycleSize() : 3);
  m_frames++;
  for(size_t i = 0; i < m_retired.size();)
    if(m_frames > m_retired[i].frame + ring + 1)
    {
      m_alloc->destroyBuffer(m_retired[i].buffer);
      m_retired.erase(m_retired.begin() + i);
    }
    else
      i++;
  if(m_vertices.size() < 2)
    return;
  NVVK_DBG_SCOPE(cmd);
  if(m_buffers.size() < ring)
  {
    m_buffers.resize(ring);
    m_capacity.resize(ring, 0);
  }
  // this frame's slot: the GPU finished the frame that last used it before the app came back to it
  m_frame              = m_app ? size_t(m_app->getFrameCycleIndex()) % m_buffers.size() : (m_frame + 1) % m_buffers.size();
  nvvk::Buffer& buffer = m_buffers[m_frame];
  const size_t  count  = m_vertices.size() & ~size_t(1);
  if(m_capacity[m_frame] < count)
  {
    // a frame still in flight may read the old one (another slot's frame, a ring that changed size)
    if(buffer.buffer != VK_NULL_HANDLE)
      m_retired.push_back({buffer, m_frames});
    buffer = {};
    const size_t capacity = std::max<size_t>(256, count * 2);
    NVVK_CHECK(m_alloc->createBuffer(buffer, capacity * sizeof(Vertex), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
    NVVK_DBG_NAME(buffer.buffer);
    m_capacity[m_frame] = capacity;
  }
  if(!buffer.mapping)
    return;
  std::memcpy(buffer.mapping, m_vertices.data(), count * sizeof(Vertex));

  shaderio::lines::LinePushConstants pc{};
  pc.viewProj        = viewProj;
  pc.sceneViewProj   = sceneViewProj;
  pc.viewportSize    = viewportSize;
  pc.depthBufferSize = depthBufferSize;
  pc.isPerspective   = isPerspective ? 1 : 0;
  pc.occludedAlpha   = 0.3f;
  pc.vertices        = reinterpret_cast<Vertex*>(buffer.address);

  m_dynamicPipeline.cmdApplyAllStates(cmd);
  m_dynamicPipeline.cmdApplyDynamicState(cmd, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
  nvvk::GraphicsPipelineState::cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});
  vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &helperDescriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
  vkCmdDraw(cmd, uint32_t(count), 1, 0, 0);
}
