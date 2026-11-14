/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// A still that keeps its lights apart (see light_groups.hpp).
//

#include "light_groups.hpp"

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>

// Pre-compiled shader
#include "_autogen/light_mix.slang.h"

static constexpr const char* kMemoryCategory = "Light groups";
static constexpr uint32_t    kImageCount     = 1 + LIGHT_GROUP_COUNT;

void LightGroups::init(Resources& resources)
{
  m_device           = resources.allocator.getDevice();
  m_appMemoryTracker = &resources.appMemoryTracker;

  m_bindings.addBinding(shaderio::LightMixBindings::eMixOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::LightMixBindings::eMixTotal, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::LightMixBindings::eMixGroups, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, LIGHT_GROUP_COUNT, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange        pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .size       = sizeof(shaderio::LightMixPushConstant)};
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
      .codeSize = light_mix_slang_sizeInBytes,
      .pCode    = light_mix_slang,
  };
  const VkComputePipelineCreateInfo pipelineInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = &shaderInfo,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .pName = "main"},
      .layout = m_pipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);

  std::vector<VkFormat> formats(kImageCount, VK_FORMAT_R16G16B16A16_SFLOAT);
  formats[0] = VK_FORMAT_R32G32B32A32_SFLOAT;
  NVVK_CHECK(m_target.init({.device = m_device, .alloc = &resources.allocator, .colorFormats = formats, .debugName = "LightGroups"}));
  m_targetInit = true;
}

void LightGroups::deinit(Resources& resources)
{
  release(resources);
  if(m_targetInit)
    m_target.deinit();
  m_targetInit = false;
  vkDestroyPipeline(m_device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_pipeline            = {};
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};
}

void LightGroups::release(Resources& resources)
{
  if(!m_allocated)
    return;
  // SYNC NOTE: rare (the setting went off); the images may be in flight
  NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_target, kImageCount);
  // the images go; the target stays to be sized again
  m_target.deinit();
  std::vector<VkFormat> formats(kImageCount, VK_FORMAT_R16G16B16A16_SFLOAT);
  formats[0] = VK_FORMAT_R32G32B32A32_SFLOAT;
  NVVK_CHECK(m_target.init({.device = m_device, .alloc = &resources.allocator, .colorFormats = formats, .debugName = "LightGroups"}));
  m_allocated  = false;
  m_holdsStill = false;
}

void LightGroups::ensure(VkCommandBuffer cmd, Resources& resources, bool on)
{
  if(!on)
  {
    release(resources);
    return;
  }
  const VkExtent2D size = resources.gBuffers.getSize();
  if(size.width == 0 || size.height == 0)
    return;
  const VkExtent2D cur = m_allocated ? m_target.getSize() : VkExtent2D{};
  if(cur.width == size.width && cur.height == size.height)
    return;
  if(m_allocated && m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_target, kImageCount);
  NVVK_CHECK(m_target.update(cmd, size));  // an accumulation's first frame writes every pixel
  if(m_appMemoryTracker)
    m_appMemoryTracker->track(kMemoryCategory, m_target, kImageCount);
  m_allocated  = true;
  m_holdsStill = false;
}

void LightGroups::cmdClearGroups(VkCommandBuffer cmd)
{
  const VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
  const VkClearColorValue       none{};
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  for(uint32_t i = 1; i < kImageCount; i++)
    vkCmdClearColorImage(cmd, m_target.getColorImage(i), VK_IMAGE_LAYOUT_GENERAL, &none, 1, &range);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

void LightGroups::cmdAdopt(VkCommandBuffer cmd, VkImage rendered)
{
  const VkExtent2D              size = m_target.getSize();
  const VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
  const VkImageCopy             region{
                  .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
                  .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
                  .extent         = {size.width, size.height, 1},
  };
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  vkCmdCopyImage(cmd, rendered, VK_IMAGE_LAYOUT_GENERAL, m_target.getColorImage(0), VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  const VkClearColorValue none{};
  for(uint32_t i = 1; i < kImageCount; i++)
    vkCmdClearColorImage(cmd, m_target.getColorImage(i), VK_IMAGE_LAYOUT_GENERAL, &none, 1, &range);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

void LightGroups::cmdMix(VkCommandBuffer cmd, const VkDescriptorImageInfo& rendered, const Mix& mix, uint32_t used)
{
  std::array<VkDescriptorImageInfo, LIGHT_GROUP_COUNT> groups;
  for(int i = 0; i < LIGHT_GROUP_COUNT; i++)
    groups[i] = group(i);
  const VkDescriptorImageInfo whole = total();

  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::LightMixBindings::eMixOut), rendered);
  writes.append(m_bindings.getWriteSet(shaderio::LightMixBindings::eMixTotal), whole);
  writes.append(m_bindings.getWriteSet(shaderio::LightMixBindings::eMixGroups), groups.data());
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  shaderio::LightMixPushConstant pushConstant{};
  for(int i = 0; i < LIGHT_GROUP_COUNT; i++)
    pushConstant.mix[i] = mix[i];
  pushConstant.used = used;
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstant), &pushConstant);
  const VkExtent2D groupCounts = nvvk::getGroupCounts(m_target.getSize(), WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groupCounts.width, groupCounts.height, 1);
  m_mixed = mix;
}
