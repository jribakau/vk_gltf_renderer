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
// HDR display output: the frame an HDR display is shown (see hdr_output.hpp).
//

#include "hdr_output.hpp"

#include <utility>

#include <nvshaders/tonemap_functions.h.slang>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>

// Pre-compiled shader
#include "_autogen/hdr_output.slang.h"

static constexpr const char* kMemoryCategory = "HDR output";

static_assert(sizeof(shaderio::HdrOutputPushConstant) <= 256, "the push constant space every device of this renderer offers");

void HdrOutput::init(Resources& resources)
{
  const VkDevice device = resources.allocator.getDevice();
  m_appMemoryTracker    = &resources.appMemoryTracker;
  m_resources           = &resources;

  // Allocated by the first frame that is HDR: an SDR display never pays for them
  NVVK_CHECK(m_target.init({.device       = device,
                            .alloc        = &resources.allocator,
                            .colorFormats = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM},
                            .debugName    = "HdrOutput"}));
  // Allocated by the frame an EXR is saved from, let go once it is on disk
  NVVK_CHECK(m_export.init({.device       = device,
                            .alloc        = &resources.allocator,
                            .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT},
                            .debugName    = "HdrOutput export"}));

  m_bindings.addBinding(shaderio::HdrOutputBindings::eHdrInput, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::HdrOutputBindings::eHdrSdr, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::HdrOutputBindings::eHdrSdrKept, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::HdrOutputBindings::eHdrOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange        pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .size       = sizeof(shaderio::HdrOutputPushConstant)};
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
      .codeSize = hdr_output_slang_sizeInBytes,
      .pCode    = hdr_output_slang,
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
    NVVK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr, &pipeline));
    NVVK_DBG_NAME(pipeline);
  };
  create("tonemapHdr", m_tonemapPipeline);
  create("composeHdr", m_composePipeline);
  create("exportLinear", m_exportPipeline);
}

void HdrOutput::deinit(Resources& resources)
{
  const VkDevice device = resources.allocator.getDevice();
  vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
  vkDestroyPipeline(device, m_composePipeline, nullptr);
  vkDestroyPipeline(device, m_exportPipeline, nullptr);
  vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_tonemapPipeline     = {};
  m_composePipeline     = {};
  m_exportPipeline      = {};
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};

  const VkExtent2D size = m_target.getSize();
  if(m_appMemoryTracker && size.width != 0 && size.height != 0)
    m_appMemoryTracker->untrack(kMemoryCategory, m_target, eImageCount);
  m_target.deinit();
  if(m_appMemoryTracker && exportImage() != VK_NULL_HANDLE)
    m_appMemoryTracker->untrack(kMemoryCategory, m_export, 1);
  m_export.deinit();
  m_active = false;
}

void HdrOutput::onResize(VkCommandBuffer cmd, VkExtent2D size)
{
  releaseExport();  // the device is idle here; the next export makes it at the new size
  const VkExtent2D current = m_target.getSize();
  if(current.width == 0 || current.height == 0)  // never allocated: the first HDR frame does
    return;
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_target, eImageCount);
  NVVK_CHECK(m_target.update(cmd, size));
  if(m_appMemoryTracker)
    m_appMemoryTracker->track(kMemoryCategory, m_target, eImageCount);
}

void HdrOutput::setDisplay(const DisplayHdr& display, bool allowed, float peakNits)
{
  m_active = allowed && display.hdr;
  m_levels = present_hdr::levels(display, peakNits);
}

VkImage HdrOutput::image() const
{
  const VkExtent2D size = m_target.getSize();
  return size.width != 0 && size.height != 0 ? m_target.getColorImage(eImageHdr) : VK_NULL_HANDLE;
}

bool HdrOutput::ensureTarget(VkCommandBuffer cmd, Resources& resources)
{
  const VkExtent2D size    = resources.gBuffers.getSize();
  const VkExtent2D current = m_target.getSize();
  if(size.width == 0 || size.height == 0)
    return false;
  if(current.width != size.width || current.height != size.height)
  {
    // The first HDR frame: nothing in flight uses these images yet (a resize goes through onResize)
    if(m_appMemoryTracker && current.width != 0 && current.height != 0)
      m_appMemoryTracker->untrack(kMemoryCategory, m_target, eImageCount);
    NVVK_CHECK(m_target.update(cmd, size));
    if(m_appMemoryTracker)
      m_appMemoryTracker->track(kMemoryCategory, m_target, eImageCount);
  }
  return true;
}

void HdrOutput::dispatch(VkCommandBuffer cmd, Resources& resources, VkPipeline pipeline, VkExtent2D size,
                         const shaderio::HdrOutputPushConstant& push, const VkDescriptorImageInfo& input,
                         const VkDescriptorImageInfo& kept, const VkDescriptorImageInfo& out)
{
  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::HdrOutputBindings::eHdrInput), input);
  writes.append(m_bindings.getWriteSet(shaderio::HdrOutputBindings::eHdrSdr),
                resources.gBuffers.getColorStorageImageInfo(Resources::eImgTonemapped));
  writes.append(m_bindings.getWriteSet(shaderio::HdrOutputBindings::eHdrSdrKept), kept);
  writes.append(m_bindings.getWriteSet(shaderio::HdrOutputBindings::eHdrOut), out);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  const VkExtent2D groups = nvvk::getGroupCounts(size, VkExtent2D{TONEMAP_WORKGROUP_SIZE, TONEMAP_WORKGROUP_SIZE});
  vkCmdDispatch(cmd, groups.width, groups.height, 1);
}

void HdrOutput::cmdTonemap(VkCommandBuffer cmd, Resources& resources, VkExtent2D size, const shaderio::TonemapperData& tm,
                           const VkDescriptorImageInfo& input)
{
  m_tonemapped = false;
  m_kept       = false;
  if(!m_active || !ensureTarget(cmd, resources))
    return;
  NVVK_DBG_SCOPE(cmd);

  shaderio::HdrOutputPushConstant push{.paperWhite = m_levels.paperWhite, .peak = m_levels.peak, .whole = 0, .tm = tm};
  // as nvshaders::Tonemapper::runCompute fills it for the SDR image
  push.tm.inputMatrix = shaderio::getColorCorrectionMatrix(tm.exposure, tm.temperature, tm.tint);

  // The previous frame's present may still copy the HDR image
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  dispatch(cmd, resources, m_tonemapPipeline, size, push, input, m_target.getColorStorageImageInfo(eImageSdrKept),
           m_target.getColorStorageImageInfo(eImageHdr));
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
  m_tm         = push.tm;
  m_tonemapped = true;
}

void HdrOutput::cmdKeepSdr(VkCommandBuffer cmd, Resources& resources)
{
  if(!m_tonemapped)
    return;
  NVVK_DBG_SCOPE(cmd);
  // The tonemapper (and DLSS-NR) wrote the SDR image; the previous frame's compose may still read the copy
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
  const VkExtent2D               size = m_target.getSize();
  const VkImageSubresourceLayers layers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1};
  const VkImageCopy              copy{.srcSubresource = layers, .dstSubresource = layers, .extent = {size.width, size.height, 1}};
  vkCmdCopyImage(cmd, resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                 m_target.getColorImage(eImageSdrKept), VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
  // The overlays write the SDR image next, and the compose reads the copy
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
  m_kept = true;
}

void HdrOutput::cmdFinish(VkCommandBuffer cmd, Resources& resources)
{
  const bool tonemapped = std::exchange(m_tonemapped, false);
  const bool kept       = std::exchange(m_kept, false);
  if(!m_active || (tonemapped && !kept) || !ensureTarget(cmd, resources))
    return;
  NVVK_DBG_SCOPE(cmd);

  const shaderio::HdrOutputPushConstant push{.paperWhite = m_levels.paperWhite, .peak = m_levels.peak, .whole = tonemapped ? 0 : 1, .tm = m_tm};
  // The overlays' writes (compute, a render pass) before the compose reads the SDR image
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  // the input isn't read here, but its binding has to hold an image
  dispatch(cmd, resources, m_composePipeline, m_target.getSize(), push,
           resources.gBuffers.getColorSampleDescriptorImageInfo(Resources::eImgRendered, resources.linearSampler),
           m_target.getColorStorageImageInfo(eImageSdrKept), m_target.getColorStorageImageInfo(eImageHdr));
  // The presenter copies the HDR image next
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

void HdrOutput::cmdExportLinear(VkCommandBuffer cmd, Resources& resources, VkExtent2D size, const shaderio::TonemapperData& tm,
                                const VkDescriptorImageInfo& input)
{
  if(size.width == 0 || size.height == 0)
    return;
  NVVK_DBG_SCOPE(cmd);
  const VkExtent2D current = m_export.getSize();
  if(current.width != size.width || current.height != size.height)
  {
    if(m_appMemoryTracker && current.width != 0 && current.height != 0)
      m_appMemoryTracker->untrack(kMemoryCategory, m_export, 1);
    NVVK_CHECK(m_export.update(cmd, size));
    if(m_appMemoryTracker)
      m_appMemoryTracker->track(kMemoryCategory, m_export, 1);
  }
  shaderio::HdrOutputPushConstant push{.paperWhite = 1.0F, .peak = 1.0F, .whole = 0, .tm = tm};
  push.tm.inputMatrix = shaderio::getColorCorrectionMatrix(tm.exposure, tm.temperature, tm.tint);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  // the export's own image in every storage binding: the display's may not exist on an SDR display
  const VkDescriptorImageInfo out = m_export.getColorStorageImageInfo(0);
  dispatch(cmd, resources, m_exportPipeline, size, push, input, out, out);
  // read back after the frame (GltfRenderer::saveExr)
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
}

VkImage HdrOutput::exportImage() const
{
  const VkExtent2D size = m_export.getSize();
  return size.width != 0 && size.height != 0 ? m_export.getColorImage(0) : VK_NULL_HANDLE;
}

void HdrOutput::releaseExport()
{
  const VkExtent2D size = m_export.getSize();
  if(size.width == 0 || size.height == 0 || !m_resources)
    return;
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_export, 1);
  // let go, and ready for the next export, which allocates at its own size
  m_export.deinit();
  NVVK_CHECK(m_export.init({.device       = m_resources->allocator.getDevice(),
                            .alloc        = &m_resources->allocator,
                            .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT},
                            .debugName    = "HdrOutput export"}));
}
