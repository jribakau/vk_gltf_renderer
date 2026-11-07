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

// Local exposure: shadows lifted and highlights held, region by region (see local_exposure.hpp).

#include "local_exposure.hpp"

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>

#include "_autogen/local_exposure.slang.h"

// The grid's sizes, as the shader has them
namespace local_exposure_kernels {
using namespace glm;
#include "shaders/local_exposure.h.slang"
}  // namespace local_exposure_kernels

static constexpr const char* kMemoryCategory = "Local exposure";
static constexpr VkFormat    kFormat         = VK_FORMAT_R16G16B16A16_SFLOAT;

void LocalExposure::init(Resources& resources)
{
  m_device           = resources.allocator.getDevice();
  m_alloc            = &resources.allocator;
  m_appMemoryTracker = &resources.appMemoryTracker;
  m_samplerPool      = &resources.samplerPool;

  // Linear, clamped to the edge: the splat's 2x2 taps on an odd side's last column or row would
  // otherwise bring in the frame's far side (the shared linear sampler repeats)
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

  NVVK_CHECK(m_out.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Local exposure out"}));

  m_bindings.addBinding(shaderio::LocalExposureBindings::eLexpImage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::LocalExposureBindings::eLexpGrid, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::LocalExposureBindings::eLexpBlurred, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::LocalExposureBindings::eLexpOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(shaderio::LocalExposurePushConstant)};
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
      .codeSize = local_exposure_slang_sizeInBytes,
      .pCode    = local_exposure_slang,
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
  create("lexpSplat", m_splatPipeline);
  create("lexpBlur", m_blurPipeline);
  create("lexpApply", m_applyPipeline);
}

void LocalExposure::deinit(Resources& /*resources*/)
{
  for(VkPipeline* p : {&m_splatPipeline, &m_blurPipeline, &m_applyPipeline})
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
  if(m_sampler != VK_NULL_HANDLE)
  {
    m_samplerPool->releaseSampler(m_sampler);
    m_sampler = VK_NULL_HANDLE;
  }
}

void LocalExposure::registerParameters(SettingsRegistry* settings)
{
  // Not persisted: the host sends both with the lighting (fv_lighting.cpp), after every load too
  settings->add({"localExposureShadows", "Local exposure: how far dark regions are lifted toward mid grey (0 = none)"},
                &m_settings.shadows, Persist::eNo, 0.0f, 1.0f);
  settings->add({"localExposureHighlights", "Local exposure: how far bright regions are held back toward mid grey (0 = none)"},
                &m_settings.highlights, Persist::eNo, 0.0f, 1.0f);
}

void LocalExposure::onResize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(m_size.width == 0 || m_size.height == 0)  // never allocated: the first frame with it on does
    return;
  releaseTargets();
  ensureTargets(cmd, size);
}

// Everything let go (the device idle, or deinit)
void LocalExposure::releaseTargets()
{
  if(m_size.width == 0 || m_size.height == 0)
    return;
  if(m_appMemoryTracker)
  {
    m_appMemoryTracker->untrack(kMemoryCategory, m_out, 1);
    m_appMemoryTracker->untrack(kMemoryCategory, m_grid.allocation);
    m_appMemoryTracker->untrack(kMemoryCategory, m_blurred.allocation);
  }
  m_out.deinit();
  NVVK_CHECK(m_out.init({.device = m_device, .alloc = m_alloc, .colorFormats = {kFormat}, .debugName = "Local exposure out"}));
  m_alloc->destroyBuffer(m_grid);
  m_alloc->destroyBuffer(m_blurred);
  m_grid    = {};
  m_blurred = {};
  m_size    = {};
  m_cell    = 0;
  m_cells   = glm::ivec2(0);
}

// Allocates the output and the grid for `size` (nothing in flight uses them: the first frame with it
// on, or a resize with the device idle)
bool LocalExposure::ensureTargets(VkCommandBuffer cmd, VkExtent2D size)
{
  if(size.width == m_size.width && size.height == m_size.height)
    return true;
  releaseTargets();
  using namespace local_exposure_kernels;
  m_cell  = lexpCellSize(int(size.height));
  m_cells = glm::ivec2(lexpCells(int(size.width), m_cell), lexpCells(int(size.height), m_cell));
  const VkDeviceSize bytes = VkDeviceSize(m_cells.x) * VkDeviceSize(m_cells.y) * LEXP_BINS * sizeof(glm::vec2);
  NVVK_CHECK(m_out.update(cmd, size));
  NVVK_CHECK(m_alloc->createBuffer(m_grid, bytes, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
  NVVK_DBG_NAME(m_grid.buffer);
  NVVK_CHECK(m_alloc->createBuffer(m_blurred, bytes, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
  NVVK_DBG_NAME(m_blurred.buffer);
  m_size = size;
  if(m_appMemoryTracker)
  {
    m_appMemoryTracker->track(kMemoryCategory, m_out, 1);
    m_appMemoryTracker->track(kMemoryCategory, m_grid.allocation);
    m_appMemoryTracker->track(kMemoryCategory, m_blurred.allocation);
  }
  return true;
}

VkDescriptorImageInfo LocalExposure::cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input,
                                              VkExtent2D size, float exposure, nvvk::ProfilerGpuTimer& timer)
{
  if(!active() || size.width == 0 || size.height == 0 || !ensureTargets(cmd, size))
    return input;
  NVVK_DBG_SCOPE(cmd);
  using namespace local_exposure_kernels;

  shaderio::LocalExposurePushConstant push{};
  push.size       = {int(size.width), int(size.height)};
  push.cells      = {m_cells.x, m_cells.y};
  push.cell       = m_cell;
  push.exposure   = exposure > 0.0f ? exposure : 1.0f;
  push.shadows    = m_settings.shadows;
  push.highlights = m_settings.highlights;

  VkDescriptorImageInfo image = input;
  image.sampler               = m_sampler;  // the splat's 2x2 means; the apply reads at the pixels' centres

  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::LocalExposureBindings::eLexpImage), image);
  writes.append(m_bindings.getWriteSet(shaderio::LocalExposureBindings::eLexpGrid), m_grid);
  writes.append(m_bindings.getWriteSet(shaderio::LocalExposureBindings::eLexpBlurred), m_blurred);
  writes.append(m_bindings.getWriteSet(shaderio::LocalExposureBindings::eLexpOut), m_out.getColorStorageImageInfo(0));
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

  // The previous frame's tone mapper and HDR output may still read the output, its apply the grid;
  // and the image is read here as soon as whatever wrote it is done. In a section of its own: the
  // profiler's sections start at the top of the pipe, so the first after this would count the wait
  {
    auto section = timer.cmdFrameSection(cmd, "Local exposure wait");
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }
  const auto between = [&] {
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
  };

  // The splat: a workgroup a cell
  {
    auto section = timer.cmdFrameSection(cmd, "Local exposure splat");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_splatPipeline);
    vkCmdDispatch(cmd, uint32_t(m_cells.x), uint32_t(m_cells.y), 1);
    between();
  }

  // The blur: a thread a cell's bin
  {
    auto           section   = timer.cmdFrameSection(cmd, "Local exposure blur");
    const uint32_t binsTotal = uint32_t(m_cells.x) * uint32_t(m_cells.y) * uint32_t(LEXP_BINS);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blurPipeline);
    vkCmdDispatch(cmd, (binsTotal + LEXP_BLUR_GROUP - 1) / LEXP_BLUR_GROUP, 1, 1);
    between();
  }

  // The gain on every pixel
  auto             section = timer.cmdFrameSection(cmd, "Local exposure apply");
  const VkExtent2D groups  = nvvk::getGroupCounts(size, VkExtent2D{LEXP_WORKGROUP_SIZE, LEXP_WORKGROUP_SIZE});
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_applyPipeline);
  vkCmdDispatch(cmd, groups.width, groups.height, 1);
  // Read by the tone mapper and the HDR output
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

  VkDescriptorImageInfo out = m_out.getColorStorageImageInfo(0);
  out.sampler               = resources.linearSampler;  // as the tone mapper reads its input
  return out;
}
