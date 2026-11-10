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
// The moving view's bounce cache, host side (see radiance_cache.hpp).
//

#include "radiance_cache.hpp"

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "shaders/shaderio.h"
#include "shaders/radiance_cache.h.slang"

// Pre-compiled shader
#include "_autogen/radiance_cache_resolve.slang.h"

static constexpr const char* kMemoryCategory = "Bounce cache";
static constexpr const char* kTimerResolve   = "Bounce cache";
// The table's slots, a power of two: a million cells, 56 MB
static constexpr uint32_t kSlots = 1u << 20;

static constexpr VkPipelineStageFlags2 kTraceStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

void RadianceCache::init(Resources& resources)
{
  const VkDevice device = resources.allocator.getDevice();
  m_appMemoryTracker    = &resources.appMemoryTracker;

  // the table is reached by its address alone: no descriptors
  const VkPushConstantRange        pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .size       = sizeof(shaderio::RadianceCacheResolvePushConstant)};
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  NVVK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  const VkShaderModuleCreateInfo shaderInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = radiance_cache_resolve_slang_sizeInBytes,
      .pCode    = radiance_cache_resolve_slang,
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

void RadianceCache::deinit(Resources& resources)
{
  const VkDevice device = resources.allocator.getDevice();
  freeTable(resources);
  vkDestroyPipeline(device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
  m_pipeline       = {};
  m_pipelineLayout = {};
}

void RadianceCache::registerParameters(SettingsRegistry* settings, std::function<void()> reset)
{
  // either changes what a moving frame draws: the view draws again
  nvutils::ParameterBase::CallbackSuccess restart;
  if(reset)
    restart = [reset](const nvutils::ParameterBase* const) { reset(); };
  settings->add({.name            = "ptRadianceCache",
                 .help            = "PathTracer: the moving view's bounce cache (radiance_cache.h.slang). On the frames DLSS "
                                    "draws, a path ends at the first surface it reaches after a rough bounce and takes the "
                                    "light that surface sends back from a world-space cache, which one path in sixteen "
                                    "feeds: a quarter to a third off a frame in a room. Biased (a cell of a few "
                                    "centimetres holds one colour, and follows a changed light over some frames); a still "
                                    "never reads it",
                 .callbackSuccess = restart},
                &m_on, Persist::eNo);
  settings->add({.name            = "ptRadianceCacheCell",
                 .help            = "PathTracer: ptRadianceCache: the width in metres of the cells nearest the eye; cells "
                                    "double with each doubling of the distance past 64 of them",
                 .callbackSuccess = restart},
                &m_cell, Persist::eNo, 0.005F, 1.0F);
  settings->add({.name            = "ptRadianceCacheStill",
                 .help            = "PathTracer (debug): ptRadianceCache on the frames of an accumulation too, which read and "
                                    "feed it as DLSS's do: a converged image with it against one without is the cache's bias",
                 .callbackSuccess = restart},
                &m_still, Persist::eNo);
}

void RadianceCache::freeTable(Resources& resources)
{
  if(m_table.buffer == VK_NULL_HANDLE)
    return;
  NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));  // the last frame may still read or resolve it
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_table.allocation);
  resources.allocator.destroyBuffer(m_table);
  m_table = {};
}

RadianceCache::Frame RadianceCache::prepare(Resources& resources, bool dlss)
{
  // Off: no frame is given the table, and it starts empty when the setting comes back. It is kept until
  // the renderer goes: 56 MB against a wait for the device and a free under whatever still held its address.
  m_served = false;
  if(!m_on)
  {
    m_fresh = true;
    return {};
  }
  if(m_table.buffer == VK_NULL_HANDLE)
  {
    if(!dlss && !m_still)
      return {};
    NVVK_CHECK(resources.allocator.createBuffer(m_table, VkDeviceSize(kSlots) * shaderio::RC_WORDS * sizeof(uint32_t),
                                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                    | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    NVVK_DBG_NAME(m_table.buffer);
    if(m_appMemoryTracker)
      m_appMemoryTracker->track(kMemoryCategory, m_table.allocation);
    m_fresh   = true;
    m_changed = false;
  }
  m_served = true;
  return {.table = reinterpret_cast<uint32_t*>(m_table.address), .slots = int(kSlots), .cell = m_cell, .frame = m_frame++, .still = m_still};
}

void RadianceCache::cmdBeforeTrace(VkCommandBuffer cmd, const Resources& resources)
{
  // Something other than the camera changed since the last frame: a light, a pose, a material. The
  // cells keep their light, as most of a scene's does not change with one edit, but at a few samples'
  // weight (the next resolve's), so what did change is replaced within some frames.
  if(resources.sceneChanges != m_sceneChanges)
    m_changed = true;
  m_sceneChanges = resources.sceneChanges;

  // cleared for the first frame that is given it: when made, when the setting comes back, after a new scene
  if(!m_served || !m_fresh)
    return;
  vkCmdFillBuffer(cmd, m_table.buffer, 0, VK_WHOLE_SIZE, 0);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, kTraceStages, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  m_fresh   = false;
  m_changed = false;
}

void RadianceCache::cmdResolve(VkCommandBuffer cmd, nvvk::ProfilerGpuTimer* profiler)
{
  if(!m_served || m_fresh)
    return;
  // the frame's paths wrote their sums; the pass reads and clears them, and the next frame's paths read its means
  nvvk::cmdMemoryBarrier(cmd, kTraceStages, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  {
    auto timerSection = profiler->cmdFrameSection(cmd, kTimerResolve);
    const shaderio::RadianceCacheResolvePushConstant pushConstant{
        .table     = reinterpret_cast<uint32_t*>(m_table.address),
        .slots     = kSlots,
        .maxWeight = m_changed ? shaderio::RC_CHANGED_WEIGHT : shaderio::RC_MAX_WEIGHT,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstant), &pushConstant);
    vkCmdDispatch(cmd, (kSlots + shaderio::RC_RESOLVE_GROUP - 1) / shaderio::RC_RESOLVE_GROUP, 1, 1);
  }
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kTraceStages, VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  m_changed = false;
}
