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

#include "gltf_scene_prev_positions_vk.hpp"

#include <algorithm>

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

namespace nvvkgltf {

namespace {
constexpr VkBufferUsageFlags2 kUsage =
    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT;
constexpr VkDeviceSize kMaxUpdateBytes = 65536;  // vkCmdUpdateBuffer's limit per call
}  // namespace

void PrevPositionsVk::init(nvvk::ResourceAllocator* alloc, VkQueue queue, SceneVk::DeferredFreeFunc deferredFree)
{
  m_alloc        = alloc;
  m_queue        = queue;
  m_deferredFree = std::move(deferredFree);
}

void PrevPositionsVk::deinit()
{
  destroyGpuBuffers();
  m_alloc = nullptr;
}

void PrevPositionsVk::freeLater(nvvk::Buffer buffer)
{
  if(buffer.buffer == VK_NULL_HANDLE)
    return;
  nvvk::ResourceAllocator* alloc   = m_alloc;
  auto                     cleanup = [=]() mutable { alloc->destroyBuffer(buffer); };
  if(m_deferredFree)
    m_deferredFree(std::move(cleanup));
  else
  {
    vkQueueWaitIdle(m_queue);
    cleanup();
  }
}

void PrevPositionsVk::destroyGpuBuffers()
{
  if(!m_alloc)
    return;
  for(nvvk::Buffer& b : m_prev)
    freeLater(b);
  m_prev.clear();
  freeLater(m_table);
  m_table = {};
}

void PrevPositionsVk::cmdCopy(VkCommandBuffer cmd, const SceneVk& scnVk, std::span<const int> renderPrimIDs, bool copyAll)
{
  if(!m_alloc || renderPrimIDs.empty())
    return;

  const std::vector<SceneVk::VertexBuffers>& vertexBufs = scnVk.vertexBuffers();
  if(m_prev.size() != vertexBufs.size())
    destroyGpuBuffers();  // A different scene, or a rebuilt one: nothing kept is valid
  m_prev.resize(vertexBufs.size());

  bool tableDirty = false;
  if(m_table.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_table, std::max<size_t>(vertexBufs.size(), 1) * sizeof(VkDeviceAddress), kUsage));
    NVVK_DBG_NAME(m_table.buffer);
    tableDirty = true;
  }

  std::vector<int> toCopy;
  for(int named : renderPrimIDs)
  {
    // the members of a vertex set read one position buffer: one copy, kept under the owner's id
    const int id = scnVk.vertexBufferOwner(named);
    if(id < 0 || id >= int(vertexBufs.size()))
      continue;
    const nvvk::Buffer& src = vertexBufs[id].position;
    if(src.buffer == VK_NULL_HANDLE)
      continue;
    nvvk::Buffer& dst   = m_prev[id];
    const bool    fresh = dst.buffer == VK_NULL_HANDLE || dst.bufferSize != src.bufferSize;
    if(fresh)
    {
      freeLater(dst);
      dst = {};
      NVVK_CHECK(m_alloc->createBuffer(dst, src.bufferSize, kUsage));
      NVVK_DBG_NAME(dst.buffer);
      tableDirty = true;
    }
    if((fresh || copyAll) && std::find(toCopy.begin(), toCopy.end(), id) == toCopy.end())
      toCopy.push_back(id);
  }
  if(toCopy.empty() && !tableDirty)
    return;

  // Last frame's writers of the positions (morph / skin compute, staging copies) before this read,
  // and last frame's path-tracer reads of the copies and table before they are overwritten.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);

  for(int id : toCopy)
  {
    const VkBufferCopy region{0, 0, vertexBufs[id].position.bufferSize};
    vkCmdCopyBuffer(cmd, vertexBufs[id].position.buffer, m_prev[id].buffer, 1, &region);
  }

  if(tableDirty)
  {
    // every render primitive points at its owner's copy
    std::vector<VkDeviceAddress> addresses(m_prev.size(), 0);
    for(size_t i = 0; i < m_prev.size(); i++)
    {
      const int owner = scnVk.vertexBufferOwner(static_cast<int>(i));
      addresses[i]    = (owner >= 0 && owner < static_cast<int>(m_prev.size())) ? m_prev[owner].address : m_prev[i].address;
    }
    const VkDeviceSize total = addresses.size() * sizeof(VkDeviceAddress);
    for(VkDeviceSize offset = 0; offset < total; offset += kMaxUpdateBytes)
    {
      const VkDeviceSize size = std::min(kMaxUpdateBytes, total - offset);
      vkCmdUpdateBuffer(cmd, m_table.buffer, offset, size, reinterpret_cast<const uint8_t*>(addresses.data()) + offset);
    }
  }

  // The copies before this frame's writers of the positions, and before the path tracer reads them.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT
                             | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

}  // namespace nvvkgltf
