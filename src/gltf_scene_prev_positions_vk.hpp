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

#pragma once

//
// #DLSS deformation motion: the vertex positions each deforming primitive had when the previous
// frame was drawn. Skinning, morph targets and host vertex uploads (set_vertices) overwrite
// SceneVk's position buffers in place, so without this a deforming surface gets camera and
// instance motion only, and Ray Reconstruction smears it while a pose, a dial or a clip moves it.
//
// The copy is taken at the start of a frame, before any of those writers runs, and only when they
// ran the frame before: when nothing wrote in between, the copy taken earlier still matches. The
// path tracer reads it through a table of device addresses, one per render primitive (0 = none),
// interpolates the hit triangle's previous corners and reprojects that point (see
// dlssPrevHitPos in shaders/pathtrace_functions.h.slang).
//

#include <span>
#include <vector>

#include <nvvk/resource_allocator.hpp>

#include "gltf_scene_vk.hpp"

namespace nvvkgltf {

class PrevPositionsVk
{
public:
  void init(nvvk::ResourceAllocator* alloc, VkQueue queue, SceneVk::DeferredFreeFunc deferredFree);
  void deinit();

  // Frees every copy and the table; the next cmdCopy allocates again. Call before the scene's
  // vertex buffers go away or are rebuilt.
  void destroyGpuBuffers();

  // Copies the current positions of `renderPrimIDs` into their previous-position buffers. A
  // primitive gets its buffer the first time it is named, or again when its vertex count changed,
  // and is copied then whatever `copyAll` says; `copyAll` false copies only those. The table follows.
  void cmdCopy(VkCommandBuffer cmd, const SceneVk& scnVk, std::span<const int> renderPrimIDs, bool copyAll);

  // Device address of the table: a VkDeviceAddress per render primitive, 0 where none is kept.
  [[nodiscard]] VkDeviceAddress tableAddress() const { return m_table.address; }
  [[nodiscard]] bool            hasBuffers() const { return m_table.buffer != VK_NULL_HANDLE; }

private:
  void freeLater(nvvk::Buffer buffer);

  nvvk::ResourceAllocator*  m_alloc = nullptr;
  VkQueue                   m_queue{};
  SceneVk::DeferredFreeFunc m_deferredFree;
  std::vector<nvvk::Buffer> m_prev;   // Per render primitive; a null handle where none is kept
  nvvk::Buffer              m_table;  // VkDeviceAddress per render primitive
};

}  // namespace nvvkgltf
