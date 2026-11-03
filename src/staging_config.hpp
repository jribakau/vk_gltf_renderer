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

#include <algorithm>
#include <span>

#include <vulkan/vulkan_core.h>

#include <nvvk/uploader_interface.hpp>

// Shared configuration for `nvvk::FrameUploader` (Resources::staging).
//
// This constant is owned by the app (not by `nvvk::FrameUploader`), because:
//   * `Resources::staging.init()` in renderer.cpp passes it as `blockSize`.
//   * `createSampledImage` in sampled_image.cpp must know it to chunk oversized
//     mip uploads into row-strips (`BufferCircularAllocator::subAllocate`
//     asserts `size <= blockSize`, so any single append must stay under it).
//
// Keeping both call sites bound to this one symbol guarantees they cannot drift.
//
// Rationale for the current values:
//   * Every block the uploader keeps is written end to end as the ring goes round, so
//     all of it stays in system memory: kept blocks are what staging costs at rest.
//     256 MiB blocks kept 512 MiB (the written one and a spare). Now at most
//     (1 + kFrameUploaderKeepBlocks) * 32 MiB = 128 MiB.
//   * A drag stages some 15-25 MB a frame with a few frames in flight, which the kept
//     blocks hold without creating or destroying one per frame.
//   * A load stages far more at once (a strand hair, some 800 MB): blocks past the kept
//     ones are made for it and destroyed once the GPU is done. kFrameUploaderMaxBlocks
//     keeps the most held at once at 16 GiB, as 256 MiB blocks had it.
//   * Big arrays and mips are appended in pieces of at most one block (below, and the
//     row strips of `createSampledImage`), so this is not a cap on a mesh or a texture.
constexpr VkDeviceSize kFrameUploaderBlockSize  = 32ull * 1024 * 1024;
constexpr uint32_t     kFrameUploaderKeepBlocks = 3;
constexpr uint32_t     kFrameUploaderMaxBlocks  = 512;  // 16 GiB of 32 MiB blocks

// Append a large array in pieces that each fit one staging block. A single append must stay
// under kFrameUploaderBlockSize (see above), but the uploader accepts several appends before a
// flush, so arrays that scale with the scene (render nodes, TLAS instances) go through here:
// 136-byte render nodes would otherwise overflow the block beyond ~246K entries.
template <typename T>
VkResult appendBufferChunked(nvvk::CmdUploaderInterface& staging, const nvvk::Buffer& buffer, size_t offset, std::span<const T> data)
{
  constexpr size_t kChunkCount = std::max<size_t>(1, static_cast<size_t>(kFrameUploaderBlockSize) / sizeof(T));
  for(size_t first = 0; first < data.size(); first += kChunkCount)
  {
    const size_t count = std::min(kChunkCount, data.size() - first);
    const VkResult result = staging.appendBuffer(buffer, offset + first * sizeof(T), count * sizeof(T), data.data() + first);
    if(result != VK_SUCCESS)
      return result;
  }
  return VK_SUCCESS;
}
