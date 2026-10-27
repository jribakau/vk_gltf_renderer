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
// What VulkanGpuBackend (scene_residency_vk.hpp) converts and plans, apart from the device: the
// backend's flags and rows as Vulkan values, glTF samplers, the barriers between its passes, and the
// BLAS scratch batches (invariant I8). Vulkan types, no device calls, so the test build covers it.
//

#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "scene_gpu_backend.hpp"

namespace rs {

VkBufferUsageFlags2                  vkBufferUsage(BufferUsageFlags usage);
VkBuildAccelerationStructureFlagsKHR vkAccelBuildFlags(AccelBuildFlags flags);
// A TLAS row with its BLAS's device address; a hole (no BLAS) references nothing and has mask 0
VkAccelerationStructureInstanceKHR vkTlasInstance(const TlasRow& row, VkDeviceAddress blasAddress);

// glTF sampler values as a Vulkan sampler: unknown values fall back to glTF's defaults, a filter of
// -1 (unset) to trilinear, and the LOD is never clamped (a texture with no sampler still mipmaps)
VkSamplerCreateInfo gltfSamplerCreateInfo(const GpuSamplerDesc& desc);

// The stages and accesses one of the backend's barriers waits on and releases
struct VkBarrierScope
{
  VkPipelineStageFlags2 srcStage  = 0;
  VkAccessFlags2        srcAccess = 0;
  VkPipelineStageFlags2 dstStage  = 0;
  VkAccessFlags2        dstAccess = 0;
};
VkBarrierScope vkBarrierScope(GpuBarrier barrier);

// Where the queued AS builds of one flush go in one scratch buffer. Builds run side by side in a
// batch, each at its own aligned offset; a batch closes when the next build would pass the budget,
// and a barrier separates batches, which reuse the buffer from offset 0. A build larger than the
// budget is a batch of its own.
struct ScratchPlan
{
  struct Place
  {
    uint32_t     batch  = 0;
    VkDeviceSize offset = 0;
  };
  std::vector<Place> places;  // per build, in order
  uint32_t           batches = 0;
  VkDeviceSize       size    = 0;  // the scratch the largest batch needs
};
// `sizes`: each build's scratch size (build or update), in queue order
ScratchPlan planScratch(std::span<const VkDeviceSize> sizes, VkDeviceSize alignment, VkDeviceSize budget);

// What VulkanGpuBackend::timeBuilds measured, the GPU time of each kind of span summed
enum class BuildSpan : uint8_t
{
  eBlas,  // one flush of queued BLAS builds and refits, every batch
  eTlas,  // one TLAS build with its instance upload
};
struct BuildTimes
{
  double   blasMs    = 0.0;
  double   tlasMs    = 0.0;
  uint32_t blasSpans = 0;
  uint32_t tlasSpans = 0;
};
// `stamps`: a begin and an end tick per span, in span order; a span whose end precedes its begin
// counts as 0
BuildTimes sumBuildSpans(std::span<const BuildSpan> spans, std::span<const uint64_t> stamps, double msPerTick);

}  // namespace rs
