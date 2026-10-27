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

#include "scene_residency_vk_util.hpp"

#include <algorithm>
#include <cstring>

#include <tinygltf/tiny_gltf.h>

namespace rs {

VkBufferUsageFlags2 vkBufferUsage(BufferUsageFlags usage)
{
  // every buffer is addressed by shaders and written and read by copies (uploads, growth)
  VkBufferUsageFlags2 out = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                            | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;
  if(usage & eBufferVertex)
    out |= VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT;
  if(usage & eBufferIndex)
    out |= VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT;
  if(usage & eBufferStorage)
    out |= VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT;
  if(usage & eBufferAccelInput)
    out |= VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  return out;
}

VkBuildAccelerationStructureFlagsKHR vkAccelBuildFlags(AccelBuildFlags flags)
{
  VkBuildAccelerationStructureFlagsKHR out = 0;
  if(flags & eAccelPreferFastTrace)
    out |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  if(flags & eAccelAllowUpdate)
    out |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  if(flags & eAccelAllowCompaction)
    out |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  return out;
}

VkAccelerationStructureInstanceKHR vkTlasInstance(const TlasRow& row, VkDeviceAddress blasAddress)
{
  VkAccelerationStructureInstanceKHR out{};
  static_assert(sizeof(out.transform.matrix) == sizeof(row.transform));
  std::memcpy(out.transform.matrix, row.transform, sizeof(row.transform));
  out.instanceCustomIndex                    = row.customIndex & 0xFFFFFFu;
  out.instanceShaderBindingTableRecordOffset = 0;
  out.mask                                   = row.blas && blasAddress ? row.mask : 0;
  out.flags                                  = row.flags & 0xFFu;
  out.accelerationStructureReference         = row.blas ? blasAddress : 0;
  return out;
}

// glTF filter/wrap enums -> Vulkan. Unknown values fall back to the glTF default instead of
// throwing -- a malformed asset should degrade, not abort the load.
static VkFilter toVkFilter(int gltfFilter)
{
  switch(gltfFilter)
  {
    case TINYGLTF_TEXTURE_FILTER_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
      return VK_FILTER_NEAREST;
    default:
      return VK_FILTER_LINEAR;
  }
}

// Only minFilter carries mip behaviour in glTF: magFilter is NEAREST or LINEAR and says nothing
// about mips. The two non-mipmapped minFilters (NEAREST / LINEAR) leave mip selection undefined,
// so keep trilinear -- the mip chain exists and using it is what avoids minification aliasing.
static VkSamplerMipmapMode toVkMipmapMode(int gltfMinFilter)
{
  switch(gltfMinFilter)
  {
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    default:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
}

static VkSamplerAddressMode toVkAddressMode(int gltfWrap)
{
  switch(gltfWrap)
  {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  }
}

VkSamplerCreateInfo gltfSamplerCreateInfo(const GpuSamplerDesc& desc)
{
  VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  info.minFilter  = VK_FILTER_LINEAR;
  info.magFilter  = VK_FILTER_LINEAR;
  info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  info.maxLod     = VK_LOD_CLAMP_NONE;
  if(desc.minFilter > -1)
  {
    info.minFilter  = toVkFilter(desc.minFilter);
    info.mipmapMode = toVkMipmapMode(desc.minFilter);
  }
  if(desc.magFilter > -1)
    info.magFilter = toVkFilter(desc.magFilter);
  info.addressModeU = toVkAddressMode(desc.wrapS);
  info.addressModeV = toVkAddressMode(desc.wrapT);
  return info;
}

VkBarrierScope vkBarrierScope(GpuBarrier barrier)
{
  constexpr VkPipelineStageFlags2 kShading = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                                             | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                             | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
  constexpr VkAccessFlags2 kShaderRead = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
  constexpr VkPipelineStageFlags2 kBuild = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
  constexpr VkAccessFlags2 kBuildRW = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  switch(barrier)
  {
    case GpuBarrier::eTransferToShader:
      return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, kShading | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              kShaderRead | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    case GpuBarrier::eTransferToBuild:
      return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, kBuild,
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
    case GpuBarrier::eComputeToBuild:
      return {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, kBuild,
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
    case GpuBarrier::eBuildToBuild:
      return {kBuild, kBuildRW, kBuild, kBuildRW};
    case GpuBarrier::eBuildToTrace:
      return {kBuild, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, kShading,
              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT};
  }
  return {};
}

ScratchPlan planScratch(std::span<const VkDeviceSize> sizes, VkDeviceSize alignment, VkDeviceSize budget)
{
  alignment           = std::max<VkDeviceSize>(alignment, 1);
  const auto  aligned = [&](VkDeviceSize v) { return (v + alignment - 1) / alignment * alignment; };
  ScratchPlan plan;
  plan.places.reserve(sizes.size());
  VkDeviceSize offset = 0;
  for(VkDeviceSize size : sizes)
  {
    const VkDeviceSize need = aligned(std::max<VkDeviceSize>(size, 1));
    if(plan.batches == 0 || (offset > 0 && offset + need > budget))
    {
      plan.batches++;
      offset = 0;
    }
    plan.places.push_back({plan.batches - 1, offset});
    offset += need;
    plan.size = std::max(plan.size, offset);
  }
  return plan;
}

BuildTimes sumBuildSpans(std::span<const BuildSpan> spans, std::span<const uint64_t> stamps, double msPerTick)
{
  BuildTimes times;
  for(size_t i = 0; i < spans.size() && 2 * i + 1 < stamps.size(); i++)
  {
    const uint64_t begin = stamps[2 * i], end = stamps[2 * i + 1];
    const double   ms = end > begin ? double(end - begin) * msPerTick : 0.0;
    if(spans[i] == BuildSpan::eBlas)
    {
      times.blasMs += ms;
      times.blasSpans++;
    }
    else
    {
      times.tlasMs += ms;
      times.tlasSpans++;
    }
  }
  return times;
}

}  // namespace rs
