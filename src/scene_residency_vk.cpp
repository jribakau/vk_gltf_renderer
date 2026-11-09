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

#include "scene_residency_vk.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>

#include <nvutils/file_mapping.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/uploader_interface.hpp>

#include "gltf_image_loader.hpp"
#include "gltf_image_loader.hpp"
#include "sampled_image.hpp"
#include "scene_material_bits.hpp"
#include "shaders/animation_io.h.slang"
#include "shaders/shaderio.h"
#include "_autogen/emissive_texels.comp.slang.h"
#include "staging_config.hpp"

// scene_material_bits.hpp is Vulkan-free; its flags are these
static_assert(nvvkgltf::kInstanceCullDisable == VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR);
static_assert(nvvkgltf::kInstanceForceOpaque == VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR);

namespace rs {

namespace {

constexpr std::string_view kMemBuffers   = "Retained/Buffers";
constexpr std::string_view kMemImages    = "Retained/Images";
constexpr std::string_view kMemBlas      = "Retained/BLAS";
constexpr std::string_view kMemTlas      = "Retained/TLAS";
constexpr std::string_view kMemScratch   = "Retained/Scratch";
constexpr std::string_view kMemInstances = "Retained/Instances";
constexpr std::string_view kMemMicromaps = "Retained/Micromaps";

// Pixels a loader decoded (`component` channels of `bits` bits) as an image the device samples: two
// channels stay two (a normal map's X and Y), one or three are widened to four (grey to RRR1)
bool fromPixels(const ImageSource& source, nvvkgltf::LoadedImageData& out)
{
  const int c = source.component, bytes = source.bits / 8;
  if(source.width <= 0 || source.height <= 0 || c < 1 || c > 4 || (bytes != 1 && bytes != 2))
    return false;
  const size_t texels = size_t(source.width) * size_t(source.height);
  if(source.pixels.size() != texels * size_t(c) * size_t(bytes))
    return false;
  const int outC = c == 2 ? 2 : 4;
  out.size       = {uint32_t(source.width), uint32_t(source.height)};
  out.format     = bytes == 2  ? (outC == 2 ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R16G16B16A16_UNORM) :
                   outC == 2   ? VK_FORMAT_R8G8_UNORM :
                   source.srgb ? VK_FORMAT_R8G8B8A8_SRGB :
                                 VK_FORMAT_R8G8B8A8_UNORM;
  std::vector<char> texelBytes(texels * size_t(outC) * size_t(bytes));
  const auto*       in  = reinterpret_cast<const char*>(source.pixels.data());
  const char        one = char(0xFF);
  for(size_t t = 0; t < texels; t++)
    for(int k = 0; k < outC; k++)
    {
      // the channel this output channel reads: grey repeats, a missing alpha is opaque
      const int from = c == 1 ? (k < 3 ? 0 : -1) : (k < c ? k : -1);
      for(int b = 0; b < bytes; b++)
        texelBytes[(t * outC + k) * bytes + b] = from < 0 ? one : in[(t * c + from) * bytes + b];
    }
  out.mipData = {std::move(texelBytes)};
  return true;
}

}  // namespace

// ---------- Backend: frame and commands ----------

static bool decodeSource(const ImageSource& source, nvvkgltf::LoadedImageData& loaded);

VulkanGpuBackend::VulkanGpuBackend(const Info& info)
    : m_info(info)
{
  m_memory.init(info.alloc);
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &asProps};
  vkGetPhysicalDeviceProperties2(info.physicalDevice, &props);
  m_scratchAlignment = std::max<VkDeviceSize>(asProps.minAccelerationStructureScratchOffsetAlignment, 1);
  m_msPerTick        = double(props.properties.limits.timestampPeriod) * 1e-6;
  m_decodes          = std::make_unique<ImageDecodePool>(
      [takePrefetched = info.takePrefetched](const ImageSource& source, nvvkgltf::LoadedImageData& loaded, bool& prefetched) {
        // a stand-in is decoded smaller than anything decoded ahead
        if(takePrefetched && source.pixels.empty() && source.bytes.empty() && !source.uri.empty() && source.downscale == 1)
          if(std::optional<nvvkgltf::LoadedImageData> taken = takePrefetched(nvutils::pathFromUtf8(source.uri), source.srgb))
          {
            loaded = std::move(*taken);
            nvvkgltf::trimLevels(loaded, source.maxExtent);
            prefetched = true;
            return true;
          }
        return decodeSource(source, loaded);
      },
      ImageDecodePool::defaultThreads());
}

VulkanGpuBackend::~VulkanGpuBackend()
{
  m_decodes.reset();  // what runs finishes; nothing is made of it
  vkDeviceWaitIdle(m_info.device);
  // what retire() handed on and the queue hasn't run: no frame can use it now
  std::map<uint64_t, std::function<void()>> pending = std::move(m_retired->pending);
  m_retired->pending.clear();
  for(auto& [id, closure] : pending)
    closure();
  // and whatever was never retired
  for(auto& [h, b] : m_buffers)
  {
    m_memory.untrack(kMemBuffers, b.buffer.allocation);
    m_info.alloc->destroyBuffer(b.buffer);
  }
  for(auto& [h, image] : m_images)
    m_info.alloc->destroyImage(image);
  for(auto& [h, b] : m_blases)
    m_info.alloc->destroyAcceleration(b.accel);
  for(auto& [h, m] : m_micromaps)
  {
    vkDestroyAccelerationStructureKHR(m_info.device, m.micromap, nullptr);
    for(nvvk::Buffer* b : {&m.storage, &m.data, &m.triangles, &m.scratch})
      m_info.alloc->destroyBuffer(*b);
  }
  for(auto& [h, t] : m_tlases)
  {
    m_info.alloc->destroyAcceleration(t.accel);
    m_info.alloc->destroyBuffer(t.instances);
    m_info.alloc->destroyBuffer(t.scratch);
  }
  m_info.alloc->destroyBuffer(m_blasScratch);
  m_deformPipelines.destroy(m_info.device);
  for(auto& [ticket, job] : m_texelJobs)
  {
    m_info.alloc->destroyBuffer(job.triangles);
    m_info.alloc->destroyBuffer(job.means);
  }
  if(m_texelPipeline)
    vkDestroyPipeline(m_info.device, m_texelPipeline, nullptr);
  if(m_texelLayout)
    vkDestroyPipelineLayout(m_info.device, m_texelLayout, nullptr);
  for(auto& [slot, sampler] : m_samplers)
    m_info.samplerPool->releaseSampler(sampler);
  m_info.alloc->destroyImage(m_white);
  m_info.alloc->destroyImage(m_missing);
  if(m_timestamps)
    vkDestroyQueryPool(m_info.device, m_timestamps, nullptr);
  if(m_compactQueries)
    vkDestroyQueryPool(m_info.device, m_compactQueries, nullptr);
}

void VulkanGpuBackend::record(VkCommandBuffer cmd)
{
  m_cmd = cmd;
  ensureDefaults();
}

void VulkanGpuBackend::finish()
{
  flushBuilds();
  flushUploads();
  m_cmd        = VK_NULL_HANDLE;
  m_timeBuilds = false;
}

// A timed span's begin; the first of a timeBuilds() resets the pool and what was recorded before
bool VulkanGpuBackend::beginSpan()
{
  if(!m_timeBuilds || !m_cmd)
    return false;
  if(!m_timestamps)
  {
    const VkQueryPoolCreateInfo info{.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                     .queryType  = VK_QUERY_TYPE_TIMESTAMP,
                                     .queryCount = 2 * kTimedSpans};
    NVVK_CHECK(vkCreateQueryPool(m_info.device, &info, nullptr, &m_timestamps));
  }
  if(m_spans.size() >= kTimedSpans)
    return false;
  if(m_spans.empty())
    vkCmdResetQueryPool(m_cmd, m_timestamps, 0, 2 * kTimedSpans);
  vkCmdWriteTimestamp(m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_timestamps, uint32_t(2 * m_spans.size()));
  return true;
}

void VulkanGpuBackend::endSpan(BuildSpan span)
{
  vkCmdWriteTimestamp(m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_timestamps, uint32_t(2 * m_spans.size() + 1));
  m_spans.push_back(span);
}

std::optional<BuildTimes> VulkanGpuBackend::buildTimes()
{
  if(m_spans.empty() || m_cmd)
    return std::nullopt;
  std::vector<uint64_t> stamps(2 * m_spans.size());
  if(vkGetQueryPoolResults(m_info.device, m_timestamps, 0, uint32_t(stamps.size()), stamps.size() * sizeof(uint64_t),
                           stamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT)
     != VK_SUCCESS)
    return std::nullopt;
  const BuildTimes times = sumBuildSpans(m_spans, stamps, m_msPerTick);
  m_spans.clear();
  return times;
}

bool VulkanGpuBackend::recording(const char* op) const
{
  if(m_cmd)
    return true;
  LOGE("VulkanGpuBackend::%s outside record() and finish(); not recorded\n", op);
  return false;
}

void VulkanGpuBackend::flushUploads()
{
  if(!m_uploadsPending || !m_cmd)
    return;
  m_info.staging->cmdUploadAppended(m_cmd);
  m_uploadsPending = false;
}

void VulkanGpuBackend::retire(std::function<void()> closure)
{
  const uint64_t id = m_retired->next++;
  m_retired->pending.emplace(id, std::move(closure));
  m_info.submitResourceFree([queue = std::weak_ptr<RetireQueue>(m_retired), id] {
    const std::shared_ptr<RetireQueue> q = queue.lock();
    if(!q)
      return;  // the backend went first and ran it
    auto it = q->pending.find(id);
    if(it == q->pending.end())
      return;
    std::function<void()> closure = std::move(it->second);
    q->pending.erase(it);
    closure();
  });
}

void VulkanGpuBackend::runRetired()
{
  std::map<uint64_t, std::function<void()>> pending = std::move(m_retired->pending);
  m_retired->pending.clear();
  for(auto& [id, closure] : pending)
    closure();
}

void VulkanGpuBackend::waitIdle()
{
  vkDeviceWaitIdle(m_info.device);
}

uint64_t VulkanGpuBackend::averageTexels(std::span<const shaderio::TexelTriangle> triangles)
{
  if(triangles.empty() || !m_info.descriptorSetLayout || !recording("averageTexels"))
    return 0;
  // call order: the queued builds and uploads (the images it samples among them) are recorded first
  flushBuilds();
  flushUploads();
  if(!m_texelPipeline)
  {
    const VkPushConstantRange push{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(shaderio::EmissiveTexelsPushConstant)};
    const VkPipelineLayoutCreateInfo layout{.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .setLayoutCount         = 1,
                                            .pSetLayouts            = &m_info.descriptorSetLayout,
                                            .pushConstantRangeCount = 1,
                                            .pPushConstantRanges    = &push};
    NVVK_CHECK(vkCreatePipelineLayout(m_info.device, &layout, nullptr, &m_texelLayout));
    const VkShaderModuleCreateInfo shader{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                          .codeSize = emissive_texels_comp_slang_sizeInBytes,
                                          .pCode    = emissive_texels_comp_slang};
    const VkComputePipelineCreateInfo pipeline{.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                               .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                          .pNext = &shader,
                                                          .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                          .pName = "main"},
                                               .layout = m_texelLayout};
    NVVK_CHECK(vkCreateComputePipelines(m_info.device, nullptr, 1, &pipeline, nullptr, &m_texelPipeline));
    NVVK_DBG_NAME(m_texelPipeline);
  }
  // small, so both sit in host memory: the triangles written here, the means read back
  TexelJob             job;
  const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  const VkDeviceSize   in = triangles.size_bytes(), out = triangles.size() * sizeof(glm::vec4);
  if(m_info.alloc->createBuffer(job.triangles, in, usage, VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT)
         != VK_SUCCESS
     || m_info.alloc->createBuffer(job.means, out, usage, VMA_MEMORY_USAGE_AUTO,
                                   VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
            != VK_SUCCESS
     || !job.triangles.mapping || !job.means.mapping)
  {
    LOGW("VulkanGpuBackend::averageTexels: no memory for %zu triangles\n", triangles.size());
    m_info.alloc->destroyBuffer(job.triangles);
    m_info.alloc->destroyBuffer(job.means);
    return 0;
  }
  std::memcpy(job.triangles.mapping, triangles.data(), size_t(in));
  vmaFlushAllocation(*m_info.alloc, job.triangles.allocation, 0, VK_WHOLE_SIZE);
  job.count = uint32_t(triangles.size());
  const shaderio::EmissiveTexelsPushConstant pc{.triangles = reinterpret_cast<shaderio::TexelTriangle*>(job.triangles.address),
                                                .means     = reinterpret_cast<glm::vec4*>(job.means.address),
                                                .count     = job.count};
  vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_texelPipeline);
  vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_texelLayout, 0, 1, &m_info.descriptorSet, 0, nullptr);
  vkCmdPushConstants(m_cmd, m_texelLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdDispatch(m_cmd, (job.count + shaderio::EMISSIVE_TEXELS_WORKGROUP - 1) / shaderio::EMISSIVE_TEXELS_WORKGROUP, 1, 1);
  const VkMemoryBarrier2 toHost{.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                                .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
  const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &toHost};
  vkCmdPipelineBarrier2(m_cmd, &dependency);
  // done once no frame in flight is still at it
  retire([done = job.done] { *done = true; });
  const uint64_t ticket = m_nextTexelTicket++;
  m_texelJobs.emplace(ticket, std::move(job));
  return ticket;
}

std::vector<GpuBackend::TexelAverages> VulkanGpuBackend::takeTexelAverages()
{
  std::vector<TexelAverages> out;
  for(auto it = m_texelJobs.begin(); it != m_texelJobs.end();)
  {
    TexelJob& job = it->second;
    if(!*job.done)
    {
      ++it;
      continue;
    }
    vmaInvalidateAllocation(*m_info.alloc, job.means.allocation, 0, VK_WHOLE_SIZE);
    TexelAverages averages{.ticket = it->first};
    averages.means.resize(job.count);
    const auto* means = reinterpret_cast<const glm::vec4*>(job.means.mapping);
    for(uint32_t i = 0; i < job.count; i++)
      averages.means[i] = glm::vec3(means[i]);
    out.push_back(std::move(averages));
    m_info.alloc->destroyBuffer(job.triangles);
    m_info.alloc->destroyBuffer(job.means);
    it = m_texelJobs.erase(it);
  }
  return out;
}

void VulkanGpuBackend::barrier(GpuBarrier barrier)
{
  if(!recording("barrier"))
    return;
  flushBuilds();
  flushUploads();
  const VkBarrierScope   scope = vkBarrierScope(barrier);
  const VkMemoryBarrier2 memory{.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                .srcStageMask  = scope.srcStage,
                                .srcAccessMask = scope.srcAccess,
                                .dstStageMask  = scope.dstStage,
                                .dstAccessMask = scope.dstAccess};
  const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memory};
  vkCmdPipelineBarrier2(m_cmd, &dependency);
}

void VulkanGpuBackend::dispatchDeform(const DeformDispatch& d)
{
  if(!recording("dispatchDeform") || !d.vertexCount)
    return;
  // call order: the queued builds and uploads are recorded before this pass
  flushBuilds();
  flushUploads();
  if(!m_deformPipelines.skin)
    m_deformPipelines.create(m_info.device);
  // a pass over a buffer that found no memory does nothing
  for(const BufferRange* r : {&d.basePositions, &d.baseNormals, &d.baseTangents, &d.outPositions, &d.outNormals, &d.outTangents, &d.positionDeltas,
                              &d.normalDeltas, &d.tangentDeltas, &d.morphWeights, &d.joints, &d.weights, &d.jointMatrices, &d.normalMatrices, &d.joints1,
                              &d.weights1, &d.jointDualQuats, &d.dualQuatShares, &d.rowStart, &d.columns, &d.stencilWeights, &d.stencilOffsets, &d.triangles, &d.uvs, &d.frameX, &d.frameY,
                              &d.outUnit})
    if(auto it = r->buffer ? m_buffers.find(r->buffer.value) : m_buffers.end(); it != m_buffers.end() && !it->second.buffer.buffer)
      return;
  // a range's device address; null for none, as the shaders test
  const auto at = [&](const BufferRange& r) -> uint64_t {
    if(!r.buffer)
      return 0;
    const uint64_t base = bufferAddress(r.buffer);
    if(!base)
      LOGE("VulkanGpuBackend::dispatchDeform: a dead or unknown buffer\n");
    return base ? base + r.offset : 0;
  };
  const uint32_t groups = (d.vertexCount + ANIMATION_WORKGROUP_SIZE - 1) / ANIMATION_WORKGROUP_SIZE;
  if(d.kind == DeformKind::eStencil)
  {
    shaderio::StencilPushConstant pc{};
    pc.baseValues  = reinterpret_cast<glm::vec3*>(at(d.basePositions));
    pc.outValues   = reinterpret_cast<glm::vec3*>(at(d.outPositions));
    pc.rowStart    = reinterpret_cast<uint32_t*>(at(d.rowStart));
    pc.columns     = reinterpret_cast<uint32_t*>(at(d.columns));
    pc.weights     = reinterpret_cast<float*>(at(d.stencilWeights));
    pc.offsets     = reinterpret_cast<glm::vec3*>(at(d.stencilOffsets));
    pc.frameX      = reinterpret_cast<uint32_t*>(at(d.frameX));
    pc.frameY      = reinterpret_cast<uint32_t*>(at(d.frameY));
    pc.outUnit     = reinterpret_cast<uint32_t*>(at(d.outUnit));
    pc.rowCount    = d.vertexCount;
    pc.sourceCount = d.sourceCount;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_deformPipelines.stencil);
    vkCmdPushConstants(m_cmd, m_deformPipelines.stencilLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  }
  else if(d.kind == DeformKind::eNormals)
  {
    shaderio::NormalsPushConstant pc{};
    pc.positions     = reinterpret_cast<glm::vec3*>(at(d.basePositions));
    pc.triangles     = reinterpret_cast<glm::uvec3*>(at(d.triangles));
    pc.triangleStart = reinterpret_cast<uint32_t*>(at(d.rowStart));
    pc.triangleList  = reinterpret_cast<uint32_t*>(at(d.columns));
    pc.outNormals    = reinterpret_cast<glm::vec3*>(at(d.outNormals));
    pc.vertexCount   = d.vertexCount;
    pc.triangleCount = d.triangleCount;
    pc.positionCount = d.sourceCount;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_deformPipelines.normals);
    vkCmdPushConstants(m_cmd, m_deformPipelines.normalsLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  }
  else if(d.kind == DeformKind::eTangents)
  {
    shaderio::TangentsPushConstant pc{};
    pc.positions     = reinterpret_cast<glm::vec3*>(at(d.basePositions));
    pc.normals       = reinterpret_cast<uint32_t*>(at(d.baseNormals));
    pc.uvs           = reinterpret_cast<glm::vec2*>(at(d.uvs));
    pc.triangles     = reinterpret_cast<glm::uvec3*>(at(d.triangles));
    pc.cornerStart   = reinterpret_cast<uint32_t*>(at(d.rowStart));
    pc.corners       = reinterpret_cast<uint32_t*>(at(d.columns));
    pc.outTangents   = reinterpret_cast<uint32_t*>(at(d.outTangents));
    pc.vertexCount   = d.vertexCount;
    pc.triangleCount = d.triangleCount;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_deformPipelines.tangents);
    vkCmdPushConstants(m_cmd, m_deformPipelines.tangentsLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  }
  else if(d.kind == DeformKind::eMorph)
  {
    shaderio::MorphPushConstant pc{};
    pc.basePositions  = reinterpret_cast<glm::vec3*>(at(d.basePositions));
    pc.baseNormals    = reinterpret_cast<uint32_t*>(at(d.baseNormals));
    pc.baseTangents   = reinterpret_cast<uint32_t*>(at(d.baseTangents));
    pc.positionDeltas = reinterpret_cast<glm::vec3*>(at(d.positionDeltas));
    pc.normalDeltas   = reinterpret_cast<glm::vec3*>(at(d.normalDeltas));
    pc.tangentDeltas  = reinterpret_cast<glm::vec3*>(at(d.tangentDeltas));
    pc.morphWeights   = reinterpret_cast<float*>(at(d.morphWeights));
    pc.outPositions   = reinterpret_cast<glm::vec3*>(at(d.outPositions));
    pc.outNormals     = reinterpret_cast<uint32_t*>(at(d.outNormals));
    pc.outTangents    = reinterpret_cast<uint32_t*>(at(d.outTangents));
    pc.vertexCount    = d.vertexCount;
    pc.numTargets     = d.targetCount;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_deformPipelines.morph);
    vkCmdPushConstants(m_cmd, m_deformPipelines.morphLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  }
  else
  {
    shaderio::SkinPushConstant pc{};
    pc.basePositions  = reinterpret_cast<glm::vec3*>(at(d.basePositions));
    pc.baseNormals    = reinterpret_cast<uint32_t*>(at(d.baseNormals));
    pc.baseTangents   = reinterpret_cast<uint32_t*>(at(d.baseTangents));
    pc.weights        = reinterpret_cast<glm::vec4*>(at(d.weights));
    pc.joints         = reinterpret_cast<glm::ivec4*>(at(d.joints));
    pc.jointMatrices  = reinterpret_cast<glm::mat4*>(at(d.jointMatrices));
    pc.normalMatrices = reinterpret_cast<glm::mat3*>(at(d.normalMatrices));
    pc.outPositions   = reinterpret_cast<glm::vec3*>(at(d.outPositions));
    pc.outNormals     = reinterpret_cast<uint32_t*>(at(d.outNormals));
    pc.outTangents    = reinterpret_cast<uint32_t*>(at(d.outTangents));
    pc.joints1        = reinterpret_cast<glm::ivec4*>(at(d.joints1));
    pc.weights1       = reinterpret_cast<glm::vec4*>(at(d.weights1));
    pc.jointDualQuats = reinterpret_cast<glm::vec4*>(at(d.jointDualQuats));
    pc.dualQuatShares = reinterpret_cast<float*>(at(d.dualQuatShares));
    pc.vertexCount    = d.vertexCount;
    pc.numJoints      = d.jointCount;
    pc.dualQuat       = d.dualQuat;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_deformPipelines.skin);
    vkCmdPushConstants(m_cmd, m_deformPipelines.skinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  }
  vkCmdDispatch(m_cmd, groups, 1, 1);
}

// ---------- Buffers ----------

BufferHandle VulkanGpuBackend::createBuffer(const BufferDesc& desc)
{
  BufferEntry entry;
  entry.size = desc.size;
  // 16-byte aligned: TLAS instances need it, and BLAS inputs need 4 (I17)
  const VkResult result = m_info.alloc->createBuffer(entry.buffer, std::max<VkDeviceSize>(desc.size, 16), vkBufferUsage(desc.usage),
                                                     VMA_MEMORY_USAGE_AUTO, {}, 16);
  if(result != VK_SUCCESS && desc.mayFail)
  {
    LOGW("VulkanGpuBackend::createBuffer: no memory for %llu bytes of %s\n", (unsigned long long)desc.size, desc.name ? desc.name : "a buffer");
    m_allocationFailed = true;
    const BufferHandle none{m_nextHandle++};
    m_buffers.emplace(none.value, BufferEntry{});
    return none;
  }
  NVVK_CHECK(result);
  nvvk::DebugUtil::getInstance().setObjectName(entry.buffer.buffer, desc.name ? desc.name : "retained buffer");
  m_memory.track(kMemBuffers, entry.buffer.allocation);
  m_ranges[entry.buffer.address] = entry.size;
  const BufferHandle handle{m_nextHandle++};
  m_buffers.emplace(handle.value, entry);
  return handle;
}

void VulkanGpuBackend::destroyBuffer(BufferHandle handle)
{
  auto it = m_buffers.find(handle.value);
  if(it == m_buffers.end())
  {
    LOGE("VulkanGpuBackend::destroyBuffer: dead or unknown buffer %llu\n", (unsigned long long)handle.value);
    return;
  }
  if(it->second.buffer.buffer)  // or it found no memory
  {
    m_ranges.erase(it->second.buffer.address);
    m_memory.untrack(kMemBuffers, it->second.buffer.allocation);
    m_info.alloc->destroyBuffer(it->second.buffer);
  }
  m_buffers.erase(it);
}

// A large load's build scratch goes once nothing has built for a while: it is only ever needed again
// for another build, and a figure's refits fit what is made for them
void VulkanGpuBackend::beginFrame()
{
  constexpr uint64_t kIdleFrames = 600;
  m_frame++;
  if(m_blasScratch.buffer && m_frame > m_blasScratchUsed + kIdleFrames)
  {
    retire([alloc = m_info.alloc, &memory = m_memory, scratch = m_blasScratch]() mutable {
      memory.untrack(kMemScratch, scratch.allocation);
      alloc->destroyBuffer(scratch);
    });
    m_blasScratch = {};
  }
}

uint64_t VulkanGpuBackend::bufferAddress(BufferHandle handle) const
{
  auto it = m_buffers.find(handle.value);
  return it == m_buffers.end() ? 0 : it->second.buffer.address;
}

uint64_t VulkanGpuBackend::bufferSize(BufferHandle handle) const
{
  auto it = m_buffers.find(handle.value);
  return it == m_buffers.end() ? 0 : it->second.size;
}

VkBuffer VulkanGpuBackend::vkBuffer(BufferHandle handle) const
{
  auto it = m_buffers.find(handle.value);
  return it == m_buffers.end() ? VK_NULL_HANDLE : it->second.buffer.buffer;
}

void VulkanGpuBackend::upload(BufferHandle dst, uint64_t offset, std::span<const std::byte> bytes)
{
  auto it = m_buffers.find(dst.value);
  if(it != m_buffers.end() && !it->second.buffer.buffer)
    return;  // it found no memory
  if(it == m_buffers.end() || offset > it->second.size || bytes.size() > it->second.size - offset)
  {
    LOGE("VulkanGpuBackend::upload: %zu bytes at %llu do not fit buffer %llu\n", bytes.size(),
         (unsigned long long)offset, (unsigned long long)dst.value);
    return;
  }
  if(bytes.empty())
    return;
  flushBuilds();  // the appended uploads always precede the queued builds (call order)
  // no system memory to stage it: the mesh it is for fails as one whose buffers found none does, and is tried again
  if(appendBufferChunked(*m_info.staging, it->second.buffer, size_t(offset), bytes) != VK_SUCCESS)
  {
    LOGW("VulkanGpuBackend::upload: no system memory to stage %zu bytes\n", bytes.size());
    m_allocationFailed = true;
  }
  m_uploadsPending = true;
}

void VulkanGpuBackend::copyBuffer(BufferHandle src, uint64_t srcOffset, BufferHandle dst, uint64_t dstOffset, uint64_t size)
{
  if(!recording("copyBuffer"))
    return;
  auto s = m_buffers.find(src.value), d = m_buffers.find(dst.value);
  if((s != m_buffers.end() && !s->second.buffer.buffer) || (d != m_buffers.end() && !d->second.buffer.buffer))
    return;  // one found no memory
  if(s == m_buffers.end() || d == m_buffers.end() || srcOffset + size > s->second.size || dstOffset + size > d->second.size)
  {
    LOGE("VulkanGpuBackend::copyBuffer: a dead buffer or a range out of bounds\n");
    return;
  }
  flushBuilds();
  flushUploads();
  if(size == 0)
    return;
  const VkBufferCopy region{.srcOffset = srcOffset, .dstOffset = dstOffset, .size = size};
  vkCmdCopyBuffer(m_cmd, s->second.buffer.buffer, d->second.buffer.buffer, 1, &region);
}

bool VulkanGpuBackend::inLiveBuffer(VkDeviceAddress address, VkDeviceSize size) const
{
  auto it = m_ranges.upper_bound(address);
  if(it == m_ranges.begin())
    return false;
  --it;
  return address - it->first <= it->second && size <= it->second - (address - it->first);
}

// ---------- Images and descriptors ----------

// An image source decoded on the calling thread (any): pixels, encoded bytes, or a file
static bool decodeSource(const ImageSource& source, nvvkgltf::LoadedImageData& loaded)
{
  if(!source.pixels.empty())
    return fromPixels(source, loaded);
  if(!source.bytes.empty())
    return nvvkgltf::loadFromMemory(loaded, source.bytes.data(), source.bytes.size(), source.srgb, 0, source.maxExtent, source.downscale);
  if(source.uri.empty())
    return false;
  nvutils::FileReadMapping file;
  if(file.open(nvutils::pathFromUtf8(source.uri)))
    return nvvkgltf::loadFromMemory(loaded, file.data(), file.size(), source.srgb, 0, source.maxExtent, source.downscale);
  LOGW("Image '%s' (%s) could not be opened\n", source.name.c_str(), source.uri.c_str());
  return false;
}

std::optional<ImageChain> VulkanGpuBackend::imageChain(const ImageSource& source)
{
  if(!source.pixels.empty())
    return std::nullopt;
  if(!source.bytes.empty())
    return nvvkgltf::readImageChain(source.bytes.data(), source.bytes.size());
  nvutils::FileReadMapping file;
  if(source.uri.empty() || !file.open(nvutils::pathFromUtf8(source.uri)))
    return std::nullopt;
  return nvvkgltf::readImageChain(file.data(), file.size());
}

ImageHandle VulkanGpuBackend::createImage(const ImageSource& source, ImageInfo& info)
{
  info = {};
  if(!recording("createImage"))
    return {};
  nvvkgltf::LoadedImageData loaded;
  if(!decodeSource(source, loaded))
    return {};
  return createFromDecoded(source, loaded, info);
}

void VulkanGpuBackend::dropPrefetch(const std::string& uri, bool srgb)
{
  if(m_info.takePrefetched)
    m_info.takePrefetched(nvutils::pathFromUtf8(uri), srgb);
}

uint64_t VulkanGpuBackend::startDecode(const ImageSource& source)
{
  return m_decodes->start(HeldImageSource(source));
}

std::vector<GpuBackend::MadeImage> VulkanGpuBackend::takeDecodes(uint64_t budget)
{
  std::vector<MadeImage> made;
  if(!recording("takeDecodes"))
    return made;
  for(ImageDecodePool::Done& done : m_decodes->take(budget))
  {
    MadeImage& m = made.emplace_back();
    m.ticket     = done.ticket;
    m.image      = done.ok ? createFromDecoded(done.source.source, done.loaded, m.info) : ImageHandle{};
    done.loaded  = {};  // its texels uploaded: let them go before the next
    m_prefetchTaken += done.prefetched;
  }
  return made;
}

void VulkanGpuBackend::cancelDecode(uint64_t ticket)
{
  m_decodes->cancel(ticket);
}

bool VulkanGpuBackend::decodesReady() const
{
  return m_decodes->ready();
}

// An image from its decoded mips, uploaded through the frame uploader in the command buffer
ImageHandle VulkanGpuBackend::createFromDecoded(const ImageSource& source, nvvkgltf::LoadedImageData& loaded, ImageInfo& info)
{
  if(loaded.mipData.empty())
    return {};
  flushBuilds();  // createSampledImage records the appended uploads with its own
  nvvk::Image image;
  if(!nvvkgltf::createSampledImage(m_cmd, *m_info.staging, *m_info.alloc, m_info.physicalDevice, m_info.generateMipmaps,
                                   loaded.format, loaded.size, loaded.mipData, loaded.componentMapping, source.name, image, &info.outOfMemory))
  {
    // its staging found no system memory part way: what was staged copies into it in this frame
    if(image.image)
      retire([alloc = m_info.alloc, image]() mutable { alloc->destroyImage(image); });
    return {};
  }
  m_uploadsPending = false;
  m_memory.track(kMemImages, image.allocation);
  info.width         = loaded.size.width;
  info.height        = loaded.size.height;
  info.twoChannel    = nvvkgltf::formatColorComponentCount(loaded.format) == 2;
  info.meanLuminance = nvvkgltf::imageMeanLuminance(loaded);
  VmaAllocationInfo allocation{};
  vmaGetAllocationInfo(*m_info.alloc, image.allocation, &allocation);
  info.bytes        = allocation.size;
  info.storedLevels = loaded.mipData.size() > 1;
  info.levels       = image.mipLevels;
  const ImageHandle handle{m_nextHandle++};
  m_images.emplace(handle.value, image);
  m_imageMade.emplace(handle.value, ImageMade{loaded.componentMapping, source.name, info});
  return handle;
}

ImageHandle VulkanGpuBackend::copyImageLevels(ImageHandle handle, uint32_t dropLevels, ImageInfo& info)
{
  info = {};
  const auto from = m_images.find(handle.value);
  const auto made = m_imageMade.find(handle.value);
  if(from == m_images.end() || made == m_imageMade.end() || !recording("copyImageLevels"))
    return {};
  const nvvk::Image source = from->second;
  if(dropLevels >= source.mipLevels)
    return {};
  VkImageCreateInfo create = DEFAULT_VkImageCreateInfo;
  create.extent            = {std::max(source.extent.width >> dropLevels, 1u), std::max(source.extent.height >> dropLevels, 1u), 1};
  create.format            = source.format;
  create.mipLevels         = source.mipLevels - dropLevels;
  create.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  VkImageViewCreateInfo view = DEFAULT_VkImageViewCreateInfo;
  view.components            = made->second.mapping;
  // memory of its own when it is large, as a decoded one has (createSampledImage)
  constexpr uint64_t kOwnMemoryBytes = 4 << 20;
  const VmaAllocationCreateInfo allocation{.flags = (made->second.info.bytes >> (2 * dropLevels)) >= kOwnMemoryBytes ? VmaAllocationCreateFlags(VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) : VmaAllocationCreateFlags(0),
                                           .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
  nvvk::Image copy;
  if(m_info.alloc->createImage(copy, create, view, allocation) != VK_SUCCESS)
  {
    info.outOfMemory = true;
    return {};
  }
  nvvk::DebugUtil::getInstance().setObjectName(copy.image, made->second.name);
  flushBuilds();
  nvvk::cmdImageMemoryBarrier(m_cmd, {copy.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});
  nvvk::cmdImageMemoryBarrier(m_cmd, {source.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
  std::vector<VkImageCopy> regions;
  for(uint32_t level = 0; level < create.mipLevels; level++)
    regions.push_back({.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level + dropLevels, 0, 1},
                       .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                       .extent         = {std::max(create.extent.width >> level, 1u), std::max(create.extent.height >> level, 1u), 1}});
  vkCmdCopyImage(m_cmd, source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), regions.data());
  nvvk::cmdImageMemoryBarrier(m_cmd, {source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
  copy.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  nvvk::cmdImageMemoryBarrier(m_cmd, {copy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
  m_memory.track(kMemImages, copy.allocation);
  info        = made->second.info;
  info.width  = create.extent.width;
  info.height = create.extent.height;
  info.levels = create.mipLevels;
  VmaAllocationInfo allocated{};
  vmaGetAllocationInfo(*m_info.alloc, copy.allocation, &allocated);
  info.bytes = allocated.size;
  const ImageHandle out{m_nextHandle++};
  m_images.emplace(out.value, copy);
  // (a name and a mapping copied, before anything new moves the map)
  ImageMade kept = made->second;
  kept.info      = info;
  m_imageMade.emplace(out.value, std::move(kept));
  return out;
}

void VulkanGpuBackend::destroyImage(ImageHandle handle)
{
  auto it = m_images.find(handle.value);
  if(it == m_images.end())
  {
    LOGE("VulkanGpuBackend::destroyImage: dead or unknown image %llu\n", (unsigned long long)handle.value);
    return;
  }
  // The invariant a frame's reads rest on: no texture slot shows an image that is gone. A slot is
  // rewritten before its image is retired, so one still showing it here names the path that didn't;
  // the slot gets the missing image, a valid read in place of a fault once the driver takes the pages.
  for(auto& [slot, shown] : m_slotImage)
    if(shown == handle.value)
    {
      const auto made = m_imageMade.find(handle.value);
      LOGE("[gpu] An image (%s, %ux%u) is destroyed while texture slot %u still shows it: whatever rewrote the slot missed it; it shows the missing image now\n",
           made != m_imageMade.end() ? made->second.name.c_str() : "unnamed",
           made != m_imageMade.end() ? made->second.info.width : 0, made != m_imageMade.end() ? made->second.info.height : 0, slot);
      writeImageDescriptor(slot, m_missing);
      shown = 0;
    }
  m_memory.untrack(kMemImages, it->second.allocation);
  m_info.alloc->destroyImage(it->second);
  m_images.erase(it);
  m_imageMade.erase(handle.value);
}

void VulkanGpuBackend::ensureDefaults()
{
  if(m_defaultsUploaded)
    return;
  m_defaultsUploaded = true;
  for(auto [image, color, name] : {std::tuple{&m_white, std::array<uint8_t, 4>{255, 255, 255, 255}, "Retained default"},
                                   std::tuple{&m_missing, std::array<uint8_t, 4>{255, 0, 255, 255}, "Retained missing"}})
  {
    VkImageCreateInfo create = DEFAULT_VkImageCreateInfo;
    create.extent            = {1, 1, 1};
    create.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    NVVK_CHECK(m_info.alloc->createImage(*image, create, DEFAULT_VkImageViewCreateInfo));
    NVVK_CHECK(m_info.staging->appendImage(*image, std::span<const uint8_t>(color.data(), 4), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    nvvk::DebugUtil::getInstance().setObjectName(image->image, name);
  }
  m_uploadsPending = true;
  flushUploads();
  writeTextureDescriptor(0, {});
  writeSamplerDescriptor(0, {});
}

void VulkanGpuBackend::writeImageDescriptor(uint32_t slot, const nvvk::Image& image)
{
  nvvk::WriteSetContainer write;
  write.append(m_info.bindings->getWriteSet(m_info.textureBinding, m_info.descriptorSet, slot, 1), image);
  vkUpdateDescriptorSets(m_info.device, write.size(), write.data(), 0, nullptr);
}

void VulkanGpuBackend::writeTextureDescriptor(uint32_t slot, ImageHandle handle)
{
  if(slot >= m_info.maxTextures)
  {
    LOGE("VulkanGpuBackend: texture slot %u past the descriptor array's %u\n", slot, m_info.maxTextures);
    return;
  }
  auto it = m_images.find(handle.value);
  if(handle && it == m_images.end())
    LOGE("VulkanGpuBackend::writeTextureDescriptor: slot %u given a dead or unknown image\n", slot);
  writeImageDescriptor(slot, it != m_images.end() ? it->second : slot == 0 ? m_white : m_missing);
  m_slotImage[slot] = it != m_images.end() ? handle.value : 0;
}

void VulkanGpuBackend::writeSamplerDescriptor(uint32_t slot, const GpuSamplerDesc& desc)
{
  if(slot >= m_info.maxSamplers)
  {
    LOGE("VulkanGpuBackend: sampler slot %u past the descriptor array's %u\n", slot, m_info.maxSamplers);
    return;
  }
  VkSampler sampler = VK_NULL_HANDLE;
  NVVK_CHECK(m_info.samplerPool->acquireSampler(sampler, gltfSamplerCreateInfo(desc)));
  // a slot is written again only after its last user retired, so no frame reads the one it held
  if(auto old = m_samplers.find(slot); old != m_samplers.end())
    m_info.samplerPool->releaseSampler(old->second);
  m_samplers[slot] = sampler;
  nvvk::WriteSetContainer write;
  const VkDescriptorImageInfo info{.sampler = sampler, .imageView = VK_NULL_HANDLE, .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  write.append(m_info.bindings->getWriteSet(m_info.samplerBinding, m_info.descriptorSet, slot, 1), info);
  vkUpdateDescriptorSets(m_info.device, write.size(), write.data(), 0, nullptr);
}

// ---------- Acceleration structures ----------

void VulkanGpuBackend::checkInput(const BlasInput& input, const char* op) const
{
  const VkDeviceSize vertexBytes = input.vertexCount ? VkDeviceSize(input.vertexCount - 1) * input.vertexStride + 12 : 0;
  if(!inLiveBuffer(input.vertexAddress, vertexBytes) || !inLiveBuffer(input.indexAddress, VkDeviceSize(input.indexCount) * 4))
    LOGE("VulkanGpuBackend::%s: its positions or indices are not inside a live buffer\n", op);
  if(input.micromap && input.micromapIndexSize
     && !inLiveBuffer(input.micromapIndexAddress, VkDeviceSize(input.indexCount / 3) * input.micromapIndexSize))
    LOGE("VulkanGpuBackend::%s: its micromap indices are not inside a live buffer\n", op);
}

nvvk::AccelerationStructureGeometryInfo VulkanGpuBackend::geometryOf(const BlasInput& input, const void* micromap) const
{
  // float3 positions, uint32 indices; and the micromap
  // link, the entry's own
  nvvk::AccelerationStructureGeometryInfo out;
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.pNext                    = micromap;
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = input.vertexAddress;
  triangles.vertexStride             = input.vertexStride;
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = input.indexAddress;
  triangles.maxVertex                = input.vertexCount ? input.vertexCount - 1 : 0;
  out.geometry.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  out.geometry.geometryType          = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  // one any-hit a triangle, as shadowCrossSurface and stochastic opacity need: dropped on every geometry
  // it traced no faster (7TEST, heavyScene at 1080p, 2026-11-08), the moving view 1-2% slower
  out.geometry.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | (input.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0);
  out.geometry.geometry.triangles = triangles;
  out.rangeInfo.primitiveCount    = input.indexCount / 3;
  return out;
}

AccelHandle VulkanGpuBackend::buildBlas(const BlasInput& input, AccelBuildFlags flags)
{
  BlasEntry entry;
  if(!input.vertexAddress || !input.indexAddress)  // over a buffer that found no memory
  {
    m_allocationFailed = true;
    const AccelHandle none{m_nextHandle++};
    m_blases.emplace(none.value, std::move(entry));
    return none;
  }
  checkInput(input, "buildBlas");
  if(input.micromap)
  {
    auto mm = m_micromaps.find(input.micromap.value);
    if(mm == m_micromaps.end())
      LOGE("VulkanGpuBackend::buildBlas: a dead or unknown micromap; built without it\n");
    else
    {
      // the micromap is linked (no indices: UINT16 at
      // address 0, the triangle's own index)
      entry.micromap = std::make_unique<VkAccelerationStructureTrianglesOpacityMicromapKHR>(
          VkAccelerationStructureTrianglesOpacityMicromapKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR});
      entry.micromap->indexType    = input.micromapIndexSize == 1 ? VK_INDEX_TYPE_UINT8_EXT :
                                     input.micromapIndexSize == 4 ? VK_INDEX_TYPE_UINT32 :
                                                                    VK_INDEX_TYPE_UINT16;
      entry.micromap->indexBuffer  = input.micromapIndexAddress;
      entry.micromap->indexStride  = input.micromapIndexSize;
      entry.micromap->baseTriangle = input.micromapBaseTriangle;
      entry.micromap->micromap     = mm->second.micromap;
    }
  }
  entry.data.asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  entry.data.addGeometry(geometryOf(input, entry.micromap.get()));
  entry.data.finalizeGeometry(m_info.device, vkAccelBuildFlags(flags));
  if(m_info.alloc->createAcceleration(entry.accel, entry.data.makeCreateInfo()) != VK_SUCCESS)
  {
    LOGW("VulkanGpuBackend::buildBlas: no memory for %u triangles\n", input.indexCount / 3);
    m_allocationFailed = true;
    entry.accel        = {};
    const AccelHandle none{m_nextHandle++};
    m_blases.emplace(none.value, std::move(entry));
    return none;
  }
  NVVK_DBG_NAME(entry.accel.accel);
  m_memory.track(kMemBlas, entry.accel.buffer.allocation);
  if(flags & eAccelAllowCompaction)
  {
    if(!m_compactQueries)
    {
      const VkQueryPoolCreateInfo info{.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                       .queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                                       .queryCount = kCompactQueries};
      NVVK_CHECK(vkCreateQueryPool(m_info.device, &info, nullptr, &m_compactQueries));
      for(uint32_t q = kCompactQueries; q-- > 0;)
        m_freeCompactQueries.push_back(q);
    }
    // none free: it stays as built
    if(!m_freeCompactQueries.empty())
    {
      entry.compactQuery = int32_t(m_freeCompactQueries.back());
      m_freeCompactQueries.pop_back();
      entry.sizeWritten = std::make_shared<bool>(false);
    }
  }
  const AccelHandle handle{m_nextHandle++};
  m_blases.emplace(handle.value, std::move(entry));
  m_queued.push_back({handle.value, false});
  return handle;
}

void VulkanGpuBackend::refitBlas(AccelHandle handle, const BlasInput& input)
{
  auto it = m_blases.find(handle.value);
  if(it != m_blases.end() && !it->second.accel.accel)
    return;  // it found no memory
  if(it == m_blases.end() || !(it->second.data.buildInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR))
  {
    LOGE("VulkanGpuBackend::refitBlas: a dead BLAS or one built without eAccelAllowUpdate\n");
    return;
  }
  checkInput(input, "refitBlas");
  // the same addresses and index count (the residency builds again otherwise); the vertex count may not move either
  it->second.data.asGeometry[0] = geometryOf(input, it->second.micromap.get()).geometry;
  m_queued.push_back({handle.value, true});
}

AccelHandle VulkanGpuBackend::buildMicromap(const MicromapInput& input)
{
  if(!m_info.opacityMicromaps)
  {
    LOGE("VulkanGpuBackend::buildMicromap: the device does not build opacity micromaps\n");
    return {};
  }
  flushBuilds();  // its inputs' uploads come after what was queued before (call order)
  MicromapEntry entry;
  entry.triangleStride = input.triangleStride;
  for(const MicromapUsage& u : input.usages)
    entry.usages.push_back(
        {.count = u.count, .subdivisionLevel = u.subdivisionLevel, .format = VkOpacityMicromapFormatKHR(u.format)});

  // the data and triangle-array addresses must be multiples of 128
  // (VUID-vkCmdBuildAccelerationStructuresKHR-micromap-11552)
  constexpr VkDeviceSize kInputAlignment = 128;
  const auto inputBuffer = [&](nvvk::Buffer& buffer, std::span<const unsigned char> bytes, const char* name) {
    NVVK_CHECK(m_info.alloc->createBuffer(buffer, std::max<VkDeviceSize>(bytes.size(), 4),
                                          VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                              | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                          VMA_MEMORY_USAGE_AUTO, {}, kInputAlignment));
    nvvk::DebugUtil::getInstance().setObjectName(buffer.buffer, name);
    m_memory.track(kMemMicromaps, buffer.allocation);
    if(!bytes.empty())
      NVVK_CHECK(appendBufferChunked(*m_info.staging, buffer, 0, bytes));
  };
  inputBuffer(entry.data, input.data, "micromap data");
  inputBuffer(entry.triangles, input.triangles, "micromap triangles");
  m_uploadsPending = true;

  // the sizes from the usage histogram and the stride; the addresses are filled when it is recorded
  VkAccelerationStructureGeometryMicromapDataKHR micromapData{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR};
  micromapData.usageCountsCount    = uint32_t(entry.usages.size());
  micromapData.pUsageCounts        = entry.usages.data();
  micromapData.triangleArrayStride = entry.triangleStride;
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.pNext        = &micromapData;
  geometry.geometryType = VK_GEOMETRY_TYPE_MICROMAP_KHR;
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR;
  buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries   = &geometry;
  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  // micromaps take no primitive counts (VUID-vkGetAccelerationStructureBuildSizesKHR-pMaxPrimitiveCounts-11613)
  vkGetAccelerationStructureBuildSizesKHR(m_info.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
                                          nullptr, &sizes);

  // its address a multiple of 256 (VUID-VkAccelerationStructureCreateInfo2KHR-addressRange-11605)
  NVVK_CHECK(m_info.alloc->createBuffer(entry.storage, std::max<VkDeviceSize>(sizes.accelerationStructureSize, 4),
                                        VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                        VMA_MEMORY_USAGE_AUTO, {}, 256));
  NVVK_DBG_NAME(entry.storage.buffer);
  m_memory.track(kMemMicromaps, entry.storage.allocation);
  NVVK_CHECK(m_info.alloc->createBuffer(entry.scratch, std::max<VkDeviceSize>(sizes.buildScratchSize, 4),
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                        VMA_MEMORY_USAGE_AUTO, {}, m_scratchAlignment));
  NVVK_DBG_NAME(entry.scratch.buffer);
  m_memory.track(kMemScratch, entry.scratch.allocation);
  // a micromap is created only through the address-range entry point
  // (VUID-VkAccelerationStructureCreateInfoKHR-type-11600)
  VkAccelerationStructureCreateInfo2KHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_2_KHR};
  create.addressRange = {entry.storage.address, sizes.accelerationStructureSize};
  create.type         = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR;
  NVVK_CHECK(vkCreateAccelerationStructure2KHR(m_info.device, &create, nullptr, &entry.micromap));
  NVVK_DBG_NAME(entry.micromap);

  const AccelHandle handle{m_nextHandle++};
  m_micromaps.emplace(handle.value, std::move(entry));
  m_queuedMicromaps.push_back(handle.value);
  return handle;
}

// The queued micromap builds, behind their inputs' uploads (flushed by the caller), and a barrier
// before the BLAS builds that link them; then their inputs and scratch retired
void VulkanGpuBackend::recordMicromaps()
{
  if(m_queuedMicromaps.empty())
    return;
  // build inputs are read by the build as any AS input: SHADER_READ
  nvvk::cmdMemoryBarrier(m_cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  for(uint64_t h : m_queuedMicromaps)
  {
    MicromapEntry& mm = m_micromaps.at(h);
    VkAccelerationStructureGeometryMicromapDataKHR micromapData{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR};
    micromapData.usageCountsCount    = uint32_t(mm.usages.size());
    micromapData.pUsageCounts        = mm.usages.data();
    micromapData.data                = mm.data.address;
    micromapData.triangleArray       = mm.triangles.address;
    micromapData.triangleArrayStride = mm.triangleStride;
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.pNext        = &micromapData;
    geometry.geometryType = VK_GEOMETRY_TYPE_MICROMAP_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type                      = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR;
    buildInfo.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure  = mm.micromap;
    buildInfo.geometryCount             = 1;
    buildInfo.pGeometries               = &geometry;
    buildInfo.scratchData.deviceAddress = mm.scratch.address;
    // micromaps take no build ranges (VUID-vkCmdBuildAccelerationStructuresKHR-ppBuildRangeInfos-11544)
    const VkAccelerationStructureBuildRangeInfoKHR* noRanges = nullptr;
    vkCmdBuildAccelerationStructuresKHR(m_cmd, 1, &buildInfo, &noRanges);
    retire([alloc = m_info.alloc, &memory = m_memory, data = mm.data, triangles = mm.triangles, scratch = mm.scratch]() mutable {
      memory.untrack(kMemMicromaps, data.allocation);
      memory.untrack(kMemMicromaps, triangles.allocation);
      memory.untrack(kMemScratch, scratch.allocation);
      alloc->destroyBuffer(data);
      alloc->destroyBuffer(triangles);
      alloc->destroyBuffer(scratch);
    });
    mm.data = mm.triangles = mm.scratch = {};
  }
  m_queuedMicromaps.clear();
  // the BLAS builds read the built micromaps
  nvvk::accelerationStructureBarrier(m_cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

// The compacted size each BLAS built for compaction comes out at, once its build has run (the
// builds ahead are recorded); readable once the frame has retired
void VulkanGpuBackend::writeCompactedSizes(std::span<const uint64_t> blases)
{
  nvvk::accelerationStructureBarrier(m_cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  for(uint64_t h : blases)
  {
    BlasEntry& blas = m_blases.at(h);
    vkCmdResetQueryPool(m_cmd, m_compactQueries, uint32_t(blas.compactQuery), 1);
    vkCmdWriteAccelerationStructuresPropertiesKHR(m_cmd, 1, &blas.accel.accel, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                                                  m_compactQueries, uint32_t(blas.compactQuery));
    // the frame's resources are freed once the queue is past it, and so is the query
    retire([written = blas.sizeWritten] { *written = true; });
  }
}

VulkanGpuBackend::Compaction VulkanGpuBackend::takeCompaction()
{
  Compaction out = std::exchange(m_compaction, {});
  out.blasBytes  = m_memory.getStats(kMemBlas).currentBytes;
  return out;
}

void VulkanGpuBackend::releaseCompactQuery(BlasEntry& blas)
{
  if(blas.compactQuery >= 0)
    m_freeCompactQueries.push_back(uint32_t(blas.compactQuery));
  blas.compactQuery = -1;
  blas.sizeWritten.reset();
}

AccelHandle VulkanGpuBackend::compactBlas(AccelHandle handle)
{
  auto it = m_blases.find(handle.value);
  if(it == m_blases.end())
  {
    LOGE("VulkanGpuBackend::compactBlas: a dead or unknown BLAS\n");
    return handle;
  }
  BlasEntry& blas = it->second;
  if(blas.compactQuery < 0 || !recording("compactBlas"))
    return handle;  // stays as built
  if(!*blas.sizeWritten)
    return {};  // its frame is still in flight
  VkDeviceSize size = 0;
  const VkResult read = vkGetQueryPoolResults(m_info.device, m_compactQueries, uint32_t(blas.compactQuery), 1, sizeof(size),
                                              &size, sizeof(size), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  releaseCompactQuery(blas);
  if(read != VK_SUCCESS || size == 0 || size >= blas.data.sizeInfo.accelerationStructureSize)
    return handle;
  // call order; and a refit of this one still queued is written before the copy reads it
  const bool refitQueued = std::any_of(m_queued.begin(), m_queued.end(), [&](const QueuedBuild& q) { return q.blas == handle.value; });
  flushBuilds();
  if(refitQueued)
    nvvk::accelerationStructureBarrier(m_cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  BlasEntry                                  compacted;
  const VkAccelerationStructureCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                                  .size  = size,
                                                  .type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
  if(m_info.alloc->createAcceleration(compacted.accel, info) != VK_SUCCESS)
    return handle;  // no room for the copy: it stays as built
  NVVK_DBG_NAME(compacted.accel.accel);
  m_memory.track(kMemBlas, compacted.accel.buffer.allocation);
  // A copy of one built to be refitted is refitted in its place (falcor's Scene::buildBlas does the same;
  // DXR calls the copy functionally equivalent, and the validation layers take its build flags from
  // the source's): it keeps the build data, and the micromap link its geometry points at. A copy of
  // one that wasn't keeps none.
  if(blas.data.buildInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
  {
    compacted.data                              = blas.data;
    compacted.data.sizeInfo.accelerationStructureSize = size;
    compacted.micromap                          = std::move(blas.micromap);
  }
  const VkCopyAccelerationStructureInfoKHR copy{.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
                                                .src   = blas.accel.accel,
                                                .dst   = compacted.accel.accel,
                                                .mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR};
  vkCmdCopyAccelerationStructureKHR(m_cmd, &copy);
  m_compaction.blases++;
  m_compaction.before += blas.data.sizeInfo.accelerationStructureSize;
  m_compaction.after += size;
  const AccelHandle out{m_nextHandle++};
  m_blases.emplace(out.value, std::move(compacted));
  return out;
}

// The queued builds and refits, side by side in scratch batches (I8), a barrier between batches
// since they share the buffer
void VulkanGpuBackend::flushBuilds()
{
  if((m_queued.empty() && m_queuedMicromaps.empty()) || !recording("flushBuilds"))
    return;
  flushUploads();
  recordMicromaps();
  if(m_queued.empty())
    return;
  std::vector<VkDeviceSize> sizes;
  sizes.reserve(m_queued.size());
  for(const QueuedBuild& q : m_queued)
  {
    const nvvk::AccelerationStructureBuildData& data = m_blases.at(q.blas).data;
    sizes.push_back(q.update ? data.sizeInfo.updateScratchSize : data.sizeInfo.buildScratchSize);
  }
  const ScratchPlan plan = planScratch(sizes, m_scratchAlignment, m_info.blasScratchBudget);
  m_blasScratchUsed      = m_frame;
  if(m_blasScratch.bufferSize < plan.size)
  {
    if(m_blasScratch.buffer)
      retire([alloc = m_info.alloc, &memory = m_memory, scratch = m_blasScratch]() mutable {
        memory.untrack(kMemScratch, scratch.allocation);
        alloc->destroyBuffer(scratch);
      });
    m_blasScratch = {};
    if(m_info.alloc->createBuffer(m_blasScratch, plan.size, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                  VMA_MEMORY_USAGE_AUTO, {}, m_scratchAlignment)
       != VK_SUCCESS)
    {
      // nothing queued can be built: a BLAS never built must not reach a TLAS, and a refit keeps what it has
      LOGW("VulkanGpuBackend::flushBuilds: no memory for %llu bytes of build scratch\n", (unsigned long long)plan.size);
      m_blasScratch      = {};
      m_allocationFailed = true;
      for(const QueuedBuild& q : m_queued)
        if(!q.update)
          if(BlasEntry& blas = m_blases.at(q.blas); blas.accel.accel)
          {
            m_memory.untrack(kMemBlas, blas.accel.buffer.allocation);
            m_info.alloc->destroyAcceleration(blas.accel);
            blas.accel = {};
          }
      m_queued.clear();
      return;
    }
    NVVK_DBG_NAME(m_blasScratch.buffer);
    m_memory.track(kMemScratch, m_blasScratch.allocation);
  }

  std::vector<VkAccelerationStructureBuildGeometryInfoKHR>     infos;
  std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> ranges;
  const auto                                                   record = [&] {
    if(infos.empty())
      return;
    vkCmdBuildAccelerationStructuresKHR(m_cmd, uint32_t(infos.size()), infos.data(), ranges.data());
    infos.clear();
    ranges.clear();
  };
  const bool timed = beginSpan();
  for(size_t i = 0; i < m_queued.size(); i++)
  {
    if(i > 0 && plan.places[i].batch != plan.places[i - 1].batch)
    {
      record();
      // the next batch writes the same scratch
      nvvk::accelerationStructureBarrier(m_cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                         VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    }
    BlasEntry&                                   blas = m_blases.at(m_queued[i].blas);
    VkAccelerationStructureBuildGeometryInfoKHR& info = blas.data.buildInfo;
    info.mode = m_queued[i].update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.srcAccelerationStructure  = m_queued[i].update ? blas.accel.accel : VK_NULL_HANDLE;
    info.dstAccelerationStructure  = blas.accel.accel;
    info.scratchData.deviceAddress = m_blasScratch.address + plan.places[i].offset;
    info.pGeometries               = blas.data.asGeometry.data();
    info.geometryCount             = uint32_t(blas.data.asGeometry.size());
    infos.push_back(info);
    ranges.push_back(blas.data.asBuildRangeInfo.data());
  }
  record();
  if(timed)
    endSpan(BuildSpan::eBlas);
  std::vector<uint64_t> compactable;
  for(const QueuedBuild& q : m_queued)
    if(!q.update && m_blases.at(q.blas).compactQuery >= 0)
      compactable.push_back(q.blas);
  if(!compactable.empty())
    writeCompactedSizes(compactable);
  m_queued.clear();
}

AccelHandle VulkanGpuBackend::createTlas(uint32_t capacity)
{
  TlasEntry entry;
  entry.capacity = std::max(capacity, 1u);
  NVVK_CHECK(m_info.alloc->createBuffer(entry.instances, VkDeviceSize(entry.capacity) * sizeof(VkAccelerationStructureInstanceKHR),
                                        VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                            | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                        VMA_MEMORY_USAGE_AUTO, {}, 16));
  NVVK_DBG_NAME(entry.instances.buffer);
  m_memory.track(kMemInstances, entry.instances.allocation);
  entry.data.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  entry.data.addGeometry(entry.data.makeInstanceGeometry(entry.capacity, entry.instances.address));
  const VkAccelerationStructureBuildSizesInfoKHR sizes =
      entry.data.finalizeGeometry(m_info.device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                     | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
  NVVK_CHECK(m_info.alloc->createAcceleration(entry.accel, entry.data.makeCreateInfo()));
  NVVK_DBG_NAME(entry.accel.accel);
  m_memory.track(kMemTlas, entry.accel.buffer.allocation);
  NVVK_CHECK(m_info.alloc->createBuffer(entry.scratch, std::max(sizes.buildScratchSize, sizes.updateScratchSize),
                                        VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                        VMA_MEMORY_USAGE_AUTO, {}, m_scratchAlignment));
  NVVK_DBG_NAME(entry.scratch.buffer);
  m_memory.track(kMemScratch, entry.scratch.allocation);
  const AccelHandle handle{m_nextHandle++};
  m_tlases.emplace(handle.value, std::move(entry));
  return handle;
}

void VulkanGpuBackend::buildTlas(AccelHandle handle, std::span<const TlasRow> rows, TlasMode mode)
{
  if(!recording("buildTlas"))
    return;
  auto it = m_tlases.find(handle.value);
  if(it == m_tlases.end() || rows.size() > it->second.capacity)
  {
    LOGE("VulkanGpuBackend::buildTlas: a dead TLAS or %zu rows past its capacity\n", rows.size());
    return;
  }
  TlasEntry& tlas = it->second;
  flushBuilds();
  flushUploads();  // what came before, outside the timed span
  const bool                                      timed = beginSpan();
  std::vector<VkAccelerationStructureInstanceKHR> instances(rows.size());
  for(size_t i = 0; i < rows.size(); i++)
  {
    VkDeviceAddress blas = 0;
    if(rows[i].blas)
    {
      auto b = m_blases.find(rows[i].blas.value);
      if(b == m_blases.end())
        LOGE("VulkanGpuBackend::buildTlas: row %zu references a dead or unknown BLAS\n", i);
      else
        blas = b->second.accel.address;
    }
    instances[i] = vkTlasInstance(rows[i], blas);
  }
  if(!instances.empty())
    NVVK_CHECK(appendBufferChunked(*m_info.staging, tlas.instances, 0, std::span<const VkAccelerationStructureInstanceKHR>(instances)));
  m_uploadsPending = true;
  flushUploads();
  // (the destination masks nvvk's helper takes)
  nvvk::accelerationStructureBarrier(m_cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
  tlas.data.asBuildRangeInfo[0].primitiveCount = uint32_t(rows.size());
  if(mode == TlasMode::eUpdate)
    tlas.data.cmdUpdateAccelerationStructure(m_cmd, tlas.accel.accel, tlas.scratch.address);
  else
    tlas.data.cmdBuildAccelerationStructure(m_cmd, tlas.accel.accel, tlas.scratch.address);
  if(timed)
    endSpan(BuildSpan::eTlas);
}

void VulkanGpuBackend::destroyAccel(AccelHandle handle)
{
  // a queued build of it would build into nothing
  std::erase_if(m_queued, [&](const QueuedBuild& q) { return q.blas == handle.value; });
  std::erase(m_queuedMicromaps, handle.value);
  if(auto m = m_micromaps.find(handle.value); m != m_micromaps.end())
  {
    vkDestroyAccelerationStructureKHR(m_info.device, m->second.micromap, nullptr);
    m_memory.untrack(kMemMicromaps, m->second.storage.allocation);
    m_info.alloc->destroyBuffer(m->second.storage);
    if(m->second.data.buffer)  // never recorded
    {
      m_memory.untrack(kMemMicromaps, m->second.data.allocation);
      m_memory.untrack(kMemMicromaps, m->second.triangles.allocation);
      m_memory.untrack(kMemScratch, m->second.scratch.allocation);
      m_info.alloc->destroyBuffer(m->second.data);
      m_info.alloc->destroyBuffer(m->second.triangles);
      m_info.alloc->destroyBuffer(m->second.scratch);
    }
    m_micromaps.erase(m);
    return;
  }
  if(auto b = m_blases.find(handle.value); b != m_blases.end())
  {
    releaseCompactQuery(b->second);
    if(b->second.accel.accel)  // or it found no memory
    {
      m_memory.untrack(kMemBlas, b->second.accel.buffer.allocation);
      m_info.alloc->destroyAcceleration(b->second.accel);
    }
    m_blases.erase(b);
    return;
  }
  if(auto t = m_tlases.find(handle.value); t != m_tlases.end())
  {
    m_memory.untrack(kMemTlas, t->second.accel.buffer.allocation);
    m_memory.untrack(kMemInstances, t->second.instances.allocation);
    m_memory.untrack(kMemScratch, t->second.scratch.allocation);
    m_info.alloc->destroyAcceleration(t->second.accel);
    m_info.alloc->destroyBuffer(t->second.instances);
    m_info.alloc->destroyBuffer(t->second.scratch);
    m_tlases.erase(t);
    return;
  }
  LOGE("VulkanGpuBackend::destroyAccel: dead or unknown acceleration structure %llu\n", (unsigned long long)handle.value);
}

// How much of an allocation lies off the device
static uint64_t spilledOf(VmaAllocator allocator, VmaAllocation allocation)
{
  if(!allocation)
    return 0;
  VmaAllocationInfo info{};
  vmaGetAllocationInfo(allocator, allocation, &info);
  VkMemoryPropertyFlags flags = 0;
  vmaGetMemoryTypeProperties(allocator, info.memoryType, &flags);
  return flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? 0 : info.size;
}

bool VulkanGpuBackend::bufferSpilled(BufferHandle buffer) const
{
  const auto it = m_buffers.find(buffer.value);
  return it != m_buffers.end() && spilledOf(*m_info.alloc, it->second.buffer.allocation) > 0;
}

bool VulkanGpuBackend::imageSpilled(ImageHandle image) const
{
  const auto it = m_images.find(image.value);
  return it != m_images.end() && spilledOf(*m_info.alloc, it->second.allocation) > 0;
}

bool VulkanGpuBackend::accelSpilled(AccelHandle accel) const
{
  const auto it = m_blases.find(accel.value);
  return it != m_blases.end() && spilledOf(*m_info.alloc, it->second.accel.buffer.allocation) > 0;
}

uint64_t VulkanGpuBackend::spilledBytes() const
{
  const VmaAllocator allocator = *m_info.alloc;
  const auto         spilled   = [&](VmaAllocation allocation) { return spilledOf(allocator, allocation); };
  uint64_t bytes = spilled(m_blasScratch.allocation);
  for(const auto& [handle, b] : m_buffers)
    bytes += spilled(b.buffer.allocation);
  for(const auto& [handle, image] : m_images)
    bytes += spilled(image.allocation);
  for(const auto& [handle, b] : m_blases)
    bytes += spilled(b.accel.buffer.allocation);
  for(const auto& [handle, t] : m_tlases)
    bytes += spilled(t.accel.buffer.allocation) + spilled(t.instances.allocation) + spilled(t.scratch.allocation);
  for(const auto& [handle, m] : m_micromaps)
    bytes += spilled(m.storage.allocation);
  return bytes;
}

uint64_t VulkanGpuBackend::accelBytes(AccelHandle handle) const
{
  if(auto m = m_micromaps.find(handle.value); m != m_micromaps.end())
    return m->second.storage.bufferSize + m->second.data.bufferSize + m->second.triangles.bufferSize + m->second.scratch.bufferSize;
  if(auto b = m_blases.find(handle.value); b != m_blases.end())
    return b->second.accel.buffer.bufferSize;
  if(auto t = m_tlases.find(handle.value); t != m_tlases.end())
    return t->second.accel.buffer.bufferSize + t->second.instances.bufferSize + t->second.scratch.bufferSize;
  return 0;
}

VkAccelerationStructureKHR VulkanGpuBackend::vkAccel(AccelHandle handle) const
{
  if(auto b = m_blases.find(handle.value); b != m_blases.end())
    return b->second.accel.accel;
  if(auto t = m_tlases.find(handle.value); t != m_tlases.end())
    return t->second.accel.accel;
  return VK_NULL_HANDLE;
}

// ---------- The view ----------

RetainedSceneVk::RetainedSceneVk(const VulkanGpuBackend::Info& info, ResidencyOptions options)
    : m_gpu([&] {
      VulkanGpuBackend::Info i = info;
      i.blasScratchBudget      = options.blasScratchBudget;
      return i;
    }())
    , m_residency(m_gpu, options)
{
}

SyncResult RetainedSceneVk::replace(VkCommandBuffer cmd, RetainedScene& scene)
{
  m_gpu.record(cmd);
  const SyncResult result = m_residency.replace(scene);
  m_gpu.finish();
  m_hasScene = true;
  afterSync(scene, result);
  return result;
}

SyncResult RetainedSceneVk::sync(VkCommandBuffer cmd, RetainedScene& scene)
{
  m_gpu.record(cmd);
  const SyncResult result = m_residency.sync(scene);
  m_gpu.finish();
  afterSync(scene, result);
  return result;
}

void RetainedSceneVk::clear()
{
  m_residency.clear();
  m_hasScene = false;
  m_lights.clear();
  m_bounds = {};
  m_transformsRevision++;
}

void RetainedSceneVk::afterSync(RetainedScene& scene, const SyncResult& result)
{
  if(!result.changed)
    return;
  m_transformsRevision++;
  m_lights.clear();
  for(const LightItem& item : scene.lightItems())
  {
    LightInfo          info;
    const LightRecord* record = scene.light(item.light);
    info.worldMatrix          = item.world;
    if(record)
    {
      info.directional = record->payload.type == "directional";
      info.skySun      = record->skySun;
      info.intensity   = item.shown ? record->payload.intensity : 0.0;  // as the light buffer holds it
      if(record->payload.color.size() == 3)
        info.maxColor = std::max({record->payload.color[0], record->payload.color[1], record->payload.color[2]});
    }
    m_lights.push_back(info);
  }
  m_bounds = {};
  for(uint32_t s = 0; s < m_residency.instanceHighWater(); s++)
  {
    const Residency::Instance* inst = m_residency.instance(s);
    const Residency::Prim*     prim = inst && inst->shown ? m_residency.prim(inst->prim) : nullptr;
    if(prim && prim->indexCount)
      m_bounds.insert(nvutils::Bbox(prim->boundsMin, prim->boundsMax).transform(inst->world));
  }
}

VkDeviceAddress RetainedSceneVk::sceneDesc() const
{
  return m_hasScene ? m_residency.sceneDesc() : 0;
}

VkAccelerationStructureKHR RetainedSceneVk::tlas() const
{
  return m_gpu.vkAccel(m_residency.tlas());
}

uint32_t RetainedSceneVk::renderNodeCapacity() const
{
  return m_hasScene ? m_residency.instanceCapacity() : 0;
}

const std::vector<uint32_t>& RetainedSceneVk::drawList(DrawList list) const
{
  switch(list)
  {
    case DrawList::eSolid:
      return m_residency.drawList(Residency::DrawList::eSolid);
    case DrawList::eSolidDoubleSided:
      return m_residency.drawList(Residency::DrawList::eSolidDoubleSided);
    case DrawList::eBlend:
      return m_residency.drawList(Residency::DrawList::eBlend);
    case DrawList::eAll:
      break;
  }
  return m_residency.drawList(Residency::DrawList::eAll);
}

SceneGpuView::DrawRecord RetainedSceneVk::drawRecord(uint32_t renderNode) const
{
  const Residency::Instance* inst = m_residency.instance(renderNode);
  const Residency::Prim*     prim = inst ? m_residency.prim(inst->prim) : nullptr;
  const Residency::MeshGpu*  mesh = prim ? m_residency.meshGpu(prim->mesh) : nullptr;
  if(!mesh)
    return {};
  // a deformed prim draws its drawer's output, which the deform passes wrote (its row says so too)
  const Residency::DrawerGpu* drawer = prim->drawer.isNone() ? nullptr : m_residency.drawer(prim->drawer, prim->mesh);
  return {.vertexBuffer = m_gpu.vkBuffer(drawer ? drawer->positions : mesh->positions),
          .indexBuffer  = m_gpu.vkBuffer(prim->indices),
          .indexCount   = prim->indexCount,
          .materialID   = int32_t(inst->materialSlot),
          .renderPrimID = int32_t(inst->prim),
          .visible      = inst->shown};
}

std::optional<glm::vec3> RetainedSceneVk::worldCentre(uint32_t renderNode) const
{
  const Residency::Instance* inst = m_residency.instance(renderNode);
  const Residency::Prim*     prim = inst ? m_residency.prim(inst->prim) : nullptr;
  if(!prim)
    return std::nullopt;
  return glm::vec3(inst->world * glm::vec4((prim->boundsMin + prim->boundsMax) * 0.5f, 1.f));
}

SceneGpuView::MemoryStats RetainedSceneVk::memoryStats() const
{
  const nvvkgltf::GpuMemoryStats total = m_gpu.memory().getTotalStats();
  return {.usedBytes = total.currentBytes, .peakBytes = total.peakBytes};
}

}  // namespace rs
