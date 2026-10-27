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
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/uploader_interface.hpp>

#include "gltf_image_loader.hpp"
#include "gltf_scene_vk.hpp"
#include "scene_residency_vk_util.hpp"
#include "staging_config.hpp"

namespace rs {

namespace {

constexpr std::string_view kMemBuffers   = "Retained/Buffers";
constexpr std::string_view kMemImages    = "Retained/Images";
constexpr std::string_view kMemBlas      = "Retained/BLAS";
constexpr std::string_view kMemTlas      = "Retained/TLAS";
constexpr std::string_view kMemScratch   = "Retained/Scratch";
constexpr std::string_view kMemInstances = "Retained/Instances";

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

VulkanGpuBackend::VulkanGpuBackend(const Info& info)
    : m_info(info)
{
  m_memory.init(info.alloc);
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &asProps};
  vkGetPhysicalDeviceProperties2(info.physicalDevice, &props);
  m_scratchAlignment = std::max<VkDeviceSize>(asProps.minAccelerationStructureScratchOffsetAlignment, 1);
}

VulkanGpuBackend::~VulkanGpuBackend()
{
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
  for(auto& [h, t] : m_tlases)
  {
    m_info.alloc->destroyAcceleration(t.accel);
    m_info.alloc->destroyBuffer(t.instances);
    m_info.alloc->destroyBuffer(t.scratch);
  }
  m_info.alloc->destroyBuffer(m_blasScratch);
  for(auto& [slot, sampler] : m_samplers)
    m_info.samplerPool->releaseSampler(sampler);
  m_info.alloc->destroyImage(m_white);
  m_info.alloc->destroyImage(m_missing);
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
  m_cmd = VK_NULL_HANDLE;
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

void VulkanGpuBackend::waitIdle()
{
  vkDeviceWaitIdle(m_info.device);
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

void VulkanGpuBackend::dispatchDeform(const DeformDispatch&)
{
  LOGE("VulkanGpuBackend::dispatchDeform: the deform passes come with P3b; nothing recorded\n");
}

// ---------- Buffers ----------

BufferHandle VulkanGpuBackend::createBuffer(const BufferDesc& desc)
{
  BufferEntry entry;
  entry.size = desc.size;
  // 16-byte aligned: TLAS instances need it, and BLAS inputs need 4 (I17)
  NVVK_CHECK(m_info.alloc->createBuffer(entry.buffer, std::max<VkDeviceSize>(desc.size, 16), vkBufferUsage(desc.usage),
                                        VMA_MEMORY_USAGE_AUTO, {}, 16));
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
  m_ranges.erase(it->second.buffer.address);
  m_memory.untrack(kMemBuffers, it->second.buffer.allocation);
  m_info.alloc->destroyBuffer(it->second.buffer);
  m_buffers.erase(it);
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
  if(it == m_buffers.end() || offset > it->second.size || bytes.size() > it->second.size - offset)
  {
    LOGE("VulkanGpuBackend::upload: %zu bytes at %llu do not fit buffer %llu\n", bytes.size(),
         (unsigned long long)offset, (unsigned long long)dst.value);
    return;
  }
  if(bytes.empty())
    return;
  flushBuilds();  // the appended uploads always precede the queued builds (call order)
  NVVK_CHECK(appendBufferChunked(*m_info.staging, it->second.buffer, size_t(offset), bytes));
  m_uploadsPending = true;
}

void VulkanGpuBackend::copyBuffer(BufferHandle src, uint64_t srcOffset, BufferHandle dst, uint64_t dstOffset, uint64_t size)
{
  if(!recording("copyBuffer"))
    return;
  auto s = m_buffers.find(src.value), d = m_buffers.find(dst.value);
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

ImageHandle VulkanGpuBackend::createImage(const ImageSource& source, ImageInfo& info)
{
  info = {};
  if(!recording("createImage"))
    return {};
  nvvkgltf::LoadedImageData loaded;
  bool                      decoded = false;
  if(!source.pixels.empty())
    decoded = fromPixels(source, loaded);
  else if(!source.bytes.empty())
    decoded = nvvkgltf::loadFromMemory(loaded, source.bytes.data(), source.bytes.size(), source.srgb);
  else if(!source.uri.empty())
  {
    nvutils::FileReadMapping file;
    if(file.open(nvutils::pathFromUtf8(source.uri)))
      decoded = nvvkgltf::loadFromMemory(loaded, file.data(), file.size(), source.srgb);
    else
      LOGW("Image '%s' (%s) could not be opened\n", source.name.c_str(), source.uri.c_str());
  }
  if(!decoded || loaded.mipData.empty())
    return {};

  flushBuilds();  // createSampledImage records the appended uploads with its own
  nvvk::Image image;
  if(!nvvkgltf::createSampledImage(m_cmd, *m_info.staging, *m_info.alloc, m_info.physicalDevice, m_info.generateMipmaps,
                                   loaded.format, loaded.size, loaded.mipData, loaded.componentMapping, source.name, image))
    return {};
  m_uploadsPending = false;
  m_memory.track(kMemImages, image.allocation);
  info.width         = loaded.size.width;
  info.height        = loaded.size.height;
  info.twoChannel    = nvvkgltf::formatColorComponentCount(loaded.format) == 2;
  info.meanLuminance = nvvkgltf::imageMeanLuminance(loaded.format, loaded.size, loaded.mipData[0]);
  const ImageHandle handle{m_nextHandle++};
  m_images.emplace(handle.value, image);
  return handle;
}

void VulkanGpuBackend::destroyImage(ImageHandle handle)
{
  auto it = m_images.find(handle.value);
  if(it == m_images.end())
  {
    LOGE("VulkanGpuBackend::destroyImage: dead or unknown image %llu\n", (unsigned long long)handle.value);
    return;
  }
  m_memory.untrack(kMemImages, it->second.allocation);
  m_info.alloc->destroyImage(it->second);
  m_images.erase(it);
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
}

nvvk::AccelerationStructureGeometryInfo VulkanGpuBackend::geometryOf(const BlasInput& input) const
{
  // as SceneRtx::renderPrimitiveToAsGeometry: float3 positions, uint32 indices
  nvvk::AccelerationStructureGeometryInfo out;
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = input.vertexAddress;
  triangles.vertexStride             = input.vertexStride;
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = input.indexAddress;
  triangles.maxVertex                = input.vertexCount ? input.vertexCount - 1 : 0;
  out.geometry.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  out.geometry.geometryType          = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  out.geometry.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | (input.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0);
  out.geometry.geometry.triangles = triangles;
  out.rangeInfo.primitiveCount    = input.indexCount / 3;
  return out;
}

AccelHandle VulkanGpuBackend::buildBlas(const BlasInput& input, AccelBuildFlags flags)
{
  checkInput(input, "buildBlas");
  BlasEntry entry;
  entry.data.asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  entry.data.addGeometry(geometryOf(input));
  entry.data.finalizeGeometry(m_info.device, vkAccelBuildFlags(flags));
  NVVK_CHECK(m_info.alloc->createAcceleration(entry.accel, entry.data.makeCreateInfo()));
  NVVK_DBG_NAME(entry.accel.accel);
  m_memory.track(kMemBlas, entry.accel.buffer.allocation);
  const AccelHandle handle{m_nextHandle++};
  m_blases.emplace(handle.value, std::move(entry));
  m_queued.push_back({handle.value, false});
  return handle;
}

void VulkanGpuBackend::refitBlas(AccelHandle handle, const BlasInput& input)
{
  auto it = m_blases.find(handle.value);
  if(it == m_blases.end() || !(it->second.data.buildInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR))
  {
    LOGE("VulkanGpuBackend::refitBlas: a dead BLAS or one built without eAccelAllowUpdate\n");
    return;
  }
  checkInput(input, "refitBlas");
  // the same addresses and index count (the residency builds again otherwise); the vertex count may not move either
  it->second.data.asGeometry[0] = geometryOf(input).geometry;
  m_queued.push_back({handle.value, true});
}

AccelHandle VulkanGpuBackend::compactBlas(AccelHandle)
{
  LOGE("VulkanGpuBackend::compactBlas: compaction comes with P7; not compacted\n");
  return {};
}

// The queued builds and refits, side by side in scratch batches (I8), a barrier between batches
// since they share the buffer
void VulkanGpuBackend::flushBuilds()
{
  if(m_queued.empty() || !recording("flushBuilds"))
    return;
  flushUploads();
  std::vector<VkDeviceSize> sizes;
  sizes.reserve(m_queued.size());
  for(const QueuedBuild& q : m_queued)
  {
    const nvvk::AccelerationStructureBuildData& data = m_blases.at(q.blas).data;
    sizes.push_back(q.update ? data.sizeInfo.updateScratchSize : data.sizeInfo.buildScratchSize);
  }
  const ScratchPlan plan = planScratch(sizes, m_scratchAlignment, m_info.blasScratchBudget);
  if(m_blasScratch.bufferSize < plan.size)
  {
    if(m_blasScratch.buffer)
      retire([alloc = m_info.alloc, &memory = m_memory, scratch = m_blasScratch]() mutable {
        memory.untrack(kMemScratch, scratch.allocation);
        alloc->destroyBuffer(scratch);
      });
    m_blasScratch = {};
    NVVK_CHECK(m_info.alloc->createBuffer(m_blasScratch, plan.size, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                          VMA_MEMORY_USAGE_AUTO, {}, m_scratchAlignment));
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
  nvvk::accelerationStructureBarrier(m_cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
  tlas.data.asBuildRangeInfo[0].primitiveCount = uint32_t(rows.size());
  if(mode == TlasMode::eUpdate)
    tlas.data.cmdUpdateAccelerationStructure(m_cmd, tlas.accel.accel, tlas.scratch.address);
  else
    tlas.data.cmdBuildAccelerationStructure(m_cmd, tlas.accel.accel, tlas.scratch.address);
}

void VulkanGpuBackend::destroyAccel(AccelHandle handle)
{
  // a queued build of it would build into nothing
  std::erase_if(m_queued, [&](const QueuedBuild& q) { return q.blas == handle.value; });
  if(auto b = m_blases.find(handle.value); b != m_blases.end())
  {
    m_memory.untrack(kMemBlas, b->second.accel.buffer.allocation);
    m_info.alloc->destroyAcceleration(b->second.accel);
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
  return {.vertexBuffer = m_gpu.vkBuffer(mesh->positions),
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
