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

#include "fake_gpu_backend.hpp"

#include <algorithm>
#include <cstring>
#include <iterator>

#include <nvutils/offset_allocator.hpp>

namespace rs {

namespace {

constexpr uint32_t kAddressUnits = 1u << 30;  // 16 GiB of fake address space

uint64_t fnv1a(uint64_t hash, std::span<const std::byte> bytes)
{
  for(std::byte b : bytes)
    hash = (hash ^ uint64_t(b)) * 0x100000001b3ull;
  return hash;
}

}  // namespace

FakeGpuBackend::FakeGpuBackend(uint32_t framesInFlight)
    : m_framesInFlight(std::max(framesInFlight, 1u))
    , m_addresses(std::make_unique<OffsetAllocator::Allocator>(kAddressUnits))
{
}

FakeGpuBackend::~FakeGpuBackend() = default;

void FakeGpuBackend::error(std::string message)
{
  m_errors.push_back(std::move(message));
}

// ---------- Frame ----------

void FakeGpuBackend::beginFrame()
{
  m_frame++;
  // the ready ones first: a closure may queue another, which waits its own frame cycle
  std::vector<Retire> ready;
  auto split = std::stable_partition(m_retires.begin(), m_retires.end(),
                                     [&](const Retire& r) { return r.frame + m_framesInFlight > m_frame; });
  std::move(split, m_retires.end(), std::back_inserter(ready));
  m_retires.erase(split, m_retires.end());
  for(Retire& r : ready)
    r.closure();
}

void FakeGpuBackend::retire(std::function<void()> closure)
{
  m_retires.push_back({m_frame, std::move(closure)});
}

void FakeGpuBackend::waitIdle()
{
  m_waitIdleCount++;
}

// ---------- Buffers ----------

FakeGpuBackend::Buffer* FakeGpuBackend::liveBuffer(BufferHandle handle, const char* op)
{
  auto it = m_buffers.find(handle.value);
  if(it == m_buffers.end())
  {
    error(std::string(op) + ": dead or unknown buffer " + std::to_string(handle.value));
    return nullptr;
  }
  return &it->second;
}

BufferHandle FakeGpuBackend::createBuffer(const BufferDesc& desc)
{
  const uint64_t                    units = std::max<uint64_t>((desc.size + kAddressUnit - 1) / kAddressUnit, 1);
  const OffsetAllocator::Allocation alloc =
      units < kAddressUnits ? m_addresses->allocate(uint32_t(units)) : OffsetAllocator::Allocation{};
  if(alloc.offset == OffsetAllocator::Allocation::NO_SPACE)
  {
    error("createBuffer: out of fake address space for " + std::to_string(desc.size) + " bytes");
    return {};
  }
  Buffer b;
  b.desc          = desc;
  b.address       = kAddressBase + uint64_t(alloc.offset) * kAddressUnit;
  b.bytes         = std::vector<std::byte>(desc.size);
  b.allocOffset   = alloc.offset;
  b.allocMetadata = alloc.metadata;
  const BufferHandle handle{m_nextHandle++};
  m_buffers.emplace(handle.value, std::move(b));
  return handle;
}

void FakeGpuBackend::destroyBuffer(BufferHandle handle)
{
  Buffer* b = liveBuffer(handle, "destroyBuffer");
  if(!b)
    return;
  m_addresses->free({b->allocOffset, b->allocMetadata});
  m_buffers.erase(handle.value);
  m_writtenBeforeBuild.erase(handle.value);
  m_writtenBeforeShading.erase(handle.value);
}

uint64_t FakeGpuBackend::bufferAddress(BufferHandle handle) const
{
  const Buffer* b = buffer(handle);
  return b ? b->address : 0;
}

uint64_t FakeGpuBackend::bufferSize(BufferHandle handle) const
{
  const Buffer* b = buffer(handle);
  return b ? b->desc.size : 0;
}

void FakeGpuBackend::upload(BufferHandle dst, uint64_t offset, std::span<const std::byte> bytes)
{
  Buffer* b = liveBuffer(dst, "upload");
  if(!b)
    return;
  if(offset > b->desc.size || bytes.size() > b->desc.size - offset)
  {
    error("upload: " + std::to_string(bytes.size()) + " bytes at " + std::to_string(offset)
          + " past the end of a buffer of " + std::to_string(b->desc.size));
    return;
  }
  if(!bytes.empty())
    std::memcpy(b->bytes.data() + offset, bytes.data(), bytes.size());
  m_uploads.push_back({dst, offset, bytes.size()});
  written(dst);
}

void FakeGpuBackend::written(BufferHandle handle)
{
  m_writtenBeforeBuild.insert(handle.value);
  m_writtenBeforeShading.insert(handle.value);
}

void FakeGpuBackend::readByBuild(uint64_t address, const char* op)
{
  for(const auto& [handle, b] : m_buffers)
    if(address >= b.address && address < b.address + std::max<uint64_t>(b.desc.size, 1) && m_writtenBeforeBuild.count(handle))
      error(std::string(op) + ": reads a buffer written with no transfer-to-build barrier between");
}

std::vector<std::string> FakeGpuBackend::unfencedForTheFrame() const
{
  std::vector<std::string> out;
  if(!m_writtenBeforeShading.empty())
    out.push_back(std::to_string(m_writtenBeforeShading.size()) + " buffers written with no transfer-to-shader barrier after");
  if(m_buildsBeforeTrace)
    out.push_back("acceleration structures built with no build-to-trace barrier after");
  return out;
}

void FakeGpuBackend::copyBuffer(BufferHandle src, uint64_t srcOffset, BufferHandle dst, uint64_t dstOffset, uint64_t size)
{
  Buffer* s = liveBuffer(src, "copyBuffer source");
  Buffer* d = liveBuffer(dst, "copyBuffer destination");
  if(!s || !d)
    return;
  if(srcOffset > s->desc.size || size > s->desc.size - srcOffset || dstOffset > d->desc.size || size > d->desc.size - dstOffset)
  {
    error("copyBuffer: " + std::to_string(size) + " bytes out of range");
    return;
  }
  if(size)
    std::memmove(d->bytes.data() + dstOffset, s->bytes.data() + srcOffset, size);
  written(dst);
}

// ---------- Images and descriptors ----------

ImageHandle FakeGpuBackend::createImage(const ImageSource& source, ImageInfo& info)
{
  info = {};
  if((source.uri.empty() && source.bytes.empty() && source.pixels.empty()) || source.uri.rfind("missing", 0) == 0)
    return {};
  info.width      = uint32_t(std::max(source.width, 1));
  info.height     = uint32_t(std::max(source.height, 1));
  info.twoChannel = source.component == 2;
  info.bytes      = uint64_t(info.width) * info.height * 4;
  const ImageHandle handle{m_nextHandle++};
  m_images.emplace(handle.value, Image{source.name, source.uri, source.srgb, info});
  return handle;
}

void FakeGpuBackend::destroyImage(ImageHandle handle)
{
  if(!m_images.erase(handle.value))
    error("destroyImage: dead or unknown image " + std::to_string(handle.value));
}

void FakeGpuBackend::writeTextureDescriptor(uint32_t slot, ImageHandle handle)
{
  if(handle && !m_images.count(handle.value))
    error("writeTextureDescriptor: slot " + std::to_string(slot) + " given a dead or unknown image");
  m_textureDescriptors[slot] = handle;
}

void FakeGpuBackend::writeSamplerDescriptor(uint32_t slot, const GpuSamplerDesc& desc)
{
  m_samplerDescriptors[slot] = desc;
}

// ---------- Acceleration structures ----------

std::span<const std::byte> FakeGpuBackend::bytesAt(uint64_t address, uint64_t size) const
{
  for(const auto& [handle, b] : m_buffers)
    if(address >= b.address && address - b.address <= b.desc.size && size <= b.desc.size - (address - b.address))
      return std::span(b.bytes).subspan(address - b.address, size);
  return {};
}

uint64_t FakeGpuBackend::contentHash(const BlasInput& input) const
{
  const uint64_t vertexBytes = input.vertexCount ? uint64_t(input.vertexCount - 1) * input.vertexStride + 12 : 0;
  const uint64_t indexBytes  = uint64_t(input.indexCount) * sizeof(uint32_t);
  std::span<const std::byte> vertices = bytesAt(input.vertexAddress, vertexBytes);
  std::span<const std::byte> indices  = bytesAt(input.indexAddress, indexBytes);
  if(vertices.size() != vertexBytes || indices.size() != indexBytes || (vertexBytes && vertices.empty())
     || (indexBytes && indices.empty()))
    return 0;
  return fnv1a(fnv1a(0xcbf29ce484222325ull, vertices), indices);
}

AccelHandle FakeGpuBackend::buildMicromap(const MicromapInput& input)
{
  if(!m_opacityMicromaps)
    error("buildMicromap: the device does not build opacity micromaps");
  if(input.triangleStride < 8 || input.triangleStride % 4 || input.triangles.size() < input.triangleStride)
    error("buildMicromap: no triangle records at a stride Vulkan takes");
  m_buildsBeforeTrace = true;
  const AccelHandle handle{m_nextHandle++};
  m_micromaps.emplace(handle.value, Micromap{.data           = {input.data.begin(), input.data.end()},
                                             .triangles      = {input.triangles.begin(), input.triangles.end()},
                                             .triangleStride = input.triangleStride,
                                             .usages         = {input.usages.begin(), input.usages.end()}});
  return handle;
}

void FakeGpuBackend::checkMicromapLink(const BlasInput& input, const char* op)
{
  if(!input.micromap)
    return;
  if(!m_micromaps.count(input.micromap.value))
    error(std::string(op) + ": it links a dead or unknown micromap");
  if(input.micromapIndexSize)
  {
    const uint64_t bytes = uint64_t(input.indexCount / 3) * input.micromapIndexSize;
    if(bytesAt(input.micromapIndexAddress, bytes).size() != bytes)
      error(std::string(op) + ": its micromap indices are not inside a live buffer");
    readByBuild(input.micromapIndexAddress, op);
  }
}

AccelHandle FakeGpuBackend::buildBlas(const BlasInput& input, AccelBuildFlags flags)
{
  Blas b;
  b.input       = input;
  b.flags       = flags;
  b.contentHash = contentHash(input);
  b.builds      = 1;
  if(!b.contentHash)
    error("buildBlas: its positions or indices are not inside a live buffer");
  readByBuild(input.vertexAddress, "buildBlas");
  readByBuild(input.indexAddress, "buildBlas");
  checkMicromapLink(input, "buildBlas");
  m_blasBeforeTlas = m_buildsBeforeTrace = true;
  const AccelHandle handle{m_nextHandle++};
  m_blases.emplace(handle.value, b);
  return handle;
}

void FakeGpuBackend::refitBlas(AccelHandle handle, const BlasInput& input)
{
  auto it = m_blases.find(handle.value);
  if(it == m_blases.end())
  {
    error("refitBlas: dead or unknown BLAS " + std::to_string(handle.value));
    return;
  }
  Blas& b = it->second;
  if(!(b.flags & eAccelAllowUpdate))
    error("refitBlas: the BLAS was not built with eAccelAllowUpdate");
  if(b.compacted)
    error("refitBlas: the BLAS is compacted");
  if(input.vertexAddress != b.input.vertexAddress || input.indexAddress != b.input.indexAddress
     || input.indexCount != b.input.indexCount)
    error("refitBlas: the addresses or the index count changed (that needs a build)");
  if(input.micromap != b.input.micromap || input.micromapIndexAddress != b.input.micromapIndexAddress
     || input.micromapIndexSize != b.input.micromapIndexSize || input.micromapBaseTriangle != b.input.micromapBaseTriangle)
    error("refitBlas: the micromap link changed (that needs a build)");
  b.input       = input;
  b.contentHash = contentHash(input);
  b.refits++;
  if(!b.contentHash)
    error("refitBlas: its positions or indices are not inside a live buffer");
  readByBuild(input.vertexAddress, "refitBlas");
  readByBuild(input.indexAddress, "refitBlas");
  checkMicromapLink(input, "refitBlas");
  m_blasBeforeTlas = m_buildsBeforeTrace = true;
}

AccelHandle FakeGpuBackend::compactBlas(AccelHandle handle)
{
  auto it = m_blases.find(handle.value);
  if(it == m_blases.end())
  {
    error("compactBlas: dead or unknown BLAS " + std::to_string(handle.value));
    return {};
  }
  if(!(it->second.flags & eAccelAllowCompaction))
    error("compactBlas: the BLAS was not built with eAccelAllowCompaction");
  Blas copy      = it->second;
  copy.compacted = true;
  const AccelHandle compacted{m_nextHandle++};
  m_blases.emplace(compacted.value, copy);
  return compacted;
}

AccelHandle FakeGpuBackend::createTlas(uint32_t capacity)
{
  const AccelHandle handle{m_nextHandle++};
  m_tlases.emplace(handle.value, Tlas{.capacity = capacity});
  return handle;
}

void FakeGpuBackend::buildTlas(AccelHandle handle, std::span<const TlasRow> rows, TlasMode mode)
{
  auto it = m_tlases.find(handle.value);
  if(it == m_tlases.end())
  {
    error("buildTlas: dead or unknown TLAS " + std::to_string(handle.value));
    return;
  }
  Tlas& t = it->second;
  if(rows.size() > t.capacity)
    error("buildTlas: " + std::to_string(rows.size()) + " rows in a TLAS of " + std::to_string(t.capacity));
  for(size_t i = 0; i < rows.size(); i++)
    if(rows[i].blas && !m_blases.count(rows[i].blas.value))
      error("buildTlas: row " + std::to_string(i) + " references a dead or unknown BLAS");
  if(m_blasBeforeTlas)
    error("buildTlas: BLASes built or refitted with no build-to-build barrier before it");
  m_buildsBeforeTrace = true;
  if(mode == TlasMode::eUpdate)
  {
    const bool sameRows = t.builds + t.updates > 0 && rows.size() == t.rows.size()
                          && std::equal(rows.begin(), rows.end(), t.rows.begin(),
                                        [](const TlasRow& a, const TlasRow& b) { return a.blas == b.blas; });
    if(!sameRows)
      error("buildTlas: an update with other rows than the last build's");
    t.updates++;
  }
  else
    t.builds++;
  t.rows.assign(rows.begin(), rows.end());
  t.lastMode = mode;
}

void FakeGpuBackend::destroyAccel(AccelHandle handle)
{
  if(m_micromaps.count(handle.value))
  {
    for(const auto& [h, b] : m_blases)
      if(b.input.micromap == handle)
        error("destroyAccel: micromap " + std::to_string(handle.value) + " goes while a BLAS links it");
    m_micromaps.erase(handle.value);
    return;
  }
  if(!m_blases.erase(handle.value) && !m_tlases.erase(handle.value))
    error("destroyAccel: dead or unknown acceleration structure " + std::to_string(handle.value));
}

// ---------- Passes ----------

void FakeGpuBackend::dispatchDeform(const DeformDispatch& dispatch)
{
  for(BufferHandle b : dispatch.inputs)
    liveBuffer(b, "dispatchDeform input");
  liveBuffer(dispatch.output, "dispatchDeform output");
  m_deforms.push_back(dispatch);
}

void FakeGpuBackend::barrier(GpuBarrier barrier)
{
  m_barriers.push_back(barrier);
  switch(barrier)
  {
    case GpuBarrier::eTransferToShader:
      m_writtenBeforeShading.clear();
      break;
    case GpuBarrier::eTransferToBuild:
      m_writtenBeforeBuild.clear();
      break;
    case GpuBarrier::eComputeToBuild:
      break;
    case GpuBarrier::eBuildToBuild:
      m_blasBeforeTlas = false;
      break;
    case GpuBarrier::eBuildToTrace:
      m_buildsBeforeTrace = false;
      break;
  }
}

// ---------- Readback ----------

const FakeGpuBackend::Buffer* FakeGpuBackend::buffer(BufferHandle handle) const
{
  auto it = m_buffers.find(handle.value);
  return it == m_buffers.end() ? nullptr : &it->second;
}

const FakeGpuBackend::Image* FakeGpuBackend::image(ImageHandle handle) const
{
  auto it = m_images.find(handle.value);
  return it == m_images.end() ? nullptr : &it->second;
}

const FakeGpuBackend::Blas* FakeGpuBackend::blas(AccelHandle handle) const
{
  auto it = m_blases.find(handle.value);
  return it == m_blases.end() ? nullptr : &it->second;
}

const FakeGpuBackend::Micromap* FakeGpuBackend::micromap(AccelHandle handle) const
{
  auto it = m_micromaps.find(handle.value);
  return it == m_micromaps.end() ? nullptr : &it->second;
}

const FakeGpuBackend::Tlas* FakeGpuBackend::tlas(AccelHandle handle) const
{
  auto it = m_tlases.find(handle.value);
  return it == m_tlases.end() ? nullptr : &it->second;
}

}  // namespace rs
