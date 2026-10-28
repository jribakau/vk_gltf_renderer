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

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "nvshaders/functions.h.slang"
#include "skin_reference.hpp"

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
  m_frameFenced = false;
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
  b.createdFrame  = m_frame;
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
  m_computeBeforeBuild.erase(handle.value);
  m_computeBeforeCompute.erase(handle.value);
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
  overwritten(*b, "upload");
  written(dst);
}

void FakeGpuBackend::written(BufferHandle handle)
{
  m_writtenBeforeBuild.insert(handle.value);
  m_writtenBeforeShading.insert(handle.value);
}

void FakeGpuBackend::overwritten(const Buffer& b, const char* op)
{
  if(b.createdFrame < m_frame && !m_frameFenced)
    error(std::string(op) + ": overwrites '" + (b.desc.name ? b.desc.name : "?")
          + "', which a frame in flight may still read, with no frame-to-write barrier before");
}

void FakeGpuBackend::readByBuild(uint64_t address, const char* op)
{
  for(const auto& [handle, b] : m_buffers)
    if(address >= b.address && address < b.address + std::max<uint64_t>(b.desc.size, 1))
    {
      if(m_writtenBeforeBuild.count(handle))
        error(std::string(op) + ": reads a buffer written with no transfer-to-build barrier between");
      if(m_computeBeforeBuild.count(handle))
        error(std::string(op) + ": reads a deform output with no compute-to-build barrier between");
    }
}

std::vector<std::string> FakeGpuBackend::unfencedForTheFrame() const
{
  std::vector<std::string> out;
  if(!m_writtenBeforeShading.empty())
    out.push_back(std::to_string(m_writtenBeforeShading.size()) + " buffers written with no transfer-to-shader barrier after");
  if(!m_computeBeforeBuild.empty())
    out.push_back(std::to_string(m_computeBeforeBuild.size()) + " deform outputs with no compute-to-build barrier after");
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
  if(m_writtenBeforeShading.count(src.value) || m_computeBeforeCompute.count(src.value))
    error("copyBuffer: copies from a buffer written this frame with no barrier between");
  if(size)
    std::memmove(d->bytes.data() + dstOffset, s->bytes.data() + srcOffset, size);
  overwritten(*d, "copyBuffer");
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
  b.builtFrame  = m_frame;
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
  if(b.builtFrame < m_frame && !m_frameFenced)
    error("refitBlas: rewrites a BLAS a frame in flight may still trace, with no frame-to-write barrier before");
  b.input       = input;
  b.contentHash = contentHash(input);
  b.builtFrame  = m_frame;
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
  if(it->second.builtFrame == m_frame)
    return {};  // the size comes once the frame that built it has run
  Blas copy        = it->second;
  copy.compacted   = true;
  copy.builtFrame  = m_frame;
  m_blasBeforeTlas = m_buildsBeforeTrace = true;
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
  if(t.builds + t.updates > 0 && t.builtFrame < m_frame && !m_frameFenced)
    error("buildTlas: rebuilds a TLAS a frame in flight may still trace, with no frame-to-write barrier before");
  t.builtFrame = m_frame;
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

namespace {

// What a deform pass reads or writes, as typed elements over a buffer's bytes
template <class T>
struct Span
{
  std::byte* at    = nullptr;
  uint64_t   count = 0;
  T          get(uint64_t i) const
  {
    T v;
    std::memcpy(&v, at + i * sizeof(T), sizeof(T));
    return v;
  }
  void set(uint64_t i, const T& v) const { std::memcpy(at + i * sizeof(T), &v, sizeof(T)); }
  explicit operator bool() const { return at != nullptr; }
};

}  // namespace

// Runs the pass on the shadow bytes as the shader would: shaders/morph.comp.slang and
// skinning.comp.slang, element for element
void FakeGpuBackend::dispatchDeform(const DeformDispatch& d)
{
  m_deforms.push_back(d);
  const uint64_t n  = d.vertexCount;
  bool           ok = true;
  // `count` elements of `size` bytes at a range; none (and fine) for no buffer
  const auto range = [&](const BufferRange& r, uint64_t size, uint64_t count, const char* what, bool output) -> std::byte* {
    if(!r.buffer)
      return nullptr;
    Buffer* b = liveBuffer(r.buffer, "dispatchDeform");
    if(!b)
    {
      ok = false;
      return nullptr;
    }
    if(r.offset > b->desc.size || size * count > b->desc.size - r.offset)
    {
      error(std::string("dispatchDeform: its ") + what + " run past the end of their buffer");
      ok = false;
      return nullptr;
    }
    if(r.offset % 4)
      error(std::string("dispatchDeform: its ") + what + " are not 4-byte aligned");
    if(output)
      overwritten(*b, "dispatchDeform");
    else
    {
      if(m_writtenBeforeShading.count(r.buffer.value))
        error(std::string("dispatchDeform: reads its ") + what + " uploaded with no transfer-to-shader barrier between");
      if(m_computeBeforeCompute.count(r.buffer.value))
        error(std::string("dispatchDeform: reads its ") + what + " written by a deform pass with no compute-to-compute barrier between");
    }
    return b->bytes.data() + r.offset;
  };
  // the figure passes, in floats as the shaders run them
  if(d.kind == DeformKind::eTangents)
  {
    const uint64_t         tc = d.triangleCount;
    const Span<glm::vec3>  pos{range(d.basePositions, 12, n, "base positions", false), n};
    const Span<glm::vec3>  nrm{range(d.baseNormals, 12, n, "base normals", false), n};
    const Span<glm::vec2>  uv{d.uvs.buffer ? range(d.uvs, 8, n, "uvs", false) : nullptr, n};
    const Span<uint32_t>   start{range(d.rowStart, 4, n + 1, "corner starts", false), n + 1};
    const uint64_t         entries = start ? start.get(n) : 0;
    const Span<uint32_t>   corners{range(d.columns, 4, entries, "corners", false), entries};
    const Span<glm::uvec3> tris{range(d.triangles, 12, tc, "triangles", false), tc};
    const Span<glm::vec4>  out{range(d.outTangents, 16, n, "outputs", true), n};
    if(!ok || !pos || !nrm || !start || !tris || !out || (entries && !corners))
    {
      if(ok)
        error("dispatchDeform: a tangent pass without its positions, normals, corners, triangles or output");
      return;
    }
    for(uint64_t v = 0; v < n; v++)
    {
      glm::vec4 t(0.f);
      for(uint32_t k = start.get(v); k < start.get(v + 1) && k < entries; k++)
      {
        const uint32_t f = corners.get(k) / 3;
        if(f >= tc)
          continue;
        const glm::uvec3 i = tris.get(f);
        if(i.x >= n || i.y >= n || i.z >= n)
          continue;
        const glm::vec3 n0 = nrm.get(i.x);
        if(!uv)
        {
          t = shaderio::makeFastTangent(n0);
          continue;
        }
        const glm::vec3 p0 = pos.get(i.x), edge1 = pos.get(i.y) - p0, edge2 = pos.get(i.z) - p0;
        const glm::vec2 uv0 = uv.get(i.x), deltaUV1 = uv.get(i.y) - uv0, deltaUV2 = uv.get(i.z) - uv0;
        float           r = 1.0F;
        const float     a = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        if(fabs(a) > 0.0F)
          r = 1.0f / a;
        const glm::vec3 tangent    = r * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
        const glm::vec3 bitangent  = r * (deltaUV2.x * edge1 - deltaUV1.x * edge2);
        const float     handedness = (glm::dot(glm::cross(tangent, bitangent), n0) > 0.0F) ? 1.0F : -1.0F;
        t                          = glm::vec4(tangent + glm::vec3(t), handedness);
      }
      const glm::vec3 n0  = nrm.get(v);
      glm::vec3       ot0 = glm::normalize(glm::vec3(t) - (glm::dot(n0, glm::vec3(t)) * n0));
      if(glm::length2(ot0) < 0.1F || glm::any(glm::isnan(ot0)))
        ot0 = glm::vec3(shaderio::makeFastTangent(n0));
      out.set(v, glm::vec4(ot0, t.w < 0.f ? -1.f : 1.f));
    }
    m_computeBeforeBuild.insert(d.outTangents.buffer.value);
    m_computeBeforeCompute.insert(d.outTangents.buffer.value);
    return;
  }
  if(d.kind == DeformKind::eStencil || d.kind == DeformKind::eNormals)
  {
    const bool            stencil = d.kind == DeformKind::eStencil;
    const uint64_t        m       = d.sourceCount;
    const Span<glm::vec3> base{range(d.basePositions, 12, m, "base positions", false), m};
    const Span<uint32_t>  start{range(d.rowStart, 4, n + 1, "row starts", false), d.rowStart.buffer ? n + 1 : 0};
    const uint64_t        entries = start ? start.get(n) : n;
    const Span<uint32_t>  columns{range(d.columns, 4, entries, "columns", false), entries};
    const Span<float>     w{stencil ? range(d.stencilWeights, 4, entries, "stencil weights", false) : nullptr, entries};
    const Span<glm::vec3> offsets{stencil && start ? range(d.stencilOffsets, 12, n, "stencil offsets", false) : nullptr, n};
    const Span<glm::uvec3> tris{stencil ? nullptr : range(d.triangles, 12, d.triangleCount, "triangles", false), d.triangleCount};
    const Span<glm::vec3> out{range(stencil ? d.outPositions : d.outNormals, 12, n, "outputs", true), n};
    if(!ok || !base || !columns || !out || (!stencil && (!start || !tris)))
    {
      if(ok)
        error("dispatchDeform: a figure pass without its base, columns, triangles or output");
      return;
    }
    if((stencil ? d.outPositions : d.outNormals).buffer == d.basePositions.buffer)
      error("dispatchDeform: writes the buffer it reads");
    for(uint64_t r = 0; r < n; r++)
    {
      glm::vec3 sum(0.f);
      if(stencil && !start)
      {
        const uint32_t c = columns.get(r);
        out.set(r, c < m ? base.get(c) : glm::vec3(0.f));
        continue;
      }
      for(uint32_t k = start.get(r); k < start.get(r + 1) && k < entries; k++)
      {
        const uint32_t c = columns.get(k);
        if(stencil)
        {
          if(c < m)
            sum += (w ? w.get(k) : 1.f) * base.get(c);
          continue;
        }
        if(c >= d.triangleCount)
          continue;
        const glm::uvec3 t = tris.get(c);
        if(t.x >= m || t.y >= m || t.z >= m)
          continue;
        const glm::vec3 a = base.get(t.x);
        sum += glm::cross(base.get(t.y) - a, base.get(t.z) - a);
      }
      if(offsets)
        sum += offsets.get(r);
      const float l = glm::length(sum);
      out.set(r, stencil ? sum : l > 0.f ? sum / l : glm::vec3(0.f));
    }
    const BufferRange& written = stencil ? d.outPositions : d.outNormals;
    m_computeBeforeBuild.insert(written.buffer.value);
    m_computeBeforeCompute.insert(written.buffer.value);
    return;
  }
  const Span<glm::vec3> basePos{range(d.basePositions, 12, n, "base positions", false), n};
  const Span<glm::vec3> baseNrm{range(d.baseNormals, 12, n, "base normals", false), n};
  const Span<glm::vec4> baseTan{range(d.baseTangents, 16, n, "base tangents", false), n};
  Span<glm::vec3>       posDelta, nrmDelta, tanDelta;
  Span<float>           weights;
  Span<glm::ivec4>      joints, joints1;
  Span<glm::vec4>       jointWeights, jointWeights1;
  Span<glm::mat4>       jointMats;
  Span<glm::mat3>       normalMats;
  Span<JointDualQuat>   dualQuats;
  Span<float>           shares;
  if(d.kind == DeformKind::eMorph)
  {
    const uint64_t deltas = n * d.targetCount;
    posDelta              = {range(d.positionDeltas, 12, deltas, "position deltas", false), deltas};
    nrmDelta              = {range(d.normalDeltas, 12, deltas, "normal deltas", false), deltas};
    tanDelta              = {range(d.tangentDeltas, 12, deltas, "tangent deltas", false), deltas};
    weights               = {range(d.morphWeights, 4, d.targetCount, "morph weights", false), d.targetCount};
    if(!posDelta || !weights)
      error("dispatchDeform: a morph without position deltas or weights");
  }
  else
  {
    joints       = {range(d.joints, 16, n, "joints", false), n};
    jointWeights = {range(d.weights, 16, n, "weights", false), n};
    jointMats    = {range(d.jointMatrices, 64, d.jointCount, "joint matrices", false), d.jointCount};
    normalMats   = {range(d.normalMatrices, kNormalMatrixStride, d.jointCount, "normal matrices", false), d.jointCount};
    if(!joints || !jointWeights || !jointMats || !normalMats)
      error("dispatchDeform: a skin without joints, weights or matrices");
    if(d.joints1.buffer && d.weights1.buffer)
    {
      joints1       = {range(d.joints1, 16, n, "joints 5-8", false), n};
      jointWeights1 = {range(d.weights1, 16, n, "weights 5-8", false), n};
    }
    if(d.jointDualQuats.buffer)
      dualQuats = {range(d.jointDualQuats, sizeof(JointDualQuat), d.jointCount, "joint dual quaternions", false), d.jointCount};
    if(d.dualQuatShares.buffer)
      shares = {range(d.dualQuatShares, 4, n, "dual quaternion shares", false), n};
  }
  const Span<glm::vec3> outPos{range(d.outPositions, 12, n, "output positions", true), n};
  const Span<glm::vec3> outNrm{range(d.outNormals, 12, n, "output normals", true), n};
  const Span<glm::vec4> outTan{range(d.outTangents, 16, n, "output tangents", true), n};
  if(!ok || !basePos || !outPos)
  {
    if(ok)
      error("dispatchDeform: no base or output positions");
    return;
  }
  // a pass reads its whole base before it writes (outputs and bases are never the same buffer)
  if(d.outPositions.buffer == d.basePositions.buffer)
    error("dispatchDeform: writes the buffer it reads");
  const bool hasNormals = baseNrm && outNrm, hasTangents = baseTan && outTan;
  // a skin's joints, as the shader reads them
  std::vector<glm::mat4>     skinMats;
  std::vector<glm::mat3>     skinNormalMats;
  std::vector<JointDualQuat> skinDualQuats;
  if(d.kind == DeformKind::eSkin && jointMats && normalMats)
    for(uint32_t j = 0; j < d.jointCount; j++)
    {
      skinMats.push_back(jointMats.get(j));
      skinNormalMats.push_back(normalMats.get(j));
      if(dualQuats)
        skinDualQuats.push_back(dualQuats.get(j));
    }
  for(uint64_t v = 0; v < n; v++)
  {
    if(d.kind == DeformKind::eMorph)
    {
      glm::vec3 pos = basePos.get(v);
      glm::vec3 nrm = baseNrm ? baseNrm.get(v) : glm::vec3(0.f);
      glm::vec4 tan = baseTan ? baseTan.get(v) : glm::vec4(0.f);
      for(uint32_t t = 0; t < d.targetCount; t++)
      {
        const float w = weights ? weights.get(t) : 0.f;
        if(w == 0.f || !posDelta)
          continue;
        const uint64_t i = uint64_t(t) * n + v;
        pos += w * posDelta.get(i);
        if(baseNrm && nrmDelta)
          nrm += w * nrmDelta.get(i);
        if(baseTan && tanDelta)
          tan = glm::vec4(glm::vec3(tan) + w * tanDelta.get(i), tan.w);
      }
      outPos.set(v, pos);
      if(hasNormals)
        outNrm.set(v, glm::normalize(nrm));
      if(hasTangents)
        outTan.set(v, glm::vec4(glm::normalize(glm::vec3(tan)), tan.w));
      continue;
    }
    if(!joints || !jointWeights || !jointMats || !normalMats)
      return;
    const glm::ivec4 j[2] = {joints.get(v), joints1 ? joints1.get(v) : glm::ivec4(0)};
    const glm::vec4  w[2] = {jointWeights.get(v), jointWeights1 ? jointWeights1.get(v) : glm::vec4(0.f)};
    const int        sets = joints1 && jointWeights1 ? 2 : 1;
    const glm::vec3  nrm  = hasNormals ? baseNrm.get(v) : glm::vec3(0.f);
    const glm::vec4  tan  = hasTangents ? baseTan.get(v) : glm::vec4(0.f);
    const float      share = !dualQuats ? 0.f : shares ? shares.get(v) : d.dualQuat;
    const SkinnedVertex out = skinVertex(basePos.get(v), hasNormals ? &nrm : nullptr, hasTangents ? &tan : nullptr,
                                         {j, size_t(sets)}, {w, size_t(sets)}, {skinMats, skinNormalMats, skinDualQuats}, share);
    outPos.set(v, out.position);
    if(hasNormals)
      outNrm.set(v, out.normal);
    if(hasTangents)
      outTan.set(v, out.tangent);
  }
  for(const BufferRange* r : {&d.outPositions, &d.outNormals, &d.outTangents})
    if(r->buffer)
    {
      m_computeBeforeBuild.insert(r->buffer.value);
      m_computeBeforeCompute.insert(r->buffer.value);
    }
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
      m_computeBeforeBuild.clear();
      m_computeBeforeCompute.clear();
      break;
    case GpuBarrier::eComputeToCompute:
      m_computeBeforeCompute.clear();
      break;
    case GpuBarrier::eFrameToWrite:
      m_frameFenced = true;
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
