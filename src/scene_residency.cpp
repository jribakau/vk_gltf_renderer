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

#include "scene_residency.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string_view>

#include <glm/gtc/packing.hpp>

#include <figure/io/timing.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/id_pool.hpp>

#include "ies_profile.hpp"
#include "scene_gpu_tables.hpp"
#include "shaders/gltf_scene_io.h.slang"

namespace rs {

namespace {

// Slot pools: a TLAS row's custom index is 24 bits, so neither instance nor prim slots go past it
constexpr uint32_t kSlotPoolSize     = 1u << 24;
constexpr uint32_t kMaterialPoolSize = 1u << 16;
// the descriptor arrays' slots; 0 is the default image and the default sampler
constexpr uint32_t kTexturePoolSize = 1u << 16;
constexpr uint32_t kSamplerPoolSize = 1u << 12;

std::tuple<int, int, int, int> keyOf(const GpuSamplerDesc& s)
{
  return {s.magFilter, s.minFilter, s.wrapS, s.wrapT};
}

GpuSamplerDesc samplerOf(const SamplerDesc& s)
{
  return {.magFilter = s.magFilter, .minFilter = s.minFilter, .wrapS = s.wrapS, .wrapT = s.wrapT};
}

// A pool whose first slot is taken (the default image or sampler)
std::shared_ptr<nvutils::IDPool> poolFromOne(uint32_t size)
{
  auto     pool = std::make_shared<nvutils::IDPool>(size);
  uint32_t zero = 0;
  pool->createID(zero);
  return pool;
}

template <class T>
std::span<const std::byte> bytesOf(const std::vector<T>& v)
{
  return std::as_bytes(std::span(v));
}

template <class T>
std::span<const std::byte> bytesOf(const T& value)
{
  return std::as_bytes(std::span(&value, 1));
}

uint32_t capacityFor(uint32_t count, uint32_t minimum)
{
  return std::max(minimum, std::bit_ceil(std::max(count, 1u)));
}

// [first, count) runs of consecutive slots
std::vector<std::pair<uint32_t, uint32_t>> runsOf(const std::set<uint32_t>& slots)
{
  std::vector<std::pair<uint32_t, uint32_t>> runs;
  for(uint32_t s : slots)
    if(!runs.empty() && runs.back().first + runs.back().second == s)
      runs.back().second++;
    else
      runs.push_back({s, 1});
  return runs;
}

TlasRow tlasRowOf(const glm::mat4& world)
{
  TlasRow row;
  for(int r = 0; r < 3; r++)
    for(int c = 0; c < 4; c++)
      row.transform[r][c] = world[c][r];
  return row;
}

}  // namespace

Residency::Residency(GpuBackend& gpu, ResidencyOptions options)
    : m_gpu(gpu)
    , m_options(options)
    , m_texturePool(poolFromOne(kTexturePoolSize))
    , m_samplerPool(poolFromOne(kSamplerPoolSize))
{
  clear();
}

Residency::~Residency()
{
  clear();
  trimGrace(0);
}

std::optional<TextureKey> textureKeyOf(const TextureRecord& record, bool srgb)
{
  const auto hashOf = [](const std::vector<unsigned char>& v) {
    return uint64_t(std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(v.data()), v.size())));
  };
  TextureKey key{.srgb = srgb};
  if(!record.bytes.empty())
  {
    key.size = record.bytes.size();
    key.hash = hashOf(record.bytes);
  }
  else if(!record.pixels.empty())
  {
    // the same pixels at another shape are another image
    key.path = std::to_string(record.width) + "x" + std::to_string(record.height) + "x"
               + std::to_string(record.component) + "x" + std::to_string(record.bits);
    key.size = record.pixels.size();
    key.hash = hashOf(record.pixels);
  }
  else if(!record.uri.empty())
  {
    std::error_code                        ec;
    const std::filesystem::path            path = std::filesystem::absolute(nvutils::pathFromUtf8(record.uri), ec);
    const std::filesystem::directory_entry entry(path, ec);
    if(ec || !entry.is_regular_file(ec))
      return std::nullopt;
    key.path  = nvutils::utf8FromPath(path);
    key.size  = entry.file_size(ec);
    key.mtime = int64_t(entry.last_write_time(ec).time_since_epoch().count());
    if(ec)
      return std::nullopt;
  }
  else
    return std::nullopt;
  return key;
}

void Residency::clear()
{
  for(auto& [id, mesh] : m_meshes)
    for(BufferHandle b : {mesh.positions, mesh.normals, mesh.tangents, mesh.uv0, mesh.uv1, mesh.colors})
      retireBuffer(b);
  for(const std::optional<Prim>& p : m_prims)
    if(p)
    {
      retireBuffer(p->indices);
      retireAccel(p->blas);
    }
  // descriptor slots are the scene descriptor set's, which a frame in flight still reads: each goes
  // back to its pool only at its retire, so the next scene is handed other slots meanwhile
  for(const auto& [key, use] : m_textureSlots)
    m_gpu.retire([&gpu = m_gpu, pool = m_texturePool, slot = use.slot] {
      gpu.writeTextureDescriptor(slot, {});
      pool->destroyID(slot);
    });
  for(const auto& [key, use] : m_samplerSlots)
    m_gpu.retire([pool = m_samplerPool, slot = use.slot] { pool->destroyID(slot); });
  for(const auto& [id, image] : m_images)
    releaseImage(image);
  retireBuffer(m_renderNodes);
  retireBuffer(m_renderPrims);
  retireBuffer(m_sceneDesc);
  retireBuffer(m_materialTable);
  retireBuffer(m_textureInfoTable);
  retireBuffer(m_lightTable);
  retireBuffer(m_emissiveTable);
  retireBuffer(m_iesTable);
  m_lightTable = m_emissiveTable = m_iesTable = {};
  m_lightCount = m_emissiveCount = m_iesProfileCount = 0;
  m_emissiveTotalWeight = m_emissiveMeanLum = 0.f;
  m_emissiveStale                           = false;
  m_shadeMaterials.clear();
  m_descDirty = false;
  retireAccel(m_tlas);
  m_meshes.clear();
  m_prims.clear();
  m_instances.clear();
  m_slotOf.clear();
  m_materials.clear();
  m_images.clear();
  m_textureSlots.clear();
  m_samplerSlots.clear();
  m_materialStates.clear();
  m_materialHighWater = 0;
  m_materialTable = m_textureInfoTable = {};
  m_materialsDirty                     = false;
  for(auto& list : m_drawLists)
    list.clear();
  m_drawListsRevision++;
  m_hasTransmission = false;
  m_renderNodes = m_renderPrims = m_sceneDesc = {};
  m_tlas                                      = {};
  m_instanceCapacity = m_primCapacity = m_tlasCapacity = 0;
  // fresh slot pools: their slots index the tables retired above, so they are never read again
  m_dirtyNodes.clear();
  m_allNodes = m_allPrims = false;
  m_instancePool          = std::make_shared<nvutils::IDPool>(kSlotPoolSize);
  m_primPool              = std::make_shared<nvutils::IDPool>(kSlotPoolSize);
  m_materialPool          = std::make_shared<nvutils::IDPool>(kMaterialPoolSize);
}

// ---------- Retiring ----------

void Residency::retireBuffer(BufferHandle buffer)
{
  if(buffer)
    m_gpu.retire([&gpu = m_gpu, buffer] { gpu.destroyBuffer(buffer); });
}

void Residency::retireAccel(AccelHandle accel)
{
  if(accel)
    m_gpu.retire([&gpu = m_gpu, accel] { gpu.destroyAccel(accel); });
}

// ---------- Records ----------

void Residency::uploadStreams(const MeshRecord& record, MeshGpu& gpu)
{
  const auto stream = [&](BufferHandle& buffer, std::span<const std::byte> bytes, BufferUsageFlags usage, const char* name) {
    if(bytes.empty())
      return;
    if(!buffer || m_gpu.bufferSize(buffer) != bytes.size())
    {
      retireBuffer(buffer);
      buffer = m_gpu.createBuffer({.size = bytes.size(), .usage = usage, .name = name});
    }
    m_gpu.upload(buffer, 0, bytes);
  };
  stream(gpu.positions, bytesOf(record.positions), eBufferVertex | eBufferStorage | eBufferAccelInput, "positions");
  stream(gpu.normals, bytesOf(record.normals), eBufferStorage, "normals");
  stream(gpu.tangents, bytesOf(record.tangents), eBufferStorage, "tangents");
  stream(gpu.uv0, bytesOf(record.uv0), eBufferStorage, "uv0");
  stream(gpu.uv1, bytesOf(record.uv1), eBufferStorage, "uv1");
  if(!record.colors.empty())
  {
    std::vector<uint32_t> packed(record.colors.size());
    for(size_t i = 0; i < packed.size(); i++)
      packed[i] = glm::packUnorm4x8(record.colors[i]);
    stream(gpu.colors, bytesOf(packed), eBufferStorage, "colors");
  }
  gpu.vertexCount = uint32_t(record.positions.size());
  gpu.contentRev  = record.contentRev;
}

Residency::MeshGpu& Residency::acquireMesh(const RetainedScene& scene, MeshId id)
{
  auto it = m_meshes.find(id);
  if(it != m_meshes.end())
    return it->second;
  MeshGpu& gpu = m_meshes[id];
  if(const MeshRecord* record = scene.mesh(id))
  {
    uploadStreams(*record, gpu);
    gpu.prims.assign(record->submeshes.size(), -1);
  }
  return gpu;
}

void Residency::retireMesh(MeshId id)
{
  auto it = m_meshes.find(id);
  if(it == m_meshes.end())
    return;
  for(BufferHandle b :
      {it->second.positions, it->second.normals, it->second.tangents, it->second.uv0, it->second.uv1, it->second.colors})
    retireBuffer(b);
  m_meshes.erase(it);
}

// The box around the vertices a prim's indices name; none named: a point at the origin
static void fitBounds(Residency::Prim& prim, const MeshRecord& record)
{
  const Submesh& run = record.submeshes[prim.submesh];
  glm::vec3      lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
  for(uint32_t i = run.firstIndex; i < run.firstIndex + run.indexCount; i++)
    if(record.indices[i] < record.positions.size())
    {
      lo = glm::min(lo, record.positions[record.indices[i]]);
      hi = glm::max(hi, record.positions[record.indices[i]]);
    }
  prim.boundsMin = lo.x <= hi.x ? lo : glm::vec3(0.f);
  prim.boundsMax = lo.x <= hi.x ? hi : glm::vec3(0.f);
}

uint32_t Residency::acquirePrim(const RetainedScene& scene, MeshId id, uint32_t submesh)
{
  MeshGpu& mesh = acquireMesh(scene, id);
  if(submesh < mesh.prims.size() && mesh.prims[submesh] >= 0)
  {
    m_prims[size_t(mesh.prims[submesh])]->users++;
    return uint32_t(mesh.prims[submesh]);
  }
  uint32_t   slot = 0;
  const bool ok   = m_primPool->createID(slot);
  assert(ok && "out of prim slots");
  (void)ok;
  if(slot >= m_prims.size())
    m_prims.resize(size_t(slot) + 1);

  const MeshRecord& record = *scene.mesh(id);
  const Submesh&    run    = record.submeshes[submesh];
  Prim              prim;
  prim.mesh       = id;
  prim.submesh    = submesh;
  prim.indexCount = run.indexCount;
  prim.users      = 1;
  const std::span<const uint32_t> indices(record.indices.data() + run.firstIndex, run.indexCount);
  prim.indices = m_gpu.createBuffer(
      {.size = indices.size_bytes(), .usage = eBufferIndex | eBufferStorage | eBufferAccelInput, .name = "indices"});
  m_gpu.upload(prim.indices, 0, std::as_bytes(indices));
  prim.input          = {.vertexAddress = m_gpu.bufferAddress(mesh.positions),
                         .vertexStride  = sizeof(glm::vec3),
                         .vertexCount   = mesh.vertexCount,
                         .indexAddress  = m_gpu.bufferAddress(prim.indices),
                         .indexCount    = run.indexCount,
                         .opaque        = false};
  fitBounds(prim, record);
  m_prims[slot]       = std::move(prim);
  mesh.prims[submesh] = int32_t(slot);
  return slot;
}

void Residency::releasePrim(uint32_t slot)
{
  Prim& prim = *m_prims[slot];
  if(--prim.users > 0)
    return;
  retireBuffer(prim.indices);
  retireAccel(prim.blas);
  auto mesh = m_meshes.find(prim.mesh);
  if(mesh != m_meshes.end())
  {
    mesh->second.prims[prim.submesh] = -1;
    if(std::all_of(mesh->second.prims.begin(), mesh->second.prims.end(), [](int32_t p) { return p < 0; }))
      retireMesh(prim.mesh);
  }
  m_prims[slot].reset();
  m_gpu.retire([pool = m_primPool, slot] { pool->destroyID(slot); });
}

uint32_t Residency::acquireMaterial(const RetainedScene& scene, MaterialId id)
{
  auto it = m_materials.find(id);
  if(it != m_materials.end())
  {
    it->second.users++;
    return it->second.slot;
  }
  uint32_t   slot = 0;
  const bool ok   = m_materialPool->createID(slot);
  assert(ok && "out of material slots");
  (void)ok;
  MaterialSlot& m = m_materials[id];
  m.slot          = slot;
  m.users         = 1;
  m.rev           = scene.material(id) ? scene.material(id)->rev : 0;
  acquireMaterialTextures(scene, id, m);
  if(slot >= m_materialStates.size())
    m_materialStates.resize(size_t(slot) + 1);
  m_materialStates[slot] = MaterialState{.material = id};
  m_materialHighWater    = std::max(m_materialHighWater, slot + 1);
  m_materialsDirty       = true;
  return slot;
}

void Residency::releaseMaterial(MaterialId id)
{
  auto it = m_materials.find(id);
  if(it == m_materials.end() || --it->second.users > 0)
    return;
  const uint32_t slot = it->second.slot;
  releaseMaterialTextures(it->second);
  m_materials.erase(it);
  m_materialStates[slot].reset();
  m_materialsDirty = true;
  m_gpu.retire([pool = m_materialPool, slot] { pool->destroyID(slot); });
}

// The textures the payload samples (as MaterialCache reads it), each with its texture slot
void Residency::acquireMaterialTextures(const RetainedScene& scene, MaterialId id, MaterialSlot& slot)
{
  const MaterialRecord* record = scene.material(id);
  slot.textures.clear();
  if(!record || !record->textures)
    return;
  nvvkgltf::MaterialCache sampled;
  sampled.append(record->payload, {});
  const std::set<int> srgb = nvvkgltf::srgbTextures(record->payload);
  slot.textures.resize(record->textures->size());
  for(size_t t = 1; t < sampled.getTextureInfos().size(); t++)
  {
    const int i = sampled.getTextureInfos()[t].index;
    if(i < 0 || i >= int(record->textures->size()) || slot.textures[size_t(i)])
      continue;
    const TextureRef&    ref     = (*record->textures)[size_t(i)];
    const GpuSamplerDesc sampler = samplerOf(ref.sampler);
    acquireTexture(scene, ref.texture, sampler, srgb.count(i) > 0);
    slot.textures[size_t(i)] = std::pair{ref.texture, sampler};
  }
}

void Residency::releaseMaterialTextures(MaterialSlot& slot)
{
  for(const auto& t : slot.textures)
    if(t)
      releaseTexture(t->first, t->second);
  slot.textures.clear();
}

void Residency::acquireTexture(const RetainedScene& scene, TextureId id, const GpuSamplerDesc& sampler, bool srgb)
{
  ImageGpu& image = m_images[id];
  if(image.users++ == 0)
    if(const TextureRecord* record = scene.texture(id))
    {
      // decoded already, for another texture or in grace
      image.key = textureKeyOf(*record, srgb);
      if(image.key)
        if(auto it = m_decoded.find(*image.key); it != m_decoded.end())
        {
          DecodedImage& decoded = it->second;
          if(decoded.users++ == 0)
            m_graceBytes -= decoded.info.bytes;
          image.image = decoded.image;
          image.info  = decoded.info;
          m_imageCounts.reused++;
        }
    }
  if(image.users == 1 && !image.image)
    if(const TextureRecord* record = scene.texture(id))
    {
      const ImageSource source{.uri       = record->uri,
                               .bytes     = record->bytes,
                               .mimeType  = record->mimeType,
                               .pixels    = record->pixels,
                               .width     = record->width,
                               .height    = record->height,
                               .component = record->component,
                               .bits      = record->bits,
                               .srgb      = srgb,
                               .name      = record->name};
      image.image = m_gpu.createImage(source, image.info);
      m_imageCounts.decoded++;
      if(image.key && image.image)
        m_decoded[*image.key] = {.image = image.image, .info = image.info, .users = 1};
      else
        image.key.reset();  // an unreadable one is tried again next time
    }

  SlotUse& use = m_textureSlots[{id, keyOf(sampler)}];
  if(use.users++ > 0)
    return;
  const bool ok = m_texturePool->createID(use.slot);
  assert(ok && "out of texture slots");
  (void)ok;
  m_gpu.writeTextureDescriptor(use.slot, image.image);  // none: the default image
  SlotUse& samplerUse = m_samplerSlots[keyOf(sampler)];
  if(samplerUse.users++ == 0)
  {
    const bool okSampler = m_samplerPool->createID(samplerUse.slot);
    assert(okSampler && "out of sampler slots");
    (void)okSampler;
    m_gpu.writeSamplerDescriptor(samplerUse.slot, sampler);
  }
}

// A slot nothing samples any more goes back to the default image, and to its pool, at its retire
// frame: a frame in flight may still sample it.
void Residency::releaseTexture(TextureId id, const GpuSamplerDesc& sampler)
{
  auto slot = m_textureSlots.find({id, keyOf(sampler)});
  if(slot != m_textureSlots.end() && --slot->second.users == 0)
  {
    m_gpu.retire([&gpu = m_gpu, pool = m_texturePool, s = slot->second.slot] {
      gpu.writeTextureDescriptor(s, {});
      pool->destroyID(s);
    });
    m_textureSlots.erase(slot);
    auto samplerUse = m_samplerSlots.find(keyOf(sampler));
    if(samplerUse != m_samplerSlots.end() && --samplerUse->second.users == 0)
    {
      m_gpu.retire([pool = m_samplerPool, s = samplerUse->second.slot] { pool->destroyID(s); });
      m_samplerSlots.erase(samplerUse);
    }
  }
  auto image = m_images.find(id);
  if(image != m_images.end() && --image->second.users == 0)
  {
    releaseImage(image->second);
    m_images.erase(image);
  }
}

// A texture's image let go of: a shared decode goes to grace when nothing uses it (reusing it is
// safe at once, since its contents never change; destroying it waits for trimGrace's retire), its
// own image to the backend at retire
void Residency::releaseImage(const ImageGpu& image)
{
  if(image.key)
  {
    auto it = m_decoded.find(*image.key);
    if(it != m_decoded.end() && --it->second.users == 0)
    {
      it->second.released = ++m_releaseClock;
      m_graceBytes += it->second.info.bytes;
    }
  }
  else if(image.image)
    m_gpu.retire([&gpu = m_gpu, handle = image.image] { gpu.destroyImage(handle); });
}

// The idle decodes, oldest first, destroyed at retire until the rest fit `budget`
void Residency::trimGrace(uint64_t budget)
{
  while(m_graceBytes > budget)
  {
    auto oldest = m_decoded.end();
    for(auto it = m_decoded.begin(); it != m_decoded.end(); ++it)
      if(it->second.users == 0 && (oldest == m_decoded.end() || it->second.released < oldest->second.released))
        oldest = it;
    if(oldest == m_decoded.end())
      break;
    m_graceBytes -= oldest->second.info.bytes;
    m_gpu.retire([&gpu = m_gpu, handle = oldest->second.image] { gpu.destroyImage(handle); });
    m_decoded.erase(oldest);
  }
  if(m_graceBytes > budget)  // only idle decodes count, so this can't be
    m_graceBytes = 0;
}

void Residency::setTextureGrace(uint64_t bytes)
{
  m_options.textureGraceBytes = bytes;
  trimGrace(bytes);
}

// Materials whose payload was edited take the textures it now samples (the new ones first, so the
// ones kept are not let go of in between).
void Residency::syncMaterials(RetainedScene& scene, const std::set<MaterialId>& edited, SyncResult& result)
{
  for(MaterialId id : edited)
  {
    auto                  it     = m_materials.find(id);
    const MaterialRecord* record = scene.material(id);
    if(it == m_materials.end() || !record || it->second.rev == record->rev)
      continue;
    MaterialSlot before = it->second;
    acquireMaterialTextures(scene, id, it->second);
    releaseMaterialTextures(before);
    it->second.rev   = record->rev;
    m_materialsDirty = result.changed = true;
  }
}

// The material and texture-info tables converted whole from the live material slots, a hole as the
// default material. Notes whether a material's draw bits changed (m_bitsChanged) and whether a
// table had to be made again (m_tablesMoved).
void Residency::writeMaterials(const RetainedScene& scene)
{
  nvvkgltf::MaterialCache cache;
  cache.buildFromMaterials({}, {});
  for(uint32_t s = 0; s < m_materialHighWater; s++)
  {
    std::optional<MaterialState>& state = m_materialStates[s];
    if(!state)
    {
      cache.append(tinygltf::Material{}, {});
      continue;
    }
    const MaterialRecord* record      = scene.material(state->material);
    const MaterialSlot&   slot        = m_materials.at(state->material);
    state->payload                    = record ? record->payload : tinygltf::Material{};
    nvvkgltf::TextureSlotTable& table = state->slots;
    table                             = {};
    const size_t n                    = slot.textures.size();
    table.samplerSlots.assign(n, 0);
    table.twoChannelSource.assign(n, 0);
    table.meanLuminance.assign(n, -1.f);
    table.gpuIndex.assign(n, -1);
    for(size_t i = 0; i < n; i++)
    {
      if(!slot.textures[i])
        continue;
      const auto& [texture, sampler] = *slot.textures[i];
      const ImageGpu& image          = m_images.at(texture);
      table.gpuIndex[i]              = int(m_textureSlots.at({texture, keyOf(sampler)}).slot);
      table.samplerSlots[i]          = int(m_samplerSlots.at(keyOf(sampler)).slot);
      table.twoChannelSource[i]      = image.info.twoChannel ? 1 : 0;
      table.meanLuminance[i]         = image.info.meanLuminance;
    }
    const nvvkgltf::MaterialBits before = state->bits;
    state->bits                         = nvvkgltf::materialBits(state->payload);
    m_bitsChanged |= state->bits != before;
    cache.append(state->payload, table);
  }

  const auto table = [&](BufferHandle& buffer, std::span<const std::byte> bytes, const char* name) {
    const uint64_t size = std::bit_ceil(std::max<uint64_t>(bytes.size(), 256));
    if(!buffer || m_gpu.bufferSize(buffer) < bytes.size())
    {
      retireBuffer(buffer);
      buffer = m_gpu.createBuffer({.size = size, .usage = eBufferStorage, .name = name});
      return true;
    }
    return false;
  };
  const auto& materials    = cache.getShadeMaterials();
  const auto& textureInfos = cache.getTextureInfos();
  m_tablesMoved |= table(m_materialTable, std::as_bytes(std::span(materials)), "materials");
  m_tablesMoved |= table(m_textureInfoTable, std::as_bytes(std::span(textureInfos)), "texture infos");
  m_gpu.upload(m_materialTable, 0, std::as_bytes(std::span(materials)));
  m_shadeMaterials = materials;
  m_gpu.upload(m_textureInfoTable, 0, std::as_bytes(std::span(textureInfos)));
  m_materialsDirty = false;
}

// The raster lists from the live instances' materials, in slot order
void Residency::buildDrawLists()
{
  for(auto& list : m_drawLists)
    list.clear();
  m_hasTransmission = false;
  for(uint32_t s = 0; s < m_instances.size(); s++)
  {
    const std::optional<Instance>& inst = m_instances[s];
    if(!inst)
      continue;
    const nvvkgltf::MaterialBits& bits = m_materialStates[inst->materialSlot]->bits;
    if(bits.solid())
      m_drawLists[size_t(DrawList::eSolid)].push_back(s);
    if(bits.solidDoubleSided())
      m_drawLists[size_t(DrawList::eSolidDoubleSided)].push_back(s);
    if(bits.blend())
      m_drawLists[size_t(DrawList::eBlend)].push_back(s);
    m_drawLists[size_t(DrawList::eAll)].push_back(s);
    m_hasTransmission |= bits.transmission;
  }
  m_drawListsRevision++;
}

// ---------- The sync ----------

SyncResult Residency::replace(RetainedScene& scene)
{
  clear();
  m_imageCounts     = {};
  SyncResult result = sync(scene);
  result.dlssReset  = true;
  result.changed    = true;
  return result;
}

SyncResult Residency::sync(RetainedScene& scene)
{
  SyncResult         result;
  const SceneChanges changes = scene.takeChanges();
  const bool         fresh   = !m_renderNodes;
  // an emitter's vertex writes have stopped long enough: the table takes the areas they left
  const bool emissiveSettled = m_emissiveStale && m_gpu.frame() >= m_emitterWriteFrame + m_options.emissiveSettleFrames;
  if(!fresh && changes.empty() && !emissiveSettled)
    return result;

  std::vector<DrawItem> items = scene.drawItems();
  if(fresh)
    m_materialsDirty = true;  // the tables exist from the first sync, even with no material in them
  if(fresh || changes.structure)
    syncStructure(scene, items, result);
  if(!changes.materials.empty())
    syncMaterials(scene, changes.materials, result);
  std::set<MeshId> written;
  if(!changes.meshes.empty())
    written = syncVertices(scene, changes.meshes, result);
  ensureCapacity(result);
  // textures before materials before geometry (see "The sync"): the textures are written as their
  // materials take them, and every material is converted here, before an instance draws with it
  if(m_materialsDirty)
  {
    writeMaterials(scene);
    result.changed = true;
  }
  if(m_tablesMoved)
    result.buffersChanged = true;
  if(result.drawListChanged || m_bitsChanged)
  {
    buildDrawLists();
    result.drawListChanged = true;
  }
  const bool materialsWritten = m_tablesMoved || result.drawListChanged || (!changes.materials.empty());
  syncRows(items, result);
  if(fresh || changes.iesProfiles)
  {
    syncIesProfiles(scene);
    result.changed = true;
  }
  if(fresh || changes.structure || !changes.lights.empty() || !changes.transforms.empty() || !changes.visibility.empty())
  {
    syncLights(scene);
    result.changed = true;
  }
  // the emissive table: its emitters came, went, moved, were shown or hidden, or their materials
  // changed (any slot change counts, so an entry never names a renumbered slot), this frame. Their
  // vertex writes only mark it stale: one O(emissive triangles) build once they stop, not one a
  // dial or drape frame, with the weights lagging meanwhile (the shader reads live geometry).
  bool emitterMoved = false, emitterWritten = false;
  for(const std::optional<Instance>& inst : m_instances)
    if(inst && emits(*inst))
    {
      emitterMoved |= changes.transforms.count(inst->entity) > 0;
      emitterWritten |= written.count(inst->mesh) > 0;
    }
  if(emitterWritten)
  {
    m_emissiveStale     = true;
    m_emitterWriteFrame = m_gpu.frame();
  }
  if(fresh || materialsWritten || result.visibilityChanged || emitterMoved || (emissiveSettled && !emitterWritten))
  {
    fv::timing::Accum time("emissive rebuild");
    syncEmissive(scene);
    m_emissiveStale = false;
    result.changed  = true;
  }
  if(m_descDirty || result.buffersChanged || !m_sceneDesc)
  {
    writeSceneDesc();
    result.changed = true;
  }
  // everything uploaded above, before the frame's shading reads it
  if(result.changed)
    m_gpu.barrier(GpuBarrier::eTransferToShader);
  // what this sync let go of and didn't take again
  trimGrace(m_options.textureGraceBytes);
  m_tablesMoved = m_bitsChanged = m_descDirty = false;
  result.featuresChanged                      = changes.features;
  return result;
}

// Instances for the draw items that are new, holes for the ones that went, and the prims, meshes
// and material slots that follow them.
void Residency::syncStructure(RetainedScene& scene, std::vector<DrawItem>& items, SyncResult& result)
{
  std::set<InstanceKey> drawn;
  for(const DrawItem& item : items)
    drawn.insert({item.entity, item.mesh, item.submesh, item.copy});

  // gone first, so a prim or a material an instance moves to is not retired under it
  for(auto it = m_slotOf.begin(); it != m_slotOf.end();)
  {
    if(drawn.count(it->first))
    {
      ++it;
      continue;
    }
    const uint32_t slot = it->second;
    Instance&      inst = *m_instances[slot];
    releasePrim(inst.prim);
    releaseMaterial(inst.material);
    m_instances[slot].reset();
    m_gpu.retire([pool = m_instancePool, slot] { pool->destroyID(slot); });
    m_dirtyNodes.insert(slot);  // a hole: its row zeroed; the slot goes back to the pool at its retire frame
    it                     = m_slotOf.erase(it);
    result.drawListChanged = true;
  }

  for(const DrawItem& item : items)
  {
    const InstanceKey key{item.entity, item.mesh, item.submesh, item.copy};
    auto              found = m_slotOf.find(key);
    if(found != m_slotOf.end())
    {
      // its material slot follows a new material binding
      Instance& inst = *m_instances[found->second];
      if(inst.material != item.material)
      {
        const uint32_t materialSlot = acquireMaterial(scene, item.material);
        releaseMaterial(inst.material);
        inst.material     = item.material;
        inst.materialSlot = materialSlot;
        m_dirtyNodes.insert(found->second);
      }
      continue;
    }
    uint32_t   slot = 0;
    const bool ok   = m_instancePool->createID(slot);
    assert(ok && "out of instance slots");
    (void)ok;
    if(slot >= m_instances.size())
      m_instances.resize(size_t(slot) + 1);
    Instance inst;
    inst.entity       = item.entity;
    inst.mesh         = item.mesh;
    inst.submesh      = item.submesh;
    inst.copy         = item.copy;
    inst.prim         = acquirePrim(scene, item.mesh, item.submesh);
    inst.material     = item.material;
    inst.materialSlot = acquireMaterial(scene, item.material);
    inst.world        = item.world;
    inst.shown        = item.shown;
    m_instances[slot] = inst;
    m_slotOf[key]     = slot;
    m_dirtyNodes.insert(slot);
    result.drawListChanged = true;
  }
  if(result.drawListChanged)
    result.changed = true;
}

// Every mesh whose vertices were edited, uploaded again; its BLASes refit, or rebuilt when a
// stream's buffer had to be replaced (the build data holds its address). Returns the meshes written.
std::set<MeshId> Residency::syncVertices(RetainedScene& scene, const std::set<MeshId>& meshes, SyncResult& result)
{
  std::set<MeshId>      written;
  std::vector<uint32_t> refits;  // after every upload, behind one barrier
  for(MeshId id : meshes)
  {
    auto              it     = m_meshes.find(id);
    const MeshRecord* record = scene.mesh(id);
    if(it == m_meshes.end() || !record || it->second.contentRev == record->contentRev)
      continue;
    MeshGpu&           gpu    = it->second;
    const BufferHandle before = gpu.positions;
    uploadStreams(*record, gpu);
    const bool moved = gpu.positions != before;
    for(int32_t p : gpu.prims)
    {
      if(p < 0)
        continue;
      Prim& prim               = *m_prims[size_t(p)];
      prim.input.vertexAddress = m_gpu.bufferAddress(gpu.positions);
      prim.input.vertexCount   = gpu.vertexCount;
      fitBounds(prim, *record);
      if(moved || !prim.blas)
      {
        retireAccel(prim.blas);
        prim.blas = {};  // built with the others in syncRows
      }
      else
        refits.push_back(uint32_t(p));
    }
    if(moved)
      result.buffersChanged = true;
    result.verticesWritten = true;
    result.changed         = true;
    written.insert(id);
  }
  if(refits.empty())
    return written;
  m_gpu.barrier(GpuBarrier::eTransferToBuild);
  for(uint32_t p : refits)
    m_gpu.refitBlas(m_prims[p]->blas, m_prims[p]->input);
  m_gpu.barrier(GpuBarrier::eBuildToBuild);  // the TLAS build reads them
  return written;
}

void Residency::ensureCapacity(SyncResult& result)
{
  const uint32_t instances = capacityFor(uint32_t(m_instances.size()), m_options.minInstances);
  if(instances != m_instanceCapacity)
  {
    retireBuffer(m_renderNodes);
    m_renderNodes = m_gpu.createBuffer(
        {.size = uint64_t(instances) * sizeof(shaderio::GltfRenderNode), .usage = eBufferStorage, .name = "render nodes"});
    m_instanceCapacity    = instances;
    m_allNodes            = true;  // a new buffer holds nothing, holes included
    result.buffersChanged = result.changed = true;
  }
  if(instances != m_tlasCapacity)
  {
    retireAccel(m_tlas);
    m_tlas                   = m_gpu.createTlas(instances);
    m_tlasCapacity           = instances;
    result.tlasHandleChanged = result.changed = true;
  }
  const uint32_t prims = capacityFor(uint32_t(m_prims.size()), m_options.minPrims);
  if(prims != m_primCapacity)
  {
    retireBuffer(m_renderPrims);
    m_renderPrims = m_gpu.createBuffer(
        {.size = uint64_t(prims) * sizeof(shaderio::GltfRenderPrimitive), .usage = eBufferStorage, .name = "render prims"});
    m_primCapacity        = prims;
    m_allPrims            = true;
    result.buffersChanged = result.changed = true;
  }
}

// The rows that differ from what is drawn, then the BLASes still to build, then the TLAS.
void Residency::syncRows(const std::vector<DrawItem>& items, SyncResult& result)
{
  std::set<uint32_t> nodes;
  nodes.swap(m_dirtyNodes);
  if(m_allNodes)
    for(uint32_t s = 0; s < m_instances.size(); s++)
      nodes.insert(s);
  m_allNodes    = false;
  bool tlasRows = !nodes.empty();
  for(const DrawItem& item : items)
  {
    const uint32_t slot = m_slotOf.at({item.entity, item.mesh, item.submesh, item.copy});
    Instance&      inst = *m_instances[slot];
    if(inst.shown != item.shown)
    {
      inst.shown               = item.shown;
      result.visibilityChanged = tlasRows = true;  // the TLAS row, not the render node
    }
    if(inst.world != item.world)
    {
      inst.world = item.world;
      nodes.insert(slot);
      tlasRows = true;
    }
  }

  // prims: new ones and the ones to build again, or every row into a new table
  std::set<uint32_t>    prims;
  std::vector<uint32_t> builds;
  for(uint32_t p = 0; p < m_prims.size(); p++)
    if(m_prims[p] && !m_prims[p]->blas)
    {
      builds.push_back(p);
      prims.insert(p);
    }
  if(m_allPrims)
    for(uint32_t p = 0; p < m_prims.size(); p++)
      prims.insert(p);
  m_allPrims = false;

  if(!nodes.empty() || !prims.empty())
    m_gpu.barrier(GpuBarrier::eTransferToBuild);
  writeRenderNodes(nodes);
  writeRenderPrims(prims);
  for(uint32_t p : builds)
  {
    Prim& prim = *m_prims[p];
    prim.blas  = m_gpu.buildBlas(prim.input, eAccelPreferFastTrace | eAccelAllowUpdate);
  }
  if(!builds.empty())
    m_gpu.barrier(GpuBarrier::eBuildToBuild);

  if(tlasRows || !builds.empty() || result.drawListChanged || result.tlasHandleChanged || result.verticesWritten || m_bitsChanged)
  {
    buildTlas();
    m_gpu.barrier(GpuBarrier::eBuildToTrace);
    result.changed = true;
  }
  if(!nodes.empty() || !prims.empty())
    result.changed = true;
}

void Residency::writeRenderNodes(const std::set<uint32_t>& slots)
{
  for(const auto& [first, count] : runsOf(slots))
  {
    std::vector<shaderio::GltfRenderNode> rows(count);
    for(uint32_t i = 0; i < count; i++)
    {
      const std::optional<Instance>& inst = first + i < m_instances.size() ? m_instances[first + i] : std::nullopt;
      shaderio::GltfRenderNode&      row  = rows[i];
      row                                 = {};
      if(!inst)
        continue;  // a hole: zeroed
      row.objectToWorld = inst->world;
      row.worldToObject = glm::inverse(inst->world);
      row.materialID    = int(inst->materialSlot);
      row.renderPrimID  = int(inst->prim);
    }
    m_gpu.upload(m_renderNodes, uint64_t(first) * sizeof(shaderio::GltfRenderNode), bytesOf(rows));
  }
}

void Residency::writeRenderPrims(const std::set<uint32_t>& slots)
{
  for(const auto& [first, count] : runsOf(slots))
  {
    std::vector<shaderio::GltfRenderPrimitive> rows(count);
    for(uint32_t i = 0; i < count; i++)
    {
      shaderio::GltfRenderPrimitive& row = rows[i];
      row                                = {};
      const std::optional<Prim>& prim    = m_prims[first + i];
      if(!prim)
        continue;
      const MeshGpu& mesh        = m_meshes.at(prim->mesh);
      row.indices                = (glm::uvec3*)m_gpu.bufferAddress(prim->indices);
      row.vertexBuffer.positions = (glm::vec3*)m_gpu.bufferAddress(mesh.positions);
      row.vertexBuffer.normals   = (glm::vec3*)m_gpu.bufferAddress(mesh.normals);
      row.vertexBuffer.tangents  = (glm::vec4*)m_gpu.bufferAddress(mesh.tangents);
      row.vertexBuffer.texCoord0 = (glm::vec2*)m_gpu.bufferAddress(mesh.uv0);
      row.vertexBuffer.texCoord1 = (glm::vec2*)m_gpu.bufferAddress(mesh.uv1);
      row.vertexBuffer.colors    = (glm::uint*)m_gpu.bufferAddress(mesh.colors);
    }
    m_gpu.upload(m_renderPrims, uint64_t(first) * sizeof(shaderio::GltfRenderPrimitive), bytesOf(rows));
  }
}

// Rows up to the high-water mark: a live instance's, a hidden one's without its BLAS, a hole's null
void Residency::buildTlas()
{
  std::vector<TlasRow> rows(m_instances.size());
  for(size_t s = 0; s < m_instances.size(); s++)
  {
    const std::optional<Instance>& inst = m_instances[s];
    if(!inst)
      continue;
    TlasRow& row                       = rows[s];
    row                                = tlasRowOf(inst->world);
    row.customIndex                    = inst->prim;
    const nvvkgltf::MaterialBits& bits = m_materialStates[inst->materialSlot]->bits;
    row.mask                           = bits.instanceMask;
    row.flags                          = bits.instanceFlags;
    if(inst->shown)
      row.blas = m_prims[inst->prim]->blas;
  }
  m_gpu.buildTlas(m_tlas, rows, TlasMode::eBuild);
}

void Residency::writeSceneDesc()
{
  if(!m_sceneDesc)
    m_sceneDesc = m_gpu.createBuffer({.size = sizeof(shaderio::GltfScene), .usage = eBufferStorage, .name = "scene"});
  shaderio::GltfScene desc{};
  desc.renderNodes           = (shaderio::GltfRenderNode*)m_gpu.bufferAddress(m_renderNodes);
  desc.renderPrimitives      = (shaderio::GltfRenderPrimitive*)m_gpu.bufferAddress(m_renderPrims);
  desc.materials             = (shaderio::GltfShadeMaterial*)m_gpu.bufferAddress(m_materialTable);
  desc.textureInfos          = (shaderio::GltfTextureInfo*)m_gpu.bufferAddress(m_textureInfoTable);
  desc.lights                = (shaderio::GltfLight*)m_gpu.bufferAddress(m_lightTable);
  desc.numLights             = int(m_lightCount);
  desc.iesProfiles           = (float*)m_gpu.bufferAddress(m_iesTable);
  desc.iesProfileSampleCount = nvvkgltf::kIesProfileSampleCount;
  desc.emissiveTriangles     = (shaderio::EmissiveTriangle*)m_gpu.bufferAddress(m_emissiveTable);
  desc.numEmissiveTriangles  = int(m_emissiveCount);
  desc.emissiveTotalWeight   = m_emissiveTotalWeight;
  desc.emissiveMeanLum       = m_emissiveMeanLum;
  m_gpu.upload(m_sceneDesc, 0, bytesOf(desc));
}

// ---------- Lights and emitters ----------

// A buffer holding `bytes`, made again only when it is too small (none for nothing)
static bool fitBuffer(GpuBackend& gpu, BufferHandle& buffer, uint64_t bytes, const char* name, const std::function<void(BufferHandle)>& retire)
{
  if(buffer && gpu.bufferSize(buffer) >= bytes && bytes)
    return false;
  if(buffer)
    retire(buffer);
  buffer = bytes ? gpu.createBuffer({.size = std::bit_ceil(bytes), .usage = eBufferStorage, .name = name}) : BufferHandle{};
  return true;
}

// Every light in document order; a hidden one (its entity or the light itself) shines with nothing
void Residency::syncLights(RetainedScene& scene)
{
  std::vector<shaderio::GltfLight> lights;
  for(const LightItem& item : scene.lightItems())
  {
    const LightRecord*  record = scene.light(item.light);
    shaderio::GltfLight light =
        record && record->ies ?
            nvvkgltf::packIesLight(item.world, int(record->ies->profile), record->ies->multiplier, record->ies->color) :
            nvvkgltf::packLight(record ? &record->payload : nullptr, item.world);
    if(!item.shown)
      light.intensity = 0.f;
    lights.push_back(light);
  }
  const uint64_t bytes = std::span(lights).size_bytes();
  m_descDirty |= fitBuffer(m_gpu, m_lightTable, bytes, "lights", [this](BufferHandle b) { retireBuffer(b); });
  if(bytes)
    m_gpu.upload(m_lightTable, 0, std::as_bytes(std::span(lights)));
  m_descDirty |= m_lightCount != lights.size();
  m_lightCount = uint32_t(lights.size());
}

// A shown instance whose material emits
bool Residency::emits(const Instance& inst) const
{
  if(!inst.shown || inst.materialSlot >= m_shadeMaterials.size())
    return false;
  const shaderio::GltfShadeMaterial& shade = m_shadeMaterials[inst.materialSlot];
  return nvvkgltf::emissiveLuminance(glm::max(shade.emissiveFactor, glm::vec3(0.0f)) * shade.emissiveLightScale) > 0.f;
}

// The profiles' samples end to end, which a light's iesProfile indexes by kIesProfileSampleCount
void Residency::syncIesProfiles(const RetainedScene& scene)
{
  std::vector<float> table;
  for(const IesProfileRecord& profile : scene.iesProfiles())
  {
    table.insert(table.end(), profile.samples.begin(), profile.samples.end());
    table.resize(table.size() + size_t(nvvkgltf::kIesProfileSampleCount) - profile.samples.size(), 1.f);
  }
  const uint64_t bytes = std::span(table).size_bytes();
  m_descDirty |= fitBuffer(m_gpu, m_iesTable, bytes, "IES profiles", [this](BufferHandle b) { retireBuffer(b); });
  if(bytes)
    m_gpu.upload(m_iesTable, 0, std::as_bytes(std::span(table)));
  m_descDirty |= m_iesProfileCount != scene.iesProfiles().size();
  m_iesProfileCount = uint32_t(scene.iesProfiles().size());
}

// The shown instances whose material emits, their triangles from their mesh records
void Residency::syncEmissive(const RetainedScene& scene)
{
  std::vector<nvvkgltf::Emitter> emitters;
  for(uint32_t s = 0; s < m_instances.size(); s++)
  {
    const std::optional<Instance>& inst = m_instances[s];
    if(!inst || !inst->shown || inst->materialSlot >= m_shadeMaterials.size())
      continue;
    const shaderio::GltfShadeMaterial& shade = m_shadeMaterials[inst->materialSlot];
    const glm::vec3   radiance = glm::max(shade.emissiveFactor, glm::vec3(0.0f)) * shade.emissiveLightScale;
    const MeshRecord* record   = scene.mesh(inst->mesh);
    if(nvvkgltf::emissiveLuminance(radiance) <= 0.f || !record)
      continue;
    const Submesh& run = record->submeshes[inst->submesh];
    emitters.push_back({.renderNodeID = s,
                        .renderPrimID = inst->prim,
                        .radiance     = radiance,
                        .doubleSided  = m_materialStates[inst->materialSlot]->payload.doubleSided,
                        .world        = inst->world,
                        .positions    = record->positions,
                        .indices      = std::span(record->indices).subspan(run.firstIndex, run.indexCount)});
  }
  const nvvkgltf::EmissiveTable table = nvvkgltf::buildEmissiveTable(emitters);
  const uint64_t                bytes = std::span(table.triangles).size_bytes();
  m_descDirty |=
      fitBuffer(m_gpu, m_emissiveTable, bytes, "emissive triangles", [this](BufferHandle b) { retireBuffer(b); });
  if(bytes)
    m_gpu.upload(m_emissiveTable, 0, std::as_bytes(std::span(table.triangles)));
  m_descDirty |= m_emissiveCount != table.triangles.size() || m_emissiveTotalWeight != table.totalWeight
                 || m_emissiveMeanLum != table.meanLum;
  m_emissiveCount       = uint32_t(table.triangles.size());
  m_emissiveTotalWeight = table.totalWeight;
  m_emissiveMeanLum     = table.meanLum;
}

// ---------- Reading ----------

uint64_t Residency::sceneDesc() const
{
  return m_gpu.bufferAddress(m_sceneDesc);
}

const Residency::Instance* Residency::instance(uint32_t slot) const
{
  return slot < m_instances.size() && m_instances[slot] ? &*m_instances[slot] : nullptr;
}

const Residency::Prim* Residency::prim(uint32_t slot) const
{
  return slot < m_prims.size() && m_prims[slot] ? &*m_prims[slot] : nullptr;
}

const Residency::MeshGpu* Residency::meshGpu(MeshId mesh) const
{
  auto it = m_meshes.find(mesh);
  return it == m_meshes.end() ? nullptr : &it->second;
}

int32_t Residency::textureSlot(TextureId texture, const GpuSamplerDesc& sampler) const
{
  auto it = m_textureSlots.find({texture, keyOf(sampler)});
  return it == m_textureSlots.end() ? -1 : int32_t(it->second.slot);
}

ImageHandle Residency::image(TextureId texture) const
{
  auto it = m_images.find(texture);
  return it == m_images.end() ? ImageHandle{} : it->second.image;
}

int32_t Residency::samplerSlot(const GpuSamplerDesc& sampler) const
{
  auto it = m_samplerSlots.find(keyOf(sampler));
  return it == m_samplerSlots.end() ? -1 : int32_t(it->second.slot);
}

const shaderio::GltfShadeMaterial* Residency::shadeMaterial(uint32_t slot) const
{
  return slot < m_shadeMaterials.size() ? &m_shadeMaterials[slot] : nullptr;
}

const Residency::MaterialState* Residency::materialState(uint32_t slot) const
{
  return slot < m_materialStates.size() && m_materialStates[slot] ? &*m_materialStates[slot] : nullptr;
}

int32_t Residency::materialSlot(MaterialId material) const
{
  auto it = m_materials.find(material);
  return it == m_materials.end() ? -1 : int32_t(it->second.slot);
}

size_t Residency::liveBuffers() const
{
  size_t n = 0;
  for(const auto& [id, mesh] : m_meshes)
    for(BufferHandle b : {mesh.positions, mesh.normals, mesh.tangents, mesh.uv0, mesh.uv1, mesh.colors})
      n += b ? 1 : 0;
  for(const std::optional<Prim>& p : m_prims)
    n += p && p->indices ? 1 : 0;
  for(BufferHandle b : {m_renderNodes, m_renderPrims, m_sceneDesc, m_materialTable, m_textureInfoTable, m_lightTable,
                        m_emissiveTable, m_iesTable})
    n += b ? 1 : 0;
  return n;
}

size_t Residency::liveImages() const
{
  size_t n = m_decoded.size();
  for(const auto& [id, image] : m_images)
    n += !image.key && image.image ? 1 : 0;
  return n;
}

size_t Residency::liveAccels() const
{
  size_t n = m_tlas ? 1 : 0;
  for(const std::optional<Prim>& p : m_prims)
    n += p && p->blas ? 1 : 0;
  return n;
}

}  // namespace rs
