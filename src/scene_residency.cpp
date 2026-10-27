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
#include <limits>
#include <span>

#include <glm/gtc/packing.hpp>

#include <nvutils/id_pool.hpp>

#include "shaders/gltf_scene_io.h.slang"

namespace rs {

namespace {

// Slot pools: a TLAS row's custom index is 24 bits, so neither instance nor prim slots go past it
constexpr uint32_t kSlotPoolSize     = 1u << 24;
constexpr uint32_t kMaterialPoolSize = 1u << 16;

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
{
  clear();
}

Residency::~Residency()
{
  clear();
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
  retireBuffer(m_renderNodes);
  retireBuffer(m_renderPrims);
  retireBuffer(m_sceneDesc);
  retireAccel(m_tlas);
  m_meshes.clear();
  m_prims.clear();
  m_instances.clear();
  m_slotOf.clear();
  m_materials.clear();
  m_renderNodes = m_renderPrims = m_sceneDesc = {};
  m_tlas                                      = {};
  m_instanceCapacity = m_primCapacity = m_tlasCapacity = 0;
  // fresh pools: what the old ones handed out is retired with its resources, and never reused
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

uint32_t Residency::acquireMaterial(MaterialId id)
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
  m_materials[id] = {slot, 1};
  return slot;
}

void Residency::releaseMaterial(MaterialId id)
{
  auto it = m_materials.find(id);
  if(it == m_materials.end() || --it->second.users > 0)
    return;
  const uint32_t slot = it->second.slot;
  m_materials.erase(it);
  m_gpu.retire([pool = m_materialPool, slot] { pool->destroyID(slot); });
}

// ---------- The sync ----------

SyncResult Residency::replace(RetainedScene& scene)
{
  clear();
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
  if(!fresh && changes.empty())
    return result;

  std::vector<DrawItem> items = scene.drawItems();
  if(fresh || changes.structure)
    syncStructure(scene, items, result);
  if(!changes.meshes.empty())
    syncVertices(scene, changes.meshes, result);
  ensureCapacity(result);
  syncRows(items, result);
  result.featuresChanged = changes.features;
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
        const uint32_t materialSlot = acquireMaterial(item.material);
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
    inst.materialSlot = acquireMaterial(item.material);
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
// stream's buffer had to be replaced (the build data holds its address).
void Residency::syncVertices(RetainedScene& scene, const std::set<MeshId>& meshes, SyncResult& result)
{
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
      if(moved || !prim.blas)
      {
        retireAccel(prim.blas);
        prim.blas = {};  // built with the others in syncRows
      }
      else
        m_gpu.refitBlas(prim.blas, prim.input);
    }
    if(moved)
      result.buffersChanged = true;
    result.verticesWritten = true;
    result.changed         = true;
  }
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

  if(tlasRows || !builds.empty() || result.drawListChanged || result.tlasHandleChanged || result.verticesWritten)
  {
    buildTlas();
    m_gpu.barrier(GpuBarrier::eBuildToTrace);
    result.changed = true;
  }
  if(result.buffersChanged || !m_sceneDesc)
    writeSceneDesc();
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
    TlasRow& row    = rows[s];
    row             = tlasRowOf(inst->world);
    row.customIndex = inst->prim;
    row.mask        = 0xFF;
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
  desc.renderNodes      = (shaderio::GltfRenderNode*)m_gpu.bufferAddress(m_renderNodes);
  desc.renderPrimitives = (shaderio::GltfRenderPrimitive*)m_gpu.bufferAddress(m_renderPrims);
  m_gpu.upload(m_sceneDesc, 0, bytesOf(desc));
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
  for(BufferHandle b : {m_renderNodes, m_renderPrims, m_sceneDesc})
    n += b ? 1 : 0;
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
