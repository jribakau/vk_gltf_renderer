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
#include <nvutils/logger.hpp>

#include "ies_profile.hpp"
#include "scene_gpu_tables.hpp"
#include "scene_skin.hpp"
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
  m_noticedLowered = m_noticedRefused = false;
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
  {
    for(BufferHandle b : {mesh.positions, mesh.normals, mesh.tangents, mesh.uv0, mesh.uv1, mesh.colors, mesh.joints, mesh.cageJoints, mesh.cageWeights, mesh.cageJoints1, mesh.cageWeights1, mesh.cageShares, mesh.joints1,
                          mesh.weights1, mesh.dualQuatShares, mesh.weights, mesh.positionDeltas, mesh.normalDeltas,
                          mesh.tangentDeltas, mesh.prevPositions})
      retireBuffer(b);
    retireRefine(mesh);
  }
  for(auto& [key, d] : m_drawers)
  {
    for(BufferHandle b : {d.positions, d.normals, d.tangents, d.morphedPositions, d.morphedNormals, d.morphedTangents, d.prevPositions})
      retireBuffer(b);
    for(BufferHandle b : d.posedBuffers())
      retireBuffer(b);
  }
  m_drawers.clear();
  for(BufferHandle b : {m_jointPack, m_normalPack, m_weightPack, m_dualQuatPack, m_prevPositionTable, m_prevWorldTable})
    retireBuffer(b);
  m_jointPack = m_normalPack = m_weightPack = m_dualQuatPack = m_prevPositionTable = m_prevWorldTable = {};
  m_packsDirty = false;
  m_drawersWritten.clear();
  m_compacting.clear();
  m_refineAcquired.clear();
  m_meshesWritten.clear();
  m_prevWorlds.clear();
  m_dirtyPrevPrims.clear();
  m_dirtyPrevNodes.clear();
  m_movedLast.clear();
  for(const std::optional<Prim>& p : m_prims)
    if(p)
    {
      retireBuffer(p->indices);
      retireBuffer(p->micromapIndices);
      retireAccel(p->blas);
    }
  for(const auto& [key, mm] : m_micromaps)  // after the BLASes that link them
    retireAccel(mm.accel);
  m_micromaps.clear();
  m_micromapPrims = 0;
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
  retireBuffer(m_emissiveGroupTable);
  retireBuffer(m_emissiveLeafTable);
  retireBuffer(m_iesTable);
  m_lightTable = m_emissiveTable = m_emissiveGroupTable = m_emissiveLeafTable = m_iesTable = {};
  m_lightCount = m_emissiveCount = m_emissiveGroupCount = m_emissiveNodeCount = m_emissiveLeafCount = m_iesProfileCount = 0;
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
  m_dirtyMaterials.clear();
  m_heldMaterials.clear();
  m_textureInfoPool = poolFromOne(m_options.textureInfoLimit);
  m_textureInfos.assign(1, {});
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

void Residency::build(Prim& prim)
{
  prim.compactable = m_options.compactStatic && prim.drawer.isNone();
  prim.compacting  = prim.compactable;
  prim.blas = m_gpu.buildBlas(prim.input, eAccelPreferFastTrace | (prim.compactable ? eAccelAllowCompaction : eAccelAllowUpdate));
  prim.refits = 0;
}

void Residency::refit(Prim& prim)
{
  if(!prim.compactable && (m_options.refitsBeforeRebuild == 0 || ++prim.refits < m_options.refitsBeforeRebuild))
  {
    m_gpu.refitBlas(prim.blas, prim.input);
    return;
  }
  // refitted enough, or built for compaction: its vertices move after all
  retireAccel(prim.blas);
  prim.blas        = m_gpu.buildBlas(prim.input, eAccelPreferFastTrace | eAccelAllowUpdate);
  prim.refits      = 0;
  prim.compactable = prim.compacting = false;
}

bool Residency::takeCompacted()
{
  bool any = false;
  for(auto it = m_compacting.begin(); it != m_compacting.end();)
  {
    const uint32_t p = *it;
    if(p >= m_prims.size() || !m_prims[p] || !m_prims[p]->compacting)
    {
      it = m_compacting.erase(it);
      continue;
    }
    Prim&             prim = *m_prims[p];
    const AccelHandle copy = m_gpu.compactBlas(prim.blas);
    if(!copy)
    {
      ++it;  // its size isn't known yet
      continue;
    }
    prim.compacting = false;
    it              = m_compacting.erase(it);
    if(copy == prim.blas)
      continue;  // kept as built
    retireAccel(prim.blas);
    prim.blas = copy;
    any       = true;
  }
  return any;
}

// ---------- Records ----------

void Residency::makeRefine(const MeshRecord& record, MeshGpu& gpu)
{
  const RefineTables* t = record.refine.get();
  if(!t || t->levels.empty() || record.cage.size() != t->cageCount || t->drawnFrom.size() != record.positions.size()
     || t->normalStart.size() != size_t(t->refinedCount()) + 1 || (!t->offsets.empty() && t->offsets.size() != t->refinedCount()))
    return;
  const auto table = [&](const auto& values, const char* name) {
    const BufferHandle b = m_gpu.createBuffer({.size = values.size() * sizeof(values[0]), .usage = eBufferStorage, .name = name});
    m_gpu.upload(b, 0, bytesOf(values));
    return b;
  };
  MeshGpu::Refine r;
  r.cage = m_gpu.createBuffer({.size = record.cage.size() * sizeof(glm::vec3), .usage = eBufferStorage, .name = "cage"});
  m_gpu.upload(r.cage, 0, bytesOf(record.cage));
  uint32_t source = t->cageCount;
  for(size_t i = 0; i < t->levels.size(); i++)
  {
    const RefineTables::Level& l = t->levels[i];
    MeshGpu::Refine::Level     g{.rows = l.rows(), .sourceCount = source};
    g.start   = table(l.start, "stencil starts");
    g.columns = table(l.columns, "stencil columns");
    g.weights = table(l.weights, "stencil weights");
    g.out = m_gpu.createBuffer({.size = uint64_t(g.rows) * sizeof(glm::vec3), .usage = eBufferStorage, .name = "refined level"});
    r.levels.push_back(g);
    source = g.rows;
  }
  r.refinedCount   = t->refinedCount();
  r.triangleCount  = uint32_t(t->triangles.size() / 3);
  r.refinedNormals = m_gpu.createBuffer(
      {.size = uint64_t(r.refinedCount) * sizeof(glm::vec3), .usage = eBufferStorage, .name = "refined normals"});
  r.drawnFrom       = table(t->drawnFrom, "drawn from");
  r.triangles       = table(t->triangles, "refined triangles");
  r.normalStart     = table(t->normalStart, "normal starts");
  r.normalTriangles = table(t->normalTriangles, "normal triangles");
  if(!t->offsets.empty())
    r.offsets = table(t->offsets, "refined offsets");
  // the tangents too, over its own corners, when it has them and the UVs they read
  if(!t->tangentIndices.empty() && gpu.tangents && t->tangentStart.size() == record.positions.size() + 1
     && t->tangentCorners.size() == t->tangentIndices.size() / 3 * 3)
  {
    r.tangentTriangles     = table(t->tangentIndices, "tangent triangles");
    r.tangentStart         = table(t->tangentStart, "tangent corner starts");
    r.tangentCorners       = table(t->tangentCorners, "tangent corners");
    r.tangentTriangleCount = uint32_t(t->tangentIndices.size() / 3);
  }
  gpu.refine        = std::move(r);
}

void Residency::refineOnGpu(std::span<MeshGpu* const> meshes)
{
  if(meshes.empty())
    return;
  std::vector<RefineRun> runs;
  for(const MeshGpu* g : meshes)
  {
    RefineRun& run     = runs.emplace_back();
    run.mesh           = g;
    run.cage           = g->refine->cage;
    run.refinedNormals = g->refine->refinedNormals;
    for(const MeshGpu::Refine::Level& l : g->refine->levels)
      run.levels.push_back(l.out);
    run.positions = g->positions;
    run.normals   = g->normals;
    run.tangents  = g->tangents;
  }
  m_gpu.barrier(GpuBarrier::eTransferToShader);  // the cages
  refineChains(runs);
  // the refits read the positions, the skin passes both, this frame's draws both
  m_gpu.barrier(GpuBarrier::eComputeToBuild);
}

void Residency::refineChains(std::span<const RefineRun> runs)
{
  size_t levels = 0;
  for(const RefineRun& r : runs)
    levels = std::max(levels, r.mesh->refine->levels.size());
  for(size_t i = 0; i < levels; i++)
  {
    for(const RefineRun& r : runs)
      if(const MeshGpu::Refine& t = *r.mesh->refine; i < t.levels.size())
      {
        const MeshGpu::Refine::Level& l    = t.levels[i];
        const bool                    last = i + 1 == t.levels.size();
        m_gpu.dispatchDeform({.kind           = DeformKind::eStencil,
                              .vertexCount    = l.rows,
                              .basePositions  = {i ? r.levels[i - 1] : r.cage},
                              .outPositions   = {r.levels[i]},
                              .rowStart       = {l.start},
                              .columns        = {l.columns},
                              .stencilWeights = {l.weights},
                              .stencilOffsets = {last ? t.offsets : BufferHandle{}},
                              .sourceCount    = l.sourceCount});
      }
    m_gpu.barrier(GpuBarrier::eComputeToCompute);
  }
  for(const RefineRun& r : runs)
  {
    const MeshGpu::Refine& t = *r.mesh->refine;
    m_gpu.dispatchDeform({.kind          = DeformKind::eNormals,
                          .vertexCount   = t.refinedCount,
                          .basePositions = {r.levels.back()},
                          .outNormals    = {r.refinedNormals},
                          .rowStart      = {t.normalStart},
                          .columns       = {t.normalTriangles},
                          .triangles     = {t.triangles},
                          .sourceCount   = t.refinedCount,
                          .triangleCount = t.triangleCount});
  }
  m_gpu.barrier(GpuBarrier::eComputeToCompute);
  // the splits: a gather of positions and of normals, each drawn vertex its refined vertex's
  for(const RefineRun& r : runs)
    for(const auto& [from, to] : {std::pair{r.levels.back(), r.positions}, std::pair{r.refinedNormals, r.normals}})
      if(to)
        m_gpu.dispatchDeform({.kind          = DeformKind::eStencil,
                              .vertexCount   = r.mesh->vertexCount,
                              .basePositions = {from},
                              .outPositions  = {to},
                              .columns       = {r.mesh->refine->drawnFrom},
                              .sourceCount   = r.mesh->refine->refinedCount});
  // the tangents, from the drawn positions, normals and UVs
  if(std::any_of(runs.begin(), runs.end(), [](const RefineRun& r) { return r.mesh->refine->tangentTriangles && r.tangents; }))
  {
    m_gpu.barrier(GpuBarrier::eComputeToCompute);
    for(const RefineRun& r : runs)
      if(const MeshGpu::Refine& t = *r.mesh->refine; t.tangentTriangles && r.normals && r.tangents)
        m_gpu.dispatchDeform({.kind          = DeformKind::eTangents,
                              .vertexCount   = r.mesh->vertexCount,
                              .basePositions = {r.positions},
                              .baseNormals   = {r.normals},
                              .outTangents   = {r.tangents},
                              .rowStart      = {t.tangentStart},
                              .columns       = {t.tangentCorners},
                              .triangles     = {t.tangentTriangles},
                              .uvs           = {r.mesh->uv0},
                              .triangleCount = t.tangentTriangleCount});
  }
}

void Residency::uploadStreams(const MeshRecord& record, MeshGpu& gpu, bool vertices)
{
  const auto stream = [&](BufferHandle& buffer, std::span<const std::byte> bytes, BufferUsageFlags usage, const char* name) {
    if(bytes.empty())
      return;
    // a refined mesh's positions and normals (and tangents, with its tables) are made on the GPU from
    // its cage after the first upload
    if(vertices && gpu.refine
       && (&buffer == &gpu.positions || &buffer == &gpu.normals || (&buffer == &gpu.tangents && gpu.refine->tangentTriangles)))
      return;
    if(!buffer || m_gpu.bufferSize(buffer) != bytes.size())
    {
      retireBuffer(buffer);
      buffer = m_gpu.createBuffer({.size = bytes.size(), .usage = usage, .name = name});
    }
    m_gpu.upload(buffer, 0, bytes);
  };
  if(vertices && gpu.refine)
    m_gpu.upload(gpu.refine->cage, 0, bytesOf(record.cage));
  stream(gpu.positions, bytesOf(record.positions), eBufferVertex | eBufferStorage | eBufferAccelInput, "positions");
  stream(gpu.normals, bytesOf(record.normals), eBufferStorage, "normals");
  if(!vertices || gpu.tangentRev != record.tangentRev)
    stream(gpu.tangents, bytesOf(record.tangents), eBufferStorage, "tangents");
  gpu.contentRev = record.contentRev;
  gpu.tangentRev = record.tangentRev;
  if(vertices)  // a vertex edit keeps the count, so the UVs and colours are as they were
    return;
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
    makeRefine(*record, gpu);
    if(gpu.refine)
      m_refineAcquired.push_back(id);
    gpu.prims.assign(record->submeshes.size(), -1);
  }
  return gpu;
}

void Residency::retireMesh(MeshId id)
{
  auto it = m_meshes.find(id);
  if(it == m_meshes.end())
    return;
  MeshGpu& m = it->second;
  for(BufferHandle b : {m.positions, m.normals, m.tangents, m.uv0, m.uv1, m.colors, m.joints, m.cageJoints, m.cageWeights, m.cageJoints1, m.cageWeights1, m.cageShares, m.joints1, m.weights1,
                        m.dualQuatShares, m.weights, m.positionDeltas, m.normalDeltas, m.tangentDeltas, m.prevPositions})
    retireBuffer(b);
  retireRefine(m);
  m_meshes.erase(it);
}

void Residency::retireRefine(MeshGpu& m)
{
  if(!m.refine)
    return;
  for(BufferHandle b : {m.refine->cage, m.refine->refinedNormals, m.refine->drawnFrom, m.refine->triangles, m.refine->normalStart,
                        m.refine->normalTriangles, m.refine->offsets, m.refine->tangentTriangles, m.refine->tangentStart,
                        m.refine->tangentCorners})
    retireBuffer(b);
  for(const MeshGpu::Refine::Level& l : m.refine->levels)
    for(BufferHandle b : {l.start, l.columns, l.weights, l.out})
      retireBuffer(b);
  m.refine.reset();
}

// The cage vertices a refined submesh's vertices are made from: its refined vertices, then each
// level's rows back to the columns they read
static std::vector<uint32_t> cageSupport(const RefineTables& t, std::span<const uint32_t> indices)
{
  std::vector<uint8_t> mark(t.refinedCount(), 0);
  for(uint32_t i : indices)
    if(i < t.drawnFrom.size() && t.drawnFrom[i] < mark.size())
      mark[t.drawnFrom[i]] = 1;
  for(size_t l = t.levels.size(); l-- > 0;)
  {
    const RefineTables::Level& level = t.levels[l];
    std::vector<uint8_t>       below(l ? t.levels[l - 1].rows() : t.cageCount, 0);
    for(uint32_t r = 0; r < level.rows() && r < mark.size(); r++)
      if(mark[r])
        for(uint32_t k = level.start[r]; k < level.start[r + 1]; k++)
          if(level.columns[k] < below.size())
            below[level.columns[k]] = 1;
    mark = std::move(below);
  }
  std::vector<uint32_t> out;
  for(uint32_t v = 0; v < mark.size(); v++)
    if(mark[v])
      out.push_back(v);
  return out;
}

// The box around the vertices a prim's indices name; none named: a point at the origin. A refined
// mesh's is its cage's around the vertices they are made from (a refined vertex is a convex sum of
// them), widened by its offsets: the drawn positions the CPU keeps fall behind its morphs (they
// place the selection box and the scene's bounds, not culling)
static void fitBounds(Residency::Prim& prim, const MeshRecord& record)
{
  const Submesh run = record.drawnRun(prim.submesh);
  glm::vec3      lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
  if(record.refine && !record.cage.empty() && run.indexCount)
  {
    if(prim.supportOf.lock() != record.refine)
    {
      prim.cageSupport = cageSupport(*record.refine, std::span(record.indices).subspan(run.firstIndex, run.indexCount));
      prim.cagePad     = 0;
      for(const glm::vec3& o : record.refine->offsets)
        prim.cagePad = std::max(prim.cagePad, glm::length(o));
      prim.supportOf = record.refine;
    }
    for(uint32_t v : prim.cageSupport)
      if(v < record.cage.size())
      {
        lo = glm::min(lo, record.cage[v]);
        hi = glm::max(hi, record.cage[v]);
      }
    lo -= glm::vec3(prim.cagePad);
    hi += glm::vec3(prim.cagePad);
  }
  else
    for(uint32_t i = run.firstIndex; i < run.firstIndex + run.indexCount; i++)
      if(record.indices[i] < record.positions.size())
      {
        lo = glm::min(lo, record.positions[record.indices[i]]);
        hi = glm::max(hi, record.positions[record.indices[i]]);
      }
  prim.boundsMin = lo.x <= hi.x ? lo : glm::vec3(0.f);
  prim.boundsMax = lo.x <= hi.x ? hi : glm::vec3(0.f);
}

uint32_t Residency::acquirePrim(const RetainedScene& scene, MeshId id, uint32_t submesh, EntityId drawer)
{
  MeshGpu&              mesh  = acquireMesh(scene, id);
  DrawerGpu*            d     = drawer.isNone() ? nullptr : &m_drawers.at({drawer, id});
  std::vector<int32_t>& slots = d ? d->prims : mesh.prims;
  if(submesh < slots.size() && slots[submesh] >= 0)
  {
    m_prims[size_t(slots[submesh])]->users++;
    return uint32_t(slots[submesh]);
  }
  uint32_t   slot = 0;
  const bool ok   = m_primPool->createID(slot);
  assert(ok && "out of prim slots");
  (void)ok;
  if(slot >= m_prims.size())
    m_prims.resize(size_t(slot) + 1);

  const MeshRecord& record = *scene.mesh(id);
  const Submesh     run    = record.drawnRun(submesh);
  Prim              prim;
  prim.mesh       = id;
  prim.drawer     = drawer;
  prim.submesh    = submesh;
  prim.indexCount = run.indexCount;
  prim.users      = 1;
  const std::span<const uint32_t> indices(record.indices.data() + run.firstIndex, run.indexCount);
  prim.indices = m_gpu.createBuffer(
      {.size = indices.size_bytes(), .usage = eBufferIndex | eBufferStorage | eBufferAccelInput, .name = "indices"});
  m_gpu.upload(prim.indices, 0, std::as_bytes(indices));
  prim.input          = {.vertexAddress = m_gpu.bufferAddress(d ? d->positions : mesh.positions),
                         .vertexStride  = sizeof(glm::vec3),
                         .vertexCount   = mesh.vertexCount,
                         .indexAddress  = m_gpu.bufferAddress(prim.indices),
                         .indexCount    = run.indexCount,
                         .opaque        = false};
  if(run.micromap && m_gpu.opacityMicromaps())
    linkMicromap(prim, *run.micromap);
  else if(run.micromap && !std::exchange(m_micromapsLeftOut, true))
    // as createVulkanScene says it: micromaps only accelerate, so without device support they are left out
    m_notices.push_back(
        "the scene has opacity micromaps; using them needs an NVIDIA RTX GPU with a recent driver "
        "(VK_KHR_opacity_micromap). Rendering is unaffected.");
  fitBounds(prim, record);
  m_prims[slot]  = std::move(prim);
  slots[submesh] = int32_t(slot);
  m_dirtyPrevPrims.insert(slot);  // its motion row: none until its positions have a previous copy
  return slot;
}

void Residency::releasePrim(uint32_t slot)
{
  Prim& prim = *m_prims[slot];
  if(--prim.users > 0)
    return;
  retireBuffer(prim.indices);
  retireAccel(prim.blas);
  unlinkMicromap(prim);  // retired after the BLAS
  if(!prim.drawer.isNone())
  {
    // the drawer goes with its last instance (releaseDrawer), and the mesh with its last drawer
    if(auto d = m_drawers.find({prim.drawer, prim.mesh}); d != m_drawers.end())
      d->second.prims[prim.submesh] = -1;
  }
  else if(auto mesh = m_meshes.find(prim.mesh); mesh != m_meshes.end())
  {
    mesh->second.prims[prim.submesh] = -1;
    if(!meshInUse(prim.mesh))
      retireMesh(prim.mesh);
  }
  m_prims[slot].reset();
  m_dirtyPrevPrims.insert(slot);
  m_gpu.retire([pool = m_primPool, slot] { pool->destroyID(slot); });
}

// The micromap a prim's triangles take their opacity from, built for the first prim that links it,
// and the prim's micromap indices uploaded (the BLAS builds read them behind syncRows' barrier)
void Residency::linkMicromap(Prim& prim, const MicromapUse& use)
{
  MicromapGpu& mm = m_micromaps[use.micromap.get()];
  if(!mm.accel)
  {
    std::vector<MicromapUsage> usages;
    for(const OpacityMicromap::Usage& u : use.micromap->usages)
      usages.push_back({.count = u.count, .subdivisionLevel = u.subdivisionLevel, .format = u.format});
    mm.record = use.micromap;
    mm.accel  = m_gpu.buildMicromap(
        {.data = use.micromap->data, .triangles = use.micromap->triangles, .triangleStride = use.micromap->triangleStride, .usages = usages});
    if(!mm.accel)
    {
      m_micromaps.erase(use.micromap.get());
      return;  // drawn without it
    }
  }
  mm.users++;
  m_micromapPrims++;
  prim.micromap                   = use.micromap.get();
  prim.input.micromap             = mm.accel;
  prim.input.micromapBaseTriangle = use.baseTriangle;
  if(use.indexComponentType)
  {
    prim.micromapIndices =
        m_gpu.createBuffer({.size = use.indices.size(), .usage = eBufferAccelInput, .name = "micromap indices"});
    m_gpu.upload(prim.micromapIndices, 0, std::as_bytes(std::span(use.indices)));
    prim.input.micromapIndexAddress = m_gpu.bufferAddress(prim.micromapIndices);
    prim.input.micromapIndexSize    = uint32_t(tinygltf::GetComponentSizeInBytes(uint32_t(use.indexComponentType)));
  }
}

void Residency::unlinkMicromap(Prim& prim)
{
  if(!prim.micromap)
    return;
  retireBuffer(prim.micromapIndices);
  m_micromapPrims--;
  auto it = m_micromaps.find(prim.micromap);
  if(it != m_micromaps.end() && --it->second.users == 0)
  {
    retireAccel(it->second.accel);
    m_micromaps.erase(it);
  }
  prim.micromap        = nullptr;
  prim.micromapIndices = {};
}

void Residency::setShaderFeatures(std::optional<nvvkgltf::SceneFeatureSet> features)
{
  if(features == m_shaderFeatures)
    return;
  m_shaderFeatures = features;
  m_retryHeld      = !m_heldMaterials.empty();
}

bool Residency::drawsBeyondShaderFeatures() const
{
  if(!m_shaderFeatures)
    return false;
  for(const std::optional<MaterialState>& state : m_materialStates)
    if(state && !m_shaderFeatures->isSupersetOf(state->features))
      return true;
  return false;
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
  m_dirtyMaterials.insert(slot);
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
  m_heldMaterials.erase(id);
  freeSpan(m_materialStates[slot]->span, m_materialStates[slot]->spanCount);
  m_materialStates[slot].reset();
  m_dirtyMaterials.insert(slot);  // a hole's row: the default material
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
    slot.textures[size_t(i)] = std::tuple{ref.texture, sampler, srgb.count(i) > 0};
  }
}

void Residency::releaseMaterialTextures(MaterialSlot& slot)
{
  for(const auto& t : slot.textures)
    if(t)
      releaseTexture(std::get<0>(*t), std::get<1>(*t), std::get<2>(*t));
  slot.textures.clear();
}

namespace {
ImageSource imageSourceOf(const TextureRecord& record, bool srgb, uint32_t maxExtent)
{
  return {.uri       = record.uri,
          .bytes     = record.bytes,
          .mimeType  = record.mimeType,
          .pixels    = record.pixels,
          .width     = record.width,
          .height    = record.height,
          .component = record.component,
          .bits      = record.bits,
          .srgb      = srgb,
          .name      = record.name,
          .maxExtent = maxExtent};
}
}  // namespace

void Residency::acquireTexture(const RetainedScene& scene, TextureId id, const GpuSamplerDesc& sampler, bool srgb)
{
  const uint32_t maxExtent = m_gpu.maxImageExtent();
  ImageGpu& image  = m_images[{id, srgb}];
  bool      shared = false;  // the image another texture decoded, or is decoding this sync
  if(image.users++ == 0)
    if(const TextureRecord* record = scene.texture(id))
    {
      // decoded already, for another texture or in grace
      image.key = textureKeyOf(*record, srgb);
      if(image.key)
      {
        image.key->maxExtent = maxExtent;
        // lowered to fit, and let go of: there may be room for it now
        if(auto it = m_decoded.find(*image.key); it != m_decoded.end() && it->second.lowered && it->second.users == 0)
        {
          m_graceBytes -= it->second.info.bytes;
          if(it->second.image)
            m_gpu.retire([&gpu = m_gpu, handle = it->second.image] { gpu.destroyImage(handle); });
          m_decoded.erase(it);
        }
      }
      if(image.key)
        if(auto it = m_decoded.find(*image.key); it != m_decoded.end())
        {
          DecodedImage& decoded = it->second;
          if(decoded.users++ == 0)
            m_graceBytes -= decoded.info.bytes;
          image.image = decoded.image;
          image.info  = decoded.info;
          shared      = true;
          m_imageCounts.reused++;
          // nothing will take what a worker decoded for this wear, and it would stay in CPU memory
          if(!record->uri.empty())
            m_gpu.dropPrefetch(record->uri, srgb);
        }
    }
  if(image.users == 1 && !shared)
    if(const TextureRecord* record = scene.texture(id))
    {
      // decoded with the sync's others (decodeQueued); its slots hold the default image until then
      m_decodeQueue.push_back({{id, srgb}, imageSourceOf(*record, srgb, maxExtent), image.key});
      if(image.key)
        m_decoded[*image.key] = {.users = 1};
    }

  SlotUse& use = m_textureSlots[{id, srgb, keyOf(sampler)}];
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
void Residency::releaseTexture(TextureId id, const GpuSamplerDesc& sampler, bool srgb)
{
  auto slot = m_textureSlots.find({id, srgb, keyOf(sampler)});
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
  auto image = m_images.find({id, srgb});
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
    if(oldest->second.image)
      m_gpu.retire([&gpu = m_gpu, handle = oldest->second.image] { gpu.destroyImage(handle); });
    m_decoded.erase(oldest);
  }
  if(m_graceBytes > budget)  // only idle decodes count, so this can't be
    m_graceBytes = 0;
}

// a texture this small is what is left of a scene that can't fit: nothing lower is worth having
constexpr uint32_t kLeastImageExtent = 256;

uint64_t Residency::lowerableImageBytes() const
{
  uint64_t             bytes = m_graceBytes;
  std::set<TextureKey> seen;  // one image, however many textures are it
  for(const auto& [id, image] : m_images)
  {
    if(!image.image || !image.info.storedLevels || (image.key && !seen.insert(*image.key).second))
      continue;
    const uint64_t extent = std::max(image.info.width, image.info.height);
    if(extent > kLeastImageExtent)
      bytes += image.info.bytes - image.info.bytes * kLeastImageExtent * kLeastImageExtent / (extent * extent);
  }
  return bytes;
}

// The queued images made to fit the GPU memory left, read from their chains before anything is
// decoded. Where they don't fit: the idle decodes go first, then top levels, the largest first, of
// these and of what is resident (decoded again, smaller, in this batch). Geometry made resident this
// sync already counts, as it allocates first, so a scene that grew past the budget comes back under
// it this way with nothing queued at all.
void Residency::fitQueuedToMemory(const RetainedScene& scene)
{
  const std::optional<int64_t> headroom = m_gpu.imageHeadroom();
  if(!headroom)
    return;
  std::vector<ImageFit> fits;
  std::vector<size_t>   queued;  // the queue's entry of each of fits
  uint64_t              need = 0;
  for(size_t i = 0; i < m_decodeQueue.size(); i++)
    if(std::optional<ImageChain> chain = m_gpu.imageChain(m_decodeQueue[i].source))
    {
      const uint32_t levels = uint32_t(chain->levelBytes.size());
      const uint32_t skip   = levelsToSkip(chain->width, chain->height, levels, m_decodeQueue[i].source.maxExtent);
      fits.push_back({std::move(*chain), skip});
      need += fits.back().bytes();
      queued.push_back(i);
    }
  int64_t room = *headroom;
  if(int64_t(need) <= room)
    return;
  room += int64_t(m_graceBytes);
  trimGrace(0);
  if(int64_t(need) <= room)
    return;
  const uint64_t wanted = uint64_t(int64_t(need) - room);

  struct Resident
  {
    ImageKey                  image;
    const TextureRecord*      record = nullptr;
    std::optional<TextureKey> key;
    uint32_t                  skip = 0;
  };
  const size_t          queuedFits = fits.size();
  std::vector<Resident> residents;
  std::set<TextureKey>  seen;  // one image, however many textures are it
  for(const auto& [id, image] : m_images)
  {
    if(!image.image || (image.key && !seen.insert(*image.key).second))
      continue;
    const TextureRecord* record = scene.texture(id.first);
    if(!record)
      continue;
    std::optional<ImageChain> chain = m_gpu.imageChain(imageSourceOf(*record, id.second, 0));
    if(!chain)
      continue;
    uint32_t skip = 0;
    while(levelExtent(chain->width, skip) > image.info.width && skip + 1 < chain->levelBytes.size())
      skip++;
    residents.push_back({id, record, image.key, skip});
    fits.push_back({std::move(*chain), skip});
  }

  const uint64_t freed = lowerToFit(fits, wanted, kLeastImageExtent);
  uint32_t       lowered = 0, largest = 0;
  for(size_t f = 0; f < queuedFits; f++)
  {
    QueuedDecode& q = m_decodeQueue[queued[f]];
    if(fits[f].skip > levelsToSkip(fits[f].chain.width, fits[f].chain.height, uint32_t(fits[f].chain.levelBytes.size()), q.source.maxExtent))
    {
      q.source.maxExtent = fits[f].extent();
      q.lowered          = true;
      lowered++;
      largest = std::max(largest, fits[f].extent());
    }
  }
  for(size_t r = 0; r < residents.size(); r++)
  {
    const ImageFit& fit = fits[queuedFits + r];
    if(fit.skip <= residents[r].skip)
      continue;
    m_decodeQueue.push_back({residents[r].image, imageSourceOf(*residents[r].record, residents[r].image.second, fit.extent()),
                             residents[r].key, true, true});
    lowered++;
    largest = std::max(largest, fit.extent());
  }
  m_imageCounts.lowered += lowered;
  if(lowered > 0)
    m_notices.push_back("to fit GPU memory, " + std::to_string(lowered) + " textures lowered to " + std::to_string(largest)
                        + " or less; " + std::to_string(freed >> 20) + " MB freed of the " + std::to_string(wanted >> 20) + " MB wanted");
  if(lowered > 0 && !std::exchange(m_noticedLowered, true))
    m_userNotices.push_back("GPU memory is nearly full, so textures are being lowered to fit. Opening the scene again restores them if there is room then.");
}

// The queued images decoded in one batch (side by side where the backend can), handed to every
// texture that is that image, and written to their slots. One nothing took meanwhile goes to grace;
// an unreadable one leaves the default image and is tried again at its next acquisition. True when an
// image already drawn was replaced, which changes the frame.
bool Residency::decodeQueued(const RetainedScene& scene, bool sceneGrew)
{
  if(m_decodeQueue.empty() && !sceneGrew)
    return false;
  fitQueuedToMemory(scene);
  if(m_decodeQueue.empty())
    return false;
  std::vector<ImageSource> sources;
  for(const QueuedDecode& q : m_decodeQueue)
    sources.push_back(q.source);
  std::vector<ImageHandle> images(sources.size());
  std::vector<ImageInfo>   infos(sources.size());
  m_gpu.createImages(sources, images, infos);
  m_imageCounts.decoded += uint32_t(sources.size());

  std::set<ImageKey> written;
  bool               replaced = false;
  uint32_t           refused  = 0;
  for(size_t i = 0; i < m_decodeQueue.size(); i++)
  {
    const QueuedDecode& q = m_decodeQueue[i];
    if(q.replaces && !images[i])
      continue;  // it keeps the image it has
    replaced |= q.replaces;
    refused += !q.replaces && infos[i].outOfMemory;
    if(q.key)
    {
      auto decoded = m_decoded.find(*q.key);
      if(decoded == m_decoded.end())
      {
        if(images[i])
          m_gpu.retire([&gpu = m_gpu, handle = images[i]] { gpu.destroyImage(handle); });
        continue;
      }
      for(auto& [id, image] : m_images)
        if(image.key == q.key)
        {
          image.image = images[i];
          image.info  = infos[i];
          if(!images[i])
            image.key.reset();
          written.insert(id);
        }
      if(!images[i])
        m_decoded.erase(decoded);
      else
      {
        if(q.replaces && decoded->second.image)
          m_gpu.retire([&gpu = m_gpu, handle = decoded->second.image] { gpu.destroyImage(handle); });
        decoded->second.image   = images[i];
        decoded->second.info    = infos[i];
        decoded->second.lowered = q.lowered;
        if(decoded->second.users == 0)
          m_graceBytes += infos[i].bytes;
      }
    }
    else if(auto image = m_images.find(q.image); image != m_images.end() && !image->second.key)
    {
      if(q.replaces && image->second.image)
        m_gpu.retire([&gpu = m_gpu, handle = image->second.image] { gpu.destroyImage(handle); });
      image->second.image = images[i];
      image->second.info  = infos[i];
      written.insert(q.image);
    }
    else if(images[i])
      m_gpu.retire([&gpu = m_gpu, handle = images[i]] { gpu.destroyImage(handle); });
  }
  m_decodeQueue.clear();
  for(const auto& [slotKey, use] : m_textureSlots)
    if(const ImageKey image{std::get<0>(slotKey), std::get<1>(slotKey)}; written.count(image))
      m_gpu.writeTextureDescriptor(use.slot, m_images.at(image).image);
  m_imageCounts.refused += refused;
  if(refused > 0)
  {
    m_notices.push_back(std::to_string(refused) + " textures did not fit in GPU memory");
    if(!std::exchange(m_noticedRefused, true))
      m_userNotices.push_back("GPU memory is full: some textures could not be loaded and are drawn without their image.");
  }
  return replaced;
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
    m_heldMaterials.erase(id);
    auto                  it     = m_materials.find(id);
    const MaterialRecord* record = scene.material(id);
    if(it == m_materials.end() || !record || it->second.rev == record->rev)
      continue;
    if(m_shaderFeatures && !m_shaderFeatures->isSupersetOf(nvvkgltf::detectMaterialFeatures(record->payload)))
    {
      m_heldMaterials.insert(id);  // tried again each sync
      continue;
    }
    MaterialSlot before = it->second;
    acquireMaterialTextures(scene, id, it->second);
    releaseMaterialTextures(before);
    it->second.rev = record->rev;
    m_dirtyMaterials.insert(it->second.slot);
    result.changed = true;
  }
}

// The changed material slots converted, each into its own span of the texture-info table (a hole
// as the default material), and uploaded: their rows and spans, or both tables whole when one had to
// be made again (m_tablesMoved) or the spans were repacked. Notes whether a material's draw bits
// changed (m_bitsChanged).
void Residency::writeMaterials(const RetainedScene& scene)
{
  m_spansRepacked = false;
  m_shadeMaterials.resize(m_materialHighWater);
  const std::set<uint32_t> dirty = std::exchange(m_dirtyMaterials, {});
  for(uint32_t s : dirty)
    if(s < m_materialHighWater)
      convertMaterial(scene, s);
  if(m_spansRepacked)  // every live span moved: every row names its texture infos again
    for(uint32_t s = 0; s < m_materialHighWater; s++)
      convertMaterial(scene, s);

  const auto table = [&](BufferHandle& buffer, uint64_t bytes, const char* name) {
    if(!buffer || m_gpu.bufferSize(buffer) < bytes)
    {
      retireBuffer(buffer);
      buffer = m_gpu.createBuffer({.size = std::bit_ceil(std::max<uint64_t>(bytes, 256)), .usage = eBufferStorage, .name = name});
      return true;
    }
    return false;
  };
  constexpr uint64_t kRow      = sizeof(shaderio::GltfShadeMaterial);
  constexpr uint64_t kInfo     = sizeof(shaderio::GltfTextureInfo);
  const bool         rowsMade  = table(m_materialTable, m_shadeMaterials.size() * kRow, "materials");
  const bool         infosMade = table(m_textureInfoTable, m_textureInfos.size() * kInfo, "texture infos");
  m_tablesMoved |= rowsMade || infosMade;
  if(rowsMade || m_spansRepacked)
    m_gpu.upload(m_materialTable, 0, std::as_bytes(std::span(m_shadeMaterials)));
  if(infosMade || m_spansRepacked)
    m_gpu.upload(m_textureInfoTable, 0, std::as_bytes(std::span(m_textureInfos)));
  for(uint32_t s : dirty)
  {
    if(s >= m_materialHighWater)
      continue;
    if(!rowsMade && !m_spansRepacked)
      m_gpu.upload(m_materialTable, s * kRow, std::as_bytes(std::span(&m_shadeMaterials[s], 1)));
    const std::optional<MaterialState>& state = m_materialStates[s];
    if(!infosMade && !m_spansRepacked && state && state->spanCount)
      m_gpu.upload(m_textureInfoTable, state->span * kInfo,
                   std::as_bytes(std::span(m_textureInfos).subspan(state->span, state->spanCount)));
  }
}

// One material slot's row and span from its state: the texture slot table its textures resolve
// through, the span sized to its texture infos (placed again when that changed; the old one freed
// at retire), the row naming them there
void Residency::convertMaterial(const RetainedScene& scene, uint32_t s)
{
  std::optional<MaterialState>&          state = m_materialStates[s];
  std::vector<shaderio::GltfTextureInfo> infos;
  if(!state)
  {
    m_shadeMaterials[s] = nvvkgltf::MaterialCache::convertOne(fv::mat::Material{}, {}, 0, infos);
    return;
  }
  const MaterialRecord* record = scene.material(state->material);
  const MaterialSlot&   slot   = m_materials.at(state->material);
  if(!m_heldMaterials.count(state->material))  // a held edit: the payload its row has, over the textures it holds
    state->payload = record ? record->payload : fv::mat::Material{};
  state->features                   = nvvkgltf::detectMaterialFeatures(state->payload);
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
    const auto& [texture, sampler, srgb] = *slot.textures[i];
    const ImageGpu& image                = m_images.at({texture, srgb});
    table.gpuIndex[i]                    = int(m_textureSlots.at({texture, srgb, keyOf(sampler)}).slot);
    table.samplerSlots[i]          = int(m_samplerSlots.at(keyOf(sampler)).slot);
    table.twoChannelSource[i]      = image.info.twoChannel ? 1 : 0;
    table.meanLuminance[i]         = image.info.meanLuminance;
  }
  const nvvkgltf::MaterialBits before = state->bits;
  state->bits                         = nvvkgltf::materialBits(state->payload);
  m_bitsChanged |= state->bits != before;

  nvvkgltf::MaterialCache::convertOne(state->payload, table, 0, infos);  // how many it needs
  const uint32_t count   = uint32_t(infos.size());
  state->texturesDropped = false;
  if(count != state->spanCount)
  {
    freeSpan(state->span, state->spanCount);
    state->span = state->spanCount = 0;
    if(count && !placeSpan(count, state->span))
    {
      // no room below the limit even repacked: drawn without its textures, never a wrapped index
      // once, though a repack converts it a second time in the same sync
      std::string notice = "the texture-info table is full (" + std::to_string(m_options.textureInfoLimit - 1)
                           + " entries): material '" + state->payload.name + "' is drawn without its textures";
      if(std::find(m_notices.begin(), m_notices.end(), notice) == m_notices.end())
        m_notices.push_back(std::move(notice));
      state->texturesDropped             = true;
      nvvkgltf::TextureSlotTable dropped = table;
      dropped.gpuIndex.assign(std::max<size_t>(n, 1), -1);
      infos.clear();
      m_shadeMaterials[s] = nvvkgltf::MaterialCache::convertOne(state->payload, dropped, 0, infos);
      return;
    }
    state->spanCount = count;
  }
  infos.clear();
  m_shadeMaterials[s] = nvvkgltf::MaterialCache::convertOne(state->payload, table, state->span, infos);
  if(m_textureInfos.size() < size_t(state->span) + count)
    m_textureInfos.resize(size_t(state->span) + count);
  std::copy(infos.begin(), infos.end(), m_textureInfos.begin() + state->span);
}

// A span of `count` entries; when the pool has no room, every live span is placed again from entry
// 1 once (a repack: both tables uploaded whole) and it is tried again
bool Residency::placeSpan(uint32_t count, uint32_t& start)
{
  if(m_textureInfoPool->createRangeID(start, count))
    return true;
  if(m_spansRepacked)
    return false;
  repackSpans();
  return m_textureInfoPool->createRangeID(start, count);
}

void Residency::repackSpans()
{
  m_spansRepacked   = true;
  m_textureInfoPool = poolFromOne(m_options.textureInfoLimit);
  m_textureInfos.assign(1, {});
  uint32_t entries = 0;
  for(std::optional<MaterialState>& state : m_materialStates)
    if(state && state->spanCount)
    {
      const bool ok = m_textureInfoPool->createRangeID(state->span, state->spanCount);
      assert(ok && "the live spans fit the limit they were placed under");
      (void)ok;
      entries += state->spanCount;
    }
  m_notices.push_back("texture-info spans repacked (" + std::to_string(entries) + " entries)");
}

// A span given back once no frame in flight reads it (the pool it came from: a repack's or a
// clear's new pool never takes it)
void Residency::freeSpan(uint32_t start, uint32_t count)
{
  if(count)
    m_gpu.retire([pool = m_textureInfoPool, start, count] { pool->destroyRangeID(start, count); });
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

// ---------- Deformation ----------

bool Residency::meshInUse(MeshId id) const
{
  if(auto it = m_meshes.find(id); it != m_meshes.end())
    if(std::any_of(it->second.prims.begin(), it->second.prims.end(), [](int32_t p) { return p >= 0; }))
      return true;
  return std::any_of(m_drawers.begin(), m_drawers.end(), [&](const auto& d) { return d.first.second == id; });
}

// Whether `entity` draws `mesh` deformed: the mesh has morph targets, or the entity skins it (a skin
// record, and joints and weights for every vertex)
bool Residency::deforms(const RetainedScene& scene, EntityId entity, MeshId mesh) const
{
  const MeshRecord* r = scene.mesh(mesh);
  const Entity*     e = scene.entity(entity);
  if(!r || !e || !e->mesh)
    return false;
  if(!r->targets.empty())
    return true;
  return scene.skin(e->mesh->skin) && !r->positions.empty() && r->joints.size() == r->positions.size()
         && r->weights.size() == r->positions.size();
}

// What a mesh's drawers deform from, beside its streams: joints and weights for a skin, the targets'
// deltas for a morph (a target without normals or tangents adds none of them)
void Residency::makeDeformInputs(const MeshRecord& record, MeshGpu& mesh, bool skin)
{
  const auto make = [&](BufferHandle& buffer, std::span<const std::byte> bytes, const char* name) {
    if(buffer || bytes.empty())
      return;
    buffer = m_gpu.createBuffer({.size = bytes.size(), .usage = eBufferStorage, .name = name});
    m_gpu.upload(buffer, 0, bytes);
  };
  if(skin)
  {
    make(mesh.joints, bytesOf(record.joints), "skin joints");  // uint4 here, int4 in the shader: the same bits
    make(mesh.weights, bytesOf(record.weights), "skin weights");
    if(record.joints1.size() == record.positions.size() && record.weights1.size() == record.positions.size())
    {
      make(mesh.joints1, bytesOf(record.joints1), "skin joints 5-8");
      make(mesh.weights1, bytesOf(record.weights1), "skin weights 5-8");
    }
    if(record.dualQuatShares.size() == record.positions.size())
      make(mesh.dualQuatShares, bytesOf(record.dualQuatShares), "dual quaternion shares");
    // DAZ's order: the cage's
    if(record.cageJoints.size() == record.cage.size() && record.cageWeights.size() == record.cage.size() && !record.cage.empty())
    {
      make(mesh.cageJoints, bytesOf(record.cageJoints), "cage skin joints");
      make(mesh.cageWeights, bytesOf(record.cageWeights), "cage skin weights");
      if(record.cageJoints1.size() == record.cage.size() && record.cageWeights1.size() == record.cage.size())
      {
        make(mesh.cageJoints1, bytesOf(record.cageJoints1), "cage skin joints 5-8");
        make(mesh.cageWeights1, bytesOf(record.cageWeights1), "cage skin weights 5-8");
      }
      if(record.cageDualQuatShares.size() == record.cage.size())
        make(mesh.cageShares, bytesOf(record.cageDualQuatShares), "cage dual quaternion shares");
    }
  }
  if(record.targets.empty() || mesh.positionDeltas)
    return;
  const size_t n = record.positions.size();
  // target-major, n deltas each; zeros for a target that has none of a stream
  const auto pack = [&](std::vector<glm::vec3> MorphTarget::*stream) {
    std::vector<glm::vec3> out;
    if(std::none_of(record.targets.begin(), record.targets.end(), [&](const MorphTarget& t) { return !(t.*stream).empty(); }))
      return out;
    out.assign(n * record.targets.size(), glm::vec3(0.f));
    for(size_t t = 0; t < record.targets.size(); t++)
      if((record.targets[t].*stream).size() == n)
        std::copy((record.targets[t].*stream).begin(), (record.targets[t].*stream).end(), out.begin() + std::ptrdiff_t(t * n));
    return out;
  };
  make(mesh.positionDeltas, bytesOf(pack(&MorphTarget::positions)), "morph position deltas");
  make(mesh.normalDeltas, bytesOf(pack(&MorphTarget::normals)), "morph normal deltas");
  make(mesh.tangentDeltas, bytesOf(pack(&MorphTarget::tangents)), "morph tangent deltas");
  mesh.targetCount = uint32_t(record.targets.size());
}

Residency::DrawerGpu& Residency::acquireDrawer(const RetainedScene& scene, EntityId entity, MeshId id)
{
  const DrawerKey key{entity, id};
  if(auto it = m_drawers.find(key); it != m_drawers.end())
    return it->second;
  MeshGpu&          mesh   = acquireMesh(scene, id);
  const MeshRecord& record = *scene.mesh(id);
  const Entity&     e      = *scene.entity(entity);
  DrawerGpu&        d      = m_drawers[key];
  d.entity               = entity;
  d.mesh                 = id;
  d.morph                = !record.targets.empty();
  d.skin = scene.skin(e.mesh->skin) && record.joints.size() == record.positions.size() && record.weights.size() == record.positions.size();
  makeDeformInputs(record, mesh, d.skin);
  // DAZ's order: a refined mesh's cage posed, then refined, when the cage has its influences
  // (tangents too, when it has them: the chain makes them, nothing skins the base's)
  const bool posed = d.skin && !d.morph && mesh.refine && mesh.cageJoints && mesh.cageWeights
                     && (!mesh.tangents || mesh.refine->tangentTriangles);
  d.dualQuat      = d.skin && (record.dualQuat > 0.f || (posed ? bool(mesh.cageShares) : bool(mesh.dualQuatShares)));
  d.dualQuatShare = record.dualQuat;
  if(posed)
  {
    const auto work = [&](uint64_t count, const char* name) {
      return m_gpu.createBuffer({.size = std::max<uint64_t>(count, 1) * sizeof(glm::vec3), .usage = eBufferStorage, .name = name});
    };
    DrawerGpu::Posed& p = d.posed.emplace();
    p.cage              = work(mesh.refine->levels.front().sourceCount, "posed cage");
    for(const MeshGpu::Refine::Level& l : mesh.refine->levels)
      p.levels.push_back(work(l.rows, "posed level"));
    p.refinedNormals = work(mesh.refine->refinedCount, "posed refined normals");
  }
  d.targetCount = d.morph ? mesh.targetCount : 0;
  d.jointCount  = d.skin ? uint32_t(scene.skin(e.mesh->skin)->joints.size()) : 0;
  const uint64_t n      = mesh.vertexCount;
  const auto     output = [&](BufferHandle& buffer, uint64_t element, BufferUsageFlags usage, const char* name) {
    buffer = m_gpu.createBuffer({.size = n * element, .usage = usage, .name = name});
  };
  output(d.positions, sizeof(glm::vec3), eBufferVertex | eBufferStorage | eBufferAccelInput, "deformed positions");
  if(mesh.normals)
    output(d.normals, sizeof(glm::vec3), eBufferStorage, "deformed normals");
  if(mesh.tangents)
    output(d.tangents, sizeof(glm::vec4), eBufferStorage, "deformed tangents");
  if(d.morph && d.skin)
  {
    output(d.morphedPositions, sizeof(glm::vec3), eBufferStorage, "morphed positions");
    if(mesh.normals)
      output(d.morphedNormals, sizeof(glm::vec3), eBufferStorage, "morphed normals");
    if(mesh.tangents)
      output(d.morphedTangents, sizeof(glm::vec4), eBufferStorage, "morphed tangents");
  }
  output(d.prevPositions, sizeof(glm::vec3), eBufferStorage, "previous positions");
  d.prims.assign(record.submeshes.size(), -1);
  m_packsDirty = true;
  return d;
}

void Residency::releaseDrawer(const DrawerKey& key)
{
  auto it = m_drawers.find(key);
  if(it == m_drawers.end() || --it->second.users > 0)
    return;
  const DrawerGpu& d = it->second;
  for(BufferHandle b : {d.positions, d.normals, d.tangents, d.morphedPositions, d.morphedNormals, d.morphedTangents, d.prevPositions})
    retireBuffer(b);
  for(BufferHandle b : d.posedBuffers())
    retireBuffer(b);
  m_drawers.erase(it);
  m_drawersWritten.erase(key);
  m_packsDirty = true;
  if(!meshInUse(key.second))
    retireMesh(key.second);
}

// The packs laid out again over the drawers there are now: each drawer's joint matrices, normal
// matrices and morph weights at offsets of their own. Every drawer then runs at the next sync,
// writing its inputs at their new offsets.
void Residency::layOutPacks(SyncResult& result)
{
  static_assert(sizeof(glm::mat3) == kNormalMatrixStride, "the packed normal matrices are float3x3 in the shader");
  uint64_t joints = 0, normals = 0, weights = 0, dualQuats = 0;
  for(auto& [key, d] : m_drawers)
  {
    d.jointOffset    = joints;
    d.normalOffset   = normals;
    d.weightOffset   = weights;
    d.dualQuatOffset = dualQuats;
    joints += uint64_t(d.jointCount) * sizeof(glm::mat4);
    normals += uint64_t(d.jointCount) * kNormalMatrixStride;
    weights += uint64_t(d.targetCount) * sizeof(float);
    if(d.dualQuat)
      dualQuats += uint64_t(d.jointCount) * sizeof(JointDualQuat);
    d.ran = false;
  }
  // a pack grows by doubling, so a wear's new drawers seldom make it again
  const auto fit = [&](BufferHandle& buffer, uint64_t bytes, const char* name) {
    if(!bytes || (buffer && m_gpu.bufferSize(buffer) >= bytes))
      return;
    retireBuffer(buffer);
    buffer         = m_gpu.createBuffer({.size = std::bit_ceil(std::max<uint64_t>(bytes, 256)), .usage = eBufferStorage, .name = name});
    result.changed = true;
  };
  fit(m_jointPack, joints, "joint matrices");
  fit(m_normalPack, normals, "normal matrices");
  fit(m_weightPack, weights, "morph weights");
  fit(m_dualQuatPack, dualQuats, "joint dual quaternions");
  m_packsDirty = false;
}

// What the last sync wrote, copied to its previous positions before anything writes it again: the
// frame about to be drawn then reprojects from where the last one drew it
void Residency::snapshotPrevPositions()
{
  bool copied = false;
  for(const DrawerKey& key : m_drawersWritten)
    if(auto it = m_drawers.find(key); it != m_drawers.end())
    {
      DrawerGpu& d = it->second;
      m_gpu.copyBuffer(d.positions, 0, d.prevPositions, 0, m_gpu.bufferSize(d.positions));
      copied = true;
      if(!std::exchange(d.prevValid, true))
        for(int32_t p : d.prims)
          if(p >= 0)
            m_dirtyPrevPrims.insert(uint32_t(p));
    }
  for(MeshId id : m_meshesWritten)
    if(auto it = m_meshes.find(id); it != m_meshes.end() && it->second.prevPositions)
    {
      MeshGpu& mesh = it->second;
      m_gpu.copyBuffer(mesh.positions, 0, mesh.prevPositions, 0, m_gpu.bufferSize(mesh.positions));
      copied = true;
      if(!std::exchange(mesh.prevValid, true))
        for(int32_t p : mesh.prims)
          if(p >= 0)
            m_dirtyPrevPrims.insert(uint32_t(p));
    }
  m_drawersWritten.clear();
  m_meshesWritten.clear();
  if(copied)
    m_gpu.barrier(GpuBarrier::eTransferToShader);  // the copies, then the uploads and passes that overwrite what they read
}

// The deform passes whose inputs changed: the morph weights, the joint matrices, or the mesh's
// vertices (`written`); their prims' BLASes refit. The meshes it deformed join `written`, for the
// emissive table's debounce.
void Residency::syncDeforms(RetainedScene& scene, std::set<MeshId>& written, SyncResult& result)
{
  m_deformCounts = {};
  if(m_drawers.empty())
    return;
  struct Run
  {
    DrawerGpu*             d = nullptr;
    std::vector<float>     weights;
    std::vector<glm::mat4> pose, joints;
    bool                   morph = false, skin = false;
  };
  std::vector<Run> runs;
  for(auto& [key, d] : m_drawers)
  {
    const Entity*     e      = scene.entity(d.entity);
    const MeshRecord* record = scene.mesh(d.mesh);
    if(!e || !e->mesh || !record)
      continue;
    Run run{.d = &d};
    if(d.morph)
    {
      // a target without a weight weighs 0 (AnimationVk); the instance's weights, else the mesh's
      const std::vector<float>& source = e->mesh->morphWeights.empty() ? record->targetWeights : e->mesh->morphWeights;
      run.weights.assign(d.targetCount, 0.f);
      std::copy_n(source.begin(), std::min<size_t>(source.size(), d.targetCount), run.weights.begin());
    }
    if(d.skin)
    {
      run.pose = scene.jointPose(d.entity);
      // (the pose has the mesh's own matrix after the joints')
      if(const uint32_t joints = run.pose.empty() ? 0 : uint32_t(run.pose.size() - 1); joints != d.jointCount)  // the skin was set again with other joints
      {
        d.jointCount = joints;
        m_packsDirty = true;
      }
    }
    runs.push_back(std::move(run));
  }
  if(m_packsDirty)
    layOutPacks(result);

  std::vector<Run*> morphs, skins;
  for(Run& run : runs)
  {
    DrawerGpu& d        = *run.d;
    const bool baseMoved = written.count(d.mesh) > 0;
    run.morph            = d.morph && (!d.ran || baseMoved || run.weights != d.lastWeights);
    // a skin over a morph reads the morph's own output, so it runs again only for what it reads
    run.skin = d.skin && d.jointCount && (!d.ran || baseMoved || run.morph || run.pose != d.lastPose);
    if(run.morph)
      morphs.push_back(&run);
    if(run.skin)
    {
      run.joints = scene.jointMatrices(d.entity);
      skins.push_back(&run);
    }
  }
  if(morphs.empty() && skins.empty())
    return;

  // this frame's weights and matrices, then the passes
  for(Run* run : morphs)
    m_gpu.upload(m_weightPack, run->d->weightOffset, bytesOf(run->weights));
  for(Run* run : skins)
  {
    std::vector<glm::mat3> normalMats(run->joints.size());
    for(size_t j = 0; j < run->joints.size(); j++)
      normalMats[j] = skinNormalMatrix(glm::mat3(run->joints[j]));
    m_gpu.upload(m_jointPack, run->d->jointOffset, bytesOf(run->joints));
    m_gpu.upload(m_normalPack, run->d->normalOffset, bytesOf(normalMats));
    if(run->d->dualQuat)
    {
      std::vector<JointDualQuat> dualQuats(run->joints.size());
      for(size_t j = 0; j < run->joints.size(); j++)
        dualQuats[j] = jointDualQuat(run->joints[j]);
      m_gpu.upload(m_dualQuatPack, run->d->dualQuatOffset, bytesOf(dualQuats));
    }
  }
  m_gpu.barrier(GpuBarrier::eTransferToShader);  // the uploads, and the base streams syncVertices wrote

  for(Run* run : morphs)
  {
    const DrawerGpu& d    = *run->d;
    const MeshGpu&   mesh = m_meshes.at(d.mesh);
    const bool       into = d.skin;  // into the intermediates the skin reads
    m_gpu.dispatchDeform({.kind           = DeformKind::eMorph,
                          .vertexCount    = mesh.vertexCount,
                          .basePositions  = {mesh.positions},
                          .baseNormals    = {mesh.normals},
                          .baseTangents   = {mesh.tangents},
                          .outPositions   = {into ? d.morphedPositions : d.positions},
                          .outNormals     = {into ? d.morphedNormals : d.normals},
                          .outTangents    = {into ? d.morphedTangents : d.tangents},
                          .positionDeltas = {mesh.positionDeltas},
                          .normalDeltas   = {mesh.normalDeltas},
                          .tangentDeltas  = {mesh.tangentDeltas},
                          .morphWeights   = {m_weightPack, d.weightOffset},
                          .targetCount    = d.targetCount});
    m_deformCounts.morphs++;
  }
  if(std::any_of(skins.begin(), skins.end(), [](const Run* r) { return r->d->morph; }) && !morphs.empty())
    m_gpu.barrier(GpuBarrier::eComputeToCompute);
  std::vector<RefineRun> posedRuns;
  for(Run* run : skins)
  {
    const DrawerGpu& d    = *run->d;
    const MeshGpu&   mesh = m_meshes.at(d.mesh);
    if(d.posed)
    {
      // the cage posed (positions only), for the refine chain below
      m_gpu.dispatchDeform({.kind           = DeformKind::eSkin,
                            .vertexCount    = mesh.refine->levels.front().sourceCount,
                            .basePositions  = {mesh.refine->cage},
                            .outPositions   = {d.posed->cage},
                            .joints         = {mesh.cageJoints},
                            .weights        = {mesh.cageWeights},
                            .jointMatrices  = {m_jointPack, d.jointOffset},
                            .normalMatrices = {m_normalPack, d.normalOffset},
                            .jointCount     = d.jointCount,
                            .joints1        = {mesh.cageJoints1},
                            .weights1       = {mesh.cageWeights1},
                            .jointDualQuats = {d.dualQuat ? m_dualQuatPack : BufferHandle{}, d.dualQuatOffset},
                            .dualQuatShares = {mesh.cageShares},
                            .dualQuat       = d.dualQuatShare});
      m_deformCounts.skins++;
      posedRuns.push_back({.mesh           = &mesh,
                           .cage           = d.posed->cage,
                           .refinedNormals = d.posed->refinedNormals,
                           .levels         = d.posed->levels,
                           .positions      = d.positions,
                           .normals        = d.normals,
                           .tangents       = d.tangents});
      continue;
    }
    m_gpu.dispatchDeform({.kind           = DeformKind::eSkin,
                          .vertexCount    = mesh.vertexCount,
                          .basePositions  = {d.morph ? d.morphedPositions : mesh.positions},
                          .baseNormals    = {d.morph ? d.morphedNormals : mesh.normals},
                          .baseTangents   = {d.morph ? d.morphedTangents : mesh.tangents},
                          .outPositions   = {d.positions},
                          .outNormals     = {d.normals},
                          .outTangents    = {d.tangents},
                          .joints         = {mesh.joints},
                          .weights        = {mesh.weights},
                          .jointMatrices  = {m_jointPack, d.jointOffset},
                          .normalMatrices = {m_normalPack, d.normalOffset},
                          .jointCount     = d.jointCount,
                          .joints1        = {mesh.joints1},
                          .weights1       = {mesh.weights1},
                          .jointDualQuats = {d.dualQuat ? m_dualQuatPack : BufferHandle{}, d.dualQuatOffset},
                          .dualQuatShares = {mesh.dualQuatShares},
                          .dualQuat       = d.dualQuatShare});
    m_deformCounts.skins++;
  }
  // DAZ's order: the posed cages refined into the drawers' outputs
  if(!posedRuns.empty())
  {
    m_gpu.barrier(GpuBarrier::eComputeToCompute);
    refineChains(posedRuns);
  }
  m_gpu.barrier(GpuBarrier::eComputeToBuild);

  // what ran is current; its prims' BLASes follow (the ones not built yet build in syncRows)
  bool refits = false;
  for(Run& run : runs)
  {
    if(!run.morph && !run.skin)
      continue;
    DrawerGpu& d = *run.d;
    d.ran        = true;
    if(run.morph)
      d.lastWeights = run.weights;
    if(run.skin)
    {
      d.lastPose   = std::move(run.pose);
      d.lastJoints = run.joints;
    }
    m_drawersWritten.insert({d.entity, d.mesh});
    written.insert(d.mesh);
    for(int32_t p : d.prims)
      if(p >= 0 && m_prims[size_t(p)]->blas)
      {
        refit(*m_prims[size_t(p)]);
        refits = true;
      }
  }
  if(refits)
    m_gpu.barrier(GpuBarrier::eBuildToBuild);  // the TLAS build reads them
  result.verticesWritten = result.changed = true;
}

// The motion tables at capacity: per prim slot the address of its previous positions (0 for none),
// per instance slot the world it was last drawn with
void Residency::writeMotionTables(SyncResult& result)
{
  const auto fit = [&](BufferHandle& buffer, uint64_t bytes, const char* name) {
    if(buffer && m_gpu.bufferSize(buffer) == bytes)
      return false;
    retireBuffer(buffer);
    buffer                = m_gpu.createBuffer({.size = bytes, .usage = eBufferStorage, .name = name});
    result.buffersChanged = result.changed = true;
    return true;
  };
  std::set<uint32_t> prims, nodes;
  prims.swap(m_dirtyPrevPrims);
  nodes.swap(m_dirtyPrevNodes);
  if(fit(m_prevPositionTable, uint64_t(m_primCapacity) * sizeof(uint64_t), "previous positions table"))
    for(uint32_t p = 0; p < m_prims.size(); p++)
      prims.insert(p);
  if(fit(m_prevWorldTable, uint64_t(m_instanceCapacity) * sizeof(glm::mat4), "previous worlds"))
    for(uint32_t s = 0; s < m_instances.size(); s++)
      nodes.insert(s);
  for(const auto& [first, count] : runsOf(prims))
  {
    std::vector<uint64_t> rows(count, 0);
    for(uint32_t i = 0; i < count; i++)
    {
      const std::optional<Prim>& prim = first + i < m_prims.size() ? m_prims[first + i] : std::nullopt;
      if(!prim)
        continue;
      if(!prim->drawer.isNone())
      {
        const DrawerGpu& d = m_drawers.at({prim->drawer, prim->mesh});
        rows[i]            = d.prevValid ? m_gpu.bufferAddress(d.prevPositions) : 0;
      }
      else if(const MeshGpu& mesh = m_meshes.at(prim->mesh); mesh.prevValid)
        rows[i] = m_gpu.bufferAddress(mesh.prevPositions);
    }
    m_gpu.upload(m_prevPositionTable, uint64_t(first) * sizeof(uint64_t), bytesOf(rows));
  }
  for(const auto& [first, count] : runsOf(nodes))
  {
    std::vector<glm::mat4> rows(count, glm::mat4(0.f));
    for(uint32_t i = 0; i < count; i++)
      if(first + i < m_prevWorlds.size() && m_instances[first + i])
        rows[i] = m_prevWorlds[first + i];
    m_gpu.upload(m_prevWorldTable, uint64_t(first) * sizeof(glm::mat4), bytesOf(rows));
  }
  if(!prims.empty() || !nodes.empty())
    result.changed = true;
}

const Residency::DrawerGpu* Residency::drawer(EntityId entity, MeshId mesh) const
{
  auto it = m_drawers.find({entity, mesh});
  return it == m_drawers.end() ? nullptr : &it->second;
}

const glm::mat4* Residency::prevWorld(uint32_t slot) const
{
  return slot < m_prevWorlds.size() && slot < m_instances.size() && m_instances[slot] ? &m_prevWorlds[slot] : nullptr;
}

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
  const bool retryHeld       = std::exchange(m_retryHeld, false);
  // what the last sync wrote or moved still has motion rows to bring back to rest
  const bool motionToSettle = !m_drawersWritten.empty() || !m_meshesWritten.empty() || !m_movedLast.empty();
  // static BLASes wait for their compacted copies (a few frames after their builds)
  const bool compacting = !m_compacting.empty();
  if(!fresh && changes.empty() && !emissiveSettled && !retryHeld && !motionToSettle && !compacting)
    return result;
  // what follows overwrites rows, streams, BLASes and the TLAS the frames in flight may still read
  m_gpu.barrier(GpuBarrier::eFrameToWrite);
  // an appearance change motion vectors can't describe
  result.dlssReset = !changes.materials.empty() || !changes.lights.empty();
  // motion: what moved last sync was drawn where it is now, and positions written last sync are
  // copied before anything writes them again
  for(uint32_t slot : m_movedLast)
    if(slot < m_instances.size() && m_instances[slot])
    {
      m_prevWorlds[slot] = m_instances[slot]->world;
      m_dirtyPrevNodes.insert(slot);
    }
  m_movedLast.clear();
  snapshotPrevPositions();

  std::vector<DrawItem> items = scene.drawItems();
  if(fresh || changes.structure)
    syncStructure(scene, items, result);
  std::set<MaterialId> edited = changes.materials;
  if(retryHeld)
    edited.insert(m_heldMaterials.begin(), m_heldMaterials.end());
  if(!edited.empty())
    syncMaterials(scene, edited, result);
  if(decodeQueued(scene, fresh || changes.structure))
    result.changed = true;
  // refined meshes made resident now: their positions and normals from their cages, as every edit after
  if(!m_refineAcquired.empty())
  {
    std::vector<MeshGpu*> made;
    for(MeshId id : std::exchange(m_refineAcquired, {}))
      if(auto it = m_meshes.find(id); it != m_meshes.end() && it->second.refine)
        made.push_back(&it->second);
    refineOnGpu(made);
  }
  std::set<MeshId> written;
  if(!changes.meshes.empty())
    written = syncVertices(scene, changes.meshes, result);
  syncDeforms(scene, written, result);
  ensureCapacity(result);
  // textures before materials before geometry (see "The sync"): the textures are written as their
  // materials take them, and every material is converted here, before an instance draws with it
  // the tables exist from the first sync, even with no material in them
  if(fresh || !m_dirtyMaterials.empty())
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
  // (an edit held until now and written on its retry counts: it may emit, or stop)
  const bool materialsWritten = m_tablesMoved || result.drawListChanged || !edited.empty();
  syncRows(items, result);
  writeMotionTables(result);
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
  // each draw item, and whether its entity draws its mesh deformed
  std::map<InstanceKey, bool> drawn;
  for(const DrawItem& item : items)
    drawn.try_emplace({item.entity, item.mesh, item.submesh, item.copy}, deforms(scene, item.entity, item.mesh));

  // gone first, so a prim or a material an instance moves to is not retired under it; an instance
  // that now draws deformed, or no longer does, goes and comes back on the other prim
  for(auto it = m_slotOf.begin(); it != m_slotOf.end();)
  {
    const auto      wanted = drawn.find(it->first);
    const Instance& kept   = *m_instances[it->second];
    // a drawer runs the passes it was made for: one that should run others is made again
    const auto sameDrawer = [&] {
      const DrawerGpu* d = kept.deformed ? &m_drawers.at({kept.entity, kept.mesh}) : nullptr;
      if(!d)
        return true;
      const MeshRecord* r = scene.mesh(kept.mesh);
      const Entity*     e = scene.entity(kept.entity);
      const bool        skin = r && e && e->mesh && scene.skin(e->mesh->skin) && r->joints.size() == r->positions.size()
                        && r->weights.size() == r->positions.size();
      return r && d->morph == !r->targets.empty() && d->skin == skin;
    };
    if(wanted != drawn.end() && wanted->second == kept.deformed && sameDrawer())
    {
      ++it;
      continue;
    }
    const uint32_t slot = it->second;
    Instance&      inst = *m_instances[slot];
    releasePrim(inst.prim);
    if(inst.deformed)
      releaseDrawer({inst.entity, inst.mesh});
    releaseMaterial(inst.material);
    m_prevWorlds[slot] = glm::mat4(0.f);  // a hole's motion row is zeroed with its render node
    m_dirtyPrevNodes.insert(slot);
    m_movedLast.erase(slot);
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
        // its raster list and its emitters follow the material, which may be drawn already (so
        // nothing is converted that would say so)
        result.drawListChanged = true;
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
    inst.entity   = item.entity;
    inst.mesh     = item.mesh;
    inst.submesh  = item.submesh;
    inst.copy     = item.copy;
    inst.deformed = drawn.at(key);
    if(inst.deformed)
      acquireDrawer(scene, item.entity, item.mesh).users++;
    inst.prim         = acquirePrim(scene, item.mesh, item.submesh, inst.deformed ? item.entity : EntityId{});
    inst.material     = item.material;
    inst.materialSlot = acquireMaterial(scene, item.material);
    inst.world        = item.world;
    inst.shown        = item.shown;
    m_instances[slot] = inst;
    m_slotOf[key]     = slot;
    m_dirtyNodes.insert(slot);
    // a new slot starts where it is: no motion, never the motion of what had the slot before
    if(slot >= m_prevWorlds.size())
      m_prevWorlds.resize(size_t(slot) + 1, glm::mat4(0.f));
    m_prevWorlds[slot] = item.world;
    m_dirtyPrevNodes.insert(slot);
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
  std::vector<MeshGpu*> refined;  // their positions and normals made from their cages
  for(MeshId id : meshes)
  {
    auto              it     = m_meshes.find(id);
    const MeshRecord* record = scene.mesh(id);
    if(it == m_meshes.end() || !record || it->second.contentRev == record->contentRev)
      continue;
    MeshGpu&           gpu    = it->second;
    const BufferHandle before = gpu.positions;
    uploadStreams(*record, gpu, true);
    if(gpu.refine)
      refined.push_back(&gpu);
    const bool moved = gpu.positions != before;
    // its static prims get motion from here on: their positions are copied at the next sync
    if(std::any_of(gpu.prims.begin(), gpu.prims.end(), [](int32_t p) { return p >= 0; }))
    {
      if(moved || !gpu.prevPositions)
      {
        retireBuffer(gpu.prevPositions);
        gpu.prevPositions = m_gpu.createBuffer(
            {.size = m_gpu.bufferSize(gpu.positions), .usage = eBufferStorage, .name = "previous positions"});
        if(std::exchange(gpu.prevValid, false))
          for(int32_t p : gpu.prims)
            if(p >= 0)
              m_dirtyPrevPrims.insert(uint32_t(p));
      }
      m_meshesWritten.insert(id);
    }
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
    // its drawers' prims: bounds of the new base (the passes and their refits follow in syncDeforms)
    for(auto& [key, d] : m_drawers)
      if(key.second == id)
        for(int32_t p : d.prims)
          if(p >= 0)
            fitBounds(*m_prims[size_t(p)], *record);
    if(moved)
      result.buffersChanged = true;
    result.verticesWritten = true;
    result.changed         = true;
    written.insert(id);
  }
  refineOnGpu(refined);
  if(refits.empty())
    return written;
  m_gpu.barrier(GpuBarrier::eTransferToBuild);
  for(uint32_t p : refits)
    refit(*m_prims[p]);
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
      // drawn last where it was: its motion row, until the next sync brings it to rest
      m_prevWorlds[slot] = inst.world;
      m_dirtyPrevNodes.insert(slot);
      m_movedLast.insert(slot);
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
  // the copies first: built in an earlier frame, they don't wait for this frame's builds
  if(!m_compacting.empty() && takeCompacted())
    tlasRows = true;
  for(uint32_t p : builds)
  {
    build(*m_prims[p]);
    if(m_prims[p]->compacting)
      m_compacting.insert(p);
  }
  if(!builds.empty() || tlasRows)
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
      const MeshGpu&   mesh      = m_meshes.at(prim->mesh);
      const DrawerGpu* d         = prim->drawer.isNone() ? nullptr : &m_drawers.at({prim->drawer, prim->mesh});
      row.indices                = (glm::uvec3*)m_gpu.bufferAddress(prim->indices);
      row.vertexBuffer.positions = (glm::vec3*)m_gpu.bufferAddress(d ? d->positions : mesh.positions);
      row.vertexBuffer.normals   = (glm::vec3*)m_gpu.bufferAddress(d ? d->normals : mesh.normals);
      row.vertexBuffer.tangents  = (glm::vec4*)m_gpu.bufferAddress(d ? d->tangents : mesh.tangents);
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
  desc.emissiveGroups        = (shaderio::EmissiveGroup*)m_gpu.bufferAddress(m_emissiveGroupTable);
  desc.numEmissiveGroups     = int(m_emissiveGroupCount);
  desc.emissiveLeafOfNode    = (int*)m_gpu.bufferAddress(m_emissiveLeafTable);
  desc.numEmissiveLeafOfNode = int(m_emissiveLeafCount);
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
    const Submesh run = record->drawnRun(inst->submesh);
    emitters.push_back({.renderNodeID = s,
                        .renderPrimID = inst->prim,
                        .radiance     = radiance,
                        .doubleSided  = m_materialStates[inst->materialSlot]->payload.doubleSided,
                        .world        = inst->world,
                        .positions    = record->positions,
                        .indices      = std::span(record->indices).subspan(run.firstIndex, run.indexCount)});
  }
  const nvvkgltf::EmissiveTable table = nvvkgltf::buildEmissiveTable(emitters, shaderio::EMISSIVE_GROUP_MAX);
  const uint64_t                bytes = std::span(table.triangles).size_bytes();
  m_descDirty |=
      fitBuffer(m_gpu, m_emissiveTable, bytes, "emissive triangles", [this](BufferHandle b) { retireBuffer(b); });
  if(bytes)
    m_gpu.upload(m_emissiveTable, 0, std::as_bytes(std::span(table.triangles)));
  const uint64_t groupBytes = std::span(table.groups).size_bytes();
  m_descDirty |= fitBuffer(m_gpu, m_emissiveGroupTable, groupBytes, "emissive groups", [this](BufferHandle b) { retireBuffer(b); });
  if(groupBytes)
    m_gpu.upload(m_emissiveGroupTable, 0, std::as_bytes(std::span(table.groups)));
  const uint64_t leafBytes = std::span(table.leafOfNode).size_bytes();
  m_descDirty |= fitBuffer(m_gpu, m_emissiveLeafTable, leafBytes, "emissive leaves", [this](BufferHandle b) { retireBuffer(b); });
  if(leafBytes)
    m_gpu.upload(m_emissiveLeafTable, 0, std::as_bytes(std::span(table.leafOfNode)));
  m_descDirty |= m_emissiveCount != table.triangles.size() || m_emissiveTotalWeight != table.totalWeight
                 || m_emissiveMeanLum != table.meanLum || m_emissiveGroupCount != table.cutCount
                 || m_emissiveLeafCount != table.leafOfNode.size();
  // what the light pick has to choose among, said when it changes (a scene's first upload, a lamp added)
  if(m_emissiveCount != table.triangles.size() || m_emissiveNodeCount != table.groups.size() || m_loggedLightCount != m_lightCount)
    LOGI("[Lights] %u punctual lights; %zu emitters, %zu emissive triangles; light tree of %zu nodes, cut %u\n", m_lightCount,
         emitters.size(), table.triangles.size(), table.groups.size(), table.cutCount);
  m_loggedLightCount    = m_lightCount;
  m_emissiveGroupCount  = table.cutCount;
  m_emissiveNodeCount   = uint32_t(table.groups.size());
  m_emissiveLeafCount   = uint32_t(table.leafOfNode.size());
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

int32_t Residency::textureSlot(TextureId texture, const GpuSamplerDesc& sampler, bool srgb) const
{
  auto it = m_textureSlots.find({texture, srgb, keyOf(sampler)});
  return it == m_textureSlots.end() ? -1 : int32_t(it->second.slot);
}

ImageHandle Residency::image(TextureId texture, bool srgb) const
{
  auto it = m_images.find({texture, srgb});
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
  {
    for(BufferHandle b : {mesh.positions, mesh.normals, mesh.tangents, mesh.uv0, mesh.uv1, mesh.colors, mesh.joints, mesh.cageJoints, mesh.cageWeights, mesh.cageJoints1, mesh.cageWeights1, mesh.cageShares, mesh.joints1,
                          mesh.weights1, mesh.dualQuatShares, mesh.weights, mesh.positionDeltas, mesh.normalDeltas,
                          mesh.tangentDeltas, mesh.prevPositions})
      n += b ? 1 : 0;
    if(mesh.refine)
    {
      for(BufferHandle b : {mesh.refine->cage, mesh.refine->refinedNormals, mesh.refine->drawnFrom, mesh.refine->triangles,
                            mesh.refine->normalStart, mesh.refine->normalTriangles, mesh.refine->offsets,
                            mesh.refine->tangentTriangles, mesh.refine->tangentStart, mesh.refine->tangentCorners})
        n += b ? 1 : 0;
      for(const MeshGpu::Refine::Level& l : mesh.refine->levels)
        for(BufferHandle b : {l.start, l.columns, l.weights, l.out})
          n += b ? 1 : 0;
    }
  }
  for(const auto& [key, d] : m_drawers)
  {
    for(BufferHandle b : {d.positions, d.normals, d.tangents, d.morphedPositions, d.morphedNormals, d.morphedTangents, d.prevPositions})
      n += b ? 1 : 0;
    n += d.posedBuffers().size();
  }
  for(const std::optional<Prim>& p : m_prims)
    n += p ? (p->indices ? 1 : 0) + (p->micromapIndices ? 1 : 0) : 0;
  for(BufferHandle b : {m_renderNodes, m_renderPrims, m_sceneDesc, m_materialTable, m_textureInfoTable, m_lightTable,
                        m_emissiveTable, m_emissiveGroupTable, m_emissiveLeafTable, m_iesTable, m_jointPack, m_normalPack, m_weightPack, m_dualQuatPack, m_prevPositionTable,
                        m_prevWorldTable})
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
  size_t n = (m_tlas ? 1 : 0) + m_micromaps.size();
  for(const std::optional<Prim>& p : m_prims)
    n += p && p->blas ? 1 : 0;
  return n;
}

}  // namespace rs
