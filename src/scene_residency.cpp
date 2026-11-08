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

#include "memory_plan.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_set>

#include <glm/gtc/packing.hpp>

#include <figure/io/timing.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/id_pool.hpp>
#include <nvutils/logger.hpp>

#include "ies_profile.hpp"
#include "scene_gpu_tables.hpp"
#include "scene_skin.hpp"
#include "scene_water.hpp"
#include "shaders/gltf_scene_io.h.slang"
#include "shaders/unit_vector_pack.h.slang"

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
std::span<const std::byte> bytesOf(std::span<const T> v)
{
  return std::as_bytes(v);
}

template <class T>
std::span<const std::byte> bytesOf(const std::vector<T>& v)
{
  return std::as_bytes(std::span(v));
}

template <class T>
std::span<const std::byte> bytesOf(const Stream<T>& v)
{
  return std::as_bytes(std::span<const T>(v));
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
  TextureKey key{.srgb = srgb, .standIn = record.standIn};
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
    for(BufferHandle b : {mesh.positions, mesh.normals, mesh.tangents, mesh.uv0, mesh.uv1, mesh.colors, mesh.masks, mesh.joints, mesh.cageJoints, mesh.cageWeights, mesh.cageJoints1, mesh.cageWeights1, mesh.cageShares, mesh.joints1,
                          mesh.weights1, mesh.dualQuatShares, mesh.weights, mesh.positionDeltas, mesh.normalDeltas,
                          mesh.tangentDeltas, mesh.prevPositions})
      retireBuffer(b);
    retireRefine(mesh);
  }
  for(auto& [key, d] : m_drawers)
  {
    for(BufferHandle b : {d.positions, d.normals, d.tangents, d.morphedPositions, d.morphedNormals, d.morphedTangents, d.prevPositions})
      retireBuffer(b);
  }
  m_drawers.clear();
  for(BufferHandle b : {m_jointPack, m_normalPack, m_weightPack, m_dualQuatPack, m_prevPositionTable, m_prevWorldTable, m_chainScratch, m_unitAxes})
    retireBuffer(b);
  m_jointPack = m_normalPack = m_weightPack = m_dualQuatPack = m_prevPositionTable = m_prevWorldTable = m_chainScratch = m_unitAxes = {};
  m_unitAxesCount = 0;
  m_packsDirty = false;
  m_drawersWritten.clear();
  m_prevResting.clear();
  m_letGoHidden.clear();
  m_bringBack.clear();
  m_detail       = {};
  m_detailAsks.clear();
  m_detailComing = 0;
  m_heldForSettleSince.reset();
  m_hits.clear();
  m_hitsTotal = 0;
  m_hitsToFollow = false;
  m_settleBuilds.clear();
  m_wakeAsked.clear();
  m_drawersWrittenLast.clear();
  m_compacting.clear();
  m_refineAcquired.clear();
  m_meshesWritten.clear();
  m_prevWorlds.clear();
  m_dirtyPrevPrims.clear();
  m_dirtyPrevNodes.clear();
  m_movedLast.clear();
  // the TLAS before the BLASes it points at
  retireAccel(m_tlas);
  m_tlasRefs->rows.clear();
  m_tlasRefs->indexed = false;
  m_tlasRefs->dropped.clear();
  retireHeld();
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
  // a texture's own image goes with the scene; one decoding for a key may still serve the next (m_decoded)
  for(auto it = m_decoding.begin(); it != m_decoding.end();)
    if(!it->second.key)
    {
      m_gpu.cancelDecode(it->first);
      m_decodingBytes -= std::min(m_decodingBytes, it->second.staged);
      it = m_decoding.erase(it);
    }
    else
      ++it;
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
  for(BufferHandle* b : {&m_waterGrids, &m_waterCells, &m_waterRefs, &m_waterItems})
    retireBuffer(std::exchange(*b, {}));
  m_waterGridCount = 0;
  m_waterSlots.clear();
  m_waterDirty = false;
  m_lightCount = m_emissiveCount = m_emissiveGroupCount = m_emissiveNodeCount = m_emissiveLeafCount = m_iesProfileCount = 0;
  m_emissiveTotalWeight = m_emissiveMeanLum = 0.f;
  m_emissiveStale                           = false;
  m_shadeMaterials.clear();
  m_descDirty = false;
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
  m_slotsFollowed.clear();
  m_relitSlots.clear();
  m_texelMeans.clear();
  m_texelAsks.clear();  // what comes for them is let go of
  m_texelAsked.clear();
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
  if(!buffer)
    return;
  const uint64_t bytes = m_gpu.bufferSize(buffer);
  *m_retiring += bytes;
  m_gpu.retire([&gpu = m_gpu, buffer, retiring = m_retiring, bytes] {
    gpu.destroyBuffer(buffer);
    *retiring -= bytes;
  });
}

void Residency::retireAccel(AccelHandle accel, const char* why)
{
  if(!accel)
    return;
  const uint64_t bytes = m_gpu.accelBytes(accel);
  *m_retiring += bytes;
  m_gpu.retire([&gpu = m_gpu, accel, retiring = m_retiring, bytes, refs = m_tlasRefs, why, retired = m_gpu.frame()] {
    // the frames in flight trace the TLAS as built when they were recorded: one built after the retire may still
    // point at it, and the one last built does
    const uint32_t* row     = refs->rowOf(accel.value);
    const auto      dropped = refs->dropped.find(accel.value);
    if(row || (dropped != refs->dropped.end() && dropped->second > retired))
    {
      if(refs->holds[accel.value]++ == 0)
      {
        if(row)
          LOGE("[gpu] A BLAS (%.1f MB) let go of in frame %llu (%s) is still in the TLAS built in frame %llu, at row %u: it "
               "is kept until a TLAS is built without it\n",
               double(bytes) / double(1 << 20), (unsigned long long)retired, why, (unsigned long long)refs->frame, *row);
        else
          LOGE("[gpu] A BLAS (%.1f MB) let go of in frame %llu (%s) was in the TLAS until frame %llu: it is kept a frame "
               "cycle past that\n",
               double(bytes) / double(1 << 20), (unsigned long long)retired, why, (unsigned long long)dropped->second);
      }
      refs->held.push_back({accel, bytes, why});
      return;
    }
    if(dropped != refs->dropped.end())
      refs->dropped.erase(dropped);
    refs->holds.erase(accel.value);
    gpu.destroyAccel(accel);
    *retiring -= bytes;
  });
}

void Residency::retireHeld()
{
  for(const TlasRefs::Held& h : std::exchange(m_tlasRefs->held, {}))
  {
    *m_retiring -= h.bytes;
    retireAccel(h.accel, h.why);
  }
}

void Residency::retireImage(ImageHandle image, uint64_t bytes)
{
  if(!image)
    return;
  *m_retiring += bytes;
  m_gpu.retire([&gpu = m_gpu, image, retiring = m_retiring, bytes] {
    gpu.destroyImage(image);
    *retiring -= bytes;
  });
}

// What a mesh's shape is read from: a refined mesh's cage, as the drawn positions the CPU keeps fall
// behind its morphs.
static const std::vector<glm::vec3>& shapeOf(const MeshRecord& record)
{
  return record.cage.empty() ? record.positions : record.cage;
}

// Some of a mesh's vertices, however many it has: enough to tell how its shape has changed, and few
// enough to look at on every step of a drag.
static std::vector<glm::vec3> shapeSample(const MeshRecord& record)
{
  constexpr size_t               kSampled = 512;
  const std::vector<glm::vec3>&  shape    = shapeOf(record);
  const size_t                   stride   = std::max<size_t>(1, shape.size() / kSampled);
  std::vector<glm::vec3>         out;
  out.reserve(shape.size() / stride + 1);
  for(size_t v = 0; v < shape.size(); v += stride)
    out.push_back(shape[v]);
  return out;
}

// Whether a mesh has changed shape so far from `built` that a refit would leave its BLAS not worth
// tracing: its vertices are, on average, more than this share of its size from where they were, once
// what they all did together is taken out (going somewhere as one leaves a BVH as good as it was).
// Toyen's dForce Long Hair falling took a still from 2.4 s to 27 s refit, and 2.3 s built again. A
// build costs a frame, and starts this count over: on 7TEST at 1080p (2026-11-08), every BLAS of a
// figure wearing FE Elegant Wave (25 M drawn vertices) built again took the sync from 7.9 to 24.2 ms
// of GPU time. Built for speed instead (PREFER_FAST_BUILD), it was 0.5-0.9 ms shorter: not kept.
static bool reshapedFar(const std::vector<glm::vec3>& built, const std::vector<glm::vec3>& now)
{
  constexpr float kFarShare = 0.02f;
  if(built.empty() || built.size() != now.size())
    return false;
  glm::vec3 lo = built[0], hi = built[0], together(0.f);
  for(size_t v = 0; v < built.size(); v++)
  {
    lo = glm::min(lo, built[v]);
    hi = glm::max(hi, built[v]);
    together += now[v] - built[v];
  }
  together /= float(built.size());
  float apart = 0;
  for(size_t v = 0; v < built.size(); v++)
    apart += glm::length(now[v] - built[v] - together);
  return apart / float(built.size()) > glm::length(hi - lo) * kFarShare;
}

void Residency::build(Prim& prim)
{
  prim.compactable = m_options.compactStatic && prim.drawer.isNone();
  prim.compacting  = prim.compactable;
  prim.blas = m_gpu.buildBlas(prim.input, eAccelPreferFastTrace | (prim.compactable ? eAccelAllowCompaction : eAccelAllowUpdate));
  prim.refits   = 0;
  prim.outgrown = false;
}

void Residency::refit(Prim& prim)
{
  if(!prim.outgrown && !prim.compactable && (m_options.refitsBeforeRebuild == 0 || ++prim.refits < m_options.refitsBeforeRebuild))
  {
    m_gpu.refitBlas(prim.blas, prim.input);
    return;
  }
  // refitted enough, out of its shape, or built for compaction: its vertices move after all
  retireAccel(prim.blas, "built again: refitted enough, out of its shape, or built for compaction");
  prim.blas        = m_gpu.buildBlas(prim.input, eAccelPreferFastTrace | eAccelAllowUpdate);
  prim.refits      = 0;
  prim.outgrown    = false;
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
    retireAccel(prim.blas, "compacted");
    prim.blas = copy;
    any       = true;
  }
  return any;
}

// ---------- Records ----------

void Residency::makeRefine(const MeshRecord& record, MeshGpu& gpu)
{
  const RefineTables* t = record.refine.get();
  if(!t || t->levels.empty() || record.cage.size() != t->cageCount || t->drawnFrom.size() != record.vertexCount()
     || t->normalStart.size() != size_t(t->refinedCount()) + 1 || (!t->offsets.empty() && t->offsets.size() != t->refinedCount()))
    return;
  const auto table = [&](const auto& values, const char* name) {
    const BufferHandle b = m_gpu.createBuffer({.size = values.size() * sizeof(values[0]), .usage = eBufferStorage, .name = name, .mayFail = true});
    m_gpu.upload(b, 0, bytesOf(values));
    return b;
  };
  MeshGpu::Refine r;
  r.cage = m_gpu.createBuffer({.size = record.cage.size() * sizeof(glm::vec3), .usage = eBufferStorage, .name = "cage", .mayFail = true});
  m_gpu.upload(r.cage, 0, bytesOf(record.cage));
  uint32_t source = t->cageCount;
  for(size_t i = 0; i < t->levels.size(); i++)
  {
    const RefineTables::Level& l = t->levels[i];
    MeshGpu::Refine::Level     g{.rows = l.rows(), .sourceCount = source};
    g.start   = table(l.start, "stencil starts");
    g.columns = table(l.columns, "stencil columns");
    g.weights = table(l.weights, "stencil weights");
    r.levels.push_back(g);
    source = g.rows;
  }
  r.refinedCount   = t->refinedCount();
  r.triangleCount  = uint32_t(t->triangles.size() / 3);
  r.drawnFrom       = table(t->drawnFrom, "drawn from");
  r.triangles       = table(t->triangles, "refined triangles");
  r.normalStart     = table(t->normalStart, "normal starts");
  r.normalTriangles = table(t->normalTriangles, "normal triangles");
  if(!t->offsets.empty())
    r.offsets = table(t->offsets, "refined offsets");
  if(record.displacement.size() == record.cage.size())
    r.displacement = table(record.displacement, "displacement");
  // the tangents too, over its own corners, when it has them and the UVs they read
  if(!t->tangentIndices.empty() && !record.tangents.empty() && t->tangentStart.size() == record.vertexCount() + 1
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
    if(g->hasBase)  // what poses its cage has no base to make
      runs.push_back({.mesh = g, .positions = g->positions, .normals = g->normals, .tangents = g->tangents});
  if(runs.empty())
    return;
  m_gpu.barrier(GpuBarrier::eTransferToShader);  // the cages
  refineChains(runs);
  // the refits read the positions, the skin passes both, this frame's draws both
  m_gpu.barrier(GpuBarrier::eComputeToBuild);
}

void Residency::ensureUnitAxes(uint64_t count)
{
  if(count <= m_unitAxesCount)
    return;
  retireBuffer(m_unitAxes);
  m_unitAxesCount = std::max<uint64_t>(std::bit_ceil(count), 1024);
  // the normals' Y, then the tangents' X, packed as the skin reads them
  std::vector<uint32_t> axes(m_unitAxesCount * 2);
  std::fill_n(axes.begin(), m_unitAxesCount, shaderio::packUnitVector(glm::vec3(0.f, 1.f, 0.f)));
  std::fill(axes.begin() + ptrdiff_t(m_unitAxesCount), axes.end(), shaderio::packTangent(glm::vec4(1.f, 0.f, 0.f, 1.f)));
  m_unitAxes = m_gpu.createBuffer({.size = axes.size() * sizeof(uint32_t), .usage = eBufferStorage, .name = "unit axes", .mayFail = true});
  m_gpu.upload(m_unitAxes, 0, bytesOf(axes));
}

void Residency::ensureChainScratch(uint64_t bytes)
{
  if(m_gpu.bufferSize(m_chainScratch) < bytes)
  {
    retireBuffer(m_chainScratch);
    m_chainScratch = m_gpu.createBuffer({.size = bytes, .usage = eBufferStorage, .name = "refine chain scratch", .mayFail = true});
    m_gpu.takeAllocationFailed();  // with none the passes do nothing, and the outputs stay as they were
  }
  m_chainScratchUsed = m_gpu.frame();
}

void Residency::refineChains(std::span<const RefineRun> runs)
{
  // what a chain makes on the way to its outputs, the first part only for one that poses its cage
  const auto chainBytes = [](const RefineRun& run) {
    const MeshGpu::Refine& t = *run.mesh->refine;
    const uint64_t         c = t.levels.front().sourceCount;
    uint64_t               bytes = uint64_t(t.refinedCount) * sizeof(glm::vec3);
    for(const MeshGpu::Refine::Level& l : t.levels)
      bytes += uint64_t(l.rows) * sizeof(glm::vec3);
    if(run.posedBy)
      bytes += c * sizeof(glm::vec3) + (t.offsets ? c * 2 * sizeof(uint32_t) + 16 : 0);
    if(run.posedBy && t.displacement)
      bytes += c * sizeof(glm::vec3);
    return bytes;
  };
  uint64_t largest = 0;
  for(const RefineRun& run : runs)
    largest = std::max(largest, chainBytes(run));
  ensureChainScratch(largest);

  bool first = true;
  for(const RefineRun& run : runs)
  {
    if(!std::exchange(first, false))
      m_gpu.barrier(GpuBarrier::eComputeToCompute);  // the next chain writes the same scratch
    const MeshGpu&         mesh = *run.mesh;
    const MeshGpu::Refine& t    = *mesh.refine;
    uint64_t               at   = 0;
    const auto             take = [&](uint64_t bytes, uint64_t alignment = 4) {
      at                  = (at + alignment - 1) / alignment * alignment;
      const BufferRange r{m_chainScratch, at};
      at += bytes;
      return r;
    };
    BufferRange    cage{t.cage}, frameX, frameY;
    const uint64_t cageCount = t.levels.front().sourceCount;
    if(run.posedBy)
    {
      const DrawerGpu& d = *run.posedBy;
      cage               = take(cageCount * sizeof(glm::vec3));
      // the axes turned with the pose: the skin's own normals and tangents of the rest's, its dual
      // quaternions included. A ribbon has one level, so its stencil's columns are cage vertices.
      const bool turns = t.offsets && t.levels.size() == 1 && m_unitAxes && m_unitAxesCount >= cageCount;
      if(turns)
      {
        frameY = take(cageCount * sizeof(uint32_t));
        frameX = take(cageCount * sizeof(uint32_t), 16);
      }
      m_gpu.dispatchDeform({.kind           = DeformKind::eSkin,
                            .vertexCount    = uint32_t(cageCount),
                            .basePositions  = {t.cage},
                            .baseNormals    = turns ? BufferRange{m_unitAxes} : BufferRange{},
                            .baseTangents   = turns ? BufferRange{m_unitAxes, m_unitAxesCount * sizeof(uint32_t)} : BufferRange{},
                            .outPositions   = cage,
                            .outNormals     = frameY,
                            .outTangents    = frameX,
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
      m_gpu.barrier(GpuBarrier::eComputeToCompute);
    }
    // where the flesh has moved on from the skin, added out of place
    if(run.posedBy && t.displacement)
    {
      const BufferRange moved = take(cageCount * sizeof(glm::vec3));
      m_gpu.dispatchDeform({.kind           = DeformKind::eStencil,
                            .vertexCount    = uint32_t(cageCount),
                            .basePositions  = cage,
                            .outPositions   = moved,
                            .stencilOffsets = {t.displacement},
                            .sourceCount    = uint32_t(cageCount)});
      m_deformCounts.displacements++;
      m_gpu.barrier(GpuBarrier::eComputeToCompute);
      cage = moved;
    }
    BufferRange level = cage;
    for(size_t i = 0; i < t.levels.size(); i++)
    {
      const MeshGpu::Refine::Level& l    = t.levels[i];
      const bool                    last = i + 1 == t.levels.size();
      const BufferRange             out  = take(uint64_t(l.rows) * sizeof(glm::vec3));
      m_gpu.dispatchDeform({.kind           = DeformKind::eStencil,
                            .vertexCount    = l.rows,
                            .basePositions  = level,
                            .outPositions   = out,
                            .rowStart       = {l.start},
                            .columns        = {l.columns},
                            .stencilWeights = {l.weights},
                            .stencilOffsets = {last ? t.offsets : BufferHandle{}},
                            .frameX         = frameX,
                            .frameY         = frameY,
                            .sourceCount    = l.sourceCount});
      m_gpu.barrier(GpuBarrier::eComputeToCompute);
      level = out;
    }
    const BufferRange refinedNormals = take(uint64_t(t.refinedCount) * sizeof(glm::vec3));
    m_gpu.dispatchDeform({.kind          = DeformKind::eNormals,
                          .vertexCount   = t.refinedCount,
                          .basePositions = level,
                          .outNormals    = refinedNormals,
                          .rowStart      = {t.normalStart},
                          .columns       = {t.normalTriangles},
                          .triangles     = {t.triangles},
                          .sourceCount   = t.refinedCount,
                          .triangleCount = t.triangleCount});
    m_gpu.barrier(GpuBarrier::eComputeToCompute);
    // the splits: a gather of positions and of normals, each drawn vertex its refined vertex's, the
    // normals packed as the streams hold them
    for(const auto& [from, to, unit] : {std::tuple{level, run.positions, false}, std::tuple{refinedNormals, run.normals, true}})
      if(to)
        m_gpu.dispatchDeform({.kind          = DeformKind::eStencil,
                              .vertexCount   = mesh.vertexCount,
                              .basePositions = from,
                              .outPositions  = unit ? BufferRange{} : BufferRange{to},
                              .columns       = {t.drawnFrom},
                              .outUnit       = unit ? BufferRange{to} : BufferRange{},
                              .sourceCount   = t.refinedCount});
    // the tangents, from the drawn positions, normals and UVs
    if(t.tangentTriangles && run.normals && run.tangents)
    {
      m_gpu.barrier(GpuBarrier::eComputeToCompute);
      m_gpu.dispatchDeform({.kind          = DeformKind::eTangents,
                            .vertexCount   = mesh.vertexCount,
                            .basePositions = {run.positions},
                            .baseNormals   = {run.normals},
                            .outTangents   = {run.tangents},
                            .rowStart      = {t.tangentStart},
                            .columns       = {t.tangentCorners},
                            .triangles     = {t.tangentTriangles},
                            .uvs           = {mesh.uv0},
                            .triangleCount = t.tangentTriangleCount});
    }
  }
}

// The normals and tangents as the vertex streams hold them (shaders/unit_vector_pack.h.slang)
static std::vector<uint32_t> packedNormals(std::span<const glm::vec3> normals)
{
  std::vector<uint32_t> out(normals.size());
  for(size_t i = 0; i < out.size(); i++)
    out[i] = shaderio::packUnitVector(normals[i]);
  return out;
}

static std::vector<uint32_t> packedTangents(std::span<const glm::vec4> tangents)
{
  std::vector<uint32_t> out(tangents.size());
  for(size_t i = 0; i < out.size(); i++)
    out[i] = shaderio::packTangent(tangents[i]);
  return out;
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
      buffer = m_gpu.createBuffer({.size = bytes.size(), .usage = usage, .name = name, .mayFail = true});
    }
    m_gpu.upload(buffer, 0, bytes);
  };
  if(vertices && gpu.refine)
    m_gpu.upload(gpu.refine->cage, 0, bytesOf(record.cage));
  if(gpu.hasBase)
  {
    stream(gpu.positions, bytesOf(record.positions), eBufferVertex | eBufferStorage | eBufferAccelInput, "positions");
    if(!(vertices && gpu.refine))  // made on the GPU, as stream() leaves them
      stream(gpu.normals, bytesOf(packedNormals(record.normals)), eBufferStorage, "normals");
    if((!vertices || gpu.tangentRev != record.tangentRev) && !(vertices && gpu.refine && gpu.refine->tangentTriangles))
      stream(gpu.tangents, bytesOf(packedTangents(record.tangents)), eBufferStorage, "tangents");
  }
  gpu.contentRev = record.contentRev;
  gpu.tangentRev = record.tangentRev;
  if(vertices)  // a vertex edit keeps the count, so the UVs and colours are as they were
    return;
  gpu.builtShape = shapeSample(record);
  stream(gpu.uv0, bytesOf(record.uv0), eBufferStorage, "uv0");
  stream(gpu.uv1, bytesOf(record.uv1), eBufferStorage, "uv1");
  if(!record.colors.empty())
  {
    std::vector<uint32_t> packed(record.colors.size());
    for(size_t i = 0; i < packed.size(); i++)
      packed[i] = glm::packUnorm4x8(record.colors[i]);
    stream(gpu.colors, bytesOf(packed), eBufferStorage, "colors");
  }
  stream(gpu.masks, bytesOf(record.masks), eBufferStorage, "masks");
  gpu.vertexCount = uint32_t(record.vertexCount());
}

// A static prim draws the mesh's own streams, and a drawer that doesn't pose its cage deforms from them. A
// refined mesh's are made from its cage (m_refineAcquired), in this sync.
void Residency::ensureBase(const MeshRecord& record, MeshGpu& gpu)
{
  if(std::exchange(gpu.hasBase, true))
    return;
  const auto stream = [&](BufferHandle& buffer, std::span<const std::byte> bytes, BufferUsageFlags usage, const char* name) {
    if(bytes.empty())
      return;
    buffer = m_gpu.createBuffer({.size = bytes.size(), .usage = usage, .name = name, .mayFail = true});
    m_gpu.upload(buffer, 0, bytes);
  };
  stream(gpu.positions, bytesOf(record.positions), eBufferVertex | eBufferStorage | eBufferAccelInput, "positions");
  stream(gpu.normals, bytesOf(packedNormals(record.normals)), eBufferStorage, "normals");
  stream(gpu.tangents, bytesOf(packedTangents(record.tangents)), eBufferStorage, "tangents");
  // none held on the CPU: the refine that follows makes them
  if(record.thinVertices)
  {
    gpu.positions = m_gpu.createBuffer({.size = record.thinVertices * sizeof(glm::vec3), .usage = eBufferVertex | eBufferStorage | eBufferAccelInput, .name = "positions", .mayFail = true});
    if(record.thinNormals)
      gpu.normals = m_gpu.createBuffer({.size = record.thinVertices * sizeof(uint32_t), .usage = eBufferStorage, .name = "normals", .mayFail = true});
  }
}

Residency::MeshGpu& Residency::acquireMesh(const RetainedScene& scene, MeshId id)
{
  auto it = m_meshes.find(id);
  if(it != m_meshes.end())
  {
    wake(scene, id);  // something new draws it
    return it->second;
  }
  MeshGpu& gpu = m_meshes[id];
  if(const MeshRecord* record = scene.mesh(id))
  {
    gpu.owner = record->owner;
    uploadStreams(*record, gpu);
    makeRefine(*record, gpu);
    // a displacement set before it came (a refined mesh's is makeRefine's)
    if(!gpu.refine && !record->displacement.empty() && record->displacement.size() == record->vertexCount())
    {
      gpu.displacement = m_gpu.createBuffer({.size = record->displacement.size() * sizeof(glm::vec3), .usage = eBufferStorage, .name = "displacement", .mayFail = true});
      m_gpu.upload(gpu.displacement, 0, bytesOf(record->displacement));
    }
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
  for(BufferHandle b : {m.positions, m.normals, m.tangents, m.uv0, m.uv1, m.colors, m.masks, m.joints, m.cageJoints, m.cageWeights, m.cageJoints1, m.cageWeights1, m.cageShares, m.joints1, m.weights1,
                        m.dualQuatShares, m.weights, m.positionDeltas, m.normalDeltas, m.tangentDeltas, m.prevPositions, m.displacement})
    retireBuffer(b);
  retireRefine(m);
  m_meshes.erase(it);
}

void Residency::retireRefine(MeshGpu& m)
{
  if(!m.refine)
    return;
  for(BufferHandle b : {m.refine->cage, m.refine->drawnFrom, m.refine->triangles, m.refine->normalStart,
                        m.refine->normalTriangles, m.refine->offsets, m.refine->tangentTriangles, m.refine->tangentStart,
                        m.refine->tangentCorners, m.refine->displacement})
    retireBuffer(b);
  for(const MeshGpu::Refine::Level& l : m.refine->levels)
    for(BufferHandle b : {l.start, l.columns, l.weights})
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
  if(!d && !mesh.hasBase)
  {
    ensureBase(record, mesh);
    if(mesh.refine)
      m_refineAcquired.push_back(id);
  }
  const Submesh     run    = record.drawnRun(submesh);
  Prim              prim;
  prim.mesh       = id;
  prim.drawer     = drawer;
  prim.submesh    = submesh;
  prim.indexCount = run.indexCount;
  prim.users      = 1;
  const std::span<const uint32_t> indices(record.indices.data() + run.firstIndex, run.indexCount);
  prim.indices = m_gpu.createBuffer(
      {.size = indices.size_bytes(), .usage = eBufferIndex | eBufferStorage | eBufferAccelInput, .name = "indices", .mayFail = true});
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
    // micromaps only accelerate, so without device support they are left out
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
  retireAccel(prim.blas, "its prim released");
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
  if(scene.material(id) && scene.material(id)->water)
    m_waterDirty = true;
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
  if(m_waterSlots.count(slot))
    m_waterDirty = true;
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
          .maxExtent = maxExtent,
          .downscale = record.standIn ? 2u : 1u};
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
          retireImage(it->second.image, it->second.info.bytes);
          m_decoded.erase(it);
        }
      }
      if(image.key)
        if(auto it = m_decoded.find(*image.key); it != m_decoded.end())
        {
          DecodedImage& decoded = it->second;
          if(decoded.users++ == 0)
            m_graceBytes -= decoded.info.bytes;
          image.image   = decoded.image;
          image.info    = decoded.info;
          image.lowered = decoded.lowered;
          shared        = true;
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
  use.since = m_gpu.frame();
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
  else
    retireImage(image.image, image.info.bytes);
}

// The slots of the textures whose image changed (`written`), moved to the new image. A slot a frame in
// flight may sample is never written again in place: a descriptor write is not ordered with the GPU's
// work, so the frame still running reads the new image at once, before the driver has mapped it for
// that frame. That was the lost device with memory full of 11-03: the harness's fault lay inside the
// image a texture move had made a moment before, while the frame before it traced. Each takes a fresh
// slot instead, which no frame uses, written now; its materials' rows point at it from this sync's
// upload, which is ordered after the frames in flight (GpuBarrier::eFrameToWrite); the old slot shows
// the old image until its retire hands it back to the pool, so that image is retired after it.
void Residency::moveSlots(const std::set<ImageKey>& written)
{
  if(written.empty())
    return;
  std::set<std::tuple<TextureId, bool, SamplerKey>> moved;
  for(auto& [slotKey, use] : m_textureSlots)
  {
    const ImageKey image{std::get<0>(slotKey), std::get<1>(slotKey)};
    if(!written.count(image))
      continue;
    const ImageHandle shown = m_images.at(image).image;
    if(use.since == m_gpu.frame())
    {
      // written this frame (a texture's first decode, in the sync that acquired it): no frame has sampled it
      m_gpu.writeTextureDescriptor(use.slot, shown);
      continue;
    }
    uint32_t fresh = 0;
    if(!m_texturePool->createID(fresh))
    {
      // every slot taken: one in use is written again only with no frame in flight
      m_gpu.waitIdle();
      m_gpu.writeTextureDescriptor(use.slot, shown);
      continue;
    }
    m_gpu.writeTextureDescriptor(fresh, shown);  // none: the default image
    m_gpu.retire([&gpu = m_gpu, pool = m_texturePool, s = use.slot] {
      gpu.writeTextureDescriptor(s, {});
      pool->destroyID(s);
    });
    use.slot  = fresh;
    use.since = m_gpu.frame();
    moved.insert(slotKey);
  }
  if(moved.empty())
    return;
  for(const auto& [id, material] : m_materials)
    for(const auto& t : material.textures)
      if(t && moved.count({std::get<0>(*t), std::get<2>(*t), keyOf(std::get<1>(*t))}))
      {
        m_slotsFollowed.insert(material.slot);
        break;
      }
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
    retireImage(oldest->second.image, oldest->second.info.bytes);
    m_decoded.erase(oldest);
  }
  if(m_graceBytes > budget)  // only idle decodes count, so this can't be
    m_graceBytes = 0;
}

uint64_t Residency::lowerableImageBytes(uint32_t leastExtent) const
{
  uint64_t             bytes = m_graceBytes;
  std::set<TextureKey> seen;  // one image, however many textures are it
  for(const auto& [id, image] : m_images)
  {
    if(!image.image || image.info.levels <= 1 || (image.key && !seen.insert(*image.key).second))
      continue;
    const uint64_t extent = std::max(image.info.width, image.info.height);
    if(extent > leastExtent)
      bytes += image.info.bytes - image.info.bytes * leastExtent * leastExtent / (extent * extent);
  }
  return bytes;
}

Residency::LoweredImages Residency::loweredImages() const
{
  LoweredImages        out;
  std::set<TextureKey> seen;  // one image, however many textures are it
  for(const auto& [id, image] : m_images)
    if(image.image && image.lowered && (!image.key || seen.insert(*image.key).second))
    {
      out.count++;
      out.largest = std::max({out.largest, image.info.width, image.info.height});
    }
  return out;
}

std::set<Residency::ImageKey> Residency::pinnedImages(const RetainedScene& scene) const
{
  std::set<ImageKey> out;
  for(const auto& [id, slot] : m_materials)
    if(const MaterialRecord* record = scene.material(id); record && record->keepFull)
      for(const auto& t : slot.textures)
        if(t)
          out.insert({std::get<0>(*t), std::get<2>(*t)});
  std::set<TextureKey> files;
  for(const ImageKey& id : out)
    if(auto it = m_images.find(id); it != m_images.end() && it->second.key)
      files.insert(*it->second.key);
  for(const auto& [id, image] : m_images)
    if(image.key && files.count(*image.key))
      out.insert(id);
  return out;
}

std::vector<std::pair<MeshId, uint64_t>> Residency::hiddenMeshes() const
{
  std::map<MeshId, bool> shown;
  for(const auto& instance : m_instances)
    if(instance)
      shown[instance->mesh] = shown[instance->mesh] || instance->shown;
  const MemoryLedger                       ledger = this->ledger();
  std::vector<std::pair<MeshId, uint64_t>> out;
  for(const MemoryLedger::Mesh& m : ledger.meshes)
    if(const auto it = shown.find(m.mesh); it != shown.end() && !it->second && !m_parked.count(m.mesh))
      out.emplace_back(m.mesh, m.bytes.total());
  return out;
}

std::vector<std::pair<MeshId, uint64_t>> Residency::settleableMeshes() const
{
  const MemoryLedger                       ledger = this->ledger();
  std::vector<std::pair<MeshId, uint64_t>> out;
  for(const MemoryLedger::Mesh& m : ledger.meshes)
  {
    const auto mesh = m_meshes.find(m.mesh);
    if(mesh == m_meshes.end() || mesh->second.settled || m_parked.count(m.mesh))
      continue;
    if(std::any_of(mesh->second.prims.begin(), mesh->second.prims.end(), [](int32_t p) { return p >= 0; }))
      continue;  // a static draw reads what would go
    bool drawn = false, quiet = true;
    for(const auto& [key, d] : m_drawers)
      if(key.second == m.mesh)
      {
        drawn = true;
        quiet = quiet && d.ran && m_gpu.frame() >= d.writtenAt + m_options.settleQuietFrames;
      }
    if(drawn && quiet)
      out.emplace_back(m.mesh, m.bytes.remake + m.bytes.working + m.bytes.unread + uint64_t(double(m.bytes.bvh) * (1.0 - kCompactedBlasShare)));
  }
  // the largest first, so that few have to wake; what is being edited after everything else
  std::stable_sort(out.begin(), out.end(), [&](const auto& a, const auto& b) {
    const bool liveA = m_live.count(a.first) > 0, liveB = m_live.count(b.first) > 0;
    return liveA != liveB ? liveB : a.second > b.second;
  });
  return out;
}

bool Residency::settlingSoon() const
{
  std::set<MeshId> soon;
  for(const auto& [key, d] : m_drawers)
    if(m_gpu.frame() < d.writtenAt + m_options.settleQuietFrames || !d.ran)
      soon.insert(key.second);
  for(MeshId id : soon)
    if(const auto mesh = m_meshes.find(id); mesh != m_meshes.end() && !mesh->second.settled && !m_parked.count(id)
       && std::none_of(mesh->second.prims.begin(), mesh->second.prims.end(), [](int32_t p) { return p >= 0; }))
      return true;
  return false;
}

void Residency::settle(MeshId id)
{
  const auto it = m_meshes.find(id);
  if(it == m_meshes.end() || it->second.settled)
    return;
  // a static draw reads what would go, and a drawer that hasn't run has nothing yet to draw
  if(std::any_of(it->second.prims.begin(), it->second.prims.end(), [](int32_t p) { return p >= 0; }))
    return;
  bool drawn = false;
  for(const auto& [key, d] : m_drawers)
    if(key.second == id)
    {
      if(!d.ran)
        return;
      drawn = true;
    }
  if(!drawn)
    return;
  MeshGpu& mesh = it->second;
  mesh.settled  = true;
  mesh.hadBase  = std::exchange(mesh.hasBase, false);
  for(BufferHandle* b : {&mesh.positions, &mesh.normals, &mesh.tangents, &mesh.joints, &mesh.weights, &mesh.joints1, &mesh.weights1, &mesh.dualQuatShares,
                         &mesh.cageJoints, &mesh.cageWeights, &mesh.cageJoints1, &mesh.cageWeights1, &mesh.cageShares, &mesh.positionDeltas, &mesh.normalDeltas,
                         &mesh.tangentDeltas})
    retireBuffer(std::exchange(*b, {}));
  retireRefine(mesh);
  for(auto& [key, d] : m_drawers)
  {
    if(key.second != id)
      continue;
    for(BufferHandle* b : {&d.morphedPositions, &d.morphedNormals, &d.morphedTangents, &d.prevPositions})
      retireBuffer(std::exchange(*b, {}));
    if(std::exchange(d.prevInUse, false))
      for(int32_t p : d.prims)
        if(p >= 0)
          m_dirtyPrevPrims.insert(uint32_t(p));
    m_prevResting.erase(key);
    for(int32_t p : d.prims)
      if(p >= 0)
        m_settleBuilds.insert(uint32_t(p));
  }
}

bool Residency::isSettled(MeshId id) const
{
  const auto it = m_meshes.find(id);
  return it != m_meshes.end() && it->second.settled;
}

size_t Residency::settledCount() const
{
  size_t n = 0;
  for(const auto& [id, mesh] : m_meshes)
    n += mesh.settled ? 1 : 0;
  return n;
}

void Residency::wake(const RetainedScene& scene, MeshId id)
{
  const auto it = m_meshes.find(id);
  if(it == m_meshes.end() || !it->second.settled)
    return;
  const MeshRecord* record = scene.mesh(id);
  if(!record)
    return;
  MeshGpu& mesh = it->second;
  mesh.settled  = false;
  m_woke        = true;
  makeRefine(*record, mesh);
  if(std::exchange(mesh.hadBase, false))
    ensureBase(*record, mesh);
  for(auto& [key, d] : m_drawers)
  {
    if(key.second != id)
      continue;
    makeDeformInputs(*record, mesh, d.skin, d.posed);
    // what the morph writes for the skin to read is gone: both run at the next change of either
    if(d.morph && d.skin)
    {
      const uint64_t n      = mesh.vertexCount;
      const auto     output = [&](BufferHandle& buffer, uint64_t element, const char* name) {
        buffer = m_gpu.createBuffer({.size = n * element, .usage = eBufferStorage, .name = name, .mayFail = true});
      };
      output(d.morphedPositions, sizeof(glm::vec3), "morphed positions");
      if(record->hasNormals())
        output(d.morphedNormals, sizeof(uint32_t), "morphed normals");
      if(!record->tangents.empty())
        output(d.morphedTangents, sizeof(uint32_t), "morphed tangents");
      d.lastWeights.clear();
    }
    // no need to compact what is about to move
    for(int32_t p : d.prims)
      if(p >= 0)
        m_settleBuilds.erase(uint32_t(p));
  }
  if(m_gpu.takeAllocationFailed())
    m_failedMeshes.insert(id);
}

void Residency::wakeWhatChanged(RetainedScene& scene, const std::set<MeshId>& edited)
{
  for(MeshId id : edited)
    wake(scene, id);
  for(MeshId id : std::exchange(m_wakeAsked, {}))
    wake(scene, id);
  for(const auto& [key, d] : m_drawers)
  {
    const auto mesh = m_meshes.find(key.second);
    if(mesh == m_meshes.end() || !mesh->second.settled)
      continue;
    const Entity*     e      = scene.entity(d.entity);
    const MeshRecord* record = scene.mesh(d.mesh);
    if(!e || !e->mesh || !record)
      continue;
    bool changed = d.skin && scene.jointPose(d.entity) != d.lastPose;
    if(d.morph && !changed)
    {
      const std::vector<float>& source = e->mesh->morphWeights.empty() ? record->targetWeights : e->mesh->morphWeights;
      std::vector<float>        weights(d.targetCount, 0.f);
      std::copy_n(source.begin(), std::min<size_t>(source.size(), d.targetCount), weights.begin());
      changed = weights != d.lastWeights;
    }
    if(changed)
      wake(scene, key.second);
  }
}

std::set<Residency::ImageKey> Residency::favouredImages(const RetainedScene& scene) const
{
  std::set<ImageKey> out;
  for(const auto& [id, slot] : m_materials)
    if(const MaterialRecord* record = scene.material(id); record && record->favoured)
      for(const auto& t : slot.textures)
        if(t)
          out.insert({std::get<0>(*t), std::get<2>(*t)});
  std::set<TextureKey> files;
  for(const ImageKey& id : out)
    if(auto it = m_images.find(id); it != m_images.end() && it->second.key)
      files.insert(*it->second.key);
  for(const auto& [id, image] : m_images)
    if(image.key && files.count(*image.key))
      out.insert(id);
  return out;
}

void Residency::residentFits(const RetainedScene& scene, bool onlyLowered, std::vector<ResidentImage>& residents, std::vector<ImageFit>& fits) const
{
  const std::set<ImageKey> favoured = favouredImages(scene), pinned = pinnedImages(scene);
  std::set<TextureKey>     seen;  // one image, however many textures are it
  // what the last count wanted of each image, and of each file several may be
  using Bins = std::array<uint64_t, kHitBins>;
  std::map<ImageKey, Bins>   imageHits;
  std::map<TextureKey, Bins> fileHits;
  for(const auto& [material, slot] : m_materials)
    if(const auto hits = m_hits.find(material); hits != m_hits.end())
      for(const auto& t : slot.textures)
        if(t)
        {
          const ImageKey key{std::get<0>(*t), std::get<2>(*t)};
          Bins&          mine  = imageHits[key];
          const auto     image = m_images.find(key);
          Bins*          file  = image != m_images.end() && image->second.key ? &fileHits[*image->second.key] : nullptr;
          for(uint32_t bin = 0; bin < kHitBins; bin++)
          {
            mine[bin] += hits->second.bins[bin];
            if(file)
              (*file)[bin] += hits->second.bins[bin];
          }
        }
  for(const auto& [id, image] : m_images)
  {
    if(!image.image || (onlyLowered && !image.lowered) || (image.key && !seen.insert(*image.key).second))
      continue;
    const TextureRecord* record = scene.texture(id.first);
    if(!record)
      continue;
    if(!std::exchange(image.chainRead, true))
    {
      image.chain = m_gpu.imageChain(imageSourceOf(*record, id.second, 0));
      // none to read, but levels on the GPU: it can come down too. Read before anything lowered it.
      if(!image.chain && image.info.levels > 1)
      {
        image.chain.emplace(ImageChain{.width = image.info.width, .height = image.info.height});
        image.levelsMade = true;
        uint64_t top     = image.info.bytes - image.info.bytes / 4;  // of all the levels, the top three quarters
        for(uint32_t level = 0; level < image.info.levels; level++, top /= 4)
          image.chain->levelBytes.push_back(std::max<uint64_t>(top, 1));
      }
    }
    if(!image.chain)
      continue;
    uint32_t skip = 0;
    while(levelExtent(image.chain->width, skip) > image.info.width && skip + 1 < image.chain->levelBytes.size())
      skip++;
    residents.push_back({id, record, image.key, skip});
    ImageFit& fit = fits.emplace_back(ImageFit{*image.chain, skip, favoured.count(id) > 0, pinned.count(id) > 0});
    fit.wholeOnly = image.levelsMade;
    const Bins* wants = nullptr;
    if(image.key)
    {
      if(const auto it = fileHits.find(*image.key); it != fileHits.end())
        wants = &it->second;
    }
    else if(const auto it = imageHits.find(id); it != imageHits.end())
      wants = &it->second;
    if(wants && m_hitsTotal > 0)
    {
      fit.wanting.emplace();
      for(uint32_t bin = 0; bin < kHitBins; bin++)
        (*fit.wanting)[bin] = float(double((*wants)[bin]) / double(m_hitsTotal));
    }
  }
}

void Residency::setItemDetail(ItemDetail detail)
{
  if(!detail.making)
    m_detailComing = 0;
  m_detail = std::move(detail);
  std::erase_if(m_detail.down, [&](const ItemDetail::Down& d) { return std::find(m_detailAsks.begin(), m_detailAsks.end(), d.item) != m_detailAsks.end(); });
}

namespace {
constexpr double kHitSeen = 0.01;  // of the scene's first hits: less, and an item is as good as out of view
}

std::optional<ItemDetail::Up> Residency::nextDetailUp(uint64_t room, bool unseenToo)
{
  size_t best      = m_detail.up.size();
  double bestShare = -1;
  for(size_t i = 0; i < m_detail.up.size(); i++)
  {
    if(m_detail.up[i].bytes > room)
      continue;
    const std::optional<double> share = hitShare(m_detail.up[i].materials);
    if((unseenToo || !share || *share >= kHitSeen) && share.value_or(0) > bestShare)
    {
      best      = i;
      bestShare = share.value_or(0);
    }
  }
  if(best == m_detail.up.size())
    return std::nullopt;
  const ItemDetail::Up up = m_detail.up[best];
  m_detail.up.erase(m_detail.up.begin() + best);
  return up;
}

void Residency::setMaterialHits(const RetainedScene& scene, std::span<const uint32_t> table)
{
  m_hits.clear();
  m_hitsTotal = 0;
  m_hitsToFollow = true;
  for(const auto& [id, slot] : m_materials)
  {
    if((size_t(slot.slot) + 1) * kHitBins > table.size())
      continue;
    MaterialHits& hits = m_hits[id];
    if(const MaterialRecord* record = scene.material(id))
      hits.owner = record->owner;
    for(uint32_t bin = 0; bin < kHitBins; bin++)
    {
      hits.bins[bin] = table[size_t(slot.slot) * kHitBins + bin];
      m_hitsTotal += hits.bins[bin];
    }
  }
}

uint32_t Residency::materialSlotCount() const
{
  uint32_t count = 0;
  for(const auto& [id, slot] : m_materials)
    count = std::max(count, slot.slot + 1);
  return count;
}

std::optional<double> Residency::hitShare(const std::vector<MaterialId>& materials) const
{
  if(m_hitsTotal == 0 || materials.empty())
    return std::nullopt;
  uint64_t sum = 0;
  for(MaterialId id : materials)
  {
    const auto hits = m_hits.find(id);
    if(hits == m_hits.end())
      return std::nullopt;
    for(uint64_t bin : hits->second.bins)
      sum += bin;
  }
  return double(sum) / double(m_hitsTotal);
}

std::optional<ItemDetail::Up> Residency::followHits(const RetainedScene& scene, int64_t room)
{
  // what the last step did has to show in what is free
  if(!m_hitsToFollow || m_hitsTotal == 0 || !m_decodeQueue.empty() || !m_decoding.empty() || m_detailComing > 0 || *m_retiring > 0)
    return std::nullopt;
  const uint64_t free = uint64_t(std::max<int64_t>(room, 0));
  constexpr uint64_t kAtATime = uint64_t(256) << 20;  // what it decodes shows in the memory before the next
  if(raiseImages(scene, std::min(free, kAtATime), 0, true) > 0)
    return std::nullopt;
  if(std::optional<ItemDetail::Up> up = nextDetailUp(free, false))
    return up;
  m_hitsToFollow = giveWayForHits(scene, uint64_t(std::max<int64_t>(-room, 0)));
  return std::nullopt;
}

bool Residency::giveWayForHits(const RetainedScene& scene, uint64_t shortBy)
{
  std::vector<ResidentImage> residents;
  std::vector<ImageFit>      fits;
  residentFits(scene, false, residents, fits);
  std::vector<uint32_t> wanted;
  for(const ImageFit& fit : fits)
    wanted.push_back(levelsToSkip(fit.chain.width, fit.chain.height, uint32_t(fit.chain.levelBytes.size()), m_gpu.maxImageExtent()));
  if(lowerForExchange(fits, wanted, kLeastImageExtent, kFitStep, shortBy) > 0)
  {
    std::string which;
    for(size_t r = 0; r < residents.size(); r++)
      if(fits[r].skip > residents[r].skip)
      {
        m_decodeQueue.push_back({residents[r].image, imageSourceOf(*residents[r].record, residents[r].image.second, fits[r].extent()),
                                 residents[r].key, true, true, fits[r].skip - residents[r].skip});
        which += (which.empty() ? "" : ", ") + residents[r].record->name + " " + std::to_string(fits[r].extent());
      }
    m_notices.push_back("what first hits want back makes room: " + which);
    return true;
  }
  // an item likewise: the one down that rays reach most, and the one not that they reach least
  double most  = kHitSeen;
  bool   found = false;
  for(const ItemDetail::Up& up : m_detail.up)
    if(const std::optional<double> share = hitShare(up.materials); share && *share >= most)
    {
      most  = *share;
      found = true;
    }
  if(!found)
    return false;
  size_t                  first = 0, last = 0;
  const std::vector<DetailStep> steps = detailSteps(first, last);
  const DetailStep*       least = nullptr;
  double                  leastShare = most / 2;
  for(const DetailStep& step : steps)
    if(const std::optional<double> share = hitShare(step.down->materials); share && !step.down->selected && !step.down->pinned && *share < leastShare)
    {
      least      = &step;
      leastShare = *share;
    }
  if(!least)
    return false;
  const uint64_t item = least->down->item;
  m_detailAsks.push_back(item);
  m_detailComing += least->bytes;
  std::erase_if(m_detail.down, [&](const ItemDetail::Down& d) { return d.item == item; });
  return true;
}

std::vector<Residency::DetailStep> Residency::detailSteps(size_t& first, size_t& last) const
{
  std::vector<DetailStep> out;
  first = last = 0;
  if(m_detail.down.empty())
    return out;
  std::map<MeshId, uint64_t> held;
  const MemoryLedger         ledger = this->ledger();
  for(const MemoryLedger::Mesh& m : ledger.meshes)
    held[m.mesh] = m.bytes.total();
  for(const ItemDetail::Down& d : m_detail.down)
  {
    double freed = 0;
    for(const auto& [mesh, share] : d.meshes)
      if(const auto it = held.find(mesh); it != held.end())
        freed += double(it->second) * (1.0 - double(share));
    if(freed >= double(m_options.leastDetailStep))
      out.push_back({&d, uint64_t(freed)});
  }
  // what is freed for what rays lose, as images come down (ImageFit::weight); and what none reach before the rest
  constexpr double kUnseen = 1.0 / 1024;
  const auto weigh = [&](const DetailStep& s) {
    const std::optional<double> share = hitShare(s.down->materials);
    return share ? double(s.bytes) / std::max(*share, kUnseen) : double(s.bytes);
  };
  const auto unseen = [&](const DetailStep& s) {
    const std::optional<double> share = hitShare(s.down->materials);
    return share && *share < kUnseen && !s.down->selected && !s.down->pinned;
  };
  std::stable_sort(out.begin(), out.end(), [&](const DetailStep& a, const DetailStep& b) {
    if(a.down->pinned != b.down->pinned)
      return b.down->pinned;
    if(a.down->selected != b.down->selected)
      return b.down->selected;
    return weigh(a) > weigh(b);
  });
  first = size_t(std::count_if(out.begin(), out.end(), unseen));
  last  = size_t(std::count_if(out.begin(), out.end(), [](const DetailStep& s) { return s.down->pinned; }));
  return out;
}

uint32_t Residency::raiseImages(const RetainedScene& scene, uint64_t room, uint32_t upToExtent, bool onlyWhatHitsWant)
{
  std::vector<ResidentImage> residents;
  std::vector<ImageFit>      fits;
  residentFits(scene, true, residents, fits);
  std::vector<uint32_t> full, wanted;
  for(const ImageFit& fit : fits)
  {
    const uint32_t levels = uint32_t(fit.chain.levelBytes.size());
    full.push_back(levelsToSkip(fit.chain.width, fit.chain.height, levels, m_gpu.maxImageExtent()));
    wanted.push_back(upToExtent ? std::max(full.back(), levelsToSkip(fit.chain.width, fit.chain.height, levels, upToExtent)) : full.back());
    if(onlyWhatHitsWant)
    {
      uint32_t missed = fit.skip;
      while(fit.wanting && missed > wanted.back() && fit.lossWithout(missed - 1) >= kLeastHitLoss)
        missed--;
      wanted.back() = missed;
    }
  }
  // one whose decode is on its way already stays: what it comes back as shows in the memory first
  for(size_t i = 0; i < residents.size(); i++)
    if(decodingNow(residents[i].image, residents[i].key))
      wanted[i] = fits[i].skip;
  raiseToFit(fits, wanted, room);
  uint32_t    raised = 0;
  std::string which;
  for(size_t i = 0; i < residents.size(); i++)
    if(fits[i].skip < residents[i].skip)
    {
      m_decodeQueue.push_back({residents[i].image, imageSourceOf(*residents[i].record, residents[i].image.second, fits[i].extent()),
                               residents[i].key, fits[i].skip > full[i], true});
      raised++;
      which += (which.empty() ? "" : ", ") + residents[i].record->name + " " + std::to_string(fits[i].extent());
    }
  if(raised > 0)
    m_notices.push_back("with room, textures raised: " + which);
  return raised;
}

uint32_t Residency::bringBack(const RetainedScene& scene, uint64_t room)
{
  uint32_t brought = 0;
  // images first: a copy is nothing
  std::set<TextureKey> seen;
  for(const auto& [id, image] : m_images)
  {
    if(!image.image || !m_gpu.imageSpilled(image.image) || image.info.bytes > room || (image.key && !seen.insert(*image.key).second))
      continue;
    const TextureRecord* record = scene.texture(id.first);
    if(!record)
      continue;
    QueuedDecode q{id, imageSourceOf(*record, id.second, 0), image.key, image.lowered, true};
    q.moveOnly = true;
    m_decodeQueue.push_back(std::move(q));
    room -= image.info.bytes;
    brought++;
  }
  if(brought)
    return brought;
  // then one mesh, the largest that fits: it is made again
  const auto  meshSpilled = [&](MeshId id) {
    const MeshGpu* mesh = meshGpu(id);
    if(!mesh)
      return false;
    for(BufferHandle b : {mesh->positions, mesh->normals, mesh->tangents, mesh->uv0, mesh->uv1, mesh->colors, mesh->masks})
      if(m_gpu.bufferSpilled(b))
        return true;
    for(const auto& [key, d] : m_drawers)
      if(key.second == id)
        for(BufferHandle b : {d.positions, d.normals, d.tangents})
          if(m_gpu.bufferSpilled(b))
            return true;
    for(const std::optional<Prim>& prim : m_prims)
      if(prim && prim->mesh == id && m_gpu.accelSpilled(prim->blas))
        return true;
    return false;
  };
  const MemoryLedger ledger = this->ledger();
  MeshId             best;
  uint64_t           bestBytes = 0;
  for(const MemoryLedger::Mesh& m : ledger.meshes)
    if(m.bytes.total() <= room && m.bytes.total() > bestBytes && meshSpilled(m.mesh))
    {
      best      = m.mesh;
      bestBytes = m.bytes.total();
    }
  if(best.isNone())
    return 0;
  m_bringBack.insert(best);
  m_parkedChanged = true;
  return 1;
}

std::string Residency::nextToGo(const RetainedScene& scene) const
{
  if(m_graceBytes > 0)
    return "textures nothing is using";
  const std::vector<std::pair<MeshId, uint64_t>> hidden = hiddenMeshes();
  if(!hidden.empty())
  {
    const auto largest = std::max_element(hidden.begin(), hidden.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    const MeshRecord* record = scene.mesh(largest->first);
    return "a hidden item" + (record && !record->name.empty() ? ": " + record->name : std::string());
  }
  const std::vector<std::pair<MeshId, uint64_t>> settleable = settleableMeshes();
  if(!settleable.empty())
  {
    const MeshRecord* record = scene.mesh(settleable.front().first);
    return "what lets " + (record && !record->name.empty() ? record->name : std::string("an item")) + " be edited at once";
  }
  std::vector<ResidentImage> residents;
  std::vector<ImageFit>      fits;
  residentFits(scene, false, residents, fits);
  const std::vector<ImageFit> before = fits;
  size_t                      detailFirst = 0, detailLast = 0;
  const std::vector<DetailStep> steps     = detailSteps(detailFirst, detailLast);
  std::vector<uint64_t>         detailBytes;
  for(const DetailStep& step : steps)
    detailBytes.push_back(step.bytes);
  const FitPlan plan = planToFit(1, {.images = fits, .leastExtent = kLeastImageExtent, .detail = detailBytes, .detailLast = detailLast, .detailAfter = kDetailAfterExtent, .detailFirst = detailFirst});
  for(size_t i = 0; i < fits.size(); i++)
    if(fits[i].skip != before[i].skip)
      return residents[i].record->name + " at " + std::to_string(before[i].extent());
  if(!plan.detail.empty())
    return "the detail of " + steps[plan.detail.front()].down->name;
  return {};
}

// The queued images made to fit the GPU memory left, read from their chains before anything is
// decoded. Where they don't fit: the idle decodes go first, then top levels, the largest first, of
// these and of what is resident (decoded again, smaller, in this batch). Geometry made resident this
// sync already counts, as it allocates first, so a scene that grew past the budget comes back under
// it this way with nothing queued at all. `slack`: as much again is left free.
void Residency::fitQueuedToMemory(const RetainedScene& scene, uint64_t slack)
{
  const std::optional<int64_t> headroom = m_gpu.imageHeadroom();
  if(!headroom)
    return;
  std::vector<ImageFit> fits;
  std::vector<size_t>   queued;  // the queue's entry of each of fits
  uint64_t              need = 0;
  for(size_t i = 0; i < m_decodeQueue.size(); i++)
    // what comes down frees room rather than taking it
    if(std::optional<ImageChain> chain = m_decodeQueue[i].dropLevels > 0 ? std::nullopt : chainOf(m_decodeQueue[i]))
    {
      const uint32_t levels = uint32_t(chain->levelBytes.size());
      const uint32_t skip   = levelsToSkip(chain->width, chain->height, levels, m_decodeQueue[i].source.maxExtent);
      fits.push_back({std::move(*chain), skip});
      need += fits.back().bytes();
      queued.push_back(i);
    }
  // what is decoding takes its memory when it is made, as it is
  for(const auto& [ticket, q] : m_decoding)
    need += q.chainBytes;
  const int64_t room = *headroom - int64_t(slack);
  if(int64_t(need) <= room)
    return;
  // with nothing new waiting, a step: the memory is read again before the next
  uint64_t wanted = uint64_t(int64_t(need) - room);
  if(queued.empty())
    wanted = std::min(wanted, kFitStep);
  // what is being made will free this before long
  wanted -= std::min(wanted, m_detailComing);

  const size_t               queuedFits = fits.size();
  const std::set<ImageKey>   favoured   = favouredImages(scene), pinned = pinnedImages(scene);
  for(size_t f = 0; f < queuedFits; f++)
  {
    fits[f].favoured = favoured.count(m_decodeQueue[queued[f]].image) > 0;
    fits[f].pinned   = pinned.count(m_decodeQueue[queued[f]].image) > 0;
  }
  std::vector<ResidentImage> residents;
  residentFits(scene, false, residents, fits);

  const std::vector<std::pair<MeshId, uint64_t>> hidden = hiddenMeshes();
  std::vector<uint64_t>                          hiddenBytes;
  for(const auto& [mesh, bytes] : hidden)
    hiddenBytes.push_back(bytes);
  const std::vector<std::pair<MeshId, uint64_t>> settleable = settleableMeshes();
  std::vector<uint64_t>                          settleBytes;
  for(const auto& [mesh, bytes] : settleable)
    settleBytes.push_back(bytes);
  size_t                        detailFirst = 0, detailLast = 0;
  const std::vector<DetailStep> steps       = detailSteps(detailFirst, detailLast);
  std::vector<uint64_t>         detailBytes;
  for(const DetailStep& step : steps)
    detailBytes.push_back(step.bytes);
  // with nothing new waiting for the room, what would show waits for what is about to settle, which won't. For
  // as long as settling takes: then on, for a while, a clip that never goes quiet doesn't hold it up again.
  const uint64_t frame = m_gpu.frame();
  if(!m_heldForSettleSince || frame >= *m_heldForSettleSince + 4 * m_options.settleQuietFrames)
    m_heldForSettleSince = frame;
  const bool hold = queued.empty() && frame < *m_heldForSettleSince + m_options.settleQuietFrames && settlingSoon();
  const FitPlan plan = planToFit(wanted, {.idleBytes = m_graceBytes, .hidden = hiddenBytes, .settle = settleBytes, .images = hold ? std::span<ImageFit>{} : std::span<ImageFit>(fits),
                                          .leastExtent = kLeastImageExtent, .detail = hold ? std::span<const uint64_t>{} : std::span<const uint64_t>(detailBytes),
                                          .detailLast = hold ? 0 : detailLast, .detailAfter = kDetailAfterExtent, .detailFirst = hold ? 0 : detailFirst});
  trimGrace(m_graceBytes - plan.idle);
  {
    std::set<uint64_t> asked;
    for(size_t i : plan.detail)
    {
      asked.insert(steps[i].down->item);
      m_detailAsks.push_back(steps[i].down->item);
      m_detailComing += steps[i].bytes;
    }
    std::erase_if(m_detail.down, [&](const ItemDetail::Down& d) { return asked.count(d.item) > 0; });  // `steps` read them: not before here
  }
  for(size_t i : plan.settled)
    settle(settleable[i].first);
  for(size_t i : plan.hidden)
  {
    m_letGoHidden.insert(hidden[i].first);
    m_parked.insert(hidden[i].first);
    m_parkedChanged = true;
  }
  const uint64_t freed = plan.freed();
  uint32_t       lowered = 0, largest = 0;
  std::string    which;  // for the log: each one's name and long edge, favoured ones marked
  const auto     noteLowered = [&](const std::string& name, const ImageFit& fit) {
    lowered++;
    largest = std::max(largest, fit.extent());
    which += (which.empty() ? "" : ", ") + name + " " + std::to_string(fit.extent()) + (fit.favoured ? "*" : "");
  };
  for(size_t f = 0; f < queuedFits; f++)
  {
    QueuedDecode& q = m_decodeQueue[queued[f]];
    if(fits[f].skip > levelsToSkip(fits[f].chain.width, fits[f].chain.height, uint32_t(fits[f].chain.levelBytes.size()), q.source.maxExtent))
    {
      q.source.maxExtent = fits[f].extent();
      q.lowered          = true;
      noteLowered(std::string(q.source.name), fits[f]);
    }
  }
  for(size_t r = 0; r < residents.size(); r++)
  {
    const ImageFit& fit = fits[queuedFits + r];
    if(fit.skip <= residents[r].skip)
      continue;
    m_decodeQueue.push_back({residents[r].image, imageSourceOf(*residents[r].record, residents[r].image.second, fit.extent()),
                             residents[r].key, true, true, fit.skip - residents[r].skip});
    noteLowered(residents[r].record->name, fit);
  }
  m_imageCounts.lowered += lowered;
  if(lowered > 0)
    m_notices.push_back("to fit GPU memory, " + std::to_string(lowered) + " textures lowered to " + std::to_string(largest)
                        + " or less; " + std::to_string(freed >> 20) + " MB freed of the " + std::to_string(wanted >> 20) + " MB wanted: " + which);
  if(lowered > 0 && !std::exchange(m_noticedLowered, true))
    m_userNotices.push_back("GPU memory is nearly full, so textures are being lowered to fit. They come back by themselves when there is room.");
}

bool Residency::decodingNow(const ImageKey& image, const std::optional<TextureKey>& key) const
{
  for(const auto& [ticket, q] : m_decoding)
    if(sameImage(q, image, key))
      return true;
  return false;
}

const std::optional<ImageChain>& Residency::chainOf(QueuedDecode& q)
{
  if(!q.chain)
    q.chain = m_gpu.imageChain(q.source);
  return *q.chain;
}

void Residency::cancelDecodesOf(const QueuedDecode& q)
{
  for(auto it = m_decoding.begin(); it != m_decoding.end();)
    if(sameImage(it->second, q.image, q.key))
    {
      m_gpu.cancelDecode(it->first);
      m_decodingBytes -= std::min(m_decodingBytes, it->second.staged);
      it = m_decoding.erase(it);
    }
    else
      ++it;
}

// The queued images begun on the backend's workers, and those it has decoded made into images, handed
// to every texture that is that image and written to their slots: the render thread never waits for a
// decode (a 4096 JPEG is a few hundred milliseconds), and the image a texture shows meanwhile stays, the
// default one before its first. What only comes down, or only moves, is copied in this frame. One
// nothing took meanwhile goes to grace; an unreadable one leaves the default image and is tried again
// at its next acquisition.
Residency::DecodesLanded Residency::decodeQueued(const RetainedScene& scene, bool sceneGrew)
{
  const std::optional<uint64_t> slack = std::exchange(m_fitSlack, std::nullopt);
  const bool                    ready = m_gpu.decodesReady();
  if(m_decodeQueue.empty() && !sceneGrew && !slack && !ready)
    return {};
  fv::timing::Accum time("textures queued, fitted and made");
  if(!m_decodeQueue.empty() || sceneGrew || slack)
    fitQueuedToMemory(scene, slack.value_or(0));
  if(m_decodeQueue.empty() && !ready)
    return {};

  DecodesLanded      landed;
  std::set<ImageKey> written;
  // the images replaced, retired once their slots are (moveSlots): a slot shows its image until its retire
  std::vector<std::pair<ImageHandle, uint64_t>> replacedImages;
  bool                                          restored = false;
  uint32_t                                      refused  = 0;
  const auto                                    noted    = [&](const QueuedDecode& q, ImageHandle made, const ImageInfo& info) {
    landed.replaced |= q.replaces;
    landed.arrived |= !q.replaces && made;
    restored |= q.replaces && !q.lowered;
    refused += !q.replaces && info.outOfMemory;
  };
  // What an image showed goes once its slots have moved: a size it was lowered to, or the same image
  // decoded twice (let go of and taken again while its first decode ran)
  const auto install = [&](const QueuedDecode& q, ImageHandle made, const ImageInfo& info) {
    if(q.key)
    {
      auto decoded = m_decoded.find(*q.key);
      if(decoded == m_decoded.end())  // let go of meanwhile, and trimmed from grace
      {
        retireImage(made, info.bytes);
        return;
      }
      if(!made && decoded->second.image)
        return;  // it keeps the image it has
      noted(q, made, info);
      for(auto& [id, image] : m_images)
        if(image.key == q.key)
        {
          image.image   = made;
          image.info    = info;
          image.lowered = q.lowered;
          if(!made)
            image.key.reset();
          written.insert(id);
        }
      if(!made)
      {
        m_decoded.erase(decoded);
        return;
      }
      DecodedImage& d = decoded->second;
      if(d.image)
      {
        replacedImages.emplace_back(d.image, d.info.bytes);
        if(d.users == 0)
          m_graceBytes -= std::min(m_graceBytes, d.info.bytes);
      }
      d.image   = made;
      d.info    = info;
      d.lowered = q.lowered;
      if(d.users == 0)
        m_graceBytes += info.bytes;
    }
    else if(auto image = m_images.find(q.image); image != m_images.end() && !image->second.key)
    {
      if(!made && image->second.image)
        return;  // it keeps the image it has
      noted(q, made, info);
      if(image->second.image)
        replacedImages.emplace_back(image->second.image, image->second.info.bytes);
      image->second.image   = made;
      image->second.info    = info;
      image->second.lowered = q.lowered;
      written.insert(q.image);
    }
    else
      retireImage(made, info.bytes);
  };

  // what only comes down, or only moves, is copied from what it is, with nothing read or decoded; one
  // that comes down undoes a raise still decoding, which a move doesn't
  std::vector<QueuedDecode> decoding;
  for(QueuedDecode& q : m_decodeQueue)
  {
    const auto image = q.dropLevels || q.moveOnly ? m_images.find(q.image) : m_images.end();
    ImageInfo  info;
    if(const ImageHandle copy = image != m_images.end() ? m_gpu.copyImageLevels(image->second.image, q.dropLevels, info) : ImageHandle{})
    {
      if(q.dropLevels)
        cancelDecodesOf(q);
      install(q, copy, info);
      m_imageCounts.copied++;
    }
    else if(!q.moveOnly)
      decoding.push_back(std::move(q));
  }
  // What decodes at once is bounded (ResidencyOptions::decodeBatchBytes), the rest waiting in the queue, in
  // order: by its chain's levels from the first kept where the header says, else its texels as given or as
  // sized. The first always begins when none is decoding.
  const auto stagedBytes = [&](QueuedDecode& q, uint64_t& chainBytes) -> uint64_t {
    chainBytes                = 0;
    const ImageSource& source = q.source;
    if(const std::optional<ImageChain>& chain = chainOf(q))
    {
      for(size_t l = levelsToSkip(chain->width, chain->height, uint32_t(chain->levelBytes.size()), source.maxExtent); l < chain->levelBytes.size(); l++)
        chainBytes += chain->levelBytes[l];
      return chainBytes;
    }
    if(!source.pixels.empty())
      return uint64_t(source.pixels.size()) * 4 / 3;
    if(source.width > 0 && source.height > 0)
      return uint64_t(source.width) * uint64_t(source.height) * 4 * 4 / 3;
    return uint64_t(64) << 20;  // nothing says: about a 4096 texture's
  };
  std::vector<QueuedDecode> later;
  for(QueuedDecode& q : decoding)
  {
    uint64_t       chainBytes = 0;
    const uint64_t bytes      = later.empty() ? stagedBytes(q, chainBytes) : 0;
    if(!later.empty() || (!m_decoding.empty() && m_decodingBytes + bytes > m_options.decodeBatchBytes))
    {
      later.push_back(std::move(q));
      continue;
    }
    cancelDecodesOf(q);
    q.staged     = bytes;
    q.chainBytes = chainBytes;
    m_decodingBytes += bytes;
    const uint64_t ticket = m_gpu.startDecode(q.source);
    m_decoding.emplace(ticket, std::move(q));
  }
  m_decodeQueue = std::move(later);
  // and those done made, as many as a sync stages: what of the decodes is still the render thread's
  std::optional<fv::timing::Accum> timeMade;
  if(ready)
    timeMade.emplace("textures made from their decodes");
  const std::vector<GpuBackend::MadeImage> made = m_gpu.takeDecodes(m_options.decodeBatchBytes);
  for(const GpuBackend::MadeImage& m : made)
  {
    auto it = m_decoding.find(m.ticket);
    if(it == m_decoding.end())  // a backend makes none it was told to cancel
    {
      retireImage(m.image, m.info.bytes);
      continue;
    }
    const QueuedDecode q = std::move(it->second);
    m_decoding.erase(it);
    m_decodingBytes -= std::min(m_decodingBytes, q.staged);
    install(q, m.image, m.info);
  }
  m_imageCounts.decoded += uint32_t(made.size());
  moveSlots(written);
  for(const auto& [image, bytes] : replacedImages)
    retireImage(image, bytes);
  m_imageCounts.refused += refused;
  if(restored && loweredImages().count == 0 && std::exchange(m_noticedLowered, false))
    m_userNotices.push_back("GPU memory has room again: the textures are back to full size.");
  if(refused > 0)
  {
    m_notices.push_back(std::to_string(refused) + " textures did not fit in GPU memory");
    if(!std::exchange(m_noticedRefused, true))
      m_userNotices.push_back("GPU memory is full: some textures could not be loaded and are drawn without their image.");
  }
  return landed;
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
// changed (m_bitsChanged), and the slots whose emitter weight changed (m_relitSlots): a texture that
// landed with another mean than the one it stood in with.
void Residency::writeMaterials(const RetainedScene& scene)
{
  m_spansRepacked = false;
  m_shadeMaterials.resize(m_materialHighWater);
  const std::set<uint32_t>                dirty = std::exchange(m_dirtyMaterials, {});
  std::vector<std::pair<uint32_t, glm::vec3>> lit;  // a repack moves no mean: the dirty slots are the ones to compare
  for(uint32_t s : dirty)
    if(s < m_materialHighWater)
    {
      lit.emplace_back(s, nvvkgltf::emitterSelectionRadiance(m_shadeMaterials[s]));
      convertMaterial(scene, s);
    }
  if(m_spansRepacked)  // every live span moved: every row names its texture infos again
    for(uint32_t s = 0; s < m_materialHighWater; s++)
      convertMaterial(scene, s);
  for(const auto& [s, before] : lit)
    if(nvvkgltf::emitterSelectionRadiance(m_shadeMaterials[s]) != before)
      m_relitSlots.insert(s);

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
  return scene.skin(e->mesh->skin) && r->skins();
}

// What a mesh's drawers deform from, beside its streams: joints and weights for a skin, the targets'
// deltas for a morph (a target without normals or tangents adds none of them)
void Residency::makeDeformInputs(const MeshRecord& record, MeshGpu& mesh, bool skin, bool posed)
{
  const auto make = [&](BufferHandle& buffer, std::span<const std::byte> bytes, const char* name) {
    if(buffer || bytes.empty())
      return;
    buffer = m_gpu.createBuffer({.size = bytes.size(), .usage = eBufferStorage, .name = name, .mayFail = true});
    m_gpu.upload(buffer, 0, bytes);
  };
  if(skin && !posed)
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
  }
  if(skin && posed)
  {
    // DAZ's order: the cage's
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
  d.skin = scene.skin(e.mesh->skin) && record.skins();
  // DAZ's order: a refined mesh's cage posed, then refined, when the cage has its influences
  // (tangents too, when it has them: the chain makes them, nothing skins the base's)
  const bool cageInfluences = !record.cage.empty() && record.cageJoints.size() == record.cage.size() && record.cageWeights.size() == record.cage.size();
  d.posed = d.skin && !d.morph && mesh.refine && cageInfluences && (record.tangents.empty() || mesh.refine->tangentTriangles);
  makeDeformInputs(record, mesh, d.skin, d.posed);
  if(!d.posed && !mesh.hasBase)
  {
    ensureBase(record, mesh);
    if(mesh.refine)
      m_refineAcquired.push_back(id);
  }
  d.dualQuat      = d.skin && (record.dualQuat > 0.f || (d.posed ? bool(mesh.cageShares) : bool(mesh.dualQuatShares)));
  d.dualQuatShare = record.dualQuat;
  d.targetCount = d.morph ? mesh.targetCount : 0;
  d.jointCount  = d.skin ? uint32_t(scene.skin(e.mesh->skin)->joints.size()) : 0;
  const uint64_t n      = mesh.vertexCount;
  const auto     output = [&](BufferHandle& buffer, uint64_t element, BufferUsageFlags usage, const char* name) {
    buffer = m_gpu.createBuffer({.size = n * element, .usage = usage, .name = name, .mayFail = true});
  };
  output(d.positions, sizeof(glm::vec3), eBufferVertex | eBufferStorage | eBufferAccelInput, "deformed positions");
  // the normals and tangents packed (shaders/unit_vector_pack.h.slang), 4 bytes each
  if(record.hasNormals())
    output(d.normals, sizeof(uint32_t), eBufferStorage, "deformed normals");
  if(!record.tangents.empty())
    output(d.tangents, sizeof(uint32_t), eBufferStorage, "deformed tangents");
  if(d.morph && d.skin)
  {
    output(d.morphedPositions, sizeof(glm::vec3), eBufferStorage, "morphed positions");
    if(record.hasNormals())
      output(d.morphedNormals, sizeof(uint32_t), eBufferStorage, "morphed normals");
    if(!record.tangents.empty())
      output(d.morphedTangents, sizeof(uint32_t), eBufferStorage, "morphed tangents");
  }
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
  m_drawers.erase(it);
  m_drawersWritten.erase(key);
  m_drawersWrittenLast.erase(key);
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
    // (its matrices are written before each run, wherever they go: a wear needn't skin every figure again)
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
  m_drawersWrittenLast.swap(m_drawersWritten);
  m_drawersWritten.clear();
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
  m_meshesWritten.clear();
  if(copied)
    m_gpu.barrier(GpuBarrier::eTransferToShader);  // the copies, then the uploads and passes that overwrite what they read
}

void Residency::copyPrevPositions(DrawerGpu& d)
{
  if(!d.ran)
    return;  // no motion the frame it appears
  if(!d.prevPositions)
  {
    d.prevPositions = m_gpu.createBuffer({.size = m_gpu.bufferSize(d.positions), .usage = eBufferStorage, .name = "previous positions", .mayFail = true});
    if(m_gpu.takeAllocationFailed())  // only its motion goes without
    {
      retireBuffer(std::exchange(d.prevPositions, {}));
      return;
    }
  }
  m_gpu.copyBuffer(d.positions, 0, d.prevPositions, 0, m_gpu.bufferSize(d.positions));
  d.prevValid = true;
  m_prevResting.erase({d.entity, d.mesh});
  if(!std::exchange(d.prevInUse, true))
    for(int32_t p : d.prims)
      if(p >= 0)
        m_dirtyPrevPrims.insert(uint32_t(p));
}

void Residency::settlePrevPositions()
{
  for(const DrawerKey& key : m_drawersWrittenLast)
    if(auto it = m_drawers.find(key); it != m_drawers.end() && !m_drawersWritten.count(key))
    {
      DrawerGpu& d = it->second;
      if(d.prevPositions)
        m_prevResting[key] = m_gpu.frame();
      d.prevInUse = false;
      d.prevValid = true;
      for(int32_t p : d.prims)
        if(p >= 0)
          m_dirtyPrevPrims.insert(uint32_t(p));
    }
  m_drawersWrittenLast.clear();
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
      // a target without a weight weighs 0; the instance's weights, else the mesh's
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

  // what the last frame drew, before the passes write over it
  for(Run& run : runs)
    if(run.morph || run.skin)
      copyPrevPositions(*run.d);
  uint64_t turningCage = 0;
  for(const Run* run : skins)
    if(const MeshGpu& mesh = m_meshes.at(run->d->mesh); run->d->posed && mesh.refine->offsets)
      turningCage = std::max<uint64_t>(turningCage, mesh.refine->levels.front().sourceCount);
  ensureUnitAxes(turningCage);

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
  // the drawn vertices' skin, its positions into `positions`
  const auto skinDrawn = [&](const DrawerGpu& d, const MeshGpu& mesh, BufferRange positions) {
    m_gpu.dispatchDeform({.kind           = DeformKind::eSkin,
                          .vertexCount    = mesh.vertexCount,
                          .basePositions  = {d.morph ? d.morphedPositions : mesh.positions},
                          .baseNormals    = {d.morph ? d.morphedNormals : mesh.normals},
                          .baseTangents   = {d.morph ? d.morphedTangents : mesh.tangents},
                          .outPositions   = positions,
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
  };
  std::vector<const Run*> displaced;  // drawn as they are, with a displacement after their skin
  for(Run* run : skins)
  {
    const DrawerGpu& d    = *run->d;
    const MeshGpu&   mesh = m_meshes.at(d.mesh);
    if(d.posed)
    {
      posedRuns.push_back({.mesh = &mesh, .posedBy = &d, .positions = d.positions, .normals = d.normals, .tangents = d.tangents});
      continue;
    }
    if(mesh.displacement)
    {
      displaced.push_back(run);
      continue;
    }
    skinDrawn(d, mesh, {d.positions});
  }
  // Where the flesh has moved on from the skin, on meshes drawn as they are: skinned into the scratch,
  // then that and the displacement added into the drawer's positions; a mesh at a time, as they share
  // the scratch. The normals stay the skin's: an offset is at most a few cm, and smooth.
  if(!displaced.empty())
  {
    uint64_t largest = 0;
    for(const Run* run : displaced)
      largest = std::max<uint64_t>(largest, uint64_t(m_meshes.at(run->d->mesh).vertexCount) * sizeof(glm::vec3));
    ensureChainScratch(largest);
    bool first = true;
    for(const Run* run : displaced)
    {
      const DrawerGpu& d    = *run->d;
      const MeshGpu&   mesh = m_meshes.at(d.mesh);
      if(!std::exchange(first, false))
        m_gpu.barrier(GpuBarrier::eComputeToCompute);  // the last one's add read the scratch
      skinDrawn(d, mesh, {m_chainScratch});
      m_gpu.barrier(GpuBarrier::eComputeToCompute);
      m_gpu.dispatchDeform({.kind           = DeformKind::eStencil,
                            .vertexCount    = mesh.vertexCount,
                            .basePositions  = {m_chainScratch},
                            .outPositions   = {d.positions},
                            .stencilOffsets = {mesh.displacement},
                            .sourceCount    = mesh.vertexCount});
      m_deformCounts.displacements++;
    }
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
    d.writtenAt  = m_gpu.frame();
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
        // at rest what the last frame drew is what its positions hold
        const DrawerGpu& d = m_drawers.at({prim->drawer, prim->mesh});
        rows[i]            = d.prevValid ? m_gpu.bufferAddress(d.prevInUse ? d.prevPositions : d.positions) : 0;
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
  // (a prim's row says what its motion is: the picture isn't changed by one coming to rest)
  if(!nodes.empty())
    result.changed = true;
  else if(!prims.empty())
    m_gpu.barrier(GpuBarrier::eTransferToShader);
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

void Residency::holdBackArrivals(const RetainedScene& scene, std::vector<DrawItem>& items)
{
  m_arriving.clear();
  std::set<MeshId> coming;
  uint64_t         triangles = 0;
  for(const DrawItem& item : items)
  {
    if(m_meshes.count(item.mesh) || coming.count(item.mesh) || m_arriving.count(item.mesh))
      continue;
    const MeshRecord* record = scene.mesh(item.mesh);
    const uint64_t    more   = record ? record->indices.size() / 3 : 0;
    if(triangles > 0 && triangles + more > m_options.arrivalTriangles)
      m_arriving.insert(item.mesh);
    else
    {
      coming.insert(item.mesh);
      triangles += more;
    }
  }
  if(!m_arriving.empty())
    std::erase_if(items, [&](const DrawItem& item) { return m_arriving.count(item.mesh) > 0; });
}

void Residency::setParked(MeshId mesh, bool parked)
{
  if(parked ? m_parked.insert(mesh).second : m_parked.erase(mesh) > 0)
    m_parkedChanged = true;
}

void Residency::retryParked()
{
  if(m_parked.size() == m_letGoHidden.size())
    return;
  m_parked        = m_letGoHidden;
  m_parkedChanged = true;
}

bool Residency::parkFailed(const RetainedScene& scene, std::vector<DrawItem>& items)
{
  if(m_failedMeshes.empty())
    return false;
  std::string names;
  for(MeshId mesh : m_failedMeshes)
  {
    m_parked.insert(mesh);
    if(const MeshRecord* record = scene.mesh(mesh); record && m_noticedParked.insert(mesh).second)
      names += (names.empty() ? "" : ", ") + record->name;
  }
  m_failedMeshes.clear();
  if(!names.empty())
    m_userNotices.push_back("GPU and system memory are both full, so these aren't drawn for now: " + names + ".");
  std::erase_if(items, [&](const DrawItem& item) { return m_parked.count(item.mesh) > 0; });
  return true;
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
  // images asked of it between syncs (raiseImages, lowerImages), or decoded since and waiting to be made
  const bool imagesAsked = !m_decodeQueue.empty() || m_fitSlack || m_gpu.decodesReady();
  // a settled mesh's BVH, looked at now and then: it waits for room
  const bool settleBuild = !m_settleBuilds.empty() && m_gpu.frame() % 30 == 0;
  const bool liveAsleep  = !m_wakeAsked.empty();
  // a textured emitter's texels being averaged: the table takes them when they come
  const bool texelsAsked = !m_texelAsks.empty();
  const bool parkedChanged = std::exchange(m_parkedChanged, false);
  // what a large chain worked in goes once nothing has posed for a while
  constexpr uint64_t kChainScratchIdleFrames = 600;
  if(m_chainScratch && m_gpu.frame() > m_chainScratchUsed + kChainScratchIdleFrames)
  {
    retireBuffer(std::exchange(m_chainScratch, {}));
    retireBuffer(std::exchange(m_unitAxes, {}));
    m_unitAxesCount = 0;
  }
  // and last frame's positions a little sooner, as a drag that only pauses keeps them
  constexpr uint64_t kPrevPositionsIdleFrames = 120;
  for(auto it = m_prevResting.begin(); it != m_prevResting.end();)
  {
    if(m_gpu.frame() <= it->second + kPrevPositionsIdleFrames)
    {
      ++it;
      continue;
    }
    if(auto d = m_drawers.find(it->first); d != m_drawers.end() && !d->second.prevInUse)
      retireBuffer(std::exchange(d->second.prevPositions, {}));
    it = m_prevResting.erase(it);
  }
  if(!fresh && changes.empty() && !emissiveSettled && !retryHeld && !motionToSettle && !compacting && !imagesAsked && !parkedChanged && !settleBuild && !liveAsleep && !texelsAsked
     && m_arriving.empty())
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
  // what was let go of while hidden comes back with its first shown draw
  bool cameBack = false;
  if(!m_letGoHidden.empty())
    for(const DrawItem& item : items)
      if(item.shown && m_letGoHidden.erase(item.mesh))
      {
        m_parked.erase(item.mesh);
        cameBack = true;
      }
  std::erase_if(items, [&](const DrawItem& item) { return m_parked.count(item.mesh) > 0; });
  if(fresh || changes.structure || parkedChanged || cameBack || !m_arriving.empty())
  {
    holdBackArrivals(scene, items);
    // what is brought back goes and comes in one sync: its old memory is still held when the new is asked for
    if(!m_bringBack.empty())
    {
      std::vector<DrawItem> without = items;
      std::erase_if(without, [&](const DrawItem& item) { return m_bringBack.count(item.mesh) > 0; });
      syncStructure(scene, without, result);
      m_bringBack.clear();
    }
    syncStructure(scene, items, result);
    if(parkFailed(scene, items))
      syncStructure(scene, items, result);
  }
  std::set<MaterialId> edited = changes.materials;
  if(retryHeld)
    edited.insert(m_heldMaterials.begin(), m_heldMaterials.end());
  if(!edited.empty())
    syncMaterials(scene, edited, result);
  // what is about to change, ready first
  std::set<MeshId> reshaped = changes.meshes;
  reshaped.insert(changes.displacements.begin(), changes.displacements.end());
  wakeWhatChanged(scene, reshaped);
  const DecodesLanded landed = decodeQueued(scene, fresh || changes.structure);
  result.imagesMoved         = landed.replaced;
  // a texture's first image, decoded since the sync that took it: drawn with the default one until now
  result.changed = result.changed || landed.arrived;
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
  if(!changes.displacements.empty())
    for(MeshId id : syncDisplacements(scene, changes.displacements))
      written.insert(id);
  if(m_gpu.takeAllocationFailed())
    m_failedMeshes.insert(written.begin(), written.end());
  syncDeforms(scene, written, result);
  settlePrevPositions();
  ensureCapacity(result);
  // textures before materials before geometry (see "The sync"): the textures are written as their
  // materials take them, and every material is converted here, before an instance draws with it
  // the tables exist from the first sync, even with no material in them. A material whose rows only
  // follow its textures to their new slots (moveSlots) draws the same picture: a still goes on
  const bool materialsChanged = fresh || !m_dirtyMaterials.empty();
  const bool slotsFollowed    = !m_slotsFollowed.empty();  // uploaded all the same: the barrier below
  m_dirtyMaterials.insert(m_slotsFollowed.begin(), m_slotsFollowed.end());
  m_slotsFollowed.clear();
  if(!m_dirtyMaterials.empty() || fresh)
  {
    writeMaterials(scene);
    result.changed = result.changed || materialsChanged;
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
  // what couldn't be built goes, and the rest is drawn
  if(parkFailed(scene, items))
  {
    syncStructure(scene, items, result);
    buildDrawLists();
    result.drawListChanged = true;
    syncRows(items, result);
  }
  writeMotionTables(result);
  if(fresh || changes.iesProfiles)
  {
    syncIesProfiles(scene);
    result.changed = true;
  }
  if(fresh || changes.water || m_waterDirty)
  {
    syncWater(scene);
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
  // dial or drape frame, with the weights lagging meanwhile (the shader reads live geometry). A shown
  // instance whose material's weight moved (a texture landed with its mean; it may start or stop
  // emitting) counts too: the table holds the weights both sides of the MIS read.
  bool emitterMoved = false, emitterWritten = false, emitterRelit = false;
  for(const std::optional<Instance>& inst : m_instances)
  {
    if(inst && inst->shown && m_relitSlots.count(inst->materialSlot))
      emitterRelit = true;
    if(inst && emits(*inst))
    {
      emitterMoved |= changes.transforms.count(inst->entity) > 0;
      emitterWritten |= written.count(inst->mesh) > 0;
    }
  }
  m_relitSlots.clear();
  // a textured emitter's own triangles weighed, now that their texels are averaged
  const bool texelsLanded = takeTexelAverages();
  if(emitterWritten)
  {
    m_emissiveStale     = true;
    m_emitterWriteFrame = m_gpu.frame();
  }
  if(fresh || materialsWritten || result.visibilityChanged || emitterMoved || emitterRelit || texelsLanded
     || (emissiveSettled && !emitterWritten))
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
  // nothing else moved than what leaves the picture as it was: a still goes on
  result.samePicture = !result.changed && (result.imagesMoved || m_compactedOnly);
  result.changed     = result.changed || result.samePicture;
  m_compactedOnly    = false;
  // everything uploaded above, before the frame's shading reads it
  if(result.changed || slotsFollowed || std::exchange(m_woke, false))
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
      const bool        skin = r && e && e->mesh && scene.skin(e->mesh->skin) && r->skins();
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
    if(m_gpu.takeAllocationFailed())
      m_failedMeshes.insert(item.mesh);
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

std::set<MeshId> Residency::syncDisplacements(const RetainedScene& scene, const std::set<MeshId>& meshes)
{
  std::set<MeshId> moved;
  for(MeshId id : meshes)
  {
    auto              it     = m_meshes.find(id);
    const MeshRecord* record = scene.mesh(id);
    if(it == m_meshes.end() || !record)
      continue;
    // a refined mesh's per cage vertex, before its levels; one drawn as it is, per drawn vertex
    MeshGpu&                      mesh   = it->second;
    BufferHandle&                 buffer = mesh.refine ? mesh.refine->displacement : mesh.displacement;
    const std::vector<glm::vec3>& d      = record->displacement;
    if(d.size() != (mesh.refine ? record->cage.size() : size_t(record->vertexCount())) || d.empty())
    {
      if(!buffer)
        continue;
      retireBuffer(std::exchange(buffer, {}));
    }
    else
    {
      if(!buffer || m_gpu.bufferSize(buffer) != d.size() * sizeof(glm::vec3))
      {
        retireBuffer(buffer);
        buffer = m_gpu.createBuffer({.size = d.size() * sizeof(glm::vec3), .usage = eBufferStorage, .name = "displacement", .mayFail = true});
      }
      m_gpu.upload(buffer, 0, bytesOf(d));
    }
    moved.insert(id);
  }
  return moved;
}

// Every mesh whose vertices were edited, uploaded again; its BLASes refit, or rebuilt when a
// stream's buffer had to be replaced (the build data holds its address). Returns the meshes written.
std::set<MeshId> Residency::syncVertices(RetainedScene& scene, const std::set<MeshId>& meshes, SyncResult& result)
{
  std::set<MeshId>      written;
  std::vector<uint32_t> refits;  // after every upload, behind one barrier
  std::vector<MeshGpu*> refined;  // their positions and normals made from their cages
  std::optional<fv::timing::Accum> timeSync;  // a frame's, when it has edits
  if(!meshes.empty())
    timeSync.emplace("vertex sync");
  for(MeshId id : meshes)
  {
    auto              it     = m_meshes.find(id);
    const MeshRecord* record = scene.mesh(id);
    if(it == m_meshes.end() || !record || it->second.contentRev == record->contentRev)
      continue;
    MeshGpu& gpu = it->second;
    // each edited mesh's upload and bounds: a refined one sends its cage, an unrefined one its whole
    // positions and normals and rescans its index lists (the host logs both once a second)
    fv::timing::Accum  timeMesh(gpu.refine ? "vertex sync: refined mesh" : "vertex sync: unrefined mesh");
    const BufferHandle before = gpu.positions;
    uploadStreams(*record, gpu, true);
    if(gpu.refine)
      refined.push_back(&gpu);
    const bool moved = gpu.positions != before;
    // come a long way from what its BLASes were built for: they are built again, static and drawn
    // deformed alike, when their refits would come
    if(std::vector<glm::vec3> shape = shapeSample(*record); reshapedFar(gpu.builtShape, shape))
    {
      gpu.builtShape = std::move(shape);
      for(int32_t p : gpu.prims)
        if(p >= 0)
          m_prims[size_t(p)]->outgrown = true;
      for(auto& [key, d] : m_drawers)
        if(key.second == id)
          for(int32_t p : d.prims)
            if(p >= 0)
              m_prims[size_t(p)]->outgrown = true;
    }
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
        retireAccel(prim.blas, "its mesh's buffers moved");
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
  m_allNodes = false;
  // (a BLAS held for the TLAS that pointed at it: one built without it, so that it can go)
  bool tlasRows = !nodes.empty() || !m_tlasRefs->held.empty();
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
  const bool otherRows = tlasRows;
  const bool compacted = !m_compacting.empty() && takeCompacted();
  tlasRows             = tlasRows || compacted;
  // a settled mesh's BVH built again to be compacted, one a sync, once there is room for it beside the old
  bool rebuilt = false;
  for(auto it = m_settleBuilds.begin(); it != m_settleBuilds.end() && !rebuilt;)
  {
    const uint32_t p = *it;
    if(p >= m_prims.size() || !m_prims[p] || !m_prims[p]->blas)
    {
      it = m_settleBuilds.erase(it);
      continue;
    }
    Prim&          prim  = *m_prims[p];
    const uint64_t bytes = m_gpu.accelBytes(prim.blas);
    if(const std::optional<int64_t> room = m_gpu.imageHeadroom(); room && *room < int64_t(bytes * 2))
      break;  // (and its scratch)
    m_gpu.barrier(GpuBarrier::eTransferToBuild);
    const AccelHandle built = m_gpu.buildBlas(prim.input, eAccelPreferFastTrace | eAccelAllowCompaction);
    if(m_gpu.takeAllocationFailed())
    {
      // no memory for it beside the one it has, which draws on: tried again in a later sync
      retireAccel(built, "found no memory to be built again compacted");
      break;
    }
    retireAccel(prim.blas, "settled: built again to be compacted");
    prim.blas        = built;
    prim.compactable = prim.compacting = true;
    prim.refits      = 0;
    prim.outgrown    = false;
    m_compacting.insert(p);
    it      = m_settleBuilds.erase(it);
    rebuilt = true;
  }
  tlasRows = tlasRows || rebuilt;
  for(uint32_t p : builds)
  {
    build(*m_prims[p]);
    if(m_prims[p]->compacting)
      m_compacting.insert(p);
  }
  if(!builds.empty() || tlasRows)
    m_gpu.barrier(GpuBarrier::eBuildToBuild);
  // (their scratch is found at that barrier, so any of them): the sync comes back here without them
  if(m_gpu.takeAllocationFailed() && !builds.empty())
  {
    for(uint32_t p : builds)
      m_failedMeshes.insert(m_prims[p]->mesh);
    return;
  }

  if(tlasRows || !builds.empty() || result.drawListChanged || result.tlasHandleChanged || result.verticesWritten || m_bitsChanged)
  {
    // the same surfaces, in a smaller BVH: nothing to start again for
    const bool onlyCompacted = (compacted || rebuilt) && !otherRows && builds.empty() && !result.drawListChanged && !result.tlasHandleChanged && !result.verticesWritten && !m_bitsChanged;
    buildTlas();
    m_gpu.barrier(GpuBarrier::eBuildToTrace);
    m_compactedOnly = onlyCompacted;
    result.changed  = result.changed || !onlyCompacted;
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
      row.vertexBuffer.normals   = (uint32_t*)m_gpu.bufferAddress(d ? d->normals : mesh.normals);
      row.vertexBuffer.tangents  = (uint32_t*)m_gpu.bufferAddress(d ? d->tangents : mesh.tangents);
      row.vertexBuffer.texCoord0 = (glm::vec2*)m_gpu.bufferAddress(mesh.uv0);
      row.vertexBuffer.texCoord1 = (glm::vec2*)m_gpu.bufferAddress(mesh.uv1);
      row.vertexBuffer.colors    = (glm::uint*)m_gpu.bufferAddress(mesh.colors);
      row.vertexBuffer.masks     = (glm::uint*)m_gpu.bufferAddress(mesh.masks);
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
  // what it points at now, and since when each it pointed at before no longer is (a drag's builds point at the same)
  std::vector<uint64_t> now(rows.size());
  for(size_t s = 0; s < rows.size(); s++)
    now[s] = rows[s].blas.value;
  if(now != m_tlasRefs->rows)
  {
    const std::unordered_set<uint64_t> kept(now.begin(), now.end());
    for(uint64_t blas : m_tlasRefs->rows)
      if(blas && !kept.count(blas))
        m_tlasRefs->dropped[blas] = m_gpu.frame();
    m_tlasRefs->rows    = std::move(now);
    m_tlasRefs->indexed = false;
  }
  m_tlasRefs->frame = m_gpu.frame();
  retireHeld();
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
  desc.waterGrids            = (shaderio::WaterGrid*)m_gpu.bufferAddress(m_waterGrids);
  desc.waterCells            = (glm::uvec2*)m_gpu.bufferAddress(m_waterCells);
  desc.waterRefs             = (uint32_t*)m_gpu.bufferAddress(m_waterRefs);
  desc.waterItems            = (shaderio::WaterItem*)m_gpu.bufferAddress(m_waterItems);
  desc.numWaterGrids         = int(m_waterGridCount);
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

std::optional<Residency::TexelKey> Residency::texelKey(const Instance& inst, shaderio::GltfTextureInfo* info) const
{
  if(inst.materialSlot >= m_materialStates.size() || !m_materialStates[inst.materialSlot] || inst.materialSlot >= m_shadeMaterials.size())
    return std::nullopt;
  const MaterialState& state = *m_materialStates[inst.materialSlot];
  const int            entry = state.payload.emissiveMap.index;
  const uint32_t       row   = m_shadeMaterials[inst.materialSlot].emissiveTexture;  // 0: none (textureInfos[0] is reserved)
  const auto           slot  = m_materials.find(state.material);
  if(entry < 0 || row == 0 || row >= m_textureInfos.size() || state.texturesDropped || slot == m_materials.end()
     || size_t(entry) >= slot->second.textures.size() || !slot->second.textures[entry])
    return std::nullopt;
  const auto& [texture, sampler, srgb] = *slot->second.textures[entry];
  const auto image                     = m_images.find({texture, srgb});
  if(image == m_images.end() || !image->second.image)
    return std::nullopt;  // drawn with the default image until it lands: nothing of its own to average
  const shaderio::GltfTextureInfo& ti = m_textureInfos[row];
  TexelKey key{.mesh = inst.mesh, .submesh = inst.submesh, .texture = texture, .srgb = srgb, .texCoord = ti.texCoord,
               .transform = {1.f, 0.f, 0.f, 1.f, 0.f, 0.f}};
#if MAT_EXT_TEXTURE_TRANSFORM
  for(int r = 0; r < 3; r++)
  {
    key.transform[2 * r]     = ti.uvTransform[r].x;
    key.transform[2 * r + 1] = ti.uvTransform[r].y;
  }
#endif
  if(info)
    *info = ti;
  return key;
}

std::vector<float> Residency::emitterTriangleLuminance(const RetainedScene& scene, uint32_t slot) const
{
  const Instance* inst = instance(slot);
  if(!inst || !inst->shown || inst->materialSlot >= m_shadeMaterials.size())
    return {};
  const std::optional<TexelKey> key = texelKey(*inst, nullptr);
  const auto                    it  = key ? m_texelMeans.find(*key) : m_texelMeans.end();
  const MeshRecord*             record = scene.mesh(inst->mesh);
  if(it == m_texelMeans.end() || it->second.empty() || !record || it->second.size() != record->drawnRun(inst->submesh).indexCount / 3)
    return {};
  const glm::vec3    factor = glm::max(m_shadeMaterials[inst->materialSlot].emissiveFactor, glm::vec3(0.f));
  std::vector<float> lum(it->second.size());
  for(size_t t = 0; t < lum.size(); t++)
    lum[t] = nvvkgltf::emissiveLuminance(factor * it->second[t]);
  return lum;
}

bool Residency::takeTexelAverages()
{
  bool landed = false;
  for(GpuBackend::TexelAverages& a : m_gpu.takeTexelAverages())
  {
    const auto ask = m_texelAsks.find(a.ticket);
    if(ask == m_texelAsks.end())
      continue;  // asked for a scene since replaced
    size_t at = 0, black = 0;
    for(const auto& [key, count] : ask->second)
    {
      if(at + count <= a.means.size())
        m_texelMeans[key].assign(a.means.begin() + at, a.means.begin() + at + count);
      m_texelAsked.erase(key);
      at += count;
    }
    for(const glm::vec3& m : a.means)
      black += nvvkgltf::emissiveLuminance(m) <= 0.f ? 1 : 0;
    LOGI("[Lights] %zu emissive triangles of %zu textured emitters weighed by their own texels; %zu of them black\n",
         a.means.size(), ask->second.size(), black);
    m_texelAsks.erase(ask);
    landed = true;
  }
  return landed;
}

// A shown instance whose material emits
bool Residency::emits(const Instance& inst) const
{
  if(!inst.shown || inst.materialSlot >= m_shadeMaterials.size())
    return false;
  return nvvkgltf::emissiveLuminance(nvvkgltf::emitterSelectionRadiance(m_shadeMaterials[inst.materialSlot])) > 0.f;
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

// Every material slot's water, packed again whole (scene_water.hpp). A slot the shader compiled without
// the procedural layer's code doesn't read is uploaded all the same: it shows once the code comes.
void Residency::syncWater(const RetainedScene& scene)
{
  std::vector<std::pair<uint32_t, const fv::mat::WaterLayer*>> layers;
  m_waterSlots.clear();
  for(const auto& [id, slot] : m_materials)
    if(const MaterialRecord* record = scene.material(id); record && record->water)
    {
      layers.emplace_back(slot.slot, record->water.get());
      m_waterSlots.insert(slot.slot);
    }
  const WaterTables tables = layers.empty() ? WaterTables{} : packWater(m_materialHighWater, layers);
  const auto        put    = [&](BufferHandle& buffer, const auto& rows, const char* name) {
    const uint64_t bytes = std::span(rows).size_bytes();
    if(!bytes && !buffer)
      return;
    m_descDirty |= fitBuffer(m_gpu, buffer, bytes, name, [this](BufferHandle b) { retireBuffer(b); });
    if(bytes)
      m_gpu.upload(buffer, 0, std::as_bytes(std::span(rows)));
  };
  put(m_waterGrids, tables.grids, "water grids");
  put(m_waterCells, tables.cells, "water cells");
  put(m_waterRefs, tables.refs, "water refs");
  put(m_waterItems, tables.items, "water items");
  m_descDirty |= m_waterGridCount != tables.grids.size();
  m_waterGridCount = uint32_t(tables.grids.size());
  m_waterDirty     = false;
}

// The shown instances whose material emits, their triangles from their mesh records, a textured
// one's each weighed by its own texels once they are averaged: those not asked for yet are, in one
// pass recorded here (the table is built again when they come, takeTexelAverages)
void Residency::syncEmissive(const RetainedScene& scene)
{
  std::vector<nvvkgltf::Emitter>              emitters;
  std::vector<std::vector<float>>             lums;  // what each emitter's triangleLuminance views
  std::vector<shaderio::TexelTriangle>        ask;
  std::vector<std::pair<TexelKey, uint32_t>>  asked;
  lums.reserve(m_instances.size());
  for(uint32_t s = 0; s < m_instances.size(); s++)
  {
    const std::optional<Instance>& inst = m_instances[s];
    if(!inst || !inst->shown || inst->materialSlot >= m_shadeMaterials.size())
      continue;
    const glm::vec3   radiance = nvvkgltf::emitterSelectionRadiance(m_shadeMaterials[inst->materialSlot]);
    const MeshRecord* record   = scene.mesh(inst->mesh);
    if(nvvkgltf::emissiveLuminance(radiance) <= 0.f || !record)
      continue;
    const Submesh                   run     = record->drawnRun(inst->submesh);
    const std::span<const uint32_t> indices = std::span(record->indices).subspan(run.firstIndex, run.indexCount);
    lums.push_back(emitterTriangleLuminance(scene, s));
    shaderio::GltfTextureInfo info{};
    if(const std::optional<TexelKey> key = lums.back().empty() ? texelKey(*inst, &info) : std::nullopt;
       key && !m_texelMeans.count(*key) && !m_texelAsked.count(*key))
    {
      // its triangles in the texture's UV space, as the shader reads it (texCoord, then the transform)
      const std::span<const glm::vec2> uv = info.texCoord == 1 ? std::span<const glm::vec2>(record->uv1) : std::span<const glm::vec2>(record->uv0);
      const size_t                     first = ask.size();
      bool                             whole = !uv.empty();
      for(size_t i = 0; whole && i + 2 < indices.size(); i += 3)
      {
        shaderio::TexelTriangle t{.texture = uint32_t(info.index), .sampler = uint32_t(info.samplerIndex)};
        glm::vec2*              corners[3] = {&t.uv0, &t.uv1, &t.uv2};
        for(int c = 0; c < 3; c++)
        {
          if(indices[i + c] >= uv.size())
          {
            whole = false;
            break;
          }
          const glm::vec2 p = uv[indices[i + c]];
          *corners[c] = glm::vec2(p.x * key->transform[0] + p.y * key->transform[2] + key->transform[4],
                                  p.x * key->transform[1] + p.y * key->transform[3] + key->transform[5]);
        }
        ask.push_back(t);
      }
      if(whole)
      {
        asked.emplace_back(*key, uint32_t(ask.size() - first));
        m_texelAsked.insert(*key);
      }
      else
      {
        ask.resize(first);
        m_texelMeans[*key] = {};  // no UVs to average over: weighed whole
      }
    }
    emitters.push_back({.renderNodeID      = s,
                        .renderPrimID      = inst->prim,
                        .radiance          = radiance,
                        .doubleSided       = m_materialStates[inst->materialSlot]->payload.doubleSided,
                        .world             = inst->world,
                        .positions         = record->positions,
                        .indices           = indices,
                        .triangleLuminance = lums.back()});
  }
  if(!ask.empty())
  {
    if(const uint64_t ticket = m_gpu.averageTexels(ask))
      m_texelAsks[ticket] = std::move(asked);
    else
      for(const auto& [key, count] : asked)  // a backend with no such pass: weighed whole
      {
        m_texelMeans[key] = {};
        m_texelAsked.erase(key);
      }
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
    for(BufferHandle b : {mesh.positions, mesh.normals, mesh.tangents, mesh.uv0, mesh.uv1, mesh.colors, mesh.masks, mesh.joints, mesh.cageJoints, mesh.cageWeights, mesh.cageJoints1, mesh.cageWeights1, mesh.cageShares, mesh.joints1,
                          mesh.weights1, mesh.dualQuatShares, mesh.weights, mesh.positionDeltas, mesh.normalDeltas,
                          mesh.tangentDeltas, mesh.prevPositions, mesh.displacement})
      n += b ? 1 : 0;
    if(mesh.refine)
    {
      for(BufferHandle b : {mesh.refine->cage, mesh.refine->drawnFrom, mesh.refine->triangles,
                            mesh.refine->normalStart, mesh.refine->normalTriangles, mesh.refine->offsets,
                            mesh.refine->tangentTriangles, mesh.refine->tangentStart, mesh.refine->tangentCorners,
                            mesh.refine->displacement})
        n += b ? 1 : 0;
      for(const MeshGpu::Refine::Level& l : mesh.refine->levels)
        for(BufferHandle b : {l.start, l.columns, l.weights})
          n += b ? 1 : 0;
    }
  }
  for(const auto& [key, d] : m_drawers)
  {
    for(BufferHandle b : {d.positions, d.normals, d.tangents, d.morphedPositions, d.morphedNormals, d.morphedTangents, d.prevPositions})
      n += b ? 1 : 0;
  }
  for(const std::optional<Prim>& p : m_prims)
    n += p ? (p->indices ? 1 : 0) + (p->micromapIndices ? 1 : 0) : 0;
  for(BufferHandle b : {m_renderNodes, m_renderPrims, m_sceneDesc, m_materialTable, m_textureInfoTable, m_lightTable,
                        m_emissiveTable, m_emissiveGroupTable, m_emissiveLeafTable, m_iesTable, m_jointPack, m_normalPack, m_weightPack, m_dualQuatPack, m_prevPositionTable,
                        m_prevWorldTable, m_chainScratch, m_unitAxes, m_waterGrids, m_waterCells, m_waterRefs, m_waterItems})
    n += b ? 1 : 0;
  return n;
}

MemoryLedger Residency::ledger() const
{
  MemoryLedger out;
  const auto   bytes = [&](BufferHandle b) { return b ? m_gpu.bufferSize(b) : 0; };
  std::map<MeshId, size_t> entryOf;
  std::set<MeshId>         posedDrawers, otherDrawers;
  for(const auto& [key, d] : m_drawers)
    (d.posed ? posedDrawers : otherDrawers).insert(key.second);
  for(const auto& [id, mesh] : m_meshes)
  {
    entryOf[id] = out.meshes.size();
    MemoryLedger::Mesh& entry = out.meshes.emplace_back();
    entry.mesh                = id;
    entry.prop                = mesh.owner.prop;
    entry.local               = mesh.owner.local;
    GpuBytes& b               = entry.bytes;
    // the mesh's own streams: drawn where a prim draws them, what a drawer deforms from where one reads
    // them, and read by nothing where every drawer makes its streams from the posed cage
    bool drawnItself = false;
    for(int32_t p : mesh.prims)
      drawnItself |= p >= 0;
    const bool readByADrawer = otherDrawers.count(id) || !posedDrawers.count(id);
    uint64_t&  ownStreams    = drawnItself ? b.drawn : readByADrawer ? b.remake : b.unread;
    for(BufferHandle s : {mesh.positions, mesh.normals, mesh.tangents})
      ownStreams += bytes(s);
    for(BufferHandle s : {mesh.uv0, mesh.uv1, mesh.colors, mesh.masks})
      b.drawn += bytes(s);
    // what deforms it: the drawn vertices' skin is the posed cage's to do when every drawer poses
    uint64_t& drawnSkin = readByADrawer ? b.remake : b.unread;
    for(BufferHandle s : {mesh.joints, mesh.weights, mesh.joints1, mesh.weights1, mesh.dualQuatShares})
      drawnSkin += bytes(s);
    for(BufferHandle s : {mesh.cageJoints, mesh.cageWeights, mesh.cageJoints1, mesh.cageWeights1, mesh.cageShares, mesh.positionDeltas, mesh.normalDeltas, mesh.tangentDeltas})
      b.remake += bytes(s);
    b.working += bytes(mesh.prevPositions) + bytes(mesh.displacement);
    if(mesh.refine)
    {
      for(BufferHandle s : {mesh.refine->cage, mesh.refine->drawnFrom, mesh.refine->triangles, mesh.refine->normalStart, mesh.refine->normalTriangles,
                            mesh.refine->offsets, mesh.refine->tangentTriangles, mesh.refine->tangentStart, mesh.refine->tangentCorners})
        b.remake += bytes(s);
      b.working += bytes(mesh.refine->displacement);
      for(const MeshGpu::Refine::Level& l : mesh.refine->levels)
        b.remake += bytes(l.start) + bytes(l.columns) + bytes(l.weights);
    }
  }
  for(const auto& [key, d] : m_drawers)
    if(auto entry = entryOf.find(key.second); entry != entryOf.end())
    {
      GpuBytes& b = out.meshes[entry->second].bytes;
      for(BufferHandle s : {d.positions, d.normals, d.tangents})
        b.drawn += bytes(s);
      for(BufferHandle s : {d.morphedPositions, d.morphedNormals, d.morphedTangents, d.prevPositions})
        b.working += bytes(s);
    }
  for(const std::optional<Prim>& p : m_prims)
    if(p)
      if(auto entry = entryOf.find(p->mesh); entry != entryOf.end())
      {
        MemoryLedger::Mesh& m = out.meshes[entry->second];
        m.triangles += p->indexCount / 3;
        m.bytes.drawn += bytes(p->indices);
        m.bytes.bvh += bytes(p->micromapIndices) + (p->blas ? m_gpu.accelBytes(p->blas) : 0);
      }
  for(const MemoryLedger::Mesh& m : out.meshes)
    out.total += m.bytes;
  out.total.working += bytes(m_chainScratch) + bytes(m_unitAxes);
  for(const auto& [key, mm] : m_micromaps)
    out.total.bvh += m_gpu.accelBytes(mm.accel);
  for(const auto& [key, decoded] : m_decoded)
    out.total.images += decoded.image ? decoded.info.bytes : 0;
  for(const auto& [id, image] : m_images)
    out.total.images += !image.key && image.image ? image.info.bytes : 0;
  out.idleImages = m_graceBytes;
  for(const auto& [id, hits] : m_hits)
    if(std::any_of(hits.bins.begin(), hits.bins.end(), [](uint64_t b) { return b > 0; }))
      out.hits.push_back({hits.owner.prop, hits.owner.local, hits.bins});
  for(BufferHandle s : {m_renderNodes, m_renderPrims, m_sceneDesc, m_materialTable, m_textureInfoTable, m_lightTable, m_emissiveTable, m_emissiveGroupTable,
                        m_emissiveLeafTable, m_iesTable, m_jointPack, m_normalPack, m_weightPack, m_dualQuatPack, m_prevPositionTable, m_prevWorldTable,
                        m_waterGrids, m_waterCells, m_waterRefs, m_waterItems})
    out.total.scene += bytes(s);
  out.total.scene += (m_tlas ? m_gpu.accelBytes(m_tlas) : 0) + m_gpu.buildScratchBytes();
  out.retiring = *m_retiring;
  out.spilled  = m_gpu.spilledBytes();
  return out;
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
