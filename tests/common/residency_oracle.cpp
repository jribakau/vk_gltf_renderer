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

#include "residency_oracle.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <cstring>
#include <set>
#include <map>
#include <tuple>

#include "gltf_material_cache.hpp"
#include "scene_gpu_tables.hpp"
#include "scene_material_bits.hpp"
#include "shaders/gltf_scene_io.h.slang"

namespace rs {

namespace {

template <class T>
bool sameBytes(std::span<const std::byte> bytes, const T* data, size_t count)
{
  return bytes.size() == count * sizeof(T) && (count == 0 || std::memcmp(bytes.data(), data, bytes.size()) == 0);
}

template <class T>
T rowAt(const FakeGpuBackend& gpu, BufferHandle table, uint32_t slot)
{
  T                             row{};
  const FakeGpuBackend::Buffer* b = gpu.buffer(table);
  if(b && uint64_t(slot + 1) * sizeof(T) <= b->bytes.size())
    std::memcpy(&row, b->bytes.data() + uint64_t(slot) * sizeof(T), sizeof(T));
  return row;
}

bool allZero(const void* p, size_t n)
{
  const auto* b = static_cast<const unsigned char*>(p);
  for(size_t i = 0; i < n; i++)
    if(b[i])
      return false;
  return true;
}

// The payload a material is drawn with: its record's, or for an edit held (D15) the one its slot has
const tinygltf::Material* drawnPayload(const RetainedScene& scene, const Residency& residency, MaterialId id)
{
  if(residency.heldMaterials().count(id) && residency.materialSlot(id) >= 0)
    if(const Residency::MaterialState* state = residency.materialState(uint32_t(residency.materialSlot(id))))
      return &state->payload;
  const MaterialRecord* m = scene.material(id);
  return m ? &m->payload : nullptr;
}

std::string at(const char* invariant, uint32_t slot)
{
  return std::string(invariant) + " slot " + std::to_string(slot) + ": ";
}

// Whether the entity draws the mesh deformed, from the scene alone
bool drawnDeformed(RetainedScene& scene, EntityId entity, const MeshRecord& r)
{
  const Entity* e = scene.entity(entity);
  if(!e || !e->mesh)
    return false;
  return !r.targets.empty()
         || (scene.skin(e->mesh->skin) && !r.positions.empty() && r.joints.size() == r.positions.size() && r.weights.size() == r.positions.size());
}

// The mesh as the entity draws it: its targets at the instance's weights (else the mesh's), then
// its skin at the joints' matrices now; arithmetic as shaders/morph.comp.slang and
// skinning.comp.slang do it
struct Deformed
{
  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec4> tangents;
};
Deformed deformedNow(RetainedScene& scene, EntityId entity, const MeshRecord& r)
{
  Deformed       out{r.positions, r.normals, r.tangents};
  const Entity&  e = *scene.entity(entity);
  const size_t   n = r.positions.size();
  const bool     hasNormals = r.normals.size() == n, hasTangents = r.tangents.size() == n;
  // a stream no target has is not morphed (its deltas are not uploaded)
  const bool normalDeltas  = std::any_of(r.targets.begin(), r.targets.end(), [](const MorphTarget& x) { return !x.normals.empty(); });
  const bool tangentDeltas = std::any_of(r.targets.begin(), r.targets.end(), [](const MorphTarget& x) { return !x.tangents.empty(); });
  if(!r.targets.empty())
  {
    const std::vector<float>& source = e.mesh->morphWeights.empty() ? r.targetWeights : e.mesh->morphWeights;
    for(size_t v = 0; v < n; v++)
    {
      glm::vec3 pos = r.positions[v];
      glm::vec3 nrm = hasNormals ? r.normals[v] : glm::vec3(0.f);
      glm::vec4 tan = hasTangents ? r.tangents[v] : glm::vec4(0.f);
      for(size_t t = 0; t < r.targets.size(); t++)
      {
        const float w = t < source.size() ? source[t] : 0.f;
        if(w == 0.f)
          continue;
        const MorphTarget& target = r.targets[t];
        pos += w * (target.positions.size() == n ? target.positions[v] : glm::vec3(0.f));
        if(hasNormals && normalDeltas)
          nrm += w * (target.normals.size() == n ? target.normals[v] : glm::vec3(0.f));
        if(hasTangents && tangentDeltas)
          tan = glm::vec4(glm::vec3(tan) + w * (target.tangents.size() == n ? target.tangents[v] : glm::vec3(0.f)), tan.w);
      }
      out.positions[v] = pos;
      if(hasNormals)
        out.normals[v] = glm::normalize(nrm);
      if(hasTangents)
        out.tangents[v] = glm::vec4(glm::normalize(glm::vec3(tan)), tan.w);
    }
  }
  const std::vector<glm::mat4> joints = scene.jointMatrices(entity);
  if(joints.empty() || r.joints.size() != n || r.weights.size() != n)
    return out;
  std::vector<glm::mat3> normalMats;
  for(const glm::mat4& j : joints)
    normalMats.push_back(glm::transpose(glm::inverse(glm::mat3(j))));
  const Deformed base = out;
  for(size_t v = 0; v < n; v++)
  {
    glm::vec3 pos(0.f), nrm(0.f), tan(0.f);
    for(int i = 0; i < 4; i++)
    {
      const int j = int(r.joints[v][i]);
      if(r.weights[v][i] > 0.f && j >= 0 && size_t(j) < joints.size())
      {
        pos += r.weights[v][i] * glm::vec3(joints[size_t(j)] * glm::vec4(base.positions[v], 1.f));
        if(hasNormals)
          nrm += r.weights[v][i] * (normalMats[size_t(j)] * base.normals[v]);
        if(hasTangents)
          tan += r.weights[v][i] * (glm::mat3(joints[size_t(j)]) * glm::vec3(base.tangents[v]));
      }
    }
    out.positions[v] = pos;
    if(hasNormals)
      out.normals[v] = glm::normalize(nrm);
    if(hasTangents)
      out.tangents[v] = glm::vec4(glm::normalize(tan), base.tangents[v].w);
  }
  return out;
}

}  // namespace

std::vector<std::string> checkResidency(RetainedScene& scene, const Residency& residency, const FakeGpuBackend& gpu)
{
  std::vector<std::string> out;
  for(const std::string& e : gpu.errors())
    out.push_back("backend: " + e);
  for(const std::string& e : gpu.unfencedForTheFrame())
    out.push_back("barriers: " + e);

  const uint32_t hwm = residency.instanceHighWater();
  // I13
  if(residency.instanceCapacity() < hwm)
    out.push_back("I13: instance capacity below the high-water mark");
  if(residency.primCapacity() < residency.primHighWater())
    out.push_back("I13: prim capacity below the high-water mark");
  if(gpu.bufferSize(residency.renderNodeTable()) < uint64_t(residency.instanceCapacity()) * sizeof(shaderio::GltfRenderNode))
    out.push_back("I13: the render-node table is smaller than its capacity");
  if(gpu.bufferSize(residency.renderPrimTable()) < uint64_t(residency.primCapacity()) * sizeof(shaderio::GltfRenderPrimitive))
    out.push_back("I13: the prim table is smaller than its capacity");
  const FakeGpuBackend::Tlas* tlas = gpu.tlas(residency.tlas());
  if(!tlas)
  {
    out.push_back("I1: no TLAS");
    return out;
  }
  if(tlas->capacity < hwm)
    out.push_back("I13: the TLAS capacity is below the high-water mark");
  if(tlas->rows.size() != hwm)
    out.push_back("I1: the TLAS was built over " + std::to_string(tlas->rows.size()) + " rows, not the high-water mark "
                  + std::to_string(hwm));

  // every draw item has exactly one instance, and no instance is not a draw item
  std::map<std::tuple<EntityId, MeshId, uint32_t, uint32_t>, uint32_t> slotOf;
  for(uint32_t s = 0; s < hwm; s++)
    if(const Residency::Instance* inst = residency.instance(s))
      if(!slotOf.emplace(std::tuple{inst->entity, inst->mesh, inst->submesh, inst->copy}, s).second)
        out.push_back(at("I1", s) + "a second instance of one draw item");
  const std::vector<DrawItem> items = scene.drawItems();
  if(items.size() != slotOf.size())
    out.push_back("I1: " + std::to_string(slotOf.size()) + " instances for " + std::to_string(items.size()) + " draw items");

  for(const DrawItem& item : items)
  {
    const auto it = slotOf.find({item.entity, item.mesh, item.submesh, item.copy});
    if(it == slotOf.end())
    {
      out.push_back("I1: a draw item has no instance");
      continue;
    }
    const uint32_t             s    = it->second;
    const Residency::Instance& inst = *residency.instance(s);
    const Residency::Prim*     prim = residency.prim(inst.prim);
    if(!prim || prim->mesh != item.mesh || prim->submesh != item.submesh)
    {
      out.push_back(at("I1", s) + "its prim is not its mesh's submesh");
      continue;
    }
    if(const MeshRecord* r = scene.mesh(item.mesh); r && prim->drawer != (drawnDeformed(scene, item.entity, *r) ? item.entity : EntityId{}))
      out.push_back(at("I1", s) + "its prim is not deformed for its entity when its mesh is, or is when it isn't");
    if(inst.world != item.world)
      out.push_back(at("I1", s) + "its world matrix is not the entity's");
    if(inst.shown != item.shown)
      out.push_back(at("I1", s) + "its shown state is not the entity's");
    if(residency.materialSlot(item.material) != int32_t(inst.materialSlot))
      out.push_back(at("I1", s) + "its material slot is not its material's");

    // TLAS row (I1, I2)
    if(s < tlas->rows.size())
    {
      const TlasRow& row = tlas->rows[s];
      if(row.blas != (item.shown ? prim->blas : AccelHandle{}))
        out.push_back(at("I1", s) + (item.shown ? "its TLAS row does not reference its prim's BLAS" : "a hidden instance's TLAS row references a BLAS"));
      for(int r = 0; r < 3; r++)
        for(int c = 0; c < 4; c++)
          if(row.transform[r][c] != item.world[c][r])
          {
            out.push_back(at("I1", s) + "its TLAS transform is not its world matrix");
            r = 3;
            break;
          }
      if(row.customIndex != inst.prim || row.customIndex >= (1u << 24))
        out.push_back(at("I2", s) + "its custom index is not its prim slot");
      if(const tinygltf::Material* payload = drawnPayload(scene, residency, item.material))
      {
        const nvvkgltf::MaterialBits bits = nvvkgltf::materialBits(*payload);
        if(row.mask != bits.instanceMask || row.flags != bits.instanceFlags)
          out.push_back(at("I1", s) + "its TLAS mask or flags are not its material's");
      }
    }

    // render-node row (I3)
    const auto node = rowAt<shaderio::GltfRenderNode>(gpu, residency.renderNodeTable(), s);
    if(node.objectToWorld != item.world || node.materialID != int(inst.materialSlot) || node.renderPrimID != int(inst.prim))
      out.push_back(at("I3", s) + "its render-node row is not the instance's");
  }

  // holes: a zeroed row and a null TLAS row
  for(uint32_t s = 0; s < hwm; s++)
  {
    if(residency.instance(s))
      continue;
    const auto node = rowAt<shaderio::GltfRenderNode>(gpu, residency.renderNodeTable(), s);
    if(!allZero(&node, sizeof(node)))
      out.push_back(at("I1", s) + "a hole's render-node row is not zeroed");
    if(s < tlas->rows.size() && (tlas->rows[s].blas || tlas->rows[s].mask))
      out.push_back(at("I1", s) + "a hole's TLAS row is not null");
  }

  // prims: their rows, bytes and BLASes (I3, I17, the build hashes), their micromap links (I21)
  std::map<const OpacityMicromap*, uint32_t> micromapUsers;
  uint32_t                                   linking = 0;
  for(uint32_t p = 0; p < residency.primHighWater(); p++)
  {
    const Residency::Prim* prim = residency.prim(p);
    if(!prim)
      continue;
    const Residency::MeshGpu* mesh   = residency.meshGpu(prim->mesh);
    const MeshRecord*         record = scene.mesh(prim->mesh);
    if(!mesh || !record)
    {
      out.push_back(at("I3", p) + "a prim of a mesh that is not resident or not in the scene");
      continue;
    }
    const Residency::DrawerGpu* drawer = prim->drawer.isNone() ? nullptr : residency.drawer(prim->drawer, prim->mesh);
    if(!prim->drawer.isNone() && !drawer)
    {
      out.push_back(at("I3", p) + "a deformed prim whose drawer is not resident");
      continue;
    }
    const BufferHandle positions = drawer ? drawer->positions : mesh->positions;
    const auto         row       = rowAt<shaderio::GltfRenderPrimitive>(gpu, residency.renderPrimTable(), p);
    if(uint64_t(row.indices) != gpu.bufferAddress(prim->indices)
       || uint64_t(row.vertexBuffer.positions) != gpu.bufferAddress(positions)
       || uint64_t(row.vertexBuffer.normals) != gpu.bufferAddress(drawer ? drawer->normals : mesh->normals)
       || uint64_t(row.vertexBuffer.tangents) != gpu.bufferAddress(drawer ? drawer->tangents : mesh->tangents)
       || uint64_t(row.vertexBuffer.texCoord0) != gpu.bufferAddress(mesh->uv0)
       || uint64_t(row.vertexBuffer.texCoord1) != gpu.bufferAddress(mesh->uv1)
       || uint64_t(row.vertexBuffer.colors) != gpu.bufferAddress(mesh->colors))
      out.push_back(at("I3", p) + "its prim row's addresses are not its buffers'");
    const Submesh& run = record->submeshes[prim->submesh];
    if(!sameBytes(gpu.bytesAt(gpu.bufferAddress(prim->indices), uint64_t(run.indexCount) * 4),
                  record->indices.data() + run.firstIndex, run.indexCount))
      out.push_back(at("I3", p) + "its index buffer does not hold its submesh's indices");
    if(!sameBytes(gpu.bytesAt(gpu.bufferAddress(mesh->positions), record->positions.size() * sizeof(glm::vec3)),
                  record->positions.data(), record->positions.size()))
      out.push_back(at("I3", p) + "its mesh's position buffer does not hold the mesh's positions");
    if(drawer)
    {
      // the passes ran on what the scene says now: the outputs hold the mesh as the entity draws it
      const Deformed want = deformedNow(scene, prim->drawer, *record);
      if(!sameBytes(gpu.bytesAt(gpu.bufferAddress(drawer->positions), want.positions.size() * sizeof(glm::vec3)),
                    want.positions.data(), want.positions.size()))
        out.push_back(at("I22", p) + "its drawer's positions are not its mesh deformed as the entity draws it now");
      if(drawer->normals
         && !sameBytes(gpu.bytesAt(gpu.bufferAddress(drawer->normals), want.normals.size() * sizeof(glm::vec3)),
                       want.normals.data(), want.normals.size()))
        out.push_back(at("I22", p) + "its drawer's normals are not its mesh deformed as the entity draws it now");
      if(drawer->tangents
         && !sameBytes(gpu.bytesAt(gpu.bufferAddress(drawer->tangents), want.tangents.size() * sizeof(glm::vec4)),
                       want.tangents.data(), want.tangents.size()))
        out.push_back(at("I22", p) + "its drawer's tangents are not its mesh deformed as the entity draws it now");
    }
    if(prim->input.vertexAddress % 4 || prim->input.indexAddress % 4)
      out.push_back(at("I17", p) + "a BLAS input address is not 4-byte aligned");
    if(prim->input.vertexAddress != gpu.bufferAddress(positions) || prim->input.indexAddress != gpu.bufferAddress(prim->indices)
       || prim->input.indexCount != run.indexCount || prim->input.vertexCount != record->positions.size())
      out.push_back(at("I3", p) + "its BLAS input is not its buffers");
    glm::vec3 lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
    for(uint32_t i = run.firstIndex; i < run.firstIndex + run.indexCount; i++)
    {
      lo = glm::min(lo, record->positions[record->indices[i]]);
      hi = glm::max(hi, record->positions[record->indices[i]]);
    }
    if(run.indexCount && (prim->boundsMin != lo || prim->boundsMax != hi))
      out.push_back(at("I3", p) + "its bounds are not its vertices'");
    const FakeGpuBackend::Blas* blas = gpu.blas(prim->blas);
    if(!blas)
      out.push_back(at("I1", p) + "its BLAS is not built");
    else if(blas->contentHash != gpu.contentHash(prim->input))
      out.push_back(at("I3", p) + "its BLAS was built from other bytes than its buffers hold");
    else if(!(blas->input == prim->input))
      out.push_back(at("I3", p) + "its BLAS was built or refit from another input than its own");

    // its submesh's micromap linked, on a device that builds them (I21)
    const MicromapUse* use = gpu.opacityMicromaps() ? run.micromap.get() : nullptr;
    if(!use)
    {
      if(prim->input.micromap || prim->micromap || prim->micromapIndices)
        out.push_back(at("I21", p) + "it links a micromap its submesh has none of");
      continue;
    }
    micromapUsers[use->micromap.get()]++;
    linking++;
    const FakeGpuBackend::Micromap* built = gpu.micromap(prim->input.micromap);
    const OpacityMicromap&          mm    = *use->micromap;
    std::vector<MicromapUsage>      usages;
    for(const OpacityMicromap::Usage& u : mm.usages)
      usages.push_back({.count = u.count, .subdivisionLevel = u.subdivisionLevel, .format = u.format});
    if(!built || prim->micromap != &mm || built->data != mm.data || built->triangles != mm.triangles
       || built->triangleStride != mm.triangleStride || built->usages != usages)
      out.push_back(at("I21", p) + "its BLAS does not link a micromap built from its submesh's");
    const uint32_t size =
        use->indexComponentType ? uint32_t(tinygltf::GetComponentSizeInBytes(uint32_t(use->indexComponentType))) : 0;
    if(prim->input.micromapBaseTriangle != use->baseTriangle || prim->input.micromapIndexSize != size)
      out.push_back(at("I21", p) + "its micromap base triangle or index size is not its submesh's");
    if(size ? prim->input.micromapIndexAddress != gpu.bufferAddress(prim->micromapIndices)
                  || !sameBytes(gpu.bytesAt(prim->input.micromapIndexAddress, use->indices.size()), use->indices.data(),
                                use->indices.size()) :
              prim->input.micromapIndexAddress || prim->micromapIndices)
      out.push_back(at("I21", p) + "its micromap index buffer does not hold its submesh's indices");
  }
  // one micromap built per record, counting the prims that link it; presence is whether any does
  for(const auto& [record, mm] : residency.micromaps())
    if(mm.record.get() != record || mm.users != micromapUsers[record] || !gpu.micromap(mm.accel))
      out.push_back("I21: a micromap's users are not the prims that link it, or it is not built");
  if(residency.micromaps().size()
     != size_t(std::count_if(micromapUsers.begin(), micromapUsers.end(), [](const auto& u) { return u.second > 0; })))
    out.push_back("I21: a micromap a prim links is not among those built");
  if(residency.hasOpacityMicromaps() != (linking > 0))
    out.push_back("I21: hasOpacityMicromaps() is not whether a prim links a micromap");

  // DLSS motion (I10): per prim slot, its previous positions' address once they have been copied, 0
  // before and for a hole; per instance slot, the world it was last drawn with, zero for a hole
  for(uint32_t p = 0; p < residency.primHighWater(); p++)
  {
    uint64_t want = 0;
    if(const Residency::Prim* prim = residency.prim(p))
    {
      if(!prim->drawer.isNone())
      {
        if(const Residency::DrawerGpu* d = residency.drawer(prim->drawer, prim->mesh); d && d->prevValid)
          want = gpu.bufferAddress(d->prevPositions);
      }
      else if(const Residency::MeshGpu* mesh = residency.meshGpu(prim->mesh); mesh && mesh->prevValid)
        want = gpu.bufferAddress(mesh->prevPositions);
    }
    if(rowAt<uint64_t>(gpu, residency.prevPositionTable(), p) != want)
      out.push_back(at("I10", p) + "its previous-positions row is not its previous positions' address");
  }
  if(gpu.bufferSize(residency.prevPositionTable()) < uint64_t(residency.primCapacity()) * sizeof(uint64_t))
    out.push_back("I13: the previous-positions table is smaller than the prim capacity");
  if(gpu.bufferSize(residency.prevWorldTable()) < uint64_t(residency.instanceCapacity()) * sizeof(glm::mat4))
    out.push_back("I13: the previous-world table is smaller than the instance capacity");
  for(uint32_t s = 0; s < hwm; s++)
  {
    const glm::mat4* prev = residency.prevWorld(s);
    if(rowAt<glm::mat4>(gpu, residency.prevWorldTable(), s) != (prev ? *prev : glm::mat4(0.f)))
      out.push_back(at("I10", s) + "its previous-world row is not the world it was last drawn with");
  }

  // the packs (I12): each drawer's joint matrices, normal matrices and weights inside their pack;
  // its streams the mesh's own count (I14: a drawer reads its mesh's streams, never a copy)
  for(const auto& [key, d] : residency.drawers())
  {
    const Residency::MeshGpu* mesh = residency.meshGpu(d.mesh);
    if(!mesh)
    {
      out.push_back("I14: a drawer of a mesh that is not resident");
      continue;
    }
    if((d.jointCount && d.jointOffset + uint64_t(d.jointCount) * sizeof(glm::mat4) > gpu.bufferSize(residency.jointPack()))
       || (d.jointCount && d.normalOffset + uint64_t(d.jointCount) * kNormalMatrixStride > gpu.bufferSize(residency.normalPack()))
       || (d.targetCount && d.weightOffset + uint64_t(d.targetCount) * sizeof(float) > gpu.bufferSize(residency.weightPack())))
      out.push_back("I12: a drawer's pack range runs past its pack");
    if(gpu.bufferSize(d.positions) != uint64_t(mesh->vertexCount) * sizeof(glm::vec3))
      out.push_back("I14: a drawer's outputs are not its mesh's vertex count");
  }

  // materials (I4): each slot's state is its record's, every texture it samples names a written
  // slot holding that texture's image; each slot's row is its state converted at its span, which
  // holds its texture infos, below the limit and overlapping no other live span (I18); a hole's row
  // is the default material
  std::vector<std::pair<uint32_t, uint32_t>> spans;  // live: start, end
  for(uint32_t m = 0; m < residency.materialHighWater(); m++)
  {
    const Residency::MaterialState*        state = residency.materialState(m);
    std::vector<shaderio::GltfTextureInfo> infos;
    nvvkgltf::TextureSlotTable             slots = state ? state->slots : nvvkgltf::TextureSlotTable{};
    if(state && state->texturesDropped)  // no room for its span: drawn without its textures
      slots.gpuIndex.assign(std::max<size_t>(slots.gpuIndex.size(), 1), -1);
    const shaderio::GltfShadeMaterial row =
        state ? nvvkgltf::MaterialCache::convertOne(state->payload, slots, state->span, infos) :
                nvvkgltf::MaterialCache::convertOne(tinygltf::Material{}, {}, 0, infos);
    constexpr size_t kRow = sizeof(shaderio::GltfShadeMaterial), kInfo = sizeof(shaderio::GltfTextureInfo);
    if(!sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.materialTable()) + m * kRow, kRow), &row, 1))
      out.push_back(at("I4", m) + "its row is not its state converted at its span");
    if(!state)
      continue;
    if(infos.size() != state->spanCount && state->spanCount)
      out.push_back(at("I18", m) + "its span is not as long as its texture infos");
    if(state->spanCount)
    {
      if(state->span == 0 || state->span + state->spanCount > residency.textureInfoLimit())
        out.push_back(at("I18", m) + "its span is not between entry 1 and the limit");
      if(!sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.textureInfoTable()) + state->span * kInfo, infos.size() * kInfo),
                    infos.data(), infos.size()))
        out.push_back(at("I4", m) + "its span does not hold its texture infos");
      for(const auto& [a, b] : spans)
        if(state->span < b && a < state->span + state->spanCount)
          out.push_back(at("I18", m) + "its span overlaps another material's");
      spans.push_back({state->span, state->span + state->spanCount});
    }
    const MaterialRecord* record = scene.material(state->material);
    if(!record)
    {
      out.push_back(at("I4", m) + "a material slot of a material not in the scene");
      continue;
    }
    if(residency.heldMaterials().count(state->material))
    {
      // held (D15): its edit needs a feature the live variant lacks
      const std::optional<nvvkgltf::SceneFeatureSet>& live = residency.shaderFeatures();
      if(!live || live->isSupersetOf(nvvkgltf::detectMaterialFeatures(record->payload)))
        out.push_back(at("I4", m) + "a held material whose edit the live variant could draw");
    }
    else if(!(state->payload == record->payload))
      out.push_back(at("I4", m) + "its row is not of the material's current payload");
    if(residency.materialSlot(state->material) != int32_t(m))
      out.push_back(at("I4", m) + "its material's slot is another");
    nvvkgltf::MaterialCache sampled;
    sampled.append(record->payload, {});
    const std::set<int> srgb = nvvkgltf::srgbTextures(record->payload);
    for(size_t t = 1; t < sampled.getTextureInfos().size(); t++)
    {
      const int i = sampled.getTextureInfos()[t].index;
      if(!record->textures || i < 0 || i >= int(record->textures->size()))
        continue;
      const TextureRef&    ref = (*record->textures)[size_t(i)];
      const GpuSamplerDesc sampler{ref.sampler.magFilter, ref.sampler.minFilter, ref.sampler.wrapS, ref.sampler.wrapT};
      const int32_t        slot = residency.textureSlot(ref.texture, sampler);
      if(slot <= 0 || state->slots.gpuTexture(i) != slot)
      {
        out.push_back(at("I4", m) + "texture " + std::to_string(i) + " does not name its texture slot");
        continue;
      }
      const auto descriptor = gpu.textureDescriptors().find(uint32_t(slot));
      if(descriptor == gpu.textureDescriptors().end() || descriptor->second != residency.image(ref.texture))
        out.push_back(at("I4", m) + "texture slot " + std::to_string(slot) + " does not hold its texture's image");
      if(state->slots.samplerSlot(i) != residency.samplerSlot(sampler)
         || !gpu.samplerDescriptors().count(uint32_t(residency.samplerSlot(sampler))))
        out.push_back(at("I4", m) + "texture " + std::to_string(i) + " does not name its sampler's written slot");
      if(const FakeGpuBackend::Image* image = gpu.image(residency.image(ref.texture)); image && srgb.count(i) && !image->srgb)
        out.push_back(at("I4", m) + "a colour texture was not decoded as sRGB");
    }
  }
  // entry 0: no texture
  const shaderio::GltfTextureInfo none{};
  if(residency.materialHighWater()
     && !sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.textureInfoTable()), sizeof(none)), &none, 1))
    out.push_back("I4: the texture-info table's entry 0 is not the empty one");

  // the raster lists: the live instances by their material's bits, in slot order
  std::array<std::vector<uint32_t>, 4> lists;
  for(uint32_t s = 0; s < hwm; s++)
    if(const Residency::Instance* inst = residency.instance(s))
      if(const tinygltf::Material* payload = drawnPayload(scene, residency, inst->material))
      {
        const nvvkgltf::MaterialBits bits = nvvkgltf::materialBits(*payload);
        if(bits.solid())
          lists[0].push_back(s);
        if(bits.solidDoubleSided())
          lists[1].push_back(s);
        if(bits.blend())
          lists[2].push_back(s);
        lists[3].push_back(s);
      }
  for(size_t l = 0; l < lists.size(); l++)
    if(residency.drawList(Residency::DrawList(l)) != lists[l])
      out.push_back("raster list " + std::to_string(l) + " is not the live instances by their materials");

  // lights: dense, in document order, a hidden one at intensity 0
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
  // EXT_lights_ies: the store's profiles' samples end to end, which the lights' iesProfile index
  std::vector<float> ies;
  for(const IesProfileRecord& profile : scene.iesProfiles())
    ies.insert(ies.end(), profile.samples.begin(), profile.samples.end());
  if(residency.iesProfileCount() != scene.iesProfiles().size()
     || !sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.iesTable()), ies.size() * sizeof(float)), ies.data(), ies.size()))
    out.push_back("lights: the IES table is not the scene's profiles");
  if(residency.lightCount() != lights.size()
     || !sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.lightTable()), lights.size() * sizeof(shaderio::GltfLight)),
                   lights.data(), lights.size()))
    out.push_back("lights: the light table is not the scene's lights in document order");

  // the emissive table (I6): the shown instances whose materials emit, by their slots, built from
  // their meshes' current positions; not while an emitter's vertex writes settle, when its areas lag
  // by design (the tests check the table once they have)
  std::vector<nvvkgltf::Emitter> emitters;
  for(uint32_t s = 0; s < hwm; s++)
  {
    const Residency::Instance* inst = residency.instance(s);
    if(!inst || !inst->shown)
      continue;
    const shaderio::GltfShadeMaterial* shade  = residency.shadeMaterial(inst->materialSlot);
    const MeshRecord*                  record = scene.mesh(inst->mesh);
    const MaterialRecord*              mat    = scene.material(inst->material);
    if(!shade || !record || !mat)
      continue;
    const glm::vec3 radiance = glm::max(shade->emissiveFactor, glm::vec3(0.f)) * shade->emissiveLightScale;
    if(nvvkgltf::emissiveLuminance(radiance) <= 0.f)
      continue;
    const Submesh& run = record->submeshes[inst->submesh];
    emitters.push_back({s, inst->prim, radiance, mat->payload.doubleSided, inst->world, record->positions,
                        std::span(record->indices).subspan(run.firstIndex, run.indexCount)});
  }
  const nvvkgltf::EmissiveTable table = nvvkgltf::buildEmissiveTable(emitters);
  if(!residency.emissiveSettling()
     && (residency.emissiveCount() != table.triangles.size() || residency.emissiveTotalWeight() != table.totalWeight
         || residency.emissiveMeanLum() != table.meanLum
         || !sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.emissiveTable()), table.triangles.size() * sizeof(shaderio::EmissiveTriangle)),
                       table.triangles.data(), table.triangles.size())))
    out.push_back("I6: the emissive table is not the shown emitters' triangles");

  // the scene descriptor
  const auto desc = rowAt<shaderio::GltfScene>(gpu, residency.sceneDescBuffer(), 0);
  if(uint64_t(desc.lights) != gpu.bufferAddress(residency.lightTable()) || desc.numLights != int(residency.lightCount())
     || uint64_t(desc.iesProfiles) != gpu.bufferAddress(residency.iesTable())
     || uint64_t(desc.emissiveTriangles) != gpu.bufferAddress(residency.emissiveTable())
     || desc.numEmissiveTriangles != int(residency.emissiveCount())
     || desc.emissiveTotalWeight != residency.emissiveTotalWeight() || desc.emissiveMeanLum != residency.emissiveMeanLum())
    out.push_back("the scene descriptor does not hold the lights and the emissive table");
  if(uint64_t(desc.renderNodes) != gpu.bufferAddress(residency.renderNodeTable())
     || uint64_t(desc.renderPrimitives) != gpu.bufferAddress(residency.renderPrimTable())
     || uint64_t(desc.materials) != gpu.bufferAddress(residency.materialTable())
     || uint64_t(desc.textureInfos) != gpu.bufferAddress(residency.textureInfoTable()))
    out.push_back("I13: the scene descriptor does not hold the tables' addresses");
  return out;
}

std::vector<std::string> residencyLeaks(const Residency& residency, const FakeGpuBackend& gpu)
{
  std::vector<std::string> out;
  if(gpu.pendingRetires())
    out.push_back("I5: " + std::to_string(gpu.pendingRetires()) + " retires still pending");
  if(gpu.liveBuffers() != residency.liveBuffers())
    out.push_back("I5: the backend holds " + std::to_string(gpu.liveBuffers()) + " buffers, the residency layer "
                  + std::to_string(residency.liveBuffers()));
  if(gpu.liveImages() != residency.liveImages())
    out.push_back("I5: the backend holds " + std::to_string(gpu.liveImages()) + " images, the residency layer "
                  + std::to_string(residency.liveImages()));
  if(gpu.liveAccels() != residency.liveAccels())
    out.push_back("I5: the backend holds " + std::to_string(gpu.liveAccels())
                  + " acceleration structures, the residency layer " + std::to_string(residency.liveAccels()));
  return out;
}

}  // namespace rs
