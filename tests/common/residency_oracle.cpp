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

std::string at(const char* invariant, uint32_t slot)
{
  return std::string(invariant) + " slot " + std::to_string(slot) + ": ";
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
      if(const MaterialRecord* m = scene.material(item.material))
      {
        const nvvkgltf::MaterialBits bits = nvvkgltf::materialBits(m->payload);
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

  // prims: their rows, bytes and BLASes (I3, I17, the build hashes)
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
    const auto row = rowAt<shaderio::GltfRenderPrimitive>(gpu, residency.renderPrimTable(), p);
    if(uint64_t(row.indices) != gpu.bufferAddress(prim->indices)
       || uint64_t(row.vertexBuffer.positions) != gpu.bufferAddress(mesh->positions)
       || uint64_t(row.vertexBuffer.normals) != gpu.bufferAddress(mesh->normals)
       || uint64_t(row.vertexBuffer.tangents) != gpu.bufferAddress(mesh->tangents)
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
    if(prim->input.vertexAddress % 4 || prim->input.indexAddress % 4)
      out.push_back(at("I17", p) + "a BLAS input address is not 4-byte aligned");
    if(prim->input.vertexAddress != gpu.bufferAddress(mesh->positions) || prim->input.indexAddress != gpu.bufferAddress(prim->indices)
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
  }

  // materials (I4): each slot's state is its record's, every texture it samples names a written
  // slot holding that texture's image, and the tables hold the states converted in slot order
  nvvkgltf::MaterialCache expected;
  expected.buildFromMaterials({}, {});
  for(uint32_t m = 0; m < residency.materialHighWater(); m++)
  {
    const Residency::MaterialState* state = residency.materialState(m);
    if(!state)
    {
      expected.append(tinygltf::Material{}, {});
      continue;
    }
    expected.append(state->payload, state->slots);
    const MaterialRecord* record = scene.material(state->material);
    if(!record)
    {
      out.push_back(at("I4", m) + "a material slot of a material not in the scene");
      continue;
    }
    if(!(state->payload == record->payload))
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
  const auto& shade = expected.getShadeMaterials();
  const auto& infos = expected.getTextureInfos();
  if(!sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.materialTable()), shade.size() * sizeof(shade[0])),
                shade.data(), shade.size()))
    out.push_back("I4: the material table does not hold the materials converted");
  if(!sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.textureInfoTable()), infos.size() * sizeof(infos[0])),
                infos.data(), infos.size()))
    out.push_back("I4: the texture-info table does not hold the materials' texture infos");

  // the raster lists: the live instances by their material's bits, in slot order
  std::array<std::vector<uint32_t>, 4> lists;
  for(uint32_t s = 0; s < hwm; s++)
    if(const Residency::Instance* inst = residency.instance(s))
      if(const MaterialRecord* m = scene.material(inst->material))
      {
        const nvvkgltf::MaterialBits bits = nvvkgltf::materialBits(m->payload);
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
    shaderio::GltfLight light  = nvvkgltf::packLight(record ? &record->payload : nullptr, item.world);
    if(!item.shown)
      light.intensity = 0.f;
    lights.push_back(light);
  }
  if(residency.lightCount() != lights.size()
     || !sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.lightTable()), lights.size() * sizeof(shaderio::GltfLight)),
                   lights.data(), lights.size()))
    out.push_back("lights: the light table is not the scene's lights in document order");

  // the emissive table (I6): the shown instances whose materials emit, by their slots, built from
  // their meshes' current positions
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
  if(residency.emissiveCount() != table.triangles.size() || residency.emissiveTotalWeight() != table.totalWeight
     || residency.emissiveMeanLum() != table.meanLum
     || !sameBytes(gpu.bytesAt(gpu.bufferAddress(residency.emissiveTable()), table.triangles.size() * sizeof(shaderio::EmissiveTriangle)),
                   table.triangles.data(), table.triangles.size()))
    out.push_back("I6: the emissive table is not the shown emitters' triangles");

  // the scene descriptor
  const auto desc = rowAt<shaderio::GltfScene>(gpu, residency.sceneDescBuffer(), 0);
  if(uint64_t(desc.lights) != gpu.bufferAddress(residency.lightTable()) || desc.numLights != int(residency.lightCount())
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
  if(gpu.liveAccels() != residency.liveAccels())
    out.push_back("I5: the backend holds " + std::to_string(gpu.liveAccels())
                  + " acceleration structures, the residency layer " + std::to_string(residency.liveAccels()));
  return out;
}

}  // namespace rs
