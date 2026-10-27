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

#pragma once

//
// The retained scene on the GPU (docs: daz-viewer docs/retained-scene-plan.md, "GPU residency",
// "The sync and the frame contract"): what a RetainedScene draws, kept resident through
// rs::GpuBackend and brought up to date once per frame from the store's changes. Vulkan-free, so
// the fake backend and ResidencyOracle (tests/common) drive it in the test build; the Vulkan
// backend binds it to the renderer.
//
// - A mesh record's vertex streams are one buffer each; a submesh drawn somewhere is a prim: its own
//   uint32 index buffer and BLAS over the mesh's streams.
// - A draw item (entity, mesh, submesh, instancing copy) is an instance: a render-node row and a TLAS
//   row, the TLAS row's custom index its prim slot.
// - Instance, prim and material slots come lowest-first from pools, so a fresh scene numbers them
//   in document order; a freed slot is a hole (a zeroed row, a null TLAS row) until its retire
//   frame, when it goes back to its pool. The tables are sized by capacity, a power of two at or
//   above the high-water mark, and the TLAS keeps its handle until capacity grows.
// - Everything freed goes through GpuBackend::retire, a frame cycle later; nothing waits for idle.
//
// Application thread only.
//

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include <glm/glm.hpp>

#include "scene_gpu_backend.hpp"
#include "scene_retained.hpp"

namespace nvutils {
class IDPool;
}

namespace rs {

struct ResidencyOptions
{
  uint32_t minInstances      = 256;           // residencyMinInstances: the least instance capacity
  uint32_t minPrims          = 256;           // and prim capacity
  uint64_t blasScratchBudget = 512ull << 20;  // blasScratchBudgetMB (the Vulkan backend's batches)
};

// What a sync did, for the renderer (see "The contract")
struct SyncResult
{
  bool changed           = false;  // any GPU op: the frame resets
  bool drawListChanged   = false;  // instances came or went
  bool buffersChanged    = false;  // a table or stream buffer was replaced
  bool visibilityChanged = false;  // an instance was shown or hidden
  bool tlasHandleChanged = false;  // the TLAS was recreated (capacity grew)
  bool verticesWritten   = false;  // a mesh's vertices were uploaded again
  bool featuresChanged   = false;
  bool dlssReset         = false;  // a scene replace
};

class Residency
{
public:
  // A mesh record's streams on the GPU
  struct MeshGpu
  {
    BufferHandle         positions, normals, tangents, uv0, uv1, colors;
    uint32_t             vertexCount = 0;
    uint32_t             contentRev  = 0;
    std::vector<int32_t> prims;  // per submesh: its prim slot, -1 while nothing draws it
  };
  // A submesh drawn: its index buffer and BLAS
  struct Prim
  {
    MeshId       mesh;
    uint32_t     submesh = 0;
    BufferHandle indices;
    uint32_t     indexCount = 0;
    AccelHandle  blas;
    BlasInput    input;
    uint32_t     users = 0;  // instances drawing it
  };
  // A draw item
  struct Instance
  {
    EntityId   entity;
    MeshId     mesh;
    uint32_t   submesh = 0;
    uint32_t   copy    = 0;
    uint32_t   prim    = 0;
    MaterialId material;
    uint32_t   materialSlot = 0;
    glm::mat4  world        = glm::mat4(1.f);
    bool       shown        = true;
  };

  explicit Residency(GpuBackend& gpu, ResidencyOptions options = {});
  ~Residency();
  Residency(const Residency&)            = delete;
  Residency& operator=(const Residency&) = delete;

  // A scene replace: what was resident is retired, and `scene` made resident whole (its pending
  // changes are taken).
  SyncResult replace(RetainedScene& scene);
  // This frame's sync: the store's changes since the last one, brought to the GPU.
  SyncResult sync(RetainedScene& scene);
  // Everything retired (the backend frees it a frame cycle later); the pools start over.
  void clear();

  // ---- what the renderers read
  uint64_t    sceneDesc() const;  // GltfScene's address
  AccelHandle tlas() const { return m_tlas; }
  uint32_t    instanceCapacity() const { return m_instanceCapacity; }
  uint32_t    instanceHighWater() const { return uint32_t(m_instances.size()); }
  uint32_t    primCapacity() const { return m_primCapacity; }

  // ---- what the oracle reads
  const Instance* instance(uint32_t slot) const;  // nullptr for a hole
  const Prim*     prim(uint32_t slot) const;
  uint32_t        primHighWater() const { return uint32_t(m_prims.size()); }
  const MeshGpu*  meshGpu(MeshId mesh) const;
  BufferHandle    renderNodeTable() const { return m_renderNodes; }
  BufferHandle    renderPrimTable() const { return m_renderPrims; }
  BufferHandle    sceneDescBuffer() const { return m_sceneDesc; }
  // the material slot a material has, -1 while nothing draws it
  int32_t materialSlot(MaterialId material) const;
  // what live resources this layer holds (meshes' streams, prims' index buffers and BLASes, the
  // tables, the TLAS), to tell a leak from what the backend still has to free
  size_t liveBuffers() const;
  size_t liveAccels() const;

private:
  using InstanceKey = std::tuple<EntityId, MeshId, uint32_t, uint32_t>;  // entity, mesh, submesh, copy

  struct MaterialSlot
  {
    uint32_t slot  = 0;
    uint32_t users = 0;
  };

  void syncStructure(RetainedScene& scene, std::vector<DrawItem>& items, SyncResult& result);
  void syncRows(const std::vector<DrawItem>& items, SyncResult& result);
  void syncVertices(RetainedScene& scene, const std::set<MeshId>& meshes, SyncResult& result);

  MeshGpu& acquireMesh(const RetainedScene& scene, MeshId id);
  uint32_t acquirePrim(const RetainedScene& scene, MeshId id, uint32_t submesh);
  void     releasePrim(uint32_t slot);
  uint32_t acquireMaterial(MaterialId id);
  void     releaseMaterial(MaterialId id);
  void     retireMesh(MeshId id);
  void     retireBuffer(BufferHandle buffer);
  void     retireAccel(AccelHandle accel);
  void     uploadStreams(const MeshRecord& record, MeshGpu& gpu);

  void ensureCapacity(SyncResult& result);
  void writeRenderNodes(const std::set<uint32_t>& slots);
  void writeRenderPrims(const std::set<uint32_t>& slots);
  void buildTlas();
  void writeSceneDesc();

  GpuBackend&      m_gpu;
  ResidencyOptions m_options;

  // shared with the retire closures that give slots back, which may outlive a clear()
  std::shared_ptr<nvutils::IDPool> m_instancePool, m_primPool, m_materialPool;

  std::map<MeshId, MeshGpu>            m_meshes;
  std::vector<std::optional<Prim>>     m_prims;      // by prim slot, up to the high-water mark
  std::vector<std::optional<Instance>> m_instances;  // by instance slot, up to the high-water mark
  std::map<InstanceKey, uint32_t>      m_slotOf;
  std::map<MaterialId, MaterialSlot>   m_materials;

  uint32_t     m_instanceCapacity = 0;
  uint32_t     m_primCapacity     = 0;
  BufferHandle m_renderNodes, m_renderPrims, m_sceneDesc;
  AccelHandle  m_tlas;
  uint32_t     m_tlasCapacity = 0;

  std::set<uint32_t> m_dirtyNodes;        // render-node rows to write: rebound or now holes
  bool               m_allNodes = false;  // every row, into a new table
  bool               m_allPrims = false;
};

}  // namespace rs
