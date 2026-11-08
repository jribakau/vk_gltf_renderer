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
// What the retained scene costs on the GPU, Vulkan-free so its tests reach it: the ledger of what is
// resident (rs::Residency::ledger), and the price of a mesh before it is made, from its counts and the
// residency's stream formats. Tests hold the price to the ledger, so a format that changes in the
// residency changes here too.
//

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "scene_retained.hpp"

namespace rs {

// GPU memory by who reads it; each byte is in one class
struct GpuBytes
{
  uint64_t drawn   = 0;  // what rays and raster read: the drawn vertex streams and indices
  uint64_t bvh     = 0;  // BLASes, opacity micromaps and their indices
  uint64_t remake  = 0;  // what makes a pose or a shape again: cages, skin inputs, refine tables, morph deltas
  uint64_t working = 0;  // only a pass's own: the refine chains' scratch, last frame's positions while moving
  uint64_t unread  = 0;  // nothing reads it: a mesh's own streams kept after what read them has gone
  uint64_t images  = 0;
  uint64_t scene   = 0;  // what the whole scene shares: its tables, the TLAS, build scratch

  uint64_t total() const { return drawn + bvh + remake + working + unread + images + scene; }
  GpuBytes& operator+=(const GpuBytes& o)
  {
    drawn += o.drawn;
    bvh += o.bvh;
    remake += o.remake;
    working += o.working;
    unread += o.unread;
    images += o.images;
    scene += o.scene;
    return *this;
  }
};

struct MemoryLedger
{
  GpuBytes total;
  // Each mesh resident, with its drawers and prims. What images and tables hold isn't any one mesh's.
  struct Mesh
  {
    MeshId   mesh;
    uint32_t prop  = 0;   // OwnerTag::prop
    int32_t  local = -1;  // OwnerTag::local
    uint64_t triangles = 0;
    GpuBytes bytes;
  };
  std::vector<Mesh> meshes;
  // What the last count of first hits saw, per material (Residency::setMaterialHits): each by the long edge it
  // could show, 256 << bin; empty: none has been counted
  struct MaterialHits
  {
    uint32_t                prop  = 0;   // OwnerTag::prop
    int32_t                 local = -1;  // OwnerTag::local
    std::array<uint64_t, 6> bins{};
  };
  std::vector<MaterialHits> hits;
  uint64_t          idleImages = 0;  // of total.images: nothing samples them (Residency's grace list)
  uint64_t          retiring   = 0;  // let go of and not in total: freed once no frame in flight reads it
  // of everything alive, what the allocator couldn't place on the device: read from system memory over
  // the bus. On an RTX 4080 that starts about at the driver's budget, an allocation then taking 125 ms
  // (scripts/memory-limit-check.py, 2026-11-01)
  uint64_t          spilled = 0;
  // the upload staging ring in system memory: its blocks, and what in them a frame in flight still reads
  uint64_t          stagingHeld = 0;
  uint64_t          stagingLive = 0;
};

// What a mesh will hold, in the counts that decide it. A refined mesh's drawn vertices are made on the
// GPU from its cage through `levels` of stencils (MeshRecord::refine).
// A host's worn items as the Fit sees them: what could be made a step coarser or finer. The host makes them,
// as new records, so all the Fit does is ask (docs/scene-memory-plan.md).
struct ItemDetail
{
  struct Down
  {
    uint64_t                              item = 0;  // the host's own
    std::string                           name;
    std::vector<std::pair<MeshId, float>> meshes;  // each with the share of what it holds it would keep
    std::vector<MaterialId>               materials;  // what rays reach of it (Residency::setMaterialHits)
    bool                                  selected = false;  // goes after the others
    bool                                  pinned   = false;  // marked Keep full: after every texture
  };
  struct Up
  {
    uint64_t                item  = 0;
    uint64_t                bytes = 0;  // what a step up would take
    std::vector<MaterialId> materials;
  };
  std::vector<Down> down;
  std::vector<Up>   up;
  bool              making = false;  // one is being made: what was asked hasn't freed anything yet
};

struct MeshCounts
{
  uint64_t vertices = 0, triangles = 0;
  bool     normals = false, tangents = false, uv0 = false, uv1 = false, colors = false;
  // deformed: its drawn streams are the drawer's
  bool     skinned = false, eightInfluences = false, dualQuatShares = false;
  uint64_t morphTargets = 0;
  bool     morphNormals = false, morphTangents = false;

  struct Level
  {
    uint64_t rows = 0, entries = 0;
  };
  std::vector<Level> levels;  // empty: not refined
  uint64_t           cageVertices = 0;
  uint64_t           refinedTriangles = 0;   // the welded normals' triangles
  bool               offsets = false;        // strand ribbons
  uint64_t           tangentTriangles = 0;   // the tangents made on the GPU, over these
  // the skin poses the cage (MeshRecord::cageJoints), and the chain runs for the drawer
  bool               posed = false, cageEightInfluences = false, cageDualQuatShares = false;

  bool deforms() const { return skinned || morphTargets > 0; }
  uint64_t refinedVertices() const { return levels.empty() ? 0 : levels.back().rows; }
};

// The buffers a mesh holds with one drawer (or none, when it doesn't deform): Residency::uploadStreams,
// makeRefine, makeDeformInputs, acquireDrawer and acquirePrim's indices, without its BLAS
inline uint64_t meshBufferBytes(const MeshCounts& m)
{
  // a normal and a tangent packed in a uint each (shaders/unit_vector_pack.h.slang); a morph's deltas are float3
  constexpr uint64_t kPosition = 12, kNormal = 4, kTangent = 4, kDelta = 12, kUv = 8, kColor = 4, kJoints = 16, kWeights = 16, kIndex3 = 12;
  const uint64_t     v = m.vertices;
  const uint64_t     outputs = kPosition + (m.normals ? kNormal : 0) + (m.tangents ? kTangent : 0);
  // the mesh's UVs and colours and indices, and its own vertices unless every drawer poses its cage
  uint64_t bytes = v * ((m.uv0 ? kUv : 0) + (m.uv1 ? kUv : 0) + (m.colors ? kColor : 0)) + m.triangles * kIndex3;
  if(!m.posed)
    bytes += v * outputs;
  // its skin and morphs
  if(m.skinned && !m.posed)
    bytes += v * ((kJoints + kWeights) * (m.eightInfluences ? 2 : 1) + (m.dualQuatShares ? 4 : 0));
  bytes += m.morphTargets * v * (kDelta + (m.morphNormals ? kDelta : 0) + (m.morphTangents ? kDelta : 0));
  // what a drawer makes
  if(m.deforms())
  {
    bytes += v * outputs;
    if(m.skinned && m.morphTargets > 0)
      bytes += v * outputs;
  }
  // refined: the cage and its tables
  if(!m.levels.empty())
  {
    const uint64_t refined = m.refinedVertices();
    bytes += m.cageVertices * kPosition;
    for(const MeshCounts::Level& l : m.levels)
      bytes += (l.rows + 1) * 4 + l.entries * 8;                       // starts, columns and weights
    bytes += v * 4;                                                    // drawn from
    bytes += m.refinedTriangles * kIndex3;                             // the triangles
    bytes += (refined + 1) * 4 + m.refinedTriangles * 3 * 4;           // each refined vertex's triangles
    if(m.offsets)
      bytes += refined * kPosition;
    if(m.tangentTriangles > 0)
      bytes += m.tangentTriangles * kIndex3 + (v + 1) * 4 + m.tangentTriangles * kIndex3;
    if(m.posed)
      bytes += m.cageVertices * ((kJoints + kWeights) * (m.cageEightInfluences ? 2 : 1) + (m.cageDualQuatShares ? 4 : 0));
  }
  return bytes;
}

// What a BLAS built for compaction holds once compacted, of what the driver says it holds as built: 98
// static BLASes of an apartment went from 65.0 to 26.5 MB (docs/improvements.md)
inline constexpr double kCompactedBlasShare = 0.41;

}  // namespace rs
