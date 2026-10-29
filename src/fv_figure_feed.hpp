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
// The figure feed (daz-viewer docs/retained-scene-plan.md, P2): a figure's scene graph straight into
// the retained scene, with no glTF in between. It holds what fv::build::buildGltf would write,
// record for record: one entity per node traversal reaches, one mesh record per scene mesh (a DAZ
// instance's nodes share it), one material record per scene material, so `m:<i>` stays put, and the
// figure's textures in the order the materials first ask for them.
//
// It diffs against what it has fed. The graph only grows, so a node past the fed ones is new and a
// node traversal no longer reaches was taken off; a mesh whose geometry or hidden polygons changed
// gets a new record; a material whose lowered payload changed is set again. Vertices moved in place
// (a dial, a fit, dForce) are not visible in the graph, so the caller names those meshes.
//
// Materials are lowered with fv::build::lowering, as the build lowers them. Vulkan-free; on the
// retained scene's owner thread.
//

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <figure/gltf/gltf_build.hpp>

#include "scene_retained.hpp"
#include "tinygltf_utils.hpp"

namespace fv {

struct FigureFeedOptions
{
  // what the figure is lowered with, as a build would take it (textures, UV flip, skin, exposure, walk-through)
  build::BuildOptions build;
  // multiplied into the figure root's scale (the host's centimetres)
  double rootScale = 1;
  // owner tags: {prop, scene index}
  uint32_t prop = 0;
  // where the figure root goes; none: a root of the retained scene
  rs::EntityId parent;
  // a subdivided mesh's cage as its morphs have moved it (MorphSet::morphedCage), xyz per cage
  // vertex: given, the feed sends it with the mesh's vertices and the GPU makes the drawn positions
  // and normals from it (figureGpuRefine, daz-viewer docs/gpu-figures-plan.md F1)
  std::function<std::vector<float>(int mesh)> cageOf;
  // DAZ's skinning (figureDazSkinning, daz-viewer docs/gpu-figures-plan.md F3): records carry up to
  // eight influences and the file's dual quaternion share; otherwise glTF's four, linear
  bool dazSkinning = false;
  // DAZ's order (figureDazOrder, gpu-figures-plan.md F4): a refined mesh's drawers skin its cage and
  // refine the posed cage; records carry the cage's influences
  bool dazOrder = false;
};

class FigureFeed
{
public:
  FigureFeed(rs::RetainedScene& scene, FigureFeedOptions options);
  // the figure's entities go, and every record the feed holds is released
  ~FigureFeed();
  FigureFeed(const FigureFeed&)            = delete;
  FigureFeed& operator=(const FigureFeed&) = delete;

  // Brings the retained scene up to `figure` below `root`. A mesh it keeps is compared with the
  // scene's vertices, and what moved is fed again with the tangents near it.
  void sync(const SceneGraph& figure, int root);
  // `options.build` changed (a texture came, the skin mode or exposure moved): the next sync
  // lowers with these (the narrow syncMaterials waits for it)
  void setOptions(const build::BuildOptions& options)
  {
    m_options.build = options;
    m_lowering.clear();
  }

  // Narrow syncs, for what changes every frame, without the lowering of every material a sync does
  // (about 10 ms on a dressed figure). Each leaves the store as a sync would for that change alone,
  // and returns false when a sync is still needed: something it was not fed with has changed (a new
  // node or mesh, a mesh's geometry or hidden polygons, a material's need for tangents), or nothing
  // has been synced since setOptions. What it could do is done either way.
  // `nodes`' local transforms and visibility (a pose step, an item shown or hidden)
  bool syncTransforms(const SceneGraph& figure, int root, std::span<const int> nodes);
  // `meshes`' vertices, moved in place (a dial, a fit, a drape), and the tangents near them.
  // `tangentsLater`: a drag's step, which keeps the tangents it had for settleTangents to make
  bool syncVertices(const SceneGraph& figure, std::span<const int> meshes, bool tangentsLater = false);
  // `meshes`' morphed cages alone (MorphSet::cageOnly, a drag's step): the GPU refines them; a mesh
  // the feed doesn't refine is fed as syncVertices feeds it
  bool syncCages(const SceneGraph& figure, std::span<const int> meshes);
  // whether the GPU makes the mesh's drawn vertices from its cage (FigureFeedOptions::cageOf)
  bool refines(int mesh) const;
  // the tangents syncVertices left for later, made for each mesh whose vertices have not moved
  // for `still` (a drag that stopped)
  void settleTangents(std::chrono::steady_clock::duration still = {});
  // `meshes`' skins: their bones' entities and inverse binds (a rebind)
  bool syncSkins(const SceneGraph& figure, std::span<const int> meshes);
  // `materials` lowered again (an edit, a texture that came), with the options the last sync lowered with
  bool syncMaterials(const SceneGraph& figure, std::span<const int> materials);
  // ... or with these (a host's own: the skin it shades with, the tone reference of an edit)
  bool syncMaterials(const SceneGraph& figure, std::span<const std::pair<int, mat::LowerOptions>> materials);
  // per scene material, the options the last sync lowered it with (empty until one has)
  const std::vector<mat::LowerOptions>& lowering() const { return m_lowering; }
  // `requests`' images changed (a cook that landed for a source image): each takes its texture's
  // place in every table that holds it, so the materials keep their indices and the old image goes.
  // Their rows may run the other way now, which a material's lowering says (its slot's transform):
  // the materials that sample them are the caller's to lower again (syncMaterials, or the host's
  // material edits). A request never fed is left to the next lowering. False when none was replaced.
  bool retexture(std::span<const mat::TextureRequest> requests);

  rs::EntityId   entityOf(int node) const;
  rs::MeshId     meshOf(int mesh) const;
  rs::MaterialId materialOf(int material) const;
  rs::SkinId     skinOf(int mesh) const;
  // the figure's texture list, in the order the materials first asked for each
  const std::shared_ptr<const rs::TextureTable>& textures() const { return m_table; }

private:
  // what a mesh record was made from: a new record when any of it changes
  struct FedMesh
  {
    rs::MeshId                                  id;
    std::weak_ptr<mesh::BuiltMesh>              geo;
    std::map<std::string, std::vector<int32_t>> hidden;  // the geometry's hidden polygons
    bool                                        flipV = false;
    std::vector<bool>                           tangentGroups;  // the drawn groups whose materials need tangents
    // the drawn groups' runs (start, count in indices) and scene materials, and the group each is
    // merged into (Submesh::mergedInto; mergePlan)
    std::vector<std::pair<uint32_t, uint32_t>> groupRuns;
    std::vector<int>                           groupMaterials;
    std::vector<uint32_t>                      merged;
    // the triangles its tangents are made over (none: it has none), and which touch each vertex,
    // so moved vertices make only the tangents near them again
    std::vector<uint32_t>                            tangentIndices;
    std::optional<tinygltf::utils::TangentAdjacency> tangentAdjacency;
    rs::SkinId                                       skin;
    // how its drawn vertices come from its cage, when the GPU refines it (FigureFeedOptions::cageOf)
    std::shared_ptr<const rs::RefineTables> refine;
    // its tangents are from before its vertices last moved, at `movedAt`
    bool                                  tangentsLate = false;
    std::chrono::steady_clock::time_point movedAt;
  };
  struct FedTexture
  {
    uint32_t slot      = 0;  // its place in m_table
    bool     flipGreen = false;
    bool     flipRows  = false;  // its rows run the other way than the options' texturesFlipped
  };

  void syncMaterials(const SceneGraph& figure, const build::Drawing& drawing);
  // a texture's slot in the table, added to `grown` (a copy of m_table) when it comes now; `missing`
  // holds the keys asked this sync that resolved to nothing
  mat::ResolvedTexture resolveTexture(const mat::TextureRequest&         req,
                                      std::shared_ptr<rs::TextureTable>& grown,
                                      std::set<std::string>&             missing);
  rs::Transform        transformOf(const SceneGraph& figure, int node, int root) const;
  // the mesh's vertices fed again into its record, the tangents remade near the ones that moved
  void moveVertices(const Mesh& m, FedMesh& f, bool tangentsLater = false);
  // whether the GPU makes the mesh's tangents with its refined vertices (RefineTables::tangentIndices)
  static bool gpuTangents(const FedMesh& f);
  // the mesh's morphed cage, when the GPU refines it; empty otherwise
  std::vector<glm::vec3> cageOf(int mesh, const FedMesh& f) const;
  // the skin the mesh draws with, if it is skinned
  std::optional<rs::SkinRecord> skinRecord(const Mesh& m) const;
  // whether the mesh is fed with its current geometry and hidden polygons
  bool fedAsItIs(const SceneGraph& figure, int mesh) const;
  void syncMeshes(const SceneGraph& figure, const build::Drawing& drawing);
  void syncNodes(const SceneGraph& figure, int root, const build::Drawing& drawing);
  // the scene material of each of a mesh's drawn groups (-1: none), and the store's material for it
  std::vector<int> groupMaterials(const SceneGraph& figure, int mesh, const build::DrawnView& drawn) const;
  rs::MaterialId   materialFor(int material);
  std::vector<bool>      tangentGroups(const std::vector<int>& materials) const;
  // Submesh::mergedInto for each drawn group: a group that follows another in the indices, both
  // drawing something, with a material the same as that group's in all but its name, is drawn and
  // traced with it (one prim, one BLAS)
  std::vector<uint32_t>  mergePlan(const std::vector<std::pair<uint32_t, uint32_t>>& runs, const std::vector<int>& materials) const;
  rs::MeshRecord         meshRecord(const SceneGraph& figure, int mesh, const build::DrawnView& drawn, const std::vector<uint32_t>& merged) const;
  std::vector<uint32_t>  tangentIndices(const rs::MeshRecord& record, const std::vector<bool>& groups) const;

  rs::RetainedScene& m_scene;
  FigureFeedOptions  m_options;

  std::vector<rs::EntityId>   m_entities;   // per scene node
  std::vector<FedMesh>        m_meshes;     // per scene mesh
  std::vector<rs::MaterialId> m_materials;  // per scene material
  std::vector<mat::LowerOptions> m_lowering;   // per scene material, what the last sync lowered with
  rs::MaterialId              m_default;    // drawn by a group with no material

  // request key -> its texture; a key that resolved to nothing is asked again next sync
  std::map<std::string, FedTexture>       m_textureOf;
  std::vector<rs::TextureId>              m_textureIds;  // m_table's textures, held by the feed
  std::shared_ptr<const rs::TextureTable> m_table;
};

}  // namespace fv
