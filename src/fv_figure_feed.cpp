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

#include "fv_figure_feed.hpp"

#include <algorithm>
#include <cstring>
#include <set>

#include <figure/dforce/skin.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "scene_import_gltf.hpp"
#include "tinygltf_utils.hpp"

namespace fv {

namespace {

rs::SamplerDesc figureSamplerDesc()
{
  const tinygltf::Sampler s = build::figureSampler();
  return {.magFilter = s.magFilter, .minFilter = s.minFilter, .wrapS = s.wrapS, .wrapT = s.wrapT};
}

// What a mesh node draws: nothing when the mesh has no geometry, no vertices or no groups (as the
// build leaves its node without a mesh).
bool draws(const SceneGraph& figure, const Node& n)
{
  if(n.kind != NodeKind::eMesh || n.mesh < 0 || n.mesh >= int(figure.meshes.size()))
    return false;
  const Mesh& m = figure.meshes[size_t(n.mesh)];
  return m.geo && !m.geo->position.empty() && !m.geo->groups.empty();
}

// How a subdivided mesh's drawn vertices come from its cage: its stencils, its splits, and per
// refined vertex the triangles NormalWelder sums for it, in triangle order
std::shared_ptr<const rs::RefineTables> refineTables(const mesh::BuiltMesh& geo)
{
  auto t       = std::make_shared<rs::RefineTables>();
  t->cageCount = uint32_t(geo.cagePositions.size() / 3);
  for(const subdiv::Stencils& s : geo.refinement->levels)
  {
    rs::RefineTables::Level& l = t->levels.emplace_back();
    l.start.assign(s.rowStart.begin(), s.rowStart.end());
    l.columns.assign(s.col.begin(), s.col.end());
    l.weights = s.weight;
  }
  t->drawnFrom   = geo.renderIndex;
  t->triangles   = geo.renderTris;
  const size_t n = size_t(std::max(geo.renderVertexCount, 0));
  t->normalStart.assign(n + 1, 0);
  for(uint32_t v : geo.renderTris)
    if(v < n)
      t->normalStart[v + 1]++;
  for(size_t v = 0; v < n; v++)
    t->normalStart[v + 1] += t->normalStart[v];
  std::vector<uint32_t> fill(t->normalStart.begin(), t->normalStart.end() - 1);
  t->normalTriangles.assign(t->normalStart[n], 0);
  for(size_t i = 0; i < geo.renderTris.size(); i++)
    if(geo.renderTris[i] < n)
      t->normalTriangles[fill[geo.renderTris[i]]++] = uint32_t(i / 3);
  // strand ribbons: each refined vertex is its strand point, moved off its guide when grown from one,
  // then across the strand by its width
  if(geo.ribbon)
  {
    const mesh::Ribbon& r = *geo.ribbon;
    t->offsets.assign(n, glm::vec3(0.f));
    for(size_t v = 0; v < n && v < geo.firstSplit.size(); v++)
      if(const int32_t d = geo.firstSplit[v]; d >= 0 && size_t(d) < r.half.size())
      {
        t->offsets[v] = glm::make_vec3(&r.side[size_t(d) * 3]) * r.half[size_t(d)];
        if(size_t(d) * 3 + 2 < r.shift.size())
          t->offsets[v] += glm::make_vec3(&r.shift[size_t(d) * 3]);
      }
  }
  return t;
}

bool sameInstance(const std::optional<rs::MeshInstance>& a, const std::optional<rs::MeshInstance>& b)
{
  if(a.has_value() != b.has_value())
    return false;
  if(!a)
    return true;
  if(a->skin != b->skin || a->morphWeights != b->morphWeights || a->copies != b->copies || a->parts.size() != b->parts.size())
    return false;
  for(size_t p = 0; p < a->parts.size(); p++)
    if(a->parts[p].mesh != b->parts[p].mesh || a->parts[p].materials != b->parts[p].materials)
      return false;
  return true;
}

// A figure texture's record: the file's path, as an import's record holds it. The image's URI is
// %-encoded, as glTF has it (a folder named "Colm Jackson" is "Colm%20Jackson" there, and opens as nothing)
rs::TextureRecord textureRecord(const mat::TextureRequest& req, const build::TextureImage& image)
{
  rs::TextureRecord t;
  t.name     = req.rel.substr(req.rel.find_last_of('/') + 1);
  t.uri      = tinygltf::utils::decodePathFromUri(image.uri);
  t.mimeType = image.mimeType;
  if(image.mimeType == "image/ktx2")
    t.sourceExtension = "KHR_texture_basisu";
  return t;
}

}  // namespace

FigureFeed::FigureFeed(rs::RetainedScene& scene, FigureFeedOptions options)
    : m_scene(scene)
    , m_options(std::move(options))
    , m_table(std::make_shared<rs::TextureTable>())
{
}

FigureFeed::~FigureFeed()
{
  for(rs::EntityId e : m_entities)
    if(m_scene.entity(e))
      m_scene.destroyEntity(e);
  for(const FedMesh& f : m_meshes)
  {
    m_scene.release(f.id);
    m_scene.release(f.skin);
  }
  for(rs::MaterialId m : m_materials)
    m_scene.release(m);
  m_scene.release(m_default);
  for(rs::TextureId t : m_textureIds)
    m_scene.release(t);
}

rs::EntityId FigureFeed::entityOf(int node) const
{
  return node >= 0 && node < int(m_entities.size()) ? m_entities[size_t(node)] : rs::EntityId{};
}

rs::MeshId FigureFeed::meshOf(int mesh) const
{
  return mesh >= 0 && mesh < int(m_meshes.size()) ? m_meshes[size_t(mesh)].id : rs::MeshId{};
}

rs::MaterialId FigureFeed::materialOf(int material) const
{
  return material >= 0 && material < int(m_materials.size()) ? m_materials[size_t(material)] : rs::MaterialId{};
}

rs::SkinId FigureFeed::skinOf(int mesh) const
{
  return mesh >= 0 && mesh < int(m_meshes.size()) ? m_meshes[size_t(mesh)].skin : rs::SkinId{};
}

void FigureFeed::sync(const SceneGraph& figure, int root)
{
  if(root < 0 || root >= int(figure.nodes.size()))
    return;
  const build::Drawing drawing = build::drawing(figure, root);
  // the materials first (a mesh's tangents follow its materials' normal maps), then the entities
  // the skins name, then the meshes and what each node draws
  syncMaterials(figure, drawing);
  syncNodes(figure, root, drawing);
  syncMeshes(figure, drawing);
}

//--------------------------------------------------------------------------------------------------
// Materials, and the textures they ask for

void FigureFeed::syncMaterials(const SceneGraph& figure, const build::Drawing& drawing)
{
  m_lowering = build::lowering(figure, drawing, m_options.build);

  // a texture that comes grows a copy of the table; the materials set from here on take it, the
  // others keep the one they index (its first entries are the same)
  std::shared_ptr<rs::TextureTable> grown;
  std::set<std::string>             missing;
  const auto resolve = [&](const mat::TextureRequest& req) { return resolveTexture(req, grown, missing); };

  std::vector<tinygltf::Material> payloads(figure.materials.size());
  std::vector<std::string>        extensionsUsed;  // the export works them out from the payloads
  for(size_t mi = 0; mi < figure.materials.size(); mi++)
    payloads[mi] = build::figureMaterial(figure, int(mi), m_lowering[mi], resolve, extensionsUsed, m_options.build);
  if(grown)
    m_table = std::move(grown);

  m_materials.resize(figure.materials.size());
  for(size_t mi = 0; mi < figure.materials.size(); mi++)
  {
    rs::MaterialId& id = m_materials[mi];
    if(id.isNone())
      id = m_scene.addMaterial({.owner = {m_options.prop, int32_t(mi)}, .payload = std::move(payloads[mi]), .textures = m_table});
    else if(const rs::MaterialRecord* r = m_scene.material(id); r && !(r->payload == payloads[mi]))
      m_scene.setMaterial(id, std::move(payloads[mi]), m_table);
  }
}

mat::ResolvedTexture FigureFeed::resolveTexture(const mat::TextureRequest&         req,
                                                std::shared_ptr<rs::TextureTable>& grown,
                                                std::set<std::string>&             missing)
{
  if(req.rel.empty() || !m_options.build.texture)
    return {};
  const std::string key = mat::requestKey(req);
  if(auto it = m_textureOf.find(key); it != m_textureOf.end())
    return {int(it->second.slot), it->second.flipGreen, it->second.flipRows};
  if(missing.count(key))
    return {};
  const auto image = m_options.build.texture(req);
  if(!image)
  {
    missing.insert(key);
    return {};
  }
  const rs::TextureId id = m_scene.addTexture(textureRecord(req, *image));
  m_textureIds.push_back(id);
  if(!grown)
    grown = std::make_shared<rs::TextureTable>(*m_table);
  grown->push_back({id, figureSamplerDesc()});
  const FedTexture fed{uint32_t(grown->size() - 1), image->flipGreen, image->flipped != m_options.build.texturesFlipped};
  m_textureOf[key] = fed;
  return {int(fed.slot), fed.flipGreen, fed.flipRows};
}

bool FigureFeed::retexture(std::span<const mat::TextureRequest> requests)
{
  if(!m_options.build.texture)
    return false;
  std::vector<std::pair<rs::TextureId, rs::TextureId>> swapped;  // the old texture, the new
  for(const mat::TextureRequest& req : requests)
  {
    const auto it = m_textureOf.find(mat::requestKey(req));
    if(it == m_textureOf.end() || it->second.slot >= m_table->size())
      continue;
    const auto image = m_options.build.texture(req);
    if(!image)
      continue;
    swapped.emplace_back((*m_table)[it->second.slot].texture, m_scene.addTexture(textureRecord(req, *image)));
    it->second.flipGreen = image->flipGreen;
    it->second.flipRows  = image->flipped != m_options.build.texturesFlipped;
  }
  if(swapped.empty())
    return false;
  // every table that holds one, copied once with it replaced (the tables are the feed's growing one
  // and the shorter ones older materials still index)
  std::map<const rs::TextureTable*, std::shared_ptr<const rs::TextureTable>> copies;
  const auto copyOf = [&](const std::shared_ptr<const rs::TextureTable>& table) -> std::shared_ptr<const rs::TextureTable> {
    if(!table)
      return table;
    auto& copy = copies[table.get()];
    if(!copy)
    {
      auto out = std::make_shared<rs::TextureTable>(*table);
      for(rs::TextureRef& ref : *out)
        for(const auto& [was, now] : swapped)
          if(ref.texture == was)
            ref.texture = now;
      copy = std::move(out);
    }
    return copy;
  };
  const auto holds = [&](const rs::TextureTable& table) {
    return std::any_of(table.begin(), table.end(), [&](const rs::TextureRef& ref) {
      return std::any_of(swapped.begin(), swapped.end(), [&](const auto& s) { return s.first == ref.texture; });
    });
  };
  m_table = copyOf(m_table);
  std::vector<rs::MaterialId> materials = m_materials;
  materials.push_back(m_default);
  for(rs::MaterialId id : materials)
    if(const rs::MaterialRecord* r = m_scene.material(id); r && r->textures && holds(*r->textures))
      m_scene.setMaterial(id, r->payload, copyOf(r->textures));
  // the feed's own hold moves to the new texture; the old goes with the last table that held it
  for(const auto& [was, now] : swapped)
  {
    std::replace(m_textureIds.begin(), m_textureIds.end(), was, now);
    m_scene.release(was);
  }
  return true;
}

bool FigureFeed::syncMaterials(const SceneGraph& figure, std::span<const int> materials)
{
  if(m_lowering.size() != figure.materials.size())
    return false;
  std::vector<std::pair<int, mat::LowerOptions>> with;
  for(int mi : materials)
    with.emplace_back(mi, mi >= 0 && mi < int(m_lowering.size()) ? m_lowering[size_t(mi)] : mat::LowerOptions{});
  return syncMaterials(figure, with);
}

bool FigureFeed::syncMaterials(const SceneGraph& figure, std::span<const std::pair<int, mat::LowerOptions>> materials)
{
  if(m_lowering.size() != figure.materials.size())
    return false;
  std::shared_ptr<rs::TextureTable> grown;
  std::set<std::string>             missing;
  const auto resolve = [&](const mat::TextureRequest& req) { return resolveTexture(req, grown, missing); };
  std::vector<std::pair<int, tinygltf::Material>> payloads;
  std::vector<std::string>                        extensionsUsed;
  bool                                            all = true;
  for(const auto& [mi, options] : materials)
  {
    if(mi < 0 || mi >= int(m_materials.size()) || m_materials[size_t(mi)].isNone())
    {
      all = false;
      continue;
    }
    payloads.emplace_back(mi, build::figureMaterial(figure, mi, options, resolve, extensionsUsed, m_options.build));
  }
  if(grown)
    m_table = std::move(grown);
  for(auto& [mi, payload] : payloads)
  {
    const rs::MaterialId      id = m_materials[size_t(mi)];
    const rs::MaterialRecord* r  = m_scene.material(id);
    if(!r || r->payload == payload)
      continue;
    // a normal map that comes or goes changes which of the meshes using it carry tangents: their
    // records are a sync's to make again
    if(rs::materialNeedsTangents(r->payload) != rs::materialNeedsTangents(payload))
      all = false;
    m_scene.setMaterial(id, std::move(payload), m_table);
  }
  // so are those whose groups merge differently now (mergePlan): an edit that sets one of a hair's
  // groups apart, or makes it the same as its neighbour again
  if(all && !payloads.empty())
    for(const FedMesh& f : m_meshes)
    {
      const bool edited = std::any_of(payloads.begin(), payloads.end(), [&](const auto& p) {
        return std::find(f.groupMaterials.begin(), f.groupMaterials.end(), p.first) != f.groupMaterials.end();
      });
      if(edited && !f.id.isNone() && f.groupMaterials.size() == f.groupRuns.size() && mergePlan(f.groupRuns, f.groupMaterials) != f.merged)
      {
        all = false;
        break;
      }
    }
  return all;
}

rs::MaterialId FigureFeed::materialFor(int material)
{
  if(material >= 0 && material < int(m_materials.size()))
    return m_materials[size_t(material)];
  // the glTF default material, as an import makes it for a primitive with none
  if(m_default.isNone())
    m_default = m_scene.addMaterial({.owner = {m_options.prop, -1}, .textures = m_table});
  return m_default;
}

//--------------------------------------------------------------------------------------------------
// Entities

void FigureFeed::syncNodes(const SceneGraph& figure, int root, const build::Drawing& drawing)
{
  m_entities.resize(figure.nodes.size());
  std::vector<bool> reached(figure.nodes.size(), false);
  for(int i : drawing.order)
    reached[size_t(i)] = true;
  // taken off: the entity goes with everything under it
  for(size_t i = 0; i < m_entities.size(); i++)
    if(!m_entities[i].isNone() && !reached[i])
    {
      if(m_scene.entity(m_entities[i]))
        m_scene.destroyEntity(m_entities[i]);
      m_entities[i] = {};
    }

  for(int i : drawing.order)
  {
    const Node&        n      = figure.nodes[size_t(i)];
    const rs::EntityId parent = i == root ? m_options.parent : m_entities[size_t(n.parent)];
    // its place among its parent's children that are fed (traversal met every earlier sibling first)
    int index = -1;
    if(i != root)
    {
      index = 0;
      for(int c : figure.nodes[size_t(n.parent)].children)
      {
        if(c == i)
          break;
        index += !m_entities[size_t(c)].isNone();
      }
    }
    rs::EntityId& e = m_entities[size_t(i)];
    if(e.isNone())
      e = m_scene.createEntity(n.name, {m_options.prop, int32_t(i)}, parent, index);
    else if(m_scene.entity(e)->parent != parent)
      m_scene.setParent(e, parent, index);

    const rs::Transform t      = transformOf(figure, i, root);
    const rs::Entity*   entity = m_scene.entity(e);
    if(!(entity->local == t))
      m_scene.setLocal(e, t);
    if(entity->visible != n.visible)
      m_scene.setVisible(e, n.visible);
  }
}

rs::Transform FigureFeed::transformOf(const SceneGraph& figure, int node, int root) const
{
  const Node&   n = figure.nodes[size_t(node)];
  rs::Transform t;
  t.translation = glm::vec3(float(n.position.x), float(n.position.y), float(n.position.z));
  t.rotation    = glm::quat(float(n.quaternion.w), float(n.quaternion.x), float(n.quaternion.y), float(n.quaternion.z));
  t.scale       = glm::vec3(float(n.scale.x), float(n.scale.y), float(n.scale.z));
  if(node == root && m_options.rootScale != 1)
    t.scale = glm::vec3(float(n.scale.x * m_options.rootScale), float(n.scale.y * m_options.rootScale),
                        float(n.scale.z * m_options.rootScale));
  return t;
}

bool FigureFeed::syncTransforms(const SceneGraph& figure, int root, std::span<const int> nodes)
{
  bool all = true;
  for(int i : nodes)
  {
    const rs::EntityId e = entityOf(i);
    if(i >= int(figure.nodes.size()) || !m_scene.entity(e))
    {
      // not fed (new, or taken off): a sync's
      all &= i >= 0 && i < int(figure.nodes.size()) && takenOff(figure, i);
      continue;
    }
    const rs::Transform t      = transformOf(figure, i, root);
    const rs::Entity*   entity = m_scene.entity(e);
    if(!(entity->local == t))
      m_scene.setLocal(e, t);
    if(entity->visible != figure.nodes[size_t(i)].visible)
      m_scene.setVisible(e, figure.nodes[size_t(i)].visible);
  }
  return all;
}

//--------------------------------------------------------------------------------------------------
// Meshes and skins

std::vector<int> FigureFeed::groupMaterials(const SceneGraph& figure, int mesh, const build::DrawnView& drawn) const
{
  const Mesh&      m = figure.meshes[size_t(mesh)];
  std::vector<int> out;
  for(const mesh::Group& g : drawn.groups())
    out.push_back(g.materialIndex >= 0 && g.materialIndex < int(m.materials.size()) ? m.materials[size_t(g.materialIndex)] : -1);
  return out;
}

std::vector<uint32_t> FigureFeed::mergePlan(const std::vector<std::pair<uint32_t, uint32_t>>& runs, const std::vector<int>& materials) const
{
  std::vector<uint32_t> plan(runs.size(), rs::Submesh::kNotMerged);
  const auto payload = [&](size_t k) -> const tinygltf::Material* {
    const int                 mi = materials[k];
    const rs::MaterialRecord* r  = mi >= 0 && mi < int(m_materials.size()) ? m_scene.material(m_materials[size_t(mi)]) : nullptr;
    return r ? &r->payload : nullptr;
  };
  const auto sameLook = [&](size_t a, size_t b) {
    if(materials[a] == materials[b])
      return true;
    const tinygltf::Material* pa = payload(a);
    const tinygltf::Material* pb = payload(b);
    if(!pa || !pb)
      return false;
    // the surfaces' names and ids aside (extras.fvId): what they are called, not what they look like
    const auto unnamed = [](tinygltf::Material m) {
      m.name.clear();
      if(m.extras.IsObject())
      {
        tinygltf::Value::Object extras = m.extras.Get<tinygltf::Value::Object>();
        extras.erase("fvId");
        m.extras = tinygltf::Value(std::move(extras));
      }
      return m;
    };
    return unnamed(*pa) == unnamed(*pb);
  };
  std::vector<size_t> order;  // the groups drawing something, in index order
  for(size_t k = 0; k < runs.size(); k++)
    if(runs[k].second > 0)
      order.push_back(k);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return runs[a].first < runs[b].first; });
  size_t head = SIZE_MAX, last = SIZE_MAX;
  for(size_t k : order)
  {
    if(last != SIZE_MAX && runs[k].first == runs[last].first + runs[last].second && sameLook(head, k))
      plan[k] = uint32_t(head);
    else
      head = k;
    last = k;
  }
  return plan;
}

std::vector<bool> FigureFeed::tangentGroups(const std::vector<int>& materials) const
{
  // as nvvkgltf::Scene chooses: a primitive with no material is weighed as material 0
  std::vector<bool> out;
  for(int mi : materials)
  {
    const int                 chosen = mi >= 0 ? mi : 0;
    const rs::MaterialRecord* r      = chosen < int(m_materials.size()) ? m_scene.material(m_materials[size_t(chosen)]) : nullptr;
    out.push_back(r && rs::materialNeedsTangents(r->payload));
  }
  return out;
}

rs::MeshRecord FigureFeed::meshRecord(const SceneGraph& figure, int mesh, const build::DrawnView& drawn, const std::vector<uint32_t>& merged) const
{
  const Mesh&            m     = figure.meshes[size_t(mesh)];
  const mesh::BuiltMesh& geo   = *m.geo;
  const size_t           count = geo.position.size() / 3;
  rs::MeshRecord         r;
  r.name  = m.name;
  r.owner = {m_options.prop, int32_t(mesh)};
  r.positions.resize(count);
  std::memcpy(r.positions.data(), geo.position.data(), count * sizeof(glm::vec3));
  if(geo.normal.size() == count * 3)
  {
    r.normals.resize(count);
    std::memcpy(r.normals.data(), geo.normal.data(), count * sizeof(glm::vec3));
  }
  if(geo.uv.size() == count * 2)
  {
    r.uv0.resize(count);
    const bool flip = !m_options.build.texturesFlipped;
    for(size_t v = 0; v < count; v++)
      r.uv0[v] = glm::vec2(geo.uv[v * 2], flip ? 1 - geo.uv[v * 2 + 1] : geo.uv[v * 2 + 1]);
  }
  if(m.skinned() && m.skin.skinIndex.size() == count * 4 && m.skin.skinWeight.size() == count * 4)
  {
    r.joints.resize(count);
    r.weights.resize(count);
    const bool eight = m_options.dazSkinning && m.skin.skinIndex8.size() == count * 8 && m.skin.skinWeight8.size() == count * 8;
    if(eight)
    {
      r.joints1.resize(count);
      r.weights1.resize(count);
    }
    for(size_t v = 0; v < count; v++)
    {
      if(eight)
      {
        const uint16_t* j = &m.skin.skinIndex8[v * 8];
        r.joints[v]       = glm::uvec4(j[0], j[1], j[2], j[3]);
        r.joints1[v]      = glm::uvec4(j[4], j[5], j[6], j[7]);
        r.weights[v]      = glm::make_vec4(&m.skin.skinWeight8[v * 8]);
        r.weights1[v]     = glm::make_vec4(&m.skin.skinWeight8[v * 8 + 4]);
        continue;
      }
      const uint16_t* j = &m.skin.skinIndex[v * 4];
      r.joints[v]       = glm::uvec4(j[0], j[1], j[2], j[3]);
      r.weights[v]      = glm::make_vec4(&m.skin.skinWeight[v * 4]);
    }
    if(m_options.dazSkinning)
    {
      r.dualQuat = m.skin.dualQuat;
      if(m.skin.dualQuatShare.size() == count)
        r.dualQuatShares = m.skin.dualQuatShare;
    }
  }
  r.indices = drawn.index();
  for(size_t k = 0; k < drawn.groups().size(); k++)
  {
    const mesh::Group& g = drawn.groups()[k];
    // every group keeps its submesh, empty or not: one a graft hides whole draws nothing until it fills
    r.submeshes.push_back({.firstIndex = g.start, .indexCount = g.count, .material = uint32_t(k), .order = int32_t(k),
                           .mergedInto = k < merged.size() ? merged[k] : rs::Submesh::kNotMerged});
  }
  return r;
}

std::vector<uint32_t> FigureFeed::tangentIndices(const rs::MeshRecord& record, const std::vector<bool>& groups) const
{
  // one set of tangents over the triangles of every group that needs them (the build's primitives
  // share the vertices, and nvvkgltf::Scene makes them so per attribute set)
  std::vector<uint32_t> indices;
  for(size_t k = 0; k < record.submeshes.size() && k < groups.size(); k++)
    if(groups[k])
    {
      const rs::Submesh& s = record.submeshes[k];
      indices.insert(indices.end(), record.indices.begin() + s.firstIndex, record.indices.begin() + s.firstIndex + s.indexCount);
    }
  return indices;
}

void FigureFeed::syncMeshes(const SceneGraph& figure, const build::Drawing& drawing)
{
  // the first node traversal meets drawing each mesh: it holds the skin (the build's other nodes
  // drawing it are DAZ instances, never skinned)
  std::vector<int> firstNode(figure.meshes.size(), -1);
  for(int i : drawing.order)
    if(const Node& n = figure.nodes[size_t(i)]; draws(figure, n) && firstNode[size_t(n.mesh)] < 0)
      firstNode[size_t(n.mesh)] = i;

  std::vector<rs::MeshId>   oldMeshes;  // let go of once the instances have moved to the new records
  std::vector<rs::SkinId>   oldSkins;
  std::vector<std::vector<rs::MaterialId>> partMaterials(figure.meshes.size());
  m_meshes.resize(figure.meshes.size());
  for(size_t mi = 0; mi < figure.meshes.size(); mi++)
  {
    FedMesh& f = m_meshes[mi];
    if(firstNode[mi] < 0)
    {
      // drawn by nothing now (taken off): its records go when the last entity lets go of them
      oldMeshes.push_back(f.id);
      oldSkins.push_back(f.skin);
      f = {};
      continue;
    }
    const Mesh&             m      = figure.meshes[mi];
    const build::DrawnView& drawn  = *drawing.drawn[mi];
    const std::vector<int>  groups = groupMaterials(figure, int(mi), drawn);
    for(int g : groups)
      partMaterials[mi].push_back(materialFor(g));
    const std::vector<bool> tangents = tangentGroups(groups);
    const bool              flipV    = !m_options.build.texturesFlipped;
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    for(const mesh::Group& g : drawn.groups())
      runs.emplace_back(g.start, g.count);
    const std::vector<uint32_t> merged = mergePlan(runs, groups);
    f.groupRuns                        = runs;
    f.groupMaterials                   = groups;
    if(f.id.isNone() || f.geo.lock() != m.geo || f.hidden != m.geo->hiddenPolys || f.flipV != flipV || f.tangentGroups != tangents
       || f.merged != merged)
    {
      rs::MeshRecord r   = meshRecord(figure, int(mi), drawn, merged);
      f.merged           = merged;
      f.tangentIndices   = tangentIndices(r, tangents);
      // subdivided, or strand ribbons once widened: the GPU makes its drawn vertices from its cage
      f.refine = mesh::drawnFromCage(*m.geo) && m_options.cageOf ? refineTables(*m.geo) : nullptr;
      if(f.refine)
      {
        r.cage = cageOf(int(mi), f);
        if(r.cage.empty())
          f.refine = nullptr;  // no cage to give: drawn as the CPU makes it
        r.refine = f.refine;
      }
      // DAZ's order: the cage's own influences, as dForce and the collider skin the cage
      if(f.refine && m_options.dazOrder && m.skinned() && m.cageSkin.start.size() == r.cage.size() + 1)
      {
        const dforce::CageSkin cs    = dforce::cageSkin(m.cageSkin, m_options.dazSkinning);
        const size_t           n     = r.cage.size();
        const bool             eight = cs.index8.size() == n * 8;
        const auto&            index = eight ? cs.index8 : cs.index;
        const auto&            wt    = eight ? cs.weight8 : cs.weight;
        const size_t           per   = eight ? 8 : 4;
        r.cageJoints.resize(n);
        r.cageWeights.resize(n);
        if(eight)
        {
          r.cageJoints1.resize(n);
          r.cageWeights1.resize(n);
        }
        for(size_t v = 0; v < n; v++)
        {
          const uint16_t* j = &index[v * per];
          r.cageJoints[v]   = glm::uvec4(j[0], j[1], j[2], j[3]);
          r.cageWeights[v]  = glm::make_vec4(&wt[v * per]);
          if(eight)
          {
            r.cageJoints1[v]  = glm::uvec4(j[4], j[5], j[6], j[7]);
            r.cageWeights1[v] = glm::make_vec4(&wt[v * per + 4]);
          }
        }
        r.cageDualQuatShares = cs.dualQuatShare;
      }
      f.tangentAdjacency = std::nullopt;
      // the GPU makes a refined mesh's tangents (zeros here, made when it is resident)
      if(f.refine && !f.tangentIndices.empty() && !r.uv0.empty())
        if(const auto adjacency = tinygltf::utils::tangentAdjacency(f.tangentIndices, r.positions.size()))
        {
          auto tables            = std::make_shared<rs::RefineTables>(*f.refine);
          tables->tangentIndices = f.tangentIndices;
          tables->tangentStart   = adjacency->start;
          tables->tangentCorners = adjacency->corners;
          f.refine               = tables;
          r.refine               = tables;
          r.tangents.assign(r.positions.size(), glm::vec4(0.f));
        }
      if(!f.tangentIndices.empty() && !gpuTangents(f))
      {
        r.tangents.assign(r.positions.size(), glm::vec4(0.f));
        tinygltf::utils::computeTangents(r.positions, r.normals, r.uv0, f.tangentIndices, r.tangents);
        f.tangentAdjacency = tinygltf::utils::tangentAdjacency(f.tangentIndices, r.positions.size());
      }
      oldMeshes.push_back(f.id);
      f.id            = m_scene.addMesh(std::move(r));
      f.geo           = m.geo;
      f.hidden        = m.geo->hiddenPolys;
      f.flipV         = flipV;
      f.tangentGroups = tangents;
      f.tangentsLate  = false;
    }
    else
      // kept: its vertices as the scene has them now, which something may have written in place
      // without a change to report (a wear's looks set the dials its file restores)
      moveVertices(m, f);

    // its skin: the bones' entities and the inverse binds the build writes
    if(std::optional<rs::SkinRecord> skin = skinRecord(m))
    {
      if(f.skin.isNone())
        f.skin = m_scene.addSkin(std::move(*skin));
      else if(const rs::SkinRecord* had = m_scene.skin(f.skin);
              had && (had->joints != skin->joints || had->inverseBind != skin->inverseBind))
        m_scene.setSkin(f.skin, std::move(*skin));
    }
    else if(!f.skin.isNone())
    {
      oldSkins.push_back(f.skin);
      f.skin = {};
    }
  }

  // what each node draws
  for(int i : drawing.order)
  {
    const Node&                     n = figure.nodes[size_t(i)];
    std::optional<rs::MeshInstance> instance;
    if(draws(figure, n))
    {
      const FedMesh& f = m_meshes[size_t(n.mesh)];
      instance.emplace();
      instance->parts.push_back({.mesh = f.id, .materials = partMaterials[size_t(n.mesh)]});
      if(firstNode[size_t(n.mesh)] == i)
        instance->skin = f.skin;
    }
    const rs::EntityId e = m_entities[size_t(i)];
    if(!sameInstance(m_scene.entity(e)->mesh, instance))
      m_scene.setMeshInstance(e, std::move(instance));
  }

  for(rs::MeshId id : oldMeshes)
    m_scene.release(id);
  for(rs::SkinId id : oldSkins)
    m_scene.release(id);
}

void FigureFeed::moveVertices(const Mesh& m, FedMesh& f, bool tangentsLater)
{
  // the same triangles over vertices that moved
  const mesh::BuiltMesh& geo   = *m.geo;
  const size_t           count = geo.position.size() / 3;
  std::vector<glm::vec3> positions(count), normals;
  std::memcpy(positions.data(), geo.position.data(), count * sizeof(glm::vec3));
  if(geo.normal.size() == count * 3)
  {
    normals.resize(count);
    std::memcpy(normals.data(), geo.normal.data(), count * sizeof(glm::vec3));
  }
  const rs::MeshRecord& had  = *m_scene.mesh(f.id);
  const bool            same = had.positions.size() == count && had.normals.size() == normals.size();
  // fed from these vertices already (a composition made its record after the dial that moved them),
  // unless its tangents are still to be made. A refined mesh's drawn vertices stay as loaded while
  // its morphs move its cage (MorphSet::cageOnly): its cage is fed whenever it moved.
  if(same && std::memcmp(had.positions.data(), positions.data(), count * sizeof(glm::vec3)) == 0
     && std::memcmp(had.normals.data(), normals.data(), normals.size() * sizeof(glm::vec3)) == 0 && (tangentsLater || !f.tangentsLate))
  {
    if(f.refine)
      if(std::vector<glm::vec3> cage = cageOf(int(&f - m_meshes.data()), f); !cage.empty() && cage != had.cage)
        m_scene.setMeshCage(f.id, std::move(cage));
    return;
  }
  std::vector<glm::vec4> t;
  // a mesh the GPU makes the tangents of has them made with its positions
  const bool cpuTangents = !f.tangentIndices.empty() && !gpuTangents(f);
  if(cpuTangents && tangentsLater)
  {
    // the drag draws with the tangents it had: settleTangents makes them once it stops
    f.tangentsLate = true;
    f.movedAt      = std::chrono::steady_clock::now();
  }
  else if(cpuTangents)
  {
    // made again only near the vertices that moved: a vertex's tangent reads the triangles
    // touching it, and the old ones were made over the same triangles (all of them, when some
    // were left for later)
    bool made = false;
    if(f.tangentAdjacency && same && had.tangents.size() == count && !f.tangentsLate)
    {
      std::vector<uint32_t> changed;
      for(uint32_t v = 0; v < uint32_t(count); v++)
        if(std::memcmp(&had.positions[v], &positions[v], sizeof(glm::vec3)) != 0
           || (!normals.empty() && std::memcmp(&had.normals[v], &normals[v], sizeof(glm::vec3)) != 0))
          changed.push_back(v);
      t    = had.tangents;
      made = tinygltf::utils::updateTangents(positions, normals, had.uv0, f.tangentIndices, *f.tangentAdjacency, changed, t);
    }
    if(!made)
    {
      t.assign(count, glm::vec4(0.f));
      tinygltf::utils::computeTangents(positions, normals, had.uv0, f.tangentIndices, t);
    }
    f.tangentsLate = false;
  }
  m_scene.setMeshVertices(f.id, std::move(positions), std::move(normals), std::move(t),
                          f.refine ? cageOf(int(&f - m_meshes.data()), f) : std::vector<glm::vec3>{});
}

void FigureFeed::settleTangents(std::chrono::steady_clock::duration still)
{
  const auto now = std::chrono::steady_clock::now();
  for(FedMesh& f : m_meshes)
  {
    if(!f.tangentsLate || now - f.movedAt < still)
      continue;
    f.tangentsLate            = false;
    const rs::MeshRecord* had = m_scene.mesh(f.id);
    if(!had || f.tangentIndices.empty())
      continue;
    std::vector<glm::vec4> t(had->positions.size(), glm::vec4(0.f));
    tinygltf::utils::computeTangents(had->positions, had->normals, had->uv0, f.tangentIndices, t);
    m_scene.setMeshVertices(f.id, had->positions, {}, std::move(t));
  }
}

bool FigureFeed::gpuTangents(const FedMesh& f)
{
  return f.refine && !f.refine->tangentIndices.empty();
}

std::vector<glm::vec3> FigureFeed::cageOf(int mesh, const FedMesh& f) const
{
  if(!f.refine || !m_options.cageOf)
    return {};
  const std::vector<float> xyz = m_options.cageOf(mesh);
  if(xyz.size() != size_t(f.refine->cageCount) * 3)
    return {};
  std::vector<glm::vec3> out(f.refine->cageCount);
  std::memcpy(out.data(), xyz.data(), xyz.size() * sizeof(float));
  return out;
}

std::optional<rs::SkinRecord> FigureFeed::skinRecord(const Mesh& m) const
{
  const size_t count = m.geo ? m.geo->position.size() / 3 : 0;
  if(!m.skinned() || m.skin.skinIndex.size() != count * 4 || m.skin.skinWeight.size() != count * 4)
    return std::nullopt;
  rs::SkinRecord skin;
  for(int bone : m.bones)
    skin.joints.push_back(entityOf(bone));
  const std::vector<float> ibm = build::inverseBindMatrices(m);
  for(size_t j = 0; j + 16 <= ibm.size(); j += 16)
    skin.inverseBind.push_back(glm::make_mat4(&ibm[j]));
  return skin;
}

bool FigureFeed::fedAsItIs(const SceneGraph& figure, int mesh) const
{
  if(mesh < 0 || mesh >= int(figure.meshes.size()) || mesh >= int(m_meshes.size()))
    return false;
  const Mesh&    m = figure.meshes[size_t(mesh)];
  const FedMesh& f = m_meshes[size_t(mesh)];
  return !f.id.isNone() && m.geo && f.geo.lock() == m.geo && f.hidden == m.geo->hiddenPolys && m_scene.mesh(f.id)
         && m_scene.mesh(f.id)->positions.size() == m.geo->position.size() / 3;
}

bool FigureFeed::syncVertices(const SceneGraph& figure, std::span<const int> meshes, bool tangentsLater)
{
  bool all = true;
  for(int mi : meshes)
  {
    if(!fedAsItIs(figure, mi))
    {
      all = false;
      continue;
    }
    moveVertices(figure.meshes[size_t(mi)], m_meshes[size_t(mi)], tangentsLater);
  }
  return all;
}

bool FigureFeed::syncCages(const SceneGraph& figure, std::span<const int> meshes)
{
  bool all = true;
  for(int mi : meshes)
  {
    if(!fedAsItIs(figure, mi))
    {
      all = false;
      continue;
    }
    FedMesh& f = m_meshes[size_t(mi)];
    if(!f.refine)
    {
      moveVertices(figure.meshes[size_t(mi)], f, true);
      continue;
    }
    std::vector<glm::vec3> cage = cageOf(mi, f);
    if(cage != m_scene.mesh(f.id)->cage)
      m_scene.setMeshCage(f.id, std::move(cage));
  }
  return all;
}

bool FigureFeed::refines(int mesh) const
{
  return mesh >= 0 && mesh < int(m_meshes.size()) && m_meshes[size_t(mesh)].refine;
}

bool FigureFeed::syncSkins(const SceneGraph& figure, std::span<const int> meshes)
{
  bool all = true;
  for(int mi : meshes)
  {
    if(!fedAsItIs(figure, mi))
    {
      all = false;
      continue;
    }
    FedMesh&                      f    = m_meshes[size_t(mi)];
    std::optional<rs::SkinRecord> skin = skinRecord(figure.meshes[size_t(mi)]);
    // a skin that comes or goes changes what the instances draw with: a sync's
    if(skin.has_value() == f.skin.isNone())
    {
      all = false;
      continue;
    }
    if(!skin)
      continue;
    if(const rs::SkinRecord* had = m_scene.skin(f.skin); had && (had->joints != skin->joints || had->inverseBind != skin->inverseBind))
      m_scene.setSkin(f.skin, std::move(*skin));
  }
  return all;
}

}  // namespace fv
