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
    return {int(it->second.slot), it->second.flipGreen};
  if(missing.count(key))
    return {};
  const auto image = m_options.build.texture(req);
  if(!image)
  {
    missing.insert(key);
    return {};
  }
  rs::TextureRecord t;
  t.name     = req.rel.substr(req.rel.find_last_of('/') + 1);
  t.uri      = image->uri;
  t.mimeType = image->mimeType;
  if(image->mimeType == "image/ktx2")
    t.sourceExtension = "KHR_texture_basisu";
  const rs::TextureId id = m_scene.addTexture(std::move(t));
  m_textureIds.push_back(id);
  if(!grown)
    grown = std::make_shared<rs::TextureTable>(*m_table);
  grown->push_back({id, figureSamplerDesc()});
  const FedTexture fed{uint32_t(grown->size() - 1), image->flipGreen};
  m_textureOf[key] = fed;
  return {int(fed.slot), fed.flipGreen};
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

rs::MeshRecord FigureFeed::meshRecord(const SceneGraph& figure, int mesh, const build::DrawnView& drawn) const
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
    for(size_t v = 0; v < count; v++)
    {
      const uint16_t* j = &m.skin.skinIndex[v * 4];
      r.joints[v]       = glm::uvec4(j[0], j[1], j[2], j[3]);
      r.weights[v]      = glm::make_vec4(&m.skin.skinWeight[v * 4]);
    }
  }
  r.indices = drawn.index();
  for(size_t k = 0; k < drawn.groups().size(); k++)
  {
    const mesh::Group& g = drawn.groups()[k];
    // every group keeps its submesh, empty or not: one a graft hides whole draws nothing until it fills
    r.submeshes.push_back({.firstIndex = g.start, .indexCount = g.count, .material = uint32_t(k), .order = int32_t(k)});
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
    if(f.id.isNone() || f.geo.lock() != m.geo || f.hidden != m.geo->hiddenPolys || f.flipV != flipV || f.tangentGroups != tangents)
    {
      rs::MeshRecord r   = meshRecord(figure, int(mi), drawn);
      f.tangentIndices   = tangentIndices(r, tangents);
      f.tangentAdjacency = std::nullopt;
      if(!f.tangentIndices.empty())
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
  // unless its tangents are still to be made
  if(same && std::memcmp(had.positions.data(), positions.data(), count * sizeof(glm::vec3)) == 0
     && std::memcmp(had.normals.data(), normals.data(), normals.size() * sizeof(glm::vec3)) == 0 && (tangentsLater || !f.tangentsLate))
    return;
  std::vector<glm::vec4> t;
  if(!f.tangentIndices.empty() && tangentsLater)
  {
    // the drag draws with the tangents it had: settleTangents makes them once it stops
    f.tangentsLate = true;
    f.movedAt      = std::chrono::steady_clock::now();
  }
  else if(!f.tangentIndices.empty())
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
  m_scene.setMeshVertices(f.id, std::move(positions), std::move(normals), std::move(t));
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
