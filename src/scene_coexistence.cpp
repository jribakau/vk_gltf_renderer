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

#include "scene_coexistence.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <set>

#include "gltf_scene.hpp"
#include "tinygltf_utils.hpp"

namespace rs {

std::vector<std::string> retainedCannotTake(const tinygltf::Model& model, bool externalAssets)
{
  std::vector<int> parent(model.nodes.size(), -1);
  for(size_t n = 0; n < model.nodes.size(); n++)
    for(int c : model.nodes[n].children)
      if(c >= 0 && c < int(parent.size()))
        parent[size_t(c)] = int(n);
  const auto under = [&](int node) {
    for(int hops = 0; parent[size_t(node)] >= 0 && hops < int(parent.size()); hops++)
      node = parent[size_t(node)];
    const std::string& name = model.nodes[size_t(node)].name;
    return name.empty() ? "node " + std::to_string(node) : "'" + name + "'";
  };
  const auto used = [&](const char* extension) {
    return std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), extension) != model.extensionsUsed.end();
  };

  // what it is, once, and where it was first found
  std::vector<std::string> out;
  std::vector<std::string> kinds;
  const auto               add = [&](const std::string& kind, const std::string& where) {
    if(std::find(kinds.begin(), kinds.end(), kind) != kinds.end())
      return;
    kinds.push_back(kind);
    out.push_back(where.empty() ? kind : kind + " (under " + where + ")");
  };
  // a clip's node channels play (the bridge carries the nodes they move); its pointer channels, which
  // write materials, lights and cameras, don't
  for(const tinygltf::Animation& a : model.animations)
    for(const tinygltf::AnimationChannel& c : a.channels)
      if(c.target_path == "pointer" || c.target_extensions.count(KHR_ANIMATION_POINTER) > 0)
        add("KHR_animation_pointer", c.target_node >= 0 && c.target_node < int(model.nodes.size()) ? under(c.target_node) : "");
  if(used(KHR_INTERACTIVITY_EXTENSION_NAME))
    add("KHR_interactivity", "");
  if(externalAssets)
    add("glTF 2.1 external assets", "");
  return out;
}

std::optional<SceneBackendSetting> parseSceneBackendSetting(std::string_view text)
{
  for(SceneBackendSetting s : {SceneBackendSetting::eAuto, SceneBackendSetting::eLegacy, SceneBackendSetting::eRetained})
    if(text == sceneBackendSettingName(s))
      return s;
  return std::nullopt;
}

const char* sceneBackendSettingName(SceneBackendSetting setting)
{
  switch(setting)
  {
    case SceneBackendSetting::eLegacy:
      return "legacy";
    case SceneBackendSetting::eRetained:
      return "retained";
    case SceneBackendSetting::eAuto:
      break;
  }
  return "auto";
}

SceneBackendChoice chooseSceneBackend(SceneBackendSetting setting, bool streamHost, const std::vector<std::string>& cannotTake)
{
  std::string missing;
  for(const std::string& what : cannotTake)
    missing += (missing.empty() ? "" : ", ") + what;
  if(!streamHost)
    return {SceneBackend::eLegacy, "a windowed session"};
  switch(setting)
  {
    case SceneBackendSetting::eLegacy:
      return {SceneBackend::eLegacy, "sceneBackend=legacy"};
    case SceneBackendSetting::eRetained:
      return {SceneBackend::eRetained, missing.empty() ? "sceneBackend=retained" : "sceneBackend=retained; not drawn: " + missing};
    case SceneBackendSetting::eAuto:
      break;
  }
  if(missing.empty())
    return {SceneBackend::eRetained, "sceneBackend=auto"};
  return {SceneBackend::eLegacy, "sceneBackend=auto; retained can't take " + missing};
}

LegacyDirtyBridge::Result LegacyDirtyBridge::carry(const nvvkgltf::Scene& legacy, RetainedScene& retained)
{
  const nvvkgltf::Scene::DirtyFlags& df = legacy.getDirtyFlags();
  if(df.isEmpty())
    return Result::eNothing;
  // a re-parse, or samplers edited: the import is redone
  if(df.allRenderNodesDirty || df.primitivesChanged || !df.samplers.empty())
    return Result::eReimport;
  // textures: appended ones are carried; anything else (an image replaced or removed, a texture
  // removed) is a re-import. The editor flags the first textures of a model with none as a full
  // change, which for an import that had none is an append too.
  const bool hadNone = !m_map.textures || (m_map.textures->empty() && m_map.images.empty());
  if(df.texturesChanged && !hadNone)
    return Result::eReimport;
  if((df.texturesChanged || df.texturesTailChanged) && !appendTextures(legacy, retained))
    return Result::eReimport;
  const tinygltf::Model& model = legacy.getModel();

  // nodes: their transform and their own visibility (the retained scene ANDs it down the hierarchy)
  for(int n : df.nodes)
  {
    if(n < 0 || n >= int(model.nodes.size()))
      continue;
    if(size_t(n) >= m_map.entities.size())
      return Result::eReimport;  // a node added since the import
    const EntityId e = m_map.entities[size_t(n)];
    if(e.isNone() || !retained.entity(e))
      continue;  // not in the scene the import took
    retained.setLocal(e, transformOf(model.nodes[size_t(n)]));
    retained.setVisible(e, tinygltf::utils::getNodeVisibility(model.nodes[size_t(n)]).visible);
    // its morph weights (a clip's weight channel), when they are not what the entity has
    if(const Entity* drawer = retained.entity(e); drawer && drawer->mesh && !model.nodes[size_t(n)].weights.empty())
    {
      const std::vector<double>& w = model.nodes[size_t(n)].weights;
      std::vector<float>         weights(w.begin(), w.end());
      if(weights != drawer->mesh->morphWeights)
        retained.setMorphWeights(e, std::move(weights));
    }
    // an EXT_lights_ies light's multiplier and colour are its node's
    const Entity*      entity = retained.entity(e);
    const LightRecord* light  = entity ? retained.light(entity->light) : nullptr;
    if(light && light->ies)
    {
      const EXT_lights_ies_ref ref = tinygltf::utils::getNodeIesLight(model.nodes[size_t(n)]);
      retained.setLightIes(entity->light,
                           IesLight{.profile = light->ies->profile, .multiplier = ref.multiplier, .color = ref.color});
    }
  }

  for(int m : df.materials)
  {
    if(m < 0 || m >= int(model.materials.size()))
      continue;
    if(size_t(m) >= m_map.materials.size())
      return Result::eReimport;
    const MaterialId      id     = m_map.materials[size_t(m)];
    const MaterialRecord* record = retained.material(id);
    if(!record)
      continue;
    if(m_map.textures && record->textures != m_map.textures)  // the table with what was appended since
      retained.setMaterial(id, model.materials[size_t(m)], m_map.textures);
    else
      retained.setMaterial(id, model.materials[size_t(m)]);
  }

  // a glTF light is a light record per node that holds it
  for(int l : df.lights)
  {
    if(l < 0 || l >= int(model.lights.size()))
      continue;
    for(size_t n = 0; n < model.nodes.size() && n < m_map.entities.size(); n++)
    {
      if(model.nodes[n].light != l || m_map.entities[n].isNone())
        continue;
      const Entity* entity = retained.entity(m_map.entities[n]);
      if(entity && !entity->light.isNone())
        retained.setLight(entity->light, model.lights[size_t(l)]);
    }
  }
  return Result::eCarried;
}

bool LegacyDirtyBridge::carryVertices(const nvvkgltf::Scene&     legacy,
                                      RetainedScene&             retained,
                                      int                        renderPrimID,
                                      std::span<const glm::vec3> positions,
                                      std::span<const glm::vec3> normals,
                                      std::span<const uint32_t>  changed)
{
  if(renderPrimID < 0 || size_t(renderPrimID) >= legacy.getRenderPrimitives().size())
    return false;
  const tinygltf::Model& model = legacy.getModel();
  // every member of the vertex set reads the buffers the upload writes: their glTF meshes' records
  // over these accessors, each once (two meshes sharing accessors are two records here)
  std::vector<int> members(legacy.vertexSetMembers(renderPrimID).begin(), legacy.vertexSetMembers(renderPrimID).end());
  if(members.empty())
    members.push_back(renderPrimID);
  std::set<MeshId> records;
  for(int member : members)
  {
    const nvvkgltf::RenderPrimitive& rp = legacy.getRenderPrimitive(size_t(member));
    if(rp.meshID < 0 || size_t(rp.meshID) >= m_map.meshes.size() || size_t(rp.meshID) >= model.meshes.size() || !rp.pPrimitive)
      continue;
    const std::vector<tinygltf::Primitive>& prims = model.meshes[size_t(rp.meshID)].primitives;
    const int32_t                            order = int32_t(rp.pPrimitive - prims.data());
    for(MeshId id : m_map.meshes[size_t(rp.meshID)])
      if(const MeshRecord* r = retained.mesh(id))
        if(std::any_of(r->submeshes.begin(), r->submeshes.end(), [&](const Submesh& s) { return s.order == order; }))
          records.insert(id);
  }

  bool carried = false;
  for(MeshId id : records)
  {
    const MeshRecord& r = *retained.mesh(id);
    if(r.positions.size() != positions.size() || (!normals.empty() && normals.size() != r.normals.size()))
      continue;
    std::vector<glm::vec4> tangents;
    if(!r.tangents.empty())
    {
      // its tangent triangles as the import chose them: the submeshes whose material needs tangents
      // (a primitive with none weighed as material 0), else all of them for tangents the file brought
      TangentRing& ring = m_tangentRings[id];
      if(!ring.adjacency)
      {
        const int mesh = legacy.getRenderPrimitive(size_t(members.front())).meshID;
        for(const Submesh& sm : r.submeshes)
        {
          const tinygltf::Primitive& p   = model.meshes[size_t(mesh)].primitives[size_t(sm.order)];
          const int                  mat = p.material >= 0 ? p.material : 0;
          if(mat < int(model.materials.size()) && materialNeedsTangents(model.materials[size_t(mat)]))
            ring.indices.insert(ring.indices.end(), r.indices.begin() + sm.firstIndex, r.indices.begin() + sm.firstIndex + sm.indexCount);
        }
        if(ring.indices.empty())
          ring.indices = r.indices;
        ring.adjacency = tinygltf::utils::tangentAdjacency(ring.indices, r.positions.size());
      }
      const std::span<const glm::vec3> n = normals.empty() ? std::span<const glm::vec3>(r.normals) : normals;
      if(!changed.empty() && ring.adjacency)
      {
        tangents = r.tangents;
        if(!tinygltf::utils::updateTangents(positions, n, r.uv0, ring.indices, *ring.adjacency, changed, tangents))
          tangents.clear();
      }
      if(tangents.empty())
      {
        tangents.assign(r.positions.size(), glm::vec4(0.f));
        tinygltf::utils::computeTangents(positions, n, r.uv0, ring.indices, tangents);
      }
    }
    retained.setMeshVertices(id, {positions.begin(), positions.end()}, {normals.begin(), normals.end()}, std::move(tangents));
    carried = true;
  }
  return carried;
}

bool LegacyDirtyBridge::carryInverseBinds(RetainedScene& retained, int skin, std::span<const glm::mat4> inverseBind)
{
  if(skin < 0 || size_t(skin) >= m_map.skins.size())
    return false;
  const SkinRecord* had = retained.skin(m_map.skins[size_t(skin)]);
  if(!had)
    return false;
  SkinRecord record  = *had;
  record.inverseBind.assign(inverseBind.begin(), inverseBind.end());
  record.inverseBind.resize(record.joints.size(), glm::mat4(1.f));
  retained.setSkin(m_map.skins[size_t(skin)], std::move(record));
  return true;
}

// The textures the legacy model gained at its end since the import (or since the last append):
// records for their images (an image the import has is shared), the texture table extended. False
// when the model's textures are not the table's with more after them.
bool LegacyDirtyBridge::appendTextures(const nvvkgltf::Scene& legacy, RetainedScene& retained)
{
  const tinygltf::Model& model = legacy.getModel();
  const size_t           had   = m_map.textures ? m_map.textures->size() : 0;
  if(model.textures.size() < had)
    return false;
  for(size_t t = 0; t < had; t++)  // the textures before keep their images
  {
    const int source = tinygltf::utils::getTextureImageIndex(model.textures[t]);
    const TextureId want = source >= 0 && size_t(source) < m_map.images.size() ? m_map.images[size_t(source)] : TextureId{};
    if((*m_map.textures)[t].texture != want)
      return false;
  }
  if(model.textures.size() == had)
    return true;
  auto table = m_map.textures ? std::make_shared<TextureTable>(*m_map.textures) : std::make_shared<TextureTable>();
  for(size_t t = had; t < model.textures.size(); t++)
  {
    const int source = tinygltf::utils::getTextureImageIndex(model.textures[t]);
    TextureId image;
    if(source >= 0 && source < int(model.images.size()))
    {
      if(size_t(source) >= m_map.images.size())
        m_map.images.resize(size_t(source) + 1);
      TextureId& known = m_map.images[size_t(source)];
      if(known.isNone() || !retained.texture(known))
      {
        known = retained.addTexture(textureRecordOf(model, size_t(source), legacy.getImageSearchPaths()));
        m_added.push_back(known);
      }
      image = known;
    }
    table->push_back({image, samplerOf(model, model.textures[t].sampler)});
  }
  m_map.textures = std::move(table);
  return true;
}

// The inventory's allow-list: whole files of the host and the sky, and GltfRenderer's members
// that replace the scene or carry the host's edits and picks
bool legacySceneUseAllowed(std::string_view file, std::string_view function)
{
  static constexpr std::array<std::string_view, 7> kFiles = {
      "fv_host.cpp",   "fv_materials.cpp",       "fv_view.cpp", "fv_compose.cpp",
      "mcp_scene.cpp", "renderer_selection.cpp", "sky_sun.cpp",
  };
  static constexpr std::array<std::string_view, 16> kMembers = {
      // the scene replace and the frame's sync
      "finalizeSceneSetup",
      "cleanupScene",
      "createRetainedScene",
      "syncRetainedScene",
      "applyEnvironmentSkyFromScene",
      "applyPendingAuthoredSky",
      // the host's edits and addressing, and the gizmo (its hooks are onAttach's lambdas)
      "editScene",
      "appendSceneFile",
      "setNodeTransforms",
      "setInverseBindMatrices",
      "renderPrimitivesOfMesh",
      "selectSceneNodeFromScript",
      "pickSceneNodeFromScript",
      "hoverSceneNodeFromScript",
      "updateGizmoAttachment",
      "onAttach",
  };
  const size_t           slash = file.find_last_of("/\\");
  const std::string_view base  = slash == std::string_view::npos ? file : file.substr(slash + 1);
  if(std::find(kFiles.begin(), kFiles.end(), base) != kFiles.end())
    return true;
  // "GltfRenderer::name" followed by its parameters, by "::" inside a lambda, or by nothing (a
  // compiler that leaves the signature out)
  constexpr std::string_view kClass = "GltfRenderer::";
  for(size_t at = function.find(kClass); at != std::string_view::npos; at = function.find(kClass, at + 1))
  {
    const std::string_view rest = function.substr(at + kClass.size());
    for(std::string_view member : kMembers)
      if(rest.starts_with(member) && (rest.size() == member.size() || rest[member.size()] == '(' || rest[member.size()] == ':'))
        return true;
  }
  return false;
}

bool LegacyUseCheck::firstUnlisted(std::string_view file, uint32_t line, std::string_view function)
{
  const uint64_t                    site = std::hash<std::string_view>{}(file) * 1000003u + line;
  const std::lock_guard<std::mutex> lock(m_mutex);
  if(!m_seen.insert(site).second)
    return false;
  return !legacySceneUseAllowed(file, function);
}

}  // namespace rs
