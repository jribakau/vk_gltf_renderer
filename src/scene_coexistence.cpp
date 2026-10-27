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
  for(size_t n = 0; n < model.nodes.size(); n++)
  {
    const tinygltf::Node& node = model.nodes[n];
    if(node.skin >= 0)
      add("skinning", under(int(n)));
    if(node.mesh >= 0 && node.mesh < int(model.meshes.size()))
      for(const tinygltf::Primitive& p : model.meshes[size_t(node.mesh)].primitives)
        if(!p.targets.empty())
          add("morph targets", under(int(n)));
  }
  for(const tinygltf::Animation& a : model.animations)
  {
    const auto target = std::find_if(a.channels.begin(), a.channels.end(), [&](const tinygltf::AnimationChannel& c) {
      return c.target_node >= 0 && c.target_node < int(model.nodes.size());
    });
    add("animation", target != a.channels.end() ? under(target->target_node) : "");
  }
  if(used(EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME))
    add("opacity micromaps", "");
  if(used(KHR_INTERACTIVITY_EXTENSION_NAME))
    add("KHR_interactivity", "");
  if(used(EXT_LIGHTS_IES_EXTENSION_NAME))
    add("IES light profiles", "");
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

LegacyDirtyBridge::Result LegacyDirtyBridge::carry(const nvvkgltf::Scene& legacy, RetainedScene& retained) const
{
  const nvvkgltf::Scene::DirtyFlags& df = legacy.getDirtyFlags();
  if(df.isEmpty())
    return Result::eNothing;
  // a re-parse, or images and samplers: the import is redone
  if(df.allRenderNodesDirty || df.primitivesChanged || df.texturesChanged || df.texturesTailChanged || !df.samplers.empty())
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
  }

  for(int m : df.materials)
  {
    if(m < 0 || m >= int(model.materials.size()))
      continue;
    if(size_t(m) >= m_map.materials.size())
      return Result::eReimport;
    const MaterialId id = m_map.materials[size_t(m)];
    if(!id.isNone() && retained.material(id))
      retained.setMaterial(id, model.materials[size_t(m)]);  // over the texture table it has
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
