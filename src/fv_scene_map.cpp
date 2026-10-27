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

#include "fv_scene_map.hpp"

#include <algorithm>

#include "gltf_scene.hpp"
#include "gltf_scene_merger.hpp"

namespace fv {

FigureSceneMap mapFigure(std::vector<int>                nodeOf,
                         std::vector<int>                meshOf,
                         std::vector<int>                skinOf,
                         const nvvkgltf::IndexRemapping& base,
                         const nvvkgltf::Scene&          scene)
{
  FigureSceneMap out;
  out.nodeOf = std::move(nodeOf);
  for(int& n : out.nodeOf)
    if(n >= 0)
      n += base.nodes;
  for(int n = 0; n < int(out.nodeOf.size()); n++)
    if(out.nodeOf[size_t(n)] >= 0)
      out.sceneOf[out.nodeOf[size_t(n)]] = n;
  out.meshOf = std::move(meshOf);
  for(int& m : out.meshOf)
    if(m >= 0)
      m += base.meshes;
  out.skinOf = std::move(skinOf);
  for(int& k : out.skinOf)
    if(k >= 0)
      k += base.skins;
  out.primsOf.assign(out.meshOf.size(), {});
  for(size_t m = 0; m < out.meshOf.size(); m++)
    if(out.meshOf[m] >= 0)
      out.primsOf[m] = vertexSetOwnersOfMesh(scene, out.meshOf[m]);
  return out;
}

std::vector<int> vertexSetOwnersOfMesh(const nvvkgltf::Scene& scene, int gltfMesh)
{
  std::vector<int> out;
  if(!scene.valid())
    return out;
  const auto& prims = scene.getRenderPrimitives();
  for(int prim = 0; prim < int(prims.size()); prim++)
  {
    if(prims[size_t(prim)].meshID != gltfMesh)
      continue;
    const int owner = scene.vertexSetOwner(prim);
    if(std::find(out.begin(), out.end(), owner) == out.end())
      out.push_back(owner);
  }
  return out;
}

void SceneMap::clear()
{
  m_where.clear();
  m_materials.clear();
}

void SceneMap::add(int prop, const PropAddress& address)
{
  m_where[address.root] = {prop, -1};
  for(int i = 0; i < int(address.gltfOf.size()); i++)
    if(address.gltfOf[size_t(i)] >= 0)
      m_where[address.gltfOf[size_t(i)]] = {prop, i};
  if(m_materials.size() <= size_t(prop))
    m_materials.resize(size_t(prop) + 1);
  m_materials[size_t(prop)] = {address.materials, address.materialCount};
}

HostAddress SceneMap::locate(int gltf) const
{
  const auto it = m_where.find(gltf);
  if(it == m_where.end())
    return {};
  return {.prop = it->second.first, .node = it->second.second};
}

int SceneMap::propOfRoot(int gltf) const
{
  const auto it = m_where.find(gltf);
  return it == m_where.end() || it->second.second != -1 ? -1 : it->second.first;
}

HostAddress SceneMap::pick(const nvvkgltf::Scene& scene, int renderNode) const
{
  if(renderNode < 0 || !scene.valid() || renderNode >= int(scene.getRenderNodes().size()))
    return {};
  const nvvkgltf::RenderNode& rn  = scene.getRenderNodes()[size_t(renderNode)];
  HostAddress                 out = locate(rn.refNodeID);
  if(out.prop < 0)
    return out;
  const auto [first, count] = m_materials[size_t(out.prop)];
  const int i               = rn.materialID - first;
  if(i >= 0 && i < count)
    out.surface = i;
  return out;
}

}  // namespace fv
