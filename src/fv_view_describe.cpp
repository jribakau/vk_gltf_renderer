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

// The "nodes/<prop>" topics: a prop's nodes as the scene tree lists them, and the map from them to
// the renderer's scene (FvView::PropNodes). No renderer and no Vulkan, so the tests build it.

#include "fv_view.hpp"

#include <algorithm>
#include <functional>
#include <set>
#include <string>

#include <tinygltf/tiny_gltf.h>

#include <figure/dson/material_preset.hpp>
#include <figure/scene/graph.hpp>

fv::Json FvView::describeFigure(PropNodes& out, const fv::SceneGraph& scene, int root, std::vector<rs::EntityId> entityOf, int morphs)
{
  out.entityOf = std::move(entityOf);
  out.nodes.assign(scene.nodes.size(), {});
  fv::Json nodes = fv::Json::object();
  int      order = 0, bones = 0, triangles = 0;
  const std::function<void(int, int)> walk = [&](int node, int depth) {
    // a taken-off item stays in the scene graph (Node::removed), but not in its tree
    if(node < 0 || node >= int(scene.nodes.size()) || scene.nodes[size_t(node)].removed)
      return;
    const fv::Node& n    = scene.nodes[node];
    std::string     kind = "group";
    if(n.kind == fv::NodeKind::eBone)
    {
      kind = "bone";
      bones++;
    }
    else if(n.kind == fv::NodeKind::eMesh)
      kind = n.mesh >= 0 && n.mesh < int(scene.meshes.size()) && scene.meshes[n.mesh].skinned() ? "skinned" : "mesh";
    out.nodes[node] = {n.name.empty() ? kind : n.name, kind, n.kind == fv::NodeKind::eMesh ? n.mesh : -1};
    nodes[std::to_string(node)] = {{"name", out.nodes[node].name}, {"parent", n.parent}, {"kind", kind}, {"order", order++}, {"depth", depth}};
    for(int child : n.children)
      walk(child, depth + 1);
  };
  if(root >= 0)
    walk(root, 0);
  // the stats count what is worn, not what was taken off
  const std::vector<bool> liveMesh = fv::liveMeshes(scene);
  size_t                  meshes   = 0;
  for(size_t m = 0; m < scene.meshes.size(); m++)
  {
    if(!liveMesh[m])
      continue;
    meshes++;
    if(const auto& geo = scene.meshes[m].geo)
      triangles += int((geo->fullIndex.empty() ? geo->renderTris.size() : geo->fullIndex.size()) / 3);
  }
  const std::vector<bool> liveMaterial = fv::dson::liveMaterials(scene);
  const size_t            materials    = size_t(std::count(liveMaterial.begin(), liveMaterial.end(), true));
  return {{"nodes", nodes},
          {"stats", {{"meshes", meshes}, {"triangles", triangles}, {"materials", materials}, {"bones", bones}, {"morphs", morphs}}}};
}

fv::Json FvView::describeModel(PropNodes& out, const tinygltf::Model& model, std::vector<rs::EntityId> entityOf)
{
  // a model the import could not take has no nodes in the scene
  const int count = entityOf.empty() ? 0 : int(model.nodes.size());
  entityOf.resize(size_t(count));
  out.entityOf = std::move(entityOf);
  out.bones = {};
  std::vector<int> parent(size_t(count), -1);
  for(int i = 0; i < count; i++)
    for(int c : model.nodes[size_t(i)].children)
      if(c >= 0 && c < count)
        parent[size_t(c)] = i;
  out.nodes.assign(size_t(count), {});
  fv::Json      nodes     = fv::Json::object();
  int           order     = 0;
  size_t        triangles = 0;
  std::set<int> meshes;
  std::set<int> materials;
  const std::function<void(int, int)> walk = [&](int node, int depth) {
    if(node < 0 || node >= count || depth > 512)
      return;
    const tinygltf::Node& n    = model.nodes[size_t(node)];
    const bool            mesh = n.mesh >= 0 && n.mesh < int(model.meshes.size());
    std::string           kind = n.mesh >= 0 ? (n.skin >= 0 ? "skinned" : "mesh") : n.camera >= 0 ? "camera" : n.light >= 0 ? "light" : "group";
    // the file's own names, else its mesh's, else one from its place in the file: never the
    // composed scene's, which change as other props come and go
    const std::string label = !n.name.empty()                              ? n.name :
                              mesh && !model.meshes[size_t(n.mesh)].name.empty() ? model.meshes[size_t(n.mesh)].name :
                                                                                   "Node-" + std::to_string(node);
    out.nodes[size_t(node)]     = {label, kind, mesh ? n.mesh : -1};
    nodes[std::to_string(node)] = {{"name", label}, {"parent", parent[size_t(node)]}, {"kind", kind}, {"order", order++}, {"depth", depth}};
    if(mesh && meshes.insert(n.mesh).second)
      for(const tinygltf::Primitive& p : model.meshes[size_t(n.mesh)].primitives)
      {
        materials.insert(p.material);
        const int acc = p.indices >= 0 ? p.indices : (p.attributes.count("POSITION") ? p.attributes.at("POSITION") : -1);
        if(acc >= 0 && acc < int(model.accessors.size()))
          triangles += model.accessors[size_t(acc)].count / 3;
      }
    for(int c : n.children)
      walk(c, depth + 1);
  };
  // the default scene's roots, as the composition hangs them from the prop root
  if(!model.scenes.empty())
  {
    const int scene = model.defaultScene >= 0 && model.defaultScene < int(model.scenes.size()) ? model.defaultScene : 0;
    for(int r : model.scenes[size_t(scene)].nodes)
      walk(r, 0);
  }
  materials.erase(-1);
  return {{"nodes", nodes},
          {"stats", {{"meshes", meshes.size()}, {"triangles", triangles}, {"materials", materials.size()}, {"bones", 0}, {"morphs", 0}}}};
}
