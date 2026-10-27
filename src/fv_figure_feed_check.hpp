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
// A fed figure read back against the scene it was fed from (daz-viewer docs/retained-scene-plan.md,
// "The figure feed"): fv::build::checkBuild (figure/gltf/gltf_check.hpp) over the retained scene.
// Every mesh a node traversal reaches is drawn from a record holding the scene's own bytes: the
// positions and their bounds, normals, UVs with the build's flip, joints widened from u16, weights,
// each drawn group's indices (a group hidden whole is an empty submesh), and with each group the
// store's material for its scene material. A skinned mesh's skin has the build's inverse bind
// matrices over the bones' entities, and a DAZ instance's nodes share one record. For the tests, and
// for fv_parity to hold beside checkBuild on the same scene, so the feed and the build stay one.
//
// checkBuild's buffer checks (one buffer, every view in it, aligned to 4, apart, zeros between them)
// and its shared-attributes check have nothing to check here: a record keeps each stream in a vector
// of its own, and its submeshes index one set of them. Indices are compared as the u32 the store
// always keeps, where the build narrows them to u16 under 65536 vertices. Transforms, visibility,
// tangents and material payloads are oracle 4's, which holds the feed to the import of the build
// (test_fv_figure_feed).
//

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include <figure/mesh/mesh_build.hpp>
#include <glm/glm.hpp>

#include "fv_figure_feed.hpp"
#include "scene_retained.hpp"

namespace fv {

// What does not match, empty when the store holds the scene's own bytes. `options`: the ones the
// feed lowers with (the UV flip follows texturesFlipped).
inline std::vector<std::string> checkFeed(const SceneGraph& scene, int root, const FigureFeed& feed,
                                          const rs::RetainedScene& store, const build::BuildOptions& options)
{
  std::vector<std::string> problems;
  if(root < 0 || root >= int(scene.nodes.size()))
  {
    problems.push_back("no root");
    return problems;
  }
  const auto same = [&](const void* got, size_t gotSize, const void* want, size_t wantSize, const std::string& what) {
    if(gotSize != wantSize || (wantSize && std::memcmp(got, want, wantSize) != 0))
      problems.push_back(what);
  };
  // a mesh node the build gives its mesh: one with geometry, vertices and groups
  const auto draws = [&](const Node& n) {
    if(n.kind != NodeKind::eMesh || n.mesh < 0 || n.mesh >= int(scene.meshes.size()))
      return false;
    const Mesh& m = scene.meshes[size_t(n.mesh)];
    return m.geo && !m.geo->position.empty() && !m.geo->groups.empty();
  };

  // the nodes traversal reaches, depth first as the feed and the build meet them (taken-off
  // subtrees left out)
  std::vector<int>  order;
  std::vector<bool> reached(scene.nodes.size(), false);
  for(std::vector<int> stack{root}; !stack.empty();)
  {
    const int i = stack.back();
    stack.pop_back();
    if(scene.nodes[size_t(i)].removed)
      continue;
    order.push_back(i);
    reached[size_t(i)] = true;
    const auto& children = scene.nodes[size_t(i)].children;
    stack.insert(stack.end(), children.rbegin(), children.rend());
  }

  // one entity per node reached, and none under the root for a node that is not (a taken-off
  // node's went with it): checkBuild's "meshes that are no scene mesh's"
  if(store.entity(feed.entityOf(root)))
    for(std::vector<rs::EntityId> stack{feed.entityOf(root)}; !stack.empty();)
    {
      const rs::EntityId id = stack.back();
      stack.pop_back();
      const rs::Entity* e = store.entity(id);
      const int         i = e->owner.local;
      if(i < 0 || i >= int(scene.nodes.size()) || !reached[size_t(i)] || feed.entityOf(i) != id)
        problems.push_back("entity " + e->name + " is no reached node's");
      stack.insert(stack.end(), e->children.begin(), e->children.end());
    }

  // per scene mesh, the nodes drawing it in traversal order: the first holds the skin (the others
  // are DAZ instances, never skinned) and every one draws the same record
  std::vector<std::vector<int>> nodesOf(scene.meshes.size());
  for(int i : order)
  {
    const Node&       n    = scene.nodes[size_t(i)];
    const std::string name = "node " + std::to_string(i) + " ";
    const rs::Entity* e    = store.entity(feed.entityOf(i));
    if(!e)
    {
      problems.push_back(name + "has no entity");
      continue;
    }
    if(!draws(n))
    {
      if(e->mesh)
        problems.push_back(name + "draws a mesh it has none of");
      continue;
    }
    if(!e->mesh || e->mesh->parts.size() != 1)
    {
      problems.push_back(name + "mesh instance");
      continue;
    }
    std::vector<int>& nodes = nodesOf[size_t(n.mesh)];
    if(!nodes.empty())
    {
      const rs::MeshInstance& first = *store.entity(feed.entityOf(nodes.front()))->mesh;
      if(e->mesh->parts[0].mesh != first.parts[0].mesh)
        problems.push_back(name + "an instance with a record of its own");
      if(!e->mesh->skin.isNone())
        problems.push_back(name + "an instance with a skin");
    }
    nodes.push_back(i);
  }

  for(size_t mi = 0; mi < scene.meshes.size(); mi++)
  {
    if(nodesOf[mi].empty())
      continue;
    const std::string       name     = "mesh " + std::to_string(mi) + " ";
    const Mesh&             m        = scene.meshes[mi];
    const mesh::BuiltMesh&  geo      = *m.geo;
    const size_t            count    = geo.position.size() / 3;
    const mesh::Drawn       drawn    = mesh::drawnTriangles(geo);
    const rs::MeshInstance& instance = *store.entity(feed.entityOf(nodesOf[mi].front()))->mesh;
    const rs::MeshRecord*   r        = store.mesh(instance.parts[0].mesh);
    if(instance.parts[0].mesh != feed.meshOf(int(mi)))
      problems.push_back(name + "not the feed's record");
    if(!r)
    {
      problems.push_back(name + "record");
      continue;
    }
    if(r->submeshes.size() != drawn.groups.size())
    {
      problems.push_back(name + "submeshes");
      continue;
    }

    same(r->positions.data(), r->positions.size() * sizeof(glm::vec3), geo.position.data(), geo.position.size() * 4, name + "positions");
    same(r->normals.data(), r->normals.size() * sizeof(glm::vec3), geo.normal.data(), geo.normal.size() * 4, name + "normals");
    std::vector<float> uv = geo.uv;
    if(!options.texturesFlipped)
      for(size_t v = 0; v < uv.size() / 2; v++)
        uv[v * 2 + 1] = 1 - uv[v * 2 + 1];
    same(r->uv0.data(), r->uv0.size() * sizeof(glm::vec2), uv.data(), uv.size() * 4, name + "UVs");
    // the bounds of the positions
    {
      glm::vec3 lo(1e30f), hi(-1e30f);
      for(size_t v = 0; v < count; v++)
      {
        lo = glm::min(lo, glm::vec3(geo.position[v * 3], geo.position[v * 3 + 1], geo.position[v * 3 + 2]));
        hi = glm::max(hi, glm::vec3(geo.position[v * 3], geo.position[v * 3 + 1], geo.position[v * 3 + 2]));
      }
      const std::array<glm::vec3, 2> bounds = r->bounds();
      if(bounds[0] != lo || bounds[1] != hi)
        problems.push_back(name + "bounds");
    }

    const bool skinned = m.skinned() && m.skin.skinIndex.size() == count * 4;
    if(skinned)
    {
      std::vector<glm::uvec4> joints(count);
      for(size_t v = 0; v < count; v++)
        joints[v] = glm::uvec4(m.skin.skinIndex[v * 4], m.skin.skinIndex[v * 4 + 1], m.skin.skinIndex[v * 4 + 2], m.skin.skinIndex[v * 4 + 3]);
      same(r->joints.data(), r->joints.size() * sizeof(glm::uvec4), joints.data(), joints.size() * sizeof(glm::uvec4), name + "joints");
      same(r->weights.data(), r->weights.size() * sizeof(glm::vec4), m.skin.skinWeight.data(), m.skin.skinWeight.size() * 4, name + "weights");
    }
    else if(!r->joints.empty())
      problems.push_back(name + "joints on a mesh with no skin");

    for(size_t k = 0; k < drawn.groups.size(); k++)
    {
      const mesh::Group& group = drawn.groups[k];
      const rs::Submesh& s     = r->submeshes[k];
      if(s.order != int32_t(k) || size_t(s.firstIndex) + s.indexCount > r->indices.size())
      {
        problems.push_back(name + "submesh " + std::to_string(k));
        continue;
      }
      same(r->indices.data() + s.firstIndex, size_t(s.indexCount) * 4, drawn.index.data() + group.start, size_t(group.count) * 4,
           name + "indices " + std::to_string(k));
      // the store's material for the group's scene material, or the default one for none, on every
      // node drawing the mesh
      const int material = group.materialIndex >= 0 && group.materialIndex < int(m.materials.size()) ? m.materials[size_t(group.materialIndex)] : -1;
      for(int node : nodesOf[mi])
      {
        const rs::MeshPart&       part = store.entity(feed.entityOf(node))->mesh->parts[0];
        const rs::MaterialId      got  = s.material < part.materials.size() ? part.materials[s.material] : rs::MaterialId{};
        const rs::MaterialRecord* mr   = store.material(got);
        if(!mr || (material >= 0 ? got != feed.materialOf(material) : mr->owner.local != -1))
        {
          problems.push_back(name + "material " + std::to_string(k));
          break;
        }
      }
    }

    if(const rs::SkinRecord* skin = store.skin(instance.skin); skin && skinned)
    {
      const std::vector<float> ibm = build::inverseBindMatrices(m);
      same(skin->inverseBind.data(), skin->inverseBind.size() * sizeof(glm::mat4), ibm.data(), ibm.size() * 4, name + "inverse binds");
      std::vector<rs::EntityId> bones;
      for(int bone : m.bones)
        bones.push_back(feed.entityOf(bone));
      if(skin->joints != bones)
        problems.push_back(name + "joint entities");
    }
    else if(skinned)
      problems.push_back(name + "no skin");
    else if(!instance.skin.isNone())
      problems.push_back(name + "a skin on a mesh with no skin");
  }
  return problems;
}

}  // namespace fv
