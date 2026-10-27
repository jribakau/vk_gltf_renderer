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

// The render nodes a selected scene node draws (renderNodesOfNode): what the selection callback,
// installed once in onAttach, puts in Resources::selectedRenderNodes for the silhouette and Fit.

#include <gtest/gtest.h>

#include <cstring>
#include <unordered_set>

#include "gltf_scene.hpp"
#include "scene_selection_render_nodes.hpp"

namespace {

// root (drawn) > child (drawn), and a drawn sibling of root; one triangle mesh shared by all three
tinygltf::Model hierarchy()
{
  tinygltf::Model m;
  const float    positions[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const uint32_t indices[]   = {0, 1, 2};
  tinygltf::Buffer& buffer   = m.buffers.emplace_back();
  buffer.data.resize(sizeof(positions) + sizeof(indices));
  std::memcpy(buffer.data.data(), positions, sizeof(positions));
  std::memcpy(buffer.data.data() + sizeof(positions), indices, sizeof(indices));
  m.bufferViews.push_back({});
  m.bufferViews[0].buffer     = 0;
  m.bufferViews[0].byteLength = sizeof(positions);
  m.bufferViews.push_back({});
  m.bufferViews[1].buffer     = 0;
  m.bufferViews[1].byteOffset = sizeof(positions);
  m.bufferViews[1].byteLength = sizeof(indices);

  tinygltf::Accessor& pos = m.accessors.emplace_back();
  pos.bufferView          = 0;
  pos.componentType       = TINYGLTF_COMPONENT_TYPE_FLOAT;
  pos.type                = TINYGLTF_TYPE_VEC3;
  pos.count               = 3;
  pos.minValues           = {0, 0, 0};
  pos.maxValues           = {1, 1, 0};
  tinygltf::Accessor& idx = m.accessors.emplace_back();
  idx.bufferView          = 1;
  idx.componentType       = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
  idx.type                = TINYGLTF_TYPE_SCALAR;
  idx.count               = 3;

  tinygltf::Primitive prim;
  prim.attributes["POSITION"] = 0;
  prim.indices                = 1;
  m.meshes.emplace_back().primitives.push_back(prim);

  for(const char* name : {"root", "child", "sibling"})
  {
    tinygltf::Node& n = m.nodes.emplace_back();
    n.name            = name;
    n.mesh            = 0;
  }
  m.nodes[0].children = {1};
  m.scenes.emplace_back().nodes = {0, 2};
  m.defaultScene                = 0;
  return m;
}

// the render nodes that draw these scene nodes
std::unordered_set<int> drawing(const nvvkgltf::Scene& scene, std::unordered_set<int> nodes)
{
  std::unordered_set<int> out;
  const auto&             renderNodes = scene.getRenderNodes();
  for(int i = 0; i < int(renderNodes.size()); i++)
    if(nodes.count(renderNodes[size_t(i)].refNodeID))
      out.insert(i);
  return out;
}

}  // namespace

TEST(SelectionRenderNodes, ANodeSelectsItsOwnAndItsDescendantsRenderNodes)
{
  nvvkgltf::Scene scene;
  scene.takeModel(hierarchy());
  ASSERT_TRUE(scene.valid());
  ASSERT_EQ(scene.getRenderNodes().size(), 3u);

  EXPECT_EQ(renderNodesOfNode(scene, 0), drawing(scene, {0, 1}));  // root and its child
  EXPECT_EQ(renderNodesOfNode(scene, 1), drawing(scene, {1}));
  EXPECT_EQ(renderNodesOfNode(scene, 2), drawing(scene, {2}));  // a sibling, nothing of root's
  EXPECT_EQ(renderNodesOfNode(scene, 0).size(), 2u);
}
