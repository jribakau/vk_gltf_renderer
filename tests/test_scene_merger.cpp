/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// SceneMerger: where a merge appends, and how each imported root picks its parent.

#include <gtest/gtest.h>

#include "gltf_scene_merger.hpp"

namespace {

// `roots` root nodes in one scene, each drawing the one mesh, which has no geometry of its own
tinygltf::Model model(int roots)
{
  tinygltf::Model m;
  m.scenes.emplace_back();
  m.meshes.emplace_back();
  for(int i = 0; i < roots; i++)
  {
    tinygltf::Node node;
    node.mesh = 0;
    m.nodes.push_back(node);
    m.scenes[0].nodes.push_back(i);
  }
  return m;
}

}  // namespace

TEST(SceneMerger, EachRootHangsFromTheNodeItChose)
{
  tinygltf::Model base = model(2);
  base.nodes[0].children = {1};
  base.scenes[0].nodes   = {0};
  const tinygltf::Model imported = model(2);

  int              asked  = 0;
  const auto       merged = nvvkgltf::SceneMerger::mergeUnder(base, imported, [&](const tinygltf::Node&) { return ++asked, 0; }, 0);
  EXPECT_EQ(asked, 2);
  ASSERT_EQ(merged.roots.size(), 2u);
  // the node keeps its own child, and there is no wrapper node and no new scene root
  EXPECT_EQ(base.nodes.size(), 4u);
  EXPECT_EQ(base.nodes[0].children, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(base.scenes[0].nodes, std::vector<int>{0});
  // the arrays are appended to, so what was there keeps its index
  EXPECT_EQ(base.meshes.size(), 2u);
  EXPECT_EQ(base.nodes[1].mesh, 0);
  EXPECT_EQ(base.nodes[2].mesh, 1);
}

TEST(SceneMerger, NoParentMakesARootOfTheScene)
{
  tinygltf::Model base = model(1);
  const auto      merged = nvvkgltf::SceneMerger::mergeUnder(base, model(2), [](const tinygltf::Node&) { return -1; }, 0);
  ASSERT_EQ(merged.roots.size(), 2u);
  EXPECT_EQ(base.scenes[0].nodes, (std::vector<int>{0, 1, 2}));
}

TEST(SceneMerger, AMissingParentRefusesAndChangesNothing)
{
  tinygltf::Model base = model(1);
  const auto      merged = nvvkgltf::SceneMerger::mergeUnder(base, model(2), [](const tinygltf::Node&) { return 6; }, 0);
  EXPECT_TRUE(merged.roots.empty());
  EXPECT_EQ(base.nodes.size(), 1u);
  EXPECT_EQ(base.meshes.size(), 1u);
  EXPECT_EQ(base.scenes[0].nodes, std::vector<int>{0});
}

// Where a merge appends: every array's size, micromaps from the root EXT_mesh_opacity_micromap list.
TEST(SceneMerger, OffsetsCountMicromaps)
{
  tinygltf::Model         m;
  tinygltf::Value::Object ext;
  ext["micromaps"] = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(tinygltf::Value::Object{}), tinygltf::Value(tinygltf::Value::Object{})});
  m.extensions["EXT_mesh_opacity_micromap"] = tinygltf::Value(ext);
  m.nodes.resize(3);
  EXPECT_EQ(nvvkgltf::SceneMerger::offsetsOf(m).micromaps, 2);
  EXPECT_EQ(nvvkgltf::SceneMerger::offsetsOf(m).nodes, 3);
}
