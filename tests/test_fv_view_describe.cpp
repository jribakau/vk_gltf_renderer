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

// A prop's "nodes/<prop>" topic and its map into the renderer's scene (FvView::describeFigure,
// describeModel): what the app's scene tree lists.

#include <gtest/gtest.h>

#include <memory>

#include <tinygltf/tiny_gltf.h>

#include <figure/scene/graph.hpp>

#include "fv_view.hpp"

TEST(FvViewDescribe, ATakenOffItemIsNotInTheFiguresTree)
{
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{.name = "Genesis 9"});
  scene.materials.resize(2);
  for(int m = 0; m < 2; m++)
  {
    fv::Mesh mesh;
    mesh.geo           = std::make_shared<fv::mesh::BuiltMesh>();
    mesh.geo->fullIndex = {0, 1, 2, 0, 2, 3};
    mesh.materials     = {m};
    scene.meshes.push_back(mesh);
  }
  fv::Node body{.name = "body", .kind = fv::NodeKind::eMesh};
  body.mesh = 0;
  scene.add(body, root);
  const int item = scene.add(fv::Node{.name = "Jacket"}, root);
  fv::Node  cloth{.name = "jacket mesh", .kind = fv::NodeKind::eMesh};
  cloth.mesh = 1;
  scene.add(cloth, item);
  const std::vector<int> nodeOf = {0, 1, 2, 3};

  FvView::PropNodes worn;
  fv::Json          all = FvView::describeFigure(worn, scene, root, nodeOf, 0);
  EXPECT_EQ(all["nodes"].size(), 4u);
  EXPECT_EQ(all["stats"]["meshes"], 2);
  EXPECT_EQ(all["stats"]["triangles"], 4);
  EXPECT_EQ(all["stats"]["materials"], 2);

  scene.nodes[size_t(item)].removed = true;
  FvView::PropNodes off;
  fv::Json          left = FvView::describeFigure(off, scene, root, {0, 1, -1, -1}, 0);
  EXPECT_EQ(left["nodes"].size(), 2u);
  EXPECT_FALSE(left["nodes"].contains(std::to_string(item)));
  EXPECT_FALSE(left["nodes"].contains(std::to_string(item + 1)));
  EXPECT_EQ(left["stats"]["meshes"], 1);
  EXPECT_EQ(left["stats"]["triangles"], 2);
  EXPECT_EQ(left["stats"]["materials"], 1);
}

TEST(FvViewDescribe, AModelsNodesAreItsFilesWithNamesFromThere)
{
  tinygltf::Model model;
  model.scenes.emplace_back().nodes = {0};
  model.meshes.resize(2);
  model.meshes[1].name = "Wheel";
  tinygltf::Node car;
  car.name     = "Car";
  car.children = {1, 2};
  tinygltf::Node body;  // unnamed, with an unnamed mesh
  body.mesh = 0;
  tinygltf::Node wheel;  // unnamed, its mesh named
  wheel.mesh = 1;
  model.nodes = {car, body, wheel};

  FvView::PropNodes out;
  const fv::Json    d = FvView::describeModel(out, model, 40);
  EXPECT_EQ(out.gltfOf, (std::vector<int>{40, 41, 42}));
  EXPECT_EQ(d["nodes"]["0"]["name"], "Car");
  EXPECT_EQ(d["nodes"]["1"]["name"], "Node-1");
  EXPECT_EQ(d["nodes"]["2"]["name"], "Wheel");
  EXPECT_EQ(d["nodes"]["2"]["parent"], 0);
  EXPECT_EQ(out.nodes[2].mesh, 1);
  EXPECT_EQ(d["stats"]["meshes"], 2);

  // a model the composition could not place has no nodes in the scene
  FvView::PropNodes none;
  EXPECT_TRUE(FvView::describeModel(none, model, -1)["nodes"].empty());
  EXPECT_TRUE(none.gltfOf.empty());
}
