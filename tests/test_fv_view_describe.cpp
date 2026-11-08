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

#include <cmath>
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
  const auto entity = [](uint32_t i) { return rs::EntityId{.index = i}; };

  FvView::PropNodes worn;
  fv::Json          all = FvView::describeFigure(worn, scene, root, {entity(0), entity(1), entity(2), entity(3)}, 0);
  EXPECT_EQ(all["nodes"].size(), 4u);
  EXPECT_EQ(all["stats"]["meshes"], 2);
  EXPECT_EQ(all["stats"]["triangles"], 4);
  EXPECT_EQ(all["stats"]["materials"], 2);
  // a pick names a mesh by the node that owns it: the figure, the item
  EXPECT_EQ(worn.nodes[size_t(item + 1)].name, "jacket mesh");
  EXPECT_EQ(worn.nodes[size_t(item + 1)].label, "Jacket");
  EXPECT_EQ(worn.nodes[size_t(root + 1)].label, "Genesis 9");
  EXPECT_EQ(worn.nodes[size_t(item)].label, "Jacket");

  scene.nodes[size_t(item)].removed = true;
  FvView::PropNodes off;
  fv::Json          left = FvView::describeFigure(off, scene, root, {entity(0), entity(1), {}, {}}, 0);
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
  const std::vector<rs::EntityId> entities = {{.index = 40}, {.index = 41}, {.index = 42}};
  const fv::Json                  d        = FvView::describeModel(out, model, entities);
  EXPECT_EQ(out.entityOf, entities);
  EXPECT_EQ(d["nodes"]["0"]["name"], "Car");
  EXPECT_EQ(d["nodes"]["1"]["name"], "Node-1");
  EXPECT_EQ(d["nodes"]["2"]["name"], "Wheel");
  EXPECT_EQ(d["nodes"]["2"]["parent"], 0);
  EXPECT_EQ(out.nodes[2].mesh, 1);
  EXPECT_EQ(out.nodes[2].label, "Wheel");  // a model's node keeps its own name
  EXPECT_EQ(d["stats"]["meshes"], 2);

  // a model the import could not take has no nodes in the scene
  FvView::PropNodes none;
  EXPECT_TRUE(FvView::describeModel(none, model, {})["nodes"].empty());
  EXPECT_TRUE(none.entityOf.empty());
}

// ---- the frame guide's field of view (fovForFrame), which the app draws the frame against (src/lib/frame.ts)

double tanHalf(double fovDeg)
{
  return std::tan(fovDeg * 3.14159265358979323846 / 360.0);
}

TEST(FvViewFrame, NoFrameIsTheLensAcrossTheViewsLongSide)
{
  // 50 mm on a landscape view: 35 mm across, so 35 / (16/9) up
  EXPECT_NEAR(tanHalf(fovForFrame(50, 16.0 / 9)), 17.5 / 50 / (16.0 / 9), 1e-9);
  // a portrait view: 35 mm up
  EXPECT_NEAR(tanHalf(fovForFrame(50, 9.0 / 16)), 17.5 / 50, 1e-9);
}

TEST(FvViewFrame, AFrameOfTheViewsShapeIsHeldWithTheMarginAround)
{
  EXPECT_NEAR(tanHalf(fovForFrame(50, 16.0 / 9, 16.0 / 9)), 17.5 / 50 / (16.0 / 9) * (1 + kFrameMargin), 1e-9);
}

TEST(FvViewFrame, APortraitFrameInAWideViewWidensItToHoldTheFramesHeight)
{
  // a 4:5 frame's 35 mm is up: the view must reach that, and its margin, vertically
  const double tanY = tanHalf(fovForFrame(50, 16.0 / 9, 4.0 / 5));
  EXPECT_NEAR(tanY, 17.5 / 50 * (1 + kFrameMargin), 1e-9);
  // and the frame's width fits across: its half width over the view's
  EXPECT_LT(17.5 / 50 * 0.8, tanY * 16.0 / 9);
}

TEST(FvViewFrame, AFrameWiderThanTheViewIsHeldByItsWidth)
{
  const double tanY = tanHalf(fovForFrame(50, 4.0 / 3, 2.39));
  EXPECT_NEAR(tanY * 4.0 / 3, 17.5 / 50 * (1 + kFrameMargin), 1e-9);
}
