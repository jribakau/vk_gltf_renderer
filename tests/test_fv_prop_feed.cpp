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

// The host's props fed as deltas (fv::PropFeed, P5 of daz-viewer docs/retained-scene-plan.md): what
// came is fed under a root of its own, what went is destroyed, and what stayed keeps its handles.

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include <figure/figure.hpp>

#include "common/fake_gpu_backend.hpp"
#include "common/residency_oracle.hpp"
#include "fv_prop_feed.hpp"
#include "scene_residency.hpp"
#include "tinygltf_utils.hpp"

namespace {

namespace fs = std::filesystem;

std::unique_ptr<fv::Figure> loadBox()
{
  const fs::path        library = fs::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions options;
  options.subdivision = 1;
  auto f = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<fs::path>{library})),
                            "People/Test/Box Figure.duf", options);
  f->settle();
  f->drain();
  return f;
}

// "chair" > "seat" (a triangle, material 0)
std::shared_ptr<const tinygltf::Model> chairFile(bool interactive = false)
{
  auto        m           = std::make_shared<tinygltf::Model>();
  const float positions[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  m->buffers.emplace_back().data.resize(sizeof(positions));
  std::memcpy(m->buffers[0].data.data(), positions, sizeof(positions));
  tinygltf::BufferView& v = m->bufferViews.emplace_back();
  v.buffer                = 0;
  v.byteLength            = sizeof(positions);
  tinygltf::Accessor& a   = m->accessors.emplace_back();
  a.bufferView            = 0;
  a.type                  = TINYGLTF_TYPE_VEC3;
  a.componentType         = TINYGLTF_COMPONENT_TYPE_FLOAT;
  a.count                 = 3;
  a.minValues             = {0, 0, 0};
  a.maxValues             = {1, 1, 0};
  m->materials.resize(1);
  tinygltf::Primitive p;
  p.attributes["POSITION"]            = 0;
  p.material                          = 0;
  m->meshes.emplace_back().primitives = {p};
  m->nodes.resize(2);
  m->nodes[0].name               = "chair";
  m->nodes[0].children           = {1};
  m->nodes[1].name               = "seat";
  m->nodes[1].mesh               = 0;
  m->scenes.emplace_back().nodes = {0};
  m->defaultScene                = 0;
  if(interactive)
    m->extensionsUsed.push_back(KHR_INTERACTIVITY_EXTENSION_NAME);
  return m;
}

tinygltf::Light pointLight(double intensity)
{
  tinygltf::Light l;
  l.type      = "point";
  l.color     = {1, 1, 1};
  l.intensity = intensity;
  return l;
}

struct Scene
{
  std::unique_ptr<fv::Figure>            box   = loadBox();
  std::shared_ptr<const tinygltf::Model> chair = chairFile(true);

  std::vector<fv::PropSpec> props() const
  {
    std::vector<fv::PropSpec> out(4);
    out[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
    out[1] = {.id = "p:2", .number = 2, .name = "Chair", .parent = "p:1", .model = chair};
    out[2] = {.id = "p:3", .number = 3, .name = "Lamp", .light = pointLight(20)};
    out[3] = {.id = "p:4", .number = 4, .name = "Camera", .camera = tinygltf::Camera{}};
    out[3].camera->type = "perspective";
    return out;
  }
};

}  // namespace

TEST(FvPropFeed, PropsThatComeAreFedUnderRootsOfTheirOwn)
{
  Scene             s;
  rs::RetainedScene scene;
  fv::PropFeed      feed(scene);
  const auto        props = s.props();
  const auto        r     = feed.sync(props);
  EXPECT_EQ(r.added, (std::vector<std::string>{"p:1", "p:2", "p:3", "p:4"}));
  EXPECT_TRUE(r.removed.empty());
  // what the chair's import leaves out, once
  ASSERT_EQ(r.notices.size(), 1u);
  EXPECT_EQ(r.notices[0].first, "p:2");

  for(const fv::PropSpec& p : props)
  {
    const fv::FedProp* fed = feed.fed(p.id);
    ASSERT_NE(fed, nullptr);
    const rs::Entity* root = scene.entity(fed->root);
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->owner, (rs::OwnerTag{p.number, -1}));
  }
  EXPECT_NE(feed.fed("p:1")->figure, nullptr);
  EXPECT_NE(scene.entity(feed.fed("p:1")->figure->entityOf(s.box->loaded.root)), nullptr);
  EXPECT_EQ(scene.entity(feed.fed("p:2")->root)->parent, feed.fed("p:1")->root);  // it rides on the figure
  EXPECT_EQ(scene.entity(feed.fed("p:2")->imported.entities[1])->owner, (rs::OwnerTag{2, 1}));
  EXPECT_EQ(scene.entity(feed.fed("p:3")->root)->light, feed.fed("p:3")->light);
  EXPECT_NE(scene.camera(feed.fed("p:4")->camera), nullptr);

  // the same props again: nothing to do, and nothing said again
  (void)scene.takeChanges();
  const auto again = feed.sync(props);
  EXPECT_TRUE(again.added.empty() && again.removed.empty() && again.notices.empty());
  EXPECT_TRUE(scene.takeChanges().empty());
}

TEST(FvPropFeed, WhatStaysKeepsItsHandlesAndWhatWentGoes)
{
  Scene             s;
  rs::RetainedScene scene;
  fv::PropFeed      feed(scene);
  auto              props = s.props();
  feed.sync(props);
  const rs::EntityId body   = feed.fed("p:1")->figure->entityOf(s.box->loaded.root);
  const rs::EntityId seat   = feed.fed("p:2")->imported.entities[1];
  const rs::LightId  light  = feed.fed("p:3")->light;
  const size_t       before = scene.entityCount();

  // the chair goes; a light and the figure's place change in place
  props.erase(props.begin() + 1);
  props[0].transform.translation = {1, 0, 0};
  props[0].visible               = false;
  props[1].light->intensity      = 5;
  const auto r                   = feed.sync(props);
  EXPECT_EQ(r.removed, std::vector<std::string>{"p:2"});
  EXPECT_EQ(feed.fed("p:2"), nullptr);
  EXPECT_EQ(scene.entity(seat), nullptr);
  EXPECT_EQ(scene.entityCount(), before - 3);  // its root and the file's two nodes
  EXPECT_EQ(feed.fed("p:1")->figure->entityOf(s.box->loaded.root), body);
  EXPECT_EQ(scene.entity(feed.fed("p:1")->root)->local.translation, glm::vec3(1, 0, 0));
  EXPECT_FALSE(scene.shown(body));  // under a hidden root
  EXPECT_EQ(feed.fed("p:3")->light, light);
  EXPECT_EQ(scene.light(light)->payload.intensity, 5);

  // a lens that changed is a new record on the same root
  const rs::CameraId camera         = feed.fed("p:4")->camera;
  props[2].camera->perspective.yfov = 0.5;
  feed.sync(props);
  EXPECT_NE(feed.fed("p:4")->camera, camera);
  EXPECT_EQ(scene.camera(camera), nullptr);  // let go of with the change
  EXPECT_EQ(scene.entity(feed.fed("p:4")->root)->camera, feed.fed("p:4")->camera);
}

TEST(FvPropFeed, AParentThatWouldMakeACycleIsRefused)
{
  Scene             s;
  rs::RetainedScene scene;
  fv::PropFeed      feed(scene);
  auto              props = s.props();
  feed.sync(props);
  props[0].parent = "p:2";  // the figure under the chair that rides on it
  feed.sync(props);
  EXPECT_TRUE(scene.entity(feed.fed("p:1")->root)->parent.isNone());
  EXPECT_EQ(scene.entity(feed.fed("p:2")->root)->parent, feed.fed("p:1")->root);
}

TEST(FvPropFeed, AFigureChangedIsFedAgainAndEverythingGoesWithTheFeed)
{
  Scene             s;
  rs::RetainedScene scene;
  {
    fv::PropFeed feed(scene);
    auto         props = s.props();
    feed.sync(props);
    const rs::EntityId root = feed.fed("p:1")->figure->entityOf(s.box->loaded.root);
    // a node hidden in the graph is fed when the host says the figure changed
    const int child = s.box->loaded.scene.nodes[size_t(s.box->loaded.root)].children.front();
    s.box->loaded.scene.nodes[size_t(child)].visible = false;
    feed.sync(props);
    EXPECT_TRUE(scene.entity(feed.fed("p:1")->figure->entityOf(child))->visible);
    props[0].figureChanged = true;
    feed.sync(props);
    EXPECT_FALSE(scene.entity(feed.fed("p:1")->figure->entityOf(child))->visible);
    EXPECT_EQ(feed.fed("p:1")->figure->entityOf(s.box->loaded.root), root);
  }
  EXPECT_EQ(scene.entityCount(), 0u);
  EXPECT_EQ(scene.meshCount(), 0u);
  EXPECT_EQ(scene.materialCount(), 0u);
  EXPECT_EQ(scene.lightCount(), 0u);
  EXPECT_EQ(scene.clipCount(), 0u);
}

// The deltas a composition makes, resident as a frame's sync takes them: every one checked by the
// residency oracle, and none waits for the device (the structural path has no idle wait from P5).
// A duplicated model holds the model its source does, and takes its source's meshes: one record,
// one prim and one BLAS for both (each with its own materials). With the source removed, the
// duplicate keeps them; another file is imported afresh.
TEST(FvPropFeed, ADuplicateSharesItsSourcesMeshes)
{
  Scene              s;
  rs::RetainedScene  scene;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu);
  fv::PropFeed       feed(scene);
  auto               props = s.props();
  feed.sync(props);
  residency.replace(scene);
  const auto step = [&](const char* what) {
    gpu.beginFrame();
    residency.sync(scene);
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << what << ": " << p;
  };
  const size_t     meshes = scene.meshCount();
  const uint32_t   prims  = residency.primHighWater();
  const rs::MeshId chair  = feed.fed("p:2")->imported.meshes[0][0];

  props.push_back({.id = "p:5", .number = 5, .name = "Chair 2", .model = s.chair});
  feed.sync(props);
  step("duplicated");
  EXPECT_EQ(feed.fed("p:5")->imported.meshes[0][0], chair);
  EXPECT_EQ(scene.meshCount(), meshes);
  EXPECT_EQ(residency.primHighWater(), prims);
  EXPECT_NE(feed.fed("p:5")->imported.materials[0], feed.fed("p:2")->imported.materials[0]);

  props.erase(props.begin() + 1);  // the source goes
  feed.sync(props);
  step("source removed");
  EXPECT_NE(scene.mesh(chair), nullptr);

  props.push_back({.id = "p:6", .number = 6, .name = "Other chair", .model = chairFile()});
  feed.sync(props);
  step("another file");
  EXPECT_NE(feed.fed("p:6")->imported.meshes[0][0], chair);

  feed.clear();
  step("cleared");
  for(uint32_t i = 0; i < gpu.framesInFlight(); i++)
    gpu.beginFrame();
  for(const std::string& p : rs::residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

TEST(FvPropFeed, EveryDeltaIsAFramesSyncWithNoDeviceWait)
{
  Scene              s;
  rs::RetainedScene  scene;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu);
  fv::PropFeed       feed(scene);
  auto               props = s.props();
  feed.sync(props);
  residency.replace(scene);
  const auto step = [&](const char* what) {
    gpu.beginFrame();
    residency.sync(scene);
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << what << ": " << p;
  };
  step("fed");

  // a prop removed and another added in the same composition, a reparent, a light retinted
  props.erase(props.begin() + 1);
  props.push_back({.id = "p:5", .number = 5, .name = "Chair again", .parent = "p:1", .model = s.chair});
  props[1].parent       = "p:1";
  props[1].light->color = {1, 0.5, 0.5};
  feed.sync(props);
  step("removed, added, reparented");

  // the figure fed again (a wardrobe change), and everything going
  props[0].figureChanged = true;
  feed.sync(props);
  step("figure fed again");
  feed.clear();
  step("cleared");
  for(uint32_t i = 0; i < gpu.framesInFlight(); i++)
    gpu.beginFrame();
  for(const std::string& p : rs::residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
  EXPECT_EQ(gpu.waitIdleCount(), 0u);
}
