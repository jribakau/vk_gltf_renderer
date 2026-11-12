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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

#include <glm/gtc/type_ptr.hpp>

#include <figure/figure.hpp>
#include <figure/gltf/gltf_check.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/mesh/normals.hpp>

#include "common/fake_gpu_backend.hpp"
#include "common/residency_oracle.hpp"
#include "fv_figure_feed_check.hpp"
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

// The host puts what rode on a removed prop at the scene's root: it stays, handles and all, though
// its root was under the one that went.
TEST(FvPropFeed, WhatRodeOnAPropThatWentStays)
{
  Scene             s;
  rs::RetainedScene scene;
  fv::PropFeed      feed(scene);
  auto              props = s.props();
  feed.sync(props);
  const rs::EntityId root = feed.fed("p:2")->root, seat = feed.fed("p:2")->imported.entities[1];
  props.erase(props.begin());  // the figure goes
  props[0].parent.clear();
  const auto r = feed.sync(props);
  EXPECT_EQ(r.removed, std::vector<std::string>{"p:1"});
  EXPECT_TRUE(r.added.empty());
  EXPECT_EQ(feed.fed("p:2")->root, root);
  ASSERT_NE(scene.entity(root), nullptr);
  EXPECT_TRUE(scene.entity(root)->parent.isNone());
  EXPECT_NE(scene.entity(seat), nullptr);
}

// A prop riding on a figure's bone (a hat on the head): its root under the bone's entity, from the
// sync that brings the figure, so the pose carries it. The figure's feed is still the figure's own.
TEST(FvPropFeed, APropRidesOnAFiguresBone)
{
  Scene             s;
  rs::RetainedScene scene;
  fv::PropFeed      feed(scene);
  auto              props = s.props();
  const int         bone  = s.box->loaded.scene.nodes[size_t(s.box->loaded.root)].children.front();
  props[1].parentNode     = bone;
  props[1].transform.translation = {0, 0.1f, 0};
  feed.sync(props);
  const fv::FigureFeed& figure = *feed.fed("p:1")->figure;
  const rs::EntityId    chair  = feed.fed("p:2")->root;
  EXPECT_EQ(scene.entity(chair)->parent, figure.entityOf(bone));
  EXPECT_EQ(fv::checkFeed(s.box->loaded.scene, s.box->loaded.root, figure, scene, {}), std::vector<std::string>{});

  // the bone moves: the chair with it
  const glm::vec3 before = glm::vec3(scene.world(chair)[3]), boneBefore = glm::vec3(scene.world(figure.entityOf(bone))[3]);
  auto&           n      = s.box->loaded.scene.nodes[size_t(bone)];
  n.position.x += 10;
  const int moved[] = {bone};
  ASSERT_TRUE(feed.fed("p:1")->figure->syncTransforms(s.box->loaded.scene, s.box->loaded.root, moved));
  const glm::vec3 boneMoved = glm::vec3(scene.world(figure.entityOf(bone))[3]) - boneBefore;
  EXPECT_GT(glm::length(boneMoved), 1e-3f);
  EXPECT_NEAR(glm::length(glm::vec3(scene.world(chair)[3]) - before - boneMoved), 0.f, 1e-5f);

  // the same props again change nothing; back on the figure itself, then on nothing
  (void)scene.takeChanges();
  feed.sync(props);
  EXPECT_TRUE(scene.takeChanges().empty());
  props[1].parentNode = -1;
  feed.sync(props);
  EXPECT_EQ(scene.entity(chair)->parent, feed.fed("p:1")->root);
  props[1].parent.clear();
  props[1].parentNode = bone;  // a bone with no figure to ride on is none
  feed.sync(props);
  EXPECT_TRUE(scene.entity(chair)->parent.isNone());
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

// What a figure's mesh is drawn from: its own streams, or with none (it poses its cage) its drawer's
struct DrawnBuffers
{
  rs::BufferHandle positions, normals, tangents;
};
DrawnBuffers drawnBuffers(const fv::SceneGraph& graph, const fv::FigureFeed& figure, const rs::Residency& residency, int mi)
{
  const rs::Residency::MeshGpu* mesh = residency.meshGpu(figure.meshOf(mi));
  if(mesh->hasBase)
    return {mesh->positions, mesh->normals, mesh->tangents};
  for(size_t n = 0; n < graph.nodes.size(); n++)
    if(graph.nodes[n].kind == fv::NodeKind::eMesh && graph.nodes[n].mesh == mi)
      if(const rs::Residency::DrawerGpu* d = residency.drawer(figure.entityOf(int(n)), figure.meshOf(mi)))
        return {d->positions, d->normals, d->tangents};
  return {};
}

// The deltas a composition makes, resident as a frame's sync takes them: every one checked by the
// residency oracle, and none waits for the device (the structural path has no idle wait from P5).
// gpu-figures-plan.md F1: a subdivided figure fed with its morphed cages is refined
// on the GPU. The oracle checks the residency's streams against the cage subdivided as the shaders do
// it; those are the CPU's drawn positions and normals to float precision, as fed and after a dial.
TEST(FvPropFeed, ASubdividedFigureIsRefinedFromItsCage)
{
  const auto                box = loadBox();
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  props[0].figureCageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); };
  feed.sync(props);
  residency.replace(scene);
  const auto check = [&](const char* when) {
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << when << ": " << p;
    size_t refined = 0;
    for(size_t mi = 0; mi < box->loaded.scene.meshes.size(); mi++)
    {
      const rs::MeshId              id     = feed.fed("p:1")->figure->meshOf(int(mi));
      const rs::MeshRecord*         record = scene.mesh(id);
      const rs::Residency::MeshGpu* mesh   = residency.meshGpu(id);
      if(!record || !mesh || !mesh->refine)
        continue;
      refined++;
      const size_t           n     = record->positions.size();
      const DrawnBuffers     drawn = drawnBuffers(box->loaded.scene, *feed.fed("p:1")->figure, residency, int(mi));
      std::vector<glm::vec3> positions(n);
      std::memcpy(positions.data(), gpu.bytesAt(gpu.bufferAddress(drawn.positions), n * 12).data(), n * 12);
      const std::vector<glm::vec3> normals = rs::streamNormals(gpu.bytesAt(gpu.bufferAddress(drawn.normals), n * 4));
      for(size_t v = 0; v < n; v++)
      {
        ASSERT_LT(glm::distance(positions[v], record->positions[v]), 1e-4f) << when << " vertex " << v;
        ASSERT_GT(glm::dot(normals[v], record->normals[v]), 0.9999f) << when << " vertex " << v;
      }
    }
    EXPECT_GT(refined, 0u) << when;
  };
  check("fed");

  box->morphs->set("Taper", 1.0);
  box->morphs->settled();
  std::vector<int> moved;
  for(const auto& c : box->drain().meshes)
    moved.push_back(c.mesh);
  ASSERT_FALSE(moved.empty());
  EXPECT_TRUE(feed.fed("p:1")->figure->syncVertices(box->loaded.scene, moved));
  gpu.beginFrame();
  residency.sync(scene);
  check("dialled");
}

// MorphSet::cageOnly (gpu-figures-plan.md F1b): a dial's step moves a refined mesh's cage alone, the
// CPU's drawn vertices left as they were; the GPU's are what the CPU makes of the dial (a second box
// dialled the same way); the CPU's stay as they were at settled() and catch up at catchUp() (an
// export's).
// A refined mesh's HD detail (MorphSet::hdDetail, here made by hand) is fed with its cage and added as
// the GPU refines it: the drawn vertices are the CPU reference's, a new weight moves them, and none
// leaves the plain refinement
TEST(FvPropFeed, AFiguresHdDetailIsAddedAsItsCageIsRefined)
{
  const auto                box = loadBox();
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  const size_t              meshes = box->loaded.scene.meshes.size();
  std::vector<std::unique_ptr<fv::subdiv::HdMixer>>        mixers(meshes);
  std::vector<std::shared_ptr<const fv::subdiv::HdTables>> tables(meshes);
  auto edits = std::make_shared<fv::subdiv::LevelEdits>(1);
  (*edits)[0] = {{0, 0, 2, {0.05f, 0.1f, -0.02f}}, {1, 1, 0, {0.f, 0.08f, 0.f}}, {1, 2, 3, {-0.03f, 0.f, 0.04f}}};
  const auto mix = [&](float w) {
    for(size_t mi = 0; mi < meshes; mi++)
    {
      const fv::mesh::BuiltMesh& geo = *box->loaded.scene.meshes[mi].geo;
      if(!geo.refinement || geo.cagePolyStart.size() < 3)
        continue;
      if(!mixers[mi])
        mixers[mi] = std::make_unique<fv::subdiv::HdMixer>(geo.refinement, geo.cagePolyStart, geo.cagePolyVerts, fv::mesh::refinedQuads(geo));
      const fv::subdiv::HdMixer::Weighted one[] = {{edits, w}};
      tables[mi]                               = mixers[mi]->mix(one);
    }
  };
  mix(1.f);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  props[0].figureCageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); };
  props[0].figureHdOf   = [&](int mesh) { return tables[size_t(mesh)]; };
  feed.sync(props);
  residency.replace(scene);
  fv::FigureFeed& figure = *feed.fed("p:1")->figure;

  // the drawn vertices against the CPU reference with the tables as they are now
  const auto check = [&](const char* when) {
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << when << ": " << p;
    size_t checked = 0;
    for(size_t mi = 0; mi < meshes; mi++)
    {
      const rs::Residency::MeshGpu* mesh = residency.meshGpu(figure.meshOf(int(mi)));
      if(!mesh || !mesh->refine)
        continue;
      const fv::mesh::BuiltMesh& geo  = *box->loaded.scene.meshes[mi].geo;
      const std::vector<float>   cage = box->morphs->morphedCage(int(mi));
      const std::vector<float>   want = fv::subdiv::refineWithTables(*geo.refinement, cage, tables[mi].get());
      const DrawnBuffers         drawnNow = drawnBuffers(box->loaded.scene, figure, residency, int(mi));
      const size_t               n        = geo.renderIndex.size();
      std::vector<glm::vec3>     positions(n);
      std::memcpy(positions.data(), gpu.bytesAt(gpu.bufferAddress(drawnNow.positions), n * 12).data(), n * 12);
      for(size_t v = 0; v < n; v++)
        ASSERT_LT(glm::distance(positions[v], glm::make_vec3(&want[size_t(geo.renderIndex[v]) * 3])), 1e-4f)
            << when << ": mesh " << mi << " vertex " << v;
      checked++;
    }
    EXPECT_GT(checked, 0u) << when;
  };
  check("fed");
  size_t withHd = 0;
  for(size_t mi = 0; mi < meshes; mi++)
    if(const rs::Residency::MeshGpu* mesh = residency.meshGpu(figure.meshOf(int(mi))); mesh && mesh->refine && mesh->refine->hdOffsets)
      withHd++;
  ASSERT_GT(withHd, 0u) << "no mesh took its HD detail: the check above compared plain refinements";
  // and the detail moved something: the tables hold edits, and the refinement with them isn't the plain one
  float moved = 0.f;
  for(size_t mi = 0; mi < meshes; mi++)
    if(tables[mi])
    {
      EXPECT_GT(tables[mi]->topology->frames(), 0u) << "mesh " << mi << ": every edit dropped";
      const fv::mesh::BuiltMesh& geo   = *box->loaded.scene.meshes[mi].geo;
      const std::vector<float>   cage  = box->morphs->morphedCage(int(mi));
      const std::vector<float>   with  = fv::subdiv::refineWithTables(*geo.refinement, cage, tables[mi].get());
      const std::vector<float>   plain = fv::subdiv::refineWithTables(*geo.refinement, cage, nullptr);
      for(size_t i = 0; i < with.size(); i++)
        moved = std::max(moved, std::abs(with[i] - plain[i]));
    }
  EXPECT_GT(moved, 0.01f);
  const auto refined = [&] {
    std::vector<int> all;
    for(size_t mi = 0; mi < meshes; mi++)
      if(figure.refines(int(mi)))
        all.push_back(int(mi));
    return all;
  };

  mix(-0.5f);
  EXPECT_TRUE(figure.syncCages(box->loaded.scene, refined()));
  gpu.beginFrame();
  residency.sync(scene);
  check("weighed again");

  std::fill(tables.begin(), tables.end(), nullptr);
  EXPECT_TRUE(figure.syncCages(box->loaded.scene, refined()));
  gpu.beginFrame();
  residency.sync(scene);
  check("dropped");
  for(size_t mi = 0; mi < meshes; mi++)
    if(const rs::Residency::MeshGpu* mesh = residency.meshGpu(figure.meshOf(int(mi))); mesh && mesh->refine)
      EXPECT_FALSE(mesh->refine->hdOffsets) << "mesh " << mi;
}

TEST(FvPropFeed, ADialMovesARefinedFiguresCageAlone)
{
  const auto                box = loadBox(), reference = loadBox();
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  props[0].figureCageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); };
  feed.sync(props);
  residency.replace(scene);
  fv::FigureFeed& figure = *feed.fed("p:1")->figure;
  const auto      moved  = [](const std::vector<fv::morph::MeshChange>& changes) {
    std::vector<int> out;
    for(const auto& c : changes)
      out.push_back(c.mesh);
    return out;
  };
  // the dial's file read
  for(fv::Figure* f : {box.get(), reference.get()})
  {
    f->morphs->set("Taper", 0.5);
    f->morphs->settled();
  }
  EXPECT_TRUE(figure.syncVertices(box->loaded.scene, moved(box->drain().meshes)));
  gpu.beginFrame();
  residency.sync(scene);
  reference->morphs->set("Taper", 1.0);
  reference->morphs->settled();

  box->morphs->cageOnly = [&](int mesh) { return figure.refines(mesh); };
  const size_t                    meshes = box->loaded.scene.meshes.size();
  std::vector<std::vector<float>> drawn;
  for(size_t mi = 0; mi < meshes; mi++)
    drawn.push_back(box->loaded.scene.meshes[mi].geo->position);
  box->morphs->set("Taper", 1.0);
  const std::vector<fv::morph::MeshChange> step = box->drain().meshes;
  size_t                                   cages = 0;
  for(const fv::morph::MeshChange& c : step)
  {
    EXPECT_EQ(c.cage, figure.refines(c.mesh)) << "mesh " << c.mesh;
    if(c.cage)
    {
      cages++;
      EXPECT_EQ(box->loaded.scene.meshes[size_t(c.mesh)].geo->position, drawn[size_t(c.mesh)]) << "mesh " << c.mesh;
    }
  }
  ASSERT_GT(cages, 0u);
  EXPECT_TRUE(figure.syncCages(box->loaded.scene, moved(step)));
  gpu.beginFrame();
  residency.sync(scene);
  for(const std::string& p : rs::checkResidency(scene, residency, gpu))
    ADD_FAILURE() << p;
  for(size_t mi = 0; mi < meshes; mi++)
  {
    const rs::Residency::MeshGpu* mesh = residency.meshGpu(figure.meshOf(int(mi)));
    if(!mesh || !mesh->refine)
      continue;
    const fv::mesh::BuiltMesh& want  = *reference->loaded.scene.meshes[mi].geo;
    const size_t               n     = want.position.size() / 3;
    const DrawnBuffers         drawnNow = drawnBuffers(box->loaded.scene, figure, residency, int(mi));
    std::vector<glm::vec3>     positions(n);
    std::memcpy(positions.data(), gpu.bytesAt(gpu.bufferAddress(drawnNow.positions), n * 12).data(), n * 12);
    const std::vector<glm::vec3> normals = rs::streamNormals(gpu.bytesAt(gpu.bufferAddress(drawnNow.normals), n * 4));
    for(size_t v = 0; v < n; v++)
    {
      ASSERT_LT(glm::distance(positions[v], glm::make_vec3(&want.position[v * 3])), 1e-4f) << "mesh " << mi << " vertex " << v;
      ASSERT_GT(glm::dot(normals[v], glm::make_vec3(&want.normal[v * 3])), 0.9999f) << "mesh " << mi << " vertex " << v;
    }
  }

  box->morphs->settled();
  EXPECT_TRUE(box->drain().meshes.empty());
  for(size_t mi = 0; mi < meshes; mi++)
    if(figure.refines(int(mi)))
      EXPECT_EQ(box->loaded.scene.meshes[mi].geo->position, drawn[mi]) << "mesh " << mi;
  box->morphs->catchUp();
  for(const fv::morph::MeshChange& c : box->drain().meshes)
  {
    EXPECT_FALSE(c.cage) << "mesh " << c.mesh;
    EXPECT_TRUE(c.caughtUp) << "mesh " << c.mesh;  // no reason for dForce or the collider to run again
  }
  for(size_t mi = 0; mi < meshes; mi++)
  {
    const std::vector<float>& got  = box->loaded.scene.meshes[mi].geo->position;
    const std::vector<float>& want = reference->loaded.scene.meshes[mi].geo->position;
    ASSERT_EQ(got.size(), want.size());
    for(size_t i = 0; i < got.size(); i++)
      ASSERT_NEAR(got[i], want[i], 1e-6f) << "mesh " << mi << " value " << i;
  }
}

// A composition that keeps a refined mesh feeds the cage its morphs moved since, though its drawn
// vertices are as they were: a restore that wears something sets the dials its file has before the
// composition, which reports no change of its own (a character's expression and the collider's
// correction were lost so, gpu-figures-plan.md F5).
TEST(FvPropFeed, AKeptRefinedMeshIsFedTheCageItsMorphsMoved)
{
  const auto                box = loadBox(), reference = loadBox();
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  props[0].figureCageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); };
  feed.sync(props);
  residency.replace(scene);
  fv::FigureFeed& figure = *feed.fed("p:1")->figure;
  box->morphs->cageOnly  = [&](int mesh) { return figure.refines(mesh); };
  reference->morphs->set("Taper", 1.0);
  reference->morphs->settled();

  box->morphs->set("Taper", 1.0);
  box->morphs->settled();
  (void)box->drain();  // nobody feeds the change: the composition below must
  props[0].figureChanged = true;
  feed.sync(props);
  gpu.beginFrame();
  residency.sync(scene);
  for(const std::string& p : rs::checkResidency(scene, residency, gpu))
    ADD_FAILURE() << p;
  size_t refined = 0;
  for(size_t mi = 0; mi < box->loaded.scene.meshes.size(); mi++)
  {
    const rs::Residency::MeshGpu* mesh = residency.meshGpu(figure.meshOf(int(mi)));
    if(!mesh || !mesh->refine)
      continue;
    refined++;
    const fv::mesh::BuiltMesh& want = *reference->loaded.scene.meshes[mi].geo;
    const size_t               n    = want.position.size() / 3;
    std::vector<glm::vec3>     positions(n);
    std::memcpy(positions.data(), gpu.bytesAt(gpu.bufferAddress(drawnBuffers(box->loaded.scene, figure, residency, int(mi)).positions), n * 12).data(), n * 12);
    for(size_t v = 0; v < n; v++)
      ASSERT_LT(glm::distance(positions[v], glm::make_vec3(&want.position[v * 3])), 1e-4f) << "mesh " << mi << " vertex " << v;
  }
  EXPECT_GT(refined, 0u);
}

// F2: a refined mesh whose materials are normal mapped has its tangents made on the GPU from its drawn
// positions, normals and UVs, with no CPU tangents fed: computeTangents of the CPU's vertices to float
// precision, as fed and after a dial's cage step (the oracle holds the skinned drawers' to them).
TEST(FvPropFeed, ARefinedNormalMappedFiguresTangentsAreMadeOnTheGpu)
{
  const auto box = loadBox();
  for(fv::Material& m : box->loaded.scene.materials)
    m.surface.normalMap = "/Runtime/Textures/Test/BoxDetail_NM.png";
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  props[0].figureOptions.texture = [](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
    return fv::build::TextureImage{.uri = (fs::path(FV_TEST_DATA) / "daz-library" / req.rel).generic_string(), .mimeType = "image/png"};
  };
  props[0].figureCageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); };
  feed.sync(props);
  residency.replace(scene);
  fv::FigureFeed& figure = *feed.fed("p:1")->figure;
  const auto      check  = [&](const char* when) {
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << when << ": " << p;
    size_t made = 0;
    for(size_t mi = 0; mi < box->loaded.scene.meshes.size(); mi++)
    {
      const rs::MeshRecord*         record = scene.mesh(figure.meshOf(int(mi)));
      const rs::Residency::MeshGpu* mesh   = residency.meshGpu(figure.meshOf(int(mi)));
      if(!record || !mesh || !mesh->refine || !mesh->refine->tangentTriangles)
        continue;
      made++;
      const fv::mesh::BuiltMesh& geo = *box->loaded.scene.meshes[mi].geo;
      const size_t               n   = geo.position.size() / 3;
      std::vector<glm::vec3>     positions(n), normals(n);
      std::memcpy(positions.data(), geo.position.data(), n * 12);
      std::memcpy(normals.data(), geo.normal.data(), n * 12);
      std::vector<glm::vec4> want(n, glm::vec4(0.f));
      tinygltf::utils::computeTangents(positions, normals, record->uv0, record->refine->tangentIndices, want);
      const std::vector<glm::vec4> got =
          rs::streamTangents(gpu.bytesAt(gpu.bufferAddress(drawnBuffers(box->loaded.scene, figure, residency, int(mi)).tangents), n * 4));
      for(size_t v = 0; v < n; v++)
      {
        ASSERT_GT(glm::dot(glm::vec3(got[v]), glm::vec3(want[v])), 0.999f) << when << " mesh " << mi << " vertex " << v;
        ASSERT_EQ(got[v].w, want[v].w) << when << " mesh " << mi << " vertex " << v;
      }
    }
    EXPECT_GT(made, 0u) << when;
  };
  check("fed");

  const auto moved = [](const std::vector<fv::morph::MeshChange>& changes) {
    std::vector<int> out;
    for(const auto& c : changes)
      out.push_back(c.mesh);
    return out;
  };
  box->morphs->set("Taper", 0.5);  // its file read
  box->morphs->settled();
  EXPECT_TRUE(figure.syncVertices(box->loaded.scene, moved(box->drain().meshes), true));
  gpu.beginFrame();
  residency.sync(scene);
  check("dialled");
  box->morphs->cageOnly = [&](int mesh) { return figure.refines(mesh); };
  box->morphs->set("Taper", 1.0);
  EXPECT_TRUE(figure.syncCages(box->loaded.scene, moved(box->drain().meshes)));
  gpu.beginFrame();
  residency.sync(scene);
  box->morphs->catchUp();  // the CPU's vertices, to check against
  (void)box->drain();
  check("a cage step");
}

// gpu-figures-plan.md F3: a figure's records carry its meshes' dual quaternion share and its drawers
// skin with the joints' dual quaternions (the oracle skins as the shader does)
TEST(FvPropFeed, AFiguresRecordsCarryTheFilesDualQuaternionShare)
{
  const auto box = loadBox();
  for(fv::Mesh& m : box->loaded.scene.meshes)
    m.skin.dualQuat = 1.f;
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  feed.sync(props);
  residency.replace(scene);
  for(const std::string& p : rs::checkResidency(scene, residency, gpu))
    ADD_FAILURE() << p;
  size_t skinned = 0;
  for(size_t mi = 0; mi < box->loaded.scene.meshes.size(); mi++)
  {
    const rs::MeshRecord* record = scene.mesh(feed.fed("p:1")->figure->meshOf(int(mi)));
    if(!record || record->joints.empty())
      continue;
    skinned++;
    EXPECT_EQ(record->dualQuat, 1.f) << mi;
  }
  EXPECT_GT(skinned, 0u);
}

// gpu-figures-plan.md F4: a refined mesh's drawer skins the cage, then refines the
// posed cage into its streams (the oracle holds them to that, exactly). At rest that is the mesh's
// refined base; a turned bone moves it.
TEST(FvPropFeed, DazOrderSkinsTheCageThenRefinesIt)
{
  const auto                box = loadBox();
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  props[0].figureCageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); };
  feed.sync(props);
  residency.replace(scene);
  const fv::FigureFeed& figure = *feed.fed("p:1")->figure;
  const auto check = [&](const char* when) {
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << when << ": " << p;
  };
  check("at rest");

  // the drawers that pose their cage, and their outputs at rest: the refined mesh as the CPU has it
  std::vector<std::pair<const rs::Residency::DrawerGpu*, rs::MeshId>> posed;
  for(size_t n = 0; n < box->loaded.scene.nodes.size(); n++)
    if(const fv::Node& node = box->loaded.scene.nodes[n]; node.kind == fv::NodeKind::eMesh && node.mesh >= 0)
      if(const rs::Residency::DrawerGpu* d = residency.drawer(figure.entityOf(int(n)), figure.meshOf(node.mesh)); d && d->posed)
        posed.emplace_back(d, figure.meshOf(node.mesh));
  ASSERT_FALSE(posed.empty());
  std::vector<std::vector<glm::vec3>> rest;
  for(const auto& [d, mesh] : posed)
  {
    const size_t           n = scene.mesh(mesh)->positions.size();
    std::vector<glm::vec3> got(n);
    std::memcpy(got.data(), gpu.bytesAt(gpu.bufferAddress(d->positions), n * 12).data(), n * 12);
    for(size_t v = 0; v < n; v++)
      ASSERT_LT(glm::distance(got[v], scene.mesh(mesh)->positions[v]), 1e-4f) << v;
    // nothing reads the mesh's own vertices or the drawn vertices' skin, so nothing holds them
    EXPECT_FALSE(residency.meshGpu(mesh)->positions);
    EXPECT_FALSE(residency.meshGpu(mesh)->joints);
    rest.push_back(got);
  }

  // a bone turned: the posed drawers move, still as the oracle has them
  rs::EntityId bone;
  for(size_t n = 0; n < box->loaded.scene.nodes.size() && bone.isNone(); n++)
    if(box->loaded.scene.nodes[n].kind == fv::NodeKind::eBone && box->loaded.scene.nodes[n].parent >= 0
       && box->loaded.scene.nodes[size_t(box->loaded.scene.nodes[n].parent)].kind == fv::NodeKind::eBone)
      bone = figure.entityOf(int(n));
  ASSERT_FALSE(bone.isNone());
  rs::Transform t = scene.entity(bone)->local;
  t.rotation      = glm::angleAxis(glm::radians(40.f), glm::vec3(0, 0, 1)) * t.rotation;
  scene.setLocal(bone, t);
  gpu.beginFrame();
  residency.sync(scene);
  check("posed");
  float moved = 0;
  for(size_t i = 0; i < posed.size(); i++)
  {
    const size_t           n = rest[i].size();
    std::vector<glm::vec3> got(n);
    std::memcpy(got.data(), gpu.bytesAt(gpu.bufferAddress(posed[i].first->positions), n * 12).data(), n * 12);
    for(size_t v = 0; v < n; v++)
      moved = std::max(moved, glm::distance(got[v], rest[i][v]));
  }
  EXPECT_GT(moved, 1e-3f);
}

// Nothing reads a posed strand hair's drawn vertices but the GPU that makes them, so its record holds none, nor
// their skin: only how many there are. Its drawers make them from the posed cage, and the oracle holds them to
// that at rest, posed, moved by a dial, and woken after settling. An export makes the skin it needs, and what
// gives light keeps its triangles for the emissive table.
TEST(FvPropFeed, APosedStrandHairHoldsNoDrawnVerticesOnTheCpu)
{
  const fs::path library = fs::path(FV_TEST_DATA) / "daz-library";
  auto           hair    = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<fs::path>{library})),
                                            "People/Test/Clothing/Box Fan Hair.duf", {});
  hair->settle();
  hair->drain();
  fv::SceneGraph& graph  = hair->loaded.scene;
  const int       root   = hair->loaded.root;
  const auto      ribbon = std::find_if(graph.meshes.begin(), graph.meshes.end(), [](const fv::Mesh& m) { return m.geo && m.geo->ribbon; });
  ASSERT_NE(ribbon, graph.meshes.end());
  ASSERT_TRUE(ribbon->skinned());
  const int                  mi  = int(ribbon - graph.meshes.begin());
  const fv::mesh::BuiltMesh& geo = *ribbon->geo;
  const size_t               n   = geo.position.size() / 3;
  // this asset's ribbons come with no width: some, as widening would have given them
  ASSERT_EQ(geo.ribbon->offsets.size(), size_t(geo.renderVertexCount) * 3);
  for(size_t v = 0; v < size_t(geo.renderVertexCount); v++)
  {
    const float across = v % 2 ? 0.2f : -0.2f;
    for(size_t k = 0; k < 3; k++)
    {
      ribbon->geo->ribbon->offsets[v * 3 + k] += across;
      ribbon->geo->position[size_t(geo.firstSplit[v]) * 3 + k] += across;
    }
  }
  EXPECT_TRUE(ribbon->skin.skinIndex.empty()) << "made only by what asks";
  // what the CPU has of them now, to hold what it makes again to (normals: a weld of these positions)
  const std::vector<int32_t> firstSplit = geo.firstSplit, nextSplit = geo.nextSplit;
  const std::vector<float>   positions  = geo.position;
  std::vector<float>         normals(positions.size());
  fv::mesh::NormalWelder(geo.renderIndex, firstSplit, nextSplit, geo.renderTris, geo.renderVertexCount).full(positions, normals);

  rs::RetainedScene  scene;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu, {.settleQuietFrames = 0});
  std::vector<float> cage = geo.cagePositions;
  fv::FigureFeed     feed(scene, {.cageOf = [&](int, std::vector<float>&) { return std::span<const float>(cage); }});
  feed.sync(graph, root);
  residency.replace(scene);
  const rs::MeshId id = feed.meshOf(mi);
  const auto check = [&](const char* when) {
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << when << ": " << p;
  };
  const auto frames = [&](int count) {
    for(int i = 0; i < count; i++)
    {
      gpu.beginFrame();
      residency.sync(scene);
    }
  };
  const auto drawnNow = [&] {
    std::vector<glm::vec3> got(n);
    std::memcpy(got.data(), gpu.bytesAt(gpu.bufferAddress(drawnBuffers(graph, feed, residency, mi).positions), n * 12).data(), n * 12);
    return got;
  };
  const auto farthest = [](const std::vector<glm::vec3>& a, const std::vector<glm::vec3>& b) {
    float most = 0;
    for(size_t v = 0; v < a.size(); v++)
      most = std::max(most, glm::distance(a[v], b[v]));
    return most;
  };

  const rs::MeshRecord* record = scene.mesh(id);
  ASSERT_TRUE(record);
  EXPECT_EQ(record->thinVertices, n);
  EXPECT_TRUE(record->positions.empty());
  EXPECT_TRUE(record->normals.empty());
  EXPECT_TRUE(record->joints.empty());
  EXPECT_TRUE(record->weights.empty());
  EXPECT_FALSE(record->indices.empty());
  ASSERT_TRUE(residency.meshGpu(id)->refine) << "a drawer has nothing else to make them from";
  check("at rest");

  // fed thin, the CPU lets go of its drawn vertices (mesh::letGoOfDrawn) and makes them again when
  // asked: what the drawers make at rest, the splits as build made them, the normals of a weld
  EXPECT_TRUE(geo.drawnLetGo);
  EXPECT_TRUE(geo.position.empty() && geo.normal.empty() && geo.firstSplit.empty() && geo.nextSplit.empty());
  EXPECT_EQ(geo.drawnCount(), n);
  EXPECT_TRUE(geo.hasNormals());
  fv::mesh::ensureDrawn(geo);
  EXPECT_FALSE(geo.drawnLetGo);
  EXPECT_EQ(geo.firstSplit, firstSplit);
  EXPECT_EQ(geo.nextSplit, nextSplit);
  ASSERT_EQ(geo.position.size(), positions.size());
  ASSERT_EQ(geo.normal.size(), normals.size());
  for(size_t i = 0; i < positions.size(); i++)
  {
    ASSERT_NEAR(geo.position[i], positions[i], 1e-4f) << i;
    ASSERT_NEAR(geo.normal[i], normals[i], 1e-3f) << i;
  }
  const std::vector<glm::vec3> rest = drawnNow();
  for(size_t v = 0; v < n; v++)
    ASSERT_LT(glm::distance(rest[v], glm::make_vec3(&geo.position[v * 3])), 1e-4f) << v;

  // its price is what it holds, as for any other mesh
  const rs::MemoryLedger ledger = residency.ledger();
  const auto             entry  = std::find_if(ledger.meshes.begin(), ledger.meshes.end(), [&](const auto& m) { return m.mesh == id; });
  ASSERT_NE(entry, ledger.meshes.end());
  rs::MeshCounts counts   = fv::meshCounts(*ribbon);
  counts.tangents         = !record->tangents.empty();
  counts.tangentTriangles = record->refine->tangentIndices.size() / 3;
  EXPECT_EQ(rs::meshBufferBytes(counts), entry->bytes.total() - entry->bytes.bvh);
  EXPECT_EQ(entry->bytes.unread, 0u);

  // a bone turned
  ASSERT_FALSE(ribbon->bones.empty());
  const rs::EntityId bone = feed.entityOf(ribbon->bones[0]);
  const auto         turn = [&](float degrees) {
    rs::Transform t = scene.entity(bone)->local;
    t.rotation      = glm::angleAxis(glm::radians(degrees), glm::vec3(0, 0, 1)) * t.rotation;
    scene.setLocal(bone, t);
  };
  turn(40);
  frames(1);
  check("posed");
  const std::vector<glm::vec3> posed = drawnNow();
  EXPECT_GT(farthest(posed, rest), 1e-3f);

  // a dial: its cage moves, in the same record, and the drawn vertices follow
  for(size_t v = 0; v < cage.size() / 3; v++)
    cage[v * 3 + 1] += 0.5f;
  feed.sync(graph, root);
  EXPECT_EQ(feed.meshOf(mi), id);
  frames(1);
  check("dialled");
  EXPECT_GT(farthest(drawnNow(), posed), 1e-3f);

  // settled and woken
  const std::vector<glm::vec3> dialled = drawnNow();
  residency.settle(id);
  frames(4);
  ASSERT_TRUE(residency.isSettled(id));
  EXPECT_EQ(farthest(drawnNow(), dialled), 0.f);
  turn(-20);
  frames(1);
  EXPECT_FALSE(residency.isSettled(id));
  check("woken");
  EXPECT_GT(farthest(drawnNow(), dialled), 1e-3f);

  // an export makes the skin it needs
  const fv::build::BuildResult built = fv::build::buildGltf(graph, root, {});
  for(const std::string& p : fv::build::checkBuild(graph, built, {}))
    ADD_FAILURE() << p;
  ASSERT_GE(built.meshOf[size_t(mi)], 0);
  EXPECT_TRUE(built.model.meshes[size_t(built.meshOf[size_t(mi)])].primitives.at(0).attributes.count("JOINTS_0"));

  // what gives light keeps its triangles: that takes a sync, and letting go of the light takes them away again
  std::vector<int> materials(graph.materials.size());
  for(size_t i = 0; i < materials.size(); i++)
    materials[i] = int(i);
  for(fv::Material& material : graph.materials)
    material.edits.emissive = fv::mat::Rgb{1, 1, 1};
  EXPECT_FALSE(feed.syncMaterials(graph, materials));
  feed.sync(graph, root);
  frames(1);
  const rs::MeshRecord* emitting = scene.mesh(feed.meshOf(mi));
  ASSERT_TRUE(emitting);
  EXPECT_EQ(emitting->thinVertices, 0u);
  EXPECT_EQ(emitting->positions.size(), n);
  EXPECT_EQ(emitting->joints.size(), n);
  EXPECT_GT(residency.emissiveCount(), 0u);
  check("giving light");
  for(fv::Material& material : graph.materials)
    material.edits.emissive = std::nullopt;
  EXPECT_FALSE(feed.syncMaterials(graph, materials));
  feed.sync(graph, root);
  frames(1);
  EXPECT_EQ(scene.mesh(feed.meshOf(mi))->thinVertices, n);
  EXPECT_EQ(residency.emissiveCount(), 0u);
  check("not any more");

  // thin again, it lets go again; an export makes them again for the file
  EXPECT_TRUE(geo.drawnLetGo);
  const fv::build::BuildResult again = fv::build::buildGltf(graph, root, {});
  EXPECT_FALSE(geo.drawnLetGo);
  for(const std::string& p : fv::build::checkBuild(graph, again, {}))
    ADD_FAILURE() << p;
  for(size_t i = 0; i < positions.size(); i++)
    ASSERT_NEAR(geo.position[i], positions[i], 1e-4f) << i;
  // and the next sync lets go of them again, its record as it was
  const rs::MeshId kept = feed.meshOf(mi);
  feed.sync(graph, root);
  EXPECT_TRUE(geo.drawnLetGo);
  EXPECT_EQ(feed.meshOf(mi), kept);
  frames(1);
  check("let go again");
}

// A procedural layer's first coming takes a sync, which gives the meshes wearing it their masks. Taken off,
// its material goes back to what it was and the mesh keeps them, so putting it back makes nothing again.
TEST(FvPropFeed, AProceduralLayerGivesItsMeshMasksAndTheyStay)
{
  const auto         box = loadBox();
  rs::RetainedScene  scene;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu);
  fv::FigureFeed     feed(scene, {.cageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); }});
  fv::SceneGraph&    graph = box->loaded.scene;
  const int          root  = box->loaded.root;
  feed.sync(graph, root);
  residency.replace(scene);

  std::vector<int> materials(graph.materials.size());
  for(size_t i = 0; i < materials.size(); i++)
    materials[i] = int(i);
  const auto masked = [&] {
    size_t n = 0;
    for(size_t mi = 0; mi < graph.meshes.size(); mi++)
      if(const rs::MeshRecord* r = scene.mesh(feed.meshOf(int(mi))); r && !r->masks.empty())
      {
        EXPECT_EQ(r->masks.size(), r->vertexCount()) << mi;
        n++;
      }
    return n;
  };
  ASSERT_EQ(masked(), 0u) << "a figure wearing none pays for none";

  for(fv::Material& m : graph.materials)
  {
    m.surface.kind = fv::mat::Kind::eSkin;
    m.edits.sweat  = 0.5f;
  }
  EXPECT_FALSE(feed.syncMaterials(graph, materials)) << "its meshes need their masks";
  feed.sync(graph, root);
  residency.replace(scene);
  EXPECT_GT(masked(), 0u);
  const rs::MaterialRecord* worn = scene.material(feed.materialOf(0));
  ASSERT_TRUE(worn);
  ASSERT_TRUE(worn->payload.procedural);
  EXPECT_FLOAT_EQ(worn->payload.procedural->amount, 0.5f);
  for(const std::string& p : rs::checkResidency(scene, residency, gpu))
    ADD_FAILURE() << p;

  // the amount moved: the materials alone
  for(fv::Material& m : graph.materials)
    m.edits.sweat = 0.8f;
  EXPECT_TRUE(feed.syncMaterials(graph, materials));

  const rs::MeshId kept = feed.meshOf(0);
  for(fv::Material& m : graph.materials)
    m.edits.sweat = std::nullopt;
  EXPECT_TRUE(feed.syncMaterials(graph, materials));
  feed.sync(graph, root);
  residency.replace(scene);
  EXPECT_EQ(feed.meshOf(0), kept);
  EXPECT_GT(masked(), 0u);
  EXPECT_FALSE(scene.material(feed.materialOf(0))->payload.procedural);
  for(const std::string& p : rs::checkResidency(scene, residency, gpu))
    ADD_FAILURE() << p;
}

TEST(FvPropFeed, RainsWaterGoesOnTheSkinsRecordsInTheUvsAsFed)
{
  const auto         box = loadBox();
  rs::RetainedScene  scene;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu);
  fv::FigureFeed     feed(scene, {.cageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); }});
  fv::SceneGraph&    graph = box->loaded.scene;
  const int          root  = box->loaded.root;
  // the first mesh's first surface is skin, everything else not
  for(fv::Material& m : graph.materials)
    m.surface.kind = fv::mat::Kind::eOther;
  ASSERT_FALSE(graph.meshes[0].materials.empty());
  const int skin                     = graph.meshes[0].materials[0];
  graph.materials[size_t(skin)].surface.kind = fv::mat::Kind::eSkin;
  feed.sync(graph, root);
  residency.replace(scene);
  ASSERT_FALSE(scene.features().has(nvvkgltf::SceneFeatureSet::eProcedural));

  // a bead on every group of the mesh, as the rain hands them over
  std::map<int, fv::mat::WaterLayer> layers;
  for(size_t g = 0; g < graph.meshes[0].materials.size(); g++)
    layers[int(g)] = {.items = {{.a = {0.25f, 0.2f}, .b = {0.25f, 0.2f}, .toTangent = {1, 0.1f, 0.2f, -1}, .radius = 1e-3f, .height = 7e-4f}}, .damp = 0.6f};
  feed.setWater(graph, 0, layers);
  const rs::MaterialRecord* wet = scene.material(feed.materialOf(skin));
  ASSERT_TRUE(wet && wet->water);
  EXPECT_FLOAT_EQ(wet->water->damp, 0.6f);
  // fed with v turned (the options' textures aren't flipped), so is the water
  ASSERT_EQ(wet->water->items.size(), 1u);
  EXPECT_FLOAT_EQ(wet->water->items[0].a[1], 0.8f);
  EXPECT_FLOAT_EQ(wet->water->items[0].toTangent[1], -0.1f);
  for(size_t g = 1; g < graph.meshes[0].materials.size(); g++)
    if(graph.meshes[0].materials[g] != skin)
      EXPECT_FALSE(scene.material(feed.materialOf(graph.meshes[0].materials[g]))->water) << "group " << g << " doesn't take water";
  // the shaders take the procedural layer's code for it, and the residency its table
  EXPECT_TRUE(scene.features().has(nvvkgltf::SceneFeatureSet::eProcedural));
  EXPECT_TRUE(scene.features().has(nvvkgltf::SceneFeatureSet::eClearcoat));
  EXPECT_TRUE(residency.sync(scene).changed);
  // the same again changes nothing; dry, it goes
  feed.setWater(graph, 0, layers);
  EXPECT_FALSE(scene.takeChanges().water);
  feed.setWater(graph, 0, {});
  EXPECT_FALSE(scene.material(feed.materialOf(skin))->water);
  EXPECT_FALSE(scene.features().has(nvvkgltf::SceneFeatureSet::eProcedural));
  residency.sync(scene);
  for(const std::string& p : rs::checkResidency(scene, residency, gpu))
    ADD_FAILURE() << p;
}

// An item in the scene comes down a step and back. All that lives on its cage stays, what it was drawn with goes
// (nothing holds the old mesh, or the memory wasn't given back), and what is drawn is what a wear at that level draws.
TEST(FvPropFeed, AWornItemComesDownAStepAndBack)
{
  const char* const shirt = "People/Test/Clothing/Box Shirt.duf";
  const auto        box = loadBox(), atLevelOne = loadBox();
  rs::RetainedScene  scene;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu);
  fv::FigureFeed     feed(scene, {.cageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); }});
  feed.sync(box->loaded.scene, box->loaded.root);
  residency.replace(scene);
  box->morphs->cageOnly = [&](int mesh) { return feed.refinesOnceFed(box->loaded.scene, mesh); };
  const auto frames = [&](const char* when) {
    for(int i = 0; i < 6; i++)
    {
      gpu.beginFrame();
      residency.sync(scene);
    }
    for(const std::string& p : rs::checkResidency(scene, residency, gpu))
      ADD_FAILURE() << when << ": " << p;
  };

  const int  mi   = int(box->loaded.scene.meshes.size());
  const auto worn = box->commitWear(box->prepareWear(shirt, 2), shirt);
  ASSERT_EQ(worn.items.size(), 1u);
  const int id = worn.items[0];
  box->morphs->set("Taper", 1.0);
  box->morphs->settled();
  box->morphs->takeChanges();
  feed.sync(box->loaded.scene, box->loaded.root);
  frames("worn");
  const fv::WornItem* item = nullptr;
  for(const fv::WornItem& w : box->items)
    if(w.id == id)
      item = &w;
  ASSERT_NE(item, nullptr);
  EXPECT_TRUE(fv::qualityOf(box->loaded.scene, item->meshes).subdivision == 2);

  std::weak_ptr<fv::mesh::BuiltMesh> old       = box->loaded.scene.meshes[size_t(mi)].geo;
  const size_t                       triangles = old.lock()->fullIndex.size();
  const std::vector<float>           cage      = box->morphs->morphedCage(mi);
  const rs::MeshId                   had       = feed.meshOf(mi);
  ASSERT_NE(cage, old.lock()->cagePositions) << "the shirt follows the dial";

  auto down = box->remakeInputs(id, {1, 1.f});
  ASSERT_NE(down, nullptr);
  fv::Figure::buildRemade(*down);
  ASSERT_TRUE(box->commitRemade(*down, 2));
  down.reset();
  EXPECT_EQ(box->loaded.scene.meshes[size_t(mi)].geo->fullIndex.size() * 4, triangles);
  EXPECT_EQ(item->reduced, "Subdivision 1");
  EXPECT_EQ(box->morphs->morphedCage(mi), cage);
  EXPECT_EQ(box->remakeInputs(id, {1, 1.f}), nullptr) << "nothing to do";

  feed.sync(box->loaded.scene, box->loaded.root);
  frames("down");
  EXPECT_NE(feed.meshOf(mi), had);
  EXPECT_TRUE(old.expired()) << "something still holds the mesh it was";

  // what an export reads, as a wear at that level with that dial would have it
  atLevelOne->morphs->set("Taper", 1.0);
  atLevelOne->morphs->settled();
  const int first = int(atLevelOne->loaded.scene.meshes.size());
  atLevelOne->commitWear(atLevelOne->prepareWear(shirt, 1), shirt);
  atLevelOne->morphs->settled();
  box->morphs->catchUp();
  box->morphs->takeChanges();
  const std::vector<float>& reference = atLevelOne->loaded.scene.meshes[size_t(first)].geo->position;
  const std::vector<float>& ours      = box->loaded.scene.meshes[size_t(mi)].geo->position;
  ASSERT_EQ(ours.size(), reference.size());
  for(size_t i = 0; i < ours.size(); i++)
    ASSERT_NEAR(ours[i], reference[i], 1e-3f) << i;

  // a cage that moved meanwhile: nothing changes, and it can be asked again
  auto up = box->remakeInputs(id, {2, 1.f});
  ASSERT_NE(up, nullptr);
  fv::Figure::buildRemade(*up);
  fv::mesh::BuiltMesh& now = *box->loaded.scene.meshes[size_t(mi)].geo;
  now.cagePositions[0] += 1;
  EXPECT_FALSE(box->commitRemade(*up, 2));
  now.cagePositions[0] -= 1;
  EXPECT_EQ(item->reduced, "Subdivision 1");
  ASSERT_TRUE(box->commitRemade(*up, 2));
  up.reset();
  EXPECT_EQ(box->loaded.scene.meshes[size_t(mi)].geo->fullIndex.size(), triangles);
  EXPECT_TRUE(item->reduced.empty());
  feed.sync(box->loaded.scene, box->loaded.root);
  frames("up");
  EXPECT_EQ(fv::checkFeed(box->loaded.scene, box->loaded.root, feed, scene, {}), std::vector<std::string>{});
}

// A wear's dials come before its first sync. What the GPU is about to refine moves its cage alone, as a fed mesh
// does: the CPU morphs nothing it doesn't draw, and holds nothing for it, until something asks.
TEST(FvPropFeed, AWearsDialsMoveOnlyItsCageBeforeItIsFed)
{
  const char* const shirt = "People/Test/Clothing/Box Shirt.duf";
  const auto        box = loadBox(), dialled = loadBox();
  box->morphs->set("Taper", 1.0);
  box->morphs->settled();
  dialled->morphs->set("Taper", 1.0);
  dialled->morphs->settled();

  rs::RetainedScene  scene;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu);
  fv::FigureFeed     feed(scene, {.cageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); }});
  feed.sync(box->loaded.scene, box->loaded.root);
  residency.replace(scene);
  box->morphs->cageOnly = [&](int mesh) { return feed.refinesOnceFed(box->loaded.scene, mesh); };

  const auto wear = [&](fv::Figure& f, std::vector<float>* came = nullptr) {
    const size_t first    = f.loaded.scene.meshes.size();
    auto         prepared = f.prepareWear(shirt, 1);
    if(came)
      *came = fv::Figure::preparedScene(*prepared).meshes.at(0).geo->position;
    f.commitWear(prepared, shirt);
    f.morphs->settled();
    EXPECT_EQ(f.loaded.scene.meshes.size(), first + 1);
    return int(first);
  };
  std::vector<float>         asItCame;
  const int                  mi      = wear(*box, &asItCame);
  const fv::mesh::BuiltMesh& worn    = *box->loaded.scene.meshes[size_t(mi)].geo;
  const fv::mesh::BuiltMesh& morphed = *dialled->loaded.scene.meshes[size_t(wear(*dialled))].geo;
  ASSERT_TRUE(fv::mesh::drawnFromCage(worn));
  ASSERT_NE(asItCame, morphed.position) << "the shirt follows the dial";

  EXPECT_FALSE(feed.refines(mi));
  EXPECT_TRUE(feed.refinesOnceFed(box->loaded.scene, mi));
  EXPECT_EQ(worn.position, asItCame);
  EXPECT_NE(box->morphs->morphedCage(mi), worn.cagePositions);

  // fed, it draws the dial all the same
  feed.sync(box->loaded.scene, box->loaded.root);
  EXPECT_TRUE(feed.refines(mi));
  gpu.beginFrame();
  residency.sync(scene);
  for(const std::string& p : rs::checkResidency(scene, residency, gpu))
    ADD_FAILURE() << p;
  EXPECT_EQ(worn.position, asItCame);
  const size_t           n = worn.position.size() / 3;
  std::vector<glm::vec3> drawn(n);
  std::memcpy(drawn.data(), gpu.bytesAt(gpu.bufferAddress(drawnBuffers(box->loaded.scene, feed, residency, mi).positions), n * 12).data(), n * 12);
  for(size_t v = 0; v < n; v++)
    ASSERT_LT(glm::distance(drawn[v], glm::make_vec3(&morphed.position[v * 3])), 1e-4f) << v;

  // and an export, which asks, finds what a CPU morph would have made
  box->morphs->catchUp();
  ASSERT_EQ(worn.position.size(), morphed.position.size());
  for(size_t i = 0; i < worn.position.size(); i++)
    ASSERT_NEAR(worn.position[i], morphed.position[i], 1e-5f) << i;
}

// A mesh's price, made before anything is fed, is what its buffers hold once resident: the residency's
// formats and scene_price.hpp stay together. The mesh's own streams are read by nothing, so aren't held.
TEST(FvPropFeed, AMeshsPriceIsWhatItHoldsOnceResident)
{
  const auto                box = loadBox();
  rs::RetainedScene         scene;
  rs::FakeGpuBackend        gpu(2);
  rs::Residency             residency(gpu);
  fv::PropFeed              feed(scene);
  std::vector<fv::PropSpec> props(1);
  props[0] = {.id = "p:1", .number = 1, .name = "Box", .figure = &box->loaded.scene, .figureRoot = box->loaded.root};
  props[0].figureCageOf = [&](int mesh, std::vector<float>& scratch) { return box->morphs->morphedCageView(mesh, scratch); };
  feed.sync(props);
  residency.replace(scene);
  const fv::FigureFeed&  figure = *feed.fed("p:1")->figure;
  const rs::MemoryLedger ledger = residency.ledger();
  size_t                 priced = 0;
  for(size_t mi = 0; mi < box->loaded.scene.meshes.size(); mi++)
  {
    const rs::MeshRecord* record = scene.mesh(figure.meshOf(int(mi)));
    if(!record)
      continue;
    const auto entry = std::find_if(ledger.meshes.begin(), ledger.meshes.end(), [&](const auto& m) { return m.mesh == figure.meshOf(int(mi)); });
    ASSERT_NE(entry, ledger.meshes.end()) << mi;
    rs::MeshCounts c = fv::meshCounts(box->loaded.scene.meshes[mi]);
    // what its materials asked, which the price can only guess
    c.tangents         = !record->tangents.empty();
    c.tangentTriangles = record->refine ? record->refine->tangentIndices.size() / 3 : 0;
    EXPECT_EQ(rs::meshBufferBytes(c), entry->bytes.total() - entry->bytes.bvh) << "mesh " << mi;
    EXPECT_EQ(entry->triangles, c.triangles) << mi;
    EXPECT_EQ(entry->prop, 1u);
    EXPECT_EQ(entry->local, int(mi));
    // what the record's tables read is the mesh's own, not a copy
    if(record->refine)
    {
      const auto& geo = *box->loaded.scene.meshes[mi].geo;
      EXPECT_EQ(static_cast<const void*>(record->refine->levels.at(0).columns.data()), static_cast<const void*>(geo.refinement->levels.at(0).col.data()));
      EXPECT_EQ(record->refine->triangles.data(), geo.renderTris.data());
      EXPECT_EQ(record->refine->owner, box->loaded.scene.meshes[mi].geo);
    }
    priced++;
  }
  EXPECT_GT(priced, 0u);
  EXPECT_EQ(ledger.total.unread, 0u);
}

// A strand's segments are drawn as two each, split at their middles: a midpoint is refined halfway
// between its segment's ends (so a pose moves it with them), halfway along the strand, and its
// ribbon lies across the strand between its ends' directions.
TEST(FvPropFeed, EachStrandSegmentIsDrawnAsTwo)
{
  fv::mesh::MeshInput input;
  fv::mesh::Lines     lines;
  lines.start    = {0, 3};
  lines.verts    = {0, 1, 2};
  lines.material = {0};
  input.positions = {0, 0, 0, 0, 2, 0, 1, 4, 0};
  input.lines     = lines;
  const fv::mesh::BuiltMesh geo = fv::mesh::build(input);
  ASSERT_TRUE(geo.ribbon && geo.refinement);
  const fv::subdiv::Refinement& r = *geo.refinement;
  EXPECT_EQ(r.vertexCount, (3 + 2) * 2);  // three points and two midpoints, each two drawn vertices
  EXPECT_EQ(geo.renderTris.size(), 4u * 2u * 3u);  // two segments, two quads each
  const fv::subdiv::Stencils& s = r.levels.back();
  for(int mid = 3; mid < 5; mid++)
    for(int k = 0; k < 2; k++)
    {
      const int row = mid * 2 + k;
      ASSERT_EQ(s.rowStart[row + 1] - s.rowStart[row], 2);
      EXPECT_EQ(s.col[s.rowStart[row]], mid - 3);
      EXPECT_EQ(s.col[s.rowStart[row] + 1], mid - 2);
      EXPECT_FLOAT_EQ(s.weight[s.rowStart[row]], 0.5f);
      EXPECT_FLOAT_EQ(s.weight[s.rowStart[row] + 1], 0.5f);
    }
  // along the strand, root to tip, every quarter
  std::vector<float> along(geo.ribbon->along);
  std::sort(along.begin(), along.end());
  along.erase(std::unique(along.begin(), along.end()), along.end());
  EXPECT_EQ(along, (std::vector<float>{0.f, 0.25f, 0.5f, 0.75f, 1.f}));
  // each drawn vertex's direction across is a unit vector
  for(size_t v = 0; v * 3 < geo.ribbon->side.size(); v++)
    EXPECT_NEAR(std::hypot(geo.ribbon->side[v * 3], geo.ribbon->side[v * 3 + 1], geo.ribbon->side[v * 3 + 2]), 1.f, 1e-5f);
}

// PR hairs: a material that grows three per guide draws each of its strands four times, the copies
// sitting off the guide within the radius, across and out from the head (never in), closer by the tip
// separation at the tip, and following the guide's points (one stencil column each, so its skin and
// morphs). A material that grows none draws its strands once, and the same seed grows the same hairs.
TEST(FvPropFeed, AStrandGrowsItsPreRenderHairsAroundIt)
{
  fv::mesh::MeshInput input;
  fv::mesh::Lines     lines;
  // two strands hanging from a head centred on the origin: one on its right (material 0), one on
  // its left (material 1)
  lines.start    = {0, 3, 6};
  lines.verts    = {0, 1, 2, 3, 4, 5};
  lines.material = {0, 1};
  lines.grow     = {{.perGuide = 3, .radius = 0.3f, .tipSeparation = -0.85f, .seed = 7}, {}};
  input.positions = {10, 0, 0, 10, -5, 0, 10, -10, 0, -10, 0, 0, -10, -5, 0, -10, -10, 0};
  input.materialCount = 2;
  input.lines         = lines;
  const fv::mesh::BuiltMesh geo = fv::mesh::build(input);
  ASSERT_TRUE(geo.ribbon && geo.refinement);
  const auto& rb = *geo.ribbon;
  ASSERT_EQ(rb.shift.size(), geo.position.size());
  // five strands: three points and two midpoints each, two drawn vertices apiece
  EXPECT_EQ(geo.refinement->vertexCount, 5 * (3 + 2) * 2);

  // each drawn vertex at its stencil's single point plus its shift (no width yet)
  const fv::subdiv::Stencils& s = geo.refinement->levels.back();
  int                         grown = 0, onLeft = 0;
  for(size_t i = 0; i < geo.renderIndex.size(); i++)
  {
    const glm::vec3 shift = glm::make_vec3(&rb.shift[i * 3]);
    const int       row   = int(geo.renderIndex[i]);
    if(s.rowStart[row + 1] - s.rowStart[row] == 1)
    {
      const int point = s.col[s.rowStart[row]];
      EXPECT_LT(glm::distance(glm::make_vec3(&geo.position[i * 3]), glm::make_vec3(&input.positions[point * 3]) + shift), 1e-5f);
      if(point >= 3)
        onLeft++;
    }
    if(shift == glm::vec3(0.f))
      continue;
    // material 0's copies, within the radius, across the strand (it runs along y) and outward (+x)
    EXPECT_EQ(rb.material[i], 0);
    EXPECT_LE(glm::length(shift), 0.3f + 1e-5f);
    EXPECT_NEAR(shift.y, 0.f, 1e-5f);
    EXPECT_GE(shift.x, -1e-6f);
    if(rb.along[i] == 1.f)
      grown++;
  }
  // three grown tips, each drawn twice
  EXPECT_EQ(grown, 3 * 2);
  // the strand on the left is drawn once: its three points' six drawn vertices
  EXPECT_EQ(onLeft, 6);

  // a copy's tip sits 0.15 as far off the guide as its root, the same way
  for(int line = 0; line < 3; line++)
  {
    // copy `line`'s points follow the guides' six: 6 + 3 * line is its root, two on its tip
    const int  root = 6 + 3 * line, tip = root + 2;
    glm::vec3  r(0.f), t(0.f);
    for(size_t i = 0; i < geo.renderIndex.size(); i++)
    {
      if(int(geo.renderIndex[i]) == root * 2)
        r = glm::make_vec3(&rb.shift[i * 3]);
      if(int(geo.renderIndex[i]) == tip * 2)
        t = glm::make_vec3(&rb.shift[i * 3]);
    }
    EXPECT_GT(glm::length(r), 0.f) << line;
    EXPECT_LT(glm::distance(t, r * 0.15f), 1e-5f) << line;
  }

  const fv::mesh::BuiltMesh again = fv::mesh::build(input);
  EXPECT_EQ(again.ribbon->shift, rb.shift);
}

// A density's hairs: a square of 10 x 10 cm with a guide of four points hanging from each corner of
// its left half (x 0..5) and from one corner on the right, so only the left polygon has guides all round.
static fv::mesh::MeshInput hairOnASurface(const fv::mesh::Growth& grow)
{
  fv::mesh::MeshInput input;
  fv::mesh::Lines     lines;
  const float         roots[4][2] = {{0, 0}, {5, 0}, {5, 10}, {0, 10}};
  for(int g = 0; g < 4; g++)
  {
    lines.start.push_back(g * 4);
    lines.material.push_back(0);
    for(int p = 0; p < 4; p++)
    {
      input.positions.insert(input.positions.end(), {roots[g][0], -3.f * float(p), roots[g][1]});
      lines.verts.push_back(g * 4 + p);
    }
  }
  lines.start.push_back(16);
  lines.grow = {grow};
  auto surface       = std::make_shared<fv::mesh::GrowthSurface>();
  surface->positions = {0, 0, 0, 5, 0, 0, 5, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0, 10};
  surface->polyStart = {0, 4, 8};
  surface->polyVerts = {0, 1, 2, 3, 1, 4, 5, 2};
  lines.surface      = surface;
  input.lines        = lines;
  return input;
}

// Each drawn strand's root (the drawn vertex with `along` 0 of each new line), with its shift.
static std::vector<glm::vec3> grownRoots(const fv::mesh::BuiltMesh& geo, size_t guidePoints)
{
  std::vector<glm::vec3> out;
  for(size_t i = 0; i < geo.renderIndex.size(); i++)
    if(geo.ribbon->along[i] == 0.f && (geo.renderIndex[i] & 1) == 0 && (geo.renderIndex[i] >> 1) >= guidePoints)
      out.push_back(glm::make_vec3(&geo.position[i * 3]));
  return out;
}

// Hairs per cm² over the polygons with guides all round, and none on a polygon that lacks one; the
// guides stay, and the same seed grows the same hairs.
TEST(FvPropFeed, ADensityGrowsOverThePolygonsWithGuidesAllRound)
{
  const fv::mesh::MeshInput input = hairOnASurface({.seed = 7, .density = 20, .singleGuide = 1, .singleBase = 1, .singleTip = 1});
  const fv::mesh::BuiltMesh geo   = fv::mesh::build(input);
  ASSERT_TRUE(geo.ribbon);
  const auto roots = grownRoots(geo, 16);
  // 50 cm² at 20 per cm², give or take a hair for each of its two triangles
  EXPECT_NEAR(double(roots.size()), 1000.0, 2.0);
  float xMax = 0;
  for(const glm::vec3& r : roots)
  {
    EXPECT_GE(r.x, -1e-4f);
    EXPECT_LE(r.x, 5.f + 1e-4f);
    EXPECT_GE(r.z, -1e-4f);
    EXPECT_LE(r.z, 10.f + 1e-4f);
    EXPECT_NEAR(r.y, 0.f, 1e-4f);
    xMax = std::max(xMax, r.x);
  }
  // it reaches across the polygon
  EXPECT_GT(xMax, 4.5f);
  EXPECT_EQ(fv::mesh::build(input).ribbon->shift, geo.ribbon->shift);

  fv::mesh::Growth other{.seed = 8, .density = 20};
  EXPECT_NE(fv::mesh::build(hairOnASurface(other)).ribbon->shift, geo.ribbon->shift);
}

// A hair is a guide's points (one stencil column each, so its skin and morphs are the guide's) moved
// off it: all the way along by where its root is when it follows one guide.
TEST(FvPropFeed, ASingleGuideHairRunsAlongsideItsGuide)
{
  const fv::mesh::BuiltMesh geo = fv::mesh::build(hairOnASurface({.seed = 7, .density = 2, .singleGuide = 1, .singleBase = 1, .singleTip = 1}));
  const fv::mesh::Ribbon&   rb  = *geo.ribbon;
  const fv::subdiv::Stencils& s     = geo.refinement->levels.back();
  const int                   hairs = int(grownRoots(geo, 16).size());
  ASSERT_GT(hairs, 10);
  for(int h = 0; h < hairs; h++)
  {
    const glm::vec3 root = glm::make_vec3(&rb.shift[size_t(geo.firstSplit[size_t(16 + h * 4) * 2]) * 3]);
    // straight down from a root at y = 0: the offset lies in the surface, the same at every height
    EXPECT_NEAR(root.y, 0.f, 1e-4f);
    for(int q = 0; q < 4; q++)
    {
      const int row = (16 + h * 4 + q) * 2;
      EXPECT_EQ(s.rowStart[row + 1] - s.rowStart[row], 1);
      EXPECT_LT(glm::distance(glm::make_vec3(&rb.shift[size_t(geo.firstSplit[row]) * 3]), root), 1e-4f);
    }
  }
}

// With no single guide, a hair lies between the guides around it: here they lean apart, so the hair
// at a polygon's centre stays near the middle of them all the way down.
TEST(FvPropFeed, AHairBetweenGuidesFollowsThemAll)
{
  fv::mesh::MeshInput input = hairOnASurface({.seed = 3, .density = 4});
  // lean the guides at x = 0 further out as they go, and the others the other way
  for(int g = 0; g < 4; g++)
    for(int p = 0; p < 4; p++)
      input.positions[(g * 4 + p) * 3] += (g == 0 || g == 3 ? -1.f : 1.f) * float(p);
  const fv::mesh::BuiltMesh geo = fv::mesh::build(input);
  // a hair is four points here, as its guide is: the refined vertices after the guides', four to a
  // hair and two to a point, then the segments' midpoints
  const int hairs = int(grownRoots(geo, 16).size());
  ASSERT_GT(hairs, 100);
  for(int h = 0; h < hairs; h++)
  {
    const int root = geo.firstSplit[size_t(16 + h * 4) * 2], tip = geo.firstSplit[size_t(16 + h * 4 + 3) * 2];
    ASSERT_GE(root, 0);
    ASSERT_GE(tip, 0);
    const float x = geo.position[size_t(root) * 3];
    // the lean runs from -3 at x = 0 to +3 at x = 5, in step with where the root is
    EXPECT_NEAR(geo.position[size_t(tip) * 3], x + (x / 5.f * 2.f - 1.f) * 3.f, 1e-3f);
  }
}

// A hair is up to the random length shorter, drawn with as many points as its length needs and at
// least the file's count, and its frizz and scraggle leave its root on the surface but move the rest.
TEST(FvPropFeed, FrizzScraggleAndLengthLeaveTheRootInPlace)
{
  fv::mesh::Growth grow{.seed = 5, .density = 2, .points = 3, .segment = 1, .singleGuide = 1, .singleBase = 1, .singleTip = 1};
  grow.randomLength           = 4;
  grow.frizzBase              = 0.5f;
  grow.frizzTip               = 0.5f;
  grow.scraggle[0]            = {.amount = 1, .frequency = 0.5f};
  const fv::mesh::BuiltMesh geo = fv::mesh::build(hairOnASurface(grow));
  const fv::mesh::Ribbon&   rb  = *geo.ribbon;
  float                     deepest = 0, shallowest = 100, moved = 0;
  for(size_t i = 0; i < geo.renderIndex.size(); i++)
  {
    if((geo.renderIndex[i] >> 1) < 16)
      continue;
    const float y = geo.position[i * 3 + 1];
    if(rb.along[i] == 0.f)
      EXPECT_NEAR(y, 0.f, 1e-4f);
    if(rb.along[i] == 1.f)
    {
      deepest    = std::max(deepest, -y);
      shallowest = std::min(shallowest, -y);
    }
    moved = std::max(moved, std::abs(rb.shift[i * 3 + 1]));
  }
  // 9 cm guides, up to 4 cm shorter (a tip lands on a guide's point, so 3 or 6 or 9)
  EXPECT_GT(deepest, 7.f);
  EXPECT_LT(shallowest, 7.f);
  EXPECT_GT(moved, 0.1f);
}

// Past a machine's worth of hairs they thin out rather than going on.
TEST(FvPropFeed, ADensityStopsSomewhere)
{
  const fv::mesh::BuiltMesh geo = fv::mesh::build(hairOnASurface({.seed = 1, .density = 1000000, .points = 2}));
  EXPECT_LE(grownRoots(geo, 16).size(), 500002u);
}

// Strand ribbons (gpu-figures-plan.md F1): each drawn vertex is its strand point moved across the
// strand by its width, which the GPU adds after copying the points; the residency's streams are the
// CPU's, as fed and with the strands moved as a morph moves them.
static void strandRibbonsAreRefinedFromTheirPoints(std::vector<fv::mesh::Growth> grow)
{
  // three strands of four points, rising and leaning from a row of roots
  fv::mesh::MeshInput input;
  fv::mesh::Lines     lines;
  lines.grow = std::move(grow);
  for(int s = 0; s < 3; s++)
  {
    lines.start.push_back(s * 4);
    lines.material.push_back(0);
    for(int p = 0; p < 4; p++)
    {
      input.positions.insert(input.positions.end(), {float(s), 2.f * float(p), 0.3f * float(p * s)});
      lines.verts.push_back(s * 4 + p);
    }
  }
  lines.start.push_back(12);
  input.lines = lines;
  auto geo    = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(input));
  ASSERT_TRUE(geo->ribbon && geo->refinement);
  fv::mesh::widenRibbons(*geo, {{0.4, 0.2}});

  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{.name = "Hair"});
  scene.materials.resize(1);
  scene.materials[0].name = "Strands";
  fv::Mesh m;
  m.name      = "hair";
  m.geo       = geo;
  m.materials = {0};
  scene.meshes.push_back(m);
  scene.add(fv::Node{.name = "Hair strands", .kind = fv::NodeKind::eMesh, .mesh = 0}, root);

  rs::RetainedScene  store;
  rs::FakeGpuBackend gpu(2);
  rs::Residency      residency(gpu);
  std::vector<float> cage = geo->cagePositions;
  fv::FigureFeed     feed(store, {.cageOf = [&](int, std::vector<float>&) { return std::span<const float>(cage); }});
  feed.sync(scene, root);
  ASSERT_TRUE(feed.refines(0));
  residency.replace(store);
  const auto check = [&](const char* when) {
    for(const std::string& p : rs::checkResidency(store, residency, gpu))
      ADD_FAILURE() << when << ": " << p;
    const rs::Residency::MeshGpu* mesh = residency.meshGpu(feed.meshOf(0));
    ASSERT_TRUE(mesh && mesh->refine) << when;
    const size_t           n = geo->position.size() / 3;
    std::vector<glm::vec3> positions(n);
    std::memcpy(positions.data(), gpu.bytesAt(gpu.bufferAddress(mesh->positions), n * 12).data(), n * 12);
    const std::vector<glm::vec3> normals = rs::streamNormals(gpu.bytesAt(gpu.bufferAddress(mesh->normals), n * 4));
    for(size_t v = 0; v < n; v++)
    {
      ASSERT_LT(glm::distance(positions[v], glm::make_vec3(&geo->position[v * 3])), 1e-5f) << when << " vertex " << v;
      // a root opens from no width: its triangles have no area, and its normal is zero both ways
      const glm::vec3 want = glm::make_vec3(&geo->normal[v * 3]);
      if(want == glm::vec3(0.f))
        ASSERT_EQ(normals[v], glm::vec3(0.f)) << when << " vertex " << v;
      else
        ASSERT_GT(glm::dot(normals[v], want), 0.9999f) << when << " vertex " << v;
    }
  };
  check("fed");

  fv::mesh::CageDeltas moved;
  for(int v = 0; v < 12; v++)
  {
    moved.verts.push_back(v);
    moved.deltas.insert(moved.deltas.end(), {0.1f * float(v), 0.f, -0.05f * float(v % 4)});
  }
  const fv::mesh::RenderTarget        target = fv::mesh::target(*geo, moved);
  const fv::mesh::RenderTarget::Drawn& t      = target.drawn(*geo);
  for(size_t i = 0; i < t.indices.size(); i++)
    for(int k = 0; k < 3; k++)
      geo->position[t.indices[i] * 3 + k] += t.deltas[i * 3 + k];
  fv::mesh::NormalWelder(geo->renderIndex, geo->firstSplit, geo->nextSplit, geo->renderTris, geo->renderVertexCount)
      .full(geo->position, geo->normal);
  for(size_t i = 0; i < moved.verts.size(); i++)
    for(int k = 0; k < 3; k++)
      cage[size_t(moved.verts[i]) * 3 + k] += moved.deltas[i * 3 + k];
  const int mesh = 0;
  EXPECT_TRUE(feed.syncCages(scene, {&mesh, 1}));
  gpu.beginFrame();
  residency.sync(store);
  check("moved");
}

TEST(FvPropFeed, StrandRibbonsAreRefinedFromTheirPoints)
{
  strandRibbonsAreRefinedFromTheirPoints({});
}

// PR hairs grown on the strands: the GPU adds each one's offset from its guide with its width.
TEST(FvPropFeed, GrownStrandsAreRefinedFromTheirGuidesPoints)
{
  strandRibbonsAreRefinedFromTheirPoints({{.perGuide = 2, .radius = 0.3f, .tipSeparation = -0.85f, .seed = 7}});
}

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
