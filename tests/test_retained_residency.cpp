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

// The retained scene on the fake GPU (src/scene_residency.hpp): every sync checked by
// ResidencyOracle, and incremental syncs held to a fresh one of the same scene (oracle 2b of
// daz-viewer docs/retained-scene-plan.md).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>

#include <glm/gtc/quaternion.hpp>

#include "common/fake_gpu_backend.hpp"
#include "common/residency_oracle.hpp"
#include "scene_residency.hpp"
#include "scene_retained.hpp"

using namespace rs;

namespace {

// A grid of n x n quads at height z, its triangles split into `groups` submeshes (material groups)
MeshRecord gridMesh(int n, float z, int groups = 1)
{
  MeshRecord m;
  m.name = "grid";
  for(int y = 0; y <= n; y++)
    for(int x = 0; x <= n; x++)
    {
      m.positions.push_back({float(x), float(y), z});
      m.normals.push_back({0, 0, 1});
      m.uv0.push_back({float(x) / n, float(y) / n});
    }
  std::vector<std::vector<uint32_t>> runs(static_cast<size_t>(groups));
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      const uint32_t a = uint32_t(y * (n + 1) + x), b = a + 1, c = a + uint32_t(n) + 1, d = c + 1;
      auto&          run = runs[size_t((y * n + x) % groups)];
      run.insert(run.end(), {a, b, d, a, d, c});
    }
  for(int g = 0; g < groups; g++)
  {
    m.submeshes.push_back(
        {.firstIndex = uint32_t(m.indices.size()), .indexCount = uint32_t(runs[size_t(g)].size()), .material = uint32_t(g), .order = g});
    m.indices.insert(m.indices.end(), runs[size_t(g)].begin(), runs[size_t(g)].end());
  }
  return m;
}

MaterialId material(RetainedScene& scene, const char* name)
{
  MaterialRecord m;
  m.payload.name = name;
  return scene.addMaterial(std::move(m));
}

// The payload with KHR_materials_transmission (factor 1), a feature the shader can compile out
fv::mat::Material withTransmission(fv::mat::Material payload)
{
  payload.transmission = fv::mat::Transmission{.factor = 1};
  return payload;
}

// A texture table over three image files: two samplers of the first, and a second image
std::shared_ptr<const TextureTable> textureTable(RetainedScene& scene)
{
  TextureRecord a, b;
  a.uri              = "a.png";
  a.width            = 4;
  b.uri              = "b.png";
  b.width            = 8;
  b.component        = 2;  // a two-channel normal map
  const TextureId ta = scene.addTexture(a), tb = scene.addTexture(b);
  auto            table = std::make_shared<TextureTable>();
  table->push_back({ta, {}});
  SamplerDesc nearest;
  nearest.magFilter = TINYGLTF_TEXTURE_FILTER_NEAREST;
  table->push_back({ta, nearest});
  table->push_back({tb, {}});
  return table;  // the add references stay the test's: a table names its textures but holds none
}

// base colour from table entry `base`, a normal map from entry 2
MaterialId textured(RetainedScene& scene, std::shared_ptr<const TextureTable> table, int base, fv::mat::AlphaMode alpha = fv::mat::AlphaMode::eOpaque)
{
  MaterialRecord m;
  m.payload.name                                        = "textured";
  m.payload.alphaMode                                   = alpha;
  m.payload.baseColorMap.index = base;
  m.payload.normalMap.index                         = 2;
  m.textures                                            = std::move(table);
  return scene.addMaterial(std::move(m));
}

MaterialId emissive(RetainedScene& scene, float strength)
{
  MaterialRecord m;
  m.payload.name           = "lamp";
  m.payload.emissive = {strength, strength, strength};
  return scene.addMaterial(std::move(m));
}

EntityId lightEntity(RetainedScene& scene, const char* type, double intensity, EntityId parent = {})
{
  LightRecord l;
  l.payload.type       = type;
  l.payload.intensity  = intensity;
  const LightId  light = scene.addLight(l);
  const EntityId e     = scene.createEntity(type, {}, parent);
  scene.setEntityLight(e, light);
  return e;
}

// An entity drawing `mesh` with one material per submesh
EntityId drawn(RetainedScene& scene, const char* name, MeshId mesh, std::vector<MaterialId> materials, EntityId parent = {})
{
  const EntityId e = scene.createEntity(name, {}, parent);
  MeshInstance   instance;
  instance.parts.push_back({mesh, std::move(materials)});
  scene.setMeshInstance(e, instance);
  return e;
}

void expectClean(RetainedScene& scene, const Residency& residency, const FakeGpuBackend& gpu, const std::string& when = "")
{
  for(const std::string& p : checkResidency(scene, residency, gpu))
    ADD_FAILURE() << when << p;
}

// frames until every retire has run
void drain(FakeGpuBackend& gpu)
{
  for(uint32_t i = 0; i < gpu.framesInFlight(); i++)
    gpu.beginFrame();
}

}  // namespace

TEST(RetainedResidency, AFreshSceneNumbersItsInstancesInDocumentOrder)
{
  RetainedScene  scene;
  const MeshId   grid  = scene.addMesh(gridMesh(2, 0.f, 2));
  const MeshId   plate = scene.addMesh(gridMesh(1, 1.f));
  MaterialId     a = material(scene, "a"), b = material(scene, "b");
  const EntityId root = drawn(scene, "root", grid, {a, b});
  drawn(scene, "child", plate, {a}, root);
  drawn(scene, "sibling", grid, {b, a});  // the same mesh again: its prims are shared

  FakeGpuBackend   gpu;
  Residency        residency(gpu);
  const SyncResult r = residency.replace(scene);
  EXPECT_TRUE(r.changed && r.dlssReset && r.drawListChanged);
  expectClean(scene, residency, gpu);

  const std::vector<DrawItem> items = scene.drawItems();
  ASSERT_EQ(items.size(), 5u);
  ASSERT_EQ(residency.instanceHighWater(), 5u);
  for(uint32_t s = 0; s < items.size(); s++)
  {
    const Residency::Instance* inst = residency.instance(s);
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->entity, items[s].entity);
    EXPECT_EQ(inst->submesh, items[s].submesh);
  }
  EXPECT_EQ(residency.primHighWater(), 3u);  // the grid's two submeshes and the plate
  EXPECT_EQ(residency.instanceCapacity(), 256u);
  EXPECT_EQ(gpu.waitIdleCount(), 0u);
}

TEST(RetainedResidency, NothingChangedIsNoWork)
{
  RetainedScene scene;
  drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f)), {material(scene, "m")});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  const size_t barriers = gpu.barriers().size();
  gpu.beginFrame();
  const SyncResult r = residency.sync(scene);
  EXPECT_FALSE(r.changed);
  EXPECT_EQ(gpu.barriers().size(), barriers);
}

TEST(RetainedResidency, HidingNullsTheTlasRowAndKeepsTheSlot)
{
  RetainedScene  scene;
  const EntityId e = drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f)), {material(scene, "m")});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);

  scene.setVisible(e, false);
  gpu.beginFrame();
  SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.changed && r.visibilityChanged);
  EXPECT_FALSE(r.drawListChanged);
  expectClean(scene, residency, gpu);
  EXPECT_FALSE(gpu.tlas(residency.tlas())->rows[0].blas);
  EXPECT_NE(residency.instance(0), nullptr);

  scene.setVisible(e, true);
  gpu.beginFrame();
  r = residency.sync(scene);
  EXPECT_TRUE(r.visibilityChanged);
  expectClean(scene, residency, gpu);
}

TEST(RetainedResidency, AMoveRewritesTheRowsOfWhatMoved)
{
  RetainedScene  scene;
  const MeshId   grid   = scene.addMesh(gridMesh(2, 0.f));
  MaterialId     m      = material(scene, "m");
  const EntityId parent = drawn(scene, "parent", grid, {m});
  drawn(scene, "child", grid, {m}, parent);
  drawn(scene, "still", grid, {m});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);

  Transform t;
  t.translation = {1.f, 2.f, 3.f};
  scene.setLocal(parent, t);
  gpu.beginFrame();
  const SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.changed);
  EXPECT_FALSE(r.drawListChanged);
  EXPECT_FALSE(r.dlssReset);  // motion vectors describe a move
  expectClean(scene, residency, gpu);
  EXPECT_EQ(gpu.tlas(residency.tlas())->rows[1].transform[0][3], 1.f);  // the child moved with its parent
  EXPECT_EQ(gpu.tlas(residency.tlas())->rows[2].transform[0][3], 0.f);
}

// A destroyed entity leaves holes; their slots come back only a frame cycle later, and the meshes and
// prims nothing draws any more are freed then.
TEST(RetainedResidency, ARemovalLeavesHolesThatAreReusedAfterTheirRetireFrame)
{
  RetainedScene  scene;
  MaterialId     m     = material(scene, "m");
  const EntityId first = drawn(scene, "first", scene.addMesh(gridMesh(2, 0.f)), {m});
  drawn(scene, "second", scene.addMesh(gridMesh(3, 1.f)), {m});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  drain(gpu);

  scene.destroyEntity(first);
  gpu.beginFrame();
  SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.drawListChanged);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.instance(0), nullptr);  // a hole
  EXPECT_EQ(residency.instanceHighWater(), 2u);

  // added in the same cycle: slot 0 is not back yet
  const MeshId   again = scene.addMesh(gridMesh(1, 2.f));
  const EntityId third = drawn(scene, "third", again, {m});
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.instanceHighWater(), 3u);
  EXPECT_EQ(residency.instance(2)->entity, third);

  // a cycle later slot 0 is free again, and the first mesh's buffers are gone
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
  drawn(scene, "fourth", again, {m});
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu);
  ASSERT_NE(residency.instance(0), nullptr);
  EXPECT_EQ(residency.instanceHighWater(), 3u);
  EXPECT_EQ(gpu.waitIdleCount(), 0u);
}

// What residencyRetireOnReplace does to a stream host's scene replace: the store's every entity
// destroyed and the scene fed again under new handles, synced into the tables the residency has.
// The old slots are holes until their retire frame, the new scene takes others meanwhile, and the
// frame after it can reuse them; what only the old scene drew is freed then.
TEST(RetainedResidency, ASceneFedAgainInTheSameStoreRetiresTheOldSlots)
{
  RetainedScene scene;
  const auto    feed = [&](float z) {
    MaterialId     m    = material(scene, "m");
    const EntityId root = drawn(scene, "root", scene.addMesh(gridMesh(2, z)), {m});
    drawn(scene, "child", scene.addMesh(gridMesh(1, z)), {m}, root);
    scene.release(m);
    return root;
  };
  const EntityId first = feed(0.f);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  drain(gpu);
  const uint32_t capacity = residency.instanceCapacity();

  scene.destroyEntity(first);
  const EntityId second = feed(1.f);
  gpu.beginFrame();
  SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.drawListChanged);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.instance(0), nullptr);  // the old scene's slots are holes
  EXPECT_EQ(residency.instance(1), nullptr);
  EXPECT_EQ(residency.instanceHighWater(), 4u);
  EXPECT_EQ(residency.instanceCapacity(), capacity);  // the tables stay

  // once retired, the next scene fed again takes them back, and nothing of the first is left
  drain(gpu);
  scene.destroyEntity(second);
  feed(2.f);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu);
  ASSERT_NE(residency.instance(0), nullptr);
  ASSERT_NE(residency.instance(1), nullptr);
  EXPECT_EQ(residency.instanceHighWater(), 4u);
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
  EXPECT_EQ(gpu.waitIdleCount(), 0u);
}

// With no memory to be found, a mesh isn't the end of anything: it is parked in the same sync, named once,
// and the rest draws. Tried again it comes back, and nothing leaks on the way. Every limit from nothing
// to room for all of it, each new try starting from whatever the last one left.
TEST(RetainedResidency, AMeshThatFindsNoMemoryIsParkedAndTheRestDraws)
{
  RetainedScene    scene;
  const MaterialId matte = material(scene, "matte");
  std::vector<MeshId> meshes;
  for(int i = 0; i < 3; i++)
  {
    meshes.push_back(scene.addMesh(gridMesh(8 + i * 8, float(i))));
    drawn(scene, "item", meshes.back(), {matte});
  }
  FakeGpuBackend whole(2);
  Residency      wholeResidency(whole);
  wholeResidency.replace(scene);
  expectClean(scene, wholeResidency, whole, "with room");
  EXPECT_TRUE(wholeResidency.parked().empty());
  const uint64_t room = whole.liveBytes();

  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  size_t         everParked = 0;
  // (the last with more than room: what is retiring counts, and tables that grew stay grown)
  for(uint64_t limit = 0; limit <= room * 2; limit += std::max<uint64_t>(room / 37, 1))
  {
    gpu.setMemoryLimit(limit);
    gpu.beginFrame();
    residency.retryParked();
    residency.sync(scene);
    expectClean(scene, residency, gpu, "at a limit of " + std::to_string(limit));
    everParked = std::max(everParked, residency.parked().size());
    drain(gpu);
    for(const std::string& p : residencyLeaks(residency, gpu))
      ADD_FAILURE() << "at a limit of " << limit << ": " << p;
  }
  EXPECT_GT(everParked, 0u);
  EXPECT_TRUE(residency.parked().empty()) << "with room for all of it, nothing stays out";
  // the first time each was parked, and not each time it was tried again
  const std::vector<std::string> notices = residency.takeUserNotices();
  EXPECT_FALSE(notices.empty());
  size_t named = 0;
  for(const std::string& n : notices)
    named += n.find("aren't drawn for now") != std::string::npos;
  EXPECT_LE(named, 3u);

  // asked for by a host: what is parked gives back all it held, and comes back whole
  const MeshId first = meshes.front();
  gpu.setMemoryLimit(std::nullopt);
  drain(gpu);
  const uint64_t before = gpu.liveBytes();
  residency.setParked(first, true);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "parked");
  EXPECT_EQ(residency.meshGpu(first), nullptr);
  drain(gpu);
  EXPECT_LT(gpu.liveBytes(), before);
  residency.setParked(first, false);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "back");
  drain(gpu);
  EXPECT_EQ(gpu.liveBytes(), before);
}

// What is new comes a few at a time, so that their BVHs don't all go in one submission: each sync takes what fits,
// and at least one however large, and goes on by itself until nothing waits.
TEST(RetainedResidency, WhatIsNewComesAFewAtATime)
{
  RetainedScene       scene;
  const MaterialId    m = material(scene, "m");
  std::vector<MeshId> meshes;
  for(int i = 0; i < 4; i++)
  {
    meshes.push_back(scene.addMesh(gridMesh(8, float(i))));
    drawn(scene, ("e" + std::to_string(i)).c_str(), meshes.back(), {m});
  }
  const uint64_t triangles = scene.mesh(meshes[0])->indices.size() / 3;
  FakeGpuBackend gpu;
  Residency      residency(gpu, {.arrivalTriangles = triangles * 2});
  residency.replace(scene);
  expectClean(scene, residency, gpu, "first");
  EXPECT_NE(residency.meshGpu(meshes[0]), nullptr);
  EXPECT_NE(residency.meshGpu(meshes[1]), nullptr);
  EXPECT_EQ(residency.meshGpu(meshes[2]), nullptr);
  EXPECT_EQ(residency.arriving(), (std::set<MeshId>{meshes[2], meshes[3]}));
  EXPECT_TRUE(residency.parked().empty()) << "waiting isn't parked";
  EXPECT_EQ(residency.parkedForMemory(), 0u);

  // with no change in the scene, the next sync takes them, and says the frame changed
  gpu.beginFrame();
  const SyncResult next = residency.sync(scene);
  EXPECT_TRUE(next.changed);
  EXPECT_TRUE(next.drawListChanged);
  expectClean(scene, residency, gpu, "second");
  EXPECT_NE(residency.meshGpu(meshes[3]), nullptr);
  EXPECT_TRUE(residency.arriving().empty());

  // nothing waits: nothing runs
  gpu.beginFrame();
  EXPECT_FALSE(residency.sync(scene).changed);

  // one too large for the sync comes alone, and then the rest
  RetainedScene       large;
  const MaterialId    lm = material(large, "m");
  const MeshId        big = large.addMesh(gridMesh(16, 0.f)), small = large.addMesh(gridMesh(2, 1.f));
  drawn(large, "big", big, {lm});
  const EntityId smallOne = drawn(large, "small", small, {lm});
  FakeGpuBackend gpu2;
  Residency      second(gpu2, {.arrivalTriangles = 1});
  second.replace(large);
  EXPECT_NE(second.meshGpu(big), nullptr);
  EXPECT_EQ(second.arriving(), std::set<MeshId>{small});
  expectClean(large, second, gpu2, "alone");

  // one that goes while it waits just goes
  large.destroyEntity(smallOne);
  gpu2.beginFrame();
  second.sync(large);
  EXPECT_TRUE(second.arriving().empty());
  expectClean(large, second, gpu2, "gone");
}

TEST(RetainedResidency, AVertexEditRefitsTheBlasesOfItsMesh)
{
  RetainedScene scene;
  const MeshId  grid = scene.addMesh(gridMesh(2, 0.f, 2));
  MaterialId    m    = material(scene, "m");
  drawn(scene, "e", grid, {m, m});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);

  std::vector<glm::vec3> moved = scene.mesh(grid)->positions;
  for(glm::vec3& p : moved)
    p.z += 0.5f;
  scene.setMeshVertices(grid, moved, {});
  gpu.beginFrame();
  const SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.changed && r.verticesWritten);
  EXPECT_FALSE(r.drawListChanged);
  expectClean(scene, residency, gpu);  // the BLASes hash the new positions
  for(uint32_t p = 0; p < residency.primHighWater(); p++)
    EXPECT_EQ(gpu.blas(residency.prim(p)->blas)->refits, 1u);
}

// blasRefitsBeforeRebuild: a BLAS refitted that many times since its build is built again in place
// of the next refit, the old one retired; the count starts over
TEST(RetainedResidency, ABlasRefittedEnoughIsBuiltAgain)
{
  RetainedScene scene;
  const MeshId  grid = scene.addMesh(gridMesh(2, 0.f));
  drawn(scene, "e", grid, {material(scene, "m")});
  FakeGpuBackend gpu;
  Residency      residency(gpu, {.refitsBeforeRebuild = 3});
  residency.replace(scene);
  const AccelHandle built = residency.prim(0)->blas;

  std::vector<glm::vec3> moved = scene.mesh(grid)->positions;
  const auto             edit  = [&] {
    for(glm::vec3& p : moved)
      p.z += 0.5f;
    scene.setMeshVertices(grid, moved, {});
    gpu.beginFrame();
    residency.sync(scene);
  };
  edit();
  edit();
  EXPECT_EQ(residency.prim(0)->blas, built);
  EXPECT_EQ(gpu.blas(built)->refits, 2u);
  edit();
  const AccelHandle again = residency.prim(0)->blas;
  EXPECT_NE(again, built);
  EXPECT_EQ(gpu.blas(again)->refits, 0u);
  expectClean(scene, residency, gpu);  // the new BLAS hashes the new positions
  edit();
  EXPECT_EQ(residency.prim(0)->blas, again);
  EXPECT_EQ(gpu.blas(again)->refits, 1u);
  drain(gpu);
  EXPECT_EQ(gpu.blas(built), nullptr);  // retired
}

// A mesh that changes shape a long way (hair falling under dForce) gets a BLAS built for its new shape:
// a refit leaves one that takes ten times as long to trace.
TEST(RetainedResidency, ABlasIsBuiltAgainWhenItsMeshChangesShapeFarAndNotOtherwise)
{
  RetainedScene scene;
  const MeshId  grid = scene.addMesh(gridMesh(8, 0.f));
  drawn(scene, "e", grid, {material(scene, "m")});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  const std::vector<glm::vec3> rest = scene.mesh(grid)->positions;
  const auto                   edit = [&](const std::function<glm::vec3(glm::vec3)>& to) {
    std::vector<glm::vec3> moved = rest;
    for(glm::vec3& p : moved)
      p = to(p);
    scene.setMeshVertices(grid, moved, {});
    gpu.beginFrame();
    residency.sync(scene);
    return residency.prim(0)->blas;
  };
  const AccelHandle built = residency.prim(0)->blas;

  // going somewhere as one, however far, leaves the BVH as good as it was
  EXPECT_EQ(edit([](glm::vec3 p) { return p + glm::vec3(0, 0, 40); }), built);
  // a small change of shape stays a refit
  EXPECT_EQ(edit([&](glm::vec3 p) { return p + glm::vec3(0, 0, p.x * 0.001f); }), built);
  EXPECT_GT(gpu.blas(built)->refits, 0u);

  // a large one is built again, once, and the count starts from there
  const auto fallen = [&](glm::vec3 p) { return p + glm::vec3(0, 0, p.x * 2.f); };
  const AccelHandle again = edit(fallen);
  EXPECT_NE(again, built);
  EXPECT_EQ(gpu.blas(again)->refits, 0u);
  expectClean(scene, residency, gpu);  // the new BLAS hashes the new positions
  EXPECT_EQ(edit([&](glm::vec3 p) { return fallen(p) + glm::vec3(0, 0, 0.001f * p.y); }), again);
  // and back again: as far the other way
  EXPECT_NE(edit([](glm::vec3 p) { return p; }), again);
  drain(gpu);
  EXPECT_EQ(gpu.blas(built), nullptr);  // retired
}

// blasCompactStatic: a static prim's BLAS is built for compaction, and a sync after the frame that
// built it puts the compacted copy in its place (the TLAS with it) and retires the original. A vertex
// edit can't refit a compacted BLAS: it builds it again, refittable from then on.
TEST(RetainedResidency, AStaticBlasIsCompactedOnceItsSizeIsKnown)
{
  RetainedScene scene;
  const MeshId  grid = scene.addMesh(gridMesh(2, 0.f));
  drawn(scene, "e", grid, {material(scene, "m")});
  FakeGpuBackend gpu;
  Residency      residency(gpu, {.compactStatic = true});
  residency.replace(scene);
  const AccelHandle built = residency.prim(0)->blas;
  EXPECT_TRUE(gpu.blas(built)->flags & eAccelAllowCompaction);
  EXPECT_FALSE(gpu.blas(built)->flags & eAccelAllowUpdate);

  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(scene).changed);
  const AccelHandle compacted = residency.prim(0)->blas;
  ASSERT_NE(compacted, built);
  EXPECT_TRUE(gpu.blas(compacted)->compacted);
  expectClean(scene, residency, gpu);
  gpu.beginFrame();
  EXPECT_FALSE(residency.sync(scene).changed);  // once
  drain(gpu);
  EXPECT_EQ(gpu.blas(built), nullptr);

  std::vector<glm::vec3> moved = scene.mesh(grid)->positions;
  for(glm::vec3& p : moved)
    p.z += 0.5f;
  scene.setMeshVertices(grid, moved, {});
  gpu.beginFrame();
  residency.sync(scene);
  const AccelHandle rebuilt = residency.prim(0)->blas;
  EXPECT_NE(rebuilt, compacted);
  EXPECT_TRUE(gpu.blas(rebuilt)->flags & eAccelAllowUpdate);
  expectClean(scene, residency, gpu);
  scene.setMeshVertices(grid, moved, {});
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_EQ(residency.prim(0)->blas, rebuilt);
  EXPECT_EQ(gpu.blas(rebuilt)->refits, 1u);
  EXPECT_TRUE(gpu.errors().empty());
}

// A vertex edit uploads the positions and normals, and the tangents only when it set them (a dial
// drag's steps keep them, and sending them again cost the drag GPU time); the UVs never
TEST(RetainedResidency, AVertexEditUploadsOnlyTheStreamsItWrote)
{
  RetainedScene scene;
  MeshRecord    record = gridMesh(2, 0.f);
  record.tangents.assign(record.positions.size(), glm::vec4(1, 0, 0, 1));
  const MeshId grid = scene.addMesh(std::move(record));
  drawn(scene, "e", grid, {material(scene, "m")});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  const Residency::MeshGpu& mesh = *residency.meshGpu(grid);
  std::vector<BufferHandle> sent;
  const auto                was  = [&](BufferHandle b) { return std::find(sent.begin(), sent.end(), b) != sent.end(); };
  const auto                send = [&] {
    gpu.beginFrame();
    gpu.clearUploads();
    residency.sync(scene);
    sent.clear();
    for(const FakeGpuBackend::UploadRecord& u : gpu.uploads())
      sent.push_back(u.buffer);
  };

  std::vector<glm::vec3> moved = scene.mesh(grid)->positions;
  for(glm::vec3& p : moved)
    p.z += 0.5f;
  scene.setMeshVertices(grid, moved, scene.mesh(grid)->normals);
  send();
  EXPECT_TRUE(was(mesh.positions) && was(mesh.normals));
  EXPECT_FALSE(was(mesh.tangents) || was(mesh.uv0));
  expectClean(scene, residency, gpu);

  scene.setMeshVertices(grid, moved, {}, std::vector<glm::vec4>(moved.size(), glm::vec4(0, 1, 0, 1)));
  send();
  EXPECT_TRUE(was(mesh.positions) && was(mesh.tangents));
  EXPECT_FALSE(was(mesh.uv0));
  expectClean(scene, residency, gpu);
}

// A grid of two submeshes of four triangles over one opacity micromap of eight triangle records: the
// first takes micromap triangles by a byte index each (the last a special one), the second its own
// from base 4
MeshId micromapped(RetainedScene& scene, float z)
{
  auto mm                    = std::make_shared<OpacityMicromap>();
  mm->data                   = std::vector<unsigned char>(32, 0x5a);
  mm->triangles              = std::vector<unsigned char>(64, 0x01);
  mm->usages                 = {{.count = 8, .subdivisionLevel = 1, .format = 1}};
  MeshRecord mesh            = gridMesh(2, z, 2);
  mesh.submeshes[0].micromap = std::make_shared<MicromapUse>(
      MicromapUse{.micromap = mm, .indices = {3, 2, 1, 0xff}, .indexComponentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE});
  mesh.submeshes[1].micromap = std::make_shared<MicromapUse>(MicromapUse{.micromap = mm, .baseTriangle = 4});
  return scene.addMesh(std::move(mesh));
}

// A micromap is built once for the prims that link it, each BLAS linking it through its own
// indices; a vertex edit refits them linked, and the micromap goes after the last of them
TEST(RetainedResidency, AnOpacityMicromapIsBuiltOnceForThePrimsThatLinkIt)
{
  RetainedScene    scene;
  const MaterialId m         = material(scene, "m");
  const MeshId     cardsMesh = micromapped(scene, 0.f);
  const EntityId   cards     = drawn(scene, "cards", cardsMesh, {m, m});
  drawn(scene, "plain", scene.addMesh(gridMesh(1, 1.f)), {m});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  ASSERT_EQ(residency.micromaps().size(), 1u);
  EXPECT_EQ(residency.micromaps().begin()->second.users, 2u);
  EXPECT_TRUE(residency.hasOpacityMicromaps());
  EXPECT_EQ(residency.prim(0)->input.micromapIndexSize, 1u);
  EXPECT_EQ(residency.prim(1)->input.micromapBaseTriangle, 4u);
  EXPECT_FALSE(residency.prim(2)->input.micromap);

  std::vector<glm::vec3> moved = scene.mesh(cardsMesh)->positions;
  for(glm::vec3& p : moved)
    p.z += 0.5f;
  scene.setMeshVertices(cardsMesh, moved, {});
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "refit: ");
  EXPECT_EQ(gpu.blas(residency.prim(0)->blas)->refits, 1u);

  scene.destroyEntity(cards);
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_FALSE(residency.hasOpacityMicromaps());
  EXPECT_TRUE(residency.micromaps().empty());
  drain(gpu);
  expectClean(scene, residency, gpu, "removed: ");
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// A replace retires the micromaps after their BLASes; on a device without them no BLAS links one,
// and the alpha path draws the cards
TEST(RetainedResidency, MicromapsGoWithAReplaceAndNeedTheDevice)
{
  RetainedScene    scene;
  const MaterialId m = material(scene, "m");
  drawn(scene, "cards", micromapped(scene, 0.f), {m, m});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  gpu.beginFrame();
  residency.replace(scene);
  expectClean(scene, residency, gpu, "replaced: ");
  EXPECT_TRUE(residency.hasOpacityMicromaps());
  drain(gpu);
  expectClean(scene, residency, gpu, "after its retires: ");
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;

  FakeGpuBackend without(2);
  without.setOpacityMicromaps(false);
  Residency plain(without);
  plain.replace(scene);
  expectClean(scene, plain, without, "no device support: ");
  EXPECT_FALSE(plain.hasOpacityMicromaps());
  EXPECT_EQ(without.liveAccels(), 3u);  // the two BLASes and the TLAS
  // and says so once
  const std::vector<std::string> notices = plain.takeNotices();
  ASSERT_EQ(notices.size(), 1u);
  EXPECT_NE(notices[0].find("VK_KHR_opacity_micromap"), std::string::npos);
  without.beginFrame();
  plain.replace(scene);
  EXPECT_TRUE(plain.takeNotices().empty());
}

// An edit needing a feature the live variant lacks is held, its slot keeping the row it had, while
// an edit within the variant's features goes through; once a sync's variant has the feature the
// held edit is written. A replace writes every material whatever the variant has, and says so.
TEST(RetainedResidency, AnEditTheVariantCantDrawIsHeldUntilItCan)
{
  RetainedScene    scene;
  const MaterialId glass = material(scene, "glass"), plain = material(scene, "plain");
  const MeshId     grid = scene.addMesh(gridMesh(2, 0.f));
  drawn(scene, "a", grid, {glass});
  drawn(scene, "b", grid, {plain});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.setShaderFeatures(nvvkgltf::SceneFeatureSet{});  // an optimal build of no features
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_FALSE(residency.drawsBeyondShaderFeatures());
  const uint32_t slot = uint32_t(residency.materialSlot(glass));

  scene.setMaterial(glass, withTransmission(scene.material(glass)->payload));
  fv::mat::Material red = scene.material(plain)->payload;
  red.baseColor         = {1, 0, 0, 1};
  scene.setMaterial(plain, red);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "held: ");
  EXPECT_EQ(residency.heldMaterials(), std::set<MaterialId>{glass});
  EXPECT_EQ(residency.shadeMaterial(slot)->transmissionFactor, 0.f);
  EXPECT_EQ(residency.shadeMaterial(uint32_t(residency.materialSlot(plain)))->pbrBaseColorFactor.x, 1.f);
  EXPECT_FALSE(residency.drawsBeyondShaderFeatures());

  gpu.beginFrame();
  residency.sync(scene);  // still held
  EXPECT_EQ(residency.heldMaterials().size(), 1u);

  residency.setShaderFeatures(scene.features());
  gpu.beginFrame();
  const SyncResult r = residency.sync(scene);
  expectClean(scene, residency, gpu, "written: ");
  EXPECT_TRUE(r.changed);
  EXPECT_TRUE(residency.heldMaterials().empty());
  EXPECT_EQ(residency.shadeMaterial(slot)->transmissionFactor, 1.f);

  residency.setShaderFeatures(nvvkgltf::SceneFeatureSet{});
  gpu.beginFrame();
  residency.replace(scene);
  expectClean(scene, residency, gpu, "replaced: ");
  EXPECT_TRUE(residency.heldMaterials().empty());
  EXPECT_TRUE(residency.drawsBeyondShaderFeatures());
  residency.setShaderFeatures(std::nullopt);  // every feature
  EXPECT_FALSE(residency.drawsBeyondShaderFeatures());
}

TEST(RetainedResidency, CapacityGrowsByDoublingWithANewTlas)
{
  RetainedScene scene;
  const MeshId  grid = scene.addMesh(gridMesh(1, 0.f));
  MaterialId    m    = material(scene, "m");
  for(int i = 0; i < 200; i++)
    drawn(scene, "e", grid, {m});
  FakeGpuBackend gpu;
  Residency      residency(gpu, {.minInstances = 128});
  residency.replace(scene);
  EXPECT_EQ(residency.instanceCapacity(), 256u);
  const AccelHandle before = residency.tlas();

  for(int i = 0; i < 100; i++)
    drawn(scene, "more", grid, {m});
  gpu.beginFrame();
  const SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.tlasHandleChanged && r.buffersChanged);
  EXPECT_EQ(residency.instanceCapacity(), 512u);
  EXPECT_NE(residency.tlas(), before);
  expectClean(scene, residency, gpu);
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

TEST(RetainedResidency, ARebindingRewritesTheMaterialSlot)
{
  RetainedScene  scene;
  const MeshId   grid = scene.addMesh(gridMesh(2, 0.f));
  MaterialId     a = material(scene, "a"), b = material(scene, "b");
  const EntityId e = drawn(scene, "e", grid, {a});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);

  MeshInstance instance;
  instance.parts.push_back({grid, {b}});
  scene.setMeshInstance(e, instance);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.materialSlot(a), -1);
  EXPECT_GE(residency.materialSlot(b), 0);
}

// Two materials over one texture table: each texture they sample once per sampler gets a slot past
// the default image's, its image is read once and decoded as colour where a material samples it as
// colour; materials that are not drawn hold nothing.
TEST(RetainedResidency, TexturesTakeSlotsPerSamplerOverOneImage)
{
  RetainedScene scene;
  const auto    table = textureTable(scene);
  const MeshId  grid  = scene.addMesh(gridMesh(2, 0.f, 2));
  drawn(scene, "e", grid, {textured(scene, table, 0), textured(scene, table, 1)});
  textured(scene, table, 1);  // not drawn
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);

  const TextureId a = (*table)[0].texture, b = (*table)[2].texture;
  EXPECT_EQ(gpu.liveImages(), 2u);
  const int32_t plain   = residency.textureSlot(a, {}, true),
                nearest = residency.textureSlot(a, {.magFilter = TINYGLTF_TEXTURE_FILTER_NEAREST}, true);
  EXPECT_GT(plain, 0);
  EXPECT_GT(nearest, 0);
  EXPECT_NE(plain, nearest);
  EXPECT_EQ(gpu.textureDescriptors().at(uint32_t(plain)), gpu.textureDescriptors().at(uint32_t(nearest)));  // one image
  EXPECT_TRUE(gpu.image(residency.image(a, true))->srgb);
  EXPECT_FALSE(gpu.image(residency.image(b, false))->srgb);
  EXPECT_EQ(residency.materialHighWater(), 2u);
  // the normal map's two channels reach its texture info
  const Residency::MaterialState* state = residency.materialState(0);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->slots.isTwoChannelSource(2), true);
}

// A second scene replace on the same residency (a re-import): once the first scene's retires have
// run, the new scene's texture slots still hold its images (the old slots' default-image writes must
// not land on slots handed out again)
TEST(RetainedResidency, AReplaceKeepsItsTexturesThroughTheOldOnesRetires)
{
  RetainedScene scene;
  const auto    table = textureTable(scene);
  const MeshId  grid  = scene.addMesh(gridMesh(2, 0.f, 2));
  drawn(scene, "e", grid, {textured(scene, table, 0), textured(scene, table, 1)});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);

  gpu.beginFrame();
  residency.replace(scene);
  expectClean(scene, residency, gpu, "replaced: ");
  drain(gpu);
  expectClean(scene, residency, gpu, "the old scene's retires run: ");
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// Images are decoded once per key: two records of the same bytes share one, a replace takes the
// previous scene's from grace, and the grace list keeps what fits textureGraceMB, oldest out first
TEST(RetainedResidency, ADecodeIsSharedAndOutlivesAReplaceInGrace)
{
  const auto record = [](unsigned char fill, int width) {
    TextureRecord t;
    t.bytes    = std::vector<unsigned char>(64, fill);
    t.mimeType = "image/png";
    t.width    = width;
    t.height   = width;
    return t;
  };
  RetainedScene scene;
  const TextureId a = scene.addTexture(record(1, 4)), same = scene.addTexture(record(1, 4)), b = scene.addTexture(record(2, 8));
  auto table = std::make_shared<TextureTable>();
  table->push_back({a, {}});
  table->push_back({same, {}});
  table->push_back({b, {}});
  const MeshId grid = scene.addMesh(gridMesh(2, 0.f, 2));
  drawn(scene, "e", grid, {textured(scene, table, 0), textured(scene, table, 1)});  // both sample b as the normal map
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.image(a, true), residency.image(same, true));
  EXPECT_EQ(gpu.liveImages(), 2u);
  EXPECT_EQ(residency.imageCounts().decoded, 2u);
  EXPECT_EQ(residency.imageCounts().reused, 1u);
  EXPECT_EQ(gpu.imageBatches(), std::vector<size_t>{2});  // decoded together, the shared one once

  // a replace decodes nothing it had
  const ImageHandle before = residency.image(b, false);
  gpu.beginFrame();
  residency.replace(scene);
  expectClean(scene, residency, gpu, "replaced: ");
  EXPECT_EQ(residency.imageCounts().decoded, 0u);
  EXPECT_EQ(residency.imageCounts().reused, 3u);
  EXPECT_EQ(gpu.imageBatches().size(), 1u);  // nothing to decode
  EXPECT_EQ(residency.image(b, false), before);
  drain(gpu);
  expectClean(scene, residency, gpu, "after its retires: ");

  // a scene without textures leaves both in grace; a budget for one keeps the newer release
  RetainedScene plain;
  drawn(plain, "e", plain.addMesh(gridMesh(2, 0.f)), {material(plain, "m")});
  gpu.beginFrame();
  residency.replace(plain);
  EXPECT_EQ(gpu.liveImages(), 2u);
  EXPECT_EQ(residency.graceBytes(), 4u * 4 * 4 + 8u * 8 * 4);
  residency.setTextureGrace(8 * 8 * 4);
  EXPECT_EQ(residency.graceBytes(), 8u * 8 * 4);
  drain(gpu);
  EXPECT_EQ(gpu.liveImages(), 1u);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;

  // and taking it back from grace
  gpu.beginFrame();
  residency.replace(scene);
  EXPECT_EQ(residency.imageCounts().reused + residency.imageCounts().decoded, 3u);
  EXPECT_EQ(residency.graceBytes(), 0u);
  expectClean(scene, residency, gpu, "back: ");
}

// ---------- Max texture size, and fitting GPU memory

TEST(ImageLevels, TheLimitKeepsWholeBlocks)
{
  EXPECT_EQ(levelsToSkip(4096, 4096, 13, 0), 0u);
  EXPECT_EQ(levelsToSkip(8192, 8192, 14, 4096), 1u);
  EXPECT_EQ(levelsToSkip(4096, 2048, 13, 1024), 2u);
  EXPECT_EQ(levelsToSkip(1024, 1024, 11, 4096), 0u);
  // 2496, 1248, 624, 312, 156, 78: a side of 78 isn't whole blocks, so 156 is as far as it goes
  EXPECT_EQ(levelsToSkip(2496, 2496, 12, 64), 4u);
  // a chain's own end
  EXPECT_EQ(levelsToSkip(4096, 4096, 2, 256), 1u);
  EXPECT_EQ(levelsToSkip(4096, 4096, 1, 256), 0u);
}

TEST(ImageLevels, TheLargestComeDownFirst)
{
  const auto fit = [](uint32_t extent) {
    ImageChain chain{.width = extent, .height = extent};
    for(uint32_t e = extent; e > 0; e /= 2)
      chain.levelBytes.push_back(uint64_t(e) * e);
    return ImageFit{chain, 0};
  };
  std::vector<ImageFit> images{fit(1024), fit(4096), fit(2048)};
  // 4096's top level alone is enough
  EXPECT_EQ(lowerToFit(images, 1, 256), 4096u * 4096u);
  EXPECT_EQ(images[1].extent(), 2048u);
  EXPECT_EQ(images[0].extent(), 1024u);
  EXPECT_EQ(images[2].extent(), 2048u);
  // the two at 2048 next, the earlier first, before 1024
  EXPECT_EQ(lowerToFit(images, 2048u * 2048u + 1, 256), 2u * 2048u * 2048u);
  EXPECT_EQ(images[0].extent(), 1024u);
  EXPECT_EQ(images[1].extent(), 1024u);
  EXPECT_EQ(images[2].extent(), 1024u);
  // and nothing goes under the least, however much is asked
  lowerToFit(images, ~0ull, 256);
  for(const ImageFit& image : images)
    EXPECT_EQ(image.extent(), 256u);
  EXPECT_EQ(lowerToFit(images, 1, 256), 0u);
}

namespace {
TextureRecord chainTexture(const char* uri, int extent)
{
  TextureRecord t;
  t.uri    = uri;
  t.name   = uri;
  t.width  = extent;
  t.height = extent;
  // the fake's file: a key from its bytes, as a path that isn't on disk has none
  t.bytes.assign(uri, uri + std::strlen(uri));
  return t;
}
// a chain's bytes at 4 bytes a texel, from `extent` down
uint64_t chainBytes(uint64_t extent)
{
  uint64_t sum = 0;
  for(; extent > 0; extent /= 2)
    sum += extent * extent * 4;
  return sum;
}
}  // namespace

// A material with only a base colour, entry `base` of the table
MaterialId coloured(RetainedScene& scene, std::shared_ptr<const TextureTable> table, int base)
{
  MaterialRecord m;
  m.payload.name               = "coloured";
  m.payload.baseColorMap.index = base;
  m.textures                   = std::move(table);
  return scene.addMaterial(std::move(m));
}

// A scene of two textures, each the base colour of a material
struct FitScene
{
  RetainedScene scene;
  TextureId     first, second;
  MeshId        mesh;
  MaterialId    firstMaterial, secondMaterial;
  FitScene(const char* a, int aExtent, const char* b, int bExtent)
  {
    first  = scene.addTexture(chainTexture(a, aExtent));
    second = scene.addTexture(chainTexture(b, bExtent));
    auto table = std::make_shared<TextureTable>();
    table->push_back({first, {}});
    table->push_back({second, {}});
    mesh           = scene.addMesh(gridMesh(2, 0.f, 2));
    firstMaterial  = coloured(scene, table, 0);
    secondMaterial = coloured(scene, table, 1);
    drawn(scene, "e", mesh, {firstMaterial, secondMaterial});
  }
};

// A sync decodes and stages so many bytes of textures (decodeBatchBytes), the first one whatever its size; the rest
// wait for the next syncs, on the default image till then, and all of them come
TEST(RetainedResidency, TexturesComeABatchOfBytesASync)
{
  const char* uris[] = {"chain_1", "chain_2", "chain_3", "chain_4", "chain_5"};
  for(const uint64_t batch : {chainBytes(64) * 5 / 2, uint64_t(1)})
  {
    SCOPED_TRACE(batch);
    RetainedScene          scene;
    std::vector<TextureId> textures;
    auto                   table = std::make_shared<TextureTable>();
    for(const char* uri : uris)
    {
      textures.push_back(scene.addTexture(chainTexture(uri, 64)));
      table->push_back({textures.back(), {}});
    }
    for(int i = 0; i < int(textures.size()); i++)
      drawn(scene, uris[i], scene.addMesh(gridMesh(2, 0.f)), {coloured(scene, table, i)});
    FakeGpuBackend gpu(2);
    Residency      residency(gpu, {.decodeBatchBytes = batch});
    residency.replace(scene);
    const size_t first = batch > 1 ? 2 : 1;
    EXPECT_EQ(gpu.imageBatches(), std::vector<size_t>{first});
    EXPECT_EQ(residency.queuedDecodes(), textures.size() - first);
    EXPECT_FALSE(residency.image(textures.back(), true)) << "the last waits";
    for(int frame = 0; frame < 8 && residency.queuedDecodes() > 0; frame++)
    {
      gpu.beginFrame();
      residency.sync(scene);
    }
    EXPECT_EQ(gpu.imageBatches(), batch > 1 ? (std::vector<size_t>{2, 2, 1}) : (std::vector<size_t>{1, 1, 1, 1, 1}));
    for(const TextureId t : textures)
      EXPECT_TRUE(residency.image(t, true));
    expectClean(scene, residency, gpu);
  }
}

// The render thread never waits for a decode: a texture is drawn with the default image while its decode
// runs, a sync meanwhile has nothing to do for it, and its image's arrival is another picture
TEST(RetainedResidency, ATextureIsDrawnWithTheDefaultImageUntilItsDecodeIsDone)
{
  FitScene       f("chain_a", 64, "chain_b", 64);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  gpu.holdDecodes(true);
  residency.replace(f.scene);
  EXPECT_FALSE(residency.image(f.first, true));
  EXPECT_FALSE(residency.image(f.second, true));
  EXPECT_EQ(gpu.decodesRunning().size(), 2u);
  EXPECT_EQ(residency.queuedDecodes(), 2u);
  expectClean(f.scene, residency, gpu, "decoding: ");
  for(int frame = 0; frame < 3; frame++)  // what the replace wrote comes to rest
  {
    gpu.beginFrame();
    residency.sync(f.scene);
  }
  gpu.beginFrame();
  EXPECT_FALSE(residency.sync(f.scene).changed) << "nothing is done: nothing to do";
  EXPECT_FALSE(residency.image(f.first, true));

  gpu.holdDecodes(false);
  gpu.beginFrame();
  const SyncResult arrived = residency.sync(f.scene);
  EXPECT_TRUE(arrived.changed);
  EXPECT_FALSE(arrived.samePicture) << "the default image gave way to the texture's";
  EXPECT_TRUE(residency.image(f.first, true));
  EXPECT_TRUE(residency.image(f.second, true));
  EXPECT_EQ(residency.queuedDecodes(), 0u);
  EXPECT_EQ(residency.imageCounts().decoded, 2u);
  expectClean(f.scene, residency, gpu, "arrived: ");
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// A raise keeps showing the lowered image while it decodes, isn't asked for again meanwhile, and is the
// same picture when it comes
TEST(RetainedResidency, ARaiseDecodingKeepsTheLoweredImageAndIsAskedOnce)
{
  FitScene       f("chain_a", 1024, "chain_b", 64);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  drain(gpu);
  const uint32_t lowered = gpu.imageInfo(residency.image(f.first, true)).width;
  ASSERT_LT(lowered, 1024u);

  gpu.setImageHeadroom(int64_t(1) << 30);
  gpu.holdDecodes(true);
  EXPECT_EQ(residency.raiseImages(f.scene, uint64_t(1) << 30), 1u);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, lowered) << "it shows what it has meanwhile";
  EXPECT_EQ(gpu.decodesRunning(), std::vector<std::string>{"chain_a"});
  EXPECT_EQ(residency.raiseImages(f.scene, uint64_t(1) << 30), 0u) << "on its way already";
  expectClean(f.scene, residency, gpu, "raising: ");

  gpu.holdDecodes(false);
  gpu.beginFrame();
  const SyncResult raised = residency.sync(f.scene);
  EXPECT_TRUE(raised.imagesMoved);
  EXPECT_TRUE(raised.samePicture);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "raised: ");
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// Memory short again while a raise decodes: the lowering is what stays, and the raise never lands
TEST(RetainedResidency, ALoweringUndoesARaiseStillDecoding)
{
  FitScene       f("chain_a", 1024, "chain_b", 64);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  drain(gpu);
  const uint32_t lowered = gpu.imageInfo(residency.image(f.first, true)).width;

  gpu.setImageHeadroom(int64_t(1) << 30);
  gpu.holdDecodes(true);
  ASSERT_EQ(residency.raiseImages(f.scene, uint64_t(1) << 30), 1u);
  gpu.beginFrame();
  residency.sync(f.scene);
  ASSERT_EQ(gpu.decodesRunning().size(), 1u);

  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.decodesCancelled(), 1u);
  EXPECT_TRUE(gpu.decodesRunning().empty());
  const uint32_t lowerStill = gpu.imageInfo(residency.image(f.first, true)).width;
  EXPECT_LT(lowerStill, lowered);

  gpu.holdDecodes(false);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, lowerStill);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "lowered again: ");
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// A scene replaced while its textures decode: a texture's own image goes with it, and one decoded for a
// key lands in grace, for the next scene to take with nothing decoded again
TEST(RetainedResidency, TexturesLetGoOfWhileTheyDecodeAreNeverDrawn)
{
  RetainedScene plain;
  drawn(plain, "e", plain.addMesh(gridMesh(2, 0.f)), {material(plain, "m")});

  {
    RetainedScene scene;
    const auto    table = textureTable(scene);  // files that aren't there: no key
    drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f, 2)), {textured(scene, table, 0), textured(scene, table, 1)});
    FakeGpuBackend gpu(2);
    Residency      residency(gpu);
    gpu.holdDecodes(true);
    residency.replace(scene);
    const size_t decoding = gpu.decodesRunning().size();
    ASSERT_GT(decoding, 0u);
    gpu.beginFrame();
    residency.replace(plain);
    EXPECT_EQ(gpu.decodesCancelled(), decoding);
    EXPECT_EQ(residency.queuedDecodes(), 0u);
    gpu.holdDecodes(false);
    gpu.beginFrame();
    residency.sync(plain);
    drain(gpu);
    EXPECT_EQ(gpu.liveImages(), 0u);
    expectClean(plain, residency, gpu, "no key: ");
    for(const std::string& p : residencyLeaks(residency, gpu))
      ADD_FAILURE() << p;
  }

  FitScene       f("chain_a", 64, "chain_b", 64);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  gpu.holdDecodes(true);
  residency.replace(f.scene);
  gpu.beginFrame();
  residency.replace(plain);
  EXPECT_EQ(gpu.decodesCancelled(), 0u);
  gpu.holdDecodes(false);
  gpu.beginFrame();
  residency.sync(plain);
  EXPECT_EQ(gpu.liveImages(), 2u) << "in grace";
  expectClean(plain, residency, gpu, "keyed, in grace: ");
  gpu.beginFrame();
  residency.replace(f.scene);
  EXPECT_EQ(residency.imageCounts().decoded, 0u);
  EXPECT_EQ(residency.imageCounts().reused, 2u);
  EXPECT_TRUE(residency.image(f.first, true));
  expectClean(f.scene, residency, gpu, "taken from grace: ");
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// Taken again while its first decode runs: a keyed image waits for that decode, one of its own is decoded
// again, and either way one image is made
TEST(RetainedResidency, ATextureTakenAgainWhileItDecodesIsMadeOnce)
{
  FitScene       f("chain_a", 64, "chain_b", 64);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  gpu.holdDecodes(true);
  residency.replace(f.scene);
  gpu.beginFrame();
  residency.replace(f.scene);
  EXPECT_EQ(gpu.decodesRunning().size(), 2u) << "the decodes begun serve it";
  gpu.holdDecodes(false);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_TRUE(residency.image(f.first, true));
  EXPECT_TRUE(residency.image(f.second, true));
  drain(gpu);
  EXPECT_EQ(gpu.liveImages(), 2u);
  expectClean(f.scene, residency, gpu, "keyed: ");

  RetainedScene  scene;
  const auto     table = textureTable(scene);
  drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f, 2)), {textured(scene, table, 0), textured(scene, table, 1)});
  FakeGpuBackend own(2);
  Residency      again(own);
  own.holdDecodes(true);
  again.replace(scene);
  const size_t decoding = own.decodesRunning().size();
  own.beginFrame();
  again.replace(scene);
  EXPECT_EQ(own.decodesCancelled(), decoding);
  EXPECT_EQ(own.decodesRunning().size(), decoding);
  own.holdDecodes(false);
  own.beginFrame();
  again.sync(scene);
  drain(own);
  EXPECT_EQ(own.liveImages(), decoding);
  expectClean(scene, again, own, "no key: ");
  for(const std::string& p : residencyLeaks(again, own))
    ADD_FAILURE() << p;
}

// Max texture size: every image is decoded under it, and a change takes effect on what is decoded next
TEST(RetainedResidency, MaxTextureSizeLimitsWhatIsDecoded)
{
  FitScene       f("chain_a", 64, "chain_b", 16);
  FakeGpuBackend gpu(2);
  gpu.setMaxImageExtent(32);
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 32u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, 16u);

  // a replace at another limit doesn't take the images in grace: they are decoded under the old one
  gpu.setMaxImageExtent(0);
  gpu.beginFrame();
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu, "no limit: ");
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 64u);
  EXPECT_EQ(residency.imageCounts().reused, 0u);
}

// With room for everything nothing changes, and no budget known is the same
TEST(RetainedResidency, ImagesThatFitAreLeftAlone)
{
  FitScene       f("chain_a", 64, "chain_b", 32);
  FakeGpuBackend gpu(2);
  gpu.setImageHeadroom(chainBytes(64) + chainBytes(32));
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 64u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, 32u);
  EXPECT_EQ(residency.imageCounts().lowered, 0u);
  EXPECT_TRUE(residency.takeUserNotices().empty());
}

// Short of room, the queued images come down, the largest first, and the user hears of it once
TEST(RetainedResidency, ImagesThatDontFitAreLowered)
{
  FitScene       f("chain_a", 1024, "chain_b", 256);
  FakeGpuBackend gpu(2);
  // room for the chains but for a little: the 1024's top level is the one to go
  gpu.setImageHeadroom(chainBytes(1024) + chainBytes(256) - 1);
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, 256u);
  EXPECT_EQ(residency.imageCounts().lowered, 1u);
  EXPECT_EQ(residency.takeUserNotices().size(), 1u);
  EXPECT_FALSE(residency.takeNotices().empty());
}

// Nothing goes under the least, however little room there is
TEST(RetainedResidency, LoweringStopsAtTheLeast)
{
  FitScene       f("chain_a", 1024, "chain_b", 128);
  FakeGpuBackend gpu(2);
  gpu.setImageHeadroom(0);
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 256u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, 128u);
}

// Idle decodes go before anything drawn comes down
TEST(RetainedResidency, IdleImagesGoBeforeLowering)
{
  FitScene       old("chain_old", 512, "chain_old2", 64);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 16 << 20});
  residency.replace(old.scene);
  expectClean(old.scene, residency, gpu);

  // the old scene's images in grace are what stands in the new one's way
  FitScene next("chain_a", 1024, "chain_b", 64);
  const uint64_t idle = gpu.imageInfo(residency.image(old.first, true)).bytes + gpu.imageInfo(residency.image(old.second, true)).bytes;
  gpu.setImageHeadroom(chainBytes(1024) + chainBytes(64) - idle);
  gpu.beginFrame();
  residency.replace(next.scene);
  EXPECT_EQ(residency.graceBytes(), 0u);
  EXPECT_EQ(gpu.imageInfo(residency.image(next.first, true)).width, 1024u);
  EXPECT_EQ(residency.imageCounts().lowered, 0u);
  drain(gpu);
  expectClean(next.scene, residency, gpu, "after its retires: ");
}

// A wear that doesn't fit takes down what is resident too: the image is decoded again, smaller, in the
// same slots, the old one retired, and the frame starts over. Let go of, it comes back whole if there is room.
TEST(RetainedResidency, ResidentImagesComeDownForNewOnes)
{
  RetainedScene scene;
  const TextureId big = scene.addTexture(chainTexture("chain_big", 1024));
  auto            table = std::make_shared<TextureTable>();
  table->push_back({big, {}});
  drawn(scene, "resident", scene.addMesh(gridMesh(2, 0.f)), {coloured(scene, table, 0)});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 16 << 20});
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  const ImageHandle before = residency.image(big, true);

  const TextureId worn = scene.addTexture(chainTexture("chain_worn", 512));
  auto            wornTable = std::make_shared<TextureTable>();
  wornTable->push_back({worn, {}});
  drawn(scene, "worn", scene.addMesh(gridMesh(2, 0.f)), {coloured(scene, wornTable, 0)});
  gpu.setImageHeadroom(chainBytes(256));  // the 1024's top level is more than enough, and the larger
  gpu.beginFrame();
  const SyncResult result = residency.sync(scene);
  EXPECT_TRUE(result.changed);
  const ImageHandle after = residency.image(big, true);
  EXPECT_NE(after, before);
  EXPECT_EQ(gpu.imageInfo(after).width, 512u);
  EXPECT_EQ(gpu.imageInfo(residency.image(worn, true)).width, 512u);
  EXPECT_EQ(residency.imageCounts().lowered, 1u);
  drain(gpu);
  expectClean(scene, residency, gpu, "after its retires: ");
  EXPECT_FALSE(gpu.hasImage(before));
}

// A mesh added past the budget takes textures down with nothing of its own to decode, and what
// lowering could free is what a new mesh is told it can count on
TEST(RetainedResidency, ASceneThatGrewPastItsBudgetLowersWhatIsThere)
{
  RetainedScene   scene;
  const TextureId big   = scene.addTexture(chainTexture("chain_big", 1024));
  auto            table = std::make_shared<TextureTable>();
  table->push_back({big, {}});
  const MaterialId shared = coloured(scene, table, 0);
  drawn(scene, "first", scene.addMesh(gridMesh(2, 0.f)), {shared});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 16 << 20});
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  // all but what a 256 keeps of a 1024: 1024 x 1024 x 4 x (1 - 1/16)
  EXPECT_EQ(residency.lowerableImageBytes(), 1024u * 1024u * 4u - 256u * 256u * 4u);
  EXPECT_EQ(residency.lowerableImageBytes(512), 1024u * 1024u * 4u - 512u * 512u * 4u);
  EXPECT_EQ(residency.lowerableImageBytes(2048), 0u);

  gpu.setImageHeadroom(-1);
  drawn(scene, "second", scene.addMesh(gridMesh(2, 0.f)), {shared});
  gpu.beginFrame();
  const SyncResult result = residency.sync(scene);
  EXPECT_TRUE(result.changed);
  EXPECT_EQ(gpu.imageInfo(residency.image(big, true)).width, 512u);
  EXPECT_EQ(residency.imageCounts().lowered, 1u);
  drain(gpu);
  expectClean(scene, residency, gpu, "after its retires: ");

  // nothing structural: no fit, whatever the headroom says
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_EQ(residency.imageCounts().lowered, 1u);
}

// One lowered and let go of is decoded again when next wanted, at what there is room for
TEST(RetainedResidency, ALoweredImageComesBackWhole)
{
  FitScene       f("chain_a", 1024, "chain_b", 16);
  FakeGpuBackend gpu(2);
  gpu.setImageHeadroom(chainBytes(256));
  Residency residency(gpu, {.textureGraceBytes = 16 << 20});
  residency.replace(f.scene);
  EXPECT_LT(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);
  EXPECT_EQ(residency.takeUserNotices().size(), 1u);

  gpu.setImageHeadroom(1ull << 30);
  gpu.beginFrame();
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu, "with room: ");
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "after its retires: ");
  EXPECT_TRUE(residency.takeUserNotices().empty());
}

// What the first hits want orders what gives way: the image they don't want goes first though it is the smaller,
// and an item none reach goes before any texture.
TEST(RetainedResidency, WhatRaysDontReachGivesWayFirst)
{
  FitScene       f("chain_a", 512, "chain_b", 1024);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20, .leastDetailStep = 1});
  residency.replace(f.scene);
  EXPECT_NE(residency.nextToGo(f.scene).find(" at 1024"), std::string::npos) << "with no hits, the largest";

  std::vector<uint32_t> table(size_t(residency.materialSlotCount()) * kHitBins, 0);
  const auto            count = [&](MaterialId m, uint32_t bin, uint32_t hits) { table[size_t(residency.materialSlot(m)) * kHitBins + bin] = hits; };
  count(f.firstMaterial, 1, 1000);  // wants its 512, whole
  count(f.secondMaterial, 0, 1);    // could show 256 of its 1024
  residency.setMaterialHits(f.scene, table);
  EXPECT_NE(residency.nextToGo(f.scene).find(" at 1024"), std::string::npos) << "still the one none want at that size";
  count(f.secondMaterial, 0, 0);
  count(f.secondMaterial, 2, 1000);  // now it wants its 1024, as the first its 512: the larger's level frees more
  residency.setMaterialHits(f.scene, table);
  const MemoryLedger ledger = residency.ledger();
  EXPECT_EQ(ledger.hits.size(), 2u);

  // the smaller goes first when the larger is wanted far more
  count(f.firstMaterial, 1, 1);
  count(f.secondMaterial, 2, 100000);
  residency.setMaterialHits(f.scene, table);
  EXPECT_NE(residency.nextToGo(f.scene).find(" at 512"), std::string::npos);
  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 256u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, 1024u);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "lowered by hits: ");

  // an item none reach goes before any texture, one they do after textures reach 2K; one not counted isn't unseen
  const ItemDetail::Down seen{.item = 1, .name = "seen", .meshes = {{f.mesh, 0.5f}}, .materials = {f.secondMaterial}};
  ItemDetail::Down       unseen{.item = 2, .name = "unseen", .meshes = {{f.mesh, 0.5f}}, .materials = {f.firstMaterial}};
  count(f.firstMaterial, 1, 0);
  residency.setMaterialHits(f.scene, table);
  residency.setItemDetail({.down = {seen, unseen}});
  EXPECT_EQ(residency.nextToGo(f.scene), "the detail of unseen");
  ItemDetail::Down notCounted = unseen;
  notCounted.materials        = {MaterialId{}};
  residency.setItemDetail({.down = {notCounted}});
  EXPECT_EQ(residency.nextToGo(f.scene), "the detail of unseen") << "a 1024 is below where textures would go first, so it goes all the same";
  residency.setItemDetail({});
  drain(gpu);
}

// Memory full and the hits new: what they want less makes way for what they want more, once for each count
TEST(RetainedResidency, WhatHitsWantBackMakesRoomForItself)
{
  FitScene       f("chain_a", 1024, "chain_b", 2048);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20, .leastDetailStep = 1});
  residency.replace(f.scene);
  std::vector<uint32_t> table(size_t(residency.materialSlotCount()) * kHitBins, 0);
  const auto            count = [&](MaterialId m, uint32_t bin, uint32_t hits) { table[size_t(residency.materialSlot(m)) * kHitBins + bin] = hits; };
  // the first is lowered, wanted by nothing at that size
  count(f.firstMaterial, 0, 100000);
  count(f.secondMaterial, 3, 100000);
  residency.setMaterialHits(f.scene, table);
  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  drain(gpu);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);

  // the view turns: now the hits want the first, and could show 256 of the second
  gpu.setImageHeadroom(0);
  count(f.firstMaterial, 0, 0);
  count(f.firstMaterial, 2, 100000);
  count(f.secondMaterial, 3, 0);
  count(f.secondMaterial, 0, 1);
  residency.setMaterialHits(f.scene, table);
  EXPECT_FALSE(residency.followHits(f.scene, 0).has_value());
  gpu.beginFrame();
  residency.sync(f.scene);
  drain(gpu);
  const uint32_t gaveWay = gpu.imageInfo(residency.image(f.second, true)).width;
  EXPECT_LT(gaveWay, 2048u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u) << "it comes up when the room shows";
  expectClean(f.scene, residency, gpu, "exchanged: ");

  // the room shows: what they want comes, and nothing else does for all there is
  const int64_t room = int64_t(1) << 30;
  gpu.setImageHeadroom(room);
  EXPECT_FALSE(residency.followHits(f.scene, room).has_value());
  gpu.beginFrame();
  residency.sync(f.scene);
  drain(gpu);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, gaveWay);
  expectClean(f.scene, residency, gpu, "came back: ");

  // that was all of this count
  EXPECT_FALSE(residency.followHits(f.scene, room).has_value());
  gpu.beginFrame();
  residency.sync(f.scene);
  drain(gpu);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, gaveWay);

  // an item down that rays reach, and one up they hardly do: the second gives way for it
  gpu.setImageHeadroom(0);
  ItemDetail detail{.down = {{.item = 5, .name = "far", .meshes = {{f.mesh, 0.5f}}, .materials = {f.secondMaterial}}},
                    .up   = {{.item = 6, .bytes = 10, .materials = {f.firstMaterial}}}};
  residency.setItemDetail(detail);
  residency.setMaterialHits(f.scene, table);
  EXPECT_FALSE(residency.followHits(f.scene, 0).has_value());
  EXPECT_EQ(residency.takeDetailAsks(), std::vector<uint64_t>{5});
  residency.setItemDetail({.up = detail.up});
  EXPECT_EQ(residency.followHits(f.scene, 100)->item, 6u) << "with the room it made";
  drain(gpu);
}

TEST(RetainedResidency, WhatRaysReachComesBackFirst)
{
  FitScene       f("chain_a", 1024, "chain_b", 256);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20, .leastDetailStep = 1});
  residency.replace(f.scene);
  std::vector<uint32_t> table(size_t(residency.materialSlotCount()) * kHitBins, 0);
  table[size_t(residency.materialSlot(f.firstMaterial)) * kHitBins + 1] = 1000;
  residency.setMaterialHits(f.scene, table);

  residency.setItemDetail({.up = {{1, 10, {f.secondMaterial}}, {2, 10, {f.firstMaterial}}, {3, 1000, {f.firstMaterial}}}});
  EXPECT_EQ(residency.nextDetailUp(50, false)->item, 2u) << "the one they reach, offered later, that fits";
  EXPECT_FALSE(residency.nextDetailUp(50, false).has_value()) << "the one they don't waits for the textures they do";
  EXPECT_EQ(residency.nextDetailUp(50, true)->item, 1u);
  EXPECT_EQ(residency.nextDetailUp(1000, false)->item, 3u);
}

// The host makes a worn item's smaller records, so the Fit only asks: once, counting on what it asked for
// until the host says it has done with it, and only for what is worth it.
TEST(RetainedResidency, AnItemsDetailIsAskedOfTheHost)
{
  FitScene       f("chain_a", 1024, "chain_b", 256);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20, .leastDetailStep = 1});
  residency.replace(f.scene);
  EXPECT_NE(residency.nextToGo(f.scene).find(" at "), std::string::npos) << "nothing offered: textures";
  const ItemDetail offered{.down = {{.item = 7, .name = "Long Hair", .meshes = {{f.mesh, 0.5f}}}}};
  residency.setItemDetail(offered);
  EXPECT_EQ(residency.nextToGo(f.scene), "the detail of Long Hair") << "a 1024 is already below where they show";

  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(residency.takeDetailAsks(), std::vector<uint64_t>{7});
  EXPECT_EQ(residency.loweredImages(), Residency::LoweredImages{});
  drain(gpu);
  expectClean(f.scene, residency, gpu);

  // being made: nothing more gives way for what it will free
  residency.setItemDetail({.making = true});
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_TRUE(residency.takeDetailAsks().empty());
  EXPECT_EQ(residency.loweredImages(), Residency::LoweredImages{});

  // done with, and still short: what was offered and what comes next
  residency.setItemDetail({});
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  drain(gpu);

  // too little to be worth a new record
  FakeGpuBackend otherGpu(2);
  Residency      small(otherGpu, {.textureGraceBytes = 1 << 20, .leastDetailStep = uint64_t(1) << 40});
  small.replace(f.scene);
  small.setItemDetail(offered);
  EXPECT_NE(small.nextToGo(f.scene).find(" at "), std::string::npos);

  // up: the first that fits, once, and textures only as far as asked
  residency.setItemDetail({.up = {{7, 1000}, {8, 10}}});
  EXPECT_TRUE(residency.detailBelowFull());
  EXPECT_EQ(residency.nextDetailUp(50)->item, 8u);
  EXPECT_FALSE(residency.nextDetailUp(50).has_value());
  EXPECT_EQ(residency.nextDetailUp(1000)->item, 7u);
  EXPECT_FALSE(residency.detailBelowFull());
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 256u);
  drain(gpu);
  gpu.setImageHeadroom(1ll << 40);
  EXPECT_EQ(residency.raiseImages(f.scene, uint64_t(1) << 40, 512), 1u);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  EXPECT_EQ(residency.raiseImages(f.scene, uint64_t(1) << 40, 512), 0u) << "as far as asked";
  residency.raiseImages(f.scene, uint64_t(1) << 40);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "raised: ");
}

// Memory another program took: nothing in the scene changed, and nothing happens until the watch asks
TEST(RetainedResidency, ASyncAskedToLowerDoesSoWithNothingChanged)
{
  FitScene       f("chain_a", 1024, "chain_b", 256);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  EXPECT_EQ(residency.loweredImages(), Residency::LoweredImages{});

  gpu.setImageHeadroom(-1);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);

  residency.lowerImages(0);
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(f.scene).changed);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  EXPECT_EQ(residency.loweredImages(), (Residency::LoweredImages{1, 512}));
  drain(gpu);
  expectClean(f.scene, residency, gpu, "after its retires: ");

  // asked once: the next sync has nothing to do, whatever the headroom says
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);

  // with slack, what stays free counts as wanted: the 512's top level is more than the slack
  gpu.setImageHeadroom(0);
  residency.lowerImages(1);
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(f.scene).changed);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 256u);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, 256u);
  EXPECT_EQ(residency.loweredImages(), (Residency::LoweredImages{1, 256}));
}

// What comes down is copied from what it was, with nothing read or decoded, and a still goes on through it.
// One that can't be copied is decoded smaller instead.
TEST(RetainedResidency, ComingDownCopiesTheLevelsItKeeps)
{
  FitScene       f("chain_a", 1024, "chain-nocopy_b", 512);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  drain(gpu);
  const size_t batches = gpu.imageBatches().size();

  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  const SyncResult down = residency.sync(f.scene);
  EXPECT_TRUE(down.changed);
  EXPECT_TRUE(down.samePicture);
  EXPECT_TRUE(down.imagesMoved);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  EXPECT_EQ(gpu.imageCopies(), 1u);
  EXPECT_EQ(residency.imageCounts().copied, 1u);
  EXPECT_EQ(gpu.imageBatches().size(), batches) << "nothing was decoded";
  drain(gpu);
  expectClean(f.scene, residency, gpu);

  // both now at 512, the one that can't be copied goes the old way
  gpu.setImageHeadroom(-int64_t(chainBytes(512)) * 2);
  residency.lowerImages(0);
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(f.scene).samePicture);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.second, true)).width, 256u);
  EXPECT_EQ(gpu.imageBatches().size(), batches + 1);
  drain(gpu);
  expectClean(f.scene, residency, gpu);
}

// Memory given back: what was lowered comes up as far as the room goes, in place, and the user hears
// when all of it is back; the same again told anew
TEST(RetainedResidency, LoweredImagesComeBackAsFarAsThereIsRoom)
{
  FitScene       f("chain_a", 1024, "chain_b", 16);
  FakeGpuBackend gpu(2);
  gpu.setImageHeadroom(chainBytes(256) + chainBytes(16));
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  EXPECT_EQ(residency.loweredImages(), (Residency::LoweredImages{1, 256}));
  EXPECT_EQ(residency.takeUserNotices().size(), 1u);
  const ImageHandle low = residency.image(f.first, true);

  gpu.setImageHeadroom(1ull << 30);
  gpu.beginFrame();
  EXPECT_EQ(residency.raiseImages(f.scene, chainBytes(512) - 1), 0u);  // decoded whole, beside the one it replaces
  residency.sync(f.scene);
  EXPECT_EQ(residency.image(f.first, true), low);

  gpu.beginFrame();
  EXPECT_EQ(residency.raiseImages(f.scene, chainBytes(512)), 1u);
  EXPECT_TRUE(residency.sync(f.scene).changed);
  EXPECT_NE(residency.image(f.first, true), low);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  EXPECT_EQ(residency.loweredImages(), (Residency::LoweredImages{1, 512}));
  EXPECT_TRUE(residency.takeUserNotices().empty());
  drain(gpu);
  expectClean(f.scene, residency, gpu, "after its retires: ");
  EXPECT_FALSE(gpu.hasImage(low));

  gpu.beginFrame();
  EXPECT_EQ(residency.raiseImages(f.scene, 1ull << 30), 1u);
  EXPECT_TRUE(residency.sync(f.scene).changed);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);
  EXPECT_EQ(residency.loweredImages(), Residency::LoweredImages{});
  EXPECT_EQ(residency.takeUserNotices().size(), 1u);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "whole again: ");
  EXPECT_EQ(residency.raiseImages(f.scene, 1ull << 30), 0u);

  // short again: told again
  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(residency.takeUserNotices().size(), 1u);
}

// Max texture size is as far as an image comes back
TEST(RetainedResidency, ARaisedImageStopsAtMaxTextureSize)
{
  FitScene       f("chain_a", 1024, "chain_b", 16);
  FakeGpuBackend gpu(2);
  gpu.setImageHeadroom(chainBytes(256) + chainBytes(16));
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  gpu.setImageHeadroom(1ull << 30);
  gpu.setMaxImageExtent(512);
  gpu.beginFrame();
  EXPECT_EQ(residency.raiseImages(f.scene, 1ull << 30), 1u);
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  EXPECT_EQ(residency.loweredImages(), Residency::LoweredImages{});
}

// Two textures of one file are one image: counted once, and both come back
TEST(RetainedResidency, ASharedFileIsLoweredAndRaisedAsOneImage)
{
  RetainedScene   scene;
  const TextureId a = scene.addTexture(chainTexture("chain_same", 1024));
  const TextureId b = scene.addTexture(chainTexture("chain_same", 1024));
  auto            table = std::make_shared<TextureTable>();
  table->push_back({a, {}});
  table->push_back({b, {}});
  drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f, 2)), {coloured(scene, table, 0), coloured(scene, table, 1)});
  FakeGpuBackend gpu(2);
  gpu.setImageHeadroom(chainBytes(512));
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.loweredImages(), (Residency::LoweredImages{1, 512}));

  gpu.setImageHeadroom(1ull << 30);
  gpu.beginFrame();
  EXPECT_EQ(residency.raiseImages(scene, 1ull << 30), 1u);
  residency.sync(scene);
  expectClean(scene, residency, gpu, "raised: ");
  EXPECT_EQ(gpu.imageInfo(residency.image(a, true)).width, 1024u);
  EXPECT_EQ(residency.image(a, true), residency.image(b, true));
  EXPECT_EQ(residency.loweredImages(), Residency::LoweredImages{});
}

// Short of memory, an item nothing shows lets go of what it holds before anything visible gives way, and
// isn't counted as what the scene is drawn without. Shown again it comes back by itself.
TEST(RetainedResidency, AHiddenItemGoesBeforeAnythingThatShows)
{
  FitScene       f("chain_a", 1024, "chain_b", 256);
  const MeshId   other = f.scene.addMesh(gridMesh(4, 3.f, 2));
  const EntityId hidden = drawn(f.scene, "hidden", other, {coloured(f.scene, nullptr, -1), coloured(f.scene, nullptr, -1)});
  f.scene.setVisible(hidden, false);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  ASSERT_TRUE(residency.meshGpu(other));
  const uint64_t holds = residency.ledger().total.total();

  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(f.scene);  // the plan
  gpu.beginFrame();
  residency.sync(f.scene);  // and it goes
  drain(gpu);
  expectClean(f.scene, residency, gpu, "let go: ");
  EXPECT_FALSE(residency.meshGpu(other));
  EXPECT_EQ(residency.letGoHidden().size(), 1u);
  EXPECT_EQ(residency.parkedForMemory(), 0u);
  EXPECT_LT(residency.ledger().total.total(), holds);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u) << "nothing that shows gave way";

  f.scene.setVisible(hidden, true);
  gpu.beginFrame();
  residency.sync(f.scene);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "shown: ");
  EXPECT_TRUE(residency.meshGpu(other));
  EXPECT_TRUE(residency.letGoHidden().empty());
}

// What is marked Keep full comes down only when nothing else can, and comes back first
TEST(RetainedResidency, WhatIsKeptFullComesDownLast)
{
  RetainedScene scene;
  const TextureId kept = scene.addTexture(chainTexture("chain_kept", 2048)), plain = scene.addTexture(chainTexture("chain_plain", 1024));
  auto            table = std::make_shared<TextureTable>();
  table->push_back({kept, {}});
  table->push_back({plain, {}});
  MaterialRecord keep;
  keep.payload.name               = "kept";
  keep.payload.baseColorMap.index = 0;
  keep.textures                   = table;
  keep.keepFull                   = true;
  drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f, 2)), {scene.addMaterial(std::move(keep)), coloured(scene, table, 1)});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(scene);

  // though the larger, it stays while the other can come down
  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(kept, true)).width, 2048u);
  EXPECT_EQ(gpu.imageInfo(residency.image(plain, true)).width, 512u);
  drain(gpu);
  expectClean(scene, residency, gpu);

  // the other at its least: now it goes too
  gpu.setImageHeadroom(-int64_t(Residency::kFitStep));
  for(int i = 0; i < 4; i++)
  {
    residency.lowerImages(0);
    gpu.beginFrame();
    residency.sync(scene);
    drain(gpu);
  }
  EXPECT_EQ(gpu.imageInfo(residency.image(plain, true)).width, kLeastImageExtent);
  EXPECT_LT(gpu.imageInfo(residency.image(kept, true)).width, 2048u);
}

// A stand-in, shown until its cook lands, is decoded at half its size, and is another image than the
// same file at its whole size
TEST(RetainedResidency, AStandInIsDecodedAtHalfItsSize)
{
  RetainedScene scene;
  TextureRecord standIn = chainTexture("mips_a", 1024);
  standIn.standIn       = true;
  const TextureId half  = scene.addTexture(standIn);
  const TextureId whole = scene.addTexture(chainTexture("mips_a", 1024));
  auto            table = std::make_shared<TextureTable>();
  table->push_back({half, {}});
  table->push_back({whole, {}});
  drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f, 2)), {coloured(scene, table, 0), coloured(scene, table, 1)});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(half, true)).width, 512u);
  EXPECT_EQ(gpu.imageInfo(residency.image(whole, true)).width, 1024u);
  drain(gpu);
  expectClean(scene, residency, gpu);
}

// An image with no stored levels (a PNG, kept whole) comes down by the levels the GPU made, and comes back all at once
TEST(RetainedResidency, AnImageWithNoStoredLevelsComesDownToo)
{
  FitScene       f("mips_a", 1024, "chain_b", 256);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  gpu.setImageHeadroom(-1);
  residency.lowerImages(0);
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(f.scene).imagesMoved);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 512u);
  EXPECT_EQ(residency.loweredImages().count, 1u);
  drain(gpu);
  expectClean(f.scene, residency, gpu);

  gpu.setImageHeadroom(int64_t(1) << 40);
  EXPECT_EQ(residency.raiseImages(f.scene, uint64_t(1) << 30), 1u);
  gpu.beginFrame();
  residency.sync(f.scene);
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 1024u);
  EXPECT_EQ(residency.loweredImages().count, 0u);
  drain(gpu);
  expectClean(f.scene, residency, gpu);
}

// What had to go to system memory comes back when there is room: an image by a copy, then a mesh made
// again, one at a time, with nothing missing from any frame between.
TEST(RetainedResidency, WhatSpilledComesBackWhenThereIsRoom)
{
  FitScene       f("chain_a", 256, "chain_b", 256);
  FakeGpuBackend gpu(2);
  gpu.setSpilling(true);
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  expectClean(f.scene, residency, gpu);
  EXPECT_GT(gpu.spilledBytes(), 0u);
  gpu.setSpilling(false);

  // no room: nothing moves
  EXPECT_EQ(residency.bringBack(f.scene, 0), 0u);

  EXPECT_EQ(residency.bringBack(f.scene, uint64_t(1) << 30), 2u) << "the images";
  gpu.beginFrame();
  const SyncResult moved = residency.sync(f.scene);
  EXPECT_TRUE(moved.samePicture);
  EXPECT_FALSE(gpu.imageSpilled(residency.image(f.first, true)));
  EXPECT_EQ(gpu.imageInfo(residency.image(f.first, true)).width, 256u);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "images moved: ");

  EXPECT_EQ(residency.bringBack(f.scene, uint64_t(1) << 30), 1u) << "the mesh";
  gpu.beginFrame();
  residency.sync(f.scene);
  expectClean(f.scene, residency, gpu, "mesh made again: ");
  EXPECT_EQ(residency.bringBack(f.scene, uint64_t(1) << 30), 0u);
  drain(gpu);
  expectClean(f.scene, residency, gpu, "after its retires: ");
  const MeshId mesh = residency.ledger().meshes.at(0).mesh;
  EXPECT_FALSE(gpu.bufferSpilled(residency.meshGpu(mesh)->positions));
}

// A favoured material's images come down after the others' and go back before them; a tie between
// images alike falls to the earlier on the way down, so the face is listed first here
TEST(RetainedResidency, AFavouredMaterialsImagesStayAboveTheOthers)
{
  RetainedScene   scene;
  const TextureId face = scene.addTexture(chainTexture("chain_face", 1024));
  const TextureId body = scene.addTexture(chainTexture("chain_body", 1024));
  auto            table = std::make_shared<TextureTable>();
  table->push_back({face, {}});
  table->push_back({body, {}});
  MaterialRecord favoured;
  favoured.payload.name               = "face";
  favoured.payload.baseColorMap.index = 0;
  favoured.textures                   = table;
  favoured.favoured                   = true;
  drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f, 2)), {scene.addMaterial(std::move(favoured)), coloured(scene, table, 1)});

  FakeGpuBackend gpu(2);
  // short by a top level of 1024: the body's goes, and the face stays
  gpu.setImageHeadroom(chainBytes(1024) + chainBytes(512));
  Residency residency(gpu, {.textureGraceBytes = 1 << 20});
  const auto widths = [&] { return std::pair{gpu.imageInfo(residency.image(face, true)).width, gpu.imageInfo(residency.image(body, true)).width}; };
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(widths(), (std::pair{1024u, 512u}));

  // the body goes on down before the face follows it, so the face is always a level or more above
  gpu.setImageHeadroom(-int64_t(512 * 512 * 4));
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_EQ(widths(), (std::pair{1024u, 256u}));
  drain(gpu);
  gpu.setImageHeadroom(-int64_t(512 * 512 * 4));
  residency.lowerImages(0);
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_EQ(widths(), (std::pair{512u, 256u}));
  drain(gpu);

  // on the way back the face is first
  gpu.setImageHeadroom(1ull << 30);
  gpu.beginFrame();
  EXPECT_EQ(residency.raiseImages(scene, chainBytes(1024)), 1u);
  residency.sync(scene);
  EXPECT_EQ(widths(), (std::pair{1024u, 256u}));
  drain(gpu);
  expectClean(scene, residency, gpu, "after its retires: ");
}

// GPU memory that can't hold an image leaves the default one, with a notice, not an exit
TEST(RetainedResidency, AnImageWithoutMemoryKeepsTheDefault)
{
  FitScene       f("nomem_a", 64, "chain_b", 16);
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureGraceBytes = 1 << 20});
  residency.replace(f.scene);
  EXPECT_FALSE(residency.image(f.first, true));
  EXPECT_TRUE(residency.image(f.second, true));
  EXPECT_EQ(residency.imageCounts().refused, 1u);
  EXPECT_EQ(residency.takeUserNotices().size(), 1u);
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// A file's key is its path, size and write time, and the colour space; no key for a missing file
TEST(RetainedResidency, ATextureKeyIsTheFileAsItIsNow)
{
  namespace fs       = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "retained_texture_key";
  fs::create_directories(dir);
  {
    std::ofstream(dir / "a.png") << "first";
  }
  TextureRecord t;
  t.uri                                 = (dir / "a.png").string();
  const std::optional<TextureKey> first = textureKeyOf(t, true);
  ASSERT_TRUE(first);
  EXPECT_NE(textureKeyOf(t, false), first);
  EXPECT_EQ(textureKeyOf(t, true), first);
  {
    std::ofstream(dir / "a.png") << "second, longer";
  }
  EXPECT_NE(textureKeyOf(t, true), first);
  t.uri = (dir / "gone.png").string();
  EXPECT_FALSE(textureKeyOf(t, true));
  fs::remove_all(dir);
}

// A material with `count` texture references (base colour, normal, emissive, occlusion,
// metallic-roughness, in that order) over table entry 0
MaterialId withTextures(RetainedScene& scene, std::shared_ptr<const TextureTable> table, int count, const char* name)
{
  MaterialRecord m;
  m.payload.name = name;
  int* fields[]  = {&m.payload.baseColorMap.index, &m.payload.normalMap.index,
                    &m.payload.emissiveMap.index, &m.payload.occlusionMap.index,
                    &m.payload.metallicRoughnessMap.index};
  for(int i = 0; i < count; i++)
    *fields[i] = 0;
  m.textures = std::move(table);
  return scene.addMaterial(std::move(m));
}

// Each material owns a span of the texture-info table: an edit writes its own row and span, and one
// that needs another span length moves it, the old one freed at retire; no other span moves
TEST(RetainedResidency, AMaterialEditWritesItsRowAndSpanOnly)
{
  RetainedScene    scene;
  const auto       table = textureTable(scene);
  const MeshId     grid  = scene.addMesh(gridMesh(2, 0.f));
  const MaterialId a = withTextures(scene, table, 2, "a"), b = withTextures(scene, table, 3, "b");
  drawn(scene, "a", grid, {a});
  drawn(scene, "b", grid, {b});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  const uint32_t slotA = uint32_t(residency.materialSlot(a)), slotB = uint32_t(residency.materialSlot(b));
  const uint32_t spanB = residency.materialState(slotB)->span;
  EXPECT_EQ(residency.materialState(slotA)->spanCount, 2u);
  EXPECT_EQ(residency.materialState(slotB)->spanCount, 3u);

  // a factor: its row and its span, nothing else of the material tables
  fv::mat::Material payload = scene.material(a)->payload;
  payload.baseColor         = {0.5, 0.5, 0.5, 1.0};
  scene.setMaterial(a, payload);
  gpu.clearUploads();
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "a factor: ");
  std::vector<std::pair<uint64_t, uint64_t>> materialUploads;
  for(const FakeGpuBackend::UploadRecord& u : gpu.uploads())
    if(u.buffer == residency.materialTable() || u.buffer == residency.textureInfoTable())
      materialUploads.push_back({u.offset, u.size});
  const uint64_t kRow = sizeof(shaderio::GltfShadeMaterial), kInfo = sizeof(shaderio::GltfTextureInfo);
  const std::vector<std::pair<uint64_t, uint64_t>> want = {{slotA * kRow, kRow},
                                                           {residency.materialState(slotA)->span * kInfo, 2 * kInfo}};
  EXPECT_EQ(materialUploads, want);

  // two more references: a longer span elsewhere; b's stays
  payload.emissiveMap.index  = 0;
  payload.occlusionMap.index = 0;
  scene.setMaterial(a, payload);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "a longer span: ");
  EXPECT_EQ(residency.materialState(slotA)->spanCount, 4u);
  EXPECT_EQ(residency.materialState(slotB)->span, spanB);
  drain(gpu);
  expectClean(scene, residency, gpu, "after its retires: ");
}

// Under the limit the spans are placed below, a span that finds no room repacks every span once;
// one that still doesn't fit is drawn without its textures, with a notice, never a wrapped index
TEST(RetainedResidency, TextureInfoSpansRepackThenRefuse)
{
  RetainedScene    scene;
  const auto       table = textureTable(scene);
  const MeshId     grid  = scene.addMesh(gridMesh(2, 0.f));
  const MaterialId a = withTextures(scene, table, 3, "a"), b = withTextures(scene, table, 3, "b"),
                   c = withTextures(scene, table, 4, "c"), d = withTextures(scene, table, 2, "d");
  const EntityId   ea = drawn(scene, "a", grid, {a});
  drawn(scene, "b", grid, {b});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureInfoLimit = 9});  // entries 1 to 8
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_TRUE(residency.takeNotices().empty());

  // a goes (entries 1-3 free once retired), c needs 4: only a repack finds them
  scene.destroyEntity(ea);
  gpu.beginFrame();
  residency.sync(scene);
  drain(gpu);
  drawn(scene, "c", grid, {c});
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "repacked: ");
  std::vector<std::string> notices = residency.takeNotices();
  ASSERT_EQ(notices.size(), 1u);
  EXPECT_NE(notices[0].find("repacked"), std::string::npos);
  EXPECT_EQ(residency.materialState(uint32_t(residency.materialSlot(c)))->spanCount, 4u);

  // d needs 2 of the 1 left: refused
  drawn(scene, "d", grid, {d});
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "full: ");
  notices = residency.takeNotices();
  ASSERT_FALSE(notices.empty());
  EXPECT_NE(notices.back().find("material 'd' is drawn without its textures"), std::string::npos);
  // once, though the repack converted it a second time
  EXPECT_EQ(std::count_if(notices.begin(), notices.end(), [](const std::string& n) { return n.find("'d'") != std::string::npos; }), 1);
  EXPECT_EQ(residency.materialState(uint32_t(residency.materialSlot(d)))->spanCount, 0u);
  const shaderio::GltfShadeMaterial* row = residency.shadeMaterial(uint32_t(residency.materialSlot(d)));
  ASSERT_NE(row, nullptr);
  EXPECT_EQ(row->pbrBaseColorTexture, 0);
  EXPECT_EQ(row->normalTexture, 0);
}

// A repack converts every slot again: a held edit's slot keeps the payload its row has
TEST(RetainedResidency, AHeldEditSurvivesARepack)
{
  RetainedScene    scene;
  const auto       table = textureTable(scene);
  const MeshId     grid  = scene.addMesh(gridMesh(2, 0.f));
  const MaterialId a = withTextures(scene, table, 3, "a"), b = withTextures(scene, table, 3, "b"),
                   c  = withTextures(scene, table, 4, "c");
  const EntityId   ea = drawn(scene, "a", grid, {a});
  drawn(scene, "b", grid, {b});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.textureInfoLimit = 9});
  residency.setShaderFeatures(nvvkgltf::SceneFeatureSet{});
  residency.replace(scene);
  scene.setMaterial(b, withTransmission(scene.material(b)->payload));
  scene.destroyEntity(ea);
  gpu.beginFrame();
  residency.sync(scene);
  drain(gpu);
  drawn(scene, "c", grid, {c});  // only a repack finds c's four entries
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "repacked: ");
  ASSERT_FALSE(residency.takeNotices().empty());
  EXPECT_EQ(residency.heldMaterials(), std::set<MaterialId>{b});
  EXPECT_EQ(residency.shadeMaterial(uint32_t(residency.materialSlot(b)))->transmissionFactor, 0.f);
}

TEST(RetainedResidency, AnUnreadableTextureLeavesTheDefaultImage)
{
  RetainedScene scene;
  TextureRecord missing;
  missing.uri = "missing.png";
  auto table  = std::make_shared<TextureTable>();
  table->push_back({scene.addTexture(missing), {}});
  MaterialRecord m;
  m.payload.baseColorMap.index = 0;
  m.textures                                            = table;
  drawn(scene, "e", scene.addMesh(gridMesh(1, 0.f)), {scene.addMaterial(m)});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  const int32_t slot = residency.textureSlot((*table)[0].texture, {}, true);
  ASSERT_GT(slot, 0);
  EXPECT_FALSE(gpu.textureDescriptors().at(uint32_t(slot)));
}

// An edit that swaps a texture frees the old slot only at its retire frame, rewritten to the default
// image; one that makes the surface a cutout moves its instances to the blend list and its TLAS rows'
// flags with them.
TEST(RetainedResidency, AMaterialEditMovesItsTexturesListsAndFlags)
{
  RetainedScene    scene;
  const auto       table = textureTable(scene);
  const MaterialId mat   = textured(scene, table, 0);
  drawn(scene, "e", scene.addMesh(gridMesh(2, 0.f)), {mat});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  EXPECT_EQ(residency.drawList(Residency::DrawList::eSolid), std::vector<uint32_t>{0});
  const int32_t  before   = residency.textureSlot((*table)[0].texture, {}, true);
  const uint64_t revision = residency.drawListsRevision();

  fv::mat::Material edited  = scene.material(mat)->payload;
  edited.baseColorMap.index = 1;  // the other sampler
  edited.alphaMode          = fv::mat::AlphaMode::eBlend;
  scene.setMaterial(mat, edited);
  gpu.beginFrame();
  const SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.changed && r.drawListChanged);
  EXPECT_TRUE(r.dlssReset);  // an appearance change motion vectors can't describe
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.textureSlot((*table)[0].texture, {}, true), -1);
  EXPECT_TRUE(gpu.textureDescriptors().at(uint32_t(before)));  // not yet: a frame in flight may sample it
  EXPECT_EQ(residency.drawList(Residency::DrawList::eBlend), std::vector<uint32_t>{0});
  EXPECT_TRUE(residency.drawList(Residency::DrawList::eSolid).empty());
  EXPECT_GT(residency.drawListsRevision(), revision);
  drain(gpu);
  EXPECT_FALSE(gpu.textureDescriptors().at(uint32_t(before)));
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

TEST(RetainedResidency, ANormalMatrixStaysFiniteForAJointScaledToZero)
{
  // the inverse transpose's direction, whatever the scale, the rotation or a mirror
  const glm::mat3 turned = glm::mat3(glm::rotate(glm::mat4(1.f), 0.7f, glm::vec3(0.f, 1.f, 0.f)));
  for(const glm::mat3& m : {turned, 2.f * turned, turned * glm::mat3(glm::scale(glm::mat4(1.f), glm::vec3(1.f, 3.f, 0.5f))), glm::mat3(glm::scale(glm::mat4(1.f), glm::vec3(-1.f, 1.f, 1.f)))})
  {
    const glm::vec3 n(0.3f, 0.5f, 0.8f);
    const glm::vec3 expected = glm::normalize(glm::transpose(glm::inverse(m)) * n);
    const glm::vec3 got      = glm::normalize(skinNormalMatrix(m) * n);
    EXPECT_NEAR(glm::dot(expected, got), 1.f, 1e-5f);
  }
  const glm::mat3 gone = skinNormalMatrix(glm::mat3(0.f));
  for(int c = 0; c < 3; c++)
    for(int r = 0; r < 3; r++)
      EXPECT_TRUE(std::isfinite(gone[c][r]));
}

TEST(RetainedResidency, AFileAlreadyThereLetsGoOfWhatAWorkerDecodedForIt)
{
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "fv residency prefetch.png";
  std::ofstream(file, std::ios::binary) << "not read by the fake";
  RetainedScene scene;
  const auto    record = [&] {
    TextureRecord t;
    t.uri      = file.generic_string();
    t.mimeType = "image/png";
    t.width    = 4;
    t.height   = 4;
    return t;
  };
  const TextureId a = scene.addTexture(record()), same = scene.addTexture(record());
  auto            table = std::make_shared<TextureTable>();
  table->push_back({a, {}});
  table->push_back({same, {}});
  const MeshId grid = scene.addMesh(gridMesh(2, 0.f, 2));
  drawn(scene, "e", grid, {textured(scene, table, 0), textured(scene, table, 1)});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.imageCounts().reused, 1u);
  EXPECT_EQ(gpu.droppedPrefetches, std::vector<std::string>{file.generic_string()});
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
  std::filesystem::remove(file);
}

// Lights are dense in document order; a hidden one shines with nothing, and a moved one is packed
// again where it went.
TEST(RetainedResidency, LightsFollowTheirEntities)
{
  RetainedScene  scene;
  const EntityId sun  = lightEntity(scene, "directional", 3.0);
  const EntityId lamp = lightEntity(scene, "point", 50.0);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.lightCount(), 2u);

  scene.setVisible(lamp, false);
  Transform t;
  t.translation = {0.f, 5.f, 0.f};
  scene.setLocal(sun, t);
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(scene).changed);
  expectClean(scene, residency, gpu);
  shaderio::GltfLight packed[2];
  std::memcpy(packed, gpu.bytesAt(gpu.bufferAddress(residency.lightTable()), sizeof(packed)).data(), sizeof(packed));
  EXPECT_EQ(packed[0].position, glm::vec3(0.f, 5.f, 0.f));
  EXPECT_EQ(packed[1].intensity, 0.f);
}

// The emissive table holds the shown emitters' triangles under their instance and prim slots, and
// follows them as they are hidden, moved and removed.
TEST(RetainedResidency, TheEmissiveTableFollowsItsEmitters)
{
  RetainedScene scene;
  const MeshId  grid = scene.addMesh(gridMesh(2, 0.f));
  drawn(scene, "dark", grid, {material(scene, "dark")});
  const EntityId glow = drawn(scene, "glow", grid, {emissive(scene, 2.f)});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.emissiveCount(), 8u);  // a 2 x 2 grid: eight triangles
  const float weight = residency.emissiveTotalWeight();

  Transform t;
  t.scale = glm::vec3(2.f);  // four times the area
  scene.setLocal(glow, t);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu);
  EXPECT_NEAR(residency.emissiveTotalWeight(), 4.f * weight, 1e-3f);

  scene.setVisible(glow, false);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.emissiveCount(), 0u);

  scene.setVisible(glow, true);
  scene.destroyEntity(scene.roots()[0]);  // the dark one: the glowing instance keeps its slot
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.emissiveCount(), 8u);
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// EXT_lights_ies: a light's row names its profile, tinted and scaled by the node's colour and
// multiplier; the profiles' table follows the store, and a profile added later is uploaded too
TEST(RetainedResidency, IesLightsNameTheirProfiles)
{
  RetainedScene      scene;
  std::vector<float> shaped(181, 1.f);
  shaped[180] = 0.f;
  scene.addIesProfile({.name = "flat", .samples = std::vector<float>(181, 1.f)});
  const uint32_t profile = scene.addIesProfile({.name = "shaped", .samples = shaped});
  const EntityId lamp    = lightEntity(scene, "spot", 40.0);
  const EntityId plain   = lightEntity(scene, "point", 5.0);
  const LightId  light   = scene.entity(lamp)->light;
  scene.setLightIes(light, IesLight{.profile = profile, .multiplier = 2.f, .color = {1.f, 0.5f, 0.5f}});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.iesProfileCount(), 2u);

  scene.setLightIes(light, IesLight{.profile = profile, .multiplier = 6.f});
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(scene).changed);
  expectClean(scene, residency, gpu, "a multiplier: ");

  scene.addIesProfile({.name = "late", .samples = std::vector<float>(181, 0.25f)});
  scene.setLightIes(scene.entity(plain)->light, IesLight{.profile = 2});
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "a profile added: ");
  EXPECT_EQ(residency.iesProfileCount(), 3u);
}

// An emitter's vertex writes (a dial drag) leave the table's areas as they were until the writes
// stop for emissiveSettleFrames, then one rebuild takes the last ones; a non-emitter's writes don't
// touch it, and any other emitter change rebuilds at once
// An emissive texture that lands after the table was built gives its emitter the weight of its mean:
// the light samples and a BSDF ray's MIS weight must read the same one (I6). Unknown until then, the
// mean counts as 1; a texture that lands again with the same mean leaves the table as it was.
TEST(RetainedResidency, AnEmissiveTextureThatLandsLateReweighsItsEmitter)
{
  RetainedScene   scene;
  const TextureId mask  = scene.addTexture(chainTexture("chain_sign_mean=0.25", 64));
  auto            table = std::make_shared<TextureTable>();
  table->push_back({mask, {}});
  MaterialRecord sign;
  sign.payload.name              = "sign";
  sign.payload.emissive          = {2, 2, 2};
  sign.payload.emissiveMap.index = 0;
  sign.textures                  = table;
  const MaterialId signMaterial  = scene.addMaterial(std::move(sign));
  drawn(scene, "sign", scene.addMesh(gridMesh(2, 0.f)), {signMaterial});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  gpu.holdDecodes(true);
  residency.replace(scene);
  expectClean(scene, residency, gpu, "decoding: ");
  const float unknown = residency.emissiveTotalWeight();
  ASSERT_GT(unknown, 0.f);

  gpu.holdDecodes(false);
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(scene).changed);
  ASSERT_TRUE(residency.image(mask, true));
  expectClean(scene, residency, gpu, "landed: ");
  EXPECT_NEAR(residency.emissiveTotalWeight(), 0.25f * unknown, 1e-4f * unknown) << "one emitter: its weight is its own luminance";
  for(int frame = 0; frame < 3; frame++)
  {
    gpu.beginFrame();
    residency.sync(scene);
  }
  gpu.beginFrame();
  EXPECT_FALSE(residency.sync(scene).changed) << "nothing moved since";
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// Row `slot` of a GPU table, as the fake holds it
template <typename T>
T tableRow(const FakeGpuBackend& gpu, BufferHandle table, size_t slot)
{
  T                             row{};
  const FakeGpuBackend::Buffer* b = gpu.buffer(table);
  if(b && (slot + 1) * sizeof(T) <= b->bytes.size())
    std::memcpy(&row, b->bytes.data() + slot * sizeof(T), sizeof(T));
  return row;
}

// A textured emitter's triangles are weighed by their own texels once those are averaged: asked for
// once, in the sync that builds the table, and weighed whole until they come; the black ones are then
// never sampled. An edit that leaves the texture and the mesh as they were reuses them.
TEST(RetainedResidency, ATexturedEmittersTrianglesWeighTheirOwnTexels)
{
  RetainedScene   scene;
  const TextureId mask  = scene.addTexture(chainTexture("chain_sign_half_mean=0.5", 64));  // lit where u >= 0.5
  auto            table = std::make_shared<TextureTable>();
  table->push_back({mask, {}});
  MaterialRecord sign;
  sign.payload.name              = "sign";
  sign.payload.emissive          = {2, 2, 2};
  sign.payload.emissiveMap.index = 0;
  sign.textures                  = table;
  const MaterialId signMaterial  = scene.addMaterial(std::move(sign));
  drawn(scene, "sign", scene.addMesh(gridMesh(2, 0.f)), {signMaterial});  // u = x / 2: two columns of four triangles
  FakeGpuBackend gpu(2);
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu, "asked: ");
  EXPECT_EQ(gpu.texelAveragesAsked(), 8u);
  ASSERT_EQ(residency.emissiveCount(), 8u);
  const auto weights = [&] {
    std::vector<float> out(8);
    for(size_t i = 0; i < 8; i++)
      out[i] = tableRow<shaderio::EmissiveTriangle>(gpu, residency.emissiveTable(), i).weight;
    return out;
  };
  for(float s : weights())
    EXPECT_GT(s, 0.f) << "weighed whole until the averages come";

  for(int frame = 0; frame < 4; frame++)
  {
    gpu.beginFrame();
    residency.sync(scene);
  }
  expectClean(scene, residency, gpu, "landed: ");
  EXPECT_EQ(residency.texelMeansHeld(), 1u);
  const std::vector<float> landed = weights();
  for(size_t t = 0; t < 8; t++)
  {
    const bool lit = (t / 2) % 2 == 1;  // cells (y, x) in order, two triangles each: x = 1 is u >= 0.5
    if(lit)
      EXPECT_NEAR(landed[t], 0.5f * 2.f, 1e-5f) << "triangle " << t << ": half a cell at the factor's luminance, its only emitter's mean";
    else
      EXPECT_EQ(landed[t], 0.f) << "triangle " << t << " is black";
  }

  // the strength changes: the texels are as they were, so nothing is asked again
  fv::mat::Material brighter = scene.material(signMaterial)->payload;
  brighter.emissive          = {4, 4, 4};
  scene.setMaterial(signMaterial, brighter);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "brighter: ");
  EXPECT_EQ(gpu.texelAveragesAsked(), 8u);
  EXPECT_NEAR(tableRow<shaderio::EmissiveTriangle>(gpu, residency.emissiveTable(), 2).weight, 0.5f * 4.f, 1e-5f);
  EXPECT_EQ(tableRow<shaderio::EmissiveTriangle>(gpu, residency.emissiveTable(), 0).weight, 0.f);
  drain(gpu);
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

TEST(RetainedResidency, AnEmittersVertexWritesRebuildTheTableOnceTheyStop)
{
  RetainedScene scene;
  const MeshId  glowing = scene.addMesh(gridMesh(2, 0.f));
  const MeshId  dark    = scene.addMesh(gridMesh(2, 1.f));
  drawn(scene, "dark", dark, {material(scene, "dark")});
  const MaterialId glow = emissive(scene, 2.f);
  drawn(scene, "glow", glowing, {glow});
  FakeGpuBackend gpu(2);
  Residency      residency(gpu, {.emissiveSettleFrames = 3});
  residency.replace(scene);
  const float                  weight = residency.emissiveTotalWeight();
  const std::vector<glm::vec3> base   = scene.mesh(glowing)->positions;
  const auto                   scaled = [&](float k) {
    std::vector<glm::vec3> out = base;
    for(glm::vec3& p : out)
      p *= k;
    return out;
  };

  // the drag: two frames of writes, then quiet ones; the weights lag until the third quiet frame
  scene.setMeshVertices(glowing, scaled(2.f), {});
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(scene).verticesWritten);
  EXPECT_TRUE(residency.emissiveSettling());
  EXPECT_EQ(residency.emissiveTotalWeight(), weight);
  expectClean(scene, residency, gpu, "dragging: ");
  scene.setMeshVertices(glowing, scaled(3.f), {});
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_EQ(residency.emissiveTotalWeight(), weight);
  for(int quiet = 1; quiet <= 2; quiet++)
  {
    gpu.beginFrame();
    EXPECT_FALSE(residency.sync(scene).changed) << quiet;
    EXPECT_EQ(residency.emissiveTotalWeight(), weight) << quiet;
  }
  gpu.beginFrame();
  EXPECT_TRUE(residency.sync(scene).changed);
  EXPECT_FALSE(residency.emissiveSettling());
  EXPECT_NEAR(residency.emissiveTotalWeight(), 9.f * weight, 1e-2f);
  expectClean(scene, residency, gpu, "settled: ");
  gpu.beginFrame();
  EXPECT_FALSE(residency.sync(scene).changed);  // once

  // a non-emitter's writes leave the table alone
  std::vector<glm::vec3> moved = scene.mesh(dark)->positions;
  for(glm::vec3& p : moved)
    p.z += 1.f;
  scene.setMeshVertices(dark, moved, {});
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_FALSE(residency.emissiveSettling());

  // a material edit while settling rebuilds at once, from the latest positions
  scene.setMeshVertices(glowing, base, {});
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_TRUE(residency.emissiveSettling());
  fv::mat::Material payload = scene.material(glow)->payload;
  payload.doubleSided       = !payload.doubleSided;
  scene.setMaterial(glow, payload);
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_FALSE(residency.emissiveSettling());
  EXPECT_NEAR(residency.emissiveTotalWeight(), weight, 1e-3f);
  expectClean(scene, residency, gpu, "material edit: ");
}

// Oracle 2b: random edits, each synced and checked; at the end the scene made resident afresh holds
// the same draws, the same geometry and the same transforms (slots may differ, content may not).
TEST(RetainedResidency, IncrementalEqualsFresh)
{
  uint32_t morphs = 0, skins = 0;  // the passes the seeds ran, so the fuzz is known to reach them
  for(uint32_t seed = 1; seed <= 20; seed++)
  {
    std::mt19937            rng(seed);
    RetainedScene           scene;
    std::vector<MeshId>     meshes;
    std::vector<MaterialId> materials;
    for(int i = 0; i < 4; i++)
      meshes.push_back(scene.addMesh(gridMesh(1 + i, float(i), 1 + i % 2)));
    // and one that deforms: two bones skin it, one target lifts it
    MeshRecord deforming = gridMesh(2, 5.f, 2);
    for(const glm::vec3& p : deforming.positions)
    {
      deforming.joints.push_back({0, 1, 0, 0});
      deforming.weights.push_back({1.f - p.x * 0.5f, p.x * 0.5f, 0.f, 0.f});
    }
    deforming.targets.push_back({.positions = std::vector<glm::vec3>(deforming.positions.size(), glm::vec3(0, 0, 1))});
    deforming.targetWeights = {0.f};
    const MeshId   deformingMesh = scene.addMesh(deforming);
    const EntityId bone0 = scene.createEntity("bone0", {}), bone1 = scene.createEntity("bone1", {}, bone0);
    const SkinId   skin  = scene.addSkin({.name = "skin", .joints = {bone0, bone1}, .inverseBind = {glm::mat4(1.f), glm::mat4(1.f)}});
    meshes.push_back(deformingMesh);
    for(const char* name : {"a", "b", "c"})
      materials.push_back(material(scene, name));
    const auto table = textureTable(scene);
    materials.push_back(textured(scene, table, 0));
    materials.push_back(textured(scene, table, 1, fv::mat::AlphaMode::eBlend));
    materials.push_back(emissive(scene, 3.f));
    materials.push_back(textured(scene, table, 2));  // the normal map's texture sampled as colour too
    std::vector<EntityId> live;

    FakeGpuBackend gpu(2);
    // odd seeds compact the static BLASes, every third builds a BLAS again after two refits, and seeds
    // 2 and 3 of every four compact the deformed ones (refitting their copies)
    Residency residency(gpu, {.minInstances         = 4,
                              .minPrims             = 4,
                              .emissiveSettleFrames = 3,
                              .refitsBeforeRebuild  = seed % 3 == 0 ? 2u : 0u,
                              .compactStatic        = seed % 2 == 1,
                              .compactDeforming     = seed % 4 >= 2});
    residency.replace(scene);
    const auto pick = [&](auto& v) -> auto& { return v[std::uniform_int_distribution<size_t>(0, v.size() - 1)(rng)]; };

    for(int step = 0; step < 60; step++)
    {
      const int op = std::uniform_int_distribution<int>(0, 9)(rng);
      if(op <= 1 || live.empty())
      {
        const MeshId            mesh = pick(meshes);
        std::vector<MaterialId> mats;
        for(size_t s = 0; s < scene.mesh(mesh)->submeshes.size(); s++)
          mats.push_back(pick(materials));
        const EntityId parent = !live.empty() && rng() % 3 == 0 ? pick(live) : EntityId{};
        live.push_back(drawn(scene, "e", mesh, mats, parent));
        if(mesh == deformingMesh && rng() % 2)  // skinned too, not only morphed
        {
          MeshInstance instance = *scene.entity(live.back())->mesh;
          instance.skin         = skin;
          scene.setMeshInstance(live.back(), instance);
        }
        if(rng() % 4 == 0)
          live.push_back(lightEntity(scene, rng() % 2 ? "point" : "spot", 10.0, live.back()));
      }
      else if(op == 2)
      {
        const EntityId e = pick(live);
        scene.destroyEntity(e);
        live.erase(std::remove_if(live.begin(), live.end(), [&](EntityId id) { return !scene.entity(id); }), live.end());
      }
      else if(op == 3)
      {
        Transform t;
        t.translation = {float(rng() % 7), float(rng() % 5), 0.f};
        scene.setLocal(pick(live), t);
      }
      else if(op == 4)
        scene.setVisible(pick(live), rng() % 2 == 0);
      else if(op == 7)
      {
        // a pose: the second bone turns
        Transform t;
        t.translation = {1, 0, 0};
        t.rotation    = glm::angleAxis(float(rng() % 90) * 0.01f, glm::vec3(0, 1, 0));
        scene.setLocal(bone1, t);
      }
      else if(op == 8)
        scene.setMorphWeights(pick(live), {float(rng() % 5) * 0.25f});
      else if(op == 9)
      {
        // a rebinding: the same meshes, other materials
        const EntityId e = pick(live);
        if(const Entity* entity = scene.entity(e); entity && entity->mesh)
        {
          MeshInstance instance = *entity->mesh;
          for(MeshPart& part : instance.parts)
            for(MaterialId& m : part.materials)
              m = pick(materials);
          scene.setMeshInstance(e, instance);
        }
      }
      else if(op == 6)
      {
        // a material edit: another texture, another alpha mode, sides
        const MaterialId   m       = pick(materials);
        fv::mat::Material payload = scene.material(m)->payload;
        static const fv::mat::AlphaMode modes[] = {fv::mat::AlphaMode::eOpaque, fv::mat::AlphaMode::eMask, fv::mat::AlphaMode::eBlend};
        payload.alphaMode   = modes[rng() % 3];
        payload.doubleSided = rng() % 2 == 0;
        if(scene.material(m)->textures)
          payload.baseColorMap.index = int(rng() % 3) - 1;
        scene.setMaterial(m, payload);
      }
      else
      {
        const MeshId           mesh  = pick(meshes);
        std::vector<glm::vec3> moved = scene.mesh(mesh)->positions;
        for(glm::vec3& p : moved)
          p.z += 0.25f;
        scene.setMeshVertices(mesh, moved, {});
      }
      // what isn't being edited settles, and wakes for whatever comes
      if(rng() % 5 == 0)
        residency.settle(pick(meshes));
      if(rng() % 7 == 0)
        residency.setLive(rng() % 2 == 0 ? std::set<MeshId>{pick(meshes)} : std::set<MeshId>{});
      gpu.beginFrame();
      residency.sync(scene);
      morphs += residency.deformCounts().morphs;
      skins += residency.deformCounts().skins;
      expectClean(scene, residency, gpu, "seed " + std::to_string(seed) + " step " + std::to_string(step) + ": ");
    }
    // the emissive table settles once the vertex writes stop
    for(int frame = 0; frame < 3; frame++)
    {
      gpu.beginFrame();
      residency.sync(scene);
    }
    EXPECT_FALSE(residency.emissiveSettling()) << "seed " << seed;
    expectClean(scene, residency, gpu, "settled, seed " + std::to_string(seed) + ": ");
    EXPECT_EQ(gpu.waitIdleCount(), 0u);
    drain(gpu);
    for(const std::string& p : residencyLeaks(residency, gpu))
      ADD_FAILURE() << "seed " << seed << ": " << p;

    // the same scene made resident afresh
    FakeGpuBackend fresh;
    Residency      again(fresh, {.minInstances = 4, .minPrims = 4});
    again.replace(scene);
    expectClean(scene, again, fresh, "fresh, seed " + std::to_string(seed) + ": ");
    const auto draws = [](const Residency& r, const FakeGpuBackend& g) {
      std::multimap<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::tuple<glm::mat4, bool, uint64_t>> out;
      for(uint32_t s = 0; s < r.instanceHighWater(); s++)
        if(const Residency::Instance* i = r.instance(s))
          out.emplace(std::tuple{i->entity.index, i->mesh.index, i->submesh, i->copy},
                      std::tuple{i->world, i->shown, g.blas(r.prim(i->prim)->blas)->contentHash});
      return out;
    };
    EXPECT_TRUE(draws(residency, gpu) == draws(again, fresh)) << "seed " << seed;
  }
  EXPECT_GT(morphs, 20u);
  EXPECT_GT(skins, 20u);
}

// Review findings, 2026-10-27: a rebinding to a material that is already drawn elsewhere moves the
// instance's raster list and its emitters; a texture sampled as colour by one material and as data by
// another; a held edit that starts emitting when its variant comes.
TEST(RetainedResidency, ARebindingToALiveMaterialMovesItsListsAndEmitters)
{
  RetainedScene    scene;
  const MeshId     grid  = scene.addMesh(gridMesh(2, 0.f));
  const MaterialId plain = material(scene, "plain"), lamp = emissive(scene, 2.f);
  MaterialRecord   blend;
  blend.payload.alphaMode  = fv::mat::AlphaMode::eBlend;
  const MaterialId glassy  = scene.addMaterial(blend);
  const EntityId   e       = drawn(scene, "e", grid, {plain});
  drawn(scene, "keep lamp", grid, {lamp});
  drawn(scene, "keep glassy", grid, {glassy});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  for(MaterialId m : {lamp, plain, glassy, plain})
  {
    MeshInstance mi;
    mi.parts.push_back({grid, {m}});
    scene.setMeshInstance(e, mi);
    gpu.beginFrame();
    residency.sync(scene);
    expectClean(scene, residency, gpu, "rebound: ");
  }
}

TEST(RetainedResidency, ATextureSampledAsColourAnywhereIsDecodedAsColour)
{
  RetainedScene scene;
  const auto    table = textureTable(scene);
  MaterialRecord normalOnly;
  normalOnly.payload.normalMap.index = 2;
  normalOnly.textures                    = table;
  const MaterialId n    = scene.addMaterial(normalOnly);
  const MeshId     grid = scene.addMesh(gridMesh(2, 0.f));
  drawn(scene, "normal", grid, {n});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu, "normal only: ");
  MaterialRecord colour;
  colour.payload.baseColorMap.index = 2;
  colour.textures                                            = table;
  drawn(scene, "colour", grid, {scene.addMaterial(colour)});
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "colour too: ");
}

TEST(RetainedResidency, AHeldEditThatEmitsJoinsTheEmissiveTableWhenItIsWritten)
{
  RetainedScene    scene;
  const MaterialId plain = material(scene, "plain");
  const MeshId     grid  = scene.addMesh(gridMesh(2, 0.f));
  drawn(scene, "a", grid, {plain});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.setShaderFeatures(nvvkgltf::SceneFeatureSet{});
  residency.replace(scene);
  fv::mat::Material glowing = scene.material(plain)->payload;
  glowing.emissive          = {2, 2, 2};
  glowing.clearcoat         = fv::mat::Clearcoat{.factor = 1};
  scene.setMaterial(plain, glowing);
  gpu.beginFrame();
  residency.sync(scene);
  ASSERT_EQ(residency.heldMaterials().size(), 1u);
  expectClean(scene, residency, gpu, "held: ");
  residency.setShaderFeatures(scene.features());
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_TRUE(residency.heldMaterials().empty());
  EXPECT_GT(residency.emissiveCount(), 0u);
  expectClean(scene, residency, gpu, "written: ");
}
