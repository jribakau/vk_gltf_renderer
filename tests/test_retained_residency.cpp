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

#include <map>
#include <random>

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
MaterialId textured(RetainedScene& scene, std::shared_ptr<const TextureTable> table, int base, const char* alpha = "OPAQUE")
{
  MaterialRecord m;
  m.payload.name                                        = "textured";
  m.payload.alphaMode                                   = alpha;
  m.payload.pbrMetallicRoughness.baseColorTexture.index = base;
  m.payload.normalTexture.index                         = 2;
  m.textures                                            = std::move(table);
  return scene.addMaterial(std::move(m));
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
  const int32_t plain   = residency.textureSlot(a, {}),
                nearest = residency.textureSlot(a, {.magFilter = TINYGLTF_TEXTURE_FILTER_NEAREST});
  EXPECT_GT(plain, 0);
  EXPECT_GT(nearest, 0);
  EXPECT_NE(plain, nearest);
  EXPECT_EQ(gpu.textureDescriptors().at(uint32_t(plain)), gpu.textureDescriptors().at(uint32_t(nearest)));  // one image
  EXPECT_TRUE(gpu.image(residency.image(a))->srgb);
  EXPECT_FALSE(gpu.image(residency.image(b))->srgb);
  EXPECT_EQ(residency.materialHighWater(), 2u);
  // the normal map's two channels reach its texture info
  const Residency::MaterialState* state = residency.materialState(0);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->slots.isTwoChannelSource(2), true);
}

TEST(RetainedResidency, AnUnreadableTextureLeavesTheDefaultImage)
{
  RetainedScene scene;
  TextureRecord missing;
  missing.uri = "missing.png";
  auto table  = std::make_shared<TextureTable>();
  table->push_back({scene.addTexture(missing), {}});
  MaterialRecord m;
  m.payload.pbrMetallicRoughness.baseColorTexture.index = 0;
  m.textures                                            = table;
  drawn(scene, "e", scene.addMesh(gridMesh(1, 0.f)), {scene.addMaterial(m)});
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu);
  const int32_t slot = residency.textureSlot((*table)[0].texture, {});
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
  const int32_t  before   = residency.textureSlot((*table)[0].texture, {});
  const uint64_t revision = residency.drawListsRevision();

  tinygltf::Material edited                          = scene.material(mat)->payload;
  edited.pbrMetallicRoughness.baseColorTexture.index = 1;  // the other sampler
  edited.alphaMode                                   = "MASK";
  scene.setMaterial(mat, edited);
  gpu.beginFrame();
  const SyncResult r = residency.sync(scene);
  EXPECT_TRUE(r.changed && r.drawListChanged);
  expectClean(scene, residency, gpu);
  EXPECT_EQ(residency.textureSlot((*table)[0].texture, {}), -1);
  EXPECT_TRUE(gpu.textureDescriptors().at(uint32_t(before)));  // not yet: a frame in flight may sample it
  EXPECT_EQ(residency.drawList(Residency::DrawList::eBlend), std::vector<uint32_t>{0});
  EXPECT_TRUE(residency.drawList(Residency::DrawList::eSolid).empty());
  EXPECT_GT(residency.drawListsRevision(), revision);
  drain(gpu);
  EXPECT_FALSE(gpu.textureDescriptors().at(uint32_t(before)));
  for(const std::string& p : residencyLeaks(residency, gpu))
    ADD_FAILURE() << p;
}

// Oracle 2b: random edits, each synced and checked; at the end the scene made resident afresh holds
// the same draws, the same geometry and the same transforms (slots may differ, content may not).
TEST(RetainedResidency, IncrementalEqualsFresh)
{
  for(uint32_t seed = 1; seed <= 20; seed++)
  {
    std::mt19937            rng(seed);
    RetainedScene           scene;
    std::vector<MeshId>     meshes;
    std::vector<MaterialId> materials;
    for(int i = 0; i < 4; i++)
      meshes.push_back(scene.addMesh(gridMesh(1 + i, float(i), 1 + i % 2)));
    for(const char* name : {"a", "b", "c"})
      materials.push_back(material(scene, name));
    const auto table = textureTable(scene);
    materials.push_back(textured(scene, table, 0));
    materials.push_back(textured(scene, table, 1, "BLEND"));
    std::vector<EntityId> live;

    FakeGpuBackend gpu(2);
    Residency      residency(gpu, {.minInstances = 4, .minPrims = 4});
    residency.replace(scene);
    const auto pick = [&](auto& v) -> auto& { return v[std::uniform_int_distribution<size_t>(0, v.size() - 1)(rng)]; };

    for(int step = 0; step < 60; step++)
    {
      const int op = std::uniform_int_distribution<int>(0, 6)(rng);
      if(op <= 1 || live.empty())
      {
        const MeshId            mesh = pick(meshes);
        std::vector<MaterialId> mats;
        for(size_t s = 0; s < scene.mesh(mesh)->submeshes.size(); s++)
          mats.push_back(pick(materials));
        const EntityId parent = !live.empty() && rng() % 3 == 0 ? pick(live) : EntityId{};
        live.push_back(drawn(scene, "e", mesh, mats, parent));
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
      else if(op == 6)
      {
        // a material edit: another texture, another alpha mode, sides
        const MaterialId   m       = pick(materials);
        tinygltf::Material payload = scene.material(m)->payload;
        static const char* modes[] = {"OPAQUE", "MASK", "BLEND"};
        payload.alphaMode          = modes[rng() % 3];
        payload.doubleSided        = rng() % 2 == 0;
        if(scene.material(m)->textures)
          payload.pbrMetallicRoughness.baseColorTexture.index = int(rng() % 3) - 1;
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
      gpu.beginFrame();
      residency.sync(scene);
      expectClean(scene, residency, gpu, "seed " + std::to_string(seed) + " step " + std::to_string(step) + ": ");
    }
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
}
