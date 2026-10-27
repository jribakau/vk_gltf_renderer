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

#include <cstring>
#include <filesystem>
#include <fstream>
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

// The payload with KHR_materials_transmission (factor 1), a feature the shader can compile out
tinygltf::Material withTransmission(tinygltf::Material payload)
{
  tinygltf::Value::Object transmission;
  transmission["transmissionFactor"]               = tinygltf::Value(1.0);
  payload.extensions["KHR_materials_transmission"] = tinygltf::Value(transmission);
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

MaterialId emissive(RetainedScene& scene, float strength)
{
  MaterialRecord m;
  m.payload.name           = "lamp";
  m.payload.emissiveFactor = {strength, strength, strength};
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
  tinygltf::Material red                   = scene.material(plain)->payload;
  red.pbrMetallicRoughness.baseColorFactor = {1, 0, 0, 1};
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
  EXPECT_EQ(residency.image(a), residency.image(same));
  EXPECT_EQ(gpu.liveImages(), 2u);
  EXPECT_EQ(residency.imageCounts().decoded, 2u);
  EXPECT_EQ(residency.imageCounts().reused, 1u);
  EXPECT_EQ(gpu.imageBatches(), std::vector<size_t>{2});  // decoded together, the shared one once

  // a replace decodes nothing it had
  const ImageHandle before = residency.image(b);
  gpu.beginFrame();
  residency.replace(scene);
  expectClean(scene, residency, gpu, "replaced: ");
  EXPECT_EQ(residency.imageCounts().decoded, 0u);
  EXPECT_EQ(residency.imageCounts().reused, 3u);
  EXPECT_EQ(gpu.imageBatches().size(), 1u);  // nothing to decode
  EXPECT_EQ(residency.image(b), before);
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
  int* fields[]  = {&m.payload.pbrMetallicRoughness.baseColorTexture.index, &m.payload.normalTexture.index,
                    &m.payload.emissiveTexture.index, &m.payload.occlusionTexture.index,
                    &m.payload.pbrMetallicRoughness.metallicRoughnessTexture.index};
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
  tinygltf::Material payload                   = scene.material(a)->payload;
  payload.pbrMetallicRoughness.baseColorFactor = {0.5, 0.5, 0.5, 1.0};
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
  payload.emissiveTexture.index  = 0;
  payload.occlusionTexture.index = 0;
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
  tinygltf::Material payload = scene.material(glow)->payload;
  payload.doubleSided        = !payload.doubleSided;
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
    materials.push_back(emissive(scene, 3.f));
    std::vector<EntityId> live;

    FakeGpuBackend gpu(2);
    Residency      residency(gpu, {.minInstances = 4, .minPrims = 4, .emissiveSettleFrames = 3});
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
}
