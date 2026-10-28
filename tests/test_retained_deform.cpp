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

// Deformation on the retained scene (daz-viewer docs/retained-scene-plan.md, P3b): skinned and
// morphed meshes drawn through their drawers on the fake GPU, which runs each pass as the shader
// does. Every sync is checked by ResidencyOracle, which deforms the mesh on the CPU from the scene
// as it is now (I22) and walks the DLSS motion tables (I10). The CPU arithmetic is the one
// test_retained_animation holds to nvvkgltf::AnimationSystem.

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "common/fake_gpu_backend.hpp"
#include "common/residency_oracle.hpp"
#include "scene_clip_player.hpp"
#include "scene_import_gltf.hpp"
#include "scene_residency.hpp"
#include "scene_retained.hpp"

using namespace rs;

namespace {

// A 2 x 2 grid of quads in the XY plane, two submeshes; the left column of vertices on joint 0, the
// middle half and half, the right on joint 1
MeshRecord skinnedGrid(bool withTargets, bool withTangents = false)
{
  MeshRecord m;
  m.name = "skinned grid";
  for(int y = 0; y <= 2; y++)
    for(int x = 0; x <= 2; x++)
    {
      m.positions.push_back({float(x), float(y), 0.f});
      m.normals.push_back({0, 0, 1});
      m.uv0.push_back({x * 0.5f, y * 0.5f});
      if(withTangents)
        m.tangents.push_back({1, 0, 0, 1});
      m.joints.push_back({0, 1, 0, 0});
      m.weights.push_back(x == 0 ? glm::vec4(1, 0, 0, 0) : x == 1 ? glm::vec4(0.5f, 0.5f, 0, 0) : glm::vec4(0, 1, 0, 0));
    }
  for(int g = 0; g < 2; g++)
  {
    std::vector<uint32_t> run;
    for(int x = 0; x < 2; x++)
    {
      const uint32_t a = uint32_t(g * 3 + x), b = a + 1, c = a + 3, d = c + 1;
      run.insert(run.end(), {a, b, d, a, d, c});
    }
    m.submeshes.push_back({.firstIndex = uint32_t(m.indices.size()), .indexCount = uint32_t(run.size()), .material = uint32_t(g), .order = g});
    m.indices.insert(m.indices.end(), run.begin(), run.end());
  }
  if(withTargets)
  {
    // target 0 lifts every vertex, target 1 pushes the right column along x and tips its normals
    MorphTarget lift, push;
    lift.positions.assign(m.positions.size(), glm::vec3(0, 0, 1));
    push.positions.assign(m.positions.size(), glm::vec3(0.f));
    push.normals.assign(m.positions.size(), glm::vec3(0.f));
    for(size_t v = 2; v < m.positions.size(); v += 3)
    {
      push.positions[v] = {0.5f, 0, 0};
      push.normals[v]   = {1, 0, 0};
    }
    m.targets       = {lift, push};
    m.targetWeights = {0.f, 0.f};
  }
  return m;
}

// A root, two bones (the second at x = 1 under the first) and an entity skinning the grid with them
struct Rig
{
  RetainedScene scene;
  EntityId      root, bone0, bone1, body;
  MeshId        mesh;
  SkinId        skin;
  MaterialId    material;

  explicit Rig(bool withTargets, bool withTangents = false)
  {
    root  = scene.createEntity("root", {});
    bone0 = scene.createEntity("bone0", {}, root);
    bone1 = scene.createEntity("bone1", {}, bone0);
    Transform t;
    t.translation = {1, 0, 0};
    scene.setLocal(bone1, t);
    mesh     = scene.addMesh(skinnedGrid(withTargets, withTangents));
    material = scene.addMaterial({});
    skin = scene.addSkin({.name = "skin", .joints = {bone0, bone1}, .inverseBind = {glm::mat4(1.f), glm::translate(glm::mat4(1.f), {-1, 0, 0})}});
    body = drawSkinned(scene, mesh, skin, material, root);
  }

  static EntityId drawSkinned(RetainedScene& scene, MeshId mesh, SkinId skin, MaterialId material, EntityId parent)
  {
    const EntityId e = scene.createEntity("body", {}, parent);
    MeshInstance   instance;
    instance.parts.push_back({mesh, {material, material}});
    instance.skin = skin;
    scene.setMeshInstance(e, instance);
    return e;
  }

  void bend(float degrees)
  {
    Transform t;
    t.translation = {1, 0, 0};
    t.rotation    = glm::angleAxis(glm::radians(degrees), glm::vec3(0, 1, 0));
    scene.setLocal(bone1, t);
  }
};

void expectClean(RetainedScene& scene, const Residency& residency, const FakeGpuBackend& gpu, const std::string& when = "")
{
  for(const std::string& p : checkResidency(scene, residency, gpu))
    ADD_FAILURE() << when << p;
}

void drain(FakeGpuBackend& gpu)
{
  for(uint32_t i = 0; i < gpu.framesInFlight(); i++)
    gpu.beginFrame();
}

template <class T>
std::vector<T> readBack(const FakeGpuBackend& gpu, BufferHandle b)
{
  const FakeGpuBackend::Buffer* buffer = gpu.buffer(b);
  std::vector<T>                out(buffer ? buffer->bytes.size() / sizeof(T) : 0);
  if(buffer)
    std::memcpy(out.data(), buffer->bytes.data(), out.size() * sizeof(T));
  return out;
}

uint32_t refitsOf(const Residency& residency, const FakeGpuBackend& gpu, const Residency::DrawerGpu& d)
{
  uint32_t n = 0;
  for(int32_t p : d.prims)
    if(p >= 0)
      n += gpu.blas(residency.prim(uint32_t(p))->blas)->refits;
  return n;
}

}  // namespace

// A skinned mesh is deformed for the entity that draws it: its prims are the drawer's, their BLASes
// built over the skin's output, which holds the mesh at the pose
TEST(RetainedDeform, ASkinnedMeshIsDrawnFromItsSkinsOutput)
{
  Rig rig(false, true);
  rig.bend(30);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  expectClean(rig.scene, residency, gpu);
  EXPECT_EQ(residency.deformCounts().skins, 1u);
  EXPECT_EQ(residency.deformCounts().morphs, 0u);
  const Residency::DrawerGpu* d = residency.drawer(rig.body, rig.mesh);
  ASSERT_NE(d, nullptr);
  ASSERT_EQ(d->prims.size(), 2u);
  for(int32_t p : d->prims)
  {
    ASSERT_GE(p, 0);
    EXPECT_EQ(residency.prim(uint32_t(p))->drawer, rig.body);
  }
  // the bent bone moved the right column out of the plane
  const std::vector<glm::vec3> skinned = readBack<glm::vec3>(gpu, d->positions);
  EXPECT_NE(skinned[2].z, 0.f);
  EXPECT_EQ(readBack<glm::vec3>(gpu, residency.meshGpu(rig.mesh)->positions), rig.scene.mesh(rig.mesh)->positions);
}

// The skin turns the normals by the packed normal matrices, read at their stride as the shader does
// (P3b's readback): bone 1 turned a quarter about y takes the right column's normals from +z to +x,
// the left column's stay, and the middle's are the normalised half and half
TEST(RetainedDeform, ASkinTurnsTheNormalsByItsBones)
{
  Rig rig(false);
  rig.bend(90);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  const std::vector<glm::vec3> normals = readBack<glm::vec3>(gpu, residency.drawer(rig.body, rig.mesh)->normals);
  ASSERT_EQ(normals.size(), 9u);
  const float half = std::sqrt(0.5f);
  for(int y = 0; y <= 2; y++)
  {
    const glm::vec3 left = normals[size_t(y * 3)], middle = normals[size_t(y * 3 + 1)], right = normals[size_t(y * 3 + 2)];
    EXPECT_NEAR(glm::distance(left, glm::vec3(0, 0, 1)), 0.f, 1e-5f) << y;
    EXPECT_NEAR(glm::distance(middle, glm::vec3(half, 0, half)), 0.f, 1e-5f) << y;
    EXPECT_NEAR(glm::distance(right, glm::vec3(1, 0, 0)), 0.f, 1e-5f) << y;
  }
}

// A pose runs the skin again and refits its prims; a frame that moves nothing it reads runs nothing
TEST(RetainedDeform, APoseRunsTheSkinAndOnlyAPoseDoes)
{
  Rig            rig(false);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  const Residency::DrawerGpu& d = *residency.drawer(rig.body, rig.mesh);

  rig.bend(45);
  gpu.beginFrame();
  SyncResult r = residency.sync(rig.scene);
  EXPECT_TRUE(r.changed && r.verticesWritten);
  EXPECT_EQ(residency.deformCounts().skins, 1u);
  EXPECT_EQ(refitsOf(residency, gpu, d), 2u);
  expectClean(rig.scene, residency, gpu, "posed: ");

  // an entity the skin does not read moves: its rows change, the skin does not run
  const EntityId other = rig.scene.createEntity("other", {}, rig.root);
  gpu.beginFrame();
  residency.sync(rig.scene);
  Transform t;
  t.translation = {0, 5, 0};
  rig.scene.setLocal(other, t);
  gpu.beginFrame();
  residency.sync(rig.scene);
  EXPECT_EQ(residency.deformCounts().skins, 0u);
  EXPECT_EQ(refitsOf(residency, gpu, d), 2u);
  expectClean(rig.scene, residency, gpu, "unrelated move: ");

  // and with nothing changed, after the motion has come to rest, a sync records nothing
  gpu.beginFrame();
  residency.sync(rig.scene);
  gpu.beginFrame();
  const size_t barriers = gpu.barriers().size();
  r                     = residency.sync(rig.scene);
  EXPECT_FALSE(r.changed);
  EXPECT_EQ(gpu.barriers().size(), barriers);
  expectClean(rig.scene, residency, gpu, "at rest: ");
}

// Morph, then skin: the skin reads the morph's output, never its own, so a pose alone runs the skin
// alone and the result is the morph skinned once
TEST(RetainedDeform, AMorphedSkinRunsOnlyThePassWhoseInputsChanged)
{
  Rig            rig(true, true);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  EXPECT_EQ(residency.deformCounts().morphs, 1u);
  EXPECT_EQ(residency.deformCounts().skins, 1u);
  expectClean(rig.scene, residency, gpu, "rest: ");

  rig.scene.setMorphWeights(rig.body, {1.f, 0.5f});
  gpu.beginFrame();
  residency.sync(rig.scene);
  EXPECT_EQ(residency.deformCounts().morphs, 1u);
  EXPECT_EQ(residency.deformCounts().skins, 1u);
  expectClean(rig.scene, residency, gpu, "weights: ");

  for(float angle : {20.f, -35.f, 60.f})
  {
    rig.bend(angle);
    gpu.beginFrame();
    residency.sync(rig.scene);
    EXPECT_EQ(residency.deformCounts().morphs, 0u);
    EXPECT_EQ(residency.deformCounts().skins, 1u);
    expectClean(rig.scene, residency, gpu, "pose " + std::to_string(angle) + ": ");
  }
  // the barrier between the passes, and the outputs fenced before the refits read them
  const std::vector<GpuBarrier>& barriers = gpu.barriers();
  EXPECT_NE(std::find(barriers.begin(), barriers.end(), GpuBarrier::eComputeToCompute), barriers.end());
}

// A morph alone: the instance's weights, else the mesh's defaults
TEST(RetainedDeform, AMorphTakesTheInstancesWeightsElseTheMeshs)
{
  RetainedScene scene;
  MeshRecord    record = skinnedGrid(true);
  record.joints.clear();
  record.weights.clear();
  record.targetWeights = {0.25f, 1.f};
  const MeshId     mesh = scene.addMesh(record);
  const MaterialId m    = scene.addMaterial({});
  const EntityId   e    = scene.createEntity("morphed", {});
  MeshInstance     instance;
  instance.parts.push_back({mesh, {m, m}});
  scene.setMeshInstance(e, instance);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  expectClean(scene, residency, gpu, "defaults: ");
  EXPECT_EQ(readBack<glm::vec3>(gpu, residency.drawer(e, mesh)->positions)[0], glm::vec3(0, 0, 0.25f));
  scene.setMorphWeights(e, {0.5f});  // the second target's weight is then 0
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_EQ(residency.deformCounts().morphs, 1u);
  expectClean(scene, residency, gpu, "instance: ");
  EXPECT_EQ(readBack<glm::vec3>(gpu, residency.drawer(e, mesh)->positions)[2], glm::vec3(2, 0, 0.5f));
}

// A host edit (a dial) to a skinned mesh's vertices: its base streams take it and the skin runs again
TEST(RetainedDeform, AVertexEditToASkinnedMeshSkinsTheNewVertices)
{
  Rig rig(false);
  rig.bend(30);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  std::vector<glm::vec3> dialled = rig.scene.mesh(rig.mesh)->positions;
  for(glm::vec3& p : dialled)
    p.y *= 1.5f;
  rig.scene.setMeshVertices(rig.mesh, dialled, {});
  gpu.beginFrame();
  residency.sync(rig.scene);
  EXPECT_EQ(residency.deformCounts().skins, 1u);
  expectClean(rig.scene, residency, gpu);
}

// Two entities drawing one skinned mesh: one set of the mesh's streams, a drawer and prims each
TEST(RetainedDeform, EachEntityDrawingASkinnedMeshHasItsOwnDrawer)
{
  Rig            rig(false);
  const EntityId second = Rig::drawSkinned(rig.scene, rig.mesh, rig.skin, rig.material, rig.root);
  Transform      t;
  t.translation = {0, 0, 3};
  rig.scene.setLocal(second, t);
  rig.bend(25);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  expectClean(rig.scene, residency, gpu);
  EXPECT_EQ(residency.drawers().size(), 2u);
  EXPECT_EQ(residency.deformCounts().skins, 2u);
  const Residency::DrawerGpu *a = residency.drawer(rig.body, rig.mesh), *b = residency.drawer(second, rig.mesh);
  ASSERT_TRUE(a && b);
  EXPECT_NE(a->positions, b->positions);
  EXPECT_NE(a->prims[0], b->prims[0]);
  // the second sits elsewhere: the joints seen from it differ, and so do its outputs
  EXPECT_NE(readBack<glm::vec3>(gpu, a->positions), readBack<glm::vec3>(gpu, b->positions));
}

// DLSS motion: a new drawer has no previous positions; positions written in one sync are copied at
// the top of the next, before anything writes them again; once nothing writes them, the copy rests
TEST(RetainedDeform, PreviousPositionsAreWhatTheLastFrameDrew)
{
  Rig            rig(false);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  const Residency::DrawerGpu& d    = *residency.drawer(rig.body, rig.mesh);
  const uint32_t              prim = uint32_t(d.prims[0]);
  const auto                  row  = [&] {
    uint64_t v = 0;
    std::memcpy(&v, gpu.buffer(residency.prevPositionTable())->bytes.data() + prim * sizeof(uint64_t), sizeof(v));
    return v;
  };
  EXPECT_EQ(row(), 0u);  // no motion the frame it appears
  const std::vector<glm::vec3> rest = readBack<glm::vec3>(gpu, d.positions);

  rig.bend(40);
  gpu.beginFrame();
  residency.sync(rig.scene);
  expectClean(rig.scene, residency, gpu, "posed: ");
  EXPECT_EQ(row(), gpu.bufferAddress(d.prevPositions));
  EXPECT_EQ(readBack<glm::vec3>(gpu, d.prevPositions), rest);  // where the last frame drew it
  const std::vector<glm::vec3> posed = readBack<glm::vec3>(gpu, d.positions);
  EXPECT_NE(posed, rest);

  gpu.beginFrame();  // nothing moves: the copy catches up, so the frame shows no motion
  residency.sync(rig.scene);
  EXPECT_EQ(readBack<glm::vec3>(gpu, d.prevPositions), posed);
  expectClean(rig.scene, residency, gpu, "rest: ");
}

// An instance that moves keeps, for one sync, the world it was drawn with; a static mesh a host edit
// writes gets previous positions from the next sync
TEST(RetainedDeform, MovedInstancesAndEditedMeshesHaveMotionForOneFrame)
{
  RetainedScene scene;
  MeshRecord    record = skinnedGrid(false);
  record.joints.clear();
  record.weights.clear();
  const MeshId     mesh = scene.addMesh(record);
  const MaterialId m    = scene.addMaterial({});
  const EntityId   e    = scene.createEntity("static", {});
  MeshInstance     instance;
  instance.parts.push_back({mesh, {m, m}});
  scene.setMeshInstance(e, instance);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(scene);
  EXPECT_TRUE(residency.drawers().empty());
  const glm::mat4 before = scene.world(e);

  Transform t;
  t.translation = {3, 0, 0};
  scene.setLocal(e, t);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "moved: ");
  EXPECT_EQ(*residency.prevWorld(0), before);
  gpu.beginFrame();
  residency.sync(scene);
  expectClean(scene, residency, gpu, "rest: ");
  EXPECT_EQ(*residency.prevWorld(0), scene.world(e));

  std::vector<glm::vec3> edited = scene.mesh(mesh)->positions;
  edited[0].z                   = 2.f;
  scene.setMeshVertices(mesh, edited, {});
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_FALSE(residency.meshGpu(mesh)->prevValid);
  gpu.beginFrame();
  residency.sync(scene);
  EXPECT_TRUE(residency.meshGpu(mesh)->prevValid);
  EXPECT_EQ(readBack<glm::vec3>(gpu, residency.meshGpu(mesh)->prevPositions), edited);
  expectClean(scene, residency, gpu, "edited: ");
}

// Taking the skin away puts the instance back on the mesh's own prims; removing the entity frees
// its drawer; nothing leaks once the retires have run
TEST(RetainedDeform, ADrawerGoesWithTheSkinOrTheEntity)
{
  Rig            rig(true);
  FakeGpuBackend gpu;
  Residency      residency(gpu);
  residency.replace(rig.scene);
  ASSERT_EQ(residency.drawers().size(), 1u);

  // still morphed without the skin: the drawer comes back without the skin pass
  MeshInstance unskinned = *rig.scene.entity(rig.body)->mesh;
  unskinned.skin         = {};
  rig.scene.setMeshInstance(rig.body, unskinned);
  gpu.beginFrame();
  residency.sync(rig.scene);
  expectClean(rig.scene, residency, gpu, "unskinned: ");
  ASSERT_EQ(residency.drawers().size(), 1u);
  EXPECT_FALSE(residency.drawer(rig.body, rig.mesh)->skin);

  rig.scene.destroyEntity(rig.body);
  gpu.beginFrame();
  residency.sync(rig.scene);
  expectClean(rig.scene, residency, gpu, "removed: ");
  EXPECT_TRUE(residency.drawers().empty());
  EXPECT_EQ(residency.meshGpu(rig.mesh), nullptr);
  drain(gpu);
  for(const std::string& leak : residencyLeaks(residency, gpu))
    ADD_FAILURE() << leak;
}

// The Khronos samples with skins and morphs, their clips played frame by frame through the residency
TEST(RetainedDeform, SampleAssetsPlayThroughTheirDrawers)
{
  namespace fs = std::filesystem;
  const fs::path models = fs::path(GLTF_SAMPLE_ASSETS_PATH) / "Models";
  int            tested = 0;
  uint32_t       morphs = 0, skins = 0;
  for(const char* file : {"RiggedSimple/glTF/RiggedSimple.gltf", "RiggedFigure/glTF/RiggedFigure.gltf", "CesiumMan/glTF/CesiumMan.gltf",
                          "Fox/glTF/Fox.gltf", "SimpleSkin/glTF/SimpleSkin.gltf", "SimpleMorph/glTF/SimpleMorph.gltf",
                          "AnimatedMorphCube/glTF/AnimatedMorphCube.gltf", "RecursiveSkeletons/glTF/RecursiveSkeletons.gltf"})
  {
    std::error_code ec;
    if(!fs::exists(models / file, ec))
      continue;
    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        err, warn;
    ASSERT_TRUE(loader.LoadASCIIFromFile(&model, &err, &warn, (models / file).string())) << file << ": " << err;
    RetainedScene      scene;
    const ImportResult imported = importGltf(scene, model);
    FakeGpuBackend     gpu;
    Residency          residency(gpu);
    residency.replace(scene);
    expectClean(scene, residency, gpu, std::string(file) + " loaded: ");
    for(ClipId clip : imported.clips)
    {
      ClipPlayer player(scene, clip);
      for(int frame = 0; frame < 6; frame++)
      {
        player.advance(std::max(player.end() - player.start(), 0.1f) / 5.f);
        player.apply(scene);
        gpu.beginFrame();
        residency.sync(scene);
        morphs += residency.deformCounts().morphs;
        skins += residency.deformCounts().skins;
        expectClean(scene, residency, gpu, std::string(file) + " frame " + std::to_string(frame) + ": ");
        if(::testing::Test::HasFailure())
          return;
      }
    }
    residency.clear();
    drain(gpu);
    for(const std::string& leak : residencyLeaks(residency, gpu))
      ADD_FAILURE() << file << ": " << leak;
    tested++;
  }
  if(tested == 0)
    GTEST_SKIP() << "no sample assets";
  EXPECT_GT(skins, 0u);
  EXPECT_GT(morphs, 0u);
  std::printf("played %d files through their drawers: %u morph and %u skin passes\n", tested, morphs, skins);
}
