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

// The retained scene's store (daz-viewer docs/retained-scene-plan.md): generational handles, the
// entity graph and its world matrices, reference-counted records, and what changes it reports.

#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>

#include "scene_retained.hpp"

namespace {

struct ThingTag;

rs::MeshRecord triangle(float x = 0)
{
  rs::MeshRecord m;
  m.positions = {{x, 0, 0}, {x + 1, 0, 0}, {x, 1, 0}};
  m.normals   = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  m.indices   = {0, 1, 2};
  m.submeshes = {{.firstIndex = 0, .indexCount = 3, .material = 0}};
  return m;
}

bool near(const glm::mat4& a, const glm::mat4& b)
{
  for(int c = 0; c < 4; c++)
    for(int r = 0; r < 4; r++)
      if(std::abs(a[c][r] - b[c][r]) > 1e-5f)
        return false;
  return true;
}

}  // namespace

TEST(RetainedHandles, AFreedSlotComesBackUnderANewGeneration)
{
  rs::SlotMap<int, ThingTag> map;
  const auto a = map.insert(1);
  const auto b = map.insert(2);
  EXPECT_EQ(map.size(), 2u);
  EXPECT_TRUE(map.erase(a));
  EXPECT_FALSE(map.contains(a));
  EXPECT_EQ(map.get(a), nullptr);
  EXPECT_FALSE(map.erase(a));  // twice is nothing
  const auto c = map.insert(3);
  EXPECT_EQ(c.index, a.index);  // the slot is reused
  EXPECT_NE(c.gen, a.gen);      // under another generation
  EXPECT_EQ(map[c], 3);
  EXPECT_FALSE(map.contains(a));  // the old handle still names nothing
  EXPECT_EQ(map[b], 2);
  EXPECT_TRUE(rs::Handle<ThingTag>{}.isNone());
  int sum = 0;
  map.forEach([&](auto, int v) { sum += v; });
  EXPECT_EQ(sum, 5);
}

TEST(RetainedScene, WhatIsGoneIsWithinNothing)
{
  rs::RetainedScene scene;
  const rs::OwnerTag owner{1, 0};
  const rs::EntityId root  = scene.createEntity("root", owner);
  const rs::EntityId child = scene.createEntity("child", owner, root);
  EXPECT_TRUE(scene.isWithin(child, root));
  EXPECT_TRUE(scene.isWithin(root, root));
  EXPECT_FALSE(scene.isWithin(root, child));
  // as a residency instance can name it between a compose and the next sync
  scene.destroyEntity(child);
  EXPECT_FALSE(scene.isWithin(child, root));
  EXPECT_FALSE(scene.isWithin(rs::EntityId{}, root));
}

TEST(RetainedScene, WorldMatricesFollowTheHierarchy)
{
  rs::RetainedScene scene;
  const rs::EntityId root  = scene.createEntity("root", {1, 0});
  const rs::EntityId child = scene.createEntity("child", {1, 1}, root);
  scene.setLocal(root, {.translation = {1, 0, 0}});
  scene.setLocal(child, {.translation = {0, 2, 0}, .scale = {2, 2, 2}});
  const glm::mat4 expected = glm::translate(glm::mat4(1.f), {1, 0, 0}) * glm::translate(glm::mat4(1.f), {0, 2, 0})
                             * glm::scale(glm::mat4(1.f), {2, 2, 2});
  EXPECT_TRUE(near(scene.world(child), expected));
  // a matrix wins over TRS, as glTF has it
  scene.setLocal(root, {.matrix = glm::translate(glm::mat4(1.f), {5, 0, 0})});
  EXPECT_NEAR(scene.world(child)[3].x, 5.f, 1e-6f);

  // reparented keeping its place in the world
  const rs::EntityId other = scene.createEntity("other", {2, 0});
  scene.setLocal(other, {.translation = {0, 0, 7}});
  const glm::mat4 before = scene.world(child);
  ASSERT_TRUE(scene.setParent(child, other, -1, true));
  EXPECT_TRUE(near(scene.world(child), before));
  EXPECT_EQ(scene.entity(child)->parent, other);
  EXPECT_TRUE(scene.entity(root)->children.empty());
  // a cycle is refused
  EXPECT_FALSE(scene.setParent(other, child));
  EXPECT_FALSE(scene.setParent(other, other));
}

TEST(RetainedScene, VisibilityCascadesAndOrderIsDepthFirst)
{
  rs::RetainedScene scene;
  const rs::EntityId a  = scene.createEntity("a", {});
  const rs::EntityId b  = scene.createEntity("b", {});
  const rs::EntityId a1 = scene.createEntity("a1", {}, a);
  const rs::EntityId a0 = scene.createEntity("a0", {}, a, 0);
  EXPECT_EQ(scene.entitiesInOrder(), (std::vector<rs::EntityId>{a, a0, a1, b}));
  scene.setVisible(a, false);
  EXPECT_FALSE(scene.shown(a1));
  EXPECT_TRUE(scene.shown(b));
  scene.setVisible(a, true);
  scene.setVisible(a1, false);
  EXPECT_TRUE(scene.shown(a0));
  EXPECT_FALSE(scene.shown(a1));
}

TEST(RetainedScene, RecordsGoWithTheirLastReference)
{
  rs::RetainedScene    scene;
  const rs::TextureId  tex   = scene.addTexture({.name = "albedo", .uri = "/tex/albedo.png"});
  auto                 table = std::make_shared<rs::TextureTable>(rs::TextureTable{{tex, {}}});
  rs::MaterialRecord   mat;
  mat.payload.baseColorMap.index = 0;
  mat.textures                                            = table;
  const rs::MaterialId material = scene.addMaterial(mat);
  const rs::MeshId     mesh     = scene.addMesh(triangle());
  EXPECT_EQ(scene.refs(tex), 2u);  // the caller's and the material's

  const rs::EntityId e1 = scene.createEntity("e1", {});
  const rs::EntityId e2 = scene.createEntity("e2", {});
  scene.setMeshInstance(e1, rs::MeshInstance{.parts = {{mesh, {material}}}});
  scene.setMeshInstance(e2, rs::MeshInstance{.parts = {{mesh, {material}}}});
  // the creator lets go: the entities hold them now
  scene.release(mesh);
  scene.release(material);
  scene.release(tex);
  EXPECT_EQ(scene.refs(mesh), 2u);
  EXPECT_EQ(scene.refs(tex), 1u);

  scene.destroyEntity(e1);
  EXPECT_NE(scene.mesh(mesh), nullptr);
  scene.setMeshInstance(e2, std::nullopt);
  EXPECT_EQ(scene.mesh(mesh), nullptr);
  EXPECT_EQ(scene.material(material), nullptr);
  EXPECT_EQ(scene.texture(tex), nullptr);  // with the material that sampled it
  EXPECT_EQ(scene.meshCount() + scene.materialCount() + scene.textureCount(), 0u);
}

TEST(RetainedScene, DrawItemsComeInSourceOrderOncePerCopy)
{
  rs::RetainedScene scene;
  rs::MeshRecord    a = triangle(), b = triangle(5);
  // two vertex sets of one glTF mesh: its primitives 0 and 2 over one, 1 over the other
  a.submeshes = {{0, 3, 0, 0}, {0, 3, 1, 2}};
  b.submeshes = {{0, 3, 0, 1}};
  const rs::MeshId     ma = scene.addMesh(a), mb = scene.addMesh(b);
  const rs::MaterialId m0 = scene.addMaterial({}), m1 = scene.addMaterial({}), m2 = scene.addMaterial({});
  const rs::EntityId   e  = scene.createEntity("e", {});
  scene.setLocal(e, {.translation = {0, 1, 0}});
  scene.setMeshInstance(e, rs::MeshInstance{.parts = {{ma, {m0, m2}}, {mb, {m1}}},
                                            .copies = {glm::mat4(1.f), glm::translate(glm::mat4(1.f), {3, 0, 0})}});
  const std::vector<rs::DrawItem> items = scene.drawItems();
  ASSERT_EQ(items.size(), 6u);
  const std::vector<rs::MaterialId> want = {m0, m0, m1, m1, m2, m2};
  for(size_t i = 0; i < items.size(); i++)
  {
    EXPECT_EQ(items[i].material, want[i]) << i;
    EXPECT_EQ(items[i].copy, i % 2) << i;
  }
  EXPECT_NEAR(items[1].world[3].x, 3.f, 1e-6f);
  EXPECT_NEAR(items[1].world[3].y, 1.f, 1e-6f);
  // bounds over what is drawn, copies included
  const auto bounds = scene.bounds();
  EXPECT_NEAR(bounds[0].x, 0.f, 1e-6f);
  EXPECT_NEAR(bounds[1].x, 9.f, 1e-6f);
}

// The store keeps one copy of an IES profile: a lamp model removed and added back (each add imports
// it again) names the profile it had, with nothing for the residency to do
TEST(RetainedScene, AnIesProfileThatComesAgainIsTheOneThere)
{
  rs::RetainedScene          scene;
  const rs::IesProfileRecord down{.name = "down", .bytes = {1, 2, 3}, .samples = std::vector<float>(181, 0.5f)};
  const uint32_t             first = scene.addIesProfile(down);
  (void)scene.takeChanges();
  EXPECT_EQ(scene.addIesProfile(down), first);
  EXPECT_FALSE(scene.takeChanges().iesProfiles);
  rs::IesProfileRecord other = down;
  other.samples[0]           = 1.f;
  EXPECT_NE(scene.addIesProfile(other), first);
  EXPECT_TRUE(scene.takeChanges().iesProfiles);
  EXPECT_EQ(scene.iesProfiles().size(), 2u);
}

TEST(RetainedScene, ChangesReportWhatTheResidencyHasToDo)
{
  rs::RetainedScene  scene;
  const rs::EntityId root  = scene.createEntity("root", {});
  const rs::EntityId child = scene.createEntity("child", {}, root);
  const rs::MeshId   mesh  = scene.addMesh(triangle());
  scene.setMeshInstance(child, rs::MeshInstance{.parts = {{mesh, {scene.addMaterial({})}}}});
  rs::SceneChanges first = scene.takeChanges();
  EXPECT_TRUE(first.structure);
  EXPECT_TRUE(scene.takeChanges().empty());

  const uint64_t structure = scene.structureRevision();
  scene.setLocal(root, {.translation = {1, 0, 0}});
  rs::SceneChanges moved = scene.takeChanges();
  EXPECT_FALSE(moved.structure);
  EXPECT_EQ(moved.transforms, (std::set<rs::EntityId>{root, child}));  // the child moves with it
  EXPECT_EQ(scene.structureRevision(), structure);

  scene.setMeshVertices(mesh, {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}}, {});
  scene.setVisible(root, false);
  rs::SceneChanges edited = scene.takeChanges();
  EXPECT_EQ(edited.meshes, (std::set<rs::MeshId>{mesh}));
  EXPECT_EQ(edited.visibility.size(), 2u);
  EXPECT_EQ(scene.mesh(mesh)->contentRev, 1u);
  // a vertex edit of the wrong size is refused
  scene.setMeshVertices(mesh, {{0, 0, 0}}, {});
  EXPECT_TRUE(scene.takeChanges().meshes.empty());

  scene.destroyEntity(root);
  rs::SceneChanges gone = scene.takeChanges();
  EXPECT_TRUE(gone.structure);
  EXPECT_TRUE(gone.transforms.empty());  // no handle to an entity that is gone
  EXPECT_EQ(scene.entityCount(), 0u);
}
