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

// The host's addressing on its retained scene (oracle 8 of daz-viewer docs/retained-scene-plan.md):
// every entity and every pick maps back to its prop, topic node and surface, by the handles the
// importer made; a handle that went maps to nothing.

#include <gtest/gtest.h>

#include <cstring>

#include "fv_scene_map.hpp"
#include "scene_import_gltf.hpp"
#include "scene_retained.hpp"

namespace {

// A file of two nodes: "seat" (mesh 0, material 0) under "chair", and an unused material 1
tinygltf::Model chairFile()
{
  tinygltf::Model m;
  const float     positions[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  m.buffers.emplace_back().data.resize(sizeof(positions));
  std::memcpy(m.buffers[0].data.data(), positions, sizeof(positions));
  tinygltf::BufferView& v = m.bufferViews.emplace_back();
  v.buffer                = 0;
  v.byteLength            = sizeof(positions);
  tinygltf::Accessor& a   = m.accessors.emplace_back();
  a.bufferView            = 0;
  a.type                  = TINYGLTF_TYPE_VEC3;
  a.componentType         = TINYGLTF_COMPONENT_TYPE_FLOAT;
  a.count                 = 3;
  a.minValues             = {0, 0, 0};
  a.maxValues             = {1, 1, 0};
  m.materials.resize(2);
  tinygltf::Primitive p;
  p.attributes["POSITION"]           = 0;
  p.material                         = 0;
  m.meshes.emplace_back().primitives = {p};
  m.nodes.resize(2);
  m.nodes[0].name               = "chair";
  m.nodes[0].children           = {1};
  m.nodes[1].name               = "seat";
  m.nodes[1].mesh               = 0;
  m.scenes.emplace_back().nodes = {0};
  m.defaultScene                = 0;
  return m;
}

struct Prop : fv::PropAddress
{
  std::string id;
};

// A prop root with the chair under it, as the host composes a model prop
Prop chairProp(rs::RetainedScene& scene, uint32_t number, rs::EntityId parent = {})
{
  Prop p;
  p.id                     = "p:" + std::to_string(number);
  p.root                   = scene.createEntity(p.id, {number, -1}, parent);
  const rs::ImportResult r = rs::importGltf(scene, chairFile(), {.prop = number, .parent = p.root});
  p.entityOf               = r.entities;
  p.materials              = r.materials;
  return p;
}

}  // namespace

TEST(FvSceneMap, EveryEntityAndPickMapsBackToItsPropNodeAndSurface)
{
  rs::RetainedScene scene;
  std::vector<Prop> props;
  props.push_back(chairProp(scene, 1));
  props.push_back(chairProp(scene, 2, props[0].root));  // rides on the first: its ranges are not the first's
  fv::SceneMap map;
  map.set(props);

  for(int prop = 0; prop < 2; prop++)
  {
    const Prop& p = props[size_t(prop)];
    EXPECT_EQ(map.propOfRoot(p.root), prop);
    EXPECT_EQ(map.locate(p.root).node, -1);
    EXPECT_EQ(map.propOfRoot(p.entityOf[0]), -1);
    for(int node = 0; node < 2; node++)
    {
      const fv::HostAddress at = map.locate(p.entityOf[size_t(node)]);
      EXPECT_EQ(at.prop, prop);
      EXPECT_EQ(at.node, node);
    }
    // what the scene draws: the seat with material 0
    ASSERT_FALSE(p.materials[0].isNone());
    EXPECT_TRUE(p.materials[1].isNone());  // nothing draws it
    const fv::HostAddress hit = map.pick(p.entityOf[1], p.materials[0]);
    EXPECT_EQ(hit.prop, prop);
    EXPECT_EQ(hit.node, 1);
    EXPECT_EQ(hit.surface, 0);
  }
  // every instance of the scene picks back to a prop
  for(const rs::DrawItem& item : scene.drawItems())
    EXPECT_GE(map.pick(item.entity, item.material).surface, 0);
}

TEST(FvSceneMap, AnotherPropsMaterialOrAHandleThatWentMapsToNothing)
{
  rs::RetainedScene scene;
  std::vector<Prop> props{chairProp(scene, 1), chairProp(scene, 2)};
  fv::SceneMap      map;
  map.set(props);
  // the first prop's seat drawing the second's material: no surface of the first
  const fv::HostAddress cross = map.pick(props[0].entityOf[1], props[1].materials[0]);
  EXPECT_EQ(cross.prop, 0);
  EXPECT_EQ(cross.surface, -1);
  // the second prop goes; a new one takes its slots under new generations
  const rs::EntityId gone = props[1].entityOf[1];
  scene.destroyEntity(props[1].root);
  props.pop_back();
  props.push_back(chairProp(scene, 3));
  map.set(props);
  EXPECT_EQ(map.locate(gone).prop, -1);
  EXPECT_EQ(map.locate(props[1].entityOf[1]).prop, 1);
}
