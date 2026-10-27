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

// The host's addressing on the legacy scene (oracle 8 of daz-viewer docs/retained-scene-plan.md):
// a figure's uploads reach every render primitive holding its mesh's bytes, and every render node's
// pick maps back to its prop, topic node and surface.

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <set>

#include "fv_scene_map.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_merger.hpp"

namespace {

// A composed scene of two props:
//   node 0 "figure" (the figure's prop root) > node 1 "body", mesh 0: two material groups (materials
//          0 and 1) over one set of positions, as a figure's mesh is built;
//   node 2 "chair"  (the model's prop root)  > node 3 "seat", mesh 1, material 2.
// The figure's own glTF had "body" at node 0, so it was merged at node offset 1.
tinygltf::Model composed()
{
  tinygltf::Model m;
  const float     positions[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
  const uint32_t  first[]     = {0, 1, 2};
  const uint32_t  second[]    = {1, 3, 2};
  const float     seat[]      = {0, 0, 1, 1, 0, 1, 0, 1, 1};
  m.buffers.emplace_back().data.resize(sizeof(positions) + sizeof(first) + sizeof(second) + sizeof(seat));
  size_t offset = 0;
  auto   view   = [&](const void* data, size_t size) {
    std::memcpy(m.buffers[0].data.data() + offset, data, size);
    tinygltf::BufferView& v = m.bufferViews.emplace_back();
    v.buffer                = 0;
    v.byteOffset            = offset;
    v.byteLength            = size;
    offset += size;
    return int(m.bufferViews.size()) - 1;
  };
  auto accessor = [&](int bufferView, int type, int componentType, size_t count) {
    tinygltf::Accessor& a = m.accessors.emplace_back();
    a.bufferView          = bufferView;
    a.type                = type;
    a.componentType       = componentType;
    a.count               = count;
    if(type == TINYGLTF_TYPE_VEC3)
    {
      a.minValues = {0, 0, 0};
      a.maxValues = {1, 1, 1};
    }
    return int(m.accessors.size()) - 1;
  };
  const int pos  = accessor(view(positions, sizeof(positions)), TINYGLTF_TYPE_VEC3, TINYGLTF_COMPONENT_TYPE_FLOAT, 4);
  const int idx0 = accessor(view(first, sizeof(first)), TINYGLTF_TYPE_SCALAR, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, 3);
  const int idx1 = accessor(view(second, sizeof(second)), TINYGLTF_TYPE_SCALAR, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, 3);
  const int seatAt = accessor(view(seat, sizeof(seat)), TINYGLTF_TYPE_VEC3, TINYGLTF_COMPONENT_TYPE_FLOAT, 3);

  m.materials.resize(3);
  auto primitive = [](int position, int indices, int material) {
    tinygltf::Primitive p;
    p.attributes["POSITION"] = position;
    p.indices                = indices;
    p.material               = material;
    return p;
  };
  m.meshes.emplace_back().primitives = {primitive(pos, idx0, 0), primitive(pos, idx1, 1)};
  m.meshes.emplace_back().primitives = {primitive(seatAt, idx0, 2)};

  m.nodes.resize(4);
  m.nodes[0].name               = "figure";
  m.nodes[0].children           = {1};
  m.nodes[1].name               = "body";
  m.nodes[1].mesh               = 0;
  m.nodes[2].name               = "chair";
  m.nodes[2].children           = {3};
  m.nodes[3].name               = "seat";
  m.nodes[3].mesh               = 1;
  m.scenes.emplace_back().nodes = {0, 2};
  m.defaultScene                = 0;
  return m;
}

struct Prop : fv::PropAddress
{
  std::string id;
};

}  // namespace

TEST(FvSceneMap, AFiguresUploadsReachEveryRenderPrimitiveOfItsMesh)
{
  nvvkgltf::Scene scene;
  scene.takeModel(composed());
  ASSERT_TRUE(scene.valid());

  nvvkgltf::IndexRemapping base;
  base.nodes                 = 1;
  const fv::FigureSceneMap f = fv::mapFigure({0}, {0}, {-1}, base, scene);
  EXPECT_EQ(f.nodeOf, std::vector<int>{1});
  EXPECT_EQ(f.sceneOf.at(1), 0);
  EXPECT_EQ(f.meshOf, std::vector<int>{0});
  EXPECT_EQ(f.skinOf, std::vector<int>{-1});
  ASSERT_EQ(f.primsOf.size(), 1u);
  EXPECT_EQ(f.primsOf[0].size(), 1u);  // the two material groups read one set of vertices: one upload

  // what an upload to each listed owner reaches, against every render primitive of the mesh
  for(size_t m = 0; m < f.meshOf.size(); m++)
  {
    std::set<int> reached, holding;
    for(int owner : f.primsOf[m])
      for(int member : scene.vertexSetMembers(owner))
        reached.insert(member);
    const auto& prims = scene.getRenderPrimitives();
    for(int r = 0; r < int(prims.size()); r++)
      if(prims[size_t(r)].meshID == f.meshOf[m])
        holding.insert(r);
    EXPECT_EQ(reached, holding) << "mesh " << m;
    EXPECT_EQ(holding.size(), 2u);
  }
}

TEST(FvSceneMap, AMeshOutsideTheSceneHasNoPrimitives)
{
  nvvkgltf::Scene scene;
  scene.takeModel(composed());
  const fv::FigureSceneMap f = fv::mapFigure({0, -1}, {-1, 5}, {-1, -1}, {}, scene);
  EXPECT_EQ(f.nodeOf, (std::vector<int>{0, -1}));
  EXPECT_EQ(f.sceneOf.size(), 1u);
  ASSERT_EQ(f.primsOf.size(), 2u);
  EXPECT_TRUE(f.primsOf[0].empty());  // not built
  EXPECT_TRUE(f.primsOf[1].empty());  // no such glTF mesh
}

TEST(FvSceneMap, EveryPickMapsBackToItsPropNodeAndSurface)
{
  nvvkgltf::Scene scene;
  scene.takeModel(composed());
  ASSERT_TRUE(scene.valid());

  std::vector<Prop> props(2);
  props[0].id            = "figure";
  props[0].root          = 0;
  props[0].gltfOf        = {1};
  props[0].materials     = 0;
  props[0].materialCount = 2;
  props[1].id            = "chair";
  props[1].root          = 2;
  props[1].gltfOf        = {3};
  props[1].materials     = 2;
  props[1].materialCount = 1;
  fv::SceneMap map;
  map.set(props);

  // glTF node -> {prop, topic node}
  const std::map<int, std::pair<int, int>> expected = {{1, {0, 0}}, {3, {1, 0}}};
  const auto&                              nodes    = scene.getRenderNodes();
  ASSERT_EQ(nodes.size(), 3u);
  for(int r = 0; r < int(nodes.size()); r++)
  {
    const nvvkgltf::RenderNode& rn = nodes[size_t(r)];
    const fv::HostAddress       at = map.pick(scene, r);
    ASSERT_TRUE(expected.count(rn.refNodeID)) << "render node " << r;
    EXPECT_EQ(at.prop, expected.at(rn.refNodeID).first) << "render node " << r;
    EXPECT_EQ(at.node, expected.at(rn.refNodeID).second) << "render node " << r;
    EXPECT_EQ(at.surface, rn.materialID - props[size_t(at.prop)].materials) << "render node " << r;
  }

  EXPECT_EQ(map.locate(0).prop, 0);
  EXPECT_EQ(map.locate(0).node, -1);  // the prop root
  EXPECT_EQ(map.propOfRoot(2), 1);
  EXPECT_EQ(map.propOfRoot(3), -1);  // a node of the prop, not its root
  EXPECT_EQ(map.locate(99).prop, -1);
  EXPECT_EQ(map.pick(scene, -1).prop, -1);
  EXPECT_EQ(map.pick(scene, int(nodes.size())).prop, -1);
}

TEST(FvSceneMap, APickOnAMaterialNotThePropsHasNoSurface)
{
  nvvkgltf::Scene scene;
  scene.takeModel(composed());
  std::vector<Prop> props(1);
  props[0].root          = 0;
  props[0].gltfOf        = {1};
  props[0].materials     = 0;
  props[0].materialCount = 1;  // material 1 is not counted as the prop's
  fv::SceneMap map;
  map.set(props);
  for(int r = 0; r < int(scene.getRenderNodes().size()); r++)
  {
    const nvvkgltf::RenderNode& rn = scene.getRenderNodes()[size_t(r)];
    if(rn.refNodeID != 1)
      EXPECT_EQ(map.pick(scene, r).prop, -1);
    else
      EXPECT_EQ(map.pick(scene, r).surface, rn.materialID == 0 ? 0 : -1);
  }
}
