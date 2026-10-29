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

// The emitters' light tree: as buildEmissiveTable lays it out (src/scene_gpu_tables.cpp), and its walks
// (shaders/light_tree.h.slang compiled as C++). A light sample's walk down from the cut must reach each
// leaf with the probability a BSDF hit's climb back up gives it, or MIS would weigh the two techniques
// wrongly and the image would be off.

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>

#include "scene_gpu_tables.hpp"

using namespace glm;
using namespace shaderio;
#include "../shaders/light_pick.h.slang"
#include "../shaders/light_tree.h.slang"

namespace {

using nvvkgltf::buildEmissiveTable;
using nvvkgltf::Emitter;

const std::vector<glm::vec3> kQuad  = {{-0.5f, -0.5f, 0}, {0.5f, -0.5f, 0}, {0.5f, 0.5f, 0}, {-0.5f, 0.5f, 0}};
const std::vector<uint32_t>  kTwo   = {0, 1, 2, 0, 2, 3};
constexpr int                kCount = 100;

// `count` quads of random size, brightness and place in a 20 m room, render nodes 3 apart
std::vector<Emitter> roomOfPanels(int count, std::mt19937& rng)
{
  std::uniform_real_distribution<float> place(-10.f, 10.f), size(0.1f, 1.5f), bright(0.5f, 20.f);
  std::vector<Emitter>                  emitters;
  for(int i = 0; i < count; i++)
  {
    const glm::mat4 world = glm::scale(glm::translate(glm::mat4(1.f), {place(rng), place(rng) * 0.2f + 2.f, place(rng)}), glm::vec3(size(rng)));
    emitters.push_back({.renderNodeID = uint32_t(3 * i + 1), .renderPrimID = uint32_t(i), .radiance = glm::vec3(bright(rng)),
                        .world = world, .positions = kQuad, .indices = kTwo});
  }
  return emitters;
}

}  // namespace

// The cut holds at most EMISSIVE_GROUP_MAX nodes and comes first; every leaf lies below exactly one of
// them; a node's children name it as their parent; a node weighs what its children do and bounds every
// corner of the emitters below it; the render-node lookup names each emitter's leaf
TEST(LightTree, IsLaidOutCutFirstOverEveryEmitter)
{
  std::mt19937               rng(7);
  const std::vector<Emitter> emitters = roomOfPanels(kCount, rng);
  const auto                 table    = buildEmissiveTable(emitters, EMISSIVE_GROUP_MAX);
  ASSERT_EQ(table.cutCount, uint32_t(EMISSIVE_GROUP_MAX));
  // a binary tree over the leaves has 2n - 1 nodes, and the cut's 32 replace the 31 above them
  ASSERT_EQ(table.groups.size(), size_t(2 * kCount - EMISSIVE_GROUP_MAX));

  std::vector<int> leafCut(table.groups.size(), -1);
  int              leaves = 0;
  for(int i = 0; i < int(table.groups.size()); i++)
  {
    const EmissiveGroup& g = table.groups[i];
    EXPECT_EQ(g.parent < 0, i < int(table.cutCount)) << "node " << i;
    if(g.left < 0)
    {
      EXPECT_EQ(g.right, -1);
      EXPECT_EQ(g.count, 2u);
      leaves++;
      // the cut node above it, and every node on the way bounds its triangles
      int node = i;
      while(true)
      {
        for(uint32_t t = g.first; t < g.first + g.count; t++)
        {
          const EmissiveTriangle& tri     = table.triangles[t];
          const Emitter&          emitter = emitters[tri.renderPrimID];
          for(const glm::vec3& corner : kQuad)
            EXPECT_LE(glm::distance(glm::vec3(emitter.world * glm::vec4(corner, 1.f)), table.groups[node].center),
                      table.groups[node].radius * 1.0001f + 1e-5f);
        }
        if(table.groups[node].parent < 0)
          break;
        node = table.groups[node].parent;
      }
      leafCut[i] = node;
      ASSERT_LT(g.renderNodeID, table.leafOfNode.size());
      EXPECT_EQ(table.leafOfNode[g.renderNodeID], i);
    }
    else
    {
      ASSERT_GE(g.right, 0);
      EXPECT_EQ(table.groups[g.left].parent, i);
      EXPECT_EQ(table.groups[g.right].parent, i);
      EXPECT_GT(g.left, i);  // breadth first below the cut
      EXPECT_NEAR(g.weight, table.groups[g.left].weight + table.groups[g.right].weight, 1e-4f * g.weight);
    }
  }
  EXPECT_EQ(leaves, kCount);
  double cutWeight = 0.0;
  for(uint32_t c = 0; c < table.cutCount; c++)
    cutWeight += table.groups[c].weight;
  EXPECT_NEAR(cutWeight, table.totalWeight, 1e-4 * table.totalWeight);
  int mapped = 0;
  for(const int32_t leaf : table.leafOfNode)
    mapped += leaf >= 0 ? 1 : 0;
  EXPECT_EQ(mapped, kCount);
}

// From each cut node, at points about the room and with their normals' cull, the walk down reaches
// each leaf below as often as the climb back up says, it reports that probability itself, and the climb
// ends at that cut node; a leaf the walk cannot reach has no probability on the way up either
TEST(LightTree, WalkDownAndClimbUpAgree)
{
  std::mt19937                     rng(11);
  const std::vector<Emitter>       emitters = roomOfPanels(kCount, rng);
  auto                             table    = buildEmissiveTable(emitters, EMISSIVE_GROUP_MAX);
  EmissiveGroup*                   nodes    = table.groups.data();
  std::uniform_real_distribution<float> place(-10.f, 10.f), unit(-1.f, 1.f);
  constexpr int                    kWalks = 20000;
  for(int point = 0; point < 6; point++)
  {
    const float3 pos(place(rng), place(rng) * 0.2f + 1.f, place(rng));
    const float3 normal = normalize(float3(unit(rng), unit(rng), unit(rng)));
    const bool   cull   = point % 2 == 0;
    for(int c = 0; c < int(table.cutCount); c++)
    {
      std::vector<double> reached(table.groups.size(), 0.0);
      bool                anyReached = false;
      for(int w = 0; w < kWalks; w++)
      {
        const EmissiveTreeWalk walk = emissiveTreeDescend(nodes, c, pos, normal, cull, (float(w) + 0.5f) / kWalks);
        if(walk.leaf < 0)
          continue;
        anyReached = true;
        ASSERT_LT(nodes[walk.leaf].left, 0);
        reached[walk.leaf] += 1.0 / kWalks;
        const EmissiveTreeWalk climb = emissiveTreeClimb(nodes, walk.leaf, pos, normal, cull);
        EXPECT_EQ(climb.leaf, c);
        EXPECT_NEAR(climb.prob, walk.prob, 1e-5f * walk.prob);
      }
      double sum = 0.0;
      for(int l = 0; l < int(table.groups.size()); l++)
      {
        if(nodes[l].left >= 0)
          continue;
        const EmissiveTreeWalk climb = emissiveTreeClimb(nodes, l, pos, normal, cull);
        if(climb.leaf != c)
          continue;
        sum += climb.prob;
        EXPECT_NEAR(reached[l], climb.prob, 2.0 / kWalks + 1e-3 * climb.prob) << "point " << point << " cut " << c << " leaf " << l;
      }
      EXPECT_NEAR(sum, anyReached ? 1.0 : 0.0, 1e-4) << "point " << point << " cut " << c;
    }
  }
}

// By weight alone, the walk reaches each leaf by its share of the cut node's weight, and leaves the
// rest of its uniform for the leaf's triangle
TEST(LightTree, WalkByWeightTakesEachLeafByItsWeight)
{
  std::mt19937               rng(5);
  const std::vector<Emitter> emitters = roomOfPanels(kCount, rng);
  auto                       table    = buildEmissiveTable(emitters, 8);
  EmissiveGroup*             nodes    = table.groups.data();
  constexpr int              kWalks   = 40000;
  for(int c = 0; c < int(table.cutCount); c++)
  {
    std::vector<double> reached(table.groups.size(), 0.0);
    double              rest = 0.0;
    for(int w = 0; w < kWalks; w++)
    {
      const EmissiveTreeStep step = emissiveTreeDescendByWeight(nodes, c, (float(w) + 0.5f) / kWalks);
      ASSERT_LT(nodes[step.leaf].left, 0);
      ASSERT_GE(step.u, 0.f);
      ASSERT_LT(step.u, 1.f);
      reached[step.leaf] += 1.0 / kWalks;
      rest += step.u / kWalks;
    }
    for(int l = 0; l < int(table.groups.size()); l++)
      if(nodes[l].left < 0 && emissiveTreeClimb(nodes, l, float3(0), float3(0, 1, 0), false).leaf == c)
        EXPECT_NEAR(reached[l], nodes[l].weight / nodes[c].weight, 2.0 / kWalks + 1e-3) << "cut " << c << " leaf " << l;
    EXPECT_NEAR(rest, 0.5, 0.02) << "cut " << c;  // what is left is about uniform
  }
}
