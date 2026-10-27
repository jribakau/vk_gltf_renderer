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

// The scene-wide tables both scenes derive (src/scene_gpu_tables.hpp): lights as the shaders read
// them, and the emissive triangles' alias table, which must sample exactly the weights it was built
// from.

#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>

#include "scene_gpu_tables.hpp"

using nvvkgltf::buildEmissiveTable;
using nvvkgltf::Emitter;
using nvvkgltf::packLight;

TEST(SceneGpuTables, ALightTakesItsNodesPlaceAndItsOwnType)
{
  tinygltf::Light spot;
  spot.type                = "spot";
  spot.intensity           = 5.0;
  spot.color               = {1.0, 0.5, 0.25};
  spot.range               = 4.0;
  spot.spot.outerConeAngle = 0.5;
  const glm::mat4 world    = glm::translate(glm::mat4(1.f), {1.f, 2.f, 3.f});
  const auto      l        = packLight(&spot, world);
  EXPECT_EQ(l.type, shaderio::eLightTypeSpot);
  EXPECT_EQ(l.position, glm::vec3(1.f, 2.f, 3.f));
  EXPECT_EQ(l.direction, glm::vec3(0.f, 0.f, -1.f));  // down -Z, as glTF lights point
  EXPECT_EQ(l.color, glm::vec3(1.f, 0.5f, 0.25f));
  EXPECT_FLOAT_EQ(l.angularSizeOrInvRange, 0.25f);
  EXPECT_FLOAT_EQ(l.outerAngle, 0.5f);

  tinygltf::Light sun;
  sun.type = "directional";
  EXPECT_EQ(packLight(&sun, glm::mat4(1.f)).type, shaderio::eLightTypeDirectional);
  EXPECT_EQ(packLight(nullptr, glm::mat4(1.f)).type, shaderio::eLightTypePoint);  // a pure IES light
  EXPECT_EQ(l.iesProfile, -1);
}

// EXT_lights_ies: a point light of the node's colour and multiplier, naming its profile
TEST(SceneGpuTables, AnIesLightIsItsNodesTintedPointLight)
{
  const glm::mat4 world = glm::translate(glm::mat4(1.f), {0.f, 3.f, 0.f});
  const auto      l     = nvvkgltf::packIesLight(world, 2, 4.f, {1.f, 0.5f, 0.f});
  EXPECT_EQ(l.type, shaderio::eLightTypePoint);
  EXPECT_EQ(l.iesProfile, 2);
  EXPECT_FLOAT_EQ(l.intensity, 4.f);
  EXPECT_EQ(l.color, glm::vec3(1.f, 0.5f, 0.f));
  EXPECT_EQ(l.position, glm::vec3(0.f, 3.f, 0.f));
}

// Two quads of different size and brightness, a degenerate triangle, and an emitter that is black
TEST(SceneGpuTables, TheAliasTableSamplesTheWeightsItWasBuiltFrom)
{
  const std::vector<glm::vec3> quad     = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const std::vector<uint32_t>  two      = {0, 1, 2, 0, 2, 3};
  const std::vector<uint32_t>  flat     = {0, 1, 1};  // zero area
  const glm::mat4              big      = glm::scale(glm::mat4(1.f), {3.f, 3.f, 1.f});
  std::vector<Emitter>         emitters = {
      {.renderNodeID = 0, .renderPrimID = 0, .radiance = glm::vec3(1.f), .world = glm::mat4(1.f), .positions = quad, .indices = two},
      {.renderNodeID = 1, .renderPrimID = 1, .radiance = glm::vec3(4.f), .doubleSided = true, .world = big, .positions = quad, .indices = two},
      {.renderNodeID = 2, .renderPrimID = 2, .radiance = glm::vec3(1.f), .positions = quad, .indices = flat},
      {.renderNodeID = 3, .renderPrimID = 3, .radiance = glm::vec3(0.f), .positions = quad, .indices = two},
  };
  const auto table = buildEmissiveTable(emitters);
  ASSERT_EQ(table.triangles.size(), 4u);  // two quads, two triangles each
  EXPECT_EQ(table.triangles[2].renderNodeID, 1u);
  EXPECT_EQ(table.triangles[2].doubleSided, 1u);
  // mean luminance by area: (1 * 1 + 4 * 9) / 10
  EXPECT_NEAR(table.meanLum, 3.7f, 1e-5f);

  // what the alias table picks each triangle with: its own bin's keep probability plus what the
  // others alias to it, over n bins
  const size_t        n = table.triangles.size();
  std::vector<double> picked(n, 0.0);
  for(size_t i = 0; i < n; i++)
  {
    picked[i] += table.triangles[i].aliasProb / double(n);
    picked[table.triangles[i].aliasIndex] += (1.0 - table.triangles[i].aliasProb) / double(n);
  }
  const float f     = shaderio::EMISSIVE_DEFENSIVE_FRACTION;
  double      total = 0.0;
  for(size_t i = 0; i < n; i++)
  {
    const float area   = i < 2 ? 0.5f : 4.5f;
    const float lum    = nvvkgltf::emissiveLuminance(table.triangles[i].radiance);
    const float weight = area * ((1.f - f) * lum + f * table.meanLum);
    total += weight;
  }
  EXPECT_NEAR(table.totalWeight, total, 1e-3);
  for(size_t i = 0; i < n; i++)
  {
    const float area = i < 2 ? 0.5f : 4.5f;
    const float lum  = nvvkgltf::emissiveLuminance(table.triangles[i].radiance);
    EXPECT_NEAR(picked[i], area * ((1.f - f) * lum + f * table.meanLum) / total, 1e-5) << "triangle " << i;
  }
}

TEST(SceneGpuTables, NoEmittersNoTable)
{
  const auto table = buildEmissiveTable({});
  EXPECT_TRUE(table.triangles.empty());
  EXPECT_EQ(table.totalWeight, 0.f);
}
