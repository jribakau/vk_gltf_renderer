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

// What a material decides for what draws with it (src/scene_material_bits.hpp): TLAS flags and
// mask, and the raster lists.

#include <gtest/gtest.h>

#include "scene_material_bits.hpp"

using nvvkgltf::materialBits;

namespace {

tinygltf::Material transmissive(double factor)
{
  tinygltf::Material      m;
  tinygltf::Value::Object ext;
  ext["transmissionFactor"]                  = tinygltf::Value(factor);
  m.extensions["KHR_materials_transmission"] = tinygltf::Value(ext);
  return m;
}

}  // namespace

TEST(MaterialBits, AnOpaqueSurfaceIsSolidAndSkipsAnyHit)
{
  const auto bits = materialBits(tinygltf::Material{});
  EXPECT_EQ(bits.instanceFlags, nvvkgltf::kInstanceForceOpaque);
  EXPECT_EQ(bits.instanceMask, INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER);
  EXPECT_TRUE(bits.solid());
  EXPECT_FALSE(bits.solidDoubleSided() || bits.blend());
}

TEST(MaterialBits, DoubleSidedTurnsCullingOff)
{
  tinygltf::Material m;
  m.doubleSided   = true;
  const auto bits = materialBits(m);
  EXPECT_EQ(bits.instanceFlags, nvvkgltf::kInstanceForceOpaque | nvvkgltf::kInstanceCullDisable);
  EXPECT_TRUE(bits.solidDoubleSided());
  EXPECT_FALSE(bits.solid() || bits.blend());
}

TEST(MaterialBits, CutoutsAndTransmissionBlend)
{
  tinygltf::Material mask;
  mask.alphaMode = "MASK";
  EXPECT_TRUE(materialBits(mask).blend());
  EXPECT_EQ(materialBits(mask).instanceFlags, 0u);

  const auto glass = materialBits(transmissive(1.0));
  EXPECT_TRUE(glass.blend() && glass.transmission);
  EXPECT_EQ(glass.instanceFlags, nvvkgltf::kInstanceCullDisable);
  EXPECT_FALSE(materialBits(transmissive(0.0)).transmission);
}

TEST(MaterialBits, AWalkThroughSurfaceLeavesTheWalkMask)
{
  tinygltf::Material      m;
  tinygltf::Value::Object extras;
  extras["fvWalkThrough"] = tinygltf::Value(true);
  m.extras                = tinygltf::Value(extras);
  EXPECT_EQ(materialBits(m).instanceMask, INSTANCE_MASK_OTHER);
}

TEST(MaterialBits, ASurfaceNoRayKeepsLeavesEveryRay)
{
  const auto transparent = [](const char* mode, double alpha) {
    tinygltf::Material m;
    m.alphaMode                               = mode;
    m.pbrMetallicRoughness.baseColorFactor[3] = alpha;
    return m;
  };
  EXPECT_EQ(materialBits(transparent("BLEND", 0.0)).instanceMask, 0);  // a hair's face dummy
  EXPECT_EQ(materialBits(transparent("MASK", 0.2)).instanceMask, 0);   // under the 0.5 cutoff
  EXPECT_NE(materialBits(transparent("MASK", 0.5)).instanceMask, 0);
  tinygltf::Material atCutoff = transparent("MASK", 0.3);  // equal once both are floats, so kept
  atCutoff.alphaCutoff        = 0.30000001192092896;
  EXPECT_NE(materialBits(atCutoff).instanceMask, 0);
  EXPECT_NE(materialBits(transparent("BLEND", 0.01)).instanceMask, 0);
  EXPECT_NE(materialBits(transparent("OPAQUE", 0.0)).instanceMask, 0);  // OPAQUE ignores alpha

  tinygltf::Material ghost = transparent("BLEND", 0.0);  // a ghost light: emits, never seen
  ghost.emissiveFactor     = {1.0, 1.0, 1.0};
  EXPECT_EQ(materialBits(ghost).instanceMask, INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER);
}
