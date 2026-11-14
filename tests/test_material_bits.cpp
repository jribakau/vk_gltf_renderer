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
  EXPECT_EQ(bits.instanceMask, INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER | INSTANCE_MASK_CAMERA | INSTANCE_MASK_SHADOWS);
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

TEST(MaterialBits, ACutoutIsSolidSoThePrepassTestsItAndTransmissionBlends)
{
  tinygltf::Material mask;
  mask.alphaMode = "MASK";
  // the prepass draws the solid lists and tests the cutoff: in the blend list its clear texels would be
  // shaded and written opaque
  EXPECT_TRUE(materialBits(mask).solid());
  EXPECT_FALSE(materialBits(mask).blend());
  // rays still stop to test it
  EXPECT_EQ(materialBits(mask).instanceFlags, 0u);
  tinygltf::Material sided = mask;
  sided.doubleSided        = true;
  EXPECT_TRUE(materialBits(sided).solidDoubleSided());
  EXPECT_FALSE(materialBits(sided).solid() || materialBits(sided).blend());

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
  EXPECT_EQ(materialBits(m).instanceMask, INSTANCE_MASK_OTHER | INSTANCE_MASK_CAMERA | INSTANCE_MASK_SHADOWS);
}

// An eye's tear film (a chamber behind it) is found by itself: the rays from the iris under it see
// nothing else
TEST(MaterialBits, ATearFilmHasAMaskBitOfItsOwn)
{
  fv::mat::Material film;
  film.transmission = fv::mat::Transmission{.factor = 1.0f};
  film.chamberIor   = 1.336f;
  film.walkThrough  = true;
  EXPECT_EQ(materialBits(film).instanceMask, INSTANCE_MASK_OTHER | INSTANCE_MASK_CAMERA | INSTANCE_MASK_SHADOWS | INSTANCE_MASK_TEAR_FILM);
  // not what lies in the chamber, nor a pane of glass
  fv::mat::Material iris;
  iris.inChamberIor = 1.336f;
  EXPECT_EQ(materialBits(iris).instanceMask & INSTANCE_MASK_TEAR_FILM, 0);
  fv::mat::Material glass;
  glass.transmission = fv::mat::Transmission{.factor = 1.0f};
  EXPECT_EQ(materialBits(glass).instanceMask & INSTANCE_MASK_TEAR_FILM, 0);
}

// What the camera looks through is met by every ray but the camera's own: a bounce or shadow ray
// (0xFF) and a walk's (INSTANCE_MASK_WALK) see it as before
TEST(MaterialBits, OffCameraAnInstanceLosesTheCamerasBitAlone)
{
  const auto bits = materialBits(tinygltf::Material{});
  EXPECT_EQ(nvvkgltf::instanceMaskOf(bits, false), bits.instanceMask);
  const uint8_t off = nvvkgltf::instanceMaskOf(bits, true);
  EXPECT_EQ(off & INSTANCE_MASK_CAMERA, 0);
  EXPECT_EQ(off, INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER | INSTANCE_MASK_SHADOWS);
  EXPECT_NE(off & 0xFF, 0);
  // what no ray keeps stays out of every ray
  nvvkgltf::MaterialBits none;
  EXPECT_EQ(nvvkgltf::instanceMaskOf(none, true), 0);
}

// Shadow linking: a lamp whose shadows are its own traces them with one bit, which every instance
// has but those that cast it none. Every other ray (0xFF, the camera's, a walk's) meets them as before
TEST(MaterialBits, AnInstanceThatCastsALampNoShadowLosesThatLampsBitAlone)
{
  const auto bits = materialBits(tinygltf::Material{});
  EXPECT_EQ(bits.instanceMask & INSTANCE_MASK_SHADOWS, INSTANCE_MASK_SHADOWS);
  const uint8_t m = nvvkgltf::instanceMaskOf(bits, false, INSTANCE_MASK_SHADOW_1);
  EXPECT_EQ(m & INSTANCE_MASK_SHADOW_1, 0);
  EXPECT_EQ(m | INSTANCE_MASK_SHADOW_1, bits.instanceMask);
  EXPECT_NE(m & INSTANCE_MASK_SHADOW_0, 0);
  // only the shadow slots can be taken away so
  EXPECT_EQ(nvvkgltf::instanceMaskOf(bits, false, 0xFF), bits.instanceMask & ~INSTANCE_MASK_SHADOWS);
  // the slots are the bits the three masks name, one each
  EXPECT_EQ(INSTANCE_MASK_SHADOW_0 | INSTANCE_MASK_SHADOW_1 | INSTANCE_MASK_SHADOW_2, INSTANCE_MASK_SHADOWS);
  EXPECT_EQ(INSTANCE_MASK_SHADOW_0 << (INSTANCE_SHADOW_SLOTS - 1), INSTANCE_MASK_SHADOW_2);
  EXPECT_EQ(INSTANCE_MASK_SHADOWS & (INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER | INSTANCE_MASK_CAMERA | INSTANCE_MASK_TEAR_FILM), 0);
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
  EXPECT_EQ(materialBits(ghost).instanceMask, INSTANCE_MASK_WALK | INSTANCE_MASK_OTHER | INSTANCE_MASK_CAMERA | INSTANCE_MASK_SHADOWS);
}
