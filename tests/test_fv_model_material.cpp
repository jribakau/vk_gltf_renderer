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

// A model prop's material with the Material panel's edits over it: what the renderer draws and what
// an export of the model writes come from this one function.

#include <gtest/gtest.h>

#include "fv_model_material.hpp"

namespace {

double ext(const tinygltf::Material& m, const char* name, const char* key)
{
  return m.extensions.at(name).Get(key).GetNumberAsDouble();
}

}  // namespace

TEST(FvModelMaterial, EditsLandOnTheFilesMaterial)
{
  tinygltf::Material original;
  original.name                                 = "paint";
  original.pbrMetallicRoughness.baseColorFactor = {0.5, 0.5, 0.5, 1.0};
  original.normalTexture.index                  = 2;
  original.pbrMetallicRoughness.baseColorTexture.index = 3;

  fv::mat::Edits e;
  e.color             = fv::mat::Rgb{1.f, 0.f, 0.f};
  e.opacity           = 0.25f;
  e.roughness         = 0.4f;
  e.metalness         = 1.f;
  e.alphaMode         = "cutout";
  e.doubleSided       = true;
  e.emissive          = fv::mat::Rgb{0.f, 1.f, 0.f};
  e.emissiveIntensity = 3.f;
  e.specular          = 0.7f;
  e.clearcoat         = 0.5f;
  e.normalStrength    = 2.f;
  e.sheenColor        = fv::mat::Rgb{0.2f, 0.3f, 0.4f};
  e.sheenRoughness    = 0.6f;
  const tinygltf::Material m = fv::model::editedMaterial(original, e);

  EXPECT_EQ(m.name, "paint");
  EXPECT_EQ(m.pbrMetallicRoughness.baseColorFactor, (std::vector<double>{1.0, 0.0, 0.0, 0.25}));
  EXPECT_DOUBLE_EQ(m.pbrMetallicRoughness.roughnessFactor, double(0.4f));
  EXPECT_DOUBLE_EQ(m.pbrMetallicRoughness.metallicFactor, 1.0);
  EXPECT_EQ(m.alphaMode, "MASK");
  EXPECT_TRUE(m.doubleSided);
  EXPECT_EQ(m.emissiveFactor, (std::vector<double>{0.0, 1.0, 0.0}));
  EXPECT_DOUBLE_EQ(ext(m, "KHR_materials_emissive_strength", "emissiveStrength"), 3.0);
  EXPECT_DOUBLE_EQ(ext(m, "KHR_materials_specular", "specularFactor"), double(0.7f));
  EXPECT_DOUBLE_EQ(ext(m, "KHR_materials_clearcoat", "clearcoatFactor"), 0.5);
  EXPECT_DOUBLE_EQ(m.normalTexture.scale, 2.0);
  // the sheen an export used to leave out
  const tinygltf::Value& sheen = m.extensions.at("KHR_materials_sheen").Get("sheenColorFactor");
  ASSERT_EQ(sheen.ArrayLen(), 3u);
  EXPECT_DOUBLE_EQ(sheen.Get(2).GetNumberAsDouble(), double(0.4f));
  EXPECT_DOUBLE_EQ(ext(m, "KHR_materials_sheen", "sheenRoughnessFactor"), double(0.6f));
  // textures stay the file's
  EXPECT_EQ(m.pbrMetallicRoughness.baseColorTexture.index, 3);
  EXPECT_EQ(m.normalTexture.index, 2);
}

TEST(FvModelMaterial, NoEditsIsTheFilesMaterialAndASheenRoughnessAloneAddsNoSheen)
{
  tinygltf::Material original;
  original.pbrMetallicRoughness.baseColorFactor = {0.1, 0.2, 0.3, 1.0};
  original.alphaMode                            = "BLEND";
  EXPECT_EQ(fv::model::editedMaterial(original, {}), original);

  fv::mat::Edits e;
  e.sheenRoughness = 0.3f;
  e.normalStrength = 2.f;  // and no normal map to scale
  const tinygltf::Material m = fv::model::editedMaterial(original, e);
  EXPECT_FALSE(m.extensions.count("KHR_materials_sheen"));
  EXPECT_DOUBLE_EQ(m.normalTexture.scale, 1.0);
}
