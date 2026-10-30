/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cstring>
#include <figure/materials/material_gltf.hpp>
#include "gltf_material_cache.hpp"
#include "gltf_image_loader.hpp"
#include "tinygltf_utils.hpp"

// M7d: MaterialCache tests — conversion, incremental updates, topology change detection

TEST(MaterialCache, BuildFromEmptyMaterials)
{
  nvvkgltf::MaterialCache         cache;
  std::vector<tinygltf::Material> materials;
  cache.buildFromMaterials(materials, {});

  EXPECT_TRUE(cache.getShadeMaterials().empty());
  EXPECT_EQ(cache.getTextureInfos().size(), 1u);  // Sentinel entry at index 0
}

TEST(MaterialCache, BuildFromSingleOpaqueMaterial)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.alphaMode                            = "OPAQUE";
  mat.doubleSided                          = false;
  mat.pbrMetallicRoughness.baseColorFactor = {1.0, 0.0, 0.0, 1.0};
  mat.pbrMetallicRoughness.metallicFactor  = 0.5;
  mat.pbrMetallicRoughness.roughnessFactor = 0.8;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials, {});

  ASSERT_EQ(cache.getShadeMaterials().size(), 1u);

  const auto& shade = cache.getShadeMaterials()[0];
  EXPECT_EQ(shade.alphaMode, 0);  // OPAQUE
  EXPECT_EQ(shade.doubleSided, 0);
  EXPECT_FLOAT_EQ(shade.pbrBaseColorFactor[0], 1.0f);
  EXPECT_FLOAT_EQ(shade.pbrBaseColorFactor[1], 0.0f);
  EXPECT_FLOAT_EQ(shade.pbrMetallicFactor, 0.5f);
  EXPECT_FLOAT_EQ(shade.pbrRoughnessFactor, 0.8f);
}

TEST(MaterialCache, BuildFromMaskAndBlendAlphaModes)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material maskMat;
  maskMat.alphaMode   = "MASK";
  maskMat.alphaCutoff = 0.3;

  tinygltf::Material blendMat;
  blendMat.alphaMode = "BLEND";

  std::vector<tinygltf::Material> materials = {maskMat, blendMat};
  cache.buildFromMaterials(materials, {});

  ASSERT_EQ(cache.getShadeMaterials().size(), 2u);
  EXPECT_EQ(cache.getShadeMaterials()[0].alphaMode, 1);  // MASK
  EXPECT_FLOAT_EQ(cache.getShadeMaterials()[0].alphaCutoff, 0.3f);
  EXPECT_EQ(cache.getShadeMaterials()[1].alphaMode, 2);  // BLEND
}

TEST(MaterialCache, BuildWithBaseColorTexture)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.pbrMetallicRoughness.baseColorTexture.index    = 0;
  mat.pbrMetallicRoughness.baseColorTexture.texCoord = 0;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials, {});

  ASSERT_EQ(cache.getShadeMaterials().size(), 1u);
  // Texture index > 0 means a texture info was added
  EXPECT_GT(cache.getShadeMaterials()[0].pbrBaseColorTexture, 0u);
  // Sentinel + at least one real texture info
  EXPECT_GE(cache.getTextureInfos().size(), 2u);
}

// One material converted on its own names its texture infos from its span's start, the same infos
// and row (but for the indices) as appending it to a cache
TEST(MaterialCache, ConvertOneNamesItsInfosFromItsBase)
{
  tinygltf::Material mat;
  mat.pbrMetallicRoughness.baseColorTexture.index = 0;
  mat.normalTexture.index                         = 1;
  mat.emissiveFactor                              = {1.0, 0.5, 0.0};

  std::vector<shaderio::GltfTextureInfo> infos;
  const shaderio::GltfShadeMaterial      row = nvvkgltf::MaterialCache::convertOne(mat, {}, 100, infos);
  ASSERT_EQ(infos.size(), 2u);
  EXPECT_EQ(row.normalTexture, 100u);  // normal before base colour, as populateShaderMaterial goes
  EXPECT_EQ(row.pbrBaseColorTexture, 101u);
  EXPECT_EQ(row.emissiveTexture, 0u);

  nvvkgltf::MaterialCache cache;
  cache.append(fv::mat::fromGltf(mat), {});
  ASSERT_EQ(cache.getTextureInfos().size(), 3u);
  for(size_t k = 0; k < 2; k++)
    EXPECT_EQ(std::memcmp(&cache.getTextureInfos()[1 + k], &infos[k], sizeof(infos[k])), 0) << k;
  EXPECT_EQ(cache.getShadeMaterials()[0].emissiveFactor, row.emissiveFactor);
}

// Every field of a material in its row, and each map in its own slot of the row
TEST(MaterialCache, EveryFieldOfAMaterialReachesItsRow)
{
  using namespace fv::mat;
  const auto slot = [](int index) {
    TextureSlot s;
    s.index = index;
    return s;
  };
  Material m;
  m.baseColor            = {0.1f, 0.2f, 0.3f, 0.4f};
  m.metallic             = 0.25f;
  m.roughness            = 0.75f;
  m.alphaMode            = AlphaMode::eMask;
  m.alphaCutoff          = 0.3f;
  m.doubleSided          = true;
  m.emissive             = {1, 0.5f, 0.25f};
  m.emissiveStrength     = 2;
  m.emissiveMap          = slot(1);
  m.normalMap            = slot(2);
  m.normalMap.scale      = 0.6f;
  m.baseColorMap         = slot(3);
  m.metallicRoughnessMap = slot(4);
  m.occlusionMap         = slot(5);
  m.occlusionMap.scale   = 0.7f;
  m.transmission         = Transmission{.factor = 0.9f, .map = slot(6)};
  m.ior                  = 1.33f;
  m.volume               = Volume{.thickness = 0.5f, .thicknessMap = slot(7), .attenuationDistance = 2, .attenuationColor = {0.3f, 0.2f, 0.1f}};
  m.clearcoat            = Clearcoat{.factor = 0.5f, .roughness = 0.2f, .map = slot(9), .roughnessMap = slot(8), .normalMap = slot(10)};
  m.clearcoat->normalMap.scale = 0.4f;
  m.specular2            = Specular2{.factor = 0.3f, .roughness = 0.15f, .map = slot(28), .roughnessMap = slot(29)};
  m.specular             = Specular{.factor = 0.8f, .color = {0.9f, 0.8f, 0.7f}, .map = slot(11), .colorMap = slot(12)};
  m.iridescence          = Iridescence{.factor = 0.5f, .ior = 1.4f, .thicknessMinimum = 150, .thicknessMaximum = 350, .map = slot(13), .thicknessMap = slot(14)};
  m.anisotropy           = Anisotropy{.strength = 0.5f, .rotation = 1.25f, .map = slot(15)};
  m.sheen                = Sheen{.color = {0.1f, 0.2f, 0.4f}, .roughness = 0.3f, .colorMap = slot(16), .roughnessMap = slot(17)};
  m.dispersion           = 0.5f;
  m.specularGlossiness   = SpecularGlossiness{.diffuse = {0.2f, 0.3f, 0.4f, 0.5f}, .specular = {0.6f, 0.7f, 0.8f}, .glossiness = 0.9f, .diffuseMap = slot(18), .specularGlossinessMap = slot(19)};
  m.diffuseTransmission  = DiffuseTransmission{.factor = 0.3f, .color = {0.4f, 0.5f, 0.6f}, .map = slot(20), .colorMap = slot(21)};
  m.retroreflection      = Retroreflection{.factor = 0.2f, .map = slot(22)};
  m.scatter              = Scatter{.strength = 0.6f, .strengthMap = slot(23), .multiscatterColor = {0.7f, 0.6f, 0.5f}, .multiscatterColorMap = slot(24), .anisotropy = -0.5f};
  m.scatterDistance      = Rgb{0.003f, 0.002f, 0.001f};
  m.detail               = Detail{.weight = 0.5f, .roughness = 0.6f, .normalMap = slot(25), .weightMap = slot(26), .roughnessMap = slot(27)};
  m.detail->normalMap.scale = 0.2f;
  m.hair                 = Hair{.longitudinal = 0.3f, .azimuthal = 0.4f, .rootAbsorption = {0.5f, 0.6f, 0.7f}, .tipAbsorption = {0.1f, 0.2f, 0.3f}, .rootFade = 0.06f, .rootV = 1, .specular = 0.5f, .strandRotation = 0.5f};
  m.dlssNrMask           = Rgba{0.1f, 0.2f, 0.3f, 0.4f};
  m.unlit                = true;

  std::vector<shaderio::GltfTextureInfo> infos;
  const shaderio::GltfShadeMaterial      row = nvvkgltf::MaterialCache::convertOne(m, {}, 0, infos);

  EXPECT_EQ(row.pbrBaseColorFactor, glm::vec4(0.1f, 0.2f, 0.3f, 0.4f));
  EXPECT_FLOAT_EQ(row.pbrMetallicFactor, 0.25f);
  EXPECT_FLOAT_EQ(row.pbrRoughnessFactor, 0.75f);
  EXPECT_EQ(row.alphaMode, 1);
  EXPECT_FLOAT_EQ(row.alphaCutoff, 0.3f);
  EXPECT_EQ(row.doubleSided, 1);
  EXPECT_EQ(row.emissiveFactor, glm::vec3(2, 1, 0.5f));
  EXPECT_FLOAT_EQ(row.normalTextureScale, 0.6f);
  EXPECT_FLOAT_EQ(row.occlusionStrength, 0.7f);
  EXPECT_FLOAT_EQ(row.transmissionFactor, 0.9f);
  EXPECT_FLOAT_EQ(row.ior, 1.33f);
  EXPECT_FLOAT_EQ(row.thicknessFactor, 0.5f);
  EXPECT_FLOAT_EQ(row.attenuationDistance, 2);
  EXPECT_EQ(row.attenuationColor, glm::vec3(0.3f, 0.2f, 0.1f));
  EXPECT_FLOAT_EQ(row.clearcoatFactor, 0.5f);
  EXPECT_FLOAT_EQ(row.clearcoatRoughness, 0.2f);
  EXPECT_FLOAT_EQ(row.clearcoatNormalTextureScale, 0.4f);
  EXPECT_FLOAT_EQ(row.specular2Factor, 0.3f);
  EXPECT_FLOAT_EQ(row.specular2Roughness, 0.15f);
  EXPECT_FLOAT_EQ(row.specularFactor, 0.8f);
  EXPECT_EQ(row.specularColorFactor, glm::vec3(0.9f, 0.8f, 0.7f));
  EXPECT_FLOAT_EQ(row.iridescenceFactor, 0.5f);
  EXPECT_FLOAT_EQ(row.iridescenceIor, 1.4f);
  EXPECT_FLOAT_EQ(row.iridescenceThicknessMinimum, 150);
  EXPECT_FLOAT_EQ(row.iridescenceThicknessMaximum, 350);
  EXPECT_FLOAT_EQ(row.anisotropyStrength, 0.5f);
  EXPECT_EQ(row.anisotropyRotation, glm::vec2(glm::sin(1.25f), glm::cos(1.25f)));
  EXPECT_EQ(row.sheenColorFactor, glm::vec3(0.1f, 0.2f, 0.4f));
  EXPECT_FLOAT_EQ(row.sheenRoughnessFactor, 0.3f);
  EXPECT_FLOAT_EQ(row.dispersion, 0.5f);
  EXPECT_EQ(row.pbrModel, shaderio::PbrModel::ePbrModelSpecularGlossiness);
  EXPECT_EQ(row.pbrDiffuseFactor, glm::vec4(0.2f, 0.3f, 0.4f, 0.5f));
  EXPECT_EQ(row.pbrSpecularFactor, glm::vec3(0.6f, 0.7f, 0.8f));
  EXPECT_FLOAT_EQ(row.pbrGlossinessFactor, 0.9f);
  EXPECT_FLOAT_EQ(row.diffuseTransmissionFactor, 0.3f);
  EXPECT_EQ(row.diffuseTransmissionColorFactor, glm::vec3(0.4f, 0.5f, 0.6f));
  EXPECT_FLOAT_EQ(row.retroreflectionFactor, 0.2f);
  EXPECT_FLOAT_EQ(row.scatterStrengthFactor, 0.6f);
  EXPECT_EQ(row.multiscatterColorFactor, glm::vec3(0.7f, 0.6f, 0.5f));
  EXPECT_FLOAT_EQ(row.scatterAnisotropy, -0.5f);
  EXPECT_EQ(row.scatterDistance, glm::vec3(0.003f, 0.002f, 0.001f));
  EXPECT_FLOAT_EQ(row.detailNormalScale, 0.2f);
  EXPECT_FLOAT_EQ(row.detailWeightFactor, 0.5f);
  EXPECT_FLOAT_EQ(row.detailRoughnessFactor, 0.6f);
  EXPECT_EQ(row.hairFactor, 1.0f);
  EXPECT_FLOAT_EQ(row.hairLongitudinal, 0.3f);
  EXPECT_FLOAT_EQ(row.hairAzimuthal, 0.4f);
  EXPECT_FLOAT_EQ(row.hairRootFade, 0.06f);
  EXPECT_FLOAT_EQ(row.hairRootV, 1);
  EXPECT_FLOAT_EQ(row.hairSpecular, 0.5f);
  EXPECT_EQ(row.hairRootSigma, glm::vec3(0.5f, 0.6f, 0.7f));
  EXPECT_EQ(row.hairTipSigma, glm::vec3(0.1f, 0.2f, 0.3f));
  EXPECT_FLOAT_EQ(row.hairStrandSin, glm::sin(0.5f));
  EXPECT_FLOAT_EQ(row.hairStrandCos, glm::cos(0.5f));
  EXPECT_EQ(row.nrMask, glm::vec4(0.1f, 0.2f, 0.3f, 0.4f));
  EXPECT_EQ(row.unlit, 1);

  // each map's info is the one its row names, by the texture it was given
  ASSERT_EQ(infos.size(), 29u);
  const auto textureOf = [&](uint16_t at) { return infos[at].index; };
  EXPECT_EQ(textureOf(row.emissiveTexture), 1);
  EXPECT_EQ(textureOf(row.normalTexture), 2);
  EXPECT_EQ(textureOf(row.pbrBaseColorTexture), 3);
  EXPECT_EQ(textureOf(row.pbrMetallicRoughnessTexture), 4);
  EXPECT_EQ(textureOf(row.occlusionTexture), 5);
  EXPECT_EQ(textureOf(row.transmissionTexture), 6);
  EXPECT_EQ(textureOf(row.thicknessTexture), 7);
  EXPECT_EQ(textureOf(row.clearcoatRoughnessTexture), 8);
  EXPECT_EQ(textureOf(row.clearcoatTexture), 9);
  EXPECT_EQ(textureOf(row.clearcoatNormalTexture), 10);
  EXPECT_EQ(textureOf(row.specular2Texture), 28);
  EXPECT_EQ(textureOf(row.specular2RoughnessTexture), 29);
  EXPECT_EQ(textureOf(row.specularTexture), 11);
  EXPECT_EQ(textureOf(row.specularColorTexture), 12);
  EXPECT_EQ(textureOf(row.iridescenceTexture), 13);
  EXPECT_EQ(textureOf(row.iridescenceThicknessTexture), 14);
  EXPECT_EQ(textureOf(row.anisotropyTexture), 15);
  EXPECT_EQ(textureOf(row.sheenColorTexture), 16);
  EXPECT_EQ(textureOf(row.sheenRoughnessTexture), 17);
  EXPECT_EQ(textureOf(row.pbrDiffuseTexture), 18);
  EXPECT_EQ(textureOf(row.pbrSpecularGlossinessTexture), 19);
  EXPECT_EQ(textureOf(row.diffuseTransmissionTexture), 20);
  EXPECT_EQ(textureOf(row.diffuseTransmissionColorTexture), 21);
  EXPECT_EQ(textureOf(row.retroreflectionTexture), 22);
  EXPECT_EQ(textureOf(row.scatterStrengthTexture), 23);
  EXPECT_EQ(textureOf(row.multiscatterColorTexture), 24);
  EXPECT_EQ(textureOf(row.detailNormalTexture), 25);
  EXPECT_EQ(textureOf(row.detailWeightTexture), 26);
  EXPECT_EQ(textureOf(row.detailRoughnessTexture), 27);

  // written to glTF and read back, the same row, byte for byte
  std::vector<shaderio::GltfTextureInfo> again;
  const shaderio::GltfShadeMaterial      through = nvvkgltf::MaterialCache::convertOne(fv::mat::toGltf(m), {}, 0, again);
  EXPECT_EQ(std::memcmp(&row, &through, sizeof(row)), 0);
  ASSERT_EQ(again.size(), infos.size());
  EXPECT_EQ(std::memcmp(infos.data(), again.data(), infos.size() * sizeof(infos[0])), 0);
}

// A lobe left out packs as one at its defaults, what the shader reads where it has that feature
TEST(MaterialCache, ALobeLeftOutPacksItsDefaults)
{
  std::vector<shaderio::GltfTextureInfo> infos;
  const shaderio::GltfShadeMaterial      row = nvvkgltf::MaterialCache::convertOne(fv::mat::Material{}, {}, 0, infos);
  EXPECT_TRUE(infos.empty());
  EXPECT_FLOAT_EQ(row.ior, 1.5f);
  EXPECT_FLOAT_EQ(row.specularFactor, 1);
  EXPECT_EQ(row.specularColorFactor, glm::vec3(1));
  EXPECT_EQ(row.attenuationDistance, FLT_MAX);
  EXPECT_EQ(row.attenuationColor, glm::vec3(1));
  EXPECT_FLOAT_EQ(row.normalTextureScale, 1);
  EXPECT_FLOAT_EQ(row.occlusionStrength, 1);
  EXPECT_FLOAT_EQ(row.iridescenceIor, 1.3f);
  EXPECT_FLOAT_EQ(row.iridescenceThicknessMaximum, 400);
  EXPECT_EQ(row.multiscatterColorFactor, glm::vec3(1));
  EXPECT_FLOAT_EQ(row.detailRoughnessFactor, 1);
  EXPECT_EQ(row.nrMask, glm::vec4(1));
  EXPECT_EQ(row.pbrModel, shaderio::PbrModel::ePbrModelMetallicRoughness);
  EXPECT_EQ(row.hairFactor, 0.0f);
}

TEST(MaterialCache, BuildResolvesSamplerIndexFromSlots)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.pbrMetallicRoughness.baseColorTexture.index = 1;  // references glTF texture 1

  std::vector<tinygltf::Material> materials = {mat};
  // Per-texture sampler slots: texture 0 -> default (0), texture 1 -> a non-default sampler (3).
  nvvkgltf::TextureSlotTable textureSlots;
  textureSlots.samplerSlots = {0, 3};
  cache.buildFromMaterials(materials, textureSlots);

  ASSERT_EQ(cache.getShadeMaterials().size(), 1u);
  const uint32_t infoSlot = cache.getShadeMaterials()[0].pbrBaseColorTexture;
  ASSERT_GT(infoSlot, 0u);  // 0 is the sentinel
  ASSERT_LT(infoSlot, cache.getTextureInfos().size());
  const auto& info = cache.getTextureInfos()[infoSlot];
  EXPECT_EQ(info.index, 1);
  EXPECT_EQ(static_cast<int>(info.samplerIndex), 3);  // propagated from textureSlots.samplerSlots[1]
}

TEST(MaterialCache, UpdateMaterialInPlace)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.alphaMode                                   = "OPAQUE";
  mat.pbrMetallicRoughness.roughnessFactor        = 0.5;
  mat.pbrMetallicRoughness.baseColorTexture.index = 0;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials, {});

  // Modify roughness
  mat.pbrMetallicRoughness.roughnessFactor = 0.9;
  auto result                              = cache.updateMaterial(0, mat, {});

  EXPECT_FALSE(result.topologyChanged);
  EXPECT_FLOAT_EQ(cache.getShadeMaterials()[0].pbrRoughnessFactor, 0.9f);
}

TEST(MaterialCache, UpdateDetectsTopologyChangeWhenTextureAdded)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.alphaMode = "OPAQUE";

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials, {});

  // Now add a base color texture — topology change
  mat.pbrMetallicRoughness.baseColorTexture.index = 0;
  auto result                                     = cache.updateMaterial(0, mat, {});

  EXPECT_TRUE(result.topologyChanged);
}

TEST(MaterialCache, UpdateDetectsTopologyChangeWhenTextureRemoved)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.pbrMetallicRoughness.baseColorTexture.index = 0;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials, {});

  // Remove the texture — topology change
  mat.pbrMetallicRoughness.baseColorTexture.index = -1;
  auto result                                     = cache.updateMaterial(0, mat, {});

  EXPECT_TRUE(result.topologyChanged);
}

TEST(MaterialCache, UpdateOutOfRangeReturnsEmpty)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material              mat;
  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials, {});

  auto result = cache.updateMaterial(5, mat, {});
  EXPECT_FALSE(result.topologyChanged);
  EXPECT_FALSE(result.span.hasAny());
}

TEST(MaterialCache, ClearResetsAll)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material              mat;
  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials, {});

  cache.clear();
  EXPECT_TRUE(cache.getShadeMaterials().empty());
  EXPECT_TRUE(cache.getTextureInfos().empty());
}

// A textured emitter weighs as a light by its texture's mean (emissiveLightScale): the mean when it
// is known, 1 when it is not (the factor alone), and none for an all-black texture. A constant
// emitter's scale is 1. An in-place update keeps it.
TEST(MaterialCache, AnEmissiveTextureWeighsByItsMean)
{
  nvvkgltf::TextureSlotTable slots;
  slots.meanLuminance = {0.25f, -1.0f, 0.0f};

  tinygltf::Material screen, unknown, black, lamp;
  screen.emissiveFactor        = {1.0, 1.0, 1.0};
  screen.emissiveTexture.index = 0;
  unknown.emissiveFactor        = {1.0, 1.0, 1.0};
  unknown.emissiveTexture.index = 1;
  black.emissiveFactor        = {1.0, 1.0, 1.0};
  black.emissiveTexture.index = 2;
  lamp.emissiveFactor = {2.0, 2.0, 2.0};

  nvvkgltf::MaterialCache cache;
  cache.buildFromMaterials({screen, unknown, black, lamp}, slots);
  const auto& shade = cache.getShadeMaterials();
  ASSERT_EQ(shade.size(), 4u);
  EXPECT_FLOAT_EQ(shade[0].emissiveLightScale, 0.25f);
  EXPECT_FLOAT_EQ(shade[1].emissiveLightScale, 1.0f);
  EXPECT_FLOAT_EQ(shade[2].emissiveLightScale, 0.0f);
  EXPECT_FLOAT_EQ(shade[3].emissiveLightScale, 1.0f);

  screen.emissiveFactor = {3.0, 3.0, 3.0};
  EXPECT_FALSE(cache.updateMaterial(0, screen, slots).topologyChanged);
  EXPECT_FLOAT_EQ(cache.getShadeMaterials()[0].emissiveLightScale, 0.25f);
}

// The mean is linear luminance: an sRGB mid grey is 0.214, a half-white checker 0.5 in either channel
// order, a pure blue Rec. 709's 0.0722; a format it cannot read is unknown.
TEST(MaterialCache, AnImagesMeanLuminanceIsLinear)
{
  const auto image = [](int w, int h, int channels, auto texel) {
    std::vector<char> t(size_t(w) * h * channels);
    for(int y = 0; y < h; ++y)
      for(int x = 0; x < w; ++x)
        for(int c = 0; c < channels; ++c)
          t[(size_t(y) * w + x) * channels + c] = char(texel(x, y, c));
    return t;
  };
  const auto grey = image(4, 4, 4, [](int, int, int c) { return c == 3 ? 255 : 128; });
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(VK_FORMAT_R8G8B8A8_SRGB, {4, 4}, grey), 0.2158f, 1e-3f);
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(VK_FORMAT_R8G8B8A8_UNORM, {4, 4}, grey), 128.0f / 255.0f, 1e-3f);

  // 256 x 256: the 64 x 64 grid lands on every fourth texel, the checker's both colours
  const auto checker = image(256, 256, 3, [](int x, int y, int) { return ((x / 4 + y / 4) % 2) ? 255 : 0; });
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(VK_FORMAT_R8G8B8_SRGB, {256, 256}, checker), 0.5f, 0.02f);

  const auto blue = image(2, 2, 4, [](int, int, int c) { return c == 2 || c == 3 ? 255 : 0; });
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(VK_FORMAT_R8G8B8A8_UNORM, {2, 2}, blue), 0.0722f, 1e-4f);
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(VK_FORMAT_B8G8R8A8_UNORM, {2, 2}, blue), 0.2126f, 1e-4f);  // read as red

  EXPECT_LT(nvvkgltf::imageMeanLuminance(VK_FORMAT_BC7_SRGB_BLOCK, {4, 4}, grey), 0.0f);
  EXPECT_LT(nvvkgltf::imageMeanLuminance(VK_FORMAT_R8G8B8A8_SRGB, {64, 64}, grey), 0.0f);  // too few bytes
}

#if MAT_EXT_TEXTURE_TRANSFORM
// KHR_texture_transform as the shader applies it (mul(float3(uv, 1), M): rows 0 and 1 by u and v,
// plus row 2) is the spec's offset + R * scale * uv. Its example samples the lower-left quadrant of
// the image turned clockwise by 90 degrees; a non-uniform scale is applied before the rotation.
TEST(MaterialCache, ATextureTransformIsTranslationRotationScale)
{
  const auto transformed = [](glm::vec2 offset, float rotation, glm::vec2 scale, glm::vec2 uv) {
    tinygltf::Material    mat;
    KHR_texture_transform tt;
    tt.offset   = offset;
    tt.rotation = rotation;
    tt.scale    = scale;
    mat.pbrMetallicRoughness.baseColorTexture.index = 0;
    tinygltf::utils::setTextureTransform(mat.pbrMetallicRoughness.baseColorTexture, tt);
    nvvkgltf::MaterialCache cache;
    cache.buildFromMaterials({mat}, {});
    const auto& m = cache.getTextureInfos()[cache.getShadeMaterials()[0].pbrBaseColorTexture].uvTransform;
    return glm::vec2(uv.x * m[0] + uv.y * m[1] + m[2]);
  };
  const float quarter = 1.57079632679f;
  // the example: (0, 0) samples the quadrant's bottom-left corner, (1, 0) its top-left, (1, 1) its top-right
  const glm::vec2 corner = transformed({0, 1}, quarter, {0.5f, 0.5f}, {0, 0});
  const glm::vec2 centre = transformed({0, 1}, quarter, {0.5f, 0.5f}, {1, 1});
  const glm::vec2 top    = transformed({0, 1}, quarter, {0.5f, 0.5f}, {1, 0});
  EXPECT_NEAR(corner.x, 0, 1e-6f);
  EXPECT_NEAR(corner.y, 1, 1e-6f);
  EXPECT_NEAR(centre.x, 0.5f, 1e-6f);
  EXPECT_NEAR(centre.y, 0.5f, 1e-6f);
  EXPECT_NEAR(top.x, 0, 1e-6f);
  EXPECT_NEAR(top.y, 0.5f, 1e-6f);

  // T * R * S with a non-uniform scale: the scale lies along u and v before the turn
  const float     r = 0.3f, c = std::cos(r), s = std::sin(r);
  const glm::vec2 uv{0.7f, 0.2f}, scale{2.0f, 0.5f}, offset{0.1f, -0.4f};
  const glm::vec2 su  = scale * uv;
  const glm::vec2 got = transformed(offset, r, scale, uv);
  EXPECT_NEAR(got.x, offset.x + c * su.x + s * su.y, 1e-5f);
  EXPECT_NEAR(got.y, offset.y - s * su.x + c * su.y, 1e-5f);
}
#endif
