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
#include <bit>
#include <cmath>
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
  m.clearcoat->color     = {0.9f, 0.5f, 0.25f};
  m.clearcoat->colorMap  = slot(30);
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
  m.chamberIor           = 1.336f;
  m.pane                 = true;
  m.film                 = true;
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
  EXPECT_EQ(row.opacityFilm, 1);
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
  EXPECT_EQ(row.clearcoatColor, glm::vec3(0.9f, 0.5f, 0.25f));
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
  EXPECT_FLOAT_EQ(row.chamberIor, 1.336f);
  {
    // a surface in the chamber says so by the sign: the film's index is positive
    fv::mat::Material iris;
    iris.inChamberIor = 1.336f;
    std::vector<shaderio::GltfTextureInfo> none;
    EXPECT_FLOAT_EQ(nvvkgltf::MaterialCache::convertOne(iris, {}, 0, none).chamberIor, -1.336f);
  }
  EXPECT_FLOAT_EQ(row.plate, 1.0f);
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
  ASSERT_EQ(infos.size(), 30u);
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
  EXPECT_EQ(textureOf(row.clearcoatColorTexture), 30);
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

// A texture whose mips came with its image (the cooker's) says so to the shader, which reads a normal
// map's mip lengths only then (toksvigVariance); one with mips made here from a raw source doesn't
TEST(MaterialCache, ATexturesCookedMipsReachItsInfo)
{
  tinygltf::Material mat;
  mat.normalTexture.index    = 0;
  mat.occlusionTexture.index = 1;
  nvvkgltf::TextureSlotTable slots;
  slots.cookedMips = {0, 1};
  std::vector<shaderio::GltfTextureInfo> infos(1);
  const shaderio::GltfShadeMaterial      row     = nvvkgltf::MaterialCache::convertOne(mat, slots, 0, infos);
  const auto                             flagged = [&](uint16_t slot) {
    return (infos.at(slot).flags & static_cast<uint8_t>(shaderio::TextureInfoFlagBits::eTexInfoCookedMips)) != 0;
  };
  EXPECT_FALSE(flagged(uint16_t(row.normalTexture)));
  EXPECT_TRUE(flagged(uint16_t(row.occlusionTexture)));
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

// A BC7 block's first texel as a full decoder gives it, in every mode: random blocks with the mode's
// bits set, decoded by texture2ddecoder (the app's Rust test dependency) on 2026-11-06, three a mode
TEST(MaterialCache, ABc7BlocksFirstTexelIsWhatADecoderGives)
{
  struct Vector
  {
    uint8_t block[16];
    uint8_t texel[4];
  };
  const Vector vectors[] = {
      {{0xad, 0x76, 0x36, 0x74, 0xec, 0x79, 0xcf, 0xea, 0x8b, 0x8e, 0x15, 0x03, 0xfd, 0x9e, 0x1f, 0xff}, {110, 63, 94, 255}},
      {{0xb9, 0x75, 0x4f, 0x19, 0x6d, 0xef, 0x1a, 0xde, 0xbd, 0xe4, 0x13, 0x3f, 0x2d, 0x7d, 0x37, 0xf5}, {208, 131, 77, 255}},
      {{0x5b, 0xec, 0xed, 0x52, 0xf6, 0x09, 0xb3, 0x20, 0x5e, 0xc2, 0xb9, 0xac, 0xbb, 0xd2, 0x0d, 0x75}, {64, 99, 51, 255}},
      {{0xba, 0xec, 0x5f, 0xd9, 0x26, 0x12, 0x10, 0x26, 0xa6, 0x79, 0xaf, 0xc6, 0xe3, 0xc8, 0x17, 0x45}, {211, 104, 131, 255}},
      {{0x72, 0xe5, 0xbe, 0x01, 0xa9, 0xce, 0x5d, 0xa9, 0x32, 0x1c, 0x80, 0x63, 0x8f, 0x4c, 0x5e, 0x30}, {149, 165, 165, 255}},
      {{0xea, 0x2f, 0xe1, 0x26, 0xfa, 0x75, 0xb5, 0x44, 0xb6, 0xbf, 0x12, 0x0c, 0x12, 0x1a, 0x1c, 0xb2}, {189, 233, 16, 255}},
      {{0x3c, 0x3f, 0xf7, 0xae, 0x4b, 0x07, 0xd5, 0xd4, 0x59, 0x2f, 0x4d, 0x97, 0xd0, 0x6b, 0x9d, 0xa2}, {255, 115, 214, 255}},
      {{0x04, 0xcb, 0x54, 0xfe, 0x59, 0x19, 0xcd, 0xba, 0x91, 0xbc, 0xe1, 0x12, 0xaa, 0x29, 0x8d, 0xe4}, {79, 145, 63, 255}},
      {{0x3c, 0xa3, 0x6a, 0xa6, 0x37, 0x3f, 0x97, 0x24, 0x82, 0x55, 0x75, 0x91, 0x7e, 0x3e, 0x52, 0x9c}, {121, 217, 123, 255}},
      {{0x88, 0x66, 0xc4, 0x05, 0x21, 0x09, 0x3f, 0x87, 0xf4, 0x67, 0xfb, 0xb2, 0xcb, 0xe6, 0xf9, 0xba}, {50, 72, 250, 255}},
      {{0x48, 0x71, 0xf7, 0x00, 0xba, 0x7b, 0xfd, 0xd2, 0x63, 0x47, 0x99, 0x04, 0x88, 0xed, 0x30, 0xee}, {184, 220, 176, 255}},
      {{0x98, 0x04, 0x32, 0xa8, 0x57, 0xad, 0x78, 0x52, 0xce, 0x15, 0x1b, 0xa1, 0x9c, 0x55, 0xb7, 0xa6}, {18, 117, 161, 255}},
      {{0x70, 0x39, 0xd5, 0x4a, 0xfc, 0x65, 0xc3, 0xe4, 0x7d, 0x61, 0x71, 0x69, 0x7d, 0xb7, 0xc6, 0xef}, {206, 173, 93, 33}},
      {{0xf0, 0x6b, 0x0f, 0x39, 0x51, 0x75, 0x53, 0x5b, 0xb3, 0x84, 0xe1, 0x60, 0xf6, 0x83, 0x72, 0xae}, {90, 24, 85, 156}},
      {{0x50, 0x7b, 0x33, 0x4b, 0xc5, 0x4c, 0x02, 0xb6, 0x27, 0x8f, 0x06, 0x50, 0xfc, 0x99, 0x06, 0xac}, {222, 181, 165, 99}},
      {{0x20, 0x15, 0x5b, 0x67, 0x91, 0xd4, 0x37, 0x2d, 0x6b, 0xd7, 0xc0, 0xd3, 0xfc, 0xcb, 0x7c, 0x98}, {42, 58, 147, 77}},
      {{0x60, 0xeb, 0x86, 0x2f, 0x51, 0xff, 0x0e, 0x20, 0xbc, 0xd5, 0xb2, 0x91, 0xd6, 0xe5, 0x62, 0x0a}, {5, 89, 221, 153}},
      {{0xe0, 0xbb, 0x40, 0xb2, 0x2d, 0xf5, 0xb2, 0xe1, 0xe4, 0x0b, 0x5f, 0x31, 0x83, 0x0e, 0xc8, 0x17}, {80, 171, 91, 173}},
      {{0x40, 0x15, 0x67, 0x91, 0x6c, 0x3e, 0x38, 0xd2, 0x2b, 0x4b, 0x87, 0x86, 0xe7, 0x62, 0x2a, 0x10}, {76, 64, 114, 92}},
      {{0x40, 0xd2, 0xed, 0x60, 0xec, 0xfb, 0x3a, 0x8a, 0x9b, 0xac, 0x85, 0xe4, 0xc9, 0xe4, 0x73, 0xa5}, {85, 56, 210, 47}},
      {{0xc0, 0x57, 0x21, 0x31, 0x99, 0x22, 0x70, 0xd2, 0xed, 0x00, 0xe0, 0xa3, 0x1c, 0x96, 0x93, 0x70}, {61, 27, 106, 134}},
      {{0x80, 0x93, 0x98, 0x8b, 0xd0, 0xa1, 0x5c, 0x3a, 0xa4, 0x85, 0xc4, 0x31, 0x85, 0xb6, 0xe1, 0x1e}, {61, 117, 67, 29}},
      {{0x80, 0x3d, 0xbd, 0x62, 0x1c, 0x34, 0xa8, 0x4b, 0x2b, 0x97, 0x04, 0xae, 0xb3, 0x94, 0x0f, 0x79}, {162, 56, 113, 40}},
      {{0x80, 0x8c, 0x97, 0xe6, 0xc3, 0x3e, 0xf8, 0x53, 0xbd, 0xde, 0x19, 0xa9, 0x76, 0x48, 0xcc, 0x67}, {212, 166, 109, 177}},
      // the cooker's own uniform blocks (texcache.rs encode_bc7): modes 6, 5, 4, 3 and 0
      {{0x40, 0x32, 0x99, 0xc7, 0xa3, 0x50, 0xfe, 0x7f, 0x12, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}, {200, 120, 40, 254}},
      {{0x20, 0x64, 0x32, 0x8f, 0x47, 0xa1, 0xfc, 0xff, 0xaf, 0xaa, 0xaa, 0xaa, 0x56, 0x55, 0x55, 0x55}, {201, 120, 40, 255}},
      {{0x10, 0xff, 0xff, 0xff, 0xff, 0xff, 0xaf, 0xaa, 0xaa, 0xaa, 0x4a, 0x92, 0x24, 0x49, 0x92, 0x24}, {255, 255, 255, 255}},
      {{0x08, 0x14, 0x08, 0x05, 0xc2, 0x68, 0x34, 0x1a, 0x05, 0x83, 0xc1, 0x20, 0xac, 0xaa, 0xaa, 0xaa}, {9, 70, 130, 255}},
      {{0xe1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x2f, 0x29, 0x49, 0x92, 0x24, 0x49}, {255, 255, 255, 255}},
  };
  for(const Vector& v : vectors)
  {
    const std::array<uint8_t, 4> t = nvvkgltf::bc7FirstTexel(v.block);
    SCOPED_TRACE(testing::Message() << "mode " << std::countr_zero(unsigned(v.block[0])) << ", block byte " << int(v.block[1]));
    for(int c = 0; c < 4; c++)
      EXPECT_EQ(int(t[c]), int(v.texel[c])) << "channel " << c;
  }
  const uint8_t reserved[16] = {};
  EXPECT_EQ(nvvkgltf::bc7FirstTexel(reserved), (std::array<uint8_t, 4>{0, 0, 0, 0}));

  const struct
  {
    uint8_t block[8];
    uint8_t value;
  } bc4[] = {{{0x8b, 0x7e, 0xfc, 0xc7, 0x4c, 0xaa, 0xc3, 0xf2}, 133}, {{0x93, 0xb2, 0x65, 0x6f, 0x71, 0x9b, 0x4a, 0xd6}, 171},
             {{0x3f, 0x83, 0x14, 0x4c, 0x00, 0xf2, 0x35, 0x65}, 103}, {{0x2d, 0x31, 0xb3, 0x5a, 0x54, 0x30, 0xca, 0x59}, 46},
             {{0x03, 0x95, 0x7a, 0x3e, 0xa2, 0x2f, 0x81, 0xfe}, 32},  {{0x29, 0x49, 0xb9, 0xd8, 0x93, 0x50, 0x02, 0x2a}, 73}};
  for(const auto& v : bc4)
    EXPECT_EQ(int(nvvkgltf::bc4FirstTexel(v.block)), int(v.value));
}

// A cooked image's mean is its 1 x 1 level's one texel, linear, through its swizzle; a chain that
// stops short of 1 x 1 has none, and an 8-bit image reads as before
TEST(MaterialCache, ACookedImagesMeanLuminanceIsItsSmallestLevel)
{
  const auto lin = [](float v) {
    const float c = v / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
  };
  // the cooker's block of (200, 120, 40) at the 1 x 1 level of a 2 x 2 image
  const uint8_t   tone[16] = {0x40, 0x32, 0x99, 0xc7, 0xa3, 0x50, 0xfe, 0x7f, 0x12, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
  const auto      level    = [](const uint8_t* b, size_t n) { return std::vector<char>(reinterpret_cast<const char*>(b), reinterpret_cast<const char*>(b) + n); };
  nvvkgltf::LoadedImageData cooked;
  cooked.format  = VK_FORMAT_BC7_SRGB_BLOCK;
  cooked.size    = {2, 2};
  cooked.mipData = {std::vector<char>(16, 0), level(tone, 16)};
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(cooked), 0.2126f * lin(200) + 0.7152f * lin(120) + 0.0722f * lin(40), 1e-5f);
  cooked.format = VK_FORMAT_BC7_UNORM_BLOCK;  // a data map: its bytes are linear
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(cooked), (0.2126f * 200 + 0.7152f * 120 + 0.0722f * 40) / 255.0f, 1e-5f);
  cooked.mipData.pop_back();
  EXPECT_LT(nvvkgltf::imageMeanLuminance(cooked), 0.0f) << "no 1 x 1 level";

  // a one-grey map, swizzled "rrr1"
  const uint8_t             grey[8] = {0x80, 0x80, 0, 0, 0, 0, 0, 0};
  nvvkgltf::LoadedImageData mask;
  mask.format           = VK_FORMAT_BC4_UNORM_BLOCK;
  mask.size             = {1, 1};
  mask.mipData          = {level(grey, 8)};
  mask.componentMapping = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(mask), 128.0f / 255.0f, 1e-5f);

  nvvkgltf::LoadedImageData rgba;
  rgba.format  = VK_FORMAT_R8G8B8A8_UNORM;
  rgba.size    = {1, 1};
  rgba.mipData = {std::vector<char>{char(255), 0, 0, char(255)}};
  EXPECT_NEAR(nvvkgltf::imageMeanLuminance(rgba), 0.2126f, 1e-5f);
}

// A normal map's slopes' variance is its texels': the mean of the two axes' variances of the slope of the
// normal the shader reads (z rebuilt from x and y in two channels, stored in three), the same from 8-bit
// texels and from BC5 and BC7 blocks of them; a flat map has none; an sRGB image is no normal map.
TEST(MaterialCache, ANormalMapsSlopeVarianceIsItsTexels)
{
  // x at 0.6 and -0.6 (bytes 204 and 51), y at 128: z^2 = 0.64 - y^2, the slopes +-0.6 / z along x
  const float y = 128.0f / 255.0f * 2.0f - 1.0f, expected = 0.5f * 0.36f / (0.64f - y * y);
  nvvkgltf::LoadedImageData rg;
  rg.format  = VK_FORMAT_R8G8_UNORM;
  rg.size    = {2, 2};
  rg.mipData = {std::vector<char>{char(204), char(128), char(51), char(128), char(51), char(128), char(204), char(128)}};
  EXPECT_NEAR(nvvkgltf::imageSlopeVariance(rg), expected, 1e-4f);
  rg.mipData = {std::vector<char>(8, char(128))};
  EXPECT_NEAR(nvvkgltf::imageSlopeVariance(rg), 0.0f, 1e-6f);

  // one BC5 block: x's endpoints 204 and 51, its texels alternating between them; y's one value, 128
  uint8_t  block[16] = {204, 51, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 0, 0};
  uint64_t bits      = 0;
  for(int t = 0; t < 16; t++)
    bits |= uint64_t(t % 2) << (3 * t);
  for(int i = 0; i < 6; i++)
    block[2 + i] = uint8_t(bits >> (8 * i));
  float x[16];
  nvvkgltf::bc4Decode(block, false, x);
  EXPECT_FLOAT_EQ(x[0], 0.8f);
  EXPECT_FLOAT_EQ(x[1], 0.2f);
  nvvkgltf::LoadedImageData bc5;
  bc5.format  = VK_FORMAT_BC5_UNORM_BLOCK;
  bc5.size    = {4, 4};
  bc5.mipData = {std::vector<char>(reinterpret_cast<const char*>(block), reinterpret_cast<const char*>(block) + 16)};
  EXPECT_NEAR(nvvkgltf::imageSlopeVariance(bc5), expected, 1e-4f);
  // a map of one block kept for its whole size (LoadedImageData::constant)
  bc5.size     = {4096, 4096};
  bc5.constant = true;
  EXPECT_NEAR(nvvkgltf::imageSlopeVariance(bc5), expected, 1e-4f);

  // three channels store z: x at +-0.6 over z = 230 / 127.5 - 1
  const float               z = 230.0f / 127.5f - 1.0f, stored = 0.5f * 0.36f / (z * z);
  nvvkgltf::LoadedImageData rgba;
  rgba.format  = VK_FORMAT_R8G8B8A8_UNORM;
  rgba.size    = {2, 1};
  rgba.mipData = {std::vector<char>{char(204), char(128), char(230), char(255), char(51), char(128), char(230), char(255)}};
  EXPECT_NEAR(nvvkgltf::imageSlopeVariance(rgba), stored, 2e-3f);
  rgba.format = VK_FORMAT_R8G8B8A8_SRGB;  // a colour, no normal map
  EXPECT_LT(nvvkgltf::imageSlopeVariance(rgba), 0.0f);

  // BC7, mode 6 (both endpoints one RGBA, so every texel it): its first texel of each block; two blocks,
  // x at 204 and 51
  const auto mode6 = [](uint8_t r, uint8_t g, uint8_t b) {
    std::array<uint8_t, 16> out{};
    uint32_t                pos = 0;
    const auto put = [&](uint32_t v, uint32_t n) {
      for(uint32_t i = 0; i < n; i++, pos++)
        out[pos >> 3] |= uint8_t(((v >> i) & 1u) << (pos & 7));
    };
    put(1u << 6, 7);  // mode 6
    for(uint8_t c : {r, g, b, uint8_t(255)})
    {
      put(c >> 1, 7);
      put(c >> 1, 7);
    }
    put(r & 1u, 1);  // the p-bits: the low bit of every channel at once (the channels here share it)
    put(r & 1u, 1);
    return out;
  };
  const std::array<uint8_t, 16> right = mode6(204, 128, 230), left = mode6(51 + 1, 128, 230);
  EXPECT_EQ(int(nvvkgltf::bc7FirstTexel(right.data())[0]), 204);
  nvvkgltf::LoadedImageData bc7;
  bc7.format = VK_FORMAT_BC7_UNORM_BLOCK;
  bc7.size   = {8, 4};
  bc7.mipData = {std::vector<char>(reinterpret_cast<const char*>(right.data()), reinterpret_cast<const char*>(right.data()) + 16)};
  bc7.mipData[0].insert(bc7.mipData[0].end(), reinterpret_cast<const char*>(left.data()), reinterpret_cast<const char*>(left.data()) + 16);
  const float x52 = 52.0f / 127.5f - 1.0f, x204 = 204.0f / 127.5f - 1.0f, mean = 0.5f * (x204 + x52) / z;
  const float bc7Expected = 0.5f * (0.5f * ((x204 / z - mean) * (x204 / z - mean) + (x52 / z - mean) * (x52 / z - mean)));
  EXPECT_NEAR(nvvkgltf::imageSlopeVariance(bc7), bc7Expected, 2e-3f);
  bc7.format = VK_FORMAT_BC7_SRGB_BLOCK;
  EXPECT_LT(nvvkgltf::imageSlopeVariance(bc7), 0.0f);
}

// The row carries its normal map's slopes' variance times its strength squared (0 without a map,
// negative where the image's is unknown), and its detail map's at unit strength; an update keeps them
TEST(MaterialCache, ARowCarriesItsReliefsSlopeVariance)
{
  nvvkgltf::TextureSlotTable slots;
  slots.slopeVariance = {0.04f, -1.0f, 0.01f};

  tinygltf::Material known, unknown, none;
  known.normalTexture.index   = 0;
  known.normalTexture.scale   = 2.0;
  unknown.normalTexture.index = 1;

  nvvkgltf::MaterialCache cache;
  cache.buildFromMaterials({known, unknown, none}, slots);
  const auto& shade = cache.getShadeMaterials();
  ASSERT_EQ(shade.size(), 3u);
  EXPECT_FLOAT_EQ(shade[0].normalSlopeVariance, 0.16f);
  EXPECT_LT(shade[1].normalSlopeVariance, 0.0f);
  EXPECT_FLOAT_EQ(shade[2].normalSlopeVariance, 0.0f);
  EXPECT_FLOAT_EQ(shade[2].detailSlopeVariance, 0.0f);

  known.normalTexture.scale = 0.5;
  EXPECT_FALSE(cache.updateMaterial(0, known, slots).topologyChanged);
  EXPECT_FLOAT_EQ(cache.getShadeMaterials()[0].normalSlopeVariance, 0.01f);

  fv::mat::Material detailed;
  detailed.detail.emplace();
  detailed.detail->normalMap.index = 2;
  std::vector<shaderio::GltfTextureInfo> infos;
  EXPECT_FLOAT_EQ(nvvkgltf::MaterialCache::convertOne(detailed, slots, 0, infos).detailSlopeVariance, 0.01f);
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
