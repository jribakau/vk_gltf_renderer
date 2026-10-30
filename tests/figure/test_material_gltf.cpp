// A material and glTF, both ways: what a file's material becomes, and that writing it back loses nothing.

#include <gtest/gtest.h>

#include <figure/materials/material_gltf.hpp>

namespace {

using namespace fv::mat;

tinygltf::Value object(tinygltf::Value::Object o = {})
{
  return tinygltf::Value(std::move(o));
}

TextureSlot slot(int index)
{
  TextureSlot s;
  s.index = index;
  return s;
}

// Every field away from its default, every map a different texture
Material everything()
{
  Material m;
  m.name                 = "everything";
  m.id                   = "m:3";
  m.walkThrough          = true;
  m.baseColor            = {0.1f, 0.2f, 0.3f, 0.4f};
  m.baseColorMap         = slot(1);
  m.metallic             = 0.25f;
  m.roughness            = 0.75f;
  m.metallicRoughnessMap = slot(2);
  m.alphaMode            = AlphaMode::eMask;
  m.alphaCutoff          = 0.3f;
  m.doubleSided          = true;
  m.normalMap            = slot(3);
  m.normalMap.scale      = 0.6f;
  m.normalMap.flipGreen  = true;
  m.occlusionMap         = slot(4);
  m.occlusionMap.scale   = 0.7f;
  m.emissive             = {0.5f, 0.6f, 0.7f};
  m.emissiveMap          = slot(5);
  m.emissiveStrength     = 12.5f;

  m.baseColorMap.texCoord  = 1;
  m.baseColorMap.transform = UvTransform{{2, 3}, {0.25f, 0.5f}, 0.125f};

  m.specular  = Specular{.factor = 0.8f, .color = {0.9f, 0.8f, 0.7f}, .map = slot(6), .colorMap = slot(7)};
  m.ior       = 1.33f;
  m.clearcoat = Clearcoat{.factor = 0.5f, .roughness = 0.2f, .map = slot(8), .roughnessMap = slot(9), .normalMap = slot(10)};
  m.clearcoat->normalMap.scale = 0.4f;
  m.specular2 = Specular2{.factor = 0.3f, .roughness = 0.15f, .map = slot(28), .roughnessMap = slot(29)};
  m.sheen        = Sheen{.color = {0.1f, 0.2f, 0.4f}, .roughness = 0.3f, .colorMap = slot(11), .roughnessMap = slot(12)};
  m.transmission = Transmission{.factor = 0.9f, .map = slot(13)};
  m.volume       = Volume{.thickness = 0.5f, .thicknessMap = slot(14), .attenuationDistance = 2, .attenuationColor = {0.3f, 0.2f, 0.1f}};
  m.scatter      = Scatter{.strength = 0.6f, .strengthMap = slot(15), .multiscatterColor = {0.7f, 0.6f, 0.5f}, .multiscatterColorMap = slot(16), .anisotropy = -0.5f};
  m.scatterDistance = Rgb{0.003f, 0.002f, 0.001f};
  m.anisotropy      = Anisotropy{.strength = 0.5f, .rotation = 1.25f, .map = slot(17)};
  m.iridescence     = Iridescence{.factor = 0.5f, .ior = 1.4f, .thicknessMinimum = 150, .thicknessMaximum = 350, .map = slot(18), .thicknessMap = slot(19)};
  m.dispersion      = 0.5f;
  m.diffuseTransmission = DiffuseTransmission{.factor = 0.3f, .color = {0.4f, 0.5f, 0.6f}, .map = slot(20), .colorMap = slot(21)};
  m.retroreflection     = Retroreflection{.factor = 0.2f, .map = slot(22)};
  m.specularGlossiness  = SpecularGlossiness{.diffuse = {0.2f, 0.3f, 0.4f, 0.5f}, .specular = {0.6f, 0.7f, 0.8f}, .glossiness = 0.9f, .diffuseMap = slot(23), .specularGlossinessMap = slot(24)};
  m.detail = Detail{.weight = 0.5f, .roughness = 0.6f, .normalMap = slot(25), .weightMap = slot(26), .roughnessMap = slot(27)};
  m.detail->normalMap.scale = 0.2f;
  m.hair = Hair{.longitudinal = 0.3f, .azimuthal = 0.4f, .rootAbsorption = {0.5f, 0.6f, 0.7f}, .tipAbsorption = {0.1f, 0.2f, 0.3f}, .rootFade = 0.06f, .rootV = 1, .specular = 0.5f, .strandRotation = 0.5f};
  m.dlssNrMask = Rgba{0.1f, 0.2f, 0.3f, 0.4f};
  m.unlit      = true;
  return m;
}

TEST(MaterialGltf, EveryFieldGoesThereAndBack)
{
  const Material m = everything();
  EXPECT_TRUE(fromGltf(toGltf(m)) == m);
}

TEST(MaterialGltf, AFibreFromItsColourGoesThereAndBack)
{
  Material m;
  m.hair = Hair{.longitudinal = 0.3f, .azimuthal = 0.4f, .specular = 0.5f, .absorptionFromColor = true};
  EXPECT_TRUE(fromGltf(toGltf(m)) == m);
}

// Which lobes are present decides the shader's features, whatever their values
TEST(MaterialGltf, ALobeLeftOutStaysOutAndOneAtItsDefaultsStaysIn)
{
  Material plain;
  EXPECT_TRUE(toGltf(plain).extensions.empty());
  EXPECT_TRUE(fromGltf(toGltf(plain)) == plain);

  Material at;
  at.clearcoat.emplace();
  at.sheen.emplace();
  at.ior = 1.5f;
  const tinygltf::Material g = toGltf(at);
  EXPECT_EQ(g.extensions.size(), 3u);
  EXPECT_TRUE(fromGltf(g) == at);
}

TEST(MaterialGltf, ADefaultExtensionReadsAsGltfDefaults)
{
  tinygltf::Material g;
  g.extensions["KHR_materials_volume"]   = object();
  g.extensions["KHR_materials_specular"] = object();
  g.extensions["FV_materials_hair"]      = object();
  const Material m = fromGltf(g);
  ASSERT_TRUE(m.volume && m.specular && m.hair);
  EXPECT_EQ(m.volume->attenuationDistance, FLT_MAX);
  EXPECT_EQ(m.specular->factor, 1.f);
  EXPECT_EQ(m.specular->color, (Rgb{1, 1, 1}));
  EXPECT_EQ(m.hair->longitudinal, 0.2f);
}

// An earlier draft of the scatter extension scatters at full strength, its colour black unless it says
// otherwise, and under another name for the colour. Written back, it takes the current name.
TEST(MaterialGltf, TheScatterDraftKeepsItsDefaults)
{
  tinygltf::Material draft;
  draft.extensions["KHR_materials_volume_scatter"] = object();
  Material m = fromGltf(draft);
  ASSERT_TRUE(m.scatter);
  EXPECT_EQ(m.scatter->strength, 1.f);
  EXPECT_EQ(m.scatter->multiscatterColor, (Rgb{0, 0, 0}));

  tinygltf::Value::Array colour{tinygltf::Value(0.25), tinygltf::Value(0.5), tinygltf::Value(0.75)};
  draft.extensions["KHR_materials_volume_scatter"] = object({{"multiscatterColor", tinygltf::Value(colour)}});
  m = fromGltf(draft);
  EXPECT_EQ(m.scatter->multiscatterColor, (Rgb{0.25f, 0.5f, 0.75f}));

  const tinygltf::Material written = toGltf(m);
  EXPECT_TRUE(written.extensions.count("KHR_materials_scatter"));
  EXPECT_FALSE(written.extensions.count("KHR_materials_volume_scatter"));
  EXPECT_TRUE(fromGltf(written) == m);

  // the current name wins where a file has both
  draft.extensions["KHR_materials_scatter"] = object({{"scatterStrengthFactor", tinygltf::Value(0.5)}});
  EXPECT_EQ(fromGltf(draft).scatter->strength, 0.5f);
}

TEST(MaterialGltf, WhatNothingReadsIsWrittenBack)
{
  tinygltf::Material g;
  g.name                     = "kept";
  g.extensions["FV_unknown"] = object({{"a", tinygltf::Value(1)}, {"b", tinygltf::Value(0.5)}, {"c", tinygltf::Value(std::string("x"))}});
  g.extras                   = object({{"fromTheFile", tinygltf::Value(true)}, {"fvId", tinygltf::Value(std::string("m:7"))}, {"fvWalkThrough", tinygltf::Value(true)}});
  g.pbrMetallicRoughness.baseColorTexture.index                    = 2;
  g.pbrMetallicRoughness.baseColorTexture.extensions["EXT_unknown"] = object({{"k", tinygltf::Value(3)}});
  g.pbrMetallicRoughness.baseColorTexture.extras                    = object({{"note", tinygltf::Value(4)}});
  g.emissiveTexture.index                                           = 3;
  g.emissiveTexture.extensions["KHR_texture_transform"] = object({{"texCoord", tinygltf::Value(1)}, {"scale", tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(2.0), tinygltf::Value(2.0)})}});

  const Material m = fromGltf(g);
  EXPECT_EQ(m.id, "m:7");
  EXPECT_TRUE(m.walkThrough);
  EXPECT_TRUE(m.foreignExtensions.count("FV_unknown"));
  EXPECT_FALSE(m.foreignExtras.Has("fvId"));
  EXPECT_EQ(m.emissiveMap.transformTexCoord, 1);

  std::vector<std::string> used;
  const tinygltf::Material out = toGltf(m, used);
  EXPECT_TRUE(out.extensions.at("FV_unknown") == g.extensions.at("FV_unknown"));
  EXPECT_TRUE(out.extras == g.extras);
  EXPECT_TRUE(out.pbrMetallicRoughness.baseColorTexture.extensions == g.pbrMetallicRoughness.baseColorTexture.extensions);
  EXPECT_TRUE(out.pbrMetallicRoughness.baseColorTexture.extras == g.pbrMetallicRoughness.baseColorTexture.extras);
  EXPECT_EQ(out.emissiveTexture.extensions.at("KHR_texture_transform").Get("texCoord").Get<int>(), 1);
  for(const char* name : {"FV_unknown", "EXT_unknown", "KHR_texture_transform"})
    EXPECT_NE(std::find(used.begin(), used.end(), name), used.end()) << name;
}

TEST(MaterialGltf, ExtensionsUsedNamesEachOnce)
{
  Material m;
  m.specular.emplace();
  m.baseColorMap           = slot(0);
  m.baseColorMap.transform = UvTransform{{2, 2}};
  m.normalMap              = slot(1);
  m.normalMap.transform    = UvTransform{{3, 3}};
  m.normalMap.flipGreen    = true;
  std::vector<std::string> used;
  (void)toGltf(m, used);
  (void)toGltf(m, used);
  std::sort(used.begin(), used.end());
  EXPECT_EQ(used, (std::vector<std::string>{"FV_texture_flip_green", "KHR_materials_specular", "KHR_texture_transform"}));
}

TEST(MaterialGltf, AlphaModesBothWays)
{
  for(const auto& [name, mode] : {std::pair{"OPAQUE", AlphaMode::eOpaque}, {"MASK", AlphaMode::eMask}, {"BLEND", AlphaMode::eBlend}})
  {
    tinygltf::Material g;
    g.alphaMode = name;
    EXPECT_EQ(fromGltf(g).alphaMode, mode);
    EXPECT_EQ(toGltf(fromGltf(g)).alphaMode, name);
  }
}

}  // namespace
