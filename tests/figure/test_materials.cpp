// A DSON surface's channels become a Surface (Iray Uber, PBRSkin), and a Surface becomes the glTF
// material the renderer shows: factors, the textures it asks the cooker for, the skin's volume, and
// edits laid over it. A port of the material pipeline in src/lib/materials and src/lib/rtx/gltf.ts.

#include <gtest/gtest.h>

#include <string>

#include <figure/dson/material_preset.hpp>
#include <figure/dson/materials.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/materials/lobes.hpp>
#include <figure/materials/lower.hpp>

namespace {

using fv::dson::Json;

fv::dson::Channels channelsOf(const char* text)
{
  fv::dson::Channels ch;
  const Json         def = Json::parse(text);
  fv::dson::collectChannels(&def, ch, {}, "x.dsf");
  return ch;
}

const char* kSkin = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,0.8,0.7],"image_file":"/Runtime/Textures/face_d.jpg"}},
  {"channel":{"id":"Translucency Weight","type":"float","current_value":0.6}},
  {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/face_n.jpg"}},
  {"channel":{"id":"Bump Strength","type":"float","current_value":0.5,"image_file":"/Runtime/Textures/face_b.jpg"}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.4}},
  {"channel":{"id":"Dual Lobe Specular Weight","type":"float","current_value":0.5}},
  {"channel":{"id":"Specular Lobe 1 Roughness","type":"float","current_value":0.2}},
  {"channel":{"id":"Specular Lobe 2 Roughness","type":"float","current_value":0.5}},
  {"channel":{"id":"Dual Lobe Specular Ratio","type":"float","current_value":0.5}}
]}]})";

const char* kEyeMoisture = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[0.5,0.5,0.5]}},
  {"channel":{"id":"Refraction Weight","type":"float","current_value":1}},
  {"channel":{"id":"Refraction Index","type":"float","current_value":1.376}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.3}}
]}]})";

TEST(Materials, ClassifiesSurfacesByName)
{
  using fv::mat::Kind;
  EXPECT_EQ(fv::mat::classify("Face"), Kind::eSkin);
  EXPECT_EQ(fv::mat::classify("Eye Moisture Left"), Kind::eCornea);
  EXPECT_EQ(fv::mat::classify("Eyelashes"), Kind::eLashes);
  EXPECT_EQ(fv::mat::classify("Fingernails"), Kind::eNails);
  EXPECT_EQ(fv::mat::classify("Scalp Hair"), Kind::eHair);
  EXPECT_EQ(fv::mat::classify("Dress"), Kind::eOther);
}

TEST(Materials, ParsesAnIrayUberSkin)
{
  const auto s = fv::dson::parseSurface("Face", channelsOf(kSkin));
  EXPECT_EQ(s.kind, fv::mat::Kind::eSkin);
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::eIray);
  EXPECT_EQ(s.base.map, "/Runtime/Textures/face_d.jpg");
  // sRGB 0.8 in the picker is 0.604 linear
  EXPECT_NEAR(s.base.color[1], 0.604f, 1e-3f);
  ASSERT_TRUE(s.dual);
  EXPECT_FLOAT_EQ(s.glossy.weight, 0);  // dual-lobe channels and no glossy weight: no glossy layer
  ASSERT_TRUE(s.translucency);
  EXPECT_FLOAT_EQ(s.translucency->weight, 0.6f);
}

TEST(Materials, FoldsTheSpecularLayersIntoOneLobeAndACoat)
{
  const auto s     = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto lobes = fv::mat::specularLobes(s);
  // no glossy layer: the main lobe is the broad dual lobe, the tight one a clearcoat
  EXPECT_FLOAT_EQ(lobes.roughness.value, 0.5f);
  ASSERT_TRUE(lobes.clearcoat);
  EXPECT_FLOAT_EQ(lobes.clearcoat->roughness.value, 0.2f);
}

TEST(Materials, SkinGoesOutAsAVolumeForTheRendererAndPlainForTheRasterizer)
{
  const auto s = fv::dson::parseSurface("Face", channelsOf(kSkin));
  {
    const auto lowered = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eVolume});
    const auto& ext    = lowered.material["extensions"];
    EXPECT_TRUE(ext.contains("KHR_materials_volume"));
    EXPECT_TRUE(ext.contains("KHR_materials_scatter"));
    EXPECT_NEAR(ext["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 0.6f, 1e-5f);
  }
  {
    const auto lowered = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
    const auto& ext    = lowered.material["extensions"];
    EXPECT_FALSE(ext.contains("KHR_materials_transmission"));
    EXPECT_FALSE(ext.contains("KHR_materials_scatter"));
  }
}

TEST(Materials, AsksTheCookerForTheBlendedNormalMap)
{
  const auto s       = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto lowered = fv::mat::lower(s, {}, {});
  ASSERT_EQ(lowered.requests.size(), 2u);
  const fv::mat::TextureRequest& normal = lowered.requests[1];
  EXPECT_EQ(normal.rel, "/Runtime/Textures/face_n.jpg");
  EXPECT_EQ(normal.channels, "ng");
  EXPECT_EQ(normal.bumpRel, "/Runtime/Textures/face_b.jpg");
  EXPECT_FLOAT_EQ(normal.bumpStrength, 0.5f);
  EXPECT_FALSE(normal.srgb);
}

TEST(Materials, EditsLieOverTheSurface)
{
  const auto     s = fv::dson::parseSurface("Face", channelsOf(kSkin));
  fv::mat::Edits e;
  e.roughness = 0.9f;
  e.color     = fv::mat::Rgb{0.2f, 0.2f, 0.2f};
  e.alphaMode = "cutout";
  const auto lowered = fv::mat::lower(s, e, {.skin = fv::mat::SkinMode::eRaster});
  const auto& pbr    = lowered.material["pbrMetallicRoughness"];
  EXPECT_NEAR(pbr["roughnessFactor"].get<float>(), 0.9f, 1e-6f);
  EXPECT_NEAR(pbr["baseColorFactor"][0].get<float>(), 0.2f, 1e-6f);
  EXPECT_EQ(lowered.material["alphaMode"], "MASK");
  EXPECT_FLOAT_EQ(lowered.effective.roughness, 0.9f);
  EXPECT_EQ(lowered.effective.alphaMode, "cutout");
}

TEST(Materials, ACorneaIsGlassNotSkin)
{
  const auto s       = fv::dson::parseSurface("Eye Moisture Left", channelsOf(kEyeMoisture));
  const auto lowered = fv::mat::lower(s, {}, {});
  const auto& ext    = lowered.material["extensions"];
  EXPECT_NEAR(ext["KHR_materials_ior"]["ior"].get<float>(), 1.376f, 1e-5f);
  EXPECT_NEAR(ext["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1.0f, 1e-6f);
  EXPECT_FALSE(ext.contains("KHR_materials_scatter"));
  // refraction does not tint what passes through: the grey base colour goes to white
  EXPECT_NEAR(lowered.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(), 1.0f, 1e-6f);
  EXPECT_EQ(lowered.material["alphaMode"], "OPAQUE");
}

TEST(Materials, ResolvesSlotsIntoTextureInfosAndDropsTheMissing)
{
  const auto s       = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto lowered = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
  std::vector<std::string> used;
  const auto               m = fv::mat::toGltf(
      lowered,
      [](const fv::mat::TextureRequest& r) {
        // the base map exists; the normal map's cook failed
        return r.channels.empty() ? fv::mat::ResolvedTexture{4, false} : fv::mat::ResolvedTexture{};
      },
      used);
  EXPECT_EQ(m.pbrMetallicRoughness.baseColorTexture.index, 4);
  EXPECT_EQ(m.normalTexture.index, -1);
  EXPECT_TRUE(m.extensions.count("KHR_materials_specular"));
}

TEST(Materials, AMaterialPresetOverridesChannelsOfTheSurfacesItNames)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Face", channelsOf(kSkin)));
  scene.materials.push_back(fv::dson::buildMaterial("Torso", channelsOf(kSkin)));
  const Json preset = Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Face:?Glossy%20Roughness/value","keys":[[0,0.15]]},
    {"url":"name://@selection#materials/Face:?diffuse/image_file","keys":[[0,"/Runtime/Textures/other_d.jpg"]]},
    {"url":"name://@selection#materials/Hat:?diffuse/value","keys":[[0,[1,0,0]]]}
  ]}})");
  const auto result = fv::dson::applyMaterialPreset(preset, scene);
  EXPECT_EQ(result.changed, std::set<int>({0}));
  EXPECT_EQ(result.missing, std::set<std::string>({"Hat"}));
  EXPECT_EQ(scene.materials[0].surface.base.map, "/Runtime/Textures/other_d.jpg");
  EXPECT_EQ(scene.materials[1].surface.base.map, "/Runtime/Textures/face_d.jpg");
  // the shared channels of the untouched material are not changed through the copy
  EXPECT_FLOAT_EQ(scene.materials[1].surface.glossy.roughness.value, 0.4f);
}

TEST(Materials, ChangingTheBumpStrengthChangesTheBakedTexture)
{
  const auto     s = fv::dson::parseSurface("Face", channelsOf(kSkin));
  fv::mat::Edits e;
  e.bumpStrength = 1.5f;
  EXPECT_NE(fv::mat::requestKey(fv::mat::lower(s, {}, {}).requests[1]), fv::mat::requestKey(fv::mat::lower(s, e, {}).requests[1]));
  // a factor does not
  fv::mat::Edits r;
  r.roughness = 0.2f;
  EXPECT_EQ(fv::mat::lower(s, {}, {}).requests.size(), fv::mat::lower(s, r, {}).requests.size());
}

}  // namespace
