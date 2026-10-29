// A DSON surface's channels become a Surface (Iray Uber, PBRSkin), and a Surface becomes the glTF
// material the renderer shows: factors, the textures it asks the cooker for, the skin's volume, and
// edits laid over it. A port of the material pipeline in src/lib/materials and src/lib/rtx/gltf.ts.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

#include <figure/dson/eye_materials.hpp>
#include <figure/dson/legacy_materials.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/dson/materials.hpp>
#include <figure/dson/skin_materials.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/materials/edits_json.hpp>
#include <figure/materials/fabric.hpp>
#include <figure/materials/lobes.hpp>
#include <figure/materials/lower.hpp>

#include "tinygltf_utils.hpp"

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
    // the medium warms what scatters; the base colour, which the walk leaves through, is the one
    // the rasterizer shows, not tinted again
    const auto  plain = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
    const auto& base  = lowered.material["pbrMetallicRoughness"]["baseColorFactor"];
    for(int i = 0; i < 3; i++)
      EXPECT_FLOAT_EQ(base[i].get<float>(), plain.material["pbrMetallicRoughness"]["baseColorFactor"][i].get<float>());
  }
  {
    const auto lowered = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
    const auto& ext    = lowered.material["extensions"];
    EXPECT_FALSE(ext.contains("KHR_materials_transmission"));
    EXPECT_FALSE(ext.contains("KHR_materials_scatter"));
  }
}

// Iray's translucency at weight 1 sends all of the diffuse light in: no 0.95 cap left over from
// when the walk was entered by refraction (the renderer enters it through a diffuse lobe now)
TEST(Materials, FullTranslucencyEntersWhollyInEveryScatterMode)
{
  auto s = fv::dson::parseSurface("Face", channelsOf(kSkin));
  ASSERT_TRUE(s.translucency);
  s.translucency->weight = 1.0f;
  for(const auto mode : {fv::mat::SkinMode::eVolume, fv::mat::SkinMode::eThin})
  {
    const auto lowered = fv::mat::lower(s, {}, {.skin = mode});
    EXPECT_FLOAT_EQ(lowered.material["extensions"]["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1.0f);
  }
}

// Measured skin ignores the file's Iray medium: every bit of diffuse light enters, the medium is
// white for the renderer to make from the skin map, and light spreads as far as in human skin.
TEST(Materials, MeasuredSkinEntersWhollyAndSpreadsAsHumanSkin)
{
  const auto  s       = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto  lowered = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eMeasured});
  const auto& ext     = lowered.material["extensions"];
  EXPECT_FLOAT_EQ(ext["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1.0f);
  EXPECT_GT(ext["KHR_materials_volume"]["thicknessFactor"].get<float>(), 0.0f);
  EXPECT_FLOAT_EQ(ext["KHR_materials_scatter"]["scatterStrengthFactor"].get<float>(), 1.0f);
  EXPECT_FLOAT_EQ(ext["KHR_materials_scatter"]["scatterAnisotropy"].get<float>(), 0.0f);
  const auto& d = ext["FV_materials_subsurface"]["scatterDistance"];
  for(int i = 0; i < 3; i++)
  {
    EXPECT_FLOAT_EQ(ext["KHR_materials_scatter"]["multiscatterColorFactor"][i].get<float>(), 1.0f);
    EXPECT_FLOAT_EQ(d[i].get<float>(), fv::mat::kSkinScatterDistance[i]);
  }
  // red spreads furthest
  EXPECT_GT(d[0].get<float>(), d[1].get<float>());
  EXPECT_GT(d[1].get<float>(), d[2].get<float>());
  // the Iray volume has none of it
  EXPECT_FALSE(fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eVolume}).material["extensions"].contains("FV_materials_subsurface"));
  // an edit scales the distance
  fv::mat::Edits e;
  e.scatterScale = 0.5f;
  EXPECT_FALSE(e.empty());
  const auto  halved = fv::mat::lower(s, e, {.skin = fv::mat::SkinMode::eMeasured});
  const auto& hd     = halved.material["extensions"]["FV_materials_subsurface"]["scatterDistance"];
  for(int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(hd[i].get<float>(), 0.5f * fv::mat::kSkinScatterDistance[i]);
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

// Genesis 9's default eyes: "Eye Color" is a white fill, the sclera, and the iris (a disc with alpha)
// over it. Taking only the first file showed the eyes white.
TEST(Materials, KeepsEveryLayerOfALayeredImage)
{
  const Json doc = Json::parse(R"({"image_library": [
    {"id": "Eye Color-3", "map": [
      {"label": "Base", "color": [1, 1, 1]},
      {"url": "/Runtime/Textures/Eyes/Sclera_01.jpg", "label": "Sclera"},
      {"url": "/Runtime/Textures/Eyes/Old_Iris.png", "label": "Off", "active": false},
      {"url": "/Runtime/Textures/Eyes/Iris_01.png", "label": "Iris"}]},
    {"id": "Plain-1", "map": [{"url": "/Runtime/Textures/plain.jpg"}]}]})");
  std::unordered_map<std::string, std::string> images;
  fv::dson::collectImages(doc, images);
  EXPECT_EQ(images["Plain-1"], "/Runtime/Textures/plain.jpg");
  const std::string eye = images["Eye Color-3"];
  EXPECT_EQ(fv::mat::mapLayers(eye), (std::vector<std::string>{"/Runtime/Textures/Eyes/Sclera_01.jpg", "/Runtime/Textures/Eyes/Iris_01.png"}));

  // the request reads the bottom layer and lays the rest over it
  auto s     = fv::dson::parseSurface("Eye Left", channelsOf(kSkin));
  s.base.map = eye;
  const auto lowered = fv::mat::lower(s, {}, {});
  const auto base    = std::find_if(lowered.requests.begin(), lowered.requests.end(), [](const auto& r) { return !r.layers.empty(); });
  ASSERT_NE(base, lowered.requests.end());
  EXPECT_EQ(base->rel, "/Runtime/Textures/Eyes/Sclera_01.jpg");
  EXPECT_EQ(base->layers, (std::vector<std::string>{"/Runtime/Textures/Eyes/Iris_01.png"}));
  EXPECT_TRUE(base->composed());
  fv::mat::TextureRequest plain = *base;
  plain.layers.clear();
  EXPECT_NE(fv::mat::requestKey(*base), fv::mat::requestKey(plain));
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

// Genesis 9's eye as Addy Florence's eyes give it: a thin-walled moisture shell over the eye with
// Iray's glossy weight 0.40 and the iris's normal map, and the eye under it with a mapped dual lobe.
const char* kG9Moisture = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1]}},
  {"channel":{"id":"Refraction Weight","type":"float","current_value":1}},
  {"channel":{"id":"Refraction Index","type":"float","current_value":1.38}},
  {"channel":{"id":"Thin Walled","type":"bool","current_value":true}},
  {"channel":{"id":"Glossy Layered Weight","type":"float","current_value":0.4016393}},
  {"channel":{"id":"Glossy Reflectivity","type":"float","current_value":0.5}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0}},
  {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/eyes_nm.jpg"}}
]}]})";

const char* kG9Eye = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/eyes_d.jpg"}},
  {"channel":{"id":"Translucency Weight","type":"float","current_value":0.8}},
  {"channel":{"id":"Dual Lobe Specular Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Dual Lobe Specular Weight","type":"float","current_value":1,"image_file":"/Runtime/Textures/eyes_s.jpg"}},
  {"channel":{"id":"Dual Lobe Specular Reflectivity","type":"float","current_value":0.5}},
  {"channel":{"id":"Dual Lobe Specular Ratio","type":"float","current_value":0.15}},
  {"channel":{"id":"Specular Lobe 1 Roughness","type":"float","current_value":0.6}},
  {"channel":{"id":"Specular Lobe 2 Roughness Mult","type":"float","current_value":0.4}},
  {"channel":{"id":"Bump Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/eyes_nm.jpg"}}
]}]})";

// The tear film is the eye's one reflecting interface, at Fresnel's full strength and smooth: G9's
// glossy weight 0.40 left its catchlights 2.5 times too dim, and the iris's normal map on it rippled them
TEST(Materials, TheTearFilmReflectsAtFullFresnelFromASmoothSurface)
{
  const auto  s       = fv::dson::parseSurface("EyeMoisture Left", channelsOf(kG9Moisture));
  ASSERT_EQ(fv::mat::eyeShell(s), fv::mat::EyeShell::eMoisture);
  const auto  lowered = fv::mat::lower(s, {}, {.underTearFilm = true});
  const auto& m       = lowered.material;
  const auto& ext     = m["extensions"];
  EXPECT_NEAR(ext["KHR_materials_specular"]["specularFactor"].get<float>(), 1.0f, 1e-6f);
  EXPECT_NEAR(ext["KHR_materials_ior"]["ior"].get<float>(), 1.38f, 1e-6f);  // the authored, a watery eye's
  EXPECT_FALSE(ext.contains("KHR_materials_clearcoat"));
  EXPECT_FALSE(m.contains("normalTexture"));
  EXPECT_LE(m["pbrMetallicRoughness"]["roughnessFactor"].get<float>(), 0.02f);
  EXPECT_NEAR(ext["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1.0f, 1e-6f);
}

// What lies under the film reflects next to nothing: the iris and sclera lose their own lobes. The
// same eye with no film over it keeps them.
TEST(Materials, AnEyeUnderTheTearFilmHasNoSpecularOfItsOwn)
{
  const auto s     = fv::dson::parseSurface("Eye Left", channelsOf(kG9Eye));
  const auto under = fv::mat::lower(s, {}, {.underTearFilm = true}).material["extensions"];
  EXPECT_NEAR(under["KHR_materials_specular"]["specularFactor"].get<float>(), 0.0f, 1e-6f);
  EXPECT_FALSE(under["KHR_materials_specular"].contains("specularTexture"));
  EXPECT_FALSE(under.contains("KHR_materials_clearcoat"));
  // with no film over it, its dual lobe shines where its weight map says
  const auto bareLowered = fv::mat::lower(s, {}, {});
  const auto bare        = bareLowered.material["extensions"];
  EXPECT_NEAR(bare["KHR_materials_specular"]["specularFactor"].get<float>(), 0.85f, 1e-5f);
  ASSERT_TRUE(bare["KHR_materials_specular"].contains("specularTexture"));
  EXPECT_EQ(bareLowered.requests[bare["KHR_materials_specular"]["specularTexture"]["slot"].get<size_t>()].rel, "/Runtime/Textures/eyes_s.jpg");
  EXPECT_TRUE(bare.contains("KHR_materials_clearcoat"));
  EXPECT_TRUE(bare["KHR_materials_clearcoat"].contains("clearcoatTexture"));
  // a panel edit still wins
  fv::mat::Edits e;
  e.specular = 0.3f;
  EXPECT_NEAR(fv::mat::lower(s, e, {.underTearFilm = true}).material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(),
              0.3f, 1e-6f);
}

// Genesis 8's two shells: the moisture over the whole eye is the film, the cornea under it is
// index-matched. A cornea with no moisture over it is the film itself. An index that is not a
// watery eye's (glass at 1.5) becomes the cornea's.
TEST(Materials, OneReflectingShellPerEye)
{
  const auto cornea = fv::dson::parseSurface("Cornea", channelsOf(kG9Moisture));
  ASSERT_EQ(fv::mat::eyeShell(cornea), fv::mat::EyeShell::eCornea);
  EXPECT_NEAR(fv::mat::lower(cornea, {}, {.underTearFilm = true}).material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(),
              0.0f, 1e-6f);
  EXPECT_NEAR(fv::mat::lower(cornea, {}, {}).material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(), 1.0f, 1e-6f);

  auto glass = fv::dson::parseSurface("EyeMoisture", channelsOf(kG9Moisture));
  glass.refraction->ior = 1.5f;
  EXPECT_NEAR(fv::mat::lower(glass, {}, {}).material["extensions"]["KHR_materials_ior"]["ior"].get<float>(), 1.376f, 1e-6f);
}

// The lid margin's tear meniscus is the same water, but its normal map is its shape and stays
TEST(Materials, TheTearMeniscusKeepsItsShape)
{
  const auto s = fv::dson::parseSurface("Tear", channelsOf(kG9Moisture));
  ASSERT_EQ(fv::mat::eyeShell(s), fv::mat::EyeShell::eMeniscus);
  const auto lowered = fv::mat::lower(s, {}, {.underTearFilm = true});
  EXPECT_TRUE(lowered.material.contains("normalTexture"));
  EXPECT_NEAR(lowered.material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(), 1.0f, 1e-6f);
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

// Two worn items under a body group, each a figure with its own surfaces: "Eyelashes Upper" and
// "Eyelashes Lower" on the lashes, "Base" on the hair, and a "Base" of the body's own.
fv::SceneGraph wornScene()
{
  fv::SceneGraph scene;
  const auto     addFigure = [&](const std::string& dson, int parent, const std::vector<std::string>& surfaces) {
    fv::Mesh mesh;
    for(const std::string& s : surfaces)
    {
      mesh.materials.push_back(int(scene.materials.size()));
      scene.materials.push_back(fv::dson::buildMaterial(s, channelsOf(kSkin)));
    }
    scene.meshes.push_back(mesh);
    fv::Node group;
    group.dsonNode = dson;
    group.parent   = parent;
    scene.nodes.push_back(group);
    const int groupId = int(scene.nodes.size()) - 1;
    fv::Node shape;
    shape.parent = groupId;
    shape.mesh   = int(scene.meshes.size()) - 1;
    scene.nodes.push_back(shape);
    return groupId;
  };
  const int body = addFigure("Genesis9", -1, {"Body", "Base"});
  addFigure("Genesis9Eyelashes", body, {"Eyelashes Upper", "Eyelashes Lower"});
  addFigure("FEMessyHair", body, {"Base", "Transition"});
  return scene;
}

TEST(Materials, APresetSurfaceLandsUnderTheNodeItNames)
{
  const fv::SceneGraph scene = wornScene();
  // the hair's Base, not the body's, though the hair hangs under the body
  EXPECT_EQ(fv::dson::presetTargets(scene, "FEMessyHair", "Base"), std::vector<int>({4}));
  EXPECT_EQ(fv::dson::presetTargets(scene, "Genesis9", "Base"), std::vector<int>({1}));
  // a node the scene doesn't have: every surface of the name
  EXPECT_EQ(fv::dson::presetTargets(scene, "SomeOtherHair", "Base"), std::vector<int>({1, 4}));
}

TEST(Materials, APresetNodeChainNarrowsToTheLongestPartTheSceneHas)
{
  fv::SceneGraph scene = wornScene();
  // a prop beside the figure with a surface of the same name
  fv::Mesh chair;
  chair.materials.push_back(int(scene.materials.size()));
  scene.materials.push_back(fv::dson::buildMaterial("Base", channelsOf(kSkin)));
  scene.meshes.push_back(chair);
  fv::Node group;
  group.dsonNode = "Chair";
  scene.nodes.push_back(group);
  fv::Node shape;
  shape.parent = int(scene.nodes.size()) - 1;
  shape.mesh   = int(scene.meshes.size()) - 1;
  scene.nodes.push_back(shape);

  // the whole chain: the hair under the figure
  EXPECT_EQ(fv::dson::presetTargets(scene, std::vector<std::string>{"Genesis9", "FEMessyHair"}, "Base"), std::vector<int>({4}));
  // an outer node the scene lacks: the part it has, the hair
  EXPECT_EQ(fv::dson::presetTargets(scene, std::vector<std::string>{"Genesis8Female", "FEMessyHair"}, "Base"), std::vector<int>({4}));
  // an item the scene lacks under the figure: the surfaces under the figure, not the chair's
  EXPECT_EQ(fv::dson::presetTargets(scene, std::vector<std::string>{"Genesis9", "OtherHair"}, "Base"), std::vector<int>({1, 4}));
  // nothing known: every surface of the name
  EXPECT_EQ(fv::dson::presetTargets(scene, std::vector<std::string>{"Nobody"}, "Base"), std::vector<int>({1, 4, 6}));

  const Json preset = Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection/Genesis9:/FEMessyHair:#materials/Base:?Glossy%20Roughness/value","keys":[[0,0.15]]}
  ]}})");
  EXPECT_EQ(fv::dson::applyMaterialPreset(preset, scene).changed, std::set<int>({4}));
}

TEST(Materials, APresetForAnItemTheSceneLacksKeepsTheMapsItWouldRemove)
{
  // fiber brows' colour preset on card brows: the cards' cutout and normal maps are what draw them
  const auto apply = [](const char* node) {
    fv::SceneGraph scene = wornScene();
    const Json     preset = Json::parse(std::string(R"({"scene":{"animations":[
      {"url":")") + node + R"(#materials/Base:?extra/studio_material_channels/channels/Normal%20Map/image","keys":[[0,null]]},
      {"url":")" + node + R"(#materials/Base:?extra/studio_material_channels/channels/Glossy%20Roughness/value","keys":[[0,0.15]]}
    ]}})");
    fv::dson::applyMaterialPreset(preset, scene);
    return scene.materials[4].surface;
  };
  const auto stray = apply("G9EyebrowFibers");
  EXPECT_EQ(stray.normalMap, "/Runtime/Textures/face_n.jpg");
  EXPECT_FLOAT_EQ(stray.glossy.roughness.value, 0.15f);
  // its own item loses the map as the preset says
  const auto own = apply("FEMessyHair");
  EXPECT_TRUE(own.normalMap.empty());
  EXPECT_FLOAT_EQ(own.glossy.roughness.value, 0.15f);
}

TEST(Materials, APresetSurfaceFallsBackToTheSurfacesItWasSplitInto)
{
  const fv::SceneGraph scene = wornScene();
  EXPECT_EQ(fv::dson::presetTargets(scene, "Genesis9Eyelashes", "Eyelashes"), std::vector<int>({2, 3}));
  // "Eyelashes" is not a prefix of "EyelashesX": only whole words split off
  EXPECT_TRUE(fv::dson::presetTargets(scene, "Genesis9Eyelashes", "Eyelash").empty());
}

TEST(Materials, APresetSurfaceFallsBackToItsNameWithoutAProductPrefix)
{
  fv::SceneGraph scene = wornScene();
  EXPECT_EQ(fv::dson::presetTargets(scene, "FEMessyHair", "FE Transition"), std::vector<int>({5}));
  const Json preset = Json::parse(R"({"scene":{"animations":[
    {"url":"FEMessyHair#materials/FE%20Base:?Glossy%20Roughness/value","keys":[[0,0.15]]},
    {"url":"FEMessyHair#materials/FE%20Cap:?Glossy%20Roughness/value","keys":[[0,0.15]]}
  ]}})");
  const auto result = fv::dson::applyMaterialPreset(preset, scene);
  EXPECT_EQ(result.changed, std::set<int>({4}));
  EXPECT_EQ(result.matched, std::set<std::string>({"FE Base"}));
  EXPECT_EQ(result.missing, std::set<std::string>({"FE Cap"}));
  EXPECT_FLOAT_EQ(scene.materials[4].surface.glossy.roughness.value, 0.15f);
  EXPECT_FLOAT_EQ(scene.materials[1].surface.glossy.roughness.value, 0.4f);
}

// OmniHair at FE Messy High Ponytail's defaults: Custom melanin 0.5 at redness 0.5, undyed.
std::string omniHair(const std::string& more = "")
{
  return R"({"extra":[{"type":"studio/material/omnihair","channels":[
    {"channel":{"id":"Melanin Presets","type":"enum","current_value":7}},
    {"channel":{"id":"Melanin","type":"float","current_value":0.5}},
    {"channel":{"id":"Melanin Redness","type":"float","current_value":0.5}},
    {"channel":{"id":"Azimuthal Roughness","type":"float","current_value":0.2}})"
         + more + "]}]}";
}

TEST(Materials, OmniHairMelaninReadsAsTheBrownItIs)
{
  // the albedo measured for medium brown hair, not the near-black a lit ribbon showed
  const auto c = fv::dson::omniHairColor(channelsOf(omniHair().c_str()));
  ASSERT_TRUE(c.has_value());
  EXPECT_NEAR((*c)[0], 0.259f, 0.003f);
  EXPECT_NEAR((*c)[1], 0.159f, 0.003f);
  EXPECT_NEAR((*c)[2], 0.062f, 0.003f);
  // the darkest preset stays darker than that, and the lightest lighter
  const auto black = fv::dson::omniHairColor(channelsOf(omniHair().replace(omniHair().find("\"current_value\":7"), 17, "\"current_value\":6").c_str()));
  const auto white = fv::dson::omniHairColor(channelsOf(omniHair().replace(omniHair().find("\"current_value\":7"), 17, "\"current_value\":1").c_str()));
  EXPECT_LT((*black)[0], 0.1f);
  EXPECT_GT((*white)[2], 0.9f);
}

TEST(Materials, OmniHairDiffuseStrengthBlendsTheDyeIn)
{
  const std::string dye = R"(,{"channel":{"id":"Hair Root Color","type":"color","current_value":[0.8,0.1,0.1]}},
    {"channel":{"id":"Hair Tip Color","type":"color","current_value":[0.8,0.1,0.1]}})";
  const auto weight = [&](double w) {
    return *fv::dson::omniHairColor(channelsOf(
        omniHair(dye + R"(,{"channel":{"id":"Diffuse Strength","type":"float","current_value":)" + std::to_string(w) + "}}").c_str()));
  };
  const auto undyed = *fv::dson::omniHairColor(channelsOf(omniHair().c_str()));
  // none of the dye at 0, all of it at 1, and in between half way
  for(int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(weight(0)[i], undyed[i]);
  EXPECT_GT(weight(1)[0] / weight(1)[1], undyed[0] / undyed[1]);
  EXPECT_GT(weight(0.5)[0] / weight(0.5)[1], undyed[0] / undyed[1]);
  EXPECT_LT(weight(0.5)[0] / weight(0.5)[1], weight(1)[0] / weight(1)[1]);
}

TEST(Materials, RibbonHairLetsLightThroughAndReflectsAsAFibre)
{
  const auto s = fv::dson::parseSurface("FE Base", channelsOf(omniHair().c_str()));
  ASSERT_TRUE(s.twoSided);
  ASSERT_TRUE(s.hairFibre);
  // OmniHair's own roughness, not the glossy fallback a sheet would use
  EXPECT_NEAR(s.hairFibre->longitudinal, 0.2f, 1e-5f);
  EXPECT_NEAR(s.hairFibre->azimuthal, 0.2f, 1e-5f);
  // undyed: the root and the tip are the same fibre
  EXPECT_NEAR(s.hairFibre->rootSigma[1], s.hairFibre->tipSigma[1], 1e-5f);
  EXPECT_GT(s.hairFibre->rootSigma[2], s.hairFibre->rootSigma[0]);

  const auto  lowered = fv::mat::lower(s, {}, {});
  const auto& ext     = lowered.material["extensions"];
  EXPECT_EQ(lowered.material["alphaMode"].get<std::string>(), "OPAQUE");
  EXPECT_FALSE(ext.contains("KHR_materials_diffuse_transmission"));
  EXPECT_FALSE(ext.contains("KHR_materials_ior"));
  ASSERT_TRUE(ext.contains("FV_materials_hair"));
  const auto& hair = ext["FV_materials_hair"];
  EXPECT_NEAR(hair["longitudinalRoughness"].get<float>(), 0.2f, 1e-5f);
  EXPECT_NEAR(hair["azimuthalRoughness"].get<float>(), 0.2f, 1e-5f);
  EXPECT_NEAR(hair["rootFade"].get<float>(), 0.06f, 1e-5f);
  EXPECT_NEAR(hair["specularFactor"].get<float>(), 1.f, 1e-5f);
  EXPECT_NEAR(hair["rootAbsorption"][1].get<float>(), s.hairFibre->rootSigma[1], 1e-5f);
  EXPECT_NEAR(hair["tipAbsorption"][2].get<float>(), s.hairFibre->tipSigma[2], 1e-5f);

  // a file for other tools has no fibre BSDF, so it keeps the sheet that lets light through
  fv::mat::LowerOptions fileOpts;
  fileOpts.forFile = true;
  const auto  filed = fv::mat::lower(s, {}, fileOpts);
  const auto& fileExt = filed.material["extensions"];
  EXPECT_FALSE(fileExt.contains("FV_materials_hair"));
  EXPECT_NEAR(fileExt["KHR_materials_ior"]["ior"].get<float>(), 1.55f, 1e-6f);
  EXPECT_NEAR(fileExt["KHR_materials_specular"]["specularFactor"].get<float>(), 1.0f, 1e-6f);
  const auto& dt = fileExt["KHR_materials_diffuse_transmission"];
  EXPECT_NEAR(dt["diffuseTransmissionFactor"].get<float>(), 0.4f, 1e-6f);
  EXPECT_NEAR(dt["diffuseTransmissionColorFactor"][0].get<float>(), filed.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(), 1e-6f);

  // card hair is not a ribbon
  const auto cards = fv::mat::lower(fv::dson::parseSurface("Scalp Hair", channelsOf(kSkin)), {}, {});
  EXPECT_FALSE(cards.material["extensions"].contains("KHR_materials_diffuse_transmission"));
  EXPECT_FALSE(cards.material["extensions"].contains("FV_materials_hair"));
}

const char* kCardHair = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/hair_d.jpg"}},
  {"channel":{"id":"Cutout Opacity","type":"float","current_value":1,"image_file":"/Runtime/Textures/hair_a.jpg"}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.4}}
]}]})";

TEST(Materials, ABumpMapsHeightIsInMillimetresOfSurface)
{
  // a 2 m square (DSON centimetres) with UVs over half the unit square: 4 m per UV unit
  fv::SceneGraph scene;
  scene.materials.resize(2);
  scene.materials[1].metresPerUv = 0.5f;  // measured before: kept
  auto geo      = std::make_shared<fv::mesh::BuiltMesh>();
  geo->position = {0, 0, 0, 200, 0, 0, 200, 200, 0, 0, 200, 0};
  geo->uv       = {0, 0, 0.5f, 0, 0.5f, 0.5f, 0, 0.5f};
  geo->fullIndex = {0, 1, 2, 0, 2, 3, 0, 1, 2};
  geo->groups    = {{0, 6, 0}, {6, 3, 1}};
  fv::Mesh m;
  m.geo       = geo;
  m.materials = {0, 1};
  scene.meshes.push_back(m);
  fv::measureUvScales(scene);
  EXPECT_NEAR(scene.materials[0].metresPerUv, 4.f, 1e-5f);
  EXPECT_NEAR(scene.materials[1].metresPerUv, 0.5f, 1e-6f);

  // the full range is kBumpHeight of surface: on a 4K map of a face's 0.43 m per UV unit, a slope
  // of about 2.4 per texel's full-range step, within the rounding
  EXPECT_NEAR(fv::mat::bumpHeightInUv(0.43f) * 4096, 2.4f, 0.1f);
  EXPECT_EQ(fv::mat::bumpHeightInUv(0), 0.f);
  // a detail tiled 4 times covers a quarter of the surface per UV unit (each to two figures)
  EXPECT_NEAR(fv::mat::bumpHeightInUv(0.4f, 4), 4 * fv::mat::bumpHeightInUv(0.4f), 0.02f * fv::mat::bumpHeightInUv(0.4f, 4));

  // the requests carry it, alone and blended onto a normal map, and the key tells them apart
  fv::mat::Surface bumped;
  bumped.bumpMap = "/Runtime/Textures/skin_b.jpg";
  fv::mat::LowerOptions o;
  o.metresPerUv = 0.43f;
  const auto alone = fv::mat::lower(bumped, {}, o).requests;
  ASSERT_EQ(alone.size(), 1u);
  EXPECT_EQ(alone[0].channels, "bn");
  EXPECT_NEAR(alone[0].bumpHeight, fv::mat::bumpHeightInUv(0.43f), 1e-7f);
  EXPECT_NE(fv::mat::requestKey(alone[0]), fv::mat::requestKey(fv::mat::lower(bumped, {}, {}).requests[0]));
  bumped.normalMap   = "/Runtime/Textures/skin_n.jpg";
  const auto blended = fv::mat::lower(bumped, {}, o).requests;
  ASSERT_EQ(blended.size(), 1u);
  EXPECT_EQ(blended[0].bumpRel, bumped.bumpMap);
  EXPECT_NEAR(blended[0].bumpHeight, fv::mat::bumpHeightInUv(0.43f), 1e-7f);
}

TEST(Materials, HairCardsAreFibresInThePathTracer)
{
  const auto s = fv::dson::parseSurface("Hair Base", channelsOf(kCardHair));
  ASSERT_TRUE(fv::mat::hairCards(s));
  fv::mat::LowerOptions o;
  o.anisotropyRotation = 0.3f;
  const auto  lowered = fv::mat::lower(s, {}, o);
  const auto& ext     = lowered.material["extensions"];
  ASSERT_TRUE(ext.contains("FV_materials_hair"));
  const auto& hair = ext["FV_materials_hair"];
  // the absorption is the colour each hit shows, the strand across the measured rotation
  EXPECT_TRUE(hair["absorptionFromColor"].get<bool>());
  EXPECT_NEAR(hair["strandRotation"].get<float>(), 0.3f, 1e-6f);
  const float roughness = lowered.material["pbrMetallicRoughness"]["roughnessFactor"].get<float>();
  EXPECT_NEAR(hair["longitudinalRoughness"].get<float>(), fv::mat::cardLongitudinalRoughness(roughness), 1e-6f);
  // the fibre's highlight as wide along the strand as the GGX lobe: glossy 0.4 is beta_m 0.2, and
  // its spread is GGX's half-maximum width as a Gaussian, at every roughness
  EXPECT_NEAR(fv::mat::cardLongitudinalRoughness(0.4f), 0.2f, 0.01f);
  for(float r : {0.1f, 0.3f, 0.5f, 0.8f})
  {
    const double bm = fv::mat::cardLongitudinalRoughness(r);
    EXPECT_NEAR(0.726 * bm + 0.812 * bm * bm + 3.7 * std::pow(bm, 20), 2.57 * r * r / 2.355, 1e-4) << r;
    EXPECT_LT(bm, r) << r;
  }
  EXPECT_NEAR(hair["azimuthalRoughness"].get<float>(), fv::mat::kCardAzimuthalRoughness, 1e-6f);
  EXPECT_NEAR(hair["specularFactor"].get<float>(), 1.f, 1e-6f);
  EXPECT_FALSE(hair.contains("rootFade"));
  // cards whose strands did not agree on a direction run along UV v
  EXPECT_FALSE(fv::mat::lower(s, {}, {}).material["extensions"]["FV_materials_hair"].contains("strandRotation"));

  // the real-time view and a file for other tools keep the GGX sheet
  fv::mat::LowerOptions raster;
  raster.skin = fv::mat::SkinMode::eRaster;
  EXPECT_FALSE(fv::mat::lower(s, {}, raster).material["extensions"].contains("FV_materials_hair"));
  fv::mat::LowerOptions file;
  file.forFile = true;
  EXPECT_FALSE(fv::mat::lower(s, {}, file).material["extensions"].contains("FV_materials_hair"));

  // a scalp cap and brows are hair by name, painted on the skin; OmniHair ribbons have a fibre of their own
  EXPECT_FALSE(fv::mat::hairCards(fv::dson::parseSurface("Hair Cap", channelsOf(kCardHair))));
  EXPECT_FALSE(fv::mat::hairCards(fv::dson::parseSurface("Eyebrows", channelsOf(kCardHair))));
  EXPECT_FALSE(fv::mat::hairCards(fv::dson::parseSurface("FE Base", channelsOf(omniHair().c_str()))));
  // hair without a cutout is no card
  EXPECT_FALSE(fv::mat::hairCards(fv::dson::parseSurface("Hair Base", channelsOf(kSkin))));
}

TEST(Materials, RibbonHairKeepsRootAndTipAbsorptionApart)
{
  const std::string dye = R"(,{"channel":{"id":"Hair Root Color","type":"color","current_value":[0.05,0.02,0.01]}},
    {"channel":{"id":"Hair Tip Color","type":"color","current_value":[0.8,0.6,0.3]}})";
  const auto s = fv::dson::parseSurface("FE Base", channelsOf(omniHair(dye).c_str()));
  ASSERT_TRUE(s.hairFibre);
  // a darker root absorbs more, in every channel
  for(int i = 0; i < 3; i++)
    EXPECT_GT(s.hairFibre->rootSigma[i], s.hairFibre->tipSigma[i]);
  // the sheet colour stays the average of the two dyes, which is what the melanin fit returns
  const auto averaged = fv::dson::omniHairColor(channelsOf(omniHair(dye).c_str()));
  ASSERT_TRUE(averaged);
  EXPECT_NEAR(s.base.color[0], (*averaged)[0], 1e-5f);
}

TEST(Materials, RibbonHairEditsMoveTheFibreAndCanStayOpaque)
{
  const auto s = fv::dson::parseSurface("FE Base", channelsOf(omniHair().c_str()));
  fv::mat::Edits e;
  e.roughness = 0.5f;
  e.specular  = 0.25f;
  e.alphaMode = "opaque";
  const auto  lowered = fv::mat::lower(s, e, {});
  const auto& hair    = lowered.material["extensions"]["FV_materials_hair"];
  EXPECT_EQ(lowered.material["alphaMode"].get<std::string>(), "OPAQUE");
  EXPECT_NEAR(hair["longitudinalRoughness"].get<float>(), 0.5f, 1e-5f);
  // azimuthal started equal to longitudinal, so it keeps that ratio
  EXPECT_NEAR(hair["azimuthalRoughness"].get<float>(), 0.5f, 1e-5f);
  EXPECT_NEAR(hair["specularFactor"].get<float>(), 0.25f, 1e-5f);
  EXPECT_NEAR(lowered.effective.roughness, 0.5f, 1e-5f);
  EXPECT_NEAR(lowered.effective.specular, 0.25f, 1e-5f);
}

// A fibre hair's tint reaches the path tracer through its absorption: the mean of root and tip goes
// to the target and the root stays darker than the tip.
TEST(Materials, ARecolourMovesAFibresAbsorptionAndKeepsRootAndTipApart)
{
  const std::string dye = R"(,{"channel":{"id":"Hair Root Color","type":"color","current_value":[0.05,0.02,0.01]}},
    {"channel":{"id":"Hair Tip Color","type":"color","current_value":[0.3,0.2,0.1]}})";
  const auto s = fv::dson::parseSurface("FE Base", channelsOf(omniHair(dye).c_str()));
  ASSERT_TRUE(s.hairFibre);
  fv::mat::Edits e;
  e.recolor                = fv::mat::Rgb{0.6f, 0.45f, 0.25f};
  const auto   lowered     = fv::mat::lower(s, e, {});
  const auto&  hair        = lowered.material["extensions"]["FV_materials_hair"];
  const float  b           = hair["azimuthalRoughness"].get<float>();
  const float  fit         = 5.969f - 0.215f * b + 2.532f * b * b - 10.73f * b * b * b + 5.574f * b * b * b * b + 0.245f * b * b * b * b * b;
  const auto   colour      = [&](float sigma) { return std::exp(-0.45f * std::sqrt(sigma) * fit); };
  for(int i = 0; i < 3; i++)
  {
    const float root = colour(hair["rootAbsorption"][i].get<float>()), tip = colour(hair["tipAbsorption"][i].get<float>());
    EXPECT_NEAR(0.5f * (root + tip), (*e.recolor)[i], 0.02f) << i;
    EXPECT_LT(root, tip) << i;
  }
  // none of it: the fibre as it was
  e.recolorStrength = 0.f;
  EXPECT_NEAR(fv::mat::lower(s, e, {}).material["extensions"]["FV_materials_hair"]["rootAbsorption"][0].get<float>(), s.hairFibre->rootSigma[0], 1e-3f);
}

// The orders of one fibre at its centre, the part of hair_bsdf.h.slang that decides whether light
// reflects or goes through. h = 0, the ray square to the strand, eta 1.55.
TEST(Materials, AFibreTransmitsUntilAbsorptionClosesIt)
{
  const auto fresnel = [](double cosI, double eta) {
    const double sin2T = (1.0 / (eta * eta)) * (1.0 - cosI * cosI);
    const double cosT  = std::sqrt(1.0 - sin2T);
    const double rPar  = (eta * cosI - cosT) / (eta * cosI + cosT);
    const double rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5 * (rPar * rPar + rPerp * rPerp);
  };
  const double F = fresnel(1.0, 1.55);
  // nothing to absorb: the transmitted order carries the fibre, the primary is only the Fresnel
  const double openTT = (1.0 - F) * (1.0 - F);
  EXPECT_GT(openTT, F * 5.0);
  // a dark fibre: one trip through kills the transmitted order, the primary Fresnel remains
  const double absorbed = std::exp(-8.0 * 2.0);
  EXPECT_LT(openTT * absorbed, F * 0.1);
}

// Longitudinal M_p peaks when the incoming angle mirrors the tilted outgoing one (pbrt's Mp:
// exp(-sinI * sinOp / v) is largest when those sines have opposite sign).
TEST(Materials, ThePrimaryLobeFacesTheMirrorAndTheTransmittedOneDoesNot)
{
  const auto mp = [](double sinI, double sinO) {
    constexpr double v = 0.05;
    const double     b = sinI * sinO / v;
    return std::exp(-b);
  };
  // square to the strand, primary tilt of about -4 degrees (two cuticle angles)
  const double sinOp = -std::sin(4.0 * 3.14159265358979323846 / 180.0);
  const double mirror = mp(-sinOp, sinOp);
  const double through = mp(0.0, sinOp);
  EXPECT_GT(mirror, through);
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

// ---- the character's own channels, and eyes chosen over them

TEST(Materials, TheBaselineIsTheChannelsAsBuiltAndPresetsLeaveIt)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Face", channelsOf(kSkin)));
  EXPECT_EQ(scene.materials[0].baseline, scene.materials[0].channels);
  const Json preset = Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Face:?diffuse/image_file","keys":[[0,"/Runtime/Textures/other_d.jpg"]]}
  ]}})");
  fv::dson::applyMaterialPreset(preset, scene);
  ASSERT_EQ(scene.materials[0].surface.base.map, "/Runtime/Textures/other_d.jpg");
  EXPECT_NE(scene.materials[0].baseline, scene.materials[0].channels);
  // back as it was built; only the materials that differed count
  EXPECT_EQ(fv::dson::restoreBaseline(scene, {0}), std::set<int>({0}));
  EXPECT_EQ(scene.materials[0].surface.base.map, "/Runtime/Textures/face_d.jpg");
  EXPECT_TRUE(fv::dson::restoreBaseline(scene, {0, 7}).empty());
}

TEST(Materials, KeepBaselineTakesTheAddOnPresetsIn)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Face", channelsOf(kSkin)));
  // the preset an add-on is worn with is the character's own; one opened later is not
  fv::dson::applyMaterialPreset(Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Face:?diffuse/image_file","keys":[[0,"/Runtime/Textures/own_d.jpg"]]}
  ]}})"),
                                scene);
  fv::dson::keepBaseline(scene);
  fv::dson::applyMaterialPreset(Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Face:?Glossy%20Roughness/value","keys":[[0,0.15]]}
  ]}})"),
                                scene);
  EXPECT_EQ(fv::dson::restoreBaseline(scene, {0}), std::set<int>({0}));
  EXPECT_EQ(scene.materials[0].surface.base.map, "/Runtime/Textures/own_d.jpg");
  EXPECT_FLOAT_EQ(scene.materials[0].surface.glossy.roughness.value, 0.4f);
}

const char* kEye = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/eyes_d.jpg"}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.3}}
]}]})";

// A Genesis 9 body as its add-ons leave it: its own surfaces, the eyes, the tear film and the lashes,
// each a figure under the body's group, and a pair of eyes taken off.
fv::SceneGraph eyeScene()
{
  fv::SceneGraph scene;
  const auto     addFigure = [&](const std::string& dson, int parent, const std::vector<std::string>& surfaces, const char* channels) {
    fv::Mesh mesh;
    for(const std::string& s : surfaces)
    {
      mesh.materials.push_back(int(scene.materials.size()));
      scene.materials.push_back(fv::dson::buildMaterial(s, channelsOf(channels)));
    }
    scene.meshes.push_back(mesh);
    fv::Node group;
    group.dsonNode = dson;
    group.parent   = parent;
    scene.nodes.push_back(group);
    const int groupId = int(scene.nodes.size()) - 1;
    fv::Node  shape;
    shape.parent = groupId;
    shape.mesh   = int(scene.meshes.size()) - 1;
    scene.nodes.push_back(shape);
    return groupId;
  };
  const int body = addFigure("Genesis9", -1, {"Body", "Head"}, kSkin);                      // 0, 1
  addFigure("Genesis9Eyes", body, {"Eye Left", "Eye Right"}, kEye);                          // 2, 3
  addFigure("Genesis9Eyes", body, {"EyeMoisture Left", "EyeMoisture Right"}, kEyeMoisture);  // 4, 5
  addFigure("Genesis9Tear", body, {"Tear"}, kEyeMoisture);                                   // 6
  addFigure("Genesis9Eyelashes", body, {"Eyelashes"}, kSkin);                                // 7
  const int gone = addFigure("OldEyes", body, {"Eye Left", "Eye Right"}, kEye);              // 8, 9
  scene.nodes[size_t(gone)].removed = true;
  return scene;
}

// A G9 eye preset, more or less Addy Florence Eyes Blue: the eyes' map, and maybe their roughness.
Json eyePreset(const std::string& map, std::optional<double> roughness = std::nullopt, const std::string& more = "")
{
  const std::string eyes = "name://@selection/Genesis9:/Genesis9Eyes:#materials/";
  std::string       anims = R"({"url":")" + eyes + R"(Eye%20Left:?diffuse/image_file","keys":[[0,")" + map + R"("]]},)"
                      + R"({"url":")" + eyes + R"(Eye%20Right:?diffuse/image_file","keys":[[0,")" + map + R"("]]})";
  if(roughness)
    anims += R"(,{"url":")" + eyes + R"(Eye%20Left:?Glossy%20Roughness/value","keys":[[0,)" + std::to_string(*roughness) + "]]}";
  return Json::parse(R"({"scene":{"animations":[)" + anims + more + "]}}");
}

TEST(Materials, TheEyeSurfacesAreTheEyesAndCorneasButTheTearFilm)
{
  const fv::SceneGraph scene = eyeScene();
  // the eyes taken off are not the figure's any more
  EXPECT_EQ(fv::dson::eyeSurfaces(scene), std::set<int>({2, 3, 4, 5}));
  // Genesis 8 has them as surfaces of the body
  for(const char* name : {"Irises", "Pupils", "Sclera", "Cornea", "EyeMoisture"})
    EXPECT_TRUE(fv::dson::isEyeSurface(fv::dson::buildMaterial(name, channelsOf(kEye)))) << name;
  for(const char* name : {"Tear", "Eyelashes", "EyeSocket", "Face"})
    EXPECT_FALSE(fv::dson::isEyeSurface(fv::dson::buildMaterial(name, channelsOf(kEye)))) << name;
}

TEST(Materials, ChoosingEyesDoesNotStack)
{
  fv::SceneGraph scene = eyeScene();
  const Json     a     = eyePreset("/a.jpg", 0.1);
  const auto     ra    = fv::dson::chooseEyes(scene, &a);
  EXPECT_EQ(ra.owned, std::set<int>({2, 3}));
  EXPECT_EQ(scene.materials[2].surface.base.map, "/a.jpg");
  EXPECT_NEAR(scene.materials[2].surface.glossy.roughness.value, 0.1f, 1e-5f);
  // B sets only the map: A's roughness does not stay under it
  const Json b  = eyePreset("/b.jpg");
  const auto rb = fv::dson::chooseEyes(scene, &b, ra.owned);
  EXPECT_EQ(scene.materials[2].surface.base.map, "/b.jpg");
  EXPECT_EQ(scene.materials[3].surface.base.map, "/b.jpg");
  EXPECT_FLOAT_EQ(scene.materials[2].surface.glossy.roughness.value, 0.3f);
  EXPECT_TRUE(rb.changed.count(2));
  // the character's own: as built, the edits left alone
  scene.materials[2].edits.irisColor = fv::mat::Rgb{0, 0, 1};
  const auto own                     = fv::dson::chooseEyes(scene, nullptr, rb.owned);
  EXPECT_EQ(own.changed, std::set<int>({2, 3}));
  EXPECT_TRUE(own.owned.empty());
  EXPECT_EQ(scene.materials[2].surface.base.map, "/Runtime/Textures/eyes_d.jpg");
  EXPECT_TRUE(scene.materials[2].edits.irisColor.has_value());
}

TEST(Materials, TheCharactersOwnEyesAreTheOnesItsAddOnPresetGave)
{
  fv::SceneGraph scene = eyeScene();
  // the eyes as the character's add-on preset left them, kept when the add-on was worn
  fv::dson::applyMaterialPreset(eyePreset("/own.jpg"), scene);
  fv::dson::keepBaseline(scene);
  const Json blue = eyePreset("/blue.jpg");
  const auto r    = fv::dson::chooseEyes(scene, &blue);
  ASSERT_EQ(scene.materials[2].surface.base.map, "/blue.jpg");
  fv::dson::chooseEyes(scene, nullptr, r.owned);
  EXPECT_EQ(scene.materials[2].surface.base.map, "/own.jpg");
  EXPECT_EQ(scene.materials[3].surface.base.map, "/own.jpg");
}

TEST(Materials, AnEyePresetIsOneThatOnlyTouchesEyes)
{
  const fv::SceneGraph scene = eyeScene();
  EXPECT_TRUE(fv::dson::isEyePreset(eyePreset("/a.jpg"), scene));
  // the head too: a character's material preset, not a pair of eyes
  const std::string head = R"(,{"url":"name://@selection/Genesis9:#materials/Head:?diffuse/image_file","keys":[[0,"/head.jpg"]]})";
  EXPECT_FALSE(fv::dson::isEyePreset(eyePreset("/a.jpg", std::nullopt, head), scene));
  EXPECT_EQ(fv::dson::materialPresetTargets(eyePreset("/a.jpg", std::nullopt, head), scene), std::set<int>({1, 2, 3}));
  // nothing it names is here
  EXPECT_FALSE(fv::dson::isEyePreset(Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Irises:?diffuse/image_file","keys":[[0,"/g8.jpg"]]}
  ]}})"),
                                     scene));
}

TEST(Materials, AMixedPresetGivesTheEyesOnlyItsEyePart)
{
  fv::SceneGraph    scene = eyeScene();
  const std::string more  = R"(,{"url":"name://@selection/Genesis9:#materials/Head:?diffuse/image_file","keys":[[0,"/head.jpg"]]},
    {"url":"name://@selection/Genesis9:/Genesis9Eyelashes:#materials/Eyelashes:?Glossy%20Roughness/value","keys":[[0,0.9]]})";
  const Json        mixed = eyePreset("/a.jpg", std::nullopt, more);
  const auto        r     = fv::dson::chooseEyes(scene, &mixed);
  EXPECT_EQ(r.owned, std::set<int>({2, 3}));
  EXPECT_EQ(r.preset.changed, std::set<int>({2, 3}));
  EXPECT_EQ(scene.materials[2].surface.base.map, "/a.jpg");
  // the head and the lashes keep what was chosen for them
  EXPECT_EQ(scene.materials[1].surface.base.map, "/Runtime/Textures/face_d.jpg");
  EXPECT_FLOAT_EQ(scene.materials[7].surface.glossy.roughness.value, 0.4f);
}

// ---- iris and sclera colours

TEST(Materials, EyeLayoutsFollowTheSurfaceNames)
{
  using fv::mat::EyeLayout;
  const auto layout = [](const char* name) { return fv::mat::eyeLayout(fv::dson::parseSurface(name, channelsOf(kEye))); };
  EXPECT_EQ(layout("Eye Left"), EyeLayout::eShared);
  EXPECT_EQ(layout("Eye Right"), EyeLayout::eShared);
  EXPECT_EQ(layout("Irises"), EyeLayout::eIris);
  EXPECT_EQ(layout("Sclera"), EyeLayout::eSclera);
  EXPECT_EQ(layout("Pupils"), EyeLayout::eNone);
  EXPECT_EQ(layout("EyeMoisture Left"), EyeLayout::eNone);
  EXPECT_EQ(layout("Face"), EyeLayout::eNone);
}

TEST(Materials, AnEyesWhiteSpreadsItsLightUnderMeasuredSkinWithNoWalk)
{
  const auto  eye     = fv::dson::parseSurface("Eye Left", channelsOf(kEye));
  const auto  lowered = fv::mat::lower(eye, {}, {.skin = fv::mat::SkinMode::eMeasured});
  const auto& ext     = lowered.material["extensions"];
  ASSERT_TRUE(ext.contains("FV_materials_subsurface"));
  for(int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(ext["FV_materials_subsurface"]["scatterDistance"][i].get<float>(), fv::mat::kScleraScatterDistance[i]);
  // no medium: the eyeball is no closed volume
  EXPECT_FALSE(ext.contains("KHR_materials_volume"));
  EXPECT_FALSE(ext.contains("KHR_materials_transmission"));
  // the Iray volume leaves the eye as it was
  EXPECT_FALSE(fv::mat::lower(eye, {}, {.skin = fv::mat::SkinMode::eVolume}).material["extensions"].contains("FV_materials_subsurface"));
}

TEST(Materials, AnIrisColourAsksForARecolouredBaseMap)
{
  const auto     eye = fv::dson::parseSurface("Eye Left", channelsOf(kEye));
  fv::mat::Edits e;
  e.irisColor        = fv::mat::Rgb{0.2f, 0.4f, 0.6f};
  e.irisStrength     = 0.5f;
  const auto lowered = fv::mat::lower(eye, e, {});
  ASSERT_EQ(lowered.requests.size(), 1u);
  const fv::mat::TextureRequest& base = lowered.requests[0];
  ASSERT_TRUE(base.iris);
  EXPECT_TRUE(base.composed());
  EXPECT_EQ(base.rel, "/Runtime/Textures/eyes_d.jpg");
  EXPECT_EQ(base.iris->mode, fv::mat::IrisRequest::Mode::eEye);
  ASSERT_EQ(base.iris->discs.size(), 2u);
  // the cooker's ir= (texcache.rs IrisRecolour::parse), before the host percent-encodes it whole
  EXPECT_EQ(fv::mat::irisQuery(*base.iris), "0.2,0.4,0.6|0.5|eye|0.249,0.249,0.108;0.749,0.249,0.108|-");
  EXPECT_NE(fv::mat::requestKey(base).find("|ir:0.2,0.4,0.6|0.5|eye|"), std::string::npos);
  // the base colour stays a live factor over the whole eye
  EXPECT_FLOAT_EQ(lowered.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(), 1.f);

  // each colour and strength is a texture of its own; without one, the eye's own map
  const auto keyOf = [&](const fv::mat::Edits& edits) { return fv::mat::requestKey(fv::mat::lower(eye, edits, {}).requests[0]); };
  fv::mat::Edits stronger = e;
  stronger.irisStrength   = 1.f;
  fv::mat::Edits other    = e;
  other.irisColor         = fv::mat::Rgb{0.6f, 0.4f, 0.2f};
  EXPECT_NE(keyOf(e), keyOf(stronger));
  EXPECT_NE(keyOf(e), keyOf(other));
  EXPECT_FALSE(fv::mat::lower(eye, {}, {}).requests[0].iris);
  // a strength alone changes nothing
  fv::mat::Edits strengthOnly;
  strengthOnly.irisStrength = 0.3f;
  EXPECT_FALSE(fv::mat::lower(eye, strengthOnly, {}).requests[0].iris);

  // the white of a Genesis 9 eye is in the same map: baked outside the irises
  fv::mat::Edits white;
  white.scleraColor = fv::mat::Rgb{1.f, 0.9f, 0.8f};
  const auto sclera = fv::mat::lower(eye, white, {}).requests[0];
  ASSERT_TRUE(sclera.iris);
  EXPECT_EQ(fv::mat::irisQuery(*sclera.iris), "-|1|eye|0.249,0.249,0.108;0.749,0.249,0.108|1,0.9,0.8");
}

TEST(Materials, AGenesis8IrisNeedsItsDiscsAndItsScleraIsAFactor)
{
  const auto     irises = fv::dson::parseSurface("Irises", channelsOf(kEye));
  fv::mat::Edits e;
  e.irisColor = fv::mat::Rgb{0.1f, 0.3f, 0.1f};
  // where the irises are in the map isn't known: no bake (material.set refuses the colour)
  EXPECT_FALSE(fv::mat::lower(irises, e, {}).requests[0].iris);
  fv::mat::LowerOptions o;
  o.irisDiscs        = {{0.25f, 0.75f, 0.1f}, {0.75f, 0.75f, 0.1f}};
  const auto lowered = fv::mat::lower(irises, e, o);
  ASSERT_TRUE(lowered.requests[0].iris);
  EXPECT_EQ(fv::mat::irisQuery(*lowered.requests[0].iris), "0.1,0.3,0.1|1|iris|0.25,0.75,0.1;0.75,0.75,0.1|-");

  // a sclera of its own multiplies its factor, live
  const auto     sclera = fv::dson::parseSurface("Sclera", channelsOf(kEye));
  fv::mat::Edits white;
  white.color       = fv::mat::Rgb{0.5f, 1.f, 1.f};
  white.scleraColor = fv::mat::Rgb{1.f, 0.8f, 0.8f};
  const auto plain  = fv::mat::lower(sclera, white, {});
  EXPECT_FALSE(plain.requests[0].iris);
  EXPECT_NEAR(plain.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(), 0.5f, 1e-6f);
  EXPECT_NEAR(plain.material["pbrMetallicRoughness"]["baseColorFactor"][1].get<float>(), 0.8f, 1e-6f);

  // a flat iris takes the colour as its factor
  const auto flat = fv::dson::parseSurface("Irises", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[0.5,0.5,0.5]}}]}]})"));
  const auto f    = fv::mat::lower(flat, e, {});
  EXPECT_TRUE(f.requests.empty());
  EXPECT_NEAR(f.material["pbrMetallicRoughness"]["baseColorFactor"][1].get<float>(), 0.3f, 1e-6f);
}

TEST(Materials, EyeColourEditsAreEditsLikeTheRest)
{
  fv::mat::Edits e;
  EXPECT_TRUE(e.empty());
  fv::mat::Edits o;
  o.irisStrength = 0.4f;
  e.merge(o);
  EXPECT_FALSE(e.empty());
  EXPECT_FLOAT_EQ(*e.irisStrength, 0.4f);
  o             = {};
  o.scleraColor = fv::mat::Rgb{1, 1, 1};
  o.irisColor   = fv::mat::Rgb{0, 0, 1};
  e.merge(o);
  EXPECT_TRUE(e.scleraColor && e.irisColor);
  EXPECT_FLOAT_EQ(*e.irisStrength, 0.4f);
}

// A mesh whose material 0 ("Irises") is two square UV islands, the second a tile to the right, and
// whose material 1 is a third.
fv::SceneGraph irisMesh()
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Irises", channelsOf(kEye)));
  scene.materials.push_back(fv::dson::buildMaterial("Sclera", channelsOf(kEye)));
  auto       geo  = std::make_shared<fv::mesh::BuiltMesh>();
  const auto quad = [&](float u0, float v0, float size) {
    const uint32_t first = uint32_t(geo->uv.size() / 2);
    for(const auto& [du, dv] : std::vector<std::pair<float, float>>{{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}})
      geo->uv.insert(geo->uv.end(), {u0 + du * size, v0 + dv * size});
    geo->fullIndex.insert(geo->fullIndex.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
  };
  quad(0.2f, 0.7f, 0.1f);  // image (0.25, 0.25)
  quad(1.7f, 0.7f, 0.1f);  // image (0.75, 0.25), a tile over
  geo->groups.push_back({0, 12, 0});
  quad(0.5f, 0.1f, 0.2f);
  geo->groups.push_back({12, 6, 1});
  fv::Mesh mesh;
  mesh.geo       = geo;
  mesh.materials = {0, 1};
  scene.meshes.push_back(mesh);
  return scene;
}

TEST(Materials, AnIrisSurfacesDiscsAreItsUVIslands)
{
  fv::SceneGraph scene = irisMesh();
  auto           discs = fv::build::irisDiscs(scene, 0);
  ASSERT_EQ(discs.size(), 2u);
  std::sort(discs.begin(), discs.end());
  EXPECT_NEAR(discs[0][0], 0.25f, 1e-5f);
  EXPECT_NEAR(discs[0][1], 0.25f, 1e-5f);
  EXPECT_NEAR(discs[0][2], 0.05f * std::sqrt(2.f), 1e-5f);
  EXPECT_NEAR(discs[1][0], 0.75f, 1e-5f);
  EXPECT_NEAR(discs[1][1], 0.25f, 1e-5f);
  EXPECT_EQ(fv::build::irisDiscs(scene, 1).size(), 1u);
  // an island of the same eye that overlaps it makes one disc around both
  fv::mesh::BuiltMesh& geo   = *scene.meshes[0].geo;
  const uint32_t       first = uint32_t(geo.uv.size() / 2);
  geo.uv.insert(geo.uv.end(), {0.22f, 0.72f, 0.32f, 0.72f, 0.32f, 0.82f});
  geo.fullIndex.insert(geo.fullIndex.end(), {first, first + 1, first + 2});
  geo.groups.push_back({18, 3, 0});
  discs = fv::build::irisDiscs(scene, 0);
  ASSERT_EQ(discs.size(), 2u);
  std::sort(discs.begin(), discs.end());
  EXPECT_GT(discs[0][2], 0.05f * std::sqrt(2.f));
  // the textures a load resolves ahead are the ones the build lowers with its discs
  scene.materials[0].edits.irisColor = fv::mat::Rgb{0, 0, 1};
  int asked                          = 0;
  for(const fv::mat::TextureRequest& r : fv::build::textureRequests(scene))
    if(r.iris)
    {
      asked++;
      EXPECT_EQ(r.iris->mode, fv::mat::IrisRequest::Mode::eIris);
      EXPECT_EQ(r.iris->discs.size(), 2u);
    }
  EXPECT_EQ(asked, 1);
}

// A worn item's colour: only its surfaces take the preset, each choice starts from how it was worn,
// and null puts it back. The lashes stand in for a hair here.
TEST(Materials, ChoosingAnItemsColourTouchesItsSurfacesAloneAndDoesNotStack)
{
  fv::SceneGraph      scene  = eyeScene();
  const std::set<int> lashes = {7};
  const auto          preset = [](const std::string& map, const std::string& more = "") {
    return Json::parse(R"({"scene":{"animations":[{"url":"name://@selection/Genesis9:/Genesis9Eyelashes:#materials/Eyelashes:?diffuse/image_file","keys":[[0,")"
                       + map + R"("]]})" + more + "]}}");
  };
  // a colour that also sets the body: the body is left alone
  const Json a  = preset("/brown.jpg", R"(,{"url":"name://@selection/Genesis9:#materials/Body:?diffuse/image_file","keys":[[0,"/tan.jpg"]]})");
  const auto ra = fv::dson::choosePreset(scene, lashes, &a);
  EXPECT_EQ(ra.owned, lashes);
  EXPECT_EQ(scene.materials[7].surface.base.map, "/brown.jpg");
  EXPECT_EQ(scene.materials[0].surface.base.map, "/Runtime/Textures/face_d.jpg");
  // the next replaces it
  const Json b = preset("/blonde.jpg");
  fv::dson::choosePreset(scene, lashes, &b, ra.owned);
  EXPECT_EQ(scene.materials[7].surface.base.map, "/blonde.jpg");
  // null: as it was worn, with its edits
  scene.materials[7].edits.recolor = fv::mat::Rgb{1, 0, 0};
  const auto own                   = fv::dson::choosePreset(scene, lashes, nullptr, ra.owned);
  EXPECT_TRUE(own.changed.count(7));
  EXPECT_TRUE(own.owned.empty());
  EXPECT_EQ(scene.materials[7].surface.base.map, "/Runtime/Textures/face_d.jpg");
  EXPECT_TRUE(scene.materials[7].edits.recolor.has_value());
}

// A recolour (a hair colour) bakes into the base map after the rest of it; a flat colour goes
// towards the target instead.
TEST(Materials, ARecolourBakesTheBaseMapOrMovesAFlatColour)
{
  const auto     hair = fv::dson::parseSurface("Hair", channelsOf(kSkin));
  fv::mat::Edits e;
  e.recolor          = fv::mat::Rgb{0.2f, 0.4f, 0.6f};
  e.recolorStrength  = 0.5f;
  const auto lowered = fv::mat::lower(hair, e, {});
  ASSERT_FALSE(lowered.requests.empty());
  const fv::mat::TextureRequest& base = lowered.requests[0];
  ASSERT_TRUE(base.recolor);
  EXPECT_TRUE(base.composed());
  EXPECT_EQ(base.rel, "/Runtime/Textures/face_d.jpg");
  // the cooker's rc= (texcache.rs Recolour::parse), before the host percent-encodes it whole
  EXPECT_EQ(fv::mat::recolorQuery(*base.recolor), "0.2,0.4,0.6|0.5");
  EXPECT_NE(fv::mat::requestKey(base).find("|rc:0.2,0.4,0.6|0.5"), std::string::npos);
  EXPECT_FALSE(fv::mat::lower(hair, {}, {}).requests[0].recolor);

  // no map: the flat colour, half way to the target
  const auto flat = fv::dson::parseSurface("Hair", channelsOf(kEyeMoisture));
  fv::mat::Edits half;
  half.recolor         = fv::mat::Rgb{1.f, 0.f, 0.f};
  half.recolorStrength = 0.5f;
  const Json factor    = fv::mat::lower(flat, half, {}).material["pbrMetallicRoughness"]["baseColorFactor"];
  EXPECT_GT(factor[0].get<float>(), factor[1].get<float>());

  // an edit like the rest: merged, and empty without it
  fv::mat::Edits merged;
  merged.merge(e);
  EXPECT_TRUE(merged.recolor.has_value());
  EXPECT_FALSE(merged.empty());
}

// The skin a skin or makeup preset may change: the character's, but its eyes, what was taken off and
// what the host leaves out (worn items); a tone shifts every skin surface against the body's map.
TEST(Materials, TheSkinIsTheCharactersSurfacesButItsEyesAndWornItems)
{
  const fv::SceneGraph scene = eyeScene();
  // the lashes stand in for a worn item; the tear film is no eye
  EXPECT_EQ(fv::dson::skinSurfaces(scene, {7}), (std::set<int>{0, 1, 6}));
  EXPECT_EQ(fv::dson::toneReference(scene), "/Runtime/Textures/face_d.jpg");
  EXPECT_EQ(fv::dson::toneReference(scene, {0, 1}), "");
}

// Makeup is a layer: the next one takes off only the channels the last one set, so a skin preset
// applied under it stays, and a channel something else set since keeps that.
TEST(Materials, MakeupComesOffAloneLeavingTheSkinUnderIt)
{
  fv::SceneGraph scene = eyeScene();
  const auto     head  = [](const std::string& channel, const std::string& field, const std::string& value) {
    return R"({"url":"name://@selection/Genesis9:#materials/Head:?extra/studio_material_channels/channels/)" + channel + "/" + field
           + R"(","keys":[[0,)" + value + "]]}";
  };
  const auto doc = [](const std::vector<std::string>& anims) {
    std::string all;
    for(const std::string& a : anims)
      all += (all.empty() ? "" : ",") + a;
    return Json::parse(R"({"scene":{"animations":[)" + all + "]}}");
  };
  const auto image = [&](const char* channel) {
    const fv::dson::Channel* c = scene.materials[1].channels->get(fv::dson::norm(channel));
    return c && c->image ? *c->image : std::string("none");
  };
  // a skin under it, not a layer
  const Json skin = doc({head("diffuse", "image_file", R"("/tan.jpg")")});
  fv::dson::applyMaterialPreset(skin, scene);
  const std::set<int> surfaces = fv::dson::skinSurfaces(scene, {7});

  const Json a  = doc({head("Makeup%20Weight", "image_file", R"("/maskA.png")"), head("Makeup%20Base%20Color", "image_file", R"("/lipsA.jpg")")});
  const auto ra = fv::dson::chooseLayer(scene, surfaces, &a, {});
  EXPECT_EQ(ra.changed, std::set<int>{1});
  EXPECT_EQ(image("Makeup Base Color"), "/lipsA.jpg");
  EXPECT_EQ(image("Makeup Weight"), "/maskA.png");

  // the next: only its colour, so A's mask comes off with it; the skin stays
  const Json b  = doc({head("Makeup%20Base%20Color", "image_file", R"("/lipsB.jpg")")});
  const auto rb = fv::dson::chooseLayer(scene, surfaces, &b, ra.layer);
  EXPECT_EQ(image("Makeup Base Color"), "/lipsB.jpg");
  EXPECT_EQ(image("Makeup Weight"), "none");
  EXPECT_EQ(scene.materials[1].surface.base.map, "/tan.jpg");

  // something else sets B's channel since: taking B off keeps that
  fv::dson::applyMaterialPreset(doc({head("Makeup%20Base%20Color", "image_file", R"("/mine.jpg")")}), scene);
  const auto off = fv::dson::chooseLayer(scene, surfaces, nullptr, rb.layer);
  EXPECT_TRUE(off.changed.empty());
  EXPECT_EQ(image("Makeup Base Color"), "/mine.jpg");

  // with nothing in between, null takes the makeup off to the skin under it
  const auto rc = fv::dson::chooseLayer(scene, surfaces, &a, {});
  fv::dson::chooseLayer(scene, surfaces, nullptr, rc.layer);
  EXPECT_EQ(image("Makeup Base Color"), "/mine.jpg");
  EXPECT_EQ(image("Makeup Weight"), "none");
}

const char* kMadeUp = R"({"extra":[{"type":"studio/material/daz_brick","channels":[
  {"channel":{"id":"Diffuse Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/face_d.jpg"}},
  {"channel":{"id":"Makeup Enable","type":"bool","current_value":true}},
  {"channel":{"id":"Makeup Weight","type":"float","current_value":0.8,"image_file":"/mask.png"}},
  {"channel":{"id":"Makeup Base Color","type":"color","current_value":[1,1,1],"image_file":"/lips.png"}}
]}]})";

// A lipstick's colour replaces its colour map over the weight map, and its weight is the panel's; a
// skin tone recolours the base map against the tone reference, the scattering colour as well.
TEST(Materials, AToneAndAMakeupColourLowerIntoTheBaseMap)
{
  const auto lips = fv::dson::parseSurface("Lips", channelsOf(kMadeUp));
  ASSERT_TRUE(lips.makeup);
  fv::mat::Edits e;
  e.makeupColor = fv::mat::Rgb{0.6f, 0.05f, 0.1f};
  e.makeupWeight = 0.5f;
  const auto m  = fv::mat::lower(lips, e, {});
  ASSERT_TRUE(m.requests[0].makeup);
  EXPECT_EQ(m.requests[0].makeup->colorRel, "");
  EXPECT_EQ(m.requests[0].makeup->weightRel, "/mask.png");
  EXPECT_FLOAT_EQ(m.requests[0].makeup->weightFactor, 0.5f);
  EXPECT_FLOAT_EQ(m.requests[0].makeup->tint[0], 0.6f);
  EXPECT_EQ(fv::mat::lower(lips, {}, {}).requests[0].makeup->colorRel, "/lips.png");

  fv::mat::Edits tone;
  tone.tone         = fv::mat::Rgb{0.3f, 0.2f, 0.15f};
  tone.toneStrength = 0.75f;
  fv::mat::LowerOptions o;
  o.toneReference = "/body_d.jpg";
  o.skin          = fv::mat::SkinMode::eThin;
  const auto face = fv::mat::lower(fv::dson::parseSurface("Face", channelsOf(kSkin)), tone, o);
  int        toned = 0;
  for(const fv::mat::TextureRequest& r : face.requests)
    if(r.recolor)
    {
      toned++;
      EXPECT_EQ(r.recolor->reference, "/body_d.jpg");
      EXPECT_FLOAT_EQ(r.recolor->strength, 0.75f);
      EXPECT_NE(fv::mat::requestKey(r).find("/body_d.jpg"), std::string::npos);
    }
  // the base colour and the thin skin's scattering colour
  EXPECT_EQ(toned, 2);
  // without a reference, against its own map
  const auto own = fv::mat::lower(fv::dson::parseSurface("Face", channelsOf(kSkin)), tone, {});
  ASSERT_TRUE(own.requests[0].recolor);
  EXPECT_EQ(own.requests[0].recolor->reference, "/Runtime/Textures/face_d.jpg");
  fv::mat::Edits merged;
  merged.merge(tone);
  merged.merge(e);
  EXPECT_TRUE(merged.tone && merged.makeupColor && merged.makeupWeight && merged.toneStrength);
}

// pbr_skin.mdl: the metal layer is there only with Metallicity Enable (false in PBRSkin.dsf), weighted
// by Metallic Weight times its map; Makeup Metallicity Enable, with the makeup on, only tints it with
// the made-up colour. IST Esme's lip presets: metal 0.2912088 x LipMask, makeup 0.7509727 x LipMask,
// both switches on: metal on the lips alone. Polina 9's head: Metallic Weight 1 and no Metallicity
// Enable, so skin. Iray Uber has no switch.
TEST(Materials, MetallicityEnableGatesTheMetalAndMakeupMetallicityOnlyTintsIt)
{
  const auto skin = [](const char* metallicity, bool makeup, bool makeupMetallicity) {
    std::string c = R"({"extra":[{"type":"studio/material/daz_brick","channels":[
      {"channel":{"id":"Diffuse Enable","type":"bool","current_value":true}},
      {"channel":{"id":"Detail Enable","type":"bool","current_value":false}},
      {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/head_d.jpg"}},
      {"channel":{"id":"Metallic Weight","type":"float","current_value":0.2912088,"image_file":"/lipmask.png"}},
      {"channel":{"id":"Makeup Base Color","type":"color","current_value":[0.6,0.1,0.1]}},
      {"channel":{"id":"Makeup Weight","type":"float","current_value":0.7509727,"image_file":"/lipmask.png"}},)";
    if(metallicity)
      c += std::string(R"({"channel":{"id":"Metallicity Enable","type":"bool","current_value":)") + metallicity + "}},";
    c += std::string(R"({"channel":{"id":"Makeup Enable","type":"bool","current_value":)") + (makeup ? "true" : "false") + "}},";
    c += std::string(R"({"channel":{"id":"Makeup Metallicity Enable","type":"bool","current_value":)") + (makeupMetallicity ? "true" : "false") + "}}";
    c += "]}]}";
    return fv::dson::parseSurface("Head", channelsOf(c.c_str()));
  };
  // both switches on: Metallic Weight and its map, untouched by the makeup's weight
  const auto lips = skin("true", true, true);
  EXPECT_FLOAT_EQ(lips.metallic.value, 0.2912088f);
  EXPECT_EQ(lips.metallic.map, "/lipmask.png");
  ASSERT_TRUE(lips.makeup);
  EXPECT_FLOAT_EQ(lips.makeup->weight, 0.7509727f);
  EXPECT_FLOAT_EQ(skin("true", true, false).metallic.value, 0.2912088f);
  // the makeup off changes nothing about the metal
  const auto bare = skin("true", false, true);
  EXPECT_FLOAT_EQ(bare.metallic.value, 0.2912088f);
  EXPECT_EQ(bare.metallic.map, "/lipmask.png");
  // Metallicity Enable off, PBRSkin.dsf's default, which every PBRSkin surface has under it (a
  // surface without the switch is an Iray Uber one with makeup laid over: ALipColourOverAnUberHead...)
  const auto off = skin("false", true, true);
  EXPECT_FLOAT_EQ(off.metallic.value, 0);
  EXPECT_TRUE(off.metallic.map.empty());
  // Iray Uber: no switch, the weight and its mask as they are
  const auto belt = fv::dson::parseSurface("BeltHolster_Metal_Mat", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"Metallic Weight","type":"float","current_value":1,"image_file":"/Belt M.jpg"}}]}]})"));
  EXPECT_FLOAT_EQ(belt.metallic.value, 1);
  EXPECT_EQ(belt.metallic.map, "/Belt M.jpg");

  // lowered: the metal reflects the made-up base colour, on the lips alone
  const auto l = fv::mat::lower(lips, {}, {});
  const auto& pbr = l.material["pbrMetallicRoughness"];
  EXPECT_FLOAT_EQ(pbr["metallicFactor"].get<float>(), 0.2912088f);
  const fv::mat::TextureRequest& metal = l.requests[pbr["metallicRoughnessTexture"]["slot"].get<size_t>()];
  EXPECT_EQ(metal.rel, "/lipmask.png");
  EXPECT_EQ(metal.channels, "mt");
  EXPECT_TRUE(metal.metalRel.empty());
  const fv::mat::TextureRequest& base = l.requests[pbr["baseColorTexture"]["slot"].get<size_t>()];
  ASSERT_TRUE(base.makeup);
  EXPECT_EQ(base.makeup->weightRel, "/lipmask.png");
}

// Iray's metal is Metallic Weight times its map, a mask (RE4R Ada Wong's belt: weight 1, metal only
// where Belt M.jpg is white). glTF's metalness is metallicFactor times metallicRoughness blue: the
// mask goes in the roughness map's blue, or makes the image alone; no metal, no mask.
TEST(Materials, AMetalMapReachesMetallicRoughnessBlue)
{
  const auto belt = [](float weight, bool roughMap) {
    std::string c = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
      {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/belt_d.jpg"}},)";
    c += std::string(R"({"channel":{"id":"Glossy Roughness","type":"float","current_value":0.4)") + (roughMap ? R"(,"image_file":"/belt_r.jpg")" : "")
         + "}},";
    c += R"({"channel":{"id":"Metallic Weight","type":"float","current_value":)" + std::to_string(weight) + R"(,"image_file":"/belt_m.jpg"}})";
    c += "]}]}";
    return fv::dson::parseSurface("Belt", channelsOf(c.c_str()));
  };
  const auto metalRequest = [](const fv::mat::Lowered& l) -> const fv::mat::TextureRequest* {
    const auto& pbr = l.material["pbrMetallicRoughness"];
    return pbr.contains("metallicRoughnessTexture") ? &l.requests[pbr["metallicRoughnessTexture"]["slot"].get<size_t>()] : nullptr;
  };

  // the mask alone: the image is the metal map's grey in blue
  const auto alone = fv::mat::lower(belt(1, false), {}, {});
  EXPECT_FLOAT_EQ(alone.material["pbrMetallicRoughness"]["metallicFactor"].get<float>(), 1);
  const fv::mat::TextureRequest* mt = metalRequest(alone);
  ASSERT_TRUE(mt);
  EXPECT_EQ(mt->rel, "/belt_m.jpg");
  EXPECT_EQ(mt->channels, "mt");
  EXPECT_FALSE(mt->srgb);
  EXPECT_TRUE(mt->metalRel.empty());

  // with a roughness map, the mask rides in its blue, and the two are different images to cook
  const auto both = fv::mat::lower(belt(1, true), {}, {});
  const fv::mat::TextureRequest* mr = metalRequest(both);
  ASSERT_TRUE(mr);
  EXPECT_EQ(mr->rel, "/belt_r.jpg");
  EXPECT_EQ(mr->channels, "mr");
  EXPECT_EQ(mr->metalRel, "/belt_m.jpg");
  fv::mat::TextureRequest unmasked = *mr;
  unmasked.metalRel.clear();
  EXPECT_NE(fv::mat::requestKey(*mr), fv::mat::requestKey(unmasked));

  // no metal: the mask is left out
  const auto none = fv::mat::lower(belt(0, false), {}, {});
  EXPECT_FALSE(metalRequest(none));
  const auto roughOnly = fv::mat::lower(belt(0, true), {}, {});
  ASSERT_TRUE(metalRequest(roughOnly));
  EXPECT_TRUE(metalRequest(roughOnly)->metalRel.empty());

  // a panel metalness on a surface authored without metal shows the authored mask
  fv::mat::Edits e;
  e.metalness       = 0.5f;
  const auto edited = fv::mat::lower(belt(0, false), e, {});
  EXPECT_FLOAT_EQ(edited.material["pbrMetallicRoughness"]["metallicFactor"].get<float>(), 0.5f);
  ASSERT_TRUE(metalRequest(edited));
  EXPECT_EQ(metalRequest(edited)->channels, "mt");
}

// ---- The top coat ----

using ChannelValues = std::vector<std::pair<std::string, Json>>;

// A surface of `shader`'s channels, each "id": value (a number, bool or [r,g,b]); an image where
// `images` gives one
fv::mat::Surface surfaceOf(const char* name, const char* shader, const ChannelValues& values, const std::map<std::string, std::string>& images = {})
{
  Json channels = Json::array();
  for(const auto& [id, value] : values)
  {
    Json c = {{"id", id}, {"type", value.is_array() ? "color" : value.is_boolean() ? "bool" : "float"}, {"current_value", value}};
    if(const auto it = images.find(id); it != images.end())
      c["image_file"] = it->second;
    channels.push_back({{"channel", c}});
  }
  const Json doc = {{"extra", Json::array({Json{{"type", shader}, {"channels", channels}}})}};
  return fv::dson::parseSurface(name, channelsOf(doc.dump().c_str()));
}

const char* kPbrSkinType = "studio/material/daz_brick";
const char* kUberType    = "studio/material/uber_iray";

const nlohmann::json& clearcoatOf(const fv::mat::Lowered& l)
{
  return l.material["extensions"]["KHR_materials_clearcoat"];
}

const fv::mat::TextureRequest& requestOf(const fv::mat::Lowered& l, const nlohmann::json& info)
{
  return l.requests[info["slot"].get<size_t>()];
}

// pbr_skin.mdl lays the coat over the skin only with Top Coat Enable, false in PBRSkin.dsf. Polina 9's
// head stores Top Coat Weight 0.25 at roughness 0.6721311 and leaves the switch off: no coat, and the
// clearcoat slot goes to her tighter dual lobe. Iray Uber has no switch: a weight makes a coat.
TEST(Materials, APbrSkinTopCoatNeedsItsEnable)
{
  const auto head = [](std::optional<bool> enable) {
    ChannelValues v = {{"Detail Enable", false},
                       {"Dual Lobe Specular Enable", true},
                       {"Dual Lobe Specular Weight", 0.5},
                       {"Dual Lobe Specular Reflectivity", 0.25},
                       {"Specular Lobe 1 Roughness", 0.6},
                       {"Specular Lobe 2 Roughness Mult", 0.4},
                       {"Dual Lobe Specular Ratio", 0.15},
                       {"Top Coat Weight", 0.25},
                       {"Top Coat Roughness", 0.6721311}};
    if(enable)
      v.push_back({"Top Coat Enable", *enable});
    return surfaceOf("Head", kPbrSkinType, v);
  };
  // off, as PBRSkin.dsf has it under her head (a surface without the switch is an Iray Uber one
  // with makeup laid over, whose coat stays: ALipColourOverAnUberHeadLeavesItsLayersOn)
  {
    const auto s = head(false);
    EXPECT_FALSE(s.topCoat);
    // the tight lobe (0.6 x 0.4, lobe 2's 0.15 of 0.5 x 0.25 x 2) is the clearcoat; the broad stays the base
    const auto  l   = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
    const auto& ext = l.material["extensions"];
    EXPECT_NEAR(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.0375f, 1e-5f);
    EXPECT_NEAR(clearcoatOf(l)["clearcoatRoughnessFactor"].get<float>(), 0.24f, 1e-5f);
    EXPECT_NEAR(ext["KHR_materials_specular"]["specularFactor"].get<float>(), 0.2125f, 1e-5f);
    EXPECT_NEAR(l.material["pbrMetallicRoughness"]["roughnessFactor"].get<float>(), 0.6f, 1e-5f);
  }
  // on: the coat takes the slot, and the tight lobe folds back into the base specular
  const auto on = head(true);
  ASSERT_TRUE(on.topCoat);
  const auto l = fv::mat::lower(on, {}, {.skin = fv::mat::SkinMode::eRaster});
  EXPECT_NEAR(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.25f, 1e-5f);
  EXPECT_NEAR(clearcoatOf(l)["clearcoatRoughnessFactor"].get<float>(), 0.6721311f, 1e-5f);
  EXPECT_NEAR(l.material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(), 0.25f, 1e-5f);

  // Iray Uber: no switch
  const auto uber = surfaceOf("Shirt", kUberType, {{"Glossy Roughness", 0.4}, {"Top Coat Weight", 0.3}, {"Top Coat Roughness", 0.2}});
  ASSERT_TRUE(uber.topCoat);
  EXPECT_FLOAT_EQ(uber.topCoat->weight, 0.3f);
  EXPECT_FLOAT_EQ(uber.topCoat->roughness, 0.2f);
  EXPECT_FLOAT_EQ(uber.topCoat->bumpWeight, 1);

  // a preset that only switches the coat off takes it away (ATK Giselle's Lip Gloss 00 Off)
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Head", channelsOf(R"({"extra":[{"type":"studio/material/daz_brick","channels":[
    {"channel":{"id":"Detail Enable","type":"bool","current_value":false}},
    {"channel":{"id":"Top Coat Enable","type":"bool","current_value":true}},
    {"channel":{"id":"Top Coat Weight","type":"float","current_value":0.25}},
    {"channel":{"id":"Top Coat Roughness","type":"float","current_value":0.11}}]}]})")));
  ASSERT_TRUE(scene.materials[0].surface.topCoat);
  fv::dson::applyMaterialPreset(Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Head:?extra/studio_material_channels/channels/Top%20Coat%20Enable/value","keys":[[0,false]]}
  ]}})"),
                                scene);
  EXPECT_FALSE(scene.materials[0].surface.topCoat);
}

// The coat's weight and roughness are each the value times its map (PBRSkin's texture instances;
// for Iray Uber, DAZ's map-times-value convention). CJ Evangeline's Lip Gloss 01: weight 0.503268
// over her lip mask, a coat on the lips alone. glTF's clearcoat is the factor times the texture's red,
// its roughness the factor times green; both maps are linear.
TEST(Materials, ATopCoatsMapsGoWhereTheCoatIs)
{
  const auto head = [](double weight) {
    return surfaceOf("Head", kPbrSkinType,
                     {{"Detail Enable", false}, {"Top Coat Enable", true}, {"Top Coat Weight", weight}, {"Top Coat Roughness", 0.4052288}},
                     {{"Top Coat Weight", "/Runtime/Textures/lipmask.png"}, {"Top Coat Roughness", "/Runtime/Textures/lip_r.png"}});
  };
  const auto s = head(0.503268);
  ASSERT_TRUE(s.topCoat);
  EXPECT_EQ(s.topCoat->weightMap, "/Runtime/Textures/lipmask.png");
  EXPECT_EQ(s.topCoat->roughnessMap, "/Runtime/Textures/lip_r.png");
  const auto  l  = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
  const auto& cc = clearcoatOf(l);
  EXPECT_NEAR(cc["clearcoatFactor"].get<float>(), 0.503268f, 1e-6f);
  EXPECT_NEAR(cc["clearcoatRoughnessFactor"].get<float>(), 0.4052288f, 1e-6f);
  ASSERT_TRUE(cc.contains("clearcoatTexture"));
  const fv::mat::TextureRequest& mask = requestOf(l, cc["clearcoatTexture"]);
  EXPECT_EQ(mask.rel, "/Runtime/Textures/lipmask.png");
  EXPECT_FALSE(mask.srgb);
  EXPECT_TRUE(mask.channels.empty());
  ASSERT_TRUE(cc.contains("clearcoatRoughnessTexture"));
  const fv::mat::TextureRequest& rough = requestOf(l, cc["clearcoatRoughnessTexture"]);
  EXPECT_EQ(rough.rel, "/Runtime/Textures/lip_r.png");
  EXPECT_FALSE(rough.srgb);

  // a panel clearcoat replaces the weight; the mask stays where the coat is
  fv::mat::Edits e;
  e.clearcoat       = 0.7f;
  const auto edited   = fv::mat::lower(s, e, {.skin = fv::mat::SkinMode::eRaster});
  EXPECT_FLOAT_EQ(clearcoatOf(edited)["clearcoatFactor"].get<float>(), 0.7f);
  EXPECT_TRUE(clearcoatOf(edited).contains("clearcoatTexture"));

  // CJ Evangeline's own head keeps the mask at weight 0: no coat
  const auto bare = head(0);
  EXPECT_FALSE(bare.topCoat);
  EXPECT_FALSE(fv::mat::lower(bare, {}, {.skin = fv::mat::SkinMode::eRaster}).material["extensions"].contains("KHR_materials_clearcoat"));

  // Iray Uber's weight map (Sienna Thornbrook's skin)
  const auto uber = surfaceOf("Torso", kUberType, {{"Glossy Roughness", 0.4}, {"Top Coat Weight", 0.45}}, {{"Top Coat Weight", "/Runtime/Textures/face_s.jpg"}});
  ASSERT_TRUE(uber.topCoat);
  EXPECT_EQ(uber.topCoat->weightMap, "/Runtime/Textures/face_s.jpg");
  const auto ul = fv::mat::lower(uber, {}, {});
  ASSERT_TRUE(clearcoatOf(ul).contains("clearcoatTexture"));
  EXPECT_EQ(requestOf(ul, clearcoatOf(ul)["clearcoatTexture"]).rel, "/Runtime/Textures/face_s.jpg");

  // Specular/Glossiness mode: the roughness is 1 - Top Coat Glossiness, and a glossiness map is no roughness map
  const auto gloss = surfaceOf("Hair", kUberType, {{"Base Mixing", 1}, {"Top Coat Weight", 0.5}, {"Top Coat Glossiness", 0.8}},
                               {{"Top Coat Glossiness", "/gloss.png"}});
  ASSERT_TRUE(gloss.topCoat);
  EXPECT_NEAR(gloss.topCoat->roughness, 0.2f, 1e-6f);
  EXPECT_TRUE(gloss.topCoat->roughnessMap.empty());
  EXPECT_FALSE(clearcoatOf(fv::mat::lower(gloss, {}, {})).contains("clearcoatRoughnessTexture"));

  // the MDL takes the weight in 0..1: the Sporty Ponytails hair's 2 is a full coat
  const auto ponytail = surfaceOf("Bangs1", kUberType, {{"Top Coat Weight", 2}, {"Top Coat Roughness", 1}});
  ASSERT_TRUE(ponytail.topCoat);
  EXPECT_FLOAT_EQ(ponytail.topCoat->weight, 1);
  EXPECT_FLOAT_EQ(clearcoatOf(fv::mat::lower(ponytail, {}, {}))["clearcoatFactor"].get<float>(), 1);
}

// PBRSkin's coat normal blends the skin's relief towards the smooth normal by Top Coat Bump Weight,
// which scales its slopes by about the weight: glTF's clearcoatNormalTexture scale. At 0 the slot is
// kept with scale 0, the smooth normal; without it the shader would give the coat the full relief.
// Iray Uber's coat follows the relief whole.
TEST(Materials, APbrSkinCoatFollowsTheReliefByItsBumpWeight)
{
  const auto head = [](std::optional<double> bump) {
    ChannelValues v = {{"Detail Enable", false}, {"Bump Enable", true}, {"Normal Map", 0.8}, {"Top Coat Enable", true}, {"Top Coat Weight", 1}};
    if(bump)
      v.push_back({"Top Coat Bump Weight", *bump});
    return surfaceOf("Head", kPbrSkinType, v, {{"Normal Map", "/Runtime/Textures/n.jpg"}});
  };
  const auto coatScale = [](const fv::mat::Surface& s) {
    const auto l      = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
    const auto normal = l.material["normalTexture"];
    EXPECT_NEAR(normal["scale"].get<float>(), 0.8f, 1e-6f);
    const auto& cc = clearcoatOf(l);
    EXPECT_TRUE(cc.contains("clearcoatNormalTexture"));
    if(!cc.contains("clearcoatNormalTexture"))
      return -1.f;
    EXPECT_EQ(cc["clearcoatNormalTexture"]["slot"], normal["slot"]);  // the same image
    return cc["clearcoatNormalTexture"]["scale"].get<float>();
  };
  EXPECT_NEAR(coatScale(head(0.5)), 0.4f, 1e-6f);
  EXPECT_NEAR(coatScale(head(0)), 0, 1e-6f);
  EXPECT_NEAR(coatScale(head(std::nullopt)), 0.8f, 1e-6f);  // PBRSkin.dsf's default is 1
  const auto uber = surfaceOf("Shirt", kUberType, {{"Normal Map", 0.8}, {"Top Coat Weight", 1}, {"Top Coat Bump Weight", 0.3}},
                              {{"Normal Map", "/Runtime/Textures/n.jpg"}});
  EXPECT_NEAR(coatScale(uber), 0.8f, 1e-6f);
}

// PBRSkin's Top Coat Color tints the coat's own reflection; the base under it keeps the share an
// untinted coat leaves it. Its luminance therefore dims the coat's weight. CHT Ava's skin: an enabled
// coat of 0.11 at roughness 0, coloured 0.502 grey (0.216 linear).
TEST(Materials, APbrSkinCoatsColourDimsItsReflection)
{
  const auto head = [](double c) {
    return surfaceOf("Head", kPbrSkinType,
                     {{"Detail Enable", false},
                      {"Top Coat Enable", true},
                      {"Top Coat Weight", 0.11},
                      {"Top Coat Roughness", 0},
                      {"Top Coat Color", Json::array({c, c, c})}});
  };
  const auto ava = head(0.5019608);
  ASSERT_TRUE(ava.topCoat);
  for(int i = 0; i < 3; i++)
    EXPECT_NEAR(ava.topCoat->color[i], fv::mat::srgbToLinear(0.5019608), 1e-6f);
  const float lum = fv::mat::srgbToLinear(0.5019608);
  EXPECT_NEAR(clearcoatOf(fv::mat::lower(ava, {}, {.skin = fv::mat::SkinMode::eRaster}))["clearcoatFactor"].get<float>(), 0.11f * lum, 1e-5f);
  EXPECT_NEAR(clearcoatOf(fv::mat::lower(head(1), {}, {.skin = fv::mat::SkinMode::eRaster}))["clearcoatFactor"].get<float>(), 0.11f, 1e-5f);

  // Iray Uber's colour is not read here (its Color Effect can change the weight itself)
  const auto uber = surfaceOf("Shirt", kUberType, {{"Top Coat Weight", 0.4}, {"Top Coat Color", Json::array({0.5, 0.5, 0.5})}});
  ASSERT_TRUE(uber.topCoat);
  EXPECT_EQ(uber.topCoat->color, (fv::mat::Rgb{1, 1, 1}));
  EXPECT_FLOAT_EQ(clearcoatOf(fv::mat::lower(uber, {}, {}))["clearcoatFactor"].get<float>(), 0.4f);
}

// ---- PBRSkin's volume ----

// A PBRSkin face with its three volume switches, absorbing 0.5 (0.214 linear) over 1 mm and scattering
// 0.5 over 0.1 mm, backwards
ChannelValues pbrSkinVolume(bool translucency, bool transmission, bool subsurface)
{
  return {{"Detail Enable", false},
          {"Translucency Enable", translucency},
          {"Translucency Weight", 0.8},
          {"Transmission Enable", transmission},
          {"Sub Surface Enable", subsurface},
          {"Transmitted Color", Json::array({0.5, 0.5, 0.5})},
          {"Transmitted Measurement Distance", 0.1},
          {"SSS Color", Json::array({0.5, 0.5, 0.5})},
          {"Scattering Measurement Distance", 0.01},
          {"SSS Direction", -0.7}};
}

// pbr_skin.mdl's volume scatters by -ln(SSS Color)/distance: its hidden log switch defaults on, and
// PBRSkin's brick graph leaves it unconnected. Iray Uber's defaults off: SSS Color/distance.
TEST(Materials, PbrSkinScattersLogarithmically)
{
  const auto s = surfaceOf("Face", kPbrSkinType, pbrSkinVolume(true, true, true));
  ASSERT_TRUE(s.translucency && s.translucency->subsurface);
  EXPECT_TRUE(s.translucency->subsurface->logScatter);
  // sigma_s = -ln(0.214)/0.1 mm = 15416 per metre, sigma_a = -ln(0.214)/1 mm = 1542: a tenth as much
  const auto v = fv::mat::volumeMedium(s.translucency->subsurface);
  EXPECT_NEAR(v.distance, 5.89711e-5f, 1e-9f);
  EXPECT_FLOAT_EQ(v.anisotropy, -0.7f);
  for(int i = 0; i < 3; i++)
  {
    EXPECT_NEAR(v.albedo[i], 0.9090909f, 1e-5f);
    EXPECT_NEAR(v.attenuation[i], 0.3678794f, 1e-5f);
    EXPECT_NEAR(v.multiscatter[i], 0.5794536f, 1e-4f);
  }
  const auto  l   = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eVolume});
  const auto& ext = l.material["extensions"];
  EXPECT_NEAR(ext["KHR_materials_volume"]["attenuationDistance"].get<float>(), 5.89711e-5f, 1e-9f);
  EXPECT_FLOAT_EQ(ext["KHR_materials_scatter"]["scatterAnisotropy"].get<float>(), -0.7f);
}

TEST(Materials, IrayUberScattersLinearly)
{
  ChannelValues v = pbrSkinVolume(true, true, true);
  v.erase(v.begin());  // no Detail Enable: Iray Uber
  v.push_back({"SSS Mode", 1});
  const auto s = surfaceOf("Face", kUberType, v);
  ASSERT_TRUE(s.translucency && s.translucency->subsurface);
  EXPECT_FALSE(s.translucency->subsurface->logScatter);
  // sigma_s = 0.214/0.1 mm = 2140 per metre against the same 1542 absorbed
  const auto m = fv::mat::volumeMedium(s.translucency->subsurface);
  EXPECT_NEAR(m.distance, 2.715916e-4f, 1e-8f);
  for(int i = 0; i < 3; i++)
  {
    EXPECT_NEAR(m.albedo[i], 0.5813178f, 1e-5f);
    EXPECT_NEAR(m.multiscatter[i], 0.2561f, 1e-3f);
  }
}

// pbr_skin.mdl gates its absorption by Transmission Enable alone and its scattering (and the
// scattering's direction) by Sub Surface Enable alone; with both off there is nothing to measure.
TEST(Materials, PbrSkinSwitchesAbsorptionAndScatteringApart)
{
  {
    const auto s = surfaceOf("Face", kPbrSkinType, pbrSkinVolume(true, false, true));
    ASSERT_TRUE(s.translucency && s.translucency->subsurface);
    const auto& sub = *s.translucency->subsurface;
    EXPECT_EQ(sub.transmittedDistance, 0);
    EXPECT_EQ(sub.scatterDistance, 0.01f);
    EXPECT_EQ(sub.direction, -0.7f);
    // scattering alone: nothing absorbed
    const auto v = fv::mat::volumeMedium(sub);
    EXPECT_NEAR(v.distance, 6.486822e-5f, 1e-9f);
    EXPECT_FLOAT_EQ(v.anisotropy, -0.7f);
    for(int i = 0; i < 3; i++)
    {
      EXPECT_NEAR(v.albedo[i], 1, 1e-6f);
      EXPECT_EQ(v.multiscatter[i], 1);
      EXPECT_NEAR(v.attenuation[i], 0.3678794f, 1e-5f);
    }
  }
  {
    const auto s = surfaceOf("Face", kPbrSkinType, pbrSkinVolume(true, true, false));
    ASSERT_TRUE(s.translucency && s.translucency->subsurface);
    const auto& sub = *s.translucency->subsurface;
    EXPECT_EQ(sub.scatterDistance, 0);
    EXPECT_EQ(sub.transmittedDistance, 0.1f);
    EXPECT_EQ(sub.direction, 0);
    // absorption alone: nothing scatters, and so no direction
    const auto v = fv::mat::volumeMedium(sub);
    EXPECT_NEAR(v.distance, 6.486822e-4f, 1e-8f);
    EXPECT_EQ(v.anisotropy, 0);
    for(int i = 0; i < 3; i++)
    {
      EXPECT_EQ(v.albedo[i], 0);
      EXPECT_EQ(v.multiscatter[i], 0);
    }
  }
  EXPECT_FALSE(surfaceOf("Face", kPbrSkinType, pbrSkinVolume(true, false, false)).translucency->subsurface);
}

// PBRSkin's brick graph makes the surface thin-walled when Translucency Enable is off: no light enters
// and there is no medium, in every skin mode, measured skin's included.
TEST(Materials, PbrSkinWithTranslucencyOffIsThinWalledInEveryMode)
{
  const auto expectPlain = [](const fv::mat::Surface& s, const char* why) {
    for(const auto mode : {fv::mat::SkinMode::eMeasured, fv::mat::SkinMode::eVolume, fv::mat::SkinMode::eThin})
    {
      const auto  l   = fv::mat::lower(s, {}, {.skin = mode});
      const auto& ext = l.material["extensions"];
      for(const char* name : {"KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_scatter", "FV_materials_subsurface"})
        EXPECT_FALSE(ext.contains(name)) << why << ", mode " << int(mode) << ": " << name;
    }
  };
  const auto off = surfaceOf("Face", kPbrSkinType, pbrSkinVolume(false, true, true));
  ASSERT_TRUE(off.translucency);
  EXPECT_TRUE(off.translucency->thinWalled);
  EXPECT_EQ(off.translucency->weight, 0);
  expectPlain(off, "Translucency Enable off");

  // on: measured skin enters wholly
  const auto on = surfaceOf("Face", kPbrSkinType, pbrSkinVolume(true, true, true));
  EXPECT_FALSE(on.translucency->thinWalled);
  EXPECT_FLOAT_EQ(fv::mat::lower(on, {}, {.skin = fv::mat::SkinMode::eMeasured})
                      .material["extensions"]["KHR_materials_transmission"]["transmissionFactor"]
                      .get<float>(),
                  1);
  // Iray Uber has no switch
  EXPECT_FALSE(fv::dson::parseSurface("Face", channelsOf(kSkin)).translucency->thinWalled);

  // a surface that looks like PBRSkin only by a stray preset's marker, without the switch, keeps its inside
  const auto stray = surfaceOf("Face", kPbrSkinType, {{"Detail Enable", false}, {"Translucency Weight", 0.8}});
  ASSERT_TRUE(stray.translucency);
  EXPECT_FALSE(stray.translucency->thinWalled);
  const auto  l   = fv::mat::lower(stray, {}, {.skin = fv::mat::SkinMode::eMeasured});
  const auto& ext = l.material["extensions"];
  EXPECT_FLOAT_EQ(ext["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1);
  EXPECT_TRUE(ext.contains("FV_materials_subsurface"));
}

// PBRSkin has no SSS Reflectance Tint (neither PBRSkin.dsf nor its MDL): a value on its surface is
// stale, left from Iray Uber's. Iray Uber's tints the light that comes back out in thin mode.
TEST(Materials, PbrSkinHasNoSssReflectanceTint)
{
  ChannelValues v = {{"Detail Enable", false},
                     {"diffuse", Json::array({1, 1, 1})},
                     {"Translucency Enable", true},
                     {"Translucency Weight", 0.8},
                     {"SSS Reflectance Tint", Json::array({0.5, 0.5, 0.5})}};
  const std::map<std::string, std::string> images = {{"diffuse", "/Runtime/Textures/face_d.jpg"}};
  const auto skin = surfaceOf("Face", kPbrSkinType, v, images);
  ASSERT_TRUE(skin.translucency);
  EXPECT_EQ(skin.translucency->reflectanceTint, (fv::mat::Rgb{1, 1, 1}));
  const auto  thin   = fv::mat::lower(skin, {}, {.skin = fv::mat::SkinMode::eThin});
  const auto& factor = thin.material["extensions"]["KHR_materials_scatter"]["multiscatterColorFactor"];
  for(int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(factor[i].get<float>(), 1);

  v.erase(v.begin());  // Iray Uber
  const auto uber = surfaceOf("Face", kUberType, v, images);
  const auto  uberThin = fv::mat::lower(uber, {}, {.skin = fv::mat::SkinMode::eThin});
  const auto& tinted   = uberThin.material["extensions"]["KHR_materials_scatter"]["multiscatterColorFactor"];
  for(int i = 0; i < 3; i++)
    EXPECT_NEAR(tinted[i].get<float>(), 0.2140411f, 1e-6f);
}

// Tri-X's eye presets give the eye moisture the eye's own map; Iray refracts through it untinted, so
// the glTF material, which tints what it transmits by its base colour, leaves the map out.
TEST(Materials, ARefractingSurfaceIsNotTintedByItsMap)
{
  const auto moisture = fv::dson::parseSurface("EyeMoisture Left", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Eyes_TX_Svetlana9_02.jpg"}},
    {"channel":{"id":"Refraction Weight","type":"float","current_value":1}},
    {"channel":{"id":"Refraction Index","type":"float","current_value":1.38}}]}]})"));
  const auto lowered = fv::mat::lower(moisture, {}, {});
  EXPECT_TRUE(lowered.requests.empty());
  const Json& pbr = lowered.material["pbrMetallicRoughness"];
  EXPECT_FALSE(pbr.contains("baseColorTexture"));
  EXPECT_FLOAT_EQ(pbr["baseColorFactor"][0].get<float>(), 1.f);
  EXPECT_FLOAT_EQ(lowered.material["extensions"]["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1.f);
}

// Colm Jackson's eye presets paint the sclera and iris on the moisture, over a black eyeball, and
// mask its refraction with a map (clear over the cornea only): the surface keeps its map, and the
// mask goes out as where it transmits.
TEST(Materials, ARefractionMaskKeepsTheSurfacesMapWhereItIsOpaque)
{
  const auto moisture = fv::dson::parseSurface("EyeMoisture Left", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/CJ_Evangeline_G9_Eyes.jpg"}},
    {"channel":{"id":"Refraction Weight","type":"float","current_value":1,"image_file":"/eye1_rf.png"}},
    {"channel":{"id":"Refraction Index","type":"float","current_value":1.40678}}]}]})"));
  ASSERT_TRUE(moisture.refraction);
  EXPECT_EQ(moisture.refraction->map, "/eye1_rf.png");
  const auto  lowered = fv::mat::lower(moisture, {}, {});
  const Json& pbr     = lowered.material["pbrMetallicRoughness"];
  ASSERT_TRUE(pbr.contains("baseColorTexture"));
  EXPECT_EQ(lowered.requests[pbr["baseColorTexture"]["slot"].get<size_t>()].rel, "/CJ_Evangeline_G9_Eyes.jpg");
  const Json& transmission = lowered.material["extensions"]["KHR_materials_transmission"];
  ASSERT_TRUE(transmission.contains("transmissionTexture"));
  const auto& mask = lowered.requests[transmission["transmissionTexture"]["slot"].get<size_t>()];
  EXPECT_EQ(mask.rel, "/eye1_rf.png");
  EXPECT_FALSE(mask.srgb);
}

}  // namespace

// PBRSkin's detail layer is a relief of its own when it is another map or the same one tiled; the
// surface's own normal map again, untiled, would only double every tilt, so its normal is dropped
// and its roughness multiplier kept.
TEST(Materials, ADetailLayerRepeatingTheNormalMapAddsNoRelief)
{
  const auto skin = [](const char* detailMap, int tiles) {
    const std::string text = std::string(R"({"extra":[{"type":"studio/material/daz_brick","channels":[
      {"channel":{"id":"Detail Enable","type":"bool","current_value":true}},
      {"channel":{"id":"Bump Enable","type":"bool","current_value":true}},
      {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/body_nm.jpg"}},
      {"channel":{"id":"Detail Weight","type":"float","current_value":1}},
      {"channel":{"id":"Detail Normal Map","type":"float","current_value":1,"image_file":")") + detailMap + R"("}},
      {"channel":{"id":"Detail Normal Map Mode","type":"enum","current_value":1}},
      {"channel":{"id":"Detail Horizontal Tiles","type":"float","current_value":)" + std::to_string(tiles) + R"(}},
      {"channel":{"id":"Detail Vertical Tiles","type":"float","current_value":)" + std::to_string(tiles) + R"(}},
      {"channel":{"id":"Detail Specular Roughness Mult","type":"float","current_value":0.8}}
    ]}]})";
    return fv::dson::parseSurface("Body", channelsOf(text.c_str()));
  };

  const auto duplicate = skin("/Runtime/Textures/body_nm.jpg", 1);
  ASSERT_TRUE(duplicate.detail);
  EXPECT_TRUE(duplicate.detail->map.empty());
  EXPECT_FLOAT_EQ(duplicate.detail->roughness, 0.8f);
  const auto lowered = fv::mat::lower(duplicate, {}, {});
  const auto& ext    = lowered.material["extensions"]["FV_materials_detail"];
  EXPECT_FALSE(ext.contains("detailNormalTexture"));
  EXPECT_FLOAT_EQ(ext["detailRoughnessFactor"].get<float>(), 0.8f);

  EXPECT_EQ(skin("/Runtime/Textures/body_nm.jpg", 8).detail->map, "/Runtime/Textures/body_nm.jpg");  // tiled: fine detail
  EXPECT_EQ(skin("/Runtime/Textures/pores_nm.jpg", 1).detail->map, "/Runtime/Textures/pores_nm.jpg");  // a map of its own
}

// ---- The Clothes tab: map slots, the UV transform, fabrics and sheen ----

namespace {

// A garment surface with a map in most slots and a top coat (so a clearcoat normal copies the normal).
fv::mat::Surface garment()
{
  fv::mat::Surface s;
  s.name                 = "Shirt";
  s.shader               = fv::mat::Surface::Shader::eIray;
  s.base.map             = "/Runtime/Textures/shirt_d.jpg";
  s.opacity.map          = "/Runtime/Textures/shirt_o.jpg";
  s.normalMap            = "/Runtime/Textures/shirt_n.jpg";
  s.glossy.roughness.map = "/Runtime/Textures/shirt_r.jpg";
  s.emission             = fv::mat::Emission{{{1, 1, 1}, "/Runtime/Textures/shirt_e.jpg"}};
  s.topCoat              = fv::mat::Surface::TopCoat{0.5f, 0.1f};
  return s;
}

// Every texture info in a lowered material.
void slotsOf(const nlohmann::json& j, std::vector<const nlohmann::json*>& out)
{
  if(j.is_object() && j.contains("slot"))
    out.push_back(&j);
  else if(j.is_structured())
    for(const auto& x : j)
      slotsOf(x, out);
}

fv::mat::UvTransform transformOf(const nlohmann::json& info)
{
  fv::mat::UvTransform t;
  if(!info.contains("extensions"))
    return t;
  const auto& tt = info["extensions"]["KHR_texture_transform"];
  t.scale        = {tt["scale"][0].get<float>(), tt["scale"][1].get<float>()};
  t.offset       = {tt["offset"][0].get<float>(), tt["offset"][1].get<float>()};
  t.rotation     = tt.value("rotation", 0.f);
  return t;
}

void expectSame(const fv::mat::UvTransform& a, const fv::mat::UvTransform& b, const std::string& what)
{
  EXPECT_NEAR(a.scale[0], b.scale[0], 1e-5f) << what;
  EXPECT_NEAR(a.scale[1], b.scale[1], 1e-5f) << what;
  EXPECT_NEAR(a.offset[0], b.offset[0], 1e-5f) << what;
  EXPECT_NEAR(a.offset[1], b.offset[1], 1e-5f) << what;
  EXPECT_NEAR(a.rotation, b.rotation, 1e-5f) << what;
}

}  // namespace

TEST(Materials, TheClothesEditsAreEditsLikeTheRest)
{
  using E = fv::mat::Edits;
  const std::vector<std::function<void(E&)>> setters = {
      [](E& e) { e.baseMap = "/a.png"; },
      [](E& e) { e.opacityMap = ""; },
      [](E& e) { e.normalMap = "/n.png"; },
      [](E& e) { e.bumpMap = "/b.png"; },
      [](E& e) { e.roughnessMap = "/r.png"; },
      [](E& e) { e.uvScaleU = 2; },
      [](E& e) { e.uvScaleV = 2; },
      [](E& e) { e.uvOffsetU = 0.5f; },
      [](E& e) { e.uvOffsetV = 0.5f; },
      [](E& e) { e.uvRotation = 45; },
      [](E& e) { e.fabric = "denim"; },
      [](E& e) { e.fabricScale = 2; },
      [](E& e) { e.fabricStrength = 0.5f; },
      [](E& e) { e.sheenColor = fv::mat::Rgb{1, 0, 0}; },
      [](E& e) { e.sheenRoughness = 0.2f; },
  };
  for(size_t i = 0; i < setters.size(); i++)
  {
    E one;
    setters[i](one);
    EXPECT_FALSE(one.empty()) << i;
    E merged;
    merged.merge(one);
    EXPECT_EQ(fv::mat::editedKeys(merged), fv::mat::editedKeys(one)) << i;
    EXPECT_EQ(fv::mat::editedKeys(one).size(), 1u) << i;
  }
  EXPECT_TRUE(E{}.empty());
}

// Iray Uber's own Horizontal/Vertical Tiles and Offset are the surface's tiling; PBRSkin's Detail
// ones stay the detail layer's.
TEST(Materials, ReadsAnIrayUbersSurfaceTiling)
{
  const auto s = fv::dson::parseSurface("Jeans", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/denim.jpg"}},
    {"channel":{"id":"Horizontal Tiles","type":"float","current_value":4}},
    {"channel":{"id":"Vertical Tiles","type":"float","current_value":0}},
    {"channel":{"id":"Horizontal Offset","type":"float","current_value":0.25}},
    {"channel":{"id":"Vertical Offset","type":"float","current_value":0.5}}]}]})"));
  EXPECT_FLOAT_EQ(s.tiles[0], 4);
  EXPECT_FLOAT_EQ(s.tiles[1], 1);  // 0 tiles is none: untiled
  EXPECT_FLOAT_EQ(s.offset[0], 0.25f);
  EXPECT_FLOAT_EQ(s.offset[1], 0.5f);
  const auto lowered = fv::mat::lower(s, {}, {});
  expectSame(transformOf(lowered.material["pbrMetallicRoughness"]["baseColorTexture"]), {{4, 1}, {0.25f, 0.5f}, 0}, "base");

  const auto plain = fv::dson::parseSurface("Shirt", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/shirt.jpg"}}]}]})"));
  EXPECT_EQ(plain.tiles, (std::array<float, 2>{1, 1}));
  EXPECT_FALSE(fv::mat::lower(plain, {}, {}).material["pbrMetallicRoughness"]["baseColorTexture"].contains("extensions"));
}

// A slot's image replaced asks for the new image; one taken away is gone, and without its cutout a
// surface is opaque and single-sided again. The texture list says which are edits.
TEST(Materials, AMapEditReplacesOrRemovesASlot)
{
  const fv::mat::Surface s = garment();
  fv::mat::Edits         e;
  e.baseMap          = "/Runtime/Textures/FigureViewer/User/0123456789abcdef/denim.png";
  e.roughnessMap     = "/Runtime/Textures/FigureViewer/User/0123456789abcdef/rough.png";
  e.bumpMap          = "/Runtime/Textures/FigureViewer/User/0123456789abcdef/bump.png";
  const auto lowered = fv::mat::lower(s, e, {});
  const auto has     = [&](const fv::mat::Lowered& l, const std::string& rel) {
    return std::any_of(l.requests.begin(), l.requests.end(), [&](const fv::mat::TextureRequest& r) { return r.rel == rel; });
  };
  const auto& pbr  = lowered.material["pbrMetallicRoughness"];
  const auto  base = lowered.requests[pbr["baseColorTexture"]["slot"].get<size_t>()];
  EXPECT_EQ(base.rel, *e.baseMap);
  EXPECT_EQ(base.alphaRel, s.opacity.map);  // the cutout still rides in its alpha
  EXPECT_EQ(lowered.requests[pbr["metallicRoughnessTexture"]["slot"].get<size_t>()].rel, *e.roughnessMap);
  const auto normal = lowered.requests[lowered.material["normalTexture"]["slot"].get<size_t>()];
  EXPECT_EQ(normal.rel, s.normalMap);
  EXPECT_EQ(normal.bumpRel, *e.bumpMap);  // a bump map given to a surface with a normal map is blended in
  EXPECT_FALSE(has(lowered, s.base.map));
  EXPECT_EQ(lowered.material["alphaMode"], "BLEND");

  fv::mat::Edits removed;
  removed.opacityMap = "";
  removed.normalMap  = "";
  const auto bare    = fv::mat::lower(s, removed, {});
  EXPECT_EQ(bare.material["alphaMode"], "OPAQUE");
  EXPECT_FALSE(bare.material["doubleSided"].get<bool>());
  EXPECT_TRUE(bare.requests[bare.material["pbrMetallicRoughness"]["baseColorTexture"]["slot"].get<size_t>()].alphaRel.empty());
  EXPECT_FALSE(bare.material.contains("normalTexture"));
  EXPECT_FALSE(bare.material["extensions"]["KHR_materials_clearcoat"].contains("clearcoatNormalTexture"));
  EXPECT_FALSE(has(bare, s.opacity.map));
  removed.alphaMode = "blend";  // an alpha edit still decides
  EXPECT_EQ(fv::mat::lower(s, removed, {}).material["alphaMode"], "BLEND");

  const auto slots = fv::mat::mapSlots(s, removed);
  const auto slot  = [&](const char* name) {
    const auto it = std::find_if(slots.begin(), slots.end(), [&](const fv::mat::MapSlot& m) { return m.slot == name; });
    return it == slots.end() ? fv::mat::MapSlot{} : *it;
  };
  EXPECT_EQ(slot("base").rel, s.base.map);
  EXPECT_FALSE(slot("base").edited);
  EXPECT_EQ(slot("opacity").rel, "");
  EXPECT_TRUE(slot("opacity").edited);
  EXPECT_EQ(slot("opacity").label, "Cutout");
  EXPECT_EQ(slot("normal").rel, "");
  EXPECT_TRUE(slot("normal").edited);
  EXPECT_EQ(slot("roughness").rel, s.glossy.roughness.map);
  EXPECT_EQ(slot("emission").rel, s.emission->color.map);
  EXPECT_TRUE(slot("bump").slot.empty());  // no bump map, no edit: not listed
  EXPECT_EQ(fv::mat::mapSlots(s, e).at(0).rel, *e.baseMap);
}

// The panel's tiling lies over the authored one on every map that samples the surface's UVs, the
// clearcoat's copy of the normal included; a detail layer's own maps keep their tiling, its weight
// map follows the surface. Under flipV each is mirrored.
TEST(Materials, TheUvTransformGoesOnEverySurfaceSlotButTheDetail)
{
  fv::mat::Surface s = garment();
  s.tiles            = {2, 3};
  s.offset           = {0.1f, 0.2f};
  fv::mat::DetailLayer d;
  d.map          = "/Runtime/Textures/pores.jpg";
  d.weight       = 1;
  d.weightMap    = "/Runtime/Textures/pores_w.jpg";
  d.tiles        = {8, 8};
  d.roughnessMap = "/Runtime/Textures/pores_r.jpg";
  s.detail       = d;
  fv::mat::Edits e;
  e.uvScaleU   = 1.5f;
  e.uvOffsetV  = 0.25f;
  e.uvRotation = 30;

  const fv::mat::UvTransform surface = fv::mat::surfaceUv(s, e);
  EXPECT_NEAR(surface.scale[0], 3, 1e-6f);
  EXPECT_NEAR(surface.scale[1], 3, 1e-6f);
  EXPECT_NEAR(surface.rotation, 30 * 3.14159265f / 180, 1e-6f);
  const fv::mat::UvTransform detail{{8, 8}, {0, 0}, 0};

  for(const bool flipV : {false, true})
  {
    fv::mat::LowerOptions o;
    o.flipV            = flipV;
    const auto lowered = fv::mat::lower(s, e, o);
    std::vector<const nlohmann::json*> infos;
    slotsOf(lowered.material, infos);
    EXPECT_GE(infos.size(), 8u);  // base, mr, normal, emissive, the clearcoat's normal, the detail's three
    for(const nlohmann::json* info : infos)
    {
      const std::string          rel  = lowered.requests[(*info)["slot"].get<size_t>()].rel;
      const fv::mat::UvTransform want = rel == d.map || rel == d.roughnessMap ? detail : surface;
      expectSame(transformOf(*info), flipV ? fv::mat::mirroredV(want) : want, rel + (flipV ? " flipped" : ""));
    }
  }
  // a transform changes no image: nothing to cook again
  const auto plain = fv::mat::lower(s, {}, {}), moved = fv::mat::lower(s, e, {});
  ASSERT_EQ(plain.requests.size(), moved.requests.size());
  for(size_t i = 0; i < plain.requests.size(); i++)
    EXPECT_EQ(fv::mat::requestKey(plain.requests[i]), fv::mat::requestKey(moved.requests[i]));
}

// The transforms against the renderer's own matrix (KHR_texture_transform::updateTransform, as the
// shader applies it): the edit lies after the authored tiling, and the flipV mirror samples the
// same texel from V turned over on both sides.
TEST(Materials, TheUvTransformComposesAndMirrorsInTheRenderersMatrix)
{
  const auto apply = [](const fv::mat::UvTransform& t, glm::vec2 uv) {
    KHR_texture_transform k;
    k.offset   = {t.offset[0], t.offset[1]};
    k.rotation = t.rotation;
    k.scale    = {t.scale[0], t.scale[1]};
    k.updateTransform();
    const glm::vec3 p{uv, 1};
    return glm::vec2(glm::dot(k.uvTransform[0], p), glm::dot(k.uvTransform[1], p));
  };
  fv::mat::Surface s;
  s.tiles  = {4, 2};
  s.offset = {0.3f, -0.1f};
  fv::mat::Edits e;
  e.uvScaleU   = 0.5f;
  e.uvScaleV   = 3;
  e.uvOffsetU  = 0.2f;
  e.uvOffsetV  = 0.7f;
  e.uvRotation = 40;
  const fv::mat::UvTransform edit{{0.5f, 3}, {0.2f, 0.7f}, 40 * 3.14159265f / 180};
  const fv::mat::UvTransform t = fv::mat::surfaceUv(s, e);
  for(const glm::vec2 uv : {glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0.3f, 0.8f), glm::vec2(-2, 5)})
  {
    const glm::vec2 authored = uv * glm::vec2(4, 2) + glm::vec2(0.3f, -0.1f);
    const glm::vec2 want = apply(edit, authored), got = apply(t, uv);
    EXPECT_NEAR(got.x, want.x, 1e-4f);
    EXPECT_NEAR(got.y, want.y, 1e-4f);
    // turned over: (u, 1 - v) in, (u', 1 - v') out
    const glm::vec2 mirrored = apply(fv::mat::mirroredV(t), {uv.x, 1 - uv.y});
    EXPECT_NEAR(mirrored.x, got.x, 1e-4f);
    EXPECT_NEAR(mirrored.y, 1 - got.y, 1e-4f);
  }
  // no rotation: the mirror the detail layers always had, offset 1 - tiles - offset
  const fv::mat::UvTransform m = fv::mat::mirroredV({{8, 4}, {0.5f, 0.25f}, 0});
  EXPECT_NEAR(m.offset[0], 0.5f, 1e-6f);
  EXPECT_NEAR(m.offset[1], 1 - 4 - 0.25f, 1e-6f);
  EXPECT_TRUE(fv::mat::surfaceUv({}, {}).identity());
}

// A source image in a figure built for the cooked, bottom-up ones (a cold texture cache): its slot
// samples it with V turned over after the surface's own transform, in the renderer's matrix, and
// only that slot; a slot with no transform of its own gets the bare flip.
TEST(Materials, AnImageWhoseRowsRunTheOtherWayIsSampledWithItsVTurnedOver)
{
  const auto apply = [](const fv::mat::UvTransform& t, glm::vec2 uv) {
    KHR_texture_transform k;
    k.offset   = {t.offset[0], t.offset[1]};
    k.rotation = t.rotation;
    k.scale    = {t.scale[0], t.scale[1]};
    k.updateTransform();
    const glm::vec3 p{uv, 1};
    return glm::vec2(glm::dot(k.uvTransform[0], p), glm::dot(k.uvTransform[1], p));
  };
  const fv::mat::UvTransform t{{3, 2}, {0.25f, -0.4f}, 0.7f};
  for(const glm::vec2 uv : {glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0.3f, 0.8f), glm::vec2(-2, 5)})
  {
    const glm::vec2 want = apply(t, uv), got = apply(fv::mat::rowsFlipped(t), uv);
    EXPECT_NEAR(got.x, want.x, 1e-4f);
    EXPECT_NEAR(got.y, 1 - want.y, 1e-4f);
  }
  expectSame(fv::mat::rowsFlipped({}), {{1, -1}, {0, 1}, 0}, "bare");

  // through toGltf: the base map's source runs the other way, the others are cooked
  const auto infoTransform = [](const tinygltf::TextureInfo& info) {
    fv::mat::UvTransform out;
    const auto           it = info.extensions.find("KHR_texture_transform");
    if(it == info.extensions.end())
      return out;
    const tinygltf::Value& tt = it->second;
    out.scale    = {float(tt.Get("scale").Get(0).GetNumberAsDouble()), float(tt.Get("scale").Get(1).GetNumberAsDouble())};
    out.offset   = {float(tt.Get("offset").Get(0).GetNumberAsDouble()), float(tt.Get("offset").Get(1).GetNumberAsDouble())};
    out.rotation = tt.Has("rotation") ? float(tt.Get("rotation").GetNumberAsDouble()) : 0.f;
    return out;
  };
  fv::mat::Surface tiled = garment();
  tiled.tiles            = {2, 3};
  for(const fv::mat::Surface& s : {garment(), tiled})
  {
    const fv::mat::UvTransform own     = fv::mat::surfaceUv(s, {});
    const auto                 lowered = fv::mat::lower(s, {}, {});
    std::vector<std::string>   used;
    const auto                 m = fv::mat::toGltf(
        lowered,
        [&](const fv::mat::TextureRequest& r) {
          const bool base = r.rel == s.base.map;
          return fv::mat::ResolvedTexture{base ? 1 : 2, false, base};
        },
        used);
    expectSame(infoTransform(m.pbrMetallicRoughness.baseColorTexture), fv::mat::rowsFlipped(own), "base");
    expectSame(infoTransform(m.emissiveTexture), own, "emissive");
    EXPECT_NE(std::find(used.begin(), used.end(), "KHR_texture_transform"), used.end());
  }
}

// A fabric is a detail layer: its height map at its physical size from metresPerUv, its relief
// fixed (a scale changes the tiling, never the image), its roughness multiplier, weight 1. A file
// for other tools leaves it out, as it does every detail.
TEST(Materials, AFabricIsADetailLayerOfItsOwnSize)
{
  const fv::mat::Fabric* denim = fv::mat::findFabric("denim");
  ASSERT_NE(denim, nullptr);
  for(const char* key : {"denim", "canvas", "linen", "knit", "satin", "corduroy", "leather"})
    EXPECT_NE(fv::mat::findFabric(key), nullptr) << key;
  EXPECT_EQ(fv::mat::fabrics().size(), 7u);
  EXPECT_EQ(fv::mat::fabricHeightMap(*denim), "/Runtime/Textures/FigureViewer/Fabrics/v1/denim-height.png");
  EXPECT_EQ(fv::mat::fabricRoughnessMap(*denim), "/Runtime/Textures/FigureViewer/Fabrics/v1/denim-rough.png");

  fv::mat::Surface s = garment();
  s.tiles            = {5, 5};  // the garment's tiling is not the weave's
  fv::mat::Edits e;
  e.fabric         = "denim";
  e.fabricStrength = 0.7f;
  fv::mat::LowerOptions o;
  o.metresPerUv      = 0.8f;
  const auto  lowered = fv::mat::lower(s, e, o);
  const auto& de      = lowered.material["extensions"]["FV_materials_detail"];
  EXPECT_FLOAT_EQ(de["detailWeightFactor"].get<float>(), 1);
  EXPECT_FLOAT_EQ(de["detailRoughnessFactor"].get<float>(), denim->roughness);
  EXPECT_FALSE(de.contains("detailWeightTexture"));
  const auto& normal = de["detailNormalTexture"];
  EXPECT_FLOAT_EQ(normal["scale"].get<float>(), 0.7f);
  const float tiles = 0.8f / denim->tileMetres;
  expectSame(transformOf(normal), {{tiles, tiles}, {0, 0}, 0}, "normal");
  expectSame(transformOf(de["detailRoughnessTexture"]), {{tiles, tiles}, {0, 0}, 0}, "roughness");
  const fv::mat::TextureRequest& r = lowered.requests[normal["slot"].get<size_t>()];
  EXPECT_EQ(r.rel, fv::mat::fabricHeightMap(*denim));
  EXPECT_EQ(r.channels, "bn");
  EXPECT_FLOAT_EQ(r.bumpHeight, denim->relief);
  EXPECT_EQ(lowered.requests[de["detailRoughnessTexture"]["slot"].get<size_t>()].rel, fv::mat::fabricRoughnessMap(*denim));

  // twice as coarse: half the tiles, the same image
  e.fabricScale       = 2;
  const auto  coarse  = fv::mat::lower(s, e, o);
  const auto& cnormal = coarse.material["extensions"]["FV_materials_detail"]["detailNormalTexture"];
  expectSame(transformOf(cnormal), {{tiles / 2, tiles / 2}, {0, 0}, 0}, "coarse");
  EXPECT_EQ(fv::mat::requestKey(coarse.requests[cnormal["slot"].get<size_t>()]), fv::mat::requestKey(r));

  // no measured size: a fixed tiling
  const auto unknown = fv::mat::lower(s, e, {});
  EXPECT_GT(transformOf(unknown.material["extensions"]["FV_materials_detail"]["detailNormalTexture"]).scale[0], 1);

  o.forFile = true;
  EXPECT_FALSE(fv::mat::lower(s, e, o).material["extensions"].contains("FV_materials_detail"));
  e.fabric = "";
  EXPECT_FALSE(fv::mat::lower(s, e, {}).material["extensions"].contains("FV_materials_detail"));

  e.fabric         = "satin";
  const auto slots = fv::mat::mapSlots(s, e);
  ASSERT_FALSE(slots.empty());
  EXPECT_EQ(slots.back().slot, "detail");
  EXPECT_EQ(slots.back().rel, fv::mat::fabricHeightMap(*fv::mat::findFabric("satin")));
  EXPECT_TRUE(slots.back().edited);
}

TEST(Materials, ASheenEditIsKhrMaterialsSheen)
{
  const fv::mat::Surface s = garment();
  fv::mat::Edits         e;
  EXPECT_FALSE(fv::mat::lower(s, e, {}).material["extensions"].contains("KHR_materials_sheen"));
  e.sheenColor        = fv::mat::Rgb{0.5f, 0.25f, 0.125f};
  const auto  lowered = fv::mat::lower(s, e, {});
  const auto& sheen   = lowered.material["extensions"]["KHR_materials_sheen"];
  EXPECT_FLOAT_EQ(sheen["sheenColorFactor"][1].get<float>(), 0.25f);
  EXPECT_FLOAT_EQ(sheen["sheenRoughnessFactor"].get<float>(), 0.5f);
  e.sheenRoughness = 0.2f;
  EXPECT_FLOAT_EQ(fv::mat::lower(s, e, {}).material["extensions"]["KHR_materials_sheen"]["sheenRoughnessFactor"].get<float>(), 0.2f);
  e.sheenColor = fv::mat::Rgb{0, 0, 0};
  EXPECT_FALSE(fv::mat::lower(s, e, {}).material["extensions"].contains("KHR_materials_sheen"));
}

// material.set's values onto edits and back: what a surface publishes for a key is what material.set
// takes; a map is a library path, never a disk path; an unknown fabric or key is refused.
TEST(Materials, TheEditValuesRoundTrip)
{
  using fv::mat::EditsJson;
  const EditsJson in = {{"color", "#804020"},
                        {"roughness", 0.3},
                        {"baseMap", "/Runtime/Textures/x.png"},
                        {"opacityMap", ""},
                        {"normalMap", "Runtime\\Textures\\n.png"},
                        {"uvScaleU", 2.5},
                        {"uvScaleV", 1000},
                        {"uvOffsetU", -0.5},
                        {"uvRotation", 90},
                        {"fabric", "corduroy"},
                        {"fabricScale", 1.5},
                        {"fabricStrength", 0.4},
                        {"sheenColor", "#336699"},
                        {"sheenRoughness", 0.25}};
  fv::mat::Edits e;
  for(auto it = in.begin(); it != in.end(); ++it)
    fv::mat::applyValue(e, it.key(), it.value());
  EXPECT_EQ(*e.normalMap, "/Runtime/Textures/n.png");
  EXPECT_FLOAT_EQ(*e.uvScaleV, 100);  // clamped

  EditsJson values = EditsJson::object();
  fv::mat::clothesValues(e, values);
  values["color"]        = fv::mat::hexOf(*e.color);
  values["roughness"]    = *e.roughness;
  const EditsJson edited = fv::mat::editedKeys(e);
  EXPECT_EQ(edited.size(), in.size());
  // the values not edited: null or their defaults
  EXPECT_TRUE(values["bumpMap"].is_null());
  EXPECT_FLOAT_EQ(values["uvOffsetV"].get<float>(), 0);
  // the published values set again give the same edits
  fv::mat::Edits again;
  for(const auto& k : edited)
    fv::mat::applyValue(again, k.get<std::string>(), values[k.get<std::string>()]);
  EditsJson republished = EditsJson::object();
  fv::mat::clothesValues(again, republished);
  for(auto it = republished.begin(); it != republished.end(); ++it)
    EXPECT_EQ(it.value(), values[it.key()]) << it.key();
  EXPECT_EQ(fv::mat::editedKeys(again), edited);
  EXPECT_EQ(values["sheenColor"], "#336699");
  EXPECT_EQ(values["opacityMap"], "");

  // null drops an edit
  fv::mat::applyValue(again, "baseMap", nullptr);
  fv::mat::applyValue(again, "fabric", nullptr);
  EXPECT_FALSE(again.baseMap);
  EXPECT_FALSE(again.fabric);

  for(const char* bad : {"C:/Users/me/denim.png", "\\\\server\\share\\denim.png", "/Runtime/../../denim.png", "..", "/a|b.png", "file:///x.png"})
    EXPECT_THROW(fv::mat::applyValue(e, "baseMap", bad), std::runtime_error) << bad;
  EXPECT_THROW(fv::mat::applyValue(e, "baseMap", 3), std::runtime_error);
  EXPECT_THROW(fv::mat::applyValue(e, "fabric", "tweed"), std::runtime_error);
  EXPECT_THROW(fv::mat::applyValue(e, "sheenColor", "red"), std::runtime_error);
  EXPECT_THROW(fv::mat::applyValue(e, "uvScale", 2), std::runtime_error);
  fv::mat::applyValue(e, "roughnessMap", "/Runtime/Textures/a..b.png");  // dots in a name are no way up
  EXPECT_EQ(*e.roughnessMap, "/Runtime/Textures/a..b.png");

  for(const char* k : {"baseMap", "roughnessMap", "uvScaleU", "uvRotation", "fabric", "fabricStrength"})
    EXPECT_TRUE(fv::mat::figureOnlyKey(k)) << k;
  for(const char* k : {"sheenColor", "sheenRoughness", "color", "roughness"})
    EXPECT_FALSE(fv::mat::figureOnlyKey(k)) << k;
}

// Legacy (3Delight) surfaces: the channels of Age of Armour's Subsurface shader, as in
// data/Age of Armour/Subsurface Shaders/AoA_Subsurface/AoA_Subsurface.dsf, where labels name other
// channels' ids, and the id-only layer a figure's scene lays over them (Genesis 3 Female.duf, Torso).
namespace {

const char* kAoaBrick = R"({"extra":[{"type":"studio/material/daz_brick","channels":[
  {"channel":{"id":"Ambient Color","label":"Subsurface Color","type":"color","current_value":[1,0.8,0.7],"image_file":"/Runtime/Textures/torso_d.jpg"}},
  {"channel":{"id":"Ambient Strength","label":"Subsurface Strength","type":"float","current_value":0.75}},
  {"channel":{"id":"color4","label":"Ambient Color","type":"color","current_value":[0,0,0]}},
  {"channel":{"id":"Ambient Strength2","label":"Ambient Strength","type":"float","current_value":0}},
  {"channel":{"id":"roughness","label":"Velvet Roughness","type":"float","current_value":0.2}},
  {"channel":{"id":"Glossiness","label":"Specular Glossiness","type":"float","current_value":0.65}},
  {"channel":{"id":"sssmaterial","label":"SSS Material Type","type":"enum","current_value":9}}
]}]})";

const char* kAoaSceneLayer = R"({"extra":[{"channels":[
  {"channel":{"id":"Ambient Color","current_value":[0.9,0.7,0.6]}},
  {"channel":{"id":"color4","current_value":[0,0,0]}},
  {"channel":{"id":"Ambient Strength2","current_value":0}}
]}]})";

// Each layer collected over the ones before, as figure_loader merges a material's layers
fv::dson::Channels layered(std::initializer_list<const char*> layers)
{
  fv::dson::Channels ch;
  for(const char* text : layers)
  {
    const Json def = Json::parse(text);
    fv::dson::collectChannels(&def, ch, {}, "x.dsf");
  }
  return ch;
}

}  // namespace

TEST(Materials, AnIdOutranksAnotherChannelsLabel)
{
  const auto brick = layered({kAoaBrick});
  // "color4", labelled "Ambient Color", is not the channel whose id is "Ambient Color"
  const auto* sss = brick.find({"Subsurface Color"});
  ASSERT_NE(sss, nullptr);
  EXPECT_EQ(sss->value, Json::parse("[1,0.8,0.7]"));
  EXPECT_EQ(brick.get("ambientcolor"), sss);
  EXPECT_EQ(brick.get("color4")->value, Json::parse("[0,0,0]"));
  EXPECT_EQ(brick.find({"Subsurface Strength"})->value, 0.75);
  EXPECT_EQ(brick.get("ambientstrength2")->value, 0);

  // the scene's id-only layer reaches each by its id: the subsurface colour keeps its map
  const auto scene = layered({kAoaBrick, kAoaSceneLayer});
  const auto* sss2 = scene.find({"Subsurface Color"});
  ASSERT_NE(sss2, nullptr);
  EXPECT_EQ(sss2->value, Json::parse("[0.9,0.7,0.6]"));
  EXPECT_EQ(sss2->image, std::optional<std::string>("/Runtime/Textures/torso_d.jpg"));
  EXPECT_EQ(scene.find({"Subsurface Strength"})->value, 0.75);
}

TEST(Materials, AnOverrideByIdStillReplacesEveryName)
{
  // a figure's "diffuse"/"Base Color" overridden by a character's "diffuse"/"Diffuse Color"
  const auto ch = layered({R"({"extra":[{"channels":[{"channel":{"id":"diffuse","label":"Base Color","current_value":[1,1,1]}}]}]})",
                           R"({"extra":[{"channels":[{"channel":{"id":"diffuse","label":"Diffuse Color","current_value":[0.5,0.4,0.3]}}]}]})"});
  const auto* byLabel = ch.find({"Base Color"});
  ASSERT_NE(byLabel, nullptr);
  EXPECT_EQ(byLabel->value, Json::parse("[0.5,0.4,0.3]"));
  EXPECT_EQ(ch.find({"Diffuse Color"}), byLabel);
  EXPECT_EQ(ch.find({"diffuse"}), byLabel);
}

TEST(Materials, AoaRoughnessIsNotVelvet)
{
  // AoA's id "roughness" is its Velvet Roughness; the specular lobe's comes from Glossiness
  const auto s = fv::dson::parseSurface("Torso", layered({kAoaBrick}));
  EXPECT_NE(s.glossy.roughness.value, 0.2f);
}

// The three 3Delight-era shaders, with values from real files: the G8M Basic Wear Boxers and the
// G3F Basic Wear Top (DAZ Default), Genesis 2 Female's Torso (UberSurface), Genesis 3 Female's
// Torso (AoA Subsurface, as in kAoaBrick).
namespace {

const char* kBoxers = R"({
  "diffuse":{"channel":{"id":"diffuse","label":"Diffuse Color","current_value":[0.7529412,0.7529412,0.7529412],"image_file":"/Runtime/Textures/boxers_d.jpg"}},
  "diffuse_strength":{"channel":{"id":"diffuse_strength","label":"Diffuse Strength","current_value":1}},
  "specular":{"channel":{"id":"specular","label":"Specular Color","current_value":[0.6,0.6,0.6]}},
  "specular_strength":{"channel":{"id":"specular_strength","label":"Specular Strength","current_value":0.3}},
  "glossiness":{"channel":{"id":"glossiness","label":"Glossiness","current_value":1}},
  "ambient":{"channel":{"id":"ambient","label":"Ambient Color","current_value":[0,0,0]}},
  "ambient_strength":{"channel":{"id":"ambient_strength","label":"Ambient Strength","current_value":1}},
  "reflection_strength":{"channel":{"id":"reflection_strength","label":"Reflection Strength","current_value":0}},
  "refraction_strength":{"channel":{"id":"refraction_strength","label":"Refraction Strength","current_value":0}},
  "ior":{"channel":{"id":"ior","label":"Index of Refraction","current_value":0}},
  "bump_min":{"channel":{"id":"bump_min","label":"Negative Bump","current_value":-0.01}},
  "bump_max":{"channel":{"id":"bump_max","label":"Positive Bump","current_value":0.01}},
  "transparency":{"channel":{"id":"transparency","label":"Opacity Strength","current_value":1}},
  "u_scale":{"channel":{"id":"u_scale","label":"Horizontal Tiles","current_value":1}},
  "extra":[{"type":"studio_material_channels","channels":[
    {"channel":{"id":"Lighting Model","label":"Lighting Model","current_value":0}}]}]})";

const char* kG2fTorso = R"({
  "diffuse":{"channel":{"id":"diffuse","label":"Diffuse Color","current_value":[1,1,1],"image_file":"/Runtime/Textures/g2f_torso.jpg"}},
  "transparency":{"channel":{"id":"transparency","label":"Opacity Strength","current_value":1}},
  "extra":[{"type":"studio/material/daz_shader","definition":"support/DAZ/Uber/shaderDefinitions/surface/omUberSurfaceDef.dse","channels":[
    {"channel":{"id":"Bump Active","current_value":true}},
    {"channel":{"id":"Bump Strength","current_value":0.25,"image_file":"/Runtime/Textures/g2f_torso_b.jpg"}},
    {"channel":{"id":"Bump Minimum","current_value":-0.1}},
    {"channel":{"id":"Bump Maximum","current_value":0.1}},
    {"channel":{"id":"Diffuse Active","current_value":true}},
    {"channel":{"id":"Diffuse Strength","current_value":0.9}},
    {"channel":{"id":"Opacity Active","current_value":true}},
    {"channel":{"id":"Map Tiling U","current_value":1}},
    {"channel":{"id":"Map Tiling V","current_value":1}},
    {"channel":{"id":"Specular Active","current_value":true}},
    {"channel":{"id":"Specular Color","current_value":[1,0.7882353,0.6627451]}},
    {"channel":{"id":"Specular Strength","current_value":0.55}},
    {"channel":{"id":"Glossiness","current_value":0.2}},
    {"channel":{"id":"Specular2 Active","current_value":true}},
    {"channel":{"id":"Specular2 Color","current_value":[0.745098,0.8627451,1]}},
    {"channel":{"id":"Specular2 Strength","current_value":0.3}},
    {"channel":{"id":"Specular2 Roughness","current_value":0.1}},
    {"channel":{"id":"Ambient Active","current_value":false}},
    {"channel":{"id":"Refraction Active","current_value":false}},
    {"channel":{"id":"Index of Refraction","current_value":1.3}},
    {"channel":{"id":"Reflection Active","current_value":false}},
    {"channel":{"id":"Reflection Strength","current_value":1}},
    {"channel":{"id":"Velvet Active","current_value":true}},
    {"channel":{"id":"Subsurface Active","current_value":true}},
    {"channel":{"id":"Subsurface Strength","current_value":0.5}}]}]})";

// The same surface switched to plain cloth: no subsurface, no bump, tiled twice across
const char* kUberCloth = R"({"extra":[{"channels":[
    {"channel":{"id":"Subsurface Active","current_value":false}},
    {"channel":{"id":"Bump Active","current_value":false}},
    {"channel":{"id":"Map Tiling U","current_value":2}},
    {"channel":{"id":"Map Tiling V","current_value":2}}]}]})";

// AoA's G3F Torso values over its brick: specular .25 at glossiness .4, specular 2 at .1, a normal map
const char* kAoaTorso = R"({"extra":[{"channels":[
    {"channel":{"id":"diffuse","current_value":[1,1,1],"image_file":"/Runtime/Textures/g3f_torso.jpg"}},
    {"channel":{"id":"Diffuse Strength","current_value":0.5}},
    {"channel":{"id":"value23","current_value":1}},
    {"channel":{"id":"Specular Strength","current_value":0.25}},
    {"channel":{"id":"Glossiness","current_value":0.4}},
    {"channel":{"id":"value222","current_value":0.1}},
    {"channel":{"id":"Normal Map","current_value":1,"image_file":"/Runtime/Textures/g3f_torso_n.jpg"}},
    {"channel":{"id":"alpha","current_value":0.6}}]}]})";

const char* kTopValues = R"({"extra":[{"channels":[
    {"channel":{"id":"specular_strength","current_value":1}},{"channel":{"id":"glossiness","current_value":0.75}}]}]})";
const char* kMetallic  = R"({"extra":[{"channels":[{"channel":{"id":"Lighting Model","current_value":1}}]}]})";
const char* kMatte     = R"({"extra":[{"channels":[{"channel":{"id":"Lighting Model","current_value":4}}]}]})";
const char* kRefracts  = R"({"extra":[{"channels":[{"channel":{"id":"refraction_strength","current_value":0.8}}]}]})";
const char* kIrayOver  = R"({"extra":[{"channels":[{"channel":{"id":"Glossy Roughness","current_value":0.4}}]}]})";
const char* kUberLens  = R"({"transparency":{"channel":{"id":"transparency","current_value":0.1}},
    "extra":[{"channels":[{"channel":{"id":"Refraction Active","current_value":true}}]}]})";

}  // namespace

TEST(Materials, ClassifyLegacyKeepsIrayFirst)
{
  using fv::dson::LegacyShader;
  EXPECT_EQ(fv::dson::classifyLegacy(channelsOf(kBoxers)), LegacyShader::eDazDefault);
  EXPECT_EQ(fv::dson::classifyLegacy(channelsOf(kG2fTorso)), LegacyShader::eUberSurface);
  EXPECT_EQ(fv::dson::classifyLegacy(layered({kAoaBrick, kAoaTorso})), LegacyShader::eAoa);
  // Iray Uber has a Glossiness and a Diffuse Strength too
  EXPECT_EQ(fv::dson::classifyLegacy(channelsOf(kSkin)), LegacyShader::eNone);
  EXPECT_EQ(fv::dson::classifyLegacy(layered({kG2fTorso, kIrayOver})), LegacyShader::eNone);
}

TEST(Materials, LegacyRoughnessCurve)
{
  using fv::dson::LegacyShader;
  using fv::dson::legacyRoughness;
  // DAZ Default narrows over the top half only: glossiness 1 is the floor, half is fully rough
  EXPECT_NEAR(legacyRoughness(LegacyShader::eDazDefault, 1, false), 0.3f, 1e-3f);
  EXPECT_NEAR(legacyRoughness(LegacyShader::eDazDefault, 0.75f, false), 0.63f, 0.01f);
  EXPECT_EQ(legacyRoughness(LegacyShader::eDazDefault, 0.4f, false), 1.f);
  EXPECT_NEAR(legacyRoughness(LegacyShader::eUberSurface, 0.5f, false), 0.5f, 1e-4f);
  // the same look: Default's 85% and UberSurface's 50%
  EXPECT_NEAR(legacyRoughness(LegacyShader::eDazDefault, 0.85f, false), legacyRoughness(LegacyShader::eUberSurface, 0.5f, false), 0.02f);
  float last = 2;
  for(int i = 0; i <= 20; i++)
  {
    const float r = legacyRoughness(LegacyShader::eUberSurface, i / 20.f, false);
    EXPECT_LE(r, last);
    EXPECT_GE(r, 0.3f);
    last = r;
  }
  // only a surface that reflects goes below the floor
  EXPECT_NEAR(legacyRoughness(LegacyShader::eUberSurface, 0.95f, true), 0.05f, 1e-4f);
  EXPECT_EQ(legacyRoughness(LegacyShader::eUberSurface, 0.95f, false), 0.3f);
}

TEST(Materials, DazDefaultTranslates)
{
  const auto s = fv::dson::parseSurface("Boxers", channelsOf(kBoxers));
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::eDazDefault);
  // DAZ's own Iray Boxers: Glossy Layered Weight 0.325 on a rough cotton, not a mirror
  EXPECT_NEAR(s.glossy.weight, 0.3f, 1e-4f);
  EXPECT_NEAR(s.glossy.roughness.value, 0.3f, 1e-3f);
  EXPECT_EQ(s.glossy.reflectivity, 0.5f);
  EXPECT_FALSE(s.refraction.has_value());  // ior 0 and no refraction strength
  EXPECT_EQ(s.metallic.value, 0.f);
  EXPECT_EQ(s.base.map, "/Runtime/Textures/boxers_d.jpg");
  EXPECT_NEAR(s.base.color[0], fv::mat::srgbToLinear(0.7529412), 1e-4f);
  EXPECT_FALSE(s.emission.has_value());  // ambient is no light

  // the G3F Basic Wear Top: full strength at glossiness 0.75 is a satin, not a plastic
  const auto top = fv::dson::parseSurface("Top", layered({kBoxers, kTopValues}));
  EXPECT_EQ(top.glossy.weight, 1.f);
  EXPECT_NEAR(top.glossy.roughness.value, 0.63f, 0.01f);

  // Lighting Model: Metallic is a metal, Matte fully rough; refraction takes an index of its own
  const auto metal = fv::dson::parseSurface("Buckle", layered({kBoxers, kMetallic}));
  EXPECT_EQ(metal.metallic.value, 1.f);
  EXPECT_LT(metal.glossy.roughness.value, 0.3f);
  const auto matte = fv::dson::parseSurface("Cloth", layered({kBoxers, kMatte}));
  EXPECT_EQ(matte.glossy.roughness.value, 1.f);
  const auto glass = fv::dson::parseSurface("Lens", layered({kBoxers, kRefracts}));
  ASSERT_TRUE(glass.refraction.has_value());
  EXPECT_NEAR(glass.refraction->weight, 0.8f, 1e-4f);
  EXPECT_EQ(glass.refraction->ior, 1.5f);
}

TEST(Materials, UberSurfaceTranslates)
{
  // skin: measured skin's specular, whatever the file's strengths and tints
  const auto torso = fv::dson::parseSurface("Torso", channelsOf(kG2fTorso));
  EXPECT_EQ(torso.shader, fv::mat::Surface::Shader::eUberSurface);
  EXPECT_EQ(torso.glossy.weight, 1.f);
  EXPECT_NEAR(torso.glossy.reflectivity, 0.35f, 1e-4f);
  EXPECT_NEAR(torso.glossy.roughness.value, 0.4f, 1e-4f);
  EXPECT_EQ(torso.base.color, (fv::mat::Rgb{1, 1, 1}));  // Diffuse Strength is not the skin's albedo
  EXPECT_EQ(torso.bumpMap, "/Runtime/Textures/g2f_torso_b.jpg");

  // cloth: both specular lobes in one, from Glossiness; the reflection is switched off
  const auto cloth = fv::dson::parseSurface("Shirt", layered({kG2fTorso, kUberCloth}));
  EXPECT_NEAR(cloth.glossy.weight, 0.85f, 1e-4f);
  EXPECT_NEAR(cloth.glossy.roughness.value, 0.8f, 1e-4f);
  EXPECT_NEAR(cloth.base.color[0], 0.9f, 1e-4f);
  EXPECT_EQ(cloth.tiles, (std::array<float, 2>{2, 2}));
  EXPECT_TRUE(cloth.bumpMap.empty());
  EXPECT_FALSE(cloth.refraction.has_value());

  // glass: what its opacity lets through refracts
  const auto lens = fv::dson::parseSurface("Lens", layered({kG2fTorso, kUberCloth, kUberLens}));
  ASSERT_TRUE(lens.refraction.has_value());
  EXPECT_NEAR(lens.refraction->weight, 0.9f, 1e-4f);
  EXPECT_NEAR(lens.refraction->ior, 1.3f, 1e-4f);
  EXPECT_EQ(lens.opacity.value, 1.f);
}

TEST(Materials, AoaTranslates)
{
  const auto torso = fv::dson::parseSurface("Torso", layered({kAoaBrick, kAoaTorso}));
  EXPECT_EQ(torso.shader, fv::mat::Surface::Shader::eAoa);
  EXPECT_NEAR(torso.glossy.roughness.value, 0.4f, 1e-4f);  // measured skin, not its velvet's 0.2
  EXPECT_EQ(torso.glossy.weight, 1.f);
  EXPECT_EQ(torso.base.color, (fv::mat::Rgb{1, 1, 1}));  // Diffuse Strength 0.5 is 3Delight's SSS split
  EXPECT_NEAR(torso.normalStrength, 0.6f, 1e-4f);       // "alpha", the Normal Map Strength

  // the same shader on a nail: its own specular, both lobes
  const auto nail = fv::dson::parseSurface("Fingernails", layered({kAoaBrick, kAoaTorso}));
  EXPECT_NEAR(nail.glossy.weight, 0.35f, 1e-4f);
  EXPECT_NEAR(nail.glossy.roughness.value, 0.6f, 1e-4f);
}

TEST(Materials, IrayPresetOverLegacyStillReadsIray)
{
  // Iray Optimized Genesis 2 Female MAT sets Iray channels over the UberSurface Torso
  fv::Material m = fv::dson::buildMaterial("Torso", channelsOf(kG2fTorso));
  ASSERT_EQ(m.surface.shader, fv::mat::Surface::Shader::eUberSurface);
  ASSERT_TRUE(fv::dson::overrideChannels(m, {{"Glossy Layered Weight", Json(1), std::nullopt},
                                             {"Glossy Reflectivity", Json(0.35), std::nullopt},
                                             {"Glossy Roughness", Json(0.45), std::nullopt},
                                             {"Translucency Weight", Json(0.5), std::nullopt}}));
  EXPECT_EQ(m.surface.shader, fv::mat::Surface::Shader::eIray);
  EXPECT_NEAR(m.surface.glossy.roughness.value, 0.45f, 1e-4f);
  EXPECT_NEAR(m.surface.glossy.reflectivity, 0.35f, 1e-4f);
  // the overrides name no shader: the files' own stays, though the channels now read as Iray
  EXPECT_EQ(m.surface.dazShader.type, "studio/material/daz_shader");
}

// The shader a material's files name, kept beside what its channels read as
namespace {

const char* kPbrSkinFile = "data/Daz 3D/Built-in Content/Daz Iray PBRSkin/PBRSkin.dsf";

// Genesis 9's own Body: the type and the channels in two extras, as the .duf writes them
const char* kG9Body = R"({"id":"Body","extra":[{"type":"studio/material/uber_iray","version":"1.1.0.0"},
  {"type":"studio_material_channels","channels":[{"channel":{"id":"Glossy Roughness","current_value":0.5}}]}]})";
// PBRSkin.dsf's library entry: the brick graph, no url
const char* kPbrSkinEntry = R"({"id":"PBRSkin","extra":[{"type":"studio/material/daz_brick","brick_settings":{}},
  {"type":"studio_material_channels","channels":[{"channel":{"id":"Specular Lobe 2 Roughness Mult","current_value":1}}]}]})";
// A character's scene material over it (Leyla 9 writes "DAZ 3D"), with an Octane extra beside
const char* kPbrSkinScene = R"({"id":"Body","url":"/data/DAZ%203D/Built-in%20Content/Daz%20Iray%20PBRSkin/PBRSkin.dsf#PBRSkin",
  "extra":[{"type":"studio/material/daz_brick"},{"type":"studio/element_data/OctaneRender"},
  {"type":"studio_material_channels","channels":[{"channel":{"id":"Detail Enable","current_value":true}}]}]})";

fv::dson::Channels collect(fv::dson::Channels ch, const char* text, const std::string& file)
{
  const Json def = Json::parse(text);
  fv::dson::collectChannels(&def, ch, {}, file);
  return ch;
}

}  // namespace

TEST(Materials, ChannelsKeepTheShaderTheirExtraNames)
{
  const auto skin = channelsOf(kSkin).shader();
  EXPECT_EQ(skin.type, "studio/material/uber_iray");
  EXPECT_EQ(skin.url, "");  // Iray Uber is defined inside DAZ Studio
  EXPECT_EQ(fv::dson::parseSurface("Face", channelsOf(kSkin)).shader, fv::mat::Surface::Shader::eIray);

  const auto torso = channelsOf(kG2fTorso).shader();
  EXPECT_EQ(torso.type, "studio/material/daz_shader");
  EXPECT_EQ(torso.url, "support/DAZ/Uber/shaderDefinitions/surface/omUberSurfaceDef.dse");

  // "studio_material_channels" holds channels and names no shader: DAZ Default names none
  EXPECT_EQ(channelsOf(kBoxers).shader().type, "");

  // a brick with neither graph nor url: the type alone, and AoA still reads as AoA
  const auto aoa = layered({kAoaBrick, kAoaTorso});
  EXPECT_EQ(aoa.shader().type, "studio/material/daz_brick");
  EXPECT_EQ(aoa.shader().url, "");
  const auto s = fv::dson::parseSurface("Torso", aoa);
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::eAoa);
  EXPECT_EQ(s.dazShader.type, "studio/material/daz_brick");
}

TEST(Materials, EachLayerNamesTheShaderAsItsFileSaysIt)
{
  // the geometry's Iray Uber, then PBRSkin.dsf's brick, then the character's instance of it
  auto ch = collect({}, kG9Body, "data/DAZ 3D/Genesis 9/Genesis 9.duf");
  EXPECT_EQ(ch.shader().type, "studio/material/uber_iray");
  EXPECT_EQ(ch.shader().url, "");
  ch = collect(ch, kPbrSkinEntry, kPbrSkinFile);
  EXPECT_EQ(ch.shader().type, "studio/material/daz_brick");
  EXPECT_EQ(ch.shader().url, "data/Daz 3D/Built-in Content/Daz Iray PBRSkin/PBRSkin.dsf#PBRSkin");
  ch = collect(ch, kPbrSkinScene, "data/DAZ 3D/Characters/Leyla 9.duf");
  EXPECT_EQ(ch.shader().type, "studio/material/daz_brick");  // the Octane extra names no shader
  EXPECT_EQ(ch.shader().url, "data/DAZ 3D/Built-in Content/Daz Iray PBRSkin/PBRSkin.dsf#PBRSkin");
  const auto skin = fv::dson::parseSurface("Body", ch);
  EXPECT_EQ(skin.shader, fv::mat::Surface::Shader::ePbrSkin);
  EXPECT_EQ(skin.dazShader, ch.shader());

  // a later layer naming another shader replaces it
  ch = collect(ch, kG9Body, "data/DAZ 3D/Genesis 9/Genesis 9.duf");
  EXPECT_EQ(ch.shader().type, "studio/material/uber_iray");
  EXPECT_EQ(ch.shader().url, "");

  // a 3Delight instance repeats the type without the definition: its library entry's stays
  const auto legs = collect(channelsOf(kG2fTorso), R"({"id":"Legs","url":"#Legs","extra":[{"type":"studio/material/daz_shader"}]})",
                            "data/DAZ 3D/Genesis 2/Female/Genesis 2 Base Female.duf");
  EXPECT_EQ(legs.shader().type, "studio/material/daz_shader");
  EXPECT_EQ(legs.shader().url, "support/DAZ/Uber/shaderDefinitions/surface/omUberSurfaceDef.dse");
}

TEST(Materials, PresetsKeepTheShaderTheFilesName)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Torso", channelsOf(kG2fTorso)));
  const Json preset = Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Torso:?Glossy%20Roughness/value","keys":[[0,0.45]]}
  ]}})");
  ASSERT_EQ(fv::dson::applyMaterialPreset(preset, scene).changed, std::set<int>({0}));
  const fv::Material& m = scene.materials[0];
  EXPECT_EQ(m.surface.shader, fv::mat::Surface::Shader::eIray);
  for(const fv::mat::ShaderRef& s : {m.surface.dazShader, m.channels->shader(), m.baseline->shader()})
  {
    EXPECT_EQ(s.type, "studio/material/daz_shader");
    EXPECT_EQ(s.url, "support/DAZ/Uber/shaderDefinitions/surface/omUberSurfaceDef.dse");
  }
}

TEST(Materials, SurfacesNamingDifferentShadersDoNotShareAMaterial)
{
  const char* uber  = R"({"extra":[{"type":"studio/material/uber_iray","channels":[{"channel":{"id":"diffuse","current_value":[1,1,1]}}]}]})";
  const char* brick = R"({"extra":[{"type":"studio/material/daz_brick","channels":[{"channel":{"id":"diffuse","current_value":[1,1,1]}}]}]})";
  EXPECT_NE(fv::dson::materialKey("Face", channelsOf(uber)), fv::dson::materialKey("Face", channelsOf(brick)));
  EXPECT_EQ(fv::dson::materialKey("Face", channelsOf(kSkin)), fv::dson::materialKey("Face", channelsOf(kSkin)));
}

TEST(Materials, LegacyBumpRangeScalesTheBakedHeight)
{
  // DAZ Default's Negative/Positive Bump: +-0.01 cm, four fifths of Iray's 0.25 mm bake
  const char* boxersBump = R"({"bump":{"channel":{"id":"bump","label":"Bump Strength","current_value":0.5,"image_file":"/Runtime/Textures/boxers_b.jpg"}}})";
  const auto  boxers     = fv::dson::parseSurface("Boxers", layered({kBoxers, boxersBump}));
  ASSERT_TRUE(boxers.bumpRange.has_value());
  EXPECT_NEAR(*boxers.bumpRange, 0.0002f, 1e-7f);
  fv::mat::LowerOptions o;
  o.metresPerUv    = 0.43f;
  const auto reqs  = fv::mat::lower(boxers, {}, o).requests;
  const auto bump  = std::find_if(reqs.begin(), reqs.end(), [](const fv::mat::TextureRequest& r) { return r.channels == "bn"; });
  ASSERT_NE(bump, reqs.end());
  EXPECT_NEAR(bump->bumpHeight, fv::mat::bumpHeightInUv(0.43f, 1, 0.0002f), 1e-7f);
  EXPECT_NEAR(bump->bumpHeight, fv::mat::bumpHeightInUv(0.43f) * 0.8f, fv::mat::bumpHeightInUv(0.43f) * 0.05f);

  // UberSurface's Bump Minimum/Maximum, +-0.1 cm: eight times Iray's bake
  const auto torso = fv::dson::parseSurface("Torso", channelsOf(kG2fTorso));
  ASSERT_TRUE(torso.bumpRange.has_value());
  EXPECT_NEAR(*torso.bumpRange, 0.002f, 1e-7f);

  // AoA's brick leaves both at 0, and Iray's has none: the fixed bake
  EXPECT_FALSE(fv::dson::parseSurface("Torso", layered({kAoaBrick, kAoaTorso})).bumpRange.has_value());
  EXPECT_FALSE(fv::dson::parseSurface("Face", channelsOf(kSkin)).bumpRange.has_value());
  EXPECT_EQ(fv::mat::bumpHeightInUv(0.43f, 1, 0), 0.f);
}

// Installed Iray presets for a 3Delight-era item: the one that keeps its look, or none.
namespace {

// The G8M Basic Wear Boxers' two surfaces as the item's own scene layer sets them: DAZ Default,
// the Black textures
const char* kBoxersBlackMaps = R"({
  "diffuse":{"channel":{"id":"diffuse","current_value":[1,1,1],"image_file":"/Runtime/Textures/ABen/Basic%20Wear/Black/Boxers_BaseColor.jpg"}},
  "normal":{"channel":{"id":"normal","current_value":1,"image_file":"/Runtime/Textures/ABen/Basic%20Wear/Black/Boxers_Normal.jpg"}}})";

fv::SceneGraph boxersScene()
{
  fv::SceneGraph scene;
  for(const char* name : {"Boxers", "BoxersWaistband"})
    scene.materials.push_back(fv::dson::buildMaterial(name, layered({kBoxers, kBoxersBlackMaps})));
  return scene;
}

// A material preset on the Boxers' surfaces: `channels` values each (the first an Iray one unless
// `threeDelight`), the colour's maps unless `folder` is empty, and a dozen maps removed
Json boxersPreset(const std::string& folder, int channels, bool threeDelight = false)
{
  Json animations = Json::array();
  auto add        = [&](const std::string& surface, const std::string& channel, const std::string& field, const Json& value) {
    animations.push_back({{"url", "name://@selection#materials/" + surface + ":?extra/studio_material_channels/channels/" + channel + "/" + field},
                                 {"keys", Json::array({Json::array({0, value})})}});
  };
  for(const char* surface : {"Boxers", "BoxersWaistband"})
  {
    for(int i = 0; i < channels; i++)
      add(surface, i == 0 ? (threeDelight ? "glossiness" : "Glossy%20Roughness") : "Channel" + std::to_string(i), "value", 0.5);
    if(!folder.empty())
    {
      add(surface, "diffuse", "image_file", "/Runtime/Textures/ABen/Basic%20Wear/" + folder + "/Boxers_BaseColor.jpg");
      add(surface, "Normal%20Map", "image_file", "/Runtime/Textures/ABen/Basic%20Wear/" + folder + "/Boxers_Normal.jpg");
    }
    for(int i = 0; i < 12; i++)
      add(surface, "Unused" + std::to_string(i), "image", nullptr);
  }
  return {{"asset_info", {{"type", "preset_material"}}}, {"scene", {{"animations", animations}}}};
}

}  // namespace

TEST(Materials, ChooseIrayPresetKeepsTheItemsLook)
{
  const fv::SceneGraph scene = boxersScene();
  ASSERT_EQ(scene.materials[0].surface.shader, fv::mat::Surface::Shader::eDazDefault);
  const Json black     = boxersPreset("Black", 60);
  const Json grey      = boxersPreset("Grey", 60);
  const Json shaderOnly = boxersPreset("", 90);  // Iray Optimized: values only, the item's maps stay
  const Json option    = boxersPreset("", 10);    // a "Low Glossiness" option on the item
  const Json delight   = boxersPreset("Black", 60, true);
  using fv::dson::chooseIrayPreset;

  // the colour the item came in, over a values-only conversion and the other colours
  EXPECT_EQ(chooseIrayPreset(scene, {&grey, &shaderOnly, &black}), 2);
  // a values-only conversion when no colour matches
  EXPECT_EQ(chooseIrayPreset(scene, {&grey, &shaderOnly}), 1);
  // another colour, an option or a 3Delight preset: none
  EXPECT_EQ(chooseIrayPreset(scene, {&grey}), -1);
  EXPECT_EQ(chooseIrayPreset(scene, {&option}), -1);
  EXPECT_EQ(chooseIrayPreset(scene, {&delight}), -1);
  EXPECT_EQ(chooseIrayPreset(scene, {nullptr}), -1);

  // an item already in Iray takes none
  fv::SceneGraph iray;
  iray.materials.push_back(fv::dson::buildMaterial("Boxers", channelsOf(kSkin)));
  EXPECT_EQ(chooseIrayPreset(iray, {&black, &shaderOnly}), -1);

  // a preset for half the item's surfaces still covers enough; the maps of the other half stay
  fv::SceneGraph three = boxersScene();
  three.materials.push_back(fv::dson::buildMaterial("Band2", layered({kBoxers, kBoxersBlackMaps})));
  three.materials.push_back(fv::dson::buildMaterial("Band3", layered({kBoxers, kBoxersBlackMaps})));
  EXPECT_EQ(chooseIrayPreset(three, {&black}), 0);
  three.materials.push_back(fv::dson::buildMaterial("Band4", layered({kBoxers, kBoxersBlackMaps})));
  EXPECT_EQ(chooseIrayPreset(three, {&black}), -1);
}

// A whole preset of another shader starts the surface from that shader, as DAZ Studio replaces the
// material; a partial one is laid over what is there.
namespace {

// PBRSkin.dsf's library entry, a few of its channels at the file's defaults: the brick graph and the
// labels, Horizontal Tiles2 labelled "Horizontal Tiles" among them
const char* kPbrSkinDefinition = R"({"id":"PBRSkin",
  "diffuse":{"channel":{"id":"diffuse","name":"Diffuse Color","label":"Base Color","type":"float_color","value":[0.7529414,0.7529414,0.7529414]}},
  "extra":[{"type":"studio/material/daz_brick","brick_settings":{}},{"type":"studio_material_channels","channels":[
  {"channel":{"id":"Metallic Weight","label":"Metallicity","type":"float","value":0}},
  {"channel":{"id":"Translucency Enable","type":"bool","value":false}},
  {"channel":{"id":"Translucency Weight","type":"float","value":0.8}},
  {"channel":{"id":"Dual Lobe Specular Enable","type":"bool","value":false}},
  {"channel":{"id":"Dual Lobe Specular Weight","type":"float","value":1}},
  {"channel":{"id":"Dual Lobe Specular Reflectivity","type":"float","value":0.5}},
  {"channel":{"id":"Dual Lobe Specular Roughness Mult","type":"float","value":1}},
  {"channel":{"id":"Specular Lobe 1 Roughness","type":"float","value":0.6}},
  {"channel":{"id":"Specular Lobe 2 Roughness Mult","type":"float","value":0.4}},
  {"channel":{"id":"Dual Lobe Specular Ratio","type":"float","value":0.15}},
  {"channel":{"id":"Normal Map","type":"float","value":1}},
  {"channel":{"id":"Bump Enable","type":"bool","value":false}},
  {"channel":{"id":"Makeup Enable","type":"bool","value":false}},
  {"channel":{"id":"Detail Enable","type":"bool","value":false}},
  {"channel":{"id":"Top Coat Weight","type":"float","value":0.5}},
  {"channel":{"id":"Horizontal Tiles2","label":"Horizontal Tiles","type":"float","value":1}},
  {"channel":{"id":"Vertical Tiles2","label":"Vertical Tiles","type":"float","value":1}}]}]})";
const char* kPbrSkinUrl = "/data/Daz%203D/Built-in%20Content/Daz%20Iray%20PBRSkin/PBRSkin.dsf#PBRSkin";

fv::dson::MaterialDefinitions pbrSkinDefinitions()
{
  return {{std::string(kPbrSkinFile) + "#PBRSkin", collect({}, kPbrSkinDefinition, kPbrSkinFile)}};
}

// Genesis 9's head in Iray Uber: its own glossy layer, a dual lobe, a normal map, tiled twice
const char* kUberHead = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,0.8,0.7],"image_file":"/Runtime/Textures/face_d.jpg"}},
  {"channel":{"id":"Translucency Weight","type":"float","current_value":0.6}},
  {"channel":{"id":"Normal Map","type":"image","current_value":1.0,"image_file":"/Runtime/Textures/face_n.jpg"}},
  {"channel":{"id":"Glossy Layered Weight","type":"float","current_value":1}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.4}},
  {"channel":{"id":"Dual Lobe Specular Weight","type":"float","current_value":0.5}},
  {"channel":{"id":"Specular Lobe 1 Roughness","type":"float","current_value":0.2}},
  {"channel":{"id":"Specular Lobe 2 Roughness","type":"float","current_value":0.5}},
  {"channel":{"id":"Dual Lobe Specular Ratio","type":"float","current_value":0.5}},
  {"channel":{"id":"Horizontal Tiles","type":"float","current_value":2}},
  {"channel":{"id":"Vertical Tiles","type":"float","current_value":2}}
]}]})";

// Genesis 9 Eyes.duf's Eye Left before its add-on MAT: Iray Uber, with the refraction, anisotropy
// and reflectance tint PBRSkin has no channel for
const char* kUberAddOnEye = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/eyes_d.jpg"}},
  {"channel":{"id":"Translucency Weight","type":"float","current_value":0.3}},
  {"channel":{"id":"Glossy Layered Weight","type":"float","current_value":0.0008046585}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0}},
  {"channel":{"id":"Refraction Weight","type":"float","current_value":0}},
  {"channel":{"id":"Refraction Index","type":"float","current_value":1.5}},
  {"channel":{"id":"Glossy Anisotropy","type":"float","current_value":0}},
  {"channel":{"id":"SSS Reflectance Tint","type":"color","current_value":[0.9,0.7,0.6]}}
]}]})";

using Values = std::vector<std::pair<std::string, Json>>;

// A whole preset of one surface: its scene material (the shader's `url` and `type`) and a value for
// each channel, an image too where `images` gives one, at `node`.
Json wholePreset(const std::string& node, const std::string& surface, const std::string& url, const std::string& type, const Values& values,
                 const std::map<std::string, std::string>& images = {})
{
  Json       animations = Json::array();
  const auto add        = [&](const std::string& channel, const std::string& field, const Json& value) {
    animations.push_back({{"url", node + "#materials/" + surface + ":?extra/studio_material_channels/channels/" + channel + "/" + field},
                                 {"keys", Json::array({Json::array({0, value})})}});
  };
  for(const auto& [channel, value] : values)
    add(channel, "value", value);
  for(const auto& [channel, image] : images)
    add(channel, "image_file", image);
  const Json entry = {{"url", url}, {"groups", Json::array({surface})}, {"extra", Json::array({Json{{"type", type}}})}};
  return Json{{"asset_info", Json{{"type", "preset_material"}}}, {"scene", Json{{"materials", Json::array({entry})}, {"animations", animations}}}};
}

// 42 PBRSkin channels at chosen values (not all PBRSkin.dsf's defaults: the dual lobe on at roughness
// 0.5, no top coat) and its base map; Specular Lobe 2 Roughness Mult is left to the definition
Json pbrSkinPreset(const std::string& surface, const std::string& node = "name://@selection")
{
  const Values values = {
      {"diffuse", Json::array({1, 1, 1})},
      {"Normal Map", 1},
      {"Dual Lobe Specular Enable", true},
      {"Dual Lobe Specular Weight", 1},
      {"Dual Lobe Specular Reflectivity", 0.5},
      {"Dual Lobe Specular Roughness Mult", 1},
      {"Specular Lobe 1 Roughness", 0.5},
      {"Dual Lobe Specular Ratio", 0.15},
      {"Top Coat Weight", 0},
      {"Makeup Enable", false},
      {"Detail Enable", false},
      {"Translucency Enable", true},
      {"Translucency Weight", 0.5},
      {"Metallic Weight", 0},
      {"Smooth On", true},
      {"Smooth Angle", 89.9},
      {"Line Preview Color", Json::array({0, 0, 0})},
      {"Line Start Width", 0.1},
      {"Line End Width", 0.1},
      {"Line UV Width", 0},
      {"Render Priority", 3},
      {"Propagate Priority", false},
      {"Metallicity Enable", false},
      {"Diffuse Enable", true},
      {"Diffuse Roughness", 0},
      {"Invert Transmission Normal", true},
      {"Bump Enable", true},
      {"Makeup Weight", 1},
      {"Makeup Roughness Mult", 1},
      {"Makeup Metallicity Enable", false},
      {"Specular Occlusion Enable", false},
      {"Ambient Occlusion Weight", 1},
      {"Specular Occlusion Grazing", 0.1},
      {"Top Coat Enable", false},
      {"Top Coat Reflectivity", 0.5},
      {"Top Coat Roughness", 0.05},
      {"Metallic Flakes Enable", false},
      {"Displacement Strength", 1},
      {"Minimum Displacement", -0.1},
      {"Maximum Displacement", 0.1},
      {"Transmission Enable", false},
      {"Sub Surface Enable", false},
  };
  return wholePreset(node, surface, kPbrSkinUrl, "studio/material/daz_brick", values, {{"diffuse", "/pb_d.jpg"}});
}

// 45 Iray Uber channels, values only, as Iray Optimized Genesis 2 Female MAT and the Basic Wear Iray
// presets set them; `more` replaces or adds
Values uberValues(const Values& more = {})
{
  Values values = {
      {"diffuse", Json::array({1, 1, 1})},
      {"Glossy Roughness", 0.4},
      {"Translucency Weight", 0.5},
      {"Metallic Weight", 0},
      {"Base Mixing", 0},
      {"Diffuse Overlay Weight", 0},
      {"Diffuse Roughness", 0},
      {"Dual Lobe Specular Weight", 0},
      {"Dual Lobe Specular Reflectivity", 0.5},
      {"Specular Lobe 1 Roughness", 0},
      {"Specular Lobe 2 Roughness", 0},
      {"Dual Lobe Specular Ratio", 1},
      {"Glossy Layered Weight", 0.325},
      {"Glossy Color", Json::array({1, 1, 1})},
      {"Glossy Color Effect", 0},
      {"Glossy Reflectivity", 0.5},
      {"Glossy Anisotropy", 0},
      {"Glossy Anisotropy Rotations", 0},
      {"Backscattering Weight", 0},
      {"Translucency Color", Json::array({1, 1, 1})},
      {"Base Color Effect", 0},
      {"SSS Reflectance Tint", Json::array({1, 1, 1})},
      {"Transmitted Measurement Distance", 0},
      {"Transmitted Color", Json::array({1, 1, 1})},
      {"Scattering Measurement Distance", 0},
      {"SSS Amount", 0},
      {"SSS Direction", 0},
      {"Refraction Index", 1.5},
      {"Refraction Weight", 0},
      {"Abbe", 0},
      {"Normal Map", 1},
      {"Top Coat Weight", 0},
      {"Top Coat Roughness", 0},
      {"Thin Walled", false},
      {"Emission Color", Json::array({0, 0, 0})},
      {"Emission Temperature", 6500},
      {"Luminance", 1500},
      {"Luminance Units", 3},
      {"Cutout Opacity", 1},
      {"Displacement Strength", 0},
      {"Horizontal Tiles", 1},
      {"Horizontal Offset", 0},
      {"Vertical Tiles", 1},
      {"Vertical Offset", 0},
      {"UV Set", 0},
  };
  for(const auto& [channel, value] : more)
  {
    const auto it = std::find_if(values.begin(), values.end(), [&](const auto& v) { return v.first == channel; });
    if(it != values.end())
      it->second = value;
    else
      values.push_back({channel, value});
  }
  return values;
}

const fv::dson::Channel* channelOf(const fv::Material& m, const char* name)
{
  return m.channels->get(fv::dson::norm(name));
}

}  // namespace

TEST(Materials, AWholePresetOfAnotherShaderStartsFromThatShader)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Head", channelsOf(kUberHead)));
  ASSERT_EQ(scene.materials[0].surface.shader, fv::mat::Surface::Shader::eIray);
  ASSERT_EQ(scene.materials[0].surface.glossy.weight, 1.f);
  ASSERT_EQ(fv::dson::applyMaterialPreset(pbrSkinPreset("Head"), scene, nullptr, pbrSkinDefinitions()).changed, std::set<int>({0}));
  const fv::Material&     m = scene.materials[0];
  const fv::mat::Surface& s = m.surface;
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::ePbrSkin);
  // Uber's glossy layer and tiling went with Uber: PBRSkin has neither
  EXPECT_EQ(channelOf(m, "Glossy Layered Weight"), nullptr);
  EXPECT_EQ(channelOf(m, "Glossy Roughness"), nullptr);
  ASSERT_NE(channelOf(m, "Horizontal Tiles"), nullptr);
  EXPECT_EQ(channelOf(m, "Horizontal Tiles")->id, "horizontaltiles2");  // PBRSkin's, by its label
  EXPECT_EQ(s.glossy.weight, 0.f);
  EXPECT_FLOAT_EQ(s.glossy.roughness.value, 0.5f);
  EXPECT_EQ(s.tiles, (std::array<float, 2>{1, 1}));
  EXPECT_EQ(s.base.map, "/pb_d.jpg");
  // the preset sets Normal Map's value alone: the map carries over, a channel of the same id
  EXPECT_EQ(s.normalMap, "/Runtime/Textures/face_n.jpg");
  ASSERT_TRUE(s.dual.has_value());
  EXPECT_FLOAT_EQ(s.dual->weight, 1.f);
  EXPECT_FLOAT_EQ(s.dual->lobe1.value, 0.5f);
  EXPECT_FLOAT_EQ(s.dual->lobe2.value, 0.2f);  // the definition's Specular Lobe 2 Roughness Mult, 0.4
  EXPECT_FLOAT_EQ(s.dual->ratio, 0.85f);
  const auto lobes = fv::mat::specularLobes(s);
  EXPECT_FLOAT_EQ(lobes.intensity, 0.85f);
  EXPECT_FLOAT_EQ(lobes.roughness.value, 0.5f);
  ASSERT_TRUE(lobes.clearcoat.has_value());
  EXPECT_FLOAT_EQ(lobes.clearcoat->weight, 0.15f);
  EXPECT_FLOAT_EQ(lobes.clearcoat->roughness.value, 0.2f);
}

TEST(Materials, AShaderSwitchRecordsTheShaderAndWorksWithoutItsDefinition)
{
  for(const bool withDefinition : {true, false})
  {
    fv::SceneGraph scene;
    scene.materials.push_back(fv::dson::buildMaterial("Head", channelsOf(kUberHead)));
    fv::dson::applyMaterialPreset(pbrSkinPreset("Head"), scene, nullptr, withDefinition ? pbrSkinDefinitions() : fv::dson::MaterialDefinitions{});
    const fv::mat::Surface& s = scene.materials[0].surface;
    EXPECT_EQ(s.shader, fv::mat::Surface::Shader::ePbrSkin) << withDefinition;
    EXPECT_EQ(s.dazShader.type, "studio/material/daz_brick");
    EXPECT_EQ(s.dazShader.url, std::string(kPbrSkinFile) + "#PBRSkin");
    ASSERT_TRUE(s.dual.has_value());
    // without the definition its lobe 2 multiplier is missing: the lobes are the same
    EXPECT_FLOAT_EQ(s.dual->lobe2.value, withDefinition ? 0.2f : 0.5f);
    EXPECT_EQ(s.tiles, (std::array<float, 2>{1, 1}));
  }
}

TEST(Materials, AnAddOnEyeSwitchedToPbrSkinLosesUbersRefractionAndTint)
{
  // Genesis 9 Eyes MAT on Genesis 9 Eyes.duf's Uber eye
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Eye Left", channelsOf(kUberAddOnEye)));
  ASSERT_TRUE(scene.materials[0].surface.refraction.has_value());
  ASSERT_TRUE(scene.materials[0].surface.glossyAnisotropy.has_value());
  fv::dson::applyMaterialPreset(pbrSkinPreset("Eye Left"), scene, nullptr, pbrSkinDefinitions());
  const fv::mat::Surface& s = scene.materials[0].surface;
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::ePbrSkin);
  EXPECT_FALSE(s.refraction.has_value());
  EXPECT_FALSE(s.glossyAnisotropy.has_value());
  EXPECT_EQ(s.glossy.weight, 0.f);
  ASSERT_TRUE(s.translucency.has_value());
  EXPECT_EQ(s.translucency->reflectanceTint, (fv::mat::Rgb{1, 1, 1}));
}

TEST(Materials, AWholeUberPresetOverPbrSkinTakesItsMakeupAway)
{
  // a PBRSkin surface with makeup, and a whole Uber preset of 45 values with another base map
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Eye Left", channelsOf(kMadeUp)));
  ASSERT_TRUE(scene.materials[0].surface.makeup.has_value());
  const Json uber = wholePreset("name://@selection", "Eye Left", "#Eye%20Left", "studio/material/uber_iray",
                                uberValues({{"Glossy Layered Weight", 0.5}}), {{"diffuse", "/iris_d.jpg"}});
  fv::dson::applyMaterialPreset(uber, scene);
  const fv::mat::Surface& s = scene.materials[0].surface;
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::eIray);
  EXPECT_FALSE(s.makeup.has_value());
  EXPECT_EQ(channelOf(scene.materials[0], "Makeup Enable"), nullptr);
  EXPECT_EQ(s.base.map, "/iris_d.jpg");
  EXPECT_FLOAT_EQ(s.glossy.weight, 0.5f);
}

TEST(Materials, AWholePresetMadeForAnItemTheSceneLacksIsLaidOver)
{
  // stray: its values land on another item's surfaces, which keep their shader's channels
  fv::SceneGraph scene = wornScene();
  fv::dson::applyMaterialPreset(pbrSkinPreset("Base", "name://@selection/SomeOtherHair:"), scene, nullptr, pbrSkinDefinitions());
  const fv::Material& m = scene.materials[4];
  EXPECT_NE(channelOf(m, "Glossy Roughness"), nullptr);
  EXPECT_EQ(m.surface.shader, fv::mat::Surface::Shader::ePbrSkin);  // read from the channels laid over, as before
}

TEST(Materials, AnIrayOptimizedPresetKeepsTheMapsOfTheShaderItReplaces)
{
  // Iray Optimized Genesis 2 Female MAT sets values only, no image, on "#Legs" and the rest: DAZ
  // keeps the 3Delight material's maps where Iray Uber has a channel of the same id
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Boxers", layered({kBoxers, kBoxersBlackMaps})));
  ASSERT_EQ(scene.materials[0].surface.shader, fv::mat::Surface::Shader::eDazDefault);
  const std::string map = scene.materials[0].surface.base.map;
  ASSERT_FALSE(map.empty());
  fv::dson::applyMaterialPreset(wholePreset("name://@selection", "Boxers", "#Boxers", "studio/material/uber_iray", uberValues()), scene);
  const fv::Material& m = scene.materials[0];
  EXPECT_EQ(m.surface.shader, fv::mat::Surface::Shader::eIray);
  EXPECT_EQ(m.surface.base.map, map);
  EXPECT_FLOAT_EQ(m.surface.glossy.roughness.value, 0.4f);
  EXPECT_EQ(channelOf(m, "diffuse_strength"), nullptr);
  EXPECT_EQ(channelOf(m, "Lighting Model"), nullptr);
}

TEST(Materials, AWholePresetOfTheSameLegacyFamilyOnlyOverrides)
{
  // UberSurface is told from DAZ Default by marker channels a preset of its own may leave out
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Torso", channelsOf(kG2fTorso)));
  ASSERT_EQ(scene.materials[0].surface.shader, fv::mat::Surface::Shader::eUberSurface);
  Values values;
  for(int i = 0; i < 45; i++)
    values.push_back({"Legacy Channel " + std::to_string(i), 0.5});
  fv::dson::applyMaterialPreset(wholePreset("name://@selection", "Torso", "#Torso", "studio/material/daz_shader", values), scene);
  EXPECT_NE(channelOf(scene.materials[0], "Map Tiling U"), nullptr);
  EXPECT_EQ(scene.materials[0].surface.shader, fv::mat::Surface::Shader::eUberSurface);
}

// FeSoul's hair colours are shader presets for the selection: "name://@selection#materials:?…"
// names no surface. They go on the surfaces with the shader they name (the OmniHair strands), not
// on the scalp cap, whose shader is Iray Uber.
TEST(Materials, APresetForTheSelectionGoesOnTheSurfacesWithItsShader)
{
  constexpr const char* kOmniHair = "data/Daz 3D/Built-in Content/OmniSurface/OmniHair Root-Tip Blend.dsf#OmniHair Root-Tip Blend";
  fv::SceneGraph        scene;
  for(const char* name : {"Hair_Front1", "Hair_Back1"})
  {
    fv::dson::Channels ch = channelsOf(omniHair().c_str());
    ch.setShader({"studio/material/daz_brick", kOmniHair});
    scene.materials.push_back(fv::dson::buildMaterial(name, ch));
  }
  scene.materials.push_back(fv::dson::buildMaterial("Cap", channelsOf(kSkin)));
  const Json preset = Json::parse(R"({"scene":{
    "materials":[{"url":"/data/Daz%203D/Built-in%20Content/OmniSurface/OmniHair%20Root-Tip%20Blend.dsf#OmniHair%20Root-Tip%20Blend",
      "extra":[{"type":"studio/material/daz_brick"}]}],
    "animations":[{"url":"name://@selection#materials:?extra/studio_material_channels/channels/Melanin/value","keys":[[0,0.9]]}]}})");
  // an item's colour checks the surfaces first: the strands are the hair's
  EXPECT_EQ(fv::dson::materialPresetTargets(preset, scene), std::set<int>({0, 1}));
  const auto result = fv::dson::applyMaterialPreset(preset, scene);
  EXPECT_EQ(result.changed, std::set<int>({0, 1}));
  EXPECT_EQ(result.matched, std::set<std::string>({fv::dson::kSelectionSurfaces}));
  EXPECT_TRUE(result.missing.empty());
  EXPECT_EQ(channelOf(scene.materials[1], "Melanin")->value, 0.9);
  EXPECT_EQ(channelOf(scene.materials[2], "Melanin"), nullptr);
  // without a shader to go by, it goes nowhere
  const Json bare = Json::parse(R"({"scene":{"animations":[{"url":"name://@selection#materials:?extra/studio_material_channels/channels/Melanin/value","keys":[[0,0.1]]}]}})");
  const auto none = fv::dson::applyMaterialPreset(bare, scene);
  EXPECT_TRUE(none.changed.empty());
  EXPECT_EQ(none.missing, std::set<std::string>({fv::dson::kSelectionSurfaces}));
}

TEST(Materials, APartialPresetOfAnotherShaderOnlyOverrides)
{
  // a CJ skin shine: one PBRSkin channel and its map, on an Uber head
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Head", channelsOf(kUberHead)));
  const Json shine = wholePreset("name://@selection", "Head", kPbrSkinUrl, "studio/material/daz_brick", {{"Specular Lobe 1 Roughness", 0.3}},
                                 {{"Specular Lobe 1 Roughness", "/SLW.jpg"}});
  fv::dson::applyMaterialPreset(shine, scene, nullptr, pbrSkinDefinitions());
  const fv::mat::Surface& s = scene.materials[0].surface;
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::eIray);
  EXPECT_EQ(s.glossy.weight, 1.f);
  EXPECT_EQ(s.tiles, (std::array<float, 2>{2, 2}));
  ASSERT_TRUE(s.dual.has_value());
  EXPECT_FLOAT_EQ(s.dual->lobe1.value, 0.3f);
  EXPECT_EQ(s.dual->lobe1.map, "/SLW.jpg");
}

// A lip colour (CJ Evangeline's, on Addy Florence's Iray Uber head) is a partial PBRSkin preset, laid
// over: its makeup channels make the head read as PBRSkin, but it carries none of PBRSkin's layer
// switches, so none of them turns the head's own layers off. Its normal map, dual lobe,
// translucency, coat and metal stay, with the lipstick over them.
TEST(Materials, ALipColourOverAnUberHeadLeavesItsLayersOn)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Head", layered({kUberHead, R"({"extra":[{"channels":[
    {"channel":{"id":"Top Coat Weight","type":"float","current_value":0.3}},
    {"channel":{"id":"Metallic Weight","type":"float","current_value":1,"image_file":"/Runtime/Textures/face_m.jpg"}},
    {"channel":{"id":"Transmitted Measurement Distance","type":"float","current_value":0.1}},
    {"channel":{"id":"Scattering Measurement Distance","type":"float","current_value":0.01}}
  ]}]})"})));
  const Json lips = wholePreset("name://@selection", "Head", kPbrSkinUrl, "studio/material/daz_brick",
                                {{"Makeup Enable", true}, {"Makeup Weight", 0.7}, {"Makeup Base Color", Json::array({1, 0, 0})}},
                                {{"Makeup Weight", "/lipmask.png"}});
  ASSERT_EQ(fv::dson::applyMaterialPreset(lips, scene, nullptr, pbrSkinDefinitions()).changed, std::set<int>({0}));
  const fv::mat::Surface& s = scene.materials[0].surface;
  ASSERT_TRUE(s.makeup.has_value());
  EXPECT_FLOAT_EQ(s.makeup->weight, 0.7f);
  EXPECT_EQ(s.makeup->weightMap, "/lipmask.png");
  EXPECT_EQ(s.normalMap, "/Runtime/Textures/face_n.jpg");
  ASSERT_TRUE(s.dual.has_value());
  EXPECT_FLOAT_EQ(s.dual->weight, 0.5f);
  ASSERT_TRUE(s.translucency.has_value());
  EXPECT_FLOAT_EQ(s.translucency->weight, 0.6f);
  EXPECT_FALSE(s.translucency->thinWalled);
  ASSERT_TRUE(s.translucency->subsurface.has_value());
  EXPECT_FLOAT_EQ(s.translucency->subsurface->transmittedDistance, 0.1f);
  EXPECT_FLOAT_EQ(s.translucency->subsurface->scatterDistance, 0.01f);
  ASSERT_TRUE(s.topCoat.has_value());
  EXPECT_FLOAT_EQ(s.topCoat->weight, 0.3f);
  EXPECT_FLOAT_EQ(s.metallic.value, 1.f);
  EXPECT_EQ(s.metallic.map, "/Runtime/Textures/face_m.jpg");
  const auto l = fv::mat::lower(s, {}, {.skin = fv::mat::SkinMode::eRaster});
  EXPECT_TRUE(l.material.contains("normalTexture"));
  EXPECT_FLOAT_EQ(l.material["extensions"]["KHR_materials_clearcoat"]["clearcoatFactor"].get<float>(), 0.3f);

  // a PBRSkin surface's own switches, off, still turn its layers off
  const auto off = surfaceOf("Head", kPbrSkinType,
                             {{"Makeup Enable", false}, {"Bump Enable", false}, {"Dual Lobe Specular Enable", false},
                              {"Translucency Enable", false}, {"Translucency Weight", 0.6}, {"Top Coat Enable", false},
                              {"Top Coat Weight", 0.3}, {"Metallicity Enable", false}, {"Metallic Weight", 1},
                              {"Normal Map", 1}, {"Dual Lobe Specular Weight", 0.5}},
                             {{"Normal Map", "/n.jpg"}});
  EXPECT_TRUE(off.normalMap.empty());
  EXPECT_FLOAT_EQ(off.dual->weight, 0.f);
  EXPECT_FLOAT_EQ(off.translucency->weight, 0.f);
  EXPECT_TRUE(off.translucency->thinWalled);
  EXPECT_FALSE(off.topCoat.has_value());
  EXPECT_FLOAT_EQ(off.metallic.value, 0.f);
}

TEST(Materials, MakeupThatSwappedTheShaderComesOffWithTheShader)
{
  // Polina 9 Face Makeup 01 is a whole PBRSkin preset on the head
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Head", channelsOf(kUberHead)));
  const Json makeup = pbrSkinPreset("Head");
  const auto r      = fv::dson::chooseLayer(scene, {0}, &makeup, {}, pbrSkinDefinitions());
  ASSERT_EQ(scene.materials[0].surface.shader, fv::mat::Surface::Shader::ePbrSkin);
  fv::dson::chooseLayer(scene, {0}, nullptr, r.layer);
  const fv::Material& m = scene.materials[0];
  EXPECT_EQ(m.surface.shader, fv::mat::Surface::Shader::eIray);
  ASSERT_NE(channelOf(m, "Glossy Layered Weight"), nullptr);
  EXPECT_EQ(channelOf(m, "Glossy Layered Weight")->value, 1);
  EXPECT_EQ(m.surface.glossy.weight, 1.f);
  EXPECT_EQ(m.surface.tiles, (std::array<float, 2>{2, 2}));
  EXPECT_EQ(m.surface.normalMap, "/Runtime/Textures/face_n.jpg");
  EXPECT_EQ(m.surface.dazShader, m.baseline->shader());
}

TEST(Materials, ASwappedShaderKeepsTheEditsAndTheBaselineGoesBack)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Head", channelsOf(kUberHead)));
  scene.materials[0].edits.roughness = 0.9f;
  const auto roughness = [&] {
    const fv::Material& m = scene.materials[0];
    return fv::mat::lower(m.surface, m.edits, {}).material["pbrMetallicRoughness"]["roughnessFactor"].get<float>();
  };
  fv::dson::applyMaterialPreset(pbrSkinPreset("Head"), scene, nullptr, pbrSkinDefinitions());
  EXPECT_EQ(scene.materials[0].surface.shader, fv::mat::Surface::Shader::ePbrSkin);
  EXPECT_EQ(scene.materials[0].edits.roughness, 0.9f);
  EXPECT_NE(scene.materials[0].baseline, scene.materials[0].channels);
  EXPECT_FLOAT_EQ(roughness(), 0.9f);
  // the character's own: the shader its channels are
  EXPECT_EQ(fv::dson::restoreBaseline(scene, {0}), std::set<int>({0}));
  const fv::mat::Surface& s = scene.materials[0].surface;
  EXPECT_EQ(s.shader, fv::mat::Surface::Shader::eIray);
  EXPECT_EQ(s.glossy.weight, 1.f);
  EXPECT_EQ(s.tiles, (std::array<float, 2>{2, 2}));
  EXPECT_EQ(s.normalMap, "/Runtime/Textures/face_n.jpg");
  EXPECT_EQ(scene.materials[0].edits.roughness, 0.9f);
  EXPECT_FLOAT_EQ(roughness(), 0.9f);
}

// ---- a map's texture instance settings (image_modification)

TEST(Materials, APresetKeepsAMapsTextureInstance)
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Face", channelsOf(kSkin)));
  const Json preset = Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Face:?extra/studio_material_channels/channels/diffuse/value","keys":[[0,[1,1,1]]]},
    {"url":"name://@selection#materials/Face:?extra/studio_material_channels/channels/diffuse/image_modification/horizontal_tiles","keys":[[0,2.5]]},
    {"url":"name://@selection#materials/Face:?extra/studio_material_channels/channels/diffuse/image_modification/SDcale","keys":[[0,4]]},
    {"url":"name://@selection#materials/Face:?extra/studio_material_channels/channels/Top%20Coat%20Weight/image_modification/scale","keys":[[0,0.5]]},
    {"url":"name://@selection#materials/Face:?Glossy%20Color/image_modification/scale","keys":[[0,3]]}
  ]}})");
  const auto result = fv::dson::applyMaterialPreset(preset, scene);
  // the settings are how a map is read, not values of their own
  EXPECT_EQ(result.edits, 1);
  const fv::Material& m = scene.materials[0];
  const auto*         d = channelOf(m, "diffuse");
  ASSERT_TRUE(d->imageModification.is_object());
  EXPECT_EQ(d->imageModification["horizontal_tiles"], 2.5);
  EXPECT_FALSE(d->imageModification.contains("SDcale"));
  // settings alone make no channel the material lacks
  EXPECT_EQ(channelOf(m, "Top Coat Weight"), nullptr);
  EXPECT_EQ(channelOf(m, "Glossy Color"), nullptr);
}

TEST(Materials, TheLoaderKeepsAnInlineTextureInstance)
{
  const auto ch = channelsOf(R"({"diffuse":{"channel":{"id":"diffuse","type":"color","current_value":[1,1,1],"image_file":"/cloth_d.jpg",
    "image_modification":{"scale":1,"offset":0,"horizontal_tiles":2.5,"vertical_tiles":2.5}}}})");
  const auto* d = ch.get("diffuse");
  ASSERT_NE(d, nullptr);
  EXPECT_EQ(d->imageModification["vertical_tiles"], 2.5);
  // a later layer replaces the fields it gives and keeps the rest
  const auto over = layered({R"({"diffuse":{"channel":{"id":"diffuse","current_value":[1,1,1],"image_modification":{"horizontal_tiles":2.5}}}})",
                             R"({"extra":[{"channels":[{"channel":{"id":"diffuse","image_modification":{"scale":0.5}}}]}]})"});
  EXPECT_EQ(over.get("diffuse")->imageModification["horizontal_tiles"], 2.5);
  EXPECT_EQ(over.get("diffuse")->imageModification["scale"], 0.5);
}

TEST(Materials, ATextureInstanceSettingTellsMaterialsApart)
{
  const auto with = [](const char* mod) {
    const std::string text = std::string(R"({"diffuse":{"channel":{"id":"diffuse","current_value":[1,1,1],"image_file":"/d.jpg")")
                             + (mod ? std::string(R"(,"image_modification":)") + mod : std::string()) + "}}}";
    return fv::dson::materialKey("Shirt", channelsOf(text.c_str()));
  };
  EXPECT_NE(with(R"({"horizontal_tiles":2.5})"), with(nullptr));
  // DAZ's defaults read as no setting
  EXPECT_EQ(with(R"({"scale":1,"horizontal_tiles":1,"vertical_tiles":1})"), with(nullptr));
  EXPECT_NE(with(R"({"scale":0.5})"), with(nullptr));
}

TEST(Materials, ATextureInstanceSettingAloneIsNoIrayPreset)
{
  // an Iray id whose map only has its scale set does not make a 3Delight preset Iray's
  const fv::SceneGraph scene  = boxersScene();
  Json                 preset = boxersPreset("", 60, true);
  for(const char* surface : {"Boxers", "BoxersWaistband"})
    preset["scene"]["animations"].push_back(
        {{"url", std::string("name://@selection#materials/") + surface + ":?extra/studio_material_channels/channels/Glossy%20Roughness/image_modification/scale"},
         {"keys", Json::array({Json::array({0, 0.5})})}});
  EXPECT_EQ(fv::dson::chooseIrayPreset(scene, {&preset}), -1);
}

// ---- a map's own tiling over the surface's
namespace {

// OOT Buttoned Up Outfit 02's shorts: the print and the bump tiled 2.5 times by their own texture
// instances, the normal map once
const char* kTiledCloth = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","type":"color","current_value":[1,1,1],"image_file":"/cloth_d.jpg",
    "image_modification":{"scale":1,"offset":0,"horizontal_tiles":2.5,"vertical_tiles":2.5}}},
  {"channel":{"id":"Bump Strength","type":"float","current_value":0.5,"image_file":"/cloth_b.jpg",
    "image_modification":{"horizontal_tiles":2.5,"vertical_tiles":2.5}}},
  {"channel":{"id":"Normal Map","type":"float","current_value":1,"image_file":"/cloth_n.jpg",
    "image_modification":{"horizontal_tiles":1,"vertical_tiles":1}}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.6}}
]}]})";

// A texture info's KHR_texture_transform, none when it has none
std::optional<fv::mat::UvTransform> tilingOf(const nlohmann::json& info)
{
  if(!info.contains("extensions") || !info["extensions"].contains("KHR_texture_transform"))
    return std::nullopt;
  const auto&          t = info["extensions"]["KHR_texture_transform"];
  fv::mat::UvTransform out;
  out.scale    = t.value("scale", std::array<float, 2>{1, 1});
  out.offset   = t.value("offset", std::array<float, 2>{0, 0});
  out.rotation = t.value("rotation", 0.f);
  return out;
}

}  // namespace

TEST(Materials, AMapsOwnTilingGoesOverTheSurfaces)
{
  const auto s = fv::dson::parseSurface("Shorts", channelsOf(kTiledCloth));
  // the normal map's one tile is DAZ's default: no tiling of its own
  EXPECT_EQ(s.mapTiling.size(), 2u);
  const auto lowered = fv::mat::lower(s, {}, {});
  const auto base    = tilingOf(lowered.material["pbrMetallicRoughness"]["baseColorTexture"]);
  ASSERT_TRUE(base.has_value());
  EXPECT_EQ(base->scale, (std::array<float, 2>{2.5f, 2.5f}));
  EXPECT_EQ(base->offset, (std::array<float, 2>{0, 0}));
  // the normal map and the bump cook into one image: tiled differently, it keeps the surface's tiling
  EXPECT_FALSE(tilingOf(lowered.material["normalTexture"]).has_value());

  // the surface's tiles under it
  const auto twice = fv::dson::parseSurface(
      "Shorts", layered({kTiledCloth, R"({"extra":[{"channels":[{"channel":{"id":"Horizontal Tiles","current_value":2}},
                                       {"channel":{"id":"Vertical Tiles","current_value":2}}]}]})"}));
  EXPECT_EQ(tilingOf(fv::mat::lower(twice, {}, {}).material["pbrMetallicRoughness"]["baseColorTexture"])->scale, (std::array<float, 2>{5, 5}));

  // the bump alone: tiled, and its height per its own tile
  const auto bumped = fv::dson::parseSurface("Shorts", layered({kTiledCloth, R"({"extra":[{"channels":[{"channel":{"id":"Normal Map","image_file":""}}]}]})"}));
  fv::mat::LowerOptions o;
  o.metresPerUv    = 0.5f;
  const auto bump  = fv::mat::lower(bumped, {}, o);
  EXPECT_EQ(tilingOf(bump.material["normalTexture"])->scale, (std::array<float, 2>{2.5f, 2.5f}));
  const auto req = std::find_if(bump.requests.begin(), bump.requests.end(), [](const fv::mat::TextureRequest& r) { return r.channels == "bn"; });
  ASSERT_NE(req, bump.requests.end());
  EXPECT_FLOAT_EQ(req->bumpHeight, fv::mat::bumpHeightInUv(0.5f, 2.5f, fv::mat::kBumpHeight));
}

TEST(Materials, AMapsOwnTilingMovesWithThePanelsOffset)
{
  const auto     s = fv::dson::parseSurface("Shorts", channelsOf(kTiledCloth));
  fv::mat::Edits e;
  e.uvOffsetU = 0.1f;
  // the print shifts as far over the surface as every other map: 0.1 of the surface is 0.25 of it
  const auto base = tilingOf(fv::mat::lower(s, e, {}).material["pbrMetallicRoughness"]["baseColorTexture"]);
  ASSERT_TRUE(base.has_value());
  EXPECT_FLOAT_EQ(base->offset[0], 0.25f);
  EXPECT_FLOAT_EQ(base->offset[1], 0.f);

  // turned by the panel, a map tiled differently across and down would turn apart from the rest: it
  // keeps the surface's transform
  const auto stretched = fv::dson::parseSurface(
      "Shorts", layered({kTiledCloth, R"({"extra":[{"channels":[{"channel":{"id":"diffuse","image_modification":{"vertical_tiles":1}}}]}]})"}));
  ASSERT_EQ(stretched.mapTiling.at("/cloth_d.jpg").tiles, (std::array<float, 2>{2.5f, 1}));
  fv::mat::Edits turned;
  turned.uvRotation = 30;
  const auto t      = tilingOf(fv::mat::lower(stretched, turned, {}).material["pbrMetallicRoughness"]["baseColorTexture"]);
  ASSERT_TRUE(t.has_value());
  EXPECT_EQ(t->scale, (std::array<float, 2>{1, 1}));
  EXPECT_NE(t->rotation, 0.f);
}

// ---- a cutout read from its map's alpha (DAZ's grayscale_mode "alpha")

TEST(Materials, ACutoutReadFromItsAlphaAsksTheCookerForTheAlpha)
{
  // booperfooper's Gamer Girl Outfit: a logo, white where its alpha cuts the panel away
  const char* panel = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","type":"color","current_value":[1,1,1],"image_file":"/panel_d.jpg"}},
    {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.5}},
    {"channel":{"id":"Cutout Opacity","type":"float","current_value":1,"image_file":"/logo.png",
      "image_modification":{"grayscale_mode":"alpha","scale":1,"offset":0,"invert":false}}}
  ]}]})";
  const auto request = [](const fv::mat::Surface& s, const fv::mat::Edits& e = {}) {
    const auto lowered = fv::mat::lower(s, e, {});
    return lowered.requests[lowered.material["pbrMetallicRoughness"]["baseColorTexture"]["slot"].get<size_t>()];
  };
  const auto s = fv::dson::parseSurface("Bottom", channelsOf(panel));
  EXPECT_TRUE(s.opacityFromAlpha);
  const auto alpha = request(s);
  EXPECT_EQ(alpha.alphaRel, "/logo.png");
  EXPECT_TRUE(alpha.alphaFromAlpha);
  // DAZ's default, "average": the map read as grey, as before
  const auto grey = fv::dson::parseSurface(
      "Bottom", layered({panel, R"({"extra":[{"channels":[{"channel":{"id":"Cutout Opacity","image_modification":{"grayscale_mode":"average"}}}]}]})"}));
  EXPECT_FALSE(grey.opacityFromAlpha);
  EXPECT_FALSE(request(grey).alphaFromAlpha);
  EXPECT_NE(fv::mat::requestKey(alpha), fv::mat::requestKey(request(grey)));
  // a cutout map of the panel's own is read as grey
  fv::mat::Edits e;
  e.opacityMap = "/mine.png";
  EXPECT_FALSE(request(s, e).alphaFromAlpha);
}

// ---- DAZ's channels read as its MDL reads them: Glossy Color, Base Mixing, squared roughness,
// Bump Enable, Makeup Roughness Mult ----

namespace {

const nlohmann::json& specularOf(const fv::mat::Lowered& l)
{
  return l.material["extensions"]["KHR_materials_specular"];
}

void expectRgb(const nlohmann::json& c, float r, float g, float b, float tol)
{
  EXPECT_NEAR(c[0].get<float>(), r, tol);
  EXPECT_NEAR(c[1].get<float>(), g, tol);
  EXPECT_NEAR(c[2].get<float>(), b, tol);
}

const char* kGreyGlossyColor = R"({"extra":[{"channels":[{"channel":{"id":"Glossy Color","type":"color","current_value":[0.5,0.5,0.5]}}]}]})";

}  // namespace

// irayubermaterial.mdl tints the glossy layer by glossy_color, normal to grazing. glTF has one
// lobe: its glossy share takes the colour, the dual lobes' share stays white. sRGB (0.5, 0.25, 1) is
// (0.21404, 0.05088, 1) linear.
TEST(Materials, GlossyColorTintsTheGlossyShareOfTheSpecular)
{
  const ChannelValues shirt = {{"Glossy Layered Weight", 1}, {"Glossy Reflectivity", 0.5}, {"Glossy Roughness", 0.4},
                               {"Glossy Color", Json::array({0.5, 0.25, 1})}};
  const auto          s     = surfaceOf("Shirt", kUberType, shirt, {{"Glossy Color", "/shirt_gc.jpg"}});
  EXPECT_NEAR(s.glossy.color[0], 0.21404f, 1e-4f);
  EXPECT_NEAR(s.glossy.color[1], 0.05088f, 1e-4f);
  EXPECT_NEAR(s.glossy.color[2], 1.0f, 1e-4f);
  const auto l = fv::mat::lower(s, {}, {});
  EXPECT_FLOAT_EQ(specularOf(l)["specularFactor"].get<float>(), 1.0f);
  expectRgb(specularOf(l)["specularColorFactor"], 0.21404f, 0.05088f, 1.0f, 1e-4f);
  ASSERT_TRUE(specularOf(l).contains("specularColorTexture"));
  const auto& map = requestOf(l, specularOf(l)["specularColorTexture"]);
  EXPECT_EQ(map.rel, "/shirt_gc.jpg");
  EXPECT_TRUE(map.srgb);
  EXPECT_FALSE(l.material["extensions"].contains("KHR_materials_ior"));  // a plain dielectric's 4%

  // a dual lobe half as strong as the glossy: the glossy is two thirds of the lobe
  ChannelValues dual = shirt;
  dual.push_back({"Dual Lobe Specular Weight", 0.5});
  const auto mixed = fv::mat::lower(surfaceOf("Shirt", kUberType, dual), {}, {});
  expectRgb(specularOf(mixed)["specularColorFactor"], 0.47603f, 0.36725f, 1.0f, 1e-4f);

  // no Glossy Color: white, and no map
  const auto skin = fv::mat::lower(fv::dson::parseSurface("Face", channelsOf(kSkin)), {}, {});
  expectRgb(specularOf(skin)["specularColorFactor"], 1, 1, 1, 0);
  EXPECT_FALSE(specularOf(skin).contains("specularColorTexture"));

  // the tear film is clear water whatever its Glossy Color
  const auto moisture = fv::dson::parseSurface("EyeMoisture Left", layered({kG9Moisture, kGreyGlossyColor}));
  expectRgb(specularOf(fv::mat::lower(moisture, {}, {.underTearFilm = true}))["specularColorFactor"], 1, 1, 1, 0);

  // a 3Delight-era surface's channel of that name is not Iray's
  const auto legacy = surfaceOf("Shirt", "studio/material/daz_shader", {{"Glossy Color", Json::array({0.5, 0.25, 1})}});
  EXPECT_EQ(legacy.glossy.color, (fv::mat::Rgb{1, 1, 1}));
  expectRgb(specularOf(fv::mat::lower(legacy, {}, {}))["specularColorFactor"], 1, 1, 1, 0);
}

// Iray tints what a refracting surface lets through by Glossy Color while Share Glossy Inputs is on,
// else by Refraction Color. glTF tints transmission by the base colour, which goes to that tint.
TEST(Materials, RefractionLetsLightThroughInItsOwnColour)
{
  const auto base = [](ChannelValues extra) {
    ChannelValues v = {{"diffuse", Json::array({0.5, 0.5, 0.5})}, {"Glossy Layered Weight", 1}, {"Glossy Roughness", 0},
                       {"Refraction Weight", 1}, {"Refraction Index", 1.33}, {"Glossy Color", Json::array({0.5, 0.25, 1})}};
    v.insert(v.end(), extra.begin(), extra.end());
    return fv::mat::lower(surfaceOf("Wine", kUberType, v), {}, {}).material["pbrMetallicRoughness"]["baseColorFactor"];
  };
  expectRgb(base({}), 0.21404f, 0.05088f, 1.0f, 1e-4f);  // shared by default
  expectRgb(base({{"Share Glossy Inputs", true}}), 0.21404f, 0.05088f, 1.0f, 1e-4f);
  expectRgb(base({{"Share Glossy Inputs", false}}), 1, 1, 1, 1e-6f);  // no Refraction Color: white
  expectRgb(base({{"Share Glossy Inputs", false}, {"Refraction Color", Json::array({1, 0.5, 1})}}), 1, 0.21404f, 1, 1e-4f);

  // the eye's water stays clear
  const auto moisture = fv::dson::parseSurface("EyeMoisture Left", layered({kG9Moisture, kGreyGlossyColor}));
  expectRgb(fv::mat::lower(moisture, {}, {.underTearFilm = true}).material["pbrMetallicRoughness"]["baseColorFactor"], 1, 1, 1, 0);
}

// Iray Uber's Weighted mode (normalized_base) mixes the diffuse and the glossy by Diffuse Weight and
// Glossy Weight, normalised past 1, with no Fresnel and no metal; Glossy Layered Weight is not read.
// glTF gets an index whose Fresnel is nearly flat (F0 0.95) scaled down to the glossy's share, and the
// diffuse scaled up again for the light that lobe already takes from under it.
TEST(Materials, WeightedMixingIsDiffuseAndGlossyByWeight)
{
  using fv::mat::Surface;
  const auto weighted = [](double d, double g, double l, ChannelValues extra = {}, std::map<std::string, std::string> images = {},
                           const char* name = "Shirt") {
    ChannelValues v = {{"diffuse", Json::array({1, 1, 1})}, {"Base Mixing", 2}, {"Metallic Weight", 1}, {"Glossy Roughness", 0.2},
                       {"Diffuse Weight", d},           {"Glossy Weight", g}, {"Glossy Layered Weight", l}};
    v.insert(v.end(), extra.begin(), extra.end());
    return surfaceOf(name, kUberType, v, images);
  };
  const auto pbrOf = [](const fv::mat::Lowered& l) { return l.material["pbrMetallicRoughness"]; };
  const auto iorOf = [](const fv::mat::Lowered& l) { return l.material["extensions"]["KHR_materials_ior"]["ior"].get<float>(); };

  EXPECT_NEAR(fv::mat::iorForReflectance(0.04f), 1.5f, 1e-4f);
  EXPECT_NEAR(fv::mat::iorForReflectance(1.f), 77.987f, 1e-2f);  // capped at kFlatF0
  EXPECT_NEAR(fv::mat::iorForReflectance(0.3333333f), 3.73205f, 1e-3f);

  // (a) a glossy weight of 0.5 with a layered weight of 0: a third glossy, two thirds diffuse
  {
    const auto s = weighted(1, 0.5, 0);
    EXPECT_EQ(s.mixing, Surface::Mixing::eWeighted);
    EXPECT_NEAR(s.diffuseWeight, 0.66667f, 1e-4f);
    EXPECT_NEAR(s.glossy.weight, 0.33333f, 1e-4f);
    EXPECT_FLOAT_EQ(s.metallic.value, 0);
    const auto l = fv::mat::lower(s, {}, {});
    EXPECT_FLOAT_EQ(pbrOf(l)["metallicFactor"].get<float>(), 0);
    EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.350877f, 1e-4f);
    EXPECT_NEAR(iorOf(l), 77.987f, 1e-2f);
    EXPECT_NEAR(pbrOf(l)["baseColorFactor"][0].get<float>(), 1.0f, 1e-4f);
  }
  // (b) weights under 1 together: the rest is lost, and the diffuse is 0.2 of what lies under 0.4
  {
    const auto l = fv::mat::lower(weighted(0.2, 0.4, 1), {}, {});
    expectRgb(pbrOf(l)["baseColorFactor"], 0.33333f, 0.33333f, 0.33333f, 1e-4f);
    EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.421053f, 1e-4f);
    EXPECT_NEAR(iorOf(l), 77.987f, 1e-2f);
    EXPECT_EQ(l.effective.color, (fv::mat::Rgb{1, 1, 1}));  // the panel shows the file's colour
  }
  // (c) hair's faint sheen
  {
    const auto s = weighted(1, 0.02, 1);
    EXPECT_NEAR(s.glossy.weight, 0.019608f, 1e-5f);
    const auto l = fv::mat::lower(s, {}, {});
    EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.020640f, 1e-5f);
    EXPECT_NEAR(iorOf(l), 77.987f, 1e-2f);
    EXPECT_NEAR(pbrOf(l)["baseColorFactor"][0].get<float>(), 1.0f, 1e-4f);
  }
  // (d) no glossy weight: no reflection, whatever the layered weight says
  {
    const auto l = fv::mat::lower(weighted(1, 0, 0.325037), {}, {});
    EXPECT_FLOAT_EQ(specularOf(l)["specularFactor"].get<float>(), 0);
    EXPECT_FALSE(l.material["extensions"].contains("KHR_materials_ior"));
  }
  // (e) Glossy Weight's map says where the surface is glossy: its grey in specularTexture's alpha.
  // Glossy Reflectivity's has no part in this mode.
  {
    const auto s = weighted(1, 1, 1, {{"Glossy Reflectivity", 0.5}}, {{"Glossy Weight", "/hair_gw.jpg"}, {"Glossy Reflectivity", "/hair_gr.jpg"}});
    EXPECT_EQ(s.glossy.weightMap, "/hair_gw.jpg");
    const auto l = fv::mat::lower(s, {}, {});
    ASSERT_TRUE(specularOf(l).contains("specularTexture"));
    const auto& r = requestOf(l, specularOf(l)["specularTexture"]);
    EXPECT_EQ(r.rel, "/hair_gw.jpg");
    EXPECT_EQ(r.channels, "wa");
    EXPECT_TRUE(r.mulRels.empty());
    EXPECT_FALSE(r.srgb);
  }
  // (f) refraction lies inside the glossy and takes its share; the glass keeps its own index
  {
    const auto s = weighted(1, 1, 1, {{"Refraction Weight", 1}, {"Refraction Index", 1.5}}, {}, "Glass");
    ASSERT_TRUE(s.refraction);
    EXPECT_NEAR(s.refraction->weight, 0.5f, 1e-6f);
    const auto l = fv::mat::lower(s, {}, {});
    EXPECT_NEAR(l.material["extensions"]["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 0.5f, 1e-6f);
    EXPECT_NEAR(iorOf(l), 1.5f, 1e-6f);
    EXPECT_FLOAT_EQ(specularOf(l)["specularFactor"].get<float>(), 1);
  }
  // (g) the dual lobes lie on the diffuse and take its share
  {
    const auto s = weighted(1, 1, 1, {{"Dual Lobe Specular Weight", 0.8}});
    ASSERT_TRUE(s.dual);
    EXPECT_NEAR(s.dual->weight, 0.4f, 1e-6f);
  }
  // PBR Metallicity/Roughness is as it was: Glossy Layered Weight, a 4% lobe, the metal
  {
    const auto s = surfaceOf("Shirt", kUberType,
                             {{"Base Mixing", 0}, {"Glossy Layered Weight", 1}, {"Glossy Reflectivity", 1}, {"Metallic Weight", 1}, {"Glossy Weight", 0.3}});
    EXPECT_EQ(s.mixing, Surface::Mixing::eMetallicRoughness);
    const auto l = fv::mat::lower(s, {}, {});
    EXPECT_FALSE(l.material["extensions"].contains("KHR_materials_ior"));
    EXPECT_FLOAT_EQ(pbrOf(l)["metallicFactor"].get<float>(), 1);
    EXPECT_FLOAT_EQ(specularOf(l)["specularFactor"].get<float>(), 1);
  }
}

// An index of its own changes nothing where 4% is all a surface reflects: skin and the eye keep
// their lobes.
TEST(Materials, APlainDielectricKeepsItsLobes)
{
  const auto skin = fv::mat::specularLobes(fv::dson::parseSurface("Face", channelsOf(kSkin)));
  EXPECT_FLOAT_EQ(skin.intensity, 0.25f);
  EXPECT_FLOAT_EQ(skin.glossyShare, 0);
  EXPECT_FLOAT_EQ(skin.reflectance, 0.04f);
  ASSERT_TRUE(skin.clearcoat);
  EXPECT_FLOAT_EQ(skin.clearcoat->weight, 0.25f);
  EXPECT_FLOAT_EQ(skin.clearcoat->roughness.value, 0.2f);

  const auto eye = fv::mat::specularLobes(fv::dson::parseSurface("Eye Left", channelsOf(kG9Eye)));
  EXPECT_FLOAT_EQ(eye.intensity, 0.85f);
  EXPECT_FLOAT_EQ(eye.reflectance, 0.04f);
  ASSERT_TRUE(eye.clearcoat);
  EXPECT_FLOAT_EQ(eye.clearcoat->weight, 0.15f);
  EXPECT_FLOAT_EQ(eye.clearcoat->roughness.value, 0.24f);
}

// Iray Uber's PBR Specular/Glossiness mode: Glossy Specular is the colour at normal incidence, white
// at grazing, Glossy Layered Weight scales the whole curve, and neither Glossy Reflectivity nor the
// metal is read (DAZ hides both in this mode). The wiring is inferred: Uber's is compiled into DAZ
// Studio.
TEST(Materials, SpecGlossReadsGlossySpecularAsF0)
{
  using fv::mat::Surface;
  const auto specGloss = [](double w, Json f0, ChannelValues extra = {}, std::map<std::string, std::string> images = {}) {
    ChannelValues v = {{"diffuse", Json::array({0.5, 0.5, 0.5})},
                       {"Base Mixing", 1},
                       {"Glossy Roughness", 0},
                       {"Glossiness", 0.7},
                       {"Glossy Reflectivity", 0.5},
                       {"Glossy Layered Weight", w},
                       {"Glossy Specular", f0},
                       {"Metallic Weight", 1}};
    v.insert(v.end(), extra.begin(), extra.end());
    return surfaceOf("Shirt", kUberType, v, images);
  };
  const auto grey  = [](double v) { return Json::array({v, v, v}); };
  const auto iorOf = [](const fv::mat::Lowered& l) { return l.material["extensions"]["KHR_materials_ior"]["ior"].get<float>(); };

  // (a) DAZ's default, 4.37% linear: a plain dielectric, no metal
  const auto s = specGloss(1, grey(0.2313726));
  EXPECT_EQ(s.mixing, Surface::Mixing::eSpecGloss);
  EXPECT_NEAR(s.glossy.reflectivity, 0.54669f, 1e-4f);
  EXPECT_EQ(s.glossy.color, (fv::mat::Rgb{1, 1, 1}));
  EXPECT_FLOAT_EQ(s.metallic.value, 0);
  const auto a = fv::mat::lower(s, {}, {});
  EXPECT_FLOAT_EQ(a.material["pbrMetallicRoughness"]["metallicFactor"].get<float>(), 0);
  EXPECT_FLOAT_EQ(specularOf(a)["specularFactor"].get<float>(), 1);
  EXPECT_NEAR(iorOf(a), 1.52886f, 1e-4f);
  expectRgb(specularOf(a)["specularColorFactor"], 1, 1, 1, 0);
  EXPECT_NEAR(a.material["pbrMetallicRoughness"]["roughnessFactor"].get<float>(), 0.3f, 1e-6f);
  // (b) the weight scales the whole curve: the index stays the colour's
  const auto b = fv::mat::lower(specGloss(0.2, grey(0.2313726)), {}, {});
  EXPECT_NEAR(specularOf(b)["specularFactor"].get<float>(), 0.2f, 1e-5f);
  EXPECT_NEAR(iorOf(b), 1.52886f, 1e-4f);
  // (c) a bright F0
  const auto c = fv::mat::lower(specGloss(1, grey(0.8078431)), {}, {});
  EXPECT_NEAR(iorOf(c), 8.3294f, 1e-2f);
  EXPECT_FLOAT_EQ(specularOf(c)["specularFactor"].get<float>(), 1);
  // (c2) Noki Veil: white at a fifth of the weight is an even fifth, not a bright rim
  const auto c2 = fv::mat::lower(specGloss(0.2, grey(1)), {}, {});
  EXPECT_NEAR(specularOf(c2)["specularFactor"].get<float>(), 0.210526f, 1e-5f);
  EXPECT_NEAR(iorOf(c2), 77.987f, 1e-2f);
  // (d) a coloured F0: the brightest channel is the index, the hue the specular colour
  const auto d = fv::mat::lower(specGloss(1, Json::array({0.5, 0.25, 0.5})), {}, {});
  expectRgb(specularOf(d)["specularColorFactor"], 1, 0.23769f, 1, 1e-4f);
  EXPECT_NEAR(iorOf(d), 2.72194f, 1e-3f);
  // (e) Glossy Reflectivity is not read in this mode
  const auto e = fv::mat::lower(specGloss(1, grey(0.2313726), {{"Glossy Reflectivity", 1}}), {}, {});
  EXPECT_EQ(e.material, a.material);
  // (f) Glossy Specular's map is the colour's
  const auto f = fv::mat::lower(specGloss(1, grey(0.2313726), {}, {{"Glossy Specular", "/bls_spec.jpg"}}), {}, {});
  ASSERT_TRUE(specularOf(f).contains("specularColorTexture"));
  EXPECT_EQ(requestOf(f, specularOf(f)["specularColorTexture"]).rel, "/bls_spec.jpg");
  EXPECT_TRUE(requestOf(f, specularOf(f)["specularColorTexture"]).srgb);
  // (h) under 4%: a plain index, the lobe scaled down
  const auto h = fv::mat::lower(specGloss(1, grey(0.1254902)), {}, {});
  EXPECT_NEAR(specularOf(h)["specularFactor"].get<float>(), 0.36110f, 1e-4f);
  EXPECT_FALSE(h.material["extensions"].contains("KHR_materials_ior"));
}

// conditional_square_roughness: Iray Uber squares Glossy Roughness into the lobe's alpha unless
// Roughness Squared is off, when the value is alpha itself (AuraVix's NYC Tenement Staircase).
// PBRSkin has no such switch.
TEST(Materials, RoughnessSquaredOffIsAlphaItself)
{
  const auto rough = [](const char* shader, std::optional<bool> squared, const char* channel = "Glossy Roughness") {
    ChannelValues v = {{channel, 0.09}, {"Metallic Weight", 0}};
    if(std::string(shader) == kPbrSkinType)
      v.push_back({"Detail Enable", false});
    if(squared)
      v.push_back({"Roughness Squared", *squared});
    return surfaceOf("Shirt", shader, v).glossy.roughness.value;
  };
  EXPECT_NEAR(rough(kUberType, false), 0.3f, 1e-6f);
  EXPECT_NEAR(rough(kUberType, true), 0.09f, 1e-6f);
  EXPECT_NEAR(rough(kUberType, std::nullopt), 0.09f, 1e-6f);
  EXPECT_NEAR(rough(kPbrSkinType, false, "Specular Lobe 1 Roughness"), 0.09f, 1e-6f);  // a stale Uber channel
}

// PBRSkin's brick graph ANDs Bump Enable into its normal map and its bump map alike (false in
// PBRSkin.dsf). Iray Uber has no switch.
TEST(Materials, PbrSkinBumpEnableSwitchesTheNormalAndBumpMaps)
{
  const auto skin = [](std::optional<bool> enable) {
    ChannelValues v = {{"Detail Enable", false}, {"Normal Map", 1}, {"Bump Strength", 0.5}};
    if(enable)
      v.push_back({"Bump Enable", *enable});
    return surfaceOf("Torso", kPbrSkinType, v, {{"Normal Map", "/n.jpg"}, {"Bump Strength", "/b.jpg"}});
  };
  const auto on = skin(true);
  EXPECT_EQ(on.normalMap, "/n.jpg");
  EXPECT_EQ(on.bumpMap, "/b.jpg");
  // off, and left to PBRSkin.dsf's default under the surface
  const auto byDefault = fv::dson::parseSurface("Torso", collect(collect({}, kPbrSkinDefinition, kPbrSkinFile), R"({"extra":[
    {"type":"studio/material/daz_brick","channels":[
      {"channel":{"id":"Normal Map","type":"float","current_value":1,"image_file":"/n.jpg"}},
      {"channel":{"id":"Bump Strength","type":"float","current_value":0.5,"image_file":"/b.jpg"}}]}]})",
                                                                 ""));
  for(const auto& off : {skin(false), byDefault})
  {
    EXPECT_TRUE(off.normalMap.empty());
    EXPECT_TRUE(off.bumpMap.empty());
    EXPECT_FALSE(fv::mat::lower(off, {}, {.skin = fv::mat::SkinMode::eRaster}).material.contains("normalTexture"));
  }
  const auto uber = surfaceOf("Torso", kUberType, {{"Normal Map", 1}, {"Bump Strength", 0.5}}, {{"Normal Map", "/n.jpg"}, {"Bump Strength", "/b.jpg"}});
  EXPECT_EQ(uber.normalMap, "/n.jpg");
  EXPECT_EQ(uber.bumpMap, "/b.jpg");
}

// pbr_skin.mdl multiplies the skin's roughness by lerp(1, Makeup Roughness Mult, Makeup Weight) with
// the makeup on, before the two lobes split it. Under a weight map it varies per texel and is left out.
TEST(Materials, MakeupRoughnessMultScalesBothLobesUnderTheMakeup)
{
  const auto lobes = [](bool enable, double weight, bool weightMap) {
    const ChannelValues v = {{"Detail Enable", false},
                             {"Makeup Enable", enable},
                             {"Makeup Weight", weight},
                             {"Makeup Roughness Mult", 1.5},
                             {"Dual Lobe Specular Enable", true},
                             {"Dual Lobe Specular Weight", 1},
                             {"Specular Lobe 1 Roughness", 0.4},
                             {"Specular Lobe 2 Roughness Mult", 0.5}};
    std::map<std::string, std::string> images;
    if(weightMap)
      images["Makeup Weight"] = "/lips_w.jpg";
    const auto s = surfaceOf("Lips", kPbrSkinType, v, images);
    return std::pair{s.dual->lobe1.value, s.dual->lobe2.value};
  };
  EXPECT_NEAR(lobes(true, 1, false).first, 0.6f, 1e-6f);
  EXPECT_NEAR(lobes(true, 1, false).second, 0.3f, 1e-6f);
  EXPECT_NEAR(lobes(true, 0.5, false).first, 0.5f, 1e-6f);
  EXPECT_NEAR(lobes(false, 1, false).first, 0.4f, 1e-6f);
  EXPECT_NEAR(lobes(false, 1, false).second, 0.2f, 1e-6f);
  EXPECT_NEAR(lobes(true, 1, true).first, 0.4f, 1e-6f);
  EXPECT_NEAR(lobes(true, 1, true).second, 0.2f, 1e-6f);
}

namespace {

float emissiveStrengthOf(const fv::mat::Lowered& l)
{
  return l.material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"].get<float>();
}

}  // namespace

// Iray Uber's emission is a flux in lumens when its units are W: DAZ converts at the file's Luminous
// Efficacy (lm/W, default 15), the one channel that only W shows. 683 lm/W belongs to pure 555 nm light.
TEST(Materials, AnEmittersWattsBecomeLumensAtItsLuminousEfficacy)
{
  const auto lamp = [](double luminance, int units, std::optional<double> efficacy) {
    ChannelValues v = {{"Emission Color", Json::array({1, 1, 1})}, {"Luminance", luminance}, {"Luminance Units", units},
                       {"Two Sided Light", false}};
    if(efficacy)
      v.push_back({"Luminous Efficacy", *efficacy});
    return surfaceOf("Bulb", kUberType, v);
  };
  const auto watts = lamp(100, 5, 60);
  ASSERT_TRUE(watts.emission && watts.emission->luminance);
  EXPECT_NEAR(watts.emission->luminance->amount, 6000.f, 1e-3f);
  EXPECT_TRUE(watts.emission->luminance->perArea);
  EXPECT_FALSE(watts.emission->luminance->twoSided);
  EXPECT_NEAR(fv::mat::emissionStrength(watts, 2.0), 0.167859f, 1e-5f);

  const auto byDefault = lamp(100, 5, std::nullopt);  // DAZ's 15 lm/W
  EXPECT_NEAR(byDefault.emission->luminance->amount, 1500.f, 1e-3f);
  EXPECT_NEAR(fv::mat::emissionStrength(byDefault, 2.0), 0.0419647f, 1e-5f);

  // lumens and luminances ignore the efficacy
  EXPECT_NEAR(lamp(500, 4, 60).emission->luminance->amount, 500.f, 1e-3f);
  const auto nits = lamp(1500, 0, 60);
  EXPECT_FALSE(nits.emission->luminance->perArea);
  EXPECT_NEAR(fv::mat::emissionStrength(nits, 0), 0.263672f, 1e-5f);

  // cd/ft2 is candela per square foot, cd/cm2 ten thousand per square metre
  EXPECT_NEAR(lamp(10, 2, std::nullopt).emission->luminance->amount, 107.639f, 1e-3f);
  EXPECT_NEAR(lamp(10, 3, std::nullopt).emission->luminance->amount, 100000.f, 1e-3f);
}

// DAZ writes Two Sided Light on by default, so a flux emitter that leaves it out spreads its flux
// over both sides, half as bright on each.
TEST(Materials, AFluxEmitterWithoutTwoSidedLightIsTwoSided)
{
  const auto panel = [](std::optional<Json> two) {
    ChannelValues v = {{"Emission Color", Json::array({1, 1, 1})}, {"Luminance", 500}, {"Luminance Units", 4}};
    if(two)
      v.push_back({"Two Sided Light", *two});
    return surfaceOf("Panel", kUberType, v);
  };
  const auto unlisted = panel(std::nullopt);
  EXPECT_TRUE(unlisted.emission->luminance->twoSided);
  EXPECT_NEAR(fv::mat::emissionStrength(unlisted, 1.0), 0.0139882f, 1e-6f);
  EXPECT_NEAR(fv::mat::emissionStrength(panel(false), 1.0), 0.0279765f, 1e-6f);
  EXPECT_FALSE(panel(0).emission->luminance->twoSided);
  EXPECT_TRUE(panel(true).emission->luminance->twoSided);
}

// irayubermaterial.mdl reads an Emission Temperature below 1 K as the colour alone, with no blackbody.
TEST(Materials, AnEmitterAtZeroKelvinKeepsItsColour)
{
  for(const float k : {0.f, 0.5f})
  {
    const auto tint = fv::mat::temperatureTint(k);
    for(int i = 0; i < 3; i++)
      EXPECT_NEAR(tint[i], 1.f, 1e-6f);
  }
  const auto s = surfaceOf("Sole", kUberType,
                           {{"Emission Color", Json::array({1, 0, 1})}, {"Emission Temperature", 0}, {"Luminance", 10},
                            {"Luminance Units", 0}, {"Two Sided Light", true}});
  const auto l = fv::mat::lower(s, {}, fv::mat::LowerOptions{.emission = 2});
  expectRgb(l.material["emissiveFactor"], 1, 0, 1, 1e-6f);  // magenta, not the red of a 1667 K tint
  EXPECT_NEAR(emissiveStrengthOf(l), 2.f, 1e-6f);
}

// The tint is MDL's math::blackbody: unit luminance in linear Rec.709, relative to D65, so a 6500 K
// blackbody (just below D65 on the Planckian locus) is a touch magenta rather than white. Values from
// the MDL SDK's mdl_blackbody, which clamps at 500 K.
TEST(Materials, AnEmittersTemperatureTintsItAsABlackbodyAgainstD65)
{
  const std::vector<std::pair<float, fv::mat::Rgb>> table = {{6500.f, {1.04760f, 0.98233f, 1.03546f}},
                                                             {2700.f, {1.92738f, 0.80603f, 0.19141f}},
                                                             {3500.f, {1.56330f, 0.89287f, 0.40321f}},
                                                             {10000.f, {0.86972f, 0.99535f, 1.43027f}},
                                                             {1111.111f, {4.22464f, 0.15182f, 0.f}},
                                                             {500.f, {6.75616f, 0.f, 0.f}},
                                                             {200.f, {6.75616f, 0.f, 0.f}}};
  for(const auto& [k, want] : table)
  {
    const auto tint = fv::mat::temperatureTint(k);
    for(int i = 0; i < 3; i++)
      EXPECT_NEAR(tint[i], want[i], 2e-3f) << k << " K, channel " << i;
  }
  for(const float k : {2700.f, 3500.f, 6500.f, 10000.f})
  {
    const auto t = fv::mat::temperatureTint(k);
    EXPECT_NEAR(0.2126f * t[0] + 0.7152f * t[1] + 0.0722f * t[2], 1.f, 2e-3f) << k << " K";
  }

  // glTF's emissiveFactor stops at 1: the peak goes into the strength
  const auto lit = [](double kelvin) {
    const auto s = surfaceOf("Panel", kUberType, {{"Emission Color", Json::array({1, 1, 1})}, {"Emission Temperature", kelvin}});
    return fv::mat::lower(s, {}, fv::mat::LowerOptions{.emission = 2});
  };
  expectRgb(lit(6500).material["emissiveFactor"], 1, 0.93770f, 0.98842f, 1e-4f);
  EXPECT_NEAR(emissiveStrengthOf(lit(6500)), 2.09519f, 1e-4f);
  expectRgb(lit(2700).material["emissiveFactor"], 1, 0.41820f, 0.09931f, 1e-4f);
  EXPECT_NEAR(emissiveStrengthOf(lit(2700)), 3.85477f, 1e-4f);

  // a non-Iray "emission" colour has no Emission Temperature and gets no blackbody
  const auto glow = fv::dson::parseSurface(
      "Glow", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
        {"channel":{"id":"emission","type":"color","current_value":[1,1,1]}}]}]})"));
  ASSERT_TRUE(glow.emission);
  EXPECT_EQ(glow.emission->temperature, 0.f);
  const auto plain = fv::mat::lower(glow, {}, {});
  expectRgb(plain.material["emissiveFactor"], 1, 1, 1, 1e-6f);
  EXPECT_NEAR(emissiveStrengthOf(plain), 1.f, 1e-6f);
}

// ---- The maps on channels that are read: each DAZ texture instance is its value times its map ----

namespace {

constexpr const char* kSlw = "/Runtime/Textures/DAZ/Characters/Genesis9/Base/Skin_SLW.jpg";

// CJ Evangeline's head (PBRSkin) as her .duf gives it: Dual Lobe Specular Weight 1 times Skin_SLW,
// a flat 0.6 grey, over both lobes; the head's own roughness map under them
fv::mat::Surface cjHead(bool weightMap = true)
{
  ChannelValues v = {{"Detail Enable", false},
                     {"Dual Lobe Specular Enable", true},
                     {"Dual Lobe Specular Weight", 1},
                     {"Dual Lobe Specular Reflectivity", 0.25},
                     {"Dual Lobe Specular Ratio", 0.4578947},
                     {"Specular Lobe 1 Roughness", 1},
                     {"Specular Lobe 2 Roughness Mult", 0.55}};
  std::map<std::string, std::string> images = {{"Specular Lobe 1 Roughness", "/head_r.jpg"}};
  if(weightMap)
    images["Dual Lobe Specular Weight"] = kSlw;
  return surfaceOf("Head", kPbrSkinType, v, images);
}

const fv::mat::TextureRequest& specularRequest(const fv::mat::Lowered& l)
{
  return requestOf(l, l.material["extensions"]["KHR_materials_specular"]["specularTexture"]);
}

}  // namespace

// pbr_skin.mdl lays both dual lobes under one weight, dual_lobe_specular_weight, which scales their
// Fresnel curve normal to grazing, and the brick graph feeds it the weight's value times its map. A
// map is a per-texel multiplier of both: specularTexture's alpha for the broad lobe, and the same
// cooked image's red for the tight lobe's clearcoat.
TEST(Materials, ADualLobeWeightMapSaysWhereBothLobesShine)
{
  const auto s = cjHead();
  ASSERT_TRUE(s.dual);
  EXPECT_EQ(s.dual->weightMap, kSlw);
  // 1 x 0.25 x 2 = 0.5, lobe 2 (1 x 0.55) is the tight one with 0.4578947 of it
  const auto lobes = fv::mat::specularLobes(s);
  EXPECT_NEAR(lobes.intensity, 0.2710526f, 1e-5f);
  ASSERT_TRUE(lobes.clearcoat);
  EXPECT_NEAR(lobes.clearcoat->weight, 0.2289474f, 1e-5f);
  EXPECT_EQ(lobes.intensityMaps, std::vector<std::string>{kSlw});
  EXPECT_EQ(lobes.clearcoat->weightMaps, lobes.intensityMaps);

  const auto l = fv::mat::lower(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.2710526f, 1e-5f);
  ASSERT_TRUE(specularOf(l).contains("specularTexture"));
  const auto& spec = specularRequest(l);
  EXPECT_EQ(spec.rel, kSlw);
  EXPECT_EQ(spec.channels, "wa");
  EXPECT_FALSE(spec.srgb);
  EXPECT_TRUE(spec.composed());
  EXPECT_NEAR(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.2289474f, 1e-5f);
  ASSERT_TRUE(clearcoatOf(l).contains("clearcoatTexture"));
  EXPECT_EQ(fv::mat::requestKey(requestOf(l, clearcoatOf(l)["clearcoatTexture"])), fv::mat::requestKey(spec));
  EXPECT_EQ(requestOf(l, clearcoatOf(l)["clearcoatRoughnessTexture"]).rel, "/head_r.jpg");

  // no map (Polina 9's and Leyla 9's skin): flat as before
  const auto flat = fv::mat::lower(cjHead(false), {}, {});
  EXPECT_NEAR(specularOf(flat)["specularFactor"].get<float>(), 0.2710526f, 1e-5f);
  EXPECT_FALSE(specularOf(flat).contains("specularTexture"));
  EXPECT_FALSE(clearcoatOf(flat).contains("clearcoatTexture"));
  EXPECT_EQ(requestOf(flat, clearcoatOf(flat)["clearcoatRoughnessTexture"]).rel, "/head_r.jpg");

  // CJ's nails: their top coat takes the slot and the tight lobe folds back into the specular,
  // 0.425 + 0.075, both under the same map, so the sum keeps it
  const auto nails = surfaceOf("Fingernails", kPbrSkinType,
                               {{"Detail Enable", false},
                                {"Dual Lobe Specular Enable", true},
                                {"Dual Lobe Specular Weight", 1},
                                {"Dual Lobe Specular Reflectivity", 0.25},
                                {"Dual Lobe Specular Ratio", 0.15},
                                {"Specular Lobe 1 Roughness", 1},
                                {"Specular Lobe 2 Roughness Mult", 0.55},
                                {"Top Coat Enable", true},
                                {"Top Coat Weight", 0.25}},
                               {{"Dual Lobe Specular Weight", kSlw}});
  const auto ln = fv::mat::lower(nails, {}, {});
  EXPECT_NEAR(specularOf(ln)["specularFactor"].get<float>(), 0.5f, 1e-5f);
  ASSERT_TRUE(specularOf(ln).contains("specularTexture"));
  EXPECT_EQ(specularRequest(ln).rel, kSlw);
  EXPECT_NEAR(clearcoatOf(ln)["clearcoatFactor"].get<float>(), 0.25f, 1e-5f);
  EXPECT_FALSE(clearcoatOf(ln).contains("clearcoatTexture"));  // the coat's weight has no map
}

// Iray Uber stacks its glossy layer over the dual lobe, two lobes with weights of their own. glTF
// sums them into one specularFactor, glossy x Mg + dual x Md, which no product of maps gives: the
// sum stays flat, and only the tight lobe (the clearcoat, the dual's alone) keeps the map.
TEST(Materials, AGlossyLayerAndADualLobeTogetherLeaveTheSumUnmapped)
{
  const auto s = fv::dson::parseSurface("Face", layered({kSkin, R"({"extra":[{"channels":[
    {"channel":{"id":"Glossy Layered Weight","type":"float","current_value":0.5}},
    {"channel":{"id":"Dual Lobe Specular Weight","type":"float","current_value":0.5,"image_file":"/Runtime/Textures/face_s.jpg"}}
  ]}]})"}));
  const auto lobes = fv::mat::specularLobes(s);
  ASSERT_TRUE(lobes.clearcoat);
  EXPECT_NEAR(lobes.clearcoat->weight, 0.25f, 1e-5f);
  EXPECT_EQ(lobes.clearcoat->weightMaps, std::vector<std::string>{"/Runtime/Textures/face_s.jpg"});
  EXPECT_NEAR(lobes.intensity, 0.75f, 1e-5f);
  EXPECT_NEAR(lobes.glossyShare, 0.6666667f, 1e-5f);
  EXPECT_TRUE(lobes.intensityMaps.empty());
  const auto l = fv::mat::lower(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.75f, 1e-5f);
  EXPECT_FALSE(specularOf(l).contains("specularTexture"));
  ASSERT_TRUE(clearcoatOf(l).contains("clearcoatTexture"));
  EXPECT_EQ(requestOf(l, clearcoatOf(l)["clearcoatTexture"]).rel, "/Runtime/Textures/face_s.jpg");
}

// Addy Florence's head (Iray Uber, with her !Material ALL preset): Dual Lobe Specular Reflectivity 1
// times Head S.jpg, a dark map. Under today's fold the reflectivity is in specularFactor, so its map
// multiplies it like the weight's (and also dims grazing, which Iray's F0-only reflectivity does not).
TEST(Materials, AReflectivityMapScalesTheDualLobeLikeItsWeight)
{
  const auto s = surfaceOf("Head", kUberType,
                           {{"Glossy Layered Weight", 0},
                            {"Glossy Roughness", 0.5},
                            {"Dual Lobe Specular Weight", 0.5486111},
                            {"Dual Lobe Specular Reflectivity", 1},
                            {"Dual Lobe Specular Ratio", 0.85},
                            {"Specular Lobe 1 Roughness", 0.5},
                            {"Specular Lobe 2 Roughness", 0.4},
                            {"Top Coat Weight", 0.01}},
                           {{"Dual Lobe Specular Reflectivity", "/head_s.jpg"}});
  ASSERT_TRUE(s.dual);
  EXPECT_EQ(s.dual->reflectivityMap, "/head_s.jpg");
  EXPECT_TRUE(s.dual->weightMap.empty());
  // 0.5486111 x 1 x 2 = 1.0972222; lobe 2 (0.4) is the tight one with 0.15 of it
  const auto lobes = fv::mat::specularLobes(s);
  ASSERT_TRUE(lobes.clearcoat);
  EXPECT_NEAR(lobes.clearcoat->weight, 0.1645833f, 1e-5f);
  EXPECT_NEAR(lobes.intensity, 0.9326389f, 1e-5f);
  EXPECT_EQ(lobes.intensityMaps, std::vector<std::string>{"/head_s.jpg"});
  // the top coat holds the slot: the tight lobe folds back, under the same map
  const auto l = fv::mat::lower(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 1.0f, 1e-5f);
  ASSERT_TRUE(specularOf(l).contains("specularTexture"));
  EXPECT_EQ(specularRequest(l).rel, "/head_s.jpg");
  EXPECT_EQ(specularRequest(l).channels, "wa");
  EXPECT_NEAR(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.01f, 1e-6f);
  EXPECT_FALSE(clearcoatOf(l).contains("clearcoatTexture"));
}

// A lobe with a weight map and a reflectivity map is their product, which the cooker multiplies
// (mul=): Madelein's face maps both of her dual lobe's.
TEST(Materials, TwoMapsOnOneLobeAreMultipliedByTheCooker)
{
  auto s = cjHead();
  s.dual->reflectivityMap = "/Runtime/Textures/head_s.jpg";
  const auto l = fv::mat::lower(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.2710526f, 1e-5f);
  EXPECT_NEAR(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.2289474f, 1e-5f);
  const auto& spec = specularRequest(l);
  EXPECT_EQ(spec.rel, kSlw);
  EXPECT_EQ(spec.mulRels, std::vector<std::string>{"/Runtime/Textures/head_s.jpg"});
  EXPECT_TRUE(spec.composed());
  const std::string key = fv::mat::requestKey(spec);
  EXPECT_NE(key.find("|mul:/Runtime/Textures/head_s.jpg"), std::string::npos);
  EXPECT_EQ(fv::mat::requestKey(requestOf(l, clearcoatOf(l)["clearcoatTexture"])), key);
  EXPECT_NE(key, fv::mat::requestKey(specularRequest(fv::mat::lower(cjHead(), {}, {}))));

  // a layered image multiplied in gives its bottom layer: the cooker multiplies plain images
  auto layeredMap = cjHead();
  layeredMap.dual->reflectivityMap = std::string("/x.png") + fv::mat::kLayerSeparator + "/y.png";
  EXPECT_EQ(specularRequest(fv::mat::lower(layeredMap, {}, {})).mulRels, std::vector<std::string>{"/x.png"});
}

// Iray Uber's glossy layer is add_top_coat with Glossy Layered Weight as its weight (the whole
// Fresnel curve) and Glossy Reflectivity as its F0, each its value times its map in the 4-Layer Uber
// graph. Clothing often puts one gloss map on both: the product squares it.
TEST(Materials, AGlossyLayersMapsSayWhereItShines)
{
  const auto s = surfaceOf("Dress", kUberType, {{"Glossy Layered Weight", 1}, {"Glossy Reflectivity", 0.5}, {"Glossy Roughness", 0.3}},
                           {{"Glossy Layered Weight", "/gloss.jpg"}, {"Glossy Reflectivity", "/gloss.jpg"}});
  EXPECT_EQ(s.glossy.weightMap, "/gloss.jpg");
  EXPECT_EQ(s.glossy.reflectivityMap, "/gloss.jpg");
  const auto lobes = fv::mat::specularLobes(s);
  EXPECT_NEAR(lobes.intensity, 1.0f, 1e-6f);
  EXPECT_EQ(lobes.intensityMaps, (std::vector<std::string>{"/gloss.jpg", "/gloss.jpg"}));
  const auto l = fv::mat::lower(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 1.0f, 1e-6f);
  ASSERT_TRUE(specularOf(l).contains("specularTexture"));
  EXPECT_EQ(specularRequest(l).rel, "/gloss.jpg");
  EXPECT_EQ(specularRequest(l).mulRels, std::vector<std::string>{"/gloss.jpg"});
  EXPECT_FALSE(l.material["extensions"].contains("KHR_materials_clearcoat"));

  // PBRSkin's leftover Glossy Layered Weight (Genesis 9's 0.0008) has no say over the dual lobe's maps
  auto head = cjHead();
  head.glossy.weight    = 0.0008046585f;
  head.glossy.weightMap = "/stale.jpg";
  EXPECT_EQ(fv::mat::specularLobes(head).intensityMaps, std::vector<std::string>{kSlw});

  // a 3Delight-era shader's glossy layer is its own: maps read under Iray's names are dropped
  const auto legacy = fv::dson::parseSurface("Dress", channelsOf(R"({"extra":[{"type":"studio/material/daz_shader","channels":[
    {"channel":{"id":"Lighting Model","type":"float","current_value":0}},
    {"channel":{"id":"specular_strength","type":"float","current_value":0.5}},
    {"channel":{"id":"Glossy Weight","type":"float","current_value":0.7,"image_file":"/gw.jpg"}},
    {"channel":{"id":"Glossy Reflectivity","type":"float","current_value":0.5,"image_file":"/gr.jpg"}}
  ]}]})"));
  EXPECT_EQ(legacy.shader, fv::mat::Surface::Shader::eDazDefault);
  EXPECT_TRUE(legacy.glossy.weightMap.empty());
  EXPECT_TRUE(legacy.glossy.reflectivityMap.empty());
}

// pbr_skin.mdl lets Translucency Weight of the diffuse light in, the value times its map: the map
// says where. The Volume (Iray) and thin modes take the weight, so they take the map, in
// transmissionTexture's red; measured flesh lets all of it in. Nails and teeth are thin in every
// path-traced mode, so theirs applies under measured skin too.
TEST(Materials, TheTranslucencyWeightsMapSaysWhereLightEnters)
{
  const auto at = [](const char* name) {
    return surfaceOf(name, kPbrSkinType,
                     {{"Specular Lobe 2 Roughness Mult", 0.5}, {"Translucency Enable", true}, {"Translucency Weight", 0.8}},
                     {{"Translucency Weight", "/trans_w.jpg"}});
  };
  const auto face = at("Face");
  ASSERT_TRUE(face.translucency);
  EXPECT_EQ(face.translucency->weightMap, "/trans_w.jpg");
  const auto transmissionOf = [](const fv::mat::Lowered& l) { return l.material["extensions"]["KHR_materials_transmission"]; };
  for(const auto mode : {fv::mat::SkinMode::eVolume, fv::mat::SkinMode::eThin})
  {
    const auto l = fv::mat::lower(face, {}, {.skin = mode});
    EXPECT_NEAR(transmissionOf(l)["transmissionFactor"].get<float>(), 0.8f, 1e-6f);
    ASSERT_TRUE(transmissionOf(l).contains("transmissionTexture"));
    const auto& r = requestOf(l, transmissionOf(l)["transmissionTexture"]);
    EXPECT_EQ(r.rel, "/trans_w.jpg");
    EXPECT_EQ(r.channels, "");
    EXPECT_FALSE(r.srgb);
  }
  const auto measured = fv::mat::lower(face, {}, {.skin = fv::mat::SkinMode::eMeasured});
  EXPECT_NEAR(transmissionOf(measured)["transmissionFactor"].get<float>(), 1.0f, 1e-6f);
  EXPECT_FALSE(transmissionOf(measured).contains("transmissionTexture"));
  EXPECT_FALSE(fv::mat::lower(face, {}, {.skin = fv::mat::SkinMode::eRaster}).material["extensions"].contains("KHR_materials_transmission"));

  const auto nails = fv::mat::lower(at("Fingernails"), {}, {.skin = fv::mat::SkinMode::eMeasured});
  EXPECT_NEAR(transmissionOf(nails)["transmissionFactor"].get<float>(), 0.8f, 1e-6f);
  ASSERT_TRUE(transmissionOf(nails).contains("transmissionTexture"));
  EXPECT_EQ(requestOf(nails, transmissionOf(nails)["transmissionTexture"]).rel, "/trans_w.jpg");
}

// Iray Uber hair sends Translucency Weight of the diffuse light through the strand's colour; with
// a weight map, only where the map is light.
TEST(Materials, HairTranslucencyKeepsItsWeightMap)
{
  const auto s = surfaceOf("Hair", kUberType, {{"Glossy Roughness", 0.4}, {"Translucency Weight", 0.5}, {"Translucency Color", Json::array({1, 0.8, 0.5})}},
                           {{"Translucency Weight", "/hair_tw.jpg"}, {"Translucency Color", "/hair_t.jpg"}});
  const auto m = fv::mat::hairTranslucency(s);
  ASSERT_TRUE(m);
  EXPECT_EQ(m->weightMap, "/hair_tw.jpg");
  EXPECT_FLOAT_EQ(m->weight, 0.5f);
  const auto l = fv::mat::lower(s, {}, {});
  ASSERT_FALSE(l.requests.empty());
  ASSERT_TRUE(l.requests[0].makeup);
  EXPECT_EQ(l.requests[0].makeup->weightRel, "/hair_tw.jpg");
}

// PBRSkin's Dual Lobe Specular Roughness Mult has no map in its brick graph: a scalar on lobe 1's
// roughness map. An image a file gives it is ignored, as in Iray.
TEST(Materials, TheDualLobeRoughnessMultHasNoMap)
{
  const auto s = surfaceOf("Face", kPbrSkinType,
                           {{"Specular Lobe 2 Roughness Mult", 1}, {"Dual Lobe Specular Weight", 1}, {"Specular Lobe 1 Roughness", 0.8},
                            {"Dual Lobe Specular Roughness Mult", 0.5}},
                           {{"Dual Lobe Specular Roughness Mult", "/mult.jpg"}});
  ASSERT_TRUE(s.dual);
  EXPECT_NEAR(s.dual->lobe1.value, 0.4f, 1e-6f);
  EXPECT_TRUE(s.dual->lobe1.map.empty());
  for(const auto& r : fv::mat::lower(s, {}, {}).requests)
    EXPECT_NE(r.rel, "/mult.jpg");
}
