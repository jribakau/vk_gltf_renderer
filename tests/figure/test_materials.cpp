// A DSON surface's channels become a Surface (Iray Uber, PBRSkin), and a Surface becomes the glTF
// material the renderer shows: factors, the textures it asks the cooker for, the skin's volume, and
// edits laid over it. A port of the material pipeline in src/lib/materials and src/lib/rtx/gltf.ts.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>

#include <figure/dson/eye_materials.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/dson/materials.hpp>
#include <figure/dson/skin_materials.hpp>
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

  // the full range is kBumpHeight of surface: on a 4K map of a face's 0.43 m per UV unit, the
  // slope per texel the fixed bake gave (10), within the rounding
  EXPECT_NEAR(fv::mat::bumpHeightInUv(0.43f) * 4096, 10.f, 1.f);
  EXPECT_EQ(fv::mat::bumpHeightInUv(0), 0.f);
  // a detail tiled 4 times covers a quarter of the surface per UV unit
  EXPECT_NEAR(fv::mat::bumpHeightInUv(0.4f, 4), 4 * fv::mat::bumpHeightInUv(0.4f), 1e-6f);

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

}  // namespace

// PBRSkin's detail layer is a relief of its own when it is another map or the same one tiled; the
// surface's own normal map again, untiled, would only double every tilt, so its normal is dropped
// and its roughness multiplier kept.
TEST(Materials, ADetailLayerRepeatingTheNormalMapAddsNoRelief)
{
  const auto skin = [](const char* detailMap, int tiles) {
    const std::string text = std::string(R"({"extra":[{"type":"studio/material/daz_brick","channels":[
      {"channel":{"id":"Detail Enable","type":"bool","current_value":true}},
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
