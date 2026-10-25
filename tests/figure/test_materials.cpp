// A DSON surface's channels become a Surface (Iray Uber, PBRSkin), and a Surface becomes the glTF
// material the renderer shows: factors, the textures it asks the cooker for, the skin's volume, and
// edits laid over it. A port of the material pipeline in src/lib/materials and src/lib/rtx/gltf.ts.

#include <gtest/gtest.h>

#include <cmath>
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

}  // namespace
