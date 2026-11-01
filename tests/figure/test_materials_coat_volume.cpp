// The top coat and PBRSkin's volume.

#include "materials_test_util.hpp"

using namespace materials_test;

namespace {

// ---- The top coat ----

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
    const auto  l   = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eRaster});
    const auto& ext = l.material["extensions"];
    EXPECT_NEAR(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.0375f, 1e-5f);
    EXPECT_NEAR(clearcoatOf(l)["clearcoatRoughnessFactor"].get<float>(), 0.24f, 1e-5f);
    EXPECT_NEAR(ext["KHR_materials_specular"]["specularFactor"].get<float>(), 0.2125f, 1e-5f);
    EXPECT_NEAR(l.material["pbrMetallicRoughness"]["roughnessFactor"].get<float>(), 0.6f, 1e-5f);
  }
  // on: the coat takes the slot, and the tight lobe folds back into the base specular
  const auto on = head(true);
  ASSERT_TRUE(on.topCoat);
  const auto l = loweredJson(on, {}, {.skin = fv::mat::SkinMode::eRaster});
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
  const auto  l  = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eRaster});
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
  const auto edited   = loweredJson(s, e, {.skin = fv::mat::SkinMode::eRaster});
  EXPECT_FLOAT_EQ(clearcoatOf(edited)["clearcoatFactor"].get<float>(), 0.7f);
  EXPECT_TRUE(clearcoatOf(edited).contains("clearcoatTexture"));

  // CJ Evangeline's own head keeps the mask at weight 0: no coat
  const auto bare = head(0);
  EXPECT_FALSE(bare.topCoat);
  EXPECT_FALSE(loweredJson(bare, {}, {.skin = fv::mat::SkinMode::eRaster}).material["extensions"].contains("KHR_materials_clearcoat"));

  // Iray Uber's weight map (Sienna Thornbrook's skin)
  const auto uber = surfaceOf("Torso", kUberType, {{"Glossy Roughness", 0.4}, {"Top Coat Weight", 0.45}}, {{"Top Coat Weight", "/Runtime/Textures/face_s.jpg"}});
  ASSERT_TRUE(uber.topCoat);
  EXPECT_EQ(uber.topCoat->weightMap, "/Runtime/Textures/face_s.jpg");
  const auto ul = loweredJson(uber, {}, {});
  ASSERT_TRUE(clearcoatOf(ul).contains("clearcoatTexture"));
  EXPECT_EQ(requestOf(ul, clearcoatOf(ul)["clearcoatTexture"]).rel, "/Runtime/Textures/face_s.jpg");

  // Specular/Glossiness mode: the roughness is 1 - Top Coat Glossiness, and a glossiness map is no roughness map
  const auto gloss = surfaceOf("Hair", kUberType, {{"Base Mixing", 1}, {"Top Coat Weight", 0.5}, {"Top Coat Glossiness", 0.8}},
                               {{"Top Coat Glossiness", "/gloss.png"}});
  ASSERT_TRUE(gloss.topCoat);
  EXPECT_NEAR(gloss.topCoat->roughness, 0.2f, 1e-6f);
  EXPECT_TRUE(gloss.topCoat->roughnessMap.empty());
  EXPECT_FALSE(clearcoatOf(loweredJson(gloss, {}, {})).contains("clearcoatRoughnessTexture"));

  // the MDL takes the weight in 0..1: the Sporty Ponytails hair's 2 is a full coat
  const auto ponytail = surfaceOf("Bangs1", kUberType, {{"Top Coat Weight", 2}, {"Top Coat Roughness", 1}});
  ASSERT_TRUE(ponytail.topCoat);
  EXPECT_FLOAT_EQ(ponytail.topCoat->weight, 1);
  EXPECT_FLOAT_EQ(clearcoatOf(loweredJson(ponytail, {}, {}))["clearcoatFactor"].get<float>(), 1);
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
    const auto l      = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eRaster});
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
TEST(Materials, ACoatsColourTintsItsReflection)
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
  EXPECT_NEAR(clearcoatOf(loweredJson(ava, {}, {.skin = fv::mat::SkinMode::eRaster}))["clearcoatFactor"].get<float>(), 0.11f * lum, 1e-5f);
  EXPECT_NEAR(clearcoatOf(loweredJson(head(1), {}, {.skin = fv::mat::SkinMode::eRaster}))["clearcoatFactor"].get<float>(), 0.11f, 1e-5f);
  EXPECT_NEAR(clearcoatOf(loweredJson(ava, {}, {.forFile = true}))["clearcoatFactor"].get<float>(), 0.11f * lum, 1e-5f);
  EXPECT_FALSE(clearcoatOf(loweredJson(ava, {}, {.skin = fv::mat::SkinMode::eRaster})).contains("fvCoatColor"));

  // The path tracer tints the reflection, and the coat keeps its weight, so the base keeps its share
  const auto traced = clearcoatOf(loweredJson(ava, {}, {}));
  EXPECT_NEAR(traced["clearcoatFactor"].get<float>(), 0.11f, 1e-6f);
  ASSERT_TRUE(traced.contains("fvCoatColor"));
  EXPECT_NEAR(traced["fvCoatColor"][0].get<float>(), lum, 1e-6f);
  // white says nothing
  EXPECT_FALSE(clearcoatOf(loweredJson(head(1), {}, {})).contains("fvCoatColor"));

  // Iray Uber's the same, its map an image of its own, read as colour
  const auto uber = surfaceOf("Shirt", kUberType, {{"Top Coat Weight", 0.4}, {"Top Coat Color", Json::array({0.5, 0.5, 0.5})}}, {{"Top Coat Color", "/Runtime/Textures/coat_c.jpg"}});
  ASSERT_TRUE(uber.topCoat);
  EXPECT_EQ(uber.topCoat->colorMap, "/Runtime/Textures/coat_c.jpg");
  const auto l = loweredJson(uber, {}, {});
  EXPECT_FLOAT_EQ(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.4f);
  const auto& request = requestOf(l, clearcoatOf(l)["fvCoatColorTexture"]);
  EXPECT_EQ(request.rel, "/Runtime/Textures/coat_c.jpg");
  EXPECT_TRUE(request.srgb);
  // and neither view without a coloured coat asks for its map
  EXPECT_FALSE(clearcoatOf(loweredJson(uber, {}, {.skin = fv::mat::SkinMode::eRaster})).contains("fvCoatColorTexture"));
  EXPECT_FALSE(clearcoatOf(loweredJson(uber, {}, {.forFile = true})).contains("fvCoatColorTexture"));
}

// Top Coat Color Effect's Scatter & Transmit modes take the colour's share out of the weight and
// normalise the tint (irayubermaterial.mdl, transmit_color_calc); Scatter Only leaves them alone
TEST(Materials, ACoatsColorEffectMovesItsShareIntoTheWeight)
{
  const auto shirt = [](int effect) {
    return surfaceOf("Shirt", kUberType, {{"Top Coat Weight", 0.5}, {"Top Coat Color Effect", effect}, {"Top Coat Color", Json::array({1.0, 0.5, 0.5})}});
  };
  const float half = fv::mat::srgbToLinear(0.5f);
  const auto  only = shirt(0);
  EXPECT_NEAR(only.topCoat->weight, 0.5f, 1e-6f);
  EXPECT_NEAR(only.topCoat->color[1], half, 1e-6f);

  const auto mean = shirt(1);
  EXPECT_NEAR(mean.topCoat->weight, 0.5f * (1 + 2 * half) / 3, 1e-5f);
  EXPECT_NEAR(mean.topCoat->color[0], 1.0f, 1e-6f);
  EXPECT_NEAR(mean.topCoat->color[1], half, 1e-6f);

  const auto luminance = shirt(2);
  EXPECT_NEAR(luminance.topCoat->weight, 0.5f * (0.2126f + 0.7874f * half), 1e-5f);

  // a brighter colour normalises: only its hue is left in the tint
  const auto bright = surfaceOf("Shirt", kUberType, {{"Top Coat Weight", 0.5}, {"Top Coat Color Effect", 1}, {"Top Coat Color", Json::array({0.5, 0.5, 0.25})}});
  EXPECT_NEAR(bright.topCoat->color[0], 1.0f, 1e-6f);
  EXPECT_NEAR(bright.topCoat->color[1], 1.0f, 1e-6f);
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
  const auto  l   = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eVolume});
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
      const auto  l   = loweredJson(s, {}, {.skin = mode});
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
  EXPECT_FLOAT_EQ(loweredJson(on, {}, {.skin = fv::mat::SkinMode::eMeasured})
                      .material["extensions"]["KHR_materials_transmission"]["transmissionFactor"]
                      .get<float>(),
                  1);
  // Iray Uber has no switch
  EXPECT_FALSE(fv::dson::parseSurface("Face", channelsOf(kSkin)).translucency->thinWalled);

  // a surface that looks like PBRSkin only by a stray preset's marker, without the switch, keeps its inside
  const auto stray = surfaceOf("Face", kPbrSkinType, {{"Detail Enable", false}, {"Translucency Weight", 0.8}});
  ASSERT_TRUE(stray.translucency);
  EXPECT_FALSE(stray.translucency->thinWalled);
  const auto  l   = loweredJson(stray, {}, {.skin = fv::mat::SkinMode::eMeasured});
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
  const auto  thin   = loweredJson(skin, {}, {.skin = fv::mat::SkinMode::eThin});
  const auto& factor = thin.material["extensions"]["KHR_materials_scatter"]["multiscatterColorFactor"];
  for(int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(factor[i].get<float>(), 1);

  v.erase(v.begin());  // Iray Uber
  const auto uber = surfaceOf("Face", kUberType, v, images);
  const auto  uberThin = loweredJson(uber, {}, {.skin = fv::mat::SkinMode::eThin});
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
  const auto lowered = loweredJson(moisture, {}, {});
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
  const auto  lowered = loweredJson(moisture, {}, {});
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
  const auto lowered = loweredJson(duplicate, {}, {});
  const auto& ext    = lowered.material["extensions"]["FV_materials_detail"];
  EXPECT_FALSE(ext.contains("detailNormalTexture"));
  EXPECT_FLOAT_EQ(ext["detailRoughnessFactor"].get<float>(), 0.8f);

  EXPECT_EQ(skin("/Runtime/Textures/body_nm.jpg", 8).detail->map, "/Runtime/Textures/body_nm.jpg");  // tiled: fine detail
  EXPECT_EQ(skin("/Runtime/Textures/pores_nm.jpg", 1).detail->map, "/Runtime/Textures/pores_nm.jpg");  // a map of its own
}
