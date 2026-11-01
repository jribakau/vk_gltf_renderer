// How DAZ's channels are read: a map's texture instance and tiling, cutouts, lobes, emitters, weight maps.

#include "materials_test_util.hpp"

using namespace materials_test;

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
  const auto lowered = loweredJson(s, {}, {});
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
  EXPECT_EQ(tilingOf(loweredJson(twice, {}, {}).material["pbrMetallicRoughness"]["baseColorTexture"])->scale, (std::array<float, 2>{5, 5}));

  // the bump alone: tiled, and its height per its own tile
  const auto bumped = fv::dson::parseSurface("Shorts", layered({kTiledCloth, R"({"extra":[{"channels":[{"channel":{"id":"Normal Map","image_file":""}}]}]})"}));
  fv::mat::LowerOptions o;
  o.metresPerUv    = 0.5f;
  const auto bump  = loweredJson(bumped, {}, o);
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
  const auto base = tilingOf(loweredJson(s, e, {}).material["pbrMetallicRoughness"]["baseColorTexture"]);
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
  const auto t      = tilingOf(loweredJson(stretched, turned, {}).material["pbrMetallicRoughness"]["baseColorTexture"]);
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
    const auto lowered = loweredJson(s, e, {});
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

const nlohmann::json& specularOf(const LoweredJson& l)
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
  const auto l = loweredJson(s, {}, {});
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
  const auto mixed = loweredJson(surfaceOf("Shirt", kUberType, dual), {}, {});
  expectRgb(specularOf(mixed)["specularColorFactor"], 0.47603f, 0.36725f, 1.0f, 1e-4f);

  // no Glossy Color: white, and no map
  const auto skin = loweredJson(fv::dson::parseSurface("Face", channelsOf(kSkin)), {}, {});
  expectRgb(specularOf(skin)["specularColorFactor"], 1, 1, 1, 0);
  EXPECT_FALSE(specularOf(skin).contains("specularColorTexture"));

  // the tear film is clear water whatever its Glossy Color
  const auto moisture = fv::dson::parseSurface("EyeMoisture Left", layered({kG9Moisture, kGreyGlossyColor}));
  expectRgb(specularOf(loweredJson(moisture, {}, {.underTearFilm = true}))["specularColorFactor"], 1, 1, 1, 0);

  // a 3Delight-era surface's channel of that name is not Iray's
  const auto legacy = surfaceOf("Shirt", "studio/material/daz_shader", {{"Glossy Color", Json::array({0.5, 0.25, 1})}});
  EXPECT_EQ(legacy.glossy.color, (fv::mat::Rgb{1, 1, 1}));
  expectRgb(specularOf(loweredJson(legacy, {}, {}))["specularColorFactor"], 1, 1, 1, 0);
}

// Iray tints what a refracting surface lets through by Glossy Color while Share Glossy Inputs is on,
// else by Refraction Color. glTF tints transmission by the base colour, which goes to that tint.
TEST(Materials, RefractionLetsLightThroughInItsOwnColour)
{
  const auto base = [](ChannelValues extra) {
    ChannelValues v = {{"diffuse", Json::array({0.5, 0.5, 0.5})}, {"Glossy Layered Weight", 1}, {"Glossy Roughness", 0},
                       {"Refraction Weight", 1}, {"Refraction Index", 1.33}, {"Glossy Color", Json::array({0.5, 0.25, 1})}};
    v.insert(v.end(), extra.begin(), extra.end());
    return loweredJson(surfaceOf("Wine", kUberType, v), {}, {}).material["pbrMetallicRoughness"]["baseColorFactor"];
  };
  expectRgb(base({}), 0.21404f, 0.05088f, 1.0f, 1e-4f);  // shared by default
  expectRgb(base({{"Share Glossy Inputs", true}}), 0.21404f, 0.05088f, 1.0f, 1e-4f);
  expectRgb(base({{"Share Glossy Inputs", false}}), 1, 1, 1, 1e-6f);  // no Refraction Color: white
  expectRgb(base({{"Share Glossy Inputs", false}, {"Refraction Color", Json::array({1, 0.5, 1})}}), 1, 0.21404f, 1, 1e-4f);

  // the eye's water stays clear
  const auto moisture = fv::dson::parseSurface("EyeMoisture Left", layered({kG9Moisture, kGreyGlossyColor}));
  expectRgb(loweredJson(moisture, {}, {.underTearFilm = true}).material["pbrMetallicRoughness"]["baseColorFactor"], 1, 1, 1, 0);
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
  const auto pbrOf = [](const LoweredJson& l) { return l.material["pbrMetallicRoughness"]; };
  const auto iorOf = [](const LoweredJson& l) { return l.material["extensions"]["KHR_materials_ior"]["ior"].get<float>(); };

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
    const auto l = loweredJson(s, {}, {});
    EXPECT_FLOAT_EQ(pbrOf(l)["metallicFactor"].get<float>(), 0);
    EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.350877f, 1e-4f);
    EXPECT_NEAR(iorOf(l), 77.987f, 1e-2f);
    EXPECT_NEAR(pbrOf(l)["baseColorFactor"][0].get<float>(), 1.0f, 1e-4f);
  }
  // (b) weights under 1 together: the rest is lost, and the diffuse is 0.2 of what lies under 0.4
  {
    const auto l = loweredJson(weighted(0.2, 0.4, 1), {}, {});
    expectRgb(pbrOf(l)["baseColorFactor"], 0.33333f, 0.33333f, 0.33333f, 1e-4f);
    EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.421053f, 1e-4f);
    EXPECT_NEAR(iorOf(l), 77.987f, 1e-2f);
    EXPECT_EQ(l.effective.color, (fv::mat::Rgb{1, 1, 1}));  // the panel shows the file's colour
  }
  // (c) hair's faint sheen
  {
    const auto s = weighted(1, 0.02, 1);
    EXPECT_NEAR(s.glossy.weight, 0.019608f, 1e-5f);
    const auto l = loweredJson(s, {}, {});
    EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.020640f, 1e-5f);
    EXPECT_NEAR(iorOf(l), 77.987f, 1e-2f);
    EXPECT_NEAR(pbrOf(l)["baseColorFactor"][0].get<float>(), 1.0f, 1e-4f);
  }
  // (d) no glossy weight: no reflection, whatever the layered weight says
  {
    const auto l = loweredJson(weighted(1, 0, 0.325037), {}, {});
    EXPECT_FLOAT_EQ(specularOf(l)["specularFactor"].get<float>(), 0);
    EXPECT_FALSE(l.material["extensions"].contains("KHR_materials_ior"));
  }
  // (e) Glossy Weight's map says where the surface is glossy: its grey in specularTexture's alpha.
  // Glossy Reflectivity's has no part in this mode.
  {
    const auto s = weighted(1, 1, 1, {{"Glossy Reflectivity", 0.5}}, {{"Glossy Weight", "/hair_gw.jpg"}, {"Glossy Reflectivity", "/hair_gr.jpg"}});
    EXPECT_EQ(s.glossy.weightMap, "/hair_gw.jpg");
    const auto l = loweredJson(s, {}, {});
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
    const auto l = loweredJson(s, {}, {});
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
    const auto l = loweredJson(s, {}, {});
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
  const auto iorOf = [](const LoweredJson& l) { return l.material["extensions"]["KHR_materials_ior"]["ior"].get<float>(); };

  // (a) DAZ's default, 4.37% linear: a plain dielectric, no metal
  const auto s = specGloss(1, grey(0.2313726));
  EXPECT_EQ(s.mixing, Surface::Mixing::eSpecGloss);
  EXPECT_NEAR(s.glossy.reflectivity, 0.54669f, 1e-4f);
  EXPECT_EQ(s.glossy.color, (fv::mat::Rgb{1, 1, 1}));
  EXPECT_FLOAT_EQ(s.metallic.value, 0);
  const auto a = loweredJson(s, {}, {});
  EXPECT_FLOAT_EQ(a.material["pbrMetallicRoughness"]["metallicFactor"].get<float>(), 0);
  EXPECT_FLOAT_EQ(specularOf(a)["specularFactor"].get<float>(), 1);
  EXPECT_NEAR(iorOf(a), 1.52886f, 1e-4f);
  expectRgb(specularOf(a)["specularColorFactor"], 1, 1, 1, 0);
  EXPECT_NEAR(a.material["pbrMetallicRoughness"]["roughnessFactor"].get<float>(), 0.3f, 1e-6f);
  // (b) the weight scales the whole curve: the index stays the colour's
  const auto b = loweredJson(specGloss(0.2, grey(0.2313726)), {}, {});
  EXPECT_NEAR(specularOf(b)["specularFactor"].get<float>(), 0.2f, 1e-5f);
  EXPECT_NEAR(iorOf(b), 1.52886f, 1e-4f);
  // (c) a bright F0
  const auto c = loweredJson(specGloss(1, grey(0.8078431)), {}, {});
  EXPECT_NEAR(iorOf(c), 8.3294f, 1e-2f);
  EXPECT_FLOAT_EQ(specularOf(c)["specularFactor"].get<float>(), 1);
  // (c2) Noki Veil: white at a fifth of the weight is an even fifth, not a bright rim
  const auto c2 = loweredJson(specGloss(0.2, grey(1)), {}, {});
  EXPECT_NEAR(specularOf(c2)["specularFactor"].get<float>(), 0.210526f, 1e-5f);
  EXPECT_NEAR(iorOf(c2), 77.987f, 1e-2f);
  // (d) a coloured F0: the brightest channel is the index, the hue the specular colour
  const auto d = loweredJson(specGloss(1, Json::array({0.5, 0.25, 0.5})), {}, {});
  expectRgb(specularOf(d)["specularColorFactor"], 1, 0.23769f, 1, 1e-4f);
  EXPECT_NEAR(iorOf(d), 2.72194f, 1e-3f);
  // (e) Glossy Reflectivity is not read in this mode
  const auto e = loweredJson(specGloss(1, grey(0.2313726), {{"Glossy Reflectivity", 1}}), {}, {});
  EXPECT_EQ(e.material, a.material);
  // (f) Glossy Specular's map is the colour's
  const auto f = loweredJson(specGloss(1, grey(0.2313726), {}, {{"Glossy Specular", "/bls_spec.jpg"}}), {}, {});
  ASSERT_TRUE(specularOf(f).contains("specularColorTexture"));
  EXPECT_EQ(requestOf(f, specularOf(f)["specularColorTexture"]).rel, "/bls_spec.jpg");
  EXPECT_TRUE(requestOf(f, specularOf(f)["specularColorTexture"]).srgb);
  // (h) under 4%: a plain index, the lobe scaled down
  const auto h = loweredJson(specGloss(1, grey(0.1254902)), {}, {});
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
    EXPECT_FALSE(loweredJson(off, {}, {.skin = fv::mat::SkinMode::eRaster}).material.contains("normalTexture"));
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

float emissiveStrengthOf(const LoweredJson& l)
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
  const auto l = loweredJson(s, {}, fv::mat::LowerOptions{.emission = 2});
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
    return loweredJson(s, {}, fv::mat::LowerOptions{.emission = 2});
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
  const auto plain = loweredJson(glow, {}, {});
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

const fv::mat::TextureRequest& specularRequest(const LoweredJson& l)
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

  const auto l = loweredJson(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.2710526f, 1e-5f);
  ASSERT_TRUE(specularOf(l).contains("specularTexture"));
  const auto& spec = specularRequest(l);
  EXPECT_EQ(spec.rel, kSlw);
  EXPECT_EQ(spec.channels, "wa");
  EXPECT_FALSE(spec.srgb);
  EXPECT_TRUE(spec.composed());
  EXPECT_NEAR(secondLobeOf(l)["specular2Factor"].get<float>(), 0.2289474f, 1e-5f);
  ASSERT_TRUE(secondLobeOf(l).contains("specular2Texture"));
  EXPECT_EQ(fv::mat::requestKey(requestOf(l, secondLobeOf(l)["specular2Texture"])), fv::mat::requestKey(spec));
  EXPECT_EQ(requestOf(l, secondLobeOf(l)["specular2RoughnessTexture"]).rel, "/head_r.jpg");

  // no map (Polina 9's and Leyla 9's skin): flat as before
  const auto flat = loweredJson(cjHead(false), {}, {});
  EXPECT_NEAR(specularOf(flat)["specularFactor"].get<float>(), 0.2710526f, 1e-5f);
  EXPECT_FALSE(specularOf(flat).contains("specularTexture"));
  EXPECT_FALSE(secondLobeOf(flat).contains("specular2Texture"));
  EXPECT_EQ(requestOf(flat, secondLobeOf(flat)["specular2RoughnessTexture"]).rel, "/head_r.jpg");

  // CJ's nails: a top coat and both lobes, each in its own place, 0.425 and 0.075 under the same map
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
  const auto ln = loweredJson(nails, {}, {});
  EXPECT_NEAR(specularOf(ln)["specularFactor"].get<float>(), 0.425f, 1e-5f);
  ASSERT_TRUE(specularOf(ln).contains("specularTexture"));
  EXPECT_EQ(specularRequest(ln).rel, kSlw);
  EXPECT_NEAR(secondLobeOf(ln)["specular2Factor"].get<float>(), 0.075f, 1e-5f);
  EXPECT_EQ(requestOf(ln, secondLobeOf(ln)["specular2Texture"]).rel, kSlw);
  EXPECT_NEAR(clearcoatOf(ln)["clearcoatFactor"].get<float>(), 0.25f, 1e-5f);
  EXPECT_FALSE(clearcoatOf(ln).contains("clearcoatTexture"));  // the coat's weight has no map
}

// Iray Uber stacks its glossy layer over the dual lobe, two lobes with weights of their own. glTF
// sums them into one specularFactor, glossy x Mg + dual x Md, which no product of maps gives: the
// sum stays flat, and only the tight lobe (the second lobe, the dual's alone) keeps the map.
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
  const auto l = loweredJson(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.75f, 1e-5f);
  EXPECT_FALSE(specularOf(l).contains("specularTexture"));
  ASSERT_TRUE(secondLobeOf(l).contains("specular2Texture"));
  EXPECT_EQ(requestOf(l, secondLobeOf(l)["specular2Texture"]).rel, "/Runtime/Textures/face_s.jpg");
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
  // with a top coat too: each lobe its own, under the same map
  const auto l = loweredJson(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.9326389f, 1e-5f);
  ASSERT_TRUE(specularOf(l).contains("specularTexture"));
  EXPECT_EQ(specularRequest(l).rel, "/head_s.jpg");
  EXPECT_EQ(specularRequest(l).channels, "wa");
  EXPECT_NEAR(secondLobeOf(l)["specular2Factor"].get<float>(), 0.1645833f, 1e-5f);
  EXPECT_EQ(requestOf(l, secondLobeOf(l)["specular2Texture"]).rel, "/head_s.jpg");
  EXPECT_NEAR(clearcoatOf(l)["clearcoatFactor"].get<float>(), 0.01f, 1e-6f);
  EXPECT_FALSE(clearcoatOf(l).contains("clearcoatTexture"));

  // the real-time view and a file have only the coat, so there the tight lobe folds back as it did
  for(const fv::mat::LowerOptions options : {fv::mat::LowerOptions{.skin = fv::mat::SkinMode::eRaster}, fv::mat::LowerOptions{.forFile = true}})
  {
    const auto folded = loweredJson(s, {}, options);
    EXPECT_FALSE(folded.material["extensions"].contains("FV_materials_specular2"));
    EXPECT_NEAR(specularOf(folded)["specularFactor"].get<float>(), 1.0f, 1e-5f);
    EXPECT_NEAR(clearcoatOf(folded)["clearcoatFactor"].get<float>(), 0.01f, 1e-6f);
  }
}

// The Material panel's clearcoat sliders move the coat where a surface has one, else its second lobe, the
// tightest highlight it has, as they did when the lobe sat in the coat's slot
TEST(Materials, TheClearcoatSlidersMoveTheSecondLobeWhereThereIsNoCoat)
{
  const auto s = cjHead();
  fv::mat::Edits e;
  e.clearcoat          = 0.6f;
  e.clearcoatRoughness = 0.3f;
  const auto l = loweredJson(s, e, {});
  EXPECT_NEAR(secondLobeOf(l)["specular2Factor"].get<float>(), 0.6f, 1e-6f);
  EXPECT_NEAR(secondLobeOf(l)["specular2RoughnessFactor"].get<float>(), 0.3f, 1e-6f);
  EXPECT_FALSE(l.material["extensions"].contains("KHR_materials_clearcoat"));
  EXPECT_NEAR(l.effective.clearcoat, 0.6f, 1e-6f);
  EXPECT_NEAR(l.effective.clearcoatRoughness, 0.3f, 1e-6f);

  // with a coat they move the coat, and the lobe keeps its own
  auto coated = cjHead();
  coated.topCoat = fv::mat::Surface::TopCoat{.weight = 0.5f, .roughness = 0.2f};
  const auto both = loweredJson(coated, e, {});
  EXPECT_NEAR(clearcoatOf(both)["clearcoatFactor"].get<float>(), 0.6f, 1e-6f);
  EXPECT_NEAR(secondLobeOf(both)["specular2Factor"].get<float>(), 0.2289474f, 1e-5f);
}

// A lobe with a weight map and a reflectivity map is their product, which the cooker multiplies
// (mul=): Madelein's face maps both of her dual lobe's.
TEST(Materials, TwoMapsOnOneLobeAreMultipliedByTheCooker)
{
  auto s = cjHead();
  s.dual->reflectivityMap = "/Runtime/Textures/head_s.jpg";
  const auto l = loweredJson(s, {}, {});
  EXPECT_NEAR(specularOf(l)["specularFactor"].get<float>(), 0.2710526f, 1e-5f);
  EXPECT_NEAR(secondLobeOf(l)["specular2Factor"].get<float>(), 0.2289474f, 1e-5f);
  const auto& spec = specularRequest(l);
  EXPECT_EQ(spec.rel, kSlw);
  EXPECT_EQ(spec.mulRels, std::vector<std::string>{"/Runtime/Textures/head_s.jpg"});
  EXPECT_TRUE(spec.composed());
  const std::string key = fv::mat::requestKey(spec);
  EXPECT_NE(key.find("|mul:/Runtime/Textures/head_s.jpg"), std::string::npos);
  EXPECT_EQ(fv::mat::requestKey(requestOf(l, secondLobeOf(l)["specular2Texture"])), key);
  EXPECT_NE(key, fv::mat::requestKey(specularRequest(loweredJson(cjHead(), {}, {}))));

  // a layered image multiplied in gives its bottom layer: the cooker multiplies plain images
  auto layeredMap = cjHead();
  layeredMap.dual->reflectivityMap = std::string("/x.png") + fv::mat::kLayerSeparator + "/y.png";
  EXPECT_EQ(specularRequest(loweredJson(layeredMap, {}, {})).mulRels, std::vector<std::string>{"/x.png"});
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
  const auto l = loweredJson(s, {}, {});
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
  const auto transmissionOf = [](const LoweredJson& l) { return l.material["extensions"]["KHR_materials_transmission"]; };
  for(const auto mode : {fv::mat::SkinMode::eVolume, fv::mat::SkinMode::eThin})
  {
    const auto l = loweredJson(face, {}, {.skin = mode});
    EXPECT_NEAR(transmissionOf(l)["transmissionFactor"].get<float>(), 0.8f, 1e-6f);
    ASSERT_TRUE(transmissionOf(l).contains("transmissionTexture"));
    const auto& r = requestOf(l, transmissionOf(l)["transmissionTexture"]);
    EXPECT_EQ(r.rel, "/trans_w.jpg");
    EXPECT_EQ(r.channels, "");
    EXPECT_FALSE(r.srgb);
  }
  const auto measured = loweredJson(face, {}, {.skin = fv::mat::SkinMode::eMeasured});
  EXPECT_NEAR(transmissionOf(measured)["transmissionFactor"].get<float>(), 1.0f, 1e-6f);
  EXPECT_FALSE(transmissionOf(measured).contains("transmissionTexture"));
  EXPECT_FALSE(loweredJson(face, {}, {.skin = fv::mat::SkinMode::eRaster}).material["extensions"].contains("KHR_materials_transmission"));

  const auto nails = loweredJson(at("Fingernails"), {}, {.skin = fv::mat::SkinMode::eMeasured});
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
  const auto l = loweredJson(s, {}, {});
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
  for(const auto& r : loweredJson(s, {}, {}).requests)
    EXPECT_NE(r.rel, "/mult.jpg");
}

// What a material becomes in the path tracer's closure: the shader's evaluateMaterial compiled as C++
// (figure/tools/closure_probe), against a baseline a change to the lowering or the closure moves.
