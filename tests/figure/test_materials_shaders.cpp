// The older shaders (AoA, DAZ Default, Uber Surface) and presets that switch a surface's shader.

#include "materials_test_util.hpp"

using namespace materials_test;

namespace {

const char* kAoaSceneLayer = R"({"extra":[{"channels":[
  {"channel":{"id":"Ambient Color","current_value":[0.9,0.7,0.6]}},
  {"channel":{"id":"color4","current_value":[0,0,0]}},
  {"channel":{"id":"Ambient Strength2","current_value":0}}
]}]})";

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

const char* kMatte     = R"({"extra":[{"channels":[{"channel":{"id":"Lighting Model","current_value":4}}]}]})";
const char* kIrayOver  = R"({"extra":[{"channels":[{"channel":{"id":"Glossy Roughness","current_value":0.4}}]}]})";

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
  const auto reqs  = loweredJson(boxers, {}, o).requests;
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

// DAZ's Displacement Strength map, which Iray moves vertices by, shaded as a height under the bump:
// read with its range, then laid in the normal slot by what else the surface has.
TEST(Materials, ADisplacementMapIsShadedAsAHeightUnderTheBump)
{
  const auto uber = [](const std::string& more) {
    return std::string(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
      {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1]}})")
           + more + "]}]}";
  };
  const auto parse = [](const std::string& text) { return fv::dson::parseSurface("Suit", channelsOf(text.c_str())); };
  const std::string disp = R"(,{"channel":{"id":"Displacement Strength","type":"float","current_value":1,"image_file":"/Runtime/Textures/suit_dp.jpg"}})";

  // DAZ's -0.1 to 0.1 cm unless set: 2 mm at strength 1; the strength kept to PBRSkin's 0-2
  EXPECT_EQ(parse(uber(disp)).displacementMap, "/Runtime/Textures/suit_dp.jpg");
  EXPECT_NEAR(parse(uber(disp)).displacementRange, 0.002f, 1e-7f);
  const std::string range = R"(,{"channel":{"id":"Minimum Displacement","type":"float","current_value":-0.2}},
      {"channel":{"id":"Maximum Displacement","type":"float","current_value":0.3}})";
  std::string half = disp;
  half.replace(half.find("\"current_value\":1"), 17, "\"current_value\":0.5");
  EXPECT_NEAR(parse(uber(half + range)).displacementRange, 0.0025f, 1e-7f);
  std::string fifteen = disp;
  fifteen.replace(fifteen.find("\"current_value\":1"), 17, "\"current_value\":15");
  EXPECT_NEAR(parse(uber(fifteen)).displacementRange, 0.004f, 1e-7f);
  std::string none = disp;
  none.replace(none.find("\"current_value\":1"), 17, "\"current_value\":0");
  EXPECT_TRUE(parse(uber(none)).displacementMap.empty());
  // a 3Delight surface's: DSON's own displacement channel and its range
  const auto legacy = fv::dson::parseSurface("Scales", channelsOf(R"({"displacement":{"channel":{"id":"displacement","current_value":0.5,
      "image_file":"/Runtime/Textures/scales_dp.jpg"}},"displacement_min":{"channel":{"id":"displacement_min","current_value":-0.25}},
      "displacement_max":{"channel":{"id":"displacement_max","current_value":0.15}}})"));
  EXPECT_EQ(legacy.displacementMap, "/Runtime/Textures/scales_dp.jpg");
  EXPECT_NEAR(legacy.displacementRange, 0.002f, 1e-7f);

  fv::mat::LowerOptions o;
  o.metresPerUv      = 0.43f;
  const float height = fv::mat::bumpHeightInUv(0.43f, 1, 0.002f);
  const auto  normalOf = [&](const fv::mat::Surface& s, const fv::mat::LowerOptions& opts) {
    const auto l = loweredJson(s, {}, opts);
    const auto r = std::find_if(l.requests.begin(), l.requests.end(), [](const fv::mat::TextureRequest& q) { return q.channels == "bn" || q.channels == "ng"; });
    const float scale = l.material.contains("normalTexture") ? l.material["normalTexture"].value("scale", 1.0f) : 0.0f;
    return std::pair{r == l.requests.end() ? std::optional<fv::mat::TextureRequest>{} : std::optional{*r}, scale};
  };

  // neither a bump nor a normal map: the displacement in the bump's place, at its own height
  fv::mat::Surface s = parse(uber(disp));
  auto [alone, aloneScale] = normalOf(s, o);
  ASSERT_TRUE(alone);
  EXPECT_EQ(alone->rel, s.displacementMap);
  EXPECT_EQ(alone->channels, "bn");
  EXPECT_NEAR(alone->bumpHeight, height, 1e-7f);
  EXPECT_TRUE(alone->displaceRel.empty());
  EXPECT_FLOAT_EQ(aloneScale, 1.0f);
  // the map's size on the surface unknown: no slope to be had, nothing
  EXPECT_FALSE(normalOf(s, {}).first);

  // under a bump of its own (the Night City bodysuit's): both in one bake, the bump's strength in its
  // height and the slot at 1
  s.bumpMap      = "/Runtime/Textures/suit_b.jpg";
  s.bumpStrength = 2;
  auto [under, underScale] = normalOf(s, o);
  ASSERT_TRUE(under);
  EXPECT_EQ(under->rel, s.bumpMap);
  EXPECT_NEAR(under->bumpHeight, 2 * fv::mat::bumpHeightInUv(0.43f), 1e-7f);
  EXPECT_EQ(under->displaceRel, s.displacementMap);
  EXPECT_NEAR(under->displaceHeight, height, 1e-7f);
  EXPECT_FLOAT_EQ(underScale, 1.0f);
  fv::mat::TextureRequest plain = *under;
  plain.displaceRel.clear();
  EXPECT_NE(fv::mat::requestKey(*under), fv::mat::requestKey(plain));
  // the bump's strength at 0: the displacement alone
  s.bumpStrength = 0;
  EXPECT_EQ(normalOf(s, o).first->rel, s.displacementMap);
  EXPECT_TRUE(normalOf(s, o).first->displaceRel.empty());
  s.bumpStrength = 2;
  // tiled otherwise than the bump: one bake can't hold both, the bump alone as before
  s.mapTiling[s.displacementMap] = fv::mat::MapTiling{{2, 2}, {0, 0}};
  EXPECT_TRUE(normalOf(s, o).first->displaceRel.empty());
  EXPECT_FLOAT_EQ(normalOf(s, o).second, 2.0f);
  s.mapTiling.clear();
  // the bump's own image: nothing to add
  s.bumpMap = s.displacementMap;
  EXPECT_TRUE(normalOf(s, o).first->displaceRel.empty());
  // a normal map, which most often holds the same relief: left out
  s.bumpMap   = "/Runtime/Textures/suit_b.jpg";
  s.normalMap = "/Runtime/Textures/suit_n.jpg";
  EXPECT_TRUE(normalOf(s, o).first->displaceRel.empty());
  EXPECT_NE(normalOf(s, o).first->rel, s.displacementMap);
  s.bumpMap.clear();
  EXPECT_EQ(normalOf(s, o).first->rel, s.normalMap);
}

// Installed Iray presets for a 3Delight-era item: the one that keeps its look, or none.

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
  const auto l = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eRaster});
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
    return loweredJson(m.surface, m.edits, {}).material["pbrMetallicRoughness"]["roughnessFactor"].get<float>();
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
