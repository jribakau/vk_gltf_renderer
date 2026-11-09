// What a character is built as, and what's laid on it: chosen eyes, iris and sclera colours, makeup, tattoos.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>

#include "materials_test_util.hpp"

using namespace materials_test;

namespace {

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

// ---- skin's micro-relief

constexpr float kFaceMetresPerUv = 0.43f;
fv::mat::LowerOptions underAWalk()
{
  return {.skin = fv::mat::SkinMode::eMeasured, .metresPerUv = kFaceMetresPerUv};
}

TEST(Materials, SkinUnderAWalkHasTheMicroReliefAtItsSize)
{
  const auto  skin    = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto  lowered = loweredJson(skin, {}, underAWalk());
  const auto& de      = lowered.material["extensions"]["FV_materials_detail"];
  EXPECT_FLOAT_EQ(de["detailWeightFactor"].get<float>(), 1);
  EXPECT_FLOAT_EQ(de["detailRoughnessFactor"].get<float>(), fv::mat::kSkinMicroRoughness);
  const auto& normal = de["detailNormalTexture"];
  EXPECT_FLOAT_EQ(normal["scale"].get<float>(), 1);
  const float tiles = kFaceMetresPerUv / fv::mat::kSkinMicroTileMetres;
  EXPECT_FLOAT_EQ(normal["extensions"]["KHR_texture_transform"]["scale"][0].get<float>(), tiles);
  EXPECT_FLOAT_EQ(normal["extensions"]["KHR_texture_transform"]["scale"][1].get<float>(), tiles);
  // a height map baked to its slopes, whose relief doesn't change with the surface's size
  const auto request = std::find_if(lowered.requests.begin(), lowered.requests.end(), [](const auto& r) { return r.rel == fv::mat::kSkinMicroMap; });
  ASSERT_NE(request, lowered.requests.end());
  EXPECT_EQ(request->channels, "bn");
  EXPECT_FLOAT_EQ(request->bumpHeight, fv::mat::kSkinMicroRelief);
  EXPECT_FALSE(request->srgb);
  EXPECT_FLOAT_EQ(*lowered.effective.microDetail, 1);
  // the shader reads how much its mips vary, for the roughness the relief they average away gives
  EXPECT_TRUE(request->normalLength);
  // the volume skin's walk melts it the same way
  const auto volume = loweredJson(skin, {}, {.skin = fv::mat::SkinMode::eVolume, .metresPerUv = kFaceMetresPerUv});
  EXPECT_TRUE(volume.material["extensions"].contains("FV_materials_detail"));
  const auto volumeRequest = std::find_if(volume.requests.begin(), volume.requests.end(), [](const auto& r) { return r.rel == fv::mat::kSkinMicroMap; });
  ASSERT_NE(volumeRequest, volume.requests.end());
  EXPECT_TRUE(volumeRequest->normalLength);
}

TEST(Materials, TheMicroReliefGoesOnlyOnSkinItCanLieOn)
{
  const auto has = [](const fv::mat::Surface& s, const fv::mat::Edits& e, const fv::mat::LowerOptions& o) {
    const auto lowered = fv::mat::lower(s, e, o);
    return std::any_of(lowered.requests.begin(), lowered.requests.end(), [](const auto& r) { return r.rel == fv::mat::kSkinMicroMap; });
  };
  const auto skin = fv::dson::parseSurface("Face", channelsOf(kSkin));
  ASSERT_TRUE(has(skin, {}, underAWalk()));
  // the views with no walk to melt it out of the diffuse light, a file, and a surface of unknown size
  EXPECT_FALSE(has(skin, {}, {.skin = fv::mat::SkinMode::eRaster, .metresPerUv = kFaceMetresPerUv}));
  EXPECT_FALSE(has(skin, {}, {.skin = fv::mat::SkinMode::eThin, .metresPerUv = kFaceMetresPerUv}));
  EXPECT_FALSE(has(skin, {}, {.skin = fv::mat::SkinMode::eMeasured, .forFile = true, .metresPerUv = kFaceMetresPerUv}));
  EXPECT_FALSE(has(skin, {}, {.skin = fv::mat::SkinMode::eMeasured}));
  // nothing but skin: a mouth's wet lining, nails, an eye
  for(const char* name : {"Mouth Cavity", "Fingernails", "Eye Left"})
    EXPECT_FALSE(has(fv::dson::parseSurface(name, channelsOf(kSkin)), {}, underAWalk())) << name;
  // the panel takes it off
  fv::mat::Edits off;
  off.microDetail = 0;
  EXPECT_FALSE(has(skin, off, underAWalk()));
  EXPECT_FLOAT_EQ(*fv::mat::lower(skin, off, underAWalk()).effective.microDetail, 0);
  // a lip gloss lies smooth over it
  auto coated    = skin;
  coated.topCoat = fv::mat::Surface::TopCoat{.weight = 0.5f};
  EXPECT_FALSE(has(coated, {}, underAWalk()));
  // a fabric on the surface wins
  fv::mat::Edits fabric;
  fabric.fabric = "denim";
  EXPECT_FALSE(has(skin, fabric, underAWalk()));
  // a detail map of the file's own stays, and with it what would be lost
  auto authored       = skin;
  authored.detail     = fv::mat::DetailLayer{.map = "/Runtime/Textures/pores.png", .weight = 1};
  EXPECT_FALSE(has(authored, {}, underAWalk()));
  auto weighted       = skin;
  weighted.detail     = fv::mat::DetailLayer{.weight = 1, .weightMap = "/Runtime/Textures/w.png", .roughness = 0.5f};
  EXPECT_FALSE(has(weighted, {}, underAWalk()));
}

// Leyla 9's skin has a detail layer with no map: on, doing nothing, or a multiplier the relief keeps
TEST(Materials, TheMicroReliefKeepsAMaplessLayersRoughness)
{
  auto skin   = fv::dson::parseSurface("Face", channelsOf(kSkin));
  skin.detail = fv::mat::DetailLayer{.weight = 0.5f, .roughness = 0.8f};
  const auto  lowered = loweredJson(skin, {}, underAWalk());
  const auto& de      = lowered.material["extensions"]["FV_materials_detail"];
  EXPECT_FLOAT_EQ(de["detailWeightFactor"].get<float>(), 1);
  EXPECT_FLOAT_EQ(de["detailRoughnessFactor"].get<float>(), 0.9f * fv::mat::kSkinMicroRoughness);
}

// Slopes' variance goes as the strength squared, and what the lobes give back with it, up to strength
// 1's: past that a strong relief is what's asked for, and it widens the highlight
TEST(Materials, TheMicroReliefsStrengthScalesItsSlopesAndItsRoughness)
{
  const auto skin = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto at   = [&](float strength) {
    fv::mat::Edits e;
    e.microDetail = strength;
    const auto lowered = loweredJson(skin, e, underAWalk());
    const auto de      = lowered.material["extensions"]["FV_materials_detail"];
    EXPECT_FLOAT_EQ(*lowered.effective.microDetail, strength);
    return std::pair{de["detailNormalTexture"]["scale"].get<float>(), de["detailRoughnessFactor"].get<float>()};
  };
  const float give = 1 - fv::mat::kSkinMicroRoughness;
  EXPECT_FLOAT_EQ(at(0.5f).first, 0.5f);
  EXPECT_FLOAT_EQ(at(0.5f).second, 1 - give * 0.25f);
  EXPECT_FLOAT_EQ(at(2).first, 2);
  EXPECT_FLOAT_EQ(at(2).second, fv::mat::kSkinMicroRoughness);
}

TEST(Materials, AnEyesWhiteSpreadsItsLightUnderMeasuredSkinWithNoWalk)
{
  const auto  eye     = fv::dson::parseSurface("Eye Left", channelsOf(kEye));
  const auto  lowered = loweredJson(eye, {}, {.skin = fv::mat::SkinMode::eMeasured});
  const auto& ext     = lowered.material["extensions"];
  ASSERT_TRUE(ext.contains("FV_materials_subsurface"));
  for(int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(ext["FV_materials_subsurface"]["scatterDistance"][i].get<float>(), fv::mat::kScleraScatterDistance[i]);
  // no medium: the eyeball is no closed volume
  EXPECT_FALSE(ext.contains("KHR_materials_volume"));
  EXPECT_FALSE(ext.contains("KHR_materials_transmission"));
  // the Iray volume leaves the eye as it was
  EXPECT_FALSE(loweredJson(eye, {}, {.skin = fv::mat::SkinMode::eVolume}).material["extensions"].contains("FV_materials_subsurface"));
}

TEST(Materials, AnIrisColourAsksForARecolouredBaseMap)
{
  const auto     eye = fv::dson::parseSurface("Eye Left", channelsOf(kEye));
  fv::mat::Edits e;
  e.irisColor        = fv::mat::Rgb{0.2f, 0.4f, 0.6f};
  e.irisStrength     = 0.5f;
  const auto lowered = loweredJson(eye, e, {});
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
  const auto keyOf = [&](const fv::mat::Edits& edits) { return fv::mat::requestKey(loweredJson(eye, edits, {}).requests[0]); };
  fv::mat::Edits stronger = e;
  stronger.irisStrength   = 1.f;
  fv::mat::Edits other    = e;
  other.irisColor         = fv::mat::Rgb{0.6f, 0.4f, 0.2f};
  EXPECT_NE(keyOf(e), keyOf(stronger));
  EXPECT_NE(keyOf(e), keyOf(other));
  EXPECT_FALSE(loweredJson(eye, {}, {}).requests[0].iris);
  // a strength alone changes nothing
  fv::mat::Edits strengthOnly;
  strengthOnly.irisStrength = 0.3f;
  EXPECT_FALSE(loweredJson(eye, strengthOnly, {}).requests[0].iris);

  // the white of a Genesis 9 eye is in the same map: baked outside the irises
  fv::mat::Edits white;
  white.scleraColor = fv::mat::Rgb{1.f, 0.9f, 0.8f};
  const auto sclera = loweredJson(eye, white, {}).requests[0];
  ASSERT_TRUE(sclera.iris);
  EXPECT_EQ(fv::mat::irisQuery(*sclera.iris), "-|1|eye|0.249,0.249,0.108;0.749,0.249,0.108|1,0.9,0.8");
}

TEST(Materials, AGenesis8IrisNeedsItsDiscsAndItsScleraIsAFactor)
{
  const auto     irises = fv::dson::parseSurface("Irises", channelsOf(kEye));
  fv::mat::Edits e;
  e.irisColor = fv::mat::Rgb{0.1f, 0.3f, 0.1f};
  // where the irises are in the map isn't known: no bake (material.set refuses the colour)
  EXPECT_FALSE(loweredJson(irises, e, {}).requests[0].iris);
  fv::mat::LowerOptions o;
  o.irisDiscs        = {{0.25f, 0.75f, 0.1f}, {0.75f, 0.75f, 0.1f}};
  const auto lowered = loweredJson(irises, e, o);
  ASSERT_TRUE(lowered.requests[0].iris);
  EXPECT_EQ(fv::mat::irisQuery(*lowered.requests[0].iris), "0.1,0.3,0.1|1|iris|0.25,0.75,0.1;0.75,0.75,0.1|-");

  // a sclera of its own multiplies its factor, live
  const auto     sclera = fv::dson::parseSurface("Sclera", channelsOf(kEye));
  fv::mat::Edits white;
  white.color       = fv::mat::Rgb{0.5f, 1.f, 1.f};
  white.scleraColor = fv::mat::Rgb{1.f, 0.8f, 0.8f};
  const auto plain  = loweredJson(sclera, white, {});
  EXPECT_FALSE(plain.requests[0].iris);
  EXPECT_NEAR(plain.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(), 0.5f, 1e-6f);
  EXPECT_NEAR(plain.material["pbrMetallicRoughness"]["baseColorFactor"][1].get<float>(), 0.8f, 1e-6f);

  // a flat iris takes the colour as its factor
  const auto flat = fv::dson::parseSurface("Irises", channelsOf(R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[0.5,0.5,0.5]}}]}]})"));
  const auto f    = loweredJson(flat, e, {});
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
  const auto lowered = loweredJson(hair, e, {});
  ASSERT_FALSE(lowered.requests.empty());
  const fv::mat::TextureRequest& base = lowered.requests[0];
  ASSERT_TRUE(base.recolor);
  EXPECT_TRUE(base.composed());
  EXPECT_EQ(base.rel, "/Runtime/Textures/face_d.jpg");
  // the cooker's rc= (texcache.rs Recolour::parse), before the host percent-encodes it whole
  EXPECT_EQ(fv::mat::recolorQuery(*base.recolor), "0.2,0.4,0.6|0.5");
  EXPECT_NE(fv::mat::requestKey(base).find("|rc:0.2,0.4,0.6|0.5"), std::string::npos);
  EXPECT_FALSE(loweredJson(hair, {}, {}).requests[0].recolor);

  // no map: the flat colour, half way to the target
  const auto flat = fv::dson::parseSurface("Hair", channelsOf(kEyeMoisture));
  fv::mat::Edits half;
  half.recolor         = fv::mat::Rgb{1.f, 0.f, 0.f};
  half.recolorStrength = 0.5f;
  const Json factor    = loweredJson(flat, half, {}).material["pbrMetallicRoughness"]["baseColorFactor"];
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

// A colour is in a bake only where it has to be, and the panel is told, so it can apply it on release:
// a drag would otherwise cook a texture a frame. Over a cutout alone it's a factor on a white image;
// laid under a makeup layer or a hair's translucency it isn't.
TEST(Materials, AColourStaysOutOfABakeWhereItCan)
{
  const auto lowered = [](const fv::mat::Surface& s, fv::mat::Rgb color, std::optional<fv::mat::Rgb> recolor = {}) {
    fv::mat::Edits e;
    e.color   = color;
    e.recolor = recolor;
    return loweredJson(s, e, {});
  };
  const fv::mat::Rgb red{0.8f, 0.1f, 0.1f}, blue{0.1f, 0.1f, 0.8f};
  const auto         keyOf = [](const LoweredJson& l) { return fv::mat::requestKey(l.requests.at(0)); };
  const auto         factorOf = [](const LoweredJson& l) { return l.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(); };

  // a cutout with no base map
  fv::mat::Surface cutout;
  cutout.name        = "Lace";
  cutout.opacity.map = "/Runtime/Textures/lace_o.jpg";
  EXPECT_FALSE(lowered(cutout, red).effective.colorBaked);
  EXPECT_EQ(keyOf(lowered(cutout, red)), keyOf(lowered(cutout, blue)));
  EXPECT_FLOAT_EQ(factorOf(lowered(cutout, red)), 0.8f);
  EXPECT_FLOAT_EQ(lowered(cutout, red).requests.at(0).makeup->factor[0], 1.f);
  EXPECT_EQ(lowered(cutout, red).requests.at(0).alphaRel, cutout.opacity.map);

  // recoloured, it's the image that's moved
  EXPECT_TRUE(lowered(cutout, red, blue).effective.colorBaked);
  EXPECT_NE(keyOf(lowered(cutout, red, blue)), keyOf(lowered(cutout, blue, blue)));
  EXPECT_FLOAT_EQ(factorOf(lowered(cutout, red, blue)), 1.f);

  // a plain map, with or without a cutout: a factor
  fv::mat::Surface mapped = cutout;
  mapped.base.map         = "/Runtime/Textures/lace_d.jpg";
  EXPECT_FALSE(lowered(mapped, red).effective.colorBaked);
  EXPECT_EQ(keyOf(lowered(mapped, red)), keyOf(lowered(mapped, blue)));
  EXPECT_FLOAT_EQ(factorOf(lowered(mapped, red)), 0.8f);

  // under makeup: in the image
  fv::mat::Surface madeUp;
  madeUp.name     = "Lips";
  madeUp.base.map = "/Runtime/Textures/lips_d.jpg";
  madeUp.makeup   = fv::mat::MakeupLayer{.map = "/Runtime/Textures/lips.png", .weight = 0.8f};
  EXPECT_TRUE(lowered(madeUp, red).effective.colorBaked);
  EXPECT_NE(keyOf(lowered(madeUp, red)), keyOf(lowered(madeUp, blue)));

  // a hair's translucency is such a layer
  fv::mat::Surface hair;
  hair.name         = "Hair";
  hair.kind         = fv::mat::Kind::eHair;
  hair.base.map     = "/Runtime/Textures/hair_d.jpg";
  hair.translucency = fv::mat::Translucency{.weight = 0.5f, .map = "/Runtime/Textures/hair_t.jpg"};
  EXPECT_TRUE(lowered(hair, red).effective.colorBaked);
  EXPECT_NE(keyOf(lowered(hair, red)), keyOf(lowered(hair, blue)));

  // no map at all: a factor
  EXPECT_FALSE(lowered(fv::mat::Surface{}, red).effective.colorBaked);
}

// Drawn from both sides by what the surface is on, as Iray draws it, but for a figure; and a light
// that says it's one-sided has a back, or it would emit twice the flux it's given. An edit says otherwise.
TEST(Materials, WhatIsDrawnFromBothSidesGoesByWhatItIsOn)
{
  using fv::mat::Context;
  const auto drawnBothWays = [](Context context, bool oneSidedLight = false, std::optional<bool> edit = {}) {
    fv::mat::Surface s;
    s.context = context;
    if(oneSidedLight)
    {
      fv::mat::Emission e;
      e.color     = {fv::mat::Rgb{1, 1, 1}, ""};
      e.luminance = fv::mat::Luminance{.amount = 100, .perArea = true, .twoSided = false};
      s.emission  = e;
    }
    fv::mat::Edits edits;
    edits.doubleSided = edit;
    return loweredJson(s, edits, {}).material["doubleSided"].get<bool>();
  };
  EXPECT_TRUE(drawnBothWays(Context::eItem));
  EXPECT_TRUE(drawnBothWays(Context::eHair));
  EXPECT_FALSE(drawnBothWays(Context::eBody));
  EXPECT_FALSE(drawnBothWays(Context::eItem, true));
  EXPECT_FALSE(drawnBothWays(Context::eHair, true));
  EXPECT_TRUE(drawnBothWays(Context::eItem, true, true));
  EXPECT_FALSE(drawnBothWays(Context::eItem, false, false));
  EXPECT_TRUE(drawnBothWays(Context::eBody, false, true));
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

namespace {

// A layered preset on the Head's base colour, of these layers
Json layeredOnHead(const Json& layers)
{
  return Json{{"asset_info", {{"type", "preset_layered_image"}}},
              {"image_library", Json::array({{{"id", "T"}, {"map_size", {4096, 4096}}, {"map", layers}}})},
              {"scene", {{"animations", Json::array({{{"url", "name://@selection/Genesis9:#materials/Head:?diffuse/image"}, {"keys", Json::array({Json::array({0, "#T"})})}}})}}}};
}

// A tattoo as they mostly come: an empty bottom, so it goes over the map there, and its image
Json tattooOn(const std::string& file, double opacity)
{
  return layeredOnHead(Json::array({{{"color", {0, 0, 0}}}, {{"url", file}, {"transparency", opacity}}}));
}

}  // namespace

// Tattoos are a stack over the skin: any of them goes, or comes back, without the skin or the others moving
TEST(Materials, TattoosComeOffOneAtATimeLeavingTheSkinAndTheRest)
{
  fv::SceneGraph scene = eyeScene();
  fv::dson::applyMaterialPreset(Json::parse(R"({"scene":{"animations":[{"url":"name://@selection/Genesis9:#materials/Head:?extra/studio_material_channels/channels/diffuse/image_file","keys":[[0,"/tan.jpg"]]}]}})"), scene);
  const std::set<int> surfaces = fv::dson::skinSurfaces(scene, {7});
  const auto          base     = [&] { return scene.materials[1].surface.base.map; };
  ASSERT_EQ(base(), "/tan.jpg");
  const auto has = [&](const char* file) { return base().find(file) != std::string::npos; };

  const Json arm = tattooOn("/arm.png", 0.75), leg = tattooOn("/leg.png", 0.75);
  auto       both = fv::dson::chooseLayers(scene, surfaces, {{&arm}, {&leg}}, {});
  EXPECT_EQ(both.changed, std::set<int>{1});
  EXPECT_TRUE(has("/tan.jpg"));
  EXPECT_TRUE(has("/arm.png"));
  EXPECT_TRUE(has("/leg.png"));
  EXPECT_LT(base().find("/arm.png"), base().find("/leg.png"));  // bottom up

  // one goes, the other and the skin stay
  auto legOnly = fv::dson::chooseLayers(scene, surfaces, {{&leg}}, both.layer);
  EXPECT_TRUE(has("/tan.jpg"));
  EXPECT_FALSE(has("/arm.png"));
  EXPECT_TRUE(has("/leg.png"));

  // back on, on top now
  auto again = fv::dson::chooseLayers(scene, surfaces, {{&leg}, {&arm}}, legOnly.layer);
  EXPECT_LT(base().find("/leg.png"), base().find("/arm.png"));

  // all off: the skin as it was, not a trace
  const auto off = fv::dson::chooseLayers(scene, surfaces, {}, again.layer);
  EXPECT_EQ(base(), "/tan.jpg");
  EXPECT_TRUE(off.layer.empty());
  EXPECT_EQ(off.changed, std::set<int>{1});
}

TEST(Materials, FadingATattooLowersEveryLayerAndKeepsTheSkinUnderAWholeImage)
{
  const Json two   = layeredOnHead(Json::array({{{"url", "/logo.png"}, {"transparency", 0.5}}, {{"url", "/b.png"}}}));
  const Json faded = fv::dson::fadedLayers(two, 0.5f);
  const Json& map  = faded["image_library"][0]["map"];
  ASSERT_EQ(map.size(), 2u);
  EXPECT_DOUBLE_EQ(map[0]["transparency"].get<double>(), 0.25);
  EXPECT_DOUBLE_EQ(map[1]["transparency"].get<double>(), 0.5);  // no opacity of its own: all of it
  EXPECT_DOUBLE_EQ(two["image_library"][0]["map"][0]["transparency"].get<double>(), 0.5);  // its own copy
  EXPECT_EQ(fv::dson::fadedLayers(Json::object(), 0.5f), Json::object());

  fv::SceneGraph scene = eyeScene();
  fv::dson::applyMaterialPreset(Json::parse(R"({"scene":{"animations":[{"url":"name://@selection/Genesis9:#materials/Head:?extra/studio_material_channels/channels/diffuse/image_file","keys":[[0,"/tan.jpg"]]}]}})"), scene);
  const std::set<int> surfaces = fv::dson::skinSurfaces(scene, {7});
  const auto          base     = [&] { return scene.materials[1].surface.base.map; };

  // A whole image replaces the map. Faded it has to show what is under it, and has to stay itself:
  // read alone, a bottom under full opacity is a placeholder and left out.
  const Json whole = layeredOnHead(Json::array({{{"url", "/whole.png"}}}));
  auto       laid  = fv::dson::chooseLayers(scene, surfaces, {{&whole}}, {});
  EXPECT_EQ(base(), "/whole.png");
  laid = fv::dson::chooseLayers(scene, surfaces, {{&whole, 0.5f}}, laid.layer);
  ASSERT_EQ(fv::mat::mapLayers(base()).size(), 2u) << base();
  EXPECT_EQ(fv::mat::mapLayers(base())[0], "/tan.jpg");
  const auto layer = fv::mat::parseLayer(fv::mat::mapLayers(base())[1]);
  EXPECT_EQ(layer.file, "/whole.png");
  EXPECT_FLOAT_EQ(layer.opacity, 0.5f);

  // A maker's logo at the bottom stays hidden through a fade
  const Json logo = layeredOnHead(Json::array({{{"url", "/logo.png"}, {"transparency", 0}}, {{"url", "/ink.png"}, {"transparency", 0.8}}}));
  laid = fv::dson::chooseLayers(scene, surfaces, {{&logo, 0.5f}}, laid.layer);
  EXPECT_EQ(base().find("/logo.png"), std::string::npos) << base();
  EXPECT_FLOAT_EQ(fv::mat::parseLayer(fv::mat::mapLayers(base())[1]).opacity, 0.4f);

  // All off, the skin
  fv::dson::chooseLayers(scene, surfaces, {}, laid.layer);
  EXPECT_EQ(base(), "/tan.jpg");
}

// A lipstick's colour replaces its colour map over the weight map, and its weight is the panel's; a
// skin tone recolours the base map against the tone reference, the scattering colour as well.
TEST(Materials, AToneAndAMakeupColourLowerIntoTheBaseMap)
{
  const auto lips = fv::dson::parseSurface("Lips", channelsOf(kMadeUp));
  ASSERT_TRUE(lips.makeup);
  fv::mat::Edits e;
  e.makeupColor = fv::mat::Rgb{0.6f, 0.05f, 0.1f};
  e.makeupWeight = 0.5f;
  const auto m  = loweredJson(lips, e, {});
  ASSERT_TRUE(m.requests[0].makeup);
  EXPECT_EQ(m.requests[0].makeup->colorRel, "");
  EXPECT_EQ(m.requests[0].makeup->weightRel, "/mask.png");
  EXPECT_FLOAT_EQ(m.requests[0].makeup->weightFactor, 0.5f);
  EXPECT_FLOAT_EQ(m.requests[0].makeup->tint[0], 0.6f);
  EXPECT_EQ(loweredJson(lips, {}, {}).requests[0].makeup->colorRel, "/lips.png");

  fv::mat::Edits tone;
  tone.tone         = fv::mat::Rgb{0.3f, 0.2f, 0.15f};
  tone.toneStrength = 0.75f;
  fv::mat::LowerOptions o;
  o.toneReference = "/body_d.jpg";
  o.skin          = fv::mat::SkinMode::eThin;
  const auto face = loweredJson(fv::dson::parseSurface("Face", channelsOf(kSkin)), tone, o);
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
  const auto own = loweredJson(fv::dson::parseSurface("Face", channelsOf(kSkin)), tone, {});
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
  const auto l = loweredJson(lips, {}, {});
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
  const auto metalRequest = [](const LoweredJson& l) -> const fv::mat::TextureRequest* {
    const auto& pbr = l.material["pbrMetallicRoughness"];
    return pbr.contains("metallicRoughnessTexture") ? &l.requests[pbr["metallicRoughnessTexture"]["slot"].get<size_t>()] : nullptr;
  };

  // the mask alone: the image is the metal map's grey in blue
  const auto alone = loweredJson(belt(1, false), {}, {});
  EXPECT_FLOAT_EQ(alone.material["pbrMetallicRoughness"]["metallicFactor"].get<float>(), 1);
  const fv::mat::TextureRequest* mt = metalRequest(alone);
  ASSERT_TRUE(mt);
  EXPECT_EQ(mt->rel, "/belt_m.jpg");
  EXPECT_EQ(mt->channels, "mt");
  EXPECT_FALSE(mt->srgb);
  EXPECT_TRUE(mt->metalRel.empty());

  // with a roughness map, the mask rides in its blue, and the two are different images to cook
  const auto both = loweredJson(belt(1, true), {}, {});
  const fv::mat::TextureRequest* mr = metalRequest(both);
  ASSERT_TRUE(mr);
  EXPECT_EQ(mr->rel, "/belt_r.jpg");
  EXPECT_EQ(mr->channels, "mr");
  EXPECT_EQ(mr->metalRel, "/belt_m.jpg");
  fv::mat::TextureRequest unmasked = *mr;
  unmasked.metalRel.clear();
  EXPECT_NE(fv::mat::requestKey(*mr), fv::mat::requestKey(unmasked));

  // no metal: the mask is left out
  const auto none = loweredJson(belt(0, false), {}, {});
  EXPECT_FALSE(metalRequest(none));
  const auto roughOnly = loweredJson(belt(0, true), {}, {});
  ASSERT_TRUE(metalRequest(roughOnly));
  EXPECT_TRUE(metalRequest(roughOnly)->metalRel.empty());

  // a panel metalness on a surface authored without metal shows the authored mask
  fv::mat::Edits e;
  e.metalness       = 0.5f;
  const auto edited = loweredJson(belt(0, false), e, {});
  EXPECT_FLOAT_EQ(edited.material["pbrMetallicRoughness"]["metallicFactor"].get<float>(), 0.5f);
  ASSERT_TRUE(metalRequest(edited));
  EXPECT_EQ(metalRequest(edited)->channels, "mt");
}

// ---- a shader preset chosen for some surfaces

const char* kGarment = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
  {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[1,1,1],"image_file":"/Runtime/Textures/shirt_d.jpg"}},
  {"channel":{"id":"Translucency Weight","type":"float","current_value":0.5}},
  {"channel":{"id":"Glossy Roughness","type":"float","current_value":0.7}}
]}]})";

const char* kThreeDelightGarment = R"({
  "diffuse":{"channel":{"id":"diffuse","type":"color","current_value":[0.5,0.5,0.5],"image_file":"/Runtime/Textures/old_d.jpg"}},
  "extra":[{"type":"studio/material/daz_shader","definition":"support/DAZ/shaders/dzDefault/dzDefault.dse"}]})";

// A shader preset as DAZ writes one: its scene material names the shader, its library entry holds the
// shader's channels at their defaults, and every url is for the selection. `filler` more values.
Json shaderPreset(const std::vector<std::pair<std::string, Json>>& values, int filler = 0, const char* type = "studio/material/uber_iray")
{
  Json animations = Json::array();
  const auto add  = [&](const std::string& channel, const Json& value) {
    animations.push_back({{"url", "name://@selection#materials:?extra/studio_material_channels/channels/" + channel}, {"keys", Json::array({Json::array({0, value})})}});
  };
  for(const auto& [channel, value] : values)
    add(channel, value);
  for(int i = 0; i < filler; i++)
    add("Filler%20" + std::to_string(i) + "/value", 0.5);
  return {{"asset_info", {{"type", "preset_shader"}}},
          {"material_library", Json::array({{{"id", "Default"},
                                              {"extra", Json::array({{{"type", type}}, {{"type", "studio_material_channels"}, {"channels", Json::array({{{"channel", {{"id", "Metallic Weight"}, {"type", "float"}, {"value", 0}}}}})}}})}}})},
          {"scene", {{"materials", Json::array({{{"url", "#Default"}, {"extra", Json::array({{{"type", type}}})}}})}, {"animations", animations}}}};
}

Json goldPreset()
{
  return shaderPreset({{"diffuse/value", Json::array({1, 0.9, 0.6})}, {"diffuse/image", nullptr}, {"Metallic%20Weight/value", 1}, {"Glossy%20Roughness/value", 0.2}}, fv::dson::kFullIrayPreset);
}

fv::SceneGraph garmentScene()
{
  fv::SceneGraph scene;
  scene.materials.push_back(fv::dson::buildMaterial("Shirt", channelsOf(kGarment)));
  scene.materials.push_back(fv::dson::buildMaterial("Collar", channelsOf(kGarment)));
  scene.materials.push_back(fv::dson::buildMaterial("Old", channelsOf(kThreeDelightGarment)));
  return scene;
}

TEST(ShaderPresets, WhatOneCanDo)
{
  using Kind = fv::dson::ShaderPresetKind;
  EXPECT_EQ(fv::dson::shaderPresetKind(goldPreset()), Kind::eWhole);
  EXPECT_EQ(fv::dson::shaderPresetKind(shaderPreset({{"Horizontal%20Tiles/value", 2}})), Kind::eOption);
  // "!Iray Uber Base" and DS Defaults' only switch the shader
  EXPECT_EQ(fv::dson::shaderPresetKind(shaderPreset({})), Kind::eNoValues);
  // NVIDIA's MDL examples and FilaToon are bricks, 3Delight's a shader of its own: their values mean nothing here
  EXPECT_EQ(fv::dson::shaderPresetKind(shaderPreset({{"Diffuse%20Color/value", "/resources/concrete.png"}}, 0, "studio/material/daz_brick")), Kind::eOtherShader);
  EXPECT_EQ(fv::dson::shaderPresetKind(shaderPreset({{"Ka/value", 0.5}}, 0, "studio/material/daz_shader")), Kind::eOtherShader);
  // made for a surface by name, it is a material preset
  EXPECT_EQ(fv::dson::shaderPresetKind(boxersPreset("Black", 50)), Kind::eNoValues);
}

TEST(ShaderPresets, AWholeOneReplacesTheSurfacesItGoesOn)
{
  fv::SceneGraph scene = garmentScene();
  // whatever their names and shaders
  EXPECT_EQ(fv::dson::applyShaderPreset(goldPreset(), scene, {0, 2, 99}, "Gold.duf"), std::set<int>({0, 2}));
  for(int i : {0, 2})
  {
    const fv::Material& m = scene.materials[size_t(i)];
    EXPECT_EQ(m.surface.shader, fv::mat::Surface::Shader::eIray) << m.name;
    EXPECT_TRUE(m.surface.base.map.empty()) << m.name;
    EXPECT_FLOAT_EQ(m.surface.metallic.value, 1.f) << m.name;
    EXPECT_FLOAT_EQ(m.surface.glossy.roughness.value, 0.2f) << m.name;
    ASSERT_TRUE(m.shaderChoice) << m.name;
    EXPECT_EQ(m.shaderChoice->rel, "Gold.duf");
  }
  // nothing of the old material stays
  EXPECT_FALSE(scene.materials[0].channels->get(fv::dson::norm("Translucency Weight")));
  // the one not picked keeps its own
  EXPECT_EQ(scene.materials[1].surface.base.map, "/Runtime/Textures/shirt_d.jpg");
  EXPECT_FALSE(scene.materials[1].shaderChoice);
  // the shader's defaults are under the values the preset sets
  EXPECT_TRUE(scene.materials[0].channels->get(fv::dson::norm("Metallic Weight")));
}

TEST(ShaderPresets, OnOverAnotherItComesOffToTheSurfaceBeforeBoth)
{
  fv::SceneGraph scene = garmentScene();
  const auto     own   = scene.materials[0].channels;
  fv::dson::applyShaderPreset(goldPreset(), scene, {0}, "Gold.duf");
  fv::dson::applyShaderPreset(shaderPreset({{"Metallic%20Weight/value", 0}}, fv::dson::kFullIrayPreset), scene, {0}, "Silk.duf");
  EXPECT_EQ(scene.materials[0].shaderChoice->rel, "Silk.duf");
  EXPECT_EQ(scene.materials[0].shaderChoice->under, own);
  // only those that have one count
  EXPECT_EQ(fv::dson::removeShaderPreset(scene, {0, 1, 99}), std::set<int>({0}));
  EXPECT_EQ(scene.materials[0].channels, own);
  EXPECT_EQ(scene.materials[0].surface.base.map, "/Runtime/Textures/shirt_d.jpg");
  EXPECT_FALSE(scene.materials[0].shaderChoice);
}

TEST(ShaderPresets, AnOptionGoesOverTheSurfaceAndTakesThePlaceOfOneThatSetsNoMore)
{
  fv::SceneGraph scene = garmentScene();
  const auto     own   = scene.materials[0].channels;
  const auto     tiles = [](int n) { return shaderPreset({{"Horizontal%20Tiles/value", n}, {"Vertical%20Tiles/value", n}}); };
  const Json     gloss = shaderPreset({{"Glossy%20Roughness/value", 0.1}});

  // alone, on the surface's own: its maps stay
  EXPECT_EQ(fv::dson::applyShaderPreset(gloss, scene, {0}, "Gloss.duf"), std::set<int>({0}));
  EXPECT_EQ(scene.materials[0].surface.base.map, "/Runtime/Textures/shirt_d.jpg");
  EXPECT_FLOAT_EQ(scene.materials[0].surface.glossy.roughness.value, 0.1f);
  ASSERT_TRUE(scene.materials[0].shaderChoice);
  EXPECT_TRUE(scene.materials[0].shaderChoice->rel.empty());

  const auto options = [&] {
    std::vector<std::string> out;
    for(const fv::Material::ShaderOption& o : scene.materials[0].shaderChoice->options)
      out.push_back(o.rel);
    return out;
  };
  fv::dson::applyShaderPreset(tiles(2), scene, {0}, "Tile 2.duf");
  EXPECT_EQ(options(), (std::vector<std::string>{"Gloss.duf", "Tile 2.duf"}));
  // another tiling in place of the first, the gloss stays
  fv::dson::applyShaderPreset(tiles(3), scene, {0}, "Tile 3.duf");
  EXPECT_EQ(options(), (std::vector<std::string>{"Gloss.duf", "Tile 3.duf"}));
  // a whole one takes them all off and goes on what they went on
  fv::dson::applyShaderPreset(goldPreset(), scene, {0}, "Gold.duf");
  EXPECT_TRUE(options().empty());
  EXPECT_EQ(scene.materials[0].shaderChoice->under, own);
  fv::dson::applyShaderPreset(tiles(2), scene, {0}, "Tile 2.duf");
  EXPECT_EQ(options(), (std::vector<std::string>{"Tile 2.duf"}));
  EXPECT_EQ(scene.materials[0].shaderChoice->rel, "Gold.duf");
  // what it went on over comes back from under all of it
  fv::dson::removeShaderPreset(scene, {0});
  EXPECT_EQ(scene.materials[0].channels, own);
}

TEST(ShaderPresets, WhatCantBeDrawnChangesNothing)
{
  fv::SceneGraph scene = garmentScene();
  const auto     own   = scene.materials[0].channels;
  EXPECT_TRUE(fv::dson::applyShaderPreset(shaderPreset({}), scene, {0}, "Base.duf").empty());
  EXPECT_TRUE(fv::dson::applyShaderPreset(shaderPreset({{"Diffuse%20Color/value", "/concrete.png"}}, 50, "studio/material/daz_brick"), scene, {0}, "Concrete.duf").empty());
  EXPECT_EQ(scene.materials[0].channels, own);
  EXPECT_FALSE(scene.materials[0].shaderChoice);
}

// What later rewrites a surface leaves no shader said to be on it
TEST(ShaderPresets, AColourOrAMaterialPresetOverOneReplacesIt)
{
  fv::SceneGraph scene = garmentScene();
  fv::dson::applyShaderPreset(goldPreset(), scene, {0}, "Gold.duf");
  EXPECT_EQ(fv::dson::restoreBaseline(scene, {0}), std::set<int>({0}));
  EXPECT_FALSE(scene.materials[0].shaderChoice);

  fv::dson::applyShaderPreset(goldPreset(), scene, {0, 1}, "Gold.duf");
  fv::dson::applyMaterialPreset(Json::parse(R"({"scene":{"animations":[
    {"url":"name://@selection#materials/Shirt:?diffuse/value","keys":[[0,[1,0,0]]]}
  ]}})"),
                                scene);
  EXPECT_FALSE(scene.materials[0].shaderChoice);
  EXPECT_TRUE(scene.materials[1].shaderChoice);
}

// Every shader preset of a library lowers to what its name says, and the table of them goes to the output. With DAZ_LIBRARY set.
TEST(ShaderPresets, TheLibrarysLowerToWhatTheirNamesSay)
{
  const char* lib = std::getenv("DAZ_LIBRARY");
  if(!lib || !*lib)
    GTEST_SKIP() << "DAZ_LIBRARY not set";
  namespace fs = std::filesystem;
  const auto library = std::make_shared<fv::dson::Library>(std::vector<fs::path>{fs::path(lib)});
  fv::dson::Store store(library);
  const fs::path  root = fs::path(lib) / "Shader Presets";
  ASSERT_TRUE(fs::exists(root));
  std::map<std::string, int> kinds;
  for(const auto& entry : fs::recursive_directory_iterator(root))
  {
    if(!entry.is_regular_file() || entry.path().extension() != ".duf")
      continue;
    const std::string rel = fs::relative(entry.path(), fs::path(lib)).generic_string();
    const auto        doc = store.load(rel);
    ASSERT_TRUE(doc) << rel;
    const fv::dson::ShaderPresetKind kind = fv::dson::shaderPresetKind(doc->json);
    kinds[kind == fv::dson::ShaderPresetKind::eWhole ? "whole" : kind == fv::dson::ShaderPresetKind::eOption ? "option" : kind == fv::dson::ShaderPresetKind::eOtherShader ? "other shader" : "no values"]++;
    if(kind != fv::dson::ShaderPresetKind::eWhole)
      continue;
    fv::SceneGraph scene = garmentScene();
    ASSERT_EQ(fv::dson::applyShaderPreset(doc->json, scene, {0}, rel, fv::dson::definitionsIn(doc->json, store)), std::set<int>({0})) << rel;
    const fv::Material& m = scene.materials[0];
    const LoweredJson   l = loweredJson(m.surface, {}, {});
    const Json&         pbr = l.material["pbrMetallicRoughness"];
    const float         metal = pbr.value("metallicFactor", 1.f), rough = pbr.value("roughnessFactor", 1.f);
    const float         transmit = l.material.contains("extensions") && l.material["extensions"].contains("KHR_materials_transmission") ? l.material["extensions"]["KHR_materials_transmission"].value("transmissionFactor", 0.f) : 0.f;
    const std::string   name = entry.path().stem().string();
    std::printf("%-58s %-8s metal %.2f rough %.2f transmit %.2f maps %zu\n", name.c_str(), m.surface.shader == fv::mat::Surface::Shader::eIray ? "iray" : "other", metal, rough, transmit, l.requests.size());
    EXPECT_EQ(m.surface.shader, fv::mat::Surface::Shader::eIray) << rel;
    // DAZ wrote these two in Weighted mode: a strong glossy lobe, no metallic
    if(name.rfind("Metal - ", 0) == 0 && name != "Metal - Magnesium Alloy" && name != "Metal - Steel - Galvanized")
      EXPECT_GT(metal, 0.9f) << rel;
    if(name.rfind("Glass - ", 0) == 0 || name.rfind("Water", 0) == 0 || name == "Diamond")
      EXPECT_GT(transmit, 0.5f) << rel;
    if(name.rfind("Fabric - ", 0) == 0 || name.rfind("Plastic - ", 0) == 0 || name.rfind("Rubber - ", 0) == 0 || name == "Leather")
      EXPECT_LT(metal, 0.1f) << rel;
  }
  for(const auto& [what, n] : kinds)
    std::printf("%s: %d\n", what.c_str(), n);
}

}  // namespace
