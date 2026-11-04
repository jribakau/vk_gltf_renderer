// A DSON surface's channels become a Surface (Iray Uber, PBRSkin), and a Surface becomes the glTF
// material the renderer shows: factors, the textures it asks the cooker for, the skin's volume, and
// edits laid over it. A port of the material pipeline in src/lib/materials and src/lib/rtx/gltf.ts.

#include "materials_test_util.hpp"

using namespace materials_test;

namespace {

// What comes down last short of GPU memory: a figure's face, and nothing on a shirt or a chair
TEST(Materials, KnowsWhatIsOnTheFace)
{
  using fv::mat::Context;
  const auto on = [](const char* name, Context context = Context::eBody) {
    fv::mat::Surface s;
    s.name    = name;
    s.context = context;
    s.kind    = fv::mat::classify(name, context);
    return fv::mat::onTheFace(s);
  };
  for(const char* face : {"Face", "Head", "Lips", "Ears", "Eye Left", "Sclera", "Irises", "Cornea", "Eye Moisture Left", "Eyelashes", "Eyebrows", "Mouth", "Teeth", "Tongue"})
    EXPECT_TRUE(on(face)) << face;
  for(const char* rest : {"Body", "Torso", "Arms", "Legs", "Fingernails", "Toenails"})
    EXPECT_FALSE(on(rest)) << rest;
  // Genesis 9's lashes and brows are worn items, as its real surfaces are named
  for(const char* worn : {"Eyelashes Upper", "Eyelashes Lower", "Eyebrows_Primary"})
  {
    EXPECT_TRUE(on(worn, Context::eItem)) << worn;
    EXPECT_TRUE(on(worn, Context::eHair)) << worn;
  }
  EXPECT_FALSE(on("Brown Leather", Context::eItem));
  EXPECT_FALSE(on("Head", Context::eItem));
  EXPECT_FALSE(on("Face", Context::eItem));
  EXPECT_FALSE(on("Eye", Context::eHair));
  EXPECT_FALSE(on("Hair", Context::eHair));
}

TEST(Materials, ClassifiesSurfacesByName)
{
  using fv::mat::Kind;
  EXPECT_EQ(fv::mat::classify("Face"), Kind::eSkin);
  EXPECT_EQ(fv::mat::classify("Eye Moisture Left"), Kind::eCornea);
  EXPECT_EQ(fv::mat::classify("Eyelashes"), Kind::eLashes);
  EXPECT_EQ(fv::mat::classify("Fingernails"), Kind::eNails);
  EXPECT_EQ(fv::mat::classify("Scalp Hair"), Kind::eHair);
  EXPECT_EQ(fv::mat::classify("Dress"), Kind::eOther);
  // whole words: what a name only happens to contain
  for(const char* name : {"Pearl", "Armor", "Leggings", "Necklace", "Clip", "Handbag", "Garment", "Bodysuit", "Wear Grime", "Footbed", "Floral", "Coral"})
    EXPECT_EQ(fv::mat::classify(name), Kind::eOther) << name;
  for(const char* name : {"Brown Leather", "LeatherChair", "Chair_03"})
    EXPECT_EQ(fv::mat::classify(name), Kind::eOther) << name;
  for(const char* name : {"Flash", "KitchenBacksplash", "Snail", "Toothbrush"})
    EXPECT_EQ(fv::mat::classify(name), Kind::eOther) << name;
  // a figure's names, however they are written
  for(const char* name : {"Lips", "Ears", "Arms", "Legs", "Neck", "Forearms", "Toes", "Fingers", "Shirt_Back", "Back", "Genitalia", "1_SkinFace", "UpperLip"})
    EXPECT_EQ(fv::mat::classify(name), Kind::eSkin) << name;
  for(const char* name : {"EyeMoisture_L", "7_Tear", "Cornea"})
    EXPECT_EQ(fv::mat::classify(name), Kind::eCornea) << name;
  for(const char* name : {"Eyelashes", "EyeLash_mat", "6_Eyelash"})
    EXPECT_EQ(fv::mat::classify(name), Kind::eLashes) << name;
  for(const char* name : {"Eye Left", "Eyes_R", "Irises", "5_Pupil", "4_Eyes"})
    EXPECT_EQ(fv::mat::classify(name), Kind::eEye) << name;
  EXPECT_EQ(fv::mat::classify("Fingernails"), Kind::eNails);
  EXPECT_EQ(fv::mat::classify("3_Toenail"), Kind::eNails);
  EXPECT_EQ(fv::mat::classify("InnerMouth"), Kind::eMouth);
  EXPECT_EQ(fv::mat::classify("EyeSocket"), Kind::eOther);
  EXPECT_EQ(fv::mat::classify("Hairline_Back1"), Kind::eHair);
  EXPECT_EQ(fv::mat::classify("Eyebrows_Primary"), Kind::eHair);

  // what it's on: no flesh on a garment or a chair, and a hair cap only on hair
  using fv::mat::Context;
  for(const char* name : {"Body", "Back", "Right Leg", "Teeth", "Tongue", "Fingernails", "Face"})
  {
    EXPECT_EQ(fv::mat::classify(name, Context::eItem), Kind::eOther) << name;
    EXPECT_EQ(fv::mat::classify(name, Context::eHair), Kind::eOther) << name;
  }
  EXPECT_EQ(fv::mat::classify("Cap", Context::eItem), Kind::eOther);
  EXPECT_EQ(fv::mat::classify("Bottle_A_Cap", Context::eItem), Kind::eOther);
  EXPECT_EQ(fv::mat::classify("Cap", Context::eHair), Kind::eHair);
  EXPECT_EQ(fv::mat::classify("Skullcap", Context::eHair), Kind::eHair);
  EXPECT_EQ(fv::mat::classify("Cap", Context::eBody), Kind::eHair);
  // an outfit made with the skin it shows says so
  EXPECT_EQ(fv::mat::classify("Skin_Mat", Context::eItem), Kind::eSkin);
  // real eyes and hair on an item stay that: a mask's, a costume's
  EXPECT_EQ(fv::mat::classify("Hair 1", Context::eItem), Kind::eHair);
  EXPECT_EQ(fv::mat::classify("Sclera", Context::eItem), Kind::eEye);
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
    const auto lowered = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eVolume});
    const auto& ext    = lowered.material["extensions"];
    EXPECT_TRUE(ext.contains("KHR_materials_volume"));
    EXPECT_TRUE(ext.contains("KHR_materials_scatter"));
    EXPECT_NEAR(ext["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 0.6f, 1e-5f);
    // the medium warms what scatters; the base colour, which the walk leaves through, is the one
    // the rasterizer shows, not tinted again
    const auto  plain = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eRaster});
    const auto& base  = lowered.material["pbrMetallicRoughness"]["baseColorFactor"];
    for(int i = 0; i < 3; i++)
      EXPECT_FLOAT_EQ(base[i].get<float>(), plain.material["pbrMetallicRoughness"]["baseColorFactor"][i].get<float>());
  }
  {
    const auto lowered = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eRaster});
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
    const auto lowered = loweredJson(s, {}, {.skin = mode});
    EXPECT_FLOAT_EQ(lowered.material["extensions"]["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1.0f);
  }
}

// Measured skin ignores the file's Iray medium: every bit of diffuse light enters, the medium is
// white for the renderer to make from the skin map, and light spreads as far as in human skin.
TEST(Materials, MeasuredSkinEntersWhollyAndSpreadsAsHumanSkin)
{
  const auto  s       = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto  lowered = loweredJson(s, {}, {.skin = fv::mat::SkinMode::eMeasured});
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
  EXPECT_FALSE(loweredJson(s, {}, {.skin = fv::mat::SkinMode::eVolume}).material["extensions"].contains("FV_materials_subsurface"));
  // an edit scales the distance
  fv::mat::Edits e;
  e.scatterScale = 0.5f;
  EXPECT_FALSE(e.empty());
  const auto  halved = loweredJson(s, e, {.skin = fv::mat::SkinMode::eMeasured});
  const auto& hd     = halved.material["extensions"]["FV_materials_subsurface"]["scatterDistance"];
  for(int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(hd[i].get<float>(), 0.5f * fv::mat::kSkinScatterDistance[i]);
}

TEST(Materials, AsksTheCookerForTheBlendedNormalMap)
{
  const auto s       = fv::dson::parseSurface("Face", channelsOf(kSkin));
  const auto lowered = loweredJson(s, {}, {});
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
  const auto lowered = loweredJson(s, {}, {});
  const auto base    = std::find_if(lowered.requests.begin(), lowered.requests.end(), [](const auto& r) { return !r.layers.empty(); });
  ASSERT_NE(base, lowered.requests.end());
  EXPECT_EQ(base->rel, "/Runtime/Textures/Eyes/Sclera_01.jpg");
  EXPECT_EQ(base->layers, (std::vector<std::string>{"/Runtime/Textures/Eyes/Iris_01.png"}));
  EXPECT_TRUE(base->composed());
  fv::mat::TextureRequest plain = *base;
  plain.layers.clear();
  EXPECT_NE(fv::mat::requestKey(*base), fv::mat::requestKey(plain));
}

// A layer as it shows: what isn't seen is left out, and what's left says how it blends.
TEST(Materials, ALayerSaysHowMuchOfItShowsHowItBlendsAndWhatMasksIt)
{
  const Json doc = Json::parse(R"({"image_library": [
    {"id": "Look", "map_size": [4096, 4096], "map": [
      {"url": "/skin.jpg", "transparency": 1},
      {"url": "/hidden.png", "transparency": 0},
      {"url": "/blush.png", "transparency": 0.7, "operation": "blend_source_over"},
      {"url": "/writing.png", "operation": "multiply"},
      {"color": [0.5, 0.25, 1], "operation": "blend_overlay", "mask": {"url": "/lips.png", "active": true}},
      {"url": "/off.png", "active": false}]}]})");
  std::unordered_map<std::string, std::string> images;
  fv::dson::collectImages(doc, images);
  const auto layers = fv::mat::mapLayers(images["Look"]);
  ASSERT_EQ(layers.size(), 4u) << images["Look"];
  EXPECT_EQ(layers[0], "/skin.jpg");

  const auto blush = fv::mat::parseLayer(layers[1]);
  EXPECT_EQ(blush.file, "/blush.png");
  EXPECT_FLOAT_EQ(blush.opacity, 0.7f);
  EXPECT_EQ(blush.blend, "");

  const auto writing = fv::mat::parseLayer(layers[2]);
  EXPECT_EQ(writing.blend, "multiply");

  const auto lips = fv::mat::parseLayer(layers[3]);
  EXPECT_EQ(lips.file, "");
  EXPECT_EQ(lips.mask, "/lips.png");
  EXPECT_EQ(lips.blend, "overlay");
  EXPECT_FLOAT_EQ(lips.color[1], 0.25f);
  // a whole image's own stack: nothing to fit to another size
  EXPECT_EQ(lips.canvas, 0);

  // a plain layer stays a plain file, with the key it always had
  EXPECT_EQ(fv::mat::layerText(fv::mat::parseLayer("/iris.png")), "/iris.png");
  EXPECT_EQ(fv::mat::layerFile(layers[3]), "");
  EXPECT_EQ(fv::mat::layerFile(layers[1]), "/blush.png");
}

// What a makeup or a tattoo is made of on disk: 3D Sugar's stack has its logo at 0% at the bottom, Daz's
// Skull and Bones an empty black. As a map either would take the skin away.
TEST(Materials, ALayeredPresetWithNoBottomOfItsOwnGoesOverTheMapThere)
{
  const Json doc = Json::parse(R"({"image_library": [
    {"id": "Logo", "map_size": [4096, 4096], "map": [
      {"url": "/3DS.jpg", "transparency": 0},
      {"url": "/liner.png", "transparency": 0.9}]},
    {"id": "Faint", "map_size": [4096, 4096], "map": [
      {"url": "/3DS.jpg", "transparency": 0.04},
      {"url": "/blush.png"}]},
    {"id": "Empty", "map_size": [2048, 2048], "map": [
      {"color": [0, 0, 0]},
      {"url": "/skull.png", "operation": "blend_multiply", "transparency": 0.5}]},
    {"id": "Own", "map_size": [4096, 4096], "map": [
      {"url": "/river.jpg"},
      {"url": "/blush.png", "transparency": 0.7}]},
    {"id": "Nothing", "map": [{"url": "/3DS.jpg", "transparency": 0}]}]})");
  std::unordered_map<std::string, std::string> images;
  fv::dson::collectImages(doc, images, true);

  const auto over = [&](const char* id) {
    const auto layers = fv::mat::mapLayers(images[id]);
    EXPECT_EQ(layers.front(), "") << id << ": " << images[id];
    EXPECT_EQ(images[id].find("3DS.jpg"), std::string::npos) << "the logo shows: " << images[id];
    return layers;
  };
  const auto logo = fv::mat::parseLayer(over("Logo")[1]);
  EXPECT_EQ(logo.file, "/liner.png");
  EXPECT_FLOAT_EQ(logo.opacity, 0.9f);
  EXPECT_EQ(logo.canvas, 4096);
  EXPECT_EQ(fv::mat::parseLayer(over("Faint")[1]).file, "/blush.png");
  const auto skull = fv::mat::parseLayer(over("Empty")[1]);
  EXPECT_EQ(skull.blend, "multiply");
  EXPECT_EQ(skull.canvas, 2048);
  EXPECT_EQ(images["Nothing"], std::string(1, fv::mat::kLayerSeparator));

  // a whole image at the bottom is the map, as it is off a scene file
  EXPECT_EQ(fv::mat::mapLayers(images["Own"]).front(), "/river.jpg");

  // and what isn't a layered preset keeps the first file for its bottom: nothing goes over
  std::unordered_map<std::string, std::string> scene;
  fv::dson::collectImages(doc, scene);
  EXPECT_EQ(fv::mat::mapLayers(scene["Empty"]).front(), "/skull.png");
  EXPECT_EQ(fv::mat::mapLayers(scene["Logo"]).front(), "/liner.png");
}

TEST(Materials, LayersOverTheMapThereGoOnOnceAndLeaveAChannelWithoutOneToItsFile)
{
  fv::Material m;
  m.name         = "Head";
  auto channels  = channelsOf(kSkin);
  m.channels     = std::make_shared<const fv::dson::Channels>(channels);
  m.surface      = fv::dson::parseSurface(m.name, *m.channels);
  const auto map = [&](const char* id) {
    const fv::dson::Channel* c = m.channels->find({id});
    return c && c->image ? *c->image : std::string();
  };
  const std::string skin = map("diffuse");
  ASSERT_FALSE(skin.empty());

  const char        sep = fv::mat::kLayerSeparator;
  const std::string liner = std::string(1, sep) + "/liner.png";
  ASSERT_TRUE(fv::dson::overrideChannels(m, {{"diffuse", std::nullopt, liner, nullptr}}));
  EXPECT_EQ(map("diffuse"), skin + liner);
  EXPECT_EQ(fv::mat::mapLayers(m.surface.base.map).front(), skin);

  // a second goes over the first; the first again adds nothing
  const std::string blush = std::string(1, sep) + "/blush.png";
  fv::dson::overrideChannels(m, {{"diffuse", std::nullopt, blush, nullptr}});
  fv::dson::overrideChannels(m, {{"diffuse", std::nullopt, liner, nullptr}});
  EXPECT_EQ(map("diffuse"), skin + liner + blush);

  // nothing to lay: the map stays
  EXPECT_FALSE(fv::dson::overrideChannels(m, {{"diffuse", std::nullopt, std::string(1, sep), nullptr}}));
  EXPECT_EQ(map("diffuse"), skin + liner + blush);

  // a channel with no map takes the first layer's file as its bottom, not a path that starts with nothing
  fv::dson::overrideChannels(m, {{"No Map Yet", std::nullopt, liner + blush, nullptr}});
  EXPECT_EQ(map("No Map Yet"), "/liner.png" + blush);
}

TEST(Materials, EditsLieOverTheSurface)
{
  const auto     s = fv::dson::parseSurface("Face", channelsOf(kSkin));
  fv::mat::Edits e;
  e.roughness = 0.9f;
  e.color     = fv::mat::Rgb{0.2f, 0.2f, 0.2f};
  e.alphaMode = "cutout";
  const auto lowered = loweredJson(s, e, {.skin = fv::mat::SkinMode::eRaster});
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
  const auto lowered = loweredJson(s, {}, {});
  const auto& ext    = lowered.material["extensions"];
  EXPECT_NEAR(ext["KHR_materials_ior"]["ior"].get<float>(), 1.376f, 1e-5f);
  EXPECT_NEAR(ext["KHR_materials_transmission"]["transmissionFactor"].get<float>(), 1.0f, 1e-6f);
  EXPECT_FALSE(ext.contains("KHR_materials_scatter"));
  // refraction does not tint what passes through: the grey base colour goes to white
  EXPECT_NEAR(lowered.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(), 1.0f, 1e-6f);
  EXPECT_EQ(lowered.material["alphaMode"], "OPAQUE");
}

TEST(Materials, AnOpacityEditOnASurfaceThatLoadedOpaqueIsSeen)
{
  const auto s = fv::dson::parseSurface("Face", channelsOf(kSkin));
  EXPECT_EQ(loweredJson(s, {}, {}).material["alphaMode"], "OPAQUE");
  fv::mat::Edits e;
  e.opacity = 0.5f;
  const auto seen = loweredJson(s, e, {});
  EXPECT_EQ(seen.material["alphaMode"], "BLEND");
  EXPECT_EQ(seen.effective.alphaMode, "blend");
  EXPECT_TRUE(seen.effective.doubleSided);
  e.opacity = 1.f;
  EXPECT_EQ(loweredJson(s, e, {}).material["alphaMode"], "OPAQUE");
  // the alpha mode's own edit has the last word
  e.opacity   = 0.5f;
  e.alphaMode = "opaque";
  EXPECT_EQ(loweredJson(s, e, {}).material["alphaMode"], "OPAQUE");
}

TEST(Materials, ASheerGarmentIsAWeaveAndAHairCardIsNot)
{
  const char* sheer = R"({"extra":[{"type":"studio/material/uber_iray","channels":[
    {"channel":{"id":"diffuse","label":"Base Color","type":"color","current_value":[0.1,0.1,0.1]}},
    {"channel":{"id":"Cutout Opacity","type":"float","current_value":0.3,"image_file":"/Runtime/Textures/sheer_o.jpg"}}
  ]}]})";
  const auto woven = [&](const char* name, fv::mat::Context context, const fv::mat::LowerOptions& o = {}) {
    return fv::mat::lower(fv::dson::parseSurface(name, channelsOf(sheer), context), {}, o).material.woven;
  };
  EXPECT_TRUE(woven("Stockings", fv::mat::Context::eItem));
  EXPECT_TRUE(woven("Curtain", fv::mat::Context::eItem));
  // strands, not threads in a plane
  EXPECT_FALSE(woven("Hair", fv::mat::Context::eHair));
  EXPECT_FALSE(woven("Eyelashes", fv::mat::Context::eItem));
  // a figure's own surface
  EXPECT_FALSE(woven("Torso", fv::mat::Context::eBody));
  // a file's cutout goes out as a mask, whose test has no weave
  fv::mat::LowerOptions file;
  file.forFile = true;
  EXPECT_FALSE(woven("Stockings", fv::mat::Context::eItem, file));
  // an opacity edit back to opaque leaves nothing to see through
  fv::mat::Edits opaque;
  opaque.alphaMode = "opaque";
  EXPECT_FALSE(fv::mat::lower(fv::dson::parseSurface("Stockings", channelsOf(sheer), fv::mat::Context::eItem), opaque, {}).material.woven);
}

TEST(Materials, IrayFlakesAreAMetalLayerUnderTheCoat)
{
  // DAZ's "Car Paint - Red - Small Flakes" Uber preset
  const ChannelValues paint = {{"diffuse", Json::array({0.73, 0.73, 0.73})},
                               {"Glossy Roughness", 0.7071068},
                               {"Top Coat Weight", 0.8},
                               {"Top Coat Roughness", 0.1},
                               {"Metallic Flakes Weight", 1},
                               {"Metallic Flakes Color", Json::array({0.7927927, 0.3172521, 0.2783647})},
                               {"Metallic Flakes Roughness", 0.3162278},
                               {"Metallic Flakes Size", 0.01},
                               {"Metallic Flakes Strength", 1},
                               {"Metallic Flakes Density", 1}};
  const fv::mat::Surface s = surfaceOf("Paint", "studio/material/uber_iray", paint, {{"Metallic Flakes Weight", "/maps/glitter_mask.png"}});
  ASSERT_TRUE(s.flakes);
  EXPECT_FLOAT_EQ(s.flakes->size, 0.01f);
  EXPECT_EQ(s.flakes->weightMap, "/maps/glitter_mask.png");

  const fv::mat::Lowered l = fv::mat::lower(s, {}, {});
  ASSERT_TRUE(l.material.flakes);
  const fv::mat::Flakes& f = *l.material.flakes;
  // the file's colour is sRGB, the layer's linear
  const auto linear = [](double c) { return float(std::pow((c + 0.055) / 1.055, 2.4)); };
  EXPECT_FLOAT_EQ(f.weight, 1.0f);
  EXPECT_NEAR(f.color[0], linear(0.7927927), 1e-4f);
  EXPECT_NEAR(f.color[2], linear(0.2783647), 1e-4f);
  EXPECT_NEAR(f.roughness, 0.3162278f, 1e-6f);
  EXPECT_FLOAT_EQ(f.size, 0.01f);
  EXPECT_TRUE(f.weightMap.has());
  EXPECT_FALSE(f.colorMap.has());
  // the coat stays over them
  ASSERT_TRUE(l.material.clearcoat);
  EXPECT_NEAR(l.material.clearcoat->factor, 0.8f, 1e-6f);
  // a file for other tools has no flakes to read
  fv::mat::LowerOptions file;
  file.forFile = true;
  EXPECT_FALSE(fv::mat::lower(s, {}, file).material.flakes);
}

TEST(Materials, NoFlakesWithoutAWeight)
{
  // the Uber's defaults: a flake size and no weight
  EXPECT_FALSE(surfaceOf("Paint", "studio/material/uber_iray", {{"Metallic Flakes Weight", 0}, {"Metallic Flakes Size", 0.001}}).flakes);
  EXPECT_FALSE(surfaceOf("Paint", "studio/material/uber_iray", {{"Glossy Roughness", 0.5}}).flakes);
  // the Intensity colour effect takes the colour's luminance into the weight (sRGB 0.5 is linear 0.214)
  const fv::mat::Surface intensity = surfaceOf("Paint", "studio/material/uber_iray",
                                               {{"Metallic Flakes Weight", 1}, {"Metallic Flakes Color", Json::array({0.5, 0.5, 0.5})}, {"Metallic Flakes Color Effect", 2}});
  ASSERT_TRUE(intensity.flakes);
  EXPECT_NEAR(intensity.flakes->weight, std::pow(0.555 / 1.055, 2.4), 1e-4f);
  EXPECT_NEAR(intensity.flakes->color[1], 1.0f, 1e-5f);
}

TEST(Materials, OnlyTheMainSourceOfARequestIsLayered)
{
  // a cutout, a makeup, a metal or a weight map of two layers: the cooker opens those as one file, and
  // the name with its separator is none, which loses the whole request and the colour with it
  const auto layered = [](const char* name) { return std::string("/maps/") + name + "_a.png" + fv::mat::kLayerSeparator + "/maps/" + name + "_b.png"; };
  auto       s       = fv::dson::parseSurface("Face", channelsOf(kSkin));
  s.opacity.map      = layered("cut");
  s.bumpMap          = layered("bump");
  s.metallic.map     = layered("metal");
  s.makeup           = fv::mat::MakeupLayer{};
  s.makeup->map      = layered("makeup");
  s.makeup->weightMap = layered("weight");
  const auto lowered = loweredJson(s, {}, {});
  ASSERT_FALSE(lowered.requests.empty());
  const auto one = [](const std::string& rel) { EXPECT_EQ(rel.find(fv::mat::kLayerSeparator), std::string::npos) << rel; };
  bool       cut = false, painted = false;
  for(const fv::mat::TextureRequest& r : lowered.requests)
  {
    one(r.rel);
    one(r.alphaRel);
    one(r.metalRel);
    one(r.bumpRel);
    for(const std::string& rel : r.mulRels)
      one(rel);
    if(r.makeup)
    {
      one(r.makeup->colorRel);
      one(r.makeup->weightRel);
      painted = r.makeup->colorRel == "/maps/makeup_a.png" && r.makeup->weightRel == "/maps/weight_a.png";
    }
    cut = cut || r.alphaRel == "/maps/cut_a.png";
  }
  EXPECT_TRUE(cut);
  EXPECT_TRUE(painted);
}

// The tear film is the eye's one reflecting interface, at Fresnel's full strength and smooth: G9's
// glossy weight 0.40 left its catchlights 2.5 times too dim, and the iris's normal map on it rippled them
TEST(Materials, TheTearFilmReflectsAtFullFresnelFromASmoothSurface)
{
  const auto  s       = fv::dson::parseSurface("EyeMoisture Left", channelsOf(kG9Moisture));
  ASSERT_EQ(fv::mat::eyeShell(s), fv::mat::EyeShell::eMoisture);
  const auto  lowered = loweredJson(s, {}, {.underTearFilm = true});
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
  const auto under = loweredJson(s, {}, {.underTearFilm = true}).material["extensions"];
  EXPECT_NEAR(under["KHR_materials_specular"]["specularFactor"].get<float>(), 0.0f, 1e-6f);
  EXPECT_FALSE(under["KHR_materials_specular"].contains("specularTexture"));
  EXPECT_FALSE(under.contains("KHR_materials_clearcoat"));
  // with no film over it, its dual lobe shines where its weight map says
  const auto bareLowered = loweredJson(s, {}, {});
  const auto bare        = bareLowered.material["extensions"];
  EXPECT_NEAR(bare["KHR_materials_specular"]["specularFactor"].get<float>(), 0.85f, 1e-5f);
  ASSERT_TRUE(bare["KHR_materials_specular"].contains("specularTexture"));
  EXPECT_EQ(bareLowered.requests[bare["KHR_materials_specular"]["specularTexture"]["slot"].get<size_t>()].rel, "/Runtime/Textures/eyes_s.jpg");
  EXPECT_FALSE(under.contains("FV_materials_specular2"));
  EXPECT_TRUE(bare.contains("FV_materials_specular2"));
  EXPECT_TRUE(bare["FV_materials_specular2"].contains("specular2Texture"));
  // a panel edit still wins
  fv::mat::Edits e;
  e.specular = 0.3f;
  EXPECT_NEAR(loweredJson(s, e, {.underTearFilm = true}).material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(),
              0.3f, 1e-6f);
}

// Two shells stacked: the moisture is the film, the cornea under it is index-matched. A cornea with no
// moisture over it (Genesis 8's stops at the limbus, the build works that out) is the film itself. An
// index that is not a watery eye's (glass at 1.5) becomes the cornea's.
TEST(Materials, OneReflectingShellPerEye)
{
  const auto cornea = fv::dson::parseSurface("Cornea", channelsOf(kG9Moisture));
  ASSERT_EQ(fv::mat::eyeShell(cornea), fv::mat::EyeShell::eCornea);
  EXPECT_NEAR(loweredJson(cornea, {}, {.underTearFilm = true}).material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(),
              0.0f, 1e-6f);
  EXPECT_NEAR(loweredJson(cornea, {}, {}).material["extensions"]["KHR_materials_specular"]["specularFactor"].get<float>(), 1.0f, 1e-6f);

  auto glass = fv::dson::parseSurface("EyeMoisture", channelsOf(kG9Moisture));
  glass.refraction->ior = 1.5f;
  EXPECT_NEAR(loweredJson(glass, {}, {}).material["extensions"]["KHR_materials_ior"]["ior"].get<float>(), 1.376f, 1e-6f);
}

// A ray entering the eye bends at the one film over it, into the chamber; only there, or a stack would
// bend it twice. The meniscus has no chamber behind it, and a file for other tools none of ours.
TEST(Materials, TheTearFilmBendsARayIntoTheEyesChamber)
{
  const auto chamberOf = [](const char* name, const fv::mat::LowerOptions& o) {
    const auto  lowered = loweredJson(fv::dson::parseSurface(name, channelsOf(kG9Moisture)), {}, o);
    const auto& ext     = lowered.material["extensions"];
    return ext.contains("FV_materials_eye") ? ext["FV_materials_eye"]["chamberIor"].get<float>() : 0.f;
  };
  EXPECT_FLOAT_EQ(chamberOf("EyeMoisture Left", {.underTearFilm = true}), fv::mat::kEyeChamberIor);
  EXPECT_FLOAT_EQ(chamberOf("Cornea", {}), fv::mat::kEyeChamberIor);
  EXPECT_EQ(chamberOf("Cornea", {.underTearFilm = true}), 0.f);
  EXPECT_EQ(chamberOf("Tear", {.underTearFilm = true}), 0.f);
  EXPECT_EQ(chamberOf("EyeMoisture Left", {.forFile = true}), 0.f);
  // the film stays thin-walled: no medium for a path or a shadow ray to track
  const auto  film = loweredJson(fv::dson::parseSurface("EyeMoisture Left", channelsOf(kG9Moisture)), {}, {});
  EXPECT_LE(film.material["extensions"]["KHR_materials_volume"].value("thicknessFactor", 0.f), 0.f);
  // what bends the ray is not what reflects: the film's own index stays on the catchlight
  EXPECT_NEAR(film.material["extensions"]["KHR_materials_ior"]["ior"].get<float>(), 1.38f, 1e-6f);
}

// The lid margin's tear meniscus is the same water, but its normal map is its shape and stays
TEST(Materials, TheTearMeniscusKeepsItsShape)
{
  const auto s = fv::dson::parseSurface("Tear", channelsOf(kG9Moisture));
  ASSERT_EQ(fv::mat::eyeShell(s), fv::mat::EyeShell::eMeniscus);
  const auto lowered = loweredJson(s, {}, {.underTearFilm = true});
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

TEST(Materials, OmniHairMelaninReadsAsTheBrownItIs)
{
  // the albedo measured for medium brown hair, not the near-black a lit ribbon showed
  const std::string fullWeight = R"(,{"channel":{"id":"Diffuse Strength","type":"float","current_value":1}})";
  const auto c = fv::dson::omniHairColor(channelsOf(omniHair(fullWeight).c_str()));
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

// OmniHairBase.mdl: base_color x base_color_weight x 2.4, at most 1, to absorption, mixed with the
// melanin's by sqrt(Melanin). A light dye absorbs nothing of its own, which is what makes blonde
// and pastel presets light: without the 2.4 they came out dark.
TEST(Materials, OmniHairDyeIsColourTimesWeightTimesTheMelaninGamma)
{
  const auto channel = [](const char* id, double r, double g, double b) {
    return std::string(R"(,{"channel":{"id":")") + id + R"(","type":"color","current_value":[)" + std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b) + "]}}";
  };
  // a colour channel reads as sRGB
  const auto linear = [](float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); };
  const auto fibre  = [&](double r, double g, double b, double w) {
    return *fv::dson::parseSurface("FE Base", channelsOf(omniHair(channel("Hair Root Color", r, g, b) + channel("Hair Tip Color", r, g, b)
                                                                  + R"(,{"channel":{"id":"Diffuse Strength","type":"float","current_value":)" + std::to_string(w) + "}}")
                                                             .c_str()))
                .hairFibre;
  };
  const auto undyed = fibre(1, 1, 1, 1);
  // melanin 0.5, redness 0.5, as OmniHairBase.mdl's own spectra
  constexpr float melanin[3] = {0.5f * 0.5f * 2.4f * (0.5f * 0.506f + 0.5f * 0.343f), 0.5f * 0.5f * 2.4f * (0.5f * 0.841f + 0.5f * 0.733f),
                                0.5f * 0.5f * 2.4f * (0.5f * 1.653f + 0.5f * 1.924f)};
  const float       share      = std::sqrt(0.5f);
  for(int i = 0; i < 3; i++)
    EXPECT_NEAR(undyed.tipSigma[i], melanin[i], 1e-4f);
  EXPECT_FLOAT_EQ(undyed.melaninMix, 1.f);

  // a light dye clamps to no absorption of its own, so only the melanin's share is left: lighter than undyed
  const auto light = fibre(0.8, 0.8, 0.8, 1);
  for(int i = 0; i < 3; i++)
    EXPECT_NEAR(light.tipSigma[i], melanin[i] * share, 1e-4f);
  EXPECT_FLOAT_EQ(light.melaninMix, share);
  EXPECT_NEAR(light.tipDye[0], linear(0.8f) * 2.4f, 1e-4f);
  // white under full weight is a dye too, not none
  const auto white = fibre(1, 1, 1, 0.8);
  EXPECT_NEAR(white.tipSigma[0], melanin[0] * share, 1e-4f);

  // what the 2.4 leaves absorbing: a dark channel, and more of it the less weight it has
  const auto  red  = fibre(0.8, 0.1, 0.1, 1);
  const float fit  = 5.969f - 0.215f * 0.2f + 2.532f * 0.04f - 10.73f * 0.008f + 5.574f * 0.0016f + 0.245f * 0.00032f;
  const float dark = std::pow(std::log(linear(0.1f) * 2.4f) / fit, 2.f);
  EXPECT_NEAR(red.tipSigma[1], dark + (melanin[1] - dark) * share, 1e-4f);
  EXPECT_NEAR(red.tipSigma[0], melanin[0] * share, 1e-4f);
  EXPECT_GT(fibre(0.8, 0.1, 0.1, 0.5).tipSigma[1], red.tipSigma[1]);
  // no weight at all is black, not undyed
  EXPECT_GT(fibre(0.8, 0.1, 0.1, 0).tipSigma[0], undyed.tipSigma[0]);
}

// The channels the brick's root-to-tip blend reads, carried to the shader; a black root stays a colour
// so the blend happens before the absorption, as it does in the MDL.
TEST(Materials, OmniHairCarriesTheBlendToTheFibre)
{
  const std::string dye = R"(,{"channel":{"id":"Hair Root Color","type":"color","current_value":[0,0,0]}},
    {"channel":{"id":"Hair Tip Color","type":"color","current_value":[0.8,0.8,0.6]}},
    {"channel":{"id":"Diffuse Strength","type":"float","current_value":1}},
    {"channel":{"id":"Root To Tip Bias","type":"float","current_value":0.4}},
    {"channel":{"id":"Root to Tip Gain","type":"float","current_value":0.15}})";
  const auto s = fv::dson::parseSurface("FE Base", channelsOf(omniHair(dye).c_str()));
  ASSERT_TRUE(s.hairFibre);
  const auto  lowered = loweredJson(s, {}, {});
  const auto& hair    = lowered.material["extensions"]["FV_materials_hair"];
  EXPECT_NEAR(hair["blendBias"].get<float>(), 0.4f, 1e-5f);
  EXPECT_NEAR(hair["blendGain"].get<float>(), 0.15f, 1e-5f);
  EXPECT_EQ(hair["rootColor"][0].get<float>(), 0.f);
  EXPECT_NEAR(hair["tipColor"][2].get<float>(), 0.3185f * 2.4f, 1e-3f);  // 0.6 sRGB, linear
  EXPECT_NEAR(hair["melaninMix"].get<float>(), std::sqrt(0.5f), 1e-5f);
  EXPECT_NEAR(hair["melaninAbsorption"][0].get<float>(), s.hairFibre->melaninSigma[0], 1e-5f);

  // a tint leaves the blend and takes the melanin out: its ends are all there is
  fv::mat::Edits e;
  e.recolor               = fv::mat::Rgb{0.6f, 0.45f, 0.25f};
  const auto  tinted      = loweredJson(s, e, {});
  const auto& tintedHair  = tinted.material["extensions"]["FV_materials_hair"];
  EXPECT_EQ(tintedHair["melaninMix"].get<float>(), 0.f);
  EXPECT_NEAR(tintedHair["blendGain"].get<float>(), 0.15f, 1e-5f);
  EXPECT_LT(tintedHair["rootColor"][0].get<float>(), tintedHair["tipColor"][0].get<float>());
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

  const auto  lowered = loweredJson(s, {}, {});
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
  const auto  filed = loweredJson(s, {}, fileOpts);
  const auto& fileExt = filed.material["extensions"];
  EXPECT_FALSE(fileExt.contains("FV_materials_hair"));
  EXPECT_NEAR(fileExt["KHR_materials_ior"]["ior"].get<float>(), 1.55f, 1e-6f);
  EXPECT_NEAR(fileExt["KHR_materials_specular"]["specularFactor"].get<float>(), 1.0f, 1e-6f);
  const auto& dt = fileExt["KHR_materials_diffuse_transmission"];
  EXPECT_NEAR(dt["diffuseTransmissionFactor"].get<float>(), 0.4f, 1e-6f);
  EXPECT_NEAR(dt["diffuseTransmissionColorFactor"][0].get<float>(), filed.material["pbrMetallicRoughness"]["baseColorFactor"][0].get<float>(), 1e-6f);

  // card hair is not a ribbon
  const auto cards = loweredJson(fv::dson::parseSurface("Scalp Hair", channelsOf(kSkin)), {}, {});
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
  const auto alone = loweredJson(bumped, {}, o).requests;
  ASSERT_EQ(alone.size(), 1u);
  EXPECT_EQ(alone[0].channels, "bn");
  EXPECT_NEAR(alone[0].bumpHeight, fv::mat::bumpHeightInUv(0.43f), 1e-7f);
  EXPECT_NE(fv::mat::requestKey(alone[0]), fv::mat::requestKey(loweredJson(bumped, {}, {}).requests[0]));
  bumped.normalMap   = "/Runtime/Textures/skin_n.jpg";
  const auto blended = loweredJson(bumped, {}, o).requests;
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
  const auto  lowered = loweredJson(s, {}, o);
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
  EXPECT_FALSE(loweredJson(s, {}, {}).material["extensions"]["FV_materials_hair"].contains("strandRotation"));

  // the real-time view and a file for other tools keep the GGX sheet
  fv::mat::LowerOptions raster;
  raster.skin = fv::mat::SkinMode::eRaster;
  EXPECT_FALSE(loweredJson(s, {}, raster).material["extensions"].contains("FV_materials_hair"));
  fv::mat::LowerOptions file;
  file.forFile = true;
  EXPECT_FALSE(loweredJson(s, {}, file).material["extensions"].contains("FV_materials_hair"));

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
  const auto  lowered = loweredJson(s, e, {});
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
  const auto   lowered     = loweredJson(s, e, {});
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
  EXPECT_NEAR(loweredJson(s, e, {}).material["extensions"]["FV_materials_hair"]["rootAbsorption"][0].get<float>(), s.hairFibre->rootSigma[0], 1e-3f);
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
  EXPECT_NE(fv::mat::requestKey(loweredJson(s, {}, {}).requests[1]), fv::mat::requestKey(loweredJson(s, e, {}).requests[1]));
  // a factor does not
  fv::mat::Edits r;
  r.roughness = 0.2f;
  EXPECT_EQ(loweredJson(s, {}, {}).requests.size(), loweredJson(s, r, {}).requests.size());
}

}  // namespace
