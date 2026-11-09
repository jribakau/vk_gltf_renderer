// What a material reflects, read through its closure, against real materials' baseline.

#include "materials_test_util.hpp"

using namespace materials_test;

namespace {

using fv::probe::ClosureAt;

constexpr int kDiffuse = 0, kTransmission = 1, kSpecular = 2, kMetal = 3, kClearcoat = 5;

std::vector<ClosureAt> gltfClosure(const std::function<void(tinygltf::Material&)>& set)
{
  tinygltf::Material m;
  m.pbrMetallicRoughness.baseColorFactor = {0.8, 0.5, 0.25, 1.0};
  m.pbrMetallicRoughness.metallicFactor  = 0.0;
  m.pbrMetallicRoughness.roughnessFactor = 0.5;
  set(m);
  return fv::probe::closureOfGltf(m);
}

struct ClosureFixture
{
  const char*              name;
  const char*              surface;
  std::vector<const char*> layers;
  fv::mat::SkinMode        skin = fv::mat::SkinMode::eVolume;
  bool                     thin = false;
};

std::vector<ClosureFixture> closureFixtures()
{
  using fv::mat::SkinMode;
  return {
      {"iray skin, measured", "Face", {kSkin}, SkinMode::eMeasured},
      {"iray skin, measured preview", "Face", {kSkin}, SkinMode::eMeasured, true},
      {"iray skin, volume", "Face", {kSkin}, SkinMode::eVolume},
      {"iray skin, thin", "Face", {kSkin}, SkinMode::eThin},
      {"iray skin, raster", "Face", {kSkin}, SkinMode::eRaster},
      {"eye moisture", "Eye Moisture Left", {kEyeMoisture}},
      {"cornea", "Cornea", {kG9Moisture}},
      {"daz default boxers", "Boxers", {kBoxers}},
      {"daz default satin", "Top", {kBoxers, kTopValues}},
      {"daz default metal", "Buckle", {kBoxers, kMetallic}},
      {"daz default glass", "Lens", {kBoxers, kRefracts}},
      {"ubersurface skin", "Torso", {kG2fTorso}, SkinMode::eMeasured},
      {"ubersurface cloth", "Shirt", {kG2fTorso, kUberCloth}},
      {"ubersurface lens", "Lens", {kG2fTorso, kUberCloth, kUberLens}},
      {"aoa skin", "Torso", {kAoaBrick}, SkinMode::eMeasured},
  };
}

double round4(float v)
{
  return std::round(double(v) * 10000.0) / 10000.0;
}

Json closureJson(const std::vector<ClosureAt>& closure)
{
  Json out = Json::object();
  for(const ClosureAt& at : closure)
  {
    Json lobes = Json::object();
    for(int l = 0; l < fv::probe::kLobeCount; l++)
      if(at.lobes[l].weight > 0.0f)
        lobes[fv::probe::lobeName(l)] = {{"weight", round4(at.lobes[l].weight)},
                                         {"albedo", {round4(at.lobes[l].albedo[0]), round4(at.lobes[l].albedo[1]), round4(at.lobes[l].albedo[2])}}};
    char key[16];
    std::snprintf(key, sizeof(key), "%.2f", at.cosine);
    out[key] = {{"total", {round4(at.total[0]), round4(at.total[1]), round4(at.total[2])}}, {"lobes", lobes}};
  }
  return out;
}

// Every number that moved, as `path: was -> now`
void closureDiff(const Json& was, const Json& now, const std::string& path, std::vector<std::string>& moved)
{
  if(now.is_object() || was.is_object())
  {
    std::set<std::string> keys;
    for(const Json* side : {&was, &now})
      if(side->is_object())
        for(const auto& [key, _] : side->items())
          keys.insert(key);
    for(const std::string& key : keys)
      closureDiff(was.is_object() && was.contains(key) ? was[key] : Json(), now.is_object() && now.contains(key) ? now[key] : Json(), path + "/" + key, moved);
    return;
  }
  if(now.is_array() && was.is_array() && now.size() == was.size())
  {
    for(size_t i = 0; i < now.size(); i++)
      closureDiff(was[i], now[i], path + "/" + std::to_string(i), moved);
    return;
  }
  // the probe's fixed seed repeats, so this is room for a compiler's rounding, not sampling noise
  if(was.is_number() && now.is_number() && std::abs(was.get<double>() - now.get<double>()) < 0.002)
    return;
  moved.push_back(path + ": " + was.dump() + " -> " + now.dump());
}

}  // namespace

// The probe sees a row as the shader does: lobes, factors and the light each lobe gets.
TEST(MaterialClosure, ThePlasticLobesAreTheirFactors)
{
  const auto closure = gltfClosure([](tinygltf::Material&) {});
  ASSERT_EQ(closure.size(), 4u);
  const ClosureAt& head = closure[0];  // a view along the normal
  EXPECT_EQ(head.cosine, 1.0f);
  EXPECT_NEAR(head.lobes[kDiffuse].albedo[0], 0.8f, 0.01f);
  EXPECT_NEAR(head.lobes[kDiffuse].albedo[2], 0.25f, 0.01f);
  EXPECT_NEAR(head.lobes[kSpecular].weight, 0.04f, 0.01f);
  EXPECT_EQ(head.lobes[kMetal].weight, 0.0f);
  EXPECT_EQ(head.lobes[kTransmission].weight, 0.0f);
  // Fresnel: the specular lobe gets more of the light toward grazing, and the diffuse less
  EXPECT_GT(closure[3].lobes[kSpecular].weight, 2.0f * head.lobes[kSpecular].weight);
  EXPECT_LT(closure[3].lobes[kDiffuse].weight, head.lobes[kDiffuse].weight);
}

TEST(MaterialClosure, AMetalIsItsMetalLobe)
{
  const auto closure = gltfClosure([](tinygltf::Material& m) { m.pbrMetallicRoughness.metallicFactor = 1.0; });
  EXPECT_NEAR(closure[0].lobes[kMetal].weight, 1.0f, 1e-4f);
  EXPECT_NEAR(closure[0].total[0], 0.8f, 0.02f);
  EXPECT_NEAR(closure[0].total[2], 0.25f, 0.02f);
  EXPECT_EQ(closure[0].lobes[kDiffuse].weight, 0.0f);
}

// A texture is neutral: the row's factors are what is measured, whatever maps it names
TEST(MaterialClosure, TexturesAreNeutral)
{
  const auto plain  = gltfClosure([](tinygltf::Material&) {});
  const auto mapped = gltfClosure([](tinygltf::Material& m) {
    m.pbrMetallicRoughness.baseColorTexture.index = 0;
    m.normalTexture.index                         = 1;
  });
  EXPECT_NEAR(mapped[1].total[0], plain[1].total[0], 1e-4f);
  EXPECT_NEAR(mapped[1].lobes[kSpecular].weight, plain[1].lobes[kSpecular].weight, 1e-4f);
}

TEST(MaterialClosure, ACoatTakesItsShareFirst)
{
  const auto coated = gltfClosure([](tinygltf::Material& m) {
    tinygltf::Value::Object coat;
    coat["clearcoatFactor"]                 = tinygltf::Value(1.0);
    coat["clearcoatRoughnessFactor"]        = tinygltf::Value(0.2);
    m.extensions["KHR_materials_clearcoat"] = tinygltf::Value(coat);
  });
  const auto plain = gltfClosure([](tinygltf::Material&) {});
  EXPECT_NEAR(coated[0].lobes[kClearcoat].weight, 0.04f, 0.01f);
  EXPECT_LT(coated[0].lobes[kDiffuse].weight, plain[0].lobes[kDiffuse].weight);
  EXPECT_GT(coated[3].lobes[kClearcoat].weight, coated[0].lobes[kClearcoat].weight);
}

namespace {

// Skin in the walk's medium (FV_materials_subsurface), as the measured mode lowers it, with a detail
// layer: its normal map is texture 2, at scale 1 and weight 0.6, as HID Kelly's
constexpr int    kDetailTexture = 2;
constexpr double kDetailWeight  = 0.6;

tinygltf::Material detailSkin(double roughnessMul = 1.0)
{
  tinygltf::Material m;
  m.pbrMetallicRoughness.baseColorFactor = {0.8, 0.5, 0.4, 1.0};
  m.pbrMetallicRoughness.metallicFactor  = 0.0;
  m.pbrMetallicRoughness.roughnessFactor = 0.5;
  tinygltf::Value::Object subsurface;
  subsurface["scatterDistance"]           = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(0.00367), tinygltf::Value(0.00137), tinygltf::Value(0.00068)});
  m.extensions["FV_materials_subsurface"] = tinygltf::Value(subsurface);
  tinygltf::Value::Object normal;
  normal["index"] = tinygltf::Value(kDetailTexture);
  normal["scale"] = tinygltf::Value(1.0);
  tinygltf::Value::Object detail;
  detail["detailNormalTexture"]       = tinygltf::Value(normal);
  detail["detailWeightFactor"]        = tinygltf::Value(kDetailWeight);
  detail["detailRoughnessFactor"]     = tinygltf::Value(roughnessMul);
  m.extensions["FV_materials_detail"] = tinygltf::Value(detail);
  return m;
}

// The detail normal map: a footprint's mip and the last mip each a flat normal of the given length,
// as a map cooked to keep it (normalLength) holds them
fv::probe::ClosureOptions detailMap(bool preview, float footprintLength, float meanLength)
{
  fv::probe::ClosureOptions options;
  options.thinScatter = preview;
  fv::probe::ProbeImage image;
  image.texture = kDetailTexture;
  for(int c = 0; c < 2; c++)
    image.colour[c] = image.mean[c] = 0.5f;
  image.colour[2] = 0.5f + 0.5f * footprintLength;
  image.mean[2]   = 0.5f + 0.5f * meanLength;
  options.images  = {image};
  return options;
}

float toksvig(float length)
{
  return (1.0f - length) / length;
}

}  // namespace

// The skin preview leaves a detail map's relief out (one sample a pixel would only shimmer with it),
// and every lobe takes what the still's samples spread a highlight by instead: a mean normal of 0.957,
// a pore map with slopes of 0.21 a side, at weight 0.6 adds 2 x 0.045 x 0.36 to alpha squared
TEST(MaterialClosure, TheSkinPreviewTakesADetailMapsReliefAsRoughness)
{
  const float added   = 2.0f * toksvig(0.957f) * float(kDetailWeight * kDetailWeight);
  const auto  still   = fv::probe::roughnessOfGltf(detailSkin(), detailMap(false, 1.0f, 0.957f));
  const auto  preview = fv::probe::roughnessOfGltf(detailSkin(), detailMap(true, 1.0f, 0.957f));
  // the still bends its normal by the relief instead
  EXPECT_NEAR(still.alpha[0], 0.25f, 1e-5f);
  EXPECT_NEAR(preview.alpha[0], std::sqrt(0.0625f + added), 1e-4f);
  EXPECT_NEAR(preview.alpha[1], preview.alpha[0], 1e-6f);
  // the coat and the second lobe lie on the relief too: alpha^2 + the same, as roughness
  const auto widened = [&](float r) { return std::sqrt(std::sqrt(r * r * r * r + added)); };
  EXPECT_NEAR(preview.clearcoat, widened(still.clearcoat), 1e-4f);
  EXPECT_NEAR(preview.specular2, widened(still.specular2), 1e-4f);
}

// Only what the footprint's mip still holds: the rest the mip has averaged away for the still as well
TEST(MaterialClosure, TheSkinPreviewTakesOnlyTheReliefTheStillResolves)
{
  const auto averaged = fv::probe::roughnessOfGltf(detailSkin(), detailMap(true, 0.957f, 0.957f));
  EXPECT_NEAR(averaged.alpha[0], 0.25f, 1e-5f);
  const auto  part  = fv::probe::roughnessOfGltf(detailSkin(), detailMap(true, 0.98f, 0.957f));
  const float added = 2.0f * (toksvig(0.957f) - toksvig(0.98f)) * float(kDetailWeight * kDetailWeight);
  EXPECT_NEAR(part.alpha[0], std::sqrt(0.0625f + added), 1e-4f);
  // a map rebuilt unit length (BC5) tells nothing: no roughness, as before it kept its length
  const auto unit = fv::probe::roughnessOfGltf(detailSkin(), detailMap(true, 1.0f, 1.0f));
  EXPECT_NEAR(unit.alpha[0], 0.25f, 1e-5f);
}

// The detail's roughness multiplier holds in the preview as in the still, so the two agree on it
TEST(MaterialClosure, TheSkinPreviewKeepsTheDetailsRoughnessMultiplier)
{
  const auto  still   = fv::probe::roughnessOfGltf(detailSkin(0.8), detailMap(false, 1.0f, 1.0f));
  const auto  preview = fv::probe::roughnessOfGltf(detailSkin(0.8), detailMap(true, 1.0f, 1.0f));
  const float mul     = 1.0f + (0.8f - 1.0f) * float(kDetailWeight);
  EXPECT_NEAR(still.alpha[0], 0.25f * mul * mul, 1e-5f);
  EXPECT_NEAR(preview.alpha[0], still.alpha[0], 1e-6f);
}

// The real channels' closures against tests/figure/data/closure_baseline.json. A lowering or closure
// change that moves one fails here, listing what moved: read the list, then rewrite the baseline by
// running with FV_UPDATE_CLOSURE_BASELINE=1 and commit it with the change.
TEST(MaterialClosure, TheBaselineHolds)
{
  const std::filesystem::path file = std::filesystem::path(FV_TEST_DATA) / "closure_baseline.json";
  Json                        now  = Json::object();
  for(const ClosureFixture& f : closureFixtures())
  {
    fv::dson::Channels ch;
    for(const char* text : f.layers)
    {
      const Json def = Json::parse(text);
      fv::dson::collectChannels(&def, ch, {}, "x.dsf");
    }
    fv::probe::ClosureOptions options;
    options.thinScatter = f.thin;
    now[f.name]         = closureJson(fv::probe::closureOf(fv::dson::buildMaterial(f.surface, ch), f.skin, options));
  }
  if(std::getenv("FV_UPDATE_CLOSURE_BASELINE"))
  {
    std::ofstream(file, std::ios::binary) << now.dump(1) << "\n";
    GTEST_SKIP() << "rewrote " << file.string();
  }
  std::ifstream in(file, std::ios::binary);
  ASSERT_TRUE(in.good()) << file.string();
  std::vector<std::string> moved;
  closureDiff(Json::parse(in), now, "", moved);
  std::string list;
  for(const std::string& line : moved)
    list += line + "\n";
  EXPECT_TRUE(moved.empty()) << moved.size() << " closure numbers moved:\n" << list;
}

// What the library sweep calls dropped: a channel with a value that parsing never looked up
TEST(MaterialClosure, ChannelReadsTellWhatTheParserLeaves)
{
  const fv::dson::Channels ch = layered({kBoxers, R"({"extra":[{"channels":[{"channel":{"id":"Nobody Reads This","current_value":1}}]}]})"});
  fv::dson::ChannelReads   reads;
  (void)fv::dson::parseSurface("Boxers", ch);
  EXPECT_TRUE(reads.read(ch.get(fv::dson::norm("diffuse"))));
  EXPECT_TRUE(reads.read(ch.get(fv::dson::norm("glossiness"))));
  // the lookups just above count too, so this one is asked for by its pointer, unread until now
  const auto unread = ch.shared(fv::dson::norm("Nobody Reads This"));
  ASSERT_TRUE(unread);
  EXPECT_FALSE(reads.read(unread.get()));
}
