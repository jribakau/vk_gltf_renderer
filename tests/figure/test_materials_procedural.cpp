// Procedural layers: where sweat lands on a body, and what the lowering makes of the amount.

#include <figure/materials/procedural.hpp>

#include "materials_test_util.hpp"

using namespace materials_test;

namespace {

// A skeleton of 170 cm in DSON's units and Genesis 8's names, facing +Z
const std::vector<std::pair<std::string, fv::Vec3>> kGenesis8 = {
    {"hip", {0, 100, 0}},          {"abdomenLower", {0, 108, 0}},  {"abdomenUpper", {0, 115, 0}}, {"chestLower", {0, 125, 0}},
    {"chestUpper", {0, 135, 0}},   {"neckLower", {0, 148, 0}},     {"head", {0, 156, 0}},         {"lEye", {3.2, 160, 6}},
    {"rEye", {-3.2, 160, 6}},      {"lShldrBend", {16, 142, 0}},   {"rShldrBend", {-16, 142, 0}}, {"lShldrTwist", {24, 135, 0}},
    {"lForearmBend", {32, 125, 0}}, {"rForearmBend", {-32, 125, 0}}, {"lHand", {48, 105, 0}},     {"rHand", {-48, 105, 0}},
    {"lShin", {10, 50, 0}},        {"rShin", {-10, 50, 0}},        {"lFoot", {10, 8, 0}},         {"rFoot", {-10, 8, 0}},
};

const fv::mat::ProceduralDef& sweatDef()
{
  return fv::mat::proceduralDefs().front();
}

float sweatAt(const fv::mat::BodyLandmarks& body, const fv::Vec3& p)
{
  return fv::mat::proceduralField(sweatDef(), sweatDef().film, body, p);
}

// The field's mean over a small cube, so the blotches' noise doesn't decide a comparison
float meanSweat(const fv::mat::BodyLandmarks& body, const fv::Vec3& p, double side = 3)
{
  float sum = 0;
  int   n   = 0;
  for(int x = -3; x <= 3; x++)
    for(int y = -3; y <= 3; y++)
      for(int z = -3; z <= 3; z++, n++)
        sum += sweatAt(body, {p.x + x * side / 6, p.y + y * side / 6, p.z + z * side / 6});
  return sum / float(n);
}

TEST(Procedural, LandmarksComeFromEitherGenerationsNamesWithTheHeightFromTheBones)
{
  const auto g8 = fv::mat::bodyLandmarks(kGenesis8);
  EXPECT_NEAR(g8.height, (160 - 8) * 1.12, 1e-6);
  EXPECT_TRUE(g8.at.count("eye"));
  EXPECT_DOUBLE_EQ(g8.at.at("eye").y, 160);
  EXPECT_DOUBLE_EQ(g8.at.at("eye").x, 0);
  EXPECT_TRUE(g8.at.count("spine4"));
  EXPECT_TRUE(g8.at.count("l:upperarm"));
  // the bend bone is the joint, not the twist bone that comes after it in another skeleton's order
  EXPECT_DOUBLE_EQ(g8.at.at("l:upperarm").x, 16);

  const std::vector<std::pair<std::string, fv::Vec3>> g9 = {
      {"spine4", {0, 135, 0}}, {"l_eye", {3, 160, 6}}, {"r_eye", {-3, 160, 6}}, {"l_upperarm", {16, 142, 0}}, {"l_foot", {10, 8, 0}}, {"r_foot", {-10, 8, 0}},
  };
  const auto nine = fv::mat::bodyLandmarks(g9);
  EXPECT_GT(nine.height, 0);
  EXPECT_TRUE(nine.at.count("spine4"));
  EXPECT_TRUE(nine.at.count("l:upperarm"));

  // Genesis 9's eyes are on a mesh of their own, which comes after the body's: bones add up, the first
  // of a slot staying
  const std::vector<std::pair<std::string, fv::Vec3>> twoMeshes = {
      {"l_foot", {10, 8, 0}}, {"r_foot", {-10, 8, 0}}, {"spine4", {0, 135, 0}}, {"spine4", {0, 999, 0}}, {"l_eye", {3, 160, 6}}, {"r_eye", {-3, 160, 6}},
  };
  const auto joined = fv::mat::bodyLandmarks(twoMeshes);
  EXPECT_GT(joined.height, 0);
  EXPECT_DOUBLE_EQ(joined.at.at("spine4").y, 135);

  // what isn't a body has none, and a whole body missing its feet can't be measured
  EXPECT_EQ(fv::mat::bodyLandmarks(std::vector<std::pair<std::string, fv::Vec3>>{{"sleeve", {0, 0, 0}}}).height, 0);
  EXPECT_EQ(fv::mat::bodyLandmarks(std::vector<std::pair<std::string, fv::Vec3>>{{"lEye", {3, 160, 6}}, {"rEye", {-3, 160, 6}}}).height, 0);
}

TEST(Procedural, SweatStartsOnTheForeheadSpineAndUnderarmsAheadOfTheLimbs)
{
  const auto     body     = fv::mat::bodyLandmarks(kGenesis8);
  const fv::Vec3 forehead{0, 164.5, 8};
  const fv::Vec3 spine{0, 125, -10};
  const fv::Vec3 underarm{16, 135, 0};
  const fv::Vec3 chest{0, 137, 10};
  const fv::Vec3 shinFront{10, 35, 6};
  const fv::Vec3 thigh{10, 75, 8};
  const float    limbs = std::max(meanSweat(body, shinFront), meanSweat(body, thigh));
  EXPECT_GT(meanSweat(body, forehead), 0.5f);
  EXPECT_GT(meanSweat(body, forehead), limbs * 2);
  EXPECT_GT(meanSweat(body, spine), limbs * 2);
  EXPECT_GT(meanSweat(body, underarm), limbs * 2);
  EXPECT_GT(meanSweat(body, chest), limbs * 1.5f);
  // a dry limb is still not nothing: the amount's top end reaches it
  EXPECT_GT(limbs, 0.05f);
}

TEST(Procedural, TheFieldStaysInRangeAndAgreesLeftAndRightApartFromItsBlotches)
{
  const auto body = fv::mat::bodyLandmarks(kGenesis8);
  for(double y = 0; y <= 180; y += 7)
    for(double x = -50; x <= 50; x += 7)
    {
      const float s = sweatAt(body, {x, y, 3});
      EXPECT_GE(s, 0.f);
      EXPECT_LE(s, 1.f);
    }
  EXPECT_NEAR(meanSweat(body, {16, 135, 0}, 24), meanSweat(body, {-16, 135, 0}, 24), 0.1f);
}

TEST(Procedural, NoSkeletonGivesAnEvenFieldRatherThanNone)
{
  const fv::mat::BodyLandmarks none;
  const float                  high = meanSweat(none, {0, 165, 8});
  EXPECT_GT(high, 0.2f);
  EXPECT_NEAR(high, meanSweat(none, {0, 30, 8}), 0.15f);
}

TEST(Procedural, MasksPutEachLayerInItsChannelInTheMeshsWorld)
{
  const auto            body = fv::mat::bodyLandmarks(kGenesis8);
  std::vector<float>    position{0, 164.5f, 8, 10, 35, 6};
  std::vector<uint32_t> masks = fv::mat::proceduralMasks(position, fv::Mat4{}, body);
  ASSERT_EQ(masks.size(), 2u);
  EXPECT_NEAR(float(masks[0] & 0xffu) / 255.f, sweatAt(body, {0, 164.5, 8}), 1.f / 255);
  EXPECT_NEAR(float((masks[0] >> 8) & 0xffu) / 255.f, fv::mat::proceduralField(sweatDef(), sweatDef().drops, body, {0, 164.5, 8}), 1.f / 255);
  EXPECT_EQ(masks[0] >> 16, 0u);

  // a mesh whose own space is 2 units a world's: its vertices land in the same places
  fv::Mat4 twice;
  twice.e[0] = twice.e[5] = twice.e[10] = 2;
  std::vector<float> halved{0, 82.25f, 4, 5, 17.5f, 3};
  EXPECT_EQ(fv::mat::proceduralMasks(halved, twice, body), masks);
}

// ---- the lowering

constexpr float kFaceMetresPerUv = 0.43f;

TEST(Procedural, AnAmountOnSkinLowersToTheLayerWithItsTileAtItsSize)
{
  const auto    skin = fv::dson::parseSurface("Face", channelsOf(kSkin));
  fv::mat::Edits e;
  e.sweat            = 0.4f;
  const auto lowered = fv::mat::lower(skin, e, {.skin = fv::mat::SkinMode::eMeasured, .metresPerUv = kFaceMetresPerUv});
  ASSERT_TRUE(lowered.material.procedural);
  const fv::mat::Procedural& l = *lowered.material.procedural;
  EXPECT_FLOAT_EQ(l.amount, 0.4f);
  EXPECT_EQ(l.channel, sweatDef().channel);
  EXPECT_EQ(l.dropChannel, sweatDef().dropChannel);
  EXPECT_NE(l.channel, l.dropChannel);
  EXPECT_FLOAT_EQ(l.coat, sweatDef().coat);
  EXPECT_FLOAT_EQ(l.filmRoughness, sweatDef().filmRoughness);
  ASSERT_TRUE(l.normalMap.has());
  ASSERT_TRUE(l.coverMap.has());
  const float tiles = kFaceMetresPerUv / fv::mat::kSweatTileMetres;
  EXPECT_FLOAT_EQ(l.normalMap.transform->scale[0], tiles);
  EXPECT_FLOAT_EQ(l.coverMap.transform->scale[1], tiles);
  // the relief is the tile's own, so one cooked image serves every surface
  const fv::mat::TextureRequest& height = lowered.requests.at(size_t(l.normalMap.index));
  EXPECT_EQ(height.rel, fv::mat::kSweatHeightMap);
  EXPECT_EQ(height.channels, "bn");
  EXPECT_FALSE(height.srgb);
  EXPECT_FLOAT_EQ(height.bumpHeight, fv::mat::kSweatRelief);
  // the shader reads its mips' lengths (Toksvig), so it is cooked whole, not as BC5's X and Y; of the
  // surface's other maps only the micro-relief's asks for that too (the skin preview reads it), and
  // the ask is a request of its own
  EXPECT_TRUE(height.normalLength);
  for(size_t i = 0; i < lowered.requests.size(); i++)
    EXPECT_EQ(lowered.requests[i].normalLength, i == size_t(l.normalMap.index) || lowered.requests[i].rel == fv::mat::kSkinMicroMap)
        << lowered.requests[i].rel;
  fv::mat::TextureRequest plainHeight = height;
  plainHeight.normalLength            = false;
  EXPECT_NE(fv::mat::requestKey(height), fv::mat::requestKey(plainHeight));
  const fv::mat::TextureRequest& cover = lowered.requests.at(size_t(l.coverMap.index));
  EXPECT_EQ(cover.rel, fv::mat::kSweatCoverMap);
  EXPECT_FALSE(cover.srgb);
  EXPECT_FLOAT_EQ(*lowered.effective.sweat, 0.4f);
}

TEST(Procedural, EverySkinModeTakesIt)
{
  const auto     skin = fv::dson::parseSurface("Face", channelsOf(kSkin));
  fv::mat::Edits e;
  e.sweat = 1;
  for(fv::mat::SkinMode mode : {fv::mat::SkinMode::eRaster, fv::mat::SkinMode::eThin, fv::mat::SkinMode::eVolume, fv::mat::SkinMode::eMeasured})
    EXPECT_TRUE(fv::mat::lower(skin, e, {.skin = mode, .metresPerUv = kFaceMetresPerUv}).material.procedural);
}

TEST(Procedural, NoAmountNoLayerAndTheSliderStillShowsOnSkin)
{
  const auto skin    = fv::dson::parseSurface("Face", channelsOf(kSkin));
  auto       lowered = fv::mat::lower(skin, {}, {.metresPerUv = kFaceMetresPerUv});
  EXPECT_FALSE(lowered.material.procedural);
  EXPECT_FLOAT_EQ(*lowered.effective.sweat, 0);
  // an amount brought back to 0 costs what none does
  fv::mat::Edits zero;
  zero.sweat = 0;
  EXPECT_FALSE(fv::mat::lower(skin, zero, {.metresPerUv = kFaceMetresPerUv}).material.procedural);
}

TEST(Procedural, OnlySkinTakesItAndAFileForOtherToolsLeavesItOut)
{
  fv::mat::Edits e;
  e.sweat              = 1;
  fv::mat::Surface eye = fv::dson::parseSurface("Face", channelsOf(kSkin));
  eye.kind             = fv::mat::Kind::eEye;
  EXPECT_FALSE(fv::mat::lower(eye, e, {.metresPerUv = kFaceMetresPerUv}).material.procedural);
  EXPECT_FALSE(fv::mat::lower(eye, e, {.metresPerUv = kFaceMetresPerUv}).effective.sweat);

  const auto skin = fv::dson::parseSurface("Face", channelsOf(kSkin));
  EXPECT_FALSE(fv::mat::lower(skin, e, {.forFile = true, .metresPerUv = kFaceMetresPerUv}).material.procedural);
}

TEST(Procedural, ASurfaceOfUnknownSizeHasTheFilmWithNoTile)
{
  const auto     skin = fv::dson::parseSurface("Face", channelsOf(kSkin));
  fv::mat::Edits e;
  e.sweat            = 1;
  const auto lowered = fv::mat::lower(skin, e, {.metresPerUv = 0});
  ASSERT_TRUE(lowered.material.procedural);
  EXPECT_FALSE(lowered.material.procedural->normalMap.has());
  EXPECT_FALSE(lowered.material.procedural->coverMap.has());
}

TEST(Procedural, TheAmountIsAnEditThatRoundTripsThroughAScene)
{
  fv::mat::Edits e;
  fv::mat::applyValue(e, "sweat", 0.7);
  ASSERT_TRUE(e.sweat);
  EXPECT_FLOAT_EQ(*e.sweat, 0.7f);
  EXPECT_FALSE(e.empty());
  fv::mat::applyValue(e, "sweat", 5);
  EXPECT_FLOAT_EQ(*e.sweat, 1.f);
  EXPECT_TRUE(fv::mat::figureOnlyKey("sweat"));

  fv::mat::Edits saved;
  saved.sweat = 0.3f;
  fv::mat::Edits merged;
  merged.merge(saved);
  EXPECT_FLOAT_EQ(*merged.sweat, 0.3f);
}

TEST(Procedural, TheLayerRoundTripsThroughGltf)
{
  fv::mat::Material m;
  fv::mat::Procedural l;
  l.amount        = 0.25f;
  l.channel       = 1;
  l.dropChannel   = 2;
  l.tint          = {0.5f, 0.6f, 0.7f};
  l.filmRoughness = 0.3f;
  l.coat          = 0.6f;
  l.dropRoughness = 0.07f;
  l.normalMap.index = 2;
  l.normalMap.scale = 1.5f;
  l.coverMap.index  = 3;
  m.procedural      = l;

  std::vector<std::string> used;
  const tinygltf::Material g = fv::mat::toGltf(m, used);
  EXPECT_NE(std::find(used.begin(), used.end(), FV_MATERIALS_PROCEDURAL_EXTENSION_NAME), used.end());
  const fv::mat::Material back = fv::mat::fromGltf(g);
  ASSERT_TRUE(back.procedural);
  EXPECT_EQ(*back.procedural, l);
}

}  // namespace
