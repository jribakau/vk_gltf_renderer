// Fitted body maps (wardrobe plan step 3b): the reference laid onto a body Genesis 9 has no clone of,
// then projected both ways; a body cut to what shows ends at its open edges, and what it lacks is left
// unmapped. Synthetic bodies only (the plan's rule 7); Genesis 8 Female fitted from Genesis 9 against
// its clone map, and an outfit's carried Skyrim body, are fv_parity's bodymap step.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include <figure/bodymap/cache.hpp>
#include <figure/bodymap/clone.hpp>
#include <figure/bodymap/fit.hpp>
#include <figure/bodymap/regions.hpp>
#include <figure/wardrobe/fit/bases.hpp>

namespace {

using fv::bodymap::Body;
using fv::bodymap::RegionMask;
using fv::rig::Region;

// A left leg as a base cage: rings every `step` cm (5, or a whole share of it) down from the hip at
// y = 100 to y = 10 (or the first ring at or above `cutAt`, left open there), each of `segments` vertices at `radius(y)` round x = 10, z = 0,
// capped at the foot. The hip's ring is held half by the pelvis and half by the thigh, the knee's at
// y = 55 half by the thigh and half by the shin; the rest by one or the other. Then scaled about the
// origin and shifted.
fv::wardrobe::BaseCage leg(int segments, const std::function<double(double)>& radius, double scale = 1, fv::Vec3 shift = {}, double cutAt = -1e9, double step = 5)
{
  fv::wardrobe::BaseCage c;
  c.skeleton = {{"hip", std::nullopt, {10, 100, 0}}, {"pelvis", "hip", {10, 100, 0}}, {"lThighBend", "pelvis", {10, 100, 0}}, {"lShin", "lThighBend", {10, 55, 0}}};
  c.skin.start.push_back(0);
  const auto add = [&](double x, double y, double z) {
    c.positions.insert(c.positions.end(), {float(x * scale + shift.x), float(y * scale + shift.y), float(z * scale + shift.z)});
    if(y >= 100)
      c.skin.bone.insert(c.skin.bone.end(), {1, 2}), c.skin.weight.insert(c.skin.weight.end(), {0.5f, 0.5f});
    else if(std::abs(y - 55) < 1e-9)
      c.skin.bone.insert(c.skin.bone.end(), {2, 3}), c.skin.weight.insert(c.skin.weight.end(), {0.5f, 0.5f});
    else
      c.skin.bone.push_back(y > 55 ? 2 : 3), c.skin.weight.push_back(1);
    c.skin.start.push_back(int32_t(c.skin.bone.size()));
  };
  int rings = 0;
  for(double y = 100; y >= 10 - 1e-9 && y >= cutAt; y -= step, rings++)
    for(int k = 0; k < segments; k++)
    {
      const double a = 2 * std::numbers::pi * k / segments;
      add(10 + radius(y) * std::cos(a), y, radius(y) * std::sin(a));
      c.uvs.insert(c.uvs.end(), {float(k) / float(segments), float(y / 100)});
    }
  for(int r = 0; r + 1 < rings; r++)
    for(int k = 0; k < segments; k++)
    {
      const uint32_t a = uint32_t(r * segments + k), b = uint32_t(r * segments + (k + 1) % segments), d = a + uint32_t(segments), e = b + uint32_t(segments);
      c.tris.insert(c.tris.end(), {a, b, d, b, e, d});
    }
  if(cutAt <= 10)
  {
    const uint32_t last = uint32_t((rings - 1) * segments), pole = uint32_t(rings * segments);
    add(10, 5, 0);
    c.uvs.insert(c.uvs.end(), {0.5f, 0.05f});
    for(int k = 0; k < segments; k++)
      c.tris.insert(c.tris.end(), {last + uint32_t(k), last + uint32_t((k + 1) % segments), pole});
  }
  return c;
}

const auto kStraight = [](double) { return 5.0; };

}  // namespace

// A region's neighbours are the one above and those below, on its side; never the other leg.
TEST(BodyMapFit, ARegionsNeighboursAreAboveAndBelowOnItsSide)
{
  using fv::bodymap::regionBit;
  const RegionMask thigh = fv::bodymap::withNeighbours(regionBit(Region::eThigh, 'l'));
  EXPECT_TRUE(thigh & regionBit(Region::ePelvis));
  EXPECT_TRUE(thigh & regionBit(Region::eShin, 'l'));
  EXPECT_FALSE(thigh & regionBit(Region::eThigh, 'r'));
  EXPECT_FALSE(thigh & regionBit(Region::eShin, 'r'));
  const RegionMask forearm = fv::bodymap::withNeighbours(regionBit(Region::eForearm, 'r'));
  EXPECT_TRUE(forearm & regionBit(Region::eUpperArm, 'r'));
  EXPECT_TRUE(forearm & regionBit(Region::eHand, 'r'));
  EXPECT_FALSE(forearm & regionBit(Region::eHand, 'l'));
  // every region hanging from one, not only the one its joint aims at; an unsided region's on both sides
  // (a body mod weighting its bust to the chest lands on Genesis 9's breasts)
  const RegionMask chest = fv::bodymap::withNeighbours(regionBit(Region::eChest));
  for(const char side : {'l', 'r'})
  {
    EXPECT_TRUE(chest & regionBit(Region::eBreast, side));
    EXPECT_TRUE(chest & regionBit(Region::eCollar, side));
  }
  EXPECT_TRUE(chest & regionBit(Region::eNeck));
  EXPECT_FALSE(chest & regionBit(Region::eUpperArm, 'l'));
  const RegionMask hand = fv::bodymap::withNeighbours(regionBit(Region::eHand, 'l'));
  for(const Region finger : {Region::eThumb, Region::eIndex, Region::eMid, Region::eRing, Region::ePinky})
  {
    EXPECT_TRUE(hand & regionBit(finger, 'l'));
    EXPECT_FALSE(hand & regionBit(finger, 'r'));
  }
  const RegionMask pelvis = fv::bodymap::withNeighbours(regionBit(Region::ePelvis));
  EXPECT_TRUE(pelvis & regionBit(Region::eAbdomen));
  EXPECT_TRUE(pelvis & regionBit(Region::eThigh, 'l'));
  EXPECT_TRUE(pelvis & regionBit(Region::eThigh, 'r'));
  EXPECT_EQ(fv::bodymap::regionLandmark(Region::eForearm), fv::rig::Landmark::eFull);
  EXPECT_EQ(fv::bodymap::regionLandmark(Region::eFoot), fv::rig::Landmark::eAlongLength);
  EXPECT_EQ(fv::bodymap::regionLandmark(Region::eCollar), fv::rig::Landmark::eNone);
}

// A hand is turned into the source's rest about the line to its middle finger too, by where its thumb
// is: the same hand turned palm down (90 degrees about the forearm) points the same way, and only the
// thumb tells them apart.
TEST(BodyMapFit, AHandIsTurnedByItsThumb)
{
  fv::wardrobe::BaseCage hand;
  hand.skeleton = {{"hip", std::nullopt, {0, 0, 0}}, {"lForearmBend", "hip", {-20, 0, 0}}, {"lHand", "lForearmBend", {0, 0, 0}},
                   {"lThumb1", "lHand", {4, 0, 3}},  {"lMid1", "lHand", {8, 0, 0}}};
  // the wrist's ring, half forearm and half hand; the palm; the middle finger's knuckles; the thumb's
  hand.positions = {0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1,  4, 0.5f, 0, 4, -0.5f, 0, 5, 0, 0.8f,
                    8, 0.3f, 0, 8, -0.3f, 0, 8, 0, 0.3f,     4, 0, 3, 4.5f, 0.3f, 3};
  hand.skin.start.push_back(0);
  const auto skin = [&](std::vector<int32_t> bones, std::vector<float> weights) {
    hand.skin.bone.insert(hand.skin.bone.end(), bones.begin(), bones.end());
    hand.skin.weight.insert(hand.skin.weight.end(), weights.begin(), weights.end());
    hand.skin.start.push_back(int32_t(hand.skin.bone.size()));
  };
  for(int i = 0; i < 4; i++)
    skin({1, 2}, {0.5f, 0.5f});
  for(int i = 0; i < 3; i++)
    skin({2}, {1});
  for(int i = 0; i < 3; i++)
    skin({2, 4}, {0.5f, 0.5f});
  for(int i = 0; i < 2; i++)
    skin({2, 3}, {0.5f, 0.5f});
  // palm down: turned 90 degrees about the forearm's line, x
  Body source;
  source.positions = hand.positions;
  for(size_t v = 0; v < source.positions.size() / 3; v++)
  {
    const float y = hand.positions[v * 3 + 1], z = hand.positions[v * 3 + 2];
    source.positions[v * 3 + 1] = -z;
    source.positions[v * 3 + 2] = y;
  }
  source.regions = fv::bodymap::vertexRegions(hand.skin, hand.skeleton, source.positions.size() / 3);

  std::vector<fv::bodymap::LimbTurn> turns;
  const std::vector<float>           posed = fv::bodymap::intoRest(hand, hand.positions, source, &turns);
  for(const auto& t : turns)
    if(t.joint == "hand")
      EXPECT_NEAR(t.degrees, 90, 1e-3);
  // all but the wrist's ring, which the forearm holds half
  for(size_t i = 4 * 3; i < posed.size(); i++)
    EXPECT_NEAR(posed[i], source.positions[i], 1e-4) << i;
}

// A forearm, hand and middle finger as one tube along x: rings every cm from the elbow at x = -10 to the
// fingertip at 16, capped there; the forearm 1.2 round, the palm bulging to 1.6 between the wrist (x = 0)
// and the knuckles (x = 8), the finger tapering from 0.8 to 0.5. The wrist's ring is half forearm and
// half hand, the knuckles' half hand and half finger. Then the palm stretched along x by `palm` and the
// finger by `finger`, from the knuckles on.
namespace {

fv::wardrobe::BaseCage handTube(double palm = 1, double finger = 1)
{
  fv::wardrobe::BaseCage c;
  const auto             stretched = [&](double x) { return x < 0 ? x : x <= 8 ? x * palm : 8 * palm + (x - 8) * finger; };
  const auto             radius    = [](double x) { return x < 0 ? 1.2 : x < 8 ? 1.2 + 0.4 * std::sin(std::numbers::pi * x / 8) : x == 8 ? 1.0 : 0.8 - 0.3 * (x - 8) / 8; };
  c.skeleton = {{"hip", std::nullopt, {-10, 0, 0}}, {"lForearmBend", "hip", {-10, 0, 0}}, {"lHand", "lForearmBend", {0, 0, 0}}, {"lMid1", "lHand", {float(8 * palm), 0, 0}}};
  c.skin.start.push_back(0);
  constexpr int segments = 12;
  int           rings    = 0;
  for(int i = -10; i < 16; i++, rings++)
    for(int k = 0; k < segments; k++)
    {
      const double x = i, a = 2 * std::numbers::pi * k / segments;
      c.positions.insert(c.positions.end(), {float(stretched(x)), float(radius(x) * std::cos(a)), float(radius(x) * std::sin(a))});
      if(i == 0)
        c.skin.bone.insert(c.skin.bone.end(), {1, 2}), c.skin.weight.insert(c.skin.weight.end(), {0.5f, 0.5f});
      else if(i == 8)
        c.skin.bone.insert(c.skin.bone.end(), {2, 3}), c.skin.weight.insert(c.skin.weight.end(), {0.5f, 0.5f});
      else
        c.skin.bone.push_back(i < 0 ? 1 : i < 8 ? 2 : 3), c.skin.weight.push_back(1);
      c.skin.start.push_back(int32_t(c.skin.bone.size()));
    }
  for(int r = 0; r + 1 < rings; r++)
    for(int k = 0; k < segments; k++)
    {
      const uint32_t a = uint32_t(r * segments + k), b = uint32_t(r * segments + (k + 1) % segments), d = a + segments, e = b + segments;
      c.tris.insert(c.tris.end(), {a, b, d, b, e, d});
    }
  const uint32_t last = uint32_t((rings - 1) * segments), tip = uint32_t(rings * segments);
  c.positions.insert(c.positions.end(), {float(stretched(16)), 0, 0});
  c.skin.bone.push_back(3), c.skin.weight.push_back(1), c.skin.start.push_back(int32_t(c.skin.bone.size()));
  for(int k = 0; k < segments; k++)
    c.tris.insert(c.tris.end(), {last + uint32_t(k), last + uint32_t((k + 1) % segments), tip});
  return c;
}

}  // namespace

// A hand and its fingers are moved and sized along their length into the source's rest as well as
// turned: a source whose palm is an eighth longer (the knuckles a centimetre further from the wrist) and
// whose finger a fifth longer has its knuckles and fingertip where the reference's land. Turned alone,
// the tip would end 2.6 cm short.
TEST(BodyMapFit, AHandAndFingerAreMovedAndSizedOntoTheSources)
{
  const fv::wardrobe::BaseCage reference = handTube();
  const Body                   source    = fv::bodymap::bodyOf(handTube(1.125, 1.2));
  std::vector<fv::bodymap::LimbTurn> turns;
  const std::vector<float>           posed = fv::bodymap::intoRest(reference, reference.positions, source, &turns, true);
  const size_t                       tip   = posed.size() / 3 - 1;
  EXPECT_NEAR(posed[tip * 3], 18.6, 0.2);
  EXPECT_NEAR(posed[tip * 3 + 1], 0, 0.05);
  EXPECT_NEAR(posed[tip * 3 + 2], 0, 0.05);
  // the knuckles' ring, the 19th (x = -10 .. 8)
  double x = 0;
  for(size_t k = 0; k < 12; k++)
    x += posed[(18 * 12 + k) * 3] / 12.0;
  EXPECT_NEAR(x, 9, 0.2);
  for(const auto& t : turns)
    if(t.joint == "mid")
      EXPECT_NEAR(t.sized, 1.2, 0.05);
  // a clone's reference is already shaped as the source: only turned, and this one's straight
  const std::vector<float> turned = fv::bodymap::intoRest(reference, reference.positions, source);
  EXPECT_NEAR(turned[tip * 3], 16, 1e-3);
}

// A body cut to what shows ends at the edges only one triangle has: a point on one is on the cut, a
// point inside or on an edge two triangles share isn't; a projection told to refuse the cut leaves a
// vertex past it unmapped.
TEST(BodyMapFit, ACutBodyEndsAtItsOpenEdges)
{
  Body square;
  square.positions = {0, 0, 0, 10, 0, 0, 10, 0, 10, 0, 0, 10};
  square.tris      = {0, 3, 1, 1, 3, 2};  // facing +y; edge 1-3 shared
  square.outer     = {1, 1};
  square.regions   = {0, 0, 0, 0};
  const fv::bodymap::CutEdges cut(square);
  const double                inside[3] = {0.4, 0.3, 0.3}, onOpen[3] = {0.5, 0.5, 0}, onShared[3] = {0, 0.5, 0.5}, atCorner[3] = {1, 0, 0};
  EXPECT_FALSE(cut.on(0, inside));
  EXPECT_TRUE(cut.on(0, onOpen));     // edge 0-3
  EXPECT_FALSE(cut.on(0, onShared));  // edge 3-1, both triangles'
  EXPECT_TRUE(cut.on(0, atCorner));   // vertex 0, on the open edges

  // a strip above it: its vertex 0 over the middle, its vertex 1 past the open edge at x = 10
  Body above;
  above.positions = {5, 1, 5, 14, 1, 5, 5, 1, 6, 14, 1, 6};
  above.tris      = {0, 2, 1, 1, 2, 3};  // facing +y
  above.outer     = {1, 1};
  above.regions   = {0, 0, 0, 0};
  fv::bodymap::ProjectOptions keep, refuse;
  refuse.refuseCut = true;
  EXPECT_TRUE(fv::bodymap::project(above, square, keep).mapped[1]);
  const auto refused = fv::bodymap::project(above, square, refuse);
  EXPECT_TRUE(refused.mapped[0]);
  EXPECT_FALSE(refused.mapped[1]);
}

// The first move: a leg scaled and shifted is found from its joint bands, and the reference lands on it.
TEST(BodyMapFit, AFitFindsTheScaleAndShiftFromTheJointBands)
{
  const fv::wardrobe::BaseCage cage   = leg(16, kStraight);
  const Body                   source = fv::bodymap::bodyOf(leg(16, kStraight, 1.1, {2, -3, 1}));
  const fv::bodymap::FitResult fit    = fv::bodymap::fitReference(cage, fv::bodymap::bodyOf(cage), source);
  EXPECT_NEAR(fit.scale, 1.1, 1e-4);
  EXPECT_NEAR(fit.shift.x, 2, 1e-3);
  EXPECT_NEAR(fit.shift.y, -3, 1e-3);
  EXPECT_NEAR(fit.shift.z, 1, 1e-3);
  EXPECT_EQ(fit.landmarks, 2u);  // the hip's and the knee's bands
  for(size_t i = 0; i < fit.positions.size(); i++)
    EXPECT_NEAR(fit.positions[i], source.positions[i], 0.02) << i;
}

// A straight leg laid onto a thicker one with a calf, tessellated otherwise (both finely: across a coarse
// polygon's corners a round trip cuts the corner): every pulled vertex ends on the source's skin, and the
// map comes back to where it started, without folds.
TEST(BodyMapFit, AFitLaysTheReferenceOntoAnotherShape)
{
  const auto calf = [](double y) { return 6 + 1.5 * std::exp(-(y - 35) * (y - 35) / 60); };
  fv::bodymap::FitInputs in;
  in.source    = std::make_shared<const fv::wardrobe::BaseCage>(leg(48, calf, 1, {}, -1e9, 2.5));
  in.g9        = std::make_shared<const fv::wardrobe::BaseCage>(leg(40, kStraight, 1, {}, -1e9, 2.5));
  in.reference = fv::bodymap::bodyOf(*in.g9);
  fv::bodymap::FitOptions options;
  options.nodes = 60;
  fv::bodymap::FitResult        fit;
  const fv::bodymap::TypeMap    made = fv::bodymap::makeFittedMap(in, options, &fit);
  EXPECT_GT(fit.pulled, size_t(1200));
  EXPECT_LT(fit.residualP95, 0.05);
  const fv::bodymap::MapQuality q = fv::bodymap::mapQuality(made.source, made.reference, made.map);
  EXPECT_EQ(q.mapped, 1.0);
  // all but the hip's ring, which lands on the source's open top: past what it has
  EXPECT_GE(q.inverseCovers, 0.9);
  EXPECT_LT(q.roundTrip.at("all").p95, 0.1);
  EXPECT_GE(q.foldFree, 0.99);
  EXPECT_EQ(made.map.frameTurn.size(), made.source.positions.size() / 3);
}

// A body cut above the knee: the reference's shin has nothing to land on, so the projection back leaves
// it unmapped, the thigh mapped; the shin isn't turned onto the stump.
TEST(BodyMapFit, APartTheSourceLacksIsLeftUnmapped)
{
  fv::bodymap::FitInputs in;
  in.source    = std::make_shared<const fv::wardrobe::BaseCage>(leg(16, kStraight, 1, {}, 70));
  in.g9        = std::make_shared<const fv::wardrobe::BaseCage>(leg(16, kStraight));
  in.reference = fv::bodymap::bodyOf(*in.g9);
  fv::bodymap::FitOptions options;
  options.nodes = 60;
  fv::bodymap::FitResult     fit;
  const fv::bodymap::TypeMap made = fv::bodymap::makeFittedMap(in, options, &fit);
  for(const auto& t : fit.turns)
    if(t.joint == "shin")
      EXPECT_NEAR(t.degrees, 0, 1e-6);
  const size_t n = made.reference.positions.size() / 3;
  size_t       below = 0, above = 0;
  for(size_t v = 0; v < n; v++)
  {
    const float y = in.g9->positions[v * 3 + 1];
    if(y <= 60)
    {
      EXPECT_FALSE(made.map.fromReference.mapped[v]) << v << " at " << y;
      below++;
    }
    else if(y >= 80 && y < 100)  // the hip's ring lands on the source's open top: past what it has
    {
      EXPECT_TRUE(made.map.fromReference.mapped[v]) << v << " at " << y;
      above++;
    }
  }
  EXPECT_GT(below, 0u);
  EXPECT_GT(above, 0u);
  EXPECT_EQ(fv::bodymap::mapQuality(made.source, made.reference, made.map).mapped, 1.0);
}

// A fitted map read back from the cache is the one made; other options are another map.
TEST(BodyMapFit, AFittedMapReadBackIsTheOneMade)
{
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "fv_bodymap_fit_cache";
  std::filesystem::remove_all(dir);
  fv::bodymap::setMapCacheDir(dir);
  fv::bodymap::FitInputs in;
  in.source    = std::make_shared<const fv::wardrobe::BaseCage>(leg(12, [](double) { return 5.5; }));
  in.g9        = std::make_shared<const fv::wardrobe::BaseCage>(leg(16, kStraight));
  in.reference = fv::bodymap::bodyOf(*in.g9);
  fv::bodymap::FitOptions options;
  options.nodes                    = 40;
  const auto                 start = fv::bodymap::mapCacheCounts();
  const fv::bodymap::TypeMap made  = fv::bodymap::cachedFittedMap(in, options);
  const fv::bodymap::TypeMap read  = fv::bodymap::cachedFittedMap(in, options);
  EXPECT_EQ(fv::bodymap::mapCacheCounts().misses, start.misses + 1);
  EXPECT_EQ(fv::bodymap::mapCacheCounts().hits, start.hits + 1);
  EXPECT_EQ(made.reference.positions, read.reference.positions);
  EXPECT_EQ(made.map.toReference.at.verts, read.map.toReference.at.verts);
  EXPECT_EQ(made.map.fromReference.mapped, read.map.fromReference.mapped);
  options.nodes = 41;
  fv::bodymap::cachedFittedMap(in, options);
  EXPECT_EQ(fv::bodymap::mapCacheCounts().misses, start.misses + 2);
  fv::bodymap::setMapCacheDir({});
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// An outfit's carried body is matched to the whole body it was cut from by UV, whatever its own vertex
// order and preset; its map is the whole body's through the match, and what it lacks stays unmapped.
TEST(BodyMapFit, ACutBodyIsMatchedToItsWholeBodyAndMappedThroughIt)
{
  const fv::wardrobe::BaseCage whole = leg(24, kStraight);
  // the leg above the knee, at another preset (2% fuller), its vertices in reverse order
  const fv::wardrobe::BaseCage above = leg(24, kStraight, 1.02, {-0.2f, -2, 0}, 70);
  fv::wardrobe::BaseCage       cut   = above;
  const size_t                 nc    = above.positions.size() / 3;
  for(size_t v = 0; v < nc; v++)
  {
    const size_t from = nc - 1 - v;
    for(size_t i = 0; i < 3; i++)
      cut.positions[v * 3 + i] = above.positions[from * 3 + i];
    for(size_t i = 0; i < 2; i++)
      cut.uvs[v * 2 + i] = above.uvs[from * 2 + i];
  }
  for(uint32_t& t : cut.tris)
    t = uint32_t(nc - 1 - t);
  cut.skin.start.assign(1, 0), cut.skin.bone.clear(), cut.skin.weight.clear();
  for(size_t v = 0; v < nc; v++)
  {
    const size_t from = nc - 1 - v;
    for(int32_t i = above.skin.start[from]; i < above.skin.start[from + 1]; i++)
      cut.skin.bone.push_back(above.skin.bone[size_t(i)]), cut.skin.weight.push_back(above.skin.weight[size_t(i)]);
    cut.skin.start.push_back(int32_t(cut.skin.bone.size()));
  }

  const std::vector<int32_t> match = fv::bodymap::matchCut(cut, whole);
  ASSERT_EQ(match.size(), nc);
  for(size_t v = 0; v < nc; v++)
    EXPECT_EQ(match[v], int32_t(nc - 1 - v)) << v;

  fv::bodymap::FitInputs in;
  in.source    = std::make_shared<const fv::wardrobe::BaseCage>(whole);
  in.g9        = std::make_shared<const fv::wardrobe::BaseCage>(leg(16, kStraight));
  in.reference = fv::bodymap::bodyOf(*in.g9);
  fv::bodymap::FitOptions options;
  options.nodes                    = 60;
  const fv::bodymap::TypeMap wholeMap = fv::bodymap::makeFittedMap(in, options);
  const fv::bodymap::Body    cutBody  = fv::bodymap::bodyOf(cut);
  const fv::bodymap::TypeMap cutMap   = fv::bodymap::cutMap(wholeMap, cutBody, match);
  for(size_t v = 0; v < nc; v++)
  {
    const size_t w = size_t(match[v]);
    EXPECT_EQ(cutMap.map.toReference.mapped[v], wholeMap.map.toReference.mapped[w]);
    for(size_t j = 0; j < 3; j++)
      EXPECT_EQ(cutMap.map.toReference.at.verts[v * 3 + j], wholeMap.map.toReference.at.verts[w * 3 + j]);
  }
  // a reference vertex below the cut has no place on it; one well above keeps the whole body's, on the
  // cut's own vertices
  for(size_t r = 0; r < cutMap.reference.positions.size() / 3; r++)
  {
    const float y = in.g9->positions[r * 3 + 1];
    if(y <= 60)
      EXPECT_FALSE(cutMap.map.fromReference.mapped[r]) << r;
    else if(y >= 80 && y < 100)
    {
      ASSERT_TRUE(cutMap.map.fromReference.mapped[r]) << r;
      for(size_t j = 0; j < 3; j++)
        EXPECT_EQ(match[cutMap.map.fromReference.at.verts[r * 3 + j]], int32_t(wholeMap.map.fromReference.at.verts[r * 3 + j])) << r;
    }
  }
}
