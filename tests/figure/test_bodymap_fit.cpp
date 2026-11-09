// Fitted body maps (wardrobe plan step 3b): the reference laid onto a body Genesis 9 has no clone of,
// then projected both ways; a body cut to what shows ends at its open edges, and what it lacks is left
// unmapped. Synthetic bodies only (the plan's rule 7); Genesis 8 Female fitted from Genesis 9 against
// its clone map, and an outfit's carried Skyrim body, are fv_parity's bodymap step.

#include <gtest/gtest.h>

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
  EXPECT_EQ(fv::bodymap::regionLandmark(Region::eForearm), fv::rig::Landmark::eFull);
  EXPECT_EQ(fv::bodymap::regionLandmark(Region::eFoot), fv::rig::Landmark::eAlongLength);
  EXPECT_EQ(fv::bodymap::regionLandmark(Region::eCollar), fv::rig::Landmark::eNone);
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
