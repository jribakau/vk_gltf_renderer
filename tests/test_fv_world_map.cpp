// A planet's base map (src/fv_world_map.hpp): the cube's grids across their edges, water that all
// runs down to the sea or a lake, rivers whose levels never rise and whose beds lie under their
// water, and the same map for a seed however many threads bake it.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include <gtest/gtest.h>

#include <figure/jobs/pool.hpp>

#include "../src/fv_world_map.hpp"

namespace {

namespace W = fv::world;

constexpr double kPi = 3.14159265358979323846;

// A small map: what the tests ask of it holds at any size
constexpr uint32_t kCells = 128;

const W::World& planet()
{
  static const W::World w = W::make({}, {.map = kCells});
  return w;
}

glm::dvec3 spread(uint32_t i, uint32_t count)
{
  const double y   = 1.0 - 2.0 * (double(i) + 0.5) / double(count);
  const double r   = std::sqrt(std::max(1.0 - y * y, 0.0));
  const double phi = double(i) * 2.399963229728653;
  return {r * std::cos(phi), y, r * std::sin(phi)};
}

glm::dvec3 middleOf(const W::RiverPiece& p)
{
  return glm::normalize(0.25 * p.a + 0.5 * p.b + 0.25 * p.c);
}

}  // namespace

TEST(WorldMapGrid, ACellsNeighboursAreBesideItAcrossTheCubesEdgesToo)
{
  const W::CubeGrid g(16);
  const double      cell = 0.5 * kPi / 16.0;  // radians, the largest
  int               none = 0;
  for(uint32_t c = 0; c < g.count(); c++)
  {
    uint32_t nb[8];
    g.around(c, nb);
    for(int k = 0; k < 8; k++)
    {
      if(nb[k] == W::CubeGrid::kNone)
      {
        none++;
        continue;
      }
      ASSERT_LT(nb[k], g.count());
      EXPECT_NE(nb[k], c);
      // beside it: no further than a cell's diagonal (by a corner a cell is a rhombus, with a long one)
      EXPECT_LT(glm::length(g.centre(nb[k]) - g.centre(c)), 1.7 * cell) << c << " " << k;
      // and it is its neighbour's neighbour
      uint32_t back[8];
      g.around(nb[k], back);
      EXPECT_NE(std::find(back, back + 8, c), back + 8) << c << " " << k;
    }
    // no neighbour twice
    std::set<uint32_t> distinct(nb, nb + 8);
    distinct.erase(W::CubeGrid::kNone);
    EXPECT_GE(distinct.size(), 7U);
  }
  // at each of the cube's eight corners three cells meet, each short of one diagonal neighbour
  EXPECT_EQ(none, 24);
}

TEST(WorldMapGrid, AGutterHoldsTheFaceBeyond)
{
  const W::CubeGrid  g(16);
  std::vector<float> cells(g.count()), padded;
  for(uint32_t c = 0; c < g.count(); c++)
    cells[c] = float(c);
  g.pad(cells, padded, false, -1.0F);
  for(uint8_t face = 0; face < 6; face++)
    for(int j = -W::kMapGutter; j < 16 + W::kMapGutter; j++)
      for(int i = -W::kMapGutter; i < 16 + W::kMapGutter; i++)
      {
        uint32_t c = 0;
        if(g.cell(face, i, j, c))
          EXPECT_EQ(padded[g.padded(face, i, j)], cells[c]);
        else
          EXPECT_EQ(padded[g.padded(face, i, j)], -1.0F);
      }
  // a direction's place on the grid is its cell's
  for(uint32_t c = 0; c < g.count(); c += 7)
  {
    const W::MapPoint p = g.at(g.centre(c));
    EXPECT_EQ(g.index(p.face, uint32_t(std::lround(p.x)), uint32_t(std::lround(p.y))), c);
    EXPECT_NEAR(p.x, std::round(p.x), 1e-6);
  }
}

TEST(WorldMapGrid, ItsCellsByThePlanetsSize)
{
  EXPECT_EQ(W::mapCells(2000.0), 64U);
  EXPECT_EQ(W::mapCells(20000.0), 512U);
  EXPECT_EQ(W::mapCells(100000.0), 1024U);
}

TEST(WorldMapBake, AllTheLandsWaterRunsDownToTheSeaOrALake)
{
  const W::BaseMap& m = *planet().base;
  ASSERT_EQ(m.grid.cells(), kCells);
  const uint32_t count = m.grid.count();
  uint32_t       land = 0, ends = 0;
  for(uint32_t c = 0; c < count; c++)
  {
    if(m.kind[c] == W::eMapSea)
      continue;
    land++;
    EXPECT_GT(m.flow[c], 0.0F);
    const uint32_t to = m.down[c];
    if(to == W::CubeGrid::kNone)
    {
      ends++;  // (a lake's hollow with no way out, which is rare; never the land itself)
      EXPECT_NE(m.kind[c], W::eMapLand) << c;
      continue;
    }
    if(m.kind[to] == W::eMapSea)
      continue;
    // downhill, where both are worn ground (a lake's hollow drains over its brim), and into no less water
    if(m.kind[c] == W::eMapLand && m.kind[to] == W::eMapLand)
      EXPECT_GT(m.height[c], m.height[to]) << c;
    EXPECT_GE(m.flow[to], m.flow[c] * 0.999F) << c;
  }
  ASSERT_GT(land, count / 5);
  EXPECT_LT(ends, 8U);
  // followed down, every way ends: at the sea, or in a lake
  for(uint32_t c = 0; c < count; c += 13)
  {
    uint32_t at = c, steps = 0;
    while(m.kind[at] != W::eMapSea && m.down[at] != W::CubeGrid::kNone && steps < count)
    {
      at = m.down[at];
      steps++;
    }
    EXPECT_LT(steps, count);
  }
}

TEST(WorldMapBake, TheWearCutsValleysAndKeepsTheShore)
{
  const W::BaseMap& m = *planet().base;
  // against the land the stages make by themselves
  const W::World bare = W::make({}, {.base = false});
  uint32_t       lower = 0, land = 0;
  for(uint32_t c = 0; c < m.grid.count(); c++)
  {
    const double before = W::sample(bare, m.grid.centre(c), m.cell).elevation;
    // what was sea is sea, what was land is land
    EXPECT_EQ(before <= 0.0, m.kind[c] == W::eMapSea) << c;
    if(m.kind[c] == W::eMapSea)
      continue;
    land++;
    EXPECT_GT(m.height[c], 0.0F);
    lower += double(m.height[c]) < before - 5.0;
  }
  // a fair share of the land is worn down by more than a few metres
  EXPECT_GT(lower, land / 10);
}

TEST(WorldMapBake, ARiversLevelNeverRisesOnItsWay)
{
  const W::BaseMap& m = *planet().base;
  ASSERT_GT(m.pieces.size(), 50U);
  for(const W::RiverPiece& p : m.pieces)
  {
    EXPECT_GE(p.level[0], p.level[1]);
    EXPECT_GE(p.level[1], p.level[2]);
    EXPECT_GE(p.level[2], 0.0F);
    for(int k = 0; k < 3; k++)
    {
      EXPECT_GT(p.half[k], 0.0F);
      EXPECT_LE(p.half[k], 0.41F * float(m.cell));
    }
    // a stretch is no longer than the lookup's window reaches: a cell and its neighbour's, and how
    // far a river's point strays in each
    EXPECT_NEAR(glm::length(p.b), 1.0, 1e-9);
    EXPECT_LT(glm::length(p.a - p.b) * planet().spec.radius, 2.2 * m.cell);
    EXPECT_LT(glm::length(p.c - p.b) * planet().spec.radius, 2.2 * m.cell);
  }
}

TEST(WorldMapBake, TheSameHoweverManyThreadsBakeIt)
{
  const W::World&   w = planet();
  const W::BaseMap& a = *w.base;
  std::shared_ptr<const W::BaseMap> b;
  {
    // (the stages before the map, as make has them while it bakes)
    W::World first = W::make({}, {.base = false});
    first.stages.resize(4);
    fv::jobs::Serial serial;
    b = W::bake(first, kCells);
  }
  ASSERT_TRUE(b);
  EXPECT_EQ(a.height, b->height);
  EXPECT_EQ(a.flow, b->flow);
  EXPECT_EQ(a.rain, b->rain);
  EXPECT_EQ(a.pieces.size(), b->pieces.size());
  EXPECT_EQ(a.lakes, b->lakes);
}

TEST(WorldMapBake, StoppedItLeavesNoMap)
{
  std::atomic<bool> stop{true};
  const W::World    w = W::make({}, {.map = kCells, .stop = &stop});
  EXPECT_FALSE(w.base);
}

TEST(WorldMapWater, ARiversBedLiesUnderItsWaterAndItsBanksOverIt)
{
  const W::World&   w = planet();
  const W::BaseMap& m = *w.base;
  int               wide = 0, wet = 0, banks = 0, dryBanks = 0;
  for(size_t k = 0; k < m.pieces.size(); k += 3)
  {
    const W::RiverPiece& p = m.pieces[k];
    if(p.half[1] < 3.0F)
      continue;
    wide++;
    const glm::dvec3 mid = middleOf(p);
    const W::Sample  s   = W::sample(w, mid, 1.0);
    if(s.inland.lake > 0.0)
      continue;  // where a river meets a lake the lake's water is over both
    // in the river's middle the ground is its bed, under its water
    EXPECT_TRUE(s.inland.river);
    EXPECT_LT(s.shore, 0.0F);
    wet += s.water > s.elevation + 0.3;
    // the water's mesh is at that level there
    EXPECT_EQ(W::waterLevel(w, mid, 1.0), std::max(s.water, 0.0));
    // and past its banks no water lies on the ground
    const glm::dvec3 side = glm::normalize(glm::cross(p.c - p.a, mid));
    const double     past = double(p.half[1]) + 1.3 * s.inland.bank;
    const W::Sample  out  = W::sample(w, glm::normalize(mid + side * (past / w.spec.radius)), 1.0);
    if(out.inland.any() || out.elevation < 0.0)
      continue;  // another river, a lake, the sea
    banks++;
    dryBanks += !(out.water > out.elevation);
  }
  ASSERT_GT(wide, 20);
  EXPECT_GT(wet, wide * 8 / 10);
  ASSERT_GT(banks, 5);
  EXPECT_EQ(dryBanks, banks);
}

TEST(WorldMapWater, WaterLiesOnTheGroundOnlyWhereARiverOrALakeIs)
{
  const W::World& w = planet();
  int             under = 0;
  for(uint32_t i = 0; i < 40000; i++)
  {
    const glm::dvec3 d = spread(i, 40000);
    for(const double footprint : {1.0, 8.0, 64.0})
    {
      const W::Sample s = W::sample(w, d, footprint);
      EXPECT_EQ(W::waterLevel(w, d, footprint), std::max(s.water, 0.0));
      if(s.water > s.elevation && s.elevation > 0.0)
      {
        under++;
        // in the water, or within what draws the sample of its edge
        EXPECT_LT(s.shore, 0.5F + 2.0F * float(footprint)) << i;
      }
    }
  }
  EXPECT_GT(under, 0);
}

TEST(WorldMapWater, ALakeIsLevelAndHeldInByItsShore)
{
  // some planet of a few has a lake
  for(const int seed : {1, 2, 3, 4})
  {
    const W::World    held = seed == 1 ? W::World{} : W::make({.seed = seed}, {.map = kCells});
    const W::World&   w    = seed == 1 ? planet() : held;
    const W::BaseMap& m    = *w.base;
    if(m.lakes == 0)
      continue;
    int in = 0, under = 0;
    for(uint32_t c = 0; c < m.grid.count(); c++)
    {
      if(m.kind[c] != W::eMapLake)
        continue;
      // a lake's cells are under its level on the map
      const float level = m.lake[m.grid.padded(uint8_t(c / (kCells * kCells)), int(c % kCells), int((c / kCells) % kCells))];
      ASSERT_GT(level, 0.0F);
      EXPECT_LT(m.height[c], level);
      // and on the ground its water is at that level (the ground's fine relief may stand out of it)
      const W::Sample s = W::sample(w, m.grid.centre(c), 4.0);
      EXPECT_NEAR(s.inland.lakeLevel, double(level), 1e-3);
      EXPECT_GE(s.water, double(level) - 1e-3);
      in++;
      under += s.water > s.elevation;
      // round it, out of its cells, the ground stands over that level
      uint32_t nb[8];
      m.grid.around(c, nb);
      for(int k = 0; k < 8; k += 2)
      {
        if(nb[k] == W::CubeGrid::kNone || m.kind[nb[k]] == W::eMapLake)
          continue;
        const glm::dvec3 out = glm::normalize(m.grid.centre(c) * -0.25 + m.grid.centre(nb[k]) * 1.25);
        const W::Sample  o   = W::sample(w, out, 4.0);
        if(o.inland.river || o.inland.lake > 0.3 || o.elevation < 0.0)
          continue;  // where it spills, or another cell of it beyond
        EXPECT_FALSE(o.water > o.elevation) << c;
      }
    }
    ASSERT_GT(in, 0);
    EXPECT_GT(under, in / 2);
    return;
  }
  FAIL() << "no lake on any of the planets";
}

// A camera in a lake starts its paths in the water (shaders/world_sea.h.slang): what it is in is the
// planet's to say
TEST(WorldMapWater, APointUnderALakesSurfaceIsInIt)
{
  // some planet of a few has a lake
  for(const int seed : {1, 2, 3, 4})
  {
    const W::World    held = seed == 1 ? W::World{} : W::make({.seed = seed}, {.map = kCells});
    const W::World&   w    = seed == 1 ? planet() : held;
    const W::BaseMap& m    = *w.base;
    if(m.lakes == 0)
      continue;
    int lakes = 0;
    for(uint32_t c = 0; c < m.grid.count(); c++)
    {
      if(m.kind[c] != W::eMapLake)
        continue;
      const glm::dvec3 d = m.grid.centre(c);
      const W::Sample  s = W::sample(w, d, 1.0);
      if(!(s.water > s.elevation + 1.0))
        continue;  // the ground's fine relief stands out of it here, or nearly
      lakes++;
      // half way down it: in it, and it says its level; over its surface: in none
      EXPECT_NEAR(W::inlandWaterOver(w, W::toScene(w, d, 0.5 * (s.water + s.elevation))), s.water, 1e-6);
      EXPECT_EQ(W::inlandWaterOver(w, W::toScene(w, d, s.water + 0.2)), W::kDry);
    }
    EXPECT_GT(lakes, 3);
    // dry ground has none over it, nor has the sea, under its surface or over it
    for(uint32_t i = 0; i < 3000; i++)
    {
      const glm::dvec3 d = spread(i, 3000);
      const W::Sample  s = W::sample(w, d, 1.0);
      if(s.water > -1.0e20)
        continue;  // a river or a lake is near
      EXPECT_EQ(W::inlandWaterOver(w, W::toScene(w, d, s.elevation < 0.0 ? 0.5 * s.elevation : s.elevation + 1.0)), W::kDry);
      EXPECT_EQ(W::inlandWaterOver(w, W::toScene(w, d, std::max(s.elevation, 0.0) + 30.0)), W::kDry);
    }
    // a planet with no map has no inland water
    const W::World bare = W::make({}, {.base = false});
    EXPECT_EQ(W::inlandWaterOver(bare, glm::dvec3(0.0, -5.0, 0.0)), W::kDry);
    return;
  }
  FAIL() << "no lake on any of the planets";
}

// A deep lake's bed can lie under the sea's level. It is the lake's bed all the same: dark mud, not
// the pale sand of the sea's shallows (which drew a disc in the lake, seen from the air)
TEST(WorldMapWater, ALakesBedUnderTheSeasLevelIsStillALakesBed)
{
  const W::World&   w = planet();
  const W::BaseMap& m = *w.base;
  int               found = 0;
  for(uint32_t c = 0; c < m.grid.count(); c++)
  {
    if(m.kind[c] != W::eMapLake)
      continue;
    const glm::dvec3 d = m.grid.centre(c);
    const W::Sample  s = W::sample(w, d, 1.0);
    if(!(s.elevation < -0.5) || !(s.water > s.elevation + 8.0))
      continue;
    found++;
    EXPECT_LT(std::max({s.colour.x, s.colour.y, s.colour.z}), 0.1F) << "elevation " << s.elevation << ", water " << s.water;
    // and the camera there is in the lake, under the sea's level or not
    EXPECT_NEAR(W::inlandWaterOver(w, W::toScene(w, d, s.elevation + 0.3)), s.water, 1e-6);
  }
  EXPECT_GT(found, 0) << "this planet's deepest lake goes under the sea's level by its south-east shore";
}

TEST(WorldMapWater, TheFiguresStandOnDryGround)
{
  for(const int seed : {1, 2, 3})
  {
    const W::World w = W::make({.seed = seed}, {.map = kCells});
    for(double a = 0.0; a < 6.28; a += 0.5)
      for(const double away : {0.0, 10.0, 30.0})
      {
        const W::Sample s = W::sample(w, W::toPlanet(w, glm::dvec3(std::cos(a), 0.0, std::sin(a)) * away), 1.0);
        EXPECT_FALSE(s.water > s.elevation) << seed;
        EXPECT_GT(s.shore, 10.0F) << seed;
      }
  }
}

TEST(WorldMapRain, TheLandIsWetInPlacesAndDryInOthers)
{
  const W::World& w = planet();
  int             land = 0, dry = 0, wet = 0;
  for(uint32_t i = 0; i < 6000; i++)
  {
    const glm::dvec3 d = spread(i, 6000);
    const float      r = W::mapRain(*w.base, d);
    EXPECT_GE(r, 0.0F);
    EXPECT_LE(r, 1.0F);
    if(W::sample(w, d, 100.0).elevation <= 0.0)
      continue;
    land++;
    dry += r < 0.25F;
    wet += r > 0.55F;
  }
  ASSERT_GT(land, 1000);
  EXPECT_GT(dry, land / 20);
  EXPECT_GT(wet, land / 20);
}

// The worn land's broad slope points uphill, lies along the ground, is one slope from either side of
// a cube's edge, and is kept again at half the cells, and half again
TEST(WorldMapSlope, ItPointsUphillAndIsTheSameAcrossTheCubesEdges)
{
  const W::World&   w = planet();
  const W::BaseMap& m = *w.base;
  ASSERT_FALSE(m.slopes.empty());
  ASSERT_EQ(m.slopes.front().cells, W::slopeCellsFor(m.grid.cells()));
  EXPECT_EQ(W::slopeCellsFor(1024), 256U);
  EXPECT_EQ(W::slopeCellsFor(64), 64U);
  for(size_t l = 1; l < m.slopes.size(); l++)
    EXPECT_EQ(m.slopes[l].cells * 2, m.slopes[l - 1].cells);
  EXPECT_EQ(W::slopeLevelFor(m, w.spec.radius, 0.5 * kPi * w.spec.radius / double(m.slopes.front().cells)), 0);
  EXPECT_EQ(W::slopeLevelFor(m, w.spec.radius, 2.0 * kPi * w.spec.radius / double(m.slopes.front().cells)), 2);
  EXPECT_EQ(W::slopeLevelFor(m, w.spec.radius, 1.0e9), int(m.slopes.size()) - 1);
  const auto heightAt = [&](const glm::dvec3& d) {
    const W::MapPoint p = m.grid.at(d);
    const int         n = int(m.grid.cells());
    return double(m.height[m.grid.index(p.face, uint32_t(std::clamp(int(std::lround(p.x)), 0, n - 1)), uint32_t(std::clamp(int(std::lround(p.y)), 0, n - 1)))]);
  };
  const double step = 3.0 * 0.5 * kPi / double(m.grid.cells());  // three cells, of the radius
  int          steep = 0, uphill = 0;
  for(uint32_t i = 0; i < 4000; i++)
  {
    const glm::dvec3 d = spread(i, 4000), g = W::mapSlope(m, d);
    EXPECT_NEAR(glm::dot(g, d), 0.0, 1e-9);
    if(glm::length(g) < 0.08)
      continue;
    const glm::dvec3 up = glm::normalize(g);
    steep++;
    uphill += heightAt(glm::normalize(d + up * step)) > heightAt(glm::normalize(d - up * step));
  }
  ASSERT_GT(steep, 200);
  EXPECT_GT(uphill, steep * 9 / 10);

  double most = 0.0;
  for(int level = 0; level < int(m.slopes.size()); level++)
    for(uint8_t face = 0; face < 6; face++)
      for(int side = 0; side < 4; side++)
        for(int k = 0; k <= 200; k++)
        {
          const double along = -1.0 + 2.0 * double(k) / 200.0;
          const auto   at    = [&](double off) {
            return W::direction(face, side == 0 ? -off : side == 1 ? off : along, side == 2 ? -off : side == 3 ? off : along);
          };
          const glm::dvec3 in = W::mapSlope(m, at(1.0 - 1e-9), level), out = W::mapSlope(m, at(1.0 + 1e-9), level);
          most = std::max(most, glm::length(in - out));
        }
  EXPECT_LT(most, 1e-5);
}

// Gullies: running water's furrows down the slopes. None where the land is level, by the sea or on
// its floor; they run down the slope, not across it; and none finer than what draws them.
TEST(WorldMapGullies, TheSlopesAreFurrowedDownTheirFall)
{
  // the land up to the gullies, and the same without them
  W::World with = planet();
  const auto stage = std::find_if(with.stages.begin(), with.stages.end(), [](const W::Stage& s) { return s.name == "gullies"; });
  ASSERT_NE(stage, with.stages.end());
  with.stages.erase(stage + 1, with.stages.end());
  W::World bare = with;
  bare.stages.pop_back();
  ASSERT_EQ(bare.stages.back().name, "base");
  const W::BaseMap& m       = *with.base;
  const double      R       = with.spec.radius;
  const double      largest = R / 40.0;                              // from one of the widest to the next
  const int         broad   = W::slopeLevelFor(m, R, 0.75 * largest);  // the land's slope they follow
  const auto        steepest = [&](const glm::dvec3& d) {
    double most = 0.0;
    for(int l = 0; l <= broad; l++)
      most = std::max(most, glm::length(W::mapSlope(m, d, l)));
    return most;
  };

  int    slopes = 0, sea = 0, level = 0;
  double furrowed = 0.0, across = 0.0, along = 0.0;
  for(uint32_t i = 0; i < 6000; i++)
  {
    const glm::dvec3 d     = spread(i, 6000);
    const W::Sample  b     = W::sample(bare, d, 2.0);
    const double     cut   = W::sample(with, d, 2.0).elevation - b.elevation;
    const double     steep = steepest(d);
    // no deeper than a share of the widest one's width, and nothing where the land is level
    EXPECT_LE(std::abs(cut), 0.35 * largest);
    if(steep < 0.03)
    {
      level++;
      EXPECT_EQ(cut, 0.0);
    }
    if(b.elevation <= 1.0)
    {
      sea++;
      EXPECT_EQ(cut, 0.0);
    }
    // what draws a gully with fewer than four points across it draws none
    EXPECT_EQ(W::sample(with, d, 0.26 * largest).elevation, W::sample(bare, d, 0.26 * largest).elevation);
    const glm::dvec3 rise = W::mapSlope(m, d, broad);
    if(glm::length(rise) < 0.15 || b.elevation < 60.0 || b.shore < 400.0F)
      continue;
    slopes++;
    furrowed += std::abs(cut);
    // the widest alone (what draws them finer has finer ones, turned into these): a step along the
    // rise stays in its furrow, a step across it leaves it
    const double     footprint = largest / 8.0, pace = 0.1 * largest / R;
    const glm::dvec3 up = glm::normalize(rise), side = glm::cross(d, up);
    const auto       cutAt = [&](const glm::dvec3& at) { return W::sample(with, at, footprint).elevation - W::sample(bare, at, footprint).elevation; };
    const double     here = cutAt(d);
    along += std::abs(cutAt(glm::normalize(d + up * pace)) - here);
    across += std::abs(cutAt(glm::normalize(d + side * pace)) - here);
  }
  ASSERT_GT(slopes, 50);
  ASSERT_GT(sea, 1000);
  ASSERT_GT(level, 100);
  EXPECT_GT(furrowed / double(slopes), 1.0);  // metres, on a planet 20 km in radius
  EXPECT_GT(across, 2.0 * along);

  // a planet with no map has none
  const W::World dry = W::make({}, {.base = false});
  EXPECT_TRUE(std::none_of(dry.stages.begin(), dry.stages.end(), [](const W::Stage& s) { return s.name == "gullies"; }));
}

// Not a test: what a quad's ground costs to make, with the gullies and without
TEST(WorldMapGullies, DISABLED_Cost)
{
 for(const double radius : {20000.0, 100000.0})
 {
  const W::World with = W::make({.radius = radius}, {.map = radius > 50000.0 ? 0U : kCells});
  W::World       bared = with;
  std::erase_if(bared.stages, [](const W::Stage& s) { return s.name == "gullies"; });
  const W::World& bare = bared;
  std::printf("a planet %.0f km in radius\n", radius / 1000.0);
  // quads over the steepest land found, a metre a vertex and sixteen
  glm::dvec3 steepest(0.0, 1.0, 0.0);
  double     most = 0.0;
  for(uint32_t i = 0; i < 6000; i++)
    if(const double s = glm::length(W::mapSlope(*with.base, spread(i, 6000))); s > most && W::sample(bare, spread(i, 6000), 8.0).elevation > 50.0)
    {
      most     = s;
      steepest = spread(i, 6000);
    }
  for(const uint8_t back : {uint8_t(0), uint8_t(4)})
  {
    const W::FacePoint p     = W::locate(steepest);
    const uint8_t      level = uint8_t(W::finestLevel(with) - back);
    const uint32_t     tiles = 1U << level;
    const W::Quad      q{p.face, level, std::min(uint32_t((p.u + 1.0) * 0.5 * tiles), tiles - 1), std::min(uint32_t((p.v + 1.0) * 0.5 * tiles), tiles - 1)};
    for(const W::World* w : {&bare, &with, &bare, &with, &bare, &with})
    {
      const auto from = std::chrono::steady_clock::now();
      for(int k = 0; k < 20; k++)
        W::groundMesh(*w, q);
      std::printf("level %u (%.1f m a quad) %s: %.2f ms a quad\n", unsigned(level), W::width(with, q), w == &with ? "gullies" : "bare   ",
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count() / 20.0);
    }
  }
 }
}

// Not a test, and an author's map: the land round a place as a shaded relief from straight above,
// north up, lit from the north-west, in the ground's colours (FV_WORLD_RELIEF = "latitude longitude kilometres pixels file.ppm [planet.json]": the
// image's width in kilometres; the default planet, or a planet file's)
TEST(WorldMapGullies, DISABLED_Relief)
{
  const char* asked = std::getenv("FV_WORLD_RELIEF");
  if(!asked)
    return;
  double lat = 0, lon = 0, km = 10;
  int    n = 512;
  char   file[512] = {}, planetFile[512] = {};
  ASSERT_GE(std::sscanf(asked, "%lf %lf %lf %d %511s %511s", &lat, &lon, &km, &n, file, planetFile), 5);
  W::Spec spec;
  bool    sited = false;
  if(planetFile[0])
  {
    std::ifstream     in(planetFile, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    const W::PlanetFile planet = W::parsePlanet(text.str());
    spec.seed     = planet.seed.value_or(spec.seed);
    spec.radius   = planet.radius.value_or(spec.radius / 1000.0) * 1000.0;
    spec.sea      = planet.sea.value_or(spec.sea);
    spec.relief   = planet.relief.value_or(spec.relief);
    spec.features = std::make_shared<const W::Features>(planet.features);
    sited         = bool(planet.features.site);
  }
  const auto       from = std::chrono::steady_clock::now();
  const W::World   w    = W::make(spec);
  std::printf("made in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count());
  // (latitude 100: where the figures stand)
  const glm::dvec3 d     = lat > 90.0 ? w.frame.site : W::directionOf(lat, lon);
  const glm::dvec3 north = glm::normalize(glm::dvec3(0.0, 1.0, 0.0) - d * d.y), east = glm::cross(d, north);
  const double     size = km * 1000.0, pixel = size / double(n);
  std::vector<W::Sample> samples(size_t(n) * size_t(n));
  fv::jobs::parallelFor(samples.size(), 256, [&](size_t begin, size_t end) {
    for(size_t k = begin; k < end; k++)
    {
      const double x = (double(k % size_t(n)) + 0.5) / n - 0.5, y = (double(k / size_t(n)) + 0.5) / n - 0.5;
      samples[k] = W::sample(w, glm::normalize(d + (east * x - north * y) * (size / w.spec.radius)), pixel);
    }
  });
  const glm::dvec3           light = glm::normalize(glm::dvec3(-1.0, 1.0, 1.1));  // east, north, up
  std::vector<unsigned char> image(samples.size() * 3);
  double                     lowest = 1e30, highest = -1e30;
  const auto                 at = [&](int i, int j) { return samples[size_t(std::clamp(j, 0, n - 1)) * n + std::clamp(i, 0, n - 1)].elevation; };
  for(int j = 0; j < n; j++)
    for(int i = 0; i < n; i++)
    {
      const W::Sample& s = samples[size_t(j) * n + i];
      lowest             = std::min(lowest, s.elevation);
      highest            = std::max(highest, s.elevation);
      const glm::dvec3 normal = glm::normalize(glm::dvec3(-(at(i + 1, j) - at(i - 1, j)) / (2.0 * pixel), (at(i, j + 1) - at(i, j - 1)) / (2.0 * pixel), 1.0));
      const double     lit    = 0.25 + 1.1 * std::max(glm::dot(normal, light), 0.0);
      glm::dvec3       c      = glm::dvec3(W::surfaceColour(w, s, float(std::sqrt(1.0 - normal.z * normal.z) / normal.z))) * lit * 2.5;
      if(s.elevation < 0.0 || s.water > s.elevation)
        c = glm::dvec3(0.02, 0.07, 0.16) * (s.elevation < 0.0 ? 1.0 : 2.0);
      for(int k = 0; k < 3; k++)
        image[(size_t(j) * n + i) * 3 + k] = (unsigned char)(255.0 * std::pow(std::clamp(c[k], 0.0, 1.0), 1.0 / 2.2));
    }
  std::printf("%d x %d, %.1f m a pixel, land from %.0f to %.0f m%s\n", n, n, pixel, lowest, highest, sited ? " (the file gives a site)" : "");
  FILE* out = std::fopen(file, "wb");
  ASSERT_NE(out, nullptr);
  std::fprintf(out, "P6\n%d %d\n255\n", n, n);
  std::fwrite(image.data(), 1, image.size(), out);
  std::fclose(out);
}

// Not a test: the ground and the water round a place of the default planet, from straight above, as
// an image (FV_WORLD_PROBE = "x y z size footprint file.ppm", the place in the scene). Grey: the
// ground's height. Blue: water over it, brighter the deeper. Red: water over ground more than 4 m
// under it that is neither in a lake's cells nor in a river's bed.
TEST(WorldMapBake, DISABLED_Probe)
{
  const char* spec = std::getenv("FV_WORLD_PROBE");
  if(!spec)
    return;
  double x = 0, y = 0, z = 0, size = 400, footprint = 2;
  char   file[512] = {};
  ASSERT_EQ(std::sscanf(spec, "%lf %lf %lf %lf %lf %511s", &x, &y, &z, &size, &footprint, file), 6);
  const W::World   w  = W::make({});
  const glm::dvec3 at(x, y, z);
  const glm::dvec3 up = glm::normalize(at - w.frame.centre);
  const glm::dvec3 a  = glm::normalize(glm::cross(up, glm::dvec3(0.3, 0.1, 0.9))), b = glm::cross(up, a);
  const int        n  = 512;
  std::vector<unsigned char> image(size_t(n) * n * 3);
  double lowest = 1e30, highest = -1e30, deepest = 0.0;
  int    stray = 0;
  std::vector<W::Sample> samples(size_t(n) * n);
  for(int j = 0; j < n; j++)
    for(int i = 0; i < n; i++)
    {
      const glm::dvec3 p = at + a * ((double(i) / n - 0.5) * size) + b * ((double(j) / n - 0.5) * size);
      samples[size_t(j) * n + i] = W::sample(w, W::toPlanet(w, p), footprint);
      lowest  = std::min(lowest, samples[size_t(j) * n + i].elevation);
      highest = std::max(highest, samples[size_t(j) * n + i].elevation);
    }
  for(size_t k = 0; k < samples.size(); k++)
  {
    const W::Sample& s    = samples[k];
    const double     grey = (s.elevation - lowest) / std::max(highest - lowest, 1e-6);
    const double     deep = std::max(s.water, 0.0) - s.elevation;
    unsigned char*   px   = &image[k * 3];
    px[0] = px[1] = px[2] = (unsigned char)(40.0 + 200.0 * grey);
    if(deep > 0.0)
    {
      deepest = std::max(deepest, deep);
      const bool held = s.inland.lake >= 0.5 || (s.inland.river && s.inland.riverAway < s.inland.riverHalf) || s.elevation < 0.0;
      if(deep > 4.0 && !held)
      {
        px[0] = 255;
        px[1] = px[2] = 0;
        stray++;
      }
      else
      {
        px[0] = px[1] = 30;
        px[2]         = (unsigned char)(120.0 + std::min(deep, 13.5) * 10.0);
      }
    }
  }
  FILE* f = std::fopen(file, "wb");
  ASSERT_TRUE(f);
  std::fprintf(f, "P6\n%d %d\n255\n", n, n);
  std::fwrite(image.data(), 1, image.size(), f);
  std::fclose(f);
  std::printf("probe: ground %.1f to %.1f m, water up to %.1f m deep, %d stray of %d\n", lowest, highest, deepest, stray, n * n);
}

// Not a test: what a bake at the planet's own size takes and makes, for whoever changes its numbers.
//   --gtest_also_run_disabled_tests --gtest_filter=*Report*
// A dev tool, not a test: where the ground steps within a metre round a place, and the stage that
// makes the step (FV_WORLD_STEPS = "x y z size", the place in the scene). The steepest few, with the
// ground's height after each stage on either side.
TEST(WorldMapBake, DISABLED_Steps)
{
  const char* spec = std::getenv("FV_WORLD_STEPS");
  if(!spec)
    return;
  double x = 0, y = 0, z = 0, size = 200;
  ASSERT_EQ(std::sscanf(spec, "%lf %lf %lf %lf", &x, &y, &z, &size), 4);
  const W::World   w  = W::make({});
  const glm::dvec3 at(x, y, z);
  const glm::dvec3 up = glm::normalize(at - w.frame.centre);
  const glm::dvec3 a  = glm::normalize(glm::cross(up, glm::dvec3(0.3, 0.1, 0.9))), b = glm::cross(up, a);
  const int        n  = int(size);
  const auto       dirOf = [&](int i, int j) { return W::toPlanet(w, at + a * (double(i) - 0.5 * n) + b * (double(j) - 0.5 * n)); };
  struct Step
  {
    double rise;
    int    i, j, di, dj;
  };
  std::vector<Step>   steps;
  std::vector<double> heights(size_t(n) * size_t(n));
  for(int j = 0; j < n; j++)
    for(int i = 0; i < n; i++)
      heights[size_t(j) * n + i] = W::sample(w, dirOf(i, j), 1.0).elevation;
  for(int j = 0; j + 1 < n; j++)
    for(int i = 0; i + 1 < n; i++)
    {
      steps.push_back({std::abs(heights[size_t(j) * n + i + 1] - heights[size_t(j) * n + i]), i, j, 1, 0});
      steps.push_back({std::abs(heights[size_t(j + 1) * n + i] - heights[size_t(j) * n + i]), i, j, 0, 1});
    }
  std::sort(steps.begin(), steps.end(), [](const Step& l, const Step& r) { return l.rise > r.rise; });
  for(size_t k = 0; k < 6 && k < steps.size(); k += 1)
  {
    const Step& st = steps[k * 37 % std::min<size_t>(steps.size(), 200)];
    std::printf("step of %.2f m at %d %d:\n", st.rise, st.i, st.j);
    W::Sample      one, two;
    const W::Point p1{dirOf(st.i, st.j), 1.0}, p2{dirOf(st.i + st.di, st.j + st.dj), 1.0};
    for(const W::Stage& stage : w.stages)
    {
      stage.apply(w, p1, one);
      stage.apply(w, p2, two);
      std::printf("  %-10s %9.3f %9.3f  (%+.3f)\n", std::string(stage.name).c_str(), one.elevation, two.elevation, two.elevation - one.elevation);
    }
    std::printf("  tiers %.3f %.3f cliff %.2f %.2f steepened %.2f %.2f river %d %d lake %.2f %.2f water %.1f %.1f\n", one.tiers, two.tiers, one.cliff, two.cliff,
                one.steepened, two.steepened, int(one.inland.river), int(two.inland.river), one.inland.lake, two.inland.lake, one.water, two.water);
  }
}

TEST(WorldMapBake, DISABLED_Report)
{
  for(const double radius : {5000.0, 20000.0, 60000.0})
  {
    const auto     t0 = std::chrono::steady_clock::now();
    const W::World w  = W::make({.radius = radius});
    const double   s  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const W::BaseMap& m = *w.base;
    const W::World bare = W::make({.radius = radius}, {.base = false});
    std::vector<float> before, after;
    uint32_t           lake = 0;
    for(uint32_t c = 0; c < m.grid.count(); c += 5)
    {
      lake += m.kind[c] == W::eMapLake;
      if(m.kind[c] == W::eMapSea)
        continue;
      before.push_back(float(W::sample(bare, m.grid.centre(c), m.cell).elevation));
      after.push_back(m.height[c]);
    }
    std::sort(before.begin(), before.end());
    std::sort(after.begin(), after.end());
    const auto at = [](const std::vector<float>& v, double share) { return v[size_t(share * double(v.size() - 1))]; };
    std::map<int, int> biomes, bareBiomes;
    int                land = 0, bareLand = 0, rain[5] = {};
    for(uint32_t i = 0; i < 20000; i++)
    {
      const W::Sample smp = W::sample(w, spread(i, 20000), 100.0);
      if(smp.elevation > 0.0)
      {
        biomes[smp.biome]++;
        land++;
        if(smp.temperature > 5.0F)
          rain[std::min(int(W::mapRain(m, spread(i, 20000)) * 5.0F), 4)]++;
      }
      const W::Sample was = W::sample(bare, spread(i, 20000), 100.0);
      if(was.elevation > 0.0)
      {
        bareBiomes[was.biome]++;
        bareLand++;
      }
    }
    std::printf("radius %.0f km: made in %.2f s; %u cells a face (%.0f m), %u lakes (%u cells of a fifth), %zu river pieces\n", radius / 1000.0, s,
                m.grid.cells(), m.cell, m.lakes, lake, m.pieces.size());
    std::printf("  land height, median / 90%% / 99%% / top: before %.0f %.0f %.0f %.0f, after %.0f %.0f %.0f %.0f\n", at(before, 0.5), at(before, 0.9),
                at(before, 0.99), at(before, 1.0), at(after, 0.5), at(after, 0.9), at(after, 0.99), at(after, 1.0));
    std::printf("  site %.1f m up, latitude %.0f; biomes:", w.frame.siteElevation, glm::degrees(std::asin(w.frame.site.y)));
    for(const auto& [b, count] : biomes)
      std::printf(" %s %.0f%%", std::string(w.biomes[size_t(b)].name).c_str(), 100.0 * count / land);
    std::printf("\n  with no map:");
    for(const auto& [b, count] : bareBiomes)
      std::printf(" %s %.0f%%", std::string(w.biomes[size_t(b)].name).c_str(), 100.0 * count / bareLand);
    std::printf("\n  rain over the land that is not cold, by fifths: %d %d %d %d %d (of %d)\n", rain[0], rain[1], rain[2], rain[3], rain[4], land);
    // places to look at, in the scene (the figures at its origin): the river and the lake nearest
    // them, and the widest river
    const auto place = [&](const char* what, const glm::dvec3& d, double level, const glm::dvec3& toward, double size) {
      const glm::dvec3 p = W::toScene(w, d, level), q = W::toScene(w, toward, level);
      std::printf("  sight %s at %.0f %.0f %.0f toward %.0f %.0f %.0f size %.1f\n", what, p.x, p.y, p.z, q.x, q.y, q.z, size);
    };
    const W::RiverPiece *nearest = nullptr, *widest = nullptr;
    for(const W::RiverPiece& p : m.pieces)
    {
      if(p.half[1] >= 5.0F && (!nearest || glm::distance(p.b, w.frame.site) < glm::distance(nearest->b, w.frame.site)))
        nearest = &p;
      if(!widest || p.half[1] > widest->half[1])
        widest = &p;
    }
    if(nearest)
      place("river", nearest->b, double(nearest->level[1]), nearest->c, double(nearest->half[1]));
    if(widest)
      place("widest", widest->b, double(widest->level[1]), widest->c, double(widest->half[1]));
    // the widest river and the highest ground on the side the look scripts' sun lights (azimuth 120, 40 up)
    const glm::dvec3 sun(std::cos(glm::radians(40.0)) * std::sin(glm::radians(120.0)), std::sin(glm::radians(40.0)),
                         std::cos(glm::radians(40.0)) * std::cos(glm::radians(120.0)));
    const auto lit = [&](const glm::dvec3& d) { return glm::dot(w.frame.turn * d, sun) > 0.5; };
    const W::RiverPiece* day = nullptr;
    for(const W::RiverPiece& p : m.pieces)
      if(lit(p.b) && (!day || p.half[1] > day->half[1]))
        day = &p;
    if(day)
      place("widest-lit", day->b, double(day->level[1]), day->c, double(day->half[1]));
    uint32_t peak = W::CubeGrid::kNone;
    for(uint32_t c = 0; c < m.grid.count(); c++)
      if(m.kind[c] == W::eMapLand && lit(m.grid.centre(c)) && (peak == W::CubeGrid::kNone || m.height[c] > m.height[peak]))
        peak = c;
    if(peak != W::CubeGrid::kNone)
      place("peak", m.grid.centre(peak), double(m.height[peak]), w.frame.site, double(m.height[peak]));
    // the cliff's face nearest the figures, and the nearest bluff by a shore
    {
      glm::dvec3 face(0.0), bluff(0.0);
      double     faceAway = 1e30, bluffAway = 1e30, faceUp = 0.0, bluffUp = 0.0;
      for(uint32_t i = 0; i < 600000; i++)
      {
        const glm::dvec3 d    = spread(i, 600000);
        const double     away = glm::distance(d, w.frame.site);
        if(away >= faceAway && away >= bluffAway)
          continue;
        const W::Sample smp = W::sample(w, d, 2.0);
        if(smp.inland.any())
          continue;
        if(smp.cliff > 0.8F && smp.elevation > 5.0 && away < faceAway)
        {
          faceAway = away;
          face     = d;
          faceUp   = smp.elevation;
        }
        if(smp.steepened > 4.0F && smp.elevation > 4.0 && away < bluffAway)
        {
          bluffAway = away;
          bluff     = d;
          bluffUp   = smp.elevation;
        }
      }
      if(faceAway < 1e30)
        place("cliff", face, faceUp, w.frame.site, faceUp);
      if(bluffAway < 1e30)
        place("bluff", bluff, bluffUp, w.frame.site, bluffUp);
    }
    // each biome's place nearest the figures (all of it that biome, off the water and the heights)
    for(size_t biome = 0; biome < w.biomes.size(); biome++)
    {
      glm::dvec3 best(0.0);
      double     nearestAway = 1e30, up = 0.0;
      for(uint32_t i = 0; i < 60000; i++)
      {
        const glm::dvec3 d = spread(i, 60000);
        if(glm::distance(d, w.frame.site) >= nearestAway)
          continue;
        const W::Sample smp = W::sample(w, d, 30.0);
        if(smp.elevation < 8.0 || size_t(smp.biome) != biome || smp.share[biome] < 0.97F || smp.shore < 150.0F || smp.rock > 0.1F)
          continue;
        nearestAway = glm::distance(d, w.frame.site);
        best        = d;
        up          = smp.elevation;
      }
      if(nearestAway < 1e30)
        place(std::string(w.biomes[biome].name).c_str(), best, up, w.frame.site, nearestAway * w.spec.radius);
    }
    // the sea's shallows nearest the figures, some 4 m deep: where it is warm (corals) and where it is cool (kelp)
    for(const bool warm : {true, false})
    {
      glm::dvec3 best(0.0);
      double     nearestAway = 1e30;
      for(uint32_t i = 0; i < 200000; i++)
      {
        const glm::dvec3 d = spread(i, 200000);
        if(glm::distance(d, w.frame.site) >= nearestAway || (W::temperatureAt(w, d, 0.0) > 18.0) != warm)
          continue;
        const W::Sample smp = W::sample(w, d, 30.0);
        if(smp.elevation > (warm ? -3.0 : -7.0) || smp.elevation < (warm ? -6.0 : -12.0) || smp.inland.any())
          continue;
        nearestAway = glm::distance(d, w.frame.site);
        best        = d;
      }
      if(nearestAway < 1e30)
        place(warm ? "sea-warm" : "sea-cool", best, 0.0, w.frame.site, nearestAway * w.spec.radius);
    }
    uint32_t lakeAt = W::CubeGrid::kNone;
    for(uint32_t c = 0; c < m.grid.count(); c++)
      if(m.kind[c] == W::eMapLake && (lakeAt == W::CubeGrid::kNone || glm::distance(m.grid.centre(c), w.frame.site) < glm::distance(m.grid.centre(lakeAt), w.frame.site)))
        lakeAt = c;
    if(lakeAt != W::CubeGrid::kNone)
    {
      const float level = m.lake[m.grid.padded(uint8_t(lakeAt / (m.grid.cells() * m.grid.cells())), int(lakeAt % m.grid.cells()), int((lakeAt / m.grid.cells()) % m.grid.cells()))];
      place("lake", m.grid.centre(lakeAt), double(level), w.frame.site, double(level - m.height[lakeAt]));
    }
  }
}
