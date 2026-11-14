// The planet (src/fv_world.hpp): the same for a seed every time, as much sea as asked for, level where
// the figures stand, quads that cover the sphere once from any height, and meshes whose edges meet.

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <gtest/gtest.h>

#include "../src/fv_world.hpp"

namespace {

namespace W = fv::world;

// `count` directions spread over the sphere
glm::dvec3 spread(uint32_t i, uint32_t count)
{
  const double y   = 1.0 - 2.0 * (double(i) + 0.5) / double(count);
  const double r   = std::sqrt(std::max(1.0 - y * y, 0.0));
  const double phi = double(i) * 2.399963229728653;
  return {r * std::cos(phi), y, r * std::sin(phi)};
}

const W::World& planet()
{
  static const W::World w = W::make({});
  return w;
}

// A scene point `height` over the ground, `along` metres from the origin toward +x
glm::dvec3 eyeAt(const W::World& w, double height, double along = 0.0)
{
  const glm::dvec3 d = W::toPlanet(w, glm::dvec3(along, 0.0, 0.0));
  return W::toScene(w, d, std::max(W::sample(w, d, 1.0).elevation, 0.0) + height);
}

}  // namespace

TEST(WorldCube, AFacesPointsGoOutAndComeBack)
{
  for(uint8_t face = 0; face < 6; face++)
    for(double u = -0.95; u < 1.0; u += 0.19)
      for(double v = -0.95; v < 1.0; v += 0.19)
      {
        const glm::dvec3   d = W::direction(face, u, v);
        const W::FacePoint p = W::locate(d);
        EXPECT_NEAR(glm::length(d), 1.0, 1e-12);
        EXPECT_EQ(p.face, face);
        EXPECT_NEAR(p.u, u, 1e-12);
        EXPECT_NEAR(p.v, v, 1e-12);
      }
}

TEST(WorldCube, TwoFacesShareTheirEdgesPoints)
{
  // the points along a face's edge are the neighbouring face's along its own: a quad's edge vertices
  // are its neighbour's across a cube edge
  for(uint8_t face = 0; face < 6; face++)
    for(double v = -1.0; v <= 1.0; v += 0.125)
    {
      const glm::dvec3   d = W::direction(face, 1.0, v);
      const W::FacePoint p = W::locate(W::direction(face, 1.0 + 1e-9, v));
      ASSERT_NE(p.face, face);
      // on the other face it is on an edge too, at a vertex of the same spacing
      const double edge = std::max(std::abs(p.u), std::abs(p.v)), along = std::min(std::abs(p.u), std::abs(p.v));
      EXPECT_NEAR(edge, 1.0, 1e-8);
      EXPECT_NEAR(along * 8.0, std::round(along * 8.0), 1e-7);
      EXPECT_NEAR(glm::length(W::direction(p.face, std::round(p.u * 8.0) / 8.0, std::round(p.v * 8.0) / 8.0) - d), 0.0, 1e-9);
    }
}

TEST(WorldFields, ASeedsPlanetIsTheSameEveryTime)
{
  const W::World a = W::make({.seed = 7}), b = W::make({.seed = 7}), c = W::make({.seed = 8});
  int            differ = 0;
  for(uint32_t i = 0; i < 200; i++)
  {
    const glm::dvec3 d = spread(i, 200);
    EXPECT_EQ(W::sample(a, d, 10.0).elevation, W::sample(b, d, 10.0).elevation);
    differ += W::sample(a, d, 10.0).elevation != W::sample(c, d, 10.0).elevation;
  }
  EXPECT_GT(differ, 190);
  EXPECT_EQ(a.frame.site, b.frame.site);
}

TEST(WorldFields, TheSeaCoversTheShareAskedFor)
{
  for(const double sea : {0.2, 0.55, 0.8})
  {
    const W::World w     = W::make({.sea = sea});
    int            under = 0;
    const int      n     = 4000;
    for(int i = 0; i < n; i++)
      under += W::sample(w, spread(uint32_t(i), uint32_t(n)), 50.0).elevation < 0.0;
    EXPECT_NEAR(double(under) / n, sea, 0.05) << sea;
  }
  // and none asked for is none
  const W::World dry = W::make({.sea = 0.0});
  for(uint32_t i = 0; i < 2000; i++)
    EXPECT_GT(W::sample(dry, spread(i, 2000), 200.0).elevation, 0.0);
}

TEST(WorldFields, FlatWithNoRelief)
{
  const W::World w = W::make({.relief = 0.0});
  for(uint32_t i = 0; i < 500; i++)
  {
    const glm::dvec3 d = spread(i, 500);
    glm::dvec3       side = glm::normalize(glm::cross(d, glm::dvec3(0.3, 0.5, 0.8)));
    // no bump a few metres across: the ground 4 m to either side is on one slope with it, the
    // continent's own
    const double before = W::sample(w, glm::normalize(d - side * (4.0 / w.spec.radius)), 1.0).elevation;
    const double after  = W::sample(w, glm::normalize(d + side * (4.0 / w.spec.radius)), 1.0).elevation;
    EXPECT_NEAR(0.5 * (before + after), W::sample(w, d, 1.0).elevation, 0.01);
  }
}

TEST(WorldFields, ItIsColdAtThePolesAndOnTheHeights)
{
  const W::World& w = planet();
  double          equator = 0.0, pole = 0.0;
  int             ne = 0, np = 0;
  const W::Sample *lowest = nullptr, *highest = nullptr;
  std::vector<W::Sample> all;
  for(uint32_t i = 0; i < 3000; i++)
    all.push_back(W::sample(w, spread(i, 3000), 100.0));
  for(uint32_t i = 0; i < 3000; i++)
  {
    const glm::dvec3 d = spread(i, 3000);
    const W::Sample& s = all[i];
    if(s.elevation < 0.0)
      continue;
    if(std::abs(d.y) < 0.2)
    {
      equator += s.temperature;
      ne++;
      // of the land near the equator, the lowest and the highest
      if(!lowest || s.elevation < lowest->elevation)
        lowest = &s;
      if(!highest || s.elevation > highest->elevation)
        highest = &s;
    }
    else if(std::abs(d.y) > 0.9)
    {
      pole += s.temperature;
      np++;
    }
  }
  ASSERT_GT(ne, 20);
  ASSERT_GT(np, 5);
  EXPECT_GT(equator / ne, pole / np + 20.0);
  ASSERT_TRUE(lowest && highest);
  EXPECT_GT(highest->elevation, lowest->elevation + 100.0);
  EXPECT_GT(lowest->temperature, highest->temperature);
}

TEST(WorldFields, EveryBiomeIsSomewhereAndColoursAreReflectances)
{
  const W::World& w = planet();
  std::set<int>   seen;
  for(uint32_t i = 0; i < 6000; i++)
  {
    const W::Sample s = W::sample(w, spread(i, 6000), 100.0);
    for(int c = 0; c < 3; c++)
    {
      EXPECT_GT(s.colour[c], 0.0F);
      EXPECT_LT(s.colour[c], 0.95F);
    }
    if(s.elevation > 0.0)
      seen.insert(s.biome);
  }
  EXPECT_EQ(seen.size(), w.biomes.size());
  for(int c = 0; c < 3; c++)
  {
    EXPECT_GT(w.albedo[c], 0.01F);
    EXPECT_LT(w.albedo[c], 0.5F);
  }
}

TEST(WorldFields, NothingFinerThanWhatDrawsIt)
{
  // along a line of steps a sixteenth of a footprint long, a sample for that footprint keeps one slope
  // from step to step, where the ground itself (a metre's footprint) has bumps a few metres across
  const W::World& w = planet();
  const double    footprint = 64.0;
  double          coarse = 0.0, fine = 0.0;
  for(uint32_t i = 0; i < 200; i++)
  {
    const glm::dvec3 d = spread(i, 200);
    const glm::dvec3 side = glm::normalize(glm::cross(d, glm::dvec3(0.3, 0.5, 0.8)));
    // on land, and off the mountains: a ridge's crest is a crease at any footprint
    const W::Sample from = W::sample(w, d, footprint), to = W::sample(w, glm::normalize(d + side * (3.0 * footprint / w.spec.radius)), footprint);
    if(from.elevation < 20.0 || from.mountain > 0.0F || to.mountain > 0.0F)
      continue;
    double           lastCoarse = 0.0, lastFine = 0.0, slopeCoarse = 0.0, slopeFine = 0.0;
    for(int k = 0; k < 48; k++)
    {
      const glm::dvec3 at = glm::normalize(d + side * (double(k) * footprint / 16.0 / w.spec.radius));
      const double     c = W::sample(w, at, footprint).elevation, f = W::sample(w, at, 1.0).elevation;
      if(k > 1)
      {
        // how much the slope changes from one step to the next: bumps under the footprint
        coarse += std::abs((c - lastCoarse) - slopeCoarse);
        fine += std::abs((f - lastFine) - slopeFine);
      }
      slopeCoarse = c - lastCoarse;
      slopeFine   = f - lastFine;
      lastCoarse  = c;
      lastFine    = f;
    }
  }
  ASSERT_GT(fine, 0.0);
  EXPECT_LT(coarse, 0.15 * fine);
}

TEST(WorldSite, TheFiguresStandOnLevelLandAtTheOrigin)
{
  for(const int seed : {1, 2, 3, 4, 5})
    for(const double radius : {5000.0, 20000.0, 80000.0})
    {
      const W::World w = W::make({.seed = seed, .radius = radius});
      EXPECT_GT(w.frame.siteElevation, 1.0) << seed << " " << radius;
      // the site is the origin, +y up
      const glm::dvec3 origin = W::toScene(w, w.frame.site, w.frame.siteElevation);
      EXPECT_NEAR(glm::length(origin), 0.0, 1e-6);
      const glm::dvec3 above = W::toScene(w, w.frame.site, w.frame.siteElevation + 10.0);
      EXPECT_NEAR(above.y, 10.0, 1e-6);
      // level within the clearing, and the ground round it within a few metres of it
      for(double a = 0.0; a < 6.28; a += 0.4)
      {
        const glm::dvec3 in = W::toPlanet(w, glm::dvec3(std::cos(a), 0.0, std::sin(a)) * (W::kClearing - 0.5));
        EXPECT_NEAR(W::sample(w, in, 1.0).elevation, w.frame.siteElevation, 1e-6);
        const glm::dvec3 out = W::toPlanet(w, glm::dvec3(std::cos(a), 0.0, std::sin(a)) * 60.0);
        EXPECT_NEAR(W::sample(w, out, 1.0).elevation, w.frame.siteElevation, 8.0) << seed << " " << radius;
        EXPECT_GT(W::sample(w, out, 1.0).elevation, 0.0);
      }
      EXPECT_NEAR(W::heightOver(w, glm::dvec3(0.0, 1.7, 0.0)), 1.7, 1e-3);
    }
}

TEST(WorldSite, APlanetNearlyAllSeaStillHasOne)
{
  const W::World w = W::make({.sea = 0.9});
  EXPECT_GT(w.frame.siteElevation, 0.0);
}

TEST(WorldSelect, TheQuadsCoverTheSphereOnceFromAnyHeight)
{
  const W::World& w = planet();
  for(const double height : {1.7, 30.0, 400.0, 5000.0, 60000.0, 400000.0})
  {
    const W::Selection s = W::select(w, eyeAt(w, height));
    // no leaf is another's ancestor, and their areas add up to the six faces
    double area = 0.0;
    for(const W::Quad& q : s.leaves)
      area += 1.0 / double(uint64_t(1) << (2 * q.level));
    EXPECT_NEAR(area, 6.0, 1e-9) << height;
    for(uint32_t i = 0; i < 500; i++)
    {
      const glm::dvec3 d = spread(i, 500);
      int              n = 0;
      for(const W::Quad& q : s.leaves)
        n += W::contains(q, d);
      EXPECT_EQ(n, 1) << height;
    }
    EXPECT_LT(s.leaves.size(), 700U) << height;
  }
}

TEST(WorldSelect, FineUnderfootAndCoarseFromOrbit)
{
  const W::World&      w      = planet();
  const W::Selection   ground = W::select(w, eyeAt(w, 1.7));
  const uint8_t        finest = W::finestLevel(w);
  EXPECT_LE(W::width(w, {0, finest, 0, 0}) / W::kGroundCells, 1.0);
  // the quad under the eye is of the finest level
  const glm::dvec3 under = W::toPlanet(w, eyeAt(w, 1.7));
  bool             found = false;
  for(const W::Quad& q : ground.leaves)
    if(W::contains(q, under))
    {
      EXPECT_EQ(q.level, finest);
      found = true;
    }
  EXPECT_TRUE(found);
  // from ten radii off, the whole planet is a few dozen quads
  const W::Selection orbit = W::select(w, eyeAt(w, 10.0 * w.spec.radius));
  EXPECT_LE(orbit.leaves.size(), 96U);
  for(const W::Quad& q : orbit.leaves)
    EXPECT_LE(q.level, 2);
}

TEST(WorldSelect, AnEyeThatHoversChangesNothing)
{
  const W::World& w = planet();
  // out and back across a split's distance: with the last selection handed back, the quads stay
  W::Selection s = W::select(w, eyeAt(w, 100.0));
  const W::Selection first = s;
  int                changes = 0;
  for(int k = 0; k < 40; k++)
  {
    const W::Selection next = W::select(w, eyeAt(w, 100.0 + (k % 2 ? 6.0 : 0.0)), &s);
    changes += next.leaves != s.leaves;
    s = next;
  }
  EXPECT_LE(changes, 1);
  // without it, the same two heights give two selections or one: either way the first again
  EXPECT_EQ(W::select(w, eyeAt(w, 100.0)).leaves, first.leaves);
}

TEST(WorldMesh, AQuadsTrianglesAreItsGround)
{
  const W::World&    w = planet();
  const W::Selection s = W::select(w, eyeAt(w, 50.0));
  // a fine quad near the eye and a coarse one far off
  for(const W::Quad& q : {s.leaves.front(), s.leaves[s.leaves.size() / 2], s.leaves.back()})
  {
    const W::ChunkMesh m = W::groundMesh(w, q);
    const uint32_t     side = W::kGroundCells + 1;
    ASSERT_EQ(m.surface, side * side);
    ASSERT_EQ(m.positions.size(), m.normals.size());
    ASSERT_EQ(m.positions.size(), m.colors.size());
    EXPECT_EQ(m.positions.size(), size_t(m.surface) + 4 * side);
    EXPECT_EQ(m.indices.size() % 3, 0U);
    for(const uint32_t i : m.indices)
      ASSERT_LT(i, m.positions.size());
    for(uint32_t v = 0; v < m.surface; v++)
    {
      const glm::dvec3 p  = m.centre + glm::dvec3(m.positions[v]);
      const glm::dvec3 up = glm::normalize(p - w.frame.centre);
      EXPECT_NEAR(glm::length(m.normals[v]), 1.0F, 1e-4F);
      EXPECT_GT(glm::dot(glm::dvec3(m.normals[v]), up), 0.2);
      // on the ground: its height over the sea is the sample's there (to a float's worth so far out)
      double           r = 0.0;
      const glm::dvec3 d = W::toPlanet(w, p, &r);
      const double     e = W::sample(w, d, W::width(w, q) / W::kGroundCells).elevation;
      EXPECT_NEAR(r - w.spec.radius, e, 0.05 + 1e-6 * W::width(w, q));
      EXPECT_GE(e, m.lowest - 0.05);
      EXPECT_LE(e, m.highest + 0.05);
      for(int c = 0; c < 4; c++)
      {
        EXPECT_GE(m.colors[v][c], 0.0F);
        EXPECT_LE(m.colors[v][c], 1.0F);
      }
    }
    // each triangle of the surface faces out of the planet
    for(size_t t = 0; t < size_t(W::kGroundCells) * W::kGroundCells * 6; t += 3)
    {
      const glm::vec3 a = m.positions[m.indices[t]], b = m.positions[m.indices[t + 1]], c = m.positions[m.indices[t + 2]];
      const glm::dvec3 up = glm::normalize(m.centre + glm::dvec3(a) - w.frame.centre);
      EXPECT_GT(glm::dot(glm::dvec3(glm::cross(b - a, c - a)), up), 0.0);
    }
    // the skirts hang under their edge
    for(size_t v = m.surface; v < m.positions.size(); v++)
      EXPECT_LT(glm::length(m.centre + glm::dvec3(m.positions[v]) - w.frame.centre),
                w.spec.radius + m.highest + 0.01);
  }
}

TEST(WorldMesh, NeighboursOfALevelMeetAlongTheirEdge)
{
  const W::World& w = planet();
  // across a face, and across a cube's edge (the quad at a face's last column and the next face's)
  const uint8_t  level = 6;
  const uint32_t last  = (1U << level) - 1;
  const W::Quad  a{0, level, last - 1, 20}, b{0, level, last, 20};
  const auto     edgeOf = [&](const W::ChunkMesh& m, bool right) {
    std::vector<glm::dvec3> out;
    const uint32_t          side = W::kGroundCells + 1;
    for(uint32_t j = 0; j < side; j++)
      out.push_back(m.centre + glm::dvec3(m.positions[j * side + (right ? side - 1 : 0)]));
    return out;
  };
  const W::ChunkMesh ma = W::groundMesh(w, a), mb = W::groundMesh(w, b);
  const auto         ea = edgeOf(ma, true), eb = edgeOf(mb, false);
  for(size_t i = 0; i < ea.size(); i++)
    EXPECT_NEAR(glm::length(ea[i] - eb[i]), 0.0, 2e-3);
  // across the cube's edge: every vertex of b's far edge is one of the mesh on the face beyond
  const auto         far = edgeOf(mb, true);
  const W::FacePoint over = W::locate(W::direction(0, 1.0 + 1e-6, -1.0 + 2.0 * (double(b.y) + 0.5) / double(1U << level)));
  ASSERT_NE(over.face, 0);
  W::Quad beyond{over.face, level, 0, 0};
  beyond.x = uint32_t(std::clamp((over.u + 1.0) * 0.5 * double(1U << level), 0.0, double(last)));
  beyond.y = uint32_t(std::clamp((over.v + 1.0) * 0.5 * double(1U << level), 0.0, double(last)));
  const W::ChunkMesh mc = W::groundMesh(w, beyond);
  for(const glm::dvec3& p : far)
  {
    double nearest = 1e30;
    for(uint32_t v = 0; v < mc.surface; v++)
      nearest = std::min(nearest, glm::length(mc.centre + glm::dvec3(mc.positions[v]) - p));
    EXPECT_LT(nearest, 5e-3);
  }
}

TEST(WorldMesh, TheSeaIsTheSphereAtItsLevel)
{
  const W::World&    w = planet();
  const W::ChunkMesh m = W::seaMesh(w, {4, 3, 2, 5});
  const uint32_t     side = W::kSeaCells + 1;
  ASSERT_EQ(m.surface, side * side);
  for(uint32_t v = 0; v < m.surface; v++)
  {
    const glm::dvec3 out = m.centre + glm::dvec3(m.positions[v]) - w.frame.centre;
    EXPECT_NEAR(glm::length(out), w.spec.radius, 0.02);
    EXPECT_NEAR(glm::dot(glm::normalize(out), glm::dvec3(m.normals[v])), 1.0, 1e-5);
  }
  for(size_t v = m.surface; v < m.positions.size(); v++)
    EXPECT_LT(glm::length(m.centre + glm::dvec3(m.positions[v]) - w.frame.centre), w.spec.radius);
}
