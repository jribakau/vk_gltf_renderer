// dForce: the settle drapes cloth over a collider and keeps it outside, holds cloth together under
// its own weight, turns colliders along their bones, the drape is the cage delta whose skinning
// gives the simulated shape, and the cloth settings are read from the scene's modifiers.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

#include <figure/dforce/dforce.hpp>
#include <figure/dforce/skin.hpp>
#include <figure/dforce/solver.hpp>
#include <figure/dforce/strands.hpp>
#include <figure/dson/dforce.hpp>
#include <figure/figure.hpp>
#include <figure/morph/morph_set.hpp>

namespace {

using namespace fv;

// An n x n grid of quads in the xz plane at height y, from -n/2 to n/2; faces up
struct Grid
{
  std::vector<float>    pos;
  std::vector<uint32_t> tris;
  std::vector<int32_t>  polyStart{0}, polyVerts;
};

Grid grid(int n, float y)
{
  Grid g;
  for(int z = 0; z <= n; z++)
    for(int x = 0; x <= n; x++)
      g.pos.insert(g.pos.end(), {float(x) - n / 2.f, y, float(z) - n / 2.f});
  const auto at = [&](int x, int z) { return uint32_t(z * (n + 1) + x); };
  for(int z = 0; z < n; z++)
    for(int x = 0; x < n; x++)
    {
      g.tris.insert(g.tris.end(), {at(x, z), at(x, z + 1), at(x + 1, z), at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)});
      g.polyVerts.insert(g.polyVerts.end(), {int32_t(at(x, z)), int32_t(at(x, z + 1)), int32_t(at(x + 1, z + 1)), int32_t(at(x + 1, z))});
      g.polyStart.push_back(int32_t(g.polyVerts.size()));
    }
  return g;
}

dforce::SettleMesh cloth_(const Grid& g);
dforce::SettleMesh cloth(const Grid& g) { return cloth_(g); }
dforce::SettleMesh cloth_(const Grid& g)
{
  dforce::SettleMesh m;
  const size_t       n = g.pos.size() / 3;
  m.rest = m.posed = g.pos;
  m.tris           = g.tris;
  m.polyStart      = g.polyStart;
  m.polyVerts      = g.polyVerts;
  m.kind           = dforce::SettleMesh::Kind::eDynamic;
  m.strength.assign(n, 1.f);
  m.stretch.assign(n, 0.8f);
  m.shear.assign(n, 0.2f);
  m.bend.assign(n, 0.5f);
  m.buckling.assign(n, 0.05f);
  m.bucklingRatio.assign(n, 0.7f);
  m.density.assign(n, 180.f);
  m.contraction.assign(n, 1.f);
  m.damping.assign(n, 0.1f);
  m.friction.assign(n, 0.4f);
  m.offset.assign(n, 0.2f);
  m.layer.assign(n, 0);
  m.selfCollide.assign(n, 1);
  m.collide.assign(n, 1);
  m.visible.assign(n, 1);
  m.gravity = {0, -980, 0};
  return m;
}

}  // namespace

TEST(Dforce, clothFallsOntoAColliderAndStaysOutsideIt)
{
  const Grid body = grid(8, 0);
  const Grid veil = grid(4, 3);
  const auto out  = dforce::settle({dforce::SettleMesh::collider(body.pos, body.pos, body.tris, dforce::kBodyLayer, 0.2f, 0.4f), cloth(veil)});
  ASSERT_EQ(out.size(), 2u);
  // the collider is where it was; the cloth came down to rest above it
  EXPECT_EQ(out[0], body.pos);
  for(size_t v = 0; v < out[1].size() / 3; v++)
  {
    const float y = out[1][v * 3 + 1];
    EXPECT_LT(y, 1.0f) << v;
    EXPECT_GT(y, 0.0f) << v;
  }
}

TEST(Dforce, clothLostInsideAColliderIsBroughtBackOut)
{
  // a veil that starts 2.5 cm under the body, deeper than the start separates: every frame puts
  // what lies behind the nearest body face back outside it, so it ends resting on the body
  const Grid body = grid(8, 0), veil = grid(4, -2.5f);
  const auto collider = dforce::SettleMesh::collider(body.pos, body.pos, body.tris, dforce::kBodyLayer, 0.2f, 0.4f);
  const auto out      = dforce::settle({collider, cloth(veil)});
  for(size_t v = 0; v < out[1].size() / 3; v++)
  {
    EXPECT_GT(out[1][v * 3 + 1], 0.0f) << v;
    EXPECT_LT(out[1][v * 3 + 1], 1.0f) << v;
  }
  // without it, the cloth stays under and falls
  dforce::SettleOptions opts;
  opts.recoverLost = false;
  const auto under  = dforce::settle({collider, cloth(veil)}, opts);
  for(size_t v = 0; v < under[1].size() / 3; v++)
    EXPECT_LT(under[1][v * 3 + 1], -2.5f) << v;
}

TEST(Dforce, restingClothGripsTheFaceItLiesOn)
{
  // a sheet on a slope of 20 degrees: friction 0.5 holds it, none lets it slide (tan 20 = 0.36)
  const auto slidOver = [](float friction) {
    const float pi = float(std::acos(-1.0)), sn = std::sin(20 * pi / 180), cs = std::cos(20 * pi / 180);
    Grid        slope = grid(60, 0), sheet = grid(12, 0.5f);
    const auto  tilt = [&](std::vector<float>& pos) {
      for(size_t v = 0; v < pos.size() / 3; v++)
      {
        const float x = pos[v * 3], y = pos[v * 3 + 1];
        pos[v * 3]     = x * cs + y * sn;
        pos[v * 3 + 1] = -x * sn + y * cs;
      }
    };
    tilt(slope.pos);
    tilt(sheet.pos);
    dforce::SettleMesh body = dforce::SettleMesh::collider(slope.pos, slope.pos, slope.tris, dforce::kBodyLayer, 0.2f, friction);
    dforce::SettleMesh m    = cloth(sheet);
    std::fill(m.friction.begin(), m.friction.end(), friction);
    dforce::ClothSim sim({body, m});
    const auto       start = sim.positions()[1];
    for(int f = 0; f < 90; f++)
      sim.frame(1, 1);
    const auto now = sim.positions()[1];
    double     mean = 0;
    for(size_t v = 0; v < now.size() / 3; v++)
      mean += double(now[v * 3] - start[v * 3]) / double(now.size() / 3);
    return float(mean);
  };
  const float gripped = slidOver(0.5f), slippery = slidOver(0.f);
  EXPECT_GT(slippery, 50.f);
  EXPECT_LT(std::fabs(gripped), 2.f);
}

TEST(Dforce, aHangingStripKeepsItsLength)
{
  // a 4 x 30 cm strip pinned along its top edge: it hangs about its rest length. The one-sweep
  // solver leaves a little sag, bounded here; the old solver let it hang several times its length
  const int w = 4, h = 30;
  Grid      strip;
  for(int r = 0; r <= h; r++)
    for(int c = 0; c <= w; c++)
      strip.pos.insert(strip.pos.end(), {float(c), float(-r), 0.f});
  const auto at = [&](int c, int r) { return uint32_t(r * (w + 1) + c); };
  for(int r = 0; r < h; r++)
    for(int c = 0; c < w; c++)
    {
      strip.tris.insert(strip.tris.end(), {at(c, r), at(c, r + 1), at(c + 1, r), at(c + 1, r), at(c, r + 1), at(c + 1, r + 1)});
      strip.polyVerts.insert(strip.polyVerts.end(), {int32_t(at(c, r)), int32_t(at(c, r + 1)), int32_t(at(c + 1, r + 1)), int32_t(at(c + 1, r))});
      strip.polyStart.push_back(int32_t(strip.polyVerts.size()));
    }
  auto m = cloth(strip);
  for(int c = 0; c <= w; c++)
    m.strength[at(c, 0)] = 0;
  const auto out = dforce::settle({m});
  for(int c = 0; c <= w; c++)
  {
    const float bottom = -out[0][at(c, h) * 3 + 1];
    EXPECT_GT(bottom, h * 0.98f) << c;
    EXPECT_LT(bottom, h * 1.05f) << c;
  }
}

namespace {

// A sheet hanging from its top row, `rows` centimetres long, in rows and columns `cell` centimetres apart.
dforce::SettleMesh hangingSheet(int columns, int rows, float cell, float stretch)
{
  Grid g;
  for(int r = 0; r <= rows; r++)
    for(int c = 0; c <= columns; c++)
      g.pos.insert(g.pos.end(), {float(c) * cell, -float(r) * cell, 0.f});
  const auto at = [&](int c, int r) { return uint32_t(r * (columns + 1) + c); };
  for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)
    {
      g.tris.insert(g.tris.end(), {at(c, r), at(c, r + 1), at(c + 1, r), at(c + 1, r), at(c, r + 1), at(c + 1, r + 1)});
      g.polyVerts.insert(g.polyVerts.end(), {int32_t(at(c, r)), int32_t(at(c, r + 1)), int32_t(at(c + 1, r + 1)), int32_t(at(c + 1, r))});
      g.polyStart.push_back(int32_t(g.polyVerts.size()));
    }
  dforce::SettleMesh m = cloth(g);
  std::fill(m.stretch.begin(), m.stretch.end(), stretch);
  for(int c = 0; c <= columns; c++)
    m.strength[at(c, 0)] = 0.f;
  return m;
}

// How far the sheet's bottom row has come down, over how long it is.
float hangsAt(const std::vector<float>& pos, float length)
{
  float lowest = 0;
  for(size_t v = 0; v < pos.size() / 3; v++)
    lowest = std::min(lowest, pos[v * 3 + 1]);
  return -lowest / length;
}

}  // namespace

TEST(Dforce, reachLinksCarryAHangingSheetToWhatHoldsIt)
{
  // two metres of fine cloth from one row: edge by edge the sweeps can't bring the load up in time
  const auto hang = [](bool links) {
    dforce::SettleOptions opts;
    opts.reachLinks = links;
    dforce::ClothSim sim({hangingSheet(10, 200, 1.f, 0.8f)}, opts);
    for(int f = 0; f < 45; f++)
      sim.frame(1, 1, 3.f);
    return hangsAt(sim.positions()[0], 200.f);
  };
  const float without = hang(false), with = hang(true);
  EXPECT_LT(with - 1.f, (without - 1.f) * 0.5f) << "without " << without << " with " << with;
}

TEST(Dforce, reachLinksFollowTheThreadsAtTheFabricsCompliance)
{
  const dforce::ClothModel flat = dforce::ClothModel::build({hangingSheet(40, 40, 1.f, 0.8f)}, 0.15f);
  int                      links = 0;
  for(const dforce::Edge& e : flat.edges)
  {
    if(e.kind != dforce::EdgeKind::eReach)
      continue;
    links++;
    const float dx = std::fabs(flat.rest[size_t(e.a) * 3] - flat.rest[size_t(e.b) * 3]), dy = std::fabs(flat.rest[size_t(e.a) * 3 + 1] - flat.rest[size_t(e.b) * 3 + 1]);
    // as long as the way along the threads, which no shear can lengthen
    EXPECT_NEAR(e.rest, dx + dy, 1e-3f);
    EXPECT_NEAR(e.alpha, flat.edges[0].alpha, flat.edges[0].alpha * 0.01f);
    // pinned to pinned holds nothing
    EXPECT_TRUE(flat.inv[size_t(e.a)] > 0 || flat.inv[size_t(e.b)] > 0);
  }
  EXPECT_GT(links, 0);
  // the links are few against the edges
  EXPECT_LT(links, int(flat.edges.size()) / 10);
  // another mesh's cloth isn't linked to this one
  const dforce::ClothModel two = dforce::ClothModel::build({hangingSheet(10, 10, 1.f, 0.8f), hangingSheet(10, 10, 1.f, 0.8f)}, 0.15f);
  for(const dforce::Edge& e : two.edges)
    EXPECT_EQ(two.meshOf[size_t(e.a)], two.meshOf[size_t(e.b)]);
}

TEST(Dforce, reachLinksFollowTheFabricsExpansion)
{
  dforce::SettleMesh m = hangingSheet(40, 40, 1.f, 0.8f);
  std::fill(m.contraction.begin(), m.contraction.end(), 1.2f);
  const dforce::ClothModel model = dforce::ClothModel::build({m}, 0.15f);
  int                      links = 0;
  for(const dforce::Edge& e : model.edges)
  {
    if(e.kind != dforce::EdgeKind::eReach)
      continue;
    links++;
    const float dx = std::fabs(model.rest[size_t(e.a) * 3] - model.rest[size_t(e.b) * 3]), dy = std::fabs(model.rest[size_t(e.a) * 3 + 1] - model.rest[size_t(e.b) * 3 + 1]);
    EXPECT_NEAR(e.rest, (dx + dy) * 1.2f, 1e-2f);
  }
  EXPECT_GT(links, 0);
}

TEST(Dforce, aReachLinkOverASoftPanelIsAsSoftAsThePanel)
{
  // stiff above and below a soft band three rows tall
  dforce::SettleMesh m = hangingSheet(40, 40, 1.f, 0.8f);
  for(int r = 20; r < 23; r++)
    for(int c = 0; c <= 40; c++)
      m.stretch[size_t(r * 41 + c)] = 0.05f;
  const dforce::ClothModel model = dforce::ClothModel::build({m}, 0.15f);
  int                      across = 0;
  float                    stiffest = INFINITY, softestEdge = 0;
  for(const dforce::Edge& e : model.edges)
  {
    if(e.kind == dforce::EdgeKind::eStretch)
      softestEdge = std::max(softestEdge, e.alpha);
    if(e.kind != dforce::EdgeKind::eReach)
      continue;
    const float ya = -model.rest[size_t(e.a) * 3 + 1], yb = -model.rest[size_t(e.b) * 3 + 1];
    if(std::min(ya, yb) < 20 && std::max(ya, yb) > 22)
    {
      across++;
      stiffest = std::min(stiffest, e.alpha);
    }
  }
  ASSERT_GT(across, 0);
  EXPECT_NEAR(stiffest, softestEdge, softestEdge * 0.01f);
}

TEST(Dforce, aVertexTakesOnlySoManyLinks)
{
  // a pinned band with strands hanging from it: the band's seeds are within reach of every strand
  const int  strands = 120, points = 30;
  Grid       fringe;
  const auto at = [&](int s, int p) { return uint32_t(s * points + p); };
  for(int s = 0; s < strands; s++)
    for(int p = 0; p < points; p++)
      fringe.pos.insert(fringe.pos.end(), {float(s) * 0.5f, -float(p) * 0.5f, 0.f});
  for(int s = 0; s < strands; s++)
  {
    for(int p = 0; p + 1 < points; p++)
    {
      fringe.polyVerts.insert(fringe.polyVerts.end(), {int32_t(at(s, p)), int32_t(at(s, p + 1))});
      fringe.polyStart.push_back(int32_t(fringe.polyVerts.size()));
    }
    if(s + 1 < strands)
    {
      fringe.polyVerts.insert(fringe.polyVerts.end(), {int32_t(at(s, 0)), int32_t(at(s + 1, 0))});
      fringe.polyStart.push_back(int32_t(fringe.polyVerts.size()));
    }
  }
  dforce::SettleMesh m = cloth(fringe);
  for(int s = 0; s < strands; s++)
    m.strength[at(s, 0)] = 0.f;
  const dforce::ClothModel model = dforce::ClothModel::build({m}, 0.15f);
  std::vector<int>         links(size_t(model.nv), 0);
  int                      total = 0;
  for(const dforce::Edge& e : model.edges)
    if(e.kind == dforce::EdgeKind::eReach)
    {
      links[size_t(e.a)]++;
      links[size_t(e.b)]++;
      total++;
    }
  EXPECT_GT(total, 0);
  EXPECT_LE(*std::max_element(links.begin(), links.end()), 12);
}

TEST(Dforce, reachLinksLeaveShearToTheFabric)
{
  // a loose weave hung from its two top corners sags on the bias, by shear. Against the same cloth with
  // no links and sweeps enough to converge: without those, it falls further only by stretching
  const auto hang = [](bool withLinks) {
    dforce::SettleMesh m = hangingSheet(60, 60, 1.f, 0.8f);
    std::fill(m.strength.begin(), m.strength.end(), 1.f);
    m.strength[0]  = 0.f;
    m.strength[60] = 0.f;
    std::fill(m.shear.begin(), m.shear.end(), 0.05f);
    dforce::SettleOptions opts;
    opts.reachLinks = withLinks;
    if(!withLinks)
    {
      opts.stretchSweeps = 40;
      opts.substeps      = 48;
    }
    dforce::ClothSim sim({m}, opts);
    sim.settle();
    return sim.positions()[0];
  };
  const auto  with = hang(true), converged = hang(false);
  // the middle of the top edge, which only the bias lets fall
  const float fell = -with[30 * 3 + 1], fellConverged = -converged[30 * 3 + 1];
  EXPECT_GT(fellConverged, 3.f);
  EXPECT_NEAR(fell, fellConverged, fellConverged * 0.25f);
}

TEST(Dforce, aStretchyFabricStaysStretchyWithReachLinks)
{
  const auto hang = [](float stretch) {
    dforce::SettleOptions opts;
    dforce::ClothSim      sim({hangingSheet(10, 100, 1.f, stretch)}, opts);
    for(int f = 0; f < 45; f++)
      sim.frame(1, 1, 3.f);
    return hangsAt(sim.positions()[0], 100.f);
  };
  EXPECT_GT(hang(0.02f), hang(0.9f) + 0.01f);
}

TEST(Dforce, aReachLinkOnlyPullsSoClothCanGather)
{
  // both ends brought to half the distance: the cloth between gathers, its links slack
  dforce::SettleMesh m = hangingSheet(40, 2, 1.f, 0.8f);
  std::fill(m.strength.begin(), m.strength.end(), 1.f);
  for(int r = 0; r <= 2; r++)
    for(int c : {0, 40})
    {
      const size_t v = size_t(r) * 41 + size_t(c);
      m.strength[v]  = 0.f;
      m.posed[v * 3] = c == 0 ? 10.f : 30.f;
    }
  dforce::SettleOptions opts;
  dforce::ClothSim      sim({m}, opts);
  sim.settle();
  const auto pos   = sim.positions()[0];
  int        slack = 0;
  for(const dforce::Edge& e : sim.model().edges)
  {
    if(e.kind != dforce::EdgeKind::eReach)
      continue;
    const float len = std::hypot(pos[size_t(e.a) * 3] - pos[size_t(e.b) * 3], pos[size_t(e.a) * 3 + 1] - pos[size_t(e.b) * 3 + 1], pos[size_t(e.a) * 3 + 2] - pos[size_t(e.b) * 3 + 2]);
    if(len < e.rest * 0.8f)
      slack++;
  }
  EXPECT_GT(slack, 0);
}

TEST(Dforce, aSoftBandFollowsItsBoneAndAStrengthGradientDoesNotTear)
{
  // a 4 x 30 cm strip skinned to one bone: its top row pinned, the next three at a fifth of the
  // strength (a skirt's band), then a ramp to free cloth. The bone moves 5 cm and turns 45 degrees:
  // the band goes with it, the rest hangs under gravity, and no edge tears at the gradient
  const int w = 4, h = 30;
  Grid      strip;
  for(int r = 0; r <= h; r++)
    for(int c = 0; c <= w; c++)
      strip.pos.insert(strip.pos.end(), {float(c), float(-r), 0.f});
  const auto at = [&](int c, int r) { return uint32_t(r * (w + 1) + c); };
  for(int r = 0; r < h; r++)
    for(int c = 0; c < w; c++)
    {
      strip.tris.insert(strip.tris.end(), {at(c, r), at(c, r + 1), at(c + 1, r), at(c + 1, r), at(c, r + 1), at(c + 1, r + 1)});
      strip.polyVerts.insert(strip.polyVerts.end(), {int32_t(at(c, r)), int32_t(at(c, r + 1)), int32_t(at(c + 1, r + 1)), int32_t(at(c + 1, r))});
      strip.polyStart.push_back(int32_t(strip.polyVerts.size()));
    }
  auto m = cloth(strip);
  for(int c = 0; c <= w; c++)
  {
    m.strength[at(c, 0)] = 0;
    for(int r = 1; r <= 3; r++)
      m.strength[at(c, r)] = 0.2f;
    for(int r = 4; r <= 8; r++)
      m.strength[at(c, r)] = 0.3f + 0.7f * float(r - 4) / 4;
  }
  const size_t n = strip.pos.size() / 3;
  m.skinIndex.assign(n * 4, 0);
  m.skinWeight.assign(n * 4, 0.f);
  for(size_t v = 0; v < n; v++)
    m.skinWeight[v * 4] = 1;
  const Mat4 bone = compose({5, 0, 0}, quatFromEuler(0, 0, std::acos(-1.0) / 4, "XYZ"), {1, 1, 1});
  m.bones          = {bone};
  m.restLocal      = m.rest;
  m.posed.resize(n * 3);
  dforce::skinMesh(m, m.bones, m.posed.data());

  // half way through the transition the band is at the bone's half turn, on the arc
  {
    dforce::ClothSim sim({m});
    sim.frame(0, 0.5f);
    const auto p    = sim.positions();
    const Mat4 half = dforce::sweepAt(dforce::poseOf(Mat4{}), dforce::poseOf(bone), 0.5f);
    for(int c = 0; c <= w; c++)
    {
      const uint32_t v = at(c, 2);
      const Vec3     a = half.transformPoint({m.rest[v * 3], m.rest[v * 3 + 1], m.rest[v * 3 + 2]});
      const double   d = std::hypot(p[0][v * 3] - a.x, p[0][v * 3 + 1] - a.y, p[0][v * 3 + 2] - a.z);
      EXPECT_LT(d, 1.0) << c;
    }
  }
  const auto out = dforce::settle({m});
  for(int c = 0; c <= w; c++)
  {
    for(int r = 1; r <= 3; r++)
    {
      const uint32_t v = at(c, r);
      const double   d = std::hypot(out[0][v * 3] - m.posed[v * 3], out[0][v * 3 + 1] - m.posed[v * 3 + 1], out[0][v * 3 + 2] - m.posed[v * 3 + 2]);
      EXPECT_LT(d, 1.5) << c << " " << r;
    }
    // the free end hangs below where the turned bone would hold it
    EXPECT_LT(out[0][at(c, h) * 3 + 1], m.posed[at(c, h) * 3 + 1] - 5) << c;
  }
  for(int r = 0; r <= h; r++)
    for(int c = 0; c <= w; c++)
    {
      const auto len = [&](uint32_t a, uint32_t b) {
        return std::hypot(out[0][a * 3] - out[0][b * 3], out[0][a * 3 + 1] - out[0][b * 3 + 1], out[0][a * 3 + 2] - out[0][b * 3 + 2]);
      };
      if(r < h)
        EXPECT_LT(len(at(c, r), at(c, r + 1)), 1.25) << c << " " << r;
      if(c < w)
        EXPECT_LT(len(at(c, r), at(c + 1, r)), 1.25) << c << " " << r;
    }
}

TEST(Dforce, aSkinnedColliderSweepsAlongItsBone)
{
  // a collider skinned to one bone that turns a quarter about y: half way through the transition it
  // is at the quarter's half, on the arc, not half way along the chord
  const std::vector<float> rest{10, 0, 0, 10, 1, 0};
  dforce::SettleMesh       c = dforce::SettleMesh::collider(rest, {0, 0, -10, 0, 1, -10}, {}, dforce::kBodyLayer, 0.2f, 0.4f);
  c.skinIndex.assign(8, 0);
  c.skinWeight = {1, 0, 0, 0, 1, 0, 0, 0};
  c.bones      = {compose({}, quatFromEuler(0, std::acos(-1.0) / 2, 0, "XYZ"), {1, 1, 1})};
  c.restLocal  = rest;
  dforce::ClothSim sim({c});
  sim.frame(0, 0.5f);
  const auto p = sim.positions();
  EXPECT_NEAR(p[0][0], 10 * std::cos(std::acos(-1.0) / 4), 1e-3);
  EXPECT_NEAR(p[0][2], -10 * std::sin(std::acos(-1.0) / 4), 1e-3);
  sim.frame(0.5f, 1);
  const auto q = sim.positions();
  EXPECT_NEAR(q[0][0], 0, 1e-3);
  EXPECT_NEAR(q[0][2], -10, 1e-3);
}

TEST(Dforce, aSweepTurnsABoneAboutItsJoint)
{
  // a skin matrix is a turn about the joint: half way through the sweep the joint has not moved and
  // a point beside it is at the half turn about the joint, not about the mesh's origin
  const Vec3 joint{0, 100, 0};
  const Mat4 bone = compose(joint, {}, {1, 1, 1}) * compose({}, quatFromEuler(0, std::acos(-1.0) / 2, 0, "XYZ"), {1, 1, 1})
                    * compose({0, -100, 0}, {}, {1, 1, 1});
  const Mat4 half = dforce::sweepAt(dforce::poseOf(Mat4{}), dforce::poseOf(bone), 0.5f);
  const Vec3 j    = half.transformPoint(joint);
  EXPECT_NEAR(j.x, joint.x, 1e-3);
  EXPECT_NEAR(j.y, joint.y, 1e-3);
  EXPECT_NEAR(j.z, joint.z, 1e-3);
  const Vec3 p = half.transformPoint({10, 100, 0});
  EXPECT_NEAR(p.x, 10 * std::cos(std::acos(-1.0) / 4), 1e-3);
  EXPECT_NEAR(p.y, 100, 1e-3);
  EXPECT_NEAR(p.z, -10 * std::sin(std::acos(-1.0) / 4), 1e-3);
  // the ends are exact
  const Vec3 end = dforce::sweepAt(dforce::poseOf(Mat4{}), dforce::poseOf(bone), 1).transformPoint({10, 100, 0});
  EXPECT_NEAR(end.x, 0, 1e-3);
  EXPECT_NEAR(end.z, -10, 1e-3);
}

TEST(Dforce, aLiveSimulationFollowsATargetThatMovesEveryFrame)
{
  // a veil over a skinned collider whose bone is given a new target every frame, as a drag does:
  // the veil rides the collider and comes to rest once the targets stop
  const Grid body = grid(8, 0), veil = grid(4, 3);
  auto       c    = dforce::SettleMesh::collider(body.pos, body.pos, body.tris, dforce::kBodyLayer, 0.2f, 0.4f);
  const size_t n = body.pos.size() / 3;
  c.skinIndex.assign(n * 4, 0);
  c.skinWeight.assign(n * 4, 0.f);
  for(size_t v = 0; v < n; v++)
    c.skinWeight[v * 4] = 1;
  c.bones     = {Mat4{}};
  c.restLocal = body.pos;
  dforce::ClothSim sim({c, cloth(veil)});
  // the collider rises 0.2 cm a frame for 20 frames, then holds
  for(int f = 1; f <= 40; f++)
  {
    const double lift = std::min(f, 20) * 0.2;
    sim.setTarget(0, {compose({0, lift, 0}, Quat{}, {1, 1, 1})});
    sim.step(f > 20 ? 6.f : 0.f);
    const auto p = sim.positions();
    EXPECT_NEAR(p[0][1], lift, 1e-4) << "collider at frame " << f;
    if(f >= 10)
      for(size_t v = 0; v < p[1].size() / 3; v++)
        EXPECT_GT(p[1][v * 3 + 1], lift) << "veil vertex " << v << " at frame " << f;
  }
  EXPECT_LT(sim.lastMove(), 0.05f);
  const auto p = sim.positions();
  for(size_t v = 0; v < p[1].size() / 3; v++)
    EXPECT_LT(p[1][v * 3 + 1], 4 + 1.0f) << v;
}

TEST(Dforce, aLiveStepAfterASettleHoldsThePoseItReached)
{
  // a strip pinned along its top row to a bone that has moved 20 cm from the rest pose: settled,
  // it hangs still from the moved bone. A live step with no new target holds that pose; it must not
  // set the pinned row out from the rest pose again and sweep it to the bone every frame, which
  // would drag the hanging cloth centimetres a frame for good
  const int w = 4, h = 10;
  Grid      strip;
  for(int r = 0; r <= h; r++)
    for(int c = 0; c <= w; c++)
      strip.pos.insert(strip.pos.end(), {float(c), float(-r), 0.f});
  const auto at = [&](int c, int r) { return uint32_t(r * (w + 1) + c); };
  for(int r = 0; r < h; r++)
    for(int c = 0; c < w; c++)
    {
      strip.tris.insert(strip.tris.end(), {at(c, r), at(c, r + 1), at(c + 1, r), at(c + 1, r), at(c, r + 1), at(c + 1, r + 1)});
      strip.polyVerts.insert(strip.polyVerts.end(), {int32_t(at(c, r)), int32_t(at(c, r + 1)), int32_t(at(c + 1, r + 1)), int32_t(at(c + 1, r))});
      strip.polyStart.push_back(int32_t(strip.polyVerts.size()));
    }
  auto m = cloth(strip);
  for(int c = 0; c <= w; c++)
    m.strength[at(c, 0)] = 0;
  const size_t n = strip.pos.size() / 3;
  m.skinIndex.assign(n * 4, 0);
  m.skinWeight.assign(n * 4, 0.f);
  for(size_t v = 0; v < n; v++)
    m.skinWeight[v * 4] = 1;
  const Mat4 bone = compose({20, 0, 0}, Quat{}, {1, 1, 1});
  m.bones          = {bone};
  m.restLocal      = m.rest;
  m.posed.resize(n * 3);
  dforce::skinMesh(m, m.bones, m.posed.data());

  dforce::ClothSim sim({m});
  sim.settle();
  const auto settled = sim.positions();
  for(int f = 0; f < 5; f++)
  {
    sim.step(6.f);
    EXPECT_LT(sim.lastMove(), 0.1f) << "step " << f;
    const auto p = sim.positions();
    for(size_t v = 0; v < n; v++)
    {
      const double d = std::hypot(p[0][v * 3] - settled[0][v * 3], p[0][v * 3 + 1] - settled[0][v * 3 + 1], p[0][v * 3 + 2] - settled[0][v * 3 + 2]);
      EXPECT_LT(d, 0.5) << "vertex " << v << " at step " << f;
    }
    for(int c = 0; c <= w; c++)
      EXPECT_NEAR(p[0][at(c, 0) * 3], 20 + c, 1e-3) << "pinned " << c << " at step " << f;
  }
  // a new target sweeps again: the bone back 10 cm over one step, the pinned row with it
  sim.setTarget(0, {compose({10, 0, 0}, Quat{}, {1, 1, 1})});
  sim.step();
  const auto p = sim.positions();
  for(int c = 0; c <= w; c++)
    EXPECT_NEAR(p[0][at(c, 0) * 3], 10 + c, 1e-3) << "pinned " << c;
}

TEST(Dforce, aSkirtDrapesOverALegThatSwings)
{
  // a tube skirt pinned at the waist over a thigh that swings forward 45 degrees about the hip: the
  // skirt stays outside the leg and its edges stay near their rest length
  const int   around = 24, rows = 15;
  const float R = 14, r = 10, len = 45, swing = 45 * float(std::acos(-1.0)) / 180;
  Grid        skirt;
  for(int i = 0; i <= rows; i++)
    for(int j = 0; j < around; j++)
    {
      const float a = 2 * float(std::acos(-1.0)) * j / around;
      skirt.pos.insert(skirt.pos.end(), {R * std::cos(a), -float(i) * (len / rows), R * std::sin(a)});
    }
  const auto at = [&](int j, int i) { return uint32_t(i * around + (j % around)); };
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < around; j++)
    {
      skirt.tris.insert(skirt.tris.end(), {at(j, i), at(j, i + 1), at(j + 1, i), at(j + 1, i), at(j, i + 1), at(j + 1, i + 1)});
      skirt.polyVerts.insert(skirt.polyVerts.end(), {int32_t(at(j, i)), int32_t(at(j, i + 1)), int32_t(at(j + 1, i + 1)), int32_t(at(j + 1, i))});
      skirt.polyStart.push_back(int32_t(skirt.polyVerts.size()));
    }
  auto m = cloth(skirt);
  for(int j = 0; j < around; j++)
    m.strength[at(j, 0)] = 0;
  // the thigh: a cylinder from just under the hip, skinned to one bone that turns about x
  const int             la = 24, lr = 30;
  std::vector<float>    lrest, lposed;
  std::vector<uint32_t> ltris;
  const float           sn = std::sin(swing), cs = std::cos(swing);
  for(int i = 0; i <= lr; i++)
    for(int j = 0; j < la; j++)
    {
      const float a = 2 * float(std::acos(-1.0)) * j / la;
      const float x = r * std::cos(a), y = -float(i) * (len + 15) / lr - 3, z = r * std::sin(a);
      lrest.insert(lrest.end(), {x, y, z});
      lposed.insert(lposed.end(), {x, cs * y - sn * z, sn * y + cs * z});
    }
  const auto lat = [&](int j, int i) { return uint32_t(i * la + (j % la)); };
  for(int i = 0; i < lr; i++)
    for(int j = 0; j < la; j++)
      ltris.insert(ltris.end(), {lat(j, i), lat(j + 1, i), lat(j, i + 1), lat(j + 1, i), lat(j + 1, i + 1), lat(j, i + 1)});
  dforce::SettleMesh leg = dforce::SettleMesh::collider(lrest, lposed, ltris, dforce::kBodyLayer, 0.2f, 0.4f);
  leg.skinIndex.assign(lrest.size() / 3 * 4, 0);
  leg.skinWeight.assign(lrest.size() / 3 * 4, 0.f);
  for(size_t v = 0; v < lrest.size() / 3; v++)
    leg.skinWeight[v * 4] = 1;
  leg.bones     = {compose({}, quatFromEuler(swing, 0, 0, "XYZ"), {1, 1, 1})};
  leg.restLocal = lrest;
  const auto out = dforce::settle({leg, m});
  ASSERT_EQ(out.size(), 2u);
  // every free skirt vertex alongside the thigh is outside it (distance from the swung axis; the
  // pinned waist cannot move out of its way), and no vertical edge is stretched past a fifth
  const float ax = 0, ay = -cs, az = -sn;
  for(size_t v = 0; v < out[1].size() / 3; v++)
  {
    if(m.strength[v] == 0)
      continue;
    const float* p = &out[1][v * 3];
    const float  d = p[0] * ax + p[1] * ay + p[2] * az;
    if(d < 3 || d > len + 12)
      continue;
    const float dx = p[0] - d * ax, dy = p[1] - d * ay, dz = p[2] - d * az;
    EXPECT_GT(std::sqrt(dx * dx + dy * dy + dz * dz), r - 0.05f) << v;
  }
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < around; j++)
    {
      const float* a = &out[1][at(j, i) * 3];
      const float* b = &out[1][at(j, i + 1) * 3];
      const float  e = std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
      EXPECT_LT(e, (len / rows) * 1.2f) << i << " " << j;
    }
}

TEST(Dforce, aTightSkirtInAStrideStaysOutOfBothThighs)
{
  // a tube skirt too tight for the stride over two thighs, one swung forward and one back: the hem
  // rides up and the tube wraps the legs like a band, folding over itself. No cloth may end inside
  // a thigh, and none is lost and put back at the end (a vertex crossing a face's edge, or caught
  // between a thigh and a fold over it, used to slip in and be recovered every frame)
  const float pi = float(std::acos(-1.0));
  const int   around = 36, rows = 14;
  const float R = 17, r = 7, len = 45, legLen = 70, stride = 25 * pi / 180;
  Grid        skirt;
  for(int i = 0; i <= rows; i++)
    for(int j = 0; j < around; j++)
    {
      const float a = 2 * pi * j / around;
      skirt.pos.insert(skirt.pos.end(), {R * std::cos(a), -float(i) * (len / rows), 0.75f * R * std::sin(a)});
    }
  const auto at = [&](int j, int i) { return uint32_t(i * around + (j % around)); };
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < around; j++)
    {
      skirt.tris.insert(skirt.tris.end(), {at(j, i), at(j, i + 1), at(j + 1, i), at(j + 1, i), at(j, i + 1), at(j + 1, i + 1)});
      skirt.polyVerts.insert(skirt.polyVerts.end(), {int32_t(at(j, i)), int32_t(at(j, i + 1)), int32_t(at(j + 1, i + 1)), int32_t(at(j + 1, i))});
      skirt.polyStart.push_back(int32_t(skirt.polyVerts.size()));
    }
  auto m = cloth(skirt);
  for(int j = 0; j < around; j++)
    m.strength[at(j, 0)] = 0;
  struct Leg
  {
    float hx, hy, angle;
  };
  const Leg                       legs[2] = {{-8.f, -3.f, stride}, {8.f, -3.f, -stride}};
  std::vector<dforce::SettleMesh> meshes;
  for(const Leg& L : legs)
  {
    const int             la = 32, lr = 40;
    std::vector<float>    lrest, lposed;
    std::vector<uint32_t> ltris;
    const float           sn = std::sin(L.angle), cs = std::cos(L.angle);
    for(int i = 0; i <= lr; i++)
      for(int j = 0; j < la; j++)
      {
        const float a = 2 * pi * j / la;
        const float x = L.hx + r * std::cos(a), y = L.hy - float(i) * legLen / lr, z = r * std::sin(a);
        lrest.insert(lrest.end(), {x, y, z});
        lposed.insert(lposed.end(), {x, L.hy + cs * (y - L.hy) - sn * z, sn * (y - L.hy) + cs * z});
      }
    const auto lat = [&](int j, int i) { return uint32_t(i * la + (j % la)); };
    for(int i = 0; i < lr; i++)
      for(int j = 0; j < la; j++)
        ltris.insert(ltris.end(), {lat(j, i), lat(j + 1, i), lat(j, i + 1), lat(j + 1, i), lat(j + 1, i + 1), lat(j, i + 1)});
    dforce::SettleMesh leg = dforce::SettleMesh::collider(lrest, lposed, ltris, dforce::kBodyLayer, 0.2f, 0.4f);
    leg.skinIndex.assign(lrest.size() / 3 * 4, 0);
    leg.skinWeight.assign(lrest.size() / 3 * 4, 0.f);
    for(size_t v = 0; v < lrest.size() / 3; v++)
      leg.skinWeight[v * 4] = 1;
    // the bone turns about the hip joint
    leg.bones = {compose({0, L.hy, 0}, {}, {1, 1, 1}) * compose({}, quatFromEuler(L.angle, 0, 0, "XYZ"), {1, 1, 1}) * compose({0, -L.hy, 0}, {}, {1, 1, 1})};
    leg.restLocal = lrest;
    meshes.push_back(std::move(leg));
  }
  meshes.push_back(m);
  dforce::ClothSim sim(meshes);
  sim.settle();
  // what is still being put back when a settle ends stops, given the time
  dforce::SettleOptions longer;
  longer.maxHold = 3;
  dforce::ClothSim given(meshes, longer);
  given.settle();
  EXPECT_EQ(given.recovered(), 0);
  const auto out = sim.positions();
  int        inside = 0;
  for(size_t v = 0; v < out[2].size() / 3; v++)
  {
    if(m.strength[v] == 0)
      continue;
    const float* p = &out[2][v * 3];
    for(const Leg& L : legs)
    {
      const float ax = 0, ay = -std::cos(L.angle), az = -std::sin(L.angle);
      const float qx = p[0] - L.hx, qy = p[1] - L.hy, qz = p[2];
      const float d  = qx * ax + qy * ay + qz * az;
      if(d < 2 || d > legLen - 2)
        continue;
      const float ex = qx - d * ax, ey = qy - d * ay, ez = qz - d * az;
      if(std::sqrt(ex * ex + ey * ey + ez * ez) < r - 0.05f)
        inside++;
    }
  }
  EXPECT_EQ(inside, 0);
}

// DISABLED: self-collision lets a falling fold through. Looking for cloth contacts every fourth
// collision pass misses it (505 edges through the lower half at 1 cm, 84 at 2 cm); looking every
// pass catches it (12 and 0) but makes a crumpled pile inflate, since a vertex-face push moves the
// vertex alone and an edge crossing an edge is never seen. Needs edge-edge contacts and one side per
// pair first (app docs/improvements.md, "Self-collision lets a fold through").
TEST(Dforce, DISABLED_aFoldFallingOntoItselfStaysOnTop)
{
  // a ribbon folded in half: the lower half on the floor, the upper half 4 cm over it, falling
  const float h = 1, W = 12, L = 60, half = L / 2;
  const int   nw = int(W / h), nl = int(L / h);
  Grid        g;
  for(int i = 0; i <= nl; i++)
    for(int j = 0; j <= nw; j++)
    {
      const float s = float(i) * h;
      const float z = s <= half ? s : 2 * half - s;
      const float y = s <= half ? 0.4f : 0.4f + 4.f * std::min(1.f, (s - half) / 4.f);
      g.pos.insert(g.pos.end(), {float(j) * h - W / 2, y, z - half / 2});
    }
  const auto at = [&](int j, int i) { return uint32_t(i * (nw + 1) + j); };
  for(int i = 0; i < nl; i++)
    for(int j = 0; j < nw; j++)
    {
      g.tris.insert(g.tris.end(), {at(j, i), at(j, i + 1), at(j + 1, i), at(j + 1, i), at(j, i + 1), at(j + 1, i + 1)});
      g.polyVerts.insert(g.polyVerts.end(), {int32_t(at(j, i)), int32_t(at(j, i + 1)), int32_t(at(j + 1, i + 1)), int32_t(at(j + 1, i))});
      g.polyStart.push_back(int32_t(g.polyVerts.size()));
    }
  const Grid floor = grid(24, 0);
  // the floor grid faces up; scaled out to 72 cm
  std::vector<float> fp = floor.pos;
  for(float& c : fp)
    c *= 3;
  dforce::ClothSim sim({dforce::SettleMesh::collider(fp, fp, floor.tris, dforce::kBodyLayer, 0.2f, 0.4f), cloth(g)});
  for(int f = 0; f < 60; f++)
    sim.frame(1, 1, f > 30 ? 6.f : 0.f);
  // the upper half lies over the lower: every upper vertex above the lower half's height
  const auto p = sim.positions()[1];
  for(int i = nl / 2 + 5; i <= nl; i++)
    for(int j = 0; j <= nw; j++)
      EXPECT_GT(p[at(j, i) * 3 + 1], 0.45f) << i << " " << j;
}

TEST(Dforce, aPinnedSurfaceStaysOnTheSkinnedPose)
{
  const Grid veil = grid(3, 3);
  auto       m    = cloth(veil);
  m.strength.assign(m.strength.size(), 0.f);
  for(size_t v = 0; v < m.posed.size() / 3; v++)
    m.posed[v * 3] += 1;
  const auto out = dforce::settle({m});
  for(size_t v = 0; v < m.posed.size(); v++)
    EXPECT_NEAR(out[0][v], m.posed[v], 1e-4) << v;
}

TEST(Dforce, aDrapeSkinsBackToTheSimulatedShape)
{
  // one bone turned a quarter about y, and scaled: the blend has a 3x3 that is not a rotation
  Mat4 bone = compose({1, 2, 3}, quatFromEuler(0, std::acos(-1.0) / 2, 0, "XYZ"), {1.5, 1, 1});
  dforce::SkinPose pose;
  pose.bones = {bone};
  mesh::SkinInput in;
  in.start  = {0, 1, 2};
  in.bone   = {0, 0};
  in.weight = {1.f, 1.f};
  const dforce::CageSkin skin = dforce::cageSkin(in);
  const std::vector<float> rest{1, 0, 0, 0, 1, 0};
  const auto posed = dforce::skinLocal(rest, skin, pose);
  std::vector<float> simulated = posed;
  simulated[1] -= 0.5f;
  simulated[5] += 0.25f;
  const auto delta = dforce::drapeDelta(rest, simulated, skin, pose);
  std::vector<float> draped = rest;
  for(size_t i = 0; i < draped.size(); i++)
    draped[i] += delta[i];
  const auto back = dforce::skinLocal(draped, skin, pose);
  for(size_t i = 0; i < back.size(); i++)
    EXPECT_NEAR(back[i], simulated[i], 1e-4) << i;
}

// DAZ's skinning on the CPU (gpu-figures-plan.md F3, E4): a vertex half on a bone twisted half a turn
// about the bone's axis keeps its distance from the axis with dual quaternions (the linear blend puts
// it on the axis), lies half way at a share of 0.5, and a drape through that skin skins back to the
// simulated shape, as with the linear blend
TEST(Dforce, aDualQuaternionSkinKeepsATwistedJointsVolumeAndDrapesBack)
{
  const double     pi = std::acos(-1.0);
  dforce::SkinPose pose;
  pose.bones = {Mat4{}, compose({0, 0, 0}, quatFromEuler(pi, 0, 0, "XYZ"), {1, 1, 1})};
  mesh::SkinInput in;
  in.start  = {0, 2, 4};
  in.bone   = {0, 1, 0, 1};
  in.weight = {0.5f, 0.5f, 0.75f, 0.25f};
  const std::vector<float> rest{1, 2, 0, 1, 0, 2};
  for(const float share : {0.f, 0.5f, 1.f})
  {
    in.dualQuat                 = share;
    const dforce::CageSkin skin = dforce::cageSkin(in, true);
    const auto             posed = dforce::skinLocal(rest, skin, pose);
    EXPECT_NEAR(std::hypot(posed[1], posed[2]), 2.0 * share, 1e-5) << share;
    EXPECT_NEAR(posed[0], 1.0, 1e-5) << share;

    std::vector<float> simulated = posed;
    simulated[1] -= 0.5f;
    simulated[5] += 0.25f;
    const auto         delta  = dforce::drapeDelta(rest, simulated, skin, pose);
    std::vector<float> draped = rest;
    for(size_t i = 0; i < draped.size(); i++)
      draped[i] += delta[i];
    const auto back = dforce::skinLocal(draped, skin, pose);
    // the half and half vertex's linear blend has no 3x3 to invert (it is left undraped)
    for(size_t i = share > 0.f ? 0 : 3; i < back.size(); i++)
      EXPECT_NEAR(back[i], simulated[i], 1e-4) << share << " " << i;
  }
  // off, the share is not read
  in.dualQuat = 1.f;
  EXPECT_EQ(dforce::cageSkin(in).dualQuat, 0.f);
}

TEST(Dforce, readsTheSettingsOfASimulationSurface)
{
  const auto scene = dson::parseDoc(R"({
    "scene": {"modifiers": [
      {"id": "sim", "url": "#dyn", "parent": "#Dress", "extra": [{"type": "studio/modifier/dynamic_simulation"}]},
      {"id": "surf", "url": "#set", "parent": "#Dress-1", "groups": ["Skirt"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation"}],
       "channel": {"id": "Dynamics Strength", "current_value": 0.5}}
    ]},
    "modifier_library": [
      {"id": "dyn", "extra": [{"type": "studio/modifier/dynamic_simulation", "channels": [
        {"channel": {"id": "Simulation Object Type", "current_value": 1}},
        {"channel": {"id": "Freeze Simulation", "current_value": false}}]}]},
      {"id": "set", "extra": [{"type": "studio/simulation_settings/dynamic_simulation", "channels": [
        {"channel": {"id": "Bend Stiffness", "current_value": 0.9}},
        {"channel": {"id": "Collision Layer", "current_value": 2}}]}]}
    ]
  })");
  const auto geo = dson::parseDoc(R"({
    "vertices": {"count": 4, "values": [[0,0,0],[1,0,0],[1,0,1],[0,0,1]]},
    "polylist": {"count": 1, "values": [[0, 0, 0, 1, 2, 3]]},
    "polygon_material_groups": {"count": 1, "values": ["Skirt"]}
  })");
  const auto data = dson::buildDforce(*scene, nullptr, "Dress", "Dress-1", *geo, geo->json, {});
  ASSERT_TRUE(data);
  EXPECT_EQ(data->kind, dson::DforceKind::eDynamic);
  EXPECT_TRUE(data->simulated());
  ASSERT_EQ(data->bend.size(), 4u);
  EXPECT_NEAR(data->bend[0], 0.9f, 1e-6);
  EXPECT_EQ(data->layer[2], 2);
  EXPECT_EQ(data->polyStart.size(), 2u);
  EXPECT_EQ(data->polyVerts.size(), 4u);

  // a node with no simulation modifier is not a simulation object
  EXPECT_FALSE(dson::buildDforce(*scene, nullptr, "Other", "Other-1", *geo, geo->json, {}));
}

// A strand hair's pre-render hairs, per surface: the definition's values under the scene's, and none
// unless a hair generator on the node has "Generate PR Hairs" on (here in its definition, which the
// geometry's file holds).
TEST(Dforce, readsThePreRenderHairsEachSurfaceGrows)
{
  const auto sceneWith = [](const char* generator) {
    return dson::parseDoc(std::string(R"({
      "scene": {"modifiers": [)") + generator + R"(
        {"id": "back", "url": "#set", "parent": "#geometry", "groups": ["Back"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation",
          "channels": [{"channel": {"id": "PreRender Hairs Per Guide", "current_value": 3}}]}]},
        {"id": "line", "url": "#set", "parent": "#geometry", "groups": ["Hairline"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation"}]}
      ]},
      "modifier_library": [
        {"id": "set", "extra": [{"type": "studio/simulation_settings/dynamic_simulation", "channels": [
          {"channel": {"id": "PreRender Hairs Per Guide", "value": 0}},
          {"channel": {"id": "PreRender Hair Distribution Radius", "value": 0.25}},
          {"channel": {"id": "PreRender Hair Tip Separation", "value": -0.7}},
          {"channel": {"id": "PreRender Hair Seed", "value": 310411}}]}]}
      ]
    })");
  };
  const auto geo = dson::parseDoc(R"({"modifier_library": [
    {"id": "gen", "extra": [{"type": "studio/modifier/dynamic_generate_hair", "generates_for_render": true},
      {"type": "studio_modifier_channels", "channels": [{"channel": {"id": "Generate PreRender Hairs", "type": "bool", "value": true}}]}]}
  ]})");
  const auto on = dson::prHairsFor(*sceneWith(R"({"id": "gen", "url": "/data/hair.dsf#gen", "parent": "#Hair", "extra": [{"type": "studio/modifier/dynamic_generate_hair"}]},)"),
                                   geo.get(), "Hair", "geometry");
  ASSERT_EQ(on.size(), 1u);
  const fv::mesh::Growth& back = on.at("Back");
  EXPECT_EQ(back.perGuide, 3);
  EXPECT_FLOAT_EQ(back.radius, 0.25f);
  EXPECT_FLOAT_EQ(back.tipSeparation, -0.7f);
  EXPECT_FLOAT_EQ(back.bias, 0.5f);
  EXPECT_EQ(back.seed, 310411u);

  // no generator, or one switched off on the node
  EXPECT_TRUE(dson::prHairsFor(*sceneWith(""), geo.get(), "Hair", "geometry").empty());
  const auto off = sceneWith(R"({"id": "gen", "url": "/data/hair.dsf#gen", "parent": "#Hair", "extra": [{"type": "studio/modifier/dynamic_generate_hair"},
    {"type": "studio_modifier_channels", "channels": [{"channel": {"id": "Generate PreRender Hairs", "type": "bool", "current_value": false}}]}]},)");
  EXPECT_TRUE(dson::prHairsFor(*off, geo.get(), "Hair", "geometry").empty());
}

// "Target Surfaces" mode grows by density, at the PR stage or else the PS one, with that stage's look.
// The other mode leaves a density alone, and a stage with its generator off grows nothing.
TEST(Dforce, readsTheDensityASurfaceGrowsAndAtWhichStage)
{
  const auto scene = [](const char* generators, const char* channels) {
    return dson::parseDoc(std::string(R"({"scene": {"modifiers": [)") + generators + R"(
        {"id": "s", "url": "#set", "parent": "#geometry", "groups": ["Back"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation"}]}
      ]},
      "modifier_library": [{"id": "set", "extra": [{"type": "studio/simulation_settings/dynamic_simulation", "channels": [)" + channels + R"(]}]}]
    })");
  };
  const std::string pr = R"({"id": "pr", "parent": "#Hair", "extra": [{"type": "studio/modifier/dynamic_generate_hair"},
    {"type": "studio_modifier_channels", "channels": [{"channel": {"id": "Generate PreRender Hairs", "type": "bool", "value": true}}]}]},)";
  const std::string ps = R"({"id": "ps", "parent": "#Hair", "extra": [{"type": "studio/modifier/dynamic_generate_hair"},
    {"type": "studio_modifier_channels", "channels": [{"channel": {"id": "Generate PreSim Hairs", "type": "bool", "value": true}}]}]},)";
  const char* const channels = R"(
    {"channel": {"id": "Hair Generation Mode", "value": 1}},
    {"channel": {"id": "PreSim Hairs Density", "value": 300}},
    {"channel": {"id": "PreSim Scraggle 1 Mode", "value": 5}},
    {"channel": {"id": "PreSim Scraggliness 1", "value": 0.3}},
    {"channel": {"id": "PreSim Scraggle 1 Frequency ", "value": 0.25}},
    {"channel": {"id": "PreSim Scraggle 2 Mode", "value": 0}},
    {"channel": {"id": "PreSim Scraggliness 2", "value": 0.9}},
    {"channel": {"id": "PreRender Hairs Density", "value": 500}},
    {"channel": {"id": "PreRender Interpolation Single Guide Strength", "value": 1}},
    {"channel": {"id": "PreRender Random Length Amount", "value": 1.5}},
    {"channel": {"id": "PreRender Frizz Tip Amount", "value": 0.5}},
    {"channel": {"id": "PreRender Interpolation Segment Length", "value": 0.3}})";

  const auto both = dson::prHairsFor(*scene((pr + ps).c_str(), channels), nullptr, "Hair", "geometry");
  ASSERT_EQ(both.size(), 1u);
  const fv::mesh::Growth& rendered = both.at("Back");
  EXPECT_FLOAT_EQ(rendered.density, 500.f);
  EXPECT_EQ(rendered.perGuide, 0);
  EXPECT_FLOAT_EQ(rendered.singleGuide, 1.f);
  EXPECT_FLOAT_EQ(rendered.randomLength, 1.5f);
  EXPECT_FLOAT_EQ(rendered.frizzTip, 0.5f);
  EXPECT_FLOAT_EQ(rendered.segment, 0.3f);

  // no PR generator: the PS stage's, its scraggle read and a scraggle of mode None left out
  const auto sim = dson::prHairsFor(*scene(ps.c_str(), channels), nullptr, "Hair", "geometry");
  ASSERT_EQ(sim.size(), 1u);
  EXPECT_FLOAT_EQ(sim.at("Back").density, 300.f);
  EXPECT_FLOAT_EQ(sim.at("Back").scraggle[0].amount, 0.3f);
  EXPECT_FLOAT_EQ(sim.at("Back").scraggle[0].frequency, 0.25f);
  EXPECT_FLOAT_EQ(sim.at("Back").scraggle[1].amount, 0.f);

  EXPECT_TRUE(dson::prHairsFor(*scene("", channels), nullptr, "Hair", "geometry").empty());

  // Root Radius grows per guide, whatever a density says
  const auto radius = dson::prHairsFor(*scene(pr.c_str(), R"({"channel": {"id": "Hair Generation Mode", "value": 0}},
    {"channel": {"id": "PreRender Hairs Density", "value": 120}},
    {"channel": {"id": "PreRender Hairs Per Guide", "value": 3}},
    {"channel": {"id": "PreRender Hair Distribution Radius", "value": 0.1}})"), nullptr, "Hair", "geometry");
  ASSERT_EQ(radius.size(), 1u);
  EXPECT_EQ(radius.at("Back").perGuide, 3);
  EXPECT_FLOAT_EQ(radius.at("Back").density, 0.f);
}

// A strand hair's points are in no polygon: each takes the settings a polygon mesh's loose vertex
// does, without the per-vertex sums a polygon mesh adds up.
TEST(Dforce, PointsInNoPolygonTakeTheSettingsALooseVertexDoes)
{
  const auto scene = dson::parseDoc(R"({
    "scene": {"modifiers": [
      {"id": "sim", "url": "#dyn", "parent": "#Hair", "extra": [{"type": "studio/modifier/dynamic_simulation"}]}
    ]},
    "modifier_library": [
      {"id": "dyn", "extra": [{"type": "studio/modifier/dynamic_simulation", "channels": [
        {"channel": {"id": "Simulation Object Type", "current_value": 1}}]}]}
    ]
  })");
  const auto strands = dson::parseDoc(R"({"vertices": {"count": 5, "values": [[0,0,0],[1,0,0],[1,0,1],[0,0,1],[2,0,2]]}})");
  const auto mesh    = dson::parseDoc(R"({
    "vertices": {"count": 5, "values": [[0,0,0],[1,0,0],[1,0,1],[0,0,1],[2,0,2]]},
    "polylist": {"count": 1, "values": [[0, 0, 0, 1, 2, 3]]}
  })");
  const auto points = dson::buildDforce(*scene, nullptr, "Hair", "Hair-1", *strands, strands->json, {});
  const auto loose  = dson::buildDforce(*scene, nullptr, "Hair", "Hair-1", *mesh, mesh->json, {});
  ASSERT_TRUE(points && loose);
  const auto same = [&](const auto& a, const auto& b, const char* what) {
    ASSERT_EQ(a.size(), 5u) << what;
    for(size_t v = 0; v < 5; v++)
      EXPECT_EQ(a[v], b[4]) << what << " at " << v;
  };
  same(points->strength, loose->strength, "strength");
  same(points->stretch, loose->stretch, "stretch");
  same(points->shear, loose->shear, "shear");
  same(points->bend, loose->bend, "bend");
  same(points->buckling, loose->buckling, "buckling");
  same(points->bucklingRatio, loose->bucklingRatio, "buckling ratio");
  same(points->density, loose->density, "density");
  same(points->contraction, loose->contraction, "contraction");
  same(points->damping, loose->damping, "damping");
  same(points->friction, loose->friction, "friction");
  same(points->offset, loose->offset, "offset");
  same(points->layer, loose->layer, "layer");
  same(points->selfCollide, loose->selfCollide, "self collide");
  same(points->surfaceVisible, loose->surfaceVisible, "visible");
  same(points->surfaceCollide, loose->surfaceCollide, "collide");
  EXPECT_EQ(points->polyStart, std::vector<int32_t>{0});
  EXPECT_TRUE(points->polyVerts.empty());
}

TEST(Dforce, aSparseInfluenceMapLeavesUnlistedVerticesOnTheSkin)
{
  // DAZ writes the influence map sparsely: a vertex it leaves out (a skirt's waistband) has weight
  // 0 and follows the skin; without a map every vertex is free
  const auto geo = dson::parseDoc(R"({
    "vertices": {"count": 4, "values": [[0,0,0],[1,0,0],[1,0,1],[0,0,1]]},
    "polylist": {"count": 1, "values": [[0, 0, 0, 1, 2, 3]]},
    "polygon_material_groups": {"count": 1, "values": ["Skirt"]}
  })");
  const auto sceneWith = [](const char* mapExtra) {
    return dson::parseDoc(std::string(R"({
      "scene": {"modifiers": [
        {"id": "sim", "url": "#dyn", "parent": "#Dress", "extra": [{"type": "studio/modifier/dynamic_simulation"}]},
        {"id": "surf", "url": "#set", "parent": "#Dress-1", "groups": ["Skirt"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation"}],
         "channel": {"id": "Dynamics Strength", "current_value": 0.5}}
      ]},
      "modifier_library": [
        {"id": "dyn", "extra": [{"type": "studio/modifier/dynamic_simulation", "vertex_count": 4)") + mapExtra + R"(}]},
        {"id": "set", "extra": [{"type": "studio/simulation_settings/dynamic_simulation", "channels": []}]}
      ]
    })");
  };
  const auto mapped = dson::buildDforce(*sceneWith(R"(, "influence_weights": {"count": 2, "values": [[0, 1.0], [1, 0.5]]})"), nullptr,
                                        "Dress", "Dress-1", *geo, geo->json, {});
  ASSERT_TRUE(mapped);
  ASSERT_EQ(mapped->influence.size(), 4u);
  ASSERT_EQ(mapped->strength.size(), 4u);
  EXPECT_NEAR(mapped->strength[0], 0.5f, 1e-6);
  EXPECT_NEAR(mapped->strength[1], 0.25f, 1e-6);
  EXPECT_NEAR(mapped->strength[2], 0.f, 1e-6);
  EXPECT_NEAR(mapped->strength[3], 0.f, 1e-6);

  const auto mapless = dson::buildDforce(*sceneWith(""), nullptr, "Dress", "Dress-1", *geo, geo->json, {});
  ASSERT_TRUE(mapless);
  EXPECT_TRUE(mapless->influence.empty());
  for(float s : mapless->strength)
    EXPECT_NEAR(s, 0.5f, 1e-6);
}

TEST(Dforce, aMorphChannelsTargetIsReplacedInPlace)
{
  mesh::BuiltMesh geo;
  geo.position.assign(9, 0.f);
  geo.normal.assign(9, 0.f);
  geo.renderIndex     = {0, 1, 2};
  geo.firstSplit      = {0, 1, 2};
  geo.nextSplit       = {-1, -1, -1};
  geo.renderVertexCount = 3;
  geo.renderTris      = {0, 1, 2};
  geo.dsonVertexCount = 3;
  morph::MorphSet morphs([&](int) { return &geo; });
  const auto make = [](float dy) {
    auto t     = std::make_shared<mesh::RenderTarget>();
    t->setDrawn({.indices = {1}, .deltas = {0, dy, 0}});
    return std::shared_ptr<const mesh::RenderTarget>(t);
  };
  morph::ChannelOptions opts;
  opts.hidden = true;
  opts.value  = 1.0;
  morphs.replaceCpuTarget("dforce:0", 0, make(2), opts);
  morphs.flush();
  EXPECT_NEAR(geo.position[4], 2, 1e-6);
  morphs.replaceCpuTarget("dforce:0", 0, make(5), opts);
  morphs.flush();
  EXPECT_NEAR(geo.position[4], 5, 1e-6);
  ASSERT_TRUE(morphs.cpuTarget("dforce:0", 0));
  morphs.set("dforce:0", 0);
  morphs.flush();
  EXPECT_NEAR(geo.position[4], 0, 1e-6);
}

namespace {

// A grid as a scene mesh: its built geometry and, for the cloth, dForce settings on every vertex.
fv::Mesh sceneMesh(const Grid& g, bool cloth)
{
  fv::mesh::MeshInput in;
  in.positions = g.pos;
  in.polyStart = g.polyStart;
  in.polyVerts = g.polyVerts;
  in.polyMaterial.assign(g.polyStart.size() - 1, 0);
  fv::Mesh m;
  m.geo = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(in));
  m.materialGroups = {"Default"};
  m.materials      = {0};
  if(cloth)
  {
    const auto s   = dforce::SettleMesh(cloth_(g));
    auto       d   = std::make_shared<fv::dson::DforceData>();
    d->strength    = s.strength;
    d->stretch     = s.stretch;
    d->shear       = s.shear;
    d->bend        = s.bend;
    d->buckling    = s.buckling;
    d->bucklingRatio = s.bucklingRatio;
    d->density     = s.density;
    d->contraction = s.contraction;
    d->damping     = s.damping;
    d->friction    = s.friction;
    d->offset      = s.offset;
    d->layer       = s.layer;
    d->selfCollide = s.selfCollide;
    d->surfaceVisible = s.visible;
    d->surfaceCollide = s.collide;
    d->polyStart   = g.polyStart;
    d->polyVerts   = g.polyVerts;
    m.dforce       = d;
  }
  return m;
}

}  // namespace

TEST(Dforce, settlesInTheFiguresCentimetresWhenTheRootIsScaledToMetres)
{
  // as the host has it: the figure root scaled to metres after the load, a body and a cloth under it
  fv::SceneGraph scene;
  fv::Node       root;
  root.scale = {0.01, 0.01, 0.01};
  const int rootNode = scene.add(root);
  const Grid body = grid(8, 0), veil = grid(4, 3);
  scene.meshes.push_back(sceneMesh(body, false));
  scene.meshes.push_back(sceneMesh(veil, true));
  scene.materials.push_back({});
  for(int i = 0; i < 2; i++)
  {
    fv::Node n;
    n.kind = fv::NodeKind::eMesh;
    n.mesh = i;
    scene.add(n, rootNode);
  }
  morph::MorphSet morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(scene, morphs, 0, rootNode);
  ASSERT_TRUE(d.active());
  d.resimulate();
  bool wrote = false;
  for(int i = 0; i < 6000 && !wrote; i++)
  {
    wrote = d.pump();
    if(!wrote)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(wrote);
  EXPECT_EQ(d.draped(), 1);
  // the drape is a morph on the cloth: it came down from 3 cm to rest just above the body, in cm
  const auto& pos = scene.meshes[1].geo->position;
  for(size_t v = 0; v < pos.size() / 3; v++)
  {
    EXPECT_LT(pos[v * 3 + 1], 1.0f) << v;
    EXPECT_GT(pos[v * 3 + 1], 0.0f) << v;
  }
}

TEST(Dforce, aWornItemWithoutSimulationSettingsIsACollider)
{
  // a plain mesh between the body and the veil (a belt, a bra): the veil rests on it, not on the
  // body, unless the host says it is not one of the figure's colliders
  const auto rig = [](std::optional<bool> plainCollides) {
    fv::SceneGraph scene;
    const int      rootNode = scene.add(fv::Node{});
    const Grid     body = grid(8, 0), belt = grid(6, 1.5f), veil = grid(4, 3);
    scene.meshes.push_back(sceneMesh(body, false));
    scene.meshes.push_back(sceneMesh(belt, false));
    scene.meshes.push_back(sceneMesh(veil, true));
    scene.materials.push_back({});
    for(int i = 0; i < 3; i++)
    {
      fv::Node n;
      n.kind = fv::NodeKind::eMesh;
      n.mesh = i;
      scene.add(n, rootNode);
    }
    morph::MorphSet morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
    dforce::Dforce  d(scene, morphs, 0, rootNode);
    if(plainCollides)
      d.collidesWith = [on = *plainCollides](int) { return on; };
    const auto p = d.prepared();
    if(!p)
      return -1.f;
    const auto out = dforce::settle(p->bodies);
    float      low = 1e9f;
    for(const auto& [mesh, slot] : p->drapes)
      for(size_t v = 0; v < out[size_t(slot)].size() / 3; v++)
        low = std::min(low, out[size_t(slot)][v * 3 + 1]);
    return low;
  };
  EXPECT_GT(rig(std::nullopt), 1.5f);
  EXPECT_GT(rig(true), 1.5f);
  const float onBody = rig(false);
  EXPECT_GT(onBody, 0.f);
  EXPECT_LT(onBody, 1.0f);
}

TEST(Dforce, anItemWithoutSimulationSettingsCanBeMadeCloth)
{
  // a worn mesh whose file gives no dForce settings: switched on, it simulates at the defaults over
  // the cage's polygons, at the strength asked; switched off, a file's dynamic surface is held
  const Grid g    = grid(4, 3);
  const auto mesh = sceneMesh(g, false);
  ASSERT_GT(mesh.geo->cagePolyStart.size(), 1u);
  EXPECT_FALSE(clothSettings(*mesh.geo, nullptr, std::nullopt, 1.f));
  const auto cloth = clothSettings(*mesh.geo, nullptr, true, 0.5f);
  ASSERT_TRUE(cloth);
  EXPECT_TRUE(cloth->simulated());
  EXPECT_EQ(cloth->polyStart.size(), size_t(16 + 1));
  ASSERT_EQ(cloth->strength.size(), g.pos.size() / 3);
  for(float s : cloth->strength)
    EXPECT_NEAR(s, 0.5f, 1e-6);
  EXPECT_NEAR(cloth->stretch[0], 0.8f, 1e-6);
  // a file's own settings, held and let go
  const auto file = sceneMesh(g, true).dforce;
  ASSERT_TRUE(file && file->simulated());
  EXPECT_EQ(clothSettings(*mesh.geo, file, std::nullopt, 1.f), file);
  const auto held = clothSettings(*mesh.geo, file, false, 1.f);
  ASSERT_TRUE(held);
  EXPECT_FALSE(held->simulated());
  EXPECT_TRUE(held->freeze);
  const auto scaled = clothSettings(*mesh.geo, file, true, 0.25f);
  ASSERT_TRUE(scaled && scaled->simulated());
  EXPECT_NEAR(scaled->strength[0], 0.25f, 1e-6);
  // no polygons: nothing to simulate
  fv::mesh::BuiltMesh bare;
  bare.dsonVertexCount = 3;
  EXPECT_FALSE(clothSettings(bare, nullptr, true, 1.f));
}

TEST(Dforce, keepShapeRunsFromTheFilesWeightsToClothAndToTheBody)
{
  const Grid g    = grid(4, 3);
  const auto mesh = sceneMesh(g, false);
  auto       soft = std::make_shared<dson::DforceData>(*sceneMesh(g, true).dforce);
  ASSERT_GE(soft->strength.size(), 3u);
  EXPECT_EQ(clothLoosest(soft), 1.f);
  EXPECT_EQ(clothLoosest(nullptr), 1.f);
  soft->strength[0] = 0.5f;
  soft->strength[1] = 0.f;
  soft->strength[2] = 1.f;
  EXPECT_EQ(clothLoosest(soft), kClothLoosest);
  const std::shared_ptr<const dson::DforceData> file = soft;
  // the middle is the file's, untouched
  EXPECT_EQ(clothSettings(*mesh.geo, file, std::nullopt, 1.f), file);
  const auto weights = [&](float strength) {
    const auto data = clothSettings(*mesh.geo, file, std::nullopt, strength);
    return std::array<float, 3>{data->strength[0], data->strength[1], data->strength[2]};
  };
  // towards the body, and towards cloth; what the file holds stays held
  EXPECT_EQ(weights(0.5f), (std::array<float, 3>{0.25f, 0.f, 0.5f}));
  EXPECT_EQ(weights(1.5f), (std::array<float, 3>{0.75f, 0.f, 1.f}));
  EXPECT_EQ(weights(kClothLoosest), (std::array<float, 3>{1.f, 0.f, 1.f}));
  EXPECT_EQ(weights(9.f), weights(kClothLoosest));
}

TEST(Dforce, liveModeDrapesThenStepsOnlyWhileSomethingMoves)
{
  fv::SceneGraph scene;
  const int      rootNode = scene.add(fv::Node{});
  const Grid     body = grid(8, 0), veil = grid(4, 3);
  scene.meshes.push_back(sceneMesh(body, false));
  scene.meshes.push_back(sceneMesh(veil, true));
  scene.materials.push_back({});
  for(int i = 0; i < 2; i++)
  {
    fv::Node n;
    n.kind = fv::NodeKind::eMesh;
    n.mesh = i;
    scene.add(n, rootNode);
  }
  morph::MorphSet morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(scene, morphs, 0, rootNode);
  d.setMode(dforce::Mode::eLive);
  EXPECT_EQ(d.mode(), dforce::Mode::eLive);
  // live starts with a settle, written when it lands
  bool wrote = false;
  for(int i = 0; i < 6000 && !wrote; i++)
  {
    wrote = d.pump();
    if(!wrote)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(wrote);
  EXPECT_EQ(d.draped(), 1);
  const auto& pos = scene.meshes[1].geo->position;
  for(size_t v = 0; v < pos.size() / 3; v++)
  {
    EXPECT_LT(pos[v * 3 + 1], 1.0f) << v;
    EXPECT_GT(pos[v * 3 + 1], 0.0f) << v;
  }
  // nothing moves and the cloth is at rest: within a few frames the steps stop
  int writes = 0;
  for(int i = 0; i < 400; i++)
  {
    if(d.pump())
      writes++;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_LT(writes, 60);
  EXPECT_FALSE(d.busy());
  // a changed mesh builds the simulation again after the debounce, and it drapes once more
  d.changed({1}, false);
  EXPECT_TRUE(d.busy());
  wrote = false;
  for(int i = 0; i < 6000 && !wrote; i++)
  {
    wrote = d.pump();
    if(!wrote)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(wrote);
}

TEST(Dforce, aLiveBoneMoveKeepsTheFiguresCentimetres)
{
  // as the host has it: the skins bound, then the figure root scaled to metres. A bone that moves no
  // mesh goes to the simulation as a target, in the mesh's own centimetres as the build's bones are:
  // given the scaled matrices the pinned row and the body would sweep to 1/100 of where they are
  fv::SceneGraph scene;
  fv::Node       root;
  root.scale         = {0.01, 0.01, 0.01};
  const int rootNode = scene.add(root);
  fv::Node  boneNode;
  boneNode.kind  = fv::NodeKind::eBone;
  const int bone = scene.add(boneNode, rootNode);

  const Grid body = grid(8, 0), veil = grid(4, 3);
  const auto skinned = [&](fv::Mesh m) {
    const size_t n = m.geo->cagePositions.size() / 3;
    m.bones        = {bone};
    m.boneInverses = {fv::Mat4{}};
    for(size_t v = 0; v <= n; v++)
      m.cageSkin.start.push_back(int32_t(v));
    m.cageSkin.bone.assign(n, 0);
    m.cageSkin.weight.assign(n, 1.f);
    return m;
  };
  scene.meshes.push_back(skinned(sceneMesh(body, false)));
  scene.meshes.push_back(skinned(sceneMesh(veil, true)));
  // the veil's first row held on the skin
  const int pinned = 5;
  {
    auto settings = std::make_shared<fv::dson::DforceData>(*scene.meshes[1].dforce);
    for(int v = 0; v < pinned; v++)
      settings->strength[size_t(v)] = 0.f;
    scene.meshes[1].dforce = settings;
  }
  scene.materials.push_back({});
  for(int i = 0; i < 2; i++)
  {
    fv::Node n;
    n.kind = fv::NodeKind::eMesh;
    n.mesh = i;
    scene.add(n, rootNode);
  }
  // the drawn vertices run in their own order: the row is the ones at its z
  const std::vector<float> rest = scene.meshes[1].geo->position;
  std::vector<size_t>      row;
  for(size_t v = 0; v < rest.size() / 3; v++)
    if(rest[v * 3 + 2] == veil.pos[2])
      row.push_back(v);
  ASSERT_EQ(row.size(), size_t(pinned));

  morph::MorphSet morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(scene, morphs, 0, rootNode);
  d.setMode(dforce::Mode::eLive);
  bool wrote = false;
  for(int i = 0; i < 6000 && !wrote; i++)
  {
    wrote = d.pump();
    if(!wrote)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(wrote);
  for(int i = 0; i < 400 && d.busy(); i++)
  {
    d.pump();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // what is drawn is the cloth's rest shape plus its drape, skinned on top: the pinned row's drape stays
  // nothing, wherever the bone is, and the free cloth stays in the size it was
  const auto& pos    = scene.meshes[1].geo->position;
  const auto  expect = [&](const char* when) {
    for(size_t v : row)
      for(size_t c = 0; c < 3; c++)
        EXPECT_NEAR(pos[v * 3 + c], rest[v * 3 + c], 0.05f) << when << " vertex " << v;
    float least = 1e9f, most = -1e9f;
    for(size_t v = 0; v < pos.size() / 3; v++)
    {
      least = std::min(least, pos[v * 3]);
      most  = std::max(most, pos[v * 3]);
    }
    EXPECT_GT(most - least, 3.f) << when;
  };
  expect("draped");

  scene.nodes[size_t(bone)].position.x += 2;
  int writes = 0;
  for(int i = 0; i < 300; i++)
  {
    if(d.pump())
    {
      writes++;
      expect("moving");
      if(HasFailure())
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // it did step, and the steps stop again
  EXPECT_GT(writes, 0);
  expect("moved");
}

namespace {

// A veil over a floor, as liveModeDrapesThenStepsOnlyWhileSomethingMoves has them.
struct VeilScene
{
  fv::SceneGraph scene;
  int            rootNode = -1;
  VeilScene()
  {
    rootNode        = scene.add(fv::Node{});
    const Grid body = grid(8, 0), veil = grid(4, 3);
    scene.meshes.push_back(sceneMesh(body, false));
    scene.meshes.push_back(sceneMesh(veil, true));
    scene.materials.push_back({});
    for(int i = 0; i < 2; i++)
    {
      fv::Node n;
      n.kind = fv::NodeKind::eMesh;
      n.mesh = i;
      scene.add(n, rootNode);
    }
  }
};

// The placement the Transform panel gives a figure it stands on its head.
const fv::Mat4 kUpsideDown = fv::compose({}, fv::Quat{1, 0, 0, 0}, {1, 1, 1});

bool pumpUntilWritten(dforce::Dforce& d)
{
  for(int i = 0; i < 6000; i++)
  {
    if(d.pump())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

float lowestOf(const std::vector<float>& pos)
{
  float y = INFINITY;
  for(size_t v = 0; v < pos.size() / 3; v++)
    y = std::min(y, pos[v * 3 + 1]);
  return y;
}

// A simulation that never comes to rest, as a vertex caught in a fold doesn't: what's asked of it is kept.
class RestlessSim final : public dforce::ClothBackend
{
public:
  explicit RestlessSim(std::vector<dforce::SettleMesh> meshes)
      : m_meshes(std::move(meshes))
  {
  }
  void frame(float, float, float extra, float s) override
  {
    seconds.push_back(s);
    damping.push_back(extra);
  }
  void setTarget(size_t, const std::vector<fv::Mat4>&) override {}
  void setGravity(const std::array<float, 3>& g) override { gravity.push_back(g); }
  void settle() override {}
  float lastMove() const override { return 1.f; }
  bool  ownerThreadOnly() const override { return true; }
  std::vector<std::vector<float>> positions() const override
  {
    std::vector<std::vector<float>> out;
    for(const dforce::SettleMesh& m : m_meshes)
      out.push_back(m.rest);
    return out;
  }
  std::vector<float>                seconds, damping;
  std::vector<std::array<float, 3>> gravity;

private:
  std::vector<dforce::SettleMesh> m_meshes;
};

// A settle that takes so many slices, on the thread that draws, as the GPU's does: it moves the cloth.
class SlicedSim final : public dforce::ClothBackend
{
public:
  SlicedSim(std::vector<dforce::SettleMesh> meshes, int slices, bool fails)
      : m_meshes(std::move(meshes))
      , m_left(slices)
      , m_fails(fails)
  {
  }
  void  frame(float, float, float, float) override {}
  void  setTarget(size_t, const std::vector<fv::Mat4>&) override {}
  void  setGravity(const std::array<float, 3>&) override {}
  void  settle() override { ADD_FAILURE() << "a settle on the drawing thread goes in slices"; }
  bool  settleSome(double) override { return --m_left <= 0; }
  bool  failed() const override { return m_fails; }
  float lastMove() const override { return 1.f; }
  bool  ownerThreadOnly() const override { return true; }
  std::vector<std::vector<float>> positions() const override
  {
    std::vector<std::vector<float>> out;
    for(const dforce::SettleMesh& m : m_meshes)
    {
      out.push_back(m.rest);
      for(size_t v = 0; v < out.back().size() / 3; v++)
        out.back()[v * 3 + 2] += 1.f;
    }
    return out;
  }

private:
  std::vector<dforce::SettleMesh> m_meshes;
  int                             m_left;
  bool                            m_fails;
};

}  // namespace

TEST(Dforce, aFrameTakesTheTimeItIsGivenAndGravityCanTurn)
{
  // free cloth, nothing under it: a frame twice as long falls four times as far
  const auto fallOver = [](float seconds) {
    dforce::SettleOptions opts;
    opts.substeps = 24;
    dforce::ClothSim sim({cloth(grid(4, 0))}, opts);
    sim.frame(1, 1, 0, seconds);
    return -sim.positions()[0][1];
  };
  EXPECT_NEAR(fallOver(1.f / 30.f) / fallOver(1.f / 15.f), 0.25f, 0.02f);

  dforce::SettleOptions opts;
  opts.substeps = 24;
  dforce::ClothSim sim({cloth(grid(4, 0))}, opts);
  sim.setGravity({0, 980, 0});
  sim.frame(1, 1);
  EXPECT_GT(sim.positions()[0][1], 0.f);
}

TEST(Dforce, aFigureTurnedOverSettlesTheOtherWay)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  // turned in the world, the scene graph as it was: the host places a figure's entity
  d.setPlacement(kUpsideDown);
  EXPECT_TRUE(d.busy());
  ASSERT_TRUE(pumpUntilWritten(d));
  EXPECT_GT(lowestOf(v.scene.meshes[1].geo->position), 3.5f);
  // the same turn again is nothing new
  EXPECT_FALSE(d.busy());
  d.setPlacement(kUpsideDown);
  EXPECT_FALSE(d.busy());
}

TEST(Dforce, aSlowTurnAddsUpAndATurnDuringAClipWaitsForItsEnd)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  // too little to matter from one frame to the next, not over a hundred
  const auto tilted = [](double degrees) {
    const double half = degrees * 0.5 * 3.14159265358979 / 180.0;
    return fv::compose({}, fv::Quat{std::sin(half), 0, 0, std::cos(half)}, {1, 1, 1});
  };
  d.setPlacement(tilted(0.01));
  EXPECT_FALSE(d.busy());
  for(int i = 2; i <= 100 && !d.busy(); i++)
    d.setPlacement(tilted(0.01 * i));
  EXPECT_TRUE(d.busy());
  d.cancel();

  // turned over during a clip: nothing until it stops, then the settle
  bool playing = true;
  d.playing    = [&] { return playing; };
  d.setPlacement(kUpsideDown);
  EXPECT_FALSE(d.busy());
  playing = false;
  d.setPlacement(kUpsideDown);
  EXPECT_TRUE(d.busy());
}

TEST(Dforce, ACancelledSettleEndsWithinAFrame)
{
  const Grid g = grid(6, 5);
  dforce::SettleMesh cloth = dforce::SettleMesh::collider(g.pos, g.pos, g.tris, 0, 0.2f, 0.4f);
  dforce::SettleOptions opts;
  int asked = 0;
  opts.cancelled = [&] { return ++asked > 3; };
  dforce::ClothSim sim({cloth}, opts);
  sim.settle();
  EXPECT_EQ(asked, 4);
}

TEST(Dforce, ResetAllLeavesTheDrapeAndASettleBringsItBackAtAnyRate)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  d.resimulate();
  ASSERT_TRUE(pumpUntilWritten(d));
  const std::string drape = "dforce:1";
  ASSERT_TRUE(morphs.has(drape));
  EXPECT_DOUBLE_EQ(morphs.get(drape), 1.0);
  morphs.resetAll();
  EXPECT_DOUBLE_EQ(morphs.get(drape), 1.0);
  // left at 0 by whatever: a settle's target wouldn't go on, for all it ran
  morphs.setMany({{drape, 0.0}});
  d.resimulate();
  ASSERT_TRUE(pumpUntilWritten(d));
  EXPECT_DOUBLE_EQ(morphs.get(drape), 1.0);
}

TEST(Dforce, WhatChangedWhileAClipPlayedSettlesOnceItStops)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  d.resimulate();
  ASSERT_TRUE(pumpUntilWritten(d));
  bool playing = true;
  d.playing    = [&] { return playing; };
  d.poke();
  EXPECT_FALSE(d.busy());
  d.setPlacement(fv::Mat4{});
  EXPECT_FALSE(d.busy());
  playing = false;
  d.setPlacement(fv::Mat4{});  // as the host does every frame
  EXPECT_TRUE(d.busy());
  ASSERT_TRUE(pumpUntilWritten(d));
  d.setPlacement(fv::Mat4{});
  EXPECT_FALSE(d.busy());  // once
}

TEST(Dforce, aSettleOnTheGpuGoesInSlicesAndAnEditDropsItAtOnce)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  std::weak_ptr<SlicedSim> sim;
  int                      made = 0;
  d.makeGpuSim = [&](std::vector<dforce::SettleMesh> meshes, const dforce::SettleOptions&) {
    auto s = std::make_shared<SlicedSim>(std::move(meshes), 3, false);
    sim    = s;
    made++;
    return s;
  };
  d.resimulate();
  EXPECT_FALSE(d.pump());  // made, and nothing stepped on the frame that made it
  ASSERT_EQ(made, 1);
  EXPECT_TRUE(d.busy());
  EXPECT_FALSE(d.pump());
  EXPECT_FALSE(d.pump());
  EXPECT_TRUE(d.busy());
  EXPECT_TRUE(d.pump());
  EXPECT_FALSE(d.busy());
  EXPECT_TRUE(sim.expired());

  // an edit partway: the old settle goes on the next frame, and the new one doesn't wait for it to finish
  d.resimulate();
  EXPECT_FALSE(d.pump());
  ASSERT_EQ(made, 2);
  EXPECT_FALSE(d.pump());
  d.resimulate();
  EXPECT_FALSE(d.pump());
  ASSERT_EQ(made, 3);
  ASSERT_TRUE(pumpUntilWritten(d));

  // turned off partway: it goes, and nothing is written
  d.resimulate();
  EXPECT_FALSE(d.pump());
  d.setEnabled(false);
  EXPECT_TRUE(sim.expired());
  EXPECT_FALSE(d.busy());
  EXPECT_FALSE(d.pump());
}

TEST(Dforce, aSettleTheGpuCouldNotFinishGoesToTheCpu)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  d.makeGpuSim = [&](std::vector<dforce::SettleMesh> meshes, const dforce::SettleOptions&) { return std::make_shared<SlicedSim>(std::move(meshes), 1, true); };
  const auto before = v.scene.meshes[1].geo->position;
  d.resimulate();
  ASSERT_TRUE(pumpUntilWritten(d));
  EXPECT_FALSE(d.busy());
  // what the CPU's settle did with the veil, not the failed simulation's shift along z
  EXPECT_LT(lowestOf(v.scene.meshes[1].geo->position), lowestOf(before));
}

TEST(Dforce, gravityFollowsAllThePlacementHas)
{
  // what hangs plumb as drawn: under a scale that isn't uniform the direction isn't a rotation's
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  std::shared_ptr<RestlessSim> sim;
  d.makeGpuSim = [&](std::vector<dforce::SettleMesh> meshes, const dforce::SettleOptions&) {
    sim = std::make_shared<RestlessSim>(std::move(meshes));
    return sim;
  };
  auto now = std::chrono::steady_clock::now();
  d.clock  = [&] { return now; };
  d.setMode(dforce::Mode::eLiveGpu);
  ASSERT_TRUE(pumpUntilWritten(d));
  const double half = 45.0 * 0.5 * 3.14159265358979 / 180.0;
  // a 45 degree tilt about z over a scale of 2 along x, and the same mirrored
  for(const double sx : {2.0, -1.0})
  {
    const fv::Mat4 placed = fv::compose({}, fv::Quat{0, 0, std::sin(half), std::cos(half)}, {sx, 1, 1});
    d.setPlacement(placed);
    now += std::chrono::milliseconds(20);
    d.pump();
    ASSERT_FALSE(sim->gravity.empty());
    const std::array<float, 3> g = sim->gravity.back();
    EXPECT_NEAR(std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]), 980.f, 0.5f);
    // drawn through the placement it points straight down
    const double wx = placed.e[0] * g[0] + placed.e[4] * g[1] + placed.e[8] * g[2];
    const double wy = placed.e[1] * g[0] + placed.e[5] * g[1] + placed.e[9] * g[2];
    const double wz = placed.e[2] * g[0] + placed.e[6] * g[1] + placed.e[10] * g[2];
    EXPECT_NEAR(wx / wy, 0.0, 1e-4) << sx;
    EXPECT_NEAR(wz / wy, 0.0, 1e-4) << sx;
    EXPECT_LT(wy, 0.0) << sx;
  }
}

TEST(Dforce, aLiveSimulationAtRestFallsTheNewWayWhenTheFigureIsTurned)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  d.setMode(dforce::Mode::eLive);
  ASSERT_TRUE(pumpUntilWritten(d));
  const auto& pos = v.scene.meshes[1].geo->position;
  ASSERT_LT(lowestOf(pos), 1.f);
  // left to come to rest
  for(int i = 0; i < 300; i++)
  {
    d.pump();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const float rested = lowestOf(pos);
  d.setPlacement(kUpsideDown);
  // no new build: the same simulation is told
  EXPECT_FALSE(d.busy());
  for(int i = 0; i < 3000 && lowestOf(pos) < rested + 2.f; i++)
  {
    d.pump();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_GE(lowestOf(pos), rested + 2.f);
}

TEST(Dforce, aStillPoseSleepsThoughTheClothNeverRestsAndATurnWakesIt)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  std::shared_ptr<RestlessSim> sim;
  d.makeGpuSim = [&](std::vector<dforce::SettleMesh> meshes, const dforce::SettleOptions&) {
    sim = std::make_shared<RestlessSim>(std::move(meshes));
    return sim;
  };
  auto       now = std::chrono::steady_clock::now();
  d.clock        = [&] { return now; };
  const auto frames = [&](int count) {
    for(int i = 0; i < count; i++)
    {
      now += std::chrono::milliseconds(20);
      d.pump();
    }
  };
  d.setMode(dforce::Mode::eLiveGpu);
  ASSERT_TRUE(pumpUntilWritten(d));
  ASSERT_TRUE(sim);
  // the settle has held this pose already: after a step to land it, nothing more
  frames(20);
  EXPECT_LE(sim->seconds.size(), 1u);
  EXPECT_TRUE(sim->gravity.empty());

  // turned over: told once, and held for live's 3 s, then asleep again
  const size_t before = sim->seconds.size();
  d.setPlacement(kUpsideDown);
  const float held = 3.f;
  frames(400);
  ASSERT_EQ(sim->gravity.size(), 1u);
  EXPECT_NEAR(sim->gravity[0][1], 980.f, 1e-2f);
  float simulated = 0;
  for(size_t i = before; i < sim->seconds.size(); i++)
    simulated += sim->seconds[i];
  EXPECT_GE(simulated, held);
  EXPECT_LT(simulated, held + 0.1f);
  const size_t asleep = sim->seconds.size();
  frames(20);
  EXPECT_EQ(sim->seconds.size(), asleep);
}

TEST(Dforce, aStillPoseLetsTheClothMoveBeforeItCalmsIt)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  std::shared_ptr<RestlessSim> sim;
  d.makeGpuSim = [&](std::vector<dforce::SettleMesh> meshes, const dforce::SettleOptions&) {
    sim = std::make_shared<RestlessSim>(std::move(meshes));
    return sim;
  };
  auto now = std::chrono::steady_clock::now();
  d.clock  = [&] { return now; };
  d.setMode(dforce::Mode::eLiveGpu);
  ASSERT_TRUE(pumpUntilWritten(d));
  d.setPlacement(kUpsideDown);
  float simulated = 0;
  for(int i = 0; i < 200; i++)
  {
    const size_t before = sim->seconds.size();
    now += std::chrono::milliseconds(20);
    d.pump();
    if(sim->seconds.size() == before)
      continue;
    simulated += sim->seconds.back();
    if(simulated < 0.9f)
      EXPECT_EQ(sim->damping.back(), 0.f) << simulated;
    if(simulated > 2.1f)
      EXPECT_EQ(sim->damping.back(), 6.f) << simulated;
  }
  EXPECT_GT(simulated, 2.5f);
}

TEST(Dforce, aLiveStepKeepsTheClockAtAnyFrameRate)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  std::shared_ptr<RestlessSim> sim;
  d.makeGpuSim = [&](std::vector<dforce::SettleMesh> meshes, const dforce::SettleOptions&) {
    sim = std::make_shared<RestlessSim>(std::move(meshes));
    return sim;
  };
  auto now = std::chrono::steady_clock::now();
  d.clock  = [&] { return now; };
  d.setMode(dforce::Mode::eLiveGpu);
  ASSERT_TRUE(pumpUntilWritten(d));
  d.setPlacement(kUpsideDown);
  const auto after = [&](int ms) {
    now += std::chrono::milliseconds(ms);
    d.pump();
  };
  after(20);
  // a frame of 30 ms
  after(30);
  EXPECT_NEAR(sim->seconds.back(), 0.03f, 1e-3f);
  // a very slow one slows the cloth rather than taking a step its substeps can't hold together
  after(200);
  EXPECT_FLOAT_EQ(sim->seconds.back(), 1.f / 20.f);
  // at 360 fps the frames add up to a step, and in all of them the time simulated is the time gone by
  const size_t steps = sim->seconds.size();
  float        simulated = 0;
  for(int i = 0; i < 360; i++)
    after(3);
  for(size_t i = steps; i < sim->seconds.size(); i++)
    simulated += sim->seconds[i];
  EXPECT_NEAR(simulated, 360 * 0.003f, 0.03f);
  EXPECT_LE(sim->seconds.size() - steps, 60u + 1u);
}

TEST(Dforce, aTurnOnAFrameTooShortToStepIsStillTaken)
{
  VeilScene       v;
  morph::MorphSet morphs([&](int m) { return v.scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(v.scene, morphs, 0, v.rootNode);
  std::shared_ptr<RestlessSim> sim;
  d.makeGpuSim = [&](std::vector<dforce::SettleMesh> meshes, const dforce::SettleOptions&) {
    sim = std::make_shared<RestlessSim>(std::move(meshes));
    return sim;
  };
  auto now = std::chrono::steady_clock::now();
  d.clock  = [&] { return now; };
  d.setMode(dforce::Mode::eLiveGpu);
  ASSERT_TRUE(pumpUntilWritten(d));
  for(int i = 0; i < 5; i++)
  {
    now += std::chrono::milliseconds(20);
    d.pump();
  }
  const size_t asleep = sim->seconds.size();
  d.setPlacement(kUpsideDown);
  d.pump();
  EXPECT_EQ(sim->seconds.size(), asleep);
  EXPECT_TRUE(sim->gravity.empty());
  now += std::chrono::milliseconds(20);
  d.pump();
  EXPECT_EQ(sim->seconds.size(), asleep + 1);
  EXPECT_EQ(sim->gravity.size(), 1u);
}

TEST(Dforce, anEditToADrapedMeshSettlesAgain)
{
  // the host reports a frame's edits before the drape is written, so a mesh just draped that an edit
  // also moved is a reason to settle again, not an echo to ignore
  fv::SceneGraph scene;
  const int      rootNode = scene.add(fv::Node{});
  const Grid     body = grid(8, 0), veil = grid(4, 3);
  scene.meshes.push_back(sceneMesh(body, false));
  scene.meshes.push_back(sceneMesh(veil, true));
  scene.materials.push_back({});
  for(int i = 0; i < 2; i++)
  {
    fv::Node n;
    n.kind = fv::NodeKind::eMesh;
    n.mesh = i;
    scene.add(n, rootNode);
  }
  morph::MorphSet morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(scene, morphs, 0, rootNode);
  d.resimulate();
  bool wrote = false;
  for(int i = 0; i < 6000 && !wrote; i++)
  {
    wrote = d.pump();
    if(!wrote)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(wrote);
  EXPECT_FALSE(d.busy());
  d.changed({1}, false);
  EXPECT_TRUE(d.busy());
}

namespace {

// Strands side by side along +x from `root`, a point every `step` cm, at the file's usual settings
// without its pull back to the authored shape.
dforce::StrandSet strandsAlongX(fv::Vec3 root, int count, int points, float step, float offset = 0.2f)
{
  dforce::StrandSet s;
  s.start = {0};
  for(int c = 0; c < count; c++)
  {
    for(int p = 0; p < points; p++)
    {
      s.verts.push_back(int32_t(s.posed.size() / 3));
      s.posed.insert(s.posed.end(), {float(root.x) + float(p) * step, float(root.y), float(root.z) + float(c)});
    }
    s.start.push_back(int32_t(s.verts.size()));
    s.strength.push_back(1.f);
    s.offset.push_back(offset);
    s.friction.push_back(0.4f);
    s.shapeRoot.push_back(0.f);
    s.shapeTip.push_back(0.f);
    s.collide.push_back(1);
  }
  return s;
}

fv::Vec3 pointOf(const std::vector<float>& a, int v)
{
  return {a[size_t(v) * 3], a[size_t(v) * 3 + 1], a[size_t(v) * 3 + 2]};
}

float lengthOf(const std::vector<float>& a, const dforce::StrandSet& s, size_t strand)
{
  float length = 0;
  for(int k = s.start[strand]; k + 1 < s.start[strand + 1]; k++)
  {
    const fv::Vec3 p = pointOf(a, s.verts[k]), q = pointOf(a, s.verts[k + 1]);
    length += float(std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z)));
  }
  return length;
}

// A hair's file settings, standing out so that it falls by itself.
std::shared_ptr<const dson::DforceData> hairSettingsForVolume()
{
  auto d          = std::make_shared<dson::DforceData>();
  d->strandStart  = {0, 3};
  d->strandVerts  = {0, 1, 2};
  d->strandStrength = {1.f};
  d->strandsOn    = true;
  return d;
}

// A closed sphere about the origin, faces out.
dforce::SettleMesh sphere(float radius, int rings, int segments)
{
  std::vector<float>    pos;
  std::vector<uint32_t> tris;
  const float           pi = 3.14159265f;
  for(int r = 0; r <= rings; r++)
    for(int s = 0; s < segments; s++)
    {
      const float polar = pi * float(r) / float(rings), around = 2 * pi * float(s) / float(segments);
      pos.insert(pos.end(), {radius * std::sin(polar) * std::cos(around), radius * std::cos(polar), radius * std::sin(polar) * std::sin(around)});
    }
  const auto at = [&](int r, int s) { return uint32_t(r * segments + (s % segments)); };
  for(int r = 0; r < rings; r++)
    for(int s = 0; s < segments; s++)
      tris.insert(tris.end(), {at(r, s), at(r, s + 1), at(r + 1, s), at(r, s + 1), at(r + 1, s + 1), at(r + 1, s)});
  return dforce::SettleMesh::collider(pos, pos, tris, dforce::kBodyLayer, 0.2f, 0.4f);
}

}  // namespace

TEST(DforceStrands, aStrandStandingOutFallsToHangFromItsRootAtItsLength)
{
  const auto s   = strandsAlongX({0, 100, 0}, 1, 31, 1);
  const auto out = dforce::settleStrands(s, {});
  ASSERT_EQ(out.size(), s.posed.size());
  EXPECT_EQ(pointOf(out, 0).x, 0);
  EXPECT_EQ(pointOf(out, 0).y, 100);
  const fv::Vec3 tip = pointOf(out, 30);
  EXPECT_NEAR(tip.y, 70, 1.5);
  EXPECT_NEAR(tip.x, 0, 2);
  EXPECT_NEAR(lengthOf(out, s, 0), 30, 0.6);
}

TEST(DforceStrands, aStrandRestsOnAColliderByItsOffset)
{
  const Grid floor = grid(80, 0);
  const auto s     = strandsAlongX({-20, 10, 0}, 1, 41, 1, 1.f);
  const auto out   = dforce::settleStrands(s, {dforce::SettleMesh::collider(floor.pos, floor.pos, floor.tris, dforce::kBodyLayer, 0.2f, 0.4f)});
  float      lowest = INFINITY;
  for(int v = 0; v < 41; v++)
    lowest = std::min(lowest, float(pointOf(out, v).y));
  EXPECT_GT(lowest, 0.7f);
  EXPECT_LT(lowest, 1.3f);
}

TEST(DforceStrands, aStrandDrapesOverARoundColliderWithoutEnteringIt)
{
  // the signs and slopes of the distance grid over a curve, from above and past its equator
  const float radius = 10, offset = 1;
  const auto  s      = strandsAlongX({0, radius + offset + 0.2f, 0}, 1, 41, 1, offset);
  const auto  out    = dforce::settleStrands(s, {sphere(radius, 48, 48)});
  for(int v = 0; v < 41; v++)
  {
    const fv::Vec3 p = pointOf(out, v);
    EXPECT_GT(std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z), radius + offset - 0.4) << v;
  }
  EXPECT_LT(pointOf(out, 40).y, 0);
}

TEST(DforceStrands, volumeStandsTheRootsOffAColliderAtTheStrandsLength)
{
  const float radius = 10, offset = 0.5f;
  auto        s      = strandsAlongX({0, radius + offset + 0.2f, 0}, 1, 41, 1, offset);
  const auto  flat   = dforce::settleStrands(s, {sphere(radius, 48, 48)});
  s.volume           = 1;
  const auto full    = dforce::settleStrands(s, {sphere(radius, 48, 48)});
  const auto off     = [&](const std::vector<float>& a, int v) {
    const fv::Vec3 p = pointOf(a, v);
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - radius;
  };
  // the hair rides a good way further out over the top of the head
  EXPECT_GT(off(full, 12), off(flat, 12) + 2);
  for(int v = 3; v < 41; v++)
    EXPECT_GT(off(full, v), offset - 0.4) << v;
  // no longer for it, and from the same root
  EXPECT_NEAR(lengthOf(full, s, 0), lengthOf(flat, s, 0), 0.01);
  EXPECT_EQ(pointOf(full, 0).y, pointOf(s.posed, 0).y);
}

TEST(DforceStrands, volumeFadesToTheFilesOffsetAtTheTip)
{
  // lying along a floor: lifted near the root, down to its offset at the tip
  const Grid floor = grid(80, 0);
  auto       s     = strandsAlongX({-20, 10, 0}, 1, 41, 1, 1.f);
  s.volume         = 1;
  const auto out   = dforce::settleStrands(s, {dforce::SettleMesh::collider(floor.pos, floor.pos, floor.tris, dforce::kBodyLayer, 0.2f, 0.4f)});
  const float tip  = float(pointOf(out, 40).y);
  EXPECT_LT(tip, 2.f);
  EXPECT_GT(tip, 0.7f);
  EXPECT_GT(pointOf(out, 21).y, tip + 1);
}

TEST(DforceStrands, volumeHoldsAStrandTheWayItGrowsNearItsRoot)
{
  // nothing to rest on: only the hold keeps the first of it out against gravity
  auto       s    = strandsAlongX({0, 100, 0}, 1, 41, 1);
  const auto flat = dforce::settleStrands(s, {});
  s.volume        = 1;
  const auto full = dforce::settleStrands(s, {});
  EXPECT_GT(pointOf(full, 9).x, pointOf(flat, 9).x + 1);
  // and past that it hangs
  const fv::Vec3 a = pointOf(full, 30), b = pointOf(full, 40);
  EXPECT_LT(std::fabs(a.x - b.x), 2);
}

TEST(DforceStrands, theClothSwitchCarriesAHairsVolume)
{
  fv::mesh::BuiltMesh geo;
  const auto          fan = hairSettingsForVolume();
  const auto          set = clothSettings(geo, fan, std::nullopt, 1.f, 0, 0.8f);
  ASSERT_TRUE(set);
  EXPECT_NE(set, fan);
  EXPECT_NEAR(set->strandVolume, 0.8f, 1e-6);
  EXPECT_NEAR(fan->strandVolume, dson::kStrandVolume, 1e-6);
  EXPECT_TRUE(set->simulated());
  // left at its default, the file's own settings stand
  EXPECT_EQ(clothSettings(geo, fan, std::nullopt, 1.f), fan);
}

TEST(DforceStrands, aPointThatStartsInsideAColliderIsBroughtOut)
{
  const float radius = 10;
  auto        s      = strandsAlongX({0, radius + 1, 0}, 1, 11, 1, 0.5f);
  // a span's worth under the surface, within the reach it can be brought back from
  for(int v = 1; v < 11; v++)
    s.posed[size_t(v) * 3 + 1] = float(radius - 1);
  // every point simulated: a point between two follows its span, in or out
  dforce::StrandOptions opts;
  opts.spacing   = 0.5f;
  const auto out = dforce::settleStrands(s, {sphere(radius, 48, 48)}, opts);
  for(int v = 0; v < 11; v++)
  {
    const fv::Vec3 p = pointOf(out, v);
    EXPECT_GT(std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z), radius) << v;
  }
}

TEST(DforceStrands, wavesShorterThanASpanStayOnAHangingStrand)
{
  auto s = strandsAlongX({0, 100, 0}, 1, 121, 0.25f);
  for(int v = 0; v < 121; v++)
    s.posed[size_t(v) * 3 + 2] = 0.3f * std::sin(float(v) * 0.5f);
  const float before = lengthOf(s.posed, s, 0);
  const auto  out    = dforce::settleStrands(s, {});
  EXPECT_NEAR(lengthOf(out, s, 0), before, before * 0.05f);
  float widest = 0;
  for(int v = 0; v < 121; v++)
    widest = std::max(widest, std::fabs(float(pointOf(out, v).z)));
  EXPECT_GT(widest, 0.2f);
  EXPECT_LT(pointOf(out, 120).y, 75);
}

TEST(DforceStrands, strengthRunsFromThePoseToTheFall)
{
  auto       s    = strandsAlongX({0, 100, 0}, 1, 21, 1);
  const auto full = dforce::settleStrands(s, {});
  s.strength      = {0.f};
  EXPECT_EQ(dforce::settleStrands(s, {}), s.posed);
  s.strength     = {0.5f};
  const auto mid = dforce::settleStrands(s, {});
  EXPECT_NEAR(pointOf(mid, 20).y, (pointOf(s.posed, 20).y + pointOf(full, 20).y) / 2, 1e-3);
}

TEST(DforceStrands, aStrandAsStiffAsItCanBeHoldsItsShape)
{
  auto s      = strandsAlongX({0, 100, 0}, 1, 21, 1);
  s.shapeRoot = {4.f};
  s.shapeTip  = {4.f};
  const auto out = dforce::settleStrands(s, {});
  EXPECT_NEAR(pointOf(out, 20).x, 20, 0.2);
  EXPECT_NEAR(pointOf(out, 20).y, 100, 0.2);
}

TEST(DforceStrands, theSettleEndsOnceTheStrandsAreStill)
{
  // already hanging: nothing to wait for, whatever the longest it may take
  auto s = strandsAlongX({0, 100, 0}, 8, 21, 1);
  for(size_t v = 0; v < s.posed.size() / 3; v++)
  {
    s.posed[v * 3 + 1] = 100 - s.posed[v * 3];
    s.posed[v * 3]     = 0;
  }
  int                   frames = 0;
  dforce::StrandOptions opts;
  opts.cancelled = [&] { frames++; return false; };
  dforce::settleStrands(s, {}, opts);
  EXPECT_LT(frames, opts.maxFrames / 2);
}

TEST(DforceStrands, anEditEndsTheSettleWithinAFrame)
{
  const auto            s = strandsAlongX({0, 100, 0}, 1, 21, 1);
  dforce::StrandOptions opts;
  opts.cancelled = [] { return true; };
  const auto out = dforce::settleStrands(s, {}, opts);
  EXPECT_NEAR(pointOf(out, 20).y, 100, 1e-3);
}

TEST(DforceStrands, aCollideOffSurfacePassesThrough)
{
  const Grid floor = grid(80, 0);
  auto       s     = strandsAlongX({-20, 10, 0}, 1, 41, 1);
  s.collide        = {0};
  const auto out   = dforce::settleStrands(s, {dforce::SettleMesh::collider(floor.pos, floor.pos, floor.tris, dforce::kBodyLayer, 0.2f, 0.4f)});
  EXPECT_LT(pointOf(out, 40).y, -20);
}

// Toyen's dForce Long Hair stands straight out; hair styled up or hanging is left as it is.
TEST(DforceStrands, onlyLongStrandsThatDontHangAreUnsettled)
{
  const auto of = [](float length, float dx, float dy) {
    std::vector<float>   pos;
    std::vector<int32_t> start{0}, verts;
    for(int c = 0; c < 4; c++)
    {
      for(int p = 0; p <= 10; p++)
      {
        verts.push_back(int32_t(pos.size() / 3));
        const float t = float(p) / 10 * length;
        pos.insert(pos.end(), {dx * t, 150 + dy * t, float(c)});
      }
      start.push_back(int32_t(verts.size()));
    }
    return dson::strandsUnsettled(pos, start, verts);
  };
  EXPECT_TRUE(of(58, 1, 0));
  EXPECT_FALSE(of(58, 0, -1));
  EXPECT_FALSE(of(8, 1, 0));
  EXPECT_FALSE(of(18, 1, 0));
  EXPECT_FALSE(dson::strandsUnsettled({}, {}, {}));
}

namespace {

// A hair of two strands on a node with a simulation modifier, one on each of two surfaces, standing
// straight out (`dy` 0) or hanging (`dy` -1).
std::shared_ptr<dson::DforceData> hairSettings(float dy)
{
  const auto scene = dson::parseDoc(R"({
    "scene": {"modifiers": [
      {"id": "sim", "url": "#dyn", "parent": "#Hair", "extra": [{"type": "studio/modifier/dynamic_simulation"}]},
      {"id": "long", "url": "#set", "parent": "#Hair-1", "groups": ["Long"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation"}],
       "channel": {"id": "Collision Offset", "current_value": 2.4}},
      {"id": "short", "url": "#set", "parent": "#Hair-1", "groups": ["Short"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation"}],
       "channel": {"id": "Collision Offset", "current_value": 0.2}}
    ]},
    "modifier_library": [
      {"id": "dyn", "extra": [{"type": "studio/modifier/dynamic_simulation", "channels": [
        {"channel": {"id": "Simulation Object Type", "current_value": 1}}]}]},
      {"id": "set", "extra": [{"type": "studio/simulation_settings/dynamic_simulation", "channels": [
        {"channel": {"id": "Friction", "current_value": 0.3}},
        {"channel": {"id": "Local Shape Constraint Stiffness", "current_value": 0.01}},
        {"channel": {"id": "Local Shape Constraint Tip Stiffness", "current_value": 0.05}}]}]}
    ]
  })");
  // two strands of 40 cm over three points, on the two groups
  const std::string y = std::to_string(100 + 20 * dy), tip = std::to_string(100 + 40 * dy);
  const std::string x1 = dy < 0 ? "0" : "20", x2 = dy < 0 ? "0" : "40";
  const auto geo = dson::parseDoc(R"({
    "vertices": {"count": 6, "values": [[0,100,0],[)" + x1 + "," + y + R"(,0],[)" + x2 + "," + tip + R"(,0],[0,100,1],[)" + x1 + "," + y + R"(,1],[)" + x2 + "," + tip + R"(,1]]},
    "polylist": {"count": 0, "values": []},
    "polyline_list": {"count": 2, "values": [[0, 0, 0, 1, 2], [0, 1, 3, 4, 5]]},
    "polygon_material_groups": {"count": 2, "values": ["Long", "Short"]}
  })");
  return dson::buildDforce(*scene, nullptr, "Hair", "Hair-1", *geo, geo->json, {});
}

}  // namespace

TEST(DforceStrands, readsEachStrandsSettingsFromItsSurface)
{
  const auto fan = hairSettings(0);
  ASSERT_TRUE(fan);
  EXPECT_FALSE(fan->hasPolygons());
  ASSERT_TRUE(fan->hasStrands());
  ASSERT_EQ(fan->strandStart, (std::vector<int32_t>{0, 3, 6}));
  EXPECT_EQ(fan->strandVerts, (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));
  ASSERT_EQ(fan->strandOffset.size(), 2u);
  EXPECT_NEAR(fan->strandOffset[0], 2.4f, 1e-6);
  EXPECT_NEAR(fan->strandOffset[1], 0.2f, 1e-6);
  EXPECT_NEAR(fan->strandFriction[0], 0.3f, 1e-6);
  EXPECT_NEAR(fan->strandShapeRoot[1], 0.01f, 1e-6);
  EXPECT_NEAR(fan->strandShapeTip[1], 0.05f, 1e-6);
  // standing out: it falls by itself. Hanging already: its file's style stays.
  EXPECT_TRUE(fan->simulated());
  const auto hanging = hairSettings(-1);
  ASSERT_TRUE(hanging && hanging->hasStrands());
  EXPECT_FALSE(hanging->simulated());
}

TEST(DforceStrands, theClothSwitchLetsStrandsFallOrHoldsThem)
{
  fv::mesh::BuiltMesh geo;
  const auto          hanging = hairSettings(-1);
  EXPECT_EQ(clothSettings(geo, hanging, std::nullopt, 1.f), hanging);
  const auto on = clothSettings(geo, hanging, true, 0.5f);
  ASSERT_TRUE(on);
  EXPECT_TRUE(on->simulated());
  EXPECT_NEAR(on->strandStrength[0], hanging->strandStrength[0] * 0.5f, 1e-6);

  const auto fan = hairSettings(0);
  EXPECT_EQ(clothSettings(geo, fan, true, 1.f), fan);
  const auto off = clothSettings(geo, fan, false, 1.f);
  ASSERT_TRUE(off);
  EXPECT_FALSE(off->simulated());
}

TEST(DforceStrands, aHairInTheSceneComesDownAsItsDrape)
{
  // through the whole job: picked out of the scene, solved on the pool, landed as the mesh's morph
  fv::SceneGraph scene;
  const int      rootNode = scene.add(fv::Node{});
  const Grid     body     = grid(8, 0);
  scene.meshes.push_back(sceneMesh(body, false));

  const auto            s = strandsAlongX({-4, 30, 0}, 2, 11, 2);
  fv::mesh::MeshInput   in;
  in.positions = s.posed;
  fv::mesh::Lines lines;
  lines.start    = s.start;
  lines.verts    = s.verts;
  lines.material = {0, 0};
  in.lines       = std::move(lines);
  fv::Mesh hair;
  hair.geo            = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(in));
  hair.materialGroups = {"Default"};
  hair.materials      = {0};
  auto data           = std::make_shared<dson::DforceData>();
  data->strandStart     = s.start;
  data->strandVerts     = s.verts;
  data->strandStrength  = s.strength;
  data->strandOffset    = s.offset;
  data->strandFriction  = s.friction;
  data->strandShapeRoot = s.shapeRoot;
  data->strandShapeTip  = s.shapeTip;
  data->strandCollide   = s.collide;
  data->strandsOn       = true;
  hair.dforce           = data;
  scene.meshes.push_back(std::move(hair));
  scene.materials.push_back({});
  for(int i = 0; i < 2; i++)
  {
    fv::Node n;
    n.kind = fv::NodeKind::eMesh;
    n.mesh = i;
    scene.add(n, rootNode);
  }
  morph::MorphSet morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(scene, morphs, 0, rootNode);
  ASSERT_TRUE(d.active());
  d.resimulate();
  ASSERT_TRUE(pumpUntilWritten(d));
  EXPECT_EQ(d.draped(), 1);
  // 20 cm of strand from a root at 30, so its tip is down about 10
  EXPECT_LT(lowestOf(scene.meshes[1].geo->position), 12.f);

  // its switch off: no settle will come, as nothing else drapes, and it goes back as it was
  auto held       = std::make_shared<dson::DforceData>(*data);
  held->strandsOn = false;
  scene.meshes[1].dforce = held;
  ASSERT_FALSE(d.active());
  d.changed({1}, false);
  EXPECT_EQ(d.draped(), 0);
  EXPECT_GT(lowestOf(scene.meshes[1].geo->position), 29.f);
}

namespace {

// A strand along +x with waves up and down of `amplitude` cm every 12 cm, longer than a span.
dforce::StrandSet wavyStrand(float amplitude)
{
  auto s = strandsAlongX({0, 100, 0}, 1, 41, 1);
  for(int v = 0; v < 41; v++)
    s.posed[size_t(v) * 3 + 1] = 100 + amplitude * std::sin(float(v) * 2 * 3.14159265f / 12);
  return s;
}

float meanMove(const std::vector<float>& a, const std::vector<float>& b)
{
  float sum = 0;
  for(size_t v = 0; v < a.size() / 3; v++)
    sum += std::sqrt((a[v * 3] - b[v * 3]) * (a[v * 3] - b[v * 3]) + (a[v * 3 + 1] - b[v * 3 + 1]) * (a[v * 3 + 1] - b[v * 3 + 1]) + (a[v * 3 + 2] - b[v * 3 + 2]) * (a[v * 3 + 2] - b[v * 3 + 2]));
  return sum / float(a.size() / 3);
}

// How much a strand still turns from span to span, radians over its length: its curls.
float curling(const std::vector<float>& a, int from, int to, int stride)
{
  float turned = 0;
  for(int v = from; v + 2 * stride <= to; v += stride)
  {
    const fv::Vec3 p = pointOf(a, v), q = pointOf(a, v + stride), r = pointOf(a, v + 2 * stride);
    const double   ax = q.x - p.x, ay = q.y - p.y, az = q.z - p.z, bx = r.x - q.x, by = r.y - q.y, bz = r.z - q.z;
    const double   c  = (ax * bx + ay * by + az * bz) / std::sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz));
    turned += float(std::acos(std::clamp(c, -1.0, 1.0)));
  }
  return turned;
}

}  // namespace

TEST(DforceStrands, styledHairKeepsItsCurlsWhereUnsettledHairHangsStraight)
{
  auto       s      = wavyStrand(1.5f);
  const float before = curling(s.posed, 0, 40, 3);
  ASSERT_GT(before, 3.f);
  const auto fallen = dforce::settleStrands(s, {});
  EXPECT_LT(curling(fallen, 0, 40, 3), before * 0.5f);
  s.styled       = true;
  s.volume       = 0.5f;
  const auto kept = dforce::settleStrands(s, {});
  EXPECT_GT(curling(kept, 0, 40, 3), before * 0.7f);
}

TEST(DforceStrands, onStyledHairMoreVolumeMeansCloserToTheStyle)
{
  auto s   = wavyStrand(1.5f);
  s.styled = true;
  float last = INFINITY;
  for(const float volume : {0.f, 0.5f, 1.f})
  {
    s.volume         = volume;
    const float move = meanMove(dforce::settleStrands(s, {}), s.posed);
    EXPECT_LT(move, last) << volume;
    last = move;
  }
  // at full it has barely drooped for a strand held straight out, but at none it has come down a long way
  EXPECT_LT(last, 3.f);
  s.volume = 0;
  EXPECT_GT(meanMove(dforce::settleStrands(s, {}), s.posed), 5.f);
  // and it keeps its length whatever the Volume
  s.volume = 0.5f;
  const auto out = dforce::settleStrands(s, {});
  EXPECT_NEAR(lengthOf(out, s, 0), lengthOf(s.posed, s, 0), lengthOf(s.posed, s, 0) * 0.01f);
}

TEST(DforceStrands, aHairStandingOutIsNotStyledAndOneThatHangsIs)
{
  const auto fan = hairSettings(0), hanging = hairSettings(-1);
  ASSERT_TRUE(fan && hanging);
  EXPECT_FALSE(fan->strandStyled);
  EXPECT_TRUE(hanging->strandStyled);
}