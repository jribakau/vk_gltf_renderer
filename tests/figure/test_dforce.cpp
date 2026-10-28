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
  EXPECT_EQ(sim.recovered(), 0);
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
    t->indices = {1};
    t->deltas  = {0, dy, 0};
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
