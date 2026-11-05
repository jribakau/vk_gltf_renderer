// The cloth island and the grid it meets faces by (docs/physics-plan.md step 12). How cloth drapes is
// tested through dForce's names in test_dforce.cpp, which build on the island; these cover what the
// move made: the island's own interface, the seam dForce runs it through, and the shared grid.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include <figure/dforce/solver.hpp>
#include <figure/physics/cloth/island.hpp>
#include <figure/physics/collide/tri_grid.hpp>

namespace {

using namespace fv;
using physics::cloth::Island;
using physics::cloth::Surface;

// A strip of `n` quads, 2 cm each, hanging in the xy plane from its top row, which is pinned and skinned
// to one bone: `bone` is that bone's target (the identity: where it is).
Surface strip(int n, const Mat4& bone)
{
  Surface m;
  for(int i = 0; i <= n; i++)
    for(int j = 0; j <= 1; j++)
      m.rest.insert(m.rest.end(), {2.f * float(j), -2.f * float(i), 0.f});
  const auto at = [](int i, int j) { return uint32_t(i * 2 + j); };
  for(int i = 0; i < n; i++)
  {
    m.tris.insert(m.tris.end(), {at(i, 0), at(i + 1, 0), at(i, 1), at(i, 1), at(i + 1, 0), at(i + 1, 1)});
    m.polyVerts.insert(m.polyVerts.end(), {int32_t(at(i, 0)), int32_t(at(i + 1, 0)), int32_t(at(i + 1, 1)), int32_t(at(i, 1))});
    m.polyStart.push_back(int32_t(m.polyVerts.size()));
  }
  const size_t v = m.rest.size() / 3;
  m.kind         = Surface::Kind::eDynamic;
  m.strength.assign(v, 1.f);
  m.strength[at(0, 0)] = m.strength[at(0, 1)] = 0;
  m.stretch.assign(v, 0.8f);
  m.shear.assign(v, 0.2f);
  m.bend.assign(v, 0.5f);
  m.buckling.assign(v, 0.05f);
  m.bucklingRatio.assign(v, 0.7f);
  m.density.assign(v, 180.f);
  m.contraction.assign(v, 1.f);
  m.damping.assign(v, 0.1f);
  m.friction.assign(v, 0.4f);
  m.offset.assign(v, 0.2f);
  m.layer.assign(v, 0);
  m.selfCollide.assign(v, 0);
  m.collide.assign(v, 1);
  m.visible.assign(v, 1);
  m.gravity = {0, -980, 0};
  m.skinIndex.assign(v * 4, 0);
  m.skinWeight.assign(v * 4, 0.f);
  for(size_t k = 0; k < v; k++)
    m.skinWeight[k * 4] = 1;
  m.bones     = {bone};
  m.restLocal = m.rest;
  m.posed.resize(m.rest.size());
  physics::cloth::skin(m, m.bones, m.posed.data());
  return m;
}

Mat4 shifted(double x)
{
  return compose(Vec3{x, 0, 0}, Quat{}, Vec3{1, 1, 1});
}

// The pinned top row's x, from positions of the strip.
std::vector<float> pinnedX(const std::vector<float>& p)
{
  return {p[0], p[3]};
}

}  // namespace

TEST(Cloth, aTargetMovesWhatThePoseHoldsAndOneItCantTakeIsRefused)
{
  // settled with its bone 5 cm along x, then sent back: the pinned row follows the new target. A
  // target with another number of bones, or for a surface there isn't, changes nothing, and the
  // island says so
  Island island({strip(4, shifted(5))});
  island.settle();
  const auto settled = pinnedX(island.positions()[0]);
  EXPECT_NEAR(settled[0], 5, 1e-4) << "settled at the target";
  EXPECT_NEAR(settled[1], 7, 1e-4);
  EXPECT_FALSE(island.setTarget(0, {Mat4{}, Mat4{}}));
  EXPECT_FALSE(island.setTarget(1, {Mat4{}}));
  EXPECT_TRUE(island.setTarget(0, {Mat4{}}));
  island.frame(0, 1);
  const auto back = pinnedX(island.positions()[0]);
  EXPECT_NEAR(back[0], 0, 1e-4);
  EXPECT_NEAR(back[1], 2, 1e-4);
}

TEST(Cloth, dforcesSimIsTheIslandBehindItsSeam)
{
  // dForce's CPU simulation is the island: the same settle to the bit. A target the island refuses
  // leaves the seam's next step holding the pose it reached, one it takes sweeps to it
  const Surface s = strip(6, shifted(5));
  Island        island({s});
  island.settle();
  dforce::ClothSim sim({s});
  sim.settle();
  EXPECT_EQ(island.positions(), sim.positions());

  sim.setTarget(0, {Mat4{}, Mat4{}});
  sim.step();
  const auto held = pinnedX(sim.positions()[0]);
  EXPECT_NEAR(held[0], 5, 1e-4) << "a refused target: it held";
  EXPECT_NEAR(held[1], 7, 1e-4);
  sim.setTarget(0, {Mat4{}});
  sim.step();
  const auto back = pinnedX(sim.positions()[0]);
  EXPECT_NEAR(back[0], 0, 1e-4) << "a target taken: it swept";
  EXPECT_NEAR(back[1], 2, 1e-4);
}

TEST(TriGrid, aQueryReachesEveryFaceWhoseGrownBoxMeetsIt)
{
  // faces scattered over a 40 cm box, some of them in the grid: a query finds every one of those whose
  // box, grown by the pad, meets the query's (it may name a face more than once, or one in a cell it
  // touches that doesn't meet it), and none that isn't in the grid
  struct Face
  {
    int     a, b, c, mesh;
    uint8_t self;
  };
  std::vector<float> at;
  std::vector<Face>  faces;
  uint32_t           seed = 12345;
  const auto         next = [&] {
    seed = seed * 1664525u + 1013904223u;
    return float(seed >> 8) / float(1 << 24);
  };
  for(int t = 0; t < 300; t++)
  {
    const float cx = next() * 40 - 20, cy = next() * 40 - 20, cz = next() * 40 - 20;
    for(int k = 0; k < 3; k++)
      at.insert(at.end(), {cx + next() * 3, cy + next() * 3, cz + next() * 3});
    faces.push_back({t * 3, t * 3 + 1, t * 3 + 2, t % 2, uint8_t(t % 3 == 0)});
  }
  std::vector<int> ids;
  for(int t = 0; t < 300; t += 2)
    ids.push_back(t);
  const float      pad = 0.25f;
  physics::TriGrid grid;
  grid.build(faces, ids, at, at, pad);
  ASSERT_FALSE(grid.empty());
  for(int q = 0; q < 40; q++)
  {
    const float lo[3] = {next() * 40 - 22, next() * 40 - 22, next() * 40 - 22};
    const float hi[3] = {lo[0] + next() * 6, lo[1] + next() * 6, lo[2] + next() * 6};
    std::set<int> found;
    grid.query(lo, hi, [&](const physics::TriGrid::Item& item) {
      found.insert(item.tri);
      const Face& f = faces[size_t(item.tri)];
      EXPECT_EQ(item.a, f.a);
      EXPECT_EQ(item.mesh, f.mesh);
      EXPECT_EQ(item.self, f.self);
    });
    for(const int t : found)
      EXPECT_EQ(t % 2, 0) << "face " << t << " isn't in the grid";
    for(const int t : ids)
    {
      bool meets = true;
      for(int c = 0; c < 3; c++)
      {
        const float flo = std::min({at[size_t(faces[size_t(t)].a) * 3 + c], at[size_t(faces[size_t(t)].b) * 3 + c], at[size_t(faces[size_t(t)].c) * 3 + c]}) - pad;
        const float fhi = std::max({at[size_t(faces[size_t(t)].a) * 3 + c], at[size_t(faces[size_t(t)].b) * 3 + c], at[size_t(faces[size_t(t)].c) * 3 + c]}) + pad;
        meets = meets && flo <= hi[c] && fhi >= lo[c];
      }
      if(meets)
        EXPECT_TRUE(found.count(t)) << "query " << q << " missed face " << t;
    }
  }
  physics::TriGrid none;
  none.build(faces, {}, at, at, pad);
  EXPECT_TRUE(none.empty());
}
