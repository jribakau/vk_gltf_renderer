// The cloth island and the grid it meets faces by (docs/physics-plan.md step 12). How cloth drapes is
// tested through dForce's names in test_dforce.cpp, which build on the island; these cover what the
// move made: the island's own interface, the seam dForce runs it through, and the shared grid. And for
// the GPU cloth bench (step 13a): the job file it replays, and the clusters the GPU solves in.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include <filesystem>
#include <fstream>
#include <map>

#include <figure/dforce/solver.hpp>
#include <figure/physics/cloth/clusters.hpp>
#include <figure/physics/cloth/island.hpp>
#include <figure/physics/cloth/job.hpp>
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

// A sheet of n x n quads, 1 cm each, hanging in the xy plane: its top row pinned, the row under it
// tethered at half strength, the rest free.
Surface sheet(int n)
{
  Surface m;
  for(int i = 0; i <= n; i++)
    for(int j = 0; j <= n; j++)
      m.rest.insert(m.rest.end(), {float(j), -float(i), 0.f});
  m.posed    = m.rest;
  const auto at = [&](int i, int j) { return uint32_t(i * (n + 1) + j); };
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
    {
      m.tris.insert(m.tris.end(), {at(i, j), at(i + 1, j), at(i, j + 1), at(i, j + 1), at(i + 1, j), at(i + 1, j + 1)});
      m.polyVerts.insert(m.polyVerts.end(), {int32_t(at(i, j)), int32_t(at(i + 1, j)), int32_t(at(i + 1, j + 1)), int32_t(at(i, j + 1))});
      m.polyStart.push_back(int32_t(m.polyVerts.size()));
    }
  const size_t v = m.rest.size() / 3;
  m.kind         = Surface::Kind::eDynamic;
  m.strength.assign(v, 1.f);
  for(int j = 0; j <= n; j++)
  {
    m.strength[at(0, j)] = 0;
    m.strength[at(1, j)] = 0.5f;
  }
  for(auto* f : {&m.stretch, &m.shear, &m.bend})
    f->assign(v, 0.5f);
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
  return m;
}

// What every cut must keep, halo or not: the free vertices split among the clusters, each row with a
// free end run where it should be, and no colour writing a vertex twice.
void expectAWholeCut(const physics::cloth::Model& model, const physics::cloth::Clusters& c, bool halo)
{
  using physics::cloth::DistanceKind;
  ASSERT_GT(c.count(), 1);
  std::vector<int> clusterOf(size_t(model.nv), -1);
  for(int k = 0; k < c.count(); k++)
    for(int l = 0; l < c.owned[size_t(k)]; l++)
    {
      const int v = c.verts[size_t(c.vertStart[size_t(k)] + l)];
      EXPECT_GT(model.inv[size_t(v)], 0) << "an owned vertex is free";
      EXPECT_EQ(clusterOf[size_t(v)], -1) << "vertex " << v << " in two clusters";
      clusterOf[size_t(v)] = k;
    }
  for(int v : model.freeVerts)
    EXPECT_GE(clusterOf[size_t(v)], 0) << "free vertex " << v << " in no cluster";
  // the rows each cluster runs, by model row, and that no colour writes a vertex twice
  std::map<int, int> held;
  const int          colours = c.colours();
  for(int k = 0; k < c.count(); k++)
  {
    const int first = c.vertStart[size_t(k)], owned = c.owned[size_t(k)];
    for(int l = owned; l < c.vertStart[size_t(k) + 1] - first; l++)
    {
      const int v = c.verts[size_t(first + l)];
      if(halo && model.inv[size_t(v)] > 0)
        EXPECT_NE(clusterOf[size_t(v)], k) << "a halo vertex is another cluster's";
      else
        EXPECT_EQ(model.inv[size_t(v)], 0) << "a ghost is pinned";
    }
    for(int col = 0; col < colours; col++)
    {
      std::set<int> written;
      for(int r = c.colourStart[size_t(k * colours + col)]; r < c.colourStart[size_t(k * colours + col + 1)]; r++)
      {
        const auto& row = c.rows[size_t(r)];
        const auto& e   = model.edges[size_t(row.edge)];
        EXPECT_EQ(c.verts[size_t(first + row.a)], e.a);
        EXPECT_EQ(c.verts[size_t(first + row.b)], e.b);
        const DistanceKind want = col < c.stretchColours ? DistanceKind::eStretch : col < c.stretchColours + c.reachColours ? DistanceKind::eReach : e.kind;
        EXPECT_EQ(e.kind, want) << "a row in another group's colour";
        if(col >= c.stretchColours + c.reachColours)
          EXPECT_TRUE(e.kind == DistanceKind::eShear || e.kind == DistanceKind::eBend);
        for(const int l : {int(row.a), int(row.b)})
          if(l < owned)
            EXPECT_TRUE(written.insert(l).second) << "cluster " << k << " colour " << col << " writes local " << l << " twice";
        held[row.edge]++;
      }
    }
  }
  for(size_t r = 0; r < model.edges.size(); r++)
  {
    const auto& e  = model.edges[r];
    const bool  fa = model.inv[size_t(e.a)] > 0, fb = model.inv[size_t(e.b)] > 0;
    if(!fa && !fb)
    {
      EXPECT_EQ(held.count(int(r)), 0u);
      continue;
    }
    const bool crosses = fa && fb && clusterOf[size_t(e.a)] != clusterOf[size_t(e.b)];
    if(!crosses)
      EXPECT_EQ(held[int(r)], 1) << "row " << r;
    else if(halo)
      EXPECT_EQ(held[int(r)], 2) << "a crossing row is held by both its clusters";
    else
      EXPECT_EQ(held.count(int(r)), 0u) << "a crossing row runs after the clusters";
  }
  // the crossing rows: each once, no colour writing a vertex twice
  std::set<int> crossing;
  for(size_t col = 0; col + 1 < c.crossingStart.size(); col++)
  {
    std::set<int> written;
    for(int k = c.crossingStart[col]; k < c.crossingStart[col + 1]; k++)
    {
      const int r = c.crossing[size_t(k)];
      EXPECT_TRUE(crossing.insert(r).second);
      EXPECT_TRUE(written.insert(model.edges[size_t(r)].a).second);
      EXPECT_TRUE(written.insert(model.edges[size_t(r)].b).second);
    }
  }
  if(halo)
    EXPECT_TRUE(crossing.empty());
  // each tethered vertex once, by the cluster that owns it
  std::set<int> soft;
  for(int k = 0; k < c.count(); k++)
    for(int s = c.softStart[size_t(k)]; s < c.softStart[size_t(k) + 1]; s++)
    {
      const auto [local, index] = c.soft[size_t(s)];
      EXPECT_LT(int(local), c.owned[size_t(k)]);
      EXPECT_EQ(c.verts[size_t(c.vertStart[size_t(k)] + local)], model.softVerts[size_t(index)]);
      EXPECT_TRUE(soft.insert(index).second);
    }
  EXPECT_EQ(soft.size(), model.softVerts.size());
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

TEST(ClothClusters, theCutSplitsTheFreeVerticesAndRunsEveryRowOnce)
{
  // a 40 cm sheet in clusters of about 200 free vertices: every free vertex in one, every row with a
  // free end in the cluster of its free ends or among the crossing rows, each colour writing a vertex
  // once, the clusters as compact as the patches they grow (few rows cross), and the same every time
  const auto model = physics::cloth::Model::build({sheet(40)}, 0.15f);
  const auto c     = physics::cloth::buildClusters(model, 200, 2048);
  expectAWholeCut(model, c, false);
  EXPECT_LE(c.mostLocal, 2048);
  EXPECT_LT(double(c.crossing.size()), 0.25 * double(model.edges.size())) << "compact patches";
  const auto again = physics::cloth::buildClusters(model, 200, 2048);
  EXPECT_EQ(again.verts, c.verts);
  EXPECT_EQ(again.colourStart, c.colourStart);
  EXPECT_EQ(again.crossing, c.crossing);
}

TEST(ClothClusters, withAHaloACrossingRowIsHeldByBothItsClusters)
{
  const auto model = physics::cloth::Model::build({sheet(40)}, 0.15f);
  const auto c     = physics::cloth::buildClusters(model, 200, 2048, true);
  expectAWholeCut(model, c, true);
  EXPECT_GT(c.haloRows, 0u);
  // as many rows cross as without a halo, now held by the clusters
  EXPECT_EQ(c.haloRows, physics::cloth::buildClusters(model, 200, 2048).crossing.size());
}

TEST(ClothClusters, smallPiecesShareACluster)
{
  // 100 cards of one quad each, as a cloth hair is many: they fill clusters together, not one each
  Surface cards = sheet(1);
  Surface one   = cards;
  cards.rest.clear();
  cards.tris.clear();
  cards.polyVerts.clear();
  cards.polyStart = {0};
  for(auto* f : {&cards.strength, &cards.stretch, &cards.shear, &cards.bend, &cards.buckling, &cards.bucklingRatio, &cards.density, &cards.contraction,
                 &cards.damping, &cards.friction, &cards.offset})
    f->clear();
  cards.layer.clear();
  cards.selfCollide.clear();
  cards.collide.clear();
  cards.visible.clear();
  for(int k = 0; k < 100; k++)
  {
    const uint32_t base = uint32_t(cards.rest.size() / 3);
    for(size_t v = 0; v < one.rest.size() / 3; v++)
    {
      cards.rest.insert(cards.rest.end(), {one.rest[v * 3] + 3.f * float(k), one.rest[v * 3 + 1], one.rest[v * 3 + 2]});
      // all free: a card's top isn't pinned here
      cards.strength.push_back(1.f);
      for(auto* f : {&cards.stretch, &cards.shear, &cards.bend})
        f->push_back(0.5f);
      cards.buckling.push_back(0.05f);
      cards.bucklingRatio.push_back(0.7f);
      cards.density.push_back(180.f);
      cards.contraction.push_back(1.f);
      cards.damping.push_back(0.1f);
      cards.friction.push_back(0.4f);
      cards.offset.push_back(0.2f);
      cards.layer.push_back(0);
      cards.selfCollide.push_back(0);
      cards.collide.push_back(1);
      cards.visible.push_back(1);
    }
    for(uint32_t t : one.tris)
      cards.tris.push_back(base + t);
    for(int32_t p : one.polyVerts)
      cards.polyVerts.push_back(int32_t(base) + p);
    cards.polyStart.push_back(int32_t(cards.polyVerts.size()));
  }
  cards.posed = cards.rest;
  const auto model = physics::cloth::Model::build({cards}, 0.15f);
  ASSERT_EQ(model.freeVerts.size(), 400u);
  const auto c = physics::cloth::buildClusters(model, 128, 2048);
  expectAWholeCut(model, c, false);
  EXPECT_LE(c.count(), 4);
}

TEST(ClothClusters, aClusterStopsBeforeItsGhostsTakeItPastTheMost)
{
  // the sheet's pinned top row is ghosts to the clusters along it: none holds more than the most asked
  const auto model = physics::cloth::Model::build({sheet(40)}, 0.15f);
  const auto c     = physics::cloth::buildClusters(model, 2000, 300);
  expectAWholeCut(model, c, false);
  EXPECT_LE(c.mostLocal, 300);
}

TEST(ClothJob, aJobReadsBackAsItWasWritten)
{
  physics::cloth::Job job;
  job.surfaces = {strip(3, shifted(2)), sheet(4)};
  job.surfaces[1].kind = Surface::Kind::eAddon;
  job.options.stretchSweeps = 3;
  job.options.transition    = 0.5f;
  job.options.recoverLost   = false;
  job.drapes                = {{1, "the sheet"}};
  const auto file           = std::filesystem::temp_directory_path() / "fv-cloth-job-test.job";
  ASSERT_TRUE(physics::cloth::writeJob(file, job));
  const auto back = physics::cloth::readJob(file);
  ASSERT_TRUE(back.has_value());
  ASSERT_EQ(back->surfaces.size(), 2u);
  for(size_t k = 0; k < 2; k++)
  {
    const Surface &a = job.surfaces[k], &b = back->surfaces[k];
    EXPECT_EQ(a.rest, b.rest);
    EXPECT_EQ(a.posed, b.posed);
    EXPECT_EQ(a.tris, b.tris);
    EXPECT_EQ(a.polyVerts, b.polyVerts);
    EXPECT_EQ(a.kind, b.kind);
    EXPECT_EQ(a.strength, b.strength);
    EXPECT_EQ(a.layer, b.layer);
    EXPECT_EQ(a.gravity, b.gravity);
    EXPECT_EQ(a.skinIndex, b.skinIndex);
    EXPECT_EQ(a.skinWeight, b.skinWeight);
    ASSERT_EQ(a.bones.size(), b.bones.size());
    for(size_t i = 0; i < a.bones.size(); i++)
      EXPECT_EQ(a.bones[i].e, b.bones[i].e);
    EXPECT_EQ(a.toSpace.e, b.toSpace.e);
  }
  EXPECT_EQ(back->options.stretchSweeps, 3);
  EXPECT_EQ(back->options.transition, 0.5f);
  EXPECT_FALSE(back->options.recoverLost);
  EXPECT_EQ(back->drapes, job.drapes);
  // and it drapes the same as what it was written from, to the bit
  EXPECT_EQ(physics::cloth::settle(job.surfaces, job.options), physics::cloth::settle(back->surfaces, back->options));

  // a file that isn't a job isn't read
  {
    std::ofstream(file, std::ios::binary) << "not a cloth job";
  }
  EXPECT_FALSE(physics::cloth::readJob(file).has_value());
  std::filesystem::remove(file);
}
