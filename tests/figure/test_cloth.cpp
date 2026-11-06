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
#include <random>

#include <figure/physics/cloth/backend.hpp>
#include <figure/physics/cloth/clusters.hpp>
#include <figure/physics/cloth/coarse.hpp>
#include <figure/physics/cloth/island.hpp>
#include <figure/physics/cloth/job.hpp>
#include <figure/physics/cloth/report.hpp>
#include <figure/physics/collide/tri_grid.hpp>
#include <figure/physics/collide/sweep_grid.hpp>
#include <figure/physics/skin.hpp>

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
  physics::cloth::CpuBackend sim({s});
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
  // the drawn skin goes with it (version 2)
  {
    Surface&     m = job.surfaces[0];
    const size_t n = m.vertexCount();
    m.skinIndex8.assign(n * 8, 0);
    m.skinWeight8.assign(n * 8, 0.f);
    for(size_t v = 0; v < n; v++)
      m.skinWeight8[v * 8] = 1;
    m.dualQuat = 0.25f;
    m.dualQuatShare.assign(n, 0.5f);
  }
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
    EXPECT_EQ(a.skinIndex8, b.skinIndex8);
    EXPECT_EQ(a.skinWeight8, b.skinWeight8);
    EXPECT_EQ(a.dualQuat, b.dualQuat);
    EXPECT_EQ(a.dualQuatShare, b.dualQuatShare);
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

// A collider skinned by two bones from `in`'s influences, its vertices `rest` (mesh-local), into the
// job's space by `to`; the drawn skin (eight influences, the dual quaternion share) when `drawn`.
Surface twoBoneCollider(const std::vector<float>& rest, const mesh::SkinInput& in, const std::vector<Mat4>& bones, const Mat4& to, bool drawn)
{
  const physics::CageSkin skin = physics::cageSkin(in);
  std::vector<uint32_t>   tris;
  Surface                 m = Surface::collider(physics::transformPoints(to, rest), physics::transformPoints(to, rest), tris, 0, 0.2f, 0.4f);
  m.skinIndex  = skin.index;
  m.skinWeight = skin.weight;
  if(drawn)
  {
    m.skinIndex8    = skin.index8;
    m.skinWeight8   = skin.weight8;
    m.dualQuat      = skin.dualQuat;
    m.dualQuatShare = skin.dualQuatShare;
  }
  m.bones     = bones;
  m.restLocal = rest;
  m.toSpace   = to;
  return m;
}

// The body cloth meets is skinned as the drawn mesh's skin pass skins it (physics-plan step 13c): its
// eight influences and its dual quaternion share, as physics::skinLocal (the drawn skin on the CPU,
// which drapeDelta inverts), into the job's space; a vertex's matrix gives the same point
TEST(ClothSurface, theBodyClothMeetsIsSkinnedAsDrawn)
{
  const double pi = std::acos(-1.0);
  // a figure's root in metres and moved: the job's space is centimetres
  const Mat4               to    = compose(Vec3{3, -1, 2}, quatFromEuler(0, 0.3, 0, "XYZ"), Vec3{100, 100, 100});
  const std::vector<Mat4>  bones = {compose(Vec3{0.1, 0, 0}, quatFromEuler(0.2, 0, 0, "XYZ"), Vec3{1, 1, 1}),
                                    compose(Vec3{0, 0.05, 0}, quatFromEuler(0, 0, pi / 2, "XYZ"), Vec3{1, 1, 1})};
  // vertex 0 on one bone; 1 half and half; 2 spread over six influences of the two (the eight's slots)
  mesh::SkinInput in;
  in.start  = {0, 1, 3, 9};
  in.bone   = {0, 0, 1, 0, 1, 0, 1, 0, 1};
  in.weight = {1.f, 0.5f, 0.5f, 0.3f, 0.2f, 0.15f, 0.15f, 0.1f, 0.1f};
  const std::vector<float> rest{0.01f, 0.02f, 0.f, 0.f, 0.03f, 0.f, -0.02f, 0.01f, 0.04f};
  physics::SkinPose pose;
  pose.bones = bones;
  for(const float share : {0.f, 0.6f, 1.f})
  {
    in.dualQuat           = share;
    const Surface m       = twoBoneCollider(rest, in, bones, to, true);
    ASSERT_TRUE(m.drawnSkin()) << share;
    const auto drawn      = physics::transformPoints(to, physics::skinLocal(rest, physics::cageSkin(in), pose));
    std::vector<float> out(rest.size());
    physics::cloth::skin(m, bones, out.data());
    const physics::cloth::SurfaceSkin s(m, bones);
    for(size_t v = 0; v < 3; v++)
    {
      for(int c = 0; c < 3; c++)
        EXPECT_NEAR(out[v * 3 + c], drawn[v * 3 + c], 1e-4) << share << " " << v;
      Mat4 mv;
      ASSERT_TRUE(s.matrix(v, mv));
      const Vec3 p = mv.transformPoint({rest[v * 3], rest[v * 3 + 1], rest[v * 3 + 2]});
      EXPECT_NEAR(p.x, out[v * 3], 1e-4);
      EXPECT_NEAR(p.y, out[v * 3 + 1], 1e-4);
      EXPECT_NEAR(p.z, out[v * 3 + 2], 1e-4);
    }
  }
}

// Why: at a bent elbow the linear blend of the two bones pulls the skin in towards the joint, so the
// drawn arm is fuller there than the body the cloth was kept off. A ring vertex half on each bone, the
// second bent a quarter turn about the joint, keeps its distance with dual quaternions; the four,
// linearly, take it to 0.71 of it
TEST(ClothSurface, aDualQuaternionBodyKeepsABentJointsVolume)
{
  const double            pi    = std::acos(-1.0);
  const std::vector<Mat4> bones = {Mat4{}, compose(Vec3{0, 0, 0}, quatFromEuler(0, 0, pi / 2, "XYZ"), Vec3{1, 1, 1})};
  mesh::SkinInput         in;
  in.start  = {0, 2};
  in.bone   = {0, 1};
  in.weight = {0.5f, 0.5f};
  const std::vector<float> rest{0, 5, 0};  // 5 cm from the joint
  for(const float share : {0.f, 1.f})
  {
    in.dualQuat     = share;
    const Surface m = twoBoneCollider(rest, in, bones, Mat4{}, share > 0);
    float         p[3];
    physics::cloth::skin(m, bones, p);
    EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), share > 0 ? 5.0 : 5.0 * std::sqrt(0.5), 1e-4) << share;
  }
}

// A surface without the drawn skin (a version 1 job, a test's) is skinned as before: the four,
// linearly, in floats, to the bit
TEST(ClothSurface, withoutTheDrawnSkinTheFourBlendLinearlyAsBefore)
{
  const double            pi    = std::acos(-1.0);
  const Mat4              to    = compose(Vec3{1, 2, 3}, Quat{}, Vec3{100, 100, 100});
  const std::vector<Mat4> bones = {compose(Vec3{0.1, 0, 0}, Quat{}, Vec3{1, 1, 1}), compose(Vec3{0, 0, 0}, quatFromEuler(0, 0, pi / 3, "XYZ"), Vec3{1, 1, 1})};
  mesh::SkinInput         in;
  in.start  = {0, 2, 3};
  in.bone   = {0, 1, 1};
  in.weight = {0.25f, 0.75f, 1.f};
  in.dualQuat = 1;  // left out below: the surface has only the four
  const std::vector<float> rest{0.01f, 0.05f, 0.f, -0.02f, 0.f, 0.03f};
  const Surface            m = twoBoneCollider(rest, in, bones, to, false);
  ASSERT_FALSE(m.drawnSkin());
  float out[6];
  physics::cloth::skin(m, bones, out);
  for(size_t v = 0; v < 2; v++)
  {
    float blend[12] = {};
    for(size_t j = 0; j < 4; j++)
    {
      const float w = m.skinWeight[v * 4 + j];
      if(w == 0)
        continue;
      const physics::cloth::Mat34 a = physics::cloth::Mat34::of(to * bones[m.skinIndex[v * 4 + j]]);
      for(int c = 0; c < 12; c++)
        blend[c] += a.m[c] * w;
    }
    const float px = rest[v * 3], py = rest[v * 3 + 1], pz = rest[v * 3 + 2];
    EXPECT_EQ(out[v * 3], blend[0] * px + blend[3] * py + blend[6] * pz + blend[9]);
    EXPECT_EQ(out[v * 3 + 1], blend[1] * px + blend[4] * py + blend[7] * pz + blend[10]);
    EXPECT_EQ(out[v * 3 + 2], blend[2] * px + blend[5] * py + blend[8] * pz + blend[11]);
  }
}

TEST(ClothSurface, sameButPoseLooksPastThePoseAlone)
{
  // a kept GPU simulation keeps its model for a surface that is the same but for its pose: the target
  // cage, the bones' matrices and gravity may change, nothing a model is built from may
  const Surface a = strip(3, Mat4{});
  Surface       b = strip(3, shifted(2));
  b.gravity       = {0, 0, -980};
  EXPECT_NE(a.posed, b.posed);
  EXPECT_TRUE(physics::cloth::sameButPose(a, b));
  EXPECT_TRUE(physics::cloth::sameButPose(a, a));
  const auto differs = [&](const auto& change) {
    Surface c = a;
    change(c);
    return !physics::cloth::sameButPose(a, c);
  };
  EXPECT_TRUE(differs([](Surface& c) { c.rest[4] += 0.01f; })) << "a dial turned";
  EXPECT_TRUE(differs([](Surface& c) { c.strength[3] = 0.5f; }));
  EXPECT_TRUE(differs([](Surface& c) { c.offset[0] = 0.3f; }));
  EXPECT_TRUE(differs([](Surface& c) { c.layer[1] = 2; }));
  EXPECT_TRUE(differs([](Surface& c) { c.visible[2] = 0; })) << "a surface hidden";
  EXPECT_TRUE(differs([](Surface& c) { c.kind = Surface::Kind::eAddon; }));
  EXPECT_TRUE(differs([](Surface& c) { c.tris.pop_back(); }));
  EXPECT_TRUE(differs([](Surface& c) { c.skinWeight[0] = 0.5f; }));
  EXPECT_TRUE(differs([](Surface& c) { c.bones.push_back(Mat4{}); })) << "another rig";
  EXPECT_TRUE(differs([](Surface& c) { c.toSpace = shifted(1); })) << "another space";
  // a collider's rest cage may count as its pose (a body's contact dents), a cloth's never
  Surface body = a, dented = a;
  body.kind = dented.kind = Surface::Kind::eCollider;
  dented.rest[4] += 0.2f;
  dented.restLocal[4] += 0.2f;
  EXPECT_FALSE(physics::cloth::sameButPose(body, dented));
  EXPECT_EQ(physics::cloth::poseFreeDifference(body, dented, true), nullptr);
  Surface cloth = a;
  cloth.rest[4] += 0.2f;
  EXPECT_STREQ(physics::cloth::poseFreeDifference(a, cloth, true), "rest");
  dented.rest.pop_back();
  EXPECT_NE(physics::cloth::poseFreeDifference(body, dented, true), nullptr) << "another size is another mesh";
}

// A soup of `faces` faces, 0.5 to 3 cm across, in a box `spread` cm wide, and the same moved by up to `move`
// cm a corner: a frame of the GPU solver's sweep.
struct Soup
{
  std::vector<int32_t> corners;
  std::vector<float>   start, end;
};

Soup soup(int faces, float spread, float move, unsigned seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> at(-spread / 2, spread / 2), size(0.5f, 3.f), jitter(-1.f, 1.f);
  Soup                                  s;
  for(int f = 0; f < faces; f++)
  {
    const float cx = at(rng), cy = at(rng), cz = at(rng), r = size(rng);
    for(int k = 0; k < 3; k++)
    {
      s.corners.push_back(int32_t(s.start.size() / 3));
      const float p[3] = {cx + r * jitter(rng), cy + r * jitter(rng), cz + r * jitter(rng)};
      for(int c = 0; c < 3; c++)
      {
        s.start.push_back(p[c]);
        s.end.push_back(p[c] + move * jitter(rng));
      }
    }
  }
  return s;
}

TEST(ClothReport, aGarmentPushedThroughAnotherIsCountedOnTheWrongSide)
{
  // two sheets 1 cm apart, the upper one's vertices on the outside (up) of the lower; at the end the
  // upper sheet's middle row has gone 0.5 cm under the lower
  const auto sheet = [](float y) {
    Surface s;
    for(int z = 0; z <= 4; z++)
      for(int x = 0; x <= 4; x++)
        s.rest.insert(s.rest.end(), {float(x), y, float(z)});
    for(int z = 0; z < 4; z++)
      for(int x = 0; x < 4; x++)
      {
        const uint32_t a = uint32_t(z * 5 + x), b = a + 1, c = a + 5, d = a + 6;
        // wound so the normal points up
        s.tris.insert(s.tris.end(), {a, c, b, b, c, d});
      }
    s.posed = s.rest;
    return s;
  };
  const Surface lower = sheet(0), upper = sheet(1);
  std::vector<float>   upperOut = upper.rest;
  for(int x = 0; x <= 4; x++)
    upperOut[size_t(2 * 5 + x) * 3 + 1] = -0.5f;
  EXPECT_EQ(physics::cloth::wrongSide(upper, upper.rest, lower, lower.rest), 0u);
  EXPECT_EQ(physics::cloth::wrongSide(upper, upperOut, lower, lower.rest), 5u);
  // and from the lower's side: none of its vertices moved, but now the upper sheet's middle lies under
  // them, which the closest faces show as a side changed for the vertices it passed
  EXPECT_GT(physics::cloth::wrongSide(lower, lower.rest, upper, upperOut), 0u);
}

TEST(SweepGrid, theGridHoldsTheReferencesCellsFrameAfterFrame)
{
  // the table sized by the cells met (an eighth full at most, grown in the middle of a frame when it
  // fills) holds the same cells with the same faces in the same order as the reference's, sized by the
  // entries: the GPU's nearest-face search finds the same faces in it, ties included
  physics::SweepGrid grid, ref;
  const struct
  {
    int   faces;
    float spread, move;
  } frames[] = {{400, 20, 0.5f}, {400, 20, 0.5f}, {3000, 60, 1}, {3000, 60, 4}, {200, 10, 0}, {6000, 120, 2}, {8000, 16, 0.5f}};
  unsigned seed = 1;
  for(const auto& f : frames)
  {
    const Soup s = soup(f.faces, f.spread, f.move, seed++);
    physics::buildSweepGrid(s.corners, s.start.data(), s.end.data(), 2.f, grid);
    physics::buildSweepGridReference(s.corners, s.start.data(), s.end.data(), 2.f, ref);
    EXPECT_TRUE(physics::sameSweepGrids(ref, grid)) << f.faces << " faces";
    EXPECT_EQ(grid.entries, ref.entries);
    // an eighth full at most, or the reference's size (half full at most) when the cells are nearly
    // as many as the entries
    EXPECT_TRUE(grid.used * 8 <= grid.cells.size() || grid.cells.size() == ref.cells.size()) << f.faces << " faces";
    EXPECT_LE(grid.cells.size(), ref.cells.size());
  }
  // a body's faces share cells: 8,000 in a 16 cm box want a table smaller than their entries do, from
  // the frame after (a frame starts at the size the last one needed)
  physics::SweepGrid dense;
  for(unsigned frame = 0; frame < 2; frame++)
  {
    const Soup s = soup(8000, 16, 0.5f, 100 + frame);
    physics::buildSweepGrid(s.corners, s.start.data(), s.end.data(), 2.f, dense);
    physics::buildSweepGridReference(s.corners, s.start.data(), s.end.data(), 2.f, ref);
    EXPECT_TRUE(physics::sameSweepGrids(ref, dense));
  }
  EXPECT_LE(dense.used * 8, dense.cells.size());
  EXPECT_LT(dense.cells.size() * 4, ref.cells.size());
}

TEST(SweepGrid, everyFaceIsInEveryCellItsSweepMeetsInAscendingOrder)
{
  const Soup         s = soup(300, 16, 1.5f, 7);
  physics::SweepGrid grid;
  physics::buildSweepGrid(s.corners, s.start.data(), s.end.data(), 2.f, grid);
  // every cell's run ascends
  size_t listed = 0;
  for(const physics::SweepCell& c : grid.cells)
    if(c.start >= 0)
    {
      listed += size_t(c.count);
      EXPECT_TRUE(std::is_sorted(grid.items.begin() + c.start, grid.items.begin() + c.start + c.count));
    }
  EXPECT_EQ(listed, grid.entries);
  // and each face is found in each cell its swept box meets, as the kernels look a cell up
  const auto lookup = [&](int x, int y, int z) -> const physics::SweepCell* {
    const uint32_t mask = grid.mask();
    for(uint32_t probe = 0, at = physics::sweepCellHash(x, y, z) & mask; probe <= mask; probe++)
    {
      const physics::SweepCell& e = grid.cells[(at + probe) & mask];
      if(e.start < 0)
        return nullptr;
      if(e.x == x && e.y == y && e.z == z)
        return &e;
    }
    return nullptr;
  };
  for(size_t f = 0; f < s.corners.size() / 3; f++)
  {
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(int k = 0; k < 3; k++)
      for(int c = 0; c < 3; c++)
        for(const std::vector<float>* p : {&s.start, &s.end})
        {
          lo[c] = std::min(lo[c], (*p)[size_t(s.corners[f * 3 + size_t(k)]) * 3 + c]);
          hi[c] = std::max(hi[c], (*p)[size_t(s.corners[f * 3 + size_t(k)]) * 3 + c]);
        }
    for(int x = int(std::floor(lo[0] / 2)); x <= int(std::floor(hi[0] / 2)); x++)
      for(int y = int(std::floor(lo[1] / 2)); y <= int(std::floor(hi[1] / 2)); y++)
        for(int z = int(std::floor(lo[2] / 2)); z <= int(std::floor(hi[2] / 2)); z++)
        {
          const physics::SweepCell* c = lookup(x, y, z);
          ASSERT_NE(c, nullptr);
          EXPECT_TRUE(std::binary_search(grid.items.begin() + c->start, grid.items.begin() + c->start + c->count, int32_t(f)));
        }
  }
}

namespace {

// A sphere of radius `r` about the origin as a collider: `rings` bands from pole to pole of 2 x `rings`
// quads each, cut in two, with a skin of four influences whose values tell each vertex apart.
Surface sphere(float r, int rings)
{
  std::vector<float>    pos;
  std::vector<uint32_t> tris;
  const int             segments = rings * 2;
  const double          pi       = 3.14159265358979323846;
  pos.insert(pos.end(), {0.f, r, 0.f});
  for(int i = 1; i < rings; i++)
    for(int j = 0; j < segments; j++)
    {
      const double theta = pi * i / rings, phi = 2 * pi * j / segments;
      pos.insert(pos.end(), {float(r * std::sin(theta) * std::cos(phi)), float(r * std::cos(theta)), float(r * std::sin(theta) * std::sin(phi))});
    }
  pos.insert(pos.end(), {0.f, -r, 0.f});
  const uint32_t south = uint32_t(pos.size() / 3 - 1);
  const auto     at    = [&](int i, int j) { return uint32_t(1 + (i - 1) * segments + (j % segments)); };
  for(int j = 0; j < segments; j++)
  {
    tris.insert(tris.end(), {0, at(1, j + 1), at(1, j)});
    tris.insert(tris.end(), {south, at(rings - 1, j), at(rings - 1, j + 1)});
  }
  for(int i = 1; i + 1 < rings; i++)
    for(int j = 0; j < segments; j++)
      tris.insert(tris.end(), {at(i, j), at(i, j + 1), at(i + 1, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)});
  Surface      s = Surface::collider(pos, pos, tris, 0, 0.2f, 0.4f);
  const size_t n = s.vertexCount();
  for(size_t v = 0; v < n; v++)
  {
    s.friction[v] = 0.3f + 1e-5f * float(v);
    s.skinIndex.insert(s.skinIndex.end(), {0, 0, 0, 0});
    s.skinWeight.insert(s.skinWeight.end(), {1.f, 0.f, 0.f, float(v)});
  }
  s.restLocal = pos;
  s.bones     = {Mat4{}};
  return s;
}

}  // namespace

TEST(ClothCoarse, aColliderFinerThanTheClothBecomesACopyAsFineAsItWithinItsError)
{
  // A necklace's links under a dress: a sphere of 3 cm in 14k faces with sides of about 0.16 cm, met by
  // cloth whose sides are a centimetre long
  const Surface fine = sphere(3.f, 60);
  ASSERT_GT(fine.tris.size() / 3, 14000u);
  const auto c = physics::cloth::coarsen(fine, 1.f, 0.25f);
  ASSERT_TRUE(c);
  const Surface& s = c->surface;
  // about as many faces as sides of a centimetre take to cover it: 4 pi 3^2 / (sqrt(3)/4) = 261
  EXPECT_LT(s.tris.size() / 3, 600u);
  EXPECT_GT(s.tris.size() / 3, 100u);
  // its vertices are the ones it says it kept, with all they carry
  ASSERT_EQ(s.vertexCount(), c->keep.size());
  EXPECT_LT(c->keep.size(), fine.vertexCount() / 10);
  EXPECT_TRUE(std::is_sorted(c->keep.begin(), c->keep.end()));
  EXPECT_TRUE(s.skinned());
  EXPECT_EQ(s.kind, Surface::Kind::eCollider);
  for(size_t i = 0; i < c->keep.size(); i++)
  {
    const size_t v = c->keep[i];
    for(size_t k = 0; k < 3; k++)
    {
      EXPECT_EQ(s.rest[i * 3 + k], fine.rest[v * 3 + k]);
      EXPECT_EQ(s.posed[i * 3 + k], fine.posed[v * 3 + k]);
      EXPECT_EQ(s.restLocal[i * 3 + k], fine.restLocal[v * 3 + k]);
    }
    EXPECT_EQ(s.friction[i], fine.friction[v]);
    EXPECT_EQ(s.skinWeight[i * 4 + 3], fine.skinWeight[v * 4 + 3]);
  }
  for(uint32_t t : s.tris)
    EXPECT_LT(t, s.vertexCount());
  // within its error of the sphere, measured and within twice what was asked, which goes on its offset:
  // the cloth keeps out of the original
  EXPECT_GT(c->error, 0.f);
  EXPECT_LE(c->error, 0.5f);
  for(float o : s.offset)
    EXPECT_FLOAT_EQ(o, 0.2f + c->error);
  double sunk = 0;
  for(size_t t = 0; t + 2 < s.tris.size(); t += 3)
  {
    double centre[3] = {};
    for(size_t k = 0; k < 3; k++)
      for(size_t a = 0; a < 3; a++)
        centre[a] += s.rest[size_t(s.tris[t + k]) * 3 + a] / 3;
    sunk = std::max(sunk, 3.0 - std::sqrt(centre[0] * centre[0] + centre[1] * centre[1] + centre[2] * centre[2]));
  }
  EXPECT_LE(sunk, c->error + 0.02) << "a face lies further under the sphere than the error it gave";
}

TEST(ClothCoarse, aChainOfSmallLinksIsCopiedThoughNoLinkCanBeSimplifiedAlone)
{
  // A necklace's chain: 100 closed links of 2.8 mm, 528 faces each, one every 2.5 mm. A simplifier that
  // keeps each piece's shape keeps most of their faces; the copy joins them
  std::vector<float>    pos;
  std::vector<uint32_t> tris;
  const Surface         link = sphere(0.14f, 12);
  for(int k = 0; k < 100; k++)
  {
    const uint32_t base = uint32_t(pos.size() / 3);
    for(size_t v = 0; v < link.vertexCount(); v++)
      pos.insert(pos.end(), {link.rest[v * 3] + 0.25f * float(k), link.rest[v * 3 + 1], link.rest[v * 3 + 2]});
    for(uint32_t t : link.tris)
      tris.push_back(base + t);
  }
  const Surface chain = Surface::collider(pos, pos, tris, 0, 0.2f, 0.4f);
  ASSERT_EQ(chain.tris.size() / 3, 52800u);
  const auto c = physics::cloth::coarsen(chain, 1.f, 0.25f);
  ASSERT_TRUE(c);
  EXPECT_LT(c->surface.tris.size() / 3, 5000u);
  EXPECT_GT(c->surface.tris.size() / 3, 0u);
  EXPECT_LE(c->error, 0.5f);
  for(float o : c->surface.offset)
    EXPECT_FLOAT_EQ(o, 0.2f + c->error);
  // the copy runs the chain's whole length
  float lo = 1e9f, hi = -1e9f;
  for(size_t v = 0; v < c->surface.vertexCount(); v++)
  {
    lo = std::min(lo, c->surface.rest[v * 3]);
    hi = std::max(hi, c->surface.rest[v * 3]);
  }
  EXPECT_LT(lo, 0.f);
  EXPECT_GT(hi, 24.75f);
}

TEST(ClothCoarse, aChainOfThinWireRingsIsCopiedWithEveryRing)
{
  // The Charming Witch necklace's chain, as the app had it in a walk: rings 1.4 mm across of wire 0.24 mm
  // thick, one every 2.5 mm. Within an eighth of the error asked the simplifier still folds a ring's wire
  // flat, and a ring that goes leaves its vertices too far off: the copy was made finer still
  constexpr int   kRings = 100, kAround = 24, kRound = 8;
  constexpr float kRing = 0.07f, kWire = 0.012f, kPitch = 0.25f;
  std::vector<float>    pos;
  std::vector<uint32_t> tris;
  for(int k = 0; k < kRings; k++)
  {
    const uint32_t base = uint32_t(pos.size() / 3);
    // each ring turned a quarter from the last, as a chain's links are
    const bool upright = k % 2 == 1;
    for(int a = 0; a < kAround; a++)
      for(int r = 0; r < kRound; r++)
      {
        const float u = 2.f * 3.14159265f * float(a) / kAround, w = 2.f * 3.14159265f * float(r) / kRound;
        const float x = (kRing + kWire * std::cos(w)) * std::cos(u), y = (kRing + kWire * std::cos(w)) * std::sin(u), z = kWire * std::sin(w);
        pos.insert(pos.end(), {x + kPitch * float(k), upright ? z : y, upright ? y : z});
      }
    for(int a = 0; a < kAround; a++)
      for(int r = 0; r < kRound; r++)
      {
        const uint32_t v00 = base + uint32_t(a * kRound + r), v01 = base + uint32_t(a * kRound + (r + 1) % kRound);
        const uint32_t v10 = base + uint32_t(((a + 1) % kAround) * kRound + r), v11 = base + uint32_t(((a + 1) % kAround) * kRound + (r + 1) % kRound);
        tris.insert(tris.end(), {v00, v10, v11, v00, v11, v01});
      }
  }
  const Surface chain = Surface::collider(pos, pos, tris, 0, 0.2f, 0.4f);
  ASSERT_EQ(chain.tris.size() / 3, size_t(kRings * kAround * kRound * 2));
  const auto c = physics::cloth::coarsen(chain, 1.f, 0.25f);
  ASSERT_TRUE(c) << "a chain of thin rings stays as fine as it is";
  EXPECT_LT(c->surface.tris.size() / 3, chain.tris.size() / 3 / 2);
  EXPECT_LE(c->error, 0.5f);
  // every ring has faces in the copy
  std::vector<int> faces(kRings, 0);
  for(size_t t = 0; t + 2 < c->surface.tris.size(); t += 3)
    faces[size_t(c->keep[c->surface.tris[t]] / (kAround * kRound))]++;
  for(int k = 0; k < kRings; k++)
    EXPECT_GT(faces[size_t(k)], 0) << "ring " << k;
}

TEST(ClothCoarse, aColliderAsFineAsTheClothOrClothItselfStaysAsItIs)
{
  // 8 bands: sides of about 1.2 cm, as the cloth's
  EXPECT_FALSE(physics::cloth::coarsen(sphere(3.f, 8), 1.f, 0.25f));
  // cloth isn't copied, however fine
  Surface cloth = sphere(3.f, 60);
  cloth.kind    = Surface::Kind::eDynamic;
  EXPECT_FALSE(physics::cloth::coarsen(cloth, 1.f, 0.25f));
  // nor for cloth with no faces to measure
  EXPECT_FALSE(physics::cloth::coarsen(sphere(3.f, 60), 0.f, 0.25f));
}

TEST(ClothCoarse, onlyTheFacesThatCollideAreCopied)
{
  // the upper half doesn't collide (DAZ's Collide off on a surface): none of it is in the copy
  Surface fine = sphere(3.f, 60);
  for(size_t v = 0; v < fine.vertexCount(); v++)
    fine.collide[v] = fine.rest[v * 3 + 1] > 0.5f ? 0 : 1;
  const auto c = physics::cloth::coarsen(fine, 1.f, 0.25f);
  ASSERT_TRUE(c);
  for(size_t i = 0; i < c->keep.size(); i++)
  {
    EXPECT_LE(c->surface.rest[i * 3 + 1], 0.5f) << i;
    EXPECT_EQ(c->surface.collide[i], 1) << i;
  }
}

TEST(ClothCoarse, theMeanEdgeIsTheFacesSides)
{
  // two right triangles of legs 1: sides 1, 1 and sqrt(2) each
  Surface s = Surface::collider({0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1}, {0, 1, 2, 0, 2, 3}, 0, 0.2f, 0.4f);
  EXPECT_NEAR(physics::cloth::meanEdge(s), (2 + std::sqrt(2.0)) / 3, 1e-6);
  s.tris.clear();
  EXPECT_EQ(physics::cloth::meanEdge(s), 0.f);
}

TEST(ClothCoarse, theSameCopyOfAColliderPosedAnewKeepsItsFacesAndTakesThePose)
{
  // a rebuild for the same shape reuses the copy, of the collider as it is posed now
  const Surface fine = sphere(3.f, 60);
  const auto    c    = physics::cloth::coarsen(fine, 1.f, 0.25f);
  ASSERT_TRUE(c);
  Surface moved = fine;
  for(size_t v = 0; v < moved.vertexCount(); v++)
    moved.posed[v * 3 + 1] += 1.f;
  moved.bones = {compose(Vec3{0, 1, 0}, Quat{}, Vec3{1, 1, 1})};
  const auto same = physics::cloth::sameCopy(moved, *c);
  ASSERT_TRUE(same);
  EXPECT_EQ(same->keep, c->keep);
  EXPECT_EQ(same->surface.tris, c->surface.tris);
  EXPECT_EQ(same->surface.rest, c->surface.rest);
  EXPECT_EQ(same->surface.offset, c->surface.offset);
  EXPECT_EQ(same->error, c->error);
  ASSERT_EQ(same->surface.bones.size(), 1u);
  for(size_t i = 0; i < same->keep.size(); i++)
    EXPECT_EQ(same->surface.posed[i * 3 + 1], fine.posed[size_t(same->keep[i]) * 3 + 1] + 1.f);
  // nor for a collider without those vertices
  EXPECT_FALSE(physics::cloth::sameCopy(sphere(3.f, 8), *c));
}

namespace {

// A cloth from points and the quads of a grid over them, `rows` by `cols` vertices, free but for the
// first row, which is held to its skin.
Surface clothOn(std::vector<float> pos, int rows, int cols)
{
  Surface s;
  s.kind = Surface::Kind::eDynamic;
  const auto at = [&](int r, int c) { return uint32_t(r * cols + c); };
  for(int r = 0; r + 1 < rows; r++)
    for(int c = 0; c + 1 < cols; c++)
    {
      s.tris.insert(s.tris.end(), {at(r, c), at(r + 1, c), at(r, c + 1), at(r, c + 1), at(r + 1, c), at(r + 1, c + 1)});
      s.polyVerts.insert(s.polyVerts.end(), {int32_t(at(r, c)), int32_t(at(r + 1, c)), int32_t(at(r + 1, c + 1)), int32_t(at(r, c + 1))});
      s.polyStart.push_back(int32_t(s.polyVerts.size()));
    }
  const size_t n = pos.size() / 3;
  s.rest         = pos;
  s.posed        = std::move(pos);
  s.strength.assign(n, 1.f);
  for(int c = 0; c < cols; c++)
    s.strength[at(0, c)] = 0.f;
  s.stretch.assign(n, 0.8f);
  s.shear.assign(n, 0.2f);
  s.bend.assign(n, 0.5f);
  s.buckling.assign(n, 0.05f);
  s.bucklingRatio.assign(n, 0.7f);
  s.density.assign(n, 180.f);
  s.contraction.assign(n, 1.f);
  s.damping.assign(n, 0.1f);
  s.friction.assign(n, 0.4f);
  s.offset.assign(n, 0.2f);
  s.layer.assign(n, 0);
  s.selfCollide.assign(n, 0);
  s.collide.assign(n, 1);
  s.visible.assign(n, 1);
  for(size_t v = 0; v < n; v++)
    s.friction[v] = 0.3f + 1e-5f * float(v);
  return s;
}

// A sheet `size` cm square in quads of size / n, in the xz plane, with a gentle swell so its vertices
// lie off a copy's flat faces.
Surface sheet(int n, float size)
{
  std::vector<float> pos;
  for(int r = 0; r <= n; r++)
    for(int c = 0; c <= n; c++)
    {
      const float x = size * (float(c) / float(n) - 0.5f), z = size * (float(r) / float(n) - 0.5f);
      pos.insert(pos.end(), {x, 0.4f * std::sin(0.5f * x) * std::cos(0.4f * z), z});
    }
  return clothOn(std::move(pos), n + 1, n + 1);
}

float sideOf(const std::vector<float>& p, uint32_t a, uint32_t b)
{
  return std::hypot(p[a * 3] - p[b * 3], p[a * 3 + 1] - p[b * 3 + 1], p[a * 3 + 2] - p[b * 3 + 2]);
}

}  // namespace

TEST(ClothLight, aFineClothBecomesALightCopyOfQuadsAlongItsThreadsThatKeepsItsSettings)
{
  // 15 cm square in quads of 0.25 cm (3,600 quads), copied with faces 2 cm wide: rings of its quads
  // collapsed three times, 0.25 to 2 cm, leaving 7 or 8 quads each way
  const Surface fine = sheet(60, 15.f);
  const auto    c    = physics::cloth::coarseCloth(fine, 2.f);
  ASSERT_TRUE(c);
  const Surface& s     = c->surface;
  const size_t   polys = s.polyStart.size() - 1;
  EXPECT_GE(polys, 40u);
  EXPECT_LE(polys, 100u);
  EXPECT_EQ(s.kind, Surface::Kind::eDynamic);
  // quads, two faces each to collide and ride on, their sides along the cloth's threads (x and z)
  EXPECT_EQ(s.polyVerts.size(), polys * 4);
  EXPECT_EQ(s.tris.size(), polys * 6);
  std::vector<float> sides;
  for(size_t q = 0; q < polys; q++)
    for(int k = 0; k < 4; k++)
    {
      const uint32_t a = uint32_t(s.polyVerts[q * 4 + size_t(k)]), b = uint32_t(s.polyVerts[q * 4 + size_t((k + 1) % 4)]);
      const float    dx = std::fabs(s.rest[a * 3] - s.rest[b * 3]), dz = std::fabs(s.rest[a * 3 + 2] - s.rest[b * 3 + 2]);
      EXPECT_LT(std::min(dx, dz), 1e-4f) << "a side across the threads";
      sides.push_back(std::max(dx, dz));
    }
  std::sort(sides.begin(), sides.end());
  EXPECT_GT(sides[sides.size() / 2], 1.5f);
  EXPECT_LT(sides[sides.size() / 2], 3.f);
  // all of it: its hems are kept
  float lo[2] = {1e9f, 1e9f}, hi[2] = {-1e9f, -1e9f};
  for(size_t v = 0; v < s.vertexCount(); v++)
    for(int a = 0; a < 2; a++)
    {
      lo[a] = std::min(lo[a], s.rest[v * 3 + size_t(a) * 2]);
      hi[a] = std::max(hi[a], s.rest[v * 3 + size_t(a) * 2]);
    }
  for(int a = 0; a < 2; a++)
  {
    EXPECT_FLOAT_EQ(lo[a], -7.5f);
    EXPECT_FLOAT_EQ(hi[a], 7.5f);
  }
  // its vertices are the cloth's own, with what each carries
  ASSERT_EQ(s.vertexCount(), c->keep.size());
  EXPECT_TRUE(std::is_sorted(c->keep.begin(), c->keep.end()));
  size_t held = 0;
  for(size_t i = 0; i < c->keep.size(); i++)
  {
    EXPECT_EQ(s.strength[i], fine.strength[c->keep[i]]);
    EXPECT_EQ(s.friction[i], fine.friction[c->keep[i]]);
    held += s.strength[i] == 0.f ? 1 : 0;
  }
  // the held row is in the copy, and every vertex of it is held as it rides
  EXPECT_GE(held, 2u);
  ASSERT_EQ(c->riding.held.size(), fine.vertexCount());
  for(size_t v = 0; v < fine.vertexCount(); v++)
    EXPECT_EQ(c->riding.held[v], fine.strength[v] == 0.f ? 1 : 0) << v;
}

TEST(ClothLight, theClothRidesOnItsCopyExactlyAtRestAndMovesRigidlyWithIt)
{
  const Surface fine = sheet(60, 15.f);
  const auto    c    = physics::cloth::coarseCloth(fine, 2.f);
  ASSERT_TRUE(c);
  // at rest: every vertex where it was
  const auto still = physics::cloth::ride(*c, c->surface.rest);
  ASSERT_EQ(still.size(), fine.rest.size());
  float worst = 0, off = 0;
  for(size_t i = 0; i < still.size(); i++)
    worst = std::max(worst, std::fabs(still[i] - fine.rest[i]));
  EXPECT_LT(worst, 1e-4f);
  for(float o : c->riding.offset)
    off = std::max(off, std::fabs(o));
  EXPECT_GT(off, 1e-3f) << "the swell lies off the copy's faces, so the offsets are tested";
  // the copy turned 30 degrees about y and moved: the cloth turns and moves with it
  const float ca = std::cos(0.5236f), sa = std::sin(0.5236f);
  const auto  move = [&](const std::vector<float>& p) {
    std::vector<float> o(p.size());
    for(size_t v = 0; v < p.size() / 3; v++)
    {
      o[v * 3]     = ca * p[v * 3] + sa * p[v * 3 + 2] + 3.f;
      o[v * 3 + 1] = p[v * 3 + 1] - 2.f;
      o[v * 3 + 2] = -sa * p[v * 3] + ca * p[v * 3 + 2] + 1.f;
    }
    return o;
  };
  const auto moved = physics::cloth::ride(*c, move(c->surface.rest));
  const auto want  = move(fine.rest);
  worst            = 0;
  for(size_t i = 0; i < moved.size(); i++)
    worst = std::max(worst, std::fabs(moved[i] - want[i]));
  EXPECT_LT(worst, 1e-3f);
}

TEST(ClothLight, eachLayerOfAFoldRidesOnItsOwn)
{
  // A strip 4 cm wide folded back on itself, as a pleat: 10 cm along x at y = 0, round a bend of
  // 0.2 cm, and back at y = 0.4, in quads of 0.2 cm. The copy's upper layer lifted 5 cm lifts the
  // cloth's upper layer, and its lower one stays
  const float        r = 0.2f, flat = 10.f, bend = 3.14159265f * r, length = 2 * flat + bend;
  const int          along = 108, across = 21;
  std::vector<float> pos;
  for(int i = 0; i < along; i++)
    for(int j = 0; j < across; j++)
    {
      const float u = length * float(i) / float(along - 1), z = 4.f * float(j) / float(across - 1);
      float       x = 0, y = 0;
      if(u <= flat)
        x = u;
      else if(u <= flat + bend)
      {
        const float a = (u - flat) / r;
        x = flat + r * std::sin(a);
        y = r - r * std::cos(a);
      }
      else
      {
        x = flat - (u - flat - bend);
        y = 2 * r;
      }
      pos.insert(pos.end(), {x, y, z});
    }
  Surface fine = clothOn(std::move(pos), along, across);
  // nothing held here: the copy's lift is all there is
  std::fill(fine.strength.begin(), fine.strength.end(), 1.f);
  const auto c = physics::cloth::coarseCloth(fine, 1.f);
  ASSERT_TRUE(c);
  std::vector<float> lifted = c->surface.rest;
  for(size_t v = 0; v < lifted.size() / 3; v++)
    if(lifted[v * 3 + 1] > r)
      lifted[v * 3 + 1] += 5.f;
  const auto rode = physics::cloth::ride(*c, lifted);
  for(size_t v = 0; v < fine.vertexCount(); v++)
  {
    if(fine.rest[v * 3] > flat - 2)
      continue;  // by the bend, the faces join the layers
    const float rise = rode[v * 3 + 1] - fine.rest[v * 3 + 1];
    if(fine.rest[v * 3 + 1] > r)
      EXPECT_NEAR(rise, 5.f, 0.05f) << "upper layer, vertex " << v;
    else
      EXPECT_NEAR(rise, 0.f, 0.05f) << "lower layer, vertex " << v;
  }
}

TEST(ClothLight, aClothAsCoarseAsAskedOrAColliderIsNotCopied)
{
  EXPECT_FALSE(physics::cloth::coarseCloth(sheet(6, 15.f), 2.f));
  Surface collider = sheet(60, 15.f);
  collider.kind    = Surface::Kind::eCollider;
  EXPECT_FALSE(physics::cloth::coarseCloth(collider, 2.f));
  EXPECT_FALSE(physics::cloth::coarseCloth(sheet(60, 15.f), 0.f));
}

TEST(ClothLight, theSameCopyOfAClothPosedAnewRidesAsItDid)
{
  const Surface fine = sheet(60, 15.f);
  const auto    c    = physics::cloth::coarseCloth(fine, 2.f);
  ASSERT_TRUE(c);
  Surface moved = fine;
  for(size_t v = 0; v < moved.vertexCount(); v++)
    moved.posed[v * 3 + 1] += 1.f;
  const auto same = physics::cloth::sameCloth(moved, *c);
  ASSERT_TRUE(same);
  EXPECT_EQ(same->keep, c->keep);
  EXPECT_EQ(same->surface.tris, c->surface.tris);
  EXPECT_EQ(same->surface.polyVerts, c->surface.polyVerts);
  EXPECT_EQ(same->riding.face, c->riding.face);
  for(size_t i = 0; i < same->keep.size(); i++)
    EXPECT_EQ(same->surface.posed[i * 3 + 1], fine.posed[size_t(same->keep[i]) * 3 + 1] + 1.f);
  EXPECT_FALSE(physics::cloth::sameCloth(sheet(30, 15.f), *c));
}

TEST(ClothLight, whatRidesIntoTheBodyIsMovedOutToItsOffset)
{
  // a body's floor at y = 0 (offset 0.2), and points riding through it: behind it within reach, in
  // front closer than the offset, in front further, too far behind, and one held to its skin
  Surface body = Surface::collider({-5, 0, -5, 5, 0, -5, 5, 0, 5, -5, 0, 5}, {-5, 0, -5, 5, 0, -5, 5, 0, 5, -5, 0, 5}, {0, 2, 1, 0, 3, 2}, 0, 0.2f, 0.4f);
  std::vector<float>   at   = {0, -0.5f, 0, 1, 0.1f, 1, 2, 1.f, 2, -1, -3.f, -1, -2, -0.5f, 2};
  std::vector<uint8_t> held = {0, 0, 0, 0, 1};
  const size_t         moved = physics::cloth::keepOut(at, held, body, body.posed);
  EXPECT_EQ(moved, 2u);
  EXPECT_NEAR(at[1], 0.2f, 1e-5f);
  EXPECT_NEAR(at[4], 0.2f, 1e-5f);
  EXPECT_EQ(at[7], 1.f);
  EXPECT_EQ(at[10], -3.f);
  EXPECT_EQ(at[13], -0.5f);
  // across the floor, not along it
  EXPECT_EQ(at[0], 0.f);
  EXPECT_EQ(at[3], 1.f);
}

TEST(ClothLight, keepingOutWithTheCopyLooksOnlyNearTheBodyAndMovesTheSame)
{
  // A 15 cm sheet whose one corner dips 1.5 cm into a ball (radius 4) and whose far side lies well off it:
  // looked for only where the copy's faces may put a vertex near the ball, every vertex ends where the
  // search of every vertex puts it, to the bit (the Kai Dress's 42,687 searched took 9 ms of a live step)
  const Surface fine = sheet(60, 15.f);
  const auto    c    = physics::cloth::coarseCloth(fine, 2.f);
  ASSERT_TRUE(c);
  Surface            ball   = sphere(4.f, 24);
  std::vector<float> ballAt = ball.posed;
  for(size_t v = 0; v < ballAt.size() / 3; v++)
  {
    ballAt[v * 3] += 5.f;
    ballAt[v * 3 + 1] -= 2.5f;
    ballAt[v * 3 + 2] += 5.f;
  }
  // the copy where the cloth is, a little turned, so a vertex is off its face's plane
  std::vector<float> copyAt = c->surface.rest;
  for(size_t v = 0; v < copyAt.size() / 3; v++)
    copyAt[v * 3 + 1] += 0.05f * copyAt[v * 3];
  const std::vector<float> rode = physics::cloth::ride(*c, copyAt);
  std::vector<float>       every = rode, near = rode;
  const size_t             all   = physics::cloth::keepOut(every, c->riding.held, ball, ballAt);
  const size_t             some  = physics::cloth::keepOut(near, c->riding.held, ball, ballAt, 2.f, &*c, &copyAt);
  EXPECT_GT(all, 50u) << "the corner in the ball is moved out";
  EXPECT_LT(all, rode.size() / 3 / 4) << "most of the sheet is off the ball";
  EXPECT_EQ(some, all);
  EXPECT_EQ(near, every);
}
