// A worn item is bound to the closest point on the body, and a body morph then moves it by what
// moved there; the body's polygons a graft replaces leave the drawn triangles. A port of
// src/lib/loaders/dson/bind.ts and follow.ts, with the MorphSet's follower.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include <memory>
#include <set>

#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/gltf/gltf_check.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/subdiv/catmark.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/wardrobe/fit/transfer.hpp>
#include <figure/wardrobe/follow.hpp>

#include "gltf_scene.hpp"

namespace {

// A flat grid of n x n quads in the XY plane at height z, one material.
fv::mesh::MeshInput grid(int n, float z)
{
  fv::mesh::MeshInput m;
  for(int y = 0; y <= n; y++)
    for(int x = 0; x <= n; x++)
      m.positions.insert(m.positions.end(), {float(x), float(y), z});
  m.polyStart.push_back(0);
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      const int a = y * (n + 1) + x;
      m.polyVerts.insert(m.polyVerts.end(), {a, a + 1, a + n + 2, a + n + 1});
      m.polyStart.push_back(int(m.polyVerts.size()));
      m.polyMaterial.push_back(0);
    }
  m.uvs.assign(size_t(n + 1) * (n + 1) * 2, 0.f);
  return m;
}

TEST(Wardrobe, BindsEachItemVertexToTheClosestBodyPoint)
{
  const auto body = fv::mesh::build(grid(4, 0));
  const auto item = fv::mesh::build(grid(2, 0.5f));
  const auto b    = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  ASSERT_EQ(b.verts.size(), item.cagePositions.size());
  // item vertex (1, 1) sits above body vertex (1, 1), index 1 * 5 + 1
  const size_t v = 1 * 3 + 1;
  float        wx = 0;
  for(int k = 0; k < 3; k++)
    if(b.verts[v * 3 + k] == 6)
      wx = b.weights[v * 3 + k];
  EXPECT_NEAR(wx, 1.0f, 1e-4f);
  EXPECT_EQ(b.used[6], 1);
}

TEST(Wardrobe, ALargeItemIsBoundInChunksAsItWouldBeWhole)
{
  // more item vertices than one chunk: each still lands on the body vertex under it
  const int  n    = 80;
  const auto body = fv::mesh::build(grid(n, 0));
  const auto item = fv::mesh::build(grid(n, 0.5f));
  ASSERT_GT(item.cagePositions.size() / 3, 4096u);
  const auto b = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  for(size_t v = 0; v < item.cagePositions.size() / 3; v++)
  {
    float under = 0;
    for(int k = 0; k < 3; k++)
      if(b.verts[v * 3 + k] == v)
        under += b.weights[v * 3 + k];
    ASSERT_NEAR(under, 1.0f, 1e-4f) << v;
  }
}

TEST(Wardrobe, AStrandBindsAlongItsLengthToTheBodyItLiesOn)
{
  // a strand lying 1 cm over the body binds point by point as bindToSurface does; once it hangs
  // away past the reach, its points ride with the last point that was within reach
  const auto body = fv::mesh::build(grid(10, 0));
  std::vector<float> strand;
  for(int k = 0; k <= 12; k++)
  {
    // along x at 1 cm for the first six points, then rising away from the body
    const float x = 0.5f + 0.75f * float(k), z = k <= 6 ? 1.f : 1.f + 2.f * float(k - 6);
    strand.insert(strand.end(), {x, 4.3f, z});
  }
  std::vector<int32_t> start{0, 13}, verts;
  for(int k = 0; k <= 12; k++)
    verts.push_back(k);
  const auto exact = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, strand);
  const auto along = fv::wardrobe::bindStrands(body.cagePositions, body.dsonTris, strand, start, verts, 3.f);
  // within 3 cm (z up to 3): the same triangle and weights
  for(int k = 0; k <= 7; k++)
    for(int c = 0; c < 3; c++)
    {
      EXPECT_EQ(along.verts[k * 3 + c], exact.verts[k * 3 + c]) << k;
      EXPECT_FLOAT_EQ(along.weights[k * 3 + c], exact.weights[k * 3 + c]) << k;
    }
  // past it: the binding of point 7, the last within reach
  for(int k = 8; k <= 12; k++)
    for(int c = 0; c < 3; c++)
    {
      EXPECT_EQ(along.verts[k * 3 + c], along.verts[7 * 3 + c]) << k;
      EXPECT_FLOAT_EQ(along.weights[k * 3 + c], along.weights[7 * 3 + c]) << k;
    }
}

TEST(Wardrobe, AStrandWhoseRootIsFarFromTheBodyStillBinds)
{
  const auto         body = fv::mesh::build(grid(4, 0));
  const std::vector<float> strand{2, 2, 10, 2, 2, 11};
  const std::vector<int32_t> start{0, 2}, verts{0, 1};
  const auto exact = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, strand);
  const auto along = fv::wardrobe::bindStrands(body.cagePositions, body.dsonTris, strand, start, verts, 3.f);
  for(int c = 0; c < 3; c++)
  {
    EXPECT_EQ(along.verts[c], exact.verts[c]);
    EXPECT_EQ(along.verts[3 + c], along.verts[c]);
  }
  float w = 0;
  for(int c = 0; c < 3; c++)
    w += along.weights[c];
  EXPECT_NEAR(w, 1.f, 1e-5f);
}

TEST(Wardrobe, AStrandStoredTipFirstBindsAsItWouldRootFirst)
{
  // the strand above, its points listed from the tip: the tip, 13 cm off the body, would otherwise
  // bind to the body under it and the points out of reach would ride with that
  const auto         body = fv::mesh::build(grid(10, 0));
  std::vector<float> strand;
  for(int k = 0; k <= 12; k++)
  {
    const float x = 0.5f + 0.75f * float(k), z = k <= 6 ? 1.f : 1.f + 2.f * float(k - 6);
    strand.insert(strand.end(), {x, 4.3f, z});
  }
  std::vector<int32_t> start{0, 13}, rootFirst, tipFirst;
  for(int k = 0; k <= 12; k++)
  {
    rootFirst.push_back(k);
    tipFirst.push_back(12 - k);
  }
  const auto a = fv::wardrobe::bindStrands(body.cagePositions, body.dsonTris, strand, start, rootFirst, 3.f);
  const auto b = fv::wardrobe::bindStrands(body.cagePositions, body.dsonTris, strand, start, tipFirst, 3.f);
  EXPECT_EQ(a.verts, b.verts);
  EXPECT_EQ(a.weights, b.weights);
}

TEST(Wardrobe, AStrandWithBothEndsOnTheBodyKeepsTheOrderItIsStoredIn)
{
  // an arch from 1 cm over the body up to 10 cm and down to a tip nearer the body than its root:
  // the points out of reach ride with the last one within reach before them, on the root's side
  const auto         body = fv::mesh::build(grid(10, 0));
  std::vector<float> strand;
  for(int k = 0; k <= 12; k++)
  {
    const float x = 0.5f + 0.75f * float(k), z = k == 12 ? 0.5f : 1.f + 9.f * std::sin(3.14159265f * float(k) / 12.f);
    strand.insert(strand.end(), {x, 4.3f, z});
  }
  std::vector<int32_t> start{0, 13}, verts;
  for(int k = 0; k <= 12; k++)
    verts.push_back(k);
  const auto exact = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, strand);
  const auto along = fv::wardrobe::bindStrands(body.cagePositions, body.dsonTris, strand, start, verts, 3.f);
  // z = 1 + 9 sin(pi k / 12) is within 3 cm at the two ends alone
  for(int k : {0, 12})
    for(int c = 0; c < 3; c++)
      EXPECT_EQ(along.verts[k * 3 + c], exact.verts[k * 3 + c]) << k;
  for(int k = 1; k <= 11; k++)
    for(int c = 0; c < 3; c++)
    {
      EXPECT_EQ(along.verts[k * 3 + c], along.verts[c]) << k;
      EXPECT_FLOAT_EQ(along.weights[k * 3 + c], along.weights[c]) << k;
    }
}

TEST(Wardrobe, GraftPairsOverrideTheClosestPoint)
{
  const auto           body  = fv::mesh::build(grid(4, 0));
  const auto           item  = fv::mesh::build(grid(2, 0.5f));
  const std::vector<int32_t> pairs{0, 24};  // item vertex 0 is welded to body vertex 24
  const auto b = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions, pairs);
  EXPECT_EQ(b.verts[0], 24u);
  EXPECT_FLOAT_EQ(b.weights[0], 1.f);
}

TEST(Wardrobe, AMorphOnTheBodyMovesTheItemThatFollowsIt)
{
  std::vector<std::shared_ptr<fv::mesh::BuiltMesh>> meshes{std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(4, 0))),
                                                           std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(2, 0.5f)))};
  fv::morph::MorphSet morphs([&](int i) { return meshes[size_t(i)].get(); });
  // a morph that lifts body vertex (1, 1) by 2
  fv::mesh::CageDeltas d;
  d.verts  = {6};
  d.deltas = {0.f, 0.f, 2.f};
  auto target = std::make_shared<fv::mesh::RenderTarget>(fv::mesh::target(*meshes[0], d));
  morphs.addTarget("Lift", 0, target);

  const auto binding = std::make_shared<fv::wardrobe::SurfaceBinding>(
      fv::wardrobe::bindToSurface(meshes[0]->cagePositions, meshes[0]->dsonTris, meshes[1]->cagePositions));
  fv::morph::Follower f;
  f.mesh    = 1;
  f.source  = 0;
  f.project = [binding, body = meshes[0], item = meshes[1], scratch = std::make_shared<std::vector<float>>()](const fv::mesh::RenderTarget& t) {
    return fv::wardrobe::projectTarget(*binding, nullptr, *body, *item, t, *scratch);
  };
  morphs.addFollower(f);
  // the drawn vertex of the item's cage vertex (1, 1)
  size_t drawn = 0;
  for(size_t d = 0; d < meshes[1]->renderIndex.size(); d++)
    if(meshes[1]->renderIndex[d] == 4)
      drawn = d;
  const auto z = [&] { return meshes[1]->position[drawn * 3 + 2]; };
  const float before = z();
  morphs.set("Lift", 1.0);
  morphs.flush();
  const float after = z();
  EXPECT_NEAR(after - before, 2.0f, 1e-4f);
  // back to zero, back where it was
  morphs.set("Lift", 0.0);
  morphs.flush();
  EXPECT_NEAR(z(), before, 1e-4f);
  // taking the mesh off stops it following
  morphs.removeMeshes({1});
  morphs.set("Lift", 1.0);
  morphs.flush();
  EXPECT_NEAR(z(), before, 1e-4f);
}

// A body morph projected onto a long item (a hair's strands, a gown at SubD) is worked a chunk at a
// time side by side: the result is the one-thread loop's, bit for bit, through the refinement and
// every UV split of a seam.
TEST(Wardrobe, AProjectionOntoALongItemIsTheOneThreadsResultBitForBit)
{
  const auto body = fv::mesh::build(grid(40, 0));
  // 70,000 cage vertices over the body, refined once, with a seam down column 100
  constexpr int       n = 263;
  fv::mesh::MeshInput in = grid(n, 0.5f);
  for(size_t i = 0; i < in.positions.size(); i += 3)
  {
    in.positions[i] *= 40.f / n;
    in.positions[i + 1] *= 40.f / n;
  }
  for(int y = 0; y < n; y++)
  {
    const int poly = y * n + 100, a = y * (n + 1) + 100;
    for(int v : {a, a + n + 1})
    {
      in.overrides.insert(in.overrides.end(), {poly, v, int(in.uvs.size() / 2)});
      in.uvs.insert(in.uvs.end(), {0.5f, float(y) / n});
    }
  }
  in.level        = 1;
  const auto item = fv::mesh::build(in);
  ASSERT_GT(item.renderIndex.size(), item.cagePositions.size() / 3);  // the seam split some
  const auto binding = fv::wardrobe::bindToSurface(body.cagePositions, body.dsonTris, item.cagePositions);
  // a morph over most of the body
  fv::mesh::CageDeltas d;
  for(int v = 0; v < 41 * 41; v++)
    if(v % 41 < 30)
    {
      d.verts.push_back(v);
      d.deltas.insert(d.deltas.end(), {float(std::sin(v * 0.3)), 0.25f, float(v % 7) * 0.1f});
    }
  const auto bodyTarget = fv::mesh::target(body, d);

  // as projectTarget was: one loop over the item, then the target made from what moves
  std::vector<float> scratch(size_t(body.dsonVertexCount) * 3, 0.f);
  for(size_t i = 0; i < bodyTarget.cageIndices.size(); i++)
    std::copy_n(&bodyTarget.cageDeltas[i * 3], 3, &scratch[size_t(bodyTarget.cageIndices[i]) * 3]);
  const size_t       nv = size_t(item.dsonVertexCount);
  std::vector<float> out(nv * 3);
  for(size_t v = 0; v < nv; v++)
  {
    const size_t k = v * 3;
    const size_t a = binding.verts[k] * 3ull, b = binding.verts[k + 1] * 3ull, c = binding.verts[k + 2] * 3ull;
    for(int i = 0; i < 3; i++)
      out[k + i] = scratch[a + i] * binding.weights[k] + scratch[b + i] * binding.weights[k + 1] + scratch[c + i] * binding.weights[k + 2];
  }
  fv::mesh::CageDeltas moved;
  for(size_t v = 0; v < nv; v++)
    if(std::abs(out[v * 3]) + std::abs(out[v * 3 + 1]) + std::abs(out[v * 3 + 2]) >= 1e-5f)
    {
      moved.verts.push_back(int32_t(v));
      moved.deltas.insert(moved.deltas.end(), {out[v * 3], out[v * 3 + 1], out[v * 3 + 2]});
    }
  fv::mesh::RenderTarget       want;
  fv::mesh::RenderTarget::Drawn wantDrawn;
  for(size_t i = 0; i < moved.verts.size(); i++)
  {
    want.cageIndices.push_back(uint32_t(moved.verts[i]));
    want.cageDeltas.insert(want.cageDeltas.end(), &moved.deltas[i * 3], &moved.deltas[i * 3] + 3);
  }
  const auto render = fv::subdiv::refineSparse(*item.refinement, want.cageIndices, want.cageDeltas, 3);
  for(size_t i = 0; i < render.indices.size(); i++)
  {
    const float x = render.data[i * 3], y = render.data[i * 3 + 1], z = render.data[i * 3 + 2];
    if(x == 0 && y == 0 && z == 0)
      continue;
    for(int32_t s = item.firstSplit[render.indices[i]]; s >= 0; s = item.nextSplit[s])
    {
      wantDrawn.indices.push_back(uint32_t(s));
      wantDrawn.deltas.insert(wantDrawn.deltas.end(), {x, y, z});
    }
  }

  std::vector<float> scratchNow;
  const auto         got = fv::wardrobe::projectTarget(binding, nullptr, body, item, bodyTarget, scratchNow);
  ASSERT_TRUE(got);
  ASSERT_GT(moved.verts.size(), 40000u);
  const auto same = [](const auto& a, const auto& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0; };
  EXPECT_TRUE(same(got->cageIndices, want.cageIndices));
  EXPECT_TRUE(same(got->cageDeltas, want.cageDeltas));
  EXPECT_TRUE(same(got->drawn(item).indices, wantDrawn.indices));
  EXPECT_TRUE(same(got->drawn(item).deltas, wantDrawn.deltas));
}

// A follower given its projections ahead (a wear's worker makes them) moves exactly as one that
// projects at commit, and project() is not asked for those targets.
TEST(Wardrobe, AFollowerWithProjectionsMadeAheadMovesLikeOneProjectingAtCommit)
{
  const auto build = [] {
    return std::vector<std::shared_ptr<fv::mesh::BuiltMesh>>{std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(4, 0))),
                                                             std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(2, 0.5f)))};
  };
  const auto           fresh = build();
  fv::mesh::CageDeltas d;
  d.verts  = {6, 7};
  d.deltas = {0.f, 0.f, 2.f, 0.f, 0.5f, 1.f};
  std::vector<float> after[2];
  for(int which = 0; which < 2; which++)
  {
    auto                meshes = build();
    fv::morph::MorphSet morphs([&](int i) { return meshes[size_t(i)].get(); });
    auto                target = std::make_shared<fv::mesh::RenderTarget>(fv::mesh::target(*meshes[0], d));
    morphs.addTarget("Lift", 0, target);
    morphs.set("Lift", 0.75);
    morphs.flush();
    const auto binding = std::make_shared<fv::wardrobe::SurfaceBinding>(
        fv::wardrobe::bindToSurface(meshes[0]->cagePositions, meshes[0]->dsonTris, meshes[1]->cagePositions));
    const auto project = [binding, body = meshes[0], item = meshes[1]](const fv::mesh::RenderTarget& t) {
      std::vector<float> scratch;
      return fv::wardrobe::projectTarget(*binding, nullptr, *body, *item, t, scratch);
    };
    fv::morph::Follower f;
    f.mesh   = 1;
    f.source = 0;
    if(which == 0)
      f.project = project;
    else
    {
      f.preprojected.emplace_back(target, project(*target));  // as the worker does, before the commit
      f.project = [](const fv::mesh::RenderTarget&) -> fv::dson::TargetPtr {
        ADD_FAILURE() << "a target projected ahead was projected again";
        return nullptr;
      };
    }
    morphs.addFollower(f);
    morphs.flush();
    after[which] = meshes[1]->position;
  }
  ASSERT_EQ(after[0].size(), after[1].size());
  EXPECT_EQ(std::memcmp(after[0].data(), after[1].data(), after[0].size() * sizeof(float)), 0);
  // and the item did move
  EXPECT_NE(std::memcmp(after[0].data(), fresh[1]->position.data(), after[0].size() * sizeof(float)), 0);
}

// Many channels dialled at once (a pose's correctives): the followers' projections are made side by
// side before the weights go in, each once, and the items end where projecting one at a time, as a
// follower added afterwards does, puts them.
TEST(Wardrobe, ProjectionsMadeSideBySideMoveTheItemsAsOneAtATime)
{
  constexpr int kItems = 3, kChannels = 5;
  const auto    build  = [] {
    std::vector<std::shared_ptr<fv::mesh::BuiltMesh>> m{std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(6, 0)))};
    for(int k = 0; k < kItems; k++)
      m.push_back(std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(3, 0.5f + 0.25f * float(k)))));
    return m;
  };
  std::vector<std::pair<std::string, double>> dials;
  for(int c = 0; c < kChannels; c++)
    dials.emplace_back("Lift" + std::to_string(c), 0.25 + 0.2 * c);
  std::vector<float> after[2][kItems];
  int                calls[2][kItems] = {};
  for(int which = 0; which < 2; which++)
  {
    auto                meshes = build();
    fv::morph::MorphSet morphs([&](int i) { return meshes[size_t(i)].get(); });
    for(int c = 0; c < kChannels; c++)
    {
      fv::mesh::CageDeltas d;
      d.verts  = {8 + c, 16 + c};
      d.deltas = {0.f, 0.f, 1.f + float(c), 0.5f, 0.f, 0.25f * float(c)};
      morphs.addTarget(dials[size_t(c)].first, 0, std::make_shared<fv::mesh::RenderTarget>(fv::mesh::target(*meshes[0], d)));
    }
    std::atomic<int> counted[kItems] = {};
    const auto       follow          = [&](int k) {
      const auto binding = std::make_shared<fv::wardrobe::SurfaceBinding>(
          fv::wardrobe::bindToSurface(meshes[0]->cagePositions, meshes[0]->dsonTris, meshes[size_t(k) + 1]->cagePositions));
      fv::morph::Follower f;
      f.mesh    = k + 1;
      f.source  = 0;
      f.project = [binding, body = meshes[0], item = meshes[size_t(k) + 1], n = &counted[k]](const fv::mesh::RenderTarget& t) {
        (*n)++;
        std::vector<float> scratch;
        return fv::wardrobe::projectTarget(*binding, nullptr, *body, *item, t, scratch);
      };
      morphs.addFollower(f);
    };
    if(which == 0)
    {
      // followers first, then every dial in one batch: the projections are made ahead
      for(int k = 0; k < kItems; k++)
        follow(k);
      morphs.setMany(dials);
    }
    else
    {
      // the dials first, then each follower, which projects what is dialled one at a time
      morphs.setMany(dials);
      for(int k = 0; k < kItems; k++)
        follow(k);
    }
    morphs.flush();
    for(int k = 0; k < kItems; k++)
    {
      after[which][k] = meshes[size_t(k) + 1]->position;
      calls[which][k] = counted[k].load();
    }
  }
  const auto fresh = build();
  for(int k = 0; k < kItems; k++)
  {
    EXPECT_EQ(calls[0][k], kChannels) << "item " << k;  // each target once
    EXPECT_EQ(calls[1][k], kChannels) << "item " << k;
    ASSERT_EQ(after[0][k].size(), after[1][k].size());
    float moved = 0;
    for(size_t i = 0; i < after[0][k].size(); i++)
    {
      EXPECT_NEAR(after[0][k][i], after[1][k][i], 1e-5f) << "item " << k << " coordinate " << i;
      moved = std::max(moved, std::abs(after[0][k][i] - fresh[size_t(k) + 1]->position[i]));
    }
    EXPECT_GT(moved, 0.1f) << "item " << k;  // and it did move
  }
}

TEST(Wardrobe, HiddenPolygonsLeaveTheDrawnTriangles)
{
  auto mesh = fv::mesh::build(grid(2, 0));
  const size_t all = mesh.fullIndex.size();
  mesh.hiddenPolys["w1"] = {0, 1};
  const auto drawn       = fv::mesh::drawnTriangles(mesh);
  EXPECT_EQ(drawn.index.size(), all - 2 * 6);  // two quads of two triangles
  ASSERT_EQ(drawn.groups.size(), 1u);
  EXPECT_EQ(drawn.groups[0].count, drawn.index.size());
  mesh.hiddenPolys.erase("w1");
  EXPECT_EQ(fv::mesh::drawnTriangles(mesh).index.size(), all);
}

// A geograft can hide every polygon of a surface group. The build still writes that group's primitive,
// with no triangles, and the scene leaves it out rather than make an empty index buffer (which fails
// to allocate and stops the renderer); the other group is drawn.
TEST(Wardrobe, AGroupAGraftHidesWholeIsNotDrawn)
{
  fv::mesh::MeshInput input = grid(2, 0);
  input.materialCount       = 2;
  input.polyMaterial        = {0, 0, 1, 1};
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  scene.materials.resize(2);
  fv::Mesh m;
  m.geo                     = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(input));
  m.geo->hiddenPolys["w1"] = {2, 3};
  m.materials               = {0, 1};
  scene.meshes.push_back(m);
  fv::Node node;
  node.kind = fv::NodeKind::eMesh;
  node.mesh = 0;
  scene.add(node, root);
  auto built = fv::build::buildGltf(scene, root, {});
  ASSERT_EQ(built.model.meshes.size(), 1u);
  ASSERT_EQ(built.model.meshes[0].primitives.size(), 2u);
  EXPECT_EQ(built.model.accessors[size_t(built.model.meshes[0].primitives[1].indices)].count, 0u);

  nvvkgltf::Scene loaded;
  loaded.takeModel(std::move(built.model));
  EXPECT_EQ(loaded.getRenderPrimitives().size(), 1u);
  EXPECT_GE(loaded.getMeshRenderPrimitive(0, 0), 0);
  EXPECT_EQ(loaded.getMeshRenderPrimitive(0, 1), -1);
  EXPECT_EQ(loaded.getRenderNodes().size(), 1u);
}

TEST(Fit, BonesAliasAcrossGenerationsByAnatomicalSlot)
{
  using fv::wardrobe::matchBone;
  const std::vector<std::string> g9{"hip", "l_thigh", "r_thigh", "l_shin", "l_foot", "l_toes", "l_upperarm", "l_forearm", "l_shoulder", "l_hand"};
  // the exact name wins
  EXPECT_EQ(matchBone("l_shin", g9), std::optional<std::string>("l_shin"));
  // Genesis 8 names reduce to the same slots; left never matches right
  EXPECT_EQ(matchBone("lThighBend", g9), std::optional<std::string>("l_thigh"));
  EXPECT_EQ(matchBone("rThighBend", g9), std::optional<std::string>("r_thigh"));
  EXPECT_EQ(matchBone("lShldrBend", g9), std::optional<std::string>("l_upperarm"));
  EXPECT_EQ(matchBone("lCollar", g9), std::optional<std::string>("l_shoulder"));
  EXPECT_EQ(matchBone("lToe", g9), std::optional<std::string>("l_toes"));
  // a toe the figure lacks falls back toward the foot
  EXPECT_EQ(matchBone("lBigToe", g9), std::optional<std::string>("l_toes"));
  EXPECT_FALSE(matchBone("lThighBend", {"r_thigh"}));
}

TEST(Fit, TwistBonesInheritTheirBendParentInsteadOfCopyingTheLeaderTwice)
{
  using namespace fv::wardrobe;
  const std::vector<std::string> g9{"l_thigh"};
  const auto plan = planLeaders({{"lThighBend", std::nullopt}, {"lThighTwist", std::string("lThighBend")}}, g9);
  EXPECT_EQ(plan.follow.count("lThighBend"), 1u);
  EXPECT_EQ(plan.follow.count("lThighTwist"), 0u);
  EXPECT_EQ(plan.inherit.count("lThighTwist"), 1u);
}

TEST(Fit, IdentifiesTheGenerationAnOutfitWasMadeFor)
{
  using namespace fv::wardrobe;
  EXPECT_EQ(generationFromPath("data/DAZ 3D/Genesis 8/Female/Wardrobe/Dress.dsf"), std::optional<Generation>(Generation::eG8));
  EXPECT_EQ(generationFromPath("data/Genesis%209/Clothing/Top.duf"), std::optional<Generation>(Generation::eG9));
  EXPECT_EQ(generationFromBones({"hip", "l_thigh", "l_shin"}), std::optional<Generation>(Generation::eG9));
  EXPECT_EQ(generationFromBones({"hip", "lThighBend", "lThighTwist"}), std::optional<Generation>(Generation::eG8));
  EXPECT_TRUE(sameRig(Generation::eG8, Generation::eG81));
  EXPECT_FALSE(sameRig(Generation::eG8, Generation::eG9));
  EXPECT_TRUE(sameTopology(Generation::eG3, Generation::eG81));
}

TEST(Fit, ARigidPlanMovesABoneToTheLeadersJointAndTurnsItOntoItsLimb)
{
  using namespace fv::wardrobe;
  // a source leg straight down from (0, 90, 0); the target's hip is at (5, 100, 0) and its knee leans out
  const std::vector<SkelBone> source{{"lThighBend", std::nullopt, {0, 90, 0}}, {"lShin", std::string("lThighBend"), {0, 50, 0}}};
  const std::vector<SkelBone> target{{"l_thigh", std::nullopt, {5, 100, 0}}, {"l_shin", std::string("l_thigh"), {10, 60, 0}}};
  const auto plan = rigidPlan(source, target);
  ASSERT_EQ(plan.count("lThighBend"), 1u);
  const Rigid& r = plan.at("lThighBend");
  EXPECT_NEAR(r.to[0], 5, 1e-9);
  EXPECT_NEAR(r.to[1], 100, 1e-9);
  // the source knee, put through it, lands on the target knee (the limb is about the same length)
  const std::vector<float> knee{0, 50, 0};
  fv::mesh::SkinInput      skin;
  skin.start  = {0, 1};
  skin.bone   = {0};
  skin.weight = {1.f};
  const auto moved = rigidPositions(knee, skin, {&r});
  EXPECT_NEAR(moved[0], 10, 0.5f);
  EXPECT_NEAR(moved[1], 60, 0.5f);
}

// Genesis 8's pectoral joint is deep in the chest and Genesis 9's near its middle: a bust carried from
// one to the other stood centimetres in front of the body (MantaMade's M7UP Top on Polina 9)
TEST(Fit, ABustMovesWithTheChestNotOntoTheTargetsPectoralJoint)
{
  using namespace fv::wardrobe;
  const std::vector<SkelBone> source{{"chestUpper", std::nullopt, {0, 130, -3}}, {"lPectoral", std::string("chestUpper"), {3, 132, -7.5}}};
  const std::vector<SkelBone> target{{"spine4", std::nullopt, {0, 124, -1}}, {"l_pectoral", std::string("spine4"), {4.3, 126, 0.9}}};
  const auto plan = rigidPlan(source, target);
  ASSERT_EQ(plan.count("chestUpper"), 1u);
  ASSERT_EQ(plan.count("lPectoral"), 1u);

  // a point on the bust, all on the pectoral, lands where the same point on the chest lands
  const std::vector<float> bust{7, 132, 13.5};
  fv::mesh::SkinInput      skin;
  skin.start  = {0, 1};
  skin.bone   = {0};
  skin.weight = {1.f};
  const auto onPectoral = rigidPositions(bust, skin, {&plan.at("lPectoral")});
  const auto onChest    = rigidPositions(bust, skin, {&plan.at("chestUpper")});
  for(int k = 0; k < 3; k++)
    EXPECT_NEAR(onPectoral[size_t(k)], onChest[size_t(k)], 1e-5f);
  // forward by the chest's own move, not the 8.4 cm between the two generations' joints
  EXPECT_NEAR(onPectoral[2], 13.5f + 2.0f, 1.0f);
}

// The glTF holds each mesh's own bytes: a mesh past 16-bit indices with polygons a worn item hides,
// a small one beside it, both with UVs flipped and as they are, every view apart and the gaps zero.
TEST(Wardrobe, TheBuiltGltfHoldsTheScenesMeshes)
{
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  scene.materials.resize(2);
  for(int n : {300, 4})
  {
    fv::Mesh m;
    m.geo       = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(n, float(n))));
    m.materials = {n == 300 ? 0 : 1};
    if(n == 300)
      m.geo->hiddenPolys["w1"] = {0, 1, 2, 500, 89999};
    scene.meshes.push_back(m);
    fv::Node node;
    node.kind = fv::NodeKind::eMesh;
    node.mesh = int(scene.meshes.size()) - 1;
    scene.add(node, root);
  }
  ASSERT_GT(scene.meshes[0].geo->position.size() / 3, 65535u);
  for(bool flipped : {false, true})
  {
    fv::build::BuildOptions options;
    options.texturesFlipped = flipped;
    const auto built        = fv::build::buildGltf(scene, root, options);
    EXPECT_EQ(built.model.meshes.size(), 2u);
    for(const std::string& p : fv::build::checkBuild(scene, built, options))
      ADD_FAILURE() << p << (flipped ? " (flipped)" : "");
    // and the hidden polygons' triangles are not drawn
    EXPECT_LT(built.model.accessors[size_t(built.model.meshes[size_t(built.meshOf[0])].primitives[0].indices)].count,
              scene.meshes[0].geo->fullIndex.size());
  }
}

TEST(Wardrobe, LightUnderTheSkinPassesThroughWhatIsWornButNotWhatBoundsTheBody)
{
  using fv::mat::Kind;
  fv::SceneGraph scene;
  scene.add(fv::Node{});
  // the body (skin, lashes), a shirt, a geograft whose path calls it clothing, Genesis 9's eyes
  // (sclera and the wet film), and a sleeve that shares the body's torso material
  for(Kind k : {Kind::eSkin, Kind::eLashes, Kind::eOther, Kind::eSkin, Kind::eEye, Kind::eCornea})
  {
    fv::Material m;
    m.surface.kind = k;
    scene.materials.push_back(m);
  }
  for(std::vector<int> mats : std::vector<std::vector<int>>{{0, 1}, {2}, {3}, {4, 5}, {0}})
  {
    fv::Mesh m;
    m.materials = mats;
    scene.meshes.push_back(m);
  }
  std::vector<fv::WornItem> items(3);
  items[0].kind   = "Clothing";
  items[0].meshes = {1, 4};
  items[1].kind        = "Clothing";
  items[1].meshes      = {2};
  items[1].hiddenPolys = {7, 8};
  items[2].kind   = "Anatomy";
  items[2].meshes = {3};
  const std::vector<bool> through = fv::walkThroughMaterials(scene, items);
  EXPECT_EQ(through, (std::vector<bool>{false, true, true, false, false, true}));

  // the build says so in the glTF materials' extras, for the renderer's walk mask
  fv::build::BuildOptions options;
  options.walkThrough = through;
  const auto built    = fv::build::buildGltf(scene, 0, options);
  ASSERT_EQ(built.model.materials.size(), through.size());
  for(size_t i = 0; i < through.size(); i++)
  {
    const tinygltf::Value& e = built.model.materials[i].extras;
    EXPECT_EQ(e.Has("fvWalkThrough") && e.Get("fvWalkThrough").Get<bool>(), bool(through[i])) << i;
    EXPECT_TRUE(e.Has("fvId")) << i;
  }
}

// Taking an item off leaves its meshes and materials in the scene (Node::removed), so nothing it
// drew still counts: a mesh that only a taken-off item draws is not in use, one no node draws is.
TEST(Wardrobe, ATakenOffItemsMeshesAreNotInUse)
{
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  scene.meshes.resize(4);
  fv::Node shown;
  shown.kind = fv::NodeKind::eMesh;
  shown.mesh = 0;
  scene.add(shown, root);
  const int item = scene.add(fv::Node{}, root);
  fv::Node  worn = shown;
  worn.mesh      = 1;
  scene.add(worn, item);
  // mesh 2 is drawn twice, once by the item and once by the figure
  worn.mesh = 2;
  scene.add(worn, item);
  scene.add(worn, root);
  // mesh 3: no node draws it
  EXPECT_EQ(fv::liveMeshes(scene), (std::vector<bool>{true, true, true, true}));
  scene.nodes[size_t(item)].removed = true;
  EXPECT_TRUE(fv::takenOff(scene, item + 1));
  EXPECT_FALSE(fv::takenOff(scene, root));
  EXPECT_EQ(fv::liveMeshes(scene), (std::vector<bool>{true, false, true, true}));
}

// A worn shirt and a taken-off jacket made with the same surface: the jacket's mesh is still in the
// scene, but no longer bounds the body, so light keeps passing through the shirt.
TEST(Wardrobe, ATakenOffItemDoesNotBoundTheBody)
{
  using fv::mat::Kind;
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  for(Kind k : {Kind::eSkin, Kind::eOther})
  {
    fv::Material m;
    m.surface.kind = k;
    scene.materials.push_back(m);
  }
  std::vector<int> itemRoot;
  for(int mat : {0, 1, 1})
  {
    fv::Mesh m;
    m.materials = {mat};
    scene.meshes.push_back(m);
    fv::Node node;
    node.kind = fv::NodeKind::eMesh;
    node.mesh = int(scene.meshes.size()) - 1;
    const int parent = mat == 0 ? root : scene.add(fv::Node{}, root);
    scene.add(node, parent);
    itemRoot.push_back(parent);
  }
  std::vector<fv::WornItem> items(2);
  items[0].kind   = "Clothing";
  items[0].meshes = {1};
  items[1].kind   = "Clothing";
  items[1].meshes = {2};
  EXPECT_EQ(fv::walkThroughMaterials(scene, items), (std::vector<bool>{false, true}));
  // the jacket comes off: out of the worn items, its node taken off
  items.pop_back();
  scene.nodes[size_t(itemRoot[2])].removed = true;
  EXPECT_EQ(fv::walkThroughMaterials(scene, items), (std::vector<bool>{false, true}));
}

// A taken-off eye moisture is no tear film over the eyes that stay, and a taken-off mesh gives an
// iris no discs: the build lowers what is left as if the item had never been worn.
TEST(Wardrobe, ATakenOffItemLeavesNoTearFilmOrIrisDiscs)
{
  using fv::mat::Kind;
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  fv::Material   iris;
  iris.surface.kind = Kind::eEye;
  iris.surface.name = "Irises";
  fv::Material moisture;
  moisture.surface.kind       = Kind::eCornea;
  moisture.surface.name       = "EyeMoisture";
  moisture.surface.refraction = fv::mat::Surface::Refraction{1.f, 1.376f};
  scene.materials = {iris, moisture};
  // the figure's own iris, a second iris and the moisture on an item
  const int item = scene.add(fv::Node{}, root);
  for(auto [mat, parent] : std::vector<std::pair<int, int>>{{0, root}, {0, item}, {1, item}})
  {
    fv::Mesh m;
    m.geo = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(1, float(scene.meshes.size()))));
    // each mesh a small UV island of its own, side by side
    const float u0 = 0.1f + 0.3f * float(scene.meshes.size());
    for(size_t v = 0; v * 2 < m.geo->uv.size(); v++)
    {
      m.geo->uv[v * 2]     = u0 + 0.1f * float(v % 2);
      m.geo->uv[v * 2 + 1] = 0.7f + 0.1f * float(v / 2);
    }
    m.materials = {mat};
    scene.meshes.push_back(m);
    fv::Node node;
    node.kind = fv::NodeKind::eMesh;
    node.mesh = int(scene.meshes.size()) - 1;
    scene.add(node, parent);
  }
  {
    const auto built = fv::build::buildGltf(scene, root, {});
    EXPECT_TRUE(built.lowering[0].underTearFilm);
    EXPECT_EQ(built.lowering[0].irisDiscs.size(), 2u);
  }
  scene.nodes[size_t(item)].removed = true;
  const auto built = fv::build::buildGltf(scene, root, {});
  EXPECT_FALSE(built.lowering[0].underTearFilm);
  EXPECT_EQ(built.lowering[0].irisDiscs.size(), 1u);
  for(const std::string& p : fv::build::checkBuild(scene, built, {}))
    ADD_FAILURE() << p;
}

// Strand ribbons make their surface double-sided; a taken-off ribbon item no longer does it to the
// polygon mesh that shares the surface.
TEST(Wardrobe, ATakenOffRibbonLeavesItsSurfaceOneSided)
{
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  scene.materials.resize(1);
  const int item = scene.add(fv::Node{}, root);
  for(int parent : {root, item})
  {
    fv::Mesh m;
    m.geo = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(1, float(scene.meshes.size()))));
    if(parent == item)
      m.geo->ribbon = fv::mesh::Ribbon{};
    m.materials = {0};
    scene.meshes.push_back(m);
    fv::Node node;
    node.kind = fv::NodeKind::eMesh;
    node.mesh = int(scene.meshes.size()) - 1;
    scene.add(node, parent);
  }
  const auto built = fv::build::buildGltf(scene, root, {});
  EXPECT_TRUE(built.model.materials[0].doubleSided);
  // a host lowering the material again with the build's options (an edit) keeps it double-sided,
  // an edit that says otherwise included
  scene.materials[0].edits.doubleSided = false;
  std::vector<std::string> used;
  EXPECT_TRUE(fv::build::materialToGltf(scene.materials[0], built.lowering[0], {}, used).doubleSided);
  scene.materials[0].edits.doubleSided.reset();
  scene.nodes[size_t(item)].removed = true;
  EXPECT_FALSE(fv::build::buildGltf(scene, root, {}).model.materials[0].doubleSided);
}

// The textures a host resolves ahead (textureRequests) are the ones the build then asks for: an
// iris with no colour edit and a skin under a tear film lower to the same requests either way.
TEST(Wardrobe, TheTexturesResolvedAheadAreTheOnesTheBuildAsksFor)
{
  using fv::mat::Kind;
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  fv::Material   iris;
  iris.surface.kind     = Kind::eEye;
  iris.surface.name     = "Irises";
  iris.surface.base.map = "/eyes/iris.jpg";
  fv::Material moisture;
  moisture.surface.kind       = Kind::eCornea;
  moisture.surface.name       = "EyeMoisture";
  moisture.surface.refraction = fv::mat::Surface::Refraction{1.f, 1.376f};
  fv::Material skin;
  skin.surface.kind     = Kind::eSkin;
  skin.surface.base.map = "/skin/torso.jpg";
  scene.materials       = {iris, moisture, skin};
  for(int mat : {0, 1, 2})
  {
    fv::Mesh m;
    m.geo       = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(1, float(mat))));
    m.materials = {mat};
    scene.meshes.push_back(m);
    fv::Node node;
    node.kind = fv::NodeKind::eMesh;
    node.mesh = mat;
    scene.add(node, root);
  }
  std::set<std::string> ahead;
  for(const fv::mat::TextureRequest& r : fv::build::textureRequests(scene))
    ahead.insert(fv::mat::requestKey(r));
  std::set<std::string> asked;
  fv::build::BuildOptions options;
  options.texture = [&](const fv::mat::TextureRequest& r) -> std::optional<fv::build::TextureImage> {
    asked.insert(fv::mat::requestKey(r));
    return std::nullopt;
  };
  (void)fv::build::buildGltf(scene, root, options);
  EXPECT_FALSE(asked.empty());
  EXPECT_EQ(asked, ahead);
}

// A figure built before any of its textures is ready still brings its repeating sampler, the one a
// texture baked in later takes; a file for other tools has none it doesn't use.
TEST(Wardrobe, AFigureWithoutTexturesStillHasItsSampler)
{
  fv::SceneGraph scene;
  const int      root = scene.add(fv::Node{});
  scene.materials.resize(1);
  fv::Mesh m;
  m.geo       = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(1, 0)));
  m.materials = {0};
  scene.meshes.push_back(m);
  fv::Node node;
  node.kind = fv::NodeKind::eMesh;
  node.mesh = 0;
  scene.add(node, root);
  const auto live = fv::build::buildGltf(scene, root, {});
  EXPECT_TRUE(live.model.textures.empty());
  ASSERT_EQ(live.model.samplers.size(), 1u);
  EXPECT_EQ(live.model.samplers[0].wrapS, TINYGLTF_TEXTURE_WRAP_REPEAT);
  EXPECT_EQ(live.model.samplers[0].wrapT, TINYGLTF_TEXTURE_WRAP_REPEAT);
  fv::build::BuildOptions file;
  file.forFile = true;
  EXPECT_TRUE(fv::build::buildGltf(scene, root, file).model.samplers.empty());
}

}  // namespace
