// A worn item is bound to the closest point on the body, and a body morph then moves it by what
// moved there; the body's polygons a graft replaces leave the drawn triangles. A port of
// src/lib/loaders/dson/bind.ts and follow.ts, with the MorphSet's follower.

#include <gtest/gtest.h>

#include <memory>

#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/follow.hpp>

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

}  // namespace
