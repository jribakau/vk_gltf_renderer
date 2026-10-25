// A worn item is bound to the closest point on the body, and a body morph then moves it by what
// moved there; the body's polygons a graft replaces leave the drawn triangles. A port of
// src/lib/loaders/dson/bind.ts and follow.ts, with the MorphSet's follower.

#include <gtest/gtest.h>

#include <cstring>

#include <memory>

#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/wardrobe/fit/transfer.hpp>
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

}  // namespace
