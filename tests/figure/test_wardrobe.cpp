// A worn item is bound to the closest point on the body, and a body morph then moves it by what
// moved there; the body's polygons a graft replaces leave the drawn triangles. A port of
// src/lib/loaders/dson/bind.ts and follow.ts, with the MorphSet's follower.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <numbers>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <figure/physics/skin.hpp>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/gltf/gltf_check.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/physics/flesh/follow.hpp>
#include <figure/subdiv/catmark.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/fit.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/wardrobe/fit/heel.hpp>
#include <figure/wardrobe/fit/transfer.hpp>
#include <figure/wardrobe/follow.hpp>

#include "../common/failing_allocations.hpp"
#include "scene_import_gltf.hpp"

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
    return fv::wardrobe::projectTarget(*binding, nullptr, *body, item->dsonVertexCount, t, *scratch);
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

// A follower holding "Lift"'s deltas for the item in a file of its own, already read.
fv::morph::Follower holdingLift(const std::vector<std::shared_ptr<fv::mesh::BuiltMesh>>& meshes, fv::dson::TargetPtr copy)
{
  fv::morph::Follower f;
  f.mesh   = 1;
  f.source = 0;
  f.hasOwn = [](const std::string& channel) { return channel == "Lift"; };
  f.own    = [copy](const std::string& channel) -> std::shared_future<std::vector<fv::dson::TargetPtr>> {
    if(channel != "Lift")
      return {};
    std::promise<std::vector<fv::dson::TargetPtr>> file;
    file.set_value({copy});
    return file.get_future().share();
  };
  const auto binding = std::make_shared<fv::wardrobe::SurfaceBinding>(
      fv::wardrobe::bindToSurface(meshes[0]->cagePositions, meshes[0]->dsonTris, meshes[1]->cagePositions));
  f.project = [binding, body = meshes[0], item = meshes[1]](const fv::mesh::RenderTarget& t) {
    std::vector<float> scratch;
    return fv::wardrobe::projectTarget(*binding, nullptr, *body, item->dsonVertexCount, t, scratch);
  };
  return f;
}

// The dial was set before the item went on, and its file has nothing for the body: wearing the item
// still applies it, and it comes off with the item. An item without a copy of its own takes nothing.
TEST(Wardrobe, WearingAnItemAfterAnEmptyDialWasSetAppliesIt)
{
  std::vector<std::shared_ptr<fv::mesh::BuiltMesh>> meshes{std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(4, 0))),
                                                           std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(grid(2, 0.5f)))};
  fv::morph::MorphSet morphs([&](int i) { return meshes[size_t(i)].get(); });
  morphs.addLazy("Lift", []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; });
  morphs.set("Lift", 0.5);
  morphs.settled();
  ASSERT_NE(morphs.channel("Lift"), nullptr);

  fv::mesh::CageDeltas d;
  d.verts  = {4};
  d.deltas = {0.f, 0.f, 2.f};
  const auto copy = std::make_shared<const fv::mesh::RenderTarget>(fv::mesh::target(*meshes[1], d));
  size_t     drawn = 0;
  for(size_t i = 0; i < meshes[1]->renderIndex.size(); i++)
    if(meshes[1]->renderIndex[i] == 4)
      drawn = i;
  const auto  z      = [&] { return meshes[1]->position[drawn * 3 + 2]; };
  const float before = z();

  fv::morph::Follower none = holdingLift(meshes, copy);
  none.hasOwn              = [](const std::string&) { return false; };
  morphs.addFollower(none);
  morphs.pump();
  EXPECT_EQ(z(), before);
  morphs.removeMeshes({1});

  morphs.addFollower(holdingLift(meshes, copy));
  morphs.pump();  // the copy, read on the pool, is taken here
  EXPECT_NEAR(z() - before, 1.0f, 1e-4f);
  morphs.set("Lift", 1.0);
  EXPECT_NEAR(z() - before, 2.0f, 1e-4f);
  morphs.set("Lift", 0.0);
  EXPECT_NEAR(z(), before, 1e-4f);
  EXPECT_NE(morphs.channel("Lift"), nullptr);
}

// A figure's own Morphs folder, skimmed: moving a dial with nothing in it reads the morphs whose files
// say that it drives them, the listing's dial and one a listing leaves out, by ids none of which are
// their files' names. Standing still it reads neither.
TEST(FigureMorphs, ADialBringsTheFilesThatReadIt)
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions       options;
  options.subdivision = 1;
  auto figure = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library})),
                                 "People/Test/Box Figure.duf", options);
  figure->settle();
  const fv::morph::Channel* pull = figure->morphs->channel("Reach Pull");
  ASSERT_NE(pull, nullptr);
  EXPECT_TRUE(pull->lazy);
  EXPECT_EQ(figure->morphs->channel("facs_cbs_ReachCombo"), nullptr);

  figure->morphs->set("Reach", 0.5);
  figure->settle();
  EXPECT_FALSE(figure->morphs->channel("Reach Pull")->lazy);
  EXPECT_NEAR(figure->morphs->weightOf("Reach Pull"), 0.5, 1e-6);
  EXPECT_DOUBLE_EQ(figure->morphs->get("Reach Pull"), 0);
  const fv::morph::Channel* combo = figure->morphs->channel("facs_cbs_ReachCombo");
  ASSERT_NE(combo, nullptr);
  EXPECT_TRUE(combo->hidden);
  EXPECT_NEAR(figure->morphs->weightOf("facs_cbs_ReachCombo"), 0.5, 1e-6);

  figure->morphs->set("Reach", 0.0);
  figure->settle();
  EXPECT_NEAR(figure->morphs->weightOf("Reach Pull"), 0, 1e-6);
  EXPECT_NEAR(figure->morphs->weightOf("facs_cbs_ReachCombo"), 0, 1e-6);
}

namespace {
std::unique_ptr<fv::Figure> boxFigure()
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions       options;
  options.subdivision = 1;
  auto figure = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library})),
                                 "People/Test/Box Figure.duf", options);
  figure->settle();
  return figure;
}
const char* const kBoxShirt = "People/Test/Clothing/Box Shirt.duf";
}  // namespace

// A conforming item keeps the binding its morphs follow the body by, for what moves the body after
// them (soft flesh, physics-plan step 9): the body's cage moved 1 cm up moves the shirt 1 cm up. Its
// mesh stands where the body's does, so the two meshes' posed spaces are one. Worn again, it has one
// binding a mesh still.
TEST(Wardrobe, AConformingItemKeepsItsBindingForTheFlesh)
{
  const auto           figure = boxFigure();
  const fv::WearResult worn   = figure->commitWear(figure->prepareWear(kBoxShirt, 1), kBoxShirt);
  figure->settle();
  const fv::WornItem* shirt = figure->item(worn.items.at(0));
  ASSERT_NE(shirt, nullptr);
  ASSERT_EQ(shirt->follows.size(), shirt->meshes.size());
  const fv::WornItem::Follow& follow = shirt->follows.front();
  EXPECT_NE(std::find(shirt->meshes.begin(), shirt->meshes.end(), follow.mesh), shirt->meshes.end());
  const fv::SceneGraph& scene = figure->loaded.scene;
  ASSERT_TRUE(follow.binding);
  ASSERT_EQ(follow.binding->verts.size(), size_t(scene.meshes[size_t(follow.mesh)].geo->dsonVertexCount) * 3);

  const int          body  = figure->wearInputs().bodyMesh;
  const int          count = scene.meshes[size_t(body)].geo->dsonVertexCount;
  std::vector<float> up(size_t(count) * 3, 0.f), out;
  for(int v = 0; v < count; v++)
    up[size_t(v) * 3 + 1] = 1;
  ASSERT_TRUE(fv::physics::followFlesh(*follow.binding, follow.rigidity.get(), up, nullptr, out));
  for(size_t v = 0; v < out.size() / 3; v++)
  {
    EXPECT_NEAR(out[v * 3], 0, 1e-5) << v;
    EXPECT_NEAR(out[v * 3 + 1], 1, 1e-5) << v;
  }
  const std::vector<int> nodeOf = fv::physics::nodeOfMesh(scene);
  const fv::Mat4         a = scene.world(nodeOf[size_t(body)]), b = scene.world(nodeOf[size_t(follow.mesh)]);
  for(size_t i = 0; i < 16; i++)
    EXPECT_NEAR(a.e[i], b.e[i], 1e-9) << i;

  ASSERT_TRUE(figure->removeItem(worn.items.at(0)));
  figure->settle();
  const fv::WearResult again = figure->commitWear(figure->prepareWear(kBoxShirt, 1), kBoxShirt);
  figure->settle();
  EXPECT_EQ(figure->item(again.items.at(0))->follows.size(), figure->item(again.items.at(0))->meshes.size());
}

// Taken off, an item's built mesh and skin go, so that try-ons don't add up. Its slot stays, the figure
// still builds, and it can go on again.
TEST(Wardrobe, TakingAnItemOffLetsGoOfItsMesh)
{
  const auto           figure = boxFigure();
  const fv::WearResult worn   = figure->commitWear(figure->prepareWear(kBoxShirt, 1), kBoxShirt);
  figure->settle();
  const std::vector<int> meshes = figure->item(worn.items.at(0))->meshes;
  ASSERT_FALSE(meshes.empty());
  std::vector<std::weak_ptr<fv::mesh::BuiltMesh>> built;
  for(int m : meshes)
    built.push_back(figure->loaded.scene.meshes[size_t(m)].geo);
  const size_t slots = figure->loaded.scene.meshes.size();

  ASSERT_TRUE(figure->removeItem(worn.items.at(0)));
  figure->settle();
  EXPECT_EQ(figure->loaded.scene.meshes.size(), slots);
  for(size_t i = 0; i < meshes.size(); i++)
  {
    const fv::Mesh& mesh = figure->loaded.scene.meshes[size_t(meshes[i])];
    EXPECT_TRUE(built[i].expired()) << meshes[i] << ": something still holds its mesh";
    EXPECT_FALSE(mesh.geo);
    EXPECT_TRUE(mesh.skin.skinIndex.empty());
    EXPECT_TRUE(mesh.cageSkin.start.empty());
  }
  (void)fv::build::buildGltf(figure->loaded.scene, figure->loaded.root, {});

  const fv::WearResult again = figure->commitWear(figure->prepareWear(kBoxShirt, 1), kBoxShirt);
  figure->settle();
  EXPECT_EQ(again.items.size(), worn.items.size());
  (void)fv::build::buildGltf(figure->loaded.scene, figure->loaded.root, {});
}

namespace {
// A box figure with a left foot, and a heel for it whose file points that foot 30 degrees
std::unique_ptr<fv::Figure> footFigure()
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions       options;
  options.subdivision = 1;
  auto figure = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library})),
                                 "People/Test/Foot Figure.duf", options);
  figure->settle();
  return figure;
}
const char* const kFootHeels  = "People/Test/Clothing/Foot Heels.duf";
const char* const kFootOutfit = "People/Test/Clothing/Foot Outfit.duf";
const char* const kFootBoots  = "People/Test/Clothing/Foot Boots.duf";
double            footX(const fv::Figure& figure)
{
  return figure.pose->posed("l_foot", "rotation", "x").value_or(NAN);
}
// The foot's turn as drawn, posed and driven together, in degrees
double footTurn(const fv::Figure& figure)
{
  const fv::Quat& q = figure.loaded.scene.nodes[size_t(figure.loaded.rigs[size_t(figure.loaded.mainRig)].find(figure.loaded.scene, "l_foot"))].quaternion;
  return 2 * std::acos(std::min(1.0, std::abs(q.w))) * 180 / std::numbers::pi;
}
}  // namespace

// A heel's file points the feet of the figure it goes on, one it was made for as well as one it is
// refitted to: the shoe rests shaped for that pose. Taken off, the feet go back.
TEST(Wardrobe, AHeelPointsTheFeetOfTheFigureItWasMadeFor)
{
  const auto figure = footFigure();
  ASSERT_TRUE(figure->pose);
  ASSERT_EQ(footX(*figure), 0);
  const fv::WearResult worn = figure->commitWear(figure->prepareWear(kFootHeels, 1), kFootHeels);
  figure->settle();
  ASSERT_EQ(worn.items.size(), 1u);
  EXPECT_DOUBLE_EQ(footX(*figure), 30);

  ASSERT_TRUE(figure->removeItem(worn.items[0]));
  EXPECT_EQ(footX(*figure), 0);
}

// An outfit's items all carry its file's pose, set once and kept by the item on the feet: the shirt
// listed first comes off and the feet stay pointed, the heel comes off and they go back.
TEST(Wardrobe, AnOutfitsFeetGoBackWithItsShoeNotItsShirt)
{
  const auto           figure = footFigure();
  const fv::WearResult worn   = figure->commitWear(figure->prepareWear(kFootOutfit, 1), kFootOutfit);
  figure->settle();
  ASSERT_EQ(worn.items.size(), 2u);
  EXPECT_DOUBLE_EQ(footX(*figure), 30);
  int shirt = 0, heel = 0;
  for(int id : worn.items)
    (figure->item(id)->label == "Foot Heel" ? heel : shirt) = id;
  ASSERT_TRUE(shirt && heel);
  EXPECT_TRUE(figure->item(shirt)->footSet.empty());

  ASSERT_TRUE(figure->removeItem(shirt));
  EXPECT_DOUBLE_EQ(footX(*figure), 30);
  ASSERT_TRUE(figure->removeItem(heel));
  EXPECT_EQ(footX(*figure), 0);
}

// A Genesis 9 heel may set only the figure's own foot dial, body_ctrl_lFootHeeledShoe, whose formula
// points the foot: the dial is set and the foot turns, a dial the figure lacks is passed over, and
// taken off, the dial goes back.
TEST(Wardrobe, AHeelSettingTheFiguresFootDialPointsTheFoot)
{
  const auto figure = footFigure();
  ASSERT_NE(figure->morphs->channel("body_ctrl_lFootHeeledShoe"), nullptr);
  ASSERT_NEAR(footTurn(*figure), 0, 1e-6);
  const fv::WearResult worn = figure->commitWear(figure->prepareWear(kFootBoots, 1), kFootBoots);
  figure->settle();
  ASSERT_EQ(worn.items.size(), 1u);
  EXPECT_DOUBLE_EQ(figure->morphs->get("body_ctrl_lFootHeeledShoe"), 0.8);
  EXPECT_EQ(footX(*figure), 0);
  EXPECT_NEAR(footTurn(*figure), 0.8 * 55, 1e-3);

  ASSERT_TRUE(figure->removeItem(worn.items[0]));
  figure->settle();
  EXPECT_EQ(figure->morphs->get("body_ctrl_lFootHeeledShoe"), 0);
  EXPECT_NEAR(footTurn(*figure), 0, 1e-6);
}

TEST(Fit, AHeelsFootChannelsAreItsBonesTurnsAndTheFiguresFootDials)
{
  const auto doc = fv::dson::Json::parse(R"({"scene": {"animations": [
    {"url": "name://@selection#body_ctrl_lFootHeeledShoe:?value/value", "keys": [[0, 0.85]]},
    {"url": "name://@selection#body_ctrl_lToesSpread:?value/value", "keys": [[0, -0.5]]},
    {"url": "name://@selection#body_ctrl_lMouthOpen:?value/value", "keys": [[0, 1]]},
    {"url": "name://@selection/l_foot:?scale/x/value", "keys": [[0, 1]]},
    {"url": "name://@selection/l_foot:?rotation/x/value", "keys": [[0, 39.3]]},
    {"url": "name://@selection/l_toes#body_ctrl_lToesSpread:?value/value", "keys": [[0, 1]]}]}})");
  const std::vector<fv::wardrobe::FootChannel> feet = fv::wardrobe::footChannels(doc);
  ASSERT_EQ(feet.size(), 3u);
  EXPECT_EQ(feet[0].dial, "body_ctrl_lFootHeeledShoe");
  EXPECT_DOUBLE_EQ(feet[0].value, 0.85);
  EXPECT_EQ(feet[1].dial, "body_ctrl_lToesSpread");
  EXPECT_EQ(feet[2].bone, "l_foot");
  EXPECT_EQ(feet[2].axis, "x");
  EXPECT_TRUE(feet[2].dial.empty());

  // a dial a hundredth from rest is a pose; all of a file's channels at rest are none
  EXPECT_EQ(fv::wardrobe::retargetFootPose({{"", "", 0.02, "body_ctrl_lFootHeeledShoe"}}, {"l_foot"}).size(), 1u);
  EXPECT_TRUE(fv::wardrobe::retargetFootPose({{"", "", 0.005, "body_ctrl_lFootHeeledShoe"}, {"l_foot", "x", 0.2}}, {"l_foot"}).empty());
}

// A wear reads the skeleton and the body of a figure, on another thread: it isn't given the rest, which
// grows with what is worn. The faces its stencils were worked out from are gone by then too.
TEST(Wardrobe, AWearIsGivenTheSkeletonAndTheBodyOnly)
{
  const auto figure = boxFigure();
  const auto worn   = figure->commitWear(figure->prepareWear(kBoxShirt, 1), kBoxShirt);
  figure->settle();
  const int  shirt = figure->item(worn.items.at(0))->meshes.at(0);
  const auto base  = figure->wearBase();
  EXPECT_EQ(base->nodes.size(), figure->loaded.scene.nodes.size());
  EXPECT_EQ(base->meshes.size(), figure->loaded.scene.meshes.size());
  EXPECT_FALSE(base->meshes[size_t(shirt)].geo);
  EXPECT_TRUE(base->meshes[size_t(shirt)].skin.skinIndex.empty());
  const size_t body = size_t(figure->loaded.morphs.meshes[figure->loaded.morphs.mainMeshes[0]].mesh);
  EXPECT_EQ(base->meshes[body].geo, figure->loaded.scene.meshes[body].geo);
  const auto& refinement = figure->loaded.scene.meshes[body].geo->refinement;
  ASSERT_TRUE(refinement);
  EXPECT_TRUE(refinement->quads.empty());
  EXPECT_TRUE(refinement->quadParent.empty());
  EXPECT_TRUE(refinement->quadValues.empty());
  EXPECT_FALSE(refinement->levels.empty());
}

// System memory running out anywhere in a commit leaves the figure as it was: nothing half worn, nothing
// of it drawn, and the same wear goes on afterwards.
TEST(Wardrobe, AWearThatRunsOutOfMemoryLeavesTheFigureAsItWas)
{
  const auto figure = boxFigure();
  const size_t had  = figure->items.size();
  long         total = 0;
  {
    auto       prepared = figure->prepareWear(kBoxShirt, 1);
    const long before   = fv::test::allocationsMade();
    const auto worn     = figure->commitWear(prepared, kBoxShirt);
    total               = fv::test::allocationsMade() - before;
    figure->removeItem(worn.items.at(0));
  }
  ASSERT_GT(total, 0);
  size_t failed = 0;
  for(long n = 0; n < total; n += 1 + total / 300)
  {
    auto prepared = figure->prepareWear(kBoxShirt, 1);
    fv::test::failAllocationAfter(n);
    try
    {
      const auto worn = figure->commitWear(prepared, kBoxShirt);
      fv::test::failAllocationAfter(-1);
      // what something on the way caught: it is on whole
      EXPECT_EQ(figure->items.size(), had + 1) << n;
      figure->removeItem(worn.items.at(0));
    }
    catch(const std::bad_alloc&)
    {
      fv::test::failAllocationAfter(-1);
      failed++;
      EXPECT_EQ(figure->items.size(), had) << n;
      figure->settle();
      (void)figure->pump();
      (void)figure->drain();
      const std::vector<bool> live = fv::liveMeshes(figure->loaded.scene);
      for(size_t m = 0; m < live.size(); m++)
        if(!figure->loaded.scene.meshes[m].geo)
          EXPECT_FALSE(live[m]) << n << ": mesh " << m << " is drawn with nothing to draw";
    }
    prepared.reset();
  }
  EXPECT_GT(failed, 0u);
  const auto worn = figure->commitWear(figure->prepareWear(kBoxShirt, 1), kBoxShirt);
  figure->settle();
  EXPECT_EQ(figure->items.size(), had + 1);
  EXPECT_EQ(worn.items.size(), 1u);
  (void)fv::build::buildGltf(figure->loaded.scene, figure->loaded.root, {});
}

// The same through a real wear. Box Figure's Sleeve Lift has an empty morph block, and Box Shirt holds
// the deltas in a file of the same name.
TEST(Wardrobe, AFiguresEmptyDialMovesTheWornItemThatHoldsItsDeltas)
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions       options;
  options.subdivision = 1;
  auto figure = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library})),
                                 "People/Test/Box Figure.duf", options);
  figure->settle();
  const std::string shirt = "People/Test/Clothing/Box Shirt.duf";
  const auto        wear  = [&] {
    const fv::WearResult worn = figure->commitWear(figure->prepareWear(shirt, 1), shirt);
    figure->settle();
    return worn.items.at(0);
  };
  const auto geoOf = [&](int id) {
    for(const fv::WornItem& item : figure->items)
      if(item.id == id)
        return figure->loaded.scene.meshes[size_t(item.meshes.at(0))].geo;
    return std::shared_ptr<fv::mesh::BuiltMesh>();
  };
  const auto moved = [](const fv::mesh::BuiltMesh& geo, const std::vector<float>& from) {
    float most = 0;
    for(size_t i = 0; i < from.size(); i++)
      most = std::max(most, std::abs(geo.position[i] - from[i]));
    return most;
  };
  // the shirt's copy loads on the pool: a frame or so, as the app sees it
  const auto waitForMove = [&](const fv::mesh::BuiltMesh& geo, const std::vector<float>& from) {
    for(int i = 0; i < 5000 && moved(geo, from) < 1e-3f; i++)
    {
      figure->pump();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return moved(geo, from);
  };

  const int  first = wear();
  const auto geo   = geoOf(first);
  ASSERT_TRUE(geo);
  const std::vector<std::string> onlyShirt{"Box Shirt"};
  EXPECT_EQ(figure->morphs->itemsMoving("Sleeve Lift"), onlyShirt);
  EXPECT_TRUE(figure->morphs->itemsMoving("Taper").empty());
  const std::vector<float> rest = geo->position;

  figure->morphs->set("Sleeve Lift", 1.0);
  figure->settle();
  const fv::morph::Channel* dial = figure->morphs->channel("Sleeve Lift");
  ASSERT_NE(dial, nullptr);
  EXPECT_TRUE(dial->inert);
  EXPECT_NEAR(waitForMove(*geo, rest), 3.0f, 1e-3f);
  figure->morphs->set("Sleeve Lift", 0.0);
  figure->settle();
  EXPECT_LT(moved(*geo, rest), 1e-4f);

  // off with the shirt the dial stays, set; on again the new shirt takes it
  figure->morphs->set("Sleeve Lift", 0.5);
  figure->settle();
  ASSERT_TRUE(figure->removeItem(first));
  ASSERT_NE(figure->morphs->channel("Sleeve Lift"), nullptr);
  EXPECT_TRUE(figure->morphs->itemsMoving("Sleeve Lift").empty());
  const auto again = geoOf(wear());
  ASSERT_TRUE(again);
  EXPECT_EQ(figure->morphs->get("Sleeve Lift"), 0.5);
  EXPECT_NEAR(waitForMove(*again, rest), 1.5f, 1e-3f);
}

// A hair loaded with a share of its strands covers what it did: each is wider by as much. A mesh a
// prefetch built of all of them isn't the one this load wants, and waits for one that does.
TEST(Wardrobe, AHairLoadedWithAShareOfItsStrandsIsAsWideAsItHasToBe)
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  auto store = std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library}));
  const std::string hair = "People/Test/Clothing/Box Fan Hair.duf";
  fv::dson::prefetchScene(store, hair, 0).wait();
  const auto ribbon = [](const fv::dson::LoadResult& r) -> const fv::mesh::BuiltMesh& {
    for(const fv::Mesh& m : r.scene.meshes)
      if(m.geo && m.geo->ribbon)
        return *m.geo;
    throw std::runtime_error("no strands");
  };
  const auto widest = [](const fv::mesh::BuiltMesh& g) {
    float most = 0;
    for(size_t i = 0; i + 2 < g.ribbon->offsets.size(); i += 3)
      most = std::max(most, std::hypot(g.ribbon->offsets[i], g.ribbon->offsets[i + 1], g.ribbon->offsets[i + 2]));
    return most;
  };
  fv::dson::LoadOptions less;
  less.strandKeep                      = 0.5f;
  const fv::dson::LoadResult thinned   = fv::dson::loadDson(*store, hair, less);
  const fv::dson::LoadResult whole     = fv::dson::loadDson(*store, hair);
  const fv::mesh::BuiltMesh& drawn     = ribbon(thinned);
  const fv::mesh::BuiltMesh& allOfIt   = ribbon(whole);
  EXPECT_EQ(drawn.fullIndex.size() * 2, allOfIt.fullIndex.size());  // one of its two strands
  EXPECT_EQ(drawn.dsonVertexCount, allOfIt.dsonVertexCount);
  EXPECT_NEAR(widest(drawn), widest(allOfIt) * 2, 1e-5f);
}

// A hair made again over its cage: the same cage, a share of its strands, each as much wider, and all of it
// again from that as it was.
TEST(Wardrobe, AHairMadeAgainIsAsWideAsItsShareHasToBe)
{
  fv::mesh::MeshInput in;
  in.materialCount = 1;
  fv::mesh::Lines lines;
  lines.start.push_back(0);
  for(int line = 0; line < 32; line++)
  {
    for(int point = 0; point < 4; point++)
    {
      lines.verts.push_back(int32_t(in.positions.size() / 3));
      for(float v : {float(line), float(point), 0.f})
        in.positions.push_back(v);
    }
    lines.start.push_back(int32_t(lines.verts.size()));
    lines.material.push_back(0);
  }
  in.lines = lines;
  fv::mesh::BuiltMesh whole = fv::mesh::build(in);
  fv::mesh::widenRibbons(whole, {{0.1, 0.05}});
  const auto widest = [](const fv::mesh::BuiltMesh& g) {
    float most = 0;
    for(size_t i = 0; i + 2 < g.ribbon->offsets.size(); i += 3)
      most = std::max(most, std::hypot(g.ribbon->offsets[i], g.ribbon->offsets[i + 1], g.ribbon->offsets[i + 2]));
    return most;
  };
  ASSERT_GT(widest(whole), 0.f);

  const auto half = fv::mesh::remade(whole, whole.cagePositions, 0, 0.5f);
  ASSERT_NE(half, nullptr);
  EXPECT_NEAR(double(half->fullIndex.size()), double(whole.fullIndex.size()) / 2, double(whole.fullIndex.size()) / 8);
  EXPECT_EQ(half->dsonVertexCount, whole.dsonVertexCount);
  EXPECT_EQ(half->cageLineVerts, whole.cageLineVerts);
  EXPECT_EQ(half->source->lines->keep, 0.5f);
  EXPECT_NEAR(widest(*half), widest(whole) * 2, 1e-5f);
  EXPECT_TRUE(fv::mesh::drawnFromCage(*half));

  const auto again = fv::mesh::remade(*half, half->cagePositions, 0, 1.f);
  EXPECT_EQ(again->fullIndex, whole.fullIndex);
  EXPECT_EQ(again->position, whole.position);
  EXPECT_EQ(again->ribbon->offsets, whole.ribbon->offsets);

  // over the cage it has now, as one built there would be
  std::vector<float> moved = whole.cagePositions;
  for(size_t i = 1; i < moved.size(); i += 3)
    moved[i] += 2;
  const auto           there = fv::mesh::remade(whole, moved, 0, 1.f);
  fv::mesh::MeshInput  atMoved = in;
  atMoved.positions           = moved;
  fv::mesh::BuiltMesh  fresh   = fv::mesh::build(atMoved);
  fv::mesh::widenRibbons(fresh, {{0.1, 0.05}});
  EXPECT_EQ(there->position, fresh.position);

  // what has nothing to be finer or coarser in isn't made again
  fv::mesh::MeshInput flat;
  flat.positions = {0, 0, 0, 1, 0, 0, 1, 1, 0};
  flat.polyStart = {0, 3};
  flat.polyVerts = {0, 1, 2};
  flat.polyMaterial = {0};
  const fv::mesh::BuiltMesh plain = fv::mesh::build(flat);
  EXPECT_EQ(plain.source, nullptr);
  EXPECT_EQ(fv::mesh::remade(plain, plain.cagePositions, 1, 1.f), nullptr);
}

// One step up undoes one step down, as far as there is to go; a mesh says what share a quality would draw.
TEST(Wardrobe, AnItemsQualityGoesUpAsItCameDown)
{
  using fv::WearQuality;
  const auto at = [](const WearQuality& q, int subdivision, float keep, int pieces) {
    return q.subdivision == subdivision && q.strandKeep == keep && q.strandPieces == pieces;
  };
  EXPECT_TRUE(at(fv::oneStepUp({0, 0.25f, 1}, 2), 1, 0.5f, 1));
  EXPECT_TRUE(at(fv::oneStepUp({1, 0.5f, 1}, 2), 2, 1.f, 1)) << "all the strands come back before their second pieces";
  EXPECT_TRUE(at(fv::oneStepUp({2, 1.f, 1}, 2), 2, 1.f, 2));
  EXPECT_TRUE(at(fv::oneStepUp({2, 1.f, 2}, 2), 2, 1.f, 2));
  EXPECT_TRUE(at(fv::oneStepUp({2, 1.f, 2}, 1), 2, 1.f, 2)) << "a setting lowered since asks nothing of what is finer";

  fv::Mesh smooth, strands, plain;
  smooth.geo                     = std::make_shared<fv::mesh::BuiltMesh>();
  smooth.geo->subdivisionSurface = true;
  auto smoothInput               = std::make_shared<fv::mesh::MeshInput>();
  smoothInput->subdivisionSurface = true;
  smoothInput->level             = 2;
  smooth.geo->source             = smoothInput;
  strands.geo                    = std::make_shared<fv::mesh::BuiltMesh>();
  auto strandInput               = std::make_shared<fv::mesh::MeshInput>();
  strandInput->lines.emplace();
  strandInput->lines->keep = 0.5f;
  strands.geo->source      = strandInput;
  plain.geo                = std::make_shared<fv::mesh::BuiltMesh>();
  EXPECT_FLOAT_EQ(fv::shareAt(smooth, {1, 1.f}), 0.25f);
  EXPECT_FLOAT_EQ(fv::shareAt(smooth, {2, 0.25f}), 1.f);
  EXPECT_FLOAT_EQ(fv::shareAt(strands, {0, 0.25f}), 0.5f);
  EXPECT_FLOAT_EQ(fv::shareAt(strands, {0, 1.f}), 2.f);
  EXPECT_FLOAT_EQ(fv::shareAt(strands, {0, 0.5f, 1}), 0.5f) << "one piece a segment is half the triangles";
  EXPECT_FLOAT_EQ(fv::shareAt(plain, {0, 0.25f}), 1.f);

  fv::SceneGraph scene;
  scene.meshes = {smooth, strands, plain};
  EXPECT_TRUE(at(fv::qualityOf(scene, {0}), 2, 1.f, 2));
  EXPECT_TRUE(at(fv::qualityOf(scene, {1}), 0, 0.5f, 2));
  EXPECT_TRUE(at(fv::qualityOf(scene, {2}), 0, 1.f, 2));
  EXPECT_TRUE(at(fv::qualityOf(scene, {0, 1, 2}), 2, 0.5f, 2));
  strandInput->lines->pieces = 1;
  scene.meshes[1].geo->source = strandInput;
  EXPECT_TRUE(at(fv::qualityOf(scene, {1}), 0, 0.5f, 1));
}

// A wearable goes down a step at a time, as far as each of its meshes has anywhere to go, and says what
// it came in without only where it has that to lose.
TEST(Wardrobe, AWearThatDoesNotFitGoesDownAStepAtATime)
{
  using fv::WearQuality;
  const auto at = [](const WearQuality& q, int subdivision, float keep, int pieces) {
    return q.subdivision == subdivision && q.strandKeep == keep && q.strandPieces == pieces;
  };
  // a hair's segments go to one piece before any strand goes
  WearQuality q{3, 1.f, 2};
  q = fv::oneStepDown(q);
  EXPECT_TRUE(at(q, 2, 1.f, 1));
  q = fv::oneStepDown(q);
  EXPECT_TRUE(at(q, 1, 0.5f, 1));
  q = fv::oneStepDown(q);
  EXPECT_TRUE(at(q, 0, 0.25f, 1));
  q = fv::oneStepDown(q);
  EXPECT_TRUE(at(q, 0, 0.25f, 1));

  const auto scene = [](bool ribbon, bool subd) {
    fv::SceneGraph s;
    fv::Mesh       m;
    m.geo                     = std::make_shared<fv::mesh::BuiltMesh>();
    m.geo->subdivisionSurface = subd;
    if(ribbon)
      m.geo->ribbon.emplace();
    s.meshes.push_back(m);
    return s;
  };
  const fv::SceneGraph hair = scene(true, false), garment = scene(false, true), prop = scene(false, false);
  EXPECT_TRUE(fv::wearCanShrink(hair, {0, 1.f}));
  EXPECT_TRUE(fv::wearCanShrink(hair, {0, 1.f, 1}));
  EXPECT_TRUE(fv::wearCanShrink(hair, {0, 0.5f, 1}));
  EXPECT_TRUE(fv::wearCanShrink(hair, {2, 0.25f, 2})) << "its second pieces are still there to give";
  EXPECT_FALSE(fv::wearCanShrink(hair, {2, 0.25f, 1}));
  EXPECT_TRUE(fv::wearCanShrink(garment, {1, 0.25f}));
  EXPECT_FALSE(fv::wearCanShrink(garment, {0, 1.f}));
  EXPECT_FALSE(fv::wearCanShrink(prop, {2, 1.f}));

  const std::vector<int> only{0};
  EXPECT_EQ(fv::wearShortfall(hair, only, {1, 0.5f}, 2), "half its strands");
  EXPECT_EQ(fv::wearShortfall(hair, only, {1, 0.5f, 1}, 2), "half its strands");
  EXPECT_EQ(fv::wearShortfall(hair, only, {0, 0.25f}, 2), "a quarter of its strands");
  EXPECT_EQ(fv::wearShortfall(hair, only, {0, 1.f, 1}, 2), "its strands in half the triangles");
  EXPECT_EQ(fv::wearShortfall(hair, only, {0, 1.f, 2}, 2), "");
  EXPECT_EQ(fv::wearShortfall(garment, only, {1, 0.5f}, 2), "Subdivision 1");
  EXPECT_EQ(fv::wearShortfall(prop, only, {1, 0.5f}, 2), "");
  EXPECT_EQ(fv::wearShortfall(garment, only, {2, 1.f}, 2), "");
}

// The store stays with the figure, and a morph a dial's formula asked for and isn't installed is no
// business of the next item's: Dejla 9's head drives a morph of a pack that isn't there, and whatever
// went on after it said it needed that file.
TEST(Wardrobe, AnItemIsNotBlamedForAFileTheFigureAskedFor)
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions       options;
  options.subdivision = 1;
  auto figure = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library})),
                                 "People/Test/Box Figure.duf", options);
  figure->settle();
  EXPECT_THROW(figure->store().load("data/Test/Not Installed.dsf"), fv::dson::MissingFileError);
  const std::string    shirt = "People/Test/Clothing/Box Shirt.duf";
  const fv::WearResult worn  = figure->commitWear(figure->prepareWear(shirt, 1), shirt);
  EXPECT_TRUE(worn.warnings.empty()) << worn.warnings.at(0);
}

// A character's own file setting a morph that isn't installed does come out wrong, and says so.
TEST(FigureMorphs, ACharacterSettingAMorphThatIsNotInstalledSaysSo)
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  const std::filesystem::path extra   = std::filesystem::temp_directory_path() / "fv_test_lost_dial";
  std::filesystem::remove_all(extra);
  std::filesystem::create_directories(extra / "People" / "Test");
  std::ifstream  in(library / "People" / "Test" / "Box Figure.duf");
  fv::dson::Json doc = fv::dson::Json::parse(in);
  fv::dson::Json dial;
  dial["id"]                       = "Gone";
  dial["url"]                      = "/data/Test/Box%20Figure/Morphs/Test/Base/Gone.dsf#Gone";
  dial["parent"]                   = "#BoxFigure";
  dial["channel"]["id"]            = "value";
  dial["channel"]["current_value"] = 1.0;
  doc["scene"]["modifiers"].push_back(dial);
  std::ofstream(extra / "People" / "Test" / "Lost Dial.duf") << doc.dump();

  fv::dson::LoadOptions options;
  options.subdivision = 1;
  auto figure = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library, extra})),
                                 "People/Test/Lost Dial.duf", options);
  figure->settle();
  const std::vector<std::string> warnings = figure->takeWarnings();
  std::filesystem::remove_all(extra);
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0],
            "Box Figure needs 1 library file that isn't installed in the assets folder or the DAZ library: data/Test/Box Figure/Morphs/Test/Base/Gone.dsf.");
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
  const auto         got = fv::wardrobe::projectTarget(binding, nullptr, body, item.dsonVertexCount, bodyTarget, scratchNow);
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
      return fv::wardrobe::projectTarget(*binding, nullptr, *body, item->dsonVertexCount, t, scratch);
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
        return fv::wardrobe::projectTarget(*binding, nullptr, *body, item->dsonVertexCount, t, scratch);
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

  rs::RetainedScene loaded;
  rs::importGltf(loaded, built.model);
  EXPECT_EQ(loaded.drawItems().size(), 1u);
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

// Julia Hair has 15 strand bones to its 5 of the figure's. That isn't another figure's skeleton: it is
// made for this rig. An item whose generation can't be told still has its bones to go by.
TEST(Fit, AnItemForTheFiguresOwnRigIsMadeForIt)
{
  fv::SceneGraph                scene;
  fv::dson::Rig                 rig;
  fv::dson::LoadResult          wearable;
  fv::dson::Item                item;
  const fv::wardrobe::FitFigure g9{scene, rig, 0, "data/Daz 3D/Genesis 9/Base"};
  const fv::wardrobe::FitFigure g81{scene, rig, 0, "data/DAZ 3D/Genesis 8/Female 8_1"};
  EXPECT_TRUE(fv::wardrobe::madeForFigure(wearable, item, "People/Genesis 9/Hair/Toyen/Julia Hair/Julia Hair.duf", g9));
  EXPECT_FALSE(fv::wardrobe::madeForFigure(wearable, item, "People/Genesis 8 Female/Hair/Some Hair.duf", g9));
  EXPECT_FALSE(fv::wardrobe::madeForFigure(wearable, item, "People/Genesis/Hair/Santa Baby/Hair.duf", g9));
  EXPECT_TRUE(fv::wardrobe::madeForFigure(wearable, item, "People/Genesis 8 Female/Hair/Some Hair.duf", g81));
  item.conformTarget = "/data/Daz%203D/Genesis%209/Base/Genesis9.dsf";
  EXPECT_TRUE(fv::wardrobe::madeForFigure(wearable, item, "Props/Some Hat.duf", g9));
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

// A cornea is under the film only where the moisture lies over it. Genesis 8's stops at the limbus,
// around a hole the cornea fills: there the cornea reflects, or the eye has no catchlight over the iris.
TEST(Wardrobe, ACorneaInTheMoisturesHoleIsTheFilm)
{
  using fv::mat::Kind;
  const auto lowerWithMoistureAt = [](float moistureX) {
    fv::SceneGraph scene;
    const int      root = scene.add(fv::Node{});
    fv::Material   cornea, moisture;
    cornea.surface.kind         = Kind::eCornea;
    cornea.surface.name         = "Cornea";
    cornea.surface.refraction   = fv::mat::Surface::Refraction{1.f, 1.376f};
    moisture.surface.kind       = Kind::eCornea;
    moisture.surface.name       = "EyeMoisture";
    moisture.surface.refraction = fv::mat::Surface::Refraction{1.f, 1.376f};
    scene.materials             = {cornea, moisture};
    for(int mat : {0, 1})
    {
      fv::Mesh m;
      fv::mesh::MeshInput in = grid(2, mat == 0 ? 0.f : 0.05f);
      if(mat == 1)
        for(size_t v = 0; v * 3 < in.positions.size(); v++)
          in.positions[v * 3] += moistureX;
      m.geo       = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(in));
      m.materials = {mat};
      scene.meshes.push_back(m);
      fv::Node node;
      node.kind = fv::NodeKind::eMesh;
      node.mesh = int(scene.meshes.size()) - 1;
      scene.add(node, root);
    }
    return fv::build::requestOptions(scene, 0, fv::mat::SkinMode::eMeasured, true, "").underTearFilm;
  };
  EXPECT_TRUE(lowerWithMoistureAt(0));      // half a millimetre over it
  EXPECT_FALSE(lowerWithMoistureAt(5.f));   // beside it, the shell ringing a hole
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
