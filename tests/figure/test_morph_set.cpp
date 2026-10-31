// Incremental morph flush matches a rebuild, a local morph reports only the vertices it moved, and
// lazy files that arrive together run the driver once. Ports test/morphs/incremental.test.ts (the
// position half; this MorphSet re-welds normals instead of applying normal deltas) and
// test/morphs/driverCoalesce.test.ts.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/mesh/normals.hpp>
#include <figure/morph/morph_set.hpp>

namespace {

fv::mesh::BuiltMesh grid(int n, int level)
{
  fv::mesh::MeshInput in;
  in.level         = level;
  in.materialCount = 1;
  for(int j = 0; j <= n; j++)
    for(int i = 0; i <= n; i++)
    {
      in.positions.push_back(float(i));
      in.positions.push_back(float(j));
      in.positions.push_back(float(std::sin(double(i)) * std::cos(double(j))));
    }
  in.polyStart.push_back(0);
  for(int j = 0; j < n; j++)
    for(int i = 0; i < n; i++)
    {
      const int a = j * (n + 1) + i;
      in.polyVerts.push_back(a);
      in.polyVerts.push_back(a + 1);
      in.polyVerts.push_back(a + 1 + (n + 1));
      in.polyVerts.push_back(a + (n + 1));
      in.polyMaterial.push_back(0);
      in.polyStart.push_back(int32_t(in.polyVerts.size()));
    }
  return fv::mesh::build(in);
}

fv::dson::TargetPtr targetOf(const fv::mesh::BuiltMesh& geo, std::vector<int32_t> verts, std::vector<float> deltas)
{
  fv::mesh::CageDeltas d;
  d.verts  = std::move(verts);
  d.deltas = std::move(deltas);
  return std::make_shared<const fv::mesh::RenderTarget>(fv::mesh::target(geo, d));
}

double worst(const std::vector<float>& got, const std::vector<float>& want)
{
  double w = 0;
  for(size_t i = 0; i < want.size() && i < got.size(); i++)
    w = std::max(w, std::abs(double(got[i]) - double(want[i])));
  return w;
}

struct CountDriver : fv::morph::Driver
{
  int calls = 0;
  void update() override { calls++; }
};

// A lazy channel whose file does not arrive until `gate` is set. `arrived` counts files that have.
fv::morph::Loader held(std::shared_future<void> gate, std::atomic<int>& arrived)
{
  return [gate, &arrived]() -> fv::morph::Finish {
    gate.wait();
    arrived.fetch_add(1);
    return [] {
      fv::morph::LazyResult r;
      r.keep = true;
      return r;
    };
  };
}

}  // namespace

class FigureMorphSet : public testing::TestWithParam<int>
{
};

TEST_P(FigureMorphSet, IncrementalFlushMatchesARebuild)
{
  const int level = GetParam();
  const int n     = 8;
  auto      geo   = std::make_shared<fv::mesh::BuiltMesh>(grid(n, level));
  const int cage  = (n + 1) * (n + 1);

  std::vector<float> localD(12);
  for(int i = 0; i < 12; i++)
    localD[i] = i % 3 == 2 ? 0.5f : 0.1f;
  std::vector<int32_t> halfV(cage / 2);
  std::vector<float>   halfD(size_t(cage / 2) * 3);
  for(int i = 0; i < cage / 2; i++)
    halfV[i] = i;
  for(size_t i = 0; i < halfD.size(); i++)
    halfD[i] = float(std::sin(double(i)) * 0.2);
  std::vector<int32_t> allV(cage);
  std::vector<float>   allD(size_t(cage) * 3);
  for(int i = 0; i < cage; i++)
    allV[i] = i;
  for(size_t i = 0; i < allD.size(); i++)
    allD[i] = float(std::cos(double(i)) * 0.1);

  const fv::dson::TargetPtr targets[] = {targetOf(*geo, {40, 41, 49, 50}, localD), targetOf(*geo, halfV, halfD), targetOf(*geo, allV, allD)};
  fv::morph::MorphSet       morphs([&](int) { return geo.get(); });
  for(int i = 0; i < 3; i++)
    morphs.addTarget("m" + std::to_string(i), 0, targets[i]);
  morphs.takeChanges();

  double weights[3] = {};
  int    seed       = 7;
  auto   rand       = [&] {
    seed = int(seed * 16807ll % 2147483647);
    return seed / 2147483647.0;
  };
  for(int step = 0; step < 120; step++)
  {
    const int k = int(rand() * 3);
    weights[k]  = rand() < 0.15 ? 0 : rand() * 2 - 0.5;
    morphs.set("m" + std::to_string(k), weights[k]);
  }
  for(int step = 0; step < 10; step++)
  {
    weights[0] = rand() * 1.5;
    morphs.set("m0", weights[0]);
  }

  // the positions a single application of the final weights would write, on an untouched copy of the grid
  auto               fresh = grid(n, level);
  std::vector<float> base  = fresh.position;
  for(int i = 0; i < 3; i++)
  {
    const fv::mesh::RenderTarget::Drawn& d = targets[i]->drawn(fresh);
    for(size_t j = 0; j < d.indices.size(); j++)
      for(int c = 0; c < 3; c++)
        base[size_t(d.indices[j]) * 3 + c] += float(d.deltas[j * 3 + c] * weights[i]);
  }
  EXPECT_LT(worst(geo->position, base), 1e-4);

  fv::mesh::NormalWelder welder(geo->renderIndex, geo->firstSplit, geo->nextSplit, geo->renderTris, geo->renderVertexCount);
  std::vector<float>     full(geo->normal.size());
  welder.full(geo->position, full);
  EXPECT_LT(worst(geo->normal, full), 1e-4);
}

INSTANTIATE_TEST_SUITE_P(Subdivision, FigureMorphSet, testing::Values(0, 1));

// The host republishes the morph topic when version() moved: a dial's value, or a weight a driver
// applied (a corrective after a pose). The same value again, or an unknown dial, is not a change.
TEST(FigureMorphSet, VersionMovesOnlyWhenAValueOrAWeightChanges)
{
  auto                geo = std::make_shared<fv::mesh::BuiltMesh>(grid(4, 0));
  fv::morph::MorphSet morphs([&](int) { return geo.get(); });
  morphs.addTarget("m0", 0, targetOf(*geo, {0, 1}, {0.1f, 0, 0, 0.1f, 0, 0}));
  const int v0 = morphs.version();
  morphs.set("m0", 0.5);
  EXPECT_GT(morphs.version(), v0);
  const int v1 = morphs.version();
  morphs.set("m0", 0.5);
  morphs.set("not a dial", 1.0);
  EXPECT_EQ(morphs.version(), v1);
  morphs.applyWeights({{"m0", 0.5}});  // the dial's own value: the weight did not change
  EXPECT_EQ(morphs.version(), v1);
  morphs.applyWeights({{"m0", 0.25}});  // a driver's weight
  EXPECT_GT(morphs.version(), v1);
  const int v2 = morphs.version();
  morphs.applyWeights({{"m0", 0.25}});
  EXPECT_EQ(morphs.version(), v2);
}

TEST(FigureMorphSet, ALocalMorphReportsOnlyWhatItMoved)
{
  auto                  geo = std::make_shared<fv::mesh::BuiltMesh>(grid(20, 0));
  const auto            t   = targetOf(*geo, {200}, {0, 0, 1});
  fv::morph::MorphSet   morphs([&](int) { return geo.get(); });
  morphs.addTarget("bump", 0, t);
  morphs.takeChanges();
  morphs.set("bump", 0.5);
  const auto changes = morphs.takeChanges();
  ASSERT_EQ(changes.size(), 1u);
  EXPECT_FALSE(changes[0].all);
  EXPECT_NE(std::find(changes[0].indices.begin(), changes[0].indices.end(), 200u), changes[0].indices.end());
  // the moved vertex, plus the neighbours whose welded normal it changed: not the whole grid
  EXPECT_LT(changes[0].indices.size(), 30u);
  EXPECT_EQ(std::find(changes[0].indices.begin(), changes[0].indices.end(), 0u), changes[0].indices.end());
  EXPECT_NEAR(geo->position[200 * 3 + 2], grid(20, 0).position[200 * 3 + 2] + 0.5f, 1e-5);
}

// The parallel full weld writes exactly what the serial weld writes: the same triangle normals
// summed per render vertex in the same order.
TEST(FigureMorphSet, ParallelFullWeldMatchesTheSerialWeld)
{
  auto geo = grid(80, 1);  // 51 200 triangles, 25 921 render vertices: several chunks of each
  fv::mesh::NormalWelder welder(geo.renderIndex, geo.firstSplit, geo.nextSplit, geo.renderTris, geo.renderVertexCount);
  std::vector<float>     parallel(geo.normal.size(), 0.f), serial(geo.normal.size(), 0.f);
  welder.full(geo.position, parallel);
  fv::mesh::weldNormals(geo.position, geo.renderIndex, geo.firstSplit, geo.renderTris, geo.renderVertexCount, serial);
  size_t different = 0;
  for(size_t i = 0; i < serial.size(); i++)
    different += parallel[i] != serial[i];
  EXPECT_EQ(different, 0u);
  // and twice over, the second time with the adjacency and triangle buffer already built
  welder.full(geo.position, parallel);
  for(size_t i = 0; i < serial.size(); i++)
    different += parallel[i] != serial[i];
  EXPECT_EQ(different, 0u);
}

// A target over more vertices than one parallel chunk moves every vertex it lists, once, and
// reports them in index order: the moved list drives the normal weld around them.
TEST(FigureMorphSet, ALargeTargetAddsItsDeltasInChunks)
{
  auto geo = std::make_shared<fv::mesh::BuiltMesh>(grid(200, 1));  // 160 801 render vertices
  // a band of 30 cage rows: about 24 000 drawn vertices, over one chunk and under a quarter of the mesh
  std::vector<int32_t> verts;
  std::vector<float>   deltas;
  for(int j = 60; j < 90; j++)
    for(int i = 0; i <= 200; i++)
    {
      verts.push_back(j * 201 + i);
      deltas.insert(deltas.end(), {0.f, 0.f, 0.01f * float(i % 7 + 1)});
    }
  const auto          t = targetOf(*geo, verts, deltas);
  ASSERT_GT(t->drawn(*geo).indices.size(), 16384u);
  ASSERT_LT(t->drawn(*geo).indices.size(), geo->position.size() / 3 / 4);
  const std::vector<float> base = geo->position;
  fv::morph::MorphSet      morphs([&](int) { return geo.get(); });
  morphs.addTarget("band", 0, t);
  morphs.takeChanges();
  morphs.set("band", 0.5);
  const auto changes = morphs.takeChanges();
  ASSERT_EQ(changes.size(), 1u);
  EXPECT_FALSE(changes[0].all);
  EXPECT_TRUE(std::is_sorted(changes[0].indices.begin(), changes[0].indices.end()));
  // every listed vertex moved by half its delta, and is among the changed
  std::set<uint32_t> changed(changes[0].indices.begin(), changes[0].indices.end());
  double             worstPos = 0;
  const fv::mesh::RenderTarget::Drawn& d = t->drawn(*geo);
  for(size_t i = 0; i < d.indices.size(); i++)
  {
    const size_t o = size_t(d.indices[i]) * 3;
    worstPos       = std::max<double>(worstPos, std::fabs(double(geo->position[o + 2]) - (double(base[o + 2]) + 0.5 * d.deltas[i * 3 + 2])));
    EXPECT_TRUE(changed.count(d.indices[i])) << d.indices[i];
  }
  EXPECT_LT(worstPos, 1e-5);
  // the normals around them agree with a weld of the whole mesh
  fv::mesh::NormalWelder welder(geo->renderIndex, geo->firstSplit, geo->nextSplit, geo->renderTris, geo->renderVertexCount);
  std::vector<float>     full(geo->normal.size());
  welder.full(geo->position, full);
  EXPECT_LT(worst(geo->normal, full), 1e-4);
}

TEST(FigureMorphSet, DriverRunsOnceForFilesThatArriveTogether)
{
  std::atomic<int> arrived{0};
  auto             gate = std::make_shared<std::promise<void>>();
  auto             fut  = gate->get_future().share();
  auto             geo  = std::make_shared<fv::mesh::BuiltMesh>();
  fv::morph::MorphSet morphs([&](int) { return geo.get(); });
  for(int i = 0; i < 40; i++)
    morphs.addLazy("m" + std::to_string(i), [fut, &arrived]() -> fv::morph::Finish {
      fut.wait();
      arrived.fetch_add(1);
      return [] {
        fv::morph::LazyResult r;
        r.keep = true;
        return r;
      };
    });
  std::vector<std::pair<std::string, double>> values;
  for(int i = 0; i < 40; i++)
    values.emplace_back("m" + std::to_string(i), 1);
  morphs.setMany(values);
  gate->set_value();
  while(arrived.load() < 40)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CountDriver driver;
  morphs.driver = &driver;
  morphs.pump();
  EXPECT_EQ(driver.calls, 1);
  morphs.settled();
  EXPECT_EQ(driver.calls, 1);
}

TEST(FigureMorphSet, DriverHasRunBeforeTheLoadSettles)
{
  auto                geo = std::make_shared<fv::mesh::BuiltMesh>();
  fv::morph::MorphSet morphs([&](int) { return geo.get(); });
  morphs.addLazy("m0", []() -> fv::morph::Finish {
    return [] {
      fv::morph::LazyResult r;
      r.keep = true;
      return r;
    };
  });
  morphs.set("m0", 1);
  CountDriver driver;
  morphs.driver = &driver;
  morphs.settled();
  EXPECT_EQ(driver.calls, 1);
}

TEST(FigureMorphSet, DriverRunsAgainWhileLoadsKeepArriving)
{
  std::atomic<int> arrived{0};
  std::promise<void> gates[2];
  auto               geo = std::make_shared<fv::mesh::BuiltMesh>();
  fv::morph::MorphSet morphs([&](int) { return geo.get(); });
  for(int i = 0; i < 2; i++)
  {
    auto fut = gates[i].get_future().share();
    morphs.addLazy("m" + std::to_string(i), held(fut, arrived));
  }
  morphs.setMany({{"m0", 1}, {"m1", 1}});
  CountDriver driver;
  morphs.driver = &driver;

  gates[0].set_value();
  while(arrived.load() < 1)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  morphs.pump();
  EXPECT_EQ(driver.calls, 0);  // the other file is still loading, and the wait has not elapsed

  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  morphs.pump();
  EXPECT_EQ(driver.calls, 1);  // not held until the slowest file

  gates[1].set_value();
  morphs.settled();
  EXPECT_EQ(driver.calls, 2);
}

// gpu-figures-plan.md F5: a cageOnly mesh's dial moves its cage alone, and its target never refines
// its drawn deltas; catchUp() (an export's) makes them and writes the drawn vertices a flush with the
// path off would have.
TEST(FigureMorphSet, ACageOnlyMeshsTargetsMakeNoDrawnDeltasUntilACatchUp)
{
  fv::mesh::BuiltMesh geo = grid(6, 2), reference = grid(6, 2);
  const auto          rest = geo.position;
  const auto          t    = targetOf(geo, {8, 9, 16}, {0, 0, 1, 0.5f, 0, 0.25f, 0, -0.5f, 0});
  fv::morph::MorphSet set([&](int) { return &geo; }), off([&](int) { return &reference; });
  set.cageOnly = [](int) { return true; };
  set.addTarget("bump", 0, t);
  off.addTarget("bump", 0, targetOf(reference, {8, 9, 16}, {0, 0, 1, 0.5f, 0, 0.25f, 0, -0.5f, 0}));
  set.takeChanges();
  set.set("bump", 0.75);
  off.set("bump", 0.75);
  const auto changes = set.takeChanges();
  ASSERT_EQ(changes.size(), 1u);
  EXPECT_TRUE(changes[0].cage);
  EXPECT_EQ(geo.position, rest);
  EXPECT_EQ(t->drawnIfMade(), nullptr);
  EXPECT_NEAR(set.morphedCage(0)[9 * 3], geo.cagePositions[9 * 3] + 0.375f, 1e-6);

  set.settled();
  set.pump();
  EXPECT_TRUE(set.takeChanges().empty());
  EXPECT_EQ(t->drawnIfMade(), nullptr);

  set.catchUp();
  const auto caught = set.takeChanges();
  ASSERT_EQ(caught.size(), 1u);
  EXPECT_TRUE(caught[0].caughtUp);
  EXPECT_NE(t->drawnIfMade(), nullptr);
  EXPECT_LT(worst(geo.position, reference.position), 1e-6);
  EXPECT_LT(worst(geo.normal, reference.normal), 1e-6);
}

// A mesh that stops being cageOnly (its feed no longer refines it) is drawn as its morphs have it at
// the next pump: a load moves cages alone before the host knows what its feed refines.
TEST(FigureMorphSet, AMeshNoLongerCageOnlyCatchesUpAtThePump)
{
  fv::mesh::BuiltMesh geo = grid(4, 1), reference = grid(4, 1);
  fv::morph::MorphSet set([&](int) { return &geo; }), off([&](int) { return &reference; });
  bool                gpu = true;
  set.cageOnly            = [&](int) { return gpu; };
  set.addTarget("lift", 0, targetOf(geo, {6, 7}, {0, 0, 0.5f, 0, 0, 0.5f}));
  off.addTarget("lift", 0, targetOf(reference, {6, 7}, {0, 0, 0.5f, 0, 0, 0.5f}));
  set.set("lift", 1);
  off.set("lift", 1);
  set.takeChanges();
  set.pump();
  EXPECT_TRUE(set.takeChanges().empty());
  gpu = false;
  set.pump();
  const auto caught = set.takeChanges();
  ASSERT_EQ(caught.size(), 1u);
  EXPECT_TRUE(caught[0].caughtUp);
  EXPECT_LT(worst(geo.position, reference.position), 1e-6);
}

TEST(FigureMorphSet, RemovedChannelsTakeTheirShapeBackOffTheMesh)
{
  fv::mesh::BuiltMesh geo  = grid(3, 0);
  const auto          rest = geo.position;
  fv::morph::MorphSet set([&](int) { return &geo; });
  fv::morph::ChannelOptions opts;
  opts.value = 0.6;
  set.addTarget("Pants Loose", 0, targetOf(geo, {5}, {0, 0, 1}), opts);
  set.flush();
  EXPECT_GT(worst(geo.position, rest), 0.5);
  // the item it came with is taken off: the dial goes, and so does what it applied
  set.removeChannels({"Pants Loose"});
  EXPECT_LT(worst(geo.position, rest), 1e-6);
  EXPECT_FALSE(set.has("Pants Loose"));
}

TEST(FigureMorphSet, AnItemsCopyOfALazyFigureMorphJoinsTheFiguresDial)
{
  fv::mesh::BuiltMesh body = grid(3, 0), shirt = grid(3, 0);
  fv::morph::MorphSet set([&](int m) { return m == 0 ? &body : &shirt; });
  set.addLazy("FBMHeavy", []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; });
  // the shirt ships its own fitted copy, stored at the value it was saved with
  fv::morph::ChannelOptions opts;
  opts.value = 0.5;
  set.addTarget("FBMHeavy", 1, targetOf(shirt, {5}, {0, 0, 1}), opts);
  const auto named = std::count_if(set.channels().begin(), set.channels().end(), [](const fv::morph::Channel& c) { return c.name == "FBMHeavy"; });
  EXPECT_EQ(named, 1);
  // one dial, at the figure's value: the shirt doesn't take a shape the body hasn't
  EXPECT_EQ(set.get("FBMHeavy"), 0);
  EXPECT_EQ(set.meshesOf("FBMHeavy"), std::set<int>{1});
}

namespace {

fv::dson::MorphInfo restingAt(double rest)
{
  fv::dson::MorphInfo info;
  info.path = "/Base";
  info.rest = rest;
  return info;
}

// A controller's file, counted when it is read.
fv::morph::Loader countingFile(std::atomic<int>& reads)
{
  return [&reads]() -> fv::morph::Finish {
    reads.fetch_add(1);
    return [] {
      fv::morph::LazyResult r;
      r.keep = true;
      return r;
    };
  };
}

}  // namespace

// A dial stands where its file says, and formulas read it there. That takes no read of the file: a
// figure has a switch of this kind for 130 corrective files. Told again, it has nothing more to do.
TEST(FigureMorphSet, ADialStandsAtItsFilesDefaultWithoutReadingTheFile)
{
  std::atomic<int>    reads{0};
  fv::morph::MorphSet set([](int) -> fv::mesh::BuiltMesh* { return nullptr; });
  CountDriver         driver;
  set.driver = &driver;
  set.addLazy("switch", countingFile(reads));
  const int seen = set.version();

  EXPECT_TRUE(set.describe("switch", restingAt(1)));
  const fv::morph::Channel& ch = *set.channel("switch");
  EXPECT_EQ(ch.value, 1);
  EXPECT_EQ(ch.rest, 1);
  EXPECT_EQ(ch.loaded, 1);
  EXPECT_FALSE(ch.dialled);
  EXPECT_EQ(set.weightOf("switch"), 1);
  EXPECT_GT(set.version(), seen);

  // the driver reads the new value on the next pump; nothing is loading
  EXPECT_FALSE(set.loading());
  set.pump();
  EXPECT_EQ(driver.calls, 1);
  set.settled();
  EXPECT_EQ(reads.load(), 0);
  EXPECT_TRUE(ch.lazy);

  EXPECT_FALSE(set.describe("switch", restingAt(1)));
  EXPECT_EQ(driver.calls, 1);
  EXPECT_FALSE(set.describe("no such dial", restingAt(1)));
}

// A default of nothing is what a channel starts at: no driver run for each of a figure's thousand dials.
TEST(FigureMorphSet, ADefaultOfNothingAsksForNothing)
{
  std::atomic<int>    reads{0};
  fv::morph::MorphSet set([](int) -> fv::mesh::BuiltMesh* { return nullptr; });
  CountDriver         driver;
  set.driver = &driver;
  set.addLazy("dial", countingFile(reads));
  EXPECT_FALSE(set.describe("dial", restingAt(0)));
  set.settled();
  EXPECT_EQ(driver.calls, 0);
  EXPECT_EQ(reads.load(), 0);
}

// Moving off the rest wants the file, as moving off 0 always has, and that includes moving to 0. Back
// at the rest asks for nothing more.
TEST(FigureMorphSet, LeavingTheRestReadsTheFile)
{
  std::atomic<int>    reads{0};
  fv::morph::MorphSet set([](int) -> fv::mesh::BuiltMesh* { return nullptr; });
  set.addLazy("switch", countingFile(reads));
  set.describe("switch", restingAt(1));
  set.set("switch", 1);
  set.settled();
  EXPECT_EQ(reads.load(), 0);
  set.set("switch", 0);
  set.settled();
  EXPECT_EQ(reads.load(), 1);
  EXPECT_EQ(set.get("switch"), 0);
  EXPECT_FALSE(set.channel("switch")->lazy);
  set.set("switch", 1);
  set.settled();
  EXPECT_EQ(reads.load(), 1);
}

// A scene's 0, or the user's, is a value: a describe that comes later doesn't replace it, whether or
// not it changed anything. The file is read as for any dial away from its rest, and a scene file holds
// the 0, so it comes back. The same for a value a scene gave when a worn item's channel was made.
TEST(FigureMorphSet, ADialSetToNothingStaysThroughALaterDescribe)
{
  std::atomic<int>    reads{0};
  auto                geo = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  fv::morph::MorphSet set([&](int) { return geo.get(); });
  set.addLazy("zero", countingFile(reads));
  set.addLazy("half", countingFile(reads));
  set.set("zero", 0);
  set.set("half", 0.5);
  fv::morph::ChannelOptions opts;
  opts.value = 0.25;
  set.addTarget("given", 0, targetOf(*geo, {0}, {0, 0, 1}), opts);

  EXPECT_TRUE(set.describe("zero", restingAt(1)));
  set.describe("half", restingAt(1));
  set.describe("given", restingAt(1));
  EXPECT_EQ(set.get("zero"), 0);
  EXPECT_EQ(set.get("half"), 0.5);
  EXPECT_EQ(set.get("given"), 0.25);
  EXPECT_EQ(set.channel("zero")->rest, 1);
  EXPECT_NE(set.channel("zero")->value, set.channel("zero")->loaded);
  set.settled();
  EXPECT_EQ(reads.load(), 2);
  EXPECT_EQ(set.get("zero"), 0);
}

// What a character's file set is how the figure loads, whatever the dial's own file says, and no
// scene file needs to hold it.
TEST(FigureMorphSet, ACharactersValueStaysWhatTheFigureLoadsAt)
{
  std::atomic<int>    reads{0};
  fv::morph::MorphSet set([](int) -> fv::mesh::BuiltMesh* { return nullptr; });
  set.addLazy("dial", countingFile(reads));
  set.set("dial", 0.6);
  set.settled();
  set.markLoaded();
  set.describe("dial", restingAt(1));
  EXPECT_EQ(set.get("dial"), 0.6);
  EXPECT_EQ(set.channel("dial")->loaded, 0.6);
  EXPECT_EQ(set.channel("dial")->rest, 1);
}

// A channel already bound when its default arrives moves the mesh to it: at once with no driver, and
// through the driver with one, the weight staying what the buffers hold until then.
TEST(FigureMorphSet, ABoundChannelsDefaultReachesTheMesh)
{
  auto                geo    = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  const auto          target = targetOf(*geo, {0}, {0, 0, 1});
  const size_t        z      = size_t(target->drawn(*geo).indices[0]) * 3 + 2;
  const float         base   = geo->position[z];
  fv::morph::MorphSet alone([&](int) { return geo.get(); });
  alone.addTarget("m", 0, target);
  EXPECT_TRUE(alone.describe("m", restingAt(0.5)));
  EXPECT_EQ(alone.weightOf("m"), 0.5);
  EXPECT_NEAR(geo->position[z], base + 0.5f, 1e-5);

  auto                geo2 = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  fv::morph::MorphSet driven([&](int) { return geo2.get(); });
  CountDriver         driver;
  driven.driver = &driver;
  driven.addTarget("m", 0, targetOf(*geo2, {0}, {0, 0, 1}));
  EXPECT_TRUE(driven.describe("m", restingAt(0.5)));
  EXPECT_EQ(driven.get("m"), 0.5);
  EXPECT_EQ(driven.weightOf("m"), 0);
  driven.pump();
  EXPECT_EQ(driver.calls, 1);
  driven.applyWeights({{"m", 0.5}});
  EXPECT_NEAR(geo2->position[z], base + 0.5f, 1e-5);
}

// Reset goes to the file's default, not 0, for a dial set to 0 as for one set elsewhere. It leaves it
// undialled: one reset before its file was read still takes the default when that comes. A channel
// no file describes (a general body slider, a worn item's dial) goes to 0 as ever.
TEST(FigureMorphSet, AResetGoesToWhereTheFileRestsIt)
{
  std::atomic<int>    reads{0};
  auto                geo = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  fv::morph::MorphSet set([&](int) { return geo.get(); });
  set.addLazy("switch", countingFile(reads));
  set.addLazy("late", countingFile(reads));
  set.addLazy("fv_shape_waist", countingFile(reads));
  set.addTarget("worn", 0, targetOf(*geo, {0}, {0, 0, 1}));
  set.describe("switch", restingAt(1));
  set.set("switch", 0);
  set.set("late", 0.3);
  set.set("fv_shape_waist", -0.5);
  set.set("worn", 0.5);
  set.settled();

  set.resetAll();
  set.settled();
  EXPECT_EQ(set.get("switch"), 1);
  EXPECT_EQ(set.get("late"), 0);
  EXPECT_EQ(set.get("fv_shape_waist"), 0);
  EXPECT_EQ(set.get("worn"), 0);
  EXPECT_EQ(set.weightOf("switch"), 1);
  EXPECT_FALSE(set.channel("switch")->dialled);

  EXPECT_TRUE(set.describe("late", restingAt(0.25)));
  EXPECT_EQ(set.get("late"), 0.25);
}

// The list's first places are for what's been changed: a dial at its default isn't.
TEST(FigureMorphSet, SortPutsMovedDialsFirstNotDefaultOnes)
{
  std::atomic<int>    reads{0};
  fv::morph::MorphSet set([](int) -> fv::mesh::BuiltMesh* { return nullptr; });
  fv::morph::ChannelOptions opts;
  opts.group = "Head";
  set.addLazy("a switch", countingFile(reads), opts);
  set.addLazy("b dial", countingFile(reads), opts);
  set.addLazy("c switch moved", countingFile(reads), opts);
  set.describe("a switch", restingAt(1));
  set.describe("c switch moved", restingAt(1));
  set.set("c switch moved", 0);
  set.settled();
  set.sort();
  ASSERT_EQ(set.channels().size(), 3u);
  EXPECT_EQ(set.channels()[0].name, "c switch moved");
}

namespace {

// A file with nothing in it for the figure, as Genesis 9's has for Eye Pupils Dilate.
fv::morph::Loader emptyFile()
{
  return []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; };
}

// A worn item holding a dial's deltas in a file of its own, read by the time it is asked for. Moving
// with the body isn't what these show, so it projects nothing.
fv::morph::Follower holdingItem(const std::string& channel, fv::dson::TargetPtr copy)
{
  fv::morph::Follower f;
  f.mesh   = 1;
  f.source = 0;
  f.label  = "Eyes";
  f.hasOwn = [channel](const std::string& name) { return name == channel; };
  f.own    = [channel, copy](const std::string& name) -> std::shared_future<std::vector<fv::dson::TargetPtr>> {
    if(name != channel)
      return {};
    std::promise<std::vector<fv::dson::TargetPtr>> file;
    file.set_value({copy});
    return file.get_future().share();
  };
  f.project = [](const fv::mesh::RenderTarget&) -> fv::dson::TargetPtr {
    ADD_FAILURE() << "the source has nothing to project";
    return nullptr;
  };
  return f;
}

// Two meshes, and a target that lifts a vertex of the second by 1 at full weight.
struct WornItemSet
{
  std::shared_ptr<fv::mesh::BuiltMesh> body = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  std::shared_ptr<fv::mesh::BuiltMesh> item = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  fv::dson::TargetPtr                  copy = targetOf(*item, {5}, {0, 0, 1});
  size_t                               z    = size_t(copy->drawn(*item).indices[0]) * 3 + 2;
  float                                base = item->position[z];
  fv::morph::MorphSet                  set{[this](int m) { return m == 0 ? body.get() : item.get(); }};
  float lift() const { return item->position[z] - base; }
};

}  // namespace

// A dial whose file has nothing for the figure stays on the list: it used to go on its first drag, and
// a preset's value for it then counted as skipped. It keeps its value, and a later listing of the same
// name makes no second dial. A body slider's loader says keep, so it isn't inert.
TEST(FigureMorphSet, ADialWithNothingInItsFileStaysListed)
{
  auto                geo = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  fv::morph::MorphSet set([&](int) { return geo.get(); });
  std::atomic<int>    reads{0};
  set.addLazy("Pupils", emptyFile());
  set.addLazy("fv_shape_waist", countingFile(reads));
  const int listed = set.size();

  set.setMany({{"Pupils", 0.5}, {"fv_shape_waist", 0.5}});
  set.settled();

  const fv::morph::Channel* pupils = set.channel("Pupils");
  ASSERT_NE(pupils, nullptr);
  EXPECT_TRUE(pupils->inert);
  EXPECT_FALSE(pupils->lazy);
  EXPECT_EQ(pupils->value, 0.5);
  EXPECT_TRUE(set.has("Pupils"));
  EXPECT_EQ(set.size(), listed);
  EXPECT_FALSE(set.channel("fv_shape_waist")->inert);

  set.addLazy("Pupils", emptyFile());
  EXPECT_EQ(set.size(), listed);
  set.set("Pupils", 0);
  set.settled();
  EXPECT_EQ(set.channel("Pupils")->value, 0);
  set.resetAll();
  EXPECT_EQ(set.size(), listed);
}

// The removal this replaces still holds for what has no slider: a hidden channel whose file is empty
// goes. So does a dial whose file isn't there at all, or won't read.
TEST(FigureMorphSet, AnEmptyHiddenChannelAndAMissingFileStillGo)
{
  fv::morph::MorphSet set([](int) -> fv::mesh::BuiltMesh* { return nullptr; });
  fv::morph::ChannelOptions hidden;
  hidden.hidden = true;
  set.addLazy("driven", emptyFile(), hidden);
  set.addLazy("missing", []() -> fv::morph::Finish { throw fv::dson::MissingFileError("missing.dsf"); });
  set.addLazy("broken", []() -> fv::morph::Finish { throw std::runtime_error("bad file"); });
  set.setMany({{"driven", 1}, {"missing", 1}, {"broken", 1}});
  set.settled();
  EXPECT_EQ(set.channel("driven"), nullptr);
  EXPECT_EQ(set.channel("missing"), nullptr);
  EXPECT_EQ(set.channel("broken"), nullptr);
  EXPECT_EQ(set.takeErrors().size(), 1u);
}

// A file that moves something isn't inert, nor is a controller, nor a switch that rests away from 0,
// including one whose rest is only known after its file was read.
TEST(FigureMorphSet, OnlyAFileWithNothingAtAllMakesADialInert)
{
  auto                geo    = std::make_shared<fv::mesh::BuiltMesh>(grid(3, 0));
  const auto          target = targetOf(*geo, {0}, {0, 0, 1});
  fv::morph::MorphSet set([&](int) { return geo.get(); });
  std::atomic<int>    reads{0};
  set.addLazy("moves", [target]() -> fv::morph::Finish {
    return [target] {
      fv::dson::MorphPart part;
      part.mesh   = 0;
      part.target = target;
      fv::morph::LazyResult r;
      r.parts.push_back(part);
      return r;
    };
  });
  set.addLazy("controller", countingFile(reads));
  set.addLazy("switch", emptyFile());
  set.addLazy("late switch", emptyFile());
  set.describe("switch", restingAt(1));
  set.setMany({{"moves", 1}, {"controller", 1}, {"switch", 0}, {"late switch", 1}});
  set.settled();
  EXPECT_FALSE(set.channel("moves")->inert);
  EXPECT_FALSE(set.channel("controller")->inert);
  EXPECT_FALSE(set.channel("switch")->inert);
  EXPECT_TRUE(set.channel("late switch")->inert);
  set.describe("late switch", restingAt(1));
  EXPECT_FALSE(set.channel("late switch")->inert);
}

// The dial moves the item that holds its deltas, though the body has nothing for it, and the dial stays.
// Back at 0 the item is back too.
TEST(FigureMorphSet, AnEmptyDialMovesAFollowerWithItsOwnCopy)
{
  WornItemSet w;
  w.set.addLazy("Pupils", emptyFile());
  w.set.addFollower(holdingItem("Pupils", w.copy));

  w.set.set("Pupils", 0.5);
  w.set.settled();
  ASSERT_NE(w.set.channel("Pupils"), nullptr);
  EXPECT_TRUE(w.set.channel("Pupils")->inert);
  w.set.pump();  // the copy is taken here, as a frame would
  EXPECT_NEAR(w.lift(), 0.5f, 1e-5);

  w.set.set("Pupils", 1);
  EXPECT_NEAR(w.lift(), 1.f, 1e-5);
  w.set.set("Pupils", 0);
  EXPECT_NEAR(w.lift(), 0.f, 1e-5);
  EXPECT_EQ(w.set.channel("Pupils")->value, 0);
}

// A copy that arrives after the dial went back has nothing to apply, and a follower without a copy of
// its own isn't touched by a dial that has nothing for it.
TEST(FigureMorphSet, ACopyArrivingAfterTheDialWentBackDoesNothing)
{
  WornItemSet                                    w;
  std::promise<std::vector<fv::dson::TargetPtr>> file;
  fv::morph::Follower                            f = holdingItem("Pupils", w.copy);
  f.own = [held = file.get_future().share()](const std::string&) { return held; };
  w.set.addLazy("Pupils", emptyFile());
  w.set.addLazy("Other", emptyFile());
  w.set.addFollower(f);
  w.set.setMany({{"Pupils", 1}, {"Other", 1}});
  w.set.settled();
  w.set.set("Pupils", 0);
  w.set.pump();
  file.set_value({w.copy});
  w.set.pump();
  EXPECT_EQ(w.lift(), 0.f);
  EXPECT_TRUE(w.set.activeTargets(1).empty());
}

// An item put on while the dial's file is on its way waits with it.
TEST(FigureMorphSet, AFollowerAddedWhileTheFileLoadsMovesWhenItArrives)
{
  WornItemSet        w;
  std::promise<void> arrive;
  w.set.addLazy("Pupils", [gate = arrive.get_future().share()]() -> fv::morph::Finish {
    gate.wait();
    return [] { return fv::morph::LazyResult{}; };
  });
  w.set.set("Pupils", 1);
  w.set.addFollower(holdingItem("Pupils", w.copy));
  w.set.pump();
  EXPECT_EQ(w.lift(), 0.f);  // nothing asked of it yet
  arrive.set_value();
  w.set.settled();
  w.set.pump();
  EXPECT_NEAR(w.lift(), 1.f, 1e-5);
}

// What the panel will say a dial moves: the items that hold something for it, each once however many
// meshes it has. A follower with neither copy nor binding isn't one, and it goes when the item does.
TEST(FigureMorphSet, ADialSaysWhichItemsItMoves)
{
  WornItemSet w;
  w.set.addLazy("Pupils", emptyFile());
  w.set.addLazy("Other", emptyFile());
  w.set.addTarget("Bound", 1, w.copy);
  const int before = w.set.listVersion();
  w.set.addFollower(holdingItem("Pupils", w.copy));
  EXPECT_GT(w.set.listVersion(), before);
  w.set.addFollower(holdingItem("Pupils", w.copy));

  const std::vector<std::string> eyes{"Eyes"};
  EXPECT_EQ(w.set.itemsMoving("Pupils"), eyes);
  EXPECT_EQ(w.set.itemsMoving("Bound"), eyes);
  EXPECT_TRUE(w.set.itemsMoving("Other").empty());

  w.set.removeMeshes({1});
  EXPECT_TRUE(w.set.itemsMoving("Pupils").empty());
  EXPECT_NE(w.set.channel("Pupils"), nullptr);
}

// A controller moves an item through the dials it drives, whose copies are the item's: its own name isn't
// in the item's folder. Once the audit has said which it drives.
TEST(FigureMorphSet, AControllerMovesTheItemsThatHoldWhatItDrives)
{
  WornItemSet w;
  w.set.addLazy("Natural", emptyFile());
  w.set.addLazy("Thick", emptyFile());
  w.set.addLazy("Curl", emptyFile());
  w.set.addFollower(holdingItem("Curl", w.copy));
  EXPECT_TRUE(w.set.itemsMoving("Natural").empty());

  fv::morph::Reach natural;
  natural.drives = {"Other", "Curl"};
  w.set.setReach("Natural", natural);
  fv::morph::Reach thick;
  thick.drives = {"Other"};
  w.set.setReach("Thick", thick);

  const std::vector<std::string> eyes{"Eyes"};
  EXPECT_EQ(w.set.itemsMoving("Natural"), eyes);
  EXPECT_TRUE(w.set.itemsMoving("Thick").empty());

  // and it says so again when a new list of what it drives comes
  const int before = w.set.listVersion();
  natural.drives   = {"Other"};
  w.set.setReach("Natural", natural);
  EXPECT_GT(w.set.listVersion(), before);
  EXPECT_TRUE(w.set.itemsMoving("Natural").empty());
}

// An item's binding to a dial doesn't make the dial the item's: with it taken off, the dial, whose own
// file was empty, stays for the next item, and a listing of the name makes no second one.
TEST(FigureMorphSet, AnEmptyDialOutlivesAnItemBoundToIt)
{
  WornItemSet w;
  w.set.addLazy("Pupils", emptyFile());
  w.set.addTarget("Pupils", 1, w.copy);
  w.set.set("Pupils", 0.5);
  w.set.settled();
  ASSERT_TRUE(w.set.channel("Pupils")->inert);
  const int listed = w.set.size();

  w.set.removeMeshes({1});
  ASSERT_NE(w.set.channel("Pupils"), nullptr);
  EXPECT_EQ(w.set.get("Pupils"), 0.5);
  EXPECT_TRUE(w.set.has("Pupils"));
  w.set.addLazy("Pupils", emptyFile());
  EXPECT_EQ(w.set.size(), listed);

  // an item's own dial, which a file never listed, goes with it as ever
  w.set.addTarget("Pants Loose", 1, w.copy);
  w.set.removeMeshes({1});
  EXPECT_EQ(w.set.channel("Pants Loose"), nullptr);
}

// A morph file's fields, as DAZ writes them, from the first few KB: the text may stop anywhere after
// them, and the deltas are never parsed.
TEST(MorphInfo, ReadsADialsFieldsFromTheStartOfItsFile)
{
  const std::string head = R"({
    "file_version": "0.6.0.0",
    "asset_info": {"id": "/data/x.dsf", "type": "modifier", "contributor": {"author": "Daz 3D"}},
    "modifier_library": [{
      "id": "body_bs_ProportionLegsLength",
      "name": "body_bs_ProportionLegsLength",
      "parent": "/data/Daz%203D/Genesis%209/Base/Genesis9.dsf#Genesis9-1",
      "presentation": {"type": "Modifier/Shape", "label": "", "colors": [[0.1, 0.2, 0.3], [1, 1, 1]]},
      "channel": {"id": "value", "type": "float", "name": "Value", "label": "Proportion Legs Length", "value": 0, "min": -2, "max": 2, "clamped": true},
      "region": "Legs",
      "group": "/Base",
      "formulas": [{"output": "stops before this", )";
  const auto info = fv::dson::morphInfoFromText(head);
  ASSERT_TRUE(info);
  EXPECT_EQ(info->id, "body_bs_ProportionLegsLength");
  EXPECT_EQ(info->label, "Proportion Legs Length");
  EXPECT_EQ(info->path, "/Base");
  EXPECT_EQ(info->region, "Legs");
  EXPECT_EQ(info->kind, "Modifier/Shape");
  EXPECT_DOUBLE_EQ(*info->min, -2);
  EXPECT_DOUBLE_EQ(*info->max, 2);
  EXPECT_EQ(info->type, "float");
  EXPECT_DOUBLE_EQ(info->rest, 0);
  EXPECT_FALSE(info->helper);
  // it stopped in the formulas: nothing of them, and no word on the deltas
  EXPECT_FALSE(info->complete);
  EXPECT_FALSE(info->formulas);
  EXPECT_TRUE(info->reads.empty());
  EXPECT_EQ(info->deltaCount, 0);

  // the value it stands at when the file doesn't say 0, a boolean's as a number, and none stated
  const auto restOf = [](const std::string& channel) {
    return fv::dson::morphInfoFromText(R"({"modifier_library": [{"id": "m", "channel": {)" + channel + "}}]}")->rest;
  };
  EXPECT_DOUBLE_EQ(restOf(R"("type": "float", "value": 0.25, "min": 0, "max": 1)"), 0.25);
  EXPECT_DOUBLE_EQ(restOf(R"("type": "float", "value": 1)"), 1);
  EXPECT_DOUBLE_EQ(restOf(R"("type": "bool", "value": true)"), 1);
  EXPECT_DOUBLE_EQ(restOf(R"("type": "bool", "value": false)"), 0);
  EXPECT_DOUBLE_EQ(restOf(R"("type": "float")"), 0);

  // the deltas' count, which comes before their values, and the formulas as they were written
  const auto whole = fv::dson::morphInfoFromText(R"({"modifier_library": [{
      "id": "m", "channel": {"type": "float", "label": "M", "value": 0.5},
      "formulas": [{"output": "#m?value", "stage": "mult", "operations": [{"op": "push", "url": "#gate?value"}, {"op": "push", "val": 2.5}, {"op": "mult"}], "on": true, "off": null}],
      "morph": {"vertex_count": 100, "deltas": {"count": 2, "values": [[0, 1, 1, )");
  ASSERT_TRUE(whole);
  EXPECT_TRUE(whole->complete);
  EXPECT_TRUE(whole->hasMorph);
  EXPECT_EQ(whole->deltaCount, 2);
  EXPECT_DOUBLE_EQ(whole->rest, 0.5);
  ASSERT_TRUE(whole->formulas);
  EXPECT_EQ(*whole->formulas, fv::dson::Json::parse(R"([{"output": "#m?value", "stage": "mult", "operations": [{"op": "push", "url": "#gate?value"}, {"op": "push", "val": 2.5}, {"op": "mult"}], "on": true, "off": null}])"));

  // what isn't a dial, each way a file says so
  const auto said = [](const std::string& presentation, const std::string& channel, const std::string& group) {
    return fv::dson::morphInfoFromText(R"({"modifier_library": [{"id": "m", "presentation": {"type": ")" + presentation + R"("}, "channel": {"label": "M")" + channel
                                       + R"(}, "group": ")" + group + R"(", "morph": {)");
  };
  EXPECT_FALSE(said("Modifier/Pose", "", "/Pose Controls/Arms")->helper);
  EXPECT_TRUE(said("Modifier/Pose", R"(, "visible": false)", "/Pose Controls/Arms")->helper);
  EXPECT_TRUE(said("Modifier/Corrective", "", "/Base")->helper);
  EXPECT_TRUE(said("Modifier/Shape", "", "/Hidden/People")->helper);
  const auto alias = said("Modifier/Pose", R"(, "type": "alias", "target_channel": "x")", "/Pose Controls/Arms");
  EXPECT_TRUE(alias->helper);
  EXPECT_TRUE(alias->alias);

  // stopping short of the first modifier, or a file with none
  EXPECT_FALSE(fv::dson::morphInfoFromText(R"({"asset_info": {"id": "/data/x)"));
  EXPECT_FALSE(fv::dson::morphInfoFromText(R"({"asset_info": {}, "scene": {}})"));
}

// A dial keeps its file's range, -2 included, as far as a slider is any use. Skimmed or loaded, the
// same: the range doesn't jump when the dial first moves.
TEST(MorphInfo, ARangeIsTheFilesAsFarAsASliderGoes)
{
  const auto range = [](const std::string& text) {
    const auto info = fv::dson::morphInfoFromText(R"({"modifier_library": [{"id": "m", "channel": {)" + text + "}, ");
    return std::pair{*info->min, *info->max};
  };
  EXPECT_EQ(range(R"("min": -2, "max": 2)"), (std::pair{-2.0, 2.0}));
  EXPECT_EQ(range(R"("min": -0.5, "max": 1.5)"), (std::pair{-0.5, 1.5}));
  EXPECT_EQ(range(R"("min": -10000, "max": 10000)"), (std::pair{-2.0, 2.0}));
}

// Told what its file says, a lazy channel takes its real label and range, and stays what it is.
TEST(MorphInfo, ALazyChannelTakesItsFilesLabelAndRange)
{
  fv::morph::MorphSet set([](int) -> fv::mesh::BuiltMesh* { return nullptr; });
  set.addLazy("body_bs_ProportionLegsLength", []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; });
  const int listed = set.listVersion();
  fv::dson::MorphInfo info;
  info.label  = "Proportion Legs Length";
  info.path   = "/Base";
  info.region = "Legs";
  info.kind   = "Modifier/Shape";
  info.min    = -2;
  info.max    = 2;
  set.describe("body_bs_ProportionLegsLength", info);
  set.describe("no such dial", info);
  const fv::morph::Channel& ch = set.channels().front();
  EXPECT_EQ(ch.label, "Proportion Legs Length");
  EXPECT_EQ(ch.path, "/Base");
  EXPECT_EQ(ch.region, "Legs");
  EXPECT_EQ(ch.min, -2);
  EXPECT_EQ(ch.max, 2);
  EXPECT_TRUE(ch.lazy);
  EXPECT_FALSE(ch.hidden);
  EXPECT_NE(set.listVersion(), listed);
}

// A file says which dials it reads, which nothing about those dials says: any sum formula's ?value
// input, whatever it sets, with deltas or without. A mult-stage input isn't one, and a head that stops
// short of the end can't tell.
TEST(MorphInfo, AFileSaysWhichDialsItReads)
{
  const std::string formula = R"("formulas": [
        {"output": "Genesis9:/data/x.dsf#Genesis9?scale/general", "operations": [{"op": "push", "url": "Genesis9:#ctrl?value"}, {"op": "push", "val": 0.25}, {"op": "mult"}]},
        {"output": "Genesis9:/data/x.dsf#ctrl?value", "operations": [{"op": "push", "url": "Genesis9:/data/y%20z.dsf#body_bs_Height?value"}, {"op": "push", "val": 1}, {"op": "mult"}]},
        {"output": "Genesis9:/data/x.dsf#ctrl?value", "operations": [{"op": "push", "url": "lHand:/data/x.dsf#lHand?rotation/x"}, {"op": "push", "val": 1}, {"op": "mult"}]},
        {"output": "Genesis9:/data/x.dsf#ctrl?value", "stage": "mult", "operations": [{"op": "push", "url": "Genesis9:#gate?value"}]},
        {"output": "Genesis9:/data/w.dsf#other?value", "stage": "sum", "operations": [{"op": "push", "url": "Genesis9:#input?value"}]}
      ])";
  const std::string start = R"({"modifier_library": [{"id": "ctrl", "channel": {"label": "C"}, )";
  const std::vector<std::string> reads{"body_bs_Height", "input"};

  const auto controller = fv::dson::morphInfoFromText(start + formula + "}]}");
  ASSERT_TRUE(controller);
  EXPECT_TRUE(controller->complete);
  EXPECT_EQ(controller->reads, reads);
  EXPECT_FALSE(controller->hasMorph);
  ASSERT_TRUE(controller->formulas);
  EXPECT_EQ(controller->formulas->size(), 5u);

  const auto withDeltas = fv::dson::morphInfoFromText(start + formula + R"(, "morph": {"deltas": {"count": 40, "values": [[0, )");
  ASSERT_TRUE(withDeltas);
  EXPECT_TRUE(withDeltas->complete);
  EXPECT_TRUE(withDeltas->hasMorph);
  EXPECT_EQ(withDeltas->deltaCount, 40);
  EXPECT_EQ(withDeltas->reads, reads);

  const auto cutShort = fv::dson::morphInfoFromText(start + formula.substr(0, formula.size() - 12));
  ASSERT_TRUE(cutShort);
  EXPECT_FALSE(cutShort->complete);
  EXPECT_TRUE(cutShort->reads.empty());
}

// facs_ctrl_MouthSmile has a morph block with nothing in it: a controller like any other. Whatever
// comes after an empty block is still read.
TEST(MorphInfo, AMorphBlockOfNoDeltasCountsAsNone)
{
  const std::string formula = R"("formulas": [{"output": "#ctrl?value", "operations": [{"op": "push", "url": "#dial?value"}]}])";
  const std::string block   = R"("morph": {"vertex_count": 100, "deltas": {"count": 0, "values": []}})";
  const std::string start   = R"({"modifier_library": [{"id": "ctrl", "channel": {"label": "C"}, )";

  const auto before = fv::dson::morphInfoFromText(start + formula + ", " + block + "}]}");
  ASSERT_TRUE(before);
  EXPECT_TRUE(before->complete);
  EXPECT_EQ(before->deltaCount, 0);
  EXPECT_FALSE(before->hasMorph);
  EXPECT_EQ(before->reads, (std::vector<std::string>{"dial"}));

  const auto after = fv::dson::morphInfoFromText(start + block + ", " + formula + "}]}");
  ASSERT_TRUE(after);
  EXPECT_FALSE(after->hasMorph);
  EXPECT_EQ(after->reads, (std::vector<std::string>{"dial"}));
}

// Formulas can run past what's read of a file at first (a character's do): it's then read whole.
TEST(MorphInfo, AHeadCutShortReadsTheWholeFile)
{
  std::string formulas;
  for(int i = 0; i < 6000; i++)
    formulas += std::string(i ? ", " : "") + R"({"output": "#other?value", "operations": [{"op": "push", "url": "#dial)" + std::to_string(i) + R"(?value"}, {"op": "push", "val": 1}, {"op": "mult"}]})";
  const std::string text = R"({"modifier_library": [{"id": "big", "channel": {"type": "float", "label": "Big", "value": 0.5}, "formulas": [)" + formulas
                           + R"(], "morph": {"deltas": {"count": 7, "values": [[0, 0, 0, 1]]}}}]})";
  ASSERT_GT(text.size(), 256u * 1024u);

  const auto cut = fv::dson::morphInfoFromText(std::string_view(text).substr(0, 256 * 1024));
  ASSERT_TRUE(cut);
  EXPECT_FALSE(cut->complete);
  EXPECT_FALSE(cut->hasMorph);
  EXPECT_EQ(cut->label, "Big");

  const std::filesystem::path path = std::filesystem::temp_directory_path() / "fv_morph_info_whole.dsf";
  {
    std::ofstream out(path, std::ios::binary);
    out << text;
  }
  const auto info = fv::dson::morphInfo(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(info);
  EXPECT_TRUE(info->complete);
  EXPECT_TRUE(info->hasMorph);
  EXPECT_EQ(info->deltaCount, 7);
  EXPECT_DOUBLE_EQ(info->rest, 0.5);
  EXPECT_EQ(info->reads.size(), 6000u);
  ASSERT_TRUE(info->formulas);
  EXPECT_EQ(info->formulas->size(), 6000u);
}

// A channel holding something other than a number is no dial: Developer Tools and LoadHeadSplit load a
// file. A stated float or bool is one, and so is a channel that states nothing.
TEST(MorphInfo, AChannelOfAnotherTypeIsAHelper)
{
  const auto said = [](const std::string& channel) {
    return fv::dson::morphInfoFromText(R"({"modifier_library": [{"id": "m", "channel": {"label": "M")" + channel + "}}]}");
  };
  const auto file = said(R"(, "type": "file", "value": "/People/Genesis 9/Developer Kit/x.duf")");
  EXPECT_TRUE(file->helper);
  EXPECT_FALSE(file->alias);
  EXPECT_EQ(file->type, "file");
  EXPECT_DOUBLE_EQ(file->rest, 0);
  EXPECT_TRUE(said(R"(, "type": "string")")->helper);
  EXPECT_FALSE(said(R"(, "type": "float")")->helper);
  EXPECT_FALSE(said(R"(, "type": "bool", "value": true)")->helper);
  EXPECT_FALSE(said("")->helper);
}
