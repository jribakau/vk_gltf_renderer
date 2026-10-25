// Incremental morph flush matches a rebuild, a local morph reports only the vertices it moved, and
// lazy files that arrive together run the driver once. Ports test/morphs/incremental.test.ts (the
// position half; this MorphSet re-welds normals instead of applying normal deltas) and
// test/morphs/driverCoalesce.test.ts.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <set>
#include <thread>
#include <vector>

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
    for(size_t j = 0; j < targets[i]->indices.size(); j++)
      for(int c = 0; c < 3; c++)
        base[size_t(targets[i]->indices[j]) * 3 + c] += float(targets[i]->deltas[j * 3 + c] * weights[i]);
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
  // a band of 50 cage rows: about 24 000 drawn vertices, over one chunk and under a quarter of the mesh
  std::vector<int32_t> verts;
  std::vector<float>   deltas;
  for(int j = 60; j < 110; j++)
    for(int i = 0; i <= 200; i++)
    {
      verts.push_back(j * 201 + i);
      deltas.insert(deltas.end(), {0.f, 0.f, 0.01f * float(i % 7 + 1)});
    }
  const auto          t = targetOf(*geo, verts, deltas);
  ASSERT_GT(t->indices.size(), 16384u);
  ASSERT_LT(t->indices.size(), geo->position.size() / 3 / 4);
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
  for(size_t i = 0; i < t->indices.size(); i++)
  {
    const size_t o = size_t(t->indices[i]) * 3;
    worstPos       = std::max<double>(worstPos, std::fabs(double(geo->position[o + 2]) - (double(base[o + 2]) + 0.5 * t->deltas[i * 3 + 2])));
    EXPECT_TRUE(changed.count(t->indices[i])) << t->indices[i];
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
