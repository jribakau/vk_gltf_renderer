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
