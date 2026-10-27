#include <benchmark/benchmark.h>
#include <filesystem>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <gltf_scene.hpp>
#include "common/test_utils.hpp"
#include "fv_figure_feed.hpp"
#include "scene_import_gltf.hpp"

// Benchmark scene loading
static void BM_SceneLoad_Simple(benchmark::State& state)
{
  try
  {
    auto path = gltf_test::TestResources::getResourcePath("cube.gltf");
    for(auto _ : state)
    {
      nvvkgltf::Scene scene;
      if(!scene.load(path))
        state.SkipWithError("load failed");
      benchmark::DoNotOptimize(scene.getRenderNodes().size());
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneLoad_Simple);

// Benchmark complex scene loading
static void BM_SceneLoad_Complex(benchmark::State& state)
{
  try
  {
    auto path = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    for(auto _ : state)
    {
      nvvkgltf::Scene scene;
      if(!scene.load(path))
        state.SkipWithError("load failed");
      benchmark::DoNotOptimize(scene.getRenderNodes().size());
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneLoad_Complex);

// Benchmark scene saving
static void BM_SceneSave(benchmark::State& state)
{
  try
  {
    // Load once
    auto            loadPath = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    nvvkgltf::Scene scene;
    if(!scene.load(loadPath))
      state.SkipWithError("load failed");

    auto savePath = gltf_test::TestResources::getTempPath("benchmark_save.gltf");

    for(auto _ : state)
    {
      bool saved = scene.save(savePath);
      benchmark::DoNotOptimize(saved);
    }

    // Cleanup
    if(std::filesystem::exists(savePath))
    {
      std::filesystem::remove(savePath);
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneSave);

// Benchmark round-trip (load + save + load)
static void BM_SceneRoundTrip(benchmark::State& state)
{
  try
  {
    auto loadPath = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    auto savePath = gltf_test::TestResources::getTempPath("benchmark_roundtrip.gltf");

    for(auto _ : state)
    {
      // Load original
      nvvkgltf::Scene scene1;
      if(!scene1.load(loadPath))
        state.SkipWithError("load failed");

      // Save
      if(!scene1.save(savePath))
        state.SkipWithError("save failed");

      // Load saved
      nvvkgltf::Scene scene2;
      if(!scene2.load(savePath))
        state.SkipWithError("load failed");

      benchmark::DoNotOptimize(scene2.getRenderNodes().size());
    }

    // Cleanup
    if(std::filesystem::exists(savePath))
    {
      std::filesystem::remove(savePath);
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneRoundTrip);

// Benchmark node matrix updates
static void BM_UpdateNodeWorldMatrices(benchmark::State& state)
{
  try
  {
    auto            path = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    nvvkgltf::Scene scene;
    if(!scene.load(path))
      state.SkipWithError("load failed");

    for(auto _ : state)
    {
      scene.updateNodeWorldMatrices();
      benchmark::DoNotOptimize(scene.getRenderNodes().size());
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_UpdateNodeWorldMatrices);

//--------------------------------------------------------------------------------------------------
// The retained scene (daz-viewer docs/retained-scene-plan.md) on the Box Figure from the figure
// library's fixtures: the figure fed straight into the store, against the build it replaces, and
// the build's glTF imported into the store, against the legacy scene taking it.

namespace {

std::unique_ptr<fv::Figure> boxFigure()
{
  const std::filesystem::path library = std::filesystem::path(FV_TEST_DATA) / "daz-library";
  fv::dson::LoadOptions       options;
  options.subdivision = 1;
  auto f = fv::Figure::load(std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{library})),
                            "People/Test/Box Figure.duf", options);
  f->settle();
  f->drain();
  return f;
}

// loaded once: what the benchmarks time is what comes after
const fv::Figure& box()
{
  static const std::unique_ptr<fv::Figure> f = boxFigure();
  return *f;
}

}  // namespace

// The glTF the renderer gets today (no textures), as fv_parity's {"build": n}
static void BM_BuildGltf(benchmark::State& state)
{
  const fv::Figure& f = box();
  for(auto _ : state)
    benchmark::DoNotOptimize(fv::build::buildGltf(f.loaded.scene, f.loaded.root, {}).model.meshes.size());
}
BENCHMARK(BM_BuildGltf)->Unit(benchmark::kMicrosecond);

// A first feed into a fresh store, as fv_parity's {"feed": n}
static void BM_FeedFigure(benchmark::State& state)
{
  const fv::Figure& f = box();
  for(auto _ : state)
  {
    rs::RetainedScene scene;
    fv::FigureFeed    feed(scene, {});
    feed.sync(f.loaded.scene, f.loaded.root);
    benchmark::DoNotOptimize(scene.entityCount());
  }
}
BENCHMARK(BM_FeedFigure)->Unit(benchmark::kMicrosecond);

// A sync with nothing changed: what the feed costs every frame once it is wired
static void BM_FeedFigureAgain(benchmark::State& state)
{
  const fv::Figure& f = box();
  rs::RetainedScene scene;
  fv::FigureFeed    feed(scene, {});
  feed.sync(f.loaded.scene, f.loaded.root);
  for(auto _ : state)
  {
    feed.sync(f.loaded.scene, f.loaded.root);
    benchmark::DoNotOptimize(scene.takeChanges().empty());
  }
}
BENCHMARK(BM_FeedFigureAgain)->Unit(benchmark::kMicrosecond);

// The build's glTF into a fresh store (oracle 3's importer)
static void BM_RetainedImport(benchmark::State& state)
{
  const fv::Figure&     f     = box();
  const tinygltf::Model model = fv::build::buildGltf(f.loaded.scene, f.loaded.root, {}).model;
  for(auto _ : state)
  {
    rs::RetainedScene scene;
    benchmark::DoNotOptimize(rs::importGltf(scene, model).entities.size());
  }
}
BENCHMARK(BM_RetainedImport)->Unit(benchmark::kMicrosecond);

// The same glTF taken by the legacy scene, as the renderer's load takes the composed model
static void BM_SceneTakeModel(benchmark::State& state)
{
  const fv::Figure&     f     = box();
  const tinygltf::Model model = fv::build::buildGltf(f.loaded.scene, f.loaded.root, {}).model;
  for(auto _ : state)
  {
    nvvkgltf::Scene scene;
    scene.takeModel(tinygltf::Model(model));
    benchmark::DoNotOptimize(scene.getRenderNodes().size());
  }
}
BENCHMARK(BM_SceneTakeModel)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
