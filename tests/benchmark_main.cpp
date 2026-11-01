#include <benchmark/benchmark.h>
#include <filesystem>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include "common/test_utils.hpp"
#include "fv_figure_feed.hpp"
#include "scene_import_gltf.hpp"

//--------------------------------------------------------------------------------------------------
// The retained scene (daz-viewer docs/retained-scene-plan.md) on the Box Figure from the figure
// library's fixtures: the figure fed straight into the store, against the build it replaces, and
// the build's glTF imported into the store.

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

BENCHMARK_MAIN();
