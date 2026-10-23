// Parity harness, native side: runs a case (test/parity/cases/*.json in the app repository) through
// the figure library and writes what it built in the shape the TypeScript dumper writes
// (test/parity/dump.parity.ts), for scripts/parity/compare.mjs to diff.
//
//   fv_parity <case.json> <out.json> [--root <dir for relative library paths>]

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

#include <figure/dson/figure_loader.hpp>

using fv::dson::Json;

namespace {

std::filesystem::path pathOf(const char* utf8)
{
  const std::string s(utf8);
  return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

// Numbers rounded to 6 decimals as the TypeScript dumper rounds them (Math.round: halves up).
double r(double v)
{
  return std::floor(v * 1e6 + 0.5) / 1e6;
}

Json dumpScene(const fv::SceneGraph& scene, int root)
{
  const auto worlds = scene.worlds();
  Json       bones  = Json::array();
  Json       meshes = Json::array();
  // depth first, parents before children, as Object3D.traverse visits
  std::vector<int> stack{root};
  while(!stack.empty())
  {
    const int i = stack.back();
    stack.pop_back();
    const fv::Node& n = scene.nodes[i];
    if(n.kind == fv::NodeKind::eBone)
    {
      Json world = Json::array();
      for(double e : worlds[i].e)
        world.push_back(r(e));
      const bool parentIsBone = n.parent >= 0 && scene.nodes[n.parent].kind == fv::NodeKind::eBone;
      bones.push_back({{"name", n.name},
                       {"parent", parentIsBone ? Json(scene.nodes[n.parent].name) : Json(nullptr)},
                       {"position", Json::array({r(n.position.x), r(n.position.y), r(n.position.z)})},
                       {"world", world}});
    }
    if(n.kind == fv::NodeKind::eMesh)
    {
      const fv::Mesh& m = scene.meshes[n.mesh];
      Json            positions = Json::array();
      for(float v : m.geo->position)
        positions.push_back(r(v));
      Json joints = Json::array();
      for(int b : m.bones)
        joints.push_back(scene.nodes[b].name);
      meshes.push_back({{"name", m.name},
                        {"vertexCount", m.geo->position.size() / 3},
                        {"indexCount", m.geo->fullIndex.size()},
                        {"positions", positions},
                        {"indices", m.geo->fullIndex},
                        {"skinned", m.skinned()},
                        {"joints", joints}});
    }
    for(auto it = n.children.rbegin(); it != n.children.rend(); ++it)
      stack.push_back(*it);
  }
  return {{"bones", bones}, {"meshes", meshes}};
}

}  // namespace

int main(int argc, char** argv)
{
  if(argc < 3)
  {
    std::cerr << "usage: fv_parity <case.json> <out.json> [--root <dir>]\n";
    return 2;
  }
  const std::filesystem::path casePath = pathOf(argv[1]);
  const std::filesystem::path outPath  = pathOf(argv[2]);
  std::filesystem::path       root     = std::filesystem::current_path();
  for(int i = 3; i + 1 < argc; i++)
    if(std::string(argv[i]) == "--root")
      root = pathOf(argv[i + 1]);
  try
  {
    std::ifstream f(casePath);
    const Json    c = Json::parse(f);
    const std::string lib     = c.at("library").get<std::string>();
    // "$DAZ_LIBRARY": the real library, from the environment; without it the case is skipped
    std::filesystem::path libraryDir;
    if(lib == "$DAZ_LIBRARY")
    {
#pragma warning(suppress : 4996)  // getenv: read once, at start-up
      const char* env = std::getenv("DAZ_LIBRARY");
      if(!env || !*env)
      {
        std::cerr << casePath.stem().string() << ": skipped (DAZ_LIBRARY is not set)\n";
        return 0;
      }
      libraryDir = pathOf(env);
    }
    else
      libraryDir = pathOf(lib.c_str());
    if(libraryDir.is_relative())
      libraryDir = root / libraryDir;
    auto store = fv::dson::Store(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{libraryDir}));
    std::optional<fv::dson::LoadResult> loaded;
    Json                                dumps = Json::object();
    for(const Json& step : c.at("steps"))
    {
      if(step.contains("load"))
        loaded = fv::dson::loadDson(store, step["load"].get<std::string>(), {.subdivision = step.value("subdivision", 0)});
      else if(step.contains("dump"))
      {
        if(!loaded)
          throw std::runtime_error("dump before load");
        dumps[step["dump"].get<std::string>()] = dumpScene(loaded->scene, loaded->root);
      }
    }
    std::filesystem::create_directories(outPath.parent_path());
    std::ofstream o(outPath);
    o << Json{{"case", casePath.stem().u8string()}, {"dumps", dumps}}.dump();
    if(loaded)
      for(const auto& w : loaded->warnings)
        std::cerr << "warning: " << w << "\n";
    return 0;
  }
  catch(const std::exception& e)
  {
    std::cerr << "fv_parity: " << e.what() << "\n";
    return 1;
  }
}
