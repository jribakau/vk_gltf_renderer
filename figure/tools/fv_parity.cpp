// Runs a case (cases/*.json; the app repository's parity cases before the TypeScript side was
// deleted) through the figure library, with no GPU, and writes what it built: bones, meshes and morph
// weights in the shape the TypeScript dumper wrote, so two runs can be diffed.
//
//   fv_parity <case.json> <out.json> [--root <dir for relative library paths>]
//
// Steps: {"load"}, {"morph"}, {"pose"}, {"preset"}, {"dump"}, and for the wardrobe {"wear": rel},
// {"remove": item id} and {"fit": label}, which measures how the worn items sit on the body: for
// each item vertex, its signed distance to the nearest body vertex along that vertex's normal.
// {"build": n} builds the glTF the renderer would get (without textures) n times, as the host
// builds it again after each wardrobe change, and prints the hair strand directions it measured;
// {"wear": rel, "measure": true} measures a wearable's strands ahead, as the host's wear does.
//
// Every step's time goes to stderr ("[step] ..."), with the library's phase timers ("[fv-time] ...")
// inside it: the case in cases/native_test.json is the profile of a scene file's load.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <unordered_map>

#include <figure/dson/material_preset.hpp>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/io/timing.hpp>

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

Json dumpScene(const fv::SceneGraph& scene, int root, const fv::morph::MorphSet* morphs)
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
  // the weight every morph is applied at, formulas included (those that are not zero)
  Json weights = Json::object();
  if(morphs)
    for(const fv::morph::Channel& ch : morphs->channels())
      if(const double w = morphs->weightOf(ch.name); w != 0)
        weights[ch.name] = r(w);
  return {{"bones", bones}, {"meshes", meshes}, {"weights", weights}};
}

// How each worn item sits on the body, in the rest pose: the signed distance (cm) of every drawn item
// vertex to the nearest body vertex, along that body vertex's normal. Negative is under the skin.
Json fitReport(const fv::Figure& fig)
{
  const fv::SceneGraph& scene  = fig.loaded.scene;
  const auto            worlds = scene.worlds();
  std::vector<int>      nodeOf(scene.meshes.size(), -1);
  for(size_t n = 0; n < scene.nodes.size(); n++)
    if(scene.nodes[n].kind == fv::NodeKind::eMesh && scene.nodes[n].mesh >= 0)
      nodeOf[size_t(scene.nodes[n].mesh)] = int(n);
  const auto& ctx = fig.loaded.morphs;
  if(ctx.mainMeshes.empty())
    throw std::runtime_error("fit without a figure body");
  const int             bodyMesh = ctx.meshes[ctx.mainMeshes[0]].mesh;
  const fv::mesh::BuiltMesh& body = *scene.meshes[size_t(bodyMesh)].geo;
  const fv::Mat4        bodyWorld = nodeOf[size_t(bodyMesh)] >= 0 ? worlds[size_t(nodeOf[size_t(bodyMesh)])] : fv::Mat4{};
  const size_t          nb        = body.position.size() / 3;
  std::vector<fv::Vec3> bp(nb), bn(nb);
  for(size_t v = 0; v < nb; v++)
  {
    bp[v] = bodyWorld.transformPoint({body.position[v * 3], body.position[v * 3 + 1], body.position[v * 3 + 2]});
    const fv::Vec3 tip = bodyWorld.transformPoint({body.position[v * 3] + body.normal[v * 3], body.position[v * 3 + 1] + body.normal[v * 3 + 1],
                                                   body.position[v * 3 + 2] + body.normal[v * 3 + 2]});
    fv::Vec3       d   = {tip.x - bp[v].x, tip.y - bp[v].y, tip.z - bp[v].z};
    const double   len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    bn[v]              = len > 0 ? fv::Vec3{d.x / len, d.y / len, d.z / len} : fv::Vec3{0, 1, 0};
  }
  // body vertices in 2 cm cells
  const double                                        cell = 2.0;
  std::unordered_map<int64_t, std::vector<uint32_t>> grid;
  const auto key = [](int64_t x, int64_t y, int64_t z) { return (x & 0x1FFFFF) | ((y & 0x1FFFFF) << 21) | ((z & 0x1FFFFF) << 42); };
  const auto cellOf = [&](double c) { return int64_t(std::floor(c / cell)); };
  for(size_t v = 0; v < nb; v++)
    grid[key(cellOf(bp[v].x), cellOf(bp[v].y), cellOf(bp[v].z))].push_back(uint32_t(v));
  const auto nearest = [&](const fv::Vec3& q) -> int64_t {
    const int64_t cx = cellOf(q.x), cy = cellOf(q.y), cz = cellOf(q.z);
    int64_t       best = -1;
    double        bestD = 1e30;
    for(int r = 0; r <= 8; r++)
    {
      for(int64_t x = cx - r; x <= cx + r; x++)
        for(int64_t y = cy - r; y <= cy + r; y++)
          for(int64_t z = cz - r; z <= cz + r; z++)
          {
            if(std::max({std::llabs(x - cx), std::llabs(y - cy), std::llabs(z - cz)}) != r)
              continue;
            auto it = grid.find(key(x, y, z));
            if(it == grid.end())
              continue;
            for(uint32_t v : it->second)
            {
              const double dx = bp[v].x - q.x, dy = bp[v].y - q.y, dz = bp[v].z - q.z;
              const double d  = dx * dx + dy * dy + dz * dz;
              if(d < bestD)
                bestD = d, best = v;
            }
          }
      // a vertex found within r cells is the nearest once the next ring can't be closer
      if(best >= 0 && std::sqrt(bestD) <= r * cell)
        break;
    }
    return best;
  };
  Json items = Json::array();
  for(const fv::WornItem& item : fig.items)
  {
    std::vector<double> dist;
    size_t              far = 0;
    for(int m : item.meshes)
    {
      const fv::mesh::BuiltMesh* geo = scene.meshes[size_t(m)].geo.get();
      if(!geo || nodeOf[size_t(m)] < 0)
        continue;
      const fv::Mat4& w = worlds[size_t(nodeOf[size_t(m)])];
      for(size_t v = 0; v < geo->position.size() / 3; v++)
      {
        const fv::Vec3 q = w.transformPoint({geo->position[v * 3], geo->position[v * 3 + 1], geo->position[v * 3 + 2]});
        const int64_t  b = nearest(q);
        if(b < 0)
        {
          far++;
          continue;
        }
        dist.push_back((q.x - bp[size_t(b)].x) * bn[size_t(b)].x + (q.y - bp[size_t(b)].y) * bn[size_t(b)].y + (q.z - bp[size_t(b)].z) * bn[size_t(b)].z);
      }
    }
    std::sort(dist.begin(), dist.end());
    const auto pct = [&](double q) { return dist.empty() ? 0.0 : r(dist[size_t(q * double(dist.size() - 1))]); };
    const auto under = [&](double depth) {
      return dist.empty() ? 0.0 : r(double(std::count_if(dist.begin(), dist.end(), [&](double d) { return d < -depth; })) / double(dist.size()));
    };
    items.push_back({{"label", item.label},
                     {"kind", item.kind},
                     {"vertices", dist.size()},
                     {"farFromBody", far},
                     {"underSkin0.1cm", under(0.1)},
                     {"underSkin0.5cm", under(0.5)},
                     {"p05", pct(0.05)},
                     {"median", pct(0.5)},
                     {"p95", pct(0.95)}});
  }
  return items;
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
    fv::timing::setSink([](const std::string& line) { std::cerr << line << "\n"; });
    auto store = std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{libraryDir}));
    std::unique_ptr<fv::Figure>         loaded;
    Json                                dumps = Json::object();
    int                                 stepNo = 0;
    const auto                          caseStart = std::chrono::steady_clock::now();
    for(const Json& step : c.at("steps"))
    {
      const auto stepStart = std::chrono::steady_clock::now();
      if(step.contains("load"))
        {
          loaded = fv::Figure::load(store, step["load"].get<std::string>(), {.subdivision = step.value("subdivision", 0)});
          loaded->settle();
        }
      else if(step.contains("morph"))
      {
        if(!loaded)
          throw std::runtime_error("morph before load");
        std::vector<std::pair<std::string, double>> values;
        for(const auto& [name, v] : step["morph"].items())
          values.emplace_back(name, v.get<double>());
        loaded->morphs->setMany(values);
        loaded->settle();
      }
      else if(step.contains("pose"))
      {
        if(!loaded || !loaded->pose)
          throw std::runtime_error("pose before a figure loads");
        // keys are "bone/kind/axis"
        std::vector<fv::pose::ChannelValue> values;
        for(const auto& [key, v] : step["pose"].items())
        {
          const size_t a = key.find('/'), b = key.rfind('/');
          fv::pose::ChannelKind kind;
          if(a == std::string::npos || a == b || !fv::pose::parseKind(key.substr(a + 1, b - a - 1), kind))
            throw std::runtime_error("bad pose key " + key);
          values.push_back({key.substr(0, a), kind, key.substr(b + 1), v.get<double>()});
        }
        loaded->pose->setMany(values);
        loaded->settle();
      }
      else if(step.contains("preset"))
      {
        if(!loaded)
          throw std::runtime_error("preset before a figure loads");
        auto                       doc = store->load(step["preset"].get<std::string>());
        const fv::PosePresetResult r   = loaded->applyPosePreset(doc->json);
        // as the host does: a property preset's surfaces too
        const fv::dson::MaterialPresetResult m = fv::dson::applyMaterialPreset(doc->json, loaded->loaded.scene);
        loaded->settle();
        std::cerr << "preset " << step["preset"].get<std::string>() << ": applied " << r.applied << ", skipped " << r.skipped.size()
                  << ", visibility " << r.visibility.size() << ", items shown or hidden " << r.itemsShown.size() << ", surfaces "
                  << m.changed.size() << "\n";
        for(const std::string& s : r.skipped)
          std::cerr << "  skipped " << s << "\n";
      }
      else if(step.contains("wear"))
      {
        if(!loaded)
          throw std::runtime_error("wear before a figure loads");
        const std::string rel = step["wear"].get<std::string>();
        auto              prepared = loaded->prepareWear(rel, step.value("subdivision", 0));
        // as the host does on its worker, so the build after the wear finds the strands measured
        if(step.value("measure", false))
        {
          fv::timing::Scope t("measureStrands");
          fv::build::measureStrands(fv::Figure::preparedScene(*prepared), false);
        }
        const fv::WearResult w = loaded->commitWear(prepared, rel);
        loaded->settle();
        Json report = {{"items", w.items}, {"fit", r(w.fit)}, {"notes", w.notes}, {"warnings", w.warnings}, {"missingBones", w.missingBones}};
        std::cerr << "wear " << rel << ": " << report.dump() << "\n";
      }
      else if(step.contains("remove"))
      {
        if(!loaded || !loaded->removeItem(step["remove"].get<int>()))
          throw std::runtime_error("no such item to remove");
        loaded->settle();
      }
      else if(step.contains("fit"))
      {
        if(!loaded)
          throw std::runtime_error("fit before a figure loads");
        dumps[step["fit"].get<std::string>()] = {{"fit", fitReport(*loaded)}};
      }
      else if(step.contains("dump"))
      {
        if(!loaded)
          throw std::runtime_error("dump before load");
        dumps[step["dump"].get<std::string>()] = dumpScene(loaded->loaded.scene, loaded->loaded.root, loaded->morphs.get());
      }
      else if(step.contains("build"))
      {
        if(!loaded)
          throw std::runtime_error("build before a figure loads");
        for(int k = 0; k < std::max(1, step["build"].get<int>()); k++)
        {
          fv::build::BuildResult built;
          {
            fv::timing::Scope t("buildGltf");
            built = fv::build::buildGltf(loaded->loaded.scene, loaded->loaded.root, {});
          }
          // the measured strand directions, to compare runs by
          std::cerr << "strands";
          for(size_t mi = 0; mi < built.lowering.size(); mi++)
            if(const auto& a = built.lowering[mi].anisotropyRotation)
              std::cerr << " " << loaded->loaded.scene.materials[mi].name << "=" << std::setprecision(6) << *a;
          std::cerr << std::defaultfloat << "\n";
        }
      }
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stepStart).count();
      std::cerr << "[step] " << ++stepNo << " " << step.begin().key() << " " << std::fixed << std::setprecision(1) << ms << " ms\n"
                << std::defaultfloat;
    }
    std::cerr << "[step] total " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - caseStart).count() << " ms\n"
              << std::defaultfloat;
    std::filesystem::create_directories(outPath.parent_path());
    std::ofstream o(outPath);
    o << Json{{"case", casePath.stem().u8string()}, {"dumps", dumps}}.dump();
    if(loaded)
      for(const auto& w : loaded->takeWarnings())
        std::cerr << "warning: " << w << "\n";
    return 0;
  }
  catch(const std::exception& e)
  {
    std::cerr << "fv_parity: " << e.what() << "\n";
    return 1;
  }
}
