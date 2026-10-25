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
// {"dforce": label} settles the worn dynamic surfaces at the pose the figure is in, on what the
// app's driver would hand the solver, and measures each drape: who moves it, how far its edges
// stretch, where it hangs against the skinned pose, and how much of it is through the body.
// "frames": n holds for n frames instead of until still; "trace": true prints every frame;
// "obj": dir writes the drapes and the posed body as OBJ files there (centimetres, figure space);
// "apply": true also runs the driver's own path (the settle on the pool, the drape written as a
// morph) and measures the drape read back through the skin, as the renderer gets it;
// "colliders": false leaves worn items without a simulation modifier out of the simulation.
//
// Every step's time goes to stderr ("[step] ..."), with the library's phase timers ("[fv-time] ...")
// inside it: the case in cases/native_test.json is the profile of a scene file's load.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <thread>
#include <unordered_map>

#include <figure/dforce/dforce.hpp>
#include <figure/dforce/skin.hpp>
#include <figure/dforce/solver.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/io/timing.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/collide.hpp>

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

// The stretch edges of one simulated mesh as a ratio of their rest length, sorted. An edge between
// two pinned vertices is the skin's doing, not the solver's, and is left out.
std::vector<double> stretchRatios(const fv::dforce::ClothModel& model, size_t slot, size_t n, const std::vector<float>& out)
{
  const int           base = model.base[slot];
  std::vector<double> ratio;
  for(const fv::dforce::Edge& e : model.edges)
  {
    if(e.kind != fv::dforce::EdgeKind::eStretch || e.rest <= 0 || e.a < base || e.b < base || e.a >= base + int(n) || e.b >= base + int(n))
      continue;
    if(model.inv[size_t(e.a)] <= 0 && model.inv[size_t(e.b)] <= 0)
      continue;
    const size_t a = size_t(e.a - base) * 3, b = size_t(e.b - base) * 3;
    const double dx = out[b] - out[a], dy = out[b + 1] - out[a + 1], dz = out[b + 2] - out[a + 2];
    ratio.push_back(std::sqrt(dx * dx + dy * dy + dz * dz) / e.rest);
  }
  std::sort(ratio.begin(), ratio.end());
  return ratio;
}

// How a simulated mesh came out, for a run to be judged without a viewport. `out` is the slot's
// positions in the job's space, `bodyNow` the body collider where the pose left it. A vertex is
// "inside" when the body's closest face has it behind, as the wardrobe collider counts it
// (pushOut's rule, within 3 cm); the signed distances are along that face's normal.
Json drapeReport(const fv::dforce::ClothModel& model, size_t slot, const fv::dforce::SettleMesh& mesh, const std::vector<float>& out,
                 const fv::dforce::SettleMesh& body, const std::vector<float>& bodyNow)
{
  const int    base   = model.base[slot];
  const size_t n      = mesh.vertexCount();
  const auto   inSlot = [&](int i) { return i >= base && i < base + int(n); };
  size_t       pinned = 0, soft = 0, free = 0;
  double       softSag = 0;
  for(int i : model.kinematicVerts)
    if(inSlot(i))
      pinned++;
  for(int i : model.freeVerts)
  {
    if(!inSlot(i))
      continue;
    if(model.strength[i] < 1 - 1e-3f)
    {
      soft++;
      const size_t v = size_t(i - base) * 3;
      const double dx = out[v] - mesh.posed[v], dy = out[v + 1] - mesh.posed[v + 1], dz = out[v + 2] - mesh.posed[v + 2];
      softSag += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    else
      free++;
  }
  const auto ratio = stretchRatios(model, slot, n, out);
  const auto pct   = [](const std::vector<double>& s, double q) { return s.empty() ? 0.0 : r(s[size_t(q * double(s.size() - 1))]); };
  const auto box   = [](const std::vector<float>& p) {
    Json lo = Json::array({1e9, 1e9, 1e9}), hi = Json::array({-1e9, -1e9, -1e9});
    for(size_t v = 0; v + 2 < p.size(); v += 3)
      for(size_t c = 0; c < 3; c++)
      {
        lo[c] = std::min(lo[c].get<double>(), double(p[v + c]));
        hi[c] = std::max(hi[c].get<double>(), double(p[v + c]));
      }
    for(size_t c = 0; c < 3; c++)
    {
      lo[c] = r(lo[c].get<double>());
      hi[c] = r(hi[c].get<double>());
    }
    return Json{{"min", lo}, {"max", hi}};
  };
  const fv::wardrobe::Surface        surface{bodyNow, body.tris};
  const fv::wardrobe::SurfaceBinding binding = fv::wardrobe::bindToSurface(bodyNow, body.tris, out);
  const std::vector<float>           pushed  = fv::wardrobe::pushOut(out, surface, binding, 0.f, 3.f);
  size_t                             inside  = 0;
  std::vector<double>                depth;
  for(size_t v = 0; v < n; v++)
  {
    const size_t k = v * 3;
    if(pushed[k] != out[k] || pushed[k + 1] != out[k + 1] || pushed[k + 2] != out[k + 2])
      inside++;
    const double w = double(binding.weights[k]) + binding.weights[k + 1] + binding.weights[k + 2];
    if(w <= 0)
      continue;
    const size_t ia = binding.verts[k], ib = binding.verts[k + 1], ic = binding.verts[k + 2];
    double       s[3], e1[3], e2[3];
    for(size_t c = 0; c < 3; c++)
    {
      s[c]  = bodyNow[ia * 3 + c] * binding.weights[k] + bodyNow[ib * 3 + c] * binding.weights[k + 1] + bodyNow[ic * 3 + c] * binding.weights[k + 2];
      e1[c] = bodyNow[ib * 3 + c] - bodyNow[ia * 3 + c];
      e2[c] = bodyNow[ic * 3 + c] - bodyNow[ia * 3 + c];
    }
    const double nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
    const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if(len < 1e-12)
      continue;
    depth.push_back(((out[k] - s[0]) * nx + (out[k + 1] - s[1]) * ny + (out[k + 2] - s[2]) * nz) / len);
  }
  std::sort(depth.begin(), depth.end());
  float minOut = 1e9f, minPosed = 1e9f;
  for(size_t v = 0; v < n; v++)
  {
    minOut   = std::min(minOut, out[v * 3 + 1]);
    minPosed = std::min(minPosed, mesh.posed[v * 3 + 1]);
  }
  return {{"vertices", n},
          {"pinned", pinned},
          {"soft", soft},
          {"free", free},
          {"softSagCm", r(soft ? softSag / double(soft) : 0.0)},
          {"stretchMax", pct(ratio, 1.0)},
          {"stretchP99", pct(ratio, 0.99)},
          {"stretchOver1.2", std::count_if(ratio.begin(), ratio.end(), [](double x) { return x > 1.2; })},
          {"insideBody", inside},
          {"bodyP05", pct(depth, 0.05)},
          {"bodyMedian", pct(depth, 0.5)},
          {"hemDropCm", r(double(minPosed - minOut))},
          {"box", box(out)},
          {"boxPosed", box(mesh.posed)}};
}

// A drape's per-vertex and per-edge detail beside its OBJ, for a script to find where it went
// wrong: each vertex's strength and signed depth against the body (null when it bound to no face),
// and every stretch edge with its rest length.
void writeSidecar(const std::filesystem::path& file, const fv::dforce::ClothModel& model, size_t slot, const fv::dforce::SettleMesh& mesh,
                  const std::vector<float>& out, const fv::dforce::SettleMesh& body, const std::vector<float>& bodyNow)
{
  const int    base = model.base[slot];
  const size_t n    = mesh.vertexCount();
  Json         strength = Json::array(), depth = Json::array(), edges = Json::array();
  const fv::wardrobe::SurfaceBinding binding = fv::wardrobe::bindToSurface(bodyNow, body.tris, out);
  for(size_t v = 0; v < n; v++)
  {
    strength.push_back(r(mesh.strength[v]));
    const size_t k = v * 3;
    const double w = double(binding.weights[k]) + binding.weights[k + 1] + binding.weights[k + 2];
    if(w <= 0)
    {
      depth.push_back(nullptr);
      continue;
    }
    const size_t ia = binding.verts[k], ib = binding.verts[k + 1], ic = binding.verts[k + 2];
    double       s[3], e1[3], e2[3];
    for(size_t c = 0; c < 3; c++)
    {
      s[c]  = bodyNow[ia * 3 + c] * binding.weights[k] + bodyNow[ib * 3 + c] * binding.weights[k + 1] + bodyNow[ic * 3 + c] * binding.weights[k + 2];
      e1[c] = bodyNow[ib * 3 + c] - bodyNow[ia * 3 + c];
      e2[c] = bodyNow[ic * 3 + c] - bodyNow[ia * 3 + c];
    }
    const double nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
    const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if(len < 1e-12)
      depth.push_back(nullptr);
    else
      depth.push_back(r(((out[k] - s[0]) * nx + (out[k + 1] - s[1]) * ny + (out[k + 2] - s[2]) * nz) / len));
  }
  for(const fv::dforce::Edge& e : model.edges)
    if(e.kind == fv::dforce::EdgeKind::eStretch && e.a >= base && e.b >= base && e.a < base + int(n) && e.b < base + int(n)
       && (model.inv[size_t(e.a)] > 0 || model.inv[size_t(e.b)] > 0))
      edges.push_back(Json::array({e.a - base, e.b - base, r(e.rest)}));
  std::ofstream o(file);
  o << Json{{"strength", strength}, {"depth", depth}, {"edges", edges}}.dump();
}

// A mesh as an OBJ: its polygons when it has them, else its triangles.
void writeObj(const std::filesystem::path& file, const std::vector<float>& pos, const fv::dforce::SettleMesh& mesh)
{
  std::filesystem::create_directories(file.parent_path());
  std::ofstream o(file);
  o << std::setprecision(6);
  for(size_t v = 0; v + 2 < pos.size(); v += 3)
    o << "v " << pos[v] << ' ' << pos[v + 1] << ' ' << pos[v + 2] << '\n';
  if(mesh.polyStart.size() > 1)
    for(size_t p = 0; p + 1 < mesh.polyStart.size(); p++)
    {
      o << 'f';
      for(int k = mesh.polyStart[p]; k < mesh.polyStart[p + 1]; k++)
        o << ' ' << mesh.polyVerts[size_t(k)] + 1;
      o << '\n';
    }
  else
    for(size_t t = 0; t + 2 < mesh.tris.size(); t += 3)
      o << "f " << mesh.tris[t] + 1 << ' ' << mesh.tris[t + 1] + 1 << ' ' << mesh.tris[t + 2] + 1 << '\n';
}

// A file name a label and a mesh name can be part of.
std::string fileSafe(std::string s)
{
  for(char& c : s)
    if(!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_')
      c = '_';
  return s;
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
      else if(step.contains("cloth"))
      {
        // {"cloth": {"item": label substring, "on": true | false | null, "strength": 1}}: dForce on
        // one worn item as the Wardrobe row sets it
        if(!loaded)
          throw std::runtime_error("cloth before a figure loads");
        const Json&         c    = step["cloth"];
        const std::string   what = c.value("item", "");
        std::optional<bool> on;
        if(c.contains("on") && c["on"].is_boolean())
          on = c["on"].get<bool>();
        int found = 0;
        for(const fv::WornItem& w : loaded->items)
          if(w.label.find(what) != std::string::npos)
          {
            const auto changed = loaded->setItemCloth(w.id, on, c.value("strength", 1.f));
            const auto state   = loaded->clothOf(w);
            std::cerr << "cloth " << w.label << ": " << changed.size() << " mesh(es) changed, able " << state.able << " native " << state.native
                      << " on " << state.on << " strength " << state.strength << "\n";
            found++;
          }
        if(!found)
          throw std::runtime_error("no worn item matches \"" + what + "\"");
      }
      else if(step.contains("fit"))
      {
        if(!loaded)
          throw std::runtime_error("fit before a figure loads");
        dumps[step["fit"].get<std::string>()] = {{"fit", fitReport(*loaded)}};
      }
      else if(step.contains("dforce"))
      {
        if(!loaded)
          throw std::runtime_error("dforce before a figure loads");
        const auto& ctx = loaded->loaded.morphs;
        if(ctx.mainMeshes.empty())
          throw std::runtime_error("dforce without a figure body");
        const std::string label = step["dforce"].get<std::string>();
        const int         body  = ctx.meshes[ctx.mainMeshes[0]].mesh;
        fv::SceneGraph&   scene = loaded->loaded.scene;
        fv::dforce::Dforce d(scene, *loaded->morphs, body, loaded->loaded.root);
        // as the host sets it: the figure's shown clothing collides, its anatomy and hair do not
        if(step.value("colliders", true))
          d.collidesWith = [fig = loaded.get()](int mesh) {
            for(const fv::WornItem& w : fig->items)
              if(w.visible && w.kind == "Clothing" && std::find(w.meshes.begin(), w.meshes.end(), mesh) != w.meshes.end())
                return true;
            return false;
          };
        else
          d.collidesWith = [](int) { return false; };
        auto p = d.prepared();
        if(!p)
          throw std::runtime_error("nothing worn drapes");
        // experiments: the cloth's own faces left out of its collisions
        if(!step.value("selfCollide", true))
          for(fv::dforce::SettleMesh& m : p->bodies)
            std::fill(m.selfCollide.begin(), m.selfCollide.end(), uint8_t(0));
        if(p->bodies.empty() || p->bodies[0].kind != fv::dforce::SettleMesh::Kind::eCollider)
          throw std::runtime_error("the body is not the first collider");
        const std::optional<std::filesystem::path> objDir =
            step.contains("obj") ? std::optional<std::filesystem::path>(root / pathOf(step["obj"].get<std::string>().c_str())) : std::nullopt;
        const auto meshName = [&](int mesh) { return fileSafe(scene.meshes[size_t(mesh)].name); };
        fv::dforce::SettleOptions opts;
        opts.air        = 0.15f;
        opts.iterations    = step.value("iterations", opts.iterations);
        opts.stretchSweeps = step.value("stretchSweeps", opts.stretchSweeps);
        opts.substeps      = step.value("substeps", opts.substeps);
        const auto           simStart = std::chrono::steady_clock::now();
        fv::dforce::ClothSim sim(p->bodies, opts);
        const bool           trace  = step.value("trace", false);
        const int            frames = step.value("frames", 0);
        if(frames > 0 || trace)
        {
          // the settle spelled out, so each frame can be reported: the transition, then the hold
          // with its extra damping until the cloth is still
          const int transition = std::max(1, int(std::lround(sim.transitionSeconds() / opts.frameDt)));
          const int hold       = frames > 0 ? frames : int(std::ceil(opts.maxHold / opts.frameDt));
          auto      before     = sim.positions();
          for(int f = 0; f < transition + hold; f++)
          {
            if(f < transition)
              sim.frame(float(f) / float(transition), float(f + 1) / float(transition));
            else
              sim.frame(1, 1, 6.f);
            if(trace)
            {
              // the frame's largest move and how many vertices moved over a centimetre, per drape:
              // a hold that will not settle shows where it is fighting
              const auto out = sim.positions();
              std::cerr << "[dforce] " << label << " frame " << f << (f < transition ? " transition" : " hold") << " move " << std::setprecision(4)
                        << sim.lastMove() << " recovered " << sim.recovered();
              for(const auto& [mesh, slot] : p->drapes)
              {
                const auto&  sm    = p->bodies[size_t(slot)];
                const auto   ratio = stretchRatios(sim.model(), size_t(slot), sm.vertexCount(), out[size_t(slot)]);
                const auto&  now   = out[size_t(slot)];
                const auto&  was   = before[size_t(slot)];
                size_t       most = 0, over = 0;
                double       mostD = 0;
                for(size_t v = 0; v < sm.vertexCount(); v++)
                {
                  const double d = std::hypot(now[v * 3] - was[v * 3], now[v * 3 + 1] - was[v * 3 + 1], now[v * 3 + 2] - was[v * 3 + 2]);
                  if(d > 1)
                    over++;
                  if(d > mostD)
                    mostD = d, most = v;
                }
                std::cerr << " | " << meshName(mesh) << " stretch " << (ratio.empty() ? 0.0 : ratio.back()) << " moved>1cm " << over << " most v" << most
                          << " " << mostD << " at y " << now[most * 3 + 1] << " s " << sm.strength[most];
              }
              std::cerr << std::defaultfloat << "\n";
              before = std::move(out);
            }
            if(f >= transition && sim.lastMove() < opts.restMove)
              break;
          }
        }
        else
          sim.settle();
        const double simMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - simStart).count();
        const auto   out   = sim.positions();
        Json         items = Json::array();
        for(const auto& [mesh, slot] : p->drapes)
        {
          Json report    = drapeReport(sim.model(), size_t(slot), p->bodies[size_t(slot)], out[size_t(slot)], p->bodies[0], out[0]);
          report["mesh"] = scene.meshes[size_t(mesh)].name;
          items.push_back(report);
          if(objDir)
          {
            const fv::dforce::SettleMesh& sm = p->bodies[size_t(slot)];
            writeObj(*objDir / (fileSafe(label) + "_" + meshName(mesh) + ".obj"), out[size_t(slot)], sm);
            writeObj(*objDir / (fileSafe(label) + "_" + meshName(mesh) + "_rest.obj"), sm.rest, sm);
            writeObj(*objDir / (fileSafe(label) + "_" + meshName(mesh) + "_posed.obj"), sm.posed, sm);
            writeSidecar(*objDir / (fileSafe(label) + "_" + meshName(mesh) + ".json"), sim.model(), size_t(slot), sm, out[size_t(slot)], p->bodies[0], out[0]);
          }
        }
        if(objDir)
          writeObj(*objDir / (fileSafe(label) + "_body.obj"), out[0], p->bodies[0]);
        std::cerr << "dforce " << label << ": " << items.dump() << "\n";
        Json dump = {{"dforce", items}, {"substeps", sim.substepsPerFrame()}, {"transition", r(sim.transitionSeconds())}, {"ms", r(simMs)}};
        if(step.value("apply", false))
        {
          // the driver's own path: the settle on the pool, the drape written as a morph, and the
          // cage read back through the skin as the renderer skins it
          d.resimulate();
          bool wrote = false;
          for(int i = 0; i < 600000 && !wrote; i++)
          {
            wrote = d.pump();
            if(!wrote)
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
          if(!wrote)
            throw std::runtime_error("the driver wrote no drape");
          loaded->morphs->flush();
          const auto     worlds    = scene.worlds();
          const auto     nodeOf    = fv::dforce::nodeOfMesh(scene);
          const fv::Mat4 fromWorld = inverse(worlds[size_t(loaded->loaded.root)]);
          Json           applied   = Json::array();
          for(const auto& [mesh, slot] : p->drapes)
          {
            const fv::Mesh&          m    = scene.meshes[size_t(mesh)];
            const std::vector<float> cage = loaded->morphs->morphedCage(mesh);
            const std::vector<float> posed =
                m.skinned() ? fv::dforce::skinLocal(cage, fv::dforce::top4(m.cageSkin), fv::dforce::currentSkin(scene, worlds, nodeOf, mesh)) : cage;
            const fv::Mat4 to = fromWorld * (nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : fv::Mat4{});
            const auto     inSpace = fv::dforce::transformPoints(to, posed);
            Json           report  = drapeReport(sim.model(), size_t(slot), p->bodies[size_t(slot)], inSpace, p->bodies[0], out[0]);
            report["mesh"]         = m.name;
            applied.push_back(report);
            if(objDir)
              writeObj(*objDir / (fileSafe(label) + "_applied_" + meshName(mesh) + ".obj"), inSpace, p->bodies[size_t(slot)]);
          }
          std::cerr << "dforce " << label << " applied: " << applied.dump() << "\n";
          dump["applied"] = applied;
        }
        dumps[label] = dump;
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
