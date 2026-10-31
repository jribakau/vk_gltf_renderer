// Runs a case (cases/*.json; the app repository's parity cases before the TypeScript side was
// deleted) through the figure library, with no GPU, and writes what it built: bones, meshes and morph
// weights in the shape the TypeScript dumper wrote, so two runs can be diffed.
//
//   fv_parity <case.json> <out.json> [--root <dir for relative library paths>] [--cache <parsed-file cache dir>]
//             [--bind-cache <surface-binding cache dir>]
//
// Steps: {"load"}, {"morph"}, {"pose"}, {"preset"}, {"dump"}, and for the wardrobe {"wear": rel},
// {"remove": item id} and {"fit": label}, which measures how the worn items sit on the body: for
// each item vertex, its signed distance to the nearest body vertex along that vertex's normal.
// {"build": n} builds the glTF the renderer would get (without textures) n times, as the host
// builds it again after each wardrobe change, and prints the hair strand directions it measured;
// {"wear": rel, "measure": true} measures a wearable's strands ahead, as the host's wear does.
// {"feed": n} feeds the figure into a fresh retained scene n times (src/fv_figure_feed, with the
// build's options), and times a second sync with nothing changed, the one every frame would make.
// Each build and each feed is read back against the scene (checkBuild, checkFeed).
// {"surfaces": label} records each material's relief maps and the surface they cover.
// {"sweep": label} measures every item in the library's materials, on no figure of the case's
// (material_sweep.hpp; cases/material_sweep.json, `pnpm library:materials`).
// {"cloth": {"item", "on", "strength"}} turns dForce on or off for the worn items it names, as the
// Wardrobe row does.
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
#include <set>
#include <thread>
#include <unordered_map>

#include <figure/dson/store.hpp>
#include <figure/dforce/dforce.hpp>
#include <figure/dforce/skin.hpp>
#include <figure/dforce/solver.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/gltf/gltf_check.hpp>
#include <figure/io/timing.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/bind_cache.hpp>
#include <figure/wardrobe/collider.hpp>
#include <figure/wardrobe/fit/collide.hpp>

#include "fv_figure_feed.hpp"
#include "material_sweep.hpp"
#include "fv_figure_feed_check.hpp"
#include "scene_retained.hpp"

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
  const std::vector<int> nodeOf = fv::dforce::nodeOfMesh(scene);
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

// How many vertices lie behind the body's closest face (pushOut's rule, within 3 cm), and how deep.
Json insideReport(std::span<const float> body, std::span<const uint32_t> bodyTris, std::span<const float> item)
{
  const fv::wardrobe::Surface        surface{body, bodyTris};
  const fv::wardrobe::SurfaceBinding binding = fv::wardrobe::bindToSurface(body, bodyTris, item);
  const std::vector<float>           pushed  = fv::wardrobe::pushOut(item, surface, binding, 0.f, 3.f);
  std::vector<double>                depth;
  for(size_t k = 0; k + 2 < item.size(); k += 3)
    if(pushed[k] != item[k] || pushed[k + 1] != item[k + 1] || pushed[k + 2] != item[k + 2])
      depth.push_back(std::hypot(pushed[k] - item[k], pushed[k + 1] - item[k + 1], pushed[k + 2] - item[k + 2]));
  std::sort(depth.begin(), depth.end());
  return {{"vertices", item.size() / 3}, {"count", depth.size()}, {"p50", depth.empty() ? 0.0 : r(depth[depth.size() / 2])}, {"max", depth.empty() ? 0.0 : r(depth.back())}};
}

// How far each cage vertex of a mesh stands from the closest body point: the percentiles and the
// counts past a few distances, over the whole mesh and over its open edges (hems).
Json standoffReport(std::span<const float> body, std::span<const uint32_t> bodyTris, std::span<const float> item, std::span<const uint32_t> tris)
{
  const fv::wardrobe::SurfaceBinding binding = fv::wardrobe::bindToSurface(body, bodyTris, item);
  const size_t                       n       = item.size() / 3;
  std::vector<double>                dist(n, 0);
  for(size_t v = 0; v < n; v++)
  {
    const size_t k = v * 3;
    double       d2 = 0;
    for(size_t c = 0; c < 3; c++)
    {
      const double s = double(body[size_t(binding.verts[k]) * 3 + c]) * binding.weights[k] + double(body[size_t(binding.verts[k + 1]) * 3 + c]) * binding.weights[k + 1]
                       + double(body[size_t(binding.verts[k + 2]) * 3 + c]) * binding.weights[k + 2];
      d2 += (item[k + c] - s) * (item[k + c] - s);
    }
    dist[v] = std::sqrt(d2);
  }
  std::map<std::pair<uint32_t, uint32_t>, int> edges;
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
    for(int e = 0; e < 3; e++)
    {
      const uint32_t a = tris[t + size_t(e)], b = tris[t + size_t(e + 1) % 3];
      edges[{std::min(a, b), std::max(a, b)}]++;
    }
  std::vector<bool> hem(n, false);
  for(const auto& [edge, count] : edges)
    if(count == 1)
      hem[edge.first] = hem[edge.second] = true;
  const auto stats = [&](bool hemOnly) {
    std::vector<double> s;
    for(size_t v = 0; v < n; v++)
      if(!hemOnly || hem[v])
        s.push_back(dist[v]);
    std::sort(s.begin(), s.end());
    const auto pct  = [&](double q) { return s.empty() ? 0.0 : r(s[size_t(q * double(s.size() - 1))]); };
    const auto over = [&](double d) { return std::count_if(s.begin(), s.end(), [d](double x) { return x > d; }); };
    return Json{{"vertices", s.size()}, {"p50", pct(0.5)}, {"p90", pct(0.9)}, {"p99", pct(0.99)}, {"max", pct(1)},
                {"over0.1", over(0.1)}, {"over0.5", over(0.5)}, {"over1", over(1)}, {"over2", over(2)}, {"over6", over(6)}};
  };
  // where the farthest of the fabric within a hand's width is: what still shows as a gap
  std::vector<size_t> order;
  for(size_t v = 0; v < n; v++)
    if(dist[v] > 0.5 && dist[v] < 8)
      order.push_back(v);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return dist[a] > dist[b]; });
  Json far = Json::array();
  for(size_t i = 0; i < order.size() && i < 12; i++)
    far.push_back({{"v", order[i]}, {"d", r(dist[order[i]])}, {"hem", bool(hem[order[i]])}, {"at", {r(item[order[i] * 3]), r(item[order[i] * 3 + 1]), r(item[order[i] * 3 + 2])}}});
  return {{"all", stats(false)}, {"hem", stats(true)}, {"farthest", far}, {"inside", insideReport(body, bodyTris, item)}};
}

// How much of each strand of the worn items whose label holds `what` lies under the body's skin, at
// rest with the figure's morphs: the buried run from the root, in cm, and the share of points under
// it by how far along the strand they are.
Json strandsInsideReport(fv::Figure& fig, const std::string& what)
{
  const int                    body     = fig.loaded.morphs.meshes[fig.loaded.morphs.mainMeshes[0]].mesh;
  const auto                   skin     = fig.morphs->morphedCage(body);
  const std::vector<uint32_t>& bodyTris = fig.loaded.scene.meshes[size_t(body)].geo->dsonTris;
  const fv::wardrobe::Surface  surface{skin, bodyTris};
  Json                         report = Json::object();
  for(const fv::WornItem& w : fig.items)
  {
    if(w.label.find(what) == std::string::npos)
      continue;
    for(int mesh : w.meshes)
    {
      const fv::mesh::BuiltMesh& geo = *fig.loaded.scene.meshes[size_t(mesh)].geo;
      if(geo.cageLineStart.size() < 2)
        continue;
      const auto                         item    = fig.morphs->morphedCage(mesh);
      const fv::wardrobe::SurfaceBinding binding = fv::wardrobe::bindToSurface(skin, bodyTris, item);
      const std::vector<float>           pushed  = fv::wardrobe::pushOut(item, surface, binding, 0.f, 3.f);
      const auto under = [&](int v) { return pushed[size_t(v) * 3] != item[size_t(v) * 3] || pushed[size_t(v) * 3 + 1] != item[size_t(v) * 3 + 1] || pushed[size_t(v) * 3 + 2] != item[size_t(v) * 3 + 2]; };
      std::vector<double> runs;
      std::array<double, 4> points{}, hidden{};
      int rootsUnder = 0;
      for(size_t l = 0; l + 1 < geo.cageLineStart.size(); l++)
      {
        const int s = geo.cageLineStart[l], e = geo.cageLineStart[l + 1];
        double    run = 0;
        bool      buried = true;
        rootsUnder += e > s && under(geo.cageLineVerts[size_t(s)]);
        for(int c = s; c < e; c++)
        {
          const int    v  = geo.cageLineVerts[size_t(c)];
          const double at = e - s > 1 ? double(c - s) / (e - s - 1) : 0;
          const size_t bin = at < 0.02 ? 0 : at < 0.05 ? 1 : at < 0.1 ? 2 : 3;
          points[bin]++;
          hidden[bin] += under(v);
          buried = buried && under(v);
          if(buried && c + 1 < e)
          {
            const int u = geo.cageLineVerts[size_t(c) + 1];
            run += std::hypot(item[size_t(u) * 3] - item[size_t(v) * 3], item[size_t(u) * 3 + 1] - item[size_t(v) * 3 + 1], item[size_t(u) * 3 + 2] - item[size_t(v) * 3 + 2]);
          }
        }
        runs.push_back(run);
      }
      std::sort(runs.begin(), runs.end());
      const auto past = [&](double cm) { return double(runs.end() - std::upper_bound(runs.begin(), runs.end(), cm)) / double(runs.size()); };
      report[w.label] = {{"strands", runs.size()},
                         {"rootsUnderSkin", rootsUnder},
                         {"buriedRunCm", {{"p50", r(runs[runs.size() / 2])}, {"p90", r(runs[runs.size() * 9 / 10])}, {"max", r(runs.back())}}},
                         {"shareBuriedPast", {{"0.5cm", r(past(0.5))}, {"1cm", r(past(1))}, {"2cm", r(past(2))}, {"4cm", r(past(4))}}},
                         {"pointsUnderSkinAlong", {{"first 2%", r(hidden[0] / points[0])}, {"2-5%", r(hidden[1] / points[1])}, {"5-10%", r(hidden[2] / points[2])}, {"rest", r(hidden[3] / points[3])}}}};
    }
  }
  return report;
}

// Skin tight on the worn items whose label holds `what`, through the real collider: the standoff of
// each of their meshes before and after, and how long a pass took on the owning thread and in all.
Json tightReport(fv::Figure& fig, const std::string& what, float amount)
{
  const int body = fig.loaded.morphs.meshes[fig.loaded.morphs.mainMeshes[0]].mesh;
  fv::wardrobe::Collider collider(fig.loaded.scene, *fig.morphs, body, fig.loaded.root);
  collider.dazSkinning = true;
  collider.settings = [&fig] {
    std::map<int, fv::dson::Smoothing> all;
    for(const fv::WornItem& w : fig.items)
      if(w.visible)
        for(auto [mesh, smoothing] : w.collision)
        {
          smoothing.tight = w.tight;
          all.emplace(mesh, smoothing);
        }
    return all;
  };
  const auto pass = [&] {
    collider.poke();
    double     owner = 0;
    const auto start = std::chrono::steady_clock::now();
    while(collider.busy())
    {
      const auto t0 = std::chrono::steady_clock::now();
      collider.pump();
      owner += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    fig.settle();
    return Json{{"ownerMs", r(owner)}, {"allMs", r(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count())}};
  };
  const std::vector<uint32_t>& bodyTris = fig.loaded.scene.meshes[size_t(body)].geo->dsonTris;
  Json                         report   = Json::object();
  report["passBefore"]                  = pass();
  std::vector<int> meshes;
  for(const fv::WornItem& w : fig.items)
    if(w.label.find(what) != std::string::npos)
    {
      for(const auto& [mesh, smoothing] : w.collision)
        meshes.push_back(mesh);
      report["items"].push_back({{"label", w.label}, {"meshes", w.meshes}, {"collision", w.collision.size()}});
    }
  if(meshes.empty())
    throw std::runtime_error("no worn item with collision matches \"" + what + "\"");
  // as the collider sees them: the cage skinned to the pose, in the figure's space
  const auto posedCage = [&](int mesh) {
    const fv::SceneGraph& scene  = fig.loaded.scene;
    const auto            worlds = scene.worlds();
    const auto            nodeOf = fv::dforce::nodeOfMesh(scene);
    const fv::Mat4        to     = fv::inverse(worlds[size_t(fig.loaded.root)]) * worlds[size_t(nodeOf[size_t(mesh)])];
    const auto            rest   = fig.morphs->morphedCage(mesh);
    const fv::Mesh&       m      = scene.meshes[size_t(mesh)];
    if(!m.skinned() || m.cageSkin.start.size() != rest.size() / 3 + 1)
      return fv::dforce::transformPoints(to, rest);
    return fv::dforce::transformPoints(to, fv::dforce::skinLocal(rest, fv::dforce::cageSkin(m.cageSkin, true), fv::dforce::currentSkin(scene, worlds, nodeOf, mesh)));
  };
  const auto measure = [&](const char* name) {
    const auto bodyNow = posedCage(body);
    for(int mesh : meshes)
      report[name][std::to_string(mesh)] = standoffReport(bodyNow, bodyTris, posedCage(mesh), fig.loaded.scene.meshes[size_t(mesh)].geo->dsonTris);
  };
  // the drawn vertices, subdivided, unposed: what shows through at the rest pose
  const auto drawn = [&](const char* name) {
    const fv::mesh::BuiltMesh& b = *fig.loaded.scene.meshes[size_t(body)].geo;
    for(int mesh : meshes)
      report[name][std::to_string(mesh)] = insideReport(b.position, b.fullIndex, fig.loaded.scene.meshes[size_t(mesh)].geo->position);
  };
  measure("before");
  drawn("drawnBefore");
  for(const fv::WornItem& w : fig.items)
    if(w.label.find(what) != std::string::npos)
      fig.setItemTight(w.id, amount);
  report["passAfter"] = pass();
  measure("after");
  drawn("drawnAfter");
  const auto hideStart = std::chrono::steady_clock::now();
  fig.refreshTightHidden(true);
  report["hideMs"] = r(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hideStart).count());
  // of the fabric still under the skin as posed, how much lies over skin that's still drawn
  const fv::mesh::BuiltMesh& bodyGeo = *fig.loaded.scene.meshes[size_t(body)].geo;
  std::set<int32_t>          hiddenPolys;
  for(const auto& [owner, polys] : bodyGeo.hiddenPolys)
    hiddenPolys.insert(polys.begin(), polys.end());
  report["hiddenPolys"] = hiddenPolys.size();
  report["bodyPolys"]   = bodyGeo.cagePolyStart.size() - 1;
  std::vector<uint8_t> shown(bodyGeo.cagePositions.size() / 3, 0);
  for(size_t p = 0; p + 1 < bodyGeo.cagePolyStart.size(); p++)
    if(!hiddenPolys.count(int32_t(p)))
      for(int32_t c = bodyGeo.cagePolyStart[p]; c < bodyGeo.cagePolyStart[p + 1]; c++)
        shown[size_t(bodyGeo.cagePolyVerts[size_t(c)])] = 1;
  const auto bodyNow = posedCage(body);
  for(int mesh : meshes)
  {
    const auto                         cage    = posedCage(mesh);
    const fv::wardrobe::Surface        surface{bodyNow, bodyTris};
    const fv::wardrobe::SurfaceBinding binding = fv::wardrobe::bindToSurface(bodyNow, bodyTris, cage);
    const std::vector<float>           pushed  = fv::wardrobe::pushOut(cage, surface, binding, 0.f, 3.f);
    Json                               showing = Json::array();
    size_t                             count   = 0;
    for(size_t k = 0; k + 2 < cage.size(); k += 3)
    {
      if(pushed[k] == cage[k] && pushed[k + 1] == cage[k + 1] && pushed[k + 2] == cage[k + 2])
        continue;
      if(!shown[binding.verts[k]] && !shown[binding.verts[k + 1]] && !shown[binding.verts[k + 2]])
        continue;
      count++;
      if(showing.size() < 40)
        showing.push_back({{"depth", r(std::hypot(pushed[k] - cage[k], pushed[k + 1] - cage[k + 1], pushed[k + 2] - cage[k + 2]))}, {"at", {r(cage[k]), r(cage[k + 1]), r(cage[k + 2])}}});
    }
    report["stillShowing"][std::to_string(mesh)] = {{"count", count}, {"first", showing}};
  }
  return report;
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
    std::cerr << "usage: fv_parity <case.json> <out.json> [--root <dir>] [--cache <dir>]\n";
    return 2;
  }
  const std::filesystem::path casePath = pathOf(argv[1]);
  const std::filesystem::path outPath  = pathOf(argv[2]);
  std::filesystem::path       root     = std::filesystem::current_path();
  for(int i = 3; i + 1 < argc; i++)
  {
    if(std::string(argv[i]) == "--root")
      root = pathOf(argv[i + 1]);
    if(std::string(argv[i]) == "--cache")  // the parsed-file cache, as the renderer's --cacheDir/dson
      fv::dson::setParsedCacheDir(pathOf(argv[i + 1]));
    if(std::string(argv[i]) == "--bind-cache")  // the surface-binding cache, as the renderer's --cacheDir/bind
      fv::wardrobe::setBindCacheDir(pathOf(argv[i + 1]));
  }
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
      for(const std::string& line : fv::timing::takeSummary())
        std::cerr << line << "\n";
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
        const fv::dson::MaterialPresetResult m = fv::dson::applyMaterialPreset(doc->json, loaded->loaded.scene, nullptr, fv::dson::definitionsIn(doc->json, *store));
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
      else if(step.contains("tight"))
      {
        // {"tight": {"item": label substring, "amount": 1, "dump": name}}: the wardrobe collider's skin
        // tight on one worn item, with how far its cage stood from the body before and after
        if(!loaded)
          throw std::runtime_error("tight before a figure loads");
        const Json& t = step["tight"];
        dumps[t.value("dump", std::string("tight"))] = tightReport(*loaded, t.value("item", ""), t.value("amount", 1.f));
      }
      else if(step.contains("strandsInside"))
      {
        // {"strandsInside": label substring, "dump": name}: how much of a worn hair's strands is under the skin
        if(!loaded)
          throw std::runtime_error("strandsInside before a figure loads");
        dumps[step.value("dump", std::string("strandsInside"))] = strandsInsideReport(*loaded, step["strandsInside"].get<std::string>());
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
        // strand hair: where it ends up against where it started, and a sample of it to look at
        if(!p->hairs.empty())
        {
          Json hairs = Json::array();
          for(const auto& [mesh, strands] : p->hairs)
          {
            fv::dforce::StrandOptions hairOpts;
            hairOpts.spacing       = step.value("spacing", hairOpts.spacing);
            hairOpts.maxFrames     = step.value("frames", hairOpts.maxFrames);
            hairOpts.substeps      = step.value("substeps", hairOpts.substeps);
            size_t colliderTris    = 0;
            for(const auto& b : p->bodies)
              colliderTris += b.tris.size() / 3;
            std::cout << "hair colliders: " << p->bodies.size() << " meshes, " << colliderTris << " tris" << std::endl;
            const auto  hairStart  = std::chrono::steady_clock::now();
            const auto  after      = fv::dforce::settleStrands(strands, p->bodies, hairOpts);
            const double hairMs    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hairStart).count();
            // what landing it costs: the move as the mesh's render target, every drawn hair
            fv::mesh::CageDeltas landed;
            for(size_t v = 0; v * 3 + 2 < after.size(); v++)
            {
              landed.verts.push_back(int32_t(v));
              landed.deltas.insert(landed.deltas.end(), {after[v * 3] - strands.posed[v * 3], after[v * 3 + 1] - strands.posed[v * 3 + 1], after[v * 3 + 2] - strands.posed[v * 3 + 2]});
            }
            const auto  landStart  = std::chrono::steady_clock::now();
            const auto  landedTarget = fv::mesh::target(*scene.meshes[size_t(mesh)].geo, landed);
            const double landMs    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - landStart).count();
            std::cout << "hair lands: " << landMs << " ms, " << scene.meshes[size_t(mesh)].geo->position.size() / 3 << " drawn points" << std::endl;
            const auto  point      = [](const std::vector<float>& a, int v) { return std::array<float, 3>{a[size_t(v) * 3], a[size_t(v) * 3 + 1], a[size_t(v) * 3 + 2]}; };
            const auto  lengthOf   = [&](const std::vector<float>& a, size_t s) {
              double l = 0;
              for(int k = strands.start[s]; k + 1 < strands.start[s + 1]; k++)
              {
                const auto u = point(a, strands.verts[size_t(k)]), w = point(a, strands.verts[size_t(k) + 1]);
                l += std::sqrt(double((u[0] - w[0]) * (u[0] - w[0]) + (u[1] - w[1]) * (u[1] - w[1]) + (u[2] - w[2]) * (u[2] - w[2])));
              }
              return l;
            };
            std::vector<double> stretch, tips;
            int                 hanging = 0;
            const size_t        count   = strands.start.size() - 1;
            Json                sample  = Json::array();
            const int           every   = std::max(1, step.value("sampleEvery", 8));
            for(size_t s = 0; s < count; s++)
            {
              const double was = lengthOf(strands.posed, s);
              if(was > 0)
                stretch.push_back(lengthOf(after, s) / was);
              const auto root = point(after, strands.verts[size_t(strands.start[s])]), tip = point(after, strands.verts[size_t(strands.start[s + 1]) - 1]);
              const float dx = tip[0] - root[0], dy = tip[1] - root[1], dz = tip[2] - root[2];
              const float chord = std::sqrt(dx * dx + dy * dy + dz * dz);
              if(chord > 0 && dy / chord < -0.5f)
                hanging++;
              tips.push_back(tip[1]);
              if(s % size_t(every) == 0)
              {
                Json line = Json::array();
                for(int k = strands.start[s]; k < strands.start[s + 1]; k += 4)
                  for(float c : point(after, strands.verts[size_t(k)]))
                    line.push_back(c);
                sample.push_back(std::move(line));
              }
            }
            std::sort(stretch.begin(), stretch.end());
            std::sort(tips.begin(), tips.end());
            Json body = Json::array();
            for(size_t v = 0; v * 3 + 2 < p->bodies[0].posed.size(); v += 7)
              for(int c = 0; c < 3; c++)
                body.push_back(p->bodies[0].posed[v * 3 + size_t(c)]);
            hairs.push_back({{"mesh", fileSafe(scene.meshes[size_t(mesh)].name)}, {"strands", count}, {"points", strands.posed.size() / 3}, {"ms", hairMs}, {"hangingShare", double(hanging) / double(count)},
                             {"stretchMin", stretch.front()}, {"stretchMax", stretch.back()}, {"tipYMedian", tips[tips.size() / 2]}, {"tipYLowest", tips.front()}, {"tipYHighest", tips.back()}});
            if(step.contains("hairDump"))
            {
              std::ofstream dump(root / pathOf(step["hairDump"].get<std::string>().c_str()));
              dump << Json{{"strands", sample}, {"body", body}}.dump();
            }
          }
          std::cout << "hair " << label << ": " << hairs.dump() << std::endl;
          // as the app has it: picked up, solved on the pool and landed, a frame's call at a time
          if(step.value("pump", false))
          {
            d.resimulate();
            const auto pumpStart = std::chrono::steady_clock::now();
            double     longest = 0;
            bool       wrote = false;
            int        calls = 0;
            while(!wrote && std::chrono::steady_clock::now() - pumpStart < std::chrono::seconds(60))
            {
              const auto callStart = std::chrono::steady_clock::now();
              wrote = d.pump();
              longest = std::max(longest, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - callStart).count());
              calls++;
              if(!wrote)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "hair pump: wrote " << wrote << " in " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pumpStart).count() << " ms, longest call " << longest << " ms over " << calls << " calls" << std::endl;
          }
          if(p->drapes.empty())
            continue;
        }
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
        opts.transition    = step.value("transition", opts.transition);
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
                m.skinned() ? fv::dforce::skinLocal(cage, fv::dforce::cageSkin(m.cageSkin), fv::dforce::currentSkin(scene, worlds, nodeOf, mesh)) : cage;
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
      else if(step.contains("sweep"))
      {
        dumps[step["sweep"].get<std::string>()] = fv::probe::materialSweep(libraryDir, root, step);
      }
      else if(step.contains("surfaces"))
      {
        // {"surfaces": label}: each material's relief maps as the figure loaded them (normal map and
        // strength, bump map and strength, PBRSkin's detail layer), with the surface it covers in
        // figure units and in UV units, so a texel's size on the body can be told from the maps
        if(!loaded)
          throw std::runtime_error("surfaces before a figure loads");
        const fv::SceneGraph& scene = loaded->loaded.scene;
        std::vector<double>   area(scene.materials.size(), 0.0), uvArea(scene.materials.size(), 0.0);
        for(const fv::Mesh& m : scene.meshes)
        {
          if(!m.geo)
            continue;
          const auto& g = *m.geo;
          for(const auto& group : g.groups)
          {
            const int mi = group.materialIndex < int(m.materials.size()) ? m.materials[size_t(group.materialIndex)] : -1;
            if(mi < 0 || mi >= int(scene.materials.size()) || g.uv.empty())
              continue;
            for(uint32_t t = group.start; t + 2 < group.start + group.count; t += 3)
            {
              const uint32_t a = g.fullIndex[t], b = g.fullIndex[t + 1], c = g.fullIndex[t + 2];
              const auto     p = [&](uint32_t v, int k) { return double(g.position[size_t(v) * 3 + size_t(k)]); };
              const auto     q = [&](uint32_t v, int k) { return double(g.uv[size_t(v) * 2 + size_t(k)]); };
              const double   e1[3] = {p(b, 0) - p(a, 0), p(b, 1) - p(a, 1), p(b, 2) - p(a, 2)};
              const double   e2[3] = {p(c, 0) - p(a, 0), p(c, 1) - p(a, 1), p(c, 2) - p(a, 2)};
              const double   n[3]  = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
              area[size_t(mi)] += 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
              uvArea[size_t(mi)] += 0.5 * std::abs((q(b, 0) - q(a, 0)) * (q(c, 1) - q(a, 1)) - (q(c, 0) - q(a, 0)) * (q(b, 1) - q(a, 1)));
            }
          }
        }
        Json list = Json::array();
        for(size_t mi = 0; mi < scene.materials.size(); mi++)
        {
          const fv::Material& mat = scene.materials[mi];
          const auto&         s   = mat.surface;
          Json                e   = {{"name", mat.name},
                                     {"normalMap", s.normalMap},
                                     {"normalStrength", s.normalStrength},
                                     {"bumpMap", s.bumpMap},
                                     {"bumpStrength", s.bumpStrength},
                                     {"area", area[mi]},
                                     {"uvArea", uvArea[mi]}};
          if(s.detail)
            e["detail"] = {{"map", s.detail->map},           {"heightMap", s.detail->heightMap}, {"strength", s.detail->strength},
                           {"weight", s.detail->weight},     {"weightMap", s.detail->weightMap}, {"tiles", s.detail->tiles}};
          list.push_back(e);
        }
        dumps[step["surfaces"].get<std::string>()] = list;
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
          // the glTF's bytes and how they are cut up, to compare builds by: FNV-1a over the buffer,
          // then over each view, accessor, primitive and skin
          uint64_t   h    = 1469598103934665603ull;
          const auto bytes = [&h](const void* p, size_t n) {
            for(size_t i = 0; i < n; i++)
              h = (h ^ static_cast<const unsigned char*>(p)[i]) * 1099511628211ull;
          };
          const auto num = [&](double v) { bytes(&v, sizeof v); };
          const tinygltf::Model& gm = built.model;
          for(const auto& b : gm.buffers)
            bytes(b.data.data(), b.data.size());
          const uint64_t bufferHash = h;
          for(const auto& v : gm.bufferViews)
            for(double x : {double(v.buffer), double(v.byteOffset), double(v.byteLength), double(v.byteStride), double(v.target)})
              num(x);
          for(const auto& a : gm.accessors)
          {
            for(double x : {double(a.bufferView), double(a.byteOffset), double(a.componentType), double(a.count), double(a.type)})
              num(x);
            for(double x : a.minValues)
              num(x);
            for(double x : a.maxValues)
              num(x);
          }
          for(const auto& m : gm.meshes)
            for(const auto& p : m.primitives)
            {
              for(const auto& [name, accessor] : p.attributes)
              {
                bytes(name.data(), name.size());
                num(accessor);
              }
              num(p.indices);
              num(p.material);
            }
          for(const auto& s : gm.skins)
          {
            num(s.inverseBindMatrices);
            for(int j : s.joints)
              num(j);
          }
          std::cerr << "glTF " << gm.buffers.front().data.size() << " bytes, " << gm.bufferViews.size() << " views, " << gm.accessors.size()
                    << " accessors, hash " << std::hex << h << " (buffer " << bufferHash << ")" << std::dec << "\n";
          // what it holds, read back against the scene (a hash differs from run to run with the figure)
          const auto problems = fv::build::checkBuild(loaded->loaded.scene, built, {});
          std::cerr << "glTF check: " << (problems.empty() ? "the scene's own bytes" : std::to_string(problems.size()) + " problems") << "\n";
          for(const std::string& p : problems)
            std::cerr << "  " << p << "\n";
        }
      }
      else if(step.contains("feed"))
      {
        if(!loaded)
          throw std::runtime_error("feed before a figure loads");
        const fv::SceneGraph& scene = loaded->loaded.scene;
        const int             root  = loaded->loaded.root;
        for(int k = 0; k < std::max(1, step["feed"].get<int>()); k++)
        {
          rs::RetainedScene store;
          fv::FigureFeed    feed(store, {});
          {
            fv::timing::Scope t("feed");
            feed.sync(scene, root);
          }
          // the sync every frame makes, with nothing changed since
          {
            fv::timing::Scope t("feed again");
            feed.sync(scene, root);
          }
          std::cerr << "store " << store.entityCount() << " entities, " << store.meshCount() << " meshes, " << store.materialCount()
                    << " materials, " << store.drawItems().size() << " draw items\n";
          // what it holds, read back against the scene
          const auto problems = fv::checkFeed(scene, root, feed, store, {});
          std::cerr << "feed check: " << (problems.empty() ? "the scene's own bytes" : std::to_string(problems.size()) + " problems") << "\n";
          for(const std::string& p : problems)
            std::cerr << "  " << p << "\n";
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
