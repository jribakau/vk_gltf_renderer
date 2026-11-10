// Runs a case (cases/*.json; the app repository's parity cases before the TypeScript side was
// deleted) through the figure library, with no GPU, and writes what it built: bones, meshes and morph
// weights in the shape the TypeScript dumper wrote, so two runs can be diffed.
//
//   fv_parity <case.json> <out.json> [--root <dir for relative library paths>] [--cache <parsed-file cache dir>]
//             [--bind-cache <surface-binding cache dir>] [--map-cache <body map cache dir>]
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
// "job": file writes what the solver is given, for the GPU cloth bench (src/tools/fv_cloth_bench.cpp);
// "apply": true also runs the driver's own path (the settle on the pool, the drape written as a
// morph) and measures the drape read back through the skin, as the renderer gets it;
// "colliders": false leaves worn items without a simulation modifier out of the simulation.
// {"retarget": label, "file": rel} plays a DAZ animation made for another Genesis on the loaded figure
// (figure/pose/retarget.hpp) beside the source figure's skeleton playing the file as it is. At every
// frame each limb's direction against its rest direction turned as the source's limb is turned: the
// largest and mean angle off ("limbsDegrees"); how far apart each limb points at rest, the two builds
// ("restDegrees"); the figure's bones turned over 1° where the file holds their counterparts at rest
// ("heldButTurnedDegrees", empty when the figure keeps its own face); how far the lower ankle rises
// and falls on each (cm); and the time the retarget takes.
// {"flesh": label} finds the body's soft lobes (figure/physics/flesh) and reports each: its particles,
// its middle and extent in the rest cage (cm), and its softest share; then bobs the skin up and down
// (2 cm at 2 Hz, a walk's) for "frames" frames at 60 fps and reports the solve's time a frame and
// each lobe's largest offset, and the time of a frame's skin of the lobes at the figure's pose.
// {"touch": label} bobs the figure's hip (2 cm at 2 Hz) for "frames" frames at the pose it is in, with
// its soft flesh on and meeting its own limbs (its ragdoll's capsules, step 11): the most particles
// pressed in a frame, how far the deepest went, which lobes, and a frame of the flesh's time.
// {"bodymap": figure} maps a Genesis type's base ("Genesis 8 Female") onto Genesis 9's shaped as it by its
// clone morph and back (figure/bodymap), and prints the gate of wardrobe plan step 3: how far the base lies
// from where it lands, the round trip by where on the body, the shares mapped each way, the stretch
// and the triangles turned over, how far the frames turn by region beside the limbs' turns into the
// rest, and the time (the loads apart, and whether the map came from the cache). Numbers only.
// {"clearance": rel} wears a garment and reports, per region, how its clearance from the body changed
// from the body it was shaped around (its generation's base cage, or a Skyrim outfit's carried body)
// to the loaded figure's, fitted: the measure of wardrobe plan step 3. Numbers only.
// {"contact": label} runs the body's contact pass at the pose it is in with nothing near it, so only
// its own parts press on one another (figure/physics/contact, step 10): the vertices pressed, the
// dents, how many vertices moved and how far at most, by part, and the pass's time.
//
// Every step's time goes to stderr ("[step] ..."), with the library's phase timers ("[fv-time] ...")
// inside it: the case in cases/native_test.json is the profile of a scene file's load.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <figure/bodymap/cache.hpp>
#include <figure/garment/pieces.hpp>
#include <figure/garment/quads.hpp>
#include <figure/bodymap/clone.hpp>
#include <figure/bodymap/fit.hpp>
#include <figure/bodymap/frames.hpp>
#include <figure/dson/store.hpp>
#include <figure/physics/cloth/backend.hpp>
#include <figure/physics/cloth/cloth.hpp>
#include <figure/physics/cloth/capsules.hpp>
#include <figure/physics/cloth/leash.hpp>
#include <figure/physics/skin.hpp>
#include <figure/physics/cloth/island.hpp>
#include <figure/physics/cloth/job.hpp>
#include <figure/physics/cloth/report.hpp>
#include <figure/dson/material_preset.hpp>
#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/gltf/gltf_check.hpp>
#include <figure/io/timing.hpp>
#include <figure/dson/rig.hpp>
#include <figure/physics/collide/surface_index.hpp>
#include <figure/physics/contact/contact.hpp>
#include <figure/physics/flesh/flesh.hpp>
#include <figure/physics/flesh/follow.hpp>
#include <figure/physics/flesh/touch.hpp>
#include <figure/physics/rig/figure_physics.hpp>
#include <figure/pose/retarget.hpp>
#include <figure/rig/spec.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/bind_cache.hpp>
#include <figure/wardrobe/collider.hpp>
#include <figure/wardrobe/fit/bases.hpp>
#include <figure/wardrobe/fit/collide.hpp>
#include <figure/wardrobe/fit/fit.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/nif/nif_wear.hpp>

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

// The bits of a float array (FNV-1a, 64 bits, hex): two drapes with the same hash came out the same to
// the bit, which a move of the solver's code must keep (physics-plan step 12, gate G1).
std::string bitsOf(const std::vector<float>& v)
{
  uint64_t h = 1469598103934665603ull;
  for(const float f : v)
  {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    for(int k = 0; k < 4; k++)
    {
      h ^= (u >> (8 * k)) & 0xFF;
      h *= 1099511628211ull;
    }
  }
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
  return buf;
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
  const std::vector<int> nodeOf = fv::physics::nodeOfMesh(scene);
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

using fv::physics::cloth::stretchRatios;

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

// A surface's vertex normals, each its faces' summed by area.
std::vector<double> vertexNormals(std::span<const float> pos, std::span<const uint32_t> tris)
{
  std::vector<double> n(pos.size(), 0.0);
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
  {
    const size_t a = size_t(tris[t]) * 3, b = size_t(tris[t + 1]) * 3, c = size_t(tris[t + 2]) * 3;
    const double u[3] = {pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]};
    const double v[3] = {pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2]};
    const double f[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    for(const size_t k : {a, b, c})
      for(size_t i = 0; i < 3; i++)
        n[k + i] += f[i];
  }
  return n;
}

// Each point's signed distance (cm) to a surface: to its closest point there, negative behind it (by
// the surface's normal at that point).
std::vector<double> signedClearance(std::span<const float> pos, std::span<const uint32_t> tris, std::span<const float> points)
{
  const std::vector<double>          nrm     = vertexNormals(pos, tris);
  const fv::wardrobe::SurfaceBinding binding = fv::wardrobe::bindToSurface(pos, tris, points);
  std::vector<double>                out(points.size() / 3, 0.0);
  for(size_t v = 0; v < out.size(); v++)
  {
    double at[3] = {}, n[3] = {};
    for(size_t j = 0; j < 3; j++)
    {
      const size_t b = size_t(binding.verts[v * 3 + j]) * 3;
      const double w = binding.weights[v * 3 + j];
      for(size_t i = 0; i < 3; i++)
      {
        at[i] += w * pos[b + i];
        n[i] += w * nrm[b + i];
      }
    }
    const double d[3] = {points[v * 3] - at[0], points[v * 3 + 1] - at[1], points[v * 3 + 2] - at[2]};
    const double len  = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    out[v]            = d[0] * n[0] + d[1] * n[1] + d[2] * n[2] < 0 ? -len : len;
  }
  return out;
}

// A body region as the measure reports it: the spec's, a finger with its hand and a toe with its foot.
std::string clearanceRegion(fv::rig::Region r)
{
  using R = fv::rig::Region;
  switch(r)
  {
    case R::ePelvis: return "pelvis";
    case R::eAbdomen: return "abdomen";
    case R::eChest: return "chest";
    case R::eBreast: return "breast";
    case R::eNeck: return "neck";
    case R::eHead: return "head";
    case R::eCollar: return "collar";
    case R::eUpperArm: return "upper arm";
    case R::eForearm: return "forearm";
    case R::eHand:
    case R::eThumb:
    case R::eIndex:
    case R::eMid:
    case R::eRing:
    case R::ePinky: return "hand";
    case R::eThigh: return "thigh";
    case R::eShin: return "shin";
    case R::eFoot:
    case R::eBigToe:
    case R::eIndexToe:
    case R::eMidToe:
    case R::eRingToe:
    case R::ePinkyToe: return "foot";
    case R::eNone: break;
  }
  return "other";
}

// Each cage vertex's region: its strongest bone's, by the skeleton spec.
std::vector<std::string> cageRegions(const fv::SceneGraph& scene, const fv::Mesh& mesh)
{
  const size_t               n = mesh.geo ? mesh.geo->cagePositions.size() / 3 : 0;
  std::vector<std::string>   out(n, "other");
  const fv::mesh::SkinInput& skin = mesh.cageSkin;
  if(skin.start.size() != n + 1)
    return out;
  for(size_t v = 0; v < n; v++)
  {
    int   best = -1;
    float most = 0;
    for(int32_t i = skin.start[v]; i < skin.start[v + 1]; i++)
      if(skin.weight[size_t(i)] > most)
        most = skin.weight[size_t(i)], best = skin.bone[size_t(i)];
    if(best >= 0 && size_t(best) < mesh.bones.size())
    {
      const fv::rig::Name name = fv::rig::parse(scene.nodes[size_t(mesh.bones[size_t(best)])].name);
      out[v]                   = clearanceRegion(name.joint ? name.joint->region : fv::rig::Region::eNone);
    }
  }
  return out;
}

// A body map's gate (wardrobe plan step 3), printed: the turns into the rest, the shares mapped, the
// stretch and folds, how far the source lies from where it lands and the round trip by class, what is
// unmapped, far or turned over by region (and turned over by surface, when the source's cage names its
// surfaces), and the frame turns by region. `source` is the type's cage, `g9` Genesis 9's.
Json reportMap(const fv::bodymap::TypeMap& m, const fv::wardrobe::BaseCage& source, const fv::wardrobe::BaseCage& g9)
{
  const fv::bodymap::MapQuality q = fv::bodymap::mapQuality(m.source, m.reference, m.map);
  Json spreads = Json::object();
  std::cerr << std::fixed << std::setprecision(1) << "  turned into its rest:";
  for(const auto& t : m.turns)
  {
    std::cerr << " " << (t.side ? std::string(1, t.side) + " " : std::string()) << t.joint << " " << t.degrees;
    if(t.moved > 0.05 || std::abs(t.sized - 1) > 0.005)
      std::cerr << std::setprecision(2) << " (moved " << t.moved << " cm, sized " << t.sized << ")" << std::setprecision(1);
  }
  std::cerr << "\n"
            << std::setprecision(4) << "  mapped " << q.mapped << ", the reference covered back " << q.inverseCovers << ", stretch p95 " << q.stretchP95
            << " max " << q.stretchMax << ", fold-free " << q.foldFree << "\n"
            << std::setprecision(3);
  for(const auto& [what, table] : {std::pair{"from where it lands", &q.distance}, std::pair{"round trip", &q.roundTrip}})
  {
    std::cerr << "  " << what << " (cm):\n";
    for(const auto& [cls, sp] : *table)
    {
      std::cerr << "    " << std::left << std::setw(13) << cls << std::right << " n " << std::setw(6) << sp.n << "  median " << std::setw(6) << sp.median
                << "  p95 " << std::setw(6) << sp.p95 << "  max " << std::setw(7) << sp.max << "\n";
      spreads[what][cls] = {{"n", sp.n}, {"median", r(sp.median)}, {"p95", r(sp.p95)}, {"max", r(sp.max)}};
    }
  }
  for(const auto& [what, counts] : {std::pair{"unmapped", &q.unmappedBy}, std::pair{"over 1 cm off", &q.farBy}, std::pair{"turned over", &q.foldedBy}, std::pair{"stretched past 1.5", &q.stretchedBy}})
  {
    std::vector<std::pair<size_t, std::string>> most;
    for(const auto& [regions, n] : *counts)
      most.emplace_back(n, regions);
    std::sort(most.rbegin(), most.rend());
    std::cerr << "  " << what << ", by region:";
    for(size_t i = 0; i < most.size() && i < 40; i++)
      std::cerr << (i ? ", " : " ") << most[i].second << " " << most[i].first;
    std::cerr << (most.size() > 40 ? ", ..." : "") << "\n";
  }
  // why a source vertex lands over 1 cm off: the reference's skin is that far from it anyway (it doesn't
  // reach there), or nearer skin was refused (facing away past 60 degrees, or of other regions)
  {
    const fv::physics::SurfaceIndex index(m.reference.positions, m.reference.tris);
    const std::vector<fv::Vec3>     normals = fv::bodymap::outerNormals(m.source);
    const auto P = [](const std::vector<float>& p, size_t v) { return fv::Vec3{p[v * 3], p[v * 3 + 1], p[v * 3 + 2]}; };
    size_t     far = 0, uncovered = 0, between = 0, refusedFacing = 0, refusedRegion = 0, facingDown = 0;
    std::map<std::string, std::array<size_t, 3>> byRegion;
    fv::physics::Stamps& stamps = fv::physics::threadStamps();
    for(size_t v = 0; v < m.map.toReference.mapped.size(); v++)
    {
      if(!m.map.toReference.mapped[v])
        continue;
      fv::Vec3 q;
      for(size_t j = 0; j < 3; j++)
        q = q + P(m.reference.positions, m.map.toReference.at.verts[v * 3 + j]) * double(m.map.toReference.at.weights[v * 3 + j]);
      const fv::Vec3 p = P(m.source.positions, v);
      if(fv::length(q - p) <= 1)
        continue;
      far++;
      facingDown += normals[v].y < -0.5;
      const int32_t id  = stamps.next(index.triangles());
      const auto    hit = index.nearest(p.x, p.y, p.z, [&](size_t t) { return t >= m.reference.outer.size() || m.reference.outer[t]; }, 100.0, stamps.stamp, id);
      auto&         row = byRegion[fv::bodymap::regionNames(m.source.regions[v])];
      if(hit.tri < 0 || hit.d2 >= 1)
      {
        uncovered++, row[0]++;
        continue;
      }
      if(hit.d2 >= 0.09)
      {
        between++, row[1]++;
        continue;
      }
      row[2]++;
      const size_t   t = size_t(hit.tri);
      const fv::Vec3 a = P(m.reference.positions, m.reference.tris[t * 3]);
      const fv::Vec3 e1 = P(m.reference.positions, m.reference.tris[t * 3 + 1]) - a, e2 = P(m.reference.positions, m.reference.tris[t * 3 + 2]) - a;
      const fv::Vec3 face = fv::normalize(fv::Vec3{e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x});
      if(fv::dot(face, normals[v]) < std::cos(60 * fv::kDeg))
        refusedFacing++;
      else
        refusedRegion++;
    }
    std::cerr << "  over 1 cm off, why: " << far << " vertices; the reference's skin over 1 cm away anyway " << uncovered << ", 3 mm to 1 cm " << between
              << ", within 3 mm but refused " << (refusedFacing + refusedRegion) << " (facing " << refusedFacing << ", regions " << refusedRegion
              << "); facing down " << facingDown << "\n    by region (skin over 1 cm, 3 mm-1 cm, within 3 mm):";
    for(const auto& [name, row] : byRegion)
      if(row[0] + row[1] + row[2] >= 20)
        std::cerr << " " << name << " " << row[0] << "/" << row[1] << "/" << row[2];
    std::cerr << "\n";
    // where the round trip (source to the reference and back) ends over 3 mm off, by region, and how
    // high (the source's height range in tenths)
    {
      const auto&                   there = m.map.toReference;
      const auto&                   back  = m.map.fromReference;
      std::map<std::string, size_t> off;
      std::array<size_t, 10>        byHeight{};
      double                        lo = INFINITY, hi = -INFINITY;
      for(size_t v = 0; v < m.source.positions.size() / 3; v++)
        lo = std::min(lo, double(m.source.positions[v * 3 + 1])), hi = std::max(hi, double(m.source.positions[v * 3 + 1]));
      size_t total = 0;
      for(size_t v = 0; v < there.mapped.size(); v++)
      {
        if(!there.mapped[v])
          continue;
        fv::Vec3 q;
        double   w = 0;
        for(size_t j = 0; j < 3; j++)
        {
          const uint32_t r = there.at.verts[v * 3 + j];
          if(r >= back.mapped.size() || !back.mapped[r])
            continue;
          fv::Vec3 s;
          for(size_t k = 0; k < 3; k++)
            s = s + P(m.source.positions, back.at.verts[r * 3 + k]) * double(back.at.weights[r * 3 + k]);
          q = q + s * double(there.at.weights[v * 3 + j]);
          w += there.at.weights[v * 3 + j];
        }
        if(w < 0.5 || fv::length(q * (1 / w) - P(m.source.positions, v)) <= 0.3)
          continue;
        total++;
        off[fv::bodymap::regionNames(m.source.regions[v])]++;
        byHeight[std::min<size_t>(9, size_t(10 * (m.source.positions[v * 3 + 1] - lo) / std::max(hi - lo, 1e-9)))]++;
      }
      std::vector<std::pair<size_t, std::string>> most;
      for(const auto& [name, n] : off)
        most.emplace_back(n, name);
      std::sort(most.rbegin(), most.rend());
      std::cerr << "  round trip over 3 mm: " << total << " vertices; by region:";
      for(size_t i = 0; i < most.size() && i < 12; i++)
        std::cerr << " " << most[i].second << " " << most[i].first;
      std::cerr << "; by height, feet to head:";
      for(const size_t n : byHeight)
        std::cerr << " " << n;
      std::cerr << "\n";
    }
  }
  // the turned-over triangles by the surface they're on (nails, ears...)
  if(const fv::wardrobe::BaseCage* cage = &source; !cage->triSurface.empty())
  {
    std::map<std::string, size_t> bySurface, all;
    for(size_t t = 0; t < cage->triSurface.size(); t++)
      all[cage->triSurface[t] < cage->surfaces.size() ? cage->surfaces[cage->triSurface[t]] : "?"]++;
    for(const uint32_t t : q.foldedTris)
      bySurface[t < cage->triSurface.size() && cage->triSurface[t] < cage->surfaces.size() ? cage->surfaces[cage->triSurface[t]] : "?"]++;
    std::cerr << "  turned over, by surface:";
    size_t skin = 0, turned = 0, nails = 0, nailsTurned = 0;
    for(const auto& [surface, n] : all)
      if(fv::bodymap::outerSkin(surface))
        (surface.find("nail") != std::string::npos ? nails : skin) += n;
    for(const auto& [surface, n] : bySurface)
    {
      std::cerr << " " << surface << " " << n << " of " << all[surface];
      (surface.find("nail") != std::string::npos ? nailsTurned : turned) += n;
    }
    std::cerr << std::setprecision(4) << "\n  fold-free outside the nails " << (skin ? 1 - double(turned) / double(skin) : 0.0) << ", on them "
              << (nails ? 1 - double(nailsTurned) / double(nails) : 0.0) << "\n";
  }
  // how far each region's frames turn, its sides together, on vertices in it alone (the nails apart):
  // on a limb, about as far as the limb was turned into the rest. In degrees, the median and p95 of
  // the whole turn and of its parts: the normal's bend onto Genesis 9's, and the twist about it after
  {
    struct Turned
    {
      std::vector<double> whole, bend, twist;
    };
    std::map<std::string, Turned>  byRegion;
    const fv::bodymap::Frames      from = fv::bodymap::vertexFrames(m.source);
    const fv::bodymap::Frames      onto = fv::bodymap::vertexFrames(fv::bodymap::bodyOf(g9));
    const auto                     angleOf = [](const fv::Quat& q) { return 2 * std::acos(std::min(1.0, std::abs(q.w))) / fv::kDeg; };
    std::vector<uint8_t>           onNail(m.source.positions.size() / 3, 0);
    const fv::wardrobe::BaseCage&  base = source;
    for(size_t t = 0; t < base.triSurface.size() && t * 3 + 2 < base.tris.size(); t++)
      if(base.triSurface[t] < base.surfaces.size() && base.surfaces[base.triSurface[t]].find("nail") != std::string::npos)
        for(size_t j = 0; j < 3; j++)
          onNail[base.tris[t * 3 + j]] = 1;
    for(size_t v = 0; v < m.map.frameTurn.size(); v++)
    {
      const fv::bodymap::RegionMask mask = m.source.regions[v];
      if(!m.map.toReference.mapped[v] || !mask || (!onNail[v] && (mask & (mask - 1))) || fv::dot(from.normal[v], from.normal[v]) == 0)
        continue;
      std::string name = onNail[v] ? "nails" : fv::bodymap::regionNames(mask);
      if(name.rfind("r ", 0) == 0)
        name = name.substr(2);
      fv::Vec3 ng;
      for(size_t j = 0; j < 3; j++)
        ng = ng + onto.normal[m.map.toReference.at.verts[v * 3 + j]] * m.map.toReference.at.weights[v * 3 + j];
      const fv::Quat bend = fv::quatFromUnitVectors(from.normal[v], fv::normalize(ng));
      Turned&        t    = byRegion[name];
      t.whole.push_back(angleOf(m.map.frameTurn[v]));
      t.bend.push_back(angleOf(bend));
      t.twist.push_back(angleOf(m.map.frameTurn[v] * bend.inverse()));
    }
    std::cerr << std::setprecision(1) << "  frames turned, degrees (median and p95 of the whole turn, the bend, the twist):\n";
    for(auto& [name, t] : byRegion)
    {
      std::cerr << "    " << std::left << std::setw(9) << name << std::right;
      for(std::vector<double>* s : {&t.whole, &t.bend, &t.twist})
      {
        std::sort(s->begin(), s->end());
        std::cerr << "  " << std::setw(5) << (*s)[s->size() / 2] << " " << std::setw(5) << (*s)[size_t(0.95 * double(s->size() - 1))];
      }
      std::cerr << "\n";
    }
  }
  std::cerr << std::defaultfloat << std::setprecision(6);
  return {{"mapped", r(q.mapped)}, {"inverseCovers", r(q.inverseCovers)}, {"stretchP95", r(q.stretchP95)}, {"foldFree", r(q.foldFree)}, {"spreads", spreads}};
}

// Where a fitted map lands each source vertex against where the clone map does (the truth, for a
// Genesis type fitted from another shape): both addresses placed on the clone map's reference, in cm.
Json reportAgainstTruth(const fv::bodymap::TypeMap& fit, const fv::bodymap::TypeMap& truth)
{
  const auto place = [&](const fv::bodymap::Projection& p, size_t v) {
    fv::Vec3 out;
    for(size_t j = 0; j < 3; j++)
    {
      const uint32_t c = p.at.verts[v * 3 + j];
      out              = out + fv::Vec3{truth.reference.positions[c * 3], truth.reference.positions[c * 3 + 1], truth.reference.positions[c * 3 + 2]} * p.at.weights[v * 3 + j];
    }
    return out;
  };
  std::map<std::string, std::vector<double>> off;
  std::map<std::string, std::pair<fv::Vec3, size_t>> mean;  // the left side's mean offset: a slide shows, a twist cancels
  for(size_t v = 0; v < fit.map.toReference.mapped.size() && v < truth.map.toReference.mapped.size(); v++)
  {
    if(!fit.map.toReference.mapped[v] || !truth.map.toReference.mapped[v])
      continue;
    const fv::Vec3 dv = place(fit.map.toReference, v) - place(truth.map.toReference, v);
    const double   d  = fv::length(dv);
    off["all"].push_back(d);
    const fv::bodymap::RegionMask m = fit.source.regions[v];
    if(m && !(m & (m - 1)))
    {
      std::string name = fv::bodymap::regionNames(m);
      if(name.rfind("r ", 0) != 0)
        mean[name].first = mean[name].first + dv, mean[name].second++;
      off[name.rfind("r ", 0) == 0 ? name.substr(2) : name].push_back(d);
    }
  }
  Json out = Json::object();
  std::cerr << std::fixed << std::setprecision(2) << "  against the clone map (cm from where it lands there: median, p95, max; share within 1 cm):\n";
  for(auto& [name, d] : off)
  {
    std::sort(d.begin(), d.end());
    const double within = double(std::lower_bound(d.begin(), d.end(), 1.0) - d.begin()) / double(d.size());
    std::cerr << "    " << std::left << std::setw(9) << name << std::right << " n " << std::setw(6) << d.size() << "  " << std::setw(6) << d[d.size() / 2] << " "
              << std::setw(6) << d[size_t(0.95 * double(d.size() - 1))] << " " << std::setw(7) << d.back() << "  " << within;
    if(const auto it = mean.find(name); it != mean.end() && it->second.second)
    {
      const fv::Vec3 a = it->second.first * (1.0 / double(it->second.second));
      std::cerr << "   mean offset (left) " << a.x << " " << a.y << " " << a.z;
    }
    std::cerr << "\n";
    out[name] = {{"n", d.size()}, {"median", r(d[d.size() / 2])}, {"p95", r(d[size_t(0.95 * double(d.size() - 1))])}, {"max", r(d.back())}};
  }
  // where each rig's joint bands lie on the body: the source's band middle against the reference's,
  // both on the clone map's reference (the source's placed through the map), so a fit holding the two
  // together is off by this much there
  {
    const std::vector<double> sourceArea = fv::bodymap::vertexAreas(truth.source), refArea = fv::bodymap::vertexAreas(truth.reference);
    std::cerr << "  joint bands, the source's against the reference's on the clone (cm):";
    for(int r = int(fv::rig::Region::eAbdomen); r <= int(fv::rig::Region::ePinkyToe); r++)
    {
      const auto region = fv::rig::Region(r), up = fv::bodymap::regionAbove(region);
      if(up == fv::rig::Region::eNone || (fv::bodymap::regionSided(region) && r > int(fv::rig::Region::eFoot)))
        continue;
      const char       side = fv::bodymap::regionSided(region) ? 'l' : 0;
      const auto       bit = fv::bodymap::regionBit(region, side), above = fv::bodymap::regionBit(up, fv::bodymap::regionSided(up) ? side : 0);
      fv::Vec3         s, t;
      double           sw = 0, tw = 0;
      for(size_t v = 0; v < truth.source.regions.size(); v++)
        if((truth.source.regions[v] & bit) && (truth.source.regions[v] & above) && truth.map.toReference.mapped[v] && sourceArea[v] > 0)
          s = s + place(truth.map.toReference, v) * sourceArea[v], sw += sourceArea[v];
      for(size_t v = 0; v < truth.reference.regions.size(); v++)
        if((truth.reference.regions[v] & bit) && (truth.reference.regions[v] & above) && refArea[v] > 0)
          t = t + fv::Vec3{truth.reference.positions[v * 3], truth.reference.positions[v * 3 + 1], truth.reference.positions[v * 3 + 2]} * refArea[v], tw += refArea[v];
      if(sw > 0 && tw > 0)
      {
        const fv::Vec3 d = s * (1 / sw) - t * (1 / tw);
        std::cerr << " " << fv::bodymap::regionNames(bit) << " " << fv::length(d) << " (" << d.x << " " << d.y << " " << d.z << "),";
      }
    }
    std::cerr << "\n";
  }
  std::cerr << std::defaultfloat << std::setprecision(6);
  return out;
}

// Per region, how a garment's clearance changed from its source body to the figure: the source's and
// the figure's median, and the change's median, 5th and 95th percentiles, over every vertex and over
// those within 3 cm of the source body (the band a garment fits by).
Json clearanceChange(const std::vector<std::string>& regions, const std::vector<double>& before, const std::vector<double>& after)
{
  std::map<std::string, std::array<std::vector<double>, 3>> all, band;
  for(size_t v = 0; v < regions.size() && v < before.size() && v < after.size(); v++)
    for(const std::string& key : {regions[v], std::string("all")})
    {
      auto& a = all[key];
      a[0].push_back(before[v]), a[1].push_back(after[v]), a[2].push_back(after[v] - before[v]);
      if(before[v] > -1 && before[v] < 3)
      {
        auto& b = band[key];
        b[0].push_back(before[v]), b[1].push_back(after[v]), b[2].push_back(after[v] - before[v]);
      }
    }
  const auto pct = [](std::vector<double> s, double q) {
    std::sort(s.begin(), s.end());
    return s.empty() ? 0.0 : r(s[size_t(q * double(s.size() - 1))]);
  };
  const auto stats = [&](const std::array<std::vector<double>, 3>& a) {
    return Json{{"n", a[2].size()},       {"source", pct(a[0], 0.5)},      {"figure", pct(a[1], 0.5)},
                {"change", pct(a[2], 0.5)}, {"changeP05", pct(a[2], 0.05)}, {"changeP95", pct(a[2], 0.95)}};
  };
  Json out = Json::object();
  for(const auto& [key, a] : all)
  {
    out[key] = {{"all", stats(a)}};
    if(const auto it = band.find(key); it != band.end())
      out[key]["band"] = stats(it->second);
  }
  return out;
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
    const auto            nodeOf = fv::physics::nodeOfMesh(scene);
    const fv::Mat4        to     = fv::inverse(worlds[size_t(fig.loaded.root)]) * worlds[size_t(nodeOf[size_t(mesh)])];
    const auto            rest   = fig.morphs->morphedCage(mesh);
    const fv::Mesh&       m      = scene.meshes[size_t(mesh)];
    if(!m.skinned() || m.cageSkin.start.size() != rest.size() / 3 + 1)
      return fv::physics::transformPoints(to, rest);
    return fv::physics::transformPoints(to, fv::physics::skinLocal(rest, fv::physics::cageSkin(m.cageSkin), fv::physics::currentSkin(scene, worlds, nodeOf, mesh)));
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
Json drapeReport(const fv::physics::cloth::Model& model, size_t slot, const fv::physics::cloth::Surface& mesh, const std::vector<float>& out,
                 const fv::physics::cloth::Surface& body, const std::vector<float>& bodyNow)
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
          {"boxPosed", box(mesh.posed)},
          {"bits", bitsOf(out)}};
}

// A drape's per-vertex and per-edge detail beside its OBJ, for a script to find where it went
// wrong: each vertex's strength and signed depth against the body (null when it bound to no face),
// and every stretch edge with its rest length.
void writeSidecar(const std::filesystem::path& file, const fv::physics::cloth::Model& model, size_t slot, const fv::physics::cloth::Surface& mesh,
                  const std::vector<float>& out, const fv::physics::cloth::Surface& body, const std::vector<float>& bodyNow)
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
  for(const fv::physics::cloth::Distance& e : model.edges)
    if(e.kind == fv::physics::cloth::DistanceKind::eStretch && e.a >= base && e.b >= base && e.a < base + int(n) && e.b < base + int(n)
       && (model.inv[size_t(e.a)] > 0 || model.inv[size_t(e.b)] > 0))
      edges.push_back(Json::array({e.a - base, e.b - base, r(e.rest)}));
  std::ofstream o(file);
  o << Json{{"strength", strength}, {"depth", depth}, {"edges", edges}}.dump();
}

// A mesh as an OBJ: its polygons when it has them, else its triangles.
void writeObj(const std::filesystem::path& file, const std::vector<float>& pos, const fv::physics::cloth::Surface& mesh)
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
    if(std::string(argv[i]) == "--map-cache")  // the body map cache, as the renderer's --cacheDir/bodymap
      fv::bodymap::setMapCacheDir(pathOf(argv[i + 1]));
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
    // {"clip": rel, "time": s[, "ahead": s, "dt": s]}: a DAZ animation's pose at that time, set as the
    // player sets it (the bones, then the dials); with "ahead", each channel carried on linearly from
    // its value "dt" before, as a guess at where the pose will be that long after
    std::map<std::string, fv::pose::PoseClip> clips;
    const auto                                poseAtClip = [&](const Json& s) {
      if(!loaded || !loaded->pose)
        throw std::runtime_error("clip before a figure loads");
      const std::string rel = s.at("clip").get<std::string>();
      auto              it  = clips.find(rel);
      if(it == clips.end())
        it = clips.emplace(rel, loaded->loadPoseClip(store->load(rel)->json, rel)).first;
      const fv::pose::PoseClip& clip  = it->second;
      const float               t     = s.value("time", 0.f);
      const float               ahead = s.value("ahead", 0.f);
      const float               dt    = s.value("dt", 1.f / 60.f);
      std::vector<fv::pose::ChannelValue>         bones;
      std::vector<std::pair<std::string, double>> dials;
      for(const fv::pose::PoseClip::Channel& ch : clip.channels)
      {
        double v = fv::pose::sample(ch, t);
        if(ahead != 0.f && dt > 0.f)
          v += (v - fv::pose::sample(ch, t - dt)) * double(ahead / dt);
        if(ch.target.kind == fv::pose::PresetTarget::Kind::eBone)
          bones.push_back({ch.target.bone, ch.target.channel, ch.target.axis, v});
        else if(ch.target.kind == fv::pose::PresetTarget::Kind::eDial)
          dials.emplace_back(ch.target.name, v);
      }
      if(!bones.empty())
        loaded->pose->setMany(bones);
      if(loaded->morphs && !dials.empty())
        loaded->morphs->setMany(dials);
      loaded->settle();
      return clip.duration;
    };
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
        auto              prepared = loaded->prepareWear(rel, fv::WearQuality{step.value("subdivision", 0), step.value("strandKeep", 1.f), step.value("strandPieces", 2)});
        // as the host does on its worker, so the build after the wear finds the strands measured
        if(step.value("measure", false))
        {
          fv::timing::Scope t("measureStrands");
          fv::build::measureStrands(fv::Figure::preparedScene(*prepared), false);
        }
        const fv::WearResult w = loaded->commitWear(prepared, rel);
        loaded->settle();
        size_t triangles = 0;
        for(const fv::WornItem& item : loaded->items)
          if(std::find(w.items.begin(), w.items.end(), item.id) != w.items.end())
            for(int m : item.meshes)
              if(const auto& geo = loaded->loaded.scene.meshes[size_t(m)].geo)
                triangles += geo->fullIndex.size() / 3;
        Json report = {{"items", w.items}, {"fit", r(w.fit)}, {"triangles", triangles}, {"notes", w.notes}, {"warnings", w.warnings}, {"missingBones", w.missingBones}};
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
      else if(step.contains("bodymap"))
      {
        // {"bodymap": figure} a Genesis type's clone map; {"bodymap": figure, "fit": shape} the same type's
        // body fitted from another Genesis shape ("Genesis 9", or a type whose clone shapes it), checked
        // against its clone map; {"bodymap": an outfit's .nif} the body it carries, fitted from Genesis 9
        // shaped by CloneGenesis8Female. "nodes", "rounds", "landmarks" (their weight) and "stiffTo" set the fit's.
        const std::string figureName = step["bodymap"].get<std::string>();
        const std::string fitFrom    = step.value("fit", std::string());
        const auto        mapStart   = std::chrono::steady_clock::now();
        const auto        g9         = fv::wardrobe::loadBaseCage(*store, fv::wardrobe::Generation::eG9, "");
        if(!g9)
          throw std::runtime_error("bodymap: Genesis 9's base isn't installed");
        std::shared_ptr<const fv::wardrobe::BaseCage> source;
        std::optional<fv::bodymap::TypeMap>           made, truth;
        fv::bodymap::FitResult                        fitted;
        std::string                                   how;
        double                                        loadMs = 0;
        fv::bodymap::FitOptions                       options;
        options.nodes          = step.value("nodes", options.nodes);
        options.rounds         = step.value("rounds", options.rounds);
        options.landmarkWeight = step.value("landmarks", options.landmarkWeight);
        options.stiffTo        = step.value("stiffTo", options.stiffTo);
        if(fv::nif::isNif(figureName) && step.contains("whole"))
        {
          // the outfit's carried body matched to the whole body mod it was cut from ("whole": its
          // femalebody nif), and mapped through the whole body's fitted map
          std::filesystem::path outfit = pathOf(figureName.c_str());
          if(std::error_code ec; !std::filesystem::is_regular_file(outfit, ec))
          {
            const auto found = store->library().find(figureName);
            if(!found)
              throw std::runtime_error(figureName + " isn't in the library");
            outfit = *found;
          }
          const auto carried = fv::nif::carriedBody(outfit);
          const auto whole   = fv::nif::modBody(pathOf(step["whole"].get<std::string>().c_str()));
          const auto shaped  = fv::bodymap::cloneReference(*store, fv::wardrobe::Generation::eG8, "female");
          if(!carried || !whole || !shaped)
            throw std::runtime_error("bodymap: no carried body, whole body or CloneGenesis8Female");
          loadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mapStart).count();
          fv::bodymap::FitInputs in{whole, g9, *shaped};
          const fv::bodymap::TypeMap wholeMap = fv::bodymap::mapCacheDir().empty() ? fv::bodymap::makeFittedMap(in, options, &fitted) : fv::bodymap::cachedFittedMap(in, options);
          const auto           matchStart = std::chrono::steady_clock::now();
          const std::vector<int32_t> match  = fv::bodymap::matchCut(*carried, *whole);
          made                              = fv::bodymap::cutMap(wholeMap, fv::bodymap::bodyOf(*carried), match);
          const double matchMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - matchStart).count();
          source               = carried;
          size_t              matched = 0;
          std::vector<double> preset;
          for(size_t c = 0; c < match.size(); c++)
            if(match[c] >= 0)
            {
              matched++;
              const size_t w = size_t(match[c]);
              preset.push_back(std::sqrt(std::pow(carried->positions[c * 3] - whole->positions[w * 3], 2) + std::pow(carried->positions[c * 3 + 1] - whole->positions[w * 3 + 1], 2)
                                         + std::pow(carried->positions[c * 3 + 2] - whole->positions[w * 3 + 2], 2)));
            }
          std::sort(preset.begin(), preset.end());
          std::ostringstream m;
          m << std::fixed << std::setprecision(2) << "its carried body (" << carried->positions.size() / 3 << " vertices) matched to the whole body's ("
            << whole->positions.size() / 3 << ") by UV: " << matched << " matched, in " << matchMs << " ms; the presets apart (cm) median "
            << (preset.empty() ? 0.0 : preset[preset.size() / 2]) << ", p95 " << (preset.empty() ? 0.0 : preset[size_t(0.95 * double(preset.size() - 1))]) << ", max "
            << (preset.empty() ? 0.0 : preset.back());
          how = m.str();
        }
        else if(fv::nif::isNif(figureName) || !fitFrom.empty())
        {
          fv::bodymap::FitInputs in;
          in.g9 = g9;
          if(fv::nif::isNif(figureName))
          {
            // a library path, or a file on disk (a body mod's meshes unpacked to look at)
            std::filesystem::path file = pathOf(figureName.c_str());
            if(std::error_code ec; !std::filesystem::is_regular_file(file, ec))
            {
              const auto found = store->library().find(figureName);
              if(!found)
                throw std::runtime_error(figureName + " isn't in the library");
              file = *found;
            }
            std::string stem = file.filename().string();
            std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            const bool modBody = stem.rfind("femalebody", 0) == 0;
            in.source          = modBody ? fv::nif::modBody(file) : fv::nif::carriedBody(file);
            if(!in.source)
              throw std::runtime_error(figureName + " carries no body");
            const auto shaped = fv::bodymap::cloneReference(*store, fv::wardrobe::Generation::eG8, "female");
            if(!shaped)
              throw std::runtime_error("bodymap: Genesis 9's CloneGenesis8Female isn't installed");
            in.reference = *shaped;
            {
              const fv::bodymap::Body b = fv::bodymap::bodyOf(*in.source);
              std::map<std::pair<uint32_t, uint32_t>, int> edges;
              for(size_t t = 0; t * 3 + 2 < b.tris.size(); t++)
                for(size_t j = 0; j < 3; j++)
                {
                  const uint32_t x = b.tris[t * 3 + j], y = b.tris[t * 3 + (j + 1) % 3];
                  edges[{std::min(x, y), std::max(x, y)}]++;
                }
              size_t open = 0;
              for(const auto& [e, k] : edges)
                open += k == 1 ? 1 : 0;
              float lo = 1e30f, hi = -1e30f;
              for(size_t v = 1; v < b.positions.size(); v += 3)
                lo = std::min(lo, b.positions[v]), hi = std::max(hi, b.positions[v]);
              how = std::string(modBody ? "the body mod's body (body, hands and feet)" : "its carried body") + " of " + std::to_string(b.positions.size() / 3)
                    + " vertices, " + std::to_string(open) + " open edges, " + std::to_string(int(lo)) + " to " + std::to_string(int(hi))
                    + " cm high, fitted from Genesis 9 shaped by CloneGenesis8Female";
            }
          }
          else
          {
            const auto generation = fv::wardrobe::generationFromPath(figureName);
            const auto from       = fv::wardrobe::generationFromPath(fitFrom);
            if(!generation || !from)
              throw std::runtime_error("bodymap: no generation in " + figureName + " or " + fitFrom);
            in.source = fv::wardrobe::loadBaseCage(*store, *generation, fv::wardrobe::genderHint(figureName));
            if(!in.source)
              throw std::runtime_error("bodymap: " + figureName + "'s base isn't installed");
            if(*from == fv::wardrobe::Generation::eG9)
              in.reference = fv::bodymap::bodyOf(*g9);
            else if(const auto shaped = fv::bodymap::cloneReference(*store, *from, fv::wardrobe::genderHint(fitFrom)))
              in.reference = *shaped;
            else
              throw std::runtime_error("bodymap: " + fitFrom + "'s clone isn't installed");
            how = "fitted from " + fitFrom;
            if(const auto inputs = fv::bodymap::cloneInputs(*store, *generation, fv::wardrobe::genderHint(figureName)))
              truth = fv::bodymap::cachedCloneMap(*inputs);
          }
          loadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mapStart).count();
          // through the cache when it's on (--map-cache), else made here with the fit's own numbers
          if(fv::bodymap::mapCacheDir().empty())
            made = fv::bodymap::makeFittedMap(in, options, &fitted);
          else
          {
            const auto hitsBefore = fv::bodymap::mapCacheCounts().hits;
            made                  = fv::bodymap::cachedFittedMap(in, options);
            how += fv::bodymap::mapCacheCounts().hits > hitsBefore ? ", the map from the cache" : ", the map made and cached";
          }
          source         = in.source;
        }
        else
        {
          const auto generation = fv::wardrobe::generationFromPath(figureName);
          if(!generation)
            throw std::runtime_error("bodymap: no generation in " + figureName);
          const auto inputs = fv::bodymap::cloneInputs(*store, *generation, fv::wardrobe::genderHint(figureName));
          if(!inputs)
            throw std::runtime_error("bodymap: " + figureName + "'s base, Genesis 9's or its clone isn't installed");
          loadMs                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mapStart).count();
          const auto hitsBefore = fv::bodymap::mapCacheCounts().hits;
          made                  = fv::bodymap::cachedCloneMap(*inputs);
          source                = inputs->base;
          how = "by " + fv::bodymap::cloneFile(*generation, fv::wardrobe::genderHint(figureName))
                + (fv::bodymap::mapCacheCounts().hits > hitsBefore ? ", the map from the cache" : fv::bodymap::mapCacheDir().empty() ? "" : ", the map made and cached");
        }
        const double mapMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mapStart).count();
        std::cerr << "[bodymap] " << figureName << " onto Genesis 9, " << how << ": " << made->source.positions.size() / 3 << " and "
                  << made->reference.positions.size() / 3 << " vertices, " << std::fixed << std::setprecision(0) << mapMs << " ms, " << loadMs << " of them loading\n";
        if(fitted.nodes)
          std::cerr << std::setprecision(3) << "  fit: scale " << fitted.scale << ", shift " << fitted.shift.x << " " << fitted.shift.y << " " << fitted.shift.z << ", "
                    << fitted.nodes << " nodes, " << fitted.landmarks << " landmarks; at the last round " << fitted.pulled << " vertices pulled, " << fitted.residualMedian << " cm median off the source, "
                    << fitted.residualP95 << " p95\n";
        std::cerr << std::defaultfloat;
        Json dump       = reportMap(*made, *source, *g9);
        dump["ms"]      = r(mapMs);
        dump["bodymap"] = figureName;
        if(truth)
          dump["againstClone"] = reportAgainstTruth(*made, *truth);
        // "points": a region's name ("hand" takes in its fingers, "foot" its toes): both bodies' points there, each with its
        // region's number, to look at outside
        if(step.contains("points"))
        {
          const std::string part = step["points"].get<std::string>();
          for(const auto& [what, body] : {std::pair{"source", &made->source}, std::pair{"reference", &made->reference}})
          {
            Json pts = Json::array();
            for(size_t v = 0; v < body->regions.size(); v++)
            {
              const fv::bodymap::RegionMask mask = body->regions[v];
              if(!mask || (mask & (mask - 1)))
                continue;
              const std::string name = fv::bodymap::regionNames(mask);
              const bool        right = name.rfind("r ", 0) == 0;
              const std::string bare  = right ? name.substr(2) : name;
              const bool        inHand = bare == "hand" || bare == "thumb" || bare == "index" || bare == "mid" || bare == "ring" || bare == "pinky";
              const bool        inFoot = bare == "foot" || bare.find("toe") != std::string::npos;
              if(right || !(bare == part || (part == "hand" && inHand) || (part == "foot" && inFoot)))
                continue;
              pts.push_back({r(body->positions[v * 3]), r(body->positions[v * 3 + 1]), r(body->positions[v * 3 + 2]), bare});
            }
            dump["points"][what] = pts;
          }
        }
        dumps[step.value("label", std::string("bodymap"))] = dump;
      }
      else if(step.contains("clearance"))
      {
        // {"clearance": rel[, "label": name][, "fit": "common"]}: wears `rel` on the loaded figure (through
        // the common layer where "fit" asks and it can, wardrobe::FitPath) and reports, per region of
        // each garment (its vertices' strongest bones, by the skeleton spec), how its clearance from the
        // body changed: against the body it was shaped around (its generation's base cage, or the body a
        // Skyrim outfit carries), as it was made, and against this figure's body, fitted and worn
        // (wardrobe plan step 3's measure), then takes it off. Numbers only: nothing of the garment is written.
        if(!loaded)
          throw std::runtime_error("clearance before a figure loads");
        const std::string     rel    = step["clearance"].get<std::string>();
        const std::string     file   = rel.substr(std::min(rel.find_first_not_of('/'), rel.size()));
        const auto            inputs = loaded->wearInputs();
        fv::dson::LoadOptions options;
        options.attach = fv::dson::AttachTarget{inputs.scene.get(), inputs.rig.get(), "measure"};
        fv::dson::LoadResult w;
        if(fv::nif::isNif(rel))
        {
          const auto found = store->library().find(file);
          if(!found)
            throw std::runtime_error(file + " isn't in the library");
          w = fv::nif::loadWearable(*found, options, {});
        }
        else
          w = fv::dson::loadDson(*store, file, options);
        const std::string             morphDir = inputs.morphDir;
        const fv::wardrobe::FitFigure figure{*inputs.scene, *inputs.rig, inputs.bodyMesh, morphDir.substr(0, morphDir.rfind("/Morphs"))};
        // each item's meshes as made: their clearance from the body they were shaped around
        struct Made
        {
          std::string                           label, source, body;
          std::vector<std::vector<std::string>> regions;
          std::vector<std::vector<double>>      clearance;
          std::vector<std::vector<float>>       cages;  // per mesh, its cage as made (empty: not measured)
          std::vector<fv::mesh::SkinInput>      skins;  // per mesh, its skin as made
        };
        std::vector<Made> made;
        for(const fv::dson::Item& item : w.items)
        {
          Made       m{item.label, "", {}, {}};
          const auto gens = fv::wardrobe::generationsOf(w, item, rel, figure);
          if(gens)
            m.source = fv::wardrobe::label(gens->first);
          const auto base = w.sourceBase ? w.sourceBase : gens ? fv::wardrobe::loadBaseCage(*store, gens->first, fv::wardrobe::genderHint(rel)) : nullptr;
          if(base)
          {
            // the body's extent, which tells a carried body cut to what shows from a whole one
            float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
            for(size_t v = 0; v + 2 < base->positions.size(); v += 3)
              for(size_t i = 0; i < 3; i++)
                lo[i] = std::min(lo[i], base->positions[v + i]), hi[i] = std::max(hi[i], base->positions[v + i]);
            std::ostringstream b;
            b << std::fixed << std::setprecision(0) << (w.sourceBase ? "carried body, " : "base cage, ") << base->positions.size() / 3 << " vertices, x "
              << lo[0] << ".." << hi[0] << " y " << lo[1] << ".." << hi[1] << " z " << lo[2] << ".." << hi[2];
            m.body = b.str();
          }
          for(int mesh : item.meshes)
          {
            const fv::Mesh& me = w.scene.meshes[size_t(mesh)];
            // a mesh with no skin (a button hung from a node of its own) is in its node's frame, and is
            // placed, not fitted: left out
            const bool skinned = me.geo && me.cageSkin.start.size() == me.geo->cagePositions.size() / 3 + 1;
            m.regions.push_back(cageRegions(w.scene, me));
            m.clearance.push_back(base && skinned ? signedClearance(base->positions, base->tris, me.geo->cagePositions) : std::vector<double>{});
            m.cages.push_back(base && skinned ? me.geo->cagePositions : std::vector<float>{});
            m.skins.push_back(base && skinned ? me.cageSkin : fv::mesh::SkinInput{});
          }
          // a body the outfit carries (cut to what shows): how much of the item has that skin under it,
          // its closest point within 5 cm and not on the body's cut, by region
          if(w.sourceBase && !w.sourceBase->tris.empty())
          {
            const fv::bodymap::Body          b = fv::bodymap::bodyOf(*w.sourceBase);
            const fv::bodymap::CutEdges      cut(b);
            const fv::physics::SurfaceIndex  index(b.positions, b.tris);
            fv::physics::Stamps&             stamps = fv::physics::threadStamps();
            std::map<std::string, std::array<size_t, 2>> byRegion;  // covered, all
            size_t                           covered = 0, all = 0;
            for(size_t j = 0; j < item.meshes.size(); j++)
            {
              const fv::Mesh& me = w.scene.meshes[size_t(item.meshes[j])];
              if(!me.geo || m.clearance[j].empty())
                continue;
              const auto& p = me.geo->cagePositions;
              for(size_t v = 0; v * 3 + 2 < p.size(); v++)
              {
                const int32_t id  = stamps.next(index.triangles());
                const auto    hit = index.nearest(p[v * 3], p[v * 3 + 1], p[v * 3 + 2], [](size_t) { return true; }, 25.0, stamps.stamp, id);
                const double  w3[3] = {hit.u, hit.v, hit.w};
                const bool    under = hit.tri >= 0 && !cut.on(size_t(hit.tri), w3);
                const std::string& region = v < m.regions[j].size() ? m.regions[j][v] : std::string("?");
                byRegion[region][0] += under, byRegion[region][1]++;
                covered += under, all++;
              }
            }
            std::ostringstream c;
            c << std::fixed << std::setprecision(0) << "carried skin under " << (all ? 100.0 * double(covered) / double(all) : 0.0) << "% of its " << all
              << " vertices:";
            for(const auto& [region, n] : byRegion)
              c << " " << region << " " << (n[1] ? 100.0 * double(n[0]) / double(n[1]) : 0.0) << "% of " << n[1];
            m.body += "; " + c.str();
          }
          made.push_back(std::move(m));
        }
        // worn: the same meshes, fitted, against the figure's body
        auto wearInputs      = loaded->wearInputs();
        wearInputs.commonFit = step.value("fit", std::string()) == "common";
        const fv::WearResult worn = loaded->commitWear(loaded->prepareWear(rel, fv::WearQuality{}, wearInputs), rel);
        loaded->settle();
        const fv::SceneGraph&      scene = loaded->loaded.scene;
        const auto&                ctx   = loaded->loaded.morphs;
        const fv::mesh::BuiltMesh& body  = *scene.meshes[size_t(ctx.meshes[ctx.mainMeshes.at(0)].mesh)].geo;
        Json                       items = Json::array();
        for(size_t k = 0; k < worn.items.size() && k < made.size(); k++)
        {
          const fv::WornItem* item = loaded->item(worn.items[k]);
          if(!item)
            continue;
          std::vector<std::string> regions;
          std::vector<double>      before, after;
          // each cage edge's length fitted over its length as made: the fit's stretch along the cloth,
          // which the clearance (off the skin only) doesn't see
          std::vector<double> stretch;
          struct Edge
          {
            double      ratio, made;
            std::string region;
            float       at[3];
          };
          std::vector<Edge> long_;  // those over 3 mm as made
          // its hard pieces (garment::hardPieces, of the cage and skin as made): how far the fit left each
          // vertex of one from where the piece's made shape, moved as a whole, would have it
          size_t              pieceCount = 0;
          std::vector<double> bent;
          // as cloth (figure.hpp, clothSettings): its triangles that pair back into quads (garment::pairQuads),
          // and of an outfit pinned by the skin (WornItem::pinBySkin), the vertices held, eased and left cloth
          size_t triangles = 0, paired = 0, pinHeld = 0, pinEased = 0, pinFree = 0;
          for(size_t j = 0; j < item->meshes.size() && j < made[k].clearance.size(); j++)
          {
            const fv::Mesh& me = scene.meshes[size_t(item->meshes[j])];
            if(!me.geo || made[k].clearance[j].size() != me.geo->cagePositions.size() / 3)
              continue;
            if(me.geo->cageLineStart.empty())
            {
              const fv::garment::Polys polys = fv::garment::pairQuads(me.geo->cagePositions, me.geo->cagePolyStart, me.geo->cagePolyVerts);
              triangles += polys.triangles;
              paired += polys.paired;
            }
            if(item->pinBySkin)
              for(const fv::WornItem::Follow& f : item->follows)
                if(f.mesh == item->meshes[j] && f.binding)
                  for(const float free : fv::clothPins(*me.geo, me.cageSkin, body.cagePositions, *f.binding))
                    (free <= 0 ? pinHeld : free >= 1 ? pinFree : pinEased)++;
            const std::vector<float>& was = made[k].cages[j];
            const std::vector<float>& is  = me.geo->cagePositions;
            if(was.size() == is.size() && me.geo->cageLineStart.empty())
            {
              const fv::garment::Pieces pieces = fv::garment::hardPieces(was, me.geo->dsonTris, made[k].skins[j]);
              const std::vector<float>  whole  = fv::garment::keepPieces(was, is, pieces);
              pieceCount += size_t(pieces.count);
              for(size_t v = 0; v < pieces.of.size(); v++)
                if(pieces.of[v] >= 0)
                  bent.push_back(std::sqrt(double(whole[v * 3] - is[v * 3]) * (whole[v * 3] - is[v * 3])
                                           + double(whole[v * 3 + 1] - is[v * 3 + 1]) * (whole[v * 3 + 1] - is[v * 3 + 1])
                                           + double(whole[v * 3 + 2] - is[v * 3 + 2]) * (whole[v * 3 + 2] - is[v * 3 + 2])));
            }
            if(was.size() == is.size())
              for(size_t t = 0; t + 2 < me.geo->dsonTris.size(); t += 3)
                for(size_t e = 0; e < 3; e++)
                {
                  const size_t a = me.geo->dsonTris[t + e], b = me.geo->dsonTris[t + (e + 1) % 3];
                  if(a >= b || b * 3 + 2 >= is.size())
                    continue;
                  double l0 = 0, l1 = 0;
                  for(size_t c = 0; c < 3; c++)
                  {
                    l0 += double(was[a * 3 + c] - was[b * 3 + c]) * (was[a * 3 + c] - was[b * 3 + c]);
                    l1 += double(is[a * 3 + c] - is[b * 3 + c]) * (is[a * 3 + c] - is[b * 3 + c]);
                  }
                  if(l0 > 1e-8)
                    stretch.push_back(std::sqrt(l1 / l0));
                  if(l0 > 0.09)
                    long_.push_back({std::sqrt(l1 / l0), std::sqrt(l0), a < made[k].regions[j].size() ? made[k].regions[j][a] : std::string("?"),
                                     {was[a * 3], was[a * 3 + 1], was[a * 3 + 2]}});
                }
            const std::vector<double> now = signedClearance(body.cagePositions, body.dsonTris, me.geo->cagePositions);
            regions.insert(regions.end(), made[k].regions[j].begin(), made[k].regions[j].end());
            before.insert(before.end(), made[k].clearance[j].begin(), made[k].clearance[j].end());
            after.insert(after.end(), now.begin(), now.end());
          }
          const Json change = clearanceChange(regions, before, after);
          items.push_back({{"label", item->label}, {"source", made[k].source}, {"regions", change}});
          std::cerr << "[clearance] " << item->label << " (" << (made[k].source.empty() ? std::string("no source body") : made[k].source) << "; "
                    << made[k].body << ")\n"
                    << std::fixed << std::setprecision(2);
          for(const auto& [region, st] : change.items())
          {
            const Json& a = st.contains("band") ? st["band"] : st["all"];
            std::cerr << "  " << std::left << std::setw(10) << region << std::right << " n " << std::setw(6) << a["n"].get<size_t>() << "  source "
                      << std::setw(6) << a["source"].get<double>() << "  figure " << std::setw(6) << a["figure"].get<double>() << "  change "
                      << std::setw(6) << a["change"].get<double>() << " [" << a["changeP05"].get<double>() << ", " << a["changeP95"].get<double>() << "]"
                      << (st.contains("band") ? "" : " (none in the band)") << "\n";
          }
          if(triangles)
          {
            std::cerr << "  quads      " << paired << " of " << triangles << " triangles pair (" << 100.0 * double(paired) / double(triangles) << "%)\n";
            items.back()["quads"] = {{"triangles", triangles}, {"paired", paired}};
          }
          if(pinHeld + pinEased + pinFree)
          {
            std::cerr << "  pinned     " << pinHeld << " held, " << pinEased << " eased, " << pinFree << " cloth\n";
            items.back()["pinned"] = {{"held", pinHeld}, {"eased", pinEased}, {"cloth", pinFree}};
          }
          if(!bent.empty())
          {
            std::sort(bent.begin(), bent.end());
            std::cerr << "  pieces     n " << std::setw(6) << pieceCount << "  of " << bent.size() << " vertices, bent by " << bent[bent.size() / 2] << " cm median, "
                      << bent[std::min(bent.size() - 1, size_t(0.95 * double(bent.size())))] << " p95, " << bent.back() << " most\n";
            items.back()["pieces"] = {{"n", pieceCount}, {"vertices", bent.size()}, {"median", bent[bent.size() / 2]}, {"max", bent.back()}};
          }
          if(!stretch.empty())
          {
            std::sort(stretch.begin(), stretch.end());
            const auto   at    = [&](double q) { return stretch[std::min(stretch.size() - 1, size_t(q * double(stretch.size())))]; };
            const size_t over  = size_t(stretch.end() - std::upper_bound(stretch.begin(), stretch.end(), 1.25));
            const size_t under = size_t(std::lower_bound(stretch.begin(), stretch.end(), 0.8) - stretch.begin());
            std::cerr << "  edges      n " << std::setw(6) << stretch.size() << "  stretch " << at(0.5) << " [" << at(0.01) << ", " << at(0.99) << "]  least "
                      << stretch.front() << "  most " << stretch.back() << "  over 1.25: " << over << "  under 0.8: " << under << "\n";
            std::sort(long_.begin(), long_.end(), [](const Edge& x, const Edge& y) { return x.ratio > y.ratio; });
            size_t longOver = 0, longUnder = 0;
            for(const Edge& e : long_)
              longOver += e.ratio > 1.5, longUnder += e.ratio < 0.5;
            std::cerr << "  edges over 3 mm: " << long_.size() << ", stretched past 1.5: " << longOver << ", shrunk under 0.5: " << longUnder << "; worst:";
            for(size_t e = 0; e < long_.size() && e < 6; e++)
              std::cerr << "  " << long_[e].ratio << "x of " << long_[e].made << " cm (" << long_[e].region << " at " << std::setprecision(0) << long_[e].at[0]
                        << " " << long_[e].at[1] << " " << long_[e].at[2] << std::setprecision(2) << ")";
            std::cerr << "\n";
            items.back()["stretch"] = {{"median", at(0.5)}, {"p01", at(0.01)}, {"p99", at(0.99)}, {"min", stretch.front()},
                                       {"max", stretch.back()}, {"over", over},  {"under", under}};
          }
          std::cerr << std::defaultfloat << std::setprecision(6);
        }
        // "poses": [pose preset, ...]: under each, how the cloth near the skin keeps its clearance from the
        // posed body (the cages skinned and morphed as they are drawn, with no collision pass): the change
        // from the rest pose for the vertices within 3 cm of the skin there, and how many end over 3 mm
        // under it. What a garment's skin weights are worth (the plan's step 5).
        if(step.contains("poses") && loaded->pose)
        {
          std::vector<int> meshes{int(ctx.meshes[ctx.mainMeshes.at(0)].mesh)};
          for(int id : worn.items)
            if(const fv::WornItem* item = loaded->item(id))
              for(int m : item->meshes)
                if(scene.meshes[size_t(m)].geo && !scene.meshes[size_t(m)].cageSkin.start.empty())
                  meshes.push_back(m);
          const auto measure = [&] {
            const auto          cages = fv::physics::posedCages(scene, *loaded->morphs, meshes, loaded->loaded.root);
            std::vector<double> out;
            for(size_t k = 1; k < cages.size(); k++)
            {
              const std::vector<double> c = signedClearance(cages[0], body.dsonTris, cages[k]);
              out.insert(out.end(), c.begin(), c.end());
            }
            return out;
          };
          loaded->pose->reset();
          loaded->settle();
          const std::vector<double> rest = measure();
          Json                      posed = Json::object();
          for(const Json& pose : step["poses"])
          {
            const std::string file = pose.get<std::string>();
            loaded->pose->reset();
            loaded->applyPosePreset(store->load(file)->json);
            loaded->settle();
            const std::vector<double> now = measure();
            std::vector<double>       change;
            size_t                    under = 0, underAtRest = 0;
            for(size_t v = 0; v < rest.size() && v < now.size(); v++)
            {
              if(rest[v] > 3)
                continue;
              change.push_back(now[v] - rest[v]);
              under += now[v] < -0.3;
              underAtRest += rest[v] < -0.3;
            }
            if(change.empty())
              continue;
            std::sort(change.begin(), change.end());
            const auto        at   = [&](double q) { return change[std::min(change.size() - 1, size_t(q * double(change.size())))]; };
            const std::string name = std::filesystem::path(file).stem().string();
            std::cerr << "  posed      n " << std::setw(6) << change.size() << std::fixed << std::setprecision(2) << "  change " << at(0.5) << " [" << at(0.05)
                      << ", " << at(0.95) << "]  least " << change.front() << "  most " << change.back() << "  under the skin over 3 mm: " << under << " ("
                      << underAtRest << " at rest)  " << name << "\n"
                      << std::defaultfloat << std::setprecision(6);
            posed[name] = {{"n", change.size()}, {"median", at(0.5)}, {"p05", at(0.05)}, {"p95", at(0.95)}, {"min", change.front()},
                           {"max", change.back()}, {"under", under}, {"underAtRest", underAtRest}};
          }
          loaded->pose->reset();
          loaded->settle();
          items.push_back({{"posed", posed}});
        }
        // off again, so the next garment measured isn't fitted over this one
        for(int id : worn.items)
          loaded->removeItem(id);
        loaded->settle();
        dumps[step.value("label", std::string("clearance"))] = {{"clearance", items}, {"notes", worn.notes}};
      }
      else if(step.contains("fit"))
      {
        if(!loaded)
          throw std::runtime_error("fit before a figure loads");
        dumps[step["fit"].get<std::string>()] = {{"fit", fitReport(*loaded)}};
      }
      else if(step.contains("clip"))
      {
        const float duration = poseAtClip(step);
        std::cerr << "clip " << step["clip"].get<std::string>() << " at " << step.value("time", 0.f) << " s of " << duration << "\n";
      }
      else if(step.contains("retarget"))
      {
        if(!loaded || !loaded->pose)
          throw std::runtime_error("retarget before a figure loads");
        const std::string label = step["retarget"].get<std::string>();
        const std::string rel   = step.at("file").get<std::string>();
        const auto        doc   = store->load(rel);
        const std::string src   = fv::pose::sourceFigureFile(doc->json, rel);
        if(src.empty())
          throw std::runtime_error(rel + ": no Genesis's bones");
        fv::SceneGraph srcScene;
        const auto     srcRig = fv::dson::buildRig(srcScene, store->load(src)->json.at("node_library"), nullptr);
        fv::pose::PoseRig from(srcScene, *srcRig, {});
        const fv::pose::PoseClip own  = fv::pose::loadPoseClip(&from, nullptr, doc->json);
        const auto               t0   = std::chrono::steady_clock::now();
        const fv::pose::PoseClip clip = loaded->loadPoseClip(doc->json, rel);
        const double             ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fv::pose::PoseRig&       to   = *loaded->pose;
        fv::SceneGraph&          toScene = loaded->loaded.scene;
        // each rig's limbs (a bone and its twist bones, by anatomy) and which way one points: from its
        // top bone's joint to its last bone's end, as the retarget aligns them
        const auto limbOf = [](const fv::pose::PoseRig& rig, const std::string& slot, char side) {
          std::vector<const fv::pose::PoseBone*> out;
          for(const auto& pb : rig.bones())
            if(const fv::rig::Name p = fv::rig::parse(pb.name); p.slot == slot && p.side == side)
              out.push_back(&pb);
          return out;
        };
        // a limb's top bone and last: the top the shallowest; the last the deepest that the next limb
        // hangs from (Genesis 8's twist bone, not Genesis 9's at the side), else the top
        const auto ends = [](const fv::SceneGraph& sc, const std::vector<const fv::pose::PoseBone*>& limb) {
          const auto depthOf = [&](const fv::pose::PoseBone* pb) {
            int d = 0;
            for(int n = pb->bone; n >= 0; n = sc.nodes[size_t(n)].parent)
              d++;
            return d;
          };
          const fv::pose::PoseBone* top = limb.front();
          for(const auto* pb : limb)
            if(depthOf(pb) < depthOf(top))
              top = pb;
          const fv::pose::PoseBone* last = top;
          for(const auto* pb : limb)
          {
            bool carries = false;
            for(size_t n = 0; n < sc.nodes.size() && !carries; n++)
              carries = sc.nodes[n].parent == pb->bone && sc.nodes[n].joint
                        && std::none_of(limb.begin(), limb.end(), [&](const fv::pose::PoseBone* q) { return q->bone == int(n); });
            if(carries && depthOf(pb) > depthOf(last))
              last = pb;
          }
          return std::pair{top, last};
        };
        // `v` turned by a bone's world matrix (a DAZ bone at rest has none, so it's the bone's turn)
        const auto turned = [](const fv::Mat4& w, const fv::Vec3& v) {
          return fv::Vec3{w.e[0] * v.x + w.e[4] * v.y + w.e[8] * v.z, w.e[1] * v.x + w.e[5] * v.y + w.e[9] * v.z, w.e[2] * v.x + w.e[6] * v.y + w.e[10] * v.z};
        };
        // which way a limb points: from its top bone's joint to its last bone's end
        const auto way = [&](const fv::SceneGraph& sc, const std::vector<const fv::pose::PoseBone*>& limb, const fv::Mat4& toFigure) {
          const auto [top, last] = ends(sc, limb);
          const fv::Mat4 wt = toFigure * sc.world(top->bone), wl = toFigure * sc.world(last->bone);
          const auto&    j  = sc.nodes[size_t(last->bone)].joint;
          const fv::Vec3 d  = turned(wl, {j->endPoint[0] - j->center[0], j->endPoint[1] - j->center[1], j->endPoint[2] - j->center[2]});
          return fv::normalize(fv::Vec3{wl.e[12], wl.e[13], wl.e[14]} + d - fv::Vec3{wt.e[12], wt.e[13], wt.e[14]});
        };
        struct Limb
        {
          std::string                            name;
          std::vector<const fv::pose::PoseBone*> src, dst;
        };
        std::vector<Limb> limbs;
        for(const char* slot : {"thigh", "shin", "foot", "toes", "collar", "upperarm", "forearm", "hand"})
          for(const char side : {'l', 'r'})
            if(auto a = limbOf(from, slot, side), b = limbOf(to, slot, side); !a.empty() && !b.empty())
              limbs.push_back({std::string(1, side) + ":" + slot, a, b});
        for(const char* slot : {"pelvis", "spine1", "spine2", "spine3", "spine4", "neck1", "neck2", "head"})
          if(auto a = limbOf(from, slot, 0), b = limbOf(to, slot, 0); !a.empty() && !b.empty())
            limbs.push_back({slot, a, b});
        const auto at = [](const fv::SceneGraph& sc, const fv::pose::PoseRig& rig, const std::string& bone) {
          const fv::Mat4 w = sc.world(rig.get(bone)->bone);
          return fv::Vec3{w.e[12], w.e[13], w.e[14]};
        };
        std::map<std::pair<std::string, char>, std::string> srcBy, dstBy;
        for(const auto& pb : from.bones())
          if(const fv::rig::Name p = fv::rig::parse(pb.name); !p.twist)
            srcBy.try_emplace({p.slot, p.side}, pb.name);
        for(const auto& pb : to.bones())
          if(const fv::rig::Name p = fv::rig::parse(pb.name); !p.twist)
            dstBy.try_emplace({p.slot, p.side}, pb.name);
        const auto apply = [](fv::pose::PoseRig& rig, const fv::pose::PoseClip& c, float t) {
          std::vector<fv::pose::ChannelValue> values;
          for(const auto& ch : c.channels)
            if(ch.target.kind == fv::pose::PresetTarget::Kind::eBone)
              values.push_back({ch.target.bone, ch.target.channel, ch.target.axis, fv::pose::sample(ch, t)});
          rig.setMany(values);
        };
        std::map<std::string, std::pair<double, double>> err;  // most, sum
        double lowSrc[2] = {1e9, -1e9}, lowDst[2] = {1e9, -1e9};
        int    frames = 0;
        const fv::Mat4 toRoot = fv::inverse(toScene.world(loaded->loaded.root));
        // how far apart each limb points with both figures at rest: the two builds
        from.reset();
        to.reset();
        Json                  restOut = Json::object();
        std::vector<fv::Vec3> restWay;
        for(const Limb& l : limbs)
        {
          restWay.push_back(way(toScene, l.dst, toRoot));
          restOut[l.name] = r(std::acos(std::clamp(fv::dot(way(srcScene, l.src, fv::Mat4{}), restWay.back()), -1.0, 1.0)) * 180 / 3.14159265358979);
        }
        for(float t = 0; t <= clip.duration + 1e-4f; t += 1.f / 30, frames++)
        {
          apply(from, own, t);
          apply(to, clip, t);
          // each limb against its own rest direction turned as the source limb is turned (not for an
          // arm re-aimed from another default pose, which isn't among these figures)
          for(size_t i = 0; i < limbs.size(); i++)
          {
            const Limb&    l  = limbs[i];
            const fv::Vec3 ds = fv::normalize(turned(srcScene.world(ends(srcScene, l.src).second->bone), restWay[i]));
            const fv::Vec3 dt = way(toScene, l.dst, toRoot);
            const double   a  = std::acos(std::clamp(fv::dot(ds, dt), -1.0, 1.0)) * 180 / 3.14159265358979;
            auto&          e  = err[l.name];
            e.first           = std::max(e.first, a);
            e.second += a;
          }
          // the lower ankle's height, each figure in its own centimetres
          const auto ankle = [&](bool target, char side) {
            const auto it = (target ? dstBy : srcBy).find({"foot", side});
            return target ? toRoot.transformPoint(at(toScene, to, it->second)).y : at(srcScene, from, it->second).y;
          };
          const double ls = std::min(ankle(false, 'l'), ankle(false, 'r')), ld = std::min(ankle(true, 'l'), ankle(true, 'r'));
          lowSrc[0] = std::min(lowSrc[0], ls), lowSrc[1] = std::max(lowSrc[1], ls);
          lowDst[0] = std::min(lowDst[0], ld), lowDst[1] = std::max(lowDst[1], ld);
        }
        Json limbsOut = Json::object();
        for(const auto& [name, e] : err)
          limbsOut[name] = {{"most", r(e.first)}, {"mean", r(e.second / frames)}};
        // the figure's bones turned (over 1°) where the file holds every counterpart at rest: a face
        // keyed at 0 must stay the figure's own
        std::set<std::pair<std::string, char>> movedSlots;
        for(const auto& ch : own.channels)
          if(ch.target.kind == fv::pose::PresetTarget::Kind::eBone && ch.target.channel == fv::pose::ChannelKind::eRotation
             && std::any_of(ch.values.begin(), ch.values.end(), [](double v) { return std::abs(v) > 0.01; }))
          {
            const fv::rig::Name p = fv::rig::parse(ch.target.bone);
            movedSlots.insert({p.slot, p.side});
          }
        Json held = Json::object();
        for(const auto& ch : clip.channels)
          if(ch.target.kind == fv::pose::PresetTarget::Kind::eBone && ch.target.channel == fv::pose::ChannelKind::eRotation)
            if(const fv::rig::Name p = fv::rig::parse(ch.target.bone); !movedSlots.count({p.slot, p.side}))
              for(double v : ch.values)
                if(std::abs(v) > 1)
                  held[ch.target.bone] = std::max(held.value(ch.target.bone, 0.0), r(std::abs(v)));
        // the channels the figure's own limits stop short (degrees past, at most): where it can't follow
        Json pastLimits = Json::object();
        for(const auto& ch : clip.channels)
          if(ch.target.kind == fv::pose::PresetTarget::Kind::eBone && ch.target.channel == fv::pose::ChannelKind::eRotation)
            if(const fv::pose::PoseBone* pb = to.get(ch.target.bone))
              if(const int a = ch.target.axis[0] - 'x'; a >= 0 && a < 3)
                if(const fv::pose::Axis* axis = pb->axis(fv::pose::ChannelKind::eRotation, a); axis && axis->clamped)
                  for(double v : ch.values)
                    if(const double past = std::max(axis->min - v, v - axis->max); past > 0.1)
                      pastLimits[ch.target.bone + "." + ch.target.axis] = std::max(pastLimits.value(ch.target.bone + "." + ch.target.axis, 0.0), r(past));
        Json dump = {{"from", src},
                     {"channels", clip.channels.size()},
                     {"duration", clip.duration},
                     {"ms", r(ms)},
                     {"limbsDegrees", limbsOut},
                     {"restDegrees", restOut},
                     {"heldButTurnedDegrees", held},
                     {"pastLimitsDegrees", pastLimits},
                     {"lowerAnkleRiseCm", {{"source", r(lowSrc[1] - lowSrc[0])}, {"figure", r(lowDst[1] - lowDst[0])}}}};
        std::cerr << "retarget " << label << ": " << dump.dump() << "\n";
        dumps[label] = dump;
      }
      else if(step.contains("touch"))
      {
        if(!loaded || !loaded->pose)
          throw std::runtime_error("touch before a figure loads");
        const auto& ctx = loaded->loaded.morphs;
        if(ctx.mainMeshes.empty())
          throw std::runtime_error("touch without a figure body");
        const std::string label = step["touch"].get<std::string>();
        const int         body  = ctx.meshes[ctx.mainMeshes[0]].mesh;
        fv::SceneGraph&   scene = loaded->loaded.scene;
        const int         root  = loaded->loaded.root;
        fv::physics::FigurePhysics physics(scene, *loaded->pose, loaded->morphs.get(), body, root);
        fv::physics::Flesh         flesh(scene, *loaded->morphs, body, root);
        if(!flesh.solver() || !physics.valid())
          throw std::runtime_error("the body has no flesh or no ragdoll");
        std::vector<int> parts;
        for(const auto& p : physics.parts())
          parts.push_back(fv::physics::partOfBone(scene, p.node));
        flesh.colliders = [&] {
          fv::physics::FleshColliders c;
          for(size_t k = 0; k < parts.size(); k++)
            c.limbs.push_back({physics.capsuleNow(k), 1.0, parts[k], physics.restCapsule(k)});
          return c;
        };
        fv::physics::FleshSettings s;
        s.on = true;
        flesh.set(s);
        const fv::physics::FleshModel& m = flesh.solver()->model();
        const double                    hipY = loaded->pose->get("hip") ? loaded->pose->get("hip")->shown(fv::pose::ChannelKind::eTranslation, 1) : 0;
        const int    frames = step.value("frames", 120);
        size_t       mostPressed = 0;
        double       deepest = 0, ms = 0;
        Json         byLobe = Json::object();
        for(int f = 0; f < frames; f++)
        {
          const double bob = 2 * std::sin(2 * 3.14159265358979 * 2 * f / 60.0);
          loaded->pose->setMany({{"hip", fv::pose::ChannelKind::eTranslation, "y", hipY + bob}});
          loaded->settle();
          const auto t0 = std::chrono::steady_clock::now();
          flesh.step(1.0 / 60);
          ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          const auto pressed = flesh.solver()->pressed();
          size_t     count   = 0;
          for(size_t p = 0; p < pressed.size(); p++)
            if(pressed[p])
            {
              count++;
              const double* d = &flesh.solver()->offsets()[p * 3];
              deepest         = std::max(deepest, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
              const std::string& lobe = m.lobes[m.lobe[p]].name;
              byLobe[lobe]            = std::max(byLobe.value(lobe, 0), 1);
            }
          mostPressed = std::max(mostPressed, count);
        }
        loaded->pose->setMany({{"hip", fv::pose::ChannelKind::eTranslation, "y", hipY}});
        loaded->settle();
        Json dump = {{"mostPressed", mostPressed}, {"deepestCm", r(deepest)}, {"lobesPressed", byLobe}, {"particles", m.vertex.size()}, {"frameMs", r(ms / frames)}};
        std::cerr << "touch " << label << ": " << dump.dump() << "\n";
        dumps[label] = dump;
      }
      else if(step.contains("contact"))
      {
        if(!loaded)
          throw std::runtime_error("contact before a figure loads");
        const auto& ctx = loaded->loaded.morphs;
        if(ctx.mainMeshes.empty())
          throw std::runtime_error("contact without a figure body");
        const std::string label = step["contact"].get<std::string>();
        const int         body  = ctx.meshes[ctx.mainMeshes[0]].mesh;
        fv::SceneGraph&   scene = loaded->loaded.scene;
        const fv::Mesh&   m     = scene.meshes[size_t(body)];
        const int         root  = loaded->loaded.root;
        // as the pass's job has it: posed and unposed, in the root's space
        const auto                 p0     = std::chrono::steady_clock::now();
        const std::vector<int>     meshes{body};
        const auto                 posed  = fv::physics::posedCages(scene, *loaded->morphs, meshes, root);
        std::vector<float>         scratch;
        const auto                 view   = loaded->morphs->morphedCageView(body, scratch);
        const auto                 worlds = scene.worlds();
        const auto                 nodeOf = fv::physics::nodeOfMesh(scene);
        const fv::Mat4             toRoot = inverse(worlds[size_t(root)]) * (nodeOf[size_t(body)] >= 0 ? worlds[size_t(nodeOf[size_t(body)])] : fv::Mat4{});
        const std::vector<float>   rest   = fv::physics::transformPoints(toRoot, std::vector<float>(view.begin(), view.end()));
        const std::vector<uint8_t> parts  = fv::physics::bodyParts(scene, m);
        const double prepareMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
        fv::physics::ContactInput in;
        in.body      = posed[0];
        in.tris      = m.geo->dsonTris;
        in.polyStart = m.geo->cagePolyStart;
        in.polyVerts = m.geo->cagePolyVerts;
        in.parts     = parts;
        in.rest      = rest;
        const auto   t0     = std::chrono::steady_clock::now();
        const auto   result = fv::physics::resolveContacts(in);
        const double passMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        static const char* const names[] = {"torso", "head", "lUpperArm", "rUpperArm", "lForearm", "rForearm", "lHand", "rHand",
                                            "lThigh", "rThigh", "lShin", "rShin", "lFoot", "rFoot"};
        Json   byPart = Json::object(), sizes = Json::object();
        int    moved  = 0;
        double most   = 0;
        for(size_t v = 0; v < parts.size(); v++)
        {
          const float* a = &posed[0][v * 3];
          const float* b = &result.positions[v * 3];
          const double d = std::sqrt(double(b[0] - a[0]) * (b[0] - a[0]) + double(b[1] - a[1]) * (b[1] - a[1]) + double(b[2] - a[2]) * (b[2] - a[2]));
          const std::string name = parts[v] < std::size(names) ? names[parts[v]] : "?";
          sizes[name]            = sizes.value(name, 0) + 1;
          most                   = std::max(most, d);
          if(d > 0.01)
          {
            moved++;
            if(!byPart.contains(name))
              byPart[name] = {{"vertices", 0}, {"largestCm", 0.0}};
            Json& p        = byPart[name];
            p["vertices"]  = p.value("vertices", 0) + 1;
            p["largestCm"] = r(std::max(p.value("largestCm", 0.0), d));
          }
        }
        Json dump = {{"pressed", result.pressed},  {"patches", result.patches}, {"moved", moved}, {"largestCm", r(most)},
                     {"byPart", byPart},          {"partSizes", sizes},        {"vertices", parts.size()},
                     {"prepareMs", r(prepareMs)}, {"passMs", r(passMs)}};
        std::cerr << "contact " << label << ": " << dump.dump() << "\n";
        dumps[label] = dump;
      }
      else if(step.contains("flesh"))
      {
        if(!loaded)
          throw std::runtime_error("flesh before a figure loads");
        const auto& ctx = loaded->loaded.morphs;
        if(ctx.mainMeshes.empty())
          throw std::runtime_error("flesh without a figure body");
        const int         body = ctx.meshes[ctx.mainMeshes[0]].mesh;
        const std::string label = step["flesh"].get<std::string>();
        fv::physics::Flesh flesh(loaded->loaded.scene, *loaded->morphs, body);
        if(!flesh.solver())
          throw std::runtime_error("the body has no skin for flesh");
        const fv::physics::FleshModel& m    = flesh.solver()->model();
        const std::vector<float>&      cage = loaded->loaded.scene.meshes[size_t(body)].geo->cagePositions;
        Json                           lobes = Json::array();
        for(const fv::physics::FleshModel::Lobe& l : m.lobes)
        {
          double lo[3] = {1e9, 1e9, 1e9}, hi[3] = {-1e9, -1e9, -1e9}, mid[3] = {0, 0, 0}, total = 0;
          float  soft = 0;
          for(uint32_t p : l.particles)
          {
            const float* x = &cage[m.vertex[p] * 3];
            for(int a = 0; a < 3; a++)
            {
              lo[a] = std::min(lo[a], double(x[a]));
              hi[a] = std::max(hi[a], double(x[a]));
              mid[a] += x[a] * m.share[p];
            }
            total += m.share[p];
            soft = std::max(soft, m.share[p]);
          }
          lobes.push_back({{"name", l.name},
                           {"particles", l.particles.size()},
                           {"middle", {r(mid[0] / total), r(mid[1] / total), r(mid[2] / total)}},
                           {"min", {r(lo[0]), r(lo[1]), r(lo[2])}},
                           {"max", {r(hi[0]), r(hi[1]), r(hi[2])}},
                           {"softest", r(soft)}});
        }
        // the skin of the lobes, as a frame makes it
        fv::physics::FleshSettings s;
        s.on = true;
        flesh.set(s);
        const auto skinStart = std::chrono::steady_clock::now();
        for(int i = 0; i < 20; i++)
          flesh.step(1.0 / 60);
        const double skinMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - skinStart).count() / 20;
        // the solve on its own, the skin bobbing
        fv::physics::FleshSolver solver(m);
        solver.set(s);
        std::vector<float> rest;
        for(uint32_t v : m.vertex)
          rest.insert(rest.end(), {cage[v * 3], cage[v * 3 + 1], cage[v * 3 + 2]});
        for(uint32_t v : m.support)
          rest.insert(rest.end(), {cage[v * 3], cage[v * 3 + 1], cage[v * 3 + 2]});
        const int           frames = step.value("frames", 240);
        std::vector<double> most(m.lobes.size(), 0.0);
        double              solveMs = 0;
        for(int f = 0; f < frames; f++)
        {
          std::vector<float> t   = rest;
          const double       bob = 2 * std::sin(2 * 3.14159265358979 * 2 * f / 60.0);
          for(size_t k = 1; k < t.size(); k += 3)
            t[k] += float(bob);
          const auto t0 = std::chrono::steady_clock::now();
          solver.step(t, 1.0 / 60);
          solveMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          for(size_t l = 0; l < m.lobes.size(); l++)
            for(uint32_t p : m.lobes[l].particles)
            {
              const double* d = &solver.offsets()[p * 3];
              most[l]         = std::max(most[l], std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
            }
        }
        for(size_t l = 0; l < m.lobes.size(); l++)
          lobes[l]["largestCm"] = r(most[l]);
        // what is worn over it, following the last frame's offsets (physics-plan step 9): each conforming
        // mesh, whether it is bound near the lobes, whether its posed space is the body's, how far it
        // moves against the body's farthest, and the gather's time
        std::vector<float> bodyOffsets(size_t(m.cageCount) * 3, 0.f), out;
        double             bodyMost = 0;
        for(size_t p = 0; p < m.vertex.size(); p++)
        {
          const double* d = &solver.offsets()[p * 3];
          for(int a = 0; a < 3; a++)
            bodyOffsets[m.vertex[p] * 3ull + a] = float(d[a]);
          bodyMost = std::max(bodyMost, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
        }
        const std::vector<int> nodeOf = fv::physics::nodeOfMesh(loaded->loaded.scene);
        Json                   worn   = Json::array();
        for(const fv::WornItem& w : loaded->items)
          for(const fv::WornItem::Follow& follow : w.follows)
          {
            const fv::Mat4 a = loaded->loaded.scene.world(nodeOf[size_t(body)]), b = loaded->loaded.scene.world(nodeOf[size_t(follow.mesh)]);
            double         apart = 0;
            for(size_t i = 0; i < 16; i++)
              apart = std::max(apart, std::abs(a.e[i] - b.e[i]));
            const bool near = fv::physics::bindsTo(*follow.binding, m.vertex);
            const auto g0   = std::chrono::steady_clock::now();
            fv::physics::followFlesh(*follow.binding, follow.rigidity.get(), bodyOffsets, nullptr, out);
            const double gatherMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
            double       itemMost = 0;
            for(size_t v = 0; v < out.size(); v += 3)
              itemMost = std::max(itemMost, std::sqrt(double(out[v]) * out[v] + double(out[v + 1]) * out[v + 1] + double(out[v + 2]) * out[v + 2]));
            worn.push_back({{"item", w.label},
                            {"vertices", out.size() / 3},
                            {"nearLobes", near},
                            {"spaceApart", r(apart)},
                            {"largestCm", r(itemMost)},
                            {"gatherMs", r(gatherMs)}});
          }
        Json dump = {{"flesh", lobes},
                     {"worn", worn},
                     {"bodyLargestCm", r(bodyMost)},
                     {"particles", m.vertex.size()},
                     {"support", m.support.size()},
                     {"edges", m.edges.size()},
                     {"cage", m.cageCount},
                     {"solveMs", r(solveMs / frames)},
                     {"skinAndSolveMs", r(skinMs)}};
        std::cerr << "flesh " << label << ": " << dump.dump() << "\n";
        dumps[label] = dump;
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
        fv::physics::Cloth d(scene, *loaded->morphs, body, loaded->loaded.root);
        // FV_DFORCE_NO_FLOOR=1 leaves the body's capsule floor out, as the app's switch does, for an A/B
#pragma warning(suppress : 4996)
        if(const char* env = std::getenv("FV_DFORCE_NO_FLOOR"); env && *env && std::atoi(env) != 0)
          d.capsuleFloor = false;
        // FV_DFORCE_PROXY_LIVE=1: a "live" walk meets the body as Live's proxy, as the app's switch has Live do
#pragma warning(suppress : 4996)
        if(const char* env = std::getenv("FV_DFORCE_PROXY_LIVE"); env && *env && std::atoi(env) != 0)
          d.liveProxy = true;
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
        // what's worn much finer than the cloth meets it as a coarse copy; "coarse": false, as it is
        d.coarseColliders = step.value("coarse", true);
        // the cloth simulates as a light copy of itself: "light": true, or the copy's faces' width in cm
        if(step.contains("light"))
        {
          d.lightCloth = !step["light"].is_boolean() || step["light"].get<bool>();
          if(step["light"].is_number())
            d.lightClothEdge = step["light"].get<float>();
        }
        auto p = d.prepared();
        if(!p)
          throw std::runtime_error("nothing worn drapes");
        {
          size_t colliderFaces = 0;
          for(const auto& b : p->bodies)
            if(b.kind == fv::physics::cloth::Surface::Kind::eCollider)
              colliderFaces += b.tris.size() / 3;
          std::cout << "dforce " << label << " colliders: " << colliderFaces << " faces" << std::endl;
        }
        // strand hair: where it ends up against where it started, and a sample of it to look at
        if(!p->hairs.empty())
        {
          Json hairs = Json::array();
          for(auto [mesh, strands] : p->hairs)
          {
            strands.volume = step.value("volume", strands.volume);
            fv::physics::hair::StrandOptions hairOpts;
            hairOpts.spacing       = step.value("spacing", hairOpts.spacing);
            hairOpts.maxFrames     = step.value("frames", hairOpts.maxFrames);
            hairOpts.substeps      = step.value("substeps", hairOpts.substeps);
            size_t colliderTris    = 0;
            for(const auto& b : p->bodies)
              colliderTris += b.tris.size() / 3;
            std::cout << "hair colliders: " << p->bodies.size() << " meshes, " << colliderTris << " tris" << std::endl;
            const auto  hairStart  = std::chrono::steady_clock::now();
            const auto  after      = fv::physics::hair::settle(strands, p->bodies, hairOpts);
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
                             {"stretchMin", stretch.front()}, {"stretchMax", stretch.back()}, {"tipYMedian", tips[tips.size() / 2]}, {"tipYLowest", tips.front()}, {"tipYHighest", tips.back()}, {"bits", bitsOf(after)}});
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
        // experiments: a light copy's bending this many times as compliant
        if(const float soften = step.value("lightBend", 1.f); soften != 1.f)
          for(size_t i = 0; i < p->drapes.size(); i++)
            if(i < p->light.size() && p->light[i])
              for(float& s : p->bodies[size_t(p->drapes[i].second)].bend)
              {
                const float c = std::max(0.01f, std::min(1.f, s));
                s             = 1.f / (1.f + soften * (1.f - c) / c);
              }
        // experiments: a cloth's polygons as its triangles (no shear diagonals, every side a thread), as a
        // light copy's are
        if(step.value("triangles", false))
          for(const auto& [mesh, slot] : p->drapes)
          {
            fv::physics::cloth::Surface& m = p->bodies[size_t(slot)];
            m.polyStart.assign(1, 0);
            m.polyVerts.clear();
            for(size_t t = 0; t + 2 < m.tris.size(); t += 3)
            {
              for(size_t k = 0; k < 3; k++)
                m.polyVerts.push_back(int32_t(m.tris[t + k]));
              m.polyStart.push_back(int32_t(m.polyVerts.size()));
            }
          }
        // experiments: the cloth's own faces left out of its collisions
        if(!step.value("selfCollide", true))
          for(fv::physics::cloth::Surface& m : p->bodies)
            std::fill(m.selfCollide.begin(), m.selfCollide.end(), uint8_t(0));
        if(p->bodies.empty() || p->bodies[0].kind != fv::physics::cloth::Surface::Kind::eCollider)
          throw std::runtime_error("the body is not the first collider");
        // How far the drawn body (eight influences, the dual quaternion share) lies from the four blended
        // linearly at the target pose, which is the body cloth met before physics-plan step 13c; and with
        // "linearBody": true the job meets that body again (every surface's drawn skin left out), for A/B
        Json bodyShift;
        {
          fv::physics::cloth::Surface linear = p->bodies[0];
          linear.skinIndex8.clear();
          linear.skinWeight8.clear();
          linear.dualQuatShare.clear();
          linear.dualQuat = 0;
          if(p->bodies[0].drawnSkin())
          {
            std::vector<float> a(linear.rest.size()), b(linear.rest.size());
            fv::physics::cloth::skin(linear, linear.bones, a.data());
            fv::physics::cloth::skin(p->bodies[0], p->bodies[0].bones, b.data());
            std::vector<double> dist(a.size() / 3);
            for(size_t v = 0; v < dist.size(); v++)
              dist[v] = std::hypot(double(a[v * 3] - b[v * 3]), double(a[v * 3 + 1] - b[v * 3 + 1]), double(a[v * 3 + 2] - b[v * 3 + 2]));
            std::sort(dist.begin(), dist.end());
            const auto at = [&](double q) { return dist.empty() ? 0.0 : dist[size_t(q * double(dist.size() - 1))]; };
            bodyShift = {{"p50", at(0.5)}, {"p90", at(0.9)}, {"p99", at(0.99)}, {"max", at(1.0)}, {"over1cm", size_t(dist.end() - std::upper_bound(dist.begin(), dist.end(), 1.0))}};
          }
        }
        if(step.value("linearBody", false))
          for(fv::physics::cloth::Surface& m : p->bodies)
          {
            m.skinIndex8.clear();
            m.skinWeight8.clear();
            m.dualQuatShare.clear();
            m.dualQuat = 0;
          }
        const std::optional<std::filesystem::path> objDir =
            step.contains("obj") ? std::optional<std::filesystem::path>(root / pathOf(step["obj"].get<std::string>().c_str())) : std::nullopt;
        const auto meshName = [&](int mesh) { return fileSafe(scene.meshes[size_t(mesh)].name); };
        fv::physics::cloth::Options opts;
        opts.air        = 0.15f;
        opts.iterations    = step.value("iterations", opts.iterations);
        opts.stretchSweeps = step.value("stretchSweeps", opts.stretchSweeps);
        opts.substeps      = step.value("substeps", opts.substeps);
        opts.transition    = step.value("transition", opts.transition);
        // the job as the solver takes it, for the GPU cloth bench to replay (physics-plan step 13a)
        if(step.contains("job"))
        {
          fv::physics::cloth::Job job{p->bodies, opts, {}};
          for(const auto& [mesh, slot] : p->drapes)
            job.drapes.emplace_back(slot, scene.meshes[size_t(mesh)].name);
          const auto file = root / pathOf(step["job"].get<std::string>().c_str());
          std::filesystem::create_directories(file.parent_path());
          if(!fv::physics::cloth::writeJob(file, job))
            throw std::runtime_error("could not write the cloth job " + file.string());
        }
        const auto           simStart = std::chrono::steady_clock::now();
        fv::physics::cloth::Island sim(p->bodies, opts);
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
          // inside the body as drawn at the target pose (its posed cage is the drawn skin), whatever body
          // the job met
          report["insideDrawn"] = fv::physics::cloth::insideBody(p->bodies[0], p->bodies[0].posed, out[size_t(slot)]);
          // and deep in it: inside a capsule floor fitted to the body (physics/cloth/capsules.hpp), whether or not the
          // job has one, as insideBody counts only what lies within 3 cm of its nearest face
          if(const auto floor = fv::physics::cloth::capsuleFloor(p->bodies[0]))
            report["insideDeep"] = fv::physics::cloth::insideFloor(*floor, floor->posed, out[size_t(slot)]);
          items.push_back(report);
          if(objDir)
          {
            const fv::physics::cloth::Surface& sm = p->bodies[size_t(slot)];
            writeObj(*objDir / (fileSafe(label) + "_" + meshName(mesh) + ".obj"), out[size_t(slot)], sm);
            writeObj(*objDir / (fileSafe(label) + "_" + meshName(mesh) + "_rest.obj"), sm.rest, sm);
            writeObj(*objDir / (fileSafe(label) + "_" + meshName(mesh) + "_posed.obj"), sm.posed, sm);
            writeSidecar(*objDir / (fileSafe(label) + "_" + meshName(mesh) + ".json"), sim.model(), size_t(slot), sm, out[size_t(slot)], p->bodies[0], out[0]);
          }
        }
        if(objDir)
          writeObj(*objDir / (fileSafe(label) + "_body.obj"), out[0], p->bodies[0]);
        // Every drape and collider as it is, with neither copied: what a light copy's cloth is measured
        // by, as it rides on the copy, and what the drapes are held against below
        const bool lightOn = d.lightCloth, coarseOn = d.coarseColliders;
        d.lightCloth       = false;
        d.coarseColliders  = false;
        const auto full    = d.prepared();
        d.lightCloth       = lightOn;
        d.coarseColliders  = coarseOn;
        // each drape as drawn: a light copy's cloth where it rides, held where it is held to its skin
        std::vector<const fv::physics::cloth::Surface*> drawnMesh(p->drapes.size());
        std::vector<std::vector<float>>                 drawnAt(p->drapes.size());
        for(size_t i = 0; i < p->drapes.size(); i++)
        {
          const size_t slot = size_t(p->drapes[i].second);
          drawnMesh[i]      = &p->bodies[slot];
          drawnAt[i]        = out[slot];
          if(i >= p->light.size() || !p->light[i] || !full)
            continue;
          const fv::physics::cloth::Surface&     fine = full->bodies[slot];
          const fv::physics::cloth::CoarseCloth& copy = *p->light[i];
          std::vector<float>                     at   = fv::physics::cloth::ride(copy, out[slot]);
          const size_t kept = step.value("keepOut", true) ? fv::physics::cloth::keepOut(at, copy.riding.held, p->bodies[0], out[0], 2.f, &copy, &out[slot]) : 0;
          for(size_t v = 0; v < copy.riding.held.size(); v++)
            if(copy.riding.held[v])
              for(size_t c = 0; c < 3; c++)
                at[v * 3 + c] = fine.posed[v * 3 + c];
          // its hem, how far its own edges stretch, how much of it is inside the body
          float minPosed = 1e9f, minOut = 1e9f;
          for(size_t v = 0; v < fine.vertexCount(); v++)
          {
            minPosed = std::min(minPosed, fine.posed[v * 3 + 1]);
            minOut   = std::min(minOut, at[v * 3 + 1]);
          }
          std::vector<std::pair<uint32_t, uint32_t>> edges;
          for(size_t t = 0; t + 2 < fine.tris.size(); t += 3)
            for(size_t k = 0; k < 3; k++)
            {
              const uint32_t a = fine.tris[t + k], b = fine.tris[t + (k + 1) % 3];
              edges.emplace_back(std::min(a, b), std::max(a, b));
            }
          std::sort(edges.begin(), edges.end());
          edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
          const auto length = [](const std::vector<float>& x, uint32_t a, uint32_t b) {
            return std::hypot(double(x[a * 3] - x[b * 3]), double(x[a * 3 + 1] - x[b * 3 + 1]), double(x[a * 3 + 2] - x[b * 3 + 2]));
          };
          std::vector<double> ratio;
          for(const auto& [a, b] : edges)
            if(const double was = length(fine.rest, a, b); was > 1e-6)
              ratio.push_back(length(at, a, b) / was);
          std::sort(ratio.begin(), ratio.end());
          items[i]["light"] = {{"copyVertices", copy.surface.vertexCount()},
                               {"keptOut", kept},
                               {"copyFaces", copy.surface.tris.size() / 3},
                               {"hemDropCm", double(minPosed - minOut)},
                               {"stretchP99", ratio.empty() ? 0.0 : ratio[size_t(0.99 * double(ratio.size() - 1))]},
                               {"stretchMax", ratio.empty() ? 0.0 : ratio.back()},
                               {"insideBody", fv::physics::cloth::insideBody(p->bodies[0], out[0], at)}};
          drawnMesh[i] = &fine;
          drawnAt[i]   = std::move(at);
          if(objDir)
            writeObj(*objDir / (fileSafe(label) + "_" + meshName(p->drapes[i].first) + "_riding.obj"), drawnAt[i], fine);
        }
        // each drape against every worn collider as it is, not its coarse copy: its vertices that ended on
        // the other side of it than they began (a dress through the underwear it met as a copy)
        if(step.value("through", true))
          for(size_t i = 0; full && i < p->drapes.size(); i++)
          {
            Json through = Json::object();
            for(size_t k = 1; k < full->bodies.size(); k++)
              if(full->bodies[k].kind == fv::physics::cloth::Surface::Kind::eCollider)
                through[full->slotMesh[k] >= 0 ? meshName(full->slotMesh[k]) : std::to_string(k)] =
                    fv::physics::cloth::wrongSide(*drawnMesh[i], drawnAt[i], full->bodies[k], full->bodies[k].posed);
            items[i]["through"] = through;
          }
        std::cerr << "dforce " << label << ": " << items.dump() << "\n";
        Json dump = {{"dforce", items}, {"substeps", sim.substepsPerFrame()}, {"transition", r(sim.transitionSeconds())}, {"ms", r(simMs)}};
        if(!bodyShift.is_null())
          dump["bodyShiftCm"] = bodyShift;
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
          const auto     nodeOf    = fv::physics::nodeOfMesh(scene);
          const fv::Mat4 fromWorld = inverse(worlds[size_t(loaded->loaded.root)]);
          Json           applied   = Json::array();
          for(const auto& [mesh, slot] : p->drapes)
          {
            const fv::Mesh&          m    = scene.meshes[size_t(mesh)];
            const std::vector<float> cage = loaded->morphs->morphedCage(mesh);
            const std::vector<float> posed =
                m.skinned() ? fv::physics::skinLocal(cage, fv::physics::cageSkin(m.cageSkin), fv::physics::currentSkin(scene, worlds, nodeOf, mesh)) : cage;
            const fv::Mat4 to = fromWorld * (nodeOf[size_t(mesh)] >= 0 ? worlds[size_t(nodeOf[size_t(mesh)])] : fv::Mat4{});
            const auto     inSpace = fv::physics::transformPoints(to, posed);
            Json           report  = drapeReport(sim.model(), size_t(slot), p->bodies[size_t(slot)], inSpace, p->bodies[0], out[0]);
            report["mesh"]         = m.name;
            applied.push_back(report);
            if(objDir)
              writeObj(*objDir / (fileSafe(label) + "_applied_" + meshName(mesh) + ".obj"), inSpace, p->bodies[size_t(slot)]);
          }
          std::cerr << "dforce " << label << " applied: " << applied.dump() << "\n";
          dump["applied"] = applied;
        }
        // {"then": {"clip": ...} or a list of them}: the drape as Live shows it a moment on, before the
        // next step lands: its deltas were taken at this pose and are drawn at that one, so each vertex
        // is carried there by its own skin (rest + delta skinned anew), against the body there
        const Json thens = !step.contains("then") ? Json::array() : step["then"].is_array() ? step["then"] : Json::array({step["then"]});
        Json       thenReports = Json::array();
        for(size_t k = 0; k < thens.size(); k++)
        {
          poseAtClip(thens[k]);
          const auto later = d.prepared();
          if(!later || later->bodies.size() != p->bodies.size())
            throw std::runtime_error("the pose after the settle prepares another job");
          const std::string tag  = fileSafe(label) + "_then" + std::to_string(k) + "_";
          Json              then = Json::array();
          for(size_t di = 0; di < p->drapes.size(); di++)
          {
            const auto [mesh, slot]                = p->drapes[di];
            const fv::physics::cloth::Surface& was = p->bodies[size_t(slot)];
            const fv::physics::cloth::Surface& now = later->bodies[size_t(slot)];
            if(was.vertexCount() != now.vertexCount())
              throw std::runtime_error("the drape's surface changed with the pose");
            const fv::physics::cloth::SurfaceSkin from(was, was.bones), to(now, now.bones);
            const std::vector<float>&         at = out[size_t(slot)];
            std::vector<float>                shown(at);
            for(size_t v = 0; v < was.vertexCount(); v++)
            {
              fv::Mat4 a, b;
              if(!from.matrix(v, a) || !to.matrix(v, b))
                continue;
              const fv::Vec3 p3 = (b * fv::inverse(a)).transformPoint(fv::Vec3{at[v * 3], at[v * 3 + 1], at[v * 3 + 2]});
              shown[v * 3]      = float(p3.x);
              shown[v * 3 + 1]  = float(p3.y);
              shown[v * 3 + 2]  = float(p3.z);
            }
            Json report = {{"mesh", scene.meshes[size_t(mesh)].name},
                           {"insideDrawn", fv::physics::cloth::insideBody(later->bodies[0], later->bodies[0].posed, shown)}};
            // Live's ride (physics/cloth/ride.hpp), as the app draws the drape between steps: what lies near
            // the body carried with it from this pose to that one, the rest by its own skin. Not for a light
            // copy, whose cloth rides on it (out[slot] is the copy)
            if(!(di < p->light.size() && p->light[di]))
            {
              using Clock    = std::chrono::steady_clock;
              const size_t n = was.vertexCount();
              std::vector<uint8_t> held(n, 0);
              for(size_t v = 0; v < n; v++)
                held[v] = !(v < was.strength.size() && was.strength[v] > 1e-4f) || (was.visible.size() == n && !was.visible[v]);
              const auto                       bindStart = Clock::now();
              const fv::physics::cloth::Ride   ride      = fv::physics::cloth::bindRide(p->bodies[0], out[0], at, held);
              const double                     bindMs    = std::chrono::duration<double, std::milli>(Clock::now() - bindStart).count();
              // a frame's work, the best of five: the body's corners, where the faces carry the drape from
              // the ease's two ends, and back through the drape's own skin as drawn there
              const fv::physics::cloth::SurfaceSkin bodySkin(later->bodies[0], later->bodies[0].bones);
              const fv::Mesh&                       sm     = scene.meshes[size_t(mesh)];
              const fv::physics::CageSkin           cs     = fv::physics::cageSkin(sm.cageSkin);
              const auto                            worlds = scene.worlds();
              const fv::physics::SkinPose           pose   = fv::physics::currentSkin(scene, worlds, fv::physics::nodeOfMesh(scene), mesh);
              const std::vector<float>              rest   = loaded->morphs->morphedCage(mesh);
              const fv::Mat4                        back   = fv::inverse(was.toSpace);
              std::vector<float>                    rode;
              double                                frameMs = INFINITY;
              for(int r = 0; r < 5; r++)
              {
                const auto               start   = Clock::now();
                const std::vector<float> corners = fv::physics::cloth::rideCorners(ride, bodySkin);
                rode                             = fv::physics::cloth::rideAt(ride, corners, ride.local);
                const std::vector<float> from    = fv::physics::cloth::rideAt(ride, corners, ride.local);
                std::vector<float>       local(rode.size());
                for(size_t i = 0; i < ride.vertices.size(); i++)
                {
                  const fv::Vec3 q  = back.transformPoint(fv::Vec3{(rode[i * 3] + from[i * 3]) * 0.5, (rode[i * 3 + 1] + from[i * 3 + 1]) * 0.5, (rode[i * 3 + 2] + from[i * 3 + 2]) * 0.5});
                  local[i * 3]      = float(q.x);
                  local[i * 3 + 1]  = float(q.y);
                  local[i * 3 + 2]  = float(q.z);
                }
                const std::vector<float> deltas = fv::physics::drapeDelta(rest, local, cs, pose, ride.vertices);
                frameMs = std::min(frameMs, std::chrono::duration<double, std::milli>(Clock::now() - start).count());
              }
              std::vector<float> ridden(shown);
              for(size_t i = 0; i < ride.vertices.size(); i++)
              {
                const size_t v = ride.vertices[i];
                for(size_t c = 0; c < 3; c++)
                  ridden[v * 3 + c] += (rode[i * 3 + c] - ridden[v * 3 + c]) * ride.weight[i];
              }
              // the next step's bind between full ones: each vertex found again on its face there
              const auto                     rebindStart = Clock::now();
              const fv::physics::cloth::Ride again       = fv::physics::cloth::bindRide(later->bodies[0], later->bodies[0].posed, ridden, held, &ride);
              const double                   rebindMs    = std::chrono::duration<double, std::milli>(Clock::now() - rebindStart).count();
              report["riding"] = {{"insideDrawn", fv::physics::cloth::insideBody(later->bodies[0], later->bodies[0].posed, ridden)},
                                  {"vertices", ride.vertices.size()},
                                  {"corners", ride.corners.size()},
                                  {"bindMs", bindMs},
                                  {"rebindMs", rebindMs},
                                  {"rebound", again.vertices.size()},
                                  {"frameMs", frameMs}};
              if(objDir)
                writeObj(*objDir / (tag + meshName(mesh) + "_ridden.obj"), ridden, was);
            }
            then.push_back(report);
            if(objDir)
              writeObj(*objDir / (tag + meshName(mesh) + ".obj"), shown, was);
          }
          if(objDir)
            writeObj(*objDir / (tag + "body.obj"), later->bodies[0].posed, later->bodies[0]);
          std::cerr << "dforce " << label << " then " << k << ": " << then.dump() << "\n";
          thenReports.push_back(then);
        }
        if(!thens.empty())
          dump["then"] = thenReports;
        // {"walk": {"clip": rel, "from": s, "every": s, "steps": n[, "seconds": s]}}: Live through a clip as the
        // app steps it when every step comes late (10TEST's walk at a 4K viewport: a step every 90 ms, each
        // simulating 1/30 s and carrying the cloth through the rest, Backend::step's late share), on a CPU
        // simulation of this job settled as above, untied and then tied to the body as Live ties it
        // (physics/cloth/leash.hpp): after each step, how many of each drape's simulated vertices are inside
        // the body as drawn at the pose it was stepped to
        if(step.contains("walk"))
        {
          namespace cloth = fv::physics::cloth;
          const Json& w       = step["walk"];
          const float from    = w.value("from", 0.f), every = w.value("every", 0.09f), seconds = w.value("seconds", opts.frameDt);
          const int   count   = w.value("steps", 20);
          const float late    = every > seconds ? 1.f - seconds / every : 0.f;
          // "live": true walks the job as Live makes it (with FV_DFORCE_PROXY_LIVE, physics/cloth/proxy.hpp: the body
          // met as its floor, what lies on the skin held to it) and counts what is drawn: each drape, or its light
          // copy's cloth, bound where its step leaves it and held out of the body as a proxy job's is. A job that
          // meets the body's faces is walked so too, untied, which its own ties and reach would draw near enough
          const bool liveWalk = w.value("live", false);
          const auto start    = liveWalk ? d.prepared(true) : p;
          if(!start)
            throw std::runtime_error("the walk's job prepares nothing");
          // each step's pose as a job takes it: every slot's bones and rest cage
          std::vector<std::vector<cloth::Surface>> poses;
          for(int k = 1; k <= count; k++)
          {
            poseAtClip(Json{{"clip", w.at("clip")}, {"time", from + every * float(k)}});
            auto later = d.prepared(liveWalk);
            if(!later || later->bodies.size() != start->bodies.size())
              throw std::runtime_error("the walk's pose prepares another job");
            poses.push_back(std::move(later->bodies));
          }
          Json walk = Json::object();
          if(liveWalk)
          {
            using Clock = std::chrono::steady_clock;
            cloth::CpuBackend sim(start->bodies, opts);
            sim.settle();
            const std::vector<int>   across = cloth::facesAcross(start->bodies[0].tris);
            std::vector<cloth::Ride> rides(start->drapes.size());
            Json                     drawn = Json::array(), simulated = Json::array(), deep = Json::array();
            size_t                   drawnMost = 0, drawnTotal = 0, simTotal = 0, deepMost = 0, deepTotal = 0, capsules = 0, drawnHeld = 0, deepHeld = 0;
            const bool               fresh = w.value("fresh", true);
            double                   holdMs = 0;
            std::map<std::string, size_t> insideAt;
            const fv::Mesh*               bodyMesh = start->slotMesh[0] >= 0 ? &scene.meshes[size_t(start->slotMesh[0])] : nullptr;
            const auto                    walkStart = Clock::now();
            for(size_t k = 0; k < poses.size(); k++)
            {
              const std::vector<cloth::Surface>& now = poses[k];
              const std::vector<cloth::Surface>& was = k > 0 ? poses[k - 1] : start->bodies;
              for(size_t slot = 0; slot < now.size(); slot++)
              {
                if(!now[slot].skinned())
                  continue;
                if(now[slot].restLocal != was[slot].restLocal)
                  sim.setRest(slot, now[slot].restLocal);
                sim.setTarget(slot, now[slot].bones);
              }
              sim.step(0, seconds, late);
              const auto         pos = sim.positions();
              std::vector<float> bodyAt(now[0].vertexCount() * 3);
              cloth::skin(now[0], now[0].bones, bodyAt.data());
              const auto floorNow = cloth::capsuleFloor(now[0]);
              capsules            = floorNow ? floorNow->capsules.size() : 0;
              Json drawnRow = Json::array(), simRow = Json::array(), deepRow = Json::array();
              for(size_t di = 0; di < start->drapes.size(); di++)
              {
                const size_t slot = size_t(start->drapes[di].second);
                // as physics::Cloth binds a proxy job's drape where its step leaves it (afresh every step; "fresh":
                // false found again on the faces the last had, the other Live jobs' between their full binds) and
                // holds what it draws out of the body; a light copy's cloth rides on it, kept out of the body where
                // the copy's faces cross a curve of it
                const auto         holdStart = Clock::now();
                const auto&        light     = di < start->light.size() ? start->light[di] : nullptr;
                std::vector<float> drawnAt   = light ? cloth::ride(*light, pos[slot]) : pos[slot];
                if(light)
                  cloth::keepOut(drawnAt, light->riding.held, now[0], pos[0], 2.f, light.get(), &pos[slot]);
                const size_t       nv   = drawnAt.size() / 3;
                const cloth::Ride* hint = !fresh && k % 8 != 0 && !rides[di].empty() ? &rides[di] : nullptr;
                cloth::Ride        ride = cloth::bindRide(now[0], bodyAt, drawnAt, std::vector<uint8_t>(nv, 0), hint);
                const cloth::Leashes l  = cloth::leashAll(ride, light ? nullptr : &now[slot], now[0]);
                std::vector<float>   at;
                for(uint32_t v : l.ride.vertices)
                  at.insert(at.end(), {drawnAt[size_t(v) * 3], drawnAt[size_t(v) * 3 + 1], drawnAt[size_t(v) * 3 + 2]});
                cloth::holdLeashes(l, bodyAt, now[0].tris, across, at);
                std::vector<float> shown = drawnAt;
                for(size_t i = 0; i < l.ride.vertices.size(); i++)
                  for(size_t c = 0; c < 3; c++)
                    shown[size_t(l.ride.vertices[i]) * 3 + c] = at[i * 3 + c];
                holdMs    = std::max(holdMs, std::chrono::duration<double, std::milli>(Clock::now() - holdStart).count());
                rides[di] = std::move(ride);
                std::vector<uint8_t> which;
                const size_t         in = cloth::insideBody(now[0], now[0].posed, shown, &which);
                const size_t         dp = floorNow ? cloth::insideFloor(*floorNow, floorNow->posed, shown) : 0;
                const size_t         is = cloth::insideBody(now[0], now[0].posed, pos[slot]);
                // of what is drawn inside, how much Live's proxy holds to the skin (no strength in the job)
                std::vector<uint8_t> heldHere(nv, 0);
                for(size_t v = 0; v < nv; v++)
                  heldHere[v] = light ? v < light->riding.held.size() && light->riding.held[v]
                                      : v < start->bodies[slot].strength.size() && start->bodies[slot].strength[v] <= 0;
                std::vector<uint8_t> deepWhich(nv, 0);
                if(floorNow)
                  for(size_t v = 0; v < deepWhich.size(); v++)
                    deepWhich[v] = cloth::insideFloor(*floorNow, floorNow->posed, {shown[v * 3], shown[v * 3 + 1], shown[v * 3 + 2]}) > 0;
                for(size_t v = 0; v < which.size() && v < nv; v++)
                {
                  drawnHeld += which[v] && heldHere[v];
                  deepHeld += deepWhich[v] && heldHere[v];
                }
                drawnRow.push_back(in);
                simRow.push_back(is);
                deepRow.push_back(dp);
                drawnMost = std::max(drawnMost, in);
                deepMost  = std::max(deepMost, dp);
                drawnTotal += in;
                simTotal += is;
                deepTotal += dp;
                // where on the body what is drawn inside lies: the strongest bone of its nearest body vertex
                const fv::wardrobe::SurfaceBinding nearest = fv::wardrobe::bindToSurface(bodyAt, now[0].tris, shown);
                for(size_t v = 0; v < which.size(); v++)
                {
                  if(!which[v])
                    continue;
                  const float* wt = &nearest.weights[v * 3];
                  const size_t bv = nearest.verts[v * 3 + size_t(wt[1] > wt[0] && wt[1] >= wt[2] ? 1 : wt[2] > wt[0] ? 2 : 0)];
                  int          most = -1;
                  float        mw   = 0;
                  for(size_t q = 0; q < 4 && bv * 4 + 3 < now[0].skinWeight.size(); q++)
                    if(now[0].skinWeight[bv * 4 + q] > mw)
                      mw = now[0].skinWeight[bv * 4 + q], most = now[0].skinIndex[bv * 4 + q];
                  std::string name = "?";
                  if(bodyMesh && most >= 0 && size_t(most) < bodyMesh->bones.size() && bodyMesh->bones[size_t(most)] >= 0)
                    name = scene.nodes[size_t(bodyMesh->bones[size_t(most)])].name;
                  insideAt[name]++;
                }
              }
              drawn.push_back(drawnRow);
              simulated.push_back(simRow);
              deep.push_back(deepRow);
            }
            const double walkMs = std::chrono::duration<double, std::milli>(Clock::now() - walkStart).count();
            walk["proxy"] = {{"drawn", drawn}, {"drawnMost", drawnMost}, {"drawnTotal", drawnTotal}, {"simulated", simulated}, {"simulatedTotal", simTotal},
                             {"deep", deep}, {"deepMost", deepMost}, {"deepTotal", deepTotal}, {"capsules", capsules}, {"holdMs", holdMs},
                             {"stepMs", walkMs / std::max<size_t>(1, poses.size())}, {"insideAt", insideAt}, {"drawnHeld", drawnHeld}, {"deepHeld", deepHeld}};
          }
          for(const bool tie : liveWalk ? std::vector<bool>{} : std::vector<bool>{false, true})
          {
            using Clock = std::chrono::steady_clock;
            cloth::CpuBackend live(p->bodies, opts);
            live.settle();
            // as physics::Cloth::tieLive: each simulated drape tied to the body where the settle, then each step,
            // left it, found again on its faces but afresh every eighth time
            std::vector<size_t> slots;
            for(const auto& [mesh, slot] : p->drapes)
              slots.push_back(size_t(slot));
            std::vector<cloth::Ride> rides;
            size_t                   ties = 0, ties2 = 0;
            double                   tieMs = 0, retieMs = 0;
            int                      tied  = 0;
            const auto               retie = [&](const std::vector<cloth::Surface>& surfaces, const std::vector<std::vector<float>>& at) {
              const auto   tieStart = Clock::now();
              const bool   fresh    = tied++ % 8 == 0;
              const size_t n        = live.tie(cloth::tieTo(surfaces, 0, slots, at, rides, fresh));
              const double ms       = std::chrono::duration<double, std::milli>(Clock::now() - tieStart).count();
              if(fresh)
                tieMs = std::max(tieMs, ms), ties = std::max(ties, n);
              else
                retieMs = std::max(retieMs, ms), ties2 = std::max(ties2, n);
            };
            if(tie)
              retie(p->bodies, live.positions());
            Json       inside    = Json::array(), deep = Json::array();
            size_t     most      = 0, total = 0, deepMost = 0, deepTotal = 0;
            // of what is inside after a step, by how it was tied as the step began: held (tied, with a
            // backstop), tied without one, and not tied (further than the ride's band, or held to its skin)
            size_t     wasHeld = 0, wasBare = 0, wasLoose = 0;
            // and of the held: gone past their reach by the step's end; inside a face within 3 cm of the one they
            // were tied to, or further off (another part of the body)
            size_t     heldPast = 0, heldNear = 0, heldFar = 0;
            // what holding the drawn drape costs a frame (physics::Cloth::holdLive's body skin and hold): the most
            double     drawnHoldMs = 0;
            const std::vector<int> bodyAcross = cloth::facesAcross(p->bodies[0].tris);
            // where on the body what is inside lies: the bone that moves its nearest body vertex most
            std::map<std::string, size_t> insideAt;
            const fv::Mesh*               bodyMesh = p->slotMesh[0] >= 0 ? &scene.meshes[size_t(p->slotMesh[0])] : nullptr;
            const auto walkStart = Clock::now();
            for(size_t k = 0; k < poses.size(); k++)
            {
              const std::vector<cloth::Surface>& now = poses[k];
              const std::vector<cloth::Surface>& was = k > 0 ? poses[k - 1] : p->bodies;
              for(size_t slot = 0; slot < now.size(); slot++)
              {
                if(!now[slot].skinned())
                  continue;
                if(now[slot].restLocal != was[slot].restLocal)
                  live.setRest(slot, now[slot].restLocal);
                live.setTarget(slot, now[slot].bones);
              }
              // how each drape vertex is tied as the step begins
              std::vector<std::vector<int8_t>> how(p->drapes.size());
              std::vector<std::vector<int>>    leashOf(p->drapes.size());
              std::vector<cloth::Leashes>      leashes(p->drapes.size());
              for(size_t di = 0; di < p->drapes.size(); di++)
              {
                how[di].assign(p->bodies[size_t(p->drapes[di].second)].vertexCount(), 0);
                leashOf[di].assign(how[di].size(), -1);
                if(di < rides.size())
                {
                  leashes[di]            = cloth::leash(rides[di], &now[size_t(p->drapes[di].second)], now[0]);
                  const cloth::Leashes& l = leashes[di];
                  for(size_t i = 0; i < l.ride.vertices.size(); i++)
                  {
                    how[di][l.ride.vertices[i]]     = l.gap[i] >= 0 ? 1 : -1;
                    leashOf[di][l.ride.vertices[i]] = int(i);
                  }
                }
              }
              live.step(0, seconds, late);
              const auto pos = live.positions();
              // deep in the body after the step: inside a capsule floor fitted to it as posed now
              const auto floorNow = cloth::capsuleFloor(now[0]);
              Json       deepRow  = Json::array();
              for(size_t di = 0; di < p->drapes.size(); di++)
              {
                const size_t slot = size_t(p->drapes[di].second);
                const size_t d    = floorNow ? cloth::insideFloor(*floorNow, floorNow->posed, pos[slot]) : 0;
                deepRow.push_back(d);
                deepMost = std::max(deepMost, d);
                deepTotal += d;
              }
              deep.push_back(deepRow);
              if(tie && !rides.empty())
              {
                const auto         holdStart = Clock::now();
                std::vector<float> bodyAt(now[0].vertexCount() * 3);
                cloth::skin(now[0], now[0].bones, bodyAt.data());
                for(size_t di = 0; di < p->drapes.size() && di < rides.size(); di++)
                {
                  const size_t         slot = size_t(p->drapes[di].second);
                  const cloth::Leashes l    = cloth::leash(rides[di], &now[slot], now[0]);
                  std::vector<float>   at;
                  for(uint32_t v : l.ride.vertices)
                    at.insert(at.end(), {pos[slot][size_t(v) * 3], pos[slot][size_t(v) * 3 + 1], pos[slot][size_t(v) * 3 + 2]});
                  cloth::holdLeashes(l, bodyAt, now[0].tris, bodyAcross, at);
                }
                drawnHoldMs = std::max(drawnHoldMs, std::chrono::duration<double, std::milli>(Clock::now() - holdStart).count());
              }
              if(tie)
                retie(now, pos);
              Json       row = Json::array();
              for(size_t di = 0; di < p->drapes.size(); di++)
              {
                const size_t         slot = size_t(p->drapes[di].second);
                std::vector<uint8_t> which;
                const size_t         in = cloth::insideBody(now[0], now[0].posed, pos[slot], &which);
                most                    = std::max(most, in);
                total += in;
                row.push_back(in);
                for(size_t v = 0; v < which.size(); v++)
                  if(which[v])
                    (how[di][v] > 0 ? wasHeld : how[di][v] < 0 ? wasBare : wasLoose)++;
                const fv::wardrobe::SurfaceBinding nearest = fv::wardrobe::bindToSurface(pos[0], now[0].tris, pos[slot]);
                const cloth::Leashes&              l       = leashes[di];
                for(size_t v = 0; v < which.size(); v++)
                {
                  if(!which[v])
                    continue;
                  const float* w   = &nearest.weights[v * 3];
                  const size_t bv  = nearest.verts[v * 3 + size_t(w[1] > w[0] && w[1] >= w[2] ? 1 : w[2] > w[0] ? 2 : 0)];
                  int          most = -1;
                  float        mw   = 0;
                  for(int k = 0; k < 4 && bv * 4 + 3 < now[0].skinWeight.size(); k++)
                    if(now[0].skinWeight[bv * 4 + size_t(k)] > mw)
                      mw = now[0].skinWeight[bv * 4 + size_t(k)], most = now[0].skinIndex[bv * 4 + size_t(k)];
                  std::string name = "?";
                  if(bodyMesh && most >= 0 && size_t(most) < bodyMesh->bones.size() && bodyMesh->bones[size_t(most)] >= 0)
                    name = scene.nodes[size_t(bodyMesh->bones[size_t(most)])].name;
                  insideAt[std::string(how[di][v] > 0 ? "held " : how[di][v] < 0 ? "bare " : "loose ") + name]++;
                }
                for(size_t v = 0; v < which.size(); v++)
                {
                  if(!which[v] || how[di][v] <= 0)
                    continue;
                  const size_t    i = size_t(leashOf[di][v]);
                  const uint32_t* c = &l.ride.at[i * 3];
                  float           corner[3][3];
                  for(int k = 0; k < 3; k++)
                    for(int a = 0; a < 3; a++)
                      corner[k][a] = pos[0][size_t(l.ride.corners[c[k]]) * 3 + size_t(a)];
                  const cloth::RideFrame f  = cloth::rideFrame(corner[0], corner[1], corner[2], &l.ride.bary[i * 3]);
                  const float*           lo = &l.ride.local[i * 3];
                  double                 toRef = 0, toNear = 0;
                  for(int a = 0; a < 3; a++)
                  {
                    const double ref = f.q[a] + lo[0] * f.e1[a] + lo[1] * f.e2[a] + lo[2] * f.n[a];
                    toRef += (pos[slot][v * 3 + size_t(a)] - ref) * (pos[slot][v * 3 + size_t(a)] - ref);
                    double nearAt = 0;
                    for(int k = 0; k < 3; k++)
                      nearAt += nearest.weights[v * 3 + size_t(k)] * pos[0][size_t(nearest.verts[v * 3 + size_t(k)]) * 3 + size_t(a)];
                    toNear += (nearAt - f.q[a]) * (nearAt - f.q[a]);
                  }
                  if(std::sqrt(toRef) > l.reach[i])
                    heldPast++;
                  else if(std::sqrt(toNear) < 3)
                    heldNear++;
                  else
                    heldFar++;
                }
              }
              inside.push_back(row);
            }
            const double walkMs = std::chrono::duration<double, std::milli>(Clock::now() - walkStart).count();
            walk[tie ? "tied" : "untied"] = {{"inside", inside}, {"most", most}, {"total", total}, {"deep", deep}, {"deepMost", deepMost}, {"deepTotal", deepTotal}, {"ties", ties}, {"tiesAgain", ties2}, {"tieMs", tieMs},
                                             {"retieMs", retieMs}, {"stepMs", walkMs / std::max<size_t>(1, poses.size())},
                                             {"insideWas", {{"held", wasHeld}, {"bare", wasBare}, {"loose", wasLoose}}},
                                             {"held", {{"pastReach", heldPast}, {"nearItsFace", heldNear}, {"elsewhere", heldFar}}}, {"insideAt", insideAt}, {"drawnHoldMs", drawnHoldMs}};
          }
          walk["late"] = late;
          std::cerr << "dforce " << label << " walk: " << walk.dump() << "\n";
          dump["walk"] = walk;
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
    // a bare file name has no folder to make (create_directories("") throws)
    if(!outPath.parent_path().empty())
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
