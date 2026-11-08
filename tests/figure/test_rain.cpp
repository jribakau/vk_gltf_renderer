// Rain on the skin (figure/physics/rain/rain_sim.hpp): the surface it runs on (shared edges, a built
// mesh's cage and UVs, refined or not), the frame its items are drawn in against the renderer's
// tangents, and the water it leaves: seeded, downhill whichever way down is, still on a flat top, falling
// off what hangs, drying away; and its layers per material group.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <thread>

#include <figure/dson/store.hpp>
#include <figure/figure.hpp>
#include <figure/materials/water.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/physics/rain/rain.hpp>
#include <figure/physics/rain/rain_sim.hpp>

#include "scene_water.hpp"

namespace {

using namespace fv;
using namespace fv::physics;

constexpr double kPi = 3.14159265358979323846;

struct Body
{
  std::vector<float>    positions;  // cm
  std::vector<uint32_t> tris;
  std::vector<float>    uv;
  std::vector<int32_t>  group;

  RainSurface surface() const { return rainSurface(tris, uv, group); }
  RainInput   input(Vec3 down = {0, -1, 0}) const { return {.posed = positions, .down = down}; }
  Vec3        at(uint32_t tri, float b1, float b2) const
  {
    const auto p = [&](int k) {
      const uint32_t v = tris[tri * 3 + uint32_t(k)];
      return Vec3{positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]};
    };
    return p(0) * (1.0 - b1 - b2) + p(1) * b1 + p(2) * b2;
  }
};

// A grid of `cols` by `rows` quads over (u, v) in [0, 1]², placed by `place`, each quad two triangles
// turned so they face along `place`'s u × v; group 0.
template <class Place>
Body grid(int cols, int rows, Place&& place)
{
  Body b;
  for(int r = 0; r <= rows; r++)
    for(int c = 0; c <= cols; c++)
    {
      const Vec3 p = place(double(c) / cols, double(r) / rows);
      b.positions.insert(b.positions.end(), {float(p.x), float(p.y), float(p.z)});
    }
  const auto vertex = [&](int c, int r) { return uint32_t(r * (cols + 1) + c); };
  const auto corner = [&](int c, int r) {
    b.uv.push_back(float(c) / cols);
    b.uv.push_back(float(r) / rows);
  };
  for(int r = 0; r < rows; r++)
    for(int c = 0; c < cols; c++)
    {
      for(auto [cc, rr] : {std::pair{c, r}, {c + 1, r}, {c + 1, r + 1}, {c, r}, {c + 1, r + 1}, {c, r + 1}})
      {
        b.tris.push_back(vertex(cc, rr));
        corner(cc, rr);
      }
      b.group.insert(b.group.end(), {0, 0});
    }
  return b;
}

// A standing cylinder of skin, `radius` and `height` cm, its seam where u comes round to 1 sharing the
// vertices it started with; the lid (group 1) on top.
Body cylinder(double radius = 15, double height = 160, int segments = 32, int rings = 40)
{
  Body       b;
  const auto vertex = [&](int s, int r) { return uint32_t(r * segments + (s % segments)); };
  for(int r = 0; r <= rings; r++)
    for(int s = 0; s < segments; s++)
    {
      const double a = 2 * kPi * s / segments;
      b.positions.insert(b.positions.end(), {float(radius * std::cos(a)), float(height * r / rings), float(-radius * std::sin(a))});
    }
  const auto corner = [&](int s, int r) {
    b.uv.push_back(float(s) / segments);
    b.uv.push_back(float(r) / rings);
  };
  for(int r = 0; r < rings; r++)
    for(int s = 0; s < segments; s++)
    {
      for(auto [ss, rr] : {std::pair{s, r}, {s + 1, r}, {s + 1, r + 1}, {s, r}, {s + 1, r + 1}, {s, r + 1}})
      {
        b.tris.push_back(vertex(ss, rr));
        corner(ss, rr);
      }
      b.group.insert(b.group.end(), {0, 0});
    }
  // the lid: a fan around its centre
  const uint32_t centre = uint32_t(b.positions.size() / 3);
  b.positions.insert(b.positions.end(), {0.f, float(height), 0.f});
  for(int s = 0; s < segments; s++)
  {
    for(uint32_t v : {vertex(s, rings), vertex(s + 1, rings), centre})
    {
      b.tris.push_back(v);
      const float* p = &b.positions[v * 3];
      b.uv.push_back(0.5f + p[0] / float(4 * radius));
      b.uv.push_back(0.5f + p[2] / float(4 * radius));
    }
    b.group.push_back(1);
  }
  return b;
}

// A ball of skin, `radius` cm, centred `lift` cm up.
Body ball(double radius = 12, double lift = 100)
{
  return grid(32, 24, [&](double u, double v) {
    const double theta = kPi * (1 - v), phi = 2 * kPi * u;
    return Vec3{radius * std::sin(theta) * std::cos(phi), lift + radius * std::cos(theta), -radius * std::sin(theta) * std::sin(phi)};
  });
}

RainSettings raining(float amount = 1, float time = 30, uint32_t seed = 1)
{
  return {.on = true, .amount = amount, .time = time, .dried = 0, .seed = seed};
}

TEST(RainSurface, LinksTheEdgesTwoTrianglesShare)
{
  // a quad: 0 1 2 and 0 2 3 share 2-0 (the first's edge 2) and 0-2 (the second's edge 0)
  const RainSurface s = rainSurface({0, 1, 2, 0, 2, 3}, std::vector<float>(12, 0.f), {0, 0});
  EXPECT_EQ(s.across, (std::vector<int32_t>{-1, -1, 3, 2, -1, -1}));
  // a third triangle on that edge meets neither
  const RainSurface three = rainSurface({0, 1, 2, 0, 2, 3, 2, 0, 4}, std::vector<float>(18, 0.f), {0, 0, 0});
  EXPECT_EQ(three.across[2], 3);
  EXPECT_EQ(three.across[6], -1);
}

TEST(RainSurface, FromABuiltMeshItsCageWithItsUvsRefinedOrNot)
{
  // two quads side by side; the second's corner at vertex 1 takes UV 6 across a seam
  mesh::MeshInput in;
  in.positions          = {0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0};
  in.polyStart          = {0, 4, 8};
  in.polyVerts          = {0, 1, 4, 3, 1, 2, 5, 4};
  in.polyMaterial       = {0, 1};
  in.materialCount      = 2;
  in.uvs                = {0, 0, 0.5f, 0, 1, 0, 0, 1, 0.5f, 1, 1, 1, 0.75f, 0.25f};
  in.overrides          = {1, 1, 6};
  in.subdivisionSurface = true;
  for(int level : {0, 1})
  {
    in.level                  = level;
    const mesh::BuiltMesh geo = mesh::build(in);
    ASSERT_EQ(geo.refined(), level > 0);
    const RainSurface s = rainSurface(geo);
    ASSERT_EQ(s.triangles(), 4u) << "level " << level;
    // each corner of the cage, wherever the triangulation put it: its vertex, UV and group
    int seamCorners = 0;
    for(size_t t = 0; t < 4; t++)
      for(int k = 0; k < 3; k++)
      {
        const uint32_t v = s.tris[t * 3 + size_t(k)];
        const float    u = s.uv[t * 6 + size_t(k) * 2], w = s.uv[t * 6 + size_t(k) * 2 + 1];
        if(s.group[t] == 1 && v == 1)
        {
          EXPECT_FLOAT_EQ(u, 0.75f);
          EXPECT_FLOAT_EQ(w, 0.25f);
          seamCorners++;
        }
        else
        {
          EXPECT_FLOAT_EQ(u, in.uvs[v * 2]) << "vertex " << v;
          EXPECT_FLOAT_EQ(w, in.uvs[v * 2 + 1]) << "vertex " << v;
        }
      }
    EXPECT_GE(seamCorners, 1);
    // the quads meet across their shared edge 1-4, whatever the UVs do there
    int links = 0;
    for(size_t e = 0; e < s.across.size(); e++)
      if(s.across[e] >= 0 && s.group[e / 3] != s.group[size_t(s.across[e]) / 3])
        links++;
    EXPECT_EQ(links, 2);
  }
}

TEST(UvToTangent, TheRenderersAxesAndTrueLengths)
{
  // in the xy plane facing +z, u along +x and v along +y at 0.2 m a UV unit: the tangent is +u, the
  // bitangent -v
  std::array<float, 4> m{};
  ASSERT_TRUE(uvToTangent({Vec3{0, 0, 0}, Vec3{0.2, 0, 0}, Vec3{0, 0.2, 0}}, {{{0, 0}, {1, 0}, {0, 1}}}, m));
  EXPECT_NEAR(m[0], 0.2, 1e-6);
  EXPECT_NEAR(m[1], 0, 1e-6);
  EXPECT_NEAR(m[2], 0, 1e-6);
  EXPECT_NEAR(m[3], -0.2, 1e-6);
  // any triangle: an offset between corners in UV is as long in metres as the edge between them
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> r(-1, 1);
  for(int i = 0; i < 50; i++)
  {
    const std::array<Vec3, 3>                p{Vec3{r(rng), r(rng), r(rng)}, Vec3{r(rng), r(rng), r(rng)}, Vec3{r(rng), r(rng), r(rng)}};
    const std::array<std::array<float, 2>, 3> uv{{{r(rng), r(rng)}, {r(rng), r(rng)}, {r(rng), r(rng)}}};
    if(!uvToTangent(p, uv, m))
      continue;
    for(auto [i0, i1] : {std::pair{0, 1}, {1, 2}, {2, 0}})
    {
      const double du = uv[size_t(i1)][0] - uv[size_t(i0)][0], dv = uv[size_t(i1)][1] - uv[size_t(i0)][1];
      const double x = m[0] * du + m[1] * dv, y = m[2] * du + m[3] * dv;
      const double edge = length(p[size_t(i1)] - p[size_t(i0)]);
      EXPECT_NEAR(std::sqrt(x * x + y * y), edge, 1e-3 * std::max(1.0, edge));
    }
  }
}

TEST(UvToTangent, FlippedVIsTheFrameOfFlippedUvs)
{
  const std::array<Vec3, 3>                p{Vec3{0.1, 0.2, 0.05}, Vec3{0.35, 0.1, 0.0}, Vec3{0.2, 0.4, 0.1}};
  const std::array<std::array<float, 2>, 3> uv{{{0.1f, 0.2f}, {0.6f, 0.15f}, {0.3f, 0.7f}}};
  std::array<float, 4>                      m{}, flipped{};
  ASSERT_TRUE(uvToTangent(p, uv, m));
  ASSERT_TRUE(uvToTangent(p, {{{uv[0][0], 1 - uv[0][1]}, {uv[1][0], 1 - uv[1][1]}, {uv[2][0], 1 - uv[2][1]}}}, flipped));
  const mat::WaterLayer layer{.items = {{.a = {0.3f, 0.4f}, .b = {0.35f, 0.45f}, .toTangent = m, .radius = 1e-3f, .height = 5e-4f}}, .damp = 0.5f};
  const mat::WaterLayer turned = mat::flippedV(layer);
  for(int k = 0; k < 4; k++)
    EXPECT_NEAR(turned.items[0].toTangent[size_t(k)], flipped[size_t(k)], 1e-6) << k;
  EXPECT_FLOAT_EQ(turned.items[0].a[1], 0.6f);
  EXPECT_FLOAT_EQ(turned.items[0].b[1], 0.55f);
  // and back again
  const mat::WaterLayer back = mat::flippedV(turned);
  for(int k = 0; k < 4; k++)
    EXPECT_FLOAT_EQ(back.items[0].toTangent[size_t(k)], m[size_t(k)]);
  EXPECT_NEAR(back.items[0].a[1], 0.4f, 1e-6);
}

TEST(SimulateRain, NothingWhenOffOrOnNothing)
{
  const Body b = cylinder();
  RainSettings off = raining();
  off.on           = false;
  const RainResult none = simulateRain(b.surface(), b.input(), off);
  EXPECT_TRUE(none.beads.empty());
  EXPECT_EQ(none.landed, 0);
  // a surface whose vertices aren't the input's
  const RainResult mismatched = simulateRain(b.surface(), {.posed = std::span(b.positions).first(30)}, raining());
  EXPECT_TRUE(mismatched.beads.empty());
}

TEST(SimulateRain, TheSameSeedTheSameWater)
{
  const Body             b = cylinder();
  const RainSurface      s = b.surface();
  const RainResult       one = simulateRain(s, b.input(), raining(0.8f, 20, 3));
  const RainResult       two = simulateRain(s, b.input(), raining(0.8f, 20, 3));
  const RainResult       other = simulateRain(s, b.input(), raining(0.8f, 20, 4));
  ASSERT_GT(one.beads.size(), 50u);
  ASSERT_EQ(one.beads.size(), two.beads.size());
  for(size_t i = 0; i < one.beads.size(); i++)
  {
    EXPECT_EQ(one.beads[i].tri, two.beads[i].tri);
    EXPECT_EQ(one.beads[i].radius, two.beads[i].radius);
  }
  EXPECT_EQ(one.tracks.size(), two.tracks.size());
  EXPECT_FALSE(one.beads.size() == other.beads.size() && one.beads[0].tri == other.beads[0].tri && one.beads[0].b1 == other.beads[0].b1);
}

// Of the tracks, the share whose way runs along `way` (cm, xyz), and how many there are
std::pair<double, size_t> alongShare(const Body& b, const RainResult& r, const Vec3& way)
{
  size_t along = 0, counted = 0;
  for(const RainTrack& t : r.tracks)
  {
    const Vec3 d = b.at(t.tri, t.b1, t.b2) - b.at(t.tri, t.a1, t.a2);
    if(length(d) < 1e-3)
      continue;
    counted++;
    if(dot(d, way) > 0.5 * length(d))
      along++;
  }
  return {counted ? double(along) / counted : 0.0, counted};
}

TEST(SimulateRain, RunsGoDownTheSkinWhicheverWayDownIs)
{
  const Body       b      = cylinder();
  const RainResult upright = simulateRain(b.surface(), b.input(), raining());
  EXPECT_GT(upright.landed, 500);
  EXPECT_GT(upright.runs, 10);
  const auto [down, counted] = alongShare(b, upright, {0, -1, 0});
  EXPECT_GT(counted, 30u);
  EXPECT_GT(down, 0.9);
  // the same body lying along x: what runs, runs along +x, round its side
  const RainResult lying = simulateRain(b.surface(), b.input({1, 0, 0}), raining());
  EXPECT_GT(lying.runs, 10);
  const auto [across, n] = alongShare(b, lying, {1, 0, 0});
  EXPECT_GT(n, 30u);
  EXPECT_GT(across, 0.8);
}

TEST(SimulateRain, AFlatTopKeepsItsBeads)
{
  const Body       flat = grid(20, 20, [](double u, double v) { return Vec3{u * 60 - 30, 100, 30 - v * 60}; });
  const RainResult r    = simulateRain(flat.surface(), flat.input(), raining(1, 40));
  EXPECT_GT(r.beads.size(), 100u);
  EXPECT_EQ(r.runs, 0);
  EXPECT_TRUE(r.tracks.empty());
  // and they merge: nothing ran or fell, so fewer beads than drops landed is drops that met
  EXPECT_LT(r.beads.size(), size_t(r.landed));
  EXPECT_EQ(r.dripped, 0);
}

TEST(SimulateRain, WhatHangsUnderABallFallsOff)
{
  const Body       b = ball();
  const RainResult r = simulateRain(b.surface(), b.input(), raining(1, 60));
  EXPECT_GT(r.runs, 5);
  EXPECT_GT(r.dripped, 0);
}

TEST(SimulateRain, ItDriesAway)
{
  const Body        b = cylinder();
  const RainSurface s = b.surface();
  RainSettings      set = raining(1, 30, 2);
  const RainResult  wet = simulateRain(s, b.input(), set);
  set.dried              = 10;
  const RainResult drying = simulateRain(s, b.input(), set);
  set.dried              = 300;
  const RainResult dry = simulateRain(s, b.input(), set);
  ASSERT_FALSE(wet.beads.empty());
  ASSERT_FALSE(wet.tracks.empty());
  EXPECT_LT(drying.beads.size(), wet.beads.size());
  EXPECT_LT(drying.damp, wet.damp);
  EXPECT_LT(drying.tracks.size() ? drying.tracks[0].height : 0.f, wet.tracks[0].height);
  EXPECT_TRUE(dry.beads.empty());
  EXPECT_TRUE(dry.tracks.empty());
  EXPECT_EQ(dry.damp, 0.f);
}

TEST(WaterLayers, PerGroupInItsUvsAllDamp)
{
  const Body        b = cylinder();
  const RainSurface s = b.surface();
  const RainInput   in = b.input();
  const RainResult  r  = simulateRain(s, in, raining());
  const auto        layers = waterLayers(s, in, r);
  ASSERT_EQ(layers.size(), 2u);
  size_t items = 0;
  for(const auto& [group, layer] : layers)
  {
    EXPECT_FLOAT_EQ(layer.damp, r.damp);
    items += layer.items.size();
    for(const mat::WaterItem& item : layer.items)
    {
      EXPECT_GT(item.radius, 0.f);
      EXPECT_GT(item.height, 0.f);
      for(float c : {item.a[0], item.a[1], item.b[0], item.b[1]})
      {
        EXPECT_GE(c, -1e-4f);
        EXPECT_LE(c, 1.0001f);
      }
    }
  }
  EXPECT_EQ(items, r.beads.size() + r.tracks.size());
  // a bead is where its own triangle's UVs put it: one on the side sits on the side's band
  const RainBead& bead = r.beads[0];
  const auto&     side = layers.at(s.group[bead.tri]);
  EXPECT_TRUE(std::any_of(side.items.begin(), side.items.end(), [&](const mat::WaterItem& i) { return i.a == i.b && i.radius == bead.radius; }));
}

// On a real figure (DAZ_LIBRARY set; RAIN_FIGURE names one, Genesis 9 by default) in its rest pose: the
// pass on the pool lands beads and runs on its skin, and what it took and what the renderer's table
// holds is printed.
TEST(RainOnAFigure, ARealOnesSkinTakesIt)
{
  const char* lib = std::getenv("DAZ_LIBRARY");
  if(!lib || !*lib)
    GTEST_SKIP() << "DAZ_LIBRARY not set";
  const std::string rel   = std::getenv("RAIN_FIGURE") ? std::getenv("RAIN_FIGURE") : "People/Genesis 9/Genesis 9.duf";
  auto              store = std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<std::filesystem::path>{std::filesystem::path(lib)}));
  auto              fig   = fv::Figure::load(store, rel, {});
  ASSERT_TRUE(fig) << rel;
  fig->settle();
  const int body = fig->loaded.morphs.meshes[size_t(fig->loaded.morphs.mainMeshes[0])].mesh;
  Rain      rain(fig->loaded.scene, *fig->morphs, body, fig->loaded.root);
  rain.set(raining(0.6f, 30));
  const auto start = std::chrono::steady_clock::now();
  while(std::chrono::steady_clock::now() - start < std::chrono::seconds(60))
  {
    rain.pump();
    if(!rain.busy() && !rain.layers().empty())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  const double     waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const RainStats& s      = rain.stats();
  std::printf("[rain] %s: %d landed, %d beads, %d tracks, %d runs, %d dripped; %.0f ms from the edit (250 of it the pause)\n", rel.c_str(), s.landed, s.beads,
              s.tracks, s.runs, s.dripped, waited);
  EXPECT_GT(s.beads, 100);
  EXPECT_GT(s.runs, 0);

  // the table the renderer would upload for it
  std::vector<std::pair<uint32_t, const fv::mat::WaterLayer*>> layers;
  uint32_t                                                     slot = 0;
  for(const auto& [group, layer] : rain.layers())
  {
    std::printf("[rain]   group %d (%s): %zu items\n", group, fig->loaded.scene.meshes[size_t(body)].materialGroups[size_t(group)].c_str(), layer.items.size());
    layers.emplace_back(slot++, &layer);
  }
  const auto            packAt = std::chrono::steady_clock::now();
  const rs::WaterTables t      = rs::packWater(slot, layers);
  const double          packMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - packAt).count();
  uint32_t              most   = 0;
  for(const glm::uvec2& c : t.cells)
    most = std::max(most, c.y);
  const size_t bytes = t.grids.size() * sizeof(shaderio::WaterGrid) + t.cells.size() * sizeof(glm::uvec2) + t.refs.size() * 4 + t.items.size() * sizeof(shaderio::WaterItem);
  std::printf("[rain] table: %zu items, %zu cells, %zu refs, at most %u in a cell, %.0f KB, packed in %.1f ms\n", t.items.size(), t.cells.size(), t.refs.size(), most,
              bytes / 1024.0, packMs);
}

}  // namespace
