// dForce: the settle drapes cloth over a collider and keeps it outside, the drape is the cage delta
// whose skinning gives the simulated shape, and the cloth settings are read from the scene's
// modifiers. A port of the TypeScript solver and skin tests.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

#include <figure/dforce/dforce.hpp>
#include <figure/dforce/skin.hpp>
#include <figure/dforce/solver.hpp>
#include <figure/dson/dforce.hpp>
#include <figure/morph/morph_set.hpp>

namespace {

using namespace fv;

// An n x n grid of quads in the xz plane at height y, from -n/2 to n/2; faces up
struct Grid
{
  std::vector<float>    pos;
  std::vector<uint32_t> tris;
  std::vector<int32_t>  polyStart{0}, polyVerts;
};

Grid grid(int n, float y)
{
  Grid g;
  for(int z = 0; z <= n; z++)
    for(int x = 0; x <= n; x++)
      g.pos.insert(g.pos.end(), {float(x) - n / 2.f, y, float(z) - n / 2.f});
  const auto at = [&](int x, int z) { return uint32_t(z * (n + 1) + x); };
  for(int z = 0; z < n; z++)
    for(int x = 0; x < n; x++)
    {
      g.tris.insert(g.tris.end(), {at(x, z), at(x, z + 1), at(x + 1, z), at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)});
      g.polyVerts.insert(g.polyVerts.end(), {int32_t(at(x, z)), int32_t(at(x, z + 1)), int32_t(at(x + 1, z + 1)), int32_t(at(x + 1, z))});
      g.polyStart.push_back(int32_t(g.polyVerts.size()));
    }
  return g;
}

dforce::SettleMesh cloth_(const Grid& g);
dforce::SettleMesh cloth(const Grid& g) { return cloth_(g); }
dforce::SettleMesh cloth_(const Grid& g)
{
  dforce::SettleMesh m;
  const size_t       n = g.pos.size() / 3;
  m.rest = m.posed = g.pos;
  m.tris           = g.tris;
  m.polyStart      = g.polyStart;
  m.polyVerts      = g.polyVerts;
  m.kind           = dforce::SettleMesh::Kind::eDynamic;
  m.strength.assign(n, 1.f);
  m.stretch.assign(n, 0.8f);
  m.shear.assign(n, 0.2f);
  m.bend.assign(n, 0.5f);
  m.buckling.assign(n, 0.05f);
  m.bucklingRatio.assign(n, 0.7f);
  m.density.assign(n, 180.f);
  m.contraction.assign(n, 1.f);
  m.damping.assign(n, 0.1f);
  m.friction.assign(n, 0.4f);
  m.offset.assign(n, 0.2f);
  m.layer.assign(n, 0);
  m.selfCollide.assign(n, 1);
  m.collide.assign(n, 1);
  m.visible.assign(n, 1);
  m.gravity = {0, -980, 0};
  return m;
}

}  // namespace

TEST(Dforce, clothFallsOntoAColliderAndStaysOutsideIt)
{
  const Grid body = grid(8, 0);
  const Grid veil = grid(4, 3);
  const auto out  = dforce::settle({dforce::SettleMesh::collider(body.pos, body.pos, body.tris, dforce::kBodyLayer, 0.2f, 0.4f), cloth(veil)});
  ASSERT_EQ(out.size(), 2u);
  // the collider is where it was; the cloth came down to rest above it
  EXPECT_EQ(out[0], body.pos);
  for(size_t v = 0; v < out[1].size() / 3; v++)
  {
    const float y = out[1][v * 3 + 1];
    EXPECT_LT(y, 1.0f) << v;
    EXPECT_GT(y, 0.0f) << v;
  }
}

TEST(Dforce, aPinnedSurfaceStaysOnTheSkinnedPose)
{
  const Grid veil = grid(3, 3);
  auto       m    = cloth(veil);
  m.strength.assign(m.strength.size(), 0.f);
  for(size_t v = 0; v < m.posed.size() / 3; v++)
    m.posed[v * 3] += 1;
  const auto out = dforce::settle({m});
  for(size_t v = 0; v < m.posed.size(); v++)
    EXPECT_NEAR(out[0][v], m.posed[v], 1e-4) << v;
}

TEST(Dforce, aDrapeSkinsBackToTheSimulatedShape)
{
  // one bone turned a quarter about y, and scaled: the blend has a 3x3 that is not a rotation
  Mat4 bone = compose({1, 2, 3}, quatFromEuler(0, M_PI / 2, 0, "XYZ"), {1.5, 1, 1});
  dforce::SkinPose pose;
  pose.bones = {bone};
  mesh::SkinInput in;
  in.start  = {0, 1, 2};
  in.bone   = {0, 0};
  in.weight = {1.f, 1.f};
  const dforce::Top4 skin = dforce::top4(in);
  const std::vector<float> rest{1, 0, 0, 0, 1, 0};
  const auto posed = dforce::skinLocal(rest, skin, pose);
  std::vector<float> simulated = posed;
  simulated[1] -= 0.5f;
  simulated[5] += 0.25f;
  const auto delta = dforce::drapeDelta(rest, simulated, skin, pose);
  std::vector<float> draped = rest;
  for(size_t i = 0; i < draped.size(); i++)
    draped[i] += delta[i];
  const auto back = dforce::skinLocal(draped, skin, pose);
  for(size_t i = 0; i < back.size(); i++)
    EXPECT_NEAR(back[i], simulated[i], 1e-4) << i;
}

TEST(Dforce, readsTheSettingsOfASimulationSurface)
{
  const auto scene = dson::parseDoc(R"({
    "scene": {"modifiers": [
      {"id": "sim", "url": "#dyn", "parent": "#Dress", "extra": [{"type": "studio/modifier/dynamic_simulation"}]},
      {"id": "surf", "url": "#set", "parent": "#Dress-1", "groups": ["Skirt"], "extra": [{"type": "studio/simulation_settings/dynamic_simulation"}],
       "channel": {"id": "Dynamics Strength", "current_value": 0.5}}
    ]},
    "modifier_library": [
      {"id": "dyn", "extra": [{"type": "studio/modifier/dynamic_simulation", "channels": [
        {"channel": {"id": "Simulation Object Type", "current_value": 1}},
        {"channel": {"id": "Freeze Simulation", "current_value": false}}]}]},
      {"id": "set", "extra": [{"type": "studio/simulation_settings/dynamic_simulation", "channels": [
        {"channel": {"id": "Bend Stiffness", "current_value": 0.9}},
        {"channel": {"id": "Collision Layer", "current_value": 2}}]}]}
    ]
  })");
  const auto geo = dson::parseDoc(R"({
    "vertices": {"count": 4, "values": [[0,0,0],[1,0,0],[1,0,1],[0,0,1]]},
    "polylist": {"count": 1, "values": [[0, 0, 0, 1, 2, 3]]},
    "polygon_material_groups": {"count": 1, "values": ["Skirt"]}
  })");
  const auto data = dson::buildDforce(*scene, nullptr, "Dress", "Dress-1", *geo, geo->json, {});
  ASSERT_TRUE(data);
  EXPECT_EQ(data->kind, dson::DforceKind::eDynamic);
  EXPECT_TRUE(data->simulated());
  ASSERT_EQ(data->bend.size(), 4u);
  EXPECT_NEAR(data->bend[0], 0.9f, 1e-6);
  EXPECT_EQ(data->layer[2], 2);
  EXPECT_EQ(data->polyStart.size(), 2u);
  EXPECT_EQ(data->polyVerts.size(), 4u);

  // a node with no simulation modifier is not a simulation object
  EXPECT_FALSE(dson::buildDforce(*scene, nullptr, "Other", "Other-1", *geo, geo->json, {}));
}

TEST(Dforce, aMorphChannelsTargetIsReplacedInPlace)
{
  mesh::BuiltMesh geo;
  geo.position.assign(9, 0.f);
  geo.normal.assign(9, 0.f);
  geo.renderIndex     = {0, 1, 2};
  geo.firstSplit      = {0, 1, 2};
  geo.nextSplit       = {-1, -1, -1};
  geo.renderVertexCount = 3;
  geo.renderTris      = {0, 1, 2};
  geo.dsonVertexCount = 3;
  morph::MorphSet morphs([&](int) { return &geo; });
  const auto make = [](float dy) {
    auto t     = std::make_shared<mesh::RenderTarget>();
    t->indices = {1};
    t->deltas  = {0, dy, 0};
    return std::shared_ptr<const mesh::RenderTarget>(t);
  };
  morph::ChannelOptions opts;
  opts.hidden = true;
  opts.value  = 1.0;
  morphs.replaceCpuTarget("dforce:0", 0, make(2), opts);
  morphs.flush();
  EXPECT_NEAR(geo.position[4], 2, 1e-6);
  morphs.replaceCpuTarget("dforce:0", 0, make(5), opts);
  morphs.flush();
  EXPECT_NEAR(geo.position[4], 5, 1e-6);
  ASSERT_TRUE(morphs.cpuTarget("dforce:0", 0));
  morphs.set("dforce:0", 0);
  morphs.flush();
  EXPECT_NEAR(geo.position[4], 0, 1e-6);
}

namespace {

// A grid as a scene mesh: its built geometry and, for the cloth, dForce settings on every vertex.
fv::Mesh sceneMesh(const Grid& g, bool cloth)
{
  fv::mesh::MeshInput in;
  in.positions = g.pos;
  in.polyStart = g.polyStart;
  in.polyVerts = g.polyVerts;
  in.polyMaterial.assign(g.polyStart.size() - 1, 0);
  fv::Mesh m;
  m.geo = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(in));
  m.materialGroups = {"Default"};
  m.materials      = {0};
  if(cloth)
  {
    const auto s   = dforce::SettleMesh(cloth_(g));
    auto       d   = std::make_shared<fv::dson::DforceData>();
    d->strength    = s.strength;
    d->stretch     = s.stretch;
    d->shear       = s.shear;
    d->bend        = s.bend;
    d->buckling    = s.buckling;
    d->bucklingRatio = s.bucklingRatio;
    d->density     = s.density;
    d->contraction = s.contraction;
    d->damping     = s.damping;
    d->friction    = s.friction;
    d->offset      = s.offset;
    d->layer       = s.layer;
    d->selfCollide = s.selfCollide;
    d->surfaceVisible = s.visible;
    d->surfaceCollide = s.collide;
    d->polyStart   = g.polyStart;
    d->polyVerts   = g.polyVerts;
    m.dforce       = d;
  }
  return m;
}

}  // namespace

TEST(Dforce, settlesInTheFiguresCentimetresWhenTheRootIsScaledToMetres)
{
  // as the host has it: the figure root scaled to metres after the load, a body and a cloth under it
  fv::SceneGraph scene;
  fv::Node       root;
  root.scale = {0.01, 0.01, 0.01};
  const int rootNode = scene.add(root);
  const Grid body = grid(8, 0), veil = grid(4, 3);
  scene.meshes.push_back(sceneMesh(body, false));
  scene.meshes.push_back(sceneMesh(veil, true));
  scene.materials.push_back({});
  for(int i = 0; i < 2; i++)
  {
    fv::Node n;
    n.kind = fv::NodeKind::eMesh;
    n.mesh = i;
    scene.add(n, rootNode);
  }
  morph::MorphSet morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
  dforce::Dforce  d(scene, morphs, 0, rootNode);
  ASSERT_TRUE(d.active());
  d.resimulate();
  bool wrote = false;
  for(int i = 0; i < 6000 && !wrote; i++)
  {
    wrote = d.pump();
    if(!wrote)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(wrote);
  EXPECT_EQ(d.draped(), 1);
  // the drape is a morph on the cloth: it came down from 3 cm to rest just above the body, in cm
  const auto& pos = scene.meshes[1].geo->position;
  for(size_t v = 0; v < pos.size() / 3; v++)
  {
    EXPECT_LT(pos[v * 3 + 1], 1.0f) << v;
    EXPECT_GT(pos[v * 3 + 1], 0.0f) << v;
  }
}
