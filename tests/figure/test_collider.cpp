// The wardrobe collider: a conforming garment that ends inside the posed body is pushed out to a
// small clearance and the push written as a hidden morph; turning it off clears it; the smoothing
// modifier's settings are read from the scene file.

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <set>
#include <thread>

#include <figure/dson/modifiers.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/wardrobe/collider.hpp>
#include <figure/wardrobe/fit/collide.hpp>

namespace {

using namespace fv;

// An n x n grid of quads in the xz plane at height y, from -n/2 to n/2; faces up.
mesh::MeshInput grid(int n, float y)
{
  mesh::MeshInput in;
  for(int z = 0; z <= n; z++)
    for(int x = 0; x <= n; x++)
      in.positions.insert(in.positions.end(), {float(x) - n / 2.f, y, float(z) - n / 2.f});
  const auto at = [&](int x, int z) { return int32_t(z * (n + 1) + x); };
  in.polyStart.push_back(0);
  for(int z = 0; z < n; z++)
    for(int x = 0; x < n; x++)
    {
      in.polyVerts.insert(in.polyVerts.end(), {at(x, z), at(x, z + 1), at(x + 1, z + 1), at(x + 1, z)});
      in.polyStart.push_back(int32_t(in.polyVerts.size()));
      in.polyMaterial.push_back(0);
    }
  return in;
}

// A mesh of the grid; skinned, with every vertex on one bone at the identity.
Mesh sceneMesh(const mesh::MeshInput& in, int bone)
{
  Mesh m;
  m.geo            = std::make_shared<mesh::BuiltMesh>(mesh::build(in));
  m.materialGroups = {"Default"};
  m.materials      = {0};
  if(bone >= 0)
  {
    const size_t n = in.positions.size() / 3;
    m.bones        = {bone};
    m.boneInverses = {Mat4{}};
    m.cageSkin.start.push_back(0);
    for(size_t v = 0; v < n; v++)
    {
      m.cageSkin.bone.push_back(0);
      m.cageSkin.weight.push_back(1.f);
      m.cageSkin.start.push_back(int32_t(v + 1));
    }
  }
  return m;
}

struct Rig
{
  SceneGraph scene;
  int        root = -1;
  int        bone = -1;
  Rig()
  {
    root = scene.add(Node{});
    Node b;
    b.kind = NodeKind::eBone;
    bone   = scene.add(b, root);
    scene.meshes.push_back(sceneMesh(grid(8, 0), -1));      // the body
    scene.meshes.push_back(sceneMesh(grid(4, -0.5f), bone));  // a garment, half a centimetre inside it
    scene.materials.push_back({});
    for(int i = 0; i < 2; i++)
    {
      Node n;
      n.kind = NodeKind::eMesh;
      n.mesh = i;
      scene.add(n, root);
    }
  }
};

bool pumpUntilWritten(wardrobe::Collider& c)
{
  for(int i = 0; i < 6000; i++)
  {
    if(c.pump())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

float minY(const mesh::BuiltMesh& geo)
{
  float y = 1e9f;
  for(size_t v = 0; v < geo.position.size() / 3; v++)
    y = std::min(y, geo.position[v * 3 + 1]);
  return y;
}

}  // namespace

// A garment standing off the body stays there until it is made tighter; tight pulls it to the
// clearance, part of the way at less, and back off when it is let go. What hangs far off stays.
TEST(Collider, pullsATightGarmentOntoTheBody)
{
  Rig r;
  r.scene.meshes[1] = sceneMesh(grid(4, 2), r.bone);
  morph::MorphSet    morphs([&](int m) { return r.scene.meshes[size_t(m)].geo.get(); });
  float              tight = 0;
  wardrobe::Collider c(r.scene, morphs, 0, r.root);
  c.settings = [&] {
    dson::Smoothing s;
    s.tight = tight;
    return std::map<int, dson::Smoothing>{{1, s}};
  };
  const auto& geo  = *r.scene.meshes[1].geo;
  const auto  pass = [&](float amount) {
    tight = amount;
    c.poke();
    return pumpUntilWritten(c);
  };
  EXPECT_NEAR(minY(geo), 2.f, 1e-4f);
  ASSERT_TRUE(pass(1));
  morphs.flush();
  EXPECT_NEAR(minY(geo), 0.05f, 0.02f);
  ASSERT_TRUE(pass(0.5f));
  morphs.flush();
  EXPECT_NEAR(minY(geo), 1.025f, 0.02f);
  ASSERT_TRUE(pass(0));
  morphs.flush();
  EXPECT_NEAR(minY(geo), 2.f, 1e-3f);

  // a garment already at the ordinary clearance still comes down onto the skin
  r.scene.meshes[1] = sceneMesh(grid(4, 0.25f), r.bone);
  morph::MorphSet    closeMorphs([&](int m) { return r.scene.meshes[size_t(m)].geo.get(); });
  wardrobe::Collider close(r.scene, closeMorphs, 0, r.root);
  close.settings = [] {
    dson::Smoothing s;
    s.tight = 1;
    return std::map<int, dson::Smoothing>{{1, s}};
  };
  close.poke();
  ASSERT_TRUE(pumpUntilWritten(close));
  closeMorphs.flush();
  EXPECT_NEAR(minY(*r.scene.meshes[1].geo), 0.05f, 0.02f);

  // a hem hanging free, past the reach
  Rig far;
  far.scene.meshes[1] = sceneMesh(grid(4, 10), far.bone);
  morph::MorphSet    farMorphs([&](int m) { return far.scene.meshes[size_t(m)].geo.get(); });
  wardrobe::Collider farCollider(far.scene, farMorphs, 0, far.root);
  farCollider.settings = [] {
    dson::Smoothing s;
    s.tight = 1;
    return std::map<int, dson::Smoothing>{{1, s}};
  };
  farCollider.poke();
  for(int i = 0; i < 500 && farCollider.busy(); i++)
  {
    farCollider.pump();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  farMorphs.flush();
  EXPECT_NEAR(minY(*far.scene.meshes[1].geo), 10.f, 1e-3f);
}

// The skin a garment covers: under its face and clear of its hems, and only that.
TEST(Collider, findsTheSkinAGarmentCovers)
{
  const mesh::BuiltMesh body = mesh::build(grid(8, 0));
  const mesh::BuiltMesh item = mesh::build(grid(4, 0.5f));
  const auto polyAt = [](int x, int z) { return z * 8 + x; };  // from the grid's corner, x and z 0 to 7
  const auto covered = [&](float reach, float hemClear) {
    const auto polys = wardrobe::coveredPolys(body.cagePositions.empty() ? body.position : body.cagePositions, body.cagePolyStart, body.cagePolyVerts,
                                              item.cagePositions.empty() ? item.position : item.cagePositions, item.dsonTris, reach, hemClear);
    return std::set<int32_t>(polys.begin(), polys.end());
  };
  // the garment is 4 by 4 over the middle, so with a hem clearance of 1 only the 2 by 2 at its centre
  const auto middle = covered(2, 1);
  EXPECT_EQ(middle.size(), 4u);
  EXPECT_TRUE(middle.count(polyAt(3, 3)));
  EXPECT_TRUE(middle.count(polyAt(4, 4)));
  EXPECT_FALSE(middle.count(polyAt(2, 2)));  // under its hem
  EXPECT_FALSE(middle.count(polyAt(0, 0)));  // bare
  // nothing when the fabric is out of reach
  EXPECT_TRUE(covered(0.25f, 1).empty());

  // behind it only: a body over the garment's outside isn't under it
  const mesh::BuiltMesh above = mesh::build(grid(8, 1));
  EXPECT_TRUE(wardrobe::coveredPolys(above.cagePositions.empty() ? above.position : above.cagePositions, above.cagePolyStart, above.cagePolyVerts,
                                     item.cagePositions.empty() ? item.position : item.cagePositions, item.dsonTris, 2, 0).empty());
}

// A drag on skin tight fits as it goes: steps closer together than the pause a pose waits for still
// get passes.
TEST(Collider, aDraggedEditDoesNotWaitForAPause)
{
  Rig r;
  r.scene.meshes[1] = sceneMesh(grid(4, 2), r.bone);
  morph::MorphSet    morphs([&](int m) { return r.scene.meshes[size_t(m)].geo.get(); });
  float              tight = 0.5f;
  wardrobe::Collider c(r.scene, morphs, 0, r.root);
  c.settings = [&] {
    dson::Smoothing s;
    s.tight = tight;
    return std::map<int, dson::Smoothing>{{1, s}};
  };
  bool wrote = false;
  for(int i = 0; i < 2000 && !wrote; i++)
  {
    // sooner than the 120 ms a pose's edit waits for, every time
    if(i % 10 == 0)
      c.pokeNow();
    wrote = c.pump();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(wrote);

  // the same steps as ordinary pokes wait until they stop
  Rig o;
  o.scene.meshes[1] = sceneMesh(grid(4, 2), o.bone);
  morph::MorphSet    otherMorphs([&](int m) { return o.scene.meshes[size_t(m)].geo.get(); });
  wardrobe::Collider other(o.scene, otherMorphs, 0, o.root);
  other.settings = c.settings;
  bool early = false;
  for(int i = 0; i < 30; i++)
  {
    other.poke();
    early = other.pump() || early;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_FALSE(early);
}

TEST(Collider, pushesAGarmentOutOfTheBodyAndClearsItWhenTurnedOff)
{
  Rig             r;
  morph::MorphSet morphs([&](int m) { return r.scene.meshes[size_t(m)].geo.get(); });
  wardrobe::Collider c(r.scene, morphs, 0, r.root);
  c.settings = [] { return std::map<int, dson::Smoothing>{{1, dson::Smoothing{}}}; };
  EXPECT_FALSE(c.busy());
  c.poke();
  EXPECT_TRUE(c.busy());
  ASSERT_TRUE(pumpUntilWritten(c));
  EXPECT_EQ(c.corrected(), 1);
  // the garment is above the skin, by about the clearance, as a hidden morph on its mesh
  const auto& geo = *r.scene.meshes[1].geo;
  EXPECT_GT(minY(geo), 0.1f);
  EXPECT_LT(minY(geo), 0.6f);
  EXPECT_TRUE(morphs.has(wardrobe::collideChannelOf(1)));
  // the body is never moved
  EXPECT_FLOAT_EQ(r.scene.meshes[0].geo->position[1], 0.f);
  // off: the garment shows as it conforms
  c.setEnabled(false);
  morphs.flush();
  EXPECT_EQ(c.corrected(), 0);
  EXPECT_NEAR(minY(geo), -0.5f, 1e-4f);
  // on again: pushed out again
  c.setEnabled(true);
  ASSERT_TRUE(pumpUntilWritten(c));
  EXPECT_GT(minY(geo), 0.1f);
}

namespace {

// A body with two garments inside it: the inner half a centimetre under the skin, the outer three
// millimetres under it, in the mesh order given. Returns the lowest and highest y of each garment
// after a pass, inner then outer.
std::array<std::pair<float, float>, 2> layered(bool innerFirst)
{
  SceneGraph scene;
  const int  root = scene.add(Node{});
  Node       b;
  b.kind         = NodeKind::eBone;
  const int bone = scene.add(b, root);
  scene.meshes.push_back(sceneMesh(grid(8, 0), -1));
  scene.meshes.push_back(sceneMesh(grid(4, innerFirst ? -0.5f : -0.3f), bone));
  scene.meshes.push_back(sceneMesh(grid(4, innerFirst ? -0.3f : -0.5f), bone));
  scene.materials.push_back({});
  for(int i = 0; i < 3; i++)
  {
    Node n;
    n.kind = NodeKind::eMesh;
    n.mesh = i;
    scene.add(n, root);
  }
  morph::MorphSet    morphs([&](int m) { return scene.meshes[size_t(m)].geo.get(); });
  wardrobe::Collider c(scene, morphs, 0, root);
  c.settings = [] { return std::map<int, dson::Smoothing>{{1, dson::Smoothing{}}, {2, dson::Smoothing{}}}; };
  c.poke();
  if(!pumpUntilWritten(c))
    return {};
  const auto range = [&](int mesh) {
    const auto& p = scene.meshes[size_t(mesh)].geo->position;
    float       lo = 1e9f, hi = -1e9f;
    for(size_t v = 0; v < p.size() / 3; v++)
    {
      lo = std::min(lo, p[v * 3 + 1]);
      hi = std::max(hi, p[v * 3 + 1]);
    }
    return std::pair{lo, hi};
  };
  const int inner = innerFirst ? 1 : 2, outer = innerFirst ? 2 : 1;
  return {range(inner), range(outer)};
}

}  // namespace

TEST(Collider, layersItemsInsideOutByTheirClearanceFromTheBody)
{
  // the inner garment ends above the skin, the outer above the inner by about the same clearance,
  // whichever order their meshes come in
  for(bool innerFirst : {true, false})
  {
    const auto [inner, outer] = layered(innerFirst);
    EXPECT_GT(inner.first, 0.1f) << innerFirst;
    EXPECT_LT(inner.second, 0.6f) << innerFirst;
    EXPECT_GT(outer.first, inner.second + 0.1f) << innerFirst;
    EXPECT_LT(outer.first, inner.second + 0.6f) << innerFirst;
  }
}

TEST(Collider, leavesAMeshASimulationOwnsAndOneWithCollisionOff)
{
  Rig             r;
  morph::MorphSet morphs([&](int m) { return r.scene.meshes[size_t(m)].geo.get(); });
  wardrobe::Collider c(r.scene, morphs, 0, r.root);
  dson::Smoothing    off;
  off.enabled = false;
  c.settings  = [off] { return std::map<int, dson::Smoothing>{{1, off}}; };
  c.poke();
  for(int i = 0; i < 300; i++)
  {
    EXPECT_FALSE(c.pump());
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_FALSE(c.busy());
  EXPECT_EQ(c.corrected(), 0);
  c.settings = [] { return std::map<int, dson::Smoothing>{{1, dson::Smoothing{}}}; };
  c.owned    = [](int mesh) { return mesh == 1; };
  c.poke();
  for(int i = 0; i < 300; i++)
  {
    EXPECT_FALSE(c.pump());
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(c.corrected(), 0);
  EXPECT_NEAR(minY(*r.scene.meshes[1].geo), -0.5f, 1e-4f);
}

TEST(Collider, readsTheSmoothingModifierOfAnItem)
{
  const auto scene = dson::parseDoc(R"({
    "scene": {"modifiers": [
      {"id": "smooth", "url": "#sm", "parent": "#Dress", "extra": [{"type": "studio/modifier/smoothing"}],
       "studio_modifier_channels": [{"channel": {"id": "Collision Iterations", "current_value": 5}}]}
    ]},
    "modifier_library": [
      {"id": "sm", "extra": [{"type": "studio/modifier/smoothing", "channels": [
        {"channel": {"id": "Enable Smoothing", "current_value": true}},
        {"channel": {"id": "Smoothing Iterations", "current_value": 4}},
        {"channel": {"id": "Collision Iterations", "current_value": 3}}]}]}
    ]
  })");
  const auto s = dson::parseSmoothing(*scene, nullptr, "Dress");
  ASSERT_TRUE(s);
  EXPECT_TRUE(s->enabled);
  EXPECT_EQ(s->smoothing, 4);
  // the instance's channel wins over the definition's
  EXPECT_EQ(s->collision, 5);
  EXPECT_FALSE(dson::parseSmoothing(*scene, nullptr, "Other"));
}
