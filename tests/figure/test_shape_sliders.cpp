// General body sliders: a girth pushes skin straight out from its bones' axes by the cap, whichever
// shape the figure is in and whichever generation named its bones, and the channels behave as any
// other dial for the mesh, a cage-only mesh and what follows the body.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/morph/shape_sliders.hpp>
#include <figure/pose/figure_driver.hpp>
#include <figure/scene/graph.hpp>

namespace {

using fv::morph::GirthRecipe;
using fv::morph::ShapeBone;

constexpr float kRadius = 10.f;
constexpr int   kAround = 8, kRings = 5;

// A tube around the y axis from y = 0 to y = 40, rings 10 apart, as a cage.
fv::mesh::MeshInput tube(float x = 0.f)
{
  fv::mesh::MeshInput in;
  for(int r = 0; r < kRings; r++)
    for(int a = 0; a < kAround; a++)
    {
      const double angle = 2 * std::numbers::pi * a / kAround;
      in.positions.push_back(x + kRadius * float(std::cos(angle)));
      in.positions.push_back(float(r) * 10.f);
      in.positions.push_back(kRadius * float(std::sin(angle)));
    }
  in.polyStart = {0};
  for(int r = 0; r + 1 < kRings; r++)
    for(int a = 0; a < kAround; a++)
    {
      const int next = (a + 1) % kAround;
      in.polyVerts.insert(in.polyVerts.end(), {r * kAround + a, r * kAround + next, (r + 1) * kAround + next, (r + 1) * kAround + a});
      in.polyStart.push_back(int32_t(in.polyVerts.size()));
      in.polyMaterial.push_back(0);
    }
  return in;
}

// Every vertex wholly on one bone.
fv::mesh::SkinInput oneBone(size_t verts, int bone, float weight = 1.f)
{
  fv::mesh::SkinInput skin;
  for(size_t v = 0; v <= verts; v++)
    skin.start.push_back(int32_t(v));
  skin.bone.assign(verts, bone);
  skin.weight.assign(verts, weight);
  return skin;
}

GirthRecipe waist(float cap = 0.2f)
{
  GirthRecipe r;
  r.channel = "fv_shape_waist";
  r.slots   = {"spine1"};
  r.cap     = cap;
  return r;
}

ShapeBone bone(const std::string& name, std::array<double, 3> center, std::array<double, 3> end)
{
  ShapeBone b;
  b.name   = name;
  b.center = center;
  b.end    = end;
  return b;
}

const float* deltaOf(const fv::mesh::CageDeltas& d, int vert)
{
  const auto it = std::find(d.verts.begin(), d.verts.end(), vert);
  return it == d.verts.end() ? nullptr : &d.deltas[size_t(it - d.verts.begin()) * 3];
}

class NoFiles : public fv::pose::FigureSource
{
public:
  std::shared_ptr<const fv::dson::Doc> doc(const std::string&) override { return nullptr; }
  std::vector<fv::dson::MorphDef>      targets(const fv::dson::Doc&, const std::string&) override { return {}; }
  std::optional<std::string>           find(const std::string&) override { return std::nullopt; }
};

// A tube skinned to one bone, with the sliders' parts around it.
struct Body
{
  fv::SceneGraph                       scene;
  std::shared_ptr<fv::mesh::BuiltMesh> geo;
  fv::dson::MeshRef                    ref;
  std::unique_ptr<fv::morph::MorphSet> morphs;
  std::unique_ptr<fv::pose::FigureDriver> driver;

  explicit Body(const std::string& boneName = "spine1")
  {
    geo = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(tube()));
    fv::Node node;
    node.name  = boneName;
    node.kind  = fv::NodeKind::eBone;
    node.joint = std::make_shared<fv::JointData>();
    node.joint->centerAuthored = {0, 0, 0};
    node.joint->center         = {0, 0, 0};
    node.joint->endPoint       = {0, 40, 0};
    const int boneNode         = scene.add(std::move(node));
    fv::Mesh  mesh;
    mesh.name     = "body";
    mesh.geo      = geo;
    mesh.bones    = {boneNode};
    mesh.cageSkin = oneBone(geo->cagePositions.size() / 3, 0);
    scene.meshes.push_back(std::move(mesh));
    ref.mesh = 0;
    ref.geo  = geo;
    morphs   = std::make_unique<fv::morph::MorphSet>([this](int) { return geo.get(); });
    driver   = std::make_unique<fv::pose::FigureDriver>(*morphs, std::make_shared<NoFiles>());
    morphs->driver = driver.get();
  }
  void addSliders() { fv::morph::addShapeSliders(*morphs, *driver, scene, {&ref}); }
};

}  // namespace

TEST(ShapeSliders, AGirthPushesStraightOutFromTheAxisByTheCap)
{
  const fv::mesh::MeshInput in   = tube();
  const size_t              n    = in.positions.size() / 3;
  const auto                d    = fv::morph::girthDeltas(in.positions, oneBone(n, 0), {bone("spine1", {0, 0, 0}, {0, 40, 0})}, waist());
  ASSERT_EQ(d.verts.size(), n);
  for(size_t i = 0; i < n; i++)
  {
    const float* p = &in.positions[i * 3];
    const float* m = &d.deltas[i * 3];
    EXPECT_NEAR(m[0], p[0] * 0.2f, 1e-5);
    EXPECT_NEAR(m[1], 0.f, 1e-5);
    EXPECT_NEAR(m[2], p[2] * 0.2f, 1e-5);
  }
}

TEST(ShapeSliders, ALeaningBonePushesAcrossItsOwnAxis)
{
  const fv::mesh::MeshInput in = tube();
  const auto                d  = fv::morph::girthDeltas(in.positions, oneBone(in.positions.size() / 3, 0), {bone("spine1", {1, 2, 3}, {11, 22, 8})}, waist());
  ASSERT_FALSE(d.verts.empty());
  const std::array<double, 3> axis{10 / 22.9129, 20 / 22.9129, 5 / 22.9129};
  for(size_t i = 0; i < d.verts.size(); i++)
    EXPECT_NEAR(d.deltas[i * 3] * axis[0] + d.deltas[i * 3 + 1] * axis[1] + d.deltas[i * 3 + 2] * axis[2], 0, 1e-4);
}

TEST(ShapeSliders, AWeightScalesTheMoveAndOtherBonesKeepTheirSkin)
{
  const fv::mesh::MeshInput in = tube();
  const size_t              n  = in.positions.size() / 3;
  const std::vector<ShapeBone> bones{bone("hip", {0, 0, 0}, {0, 40, 0}), bone("spine1", {0, 0, 0}, {0, 40, 0})};

  EXPECT_TRUE(fv::morph::girthDeltas(in.positions, oneBone(n, 0), bones, waist()).verts.empty());

  const auto half = fv::morph::girthDeltas(in.positions, oneBone(n, 1, 0.5f), bones, waist());
  ASSERT_EQ(half.verts.size(), n);
  EXPECT_NEAR(half.deltas[0], in.positions[0] * 0.1f, 1e-5);
}

TEST(ShapeSliders, ABoneWithNoLengthMovesNothing)
{
  const fv::mesh::MeshInput in = tube();
  EXPECT_TRUE(fv::morph::girthDeltas(in.positions, oneBone(in.positions.size() / 3, 0), {bone("spine1", {0, 5, 0}, {0, 5, 0})}, waist()).verts.empty());
}

TEST(ShapeSliders, ThighsHoldBackTowardTheMidlineOnly)
{
  GirthRecipe r;
  r.slots  = {"thigh"};
  r.cap    = 0.2f;
  r.inward = 0.25f;
  // a left thigh centred at x = 15: its first vertex is on the far side from the midline, the
  // vertex half way round (at x = 5) on the near side
  const fv::mesh::MeshInput in = tube(15.f);
  const auto                d  = fv::morph::girthDeltas(in.positions, oneBone(in.positions.size() / 3, 0), {bone("l_thigh", {15, 0, 0}, {15, 40, 0})}, r);
  const float*              outer = deltaOf(d, 0);
  const float*              inner = deltaOf(d, kAround / 2);
  ASSERT_TRUE(outer && inner);
  EXPECT_NEAR(outer[0], kRadius * 0.2f, 1e-5);
  EXPECT_NEAR(inner[0], -kRadius * 0.2f * 0.25f, 1e-5);
  // front and back are not toward the midline
  const float* front = deltaOf(d, kAround / 4);
  ASSERT_TRUE(front);
  EXPECT_NEAR(front[2], kRadius * 0.2f, 1e-5);
}

TEST(ShapeSliders, ABellyMovesTheFrontAloneAndFadesInFromTheFlank)
{
  GirthRecipe r = waist();
  r.facing      = 1;
  const fv::mesh::MeshInput in = tube();
  const auto d = fv::morph::girthDeltas(in.positions, oneBone(in.positions.size() / 3, 0), {bone("spine1", {0, 0, 0}, {0, 40, 0})}, r);
  // vertex kAround / 4 faces straight forward (+z), its opposite straight back; the flanks face neither
  const float* front = deltaOf(d, kAround / 4);
  ASSERT_TRUE(front);
  EXPECT_NEAR(front[2], kRadius * 0.2f, 1e-5);
  EXPECT_EQ(deltaOf(d, kAround * 3 / 4), nullptr);
  EXPECT_EQ(deltaOf(d, 0), nullptr);
  // 45 degrees round is past the fade's end
  const float* round = deltaOf(d, kAround / 8);
  ASSERT_TRUE(round);
  EXPECT_NEAR(round[2], in.positions[size_t(kAround / 8) * 3 + 2] * 0.2f, 1e-5);

  r.facing = -1;
  const auto back = fv::morph::girthDeltas(in.positions, oneBone(in.positions.size() / 3, 0), {bone("spine1", {0, 0, 0}, {0, 40, 0})}, r);
  EXPECT_EQ(deltaOf(back, kAround / 4), nullptr);
  ASSERT_TRUE(deltaOf(back, kAround * 3 / 4));
  EXPECT_NEAR(deltaOf(back, kAround * 3 / 4)[2], -kRadius * 0.2f, 1e-5);
}

TEST(ShapeSliders, ARecipeCanWeighTheAxes)
{
  GirthRecipe r = waist();
  r.axes        = {1.f, 1.f, 0.35f};
  const fv::mesh::MeshInput in = tube();
  const auto d = fv::morph::girthDeltas(in.positions, oneBone(in.positions.size() / 3, 0), {bone("spine1", {0, 0, 0}, {0, 40, 0})}, r);
  ASSERT_TRUE(deltaOf(d, 0) && deltaOf(d, kAround / 4));
  EXPECT_NEAR(deltaOf(d, 0)[0], kRadius * 0.2f, 1e-5);
  EXPECT_NEAR(deltaOf(d, kAround / 4)[2], kRadius * 0.2f * 0.35f, 1e-5);
}

TEST(ShapeSliders, AChestPushesItsPectoralsFromTheSpine)
{
  GirthRecipe r;
  r.slots    = {"spine3", "pectoral"};
  r.cap      = 0.2f;
  r.axisSlot = "spine3";
  const fv::mesh::MeshInput in = tube();
  // the skin on a pectoral whose own axis points forward, across the spine's
  const std::vector<ShapeBone> bones{bone("spine3", {0, 0, 0}, {0, 40, 0}), bone("l_pectoral", {3, 5, 8}, {3, 5, 18})};
  const auto d = fv::morph::girthDeltas(in.positions, oneBone(in.positions.size() / 3, 1), bones, r);
  for(size_t i = 0; i < d.verts.size(); i++)
  {
    const float* p = &in.positions[size_t(d.verts[i]) * 3];
    EXPECT_NEAR(d.deltas[i * 3], p[0] * 0.2f, 1e-5);
    EXPECT_NEAR(d.deltas[i * 3 + 1], 0.f, 1e-5);
    EXPECT_NEAR(d.deltas[i * 3 + 2], p[2] * 0.2f, 1e-5);
  }
  EXPECT_FALSE(d.verts.empty());
}

TEST(ShapeSliders, EveryGenerationsBonesAreFound)
{
  const auto& recipes = fv::morph::girthRecipes();
  const auto  recipe  = [&](const std::string& channel) {
    return *std::find_if(recipes.begin(), recipes.end(), [&](const GirthRecipe& r) { return r.channel == channel; });
  };
  const fv::mesh::MeshInput in = tube();
  const size_t              n  = in.positions.size() / 3;
  const auto moves = [&](const std::string& name, const GirthRecipe& r) {
    return !fv::morph::girthDeltas(in.positions, oneBone(n, 0), {bone(name, {0, 0, 0}, {0, 40, 0})}, r).verts.empty();
  };
  struct Case
  {
    const char*              channel;
    std::vector<const char*> yes, no;
  };
  const std::vector<Case> cases{
      {"fv_shape_thighs", {"l_thigh", "l_thightwist1", "l_thightwist2", "r_thigh", "lThighBend", "lThighTwist", "rThighBend", "rThighTwist"}, {"l_shin", "lShin", "pelvis", "hip"}},
      {"fv_shape_waist", {"spine1", "spine2", "abdomenLower", "abdomenUpper"}, {"spine3", "chestLower", "pelvis"}},
      {"fv_shape_chest", {"spine3", "spine4", "chestLower", "chestUpper", "l_pectoral", "rPectoral"}, {"spine2", "abdomenUpper", "l_shoulder", "lCollar"}},
      {"fv_shape_hips", {"pelvis"}, {"hip", "spine1", "l_thigh"}},
      {"fv_shape_neck", {"neck1", "neck2", "neckLower", "neckUpper"}, {"head", "spine4", "chestUpper"}},
      {"fv_shape_upperarms", {"l_upperarm", "l_upperarmtwist1", "r_upperarmtwist2", "lShldrBend", "rShldrTwist"}, {"l_forearm", "l_shoulder", "lCollar", "lForearmBend"}},
      {"fv_shape_forearms", {"l_forearm", "l_forearmtwist1", "r_forearmtwist2", "lForearmBend", "rForearmTwist"}, {"l_upperarm", "lShldrBend", "l_hand", "lHand"}},
      {"fv_shape_calves", {"l_shin", "r_shin", "lShin", "rShin"}, {"l_thigh", "lThighBend", "l_foot", "lFoot"}},
  };
  for(const Case& c : cases)
  {
    for(const char* name : c.yes)
      EXPECT_TRUE(moves(name, recipe(c.channel))) << c.channel << " " << name;
    for(const char* name : c.no)
      EXPECT_FALSE(moves(name, recipe(c.channel))) << c.channel << " " << name;
  }
  // hips and glutes share a bone, but not the same skin of it: a tube's back moves for one and its
  // flank for both
  const GirthRecipe glutes = recipe("fv_shape_glutes");
  const auto        back   = fv::morph::girthDeltas(in.positions, oneBone(n, 0), {bone("pelvis", {0, 0, 0}, {0, 40, 0})}, glutes);
  EXPECT_EQ(deltaOf(back, kAround / 4), nullptr);
  EXPECT_NE(deltaOf(back, kAround * 3 / 4), nullptr);
}

TEST(ShapeSliders, TheSliderIsADialFromMinusOneToOneThatComesBackToItsStart)
{
  Body b;
  const std::vector<float> rest = b.geo->position;
  b.addSliders();
  const fv::morph::Channel* ch = b.morphs->channel("fv_shape_waist");
  ASSERT_NE(ch, nullptr);
  EXPECT_EQ(ch->label, "Waist");
  EXPECT_EQ(ch->kind, fv::morph::kShapeKind);
  EXPECT_EQ(ch->region, "Waist");
  EXPECT_DOUBLE_EQ(ch->min, -1);
  EXPECT_DOUBLE_EQ(ch->max, 1);
  EXPECT_FALSE(ch->hidden);
  EXPECT_FALSE(ch->helper);
  EXPECT_DOUBLE_EQ(ch->loaded, 0);

  // a bare-bones figure: the other girth has no bones here, and Height no dial to drive
  EXPECT_FALSE(b.morphs->has("fv_shape_thighs"));
  EXPECT_FALSE(b.morphs->has("fv_shape_height"));

  b.morphs->set("fv_shape_waist", 1);
  b.morphs->flush();
  EXPECT_NEAR(b.geo->position[0], rest[0] * 1.22f, 1e-4);

  b.morphs->set("fv_shape_waist", -1);
  b.morphs->flush();
  EXPECT_NEAR(b.geo->position[0], rest[0] * 0.78f, 1e-4);

  b.morphs->resetAll();
  b.morphs->flush();
  EXPECT_DOUBLE_EQ(b.morphs->get("fv_shape_waist"), 0);
  for(size_t i = 0; i < rest.size(); i++)
    ASSERT_NEAR(b.geo->position[i], rest[i], 1e-4);
}

TEST(ShapeSliders, ACharactersDialsAndMovedCentresLeaveTheTargetAsItIs)
{
  Body plain;
  plain.addSliders();
  const auto expected = plain.morphs->cpuTarget("fv_shape_waist", 0);
  ASSERT_NE(expected, nullptr);

  Body shaped;
  fv::mesh::CageDeltas lift;
  for(int v = 0; v < kAround; v++)
  {
    lift.verts.push_back(v);
    lift.deltas.insert(lift.deltas.end(), {5.f, 3.f, 0.f});
  }
  shaped.morphs->addTarget("Character", 0, std::make_shared<fv::mesh::RenderTarget>(fv::mesh::target(*shaped.geo, lift)));
  shaped.morphs->set("Character", 1);
  shaped.morphs->flush();
  shaped.scene.nodes[size_t(shaped.scene.meshes[0].bones[0])].joint->center = {7, 7, 7};
  shaped.addSliders();
  const auto target = shaped.morphs->cpuTarget("fv_shape_waist", 0);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->cageIndices, expected->cageIndices);
  EXPECT_EQ(target->cageDeltas, expected->cageDeltas);
}

TEST(ShapeSliders, ACageOnlyMeshTakesTheSameMoveOnItsCage)
{
  Body gpu, cpu;
  gpu.morphs->cageOnly = [](int) { return true; };
  gpu.addSliders();
  cpu.addSliders();
  const std::vector<float> drawn = gpu.geo->position;
  gpu.morphs->set("fv_shape_waist", 0.5);
  cpu.morphs->set("fv_shape_waist", 0.5);
  gpu.morphs->flush();
  cpu.morphs->flush();
  EXPECT_EQ(gpu.geo->position, drawn);
  const std::vector<float> cage = gpu.morphs->morphedCage(0), expected = cpu.morphs->morphedCage(0);
  ASSERT_EQ(cage.size(), expected.size());
  for(size_t i = 0; i < cage.size(); i++)
    ASSERT_NEAR(cage[i], expected[i], 1e-5);
}

TEST(ShapeSliders, AShellMovesWithItsBody)
{
  Body b;
  auto shell = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(tube()));
  b.ref.shells.push_back({1, shell});
  b.morphs = std::make_unique<fv::morph::MorphSet>([&](int i) { return i == 0 ? b.geo.get() : shell.get(); });
  b.driver = std::make_unique<fv::pose::FigureDriver>(*b.morphs, std::make_shared<NoFiles>());
  b.morphs->driver = b.driver.get();
  b.addSliders();
  EXPECT_EQ(b.morphs->meshesOf("fv_shape_waist"), (std::set<int>{0, 1}));
}

TEST(ShapeSliders, WhatFollowsTheBodyFollowsTheSlider)
{
  auto         worn = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(tube()));
  Body         b;
  b.morphs = std::make_unique<fv::morph::MorphSet>([&](int i) { return i == 0 ? b.geo.get() : worn.get(); });
  b.driver = std::make_unique<fv::pose::FigureDriver>(*b.morphs, std::make_shared<NoFiles>());
  b.morphs->driver = b.driver.get();
  b.addSliders();

  fv::morph::Follower f;
  f.mesh    = 1;
  f.source  = 0;
  f.project = [worn](const fv::mesh::RenderTarget& t) -> fv::dson::TargetPtr {
    return std::make_shared<fv::mesh::RenderTarget>(fv::mesh::target(*worn, t.cageIndices, t.cageDeltas));
  };
  b.morphs->addFollower(f);

  const float before = worn->position[0];
  b.morphs->set("fv_shape_waist", 1);
  b.morphs->flush();
  EXPECT_NEAR(worn->position[0], before * 1.22f, 1e-4);
}

namespace {

void addDial(fv::morph::MorphSet& morphs, const std::string& name)
{
  morphs.addLazy(name, []() -> fv::morph::Finish {
    return [] {
      fv::morph::LazyResult r;
      r.keep = true;
      return r;
    };
  });
}

}  // namespace

TEST(ShapeSliders, HeightDrivesTheFiguresDialsWhileThoseDialsStayAtZero)
{
  Body b;
  for(const char* dial : {"body_bs_ProportionHeight", "body_bs_ProportionLegsLength", "head_ctrl_ProportionHeadSize_scl"})
    addDial(*b.morphs, dial);
  b.addSliders();
  const fv::morph::Channel* height = b.morphs->channel("fv_shape_height");
  ASSERT_NE(height, nullptr);
  EXPECT_EQ(height->label, "Height");
  EXPECT_EQ(height->kind, fv::morph::kShapeKind);
  EXPECT_DOUBLE_EQ(height->min, -1);
  EXPECT_DOUBLE_EQ(height->max, 1);

  b.morphs->set("fv_shape_height", 0.5);
  b.morphs->settled();
  EXPECT_DOUBLE_EQ(b.morphs->get("fv_shape_height"), 0.5);
  // its range is what the slider said, though its load found no file to say so
  EXPECT_DOUBLE_EQ(b.morphs->channel("fv_shape_height")->min, -1);
  EXPECT_DOUBLE_EQ(b.morphs->get("body_bs_ProportionHeight"), 0);
  EXPECT_DOUBLE_EQ(b.morphs->get("body_bs_ProportionLegsLength"), 0);
  EXPECT_NEAR(b.morphs->weightOf("body_bs_ProportionHeight"), 0.1, 1e-6);
  EXPECT_NEAR(b.morphs->weightOf("body_bs_ProportionLegsLength"), 0.25, 1e-6);
  // a little of the whole figure's scale, which the head doesn't take
  EXPECT_NEAR(b.morphs->weightOf("head_ctrl_ProportionHeadSize_scl"), -0.05, 1e-6);

  b.morphs->set("fv_shape_height", -1);
  b.morphs->settled();
  EXPECT_NEAR(b.morphs->weightOf("body_bs_ProportionLegsLength"), -0.5, 1e-6);

  b.morphs->resetAll();
  b.morphs->settled();
  EXPECT_NEAR(b.morphs->weightOf("body_bs_ProportionHeight"), 0, 1e-6);
  EXPECT_NEAR(b.morphs->weightOf("body_bs_ProportionLegsLength"), 0, 1e-6);
}

TEST(ShapeSliders, ASliderNeedsItsFirstDialAndAddsToWhatADialIsSetTo)
{
  Body b;
  // Height's others, without the dial it is made from; and a slider of its own for the legs
  addDial(*b.morphs, "body_bs_ProportionLegsLength");
  b.addSliders();
  EXPECT_FALSE(b.morphs->has("fv_shape_height"));
  ASSERT_TRUE(b.morphs->has("fv_shape_legs_length"));
  EXPECT_FALSE(b.morphs->has("fv_shape_arms_length"));

  b.morphs->set("body_bs_ProportionLegsLength", 0.25);
  b.morphs->set("fv_shape_legs_length", 1);
  b.morphs->settled();
  EXPECT_NEAR(b.morphs->weightOf("body_bs_ProportionLegsLength"), 0.75, 1e-6);
}
