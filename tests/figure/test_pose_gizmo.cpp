// Posing a bone from its local transform, as the viewport gizmo does: the channels come back from
// the node's rotation through a joint orientation and a formula offset, limits still hold, the hip's
// translation round-trips, and a click on the body finds the nearest bone.

#include <gtest/gtest.h>
#include <cmath>

#include <memory>
#include <string>

#include <figure/dson/rig.hpp>
#include <figure/math/xform.hpp>
#include <figure/pose/pose_rig.hpp>
#include <figure/scene/graph.hpp>

namespace {

using fv::pose::ChannelKind;

struct Rigged
{
  fv::SceneGraph                     scene;
  fv::dson::Rig                      rig;
  std::unique_ptr<fv::pose::PoseRig> pose;
  int                                wrap = -1;
};

std::unique_ptr<Rigged> build(const std::string& text)
{
  auto r   = std::make_unique<Rigged>();
  auto rig = fv::dson::buildRig(r->scene, fv::dson::Json::parse(text), nullptr);
  EXPECT_TRUE(rig);
  r->rig = *rig;
  // the loader stands a figure up in metres, and the user can turn the whole model
  fv::Node wrap;
  wrap.name       = "root";
  wrap.scale      = {0.01, 0.01, 0.01};
  wrap.quaternion = fv::quatFromEuler(0, 0.4, 0, "XYZ");
  r->wrap         = r->scene.add(std::move(wrap));
  r->scene.attach(r->rig.root, r->wrap);
  r->pose = std::make_unique<fv::pose::PoseRig>(r->scene, r->rig, std::vector<const fv::dson::Rig*>{});
  return r;
}

const std::string kRotation =
    R"([{"id":"x","min":-80,"max":80,"clamped":true},{"id":"y","min":-80,"max":80,"clamped":true},{"id":"z","min":-20,"max":20,"clamped":true}])";

std::unique_ptr<Rigged> arm()
{
  return build(R"([{"id":"hip","type":"figure","name":"hip","center_point":[{"id":"y","value":100}],)"
               R"("rotation":)" + kRotation + R"(,"translation":[{"id":"x","min":-500,"max":500},{"id":"y","min":-500,"max":500},{"id":"z","min":-500,"max":500}]},)"
               R"({"id":"lForearm","type":"bone","name":"lForearm","parent":"#hip","center_point":[{"id":"x","value":30},{"id":"y","value":140}],)"
               R"("rotation_order":"YZX","orientation":[{"id":"x","value":12},{"id":"y","value":30},{"id":"z","value":-25}],"rotation":)" + kRotation + "}]");
}

void zero(fv::pose::PoseRig& pose, const std::string& bone)
{
  pose.setMany({{bone, ChannelKind::eRotation, "x", 0}, {bone, ChannelKind::eRotation, "y", 0}, {bone, ChannelKind::eRotation, "z", 0}});
}

}  // namespace

TEST(PoseGizmo, rotationRoundTripsThroughOrientAndAFormulaOffset)
{
  auto r = arm();
  r->pose->setDriven({{"lForearm|rotation|y", 10}});
  r->pose->setMany({{"lForearm", ChannelKind::eRotation, "x", 25}, {"lForearm", ChannelKind::eRotation, "y", -35},
                    {"lForearm", ChannelKind::eRotation, "z", 15}});
  const int      node = r->rig.byId("lForearm");
  const fv::Quat q    = r->scene.nodes[node].quaternion;
  zero(*r->pose, "lForearm");
  r->pose->setLocalTransform("lForearm", q, std::nullopt);
  EXPECT_NEAR(r->pose->posed("lForearm", "rotation", "x").value_or(999), 25, 1e-6);
  EXPECT_NEAR(r->pose->posed("lForearm", "rotation", "y").value_or(999), -35, 1e-6);
  EXPECT_NEAR(r->pose->posed("lForearm", "rotation", "z").value_or(999), 15, 1e-6);
  const fv::Quat back = r->scene.nodes[node].quaternion;
  EXPECT_NEAR(std::abs(back.x * q.x + back.y * q.y + back.z * q.z + back.w * q.w), 1, 1e-9);
}

TEST(PoseGizmo, theBonesOwnTransformInFloatsPosesNothing)
{
  auto           r = arm();
  const int      node = r->rig.byId("lForearm");
  const fv::Quat q    = r->scene.nodes[node].quaternion;
  // the renderer holds transforms in floats
  const fv::Quat f{float(q.x), float(q.y), float(q.z), float(q.w)};
  r->pose->setLocalTransform("lForearm", f, std::nullopt);
  const fv::Vec3 p = r->scene.nodes[r->rig.root].position;
  r->pose->setLocalTransform("hip", fv::Quat{}, fv::Vec3{float(p.x), float(p.y), float(p.z)});
  EXPECT_EQ(r->pose->changed(), 0);
}

TEST(PoseGizmo, aRotationPastALimitStopsAtIt)
{
  auto                      r  = arm();
  const fv::pose::PoseBone* pb = r->pose->get("lForearm");
  ASSERT_TRUE(pb);
  // z wants 50; its limit is 20
  const fv::Quat e = fv::quatFromEuler(10 * fv::kDeg, -5 * fv::kDeg, 50 * fv::kDeg, pb->order);
  r->pose->setLocalTransform("lForearm", pb->orient * e * pb->orient.inverse(), std::nullopt);
  EXPECT_NEAR(r->pose->posed("lForearm", "rotation", "x").value_or(999), 10, 1e-6);
  EXPECT_NEAR(r->pose->posed("lForearm", "rotation", "y").value_or(999), -5, 1e-6);
  EXPECT_NEAR(r->pose->posed("lForearm", "rotation", "z").value_or(999), 20, 1e-9);
}

TEST(PoseGizmo, aDragPastHalfATurnKeepsGoingTheSameWay)
{
  auto r = build(R"([{"id":"hip","type":"figure","name":"hip","rotation":[{"id":"y","min":-360,"max":360,"clamped":true}]}])");
  r->pose->setMany({{"hip", ChannelKind::eRotation, "y", 170}});
  r->pose->setLocalTransform("hip", fv::quatFromEuler(0, 190 * fv::kDeg, 0, "XYZ"), std::nullopt);
  EXPECT_NEAR(r->pose->posed("hip", "rotation", "y").value_or(999), 190, 1e-6);
}

TEST(PoseGizmo, hipTranslationRoundTrips)
{
  auto r = arm();
  r->pose->setDriven({{"hip|translation|y", 4}});
  r->pose->setMany({{"hip", ChannelKind::eTranslation, "x", 3}, {"hip", ChannelKind::eTranslation, "y", -2},
                    {"hip", ChannelKind::eTranslation, "z", 5}});
  const int      node = r->rig.root;
  const fv::Vec3 p    = r->scene.nodes[node].position;
  r->pose->setMany({{"hip", ChannelKind::eTranslation, "x", 0}, {"hip", ChannelKind::eTranslation, "y", 0},
                    {"hip", ChannelKind::eTranslation, "z", 0}});
  r->pose->setLocalTransform("hip", fv::Quat{}, p);
  EXPECT_NEAR(r->pose->posed("hip", "translation", "x").value_or(999), 3, 1e-9);
  EXPECT_NEAR(r->pose->posed("hip", "translation", "y").value_or(999), -2, 1e-9);
  EXPECT_NEAR(r->pose->posed("hip", "translation", "z").value_or(999), 5, 1e-9);
  EXPECT_NEAR(r->pose->posed("hip", "rotation", "y").value_or(999), 0, 1e-9);
}

TEST(PoseGizmo, aClickPicksTheNearestBoneButNotTheFaceRigOrRoot)
{
  // root -> shoulder -> forearm -> hand (to its end point), a brow bone sitting on the hand, and a
  // head with a face bone named the Genesis 9 way (no Face group by name)
  const std::string rot = R"([{"id":"x","min":-80,"max":80}])";
  auto              r   = build(
      R"([{"id":"hip","type":"figure","name":"hip","rotation":)" + rot + R"(},)"
      R"({"id":"head","type":"bone","name":"head","parent":"#hip","center_point":[{"id":"y","value":160}],)"
      R"("end_point":[{"id":"y","value":180}],"rotation":)" + rot + R"(},)"
      R"({"id":"r_infraorbital","type":"bone","name":"r_infraorbital","parent":"#head","center_point":[{"id":"x","value":-3},{"id":"y","value":170},{"id":"z","value":8}],"rotation":)" + rot + R"(},)"
      R"({"id":"lShldr","type":"bone","name":"lShldr","parent":"#hip","center_point":[{"id":"x","value":10},{"id":"y","value":150}],"rotation":)" + rot + R"(},)"
      R"({"id":"lForearm","type":"bone","name":"lForearm","parent":"#lShldr","center_point":[{"id":"x","value":35},{"id":"y","value":150}],"rotation":)" + rot + R"(},)"
      R"({"id":"lHand","type":"bone","name":"lHand","parent":"#lForearm","center_point":[{"id":"x","value":60},{"id":"y","value":150}],)"
      R"("end_point":[{"id":"x","value":75},{"id":"y","value":150}],"rotation":)" + rot + R"(},)"
      R"({"id":"lBrow","type":"bone","name":"lBrow","parent":"#lHand","center_point":[{"id":"x","value":70},{"id":"y","value":151}],"rotation":)" + rot + "}]");
  // the arm bent at the elbow, so the hand's end point has to follow its pose
  r->pose->setMany({{"lForearm", ChannelKind::eRotation, "x", 40}});
  const fv::Mat4 figure = r->scene.world(r->wrap);
  auto           at     = [&](const char* bone, const fv::Vec3& local) {
    return r->scene.world(r->rig.byId(bone)).transformPoint(local);
  };
  auto nearest = [&](const fv::Vec3& world) {
    const fv::pose::PoseBone* pb = r->pose->nearestBone(world);
    return pb ? pb->id : std::string("none");
  };
  EXPECT_EQ(nearest(figure.transformPoint({20, 153, 1})), "lShldr");
  EXPECT_EQ(nearest(at("lForearm", {12, 2, 1})), "lForearm");
  EXPECT_EQ(nearest(at("lHand", {13, 1, -1})), "lHand");
  EXPECT_EQ(nearest(at("lBrow", {0, 0, 0})), "lHand");
  EXPECT_EQ(nearest(at("r_infraorbital", {0, 0, 1})), "head");
  EXPECT_EQ(nearest(figure.transformPoint({8, 120, 0})), "lShldr");
  EXPECT_EQ(r->pose->boneForNode(r->rig.byId("lHand"))->id, "lHand");
  EXPECT_EQ(r->pose->boneForNode(r->wrap), nullptr);
}

namespace {

double sizeOf(const fv::Mat4& m)
{
  return std::hypot(m.e[0], m.e[1], m.e[2]);
}

// A body, a hand, a finger and its tip along x, each with a general scale, the last two (as most of a
// Genesis figure's bones) with inherits_scale off.
std::unique_ptr<Rigged> hand()
{
  const std::string general = R"("general_scale":{"id":"general_scale","value":1,"min":-10,"max":10})";
  return build(R"([{"id":"body","type":"figure","name":"body",)" + general + R"(,"inherits_scale":true},)"
               R"({"id":"hand","type":"bone","name":"hand","parent":"#body","center_point":[{"id":"x","value":10}],)" + general + R"(,"inherits_scale":true},)"
               R"({"id":"finger","type":"bone","name":"finger","parent":"#hand","center_point":[{"id":"x","value":20}],)" + general + R"(,"inherits_scale":false},)"
               R"({"id":"tip","type":"bone","name":"tip","parent":"#finger","center_point":[{"id":"x","value":30}],)" + general + R"(,"inherits_scale":false}])");
}

}  // namespace

// With inherits_scale off a bone isn't sized by its parent's own scale: a hand and its fingers all set
// to 2 are twice the size, not 2, 4 and 8 times, and each joint moves out by what is above it.
TEST(PoseScale, ABoneThatDoesNotInheritIsNotSizedByItsParent)
{
  const auto r = hand();
  r->pose->setMany({{"hand", ChannelKind::eScale, "general", 2}, {"finger", ChannelKind::eScale, "general", 2}, {"tip", ChannelKind::eScale, "general", 2}});
  const fv::Mat4 wrap = r->scene.world(r->wrap);
  const double   unit = sizeOf(wrap);
  const auto     size = [&](const char* b) { return sizeOf(r->scene.world(r->rig.byId(b))) / unit; };
  EXPECT_NEAR(size("hand"), 2, 1e-6);
  EXPECT_NEAR(size("finger"), 2, 1e-6);
  EXPECT_NEAR(size("tip"), 2, 1e-6);

  // the finger alone: its tip stays its size, and moves with the finger
  r->pose->setMany({{"hand", ChannelKind::eScale, "general", 1}, {"tip", ChannelKind::eScale, "general", 1}});
  EXPECT_NEAR(size("hand"), 1, 1e-6);
  EXPECT_NEAR(size("finger"), 2, 1e-6);
  EXPECT_NEAR(size("tip"), 1, 1e-6);
  const auto at = [&](const char* b) {
    const fv::Mat4 m = r->scene.world(r->rig.byId(b));
    return fv::inverse(wrap).transformPoint({m.e[12], m.e[13], m.e[14]}).x;
  };
  EXPECT_NEAR(at("finger"), 20, 1e-3);
  EXPECT_NEAR(at("tip"), 40, 1e-3);
}

// What a parent inherited still reaches every bone: scaling the figure's root scales the whole figure,
// joints and all, so that a figure's height dial makes it taller. Driven by a formula as by a channel.
TEST(PoseScale, TheFiguresRootScalesEveryBone)
{
  const auto r = hand();
  r->pose->setDriven({{"body|scale|general", 0.5}});
  const fv::Mat4 wrap = r->scene.world(r->wrap);
  const double   unit = sizeOf(wrap);
  for(const char* b : {"body", "hand", "finger", "tip"})
    EXPECT_NEAR(sizeOf(r->scene.world(r->rig.byId(b))) / unit, 1.5, 1e-6) << b;
  const auto at = [&](const char* b) {
    const fv::Mat4 m = r->scene.world(r->rig.byId(b));
    return fv::inverse(wrap).transformPoint({m.e[12], m.e[13], m.e[14]}).x;
  };
  EXPECT_NEAR(at("hand"), 15, 1e-3);
  EXPECT_NEAR(at("finger"), 30, 1e-3);
  EXPECT_NEAR(at("tip"), 45, 1e-3);

  // and back
  r->pose->setDriven({});
  EXPECT_NEAR(sizeOf(r->scene.world(r->rig.byId("tip"))) / unit, 1, 1e-6);
}
