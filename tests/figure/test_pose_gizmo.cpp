// Posing a bone from its local transform, as the viewport gizmo does: the channels come back from
// the node's rotation through a joint orientation and a formula offset, limits still hold, the hip's
// translation round-trips, and a click on the body finds the nearest bone.

#include <gtest/gtest.h>

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
