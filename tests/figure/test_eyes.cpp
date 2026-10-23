// Aiming a figure's eyes at a point: both eyeballs converge, a turned and scaled figure still
// hits it, a formula offset is kept underneath, and the bone's own limits hold. A port of
// test/pose/eyes.test.ts.

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include <figure/dson/rig.hpp>
#include <figure/math/xform.hpp>
#include <figure/pose/pose_rig.hpp>
#include <figure/scene/graph.hpp>

namespace {

std::string channels(double min, double max)
{
  auto axis = [&](const char* id) {
    return std::string(R"({"id":")") + id + R"(","min":)" + std::to_string(min) + R"(,"max":)" + std::to_string(max) + R"(,"clamped":true})";
  };
  return "[" + axis("x") + "," + axis("y") + "," + axis("z") + "]";
}

struct Eyes
{
  fv::SceneGraph                     scene;
  std::unique_ptr<fv::pose::PoseRig> pose;
  fv::dson::Rig                      rig;
};

Eyes figure(double min = -80, double max = 80, double orientY = 0, const char* order = "XYZ")
{
  const std::string rot = channels(min, max);
  const std::string text =
      R"([{"id":"hip","type":"figure","name":"hip","center_point":[{"id":"y","value":0}],"rotation":)" + rot + R"(},)"
      R"({"id":"head","type":"bone","name":"head","parent":"#hip","center_point":[{"id":"y","value":15}],"rotation":)" + rot + R"(},)"
      R"({"id":"lEye","type":"bone","name":"lEye","parent":"#head","center_point":[{"id":"x","value":3},{"id":"y","value":16},{"id":"z","value":2}],)"
      R"("rotation_order":")" + order + R"(","orientation":[{"id":"y","value":)" + std::to_string(orientY) + R"(}],"rotation":)" + rot + R"(},)"
      R"({"id":"rEye","type":"bone","name":"rEye","parent":"#head","center_point":[{"id":"x","value":-3},{"id":"y","value":16},{"id":"z","value":2}],)"
      R"("rotation_order":"YZX","orientation":[{"id":"y","value":)" + std::to_string(-orientY) + R"(}],"rotation":)" + rot + "}]";
  Eyes eyes;
  auto rig = fv::dson::buildRig(eyes.scene, fv::dson::Json::parse(text), nullptr);
  EXPECT_TRUE(rig);
  eyes.rig = *rig;
  // the loader stands a figure up in metres and the user can turn the whole model
  fv::Node wrap;
  wrap.name       = "root";
  wrap.scale      = {0.01, 0.01, 0.01};
  wrap.quaternion = fv::quatFromEuler(0, 0.4, 0, "XYZ");
  const int root  = eyes.scene.add(std::move(wrap));
  eyes.scene.attach(eyes.rig.root, root);
  eyes.pose = std::make_unique<fv::pose::PoseRig>(eyes.scene, eyes.rig, std::vector<const fv::dson::Rig*>{});
  return eyes;
}

double gazeError(const fv::SceneGraph& scene, int bone, const fv::Vec3& target)
{
  const fv::Mat4 w = scene.world(bone);
  const fv::Vec3 at{w.e[12], w.e[13], w.e[14]};
  const fv::Vec3 gaze = fv::rotate({0, 0, 1}, fv::rotationOf(w));
  const fv::Vec3 want = fv::normalize(target - at);
  return std::acos(std::clamp(fv::dot(gaze, want), -1.0, 1.0));
}

}  // namespace

TEST(Eyes, bothEyesTrackAPointWhileTheHeadIsTurned)
{
  Eyes eyes = figure(-80, 80, 18);
  eyes.pose->setMany({{"head", fv::pose::ChannelKind::eRotation, "x", 22}, {"head", fv::pose::ChannelKind::eRotation, "y", -15}});
  const fv::Vec3 target{0.4, 0.2, 1.5};
  EXPECT_EQ(eyes.pose->aimEyes(target), 2);
  EXPECT_NEAR(gazeError(eyes.scene, eyes.rig.byId("lEye"), target), 0, 5e-5);
  EXPECT_NEAR(gazeError(eyes.scene, eyes.rig.byId("rEye"), target), 0, 5e-5);
  EXPECT_EQ(eyes.pose->posed("head", "rotation", "x"), 22);
}

TEST(Eyes, keepsAFormulaOffsetAndStillLandsOnTheAim)
{
  Eyes eyes = figure();
  eyes.pose->setDriven({{"lEye|rotation|y", 15}});
  const int      eye = eyes.rig.byId("lEye");
  const fv::Mat4 w   = eyes.scene.world(eye);
  const fv::Vec3 at{w.e[12], w.e[13], w.e[14]};
  const int      parent = eyes.scene.nodes[eye].parent;
  const fv::Vec3 target = at + fv::rotate({0, 0, 1}, fv::rotationOf(eyes.scene.world(parent)));
  eyes.pose->aimEyes(target);
  EXPECT_NEAR(eyes.pose->posed("lEye", "rotation", "y").value_or(999), -15, 5e-4);
  EXPECT_NEAR(gazeError(eyes.scene, eye, target), 0, 5e-5);
}

TEST(Eyes, staysInsideTheBoneLimitsWhenThePointIsOutOfReach)
{
  Eyes           eyes = figure(-8, 8);
  const int      eye  = eyes.rig.byId("lEye");
  const fv::Mat4 w    = eyes.scene.world(eye);
  const fv::Vec3 ahead{w.e[12], w.e[13], w.e[14]};
  const fv::Vec3 target = ahead + fv::Vec3{1, 0, 0};
  const double   before = gazeError(eyes.scene, eye, target);
  eyes.pose->aimEyes(target);
  const fv::pose::PoseBone* pb = eyes.pose->get("lEye");
  ASSERT_TRUE(pb);
  for(const auto& a : pb->axes)
    if(a.kind == fv::pose::ChannelKind::eRotation)
    {
      EXPECT_GE(a.value, -8);
      EXPECT_LE(a.value, 8);
    }
  EXPECT_LT(gazeError(eyes.scene, eye, target), before);
}

TEST(Eyes, doesNothingToAFigureWithNoEyeBones)
{
  const std::string rot = channels(-80, 80);
  const std::string text =
      R"([{"id":"hip","type":"figure","name":"hip","center_point":[{"id":"y","value":0}],"rotation":)" + rot
      + R"(},{"id":"head","type":"bone","name":"head","parent":"#hip","center_point":[{"id":"y","value":15}],"rotation":)" + rot + "}]";
  fv::SceneGraph scene;
  auto           rig = fv::dson::buildRig(scene, fv::dson::Json::parse(text), nullptr);
  ASSERT_TRUE(rig);
  fv::pose::PoseRig pose(scene, *rig, {});
  EXPECT_FALSE(pose.hasEyes());
  EXPECT_EQ(pose.aimEyes({0, 1, 2}), 0);
  EXPECT_EQ(pose.changed(), 0);
}
