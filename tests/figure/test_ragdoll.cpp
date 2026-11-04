// A figure's ragdoll (figure/physics/rig/figure_physics.hpp) on a small Genesis 9-named skeleton
// built inline: entering Physics moves nothing, held or seated; a pulled hand reaches the pointer and
// the elbow keeps its limits; let go, it stays; a knee never bends past its limit; let fall, it lies
// on the ground and sleeps; leaving keeps the pose in the channels; the physical layer round-trips.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>

#include <figure/dson/rig.hpp>
#include <figure/physics/rig/figure_physics.hpp>
#include <figure/pose/pose_rig.hpp>

namespace {

using fv::Vec3;
using fv::physics::FigurePhysics;
using fv::physics::PhysicsSettings;
using fv::pose::ChannelKind;

std::string channel(const char* id, double min, double max, bool clamped = true)
{
  return std::string(R"({"id":")") + id + R"(","min":)" + std::to_string(min) + R"(,"max":)" + std::to_string(max) + R"(,"clamped":)" + (clamped ? "true" : "false") + "}";
}
std::string rotation(double x0, double x1, double y0, double y1, double z0, double z1)
{
  return "[" + channel("x", x0, x1) + "," + channel("y", y0, y1) + "," + channel("z", z0, z1) + "]";
}
std::string point(const char* key, Vec3 p)
{
  return std::string("\"") + key + R"(":[{"id":"x","value":)" + std::to_string(p.x) + R"(},{"id":"y","value":)" + std::to_string(p.y) + R"(},{"id":"z","value":)" + std::to_string(p.z) + "}]";
}
std::string bone(const std::string& id, const std::string& parent, Vec3 center, Vec3 end, const std::string& rot, const char* order = "XYZ", bool translate = false)
{
  std::string s = R"({"id":")" + id + R"(","type":"bone","name":")" + id + R"(","parent":"#)" + parent + R"(",)" + point("center_point", center) + "," + point("end_point", end)
                  + R"(,"rotation_order":")" + order + R"(","rotation":)" + rot;
  if(translate)
    s += R"(,"translation":[)" + channel("x", -1000, 1000, false) + "," + channel("y", -1000, 1000, false) + "," + channel("z", -1000, 1000, false) + "]";
  return s + "}";
}

// A person 175 cm tall, arms out to the sides, standing on y = 0.
std::string skeleton()
{
  std::vector<std::string> b;
  b.push_back(R"({"id":"Genesis9","type":"figure","name":"Genesis9",)" + point("center_point", {}) + "}");
  b.push_back(bone("hip", "Genesis9", {0, 100, 0}, {0, 105, 0}, rotation(-180, 180, -180, 180, -180, 180), "YZX", true));
  b.push_back(bone("pelvis", "hip", {0, 100, 0}, {0, 92, 0}, rotation(-30, 30, -30, 30, -30, 30), "YZX"));
  b.push_back(bone("spine1", "hip", {0, 105, 0}, {0, 115, 0}, rotation(-30, 30, -20, 20, -20, 20), "YZX"));
  b.push_back(bone("spine2", "spine1", {0, 115, 0}, {0, 125, 0}, rotation(-30, 30, -20, 20, -20, 20), "YZX"));
  b.push_back(bone("spine3", "spine2", {0, 125, 0}, {0, 135, 0}, rotation(-30, 30, -20, 20, -20, 20), "YZX"));
  b.push_back(bone("spine4", "spine3", {0, 135, 0}, {0, 145, 0}, rotation(-30, 30, -20, 20, -20, 20), "YZX"));
  b.push_back(bone("neck1", "spine4", {0, 145, 0}, {0, 150, 0}, rotation(-30, 30, -30, 30, -30, 30), "YZX"));
  b.push_back(bone("neck2", "neck1", {0, 150, 0}, {0, 155, 0}, rotation(-30, 30, -30, 30, -30, 30), "YZX"));
  b.push_back(bone("head", "neck2", {0, 155, 0}, {0, 175, 0}, rotation(-40, 40, -60, 60, -30, 30), "YZX"));
  for(const char* side : {"l", "r"})
  {
    const double s = side[0] == 'l' ? 1 : -1;
    const auto   n = [&](const char* name) { return std::string(side) + "_" + name; };
    b.push_back(bone(n("shoulder"), "spine4", {4 * s, 142, 0}, {16 * s, 142, 0}, rotation(-30, 30, -30, 30, -30, 30)));
    b.push_back(bone(n("upperarm"), n("shoulder"), {16 * s, 142, 0}, {44 * s, 142, 0}, rotation(-90, 90, -100, 100, -90, 90)));
    // the elbow bends forward only: for the left arm along +x, a turn about -y
    b.push_back(bone(n("forearm"), n("upperarm"), {44 * s, 142, 0}, {70 * s, 142, 0}, rotation(-60, 60, s > 0 ? -150 : 0, s > 0 ? 0 : 150, -5, 5)));
    b.push_back(bone(n("hand"), n("forearm"), {70 * s, 142, 0}, {86 * s, 142, 0}, rotation(-70, 70, -30, 30, -70, 70)));
    b.push_back(bone(n("index1"), n("hand"), {86 * s, 142, 2}, {91 * s, 142, 2}, rotation(-10, 90, -20, 20, -20, 20)));
    b.push_back(bone(n("thigh"), "pelvis", {9 * s, 93, 0}, {9 * s, 50, 0}, rotation(-115, 35, -45, 45, -45, 85), "YZX"));
    // the knee bends one way
    b.push_back(bone(n("shin"), n("thigh"), {9 * s, 50, 0}, {9 * s, 8, 0}, rotation(0, 150, -10, 10, -5, 5), "YZX"));
    b.push_back(bone(n("foot"), n("shin"), {9 * s, 8, 0}, {9 * s, 2, 14}, rotation(-40, 60, -15, 15, -20, 20), "YZX"));
  }
  std::string out = "[";
  for(size_t i = 0; i < b.size(); i++)
    out += (i ? "," : "") + b[i];
  return out + "]";
}

struct Person
{
  fv::SceneGraph                     scene;
  fv::dson::Rig                      rig;
  std::unique_ptr<fv::pose::PoseRig> pose;
  std::unique_ptr<FigurePhysics>     physics;
  int                                root = -1;

  Person()
  {
    auto built = fv::dson::buildRig(scene, fv::dson::Json::parse(skeleton()), nullptr);
    EXPECT_TRUE(built);
    rig = *built;
    // the loader's root, in metres
    fv::Node wrap;
    wrap.name  = "root";
    wrap.scale = {0.01, 0.01, 0.01};
    root       = scene.add(std::move(wrap));
    scene.attach(rig.root, root);
    pose    = std::make_unique<fv::pose::PoseRig>(scene, rig, std::vector<const fv::dson::Rig*>{});
    physics = std::make_unique<FigurePhysics>(scene, *pose, nullptr, -1, root);
    physics->setGround(fv::physics::Plane{{0, 1, 0}, 0});
  }
  void set(const std::string& b, const char* axis, double v, ChannelKind kind = ChannelKind::eRotation) { pose->setMany({{b, kind, axis, v}}); }
  double shown(const std::string& b, int axis) const { return pose->get(b)->shown(ChannelKind::eRotation, axis); }
  Vec3 at(const std::string& b) const
  {
    const fv::Mat4 w = fv::inverse(scene.world(root)) * scene.world(pose->get(b)->bone);
    return {w.e[12], w.e[13], w.e[14]};
  }
  // every bone's shown rotation, in order
  std::vector<double> rotations() const
  {
    std::vector<double> out;
    for(const auto& pb : pose->bones())
      for(int a = 0; a < 3; a++)
        out.push_back(pb.shown(ChannelKind::eRotation, a));
    return out;
  }
  void run(double seconds)
  {
    for(int f = 0; f < int(seconds * 60); f++)
      physics->step(1.0 / 60);
  }
  void on(double strength = 0.8, bool letFall = false)
  {
    PhysicsSettings s;
    s.on       = true;
    s.strength = strength;
    s.letFall  = letFall;
    physics->set(s);
  }
};

double maxDiff(const std::vector<double>& a, const std::vector<double>& b)
{
  double d = 0;
  for(size_t i = 0; i < a.size(); i++)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

}  // namespace

TEST(Ragdoll, aGenesisSkeletonGivesItsParts)
{
  Person p;
  ASSERT_TRUE(p.physics->valid());
  // hip with pelvis, two spine parts, the neck, the head, and per side collar, upper arm, forearm,
  // hand, thigh, shin, foot
  EXPECT_EQ(p.physics->parts().size(), 1u + 2 + 1 + 1 + 2 * 7);
  EXPECT_EQ(p.physics->partOf(p.pose->indexOf("pelvis")), p.physics->partOf(p.pose->indexOf("hip")));
  EXPECT_EQ(p.physics->partOf(p.pose->indexOf("spine2")), p.physics->partOf(p.pose->indexOf("spine1")));
  EXPECT_EQ(p.physics->partOf(p.pose->indexOf("l_index1")), -1) << "a finger keeps its channels";
  EXPECT_EQ(p.physics->partOf(p.pose->indexOf("Genesis9")), -1) << "the root is the placement";
}

TEST(Ragdoll, enteringPhysicsMovesNoBone)
{
  for(const bool seated : {false, true})
  {
    Person p;
    // an A-pose, or sitting: thighs forward, shins down
    p.set("l_upperarm", "z", -45);
    p.set("r_upperarm", "z", 45);
    if(seated)
      for(const char* side : {"l_", "r_"})
      {
        p.set(std::string(side) + "thigh", "x", -90);
        p.set(std::string(side) + "shin", "x", 90);
      }
    const auto before = p.rotations();
    const Vec3 hip    = p.at("hip");
    p.on();
    for(int f = 0; f < 180; f++)
    {
      p.physics->step(1.0 / 60);
      ASSERT_LT(maxDiff(p.rotations(), before), 1e-3) << (seated ? "seated" : "standing") << ", frame " << f;
    }
    EXPECT_LT(length(p.at("hip") - hip), 1e-3);
    EXPECT_FALSE(p.physics->awake()) << "still awake after 3 s";
  }
}

// Hold sets how far a push bends it, not whether it stands: barely held, the pose still holds
// against gravity and a pull stays where it was let go; at 0 it is limp, and the arm falls
TEST(Ragdoll, aLowHoldStillHoldsThePoseAndZeroIsLimp)
{
  {
    Person p;
    p.set("l_upperarm", "z", -45);
    const auto before = p.rotations();
    p.on(0.05);
    p.run(3);
    EXPECT_LT(maxDiff(p.rotations(), before), 0.05);
    double     depth = 0;
    // the middle of the hand, down the lowered arm
    const Vec3 along = p.at("l_hand") - p.at("l_forearm");
    const Vec3 hand  = p.at("l_hand") + along * (8 / length(along));
    ASSERT_TRUE(p.physics->grab(hand + Vec3{0, 0, 100}, {0, 0, -1}, depth));
    p.physics->drag(hand + Vec3{0, 20, 10});
    p.run(1.5);
    p.pose->setMany(p.physics->release());
    p.run(0.5);
    const auto reached = p.rotations();
    EXPECT_GT(maxDiff(reached, before), 5) << "the pull didn't move the arm";
    p.run(3);
    EXPECT_LT(maxDiff(p.rotations(), reached), 0.05);
  }
  Person p;
  p.set("l_upperarm", "z", -45);
  const double arm = p.shown("l_upperarm", 2);
  p.on(0);
  p.run(3);
  EXPECT_GT(std::abs(p.shown("l_upperarm", 2) - arm), 10) << "nothing held it, yet it stayed";
  for(const double r : p.rotations())
    ASSERT_TRUE(std::isfinite(r));
}

TEST(Ragdoll, aPulledHandReachesItsTargetAndTheElbowKeepsItsLimits)
{
  Person p;
  p.on();
  const Vec3 hand = p.at("l_hand") + Vec3{8, 0, 0};
  double     depth = 0;
  // from the front, at the hand's middle
  ASSERT_TRUE(p.physics->grab(hand + Vec3{0, 0, 100}, {0, 0, -1}, depth));
  const Vec3 grabbed = hand + Vec3{0, 0, 100 - depth};
  // 20 cm up and 15 cm forward
  const Vec3 target = grabbed + Vec3{0, 20, 15};
  p.physics->drag(target);
  p.run(2);
  const auto& pull = p.physics->world().pins.back();
  const auto& body = p.physics->world().bodies[size_t(pull.body)];
  EXPECT_LT(length(body.toWorld(pull.local) - target), 0.5);
  const double elbow = p.shown("l_forearm", 1);
  EXPECT_GE(elbow, -150.5);
  EXPECT_LE(elbow, 0.5);
}

TEST(Ragdoll, aReleasedPullDoesNotDrift)
{
  Person p;
  p.on();
  double depth = 0;
  const Vec3 hand = p.at("l_hand") + Vec3{8, 0, 0};
  ASSERT_TRUE(p.physics->grab(hand + Vec3{0, 0, 100}, {0, 0, -1}, depth));
  p.physics->drag(hand + Vec3{0, 25, 10});
  p.run(1.5);
  p.pose->setMany(p.physics->release());
  p.run(0.5);
  const auto reached = p.rotations();
  p.run(3);
  EXPECT_LT(maxDiff(p.rotations(), reached), 0.01);
  EXPECT_FALSE(p.physics->awake());
}

TEST(Ragdoll, aKneeNeverBendsPastItsLimit)
{
  Person p;
  p.on(0.3);
  double depth = 0;
  const Vec3 ankle = p.at("l_foot");
  ASSERT_TRUE(p.physics->grab(ankle + Vec3{0, 3, 100}, {0, 0, -1}, depth));
  // a hard pull forward and up: the knee would bend the wrong way
  const Vec3 target = ankle + Vec3{0, 30, 60};
  for(int f = 0; f < 120; f++)
  {
    p.physics->drag(target);
    p.physics->step(1.0 / 60);
    const double knee = p.shown("l_shin", 0);
    ASSERT_GE(knee, -0.5) << "frame " << f;
    ASSERT_LE(knee, 150.5) << "frame " << f;
  }
}

TEST(Ragdoll, letFallItLiesOnTheGroundAndSleeps)
{
  Person p;
  p.on(0, true);
  int f = 0;
  for(; f < 600 && p.physics->awake(); f++)
    p.physics->step(1.0 / 60);
  EXPECT_FALSE(p.physics->awake()) << "still moving after 10 s";
  double low = INFINITY, high = -INFINITY;
  for(const auto& b : p.physics->world().bodies)
    for(const Vec3& e : {b.toWorld(b.shape.a), b.toWorld(b.shape.b)})
    {
      low  = std::min(low, e.y - b.shape.radius);
      high = std::max(high, e.y + b.shape.radius);
    }
  EXPECT_NEAR(low, 0, 1);
  EXPECT_LT(high, 60) << "it didn't fall down";
}

TEST(Ragdoll, itSleepsOnceStillAndThenWritesNothing)
{
  Person p;
  p.on();
  p.run(2);
  ASSERT_FALSE(p.physics->awake());
  p.pose->takeMoved();
  for(int f = 0; f < 30; f++)
    EXPECT_FALSE(p.physics->step(1.0 / 60));
  EXPECT_TRUE(p.pose->takeMoved().empty());
  // a target change wakes it, and it goes there, asleep within the 5 s a still may wait (the test
  // figure takes about 4)
  p.set("l_upperarm", "z", -30);
  p.run(5);
  EXPECT_NEAR(p.shown("l_upperarm", 2), -30, 0.05);
  EXPECT_FALSE(p.physics->awake());
}

TEST(Ragdoll, leavingKeepsThePoseInTheChannels)
{
  Person p;
  p.on(0.5);
  double depth = 0;
  const Vec3 hand = p.at("l_hand") + Vec3{8, 0, 0};
  ASSERT_TRUE(p.physics->grab(hand + Vec3{0, 0, 100}, {0, 0, -1}, depth));
  p.physics->drag(hand + Vec3{0, 30, 0});
  p.run(1);
  const auto shown = p.rotations();
  EXPECT_GT(std::abs(p.shown("l_upperarm", 2)), 5) << "the pull moved the arm";
  PhysicsSettings off = p.physics->settings();
  off.on              = false;
  p.physics->set(off);
  EXPECT_FALSE(p.pose->anyPhysical());
  // the channels clamp to their limits, which the ragdoll keeps to within its solve
  EXPECT_LT(maxDiff(p.rotations(), shown), 1e-3);
  EXPECT_NEAR(*p.pose->posed("l_upperarm", "rotation", "z"), shown[size_t(p.pose->indexOf("l_upperarm") * 3 + 2)], 1e-3);
}

TEST(Ragdoll, thePhysicalLayerRoundTrips)
{
  Person a;
  a.on(0.5);
  double depth = 0;
  const Vec3 hand = a.at("l_hand") + Vec3{8, 0, 0};
  ASSERT_TRUE(a.physics->grab(hand + Vec3{0, 0, 100}, {0, 0, -1}, depth));
  a.physics->drag(hand + Vec3{0, 30, 0});
  a.run(1);
  const auto values = a.physics->physicalValues();
  Person     b;
  b.on(0.5);
  b.physics->restorePhysical(values);
  EXPECT_LT(maxDiff(a.rotations(), b.rotations()), 1e-9);
  EXPECT_FALSE(b.physics->awake());
}


