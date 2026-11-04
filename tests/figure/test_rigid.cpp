// The rigid bodies of the physics core (figure/physics/rigid/world.hpp): whole substeps with the
// rest carried, the same steps for the same inputs, a fall in real time, a joint held by its drive
// with gravity's torque fed forward, a limit that holds, sleep, and rest on the ground and on a prop.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include <figure/physics/rigid/world.hpp>

namespace {

using fv::Quat;
using fv::Vec3;
using namespace fv::physics;

constexpr double kPi = 3.14159265358979323846;

// A capsule along its frame's y, `length` between its ends' centres, at `at`.
RigidBody capsule(const Vec3& at, double length, double radius)
{
  RigidBody b;
  b.x     = at;
  b.shape = {{0, -length / 2, 0}, {0, length / 2, 0}, radius};
  setCapsuleMass(b);
  return b;
}

double lowest(const RigidBody& b)
{
  return std::min(b.toWorld(b.shape.a).y, b.toWorld(b.shape.b).y) - b.shape.radius;
}

// One arm: a 30 cm capsule held out level from a joint at the world's origin, along +x.
RigidWorld levelArm()
{
  RigidWorld w;
  RigidBody  arm = capsule({15, 0, 0}, 24, 3);
  arm.q          = fv::quatFromUnitVectors({0, 1, 0}, {1, 0, 0});
  w.bodies.push_back(arm);
  Joint j;
  j.b       = 0;
  j.anchorA = {0, 0, 0};
  j.anchorB = fv::rotate(Vec3{-15, 0, 0}, arm.q.inverse());
  j.frameB  = arm.q.inverse();
  j.target  = Quat{};
  w.joints.push_back(j);
  return w;
}

// gravity's torque on body 0 about the world's origin
Vec3 gravityTorque(const RigidWorld& w)
{
  const RigidBody& b = w.bodies[0];
  return cross(b.x, w.settings.gravity * b.mass());
}

double angleDeg(const Quat& a, const Quat& b)
{
  const Quat   d = a * b.inverse();
  const double s = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  return 2 * std::atan2(s, std::abs(d.w)) * 180 / kPi;
}

}  // namespace

TEST(Physics, aFrameTakesWholeSubstepsAndCarriesTheRest)
{
  RigidWorld w;
  w.bodies.push_back(capsule({0, 100, 0}, 10, 2));
  w.settings.h = 1.0 / 600;
  EXPECT_EQ(w.step(1.0 / 60), 10);
  EXPECT_NEAR(w.carried(), 0, 1e-9);
  // 1/50 s is 12 substeps exactly; 1/70 s is 8.57: 8 now, and the rest carried
  EXPECT_EQ(w.step(1.0 / 50), 12);
  EXPECT_EQ(w.step(1.0 / 70), 8);
  EXPECT_NEAR(w.carried(), 1.0 / 70 - 8.0 / 600, 1e-9);
  EXPECT_EQ(w.step(1.0 / 70), 9);
  // a hitch is clamped to a twentieth of a second
  EXPECT_EQ(w.step(1.0), 30);
}

TEST(Physics, sameInputsSameSteps)
{
  const auto run = [] {
    RigidWorld w = levelArm();
    w.planes.push_back({{0, 1, 0}, -40});
    w.bodies.push_back(capsule({20, -10, 3}, 10, 3));
    for(int f = 0; f < 120; f++)
      w.step(1.0 / 60 + (f % 3) * 0.001);
    return w;
  };
  const RigidWorld a = run(), b = run();
  for(size_t i = 0; i < a.bodies.size(); i++)
  {
    EXPECT_EQ(std::memcmp(&a.bodies[i].x, &b.bodies[i].x, sizeof(Vec3)), 0);
    EXPECT_EQ(std::memcmp(&a.bodies[i].q, &b.bodies[i].q, sizeof(Quat)), 0);
  }
}

// A drop of 1 m lands in sqrt(2 / 9.8) = 0.452 s.
TEST(Physics, aFallKeepsRealTime)
{
  RigidWorld w;
  w.planes.push_back({{0, 1, 0}, 0});
  RigidBody b = capsule({0, 0, 0}, 20, 4);
  b.q         = fv::quatFromUnitVectors({0, 1, 0}, {1, 0, 0});
  b.x         = {0, 100 + 4, 0};
  w.bodies.push_back(b);
  double t = 0;
  while(lowest(w.bodies[0]) > 1 && t < 2)
  {
    w.step(1.0 / 60);
    t += 1.0 / 60;
  }
  // within 1 cm of the ground: 99 cm fallen
  EXPECT_NEAR(t, std::sqrt(2 * 0.99 / 9.8), 0.45 * 0.02 + 1.0 / 60);
  // and it lies there
  for(int f = 0; f < 120; f++)
    w.step(1.0 / 60);
  EXPECT_NEAR(lowest(w.bodies[0]), 0, 0.05);
}

// Held out level by a drive with gravity's torque fed forward, the arm stays where it is, to within
// 0.01 degrees, and sleeps.
TEST(Physics, aDriveWithGravityFedForwardHoldsAnArmLevel)
{
  RigidWorld w = levelArm();
  const Quat start = w.bodies[0].q;
  w.joints[0].compliance = 1e-9;
  w.joints[0].damping    = 20;
  for(int f = 0; f < 300 && !w.asleep(); f++)
  {
    w.joints[0].feedForward = gravityTorque(w) * -1.0;
    w.step(1.0 / 60);
    ASSERT_LT(angleDeg(w.bodies[0].q, start), 0.01) << "frame " << f;
  }
  EXPECT_TRUE(w.asleep());
  // without the feed forward it sags by about the compliance times the torque
  RigidWorld loose = levelArm();
  loose.joints[0].compliance = 1e-9;
  loose.joints[0].damping    = 20;
  for(int f = 0; f < 300; f++)
    loose.step(1.0 / 60);
  const double sag = angleDeg(loose.bodies[0].q, start) * kPi / 180;
  EXPECT_NEAR(sag, 1e-9 * length(gravityTorque(loose)), 0.2 * 1e-9 * length(gravityTorque(loose)));
}

// A limp arm falls until its joint's box stops it, and stays within half a degree of the bound.
TEST(Physics, aLimitStopsAFallingArm)
{
  RigidWorld w = levelArm();
  // the arm's frame: its y along the arm; a turn about the world's z (the arm's local z here) swings it down
  w.joints[0].lo = {-0.0, -0.0, -20 * kPi / 180};
  w.joints[0].hi = {0.0, 0.0, 20 * kPi / 180};
  w.joints[0].order = "XYZ";
  w.joints[0].damping = 5;
  for(int f = 0; f < 240; f++)
    w.step(1.0 / 60);
  const Vec3   along = fv::rotate(Vec3{0, 1, 0}, w.bodies[0].q);
  const double down  = std::asin(-along.y) * 180 / kPi;
  EXPECT_NEAR(down, 20, 0.5);
  EXPECT_NEAR(length(w.bodies[0].toWorld(w.joints[0].anchorB)), 0, 0.01);
}

TEST(Physics, aCapsuleRestsOnTheGroundAndSleeps)
{
  RigidWorld w;
  w.planes.push_back({{0, 1, 0}, 0});
  RigidBody b = capsule({0, 10, 0}, 30, 5);
  b.q         = fv::quatFromEuler(0.3, 0.2, 1.2, "XYZ");
  w.bodies.push_back(b);
  int f = 0;
  for(; f < 600 && !w.asleep(); f++)
    w.step(1.0 / 60);
  EXPECT_TRUE(w.asleep()) << "after " << f << " frames, last speed " << w.lastSpeed();
  EXPECT_NEAR(lowest(w.bodies[0]), 0, 0.05);
  // asleep, a frame moves nothing
  const Vec3 x = w.bodies[0].x;
  EXPECT_EQ(w.step(1.0 / 60), 0);
  EXPECT_EQ(w.bodies[0].x.y, x.y);
}

TEST(Physics, aCapsuleRestsOnABoxProp)
{
  // a closed box 40 cm wide whose top is at y = 20
  std::vector<float> p;
  for(int k = 0; k < 2; k++)
    for(int j = 0; j < 2; j++)
      for(int i = 0; i < 2; i++)
        p.insert(p.end(), {i ? 20.f : -20.f, j ? 20.f : 0.f, k ? 20.f : -20.f});
  std::vector<uint32_t> t = {0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5};
  auto set = std::make_shared<const SharedTriangles>(std::make_shared<const TriangleSet>(p, t));
  RigidWorld w;
  w.props.push_back({set, fv::Mat4{}, 1});
  RigidBody b = capsule({0, 40, 0}, 20, 4);
  b.q         = fv::quatFromUnitVectors({0, 1, 0}, {1, 0, 0});
  w.bodies.push_back(b);
  for(int f = 0; f < 600 && !w.asleep(); f++)
    w.step(1.0 / 60);
  EXPECT_TRUE(w.asleep());
  EXPECT_NEAR(lowest(w.bodies[0]), 20, 0.1);
}

// Two capsules pushed into each other part; a pair that came in overlapping keeps that much, no
// more, and the allowance shrinks as they part.
TEST(Physics, capsulesPartAndAnOverlapTheyCameInWithIsKept)
{
  RigidWorld w;
  w.settings.gravity = {};
  w.bodies.push_back(capsule({0, 0, 0}, 20, 4));
  w.bodies.push_back(capsule({7, 0, 0}, 20, 4));
  w.allowOverlaps();
  ASSERT_EQ(w.allowances.size(), 1u);
  EXPECT_NEAR(w.allowances[0].depth, 1, 1e-9);
  for(int f = 0; f < 30; f++)
    w.step(1.0 / 60);
  EXPECT_NEAR(w.overlap(0, 1), 1, 1e-6) << "a pair that came in overlapping was pushed apart";
  // pushed closer, they part again to the allowance
  w.wake();
  w.bodies[1].v = {-30, 0, 0};
  for(int f = 0; f < 60; f++)
    w.step(1.0 / 60);
  EXPECT_LE(w.overlap(0, 1), 1 + 1e-3);
  // drawn apart, the allowance is gone: pushed in again they stop at touching
  w.wake();
  w.bodies[1].x = {9, 0, 0};
  w.bodies[1].v = {};
  w.step(1.0 / 60);
  EXPECT_NEAR(w.allowances[0].depth, 0, 1e-9);
  w.bodies[1].v = {-30, 0, 0};
  for(int f = 0; f < 60; f++)
    w.step(1.0 / 60);
  EXPECT_LT(w.overlap(0, 1), 0.01);
}

TEST(Physics, eulerNearTakesTheBranchNearest)
{
  const std::array<double, 3> e{0.3, 1.4, -0.2};
  const Quat                  q = fv::quatFromEuler(e[0], e[1], e[2], "XYZ");
  const auto                  a = eulerNear(q, "XYZ", e);
  for(int i = 0; i < 3; i++)
    EXPECT_NEAR(a[size_t(i)], e[size_t(i)], 1e-9);
  // past the middle axis's 90 degrees, the other form, and a turn round on the first
  const std::array<double, 3> far{0.3 + kPi + 2 * kPi, kPi - 1.4, -0.2 + kPi};
  const auto                  b = eulerNear(q, "XYZ", far);
  for(int i = 0; i < 3; i++)
    EXPECT_NEAR(b[size_t(i)], far[size_t(i)], 1e-9);
}


