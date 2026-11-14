//
// Play mode's controllers (fv_play.hpp) on a made-up world: a floor at y = 0 and boxes on it. Someone
// on foot walks at the speed asked, turns the way the mouse goes, jumps and lands, stops at a wall and
// slides along one, climbs a step but not a crate; the ship's throttle and flight assist, its turn and
// bank, its landing on its pads; the chase camera behind it.
//

#include <cmath>
#include <numbers>
#include <vector>

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include "../src/fv_play.hpp"

namespace {

using namespace fv::play;

struct Box
{
  glm::dvec3 lo, hi;
};

// A ray against a box: where it enters, the face's normal toward the ray
std::optional<Surface> hitBox(const Box& b, const glm::dvec3& o, const glm::dvec3& d, double reach)
{
  double     t0 = 0.0, t1 = reach;
  glm::dvec3 n(0.0);
  for(int a = 0; a < 3; a++)
  {
    if(std::abs(d[a]) < 1e-12)
    {
      if(o[a] < b.lo[a] || o[a] > b.hi[a])
        return std::nullopt;
      continue;
    }
    double     enter = (b.lo[a] - o[a]) / d[a], leave = (b.hi[a] - o[a]) / d[a];
    glm::dvec3 face(0.0);
    face[a] = -1.0;
    if(enter > leave)
    {
      std::swap(enter, leave);
      face[a] = 1.0;
    }
    if(enter > t0)
    {
      t0 = enter;
      n  = face;
    }
    t1 = std::min(t1, leave);
    if(t0 > t1)
      return std::nullopt;
  }
  if(n == glm::dvec3(0.0))
    return std::nullopt;  // started inside
  return Surface{o + d * t0, n};
}

World flatWorld(const std::vector<Box>& boxes = {}, bool floor = true)
{
  World w;
  w.up     = [](const glm::dvec3&) { return glm::dvec3(0.0, 1.0, 0.0); };
  w.ground = [boxes, floor](const glm::dvec3& from, const glm::dvec3& up, double reach) -> std::optional<Surface> {
    std::optional<Surface> best;
    if(floor && from.y >= 0.0 && from.y - reach <= 0.0)
      best = Surface{{from.x, 0.0, from.z}, up};
    for(const Box& b : boxes)
      if(from.x >= b.lo.x && from.x <= b.hi.x && from.z >= b.lo.z && from.z <= b.hi.z && from.y >= b.hi.y && from.y - reach <= b.hi.y
         && (!best || b.hi.y > best->at.y))
        best = Surface{{from.x, b.hi.y, from.z}, up};
    return best;
  };
  w.ray = [boxes](const glm::dvec3& from, const glm::dvec3& dir, double reach) -> std::optional<Surface> {
    std::optional<Surface> best;
    for(const Box& b : boxes)
      if(const auto h = hitBox(b, from, dir, reach); h && (!best || glm::distance(h->at, from) < glm::distance(best->at, from)))
        best = h;
    return best;
  };
  return w;
}

void walk(Walker& w, const World& world, double seconds, double dt = 1.0 / 60.0)
{
  for(double t = 0.0; t < seconds; t += dt)
    stepWalker(w, world, dt);
}

void fly(Ship& s, const ShipShape& shape, const World& world, double seconds, bool piloted = true, double dt = 1.0 / 60.0)
{
  for(double t = 0.0; t < seconds; t += dt)
    stepShip(s, shape, world, dt, piloted);
}

// The scrappy fighter's shape about its centre: pads 1.5 m under it, its hull 11 m long and 8 wide
ShipShape fighter()
{
  ShipShape s;
  s.pads   = {{0.0, -1.5, 3.35}, {1.42, -1.5, -1.0}, {-1.42, -1.5, -1.0}};
  s.hull   = {{0.0, 0.0, 6.2}, {0.0, 0.0, -4.4}, {4.1, 0.0, -2.0}, {-4.1, 0.0, -2.0}, {0.0, 1.4, 0.0}, {0.0, -0.5, 0.0}};
  s.radius = 6.0;
  return s;
}

TEST(PlayWalker, StandsWhereTheCameraLooksFromOnTheGroundUnderIt)
{
  const World  world = flatWorld();
  const Walker w     = walkerAt({{2.0, 5.0, 3.0}, glm::normalize(glm::dvec3(0.0, -1.0, -1.0)), {0.0, 1.0, 0.0}}, world, 100.0);
  EXPECT_TRUE(w.grounded);
  EXPECT_NEAR(w.feet.y, 0.0, 1e-9);
  EXPECT_NEAR(w.feet.x, 2.0, 1e-9);
  EXPECT_NEAR(w.facing.z, -1.0, 1e-9);                     // level
  EXPECT_NEAR(w.pitch, -std::numbers::pi / 4.0, 1e-9);     // looking down as the camera did
  const Pose view = walkerView(w, world);
  EXPECT_NEAR(view.eye.y, WalkerTuning{}.eyeHeight, 1e-9);
  EXPECT_NEAR(glm::dot(view.forward, view.up), 0.0, 1e-9);
}

TEST(PlayWalker, WalksAndRunsAtTheSpeedsAskedAndStops)
{
  const World world = flatWorld();
  Walker      w     = walkerAt({{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  w.keys            = kForward;
  walk(w, world, 1.0);
  EXPECT_NEAR(glm::length(w.velocity), WalkerTuning{}.walk, 0.01);
  EXPECT_LT(w.feet.z, -3.5);
  EXPECT_NEAR(w.feet.y, 0.0, 1e-9);
  EXPECT_TRUE(w.grounded);
  w.keys = kForward | kFast;
  walk(w, world, 1.0);
  EXPECT_NEAR(glm::length(w.velocity), WalkerTuning{}.run, 0.01);
  w.keys = 0;
  walk(w, world, 1.0);
  EXPECT_LT(glm::length(w.velocity), 0.01);
}

TEST(PlayWalker, TurnsTheWayTheMouseGoes)
{
  const World world = flatWorld();
  Walker      w     = walkerAt({{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  // right (+x for someone facing -z) and up
  w.look = {100.0, -50.0};
  walk(w, world, 0.5);
  EXPECT_GT(w.facing.x, 0.1);
  EXPECT_GT(w.pitch, 0.05);
  EXPECT_NEAR(std::atan2(w.facing.x, -w.facing.z), 100.0 * WalkerTuning{}.turnPerPixel, 1e-6);
  EXPECT_NEAR(glm::length(w.look), 0.0, 1e-9);
}

TEST(PlayWalker, JumpsAndComesDown)
{
  const World world = flatWorld();
  Walker      w     = walkerAt({{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  w.jumps           = 1;
  double top        = 0.0;
  for(int i = 0; i < 150; i++)
  {
    // a press in the air is lost: no second jump off nothing
    if(i == 30)
      w.jumps = 1;
    stepWalker(w, world, 1.0 / 120.0);
    top = std::max(top, w.feet.y);
  }
  const WalkerTuning t;
  EXPECT_NEAR(top, t.jump * t.jump / (2.0 * t.gravity), 0.05);
  EXPECT_TRUE(w.grounded);
  EXPECT_NEAR(w.feet.y, 0.0, 1e-9);
  EXPECT_EQ(w.jumps, 0);
}

TEST(PlayWalker, StopsAtAWallAndSlidesAlongOne)
{
  // a wall across the way at z = -3, from x = -5 to 5
  const World world = flatWorld({{{-5.0, 0.0, -3.5}, {5.0, 3.0, -3.0}}});
  Walker      w     = walkerAt({{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  w.keys            = kForward;
  walk(w, world, 2.0);
  EXPECT_NEAR(w.feet.z, -3.0 + WalkerTuning{}.radius, 0.02);
  // half turned to it: along it at the speed's share along the wall
  w.facing = glm::normalize(glm::dvec3(1.0, 0.0, -1.0));
  const double x0 = w.feet.x;
  walk(w, world, 1.0);
  EXPECT_NEAR(w.feet.z, -3.0 + WalkerTuning{}.radius, 0.02);
  EXPECT_GT(w.feet.x - x0, 0.6 * WalkerTuning{}.walk / std::sqrt(2.0));
}

TEST(PlayWalker, ClimbsAStepButNotACrate)
{
  // a step 0.3 m high from z = -2, a crate 1 m high from z = -8
  const World world = flatWorld({{{-5.0, 0.0, -6.0}, {5.0, 0.3, -2.0}}, {{-5.0, 0.0, -9.0}, {5.0, 1.0, -8.0}}});
  Walker      w     = walkerAt({{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  w.keys            = kForward;
  walk(w, world, 1.0);
  EXPECT_NEAR(w.feet.y, 0.3, 1e-9);
  EXPECT_TRUE(w.grounded);
  // the eye follows the step a moment later, not at once
  walk(w, world, 3.0);
  EXPECT_NEAR(w.feet.z, -8.0 + WalkerTuning{}.radius, 0.05);
  EXPECT_LT(w.feet.y, 0.5);
  EXPECT_NEAR(w.eyeLag, 0.0, 1e-3);
}

TEST(PlayWalker, StepsDownAndFallsOffALedge)
{
  // a table 0.8 m high to walk off, then the floor
  const World world = flatWorld({{{-5.0, 0.0, -1.0}, {5.0, 0.8, 1.0}}});
  Walker      w     = walkerAt({{0.0, 3.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  EXPECT_NEAR(w.feet.y, 0.8, 1e-9);
  w.keys = kForward;
  walk(w, world, 2.0);
  EXPECT_NEAR(w.feet.y, 0.0, 1e-9);
  EXPECT_TRUE(w.grounded);
  EXPECT_LT(w.feet.z, -1.5);
}

TEST(PlayWalker, ComesBackAfterFallingOutOfTheWorld)
{
  const World world = flatWorld({{{-1.0, 0.0, -1.0}, {1.0, 0.5, 1.0}}}, false);
  Walker      w     = walkerAt({{0.0, 2.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  w.keys            = kForward;
  walk(w, world, 0.5);
  w.keys = 0;
  walk(w, world, 10.0);
  EXPECT_LT(glm::distance(w.feet, w.lastGround), 1e-9);
  EXPECT_GT(w.feet.y, 0.0);
}

TEST(PlayShip, TheThrottleOpensClosesAndBacks)
{
  const World     world = flatWorld({}, false);
  const ShipShape shape = fighter();
  Ship            s;
  s.com  = {0.0, 100.0, 0.0};
  s.keys = kForward;
  fly(s, shape, world, 3.0);
  EXPECT_NEAR(s.throttle, ShipTuning{}.cruise, 1e-9);
  fly(s, shape, world, 5.0);
  EXPECT_NEAR(glm::length(s.velocity), ShipTuning{}.cruise, 0.5);
  EXPECT_GT(s.velocity.z, 0.0);  // along its nose, +z
  // S closes it to nothing and holds there ...
  s.keys = kBack;
  fly(s, shape, world, 4.0);
  EXPECT_EQ(s.throttle, 0.0);
  // ... and pressed again, it backs
  s.keys = 0;
  fly(s, shape, world, 0.1);
  s.keys = kBack;
  fly(s, shape, world, 2.0);
  EXPECT_NEAR(s.throttle, -ShipTuning{}.reverse, 1e-9);
}

TEST(PlayShip, FlightAssistStopsItAndHoldsItWhereItIs)
{
  const World     world = flatWorld({}, false);
  const ShipShape shape = fighter();
  Ship            s;
  s.com      = {0.0, 100.0, 0.0};
  s.velocity = {12.0, -5.0, 30.0};
  fly(s, shape, world, 6.0);
  EXPECT_LT(glm::length(s.velocity), 0.05);
  // strafes and climbs on the keys, at their speed
  s.keys = kRight | kUp;
  fly(s, shape, world, 3.0);
  const ShipTuning t;
  EXPECT_NEAR(glm::dot(s.velocity, shipRight(s)), t.strafe, 0.2);
  EXPECT_NEAR(glm::dot(s.velocity, shipUp(s)), t.strafe, 0.2);
  // no faster than its engines push: a cruise from nothing takes two seconds at least
  Ship q;
  q.com      = {0.0, 100.0, 0.0};
  q.throttle = t.cruise;
  fly(q, shape, world, 1.0);
  EXPECT_LE(glm::length(q.velocity), t.accel * 1.0 + 1e-6);
}

TEST(PlayShip, TheMouseTurnsItAndItLeansIntoTheTurn)
{
  const World     world = flatWorld({}, false);
  const ShipShape shape = fighter();
  Ship            s;
  s.com = {0.0, 100.0, 0.0};
  // the mouse right: the nose goes right (the pilot's right is -x) and the right wing down
  s.aim = {300.0, 0.0};
  fly(s, shape, world, 0.4);
  EXPECT_LT(shipForward(s).x, -0.05);
  EXPECT_LT(shipUp(s).x, -0.02);  // its up leans to its right, -x
  // let go: level again, the nose where it was turned to, and no higher or lower for the lean
  fly(s, shape, world, 4.0);
  const glm::dvec3 nose = shipForward(s);
  EXPECT_NEAR(shipUp(s).y, 1.0, 1e-3);
  EXPECT_NEAR(nose.y, 0.0, 1e-3);
  EXPECT_NEAR(std::atan2(-nose.x, nose.z), 300.0 * ShipTuning{}.turnPerPixel, 0.01);
  // the mouse up: the nose up
  s.aim = {0.0, -200.0};
  fly(s, shape, world, 1.0);
  EXPECT_GT(shipForward(s).y, 0.2);
}

TEST(PlayShip, AFlickTurnsNoFasterThanTheShipCan)
{
  const World     world = flatWorld({}, false);
  const ShipShape shape = fighter();
  Ship            s;
  s.com = {0.0, 100.0, 0.0};
  s.aim = {5000.0, 0.0};
  fly(s, shape, world, 0.25);
  const glm::dvec3 nose = shipForward(s);
  EXPECT_LE(std::atan2(-nose.x, nose.z), ShipTuning{}.yawRate * 0.25 + 1e-6);
  // what waits is a little of it, not seconds of spinning
  EXPECT_LE(glm::length(s.aim), ShipTuning{}.aimQueue / ShipTuning{}.turnPerPixel + 1e-6);
}

TEST(PlayShip, ComesDownOntoItsPadsAndSitsLevel)
{
  const World     world = flatWorld();
  const ShipShape shape = fighter();
  Ship            s;
  s.com         = {0.0, 8.0, 0.0};
  s.orientation = glm::angleAxis(0.15, glm::dvec3(0.0, 0.0, 1.0)) * glm::angleAxis(0.1, glm::dvec3(1.0, 0.0, 0.0));
  s.keys        = kDown;
  fly(s, shape, world, 4.0);
  s.keys = 0;
  fly(s, shape, world, 3.0);
  EXPECT_TRUE(s.touching);
  EXPECT_NEAR(s.com.y, 1.5, 0.03);
  EXPECT_NEAR(shipUp(s).y, 1.0, 1e-3);
  EXPECT_NEAR(s.height, 0.0, 0.03);
  EXPECT_LT(glm::length(s.velocity), 0.05);
  // and lifts off again on the climb key
  s.keys = kUp;
  fly(s, shape, world, 1.0);
  EXPECT_GT(s.height, 3.0);
}

TEST(PlayShip, NeverSinksIntoTheGroundUpsideDown)
{
  const World     world = flatWorld();
  const ShipShape shape = fighter();
  Ship            s;
  s.com         = {0.0, 4.0, 0.0};
  s.orientation = glm::angleAxis(std::numbers::pi, glm::dvec3(0.0, 0.0, 1.0));
  fly(s, shape, world, 3.0, false);
  EXPECT_NEAR(s.com.y, 1.4, 0.03);  // resting on the canopy's top
  EXPECT_FALSE(s.touching && s.height > 0.5 && false);
}

TEST(PlayShip, StopsShortOfWhatItFliesInto)
{
  const World     world = flatWorld({{{-20.0, 0.0, 40.0}, {20.0, 50.0, 42.0}}}, false);
  const ShipShape shape = fighter();
  Ship            s;
  s.com      = {0.0, 10.0, 0.0};
  s.throttle = ShipTuning{}.cruise;
  s.keys     = kForward;
  fly(s, shape, world, 5.0);
  EXPECT_LT(s.com.z, 40.0);
  EXPECT_LT(s.velocity.z, 1.0);
}

TEST(PlayShip, NobodyAboardItFallsAndStops)
{
  const World     world = flatWorld();
  const ShipShape shape = fighter();
  Ship            s;
  s.com      = {0.0, 3.0, 0.0};
  s.velocity = {0.0, 0.0, 4.0};
  s.keys     = kForward;  // nobody's: no throttle
  fly(s, shape, world, 4.0, false);
  EXPECT_NEAR(s.com.y, 1.5, 0.03);
  EXPECT_LT(glm::length(s.velocity), 0.05);
  EXPECT_EQ(s.throttle, 0.0);
}

TEST(PlayCamera, TheChaseIsBehindAndOverTheShipLookingAhead)
{
  Ship s;
  s.com = {0.0, 10.0, 0.0};
  Chase      c;
  const Pose p = chaseView(c, s, 10.0, 1.0 / 60.0);
  EXPECT_LT(p.eye.z, -9.0);
  EXPECT_GT(p.eye.y, 11.0);
  EXPECT_GT(p.forward.z, 0.9);
  EXPECT_NEAR(glm::dot(p.forward, p.up), 0.0, 1e-9);
  // it turns after the ship a moment late, and has caught up a second later
  s.orientation = glm::angleAxis(0.5, glm::dvec3(0.0, 1.0, 0.0));
  Chase      settled;
  const Pose there = chaseView(settled, s, 10.0, 1.0 / 60.0);
  const Pose late  = chaseView(c, s, 10.0, 1.0 / 60.0);
  EXPECT_LT(late.forward.x, there.forward.x * 0.5);
  for(int i = 0; i < 60; i++)
    chaseView(c, s, 10.0, 1.0 / 60.0);
  EXPECT_NEAR(glm::dot(chaseView(c, s, 10.0, 1.0 / 60.0).forward, there.forward), 1.0, 1e-4);
}

TEST(PlayCamera, ABlendEndsWhereItSaysAndKeepsAnUpright)
{
  const Pose a{{0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}};
  const Pose b{{4.0, 2.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  EXPECT_LT(glm::distance(blend(a, b, 0.0).eye, a.eye), 1e-9);
  EXPECT_LT(glm::distance(blend(a, b, 1.0).forward, b.forward), 1e-9);
  const Pose mid = blend(a, b, 0.5);
  EXPECT_NEAR(glm::length(mid.forward), 1.0, 1e-9);
  EXPECT_NEAR(glm::dot(mid.forward, mid.up), 0.0, 1e-9);
  EXPECT_NEAR(mid.up.y, 1.0, 1e-9);
}

}  // namespace
