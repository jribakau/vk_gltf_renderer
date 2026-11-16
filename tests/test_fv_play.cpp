//
// Play mode's controllers (fv_play.hpp) on a made-up world: a floor at y = 0 and boxes on it. Someone
// on foot walks at the speed asked, turns the way the mouse goes, jumps and lands, stops at a wall and
// slides along one, climbs a step but not a crate, keeps clear of a rock beside its way; the ship's
// throttle and flight assist, its turn and bank, its wingtips stopping at what they meet, its landing on
// its pads; the chase camera behind it; the torch in hand; a tree's trunk found among its leaves.
//

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
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

TEST(PlayWalker, KeepsClearOfALowRockBesideItsWay)
{
  // a rock 0.8 m high (over a step, under the chest) beside the way: the body is 0.3 m round its middle
  const Box   rock{{0.2, 0.0, -5.2}, {0.6, 0.8, -4.8}};
  const World world = flatWorld({rock});
  Walker      w     = walkerAt({{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, world, 10.0);
  w.keys            = kForward;
  double nearest    = 1e30;
  for(double t = 0.0; t < 2.5; t += 1.0 / 60.0)
  {
    stepWalker(w, world, 1.0 / 60.0);
    const glm::dvec2 at(w.feet.x, w.feet.z);
    const glm::dvec2 on(std::clamp(at.x, rock.lo.x, rock.hi.x), std::clamp(at.y, rock.lo.z, rock.hi.z));
    nearest = std::min(nearest, glm::distance(at, on));
  }
  EXPECT_GT(nearest, WalkerTuning{}.radius - 0.08);
  EXPECT_LT(w.feet.z, -7.0);  // and past it
}

TEST(PlayShip, TheThrottleOpensClosesAndBacks)
{
  // 100 m over the floor: full throttle is the cruise there
  const World     world = flatWorld();
  const ShipShape shape = fighter();
  Ship            s;
  s.com  = {0.0, 101.5, 0.0};
  s.keys = kForward;
  fly(s, shape, world, 3.0);
  EXPECT_NEAR(s.throttle, 1.0, 1e-9);
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
  // no faster than its engines push: full throttle in open space doesn't reach the cap at once
  Ship q;
  q.com      = {0.0, 100.0, 0.0};
  q.throttle = 1.0;
  fly(q, shape, world, 1.0);
  EXPECT_LT(glm::length(q.velocity), 60.0);
}

TEST(PlayShip, FullThrottleFollowsTheRoomItHasAndStopsAtTheCap)
{
  World           world = flatWorld();
  const ShipTuning t;
  world.speedCap = 2000.0;
  Ship s;
  s.height    = 40.0;
  s.clearance = 40.0;
  EXPECT_EQ(shipSpeedRange(s, world.speedCap), t.cruise);  // near the ground: the cruise
  s.clearance = 1000.0;
  EXPECT_EQ(shipSpeedRange(s, world.speedCap), 500.0);  // higher: half the room a second
  s.clearance = 1.0e6;
  EXPECT_EQ(shipSpeedRange(s, world.speedCap), 2000.0);  // and no more than the world's cap ...
  s.keys = kFast;
  EXPECT_EQ(shipSpeedRange(s, world.speedCap), 6000.0);  // ... but Shift's three times that

  // a climb at full throttle with Shift held: as fast as the cap lets it, never a runaway
  const ShipShape shape = fighter();
  Ship            up;
  up.com         = {0.0, 51.5, 0.0};
  up.orientation = glm::angleAxis(-std::numbers::pi / 2.0, glm::dvec3(1.0, 0.0, 0.0));  // the nose up
  up.throttle    = 1.0;
  up.keys        = kFast;
  fly(up, shape, world, 60.0);
  EXPECT_LE(glm::length(up.velocity), world.speedCap * t.boost * 1.001);
  EXPECT_LT(up.height, world.speedCap * t.boost * 60.0);
}

TEST(PlayShip, ADiveAtFullThrottleSlowsAsTheGroundComesUp)
{
  World world    = flatWorld();
  world.speedCap = 2000.0;
  const ShipShape shape = fighter();
  Ship            s;
  s.com         = {0.0, 3000.0, 0.0};
  s.orientation = glm::angleAxis(0.6, glm::dvec3(1.0, 0.0, 0.0));  // the nose 34 degrees down
  s.throttle    = 1.0;
  double fastest = 0.0, atTouch = -1.0;
  for(int i = 0; i < 60 * 120 && atTouch < 0.0; i++)
  {
    stepShip(s, shape, world, 1.0 / 60.0, true);
    fastest = std::max(fastest, glm::length(s.velocity));
    if(s.touching)
      atTouch = glm::length(s.velocity);
  }
  EXPECT_GT(fastest, 500.0);  // it went fast with the room it had
  ASSERT_GE(atTouch, 0.0);    // and reached the ground
  EXPECT_LT(atTouch, 1.5 * ShipTuning{}.cruise);
}

TEST(PlayShip, SlowsForAWallAhead)
{
  // open space under it, a wall 300 m ahead: full throttle near the wall is the cruise
  World world    = flatWorld({{{-200.0, 0.0, 300.0}, {200.0, 400.0, 302.0}}}, false);
  world.speedCap = 2000.0;
  Ship s;
  s.com      = {0.0, 200.0, 0.0};
  s.throttle = 1.0;
  stepShip(s, fighter(), world, 1.0 / 60.0, true);
  EXPECT_NEAR(s.clearance, 300.0, 1e-6);
  EXPECT_EQ(shipSpeedRange(s, world.speedCap), 150.0);
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
  s.throttle = 1.0;
  s.keys     = kForward;
  fly(s, shape, world, 5.0);
  EXPECT_LT(s.com.z, 40.0);
  EXPECT_LT(s.velocity.z, 1.0);
}

TEST(PlayShip, AWingtipStopsAtWhatItMeets)
{
  // a pillar in the right wingtip's way (x 4.1), clear of the centre's
  const World     world = flatWorld({{{3.5, 0.0, 20.0}, {5.0, 30.0, 21.0}}}, false);
  const ShipShape shape = fighter();
  Ship            s;
  s.com      = {0.0, 10.0, 0.0};
  s.throttle = 0.3;
  s.keys     = kForward;
  fly(s, shape, world, 5.0);
  EXPECT_LT(s.com.z - 2.0, 20.0);  // the wingtip, 2 m behind the centre
  EXPECT_GT(s.com.z, 15.0);
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

// Where a beam from `p` passes nearest `point`: how far off its line the point is
double offBeam(const Pose& p, const glm::dvec3& point)
{
  const glm::dvec3 off = point - p.eye;
  return glm::length(off - p.forward * glm::dot(off, p.forward));
}

TEST(PlayTorch, InTheRightHandShiningWhereTheViewLooks)
{
  Torch      t;
  const Pose view{{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}};
  const Pose p = torchView(t, view, 8.0, 1.0 / 60.0);
  EXPECT_GT(p.eye.x, 0.1);   // right of the eye
  EXPECT_LT(p.eye.y, 1.45);  // and under it
  EXPECT_NEAR(glm::length(p.forward), 1.0, 1e-9);
  EXPECT_NEAR(glm::dot(p.forward, p.up), 0.0, 1e-9);
  // its beam crosses the view's line where the view meets something, 8 m ahead
  EXPECT_LT(offBeam(p, view.eye + view.forward * 8.0), 0.01);
  // looking at nothing, it crosses it a long way ahead, nearly along it
  Torch      far;
  const Pose q = torchView(far, view, std::nullopt, 1.0 / 60.0);
  EXPECT_GT(glm::dot(q.forward, view.forward), 0.999);
  // nothing nearer than a metre: the wrist turns no further in
  Torch      near;
  const Pose r = torchView(near, view, 0.1, 1.0 / 60.0);
  EXPECT_LT(offBeam(r, view.eye + view.forward * 1.0), 0.01);
}

// What the view looks at, down its line: the floor it looks down at, a box in its way, nothing up at the sky
TEST(PlayTorch, TheViewsLineMeetsTheGroundOrWhatStandsThere)
{
  const World      world = flatWorld({{{-1.0, 0.0, -6.0}, {1.0, 3.0, -5.0}}});
  const glm::dvec3 eye{0.0, 1.65, 0.0};
  // 30 degrees down: the floor 3.3 m off
  const glm::dvec3 down = glm::normalize(glm::dvec3(0.0, -std::sin(std::numbers::pi / 6.0), -std::cos(std::numbers::pi / 6.0)));
  const auto       floor = sightDistance(world, {eye, down, {0.0, 1.0, 0.0}}, 40.0);
  ASSERT_TRUE(floor.has_value());
  EXPECT_NEAR(*floor, 1.65 / std::sin(std::numbers::pi / 6.0), 0.02);
  // level, at the box 5 m ahead
  const auto box = sightDistance(world, {eye, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}}, 40.0);
  ASSERT_TRUE(box.has_value());
  EXPECT_NEAR(*box, 5.0, 1e-6);
  // level the other way, and up: nothing within reach
  EXPECT_FALSE(sightDistance(world, {eye, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}, 40.0).has_value());
  EXPECT_FALSE(sightDistance(world, {eye, glm::normalize(glm::dvec3(0.0, 1.0, 1.0)), glm::normalize(glm::dvec3(0.0, 1.0, -1.0))}, 40.0).has_value());
  // a floor further off than the reach is nothing either: 2 degrees down meets it 47 m off
  const glm::dvec3 shallow = glm::normalize(glm::dvec3(0.0, -std::sin(0.035), std::cos(0.035)));
  EXPECT_FALSE(sightDistance(world, {eye, shallow, {0.0, 1.0, 0.0}}, 40.0).has_value());
  // looking down at the floor, the torch's pool is where the view's middle is
  Torch      t;
  const Pose view{eye, down, glm::normalize(glm::cross(glm::cross(down, glm::dvec3(0.0, 1.0, 0.0)), down))};
  const Pose p = torchView(t, view, floor, 1.0 / 60.0);
  EXPECT_LT(offBeam(p, eye + down * *floor), 0.01);
}

TEST(PlayTorch, ItsAimFollowsATurnAMomentLate)
{
  Torch      t;
  const Pose before{{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}};
  const Pose after{{0.0, 1.65, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  torchView(t, before, 8.0, 1.0 / 60.0);
  const Pose first = torchView(t, after, 8.0, 1.0 / 60.0);
  EXPECT_LT(glm::dot(first.forward, after.forward), 0.7);  // still turning
  for(int i = 0; i < 60; i++)
    torchView(t, after, 8.0, 1.0 / 60.0);
  EXPECT_GT(glm::dot(torchView(t, after, 8.0, 1.0 / 60.0).forward, after.forward), 0.99);
  // and its reach a moment late too: from a wall 2 m off to a hill 30 m off, eased, the same pace either way
  Torch      r;
  const Pose view{{0.0, 1.65, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}};
  torchView(r, view, 2.0, 1.0 / 60.0);
  torchView(r, view, 30.0, 1.0 / 60.0);
  EXPECT_GT(r.reach, 2.0);
  EXPECT_LT(r.reach, 30.0);
  const double out = std::log(r.reach / 2.0);
  Torch        b;
  torchView(b, view, 30.0, 1.0 / 60.0);
  torchView(b, view, 2.0, 1.0 / 60.0);
  EXPECT_NEAR(std::log(30.0 / b.reach), out, 1e-9);
  for(int i = 0; i < 60; i++)
    torchView(r, view, 30.0, 1.0 / 60.0);
  EXPECT_NEAR(r.reach, 30.0, 0.01);
}

// A tree: a trunk of `radius` round `centre` up to 3 m (a plain column of ten sides, or rings of
// vertices every 10 cm as a scanned one has), a crown of leaves over it and a few twigs low down beside it
struct TreeMesh
{
  std::vector<float>    xyz;
  std::vector<uint32_t> tris;
};
TreeMesh treeMesh(glm::dvec2 centre, double radius, bool dense)
{
  TreeMesh   m;
  const auto put = [&](const glm::dvec3& p) {
    m.xyz.insert(m.xyz.end(), {float(p.x), float(p.y), float(p.z)});
    return uint32_t(m.xyz.size() / 3 - 1);
  };
  if(!dense)
  {
    auto [xyz, tris] = columnMesh({centre, radius, 0.0, 3.0}, glm::dmat4(1.0), 10);
    m.xyz            = std::move(xyz);
    m.tris           = std::move(tris);
  }
  else
  {
    for(int row = 0; row <= 30; row++)
      for(int k = 0; k < 24; k++)
      {
        const double a = 2.0 * std::numbers::pi * k / 24.0;
        put({centre.x + radius * std::cos(a), 0.1 * row, centre.y + radius * std::sin(a)});
      }
    for(uint32_t row = 0; row < 30; row++)
      for(uint32_t k = 0; k < 24; k++)
      {
        const uint32_t a = row * 24 + k, b = row * 24 + (k + 1) % 24, c = a + 24, d = b + 24;
        m.tris.insert(m.tris.end(), {a, b, c, c, b, d});
      }
  }
  uint32_t   seed = 7;
  const auto next = [&] {
    seed = seed * 1664525u + 1013904223u;
    return double(seed >> 8) / double(1u << 24);
  };
  const auto triangle = [&](const glm::dvec3& p, const glm::dvec3& u, const glm::dvec3& v) {
    const uint32_t a = put(p), b = put(p + u), c = put(p + v);
    m.tris.insert(m.tris.end(), {a, b, c});
  };
  for(int i = 0; i < 2000; i++)  // leaves
    triangle({centre.x + 4.0 * (next() - 0.5), 3.0 + 3.0 * next(), centre.y + 4.0 * (next() - 0.5)}, {0.1, 0.05, 0.0}, {0.0, 0.1, 0.08});
  for(int i = 0; i < 30; i++)  // twigs low down, out to a metre past the bark, half a metre tall
  {
    const double a = 2.0 * std::numbers::pi * next(), d = radius + 0.1 + 0.8 * next();
    triangle({centre.x + d * std::cos(a), 0.8 + 0.3 * next(), centre.y + d * std::sin(a)}, {0.02, 0.5, 0.0}, {0.3, 0.25, 0.02});
  }
  return m;
}

TEST(PlayTrunk, FoundAmongItsLeavesAndTwigs)
{
  for(const bool dense : {false, true})
  {
    const TreeMesh m = treeMesh({0.3, -0.1}, 0.25, dense);
    const auto     c = trunkOf(m.xyz, m.tris, glm::dmat4(1.0));
    ASSERT_TRUE(c) << (dense ? "dense" : "plain");
    EXPECT_NEAR(c->centre.x, 0.3, 0.03) << (dense ? "dense" : "plain");
    EXPECT_NEAR(c->centre.y, -0.1, 0.03) << (dense ? "dense" : "plain");
    EXPECT_NEAR(c->radius, 0.25, 0.03) << (dense ? "dense" : "plain");
    EXPECT_LT(c->bottom, 0.0);
    EXPECT_GT(c->top, 2.0);
  }
}

TEST(PlayTrunk, OfAModelLyingInItsFileAsAnotherProgramMadeIt)
{
  // the same tree with +z up in its file, stood up by its node's turn (z up to y up)
  TreeMesh m = treeMesh({0.0, 0.0}, 0.2, false);
  for(size_t i = 0; i < m.xyz.size(); i += 3)
  {
    const float y = m.xyz[i + 1], z = m.xyz[i + 2];
    m.xyz[i + 1]  = -z;
    m.xyz[i + 2]  = y;
  }
  const glm::dmat4 upright = glm::rotate(glm::dmat4(1.0), -std::numbers::pi / 2.0, glm::dvec3(1.0, 0.0, 0.0));
  const auto       c       = trunkOf(m.xyz, m.tris, upright);
  ASSERT_TRUE(c);
  EXPECT_NEAR(c->radius, 0.2, 0.03);
  EXPECT_NEAR(glm::length(c->centre), 0.0, 0.03);
}

TEST(PlayTrunk, NoneForAFlatMesh)
{
  EXPECT_FALSE(trunkOf({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f}, {0, 1, 2, 2, 1, 3}, glm::dmat4(1.0)));
  EXPECT_FALSE(trunkOf({}, {}, glm::dmat4(1.0)));
}

TEST(PlayTrunk, TheColumnIsClosedAndAsWideAsTheTrunk)
{
  const Column     c{{1.0, 2.0}, 0.3, -0.5, 4.0};
  const glm::dmat4 back = glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 10.0, 0.0));
  const auto [xyz, tris] = columnMesh(c, back, 12);
  ASSERT_EQ(xyz.size(), size_t(3 * (2 * 12 + 2)));
  ASSERT_EQ(tris.size(), size_t(12 * 12));
  for(size_t i = 0; i < 24; i++)
  {
    EXPECT_NEAR(std::hypot(xyz[i * 3] - 1.0, xyz[i * 3 + 2] - 2.0), 0.3, 1e-5);
    EXPECT_TRUE(std::abs(xyz[i * 3 + 1] - 9.5) < 1e-5 || std::abs(xyz[i * 3 + 1] - 14.0) < 1e-5);
  }
  // closed: every edge is two triangles'
  std::map<std::pair<uint32_t, uint32_t>, int> edges;
  for(size_t t = 0; t < tris.size(); t += 3)
    for(int e = 0; e < 3; e++)
    {
      const uint32_t a = tris[t + size_t(e)], b = tris[t + size_t((e + 1) % 3)];
      edges[{std::min(a, b), std::max(a, b)}]++;
    }
  for(const auto& [edge, n] : edges)
    EXPECT_EQ(n, 2);
}

}  // namespace
