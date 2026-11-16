/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// Play mode's controllers, as pure functions of their state, the keys held and a world they ask about
// (fv_host_play.cpp gives them the scene's): someone on foot who walks, runs and jumps, collides with
// what stands and climbs steps (collide and slide: rays against the props, the ground under the feet),
// with a torch in hand; a small ship that flies on a throttle with flight assist (its velocity eased
// toward what the keys ask, its turn toward what the mouse asks, levelled into a bank), stops short of
// what its nose, tail or wingtips would meet and rests on its landing pads; a car on four suspension
// rays (fv_play_car.cpp); and the cameras that follow them, behind or from the seat. A tree's trunk is
// found from its mesh (trunkOf) for the host to collide with. No Vulkan, no host: the unit tests drive
// them on made-up worlds.
//
// The ship's and the car's own axes are glTF's: forward +z, up +y, so the driver's right is -x.
//

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace fv::play {

// The keys the app holds down, a bit each (the /input event `play`'s `k`)
enum Key : uint32_t
{
  kForward   = 1u << 0,   // W
  kBack      = 1u << 1,   // S
  kLeft      = 1u << 2,   // A
  kRight     = 1u << 3,   // D
  kUp        = 1u << 4,   // Space: jump, climb
  kDown      = 1u << 5,   // Ctrl or C: sink
  kFast      = 1u << 6,   // Shift: run, boost
  kRollLeft  = 1u << 7,   // Q
  kRollRight = 1u << 8,   // E
  kUse       = 1u << 9,   // F: get in, get out
  kView      = 1u << 10,  // V: the cockpit's view or the chase camera's
  kTorch     = 1u << 11,  // L: the torch on or off
  kShipHere  = 1u << 12,  // B: a ship put down ahead
  kEye       = 1u << 13,  // X: the eye adapting to the view, or the exposure as it is
  kCarHere   = 1u << 14,  // N: a car put down ahead
};

struct Surface
{
  glm::dvec3 at{0.0};
  glm::dvec3 normal{0.0, 1.0, 0.0};  // toward where the question came from
};

// What the controllers ask of the scene
struct World
{
  // Which way is up at a point: the scene's +y, a planet's vertical away from where the figures stand
  std::function<glm::dvec3(const glm::dvec3&)> up;
  // The highest ground straight under `from` (along -up) within `reach`: the floor, the land, a prop
  std::function<std::optional<Surface>(const glm::dvec3& from, const glm::dvec3& up, double reach)> ground;
  // The nearest surface a ray from `from` along `dir` (unit) meets within `reach`: walls, props
  std::function<std::optional<Surface>(const glm::dvec3& from, const glm::dvec3& dir, double reach)> ray;
  // The fastest a ship's full throttle goes here, m/s, however much room it has (Shift goes past it):
  // what crosses this world in a sensible time (the host's: a planet's radius in ten seconds)
  double speedCap = 300.0;
};

// Where a camera is and how it is turned
struct Pose
{
  glm::dvec3 eye{0.0};
  glm::dvec3 forward{0.0, 0.0, -1.0};
  glm::dvec3 up{0.0, 1.0, 0.0};
};
// Partway from one pose to another, eased at both ends (t 0 to 1)
Pose blend(const Pose& a, const Pose& b, double t);

// ---- on foot ---------------------------------------------------------------------------------------

struct WalkerTuning
{
  double eyeHeight  = 1.65;
  double radius     = 0.3;   // how near a wall the body comes
  double stepHeight = 0.45;  // what it walks up onto
  double walk = 4.2, run = 8.5;      // metres a second
  double grip = 0.08, airGrip = 0.9;  // seconds to reach the speed asked, on the ground and in the air
  double gravity = 9.81, jump = 4.9;  // m/s², the speed a jump leaves the ground at
  double turnPerPixel = 0.0022;       // radians the view turns for a pixel of the mouse
  double lookSmoothing = 0.015;       // seconds the view follows the mouse behind
  double maxPitch     = 1.52;         // radians up or down
  double eyeFollow    = 0.07;         // seconds the eye takes to follow a step up or down
};

struct Walker
{
  glm::dvec3 feet{0.0};
  glm::dvec3 velocity{0.0};
  glm::dvec3 facing{0.0, 0.0, -1.0};  // level, where the body faces
  double     pitch    = 0.0;          // the look over the level, radians
  bool       grounded = false;
  double     eyeLag   = 0.0;  // a step's rise the eye has yet to follow, metres
  glm::dvec2 look{0.0};       // the mouse's move still to turn, pixels
  uint32_t   keys = 0;        // held
  int        jumps = 0;       // presses of the jump key not yet taken
  glm::dvec3 lastGround{0.0};  // where it last stood, to come back to after a fall through the world
};

// A step of `dt` seconds: the look, then the move with its collisions
void stepWalker(Walker& w, const World& world, double dt, const WalkerTuning& t = {});
// What it sees
Pose walkerView(const Walker& w, const World& world, const WalkerTuning& t = {});
// Someone standing where a camera is, looking where it looks: dropped onto the ground under it when
// there is one within `reach`
Walker walkerAt(const Pose& camera, const World& world, double reach, const WalkerTuning& t = {});

// A torch held in the right hand: beside and under the eye, shining at what the view looks at, so its
// pool is where the view's middle is; its aim a moment behind the view's, so it sways after a turn
struct Torch
{
  bool       on = false;
  glm::dquat aim{1.0, 0.0, 0.0, 0.0};
  double     reach  = 0.0;  // metres ahead the beam crosses the view's line, eased
  bool       placed = false;
};
// How far ahead the view's line meets what stands there or the ground, within `far`; none: nothing
// does (the sky, or ground further off)
std::optional<double> sightDistance(const World& world, const Pose& view, double far);
// Where it is and where it shines (`forward`) for the view as it is now, after `dt` seconds: at what
// the view looks at `sight` metres ahead (sightDistance), or some way ahead when it looks at nothing
Pose torchView(Torch& t, const Pose& view, std::optional<double> sight, double dt);

// ---- what stands in a world -------------------------------------------------------------------------

// A tree's trunk as a column standing up `upright`'s +y: its middle on the plan, its radius, and how far
// down and up it goes. What someone walks into among a tree's triangles, its leaves and twigs left out.
struct Column
{
  glm::dvec2 centre{0.0};  // x and z
  double     radius = 0.0;
  double     bottom = 0.0, top = 0.0;
};
// The trunk of a tree whose vertices are `xyz` (three a vertex) and triangles `tris` (three indices a
// triangle), taken by `upright` to a frame with +y up: its triangles cut at a few heights between its
// foot and a man's height, and the column the inner ring of that section (the nearest point in each
// direction from its middle), so leaves and twigs further out leave it as it is. None for a mesh too
// flat or with too little there to tell.
std::optional<Column> trunkOf(const std::vector<float>& xyz, const std::vector<uint32_t>& tris, const glm::dmat4& upright);
// The column as a closed mesh of `sides` sides with its caps, in the frame `back` takes the upright
// one to (its vertices three a vertex, its triangles three indices a triangle)
std::pair<std::vector<float>, std::vector<uint32_t>> columnMesh(const Column& c, const glm::dmat4& back, int sides = 10);

// ---- the ship ----------------------------------------------------------------------------------------

// The throttle is a share of a speed range that follows the room the ship has: its clearance (its
// height over the ground, or what is ahead of its nose if nearer) at `perClearance` a second, no slower
// than `cruise` and no faster than the world's speedCap. A dive at full throttle slows by itself as
// the ground comes up; a climb speeds up but stops at the cap, so nothing runs away.
struct ShipTuning
{
  double cruise       = 70.0;  // m/s: full throttle near the ground
  double perClearance = 0.5;   // full throttle higher up: this share of the clearance a second
  double reverse      = 0.25;  // the throttle backs to this share of the range
  double throttleRate = 0.6;   // share of the range a second while W or S is held
  double boost        = 3.0;   // Shift: the range this many times over (past the cap too)
  double strafe       = 20.0;  // m/s sideways and up or down
  double forwardTime = 0.85, strafeTime = 0.4;  // seconds to the speed asked, along the nose and across it
  double accel      = 35.0;  // m/s² at least ...
  double accelShare = 1.5;   // ... and this share of the speed a second, so it slows as fast as it got going
  double turnPerPixel = 0.0016;           // radians the nose turns for a pixel of the mouse
  double aimTime      = 0.06;             // seconds the turn follows the mouse behind
  double pitchRate = 1.7, yawRate = 1.1, rollRate = 2.6;  // rad/s at most
  double spinTime   = 0.12;               // seconds the ship takes to reach a turn rate
  double aimQueue   = 0.6;                // radians of mouse move kept waiting at most (a flick is no spin)
  double bankPerYaw = 0.5;                // radians of bank for a rad/s of yaw: it leans into a turn
  double levelGain  = 3.0;                // how hard it rolls back to the bank it should have
  double settleTime = 0.45;               // seconds it takes to sit level once on its pads
  double friction   = 3.0;                // m/s² its pads drag it to a stop with
};

// What the ship is shaped like, in its own frame about its centre of mass: what touches the ground
struct ShipShape
{
  std::vector<glm::dvec3> pads;   // its landing pads' contact points
  std::vector<glm::dvec3> hull;   // its far points: nose, tail, wingtips, top and bottom
  double                  radius = 5.0;  // about its centre, for what it flies into
};

struct Ship
{
  glm::dvec3 com{0.0};               // its centre of mass in the world
  glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
  glm::dvec3 velocity{0.0};          // m/s, the world's
  glm::dvec3 spin{0.0};              // rad/s: x pitch (nose up), y yaw (nose left), z roll (right wing down)
  double     throttle   = 0.0;       // the share of the speed range asked forward: -reverse to 1
  bool       heldAtZero = false;     // S brought the throttle to 0: it stays until S is pressed again
  glm::dvec2 aim{0.0};               // the mouse's move still to turn, pixels
  uint32_t   keys       = 0;
  bool       touching   = false;     // on its pads (or its hull) this step
  double     height     = -1.0;      // its pads' height over the ground under it (-1 with none under it)
  double     clearance  = 1e30;      // the room it has: that height, or what is ahead if nearer (1e30: nothing)
};

// Its axes in the world
glm::dvec3 shipForward(const Ship& s);
glm::dvec3 shipUp(const Ship& s);
glm::dvec3 shipRight(const Ship& s);  // the pilot's right
// What full throttle is where it is, m/s (the range the throttle is a share of), Shift's included
double shipSpeedRange(const Ship& s, double speedCap, const ShipTuning& t = {});
// A step of `dt` seconds; `piloted` false is a ship with nobody in it (it settles and stops)
void stepShip(Ship& s, const ShipShape& shape, const World& world, double dt, bool piloted, const ShipTuning& t = {});

// ---- the car -------------------------------------------------------------------------------------------
//
// A raycast vehicle, as games drive one: a rigid body on four suspension rays. Each wheel's spring and
// damper push along the ground's normal where its ray meets the ground, with the force its tyre then has
// to hold the road; the tyres' friction is solved as impulses (drive and brakes first, then the grip
// across, to a friction circle), the side grip growing with the slip angle so it corners and slides as
// a car does, and holding it still when slow. Its body's far points meet walls and the ground as the
// ship's hull does. An automatic gearbox: S stops it, then backs; Space is the handbrake. Parked, it
// sleeps, and costs nothing until a pedal or its ground wakes it. Its own axes are glTF's, as the ship's:
// forward +z, up +y, its left +x.

struct CarTuning
{
  double mass      = 2200.0;  // kg: an off-road truck's
  double frequency = 1.3;     // Hz each wheel's spring bounces at under its share of the weight
  double damping   = 0.35;    // share of critical, pushed in
  double rebound   = 2.0;     // the damper's pull coming back out, this many times its push going in
  double travelUp = 0.12, travelDown = 0.18;  // metres the wheels move from where the file has them (past the top: a stop)
  double maxLoad = 5.0;       // a spring's push at most this many times its share of the weight
  double grip      = 1.0;     // the tyres' friction, μ
  double cornering = 12.0;    // side force for a radian of slip, per newton of load (the peak at 1/12 rad)
  double slideGrip = 0.85;    // a share of the peak left to a tyre slid well past it
  double lowSpeed  = 1.0;     // m/s: slower than this a tyre holds as it stands
  double handbrakeGrip = 0.45;  // the rear's side grip with the handbrake pulled, a share
  double rolling   = 0.015;   // rolling resistance, a share of the load
  double power = 120e3, tractive = 9000.0;  // W at the wheels, and N at most
  double boost      = 1.5;    // Shift: the power this many times over
  double driveFront = 0.5;    // the share of the drive to the front wheels: 4x4
  double topSpeed = 38.0, reverseSpeed = 8.0;  // m/s
  double brake = 16000.0, frontBrake = 0.6;    // N all four together, and the front's share
  double engineBrake = 600.0;                  // N on the driven wheels off the throttle
  double holdBelow   = 0.3;                    // m/s: slower, with no pedal, the brakes hold it
  double dragArea = 0.95, air = 1.225;         // Cd·A, m²; kg/m³
  double lock = 0.6, minLock = 0.02;  // radians the steering turns at most, and at least however fast
  double turnLimit = 1.1;             // g sideways the steering's lock is cut to as it speeds
  double steerRate = 2.2, returnRate = 4.0;  // rad/s the steering turns, and comes back
  double substep   = 1.0 / 240.0;     // seconds a step of its integration is at most
  double stepUp    = 0.25;            // metres over its travel's top a wheel's ground may come: higher is a wall
  double sleepSpeed = 0.05, sleepSpin = 0.03, sleepAfter = 0.4;  // m/s, rad/s, seconds still to sleep
  double rightAfter = 2.5, rightTime = 0.8;  // seconds on its roof before it rights itself, and to do it
  double hullGrip   = 0.5, hullBounce = 0.1;  // its body against what it meets
  double turnPerPixel = 0.0022;              // radians the driver's look turns for a pixel of the mouse
  double lookYaw = 2.3, lookPitch = 1.2;     // radians the look turns at most, aside and up or down
  double lookBack = 1.0, lookReturn = 0.6;   // seconds after the mouse stops, and to come back ahead (moving)
};

struct CarWheel
{
  glm::dvec3 hub{0.0};  // its middle in the car's frame about its centre of mass, as the file has it
  double     radius = 0.4;
};

// The car's shape about its centre of mass: its wheels front left, front right, rear left, rear right
// (left is +x), and its body's far points
struct CarShape
{
  std::array<CarWheel, 4> wheels;
  std::vector<glm::dvec3> hull;
  glm::dvec3              half{0.95, 0.85, 2.4};  // the body's half sizes
  double                  radius = 2.6;           // about its centre
};
// A car's shape from its wheels and its body's box (`lo`, `hi`), all about its centre of mass: the body's
// corners (their bottom no lower than a little over the tyres'), its bumpers' middles, its sides', its
// roof's, and the tyres' fronts and backs
CarShape carShape(const std::array<CarWheel, 4>& wheels, const glm::dvec3& lo, const glm::dvec3& hi);

struct CarWheelState
{
  double                 length   = 0.12;  // the suspension's, from the top of its travel down
  double                 spin     = 0.0;   // radians about its axle
  double                 rate     = 0.0;   // rad/s
  double                 steer    = 0.0;   // radians, its own (the inner front wheel turns more)
  double                 load     = 0.0;   // N its tyre holds the ground with
  bool                   touching = false;
  std::optional<Surface> ground;           // the plane it meets, asked once a step
  glm::dvec3             normal{0.0, 1.0, 0.0};  // that plane's normal, eased
};

struct Car
{
  glm::dvec3 com{0.0};
  glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
  glm::dvec3 velocity{0.0};  // m/s, the world's
  glm::dvec3 spin{0.0};      // rad/s, the world's
  uint32_t   keys = 0;
  glm::dvec2 look{0.0};      // the mouse's move still to take, pixels
  glm::dvec2 gaze{0.0};      // the driver's look about the car: yaw (left +) and pitch (up +), radians
  double     lookIdle = 0.0; // seconds since the mouse last moved
  double     steer    = 0.0; // radians: the steering's turn, as one front wheel in the middle would have it
  int        gear     = 1;   // 1 drive, -1 reverse
  bool       handbrake = false;
  double     throttle  = 0.0;  // -1 to 1, as the pedals have it this step
  std::array<CarWheelState, 4> wheels;
  bool       asleep   = false;
  double     still    = 0.0;  // seconds at rest
  double     nap      = 0.0;  // seconds asleep since its ground was last looked at
  double     tipped   = 0.0;  // seconds on its side or roof
  double     righting = 0.0;  // seconds left of righting itself
  glm::dvec3 lastGround{0.0}; // where it last stood, to come back to after a fall through the world
};

glm::dvec3 carForward(const Car& c);
glm::dvec3 carUp(const Car& c);
glm::dvec3 carLeft(const Car& c);
// Its speed along its nose, m/s: backing is negative
double carSpeed(const Car& c);
// The height of its centre over level ground when it stands at rest on its wheels
double carRideHeight(const CarShape& shape);
// A car at `com` turned `q`, its suspension as its file has it
Car carAt(const glm::dvec3& com, const glm::dquat& q, const CarShape& shape, const CarTuning& t = {});
// A step of `dt` seconds; `driven` false is a car with nobody at the wheel (every brake on)
void stepCar(Car& c, const CarShape& shape, const World& world, double dt, bool driven, const CarTuning& t = {});

// ---- cameras ---------------------------------------------------------------------------------------------

// What a camera follows: where its centre is, how it is turned (its own axes glTF's), how fast it goes
struct Body
{
  glm::dvec3 at{0.0};
  glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
  glm::dvec3 velocity{0.0};
};

// The chase camera: behind and over what it follows, turning after it a moment late, further back as it
// speeds
struct Chase
{
  glm::dquat turn{1.0, 0.0, 0.0, 0.0};
  bool       placed = false;
  double     zoom   = 1.0;  // the wheel's: 0.5 near to 3 far
};
// How far behind and over, in its sizes, and how quickly it turns after it (the ship's)
struct ChaseTuning
{
  double lag      = 0.16;  // seconds
  double back     = 1.05;  // sizes behind ...
  double backFast = 0.25;  // ... and this many more at `fastAt` m/s
  double fastAt   = 120.0;
  double over     = 0.24;  // sizes over
  double aimOver = 0.08, aimAhead = 0.6;  // what it looks at, in sizes over and ahead of the centre
};
// A car's: nearer and lower, turning after it quicker
inline constexpr ChaseTuning kCarChase{.lag = 0.1, .back = 1.15, .backFast = 0.15, .fastAt = 40.0, .over = 0.3, .aimOver = 0.12, .aimAhead = 0.5};
Pose chaseView(Chase& c, const Body& b, double size, double dt, const ChaseTuning& t = {});
// From a seat: `eye` in its frame about its centre, the look turned `gaze` (yaw, pitch) about it
Pose cockpitView(const Body& b, const glm::dvec3& eye, const glm::dvec2& gaze = glm::dvec2(0.0));
// The view brought in front of what stands between it and `target`, and over the ground
Pose clearView(const Pose& view, const glm::dvec3& target, const World& world);

inline Body bodyOf(const Ship& s)
{
  return {s.com, s.orientation, s.velocity};
}
inline Pose chaseView(Chase& c, const Ship& s, double size, double dt)
{
  return chaseView(c, bodyOf(s), size, dt);
}
inline Pose cockpitView(const Ship& s, const glm::dvec3& eye)
{
  return cockpitView(bodyOf(s), eye);
}
// What the chase follows of a car: its heading and pitch, never its roll, turned round it by the
// driver's look
Body chaseBody(const Car& c, const glm::dvec3& up);
inline Body bodyOf(const Car& c)
{
  return {c.com, c.orientation, c.velocity};
}

}  // namespace fv::play
