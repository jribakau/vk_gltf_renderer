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
// Play mode's two controllers, as pure functions of their state, the keys held and a world they ask
// about (fv_host_play.cpp gives them the scene's): someone on foot who walks, runs and jumps, collides
// with what stands and climbs steps (collide and slide: rays against the props, the ground under the
// feet); and a small ship that flies on a throttle with flight assist (its velocity eased toward what
// the keys ask, its turn toward what the mouse asks, levelled into a bank), rests on its landing pads,
// and a camera that follows it. No Vulkan, no host: the unit tests drive them on made-up worlds.
//
// The ship's own axes are glTF's: forward +z, up +y, so the pilot's right is -x.
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

// ---- the ship ----------------------------------------------------------------------------------------

struct ShipTuning
{
  double cruise = 70.0, reverse = 15.0;   // the throttle's range, m/s
  double throttleRate = 35.0;             // m/s a second while W or S is held
  double strafe = 20.0;                   // m/s sideways and up or down
  double boost = 220.0;                   // m/s with Shift held near the ground ...
  double boostPerHeight = 0.6;            // ... and this share of the height a second higher up
  double forwardTime = 0.85, strafeTime = 0.4;  // seconds to the speed asked, along the nose and across it
  double accel = 35.0;                    // m/s² at most, cruising
  double boostAccel = 0.9;                // and boosting: this share of the speed asked, a second (no less than accel)
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
  double     throttle   = 0.0;       // the forward speed asked, m/s
  bool       heldAtZero = false;     // S brought the throttle to 0: it stays until S is pressed again
  glm::dvec2 aim{0.0};               // the mouse's move still to turn, pixels
  uint32_t   keys       = 0;
  bool       touching   = false;     // on its pads (or its hull) this step
  double     height     = 0.0;       // its pads' height over the ground under it (a large number with none)
};

// Its axes in the world
glm::dvec3 shipForward(const Ship& s);
glm::dvec3 shipUp(const Ship& s);
glm::dvec3 shipRight(const Ship& s);  // the pilot's right
// The forward speed the throttle and Shift ask for at a height over the ground
double shipTopSpeed(const Ship& s, const ShipTuning& t);
// A step of `dt` seconds; `piloted` false is a ship with nobody in it (it settles and stops)
void stepShip(Ship& s, const ShipShape& shape, const World& world, double dt, bool piloted, const ShipTuning& t = {});

// The chase camera: behind and over the ship, turning after it a moment late, further back as it speeds
struct Chase
{
  glm::dquat turn{1.0, 0.0, 0.0, 0.0};
  bool       placed = false;
  double     zoom   = 1.0;  // the wheel's: 0.5 near to 3 far
};
Pose chaseView(Chase& c, const Ship& s, double size, double dt);
// From the seat: `eye` in the ship's frame about its centre
Pose cockpitView(const Ship& s, const glm::dvec3& eye);

}  // namespace fv::play
