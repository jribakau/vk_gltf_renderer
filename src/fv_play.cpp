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

#include "fv_play.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace fv::play {

namespace {

double ease(double t)
{
  t = std::clamp(t, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// The share of what is left that a step of `dt` takes, easing at time constant `tau`: the same over
// a second however the second is cut into frames
double share(double dt, double tau)
{
  return tau > 0.0 ? 1.0 - std::exp(-dt / tau) : 1.0;
}

glm::dvec3 level(const glm::dvec3& v, const glm::dvec3& up)
{
  return v - up * glm::dot(v, up);
}

// A unit direction square to `up`, the nearest to `v`; any level one for `v` along up
glm::dvec3 levelUnit(const glm::dvec3& v, const glm::dvec3& up)
{
  for(const glm::dvec3& d : {v, glm::dvec3(0.0, 0.0, -1.0), glm::dvec3(1.0, 0.0, 0.0)})
  {
    const glm::dvec3 l = level(d, up);
    if(glm::length(l) > 1e-6)
      return glm::normalize(l);
  }
  return {1.0, 0.0, 0.0};
}

double key(uint32_t keys, Key k)
{
  return (keys & k) ? 1.0 : 0.0;
}

}  // namespace

Pose blend(const Pose& a, const Pose& b, double t)
{
  const double     e  = ease(t);
  const glm::dquat qa = glm::quatLookAt(glm::normalize(a.forward), glm::normalize(a.up));
  const glm::dquat qb = glm::quatLookAt(glm::normalize(b.forward), glm::normalize(b.up));
  const glm::dquat q  = glm::normalize(glm::slerp(qa, qb, e));
  return {glm::mix(a.eye, b.eye, e), q * glm::dvec3(0.0, 0.0, -1.0), q * glm::dvec3(0.0, 1.0, 0.0)};
}

// ---- on foot ---------------------------------------------------------------------------------------

Walker walkerAt(const Pose& camera, const World& world, double reach, const WalkerTuning& t)
{
  Walker           w;
  const glm::dvec3 up = world.up(camera.eye);
  w.feet              = camera.eye - up * t.eyeHeight;
  if(const auto g = world.ground(camera.eye, up, reach))
  {
    w.feet     = g->at;
    w.grounded = true;
  }
  const glm::dvec3 forward = glm::normalize(camera.forward);
  w.facing                 = levelUnit(forward, up);
  w.pitch                  = std::clamp(std::asin(std::clamp(glm::dot(forward, up), -1.0, 1.0)), -t.maxPitch, t.maxPitch);
  w.lastGround             = w.feet;
  return w;
}

void stepWalker(Walker& w, const World& world, double dt, const WalkerTuning& t)
{
  if(dt <= 0.0)
    return;
  const glm::dvec3 up = world.up(w.feet);

  // the look: the view follows the mouse a few milliseconds behind, so uneven batches of its moves
  // turn it evenly
  glm::dvec2 take = w.look * share(dt, t.lookSmoothing);
  if(glm::length(w.look - take) < 0.05)
    take = w.look;
  w.look -= take;
  w.facing = levelUnit(glm::angleAxis(-take.x * t.turnPerPixel, up) * w.facing, up);
  w.pitch  = std::clamp(w.pitch - take.y * t.turnPerPixel, -t.maxPitch, t.maxPitch);

  // the speed asked on the level, which its own eases toward: at once on the ground, slowly in the air
  const glm::dvec3 right = glm::normalize(glm::cross(w.facing, up));
  glm::dvec3 wish = w.facing * (key(w.keys, kForward) - key(w.keys, kBack)) + right * (key(w.keys, kRight) - key(w.keys, kLeft));
  if(glm::length(wish) > 1e-9)
    wish = glm::normalize(wish) * ((w.keys & kFast) ? t.run : t.walk);
  double     rising  = glm::dot(w.velocity, up);
  glm::dvec3 onLevel = w.velocity - up * rising;
  onLevel += (wish - onLevel) * share(dt, w.grounded ? t.grip : t.airGrip);
  if(w.jumps > 0 && w.grounded)
  {
    rising     = t.jump;
    w.grounded = false;
  }
  w.jumps = 0;
  rising  = w.grounded ? 0.0 : rising - t.gravity * dt;

  // the move on the level, slid along what it meets (collide and slide): rays from the knee to the top
  // of the head in its direction. What is lower than the knee is stepped up onto below.
  glm::dvec3 move = onLevel * dt;
  for(int pass = 0; pass < 3 && glm::length(move) > 1e-8; pass++)
  {
    const double           length = glm::length(move);
    const glm::dvec3       dir    = move / length;
    std::optional<Surface> wall;
    double                 wallAt = 1e30;
    for(const double h : {t.stepHeight + 0.1, 0.9, 1.25, 1.55, t.eyeHeight + 0.15})
    {
      // a wall met at a slant is a radius away further along the move than one met square on
      const glm::dvec3 from = w.feet + up * h;
      if(const auto hit = world.ray(from, dir, length + 3.0 * t.radius))
        if(const double d = glm::dot(hit->at - from, dir); d < wallAt)
        {
          wallAt = d;
          wall   = hit;
        }
    }
    glm::dvec3 n = wall ? level(wall->normal, up) : glm::dvec3(0.0);
    if(glm::length(n) < 1e-6)
    {
      w.feet += move;
      break;
    }
    // on up to a radius from it, square to it; what is left of the move slides along it
    n                   = glm::normalize(n);
    const double square = std::max(-glm::dot(dir, n), 1.0 / 3.0);
    const double gone   = std::clamp(wallAt - t.radius / square, 0.0, length);
    w.feet += dir * gone;
    if(gone >= length)
      break;
    move = dir * (length - gone);
    move -= n * std::min(0.0, glm::dot(move, n));
    onLevel -= n * std::min(0.0, glm::dot(onLevel, n));
  }

  // what it stands in after all (a corner's second wall, something that moved into it): out to a
  // radius from it, at the chest
  for(int k = 0; k < 8; k++)
  {
    const double     a    = k * std::numbers::pi / 4.0;
    const glm::dvec3 dir  = w.facing * std::cos(a) + right * std::sin(a);
    const glm::dvec3 from = w.feet + up * 1.0;
    if(const auto hit = world.ray(from, dir, t.radius))
      if(const glm::dvec3 l = level(hit->normal, up); glm::length(l) > 1e-6)
      {
        const glm::dvec3 n     = glm::normalize(l);
        const double     apart = -glm::dot(hit->at - from, n);  // square to it
        if(apart < t.radius)
          w.feet += n * (t.radius - apart);
      }
  }

  // up and down, then the ground under the feet: stood on when they reach it, stepped up onto when it
  // is within a step over them, kept to down a step or a slope while on it
  w.feet += up * (rising * dt);
  const bool   wasGrounded = w.grounded;
  const double snapDown    = wasGrounded ? t.stepHeight : 0.0;
  const double above       = t.stepHeight + 0.1;
  w.grounded               = false;
  if(const auto g = world.ground(w.feet + up * above, up, above + snapDown + std::max(0.0, -rising * dt)))
  {
    const double over = glm::dot(w.feet - g->at, up);
    if(over <= 0.0 || (wasGrounded && over <= snapDown && rising <= 0.0))
    {
      if(wasGrounded)
        w.eyeLag += over;  // a step: the eye stays where it was and follows
      w.feet       = g->at;
      w.grounded   = true;
      w.lastGround = w.feet;
      rising       = 0.0;
    }
  }
  w.velocity = onLevel + up * rising;
  w.eyeLag   = std::clamp(w.eyeLag * (1.0 - share(dt, t.eyeFollow)), -1.0, 1.0);

  // fallen out of the world: back where it last stood
  if(!w.grounded && glm::dot(w.lastGround - w.feet, up) > 200.0)
  {
    w.feet     = w.lastGround;
    w.velocity = glm::dvec3(0.0);
  }
}

Pose walkerView(const Walker& w, const World& world, const WalkerTuning& t)
{
  const glm::dvec3 up      = world.up(w.feet);
  const glm::dvec3 facing  = levelUnit(w.facing, up);
  const glm::dvec3 right   = glm::normalize(glm::cross(facing, up));
  const glm::dvec3 forward = facing * std::cos(w.pitch) + up * std::sin(w.pitch);
  return {w.feet + up * (t.eyeHeight + w.eyeLag), forward, glm::normalize(glm::cross(right, forward))};
}

// ---- the ship ----------------------------------------------------------------------------------------

glm::dvec3 shipForward(const Ship& s)
{
  return glm::normalize(s.orientation * glm::dvec3(0.0, 0.0, 1.0));
}

glm::dvec3 shipUp(const Ship& s)
{
  return glm::normalize(s.orientation * glm::dvec3(0.0, 1.0, 0.0));
}

glm::dvec3 shipRight(const Ship& s)
{
  return glm::normalize(s.orientation * glm::dvec3(-1.0, 0.0, 0.0));
}

double shipSpeedRange(const Ship& s, double speedCap, const ShipTuning& t)
{
  const double room = std::clamp(t.perClearance * s.clearance, t.cruise, std::max(speedCap, t.cruise));
  return (s.keys & kFast) ? room * t.boost : room;
}

void stepShip(Ship& s, const ShipShape& shape, const World& world, double dt, bool piloted, const ShipTuning& t)
{
  if(dt <= 0.0)
    return;
  const uint32_t   keys = piloted ? s.keys : 0u;
  const glm::dvec3 up   = world.up(s.com);
  const glm::dvec3 F = shipForward(s), U = shipUp(s), R = shipRight(s);

  // the throttle: W opens it, S closes it, stopping at nothing until S is let go and pressed again,
  // which backs
  const bool ahead = keys & kForward, back = keys & kBack;
  if(!back)
    s.heldAtZero = false;
  if(ahead && !back)
    s.throttle = std::min(1.0, s.throttle + t.throttleRate * dt);
  else if(back && !ahead && !s.heldAtZero)
  {
    const double next = s.throttle - t.throttleRate * dt;
    if(s.throttle > 0.0 && next <= 0.0)
    {
      s.throttle   = 0.0;
      s.heldAtZero = true;
    }
    else
      s.throttle = std::max(-t.reverse, next);
  }
  if(!piloted)
    s.throttle = 0.0;

  // the room it has: its height over the ground (last step's), or what is ahead of its nose if nearer
  s.clearance = s.height >= 0.0 ? s.height : 1e30;
  if(const auto hit = world.ray(s.com, F, std::min(s.clearance, 4.0 * std::max(world.speedCap, t.cruise) / t.perClearance)))
    s.clearance = std::min(s.clearance, glm::dot(hit->at - s.com, F));

  // flight assist: the velocity eased toward what the throttle and the keys ask, along the nose and
  // across it each at its own pace, no faster than the engines push, which grows with the speed so
  // it slows from fast as soon as it got there. Nobody aboard, it falls.
  const double forward = s.throttle * shipSpeedRange(Ship{.keys = keys, .clearance = s.clearance}, world.speedCap, t);
  const double across  = key(keys, kRight) - key(keys, kLeft);
  const double lift    = key(keys, kUp) - key(keys, kDown);
  if(piloted)
  {
    const glm::dvec3 dv    = F * forward + R * (across * t.strafe) + U * (lift * t.strafe) - s.velocity;
    const double     along = glm::dot(dv, F);
    glm::dvec3       step  = F * (along * share(dt, t.forwardTime)) + (dv - F * along) * share(dt, t.strafeTime);
    const double     most  = std::max(t.accel, t.accelShare * glm::length(s.velocity)) * dt;
    if(glm::length(step) > most)
      step *= most / glm::length(step);
    s.velocity += step;
  }
  else
    s.velocity -= up * (9.81 * dt);

  // the turn: the mouse's move taken a little at a time, at no more than the ship's rates; what they
  // leave waits for the next step, a little of it at most. Left and right turn the nose about the up
  // it would have unbanked, so leaning into a turn doesn't pitch it; up and down about its own right.
  const glm::dvec3 flat     = up - F * glm::dot(up, F);
  const bool       levelled = glm::length(flat) > 0.25;  // not pointing near straight up or down
  const glm::dvec3 unbanked = levelled ? glm::normalize(flat) : U;
  glm::dvec3       want{0.0};
  if(piloted)
  {
    glm::dvec2 take = s.aim * share(dt, t.aimTime);
    if(glm::length(s.aim - take) < 0.05)
      take = s.aim;
    s.aim -= take;
    const double pitch = -take.y * t.turnPerPixel, yaw = -take.x * t.turnPerPixel;
    want.x             = std::clamp(pitch / dt, -t.pitchRate, t.pitchRate);
    want.y             = std::clamp(yaw / dt, -t.yawRate, t.yawRate);
    s.aim += glm::dvec2(-(yaw - want.y * dt), -(pitch - want.x * dt)) / t.turnPerPixel;
    const double queue = t.aimQueue / t.turnPerPixel;
    if(glm::length(s.aim) > queue)
      s.aim *= queue / glm::length(s.aim);
    want.z = (key(keys, kRollRight) - key(keys, kRollLeft)) * t.rollRate;
    // no hand on the roll: it leans into its turn and comes back level out of it, unless it points
    // straight up or down
    if(!(keys & (kRollLeft | kRollRight)) && levelled)
    {
      const double bank = std::atan2(glm::dot(glm::cross(unbanked, U), F), glm::dot(unbanked, U));
      want.z            = std::clamp((-want.y * t.bankPerYaw - bank) * t.levelGain, -t.rollRate, t.rollRate);
    }
  }
  s.spin += (want - s.spin) * share(dt, t.spinTime);
  s.orientation = glm::normalize(glm::angleAxis(s.spin.x * dt, R) * glm::angleAxis(s.spin.y * dt, unbanked)
                                 * glm::angleAxis(s.spin.z * dt, F) * s.orientation);

  // what it flies into: the move stops short of it, and its speed into it goes
  if(const double speed = glm::length(s.velocity); speed > 1e-6)
  {
    const glm::dvec3 dir  = s.velocity / speed;
    const double     keep = 0.5 * shape.radius;
    if(const auto hit = world.ray(s.com, dir, keep + speed * dt))
    {
      const double d = glm::dot(hit->at - s.com, dir);
      s.com += dir * std::max(0.0, d - keep);
      s.velocity -= hit->normal * std::min(0.0, glm::dot(s.velocity, hit->normal));
    }
    else
      s.com += s.velocity * dt;
  }

  // what it rests on: its pads and its hull pushed up out of the ground, its fall stopped there
  const glm::dmat3 turn    = glm::mat3_cast(s.orientation);
  double           deepest = 0.0;
  bool             onPads  = false;
  const auto       touch   = [&](const glm::dvec3& local, bool pad) {
    const glm::dvec3 p = s.com + turn * local;
    if(const auto g = world.ground(p + up * 3.0, up, 3.5))
    {
      const double in = glm::dot(g->at - p, up);
      deepest         = std::max(deepest, in);
      onPads |= pad && in > -0.03;
    }
  };
  for(const glm::dvec3& p : shape.pads)
    touch(p, true);
  for(const glm::dvec3& p : shape.hull)
    touch(p, false);
  s.touching = deepest > 0.0 || onPads;
  if(deepest > 0.0)
  {
    s.com += up * deepest;
    s.velocity -= up * std::min(0.0, glm::dot(s.velocity, up));
  }
  if(s.touching)
  {
    // the pads drag it to a stop when nothing drives it ...
    if(forward == 0.0 && across == 0.0)
    {
      const glm::dvec3 slide = level(s.velocity, up);
      const double     speed = glm::length(slide);
      if(speed > 1e-9)
        s.velocity -= slide * (std::min(speed, t.friction * dt) / speed);
    }
    // ... and it sits level on them, unless it is climbing away
    if(onPads && lift <= 0.0 && std::abs(s.throttle) < 1.0)
    {
      const glm::dvec3 nose = levelUnit(F, up);
      const glm::dquat flat = glm::quat_cast(glm::dmat3(glm::cross(up, nose), up, nose));
      s.orientation         = glm::normalize(glm::slerp(s.orientation, flat, share(dt, t.settleTime)));
      s.spin.x *= 1.0 - share(dt, t.settleTime);
      s.spin.z *= 1.0 - share(dt, t.settleTime);
    }
  }

  // how high its pads are over the ground under it (-1 when there is none to tell)
  double padsUnder = 0.0;
  for(const glm::dvec3& p : shape.pads)
    padsUnder = std::max(padsUnder, -glm::dot(turn * p, up));
  const auto g = world.ground(s.com, up, 1.0e7);
  s.height     = g ? std::max(0.0, glm::dot(s.com - g->at, up) - padsUnder) : -1.0;
}

Pose chaseView(Chase& c, const Ship& s, double size, double dt)
{
  if(!c.placed)
  {
    c.turn   = s.orientation;
    c.placed = true;
  }
  c.turn                 = glm::normalize(glm::slerp(c.turn, s.orientation, share(dt, 0.16)));
  const glm::dmat3 turn  = glm::mat3_cast(c.turn);
  const double     speed = glm::length(s.velocity);
  const double     back  = size * (1.05 + 0.25 * std::min(speed / 120.0, 1.0)) * c.zoom;
  const glm::dvec3 eye   = s.com + turn * glm::dvec3(0.0, 0.24 * size * c.zoom, -back);
  const glm::dvec3 at    = s.com + turn * glm::dvec3(0.0, 0.08 * size, 0.6 * size);
  const glm::dvec3 ahead = glm::normalize(at - eye);
  glm::dvec3       up    = turn * glm::dvec3(0.0, 1.0, 0.0);
  up                     = glm::normalize(up - ahead * glm::dot(up, ahead));
  return {eye, ahead, up};
}

Pose cockpitView(const Ship& s, const glm::dvec3& eye)
{
  return {s.com + s.orientation * eye, shipForward(s), shipUp(s)};
}

}  // namespace fv::play
