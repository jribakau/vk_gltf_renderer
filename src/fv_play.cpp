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

#include "fv_play_detail.hpp"

namespace fv::play {

using namespace detail;

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

  // what it stands in after all (a corner's second wall, a post or a trunk beside the way it went,
  // something that moved into it): out to a radius from it, at the knee, the chest and the head
  for(const double h : {t.stepHeight + 0.15, 1.0, t.eyeHeight - 0.2})
    for(int k = 0; k < 8; k++)
    {
      const double     a    = k * std::numbers::pi / 4.0;
      const glm::dvec3 dir  = w.facing * std::cos(a) + right * std::sin(a);
      const glm::dvec3 from = w.feet + up * h;
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

std::optional<double> sightDistance(const World& world, const Pose& view, double far)
{
  constexpr double kFirst = 0.25;  // metres: the first point down the line asked about the ground
  constexpr double kGrow  = 1.25;  // each next one this much further, and kFirst more
  constexpr double kOver  = 20.0;  // metres over a point the ground under it is asked from
  constexpr int    kHalve = 8;     // halvings of the step the line went under the ground in
  const glm::dvec3 dir = glm::normalize(view.forward);
  std::optional<double> best;
  if(world.ray)
    if(const auto hit = world.ray(view.eye, dir, far))
      best = std::max(0.0, glm::dot(hit->at - view.eye, dir));
  if(!world.ground || !world.up)
    return best;
  // the ground: down the line in steps that grow with the distance, till a point of it is under the
  // ground there; then the step it went under in halved
  const auto under = [&](double d) {
    const glm::dvec3 p  = view.eye + dir * d;
    const glm::dvec3 up = world.up(p);
    const auto       g  = world.ground(p + up * kOver, up, 2.0 * kOver);
    return g && glm::dot(p - g->at, up) <= 0.0;
  };
  const double end = best.value_or(far);
  for(double was = 0.0, d = std::min(kFirst, end);; was = d, d = std::min(d * kGrow + kFirst, end))
  {
    if(under(d))
    {
      double lo = was, hi = d;
      for(int i = 0; i < kHalve; i++)
      {
        const double mid = 0.5 * (lo + hi);
        if(under(mid))
          hi = mid;
        else
          lo = mid;
      }
      return hi;
    }
    if(d >= end)
      return best;
  }
}

Pose torchView(Torch& t, const Pose& view, std::optional<double> sight, double dt)
{
  constexpr double kRight = 0.22, kDown = 0.3, kAhead = 0.25;  // metres from the eye: a hand at the hip
  constexpr double kNearest = 1.0;   // metres: the beam crosses the view's line no nearer (a wrist turns so far)
  constexpr double kNothing = 25.0;  // and this far ahead when the view looks at nothing
  constexpr double kLag     = 0.07;  // seconds the aim follows the view behind
  const glm::dvec3 forward = glm::normalize(view.forward);
  const glm::dvec3 right   = glm::normalize(glm::cross(forward, view.up));
  const glm::dvec3 up      = glm::cross(right, forward);
  const glm::dquat want    = glm::quatLookAt(forward, up);
  const double     reach   = std::max(sight.value_or(kNothing), kNearest);
  if(!t.placed)
  {
    t.aim    = want;
    t.reach  = reach;
    t.placed = true;
  }
  t.aim = glm::normalize(glm::slerp(t.aim, want, share(dt, kLag)));
  // the reach eased in its logarithm, so a turn from a wall to a far hill is as quick as one back
  t.reach = std::exp(std::log(std::max(t.reach, kNearest)) + std::log(reach / std::max(t.reach, kNearest)) * share(dt, kLag));
  // the hand goes with the aim, as an arm does
  const glm::dvec3 ahead  = t.aim * glm::dvec3(0.0, 0.0, -1.0);
  const glm::dvec3 aimUp  = t.aim * glm::dvec3(0.0, 1.0, 0.0);
  const glm::dvec3 aside  = glm::normalize(glm::cross(ahead, aimUp));
  const glm::dvec3 at     = view.eye + aside * kRight - aimUp * kDown + ahead * kAhead;
  const glm::dvec3 shine  = glm::normalize(view.eye + ahead * t.reach - at);
  const glm::dvec3 across = glm::normalize(glm::cross(shine, aimUp));
  return {at, shine, glm::cross(across, shine)};
}

// ---- what stands in a world -------------------------------------------------------------------------

std::optional<Column> trunkOf(const std::vector<float>& xyz, const std::vector<uint32_t>& tris, const glm::dmat4& upright)
{
  const size_t n = xyz.size() / 3;
  if(n < 3 || tris.size() < 3)
    return std::nullopt;
  const auto at = [&](uint32_t i) {
    return glm::dvec3(upright * glm::dvec4(double(xyz[i * 3ull]), double(xyz[i * 3ull + 1]), double(xyz[i * 3ull + 2]), 1.0));
  };
  // the heights first, which is all most triangles need
  std::vector<float> ys(n);
  double             lo = 1e30, hi = -1e30;
  for(size_t i = 0; i < n; i++)
  {
    const double y = at(uint32_t(i)).y;
    ys[i]          = float(y);
    lo             = std::min(lo, y);
    hi             = std::max(hi, y);
  }
  const double height = hi - lo;
  if(!(height > 0.3))
    return std::nullopt;
  // cut at three heights from over the foot's roots to a man's (a share of that on something small):
  // where the triangles' edges cross each is the outline of what stands there, trunk and leaves
  const double from = lo + std::min(0.3, 0.15 * height), to = lo + std::min(1.6, 0.6 * height);
  const double cuts[3] = {from + 0.2 * (to - from), 0.5 * (from + to), from + 0.8 * (to - from)};
  std::vector<glm::dvec2> section;
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
  {
    const uint32_t v[3] = {tris[t], tris[t + 1], tris[t + 2]};
    if(v[0] >= n || v[1] >= n || v[2] >= n)
      continue;
    const double y0 = ys[v[0]], y1 = ys[v[1]], y2 = ys[v[2]];
    if(std::max({y0, y1, y2}) < cuts[0] || std::min({y0, y1, y2}) > cuts[2])
      continue;
    for(const double h : cuts)
      for(int e = 0; e < 3; e++)
      {
        const uint32_t a = v[e], b = v[(e + 1) % 3];
        const double   ya = ys[a], yb = ys[b];
        if((ya - h) * (yb - h) >= 0.0 || ya == yb)
          continue;
        const glm::dvec3 pa = at(a), pb = at(b);
        const glm::dvec3 p  = glm::mix(pa, pb, (h - pa.y) / (pb.y - pa.y));
        section.emplace_back(p.x, p.z);
      }
  }
  if(section.size() < 6)
    return std::nullopt;
  // the trunk is the inner ring of the section: in each direction from its middle, the nearest point.
  // The middle starts as the median on each axis and moves to the middle of the circle through those
  // nearest points (a least-squares fit, Kasa's), a few times over, so a branch to one side doesn't
  // pull it off the trunk.
  const auto median = [](std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  std::vector<double> xs(section.size()), zs(section.size());
  for(size_t i = 0; i < section.size(); i++)
  {
    xs[i] = section[i].x;
    zs[i] = section[i].y;
  }
  glm::dvec2       centre(median(std::move(xs)), median(std::move(zs)));
  constexpr int    kSectors = 16;
  std::vector<int> nearest(kSectors);
  std::vector<double> reach(kSectors);
  for(int round = 0; round < 4; round++)
  {
    std::fill(nearest.begin(), nearest.end(), -1);
    std::fill(reach.begin(), reach.end(), 1e30);
    for(size_t i = 0; i < section.size(); i++)
    {
      const glm::dvec2 d = section[i] - centre;
      const double     r = glm::length(d);
      const int        k = std::clamp(int(std::floor((std::atan2(d.y, d.x) + std::numbers::pi) / (2.0 * std::numbers::pi) * kSectors)), 0, kSectors - 1);
      if(r < reach[size_t(k)])
      {
        reach[size_t(k)]   = r;
        nearest[size_t(k)] = int(i);
      }
    }
    // x^2 + y^2 + D x + E y + F = 0 through them, about the middle as it is: the circle's middle is
    // (-D/2, -E/2) from there
    glm::dmat3 normal(0.0);
    glm::dvec3 right(0.0);
    glm::dvec2 sum(0.0);
    int        count = 0;
    for(const int i : nearest)
      if(i >= 0)
      {
        const glm::dvec2 q = section[size_t(i)] - centre;
        const glm::dvec3 row(q.x, q.y, 1.0);
        normal += glm::outerProduct(row, row);
        right += row * -(q.x * q.x + q.y * q.y);
        sum += section[size_t(i)];
        count++;
      }
    if(count < 3)
      return std::nullopt;
    if(std::abs(glm::determinant(normal)) > 1e-18)
    {
      const glm::dvec3 fit = glm::inverse(normal) * right;
      centre += glm::dvec2(-0.5 * fit.x, -0.5 * fit.y);
    }
    else
      centre = sum / double(count);  // on a line: no circle to fit
  }
  std::vector<double> rings;
  for(size_t k = 0; k < size_t(kSectors); k++)
    if(nearest[k] >= 0)
      rings.push_back(glm::distance(section[size_t(nearest[k])], centre));
  Column c;
  c.centre = centre;
  c.radius = std::clamp(median(std::move(rings)), 0.03, 0.8);
  c.bottom = lo - 0.5;  // into the ground, which a slope may lift on one side
  c.top    = lo + std::min(height, 8.0);
  return c;
}

std::pair<std::vector<float>, std::vector<uint32_t>> columnMesh(const Column& c, const glm::dmat4& back, int sides)
{
  sides = std::max(3, sides);
  std::vector<float>    xyz;
  std::vector<uint32_t> tris;
  const auto            put = [&](double x, double y, double z) {
    const glm::dvec3 p = glm::dvec3(back * glm::dvec4(x, y, z, 1.0));
    xyz.insert(xyz.end(), {float(p.x), float(p.y), float(p.z)});
  };
  for(int k = 0; k < sides; k++)
  {
    const double a = 2.0 * std::numbers::pi * k / sides;
    const double x = c.centre.x + c.radius * std::cos(a), z = c.centre.y + c.radius * std::sin(a);
    put(x, c.bottom, z);  // 2k
    put(x, c.top, z);     // 2k + 1
  }
  const uint32_t below = uint32_t(2 * sides), above = below + 1;
  put(c.centre.x, c.bottom, c.centre.y);
  put(c.centre.x, c.top, c.centre.y);
  for(int k = 0; k < sides; k++)
  {
    const uint32_t a0 = uint32_t(2 * k), a1 = a0 + 1, b0 = uint32_t(2 * ((k + 1) % sides)), b1 = b0 + 1;
    tris.insert(tris.end(), {a0, b0, a1, a1, b0, b1, below, b0, a0, above, a1, b1});
  }
  return {std::move(xyz), std::move(tris)};
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

  // what it flies into: the move stops short of what its centre, its nose, its tail or a wingtip would
  // meet on the way, and its speed into that goes. (A turn is not swept: a wingtip turned into
  // something turns out of it again.)
  const glm::dmat3 turn = glm::mat3_cast(s.orientation);
  if(const double speed = glm::length(s.velocity); speed > 1e-6)
  {
    constexpr double          kHullKeep = 0.25;  // metres a far point stops short
    const glm::dvec3          dir       = s.velocity / speed;
    const double              step      = speed * dt;
    double                    go        = step;
    std::optional<glm::dvec3> into;
    const auto                meet = [&](const glm::dvec3& from, double keep) {
      if(const auto hit = world.ray(from, dir, keep + step))
        if(const double d = std::max(0.0, glm::dot(hit->at - from, dir) - keep); d < go)
        {
          go   = d;
          into = hit->normal;
        }
    };
    meet(s.com, 0.5 * shape.radius);
    for(const glm::dvec3& local : shape.hull)
      meet(s.com + turn * local, kHullKeep);
    s.com += dir * go;
    if(into)
      s.velocity -= *into * std::min(0.0, glm::dot(s.velocity, *into));
  }

  // what it rests on: its pads and its hull pushed up out of the ground, its fall stopped there
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

Pose chaseView(Chase& c, const Body& b, double size, double dt, const ChaseTuning& t)
{
  if(!c.placed)
  {
    c.turn   = b.orientation;
    c.placed = true;
  }
  c.turn                 = glm::normalize(glm::slerp(c.turn, b.orientation, share(dt, t.lag)));
  const glm::dmat3 turn  = glm::mat3_cast(c.turn);
  const double     speed = glm::length(b.velocity);
  const double     back  = size * (t.back + t.backFast * std::min(speed / t.fastAt, 1.0)) * c.zoom;
  const glm::dvec3 eye   = b.at + turn * glm::dvec3(0.0, t.over * size * c.zoom, -back);
  const glm::dvec3 at    = b.at + turn * glm::dvec3(0.0, t.aimOver * size, t.aimAhead * size);
  const glm::dvec3 ahead = glm::normalize(at - eye);
  glm::dvec3       up    = turn * glm::dvec3(0.0, 1.0, 0.0);
  up                     = glm::normalize(up - ahead * glm::dot(up, ahead));
  return {eye, ahead, up};
}

Pose cockpitView(const Body& b, const glm::dvec3& eye, const glm::dvec2& gaze)
{
  // the look turned about the body's up, then raised about its own right
  const glm::dvec3 look = glm::angleAxis(gaze.x, glm::dvec3(0.0, 1.0, 0.0)) * glm::angleAxis(-gaze.y, glm::dvec3(1.0, 0.0, 0.0)) * glm::dvec3(0.0, 0.0, 1.0);
  const glm::dvec3 forward = glm::normalize(b.orientation * look);
  glm::dvec3       up      = glm::normalize(b.orientation * glm::dvec3(0.0, 1.0, 0.0));
  if(gaze != glm::dvec2(0.0))
    up = glm::normalize(up - forward * glm::dot(up, forward));
  return {b.at + b.orientation * eye, forward, up};
}

Pose clearView(const Pose& view, const glm::dvec3& target, const World& world)
{
  constexpr double kKeep  = 0.3;  // metres short of what stands between
  constexpr double kAbove = 0.4;  // metres over the ground at least
  Pose             out    = view;
  const glm::dvec3 d      = view.eye - target;
  const double     length = glm::length(d);
  if(length < 1e-6)
    return out;
  const glm::dvec3 dir = d / length;
  if(world.ray)
    if(const auto hit = world.ray(target, dir, length + kKeep))
      if(const double t = glm::dot(hit->at - target, dir); t - kKeep < length)
        out.eye = target + dir * std::max(0.2, t - kKeep);
  if(world.ground && world.up)
  {
    const glm::dvec3 up = world.up(out.eye);
    if(const auto g = world.ground(out.eye + up * 2.0, up, 2.0 + kAbove + 1.0))
      if(const double over = glm::dot(out.eye - g->at, up); over < kAbove)
        out.eye += up * (kAbove - over);
  }
  return out;
}

}  // namespace fv::play
