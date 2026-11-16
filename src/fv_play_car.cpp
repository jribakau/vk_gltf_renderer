/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Play mode's car (fv_play.hpp): a raycast vehicle. A rigid body on four suspension rays, its tyres'
// friction solved as impulses each substep: drive and brakes along each tyre, then the grip across it
// within what the friction circle leaves, the grip growing with the slip angle (and all of it when slow,
// so it parks), a locked wheel's friction against however it slides. Its body's far points meet walls and
// the ground. Asked of the world once a step: each wheel's ground as a plane, each far point's wall and
// ground; the substeps meet those planes.

#include "fv_play.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <glm/gtc/quaternion.hpp>

#include "fv_play_detail.hpp"

namespace fv::play {

using namespace detail;

namespace {

constexpr double kGravity = 9.81;

struct Axles
{
  double front = 1.35, rear = -1.35;  // the axles along the car (z), about its centre of mass
  double base  = 2.7;                 // the wheelbase
  double mid   = 0.0;                 // the wheels' middle across (x)
};

Axles axlesOf(const CarShape& shape)
{
  const auto& w = shape.wheels;
  Axles       a;
  a.front = 0.5 * (w[0].hub.z + w[1].hub.z);
  a.rear  = 0.5 * (w[2].hub.z + w[3].hub.z);
  a.base  = std::max(a.front - a.rear, 0.5);
  a.mid   = 0.25 * (w[0].hub.x + w[1].hub.x + w[2].hub.x + w[3].hub.x);
  return a;
}

// A wheel's spring and damper, and what it pushes with at rest: its share of the weight, from where the
// centre of mass is between the axles. At rest its length is the top of its travel's, where the file has
// the wheel.
struct Spring
{
  double load = 0.0;  // N at rest
  double k    = 0.0;  // N/m
  double c    = 0.0;  // N·s/m
  double free = 0.0;  // the length it pushes nothing at
};

Spring springOf(const CarShape& shape, int wheel, const CarTuning& t)
{
  const Axles  a     = axlesOf(shape);
  const double front = std::clamp(-a.rear / a.base, 0.1, 0.9);  // the front's share of the weight
  Spring       s;
  s.load            = t.mass * kGravity * 0.5 * (wheel < 2 ? front : 1.0 - front);
  const double mass = s.load / kGravity;
  const double w    = 2.0 * std::numbers::pi * t.frequency;
  s.k               = mass * w * w;
  s.c               = 2.0 * t.damping * std::sqrt(s.k * mass);
  s.free            = t.travelUp + s.load / s.k;
  return s;
}

// The body as a solid box, about its own axes
glm::dvec3 inertiaOf(const CarShape& shape, const CarTuning& t)
{
  const glm::dvec3 d = 2.0 * shape.half;
  return t.mass / 12.0 * glm::dvec3(d.y * d.y + d.z * d.z, d.x * d.x + d.z * d.z, d.x * d.x + d.y * d.y);
}

bool finite(const glm::dvec3& v)
{
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Where a wheel meets a plane, along its suspension from the top of its travel: the suspension's length
// then, and how squarely it meets the plane (none where it meets it too slantwise to stand on)
struct Reach
{
  double length = 0.0, square = 1.0;
};
std::optional<Reach> reachOf(const glm::dvec3& top, const glm::dvec3& down, const Surface& ground, const glm::dvec3& n, double radius)
{
  const double square = -glm::dot(down, n);
  if(square < 0.3)
    return std::nullopt;
  return Reach{(glm::dot(top - ground.at, n) - radius) / square, square};
}

glm::dvec3 topOf(const Car& c, const glm::dmat3& turn, const CarWheel& w, const CarTuning& t)
{
  return c.com + turn * (w.hub + glm::dvec3(0.0, t.travelUp, 0.0));
}

// The ground a wheel meets, asked straight under the top of its travel: none within its reach, none too
// steep to stand on, and none further over the top of its travel than a wheel climbs (a wall, which the
// body meets instead)
std::optional<Surface> groundOf(const Car& c, const glm::dmat3& turn, const CarWheel& w, const World& world, const glm::dvec3& up, const CarTuning& t)
{
  const glm::dvec3 top  = topOf(c, turn, w, t);
  const double     over = t.stepUp + w.radius + 0.1;
  const auto       g    = world.ground(top + up * over, up, over + t.travelUp + t.travelDown + w.radius + 0.5);
  if(!g)
    return std::nullopt;
  const glm::dvec3 n = glm::normalize(g->normal);
  if(glm::dot(n, up) < 0.35)
    return std::nullopt;
  const auto r = reachOf(top, -turn[1], *g, n, w.radius);
  if(!r || r->length < -t.stepUp)
    return std::nullopt;
  return Surface{g->at, n};
}

// Whether each wheel still meets its ground as it did when the car went to sleep
bool restsAsItDid(const Car& c, const CarShape& shape, const World& world, const glm::dvec3& up, const CarTuning& t)
{
  const glm::dmat3 turn   = glm::mat3_cast(c.orientation);
  const double     travel = t.travelUp + t.travelDown;
  for(int i = 0; i < 4; i++)
  {
    const CarWheel&      w = shape.wheels[size_t(i)];
    const auto           g = groundOf(c, turn, w, world, up, t);
    std::optional<Reach> r;
    if(g)
      r = reachOf(topOf(c, turn, w, t), -turn[1], *g, g->normal, w.radius);
    const bool touching = r && r->length <= travel;
    if(touching != c.wheels[size_t(i)].touching || (touching && std::abs(r->length - c.wheels[size_t(i)].length) > 0.02))
      return false;
  }
  return true;
}

}  // namespace

CarShape carShape(const std::array<CarWheel, 4>& wheels, const glm::dvec3& lo, const glm::dvec3& hi)
{
  CarShape s;
  s.wheels     = wheels;
  double tyres = 1e30;
  for(const CarWheel& w : wheels)
    tyres = std::min(tyres, w.hub.y - w.radius);
  const double     bottom = std::max(lo.y, tyres + 0.08);
  const double     top    = std::max(hi.y, bottom + 0.1);
  const glm::dvec3 mid(0.5 * (lo.x + hi.x), 0.5 * (bottom + top), 0.5 * (lo.z + hi.z));
  for(int c = 0; c < 8; c++)
    s.hull.push_back({(c & 1) ? hi.x : lo.x, (c & 2) ? top : bottom, (c & 4) ? hi.z : lo.z});
  s.hull.push_back({mid.x, mid.y, hi.z});  // the bumpers
  s.hull.push_back({mid.x, mid.y, lo.z});
  s.hull.push_back({hi.x, mid.y, mid.z});  // the sides
  s.hull.push_back({lo.x, mid.y, mid.z});
  s.hull.push_back({mid.x, top, mid.z});   // the roof
  for(size_t i = 0; i < 4; i++)            // the tyres' fronts and backs
    s.hull.push_back(wheels[i].hub + glm::dvec3(0.0, 0.0, i < 2 ? wheels[i].radius : -wheels[i].radius));
  s.half   = 0.5 * glm::dvec3(hi.x - lo.x, top - std::min(lo.y, tyres), hi.z - lo.z);
  s.radius = 0.0;
  for(const glm::dvec3& p : s.hull)
    s.radius = std::max(s.radius, glm::length(p));
  return s;
}

glm::dvec3 carForward(const Car& c)
{
  return glm::normalize(c.orientation * glm::dvec3(0.0, 0.0, 1.0));
}

glm::dvec3 carUp(const Car& c)
{
  return glm::normalize(c.orientation * glm::dvec3(0.0, 1.0, 0.0));
}

glm::dvec3 carLeft(const Car& c)
{
  return glm::normalize(c.orientation * glm::dvec3(1.0, 0.0, 0.0));
}

double carSpeed(const Car& c)
{
  return glm::dot(c.velocity, carForward(c));
}

double carRideHeight(const CarShape& shape)
{
  double h = 0.0;
  for(const CarWheel& w : shape.wheels)
    h = std::max(h, w.radius - w.hub.y);
  return h;
}

Car carAt(const glm::dvec3& com, const glm::dquat& q, const CarShape& /*shape*/, const CarTuning& t)
{
  Car c;
  c.com         = com;
  c.orientation = glm::normalize(q);
  c.lastGround  = com;
  for(CarWheelState& w : c.wheels)
    w.length = t.travelUp;
  return c;
}

Body chaseBody(const Car& c, const glm::dvec3& up)
{
  const glm::dvec3 forward = carForward(c);
  glm::dvec3       left    = glm::cross(up, forward);
  left                     = glm::length(left) > 1e-6 ? glm::normalize(left) : carLeft(c);
  const glm::dvec3 over    = glm::cross(forward, left);  // square to the nose, unrolled
  const glm::dquat q       = glm::angleAxis(c.gaze.x, up) * glm::quat_cast(glm::dmat3(left, over, forward));
  return {c.com, glm::normalize(q), c.velocity};
}

void stepCar(Car& c, const CarShape& shape, const World& world, double dt, bool driven, const CarTuning& t)
{
  if(dt <= 0.0)
    return;
  const Car        before = c;
  const uint32_t   keys   = driven ? c.keys : 0u;
  const glm::dvec3 up     = world.up(c.com);
  const Axles      axles  = axlesOf(shape);
  const double     travel = t.travelUp + t.travelDown;

  // the driver's look about the car, back ahead a while after the mouse stops while it moves
  {
    const glm::dvec2 take = c.look;
    c.look                = glm::dvec2(0.0);
    c.gaze.x              = std::clamp(c.gaze.x - take.x * t.turnPerPixel, -t.lookYaw, t.lookYaw);
    c.gaze.y              = std::clamp(c.gaze.y - take.y * t.turnPerPixel, -t.lookPitch, t.lookPitch);
    c.lookIdle            = take == glm::dvec2(0.0) ? c.lookIdle + dt : 0.0;
    if(c.lookIdle > t.lookBack && std::abs(carSpeed(c)) > 3.0)
      c.gaze *= 1.0 - share(dt, t.lookReturn);
  }

  // The pedals, an automatic's: W drives forward, S brakes and once stopped backs; the pedal against the
  // way it goes brakes. With none it holds when slow. Nobody at the wheel: every brake on.
  const double along = carSpeed(c);
  const bool   fwd = (keys & kForward) != 0, back = (keys & kBack) != 0;
  double       brakeShare = 0.0;
  c.throttle              = 0.0;
  if(driven)
  {
    if(back && !fwd && along < 0.5)
      c.gear = -1;
    else if(fwd && !back && along > -0.5)
      c.gear = 1;
    const bool ahead = c.gear > 0 ? fwd : back, against = c.gear > 0 ? back : fwd;
    if(against)
      brakeShare = 1.0;
    else if(ahead)
      c.throttle = double(c.gear);
    else if(std::abs(along) < t.holdBelow)
      brakeShare = 1.0;
    c.handbrake = (keys & kUp) != 0;
  }
  else
  {
    brakeShare  = 1.0;
    c.handbrake = true;
  }

  // The steering: toward the lock asked, cut as it speeds to what the tyres turn it with (turnLimit g),
  // back to the middle quicker than it goes out. The front wheels about the turn's middle on the rear
  // axle's line (Ackermann): the inner one turns more.
  const double speed = glm::length(c.velocity);
  const double lock  = std::clamp(std::atan(t.turnLimit * kGravity * axles.base / std::max(speed * speed, 1e-9)), t.minLock, t.lock);
  const double wish  = driven ? (key(keys, kLeft) - key(keys, kRight)) * lock : 0.0;
  const bool   out   = std::abs(wish) > std::abs(c.steer) && wish * c.steer >= 0.0;
  c.steer            = std::clamp(toward(c.steer, wish, out ? t.steerRate : t.returnRate, dt), -lock, lock);
  {
    const double tanD = std::tan(c.steer);
    for(size_t i = 0; i < 4; i++)
    {
      const double x    = shape.wheels[i].hub.x - axles.mid;  // +x: the car's left
      c.wheels[i].steer = i < 2 ? std::atan2(axles.base * tanD, axles.base - x * tanD) : 0.0;
    }
  }

  // righting itself: turned upright as it heads, lifted a little over the ground, then let drop
  const double ride = carRideHeight(shape);
  if(c.righting > 0.0)
  {
    const glm::dvec3 nose    = levelUnit(glm::mat3_cast(c.orientation)[2], up);
    const glm::dquat upright = glm::normalize(glm::quat_cast(glm::dmat3(glm::cross(up, nose), up, nose)));
    const double     settle  = share(dt, t.rightTime / 4.0);
    c.orientation            = glm::normalize(glm::slerp(c.orientation, upright, settle));
    if(const auto g = world.ground(c.com + up * 3.0, up, 20.0))
      c.com += up * ((glm::dot(g->at - c.com, up) + ride + 0.3) * settle);
    c.velocity = c.spin = glm::dvec3(0.0);
    c.righting          = std::max(0.0, c.righting - dt);
    c.tipped            = 0.0;
    return;
  }

  // asleep: nothing moves until a pedal, or its ground going from under it, wakes it
  if(c.asleep)
  {
    bool wake = driven && (fwd || back);
    if(!wake && (c.nap += dt) >= 1.0)
    {
      c.nap = 0.0;
      wake  = !restsAsItDid(c, shape, world, up, t);
    }
    if(!wake)
    {
      for(CarWheelState& w : c.wheels)
        w.rate = 0.0;
      return;
    }
    c.asleep = false;
    c.still  = 0.0;
  }

  // What it meets, asked once a step: each wheel's ground as a plane (its normal eased, so a mesh's
  // facets don't shake it), each far point's wall (a ray from the centre out to it, so nothing is passed
  // through) and the ground under the low ones (all of them, on its side or off its wheels)
  struct Plane
  {
    glm::dvec3 at{0.0}, n{0.0, 1.0, 0.0};
  };
  const glm::dmat3 turn     = glm::mat3_cast(c.orientation);
  int              grounded = 0;
  for(size_t i = 0; i < 4; i++)
  {
    CarWheelState& W = c.wheels[i];
    const auto     g = groundOf(c, turn, shape.wheels[i], world, up, t);
    if(g)
    {
      W.normal = W.ground ? glm::normalize(glm::mix(W.normal, g->normal, share(dt, 0.04))) : g->normal;
      grounded++;
    }
    W.ground = g;
  }
  const double sweep  = glm::length(c.velocity) * dt;
  const bool   tipped = glm::dot(turn[1], up) < 0.7;
  std::vector<std::optional<Plane>> walls(shape.hull.size()), floors(shape.hull.size());
  for(size_t j = 0; j < shape.hull.size(); j++)
  {
    const glm::dvec3 r      = turn * shape.hull[j];
    const double     length = glm::length(r);
    if(world.ray && length > 1e-6)
      if(const auto hit = world.ray(c.com, r / length, length + 0.05 + sweep))
        walls[j] = Plane{hit->at, glm::normalize(hit->normal)};
    if(tipped || grounded < 4 || glm::dot(r, up) < 0.0)
      if(const auto g = world.ground(c.com + r + up * 1.0, up, 1.5 + sweep))
        floors[j] = Plane{g->at, glm::normalize(g->normal)};
  }

  std::array<Spring, 4> springs;
  for(int i = 0; i < 4; i++)
    springs[size_t(i)] = springOf(shape, i, t);
  const glm::dvec3   inertia = inertiaOf(shape, t);
  const double       driveShare[4] = {0.5 * t.driveFront, 0.5 * t.driveFront, 0.5 * (1.0 - t.driveFront), 0.5 * (1.0 - t.driveFront)};
  const double       brakeOf[4]    = {0.5 * t.frontBrake, 0.5 * t.frontBrake, 0.5 * (1.0 - t.frontBrake), 0.5 * (1.0 - t.frontBrake)};
  const int          steps         = std::max(1, int(std::ceil(dt / t.substep)));
  const double       h             = dt / steps;
  for(int step = 0; step < steps; step++)
  {
    const glm::dmat3 R    = glm::mat3_cast(c.orientation);
    const glm::dmat3 Iinv = R * glm::dmat3(1.0 / inertia.x, 0.0, 0.0, 0.0, 1.0 / inertia.y, 0.0, 0.0, 0.0, 1.0 / inertia.z) * glm::transpose(R);
    const glm::dvec3 down = -R[1];
    const auto       pointVelocity = [&](const glm::dvec3& r) { return c.velocity + glm::cross(c.spin, r); };
    // the mass an impulse along `u` at `r` from the centre moves
    const auto massAlong = [&](const glm::dvec3& r, const glm::dvec3& u) {
      return 1.0 / (1.0 / t.mass + glm::dot(u, glm::cross(Iinv * glm::cross(r, u), r)));
    };
    const auto impulse = [&](const glm::dvec3& r, const glm::dvec3& j) {
      c.velocity += j / t.mass;
      c.spin += Iinv * glm::cross(r, j);
    };

    // The forces: gravity, the air, each wheel's spring (along the ground's normal where its tyre meets
    // it, which is the load its tyre holds with) and its drive (no more than its tyre holds)
    glm::dvec3 force  = -up * (t.mass * kGravity) - c.velocity * (0.5 * t.air * t.dragArea * glm::length(c.velocity));
    glm::dvec3 torque = glm::dvec3(0.0);
    struct Patch
    {
      bool       on = false;
      glm::dvec3 r{0.0}, n{0.0}, f{0.0}, s{0.0};  // from the centre; the ground's normal; the tyre's forward and left on it
      double     load = 0.0, drive = 0.0;          // N
      double     bump = 0.0;                       // metres past the top of its travel, along n
    };
    std::array<Patch, 4> patch;
    const double         ahead = glm::dot(c.velocity, R[2]);
    const double         pull  = c.throttle == 0.0 ? 0.0 : std::min(t.tractive, t.power * ((keys & kFast) ? t.boost : 1.0) / std::max(std::abs(ahead), 1.0));
    const double         cap   = c.throttle > 0.0 ? t.topSpeed : t.reverseSpeed;
    const double         fade  = std::clamp((cap - c.throttle * ahead) / (0.05 * cap), 0.0, 1.0);
    for(size_t i = 0; i < 4; i++)
    {
      CarWheelState&   W   = c.wheels[i];
      const CarWheel&  w   = shape.wheels[i];
      const glm::dvec3 top = c.com + R * (w.hub + glm::dvec3(0.0, t.travelUp, 0.0));
      W.touching           = false;
      W.load               = 0.0;
      std::optional<Reach> reach;
      if(W.ground)
        reach = reachOf(top, down, *W.ground, W.normal, w.radius);
      if(!reach || reach->length > travel)
      {
        W.length = toward(W.length, travel, 3.0, h);  // it hangs
        continue;
      }
      // the spring and the damper, which pulls harder back out than in; past the top of its travel the
      // bump stop takes what is left, as a stop does (below), the spring no harder than at the top
      const Spring& s    = springs[i];
      const double  L    = reach->length;
      const double  rate = glm::dot(pointVelocity(top - c.com), W.normal) / reach->square;  // the length's change, m/s
      double        push = s.k * (s.free - std::max(L, 0.0)) - s.c * rate * (rate > 0.0 ? t.rebound : 1.0);
      push               = std::clamp(push, 0.0, t.maxLoad * s.load);
      W.length           = L;
      W.touching         = true;
      W.load             = push;
      Patch& p           = patch[i];
      p.on               = true;
      p.r                = top + down * L - W.normal * w.radius - c.com;
      p.n                = W.normal;
      p.load             = push;
      p.bump             = std::max(0.0, -L * reach->square);
      const glm::dvec3 heading = R * glm::dvec3(std::sin(W.steer), 0.0, std::cos(W.steer));
      p.f                      = glm::normalize(heading - p.n * glm::dot(heading, p.n));
      p.s                      = glm::cross(p.n, p.f);
      p.drive                  = c.handbrake && i >= 2 ? 0.0 : std::clamp(c.throttle * pull * fade * driveShare[i], -t.grip * push, t.grip * push);
      force += p.n * push + p.f * p.drive;
      torque += glm::cross(p.r, p.n * push + p.f * p.drive);
    }
    c.velocity += force * (h / t.mass);
    c.spin += Iinv * torque * h;
    if(const double w = glm::length(c.spin); w > 25.0)
      c.spin *= 25.0 / w;

    // The tyres' friction as impulses, twice round. Along each tyre the brakes (rolling resistance and
    // the engine's braking too), to what the circle leaves past the drive; across it the grip its slip
    // angle gives, past the peak falling to a slide's, all of it when slow (so it stands), within what
    // the circle leaves. A wheel the handbrake locks has its friction against however it slides.
    std::array<double, 4> along{}, aside{}, slid{}, startAlong{}, startAside{};
    for(size_t i = 0; i < 4; i++)
      if(patch[i].on)
      {
        const glm::dvec3 v = pointVelocity(patch[i].r);
        startAlong[i]      = glm::dot(v, patch[i].f);
        startAside[i]      = glm::dot(v, patch[i].s);
      }
    for(int pass = 0; pass < 2; pass++)
      for(size_t i = 0; i < 4; i++)
      {
        const Patch& p = patch[i];
        if(!p.on)
          continue;
        const double most = t.grip * p.load * h;  // the friction circle, as an impulse
        if(c.handbrake && i >= 2)
        {
          const glm::dvec3 v      = pointVelocity(p.r);
          const glm::dvec3 planar = v - p.n * glm::dot(v, p.n);
          const double     sliding = glm::length(planar);
          const double     startSpeed = std::hypot(startAlong[i], startAside[i]);
          const double     limit  = most * glm::mix(1.0, t.handbrakeGrip, std::clamp(startSpeed / t.lowSpeed, 0.0, 1.0));
          if(sliding > 1e-9 && slid[i] < limit)
          {
            const glm::dvec3 dir = planar / sliding;
            const double     j   = std::min(sliding * massAlong(p.r, dir), limit - slid[i]);
            impulse(p.r, -dir * j);
            slid[i] += j;
          }
          continue;
        }
        const double brake = brakeShare * t.brake * brakeOf[i] + t.rolling * p.load + (c.throttle == 0.0 ? t.engineBrake * driveShare[i] : 0.0);
        double       sum   = std::clamp(along[i] - glm::dot(pointVelocity(p.r), p.f) * massAlong(p.r, p.f), -brake * h, brake * h);
        sum                = std::clamp(sum, -most - p.drive * h, most - p.drive * h);
        impulse(p.r, p.f * (sum - along[i]));
        along[i] = sum;

        const double slip = std::atan(std::abs(startAside[i]) / std::max(std::abs(startAlong[i]), t.lowSpeed));
        double       grip = t.cornering * p.load * slip * h;
        if(grip > most)
          grip = most * glm::mix(1.0, t.slideGrip, std::clamp(grip / most - 1.0, 0.0, 1.0));
        grip              = std::max(grip, most * std::clamp(1.0 - std::abs(startAlong[i]) / t.lowSpeed, 0.0, 1.0));
        const double used = std::abs(p.drive * h + along[i]);
        grip              = std::min(grip, std::sqrt(std::max(0.0, most * most - used * used)));
        const double side = std::clamp(aside[i] - glm::dot(pointVelocity(p.r), p.s) * massAlong(p.r, p.s), -grip, grip);
        impulse(p.r, p.s * (side - aside[i]));
        aside[i] = side;
      }

    // Its body against what it meets: what moves into a wall or the ground stops (a little bounce when
    // fast), slides with some friction, and is pushed back out of it a little at a time. A wheel past the
    // top of its travel is on its bump stop: it stops as dead, its tyre's friction its own (above).
    double     deepest = 0.0;
    glm::dvec3 outOf(0.0);
    for(const Patch& p : patch)
      if(p.on && p.bump > 0.0)
      {
        if(const double into = glm::dot(pointVelocity(p.r), p.n); into < 0.0)
          impulse(p.r, p.n * (-into * massAlong(p.r, p.n)));
        if(p.bump > deepest)
        {
          deepest = p.bump;
          outOf   = p.n;
        }
      }
    for(size_t j = 0; j < shape.hull.size(); j++)
    {
      const glm::dvec3 r = R * shape.hull[j];
      const glm::dvec3 p = c.com + r;
      for(const std::optional<Plane>* plane : {&walls[j], &floors[j]})
      {
        if(!*plane)
          continue;
        const glm::dvec3 n     = (*plane)->n;
        const double     depth = glm::dot((*plane)->at - p, n);
        if(depth <= 0.0)
          continue;
        if(const double into = glm::dot(pointVelocity(r), n); into < 0.0)
        {
          const double stop = -(1.0 + (into < -0.5 ? t.hullBounce : 0.0)) * into * massAlong(r, n);
          impulse(r, n * stop);
          const glm::dvec3 v     = pointVelocity(r);
          const glm::dvec3 slide = v - n * glm::dot(v, n);
          if(const double length = glm::length(slide); length > 1e-9)
          {
            const glm::dvec3 dir = slide / length;
            impulse(r, -dir * std::min(length * massAlong(r, dir), t.hullGrip * stop));
          }
        }
        if(depth > deepest)
        {
          deepest = depth;
          outOf   = n;
        }
      }
    }
    if(deepest > 0.01)
      c.com += outOf * (0.5 * (deepest - 0.01));

    // on: where it goes and how it turns, and the wheels rolling with the ground under them (a locked
    // one not; one off the ground slowing, or spinning up under the throttle)
    c.com += c.velocity * h;
    const glm::dquat w(0.0, c.spin.x, c.spin.y, c.spin.z);
    c.orientation = glm::normalize(c.orientation + (w * c.orientation) * (0.5 * h));
    for(size_t i = 0; i < 4; i++)
    {
      CarWheelState& W = c.wheels[i];
      if(patch[i].on)
        W.rate = c.handbrake && i >= 2 ? 0.0 : glm::dot(pointVelocity(patch[i].r), patch[i].f) / shape.wheels[i].radius;
      else
        W.rate = W.rate * (1.0 - share(h, 1.5)) + c.throttle * driveShare[i] * 60.0 * h;
      W.spin = std::remainder(W.spin + W.rate * h, 2.0 * std::numbers::pi);
    }
  }

  int touching = 0;
  for(const CarWheelState& w : c.wheels)
    touching += w.touching ? 1 : 0;
  if(touching >= 2)
    c.lastGround = c.com;
  // at rest it sleeps
  if(c.throttle == 0.0 && glm::length(c.velocity) < t.sleepSpeed && glm::length(c.spin) < t.sleepSpin && touching >= 3)
    c.still += dt;
  else
    c.still = 0.0;
  if(c.still >= t.sleepAfter)
  {
    c.asleep   = true;
    c.nap      = 0.0;
    c.velocity = c.spin = glm::dvec3(0.0);
    for(CarWheelState& w : c.wheels)
      w.rate = 0.0;
  }
  // on its side or its roof, all but still: in a while it rights itself
  if(glm::dot(carUp(c), up) < 0.3 && glm::length(c.velocity) < 2.0)
    c.tipped += dt;
  else
    c.tipped = 0.0;
  if(c.tipped >= t.rightAfter)
  {
    c.righting = t.rightTime;
    c.tipped   = 0.0;
  }
  // fallen out of the world: back where it last stood
  if(glm::dot(c.lastGround - c.com, up) > 200.0)
  {
    c.com      = c.lastGround + up * 1.0;
    c.velocity = c.spin = glm::dvec3(0.0);
  }
  if(!finite(c.com) || !finite(c.velocity) || !finite(c.spin) || !std::isfinite(c.orientation.w) || !std::isfinite(c.orientation.x)
     || !std::isfinite(c.orientation.y) || !std::isfinite(c.orientation.z))
  {
    c          = before;
    c.velocity = c.spin = glm::dvec3(0.0);
  }
}

}  // namespace fv::play
