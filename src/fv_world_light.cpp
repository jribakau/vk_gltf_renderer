/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fv_world_light.hpp"

#include <algorithm>
#include <cmath>

namespace fv::world_light {

// The shader's own functions for a body's face, as C++ (tests/test_world_bodies.cpp does the same)
namespace shader {
using namespace glm;
using float3 = glm::vec3;
using std::lerp;
#include "../shaders/world_bodies.h.slang"
}  // namespace shader

namespace {

constexpr double kPi = 3.14159265358979323846;

// The sphere `source` is (-1: the planet)
struct Sphere
{
  glm::dvec3 centre;
  double     radius;
};

Sphere sphereOf(const Sky& sky, int source)
{
  if(source < 0)
    return {sky.planetCentre, sky.planetRadius};
  const Body& b = sky.bodies[size_t(source)];
  return {b.centre, b.radius};
}

int bodyCount(const Sky& sky)
{
  return std::min(int(sky.bodies.size()), kMost);
}

// What the planet and the bodies but `but` leave of the sun's disk at `point`
float sunPast(const Sky& sky, const glm::dvec3& point, int but)
{
  const glm::vec3 p(point);
  const glm::vec3 toSun(sky.toSun);
  const float     sunRadius = float(sky.sunAngularRadius);
  float           share     = 1.0F;
  if(but >= 0 && sky.planetRadius > 0.0)
    share *= shader::worldSunPast(p, glm::vec3(sky.planetCentre), float(sky.planetRadius), toSun, sunRadius);
  for(int k = 0; k < bodyCount(sky); k++)
    if(k != but)
      share *= shader::worldSunPast(p, glm::vec3(sky.bodies[size_t(k)].centre), float(sky.bodies[size_t(k)].radius), toSun, sunRadius);
  return share;
}

// Where a ray from `origin` along unit `dir` enters the sphere, -1 for a miss (double: the bodies are
// thousands of kilometres off and their edges are what the sums are made of)
double hitSphere(const glm::dvec3& origin, const glm::dvec3& dir, const Sphere& s)
{
  const glm::dvec3 from  = origin - s.centre;
  const double     along = glm::dot(from, dir);
  const glm::dvec3 off   = from - dir * along;
  const double     disc  = s.radius * s.radius - glm::dot(off, off);
  if(disc < 0.0)
    return -1.0;
  const double t = -along - std::sqrt(disc);
  return t > 0.0 ? t : -1.0;
}

}  // namespace

glm::vec3 faceRadiance(const Sky& sky, const Lights* fill, int source, const glm::dvec3& point, const glm::dvec3& toEye)
{
  const Sphere    s = sphereOf(sky, source);
  const glm::vec3 n(glm::normalize(point - s.centre));
  const glm::vec3 toSun(sky.toSun);
  const float     lit = sunPast(sky, point, source);
  if(source < 0)
    return sky.planetAlbedo * (std::max(glm::dot(n, toSun), 0.0F) * float(1.0 / kPi) * lit);

  const Body&     b         = sky.bodies[size_t(source)];
  const float     markings  = shader::worldBodyMarkings(n, b.kind, float(source) + 1.0F);
  const glm::vec3 radiance  = b.color * (markings * shader::worldBodyShade(n, glm::vec3(toEye), toSun) * lit);
  glm::vec3       fromOther = glm::vec3(0.0F);
  if(fill != nullptr)
    for(int k = 0; k < bodyCount(sky); k++)
    {
      const glm::vec3 light = fill->fill[size_t(source * kMost + k)];
      if(light == glm::vec3(0.0F))
        continue;
      const Sphere     other = sphereOf(sky, k == source ? -1 : k);
      const glm::dvec3 to    = other.centre - point;
      const double     d     = glm::length(to);
      fromOther += light * shader::worldBodyFillShade(n, glm::vec3(to / d), float(std::min(other.radius / d, 1.0)));
    }
  return radiance + b.color * markings * fromOther;
}

Face faceLight(const Sky& sky, const Lights* fill, int source, const glm::dvec3& from, int steps)
{
  Face             face;
  const Sphere     s  = sphereOf(sky, source);
  const glm::dvec3 to = s.centre - from;
  const double     d  = glm::length(to);
  if(!(s.radius > 0.0) || d <= s.radius)
    return face;
  const glm::dvec3 w = to / d;
  face.toward        = w;
  // the cone the sphere fills, by its sine: 1 - cos of a far body's half angle is all in the last digits
  const double sin2 = (s.radius * s.radius) / (d * d);
  const double omc  = sin2 / (1.0 + std::sqrt(1.0 - sin2));
  const glm::dvec3 u = glm::normalize(std::abs(w.x) < 0.9 ? glm::cross(w, glm::dvec3(1.0, 0.0, 0.0)) : glm::cross(w, glm::dvec3(0.0, 0.0, 1.0)));
  const glm::dvec3 v = glm::cross(w, u);

  glm::dvec3 sum(0.0);
  glm::dvec3 mean(0.0);
  for(int a = 0; a < steps; a++)
  {
    // even in solid angle: 1 - cos theta even over the cap
    const double down = omc * (double(a) + 0.5) / double(steps);
    const double sinT = std::sqrt(std::max(down * (2.0 - down), 0.0));
    const double cosT = 1.0 - down;
    for(int c = 0; c < steps; c++)
    {
      const double     phi = 2.0 * kPi * (double(c) + 0.5) / double(steps);
      const glm::dvec3 dir = u * (sinT * std::cos(phi)) + v * (sinT * std::sin(phi)) + w * cosT;
      const double     t   = hitSphere(from, dir, s);
      if(t < 0.0)
        continue;
      // another body before it along the way
      bool hidden = false;
      for(int k = 0; k < bodyCount(sky) && !hidden; k++)
        if(k != source)
        {
          const double tk = hitSphere(from, dir, sphereOf(sky, k));
          hidden          = tk > 0.0 && tk < t;
        }
      if(hidden)
        continue;
      const glm::vec3 r = faceRadiance(sky, fill, source, from + dir * t, -dir);
      sum += glm::dvec3(r);
      mean += dir * double(luminance(r));
    }
  }
  const double each = 2.0 * kPi * omc / double(steps * steps);
  face.light        = glm::vec3(sum * each);
  const double lum  = double(luminance(face.light));
  if(lum > 0.0)
  {
    const glm::dvec3 m = mean * each / lum;
    const double     l = glm::length(m);
    if(l > 0.0)
    {
      face.toward = m / l;
      face.spread = std::min(l, 1.0);
    }
  }
  return face;
}

Lights sumLights(const Sky& sky, const glm::dvec3& place)
{
  Lights    lights;
  const int count = bodyCount(sky);
  // what each body and the planet give each body's centre: their sunlit faces
  constexpr int kFillSteps = 16;
  for(int i = 0; i < count; i++)
    for(int k = 0; k < count; k++)
    {
      const int source = k == i ? -1 : k;
      if(source < 0 && !(sky.planetRadius > 0.0))
        continue;
      lights.fill[size_t(i * kMost + k)] = faceLight(sky, nullptr, source, sky.bodies[size_t(i)].centre, kFillSteps).light;
    }
  // and their light at the place, with what lights their dark sides
  constexpr int kSteps = 32;
  float         most   = 0.0F;
  for(int i = 0; i < count; i++)
  {
    lights.shine[size_t(i)] = faceLight(sky, &lights, i, place, kSteps);
    most                    = std::max(most, luminance(lights.shine[size_t(i)].light));
  }
  // the sun's on a level surface there
  glm::dvec3 up(0.0, 1.0, 0.0);
  if(sky.planetRadius > 0.0 && glm::length(place - sky.planetCentre) > 0.0)
    up = glm::normalize(place - sky.planetCentre);
  const float sun   = float(std::max(glm::dot(up, sky.toSun), 0.0));
  const float least = kLeast * std::max(sun, most);
  for(int i = 0; i < count; i++)
  {
    const float l          = luminance(lights.shine[size_t(i)].light);
    lights.lit[size_t(i)] = l > 0.0F && l >= least;
  }
  return lights;
}

}  // namespace fv::world_light
