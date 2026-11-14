/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Other bodies round a planet that stands in the scene (the backdrop's `bodies`, src/fv_backdrop.hpp):
// moons and planets as spheres where they are, lit by the sun. Not meshes: a ray is tested against
// them where it would otherwise see the sky (evalWorldSky in gltf_pathtrace.slang), and the sun's
// light at a point is what they leave of its disk. So a body has its phase from where the sun and the
// eye stand, goes dark in the planet's shadow, and darkens the ground it passes in front of the sun
// for. Slang and plain C++ (glm, no swizzles): tests/test_world_bodies.cpp tests what is here.
#ifndef WORLD_BODIES_H_SLANG
#define WORLD_BODIES_H_SLANG

// What a body is made of (SceneFrameInfo::worldBodyLook.w): rock, cratered and mottled, or gas, banded
static const int WORLD_BODY_ROCK = 0;
static const int WORLD_BODY_GAS  = 1;

// Where a ray from `origin` along `dir` (unit) meets the sphere at `centre`: the distance to its near
// side, or -1 when it misses or starts inside. The sphere is far off and large, so its distance from
// the ray is taken first: b * b - c loses the sphere's radius among the squares of its distance.
inline float worldBodyHit(float3 origin, float3 dir, float3 centre, float radius)
{
  const float3 from  = origin - centre;
  const float  along = dot(from, dir);
  const float3 off   = from - dir * along;
  const float  disc  = radius * radius - dot(off, off);
  if(disc < 0.0F)
    return -1.0F;
  const float t = -along - sqrt(disc);
  return t > 0.0F ? t : -1.0F;
}

// The share of a disk of angular radius `a` hidden by one of angular radius `b` whose centre is `s`
// from its own (radians, small: taken as flat disks): none apart, all of it under a larger one, the
// ratio of their areas round a smaller one, the lens between.
inline float worldDiskCovered(float a, float b, float s)
{
  if(s >= a + b)
    return 0.0F;
  if(s <= abs(b - a))
    return b >= a ? 1.0F : (b * b) / (a * a);
  const float a2   = a * a;
  const float b2   = b * b;
  const float x    = (a2 - b2 + s * s) / (2.0F * s);  // from a's centre to the chord they share
  const float y    = s - x;
  const float lens = a2 * acos(clamp(x / a, -1.0F, 1.0F)) - x * sqrt(max(a2 - x * x, 0.0F))
                     + b2 * acos(clamp(y / b, -1.0F, 1.0F)) - y * sqrt(max(b2 - y * y, 0.0F));
  return clamp(lens / (3.14159265F * a2), 0.0F, 1.0F);
}

// What the sphere at `centre` leaves of the sun's disk as `p` sees it: the sun toward `sunDirection`,
// its disk `sunAngularRadius` across the half. 1 with the sphere behind `p` or clear of the disk, 0
// in its full shadow, the penumbra between.
inline float worldSunPast(float3 p, float3 centre, float radius, float3 sunDirection, float sunAngularRadius)
{
  const float3 to    = centre - p;
  const float  d     = length(to);
  const float  along = dot(to, sunDirection);
  if(d <= radius)
    return 0.0F;
  if(along <= 0.0F)
    return 1.0F;
  // the angle between them from the cross product: the cosine of a small angle has no digits left
  const float s = atan2(length(cross(to, sunDirection)), along);
  const float b = asin(min(radius / d, 1.0F));
  return 1.0F - worldDiskCovered(sunAngularRadius, b, s);
}

// Value noise on a lattice, 0 to 1, for a body's markings
inline float worldBodyHash(float3 p)
{
  const float h = sin(dot(p, float3(127.1F, 311.7F, 74.7F))) * 43758.5453F;
  return h - floor(h);
}

inline float worldBodyNoise(float3 p)
{
  const float3 i = floor(p);
  const float3 f = p - i;
  const float3 u = f * f * (float3(3.0F, 3.0F, 3.0F) - 2.0F * f);
  const float  a = lerp(worldBodyHash(i), worldBodyHash(i + float3(1.0F, 0.0F, 0.0F)), u.x);
  const float  b = lerp(worldBodyHash(i + float3(0.0F, 1.0F, 0.0F)), worldBodyHash(i + float3(1.0F, 1.0F, 0.0F)), u.x);
  const float  c = lerp(worldBodyHash(i + float3(0.0F, 0.0F, 1.0F)), worldBodyHash(i + float3(1.0F, 0.0F, 1.0F)), u.x);
  const float  d = lerp(worldBodyHash(i + float3(0.0F, 1.0F, 1.0F)), worldBodyHash(i + float3(1.0F, 1.0F, 1.0F)), u.x);
  return lerp(lerp(a, b, u.y), lerp(c, d, u.y), u.z);
}

// A body's albedo against its colour at the point of it whose normal is `n`: about 1 on average.
// Rock: dark plains among bright highlands, mottled finer. Gas: bands by latitude, pulled about.
// `seed` tells the bodies apart.
inline float worldBodyMarkings(float3 n, int kind, float seed)
{
  const float3 at = n + float3(seed, seed * 1.7F, seed * 2.3F);
  if(kind == WORLD_BODY_GAS)
  {
    const float pull = worldBodyNoise(at * 3.0F) - 0.5F;
    const float band = sin((n.y + 0.08F * pull) * 14.0F) * 0.5F + 0.5F;
    const float fine = sin((n.y + 0.05F * pull) * 47.0F) * 0.5F + 0.5F;
    return 0.72F + 0.4F * band + 0.12F * fine;
  }
  const float plains = clamp((worldBodyNoise(at * 2.2F) - 0.45F) * 6.0F, 0.0F, 1.0F);
  const float fine   = 0.5F * worldBodyNoise(at * 9.0F) + 0.3F * worldBodyNoise(at * 21.0F) + 0.2F * worldBodyNoise(at * 47.0F);
  return lerp(0.62F, 1.15F, plains) * (0.8F + 0.4F * fine);
}

// How bright the body's surface is toward the eye where its normal is `n`, per unit of the sun's
// illuminance on a surface facing it and of the body's albedo: Lommel-Seeliger with a Lambert share,
// which keeps a full moon's limb as bright as its middle and still rounds a crescent. `toEye` and
// `toSun` are unit vectors from the point.
inline float worldBodyShade(float3 n, float3 toEye, float3 toSun)
{
  const float cosSun = dot(n, toSun);
  const float cosEye = max(dot(n, toEye), 0.0F);
  if(cosSun <= 0.0F)
    return 0.0F;
  const float seeliger = cosSun / max(cosSun + cosEye, 1e-4F);
  return (0.6F * 2.0F * seeliger + 0.4F * cosSun) * (1.0F / 3.14159265F);
}

#endif  // WORLD_BODIES_H_SLANG
