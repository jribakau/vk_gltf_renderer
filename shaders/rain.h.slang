/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Rain in the air: how many drops a rain of a given rate holds, how big they are, how fast they
// fall, and where each one is (src/rain.hpp draws them; docs/sky.md, Rain). Slang and C++
// (tests/test_rain.cpp).
//
// The drops live in cells a metre across, in coordinates sheared along the way the wind blows them
// (`shear`), so a drop keeps its cell as it falls: each cell's drops fall through it and come in
// again at its top, at the speed their size gives them. The cells are the world's, so the rain
// stands still in it when the camera turns.
//

#ifndef RAIN_H_SLANG
#define RAIN_H_SLANG 1

// Marshall and Palmer's drop sizes (1948): N(D) = N0 exp(-L D) drops a cubic metre a millimetre of
// diameter, L = 4.1 R^-0.21 for a rate R in mm/h. With Atlas's fall speeds below they give back the
// rate within a fifth from drizzle to a downpour (tests/test_rain.cpp).
static const float RAIN_N0 = 8000.0F;
// The drops drawn: from half a millimetre (smaller ones are a haze, not streaks) to six (larger
// ones break up as they fall)
static const float RAIN_MIN_DIAMETER = 0.5F;
static const float RAIN_MAX_DIAMETER = 6.0F;
// The fall speed the cells are sheared by: a middling drop's (each drop's streak takes its own)
static const float RAIN_SHEAR_SPEED = 5.0F;

// L, per millimetre, for a rate in mm/h
inline float rainSlope(float rate)
{
  return 4.1F * pow(max(rate, 1e-3F), -0.21F);
}

// Drops a cubic metre at least `minDiameter` millimetres across
inline float rainDropsPerCubicMetre(float rate, float minDiameter)
{
  const float slope = rainSlope(rate);
  return RAIN_N0 / slope * exp(-slope * minDiameter);
}

// A drop's diameter, mm, for a uniform number `u` in [0, 1): the distribution above from
// RAIN_MIN_DIAMETER, the largest cut at RAIN_MAX_DIAMETER
inline float rainDropDiameter(float rate, float u)
{
  return min(RAIN_MIN_DIAMETER - log(max(1.0F - u, 1e-7F)) / rainSlope(rate), RAIN_MAX_DIAMETER);
}

// A drop's terminal speed, m/s, by its diameter in mm (Atlas, Srivastava and Sekhon 1973): 4 m/s at
// a millimetre, 8 at three, nearly 9.6 at the largest
inline float rainFallSpeed(float diameter)
{
  return max(9.65F - 10.3F * exp(-0.6F * diameter), 0.3F);
}

// How far a drop drifts sideways for each metre it falls, against the wind `wind` (m/s, x and z):
// the cells' shear
inline float2 rainShear(float2 wind)
{
  return wind * (-1.0F / RAIN_SHEAR_SPEED);
}

// A drop's light seen against collimated light (the sun's) at `cosTheta` between the light's way and
// the way to the eye, per steradian: a water sphere's refraction without its diffraction peak,
// forward-scattering (Henyey-Greenstein, g 0.8). A drop backlit by the sun glows
inline float rainSunPhase(float cosTheta)
{
  const float g     = 0.8F;
  const float denom = 1.0F + g * g - 2.0F * g * cosTheta;
  return (1.0F - g * g) / (12.5663706F * denom * sqrt(denom));
}

inline uint rainHash(uint x)
{
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  x ^= x >> 16U;
  return x;
}

inline float rainUnit(uint x)
{
  return float(rainHash(x) >> 8U) * (1.0F / 16777216.0F);
}

struct RainDrop
{
  float3 position;  // where it is at `time`, metres
  float  diameter;  // mm
  float  speed;     // its fall, m/s
  int    present;   // 0: the cell has fewer drops than its slots
};

// Drop `index` of the cell `cell` (x and z along the sheared ground, y up; `cellSize` metres) at
// `time` seconds: a cell has as many slots as its most drops, each a drop at `chance`
inline RainDrop rainDropAt(int3 cell, uint index, float cellSize, float2 shear, float time, float chance, uint seed, float rate)
{
  const uint key = rainHash((uint(cell.x) * 73856093U) ^ (uint(cell.y) * 19349663U) ^ (uint(cell.z) * 83492791U) ^ (seed * 2654435761U))
                   + index * 0x9E3779B9U;
  RainDrop d;
  d.present         = rainUnit(key) < chance ? 1 : 0;
  d.diameter        = rainDropDiameter(rate, rainUnit(key + 1U));
  d.speed           = rainFallSpeed(d.diameter);
  const float hx    = (float(cell.x) + rainUnit(key + 2U)) * cellSize;
  const float hz    = (float(cell.z) + rainUnit(key + 3U)) * cellSize;
  float       share = rainUnit(key + 4U) - d.speed * time / cellSize;
  share -= floor(share);
  const float y = (float(cell.y) + share) * cellSize;
  d.position    = float3(hx + shear.x * y, y, hz + shear.y * y);
  return d;
}

// The shelter map (rain_shelter.slang): a square of RAIN_SHELTER_SIZE texels (25 cm each), RAIN_SHELTER_EXTENT
// metres across on the sheared ground round the camera, each the height of the first surface a drop
// falling there meets (RAIN_SHELTER_OPEN for none)
#define RAIN_SHELTER_SIZE 256
static const float RAIN_SHELTER_EXTENT = 64.0F;
static const float RAIN_SHELTER_OPEN   = -1e30F;

// The texel of the shelter map under `position`, or -1 off the map
inline int rainShelterTexel(float3 position, float2 shear, float2 centre)
{
  const float2 h  = float2(position.x - shear.x * position.y, position.z - shear.y * position.y);
  const float2 uv = (h - centre) * (1.0F / RAIN_SHELTER_EXTENT) + float2(0.5F, 0.5F);
  if(uv.x < 0.0F || uv.y < 0.0F || uv.x >= 1.0F || uv.y >= 1.0F)
    return -1;
  return int(uv.y * float(RAIN_SHELTER_SIZE)) * RAIN_SHELTER_SIZE + int(uv.x * float(RAIN_SHELTER_SIZE));
}

// ---- where it lands ---------------------------------------------------------------------------------

// How wet a surface open to the rain is at a rate (mm/h), 0 to 1: a drizzle 0.6, light rain 0.9 (the
// steady state: not how long it has rained)
inline float rainWetness(float rate)
{
  return rate > 0.0F ? 1.0F - exp(-rate / 1.0F) : 0.0F;
}

// The share of the puddles flat ground can hold that it holds (groundWaterAt's): none in a drizzle,
// from 1 mm/h up, all from 10
inline float rainPuddleShare(float rate)
{
  return clamp((rate - 1.0F) / 9.0F, 0.0F, 1.0F);
}

// How far out from a surface the shelter map is read, metres: past its texels, so a wall's own top
// doesn't shelter its face
static const float RAIN_SHELTER_OUT = 0.3F;

// How much of the rain reaches a surface at `position` facing `normal` (0 to 1): none under what the
// shelter map `shelter` (null: open everywhere) holds over it, a roof, a tree, a figure; none on what
// faces down; half on a wall (what drift and splashes bring); more on a face turned into the wind
inline float rainExposureAt(float3 position, float3 normal, float2 shear, float2 centre, float* shelter)
{
  const float3 fall   = normalize(float3(-shear.x, -1.0F, -shear.y));
  const float  facing = clamp(0.5F - 1.5F * dot(normal, fall), 0.0F, 1.0F);
  if(facing <= 0.0F)
    return 0.0F;
  if(shelter != nullptr)
  {
    const float3 out   = position + normal * RAIN_SHELTER_OUT;
    const int    texel = rainShelterTexel(out, shear, centre);
    if(texel >= 0 && out.y < shelter[texel] - 0.02F)
      return 0.0F;
  }
  return facing;
}

// The light table (rainLight): what a drop shows, a cosine-weighted mean of the environment round a
// direction, on a lat-long grid
#define RAIN_LIGHT_WIDTH 32
#define RAIN_LIGHT_HEIGHT 16

#endif  // RAIN_H_SLANG
