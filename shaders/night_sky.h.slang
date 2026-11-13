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

//
// The night sky past the air: the upper air's own glow (airglow), the Milky Way and the stars, all
// in the sun's light above the air, so the sky's units hold them as they hold the sun (night_sky
// in docs/sky.md). Slang and C++ (tests/test_night_sky.cpp).
//
// Directions here are celestial (equatorial, J2000): x toward right ascension 0 on the equator, y
// toward 90 degrees, z the north pole. The host turns them into the world by the clock
// (SceneFrameInfo::celestialX, Y, Z: where those three point).
//

#ifndef NIGHT_SKY_H
#define NIGHT_SKY_H 1

// ---- the night's glow ------------------------------------------------------------------------------

// A moonless night's sky at the zenith, far from towns: 22 magnitudes a square arcsecond, 2e-4 cd/m^2,
// the airglow with the faint stars and the zodiacal light in it; against the sun's disk above the air,
// 1.87e9 cd/m^2 (its 1.27e5 lux over its 6.8e-5 sr).
static const float NIGHT_GLOW_ZENITH = 1.07e-13F;
// The airglow's layer, km up (the oxygen line's): seen at a slant, through more of it (nightGlowPath)
static const float NIGHT_GLOW_HEIGHT = 90.0F;
// The Milky Way's brightest, the star clouds toward the galaxy's centre, against that glow: some
// 19.5 magnitudes a square arcsecond against 22, at the band's middle (milkyWayBand's 2.2)
static const float NIGHT_MILKY_WAY = 1.1F;

// How much more of the glow's layer a ray at `mu` (the cosine of its angle from the zenith) crosses
// than one straight up, from a planet of radius `bottomRadius` km: van Rhijn's, 6 at the horizon.
inline float nightGlowPath(float mu, float bottomRadius)
{
  const float s = bottomRadius / (bottomRadius + NIGHT_GLOW_HEIGHT);
  return 1.0F / sqrt(max(1.0F - s * s * (1.0F - mu * mu), 1e-4F));
}

// The glow's colour, luminance 1: the oxygen line's green with the starlight the air scatters, the
// teal grey a dark site's sky photographs as (the air's reddening on the way in is the caller's)
inline float3 nightGlowColor()
{
  return float3(0.913F, 1.014F, 1.116F);
}

// ---- the stars' grid ---------------------------------------------------------------------------------

// The stars are binned by right ascension and declination, two degrees a cell (NightStarCell), so a
// ray looks at its own cell and the eight round it.
#define NIGHT_STAR_CELLS_RA 180
#define NIGHT_STAR_CELLS_DEC 90

inline float3 celestialUnit(float raDeg, float decDeg)
{
  const float ra  = raDeg * 0.0174532925F;
  const float dec = decDeg * 0.0174532925F;
  return float3(cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec));
}

// The cell column of a celestial direction, 0 at right ascension 0, eastward
inline int nightStarColumn(float3 v)
{
  float ra = atan2(v.y, v.x);
  if(ra < 0.0F)
    ra += 6.28318530718F;
  return min(int(ra * (float(NIGHT_STAR_CELLS_RA) / 6.28318530718F)), NIGHT_STAR_CELLS_RA - 1);
}

// ... and its row, 0 at the south pole
inline int nightStarRow(float3 v)
{
  const float dec = asin(clamp(v.z, -1.0F, 1.0F));
  return clamp(int((dec + 1.57079632679F) * (float(NIGHT_STAR_CELLS_DEC) / 3.14159265359F)), 0, NIGHT_STAR_CELLS_DEC - 1);
}

// A star's light against the sun's above the air, by its V magnitude (the sun's -26.74)
inline float starLightShare(float vMagnitude)
{
  return pow(10.0F, -0.4F * (vMagnitude + 26.74F));
}

// ---- the Milky Way -----------------------------------------------------------------------------------

// The galactic frame (IAU 1958, in J2000): its north pole, and its centre in Sagittarius
inline float3 galacticNorth()
{
  return celestialUnit(192.85948F, 27.12825F);
}
inline float3 galacticCentre()
{
  return celestialUnit(266.40510F, -28.93617F);
}

// A celestial direction's galactic longitude (0 toward the centre, -pi to pi) and latitude, radians
inline float2 galactic(float3 v)
{
  const float3 z = galacticNorth();
  const float3 x = normalize(galacticCentre() - z * dot(galacticCentre(), z));
  const float3 y = cross(z, x);
  return float2(atan2(dot(v, y), dot(v, x)), asin(clamp(dot(v, z), -1.0F, 1.0F)));
}

// The band's smooth light, before its clouds and dust: a disc some ten degrees wide and brightest
// toward the centre, and the bulge round the centre. 2.2 at the centre, a quarter of a unit at the
// anticentre, nothing far off the plane.
inline float milkyWayBand(float l, float b)
{
  const float toCentre = 0.5F + 0.5F * cos(l);
  const float width    = 0.09F + 0.05F * toCentre;
  const float disc     = exp(-(b * b) / (2.0F * width * width)) * (0.25F + 0.75F * toCentre * toCentre);
  const float bulge    = exp(-(l * l + b * b) / (2.0F * 0.15F * 0.15F));
  return disc + 1.2F * bulge;
}

inline float nightHash(int3 c)
{
  uint h = uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u;
  h ^= h >> 13u;
  h *= 0x5bd1e995u;
  h ^= h >> 15u;
  return float(h & 0xffffffu) * (1.0F / 16777215.0F);
}

// Value noise, 0 to 1, smooth between the lattice's points
inline float nightNoise(float3 p)
{
  const float3 fl = floor(p);
  const int3   c  = int3(int(fl.x), int(fl.y), int(fl.z));
  float3       f  = p - fl;
  f               = f * f * (float3(3.0F) - 2.0F * f);
  const float a = lerp(nightHash(c), nightHash(c + int3(1, 0, 0)), f.x);
  const float b = lerp(nightHash(c + int3(0, 1, 0)), nightHash(c + int3(1, 1, 0)), f.x);
  const float d = lerp(nightHash(c + int3(0, 0, 1)), nightHash(c + int3(1, 0, 1)), f.x);
  const float e = lerp(nightHash(c + int3(0, 1, 1)), nightHash(c + int3(1, 1, 1)), f.x);
  return lerp(lerp(a, b, f.y), lerp(d, e, f.y), f.z);
}

inline float nightFbm(float3 p)
{
  float sum = 0.0F, amplitude = 0.5F;
  for(int i = 0; i < 5; ++i)
  {
    sum += amplitude * nightNoise(p);
    p *= 2.03F;
    amplitude *= 0.5F;
  }
  return sum / 0.96875F;
}

// The Milky Way along a celestial direction, against NIGHT_GLOW_ZENITH, in colour: the band, broken
// into star clouds, and the dust that splits it from Cygnus to Sagittarius (the Great Rift). Its warm
// bulge and bluer arms.
inline float3 milkyWay(float3 v)
{
  const float2 g = galactic(v);
  const float  l = g.x, b = g.y;
  const float  band = milkyWayBand(l, b);
  if(band < 1e-3F)
    return float3(0.0F);
  const float clouds = nightFbm(v * 9.0F);
  const float lane   = (b - 0.015F * sin(3.0F * l)) / 0.03F;
  const float rift   = exp(-lane * lane) * (1.0F - smoothstep(0.3F, 1.5F, abs(l)));
  // and the dark nebulae along the plane, patches of dust the band's light is cut by
  const float onPlane = exp(-(b * b) / (2.0F * 0.12F * 0.12F));
  const float patches = smoothstep(0.52F, 0.72F, nightFbm(v * 16.0F + float3(3.0F))) * onPlane;
  const float dust    = max(rift * (0.55F + 0.45F * nightFbm(v * 23.0F + float3(7.0F))), patches);
  const float light   = band * (0.2F + 1.5F * clouds * clouds * clouds) * (1.0F - 0.85F * dust);
  const float warm   = clamp(exp(-(l * l + b * b) / (2.0F * 0.3F * 0.3F)) + 0.3F * (0.5F + 0.5F * cos(l)), 0.0F, 1.0F);
  return NIGHT_MILKY_WAY * light * lerp(float3(0.88F, 0.98F, 1.12F), float3(1.12F, 0.98F, 0.8F), float3(warm));
}

#endif  // NIGHT_SKY_H
