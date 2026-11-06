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
// The noise the clouds are cut from, as functions of a point in the unit cube (or square) that tile
// with it: what sky_clouds_noise.slang writes into the volumes once, and what
// tests/test_sky_clouds.cpp samples on the CPU to see how much sky a coverage covers.
//
// The recipe is the usual one for real-time clouds (Schneider, "The Real-Time Volumetric Cloudscapes
// of Horizon Zero Dawn", 2015; Hillaire's TileableVolumeNoise): billows from Perlin noise reshaped by
// inverted Worley noise, cut back by coarser Worley, and edges eroded by finer Worley. Every lattice
// is periodic in its cell count, so each volume tiles.
//

#ifndef SKY_CLOUDS_NOISE_H
#define SKY_CLOUDS_NOISE_H 1

#include "sky_clouds.h.slang"  // cloudRemap

// A well-mixed 3D integer hash (Jarzynski and Olano, "Hash Functions for GPU Rendering", 2020).
inline uint3 cloudPcg3d(uint3 v)
{
  v   = v * 1664525U + 1013904223U;
  v.x = v.x + v.y * v.z;
  v.y = v.y + v.z * v.x;
  v.z = v.z + v.x * v.y;
  v   = v ^ (v >> 16U);
  v.x = v.x + v.y * v.z;
  v.y = v.y + v.z * v.x;
  v.z = v.z + v.x * v.y;
  return v;
}

// Three uniform numbers in [0, 1) for a lattice cell, one stream per `salt`.
inline float3 cloudHash(int3 cell, int salt)
{
  const uint3 h = cloudPcg3d(uint3(uint(cell.x), uint(cell.y), uint(cell.z) + uint(salt) * 7919U));
  return float3(float(h.x), float(h.y), float(h.z)) * (1.0F / 4294967296.0F);
}

inline int cloudWrap(int i, int period)
{
  const int m = i % period;
  return m < 0 ? m + period : m;
}

inline int3 cloudWrap3(int3 c, int period)
{
  return int3(cloudWrap(c.x, period), cloudWrap(c.y, period), cloudWrap(c.z, period));
}

inline float cloudFade(float t)
{
  return t * t * t * (t * (t * 6.0F - 15.0F) + 10.0F);
}

// Gradient noise at `p` (in cells), periodic in `period` cells, about -1..1.
inline float cloudPerlin(float3 p, int period, int salt)
{
  const float3 fl = floor(p);
  const int3   i  = int3(int(fl.x), int(fl.y), int(fl.z));
  const float3 f  = p - fl;
  const float3 u  = float3(cloudFade(f.x), cloudFade(f.y), cloudFade(f.z));

  float n = 0.0F;
  for(int c = 0; c < 8; ++c)
  {
    const int3   o    = int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
    const float3 g    = normalize(cloudHash(cloudWrap3(i + o, period), salt) * 2.0F - float3(1.0F));
    const float3 d    = f - float3(float(o.x), float(o.y), float(o.z));
    const float  wx   = o.x == 1 ? u.x : 1.0F - u.x;
    const float  wy   = o.y == 1 ? u.y : 1.0F - u.y;
    const float  wz   = o.z == 1 ? u.z : 1.0F - u.z;
    n                += wx * wy * wz * dot(g, d);
  }
  // A 3D gradient noise peaks at sqrt(3)/2; scaled toward the unit range.
  return n * 1.1547F;
}

// Inverted cellular noise at `p` (in cells), periodic in `period` cells: 1 on a feature point,
// falling to 0 a cell away. One feature point per cell.
inline float cloudWorley(float3 p, int period, int salt)
{
  const float3 fl = floor(p);
  const int3   i  = int3(int(fl.x), int(fl.y), int(fl.z));
  const float3 f  = p - fl;

  float best = 1e9F;
  for(int z = -1; z <= 1; ++z)
    for(int y = -1; y <= 1; ++y)
      for(int x = -1; x <= 1; ++x)
      {
        const int3   o     = int3(x, y, z);
        const float3 point = float3(float(x), float(y), float(z)) + cloudHash(cloudWrap3(i + o, period), salt);
        const float3 d     = point - f;
        best               = min(best, dot(d, d));
      }
  return 1.0F - clamp(sqrt(best), 0.0F, 1.0F);
}

// Three octaves of it from `cells` cells across the unit cube up, weighted as Schneider's.
inline float cloudWorleyFbm(float3 p, int cells, int salt)
{
  return cloudWorley(p * float(cells), cells, salt) * 0.625F + cloudWorley(p * float(cells * 2), cells * 2, salt) * 0.25F
         + cloudWorley(p * float(cells * 4), cells * 4, salt) * 0.125F;
}

// Fractal gradient noise from `cells` across, `octaves` deep, about -1..1.
inline float cloudPerlinFbm(float3 p, int cells, int octaves, int salt)
{
  float sum = 0.0F;
  float amp = 1.0F;
  float norm = 0.0F;
  int   c   = cells;
  for(int o = 0; o < octaves; ++o)
  {
    sum += cloudPerlin(p * float(c), c, salt + o) * amp;
    norm += amp;
    amp *= 0.5F;
    c *= 2;
  }
  return sum / norm;
}

// The shape volume at `p` in the unit cube: rounded billows, 0 to 1.
inline float cloudShapeNoise(float3 p)
{
  const float perlin = clamp(cloudPerlinFbm(p, 4, 4, 11) * 0.5F + 0.5F, 0.0F, 1.0F);
  const float worley = cloudWorleyFbm(p, 4, 23);
  // Perlin-Worley: the Perlin's range lifted onto the Worley, so its lows become cell boundaries
  const float perlinWorley = cloudRemap(perlin, 0.0F, 1.0F, worley, 1.0F);
  const float coarse = cloudWorleyFbm(p, 4, 37) * 0.625F + cloudWorleyFbm(p, 8, 41) * 0.25F + cloudWorleyFbm(p, 16, 43) * 0.125F;
  const float shape  = cloudRemap(perlinWorley, coarse - 1.0F, 1.0F, 0.0F, 1.0F);
  // That lands between 0.74 and 0.90 at nine points in ten (measured, tests/test_sky_clouds.cpp), so
  // a coverage threshold below the band makes all of it cloud and one above none. Stretched over the
  // unit range the threshold sweeps all of it, and the volume's eight bits keep their levels.
  return clamp(cloudRemap(shape, 0.70F, 0.95F, 0.0F, 1.0F), 0.0F, 1.0F);
}

// The detail volume at `p` in the unit cube: finer cells to erode the edges with, 0 to 1.
inline float cloudDetailNoise(float3 p)
{
  return cloudWorleyFbm(p, 2, 53) * 0.625F + cloudWorleyFbm(p, 4, 59) * 0.25F + cloudWorleyFbm(p, 8, 61) * 0.125F;
}

// The weather map at `p` in the unit square: where over the land there is cloud, 0 to 1, about half
// on average.
inline float cloudWeatherNoise(float2 p)
{
  const float n = cloudPerlinFbm(float3(p.x, p.y, 0.0F), 4, 5, 71);
  return clamp(0.5F + 0.9F * n, 0.0F, 1.0F);
}

#endif  // SKY_CLOUDS_NOISE_H
