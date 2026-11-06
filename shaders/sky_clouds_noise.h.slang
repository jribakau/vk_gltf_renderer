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
// with it: what sky_clouds_noise.slang writes into the textures once, and what
// tests/test_sky_clouds.cpp evaluates on the CPU to fit the coverage.
//
// Ported from takram's three-geospatial clouds (packages/clouds/src/shaders: perlin.glsl,
// tileableNoise.glsl, cloudShape.frag, cloudShapeDetail.frag, localWeather.frag, turbulence.frag),
// GLSL to the subset Slang and C++ share. Their notices, which travel with the code:
//
//   three-geospatial -- The MIT License (MIT), Copyright (c) 2024 Shota Matsuda.
//
//   perlin.glsl is ported from GLM (glm/gtc/noise.inl): OpenGL Mathematics (GLM) is licensed under
//   The Happy Bunny License or MIT License. The MIT License, Copyright (c) 2005 - G-Truc Creation.
//
//   cloudShape.frag, cloudShapeDetail.frag and tileableNoise.glsl are based on
//   https://github.com/sebh/TileableVolumeNoise -- The MIT License (MIT), Copyright(c) 2017
//   Sebastien Hillaire.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//   and associated documentation files (the "Software"), to deal in the Software without
//   restriction, including without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all copies or
//   substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#ifndef SKY_CLOUDS_NOISE_H
#define SKY_CLOUDS_NOISE_H 1

// ---- GLSL's mod and fract, which Slang spells differently and C++ not at all -----------------

inline float cloudFract(float x)
{
  return x - floor(x);
}

inline float4 cloudFract4(float4 x)
{
  return x - floor(x);
}

inline float3 cloudMod3(float3 x, float y)
{
  return x - y * floor(x / y);
}

inline float4 cloudMod4(float4 x, float4 y)
{
  return x - y * floor(x / y);
}

// ---- Classic Perlin noise, periodic version (GLM's, by way of perlin.glsl) ------------------

inline float4 cloudMod289(float4 x)
{
  return x - floor(x * (1.0F / 289.0F)) * 289.0F;
}

inline float4 cloudPermute(float4 v)
{
  return cloudMod289((v * 34.0F + 1.0F) * v);
}

inline float4 cloudTaylorInvSqrt(float4 r)
{
  return 1.79284291400159F - 0.85373472095314F * r;
}

inline float4 cloudFade(float4 v)
{
  return v * v * v * (v * (v * 6.0F - 15.0F) + 10.0F);
}

// One corner set's gradients: four lattice hashes `ixy` to 4D vectors, one component per field.
// A struct rather than out parameters, which C++ has no keyword for.
struct CloudGradients
{
  float4 x;
  float4 y;
  float4 z;
  float4 w;
};

inline CloudGradients cloudGradients(float4 ixy)
{
  CloudGradients g;
  g.x = ixy / 7.0F;
  g.y = floor(g.x) / 7.0F;
  g.z = floor(g.y) / 6.0F;
  g.x = cloudFract4(g.x) - 0.5F;
  g.y = cloudFract4(g.y) - 0.5F;
  g.z = cloudFract4(g.z) - 0.5F;
  g.w = float4(0.75F) - abs(g.x) - abs(g.y) - abs(g.z);
  const float4 sw = step(g.w, float4(0.0F));
  g.x -= sw * (step(float4(0.0F), g.x) - 0.5F);
  g.y -= sw * (step(float4(0.0F), g.y) - 0.5F);
  return g;
}

// 4D gradient noise at `position`, periodic in `rep` along each axis, about -1..1.
inline float cloudPerlin(float4 position, float4 rep)
{
  const float4 Pi0 = cloudMod4(floor(position), rep);
  const float4 Pi1 = cloudMod4(Pi0 + 1.0F, rep);
  const float4 Pf0 = cloudFract4(position);
  const float4 Pf1 = Pf0 - 1.0F;
  const float4 ix  = float4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  const float4 iy  = float4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
  const float4 iz0 = float4(Pi0.z);
  const float4 iz1 = float4(Pi1.z);
  const float4 iw0 = float4(Pi0.w);
  const float4 iw1 = float4(Pi1.w);

  const float4 ixy   = cloudPermute(cloudPermute(ix) + iy);
  const float4 ixy0  = cloudPermute(ixy + iz0);
  const float4 ixy1  = cloudPermute(ixy + iz1);
  const float4 ixy00 = cloudPermute(ixy0 + iw0);
  const float4 ixy01 = cloudPermute(ixy0 + iw1);
  const float4 ixy10 = cloudPermute(ixy1 + iw0);
  const float4 ixy11 = cloudPermute(ixy1 + iw1);

  const CloudGradients c00 = cloudGradients(ixy00);
  const CloudGradients c01 = cloudGradients(ixy01);
  const CloudGradients c10 = cloudGradients(ixy10);
  const CloudGradients c11 = cloudGradients(ixy11);
  const float4 gx00 = c00.x, gy00 = c00.y, gz00 = c00.z, gw00 = c00.w;
  const float4 gx01 = c01.x, gy01 = c01.y, gz01 = c01.z, gw01 = c01.w;
  const float4 gx10 = c10.x, gy10 = c10.y, gz10 = c10.z, gw10 = c10.w;
  const float4 gx11 = c11.x, gy11 = c11.y, gz11 = c11.z, gw11 = c11.w;

  float4 g0000 = float4(gx00.x, gy00.x, gz00.x, gw00.x);
  float4 g1000 = float4(gx00.y, gy00.y, gz00.y, gw00.y);
  float4 g0100 = float4(gx00.z, gy00.z, gz00.z, gw00.z);
  float4 g1100 = float4(gx00.w, gy00.w, gz00.w, gw00.w);
  float4 g0010 = float4(gx10.x, gy10.x, gz10.x, gw10.x);
  float4 g1010 = float4(gx10.y, gy10.y, gz10.y, gw10.y);
  float4 g0110 = float4(gx10.z, gy10.z, gz10.z, gw10.z);
  float4 g1110 = float4(gx10.w, gy10.w, gz10.w, gw10.w);
  float4 g0001 = float4(gx01.x, gy01.x, gz01.x, gw01.x);
  float4 g1001 = float4(gx01.y, gy01.y, gz01.y, gw01.y);
  float4 g0101 = float4(gx01.z, gy01.z, gz01.z, gw01.z);
  float4 g1101 = float4(gx01.w, gy01.w, gz01.w, gw01.w);
  float4 g0011 = float4(gx11.x, gy11.x, gz11.x, gw11.x);
  float4 g1011 = float4(gx11.y, gy11.y, gz11.y, gw11.y);
  float4 g0111 = float4(gx11.z, gy11.z, gz11.z, gw11.z);
  float4 g1111 = float4(gx11.w, gy11.w, gz11.w, gw11.w);

  const float4 norm00 = cloudTaylorInvSqrt(float4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 *= norm00.x;
  g0100 *= norm00.y;
  g1000 *= norm00.z;
  g1100 *= norm00.w;

  const float4 norm01 = cloudTaylorInvSqrt(float4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 *= norm01.x;
  g0101 *= norm01.y;
  g1001 *= norm01.z;
  g1101 *= norm01.w;

  const float4 norm10 = cloudTaylorInvSqrt(float4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 *= norm10.x;
  g0110 *= norm10.y;
  g1010 *= norm10.z;
  g1110 *= norm10.w;

  const float4 norm11 = cloudTaylorInvSqrt(float4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 *= norm11.x;
  g0111 *= norm11.y;
  g1011 *= norm11.z;
  g1111 *= norm11.w;

  const float n0000 = dot(g0000, Pf0);
  const float n1000 = dot(g1000, float4(Pf1.x, Pf0.y, Pf0.z, Pf0.w));
  const float n0100 = dot(g0100, float4(Pf0.x, Pf1.y, Pf0.z, Pf0.w));
  const float n1100 = dot(g1100, float4(Pf1.x, Pf1.y, Pf0.z, Pf0.w));
  const float n0010 = dot(g0010, float4(Pf0.x, Pf0.y, Pf1.z, Pf0.w));
  const float n1010 = dot(g1010, float4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  const float n0110 = dot(g0110, float4(Pf0.x, Pf1.y, Pf1.z, Pf0.w));
  const float n1110 = dot(g1110, float4(Pf1.x, Pf1.y, Pf1.z, Pf0.w));
  const float n0001 = dot(g0001, float4(Pf0.x, Pf0.y, Pf0.z, Pf1.w));
  const float n1001 = dot(g1001, float4(Pf1.x, Pf0.y, Pf0.z, Pf1.w));
  const float n0101 = dot(g0101, float4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  const float n1101 = dot(g1101, float4(Pf1.x, Pf1.y, Pf0.z, Pf1.w));
  const float n0011 = dot(g0011, float4(Pf0.x, Pf0.y, Pf1.z, Pf1.w));
  const float n1011 = dot(g1011, float4(Pf1.x, Pf0.y, Pf1.z, Pf1.w));
  const float n0111 = dot(g0111, float4(Pf0.x, Pf1.y, Pf1.z, Pf1.w));
  const float n1111 = dot(g1111, Pf1);

  const float4 f    = cloudFade(Pf0);
  const float4 n_0w = lerp(float4(n0000, n1000, n0100, n1100), float4(n0001, n1001, n0101, n1101), float4(f.w));
  const float4 n_1w = lerp(float4(n0010, n1010, n0110, n1110), float4(n0011, n1011, n0111, n1111), float4(f.w));
  const float4 n_zw = lerp(n_0w, n_1w, float4(f.z));
  const float  n_y0 = lerp(n_zw.x, n_zw.z, f.y);
  const float  n_y1 = lerp(n_zw.y, n_zw.w, f.y);
  return 2.2F * lerp(n_y0, n_y1, f.x);
}

// Octaves of it from `frequency` cells across the unit cube, each twice the last at half the weight.
inline float cloudPerlinNoise(float3 point, float3 frequency, int octaveCount)
{
  float  sum       = 0.0F;
  float  weightSum = 0.0F;
  float  weight    = 1.0F;
  float3 next      = frequency;
  for(int i = 0; i < octaveCount; ++i)
  {
    const float4 p = float4(point.x * next.x, point.y * next.y, point.z * next.z, 0.0F);
    sum += cloudPerlin(p, float4(next.x, next.y, next.z, 1.0F)) * weight;
    weightSum += weight;
    weight *= 0.5F;
    next = next * 2.0F;
  }
  return sum / weightSum;  // intentionally unclamped, as takram's
}

// ---- Tileable cellular noise (Hillaire's, by way of tileableNoise.glsl) ---------------------

inline float cloudHash(float n)
{
  return cloudFract(sin(n + 1.951F) * 43758.5453F);
}

inline float cloudValueNoise(float3 x)
{
  const float3 p = floor(x);
  float3       f = x - p;
  f              = f * f * (3.0F - 2.0F * f);
  const float n  = p.x + p.y * 57.0F + 113.0F * p.z;
  const float a  = lerp(lerp(cloudHash(n + 0.0F), cloudHash(n + 1.0F), f.x), lerp(cloudHash(n + 57.0F), cloudHash(n + 58.0F), f.x), f.y);
  const float b  = lerp(lerp(cloudHash(n + 113.0F), cloudHash(n + 114.0F), f.x), lerp(cloudHash(n + 170.0F), cloudHash(n + 171.0F), f.x), f.y);
  return lerp(a, b, f.z);
}

// The squared distance to the nearest feature point, `cellCount` cells across the unit cube and
// tiling with it, clamped to 1.
inline float cloudWorleyNoise(float3 p, float cellCount)
{
  const float3 cell = p * cellCount;
  const float3 base = floor(cell);
  float        d    = 1.0e10F;
  for(int x = -1; x <= 1; ++x)
    for(int y = -1; y <= 1; ++y)
      for(int z = -1; z <= 1; ++z)
      {
        const float3 tp = base + float3(float(x), float(y), float(z));
        const float3 to = cell - tp - cloudValueNoise(cloudMod3(tp, cellCount));
        d               = min(d, dot(to, to));
      }
  return clamp(d, 0.0F, 1.0F);
}

// ---- The four textures (cloudShape.frag, cloudShapeDetail.frag, localWeather.frag, turbulence.frag)

inline float cloudRemapNoise(float x, float min1, float max1, float min2, float max2)
{
  return min2 + (x - min1) / (max1 - min1) * (max2 - min2);
}

// The shape volume at `p` in the unit cube: Perlin lifted onto inverted Worley, cut back by coarser
// Worley. Unclamped, as takram's; the volume's eight bits clamp it.
inline float cloudShapeNoise(float3 p)
{
  const float perlin = clamp(cloudPerlinNoise(p, float3(8.0F), 3), 0.0F, 1.0F);
  const float cells  = 4.0F;
  const float fbm    = (1.0F - cloudWorleyNoise(p, cells * 2.0F)) * 0.625F + (1.0F - cloudWorleyNoise(p, cells * 8.0F)) * 0.25F
                    + (1.0F - cloudWorleyNoise(p, cells * 14.0F)) * 0.125F;
  const float perlinWorley = cloudRemapNoise(perlin, 0.0F, 1.0F, fbm, 1.0F);

  const float w0        = 1.0F - cloudWorleyNoise(p, cells * 2.0F);
  const float w1        = 1.0F - cloudWorleyNoise(p, cells * 4.0F);
  const float w2        = 1.0F - cloudWorleyNoise(p, cells * 8.0F);
  const float w3        = 1.0F - cloudWorleyNoise(p, cells * 16.0F);
  const float fbm0      = w0 * 0.625F + w1 * 0.25F + w2 * 0.125F;
  const float fbm1      = w1 * 0.625F + w2 * 0.25F + w3 * 0.125F;
  const float fbm2      = w2 * 0.75F + w3 * 0.25F;
  const float worleyFbm = fbm0 * 0.625F + fbm1 * 0.25F + fbm2 * 0.125F;
  return cloudRemapNoise(perlinWorley, worleyFbm - 1.0F, 1.0F, 0.0F, 1.0F);
}

// The detail volume at `p` in the unit cube: finer inverted Worley, 0 to 1.
inline float cloudDetailNoise(float3 p)
{
  const float cells = 2.0F;
  const float w0    = 1.0F - cloudWorleyNoise(p, cells * 1.0F);
  const float w1    = 1.0F - cloudWorleyNoise(p, cells * 2.0F);
  const float w2    = 1.0F - cloudWorleyNoise(p, cells * 4.0F);
  const float w3    = 1.0F - cloudWorleyNoise(p, cells * 8.0F);
  const float fbm0  = w0 * 0.625F + w1 * 0.25F + w2 * 0.125F;
  const float fbm1  = w1 * 0.625F + w2 * 0.25F + w3 * 0.125F;
  const float fbm2  = w2 * 0.75F + w3 * 0.25F;
  return fbm0 * 0.625F + fbm1 * 0.25F + fbm2 * 0.125F;
}

inline float cloudWorleyFbm(float3 point, float frequency, float amplitude, float lacunarity, float gain, int octaveCount)
{
  float noise = 0.0F;
  for(int i = 0; i < octaveCount; ++i)
  {
    noise += amplitude * (1.0F - cloudWorleyNoise(point, frequency));
    frequency *= lacunarity;
    amplitude *= gain;
  }
  return noise;
}

// The weather map at `p` in the unit square, one layer to a channel: r the low cumulus (where the
// deeper ones are not), g the deeper cumulus in large fields, b the high cloud in streaks, a unused.
inline float4 cloudWeatherNoise(float2 p)
{
  const float3 point = float3(p.x, p.y, 0.0F);
  const float  mid   = smoothstep(1.0F, 1.4F, cloudWorleyFbm(point + float3(0.5F), 8.0F, 0.4F, 2.0F, 0.95F, 4));
  const float  low   = clamp(smoothstep(0.8F, 1.4F, cloudWorleyFbm(point, 16.0F, 0.4F, 2.0F, 0.95F, 4)) - mid, 0.0F, 1.0F);
  const float  high  = smoothstep(-0.5F, 0.5F, cloudPerlinNoise(point, float3(6.0F, 12.0F, 1.0F), 8));
  return float4(low, mid, high, 1.0F);
}

inline float3 cloudPerlin3d(float3 point)
{
  const float3 frequency = float3(12.0F);
  return float3(cloudPerlinNoise(point, frequency, 3),
                cloudPerlinNoise(float3(point.y, point.z, point.x) + float3(-19.1F, 33.4F, 47.2F), frequency, 3),
                cloudPerlinNoise(float3(point.z, point.x, point.y) + float3(74.2F, -124.5F, 99.4F), frequency, 3));
}

// The turbulence map at `p` in the unit square: a unit curl of gradient noise, packed to 0..1.
inline float3 cloudTurbulenceNoise(float2 p)
{
  const float  delta = 0.1F;
  const float3 point = float3(p.x, p.y, 0.0F);
  const float3 dx    = float3(delta, 0.0F, 0.0F);
  const float3 dy    = float3(0.0F, delta, 0.0F);
  const float3 dz    = float3(0.0F, 0.0F, delta);
  const float3 px0   = cloudPerlin3d(point - dx);
  const float3 px1   = cloudPerlin3d(point + dx);
  const float3 py0   = cloudPerlin3d(point - dy);
  const float3 py1   = cloudPerlin3d(point + dy);
  const float3 pz0   = cloudPerlin3d(point - dz);
  const float3 pz1   = cloudPerlin3d(point + dz);
  const float  x     = py1.z - py0.z - pz1.y + pz0.y;
  const float  y     = pz1.x - pz0.x - px1.z + px0.z;
  const float  z     = px1.y - px0.y - py1.x + py0.x;
  return 0.5F * normalize(float3(x, y, z) * (1.0F / (2.0F * delta))) + 0.5F;
}

#endif  // SKY_CLOUDS_NOISE_H
