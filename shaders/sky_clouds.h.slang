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
// The physical sky's clouds: one layer of them over the planet, what is cloud and what is not, how
// it is lit, and the image it is baked into. Plain arithmetic in the subset Slang and C++ share, so
// tests/test_sky_clouds.cpp runs the same functions the shaders do; the noise sampling and the march
// itself are sky_clouds_bake.slang.
//
// Units are the atmosphere's: kilometres, and the observer on the +Y axis of a planet centred at the
// origin (sky_bruneton_radiance.h.slang).
//
// **Why one value for the sun.** A scene is metres across and the clouds are kilometres up, so the
// whole scene is in one cloud's shadow or none: the sun's transmittance through the layer, toward the
// sun from the observer, is a single number, published with the solar disk. Nothing per pixel.
//
// **What the image holds.** For each direction the observer can see above its horizon:
//
//     rgb = airTransmittance * cloudLight + (1 - cloudTransmittance) * airInFront
//     a   = cloudTransmittance
//
// so that the sky seen through it is `sky * a + rgb` (composeClouds). `sky` is the clear sky along
// the whole ray, which already holds the air in front of the clouds; where a cloud hides what is
// behind, that air is all that is left of it, which is the second term.
//

#ifndef SKY_CLOUDS_H
#define SKY_CLOUDS_H 1

// ---- the layer -----------------------------------------------------------------------------

// The base: a fair-weather cumulus base over land. Fixed in this first slice.
static const float CLOUD_BOTTOM_KM = 1.5F;

// Extinction at full density, 1/km. Cumulus water content (~0.5 g/m^3) in droplets of ~10 um is
// 1.5 * LWC / (rho * r_eff) = 75 /km; the noise rarely reaches full density, so a cloud's core sits
// at 20-50, an optical depth of tens through a kilometre of it, as real ones measure.
static const float CLOUD_EXTINCTION = 60.0F;

// The noise's periods, in km. The shape's billows are about a quarter of its period across, the
// detail's eddies a quarter of its; the weather map's cloud fields a fifth of its. Its second,
// smaller octave is at a period that shares no factor with the first, so the two do not repeat
// together within anything the observer can see.
static const float CLOUD_SHAPE_PERIOD   = 8.0F;
static const float CLOUD_DETAIL_PERIOD  = 1.3F;
static const float CLOUD_WEATHER_PERIOD = 61.0F;
static const float CLOUD_WEATHER_PERIOD2 = 23.7F;

// How far a view ray is marched at most, km: the layer seen edge-on at the horizon is ~40 km deep at
// 140 km, and past that what the clouds add is mostly the haze in front of them.
static const float CLOUD_MAX_DISTANCE = 240.0F;

// Where a ray from inside a sphere of radius R leaves it.
inline float cloudSphereExit(float r0, float mu, float R)
{
  const float disc = r0 * r0 * (mu * mu - 1.0F) + R * R;
  return -r0 * mu + sqrt(max(disc, 0.0F));
}

// Where a ray from outside a sphere of radius R first meets it, or -1 when it misses.
inline float cloudSphereEntry(float r0, float mu, float R)
{
  const float disc = r0 * r0 * (mu * mu - 1.0F) + R * R;
  if(disc < 0.0F)
    return -1.0F;
  const float t = -r0 * mu - sqrt(disc);
  return t >= 0.0F ? t : -1.0F;
}

// The stretch of a ray inside the layer between radii rBottom and rTop: (enter, leave), with
// leave <= enter when the ray never reaches it. The observer is `r0` from the planet's centre and
// the ray's zenith cosine is `mu`. A ray that meets the ground first is the caller's to drop.
inline float2 cloudLayerSegment(float r0, float mu, float rBottom, float rTop)
{
  if(r0 < rBottom)
    return float2(cloudSphereExit(r0, mu, rBottom), cloudSphereExit(r0, mu, rTop));
  if(r0 <= rTop)
  {
    const float down = cloudSphereEntry(r0, mu, rBottom);
    return float2(0.0F, down >= 0.0F ? down : cloudSphereExit(r0, mu, rTop));
  }
  const float enter = cloudSphereEntry(r0, mu, rTop);
  if(enter < 0.0F)
    return float2(0.0F, -1.0F);
  const float down = cloudSphereEntry(r0, mu, rBottom);
  return float2(enter, down >= 0.0F ? down : cloudSphereExit(r0, mu, rTop));
}

// ---- what is cloud -------------------------------------------------------------------------

inline float cloudRemap(float x, float a, float b, float c, float d)
{
  return c + (x - a) * (d - c) / max(b - a, 1e-6F);
}

// The weather over a point, from the map read at its two periods (`a` at CLOUD_WEATHER_PERIOD, `b` at
// CLOUD_WEATHER_PERIOD2): mostly the large fields, broken up by the smaller ones, and stretched back
// to the map's own spread, which blending two narrows.
inline float cloudWeather(float a, float b)
{
  return clamp(0.5F + (0.75F * a + 0.25F * b - 0.5F) * 1.25F, 0.0F, 1.0F);
}

// How much cloud the weather allows over a point: 0 none, 1 all the shape noise holds there, and
// past 1 more than it holds, which closes the gaps a deck would otherwise keep. The weather `level`
// moves the map through its whole range: from a fifth its highest fields pass, as small fair-weather
// cumulus; by a half there is broken cloud everywhere; at 1 a deck with no holes.
inline float cloudLocalCoverage(float weather, float level)
{
  return clamp(weather + 2.0F * level - 1.0F, 0.0F, 1.3F);
}

// The weather level for a cloud cover: the share of the land the clouds shade with the sun overhead,
// which is what the panel's Coverage says (and, near enough, the share of the sky overhead they
// hide, how cloud cover is reported). Measured for a 1.2 km layer by marching columns through it
// (SkyClouds.DISABLED_CoverageCurve in tests/test_sky_clouds.cpp, which re-fits this table if the
// noise or the density changes) and inverted, a knot every 5%. A thinner layer shades a little less.
static const float CLOUD_LEVEL_FOR_COVER[21] = {0.2000F, 0.3452F, 0.3833F, 0.4130F, 0.4388F, 0.4614F, 0.4811F,
                                                0.4986F, 0.5154F, 0.5328F, 0.5511F, 0.5697F, 0.5882F, 0.6068F,
                                                0.6261F, 0.6476F, 0.6728F, 0.7030F, 0.7369F, 0.7791F, 1.0000F};

inline float cloudLevel(float cover)
{
  const float x = clamp(cover, 0.0F, 1.0F) * 20.0F;
  const int   i = min(int(x), 19);
  return lerp(CLOUD_LEVEL_FOR_COVER[i], CLOUD_LEVEL_FOR_COVER[i + 1], x - float(i));
}

// The layer's vertical shape, 0 at the base to 1 at the highest tops: a flat base that fills within a
// tenth of the depth, tops that round off over the upper half.
inline float cloudHeightProfile(float h)
{
  return clamp(h / 0.1F, 0.0F, 1.0F) * clamp((1.0F - h) / 0.6F, 0.0F, 1.0F);
}

// Density, 0 to 1, at a point `h` of the way up the layer. `shape` and `detail` are the noise
// volumes there, `weather` the map and `level` its cloudLevel: the shape is cut down to what the
// weather allows, the cut lowest at the rounded tops, and its edges eroded by the detail -- wisps at
// the base, billows above.
inline float cloudDensity(float shape, float detail, float weather, float h, float level)
{
  if(h <= 0.0F || h >= 1.0F)
    return 0.0F;
  const float local = cloudLocalCoverage(weather, level);
  if(local <= 0.0F)
    return 0.0F;
  const float base = clamp(cloudRemap(shape * cloudHeightProfile(h), 1.0F - local, 1.0F, 0.0F, 1.0F), 0.0F, 1.0F);
  if(base <= 0.0F)
    return 0.0F;
  const float erode = lerp(detail, 1.0F - detail, clamp(h * 4.0F, 0.0F, 1.0F));
  return clamp(cloudRemap(base, erode * 0.35F, 1.0F, 0.0F, 1.0F), 0.0F, 1.0F);
}

// The share of the land under the layer, seen from afar, that the sun still reaches: what the
// distant ground and the haze in front of the clouds are lit by. Coverage is measured as the share
// the clouds shade (cloudLevel), so this is what is left of it.
inline float cloudSunlitAir(float cover)
{
  return 1.0F - clamp(cover, 0.0F, 1.0F);
}

// ---- how it is lit -------------------------------------------------------------------------

// Henyey-Greenstein, normalised over the sphere.
inline float cloudHg(float g, float cosTheta)
{
  const float g2 = g * g;
  const float d  = max(1.0F + g2 - 2.0F * g * cosTheta, 1e-6F);
  return (1.0F - g2) / (4.0F * 3.14159265F * d * sqrt(d));
}

// Cloud droplets' phase: a strong forward lobe (the silver lining, the glow round a sun behind thin
// cloud) and a weak backward one, still normalised. `cosTheta` is between the way the light travels
// and the way it leaves, so 1 is straight on.
inline float cloudPhase(float cosTheta)
{
  return lerp(cloudHg(0.8F, cosTheta), cloudHg(-0.3F, cosTheta), 0.2F);
}

// A thick cloud's plane albedo: what its sunlit face reflects, as a white surface would.
static const float CLOUD_DIFFUSE_ALBEDO = 0.8F;
// How fast the diffused light thins with depth: a conservative slab of optical depth tau passes
// 1 / (1 + 3/4 (1 - g) tau) of it, with real droplets' g of 0.85.
static const float CLOUD_DIFFUSION = 0.75F * (1.0F - 0.85F);

// What a point of cloud sends toward the eye, per unit of optical depth along the view ray: a
// radiance. `sunIrradiance` is the sunlight arriving at the layer, after the air above it;
// `sunDepth` the cloud's own optical depth between the point and the sun; `skyIrradiance` the sky's
// light on the layer from above; `h` the height in it.
//
// Three terms. The sunlight scattered once on its way to the eye, through the cloud's phase. The
// sunlight diffused through the cloud by many scatterings: a cloud's brightness is almost all this,
// as thick ones are near white, and it is what the slab law above carries down into the cloud. And
// the sky's light, diffused alike from above, less of it toward the base. Their albedo is the
// cloud's: droplets absorb next to nothing.
inline float3 cloudLight(float3 sunIrradiance, float sunDepth, float cosTheta, float3 skyIrradiance, float h)
{
  const float kInvPi  = 0.318309886F;
  const float single  = cloudPhase(cosTheta) * exp(-sunDepth);
  const float diffuse = CLOUD_DIFFUSE_ALBEDO * kInvPi / (1.0F + CLOUD_DIFFUSION * sunDepth);
  const float sky     = CLOUD_DIFFUSE_ALBEDO * kInvPi * lerp(0.35F, 1.0F, clamp(h, 0.0F, 1.0F));
  return sunIrradiance * (single + diffuse) + skyIrradiance * sky;
}

// ---- the image -----------------------------------------------------------------------------

// The cloud image covers what the observer sees above its horizon: azimuth across the columns as the
// lat-long's (latlongToDir), elevation up the rows from the horizon (row 0) to the zenith, on a
// square root so the rows crowd at the horizon where the clouds do. `minElevation` is the
// horizon's, in radians: asin(SkySunDisk::horizonMu).
inline float2 cloudImageUv(float3 dir, float minElevation)
{
  const float kPi = 3.14159265F;
  const float u   = atan2(dir.z, dir.x) * (0.5F / kPi) + 0.5F;
  const float e   = asin(clamp(dir.y, -1.0F, 1.0F));
  const float v   = sqrt(clamp((e - minElevation) / (0.5F * kPi - minElevation), 0.0F, 1.0F));
  return float2(u, v);
}

inline float3 cloudImageDir(float2 uv, float minElevation)
{
  const float kPi = 3.14159265F;
  const float phi = 2.0F * kPi * uv.x - kPi;
  const float e   = minElevation + uv.y * uv.y * (0.5F * kPi - minElevation);
  const float c   = cos(e);
  return float3(cos(phi) * c, sin(e), sin(phi) * c);
}

// One texel of the image (see the top of the file).
inline float4 cloudTexel(float3 cloudLightSeen, float cloudTransmittance, float3 airTransmittance, float3 airInFront)
{
  return float4(cloudLightSeen * airTransmittance + airInFront * (1.0F - cloudTransmittance), cloudTransmittance);
}

// The sky along a ray, with the clouds over it. No clouds is a texel of (0, 0, 0, 1), which gives
// back `sky` exactly.
inline float3 composeClouds(float3 sky, float4 texel)
{
  return sky * texel.w + float3(texel.x, texel.y, texel.z);
}

#endif  // SKY_CLOUDS_H
