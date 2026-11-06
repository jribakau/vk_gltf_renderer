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
// The physical sky's clouds: the layers, what is cloud, how it is lit, and the image it is baked
// into. Plain arithmetic in the subset Slang and C++ share, so tests/test_sky_clouds.cpp runs the
// same functions the shaders do; the texture reads and the march are sky_clouds_bake.slang.
//
// The cloud model -- weather, density, phase, multiple scattering -- is ported from takram's
// three-geospatial clouds (packages/clouds/src/shaders/clouds.glsl and clouds.frag), MIT,
// Copyright (c) 2024 Shota Matsuda; the full notice is in sky_clouds_noise.h.slang. It is theirs
// in the atmosphere's kilometres instead of metres. What is ours is around it: the image, its
// composition over the sky, and what the cover on the panel means (cloudCoverageFor).
//
// The observer is on the +Y axis of a planet centred at the origin (sky_bruneton_radiance.h.slang).
//
// **Why one value for the sun.** A scene is metres across and the clouds are kilometres up, so the
// whole scene is in one cloud's shadow or none: the sun's transmittance through the layers, toward
// the sun from the observer, is a single number, published with the solar disk. Nothing per pixel.
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

// ---- the layers, from the observer -------------------------------------------------------------

// takram's texture tilings (CloudsEffect.ts: localWeatherRepeat, shapeRepeat, shapeDetailRepeat,
// turbulenceRepeat), as periods in km. Their weather repeats 100 times across a cube-sphere face of
// the Earth, so about every 100 km.
static const float CLOUD_WEATHER_PERIOD    = 100.0F;
static const float CLOUD_TURBULENCE_PERIOD = 5.0F;
static const float CLOUD_SHAPE_PERIOD      = 1.0F / 0.3F;
static const float CLOUD_DETAIL_PERIOD     = 1.0F / 6.0F;

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

// The stretch of a ray inside the shell between radii rBottom and rTop: (enter, leave), with
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

// Whether a height falls in one of the gaps between layers, which the march steps over.
inline bool cloudInsideIntervals(SkyCloudsParameters p, float height)
{
  return (height > p.minIntervalHeights.x && height < p.maxIntervalHeights.x)
         || (height > p.minIntervalHeights.y && height < p.maxIntervalHeights.y)
         || (height > p.minIntervalHeights.z && height < p.maxIntervalHeights.z);
}

// ---- what is cloud (clouds.glsl) ------------------------------------------------------------------

inline float4 cloudSaturate4(float4 x)
{
  return clamp(x, float4(0.0F), float4(1.0F));
}

// Rounds the clouds toward their tops: a semicircle over each layer's height.
inline float4 cloudShapeAltering(float4 heightFraction, float4 bias)
{
  const float4 x = clamp(pow(heightFraction, bias) * 2.0F - 1.0F, float4(-1.0F), float4(1.0F));
  return float4(1.0F) - x * x;
}

struct CloudWeather
{
  float4 heightFraction;  // how far up each layer the point is
  float4 density;         // what the weather and the coverage allow in each
};

// The weather at a height, from the map's four channels there (`localWeather`, one per layer). The
// coverage lowers each layer's threshold most at the height its clouds are widest (Skybolt's
// modulation, as takram has it).
inline CloudWeather cloudSampleWeather(SkyCloudsParameters p, float4 localWeather, float height)
{
  CloudWeather w;
  w.heightFraction = cloudSaturate4((float4(height) - p.minLayerHeights) / max(p.maxLayerHeights - p.minLayerHeights, float4(1e-6F)));
  const float4 weather     = pow(localWeather, p.weatherExponents);
  const float4 heightScale = cloudShapeAltering(w.heightFraction, p.shapeAlteringBiases);
  const float4 factor      = float4(1.0F) - p.coverage * heightScale;
  w.density = cloudSaturate4((lerp(weather, float4(1.0F), p.coverageFilterWidths) - factor) / max(p.coverageFilterWidths, float4(1e-6F)));
  return w;
}

inline bool cloudWeatherHasDensity(SkyCloudsParameters p, CloudWeather w)
{
  return w.density.x > p.minDensity || w.density.y > p.minDensity || w.density.z > p.minDensity || w.density.w > p.minDensity;
}

// How far the turbulence displaces the shape: near the bottom of whichever layers have cloud there.
inline float cloudTurbulenceWeight(CloudWeather w)
{
  return dot(w.density, cloudSaturate4(float4(1.0F) - w.heightFraction / 0.3F));
}

struct CloudMedia
{
  float4 weight;      // each layer's share of the density
  float  scattering;  // 1/km
  float  extinction;  // 1/km
};

// The medium at a point from the weather there and the two volumes: the shape erodes the weather's
// density, the detail erodes its edges -- fluffy at the top, whippy at the bottom -- and each layer's
// profile and scale finish it. `withDetail` false leaves the detail out, as takram does at distance.
inline CloudMedia cloudSampleMedia(SkyCloudsParameters p, CloudWeather w, float shape, float detail, bool withDetail)
{
  const float4 floor0  = (1.0F - shape) * p.shapeAmounts;
  float4       density = cloudSaturate4((w.density - floor0) / max(float4(1.0F) - floor0, float4(1e-6F)));
  if(withDetail)
  {
    const float4 fluffy   = lerp(float4(pow(detail, 6.0F)), float4(1.0F - detail), cloudSaturate4((w.heightFraction - 0.2F) / 0.2F));
    const float4 modifier = fluffy * p.shapeDetailAmounts * 0.5F;
    density               = cloudSaturate4((density * 2.0F - modifier) / max(float4(1.0F) - modifier, float4(1e-6F)));
  }
  const float4 profile = p.profileExpTerms * exp(p.profileExponents * w.heightFraction) + p.profileLinearTerms * w.heightFraction
                         + p.profileConstantTerms;
  density = cloudSaturate4(density * p.densityScales * profile);

  CloudMedia  media;
  const float sum  = density.x + density.y + density.z + density.w;
  media.weight     = sum > 0.0F ? density / sum : float4(0.0F);
  media.scattering = sum * p.scattering;
  media.extinction = sum * p.absorption + media.scattering;
  return media;
}

// ---- how it is lit (clouds.frag) ----------------------------------------------------------------

// Two Henyey-Greenstein lobes, each normalised over the sphere.
inline float2 cloudHg2(float2 g, float cosTheta)
{
  const float2 g2 = g * g;
  return 0.0795774715F * ((float2(1.0F) - g2) / max(float2(1e-7F), pow(float2(1.0F) + g2 - 2.0F * g * cosTheta, float2(1.5F))));
}

// takram's droplets: a forward and a backward lobe, half each; `attenuation` flattens both, for
// the multiple-scattering octaves. `cosTheta` is between the way the light travels and the way it
// leaves, so 1 is straight on.
inline float cloudPhase(float cosTheta, float attenuation)
{
  return dot(cloudHg2(float2(0.7F, -0.2F) * attenuation, cosTheta), float2(0.5F, 0.5F));
}

// Wrenninge's octaves ("Oz: The Great and Volumetric", 2013): each a fainter, flatter scattering
// through less of the optical depth to the sun, which stands in for the light that has bounced
// many times. `octaves` of them; takram's eight.
inline float cloudMultipleScattering(float opticalDepth, float cosTheta, int octaves)
{
  float a          = 1.0F;  // its share
  float b          = 1.0F;  // how much of the optical depth it sees
  float c          = 1.0F;  // how anisotropic it stays
  float scattering = 0.0F;
  for(int i = 0; i < octaves; ++i)
  {
    scattering += a * exp(-opticalDepth * b) * cloudPhase(cosTheta, c);
    a *= 0.5F;
    b *= 0.5F;
    c *= 0.5F;
  }
  return scattering;
}

// ---- the types the panel offers ------------------------------------------------------------------

// Each a set of takram's layer parameters (cloudDefaultParameters) and its own cover table.
static const int CLOUD_TYPE_CUMULUS       = 0;  // takram's default: heaped cumulus and a thin high layer
static const int CLOUD_TYPE_STRATOCUMULUS = 1;  // a low deck of lumps and rolls, with gaps
static const int CLOUD_TYPE_STRATUS       = 2;  // a low, flat, grey sheet: overcast
static const int CLOUD_TYPE_COUNT         = 3;

inline int cloudTypeIndex(int type)
{
  return clamp(type, 0, CLOUD_TYPE_COUNT - 1);
}

// Each type's depth from base to tops before the panel's Thickness moves it, in km: takram's
// cumulus; stratocumulus and stratus as they are typically observed (a few hundred metres).
inline float cloudDefaultThickness(int type)
{
  const int t = cloudTypeIndex(type);
  return t == CLOUD_TYPE_STRATOCUMULUS ? 0.6F : t == CLOUD_TYPE_STRATUS ? 0.4F : 1.45F;
}

// ---- what the panel's cover means ------------------------------------------------------------------

// takram's coverage for a cover on the panel: the share of the land the clouds shade with the sun
// overhead, which is near enough the share of the sky overhead they hide, the way cloud cover is
// reported. Measured by marching columns through each type's default layers
// (SkyClouds.DISABLED_CoverageCurve in tests/test_sky_clouds.cpp prints these rows; re-fit them when
// the model changes) and inverted, a knot every 5%. Takram's own default of 0.3 shades about a
// quarter of the land under cumulus; by 0.65 the deck has closed. The first knot is where the first
// clouds appear rather than 0: a cover of exactly 0 is no clouds, and the bake does not run
// (SkyClouds::hasClouds). One row per type, in CLOUD_TYPE_* order.
static const float CLOUD_COVERAGE_FOR_COVER[CLOUD_TYPE_COUNT][21] = {
    {0.1000F, 0.2037F, 0.2385F, 0.2659F, 0.2880F, 0.3073F, 0.3247F, 0.3404F, 0.3560F, 0.3716F, 0.3869F,
     0.4020F, 0.4162F, 0.4306F, 0.4457F, 0.4613F, 0.4774F, 0.4954F, 0.5152F, 0.5409F, 0.6500F},
    {0.1146F, 0.1910F, 0.2172F, 0.2354F, 0.2498F, 0.2622F, 0.2732F, 0.2833F, 0.2929F, 0.3020F, 0.3109F,
     0.3197F, 0.3284F, 0.3374F, 0.3467F, 0.3569F, 0.3681F, 0.3811F, 0.3973F, 0.4211F, 0.5009F},
    {0.0459F, 0.0835F, 0.0970F, 0.1067F, 0.1145F, 0.1213F, 0.1273F, 0.1327F, 0.1378F, 0.1426F, 0.1473F,
     0.1520F, 0.1568F, 0.1617F, 0.1669F, 0.1726F, 0.1790F, 0.1867F, 0.1967F, 0.2125F, 0.2771F},
};

inline float cloudCoverageFor(float cover, int type)
{
  const int   t = cloudTypeIndex(type);
  const float x = clamp(cover, 0.0F, 1.0F) * 20.0F;
  const int   i = min(int(x), 19);
  return lerp(CLOUD_COVERAGE_FOR_COVER[t][i], CLOUD_COVERAGE_FOR_COVER[t][i + 1], x - float(i));
}

// takram's medium (uniforms.ts, CloudsMaterial.ts) and a type's layers, in km, for a cover and a
// depth. At takram's 'high' quality preset; cloudWithQuality lowers it for a preview.
//
// - Cumulus is takram's default (CloudLayers.DEFAULT): a low cumulus layer from 750 m and a deeper
//   one from 1 km, whose depths `thicknessKm` scales from their 1.45 km together, and a thin high
//   layer at 7.5 km.
// - Stratocumulus is one deck from 900 m, read through the two low weather channels at once, so the
//   fields of both make one broad deck: billows still cut it into lumps and rolls (shape amount 0.8),
//   with less of the detail's erosion than a cumulus and a softer edge to the weather's fields.
// - Stratus is one deck from 500 m, after takram's own example of a fog layer: the billows barely
//   mark it (shape amount 0.25), no detail, the widest edges to the weather's fields, density nearly
//   even with height, and little turbulence, so it reads as a flat grey sheet.
//
// The fourth layer stays off in every type.
static const float CLOUD_DEFAULT_THICKNESS = 1.45F;  // cumulus's (cloudDefaultThickness)

inline SkyCloudsParameters cloudDefaultParameters(float cover, float thicknessKm, int type)
{
  const int           t = cloudTypeIndex(type);
  const float         k = thicknessKm / cloudDefaultThickness(t);
  SkyCloudsParameters p;
  // takram's CloudLayer defaults, every layer off until the type turns it on
  p.minLayerHeights        = float4(0.0F);
  p.maxLayerHeights        = float4(0.0F);
  p.densityScales          = float4(0.0F);
  p.shapeAmounts           = float4(1.0F);
  p.shapeDetailAmounts     = float4(1.0F);
  p.weatherExponents       = float4(1.0F);
  p.shapeAlteringBiases    = float4(0.35F);
  p.coverageFilterWidths   = float4(0.6F);
  p.profileExpTerms        = float4(0.0F);
  p.profileExponents       = float4(0.0F);
  p.profileLinearTerms     = float4(0.75F);
  p.profileConstantTerms   = float4(0.25F);
  p.minIntervalHeights     = float3(0.0F);  // no gaps to step over
  p.maxIntervalHeights     = float3(0.0F);
  p.turbulenceDisplacement = 0.35F;  // 350 m

  if(t == CLOUD_TYPE_STRATOCUMULUS)
  {
    const float base = 0.9F, top = base + 0.6F * k;
    p.minLayerHeights        = float4(base, base, 0.0F, 0.0F);
    p.maxLayerHeights        = float4(top, top, 0.0F, 0.0F);
    p.densityScales          = float4(0.05F, 0.05F, 0.0F, 0.0F);
    p.shapeAmounts           = float4(0.8F, 0.8F, 1.0F, 1.0F);
    p.shapeDetailAmounts     = float4(0.5F, 0.5F, 1.0F, 1.0F);
    p.shapeAlteringBiases    = float4(0.5F);
    p.coverageFilterWidths   = float4(0.8F, 0.8F, 0.6F, 0.6F);
    p.profileLinearTerms     = float4(0.5F);
    p.profileConstantTerms   = float4(0.5F);
    p.turbulenceDisplacement = 0.15F;
    p.minHeight              = base;
    p.maxHeight              = top;
    p.shadowTopHeight        = top;
  }
  else if(t == CLOUD_TYPE_STRATUS)
  {
    const float base = 0.5F, top = base + 0.4F * k;
    p.minLayerHeights        = float4(base, base, 0.0F, 0.0F);
    p.maxLayerHeights        = float4(top, top, 0.0F, 0.0F);
    p.densityScales          = float4(0.06F, 0.06F, 0.0F, 0.0F);
    p.shapeAmounts           = float4(0.25F, 0.25F, 1.0F, 1.0F);
    p.shapeDetailAmounts     = float4(0.0F, 0.0F, 1.0F, 1.0F);
    p.shapeAlteringBiases    = float4(0.5F);
    p.coverageFilterWidths   = float4(0.9F, 0.9F, 0.6F, 0.6F);
    p.profileLinearTerms     = float4(0.3F);
    p.profileConstantTerms   = float4(0.7F);
    p.turbulenceDisplacement = 0.08F;
    p.minHeight              = base;
    p.maxHeight              = top;
    p.shadowTopHeight        = top;
  }
  else
  {
    p.minLayerHeights      = float4(0.75F, 1.0F, 7.5F, 0.0F);
    p.maxLayerHeights      = float4(0.75F + 0.65F * k, 1.0F + 1.2F * k, 8.0F, 0.0F);
    p.densityScales        = float4(0.2F, 0.2F, 0.003F, 0.0F);
    p.shapeAmounts         = float4(1.0F, 1.0F, 0.4F, 1.0F);
    p.shapeDetailAmounts   = float4(1.0F, 1.0F, 0.0F, 1.0F);
    p.coverageFilterWidths = float4(0.6F, 0.6F, 0.5F, 0.6F);
    // the one gap: from the cumulus tops to the high layer's base
    const float cumulusTop = max(p.maxLayerHeights.x, p.maxLayerHeights.y);
    p.minIntervalHeights   = float3(cumulusTop, 0.0F, 0.0F);
    p.maxIntervalHeights   = float3(max(cumulusTop, p.minLayerHeights.z), 0.0F, 0.0F);
    p.minHeight            = 0.75F;
    p.maxHeight            = p.maxLayerHeights.z;
    p.shadowTopHeight      = cumulusTop;
  }

  p.coverage          = cloudCoverageFor(cover, t);
  p.weatherOffset     = float2(0.0F);
  p.shapeOffset       = float3(0.0F);
  p.scattering        = 1000.0F;  // 1/m
  p.absorption        = 0.0F;
  p.skyLightScale     = 1.0F;
  p.groundBounceScale = 1.0F;
  p.powderScale       = 0.8F;
  p.powderExponent    = 0.15F;  // 150 m
  // Takram averages the march's jitter over frames; a bake is one frame, so it averages it over
  // marches per texel instead. One leaves the jitter's crosshatch frozen in the image.
  p.samples = 8;
  p._pad1   = 0.0F;

  p.maxIterations        = 500;
  p.minStepSize          = 0.05F;
  p.maxStepSize          = 1.0F;
  p.maxRayDistance       = 200.0F;
  p.perspectiveStepScale = 1.01F;
  p.minDensity           = 1e-5F;
  p.minExtinction        = 1e-2F;  // 1e-5 /m
  p.minTransmittance     = 1e-2F;
  p.sunIterations        = 2;
  p.groundIterations     = 3;
  p.minSecondaryStepSize = 0.1F;
  p.secondaryStepScale   = 2.0F;
  p.farSunIterations     = 5;
  p.octaves              = 8;
  p.shapeDetail          = 1;
  p.turbulence           = 1;
  return p;
}

// A drag's preview at takram's 'low' preset: shorter and coarser marches, no detail, turbulence or
// ground bounce. The octaves stay, since fewer darken the clouds (takram's own note).
inline SkyCloudsParameters cloudWithQuality(SkyCloudsParameters p, bool preview)
{
  if(preview)
  {
    p.maxIterations    = 200;
    p.minStepSize      = 0.1F;
    p.maxRayDistance   = 100.0F;
    p.minDensity       = 1e-4F;
    p.minExtinction    = 1e-1F;
    p.minTransmittance = 1e-1F;
    p.groundIterations = 0;
    p.sunIterations    = 1;
    p.farSunIterations = 3;
    p.shapeDetail      = 0;
    p.turbulence       = 0;
    p.samples          = 1;  // a drag's frames move on before the dither would show
  }
  return p;
}

// The share of the land under the layers, seen from afar, that the sun still reaches: what the
// distant ground and the haze in front of the clouds are lit by. The cover is the share shaded, so
// this is what is left of it.
inline float cloudSunlitAir(float cover)
{
  return 1.0F - clamp(cover, 0.0F, 1.0F);
}

// ---- the image ------------------------------------------------------------------------------------

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
