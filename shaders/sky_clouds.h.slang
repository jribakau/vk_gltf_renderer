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
  const float4 factor      = float4(1.0F) - p.coverages * heightScale;
  w.density = cloudSaturate4((lerp(weather, float4(1.0F), p.coverageFilterWidths) - factor) / max(p.coverageFilterWidths, float4(1e-6F)));
  return w;
}

inline bool cloudWeatherHasDensity(SkyCloudsParameters p, CloudWeather w)
{
  return w.density.x > p.minDensity || w.density.y > p.minDensity || w.density.z > p.minDensity || w.density.w > p.minDensity;
}

// How far the turbulence displaces the shape, in km: near the bottom of whichever layers have cloud
// there, each by its own displacement (takram's one, turbulenceDisplacement, per layer).
inline float cloudTurbulenceDisplacement(SkyCloudsParameters p, CloudWeather w)
{
  return dot(w.density * cloudSaturate4(float4(1.0F) - w.heightFraction / 0.3F), p.turbulenceDisplacements);
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
// `shape` and `detail` are each layer's, read at its own scale (shapeScales).
inline CloudMedia cloudSampleMedia(SkyCloudsParameters p, CloudWeather w, float4 shape, float4 detail, bool withDetail)
{
  const float4 floor0  = (float4(1.0F) - shape) * p.shapeAmounts;
  float4       density = cloudSaturate4((w.density - floor0) / max(float4(1.0F) - floor0, float4(1e-6F)));
  if(withDetail)
  {
    const float4 fluffy   = lerp(pow(detail, float4(6.0F)), float4(1.0F) - detail, cloudSaturate4((w.heightFraction - 0.2F) / 0.2F));
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

// ---- the kinds of cloud, in three tiers -----------------------------------------------------------

// The sky is three tiers, as cloud cover is reported -- low, mid and high -- each with its own cover,
// so they lie over one another as real ones do: cumulus under a mackerel sky under cirrus. The low
// tier is one of three kinds; the mid tier is altocumulus and the high tier cirrus. Each kind is a set
// of takram's layer parameters and its own cover table, and owns its layers of the four:
//
//     layers 0 and 1 (the weather map's r, g)  the low tier
//     layer 2 (b, takram's high streaks)        the high tier, cirrus
//     layer 3 (a, ours)                         the mid tier, altocumulus
static const int CLOUD_TYPE_CUMULUS       = 0;  // low: heaped clouds, flat bases and billowing tops
static const int CLOUD_TYPE_STRATOCUMULUS = 1;  // low: a deck of lumps and rolls, with gaps
static const int CLOUD_TYPE_STRATUS       = 2;  // low: a flat, grey sheet, overcast
static const int CLOUD_TYPE_ALTOCUMULUS   = 3;  // mid: a deck of small rounded cloudlets
static const int CLOUD_TYPE_CIRRUS        = 4;  // high: thin streaks of ice
static const int CLOUD_TYPE_COUNT         = 5;
static const int CLOUD_LOW_TYPE_COUNT     = 3;  // the low tier's kinds, the first three

inline int cloudTypeIndex(int type)
{
  return clamp(type, 0, CLOUD_TYPE_COUNT - 1);
}

// Each kind's depth from base to tops before the panel's Thickness moves it (the low tier's only),
// in km: takram's cumulus and high layer; the others as they are typically observed.
inline float cloudDefaultThickness(int type)
{
  const int t = cloudTypeIndex(type);
  if(t == CLOUD_TYPE_STRATOCUMULUS)
    return 0.6F;
  if(t == CLOUD_TYPE_STRATUS || t == CLOUD_TYPE_ALTOCUMULUS)
    return 0.4F;
  if(t == CLOUD_TYPE_CIRRUS)
    return 0.5F;
  return 1.45F;
}

// ---- what the panel's covers mean ------------------------------------------------------------------

// takram's coverage for a tier's cover on the panel. For the low and mid tiers, the share of the land
// the tier shades with the sun overhead, which is near enough the share of the sky overhead it hides,
// the way cloud cover is reported. Cirrus hardly shades -- a veil the sun shines through -- so for it
// the cover is the share of the sky it veils at all (an optical depth of 0.05 or more), and it shades
// CLOUD_CIRRUS_SHADE of what it covers. Measured by marching columns through each kind's layers alone
// (SkyClouds.DISABLED_CoverageCurve in tests/test_sky_clouds.cpp prints these rows; re-fit them when
// the model changes) and inverted, a knot every 5%. Takram's own default of 0.3 shades about a
// quarter of the land under cumulus; by 0.6 the deck has closed. The first knot is where the first
// clouds appear rather than 0: a cover of exactly 0 is no tier at all, and with no tier the bake does
// not run (SkyClouds::hasClouds). One row per kind, in CLOUD_TYPE_* order.
static const float CLOUD_COVERAGE_FOR_COVER[CLOUD_TYPE_COUNT][21] = {
    {0.1157F, 0.2138F, 0.2518F, 0.2800F, 0.3028F, 0.3221F, 0.3398F, 0.3562F, 0.3729F, 0.3888F, 0.4042F,
     0.4182F, 0.4318F, 0.4463F, 0.4606F, 0.4752F, 0.4906F, 0.5064F, 0.5235F, 0.5467F, 0.6103F},
    {0.1146F, 0.1910F, 0.2172F, 0.2354F, 0.2498F, 0.2622F, 0.2732F, 0.2833F, 0.2929F, 0.3020F, 0.3109F,
     0.3197F, 0.3284F, 0.3374F, 0.3467F, 0.3569F, 0.3681F, 0.3811F, 0.3973F, 0.4211F, 0.5009F},
    {0.0459F, 0.0835F, 0.0970F, 0.1067F, 0.1145F, 0.1213F, 0.1273F, 0.1327F, 0.1378F, 0.1426F, 0.1473F,
     0.1520F, 0.1568F, 0.1617F, 0.1669F, 0.1726F, 0.1790F, 0.1867F, 0.1967F, 0.2125F, 0.2771F},
    {0.1788F, 0.3070F, 0.3550F, 0.3898F, 0.4192F, 0.4447F, 0.4675F, 0.4889F, 0.5091F, 0.5283F, 0.5468F,
     0.5650F, 0.5834F, 0.6022F, 0.6219F, 0.6427F, 0.6654F, 0.6916F, 0.7240F, 0.7701F, 0.9550F},
    {0.1275F, 0.2306F, 0.2801F, 0.3050F, 0.3278F, 0.3490F, 0.3683F, 0.3854F, 0.4038F, 0.4198F, 0.4353F,
     0.4536F, 0.4688F, 0.4857F, 0.5040F, 0.5230F, 0.5443F, 0.5655F, 0.5918F, 0.6275F, 0.7450F},
};

// The share of what cirrus covers that it shades, on average (DISABLED_CoverageCurve prints it).
static const float CLOUD_CIRRUS_SHADE = 0.70F;

inline float cloudCoverageFor(float cover, int type)
{
  const int   t = cloudTypeIndex(type);
  const float x = clamp(cover, 0.0F, 1.0F) * 20.0F;
  const int   i = min(int(x), 19);
  return lerp(CLOUD_COVERAGE_FOR_COVER[t][i], CLOUD_COVERAGE_FOR_COVER[t][i + 1], x - float(i));
}

// ---- the layers -------------------------------------------------------------------------------------

// takram's medium (uniforms.ts, CloudsMaterial.ts) and march, every layer off. At takram's 'high'
// quality preset; cloudWithQuality lowers it for a preview.
inline SkyCloudsParameters cloudNoLayers()
{
  SkyCloudsParameters p;
  // takram's CloudLayer defaults, every layer off until a tier turns it on
  p.minLayerHeights         = float4(0.0F);
  p.maxLayerHeights         = float4(0.0F);
  p.densityScales           = float4(0.0F);
  p.shapeAmounts            = float4(1.0F);
  p.shapeDetailAmounts      = float4(1.0F);
  p.weatherExponents        = float4(1.0F);
  p.shapeAlteringBiases     = float4(0.35F);
  p.coverageFilterWidths    = float4(0.6F);
  p.profileExpTerms         = float4(0.0F);
  p.profileExponents        = float4(0.0F);
  p.profileLinearTerms      = float4(0.75F);
  p.profileConstantTerms    = float4(0.25F);
  p.coverages               = float4(0.0F);
  p.shapeScales             = float4(1.0F);  // takram's
  p.shapeStretches          = float4(1.0F);
  p.powderScales            = float4(0.8F);  // takram's
  p.turbulenceDisplacements = float4(0.0F);
  p.minIntervalHeights      = float3(0.0F);
  p._pad0                   = 0.0F;
  p.maxIntervalHeights      = float3(0.0F);
  p.minHeight               = 0.0F;
  p.weatherOffset           = float2(0.0F);
  p.maxHeight               = 0.0F;
  p.shadowTopHeight         = 0.0F;

  p.shapeOffset       = float3(0.0F);
  p.scattering        = 1000.0F;  // 1/m
  p.absorption        = 0.0F;
  p.skyLightScale     = 1.0F;
  p.groundBounceScale = 1.0F;
  p._pad3             = 0.0F;
  p.powderExponent    = 0.15F;  // 150 m
  p._pad1             = 0.0F;
  // Takram averages the march's jitter over frames; a bake is one frame, so it averages it over
  // marches per texel instead. One leaves the jitter's crosshatch frozen in the image.
  p.samples = 8;
  p._pad2   = 0.0F;

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

// The low tier in layers 0 and 1, for its kind, cover and depth.
//
// - Cumulus is takram's default (CloudLayers.DEFAULT) less its high layer, which is the high tier: a
//   low cumulus layer from 750 m and a deeper one in larger fields from 1 km, whose depths
//   `thicknessKm` scales from their 1.45 km together.
// - Stratocumulus is one deck from 900 m, read through both low weather channels at once, so the
//   fields of both make one broad deck: billows still cut it into lumps and rolls (shape amount 0.8),
//   with less of the detail's erosion than a cumulus and a softer edge to the weather's fields.
// - Stratus is one deck from 500 m, after takram's own example of a fog layer: the billows barely
//   mark it (shape amount 0.25), no detail, the widest edges to the weather's fields, density nearly
//   even with height, and little turbulence, so it reads as a flat grey sheet.
inline SkyCloudsParameters cloudLowTier(SkyCloudsParameters p, float cover, float thicknessKm, int type)
{
  const int   t        = clamp(type, 0, CLOUD_LOW_TYPE_COUNT - 1);
  const float k        = thicknessKm / cloudDefaultThickness(t);
  const float coverage = cloudCoverageFor(cover, t);
  p.coverages.x        = coverage;
  p.coverages.y        = coverage;
  if(t == CLOUD_TYPE_STRATOCUMULUS)
  {
    const float base = 0.9F, top = base + 0.6F * k;
    p.minLayerHeights.x = base;
    p.minLayerHeights.y = base;
    p.maxLayerHeights.x = top;
    p.maxLayerHeights.y = top;
    p.densityScales.x = 0.05F;
    p.densityScales.y = 0.05F;
    p.shapeAmounts.x = 0.8F;
    p.shapeAmounts.y = 0.8F;
    p.shapeDetailAmounts.x = 0.5F;
    p.shapeDetailAmounts.y = 0.5F;
    p.shapeAlteringBiases.x = 0.5F;
    p.shapeAlteringBiases.y = 0.5F;
    p.coverageFilterWidths.x = 0.8F;
    p.coverageFilterWidths.y = 0.8F;
    p.profileLinearTerms.x = 0.5F;
    p.profileLinearTerms.y = 0.5F;
    p.profileConstantTerms.x = 0.5F;
    p.profileConstantTerms.y = 0.5F;
    p.turbulenceDisplacements.x = 0.15F;
    p.turbulenceDisplacements.y = 0.15F;
  }
  else if(t == CLOUD_TYPE_STRATUS)
  {
    const float base = 0.5F, top = base + 0.4F * k;
    p.minLayerHeights.x = base;
    p.minLayerHeights.y = base;
    p.maxLayerHeights.x = top;
    p.maxLayerHeights.y = top;
    p.densityScales.x = 0.06F;
    p.densityScales.y = 0.06F;
    p.shapeAmounts.x = 0.25F;
    p.shapeAmounts.y = 0.25F;
    p.shapeDetailAmounts.x = 0.0F;
    p.shapeDetailAmounts.y = 0.0F;
    p.shapeAlteringBiases.x = 0.5F;
    p.shapeAlteringBiases.y = 0.5F;
    p.coverageFilterWidths.x = 0.9F;
    p.coverageFilterWidths.y = 0.9F;
    p.profileLinearTerms.x = 0.3F;
    p.profileLinearTerms.y = 0.3F;
    p.profileConstantTerms.x = 0.7F;
    p.profileConstantTerms.y = 0.7F;
    p.turbulenceDisplacements.x = 0.08F;
    p.turbulenceDisplacements.y = 0.08F;
  }
  else
  {
    p.minLayerHeights.x = 0.75F;
    p.minLayerHeights.y = 1.0F;
    p.maxLayerHeights.x = 0.75F + 0.65F * k;
    p.maxLayerHeights.y = 1.0F + 1.2F * k;
    p.densityScales.x = 0.2F;
    p.densityScales.y = 0.2F;
    p.turbulenceDisplacements.x = 0.35F;  // 350 m
    p.turbulenceDisplacements.y = 0.35F;
  }
  return p;
}

// The mid tier in layer 3, altocumulus: one deck from 3 km, 400 m deep, in the weather map's own
// channel for it. Its cloudlets are a few hundred metres across, a degree or a few as seen from
// below, where takram's billows are a kilometre: the noise is read at twice the frequency
// (shapeScales), and it is thinner than the low decks. A shape amount of 1 cuts takram's clouds only
// where the weather allows less than all, at a field's edge, which inside a field leaves a solid
// deck; at 2 the billows cut it everywhere at a threshold the cover moves, into dense rounded
// cloudlets with sky between, fields of them where the weather has patches.
inline SkyCloudsParameters cloudMidTier(SkyCloudsParameters p, float cover)
{
  const float base = 3.0F, top = base + cloudDefaultThickness(CLOUD_TYPE_ALTOCUMULUS);
  p.coverages.w               = cloudCoverageFor(cover, CLOUD_TYPE_ALTOCUMULUS);
  p.minLayerHeights.w         = base;
  p.maxLayerHeights.w         = top;
  p.densityScales.w           = 0.06F;
  p.shapeAmounts.w            = 2.0F;
  p.shapeDetailAmounts.w      = 0.4F;
  p.shapeAlteringBiases.w     = 0.5F;
  p.shapeScales.w             = 2.0F;
  p.turbulenceDisplacements.w = 0.1F;
  return p;
}

// The high tier in layer 2, cirrus: a thin sheet at 7.5 km, takram's high layer, in streaks where
// its channel of the weather map has them. Takram's billows barely mark it, which leaves soft smears;
// cirrus is fibres, so here the billows are drawn out sixfold along one way (shapeStretches) and cut
// the sheet hard (a shape amount past 1, as altocumulus's) into long thin strands. No Beer-Powder:
// it darkens a dense cloud's thin edges, where light has yet to build up by scattering, and a veil
// of ice is all thin edge, which it greys.
inline SkyCloudsParameters cloudHighTier(SkyCloudsParameters p, float cover)
{
  const float base = 7.5F, top = base + cloudDefaultThickness(CLOUD_TYPE_CIRRUS);
  p.coverages.z            = cloudCoverageFor(cover, CLOUD_TYPE_CIRRUS);
  p.minLayerHeights.z      = base;
  p.maxLayerHeights.z      = top;
  p.densityScales.z        = 0.012F;
  p.shapeAmounts.z         = 1.6F;
  p.shapeDetailAmounts.z   = 0.0F;
  p.coverageFilterWidths.z = 0.5F;
  p.shapeStretches.z       = 1.0F / 6.0F;
  p.powderScales.z         = 0.0F;
  return p;
}

// What the layers the tiers turned on add up to, for the march: the lowest base and highest top it
// marches between, the gaps between them it steps over (up to three, between the merged spans of the
// layers), and the top of the lowest span, whose bases the ground's light reaches.
inline SkyCloudsParameters cloudFinishLayers(SkyCloudsParameters p)
{
  // the spans of the layers that are on, in order of their bases (four at most: an insertion sort)
  float lo[4], hi[4];
  int   n = 0;
  for(int i = 0; i < 4; ++i)
  {
    if(p.densityScales[i] <= 0.0F || p.maxLayerHeights[i] <= p.minLayerHeights[i])
      continue;
    int j = n++;
    for(; j > 0 && lo[j - 1] > p.minLayerHeights[i]; --j)
    {
      lo[j] = lo[j - 1];
      hi[j] = hi[j - 1];
    }
    lo[j] = p.minLayerHeights[i];
    hi[j] = p.maxLayerHeights[i];
  }
  p.minIntervalHeights = float3(0.0F);
  p.maxIntervalHeights = float3(0.0F);
  if(n == 0)
  {
    p.minHeight = p.maxHeight = p.shadowTopHeight = 0.0F;
    return p;
  }
  // merged into spans; the gap before each span after the first is one to step over
  float spanTop = hi[0];
  int   gaps    = 0;
  p.minHeight       = lo[0];
  p.shadowTopHeight = -1.0F;
  for(int i = 1; i < n; ++i)
  {
    if(lo[i] <= spanTop)
    {
      spanTop = max(spanTop, hi[i]);
      continue;
    }
    if(p.shadowTopHeight < 0.0F)
      p.shadowTopHeight = spanTop;
    p.minIntervalHeights[gaps] = spanTop;
    p.maxIntervalHeights[gaps] = lo[i];
    ++gaps;
    spanTop = hi[i];
  }
  if(p.shadowTopHeight < 0.0F)
    p.shadowTopHeight = spanTop;
  p.maxHeight = spanTop;
  return p;
}

// One kind's tier alone, at a cover and (the low tier's) depth: what the cover tables are fitted on.
inline SkyCloudsParameters cloudDefaultParameters(float cover, float thicknessKm, int type)
{
  const int           t = cloudTypeIndex(type);
  SkyCloudsParameters p = cloudNoLayers();
  if(t == CLOUD_TYPE_ALTOCUMULUS)
    p = cloudMidTier(p, cover);
  else if(t == CLOUD_TYPE_CIRRUS)
    p = cloudHighTier(p, cover);
  else
    p = cloudLowTier(p, cover, thicknessKm, t);
  return cloudFinishLayers(p);
}

// The sky the panel sets: each tier whose cover is above 0, over one another.
inline SkyCloudsParameters cloudSkyParameters(float lowCover, float lowThicknessKm, int lowType, float midCover, float highCover)
{
  SkyCloudsParameters p = cloudNoLayers();
  if(lowCover > 0.0F)
    p = cloudLowTier(p, lowCover, lowThicknessKm, lowType);
  if(midCover > 0.0F)
    p = cloudMidTier(p, midCover);
  if(highCover > 0.0F)
    p = cloudHighTier(p, highCover);
  return cloudFinishLayers(p);
}

static const float CLOUD_DEFAULT_THICKNESS = 1.45F;  // cumulus's (cloudDefaultThickness)

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

// The same under the three tiers, each shading its own share wherever the others leave the sun, as
// if they lay over one another independently; cirrus shades only CLOUD_CIRRUS_SHADE of its cover.
inline float cloudSunlitAirOfTiers(float lowCover, float midCover, float highCover)
{
  return cloudSunlitAir(lowCover) * cloudSunlitAir(midCover) * cloudSunlitAir(CLOUD_CIRRUS_SHADE * highCover);
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
