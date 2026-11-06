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
// Host/device interface of the physical sky's clouds (src/sky_clouds.*): the noise they are cut from,
// the pass that bakes them into the cloud image, and the image's set that the bake and the path
// tracer read it through. What the clouds are is shaders/sky_clouds.h.slang.
//

#ifndef SKY_CLOUDS_IO_H
#define SKY_CLOUDS_IO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// The noise, generated once (sky_clouds_noise.slang), each tiling, at takram's sizes.
#define CLOUD_SHAPE_NOISE_SIZE 128  // 3D R8, mipmapped: the billows, Perlin-Worley
#define CLOUD_DETAIL_NOISE_SIZE 32  // 3D R8, mipmapped: the eddies the edges are eroded by, Worley
#define CLOUD_WEATHER_SIZE 512      // 2D RGBA8, mipmapped: where each layer may have cloud
#define CLOUD_TURBULENCE_SIZE 128   // 2D RGBA8: a curl field that shreds the cloud bottoms

#define CLOUD_WORKGROUP_SIZE 8  // 8x8 for the 2D passes, 4x4x4 for the volumes

// The cloud pass's own set (sky_clouds_bake.slang). Set 1 of that pass is SkyBruneton's runtime set.
enum SkyCloudsBakeBindings
{
  eCloudBakeOut        = 0,  // RWTexture2D<float4>: what the march writes -- the raw image, or the preview
  eCloudBakeShape      = 1,  // Sampler3D<float>, repeating, mipmapped
  eCloudBakeDetail     = 2,  // Sampler3D<float>, repeating, mipmapped
  eCloudBakeWeather    = 3,  // Sampler2D<float4>, repeating, mipmapped
  eCloudBakeTurbulence = 4,  // Sampler2D<float4>, repeating
  eCloudBakeParams     = 5,  // ConstantBuffer<SkyCloudsParameters>
  eCloudBakeFiltered   = 6,  // RWTexture2D<float4>: mip 0 of the cloud image, the raw one filtered (filterMain)
};

// The noise pass's set: what it writes, once.
enum SkyCloudsNoiseBindings
{
  eCloudNoiseShape      = 0,  // RWTexture3D<float>
  eCloudNoiseDetail     = 1,  // RWTexture3D<float>
  eCloudNoiseWeather    = 2,  // RWTexture2D<float4>, mip 0
  eCloudNoiseTurbulence = 3,  // RWTexture2D<float4>
};

// The set the cloud image is *read* through, by the environment bake (its set 2) and the path
// tracer (its set 4): Sampler2D<float4>s, azimuth repeating. The full image, mipmapped, for a
// settled sky; a quarter-size one for a drag's preview (SkySunDisk::cloudPreview says which holds
// the clouds of the moment).
#define CLOUD_IMAGE_BINDING 0
#define CLOUD_PREVIEW_BINDING 1

// The layers and the medium, as takram's CloudsEffect uniforms (packages/clouds/src/uniforms.ts),
// in the atmosphere's kilometres. Four layers, one per channel of the weather map; a layer with no
// height and no density is off. Built by SkyClouds from the panel's tiers -- low, mid and high, each
// its own layers and cover (cloudSkyParameters) -- and the arrangement.
struct SkyCloudsParameters
{
  float4 minLayerHeights;  // km above the surface: each layer's base
  float4 maxLayerHeights;  // ... and its top
  float4 densityScales;
  float4 shapeAmounts;
  float4 shapeDetailAmounts;
  float4 weatherExponents;
  float4 shapeAlteringBiases;
  float4 coverageFilterWidths;
  // Density across each layer's height: expTerm * exp(exponent * h) + linearTerm * h + constantTerm
  float4 profileExpTerms;
  float4 profileExponents;
  float4 profileLinearTerms;
  float4 profileConstantTerms;
  // Takram has one coverage and one noise scale for every layer; the tiers each have their own
  float4 coverages;                // takram's coverage, 0..1 (cloudCoverageFor: from the tier's cover)
  float4 shapeScales;              // the billows' and the detail's frequency against takram's (2: half the size)
  float4 shapeStretches;           // ... along x against the other two: below 1 draws them out into streaks
  float4 powderScales;             // Beer-Powder's darkening of thin cloud, takram's powderScale per layer
  float4 turbulenceDisplacements;  // km the curl field moves the shape at each layer's bottom

  float3 minIntervalHeights;  // the gaps between layers, which the march steps over
  float  _pad0;
  float3 maxIntervalHeights;
  float  minHeight;  // the lowest base and the highest top of all the layers, km
  float2 weatherOffset;  // the arrangement: where in the weather map's tiling the observer stands
  float  maxHeight;
  float  shadowTopHeight;  // the top of the layers that shade the ground (their ground bounce stops above)

  float3 shapeOffset;
  float  scattering;  // 1/km at density 1
  float  absorption;  // 1/km at density 1
  float  skyLightScale;
  float  groundBounceScale;
  float  _pad3;
  float  powderExponent;  // km
  float  _pad1;
  int    samples;  // marches per texel, each at another jitter and spot in it, averaged
  float  _pad2;

  // The march: takram's quality preset, in kilometres
  int   maxIterations;
  float minStepSize;
  float maxStepSize;
  float maxRayDistance;
  float perspectiveStepScale;
  float minDensity;
  float minExtinction;  // 1/km
  float minTransmittance;
  int   sunIterations;     // the short march toward the sun, for the edges
  int   groundIterations;  // toward the ground, for the bounce; 0 skips it
  float minSecondaryStepSize;
  float secondaryStepScale;
  int   farSunIterations;  // the long march toward the sun (what takram's shadow map supplies)
  int   octaves;           // multiple-scattering octaves
  int   shapeDetail;       // 0 skips the detail noise
  int   turbulence;        // 0 skips the turbulence
};

// What changes per bake: where the sun and the observer are, and which image it fills.
struct SkyCloudsPushConstant
{
  uint2  imageSize;         // the image written: the full one, or the preview
  float  observerAltitude;  // km above the surface, the bake's fixed observer
  float  sunlitAir;         // share of the air under the layers the sun reaches (cloudSunlitAir)
  float3 sunDirection;      // unit, the environment's frame, toward the sun
  float  texelAngle;        // radians a column of the image spans: its footprint at a distance
};

NAMESPACE_SHADERIO_END()

#endif  // SKY_CLOUDS_IO_H
