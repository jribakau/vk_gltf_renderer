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
// Host/device interface of the physical sky's cloud layer (src/sky_clouds.*): the noise volumes it
// is built from, the pass that bakes it into the cloud image, and the image's set that the bake and
// the path tracer read it through. What the clouds are, and how the image is laid out, is
// shaders/sky_clouds.h.slang.
//

#ifndef SKY_CLOUDS_IO_H
#define SKY_CLOUDS_IO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// The noise the clouds are cut from, generated once (sky_clouds_noise.slang). Each tiles, so a
// sampler set to repeat walks it forever; the sizes are its texels per period.
#define CLOUD_SHAPE_NOISE_SIZE 128  // 3D: the billows, Perlin-Worley
#define CLOUD_DETAIL_NOISE_SIZE 64  // 3D: the edges they are eroded by, Worley
#define CLOUD_WEATHER_SIZE 512      // 2D: where over the land there is cloud, Perlin

#define CLOUD_WORKGROUP_SIZE 8      // 8x8 for the image and the weather map, 4x4x4 for the volumes

// The cloud pass's own set (sky_clouds_bake.slang). Set 1 of that pass is SkyBruneton's runtime set.
enum SkyCloudsBakeBindings
{
  eCloudBakeOut     = 0,  // RWTexture2D<float4>: the cloud image (sky_clouds.h.slang says what it holds)
  eCloudBakeShape   = 1,  // Sampler3D<float>: CLOUD_SHAPE_NOISE_SIZE^3, repeating
  eCloudBakeDetail  = 2,  // Sampler3D<float>: CLOUD_DETAIL_NOISE_SIZE^3, repeating
  eCloudBakeWeather = 3,  // Sampler2D<float>: CLOUD_WEATHER_SIZE^2, repeating
};

// The noise pass's set: the three volumes it writes, once.
enum SkyCloudsNoiseBindings
{
  eCloudNoiseShape   = 0,  // RWTexture3D<float>
  eCloudNoiseDetail  = 1,  // RWTexture3D<float>
  eCloudNoiseWeather = 2,  // RWTexture2D<float>
};

// The set the cloud image is *read* through, by the environment bake (its set 2) and the path
// tracer (its set 4): binding 0, Sampler2D<float4>, azimuth repeating.
#define CLOUD_IMAGE_BINDING 0

// What one bake of the cloud image needs. The layer, in the atmosphere's kilometres, and where the
// sun and the observer are -- the same two the sky's bake takes.
struct SkyCloudsPushConstant
{
  uint2  imageSize;
  float  level;      // the weather level for the cover (cloudLevel); 0 for no clouds
  float  bottom;     // km above the surface: the cloud base
  float3 sunDirection;  // unit, the environment's frame, toward the sun
  float  thickness;     // km from the base to the highest tops
  float2 offset;        // km: which stretch of the endless noise is overhead (the arrangement)
  float  observerAltitude;  // km above the surface, the bake's fixed observer
  float  sunlitAir;         // share of the air under the layer the sun reaches (cloudSunlitAir)
  int    viewSteps;         // samples along a view ray through the layer
  int    lightSteps;        // samples toward the sun from each of them
  int    _pad0;
  int    _pad1;
};

NAMESPACE_SHADERIO_END()

#endif  // SKY_CLOUDS_IO_H
