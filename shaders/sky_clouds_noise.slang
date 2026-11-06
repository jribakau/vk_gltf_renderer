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
// Writes the clouds' four noise textures, once, when SkyClouds starts (src/sky_clouds.cpp). One
// entry point per texture; what each holds is sky_clouds_noise.h.slang, ported from takram's.
//

#include "sky_clouds_io.h.slang"
#include "sky_clouds_noise.h.slang"

// clang-format off
[[vk::binding(SkyCloudsNoiseBindings::eCloudNoiseShape, 0)]]      [[vk::image_format("r8")]]    RWTexture3D<float>  g_shape;
[[vk::binding(SkyCloudsNoiseBindings::eCloudNoiseDetail, 0)]]     [[vk::image_format("r8")]]    RWTexture3D<float>  g_detail;
[[vk::binding(SkyCloudsNoiseBindings::eCloudNoiseWeather, 0)]]    [[vk::image_format("rgba8")]] RWTexture2D<float4> g_weather;
[[vk::binding(SkyCloudsNoiseBindings::eCloudNoiseTurbulence, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> g_turbulence;
// clang-format on

[shader("compute")]
[numthreads(4, 4, 4)]
void shapeMain(uint3 id: SV_DispatchThreadID)
{
  if(any(id >= uint3(CLOUD_SHAPE_NOISE_SIZE)))
    return;
  g_shape[id] = cloudShapeNoise((float3(id) + 0.5F) / float(CLOUD_SHAPE_NOISE_SIZE));
}

[shader("compute")]
[numthreads(4, 4, 4)]
void detailMain(uint3 id: SV_DispatchThreadID)
{
  if(any(id >= uint3(CLOUD_DETAIL_NOISE_SIZE)))
    return;
  g_detail[id] = cloudDetailNoise((float3(id) + 0.5F) / float(CLOUD_DETAIL_NOISE_SIZE));
}

[shader("compute")]
[numthreads(CLOUD_WORKGROUP_SIZE, CLOUD_WORKGROUP_SIZE, 1)]
void weatherMain(uint3 id: SV_DispatchThreadID)
{
  if(any(id.xy >= uint2(CLOUD_WEATHER_SIZE)))
    return;
  g_weather[id.xy] = cloudWeatherNoise((float2(id.xy) + 0.5F) / float(CLOUD_WEATHER_SIZE));
}

[shader("compute")]
[numthreads(CLOUD_WORKGROUP_SIZE, CLOUD_WORKGROUP_SIZE, 1)]
void turbulenceMain(uint3 id: SV_DispatchThreadID)
{
  if(any(id.xy >= uint2(CLOUD_TURBULENCE_SIZE)))
    return;
  g_turbulence[id.xy] = float4(cloudTurbulenceNoise((float2(id.xy) + 0.5F) / float(CLOUD_TURBULENCE_SIZE)), 1.0F);
}
