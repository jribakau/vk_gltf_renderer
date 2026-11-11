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

// The still's per-pixel Keep detail (StillKeep in src/still_keep.hpp, still_keep.h.slang): over the
// denoised image, each pixel the accumulation has converged takes back its raw value. The noise is
// averaged over the pixel and its eight neighbours, so the share kept changes smoothly and a lone
// pixel does not flip between the two images. Each pixel writes only itself and reads only its own
// denoised value, so the output is the denoised image.

#include "shaderio.h"
#include "adaptive_stop.h.slang"
#include "still_keep.h.slang"

// clang-format off
[[vk::binding(StillKeepBindings::eKeepDenoised)]] RWTexture2D<float4> u_denoised;
[[vk::binding(StillKeepBindings::eKeepRaw)]]      RWTexture2D<float4> u_raw;
[[vk::binding(StillKeepBindings::eKeepMoments)]]  RWTexture2D<float4> u_moments;

[[vk::push_constant]] ConstantBuffer<StillKeepPushConstant> pushConst;
// clang-format on

float luminanceOf(float3 c)
{
  return dot(c, float3(0.2126F, 0.7152F, 0.0722F));
}

// A pixel's noise squared (stillKeepNoise): its samples the stop's count where it kept one, else the still's
float noiseAt(int2 pixel)
{
  const float2 moments = u_moments[pixel].xy;
  const int    samples = moments.y != 0.0F ? adaptiveStateSamples(moments.y) : pushConst.samples;
  return stillKeepNoise(moments.x, luminanceOf(u_raw[pixel].xyz), samples);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  int2 size;
  u_denoised.GetDimensions(size.x, size.y);
  const int2 pixel = int2(dispatchThreadID.xy);
  if(pixel.x >= size.x || pixel.y >= size.y)
    return;

  // the mean of the 3x3 neighbourhood's noise squared; any with too few samples, none kept
  float sum   = 0.0F;
  bool  tooFew = false;
  for(int dy = -1; dy <= 1; dy++)
    for(int dx = -1; dx <= 1; dx++)
    {
      const float n = noiseAt(clamp(pixel + int2(dx, dy), int2(0), size - 1));
      tooFew        = tooFew || n < 0.0F;
      sum += n;
    }
  const float share = stillKeepShare(tooFew ? -1.0F : sum / 9.0F, pushConst.threshold);

  if(pushConst.view != 0)
  {
    u_denoised[pixel] = float4(float3(share), 1.0F);
    return;
  }
  if(share <= 0.0F)
    return;
  // a weighted sum, not a lerp: an infinite raw value (a sun disk) would make that NaN
  u_denoised[pixel] = u_denoised[pixel] * (1.0F - share) + u_raw[pixel] * share;
}
