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

// The lens pass (src/lens.hpp): the rendered image as a cheap wide lens draws it, bent and with its
// colours split toward the corners (shaders/lens.h.slang), for the tone mapper to take.

#include "lens_io.h.slang"
#include "lens.h.slang"

// clang-format off
[[vk::binding(LensBindings::eLensIn)]]  Sampler2D<float4>   u_in;
[[vk::binding(LensBindings::eLensOut)]] RWTexture2D<float4> u_out;

[[vk::push_constant]] ConstantBuffer<LensPushConstant> pc;
// clang-format on

static const float kHalfMax = 65504.0;  // Largest half float: the output is RGBA16F

// A tap that is a finite colour: an infinite pixel (a sun disk) or a NaN would spread to its
// neighbours through the filter
float4 tap(float2 uv)
{
  const float4 c = u_in.SampleLevel(uv, 0);
  return float4(all(isfinite(c.rgb)) ? clamp(c.rgb, float3(-kHalfMax), float3(kHalfMax)) : float3(0.0), c.a);
}

[shader("compute")]
[numthreads(LENS_WORKGROUP_SIZE, LENS_WORKGROUP_SIZE, 1)]
void lens(uint3 id: SV_DispatchThreadID)
{
  const int2 p = int2(id.xy);
  if(p.x >= pc.outSize.x || p.y >= pc.outSize.y)
    return;
  const float2 uv     = (float2(p) + 0.5) / float2(pc.outSize);
  const float  aspect = float(pc.outSize.x) / float(pc.outSize.y);
  const float4 green  = tap(lensSource(uv, aspect, pc.bend, pc.fringe, 0.0));
  float3       rgb    = green.rgb;
  if(pc.fringe > 0.0)
  {
    rgb.r = tap(lensSource(uv, aspect, pc.bend, pc.fringe, -1.0)).r;
    rgb.b = tap(lensSource(uv, aspect, pc.bend, pc.fringe, 1.0)).b;
  }
  u_out[p] = float4(rgb, green.a);
}
