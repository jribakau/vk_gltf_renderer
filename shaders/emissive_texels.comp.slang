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

// A textured emitter's mean colour over each of its triangles, so each is weighed as a light by its
// own texels, not its material's (rs::Residency, the emissive table). One thread a triangle: points
// stratified over its area, read at the level of detail whose texels they spread over, so the
// filtering covers the rest. As Falcor's EmissiveIntegrator does by rasterizing in texture space.

#include "shaderio.h"
#include "emissive_texels_io.h.slang"

// clang-format off
[[vk::push_constant]]                        ConstantBuffer<EmissiveTexelsPushConstant> pc;
[[vk::binding(BindingPoints::eTextures, 0)]] Texture2D allImages[];
[[vk::binding(BindingPoints::eSamplers, 0)]] SamplerState allSamplers[];
// clang-format on

// Van der Corput's radical inverse of i in base 2: the second coordinate of a Hammersley set
float radicalInverse(uint i)
{
  return float(reversebits(i)) * 2.3283064365386963e-10F;
}

[shader("compute")]
[numthreads(EMISSIVE_TEXELS_WORKGROUP, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  const uint i = tid.x;
  if(i >= pc.count)
    return;
  const TexelTriangle t     = pc.triangles[i];
  Texture2D           image = allImages[NonUniformResourceIndex(t.texture)];
  SamplerState        smp   = allSamplers[NonUniformResourceIndex(t.sampler)];
  uint                w, h, levels;
  image.GetDimensions(0, w, h, levels);
  // the triangle's area in texels, shared out among the points: the level whose texel is that big
  const float2 e1    = t.uv1 - t.uv0, e2 = t.uv2 - t.uv0;
  const float  area  = 0.5F * abs(e1.x * e2.y - e1.y * e2.x) * float(w) * float(h);
  const float  lod   = clamp(0.5F * log2(max(area / float(EMISSIVE_TEXELS_SAMPLES), 1.0F)), 0.0F, float(max(levels, 1u) - 1u));
  float3       sum   = float3(0.0F);
  for(uint s = 0; s < EMISSIVE_TEXELS_SAMPLES; s++)
  {
    // a Hammersley point warped onto the triangle uniformly by area
    const float  u  = (float(s) + 0.5F) / float(EMISSIVE_TEXELS_SAMPLES);
    const float  v  = radicalInverse(s);
    const float  su = sqrt(u);
    const float2 uv = t.uv0 + e1 * (1.0F - su) + e2 * (v * su);
    sum += image.SampleLevel(smp, uv, lod).rgb;
  }
  pc.means[i] = float4(sum / float(EMISSIVE_TEXELS_SAMPLES), 1.0F);
}
