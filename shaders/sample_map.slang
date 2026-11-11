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

// The Samples view (src/sample_map.hpp): each pixel of a still in viridis by the samples it took, on a
// log scale from the adaptive stop's minimum to the still's cap (adaptiveSamplesRamp). Display values,
// written where the tone mapper would write the picture.

#include "sample_map_io.h.slang"
#include "adaptive_stop.h.slang"
#include "debug_colors.h.slang"

// clang-format off
[[vk::binding(SampleMapBindings::eSampleMapState)]] RWTexture2D<float4> u_state;
[[vk::binding(SampleMapBindings::eSampleMapOut)]]   RWTexture2D<float4> u_out;

[[vk::push_constant]] ConstantBuffer<SampleMapPushConstant> pc;
// clang-format on

[shader("compute")]
[numthreads(SAMPLE_MAP_WORKGROUP_SIZE, SAMPLE_MAP_WORKGROUP_SIZE, 1)]
void sampleMap(uint3 id: SV_DispatchThreadID)
{
  const int2 pixel = int2(id.xy);
  if(any(pixel >= pc.outSize))
    return;
  const float state = u_state[pixel].y;
  // nothing written by the stop there: grey, as the Density view's pixels without a mesh
  const float3 color = state == 0.0F ? float3(0.35F) :
                                       viridis(adaptiveSamplesRamp(adaptiveStateSamples(state), pc.minSamples, pc.cap));
  u_out[pixel] = float4(color, 1.0F);
}
