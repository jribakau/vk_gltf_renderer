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

//=============================================================================
// World-space helper lines (gizmo_lines_vk.hpp): a list of segments read from a buffer by vertex
// index, drawn over the tonemapped image. Behind scene geometry a line still shows, faintly, so a
// camera or light inside a wall can be found.
//=============================================================================

#include "gizmo_lines_shaderio.h.slang"

[[vk::push_constant]]
ConstantBuffer<LinePushConstants> pc;

[[vk::binding(0, 0)]]
Texture2D<float> sceneDepth;
[[vk::binding(1, 0)]]
SamplerState depthSampler;

struct LineVertexOutput
{
  float4                 position : SV_Position;
  float3                 worldPos : TEXCOORD0;
  nointerpolation float4 color : TEXCOORD1;
};

[shader("vertex")]
LineVertexOutput vertmain(uint vertexId: SV_VertexID)
{
  const LineVertex v = pc.vertices[vertexId];
  LineVertexOutput o;
  o.worldPos = v.position;
  o.color    = float4(float(v.color & 0xFFu), float((v.color >> 8) & 0xFFu), float((v.color >> 16) & 0xFFu), float(v.color >> 24)) / 255.0;
  o.position = mul(float4(v.position, 1.0), pc.viewProj);
  return o;
}

// As the grid does it (gizmo_grid.slang): reprojected with the scene camera, compared as distance.
bool occludedByScene(float2 fragCoord, float3 worldPos)
{
  const float4 clip = mul(float4(worldPos, 1.0), pc.sceneViewProj);
  if(clip.w <= 0.0)
    return false;
  const int2  texel = int2(fragCoord * (pc.depthBufferSize / pc.viewportSize));
  const float scene = sceneDepth.Load(int3(texel, 0));
  const float line  = clip.z / clip.w;
  if(pc.isPerspective != 0)
    return (1.0 - scene) > (1.0 - line) * 1.002;
  return scene < line - 1e-5;
}

[shader("fragment")]
float4 fragmain(LineVertexOutput input) : SV_Target0
{
  float4 c = input.color;
  if(occludedByScene(input.position.xy, input.worldPos))
    c.a *= pc.occludedAlpha;
  return c;
}
