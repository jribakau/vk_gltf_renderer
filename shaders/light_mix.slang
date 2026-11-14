/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The light groups' mix over a still (LightGroups in src/light_groups.hpp, light_mix.h.slang): the
// rendered image is the accumulation of every light with each group's share scaled and tinted. Each
// pixel reads and writes itself alone.

#include "shaderio.h"
#include "light_mix.h.slang"

// clang-format off
[[vk::binding(LightMixBindings::eMixOut)]]    RWTexture2D<float4> u_out;
[[vk::binding(LightMixBindings::eMixTotal)]]  RWTexture2D<float4> u_total;
[[vk::binding(LightMixBindings::eMixGroups)]] RWTexture2D<float4> u_groups[LIGHT_GROUP_COUNT];

[[vk::push_constant]] ConstantBuffer<LightMixPushConstant> pushConst;
// clang-format on

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  int2 size;
  u_out.GetDimensions(size.x, size.y);
  const int2 pixel = int2(dispatchThreadID.xy);
  if(pixel.x >= size.x || pixel.y >= size.y)
    return;

  // the least factor among the groups with light: the share of the whole picture the mix can keep
  float3 floor = float3(1e30F);
  for(int g = 0; g < LIGHT_GROUP_COUNT; g++)
    floor = lightMixFloor(floor, pushConst.mix[g].xyz, ((pushConst.used >> g) & 1u) != 0u);
  if(floor.x > 1e29F)
    floor = float3(1.0F);  // no group has light: the picture as it is

  const float4 whole = u_total[pixel];
  float3       terms = float3(0.0F);
  for(int g = 0; g < LIGHT_GROUP_COUNT; g++)
    if(((pushConst.used >> g) & 1u) != 0u)
      terms += lightMixTerm(u_groups[g][pixel].xyz, pushConst.mix[g].xyz, floor);
  // (the alpha is the whole's: whether the camera's ray met anything)
  u_out[pixel] = float4(lightMixed(whole.xyz, floor, terms), whole.w);
}
