// The bounce cache's pass after a frame (radiance_cache.h.slang, RadianceCache in
// src/radiance_cache.hpp): each cell's sums of the frame folded into what it keeps and cleared
// for the next, and a cell no path has fed for RC_MAX_AGE frames freed. One thread a slot; the paths
// that wrote the sums have ended, and the next frame's start after this.

#include "shaderio.h"
#include "radiance_cache.h.slang"

// clang-format off
[[vk::push_constant]] ConstantBuffer<RadianceCacheResolvePushConstant> pushConst;
// clang-format on

[shader("compute")]
[numthreads(RC_RESOLVE_GROUP, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  if(dispatchThreadID.x >= pushConst.slots)
    return;
  uint*      table = pushConst.table;
  const uint base  = dispatchThreadID.x * uint(RC_WORDS);
  if(table[base + uint(RC_KEY)] == 0u)
    return;

  const RadianceCacheCell held = radianceCacheRead(table, int(dispatchThreadID.x));
  float3                  sumLight;
  float3                  sumWeight;
  [ForceUnroll]
  for(int c = 0; c < 3; c++)
  {
    sumLight[c]  = asfloat(table[base + uint(RC_SUM_LIGHT + c)]);
    sumWeight[c] = asfloat(table[base + uint(RC_SUM_WEIGHT + c)]);
  }

  uint age = table[base + uint(RC_AGE)];
  age      = radianceCacheReceived(sumWeight) > 0.0F ? 0u : age + 1u;
  if(age > RC_MAX_AGE)
  {
    for(int i = 0; i < RC_WORDS; i++)
      table[base + uint(i)] = 0u;
    return;
  }

  RadianceCacheCell merged = radianceCacheMerge(held, sumLight, sumWeight, pushConst.maxWeight);
  // sums that are no number (one overflowed) start the cell over
  const float total = merged.light.x + merged.light.y + merged.light.z + merged.weight.x + merged.weight.y + merged.weight.z;
  if(!(total >= 0.0F) || !(total < 3.0e38F))
  {
    merged.light  = float3(0.0F);
    merged.weight = float3(0.0F);
  }

  table[base + uint(RC_AGE)] = age;
  [ForceUnroll]
  for(int c = 0; c < 3; c++)
  {
    table[base + uint(RC_LIGHT + c)]      = asuint(merged.light[c]);
    table[base + uint(RC_WEIGHT + c)]     = asuint(merged.weight[c]);
    table[base + uint(RC_SUM_LIGHT + c)]  = 0u;
    table[base + uint(RC_SUM_WEIGHT + c)] = 0u;
  }
}
