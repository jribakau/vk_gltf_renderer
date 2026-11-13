/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The rain's shelter map (src/rain.hpp): over a square of the ground round the camera, the height of
// the first surface a drop falling there meets, traced down the way the wind slants the rain. No
// drop is drawn below it: none falls under a roof, a tree or a figure. Every surface stops it, alpha
// cut-outs whole (a tree's leaves shelter what is under them).

#include "rain_io.h.slang"
#include "rain.h.slang"

// clang-format off
[[vk::binding(RainBindings::eRainTlas, 1)]] RaytracingAccelerationStructure u_scene;

[[vk::push_constant]] ConstantBuffer<RainPush> pc;
// clang-format on

[shader("compute")]
[numthreads(16, 16, 1)]
void rainShelter(uint3 id: SV_DispatchThreadID)
{
  if(id.x >= RAIN_SHELTER_SIZE || id.y >= RAIN_SHELTER_SIZE)
    return;
  RainFrame*   f   = pc.frame;
  const float2 h   = f->shelterCentre + ((float2(id.xy) + 0.5F) / float(RAIN_SHELTER_SIZE) - 0.5F) * RAIN_SHELTER_EXTENT;
  const float  top = f->shelterTop;
  RayDesc      ray;
  ray.Origin    = float3(h.x + f->shear.x * top, top, h.y + f->shear.y * top);
  ray.Direction = normalize(float3(-f->shear.x, -1.0F, -f->shear.y));
  ray.TMin      = 0.0F;
  ray.TMax      = 1e5F;
  RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
  query.TraceRayInline(u_scene, RAY_FLAG_NONE, 0xFF, ray);
  while(query.Proceed())
  {
  }
  float height = RAIN_SHELTER_OPEN;
  if(query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    height = ray.Origin.y + ray.Direction.y * query.CommittedRayT();
  f->shelter[id.y * RAIN_SHELTER_SIZE + id.x] = height;
}
