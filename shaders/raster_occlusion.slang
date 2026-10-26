/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// The real-time view's traced occlusion and shadows, once per pixel (Rasterizer::traceOcclusion).
//
// Traced in the raster's fragment shader they cost about 60 ms per ray at 4K: every overdrawn
// fragment traced, and the uber-shader hid the rays' latency poorly. Here a depth prepass has
// already found each pixel's visible surface (the G-buffer: its position and geometric normal), so
// traceMain traces the ambient occlusion and the shadows of the lights the shadow map does not
// cover once per half-resolution texel, from a lean shader, and blurMain filters the result over
// the texels at the same depth. The raster's fragments read it back with a depth-aware upsample.

#include "gltf_scene_io.h.slang"
#include "gltf_vertex_access.h.slang"
#include "gltf_material_eval.h.slang"
#include "shaderio.h"
#include "nvshaders/functions.h.slang"
#include "nvshaders/light_contrib.h.slang"
#include "nvshaders/ray_utils.h.slang"  // pointOffset, for get_hit.h.slang
#include "sky_background.h.slang"  // makeSkySunLight, applySkySunLight
#include "ground.h.slang"

// clang-format off
[[vk::push_constant]]                                           ConstantBuffer<RasterOcclusionPushConstant> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]                    Texture2D                       allImages[];
[[vk::binding(BindingPoints::eSamplers, 0)]]                    SamplerState                    allSamplers[];
[[vk::binding(RasterOcclusionBindings::eRasterOccTlas, 1)]]     RaytracingAccelerationStructure g_tlas;
[[vk::binding(RasterOcclusionBindings::eRasterOccGbuffer, 1)]]  RWTexture2D<uint4>              g_gbuffer;
[[vk::binding(RasterOcclusionBindings::eRasterOccRaw, 1)]]      RWTexture2DArray<float4>        g_raw;
[[vk::binding(RasterOcclusionBindings::eRasterOccFiltered, 1)]] RWTexture2DArray<float4>        g_filtered;
// clang-format on

#include "surface_opacity.h.slang"
#include "raster_rays.h.slang"

// A world-space point on the pixel's camera ray at NDC depth `z` (fragmentGroundMain's)
float3 unprojectPixel(float2 ndc, float z)
{
  float4 v = mul(float4(ndc, z, 1.0), pushConst.frameInfo.projInv);
  v /= v.w;
  return mul(float4(v.xyz, 1.0), pushConst.frameInfo.viewInv).xyz;
}

// Where pixel `pixel`'s ray meets the floor, as the ground pass draws it; false where it draws none
bool groundPoint(float2 pixel, out float3 P)
{
  SceneFrameInfo* fi  = pushConst.frameInfo;
  const float2    ndc = ((pixel + 0.5) / fi.imageSize) * 2.0 - 1.0 + fi.jitter * (2.0 / fi.imageSize);
  const float3    a   = unprojectPixel(ndc, 0.25);
  const float3    b   = unprojectPixel(ndc, 0.75);
  const float3    eye = float3(fi.viewInv[3].x, fi.viewInv[3].y, fi.viewInv[3].z);
  const float3    fwd = normalize(mul(float4(0.0, 0.0, -1.0, 0.0), fi.viewInv).xyz);
  float3          dir = normalize(b - a);
  if(dot(dir, fwd) < 0.0)
    dir = -dir;
  const float3 origin = a - dir * dot(a - eye, dir);
  P                   = origin;
  const float planeY  = fi.infinitePlaneDistance;
  if(origin.y <= planeY || dir.y >= -1e-6)
    return false;
  P = origin + dir * ((planeY - origin.y) / dir.y);
  return groundFadeWeight(P, fi) > 0.0;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void traceMain(uint3 tid: SV_DispatchThreadID)
{
  uint3 rawSize;
  g_raw.GetDimensions(rawSize.x, rawSize.y, rawSize.z);
  if(tid.x >= rawSize.x || tid.y >= rawSize.y)
    return;
  uint2 fullSize;
  g_gbuffer.GetDimensions(fullSize.x, fullSize.y);

  // The texel stands for the top-left pixel of its 2x2 block
  const uint2  pixel = min(tid.xy * 2u, fullSize - 1u);
  const uint4  g     = g_gbuffer[pixel];
  const float3 eye   = float3(pushConst.frameInfo.viewInv[3].x, pushConst.frameInfo.viewInv[3].y, pushConst.frameInfo.viewInv[3].z);

  float3 P, N;
  float  radius;
  if(g.w != RASTER_GBUFFER_EMPTY)
  {
    P      = asfloat(g.xyz);
    N      = unpackOctNormal(g.w);
    radius = AO_RADIUS_SURFACE;
  }
  else if(pushConst.groundOn != 0 && groundPoint(float2(pixel), P))
  {
    N      = float3(0.0, 1.0, 0.0);
    radius = AO_RADIUS_FLOOR;
  }
  else
  {
    g_raw[uint3(tid.xy, 0)] = float4(1.0, 0.0, 1.0, 1.0);
    g_raw[uint3(tid.xy, 1)] = float4(1.0);
    return;
  }

  // The surface as a hit: flat at the pixel, its rounding bound a float's at that distance, and
  // stepped off it no less than the raster's interpolated position can be off the BLAS's triangle
  HitState hit;
  hit.pos            = P;
  hit.nrm            = N;
  hit.geonrm         = N;
  hit.shadowPos      = P;
  hit.offsetDistance = max(2e-6 * max(1.0, max(abs(P.x), max(abs(P.y), abs(P.z)))), 2e-5);

  const float2 fp = float2(pixel);
  const float  ao = rayTracedOcclusion(hit, N, radius, fp, pushConst.aoRays);

  float shadows[RASTER_OCCLUSION_LIGHTS] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  if(pushConst.shadowRays > 0)
  {
    SceneFrameInfo* fi         = pushConst.frameInfo;
    const int       lightCount = min(pushConst.gltfScene.numLights + (fi.rendererSunLight != 0 ? 1 : 0), RASTER_OCCLUSION_LIGHTS);
    for(int i = 0; i < lightCount; i++)
    {
      GltfLight light = (i >= pushConst.gltfScene.numLights) ? makeSkySunLight(fi) : pushConst.gltfScene.lights[i];
      applySkySunLight(light, i, fi);
      if(i == fi.shadowLightIndex && light.type == LightType::eLightTypeDirectional)
        continue;  // the shadow map's
      // out of the light's reach (range, cone, no intensity): nothing to shadow
      const float3 toLight = light.type == LightType::eLightTypeDirectional ? -light.direction : normalize(light.position - P);
      if(all(singleLightContribution(light, P, toLight).intensity == 0.0))
        continue;
      shadows[i] = rayTracedShadow(i, light, hit, N, fp, pushConst.shadowRays);
    }
  }

  g_raw[uint3(tid.xy, 0)] = float4(ao, length(P - eye), shadows[0], shadows[1]);
  g_raw[uint3(tid.xy, 1)] = float4(shadows[2], shadows[3], shadows[4], shadows[5]);
}

// A 5x5 filter over the texels at the same depth: the traced rays' per-pixel rotations average out
// into a smooth occlusion, and nothing bleeds across a silhouette (a texel more than 1% of its
// distance nearer or farther counts for nothing).
[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void blurMain(uint3 tid: SV_DispatchThreadID)
{
  uint3 size;
  g_raw.GetDimensions(size.x, size.y, size.z);
  if(tid.x >= size.x || tid.y >= size.y)
    return;

  const float4 c0 = g_raw[uint3(tid.xy, 0)];
  const float  d0 = c0.y;
  if(d0 <= 0.0)
  {
    g_filtered[uint3(tid.xy, 0)] = c0;
    g_filtered[uint3(tid.xy, 1)] = g_raw[uint3(tid.xy, 1)];
    return;
  }

  const float tolerance = 0.01 * d0;
  float4      sum0      = float4(0.0);
  float4      sum1      = float4(0.0);
  float       wsum      = 0.0;
  for(int y = -2; y <= 2; y++)
  {
    for(int x = -2; x <= 2; x++)
    {
      const int2 q = int2(tid.xy) + int2(x, y);
      if(any(q < 0) || any(q >= int2(size.xy)))
        continue;
      const float4 a = g_raw[uint3(q, 0)];
      if(a.y <= 0.0)
        continue;
      const float w = saturate(1.0 - abs(a.y - d0) / tolerance) * exp(-0.25 * float(x * x + y * y));
      if(w <= 0.0)
        continue;
      sum0 += w * a;
      sum1 += w * g_raw[uint3(q, 1)];
      wsum += w;
    }
  }
  float4 f0 = sum0 / wsum;
  f0.y      = d0;  // the texel's own distance, for the raster's upsample
  g_filtered[uint3(tid.xy, 0)] = f0;
  g_filtered[uint3(tid.xy, 1)] = sum1 / wsum;
}
