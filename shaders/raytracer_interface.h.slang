/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "get_hit.h.slang"
#include "shaderio.h"

// NVSHADERS are under: https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders
#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"

// Debug-visualization gate. Set to 1 by the host (see PathTracer::compileShader) when a
// visualization mode other than eRendered is selected; 0 in the shipping render path. Gates the
// OMM debug payload field and the visualization branches so they cost nothing in normal rendering.
// Default here so the embedded build (which does not pass the macro) compiles it out cleanly.
#ifndef USE_VISUALIZE
#define USE_VISUALIZE 0
#endif

// Payload for the path tracer
struct HitPayload
{
  uint   seed;
  float  hitT        = 0.0f;
  int    rnodeID     = -1;
  int    rprimID     = -1;
  int    primitiveID = -1;
  float2 bary        = float2(0);  // BuiltInTriangleIntersectionAttributes.barycentrics; third = 1 - x - y
#if USE_VISUALIZE
  bool ommUnknown = false;  // Opacity-micromap debug view: set when any-hit ran (micro-triangle is "unknown")
#endif
};

// Shadow payload for the path tracer
struct ShadowPayload
{
  uint seed;
  float prevHitT = 0.0f;  // RayT of the last medium boundary crossed (the origin until one is). shadowCrossSurface charges abs(hitT - prevHitT) of Beer's law at the next boundary; any-hit order is not guaranteed, so abs() is an approximation that works well for thin/convex volumes.
  bool   isInside          = false;         // If true, the shadow ray is considered to start INSIDE a volume
  float3 extinction        = float3(0.0f);  // sigma_t of the medium the ray is in, while isInside
  float3 totalTransmission = float3(1.0f);  // The total transmission of the shadow ray through the volume
  bool   missed            = false;         // Set by rmissShadow; shadow rays skip closest-hit, so a hit leaves it false
};

// Shadow rays only ask whether anything blocks the light: any committed hit (an opaque instance,
// or rahitShadow's AcceptHitAndEndSearch once shadowCrossSurface blocks the ray) means fully occluded, so traversal
// can stop at the first one instead of searching for the closest.
static const uint SHADOW_RAY_FLAGS = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

// Which sides of a surface a path's ray sees. glTF's single-sidedness is about what is drawn: a camera
// ray doesn't see a single-sided surface from behind (SceneRtx sets
// VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR on double-sided and transmissive instances,
// which see both). To light it is solid from either side, so a bounce stops at a back face as a shadow
// ray does (SHADOW_RAY_FLAGS has no cull: a roof that faces the sun still shades what's under it), and
// the BSDF's and the light sample's estimates of one light agree. A walk inside a medium keeps the
// camera's: a back face there says its medium ended, and only the medium's own mesh may say that.
static const uint CAMERA_RAY_FLAGS = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
static const uint BOUNCE_RAY_FLAGS = RAY_FLAG_NONE;

// Raytracer interface definition
interface IRaytracer
{
  // `instanceMask`: the TLAS instances the ray can hit (shaderio.h INSTANCE_MASK_*; 0xFF for all);
  // `rayFlags`: CAMERA_RAY_FLAGS or BOUNCE_RAY_FLAGS
  void Trace(RayDesc ray, inout HitPayload payload, inout uint seed, uint instanceMask, uint rayFlags);
  void TraceLow(RayDesc ray, inout HitPayload payload);
  // A shadow ray that starts inside a medium says so, with that medium's extinction: it attenuates
  // every segment it travels inside one, up to TMax.
  float3 TraceShadow(RayDesc ray, inout uint seed, bool initialInside = false, float3 initialExtinction = float3(0.0));
}

// Implementation using RayQuery
struct RayQueryRaytracer : IRaytracer
{
  void Trace(RayDesc ray, inout HitPayload payload, inout uint seed, uint instanceMask, uint rayFlags)
  {
    payload.hitT = INFINITE;  // Default when not hitting anything
#if USE_VISUALIZE
    payload.ommUnknown = false;  // Clear stale debug state; the OMM viz re-sets it below per ray
#endif

    RayQuery rayQuery;
    rayQuery.TraceRayInline(topLevelAS, rayFlags, instanceMask, ray);

    while(rayQuery.Proceed())
    {
      int    instanceID   = rayQuery.CandidateInstanceIndex();
      int    renderPrimID = rayQuery.CandidateInstanceID();
      int    triangleID   = rayQuery.CandidatePrimitiveIndex();
      float2 bary         = rayQuery.CandidateTriangleBarycentrics();

      // Barycentric coordinate on the triangle
      const float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

#if USE_VISUALIZE
      // Opacity-micromap debug view: reaching a candidate here means the micro-triangle is
      // "unknown" (OMM-opaque micro-triangles are committed without yielding a candidate,
      // transparent ones are culled). Commit it and stop, so it is colored as unknown.
      if(pushConst.frameInfo->visualization == Visualization::eOpacityMicromap)
      {
        payload.ommUnknown = true;
        rayQuery.CommitNonOpaqueTriangleHit();
        break;
      }
#endif

      GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
      GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

      float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);

      // do alpha blending the stochastically way
      if(stochasticOpacityU(uint(instanceID), uint(triangleID), barycentrics, seed) < opacity)
        rayQuery.CommitNonOpaqueTriangleHit();
    }

    if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      payload.hitT        = rayQuery.CommittedRayT();
      payload.rprimID     = rayQuery.CommittedInstanceID();
      payload.rnodeID     = rayQuery.CommittedInstanceIndex();
      payload.primitiveID = rayQuery.CommittedPrimitiveIndex();
      payload.bary        = rayQuery.CommittedTriangleBarycentrics();
    }
  }

  void TraceLow(RayDesc ray, inout HitPayload payload)
  {
    payload.hitT = INFINITE;
    RayQuery rayQuery;
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    while(rayQuery.Proceed())
    {
    }  // All geometry treated as opaque, nothing to process
    if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      payload.hitT    = rayQuery.CommittedRayT();
      payload.rnodeID = rayQuery.CommittedInstanceIndex();
    }
  }

  float3 TraceShadow(RayDesc ray, inout uint seed, bool initialInside = false, float3 initialExtinction = float3(0.0))
  {
    const float ground = infinitePlaneShadowTransmittance(ray.Origin, ray.Direction, ray.TMax, pushConst.frameInfo);
    if(ground <= 0.0F)
      return float3(0.0);

    float3      totalTransmission = float3(1.0);
    bool        isInside          = initialInside;
    float3      extinction        = initialExtinction;
    float       prevHitT          = 0;

    // Visibility needs any occluder, not the closest one: the first committed (opaque) hit ends the
    // traversal. Transmissive surfaces are never committed -- they only scale the transmission below.
    RayQuery rayQuery;
    rayQuery.TraceRayInline(topLevelAS, SHADOW_RAY_FLAGS, 0xFF, ray);

    while(rayQuery.Proceed())
    {
      float    hitT          = rayQuery.CandidateTriangleRayT();
      int      renderPrimID  = rayQuery.CandidateInstanceID();
      int      instanceID    = rayQuery.CandidateInstanceIndex();
      int      triangleID    = rayQuery.CandidatePrimitiveIndex();
      float2   bary          = rayQuery.CandidateTriangleBarycentrics();
      float4x3 worldToObject = rayQuery.CandidateWorldToObject4x3();

      GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
      GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

      float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
      if(!shadowCrossSurface(renderNode, renderPrim, uint(instanceID), triangleID, barycentrics, worldToObject, ray.Direction,
                             hitT, totalTransmission, isInside, extinction, prevHitT, seed))
        return float3(0.0);
    }

    if(rayQuery.CommittedStatus() != COMMITTED_NOTHING)
    {
      return float3(0., 0., 0.);
    }

    return ground * totalTransmission * shadowTailTransmittance(ray, isInside, extinction, prevHitT);
  }
}

// Implementation using traditional ray tracing
struct TraditionalRaytracer : IRaytracer
{
  void Trace(RayDesc ray, inout HitPayload payload, inout uint seed, uint instanceMask, uint rayFlags)
  {
    payload.seed = seed;
#if USE_VISUALIZE
    payload.ommUnknown = false;  // Clear stale debug state; any-hit re-sets it for the OMM viz
#endif
    if(USE_SER == 1 && AVAILABLE_SER == 1)
    {
      HitObject hitObj = HitObject::TraceRay(topLevelAS, rayFlags, instanceMask, 0, 0, 0, ray, payload);

      // Simple default thread reordering, not coherence hinting
      ReorderThread(hitObj);

      // Miss everything, return early
      if(hitObj.IsMiss())
      {
        payload.hitT = INFINITE;
        seed         = payload.seed;
        return;
      }

      // There is a hit: record only the raw triangle hit (HitState is rebuilt in pathTrace).
      BuiltInTriangleIntersectionAttributes attr = hitObj.GetAttributes<BuiltInTriangleIntersectionAttributes>();

      payload.hitT        = hitObj.GetRayDesc().TMax;
      payload.rprimID     = hitObj.GetInstanceID();
      payload.rnodeID     = hitObj.GetInstanceIndex();
      payload.primitiveID = hitObj.GetPrimitiveIndex();
      payload.bary        = attr.barycentrics;
    }
    else
    {
      TraceRay(topLevelAS, rayFlags, instanceMask, 0, 0, 0, ray, payload);
    }
    seed = payload.seed;
  }

  void TraceLow(RayDesc ray, inout HitPayload payload)
  {
    if(USE_SER == 1 && AVAILABLE_SER == 1)
    {
      HitObject hitObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, ray, payload);
      if(hitObj.IsMiss())
      {
        payload.hitT = INFINITE;
        return;
      }
      payload.hitT    = hitObj.GetRayDesc().TMax;
      payload.rnodeID = hitObj.GetInstanceIndex();
    }
    else
    {
      // Accepted trade-off: this dispatches rchitMain, but only hitT/rnodeID are used.
      // A dedicated minimal CHS would require an extra SBT entry. The SER path above
      // already avoids that dispatch overhead.
      TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);
    }
  }

  // Visibility needs any occluder, not the closest one: the first committed hit (opaque, or accepted
  // by rahitShadow once the transmission is spent) ends the traversal; transmissive surfaces are
  // ignored by rahitShadow and never committed.
  float3 TraceShadow(RayDesc ray, inout uint seed, bool initialInside = false, float3 initialExtinction = float3(0.0))
  {
    const float ground = infinitePlaneShadowTransmittance(ray.Origin, ray.Direction, ray.TMax, pushConst.frameInfo);
    if(ground <= 0.0F)
      return float3(0.0);

    ShadowPayload shadowPayload = {};
    shadowPayload.seed          = seed;
    shadowPayload.isInside      = initialInside;
    shadowPayload.extinction    = initialExtinction;

    if(USE_SER == 1 && AVAILABLE_SER == 1)
    {
      // Any-hit (rahitShadow) still runs during HitObject traversal and accumulates transmission;
      // only closest-hit/miss are deferred, so a committed hit means the ray was blocked.
      HitObject hitObj = HitObject::TraceRay(topLevelAS, SHADOW_RAY_FLAGS, 0xFF, 1, 0, 1, ray, shadowPayload);
      if(hitObj.IsHit())
      {
        shadowPayload.totalTransmission = float3(0.0);
      }
    }
    else
    {
      // No closest-hit shader runs; the miss shader reports a clear path instead.
      TraceRay(topLevelAS, SHADOW_RAY_FLAGS | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 1, 0, 1, ray, shadowPayload);
      if(!shadowPayload.missed)
      {
        shadowPayload.totalTransmission = float3(0.0);
      }
    }
    seed = shadowPayload.seed;  // any-hit consumed random numbers; propagate on every path

    return ground * shadowPayload.totalTransmission
           * shadowTailTransmittance(ray, shadowPayload.isInside, shadowPayload.extinction, shadowPayload.prevHitT);
  }
}
