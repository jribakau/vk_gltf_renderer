/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The ground (infinite plane), as both renderers draw it.
#ifndef GROUND_H_SLANG
#define GROUND_H_SLANG

#include "shaderio.h"

// How much of the ground there is at `pos`: 1 over the inner 55% of the fade disk, falling linearly
// to 0 at its edge -- the radial gradient the Figure Viewer's three.js floor disk is drawn with.
// 1 everywhere when there is no fade.
float groundFadeWeight(float3 pos, SceneFrameInfo* frameInfo)
{
  if(frameInfo->groundFadeRadius <= 0.0F)
    return 1.0F;
  const float d = length(pos.xz - float2(frameInfo->groundFadeCenterX, frameInfo->groundFadeCenterZ)) / frameInfo->groundFadeRadius;
  return saturate((1.0F - d) / (1.0F - 0.55F));
}

// Where a ray from above the ground crosses it before tMax: the distance, or -1 for no crossing
float infinitePlaneCrossing(float3 origin, float3 direction, float tMax, SceneFrameInfo* frameInfo)
{
  if((frameInfo->flags & int(SceneFrameInfoFlags::eSceneUseInfinitePlane)) == 0)
    return -1.0F;
  const float planeHeight = frameInfo->infinitePlaneDistance;
  if(origin.y <= planeHeight || abs(direction.y) <= 1e-6F)
    return -1.0F;
  const float t = (planeHeight - origin.y) / direction.y;
  return (t > 0.0F && t < tMax) ? t : -1.0F;
}

// The ground as a path tracer's shadow ray sees it. A path's own ray hits the plane with
// probability groundFadeWeight (checkInfinitePlaneIntersection), so what passes is, in expectation,
// its complement there. Without it the light samples and the BSDF rays integrated different scenes:
// a figure's undersides were lit through the floor by the environment's lower half. The shadow
// catcher is skipped: it is a compositing effect, not geometry. Path rays that cross it are still
// darkened by handleShadowCatcher, so in catcher mode the two do not integrate the same scene.
float infinitePlaneShadowTransmittance(float3 origin, float3 direction, float tMax, SceneFrameInfo* frameInfo)
{
  if((frameInfo->flags & int(SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher)) != 0)
    return 1.0F;
  const float t = infinitePlaneCrossing(origin, direction, tMax, frameInfo);
  return t < 0.0F ? 1.0F : 1.0F - groundFadeWeight(origin + direction * t, frameInfo);
}

// The rasterizer's catcher has no path to trace, so it darkens by a fraction of what it catches.
// Most of it at darkness 0, all of it at 1, so the setting reads the same way in both renderers.
float groundCatcherStrength(SceneFrameInfo* frameInfo)
{
  return lerp(0.6F, 1.0F, saturate(frameInfo->shadowCatcherDarkenAmount));
}

#endif  // GROUND_H_SLANG
