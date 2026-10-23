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

// The rasterizer's catcher has no path to trace, so it darkens by a fraction of what it catches.
// Most of it at darkness 0, all of it at 1, so the setting reads the same way in both renderers.
float groundCatcherStrength(SceneFrameInfo* frameInfo)
{
  return lerp(0.6F, 1.0F, saturate(frameInfo->shadowCatcherDarkenAmount));
}

#endif  // GROUND_H_SLANG
