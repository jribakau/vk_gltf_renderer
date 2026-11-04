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

// How much of the light it catches a shadow catcher darkens by: most of it at darkness 0, all of it
// at 1.
float groundCatcherStrength(SceneFrameInfo* frameInfo)
{
  return lerp(0.6F, 1.0F, saturate(frameInfo->shadowCatcherDarkenAmount));
}

//-----------------------------------------------------------------------
// Water on the ground
//-----------------------------------------------------------------------
// Standing water and the damp round it, from one noise field over the floor in world space
// (metres) whose low ground fills first. The amount wets patches of the floor first and all of it
// by half; from 0.3 puddles form, and they join until at 1 the floor is under a sheet of water,
// whose surface ripples. Evaluated where a ray meets the floor: no texture and no extra ray.

static const float GROUND_WATER_IOR = 1.33F;
// The field's largest octave, metres: puddles about the size of a figure's stance
static const float GROUND_PUDDLE_SCALE = 1.2F;
// The edges, as logistic widths in the field's logit (groundWaterAt): about 1.3 cm for the water's,
// 20 cm for the damp's
static const float GROUND_PUDDLE_EDGE = 0.012F;
static const float GROUND_DAMP_EDGE   = 0.2F;
// The ripples' longest wavelength, metres, and their RMS slope per axis at 1: a breeze over a
// sheltered puddle
static const float GROUND_RIPPLE_SCALE = 0.25F;
static const float GROUND_RIPPLE_SLOPE = 0.05F;
// How glossy damp ground gets, as perceptual roughness: water in its pores, no film on it
static const float GROUND_DAMP_ROUGHNESS = 0.35F;
// groundNoise's value variance and its gradient's per axis, measured (test_ground.cpp)
static const float GROUND_NOISE_VARIANCE       = 0.0311F;
static const float GROUND_NOISE_SLOPE_VARIANCE = 0.26F;

// A lattice point's gradient, each component in [-1, 1]: pcg2d (Jarzynski and Olano 2020)
float2 groundNoiseGradient(int2 cell)
{
  uint2 v = uint2(cell) * 1664525u + 1013904223u;
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v ^= v >> 16u;
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v ^= v >> 16u;
  return float2(v) * (2.0F / 4294967295.0F) - 1.0F;
}

// Gradient noise at `p`, quintic, with its derivatives (Quilez): x the value, yz its gradient
float3 groundNoise(float2 p)
{
  const float2 cellCorner = floor(p);
  const int2   cell       = int2(cellCorner);
  const float2 f          = p - cellCorner;
  const float2 u          = f * f * f * (f * (f * 6.0F - 15.0F) + 10.0F);
  const float2 du         = 30.0F * f * f * (f * (f - 2.0F) + 1.0F);
  const float2 ga         = groundNoiseGradient(cell);
  const float2 gb         = groundNoiseGradient(cell + int2(1, 0));
  const float2 gc         = groundNoiseGradient(cell + int2(0, 1));
  const float2 gd         = groundNoiseGradient(cell + int2(1, 1));
  const float  va         = dot(ga, f);
  const float  vb         = dot(gb, f - float2(1.0F, 0.0F));
  const float  vc         = dot(gc, f - float2(0.0F, 1.0F));
  const float  vd         = dot(gd, f - float2(1.0F, 1.0F));
  const float  k          = va - vb - vc + vd;
  const float  value      = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
  const float2 gradient   = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
                          + du * (float2(u.y, u.x) * k + float2(vb - va, vc - va));
  return float3(value, gradient.x, gradient.y);
}

// How much of the ground at logit `z` the top `share` of the floor covers, with a logistic edge of
// width `edge`: none at share 0, all at 1, and where the edge is as wide as the field (a footprint
// that averages it all) `share` everywhere.
float groundCover(float z, float share, float edge)
{
  if(share <= 0.0F)
    return 0.0F;
  if(share >= 1.0F)
    return 1.0F;
  const float threshold = log((1.0F - share) / share);
  return 1.0F / (1.0F + exp(min((threshold - z) / edge, 80.0F)));
}

struct GroundWater
{
  float  damp;           // 0..1: the ground wet, darker (groundWetAlbedo) and glossier
  float  puddle;         // 0..1: under standing water, smooth but for its ripples
  float2 slope;          // the water surface's slope along x and z, from its ripples
  float  slopeVariance;  // per axis: the ripples finer than the footprint, for the roughness
};

// The water at `xz` on the ground, for a ray footprint `footprint` metres wide. An octave of the
// field or the ripples as fine as the footprint fades out: the field's widens the edges, the
// ripples' goes into slopeVariance, so a far floor averages instead of aliasing.
GroundWater groundWaterAt(float2 xz, float footprint, float amount, float ripples)
{
  GroundWater water;
  water.damp          = 0.0F;
  water.puddle        = 0.0F;
  water.slope         = float2(0.0F, 0.0F);
  water.slopeVariance = 0.0F;
  if(amount <= 0.0F)
    return water;

  // Four octaves, each half as large and turned by 36.87 degrees, so their lattices never line up
  float2 axisX  = float2(1.0F, 0.0F);
  float2 axisZ  = float2(0.0F, 1.0F);
  float  height = 0.0F;
  float  lost   = 0.0F;  // the variance of the octaves the footprint covers
  float  total  = 0.0F;
  float  scale  = GROUND_PUDDLE_SCALE;
  float  weight = 1.0F;
  for(int k = 0; k < 4; ++k)
  {
    const float  kept = saturate(2.0F - 4.0F * footprint / scale);
    const float2 q    = float2(dot(axisX, xz), dot(axisZ, xz)) / scale + float2(0.37F + 5.1F * float(k), 0.71F - 3.3F * float(k));
    height += weight * kept * groundNoise(q).x;
    lost += weight * weight * (1.0F - kept * kept);
    total += weight * weight;
    const float2 turned = 0.8F * axisX - 0.6F * axisZ;
    axisZ               = 0.6F * axisX + 0.8F * axisZ;
    axisX               = turned;
    scale *= 0.5F;
    weight *= 0.5F;
  }
  // The field is about normal and the logistic about its CDF, so its logit is uniform over the
  // floor: a share of the floor is a threshold on it. What the footprint averaged out is an
  // uncertainty of the same scale on top of the edge.
  const float z    = 1.702F * height / sqrt(total * GROUND_NOISE_VARIANCE);
  const float blur = lost / total;
  water.damp       = groundCover(z, saturate(2.0F * amount), sqrt(GROUND_DAMP_EDGE * GROUND_DAMP_EDGE + blur));
  water.puddle     = groundCover(z, saturate((amount - 0.3F) / 0.7F), sqrt(GROUND_PUDDLE_EDGE * GROUND_PUDDLE_EDGE + blur));
  water.damp       = max(water.damp, water.puddle);

  if(water.puddle < 1e-3F || ripples <= 0.0F)
    return water;
  // Three octaves of equal slope from GROUND_RIPPLE_SCALE down, each gradient turned back into
  // world axes
  const float slopeWeight = ripples * GROUND_RIPPLE_SLOPE / sqrt(3.0F * GROUND_NOISE_SLOPE_VARIANCE);
  axisX                   = float2(1.0F, 0.0F);
  axisZ                   = float2(0.0F, 1.0F);
  scale                   = GROUND_RIPPLE_SCALE;
  for(int k = 0; k < 3; ++k)
  {
    const float  kept = saturate(2.0F - 4.0F * footprint / scale);
    const float2 q    = float2(dot(axisX, xz), dot(axisZ, xz)) / scale + float2(-2.9F + 7.3F * float(k), 4.3F + 1.9F * float(k));
    const float3 n    = groundNoise(q);
    water.slope += slopeWeight * kept * (axisX * n.y + axisZ * n.z);
    water.slopeVariance += slopeWeight * slopeWeight * GROUND_NOISE_SLOPE_VARIANCE * (1.0F - kept * kept);
    const float2 turned = 0.8F * axisX - 0.6F * axisZ;
    axisZ               = 0.6F * axisX + 0.8F * axisZ;
    axisX               = turned;
    scale *= 0.4F;
  }
  water.slope *= water.puddle;
  return water;
}

// A diffuse surface under water (Lekner and Dorf 1988): the water's underside sends back r of the
// light the surface scatters up, most of it by total internal reflection, for another bounce, so
// what comes out is a / (n^2 (1 - a r)) over what the water's surface lets in, r = 1 - (1 - re) / n^2
// with re = 0.066 the surface's diffuse reflectance from above. The darker a surface, the more it
// darkens: 20% grey goes to 12.5%. Never brighter than dry (near white the formula passes 1).
float3 groundWetAlbedo(float3 dry)
{
  const float n2       = GROUND_WATER_IOR * GROUND_WATER_IOR;
  const float internal = 1.0F - (1.0F - 0.066F) / n2;
  return min(dry, dry / (n2 * (float3(1.0F, 1.0F, 1.0F) - dry * internal)));
}

#endif  // GROUND_H_SLANG
