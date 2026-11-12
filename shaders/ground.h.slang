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

// A water surface's slope at a point, as a footprint sees it
struct GroundSlope
{
  float2 slope;     // along x and z
  float  variance;  // per axis: the detail finer than the footprint, for the roughness
};

// The ripples at `xz`, for a footprint `footprint` metres wide: three octaves of equal slope from
// GROUND_RIPPLE_SCALE down, each gradient turned back into world axes. An octave as fine as the
// footprint fades out of the slope and into the variance.
GroundSlope groundRipplesAt(float2 xz, float footprint, float ripples)
{
  GroundSlope s;
  s.slope    = float2(0.0F, 0.0F);
  s.variance = 0.0F;
  if(ripples <= 0.0F)
    return s;
  const float weight = ripples * GROUND_RIPPLE_SLOPE / sqrt(3.0F * GROUND_NOISE_SLOPE_VARIANCE);
  float2      axisX  = float2(1.0F, 0.0F);
  float2      axisZ  = float2(0.0F, 1.0F);
  float       scale  = GROUND_RIPPLE_SCALE;
  for(int k = 0; k < 3; ++k)
  {
    const float  kept = saturate(2.0F - 4.0F * footprint / scale);
    const float2 q    = float2(dot(axisX, xz), dot(axisZ, xz)) / scale + float2(-2.9F + 7.3F * float(k), 4.3F + 1.9F * float(k));
    const float3 n    = groundNoise(q);
    s.slope += weight * kept * (axisX * n.y + axisZ * n.z);
    s.variance += weight * weight * GROUND_NOISE_SLOPE_VARIANCE * (1.0F - kept * kept);
    const float2 turned = 0.8F * axisX - 0.6F * axisZ;
    axisZ               = 0.6F * axisX + 0.8F * axisZ;
    axisX               = turned;
    scale *= 0.4F;
  }
  return s;
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

  if(water.puddle < 1e-3F)
    return water;
  const GroundSlope rippled = groundRipplesAt(xz, footprint, ripples);
  water.slope               = rippled.slope * water.puddle;
  water.slopeVariance       = rippled.variance;
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

//-----------------------------------------------------------------------
// A body of water over the ground
//-----------------------------------------------------------------------
// The ground's water mode: a surface `waterDepth` above the floor, its gentle waves a heightfield
// the rays meet (groundWaterCrossing), the ripples on them in its normal only. Under it light is
// absorbed as by pure water, the red first, so depth turns the floor and what stands in it cyan and
// then blue, and scattered as by pure seawater, so deep water past the floor glows blue in the sun
// and the clouds' shadows darken it (groundWaterInScatter). A path's ray refracts at the surface; a shadow ray crosses it straight, losing its
// Fresnel share, as through clear glass, and from under it takes the caustic the surface focuses
// there (groundWaterShadowTransmittance, groundWaterCaustic).

// Pure water's absorption, per metre, over the bands of the sRGB primaries (Pope and Fry 1997: about
// 630, 550 and 470 nm)
static const float3 GROUND_WATER_ABSORPTION = float3(0.29F, 0.055F, 0.011F);
// Pure seawater's scattering, per metre, over the same bands: Morel's 0.00288 at 500 nm, as the
// wavelength to the -4.32. Deep water is blue by it: of light straight down it sends back about 4.6% of
// the blue, 0.6% of the green and 0.07% of the red (once scattered; groundWaterInScatter), the open
// sea's colour. Its phase function is the molecules' (groundWaterPhase).
static const float3 GROUND_WATER_SCATTERING = float3(0.00106F, 0.00191F, 0.00376F);
static const float3 GROUND_WATER_EXTINCTION = GROUND_WATER_ABSORPTION + GROUND_WATER_SCATTERING;

// Seawater's phase function, per steradian: the molecules', (1 + 0.835 cos^2) normalised (Morel 1974)
float groundWaterPhase(float cosTheta)
{
  return 0.06225F * (1.0F + 0.835F * cosTheta * cosTheta);
}
static const float  GROUND_TWO_PI           = 6.28318531F;
// The swell at Waves 1: four trains of one steepness (their slope's amplitude) from a wind
// GROUND_SWELL_WIND radians round from +x, each with its wavelength (metres), heading off the wind
// (radians) and phase; and two octaves of noise, GROUND_SWELL_NOISE metres across and as high as
// GROUND_SWELL_NOISE_HEIGHT times steepness times that, that keep the trains from looking ruled.
// About 6 cm from trough to crest at most, RMS slope 0.06.
static const float GROUND_SWELL_STEEPNESS    = 0.04F;
static const float GROUND_SWELL_WIND         = 0.52F;
static const float GROUND_SWELL_LENGTH[4]    = {3.1F, 1.9F, 1.3F, 0.83F};
static const float GROUND_SWELL_HEADING[4]   = {0.0F, 0.4F, -0.55F, 0.82F};
static const float GROUND_SWELL_PHASE[4]     = {0.0F, 1.7F, 4.1F, 2.6F};
static const float GROUND_SWELL_NOISE[2]     = {1.1F, 0.45F};
static const float GROUND_SWELL_NOISE_HEIGHT = 0.35F;
// groundNoise never leaves -1..1 (its gradients' components do not)
static const float GROUND_NOISE_BOUND = 1.0F;

struct GroundWaves
{
  float       height;  // above the water's level, with every train and octave
  GroundSlope slope;   // as the footprint sees it
};

// The swell at `xz`, for a footprint `footprint` metres wide. A train or octave as fine as the
// footprint fades out of the slope and into its variance; the height keeps them all, since it is
// the surface the rays meet.
GroundWaves groundWavesAt(float2 xz, float footprint, float waves)
{
  GroundWaves w;
  w.height         = 0.0F;
  w.slope.slope    = float2(0.0F, 0.0F);
  w.slope.variance = 0.0F;
  if(waves <= 0.0F)
    return w;
  const float steepness = GROUND_SWELL_STEEPNESS * waves;
  for(int i = 0; i < 4; ++i)
  {
    const float  heading = GROUND_SWELL_WIND + GROUND_SWELL_HEADING[i];
    const float2 dir     = float2(cos(heading), sin(heading));
    const float  k       = GROUND_TWO_PI / GROUND_SWELL_LENGTH[i];
    const float  phase   = k * dot(dir, xz) + GROUND_SWELL_PHASE[i];
    const float  kept    = saturate(2.0F - 4.0F * footprint / GROUND_SWELL_LENGTH[i]);
    w.height += steepness / k * sin(phase);
    w.slope.slope += kept * steepness * cos(phase) * dir;
    // a train's slope varies along its heading only: half its square there, a quarter per axis
    w.slope.variance += (1.0F - kept * kept) * 0.25F * steepness * steepness;
  }
  for(int j = 0; j < 2; ++j)
  {
    const float  scale  = GROUND_SWELL_NOISE[j];
    const float  height = GROUND_SWELL_NOISE_HEIGHT * steepness * scale;
    const float3 n      = groundNoise(xz / scale + float2(11.3F + 6.1F * float(j), -5.7F + 2.3F * float(j)));
    const float  kept   = saturate(2.0F - 4.0F * footprint / scale);
    w.height += height * n.x;
    w.slope.slope += kept * height / scale * float2(n.y, n.z);
    w.slope.variance += (1.0F - kept * kept) * (height / scale) * (height / scale) * GROUND_NOISE_SLOPE_VARIANCE;
  }
  return w;
}

// How far the waves reach above and below the level: all their crests at once, which bounds them
float groundWaveBound(float waves)
{
  const float steepness = GROUND_SWELL_STEEPNESS * max(waves, 0.0F);
  float       bound     = 0.0F;
  for(int i = 0; i < 4; ++i)
    bound += steepness * GROUND_SWELL_LENGTH[i] / GROUND_TWO_PI;
  for(int j = 0; j < 2; ++j)
    bound += GROUND_NOISE_BOUND * GROUND_SWELL_NOISE_HEIGHT * steepness * GROUND_SWELL_NOISE[j];
  return bound;
}

struct GroundWaterCrossing
{
  float t;          // along the ray; negative: it meets no water
  bool  fromAbove;  // it comes down onto the water from the air
};

// The ray's height above the water's surface at t
float groundHeightAboveWater(float3 origin, float3 direction, float t, float level, float waves)
{
  return origin.y + t * direction.y - level - groundWavesAt(float2(origin.x + t * direction.x, origin.z + t * direction.z), 0.0F, waves).height;
}

// Where a ray first meets the water surface, the heightfield over `level`, between 0 and tMax: in
// the band the waves keep to, stepped across for its first change of side and closed by the
// Illinois false position. A grazing ray can skim a crest and cross twice in the band; steps of
// 15 cm across the water, a third of its shortest swell, find the first (at most 8: a ray within a
// degree or so of the horizontal, far out, may still miss a crest).
GroundWaterCrossing groundWaterCrossing(float3 origin, float3 direction, float tMax, float level, float waves)
{
  GroundWaterCrossing c;
  c.t         = -1.0F;
  c.fromAbove = false;
  if(abs(direction.y) < 1e-6F)
    return c;
  // A still surface's band is a hair thick, so a flat level brackets like a wavy one
  const float bound   = groundWaveBound(waves) + 1e-4F;
  const float tTop    = (level + bound - origin.y) / direction.y;
  const float tBottom = (level - bound - origin.y) / direction.y;
  float       ta      = max(0.0F, min(tTop, tBottom));
  float       tb      = min(tMax, max(tTop, tBottom));
  if(!(ta < tb))
    return c;
  float       fa    = groundHeightAboveWater(origin, direction, ta, level, waves);
  float       fb    = fa;
  const float span  = (tb - ta) * sqrt(direction.x * direction.x + direction.z * direction.z);
  const int   steps = clamp(int(ceil(span / 0.15F)), 1, 8);
  const float end   = tb;
  bool        found = false;
  for(int s = 1; s <= steps && !found; ++s)
  {
    const float t = s == steps ? end : ta + (end - ta) * float(s) / float(steps);
    const float f = groundHeightAboveWater(origin, direction, t, level, waves);
    if((f > 0.0F) != (fa > 0.0F))
    {
      tb    = t;
      fb    = f;
      found = true;
    }
    else
    {
      ta = t;
      fa = f;
    }
  }
  if(!found)
    return c;
  c.fromAbove = fa > 0.0F;
  int   side  = 0;
  float t     = ta;
  for(int i = 0; i < 10; ++i)
  {
    t             = (ta * fb - tb * fa) / (fb - fa);
    const float f = groundHeightAboveWater(origin, direction, t, level, waves);
    if(abs(f) < 1e-6F)
      break;
    if((f > 0.0F) == (fb > 0.0F))
    {
      tb = t;
      fb = f;
      if(side == -1)
        fa *= 0.5F;
      side = -1;
    }
    else
    {
      ta = t;
      fa = f;
      if(side == 1)
        fb *= 0.5F;
      side = 1;
    }
  }
  c.t = t;
  return c;
}

// The share of light the water's surface reflects, unpolarised, for light meeting it from the air
// at `cosTheta` to its normal (or leaving it into the air there: the same share)
float groundWaterFresnel(float cosTheta)
{
  const float c     = saturate(cosTheta);
  const float n     = GROUND_WATER_IOR;
  const float sinT2 = (1.0F - c * c) / (n * n);
  const float cosT  = sqrt(max(1.0F - sinT2, 0.0F));
  const float rs    = (c - n * cosT) / (c + n * cosT);
  const float rp    = (n * c - cosT) / (n * c + cosT);
  return 0.5F * (rs * rs + rp * rp);
}

// Whether there is a body of water: the plane is a floor (not the catcher) with a depth over it
bool groundHasWater(SceneFrameInfo* frameInfo)
{
  return (frameInfo->flags & int(SceneFrameInfoFlags::eSceneUseInfinitePlane)) != 0
         && (frameInfo->flags & int(SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher)) == 0 && frameInfo->waterDepth > 0.0F;
}

float groundWaterLevel(SceneFrameInfo* frameInfo)
{
  return frameInfo->infinitePlaneDistance + frameInfo->waterDepth;
}

// The waves the water has: those asked for, but in shallow water no higher than keeps every trough
// a fifth of the depth off the floor
float groundWaterWaves(SceneFrameInfo* frameInfo)
{
  return min(frameInfo->waterWaves, 0.8F * frameInfo->waterDepth / groundWaveBound(1.0F));
}

// Whether `pos` is under the water's surface, waves and all
bool groundUnderWater(float3 pos, SceneFrameInfo* frameInfo)
{
  return groundHasWater(frameInfo)
         && pos.y < groundWaterLevel(frameInfo) + groundWavesAt(float2(pos.x, pos.z), 0.0F, groundWaterWaves(frameInfo)).height;
}

// The most a caustic brightens the light. Where the surface focuses it to a line the brightness has
// no bound for light from one direction; in nature the sun's width and the water's depth spread the
// line, and here the light samples of a pixel do, the sun disc being sampled, past this cut.
static const float GROUND_CAUSTIC_MAX = 8.0F;
// How deep light goes before an octave of the ripples (or the swell's noise) focuses it, over the
// octave's size, per unit of the ripples (waves): an octave as small as this times the depth the
// light has come down is left out of the bending, as a footprint leaves out what it averages. Past
// its focus an octave crosses its light into a web finer than itself, which one place of the surface
// per floor point cannot follow; left in, the floor got up to 66% more light than came in. Set so
// the landing never folds over at Waves and Ripples 1 (test_ground.cpp).
static const float GROUND_CAUSTIC_RIPPLE_REACH = 0.045F;
static const float GROUND_CAUSTIC_SWELL_REACH  = 0.045F;

// Light from the air going `down`, bent into the water through a surface of normal `n` (Snell; from
// the air there is no total reflection)
float3 groundIntoWater(float3 down, float3 n)
{
  const float eta   = 1.0F / GROUND_WATER_IOR;
  const float cosI  = max(-dot(down, n), 0.0F);
  const float sin2T = eta * eta * (1.0F - cosI * cosI);
  return eta * down + (eta * cosI - sqrt(max(1.0F - sin2T, 0.0F))) * n;
}

// The surface as light coming down to a depth sees it: the octaves too small to follow that far
// left out (GROUND_CAUSTIC_RIPPLE_REACH)
struct GroundCausticReach
{
  float waves, ripples;      // amounts
  float swellOut, rippleOut;  // footprints, metres
};

// Where light from `toLight`, coming in through the surface over `xz`, lands at height `y` under it:
// bent there by the swell's and the ripples' slope together, as a path's ray is (waterSurfaceMaterial),
// but for the octaves `reach` leaves out
float2 groundWaterLanding(float2 xz, float3 toLight, float y, float level, GroundCausticReach reach)
{
  const GroundWaves w     = groundWavesAt(xz, reach.swellOut, reach.waves);
  const float2      slope = w.slope.slope + groundRipplesAt(xz, reach.rippleOut, reach.ripples).slope;
  const float3      bent  = groundIntoWater(-toLight, normalize(float3(-slope.x, 1.0F, -slope.y)));
  return xz + float2(bent.x, bent.z) * ((level + w.height - y) / max(-bent.y, 1e-4F));
}

// How much more light from `toLight`, above the water, reaches `origin` under it than still water
// would let through: the waves and ripples focus it under a crest and spread it under a trough, the
// floor's caustics. The light that lands at `origin` came in where the surface bends it there, found
// from where still water would have let it in by steps towards that landing. Its brightness is the
// inverse of how a patch of the surface there grows on its way down: the landing's Jacobian, taken
// across a millimetre. With what has focused by that depth left out the landing does not fold, so
// each floor point has the one place it is lit from, and the floor keeps the light that came in.
float groundWaterCaustic(float3 origin, float3 toLight, float level, float waves, float ripples)
{
  if(waves <= 0.0F && ripples <= 0.0F)
    return 1.0F;
  const float2 p     = float2(origin.x, origin.z);
  const float3 still = groundIntoWater(-toLight, float3(0.0F, 1.0F, 0.0F));
  // how far the light comes down through the water
  const float        path = max(level - origin.y, 0.0F) / max(-still.y, 1e-4F);
  GroundCausticReach reach;
  reach.waves     = waves;
  reach.ripples   = ripples;
  reach.swellOut  = GROUND_CAUSTIC_SWELL_REACH * waves * path;
  reach.rippleOut = GROUND_CAUSTIC_RIPPLE_REACH * ripples * path;
  float2 entry    = p - float2(still.x, still.z) * path;
  for(int i = 0; i < 3; ++i)
    entry += p - groundWaterLanding(entry, toLight, origin.y, level, reach);
  const float  h   = 1e-3F;
  const float2 at  = groundWaterLanding(entry, toLight, origin.y, level, reach);
  const float2 dx  = (groundWaterLanding(entry + float2(h, 0.0F), toLight, origin.y, level, reach) - at) / h;
  const float2 dz  = (groundWaterLanding(entry + float2(0.0F, h), toLight, origin.y, level, reach) - at) / h;
  const float  det = abs(dx.x * dz.y - dx.y * dz.x);
  return 1.0F / max(det, 1.0F / GROUND_CAUSTIC_MAX);
}

// What a shadow ray keeps of its light through the water: its Fresnel share where it crosses the
// surface, which it does straight, what the water absorbs along the part of it under the surface, and
// from under the water to a light above, the caustic there (groundWaterCaustic). The share is the one
// light from the air meets at the ray's angle to the level: it is that light, bent on the way in,
// that the ray stands for. Whether the ray starts under water is the surface's, waves and all, as for
// the path's own rays: a leg under a crest is in the water.
float3 groundWaterShadowTransmittance(float3 origin, float3 direction, float tMax, SceneFrameInfo* frameInfo)
{
  const float3 clear = float3(1.0F, 1.0F, 1.0F);
  if(!groundHasWater(frameInfo))
    return clear;
  const float level = groundWaterLevel(frameInfo);
  const float waves = groundWaterWaves(frameInfo);
  // Over every crest and not going down, as most shadow rays are: in the air all the way
  if(origin.y > level + groundWaveBound(waves) && direction.y >= 0.0F)
    return clear;
  const bool                under    = groundUnderWater(origin, frameInfo);
  const GroundWaterCrossing crossing = groundWaterCrossing(origin, direction, tMax, level, waves);
  if(!under && crossing.t < 0.0F)
    return clear;  // the ray stays in the air
  // how far the ray runs in the water: to the surface, or from it to the light; all of it when the
  // light is in the water too
  const float wet     = under ? (crossing.t >= 0.0F ? crossing.t : tMax) : tMax - crossing.t;
  const float through = crossing.t >= 0.0F ? 1.0F - groundWaterFresnel(abs(direction.y)) : 1.0F;
  const float focus   = under && crossing.t >= 0.0F && direction.y > 0.0F
                            ? groundWaterCaustic(origin, direction, level, waves, frameInfo->groundRipples)
                            : 1.0F;
  return through * focus * exp(-GROUND_WATER_EXTINCTION * wet);
}

#endif  // GROUND_H_SLANG
