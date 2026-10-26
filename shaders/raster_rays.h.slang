#ifndef RASTER_RAYS_H
#define RASTER_RAYS_H

// The real-time view's traced rays: shadows of the lights the shadow map does not cover and ambient
// occlusion of the environment light, through the path tracer's opacity. Shared by the occlusion
// pass (raster_occlusion.slang), which traces them once per pixel, and the raster's fragment shader
// (gltf_raster.slang), which traces them only for what that pass does not see (blended and cutout
// surfaces). Reads the including shader's g_tlas, and what surface_opacity.h.slang reads.

#include "get_hit.h.slang"
#include "nvshaders/functions.h.slang"  // orthonormalBasis

// How far the traced ambient occlusion reaches, in metres
static const float AO_RADIUS_SURFACE = 0.3;  // the figure: folds, the chin, under the arms
static const float AO_RADIUS_FLOOR   = 1.0;  // the floor: the figure's contact shadow

// Interleaved gradient noise (Jimenez 2014): a per-pixel rotation that turns the filter's banding
// into fine grain.
float interleavedGradientNoise(float2 pixel)
{
  return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

// The i-th of n points of a Vogel (golden-angle) disk of radius 1, turned by `phi`.
float2 vogelDisk(int i, int n, float phi)
{
  const float r     = sqrt((float(i) + 0.5) / float(n));
  const float theta = float(i) * 2.39996323 + phi;
  return r * float2(cos(theta), sin(theta));
}

// How much of a ray from `origin` along `dir` gets past the scene within `maxT`: 0 at an opaque hit,
// the product of (1 - opacity) over the alpha-tested ones (lash and hair cards, cutouts) otherwise,
// so their shadows are soft and need no noise. The path tracer's any-hit opacity (getOpacity).
// A ray through a hair's cards meets dozens of them, each an opacity fetch: the walk stops once
// less than 1/64 gets through (as blocked), or after RAY_VISIBILITY_MAX_CANDIDATES cards (with what
// has got through so far), so a hair's fragments cost a bounded number of fetches per ray.
static const int RAY_VISIBILITY_MAX_CANDIDATES = 8;

// How much of the light a hit stops: the path tracer's opacity, less what a transmitting surface
// (KHR_materials_transmission: a cornea, the eye's moisture) lets through, as the sun's shadow map
// takes it (fragmentShadowMain). Without that, the eye inside its shell saw nothing but the shell
// and went black.
float rayOpacity(uint instance, uint primitive, uint triangle, float2 b)
{
  const GltfRenderNode renderNode = pushConst.gltfScene->renderNodes[instance];
  const float transmission = pushConst.gltfScene->materials[max(0, renderNode.materialID)].transmissionFactor;
  return getOpacity(renderNode, pushConst.gltfScene->renderPrimitives[primitive], int(triangle), float3(1.0 - b.x - b.y, b.x, b.y))
         * (1.0 - transmission);
}

float rayVisibility(float3 origin, float3 dir, float maxT)
{
  RayDesc ray;
  ray.Origin    = origin;
  ray.Direction = dir;
  ray.TMin      = 0.0;
  ray.TMax      = maxT;
  RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
  // every hit a candidate, so a transmitting surface built opaque into the BLAS is tested too; the
  // first one that stops the ray still ends the search
  q.TraceRayInline(g_tlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);
  float transmit   = 1.0;
  int   candidates = 0;
  while(q.Proceed())
  {
    if(q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
      continue;
    const float opacity = rayOpacity(q.CandidateInstanceIndex(), q.CandidateInstanceID(), q.CandidatePrimitiveIndex(),
                                     q.CandidateTriangleBarycentrics());
    if(opacity >= 0.999)
    {
      q.CommitNonOpaqueTriangleHit();  // the search ends here (RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH)
      continue;
    }
    transmit *= 1.0 - opacity;
    if(transmit < 1.0 / 64.0)
    {
      q.Abort();
      return 0.0;
    }
    if(++candidates >= RAY_VISIBILITY_MAX_CANDIDATES)
    {
      q.Abort();
      return transmit;
    }
  }
  return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : transmit;
}

// A light's shadow traced against the scene: `rays` rays over the light's sphere (point, spot:
// GltfLight.radius) or the sun's disc (directional: its angular size), a Vogel disk turned per
// pixel, so the penumbra is the one the path tracer gives. From the hit's terminator-safe point,
// stepped off the surface on the light's side by the hit's own rounding bound.
float rayTracedShadow(int lightIndex, GltfLight light, HitState hit, float3 geoNormal, float2 pixel, int rays)
{
  float3 toLight;
  float  maxT, tanHalf;
  if(light.type == LightType::eLightTypeDirectional)
  {
    toLight = -light.direction;
    maxT    = 1e30;
    tanHalf = tan(0.5 * light.angularSizeOrInvRange);
  }
  else
  {
    const float3 d    = light.position - hit.pos;
    const float  dist = length(d);
    if(dist <= light.radius)
      return 1.0;
    toLight = d / dist;
    maxT    = dist - light.radius;
    tanHalf = light.radius / dist;
  }
  const float3 n      = dot(geoNormal, toLight) < 0.0 ? -geoNormal : geoNormal;
  const float3 origin = (dot(hit.nrm, toLight) > 0.0 ? hit.shadowPos : hit.pos) + n * max(2.0 * hit.offsetDistance, 1e-5);
  float3       t, b;
  orthonormalBasis(toLight, t, b);
  const float phi = interleavedGradientNoise(pixel + float2(float(lightIndex) * 7.0, 0.0)) * 6.28318531;
  float       lit = 0.0;
  for(int i = 0; i < rays; ++i)
  {
    const float2 o = vogelDisk(i, rays, phi) * tanHalf;
    lit += rayVisibility(origin, normalize(toLight + o.x * t + o.y * b), maxT);
  }
  return lit / float(rays);
}

// How open the hemisphere over `N` is within `radius`: `rays` cosine-distributed rays on a Vogel
// disk turned per pixel, alpha-tested hits letting (1 - opacity) through.
float rayTracedOcclusion(HitState hit, float3 N, float radius, float2 pixel, int rays)
{
  if(rays <= 0)
    return 1.0;
  const float3 n      = dot(hit.geonrm, N) < 0.0 ? -hit.geonrm : hit.geonrm;
  const float3 origin = hit.pos + n * max(2.0 * hit.offsetDistance, 1e-5);
  float3       t, b;
  orthonormalBasis(N, t, b);
  const float phi  = interleavedGradientNoise(pixel + float2(3.0, 11.0)) * 6.28318531;
  float       open = 0.0;
  for(int i = 0; i < rays; ++i)
  {
    // a point on the unit disk lifted onto the hemisphere: cosine-distributed directions
    const float2 d   = vogelDisk(i, rays, phi);
    const float3 dir = d.x * t + d.y * b + sqrt(saturate(1.0 - dot(d, d))) * N;
    // a direction under the geometry (a normal-mapped N) sees the surface itself (an if: ?: would
    // trace it anyway)
    if(dot(dir, n) > 0.0)
      open += rayVisibility(origin, dir, radius);
  }
  return open / float(rays);
}

// The occlusion pass's image (Rasterizer::traceOcclusion), two layers per half-resolution texel:
// layer 0 (ambient occlusion, distance from the eye, shadows of lights 0 and 1), layer 1 (shadows
// of lights 2 to 5). A distance of 0 marks a texel with no surface on it.
static const int RASTER_OCCLUSION_LIGHTS = 6;

// The shadow of light `i` from the pass's two layers
float rasterOcclusionShadow(float4 layer0, float4 layer1, int i)
{
  return i < 2 ? (i == 0 ? layer0.z : layer0.w) : layer1[i - 2];
}

// Octahedral normal encoding in 32 bits (16 per axis), for the prepass's G-buffer
uint packOctNormal(float3 n)
{
  n /= abs(n.x) + abs(n.y) + abs(n.z);
  float2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
  const uint2 q = uint2(round(saturate(e * 0.5 + 0.5) * 65534.0));  // 65535 never: 0xFFFFFFFF is "no surface"
  return q.x | (q.y << 16);
}

float3 unpackOctNormal(uint p)
{
  const float2 e = float2(float(p & 0xFFFFu), float(p >> 16)) / 65534.0 * 2.0 - 1.0;
  float3       n = float3(e, 1.0 - abs(e.x) - abs(e.y));
  if(n.z < 0.0)
    n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
  return normalize(n);
}

static const uint RASTER_GBUFFER_EMPTY = 0xFFFFFFFFu;

#endif
