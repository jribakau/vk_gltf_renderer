/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PATHTRACE_FUNCTIONS_H
#define PATHTRACE_FUNCTIONS_H 1

#include "gltf_eval_config.h"
#include "gltf_light_ies.h.slang"
#include "camera_ray.h.slang"
#include "sky_background.h.slang"
#include "ground.h.slang"
#include "light_pick.h.slang"
#include "walk_guide.h.slang"
#include "light_tree.h.slang"

//-----------------------------------------------------------------------
// Types and tuning constants
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// Direct light structure
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// Light pick probabilities at a shading vertex (light_pick.h.slang)
//-----------------------------------------------------------------------
// The share of next-event estimation's samples that went, at one vertex, to each light a BSDF ray
// can also reach. They depend on the vertex, and that vertex's BSDF ray is weighed against them on
// the next bounce, so the path carries them (PathTracerState::lightPick).
struct LightPick
{
  float env;       // the environment
  float sun;       // the sky's sun (0 without one, or below this vertex's horizon)
  float emissive;  // the emissive triangles together
  // Where the pick was made, for a sized light's share there (sizedLightNeePdf): the vertex, and the
  // normal it culled lights by (0: lit from both sides, nothing culled)
  float3 pos;
  float3 normal;
};

struct DirectLight
{
  float3    direction;        // Direction to the light
  float3    radianceOverPdf;  // Radiance over pdf
  float     distance;         // Distance to the light
  float     pdf;              // Probability of sampling this light
  LightPick pick;             // What the pick gave each light at this vertex, for the BSDF side's MIS
};

static const float MIN_TRANSMISSION = 0.01f;  // Minimum shadow transmission factor for transparent materials
// A clear surface this smooth (GGX alpha; perceptual roughness 0.1, the cap the figure's lowering puts
// on refracting surfaces) passes shadow rays straight through, and a path's ray that refracts through
// it is a null event for MIS (pathTraceOneBounce). A rougher one blocks shadow rays: the light behind
// it is the BSDF ray's alone. Both sides read the material's own roughness, never the regularised one.
static const float CLEAR_CROSSING_ALPHA = 0.01f;
static const float ANTIALIASING_STANDARD_DEVIATION = 0.4246609f;  // Standard deviation for antialiasing
static const int   RR_MIN_DEPTH                    = 3;           // Minimum depth for Russian Roulette
static const float VOLUME_RAND_FLOOR               = 1.0e-10f;    // Minimum random floor for volume rendering
static const float VOLUME_RAND_CEIL = 1.0f - 1.0e-7f;  // rand() can return exactly 1.0; keep u in the half-open [0,1) the bucket remap assumes
static const float RR_PCONT_FLOOR = 0.001f;  // Minimum probability for Russian Roulette
// In-volume Russian Roulette, then a hard stop. A walk has no bounce limit, so its scattering is
// bounded by an event budget. A dense, high-albedo medium keeps the throughput near 1, so a capped
// continuation probability (the 0.95 the surface loop once used) culls 5% of the fully energetic
// paths at every scatter and the medium renders far too dark. Following the throughput (cap 0.999)
// leaves those paths alone until VOLUME_HARD_CAP ends the walk.
// Without that stop an unbounded walk in a non-absorbing medium does not terminate (measured:
// device loss, not merely a slow frame). Both count the events of one walk: entering a medium starts
// the count again, so a path's second walk is not cut short by its first.
static const int   VOLUME_FREE_BUDGET  = 64;      // Scatter events of a walk before Russian Roulette starts culling it
static const int   VOLUME_HARD_CAP     = 256;     // Scatter events at which the walk is ended outright
static const float RR_PCONT_CAP_VOLUME = 0.999f;  // Maximum Russian Roulette probability inside a volume

//-----------------------------------------------------------------------
// Denoiser guide buffers (shared by DLSS, OptiX and OIDN)
//
// Guide-buffer scratch (GuideOutput / GuideScratch + the helpers further down + the
// per-bounce captures in pathTraceOneBounce) only exists when USE_GUIDE_SHADER is
// defined (set by CMake when USE_DLSS, USE_OPTIX_DENOISER or USE_OIDN_DENOISER is ON, or by the runtime
// scene-aware variant rebuild). When it's off we drop those scratch fields from
// PathTracerState and SampleResult to shrink per-thread live state.
//
// Two-tier split inside the guide block:
//   USE_GUIDE_SHADER : albedo + normalRoughness (+ motion/depth). Consumed by DLSS, OptiX
//                      and OIDN, so present whenever any denoiser guide is required.
//   USE_DLSS_SHADER  : specularAlbedo + specularHitDist (+ their per-bounce capture).
//                      DLSS-only; OptiX needs neither, so they are compiled out of the
//                      OptiX/no-denoiser variant to shrink live state further.
//
// Exception: hitPosition is promoted OUT of the guide-gated block because it is
// also consumed by the always-on NDC depth output (used for selection/picking on
// the first frame). Keeping it gated would break picking when DLSS/OptiX are off.
//-----------------------------------------------------------------------
#if USE_GUIDE_SHADER
struct GuideOutput
{
  float16_t4 albedo          = float16_t4(0);
  float16_t4 normalRoughness = float16_t4(0);
#if USE_DLSS_SHADER
  float16_t3 specularAlbedo  = float16_t3(0);
  float      specularHitDist = 0.0f;  // Distance to reflected surface (0=no reflection, 65504=sky reflection)
  float16_t4 nrMask          = float16_t4(1, 1, 1, 1);
  float16_t  responsivity    = float16_t(0);  // RR's responsivity mask (eDlssResponsivity)
  bool       psrValid        = false;      // The primary hit is clear glass with a surface behind it
  float3     psrPos          = float3(0);  // Virtual position of that surface (DLSS depth + motion)
#endif                                     // USE_DLSS_SHADER
};

//-----------------------------------------------------------------------
// DLSS guide temps accumulated during pathTrace (written to SampleResult at end).
//-----------------------------------------------------------------------
struct GuideScratch
{
  float16_t3 albedo          = float16_t3(0);
  float16_t4 normalRoughness = float16_t4(0);
#if USE_DLSS_SHADER
  float16_t3 specularAlbedo     = float16_t3(0);
  float16_t  specularHitDist    = float16_t(0);
  bool       captureSpecularHit = false;
  float16_t4 nrMask             = float16_t4(1, 1, 1, 1);
  float16_t  responsivity       = float16_t(0);  // See GuideOutput
  bool       psrValid           = false;      // See GuideOutput
  float3     psrPos             = float3(0);  // See GuideOutput
#endif                                        // USE_DLSS_SHADER
};
#endif  // USE_GUIDE_SHADER

//-----------------------------------------------------------------------
// This is the result of sampling a pixel
//-----------------------------------------------------------------------
struct SampleResult
{
  float4 radiance    = float4(0, 0, 0, 0);
  float3 hitPosition = 1e34f;  // World-space first-hit position; always present (drives NDC depth + DLSS motion).
#if USE_GUIDE_SHADER
  float3      hitPositionPrev = 1e34f;  // Previous-frame world-space position of the first-hit point (instance motion).
  GuideOutput guideOutput     = {};
#endif
};

//-----------------------------------------------------------------------
// Ray-cone footprint (Akenine-Moller et al. 2019).
//-----------------------------------------------------------------------
struct RayCone
{
  float width       = 0.0f;
  float spreadAngle = 0.0f;
};

// VolumeMedium and its helpers only exist when KHR_materials_volume is in use.
// When GLTF_USE_VOLUME=0 (the scene has no volume materials in optimal mode) the medium
// struct, the bounce-loop in-volume tracking, and the corresponding fields on
// PathTracerState are dropped entirely (reduces per-thread live registers).
#if GLTF_USE_VOLUME
// Stored as absorption + scattering rather than extinction + scattering. The two forms are
// equivalent in exact arithmetic, but a dense, high-albedo medium has sigma_a as a tiny
// difference of two large numbers (sigma_t = 2712, sigma_s = 2711.6), and recovering it by
// subtracting stored values destroys it under limited precision. Keeping sigma_a as its own
// field gives every derived quantity -- sigma_t and, more importantly, the single-scatter
// albedo -- full precision, with no cancellation. That matters here: a medium authored with
// multiscatterColor near white has an albedo like 0.9999, and recovering it as sigma_s / sigma_t
// from two lower-precision values can quantize it straight to 1.0 (originally observed with
// float16 storage; kept as float32 headroom against the same failure mode at higher albedos).
struct VolumeMedium
{
  float32_t3 absorption;          // sigma_a
  float32_t3 scatterCoefficient;  // sigma_s
  float32_t  scatterAnisotropy;   // g
};

VolumeMedium makeVolumeMedium(PbrMaterial pbrMat)
{
  // sigma_t is the total extinction (KHR_materials_volume) and sigma_s the scattering part of it
  // (KHR_materials_scatter), so the difference is the absorption. Done in full precision here.
  float3       sigmaT = volumeExtinctionCoefficient(pbrMat);
  float3       sigmaS = min(pbrMat.scatterCoefficient, sigmaT);
  VolumeMedium medium;
  medium.absorption         = float32_t3(sigmaT - sigmaS);
  medium.scatterCoefficient = float32_t3(sigmaS);
  medium.scatterAnisotropy  = float32_t(pbrMat.scatterAnisotropy);
  return medium;
}

float3 volumeExtinction(VolumeMedium medium)
{
  return float3(medium.absorption) + float3(medium.scatterCoefficient);
}

bool hasVolumeMedium(VolumeMedium medium)
{
  float3 extinction = volumeExtinction(medium);
  return max(max(extinction.x, extinction.y), extinction.z) > 0.0F;
}
#endif  // GLTF_USE_VOLUME


//-----------------------------------------------------------------------
// Ray utilities
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// The next float above / below `v` (PBRT's NextFloatUp / NextFloatDown).
//-----------------------------------------------------------------------
float nextFloatUp(float v)
{
  if(isinf(v) && v > 0.0F)
    return v;
  v = (v == 0.0F) ? 0.0F : v;  // -0 steps up like +0
  return asfloat(v >= 0.0F ? asuint(v) + 1u : asuint(v) - 1u);
}

float nextFloatDown(float v)
{
  if(isinf(v) && v < 0.0F)
    return v;
  v = (v == 0.0F) ? -0.0F : v;  // +0 steps down like -0
  return asfloat(v > 0.0F ? asuint(v) - 1u : asuint(v) + 1u);
}

//-----------------------------------------------------------------------
// Self-intersection offset (PBRT 6.8.6, OffsetRayOrigin). `offsetDistance` covers the rounding errors
// between the computed hit point and the surface, and the hardware's test of the next ray against it
// (HitState::offsetDistance); `offsetDirection` is the geometric normal, turned to the side the ray
// leaves on. Stepping that far puts the surface behind the ray, and rounding the result one float
// further away keeps the step from being lost to the addition's own rounding.
//-----------------------------------------------------------------------
float3 safeOffsetRay(float3 worldPosition, float3 offsetDirection, float offsetDistance)
{
  const float3 offset = offsetDistance * offsetDirection;
  float3       origin = worldPosition + offset;
  [unroll]
  for(int i = 0; i < 3; i++)
  {
    if(offset[i] > 0.0F)
      origin[i] = nextFloatUp(origin[i]);
    else if(offset[i] < 0.0F)
      origin[i] = nextFloatDown(origin[i]);
  }
  return origin;
}


//-----------------------------------------------------------------------
// Ray-cone footprint
//-----------------------------------------------------------------------

float rayConeWorldFootprint(RayCone cone, float hitT, float3 N, float3 V)
{
  float w = cone.width + cone.spreadAngle * hitT;
  return w / max(abs(dot(N, V)), 1e-3);
}


//-----------------------------------------------------------------------
// Alpha testing
//-----------------------------------------------------------------------

#include "surface_opacity.h.slang"  // getOpacity, shared with the raster's shadow rays

// Cells across one barycentric axis. A fixed grid, so the decision is a property of the
// triangle: a dolly does not reshuffle it, and motion vectors can follow it. On a card a few
// hundred pixels wide a cell is several pixels, wider than DLSS's half-pixel jitter except at a
// cell border. Finer bins flicker under that jitter; coarser bins clump a soft ramp.
//
// Tried and dropped (2026-10-22): hashed alpha testing (Wyman and McGuire 2017) with object-space
// cells one pixel wide, plus BLEND alpha sampled at the pixel's mip. Filtering turned the hair
// maps' crisp 0/1 strands into partial opacity, and pixel-sized cells turned that into per-pixel
// dither that Ray Reconstruction kept as grain; hair looked frizzy when still and shimmered no less
// in motion. Point-sampled hair texels are mostly 0 or 1, which the hash does not touch.
static const uint ALPHA_HASH_BINS = 64;

// Largest float below 1.
static const float ALPHA_U_MAX = 0.99999994f;

// Every alpha test keeps a hit when u < opacity, with u in [0, 1). An opaque or mask-on texel
// (opacity 1) is then always kept and a cut-out texel (opacity 0) never is, whichever ray asks.

// Hash of the hit in [0, 1): render node, triangle, and the two independent barycentrics
// (barycentrics.y and .z; .x is 1 - y - z) quantized onto ALPHA_HASH_BINS. Raw barycentric bits
// avalanche when jitter moves the hit by a fraction of a pixel, so they are quantized first. The
// render node, not the shared render primitive, so instances of one mesh cut independently. The
// top 24 bits are exact in a float, so the value never rounds up to 1 (rand() can).
float alphaHashU(uint renderNodeID, uint triangleID, float3 barycentrics)
{
  uint bx = min(uint(saturate(barycentrics.y) * float(ALPHA_HASH_BINS)), ALPHA_HASH_BINS - 1);
  uint by = min(uint(saturate(barycentrics.z) * float(ALPHA_HASH_BINS)), ALPHA_HASH_BINS - 1);
  uint h  = xxhash32(uint3(renderNodeID, triangleID, bx * ALPHA_HASH_BINS + by));
  return float(h >> 8) * (1.F / 16777216.F);
}

// Uniform in [0, 1) for the stochastic alpha test.
//
// While DLSS denoises this frame the value is alphaHashU and the path seed is left alone. That
// seed is xxhash32(pixel, frameCount), and with DLSS frameCount is the Halton index, so drawing
// from it re-rolls the edge every frame. Ray Reconstruction holds a decision that sticks to the
// card.
//
// Every other frame draws from the path seed, which advances it. A frozen hash averages to 0 or
// 1 across accumulated frames, never to the soft opacity. USE_DLSS_SHADER stays on for every
// frame while DLSS is enabled, including the frames OptiX accumulates, so ePtUseDlss — set only
// when DLSS denoises this frame — is the gate.
float stochasticOpacityU(uint renderNodeID, uint triangleID, float3 barycentrics, inout uint seed)
{
#if USE_DLSS_SHADER
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
    return alphaHashU(renderNodeID, triangleID, barycentrics);
#endif
  return min(rand(seed), ALPHA_U_MAX);
}

//-----------------------------------------------------------------------
// Shadow transmission
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// Sample a material texture at a shadow-ray hit (LOD 0: a shadow ray carries no footprint)
//-----------------------------------------------------------------------
float4 sampleShadowTexture(GltfRenderPrimitive renderPrim, GltfTextureInfo texInfo, uint3 indices, float3 barycentrics)
{
  float2 uv = getInterpolatedVertexTexCoord(renderPrim, texInfo.texCoord, indices, barycentrics);
  uv        = applyTextureTransform(texInfo, uv);
  return allImages[NonUniformResourceIndex(texInfo.index)].SampleLevel(allSamplers[NonUniformResourceIndex(texInfo.samplerIndex)],
                                                                       uv, 0.0f);
}

//-----------------------------------------------------------------------
// Calculate the transmission of a ray through a surface. The medium behind the surface is the
// caller's to apply (shadowMediumTransmittance); crossing a volume boundary flips `isInside`, and
// entering one sets `extinction` to the medium the ray travels through next.
//-----------------------------------------------------------------------
float3 getShadowTransmission(GltfRenderNode      renderNode,
                             GltfRenderPrimitive renderPrim,
                             int                 triangleID,
                             float3              barycentrics,
                             float4x3            worldToObject,
                             float3              rayDirection,
                             inout bool          isInside,
                             inout float3        extinction)
{
#if !GLTF_USE_TRANSMISSION
  return float3(0.0);
#else
  // Straight-through shadow transmission ignores refraction: exact through a thin-walled smooth
  // surface, an approximation through a thick one. The path's own ray crosses the same surfaces as a
  // null event for MIS, so the light behind them is counted once. When disabled the surface is
  // opaque to shadow rays; stochastic alpha in the callers is unaffected.
  if(!hasFlag(pushConst.flags, PathtracerFlags::ePtShadowTransmission))
    return float3(0.0);

  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  // Textures only scale the factors down, so a low factor is final without sampling.
  float tFactor = mat.transmissionFactor;
  if(tFactor <= MIN_TRANSMISSION)
  {
    return float3(0.0);
  }

  uint3 indices = getTriangleIndices(renderPrim, triangleID);

  // From here on, follow evaluateMaterial texel by texel, so the shadow ray agrees with the surface.
  if(isTexturePresent(mat.transmissionTexture))
    tFactor *= sampleShadowTexture(renderPrim, texInfos[mat.transmissionTexture], indices, barycentrics).r;

  // thicknessFactor alone selects thin-walled vs. volume; the thickness texture is a rasterizer
  // stand-in for the path length, which a traced ray measures (hitT).
  float thickness = 0.0;
#if GLTF_USE_VOLUME
  thickness = mat.thicknessFactor;
#endif

#if GLTF_USE_SCATTER
  // The preview flag shades a volumetric scatter material as thin-walled, so its shadow must too:
  // Beer-Lambert through the measured medium would blacken a lobe the camera path never enters.
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin) && mat.scatterStrengthFactor > 0.0)
    thickness = 0.0;
  // Volumetric scatter is not a window. A shadow ray that hits the shell from outside stops; one
  // already in the medium (the walk's next-event connection) leaves it carrying only the
  // Beer-Lambert the caller applied over its segment: tinting it by the base colour, Fresnel and
  // roughness was a second colour on top of the walk.
  if(thickness > 0.0 && mat.scatterStrengthFactor > 0.0)
  {
    if(!isInside)
      return float3(0.0);
    isInside = false;
    return float3(1.0);
  }
  // Thin-walled KHR_materials_scatter turns `scatterStrength` of the transmission into a diffuse
  // lobe (see evaluateMaterial), which a straight-through shadow ray must not pass. Nor its specular
  // rest: a path's ray through such a surface is a real vertex (no clear crossing), so light behind
  // it is the BSDF ray's.
  if(thickness <= 0.0 && mat.scatterStrengthFactor > 0.0)
    return float3(0.0);
#endif

  if(tFactor <= MIN_TRANSMISSION)
  {
    return float3(0.0);
  }

  // Only a smooth surface is straight-through (CLEAR_CROSSING_ALPHA), and only what is not metal
  float metallic = mat.pbrMetallicFactor;
  {
    float roughness = mat.pbrRoughnessFactor;
    if(isTexturePresent(mat.pbrMetallicRoughnessTexture))
    {
      float4 mr_sample = sampleShadowTexture(renderPrim, texInfos[mat.pbrMetallicRoughnessTexture], indices, barycentrics);
      roughness *= mr_sample.g;
      metallic *= mr_sample.b;
    }
    if(roughness * roughness > CLEAR_CROSSING_ALPHA)
      return float3(0.0);
  }

  float3 normal;
  {
    float3 v0 = getVertexPosition(renderPrim, indices.x);
    float3 v1 = getVertexPosition(renderPrim, indices.y);
    float3 v2 = getVertexPosition(renderPrim, indices.z);
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    normal    = normalize(cross(e1, e2));
    normal    = normalize(float3(mul(worldToObject, normal).xyz));
  }

  float cosTheta = abs(dot(rayDirection, normal));
#if GLTF_USE_IOR
  float fresnel = schlickFresnel(mat.ior, cosTheta);
#else
  float fresnel = schlickFresnel(1.5F, cosTheta);
#endif
  // What the surface reflects, as the BSDF has it: KHR_materials_specular weighs Fresnel (a cornea
  // under the eye's tear film reflects none), and a clearcoat and second lobe reflect their shares over that.
#if GLTF_USE_SPECULAR
  fresnel *= saturate(mat.specularFactor * max(mat.specularColorFactor.x, max(mat.specularColorFactor.y, mat.specularColorFactor.z)));
#endif
#if GLTF_USE_CLEARCOAT
  fresnel = 1.0F - (1.0F - fresnel) * (1.0F - mat.clearcoatFactor * schlickFresnel(1.5F, cosTheta)) * (1.0F - mat.specular2Factor * schlickFresnel(1.5F, cosTheta));
#endif

  float3 currentTransmission = tFactor * (1.0F - metallic) * mat.pbrBaseColorFactor.rgb;
  currentTransmission *= (1.0F - fresnel);

#if GLTF_USE_VOLUME
  if(thickness > 0.0)
  {
    if(!isInside)
    {
      // The TOTAL extinction sigma_t = sigma_a + sigma_s: the out-scattering portion is already
      // included, so the scatter coefficient must not be added on top (that would double-count it
      // and over-darken the ray). The same call as the path's medium (makeVolumeMedium), so the
      // shadow ray sees the medium the random walk does. CPU sanitizes attenuationDistance to
      // FLT_MAX for the spec-invalid value 0; dividing by FLT_MAX gives ~0 (= no attenuation).
      extinction = volumeExtinctionCoefficient(mat.attenuationColor, mat.attenuationDistance);
    }
    isInside = !isInside;
  }
#endif

  return currentTransmission;
#endif
}

// Beer-Lambert transmittance of a shadow-ray segment: every segment travelled inside a medium,
// whatever ends it -- a boundary, another surface, or the light itself.
float3 shadowMediumTransmittance(bool isInside, float3 extinction, float segmentLength)
{
  return isInside ? exp(-segmentLength * extinction) : float3(1.0);
}

// The last segment, from the last boundary (or the origin) to the end of a shadow ray that ends
// inside a medium: a light in a fog, or a surface and its light in the same medium. A ray to
// infinity cannot end inside a closed medium; one that thinks it does crossed an open mesh, and is
// left unattenuated rather than black.
float3 shadowTailTransmittance(RayDesc ray, bool isInside, float3 extinction, float lastHitT)
{
  return shadowMediumTransmittance(isInside && ray.TMax < INFINITE, extinction, ray.TMax - lastHitT);
}

// Whether a shadow ray crossing this surface enters or leaves a medium (getShadowTransmission's
// thickness test): the rest of the ray then travels through something else.
bool shadowSurfaceBoundsMedium(GltfRenderNode renderNode)
{
#if GLTF_USE_TRANSMISSION && GLTF_USE_VOLUME
  if(!hasFlag(pushConst.flags, PathtracerFlags::ePtShadowTransmission))
    return false;
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  const uint         matIndex  = max(0, renderNode.materialID);
#if GLTF_USE_SCATTER
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin) && materials[matIndex].scatterStrengthFactor > 0.0)
    return false;
#endif
  return materials[matIndex].thicknessFactor > 0.0;
#else
  return false;
#endif
}

// Below this a shadow ray's transmission is rouletted rather than carried on
static const float SHADOW_ROULETTE_BELOW = 0.05F;

// One surface a shadow ray meets: the medium segment before it, then what the surface lets
// through. A surface that only filters the light (alpha, thin glass) passes its expected
// transmission, (1 - opacity) + opacity * T, exact and noise-free: the stochastic test made lash
// and hair-card shadows noisy in stills and froze them into a stipple on DLSS frames (alphaHashU).
// One that bounds a medium is either crossed or not, as the rest of the ray depends on it, so it
// keeps the stochastic test. Returns false once the ray is blocked.
bool shadowCrossSurface(GltfRenderNode      renderNode,
                        GltfRenderPrimitive renderPrim,
                        uint                instanceID,
                        int                 triangleID,
                        float3              barycentrics,
                        float4x3            worldToObject,
                        float3              rayDirection,
                        float               hitT,
                        inout float3        transmission,
                        inout bool          isInside,
                        inout float3        extinction,
                        inout float         prevHitT,
                        inout uint          seed)
{
  const float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);
  if(opacity <= 0.0F)
    return true;
  const bool boundsMedium = shadowSurfaceBoundsMedium(renderNode);
  if(boundsMedium && stochasticOpacityU(instanceID, uint(triangleID), barycentrics, seed) >= opacity)
    return true;

  // The medium changes only at a boundary, so the segment travelled in it is charged there, from
  // the boundary before. Candidates are not guaranteed to arrive in spatial order: abs() keeps one
  // that comes back out of order from zeroing the segment (exact for a single convex crossing), and
  // charging only at boundaries keeps that approximation off the surfaces in between. A lash card
  // met before the skin's exit would otherwise charge the skin's extinction past the exit.
  if(boundsMedium)
  {
    transmission *= shadowMediumTransmittance(isInside, extinction, abs(hitT - prevHitT));
    prevHitT = hitT;
  }
  const float3 surface =
      getShadowTransmission(renderNode, renderPrim, triangleID, barycentrics, worldToObject, rayDirection, isInside, extinction);
  transmission *= boundsMedium ? surface : (1.0F - opacity) + opacity * surface;

  // Russian roulette instead of a cut: a ray through many faint layers is still unbiased
  const float largest = max(transmission.x, max(transmission.y, transmission.z));
  if(largest >= SHADOW_ROULETTE_BELOW)
    return true;
  if(largest <= 0.0F || rand(seed) * SHADOW_ROULETTE_BELOW >= largest)
  {
    transmission = float3(0.0F);
    return false;
  }
  transmission *= SHADOW_ROULETTE_BELOW / largest;
  return true;
}


//-----------------------------------------------------------------------
// Raytracer interface
//-----------------------------------------------------------------------

#include "raytracer_interface.h.slang"


//-----------------------------------------------------------------------
// Direct lighting and environment
//-----------------------------------------------------------------------

// Punctual lights the sampler can choose from: the scene's, plus the sky's own sun when the
// renderer is supplying one. Every count and every index has to agree with this, or the selection
// pdf stops matching what was actually sampled.
int punctualLightCount()
{
  return pushConst.gltfScene.numLights + (pushConst.frameInfo->rendererSunLight != 0 ? 1 : 0);
}

// Index into that set. The scene's lights come first, so the sky's sun is the last one.
GltfLight getPunctualLight(int index)
{
  if(index >= pushConst.gltfScene.numLights)
    return makeSkySunLight(pushConst.frameInfo);

  GltfLight light = pushConst.gltfScene.lights[index];
  applySkySunLight(light, index, pushConst.frameInfo);  // no-op unless this is the marked sun
  return light;
}

// Index of the sky's sun in that set, or -1: the renderer's own, else the scene's marked one. It is
// the one punctual light a BSDF ray can also reach, because its disk is drawn in the sky.
int skySunPunctualIndex()
{
  if(pushConst.frameInfo->rendererSunLight != 0)
    return pushConst.gltfScene.numLights;
  return pushConst.frameInfo->sunLightIndex;
}

// The environment's pick estimate (light_pick.h.slang), 0 where there is none to sample.
float environmentPickEstimate()
{
  SceneFrameInfo* frameInfo = pushConst.frameInfo;
  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseNoEnvironment))
    return 0.0F;
  return max(frameInfo->envPickEstimate * frameInfo->envIntensity, 0.0F);
}

float punctualPickEstimate(GltfLight light, float3 pos, float3 normal, bool cull)
{
  const bool directional = light.type == LightType::eLightTypeDirectional;
  return punctualLightPickEstimate(directional, light.type == LightType::eLightTypeSpot, light.position, light.direction,
                                   light.radius, max(light.color.x, max(light.color.y, light.color.z)) * light.intensity,
                                   directional ? 0.0F : light.angularSizeOrInvRange, light.innerAngle, light.outerAngle,
                                   pos, normal, cull);
}

// What a node of the light tree's cut (EmissiveGroup) gives a point (emissiveNodeEstimate)
float emissiveGroupPickEstimate(int group, float3 pos, float3 normal, bool cull)
{
  return emissiveNodeEstimate(pushConst.gltfScene.emissiveGroups[group], pos, normal, cull);
}

// What next-event estimation picks from at one vertex: the environment, the punctual lights and
// the light tree's cut (GltfScene::numEmissiveGroups) split it by their estimates. Without a tree,
// or past LIGHT_PICK_MAX_LIGHTS punctual lights, the emissive triangles take a fixed share by power
// instead (they have one estimate for all of them and no distance to put it at).
struct LightPickSet
{
  LightPick pick;
  float     restShare;    // 1 - pick.emissive: the environment's, the punctual lights' and the groups' together
  float     envEstimate;  // 0: no environment to pick
  float     total;        // the candidates' estimates summed
  int       candidates;   // the environment, the lights and the emissive groups that can light the vertex
  int       lights;       // of those, punctual lights
  bool      uniform;      // over LIGHT_PICK_MAX_LIGHTS lights: half to the environment, the lights evenly, none culled
  bool      groups;       // the light tree's cut nodes are candidates (not uniform, and emissiveTreeOn)
};

// `cull`: drop the lights below the plane of `normal` (not for a vertex lit from both sides).
LightPickSet lightPickSet(float3 pos, float3 normal, bool cull)
{
  LightPickSet s;
  const int numPunctual = punctualLightCount();
  const int sunIndex    = skySunPunctualIndex();
  s.envEstimate         = environmentPickEstimate();
  s.uniform             = numPunctual > LIGHT_PICK_MAX_LIGHTS;
  s.total               = s.envEstimate;
  s.lights              = 0;
  float sunEstimate     = 0.0F;
  s.groups              = !s.uniform && emissiveTreeOn();
  int emitters          = 0;
  if(s.uniform)
    s.lights = numPunctual;
  else
  {
    for(int i = 0; i < numPunctual; i++)
    {
      const float e = punctualPickEstimate(getPunctualLight(i), pos, normal, cull);
      if(e > 0.0F)
      {
        s.total += e;
        s.lights++;
      }
      if(i == sunIndex)
        sunEstimate = e;
    }
  }
  if(s.groups)
    for(int g = 0; g < pushConst.gltfScene.numEmissiveGroups; g++)
    {
      const float e = emissiveGroupPickEstimate(g, pos, normal, cull);
      if(e > 0.0F)
      {
        s.total += e;
        emitters++;
      }
    }
  const bool hasEnv = s.envEstimate > 0.0F;
  s.candidates      = s.lights + emitters + (hasEnv ? 1 : 0);

  // Each kind present gets as much as the emissive triangles: a half beside one, a third beside both
  const int kinds = (hasEnv ? 1 : 0) + (s.lights > 0 ? 1 : 0);
  s.pick.emissive = (!s.groups && pushConst.gltfScene.numEmissiveTriangles > 0) ? 1.0F / float(1 + kinds) : 0.0F;
  s.restShare     = 1.0F - s.pick.emissive;
  s.pick.pos      = pos;
  s.pick.normal   = cull ? normal : float3(0.0F);

  if(s.uniform)
  {
    const float envShare = hasEnv ? 0.5F : 0.0F;
    s.pick.env           = s.restShare * envShare;
    s.pick.sun           = (sunIndex >= 0) ? s.restShare * (1.0F - envShare) / float(numPunctual) : 0.0F;
  }
  else
  {
    s.pick.env = s.restShare * lightPickProbability(s.envEstimate, s.total, s.candidates);
    s.pick.sun = s.restShare * lightPickProbability(sunEstimate, s.total, s.candidates);
  }
  return s;
}

// A light's share of the rest at this vertex, `estimate` its pick estimate there.
float punctualPickShare(LightPickSet s, float estimate)
{
  if(s.uniform)
    return (s.envEstimate > 0.0F ? 0.5F : 1.0F) / float(s.lights);
  return lightPickProbability(estimate, s.total, s.candidates);
}

//-----------------------------------------------------------------------
// Emissive-mesh area lights (next-event estimation)
//
// Emissive triangles are sampled directly and MIS-combined with BSDF sampling, mirroring the
// environment-light MIS. Selection is POWER-WEIGHTED: triangle i is chosen with probability
// proportional to (world area * luminance), in O(1) via a Vose alias table. The key simplification
// is that the world area cancels in the solid-angle pdf:
//     pdf(i) = P(select i) * dist^2 / (area_i * cos) = luminance_i * dist^2 / (W_total * cos)
// so the MIS pdf for a BSDF ray landing on an emitter is still recovered from local hit geometry
// (its own radiance + W_total) with NO per-triangle lookup table. Triangles are stored as
// references, so vertices/transforms are read live here.
//-----------------------------------------------------------------------
// Defensive luminance: pull the emitter's luminance toward the scene's area-weighted mean so a dim
// emitter is never starved. Uses the shared Rec. 709 luminance() (nvshaders/functions.h.slang); the
// host weight (gltf_scene_vk.cpp uploadEmissiveTriangles) must use the same coefficients.
float emitterDefensiveLuminance(float3 radiance)
{
  float f = EMISSIVE_DEFENSIVE_FRACTION;
  return (1.0f - f) * luminance(radiance) + f * pushConst.gltfScene.emissiveMeanLum;
}

struct EmissiveSample
{
  float3 direction;  // from the shading point toward the sampled point on the emitter
  float  distance;   // shadow-ray TMax (stops just short of the emitter surface)
  float3 radiance;   // emitted radiance at the sampled point (a textured emitter's texel)
  float  pdf;        // solid-angle pdf of the sample; 0 when the sample is invalid
};

// The radiance material `materialIndex` was picked by among the emitters: emissiveFactor (strength
// included) times its emissive texture's mean (emissiveLightScale), as SceneVk::uploadEmissiveTriangles
// stored it. Not what it emits at any one point of a textured emitter; what the pick's pdf is made of.
float3 emitterSelectionRadiance(int materialIndex)
{
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  return max(materials[materialIndex].emissiveFactor, float3(0.0F)) * materials[materialIndex].emissiveLightScale;
}

// Select an emissive triangle of [first, first + n) proportional to its selection weight in O(1) via
// that range's Vose alias table. One uniform picks a bin; its fractional part chooses between that bin
// and its alias.
int selectEmissiveTriangleIn(int first, int n, float u01)
{
  float            scaled = u01 * float(n);
  int              bin    = min(int(scaled), n - 1);
  float            uf     = scaled - float(bin);  // reuse the fractional part as the second uniform
  EmissiveTriangle et     = pushConst.gltfScene.emissiveTriangles[first + bin];
  return (uf < et.aliasProb) ? first + bin : int(et.aliasIndex);
}

// Select an emissive triangle proportional to its selection weight among all of them: the one table,
// or with the light tree, a cut node by its weight, a leaf below it by weight, and then a triangle in
// it (the same distribution)
int selectEmissiveTriangle(float u01)
{
  const int groups = pushConst.gltfScene.numEmissiveGroups;
  if(groups <= 0)
    return selectEmissiveTriangleIn(0, pushConst.gltfScene.numEmissiveTriangles, u01);
  float     target = u01 * pushConst.gltfScene.emissiveTotalWeight;
  float     before = 0.0F;
  int       cut    = 0;
  for(int i = 0; i < groups; i++)
  {
    cut = i;
    if(target < before + pushConst.gltfScene.emissiveGroups[i].weight || i == groups - 1)
      break;
    before += pushConst.gltfScene.emissiveGroups[i].weight;
  }
  const float            u    = saturate((target - before) / max(pushConst.gltfScene.emissiveGroups[cut].weight, 1e-20F)) * 0.99999994F;
  const EmissiveTreeStep step = emissiveTreeDescendByWeight(pushConst.gltfScene.emissiveGroups, cut, u);
  const EmissiveGroup    leaf = pushConst.gltfScene.emissiveGroups[step.leaf];
  return selectEmissiveTriangleIn(int(leaf.first), int(leaf.count), step.u);
}

// The light tree's leaf drawn by render node `renderNodeID`; -1 for none
int emissiveLeafOf(uint renderNodeID)
{
  if(renderNodeID >= uint(pushConst.gltfScene.numEmissiveLeafOfNode))
    return -1;
  return pushConst.gltfScene.emissiveLeafOfNode[renderNodeID];
}

// The light tree is a candidate in the light pick (ePtLightTree, and the scene has one); otherwise
// the emissive triangles take the one table's fixed share by power
bool emissiveTreeOn()
{
  return hasFlag(pushConst.flags, PathtracerFlags::ePtLightTree) && pushConst.gltfScene.numEmissiveGroups > 0;
}

// Power-weighted sample of emissive triangle `emissiveIndex` as seen from `shadingPos`, at the
// uniform point `u`. `totalWeight`: the weight of the triangles it was picked among (all of them,
// or its group's).
EmissiveSample sampleEmissiveTriangle(int emissiveIndex, float3 shadingPos, float2 u, float totalWeight)
{
  EmissiveSample es = {float3(0, 0, 1), 0.0, float3(0), 0.0};

  EmissiveTriangle    et   = pushConst.gltfScene.emissiveTriangles[emissiveIndex];
  GltfRenderNode      node = pushConst.gltfScene.renderNodes[et.renderNodeID];
  GltfRenderPrimitive prim = pushConst.gltfScene.renderPrimitives[et.renderPrimID];
  uint3               idx  = getTriangleIndices(prim, int(et.triangleID));
  const float3        o0   = getVertexPosition(prim, idx.x);
  const float3        o1   = getVertexPosition(prim, idx.y);
  const float3        o2   = getVertexPosition(prim, idx.z);

  // Uniform barycentric sample over the triangle area, placed as getHitState places a hit
  const float  su0      = sqrt(u.x);
  const float  b1       = 1.0 - su0;
  const float  b2       = u.y * su0;
  const float3 bary     = float3(1.0 - b1 - b2, b1, b2);
  const float3 lightPos = mul(float4(o0 * bary.x + o1 * bary.y + o2 * bary.z, 1.0), node.objectToWorld).xyz;

  float3 toLight = lightPos - shadingPos;
  float  dist2   = dot(toLight, toLight);
  if(dist2 <= 1e-12)
    return es;
  float  dist = sqrt(dist2);
  float3 dir  = toLight / dist;

  const float3 we1     = mul(float4(o1 - o0, 0.0), node.objectToWorld).xyz;
  const float3 we2     = mul(float4(o2 - o0, 0.0), node.objectToWorld).xyz;
  float3       ng      = cross(we1, we2);
  float        twoArea = length(ng);
  if(twoArea <= 1e-12)
    return es;  // degenerate triangle
  float3 lightN = ng / twoArea;

  float cosLight = dot(lightN, -dir);
  if(et.doubleSided != 0)
    cosLight = abs(cosLight);
  if(cosLight <= 1e-6)
    return es;  // one-sided emitter facing away, or grazing

  float wtot = totalWeight;
  if(wtot <= 0.0)
    return es;

  es.direction = dir;
  // Stop short of the emitter by no more than the rounding of the sampled point (as a hit's offset,
  // HitState::offsetDistance) and of the shadow ray's end, shadingPos + dir * distance, so the ray
  // does not hit the emitter itself. The fixed centimetre it stopped short by let light through an
  // occluder that close: a lamp's housing, a panel's backing. Half the way for a grazing emitter.
  const float3 we3          = we2 - we1;
  const float  triangleSize = sqrt(max(dot(we1, we1), max(dot(we2, we2), dot(we3, we3))));
  const float  surfaceError = triangleOffsetDistance(o0, o1, o2, bary, node.objectToWorld, lightN, triangleSize)
                             + roundingGamma(8) * (dot(abs(lightN), abs(shadingPos)) + dist);
  es.distance = max(dist - surfaceError / cosLight, 0.5F * dist);
  es.radiance = et.radiance;
  // A textured emitter emits its texel there; it was picked by its mean (et.radiance), which the pdf keeps
  GltfShadeMaterial* materials   = pushConst.gltfScene->materials;
  const int          materialID = max(0, node.materialID);
  if(isTexturePresent(materials[materialID].emissiveTexture))
  {
    const GltfTextureInfo texInfo = pushConst.gltfScene->textureInfos[materials[materialID].emissiveTexture];
    es.radiance = max(materials[materialID].emissiveFactor, float3(0.0F)) * sampleShadowTexture(prim, texInfo, idx, bary).rgb;
  }
  es.radiance *= bsdfCoatEmission(materials[materialID].clearcoatFactor, materials[materialID].clearcoatRoughness,
                                  materials[materialID].pbrBaseColorFactor.rgb, materials[materialID].pbrMetallicFactor,
                                  materials[materialID].scatterStrengthFactor > 0.0F ? 0.0F : materials[materialID].transmissionFactor,
                                  materials[materialID].pbrRoughnessFactor * materials[materialID].pbrRoughnessFactor, cosLight);
  // A ray meets a see-through surface with the probability of its opacity there (the any-hit's
  // stochastic test), and so does what a light sample takes from it. A cutout's holes (a candle
  // flame's card, a lamp's pattern) emitted to light samples alone, so a masked panel lit a floor
  // 1.9x a panel with the same holes cut (tests/render/emitter_cutout.py).
  es.radiance *= getOpacity(node, prim, int(et.triangleID), bary);
  // Power-weighted solid-angle pdf; the triangle area cancels (see header). Uses the defensive
  // luminance so it matches the host selection weight.
  es.pdf = emitterDefensiveLuminance(et.radiance) * dist2 / (wtot * cosLight);
  return es;
}

// NEE solid-angle pdf (within the light technique) of sampling the hit emissive triangle in the hit
// direction. Power-weighted selection makes the area cancel, so this needs only the emitter's own
// radiance, the geometric normal, and the weight it was picked among (`totalWeight`: all of them, or
// its group's) -- no per-triangle lookup. 0 if not a valid emitter dir.
float emitterHitSolidAnglePdf(float3 radiance, float3 geoNormal, float3 rayDir, float hitDist, float totalWeight)
{
  float wtot = totalWeight;
  if(pushConst.gltfScene.numEmissiveTriangles <= 0 || wtot <= 0.0)
    return 0.0;
  // No sidedness gate needed: the caller counts a one-sided emitter's front faces only (emitsThisWay),
  // exactly the side the sampler samples. abs() gives the right cosine magnitude for both one-sided
  // (front) and double-sided (either face, sampler also uses abs) emitters.
  float cosLight = abs(dot(geoNormal, rayDir));  // geonrm is pre-flipped toward the ray
  if(cosLight <= 1e-6)
    return 0.0;
  return emitterDefensiveLuminance(radiance) * (hitDist * hitDist) / (wtot * cosLight);
}

// MIS weight for a BSDF ray that landed on an emissive triangle, weighted against the emissive-NEE
// technique. Mirrors computeEnvHitMisWeight; the competing NEE pdf is the emissive triangles' pick
// at the vertex the ray left (`pick`) times the emitter's solid-angle pdf. With the light tree, the
// share of the cut node above the leaf drawn by `renderNodeID` there, as sizedLightNeePdf recovers a
// sized light's, times the walk's probability from it down to that leaf (emissiveTreeClimb).
// `radiance`: the emitter's selection radiance (emitterSelectionRadiance), not its texel; black: it is
// not in the light list, and only BSDF rays reach it.
float computeEmitterHitMisWeight(float lastSamplePdf, LightPick pick, float3 radiance, float3 geoNormal, float3 rayDir,
                                 float hitDist, uint renderNodeID)
{
  if(lastSamplePdf == DIRAC || !(luminance(radiance) > 0.0f))
    return 1.0f;
  float emitterNeePdf = pick.emissive * emitterHitSolidAnglePdf(radiance, geoNormal, rayDir, hitDist, pushConst.gltfScene.emissiveTotalWeight);
  if(emissiveTreeOn() && punctualLightCount() <= LIGHT_PICK_MAX_LIGHTS)
  {
    const int leaf = emissiveLeafOf(renderNodeID);
    if(leaf < 0)
      return 1.0f;
    const bool             cull  = any(pick.normal != float3(0.0F));
    const EmissiveTreeWalk climb = emissiveTreeClimb(pushConst.gltfScene.emissiveGroups, leaf, pick.pos, pick.normal, cull);
    const LightPickSet     set   = lightPickSet(pick.pos, pick.normal, cull);
    const float            share = set.restShare * climb.prob
                        * lightPickProbability(emissiveGroupPickEstimate(climb.leaf, pick.pos, pick.normal, cull), set.total, set.candidates);
    emitterNeePdf = share
                    * emitterHitSolidAnglePdf(radiance, geoNormal, rayDir, hitDist, pushConst.gltfScene.emissiveGroups[leaf].weight);
  }
  return misPowerWeight(lastSamplePdf, emitterNeePdf);
}

// `normal` is used to reject punctual lights that sit below the surface's hemisphere: the pick
// gives them nothing (lightPickSet), so their share goes to the lights that can reach the point.
// `lightsFromBothSides` disables that rejection for shading points that receive light from the
// back hemisphere as well: volume scattering events, and surfaces with a diffuse transmission
// lobe (KHR_materials_diffuse_transmission, and the thin-walled mode of KHR_materials_scatter --
// a backlit leaf or sheet of paper is exactly the case those describe).
//
// `directLight.pick` is what the pick gave each light a BSDF ray can also reach, which that ray's MIS
// weight needs on the next bounce.
//
// `compensatedEnv`: a BSDF sample of this vertex is traced and weighed against this one, so the
// environment is sampled from the MIS-compensated table (EnvCompensatedCell): its bright part, the
// even rest left to the BSDF sample. Every other caller -- the last vertex, a volume scatter, the
// shadow catcher -- has no such partner and samples the whole environment.
void sampleLights(in float3       pos,
                  float3          normal,
                  in float3       worldRayDirection,
                  inout uint      seed,
                  out DirectLight directLight,
                  in bool         isVolumeSample      = false,
                  in bool         lightsFromBothSides = false,
                  in bool         compensatedEnv      = false,
                  in float4       xi                  = float4(-1.0F))
{
  // `xi`: the sample's random numbers (x picks the light, yz place the sample, w is the
  // environment's third and the emitter triangle's pick), e.g. a Sobol point; negative: drawn from
  // `seed`, in the order they are used
  const bool given = xi.x >= 0.0F;
  directLight.direction       = float3(0.0f, 0.0f, 1.0f);
  directLight.pdf             = 0.0f;
  directLight.distance        = INFINITE;
  directLight.radianceOverPdf = float3(0.0f);

  const bool         bothSides = isVolumeSample || lightsFromBothSides;
  const LightPickSet set       = lightPickSet(pos, normal, /*cull=*/!bothSides);
  directLight.pick             = set.pick;
  if(set.pick.emissive <= 0.0f && set.candidates == 0)
    return;  // nothing can light this point

  // rand() can return exactly 1.0; the walk below wants [0, 1)
  float u = min(given ? xi.x : rand(seed), 0.99999994f);

  if(set.pick.emissive > 0.0f && (u < set.pick.emissive || set.candidates == 0))
  {
    // Emissive triangle (area light): power-weighted selection, then divide by the solid-angle pdf.
    int            tri    = selectEmissiveTriangle(given ? xi.w : rand(seed));
    float2         uTri   = given ? xi.yz : float2(0.0F);
    if(!given)
    {
      uTri.x = rand(seed);
      uTri.y = rand(seed);
    }
    EmissiveSample es     = sampleEmissiveTriangle(tri, pos, uTri, pushConst.gltfScene.emissiveTotalWeight);
    directLight.direction = es.direction;
    directLight.distance  = es.distance;
    if(es.pdf > 0.0)
    {
      // es.pdf already includes the power-weighted selection; fold in the emissive share.
      directLight.pdf             = set.pick.emissive * es.pdf;  // left as 0 otherwise -> caller skips NEE
      directLight.radianceOverPdf = es.radiance / directLight.pdf;
    }
    return;
  }
  u = min((u - set.pick.emissive) / set.restShare, 0.99999994f);  // the rest, stretched back over [0, 1)

  const float envShare = set.pick.env / set.restShare;
  if(u < envShare)
  {
    // One environment representation, whatever the sky is. Every EnvSystem that reaches this point
    // is a lat-long image by now: the HDR file is one, and the analytic skies are baked into one
    // (see GltfRenderer::isBakedEnvironment). eNone has no estimate and is never picked.
    float3 rand_val = given ? xi.yzw : float3(rand(seed), rand(seed), rand(seed));
    // The alias table may be coarser than the image (a baked sky), so sample on its own grid.
    const uint2 samplingGrid = uint2(pushConst.frameInfo->envSamplingGridWidth, pushConst.frameInfo->envSamplingGridHeight);
    float4 radiance_pdf;
    // A Sobol point inverts the table's CDF, which keeps the points stratified over the environment;
    // the alias method's O(1) pick scatters them, fine for the independent samples of DLSS frames
    if(compensatedEnv && given)
      radiance_pdf.xyz = environmentSampleCompensatedStratified(texturesHdr[HDR_IMAGE_INDEX], envSamplingCompensated, samplingGrid,
                                                                rand_val, directLight.direction, radiance_pdf.w);
    else if(compensatedEnv)
      radiance_pdf.xyz = environmentSampleCompensated(texturesHdr[HDR_IMAGE_INDEX], envSamplingCompensated, samplingGrid,
                                                      rand_val, directLight.direction, radiance_pdf.w);
    else
      radiance_pdf = environmentSample(texturesHdr[HDR_IMAGE_INDEX], envSamplingData, samplingGrid, rand_val, directLight.direction);
    const float envPdf = radiance_pdf.w;
    if(envPdf > 0.0)
    {
      // The pdf the caller weighs against the BSDF: computeEnvHitMisWeight's on the BSDF side
      directLight.pdf             = set.pick.env * envPdf;  // left as 0 otherwise -> caller skips NEE
      directLight.radianceOverPdf = radiance_pdf.xyz * pushConst.frameInfo->envIntensity / directLight.pdf;
      directLight.direction       = quatRotate(pushConst.frameInfo->envRotation, directLight.direction);
    }
    return;
  }

  // A punctual light: walk the lights' shares past the environment's to the one `u` falls in. The
  // last light that can light the point catches what rounding leaves over.
  const int numPunctual = punctualLightCount();
  float     cumulative  = envShare;
  int       lightIndex  = -1;
  float     share       = 0.0f;
  bool      found       = false;
  GltfLight light;
  for(int i = 0; i < numPunctual; i++)
  {
    const GltfLight candidate = getPunctualLight(i);
    const float     p = punctualPickShare(set, set.uniform ? 1.0f : punctualPickEstimate(candidate, pos, normal, !bothSides));
    if(p <= 0.0f)
      continue;
    lightIndex = i;
    share      = p;
    light      = candidate;
    cumulative += p;
    if(u < cumulative)
    {
      found = true;
      break;
    }
  }
  // A node of the light tree's cut past the lights (the last that can light the point catches what
  // rounding leaves over), then a leaf below it (what is left of `u` walks the tree), then a triangle
  // of it by its own weight
  if(!found && set.groups)
  {
    int   group      = -1;
    float groupShare = 0.0f;
    float groupStart = 0.0f;
    for(int g = 0; g < pushConst.gltfScene.numEmissiveGroups; g++)
    {
      const float p = lightPickProbability(emissiveGroupPickEstimate(g, pos, normal, !bothSides), set.total, set.candidates);
      if(p <= 0.0f)
        continue;
      group      = g;
      groupShare = p;
      groupStart = cumulative;
      cumulative += p;
      if(u < cumulative)
        break;
    }
    EmissiveTreeWalk walk;
    walk.leaf = -1;
    if(group >= 0)
      walk = emissiveTreeDescend(pushConst.gltfScene.emissiveGroups, group, pos, normal, !bothSides,
                                 saturate((u - groupStart) / groupShare) * 0.99999994F);
    if(walk.leaf >= 0)
    {
      const EmissiveGroup grp  = pushConst.gltfScene.emissiveGroups[walk.leaf];
      const int           tri  = selectEmissiveTriangleIn(int(grp.first), int(grp.count), given ? xi.w : rand(seed));
      float2              uTri = given ? xi.yz : float2(0.0F);
      if(!given)
      {
        uTri.x = rand(seed);
        uTri.y = rand(seed);
      }
      EmissiveSample es     = sampleEmissiveTriangle(tri, pos, uTri, grp.weight);
      directLight.direction = es.direction;
      directLight.distance  = es.distance;
      if(es.pdf > 0.0)
      {
        directLight.pdf             = set.restShare * groupShare * walk.prob * es.pdf;
        directLight.radianceOverPdf = es.radiance / directLight.pdf;
      }
      return;
    }
    if(group >= 0)
      return;  // no leaf below the cut node can light the point: this sample takes nothing
  }
  if(lightIndex < 0)
    return;
  const float selectionPdf = set.restShare * share;

  // singleLightContribution() drops any punctual light below the hemisphere of the normal it
  // is given. When the shading point is lit from both sides, hand it a normal pointing at the
  // light so nothing is rejected; the BSDF (or phase function) decides what the light does.
  float3 cullNormal = normal;
  if(bothSides)
  {
    cullNormal = (light.type == LightType::eLightTypeDirectional) ? -light.direction : normalize(light.position - pos);
  }
  LightContrib contrib = singleLightContribution(light, pos, cullNormal, given ? xi.yz : float2(rand(seed), rand(seed)));
  contrib.intensity *= evalIesFactor(pushConst.gltfScene, light, pos);  // EXT_lights_ies

  directLight.direction       = -contrib.incidentVector;
  directLight.distance        = contrib.distance;
  directLight.radianceOverPdf = contrib.intensity / selectionPdf;
  // A sphere's sample ends on the sphere: the cone reaches its silhouette, where the centre's
  // distance runs past the light into whatever is behind it
  if(light.type != LightType::eLightTypeDirectional && light.radius > 0.0F)
    directLight.distance = sphereLightSampleDistance(pos, directLight.direction, light.position, light.radius);

  // A sized light's cone sample carries its pdf: a BSDF ray can hit the light too (sizedLightHits),
  // weighed against this with sizedLightNeePdf. The sky's sun is the sky's disk, which a BSDF ray
  // sees on escape: skySunNeePdf() gives computeSkySunHitMisWeight() the same pdf from `pick.sun`.
  // A point is DIRAC: no BSDF ray reaches it, so this is its only technique, of weight 1. So is
  // every light at a volume scatter, whose phase-sampled ray gathers no light (sizedLightHits).
  const float sunSolidAngle = (lightIndex == skySunPunctualIndex()) ? skySunSolidAngle(pushConst.frameInfo) : 0.0F;
  if(sunSolidAngle > 0.0F)
    directLight.pdf = selectionPdf / sunSolidAngle;
  else if(contrib.pdf != DIRAC && !isVolumeSample)
    directLight.pdf = selectionPdf * contrib.pdf;
  else
    directLight.pdf = DIRAC;
}

// The pdf with which sampleLights(), at the vertex `pick` was made at, draws a direction in the
// cone of sized light `light`, of `oneMinusCos` from there: its share of the pick over the cone's
// solid angle. 0 where that vertex's pick gave it nothing.
float sizedLightNeePdf(LightPick pick, GltfLight light, float oneMinusCos)
{
  const bool         cull  = any(pick.normal != float3(0.0F));
  const LightPickSet set   = lightPickSet(pick.pos, pick.normal, cull);
  const float        share = punctualPickShare(set, set.uniform ? 1.0F : punctualPickEstimate(light, pick.pos, pick.normal, cull));
  return set.restShare * share / (M_TWO_PI * max(oneMinusCos, 1e-10F));
}

void sampleEnvironment(float3 direction, SceneFrameInfo* frameInfo, out float3 envColor, out float envPdf)
{
  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseNoEnvironment))
  {
    // No environment: contributes no radiance and cannot be importance-sampled.
    envColor = float3(0.0f);
    envPdf   = 0.0f;
    return;
  }

  // Same as the sampling branch above: by the time a ray escapes, every sky is a lat-long image.
  float3 dir = quatRotateInverse(frameInfo->envRotation, direction);
  float2 uv  = getSphericalUv(dir);
  float4 env = texturesHdr[HDR_IMAGE_INDEX].SampleLevel(uv, 0);
  envColor   = env.rgb * frameInfo->envIntensity;
  envPdf     = env.w;
}

// MIS weight for a BSDF ray that escaped to the environment, against the environment sample the
// vertex it left drew with probability `pick.env`.
// The compensated table's pdf (sampleLights' `compensatedEnv`) of world direction `direction`.
float environmentCompensatedPdfAt(float3 direction, SceneFrameInfo* frameInfo)
{
  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseNoEnvironment))
    return 0.0F;
  const uint2 samplingGrid = uint2(frameInfo->envSamplingGridWidth, frameInfo->envSamplingGridHeight);
  return environmentCompensatedPdf(envSamplingCompensated, samplingGrid, quatRotateInverse(frameInfo->envRotation, direction));
}

float computeEnvHitMisWeight(float lastSamplePdf, float envPdf, LightPick pick)
{
  if(lastSamplePdf == DIRAC)
    return 1.0f;
  return misPowerWeight(lastSamplePdf, pick.env * envPdf);
}

// Solid-angle pdf with which sampleLights() lands on the sky's sun at a vertex that picked it with
// probability `pick.sun`: that over the disk's solid angle, exactly as it computes it. Only called
// where the disk exists, so the solid angle is not zero.
float skySunNeePdf(LightPick pick)
{
  return pick.sun / skySunSolidAngle(pushConst.frameInfo);
}

// MIS weight for a BSDF ray that escaped onto the sky's sun disk, against the sun's NEE sample.
// Never called after a delta bounce: the caller adds the disk unweighted there.
float computeSkySunHitMisWeight(float lastSamplePdf, LightPick pick)
{
  return misPowerWeight(lastSamplePdf, skySunNeePdf(pick));
}

//-----------------------------------------------------------------------
// Infinite plane and shadow catcher
//-----------------------------------------------------------------------

// A shadow catcher composites, as the rasterizer's does: the ray passes through the plane and sees
// exactly what it would without it -- the same background, the same camera-ray status -- darkened by
// how much of the light the plane would receive is blocked there. One next-event sample's
// visibility per pass, so the darkening converges to the occluded share of the light, sun and
// environment alike; its strength is the rasterizer's (groundCatcherStrength).
//
// Returns true: the path always continues, below the plane.
bool handleShadowCatcher(IRaytracer      raytracer,
                         HitState        hit,
                         PbrMaterial     pbrMat,
                         inout RayDesc   ray,
                         inout float3    radiance,
                         inout float3    throughput,
                         inout float     lastSamplePdf,
                         inout uint      seed,
                         bool            shadowInside,      // the catcher lies inside a medium...
                         float3          shadowExtinction,  // ...of this extinction
                         SceneFrameInfo* frameInfo)
{
  DirectLight directLight;
  sampleLights(hit.pos, pbrMat.N, ray.Direction, seed, directLight);

  float visibility = 1.0F;
  if(dot(directLight.direction, hit.nrm) > 0.0F && directLight.pdf != 0.0F && any(directLight.radianceOverPdf > 0.0F))
  {
    RayDesc shadowRay;
    shadowRay.Origin    = safeOffsetRay(hit.pos, hit.geonrm, hit.offsetDistance);
    shadowRay.Direction = directLight.direction;
    shadowRay.TMin      = 0.0;
    shadowRay.TMax      = directLight.distance;  // short of the light: an emitter is no shadow on what it lights
    const float3 shadow = raytracer.TraceShadow(shadowRay, seed);
    visibility          = (shadow.x + shadow.y + shadow.z) / 3.0F;
  }

  throughput *= 1.0F - groundCatcherStrength(frameInfo) * (1.0F - visibility);
  ray.Origin = safeOffsetRay(hit.pos, -hit.geonrm, hit.offsetDistance);
  return true;
}

// The plane fades out at the edge of its disk as the rasterizer's floor does (groundFadeWeight): a
// ray hits it with probability equal to the fade there, and otherwise passes on to whatever is behind.
bool checkInfinitePlaneIntersection(RayDesc ray, inout HitPayload payload, inout HitState hit, SceneFrameInfo* frameInfo, inout uint seed)
{
  const float intersectionDist = infinitePlaneCrossing(ray.Origin, ray.Direction, payload.hitT, frameInfo);
  if(intersectionDist < 0.0F)
    return false;
  const float3 normal = float3(0, 1, 0);

  const float3 planePos = ray.Origin + ray.Direction * intersectionDist;
  const float  fade     = groundFadeWeight(planePos, frameInfo);
  if(fade < 1.0F && rand(seed) >= fade)
    return false;

  payload.hitT  = intersectionDist;
  hit.pos       = planePos;
  hit.shadowPos = hit.pos;
  hit.nrm       = normal;
  hit.geonrm    = normal;
  hit.tangent   = float3(1, 0, 0);
  hit.bitangent = float3(0, 0, 1);
  // Intersected here, analytically, never by a traced ray: the only error is the rounding of
  // origin + direction * t
  hit.offsetDistance = roundingGamma(3) * dot(abs(normal), abs(ray.Origin) + abs(ray.Direction * payload.hitT));

  return true;
}


//-----------------------------------------------------------------------
// In-volume step: Beer's-law absorption (always when GLTF_USE_VOLUME) +
// Henyey-Greenstein scatter sampling (only when GLTF_USE_SCATTER).
//-----------------------------------------------------------------------
// IMPORTANT: this function exists whenever GLTF_USE_VOLUME is on, NOT only when
// GLTF_USE_SCATTER is on. KHR_materials_volume by itself (no scatter
// extension) still needs per-segment Beer's-law absorption applied along the
// camera ray, which is what makes a glass with attenuationColor="green" look
// green. Earlier the entire function was gated on GLTF_USE_SCATTER which
// silently dropped that absorption pass in optimal mode for any scene that
// uses volume but not scatter (e.g. CompareVolume.gltf).
//
// Returns true only when a real scatter event happened (GLTF_USE_SCATTER
// must be on for that to be possible). Returns false when only absorption was
// applied. The bounce-loop caller uses the return value to decide whether to
// run scatter-only follow-up work (NEE at the scatter point, scatterBounces++).
// The free-flight distance of a step through a scattering medium, and the channel weights it was
// drawn with. Drawn before the step's ray is traced (pathTraceOneBounce), so the ray is traced only
// that far: with a free path well under a millimetre almost every step scatters, and a traversal to
// the surface, the any-hit on the way and a hit state for it would all be thrown away. A miss of
// such a ray is the scatter. `drawn` is false for a medium that only absorbs: no distance to draw.
// Declared without GLTF_USE_VOLUME too: the bounce passes one (never drawn) whatever the scene has.
struct FreeFlight
{
  float  dist;         // INFINITE when the channel drawn has no extinction
  float3 channelProb;  // the balance heuristic's weights, from the throughput
  bool   drawn;
  // Walk guiding (walk_guide.h.slang) for a step after a guided walk's scatter: the share of guided
  // steps (0: this step was classic only), the guided extinction's stretch along this step, and the
  // guided over the phase function's direction pdf
  float guideFraction;
  float guideStretch;
  float guideFactor;
};

#if GLTF_USE_VOLUME

#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
// The walk guide of a medium (walk_guide.h.slang), from its largest single-scattering albedo
WalkGuide mediumWalkGuide(VolumeMedium medium)
{
  const float3 sigmaS = float3(medium.scatterCoefficient);
  const float3 albedo = sigmaS / max(volumeExtinction(medium), float3(1e-30F));
  return makeWalkGuide(max(albedo.x, max(albedo.y, albedo.z)), float(medium.scatterAnisotropy));
}

// `walkGuide`: what the walk's last scatter chose (PathTracerState::walkGuide), `walkNormal` its entry
// normal, `direction` this step's.
FreeFlight sampleFreeFlight(VolumeMedium medium, float3 throughput, inout uint seed, float walkGuide, float3 walkNormal, float3 direction)
{
  FreeFlight ff = {};
  float3 scatterCoeff = float3(medium.scatterCoefficient);
  float  maxScatter   = max(scatterCoeff.x, max(scatterCoeff.y, scatterCoeff.z));
  if(!(maxScatter > VOLUME_MIN_SCATTER))
    return ff;
  float3 extinction = volumeExtinction(medium);

  // KHR_materials_volume media are homogeneous, so the free-flight distance can be sampled
  // analytically -- no delta tracking and no majorant. The medium is chromatic (sigma_t and
  // sigma_s are per-channel), so one channel is picked to sample the distance and the three
  // per-channel distance pdfs are combined with the balance heuristic (spectral MIS). Every
  // channel then keeps exactly its own transmittance and its own scattering rate.
  //
  // A single-majorant tracker would instead scatter every channel at the majorant's rate:
  // the low-extinction channels would random-walk far too often, and since their extra
  // "collisions" are lossless null collisions, their effective per-collision albedo -- which
  // is what sets the perceived color of a dense medium -- would drift towards 1. That breaks
  // the multi-scatter albedo mapping the extension is built on (a yellow multiscatterColor
  // over a slightly red attenuationColor came out pink).
  float3 channelProb = max(throughput, float3(0.0));
  float  probSum     = channelProb.x + channelProb.y + channelProb.z;
  channelProb        = (probSum > 0.0) ? channelProb / probSum : float3(1.0 / 3.0);

  // Pick the channel whose free-flight distribution is sampled, then draw the distance. The
  // selection consumes only the bucket it lands in, so rescaling within that bucket recovers a
  // fresh uniform for the distance -- one rand() per scatter event instead of two, and in a
  // dense medium there is one of these per event.
  float u             = min(rand(seed), VOLUME_RAND_CEIL);
  float sigmaTChannel = extinction.z;
  if(u < channelProb.x)
  {
    sigmaTChannel = extinction.x;
    u /= channelProb.x;
  }
  else if(u < channelProb.x + channelProb.y)
  {
    sigmaTChannel = extinction.y;
    u             = (u - channelProb.x) / channelProb.y;
  }
  else
  {
    u = (u - channelProb.x - channelProb.y) / channelProb.z;
  }
  if(walkGuide != 0.0F)
  {
    const WalkGuide guide = mediumWalkGuide(medium);
    ff.guideFraction      = guide.fraction;
    ff.guideStretch       = walkGuideStretch(guide, dot(direction, walkNormal));
    ff.guideFactor        = abs(walkGuide);
    if(walkGuide < 0.0F)
      sigmaTChannel *= ff.guideStretch;  // drawn guided
  }
  ff.dist        = (sigmaTChannel > 0.0) ? -log(max(u, VOLUME_RAND_FLOOR)) / sigmaTChannel : INFINITE;
  ff.channelProb = channelProb;
  ff.drawn       = true;
  return ff;
}
#endif

// `freeFlight`: the step's distance, drawn before the trace; not drawn, it is drawn here.
// `walkGuide` / `walkNormal`: PathTracerState's. This step uses the guide its last scatter left, and a
// scatter leaves the next step's; reaching the surface leaves none.
bool handleVolumeScatter(VolumeMedium medium, float hitDistance, inout RayDesc ray, inout float3 throughput, inout float lastSamplePdf,
                         inout uint seed, FreeFlight freeFlight, inout float walkGuide, float3 walkNormal)
{
  float3 extinction = volumeExtinction(medium);

#if GLTF_USE_SCATTER
  float3 scatterCoeff = float3(medium.scatterCoefficient);
  float  maxScatter   = max(scatterCoeff.x, max(scatterCoeff.y, scatterCoeff.z));
  if(maxScatter > VOLUME_MIN_SCATTER)
  {
    const FreeFlight ff = freeFlight.drawn ? freeFlight : sampleFreeFlight(medium, throughput, seed, walkGuide, walkNormal, ray.Direction);
    const float3 channelProb = ff.channelProb;
    const float  scatterDist = ff.dist;
    walkGuide                = 0.0F;

    // A finite free path is a scatter, including when the ray left the mesh (hitDistance is
    // INFINITE). Both unbounded means this channel has no extinction: fall through and let the
    // miss path below decide, which never multiplies 0 by INFINITE.
    if(isfinite(scatterDist) && scatterDist < hitDistance)
    {
      // Scattering event: pdf_i(t) = sigma_t_i * exp(-sigma_t_i * t), combined over the channels
      // (and with the guided step's, walkStepPdf).
      float3 transmittance = exp(-extinction * scatterDist);
      float  pdf = dot(channelProb, walkStepPdf(extinction, scatterDist, false, ff.guideFraction, ff.guideStretch, ff.guideFactor));
      if(pdf <= 0.0)
      {
        // Every channel's transmittance underflowed: the medium is opaque along this segment.
        throughput = float3(0.0);
        return false;
      }
      throughput *= scatterCoeff * transmittance / pdf;

      ray.Origin        = ray.Origin + ray.Direction * scatterDist;
      float3 wi         = ray.Direction;
      float  anisotropy = float(medium.scatterAnisotropy);
      // A walk entered from a surface draws the next step guided or classic (walk_guide.h.slang);
      // the next step weighs its outcome against both.
      WalkGuide guide = {};
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtWalkGuiding) && any(walkNormal != float3(0.0F)))
        guide = mediumWalkGuide(medium);
      float u = rand(seed);
      if(u < guide.fraction)
      {
        const float cosTheta = walkGuideSampleCos(guide, u / guide.fraction);
        const float phi      = M_TWO_PI * rand(seed);
        const float sinTheta = sqrt(max(1.0F - cosTheta * cosTheta, 0.0F));
        float3      tangent, bitangent;
        orthonormalBasis(walkNormal, tangent, bitangent);
        ray.Direction = normalize(tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + walkNormal * cosTheta);
      }
      else
        ray.Direction = sampleHenyeyGreenstein(float2(guide.fraction > 0.0F ? rand(seed) : u, rand(seed)), anisotropy, wi);

      lastSamplePdf = henyeyGreensteinPdf(dot(wi, ray.Direction), anisotropy);
      if(guide.fraction > 0.0F)
      {
        const float factor = walkGuidePdf(guide, dot(ray.Direction, walkNormal)) / lastSamplePdf;
        walkGuide          = u < guide.fraction ? -factor : factor;
      }

      return true;  // scatter event
    }

    // Reached the surface, or left the mesh, without scattering.
    if(!(hitDistance < INFINITE))
    {
      // Any extinction over an unbounded distance absorbs the path. A zero extinction stays put:
      // exp(0 * INFINITE) is NaN, and this medium had nothing to apply.
      if(max(extinction.x, max(extinction.y, extinction.z)) > 0.0)
        throughput = float3(0.0);
      return false;
    }
    // The probability of reaching the surface is exp(-sigma_t_i * d) per channel, combined with
    // the balance heuristic (and with the guided step's).
    float3 transmittance = exp(-extinction * hitDistance);
    float  pdf = dot(channelProb, walkStepPdf(extinction, hitDistance, true, ff.guideFraction, ff.guideStretch, ff.guideFactor));
    throughput *= (pdf > 0.0) ? transmittance / pdf : float3(0.0);
    return false;
  }
#endif  // GLTF_USE_SCATTER

  // Pure-absorption path (no scatter coefficient, or GLTF_USE_SCATTER disabled):
  // straight Beer's law applied over the segment length. This is the case for plain
  // KHR_materials_volume materials and is what produces the colored tint of a glass
  // filled with an absorbing medium. An unbounded segment with any extinction absorbs fully.
  if(!(hitDistance < INFINITE))
  {
    if(max(extinction.x, max(extinction.y, extinction.z)) > 0.0)
      throughput = float3(0.0);
    return false;
  }
  throughput *= exp(-hitDistance * extinction);
  return false;
}
#endif  // GLTF_USE_VOLUME

// Scatter-only: next-event estimation at the scatter point. Only meaningful when a
// scatter event can actually happen, so gated on GLTF_USE_SCATTER.
#if GLTF_USE_SCATTER
float3 volumeScatterNEE(IRaytracer raytracer, VolumeMedium medium, float3 scatterPos, float3 wiBeforeScatter, float3 throughput, inout uint seed)
{
  DirectLight directLight;
  sampleLights(scatterPos, /*normal=*/wiBeforeScatter, wiBeforeScatter, seed, directLight, /*isVolumeSample=*/true);

  // pdf 0 is a failed sample; a punctual light is DIRAC (negative) and must go through, since no
  // phase-sampled ray can reach it. Skip lights that give nothing here (a spot whose cone misses
  // this point) before paying for the shadow ray.
  if(directLight.pdf == 0.0F || all(directLight.radianceOverPdf == 0.0F))
    return float3(0.0F);

  float cosTheta = dot(wiBeforeScatter, directLight.direction);
  // The phase function value. It is also the pdf of sampling this direction, but volume NEE is the
  // only direct-light estimator at a scatter vertex (the exit surface's NEE faces the interior and
  // does not run), so the MIS weight against the phase-sampled path is 1.
  float phase = henyeyGreensteinPdf(cosTheta, float(medium.scatterAnisotropy));
  float misWeight = 1.0F;

  RayDesc shadowRay;
  shadowRay.Origin    = scatterPos;
  shadowRay.Direction = directLight.direction;
  shadowRay.TMin      = 0.0F;
  shadowRay.TMax      = directLight.distance;
  float3 shadowFactor = raytracer.TraceShadow(shadowRay, seed, /*initialInside=*/true, volumeExtinction(medium));

  return throughput * directLight.radianceOverPdf * misWeight * phase * shadowFactor;
}
#endif  // GLTF_USE_SCATTER


//-----------------------------------------------------------------------
// DLSS guide buffers
//-----------------------------------------------------------------------

#if USE_GUIDE_SHADER
// Previous-frame world position of the surface point at `hitPos`, on triangle `triangleID` of
// render primitive `rprimID` in render node `rnodeID`, with the barycentrics getHitState used.
// A primitive skinned, morphed or re-uploaded this frame takes its triangle's previous corners
// (prevVertexPositions; the shape moved under the point), a node that moved takes its previous
// transform (prevRenderNodeObjectToWorld), and a point with neither is where it was.
float3 dlssPrevHitPos(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int rnodeID, int rprimID, int triangleID,
                      float3 barys, float3 hitPos)
{
  float3* prevPositions = nullptr;
  if(pushConst.prevVertexPositions != nullptr)
    prevPositions = pushConst.prevVertexPositions[rprimID];
  if(prevPositions == nullptr && pushConst.prevRenderNodeObjectToWorld == nullptr)
    return hitPos;

  float3 objPos;
  if(prevPositions != nullptr)
  {
    const uint3 idx = getTriangleIndices(renderPrim, triangleID);
    objPos          = prevPositions[idx.x] * barys.x + prevPositions[idx.y] * barys.y + prevPositions[idx.z] * barys.z;
  }
  else
    objPos = mul(float4(hitPos, 1.0), renderNode.worldToObject).xyz;

  const float4x4 objectToWorld =
      pushConst.prevRenderNodeObjectToWorld != nullptr ? pushConst.prevRenderNodeObjectToWorld[rnodeID] : renderNode.objectToWorld;
  return mul(float4(objPos, 1.0), objectToWorld).xyz;
}
#endif  // USE_GUIDE_SHADER

#if USE_DLSS_SHADER
float3 dlssEnforceMinAlbedo(float3 albedo, float3 specularAlbedo)
{
  float sumAlbedo = albedo.x + albedo.y + albedo.z;
  float sumSpec   = specularAlbedo.x + specularAlbedo.y + specularAlbedo.z;
  if(sumAlbedo + sumSpec < 0.05f)
  {
    return max(albedo, float3(0.2f));
  }
  return albedo;
}

bool dlssIsClearGlass(PbrMaterial pbrMat)
{
  return (pbrMat.transmission > 0.9f) && (pbrMat.roughness.x < 0.05f) && (pbrMat.opacity >= 0.99f);
}

// Clear-glass test from the material factors alone, without sampling a texture: 1 = clear glass,
// 0 = not, -1 = a texture decides (evaluate the material). Textures only scale the factors down,
// so a low transmission factor or an already-smooth roughness factor is final. Mirrors
// dlssIsClearGlass; opacity is checked separately (getOpacity). Scatter and anisotropy change
// the evaluated transmission / roughness, so they defer to the evaluation too.
int dlssClearGlassFromFactors(GltfShadeMaterial* mat)
{
  if(mat.pbrModel != PbrModel::ePbrModelMetallicRoughness || mat.transmissionFactor <= 0.9f)
    return 0;
  if(isTexturePresent(mat.transmissionTexture) || mat.scatterStrengthFactor > 0.0f || mat.anisotropyStrength > 0.0f)
    return -1;
  if(mat.pbrRoughnessFactor * mat.pbrRoughnessFactor < 0.05f)
    return 1;
  return isTexturePresent(mat.pbrMetallicRoughnessTexture) ? -1 : 0;
}

bool dlssIsThinWalled(GltfShadeMaterial* mat)
{
#if GLTF_USE_VOLUME
  return mat.thicknessFactor <= 0.0f;
#else
  return true;
#endif
}

// One traversal to the next surface the clear-glass chain has to look at. Thin clear glass and
// cut-out texels are rejected inside the traversal itself, so a stack of glass panes costs a
// single ray; alpha-blend texels go through alphaHashU, the same deterministic cells the color
// path keeps while DLSS denoises, so guides and color agree on which cells are solid.
// FORCE_NON_OPAQUE makes the (opaque-flagged) glass report candidates.
//
// `glassCount` grows by the panes skipped in front of the hit. Candidates arrive in no set order,
// so a pane is counted when it is nearer than the hit committed so far; one found before a nearer
// commit is still counted, which only errs toward too many (see dlssTraceClearGlass).
HitPayload dlssTraceSkipThinGlass(RayDesc ray, inout int glassCount)
{
  GltfShadeMaterial*   materials        = pushConst.gltfScene->materials;
  GltfRenderNode*      renderNodes      = pushConst.gltfScene->renderNodes;
  GltfRenderPrimitive* renderPrimitives = pushConst.gltfScene->renderPrimitives;

  HitPayload payload = {};
  payload.hitT       = INFINITE;

  float    committedT = INFINITE;
  RayQuery rayQuery;
  rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, ray);
  while(rayQuery.Proceed())
  {
    GltfRenderNode     node       = renderNodes[rayQuery.CandidateInstanceIndex()];
    GltfShadeMaterial* mat        = materials + max(0, node.materialID);
    const float        candidateT = rayQuery.CandidateTriangleRayT();
    const bool         clearGlass = dlssClearGlassFromFactors(mat) == 1 && dlssIsThinWalled(mat);
    float              opacity    = 1.0f;
    float              u          = 0.0f;
    if(!clearGlass || mat.alphaMode != AlphaMode::eAlphaModeOpaque)
    {
      const float2 bary         = rayQuery.CandidateTriangleBarycentrics();
      const float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
      const uint   triangleID   = rayQuery.CandidatePrimitiveIndex();
      opacity = getOpacity(node, renderPrimitives[rayQuery.CandidateInstanceID()], triangleID, barycentrics);
      u       = alphaHashU(rayQuery.CandidateInstanceIndex(), triangleID, barycentrics);
    }

    if(clearGlass && opacity >= 0.99f)
    {
      if(candidateT < committedT)
        glassCount++;  // Skipped pane in front of the hit so far
    }
    else if(u < opacity)
    {
      rayQuery.CommitNonOpaqueTriangleHit();
      committedT = candidateT;
    }
  }

  if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    payload.hitT        = rayQuery.CommittedRayT();
    payload.rprimID     = rayQuery.CommittedInstanceID();
    payload.rnodeID     = rayQuery.CommittedInstanceIndex();
    payload.primitiveID = rayQuery.CommittedPrimitiveIndex();
    payload.bary        = rayQuery.CommittedTriangleBarycentrics();
  }
  return payload;
}

// Primary surface replacement (PSR) through clear glass. Follows the specular transmission
// direction -- straight through a thin-walled surface, refracted through a volume -- across every
// clear-glass layer, and returns the first surface that is not clear glass. DLSS-RR then gets the
// guides of what the camera actually sees through the glass: its albedo, normal, roughness, and a
// virtual position along the camera ray at the full path length, whose depth and motion move with
// the content behind the glass rather than with the glass itself.
//
// The chain is deterministic, so the guides do not flicker from frame to frame: alpha is resolved
// by alphaHashU, the per-cell decision the path makes while DLSS denoises. Thin clear glass costs nothing
// beyond the one traversal (dlssTraceSkipThinGlass); only refracting glass, or glass whose
// clarity depends on a texture, starts another one.
//
// The guides must agree with the noisy color. The path spends one bounce per glass surface, so
// when the chain crosses as many surfaces as the path has bounces (maxDepth), the path never
// reached the surface behind and the pixel is dark there; the glass keeps its own guides.
static const int DLSS_MAX_GLASS_LAYERS = 8;

struct DlssGlassPsr
{
  bool   valid      = false;         // A non-glass surface was found behind the glass
  float3 albedo     = float3(0.5f);  // Its base color (0.5 grey when the chain left the scene)
  float3 normal     = float3(0.0f);  // Its shading normal
  float  roughness  = 0.0f;          // Its roughness
  float  pathLength = 0.0f;          // Distance from the first glass hit to it, along the chain
  float3 posDelta   = float3(0.0f);  // Its instance motion: previous-frame position minus current position
};

DlssGlassPsr dlssTraceClearGlass(PbrMaterial glassMat, HitState glassHit, float3 dir, bool isInside)
{
  GltfShadeMaterial*   materials       = pushConst.gltfScene->materials;
  GltfRenderNode*      renderNodes     = pushConst.gltfScene->renderNodes;
  GltfRenderPrimitive* renderPrimitive = pushConst.gltfScene->renderPrimitives;
  GltfTextureInfo*     texInfos        = pushConst.gltfScene->textureInfos;

  DlssGlassPsr result = {false, float3(0.5f), float3(0.0f), 0.0f, 0.0f, float3(0.0f)};

  PbrMaterial mat          = glassMat;
  float3      origin       = glassHit.pos;
  float3      originNormal = glassHit.geonrm;          // The geometric normal at `origin`...
  float       originOffset = glassHit.offsetDistance;  // ...and the step off it (safeOffsetRay)
  int         glassCount   = 1;                        // Glass surfaces crossed, the primary hit included
  for(int layer = 0; layer < DLSS_MAX_GLASS_LAYERS; layer++)
  {
    // Smooth thin-walled transmission does not bend the ray (see bsdfSample); a volume refracts,
    // or reflects on total internal reflection.
    if(mat.thickness > 0.0f)
    {
      const float3 N       = dot(mat.N, dir) < 0.0f ? mat.N : -mat.N;
      const float3 refrDir = refract(dir, N, mat.ior1 / mat.ior2);
      if(dot(refrDir, refrDir) > 0.0f)
      {
        dir      = refrDir;
        isInside = !isInside;
      }
      else
        dir = reflect(dir, N);
    }

    RayDesc guideRay;
    guideRay.Origin    = safeOffsetRay(origin, dot(dir, originNormal) > 0.0f ? originNormal : -originNormal, originOffset);
    guideRay.Direction = dir;
    guideRay.TMin      = 0.0f;
    guideRay.TMax      = INFINITE;
    HitPayload payload = dlssTraceSkipThinGlass(guideRay, glassCount);
    if(payload.hitT == INFINITE || glassCount >= pushConst.maxDepth)
      return result;
    result.pathLength += payload.hitT;

    GltfRenderNode      node  = renderNodes[payload.rnodeID];
    GltfRenderPrimitive prim  = renderPrimitive[payload.rprimID];
    float3              barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
    HitState  hit  = getHitState(prim, barys, node.worldToObject, node.objectToWorld, payload.primitiveID, dir);
    MeshState mesh = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, isInside, 0.0f);
    mesh.baseColorVertexMul = hit.color;
    mat    = evaluateMaterial(max(0, node.materialID), mesh, allImages, allSamplers, texInfos, materials,
                              hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin));
    origin       = hit.pos;
    originNormal = hit.geonrm;
    originOffset = hit.offsetDistance;

    if(dlssIsClearGlass(mat))
    {
      glassCount++;
      continue;  // Refracting glass, or glass only a texture showed to be clear
    }

    result.valid     = true;
    result.albedo    = mat.baseColor.xyz;
    result.normal    = mat.N;
    result.roughness = mat.roughness.x;
    result.posDelta =
        dlssPrevHitPos(node, prim, payload.rnodeID, payload.rprimID, payload.primitiveID, barys, hit.pos) - hit.pos;
    return result;
  }
  return result;  // Too many layers: keep the glass's own guides
}
#endif  // USE_DLSS_SHADER


//-----------------------------------------------------------------------
// Wireframe overlay
//-----------------------------------------------------------------------

#if WIREFRAME

float computeMeshWireframe(float4x4 objectToWorld, float3 v0, float3 v1, float3 v2, float2 bary, float hitT, float pixelAngle, float3 N, float3 V)
{
  float3 wp0 = mul(float4(v0, 1.0), objectToWorld).xyz;
  float3 wp1 = mul(float4(v1, 1.0), objectToWorld).xyz;
  float3 wp2 = mul(float4(v2, 1.0), objectToWorld).xyz;

  float3 barys = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

  float3 e1 = wp1 - wp0;
  float3 e2 = wp2 - wp0;

  float worldFoot = hitT * pixelAngle / max(abs(dot(N, V)), 1e-3);

  float  twoArea  = length(cross(e1, e2));
  float3 edgesOpp = float3(length(wp2 - wp1), length(wp0 - wp2), length(wp1 - wp0));
  float3 deltas   = worldFoot * edgesOpp / max(twoArea, 1e-20);

  return processWireframe(WIREFRAME_PRESETS[WIREFRAME_THIN], barys, deltas);
}

#endif  // WIREFRAME


//-----------------------------------------------------------------------
// Camera and selection ray
//-----------------------------------------------------------------------

// Box-Muller transform: two uniform numbers in [0, 1] to two independent standard normal values.
// rand() returns exactly 0 once in 2^32 draws, and log(0) would make the sample infinite and the
// camera ray NaN. The floor is the smallest non-zero value rand() returns, which keeps r finite
// (about 6.7). It must be a normal float: a denormal such as 1e-38 is flushed to zero on the GPU.
float2 sampleGaussian(float2 u)
{
  const float r     = sqrt(-2.0f * log(max(u.x, 2.3283064e-10f)));
  const float theta = 2.0f * M_PI * u.y;
  return r * float2(cos(theta), sin(theta));
}


uint traceSelectionRay(IRaytracer raytracer, float2 samplePos, float2 imageSize)
{
  const bool isOrthographic = hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneIsOrthographic);
  RayDesc ray = getRay(samplePos, float2(0.5f), imageSize, pushConst.frameInfo->projInv, pushConst.frameInfo->viewInv, isOrthographic);
  HitPayload payload = {};
  raytracer.TraceLow(ray, payload);
  return (payload.hitT != INFINITE) ? uint(payload.rnodeID + 1) : 0;
}


//-----------------------------------------------------------------------
// Path trace iteration
//-----------------------------------------------------------------------

// Step results for the path tracing
enum PathStepResult
{
  eOutOfVolume,     // Continue tracing after volume processing
  eVolumeContinue,  // Continue tracing the ray through the volume
  eEarlyContinue,   // Early continue tracing the ray (infinite plane)
  eBreak,           // Break the path tracing
}


// Path tracer state
//
// isInside / volumeMedium / scatterBounces / guide are only carried in builds that
// actually use them. Each field dropped removes that many live registers from the
// per-thread state that survives across every Trace() call in the bounce loop,
// which is what determines warps-per-SM (occupancy) on the megakernel.
//
// isInside is conditional on either GLTF_USE_TRANSMISSION (toggled at refraction)
// or GLTF_USE_VOLUME (used to gate the in-volume scatter check).
//
// firstHitPos is always present (drives the selection/picking NDC depth) even
// when DLSS/OptiX is disabled; it is written on the first hit (or by the
// primary-miss backplate path).
struct PathTracerState
{
  float3  radiance      = float3(0.0F, 0.0F, 0.0F);
  float3  throughput    = float3(1.0F, 1.0F, 1.0F);
  float3  firstHitPos   = float3(1e34f);  // matches SampleResult::hitPosition sentinel
  float   lastSamplePdf = DIRAC;
  // What the last vertex's light pick gave the lights its BSDF ray can also reach, for that ray's MIS
  // weight. Read only when lastSamplePdf is not DIRAC, so only after a vertex that sampled the lights.
  LightPick lightPick    = {0.0F, 0.0F, 0.0F, float3(0.0F), float3(0.0F)};
  float2  maxRoughness  = float2(0.0);
  // How rough the path's scattering has been (perceptual roughness, 0..1): the largest of the lobes
  // it sampled, a diffuse one and a scattering walk counting as 1 (Unreal's path roughness). It decides
  // which light the firefly clamp may cut (ePtClampIndirect) and, with ePtApproximateCaustics, how
  // rough later lobes are. `prefixIndirect`: the path was already indirect before its last event, which
  // is the class of the light that event's own ray reaches.
  float pathRoughness  = 0.0F;
  bool  prefixIndirect = false;
  // The length of the clear crossings (CLEAR_CROSSING_ALPHA) since the vertex whose light sample the
  // next hit is weighed against: an emitter's MIS pdf converts by the whole distance from there.
  float nullDistance = 0.0F;
  bool    solid         = true;
  bool    firstHitHazed = true;  // false when the camera ray landed on the shadow catcher
  RayCone cone          = {0.0F, pushConst.pixelAngle};
  int     surfaceDepth  = 0;
#if GLTF_USE_HAIR
  int hairBounces = 0;  // fibre events, on the hair's own budget (PathtracePushConstant::hairBudget)
#endif
#if GLTF_USE_TRANSMISSION || GLTF_USE_VOLUME
  bool isInside = false;
#endif
#if GLTF_USE_VOLUME
  VolumeMedium volumeMedium = {};
  // Set on a scatter. Volume NEE is the only direct-light estimator for the walk, so an
  // environment or emitter hit while this is set contributes nothing. Cleared at the first
  // surface hit outside the medium, which is shaded on its own.
  bool exitedVolumeWalk = false;
#endif
#if GLTF_USE_SCATTER
  int scatterBounces = 0;  // Scatter events of the current walk
#endif
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
  // The material whose medium the path entered (-1: none, or the camera's medium). The diffuse exit
  // divides out this medium's albedo, whatever surface the walk leaves through.
  int volumeMaterial = -1;
  // Walk guiding (walk_guide.h.slang): the normal where the walk entered, facing out of the medium
  // (zero: a walk in the camera's medium, never guided), and what its last scatter drew the next step
  // with: 0 for no guiding (a walk's first step, or after the surface), else the guided over the phase
  // function's direction pdf, negative when the guided technique drew it
  float3 walkNormal = float3(0.0F);
  float  walkGuide  = 0.0F;
#endif
#if USE_GUIDE_SHADER
  GuideScratch guide           = {};
  float3       firstHitPosPrev = float3(1e34f);  // Previous-frame world position of the first hit (instance motion).
#endif
#if WIREFRAME
  float firstHitWireframeAlpha = 0.0;
#endif
};

struct BounceScratch
{
  bool   nextEventValid = false;
  bool   indirect       = false;  // The contribution reached the path after a rough bounce: the clamp may cut it
  float3 contribution   = float3(0.0F);
  // Keep shadow rays full precision: half packing erases the self-intersection offset.
  float3 shadowRayPos        = float3(0.0F);
  float3 shadowRayDir        = float3(0.0F);
  float  shadowRayDist       = 0.0F;
  bool   shadowRayInside     = false;         // The shadow ray starts inside a medium...
  float3 shadowRayExtinction = float3(0.0F);  // ...of this extinction.
  bool   hairEvent           = false;         // This vertex was a fibre's and spent the hair budget, not a surface bounce
  bool   fibre               = false;         // This vertex was a fibre's, within the budget or past it
};

// The hair's own budget (PathtracePushConstant::hairBudget, the host's ptHairBounces and
// ptHairBouncesDlss): fibre events a path may take before they count as surface bounces. The light a
// lock scatters between its fibres is the path's own (hair_bsdf.h.slang has no stand-in for it), and
// light hair sends most of it forward through the lock: under the surface limit it was cut after a
// few fibres. A clear fibre keeps its energy, so the roulette that follows the throughput never ended
// it and white hair took every event of the budget, each with a shadow ray through the hair: past the
// first HAIR_FREE_EVENTS a fibre's path continues at most with HAIR_RR_CAP, bounding its expected
// length whatever the hair's colour.
static const int   HAIR_FREE_EVENTS = 4;
static const float HAIR_RR_CAP      = 0.85F;

// Light that reaches the path while its roughness is at most this is direct: the first surface's, and
// that of surfaces seen through a mirror, glass or a cornea. Above it, light came by a rough bounce.
static const float DIRECT_PATH_ROUGHNESS = 0.1F;
// A path at least this rough meets a scattering medium as the thin-walled preview's diffuse lobes, not
// a random walk (ePtSimplifyScatter; Unreal's bSimplifySSS): its light is a bounce's, which blurs it anyway.
static const float SCATTER_SIMPLIFY_ROUGHNESS = 0.15F;

// The firefly clamp's level for this frame: an accumulation's frames may have their own, looser one
// (fireflyClampStill; 0: the same), a still having its samples and its denoiser against the outliers
// where a DLSS frame has one sample. 0 or less: no clamp.
float fireflyClampLevel()
{
  if(pushConst.fireflyClampStill > 0.0F && !hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
    return pushConst.fireflyClampStill;
  return pushConst.fireflyClampThreshold;
}

// Adds a contribution to the path's radiance. With ePtClampIndirect, an indirect one (`indirect`: the
// path's roughness before it is above DIRECT_PATH_ROUGHNESS) is scaled down to the clamp's level by
// its largest channel, keeping its colour; direct light never is (Unreal's FPathState::AccumulateRadiance).
// The clamp biases the mean where rare bright paths carry the light, so it cuts only bounced light.
void addRadiance(inout PathTracerState pt, float3 contribution, bool indirect)
{
  if(indirect && hasFlag(pushConst.flags, PathtracerFlags::ePtClampIndirect))
  {
    const float level   = fireflyClampLevel();
    const float largest = max(contribution.x, max(contribution.y, contribution.z));
    if(level > 0.0F && largest > level)
      contribution *= level / largest;
  }
  pt.radiance += contribution;
}

// Safe accessor for pt.isInside. The field only exists when GLTF_USE_TRANSMISSION or
// GLTF_USE_VOLUME is compiled in (without those, a ray can never enter a medium so the
// answer is always false). Used by call sites that aren't themselves gated, e.g.
// MeshState construction in the mesh-hit path or the DLSS clear-glass guide.
bool ptIsInside(PathTracerState pt)
{
#if GLTF_USE_TRANSMISSION || GLTF_USE_VOLUME
  return pt.isInside;
#else
  return false;
#endif
}

// Whether the path has met nothing yet. Not surfaceDepth alone: a fibre event spends the hair's own
// budget and leaves it at 0, and a ray leaving a fibre is no camera ray. It sees no backplate (one that
// reached the sky replaced the path's light with the backdrop, so black hair showed the sky behind it)
// and captures no first hit.
bool ptOnCameraRay(PathTracerState pt)
{
#if GLTF_USE_HAIR
  return pt.surfaceDepth == 0 && pt.hairBounces == 0;
#else
  return pt.surfaceDepth == 0;
#endif
}

// The light a path's ray meets on sized lights before `tMax` (with `escaped`, the discs of
// directional ones too), each weighed against the light sample of the vertex the ray left. Without
// it the cone samples were their only technique, and they almost never land in a smooth lobe: a
// cornea, wet lips or chrome never showed the key. The sky's sun is the sky's disk, not this.
//
// The radiance is what the light sample carries from that vertex -- singleLightContribution's
// irradiance over the cone's solid angle, a spot's falloff and an IES profile read along the cone's
// axis -- so both techniques estimate the same light. After a delta event (no light sample to weigh
// against) the vertex is the ray's origin. A camera ray sees the lights with ePtLightsVisible only.
float3 sizedLightHits(RayDesc ray, float tMax, bool escaped, PathTracerState pt)
{
  if(ptOnCameraRay(pt) && !hasFlag(pushConst.flags, PathtracerFlags::ePtLightsVisible))
    return float3(0.0F);
  const bool   delta    = pt.lastSamplePdf == DIRAC;
  const float3 vertex   = delta ? ray.Origin : pt.lightPick.pos;
  const int    sunIndex = skySunPunctualIndex();
  float3       sum      = float3(0.0F);
  for(int i = 0; i < punctualLightCount(); i++)
  {
    const GltfLight light = getPunctualLight(i);
    if(i == sunIndex || !(light.intensity > 0.0F))
      continue;
    const bool directional = light.type == LightType::eLightTypeDirectional;
    if(directional && !escaped)
      continue;
    const float omc = sizedLightOneMinusCos(directional, light.position, light.radius, light.angularSizeOrInvRange, vertex);
    if(!(omc > 0.0F))
      continue;
    const bool hit = directional ? directionalLightHit(ray.Direction, light.direction, omc) :
                                   sphereLightHitDistance(ray.Origin, ray.Direction, light.position, light.radius, tMax) >= 0.0F;
    if(!hit)
      continue;
    // Facing the light, so nothing is culled: the light sample's own test was its pick's
    const float3 toLight = directional ? -light.direction : normalize(light.position - vertex);
    LightContrib contrib = singleLightContribution(light, vertex, toLight, float2(0.0F));
    if(all(contrib.intensity == float3(0.0F)))
      continue;
    const float3 radiance = contrib.intensity * evalIesFactor(pushConst.gltfScene, light, vertex) / (M_TWO_PI * max(omc, 1e-10F));
    const float  w        = delta ? 1.0F : misPowerWeight(pt.lastSamplePdf, sizedLightNeePdf(pt.lightPick, light, omc));
    sum += w * radiance;
  }
  return sum;
}

// Handles the in-volume segment before normal surface shading:
// - Always applies segment transmittance when GLTF_USE_VOLUME is enabled.
// - Handles scatter-only follow-up (cone advance, NEE, RR) when a real scatter happens.
// `freeFlight`: the step's free-flight distance when it was drawn before the trace (FreeFlight).
PathStepResult processVolumeSegment(IRaytracer raytracer, float hitDistance, inout RayDesc ray, inout PathTracerState pt, inout uint seed, FreeFlight freeFlight)
{
#if GLTF_USE_VOLUME
  if(pt.isInside && hasVolumeMedium(pt.volumeMedium))
  {
#if GLTF_USE_SCATTER
    // handleVolumeScatter may rotate/translate the ray during a true scatter event.
    // Keep the pre-scatter ray state for HG cosine evaluation and cone travel update.
    const float3 wiBeforeScatter     = ray.Direction;
    const float3 originBeforeScatter = ray.Origin;
#endif

    float  walkGuide  = 0.0F;
    float3 walkNormal = float3(0.0F);
#if GLTF_USE_SCATTER
    walkGuide  = pt.walkGuide;
    walkNormal = pt.walkNormal;
#endif
    const bool scattered = handleVolumeScatter(pt.volumeMedium, hitDistance, ray, pt.throughput, pt.lastSamplePdf, seed, freeFlight, walkGuide, walkNormal);
#if GLTF_USE_SCATTER
    pt.walkGuide = walkGuide;
#endif
    if(scattered)
    {
#if GLTF_USE_SCATTER
      pt.scatterBounces++;
      pt.nullDistance = 0.0F;  // a scatter is a vertex: the next light hit is weighed from here

      // Scatter events translate the cone through the medium segment (no slant correction).
      pt.cone.width += pt.cone.spreadAngle * length(ray.Origin - originBeforeScatter);
      // With the diffuse exit the walk gathers nothing inside: the surface it leaves through is
      // shaded instead (gltf_pathtrace.slang), with the usual surface NEE and MIS.
      if(!hasFlag(pushConst.flags, PathtracerFlags::ePtVolumeDiffuseExit))
      {
        pt.exitedVolumeWalk = true;
        // A walk leaves the path's roughness as it was: under a surface seen directly, its light is direct
        addRadiance(pt, volumeScatterNEE(raytracer, pt.volumeMedium, ray.Origin, wiBeforeScatter, pt.throughput, seed),
                    pt.pathRoughness > DIRECT_PATH_ROUGHNESS);
      }
      else
        pt.lastSamplePdf = DIRAC;  // no light sample at this scatter: a light the next segment reaches has weight 1
      // A scatter in a medium no surface walk entered (the camera's) is a diffuse event like any other
      if(pt.volumeMaterial < 0)
      {
        pt.prefixIndirect = pt.pathRoughness > DIRECT_PATH_ROUGHNESS;
        pt.pathRoughness  = 1.0F;
      }

      if(pt.scatterBounces >= VOLUME_FREE_BUDGET)
      {
        float rrPcont = min(max(pt.throughput.x, max(pt.throughput.y, pt.throughput.z)) + RR_PCONT_FLOOR, RR_PCONT_CAP_VOLUME);
        if(rand(seed) >= rrPcont)
          return PathStepResult.eBreak;
        pt.throughput /= rrPcont;
      }
      if(pt.scatterBounces >= VOLUME_HARD_CAP)
        return PathStepResult.eBreak;
      return PathStepResult.eVolumeContinue;  // skips surface-depth increment in the main loop
#endif                                        // GLTF_USE_SCATTER
    }
  }
#endif  // GLTF_USE_VOLUME

  return PathStepResult.eOutOfVolume;
}


// Primary ray missed geometry: DLSS far hit + optional backplate (not part of the lighting integral).
// Returns true when radiance is final and path tracing should stop.
bool tryPrimaryMissBackplate(RayDesc ray, SceneFrameInfo* frameInfo, inout PathTracerState pt)
{
  pt.solid = false;
  // Sky/background: represent it as a point at infinity along the (world-space) view direction.
  // processPixel projects this with w=0 (camera translation cancels). NDC depth for a miss is written
  // as far (1.0) on the !hasSolidHit branch, so this value is consumed only by motion vectors.
  pt.firstHitPos = ray.Direction;
#if USE_GUIDE_SHADER
  // Sky is fixed in world space; the previous-frame view direction is identical (w=0 yields camera-only motion).
  pt.firstHitPosPrev = ray.Direction;
#endif

  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseSolidBackground))
  {
    pt.radiance = frameInfo->backgroundColor;
    return true;
  }

  // Authored sky: the visible background is evaluated analytically, sun and all. Sampling the
  // baked lat-long here instead would show a sky with no sun in it -- see sky_background.h.slang.
  if(hasAuthoredSkyBackground(frameInfo->envSkyType))
  {
    const float3 dir = quatRotateInverse(frameInfo->envRotation, ray.Direction);
    pt.radiance      = evalAuthoredSkyBackground(dir, frameInfo) * frameInfo->envIntensity;
    return true;
  }

  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseHdrEnvironment) && frameInfo->envBlur > 0)
  {
    float3 dir = quatRotateInverse(frameInfo->envRotation, ray.Direction);
    float2 uv  = getSphericalUv(dir);
    // The physical sky's disk is not in its baked image, so it is added here as on the unblurred
    // path; a no-op for every other environment.
    const float3 blurred = smoothHDRBlur(texturesHdr[HDR_IMAGE_INDEX], uv, frameInfo->envBlur).xyz;
    pt.radiance          = (blurred + evalSunDiskBackground(dir, frameInfo)) * frameInfo->envIntensity;
    return true;
  }

  return false;
}

SampleResult packSampleResult(PathTracerState pt)
{
  SampleResult sampleResult = {};
  sampleResult.radiance     = float4(pt.radiance, pt.solid ? 1 : 0);
  sampleResult.hitPosition  = pt.firstHitPos;
#if USE_GUIDE_SHADER
  sampleResult.hitPositionPrev             = pt.firstHitPosPrev;
  sampleResult.guideOutput.albedo          = float16_t4(pt.guide.albedo, pt.solid ? 1.0h : 0.0h);
  sampleResult.guideOutput.normalRoughness = pt.guide.normalRoughness;
#if USE_DLSS_SHADER
  sampleResult.guideOutput.specularAlbedo  = pt.guide.specularAlbedo;
  sampleResult.guideOutput.specularHitDist = pt.guide.specularHitDist;
#if MAT_EXT_DLSS_NR
  sampleResult.guideOutput.nrMask = pt.guide.nrMask;
#endif  // MAT_EXT_DLSS_NR
  sampleResult.guideOutput.responsivity = pt.guide.responsivity;
  sampleResult.guideOutput.psrValid = pt.guide.psrValid;
  sampleResult.guideOutput.psrPos   = pt.guide.psrPos;
#endif  // USE_DLSS_SHADER
#endif  // USE_GUIDE_SHADER
  return sampleResult;
}


#endif  // PATHTRACE_FUNCTIONS_H
