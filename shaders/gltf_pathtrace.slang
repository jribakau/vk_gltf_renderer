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

#ifndef AVAILABLE_SER
#define AVAILABLE_SER 1
#endif

#ifndef WIREFRAME
#define WIREFRAME 0
#endif

// VK_KHR_opacity_micromap: an entry point whose ray queries traverse a BLAS carrying an opacity
// micromap must opt in with the OpacityMicromapIdKHR execution mode (VUID-vkCmdTraceRaysKHR-micromap-11636).
// The capability it declares is only valid on a device with the extension, so the host sets USE_OMM
// only when the loaded scene has micromaps; the embedded SPIR-V is built without it.
#ifndef USE_OMM
#define USE_OMM 0
#endif
// The Time view (pixel_time.h.slang): each pixel's thread timed by the device's clock. The capability
// it declares needs VK_KHR_shader_clock, so the host sets it only for that view, where the device has it.
#ifndef USE_PIXEL_TIME
#define USE_PIXEL_TIME 0
#endif
// clang-format off: the SPIR-V ids (%ommEnable) must stay as written
#if USE_OMM
#define RAYQUERY_USE_OPACITY_MICROMAP(entryPoint)                                                                      \
  spirv_asm                                                                                                            \
  {                                                                                                                    \
    OpCapability RayTracingOpacityMicromapExecutionModeKHR;                                                            \
    OpExtension "SPV_KHR_opacity_micromap";                                                                            \
    %ommEnable = OpSpecConstantTrue $$bool;                                                                            \
    OpExecutionModeId $entryPoint OpacityMicromapIdKHR %ommEnable                                                      \
  }
#else
#define RAYQUERY_USE_OPACITY_MICROMAP(entryPoint)
#endif
// clang-format on

// This is a debug flag to enable/disable debug output
// For example, to use print statements in the shader, set this to true.
// Add printf and enable the Validation layer debug output.
static bool doDebug = false;

#include "gltf_material_eval.h.slang"
#include "gltf_scene_io.h.slang"
#include "water.h.slang"
#if GLTF_USE_HAIR
#include "hair_bsdf.h.slang"
#endif
#include "adaptive_stop.h.slang"
#include "hit_signal.h.slang"
#include "pixel_time.h.slang"
#include "gltf_vertex_access.h.slang"
#include "shaderio.h"
#include "get_hit.h.slang"
#include "dlss_util.h"
#include "common.h.slang"
#include "aerial_perspective.h.slang"
#include "contribution_roulette.h.slang"
#include "radiance_cache.h.slang"
#include "skin_slab.h.slang"
#include "skin_spread.h.slang"
#include "aperture.h.slang"
#include "sky_bruneton_radiance.h.slang"
#include "sky_clouds.h.slang"
#include "sky_clouds_air.h.slang"
#include "sky_clouds_io.h.slang"

// NVSHADERS are under: https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders
#include "nvshaders/functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/light_contrib.h.slang"
#include "nvshaders/pbr_ggx_microfacet.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sample_blur.h.slang"
#if WIREFRAME
#include "nvshaders/wireframe.h.slang"
#endif


// clang-format off

// Bindings
[[vk::push_constant]]                           ConstantBuffer<PathtracePushConstant> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]    Texture2D allImages[];
[[vk::binding(BindingPoints::eSamplers, 0)]]    SamplerState allSamplers[];
[[vk::binding(BindingPoints::eTexturesHdr, 0)]] Sampler2D texturesHdr[];
[[vk::binding(BindingPoints::eTlas, 1)]]        RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::eOutImages, 1)]]   RWTexture2D<float4> outImages[];
[[vk::binding(BindingPoints::eOutDepth, 1)]]    RWTexture2D<float>  outDepth;

// HDR Environment
[[vk::binding(EnvBindings::eImpSamples, 2)]]    StructuredBuffer<EnvAccel> envSamplingData;
// The MIS-compensated table, for light samples a BSDF sample of the same vertex is weighed against
[[vk::binding(EnvBindings::eImpSamplesCompensated, 2)]] StructuredBuffer<EnvCompensatedCell> envSamplingCompensated;

// Set 3 is SkyBruneton's runtime set -- the four scattering LUTs and the atmosphere they were
// built from -- bound for aerial perspective alone. The bake reads the same set at index 1 of its
// own layout; a set number is a property of the pipeline, not of the resources, so the shared
// header takes them as arguments rather than declaring them.
[[vk::binding(0, 3)]] Sampler2D<float3>                       g_transmittance;
[[vk::binding(1, 3)]] Sampler3D<float4>                       g_scattering;
[[vk::binding(2, 3)]] Sampler3D<float4>                       g_singleMie;
[[vk::binding(3, 3)]] Sampler2D<float3>                       g_irradiance;
[[vk::binding(4, 3)]] ConstantBuffer<SkyAtmosphereParameters> g_atmosphere;

// Set 4 is SkyClouds' runtime set: the cloud image the physical sky's bake composed into the
// lighting environment, composed here over what the camera and mirrors see. Clear, it is (0, 0, 0, 1)
// everywhere, which gives the sky back unchanged. With it the images that hold the clouds as the
// camera sees them from where it is: the sky images that follow it, each one's depth image, and the
// view image.
[[vk::binding(CLOUD_IMAGE_BINDING, 4)]]   Sampler2D<float4>     g_clouds;
[[vk::binding(CLOUD_PREVIEW_BINDING, 4)]] Sampler2D<float4>     g_cloudsPreview;
[[vk::binding(CLOUD_VIEW_BINDING, 4)]]    Sampler2D<float4>     g_cloudView;
[[vk::binding(CLOUD_FOLLOW_BINDING, 4)]]  Sampler2D<float4>     g_cloudFollow[2];
[[vk::binding(CLOUD_DEPTH_BINDING, 4)]]   Sampler2D<float>      g_cloudDepth[3];
[[vk::binding(CLOUD_SHADOW_BINDING, 4)]]  Sampler2D<float>      g_cloudShadow;

// The share of the sun's light the clouds let through to a place in the scene: their shadow on the
// land (the shadow image, baked with the clouds), read where the sun's ray through the place comes
// up from the ground. 1 under a clear sky and for every sky but the physical one. Past the image's
// edge, kilometres out, the land's mean share.
float cloudSunShare(float3 worldPos, SceneFrameInfo* frameInfo)
{
  if(frameInfo->sunDisk == nullptr || !(frameInfo->sunDisk->cloudTop > 0.0F))
    return 1.0F;
  const float3 p      = quatRotateInverse(frameInfo->envRotation, worldPos) * 0.001F;
  const float3 sunDir = quatRotateInverse(frameInfo->envRotation, frameInfo->sunDirection);
  const float2 uv     = cloudShadowUv(p, sunDir);
  return lerp(frameInfo->sunDisk->sunlitGround, g_cloudShadow.SampleLevel(uv, 0.0F), cloudShadowWeight(uv));
}

// The physical sky along a ray, without the solar disk: its radiance, evaluated per ray through the
// same function the bake uses, the clouds over it. Per ray rather than read from the baked image so
// what the camera, a mirror or still water sees does not depend on the bake resolution -- the
// horizon, where sky meets ground in a hard edge, is what a lat-long would otherwise smear, and the
// clouds' edges what its 0.35 degrees would.
// `cloudsLeave` is what the clouds along the ray leave of what is behind them, 1 with none.
float3 evalPhysicalSky(float3 worldDir, SceneFrameInfo* frameInfo, out float cloudsLeave)
{
  cloudsLeave         = 1.0F;
  const float3 dir    = quatRotateInverse(frameInfo->envRotation, worldDir);
  const float3 sunDir = quatRotateInverse(frameInfo->envRotation, frameInfo->sunDirection);
  // observerRadius is the bake's: selecting the physical sky commits a bake at the top of the frame,
  // before any frame that draws it is recorded (GltfRenderer::applyPendingEnvironmentRefresh).
  const SkySunDisk disk = *frameInfo->sunDisk;
  float3           sky  = evalBrunetonSkyRadiance(g_atmosphere, g_transmittance, g_scattering, g_singleMie, g_irradiance,
                                                  disk.observerRadius, dir, sunDir, disk.sunlitGround);
  // The clouds over it, as the bake composes them, above the horizon. The disk goes on top, dimmed by
  // the clouds in front of it (evalPhysicalSkyBackground).
  if(dir.y > disk.horizonMu)
  {
    float4 clouds;
    float  depth = CLOUD_NO_DEPTH;  // how far the clouds are along the ray, km
    if(disk.cloudPreview != 0)
    {
      // a drag's preview, as the scene's origin sees it
      clouds = g_cloudsPreview.SampleLevel(cloudImageUv(dir, asin(disk.horizonMu)), 0.0F);
    }
    else
    {
      // The sky image the frame takes was baked from one place (its anchor), and the clouds are a
      // place too, kilometres up: a camera that stands off the anchor sees each cloud where the
      // anchor saw the same point of it. The depth image says how far the clouds are; the first read
      // of it is where the camera looks, the second where that put the cloud in the image.
      const int    front        = clamp(frameInfo->cloudFront, 0, 2);
      const float3 eye          = frameInfo->cloudEye.xyz;
      const float  minElevation = asin(brunetonHorizonMu(g_atmosphere, g_atmosphere.bottomRadius + frameInfo->cloudEye.w));
      float3       look         = dir;
      depth                     = g_cloudDepth[front].SampleLevel(cloudImageUv(dir, minElevation), 0.0F);
      if(any(eye != 0.0F))
      {
        look  = cloudDirFromAnchor(dir, eye, depth);
        depth = cloudDepthAlong(dir, eye, look, g_cloudDepth[front].SampleLevel(cloudImageUv(look, minElevation), 0.0F));
        look  = cloudDirFromAnchor(dir, eye, depth);
      }
      const float2 uv = cloudImageUv(look, minElevation);
      if(front == 0)
        clouds = g_clouds.SampleLevel(uv, 0.0F);
      else
        clouds = g_cloudFollow[front - 1].SampleLevel(uv, 0.0F);
    }
    // ... and where the camera looks, the view image, at the screen's resolution, shifted likewise
    // from where it was baked. The two hold the same clouds, the view's sharper; the texel composes
    // linearly, so a blend of them is too.
    if(frameInfo->cloudViewForward.w != 0.0F)
    {
      const float3 seen  = cloudDirFromAnchor(dir, frameInfo->cloudViewEye.xyz, depth);
      const float3 view  = cloudViewUv(seen, frameInfo->cloudViewRight.xyz, frameInfo->cloudViewRight.w, frameInfo->cloudViewUp.xyz,
                                       frameInfo->cloudViewUp.w, frameInfo->cloudViewForward.xyz);
      const float  share = cloudViewShare(view.z);
      if(share > 0.0F)
      {
        const float2 viewUv = float2(view.x * frameInfo->cloudViewUvScaleX, view.y * frameInfo->cloudViewUvScaleY);
        clouds              = lerp(clouds, g_cloudView.SampleLevel(viewUv, 0.0F), share);
      }
    }
    sky         = composeClouds(sky, clouds);
    cloudsLeave = clamp(clouds.w, 0.0F, 1.0F);
  }
  else if(disk.cloudTop > 0.0F)
  {
    // Below the horizon the image does not reach: the air in front of the ground, all of it under the
    // clouds, under their shade (sky_clouds_air.h.slang). The ground's own light is the sunlit share's.
    const float3 air = evalBrunetonSkyInScatter(g_atmosphere, g_transmittance, g_scattering, g_singleMie, disk.observerRadius, dir, sunDir);
    sky -= cloudShadeOfClearSky(g_atmosphere, g_transmittance, g_irradiance, disk.observerRadius, dir, sunDir, disk.cloudTop,
                                disk.sunlitGround, disk.cloudSkyShare, air);
  }
  return sky;
}

float3 evalPhysicalSky(float3 worldDir, SceneFrameInfo* frameInfo)
{
  float unused;
  return evalPhysicalSky(worldDir, frameInfo, unused);
}

// What a camera ray or a mirror bounce sees of the physical sky: the sky per ray, plus the solar disk
// through the clouds drawn in front of it, pixel by pixel, so a cloud's edge crosses the disk.
float3 evalPhysicalSkyBackground(float3 worldDir, SceneFrameInfo* frameInfo)
{
  float        cloudsLeave;
  const float3 sky = evalPhysicalSky(worldDir, frameInfo, cloudsLeave);
  return sky + evalSunDiskBackground(quatRotateInverse(frameInfo->envRotation, worldDir), frameInfo) * cloudsLeave;
}

// A lobe whose pdf at the sampled direction is above this, per steradian, is narrower than about a
// degree: three of the lighting field's texels (kEnvBakeSize, 1024 x 512, 0.35 degrees). Still water,
// the facing slopes of near waves, polished stone and car paint reflect a sky that sharp; GGX at alpha
// 0.009 peaks here. Their rays see the physical sky per ray, as a mirror's do (envColor below).
static const float SKY_NARROW_LOBE_PDF = 1000.0F;

// Specialization constants
[[vk::constant_id(0)]]  int USE_SER = 1;
[[vk::constant_id(1)]]  int USE_DLSS_TRANSP = 0; // Dlss::TransparencyMode: 1 = primary surface replacement through clear glass
// ptInlineShadow: the RTX technique's shadow rays as inline ray queries in raygen (traceShadowInline,
// the default: 12-18% off the path trace on 1TEST, 7TEST and 10TEST) instead of through rahitShadow;
// the ray-query technique traces them inline always
[[vk::constant_id(2)]]  int USE_INLINE_SHADOW = 1;

// clang-format on

#include "pathtrace_functions.h.slang"
#include "sobol.h.slang"

//-----------------------------------------------------------------------
// The sample's fixed sampling decisions (ePtSobol): the pixel's jitter, the lens, and at every
// surface vertex its light sample and its BSDF sample, each one 4D Owen-scrambled Sobol point at the
// pixel's sample index (sobol.h.slang). They are numbered in the order a path meets them, so two
// vertices never share one. Everything that draws a varying number of times -- the opacity test, a
// walk, Russian roulette, the hair's offset -- keeps `seed`. Thread-private.
//-----------------------------------------------------------------------
static bool g_sobol         = false;
static uint g_sobolIndex    = 0u;  // the pixel's sample index: the accumulation's samples before it
static uint g_sobolPixel    = 0u;  // the pixel's scramble, the same in every frame
static uint g_sobolDecision = 0u;
static bool g_hitToCount    = false;  // the pixel counts, and its first hit is still to come (pushConst.hitSignal)
// The pixel's mean so far, a luminance, for its shadow rays' roulette (SceneFrameInfo::shadowRayShare);
// 0 where there is none to go by, and every ray is traced
static float g_shadowRest = 0.0F;
// The accumulation's samples a pixel has before its mean is taken for that
static const int SHADOW_ROULETTE_FROM = 8;
#if USE_PIXEL_TIME
static uint g_timeSlot = kPixelTimeUnset;  // the Time view: what the pixel's first camera ray met (pixel_time.h.slang)
#endif
#if USE_LIGHT_RIS
static int2 g_reusePixel       = int2(0);  // ePtLightReuse: the pixel whose record its first hit reads and writes...
static bool g_reuseFirstSample = true;     // ...its frame's first sample, which reads last frame's record (the rest, this frame's)
#endif

#if USE_RADIANCE_CACHE
// The bounce cache (radiance_cache.h.slang), set per pixel in processPixel: the frame's paths read it
// (a DLSS frame the host gave a table), and this pixel's path feeds it
static bool g_cacheOn     = false;
static bool g_cacheUpdate = false;

// The slot of `key`'s cell, taking a free one of its bucket where it has none yet; -1: the bucket is full
int radianceCacheClaim(uint* table, RadianceCacheKey key)
{
  for(int i = 0; i < RC_BUCKET; i++)
  {
    const uint slot = key.bucket + uint(i);
    const uint word = slot * uint(RC_WORDS) + uint(RC_KEY);
    uint       seen = table[word];
    if(seen == 0u)
      InterlockedCompareExchange(table[word], 0u, key.fingerprint, seen);  // another path may take it first
    if(seen == 0u || seen == key.fingerprint)
      return int(slot);
  }
  return -1;
}

// `value` added to the float a word of the table holds, while other paths add to it. Given up after
// sixteen of them got in first, which would take as many paths ending on one cell at the same instant.
void radianceCacheAddFloat(uint* table, uint word, float value)
{
  if(value == 0.0F)
    return;
  uint seen = table[word];
  for(int i = 0; i < 16; i++)
  {
    uint before;
    InterlockedCompareExchange(table[word], seen, asuint(asfloat(seen) + value), before);
    if(before == seen)
      return;
    seen = before;
  }
}

// An update path's sample, into its cell's sums of the frame
void radianceCacheAdd(uint* table, int slot, RadianceCacheSample s)
{
  const uint base = uint(slot) * uint(RC_WORDS);
  [ForceUnroll]
  for(int c = 0; c < 3; c++)
  {
    radianceCacheAddFloat(table, base + uint(RC_SUM_LIGHT + c), s.light[c]);
    radianceCacheAddFloat(table, base + uint(RC_SUM_WEIGHT + c), s.weight[c]);
  }
}
#endif

void beginSobolSample(uint index)
{
  g_sobolIndex    = index;
  g_sobolDecision = 0u;
}

// The next decision's point (only while g_sobol: otherwise the caller draws from `seed`)
float4 nextSobolDecision()
{
  return sobolOwen4(g_sobolIndex, sobolHash(sobolHashCombine(g_sobolPixel, g_sobolDecision++)));
}

//-----------------------------------------------------------------------
// PATH TRACE ONE BOUNCE
//
// This function performs a single bounce of path tracing in the renderer. It traces a ray through the scene,
// handles intersection and shading logic for the first or subsequent bounces, and updates path tracing state
// and bounce scratch data accordingly. It manages ray-scene intersections, material retrieval, handling
// of missed rays (e.g., environment hits), and prepares data for direct/indirect lighting and next events.
//-----------------------------------------------------------------------
// `bounce` arrives default-initialized (BounceScratch's member defaults) and is filled in here.
// DLSS frames (hairCutDlss): a path through hair ends at this fibre event, after its light sample.
// `hairBounces`: the fibre events it has taken before this one. A negative hairCutDlss cuts the
// still's paths too, to measure the tail against a converged image.
bool hairCutHere(bool fibre, int hairBounces)
{
  return fibre && pushConst.hairCutDlss != 0 && (pushConst.hairCutDlss < 0 || hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
         && hairBounces + 1 >= abs(pushConst.hairCutDlss);
}

#if GLTF_USE_VOLUME
// What a mesh's surface is to the media the path is in: mediumBoundary (medium_stack.h.slang) from the
// path's state and the surface's material. Worked out at each place that needs it (the hit, the shadow
// ray's start behind the surface, the crossing) and not kept between them: once the hit has put the
// path in `here`, the state is the same at all three and so is the answer.
// `shadedThin`: scatter materials are shaded as the thin preview here, which has no inside to enter.
MediumBoundary surfaceMedium(PathTracerState pt, GltfShadeMaterial* materials, int materialIndex, bool frontFace, bool shadedThin)
{
  bool walk     = false;  // the current medium scatters
  bool scatters = false;  // the surface's own medium scatters
#if GLTF_USE_SCATTER
  const float3 sigmaS = float3(pt.volumeMedium.scatterCoefficient);
  walk                = pt.isInside && max(sigmaS.x, max(sigmaS.y, sigmaS.z)) > VOLUME_MIN_SCATTER;
  scatters            = materials[materialIndex].scatterStrengthFactor > 0.0F;
#endif
  const bool bounds = surfaceBoundsMedium(materials + materialIndex) && !(shadedThin && scatters);
  // The back of a scattering surface met outside every medium is a way into it, as it was before the
  // stack: a walk that ended at an eyeball goes on through the head from the lid's inside. (An
  // absorbing solid's is its way out: the path was in it unknown, and is outside after.)
  return mediumBoundary(ptMedium(pt), materialIndex, bounds, frontFace || (bounds && scatters && !pt.isInside), walk);
}
#endif

// The light sample's resampling (light_ris.h.slang, ptLightCandidates) is compiled in only where a build
// asks for it: switched off, its loop still cost a still frame 0.8% (7TEST, 272.0 against 269.7 ms,
// 2026-11-05), and switched on it bought too little without the shadows in the weights (the candidate
// stage of ReSTIR; its reuse is what would): docs/improvements.md, "Light samples chosen among candidates".
// USE_LIGHT_RIS itself (default 0) is light_ris.h.slang's, which pathtrace_functions.h.slang includes.

#if USE_LIGHT_RIS
// Pixel `p`'s record in lightReservoirs: in its first or its second half (ePtLightReuseParity), two uint4 a pixel
uint lightReuseSlot(int2 p, bool secondHalf)
{
  const uint width  = pushConst.renderSize.x;
  const uint pixels = width * pushConst.renderSize.y;
  return ((secondHalf ? pixels : 0u) + uint(p.y) * width + uint(p.x)) * 2u;
}
#endif

// How many light samples a surface vertex draws to keep one of (light_ris.h.slang): the host's count at
// the path's first lightCandidatesDepth surface vertices, 1 elsewhere and on a fibre, whose BSDF costs
// three times a surface's to evaluate and whose light mostly comes through the lock anyway.
int lightCandidatesHere(bool fibre, int surfaceDepth)
{
#if USE_LIGHT_RIS
  if(fibre || pushConst.lightCandidates <= 1)
    return 1;
  if(pushConst.lightCandidatesDepth > 0 && surfaceDepth >= pushConst.lightCandidatesDepth)
    return 1;
  return min(pushConst.lightCandidates, LIGHT_RIS_MAX_CANDIDATES);
#else
  return 1;
#endif
}

#if USE_LIGHT_RIS
// What a surface does with light from `toLight`: the BSDF next-event estimation below evaluates (every lobe
// with `allLobes`), with the diffuse exit's Fresnel. `k1` faces the incoming ray.
BsdfEvaluateData evaluateSurfaceForLight(PbrMaterial pbrMat, float3 k1, float3 toLight, bool allLobes, float exitFresnelEta, inout uint seed)
{
  BsdfEvaluateData evalData;
  evalData.k1 = k1;
  evalData.k2 = toLight;
  evalData.xi = float3(rand(seed), rand(seed), rand(seed));
  if(allLobes)
    bsdfEvaluateAll(evalData, pbrMat);
  else
    bsdfEvaluate(evalData, pbrMat);
  if(exitFresnelEta > 0.0F)
    evalData.bsdf_diffuse *= diffuseExitFresnelWeight(dot(pbrMat.N, toLight), exitFresnelEta);
  return evalData;
}

// The material at a light reuse record's shading point, as this point's with the record's shading
// normal: the tangent frame turned with it, its handedness kept. On a still that is the record's own
// point up to the textures across the jitter; on a moving view's reprojected pixel, nearly.
PbrMaterial lightReuseMaterialThere(PbrMaterial m, float3 shading)
{
  const float3 n = normalize(shading);
  const float3 t = m.T - n * dot(n, m.T);
  if(dot(t, t) > 1e-12F)
  {
    const float handed = (dot(cross(m.N, m.T), m.B) < 0.0F) ? -1.0F : 1.0F;
    m.T                = normalize(t);
    m.B                = cross(n, m.T) * handed;
  }
  m.N = n;
  return m;
}
#endif

#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
// The far side of thin skin under a thin-preview hit (skin_slab.h.slang, the slab in pathTraceOneBounce):
// straight in from `origin` along `inward`, the first surface a walk would stop at (INSTANCE_MASK_WALK:
// not the hair, lashes or clothing over it) within `reach`, if it is the same material, which a back
// of an ear or a finger is and an eyeball under a lid or a tooth behind a lip is not. As a hit seen
// from outside that side: getHitState turned its frame toward the probe, into the skin, and it turns
// back out, as a walk's diffuse exit turns its own.
struct SlabFarSide
{
  bool     found;
  float    depth;  // how far in, along `inward`
  HitState hit;
};
SlabFarSide probeSlabFarSide(float3 origin, float3 inward, float reach, int materialID)
{
  SlabFarSide far = {};
  far.found       = false;
  RayDesc probe;
  probe.Origin    = origin;
  probe.Direction = inward;
  probe.TMin      = 0.0F;
  probe.TMax      = reach;
  PIXEL_TIME_COUNT(kPixelTimeCountRays);
  RayQuery rayQuery;
  rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, INSTANCE_MASK_WALK, probe);
  while(rayQuery.Proceed())
  {
  }  // opaque: nothing to decide
  if(rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    return far;
  const GltfRenderNode node = pushConst.gltfScene->renderNodes[rayQuery.CommittedInstanceIndex()];
  if(node.materialID != materialID)
    return far;
  const GltfRenderPrimitive prim = pushConst.gltfScene->renderPrimitives[rayQuery.CommittedInstanceID()];
  const float2              b    = rayQuery.CommittedTriangleBarycentrics();
  HitState hit = getHitState(prim, float3(1.0F - b.x - b.y, b.x, b.y), node.worldToObject, node.objectToWorld,
                             rayQuery.CommittedPrimitiveIndex(), inward);
  hit.nrm       = -hit.nrm;
  hit.geonrm    = -hit.geonrm;
  hit.tangent   = -hit.tangent;
  hit.bitangent = -hit.bitangent;
  hit.frameSign = -hit.frameSign;
  hit.shadowPos = hit.shadowPosOther;  // the terminator offset for the outer side, which the light is on
  far.found     = true;
  far.depth     = rayQuery.CommittedRayT();
  far.hit       = hit;
  return far;
}

// The skin under a point of the thin preview's spread (skin_spread.h.slang, in pathTraceOneBounce):
// straight down from `origin` along `down` within `reach`, the first surface a walk would stop at
// (INSTANCE_MASK_WALK: not the hair, lashes or anything worn lying on it), if it is skin of a scattering
// medium met from outside, any such skin, so the spread runs on across the seam between a head's and a
// body's surfaces. Its frame faces the probe, out of the skin, as a hit seen from outside does.
struct SpreadPoint
{
  bool     found;
  HitState hit;
};
SpreadPoint probeSpreadPoint(float3 origin, float3 down, float reach)
{
  SpreadPoint point = {};
  point.found       = false;
  RayDesc probe;
  probe.Origin    = origin;
  probe.Direction = down;
  probe.TMin      = 0.0F;
  probe.TMax      = reach;
  PIXEL_TIME_COUNT(kPixelTimeCountRays);
  RayQuery rayQuery;
  rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, INSTANCE_MASK_WALK, probe);
  while(rayQuery.Proceed())
  {
  }  // opaque: nothing to decide
  if(rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    return point;
  const GltfRenderNode node = pushConst.gltfScene->renderNodes[rayQuery.CommittedInstanceIndex()];
  if(node.materialID < 0)
    return point;
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  if(materials[node.materialID].scatterStrengthFactor <= 0.0F || !any(materials[node.materialID].scatterDistance > 0.0F))
    return point;
  const GltfRenderPrimitive prim = pushConst.gltfScene->renderPrimitives[rayQuery.CommittedInstanceID()];
  const float2              b    = rayQuery.CommittedTriangleBarycentrics();
  point.hit = getHitState(prim, float3(1.0F - b.x - b.y, b.x, b.y), node.worldToObject, node.objectToWorld,
                          rayQuery.CommittedPrimitiveIndex(), down);
  point.found = point.hit.frontFace;
  return point;
}
#endif

PathStepResult pathTraceOneBounce(IRaytracer raytracer, inout RayDesc ray, inout uint seed, inout PathTracerState pt, inout BounceScratch bounce)
{
  {
    //DirectLight directLight;
    SceneFrameInfo* frameInfo = pushConst.frameInfo;
    HitPayload      payload   = {};

    // Inside a scattering medium, draw the step's free-flight distance first and trace only that
    // far: a miss is the scatter, and the traversal to the surface (and its hit state) is paid only
    // by the steps that reach it. See FreeFlight.
    FreeFlight freeFlight = {};
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
    if(pt.isInside && hasVolumeMedium(pt.volumeMedium))
    {
      freeFlight = sampleFreeFlight(pt.volumeMedium, pt.throughput, seed, pt.walkGuide, pt.walkNormal, ray.Direction);
      if(freeFlight.drawn && isfinite(freeFlight.dist))
        ray.TMax = freeFlight.dist;
    }
#endif
#if USE_PIXEL_TIME
    if(freeFlight.drawn)
      PIXEL_TIME_COUNT(kPixelTimeCountWalkSteps);
#endif

    // Trace the ray through the scene. A walk's step sees only what can bound its medium: it
    // passes through lashes, hair and clothing that reach under the skin (INSTANCE_MASK_WALK).
    raytracer.Trace(ray, payload, seed, freeFlight.drawn ? INSTANCE_MASK_WALK : 0xFF,
                    ptOnCameraRay(pt) || ptIsInside(pt) ? CAMERA_RAY_FLAGS : BOUNCE_RAY_FLAGS);
    ray.TMax = INFINITE;

    // Getting the scene information
    GltfRenderNode*      renderNodes     = pushConst.gltfScene->renderNodes;
    GltfRenderPrimitive* renderPrimitive = pushConst.gltfScene->renderPrimitives;

    // If we hit something, get the hit state
    HitState            hit        = {};
    GltfRenderNode      renderNode = {};
    GltfRenderPrimitive renderPrim = {};
    if(payload.hitT != INFINITE)
    {
      renderNode   = renderNodes[payload.rnodeID];
      renderPrim   = renderPrimitive[payload.rprimID];
      float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
      hit = getHitState(renderPrim, barys, renderNode.worldToObject, renderNode.objectToWorld, payload.primitiveID, ray.Direction);
    }

    // Check if we are on the first ray
    bool firstRay = ptOnCameraRay(pt);

    // Check if we hit the infinite plane
    bool hitInfinitePlane = checkInfinitePlaneIntersection(ray, payload, hit, frameInfo, seed);

    // Then the water's surface, over the floor: before it, it is the hit
    bool       waterFromAbove = false;
    const bool hitWater       = checkWaterIntersection(ray, payload, hit, frameInfo, waterFromAbove);
    if(hitWater)
      hitInfinitePlane = false;
#if USE_PIXEL_TIME
    // the Time view sums a pixel's time by what its first sample's camera ray met
    if(firstRay && g_timeSlot == kPixelTimeUnset)
    {
      const uint instance = kPixelTimeInstance + uint(payload.rnodeID);
      g_timeSlot          = hitWater                  ? kPixelTimeWater :
                            hitInfinitePlane          ? kPixelTimeGround :
                            payload.hitT == INFINITE  ? kPixelTimeSky :
                            instance < uint(frameInfo->pixelTimeSlots) ? instance : kPixelTimeOther;
    }
#endif

    // A segment under the water, outside any mesh's own medium, is charged its absorption. One that
    // never meets the surface or the floor (past the floor's fade) is lost in it.
    if(pt.inWater && !ptIsInside(pt))
    {
      if(payload.hitT == INFINITE)
        return PathStepResult.eBreak;
      pt.throughput *= exp(-GROUND_WATER_ABSORPTION * payload.hitT);
    }

    // Charge the in-volume segment up to this hit, or the unbounded one of a miss, before anything
    // else: the sky, the front-face escape test, material evaluation or emission. With scattering an
    // event is sampled, and a scatter returns here, so a surface the walk never reaches is not shaded;
    // a walk that leaves the mesh scatters or is lost before the sky is considered (with absorption
    // only, its transmittance is 0). A walk that survives to an eyeball or a tooth has paid for the
    // trip, and the front-face clear below then drops the medium. This is the segment's one charge:
    // what reaches the surface -- its emission included -- comes through the medium's transmittance
    // once, and nothing below may charge it again (a second call squared the transmittance of every
    // absorbing medium and of a walk's refractive exit). One call for the hit and the miss alike: the
    // driver inlines a walk's step, its light sample and shadow ray with it, at each call, and two
    // calls made the path tracer's build some 8 s longer (2026-11-07).
    {
      PathStepResult volumeStep = processVolumeSegment(raytracer, payload.hitT, ray, pt, seed, freeFlight);
      if(volumeStep != PathStepResult.eOutOfVolume)
        return volumeStep;
    }

    //-----------------------------------------------------------------------
    // Hitting the environment
    //-----------------------------------------------------------------------
    if(payload.hitT == INFINITE)
    {
      // Check if we hit the backplate
      if(firstRay && tryPrimaryMissBackplate(ray, frameInfo, pt))
        return PathStepResult.eBreak;

#if USE_DLSS_SHADER
      // #DLSS - If capturing specular hit and we hit environment, use max distance (65504 = fp16 max)
      // This happens when a glossy reflection bounces into the sky.
      if(pt.guide.captureSpecularHit)
      {
        pt.guide.specularHitDist    = float16_t(65504.0f);
        pt.guide.captureSpecularHit = false;
      }
#endif  // USE_DLSS_SHADER

      // A ray that left a delta/specular event -- a camera ray, or a mirror bounce -- must see the
      // *background* field, not the lighting one: a mirror has to show a sharp sun. That is safe
      // precisely here and nowhere else -- next-event estimation cannot reach a delta lobe, so the
      // sun cannot be counted twice. Any other bounce gets the disk-free lighting field, plus the
      // sun disk weighted against the NEE sample that also reaches it. See sky_background.h.slang.
      //
      // Every sky that has a background field evaluates it per ray, so what the camera sees does
      // not depend on the bake resolution. The physical sky's is its per-ray radiance plus the disk
      // (sunDisk is non-null for that sky alone); the lighting field it replaces is the same
      // function, baked, so the two agree everywhere but in sharpness.
      float3     envColor;
      float      envPdf = 0.0F;  // unread on a delta bounce: the MIS weight there is 1 whatever it is
      const bool delta  = pt.lastSamplePdf == DIRAC;
      if(delta && hasAuthoredSkyBackground(frameInfo->envSkyType))
      {
        const float3 bgDir = quatRotateInverse(frameInfo->envRotation, ray.Direction);
        envColor           = evalAuthoredSkyBackground(bgDir, frameInfo) * frameInfo->envIntensity;
      }
      else if(delta && frameInfo->sunDisk != nullptr)
      {
        envColor = evalPhysicalSkyBackground(ray.Direction, frameInfo) * frameInfo->envIntensity;
      }
      else
      {
        // The lighting field: the baked image for a sky, the file for an HDR.
        sampleEnvironment(ray.Direction, frameInfo, envColor, envPdf);
        // The vertex this ray left drew its environment sample from the compensated table: it
        // traced this ray (sampleLights' `compensatedEnv`)
        envPdf = environmentCompensatedPdfAt(ray.Direction, frameInfo);
        // A lobe narrower than the field resolves (SKY_NARROW_LOBE_PDF) reflects the physical sky per
        // ray, as sharp as the camera sees it: still water read the field's blurred clouds under the
        // sky's sharp ones. The disk is not in it: it comes below, weighed against its light sample,
        // and so does this against the environment's, a sample of the field that agrees with it but
        // in sharpness.
        if(frameInfo->sunDisk != nullptr && pt.lastSamplePdf > SKY_NARROW_LOBE_PDF)
          envColor = evalPhysicalSky(ray.Direction, frameInfo) * frameInfo->envIntensity;
      }

      // We may hit the environment twice: once via sampleLights() and once when hitting the sky while probing
      // for more indirect hits. This is the counter part of the MIS weighting in sampleLights()
      float misWeight = computeEnvHitMisWeight(pt.lastSamplePdf, envPdf, pt.lightPick);

      // Volume NEE already estimated direct light for a walk. The sky reached by the phase-sampled
      // path would count it again; that path contributes nothing.
#if GLTF_USE_VOLUME
      const bool walkOwnsSky = pt.exitedVolumeWalk;
      pt.exitedVolumeWalk    = false;
#else
      const bool walkOwnsSky = false;
#endif
      if(!walkOwnsSky)
      {
        // The partner of the last vertex's light sample, so the same class: the path before that vertex
        const bool indirect = pt.prefixIndirect;
        addRadiance(pt, pt.throughput * misWeight * envColor, indirect);
        // Sized lights it meets on the way out, a sun's disc among them (not in a medium: it scatters there)
        if(!ptIsInside(pt))
          addRadiance(pt, pt.throughput * sizedLightHits(ray, INFINITE, /*escaped=*/true, pt), indirect);

        // The sun the lighting field leaves out. Without it a lobe narrower than the disk -- smooth
        // glass, a polished mirror -- shows no sun at all, as NEE's cone samples almost never land in it.
        if(!delta)
        {
          const float3 bgDir = quatRotateInverse(frameInfo->envRotation, ray.Direction);
          // through the clouds over where the ray set out, as the light sample it is weighed against
          const float3 disk  = evalSkySunDisk(bgDir, frameInfo) * cloudSunShare(ray.Origin, frameInfo);
          if(any(disk > 0.0F))
            addRadiance(pt, pt.throughput * computeSkySunHitMisWeight(pt.lastSamplePdf, pt.lightPick) * disk * frameInfo->envIntensity, indirect);
        }
      }

      return PathStepResult.eBreak;
    }

    // Sized lights before this hit, by this segment's own state: the medium rules below may leave
    // the medium it ran through. Not from inside one, where the path scatters, nor on a walk's way
    // out, whose light volume NEE gathered (as for an emitter, below).
    {
      bool walkExit = false;
#if GLTF_USE_VOLUME
      walkExit = pt.exitedVolumeWalk;
#endif
      if(!ptIsInside(pt) && !walkExit)
        addRadiance(pt, pt.throughput * sizedLightHits(ray, payload.hitT, /*escaped=*/false, pt), pt.prefixIndirect);
    }

    GltfShadeMaterial* materials     = pushConst.gltfScene->materials;
    GltfTextureInfo*   texInfos      = pushConst.gltfScene->textureInfos;
    int                materialIndex = -1;
    PbrMaterial        pbrMat;
    // A mesh's surface among the media the path is in (surfaceMedium), for its index pair: the ray is
    // on the side of the surface's own medium, and the material whose medium its index is against (-1: air)
    bool mediumOwnSide = false;
    int  mediumAgainst = -1;
    // A diffuse exit's boundary index, for its light-side Fresnel (ePtExitFresnel); 0: none
    float exitFresnelEta = 0.0F;
    // Measured skin shaded as the thin preview, seen from outside, and its medium's single-scattering
    // albedo as the still's walk makes it from the map (set where the material is evaluated)
    bool   previewSkin      = false;
    float3 previewSkinAlpha = float3(0.0F);
    // Volumetric scatter shades as the thin preview at this hit: the preview flag, or a rough path
    // arriving from outside any medium (a walk's own exit, from inside, stays volumetric)
    bool simplifyScatter = hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin)
                           || (hasFlag(pushConst.flags, PathtracerFlags::ePtSimplifyScatter) && !ptIsInside(pt)
                               && pt.pathRoughness >= SCATTER_SIMPLIFY_ROUGHNESS);

#if USE_RADIANCE_CACHE
    // The bounce cache: only a vertex reached through no medium, as the segment came (the rules below
    // may leave one: a walk's exit is the walk's own estimate, not a place to stop)
    bool cacheSurface = g_cacheOn && !ptIsInside(pt) && !pt.inWater && !hitWater && pt.pathRoughness >= RC_PATH_ROUGHNESS;
#endif

    // Compute the ray-cone footprint at the hit point which is used to compute the texture-LOD gradient
    float worldFoot = rayConeWorldFootprint(pt.cone, payload.hitT, hit.geonrm, -ray.Direction);

    //-----------------------------------------------------------------------
    // Hitting the infinite plane
    //-----------------------------------------------------------------------
    if(hitInfinitePlane)
    {
      // Replaced the material at the hit point with the infinite plane material, and its water
      pbrMat = groundMaterial(hit.pos, worldFoot, frameInfo);
      // Under a body of water, what is over the floor is water
      if(pt.inWater)
        pbrMat.ior1 = GROUND_WATER_IOR;

      // Shadow catcher: invisible plane that shows only shadows for compositing
      if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher))
      {
        // The medium in front of the plane was charged above, as for a mesh; the catcher's shadow
        // ray starts in it.
        float3 shadowExtinction = float3(0.0F);
#if GLTF_USE_VOLUME
        shadowExtinction = volumeExtinction(pt.volumeMedium);
#endif

        // Propagate cone through plane bounce before potential early continue from shadow-catcher path.
        pt.cone.width = worldFoot;
        // What the catcher shows is the environment behind it, which already carries the whole
        // atmosphere; aerial perspective at the plane would haze it a second time.
        if(firstRay)
          pt.firstHitHazed = false;
        bool continuePath = handleShadowCatcher(raytracer, hit, pbrMat, ray, pt.radiance, pt.throughput,
                                                pt.lastSamplePdf, seed, ptIsInside(pt), shadowExtinction, frameInfo);
        if(!continuePath)
          return PathStepResult.eBreak;
        return PathStepResult.eEarlyContinue;  // On through the plane, as if it were not there
      }
    }
    //-----------------------------------------------------------------------
    // Hitting the water's surface
    //-----------------------------------------------------------------------
    else if(hitWater)
    {
      pbrMat = waterSurfaceMaterial(hit.pos, hit.geonrm, waterFromAbove, worldFoot, frameInfo);
    }
    //-----------------------------------------------------------------------
    // Hitting the mesh
    //-----------------------------------------------------------------------
    else
    {
      // Find material at the hit point and evaluate the material (PBR material evaluation)

      materialIndex = max(0, renderNode.materialID);

#if GLTF_USE_VOLUME
      // What the surface is to the media the path is in (medium_stack.h.slang); the segment up to here
      // was charged above either way.
      //  - A front face met in a medium that scatters, a walk through skin, is an opening the walk
      //    escaped through (an eye socket, a mouth) or what sits in one (an eyeball, a tooth): the
      //    medium ends there, and the path is back in the one around it, if any.
      //  - In one that only absorbs (glass, a liquid: closed solids) it is something inside the medium,
      //    which the path meets and leaves in it, so that its light sample's shadow ray and the path
      //    from it go on through the medium: the absorbing box's panel in volume_transmittance came out
      //    61% bright when the medium was dropped, its light's leg uncharged. A surface that bounds a
      //    medium of its own is met in it too (oil in a bottle that is one solid of glass, a hand in a
      //    tub): its index is against this medium's, and the path is back in this one on leaving it.
      //  - The back of the medium around the current one, where the two overlap, is no boundary.
      // Only in a medium, or at the back of a surface with one: from outside, a front face is just a
      // way in, and there is nothing to decide.
      if(pt.isInside || (!hit.frontFace && materials[materialIndex].thicknessFactor > 0.0F))
      {
        const MediumState    arrived  = ptMedium(pt);
        const MediumBoundary boundary = surfaceMedium(pt, materials, materialIndex, hit.frontFace, simplifyScatter);
#if USE_NESTED_MEDIA
        if(boundary.passed)
        {
          pt.outerMaterial = -1;
          pt.cone.width    = worldFoot;
          pt.nullDistance += payload.hitT;
          ray.Origin = safeOffsetRay(hit.pos, -hit.geonrm, hit.offsetDistance);  // getHitState turned geonrm to the ray
          return PathStepResult.eEarlyContinue;
        }
#endif
        if(boundary.here.inside != arrived.inside || boundary.here.current != arrived.current)
        {
#if GLTF_USE_SCATTER
          // Light that leaves a walk has scattered: the path is rough from here, as after the diffuse exit
          const float3 sigmaS = float3(pt.volumeMedium.scatterCoefficient);
          if(max(sigmaS.x, max(sigmaS.y, sigmaS.z)) > VOLUME_MIN_SCATTER && pt.scatterBounces > 0)
          {
            pt.prefixIndirect = true;
            pt.pathRoughness  = 1.0F;
          }
#endif
          ptSetMedium(pt, boundary.here, materials);
        }
#if USE_NESTED_MEDIA
        // The medium a surface with one of its own is met in is the one around it, once entered: kept
        // here, at the hit, whether or not the path goes in (medium_stack.h.slang)
        if(boundary.here.outer != pt.outerMaterial)
          pt.outerMaterial = boundary.here.outer;
#endif
        mediumOwnSide = boundary.ownSide;
        mediumAgainst = boundary.against;
      }
#endif

      // Diffuse exit (ePtVolumeDiffuseExit): a scatter walk that reaches the surface from inside
      // leaves through it as a Lambertian point, carrying the colour the walk gave its throughput,
      // and is shaded below like any surface: one next-event estimate at the exit with MIS against
      // the cosine-sampled continuation. The walk gathered nothing inside (pathtrace_functions),
      // so nothing is counted twice, and the boundary's Fresnel and total internal reflection are
      // folded into the exit lobe rather than being skipped by straight-through shadow rays (the
      // estimator production renderers use for dense media). A medium that only absorbs keeps its
      // refractive exit: this is for scattering media.
      bool diffuseExit = false;
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
      const int walkMaterial = pt.volumeMaterial;
      if(ptIsInside(pt) && !hit.frontFace && hasFlag(pushConst.flags, PathtracerFlags::ePtVolumeDiffuseExit))
      {
        const float3 sigmaS = float3(pt.volumeMedium.scatterCoefficient);
        diffuseExit         = max(sigmaS.x, max(sigmaS.y, sigmaS.z)) > VOLUME_MIN_SCATTER;
      }
      if(diffuseExit)
      {
        // out of the walk's medium, into the one around it (a hand in a tub: the water)
        ptSetMedium(pt, mediumLeft(ptMedium(pt)), materials);
        mediumOwnSide       = false;
        mediumAgainst       = -1;
        pt.exitedVolumeWalk = false;
        // getHitState turned the normals towards the ray, i.e. into the medium; the exit faces out.
        // The whole frame turns back, as getHitState turned it: a bitangent left as it was mirrors
        // the normal map's green channel, turning its relief inside out along one axis, and the
        // exit's diffuse went dark where that relief faced the light (a still's sternum, collarbones
        // and neck on measured skin, which the DLSS preview without the walk did not show).
        hit.nrm       = -hit.nrm;
        hit.geonrm    = -hit.geonrm;
        hit.tangent   = -hit.tangent;
        hit.bitangent = -hit.bitangent;
        hit.frameSign = -hit.frameSign;
        hit.shadowPos = hit.shadowPosOther;  // the terminator offset for the outer side, which the light is on
      }
#endif

      float     texGrad = worldFoot * hit.texelDensity * pushConst.texGradScale;
      if(g_hitToCount)
      {
        g_hitToCount = false;
        if(materialIndex < pushConst.hitSignalSlots)
          InterlockedAdd(pushConst.hitSignal[materialIndex * HIT_SIGNAL_BINS + hitSignalBin(worldFoot * hit.texelDensity)], 1u);
      }
      // Whether the ray is on the side of the surface's own medium, for its index pair
      bool ownSide = ptIsInside(pt);
#if GLTF_USE_VOLUME
      ownSide = mediumOwnSide;
#endif
      MeshState mesh    = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, ownSide, texGrad);
      mesh.baseColorVertexMul = hit.color;
      mesh.masks              = hit.masks;
#if GLTF_USE_PROCEDURAL
      mesh.water = sampleWater(pushConst.gltfScene, uint(materialIndex), hit.uv[0]);
#endif
      pbrMat                  = evaluateMaterial(materialIndex, mesh, allImages, allSamplers, texInfos, materials, simplifyScatter,
                                                 hasFlag(pushConst.flags, PathtracerFlags::ePtSkinFaceOn));
      // Under the water, what is outside a surface is water, not air: submerged skin all but loses its
      // sheen, and glass bends less
      if(pt.inWater)
      {
        if(!ptIsInside(pt))
          pbrMat.ior1 = GROUND_WATER_IOR;
        else if(pbrMat.ior2 == 1.0F)
          pbrMat.ior2 = GROUND_WATER_IOR;
      }
#if GLTF_USE_VOLUME
      // In a mesh's medium likewise (medium_stack.h.slang): oil in a bottle that is one solid of glass
      // meets the glass, not air, and so does a spoon in a drink. A medium that only absorbs: a walk's
      // boundary is no smooth glass, and what it meets inside ends it.
      if(mediumAgainst >= 0)
      {
        bool walkAround = false;
#if GLTF_USE_SCATTER
        walkAround = materials[mediumAgainst].scatterStrengthFactor > 0.0F;
#endif
        float againstIor = 1.5F;  // glTF's default, as evaluateMaterial has it without KHR_materials_ior
#if GLTF_USE_IOR
        againstIor = materials[mediumAgainst].ior;
#endif
        if(!walkAround && mediumOwnSide)
          pbrMat.ior2 = againstIor;
        else if(!walkAround)
          pbrMat.ior1 = againstIor;
      }
#endif
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
      // The thin preview's skin as the still's walk shades it (skin_albedo.h.slang). The walk enters by
      // refraction along the view and comes back more at a slant, darker channels the most; it leaves
      // through a Lambertian exit that pays the boundary's Fresnel toward the light. The preview's
      // Lambertian map had neither, so under the sun it read 0.92-0.98 of the still where the sun was
      // steep on the skin, and below that seen at a slant; over a sky the exit's Fresnel averages out.
      // Before the guides, so the albedo guide is the colour shaded.
      if(simplifyScatter && hit.frontFace && !ptIsInside(pt) && pbrMat.thickness > 0.0F && materials[materialIndex].scatterStrengthFactor > 0.0F
         && any(materials[materialIndex].scatterDistance > 0.0F))
      {
        const bool   faceOn = hasFlag(pushConst.flags, PathtracerFlags::ePtSkinFaceOn);
        const float3 A      = clamp(pbrMat.baseColor, float3(1e-3F), float3(0.999F));
        const float  eta    = clamp(materials[materialIndex].ior, 1.0F, 3.0F);
        previewSkin         = true;
        previewSkinAlpha    = faceOn ? faceOnSingleScatterAlbedo(A) : multiToSingleScatterAlbedo(A, 0.0F);
        if(frameInfo->skinView != 0)
        {
          pbrMat.baseColor = skinViewAlbedo(A, previewSkinAlpha, dot(-ray.Direction, pbrMat.N), eta, faceOn);
          if(hasFlag(pushConst.flags, PathtracerFlags::ePtExitFresnel))
            exitFresnelEta = eta;
        }
      }
      if(diffuseExit)
      {
        // The exit lobe: Lambertian with no specular layer, since the walk already paid the
        // boundary's share on the way in. The walk hands back about the medium's multiple-scattering
        // albedo, already a skin colour; the exit divides it out, so lit from the front the skin is
        // its colour map (as a random walk with albedo inversion renders it), while how far each
        // channel spread under the surface still shows at shadow edges and through thin parts.
        // The albedo divided out is the walk's own medium's, the one its throughput carries, not
        // that of the surface it leaves through: a walk in the face that reaches the back of a lash
        // card (no medium: the 0.05 floor made it twenty times its colour) or the lips' boundary
        // leaves in that surface's colour with the face's tint taken out. Only a path that started
        // in the camera's medium has no material to name and falls back to the surface's. Factors
        // only: a scatter strength or multiscatter texture (which the figures don't use) is not in it.
        // A medium made from its surface's colour (FV_materials_subsurface) already carries the
        // colour of where the walk entered: the walk leaves white.
        const int    albedoMaterial = walkMaterial >= 0 ? walkMaterial : materialIndex;
        const float3 walkAlbedo     = max(materials[albedoMaterial].multiscatterColorFactor
                                              * materials[albedoMaterial].scatterStrengthFactor,
                                          float3(0.05F));
        const float3 exitAlbedo = any(materials[albedoMaterial].scatterDistance > 0.0F) ? float3(1.0F) : pbrMat.baseColor / walkAlbedo;
        const float3 N          = pbrMat.N;
        const float3 Ng         = pbrMat.Ng;
        const float3 T          = pbrMat.T;
        const float3 B          = pbrMat.B;
        const float3 Ns         = pbrMat.Ns;
        pbrMat                  = defaultPbrMaterial(exitAlbedo, 0.0F, 1.0F, N, Ng);
        pbrMat.specular         = 0.0F;
        pbrMat.ior2             = 1.0F;
        pbrMat.T                = T;
        pbrMat.B                = B;
        pbrMat.Ns               = Ns;
        // The boundary the light crosses on its way in, which the camera's side paid at the entry
        // (diffuseExitFresnelWeight): on the light sample and on the continuation alike
        if(hasFlag(pushConst.flags, PathtracerFlags::ePtExitFresnel))
          exitFresnelEta = clamp(materials[materialIndex].ior, 1.0F, 3.0F);
      }

      // Diffuse entry (ePtDiffuseEntry): light from outside enters a scatter walk through a
      // Lambertian lobe, as the walk leaves through one, and not by a rough refraction whose
      // direction follows the view. A dense medium forgets where its light came from within a few
      // events, and the boundary of skin is no smooth glass: it is the light that gets past the
      // specular layer that matters, and that the lobe weights keep. Unreal's path tracer enters
      // subsurface this way; Iray's translucency is a diffuse transmission. The transmission's share
      // moves to the diffuse-transmission lobe whole (the BTDF's tint was already white for a walk,
      // see evaluateMaterial): diffuse transmission T, diffuse reflection 1 - T, as before.
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtDiffuseEntry) && !ptIsInside(pt) && !simplifyScatter
         && pbrMat.thickness > 0.0F && pbrMat.transmission > 0.0F
         && max(pbrMat.scatterCoefficient.x, max(pbrMat.scatterCoefficient.y, pbrMat.scatterCoefficient.z)) > VOLUME_MIN_SCATTER)
      {
        const float t                         = pbrMat.transmission;
        const float dt                        = t + (1.0F - t) * pbrMat.diffuseTransmissionFactor;
        pbrMat.diffuseTransmissionColorFactor = lerp(pbrMat.diffuseTransmissionColorFactor, float3(1.0F), t / dt);
        pbrMat.diffuseTransmissionFactor      = dt;
        pbrMat.transmission                   = 0.0F;
      }
#endif

#if WIREFRAME
      // Wireframe is only used for the camera-visible surface, so compute it once here on the
      // first hit (instead of every Trace() like the old payload-routed version did). All inputs
      // are already in scope, so this avoids carrying wireframeAlpha + pixelAngle in the payload.
      if(firstRay)
      {
        uint3 triIdx = getTriangleIndices(renderPrim, payload.primitiveID);
        pt.firstHitWireframeAlpha =
            computeMeshWireframe(renderNode.objectToWorld, getVertexPosition(renderPrim, triIdx.x),
                                 getVertexPosition(renderPrim, triIdx.y), getVertexPosition(renderPrim, triIdx.z),
                                 payload.bary, payload.hitT, pushConst.pixelAngle, hit.geonrm, -ray.Direction);
      }
#endif
    }

#if USE_DLSS_SHADER
    // #DLSS - Capture specular hit distance on depth 1 (first reflection bounce)
    if(pt.guide.captureSpecularHit)
    {
      pt.guide.specularHitDist    = float16_t(min(payload.hitT, 65504.0f));
      pt.guide.captureSpecularHit = false;
      // How the reflection's image moves (RR's reflection motion): this point's motion mirrored in the
      // first hit's plane, which moves with the first hit (mirrorImageDelta). The ground and the water
      // stay put; a first hit seen through glass has its guides elsewhere and no motion of its own here.
      float3 hitDelta = float3(0.0F);
      if(!hitInfinitePlane && !hitWater)
      {
        const float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
        hitDelta = dlssPrevHitPos(renderNode, renderPrim, payload.rnodeID, payload.rprimID, payload.primitiveID, barys, hit.pos) - hit.pos;
      }
      const float3 mirrorDelta = pt.guide.psrValid ? float3(0.0F) : pt.firstHitPosPrev - pt.firstHitPos;
      const float3 mirrorN     = float3(pt.guide.normalRoughness.xyz);
      const float  mirrorN2    = dot(mirrorN, mirrorN);
      // a first hit that left no guide normal: the point's own motion, unmirrored (never a NaN for RR)
      pt.guide.specularDelta = float16_t3(mirrorN2 > 1e-8F ? mirrorImageDelta(hitDelta, mirrorDelta, mirrorN * rsqrt(mirrorN2)) : hitDelta);
    }
#endif  // USE_DLSS_SHADER

    //-----------------------------------------------------------------------
    // First-hit captures (NDC-depth always; denoiser guide buffers when on).
    //-----------------------------------------------------------------------
    if(firstRay)
    {
      // Always record the first hit position; drives the NDC depth output used by
      // selection/picking, independent of the denoisers.
      pt.firstHitPos = hit.pos;

#if USE_GUIDE_SHADER
#if USE_DLSS_SHADER
      // #DLSS instance and deformation motion - Previous-frame world position of this surface point.
      // Only DLSS's motion vectors read it: the other denoisers' variant leaves it out.
      pt.firstHitPosPrev = hit.pos;
      if(!hitInfinitePlane && !hitWater)
      {
        const float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
        pt.firstHitPosPrev = dlssPrevHitPos(renderNode, renderPrim, payload.rnodeID, payload.rprimID, payload.primitiveID, barys, hit.pos);
      }
      // the hit's own motion, which a reflection in it moves with (primary surface replacement below
      // puts the surface behind glass in firstHitPosPrev)
      const float3 firstHitDelta = pt.firstHitPosPrev - hit.pos;
#endif  // USE_DLSS_SHADER

      // Base-color guide albedo (shared by DLSS and the other denoisers). Stored temporarily, written at end to
      // reduce live state. Without RR's guides it is the surface colour, metal included; DLSS-RR's is the
      // diffuse albedo, its specular share going to the specular guide (NVIDIA's RR guide). Transmission is not
      // taken off: skin's (up to 0.95) would leave RR a near-black albedo.
      float3 guideAlbedo = pbrMat.baseColor.xyz;
#if USE_DLSS_SHADER
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss) || hasFlag(pushConst.flags, PathtracerFlags::ePtRrStillGuides))
        guideAlbedo *= 1.0F - pbrMat.metallic;
      // #DLSS specular guide - specular albedo + clear-glass albedo refinement. DLSS-only quality
      // path, compiled out of the variant without DLSS's guides.
      float3 specularAlbedo = dlssSpecularAlbedo(pbrMat, ray.Direction);
      float3 guideNormal    = pbrMat.N;
      // RR reads linear (perceptual) roughness; the material keeps GGX alpha
      float  guideRoughness = sqrt(pbrMat.roughness.x);
      // also on the accumulation's frames, so the still's guides see the iris behind a cornea too (only
      // DLSS's frames read the replaced depth and motion)
      float replacedDepth = 0.0F;  // how far behind the hit DLSS's depth is put (primary surface replacement)
      if(USE_DLSS_TRANSP == 1 && dlssIsClearGlass(pbrMat))
      {
        // #DLSS primary surface replacement: describe what is seen through the glass. The albedo
        // keeps the glass's Fresnel share, and the surface behind comes as it comes through: tinted
        // by every glass and pane on the way, dimmed by a volume's and the water's absorption
        // (DlssGlassPsr::through). What reflects is the glass's own and, through it, the surface's: a
        // metal or a gloss behind a window keeps its specular albedo as seen directly. Added, not
        // blended by the Fresnel, which would leave a matte surface behind glass about F^2 of
        // specular albedo against the glass's own reflection of F. Depth, normal and motion are the
        // surface's behind, unless the glass's reflection outweighs what comes through
        // (dlssReflectionShare: water from a low camera, a window edge-on): then its own, which the
        // reflection moves with. The roughness follows the share.
        DlssGlassPsr psr     = hitWater ? dlssTraceWater(pbrMat, hit, ray.Direction, frameInfo)
                                        : dlssTraceClearGlass(pbrMat, hit, materialIndex >= 0 ? rayIntoEyeChamber(ray.Direction, pbrMat.N, hit, materials + materialIndex) : ray.Direction, ptIsInside(pt));
        float        ndotv   = saturate(dot(pbrMat.N, -ray.Direction));
        float        fresnel = schlickFresnel((pbrMat.ior2 > 0.0f) ? (pbrMat.ior2 / pbrMat.ior1) : 1.5f, ndotv);
        if(bsdfIsPlate(pbrMat))
          fresnel = 2.0F * fresnel / (1.0F + fresnel);  // a pane's two faces
        specularAlbedo       = saturate(specularAlbedo + (1.0F - fresnel) * psr.through * psr.specularAlbedo);
        guideAlbedo = dlssEnforceMinAlbedo(dlssSeeThroughAlbedo(psr.albedo, pbrMat.baseColor.xyz, fresnel, psr.through), specularAlbedo);
        const float reflectionShare = dlssReflectionShare(fresnel, psr.through);
        if(psr.valid)
          guideRoughness = lerp(sqrt(psr.roughness), guideRoughness, reflectionShare);
        if(psr.valid && reflectionShare <= DLSS_REFLECTION_KEEPS_SURFACE)
        {
          guideNormal        = psr.normal;
          pt.guide.psrValid  = true;
          // where the eye sees it: nearer under water and through solid glass (dlssSeenScale)
          pt.guide.psrPos    = hit.pos + ray.Direction * psr.seenLength;
          pt.firstHitPosPrev = pt.guide.psrPos + psr.posDelta;
          replacedDepth      = psr.seenLength;
        }
        // Its own reflection, on every DLSS frame (dlssTraceMirror): how far it reaches and how its
        // image moves, for RR's specular hit distance and reflection motion. The bounce below then
        // captures nothing. Not on an eye: its tear film's image is set below, with no ray.
        const bool tearFilm = materialIndex >= 0 && materials[materialIndex].chamberIor > 0.0F;
        if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss) && !tearFilm)
        {
          const DlssMirror mirror  = dlssTraceMirror(pbrMat, hit, ray.Direction, firstHitDelta, frameInfo);
          pt.guide.specularHitDist = float16_t(mirror.hitDist);
          pt.guide.specularDelta   = float16_t3(mirror.delta);
        }
      }
#if GLTF_USE_TRANSMISSION
      // #DLSS an eye's tear film: its reflection lies where the curved film shows it, half its radius
      // behind it (mirror_image.h.slang), whichever lobe this sample took, so the catchlight goes
      // with the eye while the camera moves. The bounce below then captures nothing. That point is
      // fixed to the eye, so it moves as the film does.
      if(materialIndex >= 0 && hit.frontFace && materials[materialIndex].chamberIor > 0.0F)
      {
        const float radius = dlssHitMirrorRadius(renderNode, renderPrim, payload.primitiveID);
        if(radius > 0.0F)
        {
          pt.guide.specularHitDist = float16_t(mirrorImageDistance(radius, replacedDepth));
          pt.guide.specularDelta   = float16_t3(firstHitDelta);
        }
      }
#endif
      pt.guide.specularAlbedo = float16_t3(specularAlbedo);
#if MAT_EXT_DLSS_NR
      if(materialIndex >= 0)
        pt.guide.nrMask = float16_t4(materials[materialIndex].nrMask);
#endif  // MAT_EXT_DLSS_NR
      pt.guide.normalRoughness = float16_t4(float16_t3(guideNormal), float16_t(guideRoughness));
#else
      pt.guide.normalRoughness = float16_t4(float16_t3(pbrMat.N), float16_t(pbrMat.roughness.x));
#endif  // USE_DLSS_SHADER
      pt.guide.albedo = float16_t3(guideAlbedo);
#endif  // USE_GUIDE_SHADER
    }

#if USE_DLSS_SHADER && GLTF_USE_TRANSMISSION
    // A film's cover (opacityFilmKept): the traversal kept this hit whatever its opacity, and the
    // path goes on through it here by its open share, as a still's any-hit lets a sample through.
    // After the guides: the camera's film is the first hit on every frame, whichever way this one's
    // path went. The crossing is no event: nothing is lit or scattered here, and the light the ray
    // reaches beyond is weighed against the vertex before (a film passes that vertex's shadow ray by
    // the same share, shadowCrossSurface). It spends the glass's budget, as a pane does, then bounces.
    if(materialIndex >= 0 && materials[materialIndex].opacityFilm != 0 && hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
    {
      const float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
      // the opacity the traversal read: a weave's by the ray's direction, in the instance's space
      const float cover = getOpacity(renderNode, renderPrim, payload.primitiveID, barys, mul(float4(ray.Direction, 0.0F), renderNode.worldToObject).xyz);
      if(cover < 1.0F && rand(seed) >= cover)
      {
        pt.cone.width = worldFoot;
        pt.nullDistance += payload.hitT;
        ray.Origin = safeOffsetRay(hit.pos, -hit.geonrm, hit.offsetDistance);  // getHitState turned geonrm to the ray
        if(pt.glassBounces < pushConst.glassBudget)
          pt.glassBounces++;
        else
          pt.surfaceDepth++;
        return PathStepResult.eEarlyContinue;
      }
    }
#endif

    // The material's own roughness, before the regularisation below, read as the shadow rays read it
    // (crossingRoughnessMetallic): whether a refraction through it is a clear crossing
    // (CLEAR_CROSSING_ROUGHNESS) is decided by it, texel by texel as they decide. 1: no refracting mesh.
    float crossingRoughness = 1.0F;
#if GLTF_USE_TRANSMISSION
    if(materialIndex >= 0 && pbrMat.transmission > MIN_TRANSMISSION)
    {
      const float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
      crossingRoughness  = crossingRoughnessMetallic(materials[materialIndex], renderPrim, getTriangleIndices(renderPrim, payload.primitiveID), barys).x;
    }
#endif

    // Regularisation: later bounces are made at least as rough as the path, against fireflies.
    // Approximate caustics (Unreal's): as rough as the path's scattering, a diffuse bounce making
    // every later lobe fully rough. Otherwise (upstream): as rough as the roughest material so far,
    // whichever of its lobes the path took, where ePtRegularize asks for it (ptRegularize: DLSS frames
    // by default; the still has the indirect clamp and the denoiser) and never on a surface that
    // refracts without scattering: after a matte bounce the cornea went rough, and the iris behind it.
    if(hasFlag(pushConst.flags, PathtracerFlags::ePtApproximateCaustics))
    {
      pbrMat.roughness          = max(pbrMat.roughness, float2(pt.pathRoughness * pt.pathRoughness));
      pbrMat.clearcoatRoughness = max(pbrMat.clearcoatRoughness, pt.pathRoughness);
      pbrMat.specular2Roughness = max(pbrMat.specular2Roughness, pt.pathRoughness);
    }
    else
    {
      pt.maxRoughness            = max(pbrMat.roughness, pt.maxRoughness);
      const bool clearRefraction = pbrMat.transmission > 0.0F
                                   && max(pbrMat.scatterCoefficient.x, max(pbrMat.scatterCoefficient.y, pbrMat.scatterCoefficient.z)) <= VOLUME_MIN_SCATTER;
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtRegularize) && !clearRefraction)
        pbrMat.roughness = pt.maxRoughness;
    }

    //-----------------------------------------------------------------------
    // Visualization override (Debug mode)
    // Note: clay visualization mode continues shading; flat-color modes break
    //-----------------------------------------------------------------------
    // Debug-only; compiled out of the normal render path (USE_VISUALIZE, set by the host when the
    // visualization mode is not eRendered).
#if USE_VISUALIZE
    if(frameInfo->visualization != Visualization::eRendered)
    {
      float3     vizColor;
      const bool onMesh = !hitInfinitePlane && !hitWater;
      // the hit triangle's size, for the Density view alone: the side of the square of twice its area
      float triangleSize = 0.0;
      if(onMesh && frameInfo->visualization == Visualization::eTriangleSize)
      {
        const uint3  idx = getTriangleIndices(renderPrim, payload.primitiveID);
        const float3 p0  = mul(float4(getVertexPosition(renderPrim, idx.x), 1.0), renderNode.objectToWorld).xyz;
        const float3 p1  = mul(float4(getVertexPosition(renderPrim, idx.y), 1.0), renderNode.objectToWorld).xyz;
        const float3 p2  = mul(float4(getVertexPosition(renderPrim, idx.z), 1.0), renderNode.objectToWorld).xyz;
        triangleSize     = sqrt(length(cross(p1 - p0, p2 - p0)));
      }
      VisualizationResult vizResult = applyVisualization(pbrMat, hit, frameInfo->visualization, payload.rprimID,
                                                         payload.primitiveID, payload.ommUnknown ? 1 : 0,
                                                         onMesh ? materialIndex : -1, onMesh ? renderNode.ownerID : -1,
                                                         triangleSize, -ray.Direction, vizColor);
      if(vizResult == VisualizationResult::eColorOverride && firstRay)
      {
        pt.radiance.xyz = vizColor;
        return PathStepResult.eBreak;
      }
    }
#endif

    // glTF's occlusion map stands in for the indirect light a rasterizer cannot trace. A path tracer
    // traces that occlusion, and the BSDF only applied the map to light samples, never to the BSDF
    // rays: sunlit cavities were painted dark while the light MIS gave the BSDF side kept its own.
    pbrMat.occlusion = 1.0F;

    //-----------------------------------------------------------------------
    // Adding emissive
    //-----------------------------------------------------------------------
    // The first surface outside a volume walk is shaded on its own. An emitter reached directly
    // from the walk contributes nothing: volume NEE already estimated that light.
#if GLTF_USE_VOLUME
    const bool walkOwnsDirect = pt.exitedVolumeWalk && !ptIsInside(pt);
    if(!ptIsInside(pt))
      pt.exitedVolumeWalk = false;
#else
    const bool walkOwnsDirect = false;
#endif
    // Emitters are also sampled as area lights, textured ones by their texture's mean; a BSDF ray that
    // hits one is MIS-weighted against the light technique, mirroring the env-hit MIS above. A one-sided
    // emitter met from behind is only a surface, as the light samples have it (sampleEmissiveTriangle).
    const bool emitsThisWay = hitInfinitePlane || hitWater || hit.frontFace || materials[materialIndex].doubleSided != 0;
    if(!walkOwnsDirect && emitsThisWay && any(pbrMat.emissive > float3(0.0)))
    {
      float emissiveMis = 1.0;
      if(!hitInfinitePlane && !hitWater)
      {
        // The light sample's pdf of this point: the hit triangle's weight in the light table (not the
        // material row's), its area as it is here, its geometric normal, and what it was picked among.
        emissiveMis = computeEmitterHitMisWeight(pt.lastSamplePdf, pt.lightPick, emitterWeight(payload.rnodeID, payload.primitiveID),
                                                 hit.geonrm, ray.Direction, pt.nullDistance + payload.hitT, payload.rnodeID,
                                                 renderPrim, payload.primitiveID, renderNode.objectToWorld);
      }
      const float3 throughCoat = bsdfCoatEmission(pbrMat.clearcoat, pbrMat.clearcoatRoughness, pbrMat.baseColor, pbrMat.metallic,
                                                  bsdfCoatClearTransmission(pbrMat), 0.5F * (pbrMat.roughness.x + pbrMat.roughness.y), abs(dot(pbrMat.N, ray.Direction)));
      addRadiance(pt, emissiveMis * pbrMat.emissive * throughCoat * pt.throughput, pt.prefixIndirect);
    }

    //-----------------------------------------------------------------------
    // Unlit
    //-----------------------------------------------------------------------
#if GLTF_USE_UNLIT
    if(materialIndex >= 0 && materials[materialIndex].unlit > 0)
    {
      // Through the path's throughput, like any emission: an unlit backdrop seen in a 4% reflection
      // is 4% of itself
      addRadiance(pt, pt.throughput * pbrMat.baseColor, pt.prefixIndirect);
      return PathStepResult.eBreak;
    }
#endif

#if USE_RADIANCE_CACHE
    // The bounce cache (radiance_cache.h.slang). The path has scattered roughly and this vertex is a
    // surface that lets no light through: what it sends back, its own emission aside (added above, with
    // its MIS weight), is in the cell here once enough paths have fed it, and the path ends with that in
    // place of its light sample and every bounce to come. An update path instead marks its first such
    // vertex and goes on, to add what it gathers from here to the cell (pathTrace); at its later ones it
    // reads the cache as the others do, so a cell's light comes to hold every bounce after it.
#if GLTF_USE_HAIR
    if(materialIndex >= 0 && materials[materialIndex].hairFactor > 0.0F)
      cacheSurface = false;
#endif
    if(cacheSurface && pbrMat.transmission <= MIN_TRANSMISSION)
    {
      const float3 eye  = frameInfo->viewInv[3].xyz;
      const float  cell = frameInfo->radianceCacheCell;
      const float  size = radianceCacheCellSize(radianceCacheLevel(length(hit.pos - eye), cell), cell);
      if(radianceCacheFarEnough(payload.hitT, size, rand(seed)))
      {
        uint*      table = frameInfo->radianceCache;
        const uint slots = uint(frameInfo->radianceCacheSlots);
        if(g_cacheUpdate && pt.cacheSlot == -1)
        {
          const int slot     = radianceCacheClaim(table, radianceCacheKey(hit.pos, hit.geonrm, eye, cell, slots));
          pt.cacheSlot       = slot >= 0 ? slot : -2;
          pt.cacheThroughput = pt.throughput;
          pt.cacheRadiance   = pt.radiance;
        }
        else
        {
          // the cell of a point up to half a cell off, so the cells' edges do not show; where that one
          // is empty (past a wall's end), this point's own
          const float3 near = radianceCacheJitter(hit.pos, radianceCacheFace(hit.geonrm), size, float2(rand(seed), rand(seed)));
          int          slot = radianceCacheFind(table, radianceCacheKey(near, hit.geonrm, eye, cell, slots));
          RadianceCacheCell held;
          held.light  = float3(0.0F);
          held.weight = float3(0.0F);
          if(slot >= 0)
            held = radianceCacheRead(table, slot);
          if(!radianceCacheValid(held))
          {
            slot = radianceCacheFind(table, radianceCacheKey(hit.pos, hit.geonrm, eye, cell, slots));
            if(slot >= 0)
              held = radianceCacheRead(table, slot);
          }
          if(radianceCacheValid(held))
          {
            addRadiance(pt, pt.throughput * radianceCacheRadiance(held), true);
            return PathStepResult.eBreak;
          }
        }
      }
    }
#endif

    // The thin preview's spread (skin_spread.h.slang). The preview takes skin's light where it lands,
    // and the still's walk lets it out millimetres away, so a shadow on skin -- a pendant's, the jaw's
    // on the neck -- had the hard edge and the dark core the still's lacks. On the camera's skin the
    // diffuse share of the light sample is taken, each channel, at a point of the skin as far off as
    // that channel's profile puts it (its rate here, from the medium the map makes, previewSkinAlpha),
    // with a shadow ray of its own; its glossy share keeps this point's. Below, at the shadow rays.
    bool   spreadHere = false;
    float3 spreadRate = float3(0.0F);
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
    if(previewSkin && firstRay && frameInfo->skinSpread != 0 && !hitInfinitePlane && !hitWater && materialIndex >= 0)
    {
      spreadRate = skinSpreadRate(previewSkinAlpha, materials[materialIndex].scatterDistance);
      spreadHere = true;
    }
#endif

    // The thin preview's slab (skin_slab.h.slang). The preview sends all of skin's light back out where
    // it went in, which a thick body does, but where measured skin is thin enough for light to reach
    // its far side -- an ear, a finger, the wing of a nostril -- the still's walk lets some through,
    // red the most, and a back-lit ear glows. A probe straight in along the face finds that far side
    // (probeSlabFarSide, within a cone about the smooth normal: SKIN_SLAB_CONE_COS), Kubelka-Munk splits
    // the map's light into what comes back here and what goes through, and one sample takes one of the
    // two: this point with what comes back, or the far side's point as a Lambertian surface with what
    // goes through (its light sample and its bounce there), each weighed by one over its chance. What
    // goes through came past this side's specular layer, as the diffuse share does, and leaves with the
    // walk's exit Fresnel. Thick skin, where no far side is within reach, is shaded as before (the
    // probe's direction takes two numbers from the seed). After the guides: the camera sees this side.
    bool slabFarSide = false;
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
    if(simplifyScatter && frameInfo->skinSlab != 0 && !hitInfinitePlane && !hitWater && materialIndex >= 0 && hit.frontFace && !ptIsInside(pt)
       && pbrMat.thickness > 0.0F && materials[materialIndex].scatterStrengthFactor > 0.0F && any(materials[materialIndex].scatterDistance > 0.0F))
    {
      const float3 diffusion = max(materials[materialIndex].scatterDistance, float3(1e-7F));
      // in within the cone about the smooth normal (SKIN_SLAB_CONE_COS), never out past the face
      const float3 axis = -hit.nrm;
      const float  cosT = 1.0F - rand(seed) * (1.0F - SKIN_SLAB_CONE_COS);
      const float  phi  = M_TWO_PI * rand(seed);
      float3       tangent, bitangent;
      orthonormalBasis(axis, tangent, bitangent);
      float3 inward = normalize(axis * cosT + (tangent * cos(phi) + bitangent * sin(phi)) * sqrt(max(1.0F - cosT * cosT, 0.0F)));
      if(dot(inward, hit.geonrm) >= -0.05F)
        inward = -hit.geonrm;
      const float       along = max(dot(inward, axis), SKIN_SLAB_CONE_COS);
      const float       reach = SKIN_SLAB_REACH * max(diffusion.x, max(diffusion.y, diffusion.z)) / along;
      const SlabFarSide far   = probeSlabFarSide(safeOffsetRay(hit.pos, -hit.geonrm, hit.offsetDistance), inward, reach, renderNode.materialID);
      if(far.found)
      {
        const SkinSlab slab    = skinSlab(pbrMat.baseColor, far.depth * along / diffusion);
        const float    through = luminance(slab.transmittance);
        const float    chance  = min(through / max(through + luminance(slab.reflectance), 1e-6F), SKIN_SLAB_MAX_CHANCE);
        if(chance > SKIN_SLAB_MIN_CHANCE)
        {
          if(rand(seed) < chance)
          {
            // this side's diffuse share at the view, and what a coat over it leaves of the colour
            float3 tint = pbrMat.baseColor;
            ARRAY_TYPE(float, LOBE_COUNT, lobeWeights) = computeLobeWeights(pbrMat, dot(-ray.Direction, pbrMat.N), tint);
            const float3 underCoat = tint / max(pbrMat.baseColor, float3(1e-4F));
            hit                    = far.hit;
            pbrMat                 = defaultPbrMaterial(slab.transmittance * underCoat, 0.0F, 1.0F, hit.nrm, hit.geonrm);
            pbrMat.specular        = 0.0F;
            pbrMat.ior2            = 1.0F;
            pbrMat.T               = hit.tangent;
            pbrMat.B               = hit.bitangent;
            pbrMat.Ns              = hit.nrm;
            if(hasFlag(pushConst.flags, PathtracerFlags::ePtExitFresnel))
              exitFresnelEta = clamp(materials[materialIndex].ior, 1.0F, 3.0F);
            pt.throughput *= lobeWeights[LOBE_DIFFUSE_REFLECTION] / chance;
            slabFarSide = true;
            spreadHere  = false;  // shaded at the far side's point already
          }
          else
          {
            pbrMat.baseColor = slab.reflectance;
            pt.throughput /= 1.0F - chance;
          }
        }
      }
    }
#endif

    // Normal surface bounce: propagate the cone footprint to the next segment. The new ray
    // starts at hit.pos with the slant-corrected worldFoot as its starting width, so the
    // next Trace()'s texture LOD inherits the full accumulated footprint.
    pt.cone.width = worldFoot;

    //-----------------------------------------------------------------------
    // Sampling the lights
    // - Get the light contribution from one light source or the HDR environment
    //-----------------------------------------------------------------------
    // A ribbon fibre is lit from both sides: the transmitted lobe leaves through the sheet.
    bool hairOn = false;
#if GLTF_USE_HAIR
    HairHit hairHit = hairHitZero();
    if(materialIndex >= 0 && materials[materialIndex].hairFactor > 0.0)
      hairOn = makeHairHit(materials[materialIndex], hit.uv[0], hit.bitangent, hit.tangent, hit.frameSign, pbrMat.N, pbrMat.baseColor, hairHit);
    if(hairOn && hasFlag(pushConst.flags, PathtracerFlags::ePtApproximateCaustics))
      hairHit.betaM = max(hairHit.betaM, pt.pathRoughness);  // the fibre's longitudinal roughness, as the lobes above
    // One offset across the fibre for this hit: the samples average over it to the fibre's response,
    // at a quarter of the cost of evaluating four fixed offsets in both the light and the BSDF sample
    if(hairOn)
      hairHit.h = 2.0 * rand(seed) - 1.0;
#if USE_DLSS_SHADER
    // RR's guides describe the fibre, not the sheet's GGX: the base colour stays the albedo, the
    // specular is the R lobe's (Fresnel at the fibre's 1.55, about 4.6%), the roughness its longitudinal one
    if(firstRay && hairOn)
    {
      pt.guide.specularAlbedo  = float16_t3(float3(0.046F * saturate(hairHit.specular)));
      pt.guide.normalRoughness = float16_t4(pt.guide.normalRoughness.xyz, float16_t(saturate(hairHit.betaM)));
      // RR weighs its history on hair by this (ptHairResponsivity), against the shimmer of thin fibres
      pt.guide.responsivity = float16_t(pushConst.hairResponsivity);
    }
#endif
#endif
    // Rough thin glass (frosted, the envelope of a bulb) blocks shadow rays, so the lights behind it are
    // sampled from here, through its transmission lobe, and its BSDF ray that reaches one is weighed
    // against that (samplesLightsThroughGlass)
    bool roughThinGlass = false;
#if GLTF_USE_TRANSMISSION
    roughThinGlass = !hairOn && materialIndex >= 0
                     && samplesLightsThroughGlass(pbrMat.transmission > MIN_TRANSMISSION, pbrMat.thickness, crossingRoughness,
                                                  materials[materialIndex].scatterStrengthFactor);
#endif
    const bool bothSides = pbrMat.diffuseTransmissionFactor > 0.0f || hairOn || roughThinGlass;

    DirectLight directLight;
    // A fibre event spends the hair's own budget while it lasts, not a surface bounce
#if GLTF_USE_HAIR
    bounce.hairEvent = hairOn && pt.hairBounces < pushConst.hairBudget;
    bounce.fibre     = hairOn;
#endif
    // So does an event on clear glass, the glass's (glassBudget): the surface shadow rays pass straight
    // through, so what is behind a pane is lit, and the path goes on to see it
#if GLTF_USE_TRANSMISSION
    bounce.glassEvent = !hairOn && materialIndex >= 0 && pt.glassBounces < pushConst.glassBudget && pbrMat.transmission > MIN_TRANSMISSION
                        && isClearCrossing(crossingRoughness) && materials[materialIndex].scatterStrengthFactor <= 0.0F;
#endif
    // The water's surface likewise, on a count of its own the glass's size: its rays go into the water
    // to the floor and what stands in it, and back out
    bounce.waterEvent = hitWater && pt.waterBounces < pushConst.glassBudget;
    // The last vertex's BSDF continuation is never traced (the loop ends first), so its light samples
    // have no BSDF-sampled partner: weight 1, as for a punctual light, and the full environment table.
    bool hairCut = false;
#if GLTF_USE_HAIR
    hairCut = hairCutHere(bounce.fibre, pt.hairBounces);
#endif
    const bool lastVertex = hairCut || (!bounce.hairEvent && !bounce.glassEvent && !bounce.waterEvent && pt.surfaceDepth + 1 >= pushConst.maxDepth);
    // The lobe mixture for this vertex's light sample and BSDF sample alike, so their MIS agrees. With
    // ePtLobeMixtureDirect only where the path is still direct: there a catchlight is seen, through a
    // cornea one sample in forty with a single lobe; past a rough bounce one lobe costs a third less.
    const bool lobeMixture = hasFlag(pushConst.flags, PathtracerFlags::ePtLobeMixture)
                             && (!hasFlag(pushConst.flags, PathtracerFlags::ePtLobeMixtureDirect) || pt.pathRoughness <= DIRECT_PATH_ROUGHNESS);
    // A vertex whose every weighted lobe is a mirror (smoothVertex: clear glass, a tear film, polished
    // metal), judged after the regularisation above, leaves the lights its BSDF ray reaches to that ray:
    // no BSDF evaluation and no shadow ray for a sample whose value and MIS weight are next to nothing,
    // and the ray's hits weigh 1 (LightPick::bsdfOnly), so the image is the same in expectation. The
    // pick is drawn as before: a punctual light keeps its sample, and so does one where a lobe may
    // scatter that the BSDF ray never leaves by (smoothVertexSamplesLight), weighed 1 only there
    // (smoothVertexLightWeight). Not at the last vertex, whose BSDF ray is never traced, nor on a fibre
    // (ePtSmoothBsdfOnly; off, it samples them as any vertex).
    const bool smoothHere = hasFlag(pushConst.flags, PathtracerFlags::ePtSmoothBsdfOnly) && !hairOn && !lastVertex && smoothVertex(pbrMat);
    // The normal only culls directional lights below the surface (singleLightContribution). Give it
    // the one bounce.nextEventValid tests, so NEE and the BSDF side (R4) agree on which directions
    // next-event estimation can reach; the BSDF then decides what a light below the shading normal does.
    // The vertex before this one, for a ray that crosses this surface clear (below)
    const float     prevSamplePdf = pt.lastSamplePdf;
    const LightPick prevLightPick = pt.lightPick;
    sampleLights(hit.pos, hit.nrm, ray.Direction, seed, directLight, /*isVolumeSample=*/false,
                 /*lightsFromBothSides=*/bothSides, /*compensatedEnv=*/!lastVertex, g_sobol ? nextSobolDecision() : float4(-1.0F));
    // The BSDF ray sampled below is weighed against this vertex's pick when it reaches a light
    pt.lightPick          = directLight.pick;
    pt.lightPick.bsdfOnly = smoothHere;
    if(smoothHere && !smoothVertexSamplesLight(pbrMat, directLight.direction, directLight.pdf))
      directLight.pdf = 0.0F;  // the BSDF ray's: nothing to evaluate or trace (bounce.nextEventValid)

    // Several candidates, one kept (light_ris.h.slang): the first is the sample above, the rest drawn
    // the same way from the seed. Each is weighed by its unshadowed estimate at this vertex, its BSDF
    // evaluated here and kept with it, so the kept one's estimate below is the one it was weighed by.
    // Every candidate shares the pick (the same point and normal), so pt.lightPick holds for all.
    float            risScale   = 1.0F;
#if USE_LIGHT_RIS
    // The camera's first surface keeps its kept sample from frame to frame (ePtLightReuse): last
    // frame's record at this pixel (reprojected under DLSS) is one more candidate, re-weighed here, and
    // what this vertex shades is written for the next frame, its W zeroed below if it was shadowed.
    // A smooth vertex resamples nothing: its one sample stands (smoothHere)
    // Not from a slab's far side: the record is this pixel's first surface, and that one is the near side
    const bool reuseHere = firstRay && !hairOn && !smoothHere && !slabFarSide && hasFlag(pushConst.flags, PathtracerFlags::ePtLightReuse)
                           && pushConst.lightReservoirs != nullptr;
    const int        candidates = reuseHere ? clamp(pushConst.lightCandidates, 1, LIGHT_RIS_MAX_CANDIDATES) :
                                              (smoothHere ? 1 : lightCandidatesHere(hairOn, pt.surfaceDepth));
    BsdfEvaluateData keptEval   = {};
    if(candidates > 1 || reuseHere)
    {
      // Last frame's record at this pixel, or what this frame's earlier sample of it wrote
      const bool       secondHalf = hasFlag(pushConst.flags, PathtracerFlags::ePtLightReuseParity);
      LightReuseRecord record = lightReuseUnpack(uint4(0u), uint4(0u));  // none
      float recordM = 0.0F;
      if(reuseHere)
      {
        int2  readPixel   = g_reusePixel;
        float recordDepth = mul(float4(hit.pos, 1.0F), frameInfo->viewProjMatrix).w;  // this point's depth, as the record has its own
#if USE_DLSS_SHADER
        if(g_reuseFirstSample && hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
        {
          // where the surface was on screen last frame, as its motion vector has it, and its depth then
          const float4 prevClip = mul(float4(pt.firstHitPosPrev, 1.0F), frameInfo->prevMVP);
          recordDepth           = prevClip.w;
          readPixel             = (prevClip.w > 0.0F) ? int2(floor((prevClip.xy / prevClip.w * 0.5F + 0.5F) * float2(pushConst.renderSize))) : int2(-1);
        }
#endif
        const bool inside = all(readPixel >= int2(0)) && all(readPixel < int2(pushConst.renderSize));
        if(inside && (!g_reuseFirstSample || hasFlag(pushConst.flags, PathtracerFlags::ePtLightReuseHistory)))
        {
          const uint slot = lightReuseSlot(readPixel, g_reuseFirstSample ? !secondHalf : secondHalf);
          record          = lightReuseUnpack(pushConst.lightReservoirs[slot], pushConst.lightReservoirs[slot + 1]);
          // a record of another surface (an edge the jitter crossed, a disocclusion) is none: that depends
          // on the geometry, not on what was drawn, so dropping it keeps the M weights whole
          if(!lightReuseSameSurface(record, recordDepth, hit.nrm))
            record.M = 0.0F;
        }
        recordM = lightReuseRecordM(record, float(candidates), pushConst.lightReuseCap);
      }

      LightReservoir reservoir = lightReservoirEmpty();
      DirectLight    kept      = directLight;
      for(int c = 0; c < candidates; c++)
      {
        DirectLight candidate = directLight;
        if(c > 0)
          sampleLights(hit.pos, hit.nrm, ray.Direction, seed, candidate, /*isVolumeSample=*/false,
                       /*lightsFromBothSides=*/bothSides, /*compensatedEnv=*/!lastVertex);
        BsdfEvaluateData candidateEval = {};
        float weight      = 0.0F;
        if((dot(candidate.direction, hit.nrm) > 0.0f || bothSides) && candidate.pdf != 0.0f)
        {
          const bool candidateAllLobes = lobeMixture || (candidate.pdf == DIRAC && hasFlag(pushConst.flags, PathtracerFlags::ePtLobeSumPunctual));
          candidateEval = evaluateSurfaceForLight(pbrMat, -ray.Direction, candidate.direction, candidateAllLobes, exitFresnelEta, seed);
          if(bsdfEvaluated(candidateEval))
          {
            const float mis = (candidate.pdf == DIRAC || lastVertex) ? 1.0F : misLightWeight(candidate.pdf, candidateEval.pdf, pushConst.frameInfo->estimator);
            weight = luminance(pt.throughput * candidate.radianceOverPdf * mis * (candidateEval.bsdf_diffuse + candidateEval.bsdf_glossy));
          }
        }
        reservoir = lightReservoirAdd(reservoir, weight, rand(seed));
        if(reservoir.keptLast)
        {
          kept      = candidate;
          keptEval  = candidateEval;
        }
      }
      // None weighed anything: the scale is 0, and the sample below adds nothing and traces no ray
      directLight = kept;
      risScale    = lightReservoirScale(reservoir);
      if(reuseHere)
      {
        // Two sides: this frame's candidates, resampled above into the kept one, and the record. Each is
        // weighed by its MIS weight (lightReuseMis: M times its target here and at the record's point,
        // the second taken here with that point's shading normal and view direction) times its target
        // here times its W.
        const PbrMaterial thereMat = (recordM > 0.0F) ? lightReuseMaterialThere(pbrMat, record.shading) : pbrMat;

        // this frame's side: the kept candidate's target here is its weight; there, only for a light a
        // record can hold (a punctual one is this side's alone)
        float keptThere = 0.0F;
        if(recordM > 0.0F && reservoir.kept > 0.0F && directLight.reuseKind != LIGHT_REUSE_NONE && directLight.pdf != DIRAC)
        {
          const BsdfEvaluateData e = evaluateSurfaceForLight(thereMat, record.view, directLight.direction, lobeMixture, exitFresnelEta, seed);
          if(bsdfEvaluated(e))
          {
            const float mis = lastVertex ? 1.0F : misLightWeight(directLight.pdf, e.pdf, pushConst.frameInfo->estimator);
            keptThere       = luminance(pt.throughput * directLight.radianceOverPdf * mis * (e.bsdf_diffuse + e.bsdf_glossy));
          }
        }
        const float meanWeight    = (reservoir.count > 0) ? reservoir.weightSum / float(reservoir.count) : 0.0F;
        const float currentWeight = lightReuseWeight(lightReuseMis(true, float(candidates), reservoir.kept, recordM, keptThere), meanWeight);

        // the record's side: its sample re-weighed here and at its own point, in the record's measure
        float            recordTarget = 0.0F;
        float            recordWeight = 0.0F;
        ReusedLight      reused       = {};
        BsdfEvaluateData reusedEval   = {};
        if(recordM > 0.0F && record.W > 0.0F)
        {
          reused = evaluateReusedLight(record, hit.pos, hit.nrm, directLight.pick, bothSides, /*compensatedEnv=*/!lastVertex);
          if(any(reused.radiance > 0.0F))
          {
            reusedEval = evaluateSurfaceForLight(pbrMat, -ray.Direction, reused.direction, lobeMixture, exitFresnelEta, seed);
            if(bsdfEvaluated(reusedEval))
            {
              const float mis = lastVertex ? 1.0F : misLightWeight(reused.pdf, reusedEval.pdf, pushConst.frameInfo->estimator);
              recordTarget    = luminance(pt.throughput * reused.radiance * mis * (reusedEval.bsdf_diffuse + reusedEval.bsdf_glossy));
            }
            float                  recordThere = 0.0F;
            const BsdfEvaluateData e           = evaluateSurfaceForLight(thereMat, record.view, reused.direction, lobeMixture, exitFresnelEta, seed);
            if(bsdfEvaluated(e))
            {
              const float mis = lastVertex ? 1.0F : misLightWeight(reused.pdf, e.pdf, pushConst.frameInfo->estimator);
              recordThere     = luminance(pt.throughput * reused.radiance * mis * (e.bsdf_diffuse + e.bsdf_glossy));
            }
            recordWeight = lightReuseWeight(lightReuseMis(false, float(candidates), recordTarget, recordM, recordThere), recordTarget * record.W);
          }
        }

        // one of the two, in proportion to its weight. What multiplies the kept sample's estimate: the
        // weights' sum over its own target here. A current candidate's record W is that per solid angle
        // and per the record's measure.
        LightReservoir pair = lightReservoirAdd(lightReservoirEmpty(), currentWeight, rand(seed));
        pair                = lightReservoirAdd(pair, recordWeight, rand(seed));
        float measureOverScale = directLight.pdf * directLight.toMeasure;
        if(pair.keptLast)
        {
          directLight.direction       = reused.direction;
          directLight.distance        = reused.distance;
          directLight.radianceOverPdf = reused.radiance;  // in the record's measure, W taking the pdf's place
          directLight.pdf             = (lastVertex && !(reused.pdf > 0.0F)) ? 1e-30F : reused.pdf;  // weight 1 there
          directLight.reuseKind       = record.kind;
          directLight.reuseIndex      = record.index;
          directLight.reuseUv         = record.uv;
          keptEval                    = reusedEval;
          risScale                    = pair.weightSum / recordTarget;
          measureOverScale            = 1.0F;
        }
        else
          risScale = (reservoir.kept > 0.0F) ? pair.weightSum / reservoir.kept : 0.0F;

        // The record for the next frame (and this frame's next sample of the pixel): M whatever was
        // drawn; W 0 for what a record cannot hold or what weighed nothing, and if the shadow ray below
        // finds the sample shadowed
        LightReuseRecord next;
        next.depth   = mul(float4(hit.pos, 1.0F), frameInfo->viewProjMatrix).w;
        next.normal  = hit.nrm;
        next.shading = pbrMat.N;
        next.view    = -ray.Direction;
        next.kind    = directLight.reuseKind;
        next.index   = directLight.reuseIndex;
        next.uv      = directLight.reuseUv;
        next.W       = (next.kind != LIGHT_REUSE_NONE && directLight.pdf != DIRAC && measureOverScale > 0.0F) ? risScale / measureOverScale : 0.0F;
        next.M       = float(candidates) + recordM;
        const uint slot                     = lightReuseSlot(g_reusePixel, secondHalf);
        pushConst.lightReservoirs[slot]     = lightReusePackA(next);
        pushConst.lightReservoirs[slot + 1] = lightReusePackB(next);
        bounce.lightReuseWritten            = next.W > 0.0F;
      }
    }
#endif

    // This checks if the light could contribute, but doesn't send a next event estimation (NEE) shadow ray yet.
    // The actual test and contribution are delayed to minimize per-bounce state.
    bounce.nextEventValid = (dot(directLight.direction, hit.nrm) > 0.0f || bothSides)
                            && directLight.pdf != 0.0f && risScale > 0.0F;

    //-----------------------------------------------------------------------
    // Evaluating the BSDF for the light
    //-----------------------------------------------------------------------
    if(bounce.nextEventValid)
    {
      // Evaluating the BSDF at the hit point for the light
      BsdfEvaluateData evalData;
      // Every lobe for a punctual light (no BSDF ray reaches it; its weight is 1 whatever the pdf), and
      // for every light when MIS weighs against the lobe mixture on both sides (bsdfSampleCore below).
      const bool allLobes = lobeMixture || (directLight.pdf == DIRAC && hasFlag(pushConst.flags, PathtracerFlags::ePtLobeSumPunctual));
#if USE_LIGHT_RIS
      if(candidates > 1)
        evalData = keptEval;  // evaluated as it was weighed, never a fibre (lightCandidatesHere)
      else
#endif
      {
        evalData.k1 = -ray.Direction;
        evalData.k2 = directLight.direction;
        evalData.xi = float3(rand(seed), rand(seed), rand(seed));
#if GLTF_USE_HAIR
        if(hairOn)
        {
          hairBsdfEvaluate(evalData, hairHit);  // already the whole fibre BSDF, with its mixture pdf
          // A cut path's light sample also stands for the light the lock would have scattered on to it:
          // its colour, spread over the sphere (hairCutTail)
          if(hairCut)
            evalData.bsdf_diffuse += pushConst.hairCutTail * (0.25F * M_1_PI) * hairLockColor(hairHit);
        }
        else
#endif
        if(allLobes)
          bsdfEvaluateAll(evalData, pbrMat);
        else
          bsdfEvaluate(evalData, pbrMat);
        // bsdfEvaluateSimple(evalData, pbrMat);

        if(exitFresnelEta > 0.0F)
          evalData.bsdf_diffuse *= diffuseExitFresnelWeight(dot(pbrMat.N, directLight.direction), exitFresnelEta);
      }

      // Some lobe can scatter toward the light (in every mode): the shadow ray is worth tracing. Its
      // pdf is 0 where the light lies past the face from where the lobe would send a bounce
      // (bsdfFaceAllows): no BSDF sample reaches it, and the light sample's weight is 1.
      if(bsdfEvaluated(evalData))
      {
        // Weight for combining light and BSDF sampling strategies (Multiple Importance Sampling);
        // 1 at the last vertex (above). A smooth vertex's kept sample: 1 only where the BSDF ray cannot go.
        float mis_weight = (directLight.pdf == DIRAC || lastVertex) ? 1.0F : misLightWeight(directLight.pdf, evalData.pdf, pushConst.frameInfo->estimator);
        if(smoothHere)
          mis_weight = smoothVertexLightWeight(directLight.pdf, evalData.pdf);

        // sample weight; a kept candidate's scaled by the candidates' mean weight over its own
        const float3 w = pt.throughput * directLight.radianceOverPdf * mis_weight * risScale;
        // the spread's diffuse share has shadow rays of its own (below)
        if(spreadHere)
          bounce.spreadContribution = w * evalData.bsdf_diffuse;
        else
          bounce.contribution += w * evalData.bsdf_diffuse;
        bounce.contribution += w * evalData.bsdf_glossy;
        // Classed by the path before this vertex, as the sample below is about to change it
        bounce.indirect = pt.pathRoughness > DIRECT_PATH_ROUGHNESS;
      }
      // The shadow ray could only multiply zero: the BSDF cannot scatter toward the light, or the
      // light sends nothing this way (a black texel of an emissive texture, a sized spot's sample
      // outside its cone, a light picked evenly among more than LIGHT_PICK_MAX_LIGHTS). A non-finite
      // contribution still traces, so its sample is dropped as before (the end of samplePixel).
      if(all(bounce.contribution == 0.0F) && all(bounce.spreadContribution == 0.0F))
        bounce.nextEventValid = false;

      // Traced only as often as it can matter (contribution_roulette.h.slang): a sample that would add
      // less than the share of what the pixel holds, were the light to reach it, is traced with the
      // chance of its size against that and weighed by one over it. Unbiased: the mean it goes by is
      // that of the accumulation's earlier frames, which this sample has no part in. A sample that is
      // not finite is traced, and dropped as before. Not a vertex that wrote the pixel's record for the
      // next frame, whose shadow ray says whether the record stands.
      bool rouletteHere = bounce.nextEventValid && g_shadowRest > 0.0F;
#if USE_LIGHT_RIS
      rouletteHere = rouletteHere && !bounce.lightReuseWritten;
#endif
      if(rouletteHere)
      {
        const float term = luminance(bounce.contribution + bounce.spreadContribution);
        if(term > 0.0F && term < pushConst.frameInfo->shadowRayShare * g_shadowRest)
        {
          const float chance = contributionTraceChance(term, g_shadowRest, pushConst.frameInfo->shadowRayShare);
          if(rand(seed) < chance)
          {
            bounce.contribution       /= chance;
            bounce.spreadContribution /= chance;
          }
          else
            bounce.nextEventValid = false;
        }
      }

#if GLTF_USE_VOLUME
      // The medium the shadow ray starts in, read before the BSDF sample below moves the path into
      // another one: the medium the ray arrived through, unless the light is behind a surface that
      // bounds a volume (diffuse transmission), which the shadow ray then starts on the far side of.
      bounce.shadowRayInside     = pt.isInside;
      bounce.shadowRayExtinction = volumeExtinction(pt.volumeMedium);
      // (not a scatter material shaded thin here: it has no far side to start on)
      bool thinScatter = false;
#if GLTF_USE_SCATTER
      thinScatter = simplifyScatter && materialIndex >= 0 && materials[materialIndex].scatterStrengthFactor > 0.0F;
#endif
      if(dot(directLight.direction, hit.nrm) <= 0.0F && pbrMat.thickness > 0.0F && !thinScatter && materialIndex >= 0)
      {
        // beyond the surface: in its own medium, or out of it in the one around it
        bool backEntry = false;
#if GLTF_USE_SCATTER
        backEntry = !pt.isInside && materialIndex >= 0 && materials[materialIndex].scatterStrengthFactor > 0.0F;
#endif
        const MediumState left = mediumLeft(ptMedium(pt));
        bounce.shadowRayInside     = hit.frontFace || backEntry || left.inside;
        bounce.shadowRayExtinction = float3(0.0F);
        if(hit.frontFace || backEntry)
          bounce.shadowRayExtinction = volumeExtinction(makeVolumeMedium(pbrMat));
        else if(left.inside)
          bounce.shadowRayExtinction = volumeExtinction(absorbingMedium(materials + left.current));
      }
#endif
    }

    //-----------------------------------------------------------------------
    // Sampling the BSDF :
    // - Finding the next direction and updating the throughput
    // Not at the last vertex: the loop ends before its ray is traced, and what follows here (the
    // shadow ray's set-up) reads the hit and the light only. The camera ray keeps it for RR's
    // specular hit guide.
    //-----------------------------------------------------------------------
    if(!lastVertex || firstRay)
    {
      // Sampling the BSDF at the hit point
      BsdfSampleData sampleData;
      sampleData.k1 = -ray.Direction;                              // outgoing direction
      // random numbers: a Sobol point's, the fourth for the fibre's azimuth
      const float4 bsdfXi = g_sobol ? nextSobolDecision() : float4(rand(seed), rand(seed), rand(seed), 0.0F);
      sampleData.xi       = bsdfXi.xyz;
      int sampledLobe     = LOBE_DIFFUSE_REFLECTION;  // the lobe the sample came from; a fibre's are its own
#if GLTF_USE_HAIR
      if(hairOn)
      {
        hairBsdfSample(sampleData, hairHit, g_sobol ? bsdfXi.w : rand(seed));
      }
      else
#endif
      {
        // The lobe mixture's pdf, which next-event estimation's bsdfEvaluateAll weighs against. Not at
        // the last vertex: the loop ends before this ray is traced, so nothing reads it.
        const bool mixturePdf = lobeMixture && !lastVertex;
        sampledLobe           = bsdfSampleCore(sampleData, pbrMat, mixturePdf);
      }
      // bsdfSampleSimple(sampleData, pbrMat);
      // The exit's Fresnel weighs the sample, not its pdf: both MIS sides keep the cosine lobe's. The
      // diffuse lobe's alone (the sample's weight is its lobe's, under the mixture too): the preview's
      // skin keeps its specular lobes beside it.
      if(exitFresnelEta > 0.0F && sampleData.event_type != BSDF_EVENT_ABSORB && sampledLobe == LOBE_DIFFUSE_REFLECTION)
        sampleData.bsdf_over_pdf *= diffuseExitFresnelWeight(dot(pbrMat.N, sampleData.k2), exitFresnelEta);

      // Update the pt.throughput
      pt.throughput *= sampleData.bsdf_over_pdf;
      ray.Direction    = sampleData.k2;  // new direction
      if(materialIndex >= 0 && (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0)
        ray.Direction = rayIntoEyeChamber(ray.Direction, pbrMat.N, hit, materials + materialIndex);
      pt.lastSamplePdf = sampleData.pdf;
      // Next-event estimation above samples a light only where this holds (bounce.nextEventValid), so a
      // light this ray reaches elsewhere has no light-sample partner: its weight is 1, as after a delta
      // lobe (the env hit's delta branch rests on the same fact). Rough thin glass lit from behind, a
      // refractive exit from inside a volume: no shadow ray passes them (getShadowTransmission).
      // On the mode, not this vertex's lobe mixture: it holds whichever pdf the sample has.
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtLobeMixture) && sampleData.event_type != BSDF_EVENT_ABSORB
         && !(dot(sampleData.k2, hit.nrm) > 0.0f || bothSides))
        pt.lastSamplePdf = DIRAC;

      // A clear crossing: a refraction through a smooth surface that shadow rays pass straight
      // through (getShadowTransmission, CLEAR_CROSSING_ROUGHNESS). The previous vertex's shadow ray already
      // reached the light behind it, so the light this ray reaches beyond is weighed against that
      // vertex's light sample: the crossing is a null event for MIS. Weighed against its own (next to
      // nothing through smooth glass), the light behind a window or a cornea was counted 1 + w times.
      // A camera ray through it keeps its DIRAC: through a window, the background.
      bool clearCrossing = false;
#if GLTF_USE_TRANSMISSION
      clearCrossing = hasFlag(pushConst.flags, PathtracerFlags::ePtShadowTransmission) && !hairOn && materialIndex >= 0
                      && (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0
                      && (sampleData.event_type & (BSDF_EVENT_GLOSSY | BSDF_EVENT_IMPULSE)) != 0
                      && isClearCrossing(crossingRoughness) && materials[materialIndex].scatterStrengthFactor <= 0.0F;
#endif
      // The water's surface is crossed so by shadow rays whatever its roughness (groundWaterShadowTransmittance)
      clearCrossing = clearCrossing || (hitWater && (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0);
      if(clearCrossing)
      {
        pt.lastSamplePdf = prevSamplePdf;
        pt.lightPick     = prevLightPick;
        pt.nullDistance += payload.hitT;
      }
      else
        pt.nullDistance = 0.0F;

#if USE_DLSS_SHADER
      // #DLSS - Enable specular hit distance capture on first bounce if this is a glossy reflection
      // Matches reference implementation: only capture for BSDF_EVENT_GLOSSY_REFLECTION
      // A distance already given at the hit (an eye's tear film) stands.
      if(firstRay && pt.guide.specularHitDist == float16_t(0)
         && (sampleData.event_type == BSDF_EVENT_GLOSSY_REFLECTION || sampleData.event_type == BSDF_EVENT_IMPULSE_REFLECTION))
      {
        pt.guide.captureSpecularHit = true;
      }
#endif  // USE_DLSS_SHADER

      // The ray wasn't absorbed, so we can continue the path
      if(sampleData.event_type != BSDF_EVENT_ABSORB)
      {
        // How rough this scattering was (perceptual): a diffuse lobe 1, an impulse 0, a glossy one
        // the roughness of the lobe that took it (bsdfLobeRoughness): a bounce off the rough base
        // under a sharp coat is rough, as Falcor's isSampleRough judges it. A fibre: its longitudinal
        // roughness.
        float eventRoughness = 1.0F;
        if((sampleData.event_type & BSDF_EVENT_IMPULSE) != 0)
          eventRoughness = 0.0F;
        else if((sampleData.event_type & BSDF_EVENT_GLOSSY) != 0)
          eventRoughness = bsdfLobeRoughness(pbrMat, sampledLobe);
#if GLTF_USE_HAIR
        if(hairOn && (sampleData.event_type & BSDF_EVENT_GLOSSY) != 0)
          eventRoughness = hairHit.betaM;  // the fibre lobes; its multiple-scattering lobe is diffuse
#endif
        // A walk's own events are no bounce: entering the medium, and reflecting back into it at its
        // boundary. Leaving it by refraction after it scattered is diffuse light, as the diffuse exit is.
        bool entersWalk = false;  // set below
        bool inWalk     = false;
        int  walkEvents = 0;
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
        {
          const float3 walkSigmaS = float3(pt.volumeMedium.scatterCoefficient);
          inWalk     = ptIsInside(pt) && max(walkSigmaS.x, max(walkSigmaS.y, walkSigmaS.z)) > VOLUME_MIN_SCATTER;
          walkEvents = pt.scatterBounces;
        }
#endif

        // Continue path (bounce)
        bool isSpecular     = (sampleData.event_type & BSDF_EVENT_IMPULSE) != 0;
        bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

        // Compute the offset position for the next bounce: the side the event leaves by, since a sample
        // leaving on the other side of the face was dropped (bsdfFaceAllows; not on the ground and water,
        // where the direction says)
        float3 offsetDir = dot(ray.Direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
        ray.Origin       = safeOffsetRay(hit.pos, offsetDir, hit.offsetDistance);

        // Toggle 'pt.isInside' to track whether the ray is currently inside or outside geometry.
        // This update is necessary for correctly handling transmission (refraction) through solid (non-thin-walled) materials.
        // For thin-walled surfaces - like decals or leaves - this state isn't toggled, because there's no real "inside":
        // thickness 0 is the same test the BSDF uses to pass straight through instead of refracting, and the one shadow
        // rays use to count boundaries.
        //
        // Gated on GLTF_USE_TRANSMISSION: scenes without transmission never produce
        // BSDF_EVENT_TRANSMISSION, so this branch is dead. makeVolumeMedium only when
        // GLTF_USE_VOLUME is on (medium has nowhere to live otherwise).
#if GLTF_USE_TRANSMISSION
        // Thin-walled transmission (thickness == 0) has no interior, so it must not flip isInside.
        // The scatter preview evaluates a volumetric material as thin and likewise stays outside.
        if(isTransmission && pbrMat.thickness > 0.0f && !hitWater)
        {
          bool previewThin = false;
#if GLTF_USE_SCATTER
          previewThin = simplifyScatter && materialIndex >= 0 && materials[materialIndex].scatterStrengthFactor > 0.0f;
#endif
          if(!previewThin)
          {
#if GLTF_USE_VOLUME
            // Through the boundary (medium_stack.h.slang): into the surface's own medium, or out of it
            // into the one around it, if any
            // the back of a scattering surface met outside every medium is a way in (surfaceMedium)
            bool backEntry = false;
#if GLTF_USE_SCATTER
            backEntry = !pt.isInside && materialIndex >= 0 && materials[materialIndex].scatterStrengthFactor > 0.0F;
#endif
            const bool enters = hit.frontFace || backEntry;
            if(enters)
            {
              // (the medium around it is the path's already: set at the hit, surfaceMedium's `here`)
              pt.isInside       = true;
              pt.volumeMaterial = materialIndex;
              pt.volumeMedium   = makeVolumeMedium(pbrMat);  // Entering volume: capture the medium for scatter/absorption
            }
            else
              ptSetMedium(pt, mediumLeft(ptMedium(pt)), materials);
#if GLTF_USE_SCATTER
            // A walk starts here: its material for the diffuse exit, a fresh event budget, and the
            // entry's normal, facing out of the medium, for walk guiding (its first step is unguided)
            if(enters)
            {
              pt.scatterBounces = 0;
              pt.walkNormal     = dot(ray.Direction, hit.nrm) < 0.0F ? hit.nrm : -hit.nrm;
              pt.walkGuide      = 0.0F;
            }
            const float3 sigmaS = float3(pt.volumeMedium.scatterCoefficient);
            entersWalk          = enters && max(sigmaS.x, max(sigmaS.y, sigmaS.z)) > VOLUME_MIN_SCATTER;
#endif
#else
            pt.isInside = !pt.isInside;
#endif
          }
        }
#endif

        // Through the water's surface the path goes into the water or out of it; reflected, it stays
        // on the side it came from
        if(hitWater)
          pt.inWater = isTransmission ? waterFromAbove : !waterFromAbove;

        // A walk entered from a surface seen directly keeps the path direct until it leaves, as in
        // Unreal: the skin's light is the walk's, and the exit's own diffuse sample makes the path rough.
        if(inWalk && !ptIsInside(pt) && walkEvents > 0)
          eventRoughness = 1.0F;
        pt.prefixIndirect = pt.pathRoughness > DIRECT_PATH_ROUGHNESS;
        if(!entersWalk && !(inWalk && ptIsInside(pt)))
          pt.pathRoughness = max(pt.pathRoughness, eventRoughness);
      }
      else if(sampleData.event_type == BSDF_EVENT_ABSORB)  // If the ray is absorbed , then break
      {
        // Exit tracing rays, but still finish this iteration; in particular the visibility test
        // for the light that we may have hit.
        pt.surfaceDepth = pushConst.maxDepth;
      }

      // End scope to free live states
    }

    //-----------------------------------------------------------------------
    // Finding the shadow ray position and direction (only for a shadow ray pathTrace() will trace)
    //-----------------------------------------------------------------------
    if(bounce.nextEventValid)
    {
      bool   shadowSideForward = dot(directLight.direction, hit.nrm) > 0.0F;
      float3 shadowOffsetDir   = shadowSideForward ? hit.geonrm : -hit.geonrm;
      float3 shadowOffsetBase  = shadowSideForward ? hit.shadowPos : hit.pos;
#if GLTF_USE_SCATTER
      // FV_materials_subsurface on a surface with no volume (the white of an eye): light that enters
      // it comes back out a short way off, so what the light sees is taken at a point spread from
      // this one by an exponential radial profile of mean d (the diffusion length, averaged over
      // the channels), in the tangent plane. Thin shadows, the lashes' on the eye, blur as they do
      // in tissue; no walk, so nothing is lost in an eyeball that is not one closed medium. The tail
      // is cut at 2.5 d, short of most lid margins. The material's thickness, not the lobe's: a walk's
      // diffuse exit shades with a thin default lobe, and measured skin's exits took the offset too, a
      // centimetre at twice the scatter distance. In a hollow (the throat, between the breasts, under the
      // collarbones) the tangent plane runs into the body, the shadow ray started inside it, and the
      // walk's light was lost there: grey patches in the still.
      if(materialIndex >= 0 && materials[materialIndex].thicknessFactor <= 0.0F && any(materials[materialIndex].scatterDistance > 0.0F)
         && shadowSideForward)
      {
        const float3 dist = materials[materialIndex].scatterDistance;
        const float  d    = (dist.x + dist.y + dist.z) / 3.0F;
        const float  r    = min(-d * log(max(1.0F - rand(seed), 1e-6F)), 2.5F * d);
        const float  phi  = rand(seed) * M_TWO_PI;
        // a frame of its own: hit.bitangent carries the tangent's handedness, 0 without tangents
        float3 t, b;
        orthonormalBasis(hit.geonrm, t, b);
        shadowOffsetBase += r * (cos(phi) * t + sin(phi) * b);
      }
#endif
      bounce.shadowRayPos      = safeOffsetRay(shadowOffsetBase, shadowOffsetDir, hit.offsetDistance);
      bounce.shadowRayDir      = directLight.direction;
      bounce.shadowRayDist     = directLight.distance;

      // The light was sampled from `hit.pos`, stopping just short of the emitter. The terminator
      // offset moves the origin, so aim at that same end point: keeping direction and distance would
      // shift it through a grazing emissive triangle, which then shadows itself.
      if(directLight.distance < INFINITE)
      {
        const float3 toTarget = hit.pos + directLight.direction * directLight.distance - bounce.shadowRayPos;
        bounce.shadowRayDist  = length(toTarget);
        bounce.shadowRayDir   = toTarget / max(bounce.shadowRayDist, 1e-12F);
      }

#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
      // The spread's points (above, where spreadHere is set): one radius of the unit profile and one
      // direction in the tangent plane for all three channels, each channel's point at that radius over
      // its own rate, so each has a sample of its own profile, never weighed up or down: where nothing
      // casts a shadow nearby every point sees what this one does and no noise is added. Each point is
      // found on the skin by a probe straight down through it from as high above the plane as it is
      // far out (the skin curves away from the plane by much less), and sees the light by a shadow ray
      // of its own to the same point of the light. A channel whose probe finds no skin, or whose point
      // has the light below its horizon, keeps this point's shadow (-1, pathTrace): the spread blurs
      // what something casts on the skin, while the terminator stays as the preview shades it. So does
      // a channel whose point falls within the pixel's footprint, where a spread could not show: a figure
      // seen whole traces few of these rays, a close-up of skin all of them (on 10TEST's neck, 9-11%
      // of the moving frame's trace).
      if(spreadHere && any(bounce.spreadContribution != 0.0F))
      {
        const float3 lightPoint = hit.pos + directLight.direction * directLight.distance;
        const float  unitRadius = skinSpreadUnitRadius(rand(seed), rand(seed));
        const float  phi        = M_TWO_PI * rand(seed);
        float3       t, b;
        orthonormalBasis(hit.nrm, t, b);
        const float3 along = t * cos(phi) + b * sin(phi);
        bounce.spreadVisibility = float3(-1.0F);
        [ForceUnroll]
        for(int c = 0; c < 3; c++)
        {
          if(bounce.spreadContribution[c] == 0.0F)
            continue;
          const float r = unitRadius / spreadRate[c];
          if(r <= max(hit.offsetDistance, worldFoot))
            continue;  // within the pixel's own footprint, where no spread shows: this point's shadow
          const SpreadPoint point = probeSpreadPoint(hit.pos + along * r + hit.nrm * r, -hit.nrm, 2.0F * r);
          if(!point.found)
            continue;
          RayDesc shadowRay;
          shadowRay.Origin    = safeOffsetRay(point.hit.shadowPos, point.hit.geonrm, point.hit.offsetDistance);
          shadowRay.Direction = directLight.direction;
          shadowRay.TMin      = 0.0F;
          shadowRay.TMax      = directLight.distance;
          if(directLight.distance < INFINITE)
          {
            const float3 toTarget = lightPoint - shadowRay.Origin;
            shadowRay.TMax        = length(toTarget);
            shadowRay.Direction   = toTarget / max(shadowRay.TMax, 1e-12F);
          }
          if(dot(shadowRay.Direction, point.hit.geonrm) <= 0.0F || dot(shadowRay.Direction, point.hit.nrm) <= 0.0F)
            continue;  // the light is below this point's horizon: the terminator, not a cast shadow
          bounce.spreadVisibility[c] = raytracer.TraceShadow(shadowRay, seed, false, float3(0.0F))[c];
        }
      }
#endif
    }
  }

  return PathStepResult.eOutOfVolume;
}

//-----------------------------------------------------------------------
// PATH TRACE
//
// This function performs the main path tracing loop. It initializes the path tracer state,
// traces a ray through the scene, handles intersection and shading logic for the first or subsequent bounces,
// and updates path tracing state and bounce scratch data accordingly. It manages ray-scene intersections,
// material retrieval, handling of missed rays (e.g., environment hits),
// and prepares data for direct/indirect lighting and next events.
//-----------------------------------------------------------------------
SampleResult pathTrace(IRaytracer raytracer, RayDesc ray, inout uint seed)
{
  PathTracerState pt = {};
  pt.inWater         = groundUnderWater(ray.Origin, pushConst.frameInfo);

#if GLTF_USE_VOLUME
  // A camera path starts in the medium the camera is in (probeCameraMedium). No pointer: the scene
  // has no medium to be in.
  if(pushConst.cameraMedium != nullptr && pushConst.cameraMedium->inside != 0)
  {
    const CameraMedium cameraMedium    = *pushConst.cameraMedium;
    pt.isInside                        = true;
    pt.volumeMedium.absorption         = float32_t3(cameraMedium.absorption);
    pt.volumeMedium.scatterCoefficient = float32_t3(cameraMedium.scattering);
    pt.volumeMedium.scatterAnisotropy  = float32_t(cameraMedium.anisotropy);
  }
#endif

  while(pt.surfaceDepth < pushConst.maxDepth)
  {
    ray.Direction = normalize(ray.Direction);

    //-----------------------------------------------------------------------
    // Tracing the ray and finding the radiance, the throughput and the next event valid flag
    //-----------------------------------------------------------------------
    BounceScratch  bounce = {};
    PathStepResult step   = pathTraceOneBounce(raytracer, ray, seed, pt, bounce);
    if(step == PathStepResult.eBreak)
      break;
    if(step == PathStepResult.eVolumeContinue || step == PathStepResult.eEarlyContinue)
      continue;

    //-----------------------------------------------------------------------
    // Tracing the shadow ray
    //-----------------------------------------------------------------------
    if(bounce.nextEventValid)
    {
      RayDesc shadowRay;
      shadowRay.Origin    = bounce.shadowRayPos;
      shadowRay.Direction = bounce.shadowRayDir;
      shadowRay.TMin      = 0.0;
      shadowRay.TMax      = bounce.shadowRayDist;
      float3 shadowFactor = raytracer.TraceShadow(shadowRay, seed, bounce.shadowRayInside, bounce.shadowRayExtinction);
      // the spread's diffuse share by each channel's own shadow, or this one's where it found no point
      const float3 spreadShadow = lerp(shadowFactor, bounce.spreadVisibility, float3(bounce.spreadVisibility >= 0.0F));
      addRadiance(pt, bounce.contribution * shadowFactor + bounce.spreadContribution * spreadShadow, bounce.indirect);
#if USE_LIGHT_RIS
      // what the pixel carries to the next frame is what it found lit (light_ris.h.slang)
      if(bounce.lightReuseWritten && all(shadowFactor <= float3(0.0F)))
        pushConst.lightReservoirs[lightReuseSlot(g_reusePixel, hasFlag(pushConst.flags, PathtracerFlags::ePtLightReuseParity)) + 1].z = 0u;
#endif
    }

    //-----------------------------------------------------------------------
    // Russian Roulette
    //-----------------------------------------------------------------------
    // Not at the last vertex: the loop ends there whatever the roulette says. No cap below 1: a path
    // that keeps its energy (a white wall, blond hair) is cut by the bounce limit, not culled at
    // random and weighted up, which only made its survivors brighter and noisier.
    int  depth         = pt.surfaceDepth;
    bool fibreRoulette = false;
#if GLTF_USE_HAIR
    depth += pt.hairBounces;
    // DLSS frames: past its first fibre events a path through a lock continues at most with
    // hairRouletteDlss. Unbiased (the survivors carry the rest), noisier, for Ray Reconstruction to
    // denoise: the light a lock scatters between its fibres was three quarters of a moving frame.
    fibreRoulette = bounce.fibre && hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss)
                    && pushConst.hairRouletteDlss < 1.0F && pt.hairBounces >= pushConst.hairRouletteFrom;
#endif
    if((depth >= RR_MIN_DEPTH || fibreRoulette) && (bounce.hairEvent || bounce.glassEvent || bounce.waterEvent || pt.surfaceDepth + 1 < pushConst.maxDepth))
    {
      float rrCap = 1.0F;
#if GLTF_USE_HAIR
      if(bounce.hairEvent && pt.hairBounces >= HAIR_FREE_EVENTS)
        rrCap = HAIR_RR_CAP;  // a clear fibre's path, bounded (HAIR_FREE_EVENTS)
      if(fibreRoulette)
        rrCap = min(rrCap, pushConst.hairRouletteDlss);
#endif
      const float rrPcont = min(max(pt.throughput.x, max(pt.throughput.y, pt.throughput.z)) + RR_PCONT_FLOOR, rrCap);
      if(rrPcont < 1.0F)  // (rand() can return 1, which would end a path that has to survive)
      {
        if(rand(seed) >= rrPcont)
          break;
        pt.throughput /= rrPcont;
      }
    }

    // Consume one surface bounce, or one of the hair's, the glass's or the water's; a path cut at a fibre (hairCutDlss) ends here
#if GLTF_USE_HAIR
    if(hairCutHere(bounce.fibre, pt.hairBounces))
      pt.surfaceDepth = pushConst.maxDepth;
    else if(bounce.hairEvent)
      pt.hairBounces++;
    else
#endif
#if GLTF_USE_TRANSMISSION
    if(bounce.glassEvent)
      pt.glassBounces++;
    else
#endif
    if(bounce.waterEvent)
      pt.waterBounces++;
    else
      pt.surfaceDepth++;
  }

#if USE_RADIANCE_CACHE
  // An update path of the bounce cache: what it gathered after its vertex, and what it arrived there
  // with (before the air of the camera's segment below, which is not the vertex's)
  if(pt.cacheSlot >= 0)
  {
    const RadianceCacheSample cached = radianceCacheSample(pt.radiance - pt.cacheRadiance, pt.cacheThroughput);
    if(cached.valid)
      radianceCacheAdd(pushConst.frameInfo->radianceCache, pt.cacheSlot, cached);
  }
#endif

#if WIREFRAME
  if(pt.firstHitWireframeAlpha > 0.0)
    pt.radiance = lerp(pt.radiance, float3(0.9, 0.0, 0.0), pt.firstHitWireframeAlpha);
#endif

  // Aerial perspective, applied once to the finished path rather than per bounce: what it
  // models is the air in the *first* segment, between the eye and whatever the camera ray
  // hit, and every later bounce reaches the eye through exactly that same segment.
  //
  // `pt.solid` is the primary-hit test -- a miss saw the sky, which already integrated the whole
  // atmosphere, and hazing it a second time would grey out the horizon it just drew.
  if(pt.solid && pt.firstHitHazed && hasAerialPerspective(pushConst.frameInfo))
  {
    const float3 eye = pushConst.frameInfo->viewInv[3].xyz;
    const float2 u   = float2(rand(seed), rand(seed));
    AerialMarch  march =
        evalAerialPerspective(pt.firstHitPos, eye, pushConst.frameInfo, g_atmosphere, g_transmittance, g_irradiance, u);

    // The atmosphere lights every point of the segment with the sun; the scene may not. One
    // shadow ray from a point picked in proportion to the sun's share turns that share into an
    // unbiased estimate of what actually reaches the air -- so the air inside a helmet, a room or
    // a canyon is dark, and shafts form where the sun gets through. Only the sun's direct term:
    // the multiple scattering is light from the whole sky and is left unshadowed.
    //
    // Traced only as often as it can matter (contribution_roulette.h.slang): over a scene's few
    // metres the air's sun is a sliver of what the pixel holds, so most paths skip the ray and the
    // ones that trace it count for those that didn't. Unbiased, and the noise it adds stays under
    // the share (SceneFrameInfo::aerialSunShare) of the pixel; a distant mountain's haze, a large
    // part of its pixel, is traced every time. A share of 0 traces it on every path, as before (with
    // a chance of 1 no number is drawn).
    float3 inScatter = march.inScatter;
    if(any(march.sunEstimate > 0.0F))
    {
      inScatter -= march.sunInScatter;
      // the clouds' shadow where the picked point stands: the air under a gap is lit, the rest not
      const float3 sunPoint    = lerp(eye, pt.firstHitPos, march.sunPoint);
      const float3 sunEstimate = march.sunEstimate * cloudSunShare(sunPoint, pushConst.frameInfo);
      const float  chance      = contributionTraceChance(luminance(sunEstimate), luminance(pt.radiance * march.transmittance + inScatter),
                                                         pushConst.frameInfo->aerialSunShare);
      if(any(sunEstimate > 0.0F) && (chance >= 1.0F || rand(seed) < chance))
      {
        RayDesc shadowRay;
        shadowRay.Origin        = sunPoint;
        shadowRay.Direction     = pushConst.frameInfo->sunDirection;
        shadowRay.TMin          = 0.0F;
        shadowRay.TMax          = INFINITE;
        const float3 visibility = raytracer.TraceShadow(shadowRay, seed);
        inScatter += visibility * sunEstimate / chance;
      }
    }
    pt.radiance = pt.radiance * march.transmittance + inScatter;
  }

  // Haze in the air (haze.h.slang), on the same segment, taken as in front of the atmosphere's air:
  // at a scene's size the order makes no difference. Unlike that air it hazes a miss too, to the edge
  // of the sphere it fills, so a beam in front of the sky shows. From the pinhole's eye, as above: the
  // haze is not defocused. Not over the shadow catcher, which shows the environment behind it.
  if(pt.firstHitHazed && pushConst.frameInfo->hazeExtinction > 0.0F
     && !hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneIsOrthographic)
     && (!pt.solid || all(abs(pt.firstHitPos) < float3(1e33F))))  // a path ended before its first hit has none
  {
    const float3            eye   = pushConst.frameInfo->viewInv[3].xyz;
    const float3            toHit = pt.firstHitPos - eye;
    const float             tEnd  = pt.solid ? length(toHit) : INFINITE;
    const float3            dir   = pt.solid ? toHit / max(tEnd, 1e-20F) : pt.firstHitPos;  // a miss keeps its direction there
    const HazeSegmentResult haze  = hazeCameraSegment(raytracer, eye, dir, tEnd, seed);
    pt.radiance                   = pt.radiance * haze.transmittance + haze.inScatter;
  }

  return packSampleResult(pt);
}


//-----------------------------------------------------------------------
// Sampling the pixel
// This function samples a pixel (trace a path of rays through the scene)
// and returns the result
//-----------------------------------------------------------------------
SampleResult samplePixel(IRaytracer raytracer,
                         inout uint seed,
                         float2     samplePos,
                         float2     subpixelJitter,
                         float2     imageSize,
                         float4x4   projMatrixI,
                         float4x4   viewMatrixI,
                         float      focalDist,
                         float      aperture)
{
  const bool isOrthographic = hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneIsOrthographic);
  RayDesc    ray            = getRay(samplePos, subpixelJitter, imageSize, projMatrixI, viewMatrixI, isOrthographic);

  // Depth-of-Field (disabled for orthographic cameras)
  if(!isOrthographic)
  {
    // The focal distance is measured along the view axis, as autofocus measures it: everything on
    // the plane at that depth is sharp, not on a sphere around the eye (which focused the frame's
    // corners short of the plane).
    // The matrices are row-major, applied as mul(v, M) (as getRay does): mul(M, v) is the
    // transpose, which only an unturned camera does not tell apart, and which tilted the aperture.
    const float3 camForward  = normalize(mul(float4(0, 0, -1, 0), viewMatrixI).xyz);
    float3 focalPoint        = (focalDist / max(dot(ray.Direction, camForward), 1e-4F)) * ray.Direction;
    // `aperture` is the radius squared; the blades shape it (aperture.h.slang)
    float2 u = float2(0.0F);
    if(g_sobol)
      u = nextSobolDecision().xy;
    else
    {
      u.x = rand(seed);
      u.y = rand(seed);
    }
    const float2 lens        = sampleAperture(u.x, u.y, pushConst.bladeCount, pushConst.bladeRotation) * sqrt(aperture);
    float4 cam_right         = mul(float4(1, 0, 0, 0), viewMatrixI);
    float4 cam_up            = mul(float4(0, 1, 0, 0), viewMatrixI);
    float3 randomAperturePos = lens.x * cam_right.xyz + lens.y * cam_up.xyz;
    float3 finalRayDir       = normalize(focalPoint - randomAperturePos);

    // Set the new ray origin and direction with depth-of-field
    ray.Origin += randomAperturePos;
    ray.Direction = finalRayDir;
  }

  SampleResult sampleResult = pathTrace(raytracer, ray, seed);

  // A non-finite sample would stick in the accumulation for the rest of the still and reach the
  // denoiser: it is dropped (the clamp below passes NaN, and Inf times 0 is NaN in the mean)
  if(any(isnan(sampleResult.radiance.xyz)) || any(isinf(sampleResult.radiance.xyz)))
    sampleResult.radiance.xyz = float3(0.0F);

  // Removing fireflies. With ePtClampIndirect each indirect contribution was clamped as it was added
  // (addRadiance), and what is left here is a cap that keeps a sample within half float, which the
  // DLSS input is. Otherwise (upstream) the whole path is scaled to the clamp's level by its mean,
  // direct light and all. Only the colour: the alpha says whether the camera ray hit something.
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtClampIndirect))
  {
    const float largest = max(sampleResult.radiance.x, max(sampleResult.radiance.y, sampleResult.radiance.z));
    if(largest > 60000.0F)
      sampleResult.radiance.xyz *= 60000.0F / largest;
  }
  else
  {
    const float threshold = fireflyClampLevel();
    const float lum       = dot(sampleResult.radiance.xyz, float3(1.0F / 3.0F));
    if(threshold > 0.0F && lum > threshold)
      sampleResult.radiance.xyz *= threshold / lum;
  }

  return sampleResult;
}

//-----------------------------------------------------------------------
// Antialiasing: position of a sample inside the pixel, a Gaussian around its center. Every sample
// of a pixel uses it, so the converged image does not depend on the samples per frame.
//-----------------------------------------------------------------------
float2 gaussianSubpixelJitter(inout uint seed)
{
  if(g_sobol)
    return float2(0.5f, 0.5f) + ANTIALIASING_STANDARD_DEVIATION * sampleGaussian(nextSobolDecision().xy);
  return float2(0.5f, 0.5f) + ANTIALIASING_STANDARD_DEVIATION * sampleGaussian(float2(rand(seed), rand(seed)));
}

#if USE_GUIDE_SHADER
// Adds one sample's denoiser guides. The normal is the world-space shading normal, zero on a miss.
// Where nothing is hit, the albedo is the normalized background color, as Open Image Denoise
// recommends for emitters: that radiance is signal, not noise.
void addDenoiserGuides(SampleResult sampleResult, inout float3 albedoSum, inout float3 normalSum)
{
  float3 albedo = sampleResult.guideOutput.albedo.xyz;
#if USE_DLSS_SHADER
  // DLSS-RR's still: the sky's albedo is 0.5, as on its moving frames (NVIDIA's RR guide)
  if(sampleResult.radiance.w <= 0.0f && hasFlag(pushConst.flags, PathtracerFlags::ePtRrStillGuides))
    albedo = float3(0.5f);
  else
#endif
  if(sampleResult.radiance.w <= 0.0f)
  {
    const float3 background = max(sampleResult.radiance.xyz, float3(0.0f));
    const float  maxComp    = max(background.x, max(background.y, background.z));
    albedo                  = maxComp > 0.0f ? saturate(background / maxComp) : float3(0.0f);
  }
  albedoSum += albedo;
  normalSum += sampleResult.guideOutput.normalRoughness.xyz;
}

#if USE_DLSS_SHADER
// DLSS-RR's still (ePtRrStillGuides): one sample's specular albedo and linear roughness (w), and its guide
// depth, the surface behind clear glass's (primary surface replacement), far where nothing is hit. The
// albedo addDenoiserGuides takes is the diffuse one on this variant (the metal's share is the specular's).
void addRrStillGuides(SampleResult sampleResult, inout float4 specularSum, inout float depthMin)
{
  specularSum += float4(float3(sampleResult.guideOutput.specularAlbedo), float(sampleResult.guideOutput.normalRoughness.w));
  if(sampleResult.radiance.w <= 0.0f)
    return;
  const float3 pos  = sampleResult.guideOutput.psrValid ? sampleResult.guideOutput.psrPos : sampleResult.hitPosition;
  const float4 clip = mul(float4(pos, 1.0), pushConst.frameInfo->viewProjMatrix);
  depthMin          = min(depthMin, clip.z / clip.w);
}
#endif  // USE_DLSS_SHADER
#endif  // USE_GUIDE_SHADER

//-----------------------------------------------------------------------
// Per-pixel accumulation and GBuffer output
//-----------------------------------------------------------------------
#if USE_PIXEL_TIME
// The Time view: adds the pixels' time and samples to the sums of what each met (pixel_time.h.slang),
// one set of atomics for each slot the warp's pixels hold, a 64-bit sum by the carry of its low half
void addPixelTime(uint slot, uint ticks, uint samples)
{
  uint* sums = pushConst.frameInfo->pixelTime;
  [loop]
  for(;;)
  {
    if(WaveReadLaneFirst(slot) == slot)
    {
      const uint warpTicks   = WaveActiveSum(ticks);
      const uint warpSamples = WaveActiveSum(samples);
      if(WaveIsFirstLane())
      {
        uint before = 0u;
        InterlockedAdd(sums[slot * 4u], warpTicks, before);
        if(before + warpTicks < before)
          InterlockedAdd(sums[slot * 4u + 1u], 1u);
        InterlockedAdd(sums[slot * 4u + 2u], warpSamples);
      }
      break;
    }
  }
}

uint64_t pixelTimeClock()
{
  const uint2 c = getRealtimeClock();
  return (uint64_t(c.y) << 32) | uint64_t(c.x);
}
#endif

void processPixel(IRaytracer raytracer, float2 samplePos, float2 imageSize)
{
  // Check if the sample position is within the image bounds
  if(samplePos.x >= imageSize.x || samplePos.y >= imageSize.y)
    return;
#if USE_PIXEL_TIME
  // from here to the result: the samples, the selection ray, the guides' sums (the host leaves the
  // adaptive stop off, so no warp returns early)
  const uint64_t clockStart = pixelTimeClock();
  g_timeSlot                = kPixelTimeUnset;
  g_timeCount               = 0u;
#endif

  // Check if the sample position is the mouse coordinate
  if(samplePos.x == pushConst.mouseCoord.x && samplePos.y == pushConst.mouseCoord.y)
  {
    // Set the debug flag to true (used for printing debug information)
    doDebug = true;
  }

  // Converged pixels of an accumulation take no more samples (ePtAdaptiveStop), a warp at a time: a
  // thread whose warp still works costs the same idle, so a converged pixel among unconverged ones goes
  // on sampling (and is unconverged again if a bright path arrives). Before any trace, while the warp
  // is the launch's tile and not one SER regrouped. The value and the guides of a pixel that stops stay
  // as they are; eConvergence is written by the accumulation's first frame below.
  const bool adaptiveStop = hasFlag(pushConst.flags, PathtracerFlags::ePtAdaptiveStop);
  const bool accumulating = !hasFlag(pushConst.flags, PathtracerFlags::ePtFirstFrame) && pushConst.totalSamples > 0;
  if(adaptiveStop && accumulating && WaveActiveAllTrue(outImages[int(OutputImage::eConvergence)][int2(samplePos)].y > 0.0F))
    return;

  // A pass traces one sample a pixel, and the host runs a pass for each sample of the frame
  // (PathTracer::onRender): this one is the frame's first, or a later one (frameSample)
  const bool laterSample = pushConst.frameSample > 0;

#if USE_LIGHT_RIS
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtLightReuse) && pushConst.lightReservoirs != nullptr)
  {
    g_reusePixel       = int2(samplePos);
    g_reuseFirstSample = !laterSample;
    if(!laterSample)
      pushConst.lightReservoirs[lightReuseSlot(g_reusePixel, hasFlag(pushConst.flags, PathtracerFlags::ePtLightReuseParity)) + 1] = uint4(0u);
  }
#endif

#if USE_RADIANCE_CACHE
  // The bounce cache, on a DLSS frame the host gave a table for (a frame after the setting went off has
  // none), and on an accumulation's only for the debug setting that measures it; one pixel in sixteen
  // feeds it
  g_cacheOn = (hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss) || pushConst.frameInfo->radianceCacheStill != 0)
              && pushConst.frameInfo->radianceCache != nullptr && pushConst.frameInfo->radianceCacheSlots > 0;
  g_cacheUpdate = g_cacheOn && radianceCacheUpdatePixel(int(samplePos.x), int(samplePos.y), pushConst.frameInfo->radianceCacheFrame);
#endif

  // The shadow rays' roulette goes by the pixel's mean once the accumulation has a few samples of it
  // (the Time view's image holds no light)
  g_shadowRest = 0.0F;
#if !USE_PIXEL_TIME
  if(pushConst.frameInfo->shadowRayShare > 0.0F && accumulating && pushConst.totalSamples >= SHADOW_ROULETTE_FROM
     && !hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
  {
    const float mean = luminance(outImages[int(OutputImage::eResultImage)][int2(samplePos)].xyz);
    g_shadowRest     = (mean > 0.0F && mean < 1e30F) ? mean : 0.0F;
  }
#endif

  // Initialize the random number
  // pushConst.seed selects another sequence (0: the default one), e.g. for a reference independent of a test render;
  // each of a frame's samples has its own
  uint seed = xxhash32(uint3(uint2(samplePos.xy), uint(pushConst.frameCount) + uint(pushConst.seed) * 0x9E3779B9u
                                                      + uint(pushConst.frameSample) * 0x85EBCA6Bu));
  // A Sobol pixel draws its points at its sample index, under a scramble of its own that every frame keeps
  g_sobol      = hasFlag(pushConst.flags, PathtracerFlags::ePtSobol) && !hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss);
  g_sobolPixel = xxhash32(uint3(uint2(samplePos.xy), uint(pushConst.seed) + 0x5bd1e995u));
  beginSobolSample(uint(pushConst.totalSamples));
  g_hitToCount = pushConst.hitSignal != nullptr && hitSignalPixel(int(samplePos.x), int(samplePos.y), pushConst.frameCount);

  bool firstFrame = hasFlag(pushConst.flags, PathtracerFlags::ePtFirstFrame);

  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  // If DLSS is used, the jitter is on the entire frame, not just the pixel.
  float2 subpixelJitter = float2(0.5f, 0.5f);

  // #DLSS - use the DLSS jitter and frame index (not resetting to zero)
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
  {
    subpixelJitter += pushConst.frameInfo.jitter;
  }
  else
  {
    subpixelJitter = gaussianSubpixelJitter(seed);
  }

  // One sample of the pixel. The frame traces exactly the samples the host counts, a pass each: a
  // frame DLSS denoises gets ptSamplesDlss (1 but for the DLSS refine of a still: DLSS does its own
  // temporal accumulation); with DLSS-RR's still on, the frames after DLSS's frame 0 get Auto SPP's
  // or ptSamples' count. samplePixel is called here alone: called again in a loop for a frame's later
  // samples, the driver compiled the whole kernel twice, and the second copy carried the frame's sums
  // through every trace, a fifth to a half slower a sample (RENDERING_ARCHITECTURE.md, "The path
  // tracer's build time").
  SampleResult sampleResult = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo->projInv,
                                          pushConst.frameInfo->viewInv, pushConst.focalDistance, pushConst.aperture);
  float4 pixelColor = sampleResult.radiance;
  float  lumSquared = square(luminance(sampleResult.radiance.xyz));  // the sample's second moment, for ePtAdaptiveStop

#if USE_GUIDE_SHADER
  // The denoiser guides average every sample of the frame, like the color
  const bool denoiserGuides = hasFlag(pushConst.flags, PathtracerFlags::ePtDenoiserGuides);
  float3     albedoSum      = float3(0.0f);
  float3     normalSum      = float3(0.0f);
  int        guideCount     = 0;
#if USE_DLSS_SHADER
  const bool rrStillGuides = denoiserGuides && hasFlag(pushConst.flags, PathtracerFlags::ePtRrStillGuides);
  float4     specularSum   = float4(0.0f);
  float      depthMin      = 1.0f;
#endif
  if(denoiserGuides)
  {
    addDenoiserGuides(sampleResult, albedoSum, normalSum);
    guideCount++;
#if USE_DLSS_SHADER
    if(rrStillGuides)
      addRrStillGuides(sampleResult, specularSum, depthMin);
#endif
  }
#endif


  // NDC depth from first-hit world position (Vulkan [0,1] range, 1.0 = far/miss).
  // Written to two targets: outDepth (the main G-buffer's, at this frame's size: a DLSS frame's fills
  // the image's corner) and eDlssDepth (DLSS guide buffer, every frame when DLSS is active).
  const bool hasSolidHit = sampleResult.radiance.w > 0.0f;
  float4     clipPos =
      hasSolidHit ? mul(float4(sampleResult.hitPosition, 1.0), pushConst.frameInfo->viewProjMatrix) : float4(0, 0, 1, 1);
  float ndcDepth = hasSolidHit ? (clipPos.z / clipPos.w) : 1.0f;

  // The object IDs, from a selection ray of their own, when the host asks (PathTracer::setupPushConstant:
  // every DLSS frame is a first frame, and most have nothing reading them); the NDC depth on the first frame
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtWriteSelection))
  {
    uint objectID                                              = traceSelectionRay(raytracer, samplePos, imageSize);
    outImages[int(OutputImage::eSelectImage)][int2(samplePos)] = float4(asfloat(objectID), 0.0, 0.0, 0.0);
  }
  // Through an accumulation, the nearest depth any of its samples saw: the first frame's one jittered
  // sample often misses the strands that cover part of a pixel of hair, and the grid's axis showed
  // through the hair of a still as a dotted line.
  // The depth starts over where the image does (sample_schedule::frameStartsDepth, which tells the
  // overlays its size): on a first frame, and on the frame a still takes over from DLSS, its samples at
  // none again. DLSS's depth is at its input size, in the image's corner: the nearest of that and this
  // frame's, at the image's own size, was two pictures in one.
  const bool dlssFrame = hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss);
  if(firstFrame || (!dlssFrame && pushConst.totalSamples == 0))
    outDepth[int2(samplePos)] = ndcDepth;
  else if(!dlssFrame && ndcDepth < outDepth[int2(samplePos)])
    outDepth[int2(samplePos)] = ndcDepth;

#if USE_PIXEL_TIME
  // The Time view: the image is the pixel's time a sample against the frame's average, in colour (a
  // display value, as the debug views' are); an accumulation keeps the mean of its frames' times in the
  // convergence image (bound for it, its adaptive stop off), a DLSS frame shows its own. A count mode
  // (pixelTimeCount) shows and sums the pixel's count in place of its time, unshifted.
  {
    const bool     counting  = pushConst.frameInfo->pixelTimeCount != kPixelTimeCountClock;
    const uint64_t elapsed   = counting ? uint64_t(g_timeCount) : pixelTimeClock() - clockStart;
    const float    perSample = float(elapsed);
    float          mean      = perSample;
    if(!hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
    {
      if(!firstFrame && pushConst.totalSamples > 0)
        mean = (outImages[int(OutputImage::eConvergence)][int2(samplePos)].x * float(pushConst.totalSamples)
                + perSample)
               / float(pushConst.totalSamples + 1);
      outImages[int(OutputImage::eConvergence)][int2(samplePos)] = float4(mean, 0.0F, 0.0F, 0.0F);
    }
    // before a frame has been summed, against a tenth of a millisecond of NVIDIA's clock (or ten of a count)
    const float scale = pushConst.frameInfo->pixelTimeScale > 0.0F ? pushConst.frameInfo->pixelTimeScale : counting ? 0.1F : 1e-5F;
    pixelColor        = float4(pixelTimeColor(mean * scale), pixelColor.w);
    if(pushConst.frameInfo->pixelTime != nullptr)
      addPixelTime(g_timeSlot == kPixelTimeUnset ? kPixelTimeSky : g_timeSlot,
                   uint(min(counting ? elapsed : elapsed >> kPixelTimeUnitShift, uint64_t(0x7FFFFFFFu))), 1u);
  }
  outImages[int(OutputImage::eResultImage)][int2(samplePos)] = pixelColor;
#else
  // Saving result
  // Nothing accumulated yet (the first frame, or the first after DLSS's frame 0): replace the value in
  // the buffer, never weigh it by 0, which an infinite value would turn into NaN. A DLSS frame's first
  // sample replaces it too, and its later ones (totalSamples then counts the frame's so far) join it.
  if(firstFrame || pushConst.totalSamples == 0 || (hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss) && !laterSample))
  {
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] = pixelColor;
  }
  else
  {
    // Do accumulation over time using uniform weighting
    float  totalSamplesAfter = float(pushConst.totalSamples + 1);
    float4 old_color         = outImages[0][int2(samplePos)];
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] =
        (old_color * pushConst.totalSamples + pixelColor) / totalSamplesAfter;
  }
#endif

  // Whether the pixel's mean has converged, after this frame (adaptiveConverged)
  if(adaptiveStop)
  {
    const int2  pixel   = int2(samplePos);
    const int   samples = (accumulating ? pushConst.totalSamples : 0) + 1;
    float       second  = lumSquared;
    if(accumulating)
      second = (outImages[int(OutputImage::eConvergence)][pixel].x * pushConst.totalSamples + lumSquared) / float(samples);
    const float mean = luminance(outImages[int(OutputImage::eResultImage)][pixel].xyz);
    // where the tone mapper flattens the pixel, the table's widening, between its entries
    float       relax    = 1.0F;
    const float position = adaptiveRelaxPosition(mean, pushConst.adaptiveRelaxBase);
    if(position >= 0.0F)
    {
      const int   i = int(position);
      const int   j = min(i + 1, ADAPTIVE_RELAX_STOPS - 1);
      relax         = lerp(pushConst.adaptiveRelax[i >> 2][i & 3], pushConst.adaptiveRelax[j >> 2][j & 3], position - float(i));
    }
    const bool stop = adaptiveConverged(second, mean, samples, pushConst.adaptiveThreshold, pushConst.adaptiveMin, relax);
    outImages[int(OutputImage::eConvergence)][pixel] = float4(second, stop ? 1.0F : 0.0F, 0.0F, 0.0F);
    // the pixels still sampling, a warp's at a time, for the host's stop of the whole still
    // (ptAdaptiveDone); a warp that skipped the frame above has none
    const uint active = WaveActiveCountBits(!stop);
    if(pushConst.adaptiveActive != nullptr && active > 0 && WaveIsFirstLane())
      InterlockedAdd(*pushConst.adaptiveActive, active);
  }

#if USE_DLSS_SHADER
  // #DLSS - Storing the GBuffer for the DLSS denoiser (DLSS-only variant; the specular guide targets
  // do not exist in the build for the other denoisers or none).
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
  {
    // Combined camera + instance motion. w selects the projection model: surface hits use w=1 (the
    // previous position carries object motion); sky/background uses w=0, i.e. a point at infinity along
    // the view direction, so camera translation cancels and only rotation moves the sky (no far-plane
    // magic). sampleResult.hitPosition holds a world position for surfaces and a view direction for sky.
    //
    // Through clear glass in the Improved transparency mode, depth and motion describe the surface
    // seen through it (primary surface replacement); hitPositionPrev already holds its previous
    // position. The main depth (outDepth) keeps the glass, which picking and overlays rely on.
    float  hitW       = hasSolidHit ? 1.0f : 0.0f;
    float3 guidePos   = sampleResult.hitPosition;
    float  guideDepth = ndcDepth;
    if(sampleResult.guideOutput.psrValid)
    {
      guidePos             = sampleResult.guideOutput.psrPos;
      const float4 psrClip = mul(float4(guidePos, 1.0), pushConst.frameInfo->viewProjMatrix);
      guideDepth           = psrClip.z / psrClip.w;
    }
    float2 motionVec = calculateMotionVector(float4(guidePos, hitW), float4(sampleResult.hitPositionPrev, hitW),
                                             pushConst.frameInfo->prevMVP, pushConst.frameInfo->viewProjMatrix, imageSize);
    // The motion of the reflection the pixel shows (RR's pInMotionVectorsReflections), which RR then
    // takes instead of working it out from the specular hit distance, so it must cover every pixel.
    // A reflection's image lies the hit distance beyond the first hit along the camera ray, as RR puts
    // it (mirrorImagePoint), and moves as specularDelta says: in a still scene, RR's own. The sky's
    // lies at infinity that way and moves as the sky does. A pixel with no reflection: its surface's motion.
    // An eye's distance is counted from its iris (mirrorImageDistance), under a millimetre behind the
    // film this counts from: a shift along the camera ray, which moves nothing on screen.
    float2      specularMotion = motionVec;
    const float specularDist   = sampleResult.guideOutput.specularHitDist;
    if(hasSolidHit && specularDist > 0.0f)
    {
      const float3 viewDir = normalize(sampleResult.hitPosition - pushConst.frameInfo->viewInv[3].xyz);
      if(specularDist >= 65504.0f)
        specularMotion = calculateMotionVector(float4(viewDir, 0.0f), float4(viewDir, 0.0f), pushConst.frameInfo->prevMVP,
                                               pushConst.frameInfo->viewProjMatrix, imageSize);
      else
      {
        const float3 image = mirrorImagePoint(sampleResult.hitPosition, viewDir, specularDist);
        specularMotion = calculateMotionVector(float4(image, 1.0f), float4(image + float3(sampleResult.guideOutput.specularDelta), 1.0f),
                                               pushConst.frameInfo->prevMVP, pushConst.frameInfo->viewProjMatrix, imageSize);
      }
    }
    outImages[int(OutputImage::eDlssDepth)][int2(samplePos)]           = float4(guideDepth);
    outImages[int(OutputImage::eDlssMotion)][int2(samplePos)]          = float4(motionVec, 0, 0);
    outImages[int(OutputImage::eDlssSpecularMotion)][int2(samplePos)]  = float4(specularMotion, 0, 0);
    outImages[int(OutputImage::eDlssNormalRoughness)][int2(samplePos)] = sampleResult.guideOutput.normalRoughness;
    // the sky's albedo is 0.5, as NVIDIA's RR guide asks: 0 makes it boil at the silhouette
    float4 guideAlbedo = sampleResult.guideOutput.albedo;
    if(!hasSolidHit)
      guideAlbedo = float4(0.5f);
    outImages[int(OutputImage::eDlssAlbedo)][int2(samplePos)] = guideAlbedo;
    outImages[int(OutputImage::eDlssSpecAlbedo)][int2(samplePos)] = float4(sampleResult.guideOutput.specularAlbedo.xyz, 1.0f);
    outImages[int(OutputImage::eDlssSpecularHitDist)][int2(samplePos)] = float4(sampleResult.guideOutput.specularHitDist);
    if(hasFlag(pushConst.flags, PathtracerFlags::ePtWriteNrMask))
      outImages[int(OutputImage::eNrMask)][int2(samplePos)] = float4(sampleResult.guideOutput.nrMask);
    if(pushConst.hairResponsivity != 0.0f)
      outImages[int(OutputImage::eDlssResponsivity)][int2(samplePos)] = float4(sampleResult.guideOutput.responsivity);
  }
#endif  // USE_DLSS_SHADER

#if USE_GUIDE_SHADER
  if(denoiserGuides)
  {
    // The guides of the denoiser of the accumulation are averaged over the samples like the color,
    // so their edges are anti-aliased (and blurred by depth of field) the same way. They have their
    // own sample count: they restart when the denoiser is turned on, while the color goes on. Each
    // denoiser converts them to what it reads.
    const float3 albedo  = albedoSum / float(guideCount);
    const float3 normal  = normalSum / float(guideCount);
    const int2   pixel   = int2(samplePos);
    float        normalW = 1.0f;
#if USE_DLSS_SHADER
    // DLSS-RR's still reads the roughness packed in the normal's w (RrStillDenoiser)
    const float4 specular = specularSum / float(guideCount);
    if(rrStillGuides)
      normalW = specular.w;
#endif
    const bool restart = firstFrame || pushConst.guideSamples == 0;
    if(restart)
    {
      outImages[int(OutputImage::eDenoiserAlbedo)][pixel] = float4(albedo, 1.0f);
      outImages[int(OutputImage::eDenoiserNormal)][pixel] = float4(normal, normalW);
    }
    else
    {
      const float weight = float(guideCount) / float(pushConst.guideSamples + guideCount);
      outImages[int(OutputImage::eDenoiserAlbedo)][pixel] =
          lerp(outImages[int(OutputImage::eDenoiserAlbedo)][pixel], float4(albedo, 1.0f), weight);
      outImages[int(OutputImage::eDenoiserNormal)][pixel] =
          lerp(outImages[int(OutputImage::eDenoiserNormal)][pixel], float4(normal, normalW), weight);
    }
#if USE_DLSS_SHADER
    if(rrStillGuides)
    {
      const float4 spec = float4(specular.xyz, 1.0f);
      outImages[int(OutputImage::eRrStillSpecAlbedo)][pixel] =
          restart ? spec :
                    lerp(outImages[int(OutputImage::eRrStillSpecAlbedo)][pixel], spec, float(guideCount) / float(pushConst.guideSamples + guideCount));
      outImages[int(OutputImage::eRrStillDepth)][pixel] =
          float4(restart ? depthMin : min(outImages[int(OutputImage::eRrStillDepth)][pixel].x, depthMin));
    }
#endif
  }
#endif  // USE_GUIDE_SHADER
}

//-----------------------------------------------------------------------
// The medium the camera is in
//
// A path knows it is inside a volume only by having crossed into one, and a camera ray starts in
// clear air. From a camera inside a fog, the path would take the fog's far wall for its near one:
// no fog in front of the camera, and fog behind the wall. So once per accumulation of a scene with a
// KHR_materials_volume material, the host runs the path tracer's own pipeline with
// ePtProbeCameraMedium, and one thread follows the center camera ray to the nearest surface that
// bounds a medium: met from behind, the camera is inside it. Other surfaces are passed through, so
// an object between the camera and the wall does not hide it. A path is in one medium at a time, so
// the nearest boundary is the only one that counts. Depth of field moves the origin within the
// aperture, which is assumed to stay in the same medium.
//
// It traces and evaluates surfaces exactly as the paths do. Being compiled into the path-tracing
// kernel costs it 1-2% of ray-query throughput (Sponza), most of it for the branch alone: a leaner
// probe (opaque, material factors only) measured the same, so only a pipeline of its own would
// remove it.
//-----------------------------------------------------------------------
static const int CAMERA_MEDIUM_MAX_SURFACES = 32;  // Surfaces passed through before giving up (clear air)

void probeCameraMedium(IRaytracer raytracer)
{
  CameraMedium result = {};

#if GLTF_USE_VOLUME
  GltfShadeMaterial*   materials       = pushConst.gltfScene->materials;
  GltfTextureInfo*     texInfos        = pushConst.gltfScene->textureInfos;
  GltfRenderNode*      renderNodes     = pushConst.gltfScene->renderNodes;
  GltfRenderPrimitive* renderPrimitive = pushConst.gltfScene->renderPrimitives;

  const float2 imageSize      = float2(pushConst.renderSize);
  const bool   isOrthographic = hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneIsOrthographic);
  RayDesc      ray            = getRay(imageSize * 0.5F, float2(0.0F), imageSize, pushConst.frameInfo->projInv,
                                       pushConst.frameInfo->viewInv, isOrthographic);
  uint         seed           = xxhash32(uint3(0, 0, uint(pushConst.frameCount)));

  for(int i = 0; i < CAMERA_MEDIUM_MAX_SURFACES; i++)
  {
    HitPayload payload;
    raytracer.Trace(ray, payload, seed, 0xFF, CAMERA_RAY_FLAGS);
    if(payload.hitT == INFINITE)
      break;

    GltfRenderNode      node  = renderNodes[payload.rnodeID];
    GltfRenderPrimitive prim  = renderPrimitive[payload.rprimID];
    float3              barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
    HitState hit = getHitState(prim, barys, node.worldToObject, node.objectToWorld, payload.primitiveID, ray.Direction);

    // A boundary the path would cross into a medium through
    if(surfaceBoundsMedium(materials + max(0, node.materialID)))
    {
      if(!hit.frontFace)
      {
        MeshState mesh          = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, !hit.frontFace, 0.0F);
        mesh.baseColorVertexMul = hit.color;
        mesh.masks              = hit.masks;
        // the thin preview has no medium to be inside of
        PbrMaterial mat = evaluateMaterial(max(0, node.materialID), mesh, allImages, allSamplers, texInfos, materials,
                                           hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin),
                                           hasFlag(pushConst.flags, PathtracerFlags::ePtSkinFaceOn));
        const VolumeMedium medium = makeVolumeMedium(mat);
        result.inside             = 1;
        result.absorption         = float3(medium.absorption);
        result.scattering         = float3(medium.scatterCoefficient);
        result.anisotropy         = float(medium.scatterAnisotropy);
      }
      break;
    }

    // Past this surface: the same ray, from just beyond the hit. The hardware finds the same distance
    // to the same triangle again, so the next float excludes it, with no offset to size.
    ray.TMin = nextFloatUp(payload.hitT);
  }
#endif

  *pushConst.cameraMedium = result;
}

//-----------------------------------------------------------------------
// The pixel readout (view.inspect, PixelInspect)
//
// One thread, with ePtInspectPixel: the camera ray through the centre of render pixel mouseCoord, what
// it meets first (a mesh, the ground or the water's surface, as the paths find them) evaluated as the
// paths evaluate their first hit, and what the frame's image holds at that pixel. A fixed draw for
// stochastic opacity, so a strand or a lash gives the same answer each time it is read. It shares the
// camera medium's branch in the entry points: a second branch of its own would cost the path tracing
// kernel as that one does (probeCameraMedium's note).
//-----------------------------------------------------------------------
void inspectPixel(IRaytracer raytracer)
{
  PixelInspect result = {};
  const int2   pixel  = clamp(int2(pushConst.mouseCoord), int2(0), max(int2(pushConst.renderSize) - 1, int2(0)));
  result.ids          = int4(-1, -1, -1, hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss) ? 1 : 0);
  result.uv           = float4(0.0F, 0.0F, float2(pixel));
  // a still's mean so far, or a moving frame's sample; the host says how many samples it holds
  result.radiance = float4(outImages[int(OutputImage::eResultImage)][pixel].xyz, float(pushConst.totalSamples));

  SceneFrameInfo* frameInfo      = pushConst.frameInfo;
  const bool      isOrthographic = hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneIsOrthographic);
  RayDesc ray = getRay(float2(pixel), float2(0.5F), float2(pushConst.renderSize), frameInfo->projInv, frameInfo->viewInv, isOrthographic);
  uint    seed = 0x9E3779B9u;

  HitPayload payload;
  raytracer.Trace(ray, payload, seed, 0xFF, CAMERA_RAY_FLAGS);
  GltfRenderNode      node = {};
  GltfRenderPrimitive prim = {};
  HitState            hit  = {};
  if(payload.hitT != INFINITE)
  {
    node         = pushConst.gltfScene->renderNodes[payload.rnodeID];
    prim         = pushConst.gltfScene->renderPrimitives[payload.rprimID];
    float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
    hit          = getHitState(prim, barys, node.worldToObject, node.objectToWorld, payload.primitiveID, ray.Direction);
  }
  bool       hitPlane       = checkInfinitePlaneIntersection(ray, payload, hit, frameInfo, seed);
  bool       waterFromAbove = false;
  const bool hitWater       = checkWaterIntersection(ray, payload, hit, frameInfo, waterFromAbove);
  if(hitWater)
    hitPlane = false;
  if(payload.hitT == INFINITE)
  {
    *pushConst.pixelInspect = result;
    return;
  }

  const RayCone cone      = {0.0F, pushConst.pixelAngle};
  const float   worldFoot = rayConeWorldFootprint(cone, payload.hitT, hit.geonrm, -ray.Direction);
  PbrMaterial   mat;
  bool          front = hit.frontFace;
  if(hitPlane)
  {
    mat        = groundMaterial(hit.pos, worldFoot, frameInfo);
    front      = ray.Direction.y < 0.0F;
    result.ids = int4(-2, -1, -1, result.ids.w);
  }
  else if(hitWater)
  {
    mat        = waterSurfaceMaterial(hit.pos, hit.geonrm, waterFromAbove, worldFoot, frameInfo);
    front      = waterFromAbove;
    result.ids = int4(-3, -1, -1, result.ids.w);
  }
  else
  {
    const int   materialIndex   = max(0, node.materialID);
    MeshState   mesh            = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, false,
                                            worldFoot * hit.texelDensity * pushConst.texGradScale);
    mesh.baseColorVertexMul     = hit.color;
    mesh.masks                  = hit.masks;
#if GLTF_USE_PROCEDURAL
    mesh.water = sampleWater(pushConst.gltfScene, uint(materialIndex), hit.uv[0]);
#endif
    mat = evaluateMaterial(materialIndex, mesh, allImages, allSamplers, pushConst.gltfScene->textureInfos,
                           pushConst.gltfScene->materials, hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin),
                           hasFlag(pushConst.flags, PathtracerFlags::ePtSkinFaceOn));
    result.ids  = int4(payload.rnodeID, payload.primitiveID, materialIndex, result.ids.w);
    result.uv.xy = hit.uv[0];
  }

  result.position  = float4(hit.pos, payload.hitT);
  result.normal    = float4(mat.N, front ? 1.0F : 0.0F);
  result.baseColor = float4(mat.baseColor, mat.opacity);
  result.emission  = float4(mat.emissive, mat.transmission);
  result.surface   = float4(mat.metallic, sqrt(max(mat.roughness.x, 0.0F)), mat.ior2, mat.specular);
  result.layers    = float4(mat.clearcoat, mat.clearcoatRoughness, max(mat.sheenColor.x, max(mat.sheenColor.y, mat.sheenColor.z)), mat.thickness);
  result.medium    = float4(0.0F, 0.0F, 0.0F, mat.diffuseTransmissionFactor);
#if GLTF_USE_VOLUME
  // a scattering medium's single-scattering albedo (skin's, wax's), where it scatters at all
  if(max(mat.scatterCoefficient.x, max(mat.scatterCoefficient.y, mat.scatterCoefficient.z)) > 0.0F)
  {
    const VolumeMedium medium = makeVolumeMedium(mat);
    result.medium.xyz = float3(medium.scatterCoefficient) / max(float3(medium.scatterCoefficient + medium.absorption), float3(1e-12F));
  }
#endif
  *pushConst.pixelInspect = result;
}

//-----------------------------------------------------------------------
// RAY-QUERY COMPUTE
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(PT_WORKGROUP_X, PT_WORKGROUP_Y, 1)]
// (subgroup vote and ballot: ePtAdaptiveStop's warp-wide stop and its count of the pixels still sampling, in processPixel)
[require(spvGroupNonUniformVote)]
[require(spvGroupNonUniformBallot)]
#if USE_PIXEL_TIME
// (the Time view's clock, and its warp sums)
[require(spvShaderClockKHR)]
[require(spvGroupNonUniformArithmetic)]
#endif
void computeMain(uint3 threadIdx: SV_DispatchThreadID)
{
  RAYQUERY_USE_OPACITY_MICROMAP(computeMain);
  // The render size, not the output image's: DLSS's frames trace its smaller input size.
  RayQueryRaytracer raytracer;
  // the one-thread modes, in one branch (inspectPixel's note)
  if(hasFlag(pushConst.flags, int(PathtracerFlags::ePtProbeCameraMedium) | int(PathtracerFlags::ePtInspectPixel)))
  {
    if(all(threadIdx.xy == 0))
    {
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtProbeCameraMedium))
        probeCameraMedium(raytracer);
      else
        inspectPixel(raytracer);
    }
    return;
  }
  float2 samplePos = (float2)threadIdx.xy;
  processPixel(raytracer, samplePos, (float2)pushConst.renderSize);
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("raygeneration")]
// (subgroup vote and ballot: ePtAdaptiveStop's warp-wide stop and its count of the pixels still sampling, in processPixel)
[require(spvGroupNonUniformVote)]
[require(spvGroupNonUniformBallot)]
#if USE_PIXEL_TIME
// (the Time view's clock, and its warp sums)
[require(spvShaderClockKHR)]
[require(spvGroupNonUniformArithmetic)]
#endif
void rgenMain()
{
  RAYQUERY_USE_OPACITY_MICROMAP(rgenMain);  // Ray queries here too (e.g. dlssTraceSkipThinGlass)
  TraditionalRaytracer raytracer;
  // the one-thread modes, in one branch (inspectPixel's note)
  if(hasFlag(pushConst.flags, int(PathtracerFlags::ePtProbeCameraMedium) | int(PathtracerFlags::ePtInspectPixel)))
  {
    if(all(DispatchRaysIndex().xy == 0))
    {
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtProbeCameraMedium))
        probeCameraMedium(raytracer);
      else
        inspectPixel(raytracer);
    }
    return;
  }
  float2 samplePos = (float2)DispatchRaysIndex().xy;
  processPixel(raytracer, samplePos, (float2)pushConst.renderSize);
}

//-----------------------------------------------------------------------
// CLOSEST HIT
//-----------------------------------------------------------------------
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  payload.hitT        = RayTCurrent();
  payload.rprimID     = InstanceID();
  payload.rnodeID     = InstanceIndex();
  payload.primitiveID = PrimitiveIndex();
  payload.bary        = attr.barycentrics;
}


//-----------------------------------------------------------------------
// MISS
//-----------------------------------------------------------------------
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
  payload.hitT = INFINITE;
}

[shader("miss")]
void rmissShadow(inout ShadowPayload payload)
{
  // When we miss, we've passed through all surfaces along the ray.
  // Keep the accumulated transmission -- don't reset it. totalTransmission already contains
  // the accumulated value from rahitShadow. Shadow rays skip closest-hit (TraceShadow), so this
  // flag is the only way the caller tells a clear path from a blocked one.
  payload.missed = true;
}

//-----------------------------------------------------------------------
// INTERSECTION (Any Hit)
//-----------------------------------------------------------------------
[shader("anyhit")]
void rahitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
#if USE_VISUALIZE
  // Opacity-micromap debug view: any-hit only runs for "unknown" micro-triangles (OMM-opaque
  // ones are committed by the hardware without any-hit, transparent ones are culled). Flag it and
  // accept the hit so it is colored as unknown; the path breaks at the first hit anyway.
  if(pushConst.frameInfo->visualization == Visualization::eOpacityMicromap)
  {
    payload.ommUnknown = true;
    AcceptHitAndEndSearch();
  }
#endif

  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint instanceID   = InstanceIndex();
  uint renderPrimID = InstanceID();
  uint triangleID   = PrimitiveIndex();

  // Retrieve the Primitive mesh buffer information
  GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
  GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

  float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics, ObjectRayDirection());
  if(stochasticOpacityU(instanceID, triangleID, barycentrics, payload.seed) >= opacity && !opacityFilmKept(renderNode, opacity))
  {
    IgnoreHit();
  }
}


[shader("anyhit")]
void rahitShadow(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint instanceID   = InstanceIndex();
  uint renderPrimID = InstanceID();
  uint primitiveID  = PrimitiveIndex();

  // Get the built-in ray tracing variables
  float4x3 worldToObject = WorldToObject4x3();
  float3   worldRayDir   = WorldRayDirection();
  float    hitT          = RayTCurrent();

  // Retrieve the Primitive mesh buffer information
  GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
  GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

  // Any-hit order along the ray is not guaranteed; a fully correct version would chain
  // closest-hit traces instead of accumulating here (see shadowCrossSurface's segment).
  if(!shadowCrossSurface(renderNode, renderPrim, instanceID, int(primitiveID), barycentrics, worldToObject, worldRayDir,
                         hitT, payload.totalTransmission, payload.isInside, payload.extinction, payload.prevHitT, payload.seed))
    AcceptHitAndEndSearch();
  // We want all possible intersections
  IgnoreHit();
}
