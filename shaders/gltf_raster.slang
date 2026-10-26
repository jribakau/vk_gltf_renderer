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

#include "gltf_scene_io.h.slang"
#include "gltf_vertex_access.h.slang"
// Raster runs in a fragment shader, where implicit-LOD .Sample() has valid screen-space
// derivatives and gives correct hardware mipmapping. Opt into that path in getTexture();
// the path tracer (compute/raygen) has no such derivatives and uses explicit mip 0 instead.
#define RASTER_PIPELINE 1
#include "gltf_material_eval.h.slang"
#include "shaderio.h"
#ifdef HAS_DLSS_MOTION
#include "dlss_util.h"  // calculateMotionVector (clip-pos overload) - only when DLAA motion attachment is built in
#endif
#include "get_hit.h.slang"
#include "common.h.slang"

// NVSHADERS are under: https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/functions.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/light_contrib.h.slang"

#include "sky_background.h.slang"  // makeSkySunLight
#include "ground.h.slang"
#include "aerial_perspective.h.slang"
#include "gltf_light_ies.h.slang"
#include "nvshaders/pbr_ggx_microfacet.h.slang"
#include "nvshaders/pbr_direct_eval.h.slang"
#include "nvshaders/pbr_ibl_eval.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sample_blur.h.slang"


// Bindings
// clang-format off
[[vk::push_constant]]                                       ConstantBuffer<RasterPushConstant>  pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]                Texture2D                           allImages[];
[[vk::binding(BindingPoints::eSamplers, 0)]]                SamplerState                        allSamplers[];
[[vk::binding(BindingPoints::eTexturesHdr, 0)]]             Sampler2D                           texturesHdr[];
[[vk::binding(BindingPoints::eTexturesCube, 0)]]            SamplerCube                         texturesCube[];

// Set 1 is SkyBruneton's runtime set -- the scattering LUTs and the atmosphere they were built
// from -- bound for aerial perspective alone. The path tracer binds the same set at index 3 of its
// own layout; a set number belongs to the pipeline, not to the resources, which is why the shared
// header takes them as arguments rather than declaring them.
[[vk::binding(0, 1)]] Sampler2D<float3>                       g_transmittance;
[[vk::binding(1, 1)]] Sampler3D<float4>                       g_scattering;
[[vk::binding(2, 1)]] Sampler3D<float4>                       g_singleMie;
[[vk::binding(3, 1)]] Sampler2D<float3>                       g_irradiance;
[[vk::binding(4, 1)]] ConstantBuffer<SkyAtmosphereParameters> g_atmosphere;

// Set 2 is the rasterizer's sun shadow map (Rasterizer::renderShadowMap): depth along the shadow
// light, a comparison sampler for the filter and a point sampler for the blocker search.
[[vk::binding(0, 2)]] Texture2D<float>        g_shadowMap;
[[vk::binding(1, 2)]] SamplerComparisonState  g_shadowCompare;
[[vk::binding(2, 2)]] SamplerState            g_shadowPoint;

// Set 3 is the scene's TLAS (a push descriptor), for the ray-traced shadows of the lights the
// shadow map does not cover. Pushed only when the scene has one; pushConst.rtShadowRays is 0 then.
[[vk::binding(0, 3)]] RaytracingAccelerationStructure g_tlas;
// clang-format on

#include "surface_opacity.h.slang"

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

// How much of the shadow light reaches `worldPos`: 1 lit, 0 in shadow.
//
// Contact-hardening soft shadows (PCSS). A blocker search finds the average occluder depth, the
// penumbra is (receiver - blocker) x tan(the light's angular radius) -- the same angular size the
// path tracer samples the light's disk with, so one softness matches between the two -- and a
// rotated Vogel-disk PCF filters over it. Without contact hardening the radius is fixed by the
// light's size at a typical occluder distance. Everything is in metres until the last step.
float sunShadow(float3 worldPos, float3 geoNormal, float3 toLight, float angularSize, float2 pixel)
{
  SceneFrameInfo* fi = pushConst.frameInfo;

  // Offset along the normal by how grazing the light is: what the depth bias alone cannot fix.
  const float  NoL      = dot(geoNormal, toLight);
  const float3 n        = NoL < 0.0 ? -geoNormal : geoNormal;
  const float  sinTheta = sqrt(saturate(1.0 - NoL * NoL));
  const float3 p        = worldPos + n * fi.shadowNormalBias * max(sinTheta, 0.2);

  const float4 clip = mul(float4(p, 1.0), fi.shadowViewProj);
  const float2 uv   = clip.xy * 0.5 + 0.5;
  if(any(uv < 0.0) || any(uv > 1.0) || clip.z <= 0.0)
    return 1.0;  // outside the box: nothing there casts
  // Past the far plane is beyond every caster: shadowed wherever the map has a blocker.
  const float receiver = min(clip.z, 1.0) - fi.shadowBias / fi.shadowDepthRange;

  const float tanHalf    = tan(0.5 * angularSize);
  const float uvPerMetre = 1.0 / (fi.shadowTexelWorld * float(fi.shadowMapSize));
  const float maxRadius  = 48.0 * fi.shadowTexelWorld;
  const float phi        = interleavedGradientNoise(pixel) * 6.28318531;

  float radius;
  if(fi.shadowContactHardening != 0)
  {
    // The widest penumbra any occluder could throw here comes from one at the near plane.
    const float search = clamp(receiver * fi.shadowDepthRange * tanHalf, 2.0 * fi.shadowMinRadius, maxRadius);
    float       sum    = 0.0;
    int         count  = 0;
    [unroll]
    for(int i = 0; i < 16; ++i)
    {
      const float d = g_shadowMap.SampleLevel(g_shadowPoint, uv + vogelDisk(i, 16, phi) * search * uvPerMetre, 0.0);
      if(d < receiver)
      {
        sum += d;
        ++count;
      }
    }
    if(count == 0)
      return 1.0;
    const float blocker = sum / float(count);
    radius              = (receiver - blocker) * fi.shadowDepthRange * tanHalf;
  }
  else
  {
    radius = 0.25 * fi.shadowDepthRange * tanHalf;
  }
  radius = clamp(radius, fi.shadowMinRadius, maxRadius);

  float lit = 0.0;
  [unroll]
  for(int i = 0; i < 32; ++i)
    lit += g_shadowMap.SampleCmpLevelZero(g_shadowCompare, uv + vogelDisk(i, 32, phi) * radius * uvPerMetre, receiver);
  return lit / 32.0;
}

// How much of a ray from `origin` along `dir` gets past the scene within `maxT`: 0 at an opaque hit,
// the product of (1 - opacity) over the alpha-tested ones (lash and hair cards, cutouts) otherwise,
// so their shadows are soft and need no noise. The path tracer's any-hit opacity (getOpacity).
float rayVisibility(float3 origin, float3 dir, float maxT)
{
  RayDesc ray;
  ray.Origin    = origin;
  ray.Direction = dir;
  ray.TMin      = 0.0;
  ray.TMax      = maxT;
  RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
  q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);
  float transmit = 1.0;
  while(q.Proceed())
  {
    if(q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
      continue;
    const float2 b       = q.CandidateTriangleBarycentrics();
    const float  opacity = getOpacity(pushConst.gltfScene->renderNodes[q.CandidateInstanceIndex()],
                                      pushConst.gltfScene->renderPrimitives[q.CandidateInstanceID()],
                                      int(q.CandidatePrimitiveIndex()), float3(1.0 - b.x - b.y, b.x, b.y));
    if(opacity >= 0.999)
      q.CommitNonOpaqueTriangleHit();
    else
      transmit *= 1.0 - opacity;
  }
  return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : transmit;
}

// A light's shadow traced against the scene: pushConst.rtShadowRays rays over the light's sphere
// (point, spot: GltfLight.radius) or the sun's disc (directional: its angular size), a Vogel disk
// turned per pixel, so the penumbra is the one the path tracer gives. From the hit's terminator-safe
// point, stepped off the surface on the light's side by the hit's own rounding bound.
float rayTracedShadow(int lightIndex, GltfLight light, HitState hit, float3 geoNormal, float2 pixel)
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
  const int   rays = pushConst.rtShadowRays;
  const float phi  = interleavedGradientNoise(pixel + float2(float(lightIndex) * 7.0, 0.0)) * 6.28318531;
  float       lit  = 0.0;
  for(int i = 0; i < rays; ++i)
  {
    const float2 o = vogelDisk(i, rays, phi) * tanHalf;
    lit += rayVisibility(origin, normalize(toLight + o.x * t + o.y * b), maxT);
  }
  return lit / float(rays);
}

// The shadow term for light `lightIndex` of the raster light loop: the shadow map's for the shadow
// light, traced for every other light when the scene has a TLAS, else 1.
float lightShadow(int lightIndex, GltfLight light, HitState hit, float3 geoNormal, float2 pixel)
{
  if(lightIndex == pushConst.frameInfo.shadowLightIndex && light.type == LightType::eLightTypeDirectional)
    return sunShadow(hit.pos, geoNormal, -light.direction, light.angularSizeOrInvRange, pixel);
  if(pushConst.rtShadowRays <= 0)
    return 1.0;
  return rayTracedShadow(lightIndex, light, hit, geoNormal, pixel);
}

// How open the hemisphere over `N` is within `radius`: pushConst.aoRays cosine-distributed rays on
// a Vogel disk turned per pixel, alpha-tested hits letting (1 - opacity) through. 1 without a TLAS.
float rayTracedOcclusion(HitState hit, float3 N, float radius, float2 pixel)
{
  const int rays = pushConst.aoRays;
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
    // a direction under the geometry (a normal-mapped N) sees the surface itself
    open += dot(dir, n) > 0.0 ? rayVisibility(origin, dir, radius) : 0.0;
  }
  return open / float(rays);
}

// What occlusion `ao` leaves of a diffuse lobe of `albedo` once the light it blocks bounces back off
// the nearby surfaces (Jimenez et al. 2016, "Practical Real-Time Strategies for Accurate Indirect
// Occlusion"; Unreal's AOMultiBounce).
float3 aoMultiBounce(float ao, float3 albedo)
{
  const float3 a = 2.0404 * albedo - 0.3324;
  const float3 b = -4.7951 * albedo + 0.6417;
  const float3 c = 2.7552 * albedo + 0.6903;
  return max(float3(ao), ((ao * a + b) * ao + c) * ao);
}

// The share of a specular lobe's environment light that occlusion `ao` leaves, at the view's angle
// and the lobe's GGX alpha (Lagarde and de Rousiers 2014, "Moving Frostbite to PBR").
float specularOcclusion(float NoV, float ao, float alpha)
{
  return saturate(pow(NoV + ao, exp2(-16.0 * alpha - 1.0)) - 1.0 + ao);
}

// How far the traced ambient occlusion reaches, in metres
static const float AO_RADIUS_SURFACE = 0.3;  // the figure: folds, the chin, under the arms
static const float AO_RADIUS_FLOOR   = 1.0;  // the floor: the figure's contact shadow


// Define the vertex attributes that will be provided by the vertex buffer
// Note: In Slang, we use semantic annotations (: POSITION, etc.) to specify
// how these fields map to vertex buffer data
struct VertexInput
{
  [[vk::binding(0)]]
  float3 position : POSITION;  // Vertex position in model space
};

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
//
// currClipUnjit/prevClipUnjit are clip-space positions WITHOUT the DLSS sub-pixel jitter.
// They are used in the fragment shader to compute screen-space motion vectors that the
// DLSS network expects (NGX wants un-jittered MVs and de-jitters them itself via
// InJitterOffsetX/Y). When DLAA is disabled `position` and `currClipUnjit` are identical
// because frameInfo.jitter defaults to (0,0). The unjittered fields are only emitted in
// builds that ship the motion attachment (USE_DLSS=ON); USE_DLSS=OFF builds drop them so
// Slang's DCE removes the prevMVP multiply and the extra interpolators.
struct VertexOutput
{
  float4 position : SV_Position;  // Clip space position (jittered when DLAA on) - required
  float3 worldPos;
#ifdef HAS_DLSS_MOTION
  float4 currClipUnjit;  // Current-frame unjittered clip-space position
  float4 prevClipUnjit;  // Previous-frame unjittered clip-space position
#endif
};

// Define the final output of the fragment shader. Color attachments:
//   0 = shaded color (or DLAA input when enabled)
//   1 = render-node ID for silhouette/picking
//   2 = screen-space motion vector ([prev - curr] in normalized 0..1 screen UV) for DLAA -
//       ONLY in USE_DLSS=ON builds. USE_DLSS=OFF binaries drop this attachment entirely so
//       the rasterizer pipeline has 2 color outputs and the rasterizer's Dlss (Kind::SR)
//       instance never allocates a motion image. HAS_DLSS_MOTION is passed by CMake when USE_DLSS=ON.
struct PixelOutput
{
  float4 color : SV_TARGET0;
  float4 selection : SV_TARGET1;
#ifdef HAS_DLSS_MOTION
  float2 motion : SV_TARGET2;
#endif
};


// ---------- Screen-space transmission (KHR_materials_transmission, ibl.glsl:58-114) ----------
//
// `texturesHdr[HDR_OPAQUE_INDEX]` was filled by the rasterizer's first pass with the resolved
// opaque image downsized to OPAQUE_COLOR_SIZE x OPAQUE_COLOR_SIZE plus a full mip chain.
// Roughness drives the mip LOD via `applyIorToRoughness(roughness, ior) * log2(size)`, the same
// mapping the Khronos reference uses.

float3 getTransmissionSample(float2 uv, float roughness, float ior)
{
  float framebufferLod = log2(float(OPAQUE_COLOR_SIZE)) * applyIorToRoughness(roughness, ior);
  return texturesHdr[HDR_OPAQUE_INDEX].SampleLevel(uv, framebufferLod).rgb;
}

#if GLTF_USE_RETROREFLECTION
// KHR_materials_retroreflection excludes additional lobes -- transmission, clearcoat, and sheen --
// from the mirrored-V blend: they see the real V and are added back untouched (see the
// `cf - cfBrdf` isolation below). Anisotropy and iridescence stay in, since they're direction-
// dependent modifiers of the base metal/dielectric lobe and must see the same (mirrored) V that
// lobe uses (per Khronos spec author confirmation, github.com/KhronosGroup/glTF/pull/2610).
PbrMaterial disableExcludedRetroLobes(PbrMaterial mat)
{
  mat.transmission              = 0.0f;
  mat.diffuseTransmissionFactor = 0.0f;
  mat.clearcoat                 = 0.0f;
  mat.sheenColor                = float3(0.0f);
  return mat;
}

// MRM (Portsmouth et al. 2026): blend reflection lobes only; keep regular BTDF (Portsmouth Sec. 5).
float3 bsdfEvalPunctualKhronosMRM(PbrMaterial mat, float3 V, float3 pointToLight, float3 lightIntensity, float3 transmissionRay, float dttThickness, Sampler2D sheenLUT)
{
  float3 cf = bsdfEvaluatePunctualKhronos(mat, V, pointToLight, lightIntensity, transmissionRay, dttThickness, sheenLUT);
  if(mat.retroreflection <= 0.0f)
    return cf;

  PbrMaterial matBrdf = disableExcludedRetroLobes(mat);
  float3 cfBrdf = bsdfEvaluatePunctualKhronos(matBrdf, V, pointToLight, lightIntensity, transmissionRay, dttThickness, sheenLUT);

  // V is outgoing (toward camera); MRM uses the incident view ray: reflect(-V, N).
  float3 Vp = reflect(-V, mat.N);
  float3 crBrdf = bsdfEvaluatePunctualKhronos(matBrdf, Vp, pointToLight, lightIntensity, transmissionRay, dttThickness, sheenLUT);
  return (cf - cfBrdf) + lerp(cfBrdf, crBrdf, mat.retroreflection);
}

float3 evalKhronosIBLMRM(PbrMaterial mat,
                         float3      V,
                         SamplerCube diffuseCube,
                         SamplerCube glossyCube,
                         Sampler2D   lut,
                         Sampler2D   sheenLUT,
                         float4      envRotation,
                         float       envIntensity,
                         float       dttThickness,
                         float3      f_specular_transmission,
                         bool        haveTransmissionSample,
                         float3      diffuseOcclusion,
                         float       specularOcclusion)
{
  float3 cf = evalKhronosIBL(mat, V, diffuseCube, glossyCube, lut, sheenLUT, envRotation, envIntensity, dttThickness,
                             f_specular_transmission, haveTransmissionSample, diffuseOcclusion, specularOcclusion);
  if(mat.retroreflection <= 0.0f)
    return cf;

  PbrMaterial matBrdf = disableExcludedRetroLobes(mat);
  float3      cfBrdf  = evalKhronosIBL(matBrdf, V, diffuseCube, glossyCube, lut, sheenLUT, envRotation, envIntensity,
                                       dttThickness, float3(0.0f), false, diffuseOcclusion, specularOcclusion);

  float3 Vp     = reflect(-V, mat.N);
  float3 crBrdf = evalKhronosIBL(matBrdf, Vp, diffuseCube, glossyCube, lut, sheenLUT, envRotation, envIntensity,
                                 dttThickness, float3(0.0f), false, diffuseOcclusion, specularOcclusion);
  return (cf - cfBrdf) + lerp(cfBrdf, crBrdf, mat.retroreflection);
}
#endif

float3 getIBLVolumeRefraction(float3   n,
                              float3   v,
                              float    perceptualRoughness,
                              float3   baseColor,
                              float3   worldPos,
                              float3x3 modelMatrix3,
                              float    ior,
                              float    thickness,
                              float3   attenuationColor,
                              float    attenuationDistance,
                              float    dispersion)
{
  // Khronos uses a 4x4 model matrix; we only need column lengths for thickness scaling. We pass
  // a mat3 of the upper-left of objectToWorld which is sufficient.
  // Emulate getVolumeTransmissionRay() inline using mat3.
  // (Keeps signatures simple and avoids passing full mat4.)

  float3 modelScale;
  modelScale.x = length(modelMatrix3[0]);
  modelScale.y = length(modelMatrix3[1]);
  modelScale.z = length(modelMatrix3[2]);

  float3 transmittedLight;
  float  transmissionRayLength = 0.0;

  if(dispersion > 0.0)
  {
    // Per-channel IOR for chromatic dispersion (ibl.glsl:73-95).
    float halfSpread = (ior - 1.0) * 0.025 * dispersion;
    float iors[3]    = { ior - halfSpread, ior, ior + halfSpread };

    [unroll]
    for(int i = 0; i < 3; i++)
    {
      float3 refractionVector = refract(-v, normalize(n), 1.0 / iors[i]);
      float3 transmissionRay  = normalize(refractionVector) * thickness * modelScale;
      transmissionRayLength   = length(transmissionRay);  // overwritten by blue ray; matches reference
      float3 refractedRayExit = worldPos + transmissionRay;

      float4 ndcPos           = mul(float4(refractedRayExit, 1.0), pushConst.frameInfo.viewProjMatrix);
      float2 refractionCoords = ndcPos.xy / max(abs(ndcPos.w), 1e-6);
      refractionCoords        = refractionCoords * 0.5 + 0.5;

      transmittedLight[i] = getTransmissionSample(refractionCoords, perceptualRoughness, iors[i])[i];
    }
  }
  else
  {
    float3 refractionVector = refract(-v, normalize(n), 1.0 / ior);
    float3 transmissionRay  = normalize(refractionVector) * thickness * modelScale;
    transmissionRayLength   = length(transmissionRay);
    float3 refractedRayExit = worldPos + transmissionRay;

    float4 ndcPos           = mul(float4(refractedRayExit, 1.0), pushConst.frameInfo.viewProjMatrix);
    float2 refractionCoords = ndcPos.xy / max(abs(ndcPos.w), 1e-6);
    refractionCoords        = refractionCoords * 0.5 + 0.5;

    transmittedLight = getTransmissionSample(refractionCoords, perceptualRoughness, ior);
  }

  float3 attenuatedColor = applyVolumeAttenuation(transmittedLight, transmissionRayLength, attenuationColor, attenuationDistance);
  return attenuatedColor * baseColor;
}

//------------------------------------------------------------------------------
// Vertex Shader
//
// SV_Position is the UN-JITTERED clip-space position with a post-projection NDC
// shift applied from frameInfo.jitter so the rasterizer samples sub-pixel offsets
// for DLAA. The shift mirrors the standard "translate viewProj in clip-space"
// trick (`clip.xy -= jitter * 2/imageSize * clip.w`)
//
// NOTE: per-object animation/skinning is NOT yet propagated into prevMVP, so
// MVs only reflect camera motion. Animated meshes will still antialias correctly
// from the screen-space MV at the camera level; deferring object-level MVs.
//------------------------------------------------------------------------------
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
  GltfRenderNode renderNode = pushConst.gltfScene.renderNodes[pushConst.renderNodeID];

  float3 pos = mul(float4(input.position, 1.0), renderNode.objectToWorld).xyz;

  VertexOutput output;
  output.worldPos = pos;

#ifdef HAS_DLSS_MOTION
  // DLSS path: emit unjittered clip-space positions for fragment-side MV calc, then
  // post-multiply by the per-frame jitter shift to produce the rasterizer's sample positions.
  output.currClipUnjit = mul(float4(pos, 1.0), pushConst.frameInfo.viewProjMatrix);
  output.prevClipUnjit = mul(float4(pos, 1.0), pushConst.frameInfo.prevMVP);

  // Apply DLAA sub-pixel jitter as a post-projection clip-space shift. Equivalent to multiplying
  // by `translate(-jitter*(2/imgSize, 2/imgSize, 0)) * viewProj` on the host. When DLAA is
  // disabled at runtime the host writes jitter=(0,0) so output.position == currClipUnjit.
  float4 jittered = output.currClipUnjit;
  jittered.xy -= pushConst.frameInfo.jitter * (2.0 / pushConst.frameInfo.imageSize) * jittered.w;
  output.position = jittered;
#else
  // Non-DLSS build: no jitter, no MV - the standard rasterizer clip-space position is enough.
  output.position = mul(float4(pos, 1.0), pushConst.frameInfo.viewProjMatrix);
#endif

  return output;
}

//------------------------------------------------------------------------------
// Fragment/Pixel Shader
//------------------------------------------------------------------------------
// Pixel-space motion vector shared with the path tracer (dlss_util.h::calculateMotionVector).
// We pass UN-JITTERED clip-space positions before the perspective divide (interpolated across
// the triangle by the rasterizer) so the projective interpolation is correct, then the helper
// performs the /w per fragment. NGX's InMVScaleX/Y stays at 1.0, matching DLSS-RR.
[shader("fragment")]
PixelOutput fragmentMain(VertexOutput input, uint primitiveID: SV_PrimitiveID, float3 baryWeights: SV_Barycentrics)
{
  PixelOutput output;
  output.color.a = 1.0;
#ifdef HAS_DLSS_MOTION
  output.motion = calculateMotionVector(input.currClipUnjit, input.prevClipUnjit, pushConst.frameInfo.imageSize);
#endif

  // Setting up scene info
  GltfShadeMaterial material   = pushConst.gltfScene->materials[pushConst.materialID];      // Buffer of materials
  GltfRenderNode    renderNode = pushConst.gltfScene->renderNodes[pushConst.renderNodeID];  // Buffer of render nodes
  GltfRenderPrimitive renderPrimitive = pushConst.gltfScene->renderPrimitives[pushConst.renderPrimID];  // Buffer of meshes

  float3 cameraPos =
      float3(pushConst.frameInfo.viewInv[3].x, pushConst.frameInfo.viewInv[3].y, pushConst.frameInfo.viewInv[3].z);
  float3 worldRayDirection;
  if(hasFlag(pushConst.frameInfo.flags, SceneFrameInfoFlags::eSceneIsOrthographic))
  {
    worldRayDirection = normalize(mul(float4(0, 0, -1, 0), pushConst.frameInfo.viewInv).xyz);
  }
  else
  {
    worldRayDirection = normalize(input.worldPos - cameraPos);
  }

  HitState hit = getHitState(renderPrimitive, baryWeights, renderNode.worldToObject, renderNode.objectToWorld,
                             primitiveID, worldRayDirection);

  // Evaluate the material at the hit point
  MeshState mesh          = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, false, 0.0f);
  mesh.baseColorVertexMul = hit.color;
  PbrMaterial pbrMat      = evaluateMaterial(material, mesh, allImages, allSamplers, pushConst.gltfScene->textureInfos);

  output.color.xyz = pbrMat.baseColor;
  output.color.a   = pbrMat.opacity * (1.0 - pbrMat.transmission);

  // Write ObjectID for silhouette (0 = no hit, N+1 = render node N); silhouette uses bitmask for selection
  output.selection = float4(asfloat(pushConst.renderNodeID + 1), 0, 0, 0);


  if(pushConst.frameInfo.visualization != Visualization::eRendered)
  {
    float3              vizColor;
    VisualizationResult vizResult =
        applyVisualization(pbrMat, hit, pushConst.frameInfo.visualization, pushConst.renderPrimID, primitiveID, -1, vizColor);
    if(vizResult == VisualizationResult::eColorOverride)
    {
      output.color.xyz = vizColor;
      output.color.a   = 1.0;
      return output;
    }
  }

  if(material.alphaMode == AlphaMode::eAlphaModeMask)
  {
    if(pbrMat.opacity < material.alphaCutoff)
      discard;
  }

#if GLTF_USE_UNLIT
  // KHR_materials_unlit: skip lighting entirely. Match Khronos pbr.frag (color = baseColor + emissive).
  if(material.unlit > 0)
  {
    float finalA = (material.alphaMode == AlphaMode::eAlphaModeBlend) ? pbrMat.opacity : 1.0;
    output.color = float4(pbrMat.baseColor + pbrMat.emissive, finalA);
    return output;
  }
#endif

  const float3 V = -worldRayDirection;

  // Result
  float3 contribution = float3(0);

  // Effective IOR for transmission ray and screen-space refraction (path tracer uses ior2/ior1
  // already; we mirror that to keep raster geometry identical).
  float effectiveIor        = pbrMat.ior2 / max(pbrMat.ior1, 1e-5);
  float perceptualRoughness = sqrt(pbrMat.roughness.y);

  // Model scale (length of each objectToWorld column) is needed by the volume transmission ray
  // and by the diffuse-transmission thickness scaling. Compute once.
  float3 modelScale = float3(length(renderNode.objectToWorld[0].xyz), length(renderNode.objectToWorld[1].xyz),
                             length(renderNode.objectToWorld[2].xyz));

  // World-space thickness for KHR_materials_diffuse_transmission's volume attenuation
  // (`pbr.frag:145`). Hoisted here so it's available to the punctual evaluator regardless of
  // the env mode.
  float dttThickness = (pbrMat.thickness > 0.0) ? (pbrMat.thickness * (modelScale.x + modelScale.y + modelScale.z) / 3.0) : 0.0;

  // Pre-compute the volume transmission ray (object-space refraction direction * thickness *
  // model scale). Used by both the screen-space sample and the punctual BTDF.
  float3 transmissionRay = float3(0.0);
#if GLTF_USE_TRANSMISSION
  if(pbrMat.transmission > 0.0 && pbrMat.thickness > 0.0)
  {
    float3 refractionVector = refract(-V, normalize(pbrMat.N), 1.0 / max(effectiveIor, 1e-5));
    transmissionRay         = normalize(refractionVector) * pbrMat.thickness * modelScale;
  }
#endif

  // Screen-space transmitted radiance (KHR_materials_transmission) computed once for both
  // env-mode branches when the opaque-color framebuffer was captured this frame.
  bool   haveTransmissionSample  = false;
  float3 f_specular_transmission = float3(0.0);
#if GLTF_USE_TRANSMISSION
  if(pbrMat.transmission > 0.0 && pushConst.opaqueColorReady != 0)
  {
    float3x3 m3 = float3x3(renderNode.objectToWorld[0].xyz, renderNode.objectToWorld[1].xyz, renderNode.objectToWorld[2].xyz);
    f_specular_transmission =
        getIBLVolumeRefraction(pbrMat.N, V, perceptualRoughness, pbrMat.baseColor, hit.pos, m3, effectiveIor,
                               pbrMat.thickness, pbrMat.attenuationColor, pbrMat.attenuationDistance, pbrMat.dispersion);
    haveTransmissionSample = true;
  }
#endif

  // Indirect lighting. Every sky type goes through the same prefiltered cubemaps: the analytic
  // skies are baked into them by EnvBaker and a loaded HDR fills them on load. This used to branch, with a hardcoded hemispherical ambient
  // plus a punctual sun standing in whenever the environment was not an HDR -- which meant raster
  // sky lighting never matched the path tracer, for any sky. There is now one path and no
  // stand-in.
  if(hasFlag(pushConst.frameInfo.flags, SceneFrameInfoFlags::eSceneUseNoEnvironment))
  {
    // No environment: no indirect lighting. Only punctual/emissive scene lights contribute.
    // Screen-space transmission is unrelated to the environment (it refracts whatever is behind
    // the surface), so transmissive materials still compose it — matching the Sky/HDR branches.
#if GLTF_USE_TRANSMISSION
    if(haveTransmissionSample)
      contribution = lerp(contribution, f_specular_transmission, pbrMat.transmission);
#endif
  }
  else
  {
    // Occlusion per lobe: the material's map times the traced occlusion, multi-bounce on the
    // diffuse lobes and specular occlusion on the glossy ones (Unreal's AOMultiBounce and
    // GetSpecularOcclusion), instead of one factor over the whole IBL.
    const float  ao        = pbrMat.occlusion * rayTracedOcclusion(hit, pbrMat.N, AO_RADIUS_SURFACE, input.position.xy);
    const float3 diffuseAO = aoMultiBounce(ao, pbrMat.baseColor);
    const float  specAO    = specularOcclusion(saturate(dot(pbrMat.N, V)), ao, pbrMat.roughness.y);

    // Single-call Khronos split-sum evaluator (matches `pbr.frag:162-228`).
    // Handles diffuse, specular (isotropic/anisotropic), sheen, clearcoat, iridescence,
    // diffuse-transmission volume attenuation, and optional screen-space transmission mixing.
#if GLTF_USE_RETROREFLECTION
    contribution += evalKhronosIBLMRM(pbrMat, V, texturesCube[HDR_DIFFUSE_INDEX], texturesCube[HDR_GLOSSY_INDEX],
                                      texturesHdr[HDR_LUT_INDEX], texturesHdr[HDR_SHEEN_INDEX],
                                      pushConst.frameInfo.envRotation, pushConst.frameInfo.envIntensity, dttThickness,
                                      f_specular_transmission, haveTransmissionSample, diffuseAO, specAO);
#else
    contribution += evalKhronosIBL(pbrMat, V, texturesCube[HDR_DIFFUSE_INDEX], texturesCube[HDR_GLOSSY_INDEX],
                                   texturesHdr[HDR_LUT_INDEX], texturesHdr[HDR_SHEEN_INDEX], pushConst.frameInfo.envRotation,
                                   pushConst.frameInfo.envIntensity, dttThickness, f_specular_transmission,
                                   haveTransmissionSample, diffuseAO, specAO);
#endif

    // KHR_materials_scatter needs no separate term here: evaluateMaterial() maps it onto the
    // diffuse-transmission / diffuse-reflection lobes for the raster pipeline (the spec's
    // thin-walled approximation of volumetric scattering), so the IBL composite above already
    // accounts for it.
  }

  // Ambient occlusion is applied to the indirect lighting only (Khronos pbr.frag), per lobe in the
  // environment branch above; with no environment, to the screen-space transmission alone.
  if(hasFlag(pushConst.frameInfo.flags, SceneFrameInfoFlags::eSceneUseNoEnvironment))
    contribution *= pbrMat.occlusion;

  // Emissive under the clearcoat layer (`pbr.frag:373`).
  contribution += pbrMat.emissive * (float3(1.0) - pbrMat.clearcoat * khronosClearcoatFresnel(pbrMat, V));

  // Punctual lights — Khronos-faithful direct lighting: spec-correct KHR_lights_punctual
  // attenuation via `singleLightContribution`, then per-light layered BSDF via
  // `bsdfEvaluatePunctualKhronos` (VdotH Schlick + height-correlated Smith + Khronos composite).
  // Intensity is computed once per light at the front-surface position; only the BTDF lobe gets
  // the transmission-ray shift applied internally (intentional fix vs. `pbr.frag:316`).
  // One past the scene's lights when the sky supplies its own sun, matching the path tracer's
  // punctualLightCount(); getSkySunLightRaster() returns it for that last index.
  const int rasterLightCount = pushConst.gltfScene.numLights + (pushConst.frameInfo->rendererSunLight != 0 ? 1 : 0);
  for(int i = 0; i < rasterLightCount; i++)
  {
    GltfLight light =
        (i >= pushConst.gltfScene.numLights) ? makeSkySunLight(pushConst.frameInfo) : pushConst.gltfScene.lights[i];
    applySkySunLight(light, i, pushConst.frameInfo);  // no-op unless this is the marked sun
    LightContrib contrib = singleLightContribution(light, hit.pos, pbrMat.N);
    contrib.intensity *= evalIesFactor(pushConst.gltfScene, light, hit.pos);  // EXT_lights_ies
    if(all(contrib.intensity == 0.0))
      continue;
    contrib.intensity *= lightShadow(i, light, hit, hit.geonrm, input.position.xy);
    if(all(contrib.intensity == 0.0))
      continue;

    // contrib.incidentVector points from the light toward the surface (= propagation direction).
    // The evaluator wants the unnormalized surface-to-light vector (length matters for the
    // BTDF transmission-ray shift to match `pbr.frag:303`).
    float3 pointToLight;
    if(light.type == LightType::eLightTypeDirectional)
      pointToLight = -contrib.incidentVector;
    else
      pointToLight = -contrib.incidentVector * contrib.distance;

#if GLTF_USE_RETROREFLECTION
    contribution += bsdfEvalPunctualKhronosMRM(pbrMat, V, pointToLight, contrib.intensity, transmissionRay,
                                               dttThickness, texturesHdr[HDR_SHEEN_INDEX]);
#else
    contribution += bsdfEvaluatePunctualKhronos(pbrMat, V, pointToLight, contrib.intensity, transmissionRay,
                                                dttThickness, texturesHdr[HDR_SHEEN_INDEX]);
#endif
  }


  // Final alpha. Khronos pbr.frag writes a=1.0 for opaque/mask and `baseColor.a` for blend, and
  // relies on the screen-space transmission sample to make transmissive surfaces see-through.
  //
  // When the transmission framebuffer sample IS available (`haveTransmissionSample`), the
  // fragment color already contains the refracted backdrop, so alpha stays at the authored
  // opacity -- reducing it by `(1 - transmission)` would multiply SRC_ALPHA=0 and the surface
  // would vanish entirely (SRC_ALPHA blending: `0*src + 1*dst` = background kept as-is).
  //
  // When we do NOT have a sample (sky mode with no capture, or transmissive material in a
  // scene that never triggered the opaque-color pass), we fall back to alpha-based see-through
  // so the surface still lets the background show by reducing alpha by transmission.
  float finalAlpha = 1.0;
  if(material.alphaMode == AlphaMode::eAlphaModeBlend)
    finalAlpha = pbrMat.opacity;
#if GLTF_USE_TRANSMISSION
  if(pbrMat.transmission > 0.0 && !haveTransmissionSample)
    finalAlpha *= (1.0 - pbrMat.transmission);
#endif

  // Aerial perspective: the air between the eye and this fragment. The same function the path
  // tracer calls, on the same inputs, so the two paths' haze cannot drift. Unshadowed -- the
  // rasterizer has no acceleration structure, so it cannot ask whether the sun is visible from a
  // point in the air; light shafts are a path-tracer capability and a phase of their own.
  if(hasAerialPerspective(pushConst.frameInfo))
  {
    const float3 eye =
        float3(pushConst.frameInfo.viewInv[3].x, pushConst.frameInfo.viewInv[3].y, pushConst.frameInfo.viewInv[3].z);
    contribution = applyAerialPerspective(contribution, hit.pos, eye, pushConst.frameInfo, g_atmosphere, g_transmittance, g_irradiance);
  }

  output.color = float4(contribution, finalAlpha);


  return output;
}

//------------------------------------------------------------------------------
// Wireframe Shader
// Return a constant color for wireframe rendering. Selection/motion are zero - the
// wireframe pass is a visual debug overlay and doesn't participate in DLAA.
[shader("fragment")]
PixelOutput fragmentWireframeMain(VertexOutput input)
{
  PixelOutput output;
  output.color     = float4(0.4, 0.01, 0.01, 1);
  output.selection = float4(0.0);
#ifdef HAS_DLSS_MOTION
  output.motion = float2(0.0);
#endif
  return output;
}

//------------------------------------------------------------------------------
// Sun shadow map (Rasterizer::renderShadowMap): depth along the shadow light.
//------------------------------------------------------------------------------
struct ShadowVertexOutput
{
  float4 position : SV_Position;
};

[shader("vertex")]
ShadowVertexOutput vertexShadowMain(VertexInput input)
{
  GltfRenderNode     renderNode = pushConst.gltfScene.renderNodes[pushConst.renderNodeID];
  const float3       pos        = mul(float4(input.position, 1.0), renderNode.objectToWorld).xyz;
  ShadowVertexOutput output;
  output.position = mul(float4(pos, 1.0), pushConst.frameInfo.shadowViewProj);
  return output;
}

// Opaque surfaces only write depth. A mask material is cut at its cutoff, as it is drawn; a blended
// one is hashed against its opacity, so hair cards and lashes cast the cutout shadow their texture
// says, and a nearly clear surface (an eye's moisture layer) casts almost none.
[shader("fragment")]
void fragmentShadowMain(ShadowVertexOutput input, uint primitiveID: SV_PrimitiveID, float3 baryWeights: SV_Barycentrics)
{
  GltfShadeMaterial material = pushConst.gltfScene->materials[pushConst.materialID];
  if(material.alphaMode == AlphaMode::eAlphaModeOpaque)
    return;

  GltfRenderNode      renderNode      = pushConst.gltfScene->renderNodes[pushConst.renderNodeID];
  GltfRenderPrimitive renderPrimitive = pushConst.gltfScene->renderPrimitives[pushConst.renderPrimID];
  const float3        rayDir          = normalize(float3(0.0, -1.0, 0.0));
  HitState hit = getHitState(renderPrimitive, baryWeights, renderNode.worldToObject, renderNode.objectToWorld, primitiveID, rayDir);

  MeshState mesh          = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, false, 0.0f);
  mesh.baseColorVertexMul = hit.color;
  PbrMaterial pbrMat      = evaluateMaterial(material, mesh, allImages, allSamplers, pushConst.gltfScene->textureInfos);

  if(material.alphaMode == AlphaMode::eAlphaModeMask)
  {
    if(pbrMat.opacity < material.alphaCutoff)
      discard;
    return;
  }
  const float coverage = pbrMat.opacity * (1.0 - pbrMat.transmission);
  if(coverage <= interleavedGradientNoise(input.position.xy))
    discard;
}

//------------------------------------------------------------------------------
// Ground (Rasterizer::renderGround): the infinite plane, as one fullscreen triangle.
//------------------------------------------------------------------------------
struct GroundVertexOutput
{
  float4 position : SV_Position;
};

[shader("vertex")]
GroundVertexOutput vertexGroundMain(uint vertexId: SV_VertexID)
{
  const float2       uv = float2(float((vertexId << 1) & 2), float(vertexId & 2));
  GroundVertexOutput output;
  output.position = float4(uv * 2.0 - 1.0, 0.5, 1.0);
  return output;
}

struct GroundOutput
{
  float4 color : SV_TARGET0;
  float4 selection : SV_TARGET1;
#ifdef HAS_DLSS_MOTION
  float2 motion : SV_TARGET2;
#endif
  float depth : SV_Depth;
};

// A world-space point on the pixel's camera ray at NDC depth `z`.
float3 unprojectPixel(float2 ndc, float z)
{
  float4 v = mul(float4(ndc, z, 1.0), pushConst.frameInfo.projInv);
  v /= v.w;
  return mul(float4(v.xyz, 1.0), pushConst.frameInfo.viewInv).xyz;
}

// Intersect the pixel's ray with the plane and draw it there, with its own depth, so the scene
// occludes it and it occludes what is under it. A floor is lit as the path tracer lights its plane
// -- image-based light plus the punctual lights, the shadow light through the shadow map -- and
// fades out at the edge of its disk; a shadow catcher draws nothing but the darkening it catches.
[shader("fragment")]
GroundOutput fragmentGroundMain(GroundVertexOutput input)
{
  SceneFrameInfo* fi = pushConst.frameInfo;
  GroundOutput    output;
  output.selection = float4(0.0);

  // The scene is drawn jittered (see vertexMain); the ray is the unjittered pixel's.
  const float2 ndc = (input.position.xy / fi.imageSize) * 2.0 - 1.0 + fi.jitter * (2.0 / fi.imageSize);

  // Two points on the ray, at depths every projection here maps to finite points, then the ray
  // through them oriented away from the camera and started at the eye's plane.
  const float3 a       = unprojectPixel(ndc, 0.25);
  const float3 b       = unprojectPixel(ndc, 0.75);
  const float3 eye     = float3(fi.viewInv[3].x, fi.viewInv[3].y, fi.viewInv[3].z);
  const float3 forward = normalize(mul(float4(0.0, 0.0, -1.0, 0.0), fi.viewInv).xyz);
  float3       dir     = normalize(b - a);
  if(dot(dir, forward) < 0.0)
    dir = -dir;
  const float3 origin = a - dir * dot(a - eye, dir);

  const float planeY = fi.infinitePlaneDistance;
  if(origin.y <= planeY || dir.y >= -1e-6)
    discard;  // below the plane, or looking at or above the horizon
  const float  t = (planeY - origin.y) / dir.y;
  const float3 P = origin + dir * t;

  const float fade = groundFadeWeight(P, fi);
  if(fade <= 0.0)
    discard;

  const float4 clip = mul(float4(P, 1.0), fi.viewProjMatrix);
  output.depth      = clip.z / clip.w;
#ifdef HAS_DLSS_MOTION
  output.motion = calculateMotionVector(clip, mul(float4(P, 1.0), fi.prevMVP), fi.imageSize);
#endif

  const float3 up = float3(0.0, 1.0, 0.0);
  const float3 V  = -dir;

  const int lightCount = pushConst.gltfScene.numLights + (fi.rendererSunLight != 0 ? 1 : 0);

  if(hasFlag(fi.flags, SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher))
  {
    float shadow = 1.0;
    const int i  = fi.shadowLightIndex;
    if(i >= 0 && i < lightCount)
    {
      GltfLight light = (i >= pushConst.gltfScene.numLights) ? makeSkySunLight(fi) : pushConst.gltfScene.lights[i];
      applySkySunLight(light, i, fi);
      if(dot(up, -light.direction) > 0.0 && any(light.color * light.intensity > 0.0))
        shadow = sunShadow(P, up, -light.direction, light.angularSizeOrInvRange, input.position.xy);
    }
    const float alpha = fade * groundCatcherStrength(fi) * (1.0 - shadow);
    if(alpha <= 1.0 / 512.0)
      discard;
    output.color = float4(0.0, 0.0, 0.0, alpha);
    return output;
  }

  PbrMaterial mat = defaultPbrMaterial(fi.infinitePlaneBaseColor, fi.infinitePlaneMetallic, fi.infinitePlaneRoughness, up, up);

  float3 contribution = float3(0.0);
  if(!hasFlag(fi.flags, SceneFrameInfoFlags::eSceneUseNoEnvironment))
  {
    // the figure's contact shadow: the floor's environment light occluded within a metre
    HitState floorHit;
    floorHit.pos            = P;
    floorHit.nrm            = up;
    floorHit.geonrm         = up;
    floorHit.shadowPos      = P;
    floorHit.offsetDistance = 1e-6 * max(1.0, max(abs(P.x), max(abs(P.y), abs(P.z))));
    const float ao          = rayTracedOcclusion(floorHit, up, AO_RADIUS_FLOOR, input.position.xy);
    contribution += evalKhronosIBL(mat, V, texturesCube[HDR_DIFFUSE_INDEX], texturesCube[HDR_GLOSSY_INDEX],
                                   texturesHdr[HDR_LUT_INDEX], texturesHdr[HDR_SHEEN_INDEX], fi.envRotation,
                                   fi.envIntensity, 0.0, float3(0.0), false, aoMultiBounce(ao, mat.baseColor),
                                   specularOcclusion(saturate(dot(up, V)), ao, mat.roughness.y));
  }

  // the floor as a hit for the traced shadows: flat, its rounding bound a float's at that distance
  HitState groundHit;
  groundHit.pos            = P;
  groundHit.nrm            = up;
  groundHit.geonrm         = up;
  groundHit.shadowPos      = P;
  groundHit.offsetDistance = 1e-6 * max(1.0, max(abs(P.x), max(abs(P.y), abs(P.z))));
  for(int i = 0; i < lightCount; i++)
  {
    GltfLight light = (i >= pushConst.gltfScene.numLights) ? makeSkySunLight(fi) : pushConst.gltfScene.lights[i];
    applySkySunLight(light, i, fi);
    LightContrib contrib = singleLightContribution(light, P, up);
    contrib.intensity *= evalIesFactor(pushConst.gltfScene, light, P);
    if(all(contrib.intensity == 0.0))
      continue;
    contrib.intensity *= lightShadow(i, light, groundHit, up, input.position.xy);
    if(all(contrib.intensity == 0.0))
      continue;
    const float3 pointToLight =
        (light.type == LightType::eLightTypeDirectional) ? -contrib.incidentVector : -contrib.incidentVector * contrib.distance;
    contribution += bsdfEvaluatePunctualKhronos(mat, V, pointToLight, contrib.intensity, float3(0.0), 0.0, texturesHdr[HDR_SHEEN_INDEX]);
  }

  if(hasAerialPerspective(fi))
    contribution = applyAerialPerspective(contribution, P, eye, fi, g_atmosphere, g_transmittance, g_irradiance);

  output.color = float4(contribution, fade);
  return output;
}
