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
#if GLTF_USE_HAIR
#include "hair_bsdf.h.slang"
#endif
#include "gltf_vertex_access.h.slang"
#include "shaderio.h"
#include "get_hit.h.slang"
#include "dlss_util.h"
#include "common.h.slang"
#include "aerial_perspective.h.slang"
#include "aperture.h.slang"
#include "sky_bruneton_radiance.h.slang"

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

// What a camera ray or a mirror bounce sees of the physical sky: its radiance, evaluated per ray
// through the same function the bake uses, plus the solar disk. Per ray rather than read from the
// baked image so what the camera sees does not depend on the bake resolution -- the horizon, where
// sky meets ground in a hard edge, is what a lat-long would otherwise smear.
float3 evalPhysicalSkyBackground(float3 worldDir, SceneFrameInfo* frameInfo)
{
  const float3 dir    = quatRotateInverse(frameInfo->envRotation, worldDir);
  const float3 sunDir = quatRotateInverse(frameInfo->envRotation, frameInfo->sunDirection);
  // observerRadius is the bake's: selecting the physical sky commits a bake at the top of the frame,
  // before any frame that draws it is recorded (GltfRenderer::applyPendingEnvironmentRefresh).
  const float3 sky = evalBrunetonSkyRadiance(g_atmosphere, g_transmittance, g_scattering, g_singleMie, g_irradiance,
                                             frameInfo->sunDisk->observerRadius, dir, sunDir);
  return sky + evalSunDiskBackground(dir, frameInfo);
}

// Specialization constants
[[vk::constant_id(0)]]  int USE_SER = 1;
[[vk::constant_id(1)]]  int USE_DLSS_TRANSP = 0; // Dlss::TransparencyMode: 1 = primary surface replacement through clear glass

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
      freeFlight = sampleFreeFlight(pt.volumeMedium, pt.throughput, seed);
      if(freeFlight.drawn && isfinite(freeFlight.dist))
        ray.TMax = freeFlight.dist;
    }
#endif

    // Trace the ray through the scene. A walk's step sees only what can bound its medium: it
    // passes through lashes, hair and clothing that reach under the skin (INSTANCE_MASK_WALK).
    raytracer.Trace(ray, payload, seed, freeFlight.drawn ? INSTANCE_MASK_WALK : 0xFF);
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
    bool firstRay = (pt.surfaceDepth == 0);

    // Check if we hit the infinite plane
    bool hitInfinitePlane = checkInfinitePlaneIntersection(ray, payload, hit, frameInfo, seed);

    //-----------------------------------------------------------------------
    // Hitting the environment
    //-----------------------------------------------------------------------
    if(payload.hitT == INFINITE)
    {
      // A walk that leaves the mesh pays for the unbounded segment before the sky is considered.
      // With scattering an event is sampled; with absorption only the transmittance is 0.
      PathStepResult escaped = processVolumeSegment(raytracer, INFINITE, ray, pt, seed, freeFlight);
      if(escaped != PathStepResult.eOutOfVolume)
        return escaped;

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
          const float3 disk  = evalSkySunDisk(bgDir, frameInfo);
          if(any(disk > 0.0F))
            addRadiance(pt, pt.throughput * computeSkySunHitMisWeight(pt.lastSamplePdf, pt.lightPick) * disk * frameInfo->envIntensity, indirect);
        }
      }

      return PathStepResult.eBreak;
    }

    // Charge the in-volume segment up to this hit before the front-face escape test, material
    // evaluation or emission. A scatter returns here, so a surface the walk never reaches is not
    // shaded. A walk that survives to an eyeball or a tooth has paid for the trip, and the
    // front-face clear below then drops the medium. This is the segment's one charge: what reaches
    // the surface -- its emission included -- comes through the medium's transmittance once, and
    // nothing below may charge it again (a second call squared the transmittance of every
    // absorbing medium and of a walk's refractive exit).
    {
      PathStepResult volumeStep = processVolumeSegment(raytracer, payload.hitT, ray, pt, seed, freeFlight);
      if(volumeStep != PathStepResult.eOutOfVolume)
        return volumeStep;
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
    // A diffuse exit's boundary index, for its light-side Fresnel (ePtExitFresnel); 0: none
    float exitFresnelEta = 0.0F;
    // Volumetric scatter shades as the thin preview at this hit: the preview flag, or a rough path
    // arriving from outside any medium (a walk's own exit, from inside, stays volumetric)
    bool simplifyScatter = hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin)
                           || (hasFlag(pushConst.flags, PathtracerFlags::ePtSimplifyScatter) && !ptIsInside(pt)
                               && pt.pathRoughness >= SCATTER_SIMPLIFY_ROUGHNESS);

    // Compute the ray-cone footprint at the hit point which is used to compute the texture-LOD gradient
    float worldFoot = rayConeWorldFootprint(pt.cone, payload.hitT, hit.geonrm, -ray.Direction);

    //-----------------------------------------------------------------------
    // Hitting the infinite plane
    //-----------------------------------------------------------------------
    if(hitInfinitePlane)
    {
      // Replaced the material at the hit point with the infinite plane material
      pbrMat = defaultPbrMaterial(frameInfo->infinitePlaneBaseColor, frameInfo->infinitePlaneMetallic,
                                  frameInfo->infinitePlaneRoughness, hit.nrm, hit.nrm);

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
    // Hitting the mesh
    //-----------------------------------------------------------------------
    else
    {
      // Find material at the hit point and evaluate the material (PBR material evaluation)

      materialIndex = max(0, renderNode.materialID);

#if GLTF_USE_VOLUME
      // A path inside a medium can only leave it through a back face. Meeting a front face means
      // the mesh had an opening (an eye socket, a mouth on a figure) and the walk escaped through
      // it: the medium ended at the opening, so the path continues outside, with no transmittance
      // charged for the way here and this surface shaded from its front.
      if(ptIsInside(pt) && hit.frontFace)
      {
        pt.isInside     = false;
        pt.volumeMedium = {};
#if GLTF_USE_SCATTER
        pt.volumeMaterial = -1;
        // Light that leaves a walk has scattered: the path is rough from here, as after the diffuse exit
        if(pt.scatterBounces > 0)
        {
          pt.prefixIndirect = true;
          pt.pathRoughness  = 1.0F;
        }
#endif
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
        pt.isInside         = false;
        pt.volumeMedium     = {};
        pt.volumeMaterial   = -1;
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
        hit.shadowPos = hit.shadowPosOther;  // the terminator offset for the outer side, which the light is on
      }
#endif

      float     texGrad = worldFoot * hit.texelDensity * pushConst.texGradScale;
      MeshState mesh    = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, ptIsInside(pt), texGrad);
      mesh.baseColorVertexMul = hit.color;
      pbrMat                  = evaluateMaterial(materialIndex, mesh, allImages, allSamplers, texInfos, materials, simplifyScatter);
#if GLTF_USE_VOLUME && GLTF_USE_SCATTER
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
      pt.guide.specularHitDist    = float16_t(payload.hitT);
      pt.guide.captureSpecularHit = false;
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
      // #DLSS instance and deformation motion - Previous-frame world position of this surface point.
      pt.firstHitPosPrev = hit.pos;
      if(!hitInfinitePlane)
      {
        const float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
        pt.firstHitPosPrev = dlssPrevHitPos(renderNode, renderPrim, payload.rnodeID, payload.rprimID, payload.primitiveID, barys, hit.pos);
      }

      // Base-color guide albedo (shared by DLSS and the other denoisers). Stored temporarily, written at end to
      // reduce live state. OptiX and OIDN take the surface colour, metal included; DLSS-RR the diffuse
      // albedo, its specular share going to the specular guide (NVIDIA's RR guide). Transmission is not
      // taken off: skin's (up to 0.95) would leave RR a near-black albedo.
      float3 guideAlbedo = pbrMat.baseColor.xyz;
#if USE_DLSS_SHADER
      if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
        guideAlbedo *= 1.0F - pbrMat.metallic;
      // #DLSS specular guide - specular albedo + clear-glass albedo refinement. DLSS-only quality
      // path; the other denoisers need neither, so it is compiled out of their variant.
      // KHR_materials_specular: dielectric_f0 = min(f0_from_ior * specularColor, 1.0) * specular; a metal's
      // F0 is its base colour; the clearcoat (DAZ skin's top lobe) adds its own at 0.04.
      float  f0_from_ior = pow((pbrMat.ior2 - pbrMat.ior1) / (pbrMat.ior2 + pbrMat.ior1), 2.0F);
      float3 dielectricF0 = min(f0_from_ior * pbrMat.specularColor, float3(1.0F)) * pbrMat.specular;
      float3 f0           = lerp(dielectricF0, pbrMat.baseColor.xyz, pbrMat.metallic);
      const float ndotv   = dot(pbrMat.N, ray.Direction);
      float3 specularAlbedo = EnvBRDFApprox2(f0, pbrMat.roughness.x, ndotv);
      // A metal's multiple scattering, as its lobe gives it back (ggxMultiScatterFactor)
      specularAlbedo *= lerp(float3(1.0F), ggxMultiScatterFactor(pbrMat.baseColor, pbrMat.roughness.x, abs(ndotv)), pbrMat.metallic);
      if(pbrMat.clearcoat > 0.0F)
        specularAlbedo = saturate(specularAlbedo
                                  + pbrMat.clearcoat * EnvBRDFApprox2(float3(0.04F), pbrMat.clearcoatRoughness * pbrMat.clearcoatRoughness, ndotv));
      float3 guideNormal    = pbrMat.N;
      // RR reads linear (perceptual) roughness; the material keeps GGX alpha
      float  guideRoughness = sqrt(pbrMat.roughness.x);
      // also on the accumulation's frames, so OptiX and OIDN see the iris behind a cornea too (only DLSS
      // reads the replaced depth and motion)
      if(USE_DLSS_TRANSP == 1 && dlssIsClearGlass(pbrMat))
      {
        // #DLSS primary surface replacement: describe what is seen through the glass. The albedo
        // keeps the glass's Fresnel share; depth and motion come from the replaced surface.
        DlssGlassPsr psr     = dlssTraceClearGlass(pbrMat, hit, ray.Direction, ptIsInside(pt));
        float        ndotv   = saturate(dot(pbrMat.N, -ray.Direction));
        float        fresnel = schlickFresnel((pbrMat.ior2 > 0.0f) ? (pbrMat.ior2 / pbrMat.ior1) : 1.5f, ndotv);
        guideAlbedo          = dlssEnforceMinAlbedo(lerp(psr.albedo, pbrMat.baseColor.xyz, fresnel), specularAlbedo);
        if(psr.valid)
        {
          guideNormal        = psr.normal;
          guideRoughness     = sqrt(psr.roughness);
          pt.guide.psrValid  = true;
          pt.guide.psrPos    = hit.pos + ray.Direction * psr.pathLength;
          pt.firstHitPosPrev = pt.guide.psrPos + psr.posDelta;
        }
      }
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

    // The material's own roughness, before the regularisation below: whether a refraction through it
    // is a clear crossing (CLEAR_CROSSING_ALPHA) is decided by it, as the shadow rays decide.
    const float materialAlpha = max(pbrMat.roughness.x, pbrMat.roughness.y);

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
      float3              vizColor;
      VisualizationResult vizResult = applyVisualization(pbrMat, hit, frameInfo->visualization, payload.rprimID,
                                                         payload.primitiveID, payload.ommUnknown ? 1 : 0, vizColor);
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
    // hits one is MIS-weighted against the light technique, mirroring the env-hit MIS above.
    if(!walkOwnsDirect && any(pbrMat.emissive > float3(0.0)))
    {
      float emissiveMis = 1.0;
      if(!hitInfinitePlane)
      {
        // Power-weighted MIS pdf needs only the radiance the emitter was picked by, its geometric
        // normal, and W_total (the area cancels) -- no per-triangle lookup.
        emissiveMis = computeEmitterHitMisWeight(pt.lastSamplePdf, pt.lightPick, emitterSelectionRadiance(materialIndex),
                                                 hit.geonrm, ray.Direction, pt.nullDistance + payload.hitT);
      }
      addRadiance(pt, emissiveMis * pbrMat.emissive * pt.throughput, pt.prefixIndirect);
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
      hairOn = makeHairHit(materials[materialIndex], hit.uv[0], hit.bitangent, hit.tangent, pbrMat.N, pbrMat.baseColor, hairHit);
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
    }
#endif
#endif
    const bool bothSides = pbrMat.diffuseTransmissionFactor > 0.0f || hairOn;

    DirectLight directLight;
    // A fibre event spends the hair's own budget while it lasts, not a surface bounce
#if GLTF_USE_HAIR
    bounce.hairEvent = hairOn && pt.hairBounces < pushConst.hairBudget;
    bounce.fibre     = hairOn;
#endif
    // The last vertex's BSDF continuation is never traced (the loop ends first), so its light samples
    // have no BSDF-sampled partner: weight 1, as for a punctual light, and the full environment table.
    bool hairCut = false;
#if GLTF_USE_HAIR
    hairCut = hairCutHere(bounce.fibre, pt.hairBounces);
#endif
    const bool lastVertex = hairCut || (!bounce.hairEvent && pt.surfaceDepth + 1 >= pushConst.maxDepth);
    // The lobe mixture for this vertex's light sample and BSDF sample alike, so their MIS agrees. With
    // ePtLobeMixtureDirect only where the path is still direct: there a catchlight is seen, through a
    // cornea one sample in forty with a single lobe; past a rough bounce one lobe costs a third less.
    const bool lobeMixture = hasFlag(pushConst.flags, PathtracerFlags::ePtLobeMixture)
                             && (!hasFlag(pushConst.flags, PathtracerFlags::ePtLobeMixtureDirect) || pt.pathRoughness <= DIRECT_PATH_ROUGHNESS);
    // The normal only culls directional lights below the surface (singleLightContribution). Give it
    // the one bounce.nextEventValid tests, so NEE and the BSDF side (R4) agree on which directions
    // next-event estimation can reach; the BSDF then decides what a light below the shading normal does.
    // The vertex before this one, for a ray that crosses this surface clear (below)
    const float     prevSamplePdf = pt.lastSamplePdf;
    const LightPick prevLightPick = pt.lightPick;
    sampleLights(hit.pos, hit.nrm, ray.Direction, seed, directLight, /*isVolumeSample=*/false,
                 /*lightsFromBothSides=*/bothSides, /*compensatedEnv=*/!lastVertex, g_sobol ? nextSobolDecision() : float4(-1.0F));
    // The BSDF ray sampled below is weighed against this vertex's pick when it reaches a light
    pt.lightPick = directLight.pick;

    // This checks if the light could contribute, but doesn't send a next event estimation (NEE) shadow ray yet.
    // The actual test and contribution are delayed to minimize per-bounce state.
    bounce.nextEventValid = (dot(directLight.direction, hit.nrm) > 0.0f || bothSides)
                            && directLight.pdf != 0.0f;

    //-----------------------------------------------------------------------
    // Evaluating the BSDF for the light
    //-----------------------------------------------------------------------
    if(bounce.nextEventValid)
    {
      // Evaluating the BSDF at the hit point for the light
      BsdfEvaluateData evalData;
      evalData.k1 = -ray.Direction;
      evalData.k2 = directLight.direction;
      evalData.xi = float3(rand(seed), rand(seed), rand(seed));
      // Every lobe for a punctual light (no BSDF ray reaches it; its weight is 1 whatever the pdf), and
      // for every light when MIS weighs against the lobe mixture on both sides (bsdfSampleCore below).
      const bool allLobes = lobeMixture || (directLight.pdf == DIRAC && hasFlag(pushConst.flags, PathtracerFlags::ePtLobeSumPunctual));
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

      // Some lobe can scatter toward the light (in every mode): the shadow ray is worth tracing
      if(evalData.pdf > 0.0)
      {
        // Weight for combining light and BSDF sampling strategies (Multiple Importance Sampling);
        // 1 at the last vertex (above).
        const float mis_weight = (directLight.pdf == DIRAC || lastVertex) ? 1.0F : misPowerWeight(directLight.pdf, evalData.pdf);

        // sample weight
        const float3 w = pt.throughput * directLight.radianceOverPdf * mis_weight;
        bounce.contribution += w * evalData.bsdf_diffuse;
        bounce.contribution += w * evalData.bsdf_glossy;
        // Classed by the path before this vertex, as the sample below is about to change it
        bounce.indirect = pt.pathRoughness > DIRECT_PATH_ROUGHNESS;
      }
      else
      {
        // The BSDF cannot scatter toward the light, so the shadow ray could only multiply zero
        bounce.nextEventValid = false;
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
      if(dot(directLight.direction, hit.nrm) <= 0.0F && pbrMat.thickness > 0.0F && !thinScatter)
      {
        bounce.shadowRayInside     = !pt.isInside;
        bounce.shadowRayExtinction = volumeExtinction(makeVolumeMedium(pbrMat));
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
        bsdfSampleCore(sampleData, pbrMat, mixturePdf);
      }
      // bsdfSampleSimple(sampleData, pbrMat);
      // The exit's Fresnel weighs the sample, not its pdf: both MIS sides keep the cosine lobe's
      if(exitFresnelEta > 0.0F && sampleData.event_type != BSDF_EVENT_ABSORB)
        sampleData.bsdf_over_pdf *= diffuseExitFresnelWeight(dot(pbrMat.N, sampleData.k2), exitFresnelEta);

      // Update the pt.throughput
      pt.throughput *= sampleData.bsdf_over_pdf;
      ray.Direction    = sampleData.k2;  // new direction
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
      // through (getShadowTransmission, CLEAR_CROSSING_ALPHA). The previous vertex's shadow ray already
      // reached the light behind it, so the light this ray reaches beyond is weighed against that
      // vertex's light sample: the crossing is a null event for MIS. Weighed against its own (next to
      // nothing through smooth glass), the light behind a window or a cornea was counted 1 + w times.
      // A camera ray through it keeps its DIRAC: through a window, the background.
      bool clearCrossing = false;
#if GLTF_USE_TRANSMISSION
      clearCrossing = hasFlag(pushConst.flags, PathtracerFlags::ePtShadowTransmission) && !hairOn && materialIndex >= 0
                      && (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0
                      && (sampleData.event_type & (BSDF_EVENT_GLOSSY | BSDF_EVENT_IMPULSE)) != 0
                      && materialAlpha <= CLEAR_CROSSING_ALPHA && materials[materialIndex].scatterStrengthFactor <= 0.0F;
#endif
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
      if(firstRay && (sampleData.event_type == BSDF_EVENT_GLOSSY_REFLECTION || sampleData.event_type == BSDF_EVENT_IMPULSE_REFLECTION))
      {
        pt.guide.captureSpecularHit = true;
      }
#endif  // USE_DLSS_SHADER

      // The ray wasn't absorbed, so we can continue the path
      if(sampleData.event_type != BSDF_EVENT_ABSORB)
      {
        // How rough this scattering was (perceptual): a diffuse lobe 1, an impulse 0, a glossy one
        // its own roughness. The event does not say which glossy lobe it was: take the smoother of
        // the specular and the clearcoat, so a wrong guess clamps less, not more. A fibre: its
        // longitudinal roughness.
        float eventRoughness = 1.0F;
        if((sampleData.event_type & BSDF_EVENT_IMPULSE) != 0)
          eventRoughness = 0.0F;
        else if((sampleData.event_type & BSDF_EVENT_GLOSSY) != 0)
        {
          float alpha = max(pbrMat.roughness.x, pbrMat.roughness.y);
          if(pbrMat.clearcoat > 0.0F)
            alpha = min(alpha, pbrMat.clearcoatRoughness * pbrMat.clearcoatRoughness);
          eventRoughness = sqrt(alpha);
        }
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

        // Compute the offset position for the next bounce
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
        if(isTransmission && pbrMat.thickness > 0.0f)
        {
          bool previewThin = false;
#if GLTF_USE_SCATTER
          previewThin = simplifyScatter && materialIndex >= 0 && materials[materialIndex].scatterStrengthFactor > 0.0f;
#endif
          if(!previewThin)
          {
            pt.isInside = !pt.isInside;
#if GLTF_USE_VOLUME
            if(pt.isInside)
              pt.volumeMedium = makeVolumeMedium(pbrMat);  // Entering volume: capture the medium for scatter/absorption
            else
              pt.volumeMedium = {};
#if GLTF_USE_SCATTER
            // A walk starts here: its material for the diffuse exit, and a fresh event budget
            pt.volumeMaterial = pt.isInside ? materialIndex : -1;
            if(pt.isInside)
              pt.scatterBounces = 0;
            const float3 sigmaS = float3(pt.volumeMedium.scatterCoefficient);
            entersWalk          = pt.isInside && max(sigmaS.x, max(sigmaS.y, sigmaS.z)) > VOLUME_MIN_SCATTER;
#endif
#endif
          }
        }
#endif

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
      addRadiance(pt, bounce.contribution * shadowFactor, bounce.indirect);
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
    if((depth >= RR_MIN_DEPTH || fibreRoulette) && (bounce.hairEvent || pt.surfaceDepth + 1 < pushConst.maxDepth))
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

    // Consume one surface bounce, or one of the hair's; a path cut at a fibre (hairCutDlss) ends here
#if GLTF_USE_HAIR
    if(hairCutHere(bounce.fibre, pt.hairBounces))
      pt.surfaceDepth = pushConst.maxDepth;
    else if(bounce.hairEvent)
      pt.hairBounces++;
    else
#endif
      pt.surfaceDepth++;
  }

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
    float3 inScatter = march.inScatter;
    if(any(march.sunEstimate > 0.0F))
    {
      RayDesc shadowRay;
      shadowRay.Origin        = lerp(eye, pt.firstHitPos, march.sunPoint);
      shadowRay.Direction     = pushConst.frameInfo->sunDirection;
      shadowRay.TMin          = 0.0F;
      shadowRay.TMax          = INFINITE;
      const float3 visibility = raytracer.TraceShadow(shadowRay, seed);
      inScatter += visibility * march.sunEstimate - march.sunInScatter;
    }
    pt.radiance = pt.radiance * march.transmittance + inScatter;
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
  if(sampleResult.radiance.w <= 0.0f)
  {
    const float3 background = max(sampleResult.radiance.xyz, float3(0.0f));
    const float  maxComp    = max(background.x, max(background.y, background.z));
    albedo                  = maxComp > 0.0f ? saturate(background / maxComp) : float3(0.0f);
  }
  albedoSum += albedo;
  normalSum += sampleResult.guideOutput.normalRoughness.xyz;
}
#endif  // USE_GUIDE_SHADER

//-----------------------------------------------------------------------
// Per-pixel accumulation and GBuffer output
//-----------------------------------------------------------------------
void processPixel(IRaytracer raytracer, float2 samplePos, float2 imageSize)
{
  // Check if the sample position is within the image bounds
  if(samplePos.x >= imageSize.x || samplePos.y >= imageSize.y)
    return;

  // Check if the sample position is the mouse coordinate
  if(samplePos.x == pushConst.mouseCoord.x && samplePos.y == pushConst.mouseCoord.y)
  {
    // Set the debug flag to true (used for printing debug information)
    doDebug = true;
  }

  // Initialize the random number
  // pushConst.seed selects another sequence (0: the default one), e.g. for a reference independent of a test render
  uint seed = xxhash32(uint3(uint2(samplePos.xy), uint(pushConst.frameCount) + uint(pushConst.seed) * 0x9E3779B9u));
  // A Sobol pixel draws its points at its sample index, under a scramble of its own that every frame keeps
  g_sobol      = hasFlag(pushConst.flags, PathtracerFlags::ePtSobol) && !hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss);
  g_sobolPixel = xxhash32(uint3(uint2(samplePos.xy), uint(pushConst.seed) + 0x5bd1e995u));
  beginSobolSample(uint(pushConst.totalSamples));

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

  // Sampling n times the pixel
  SampleResult sampleResult = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo->projInv,
                                          pushConst.frameInfo->viewInv, pushConst.focalDistance, pushConst.aperture);
  float4 pixelColor = sampleResult.radiance;

#if USE_GUIDE_SHADER
  // The denoiser guides average every sample of the frame, like the color
  const bool denoiserGuides = hasFlag(pushConst.flags, PathtracerFlags::ePtDenoiserGuides);
  float3     albedoSum      = float3(0.0f);
  float3     normalSum      = float3(0.0f);
  int        guideCount     = 0;
  if(denoiserGuides)
  {
    addDenoiserGuides(sampleResult, albedoSum, normalSum);
    guideCount++;
  }
#endif

  // The frame traces exactly the samples the host counts (numSamples): the host decides, per frame.
  // A frame DLSS denoises gets 1 (DLSS does its own temporal accumulation); with OptiX or OIDN also
  // on, the frames after DLSS's frame 0 get Auto SPP's or ptSamples' count.
  for(int s = 1; s < pushConst.numSamples; s++)
  {
    beginSobolSample(uint(pushConst.totalSamples + s));
    subpixelJitter = gaussianSubpixelJitter(seed);  // the same pixel filter as the first sample
    sampleResult   = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo->projInv,
                                 pushConst.frameInfo->viewInv, pushConst.focalDistance, pushConst.aperture);
    pixelColor += sampleResult.radiance;
#if USE_GUIDE_SHADER
    if(denoiserGuides)
    {
      addDenoiserGuides(sampleResult, albedoSum, normalSum);
      guideCount++;
    }
#endif
  }
  pixelColor /= pushConst.numSamples;


  // NDC depth from first-hit world position (Vulkan [0,1] range, 1.0 = far/miss).
  // Written to two targets: outDepth (main GBuffer, full viewport resolution, frame 0 only)
  // and eDlssDepth (DLSS guide buffer, DLSS render resolution, every frame when DLSS active).
  // Both use the same NDC value but the images differ in resolution.
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
  if(firstFrame)
    outDepth[int2(samplePos)] = ndcDepth;

  // Saving result
  // Nothing accumulated yet (the first frame, or the first after DLSS's frame 0): replace the value in
  // the buffer, never weigh it by 0, which an infinite value would turn into NaN
  if(firstFrame || pushConst.totalSamples == 0 || hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
  {
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] = pixelColor;
  }
  else
  {
    // Do accumulation over time using uniform weighting
    float  totalSamplesAfter = float(pushConst.totalSamples + pushConst.numSamples);
    float4 old_color         = outImages[0][int2(samplePos)];
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] =
        (old_color * pushConst.totalSamples + pixelColor * pushConst.numSamples) / totalSamplesAfter;
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
    outImages[int(OutputImage::eDlssDepth)][int2(samplePos)]           = float4(guideDepth);
    outImages[int(OutputImage::eDlssMotion)][int2(samplePos)]          = float4(motionVec, 0, 0);
    outImages[int(OutputImage::eDlssNormalRoughness)][int2(samplePos)] = sampleResult.guideOutput.normalRoughness;
    // the sky's albedo is 0.5, as NVIDIA's RR guide asks: 0 makes it boil at the silhouette
    float4 guideAlbedo = sampleResult.guideOutput.albedo;
    if(!hasSolidHit)
      guideAlbedo = float4(0.5f);
    outImages[int(OutputImage::eDlssAlbedo)][int2(samplePos)] = guideAlbedo;
    outImages[int(OutputImage::eDlssSpecAlbedo)][int2(samplePos)] = float4(sampleResult.guideOutput.specularAlbedo.xyz, 1.0f);
    outImages[int(OutputImage::eDlssSpecularHitDist)][int2(samplePos)] = float4(sampleResult.guideOutput.specularHitDist);
    outImages[int(OutputImage::eNrMask)][int2(samplePos)] = float4(sampleResult.guideOutput.nrMask);
  }
#endif  // USE_DLSS_SHADER

#if USE_GUIDE_SHADER
  if(denoiserGuides)
  {
    // The guides of the denoiser of the accumulation are averaged over the samples like the color,
    // so their edges are anti-aliased (and blurred by depth of field) the same way. They have their
    // own sample count: they restart when the denoiser is turned on, while the color goes on. Each
    // denoiser converts them to what it reads.
    const float3 albedo = albedoSum / float(guideCount);
    const float3 normal = normalSum / float(guideCount);
    const int2   pixel  = int2(samplePos);
    if(firstFrame || pushConst.guideSamples == 0)
    {
      outImages[int(OutputImage::eDenoiserAlbedo)][pixel] = float4(albedo, 1.0f);
      outImages[int(OutputImage::eDenoiserNormal)][pixel] = float4(normal, 1.0f);
    }
    else
    {
      const float weight = float(guideCount) / float(pushConst.guideSamples + guideCount);
      outImages[int(OutputImage::eDenoiserAlbedo)][pixel] =
          lerp(outImages[int(OutputImage::eDenoiserAlbedo)][pixel], float4(albedo, 1.0f), weight);
      outImages[int(OutputImage::eDenoiserNormal)][pixel] =
          lerp(outImages[int(OutputImage::eDenoiserNormal)][pixel], float4(normal, 1.0f), weight);
    }
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
    raytracer.Trace(ray, payload, seed, 0xFF);
    if(payload.hitT == INFINITE)
      break;

    GltfRenderNode      node  = renderNodes[payload.rnodeID];
    GltfRenderPrimitive prim  = renderPrimitive[payload.rprimID];
    float3              barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
    HitState hit = getHitState(prim, barys, node.worldToObject, node.objectToWorld, payload.primitiveID, ray.Direction);

    // A boundary the path would cross into a medium through (see the pt.isInside toggle). The mesh
    // bounds the medium, so the material's factors decide, not their textured values at the hit: a
    // texel with no transmission (a logo on glass) is still the medium's wall, and passing it would
    // take the far wall, met from behind, for the camera being inside (upstream 10f2de0). Only the
    // transmissions this shader evaluates count, as in evaluateMaterial().
    const GltfShadeMaterial factors      = materials[max(0, node.materialID)];
    bool                    transmissive = false;
#if GLTF_USE_TRANSMISSION
    transmissive = transmissive || factors.transmissionFactor > 0.0F;
#endif
#if GLTF_USE_DIFFUSE_TRANSMISSION
    transmissive = transmissive || factors.diffuseTransmissionFactor > 0.0F;
#endif
    if(factors.thicknessFactor > 0.0F && transmissive)
    {
      if(!hit.frontFace)
      {
        MeshState mesh          = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, !hit.frontFace, 0.0F);
        mesh.baseColorVertexMul = hit.color;
        // the thin preview has no medium to be inside of
        PbrMaterial mat = evaluateMaterial(max(0, node.materialID), mesh, allImages, allSamplers, texInfos, materials,
                                           hasFlag(pushConst.flags, PathtracerFlags::ePtScatterAsThin));
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
// RAY-QUERY COMPUTE
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(PT_WORKGROUP_X, PT_WORKGROUP_Y, 1)]
void computeMain(uint3 threadIdx: SV_DispatchThreadID)
{
  RAYQUERY_USE_OPACITY_MICROMAP(computeMain);
  // The render size, not the output image's: OptiX 2x upscale traces only its top-left quarter.
  RayQueryRaytracer raytracer;
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtProbeCameraMedium))
  {
    if(all(threadIdx.xy == 0))
      probeCameraMedium(raytracer);
    return;
  }
  float2 samplePos = (float2)threadIdx.xy;
  processPixel(raytracer, samplePos, (float2)pushConst.renderSize);
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("raygeneration")]
void rgenMain()
{
  RAYQUERY_USE_OPACITY_MICROMAP(rgenMain);  // Ray queries here too (e.g. dlssTraceSkipThinGlass)
  TraditionalRaytracer raytracer;
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtProbeCameraMedium))
  {
    if(all(DispatchRaysIndex().xy == 0))
      probeCameraMedium(raytracer);
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

  float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);
  if(stochasticOpacityU(instanceID, triangleID, barycentrics, payload.seed) >= opacity)
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
