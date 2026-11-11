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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "nvshaders/slang_types.h"
#include "sky_omi_shaderio.h.slang"
// Bruneton atmosphere parameters. Its own header rather than part of this bundle: only the sky
// subsystem needs it, and it carries an upstream BSD-3 notice that should stay with its content.
#include "sky_bruneton_io.h.slang"
#include "gltf_scene_io.h.slang"
#include "nvshaders/hdr_io.h.slang"

NAMESPACE_SHADERIO_BEGIN()

#define WORKGROUP_SIZE 16
// The ray-query path tracer's workgroup (computeMain), apart from WORKGROUP_SIZE because its size
// steers how the compiler budgets registers for that one large kernel. At 16x16 (256 threads) the
// kernel with the DLSS guides got 202 registers and 8 warps per SM; at 128 threads or fewer, 168
// registers with a small spill and 12 warps -- ~25% faster path tracing with DLSS, ~6-18% without
// (RTX PRO 4500 Blackwell). The shape among those barely matters; 8x8 keeps rays coherent.
#define PT_WORKGROUP_X 8
#define PT_WORKGROUP_Y 8
#define SILHOUETTE_WORKGROUP_SIZE 16

// TLAS instance masks. Every ray sees every instance but the rays of a scatter walk, which see only
// what can bound the medium: a material whose extras say `"fvWalkThrough": true` (lashes, hair,
// clothing on a figure) has no INSTANCE_MASK_WALK, so a walk under the skin passes through it
// instead of leaving the skin early.
#define INSTANCE_MASK_WALK 0x01
#define INSTANCE_MASK_OTHER 0x02


// Indices into `texturesHdr[]` (eTexturesHdr binding).
#define HDR_IMAGE_INDEX 0    // Lat-long HDR environment image
#define HDR_TEXTURE_COUNT 1  // Count for descriptor binding

// Environment types.
//
// Values are explicit and appended, never reordered: `envSystem` is persisted as a raw integer in
// the .ini and is part of the command-line and MCP surface, so inserting a value would silently
// remap every existing user's setting. The UI shows them in a different, more readable order
// through an explicit label/value table rather than a positional combo string.
enum class EnvSystem
{
  eSky      = 0,  // Procedural physical sky
  eHdr      = 1,  // Lat-long HDR image
  eNone     = 2,  // No environment: no sky/HDR lighting or background (black backdrop)
  ePlain    = 3,  // OMI_environment_sky `plain`: one authored solid color
  eGradient = 4,  // OMI_environment_sky `gradient`: authored bottom/horizon/top bands + sun
};

// Output image types
enum OutputImage
{
  eResultImage = 0,      // Output image (RGBA32)
  eSelectImage,          // Selection image (R32)
  eDlssAlbedo = 2,       // Diffuse albedo (RGBA8)
  eDlssSpecAlbedo,       // Specular albedo (RGBA32)
  eDlssNormalRoughness,  // Normal and roughness (RGBA32)
  eDlssMotion,           // Motion (RGBA32)
  eDlssDepth,            // Depth (R32)
  eDlssSpecularHitDist,  // Specular hit distance (R16F)
  eNrMask,               // NR per-material control mask (RGBA16F)
  eDenoiserAlbedo = 2,   // Accumulated albedo guide of DLSS-RR's still (RGBA32)
  eDenoiserNormal = 3,   // Accumulated world-space normal guide (RGBA32), its w the accumulated linear roughness.
                         // Both alias DLSS slots: the still's images are bound there, never on DLSS's frames
  eRrStillSpecAlbedo = 4,  // DLSS-RR's still (ePtRrStillGuides): accumulated specular albedo (RGBA32)
  eRrStillDepth      = 6,  // DLSS-RR's still: the nearest guide depth of its samples (R32), the surface behind clear glass's
  eConvergence = 9,      // Per pixel of an accumulation (RG32): the mean of its samples' luminance squared, and 1 while it has converged (ePtAdaptiveStop)
  eDlssResponsivity = 9, // DLSS-RR responsivity mask (R16F): ptHairResponsivity where the first hit is hair, 0 elsewhere.
                         // DLSS frames only, so it shares the still's eConvergence slot
  eDlssSpecularMotion = 10,  // DLSS-RR's motion of reflections (RG16F, pixels, as eDlssMotion): of a reflection's
                             // image, the surface's own where there is none
  eOutputImageSlots = 11,    // How many slots the output image binding holds: one past the last above
};

// Binding points for descriptors
enum BindingPoints
{
  eTlas = 0,      // Top level acceleration structure
  eOutImages,     // Output image (RGBA32); eSelectImage slot = ObjectID in .r (R32_SFLOAT)
  eOutDepth,      // Scene depth output (D32_SFLOAT as r32f storage image)
  eTextures,      // glTF material images (bindless SAMPLED_IMAGE array; indexed by GltfTextureInfo.index)
  eTexturesHdr,   // 2D environment textures (HDR_IMAGE_INDEX)
  eSamplers,      // glTF samplers (bindless SAMPLER array; indexed by GltfTextureInfo.samplerIndex, slot 0 = default)
};

// DLSS hand-off (denoiser_handoff.slang): storage images, in VK_IMAGE_LAYOUT_GENERAL
enum DenoiserHandoffBindings
{
  eHandoffDlss,      // In: DLSS's image, kept from frame 0 (RGBA32F)
  eHandoffDenoised,  // In: the accumulation denoiser's image (RGBA32F)
  eHandoffOut,       // Out: the image shown (RGBA16F)
};

// Binding points for descriptors
enum SilhouetteBindings
{
  eObjectID,          // In: the object ID image (R32_UINT)
  eRGBAIImage,        // Out: the output image
  eSelectionBitMask,  // In: storage buffer of uint32_t (one bit per render node)
};

enum Visualization
{
  eRendered,
  eBaseColor,
  eMetallic,
  eRoughness,
  eNormalShading,
  eNormalGeometric,
  eTangent,
  eBitangent,
  eEmissive,
  eOpacity,
  eTexCoord0,
  eTexCoord1,
  eClay,
  eTriangleID,
  eFaceOrientation,
  // Khronos glTF-Sample-Renderer DEBUG_* parity (subset — only the modes that map to fields
  // already populated in PbrMaterial; otherwise we render the base color as a fallback).
  eOcclusion,
  eClearcoatFactor,
  eClearcoatRoughness,
  eClearcoatNormal,
  eSheenColor,
  eSheenRoughness,
  eSpecularFactor,
  eSpecularColor,
  eTransmissionFactor,
  eIridescenceFactor,
  eIridescenceThickness,
  eAnisotropyStrength,
  eDiffuseTransmissionFactor,
  eDiffuseTransmissionColor,
  eOpacityMicromap,
  // The app's Display views beyond Clay and Normals (view.set display, fv_view.cpp); appended, as
  // dbgVisualization is saved by its number
  eMaterialId,        // which material: a colour each (idColor)
  eObjectId,          // which part of which prop: a figure's body, each garment, the hair (GltfRenderNode::ownerID)
  eTriangleSize,      // how large the mesh's triangles are, on a log scale (triangleSizeRamp, viridis)
  eBackFacingNormal,  // where the shading normal turns away from the eye, the bump past the silhouette
};


// Bit flags for SceneFrameInfo::flags
enum SceneFrameInfoFlags
{
  eSceneIsOrthographic             = 1 << 0,
  eSceneUseSolidBackground         = 1 << 1,
  eSceneUseHdrEnvironment          = 1 << 2,
  eSceneUseInfinitePlane           = 1 << 3,
  eSceneInfinitePlaneShadowCatcher = 1 << 4,
  eSceneUseNoEnvironment           = 1 << 5,  // No sky/HDR environment lighting or background
};

// Camera info
// The solar disk the Bruneton bake leaves out of its image, written by that bake and read by the
// background passes. Kept out of SceneFrameInfo's own storage because the value is produced on the
// GPU -- the transmittance toward the sun comes from the atmosphere LUTs -- and the host never
// learns it. See publishSunDisk() in shaders/env_bake.slang.
struct SkySunDisk
{
  float3 radiance;          // in the baked image's units, under a clear sky: the clouds' share is taken where its light lands
  float  cosAngularRadius;  // cos of the sun's angular radius; the disk test is a dot product
  float  angularRadius;     // radians. Stored rather than derived so no one pays an acos per ray
  // Cosine of the zenith angle at which the ground starts, for the observer the sky was baked
  // from. Rays below it end on the planet, so the sun is behind it and must not be drawn.
  float horizonMu;
  // Distance of that observer from the planet's centre, in km. A per-ray evaluation of the sky has
  // to stand exactly where the bake stood -- see sky_bruneton_radiance.h.slang.
  float observerRadius;
  // The clouds' (sky_clouds.h.slang), 1 and 1 without them. `cloudShadow` is the sun's transmittance
  // through the layer from the observer, the scene's origin: what the meter reads. It is not in
  // `radiance`: a light sample of the sun takes the clouds' shadow where it lands, from the shadow
  // image (cloudSunShare in gltf_pathtrace.slang). `sunlitGround` is the share of the distant land and
  // haze under the layer the sun still reaches, as the cloud bake measures it toward this sun (airMain
  // in sky_clouds_bake.slang): the cover's complement overhead, all but none through a broken deck at
  // a low sun's slant.
  float cloudShadow;
  float sunlitGround;
  // 1 while a drag's preview holds the clouds of the moment (SkyClouds' quarter-size image, at
  // CLOUD_PREVIEW_BINDING), 0 once the full image does.
  int cloudPreview;
  // The top of the clouds' highest layer, km above the surface, 0 without clouds, and the share of
  // the sky's light they leave on the air (1 clear): below the horizon the air in front of the ground
  // is under their shade, by that and `sunlitGround` (sky_clouds_air.h.slang).
  float cloudTop;
  float cloudSkyShare;
};

struct SceneFrameInfo
{
  float4x4 viewMatrix;      // View matrix
  float4x4 projInv;         // Inverse projection matrix
  float4x4 viewInv;         // Inverse view matrix
  float4x4 viewProjMatrix;  // View-projection matrix (P*V)
  float4x4 prevMVP;         // Previous view-projection matrix
  float2   jitter;          // DLSS sub-pixel jitter in pixel units, range [-0.5, +0.5]
  float2   imageSize;       // Render extent in pixels (W,H)
  int      flags = 0;       // Bit flags: see SceneFrameInfoFlags
  float4 envRotation;  // Environment orientation: unit quaternion (x, y, z, w), environment frame -> world. Lookups use quatRotateInverse(envRotation, worldDir)
  float         envBlur;             // Level of blur for the environment map (0.0: no blur, 1.0: full blur)
  float         envIntensity = 1.f;  // Environment intensity
  float3        backgroundColor;     // Background color when using solid background
  Visualization visualization             = Visualization::eRendered;  // Visualization mode
  float         infinitePlaneDistance     = 0;
  float3        infinitePlaneBaseColor    = float3(0.5, 0.5, 0.5);  // Default gray color
  float         infinitePlaneMetallic     = 0.0;                    // Default non-metallic
  float         infinitePlaneRoughness    = 0.5;                    // Default medium roughness
  float         shadowCatcherDarkenAmount = 0.0;  // Non-physical shadow darkening (precomputed from darkness slider)

  // Background field for the authored sky types. This is what a camera ray -- and a ray leaving a
  // mirror -- sees, and it is evaluated analytically per ray rather than read from the baked
  // lat-long: the bake is the *lighting* field and deliberately has no sun in it. Ignored unless
  // `envSkyType` names an authored type.
  //
  // The solar disk for the physical sky, or null for every other environment -- which is also how
  // a background pass knows whether to draw one. `sunDirection` is the world-space direction the
  // bake used, so the disk lands on the aureole the bake drew around it.
  SkySunDisk* sunDisk      = nullptr;
  float3      sunDirection = {0.0F, 1.0F, 0.0F};
  // 1 when the renderer supplies the sun as a light of its own: the sky has one and the scene
  // carries no marked sun light. It is not a glTF node -- nothing to save, nothing to undo, and no
  // scene edit as a side effect of picking an environment.
  int rendererSunLight = 0;
  // Index into the scene's punctual lights of the light the sky's sun *is*, or -1. Set only for a
  // light the scene marks as the sky's sun (gltf_environment_sky.hpp): that one is driven by the
  // sky, every other light is used exactly as authored.
  int sunLightIndex = -1;

  // `envSkyType` is a SkyType, or -1 when the environment is not an authored sky (a loaded HDR,
  // the procedural sky, or none) and the existing background paths apply instead.
  int              envSkyType = -1;
  SkyOmiParameters skyOmi;

  // Aerial perspective -- the air between the camera and a surface. Physical sky only; the gate
  // is `sunDisk != nullptr`, the same one the solar disk uses, so there is no second flag to
  // keep in step with the first. See shaders/aerial_perspective.h.slang.
  //
  // The scale is a *distance* multiplier on top of glTF's metre rather than an opacity: 2.0 says
  // the scene is twice as big, which stays inside the model. 0 switches the term off.
  //
  // **Appended, not inserted.** These two first went in beside `envSkyType`, where the host and
  // Slang disagreed about the resulting layout and every field read back as garbage -- silently,
  // because a wrong float is still a float. Nothing in this struct may be inserted into; new
  // fields go on the end.
  float aerialPerspectiveScale = 1.0F;
  // Aerial perspective's eye, in km above the planet surface. Not the bake's observer: that one is
  // fixed, while this follows the camera -- see GltfRenderer::aerialPerspectiveEyeAltitudeKm(). The
  // bake's is SkySunDisk::observerRadius.
  float observerAltitudeKm = 0.3F;

  // Illuminance of the gradient sky's sun, before `envIntensity`. Unused by every other
  // environment: the physical sky's sun is measured by the bake and read from `sunDisk`, and no
  // other sky has one.
  //
  // It is computed on the host rather than in makeSkySunLight() because the gradient sky's sun has
  // no radiance to convert -- `sunColor` says what the disk is drawn in, not how bright it is --
  // so the number comes from integrating the dome it sits under. See
  // GltfRenderer::gradientSunIlluminance(). (Appended, per the note above.)
  float3 skySunIlluminance = {0.0F, 0.0F, 0.0F};

  // Resolution of the environment's alias table -- HdrIbl::getSamplingGrid() -- which a baked sky
  // builds coarser than its image. environmentSample() needs it to turn an alias entry back into a
  // direction; the image's own size would be wrong. Two scalars rather than a uint2, so the host and
  // Slang cannot disagree about alignment after the float3 above. (Appended, per the note above.)
  uint envSamplingGridWidth  = 1;
  uint envSamplingGridHeight = 1;

  // The ground (infinite plane) fades out to nothing over the outer part of a disk around the
  // scene. Radius 0: no fade.
  float groundFadeCenterX = 0.0F;
  float groundFadeCenterZ = 0.0F;
  float groundFadeRadius  = 0.0F;

  // A multiplier on the sky's sun light, on top of what the sky measures: a host's sun dimmer (0
  // turns the sun off and leaves the sky). The disk drawn in the background keeps the sky's own.
  // (Appended, per the note above.)
  float skySunScale = 1.0F;

  // The environment's estimate for the path tracer's light pick (light_pick.h.slang), before
  // `envIntensity`: the integral of its lighting field's largest channel over the sphere
  // (HdrIbl::getIntegral) over 4, the irradiance a uniform sky of that power gives a surface facing
  // it. (Appended, per the note above.)
  float envPickEstimate = 0.0F;

  // Water on the ground (groundWaterAt in ground.h.slang): how much, 0 dry to 1 a sheet of it, and
  // how rippled its surface is, 0 still to 1. (Appended, per the note above.)
  float groundWater   = 0.0F;
  float groundRipples = 0.0F;

  // A body of water over the ground (groundHasWater): its surface this far above the plane, in
  // metres (0: none), and how high its waves run, 0 still to 1. The ripples are groundRipples.
  // (Appended, per the note above.)
  float waterDepth = 0.0F;
  float waterWaves = 0.0F;

  // The physical sky's clouds where the camera looks, at the screen's resolution: the view image at
  // CLOUD_VIEW_BINDING (SkyClouds::recordView, cloudViewUv). Its right, up and forward in the
  // environment's frame; right's and up's .w the tangents of its half-width and half-height, and
  // forward's 1 while it holds the clouds of the moment, 0 while the sky image alone does. The bake
  // fills this share of the image's texels across and down. (Appended, per the note above.)
  float4 cloudViewRight    = {0.0F, 0.0F, 0.0F, 0.0F};
  float4 cloudViewUp       = {0.0F, 0.0F, 0.0F, 0.0F};
  float4 cloudViewForward  = {0.0F, 0.0F, 0.0F, 0.0F};
  float  cloudViewUvScaleX = 1.0F;
  float  cloudViewUvScaleY = 1.0F;

  // Haze in the air (haze.h.slang): its extinction per metre (0: none), the anisotropy of its
  // Henyey-Greenstein phase function, and the sphere it fills, around the scene, its lights and the
  // eye (GltfRenderer::fillHaze). (Appended, per the note above.)
  float  hazeExtinction = 0.0F;
  float  hazeAnisotropy = 0.7F;
  float3 hazeCenter     = {0.0F, 0.0F, 0.0F};
  float  hazeRadius     = 0.0F;
  // A still's scatter point picks its light from this many candidates by what each gives the eye
  // (HazeReservoir); a moving frame takes one. (Appended, per the note above.)
  int hazeCandidates = 1;
  // Uneven haze (HazeShape): the floor's height, one over the height it thins over above it (0:
  // even through the height), how patchy it is (0 to 1) and the patches' frequency per metre.
  // (Appended, per the note above.)
  float hazeFloor        = 0.0F;
  float hazeInvHeight    = 0.0F;
  float hazePatches      = 0.0F;
  float hazeInvPatchSize = 0.0F;
  // ptEstimator, a debug check (light_pick.h.slang): which technique counts what both a light sample and
  // a BSDF ray reach, ESTIMATOR_MIS (0, both), _LIGHT (1) or _BSDF (2). (Appended, per the note above.)
  int estimator = 0;
  // The Time view (USE_PIXEL_TIME, pixel_time.h.slang): one over the clock ticks the frame's average
  // pixel took a sample, from a recent frame (0: none yet); the slots the frame's sums have, and where
  // they go (null: not summed). (Appended, per the note above.)
  float pixelTimeScale = 0.0F;
  int   pixelTimeSlots = 0;
  uint* pixelTime;
  // ptAerialSunShare: the air's sun-shadow ray is traced with a chance that keeps the noise it adds
  // under this share of the pixel (contribution_roulette.h.slang); 0 traces it on every path.
  // (Appended, per the note above.)
  float aerialSunShare = 0.05F;
  // What the Time view sums (pixel_time.h.slang kPixelTimeCount*): the clock (0), or a count of rays,
  // walk steps or shadow rays. (Appended, per the note above.)
  int pixelTimeCount = 0;
  // ptSkinSlab (debug): the thin preview lets light through thin skin to its far side
  // (skin_slab.h.slang); 0 sends it all back where it went in, as before. (Appended, per the note above.)
  int skinSlab = 1;
  // ptSkinSpread (debug): the thin preview takes the light of the camera's skin at points as far off as
  // the still's walk lets it out (skin_spread.h.slang); 0 takes it where it lands, as before.
  // (Appended, per the note above.)
  int skinSpread = 1;
  // ptSkinView (debug): the thin preview shades measured skin as the still's walk returns it, its colour
  // by the view's angle and its diffuse by the exit's Fresnel toward the light (skin_albedo.h.slang); 0
  // the map's Lambertian, as before. (Appended, per the note above.)
  int skinView = 1;
  // The moving view's bounce cache (USE_RADIANCE_CACHE, radiance_cache.h.slang): its table's slots (a
  // power of two), where it is (null: the frame has none), the width of the cells nearest the eye in
  // metres, and the frame's number, which says which pixels feed it. (Appended, per the note above; the
  // count before the pointer keeps it 8-byte aligned.)
  int   radianceCacheSlots = 0;
  uint* radianceCache;
  float radianceCacheCell  = 0.04F;
  int   radianceCacheFrame = 0;
  // ptRadianceCacheStill (debug): the frames of an accumulation read and feed the cache too, so its bias
  // can be measured on a converged image. (Appended, per the note above.)
  int radianceCacheStill = 0;
  // ptShadowShare: an accumulation's light sample traces its shadow ray with a chance that keeps the
  // noise it adds under this share of the pixel's mean so far (contribution_roulette.h.slang); 0 traces
  // every one. (Appended, per the note above.)
  float shadowRayShare = 0.0F;
  // The physical sky's clouds are a place, and the images that hold them were baked from one
  // (SkyCloudFollow, SkyCloudView in src/sky_cloud_view.hpp). Which sky image the frame takes
  // (cloudFront: 0 the sky bake's, 1 and 2 the follow images at CLOUD_FOLLOW_BINDING); where the
  // camera stands from where that image was baked, km in the environment's frame, with that place's
  // altitude in .w, km above the planet's surface (cloudEye); and where it stands from where the view
  // image was baked (cloudViewEye). A reader shifts each image by the clouds' distance
  // (cloudDirFromAnchor). (Appended, per the note above.)
  int    cloudFront   = 0;
  float4 cloudEye     = {0.0F, 0.0F, 0.0F, 0.0F};
  float4 cloudViewEye = {0.0F, 0.0F, 0.0F, 0.0F};
};

enum PathtracerFlags
{
  ePtUseDlss             = 1 << 0,
  ePtRrStillGuides       = 1 << 1,  // An accumulation frame on the DLSS variant averages DLSS-RR's guides into the still's denoiser's images (RrStillDenoiser)
  ePtFirstFrame          = 1 << 2,
  ePtShadowTransmission  = 1 << 3,  // Biased: shadow rays pass straight through transmissive surfaces
  ePtDenoiserGuides      = 1 << 4,  // Accumulate the albedo/normal guides (eDenoiserAlbedo, eDenoiserNormal)
  ePtProbeCameraMedium   = 1 << 5,  // Trace nothing but the camera's medium, into PathtracePushConstant::cameraMedium
  ePtScatterAsThin       = 1 << 6,  // Volumetric scatter shades as the thin-walled preview (no random walk)
  ePtVolumeDiffuseExit   = 1 << 7,  // A scatter walk gathers no light inside; it leaves the surface as a Lambertian point
  ePtClampIndirect       = 1 << 8,  // The firefly clamp cuts each indirect contribution on its own, never direct light
  ePtApproximateCaustics = 1 << 9,  // Every lobe after a rough bounce is at least as rough as the path (biased)
  ePtLobeSumPunctual     = 1 << 10,  // A punctual (DIRAC) light sample evaluates every BSDF lobe, each by its weight
  ePtLobeMixture         = 1 << 11,  // Every light sample does, and MIS uses the lobe-mixture pdf on both sides
  ePtSimplifyScatter     = 1 << 12,  // After a rough bounce, a scattering medium shades as the thin preview (no walk)
  ePtSobol               = 1 << 13,  // An accumulation frame draws its fixed sampling decisions from Owen-scrambled Sobol points
  ePtDiffuseEntry        = 1 << 14,  // Light enters a scatter walk through a Lambertian lobe, not by refraction
  ePtRegularize          = 1 << 15,  // Later lobes are at least as rough as the roughest so far, clear refraction aside (biased)
  ePtLightsVisible       = 1 << 16,  // Camera rays see sized lights (sizedLightHits); later rays always do
  ePtExitFresnel         = 1 << 17,  // A diffuse exit weighs its light by the boundary's Fresnel transmittance
  ePtLobeMixtureDirect   = 1 << 18,  // ePtLobeMixture only while the path is direct (pathRoughness at most DIRECT_PATH_ROUGHNESS)
  ePtWriteSelection      = 1 << 19,  // Trace a selection ray per pixel into eSelectImage (a still's frame 0; a DLSS frame only while something reads it)
  ePtAdaptiveStop        = 1 << 20,  // An accumulation's pixel stops once its mean has converged (eConvergence, adaptiveThreshold)
  ePtWalkGuiding         = 1 << 21,  // A scatter walk entered from a surface draws half its steps towards it (Dwivedi guiding, unbiased)
  ePtLightTree           = 1 << 22,  // The emitters' light tree is picked among the lights by distance (light_tree.h.slang); off: the one table by power
  ePtWriteNrMask         = 1 << 23,  // A DLSS frame writes the per-material DLSS-NR mask (eNrMask): only while DLSS-NR runs
  ePtLightReuse          = 1 << 24,  // The first hit's light sample takes the last frame's in as a candidate (lightReservoirs)
  ePtLightReuseHistory   = 1 << 25,  // ...which the last frame wrote for this view: read it (else start afresh)
  ePtLightReuseParity    = 1 << 26,  // ...this frame writes the second half of lightReservoirs and reads the first
  ePtSkinFaceOn          = 1 << 27,  // A medium made from its skin map (FV_materials_subsurface) returns the map lit head-on, not averaged over directions
  ePtStillPickAsScored   = 1 << 28,  // A still's light draws pick as they score where the light tree is on (lightPickScore), not by the walk
  ePtSmoothBsdfOnly      = 1 << 29,  // A vertex whose lobes are all mirrors (smoothVertex) leaves the lights its BSDF ray reaches to that ray: no shadow ray
  ePtInspectPixel        = 1 << 30,  // Trace nothing but the camera ray through pixel mouseCoord, into PathtracePushConstant::pixelInspect
};

// The participating medium the camera is in, for every camera path to start in. Written once per
// accumulation by the path tracer's own pipeline (ePtProbeCameraMedium, probeCameraMedium()).
struct CameraMedium
{
  float3 absorption;  // sigma_a, when `inside`
  int    inside;      // The camera is inside a KHR_materials_volume medium
  float3 scattering;  // sigma_s
  float  anisotropy;  // Henyey-Greenstein g
};


// What is at one pixel and how bright it is (the pixel readout, view.inspect): written by the path
// tracer's own pipeline (ePtInspectPixel, inspectPixel()) for the camera ray through the pixel's centre,
// the surface evaluated as the paths evaluate it at their first hit (textures, vertex colour, the thin
// preview and skin flags of the frame), before any regularisation. Every member a 16-byte row, so the
// host reads it as Slang writes it.
struct PixelInspect
{
  float4 position;   // xyz the hit, world space; w its distance from the camera (0: nothing hit)
  float4 normal;     // xyz the shading normal; w 1 when the camera sees the surface's front
  float4 baseColor;  // rgb linear, after textures and vertex colour; a the opacity
  float4 emission;   // rgb the emitted radiance (the renderer's units); a the transmission
  float4 surface;    // x metallic, y roughness (perceptual), z the index of refraction, w the specular weight
  float4 layers;     // x clearcoat, y its roughness, z the sheen (its colour's largest channel), w the thickness (0: thin-walled)
  float4 medium;     // rgb the scattering medium's colour (subsurface; 0: none), a the diffuse transmission
  float4 radiance;   // rgb what the frame's image holds at the pixel (a still's mean so far); a the samples in it
  float4 uv;         // xy the texture coordinates; z the render pixel's x, w its y
  int4   ids;        // x the instance slot hit (-1: nothing, -2: the ground, -3: the water), y the primitive, z the material, w 1 on a moving (DLSS) frame
};

// Push constant
struct PathtracePushConstant
{
  int             maxDepth              = 5;       // Maximum depth of the ray
  int             frameCount            = 0;       // Frame number
  float           fireflyClampThreshold = 10.f;    // Firefly clamp threshold
  float           texGradScale          = 0.f;     // Ray-footprint gradient scale
  float           fireflyClampStill     = 0.f;     // Clamp for the frames that accumulate a still (0: fireflyClampThreshold)
  int             hairBudget            = 0;       // Fibre events a path takes before they count as surface bounces (also keeps mouseCoord 8-byte aligned)
  int             frameSample           = 0;       // Which of the frame's samples this pass traces, 0 the first (a pass traces one a pixel)
  int             totalSamples          = 0;       // Total samples accumulated so far
  float           focalDistance         = 0.0f;    // Focal distance for depth of field
  float           aperture              = 0.0f;    // Aperture for depth of field
  int             flags                 = 0;       // Bit flags: see PathtracerFlags
  float           pixelAngle            = 0.0f;    // Angular size of one pixel (radians) for ray-cone footprint LOD
  int             seed                  = 0;       // Random sequence: another value gives independent noise
  int             guideSamples          = 0;       // Samples in the denoiser guides (0: replace them)
  float2          mouseCoord            = {0, 0};  // Mouse coordinates (use for debug)
  uint2           renderSize = {0, 0};          // Traced region: DLSS's input size on its frames, smaller than the view
  int             bladeCount    = 0;            // Aperture blades: below 3 a round aperture (aperture.h.slang)
  float           bladeRotation = 0.0f;         // Angle of the first blade's corner, radians from the camera's right
  float           hairRouletteDlss = 1.0f;      // DLSS frames: a path continues from a fibre at most with this... (1: off)
  int             hairRouletteFrom = 1;         // ...once it has taken this many fibre events (two fields keep the pointers 8-byte aligned)
  int             hairCutDlss      = 3;         // DLSS frames: a path through hair ends at this fibre event, its light sample its last (0: off)...
  float           hairCutTail      = 4.0f;      // ...which also carries this times the lock's colour over 4 pi, for the light it would have scattered on
  float           adaptiveThreshold = 0.09f;    // ePtAdaptiveStop: a pixel stops once its mean's standard error is at most this times the root of its luminance...
  int             adaptiveMin       = 32;       // ...and it has at least this many samples
  float           hairResponsivity  = 0.0f;     // DLSS frames: the responsivity mask where the first hit is hair (0: no mask)
  int             glassBudget       = 8;        // Clear-glass events a path takes before they count as surface bounces (also keeps the table 16-byte aligned)
  // ePtAdaptiveStop: how much more noise a pixel may keep where the tone mapper flattens its brightness,
  // a stop apart from mid grey's luminance in the scene (adaptiveRelaxBase; 0: none). adaptive_relax.hpp
  float4          adaptiveRelax[3];
  float           adaptiveRelaxBase = 0.0f;
  int             hitSignalSlots    = 0;        // hitSignal's materials; keeps the pointers 8-byte aligned
  int             lightCandidates      = 1;     // Light samples a surface vertex draws to keep one of (light_ris.h.slang; 1: the one it traces)...
  int             lightCandidatesDepth = 0;     // ...at its first this many surface vertices (0: at all of them); two fields keep the pointers aligned
  float           lightReuseCap        = 1.0F;  // ePtLightReuse: a record stands for at most this many times the frame's own candidates...
  int             lightReusePad        = 0;     // ...(and keeps the pointers 8-byte aligned)
  uint*           hitSignal;                    // first hits counted per material slot and bin (hit_signal.h.slang; null: not counting)
  uint*           adaptiveActive;               // ePtAdaptiveStop: the frame counts the pixels still sampling here (null: not counted)
  SceneFrameInfo* frameInfo;                    // Camera info (incl. SceneFrameInfo::jitter when DLSS is active)
  GltfScene*      gltfScene;                    // GLTF scene
  float4x4*       prevRenderNodeObjectToWorld;  // #DLSS instance motion: previous-frame objectToWorld per render node
  float3**        prevVertexPositions;  // #DLSS deformation motion: previous-frame positions per render primitive (null entry = none)
  CameraMedium*   cameraMedium;                 // The medium camera paths start in; null when the scene has none
  uint4*          lightReservoirs;              // ePtLightReuse: two halves of renderSize records, two uint4 a pixel (light_ris.h.slang)
  PixelInspect*   pixelInspect;                 // ePtInspectPixel: where the pixel readout goes (the last 8 of 256 bytes)
};

struct DenoiserHandoffPushConstant
{
  float weight = 0.0F;      // Share of the second image (the hand-off: the denoised one, handoffWeight)
  float limit  = 65504.0F;  // Largest magnitude written: a half float's for an RGBA16F output
};

// The bounce cache's pass after a frame (radiance_cache_resolve.slang): each cell's sums of the frame
// folded into what it keeps, at most `maxWeight` of throughput received
struct RadianceCacheResolvePushConstant
{
  uint* table;
  uint  slots     = 0;
  float maxWeight = 0.0F;
};

struct SilhouettePushConstant
{
  float3 color;
  uint   selectionBitMaskWordCount;  // Number of uint32 words in selection bitmask (for bounds check)
};

NAMESPACE_SHADERIO_END()

#endif  // HOST_DEVICE_H
