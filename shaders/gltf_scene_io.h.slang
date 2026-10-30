/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*-------------------------------------------------------------------------------------------------
Local fork of nvpro_core2/nvshaders/gltf_scene_io.h.slang.

Keeps the include guard DH_SCN_DESC_H identical to the upstream file so any transitive
include of the nvpro_core2 version (e.g. from nvshaders/light_contrib.h.slang) is a no-op
as long as this file is #included first in the translation unit.

Adds MAT_EXT_* preprocessor gates around every glTF extension's fields so the
GltfShadeMaterial struct can be shrunk for scenes that don't use those extensions.
With all flags = 1 (the default) the struct layout is byte-identical to upstream.
-------------------------------------------------------------------------------------------------*/

#ifndef DH_SCN_DESC_H
#define DH_SCN_DESC_H 1

#include "nvshaders/slang_types.h"
#include "gltf_material_config.h"

NAMESPACE_SHADERIO_BEGIN()

// This is the GLTF Node structure, but flattened
struct GltfRenderNode
{
  float4x4 objectToWorld;
  float4x4 worldToObject;
  int      materialID;
  int      renderPrimID;
};

// This is all the information about a vertex buffer
struct VertexBuffers
{
  float3* positions;
  float3* normals;
  uint*   colors;  // packed as 4 uint8_t
  float4* tangents;
  // TEXCOORD_0 and TEXCOORD_1 as two members, not a `float2* texCoords[2]` array: AMD's Vulkan
  // driver crashes compiling a shader that loads a struct holding an array of buffer pointers.
  // Pick a channel at runtime with vertexTexCoords() (gltf_vertex_access.h.slang).
  float2* texCoord0;
  float2* texCoord1;
};

// This is the GLTF Primitive structure
struct GltfRenderPrimitive
{
  uint3*        indices;
  VertexBuffers vertexBuffer;
};

// One emissive triangle referenced for area-light sampling (next-event estimation). Stored as a
// reference (render node + primitive + triangle) rather than baked world-space vertices, so the
// sampler reads live geometry/transforms and the list depends only on materials (rebuilt on the
// material dirty flag; see docs/RENDERING_ARCHITECTURE.md).
struct EmissiveTriangle
{
  float3 radiance;      // constant emitted radiance (emissiveFactor * KHR_materials_emissive_strength)
  float  aliasProb;     // Vose alias table: probability of keeping this bin (else take aliasIndex). O(1) select.
  uint   renderNodeID;  // index into GltfScene.renderNodes
  uint   renderPrimID;  // index into GltfScene.renderPrimitives
  uint   triangleID;    // triangle within the primitive
  uint   doubleSided;   // 1 = emit from both faces
  uint   aliasIndex;    // Vose alias table: fallback bin (with emitter groups, within the triangle's group)
};

// A node of the emitters' light tree (light_tree.h.slang). A leaf is the emissive triangles of one
// drawn surface (an instance's prim), sampled as one light: a triangle among them by its own weight
// (EmissiveTriangle::aliasProb and aliasIndex, within the leaf). An inner node bounds its two children.
// GltfScene::emissiveGroups holds the tree's cut first, at most EMISSIVE_GROUP_MAX nodes that are
// picked among the environment and the punctual lights by an estimate of what they give the shading
// point (their weight over the distance to their bounds); then their subtrees. A scene with no more
// emitters than that has leaves alone. GltfScene::numEmissiveGroups (the cut) 0: one table over every
// triangle, which takes a fixed share of the light samples.
struct EmissiveGroup
{
  float3 center;        // the bounding sphere of its triangles, in world space when the table was built
  float  radius;
  float3 axis;          // the cone holding its triangles' geometric normals (the side a one-sided triangle emits to)
  float  cosSpread;     // cos of the cone's half angle; -1: every way (a double-sided emitter, a bulb)
  float  weight;        // its triangles' selection weights summed
  uint   first;         // a leaf's triangles: emissiveTriangles[first, first + count)
  uint   count;
  uint   renderNodeID;  // the render node a leaf's triangles are drawn by, for a BSDF ray that hits one
  int    left;          // an inner node's children; -1 for a leaf
  int    right;
  int    parent;        // -1 for a node of the cut
  int    pad;
};
// The cut: at most this many nodes compete with the lights, each at every light sample. A node of the
// cut is judged by its own place, and a walk below one by bounds that hold several lights: 2560x1440,
// 64 samples, the noise at a cut of 16 / 32 / 64+: 1TEST (65 emitters) 0.047 / 0.0364 / 0.0347 at 7.5 /
// 7.7 / 7.8 s; 120 panels 32 / 64 / 120: 0.0142 / 0.0139 / 0.0135 at 4.8 / 4.9 / 5.1 s (2026-10-30)
static const int EMISSIVE_GROUP_MAX = 64;


/*-------------------------------------------------------------------------------------------------
Common structures used for lights other than environment lighting.
-------------------------------------------------------------------------------------------------*/

// Light type
enum LightType
{
  eLightTypeNone,
  eLightTypeDirectional,
  eLightTypeSpot,
  eLightTypePoint,
};


// Light
// Defaults produce a dark directional light pointing roughly down and away; match what
// defaultLight() used to return. normalize()/radians() resolve to glm on host and to
// Slang builtins on device (via `using namespace glm` in slang_types.h).
struct GltfLight
{
  float3 direction = normalize(float3(0.0F, -0.7F, -0.7F));  // Direction of the light
  int    type      = LightType::eLightTypeDirectional;       // Type of the light

  float3 position = float3(5.0F, 5.0F, 5.0F);  // Position of the light
  float  radius   = 1.0F;                      // Radius of the light

  float3 color = float3(1.0F);  // Color of the light
  float intensity = 0.0F;  // illuminance (lm/m2) for directional lights, luminous intensity (lm/sr) for positional lights

  float angularSizeOrInvRange = radians(0.53F);  // angular size for directional lights, 1/range for spot and point lights
  float innerAngle = radians(10.0F);             // Inner angle of the light
  float outerAngle = radians(30.0F);             // Outer angle of the light
  int   iesProfile = -1;  // EXT_lights_ies: index into GltfScene.iesProfiles (stride iesProfileSampleCount), -1 = none
};


// alphaMode
enum AlphaMode
{
  eAlphaModeOpaque,
  eAlphaModeMask,
  eAlphaModeBlend,
};

// PBR shading model selector. Defaults to metallic-roughness; set to specular-glossiness
// when KHR_materials_pbrSpecularGlossiness is present (and MAT_EXT_SPECULAR_GLOSSINESS is on).
enum PbrModel
{
  ePbrModelMetallicRoughness,
  ePbrModelSpecularGlossiness,
};

// GltfTextureInfo::flags bits.
enum TextureInfoFlagBits
{
  // The source image format carries exactly two color components (BC5, R8G8, EAC_R11G11, ...), so
  // red and green are authored data but the sampled blue channel is a synthetic 0. Set by SceneVk
  // from the decoded VkFormat. Consumers that need the third channel must reconstruct it from the
  // first two -- see the normal map handling in gltf_material_eval.h.slang.
  eTexInfoTwoChannelSource = 1 << 0,
  // FV_texture_flip_green on the texture info: a tangent-space normal map stored with Y down
  // (DirectX convention, or rows stored bottom-up as Figure Viewer's texture cache keeps them).
  // The decoded normal's Y is negated, so one image serves both conventions without a second copy.
  eTexInfoFlipGreen = 1 << 1,
};

// Texture info
// Total 32 bytes with MAT_EXT_TEXTURE_TRANSFORM = 1, 8 bytes when off.
struct GltfTextureInfo
{
#if MAT_EXT_TEXTURE_TRANSFORM
  float3x2 uvTransform = float3x2(1);  // KHR_texture_transform (24 bytes, 3x2 matrix)
#endif
  int     index        = -1;  // 4 bytes  glTF texture index -> images[] (eTextures)
  int16_t samplerIndex = 0;   // 2 bytes  resolved GPU sampler slot -> samplers[] (eSamplers); 0 = default sampler
  int8_t  texCoord     = 0;   // 1 byte   texcoord set: 0 or 1
  uint8_t flags        = 0;   // 1 byte   TextureInfoFlagBits
};

/*-------------------------------------------------------------------------------------------------
GltfShadeMaterial

Field layout is organized as:
  1. Base (always-on) float/int block at the top.
  2. Extension float/int blocks, each wrapped in #if MAT_EXT_*.
  3. Base uint16_t texture slots.
  4. Extension uint16_t texture slots, each wrapped in #if MAT_EXT_*.
  5. Trailing uint64_t _pad to keep sizeof a multiple of 8 and lift struct alignment to 8
     for every MAT_EXT_* combination (enables LDG.64 for aligned field pairs in the BDA array).

This ensures the base block's offsets are stable regardless of which MAT_EXT_* flags are
enabled — e.g. occlusionStrength, doubleSided, and the base texture slots always sit at
the same position. Extension blocks are collapsed out of the struct when gated, shrinking
the total size. The layout is NOT identical to upstream nvpro_core2; that's a deliberate
trade-off for forkability (see docs/developer.md > Material System).
-------------------------------------------------------------------------------------------------*/
struct GltfShadeMaterial
{
  float4 pbrBaseColorFactor = float4(1);
  float3 emissiveFactor     = float3(0);
  float  normalTextureScale = 1;

  float pbrRoughnessFactor = 1;
  float pbrMetallicFactor  = 1;
  int   alphaMode          = AlphaMode::eAlphaModeOpaque;
  float alphaCutoff        = 0.5;

  float occlusionStrength = 1;
  int   doubleSided       = 0;
  // An emitter's weight among the light samples is emissiveFactor (strength included) times this:
  // the emissive texture's mean luminance (SceneVk, from the decoded image), 1 without a texture or
  // where the mean is unknown, 0 for an all-black texture (not a light). The path tracer's hit side
  // reads it to recover the pick the light sample made (emitterSelectionRadiance).
  float emissiveLightScale = 1;

#if MAT_EXT_VOLUME
  float3 attenuationColor = float3(1);
#endif

#if MAT_EXT_IOR
  float ior = 1.5;
#endif

#if MAT_EXT_TRANSMISSION
  float transmissionFactor = 0;
#endif

#if MAT_EXT_VOLUME
  float thicknessFactor     = 0;              // KHR_materials_volume
  float attenuationDistance = 3.402823e+38f;  // +Infinity per spec default
#endif

#if MAT_EXT_CLEARCOAT
  float clearcoatFactor = 0;
#endif

#if MAT_EXT_SPECULAR
  float3 specularColorFactor = float3(1);
#endif

#if MAT_EXT_CLEARCOAT
  float clearcoatRoughness          = 0;
  float clearcoatNormalTextureScale = 1;  // clearcoatNormalTexture.scale
  float clearcoatDualLobe           = 0;  // 1: the slot holds DAZ's tight specular lobe, not a coat (fvDualLobe)
  float _clearcoatPad               = 0;  // keeps the row a multiple of 8 with no padding: rows are compared byte by byte
#endif

#if MAT_EXT_SPECULAR
  float specularFactor = 0;
#endif

#if MAT_EXT_UNLIT
  int unlit = 0;
#endif

#if MAT_EXT_IRIDESCENCE
  float iridescenceFactor           = 0;
  float iridescenceThicknessMinimum = 100;
  float iridescenceThicknessMaximum = 400;
  float iridescenceIor              = 1.3f;
#endif

#if MAT_EXT_ANISOTROPY
  float2 anisotropyRotation = float2(0);
#endif

#if MAT_EXT_SHEEN
  float3 sheenColorFactor = float3(0);
#endif

#if MAT_EXT_ANISOTROPY
  float anisotropyStrength = 0;
#endif

#if MAT_EXT_SHEEN
  float sheenRoughnessFactor = 0;
#endif

#if MAT_EXT_DISPERSION
  float dispersion = 0;
#endif

#if MAT_EXT_SPECULAR_GLOSSINESS
  int pbrModel = PbrModel::ePbrModelMetallicRoughness;

  float4 pbrDiffuseFactor    = float4(1);
  float3 pbrSpecularFactor   = float3(1);
  float  pbrGlossinessFactor = 1;
#endif

#if MAT_EXT_DIFFUSE_TRANSMISSION
  float3 diffuseTransmissionColorFactor = float3(1);
  float  diffuseTransmissionFactor      = 0;
#endif

#if MAT_EXT_RETROREFLECTION
  float retroreflectionFactor = 0;
#endif

#if MAT_EXT_SCATTER
  float3 multiscatterColorFactor = float3(1);
  float  scatterStrengthFactor   = 0;
  float  scatterAnisotropy       = 0;
  float3 scatterDistance         = float3(0);  // FV_materials_subsurface; 0: the medium as authored
#endif

#if MAT_EXT_DETAIL
  float detailNormalScale     = 1;  // detailNormalTexture.scale
  float detailWeightFactor    = 0;
  float detailRoughnessFactor = 1;
#endif

#if MAT_EXT_HAIR
  // FV_materials_hair. hairFactor is 0 unless the extension is present: a ribbon fibre, shaded
  // with a hair BSDF instead of the glTF lobes. Absorption is Chiang's σ_a; the hit's along-UV
  // blends root to tip. rootV is the UV.y of the root (1 when the build flipped V).
  float  hairFactor        = 0;
  float  hairLongitudinal  = 0.2f;  // β_m, 0..1
  float  hairAzimuthal     = 0.2f;  // β_n, 0..1
  float  hairRootFade      = 0;    // along-UV width of the root opacity ramp; 0 = none
  float  hairRootV         = 0;    // UV.y at the root
  float  hairSpecular      = 1.f;  // scales the reflecting orders; 1 is the fibre's Fresnel
  float3 hairRootSigma     = float3(0);
  float3 hairTipSigma      = float3(0);
  // Hair cards: 1 takes each hit's absorption from its base colour instead of root and tip. The
  // strand rotation's sine and cosine turn the UV tangent to across the strand as anisotropy does;
  // both 0, the strand is the UV bitangent.
  float hairFromColor   = 0;
  float hairStrandSin   = 0;
  float hairStrandCos   = 0;
#endif

  // Texture infos (uint16_t, 2 bytes each). Index 0 is the sentinel "no texture" slot
  // (textureInfos[0] is reserved as invalid on the host), so all texture members default to 0.
  // Base (always-on) texture slots come first so their offsets within the texture section
  // are stable across MAT_EXT_* flag combinations.
  uint16_t pbrBaseColorTexture         = 0;
  uint16_t normalTexture               = 0;
  uint16_t pbrMetallicRoughnessTexture = 0;
  uint16_t emissiveTexture             = 0;
  uint16_t occlusionTexture            = 0;

#if MAT_EXT_TRANSMISSION
  uint16_t transmissionTexture = 0;
#endif

#if MAT_EXT_VOLUME
  uint16_t thicknessTexture = 0;
#endif

#if MAT_EXT_CLEARCOAT
  uint16_t clearcoatTexture          = 0;
  uint16_t clearcoatRoughnessTexture = 0;
  uint16_t clearcoatNormalTexture    = 0;
#endif

#if MAT_EXT_SPECULAR
  uint16_t specularTexture      = 0;
  uint16_t specularColorTexture = 0;
#endif
#if MAT_EXT_IRIDESCENCE
  uint16_t iridescenceTexture          = 0;
  uint16_t iridescenceThicknessTexture = 0;
#endif

#if MAT_EXT_ANISOTROPY
  uint16_t anisotropyTexture = 0;
#endif

#if MAT_EXT_SHEEN
  uint16_t sheenColorTexture     = 0;
  uint16_t sheenRoughnessTexture = 0;
#endif

#if MAT_EXT_SPECULAR_GLOSSINESS
  uint16_t pbrDiffuseTexture            = 0;
  uint16_t pbrSpecularGlossinessTexture = 0;
#endif

#if MAT_EXT_DIFFUSE_TRANSMISSION
  uint16_t diffuseTransmissionTexture      = 0;
  uint16_t diffuseTransmissionColorTexture = 0;
#endif

#if MAT_EXT_RETROREFLECTION
  uint16_t retroreflectionTexture = 0;
#endif

#if MAT_EXT_SCATTER
  uint16_t scatterStrengthTexture   = 0;
  uint16_t multiscatterColorTexture = 0;
#endif

#if MAT_EXT_DETAIL
  uint16_t detailNormalTexture    = 0;
  uint16_t detailWeightTexture    = 0;
  uint16_t detailRoughnessTexture = 0;
  uint16_t _detailPad             = 0;  // keeps the texture block a whole number of 4-byte words
#endif

#if MAT_EXT_DLSS_NR
  float4 nrMask = float4(1, 1, 1, 1);  // EXT_DLSS_NR: per-material NR control mask (R=intensity, G=localTone, B=localStructure, A=globalTone)
#endif

  // Trailing 8-byte explicit pad. Two reasons for uint64_t (rather than just uint32_t):
  //  1. It guarantees sizeof(GltfShadeMaterial) is a multiple of 8 for every MAT_EXT_*
  //     combination (struct alignment follows max member alignment, which uint64_t makes 8).
  //  2. With the base BDA pointer already 8-aligned (Vulkan guarantees this for storage
  //     buffers), an 8-multiple stride keeps every `materials[i]` 8-aligned, letting the
  //     driver emit LDG.64 for adjacent 4-byte field pairs across all elements.
  uint64_t _pad = 0;
};

// The scene description is a pointer to the material, render node and render primitive
// The buffers are all arrays of the above structures
struct GltfScene
{
  GltfShadeMaterial*   materials;
  GltfTextureInfo*     textureInfos;
  GltfRenderNode*      renderNodes;
  GltfRenderPrimitive* renderPrimitives;
  GltfLight*           lights;
  int                  numLights;  // number of punctual lights
  EmissiveTriangle*    emissiveTriangles;
  int                  numEmissiveTriangles;  // number of sampled emissive triangles (area lights)
  float                emissiveTotalWeight;   // sum of per-triangle selection weights (area * defensive luminance)
  float                emissiveMeanLum;       // area-weighted mean emitter luminance (for defensive sampling)
  int                  numEmissiveGroups;     // the light tree's cut; 0: the triangles are one table (EmissiveGroup)
  EmissiveGroup*       emissiveGroups;        // the cut, then the rest of the tree
  int*                 emissiveLeafOfNode;    // [render node]: the leaf whose triangles it draws, -1 for none
  int                  numEmissiveLeafOfNode;

  // EXT_lights_ies: flattened [profile][sample] table, sample = normalized candela (peak = 1) at
  // vertical angle sample/(iesProfileSampleCount-1) * 180 degrees off the light's photometric axis,
  // azimuthally averaged (see docs/developer.md > Lighting). GltfLight.iesProfile indexes the profile.
  float* iesProfiles;
  int    iesProfileSampleCount;
};

// Defensive emissive selection: blend power-weighted selection (area * luminance) with area-uniform
// selection so a dim emitter is never starved to a near-zero selection probability. Each emitter's
// luminance is pulled toward the scene's area-weighted mean: lum' = (1-f)*lum + f*meanLum.
// 0 = pure power-weighted, 1 = pure area-uniform. Host and device must use the same value.
static const float EMISSIVE_DEFENSIVE_FRACTION = 0.5f;


NAMESPACE_SHADERIO_END()

#endif
