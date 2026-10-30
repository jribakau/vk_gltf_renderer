/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The path tracer's OpenPBR closure (phase M2 of the app's docs/material-model-plan.md): today's
// resolved material (evaluateMaterial's PbrMaterial) turned into OpenPBR's parameters, and Adobe's
// closure (adobe/, see its NOTICE.md) behind the path tracer's BSDF interface. A polyglot, so the unit
// tests and the sweep compile it as C++ (tests/test_bsdf_lobes.cpp, figure/tools/closure_probe).
//
// Include it after bsdf_functions.h.slang.

#ifndef FV_OPENPBR_H_SLANG
#define FV_OPENPBR_H_SLANG

#ifndef __cplusplus
#define OPENPBR_LANGUAGE_TARGET_SLANG 1
// Adobe's Slang layer says `unsigned short`, which our Slang doesn't parse
#define OPENPBR_UINT16 uint16_t
#endif
#define OPENPBR_USE_CUSTOM_SATURATE 1
#include "adobe/openpbr.h"

// KHR_materials_clearcoat's coat has this index, and OpenPBR's default is another
#define FV_OPENPBR_COAT_IOR 1.5F

// The IOR of the material itself, whichever side of it the path is on (evaluateMaterial swaps them)
float fvOpenPbrIor(PbrMaterial mat)
{
  return mat.ior2 > 1.0F || mat.ior1 <= 1.0F ? mat.ior2 : mat.ior1;
}

// A frame about `N`, with the view above it: a normal map can bend it past the view, which OpenPBR
// reads as the path being inside the surface. `away`: the path is inside, and the normal faces away.
float3 fvOpenPbrFacing(float3 N, float3 V, bool away)
{
  const float NoV = dot(N, V);
  if(away)
    return NoV < 0.0F ? N : -N;
  return NoV >= 1e-3F ? N : normalize(N + (1e-3F - NoV) * V);
}

OpenPBR_Basis fvOpenPbrBasis(float3 N, float3 T, float3 B)
{
  // T orthogonal to N, B keeping its side (the anisotropy's direction and the normal map's handedness)
  const float3 t = normalize(T - N * dot(N, T));
  float3       b = cross(N, t);
  if(dot(b, B) < 0.0F)
    b = -b;
  OpenPBR_Basis basis;
  basis.t = t;
  basis.b = b;
  basis.n = N;
  return basis;
}

// Today's material as OpenPBR's: the published KHR equivalences. `V` toward the viewer.
OpenPBR_ResolvedInputs fvOpenPbrInputs(PbrMaterial mat, float3 V)
{
  OpenPBR_ResolvedInputs in = openpbr_make_default_resolved_inputs();

  in.base_color     = mat.baseColor;
  in.base_metalness = mat.metallic;

  // Transmission with no medium of OpenPBR's: the path tracer's own walk carries the medium
  // (makeVolumeMedium), and the crossing's tint is the surface's, as it is on the KHR lobes
  in.transmission_weight = mat.transmission;
  in.transmission_color  = mat.baseColor * mat.transmissionTint;
  in.transmission_depth  = 0.0F;
  if(mat.dispersion > 0.0F)
  {
    in.transmission_dispersion_scale       = 1.0F;
    in.transmission_dispersion_abbe_number = 20.0F / mat.dispersion;
  }

  // Thin-walled: nothing bends, and both sides shade alike. A surface whose only way through is
  // diffuse is thin to the closure too, whether or not a walk starts behind it (ePtDiffuseEntry).
  const bool thin = mat.thickness <= 0.0F || mat.transmission <= 0.0F || mat.diffuseTransmissionFactor > 0.0F;
  in.geometry_thin_walled = thin;

  // KHR_materials_diffuse_transmission's factor of the diffuse bucket goes through in its colour,
  // the rest reflects in the base colour: the subsurface part of a thin wall, all forward
  in.subsurface_weight             = mat.diffuseTransmissionFactor;
  in.subsurface_color              = mat.diffuseTransmissionColorFactor;
  in.subsurface_scatter_anisotropy = 1.0F;

  // alpha_x = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_y = (1 - a) alpha_x (OpenPBR's anisotropy), with
  // x along the tangent evaluateMaterial turned to the direction of the stretch
  const float alphaX = max(mat.roughness.x, mat.roughness.y);
  const float alphaY = min(mat.roughness.x, mat.roughness.y);
  const float aniso  = alphaX > 0.0F ? 1.0F - alphaY / alphaX : 0.0F;
  in.specular_roughness            = sqrt(alphaX / sqrt(2.0F / (1.0F + (1.0F - aniso) * (1.0F - aniso))));
  in.specular_roughness_anisotropy = aniso;
  // A metal's specular_color is OpenPBR's edge tint, which glTF's metal doesn't have
  in.specular_weight = lerp(mat.specular, 1.0F, mat.metallic);
  in.specular_color  = lerp(mat.specularColor, float3(1.0F), mat.metallic);
  in.specular_ior    = fvOpenPbrIor(mat);

  in.coat_weight    = mat.clearcoat;
  in.coat_roughness = mat.clearcoatRoughness;
  in.coat_ior       = FV_OPENPBR_COAT_IOR;
  // decision M10: physical, the base darkens under its coat
  in.coat_darkening = 1.0F;

  const float fuzz = max(mat.sheenColor.x, max(mat.sheenColor.y, mat.sheenColor.z));
  in.fuzz_weight    = min(fuzz, 1.0F);
  in.fuzz_color     = fuzz > 0.0F ? mat.sheenColor / fuzz : float3(1.0F);
  in.fuzz_roughness = mat.sheenRoughness;

  in.thin_film_weight    = mat.iridescence;
  in.thin_film_thickness = mat.iridescenceThickness * 0.001F;  // nm to OpenPBR's microns
  in.thin_film_ior       = mat.iridescenceIor;

  // Inside only a surface that refracts has an inside: evaluateMaterial has swapped its IORs
  const bool   away = !thin && mat.ior1 > 1.0F;
  const float3 N    = fvOpenPbrFacing(mat.N, V, away);
  in.geometry_basis = fvOpenPbrBasis(N, mat.T, mat.B);
  // the coat's normal is its own map's, or the base's as bent
  const float3 Nc        = mat.clearcoat > 0.0F ? fvOpenPbrFacing(mat.Nc, V, away) : N;
  in.geometry_coat_basis = fvOpenPbrBasis(Nc, mat.T, mat.B);
  return in;
}

// Whether the closure takes this material: retroreflection has no lobe in OpenPBR
bool fvOpenPbrTakes(PbrMaterial mat)
{
  return mat.retroreflection <= 0.0F;
}

// What the diffuse reflection carries toward `L`, on top of OpenPBR's: the terminator's fade on a bumped
// lobe, and the light a medium has evened out (diffuseSoftening), as brdf_diffuse_eval has them. The
// pdf stays the closure's. Light through the surface has neither, as the path tracer's diffuse
// transmission doesn't.
float3 fvOpenPbrDiffuseTerms(PbrMaterial mat, float3 L)
{
  const float NoL = dot(mat.N, L);
  if(NoL <= 0.0F)
    return float3(1.0F);
  float3       terms = float3(shadowTerminatorTerm(L, mat.N, mat.Ns));
  const float3 soft  = mat.diffuseSoftening;
  if(soft.x > 0.0F || soft.y > 0.0F || soft.z > 0.0F)
  {
    // the softened part's cosine is the smooth normal's, kept within what a fireball wouldn't notice
    const float ratio = min(max(dot(mat.Ns, L), 0.0F) / max(NoL, 1e-4F), 4.0F);
    terms *= float3(1.0F) + soft * (ratio - 1.0F);
  }
  return terms;
}

// The closure at a hit. One per vertex: the light sample and the BSDF sample below share it.
OpenPBR_PreparedBsdf fvOpenPbrPrepare(PbrMaterial mat, float3 V)
{
  return openpbr_prepare(fvOpenPbrInputs(mat, V), float3(1.0F), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, V);
}

// bsdfEvaluateAll's outputs: the whole closure toward k2, and the pdf of its own sampler (a mixture of
// its lobes, as ePtLobeMixture's)
void fvOpenPbrEvaluate(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, OpenPBR_PreparedBsdf prepared, PbrMaterial mat)
{
  const OpenPBR_DiffuseSpecular value = openpbr_eval(prepared, data.k2);
  data.bsdf_diffuse = value.diffuse * fvOpenPbrDiffuseTerms(mat, data.k2) * mat.occlusion;
  data.bsdf_glossy  = value.specular * mat.occlusion;
  data.pdf          = openpbr_pdf(prepared, data.k2);
}

// bsdfSampleCore's outputs. A sample with no direction is an absorption.
void fvOpenPbrSample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, OpenPBR_PreparedBsdf prepared, PbrMaterial mat)
{
  float3                  L;
  OpenPBR_DiffuseSpecular weight;
  float                   pdf = 0.0F;
  OpenPBR_BsdfLobeType    type = OpenPBR_BsdfLobeTypeNone;
  openpbr_sample(prepared, data.xi, L, weight, pdf, type);
  if(!(pdf > 0.0F))
  {
    data.pdf           = 0.0F;
    data.bsdf_over_pdf = float3(0.0F);
    data.event_type    = BSDF_EVENT_ABSORB;
    return;
  }
  data.k2            = normalize(L);
  data.pdf           = pdf;
  data.bsdf_over_pdf = (weight.diffuse * fvOpenPbrDiffuseTerms(mat, data.k2) + weight.specular) * mat.occlusion;
  // the closure says which lobe drew the direction, as the path tracer's events do
  const bool transmitted = (type & OpenPBR_BsdfLobeTypeTransmission) != 0;
  const bool diffuse     = (type & OpenPBR_BsdfLobeTypeDiffuse) != 0;
  data.event_type        = diffuse ? (transmitted ? BSDF_EVENT_DIFFUSE_TRANSMISSION : BSDF_EVENT_DIFFUSE_REFLECTION) :
                                     (transmitted ? BSDF_EVENT_GLOSSY_TRANSMISSION : BSDF_EVENT_GLOSSY_REFLECTION);
}

#endif  // FV_OPENPBR_H_SLANG
