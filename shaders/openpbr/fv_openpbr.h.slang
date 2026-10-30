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

  // decision M7: DAZ's tight lobe isn't a coat, it is a lobe beside the specular (fvOpenPbrLobeShare)
  in.coat_weight    = mat.clearcoatIsLobe > 0.0F ? 0.0F : mat.clearcoat;
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
  const float3 Nc        = in.coat_weight > 0.0F ? fvOpenPbrFacing(mat.Nc, V, away) : N;
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

// The share of the light DAZ's tight specular lobe takes at this view (0: no such lobe), by its albedo
// as today's closure has it: the rest of the closure lies under it, scaled by what is left, which is
// how OpenPBR 1.2's haze layers. It is drawn by today's GGX lobe, the one the path tracer tested.
float fvOpenPbrLobeShare(PbrMaterial mat, float3 V)
{
  return mat.clearcoatIsLobe > 0.0F && mat.clearcoat > 0.0F ? bsdfClearcoatAlbedo(mat, abs(dot(V, mat.Nc))) : 0.0F;
}

// The closure at a hit. One per vertex: the light sample and the BSDF sample below share it.
OpenPBR_PreparedBsdf fvOpenPbrPrepare(PbrMaterial mat, float3 V)
{
  return openpbr_prepare(fvOpenPbrInputs(mat, V), float3(1.0F), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, V);
}

// The closure says which lobe drew the direction, as the path tracer's events do
int fvOpenPbrEvent(OpenPBR_BsdfLobeType type)
{
  const bool transmitted = (type & OpenPBR_BsdfLobeTypeTransmission) != 0;
  const bool diffuse     = (type & OpenPBR_BsdfLobeTypeDiffuse) != 0;
  return diffuse ? (transmitted ? BSDF_EVENT_DIFFUSE_TRANSMISSION : BSDF_EVENT_DIFFUSE_REFLECTION) :
                   (transmitted ? BSDF_EVENT_GLOSSY_TRANSMISSION : BSDF_EVENT_GLOSSY_REFLECTION);
}

// bsdfEvaluateAll's outputs: the whole closure toward k2, and the pdf of its own sampler (a mixture of
// its lobes, as ePtLobeMixture's)
void fvOpenPbrEvaluate(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, OpenPBR_PreparedBsdf prepared, PbrMaterial mat)
{
  const OpenPBR_DiffuseSpecular value = openpbr_eval(prepared, data.k2);
  const float                   share = fvOpenPbrLobeShare(mat, data.k1);
  data.bsdf_diffuse = (1.0F - share) * value.diffuse * fvOpenPbrDiffuseTerms(mat, data.k2) * mat.occlusion;
  data.bsdf_glossy  = (1.0F - share) * value.specular * mat.occlusion;
  data.pdf          = (1.0F - share) * openpbr_pdf(prepared, data.k2);
  if(share > 0.0F)
  {
    BsdfEvaluateData lobe = data;
    bsdfEvaluateLobe(lobe, mat, LOBE_CLEARCOAT_REFLECTION, mat.baseColor);
    data.bsdf_glossy += share * lobe.bsdf_glossy * mat.occlusion;
    data.pdf += share * lobe.pdf;
  }
}

// bsdfSampleCore's outputs. A sample with no direction is an absorption.
void fvOpenPbrSample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, OpenPBR_PreparedBsdf prepared, PbrMaterial mat)
{
  const float share = fvOpenPbrLobeShare(mat, data.k1);
  if(share > 0.0F)
  {
    // One of the two by its share, then the mixture's value over the mixture's pdf at what it drew,
    // whichever drew it. xi.z chose, so it is stretched back over [0,1] for the one chosen.
    const float3 xi   = data.xi;
    const bool   lobe = xi.z < share;
    int          event = BSDF_EVENT_GLOSSY_REFLECTION;
    if(lobe)
    {
      BsdfSampleData drawn = data;
      drawn.xi.z           = xi.z / share;
      bsdfSampleLobe(drawn, mat, LOBE_CLEARCOAT_REFLECTION, mat.baseColor);
      data.k2 = drawn.k2;
      event   = drawn.event_type;
    }
    else
    {
      float3                  L;
      OpenPBR_DiffuseSpecular weight;
      float                   pdf  = 0.0F;
      OpenPBR_BsdfLobeType    type = OpenPBR_BsdfLobeTypeNone;
      openpbr_sample(prepared, float3(xi.x, xi.y, (xi.z - share) / (1.0F - share)), L, weight, pdf, type);
      event   = pdf > 0.0F ? fvOpenPbrEvent(type) : BSDF_EVENT_ABSORB;
      data.k2 = pdf > 0.0F ? normalize(L) : data.k1;
    }
    if(event == BSDF_EVENT_ABSORB)
    {
      data.pdf           = 0.0F;
      data.bsdf_over_pdf = float3(0.0F);
      data.event_type    = BSDF_EVENT_ABSORB;
      return;
    }
    BsdfEvaluateData at;
    at.k1 = data.k1;
    at.k2 = data.k2;
    at.xi = xi;
    fvOpenPbrEvaluate(at, prepared, mat);
    data.pdf           = at.pdf;
    data.bsdf_over_pdf = at.pdf > 0.0F ? (at.bsdf_diffuse + at.bsdf_glossy) / at.pdf : float3(0.0F);
    data.event_type    = at.pdf > 0.0F ? event : BSDF_EVENT_ABSORB;
    return;
  }

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
  data.event_type    = fvOpenPbrEvent(type);
}

#endif  // FV_OPENPBR_H_SLANG
