/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HAIR_BSDF_H
#define HAIR_BSDF_H 1

#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/functions.h.slang"

// A fibre BSDF for an OmniHair ribbon. The ribbon stands for a lock, so the response is Chiang's
// near-field model (the practical one: R, TT, TRT, and the remaining orders) averaged over the
// fibre offset h. Longitudinal and azimuthal roughness are OmniHair's β_m and β_n. The strand
// direction is the ribbon's bitangent, per hit, so a wavy bob does not need one angle for the
// whole material.
//
// The value handed back is the quantity the path tracer stores as f·cos: Chiang's sum of M_p N_p A_p,
// which is what a path integral uses for a fibre (the extra cosine of a flat sheet is not applied
// again). Nothing else: the light a lock scatters between its fibres is the path's, bounce after
// bounce (the path tracer's hair budget, PathtracePushConstant::hairBudget). A Lambert lobe on both sides used to
// stand in for it, 35% of every hit: it cut the highlight and every order to 65% and, tinted c^0.45
// for a lock of colour c, lifted dark hair to grey (9% diffuse over a 5% lock).

// Compiled as C++ by the unit tests (tests/test_hair_bsdf.cpp), as the nvshaders headers are
#ifdef __cplusplus
#define HAIR_UNROLL
#else
#define HAIR_UNROLL [unroll]
#endif

static const float HAIR_ETA   = 1.55F;
static const float HAIR_ALPHA = 0.034906585F;  // 2 degrees, the cuticle tilt
static const int   HAIR_PMAX  = 3;
static const float HAIR_SQRT_PI_OVER_8 = 0.626657068F;

struct HairHit
{
  float3 sigma;
  float  betaM;
  float  betaN;
  float3 strand;  // along the ribbon
  float3 phiY;    // across the ribbon, perpendicular to the strand
  float3 phiZ;    // completes the strand frame (phi = atan2(dot(w, phiZ), dot(w, phiY)))
  float3 normal;  // shading normal: which side of the ribbon a direction is on
  float  specular;  // scales the reflecting orders (R, TRT); 1 is the fibre's own Fresnel
  float  h;         // where across the fibre this hit is, -1..1: one offset per hit, shared by its light
                    // sample, its BSDF sample and the pdf the next hit's MIS reads (pbrt, Unreal)
};

struct HairResponse
{
  float3 f;
  float  pdf;
};

float hairSafeSqrt(float x)
{
  return sqrt(max(x, 0.0F));
}

float hairLuma(float3 c)
{
  return dot(c, float3(0.2126F, 0.7152F, 0.0722F));
}

float hairI0(float x)
{
  float val  = 0.0F;
  float x2i  = 1.0F;
  float fact = 1.0F;
  float i4   = 1.0F;
  HAIR_UNROLL
  for(int i = 0; i < 10; i++)
  {
    if(i > 1)
      fact *= float(i);
    val += x2i / (i4 * fact * fact);
    x2i *= x * x;
    i4 *= 4.0F;
  }
  return val;
}

float hairLogI0(float x)
{
  if(x > 12.0F)
    return x + 0.5F * (-log(2.0F * M_PI) + log(1.0F / x) + 1.0F / (8.0F * x));
  return log(max(hairI0(x), 1e-20F));
}

// Longitudinal scattering M_p. v is the variance of this lobe.
float hairMp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
{
  v             = max(v, 1e-5F);
  float a       = cosThetaI * cosThetaO / v;
  float b       = sinThetaI * sinThetaO / v;
  if(v <= 0.1F)
    return exp(hairLogI0(a) - b - 1.0F / v + 0.693147F + log(0.5F / v));
  return exp(-b) * hairI0(a) / (sinh(1.0F / v) * 2.0F * v);
}

float hairFresnel(float cosThetaI, float eta)
{
  cosThetaI       = saturate(abs(cosThetaI));
  float sin2ThetaT = (1.0F / (eta * eta)) * (1.0F - cosThetaI * cosThetaI);
  if(sin2ThetaT >= 1.0F)
    return 1.0F;
  float cosThetaT = sqrt(1.0F - sin2ThetaT);
  float rPar      = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
  float rPerp     = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
  return 0.5F * (rPar * rPar + rPerp * rPerp);
}

float hairLogistic(float x, float s)
{
  x       = abs(x);
  float e = exp(-x / s);
  return e / (s * (1.0F + e) * (1.0F + e));
}

// The logistic's mass on [-pi, pi], in closed form: CDF(pi) - CDF(-pi) = (1 - e) / (1 + e), e = exp(-pi/s)
float hairLogisticMassPi(float s)
{
  const float e = exp(-M_PI / s);
  return (1.0F - e) / (1.0F + e);
}

// The logistic trimmed to [-pi, pi], the only range the azimuth uses
float hairTrimmedLogistic(float x, float s)
{
  return hairLogistic(x, s) / hairLogisticMassPi(s);
}

float hairSampleTrimmedLogistic(float u, float s)
{
  const float k = hairLogisticMassPi(s);
  const float x = -s * log(1.0F / (u * k + 0.5F * (1.0F - k)) - 1.0F);
  return clamp(x, -M_PI, M_PI);
}

float hairWrapPi(float dphi)
{
  // One or two turns is all a logistic sample produces.
  if(dphi > M_PI)
    dphi -= M_TWO_PI;
  if(dphi < -M_PI)
    dphi += M_TWO_PI;
  if(dphi > M_PI)
    dphi -= M_TWO_PI;
  if(dphi < -M_PI)
    dphi += M_TWO_PI;
  return dphi;
}

float hairNp(float phi, int p, float s, float gammaO, float gammaT)
{
  float dphi = hairWrapPi(phi - (2.0F * float(p) * gammaT - 2.0F * gammaO + float(p) * M_PI));
  return hairTrimmedLogistic(dphi, s);
}

struct HairScales
{
  float sinA[3];
  float cosA[3];
  float v[4];
  float s;
};

HairScales hairScales(float betaM, float betaN)
{
  HairScales sc;
  betaM = clamp(betaM, 0.0F, 1.0F);
  betaN = clamp(betaN, 0.0F, 1.0F);
  float bm2 = betaM * betaM;
  float bn2 = betaN * betaN;
  sc.v[0]   = pow(0.726F * betaM + 0.812F * bm2 + 3.7F * pow(betaM, 20.0F), 2.0F);
  sc.v[0]   = max(sc.v[0], 1e-5F);
  sc.v[1]   = 0.25F * sc.v[0];
  sc.v[2]   = 4.0F * sc.v[0];
  sc.v[3]   = sc.v[2];
  sc.s      = max(HAIR_SQRT_PI_OVER_8 * (0.265F * betaN + 1.194F * bn2 + 5.372F * pow(betaN, 22.0F)), 1e-3F);

  float sa  = sin(HAIR_ALPHA);
  float ca  = cos(HAIR_ALPHA);
  sc.sinA[0] = sa;
  sc.cosA[0] = ca;
  sc.sinA[1] = 2.0F * ca * sa;
  sc.cosA[1] = ca * ca - sa * sa;
  sc.sinA[2] = 2.0F * sc.cosA[1] * sc.sinA[1];
  sc.cosA[2] = sc.cosA[1] * sc.cosA[1] - sc.sinA[1] * sc.sinA[1];
  return sc;
}

void hairTilt(int p, float sinThetaO, float cosThetaO, HairScales sc, NVSHADERS_OUT_TYPE(float) sinOp, NVSHADERS_OUT_TYPE(float) cosOp)
{
  if(p == 0)
  {
    sinOp = sinThetaO * sc.cosA[1] - cosThetaO * sc.sinA[1];
    cosOp = cosThetaO * sc.cosA[1] + sinThetaO * sc.sinA[1];
  }
  else if(p == 1)
  {
    sinOp = sinThetaO * sc.cosA[0] + cosThetaO * sc.sinA[0];
    cosOp = cosThetaO * sc.cosA[0] - sinThetaO * sc.sinA[0];
  }
  else if(p == 2)
  {
    sinOp = sinThetaO * sc.cosA[2] + cosThetaO * sc.sinA[2];
    cosOp = cosThetaO * sc.cosA[2] - sinThetaO * sc.sinA[2];
  }
  else
  {
    sinOp = sinThetaO;
    cosOp = cosThetaO;
  }
  cosOp = abs(cosOp);
}

// Attenuation of one fibre at offset h, and the pdf of picking each order. The pdf uses luminance,
// so a red dye is not sampled as if its energy were in the green channel. `specular` scales the
// reflecting orders; the transmitted order is the fibre's, not the slider's.
struct HairOrders
{
  float3 ap[4];
  float  pdf[4];
  float  gammaO;
  float  gammaT;
  float  sinThetaO;
  float  cosThetaO;
};

HairOrders hairOrders(float h, float sinThetaO, float cosThetaO, float3 sigma, float specular)
{
  HairOrders o;
  o.sinThetaO = sinThetaO;
  o.cosThetaO = max(cosThetaO, 1e-4F);
  float sinThetaT = sinThetaO / HAIR_ETA;
  float cosThetaT = max(hairSafeSqrt(1.0F - sinThetaT * sinThetaT), 1e-4F);
  float etap      = sqrt(max(HAIR_ETA * HAIR_ETA - sinThetaO * sinThetaO, 0.0F)) / o.cosThetaO;
  float sinGammaT = clamp(h / max(etap, 1e-4F), -1.0F, 1.0F);
  float cosGammaT = hairSafeSqrt(1.0F - sinGammaT * sinGammaT);
  o.gammaO        = asin(clamp(h, -1.0F, 1.0F));
  o.gammaT        = asin(sinGammaT);

  float3 T = exp(-sigma * (2.0F * cosGammaT / cosThetaT));
  float  F = hairFresnel(o.cosThetaO * hairSafeSqrt(1.0F - h * h), HAIR_ETA);
  float  spec = saturate(specular);
  o.ap[0]  = float3(F) * spec;
  o.ap[1]  = (1.0F - F) * (1.0F - F) * T;
  o.ap[2]  = o.ap[1] * T * F * spec;
  o.ap[3]  = o.ap[2] * F * T / max(float3(1.0F) - T * F, float3(1e-4F));

  float sum = 0.0F;
  HAIR_UNROLL
  for(int i = 0; i < 4; i++)
    sum += hairLuma(o.ap[i]);
  // Each lobe's probability is its share of the weight the sum is of, so the four sum to 1 and the
  // remainder the sampler gives lobe 3 is pdf[3]. With no weight at all (no specular, opaque
  // fibre) the sampler's remainder is all of lobe 3, and the density says so.
  HAIR_UNROLL
  for(int j = 0; j < 4; j++)
    o.pdf[j] = sum > 1e-6F ? hairLuma(o.ap[j]) / sum : (j == HAIR_PMAX ? 1.0F : 0.0F);
  return o;
}

// Fibre response at one offset, in world space. f is M N A summed, pdf is the solid-angle density
// of sampling that direction at this offset (pbrt's pdf, which pairs with this f).
HairResponse hairFibreAt(float h, float3 wo, float3 wi, HairHit hair, HairScales sc)
{
  float sinO = dot(wo, hair.strand);
  float sinI = dot(wi, hair.strand);
  sinO       = clamp(sinO, -1.0F, 1.0F);
  sinI       = clamp(sinI, -1.0F, 1.0F);
  float cosO = hairSafeSqrt(1.0F - sinO * sinO);
  float cosI = hairSafeSqrt(1.0F - sinI * sinI);

  // phi around the strand. The zero is arbitrary; only the difference is used.
  float phiO = atan2(dot(wo, hair.phiZ), dot(wo, hair.phiY));
  float phiI = atan2(dot(wi, hair.phiZ), dot(wi, hair.phiY));
  float phi  = phiI - phiO;

  HairOrders ord = hairOrders(h, sinO, max(cosO, 1e-4F), hair.sigma, hair.specular);
  HairResponse r;
  r.f   = float3(0.0F);
  r.pdf = 0.0F;
  HAIR_UNROLL
  for(int p = 0; p < HAIR_PMAX; p++)
  {
    float sinOp, cosOp;
    hairTilt(p, sinO, max(cosO, 1e-4F), sc, sinOp, cosOp);
    float mp = hairMp(cosI, cosOp, sinI, sinOp, sc.v[p]);
    float np = hairNp(phi, p, sc.s, ord.gammaO, ord.gammaT);
    r.f += mp * np * ord.ap[p];
    r.pdf += mp * ord.pdf[p] * np;
  }
  float mpR = hairMp(cosI, max(cosO, 1e-4F), sinI, sinO, sc.v[HAIR_PMAX]);
  r.f += mpR * ord.ap[HAIR_PMAX] * (0.5F * M_1_PI);
  r.pdf += mpR * ord.pdf[HAIR_PMAX] * (0.5F * M_1_PI);
  return r;
}

HairResponse hairCombined(float3 wo, float3 wi, HairHit hair)
{
  return hairFibreAt(hair.h, wo, wi, hair, hairScales(hair.betaM, hair.betaN));
}

HairHit hairHitZero()
{
  HairHit hair;
  hair.sigma  = float3(0.0F);
  hair.betaM  = 0.0F;
  hair.betaN  = 0.0F;
  hair.strand = float3(0.0F, 1.0F, 0.0F);
  hair.phiY   = float3(1.0F, 0.0F, 0.0F);
  hair.phiZ   = float3(0.0F, 0.0F, 1.0F);
  hair.normal = float3(0.0F, 0.0F, 1.0F);
  hair.specular = 1.0F;
  hair.h        = 0.0F;
  return hair;
}

// Chiang et al. 2016's fit of the colour a lock of fibres shows against their absorption, at
// azimuthal roughness b: colour = exp(-sqrt(σ_a) · fit).
float hairColorFit(float b)
{
  b        = saturate(b);
  float b2 = b * b;
  float b3 = b2 * b;
  return 5.969F - 0.215F * b + 2.532F * b2 - 10.73F * b3 + 5.574F * b2 * b2 + 0.245F * b2 * b3;
}

// The absorption that shows `color`: the fit inverted (Unreal's HairColorToAbsorption). Clamped,
// since painted root shading near black inverts to an absorption no light gets through, and white
// to none at all.
float3 hairColorToAbsorption(float3 color, float b)
{
  float3 l = log(clamp(color, float3(0.02F), float3(0.98F))) / hairColorFit(b);
  return l * l;
}

// OmniHair's root to tip blend (daz_3d/basic.mdl): Schlick's fast bias, and the gain built on it
float hairBias(float t, float bias)
{
  return t / ((1.0F / bias - 2.0F) * (1.0F - t) + 1.0F);
}

float hairBlend(float along, float bias, float gain)
{
  float t = hairBias(along, bias);
  return t < 0.5F ? hairBias(t * 2.0F, gain) * 0.5F : hairBias(t * 2.0F - 1.0F, 1.0F - gain) * 0.5F + 0.5F;
}

// OmniHairBase.mdl's absorption at a point of the strand
float3 hairDyedAbsorption(GltfShadeMaterial mat, float along)
{
  float3 color = lerp(mat.hairRootColor, mat.hairTipColor, hairBlend(along, mat.hairBlendBias, mat.hairBlendGain));
  float3 l     = log(clamp(color, float3(0.001F), float3(1.0F))) / hairColorFit(mat.hairAzimuthal);
  return lerp(l * l, mat.hairMelaninSigma, mat.hairMelaninMix);
}

// The colour a lock of these fibres shows (hairColorFit): what its multiply scattered light returns
float3 hairLockColor(HairHit hair)
{
  return exp(-sqrt(hair.sigma) * hairColorFit(hair.betaN));
}

// `baseColor`: the colour the hit shows, for a card's absorption (hairFromColor)
bool makeHairHit(GltfShadeMaterial mat, float2 uv, float3 geoBitangent, float3 geoTangent, float3 normal, float3 baseColor, NVSHADERS_OUT_TYPE(HairHit) hair)
{
  hair = hairHitZero();
  // A card's strand is across its measured anisotropy direction, which turns the UV tangent by
  // (cos t, -sin t) in (tangent, bitangent): the strand is (sin t, cos t), UV v at t = 0
  if(mat.hairStrandSin != 0.0F || mat.hairStrandCos != 0.0F)
  {
    const float3 t = geoTangent;
    geoTangent     = t * mat.hairStrandCos - geoBitangent * mat.hairStrandSin;
    geoBitangent   = t * mat.hairStrandSin + geoBitangent * mat.hairStrandCos;
  }
  float len2 = dot(geoBitangent, geoBitangent);
  if(len2 < 1e-8F)
    return false;
  float3 strand = geoBitangent / sqrt(len2);
  // A card's normal map tilts its strands: they lie in the plane of the shading normal, so a bump
  // along a strand moves its highlight along it and the band breaks up strand by strand. (Turning a
  // fibre about its own axis changes nothing; only the tilt along it shows.)
  if(mat.hairFromColor > 0.0F)
  {
    const float3 tilted = strand - normal * dot(strand, normal);
    if(dot(tilted, tilted) > 1e-8F)
      strand = normalize(tilted);
  }
  float3 across = geoTangent - strand * dot(geoTangent, strand);
  if(dot(across, across) < 1e-8F)
    across = normal - strand * dot(normal, strand);
  if(dot(across, across) < 1e-8F)
    return false;
  across = normalize(across);

  float along = saturate(abs(uv.y - mat.hairRootV));
  hair.sigma  = mat.hairFromColor > 0.0F ? hairColorToAbsorption(baseColor, mat.hairAzimuthal)
                : mat.hairDyed > 0.0F    ? hairDyedAbsorption(mat, along)
                                         : lerp(mat.hairRootSigma, mat.hairTipSigma, along);
  hair.betaM     = mat.hairLongitudinal;
  hair.betaN     = mat.hairAzimuthal;
  hair.strand    = strand;
  hair.phiY      = across;
  hair.phiZ      = normalize(cross(strand, across));
  hair.normal    = normal;
  hair.specular  = mat.hairSpecular;
  return true;
}

void hairBsdfEvaluate(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, HairHit hair)
{
  data.bsdf_diffuse = float3(0.0F);
  data.bsdf_glossy  = float3(0.0F);
  data.pdf          = 0.0F;
  HairResponse r    = hairCombined(data.k1, data.k2, hair);
  if(r.pdf <= 0.0F || any(isnan(r.f)) || isnan(r.pdf))
    return;
  // Every order is a fibre lobe: the glossy slot
  data.bsdf_glossy = r.f;
  data.pdf         = r.pdf;
}

float3 hairSampleDirection(float h, float3 wo, HairHit hair, HairScales sc, float uLobe, float uAzimuth, float uLong, float uPhi, NVSHADERS_OUT_TYPE(int) lobe)
{
  float sinO = clamp(dot(wo, hair.strand), -1.0F, 1.0F);
  float cosO = max(hairSafeSqrt(1.0F - sinO * sinO), 1e-4F);
  HairOrders ord = hairOrders(h, sinO, cosO, hair.sigma, hair.specular);

  lobe = HAIR_PMAX;
  float u = saturate(uLobe);
  for(int p = 0; p < HAIR_PMAX; p++)
  {
    if(u < ord.pdf[p])
    {
      lobe = p;
      break;
    }
    u -= ord.pdf[p];
  }

  float sinOp, cosOp;
  hairTilt(lobe, sinO, cosO, sc, sinOp, cosOp);
  float v = sc.v[min(lobe, HAIR_PMAX)];
  uLong   = max(uLong, 1e-5F);
  float cosTheta = 1.0F + v * log(uLong + (1.0F - uLong) * exp(-2.0F / v));
  cosTheta       = clamp(cosTheta, -1.0F, 1.0F);
  float sinTheta = hairSafeSqrt(1.0F - cosTheta * cosTheta);
  float cosPhi   = cos(M_TWO_PI * uPhi);
  float sinI     = clamp(-cosTheta * sinOp + sinTheta * cosPhi * cosOp, -1.0F, 1.0F);
  float cosI     = max(hairSafeSqrt(1.0F - sinI * sinI), 0.0F);

  float dphi;
  if(lobe < HAIR_PMAX)
    dphi = (2.0F * float(lobe) * ord.gammaT - 2.0F * ord.gammaO + float(lobe) * M_PI) + hairSampleTrimmedLogistic(saturate(uAzimuth), sc.s);
  else
    dphi = M_TWO_PI * uAzimuth;

  float phiO = atan2(dot(wo, hair.phiZ), dot(wo, hair.phiY));
  float phiI = phiO + dphi;
  return sinI * hair.strand + (cosI * cos(phiI)) * hair.phiY + (cosI * sin(phiI)) * hair.phiZ;
}

void hairBsdfSample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, HairHit hair, float uAzimuth)
{
  data.pdf           = 0.0F;
  data.bsdf_over_pdf = float3(0.0F);
  data.event_type    = BSDF_EVENT_ABSORB;

  // At the hit's offset, which hairCombined evaluates the density at
  HairScales sc   = hairScales(hair.betaM, hair.betaN);
  int        lobe = 0;
  data.k2         = normalize(hairSampleDirection(hair.h, data.k1, hair, sc, data.xi.x, uAzimuth, data.xi.y, data.xi.z, lobe));

  HairResponse r = hairCombined(data.k1, data.k2, hair);
  if(r.pdf <= 1e-8F || any(isnan(r.f)) || isnan(r.pdf) || any(isnan(data.k2)))
  {
    data.event_type = BSDF_EVENT_ABSORB;
    data.pdf        = 0.0F;
    return;
  }
  data.pdf           = r.pdf;
  data.bsdf_over_pdf = r.f / r.pdf;
  data.event_type    = dot(data.k2, hair.normal) < 0.0F ? BSDF_EVENT_GLOSSY_TRANSMISSION : BSDF_EVENT_GLOSSY_REFLECTION;
}

#endif  // HAIR_BSDF_H
