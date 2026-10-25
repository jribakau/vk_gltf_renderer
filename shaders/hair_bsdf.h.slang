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
// again). A fixed share is a Lambert lobe on both sides of the ribbon, tinted by the same albedo
// the melanin fit shows, so a stack of ribbons still passes light where the lobes are narrow.

static const float HAIR_ETA   = 1.55;
static const float HAIR_ALPHA = 0.034906585;  // 2 degrees, the cuticle tilt
static const int   HAIR_PMAX  = 3;
static const float HAIR_MS    = 0.35;  // share of the response that is the residual multiple scatter
static const float HAIR_SQRT_PI_OVER_8 = 0.626657068;

struct HairHit
{
  float3 sigma;
  float3 albedo;
  float  betaM;
  float  betaN;
  float3 strand;  // along the ribbon
  float3 phiY;    // across the ribbon, perpendicular to the strand
  float3 phiZ;    // completes the strand frame (phi = atan2(dot(w, phiZ), dot(w, phiY)))
  float3 normal;  // shading normal, for the residual scatter
  float3 msT;
  float3 msB;
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
  return sqrt(max(x, 0.0));
}

float hairLuma(float3 c)
{
  return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float hairI0(float x)
{
  float val  = 0.0;
  float x2i  = 1.0;
  float fact = 1.0;
  float i4   = 1.0;
  [unroll]
  for(int i = 0; i < 10; i++)
  {
    if(i > 1)
      fact *= float(i);
    val += x2i / (i4 * fact * fact);
    x2i *= x * x;
    i4 *= 4.0;
  }
  return val;
}

float hairLogI0(float x)
{
  if(x > 12.0)
    return x + 0.5 * (-log(2.0 * M_PI) + log(1.0 / x) + 1.0 / (8.0 * x));
  return log(max(hairI0(x), 1e-20));
}

// Longitudinal scattering M_p. v is the variance of this lobe.
float hairMp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
{
  v             = max(v, 1e-5);
  float a       = cosThetaI * cosThetaO / v;
  float b       = sinThetaI * sinThetaO / v;
  if(v <= 0.1)
    return exp(hairLogI0(a) - b - 1.0 / v + 0.693147 + log(0.5 / v));
  return exp(-b) * hairI0(a) / (sinh(1.0 / v) * 2.0 * v);
}

float hairFresnel(float cosThetaI, float eta)
{
  cosThetaI       = saturate(abs(cosThetaI));
  float sin2ThetaT = (1.0 / (eta * eta)) * (1.0 - cosThetaI * cosThetaI);
  if(sin2ThetaT >= 1.0)
    return 1.0;
  float cosThetaT = sqrt(1.0 - sin2ThetaT);
  float rPar      = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
  float rPerp     = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
  return 0.5 * (rPar * rPar + rPerp * rPerp);
}

float hairLogistic(float x, float s)
{
  x       = abs(x);
  float e = exp(-x / s);
  return e / (s * (1.0 + e) * (1.0 + e));
}

// The logistic's mass on [-pi, pi], in closed form: CDF(pi) - CDF(-pi) = (1 - e) / (1 + e), e = exp(-pi/s)
float hairLogisticMassPi(float s)
{
  const float e = exp(-M_PI / s);
  return (1.0 - e) / (1.0 + e);
}

// The logistic trimmed to [-pi, pi], the only range the azimuth uses
float hairTrimmedLogistic(float x, float s)
{
  return hairLogistic(x, s) / hairLogisticMassPi(s);
}

float hairSampleTrimmedLogistic(float u, float s)
{
  const float k = hairLogisticMassPi(s);
  const float x = -s * log(1.0 / (u * k + 0.5 * (1.0 - k)) - 1.0);
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
  float dphi = hairWrapPi(phi - (2.0 * float(p) * gammaT - 2.0 * gammaO + float(p) * M_PI));
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
  betaM = clamp(betaM, 0.0, 1.0);
  betaN = clamp(betaN, 0.0, 1.0);
  float bm2 = betaM * betaM;
  float bn2 = betaN * betaN;
  sc.v[0]   = pow(0.726 * betaM + 0.812 * bm2 + 3.7 * pow(betaM, 20.0), 2.0);
  sc.v[0]   = max(sc.v[0], 1e-5);
  sc.v[1]   = 0.25 * sc.v[0];
  sc.v[2]   = 4.0 * sc.v[0];
  sc.v[3]   = sc.v[2];
  sc.s      = max(HAIR_SQRT_PI_OVER_8 * (0.265 * betaN + 1.194 * bn2 + 5.372 * pow(betaN, 22.0)), 1e-3);

  float sa  = sin(HAIR_ALPHA);
  float ca  = cos(HAIR_ALPHA);
  sc.sinA[0] = sa;
  sc.cosA[0] = ca;
  sc.sinA[1] = 2.0 * ca * sa;
  sc.cosA[1] = ca * ca - sa * sa;
  sc.sinA[2] = 2.0 * sc.cosA[1] * sc.sinA[1];
  sc.cosA[2] = sc.cosA[1] * sc.cosA[1] - sc.sinA[1] * sc.sinA[1];
  return sc;
}

void hairTilt(int p, float sinThetaO, float cosThetaO, HairScales sc, out float sinOp, out float cosOp)
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
  o.cosThetaO = max(cosThetaO, 1e-4);
  float sinThetaT = sinThetaO / HAIR_ETA;
  float cosThetaT = max(hairSafeSqrt(1.0 - sinThetaT * sinThetaT), 1e-4);
  float etap      = sqrt(max(HAIR_ETA * HAIR_ETA - sinThetaO * sinThetaO, 0.0)) / o.cosThetaO;
  float sinGammaT = clamp(h / max(etap, 1e-4), -1.0, 1.0);
  float cosGammaT = hairSafeSqrt(1.0 - sinGammaT * sinGammaT);
  o.gammaO        = asin(clamp(h, -1.0, 1.0));
  o.gammaT        = asin(sinGammaT);

  float3 T = exp(-sigma * (2.0 * cosGammaT / cosThetaT));
  float  F = hairFresnel(o.cosThetaO * hairSafeSqrt(1.0 - h * h), HAIR_ETA);
  float  spec = saturate(specular);
  o.ap[0]  = float3(F) * spec;
  o.ap[1]  = (1.0 - F) * (1.0 - F) * T;
  o.ap[2]  = o.ap[1] * T * F * spec;
  o.ap[3]  = o.ap[2] * F * T / max(float3(1.0) - T * F, float3(1e-4));

  float sum = 0.0;
  [unroll]
  for(int i = 0; i < 4; i++)
    sum += hairLuma(o.ap[i]);
  // Each lobe's probability is its share of the weight the sum is of, so the four sum to 1 and the
  // remainder the sampler gives lobe 3 is pdf[3]. With no weight at all (no specular, opaque
  // fibre) the sampler's remainder is all of lobe 3, and the density says so.
  [unroll]
  for(int j = 0; j < 4; j++)
    o.pdf[j] = sum > 1e-6 ? hairLuma(o.ap[j]) / sum : (j == HAIR_PMAX ? 1.0 : 0.0);
  return o;
}

// Fibre response at one offset, in world space. f is M N A summed, pdf is the solid-angle density
// of sampling that direction at this offset (pbrt's pdf, which pairs with this f).
HairResponse hairFibreAt(float h, float3 wo, float3 wi, HairHit hair, HairScales sc)
{
  float sinO = dot(wo, hair.strand);
  float sinI = dot(wi, hair.strand);
  sinO       = clamp(sinO, -1.0, 1.0);
  sinI       = clamp(sinI, -1.0, 1.0);
  float cosO = hairSafeSqrt(1.0 - sinO * sinO);
  float cosI = hairSafeSqrt(1.0 - sinI * sinI);

  // phi around the strand. The zero is arbitrary; only the difference is used.
  float phiO = atan2(dot(wo, hair.phiZ), dot(wo, hair.phiY));
  float phiI = atan2(dot(wi, hair.phiZ), dot(wi, hair.phiY));
  float phi  = phiI - phiO;

  HairOrders ord = hairOrders(h, sinO, max(cosO, 1e-4), hair.sigma, hair.specular);
  HairResponse r;
  r.f   = float3(0.0);
  r.pdf = 0.0;
  [unroll]
  for(int p = 0; p < HAIR_PMAX; p++)
  {
    float sinOp, cosOp;
    hairTilt(p, sinO, max(cosO, 1e-4), sc, sinOp, cosOp);
    float mp = hairMp(cosI, cosOp, sinI, sinOp, sc.v[p]);
    float np = hairNp(phi, p, sc.s, ord.gammaO, ord.gammaT);
    r.f += mp * np * ord.ap[p];
    r.pdf += mp * ord.pdf[p] * np;
  }
  float mpR = hairMp(cosI, max(cosO, 1e-4), sinI, sinO, sc.v[HAIR_PMAX]);
  r.f += mpR * ord.ap[HAIR_PMAX] * (0.5 * M_1_PI);
  r.pdf += mpR * ord.pdf[HAIR_PMAX] * (0.5 * M_1_PI);
  return r;
}

float3 hairMsF(float3 wi, HairHit hair)
{
  return hair.albedo * (HAIR_MS * 0.5) * abs(dot(wi, hair.normal)) * M_1_PI;
}

float hairMsPdf(float3 wi, HairHit hair)
{
  return (HAIR_MS * 0.5) * abs(dot(wi, hair.normal)) * M_1_PI;
}

HairResponse hairCombined(float3 wo, float3 wi, HairHit hair)
{
  HairScales   sc = hairScales(hair.betaM, hair.betaN);
  HairResponse m  = hairFibreAt(hair.h, wo, wi, hair, sc);
  HairResponse r;
  r.f   = (1.0 - HAIR_MS) * m.f + hairMsF(wi, hair);
  r.pdf = (1.0 - HAIR_MS) * m.pdf + hairMsPdf(wi, hair);
  return r;
}

HairHit hairHitZero()
{
  HairHit hair;
  hair.sigma  = float3(0.0);
  hair.albedo = float3(0.0);
  hair.betaM  = 0.0;
  hair.betaN  = 0.0;
  hair.strand = float3(0.0, 1.0, 0.0);
  hair.phiY   = float3(1.0, 0.0, 0.0);
  hair.phiZ   = float3(0.0, 0.0, 1.0);
  hair.normal = float3(0.0, 0.0, 1.0);
  hair.msT      = float3(1.0, 0.0, 0.0);
  hair.msB      = float3(0.0, 1.0, 0.0);
  hair.specular = 1.0;
  hair.h        = 0.0;
  return hair;
}

bool makeHairHit(GltfShadeMaterial mat, float2 uv, float3 geoBitangent, float3 geoTangent, float3 normal, float3 shadingT, float3 shadingB, out HairHit hair)
{
  hair       = hairHitZero();
  float len2 = dot(geoBitangent, geoBitangent);
  if(len2 < 1e-8)
    return false;
  float3 strand = geoBitangent * rsqrt(len2);
  float3 across = geoTangent - strand * dot(geoTangent, strand);
  if(dot(across, across) < 1e-8)
    across = normal - strand * dot(normal, strand);
  if(dot(across, across) < 1e-8)
    return false;
  across = normalize(across);

  float along = saturate(abs(uv.y - mat.hairRootV));
  hair.sigma  = lerp(mat.hairRootSigma, mat.hairTipSigma, along);
  // The same fit the loader uses to turn absorption into the colour a ribbon shows, so the
  // multiple-scatter body stays the melanin colour and still changes from root to tip.
  float b   = saturate(mat.hairAzimuthal);
  float b2  = b * b;
  float b3  = b2 * b;
  float fit = 5.969 - 0.215 * b + 2.532 * b2 - 10.73 * b3 + 5.574 * b2 * b2 + 0.245 * b2 * b3;
  hair.albedo    = exp(-0.45 * sqrt(max(hair.sigma, float3(0.0))) * fit);
  hair.betaM     = mat.hairLongitudinal;
  hair.betaN     = mat.hairAzimuthal;
  hair.strand    = strand;
  hair.phiY      = across;
  hair.phiZ      = normalize(cross(strand, across));
  hair.normal    = normal;
  hair.msT       = shadingT;
  hair.msB       = shadingB;
  hair.specular  = mat.hairSpecular;
  return true;
}

void hairBsdfEvaluate(inout BsdfEvaluateData data, HairHit hair)
{
  data.bsdf_diffuse = float3(0.0);
  data.bsdf_glossy  = float3(0.0);
  data.pdf          = 0.0;
  HairResponse r    = hairCombined(data.k1, data.k2, hair);
  if(r.pdf <= 0.0 || any(isnan(r.f)) || isnan(r.pdf))
    return;
  // The lobes are specular; the residual scatter is the diffuse slot, so a denoiser can tell them apart.
  data.bsdf_diffuse = hairMsF(data.k2, hair);
  data.bsdf_glossy  = r.f - data.bsdf_diffuse;
  data.pdf          = r.pdf;
}

float3 hairSampleDirection(float h, float3 wo, HairHit hair, HairScales sc, float uLobe, float uAzimuth, float uLong, float uPhi, out int lobe)
{
  float sinO = clamp(dot(wo, hair.strand), -1.0, 1.0);
  float cosO = max(hairSafeSqrt(1.0 - sinO * sinO), 1e-4);
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
  uLong   = max(uLong, 1e-5);
  float cosTheta = 1.0 + v * log(uLong + (1.0 - uLong) * exp(-2.0 / v));
  cosTheta       = clamp(cosTheta, -1.0, 1.0);
  float sinTheta = hairSafeSqrt(1.0 - cosTheta * cosTheta);
  float cosPhi   = cos(M_TWO_PI * uPhi);
  float sinI     = clamp(-cosTheta * sinOp + sinTheta * cosPhi * cosOp, -1.0, 1.0);
  float cosI     = max(hairSafeSqrt(1.0 - sinI * sinI), 0.0);

  float dphi;
  if(lobe < HAIR_PMAX)
    dphi = (2.0 * float(lobe) * ord.gammaT - 2.0 * ord.gammaO + float(lobe) * M_PI) + hairSampleTrimmedLogistic(saturate(uAzimuth), sc.s);
  else
    dphi = M_TWO_PI * uAzimuth;

  float phiO = atan2(dot(wo, hair.phiZ), dot(wo, hair.phiY));
  float phiI = phiO + dphi;
  return sinI * hair.strand + (cosI * cos(phiI)) * hair.phiY + (cosI * sin(phiI)) * hair.phiZ;
}

void hairBsdfSample(inout BsdfSampleData data, HairHit hair, float uPick, float uAzimuth)
{
  data.pdf           = 0.0;
  data.bsdf_over_pdf = float3(0.0);
  data.event_type    = BSDF_EVENT_ABSORB;

  HairScales sc = hairScales(hair.betaM, hair.betaN);
  bool       ms = uPick < HAIR_MS;
  if(ms)
  {
    bool   back  = data.xi.z < 0.5;
    float3 n     = back ? -hair.normal : hair.normal;
    float3 local = cosineSampleHemisphere(data.xi.x, data.xi.y);
    data.k2      = hair.msT * local.x + hair.msB * local.y + n * local.z;
  }
  else
  {
    // At the hit's offset, which hairCombined evaluates the density at
    int lobe = 0;
    data.k2  = hairSampleDirection(hair.h, data.k1, hair, sc, data.xi.x, uAzimuth, data.xi.y, data.xi.z, lobe);
  }
  data.k2 = normalize(data.k2);

  HairResponse r = hairCombined(data.k1, data.k2, hair);
  if(r.pdf <= 1e-8 || any(isnan(r.f)) || isnan(r.pdf) || any(isnan(data.k2)))
  {
    data.event_type = BSDF_EVENT_ABSORB;
    data.pdf        = 0.0;
    return;
  }
  data.pdf           = r.pdf;
  data.bsdf_over_pdf = r.f / r.pdf;
  bool transmit      = dot(data.k2, hair.normal) < 0.0;
  if(ms)
    data.event_type = transmit ? BSDF_EVENT_DIFFUSE_TRANSMISSION : BSDF_EVENT_DIFFUSE_REFLECTION;
  else
    data.event_type = transmit ? BSDF_EVENT_GLOSSY_TRANSMISSION : BSDF_EVENT_GLOSSY_REFLECTION;
}

#endif  // HAIR_BSDF_H
