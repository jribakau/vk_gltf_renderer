/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// KHR_materials_scatter's thin-walled lobe on the BSDF's own lobes (evaluateMaterial). Plain math,
// so the unit tests compile it as C++ (tests/test_thin_scatter.cpp).
#ifndef THIN_SCATTER_H_SLANG
#define THIN_SCATTER_H_SLANG

struct ThinScatterLobes
{
  float  transmission;               // the specular BTDF's weight that is left
  float  diffuseTransmissionFactor;  // the share of the diffuse bucket sent through the surface
  float3 diffuseTransmissionColor;   // KHR_materials_diffuse_transmission's colour, blended
  float3 baseColor;                  // the diffuse reflection's colour, blended
  float3 transmissionTint;           // cancels the blend for the residual specular BTDF
};

// The thin-walled branch of the spec replaces the transmissive part of the dielectric base with
//   mix(specular_btdf * baseColor, scatter_bsdf(multiscatterColor, g), scatterStrength)
// where scatter_bsdf is a Lambertian BSDF sending (1+g)/2 of its energy through the surface and
// (1-g)/2 back. There is no dedicated scatter lobe in the BSDF, so it is expressed with the existing
// diffuse-transmission / diffuse-reflection lobes, whose weights below the specular layer are
// `transmission` for the BTDF and `1 - transmission` for the diffuse bucket (split by
// diffuseTransmissionFactor). Moving `scatterWeight = transmission * scatterStrength` from the BTDF
// into the diffuse bucket reproduces the spec lobe weights exactly, for any transmission and any g.
// Note both factors: without transmission the extension must have no effect at all. For
// scatterWeight > 0 only.
ThinScatterLobes thinScatterLobes(float  transmission,
                                  float  scatterStrength,
                                  float  g,
                                  float  diffuseTransmissionFactor,
                                  float3 diffuseTransmissionColor,
                                  float3 baseColor,
                                  float3 multiscatterColor)
{
  const float scatterWeight = transmission * scatterStrength;  // energy moved to scatter_bsdf

  // The diffuse bucket holds four shares: the material's own diffuse reflection and diffuse
  // transmission (untouched by scatter), plus the two halves of scatter_bsdf.
  const float forwardFraction = 0.5F * (1.0F + g);
  const float baseDiffuse     = 1.0F - transmission;                    // base, untouched by scatter
  const float baseTransWeight = baseDiffuse * diffuseTransmissionFactor;  // 0 unless KHR_materials_diffuse_transmission
  const float baseReflWeight  = baseDiffuse * (1.0F - diffuseTransmissionFactor);
  const float forwardWeight   = scatterWeight * forwardFraction;
  const float backwardWeight  = scatterWeight - forwardWeight;
  const float diffuseWeight   = baseDiffuse + scatterWeight;  // the four shares; >= scatterWeight > 0

  ThinScatterLobes lobes;
  lobes.transmission = transmission - scatterWeight;  // = transmission * (1 - scatterStrength)

  // Split the bucket. Note this is the forward share *of the bucket*, not the anisotropy's forward
  // fraction: the bucket also carries the untouched base, so it has to be normalised by the
  // bucket's own weight. (Using the bare fraction is only right at transmission = 1.)
  lobes.diffuseTransmissionFactor = (baseTransWeight + forwardWeight) / diffuseWeight;

  // Each of the two diffuse lobes now carries contributions the spec colors differently, so each
  // takes the energy-weighted average of its own two shares. Both shares of a lobe have the same
  // Lambertian shape, so averaging their colors is exact, not an approximation.
  const float forwardShare       = forwardWeight / max(baseTransWeight + forwardWeight, 1e-6F);
  const float backwardShare      = backwardWeight / max(baseReflWeight + backwardWeight, 1e-6F);
  lobes.diffuseTransmissionColor = diffuseTransmissionColor + (multiscatterColor - diffuseTransmissionColor) * forwardShare;
  lobes.baseColor                = baseColor + (multiscatterColor - baseColor) * backwardShare;

  // ...but baseColor also tints the residual specular BTDF, which the spec leaves at the material's
  // baseColor. transmissionTint cancels the blend back out for that lobe alone (the BSDF tints it by
  // `baseColor * transmissionTint`), and the product is bounded by baseColor so it can never add
  // energy. It degrades in one corner: at transmission = 1 with a multiscatterColor channel of
  // exactly 0, the blended color is 0 in that channel and a multiplier cannot recover baseColor from
  // it, so the residual BTDF loses that channel.
  lobes.transmissionTint = (lobes.transmission > 0.0F) ? baseColor / max(lobes.baseColor, float3(1e-4F)) : float3(1.0F);
  return lobes;
}

// The diffuse exit's light-side Fresnel (gltf_pathtrace.slang, ePtExitFresnel). A walk that leaves
// skin as a Lambertian point skipped the boundary on the light's side, while the camera's side paid
// its (1 - F) on the way in: grazing light was over-counted by about F there. Light refracts in from
// outside in proportion to Ft(theta) = 1 - F(theta), so the exit is Lambertian times Ft / c, with c
// the cosine-weighted mean of Ft, which keeps what the walk hands back (the walk's albedo is
// inverted from the surface's colour) and only moves it from grazing light to head-on light.

// Unpolarised Fresnel reflectance for light from outside (index 1) at `cosI` onto a medium of `eta`.
float dielectricFresnelOutside(float cosI, float eta)
{
  cosI              = clamp(cosI, 0.0F, 1.0F);
  const float sinT2 = (1.0F - cosI * cosI) / (eta * eta);
  const float cosT  = sqrt(max(1.0F - sinT2, 0.0F));
  const float rs    = (cosI - eta * cosT) / (cosI + eta * cosT);
  const float rp    = (eta * cosI - cosT) / (eta * cosI + cosT);
  return 0.5F * (rs * rs + rp * rp);
}

// c = 2 * integral of Ft(mu) mu dmu over [0, 1]: by reciprocity eta^2 (1 - Fdr), Fdr the diffuse
// reflectance from inside, in Egan and Hilgeman's fit (as Jensen et al. 2001 use it).
float diffuseExitTransmittance(float eta)
{
  const float fdr = -1.440F / (eta * eta) + 0.710F / eta + 0.668F + 0.0636F * eta;
  return eta * eta * (1.0F - fdr);
}

// The exit lobe's weight toward a direction at `cosTheta` from its normal, for a boundary of `eta`.
float diffuseExitFresnelWeight(float cosTheta, float eta)
{
  return (1.0F - dielectricFresnelOutside(cosTheta, eta)) / diffuseExitTransmittance(eta);
}

#endif  // THIN_SCATTER_H_SLANG
