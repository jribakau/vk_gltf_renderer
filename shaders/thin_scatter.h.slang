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

#endif  // THIN_SCATTER_H_SLANG
