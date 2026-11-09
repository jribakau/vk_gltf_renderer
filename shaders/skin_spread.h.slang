/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The thin preview's spread (gltf_pathtrace.slang): how far from where light enters skin the still's
// walk lets it out, as a radial profile, so the preview can take a light's shadow where the walk would
// have gathered it and a shadow's edge on skin softens as in the still, red the most. Plain math,
// compiled as C++ for its test (tests/test_skin_albedo.cpp).
//
// The shape is Burley's normalized diffusion, R(r) = s (e^{-s r} + e^{-s r / 3}) / (8 pi r), which
// integrates to 1 over the plane (Christensen and Burley 2015). Its scale is fitted to our own walk:
// through the medium FV_materials_subsurface makes from a skin map and a scatter distance d (the
// diffusion length 1 / sigma_tr, gltf_material_eval.h.slang), entered straight in and leaving the first
// time it reaches the surface (ePtVolumeDiffuseExit). Then s = k / d, k a function of the medium's
// single-scattering albedo alpha alone, by c = sqrt(1 - alpha):
//   k = 2.1772 + 0.3585 / c + 0.6248 c,
// within 3.3% of the k that fits the walk best over maps of 0.02 to 0.95, the walk's radial
// distribution within 0.02 of the shape's up to a map of 0.8 (0.045 at 0.95). Christensen and Burley's
// own fit (s = 1.9 - A + 3.5 (A - 0.8)^2 over their diffuse mean free path) spreads two to five times
// wider at these distances: a lighter map spreads less far against d here, since d is the far field's
// decay and a light channel's light mostly leaves early.
#ifndef SKIN_SPREAD_H_SLANG
#define SKIN_SPREAD_H_SLANG

// The unit radius s r a sample is cut at: 99.9% of the profile lies within it
static const float SKIN_SPREAD_MAX = 20.0F;

// k of the fit above, for a medium of single-scattering albedo `alpha`
inline float skinSpreadScale(float alpha)
{
  const float c = sqrt(max(1.0F - alpha, 2.5e-5F));
  return 2.1772F + 0.3585F / c + 0.6248F * c;
}

// s per channel, per scene unit: the medium's single-scattering albedo (as the walk makes it, by the
// face-on or the direction-averaged inversion) and its scatter distance
inline float3 skinSpreadRate(float3 alpha, float3 scatterDistance)
{
  const float3 k = float3(skinSpreadScale(alpha.x), skinSpreadScale(alpha.y), skinSpreadScale(alpha.z));
  return k / max(scatterDistance, float3(1e-7F));
}

// A radius of the unit profile (s = 1) from two uniform numbers: the shape is two exponentials, e^{-a}
// a quarter of the time and e^{-a / 3} the rest. One radius of it, over each channel's s, is a sample
// of every channel's own profile, all three from the same numbers.
inline float skinSpreadUnitRadius(float u1, float u2)
{
  const float scale = (u1 < 0.25F) ? 1.0F : 3.0F;
  return min(-scale * log(max(1.0F - u2, 1e-7F)), SKIN_SPREAD_MAX);
}

// The share of the unit profile within unit radius `a`
inline float skinSpreadUnitCdf(float a)
{
  return 1.0F - 0.25F * exp(-a) - 0.75F * exp(-a / 3.0F);
}

#endif  // SKIN_SPREAD_H_SLANG
