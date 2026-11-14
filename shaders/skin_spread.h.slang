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

// The spread of a surface with no volume under it (FV_materials_subsurface without a thickness: the
// white of an eye, and a Genesis 9 iris in the same surface): where a light sample there looks for its
// shadow, from the shaded point. An exponential radius of mean `d` (the scatter distance), cut at
// SURFACE_SPREAD_CUT d, in the tangent plane, and lifted off that plane by SURFACE_SPREAD_LIFT of the
// radius. The lift is for a surface that curves up to meet its own tangent plane: the iris is a shallow
// bowl (it rises 0.02 to 0.07 mm a millimetre toward its rim), the points lay microns under it, and its
// own surface shadowed a third of its direct light. A tenth clears the inside of a sphere out to a
// fifth of its radius (the back of an eyeball, 12 mm, to 2.4 mm), and puts a point 2 mm out 0.2 mm
// above the white.
static const float SURFACE_SPREAD_CUT  = 2.5F;
static const float SURFACE_SPREAD_LIFT = 0.1F;

inline float surfaceSpreadRadius(float d, float u)
{
  return min(-d * log(max(1.0F - u, 1e-6F)), SURFACE_SPREAD_CUT * d);
}

// `normal`: the surface's, on the light's side; `t`, `b`: two unit vectors square to it and each other
inline float3 surfaceSpreadOffset(float3 normal, float3 t, float3 b, float radius, float phi)
{
  return radius * (cos(phi) * t + sin(phi) * b + SURFACE_SPREAD_LIFT * normal);
}

#endif  // SKIN_SPREAD_H_SLANG
