/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// What a skin's colour map means as the medium under it (FV_materials_subsurface): the base colour
// A, inverted to the single-scattering albedo of the isotropic medium the random walk goes through.
// Compiled as C++ for its test (tests/test_skin_albedo.cpp).
//
// Chiang, Kutz and Burley's inversion (multiToSingleScatterAlbedo) makes A the walk's albedo
// averaged over every direction light comes from, which is what a diffuse entry returns. Light that
// refracts in enters steeply (at most 46 degrees from the normal at an index of 1.4), goes deeper
// and comes back less: lit from the front such skin returned 0.94, 0.88 and 0.85 of a pale skin's
// map in red, green and blue, Chandrasekhar's H-function for a half-space, 1 - H(1) sqrt(1 - alpha).
// This inversion is fitted to that head-on albedo instead, in the same form, so skin lit from the
// front comes back at its map, as the moving view's preview shows it: the H-function of the alpha it
// gives is within 0.07% of A over 0.001 to 0.999, and rendered (tests/render/skin_furnace.py) the
// pale skin came back at 0.990, 0.992 and 0.984 of it, as the diffuse entry does under the other
// inversion (0.995, 0.993, 0.985: the walk's own percent). Seen from an angle the camera's ray enters less
// steeply and comes back more: at 60 degrees the pale skin reads 3, 6 and 8% above its map (the
// direction-averaged fit 2, 5 and 6% below), more in a darker channel. A Lambertian preview reads A
// at every angle; no one inversion of a refracting entry does.

#ifndef SKIN_ALBEDO_H
#define SKIN_ALBEDO_H

/*-------------------------------------------------------------------------------------------------
 * Convert multi-scatter albedo to single-scatter albedo (Kulla-Conty mapping).
 * Formula from the KHR_materials_scatter specification. The anisotropy g of the
 * Henyey-Greenstein phase function enters the denominator, so forward/backward
 * scattering media get the correct single-scatter albedo.
 * -------------------------------------------------------------------------------------------------*/
float3 multiToSingleScatterAlbedo(float3 rho_ms, float g)
{
  float3 s = 4.09712F + 4.20863F * rho_ms - sqrt(9.59217F + 41.6808F * rho_ms + 17.7126F * rho_ms * rho_ms);
  // g in (-1,1) and s in [0,1] keep the denominator strictly positive. Clamp the
  // result to [0,1] so scatterCoefficient never exceeds the total extinction.
  return clamp((1.0F - s * s) / (1.0F - g * s * s), float3(0.0F), float3(1.0F));
}

float3 faceOnSingleScatterAlbedo(float3 A)
{
  const float3 s = 6.73841F + 13.0827F * A - sqrt(max(32.9292F + 187.577F * A + 172.375F * A * A, float3(0.0F)));
  const float3 c = clamp(s, float3(0.0F), float3(1.0F));
  return clamp(1.0F - c * c, float3(0.0F), float3(1.0F));
}

#endif
