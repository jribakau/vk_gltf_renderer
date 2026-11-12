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

/*-------------------------------------------------------------------------------------------------
 * The thin preview's skin as the walk returns it from the camera's angle. The still's walk enters by
 * refraction along the view, so its albedo is 1 - H(mu0) sqrt(1 - alpha) at the refracted direction's
 * cosine mu0, rising as the view slants (the light side is the diffuse exit's: diffuseExitFresnelWeight).
 * Measured on a slab under one directional light against the preview's Lambertian map, the walk came
 * back 3, 6 and 7% above it at a 45-degree view (A 0.6, 0.35, 0.25) and 6, 14 and 17% at 70; the
 * H-function says 3.5, 5.8 and 6.8%, and 6.8, 11.7 and 13.8% (a rough boundary lets some light in
 * shallower at a grazing view).
 * -------------------------------------------------------------------------------------------------*/

// Chandrasekhar's H-function of an isotropic half-space of single-scattering albedo omega, at direction
// cosine mu, by Hapke's (2002) approximation: about 1% above the exact, by nearly the same at every mu
float3 hFunctionApprox(float3 omega, float mu)
{
  const float  m  = clamp(mu, 1e-4F, 1.0F);
  const float3 g  = sqrt(max(float3(1.0F) - omega, float3(0.0F)));
  const float3 r0 = (float3(1.0F) - g) / (float3(1.0F) + g);
  return float3(1.0F) / (float3(1.0F) - omega * m * (r0 + (float3(1.0F) - 2.0F * m * r0) * (0.5F * log((1.0F + m) / m))));
}

// What the walk returns entered by refraction from outside at cosine `cosView` through a boundary of
// index `eta`
float3 walkAlbedoAt(float3 alpha, float cosView, float eta)
{
  const float c   = clamp(cosView, 0.0F, 1.0F);
  const float mu0 = sqrt(max(1.0F - (1.0F - c * c) / (eta * eta), 0.0F));
  return float3(1.0F) - hFunctionApprox(alpha, mu0) * sqrt(max(float3(1.0F) - alpha, float3(0.0F)));
}

// The colour the thin preview gives measured skin of map A seen at cosine `cosView`: what the still's
// walk through its medium (single-scattering albedo alpha) returns from that view. With the face-on
// inversion A is that albedo head-on, so A is scaled by the walk's change from head-on, which cancels
// the approximation's error; with the direction-averaged one the walk's own albedo at the view stands
// (head-on 6-15% below the map, as tests/render/skin_furnace.py found).
float3 skinViewAlbedo(float3 A, float3 alpha, float cosView, float eta, bool faceOn)
{
  const float3 atView = walkAlbedoAt(alpha, cosView, eta);
  if(faceOn)
    return A * atView / max(walkAlbedoAt(alpha, 1.0F, eta), float3(1e-4F));
  return atView;
}

// The same for the preview's softened diffuse (PbrMaterial::diffuseSoftening `s`, per channel): the
// share s the walk's spread evens out sees the view at the smooth normal's cosine, the rest at the map's.
// At the map's alone every bump took its own colour from the angle it leans to the camera, and the
// relief the softening evened out came back.
float3 softenedSkinViewAlbedo(float3 A, float3 alpha, float cosMap, float cosSmooth, float3 s, float eta, bool faceOn)
{
  return lerp(skinViewAlbedo(A, alpha, cosMap, eta, faceOn), skinViewAlbedo(A, alpha, cosSmooth, eta, faceOn), s);
}

#endif
