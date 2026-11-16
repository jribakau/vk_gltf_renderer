/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The exact dielectric Fresnel, for the BSDF's neighbours that need it unpolarised and whole: the skin's
// diffuse exit (thin_scatter.h.slang), the hair fibre's (hair_bsdf.h.slang) and the ground's water
// (ground.h.slang). Plain math, so the unit tests compile it as C++ through them.
#ifndef FRESNEL_H_SLANG
#define FRESNEL_H_SLANG

// Unpolarised Fresnel reflectance for light at `cosI` onto a medium of relative index `eta` (the far
// side's over the near side's): 1 past the critical angle where eta < 1.
inline float dielectricFresnelOutside(float cosI, float eta)
{
  cosI              = clamp(cosI, 0.0F, 1.0F);
  const float sinT2 = (1.0F - cosI * cosI) / (eta * eta);
  const float cosT  = sqrt(max(1.0F - sinT2, 0.0F));
  // grazing onto a medium of the same index (a light below the map's normal, eta 1): both ratios are
  // 0 over 0, where every other index reflects it all
  if(cosI + eta * cosT <= 0.0F)
    return 1.0F;
  const float rs    = (cosI - eta * cosT) / (cosI + eta * cosT);
  const float rp    = (eta * cosI - cosT) / (eta * cosI + cosT);
  return 0.5F * (rs * rs + rp * rp);
}

#endif  // FRESNEL_H_SLANG
