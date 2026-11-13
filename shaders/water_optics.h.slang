/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The ground's water as a medium: its index, absorption and scattering, and what a body of it
// reflects as the physical sky's ground. Slang and plain C++ (glm, no swizzles): ground.h.slang draws
// with it, src/sky_ground.cpp gives the sky its ground from it.
#ifndef WATER_OPTICS_H_SLANG
#define WATER_OPTICS_H_SLANG

static const float GROUND_WATER_IOR = 1.33F;

// Pure water's absorption, per metre, over the bands of the sRGB primaries (Pope and Fry 1997: about
// 630, 550 and 470 nm)
static const float3 GROUND_WATER_ABSORPTION = float3(0.29F, 0.055F, 0.011F);
// Pure seawater's scattering, per metre, over the same bands: Morel's 0.00288 at 500 nm, as the
// wavelength to the -4.32. Deep water is blue by it: of light straight down it sends back about 4.6% of
// the blue, 0.6% of the green and 0.07% of the red (once scattered; groundWaterInScatter), the open
// sea's colour. Its phase function is the molecules' (groundWaterPhase).
static const float3 GROUND_WATER_SCATTERING = float3(0.00106F, 0.00191F, 0.00376F);
static const float3 GROUND_WATER_EXTINCTION = GROUND_WATER_ABSORPTION + GROUND_WATER_SCATTERING;

// What a body of water `depth` metres deep over a floor of albedo `floorAlbedo` reflects of diffuse
// light, as the physical sky's ground (the planet's surface its tables and the clouds' ground bounce
// take as matte; src/sky_ground.hpp): the surface's own share (0.066, groundWetAlbedo's re), and what
// comes back up through it, out as groundWetAlbedo has it. Below the surface: the floor, seen through
// the water down and up again, and the water's own backscatter over what the floor leaves (Gordon's
// 0.33 b_b / a, b_b half the scattering, the molecules' phase being symmetric). Diffuse light crosses
// the water at a slant, 1 / 0.83 of the depth each way (the mean cosine of a sky's light refracted).
// The open sea: about 0.066 red and green, 0.096 blue; two metres over white sand, a lagoon's cyan,
// which lights the clouds above it so.
inline float3 groundWaterAlbedo(float3 floorAlbedo, float depth)
{
  const float  n2       = GROUND_WATER_IOR * GROUND_WATER_IOR;
  const float  re       = 0.066F;
  const float  internal = 1.0F - (1.0F - re) / n2;
  const float3 deep     = (0.33F * 0.5F) * GROUND_WATER_SCATTERING / GROUND_WATER_ABSORPTION;
  const float3 through  = exp(-GROUND_WATER_EXTINCTION * (2.0F * max(depth, 0.0F) / 0.83F));
  const float3 below    = deep + (floorAlbedo - deep) * through;
  return float3(re, re, re) + (1.0F - re) * below / (n2 * (float3(1.0F, 1.0F, 1.0F) - below * internal));
}

#endif  // WATER_OPTICS_H_SLANG
