/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The thin preview's slab (gltf_pathtrace.slang): what a layer of skin reflects and what it lets
// through to its far side, where it is thin enough for light to reach that side, as the still's walk
// lets it through an ear or between fingers. Kubelka-Munk's slab, its reflectance at infinite depth
// the skin map A (what the preview shows on thick skin, and the walk's albedo by construction:
// skin_albedo.h.slang) and its optical depth the thickness over the diffusion length d, the measured
// scatter distance the walk's medium is made from (sigma_tr = 1 / d). Against the walk itself, through
// a slab of that medium entered diffusely (tests/test_skin_albedo.cpp), it is within a few percent for
// the light channels that carry the glow (A 0.6: 1-6% of the transmittance from 0.25 to 5 diffusion
// lengths) and low for a dark channel deep in, where little gets through (A 0.2: 0.048 against 0.062
// at three lengths). The walk's exact decay rate in place of 1 / d (walk_guide.h.slang's diffusion
// length) moves that error to the other side, not away. Plain math, compiled as C++ for its test.
#ifndef SKIN_SLAB_H_SLANG
#define SKIN_SLAB_H_SLANG

struct SkinSlab
{
  float3 reflectance;    // back out through the side the light came in by
  float3 transmittance;  // out through the far side
};

// How deep the preview looks for a far side, in diffusion lengths of the channel that reaches
// furthest: past it under 0.1% of that channel's light gets through (A 0.6: 0.02% at eight)
static const float SKIN_SLAB_REACH = 8.0F;
// The cosine of the half-angle of the cone about the smooth normal the probe goes in by (45 degrees),
// its hit's depth along that normal taken for the thickness: through a flat slab every direction gives
// the slab's own thickness, while beside a fold some find the thin part next to it, as the walk's light
// spreads sideways under the skin. Straight in alone, the glow's edge was as sharp as the ear's folds
// and stepped with each triangle, and the bowl of the ear, backed by the head, stayed dark.
static const float SKIN_SLAB_CONE_COS = 0.70710678F;
// The chance of the far side under which a hit is shaded as thick skin, as the preview always did
static const float SKIN_SLAB_MIN_CHANCE = 1e-3F;
// ...and over which the near side's weight (one over its chance) would grow past ten
static const float SKIN_SLAB_MAX_CHANCE = 0.9F;

// `A` the reflectance at infinite depth, `depth` the thickness in diffusion lengths, per channel.
// Kubelka-Munk with a = (1 + A^2) / 2A, b = (1 - A^2) / 2A and e = exp(-depth):
// R = (1 - e^2) / (a (1 - e^2) + b (1 + e^2)), T = 2 b e / (same), the sinh/cosh form over 2 / e,
// which overflows nowhere. No depth lets all through (R 0, T 1); infinite depth gives back A.
inline SkinSlab skinSlab(float3 A, float3 depth)
{
  const float3 r     = clamp(A, float3(1e-3F), float3(0.999F));
  const float3 a     = (float3(1.0F) + r * r) / (2.0F * r);
  const float3 b     = (float3(1.0F) - r * r) / (2.0F * r);
  const float3 e     = exp(-max(depth, float3(0.0F)));
  const float3 e2    = e * e;
  const float3 denom = a * (float3(1.0F) - e2) + b * (float3(1.0F) + e2);
  SkinSlab     slab;
  slab.reflectance   = (float3(1.0F) - e2) / denom;
  slab.transmittance = 2.0F * b * e / denom;
  return slab;
}

#endif  // SKIN_SLAB_H_SLANG
