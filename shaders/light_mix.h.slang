/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The light groups' mix: a picture is the sum of what each of its lights gives it, so a group of lights
// made twice as strong, or tinted, is that group's share of the picture times two, or times the tint,
// with nothing traced again. Every light is in one group (shaderio.h LIGHT_GROUP_*), and a still that
// keeps its lights apart (USE_LIGHT_GROUPS) holds an image of each group's share beside the whole.
//
// A path that carried every group's share cost a still half again its time (a face at 1280x960: 33 ms
// a sample against 21; one share alone, about a twentieth), so a sample carries one group's: the groups
// the scene has light in take turns, a pixel's samples in order from a start of its own
// (lightGroupOfSample). A group's image is the mean of the samples that were its turn
// (lightGroupTurnsBefore): a K-th of them with K groups, so it is noisier than the whole, which has
// them all.
//
// The mix leans on the whole as far as it can. With each group's factor m_g and the least of them a
// (lightMixFloor), it is a times the whole plus (m_g - a) times each group's image: every term is a
// light times a factor that is not negative, so the mix is never under zero, and it is right on
// average whatever a is (the images sum to the whole on average). With every factor 1 it is the whole,
// bit for bit; a group made brighter adds only its own image's noise; a group turned off leaves a at
// 0, and the mix is then the other groups' images alone, as noisy as a still of a K-th of the samples.
//
// Plain math, compiled as C++ for its test (tests/test_light_mix.cpp).
#ifndef LIGHT_MIX_H_SLANG
#define LIGHT_MIX_H_SLANG

// The least a factor has taken so far, with `mix` (a group's factor) if `used` (the scene has light
// in the group): start from float3(1e30)
inline float3 lightMixFloor(float3 floor, float3 mix, bool used)
{
  return used ? min(floor, mix) : floor;
}

// What a group's image adds to the mix over the whole at the floor: `mix` its factor
inline float3 lightMixTerm(float3 share, float3 mix, float3 floor)
{
  return max(mix - floor, float3(0.0F)) * share;
}

// The mix: the whole at the floor, and each group's term (their sum, `terms`)
inline float3 lightMixed(float3 whole, float3 floor, float3 terms)
{
  return max(floor * whole + terms, float3(0.0F));
}

// How many groups the scene has light in: `used` has bit g set for group g
inline uint lightGroupsInTurn(uint used)
{
  uint count = 0u;
  for(uint g = 0u; g < uint(LIGHT_GROUP_COUNT); g++)
    count += (used >> g) & 1u;
  return count;
}

// The group whose share sample `sample` of a pixel carries (LIGHT_GROUP_COUNT: none, no group is
// used): the used groups in turn, the turns going round from the pixel's own `start`
inline uint lightGroupOfSample(uint used, uint sample, uint start)
{
  const uint count = lightGroupsInTurn(used);
  if(count == 0u)
    return uint(LIGHT_GROUP_COUNT);
  uint turn = (sample % count + start % count) % count;  // (the sum itself would wrap at 2^32 and skip turns)
  for(uint g = 0u; g < uint(LIGHT_GROUP_COUNT); g++)
  {
    if(((used >> g) & 1u) == 0u)
      continue;
    if(turn == 0u)
      return g;
    turn--;
  }
  return uint(LIGHT_GROUP_COUNT);
}

// How many of a pixel's samples before `sample` carried the same group as `sample` does: what its
// image's mean so far is of
inline uint lightGroupTurnsBefore(uint used, uint sample)
{
  const uint count = lightGroupsInTurn(used);
  return count == 0u ? 0u : sample / count;
}

#endif  // LIGHT_MIX_H_SLANG
