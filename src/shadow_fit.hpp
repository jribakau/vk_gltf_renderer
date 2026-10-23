/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// The rasterizer's sun shadow map: where its orthographic box goes.
//
// Pure, so it is unit tested (tests/test_shadow_fit.cpp) rather than checked by looking at shadows.
#pragma once

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace shadow_fit {

struct Input
{
  glm::vec3 center{0.0F};   // the casters' bounding sphere
  float     radius = 1.0F;  //
  glm::vec3 toLight{0.0F, 1.0F, 0.0F};  // unit, from the scene toward the light
  int       mapSize = 4096;
  // The ground that receives: a disk of `groundReach` around (groundCenterX, groundCenterZ) at
  // height `groundY`. Reach 0 means no ground, and the box ends at the sphere.
  float groundY       = 0.0F;
  float groundCenterX = 0.0F;
  float groundCenterZ = 0.0F;
  float groundReach   = 0.0F;
};

struct Result
{
  bool      valid = false;
  glm::mat4 view{1.0F};
  glm::mat4 proj{1.0F};
  glm::mat4 viewProj{1.0F};
  float     halfExtent = 0.0F;  // metres from the box's centre to its side
  float     texelWorld = 0.0F;  // metres per shadow-map texel
  float     nearDepth  = 0.0F;  // distance along the light's rays, from the light-space origin
  float     farDepth   = 0.0F;
  float     depthRange = 0.0F;  // farDepth - nearDepth: what [0, 1] of the map's depth spans
};

// A half extent that changes in steps of 1/16 of its power of two, so a sphere that grows or
// shrinks a little keeps the same texel size -- otherwise every texel's footprint moves and the
// edges crawl.
inline float quantizeExtent(float r)
{
  if(!(r > 0.0F))
    return 0.0F;
  const float step = std::exp2(std::floor(std::log2(r))) / 16.0F;
  return std::ceil(r / step) * step;
}

// The light-space rotation: world to a frame whose -Z runs along the light's rays. Origin stays at
// the world origin, which is what lets the box snap to a grid that does not move with the scene.
inline glm::mat4 lightRotation(const glm::vec3& toLight)
{
  const glm::vec3 up = (std::abs(toLight.y) > 0.99F) ? glm::vec3(0.0F, 0.0F, 1.0F) : glm::vec3(0.0F, 1.0F, 0.0F);
  return glm::lookAtRH(glm::vec3(0.0F), -toLight, up);
}

// Upper bound on how far past the sphere the box may reach, in sphere radii. At a sun a few degrees
// above the horizon the ground's shadow runs off to the horizon; past this the receivers clamp to
// the far plane, which still reads as shadowed -- only the penumbra estimate degrades.
inline constexpr float kMaxFarRadii = 64.0F;

inline Result fit(const Input& in)
{
  Result r;
  const float lightLen = glm::length(in.toLight);
  if(!(in.radius > 0.0F) || !(lightLen > 1e-6F) || in.mapSize <= 0)
    return r;
  const glm::vec3 toLight = in.toLight / lightLen;
  const glm::vec3 along   = -toLight;  // the rays' direction

  const glm::mat4 rot = lightRotation(toLight);
  r.halfExtent        = quantizeExtent(in.radius * 1.02F);
  r.texelWorld        = 2.0F * r.halfExtent / float(in.mapSize);

  // Snap the box centre to whole texels in the fixed light frame.
  glm::vec3 c = glm::vec3(rot * glm::vec4(in.center, 1.0F));
  c.x         = std::round(c.x / r.texelWorld) * r.texelWorld;
  c.y         = std::round(c.y / r.texelWorld) * r.texelWorld;

  // Depth runs along the rays: in this right-handed view, a point's distance is -z.
  const float centerDepth = -c.z;
  float       farReach    = in.radius;
  if(in.groundReach > 0.0F)
  {
    // The deepest point of the ground disk: its centre's depth plus the disk's radius times how
    // much of the ray direction lies in the ground plane.
    const glm::vec3 g(in.groundCenterX, in.groundY, in.groundCenterZ);
    const float     horiz = std::sqrt(std::max(0.0F, along.x * along.x + along.z * along.z));
    const float     deep  = glm::dot(g - in.center, along) + in.groundReach * horiz;
    farReach              = std::max(farReach, deep);
  }
  farReach = std::min(farReach, in.radius * kMaxFarRadii);

  // A little slack in front, so a caster on the sphere's surface is never clipped by the near plane.
  r.nearDepth  = centerDepth - in.radius * 1.05F;
  r.farDepth   = centerDepth + farReach * 1.01F;
  r.depthRange = r.farDepth - r.nearDepth;

  r.view     = rot;
  r.proj     = glm::orthoRH_ZO(c.x - r.halfExtent, c.x + r.halfExtent, c.y - r.halfExtent, c.y + r.halfExtent,
                               r.nearDepth, r.farDepth);
  r.viewProj = r.proj * r.view;
  r.valid    = true;
  return r;
}

// The ground's fade radius from Lighting.ts's reach: the floor disk is `fade` scene radii.
inline float groundReach(float sceneRadius, float fade)
{
  return std::max(0.0F, sceneRadius * fade);
}

}  // namespace shadow_fit
