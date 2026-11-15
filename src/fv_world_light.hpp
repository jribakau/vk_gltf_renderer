/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// The light of the other bodies in a planet's sky (shaders/world_bodies.h.slang): each one's face, lit
// by the sun as it is drawn, summed over the sphere as a place sees it. What it gives a surface facing
// it, and from where on average, is what the path tracer weighs a body's light sample by and what the
// air and the clouds are lit with; what each body and the planet give each other body is the light on
// its dark side (one bounce: what a body receives from the others is their sunlit faces'). Pure, tested
// without a device (tests/test_fv_world_light.cpp); the renderer sums it when the sun or a body moves.
//
// Light here is against the sun's illuminance above the air: a body's 0.01 is a hundredth of the sun.
// Positions are metres in the scene's axes, as the frame's (SceneFrameInfo::worldBodyPlace).

#include <array>
#include <vector>

#include <glm/glm.hpp>

namespace fv::world_light {

// As many bodies as a planet's sky holds (WORLD_BODY_MAX in shaders/shaderio.h)
inline constexpr int kMost = 4;

// A body that stands in the sky: its centre and radius (metres), its linear albedo, what it is made of
// (WORLD_BODY_ROCK, WORLD_BODY_GAS), as SceneFrameInfo::worldBodyPlace and worldBodyLook carry it
struct Body
{
  glm::dvec3 centre{0.0};
  double     radius = 0.0;
  glm::vec3  color{0.0F};
  int        kind = 0;
};

struct Sky
{
  std::vector<Body> bodies;  // up to kMost
  // the planet under the scene: a sphere of its mean albedo (fv::world::World::albedo) to the bodies
  glm::dvec3 planetCentre{0.0};
  double     planetRadius = 0.0;
  glm::vec3  planetAlbedo{0.0F};
  // toward the sun (unit), and its disk's angular radius
  glm::dvec3 toSun{0.0, 1.0, 0.0};
  double     sunAngularRadius = 0.004675;
};

// A sphere's face as a place sees it
struct Face
{
  glm::vec3  light{0.0F};          // the illuminance on a surface facing it
  glm::dvec3 toward{0.0, 1.0, 0.0};  // the light's mean direction from the place, unit
  double     spread = 1.0;         // that mean's length: 1 for a point, less the wider the light comes from
};

struct Lights
{
  // each body's light at the place it was summed for (sumLights' `place`)
  std::array<Face, kMost> shine{};
  // whether the path tracer takes it as a light: not one too dim beside the sun or the brightest body
  // there to matter (kLeast)
  std::array<bool, kMost> lit{};
  // what lights each body's dark side: [lit * kMost + source], the source's light on the lit body's
  // centre; the lit body's own slot is the planet's
  std::array<glm::vec3, kMost * kMost> fill{};
};

// A body is a light where its light is at least this share of the sun's on a level surface there, or of
// the brightest body's (Celestia's cutoff for planetshine): a full moon by day is not, at night it is.
inline constexpr float kLeast = 1.0e-4F;

// The light of the face of body `source` (-1: the planet) toward a place `toEye` from its `point`, per
// unit of the sun's illuminance: the sun's on it (its phase, and where the planet or another body stands
// before the sun, nothing), and with `fill` what the others give it.
glm::vec3 faceRadiance(const Sky& sky, const Lights* fill, int source, const glm::dvec3& point, const glm::dvec3& toEye);

// The face of body `source` (-1: the planet) summed over `steps` x `steps` directions spread evenly over
// the solid angle it covers from `from`, those another body hides left out. Nothing from inside it.
Face faceLight(const Sky& sky, const Lights* fill, int source, const glm::dvec3& from, int steps);

// Every body's light at `place` (the scene's origin), and what they and the planet give each other
Lights sumLights(const Sky& sky, const glm::dvec3& place);

// The luminance of a linear colour
inline float luminance(const glm::vec3& c)
{
  return 0.2126F * c.x + 0.7152F * c.y + 0.0722F * c.z;
}

}  // namespace fv::world_light
