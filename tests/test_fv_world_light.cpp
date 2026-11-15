/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// The light of a planet's other bodies (src/fv_world_light.hpp): a face summed against the closed
// forms, a crescent's light from its lit edge, eclipses and hiding, what lights a body's dark side, and
// which bodies are lights.
//

#include <cmath>

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include "../src/fv_world_light.hpp"

namespace {

using namespace fv::world_light;

constexpr double kPi = 3.14159265358979323846;

// The angle between two unit vectors, radians
double angle(const glm::dvec3& a, const glm::dvec3& b)
{
  return std::atan2(glm::length(glm::cross(a, b)), glm::dot(a, b));
}

// A planet 20 km in radius under the place, nothing else
Sky planetOnly(const glm::dvec3& toSun)
{
  Sky s;
  s.planetCentre = glm::dvec3(0.0, -20000.0, 0.0);
  s.planetRadius = 20000.0;
  s.planetAlbedo = glm::vec3(0.3F);
  s.toSun        = glm::normalize(toSun);
  return s;
}

TEST(WorldLight, AFullLambertSphereGivesTwoThirdsOfItsAlbedoTimesItsDiskOverItsDistance)
{
  // the planet seen from 1000 radii off along the sun: full, Lambert's 2/3 p (R/d)^2
  const Sky        s    = planetOnly(glm::dvec3(1.0, 0.0, 0.0));
  const glm::dvec3 from = s.planetCentre + glm::dvec3(1.0, 0.0, 0.0) * (1000.0 * s.planetRadius);
  const Face       f    = faceLight(s, nullptr, -1, from, 64);
  const double     want = 2.0 / 3.0 * 0.3 * 1.0e-6;
  EXPECT_NEAR(f.light.x / want, 1.0, 0.01);
  EXPECT_LT(angle(f.toward, glm::normalize(s.planetCentre - from)), 1e-6);
  EXPECT_GT(f.spread, 0.999999);
}

TEST(WorldLight, AHalfLitSphereGivesLambertsPhaseAndItsLightFromTheLitSide)
{
  // a quarter: the sun across the line of sight; Lambert's phase function at 90 degrees is 1 / pi
  const Sky        s    = planetOnly(glm::dvec3(1.0, 0.0, 0.0));
  const glm::dvec3 from = s.planetCentre + glm::dvec3(0.0, 0.0, 1.0) * (1000.0 * s.planetRadius);
  const Face       f    = faceLight(s, nullptr, -1, from, 64);
  const double     want = 2.0 / 3.0 * 0.3 * 1.0e-6 / kPi;
  EXPECT_NEAR(f.light.x / want, 1.0, 0.02);
  // from the half toward the sun: off the centre toward +x, inside the disk
  const glm::dvec3 centre = glm::normalize(s.planetCentre - from);
  EXPECT_GT(glm::dot(f.toward - centre, glm::dvec3(1.0, 0.0, 0.0)), 0.0);
  EXPECT_LT(angle(f.toward, centre), 1.0e-3);
}

TEST(WorldLight, ACrescentsLightComesFromItsLitEdge)
{
  // a giant 60 degrees across (radius half its distance) as in the user's sky, the sun 60 degrees off
  // its centre: from this near a crescent (the eye sees only the cap within 60 degrees of it, so a sun
  // nearer than that lights none of it), whose light comes from well off the centre toward the sun
  Sky s = planetOnly(glm::dvec3(0.0, 1.0, 0.0));
  s.planetRadius = 0.0;  // no planet in the way
  Body giant;
  giant.centre = glm::dvec3(0.0, 0.0, -3991.0e3);
  giant.radius = 1995.0e3;
  giant.color  = glm::vec3(0.57F, 0.41F, 0.25F);
  giant.kind   = 1;
  s.bodies     = {giant};
  const double turn = 120.0 * kPi / 180.0;
  s.toSun           = glm::dvec3(0.0, std::sin(turn), std::cos(turn));  // 60 degrees from -z, over +y
  const Face  f     = faceLight(s, nullptr, 0, glm::dvec3(0.0), 48);
  const double half = std::asin(0.5);
  EXPECT_GT(f.light.x, 0.0F);
  EXPECT_GT(f.toward.y, 0.0);                                        // toward the sun's side
  EXPECT_GT(angle(f.toward, glm::dvec3(0.0, 0.0, -1.0)), 0.4 * half);  // well off the centre
  EXPECT_LT(angle(f.toward, glm::dvec3(0.0, 0.0, -1.0)), half);        // and on the disk
  // and far less than at full
  s.toSun          = glm::dvec3(0.0, 0.0, 1.0);
  const Face full  = faceLight(s, nullptr, 0, glm::dvec3(0.0), 48);
  EXPECT_LT(f.light.x, 0.1F * full.light.x);
  // the full giant: some 9-12% of the sun at its colour. A far sphere's would be its disk times the
  // face's mean over it (0.6 + 0.4 * 2/3); from this near more, as the eye meets the edge of the cap it
  // sees at a graze while the sun still stands well up there, and Lommel-Seeliger is brightest so
  const double far = 0.57 * 0.25 * (0.6 + 0.4 * 2.0 / 3.0);
  EXPECT_GT(full.light.x / far, 1.0);
  EXPECT_LT(full.light.x / far, 1.4);
}

TEST(WorldLight, ABodyInThePlanetsShadowGivesNothing)
{
  Sky  s = planetOnly(glm::dvec3(1.0, 0.0, 0.0));
  Body moon;
  moon.centre = s.planetCentre + glm::dvec3(-100.0e3, 0.0, 0.0);  // 100 km behind it: in the umbra
  moon.radius = 1000.0;
  moon.color  = glm::vec3(0.3F);
  s.bodies    = {moon};
  EXPECT_EQ(faceLight(s, nullptr, 0, glm::dvec3(0.0, 0.0, 50.0e3), 32).light.x, 0.0F);
  // out of it, lit
  s.bodies[0].centre = s.planetCentre + glm::dvec3(0.0, 0.0, -100.0e3);
  EXPECT_GT(faceLight(s, nullptr, 0, glm::dvec3(0.0, 0.0, 50.0e3), 32).light.x, 0.0F);
}

TEST(WorldLight, ABodyBeforeAnotherHidesIt)
{
  Sky s = planetOnly(glm::dvec3(0.0, 0.0, 1.0));
  s.planetRadius = 0.0;
  Body near, far;
  near.centre = glm::dvec3(0.0, 0.0, -1000.0e3);
  near.radius = 200.0e3;
  near.color  = glm::vec3(0.3F);
  far.centre  = glm::dvec3(0.0, 0.0, -3000.0e3);
  far.radius  = 300.0e3;  // smaller in the sky than the near one
  far.color   = glm::vec3(0.3F);
  s.bodies    = {near, far};
  // the sun behind the place, so the near one casts no shadow on the far one's face toward it... but
  // it stands before it from the place
  EXPECT_EQ(faceLight(s, nullptr, 1, glm::dvec3(0.0), 32).light.x, 0.0F);
  s.bodies[0].centre.x = 800.0e3;  // moved aside
  EXPECT_GT(faceLight(s, nullptr, 1, glm::dvec3(0.0), 32).light.x, 0.0F);
}

TEST(WorldLight, ANeighbourLightsABodysDarkSide)
{
  // a moon between the place and the sun's side, a full giant beside it: the moon's side toward the
  // place has no sun, but the giant's light
  Sky s = planetOnly(glm::dvec3(0.0, 0.0, -1.0));
  Body moon, giant;
  moon.centre  = glm::dvec3(0.0, 1000.0e3, -5000.0e3);
  moon.radius  = 300.0e3;
  moon.color   = glm::vec3(0.3F);
  giant.centre = glm::dvec3(0.0, 1000.0e3, -5000.0e3) + glm::dvec3(2000.0e3, 0.0, 3000.0e3);
  giant.radius = 1000.0e3;
  giant.color  = glm::vec3(0.6F);
  giant.kind   = 1;
  s.bodies     = {moon, giant};
  const Lights l = sumLights(s, glm::dvec3(0.0));
  // the giant on the moon, more than the small far planet on it
  EXPECT_GT(luminance(l.fill[0 * kMost + 1]), 0.0F);
  EXPECT_GT(luminance(l.fill[0 * kMost + 1]), 100.0F * luminance(l.fill[0 * kMost + 0]));
  // the moon's dark face is lit by it
  const Face dark = faceLight(s, nullptr, 0, glm::dvec3(0.0), 32);
  EXPECT_GT(luminance(l.shine[0].light), luminance(dark.light));
  EXPECT_GT(luminance(l.shine[0].light), 0.0F);
}

TEST(WorldLight, AMoonIsALightAtNightAndNotByDay)
{
  // a moon as Earth's: 1737 km at 384,400 km, opposite the sun
  for(const bool day : {true, false})
  {
    Sky s = planetOnly(day ? glm::dvec3(0.3, 1.0, 0.0) : glm::dvec3(0.3, -1.0, 0.0));
    Body moon;
    moon.centre = glm::normalize(glm::dvec3(-0.3, day ? 0.2 : 1.0, 0.0)) * 384400.0e3;
    moon.radius = 1737.0e3;
    moon.color  = glm::vec3(0.12F);
    s.bodies    = {moon};
    const Lights l = sumLights(s, glm::dvec3(0.0));
    EXPECT_EQ(l.lit[0], !day) << day;
    EXPECT_GT(luminance(l.shine[0].light), 0.0F);
    // about a full moon's 2.5e-6 of the sun at night (gibbous)
    if(!day)
      EXPECT_NEAR(std::log10(luminance(l.shine[0].light)), std::log10(2.5e-6), 0.5);
  }
}

}  // namespace
