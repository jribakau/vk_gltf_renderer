/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Where the sun's shadow map goes (src/shadow_fit.hpp): it holds every caster, reaches the ground
// the floor shows, and moves in whole texels so small changes in the bounds don't make edges crawl.

#include <cmath>

#include <gtest/gtest.h>

#include "shadow_fit.hpp"

namespace {

constexpr float kPi = 3.14159265F;

glm::vec3 toLight(float azimuthDeg, float elevationDeg)
{
  const float az = azimuthDeg * kPi / 180.0F, el = elevationDeg * kPi / 180.0F;
  return {std::cos(el) * std::sin(az), std::sin(el), std::cos(el) * std::cos(az)};
}

glm::vec3 project(const shadow_fit::Result& r, const glm::vec3& p)
{
  const glm::vec4 c = r.viewProj * glm::vec4(p, 1.0F);
  return glm::vec3(c) / c.w;
}

bool inside(const glm::vec3& ndc, float eps = 1e-4F)
{
  return std::abs(ndc.x) <= 1.0F + eps && std::abs(ndc.y) <= 1.0F + eps && ndc.z >= -eps && ndc.z <= 1.0F + eps;
}

}  // namespace

TEST(ShadowFit, BoxHoldsTheSphere)
{
  for(float el : {80.0F, 45.0F, 12.0F, 89.9F})
    for(float az : {0.0F, 35.0F, -120.0F, 180.0F})
    {
      shadow_fit::Input in;
      in.center  = {0.3F, 0.9F, -0.2F};
      in.radius  = 0.95F;
      in.toLight = toLight(az, el);
      const shadow_fit::Result r = shadow_fit::fit(in);
      ASSERT_TRUE(r.valid);
      // the poles and a ring of points on the sphere
      for(int i = 0; i < 64; i++)
      {
        const float     t = float(i) / 64.0F * 2.0F * kPi;
        const glm::vec3 dirs[] = {{std::cos(t), 0, std::sin(t)}, {std::cos(t), std::sin(t), 0}, {0, std::cos(t), std::sin(t)}};
        for(const glm::vec3& d : dirs)
          EXPECT_TRUE(inside(project(r, in.center + d * in.radius))) << "el " << el << " az " << az;
      }
    }
}

TEST(ShadowFit, FarPlaneReachesTheGroundAtALowSun)
{
  shadow_fit::Input in;
  in.center      = {0.0F, 0.9F, 0.0F};
  in.radius      = 0.95F;
  in.toLight     = toLight(30.0F, 8.0F);
  in.groundY     = 0.0F;
  in.groundReach = shadow_fit::groundReach(in.radius, 3.0F);
  const shadow_fit::Result r = shadow_fit::fit(in);
  ASSERT_TRUE(r.valid);
  // every point of the ground disk that lies inside the box's sides is in front of its far plane
  int counted = 0;
  for(int i = 0; i < 128; i++)
  {
    const float     t = float(i) / 128.0F * 2.0F * kPi;
    const glm::vec3 g(std::cos(t) * in.groundReach, in.groundY, std::sin(t) * in.groundReach);
    const glm::vec3 ndc = project(r, g);
    if(std::abs(ndc.x) > 1.0F || std::abs(ndc.y) > 1.0F)
      continue;
    counted++;
    EXPECT_LE(ndc.z, 1.0F + 1e-4F);
    EXPECT_GE(ndc.z, 0.0F);
  }
  EXPECT_GT(counted, 0);

  // without the ground the box ends at the sphere, so the same far point falls behind it
  in.groundReach              = 0.0F;
  const shadow_fit::Result r0 = shadow_fit::fit(in);
  EXPECT_LT(r0.farDepth, r.farDepth);
}

TEST(ShadowFit, SmallChangesMoveTheBoxInWholeTexels)
{
  shadow_fit::Input in;
  in.center  = {0.1F, 0.9F, 0.05F};
  in.radius  = 0.95F;
  in.toLight = toLight(35.0F, 50.0F);
  in.mapSize = 4096;
  const shadow_fit::Result r0 = shadow_fit::fit(in);
  ASSERT_TRUE(r0.valid);
  const glm::vec3 fixedPoint(0.37F, 1.21F, -0.4F);
  const glm::vec3 p0 = project(r0, fixedPoint);

  for(int i = 1; i <= 20; i++)
  {
    shadow_fit::Input moved = in;
    moved.center += glm::vec3(0.0013F * float(i), -0.0007F * float(i), 0.0009F * float(i));
    moved.radius *= 1.0F + 0.0005F * float(i);  // a breath of animation
    const shadow_fit::Result r = shadow_fit::fit(moved);
    ASSERT_TRUE(r.valid);
    EXPECT_EQ(r.halfExtent, r0.halfExtent);
    EXPECT_EQ(r.texelWorld, r0.texelWorld);
    // the same world point lands on the map a whole number of texels from where it was
    const glm::vec3 p      = project(r, fixedPoint);
    const float     texels = 0.5F * float(in.mapSize);  // NDC units to texels
    const float     dx = (p.x - p0.x) * texels, dy = (p.y - p0.y) * texels;
    EXPECT_NEAR(dx, std::round(dx), 0.02F) << i;
    EXPECT_NEAR(dy, std::round(dy), 0.02F) << i;
  }
}

TEST(ShadowFit, ExtentIsQuantized)
{
  EXPECT_EQ(shadow_fit::quantizeExtent(0.0F), 0.0F);
  EXPECT_GE(shadow_fit::quantizeExtent(0.97F), 0.97F);
  EXPECT_EQ(shadow_fit::quantizeExtent(0.97F), shadow_fit::quantizeExtent(0.971F));
  // steps of 1/16 of the power of two below: never more than 1/16 too big
  for(float r : {0.013F, 0.5F, 0.97F, 3.3F, 120.0F})
    EXPECT_LE(shadow_fit::quantizeExtent(r), r * (1.0F + 1.0F / 16.0F) + 1e-6F);
}

TEST(ShadowFit, DegenerateInputIsInvalid)
{
  shadow_fit::Input in;
  in.radius = 0.0F;
  EXPECT_FALSE(shadow_fit::fit(in).valid);
  in.radius  = 1.0F;
  in.toLight = glm::vec3(0.0F);
  EXPECT_FALSE(shadow_fit::fit(in).valid);
}
