// A woven cutout's opacity (shaders/woven_opacity.h.slang) compiled as C++: the authored opacity
// head-on, the gaps of one thread set closing as the cloth tilts across it, and the same answer for
// either side and either direction of travel.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace glm;
#include "../shaders/shaderio.h"
using namespace shaderio;
#include "../shaders/woven_opacity.h.slang"

namespace {

const float3 kNormal(0, 0, 1);
const float3 kWarp(1, 0, 0);

// a direction `degrees` off the normal, turned `azimuth` degrees from the warp
float3 tilted(float degrees, float azimuth = 0)
{
  const float t = glm::radians(degrees), a = glm::radians(azimuth);
  return float3(std::sin(t) * std::cos(a), std::sin(t) * std::sin(a), std::cos(t));
}

}  // namespace

TEST(WovenOpacity, HeadOnItIsTheAuthoredOpacity)
{
  for(float alpha : {0.05f, 0.3f, 0.7f, 0.95f})
    EXPECT_NEAR(wovenOpacity(alpha, kNormal, kNormal, kWarp), alpha, 1e-6f) << alpha;
}

TEST(WovenOpacity, AHoleAndSolidClothStayAsTheyAre)
{
  EXPECT_EQ(wovenOpacity(0.0f, tilted(80), kNormal, kWarp), 0.0f);
  EXPECT_EQ(wovenOpacity(1.0f, tilted(80), kNormal, kWarp), 1.0f);
}

TEST(WovenOpacity, TheGapsCloseAsTheClothTurnsAway)
{
  float last = 0.3f;
  for(float degrees = 10; degrees < 90; degrees += 10)
  {
    const float seen = wovenOpacity(0.3f, tilted(degrees, 30), kNormal, kWarp);
    EXPECT_GT(seen, last) << degrees;
    last = seen;
  }
  EXPECT_NEAR(wovenOpacity(0.3f, tilted(90), kNormal, kWarp), 1.0f, 1e-6f);
}

TEST(WovenOpacity, LookingAlongTheWarpOnlyTheWeftCloses)
{
  // in the warp's plane the warp's gaps run with the view; the weft's foreshorten by cos
  const float alpha = 0.3f, c = 1 - std::sqrt(1 - alpha), cosine = std::cos(glm::radians(50.0f));
  EXPECT_NEAR(wovenOpacity(alpha, tilted(50, 0), kNormal, kWarp), 1 - (1 - c) * (1 - c / cosine), 1e-5f);
  // the same tilt across both sets at once closes more
  EXPECT_GT(wovenOpacity(alpha, tilted(50, 45), kNormal, kWarp), wovenOpacity(alpha, tilted(50, 0), kNormal, kWarp));
}

TEST(WovenOpacity, EitherSideAndEitherWayGiveTheSame)
{
  const float3 v    = tilted(65, 20);
  const float  seen = wovenOpacity(0.4f, v, kNormal, kWarp);
  EXPECT_NEAR(wovenOpacity(0.4f, -v, kNormal, kWarp), seen, 1e-6f);
  EXPECT_NEAR(wovenOpacity(0.4f, v, -kNormal, kWarp), seen, 1e-6f);
  EXPECT_NEAR(wovenOpacity(0.4f, v, kNormal, -kWarp), seen, 1e-6f);
}

TEST(WovenOpacity, AnyLengthAndATiltedWarpServe)
{
  const float3 v    = tilted(60, 35);
  const float  seen = wovenOpacity(0.25f, v, kNormal, kWarp);
  // unnormalised, and a warp leaning out of the plane (an interpolated tangent)
  EXPECT_NEAR(wovenOpacity(0.25f, v * 3.0f, kNormal * 0.2f, kWarp * 5.0f + kNormal), seen, 1e-5f);
  // a warp along the normal has no direction in the plane: some axis there, and still an opacity
  const float degenerate = wovenOpacity(0.25f, v, kNormal, kNormal);
  EXPECT_TRUE(std::isfinite(degenerate));
  EXPECT_GT(degenerate, 0.25f);
  EXPECT_LE(degenerate, 1.0f);
}
