// An EXR dome's texels made fit to light by (src/env_texels.hpp): NaN, infinite and negative values,
// any one of which would make the dome's importance table non-finite, and then fed to that table's
// builder to show it stays finite.

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <nvvk/env_aliasmap.hpp>

#include "env_texels.hpp"

namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();
}  // namespace

TEST(EnvTexels, BadValuesBecomeFiniteAndAreCounted)
{
  // four texels: clean, a NaN, +Inf and a negative, -Inf
  std::vector<float> rgba = {0.5f, 1.0f, 2.0f, 1.0f,  //
                             kNaN, 0.2f, 0.3f, 1.0f,  //
                             kInf, 0.0f, 4.0f, 1.0f,  //
                             -0.1f, -kInf, 0.7f, 1.0f};
  const EnvTexelRepairs r = repairEnvTexels(rgba);
  EXPECT_EQ(r.nan, 1u);
  EXPECT_EQ(r.infinite, 1u);
  EXPECT_EQ(r.negative, 2u);
  EXPECT_EQ(r.total(), 4u);
  // the clean texel is untouched
  EXPECT_EQ(rgba[0], 0.5f);
  EXPECT_EQ(rgba[2], 2.0f);
  EXPECT_EQ(rgba[4], 0.0f);
  EXPECT_EQ(rgba[8], kEnvTexelBrightest);
  EXPECT_EQ(rgba[12], 0.0f);
  EXPECT_EQ(rgba[13], 0.0f);
  EXPECT_EQ(rgba[14], 0.7f);
  for(float v : rgba)
  {
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GE(v, 0.0f);
  }
}

TEST(EnvTexels, AlphaIsLeftAlone)
{
  std::vector<float> rgba = {1.0f, 1.0f, 1.0f, kNaN};
  EXPECT_EQ(repairEnvTexels(rgba).total(), 0u);
  EXPECT_TRUE(std::isnan(rgba[3]));
}

TEST(EnvTexels, ARepairedDomeHasAFiniteImportanceIntegral)
{
  // one NaN texel among clean ones: unrepaired, the alias table's integral is not finite
  const int          texels = 64;
  std::vector<float> rgba(texels * 4, 0.25f);
  rgba[4 * 17 + 1] = kNaN;
  rgba[4 * 40]     = kInf;
  auto integralOf  = [&](const std::vector<float>& image) {
    std::vector<float> importance(texels);
    for(int i = 0; i < texels; i++)
      importance[i] = std::max(image[i * 4], std::max(image[i * 4 + 1], image[i * 4 + 2]));
    std::vector<shaderio::EnvAccel> accel(texels);
    return nvvk::buildEnvAliasmap(importance, accel);
  };
  EXPECT_FALSE(std::isfinite(integralOf(rgba)));
  repairEnvTexels(rgba);
  const float integral = integralOf(rgba);
  EXPECT_TRUE(std::isfinite(integral));
  EXPECT_GT(integral, 0.0f);
}

TEST(EnvTexels, AnEmptyImageNeedsNothing)
{
  std::vector<float> none;
  EXPECT_EQ(repairEnvTexels(none).total(), 0u);
}
