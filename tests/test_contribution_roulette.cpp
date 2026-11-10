// Russian roulette by what a ray can add (shaders/contribution_roulette.h.slang), as the path tracer
// takes the air's sun-shadow ray and a still's light samples: traced always once it can matter, else
// with a chance that keeps the estimate unbiased and the noise it adds under the share of the pixel.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include <nvshaders/slang_types.h>

#include "../shaders/contribution_roulette.h.slang"

// Always once the term reaches the share of what the pixel holds; never for nothing; always for a
// pixel that holds nothing else, or with no share to spend
TEST(ContributionRoulette, TheChanceFollowsTheTermsShareOfThePixel)
{
  EXPECT_FLOAT_EQ(contributionTraceChance(0.05F, 1.0F, 0.05F), 1.0F);
  EXPECT_FLOAT_EQ(contributionTraceChance(2.0F, 1.0F, 0.05F), 1.0F);
  EXPECT_FLOAT_EQ(contributionTraceChance(0.001F, 1.0F, 0.05F), 0.02F);
  EXPECT_FLOAT_EQ(contributionTraceChance(0.001F, 4.0F, 0.05F), 0.005F);
  EXPECT_EQ(contributionTraceChance(0.0F, 1.0F, 0.05F), 0.0F);
  EXPECT_EQ(contributionTraceChance(0.001F, 0.0F, 0.05F), 1.0F);
  EXPECT_EQ(contributionTraceChance(0.001F, 1.0F, 0.0F), 1.0F);  // ptAerialSunShare 0: every path
}

// As a still's light samples take it (ptShadowShare, the rest the pixel's mean so far): a sample under
// the share of the pixel is weighed up to exactly that share, so none it lets through is brighter
// than a sample it never touches, and each adds at most the share squared of the pixel's mean squared
TEST(ContributionRoulette, ASampleUnderTheShareIsWeighedUpToIt)
{
  const float share = 0.05F;
  const float rest  = 3.0F;
  for(const float term : {1e-5F, 1e-3F, 0.1F, 0.14F})
  {
    const float chance = contributionTraceChance(term, rest, share);
    ASSERT_LT(chance, 1.0F) << term;
    EXPECT_NEAR(term / chance, share * rest, 1e-5F) << term;
    EXPECT_LE(term * term * (1.0F / chance - 1.0F), share * share * rest * rest) << term;
  }
  EXPECT_EQ(contributionTraceChance(0.16F, rest, share), 1.0F);
  EXPECT_EQ(contributionTraceChance(5.0F, rest, share), 1.0F);
}

// Traced with that chance and weighed by one over it, the term's mean is what tracing it every time
// gives, and its variance is at most share * term * rest: a relative deviation of at most the share
TEST(ContributionRoulette, TheEstimateIsUnbiasedAndItsNoiseBoundedByTheShare)
{
  const float share = 0.05F;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(0.0F, 1.0F);
  for(const float term : {1e-4F, 3e-3F, 0.02F, 0.2F})
    for(const float visible : {0.0F, 0.4F, 1.0F})
    {
      const float rest   = 1.0F;
      const float chance = contributionTraceChance(term, rest, share);
      const int   n      = 400000;
      double      sum = 0.0, sum2 = 0.0;
      for(int i = 0; i < n; i++)
      {
        const double v = (chance >= 1.0F || u(rng) < chance) ? double(visible) * term / chance : 0.0;
        sum += v;
        sum2 += v * v;
      }
      const double mean = sum / n;
      const double var  = sum2 / n - mean * mean;
      const double exact = double(visible) * term;
      // the mean within four of its standard errors
      EXPECT_NEAR(mean, exact, 4.0 * std::sqrt(std::max(var, 1e-30) / n) + 1e-12) << term << " " << visible;
      EXPECT_LE(var, 1.02 * share * term * rest + 1e-12) << term << " " << visible;
    }
}
