// The path tracer's per-pixel stop (shaders/adaptive_stop.h.slang) compiled as C++: a pixel stops once
// its mean's standard error is small against the root of its luminance, never before its minimum.

#include <cmath>
#include <random>

#include <gtest/gtest.h>

using std::log2;
using std::sqrt;
#include "../shaders/adaptive_stop.h.slang"
#include "../src/adaptive_relax.hpp"

namespace {

// The moments of n samples of a pixel: luminance `value` with a Gaussian spread `sigma`
struct Moments
{
  float mean = 0, second = 0;
};

Moments sample(float value, float sigma, int n, unsigned seed)
{
  std::mt19937                    rng(seed);
  std::normal_distribution<float> noise(value, sigma);
  double                          sum = 0, sum2 = 0;
  for(int i = 0; i < n; i++)
  {
    const double x = noise(rng);
    sum += x;
    sum2 += x * x;
  }
  return {float(sum / n), float(sum2 / n)};
}

}  // namespace

TEST(AdaptiveStop, AFlatPixelStopsAtItsMinimum)
{
  const Moments m = sample(0.5f, 0.0f, 32, 1);
  EXPECT_FALSE(adaptiveConverged(m.second, m.mean, 31, 0.1f, 32, 1.0f));
  EXPECT_TRUE(adaptiveConverged(m.second, m.mean, 32, 0.1f, 32, 1.0f));
}

TEST(AdaptiveStop, ANoisyPixelGoesOnUntilItsErrorIsSmall)
{
  // sigma 0.6 at luminance 0.5: the standard error at n samples is 0.6 / sqrt(n), against 0.1 sqrt(0.5)
  // = 0.0707, so it converges past n = 72
  const Moments early = sample(0.5f, 0.6f, 32, 2);
  EXPECT_FALSE(adaptiveConverged(early.second, early.mean, 32, 0.1f, 32, 1.0f));
  const Moments late = sample(0.5f, 0.6f, 128, 3);
  EXPECT_TRUE(adaptiveConverged(late.second, late.mean, 128, 0.1f, 32, 1.0f));
}

TEST(AdaptiveStop, ADarkPixelStopsSooner)
{
  // the same relative noise: the dark pixel's error is smaller against the root of its luminance
  const Moments bright = sample(1.0f, 1.2f, 48, 4);
  const Moments dark   = sample(0.01f, 0.012f, 48, 4);
  EXPECT_FALSE(adaptiveConverged(bright.second, bright.mean, 48, 0.1f, 32, 1.0f));
  EXPECT_TRUE(adaptiveConverged(dark.second, dark.mean, 48, 0.1f, 32, 1.0f));
}

TEST(AdaptiveStop, TheThresholdFollowsTheExposure)
{
  EXPECT_FLOAT_EQ(adaptiveSceneThreshold(0.1f, 1.0f), 0.1f);
  // four times the exposure shows the noise four times brighter and the root of the mean twice: half
  EXPECT_FLOAT_EQ(adaptiveSceneThreshold(0.1f, 4.0f), 0.05f);
  EXPECT_GT(adaptiveSceneThreshold(0.1f, 0.0f), 0.0f);  // never a division by zero
}

TEST(AdaptiveStop, RelaxWidensTheBound)
{
  // sigma 6 at luminance 5: not converged at 64 samples under the plain rule, converged at four times it
  const Moments m = sample(5.0f, 6.0f, 64, 5);
  EXPECT_FALSE(adaptiveConverged(m.second, m.mean, 64, 0.1f, 32, 1.0f));
  EXPECT_TRUE(adaptiveConverged(m.second, m.mean, 64, 0.1f, 32, 4.0f));
  EXPECT_FALSE(adaptiveConverged(m.second, m.mean, 31, 0.1f, 32, 1.0e4f));  // never before the minimum
}

TEST(AdaptiveStop, RelaxPositionIsStopsAboveMidGrey)
{
  EXPECT_EQ(adaptiveRelaxPosition(0.1f, 0.18f), -1.0f);  // the shadows and midtones: no table entry
  EXPECT_EQ(adaptiveRelaxPosition(0.18f, 0.18f), -1.0f);
  EXPECT_EQ(adaptiveRelaxPosition(5.0f, 0.0f), -1.0f);   // no table
  EXPECT_NEAR(adaptiveRelaxPosition(0.18f * 8.0f, 0.18f), 3.0f, 1e-5f);
  EXPECT_EQ(adaptiveRelaxPosition(1.0e9f, 0.18f), float(ADAPTIVE_RELAX_STOPS - 1));
}

TEST(AdaptiveRelax, TheShoulderOfAFilmicCurveRelaxesHighlightsNotMidGrey)
{
  const adaptive_relax::Table t = adaptive_relax::build(shaderio::TonemapperData{});
  ASSERT_NEAR(t.base, 0.18f, 1e-6f);
  EXPECT_NEAR(t.relax[0], 1.0f, 0.05f);
  for(int i = 1; i < ADAPTIVE_RELAX_STOPS; i++)
    EXPECT_GE(t.relax[i], t.relax[i - 1] * 0.999f) << i;
  EXPECT_GT(t.relax[6], 3.0f);  // 11.5 in the scene, deep in the shoulder
}

TEST(AdaptiveRelax, TheTableFollowsTheExposure)
{
  shaderio::TonemapperData tm;
  const adaptive_relax::Table one = adaptive_relax::build(tm);
  tm.exposure                     = 4.0f;
  const adaptive_relax::Table four = adaptive_relax::build(tm);
  EXPECT_NEAR(four.base, 0.045f, 1e-6f);  // mid grey on screen is a quarter of the light in the scene
  for(int i = 0; i < ADAPTIVE_RELAX_STOPS; i++)
    EXPECT_NEAR(four.relax[i], one.relax[i], one.relax[i] * 1e-3f) << i;
}

TEST(AdaptiveRelax, AClippedCurveStopsWorryingPastWhite)
{
  shaderio::TonemapperData tm;
  tm.method                     = shaderio::TONEMAP_METHOD(eClip);
  const adaptive_relax::Table t = adaptive_relax::build(tm);
  EXPECT_LT(t.relax[1], 1.5f);  // 0.36, below white: the plain rule
  EXPECT_EQ(t.relax[4], adaptive_relax::kMaxRelax);  // 2.9, clipped
}

TEST(AdaptiveRelax, NoTableWithAutoExposure)
{
  shaderio::TonemapperData tm;
  tm.autoExposure = 1;
  EXPECT_EQ(adaptive_relax::build(tm).base, 0.0f);
}
