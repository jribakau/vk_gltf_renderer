// The path tracer's per-pixel stop (shaders/adaptive_stop.h.slang) compiled as C++: a pixel stops once
// its mean's standard error is small against the root of its luminance, never before its minimum.

#include <cmath>
#include <random>

#include <gtest/gtest.h>

using std::sqrt;
#include "../shaders/adaptive_stop.h.slang"

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
  EXPECT_FALSE(adaptiveConverged(m.second, m.mean, 31, 0.1f, 32));
  EXPECT_TRUE(adaptiveConverged(m.second, m.mean, 32, 0.1f, 32));
}

TEST(AdaptiveStop, ANoisyPixelGoesOnUntilItsErrorIsSmall)
{
  // sigma 0.6 at luminance 0.5: the standard error at n samples is 0.6 / sqrt(n), against 0.1 sqrt(0.5)
  // = 0.0707, so it converges past n = 72
  const Moments early = sample(0.5f, 0.6f, 32, 2);
  EXPECT_FALSE(adaptiveConverged(early.second, early.mean, 32, 0.1f, 32));
  const Moments late = sample(0.5f, 0.6f, 128, 3);
  EXPECT_TRUE(adaptiveConverged(late.second, late.mean, 128, 0.1f, 32));
}

TEST(AdaptiveStop, ADarkPixelStopsSooner)
{
  // the same relative noise: the dark pixel's error is smaller against the root of its luminance
  const Moments bright = sample(1.0f, 1.2f, 48, 4);
  const Moments dark   = sample(0.01f, 0.012f, 48, 4);
  EXPECT_FALSE(adaptiveConverged(bright.second, bright.mean, 48, 0.1f, 32));
  EXPECT_TRUE(adaptiveConverged(dark.second, dark.mean, 48, 0.1f, 32));
}

TEST(AdaptiveStop, TheThresholdFollowsTheExposure)
{
  EXPECT_FLOAT_EQ(adaptiveSceneThreshold(0.1f, 1.0f), 0.1f);
  // four times the exposure shows the noise four times brighter and the root of the mean twice: half
  EXPECT_FLOAT_EQ(adaptiveSceneThreshold(0.1f, 4.0f), 0.05f);
  EXPECT_GT(adaptiveSceneThreshold(0.1f, 0.0f), 0.0f);  // never a division by zero
}
