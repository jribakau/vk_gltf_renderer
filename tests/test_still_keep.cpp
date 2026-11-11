// The still's per-pixel Keep detail (shaders/still_keep.h.slang): a pixel's noise from its luminance
// moments, and the share of its raw value it keeps over the denoised image.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <nvshaders/slang_types.h>

#include "../shaders/still_keep.h.slang"

namespace {

struct Moments
{
  float second = 0.0f;  // mean of the samples' luminance squared
  float mean   = 0.0f;  // mean luminance
};

Moments momentsOf(const std::vector<float>& samples)
{
  double sum = 0.0, sumSq = 0.0;
  for(const float s : samples)
  {
    sum += s;
    sumSq += double(s) * s;
  }
  return {float(sumSq / samples.size()), float(sum / samples.size())};
}

}  // namespace

TEST(StillKeep, AConvergedPixelIsKeptAndANoisyOneIsNot)
{
  const float threshold = 0.02f;
  // every sample the same: no noise, the raw value kept whole
  EXPECT_FLOAT_EQ(stillKeepShare(stillKeepNoise(0.25f, 0.5f, 64), threshold), 1.0f);
  // the noise at the threshold: half; at half of it or less, all; at one and a half times, none
  const auto at = [&](float noise) { return stillKeepShare(noise * noise, threshold); };
  EXPECT_NEAR(at(threshold), 0.5f, 1e-5f);
  EXPECT_FLOAT_EQ(at(0.5f * threshold), 1.0f);
  EXPECT_FLOAT_EQ(at(1.5f * threshold), 0.0f);
  EXPECT_GT(at(0.8f * threshold), at(1.2f * threshold));
}

TEST(StillKeep, TooFewSamplesOrOffKeepsNothing)
{
  EXPECT_LT(stillKeepNoise(0.25f, 0.5f, STILL_KEEP_MIN_SAMPLES - 1), 0.0f);
  EXPECT_FLOAT_EQ(stillKeepShare(stillKeepNoise(0.25f, 0.5f, STILL_KEEP_MIN_SAMPLES - 1), 0.02f), 0.0f);
  EXPECT_FLOAT_EQ(stillKeepShare(0.0f, 0.0f), 0.0f);  // the setting at 0: off
}

// The adaptive stop's measure: the same spread relative to the brightness counts for less in a dark
// pixel, whose noise shows less; and more samples, less noise
TEST(StillKeep, DarkerAndLongerPixelsAreQuieter)
{
  const auto noiseOf = [](float mean, float relativeSpread, int samples) {
    const float sd = relativeSpread * mean;
    return stillKeepNoise(mean * mean + sd * sd * float(samples - 1) / float(samples), mean, samples);
  };
  EXPECT_LT(noiseOf(0.05f, 0.5f, 64), noiseOf(0.5f, 0.5f, 64));
  EXPECT_LT(noiseOf(0.5f, 0.5f, 256), noiseOf(0.5f, 0.5f, 64));
}

// From samples drawn at random, the noise is the mean's standard error squared over the brightness
TEST(StillKeep, TheNoiseIsTheStandardErrorOverTheRootOfTheBrightness)
{
  std::mt19937                    rng(7);
  std::normal_distribution<float> draw(0.4f, 0.12f);
  std::vector<float>              samples(64);
  for(float& s : samples)
    s = draw(rng);
  const Moments m        = momentsOf(samples);
  const float   expected = 0.12f * 0.12f / 64.0f / 0.4f;
  EXPECT_NEAR(stillKeepNoise(m.second, m.mean, 64), expected, 0.35f * expected);

  // a firefly among them: far noisier, nothing kept at a threshold the quiet pixel passes
  const float quiet = stillKeepNoise(m.second, m.mean, 64);
  samples[10]       = 40.0f;
  const Moments f   = momentsOf(samples);
  const float threshold = 1.2f * std::sqrt(quiet);
  EXPECT_GT(stillKeepShare(quiet, threshold), 0.5f);
  EXPECT_FLOAT_EQ(stillKeepShare(stillKeepNoise(f.second, f.mean, 64), threshold), 0.0f);
}
