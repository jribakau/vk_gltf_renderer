// The light sample's resampling (shaders/light_ris.h.slang) compiled as C++: a candidate is kept in
// proportion to its weight, the scaled estimate is unbiased with the shadowing outside the weights and
// with candidates of different kinds mixed, one candidate is today's single sample, and what is not
// finite or weighs nothing is never kept.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/light_ris.h.slang"

namespace {

float luminance709(vec3 c)
{
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

TEST(LightRis, KeepsACandidateInProportionToItsWeight)
{
  const std::vector<float> weights = {1.0f, 0.0f, 3.0f, 6.0f};
  std::mt19937             rng(7);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<int>                      kept(weights.size(), 0);
  const int                             trials = 200000;
  for(int t = 0; t < trials; t++)
  {
    LightReservoir r    = lightReservoirEmpty();
    int            pick = -1;
    for(size_t i = 0; i < weights.size(); i++)
    {
      r = lightReservoirAdd(r, weights[i], uniform(rng));
      if(r.keptLast)
        pick = int(i);
    }
    ASSERT_GE(pick, 0);
    kept[pick]++;
  }
  EXPECT_EQ(kept[1], 0);  // weight 0: never kept
  EXPECT_NEAR(kept[0] / double(trials), 0.1, 0.004);
  EXPECT_NEAR(kept[2] / double(trials), 0.3, 0.004);
  EXPECT_NEAR(kept[3] / double(trials), 0.6, 0.004);
}

TEST(LightRis, OneCandidateIsTheSingleSample)
{
  LightReservoir r = lightReservoirAdd(lightReservoirEmpty(), 0.25f, 0.9f);
  EXPECT_TRUE(r.keptLast);
  EXPECT_FLOAT_EQ(lightReservoirScale(r), 1.0f);
}

TEST(LightRis, NothingToKeepScalesToZero)
{
  LightReservoir r = lightReservoirEmpty();
  r                = lightReservoirAdd(r, 0.0f, 0.5f);
  r                = lightReservoirAdd(r, std::numeric_limits<float>::quiet_NaN(), 0.5f);
  r                = lightReservoirAdd(r, std::numeric_limits<float>::infinity(), 0.5f);
  r                = lightReservoirAdd(r, -1.0f, 0.5f);
  EXPECT_FALSE(r.keptLast);
  EXPECT_EQ(r.count, 4);
  EXPECT_EQ(lightReservoirScale(r), 0.0f);
}

TEST(LightRis, NotFiniteAddsNothingToWhatIsKept)
{
  LightReservoir r = lightReservoirEmpty();
  r                = lightReservoirAdd(r, 2.0f, 0.5f);
  r                = lightReservoirAdd(r, std::numeric_limits<float>::quiet_NaN(), 0.0f);
  EXPECT_FALSE(r.keptLast);
  EXPECT_FLOAT_EQ(r.weightSum, 2.0f);
  EXPECT_FLOAT_EQ(lightReservoirScale(r), 0.5f);  // mean over both candidates: 2 / (2 * 2)
}

// A vertex lit by a row of lamps along [0, 1): x drawn uniformly (as a pick that knows nothing of
// the BSDF), a coloured integrand f(x) whose BSDF favours one end, and a shadow over part of the row.
// The resampled estimate, shadow applied to the kept candidate only, must average to the integral of
// V f, channel by channel, for every candidate count.
vec3 integrand(float x)
{
  const float bsdf = std::pow(x, 6.0f) * 7.0f;               // a glossy lobe toward x = 1
  return vec3(1.0f, 0.6f + 0.4f * x, 0.2f + x * x) * (0.05f + bsdf);  // the lamps' colour changes along the row
}

float visibility(float x)
{
  return (x > 0.7f && x < 0.85f) ? 0.0f : 1.0f;  // an occluder over part of the bright end
}

vec3 referenceIntegral()
{
  vec3      sum(0.0f);
  const int n = 1 << 20;
  for(int i = 0; i < n; i++)
  {
    const float x = (float(i) + 0.5f) / float(n);
    sum += visibility(x) * integrand(x);
  }
  return sum / float(n);
}

TEST(LightRis, TheScaledEstimateIsUnbiased)
{
  const vec3 reference = referenceIntegral();
  for(int candidates : {1, 2, 4, 8, 16})
  {
    std::mt19937                          rng(1234 + candidates);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const int                             trials = 400000;
    dvec3                                 sum(0.0);
    for(int t = 0; t < trials; t++)
    {
      LightReservoir r = lightReservoirEmpty();
      vec3           keptEstimate(0.0f);
      float          keptX = 0.0f;
      for(int c = 0; c < candidates; c++)
      {
        const float x        = uniform(rng);
        const vec3  estimate = integrand(x) / 1.0f;  // over the uniform pdf
        r                    = lightReservoirAdd(r, luminance709(estimate), uniform(rng));
        if(r.keptLast)
        {
          keptEstimate = estimate;
          keptX        = x;
        }
      }
      sum += dvec3(visibility(keptX) * keptEstimate * lightReservoirScale(r));
    }
    const dvec3 mean = sum / double(trials);
    for(int ch = 0; ch < 3; ch++)
      EXPECT_NEAR(mean[ch], reference[ch], 0.01 * reference[ch]) << candidates << " candidates, channel " << ch;
  }
}

// Candidates of different kinds in one reservoir: a point light (a fixed estimate: its intensity over
// its pick probability) beside an area light sampled over [0, 1), picked between them at random as
// sampleLights does. Each candidate's weight is its own estimate's luminance, in whatever measure it
// was drawn in, and the mixture's expectation stays the sum of both lights.
TEST(LightRis, PointAndAreaCandidatesMix)
{
  const vec3  pointLight(0.4f, 0.3f, 0.1f);  // what the point light gives the vertex
  const float pickPoint = 0.3f;
  const vec3  reference = pointLight + referenceIntegral();  // the point light, and the row with its shadow
  for(int candidates : {1, 4})
  {
    std::mt19937                          rng(99 + candidates);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const int                             trials = 400000;
    dvec3                                 sum(0.0);
    for(int t = 0; t < trials; t++)
    {
      LightReservoir r = lightReservoirEmpty();
      vec3           keptEstimate(0.0f);
      float          keptVisibility = 0.0f;
      for(int c = 0; c < candidates; c++)
      {
        vec3  estimate;
        float v;
        if(uniform(rng) < pickPoint)
        {
          estimate = pointLight / pickPoint;
          v        = 1.0f;
        }
        else
        {
          const float x = uniform(rng);
          estimate      = integrand(x) / (1.0f - pickPoint);
          v             = visibility(x);
        }
        r = lightReservoirAdd(r, luminance709(estimate), uniform(rng));
        if(r.keptLast)
        {
          keptEstimate   = estimate;
          keptVisibility = v;
        }
      }
      sum += dvec3(keptVisibility * keptEstimate * lightReservoirScale(r));
    }
    const dvec3 mean = sum / double(trials);
    for(int ch = 0; ch < 3; ch++)
      EXPECT_NEAR(mean[ch], reference[ch], 0.01 * reference[ch]) << candidates << " candidates, channel " << ch;
  }
}

// What the resampling is for: with the integrand's bright end narrow, more candidates give the same
// mean with less spread
TEST(LightRis, MoreCandidatesLessNoise)
{
  auto variance = [](int candidates) {
    std::mt19937                          rng(5);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const int                             trials = 100000;
    double                                sum = 0.0, sum2 = 0.0;
    for(int t = 0; t < trials; t++)
    {
      LightReservoir r = lightReservoirEmpty();
      float          kept = 0.0f;
      for(int c = 0; c < candidates; c++)
      {
        const float e = luminance709(integrand(uniform(rng)));
        r             = lightReservoirAdd(r, e, uniform(rng));
        if(r.keptLast)
          kept = e;
      }
      const double v = kept * lightReservoirScale(r);
      sum += v;
      sum2 += v * v;
    }
    const double mean = sum / trials;
    return sum2 / trials - mean * mean;
  };
  const double one = variance(1), four = variance(4), eight = variance(8);
  EXPECT_LT(four, 0.5 * one);
  EXPECT_LT(eight, four);
}

}  // namespace
