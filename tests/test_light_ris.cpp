// The light sample's resampling (shaders/light_ris.h.slang) compiled as C++: a candidate is kept in
// proportion to its weight, the scaled estimate is unbiased with the shadowing outside the weights and
// with candidates of different kinds mixed, one candidate is today's single sample, and what is not
// finite or weighs nothing is never kept.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
// as test_bsdf_lobes.cpp has them (the same inline definitions)
namespace shaderio {
inline uint32_t asuint(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float asfloat(uint32_t u)
{
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace shaderio
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

// The row's shadowed integral over [0, end)
vec3 referenceIntegral(float end = 1.0f)
{
  vec3      sum(0.0f);
  const int n = 1 << 20;
  for(int i = 0; i < n; i++)
  {
    const float x = (float(i) + 0.5f) / float(n);
    if(x < end)
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

TEST(LightReuse, ARecordSurvivesItsPacking)
{
  LightReuseRecord r;
  r.pos    = vec3(1.5f, -2.25f, 300.125f);
  r.normal = normalize(vec3(0.3f, -0.8f, -0.5f));
  r.kind   = LIGHT_REUSE_EMISSIVE;
  r.index  = 123456789u & 0x3FFFFFFFu;
  r.uv     = vec2(0.25f, 0.7f);
  r.W      = 3.5e-3f;
  r.M      = 17.0f;
  LightReuseRecord u = lightReuseUnpack(lightReusePackA(r), lightReusePackB(r));
  EXPECT_EQ(u.pos, r.pos);
  EXPECT_GT(dot(u.normal, r.normal), 0.99999f);
  EXPECT_EQ(u.kind, r.kind);
  EXPECT_EQ(u.index, r.index);
  EXPECT_NEAR(u.uv.x, 0.25f, 1e-4f);
  EXPECT_NEAR(u.uv.y, 0.7f, 1e-4f);
  EXPECT_EQ(u.W, r.W);
  EXPECT_EQ(u.M, r.M);

  // an environment direction keeps its sign and its last bit of the sphere
  r.kind = LIGHT_REUSE_ENVIRONMENT;
  for(vec3 d : {vec3(0, 0, -1), vec3(0.6f, -0.48f, -0.64f), vec3(-0.001f, 0.99999f, 0.0f)})
  {
    r.uv = lightReuseOctEncode(normalize(d));
    u    = lightReuseUnpack(lightReusePackA(r), lightReusePackB(r));
    EXPECT_GT(dot(lightReuseOctDecode(u.uv), normalize(d)), 0.999999f) << d.x << " " << d.y << " " << d.z;
  }
}

TEST(LightReuse, ABufferNoFrameWroteIsNoRecord)
{
  const LightReuseRecord zero = lightReuseUnpack(uvec4(0), uvec4(0));
  EXPECT_EQ(zero.M, 0.0f);
  const uint32_t nan = 0x7FC00000u;
  EXPECT_EQ(lightReuseUnpack(uvec4(0), uvec4(0, 0, nan, asuint(4.0f))).M, 0.0f);
}

TEST(LightReuse, TheRecordsHistoryIsCappedAndShared)
{
  LightReuseRecord r = lightReuseUnpack(uvec4(0), uvec4(0));
  r.M                = 1000.0f;
  r.W                = 2.0f;
  const float recordM = lightReuseRecordM(r, 4.0f, 20.0f);
  EXPECT_FLOAT_EQ(recordM, 80.0f);
  EXPECT_FLOAT_EQ(lightReuseCandidateShare(true, 4.0f, recordM), 1.0f / 84.0f);
  EXPECT_FLOAT_EQ(lightReuseCandidateShare(false, 4.0f, recordM), 1.0f / 4.0f);  // no record holds a punctual light
  EXPECT_FLOAT_EQ(lightReuseRecordWeight(r, 0.5f, 4.0f, recordM), 80.0f / 84.0f * 0.5f * 2.0f);
  // the M weights of a light a record holds sum to one: 4 candidates' and the record's
  EXPECT_NEAR(4.0f * lightReuseCandidateShare(true, 4.0f, recordM) + recordM / (4.0f + recordM), 1.0f, 1e-6f);
}

// A pixel's light over frames at one shading point, as the shader does it: each frame's candidates
// are a point light (no record can hold it) or a point of a row of lamps along [0, 1), drawn
// uniformly, some taking nothing; part of the row is shadowed. Each candidate weighs its M weight
// times its estimate, the record its M weight times its target here times W; the kept sample is
// shaded with its shadow and carried on, W 0 if it was shadowed or is the point light, M always.
// Every frame's estimate must average to the point light plus the row's shadowed integral, and the
// history must make the later frames less noisy.
TEST(LightReuse, AChainOverFramesStaysUnbiased)
{
  const vec3  pointLight(0.08f, 0.06f, 0.02f);
  const float pickPoint = 0.2f;
  const vec3  reference = pointLight + referenceIntegral(0.93f);  // the samples past 0.93 take nothing: no light there
  const int   frames = 12, trials = 200000, candidates = 2;
  std::vector<dvec3>  sum(frames, dvec3(0.0));
  std::vector<double> sum2(frames, 0.0);
  std::mt19937                          rng(321);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for(int t = 0; t < trials; t++)
  {
    LightReuseRecord record = lightReuseUnpack(uvec4(0), uvec4(0));
    for(int f = 0; f < frames; f++)
    {
      const float    recordM = lightReuseRecordM(record, float(candidates), 20.0f);
      LightReservoir r       = lightReservoirEmpty();
      bool           keptPoint = false;
      float          keptX     = 0.0f;
      for(int c = 0; c < candidates; c++)
      {
        const bool point = uniform(rng) < pickPoint;
        const float x    = uniform(rng);
        float raw = 0.0f;
        if(point)
          raw = luminance709(pointLight / pickPoint);
        else if(x < 0.93f)  // past it the sample takes nothing, as a cut node below the horizon does
          raw = luminance709(integrand(x) / (1.0f - pickPoint));
        r = lightReservoirAdd(r, lightReuseCandidateShare(!point, float(candidates), recordM) * raw, uniform(rng));
        if(r.keptLast)
        {
          keptPoint = point;
          keptX     = x;
        }
      }
      // the record re-weighed here: the same point, so its target is the row's luminance at its sample
      const float recordTarget = record.M > 0.0f ? luminance709(integrand(record.uv.x)) : 0.0f;
      r                        = lightReservoirAdd(r, lightReuseRecordWeight(record, recordTarget, float(candidates), recordM), uniform(rng));
      const bool  tookRecord = r.keptLast;
      const bool  point      = !tookRecord && keptPoint;
      const float y          = tookRecord ? record.uv.x : keptX;
      const vec3  fy         = point ? pointLight : integrand(y);  // the sample's own measure: the light, or the row's
      const float W          = r.kept > 0.0f ? r.weightSum / luminance709(fy) : 0.0f;
      const vec3  e          = (point ? 1.0f : visibility(y)) * fy * W;
      sum[f] += dvec3(e);
      sum2[f] += double(luminance709(e)) * luminance709(e);
      record.uv = vec2(y, 0.0f);
      record.W  = (!point && visibility(y) > 0.0f) ? W : 0.0f;
      record.M  = float(candidates) + recordM;
    }
  }
  for(int f = 0; f < frames; f++)
    for(int ch = 0; ch < 3; ch++)
      EXPECT_NEAR(sum[f][ch] / trials, reference[ch], 0.015 * reference[ch]) << "frame " << f << ", channel " << ch;
  auto variance = [&](int f) {
    const double mean = luminance709(vec3(sum[f] / double(trials)));
    return sum2[f] / trials - mean * mean;
  };
  EXPECT_LT(variance(frames - 1), 0.6 * variance(0)) << "the history should cut the noise";
}

}  // namespace
