// The light sample's resampling (shaders/light_ris.h.slang) compiled as C++: a candidate is kept in
// proportion to its weight, the scaled estimate is unbiased with the shadowing outside the weights and
// with candidates of different kinds mixed, one candidate is today's single sample, and what is not
// finite or weighs nothing is never kept.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <cstdio>
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
  r.depth   = 300.125f;
  r.normal  = normalize(vec3(0.3f, -0.8f, -0.5f));
  r.shading = normalize(vec3(0.35f, -0.75f, -0.5f));
  r.view    = normalize(vec3(-0.2f, 0.1f, 0.97f));
  r.kind   = LIGHT_REUSE_EMISSIVE;
  r.index  = 123456789u & 0x3FFFFFFFu;
  r.uv     = vec2(0.25f, 0.7f);
  r.W      = 3.5e-3f;
  r.M      = 17.0f;
  LightReuseRecord u = lightReuseUnpack(lightReusePackA(r), lightReusePackB(r));
  EXPECT_EQ(u.depth, r.depth);
  EXPECT_GT(dot(u.normal, r.normal), 0.99999f);
  EXPECT_GT(dot(u.shading, r.shading), 0.99999f);
  EXPECT_GT(dot(u.view, r.view), 0.99999f);
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

TEST(LightReuse, TheMisWeightsSumToOneAndFollowTheTargets)
{
  std::mt19937                          rng(5);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for(int i = 0; i < 1000; i++)
  {
    const float m = 1.0f + 8.0f * uniform(rng), here = uniform(rng) * 10.0f, there = uniform(rng) * 10.0f;
    EXPECT_NEAR(lightReuseMis(true, 1.0f, here, m, there) + lightReuseMis(false, 1.0f, here, m, there), 1.0f, 1e-6f);
  }
  // a sample as bright here as there: the M weights, 1 to 4
  EXPECT_FLOAT_EQ(lightReuseMis(false, 1.0f, 0.5f, 4.0f, 0.5f), 0.8f);
  // one the record's point cannot see (or no record can hold): this frame's alone; one neither sees, too
  EXPECT_FLOAT_EQ(lightReuseMis(true, 1.0f, 0.5f, 4.0f, 0.0f), 1.0f);
  EXPECT_FLOAT_EQ(lightReuseMis(false, 1.0f, 0.5f, 4.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(lightReuseMis(true, 1.0f, 0.0f, 4.0f, 0.0f), 1.0f);
  // a record's sample many times brighter here than where it was found (the next bump of a grain)
  // weighs at most its M share of what it weighed there, however bright it looks here
  const float W = 2.0f, there = 0.01f;
  for(float here : {0.01f, 1.0f, 100.0f, 1e6f})
    EXPECT_LE(lightReuseWeight(lightReuseMis(false, 1.0f, here, 4.0f, there), here * W), 4.0f * there * W * 1.0001f) << here;
  EXPECT_EQ(lightReuseWeight(1.0f, std::numeric_limits<float>::infinity()), 0.0f);
}

TEST(LightReuse, ASurfaceIsTheSameAtItsDepthAndNormal)
{
  LightReuseRecord r = lightReuseUnpack(uvec4(0), uvec4(0));
  r.depth            = 10.0f;
  r.normal           = vec3(0, 1, 0);
  EXPECT_TRUE(lightReuseSameSurface(r, 10.15f, vec3(0, 1, 0)));
  EXPECT_FALSE(lightReuseSameSurface(r, 10.5f, vec3(0, 1, 0)));                        // another surface behind
  EXPECT_FALSE(lightReuseSameSurface(r, 10.0f, normalize(vec3(0.6f, 0.8f, 0.0f))));  // a turned one
}

TEST(LightReuse, TheRecordsHistoryIsCapped)
{
  LightReuseRecord r = lightReuseUnpack(uvec4(0), uvec4(0));
  r.M                = 1000.0f;
  r.W                = 2.0f;
  EXPECT_FLOAT_EQ(lightReuseRecordM(r, 4.0f, 20.0f), 80.0f);
  r.M = 3.0f;
  EXPECT_FLOAT_EQ(lightReuseRecordM(r, 4.0f, 20.0f), 3.0f);
}

// A pixel's light over frames, as the shader does it. Each frame's candidates are a point light (no
// record can hold it) or a point of a row of lamps along [0, 1), drawn uniformly, some taking nothing;
// part of the row is shadowed. The candidates are resampled into one, which then competes with the
// record: each side weighs its MIS weight (M times its target here and at the record's point) times
// its target here times its W. The kept sample is shaded with its shadow and carried on, W 0 if it
// was shadowed or is the point light, M always.
//
// The shading point is the record's (`bumpy` false: a flat surface), or moves every frame across the
// bumps of a grain (true): a sharp highlight on the row, its place drawn anew each frame, as a still's
// jitter moves the bumped normal of the club's leather. Every frame's estimate must average to the
// point light plus the row's shadowed integral (over the bumps), and the history must cut the noise.
struct ReuseChain
{
  bool  bumpy = false;
  int   candidates = 2;
  float cap        = 1.0f;  // the renderer's ptLightReuseCap
  bool  reuse      = true;
  bool  constantM  = false;  // the M weights alone (M over the sum of M), as before the targets were in them
};

const vec3  kPointLight(0.08f, 0.06f, 0.02f);
const float kPickPoint = 0.2f;

// the row's light at x from a point whose bump puts its highlight at `peak` (flat: the lobe of integrand)
vec3 bumpyIntegrand(float x, float peak)
{
  const float d    = (x - peak) / 0.04f;
  const float bsdf = 12.0f * std::exp(-d * d);
  return vec3(1.0f, 0.6f + 0.4f * x, 0.2f + x * x) * (0.05f + bsdf);
}

float drawPeak(std::mt19937& rng)
{
  return std::uniform_real_distribution<float>(0.2f, 0.95f)(rng);
}

vec3 rowAt(const ReuseChain& c, float x, float peak)
{
  if(x >= 0.93f)  // past it the sample takes nothing, as a cut node below the horizon does
    return vec3(0.0f);
  return c.bumpy ? bumpyIntegrand(x, peak) : integrand(x);
}

vec3 chainReference(const ReuseChain& c)
{
  if(!c.bumpy)
    return kPointLight + referenceIntegral(0.93f);
  // over the bumps: the peak uniform in [0.2, 0.95)
  dvec3     sum(0.0);
  const int nx = 1 << 13, np = 1 << 9;
  for(int j = 0; j < np; j++)
  {
    const float peak = 0.2f + 0.75f * (float(j) + 0.5f) / float(np);
    for(int i = 0; i < nx; i++)
    {
      const float x = (float(i) + 0.5f) / float(nx);
      sum += dvec3(visibility(x) * rowAt(c, x, peak));
    }
  }
  return kPointLight + vec3(sum / double(nx * np));
}

struct ChainResult
{
  std::vector<dvec3>  mean;
  std::vector<double> variance;
};

ChainResult runChain(const ReuseChain& c, int frames, int trials, uint32_t seed)
{
  std::vector<dvec3>                    sum(frames, dvec3(0.0));
  std::vector<double>                   sum2(frames, 0.0);
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for(int t = 0; t < trials; t++)
  {
    LightReuseRecord record     = lightReuseUnpack(uvec4(0), uvec4(0));
    float            recordPeak = 0.0f;  // the record's point: its bump, as the record's shading normal
    for(int f = 0; f < frames; f++)
    {
      const float peak    = c.bumpy ? drawPeak(rng) : 0.0f;
      const float recordM = c.reuse ? lightReuseRecordM(record, float(c.candidates), c.cap) : 0.0f;
      // this frame's candidates, resampled
      LightReservoir r         = lightReservoirEmpty();
      bool           keptPoint = false;
      float          keptX     = 0.0f;
      for(int k = 0; k < c.candidates; k++)
      {
        const bool  point = uniform(rng) < kPickPoint;
        const float x     = uniform(rng);
        const float raw   = point ? luminance709(kPointLight / kPickPoint) : luminance709(rowAt(c, x, peak) / (1.0f - kPickPoint));
        r                 = lightReservoirAdd(r, raw, uniform(rng));
        if(r.keptLast)
        {
          keptPoint = point;
          keptX     = x;
        }
      }
      // this frame's side, its kept candidate's target at the record's point in the same measure
      const float keptThere = (recordM > 0.0f && r.kept > 0.0f && !keptPoint) ? luminance709(rowAt(c, keptX, recordPeak) / (1.0f - kPickPoint)) : 0.0f;
      const float meanWeight    = r.count > 0 ? r.weightSum / float(r.count) : 0.0f;
      const float currentMis    = !c.constantM ? lightReuseMis(true, float(c.candidates), r.kept, recordM, keptThere)
                                  : (keptPoint ? 1.0f : float(c.candidates) / (float(c.candidates) + recordM));
      const float currentWeight = lightReuseWeight(currentMis, meanWeight);
      // the record's side, in the row's measure
      float recordTarget = 0.0f, recordWeight = 0.0f;
      if(recordM > 0.0f && record.W > 0.0f)
      {
        recordTarget            = luminance709(rowAt(c, record.uv.x, peak));
        const float recordThere = luminance709(rowAt(c, record.uv.x, recordPeak));
        const float recordMis   = !c.constantM ? lightReuseMis(false, float(c.candidates), recordTarget, recordM, recordThere)
                                  : recordM / (float(c.candidates) + recordM);
        recordWeight = lightReuseWeight(recordMis, recordTarget * record.W);
      }
      LightReservoir pair = lightReservoirAdd(lightReservoirEmpty(), currentWeight, uniform(rng));
      pair                = lightReservoirAdd(pair, recordWeight, uniform(rng));
      const bool  tookRecord = pair.keptLast;
      const bool  point      = !tookRecord && keptPoint;
      const float y          = tookRecord ? record.uv.x : keptX;
      const vec3  fy         = point ? kPointLight : rowAt(c, y, peak);  // the sample's own measure: the light, or the row's
      const float W          = (pair.weightSum > 0.0f && luminance709(fy) > 0.0f) ? pair.weightSum / luminance709(fy) : 0.0f;
      const vec3  e          = (point ? 1.0f : visibility(y)) * fy * W;
      sum[f] += dvec3(e);
      sum2[f] += double(luminance709(e)) * luminance709(e);
      record.uv  = vec2(y, 0.0f);
      record.W   = (!point && visibility(y) > 0.0f) ? W : 0.0f;
      record.M   = float(c.candidates) + recordM;
      recordPeak = peak;
    }
  }
  ChainResult out;
  for(int f = 0; f < frames; f++)
  {
    out.mean.push_back(sum[f] / double(trials));
    const double m = luminance709(vec3(out.mean.back()));
    out.variance.push_back(sum2[f] / trials - m * m);
  }
  return out;
}

void expectUnbiased(const ReuseChain& c, const ChainResult& r, double tolerance)
{
  const vec3 reference = chainReference(c);
  for(size_t f = 0; f < r.mean.size(); f++)
    for(int ch = 0; ch < 3; ch++)
      EXPECT_NEAR(r.mean[f][ch], reference[ch], tolerance * reference[ch]) << "frame " << f << ", channel " << ch;
}

TEST(LightReuse, AChainOverFramesStaysUnbiased)
{
  ReuseChain        c;
  const ChainResult r = runChain(c, 12, 200000, 321);
  expectUnbiased(c, r, 0.015);
  EXPECT_LT(r.variance.back(), 0.6 * r.variance.front()) << "the history should cut the noise";
}

// The club's leather: the highlight moves from frame to frame. Still unbiased, and the history still
// cuts the noise rather than adding to it (with constant M weights a sample found off the highlight
// and carried onto it weighed its whole history there).
TEST(LightReuse, AChainOverABumpyGlossyPixelStaysUnbiased)
{
  ReuseChain c;
  c.bumpy             = true;
  const ChainResult r = runChain(c, 12, 400000, 99);
  expectUnbiased(c, r, 0.015);
  ReuseChain plain = c;
  plain.reuse      = false;
  const ChainResult p = runChain(plain, 2, 400000, 99);
  EXPECT_LT(r.variance.back(), p.variance.back()) << "reuse should not add noise on a bumpy glossy pixel";
  ReuseChain constant = c;
  constant.constantM  = true;
  const ChainResult k = runChain(constant, 12, 400000, 99);
  EXPECT_LT(r.variance.back(), 0.5 * k.variance.back()) << "the targets in the MIS weights are what keep it down";
  std::printf("bumpy pixel, last frame's variance: no reuse %.4g, M weights alone %.4g, with the targets %.4g\n", p.variance.back(),
              k.variance.back(), r.variance.back());
}

}  // namespace
