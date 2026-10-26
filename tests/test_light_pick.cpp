// The path tracer's light pick (shaders/light_pick.h.slang) compiled as C++: the estimates are the
// irradiance singleLightContribution gives, a culled light is one whose every sample next-event
// estimation would discard, the probabilities sum to one, and the power heuristic's two weights do.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/light_pick.h.slang"

namespace {

constexpr float kPi = 3.14159265358979f;

float pointEstimate(float3 lightPos, float radius, float power, float3 pos, float3 normal, bool cull = true)
{
  return punctualLightPickEstimate(false, false, lightPos, float3(0, -1, 0), radius, power, 0.0f, 0.0f, 0.0f, pos, normal, cull);
}

// A direction uniform over the cone a sphere of `radius` at `center` subtends from `pos`, as
// singleLightContribution draws it.
float3 sampleSphereCone(float3 center, float radius, float3 pos, float u1, float u2)
{
  const float3 toLight = center - pos;
  const float  d       = length(toLight);
  const float  r       = std::min(radius / d, 1.0f);
  const float  r2p1    = r * r + 1.0f;
  const float  omc     = r * r / (r2p1 + std::sqrt(r2p1));
  const float  z       = 1.0f - u2 * omc;
  const float  s       = std::sqrt(std::max(1.0f - z * z, 0.0f));
  const float3 w       = toLight / d;
  const float3 a       = std::abs(w.x) > 0.5f ? float3(0, 1, 0) : float3(1, 0, 0);
  const float3 t       = normalize(cross(a, w));
  const float3 b       = cross(w, t);
  return normalize(t * (std::cos(2.0f * kPi * u1) * s) + b * (std::sin(2.0f * kPi * u1) * s) + w * z);
}

}  // namespace

// A small light gives intensity over distance squared, a sphere far away the same, and the range
// and the spot cone shape it as KHR_lights_punctual says.
TEST(LightPick, EstimatesAreTheIrradianceAtNormalIncidence)
{
  const float3 pos(0, 0, 0), up(0, 1, 0);
  EXPECT_NEAR(pointEstimate(float3(0, 2, 0), 0.0f, 100.0f, pos, up), 25.0f, 1e-4f);
  EXPECT_NEAR(pointEstimate(float3(0, 200, 0), 0.01f, 1e6f, pos, up), 25.0f, 0.01f);
  // Inside its sphere a light gives a hemisphere of its radiance (I / r^2 * pi), not infinity
  EXPECT_NEAR(pointEstimate(float3(0, 0.01f, 0), 0.5f, 1.0f, pos, up), 4.0f * (2.0f - std::sqrt(2.0f)), 1e-4f);

  // range 4 at distance 2: 1 - (1/2)^4
  const float ranged = punctualLightPickEstimate(false, false, float3(0, 2, 0), float3(0, -1, 0), 0.0f, 100.0f, 0.25f, 0, 0, pos, up, true);
  EXPECT_NEAR(ranged, 25.0f * (1.0f - 1.0f / 16.0f), 1e-4f);
  EXPECT_EQ(punctualLightPickEstimate(false, false, float3(0, 5, 0), float3(0, -1, 0), 0.0f, 100.0f, 0.25f, 0, 0, pos, up, true), 0.0f);

  // A spot aiming 30 degrees off the point, cone 20..40 degrees: the ramp's square
  const float  inner = 20.0f * kPi / 180.0f, outer = 40.0f * kPi / 180.0f, off = 30.0f * kPi / 180.0f;
  const float3 aim(std::sin(off), -std::cos(off), 0.0f);
  const float  t = (std::cos(off) - std::cos(outer)) / (std::cos(inner) - std::cos(outer));
  EXPECT_NEAR(punctualLightPickEstimate(false, true, float3(0, 2, 0), aim, 0.0f, 100.0f, 0.0f, inner, outer, pos, up, true), 25.0f * t * t, 1e-3f);
  const float3 away(std::sin(0.8f), -std::cos(0.8f), 0.0f);
  EXPECT_EQ(punctualLightPickEstimate(false, true, float3(0, 2, 0), away, 0.0f, 100.0f, 0.0f, inner, outer, pos, up, true), 0.0f);

  // A directional light is its illuminance
  EXPECT_EQ(punctualLightPickEstimate(true, false, float3(0), normalize(float3(1, -1, 0)), 0, 5000.0f, 0, 0, 0, pos, up, true), 5000.0f);
}

// A directional light below the horizon is culled, as singleLightContribution drops it; lit from
// both sides (no cull) it is not.
TEST(LightPick, ASunBelowTheHorizonGetsNothing)
{
  const float3 pos(0), up(0, 1, 0), sunTravel = normalize(float3(0.3f, 0.2f, 0.0f));  // travels upward: shines from below
  EXPECT_EQ(punctualLightPickEstimate(true, false, float3(0), sunTravel, 0, 1e5f, 0, 0, 0, pos, up, true), 0.0f);
  EXPECT_EQ(punctualLightPickEstimate(true, false, float3(0), sunTravel, 0, 1e5f, 0, 0, 0, pos, up, false), 1e5f);
}

// Culling is conservative: a sphere light the pick gives nothing has no direction of its cone above
// the plane, where next-event estimation would keep the sample; one that reaches above it is kept.
TEST(LightPick, ACulledSphereHasNoSampleAboveTheHorizon)
{
  std::mt19937                          rng(11);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const float3                          up(0, 1, 0);
  int                                   culled = 0, kept = 0;
  for(int i = 0; i < 2000; i++)
  {
    const float3 pos(0);
    const float  radius = 0.05f + u(rng) * 0.5f;
    const float3 center(2.0f * u(rng) - 1.0f, (u(rng) - 0.6f) * 1.5f, 2.0f * u(rng) - 1.0f);
    if(length(center) <= radius * 1.01f)
      continue;
    const bool isCulled = pointEstimate(center, radius, 1.0f, pos, up) == 0.0f;
    bool       anyAbove = false;
    for(int k = 0; k < 400; k++)
      anyAbove |= dot(sampleSphereCone(center, radius, pos, u(rng), u(rng)), up) > 1e-4f;
    if(isCulled)
    {
      culled++;
      EXPECT_FALSE(anyAbove) << "centre y " << center.y << " radius " << radius;
    }
    else
      kept++;
    // The sphere's top above the plane by a clear margin: some of its cone is too, and it is kept
    if(center.y > -radius + 0.05f)
      EXPECT_FALSE(isCulled);
  }
  EXPECT_GT(culled, 100);
  EXPECT_GT(kept, 100);
}

// Over the candidates the probabilities sum to one, each at least the defensive share; a light that
// cannot light the point has none.
TEST(LightPick, ProbabilitiesSumToOne)
{
  const std::vector<float> estimates = {1e5f, 3.0f, 0.0f, 2500.0f, 0.02f};
  float                    total = 0.0f;
  int                      candidates = 0;
  for(float e : estimates)
  {
    total += e;
    candidates += e > 0.0f;
  }
  float sum = 0.0f;
  for(float e : estimates)
  {
    const float p = lightPickProbability(e, total, candidates);
    if(e > 0.0f)
      EXPECT_GE(p, LIGHT_PICK_DEFENSIVE / candidates - 1e-7f);
    else
      EXPECT_EQ(p, 0.0f);
    sum += p;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
  EXPECT_EQ(lightPickProbability(1.0f, 0.0f, 0), 0.0f);
}

// A furnace of the pick: lights of known contribution at a vertex, one picked per sample and
// divided by its probability, average to their sum (none is starved, none counted twice).
TEST(LightPick, ThePickedSampleAveragesToEveryLight)
{
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  // What each light gives the point: the estimates are off (occlusion, the BSDF), never zero where this is not
  const std::vector<float> estimate     = {8e4f, 9000.0f, 40.0f, 0.0f, 300.0f};
  const std::vector<float> contribution = {1e4f, 7000.0f, 90.0f, 0.0f, 10.0f};
  float                    total = 0.0f, expected = 0.0f;
  int                      candidates = 0;
  for(size_t i = 0; i < estimate.size(); i++)
  {
    total += estimate[i];
    candidates += estimate[i] > 0.0f;
    expected += contribution[i];
  }
  const int N   = 400000;
  double    sum = 0.0;
  for(int s = 0; s < N; s++)
  {
    // The shader's walk: the first light whose cumulative share passes u
    const float x          = std::min(u(rng), 0.99999994f);
    float       cumulative = 0.0f;
    for(size_t i = 0; i < estimate.size(); i++)
    {
      const float p = lightPickProbability(estimate[i], total, candidates);
      if(p <= 0.0f)
        continue;
      cumulative += p;
      if(x < cumulative)
      {
        sum += contribution[i] / p;
        break;
      }
    }
  }
  EXPECT_NEAR(sum / N / expected, 1.0, 0.01);
}

// The two techniques' weights for one sample sum to one, the larger pdf takes more than its balance
// share, and neither a huge pdf nor a missing technique gives NaN.
TEST(LightPick, PowerHeuristicWeightsSumToOne)
{
  const float pdfs[] = {1e-30f, 1e-6f, 0.1f, 1.0f, 3.0f, 1e6f, 1e30f, 3e38f};
  for(float a : pdfs)
    for(float b : pdfs)
    {
      const float wa = misPowerWeight(a, b), wb = misPowerWeight(b, a);
      ASSERT_TRUE(std::isfinite(wa) && std::isfinite(wb)) << a << " " << b;
      EXPECT_NEAR(wa + wb, 1.0f, 1e-6f) << a << " " << b;
      if(a > b)
        EXPECT_GE(wa, a / (a + b) - 1e-6f);
    }
  EXPECT_NEAR(misPowerWeight(3.0f, 1.0f), 0.9f, 1e-6f);
  EXPECT_EQ(misPowerWeight(2.0f, 0.0f), 1.0f);
}
