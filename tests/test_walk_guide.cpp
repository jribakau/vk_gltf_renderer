// Walk guiding (shaders/walk_guide.h.slang) compiled as C++: the diffusion length fit solves the
// half-space's transport equation, the guided directions are sampled by their pdf, and a walk in a
// half-space weighed with walkStepPdf, as handleVolumeScatter weighs it, reflects the same light
// guided as unguided, in fewer events.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <random>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/walk_guide.h.slang"

namespace {

using double3 = glm::dvec3;

constexpr float kPi = 3.14159265358979f;

// The exact diffusion length v of an isotropic half-space of albedo a solves a v / 2 log((v + 1) / (v - 1)) = 1
double transportResidual(double albedo, double v)
{
  return albedo * v / 2.0 * std::log((v + 1.0) / (v - 1.0)) - 1.0;
}

float hgPdf(float cosTheta, float g)
{
  const float d = 1.0f + g * g - 2.0f * g * cosTheta;
  return (1.0f - g * g) / (4.0f * kPi * d * std::sqrt(d));
}

// A direction at `cosTheta` to unit `axis`, turned by `phi` about it
float3 aroundAxis(float3 axis, float cosTheta, float phi)
{
  const float  s = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));
  const float3 a = std::abs(axis.x) > 0.5f ? float3(0, 1, 0) : float3(1, 0, 0);
  const float3 t = normalize(cross(a, axis));
  const float3 b = cross(axis, t);
  return normalize(t * (std::cos(phi) * s) + b * (std::sin(phi) * s) + axis * cosTheta);
}

float3 sampleHg(float3 wi, float g, float u1, float u2)
{
  float cosTheta = 1.0f - 2.0f * u1;
  if(std::abs(g) > 1e-3f)
  {
    const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
    cosTheta      = (1.0f + g * g - s * s) / (2.0f * g);
  }
  return aroundAxis(wi, std::clamp(cosTheta, -1.0f, 1.0f), 2.0f * kPi * u2);
}

struct WalkStats
{
  double3 sum{0.0}, sumSq{0.0};
  double  events = 0.0;
  int     walks  = 0;

  double3 mean() const { return sum / double(walks); }
  double3 stderr2() const
  {
    const double3 m = mean();
    return (sumSq / double(walks) - m * m) / double(walks);
  }
};

// Walks entering the half-space z < 0 at the origin, straight down, until they leave through z = 0:
// the light each brings back per channel, as the path tracer's walk estimates it (sampleFreeFlight,
// handleVolumeScatter): a channel picked by the throughput, the balance heuristic over the channels,
// and with `guided` the guided and classic steps weighed together.
WalkStats walkHalfSpace(float3 sigmaS, float3 sigmaT, float g, bool guided, int walks, unsigned seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  const float3                          albedo = sigmaS / sigmaT;
  const float3                          normal(0, 0, 1);
  const WalkGuide guide = guided ? makeWalkGuide(std::max(albedo.x, std::max(albedo.y, albedo.z)), g) : WalkGuide{0, 0, 0};

  WalkStats stats;
  for(int w = 0; w < walks; ++w)
  {
    float3 pos(0.0f), dir(0, 0, -1), throughput(1.0f);
    float  walkGuide = 0.0f;
    for(int event = 0; event < 100000; ++event)
    {
      stats.events += 1.0;
      float3 channelProb = throughput / (throughput.x + throughput.y + throughput.z);
      float  u           = std::min(uniform(rng), 1.0f - 1e-7f);
      float  sigma       = sigmaT.z;
      if(u < channelProb.x)
        sigma = sigmaT.x;
      else if(u < channelProb.x + channelProb.y)
        sigma = sigmaT.y;
      u = uniform(rng);

      float fraction = 0.0f, stretch = 1.0f, factor = 0.0f;
      if(walkGuide != 0.0f)
      {
        fraction = guide.fraction;
        stretch  = walkGuideStretch(guide, dot(dir, normal));
        factor   = std::abs(walkGuide);
        if(walkGuide < 0.0f)
          sigma *= stretch;
      }
      const float dist    = -std::log(std::max(1.0f - u, 1e-10f)) / sigma;
      const float surface = dir.z > 0.0f ? -pos.z / dir.z : INFINITY;
      if(dist >= surface)
      {
        const float3 transmittance = exp(-sigmaT * surface);
        throughput *= transmittance / dot(channelProb, walkStepPdf(sigmaT, surface, true, fraction, stretch, factor));
        const double3 out(throughput);
        stats.sum += out;
        stats.sumSq += out * out;
        break;
      }
      const float3 transmittance = exp(-sigmaT * dist);
      throughput *= sigmaS * transmittance / dot(channelProb, walkStepPdf(sigmaT, dist, false, fraction, stretch, factor));
      pos += dir * dist;
      if(std::max(throughput.x, std::max(throughput.y, throughput.z)) < 1e-7f)
        break;  // what it could still bring back is far below the test's tolerance

      const float3 wi = dir;
      walkGuide       = 0.0f;
      const float c   = uniform(rng);
      if(c < guide.fraction)
        dir = aroundAxis(normal, walkGuideSampleCos(guide, c / guide.fraction), 2.0f * kPi * uniform(rng));
      else
        dir = sampleHg(wi, g, uniform(rng), uniform(rng));
      if(guide.fraction > 0.0f)
      {
        const float f = walkGuidePdf(guide, dot(dir, normal)) / hgPdf(dot(wi, dir), g);
        walkGuide     = c < guide.fraction ? -f : f;
      }
    }
    ++stats.walks;
  }
  return stats;
}

}  // namespace

TEST(WalkGuide, DiffusionLengthSolvesTheHalfSpace)
{
  for(float albedo : {0.4f, 0.5f, 0.8f, 0.9f, 0.95f, 0.99f, 0.999f})
  {
    const WalkGuide w = makeWalkGuide(albedo, 0.0f);
    ASSERT_GT(w.fraction, 0.0f) << albedo;
    // The fit is good to about a percent of the residual over the range Cycles uses it in
    EXPECT_NEAR(transportResidual(albedo, w.diffusionLength), 0.0, 0.02) << albedo << " v " << w.diffusionLength;
    EXPECT_NEAR(w.phaseLog, std::log((w.diffusionLength + 1.0f) / (w.diffusionLength - 1.0f)), 1e-5f);
  }
}

TEST(WalkGuide, FractionFollowsTheAnisotropy)
{
  EXPECT_FLOAT_EQ(makeWalkGuide(0.9f, 0.0f).fraction, 0.5f);
  EXPECT_NEAR(makeWalkGuide(0.9f, 0.5f).fraction, 1.0f - std::pow(0.5f, 0.125f), 1e-6f);
  EXPECT_NEAR(makeWalkGuide(0.9f, -0.5f).fraction, makeWalkGuide(0.9f, 0.5f).fraction, 1e-6f);
  EXPECT_FLOAT_EQ(makeWalkGuide(0.9f, 1.0f).fraction, 0.0f);
}

TEST(WalkGuide, NoGuidingWhereItCannotHelp)
{
  EXPECT_EQ(makeWalkGuide(0.0f, 0.0f).fraction, 0.0f);   // nothing scatters
  EXPECT_EQ(makeWalkGuide(1.0f, 0.0f).fraction, 0.0f);   // nothing absorbs: v is unbounded
  EXPECT_EQ(makeWalkGuide(1e-4f, 0.0f).fraction, 0.0f);  // v is 1: the pdf is not normalisable
  EXPECT_EQ(makeWalkGuide(0.2f, 0.0f).fraction, 0.0f);   // v is within 1e-4 of 1: a walk of an event or two
}

TEST(WalkGuide, PdfIsNormalisedAndSampled)
{
  for(float albedo : {0.5f, 0.9f, 0.99f})
  {
    const WalkGuide w = makeWalkGuide(albedo, 0.0f);
    // Over the sphere: 2 pi times the integral over the cosine
    double    integral = 0.0;
    const int n        = 200000;
    for(int i = 0; i < n; ++i)
    {
      const float c = -1.0f + (i + 0.5f) * 2.0f / n;
      integral += 2.0 * kPi * walkGuidePdf(w, c) * (2.0 / n);
    }
    EXPECT_NEAR(integral, 1.0, 1e-4) << albedo;
    // The sample inverts the cosine's CDF, log((v + 1) / (v - c)) / phaseLog
    for(float u : {0.0f, 0.1f, 0.5f, 0.9f, 0.999f})
    {
      const float c   = walkGuideSampleCos(w, u);
      const float cdf = std::log((w.diffusionLength + 1.0f) / (w.diffusionLength - c)) / w.phaseLog;
      EXPECT_NEAR(cdf, u, 1e-4f) << albedo << " u " << u;
    }
    // Towards the surface: the median direction leans out of the medium
    EXPECT_GT(walkGuideSampleCos(w, 0.5f), 0.0f);
  }
}

TEST(WalkGuide, StepPdfIsClassicWithoutGuiding)
{
  const float3 sigmaT(2.0f, 5.0f, 9.0f);
  const float3 classic = sigmaT * exp(-sigmaT * 0.3f);
  const float3 pdf     = walkStepPdf(sigmaT, 0.3f, false, 0.0f, 0.7f, 3.0f);
  EXPECT_FLOAT_EQ(pdf.x, classic.x);
  EXPECT_FLOAT_EQ(pdf.z, classic.z);
  EXPECT_FLOAT_EQ(walkStepPdf(sigmaT, 0.3f, true, 0.0f, 0.7f, 3.0f).y, std::exp(-5.0f * 0.3f));
}

// Over the outcomes of one step to a surface at distance d, a scatter anywhere before it or reaching
// it, each technique's pdf integrates to 1: the mixture to 1 - fraction + fraction * factor
TEST(WalkGuide, StepPdfCoversEveryOutcome)
{
  const float3 sigmaT(2.0f, 5.0f, 9.0f);
  const float  d = 0.4f, fraction = 0.5f, stretch = 0.6f, factor = 1.7f;
  double3      integral(0.0);
  const int    n = 100000;
  for(int i = 0; i < n; ++i)
    integral += double3(walkStepPdf(sigmaT, (i + 0.5f) * d / n, false, fraction, stretch, factor)) * double(d / n);
  integral += double3(walkStepPdf(sigmaT, d, true, fraction, stretch, factor));
  const double expected = 1.0 - fraction + fraction * factor;
  EXPECT_NEAR(integral.x, expected, 1e-4);
  EXPECT_NEAR(integral.y, expected, 1e-4);
  EXPECT_NEAR(integral.z, expected, 1e-4);
}

// The estimate is unbiased: guided and unguided walks bring back the same light from a chromatic
// medium like skin's (red scatters far and barely absorbs, blue the reverse), within their noise, for
// an isotropic medium and a forward-scattering one. Guided walks take fewer events.
TEST(WalkGuide, GuidedWalkReflectsTheSameLight)
{
  const float3 sigmaS(0.97f, 0.85f, 0.6f);
  const float3 sigmaT(1.0f, 1.0f, 1.0f);
  for(float g : {0.0f, 0.5f})
  {
    const float3    s = sigmaS * float3(1, 2, 3), t = sigmaT * float3(1, 2, 3);
    const WalkStats classic = walkHalfSpace(s, t, g, false, 200000, 1u);
    const WalkStats guided  = walkHalfSpace(s, t, g, true, 200000, 2u);
    const double3   a = classic.mean(), b = guided.mean();
    const double3   tol = 4.0 * sqrt(classic.stderr2() + guided.stderr2());
    EXPECT_NEAR(b.x, a.x, tol.x) << "g " << g;
    EXPECT_NEAR(b.y, a.y, tol.y) << "g " << g;
    EXPECT_NEAR(b.z, a.z, tol.z) << "g " << g;
    if(g == 0.0f)
      EXPECT_LT(guided.events / guided.walks, classic.events / classic.walks);
    std::printf("g %.1f  reflectance classic (%.4f %.4f %.4f) guided (%.4f %.4f %.4f)  events/walk %.1f -> %.1f  "
                "stderr red %.5f -> %.5f\n",
                g, a.x, a.y, a.z, b.x, b.y, b.z, classic.events / classic.walks, guided.events / guided.walks,
                std::sqrt(classic.stderr2().x), std::sqrt(guided.stderr2().x));
  }
}
