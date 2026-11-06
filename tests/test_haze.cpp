// The haze on the camera's segment (shaders/haze.h.slang) compiled as C++: the stretch of a ray in
// the haze's sphere, the scatter point's distance and its density, and the estimate they make of a
// point light's single scattering, against the integral done by quadrature (itself held to the
// closed form of thin, isotropic haze). The shader composes the same functions with sampleLights,
// which for one point light gives what these tests give it: the light's position and I / r^2.

// GLM as test_bsdf_lobes.cpp, another file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <vector>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/haze.h.slang"

namespace {

constexpr double kPi = std::numbers::pi;

double henyeyGreenstein(double cosTheta, double g)
{
  return (1 - g * g) / (4 * kPi * std::pow(1 + g * g - 2 * g * cosTheta, 1.5));
}

// The scene of these tests: a ray along +x from the origin through unbounded haze, a point light of
// intensity I at `along` on it and `h` beside it
struct HaloScene
{
  float sigma = 3.912023F / 50.0F;  // a visibility of 50 m
  float g     = 0.7F;
  float t0 = 0.0F, t1 = 10.0F;
  float along = 5.0F, h = 0.3F;
  float intensity = 2.0F;

  vec3 o() const { return vec3(0.0F); }
  vec3 d() const { return vec3(1.0F, 0.0F, 0.0F); }
  vec3 light() const { return vec3(along, h, 0.0F); }
};

// The single scattering the eye gets from the light: the integral over the segment of
// sigma e^(-sigma (t - t0)) phase I e^(-sigma r) / r^2, by Simpson's rule after t = along + h tan(theta),
// which takes the 1/r^2 out (dt / r^2 = dtheta / h) and leaves it smooth
double reference(const HaloScene& s, int intervals = 20000)
{
  const double a = std::atan2(s.t0 - s.along, s.h), b = std::atan2(s.t1 - s.along, s.h);
  const auto   f = [&](double theta) {
    const double t = s.along + s.h * std::tan(theta), r = s.h / std::cos(theta);
    // the light is at -sin(theta) along the ray as seen from the point
    return s.sigma * std::exp(-s.sigma * (t - s.t0)) * henyeyGreenstein(-std::sin(theta), s.g) * s.intensity
           * std::exp(-s.sigma * r) / s.h;
  };
  const double step = (b - a) / intervals;
  double       sum  = f(a) + f(b);
  for(int i = 1; i < intervals; i++)
    sum += f(a + i * step) * (i % 2 ? 4 : 2);
  return sum * step / 3;
}

// One sample of the shader's estimate (hazeCameraSegment) with the light as its only light
double estimate(const HaloScene& s, const HazeGuide& guide, std::mt19937& rng)
{
  std::uniform_real_distribution<float> u01(0.0F, 1.0F);
  const float choice = u01(rng);
  const float t      = hazeSampleDistance(u01(rng), choice, s.sigma, s.t0, s.t1, guide);
  const vec3  toward = s.light() - (s.o() + t * s.d());
  const float r      = length(toward);
  return double(hazeScatterWeight(t, s.sigma, s.t0, s.t1, guide)) * henyeyGreenstein(dot(s.d(), toward / r), s.g)
         * std::exp(-s.sigma * r) * s.intensity / (r * r);
}

struct Stats
{
  double mean, relativeNoise;  // the mean, and one sample's standard deviation over it
};

Stats run(const HaloScene& s, const HazeGuide& guide, int n, unsigned seed)
{
  std::mt19937 rng(seed);
  double       sum = 0, sum2 = 0;
  for(int i = 0; i < n; i++)
  {
    const double x = estimate(s, guide, rng);
    sum += x;
    sum2 += x * x;
  }
  const double mean = sum / n;
  return {mean, std::sqrt(std::max(sum2 / n - mean * mean, 0.0)) / mean};
}

TEST(Haze, AVisibilityIsTheDistanceThatLeavesTwoPercent)
{
  EXPECT_NEAR(std::exp(-hazeExtinctionFromVisibility(200.0F) * 200.0F), 0.02, 1e-6);
  EXPECT_EQ(hazeExtinctionFromVisibility(0.0F), 0.0F);
  for(const double x : {1e-7, 1e-5, 2e-3, 0.05, 0.099, 0.101, 0.5, 2.0})
  {
    EXPECT_NEAR(hazeOneMinusExp(float(x)) / -std::expm1(-double(float(x))), 1.0, 1e-6) << x;
    if(x < 1.0)
      EXPECT_NEAR(hazeMinusLogOneMinus(float(x)) / -std::log1p(-double(float(x))), 1.0, 1e-5) << x;
  }
}

TEST(Haze, TheSegmentIsTheRaysStretchInsideTheSphere)
{
  const vec3 c(0.0F), x(1.0F, 0.0F, 0.0F);
  // from inside: to the hit, or to the edge where it hits nothing
  vec2 seg = hazeSegment(vec3(0.0F), x, 3.0F, c, 5.0F);
  EXPECT_FLOAT_EQ(seg.x, 0.0F);
  EXPECT_FLOAT_EQ(seg.y, 3.0F);
  seg = hazeSegment(vec3(0.0F), x, 1e32F, c, 5.0F);
  EXPECT_FLOAT_EQ(seg.y, 5.0F);
  // from outside: from where the ray enters
  seg = hazeSegment(vec3(-8.0F, 0.0F, 0.0F), x, 1e32F, c, 5.0F);
  EXPECT_FLOAT_EQ(seg.x, 3.0F);
  EXPECT_FLOAT_EQ(seg.y, 13.0F);
  // a hit before the sphere, a sphere behind the eye, a ray that passes it by: none
  seg = hazeSegment(vec3(-8.0F, 0.0F, 0.0F), x, 2.0F, c, 5.0F);
  EXPECT_LE(seg.y, seg.x);
  seg = hazeSegment(vec3(8.0F, 0.0F, 0.0F), x, 1e32F, c, 5.0F);
  EXPECT_LE(seg.y, seg.x);
  seg = hazeSegment(vec3(-8.0F, 6.0F, 0.0F), x, 1e32F, c, 5.0F);
  EXPECT_LE(seg.y, seg.x);
  // a shadow ray leaves it at the edge
  EXPECT_NEAR(hazeExitDistance(vec3(1.0F, 0.0F, 0.0F), x, c, 5.0F), 4.0F, 1e-5);
  EXPECT_NEAR(hazeExitDistance(vec3(0.0F, 3.0F, 0.0F), x, c, 5.0F), 4.0F, 1e-5);
}

// The density integrates to 1, and the distances drawn follow it, with the guide at the light, at
// the segment's end, beyond it and nowhere
TEST(Haze, TheDistancesDrawnFollowTheirDensity)
{
  const HaloScene s;
  const struct
  {
    const char* what;
    HazeGuide   guide;
  } cases[] = {
      {"at the light", hazeGuide(s.o(), s.d(), s.t0, s.t1, s.light(), true)},
      {"near the light", hazeGuide(s.o(), s.d(), s.t0, s.t1, vec3(5.0F, 0.02F, 0.0F), true)},
      {"at the end", hazeGuide(s.o(), s.d(), s.t0, s.t1, vec3(9.9F, 0.5F, 0.0F), true)},
      {"beyond it", hazeGuide(s.o(), s.d(), s.t0, s.t1, vec3(14.0F, 1.0F, 2.0F), true)},
      {"none", hazeNoGuide()},
  };
  for(const auto& c : cases)
  {
    // integral by the midpoint rule, fine enough for a 2 cm peak
    const int n        = 400000;
    double    integral = 0;
    for(int i = 0; i < n; i++)
    {
      const float t = s.t0 + (i + 0.5F) * (s.t1 - s.t0) / n;
      integral += hazeDistancePdf(t, s.sigma, s.t0, s.t1, c.guide) * double(s.t1 - s.t0) / n;
    }
    EXPECT_NEAR(integral, 1.0, 2e-3) << c.what;

    // 40 bins of 400000 draws against the density's mass in each
    const int                             bins = 40, draws = 400000;
    std::vector<int>                      counts(bins, 0);
    std::mt19937                          rng(7);
    std::uniform_real_distribution<float> u01(0.0F, 1.0F);
    for(int i = 0; i < draws; i++)
    {
      const float choice = u01(rng);
      const float t      = hazeSampleDistance(u01(rng), choice, s.sigma, s.t0, s.t1, c.guide);
      ASSERT_GE(t, s.t0);
      ASSERT_LE(t, s.t1);
      counts[std::min(int((t - s.t0) / (s.t1 - s.t0) * bins), bins - 1)]++;
    }
    for(int b = 0; b < bins; b++)
    {
      double mass = 0;
      for(int k = 0; k < 2000; k++)
      {
        const float t = s.t0 + (b + (k + 0.5F) / 2000) * (s.t1 - s.t0) / bins;
        mass += hazeDistancePdf(t, s.sigma, s.t0, s.t1, c.guide) * double(s.t1 - s.t0) / bins / 2000;
      }
      const double expected = mass * draws, spread = std::sqrt(expected) + 1;
      EXPECT_NEAR(counts[b], expected, 5 * spread + 0.002 * expected) << c.what << ", bin " << b;
    }
  }
}

// Drawn by the transmittance alone, a point carries what the whole segment scatters
TEST(Haze, WithoutAGuideAPointCarriesWhatTheSegmentScatters)
{
  const HaloScene     s;
  const HazeGuide none = hazeNoGuide();
  for(float t : {0.0F, 2.5F, 9.0F})
    EXPECT_NEAR(hazeScatterWeight(t, s.sigma, s.t0, s.t1, none), 1.0F - std::exp(-s.sigma * (s.t1 - s.t0)), 1e-6);
}

// The estimate of a point light's halo is unbiased: with the guide at the light (what sampleLights
// gives for one), at another light (a pick of another one), and with none (a light at infinity
// picked for the guide)
TEST(Haze, APointLightsHaloMatchesItsSingleScatter)
{
  // the quadrature, against the closed form of thin isotropic haze, sigma I (theta_b - theta_a) / (4 pi h)
  {
    HaloScene thin;
    thin.sigma = 1e-6F;
    thin.g     = 0.0F;
    const double closed =
        thin.sigma * thin.intensity
        * (std::atan2(thin.t1 - thin.along, thin.h) - std::atan2(thin.t0 - thin.along, thin.h)) / (4 * kPi * thin.h);
    EXPECT_NEAR(reference(thin) / closed, 1.0, 1e-4);
  }
  for(const float g : {0.0F, 0.7F})
    for(const float h : {0.3F, 1.5F})
    {
      HaloScene s;
      s.g              = g;
      s.h              = h;
      const double ref = reference(s);
      const Stats at    = run(s, hazeGuide(s.o(), s.d(), s.t0, s.t1, s.light(), true), 400000, 11);
      const Stats other = run(s, hazeGuide(s.o(), s.d(), s.t0, s.t1, vec3(1.0F, -2.0F, 0.5F), true), 400000, 12);
      const Stats none  = run(s, hazeNoGuide(), 400000, 13);
      EXPECT_NEAR(at.mean / ref, 1.0, 0.01) << "g " << g << ", h " << h;
      EXPECT_NEAR(other.mean / ref, 1.0, 0.015) << "g " << g << ", h " << h;
      EXPECT_NEAR(none.mean / ref, 1.0, 0.015) << "g " << g << ", h " << h;
    }
}

// The equiangular half takes the 1/r^2 out of a halo: a ray passing a light at 5 cm on a 10 m segment
TEST(Haze, AGuideAtTheLightTakesItsHalosNoiseAway)
{
  HaloScene s;
  s.h               = 0.05F;
  s.g               = 0.0F;
  const double ref  = reference(s);
  const Stats  at   = run(s, hazeGuide(s.o(), s.d(), s.t0, s.t1, s.light(), true), 400000, 21);
  const Stats  none = run(s, hazeNoGuide(), 400000, 22);
  std::printf("halo at 5 cm, one sample's noise over its mean: guided %.3f, by the transmittance alone %.2f\n",
              at.relativeNoise, none.relativeNoise);
  EXPECT_NEAR(at.mean / ref, 1.0, 0.01);
  EXPECT_NEAR(none.mean / ref, 1.0, 0.05);
  // half the points by the transmittance cost at most twice the pure equiangular variance
  EXPECT_LT(at.relativeNoise, 1.0);
  EXPECT_GT(none.relativeNoise, 4 * at.relativeNoise);
}

// A still's light among candidates (HazeReservoir): lights the pick weighs by power alone, half ahead
// of the eye and half behind it, some hidden from the point. Kept by what they give the eye, one shadow
// ray for the kept one, the estimate is still each light's share summed, and quieter than the pick's
// one sample by more the more candidates there are.
TEST(Haze, ALightKeptAmongCandidatesCountsForThemAll)
{
  struct Light
  {
    vec3  colour;  // its light at the point over nothing: what the pick's sample over its pdf scales
    float pick;    // the pick's chance of it, by power and distance
    float phase;   // toward it, at a glow of 0.7: ahead of the eye or behind it
    bool  seen;    // the shadow ray reaches it
  };
  std::vector<Light> lights;
  std::mt19937       setup(5);
  std::uniform_real_distribution<float> u01(0.0F, 1.0F);
  float              total = 0;
  for(int i = 0; i < 24; i++)
  {
    const bool ahead = i % 2 == 0;
    Light      l;
    l.colour = vec3(0.5F + u01(setup), 0.5F + u01(setup), 0.5F + u01(setup)) * (0.2F + 2 * u01(setup));
    l.pick   = 0.2F + u01(setup);
    // the phase function's two ends at g 0.7, forward ((1 + g) / (1 - g))^3 times backward, and between
    l.phase  = float(henyeyGreenstein(ahead ? 0.9 - 0.3 * u01(setup) : -0.9 + 0.3 * u01(setup), 0.7));
    l.seen   = u01(setup) < 0.6F;
    lights.push_back(l);
    total += l.pick;
  }
  vec3 exact(0.0F);
  for(Light& l : lights)
  {
    l.pick /= total;
    exact += l.colour * l.phase * (l.seen ? 1.0F : 0.0F);
  }
  // one draw of the pick: the light, and its light over its pdf
  const auto draw = [&](std::mt19937& rng, vec3& overPdf) {
    float u = u01(rng);
    for(size_t i = 0; i < lights.size(); i++)
    {
      if(u < lights[i].pick || i + 1 == lights.size())
      {
        overPdf = lights[i].colour / lights[i].pick;
        return int(i);
      }
      u -= lights[i].pick;
    }
    return 0;
  };
  const auto lum = [](vec3 c) { return 0.2126F * c.x + 0.7152F * c.y + 0.0722F * c.z; };
  double noiseOfOne = 0;
  for(const int count : {1, 2, 4, 8})
  {
    std::mt19937 rng(100 + count);
    const int    n = 400000;
    dvec3        sum(0.0), sum2(0.0);
    for(int k = 0; k < n; k++)
    {
      HazeReservoir r = hazeReservoir();
      int           kept = -1;
      vec3          keptOverPdf(0.0F);
      for(int c = 0; c < count; c++)
      {
        vec3        overPdf;
        const int   i = draw(rng, overPdf);
        const float w = lights[i].phase * lum(overPdf);
        const float u = hazeReservoirNeedsDraw(r, w) ? u01(rng) : 0.0F;
        r             = hazeReservoirTake(r, w, u);
        if(r.taken)
        {
          kept        = i;
          keptOverPdf = overPdf;
        }
      }
      vec3 x(0.0F);
      if(kept >= 0 && lights[kept].seen)
        x = keptOverPdf * lights[kept].phase * (count > 1 ? hazeReservoirWeight(r, count) : 1.0F);
      sum += dvec3(x);
      sum2 += dvec3(x) * dvec3(x);
    }
    const dvec3  mean  = sum / double(n);
    const double noise = std::sqrt(sum2.y / n - mean.y * mean.y) / mean.y;
    if(count == 1)
      noiseOfOne = noise;
    std::printf("%d candidates: one sample's noise over its mean %.3f\n", count, noise);
    for(int ch = 0; ch < 3; ch++)
      EXPECT_NEAR(mean[ch] / exact[ch], 1.0, 0.01) << count << " candidates, channel " << ch;
    if(count == 4)
      EXPECT_LT(noise, 0.75 * noiseOfOne);
  }
  // nothing to keep: no light
  EXPECT_EQ(hazeReservoirWeight(hazeReservoir(), 4), 0.0F);
}

}  // namespace
