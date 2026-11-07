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
  // one draw of the pick at `u`: the light, and its light over its pdf
  const auto drawAt = [&](float u, vec3& overPdf) {
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
  double noiseOfOne = 0, noiseOfFour = 0;
  // independent candidates, then the shader's: one scoring, the c-th from the c-th slice of it
  for(const bool sliced : {false, true})
  for(const int count : {1, 2, 4, 8})
  {
    std::mt19937 rng(100 + count + (sliced ? 50 : 0));
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
        const float pick = sliced ? (float(c) + u01(rng)) / float(count) : u01(rng);
        const int   i    = drawAt(pick, overPdf);
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
    if(count == 1 && !sliced)
      noiseOfOne = noise;
    std::printf("%d candidates%s: one sample's noise over its mean %.3f\n", count, sliced ? ", sliced" : "", noise);
    for(int ch = 0; ch < 3; ch++)
      EXPECT_NEAR(mean[ch] / exact[ch], 1.0, 0.01) << count << " candidates, channel " << ch << (sliced ? ", sliced" : "");
    if(count == 4 && !sliced)
    {
      noiseOfFour = noise;
      EXPECT_LT(noise, 0.75 * noiseOfOne);
    }
    if(count == 4 && sliced)
      EXPECT_LE(noise, 1.02 * noiseOfFour);  // the slices spread them: no noisier than independent draws
  }
  // nothing to keep: no light
  EXPECT_EQ(hazeReservoirWeight(hazeReservoir(), 4), 0.0F);
}

//-----------------------------------------------------------------------
// Uneven haze
//-----------------------------------------------------------------------

// A ray of the uneven haze's tests and the integral of the height's factor along it by the midpoint rule
struct HeightRay
{
  vec3  o, d;
  float ta, tb;
};

double heightDepthByQuadrature(const HazeShape& shape, const HeightRay& r, double upTo, int steps = 200000)
{
  const double len = upTo - r.ta, step = len / steps;
  double       sum = 0;
  for(int i = 0; i < steps; i++)
  {
    const double t = r.ta + (i + 0.5) * step;
    const double z = double(r.o.y) + double(r.d.y) * t - shape.floorY;
    sum += z > 0 ? std::exp(-z * shape.invHeight) : 1.0;
  }
  return sum * step;
}

std::vector<HeightRay> heightRays()
{
  return {
      {vec3(0, 1.0F, 0), vec3(1, 0, 0), 0.0F, 20.0F},                           // level, 1 m up
      {vec3(0, 0.0F, 0), normalize(vec3(1, 0.4F, 0)), 0.0F, 30.0F},             // rising from the floor
      {vec3(0, 30.0F, 0), normalize(vec3(1, -1.5F, 0.2F)), 0.0F, 21.0F},        // falling from 30 m to near the floor
      {vec3(0, 1.7F, 0), normalize(vec3(1, -0.08F, 0)), 0.5F, 25.0F},           // an eye looking a little down, through the floor
      {vec3(0, -0.5F, 0), normalize(vec3(1, 0.3F, 0)), 0.0F, 12.0F},            // from below the floor up out of it
      {vec3(0, 2.0F, 0), normalize(vec3(1, 1e-6F, 0)), 0.0F, 50.0F},            // all but level
      {vec3(0, 80.0F, 0), normalize(vec3(0.2F, -1.0F, 0)), 0.0F, 81.0F},        // from far above down to the floor
  };
}

// The height's integral along a ray is its quadrature, rising, falling, level and through the floor,
// and the inversion lands where the integral reaches a given depth
TEST(Haze, TheHeightsDepthIsItsIntegralAndItsInverse)
{
  for(const float height : {0.3F, 1.5F, 20.0F})
  {
    const HazeShape shape{0.0F, 1.0F / height, 0.0F, 1.0F};
    for(const HeightRay& r : heightRays())
    {
      const double exact = heightDepthByQuadrature(shape, r, r.tb);
      const float  depth = hazeHeightDepth(shape, r.o, r.d, r.ta, r.tb);
      EXPECT_NEAR(depth / exact, 1.0, 1e-3) << "H " << height << ", from y " << r.o.y << " along y " << r.d.y;
      for(const double share : {0.0, 0.01, 0.3, 0.7, 0.999})
      {
        const float t = hazeHeightInvert(shape, r.o, r.d, r.ta, r.tb, float(share * depth));
        ASSERT_GE(t, r.ta);
        ASSERT_LE(t, r.tb);
        EXPECT_NEAR(hazeHeightDepth(shape, r.o, r.d, r.ta, t), share * depth, 2e-4 * depth + 1e-6)
            << "H " << height << ", from y " << r.o.y << " along y " << r.d.y << ", share " << share;
      }
    }
  }
  // even through the height: the length
  const HazeShape even{0.0F, 0.0F, 0.0F, 1.0F};
  EXPECT_EQ(hazeHeightDepth(even, vec3(0, 3, 0), vec3(1, 0, 0), 2.0F, 9.0F), 7.0F);
  EXPECT_EQ(hazeHeightInvert(even, vec3(0, 3, 0), vec3(1, 0, 0), 2.0F, 9.0F, 4.0F), 6.0F);
  EXPECT_TRUE(hazeEven(even));
  EXPECT_FALSE(hazeEven(HazeShape{0.0F, 0.0F, 0.5F, 1.0F}));
}

// The patches' noise: within [-1, 1], about 0 on average with a spread, and smooth
TEST(Haze, ThePatchesAreASmoothNoise)
{
  std::mt19937                          rng(5);
  std::uniform_real_distribution<float> place(-40.0F, 40.0F);
  double                                sum = 0, sum2 = 0, largestStep = 0;
  const int                             n   = 200000;
  for(int i = 0; i < n; i++)
  {
    const vec3  p = vec3(place(rng), place(rng), place(rng));
    const float v = hazeNoise(p);
    ASSERT_GE(v, -1.0F);
    ASSERT_LE(v, 1.0F);
    sum += v;
    sum2 += double(v) * v;
    largestStep = std::max(largestStep, double(std::abs(hazeNoise(p + vec3(0.01F, 0, 0)) - v)));
  }
  const double mean = sum / n, spread = std::sqrt(sum2 / n - mean * mean);
  EXPECT_NEAR(mean, 0.0, 0.02);
  EXPECT_GT(spread, 0.15);
  EXPECT_LT(largestStep, 0.06);  // a hundredth of a cell moves it by little
}

// The uneven haze of these tests: a fog layer 0.8 m deep with patches, an eye at 3 m looking down
// at the floor 10 m away, a point light near where the ray meets it
struct UnevenScene
{
  HazeShape shape{0.0F, 1.0F / 0.8F, 0.6F, 1.0F / 1.5F};
  float     sigma = 3.912023F / 20.0F;  // a visibility of 20 m at the floor
  float     g     = 0.7F;
  vec3      o     = vec3(0.0F, 3.0F, 0.0F);
  vec3      d     = normalize(vec3(1.0F, -0.29F, 0.1F));
  float     t0 = 0.0F, t1 = 10.0F;
  vec3      light     = vec3(8.0F, 0.6F, 1.5F);
  float     intensity = 2.0F;

  // the light's way to a point: dimmed by the height alone, as the shader's air
  double air(vec3 x) const
  {
    const vec3 toward = light - x;
    return std::exp(-sigma * hazeHeightDepth(shape, x, normalize(toward), 0.0F, length(toward)));
  }
  double lit(vec3 x) const
  {
    const vec3   toward = light - x;
    const double r2     = dot(toward, toward);
    return henyeyGreenstein(dot(d, toward / float(std::sqrt(r2))), g) * air(x) * intensity / r2;
  }
};

// The single scattering by fine quadrature, the patches in the transmittance and the density
double unevenReference(const UnevenScene& s, double* transmittance = nullptr, int steps = 400000)
{
  const double step = (s.t1 - s.t0) / steps;
  double       tau = 0, sum = 0;
  for(int i = 0; i < steps; i++)
  {
    const float  t       = float(s.t0 + (i + 0.5) * step);
    const vec3   x       = s.o + s.d * t;
    const double density = hazeDensity(s.shape, x);
    sum += s.sigma * density * std::exp(-s.sigma * (tau + 0.5 * density * step)) * s.lit(x) * step;
    tau += density * step;
  }
  if(transmittance)
    *transmittance = std::exp(-s.sigma * tau);
  return sum;
}

// The shader's uneven estimate, guided at the light: the column, the mixture's distance and its weight
TEST(Haze, AnUnevenHazesHaloMatchesItsSingleScatter)
{
  for(const float patches : {0.0F, 0.6F})
  {
    UnevenScene s;
    s.shape.patches = patches;
    double       exactT = 0;
    const double ref    = unevenReference(s, &exactT);
    const HazeGuide guide = hazeGuide(s.o, s.d, s.t0, s.t1, s.light, true);
    std::mt19937                          rng(21 + int(patches * 10));
    std::uniform_real_distribution<float> u01(0.0F, 0.99999994F);
    double                                sum = 0, sumT = 0;
    const int                             n   = 400000;
    for(int k = 0; k < n; k++)
    {
      const HazeColumn c = hazeColumn(s.shape, s.o, s.d, s.t0, s.t1, u01(rng));
      sumT += std::exp(-s.sigma * c.total);
      const float choice = u01(rng), u = u01(rng);
      const float t = (guide.on && choice < HAZE_GUIDE_SHARE) ? hazeSampleEquiangular(u, s.t0, s.t1, guide)
                                                               : hazeColumnSampleTransmittance(s.shape, c, s.o, s.d, s.sigma, s.t0, s.t1, u);
      sum += double(hazeColumnScatterWeight(s.shape, c, s.o, s.d, s.sigma, s.t0, t, guide)) * s.lit(s.o + s.d * t);
    }
    std::printf("uneven, patches %.1f: in-scatter %.5g against %.5g, transmittance %.5f against %.5f\n", patches, sum / n, ref,
                sumT / n, exactT);
    EXPECT_NEAR(sum / n / ref, 1.0, 0.015) << "patches " << patches;
    // the column's transmittance: exact without patches, a hair high with them
    EXPECT_NEAR(sumT / n / exactT, 1.0, patches > 0.0F ? 0.01 : 1e-4) << "patches " << patches;
  }
}

// Drawn by the column's transmittance alone, the distances follow hazeColumnDistancePdf
TEST(Haze, TheUnevenDistancesFollowTheirDensity)
{
  UnevenScene      s;
  const HazeColumn c = hazeColumn(s.shape, s.o, s.d, s.t0, s.t1, 0.37F);
  const int        bins = 40;
  std::vector<int> count(bins, 0);
  std::mt19937     rng(3);
  std::uniform_real_distribution<float> u01(0.0F, 0.99999994F);
  const int        n = 800000;
  for(int k = 0; k < n; k++)
  {
    const float t = hazeColumnSampleTransmittance(s.shape, c, s.o, s.d, s.sigma, s.t0, s.t1, u01(rng));
    count[std::min(int((t - s.t0) / (s.t1 - s.t0) * bins), bins - 1)]++;
  }
  for(int b = 0; b < bins; b++)
  {
    // the bin's probability by the pdf, midpoint rule inside it
    double p = 0;
    for(int j = 0; j < 200; j++)
    {
      const float t = s.t0 + (s.t1 - s.t0) * (b + (j + 0.5F) / 200.0F) / bins;
      p += hazeColumnDistancePdf(s.shape, c, s.o, s.d, s.sigma, s.t0, t, hazeNoGuide());
    }
    p *= (s.t1 - s.t0) / bins / 200.0;
    const double sd = std::sqrt(p * (1 - p) / n);
    EXPECT_NEAR(double(count[b]) / n, p, 5 * sd + 1e-5) << "bin " << b;
  }
}

}  // namespace
