// The physical sky's clouds (shaders/sky_clouds.h.slang and sky_clouds_noise.h.slang) compiled as
// C++: a clear sky comes back exactly, the cloud image's mapping inverts and crowds its rows at the
// horizon, the layer is where it should be from every observer, the phase is normalised, the
// diffused light thins with depth as a slab's does, the noise tiles, and the coverage covers as much
// of the land as the bake's sunlit share assumes.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>

using namespace glm;
#include "../shaders/shaderio.h"
using namespace shaderio;
#include "../shaders/sky_clouds.h.slang"
#include "../shaders/sky_clouds_noise.h.slang"

namespace {

constexpr float kPi       = 3.14159265F;
constexpr float kEarth    = 6360.0F;  // km, the Earth preset's radius
constexpr float kObserver = kEarth + 0.3F;

float horizonElevation(float r)
{
  return -std::acos(kEarth / r);
}

float wrap01(float x)
{
  return x - std::floor(x);
}

// The bake's density at a planet-centred point, with the noise evaluated directly instead of read
// from its volumes (sky_clouds_bake.slang densityAt).
float densityAt(float3 p, float level, float thickness)
{
  const float r        = length(p);
  const float altitude = r - kEarth;
  const float h        = (altitude - CLOUD_BOTTOM_KM) / thickness;
  if(h <= 0.0F || h >= 1.0F)
    return 0.0F;
  const float2 xz = float2(p.x, p.z);
  const float  weather =
      cloudWeather(cloudWeatherNoise(float2(wrap01(xz.x / CLOUD_WEATHER_PERIOD), wrap01(xz.y / CLOUD_WEATHER_PERIOD))),
                   cloudWeatherNoise(float2(wrap01(xz.x / CLOUD_WEATHER_PERIOD2 + 0.37F), wrap01(xz.y / CLOUD_WEATHER_PERIOD2 + 0.61F))));
  if(cloudLocalCoverage(weather, level) <= 0.0F)
    return 0.0F;
  const float3 q = float3(xz.x, altitude, xz.y);
  const float  shape =
      cloudShapeNoise(float3(wrap01(q.x / CLOUD_SHAPE_PERIOD), wrap01(q.y / CLOUD_SHAPE_PERIOD), wrap01(q.z / CLOUD_SHAPE_PERIOD)));
  const float detail =
      cloudDetailNoise(float3(wrap01(q.x / CLOUD_DETAIL_PERIOD), wrap01(q.y / CLOUD_DETAIL_PERIOD), wrap01(q.z / CLOUD_DETAIL_PERIOD)));
  return cloudDensity(shape, detail, weather, h, level);
}

// The share of the land the layer shades with the sun overhead, over columns straight up from the
// ground at `ups` (on the sphere, a radius, not +Y).
double shadedShare(const std::vector<float3>& ups, float level, float thickness)
{
  const int steps  = 24;
  double    shaded = 0.0;
  for(const float3 up : ups)
  {
    double depth = 0.0;
    for(int i = 0; i < steps; ++i)
    {
      const float altitude = CLOUD_BOTTOM_KM + (float(i) + 0.5F) / float(steps) * thickness;
      depth += densityAt(up * (kEarth + altitude), level, thickness) * CLOUD_EXTINCTION * (thickness / steps);
    }
    shaded += 1.0 - std::exp(-depth);
  }
  return shaded / double(ups.size());
}

std::vector<float3> columns(int count, unsigned seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> spread(-600.0F, 600.0F);
  std::vector<float3>                   ups(count);
  for(float3& up : ups)
    up = normalize(float3(spread(rng), kEarth, spread(rng)));
  return ups;
}

}  // namespace

TEST(SkyClouds, AClearTexelGivesTheSkyBackExactly)
{
  for(const float3 sky : {float3(0.0F), float3(1.0F, 2.0F, 3.0F), float3(1e-7F, 3.7e4F, 0.123456F)})
  {
    const float3 seen = composeClouds(sky, float4(0.0F, 0.0F, 0.0F, 1.0F));
    EXPECT_EQ(seen.x, sky.x);
    EXPECT_EQ(seen.y, sky.y);
    EXPECT_EQ(seen.z, sky.z);
  }
}

TEST(SkyClouds, AnOpaqueCloudHidesTheSkyBehindIt)
{
  // What is left of the sky is the air in front, and the cloud through it
  const float4 texel = cloudTexel(float3(2.0F), 0.0F, float3(0.5F), float3(0.25F));
  const float3 seen  = composeClouds(float3(100.0F), texel);
  EXPECT_FLOAT_EQ(seen.x, 2.0F * 0.5F + 0.25F);
  EXPECT_FLOAT_EQ(texel.w, 0.0F);
}

TEST(SkyClouds, TheImageMappingInvertsAndSharesTheLatLongsAzimuth)
{
  const float minElevation = horizonElevation(kObserver);
  for(float u = 0.03F; u < 1.0F; u += 0.11F)
    for(float v = 0.02F; v <= 1.0F; v += 0.07F)
    {
      const float3 dir = cloudImageDir(float2(u, v), minElevation);
      EXPECT_NEAR(length(dir), 1.0F, 1e-5F);
      const float2 back = cloudImageUv(dir, minElevation);
      EXPECT_NEAR(back.x, u, 1e-4F);
      EXPECT_NEAR(back.y, v, 1e-4F);
      // the lat-long's azimuth (latlongToDir: phi = 2 pi u - pi = atan2(z, x))
      EXPECT_NEAR(std::atan2(dir.z, dir.x), 2.0F * kPi * u - kPi, 1e-4F);
    }
  // row 0 at the horizon, the last at the zenith
  EXPECT_NEAR(std::asin(cloudImageDir(float2(0.5F, 0.0F), minElevation).y), minElevation, 1e-5F);
  EXPECT_NEAR(cloudImageDir(float2(0.5F, 1.0F), minElevation).y, 1.0F, 1e-6F);
}

TEST(SkyClouds, TheRowsCrowdAtTheHorizonWhereTheCloudsDo)
{
  const float minElevation = horizonElevation(kObserver);
  const auto  rowsPerDegree = [&](float degrees) {
    const float e  = degrees * kPi / 180.0F;
    const float de = 0.01F * kPi / 180.0F;
    const float v0 = cloudImageUv(float3(std::cos(e), std::sin(e), 0.0F), minElevation).y;
    const float v1 = cloudImageUv(float3(std::cos(e + de), std::sin(e + de), 0.0F), minElevation).y;
    return (v1 - v0) / 0.01F * 512.0F;
  };
  EXPECT_GT(rowsPerDegree(2.0F), 3.0F * rowsPerDegree(60.0F));
  // and the zenith keeps a few: its clouds are the nearest and the largest
  EXPECT_GT(rowsPerDegree(85.0F), 2.0F);
}

TEST(SkyClouds, TheLayerIsWhereItIsFromBelowInsideAndAbove)
{
  const float rBottom = kEarth + 1.5F;
  const float rTop    = rBottom + 1.0F;

  // Overhead from the ground: from the base to the tops
  float2 s = cloudLayerSegment(kObserver, 1.0F, rBottom, rTop);
  EXPECT_NEAR(s.x, 1.2F, 1e-3F);
  EXPECT_NEAR(s.y, 2.2F, 1e-3F);

  // Along the horizontal the layer is far away and long: the base ~138 km off, the tops ~40 km on
  s = cloudLayerSegment(kObserver, 0.0F, rBottom, rTop);
  EXPECT_NEAR(s.x, std::sqrt(rBottom * rBottom - kObserver * kObserver), 0.05F);
  EXPECT_GT(s.y - s.x, 30.0F);

  // Inside the layer, looking down: from the eye to the base
  const float inside = kEarth + 2.0F;
  s                  = cloudLayerSegment(inside, -1.0F, rBottom, rTop);
  EXPECT_NEAR(s.x, 0.0F, 1e-6F);
  EXPECT_NEAR(s.y, 0.5F, 1e-3F);

  // Above it, looking down: from the tops to the base
  const float above = kEarth + 5.0F;
  s                 = cloudLayerSegment(above, -1.0F, rBottom, rTop);
  EXPECT_NEAR(s.x, 2.5F, 1e-3F);
  EXPECT_NEAR(s.y, 3.5F, 1e-3F);

  // ... and up, away from it: none
  s = cloudLayerSegment(above, 1.0F, rBottom, rTop);
  EXPECT_LE(s.y, s.x);
}

TEST(SkyClouds, CoverageRunsFromNoneToAll)
{
  for(float w = 0.0F; w <= 1.0F; w += 0.05F)
  {
    EXPECT_EQ(cloudLocalCoverage(w, 0.0F), 0.0F) << w;
    EXPECT_GE(cloudLocalCoverage(w, 1.0F), 1.0F) << w;  // past what the shape holds: a deck without holes
    float last = 0.0F;
    for(float c = 0.0F; c <= 1.0F; c += 0.05F)
    {
      const float local = cloudLocalCoverage(w, c);
      EXPECT_GE(local, last);
      last = local;
    }
  }
  // Nothing outside the layer, however much cloud
  EXPECT_EQ(cloudDensity(1.0F, 0.5F, 1.0F, 0.0F, 1.0F), 0.0F);
  EXPECT_EQ(cloudDensity(1.0F, 0.5F, 1.0F, 1.0F, 1.0F), 0.0F);
  EXPECT_GT(cloudDensity(1.0F, 0.5F, 1.0F, 0.3F, 1.0F), 0.5F);
  // and the sunlit share follows: all of it clear, none of it overcast
  EXPECT_EQ(cloudSunlitAir(0.0F), 1.0F);
  EXPECT_EQ(cloudSunlitAir(1.0F), 0.0F);
  // the cover's level rises with it, from where the first clouds pass to a full deck
  float last = 0.0F;
  for(float cover = 0.0F; cover <= 1.0F; cover += 0.01F)
  {
    EXPECT_GT(cloudLevel(cover), last);
    last = cloudLevel(cover);
  }
  EXPECT_FLOAT_EQ(cloudLevel(1.0F), 1.0F);
}

TEST(SkyClouds, ThePhaseIsNormalisedAndScattersForward)
{
  double sum = 0.0;
  const int n = 20000;
  for(int i = 0; i < n; ++i)
  {
    const float c = -1.0F + 2.0F * (float(i) + 0.5F) / float(n);
    sum += double(cloudPhase(c)) * 2.0 * double(kPi) * (2.0 / n);
  }
  EXPECT_NEAR(sum, 1.0, 1e-3);
  EXPECT_GT(cloudPhase(1.0F), 50.0F * cloudPhase(-1.0F));
}

TEST(SkyClouds, DiffusedLightThinsWithDepthAsASlabPassesIt)
{
  // The sunlit face reflects as a white surface of the cloud's albedo would, plus the single scatter
  const float3 sun  = float3(1.0F);
  const float  face = cloudLight(sun, 0.0F, -1.0F, float3(0.0F), 1.0F).x;
  EXPECT_NEAR(face, CLOUD_DIFFUSE_ALBEDO / kPi + cloudPhase(-1.0F), 1e-5F);

  // A deck of optical depth 20 passes about a third of the light to its base: 1 / (1 + 3/4 (1 - g) tau)
  const float base = cloudLight(sun, 20.0F, -1.0F, float3(0.0F), 0.0F).x;
  EXPECT_NEAR(base / (CLOUD_DIFFUSE_ALBEDO / kPi), 1.0F / (1.0F + 0.1125F * 20.0F), 1e-3F);

  // Toward the sun, a thin edge glows: the forward lobe outshines the sunlit face several times over
  EXPECT_GT(cloudLight(sun, 0.5F, 1.0F, float3(0.0F), 0.5F).x, 5.0F * face);
}

TEST(SkyClouds, TheNoiseTilesAndStaysInRange)
{
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  for(int i = 0; i < 64; ++i)
  {
    const float3 p(unit(rng), unit(rng), unit(rng));
    const float  shape = cloudShapeNoise(p);
    EXPECT_GE(shape, 0.0F);
    EXPECT_LE(shape, 1.0F);
    EXPECT_NEAR(cloudShapeNoise(p + float3(1.0F, 0.0F, 0.0F)), shape, 1e-4F);
    EXPECT_NEAR(cloudShapeNoise(p + float3(0.0F, 0.0F, -1.0F)), shape, 1e-4F);

    const float detail = cloudDetailNoise(p);
    EXPECT_GE(detail, 0.0F);
    EXPECT_LE(detail, 1.0F);
    EXPECT_NEAR(cloudDetailNoise(p + float3(0.0F, 1.0F, 0.0F)), detail, 1e-4F);

    const float2 q(p.x, p.y);
    const float  weather = cloudWeatherNoise(q);
    EXPECT_GE(weather, 0.0F);
    EXPECT_LE(weather, 1.0F);
    EXPECT_NEAR(cloudWeatherNoise(q + float2(1.0F, 0.0F)), weather, 1e-4F);
  }
}

// Coverage is the share of the land the clouds shade with the sun overhead: what the panel says, and
// what the bake assumes for the distant ground and haze (cloudSunlitAir). Other columns than the
// table was fitted on.
TEST(SkyClouds, TheCoverageShadesAsMuchOfTheLandAsItSays)
{
  const std::vector<float3> ups = columns(800, 11);
  for(const float cover : {0.1F, 0.3F, 0.5F, 0.7F, 0.9F, 1.0F})
  {
    const double shaded = shadedShare(ups, cloudLevel(cover), 1.2F);
    std::printf("[ cover %.2f ] shades %.3f of the land\n", cover, shaded);
    EXPECT_NEAR(shaded, 1.0 - cloudSunlitAir(cover), 0.06) << cover;
  }
}

// Not a test: what CLOUD_LEVEL_FOR_COVER is fitted from. Prints the shaded share for weather levels
// from 0.2 to 1; invert it at every 5% of cover (the knots) when the noise or the density changes.
TEST(SkyClouds, DISABLED_CoverageCurve)
{
  const std::vector<float3> ups = columns(3000, 5);
  for(int k = 0; k <= 32; ++k)
  {
    const float level = 0.2F + 0.025F * float(k);
    std::printf("level %.3f shades %.4f\n", level, shadedShare(ups, level, 1.2F));
  }
}
