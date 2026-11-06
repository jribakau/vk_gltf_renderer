// The physical sky's clouds (shaders/sky_clouds.h.slang and sky_clouds_noise.h.slang, takram's model)
// compiled as C++: a clear sky comes back exactly, the cloud image's mapping inverts and crowds its
// rows at the horizon, the layers are where they should be from every observer, the coverage runs
// from none to all, the phase is normalised and the octaves carry light deep into a cloud, the noise
// tiles, and the panel's cover shades as much of the land as it says.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace glm;
#include "../shaders/shaderio.h"
#include "../shaders/sky_clouds_io.h.slang"
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

float3 wrap01(float3 x)
{
  return x - floor(x);
}

// One column straight up from the ground, with what does not depend on the coverage read once: the
// weather map over it and the two volumes at each height, as the bake reads them (sky_clouds_bake.slang
// mediaAt), the noise evaluated directly instead of from its textures. Without the turbulence, which
// moves the billows about but not how many there are.
struct Column
{
  float4             weather;
  std::vector<float> heights;
  std::vector<float> steps;  // km of column each height stands for
  std::vector<float> shape;
  std::vector<float> detail;
};

std::vector<Column> columns(int count, unsigned seed, const SkyCloudsParameters& p)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> spread(-600.0F, 600.0F);
  std::vector<Column>                   out(count);
  for(Column& c : out)
  {
    const float x = spread(rng), z = spread(rng);
    c.weather     = cloudWeatherNoise(float2(wrap01(x / CLOUD_WEATHER_PERIOD), wrap01(z / CLOUD_WEATHER_PERIOD)));
    // the cumulus layers finely, the high one coarsely, the gap between not at all
    const auto band = [&](float bottom, float top, int n) {
      for(int i = 0; i < n; ++i)
      {
        const float  h = bottom + (float(i) + 0.5F) / float(n) * (top - bottom);
        const float3 q(x, h, z);
        c.heights.push_back(h);
        c.steps.push_back((top - bottom) / float(n));
        c.shape.push_back(std::clamp(cloudShapeNoise(wrap01(q / CLOUD_SHAPE_PERIOD)), 0.0F, 1.0F));
        c.detail.push_back(cloudDetailNoise(wrap01(q / CLOUD_DETAIL_PERIOD)));
      }
    };
    band(p.minHeight, p.shadowTopHeight, 24);
    band(p.minLayerHeights.z, p.maxLayerHeights.z, 4);
  }
  return out;
}

// The share of the land the layers shade with the sun overhead, at a coverage.
double shadedShare(const std::vector<Column>& cols, SkyCloudsParameters p, float coverage)
{
  p.coverage    = coverage;
  double shaded = 0.0;
  for(const Column& c : cols)
  {
    double depth = 0.0;
    for(size_t i = 0; i < c.heights.size(); ++i)
    {
      const CloudWeather w = cloudSampleWeather(p, c.weather, c.heights[i]);
      if(cloudWeatherHasDensity(p, w))
        depth += cloudSampleMedia(p, w, c.shape[i], c.detail[i], true).extinction * c.steps[i];
    }
    shaded += 1.0 - std::exp(-depth);
  }
  return shaded / double(cols.size());
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
  const float minElevation  = horizonElevation(kObserver);
  const float rows          = 1024.0F;  // kCloudImageSize.height
  const auto  rowsPerDegree = [&](float degrees) {
    const float e  = degrees * kPi / 180.0F;
    const float de = 0.01F * kPi / 180.0F;
    const float v0 = cloudImageUv(float3(std::cos(e), std::sin(e), 0.0F), minElevation).y;
    const float v1 = cloudImageUv(float3(std::cos(e + de), std::sin(e + de), 0.0F), minElevation).y;
    return (v1 - v0) / 0.01F * rows;
  };
  EXPECT_GT(rowsPerDegree(2.0F), 3.0F * rowsPerDegree(60.0F));
  // and the zenith keeps a few: its clouds are the nearest and the largest
  EXPECT_GT(rowsPerDegree(85.0F), 4.0F);
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

TEST(SkyClouds, TheDefaultLayersAreTakramsAndTheDepthScalesTheCumulus)
{
  const SkyCloudsParameters p = cloudDefaultParameters(0.5F, CLOUD_DEFAULT_THICKNESS);
  EXPECT_FLOAT_EQ(p.minLayerHeights.x, 0.75F);
  EXPECT_FLOAT_EQ(p.maxLayerHeights.x, 1.4F);
  EXPECT_FLOAT_EQ(p.maxLayerHeights.y, 2.2F);
  EXPECT_FLOAT_EQ(p.minLayerHeights.z, 7.5F);
  EXPECT_FLOAT_EQ(p.maxHeight, 8.0F);
  // the march steps over the gap between the cumulus and the high layer, and nothing else
  EXPECT_TRUE(cloudInsideIntervals(p, 5.0F));
  EXPECT_FALSE(cloudInsideIntervals(p, 1.5F));
  EXPECT_FALSE(cloudInsideIntervals(p, 7.7F));

  const SkyCloudsParameters deep = cloudDefaultParameters(0.5F, 2.0F * CLOUD_DEFAULT_THICKNESS);
  EXPECT_FLOAT_EQ(deep.maxLayerHeights.y - deep.minLayerHeights.y, 2.4F);
  EXPECT_FLOAT_EQ(deep.minLayerHeights.y, 1.0F);
  EXPECT_FLOAT_EQ(deep.maxLayerHeights.z, 8.0F);  // the high layer stays

  // a preview is coarser but keeps the octaves, which hold the clouds' brightness
  const SkyCloudsParameters preview = cloudWithQuality(p, true);
  EXPECT_LT(preview.maxIterations, p.maxIterations);
  EXPECT_EQ(preview.shapeDetail, 0);
  EXPECT_EQ(preview.octaves, p.octaves);

  // the settled image averages the march's jitter over several marches a texel, or its crosshatch
  // stays frozen in the clouds' edges; a drag's preview makes do with one
  EXPECT_GE(p.samples, 8);
  EXPECT_EQ(cloudWithQuality(p, false).samples, p.samples);
  EXPECT_EQ(preview.samples, 1);
}

TEST(SkyClouds, CoverageRunsFromNoneToAll)
{
  const SkyCloudsParameters p = cloudDefaultParameters(0.0F, CLOUD_DEFAULT_THICKNESS);
  for(float w = 0.0F; w <= 1.0F; w += 0.1F)
    for(float h = 0.8F; h < 2.2F; h += 0.1F)
    {
      SkyCloudsParameters none = p;
      none.coverage            = 0.0F;
      EXPECT_FALSE(cloudWeatherHasDensity(none, cloudSampleWeather(none, float4(w), h))) << w << " " << h;
      // rising with the coverage everywhere
      float last = 0.0F;
      for(float c = 0.0F; c <= 1.0F; c += 0.1F)
      {
        SkyCloudsParameters at = p;
        at.coverage            = c;
        const float4 d         = cloudSampleWeather(at, float4(w), h).density;
        EXPECT_GE(d.x + d.y, last - 1e-6F);
        last = d.x + d.y;
      }
    }
  // The panel's cover maps onto it from the first clouds to a closed deck, rising
  EXPECT_FLOAT_EQ(cloudCoverageFor(0.0F), CLOUD_COVERAGE_FOR_COVER[0]);
  EXPECT_FLOAT_EQ(cloudCoverageFor(1.0F), CLOUD_COVERAGE_FOR_COVER[20]);
  float last = -1.0F;
  for(float cover = 0.0F; cover <= 1.0F; cover += 0.01F)
  {
    EXPECT_GT(cloudCoverageFor(cover), last);
    last = cloudCoverageFor(cover);
  }
  EXPECT_EQ(cloudSunlitAir(0.0F), 1.0F);
  EXPECT_EQ(cloudSunlitAir(1.0F), 0.0F);
}

TEST(SkyClouds, ThePhaseIsNormalisedAndScattersForward)
{
  for(const float attenuation : {1.0F, 0.5F, 0.125F})
  {
    double    sum = 0.0;
    const int n   = 20000;
    for(int i = 0; i < n; ++i)
    {
      const float c = -1.0F + 2.0F * (float(i) + 0.5F) / float(n);
      sum += double(cloudPhase(c, attenuation)) * 2.0 * double(kPi) * (2.0 / n);
    }
    EXPECT_NEAR(sum, 1.0, 2e-3) << attenuation;
  }
  EXPECT_GT(cloudPhase(1.0F, 1.0F), 10.0F * cloudPhase(0.0F, 1.0F));
}

TEST(SkyClouds, TheOctavesCarryLightDeepIntoACloud)
{
  // At the surface the octaves add up to their shares, each phase normalised: 2 (1 - 2^-8)
  double    sum = 0.0;
  const int n   = 20000;
  for(int i = 0; i < n; ++i)
  {
    const float c = -1.0F + 2.0F * (float(i) + 0.5F) / float(n);
    sum += double(cloudMultipleScattering(0.0F, c, 8)) * 2.0 * double(kPi) * (2.0 / n);
  }
  EXPECT_NEAR(sum, 2.0 * (1.0 - 1.0 / 256.0), 5e-3);

  // Deep in it, where single scattering is gone, the flattest octaves still pass light
  const float deep = cloudMultipleScattering(20.0F, 0.0F, 8);
  EXPECT_GT(deep, 1e5F * std::exp(-20.0F) * cloudPhase(0.0F, 1.0F));
  EXPECT_LT(deep, cloudMultipleScattering(5.0F, 0.0F, 8));
}

TEST(SkyClouds, TheNoiseTilesAndStaysInRange)
{
  std::mt19937                          rng(7);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  for(int i = 0; i < 24; ++i)
  {
    const float3 p(unit(rng), unit(rng), unit(rng));
    const float  shape = cloudShapeNoise(p);
    EXPECT_NEAR(cloudShapeNoise(p + float3(1.0F, 0.0F, 0.0F)), shape, 2e-3F);
    EXPECT_NEAR(cloudShapeNoise(p + float3(0.0F, 0.0F, -1.0F)), shape, 2e-3F);

    const float detail = cloudDetailNoise(p);
    EXPECT_GE(detail, 0.0F);
    EXPECT_LE(detail, 1.0F);
    EXPECT_NEAR(cloudDetailNoise(p + float3(0.0F, 1.0F, 0.0F)), detail, 2e-3F);

    const float2 q(p.x, p.y);
    const float4 weather = cloudWeatherNoise(q);
    const float4 wrapped = cloudWeatherNoise(q + float2(1.0F, 0.0F));
    for(int c = 0; c < 4; ++c)
    {
      EXPECT_GE(weather[c], 0.0F);
      EXPECT_LE(weather[c], 1.0F);
      EXPECT_NEAR(wrapped[c], weather[c], 2e-3F) << c;
    }

    const float3 turbulence = cloudTurbulenceNoise(q);
    EXPECT_NEAR(length(turbulence * 2.0F - 1.0F), 1.0F, 1e-3F);
    EXPECT_NEAR(length(cloudTurbulenceNoise(q + float2(0.0F, 1.0F)) - turbulence), 0.0F, 5e-3F);
  }
}

// Coverage is the share of the land the clouds shade with the sun overhead: what the panel says, and
// what the bake assumes for the distant ground and haze (cloudSunlitAir). Other columns than the
// table was fitted on.
TEST(SkyClouds, TheCoverShadesAsMuchOfTheLandAsItSays)
{
  const SkyCloudsParameters p    = cloudDefaultParameters(0.0F, CLOUD_DEFAULT_THICKNESS);
  const std::vector<Column> cols = columns(600, 11, p);
  for(const float cover : {0.02F, 0.1F, 0.3F, 0.5F, 0.7F, 0.9F, 1.0F})
  {
    const double shaded = shadedShare(cols, p, cloudCoverageFor(cover));
    std::printf("[ cover %.2f ] shades %.3f of the land\n", cover, shaded);
    EXPECT_NEAR(shaded, 1.0 - cloudSunlitAir(cover), 0.06) << cover;
  }
}

// Not a test: what CLOUD_COVERAGE_FOR_COVER is fitted from. Prints the shaded share for takram's
// coverage from 0 to 1; invert it at every 5% of cover (the table's knots) when the model changes.
TEST(SkyClouds, DISABLED_CoverageCurve)
{
  const SkyCloudsParameters p    = cloudDefaultParameters(0.0F, CLOUD_DEFAULT_THICKNESS);
  const std::vector<Column> cols = columns(2500, 5, p);
  for(int k = 0; k <= 40; ++k)
  {
    const float coverage = 0.025F * float(k);
    std::printf("coverage %.3f shades %.4f\n", coverage, shadedShare(cols, p, coverage));
  }
}
