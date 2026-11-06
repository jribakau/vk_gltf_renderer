// The physical sky's clouds (shaders/sky_clouds.h.slang and sky_clouds_noise.h.slang, takram's model)
// compiled as C++: a clear sky comes back exactly, the cloud image's mapping inverts and crowds its
// rows at the horizon, the layers are where they should be from every observer, the coverage runs
// from none to all, the phase is normalised and the octaves carry light deep into a cloud, the noise
// tiles, each kind is the cloud it says, the tiers lie over one another, and each tier's cover shades
// (cirrus: veils) as much of the land as it says.

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
// mediaAt) -- at the first layer's scale and at the fourth's -- the noise evaluated directly instead of
// from its textures. The heights are the spans the march marches, the gaps between them left out.
// Without the turbulence, which moves the billows about but not how many there are.
struct Column
{
  float4              weather;
  std::vector<float>  heights;
  std::vector<float>  steps;  // km of column each height stands for
  std::vector<float4> shape;
  std::vector<float4> detail;
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
    const auto band = [&](float bottom, float top, int n) {
      for(int i = 0; i < n; ++i)
      {
        const float  h = bottom + (float(i) + 0.5F) / float(n) * (top - bottom);
        const float3 q(x, h, z);
        // as the bake's noiseAt: layer 1 shares layer 0's
        float4 shape(0.0F), detail(0.0F);
        for(int l : {0, 2, 3})
        {
          const float  scale = p.shapeScales[l];
          const float3 f     = float3(scale * p.shapeStretches[l], scale, scale);
          shape[l]           = std::clamp(cloudShapeNoise(wrap01(q * f / CLOUD_SHAPE_PERIOD)), 0.0F, 1.0F);
          detail[l]          = cloudDetailNoise(wrap01(q * f / CLOUD_DETAIL_PERIOD));
        }
        shape.y  = shape.x;
        detail.y = detail.x;
        c.heights.push_back(h);
        c.steps.push_back((top - bottom) / float(n));
        c.shape.push_back(shape);
        c.detail.push_back(detail);
      }
    };
    float bottom = p.minHeight;
    for(int g = 0; g < 3; ++g)
      if(p.maxIntervalHeights[g] > p.minIntervalHeights[g])
      {
        band(bottom, p.minIntervalHeights[g], 24);
        bottom = p.maxIntervalHeights[g];
      }
    band(bottom, p.maxHeight, 24);
  }
  return out;
}

// Takram's coverage on every layer a tier turned on.
SkyCloudsParameters withCoverage(SkyCloudsParameters p, float coverage)
{
  for(int i = 0; i < 4; ++i)
    p.coverages[i] = p.densityScales[i] > 0.0F ? coverage : 0.0F;
  return p;
}

// What the layers do to the land with the sun overhead, at a coverage: the share they shade, and the
// share they veil at all (an optical depth of 0.05 or more, cirrus's cover).
struct Shade
{
  double shaded;
  double veiled;
};

Shade shadeAt(const std::vector<Column>& cols, const SkyCloudsParameters& layers, float coverage)
{
  const SkyCloudsParameters p = withCoverage(layers, coverage);
  Shade                     out{0.0, 0.0};
  for(const Column& c : cols)
  {
    double depth = 0.0;
    for(size_t i = 0; i < c.heights.size(); ++i)
    {
      const CloudWeather w = cloudSampleWeather(p, c.weather, c.heights[i]);
      if(cloudWeatherHasDensity(p, w))
        depth += cloudSampleMedia(p, w, c.shape[i], c.detail[i], true).extinction * c.steps[i];
    }
    out.shaded += 1.0 - std::exp(-depth);
    out.veiled += depth >= 0.05 ? 1.0 : 0.0;
  }
  out.shaded /= double(cols.size());
  out.veiled /= double(cols.size());
  return out;
}

// The share a kind's cover means: what cirrus veils, what the others shade.
double coverShare(const Shade& s, int type)
{
  return type == CLOUD_TYPE_CIRRUS ? s.veiled : s.shaded;
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
  const SkyCloudsParameters p = cloudDefaultParameters(0.5F, CLOUD_DEFAULT_THICKNESS, CLOUD_TYPE_CUMULUS);
  EXPECT_FLOAT_EQ(p.minLayerHeights.x, 0.75F);
  EXPECT_FLOAT_EQ(p.maxLayerHeights.x, 1.4F);
  EXPECT_FLOAT_EQ(p.maxLayerHeights.y, 2.2F);
  EXPECT_FLOAT_EQ(p.minHeight, 0.75F);
  EXPECT_FLOAT_EQ(p.maxHeight, 2.2F);
  EXPECT_FLOAT_EQ(p.shadowTopHeight, 2.2F);
  // takram's high layer is the high tier's now: alone, the cumulus are one span with no gap in it
  EXPECT_EQ(p.densityScales.z, 0.0F);
  EXPECT_FALSE(cloudInsideIntervals(p, 1.5F));

  const SkyCloudsParameters deep = cloudDefaultParameters(0.5F, 2.0F * CLOUD_DEFAULT_THICKNESS, CLOUD_TYPE_CUMULUS);
  EXPECT_FLOAT_EQ(deep.maxLayerHeights.y - deep.minLayerHeights.y, 2.4F);
  EXPECT_FLOAT_EQ(deep.minLayerHeights.y, 1.0F);

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
  for(int type = 0; type < CLOUD_TYPE_COUNT; ++type)
  {
    const SkyCloudsParameters p = cloudDefaultParameters(0.0F, cloudDefaultThickness(type), type);
    for(float w = 0.0F; w <= 1.0F; w += 0.1F)
      for(float h = p.minHeight + 0.05F; h < p.shadowTopHeight; h += 0.05F)
      {
        const SkyCloudsParameters none = withCoverage(p, 0.0F);
        EXPECT_FALSE(cloudWeatherHasDensity(none, cloudSampleWeather(none, float4(w), h))) << type << " " << w << " " << h;
        // rising with the coverage everywhere
        float last = 0.0F;
        for(float c = 0.0F; c <= 1.0F; c += 0.1F)
        {
          const SkyCloudsParameters at = withCoverage(p, c);
          const float4              d  = cloudSampleWeather(at, float4(w), h).density;
          EXPECT_GE(d.x + d.y + d.z + d.w, last - 1e-6F) << type;
          last = d.x + d.y + d.z + d.w;
        }
      }
    // The panel's cover maps onto it from the first clouds to a closed deck, rising
    EXPECT_FLOAT_EQ(cloudCoverageFor(0.0F, type), CLOUD_COVERAGE_FOR_COVER[type][0]);
    EXPECT_FLOAT_EQ(cloudCoverageFor(1.0F, type), CLOUD_COVERAGE_FOR_COVER[type][20]);
    float last = -1.0F;
    for(float cover = 0.0F; cover <= 1.0F; cover += 0.01F)
    {
      EXPECT_GT(cloudCoverageFor(cover, type), last) << type;
      last = cloudCoverageFor(cover, type);
    }
  }
  // a type out of range clamps to the nearest, never a read past the table
  EXPECT_EQ(cloudCoverageFor(0.5F, 99), cloudCoverageFor(0.5F, CLOUD_TYPE_COUNT - 1));
  EXPECT_EQ(cloudCoverageFor(0.5F, -1), cloudCoverageFor(0.5F, CLOUD_TYPE_CUMULUS));
  EXPECT_EQ(cloudSunlitAir(0.0F), 1.0F);
  EXPECT_EQ(cloudSunlitAir(1.0F), 0.0F);
}

// The stratiform types are single decks, low and flat: one band of the sky marched, no gap to step
// over and no high layer, the billows marking stratus least, and Thickness their depth.
TEST(SkyClouds, EachTypeIsTheCloudItSays)
{
  const SkyCloudsParameters cu = cloudDefaultParameters(0.5F, cloudDefaultThickness(CLOUD_TYPE_CUMULUS), CLOUD_TYPE_CUMULUS);
  const SkyCloudsParameters sc =
      cloudDefaultParameters(0.5F, cloudDefaultThickness(CLOUD_TYPE_STRATOCUMULUS), CLOUD_TYPE_STRATOCUMULUS);
  const SkyCloudsParameters st = cloudDefaultParameters(0.5F, cloudDefaultThickness(CLOUD_TYPE_STRATUS), CLOUD_TYPE_STRATUS);
  EXPECT_FLOAT_EQ(cloudDefaultThickness(CLOUD_TYPE_CUMULUS), CLOUD_DEFAULT_THICKNESS);

  for(const SkyCloudsParameters* p : {&sc, &st})
  {
    EXPECT_FLOAT_EQ(p->minLayerHeights.x, p->minHeight);
    EXPECT_FLOAT_EQ(p->maxLayerHeights.x, p->maxHeight);
    EXPECT_FLOAT_EQ(p->shadowTopHeight, p->maxHeight);
    EXPECT_EQ(p->densityScales.z, 0.0F);  // no high layer
    EXPECT_EQ(p->densityScales.w, 0.0F);
    for(float h = p->minHeight; h <= p->maxHeight; h += 0.05F)
      EXPECT_FALSE(cloudInsideIntervals(*p, h)) << h;
  }
  // low: stratus under 1 km, stratocumulus under 2, both below cumulus's tops
  EXPECT_LT(st.maxHeight, 1.0F);
  EXPECT_LT(st.minHeight, sc.minHeight);
  EXPECT_LT(sc.maxHeight, 2.0F);
  EXPECT_LT(sc.maxHeight, cu.shadowTopHeight);
  // flat: the billows cut cumulus most and stratus least, and stratus has no detail to erode it
  EXPECT_GT(cu.shapeAmounts.x, sc.shapeAmounts.x);
  EXPECT_GT(sc.shapeAmounts.x, st.shapeAmounts.x);
  EXPECT_EQ(st.shapeDetailAmounts.x, 0.0F);
  EXPECT_GT(st.coverageFilterWidths.x, cu.coverageFilterWidths.x);
  EXPECT_LT(st.turbulenceDisplacements.x, cu.turbulenceDisplacements.x);

  // Thickness is each deck's depth from its base
  const SkyCloudsParameters deepSt = cloudDefaultParameters(0.5F, 1.0F, CLOUD_TYPE_STRATUS);
  EXPECT_FLOAT_EQ(deepSt.minHeight, st.minHeight);
  EXPECT_NEAR(deepSt.maxHeight - deepSt.minHeight, 1.0F, 1e-5F);
  // and the cover means the same under each: takram's coverage differs by type
  EXPECT_NE(cloudCoverageFor(0.5F, CLOUD_TYPE_STRATUS), cloudCoverageFor(0.5F, CLOUD_TYPE_CUMULUS));

  // Altocumulus: a mid-level deck in the fourth layer alone, of cloudlets smaller than takram's billows
  // that the billows cut everywhere (a shape amount past 1), shallower than the low decks; every other
  // type at takram's scale
  const SkyCloudsParameters ac =
      cloudDefaultParameters(0.5F, cloudDefaultThickness(CLOUD_TYPE_ALTOCUMULUS), CLOUD_TYPE_ALTOCUMULUS);
  EXPECT_GE(ac.minHeight, 2.0F);
  EXPECT_LE(ac.maxHeight, 6.0F);
  EXPECT_FLOAT_EQ(ac.minLayerHeights.w, ac.minHeight);
  EXPECT_FLOAT_EQ(ac.maxLayerHeights.w, ac.maxHeight);
  EXPECT_GT(ac.densityScales.w, 0.0F);
  EXPECT_EQ(ac.densityScales.x + ac.densityScales.y + ac.densityScales.z, 0.0F);
  EXPECT_LT(cloudDefaultThickness(CLOUD_TYPE_ALTOCUMULUS), cloudDefaultThickness(CLOUD_TYPE_STRATOCUMULUS));
  EXPECT_GT(ac.shapeAmounts.w, 1.0F);
  EXPECT_GT(ac.shapeScales.w, 1.0F);
  for(const SkyCloudsParameters* p : {&cu, &sc, &st})
    EXPECT_EQ(p->shapeScales.x, 1.0F);
  for(float h = ac.minHeight; h <= ac.maxHeight; h += 0.05F)
    EXPECT_FALSE(cloudInsideIntervals(ac, h)) << h;
  // its channel of the weather map has patches and clear sky both
  int patches = 0, clear = 0;
  for(int i = 0; i < 400; ++i)
  {
    const float a = cloudWeatherNoise(float2((float(i % 20) + 0.5F) / 20.0F, (float(i / 20) + 0.5F) / 20.0F)).w;
    patches += a > 0.5F ? 1 : 0;
    clear += a < 0.05F ? 1 : 0;
  }
  EXPECT_GT(patches, 40);
  EXPECT_GT(clear, 40);

  // Cirrus: a thin sheet high above the others in the third layer alone, a veil rather than a deck
  const SkyCloudsParameters ci = cloudDefaultParameters(0.5F, cloudDefaultThickness(CLOUD_TYPE_CIRRUS), CLOUD_TYPE_CIRRUS);
  EXPECT_GE(ci.minHeight, 6.0F);
  EXPECT_FLOAT_EQ(ci.minLayerHeights.z, ci.minHeight);
  EXPECT_FLOAT_EQ(ci.maxLayerHeights.z, ci.maxHeight);
  EXPECT_GT(ci.densityScales.z, 0.0F);
  EXPECT_EQ(ci.densityScales.x + ci.densityScales.y + ci.densityScales.w, 0.0F);
  EXPECT_LT(ci.densityScales.z, ac.densityScales.w / 4.0F);
  EXPECT_EQ(ci.turbulenceDisplacements.z, 0.0F);
  EXPECT_LT(ci.shapeStretches.z, 0.5F);  // drawn out into strands
  EXPECT_EQ(ci.powderScales.z, 0.0F);    // a veil, all thin edge
  EXPECT_GT(ci.shapeAmounts.z, 1.0F);
  for(const SkyCloudsParameters* p : {&cu, &sc, &st, &ac})
    for(int i = 0; i < 4; ++i)
      if(p->densityScales[i] > 0.0F)
        EXPECT_EQ(p->shapeStretches[i], 1.0F);
  EXPECT_GT(CLOUD_CIRRUS_SHADE, 0.0F);
  EXPECT_LT(CLOUD_CIRRUS_SHADE, 0.8F);  // a veil: the decks shade about all they cover
}

// The tiers lie over one another: each in its own layers with its own coverage, the march's span
// from the lowest base to the highest top with the gaps between them stepped over, the ground's
// light on the lowest tier's bases, and a tier at no cover not there at all.
TEST(SkyClouds, TheTiersLieOverOneAnother)
{
  const SkyCloudsParameters all = cloudSkyParameters(0.5F, CLOUD_DEFAULT_THICKNESS, CLOUD_TYPE_CUMULUS, 0.3F, 0.7F);
  const SkyCloudsParameters cu  = cloudDefaultParameters(0.5F, CLOUD_DEFAULT_THICKNESS, CLOUD_TYPE_CUMULUS);
  const SkyCloudsParameters ac  = cloudDefaultParameters(0.3F, 0.0F, CLOUD_TYPE_ALTOCUMULUS);
  const SkyCloudsParameters ci  = cloudDefaultParameters(0.7F, 0.0F, CLOUD_TYPE_CIRRUS);
  // each tier as it is alone, in its own layers
  EXPECT_EQ(all.coverages.x, cu.coverages.x);
  EXPECT_EQ(all.coverages.y, cu.coverages.y);
  EXPECT_EQ(all.coverages.w, ac.coverages.w);
  EXPECT_EQ(all.coverages.z, ci.coverages.z);
  for(int i = 0; i < 4; ++i)
  {
    const SkyCloudsParameters& alone = i < 2 ? cu : i == 2 ? ci : ac;
    EXPECT_EQ(all.minLayerHeights[i], alone.minLayerHeights[i]) << i;
    EXPECT_EQ(all.maxLayerHeights[i], alone.maxLayerHeights[i]) << i;
    EXPECT_EQ(all.densityScales[i], alone.densityScales[i]) << i;
    EXPECT_EQ(all.shapeAmounts[i], alone.shapeAmounts[i]) << i;
    EXPECT_EQ(all.shapeScales[i], alone.shapeScales[i]) << i;
    EXPECT_EQ(all.shapeStretches[i], alone.shapeStretches[i]) << i;
    EXPECT_EQ(all.turbulenceDisplacements[i], alone.turbulenceDisplacements[i]) << i;
  }
  // marched from the cumulus bases to the cirrus tops, over the two gaps between the tiers
  EXPECT_FLOAT_EQ(all.minHeight, cu.minHeight);
  EXPECT_FLOAT_EQ(all.maxHeight, ci.maxHeight);
  EXPECT_FLOAT_EQ(all.shadowTopHeight, cu.maxHeight);
  EXPECT_TRUE(cloudInsideIntervals(all, 0.5F * (cu.maxHeight + ac.minHeight)));
  EXPECT_TRUE(cloudInsideIntervals(all, 0.5F * (ac.maxHeight + ci.minHeight)));
  for(const SkyCloudsParameters* t : {&cu, &ac, &ci})
    for(float h = t->minHeight + 0.01F; h < t->maxHeight; h += 0.05F)
      EXPECT_FALSE(cloudInsideIntervals(all, h)) << h;
  // a deep cumulus up into the mid tier makes one span of the two: no gap between them
  const SkyCloudsParameters tall = cloudSkyParameters(0.5F, 4.0F, CLOUD_TYPE_CUMULUS, 0.3F, 0.0F);
  EXPECT_GT(tall.maxLayerHeights.y, tall.minLayerHeights.w);
  for(float h = tall.minHeight; h < tall.maxHeight; h += 0.05F)
    EXPECT_FALSE(cloudInsideIntervals(tall, h)) << h;
  EXPECT_FLOAT_EQ(tall.maxHeight, std::max(tall.maxLayerHeights.y, tall.maxLayerHeights.w));
  // a tier at no cover is not there
  const SkyCloudsParameters high = cloudSkyParameters(0.0F, CLOUD_DEFAULT_THICKNESS, CLOUD_TYPE_CUMULUS, 0.0F, 0.4F);
  EXPECT_EQ(high.densityScales.x + high.densityScales.y + high.densityScales.w, 0.0F);
  EXPECT_FLOAT_EQ(high.minHeight, ci.minHeight);
  EXPECT_FLOAT_EQ(high.shadowTopHeight, ci.maxHeight);
  const SkyCloudsParameters none = cloudSkyParameters(0.0F, CLOUD_DEFAULT_THICKNESS, CLOUD_TYPE_CUMULUS, 0.0F, 0.0F);
  EXPECT_EQ(none.densityScales.x + none.densityScales.y + none.densityScales.z + none.densityScales.w, 0.0F);
  // the sun reaches the land where every tier leaves it, cirrus shading little of what it covers
  EXPECT_FLOAT_EQ(cloudSunlitAirOfTiers(0.5F, 0.0F, 0.0F), cloudSunlitAir(0.5F));
  EXPECT_FLOAT_EQ(cloudSunlitAirOfTiers(0.5F, 0.5F, 0.0F), 0.25F);
  EXPECT_FLOAT_EQ(cloudSunlitAirOfTiers(0.0F, 0.0F, 1.0F), 1.0F - CLOUD_CIRRUS_SHADE);
  EXPECT_GT(cloudSunlitAirOfTiers(0.0F, 0.0F, 1.0F), cloudSunlitAirOfTiers(1.0F, 0.0F, 0.0F));  // a deck shuts it all out
  EXPECT_LT(cloudSunlitAirOfTiers(0.5F, 0.5F, 1.0F), cloudSunlitAirOfTiers(0.5F, 0.5F, 0.0F));
  // and the low kind out of the low tier's range is one of its own, never another tier
  EXPECT_EQ(cloudSkyParameters(0.5F, 1.0F, CLOUD_TYPE_CIRRUS, 0.0F, 0.0F).densityScales.z, 0.0F);
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

// Coverage is the share of the land a tier shades with the sun overhead (cirrus: veils): what the
// panel says, and what the bake assumes for the distant ground and haze (cloudSunlitAirOfTiers).
// Other columns than the table was fitted on.
TEST(SkyClouds, TheCoverShadesAsMuchOfTheLandAsItSays)
{
  for(int type = 0; type < CLOUD_TYPE_COUNT; ++type)
  {
    const SkyCloudsParameters p    = cloudDefaultParameters(0.0F, cloudDefaultThickness(type), type);
    const std::vector<Column> cols = columns(600, 11, p);
    for(const float cover : {0.02F, 0.1F, 0.3F, 0.5F, 0.7F, 0.9F, 1.0F})
    {
      const double share = coverShare(shadeAt(cols, p, cloudCoverageFor(cover, type)), type);
      std::printf("[ type %d cover %.2f ] covers %.3f of the land\n", type, cover, share);
      EXPECT_NEAR(share, 1.0 - cloudSunlitAir(cover), 0.06) << type << " " << cover;
    }
  }
}

// Not a test: what CLOUD_COVERAGE_FOR_COVER is fitted from. For each kind alone, the share its cover
// means (coverShare) at takram's coverage from 0 to 1, inverted at every 5% of cover: a row of the
// table, to paste when the model changes. Its first knot is where the share passes 0.2% (the first
// clouds), its last where it passes 99.8% (the deck closed). For cirrus, too, the share of what it
// covers that it shades at full cover: CLOUD_CIRRUS_SHADE. Other columns than the test above checks.
TEST(SkyClouds, DISABLED_CoverageCurve)
{
  constexpr int kSteps = 400;
  for(int type = 0; type < CLOUD_TYPE_COUNT; ++type)
  {
    const SkyCloudsParameters p    = cloudDefaultParameters(0.0F, cloudDefaultThickness(type), type);
    const std::vector<Column> cols = columns(2500, 5, p);
    std::vector<double>       share(kSteps + 1);
    for(int k = 0; k <= kSteps; ++k)
      share[k] = coverShare(shadeAt(cols, p, float(k) / float(kSteps)), type);
    // the coverage where the share first reaches `target`, between the grid's points
    const auto coverageAt = [&](double target) {
      for(int k = 1; k <= kSteps; ++k)
        if(share[k] >= target)
        {
          const double t = (target - share[k - 1]) / std::max(share[k] - share[k - 1], 1e-9);
          return (double(k - 1) + std::clamp(t, 0.0, 1.0)) / double(kSteps);
        }
      return 1.0;
    };
    std::printf("type %d: {", type);
    for(int knot = 0; knot <= 20; ++knot)
    {
      const double target = knot == 0 ? 0.002 : knot == 20 ? 0.998 : 0.05 * knot;
      std::printf("%.4fF%s", coverageAt(target), knot < 20 ? ", " : "}\n");
    }
    if(type == CLOUD_TYPE_CIRRUS)
    {
      const Shade full = shadeAt(cols, p, float(coverageAt(0.998)));
      std::printf("cirrus at full cover shades %.4f of what it covers (CLOUD_CIRRUS_SHADE)\n", full.shaded / std::max(full.veiled, 1e-9));
    }
  }
}
