// The physical sky's clouds (shaders/sky_clouds.h.slang and sky_clouds_noise.h.slang, takram's model)
// compiled as C++: a clear sky comes back exactly, the cloud image's mapping inverts and crowds its
// rows at the horizon, the view image's inverts and covers a camera's frame a texel a pixel, the layers are where they should be from every observer, the coverage runs
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
#include "../src/sky_cloud_view.hpp"
#include "../src/sky_visibility.hpp"

#include <glm/gtc/matrix_transform.hpp>

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

// The turbulence map at a point of its tile, as the noise pass writes it: the curl and the variety.
float4 turbulenceMap(float u, float v)
{
  const float2 p(wrap01(u), wrap01(v));
  return float4(cloudTurbulenceNoise(p), cloudVarietyNoise(p));
}

// One column straight up from the ground, with what does not depend on the coverage read once: the
// weather map over it and the two volumes at each height, as the bake reads them (sky_clouds_bake.slang
// mediaAt) -- at the first layer's scale and at the fourth's -- the noise evaluated directly instead of
// from its textures, the land's variety and its bends included. The heights are the spans the march
// marches, the gaps between them left out. Without the turbulence, which moves the billows about but
// not how many there are.
struct Column
{
  float4              weather;
  CloudVariety        variety;
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
    const float  x  = spread(rng), z = spread(rng);
    const float2 xz = float2(x, z), sys = cloudSystemUv(xz), reg = cloudRegionUv(xz), loc = cloudLocalUv(xz);
    c.variety       = cloudVariety(turbulenceMap(sys.x, sys.y), turbulenceMap(reg.x, reg.y), turbulenceMap(loc.x, loc.y));
    c.weather       = cloudWeatherNoise(float2(wrap01((x + c.variety.weatherWarp.x) / CLOUD_WEATHER_PERIOD),
                                               wrap01((z + c.variety.weatherWarp.y) / CLOUD_WEATHER_PERIOD)));
    const auto band = [&](float bottom, float top, int n) {
      for(int i = 0; i < n; ++i)
      {
        const float  h = bottom + (float(i) + 0.5F) / float(n) * (top - bottom);
        const float3 q(x + c.variety.shapeWarp.x, h, z + c.variety.shapeWarp.y);
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
      const CloudWeather w = cloudSampleVariedWeather(p, c.weather, c.heights[i], c.variety);
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

TEST(SkyClouds, TheShadeOfTheAirSeenThroughThemNeverTakesTheSkyPastNothing)
{
  // A gap's texel takes off what the clouds' shade takes from the clear sky's air (sky_clouds_air.h.slang)
  const float4 gap(-30.0F, -20.0F, -10.0F, 1.0F);
  EXPECT_FLOAT_EQ(composeClouds(float3(100.0F), gap).x, 70.0F);
  // and a coarser read of the sky than the bake's own may hold less than it takes
  const float3 seen = composeClouds(float3(5.0F), gap);
  EXPECT_EQ(seen.x, 0.0F);
  EXPECT_EQ(seen.z, 0.0F);
}

TEST(SkyClouds, TheShadedAirRunsToTheCloudsTopOrToTheGround)
{
  // The shell sky_clouds_air.h.slang shades: from the ground to the top of the highest layer, here a
  // stratus deck's 0.66 km, seen from 0.3 km up.
  const float top = kEarth + 0.66F;
  // level: out to where the ray climbs to the top, some 68 km off
  const float2 level = cloudLayerSegment(kObserver, 0.0F, kEarth, top);
  EXPECT_EQ(level.x, 0.0F);
  EXPECT_NEAR(level.y, std::sqrt(top * top - kObserver * kObserver), 1e-2F);
  EXPECT_GT(level.y, 60.0F);
  // up steeply: the deck's depth above the eye, a fraction of a kilometre
  const float2 up = cloudLayerSegment(kObserver, 0.8F, kEarth, top);
  EXPECT_NEAR(up.y, 0.36F / 0.8F, 1e-2F);
  // down: to the ground, the whole of the air in front of it
  const float2 down = cloudLayerSegment(kObserver, -0.1F, kEarth, top);
  EXPECT_EQ(down.x, 0.0F);
  EXPECT_NEAR(down.y, 3.0F, 0.05F);
  // from above the deck, from where the ray comes down into it
  const float2 above = cloudLayerSegment(kEarth + 2.0F, -0.5F, kEarth, top);
  EXPECT_NEAR(above.x, (2.0F - 0.66F) / 0.5F, 0.05F);
  EXPECT_NEAR(above.y, 2.0F / 0.5F, 0.05F);
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

// From above the clouds' bases an image holds what is under its horizon too: the clouds beside the
// camera and below it, which it had no rows for (the sky cut off in a line at the horizon, and the
// land crisp from inside a cloud).
TEST(SkyClouds, AWholeImageRunsFromStraightDownToTheZenith)
{
  const float minElevation = horizonElevation(kEarth + 1.2F);
  for(float u = 0.03F; u < 1.0F; u += 0.11F)
    for(float v = 0.01F; v < 1.0F; v += 0.045F)
    {
      const float3 dir = cloudImageDir(float2(u, v), minElevation, true);
      EXPECT_NEAR(length(dir), 1.0F, 1e-5F);
      const float2 back = cloudImageUv(dir, minElevation, true);
      EXPECT_NEAR(back.x, u, 1e-4F);
      EXPECT_NEAR(back.y, v, 2e-4F);
      EXPECT_NEAR(std::atan2(dir.z, dir.x), 2.0F * kPi * u - kPi, 1e-4F);
    }
  // straight down, the horizon at the middle row, the zenith
  EXPECT_NEAR(cloudImageDir(float2(0.5F, 0.0F), minElevation, true).y, -1.0F, 1e-6F);
  EXPECT_NEAR(std::asin(cloudImageDir(float2(0.5F, 0.5F), minElevation, true).y), minElevation, 1e-5F);
  EXPECT_NEAR(cloudImageDir(float2(0.5F, 1.0F), minElevation, true).y, 1.0F, 1e-6F);
  // the rows crowd at the horizon from both sides: a degree there is more of them than one halfway up or down
  const auto rowsPerDegree = [&](float elevationDegrees) {
    const float e  = elevationDegrees * kPi / 180.0F;
    const float de = 0.5F * kPi / 180.0F;
    return std::abs(cloudImageUv(float3(std::cos(e + de), std::sin(e + de), 0.0F), minElevation, true).y
                    - cloudImageUv(float3(std::cos(e - de), std::sin(e - de), 0.0F), minElevation, true).y);
  };
  EXPECT_GT(rowsPerDegree(minElevation * 180.0F / kPi + 2.0F), 3.0F * rowsPerDegree(45.0F));
  EXPECT_GT(rowsPerDegree(minElevation * 180.0F / kPi - 2.0F), 3.0F * rowsPerDegree(-45.0F));

  // which images are whole: those baked from above the lowest base, when there is one
  EXPECT_FALSE(cloudImageWhole(0.6F, 0.75F));
  EXPECT_TRUE(cloudImageWhole(0.76F, 0.75F));
  EXPECT_FALSE(cloudImageWhole(5.0F, 0.0F));
}

// What a whole image's clouds hide of what the camera sees past them.
TEST(SkyClouds, CloudsAreInFrontOfTheLandUnderThemAndOfWhatIsPastThemAmongThem)
{
  const float base = 0.75F;
  // land under the bases, seen from above them: every cloud of the ray is in front of it, whatever
  // depth the image's coarse depth gives the texel
  EXPECT_EQ(cloudShareInFront(1.5F, 0.4F, 0.31F, base), 1.0F);
  EXPECT_EQ(cloudShareInFront(1.5F, 40.0F, 0.31F, base), 1.0F);
  // up among the clouds: a thing nearer than they are is clear of them, one past them behind them
  EXPECT_EQ(cloudShareInFront(0.2F, 2.0F, 1.0F, base), 0.0F);
  EXPECT_EQ(cloudShareInFront(5.0F, 2.0F, 1.0F, base), 1.0F);
  EXPECT_GT(cloudShareInFront(2.0F, 2.0F, 1.0F, base), 0.0F);
  EXPECT_LT(cloudShareInFront(2.0F, 2.0F, 1.0F, base), 1.0F);
  // the camera in the cloud itself, which is no distance off: fog over everything
  EXPECT_EQ(cloudShareInFront(0.05F, 1e-3F, 1.0F, base), 1.0F);

  // a ray down from above the bases ends at them: the march never looks for cloud in the ground
  const float  top   = kEarth + 1.6F;
  const float2 above = cloudLayerSegment(kEarth + 2.0F, -0.8F, kEarth + base, top);
  EXPECT_NEAR(above.x, 0.4F / 0.8F, 0.01F);
  EXPECT_NEAR(above.y, 1.25F / 0.8F, 0.01F);
  const float2 inside = cloudLayerSegment(kEarth + 1.0F, -0.8F, kEarth + base, top);
  EXPECT_EQ(inside.x, 0.0F);
  EXPECT_NEAR(inside.y, 0.25F / 0.8F, 0.01F);
}

TEST(SkyClouds, TheViewImageMappingInverts)
{
  const float3 forward = normalize(float3(0.3F, 0.4F, -0.8F));
  const float3 right   = normalize(cross(forward, float3(0.0F, 1.0F, 0.0F)));
  const float3 up      = cross(right, forward);
  for(float u = 0.01F; u < 1.0F; u += 0.07F)
    for(float v = 0.02F; v < 1.0F; v += 0.09F)
    {
      const float3 dir = cloudViewDir(float2(u, v), right, 0.31F, up, 0.2F, forward);
      EXPECT_NEAR(length(dir), 1.0F, 1e-5F);
      const float3 back = cloudViewUv(dir, right, 0.31F, up, 0.2F, forward);
      EXPECT_NEAR(back.x, u, 1e-4F);
      EXPECT_NEAR(back.y, v, 1e-4F);
      EXPECT_NEAR(back.z, std::max(std::abs(2.0F * u - 1.0F), std::abs(2.0F * v - 1.0F)), 1e-4F);
    }
  // v runs down: the top row looks up
  EXPECT_GT(dot(cloudViewDir(float2(0.5F, 0.0F), right, 0.31F, up, 0.2F, forward), up), 0.0F);
  // behind the image, and beside it, the sky image alone; inside, the view image alone, fading
  // between over the outer part of it
  EXPECT_EQ(cloudViewShare(cloudViewUv(-forward, right, 0.31F, up, 0.2F, forward).z), 0.0F);
  EXPECT_EQ(cloudViewShare(cloudViewUv(normalize(forward + right), right, 0.31F, up, 0.2F, forward).z), 0.0F);
  EXPECT_EQ(cloudViewShare(0.5F), 1.0F);
  EXPECT_EQ(cloudViewShare(1.0F - CLOUD_VIEW_FADE), 1.0F);
  EXPECT_NEAR(cloudViewShare(1.0F - 0.5F * CLOUD_VIEW_FADE), 0.5F, 1e-4F);
}

TEST(SkyClouds, TheViewImageCoversTheCamerasFrameATexelAPixel)
{
  // An 85 mm lens (16 degrees down a full frame's 24 mm) on a 2160 x 1395 viewport, looking past a
  // figure and up a little, under an environment turned 40 degrees; the projection flipped as
  // Vulkan's is.
  const glm::uvec2 frame(2160, 1395);
  glm::mat4        projection = glm::perspective(glm::radians(16.0F), float(frame.x) / float(frame.y), 0.1F, 1000.0F);
  projection[1][1] *= -1.0F;
  const glm::mat4 view = glm::lookAt(glm::vec3(0.3F, 1.5F, 3.0F), glm::vec3(0.0F, 1.9F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
  const glm::quat env  = glm::angleAxis(glm::radians(40.0F), glm::vec3(0.0F, 1.0F, 0.0F));
  const std::optional<SkyCloudView> image = cloudViewFor(view, projection, env, frame);
  ASSERT_TRUE(image.has_value());
  EXPECT_LE(image->size.x, uint32_t(CLOUD_VIEW_CAPACITY));
  EXPECT_LE(image->size.y, uint32_t(CLOUD_VIEW_CAPACITY));

  // A pixel's ray as getRay aims it (camera_ray.h.slang), in the environment's frame, as the path
  // tracer looks it up, and where it falls in the view image's texels.
  const glm::mat4 projectionInverse = glm::inverse(projection);
  const glm::mat4 viewInverse       = glm::inverse(view);
  const auto      texelOf           = [&](float x, float y) {
    const glm::vec2 clip = glm::vec2(x, y) / glm::vec2(frame) * 2.0F - 1.0F;
    glm::vec4       p    = projectionInverse * glm::vec4(clip, -1.0F, 1.0F);
    p /= p.w;
    const glm::vec3 world = glm::normalize(glm::vec3(viewInverse * p) - glm::vec3(viewInverse[3]));
    const glm::vec3 dir   = glm::conjugate(env) * world;
    const float3    uv    = cloudViewUv(dir, float3(image->right), image->right.w, float3(image->up), image->up.w, image->forward);
    return glm::vec3(uv.x * float(image->size.x), uv.y * float(image->size.y), uv.z);
  };
  // every part of the frame, its corners too, in the view image wholly
  for(float x = 0.0F; x <= float(frame.x); x += float(frame.x) / 8.0F)
    for(float y = 0.0F; y <= float(frame.y); y += float(frame.y) / 8.0F)
    {
      const glm::vec3 t = texelOf(x, y);
      EXPECT_LT(t.z, 1.0F / (1.0F + kCloudViewMargin) + 1e-3F) << x << ", " << y;
      EXPECT_EQ(cloudViewShare(t.z), 1.0F) << x << ", " << y;
    }
  // about a texel a pixel, here a little under (the image's capacity), against the sky image's 7
  for(const glm::vec2 at : {glm::vec2(1080.0F, 697.0F), glm::vec2(2.0F, 2.0F), glm::vec2(2150.0F, 1390.0F)})
  {
    const float across = glm::length(glm::vec2(texelOf(at.x + 1.0F, at.y)) - glm::vec2(texelOf(at.x, at.y)));
    const float down   = glm::length(glm::vec2(texelOf(at.x, at.y + 1.0F)) - glm::vec2(texelOf(at.x, at.y)));
    EXPECT_GT(across, 0.8F);
    EXPECT_LT(across, 1.0F);
    EXPECT_GT(down, 0.8F);
    EXPECT_LT(down, 1.0F);
  }
  // the sky image's columns at this camera: 0.088 degrees, against a pixel's 16 / 1395
  EXPECT_GT(360.0F / 4096.0F / (16.0F / float(frame.y)), 7.0F);

  // A camera that has not moved asks for the same image; one turned a hair, for another.
  EXPECT_TRUE(cloudViewFor(view, projection, env, frame) == image);
  const glm::mat4 turned = glm::rotate(view, 1e-4F, glm::vec3(0.0F, 1.0F, 0.0F));
  EXPECT_FALSE(cloudViewFor(turned, projection, env, frame) == image);
}

TEST(SkyClouds, TheViewImageIsSeenFromWhereTheCameraStands)
{
  const glm::uvec2 frame(1920, 1080);
  const glm::mat4  projection = glm::perspective(glm::radians(40.0F), float(frame.x) / float(frame.y), 0.1F, 1000.0F);
  const auto       viewFrom   = [](glm::vec3 eye) { return glm::lookAt(eye, eye + glm::vec3(0.0F, 0.2F, -1.0F), glm::vec3(0.0F, 1.0F, 0.0F)); };
  const glm::quat  none(1.0F, 0.0F, 0.0F, 0.0F);

  // At the scene's origin the camera is the sky bake's observer, a head's height above it
  const std::optional<SkyCloudView> home = cloudViewFor(viewFrom({0.0F, 1.6F, 0.0F}), projection, none, frame, 0.3F);
  ASSERT_TRUE(home.has_value());
  EXPECT_NEAR(home->eye.land.x, 0.0F, 1e-6F);
  EXPECT_NEAR(home->eye.land.y, 0.0F, 1e-6F);
  EXPECT_NEAR(home->eye.altitude, 0.3016F, 1e-5F);

  // 300 m east, 400 m south and 50 m up: that far on over the land, in kilometres, and as much higher
  const std::optional<SkyCloudView> away = cloudViewFor(viewFrom({300.0F, 51.6F, 400.0F}), projection, none, frame, 0.3F);
  ASSERT_TRUE(away.has_value());
  EXPECT_NEAR(away->eye.land.x, 0.3F, 1e-5F);
  EXPECT_NEAR(away->eye.land.y, 0.4F, 1e-5F);
  EXPECT_NEAR(away->eye.altitude, 0.3516F, 1e-5F);
  EXPECT_TRUE(away->eye == cloudEyeOf(viewFrom({300.0F, 51.6F, 400.0F}), none, 0.3F));
  // it looks the same way, and is another image
  EXPECT_NEAR(glm::length(away->forward - home->forward), 0.0F, 1e-5F);
  EXPECT_FALSE(away == home);
  EXPECT_NEAR(cloudAnchorDistance(home->eye, away->eye), std::sqrt(0.3F * 0.3F + 0.4F * 0.4F + 0.05F * 0.05F), 1e-5F);
  const glm::vec3 from = cloudEyeFrom(home->eye, away->eye);
  EXPECT_NEAR(from.x, 0.3F, 1e-5F);
  EXPECT_NEAR(from.y, 0.05F, 1e-5F);
  EXPECT_NEAR(from.z, 0.4F, 1e-5F);

  // The land is the environment's: under a sky turned a quarter round, the world's x is its z
  const glm::quat      quarter = glm::angleAxis(glm::radians(90.0F), glm::vec3(0.0F, 1.0F, 0.0F));
  const SkyCloudAnchor turned  = cloudEyeOf(viewFrom({300.0F, 1.6F, 0.0F}), quarter, 0.3F);
  EXPECT_NEAR(turned.land.x, 0.0F, 1e-5F);
  EXPECT_NEAR(std::abs(turned.land.y), 0.3F, 1e-5F);

  // A step to the side is another image: the clouds are marched again from there
  EXPECT_FALSE(cloudViewFor(viewFrom({0.05F, 1.6F, 0.0F}), projection, none, frame, 0.3F) == home);
  // A camera under its scene's floor, on a planet's surface, still looks from above ground
  EXPECT_EQ(cloudEyeOf(viewFrom({0.0F, -40.0F, 0.0F}), none, 0.0F).altitude, kCloudViewMinAltitude);
}

// A cloud is a point in the world. An image baked from one place holds it in the direction that
// place saw it; a camera elsewhere finds it there by how far along its own ray the cloud is.
TEST(SkyClouds, AnImageBakedElsewhereIsReadWhereItsAnchorSawTheCloud)
{
  // a cloud base 750 m over a spot 400 m north of the anchor
  const float3 cloud(0.0F, 0.75F, -0.4F);
  // the camera 10 m east of the anchor and 2 m up, looking at it
  const float3 eye(0.010F, 0.002F, 0.0F);
  const float  depth = length(cloud - eye);
  const float3 dir   = (cloud - eye) / depth;
  // the image holds it where the anchor saw it
  const float3 look = cloudDirFromAnchor(dir, eye, depth);
  EXPECT_NEAR(length(look - normalize(cloud)), 0.0F, 1e-6F);
  // which is not where the camera looks: 0.67 degrees off, eight of the sky image's columns
  const float off = degrees(acos(dot(look, dir)));
  EXPECT_NEAR(off, 0.67F, 0.02F);
  EXPECT_GT(off, 7.0F * 360.0F / 4096.0F);
  // from the anchor itself the image is read where the camera looks
  EXPECT_NEAR(length(cloudDirFromAnchor(dir, float3(0.0F), depth) - dir), 0.0F, 1e-6F);

  // The depth image is read first where the camera looks, which gives the anchor's distance to what
  // it saw there; the cloud's own distance along the camera's ray comes from the second read.
  EXPECT_NEAR(cloudDepthAlong(dir, eye, look, length(cloud)), depth, 1e-5F);
  // never behind the camera, whatever the image says
  EXPECT_GT(cloudDepthAlong(dir, eye, -dir, 5.0F), 0.0F);

  // A cloud ten times as far is ten times less off, and a direction with no cloud not at all
  const float3 far     = cloud * 10.0F;
  const float  farAway = length(far - eye);
  const float3 farDir  = (far - eye) / farAway;
  EXPECT_NEAR(degrees(acos(dot(cloudDirFromAnchor(farDir, eye, farAway), farDir))), 0.067F, 0.01F);
  EXPECT_LT(length(cloudDirFromAnchor(dir, eye, CLOUD_NO_DEPTH) - dir), 2e-5F);
}

TEST(SkyClouds, AClearDirectionIsAsFarAsAShellAtTheBasesHeight)
{
  const float shell = kEarth + 0.75F;
  // from below it: straight up the height left, and further toward the horizon
  EXPECT_NEAR(cloudShellDistance(kObserver, 1.0F, shell), 0.45F, 1e-3F);
  EXPECT_GT(cloudShellDistance(kObserver, 0.1F, shell), 4.0F);
  EXPECT_LT(cloudShellDistance(kObserver, 0.1F, shell), 4.6F);
  // from above it: down to it, and nothing at all looking up
  EXPECT_NEAR(cloudShellDistance(kEarth + 2.0F, -1.0F, shell), 1.25F, 1e-3F);
  EXPECT_EQ(cloudShellDistance(kEarth + 2.0F, 0.5F, shell), CLOUD_NO_DEPTH);
}

// The clouds' shadow on the land: where the shadow image is read for a place in the scene.
TEST(SkyClouds, AShadowIsReadWhereTheSunsRayComesUpFromTheGround)
{
  const float3 overhead(0.0F, 1.0F, 0.0F);
  // the scene's origin is the image's middle, and the image's corners the corners of its square
  EXPECT_EQ(cloudShadowUv(float3(0.0F), overhead), float2(0.5F));
  EXPECT_EQ(cloudShadowGround(float2(0.5F)), float2(0.0F));
  EXPECT_EQ(cloudShadowGround(float2(1.0F, 0.0F)), float2(0.5F * CLOUD_SHADOW_EXTENT, -0.5F * CLOUD_SHADOW_EXTENT));
  // a point of the ground reads the texel baked from it
  const float2 ground(1.3F, -2.1F);
  const float2 uv = cloudShadowUv(float3(ground.x, 0.0F, ground.y), normalize(float3(0.4F, 0.6F, -0.2F)));
  EXPECT_NEAR(length(cloudShadowGround(uv) - ground), 0.0F, 1e-5F);

  // A treetop 30 m up with the sun 45 degrees up in the east is in the shadow of the cloud that
  // shades the ground 30 m west of it: the sun's ray through it comes up from there
  const float3 east = normalize(float3(1.0F, 1.0F, 0.0F));
  const float2 top  = cloudShadowGround(cloudShadowUv(float3(0.2F, 0.03F, 0.1F), east));
  EXPECT_NEAR(top.x, 0.17F, 1e-5F);
  EXPECT_NEAR(top.y, 0.1F, 1e-5F);
  // under an overhead sun, the ground under it
  EXPECT_NEAR(length(cloudShadowGround(cloudShadowUv(float3(0.2F, 0.03F, 0.1F), overhead)) - float2(0.2F, 0.1F)), 0.0F, 1e-5F);
  // a sun on the horizon would carry a hill's height out of the image: it counts as no lower than
  // CLOUD_SHADOW_MIN_SUN, three degrees
  const float2 low = cloudShadowGround(cloudShadowUv(float3(0.0F, 0.1F, 0.0F), normalize(float3(1.0F, 0.001F, 0.0F))));
  EXPECT_NEAR(low.x, -0.1F / CLOUD_SHADOW_MIN_SUN, 1e-3F);
  EXPECT_GT(low.x, -0.5F * CLOUD_SHADOW_EXTENT);

  // The image counts whole inside, fades over the outer tenth of its half-width, and past its edge
  // the land's mean share stands in
  EXPECT_EQ(cloudShadowWeight(float2(0.5F)), 1.0F);
  EXPECT_EQ(cloudShadowWeight(float2(0.94F, 0.5F)), 1.0F);
  EXPECT_NEAR(cloudShadowWeight(float2(0.975F, 0.5F)), 0.5F, 1e-4F);
  EXPECT_EQ(cloudShadowWeight(float2(1.0F, 0.5F)), 0.0F);
  EXPECT_EQ(cloudShadowWeight(float2(0.3F, -0.2F)), 0.0F);
  // a 900 m ground with its hills is well inside
  EXPECT_EQ(cloudShadowWeight(cloudShadowUv(float3(0.9F, 0.06F, -0.9F), normalize(float3(0.5F, 0.5F, 0.0F)))), 1.0F);

  // Clouds that drifted 0.3 km east since the bake shade a point with the shadow the image has 0.3 km
  // west of it, whatever the sun: the shadow went with them
  const float3 sun   = normalize(float3(0.4F, 0.6F, -0.2F));
  const float3 point = float3(1.0F, 0.02F, -0.5F);
  const float2 moved = cloudShadowUv(point, sun, float2(0.3F, 0.0F));
  const float2 west  = cloudShadowUv(point - float3(0.3F, 0.0F, 0.0F), sun);
  EXPECT_NEAR(length(moved - west), 0.0F, 1e-6F);
  EXPECT_EQ(cloudShadowUv(point, sun, float2(0.0F)), cloudShadowUv(point, sun));
}

// The wind carries the clouds as a camera flying the other way under them would see them: an anchor
// is a place under the clouds as they have drifted, the follow bakes from where a camera at rest has
// drifted to among them, and the shadow image is baked again once they are far enough on.
TEST(SkyClouds, DriftingCloudsPassOverACameraAtRest)
{
  // From where the wind blows, as the sun's azimuth: from 90 (+x) at 36 km/h, 10 m a second toward -x
  const glm::vec2 v = cloudDriftVelocity(36.0F, 90.0F);
  EXPECT_NEAR(v.x, -0.01F, 1e-7F);
  EXPECT_NEAR(v.y, 0.0F, 1e-7F);
  const glm::vec2 fromNorth = cloudDriftVelocity(36.0F, 0.0F);
  EXPECT_NEAR(fromNorth.x, 0.0F, 1e-7F);
  EXPECT_NEAR(fromNorth.y, -0.01F, 1e-7F);
  EXPECT_EQ(cloudDriftVelocity(0.0F, 45.0F), glm::vec2(0.0F));
  EXPECT_EQ(cloudDriftVelocity(-5.0F, 45.0F), glm::vec2(0.0F));

  // A frame moves them by its length, a stall no further than kCloudDriftLongestFrame
  EXPECT_NEAR(cloudDriftStep(glm::vec2(0.0F), v, 0.5F * kCloudDriftLongestFrame).x, -0.0005F, 1e-7F);
  EXPECT_EQ(cloudDriftStep(glm::vec2(0.0F), v, 30.0F), cloudDriftStep(glm::vec2(0.0F), v, kCloudDriftLongestFrame));
  EXPECT_EQ(cloudDriftStep(glm::vec2(1.0F), v, -1.0F), glm::vec2(1.0F));

  // A camera 300 m east under clouds gone 2 km west stands 2.3 km east among them
  const glm::mat4 view = glm::lookAt(glm::vec3(300.0F, 1.6F, 0.0F), glm::vec3(300.0F, 1.8F, -1.0F), glm::vec3(0.0F, 1.0F, 0.0F));
  const glm::quat none(1.0F, 0.0F, 0.0F, 0.0F);
  const SkyCloudAnchor eye = cloudEyeOf(view, none, 0.3F, glm::vec2(-2.0F, 0.0F));
  EXPECT_NEAR(eye.land.x, 2.3F, 1e-5F);
  EXPECT_NEAR(eye.land.y, 0.0F, 1e-6F);
  const glm::mat4 projection = glm::perspective(glm::radians(40.0F), 16.0F / 9.0F, 0.1F, 1000.0F);
  const std::optional<SkyCloudView> seen = cloudViewFor(view, projection, none, {1920, 1080}, 0.3F, glm::vec2(-2.0F, 0.0F));
  ASSERT_TRUE(seen.has_value());
  EXPECT_TRUE(seen->eye == eye);

  // At rest at the origin under a 10 m/s wind, 60 frames a second: the sky bake's image, shifted, for
  // its first kCloudFollowDistance; then a bake from where the camera has drifted to among the clouds
  const SkyCloudAnchor home{.land = {0.0F, 0.0F}, .altitude = 0.3016F};
  SkyCloudFollow       f;
  f.reset(home);
  glm::vec2 drift(0.0F);
  int       frames = 0;
  bool      baking = false;
  while(!baking && frames < 600)
  {
    drift = cloudDriftStep(drift, v, 1.0F / 60.0F);
    ++frames;
    const SkyCloudAnchor at{.land = -drift, .altitude = home.altitude};
    baking = cloudFollowStep(f, at, 1024, 8).bake;
  }
  EXPECT_TRUE(baking);
  EXPECT_GT(frames, 55);  // just past a second
  EXPECT_LT(frames, 65);
  EXPECT_GT(f.bakingFrom.land.x, kCloudFollowDistance);  // east of the origin among clouds gone west
}

// Among the clouds they are tens of metres from the camera: the image is baked again after fewer
// metres and over fewer frames, and what the camera has gone meanwhile never turns a near cloud
// through more than a few degrees (it smeared across the frame).
TEST(SkyClouds, AmongTheCloudsTheFollowIsSoonerAndTheShiftBounded)
{
  const SkyCloudAnchor origin{.land = {0.0F, 0.0F}, .altitude = 0.3F};
  SkyCloudFollow       f;
  f.reset(origin);
  const auto at = [&](float metres) { return SkyCloudAnchor{.land = {metres * 0.001F, 0.0F}, .altitude = 1.1F}; };
  // put down up there: the jump's bake, whole at once
  EXPECT_TRUE(cloudFollowStep(f, at(0.0F), 1024, 8, true).completes);
  // 5 m on is not yet far for a camera under the clouds, and is for one among them
  SkyCloudFollow under = f;
  EXPECT_FALSE(cloudFollowStep(under, at(5.0F), 1024, 8).bake);
  float x      = 5.0F;
  int   frames = 0;
  for(bool whole = false; !whole; ++frames)
  {
    const SkyCloudFollowStep step = cloudFollowStep(f, at(x), 1024, 8, true);
    ASSERT_TRUE(step.bake);
    EXPECT_EQ(step.rows % 8U, 0U);
    whole = step.completes;
    x += 0.25F;
    ASSERT_LE(frames, 64);
  }
  EXPECT_EQ(frames, int(kCloudFollowFramesAmong));
  EXPECT_LT(kCloudFollowDistanceAmong, kCloudFollowDistance);

  // A cloud 30 m off, the camera 8 m from where the image was baked: shifted by its own depth it
  // would turn through 15 degrees; the shift takes it as no nearer than CLOUD_SHIFT_NEAREST times
  // the camera's distance
  const float3 eye(0.008F, 0.0F, 0.0F);
  const float3 dir(0.0F, 0.0F, 1.0F);
  EXPECT_GT(degrees(acos(dot(cloudDirFromAnchor(dir, eye, 0.03F), dir))), 14.0F);
  EXPECT_LT(degrees(acos(dot(cloudDirFromAnchor(dir, eye, cloudShiftDepth(0.03F, eye)), dir))), 9.6F);
  // a cloud at its usual distance is shifted by that, as before
  EXPECT_EQ(cloudShiftDepth(0.75F, eye), 0.75F);
  EXPECT_EQ(cloudShiftDepth(0.75F, float3(0.0F)), 0.75F);
}

// In a fast flight a bake begins as the last is whole: baked from where it began, the image the frames
// had was up to two bakes' flight behind the camera (22 m among the clouds at 1 m a frame). Baked from
// ahead of where the camera will be, about a quarter of that; and a camera that stops gets an image of
// where it stopped.
TEST(SkyClouds, AFastFlightsImageIsBakedFromWhereTheCameraWillBe)
{
  const SkyCloudAnchor origin{.land = {0.0F, 0.0F}, .altitude = 0.3F};
  SkyCloudFollow       f;
  f.reset(origin);
  const auto at = [&](float metres) { return SkyCloudAnchor{.land = {metres * 0.001F, 0.0F}, .altitude = 1.1F + metres * 0.0002F}; };
  EXPECT_TRUE(cloudFollowStep(f, at(0.0F), 1024, 8, true).completes);  // put down up there
  float x = 0.0F, behind = 0.0F;
  int   whole = 0;
  for(int frame = 0; frame < 400; frame++)
  {
    x += 1.0F;
    whole += cloudFollowStep(f, at(x), 1024, 8, true).completes ? 1 : 0;
    if(frame >= 60)
      behind = std::max(behind, 1000.0F * cloudAnchorDistance(f.anchor, at(x)));
  }
  EXPECT_GT(whole, 25);  // a bake about every dozen frames, as before
  EXPECT_LT(behind, 7.5F);
  // stopped: within a few dozen frames the image is of where the camera is, and no more is baked
  int bakes = 0;
  for(int frame = 0; frame < 120; frame++)
    bakes += cloudFollowStep(f, at(x), 1024, 8, true).bake && frame >= 40 ? 1 : 0;
  EXPECT_EQ(bakes, 0);
  EXPECT_LT(1000.0F * cloudAnchorDistance(f.anchor, at(x)), 1.5F);
  // a camera put down far off has no pace: its image is of where it is
  EXPECT_TRUE(cloudFollowStep(f, at(x + 500.0F), 1024, 8, true).completes);
  EXPECT_LT(1000.0F * cloudAnchorDistance(f.anchor, at(x + 500.0F)), 1e-3F);
}

// The sky image that follows the camera (cloudFollowStep), a 1024-row image baked in bands of 8.
TEST(SkyClouds, TheSkyImageFollowsACameraInFlight)
{
  const SkyCloudAnchor origin{.land = {0.0F, 0.0F}, .altitude = 0.3F};
  SkyCloudFollow       f;
  f.reset(origin);
  const auto at = [&](float metres) { return SkyCloudAnchor{.land = {metres * 0.001F, 0.0F}, .altitude = 0.3016F}; };

  // Round a figure at the origin nothing is baked: the sky bake's image, shifted
  for(float x : {0.0F, 3.0F, -6.0F, 9.0F})
  {
    const SkyCloudFollowStep step = cloudFollowStep(f, at(x), 1024, 8);
    EXPECT_FALSE(step.bake) << x;
    EXPECT_FALSE(step.frontChanged) << x;
  }
  EXPECT_EQ(f.front, 0);

  // A flight at 15 m/s, 60 frames a second, after a moment held still: past 10 m a bake starts, a
  // band of rows each frame, and the frames keep the image they had until it is whole. It is baked
  // from ahead of where the camera will be when it is whole, its pace (a quarter of a metre a frame,
  // smoothed) times the bake's 31 frames after the first on
  for(int i = 0; i < 60; i++)
    cloudFollowStep(f, at(9.0F), 1024, 8);
  float              x = 9.0F;
  SkyCloudFollowStep first;
  for(int i = 0; !first.bake; i++)
  {
    x += 0.25F;
    first = cloudFollowStep(f, at(x), 1024, 8);
    ASSERT_LE(i, 64);
  }
  const float began = first.from.land.x;
  EXPECT_GT(began, at(x).land.x + 31.0F * f.pace.x);
  EXPECT_LT(began, at(x).land.x + kCloudFollowLeadMost);
  uint32_t baked  = first.rows;
  int      frames = 1;
  bool     whole  = first.completes;
  while(!whole)
  {
    x += 0.25F;
    const SkyCloudFollowStep step = cloudFollowStep(f, at(x), 1024, 8);
    ASSERT_TRUE(step.bake);
    EXPECT_EQ(step.target, 1);
    EXPECT_EQ(step.firstRow, baked);
    EXPECT_EQ(step.rows % 8U, 0U);
    EXPECT_EQ(step.from.land.x, began);  // where the bake was from, every band of it
    baked += step.rows;
    ++frames;
    whole = step.completes;
    EXPECT_EQ(step.frontChanged, whole);
    EXPECT_EQ(f.front, whole ? 1 : 0);
    ASSERT_LE(frames, 64);
  }
  EXPECT_EQ(baked, 1024U);
  EXPECT_EQ(frames, int(kCloudFollowFrames));
  EXPECT_EQ(f.anchor.land.x, began);
  // the camera is then short of the image's anchor, and passes it before the next bake
  EXPECT_LT(cloudAnchorDistance(f.anchor, at(x)), kCloudFollowDistance);
  EXPECT_LT(at(x).land.x, f.anchor.land.x);
  EXPECT_FALSE(cloudFollowStep(f, at(x + 0.25F), 1024, 8).bake);
  x += 0.25F;

  // ... and the next one goes into the other follow image, never the one the frames are taking
  SkyCloudFollowStep next;
  for(int i = 0; !next.bake; i++)
  {
    x += 0.25F;
    next = cloudFollowStep(f, at(x), 1024, 8);
    ASSERT_LE(i, 200);
  }
  EXPECT_EQ(next.target, 2);
  EXPECT_EQ(next.firstRow, 0U);
  EXPECT_EQ(f.front, 1);

  // The camera stops as it begins. A frame in which a flight's camera did not move looks no
  // different, so the bake goes on as it was, a band a frame, from where the camera was going; the
  // camera at rest gets that image all the same, then, having no pace now, one from where it stopped
  int bands = 1;
  for(SkyCloudFollowStep step = next; !step.completes; ++bands)
  {
    step = cloudFollowStep(f, at(x), 1024, 8);
    ASSERT_TRUE(step.bake);
    EXPECT_EQ(step.target, 2);
    EXPECT_EQ(step.from.land.x, next.from.land.x);
    ASSERT_LE(bands, 64);
  }
  EXPECT_EQ(bands, int(kCloudFollowFrames));
  EXPECT_EQ(f.front, 2);
  EXPECT_FALSE(f.baking);
  SkyCloudFollowStep last;
  for(int i = 0; !last.completes; i++)
  {
    last = cloudFollowStep(f, at(x), 1024, 8);
    ASSERT_TRUE(last.bake);
    ASSERT_LE(i, 64);
  }
  EXPECT_EQ(last.target, 1);
  EXPECT_LT(cloudAnchorDistance(f.anchor, at(x)), 0.00001F);  // under a centimetre
  // at rest under it, nothing more; nor a few metres on, within reach of the shift
  EXPECT_FALSE(cloudFollowStep(f, at(x), 1024, 8).bake);
  EXPECT_FALSE(cloudFollowStep(f, at(x + 4.0F), 1024, 8).bake);

  // Put down 400 m off (a saved view, a jump to a selection): the shift is no picture of the sky
  // from that far, so the image is baked whole in that frame
  const SkyCloudFollowStep jump = cloudFollowStep(f, at(430.0F), 1024, 8);
  EXPECT_TRUE(jump.bake);
  EXPECT_TRUE(jump.completes);
  EXPECT_TRUE(jump.frontChanged);
  EXPECT_EQ(jump.target, 2);
  EXPECT_EQ(jump.firstRow, 0U);
  EXPECT_EQ(jump.rows, 1024U);
  EXPECT_TRUE(f.anchor == at(430.0F));
  // ... and a jump while a bake is under way bakes the new place, not the rest of the old one
  EXPECT_FALSE(cloudFollowStep(f, at(445.0F), 1024, 8).completes);
  const SkyCloudFollowStep again = cloudFollowStep(f, at(900.0F), 1024, 8);
  EXPECT_TRUE(again.completes);
  EXPECT_EQ(again.firstRow, 0U);
  EXPECT_EQ(again.rows, 1024U);
  EXPECT_TRUE(f.anchor == at(900.0F));

  // Back at the origin the frames take the sky bake's own image again, with nothing baked
  const SkyCloudFollowStep back = cloudFollowStep(f, at(5.0F), 1024, 8);
  EXPECT_FALSE(back.bake);
  EXPECT_TRUE(back.frontChanged);
  EXPECT_EQ(f.front, 0);
  EXPECT_TRUE(f.anchor == origin);
  EXPECT_FALSE(cloudFollowStep(f, at(5.0F), 1024, 8).frontChanged);

  // A climb counts as a flight does: 40 m up over the origin is 40 m from it
  SkyCloudFollow g;
  g.reset(origin);
  EXPECT_TRUE(cloudFollowStep(g, SkyCloudAnchor{.land = {0.0F, 0.0F}, .altitude = 0.34F}, 1024, 8).bake);
  EXPECT_NEAR(g.bakingFrom.altitude, 0.34F, 1e-6F);
  // ... and a sky bake puts the frames back on its image from wherever they were
  g.reset(origin);
  EXPECT_EQ(g.front, 0);
  EXPECT_FALSE(g.baking);
  EXPECT_TRUE(g.anchor == origin);
}

TEST(SkyClouds, AnOrthographicOrVeryWideCameraKeepsTheSkyImage)
{
  const glm::mat4 view = glm::lookAt(glm::vec3(0.0F, 1.5F, 3.0F), glm::vec3(0.0F, 1.5F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
  const glm::quat env(1.0F, 0.0F, 0.0F, 0.0F);
  EXPECT_FALSE(cloudViewFor(view, glm::ortho(-2.0F, 2.0F, -1.0F, 1.0F, 0.1F, 100.0F), env, {1920, 1080}).has_value());
  EXPECT_FALSE(cloudViewFor(view, glm::perspective(glm::radians(120.0F), 1.0F, 0.1F, 100.0F), env, {1080, 1080}).has_value());
  EXPECT_FALSE(cloudViewFor(view, glm::perspective(glm::radians(40.0F), 1.0F, 0.1F, 100.0F), env, {0, 0}).has_value());
  // a 24 mm lens still has one
  EXPECT_TRUE(cloudViewFor(view, glm::perspective(glm::radians(53.0F), 1.5F, 0.1F, 100.0F), env, {1920, 1280}).has_value());
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

// The air's visibility as a setting: the aerosol scaled until the air at the ground has it
TEST(SkyVisibility, TheAerosolIsScaledToTheVisibilityAskedFor)
{
  namespace V = sky_visibility;
  // Earth's own air as the preset has it: a clear day
  const float rayleigh = 0.0135578F, aerosol = 0.06F / 0.9F;
  EXPECT_NEAR(V::of(rayleigh, aerosol), 48.8F, 0.5F);
  // none asked for, or no aerosol to scale: as it is
  EXPECT_FLOAT_EQ(V::aerosolScale(0.0F, rayleigh, aerosol), 1.0F);
  EXPECT_FLOAT_EQ(V::aerosolScale(-3.0F, rayleigh, aerosol), 1.0F);
  EXPECT_FLOAT_EQ(V::aerosolScale(10.0F, rayleigh, 0.0F), 1.0F);
  // asked for what it has: unchanged; and the air scaled has the visibility asked for
  EXPECT_NEAR(V::aerosolScale(V::of(rayleigh, aerosol), rayleigh, aerosol), 1.0F, 1e-4F);
  for(const float km : {3.0F, 10.0F, 20.0F, 100.0F})
  {
    const float scale = V::aerosolScale(km, rayleigh, aerosol);
    EXPECT_NEAR(V::of(rayleigh, aerosol * scale), km, 1e-3F * km) << km;
  }
  // a summer haze of 10 km is some six times the clear day's aerosol
  EXPECT_NEAR(V::aerosolScale(10.0F, rayleigh, aerosol), 5.66F, 0.1F);
  // past what Rayleigh alone allows (about 290 km) there is still a little aerosol; and a fog is not
  // this model's to make
  EXPECT_FLOAT_EQ(V::aerosolScale(1000.0F, rayleigh, aerosol), V::kLeast);
  EXPECT_FLOAT_EQ(V::aerosolScale(0.01F, rayleigh, aerosol), V::kMost);

  // Clearer air is less of the same aerosol; hazier is the atmosphere's own and water haze added,
  // whose extinction gives the visibility asked for and which absorbs next to nothing
  const glm::vec3 ownScatter(0.06F), ownExtinction(aerosol);
  for(const float km : {30.0F, 10.0F, 2.0F, 100.0F})
  {
    const float scale     = V::aerosolScale(km, rayleigh, aerosol);
    glm::vec3   scatter   = ownScatter;
    glm::vec3   extinct   = ownExtinction;
    V::applyAerosolScale(scale, scatter, extinct);
    EXPECT_NEAR(V::of(rayleigh, extinct.y), km, 1e-3F * km) << km;
    const float absorbed  = extinct.y - scatter.y;
    const float ownAbsorb = ownExtinction.y - ownScatter.y;
    if(scale > 1.0F)
    {
      EXPECT_NEAR(absorbed, ownAbsorb + (extinct.y - ownExtinction.y) * (1.0F - V::kAddedHazeAlbedo), 1e-5F) << km;
      EXPECT_GT(scatter.y / extinct.y, 0.9F) << km;
    }
    else
      EXPECT_NEAR(scatter.y / extinct.y, 0.9F, 1e-5F) << km;  // the same aerosol, less of it
  }
  // a 2 km mist scatters 98% of what it meets, against Earth's dust's 90%
  {
    glm::vec3 scatter = ownScatter, extinct = ownExtinction;
    V::applyAerosolScale(V::aerosolScale(2.0F, rayleigh, aerosol), scatter, extinct);
    EXPECT_GT(scatter.y / extinct.y, 0.98F);
  }
}

// The sky's tables bounce the light as many times as the air is deep: a clear day keeps the four it
// always had, a mist takes more, and none takes more than kMostOrders
TEST(SkyVisibility, AHazierAirIsBouncedMoreTimes)
{
  namespace V = sky_visibility;
  constexpr float rayleigh = 0.0135578F, aerosol = 0.06F / 0.9F;  // Earth's own as the preset has it, green, per km
  constexpr float rayleighHeight = 8.0F, aerosolHeight = 1.2F;
  const auto ordersAt = [&](float visibilityKm) {
    const float aerosolNow = aerosol * V::aerosolScale(visibilityKm, rayleigh, aerosol);
    return V::scatteringOrders(rayleigh * rayleighHeight + aerosolNow * aerosolHeight, 4);
  };
  EXPECT_EQ(ordersAt(0.0F), 4);  // the atmosphere's own
  EXPECT_EQ(ordersAt(150.0F), 4);
  EXPECT_EQ(ordersAt(15.0F), 5);
  EXPECT_EQ(ordersAt(8.0F), 6);
  EXPECT_EQ(ordersAt(5.0F), 7);
  EXPECT_EQ(ordersAt(2.0F), 11);
  EXPECT_EQ(ordersAt(1.0F), 18);
  EXPECT_EQ(ordersAt(0.5F), V::kMostOrders);
  // never fewer than the clear air's, whatever it asks
  EXPECT_EQ(V::scatteringOrders(0.0F, 4), 4);
  EXPECT_EQ(V::scatteringOrders(1e9F, 4), V::kMostOrders);
}
