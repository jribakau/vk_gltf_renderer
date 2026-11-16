// The ground (shaders/ground.h.slang) compiled as C++: a shadow ray that crosses the floor from
// above is blocked by as much of it as a path's own ray would hit there, and the water on it covers
// the share of the floor its amount asks for.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace glm;
#include "../shaders/shaderio.h"
using namespace shaderio;
inline float saturate(float x) { return glm::clamp(x, 0.0f, 1.0f); }
#include "../shaders/ground.h.slang"
#include "../shaders/world_sea.h.slang"
#include "../shaders/world_ground.h.slang"
#include "../src/sky_ground.hpp"

namespace {

SceneFrameInfo floorAt(float height, float fadeRadius = 0.0f)
{
  SceneFrameInfo frame{};
  frame.flags                 = SceneFrameInfoFlags::eSceneUseInfinitePlane;
  frame.infinitePlaneDistance = height;
  frame.groundFadeRadius      = fadeRadius;
  return frame;
}

float through(SceneFrameInfo& frame, float3 origin, float3 direction, float tMax = 1e30f)
{
  return infinitePlaneShadowTransmittance(origin, normalize(direction), tMax, &frame);
}

}  // namespace

TEST(Ground, AShadowRayThroughTheFloorIsBlocked)
{
  SceneFrameInfo floor = floorAt(-1.0f);
  EXPECT_EQ(through(floor, float3(0, 0, 0), float3(0.3f, -1, 0)), 0.0f);  // down through it
  EXPECT_EQ(through(floor, float3(0, 0, 0), float3(0, 1, 0)), 1.0f);      // up, away from it
  EXPECT_EQ(through(floor, float3(0, 0, 0), float3(1, 0, 0)), 1.0f);      // along it
  EXPECT_EQ(through(floor, float3(0, 0, 0), float3(0, -1, 0), 0.5f), 1.0f);  // a light above the floor
  EXPECT_EQ(through(floor, float3(0, -2, 0), float3(0, -1, 0)), 1.0f);       // from under it
  EXPECT_EQ(through(floor, float3(0, -1, 0), float3(0.3f, -1, 0)), 1.0f);    // from on it (the floor's own light sample)
  EXPECT_EQ(through(floor, float3(0, 0, 0), float3(0, -1, 0), 1.0f), 1.0f);  // a light exactly on it

  SceneFrameInfo none = floor;
  none.flags          = 0;
  EXPECT_EQ(through(none, float3(0, 0, 0), float3(0, -1, 0)), 1.0f);

  // The catcher only shows shadows; it is not there for the light
  SceneFrameInfo catcher = floor;
  catcher.flags |= SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher;
  EXPECT_EQ(through(catcher, float3(0, 0, 0), float3(0, -1, 0)), 1.0f);
}

// A path's ray hits the faded floor with probability groundFadeWeight; a shadow ray passes the rest
TEST(Ground, TheFadeLetsThroughWhatAPathWouldMiss)
{
  SceneFrameInfo floor = floorAt(0.0f, 10.0f);
  EXPECT_NEAR(through(floor, float3(0, 1, 0), float3(0, -1, 0)), 0.0f, 1e-6f);
  EXPECT_NEAR(through(floor, float3(7.75f, 1, 0), float3(0, -1, 0)), 0.5f, 1e-5f);  // halfway through the fade
  EXPECT_NEAR(through(floor, float3(0, 1, 12), float3(0, -1, 0)), 1.0f, 1e-6f);     // past its edge
  // Where the ray crosses, not where it starts
  EXPECT_NEAR(through(floor, float3(0, 1, 0), float3(7.75f, -1, 0)), 0.5f, 1e-5f);
}

namespace {

// Points spread over 200 m of floor, the same for every test
std::vector<float2> floorPoints(int count)
{
  std::vector<float2> points;
  uint                state = 12345u;
  const auto          next  = [&state]() {
    state = state * 747796405u + 2891336453u;
    return float(state >> 8) / 16777216.0f;
  };
  for(int i = 0; i < count; ++i)
  {
    const float x = next(), z = next();
    points.push_back(float2(x * 200.0f - 100.0f, z * 200.0f - 100.0f));
  }
  return points;
}

struct Coverage
{
  float damp = 0, puddle = 0;
};

Coverage coverage(float amount, float footprint)
{
  Coverage c;
  const std::vector<float2> points = floorPoints(100000);
  for(const float2& p : points)
  {
    const GroundWater w = groundWaterAt(p, footprint, amount, 0.0f);
    c.damp += w.damp;
    c.puddle += w.puddle;
  }
  c.damp /= float(points.size());
  c.puddle /= float(points.size());
  return c;
}

}  // namespace

// The variances the water's field and ripples are scaled by are the noise's own
TEST(GroundWater, TheNoiseHasTheVarianceItIsScaledBy)
{
  double value = 0, slopeX = 0, slopeZ = 0, mean = 0, worstDerivative = 0;
  for(const float2& p : floorPoints(200000))
  {
    const float3 n = groundNoise(p);
    mean += n.x;
    value += double(n.x) * n.x;
    slopeX += double(n.y) * n.y;
    slopeZ += double(n.z) * n.z;
    // the analytic gradient is the value's (near the origin, where a float resolves the step)
    const float2 q = p * 0.05f;
    const float  h = 1e-3f;
    const float  d = (groundNoise(q + float2(h, 0.0f)).x - groundNoise(q - float2(h, 0.0f)).x) / (2.0f * h);
    worstDerivative = std::max(worstDerivative, double(std::abs(d - groundNoise(q).y)));
  }
  const double count = 200000;
  EXPECT_NEAR(mean / count, 0.0, 0.002);
  EXPECT_NEAR(value / count, GROUND_NOISE_VARIANCE, 0.03 * GROUND_NOISE_VARIANCE);
  EXPECT_NEAR(slopeX / count, GROUND_NOISE_SLOPE_VARIANCE, 0.03 * GROUND_NOISE_SLOPE_VARIANCE);
  EXPECT_NEAR(slopeZ / count, GROUND_NOISE_SLOPE_VARIANCE, 0.03 * GROUND_NOISE_SLOPE_VARIANCE);
  EXPECT_LT(worstDerivative, 0.01);
}

TEST(GroundWater, NoneIsDryAndAllIsASheet)
{
  for(const float2& p : floorPoints(2000))
  {
    const GroundWater dry = groundWaterAt(p, 0.0f, 0.0f, 1.0f);
    EXPECT_EQ(dry.damp, 0.0f);
    EXPECT_EQ(dry.puddle, 0.0f);
    EXPECT_EQ(dry.slope.x, 0.0f);
    // damp ground, no water standing on it
    EXPECT_EQ(groundWaterAt(p, 0.0f, 0.3f, 1.0f).puddle, 0.0f);
    EXPECT_EQ(groundWaterAt(p, 0.0f, 0.5f, 1.0f).damp, 1.0f);
    const GroundWater sheet = groundWaterAt(p, 0.0f, 1.0f, 0.0f);
    EXPECT_EQ(sheet.damp, 1.0f);
    EXPECT_EQ(sheet.puddle, 1.0f);
    EXPECT_EQ(sheet.slope.x, 0.0f);  // still: no ripples
  }
}

// The amount is the share of the floor: damp over twice it, puddles from 0.3 to all of it at 1
TEST(GroundWater, TheAmountIsTheShareOfTheFloorCovered)
{
  for(const float amount : {0.1f, 0.2f, 0.4f, 0.5f, 0.65f, 0.8f, 0.95f})
  {
    const Coverage c = coverage(amount, 0.0f);
    EXPECT_NEAR(c.damp, std::min(1.0f, 2.0f * amount), 0.03f) << amount;
    EXPECT_NEAR(c.puddle, std::max(0.0f, (amount - 0.3f) / 0.7f), 0.03f) << amount;
  }
}

// A footprint that averages the field out keeps the share: a far floor is as wet as a near one
TEST(GroundWater, AFarFloorKeepsTheShare)
{
  for(const float footprint : {0.2f, 0.5f, 1.0f})
    for(const float amount : {0.2f, 0.65f})
    {
      const Coverage near = coverage(amount, 0.0f);
      const Coverage far  = coverage(amount, footprint);
      EXPECT_NEAR(far.damp, near.damp, 0.02f) << footprint << " " << amount;
      EXPECT_NEAR(far.puddle, near.puddle, 0.02f) << footprint << " " << amount;
    }
}

// The ripples' slope has the RMS the slider asks for, and what a footprint averages out of it goes
// into the variance the roughness takes, so the water reflects as rough a sky either way
TEST(GroundWater, RipplesTheFootprintCoversBecomeRoughness)
{
  for(const float footprint : {0.0f, 0.03f, 0.08f, 1.0f})
  {
    double kept = 0, averaged = 0;
    const std::vector<float2> points = floorPoints(100000);
    for(const float2& p : points)
    {
      const GroundWater w = groundWaterAt(p, footprint, 1.0f, 1.0f);
      kept += 0.5 * (double(w.slope.x) * w.slope.x + double(w.slope.y) * w.slope.y);
      averaged += w.slopeVariance;
    }
    const double rms = GROUND_RIPPLE_SLOPE;
    EXPECT_NEAR((kept + averaged) / double(points.size()), rms * rms, 0.05 * rms * rms) << footprint;
    if(footprint == 0.0f)
      EXPECT_EQ(averaged, 0.0);
    if(footprint == 1.0f)
      EXPECT_EQ(kept, 0.0);
  }
}

// The sea as the sky's ground: the open sea a little blue over its surface's 6.6%, whatever floor lies
// far below; shallow water over white sand a lagoon's cyan; deeper, darker, down to the sea
TEST(GroundWater, TheSeaAsTheSkysGround)
{
  const float3 sea = groundWaterAlbedo(float3(0.8f), 1000.0f);
  EXPECT_NEAR(sea.x, 0.066f, 0.002f);
  EXPECT_NEAR(sea.y, 0.069f, 0.002f);
  EXPECT_NEAR(sea.z, 0.096f, 0.004f);
  EXPECT_NEAR(groundWaterAlbedo(float3(0.05f), 1000.0f).z, sea.z, 1e-4f);

  const float3 lagoon = groundWaterAlbedo(float3(0.6f), 2.0f);
  EXPECT_LT(lagoon.x, 0.2f);
  EXPECT_GT(lagoon.y, 0.3f);
  EXPECT_GT(lagoon.z, lagoon.y);

  float last = 1.0f;
  for(float d = 0.0f; d <= 40.0f; d += 2.0f)
  {
    const float g = groundWaterAlbedo(float3(0.6f), d).y;
    EXPECT_LE(g, last);
    EXPECT_GE(g, sea.y - 1e-4f);
    last = g;
  }
  // a floor under a film of water is no brighter than dry
  EXPECT_LE(groundWaterAlbedo(float3(0.3f), 0.0f).y, 0.3f);
}

// The sky's ground: the backdrop's land first, then the water, then the atmosphere's own setting
TEST(GroundWater, TheSkysGroundIsTheLandThenTheWater)
{
  const glm::vec3 none(0.0f), land(0.1f, 0.12f, 0.05f), floor(0.5f), ground(0.2f);
  EXPECT_EQ(sky_ground::albedo(land, 3.0f, floor, ground), land);
  EXPECT_EQ(sky_ground::albedo(none, 0.0f, floor, ground), ground);
  const glm::vec3 water = sky_ground::albedo(none, 3.0f, floor, ground);
  EXPECT_NEAR(water.x, groundWaterAlbedo(float3(0.5f), 3.0f).x, 1e-6f);
  EXPECT_LT(water.x, ground.x);
}

// Wet ground is darker, a dark one by more, and never brighter than dry
TEST(GroundWater, WetGroundDarkens)
{
  EXPECT_NEAR(groundWetAlbedo(float3(0.2f)).x, 0.125f, 0.002f);
  float lastRatio = 0;
  for(float a = 0.05f; a <= 1.0f; a += 0.05f)
  {
    const float wet = groundWetAlbedo(float3(a)).x;
    EXPECT_LE(wet, a);
    EXPECT_GE(wet / a, lastRatio);  // the darker, the more it darkens
    lastRatio = wet / a;
  }
}

namespace {

SceneFrameInfo lakeAt(float floor, float depth, float waves)
{
  SceneFrameInfo frame{};
  frame.flags                 = SceneFrameInfoFlags::eSceneUseInfinitePlane;
  frame.infinitePlaneDistance = floor;
  frame.waterDepth            = depth;
  frame.waterWaves            = waves;
  return frame;
}

// The ray's height above the water's surface at t
float aboveWater(float3 origin, float3 direction, float t, float level, float waves)
{
  const float3 p = origin + direction * t;
  return p.y - level - groundWavesAt(float2(p.x, p.z), 0.0f, waves).height;
}

// A direction `elevation` radians off the horizontal, down (or up), at azimuth `azimuth`
float3 slanted(float elevation, float azimuth, bool down)
{
  return float3(std::cos(elevation) * std::cos(azimuth), (down ? -1.0f : 1.0f) * std::sin(elevation), std::cos(elevation) * std::sin(azimuth));
}

}  // namespace

// The swell's slope is its height's, and its bound holds it: gentle, a few centimetres
TEST(GroundWaves, TheSlopeIsTheHeightsAndTheBoundHoldsIt)
{
  const float bound = groundWaveBound(1.0f);
  EXPECT_GT(bound, 0.04f);
  EXPECT_LT(bound, 0.08f);
  EXPECT_EQ(groundWaveBound(0.0f), 0.0f);
  float highest = 0, worst = 0;
  for(const float2& p : floorPoints(50000))
  {
    const float2      q = p * 0.05f;  // within 5 m, where a float resolves the step
    const GroundWaves w = groundWavesAt(q, 0.0f, 1.0f);
    highest             = std::max(highest, std::abs(w.height));
    const float h       = 1e-3f;
    const float dx = (groundWavesAt(q + float2(h, 0.0f), 0.0f, 1.0f).height - groundWavesAt(q - float2(h, 0.0f), 0.0f, 1.0f).height) / (2.0f * h);
    const float dz = (groundWavesAt(q + float2(0.0f, h), 0.0f, 1.0f).height - groundWavesAt(q - float2(0.0f, h), 0.0f, 1.0f).height) / (2.0f * h);
    worst = std::max(worst, std::max(std::abs(dx - w.slope.slope.x), std::abs(dz - w.slope.slope.y)));
    EXPECT_EQ(w.slope.variance, 0.0f);
  }
  EXPECT_LE(highest, bound);
  EXPECT_GT(highest, 0.5f * bound);
  EXPECT_LT(worst, 1e-3f);
  const GroundWaves still = groundWavesAt(float2(1.0f, 2.0f), 0.0f, 0.0f);
  EXPECT_EQ(still.height, 0.0f);
  EXPECT_EQ(still.slope.slope.x, 0.0f);
}

// What a footprint averages out of the swell's slope goes into the variance, so a far surface
// reflects as rough a sky as a near one does on average
TEST(GroundWaves, WavesTheFootprintCoversBecomeRoughness)
{
  double reference = -1;
  for(const float footprint : {0.0f, 0.3f, 1.0f, 10.0f})
  {
    double                    total  = 0;
    const std::vector<float2> points = floorPoints(100000);
    for(const float2& p : points)
    {
      const GroundWaves w = groundWavesAt(p, footprint, 1.0f);
      total += 0.5 * (double(w.slope.slope.x) * w.slope.slope.x + double(w.slope.slope.y) * w.slope.slope.y) + w.slope.variance;
    }
    total /= double(points.size());
    if(reference < 0)
      reference = total;
    EXPECT_NEAR(total, reference, 0.1 * reference) << footprint;
  }
  // gentle: an RMS slope of a few hundredths per axis
  EXPECT_GT(std::sqrt(reference), 0.02);
  EXPECT_LT(std::sqrt(reference), 0.06);
}

// A ray meets the surface where it is, the first time it does. Steeper than the steepest wave, a
// ray's height over the surface only falls (or rises), so the crossing is the one there is.
TEST(GroundWaves, ARayMeetsTheSurfaceWhereItIs)
{
  const float level = 0.7f, waves = 1.0f;
  const float bound = groundWaveBound(waves);
  uint        state = 99u;
  const auto  next  = [&state]() {
    state = state * 747796405u + 2891336453u;
    return float(state >> 8) / 16777216.0f;
  };
  for(int i = 0; i < 2000; ++i)
  {
    const bool   down      = (i & 1) == 0;
    const float  elevation = 0.27f + next() * 1.3f;  // 15 to 90 degrees
    const float3 direction = slanted(elevation, next() * 6.2831853f, down);
    const float3 origin    = float3(next() * 20.0f - 10.0f, level + (down ? 0.3f + 2.0f * next() : -bound - 0.02f - 0.5f * next()), next() * 20.0f - 10.0f);
    const GroundWaterCrossing c = groundWaterCrossing(origin, direction, 1e30f, level, waves);
    ASSERT_GT(c.t, 0.0f) << i;
    EXPECT_EQ(c.fromAbove, down) << i;
    EXPECT_LT(std::abs(aboveWater(origin, direction, c.t, level, waves)), 1e-4f) << i;
    for(int k = 0; k < 50; ++k)  // nothing nearer
    {
      const float f = aboveWater(origin, direction, c.t * float(k) / 50.0f * 0.999f, level, waves);
      EXPECT_TRUE(down ? f > 0.0f : f < 0.0f) << i << " " << k;
    }
    // short of the surface, or away from it: none
    EXPECT_LT(groundWaterCrossing(origin, direction, 0.5f * c.t, level, waves).t, 0.0f) << i;
    EXPECT_LT(groundWaterCrossing(origin, -direction, 1e30f, level, waves).t, 0.0f) << i;
  }
  // Still water is the level
  const GroundWaterCrossing still = groundWaterCrossing(float3(0, 2, 0), normalize(float3(0.5f, -1, 0)), 1e30f, level, 0.0f);
  EXPECT_NEAR(still.t, (2.0f - level) * std::sqrt(1.25f), 1e-4f);
}

TEST(GroundWaves, TheSurfaceReflectsWhatWaterDoes)
{
  EXPECT_NEAR(groundWaterFresnel(1.0f), (0.33f / 2.33f) * (0.33f / 2.33f), 1e-5f);
  EXPECT_NEAR(groundWaterFresnel(0.0f), 1.0f, 1e-5f);
  float last = 1.0f;
  for(float c = 0.05f; c <= 1.0f; c += 0.05f)
  {
    EXPECT_LE(groundWaterFresnel(c), last);
    last = groundWaterFresnel(c);
  }
}

// A shadow ray keeps what the surface lets through and the water does not absorb, along the part of
// it under the water
// Seawater's phase function takes in all the light it scatters (over the sphere, 1), and deep water
// lit and seen straight down sends back, once scattered, what ground.h.slang says: about 4.6% of the
// blue, 0.6% of the green, 0.07% of the red -- the open sea's blue. Its single scattering over an
// unbounded depth: sigma_s p(180) / (2 sigma_t) per unit of the sun's irradiance, times pi.
TEST(GroundWater, DeepWaterSendsBackBlue)
{
  double total = 0.0;
  const int n  = 2000;
  for(int i = 0; i < n; ++i)
  {
    const float c = -1.0f + 2.0f * (float(i) + 0.5f) / float(n);
    total += groundWaterPhase(c) * 2.0 * 3.14159265 * (2.0 / n);
  }
  EXPECT_NEAR(total, 1.0, 1e-3);

  const float3 sent = float(3.14159265) * GROUND_WATER_SCATTERING * groundWaterPhase(-1.0f) / (2.0f * GROUND_WATER_EXTINCTION);
  EXPECT_NEAR(sent.z, 0.046f, 0.002f);
  EXPECT_NEAR(sent.y, 0.006f, 0.0005f);
  EXPECT_NEAR(sent.x, 0.0007f, 0.0001f);
  EXPECT_GT(sent.z, 5.0f * sent.y);
}

TEST(GroundWaves, AShadowRayThroughTheWater)
{
  SceneFrameInfo lake    = lakeAt(0.0f, 1.0f, 0.4f);
  const float    through = 1.0f - groundWaterFresnel(1.0f);
  const auto     near3   = [](float3 a, float3 b) {
    EXPECT_NEAR(a.x, b.x, 1e-5f);
    EXPECT_NEAR(a.y, b.y, 1e-5f);
    EXPECT_NEAR(a.z, b.z, 1e-5f);
  };
  // the surface over the origin, waves and all
  const float crest = groundWavesAt(float2(0.0f, 0.0f), 0.0f, 0.4f).height;
  // up out of half a metre of water (and the wave there), to the sun, with the caustic there
  const float focus = groundWaterCaustic(float3(0, 0.5f, 0), float3(0, 1, 0), 1.0f, 0.4f, 0.0f);
  near3(groundWaterShadowTransmittance(float3(0, 0.5f, 0), float3(0, 1, 0), 1e32f, &lake), through * focus * exp(-(0.5f + crest) * GROUND_WATER_EXTINCTION));
  // down into it, to a light at the same point
  near3(groundWaterShadowTransmittance(float3(0, 2, 0), float3(0, -1, 0), 1.5f, &lake), through * exp(-(0.5f + crest) * GROUND_WATER_EXTINCTION));
  // in the air, and to a light still above the water
  near3(groundWaterShadowTransmittance(float3(0, 2, 0), float3(0, 1, 0), 1e32f, &lake), float3(1.0f));
  near3(groundWaterShadowTransmittance(float3(0, 2, 0), float3(0, -1, 0), 0.5f, &lake), float3(1.0f));
  // all of it under water: absorbed only
  near3(groundWaterShadowTransmittance(float3(0, 0.5f, 0), float3(0, -1, 0), 0.25f, &lake), exp(-0.25f * GROUND_WATER_EXTINCTION));
  // red goes first
  const float3 deep = groundWaterShadowTransmittance(float3(0, 0.01f, 0), float3(0, 1, 0), 1e32f, &lake);
  EXPECT_LT(deep.x, deep.y);
  EXPECT_LT(deep.y, deep.z);
  // no water: a floor without depth, or the catcher
  SceneFrameInfo dry = lakeAt(0.0f, 0.0f, 0.4f);
  near3(groundWaterShadowTransmittance(float3(0, 0.5f, 0), float3(0, 1, 0), 1e32f, &dry), float3(1.0f));
  SceneFrameInfo catcher = lake;
  catcher.flags |= SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher;
  near3(groundWaterShadowTransmittance(float3(0, 0.5f, 0), float3(0, 1, 0), 1e32f, &catcher), float3(1.0f));
}

// Under a crest, over the level, is in the water: a shadow ray from there crosses the surface and is
// charged the water up to it, as the path's own rays see it
TEST(GroundWaves, UnderACrestIsInTheWater)
{
  SceneFrameInfo lake = lakeAt(0.0f, 1.0f, 1.0f);
  // the highest crest near the origin
  float2 top = float2(0.0f);
  float  h   = -1.0f;
  for(int i = 0; i < 400; ++i)
    for(int j = 0; j < 400; ++j)
    {
      const float2 p = float2(float(i), float(j)) * 0.01f;
      const float  w = groundWavesAt(p, 0.0f, 1.0f).height;
      if(w > h)
      {
        h   = w;
        top = p;
      }
    }
  ASSERT_GT(h, 0.02f);
  const float3 under = float3(top.x, 1.0f + 0.5f * h, top.y);
  EXPECT_TRUE(groundUnderWater(under, &lake));
  const float3 t = groundWaterShadowTransmittance(under, float3(0, 1, 0), 1e32f, &lake);
  // a centimetre or so under the crest: next to no caustic yet
  const float focus = groundWaterCaustic(under, float3(0, 1, 0), 1.0f, 1.0f, 0.0f);
  EXPECT_NEAR(focus, 1.0f, 0.01f);
  EXPECT_LT(t.x, 1.0f - groundWaterFresnel(1.0f));
  EXPECT_NEAR(t.z, (1.0f - groundWaterFresnel(1.0f)) * focus * std::exp(-0.5f * h * GROUND_WATER_EXTINCTION.z), 1e-5f);
}

namespace {

// The caustic over a square of floor `side` metres across, `n` points a side, `depth` under the
// level, lit from `toLight`
struct CausticSpread
{
  double mean = 0, lowest = 1e9, highest = 0, bright = 0, dark = 0;  // bright: share over 1.5, dark: under 0.67
};

CausticSpread causticsOver(float depth, float waves, float ripples, float3 toLight, float side = 2.0f, int n = 200)
{
  CausticSpread s;
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
    {
      const float3 p(side * (float(i) + 0.37f) / float(n), 1.0f - depth, side * (float(j) + 0.61f) / float(n));
      const float  c = groundWaterCaustic(p, normalize(toLight), 1.0f, waves, ripples);
      s.mean += c;
      s.lowest  = std::min(s.lowest, double(c));
      s.highest = std::max(s.highest, double(c));
      s.bright += c > 1.5f;
      s.dark += c < 0.67f;
    }
  const double count = double(n) * double(n);
  s.mean /= count;
  s.bright /= count;
  s.dark /= count;
  return s;
}

}  // namespace

// Still water focuses nothing; waves and ripples make bands of light on the floor that grow with
// the depth, never past the cut, and move the light about rather than adding to it or taking it away
TEST(GroundWaves, TheSurfaceFocusesTheSunIntoCaustics)
{
  EXPECT_EQ(groundWaterCaustic(float3(0.3f, 0.2f, 0.7f), float3(0, 1, 0), 1.0f, 0.0f, 0.0f), 1.0f);
  // Measured 11-04 at Waves and Ripples 1, over 2 m of floor: the mean 0.98 to 1.0 overhead and 0.96
  // to 1.02 at 48 degrees; at 0.6 m overhead 0.43x to 4.7x, 8% of the floor over 1.5x
  for(const float3 sun : {float3(0, 1, 0), float3(0.6f, 0.7f, 0.2f)})
    for(const float depth : {0.1f, 0.3f, 0.6f, 1.2f, 2.5f})
    {
      const CausticSpread c = causticsOver(depth, 1.0f, 1.0f, sun);
      EXPECT_NEAR(c.mean, 1.0, 0.05) << depth;
      EXPECT_LE(c.highest, GROUND_CAUSTIC_MAX + 1e-4) << depth;
      EXPECT_GT(c.lowest, 0.25) << depth;
      if(depth >= 0.3f && depth <= 1.2f)
      {
        EXPECT_GT(c.bright, 0.03) << depth;
        EXPECT_GT(c.dark, 0.05) << depth;
      }
      // with the sun overhead the landing never folds, so nothing reaches the cut
      if(sun.y == 1.0f)
        EXPECT_LT(c.highest, GROUND_CAUSTIC_MAX - 1.0) << depth;
    }
  // light seen through a shallow film barely changes; deeper, the bands sharpen
  EXPECT_LT(causticsOver(0.1f, 1.0f, 1.0f, float3(0, 1, 0)).bright, causticsOver(0.6f, 1.0f, 1.0f, float3(0, 1, 0)).bright);
  // the ripples make most of it: the swell alone, a metre down, moves the light by a few percent
  const CausticSpread swell = causticsOver(1.0f, 1.0f, 0.0f, float3(0, 1, 0));
  EXPECT_GT(swell.lowest, 0.8);
  EXPECT_LT(swell.highest, 1.25);
}

// In water shallower than its swell the waves are kept lower, so no trough reaches the floor
TEST(GroundWaves, ShallowWaterKeepsItsTroughsOffTheFloor)
{
  for(const float depth : {0.01f, 0.03f, 0.06f, 0.2f})
  {
    SceneFrameInfo lake = lakeAt(0.0f, depth, 1.0f);
    EXPECT_LE(groundWaveBound(groundWaterWaves(&lake)), 0.8f * depth + 1e-6f) << depth;
  }
  // deep enough, the waves are those asked for
  SceneFrameInfo deep = lakeAt(0.0f, 0.5f, 0.7f);
  EXPECT_EQ(groundWaterWaves(&deep), 0.7f);
}

// A ray low over the water, which can skim a crest and cross twice in the waves' band, meets the
// surface where it first does
TEST(GroundWaves, AGrazingRayMeetsTheFirstCrossing)
{
  const float level = 0.0f, waves = 1.0f;
  uint        state = 7u;
  const auto  next  = [&state]() {
    state = state * 747796405u + 2891336453u;
    return float(state >> 8) / 16777216.0f;
  };
  for(int i = 0; i < 2000; ++i)
  {
    const float  elevation = 0.1f + next() * 0.17f;  // 6 to 15 degrees
    const float3 direction = slanted(elevation, next() * 6.2831853f, true);
    const float3 origin    = float3(next() * 20.0f - 10.0f, 0.5f + next(), next() * 20.0f - 10.0f);
    const GroundWaterCrossing c = groundWaterCrossing(origin, direction, 1e30f, level, waves);
    ASSERT_GT(c.t, 0.0f) << i;
    EXPECT_TRUE(c.fromAbove) << i;
    EXPECT_LT(std::abs(aboveWater(origin, direction, c.t, level, waves)), 1e-4f) << i;
    for(int k = 0; k < 400; ++k)  // nothing nearer, a few millimetres apart
      EXPECT_GT(aboveWater(origin, direction, c.t * float(k) / 400.0f * 0.999f, level, waves), 0.0f) << i << " " << k;
  }
}

TEST(GroundWaves, WhatIsUnderTheWater)
{
  SceneFrameInfo lake = lakeAt(-0.2f, 1.0f, 1.0f);
  EXPECT_TRUE(groundUnderWater(float3(0.3f, 0.5f, 0.1f), &lake));
  EXPECT_FALSE(groundUnderWater(float3(0.3f, 0.9f, 0.1f), &lake));
  // the waves decide near the level
  const float2 p = float2(1.7f, -0.4f);
  const float  h = groundWavesAt(p, 0.0f, 1.0f).height;
  EXPECT_TRUE(groundUnderWater(float3(p.x, 0.8f + h - 1e-3f, p.y), &lake));
  EXPECT_FALSE(groundUnderWater(float3(p.x, 0.8f + h + 1e-3f, p.y), &lake));
  SceneFrameInfo dry = lakeAt(-0.2f, 0.0f, 1.0f);
  EXPECT_FALSE(groundUnderWater(float3(0.3f, 0.0f, 0.1f), &dry));
}

//-----------------------------------------------------------------------
// A planet's sea (shaders/world_sea.h.slang): the water above, on a sphere
//-----------------------------------------------------------------------

namespace {

// A sea 20 km in radius whose top is 12.2 m under the scene's origin, as a planet's is under its site
Sea seaOf(float waves = 1.0f, float ripples = 1.0f, float radius = 20000.0f)
{
  Sea sea;
  sea.top     = float3(0.0f, -12.2f, 0.0f);
  sea.radius  = radius;
  sea.waves   = waves;
  sea.ripples = ripples;
  return sea;
}

// A place on the sea's sphere, with the plane that touches it there: `x`, `up` and `z` in the world
struct Spot
{
  float3 at, x, up, z;
  float3 world(float3 local) const { return at + x * local.x + up * local.y + z * local.z; }
  float3 turned(float3 local) const { return x * local.x + up * local.y + z * local.z; }
};

Spot spotOn(const Sea& sea, float3 direction)
{
  Spot s;
  s.up = normalize(direction);
  s.x  = normalize(cross(std::abs(s.up.y) < 0.9f ? float3(0, 1, 0) : float3(1, 0, 0), s.up));
  s.z  = cross(s.up, s.x);
  // (in doubles: the centre is 20 km off, where a float is good to two millimetres)
  s.at = float3(dvec3(sea.top) + (dvec3(s.up) - dvec3(0.0, 1.0, 0.0)) * double(sea.radius));
  return s;
}

// A point's height over the sea, in doubles from the centre
double heightOver(const Sea& sea, float3 p)
{
  const dvec3 centre = dvec3(sea.top) - dvec3(0.0, double(sea.radius), 0.0);
  return length(dvec3(p) - centre) - double(sea.radius);
}

}  // namespace

// By the top of the sea, where the scene is, a height over it is good to a micrometre: taken from the
// centre, 20 km off, it would be good to two millimetres, more than a ray's offset from a surface
TEST(Sea, HeightsKeepTheirPrecisionByTheTop)
{
  const Sea sea = seaOf();
  for(const float3& p : {float3(3.0f, -12.199f, 4.0f), float3(-40.0f, -12.3f, 15.0f), float3(0.2f, -11.0f, -0.7f), float3(120.0f, 3.0f, -80.0f)})
  {
    EXPECT_NEAR(seaHeight(sea, p), heightOver(sea, p), 2e-6) << p.x;
    EXPECT_EQ(seaUnder(sea, p), heightOver(sea, p) < 0.0) << p.x;
    EXPECT_NEAR(dot(seaUp(sea, p), float3(0, 1, 0)), 1.0f, 1e-4f);
  }
  // and anywhere on the planet as good as the point itself is
  for(const float3& d : {float3(1, 0, 0), float3(0, -1, 0), float3(0.5f, 0.3f, -0.8f)})
  {
    const Spot spot = spotOn(sea, d);
    EXPECT_NEAR(seaHeight(sea, spot.world(float3(5.0f, 1.5f, -3.0f))), heightOver(sea, spot.world(float3(5.0f, 1.5f, -3.0f))), 0.02) << d.x;
    EXPECT_NEAR(dot(seaUp(sea, spot.at), spot.up), 1.0f, 1e-5f);
  }
}

// Round the top the sea's waves are the plane's own, the swell SEA_SWELL_SCALE times as long;
// everywhere they lie in the sphere's surface, and what a footprint covers of them is roughness
TEST(Sea, ItsWavesAreThePlanesLaidOnTheSphere)
{
  const Sea sea = seaOf();
  for(const float2& p : floorPoints(2000))
  {
    const float3      pos     = float3(p.x, sea.top.y, p.y);
    const SeaWaves    w       = seaWavesAt(sea, pos, 0.0f, 1.0f, 0.0f, 1.0f);
    const GroundWaves flat    = groundWavesAt(p / SEA_SWELL_SCALE, 0.0f, 1.0f);
    const GroundSlope ripples = groundRipplesAt(p, 0.0f, 1.0f);
    EXPECT_NEAR(w.slope.x, flat.slope.slope.x + ripples.slope.x, 2e-4f);
    EXPECT_NEAR(w.slope.z, flat.slope.slope.y + ripples.slope.y, 2e-4f);
    // the surface a footprint sees: its normal the sphere's, leaning against the waves' slope
    const SeaSurface surface = seaSurfaceAt(sea, pos, 0.0f);
    EXPECT_NEAR(surface.up.x / surface.up.y, p.x / sea.radius - w.slope.x, 2e-4f);
    EXPECT_NEAR(surface.up.z / surface.up.y, p.y / sea.radius - w.slope.z, 2e-4f);
  }
  uint       state = 5u;
  const auto next  = [&state]() {
    state = state * 747796405u + 2891336453u;
    return float(state >> 8) / 16777216.0f;
  };
  double steep = 0.0;
  for(int i = 0; i < 20000; ++i)
  {
    const float3   d    = normalize(float3(next() * 2.0f - 1.0f, next() * 2.0f - 1.0f, next() * 2.0f - 1.0f) + float3(1e-3f));
    const Spot     spot = spotOn(sea, d);
    const SeaWaves w    = seaWavesAt(sea, spot.at, 0.0f, 1.0f, 0.0f, 1.0f);
    EXPECT_NEAR(dot(w.slope, spot.up), 0.0f, 1e-4f) << i;
    EXPECT_LT(length(w.slope), 0.6f) << i;
    EXPECT_EQ(w.variance, 0.0f);
    steep += double(dot(w.slope, w.slope));
    // what a footprint as wide as the waves covers is in the variance, none of it in the slope
    const SeaWaves far = seaWavesAt(sea, spot.at, 100.0f, 1.0f, 100.0f, 1.0f);
    EXPECT_LT(length(far.slope), 1e-6f);
    EXPECT_GT(far.variance, 0.001f);
    EXPECT_NEAR(dot(seaSurfaceAt(sea, spot.at, 100.0f).up, spot.up), 1.0f, 1e-5f);
  }
  // gentle all over the sphere: an RMS slope of a few hundredths an axis, where planes blend too
  EXPECT_GT(std::sqrt(0.5 * steep / 20000.0), 0.03);
  EXPECT_LT(std::sqrt(0.5 * steep / 20000.0), 0.09);
  // along an arc from the top to the side, 100 m across each place where a plane's waves come in
  // (60 degrees off its axis): 5 cm on, the slope has moved by no more than the ripples turn it and
  // a float's rounding 20 km from the centre allow, so no seam shows
  for(const double across : {0.6751, 0.9851, 1.0472})
  {
    float3 last(0.0f);
    for(int i = 0; i < 2000; ++i)
    {
      const double angle = across + (double(i) - 1000.0) * 0.05 / double(sea.radius);
      const dvec3  d(std::sin(angle) * 0.8, std::cos(angle), std::sin(angle) * 0.6);
      const float3 at    = float3(dvec3(sea.top) - dvec3(0.0, double(sea.radius), 0.0) + d * double(sea.radius));
      const float3 slope = seaWavesAt(sea, at, 0.0f, 1.0f, 0.0f, 0.0f).slope;
      if(i > 0)
        EXPECT_LT(length(slope - last), 0.03f) << across << " " << i;
      last = slope;
    }
  }
  // a still sea is the sphere
  const SeaWaves still = seaWavesAt(sea, float3(1.0f, sea.top.y, 2.0f), 0.0f, 0.0f, 0.0f, 0.0f);
  EXPECT_EQ(still.slope.x, 0.0f);
  EXPECT_EQ(still.variance, 0.0f);
}

// A ray meets the sea where its sphere is, from over it and from under it: by the top, where the
// scene is, and anywhere round the planet
TEST(Sea, ARayMeetsTheSurfaceWhereItIs)
{
  const Sea  sea   = seaOf();
  uint       state = 99u;
  const auto next  = [&state]() {
    state = state * 747796405u + 2891336453u;
    return float(state >> 8) / 16777216.0f;
  };
  int place = 0;
  for(const float3& d : {float3(0, 1, 0), float3(1, 0, 0), float3(0.5f, 0.3f, -0.8f), float3(-0.6f, -0.7f, 0.4f)})
  {
    const Spot spot = spotOn(sea, d);
    // by the top a height is exact; elsewhere as fine as a float 20 km from the origin
    const float fine = place == 0 ? 2e-5f : 0.02f;
    for(int i = 0; i < 500; ++i)
    {
      const bool   down      = (i & 1) == 0;
      const float  elevation = 0.02f + next() * 1.55f;  // 1 to 90 degrees
      const float3 direction = spot.turned(slanted(elevation, next() * 6.2831853f, down));
      const float3 origin    = spot.world(float3(next() * 20.0f - 10.0f, down ? 0.3f + 2.0f * next() : -0.05f - 0.5f * next(), next() * 20.0f - 10.0f));
      const SeaCrossing c = seaCrossing(sea, origin, direction, 1e30f);
      ASSERT_GT(c.t, 0.0f) << place << " " << i;
      EXPECT_EQ(c.fromAbove, down) << place << " " << i;
      EXPECT_EQ(c.under, !down) << place << " " << i;
      EXPECT_LT(std::abs(heightOver(sea, origin + direction * c.t)), double(fine) + 1e-6 * double(c.t)) << place << " " << i;
      // short of the surface, or away from it: none
      EXPECT_LT(seaCrossing(sea, origin, direction, 0.5f * c.t).t, 0.0f) << place << " " << i;
      if(down)
        EXPECT_LT(seaCrossing(sea, origin, -direction, 1e30f).t, 0.0f) << place << " " << i;
      // a ray's offset off the surface is on the side it was put: no crossing again on the way out,
      // and one at once on the way back
      const float3 at    = origin + direction * c.t;
      const float  clear = 1e-4f + 4e-7f * (std::abs(at.x) + std::abs(at.y) + std::abs(at.z));
      EXPECT_FALSE(seaUnder(sea, at + spot.up * (2.0f * clear))) << place << " " << i;
      EXPECT_TRUE(seaUnder(sea, at - spot.up * (2.0f * clear))) << place << " " << i;
      EXPECT_LT(seaCrossing(sea, at + spot.up * (2.0f * clear), spot.up, 1e30f).t, 0.0f) << place << " " << i;
      EXPECT_LT(seaCrossing(sea, at - spot.up * (2.0f * clear), spot.up, 1e30f).t, 4.0f * clear) << place << " " << i;
      EXPECT_GT(seaCrossing(sea, at - spot.up * (2.0f * clear), spot.up, 1e30f).t, 0.0f) << place << " " << i;
    }
    place++;
  }
}

// From orbit a ray straight down meets the sea at its height over it, one past the planet's edge
// meets nothing, and one from the sea's depths comes up to the surface
TEST(Sea, FromFarItIsTheSphere)
{
  const Sea         sea  = seaOf();
  const SeaCrossing down = seaCrossing(sea, sea.top + float3(0, 5000.0f, 0), float3(0, -1, 0), 1e30f);
  EXPECT_NEAR(down.t, 5000.0f, 0.01f);
  EXPECT_TRUE(down.fromAbove);
  // towards the planet's edge, 20 km off its centre and 30 km out: a ray to within the radius meets
  // it, one outside it does not
  const float3      out   = sea.top + float3(0, -sea.radius, 30000.0f);
  const SeaCrossing meets = seaCrossing(sea, out, normalize(float3(25000.0f, 0, -30000.0f)), 1e30f);
  EXPECT_GT(meets.t, 10000.0f);
  EXPECT_TRUE(meets.fromAbove);
  EXPECT_NEAR(heightOver(sea, out + normalize(float3(25000.0f, 0, -30000.0f)) * meets.t), 0.0, 0.05);
  EXPECT_LT(seaCrossing(sea, out, normalize(float3(28000.0f, 0, -30000.0f)), 1e30f).t, 0.0f);
  // something nearer than the sea hides it
  EXPECT_LT(seaCrossing(sea, sea.top + float3(0, 5000.0f, 0), float3(0, -1, 0), 4000.0f).t, 0.0f);
  // from 300 m down, up to the surface; and down, out of the planet's far side
  const SeaCrossing up = seaCrossing(sea, sea.top + float3(7.0f, -300.0f, 2.0f), float3(0, 1, 0), 1e30f);
  EXPECT_NEAR(up.t, 300.0f, 0.01f);
  EXPECT_FALSE(up.fromAbove);
  EXPECT_TRUE(up.under);
  EXPECT_NEAR(seaCrossing(sea, sea.top + float3(0.0f, -300.0f, 0.0f), float3(0, -1, 0), 1e30f).t, 2.0f * sea.radius - 300.0f, 0.1f);
  // a little way off, exactly
  const SeaCrossing level = seaCrossing(sea, sea.top + float3(0, 2.0f, 0), normalize(float3(0.5f, -1, 0)), 1e30f);
  EXPECT_NEAR(level.t, 2.0f * std::sqrt(1.25f), 1e-4f);
}

TEST(Sea, WhatIsUnderIt)
{
  const Sea sea = seaOf();
  EXPECT_TRUE(seaUnder(sea, sea.top + float3(0.3f, -0.5f, 0.1f)));
  EXPECT_FALSE(seaUnder(sea, sea.top + float3(0.3f, 0.5f, 0.1f)));
  EXPECT_FALSE(seaUnder(sea, float3(0.0f, 1.6f, 0.0f)));  // a figure's eyes, on the land over it
  // on the planet's far side down is up
  EXPECT_TRUE(seaUnder(sea, sea.top + float3(0.0f, -2.0f * sea.radius + 3.0f, 0.0f)));
  EXPECT_FALSE(seaUnder(sea, sea.top + float3(0.0f, -2.0f * sea.radius - 3.0f, 0.0f)));
  // with no planet the water is the ground's
  SceneFrameInfo lake = lakeAt(-0.2f, 1.0f, 1.0f);
  EXPECT_FALSE(worldHasSea(&lake));
  EXPECT_TRUE(waterUnder(float3(0.3f, 0.5f, 0.1f), &lake));
  EXPECT_FALSE(waterUnder(float3(0.3f, 0.9f, 0.1f), &lake));
  EXPECT_EQ(waterExtinction(&lake).x, GROUND_WATER_EXTINCTION.x);
  // with one, its sea, whatever the ground's plane says
  lake.worldSeaRadius = sea.radius;
  lake.worldSeaTop    = sea.top;
  EXPECT_TRUE(worldHasSea(&lake));
  EXPECT_FALSE(waterUnder(float3(0.3f, 0.5f, 0.1f), &lake));
  EXPECT_TRUE(waterUnder(float3(0.3f, -13.0f, 0.1f), &lake));
  EXPECT_EQ(waterExtinction(&lake).x, SEA_EXTINCTION.x);
}

// A shadow ray keeps what the surface lets through and the sea does not take, along the part of it
// under the surface, with the caustic there
TEST(Sea, AShadowRayThroughIt)
{
  const Sea   sea     = seaOf(0.4f, 0.0f);
  const float through = 1.0f - groundWaterFresnel(1.0f);
  // by the top a depth is exact; on the planet's far side, 40 km from the origin, good to millimetres
  float      fine  = 1e-4f;
  const auto near3 = [&fine](float3 a, float3 b) {
    EXPECT_NEAR(a.x, b.x, fine);
    EXPECT_NEAR(a.y, b.y, fine);
    EXPECT_NEAR(a.z, b.z, fine);
  };
  for(const float3& d : {float3(0, 1, 0), float3(0, -1, 0)})
  {
    const Spot   spot  = spotOn(sea, d);
    fine               = d.y > 0.0f ? 1e-4f : 5e-3f;
    const float3 under = spot.world(float3(0, -0.5f, 0));
    const float  focus = seaCaustic(sea, under, spot.up);
    // up out of half a metre of water, to the sun
    near3(seaShadowTransmittance(sea, under, spot.up, 1e32f), through * focus * exp(-0.5f * SEA_EXTINCTION));
    // down into it, to a light at the same point
    near3(seaShadowTransmittance(sea, spot.world(float3(0, 2, 0)), -spot.up, 2.5f), through * exp(-0.5f * SEA_EXTINCTION));
    // in the air, and to a light still above the water
    near3(seaShadowTransmittance(sea, spot.world(float3(0, 2, 0)), spot.up, 1e32f), float3(1.0f));
    near3(seaShadowTransmittance(sea, spot.world(float3(0, 2, 0)), -spot.up, 0.5f), float3(1.0f));
    // all of it under water: absorbed only
    near3(seaShadowTransmittance(sea, under, -spot.up, 0.25f), exp(-0.25f * SEA_EXTINCTION));
  }
  // red goes first
  const float3 deep = seaShadowTransmittance(sea, sea.top + float3(0, -3.0f, 0), float3(0, 1, 0), 1e32f);
  EXPECT_LT(deep.x, deep.y);
  EXPECT_LT(deep.y, deep.z);
  // light that grazes the surface is mostly reflected off it; light straight down goes in
  const float3 low   = seaShadowTransmittance(sea, sea.top + float3(0, 1.0f, 0), normalize(float3(1, -0.1f, 0)), 12.0f);
  const float3 steep = seaShadowTransmittance(sea, sea.top + float3(0, 1.0f, 0), float3(0, -1, 0), 1.5f);
  EXPECT_LT(low.z, 0.6f);
  EXPECT_GT(low.z, 0.2f);
  EXPECT_GT(steep.z, 0.95f);
}

// Still water focuses nothing; the sea's ripples and swell make bands of light on its floor that move
// the light about without adding to it; so deep that the water has spread it all, there are none
TEST(Sea, ItsSurfaceFocusesTheSunIntoCaustics)
{
  EXPECT_EQ(seaCaustic(seaOf(0.0f, 0.0f), float3(0.3f, -13.0f, 0.7f), float3(0, 1, 0)), 1.0f);
  const Sea sea = seaOf(0.5f, 0.3f);
  for(const float3& d : {float3(0, 1, 0), float3(0.5f, 0.3f, -0.8f)})
  {
    const Spot spot = spotOn(sea, d);
    for(const float3& sun : {float3(0, 1, 0), float3(0.6f, 0.7f, 0.2f)})
      for(const float depth : {0.5f, 2.0f, 6.0f})
      {
        const int n = 120;
        double    mean = 0, lowest = 1e9, highest = 0;
        for(int i = 0; i < n; ++i)
          for(int j = 0; j < n; ++j)
          {
            const float3 p = spot.world(float3(16.0f * (float(i) + 0.37f) / float(n), -depth, 16.0f * (float(j) + 0.61f) / float(n)));
            const float  c = seaCaustic(sea, p, spot.turned(normalize(sun)));
            mean += c;
            lowest  = std::min(lowest, double(c));
            highest = std::max(highest, double(c));
          }
        mean /= double(n) * double(n);
        EXPECT_NEAR(mean, 1.0, 0.06) << depth << " " << d.x << " " << sun.x;
        EXPECT_LE(highest, GROUND_CAUSTIC_MAX + 1e-4) << depth;
        EXPECT_GT(lowest, 0.2) << depth;
        if(depth <= 2.0f)
          EXPECT_GT(highest - lowest, 0.1) << depth;  // bands, not a flat floor
      }
  }
  EXPECT_EQ(seaCaustic(sea, sea.top + float3(1.0f, -2000.0f, 3.0f), float3(0, 1, 0)), 1.0f);
}

// The sea's particles send on all the light they scatter, most of it forward, so a ray towards the
// sun through the water gathers far more than one across it; and a ray loses to the water only what
// does not go on the way it went
TEST(Sea, ItsWaterScattersForward)
{
  double total = 0.0;
  const int n  = 20000;
  for(int i = 0; i < n; ++i)
  {
    const float c = -1.0f + 2.0f * (float(i) + 0.5f) / float(n);
    total += seaParticlePhase(c) * 2.0 * 3.14159265 * (2.0 / n);
  }
  EXPECT_NEAR(total, 1.0, 5e-3);
  EXPECT_GT(seaScattered(1.0f).y, 50.0f * seaScattered(0.0f).y);
  EXPECT_GT(seaScattered(0.0f).y, seaScattered(-1.0f).y * 0.5f);
  // sideways and back the molecules still count: blue more than red
  EXPECT_GT(seaScattered(-1.0f).z, seaScattered(-1.0f).x);
  EXPECT_GT(SEA_EXTINCTION.x, GROUND_WATER_EXTINCTION.x);
  EXPECT_LT(SEA_EXTINCTION.z, GROUND_WATER_EXTINCTION.z + SEA_PARTICLES);
  // seen from above, deep water still sends back the open sea's blue (DeepWaterSendsBackBlue), a
  // little less of it
  const float3 sent = float(3.14159265) * seaScattered(-1.0f) / (2.0f * SEA_EXTINCTION);
  EXPECT_GT(sent.z, 0.025f);
  EXPECT_LT(sent.z, 0.05f);
  EXPECT_GT(sent.z, 3.0f * sent.y);
}

// In a wind the water's waves travel downwind (g_waterTime, g_waterHeading): the same water a moment
// later, as rough as it was, and in a calm exactly the water there was before it could move
TEST(GroundWaves, TravelDownwindInAWind)
{
  const auto heightsAt = [](float2 from, float2 step, int count) {
    std::vector<float> h;
    for(int i = 0; i < count; i++)
      h.push_back(groundWavesAt(from + step * float(i), 0.0f, 1.0f).height);
    return h;
  };
  const auto correlation = [](const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for(size_t i = 0; i < a.size(); i++)
    {
      ab += double(a[i]) * b[i];
      aa += double(a[i]) * a[i];
      bb += double(b[i]) * b[i];
    }
    return ab / std::sqrt(aa * bb);
  };
  const auto rmsSlope = [] {
    double sum = 0.0;
    for(int i = 0; i < 4000; i++)
    {
      const float2      at(float(i % 63) * 0.37f - 9.0f, float(i / 63) * 0.41f + 3.0f);
      const GroundWaves w = groundWavesAt(at, 0.0f, 1.0f);
      const GroundSlope r = groundRipplesAt(at, 0.0f, 1.0f);
      sum += double(dot(w.slope.slope + r.slope, w.slope.slope + r.slope));
    }
    return std::sqrt(sum / 4000.0);
  };

  for(const float heading : {GROUND_SWELL_WIND, 2.4f, -1.1f})
  {
    const float2 way(std::cos(heading), std::sin(heading));
    // a line of the water along the wind, in a calm
    groundWaterMotion(0.0f, heading);
    const float2             from(4.0f, -7.0f);
    const std::vector<float> calm     = heightsAt(from, way * 0.05f, 400);
    const std::vector<float> calmUp   = heightsAt(from - way * 0.38f, way * 0.05f, 400);
    const std::vector<float> calmDown = heightsAt(from + way * 0.38f, way * 0.05f, 400);
    const double             rough = rmsSlope();
    const GroundSlope        still = groundRipplesAt(from, 0.0f, 1.0f);
    // a fifth of a second on, the water is what stood some 38 cm upwind (its trains go 1.1 to
    // 2.2 m/s, the shorter ones more across the wind), not what stood here, and still less what
    // stood as far downwind
    groundWaterMotion(0.2f, heading);
    const std::vector<float> later = heightsAt(from, way * 0.05f, 400);
    const double             here  = correlation(later, calm);
    const double             up    = correlation(later, calmUp);
    const double             down  = correlation(later, calmDown);
    EXPECT_GT(up, 0.9) << heading;
    EXPECT_GT(up, here + 0.05) << heading;
    EXPECT_LT(down, up - 0.3) << heading;
    // the ripples have moved too, and the water is as rough as it was, then and a quarter of an hour on
    groundWaterMotion(0.2f, heading);
    const GroundSlope moved = groundRipplesAt(from, 0.0f, 1.0f);
    EXPECT_GT(length(moved.slope - still.slope), 1e-3f);
    EXPECT_NEAR(rmsSlope(), rough, 0.05 * rough);
    groundWaterMotion(900.0f, heading);
    EXPECT_NEAR(rmsSlope(), rough, 0.05 * rough);
    // the ripples drift as one field: a quarter of a second on they are the ripples that stood
    // 15 cm upwind, which is what lets DLSS follow them (groundPatternVelocity)
    groundWaterMotion(3.0f, heading);
    const GroundSlope then = groundRipplesAt(from, 0.0f, 1.0f);
    groundWaterMotion(3.25f, heading);
    const GroundSlope now = groundRipplesAt(from + way * (GROUND_RIPPLE_DRIFT * 0.25f), 0.0f, 1.0f);
    EXPECT_NEAR(now.slope.x, then.slope.x, 2e-3f);
    EXPECT_NEAR(now.slope.y, then.slope.y, 2e-3f);
    // a moment's difference is a little water's difference: no jump from one frame to the next
    groundWaterMotion(900.0f, heading);
    const float before = groundWavesAt(from, 0.0f, 1.0f).height;
    groundWaterMotion(900.004f, heading);
    EXPECT_NEAR(groundWavesAt(from, 0.0f, 1.0f).height, before, 2e-3f);
  }

  // a swell four times as long turns over half as fast: the sea's (deep water's dispersion)
  EXPECT_FLOAT_EQ(SEA_SWELL_TIME, 1.0f / std::sqrt(SEA_SWELL_SCALE));
  EXPECT_NEAR(groundWaveSpeed(3.1f), 2.2f, 0.01f);

  // in a calm, with the wind's way never said: the water as it was before it could move
  groundWaterMotion(0.0f, GROUND_SWELL_WIND);
  EXPECT_EQ(g_waterTime, 0.0f);
  const GroundWaves w = groundWavesAt(float2(1.3f, -2.1f), 0.0f, 1.0f);
  float             h = 0.0f;
  for(int i = 0; i < 4; i++)
  {
    const float heading = GROUND_SWELL_WIND + GROUND_SWELL_HEADING[i];
    const float k       = GROUND_TWO_PI / GROUND_SWELL_LENGTH[i];
    h += GROUND_SWELL_STEEPNESS / k * std::sin(k * dot(float2(std::cos(heading), std::sin(heading)), float2(1.3f, -2.1f)) + GROUND_SWELL_PHASE[i]);
  }
  for(int j = 0; j < 2; j++)
    h += GROUND_SWELL_NOISE_HEIGHT * GROUND_SWELL_STEEPNESS * GROUND_SWELL_NOISE[j]
         * groundNoise(float2(1.3f, -2.1f) / GROUND_SWELL_NOISE[j] + float2(11.3f + 6.1f * float(j), -5.7f + 2.3f * float(j))).x;
  EXPECT_NEAR(w.height, h, 1e-6f);
}

// What DLSS is told a pixel of water moves by: the way of the waves its footprint shows
TEST(GroundWaves, TheLookGoesWithTheWavesAFootprintShows)
{
  const float  heading = 0.3f;
  const float2 way(std::cos(heading), std::sin(heading));
  groundWaterMotion(5.0f, heading, 0.01f);
  // near, on water with no swell: the ripples' drift
  const float2 ripples = groundPatternVelocity(0.001f, 0.0f, 1.0f, 1.0f);
  EXPECT_NEAR(ripples.x, GROUND_RIPPLE_DRIFT * way.x, 1e-6f);
  EXPECT_NEAR(ripples.y, GROUND_RIPPLE_DRIFT * way.y, 1e-6f);
  // with no ripples: the trains' mean, downwind, no faster than the longest train
  const float2 swell = groundPatternVelocity(0.001f, 1.0f, 0.0f, 1.0f);
  EXPECT_GT(dot(swell, way), 1.0f);
  EXPECT_LT(length(swell), groundWaveSpeed(GROUND_SWELL_LENGTH[0]));
  EXPECT_LT(std::abs(swell.x * way.y - swell.y * way.x), 0.3f * length(swell));
  // a swell laid out four times as long goes twice as fast
  const float2 sea = groundPatternVelocity(0.001f, 1.0f, 0.0f, 4.0f);
  EXPECT_NEAR(sea.x, 2.0f * swell.x, 1e-4f);
  EXPECT_NEAR(sea.y, 2.0f * swell.y, 1e-4f);
  // both: the ripples', alone, while they show whole (they are the finest of the look, and theirs
  // is the one motion that is exact)
  const float2 both = groundPatternVelocity(0.001f, 1.0f, 1.0f, 4.0f);
  EXPECT_NEAR(both.x, ripples.x, 1e-6f);
  EXPECT_NEAR(both.y, ripples.y, 1e-6f);
  // ... going over to the swell's as a footprint averages them out, and the swell's from there
  const float2 fading = groundPatternVelocity(0.1f, 1.0f, 1.0f, 4.0f);
  EXPECT_GT(dot(fading, way), GROUND_RIPPLE_DRIFT + 0.1f);
  EXPECT_LT(dot(fading, way), dot(sea, way) - 0.1f);
  const float2 far = groundPatternVelocity(1.0f, 1.0f, 1.0f, 4.0f);
  EXPECT_GT(dot(far, way), 2.0f);
  EXPECT_NEAR(far.x, groundPatternVelocity(1.0f, 1.0f, 0.0f, 4.0f).x, 1e-6f);
  EXPECT_EQ(groundPatternVelocity(100.0f, 1.0f, 1.0f, 4.0f), float2(0.0f, 0.0f));
  EXPECT_EQ(groundPatternVelocity(0.001f, 0.0f, 0.0f, 1.0f), float2(0.0f, 0.0f));

  // in the scene, since the frame before: the ground's water along x and z ...
  SceneFrameInfo flat{};
  flat.flags         = int(SceneFrameInfoFlags::eSceneUseInfinitePlane);
  flat.waterDepth    = 2.0f;
  flat.waterWaves    = 0.0f;
  flat.groundRipples = 1.0f;
  const float3 moved = waterPatternStep(&flat, float3(3.0f, 2.0f, -1.0f), 0.001f, false);
  EXPECT_NEAR(moved.x, GROUND_RIPPLE_DRIFT * way.x * 0.01f, 1e-7f);
  EXPECT_EQ(moved.y, 0.0f);
  EXPECT_NEAR(moved.z, GROUND_RIPPLE_DRIFT * way.y * 0.01f, 1e-7f);
  // ... a planet's along its surface, by the scene on the plane the scene lies in, and round its
  // side on that side's plane; a lake's swell is calmer, so its look is more its ripples'
  SceneFrameInfo planet{};
  planet.worldSeaRadius  = 20000.0f;
  planet.worldSeaTop     = float3(0.0f, -12.0f, 0.0f);
  planet.worldSeaWaves   = 0.0f;
  planet.worldSeaRipples = 1.0f;
  const Sea    water = worldSea(&planet);
  const float3 top   = waterPatternStep(&planet, water.top + float3(200.0f, -1.0f, 100.0f), 0.001f, false);
  EXPECT_NEAR(top.x, GROUND_RIPPLE_DRIFT * way.x * 0.01f, 1e-5f);
  EXPECT_NEAR(top.z, GROUND_RIPPLE_DRIFT * way.y * 0.01f, 1e-5f);
  EXPECT_LT(std::abs(top.y), 1e-4f);
  const float3 centre = water.top - float3(0.0f, water.radius, 0.0f);
  const float3 side   = waterPatternStep(&planet, centre + float3(water.radius, 0.0f, 0.0f), 0.001f, false);
  EXPECT_NEAR(length(side), GROUND_RIPPLE_DRIFT * 0.01f, 1e-5f);
  EXPECT_LT(std::abs(side.x), 1e-5f);  // (along the surface there: y and z)
  // far off, where only the swell shows, a lake's goes as the sea's does (calmer, not slower)
  planet.worldSeaWaves = 1.0f;
  EXPECT_GT(length(waterPatternStep(&planet, water.top, 1.0f, false)), 2.0f * 0.01f);
  EXPECT_EQ(waterPatternStep(&planet, water.top, 1.0f, true), waterPatternStep(&planet, water.top, 1.0f, false));
  // in a calm there is no step
  groundWaterMotion(0.0f, GROUND_SWELL_WIND);
  EXPECT_EQ(waterPatternStep(&planet, water.top, 0.001f, false), float3(0.0f));
}

// A planet's lakes and rivers are the same water standing higher (worldSeaAt): what a shadow ray
// from under one keeps is what it would keep under a sea at that level, while the path is in it
TEST(Sea, InlandWaterIsTheSeaStandingHigher)
{
  SceneFrameInfo frame{};
  frame.worldSeaRadius  = 20000.0f;
  frame.worldSeaTop     = float3(0.0f, -12.0f, 0.0f);
  frame.worldSeaWaves   = 0.6f;
  frame.worldSeaRipples = 0.5f;
  // no slot is its lakes' until the host says one, and no mesh without a material is
  EXPECT_FALSE(worldWaterIs(&frame, -1));
  EXPECT_FALSE(worldWaterIs(&frame, 3));
  frame.worldWater = 3;
  EXPECT_TRUE(worldWaterIs(&frame, 3));
  EXPECT_FALSE(worldWaterIs(&frame, 4));

  const Sea sea = worldSea(&frame), same = worldSeaAt(&frame, 0.0f), lake = worldSeaAt(&frame, 50.0f);
  EXPECT_EQ(same.radius, sea.radius);
  EXPECT_EQ(same.top, sea.top);
  EXPECT_EQ(same.waves, sea.waves);
  EXPECT_FLOAT_EQ(lake.radius, 20050.0f);
  EXPECT_FLOAT_EQ(lake.top.y, 38.0f);
  EXPECT_FLOAT_EQ(lake.waves, 0.6f * SEA_INLAND_WAVES);  // calmer than the sea
  EXPECT_EQ(lake.ripples, sea.ripples);
  // a point 49 m over the sea is a metre under that lake's surface, 300 m from the planet's top
  const float3 at = sea.top + float3(300.0f, 49.0f - 300.0f * 300.0f / 40000.0f, 0.0f);
  EXPECT_FALSE(seaUnder(sea, at));
  EXPECT_TRUE(seaUnder(lake, at));
  EXPECT_NEAR(seaHeight(lake, at), -1.0f, 0.02f);

  const float3 toSun = normalize(float3(0.3f, 1.0f, 0.2f));
  // a path in no water, or in the sea: the ray from there is in the air
  g_waterLevel = 0.0f;
  EXPECT_EQ(waterShadowTransmittance(at, toSun, 1e30f, &frame), float3(1.0f));
  // in the lake: it crosses the lake's surface, with what the surface reflects and the water takes
  g_waterLevel        = 50.0f;
  const float3 through = waterShadowTransmittance(at, toSun, 1e30f, &frame);
  EXPECT_EQ(through, seaShadowTransmittance(lake, at, toSun, 1e30f));
  EXPECT_LT(through.x, through.z);  // red goes first
  EXPECT_GT(through.z, 0.0f);
  // over the lake's surface the ray is in the air again
  EXPECT_EQ(waterShadowTransmittance(at + float3(0.0f, 2.0f, 0.0f), toSun, 1e30f, &frame), float3(1.0f));
  g_waterLevel = 0.0f;
}

// Their mesh is the water's surface to a ray that comes to it from the other side of it, and no
// surface otherwise (worldInlandHit)
TEST(Sea, ALakesMeshIsItsSurfaceFromTheOtherSideOnly)
{
  SceneFrameInfo frame{};
  frame.worldSeaRadius = 20000.0f;
  frame.worldSeaTop    = float3(0.0f, -12.0f, 0.0f);
  const Sea    sea     = worldSea(&frame);
  const float3 at      = sea.top + float3(300.0f, 49.0f, 0.0f);  // over the sea
  const float3 up      = seaUp(sea, at);
  const float3 side    = normalize(cross(up, float3(0.0f, 0.0f, 1.0f)));
  const float3 down    = normalize(-up + 0.4f * side), rising = normalize(up + 0.4f * side);

  // from the air down onto it, and from the water up to it: its surface, the normal turned to the
  // ray whichever way round the face is
  for(const float3 face : {up, -up})
  {
    const InlandHit in = worldInlandHit(sea, at, down, face, false);
    EXPECT_TRUE(in.fromAbove);
    EXPECT_FALSE(in.passed);
    EXPECT_NEAR(dot(in.normal, up), 1.0f, 1e-5f);
    const InlandHit out = worldInlandHit(sea, at, rising, face, true);
    EXPECT_FALSE(out.fromAbove);
    EXPECT_FALSE(out.passed);
    EXPECT_NEAR(dot(out.normal, up), -1.0f, 1e-5f);
  }
  // met from the side the path is on already: no surface
  EXPECT_TRUE(worldInlandHit(sea, at, down, up, true).passed);
  EXPECT_TRUE(worldInlandHit(sea, at, rising, up, false).passed);

  // a wall of the mesh's skirt: nothing to a path in the water, either way it goes; the surface,
  // level, to one that came down a crack from the air
  EXPECT_TRUE(worldInlandHit(sea, at, down, side, true).passed);
  EXPECT_TRUE(worldInlandHit(sea, at, rising, side, true).passed);
  const InlandHit crack = worldInlandHit(sea, at, down, side, false);
  EXPECT_TRUE(crack.fromAbove);
  EXPECT_FALSE(crack.passed);
  EXPECT_NEAR(dot(crack.normal, up), 1.0f, 1e-5f);
  // a river's water running downhill is no wall
  const float3 sloped = normalize(up + 0.3f * side);
  EXPECT_FALSE(worldInlandHit(sea, at, down, sloped, false).passed);
  EXPECT_NEAR(dot(worldInlandHit(sea, at, down, sloped, false).normal, sloped), 1.0f, 1e-5f);

  // the end of a river's mesh under the sea, at its mouth: the sea's own surface is over it
  const float3 under = sea.top + float3(300.0f, -300.0f * 300.0f / 40000.0f - 0.2f, 0.0f);
  ASSERT_TRUE(seaUnder(sea, under));
  EXPECT_TRUE(worldInlandHit(sea, under, down, up, false).passed);
  EXPECT_TRUE(worldInlandHit(sea, under, rising, up, true).passed);
}

// ---- a planet's ground (shaders/world_ground.h.slang) --------------------------------------------

namespace {

SceneFrameInfo groundWithMaps()
{
  SceneFrameInfo frame{};
  frame.worldGround          = 7;
  frame.worldGroundLayer     = int4(10, 11, 12, 13);                    // grass, a wood's floor, rock, sand
  frame.worldGroundTile      = float4(1.0f / 2.5f, 0.5f, 0.25f, 0.5f);  // repeats a metre
  frame.worldGroundLayerMore = int4(14, 15, 16, -1);                    // snow, dry earth, stony soil; no pebbles
  frame.worldGroundTileMore  = float4(0.5f, 0.25f, 0.5f, 0.5f);
  return frame;
}

// A place with shares of the first four layers only
int worldGroundLayer(float4 shares, float pick)
{
  return ::worldGroundLayer(shares, float4(0, 0, 0, 0), pick);
}
WorldGroundHit worldGroundAt(SceneFrameInfo* frame, float3 pos, float3 normal, float4 shares, float footprint)
{
  return ::worldGroundAt(frame, pos, normal, shares, float4(0, 0, 0, 0), footprint);
}

}  // namespace

TEST(WorldGround, OnlyThePlanetsGroundIsDrawnByIt)
{
  SceneFrameInfo none{};
  EXPECT_FALSE(worldGroundIs(&none, 0));  // no planet, or one without maps: no slot is its ground
  EXPECT_FALSE(worldGroundIs(&none, 7));
  SceneFrameInfo frame = groundWithMaps();
  EXPECT_TRUE(worldGroundIs(&frame, 7));
  EXPECT_FALSE(worldGroundIs(&frame, 10));
}

TEST(WorldGround, ThePickIsTheSameAtAPlaceAndSpreadOverTheGround)
{
  EXPECT_EQ(worldGroundPick(float3(3.1f, -2.0f, 8.4f)), worldGroundPick(float3(3.1f, -2.0f, 8.4f)));
  // smooth: a centimetre on, nearly the same
  const float4 here = worldGroundPick(float3(3.1f, -2.0f, 8.4f)), near = worldGroundPick(float3(3.11f, -2.0f, 8.4f));
  for(int c = 0; c < 4; c++)
    EXPECT_LT(std::abs(here[c] - near[c]), 0.1f);
  // over a field, about as often anywhere from 0 to 1, each of the four, and no two together
  int    low[4] = {}, high[4] = {}, n = 0;
  double sum[4] = {}, both[4] = {};
  for(int i = 0; i < 200; i++)
    for(int j = 0; j < 200; j++, n++)
    {
      const float4 p = worldGroundPick(float3(-800.0f + 0.37f * float(i), 12.0f, 300.0f + 0.41f * float(j)));
      for(int c = 0; c < 4; c++)
      {
        ASSERT_GE(p[c], 0.0f);
        ASSERT_LE(p[c], 1.0f);
        low[c] += p[c] < 0.25f;
        high[c] += p[c] > 0.75f;
        sum[c] += p[c];
        both[c] += (p[c] - 0.5) * (p[(c + 1) % 4] - 0.5);
      }
    }
  for(int c = 0; c < 4; c++)
  {
    EXPECT_NEAR(sum[c] / n, 0.5, 0.05);
    EXPECT_NEAR(double(low[c]) / n, 0.25, 0.08);
    EXPECT_NEAR(double(high[c]) / n, 0.25, 0.08);
    EXPECT_LT(std::abs(both[c] / n), 0.02);
  }
}

TEST(WorldGround, TheLayersPastTheFourthAreItsTextureCoordinates)
{
  const float4 none(0, 0, 0, 0);
  for(const float pick : {0.0f, 0.5f, 1.0f})
    for(int l = 0; l < 4; l++)
    {
      float4 more(0, 0, 0, 0);
      more[l] = 1.0f;
      EXPECT_EQ(worldGroundLayer(none, more, pick), 4 + l);
    }
  // rock gives way to the stones under it, pebbles to a shore's sand
  EXPECT_EQ(worldGroundLayer(float4(0, 0, 0.5f, 0), float4(0, 0, 0.5f, 0), 0.2f), 2);
  EXPECT_EQ(worldGroundLayer(float4(0, 0, 0.5f, 0), float4(0, 0, 0.5f, 0), 0.8f), 6);
  EXPECT_EQ(worldGroundLayer(float4(0, 0, 0, 0.5f), float4(0, 0, 0, 0.5f), 0.2f), 7);
  EXPECT_EQ(worldGroundLayer(float4(0, 0, 0, 0.5f), float4(0, 0, 0, 0.5f), 0.8f), 3);
  // all eight make the whole place between them
  EXPECT_EQ(worldGroundLayer(float4(0.1f, 0.1f, 0.1f, 0.1f), float4(0.1f, 0.1f, 0.1f, 0.1f), 0.97f), -1);
  EXPECT_EQ(worldGroundLayer(float4(0.125f, 0.125f, 0.125f, 0.125f), float4(0.125f, 0.125f, 0.125f, 0.125f), 0.97f), 0);
  // drawn with its own material and maps
  SceneFrameInfo       frame = groundWithMaps();
  const float3         up(0, 1, 0);
  const WorldGroundHit snow  = worldGroundAt(&frame, float3(3, 0, 4), up, none, float4(1, 0, 0, 0), 0.0f);
  ASSERT_TRUE(snow.mapped);
  EXPECT_EQ(snow.material, 14);
  EXPECT_TRUE(snow.texelDensity == 0.5f || snow.texelDensity == 0.5f * WORLD_GROUND_SECOND_SCALE);
  const WorldGroundHit dry = worldGroundAt(&frame, float3(3, 0, 4), up, none, float4(0, 1, 0, 0), 0.0f);
  EXPECT_EQ(dry.material, 15);
  EXPECT_TRUE(dry.texelDensity == 0.25f || dry.texelDensity == 0.25f * WORLD_GROUND_SECOND_SCALE);
  // a layer the kit has no maps for is the ground's own material
  EXPECT_FALSE(worldGroundAt(&frame, float3(3, 0, 4), up, none, float4(0, 0, 0, 1), 0.0f).mapped);
}

TEST(WorldGround, ALayerHasItsShareOfThePlace)
{
  // all of one: that one, wherever the noise is
  for(const float pick : {0.0f, 0.3f, 1.0f})
  {
    EXPECT_EQ(worldGroundLayer(float4(1, 0, 0, 0), pick), 0);
    EXPECT_EQ(worldGroundLayer(float4(0, 0, 1, 0), pick), 2);
    EXPECT_EQ(worldGroundLayer(float4(0, 0, 0, 0), pick), -1);  // none: the ground's own material
  }
  // rock first along the noise, then sand, a wood's floor, grass, and what they leave
  const float4 mixed(0.2f, 0.2f, 0.2f, 0.2f);
  EXPECT_EQ(worldGroundLayer(mixed, 0.1f), 2);
  EXPECT_EQ(worldGroundLayer(mixed, 0.3f), 3);
  EXPECT_EQ(worldGroundLayer(mixed, 0.5f), 1);
  EXPECT_EQ(worldGroundLayer(mixed, 0.7f), 0);
  EXPECT_EQ(worldGroundLayer(mixed, 0.95f), -1);
  // a trace of a layer shows nowhere, though the noise is at its end over some of the ground; nor
  // does the ground's own where a byte's rounding leaves a trace of it
  EXPECT_EQ(worldGroundLayer(float4(0.99f, 0, 0.01f, 0), 0.0f), 0);
  EXPECT_EQ(worldGroundLayer(float4(0.6f, 0, 0.392f, 0), 1.0f), 0);
  // over a field of the noise a layer has about its share
  int rock = 0, n = 0;
  for(int i = 0; i < 150; i++)
    for(int j = 0; j < 150; j++, n++)
      rock += worldGroundLayer(float4(0.7f, 0, 0.3f, 0), worldGroundPick(float3(40.0f + 0.3f * float(i), 0.0f, 0.3f * float(j))).x) == 2;
  EXPECT_NEAR(double(rock) / n, 0.3, 0.08);
}

TEST(WorldGround, ItsMapsAreLaidOnThePlaneItFaces)
{
  // level ground on y's plane, a cliff's face on the plane it faces, whichever way it faces it
  for(const float pick : {0.0f, 0.5f, 1.0f})
  {
    EXPECT_EQ(worldGroundPlane(normalize(float3(0.1f, 1.0f, -0.2f)), pick), 1);
    EXPECT_EQ(worldGroundPlane(normalize(float3(-1.0f, 0.2f, 0.1f)), pick), 0);
    EXPECT_EQ(worldGroundPlane(normalize(float3(0.3f, -0.1f, -1.0f)), pick), 2);
    // 35 degrees from level is still level ground's plane
    EXPECT_EQ(worldGroundPlane(normalize(float3(0.7f, 1.0f, 0.0f)), pick), 1);
  }
  // where it faces two alike, either, by the noise
  EXPECT_EQ(worldGroundPlane(normalize(float3(1.0f, 1.0f, 0.0f)), 0.2f), 1);
  EXPECT_EQ(worldGroundPlane(normalize(float3(1.0f, 1.0f, 0.0f)), 0.8f), 0);

  SceneFrameInfo frame = groundWithMaps();
  // Sand, level. A map lies one of two ways, by the noise: a centimetre along x is so much of a
  // repeat along u and none along v, or, turned a quarter and a little larger, the other way round.
  const float3 up(0, 1, 0);
  const float4 sand(0, 0, 0, 1);
  int          first = 0, second = 0;
  for(int i = 0; i < 400; i++)
  {
    const float3         at(10.0f + 0.83f * float(i), 0.0f, 20.0f - 0.61f * float(i));
    const WorldGroundHit a = worldGroundAt(&frame, at, up, sand, 0.0f);
    const WorldGroundHit b = worldGroundAt(&frame, at + float3(0.01f, 0, 0), up, sand, 0.0f);
    const WorldGroundHit c = worldGroundAt(&frame, at + float3(0, 0, 0.01f), up, sand, 0.0f);
    ASSERT_TRUE(a.mapped);
    EXPECT_EQ(a.material, 13);
    if(b.texelDensity != a.texelDensity || c.texelDensity != a.texelDensity)
      continue;  // the three are not in one patch
    const float tol = 2e-4f;  // (a float's worth of a coordinate tens of repeats out)
    if(a.texelDensity == 0.5f)
    {
      first++;
      EXPECT_NEAR(b.uv.x - a.uv.x, 0.005f, tol);
      EXPECT_NEAR(b.uv.y - a.uv.y, 0.0f, tol);
      EXPECT_NEAR(c.uv.y - a.uv.y, 0.005f, tol);
      // the frame is the map's: u along x, and up the map against v, which runs along z
      EXPECT_GT(a.tangent.x, 0.99f);
      EXPECT_LT(a.bitangent.z, -0.99f);
    }
    else
    {
      second++;
      EXPECT_FLOAT_EQ(a.texelDensity, 0.5f * WORLD_GROUND_SECOND_SCALE);
      EXPECT_NEAR(b.uv.y - a.uv.y, -0.005f * WORLD_GROUND_SECOND_SCALE, tol);
      EXPECT_NEAR(b.uv.x - a.uv.x, 0.0f, tol);
      EXPECT_NEAR(c.uv.x - a.uv.x, 0.005f * WORLD_GROUND_SECOND_SCALE, tol);
      EXPECT_GT(a.tangent.z, 0.99f);
      EXPECT_GT(a.bitangent.x, 0.99f);
    }
  }
  EXPECT_GT(first, 100);
  EXPECT_GT(second, 100);

  // Rock on a face that looks along +x: the map's top is up either way it lies, so v falls as the
  // face is climbed, and the frame is the map's: the tangent along u, the bitangent up the map, both
  // in the face
  const float3 out = normalize(float3(1.0f, 0.15f, 0.1f));
  const float4 rock(0, 0, 1, 0);
  for(int i = 0; i < 50; i++)
  {
    const float3         at(5.0f, 3.0f + 1.3f * float(i), 7.0f + 2.9f * float(i));
    const WorldGroundHit f = worldGroundAt(&frame, at, out, rock, 0.0f);
    const WorldGroundHit g = worldGroundAt(&frame, at + float3(0, 0.01f, 0), out, rock, 0.0f);
    const WorldGroundHit h = worldGroundAt(&frame, at + float3(0, 0, 0.01f), out, rock, 0.0f);
    ASSERT_TRUE(f.mapped);
    EXPECT_EQ(f.material, 12);
    if(g.texelDensity != f.texelDensity || h.texelDensity != f.texelDensity)
      continue;
    EXPECT_TRUE(f.texelDensity == 0.25f || f.texelDensity == 0.25f * WORLD_GROUND_SECOND_SCALE);
    EXPECT_NEAR(g.uv.y - f.uv.y, -0.01f * f.texelDensity, 2e-4f);
    EXPECT_NEAR(h.uv.x - f.uv.x, 0.01f * f.texelDensity, 2e-4f);
    EXPECT_NEAR(length(f.tangent), 1.0f, 1e-5f);
    EXPECT_NEAR(length(f.bitangent), 1.0f, 1e-5f);
    EXPECT_NEAR(dot(f.tangent, out), 0.0f, 1e-5f);
    EXPECT_NEAR(dot(f.bitangent, out), 0.0f, 1e-5f);
    EXPECT_NEAR(dot(f.tangent, f.bitangent), 0.0f, 1e-5f);
    EXPECT_GT(f.tangent.z, 0.9f);    // along u, which runs along z on this plane
    EXPECT_GT(f.bitangent.y, 0.9f);  // up the map: against v
  }

  // a place with no layer, or a layer the kit has no maps for, is the ground's own material
  EXPECT_FALSE(worldGroundAt(&frame, float3(1, 2, 3), up, float4(0, 0, 0, 0), 0.0f).mapped);
  frame.worldGroundLayer = int4(10, 11, -1, 13);
  const WorldGroundHit bare = worldGroundAt(&frame, float3(5, 3, 7), out, rock, 0.0f);
  EXPECT_FALSE(bare.mapped);
  EXPECT_EQ(bare.material, 7);
}

// From far, a map is laid larger (distance tiling) so that a repeat stays tens of pixels across, and read at
// the footprint's mip; past the largest the ground is its own material again
TEST(WorldGround, FromFarItsMapsAreLaidLargerThenItIsItsOwnMaterial)
{
  // a repeat forty pixels across or more: as laid near, everywhere
  for(const float pick : {0.0f, 0.5f, 1.0f})
  {
    EXPECT_EQ(worldGroundFarStep(0.0f, 0.5f, pick), 0);
    EXPECT_EQ(worldGroundFarStep(0.04f, 0.5f, pick), 0);  // sand's two metres from 40 m at a milliradian a pixel
    EXPECT_EQ(worldGroundFarStep(0.08f, 0.25f, pick), 0);  // rock's four metres are seen from twice as far
  }
  // between a fortieth and an eighth of a repeat, near in patches, larger in the rest
  EXPECT_EQ(worldGroundFarStep(0.15f, 0.5f, 0.2f), 1);
  EXPECT_EQ(worldGroundFarStep(0.15f, 0.5f, 0.8f), 0);
  // whatever the footprint and the noise: the fewest steps that leave the footprint within its share of a
  // repeat, and none past the last
  for(float footprint = 1e-3f; footprint < 1e3f; footprint *= 1.37f)
    for(const float pick : {0.0f, 0.3f, 0.7f, 1.0f})
    {
      const float most = WORLD_GROUND_FADE_FROM + (WORLD_GROUND_FADE_TO - WORLD_GROUND_FADE_FROM) * pick;
      const int   step = worldGroundFarStep(footprint, 0.4f, pick);
      SCOPED_TRACE(testing::Message() << footprint << " " << pick << " " << step);
      ASSERT_GE(step, -1);
      ASSERT_LE(step, WORLD_GROUND_FAR_STEPS);
      if(step < 0)
      {
        EXPECT_GT(footprint * 0.4f / std::pow(WORLD_GROUND_FAR_SCALE, float(WORLD_GROUND_FAR_STEPS)), most);
        continue;
      }
      EXPECT_LE(footprint * 0.4f / std::pow(WORLD_GROUND_FAR_SCALE, float(step)), most * 1.0001f);
      if(step > 0)
        EXPECT_GT(footprint * 0.4f / std::pow(WORLD_GROUND_FAR_SCALE, float(step - 1)), most);
    }

  // laid larger: its repeats a metre as many times fewer, and a centimetre along so much less of one
  SceneFrameInfo frame = groundWithMaps();
  const float3   up(0, 1, 0);
  const float4   sand(0, 0, 0, 1);
  int            steps[WORLD_GROUND_FAR_STEPS + 1] = {};
  for(int i = 0; i < 400; i++)
  {
    const float3         at(3.0f + 0.9f * float(i), 0.0f, -2.0f * float(i));
    const WorldGroundHit g = worldGroundAt(&frame, at, up, sand, 0.15f);
    ASSERT_TRUE(g.mapped);
    ASSERT_GE(g.farStep, 0);
    ASSERT_LE(g.farStep, 1);
    steps[g.farStep]++;
    const float laid = 0.5f / std::pow(WORLD_GROUND_FAR_SCALE, float(g.farStep));
    EXPECT_TRUE(g.texelDensity == laid || g.texelDensity == laid * WORLD_GROUND_SECOND_SCALE) << g.texelDensity;
    const WorldGroundHit b = worldGroundAt(&frame, at + float3(0.01f, 0, 0), up, sand, 0.15f);
    if(b.texelDensity == g.texelDensity && b.farStep == g.farStep)
      EXPECT_NEAR(std::abs(b.uv.x - g.uv.x) + std::abs(b.uv.y - g.uv.y), 0.01f * g.texelDensity, 2e-4f);
  }
  EXPECT_GT(steps[0], 100);
  EXPECT_GT(steps[1], 100);
  // read at the footprint's mip where it is laid larger, as the frame says where it isn't
  EXPECT_EQ(worldGroundTexGradScale(0, 0.0f), 0.0f);
  EXPECT_EQ(worldGroundTexGradScale(0, 0.5f), 0.5f);
  EXPECT_EQ(worldGroundTexGradScale(1, 0.0f), 1.0f);
  EXPECT_EQ(worldGroundTexGradScale(WORLD_GROUND_FAR_STEPS, 0.0f), 1.0f);

  // the last step at a footprint of 2.2 repeats (a 64th of one laid largest): there in most patches, the
  // ground's own material in the rest; past it, the ground's own material everywhere
  int last = 0;
  for(int i = 0; i < 200; i++)
  {
    const WorldGroundHit g = worldGroundAt(&frame, float3(1.3f * float(i), 0.0f, -0.7f * float(i)), up, sand, 4.4f);
    if(g.mapped)
    {
      EXPECT_EQ(g.farStep, WORLD_GROUND_FAR_STEPS);
      last++;
    }
  }
  EXPECT_GT(last, 100);
  EXPECT_FALSE(worldGroundAt(&frame, float3(3, 0, 4), up, sand, 1000.0f).mapped);
  EXPECT_EQ(worldGroundAt(&frame, float3(3, 0, 4), up, sand, 1000.0f).farStep, 0);
}
