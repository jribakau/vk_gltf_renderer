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
