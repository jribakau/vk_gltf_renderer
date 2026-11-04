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
