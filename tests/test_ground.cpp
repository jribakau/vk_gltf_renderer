// The ground's shadow test (shaders/ground.h.slang) compiled as C++: a shadow ray that crosses the
// floor from above is blocked by as much of it as a path's own ray would hit there.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

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
