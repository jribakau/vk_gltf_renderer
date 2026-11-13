// Rain in the air (shaders/rain.h.slang, src/rain_cells.hpp) compiled as C++: the drops a rate holds
// give that rate back, they fall as fast as their size says, each stays in its cell and falls through
// it, and a frame draws the cells round the camera it can see.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "rain_cells.hpp"

using namespace glm;
#include "../shaders/shaderio.h"
using namespace shaderio;
#include "../shaders/rain.h.slang"

// Marshall and Palmer's sizes falling at Atlas's speeds carry the rate they were made for (within a
// fifth: the two were fitted apart), from drizzle to a downpour
TEST(Rain, TheDropsCarryTheirRate)
{
  for(float rate : {1.0f, 2.5f, 5.0f, 10.0f, 25.0f, 60.0f})
  {
    const float slope = rainSlope(rate);
    double      sum   = 0.0;
    const float step  = 0.005f;
    for(float d = 0.5f * step; d < 8.0f; d += step)
      sum += RAIN_N0 * std::exp(-slope * d) * double(d) * d * d * rainFallSpeed(d) * step;
    const double carried = 6.0 * 3.14159265 * 1e-4 * sum;  // mm/h
    EXPECT_NEAR(carried / rate, 1.0, 0.25) << rate;
  }
  // hundreds of drops a cubic metre at a moderate rate, more the harder it rains
  EXPECT_NEAR(rainDropsPerCubicMetre(5.0f, RAIN_MIN_DIAMETER), 634.0f, 10.0f);
  EXPECT_GT(rainDropsPerCubicMetre(25.0f, RAIN_MIN_DIAMETER), rainDropsPerCubicMetre(5.0f, RAIN_MIN_DIAMETER));
}

TEST(Rain, BiggerDropsFallFaster)
{
  EXPECT_NEAR(rainFallSpeed(1.0f), 4.0f, 0.1f);
  EXPECT_NEAR(rainFallSpeed(2.0f), 6.5f, 0.1f);
  EXPECT_NEAR(rainFallSpeed(5.0f), 9.1f, 0.1f);
  float last = 0.0f;
  for(float d = 0.5f; d <= RAIN_MAX_DIAMETER; d += 0.25f)
  {
    EXPECT_GT(rainFallSpeed(d), last);
    last = rainFallSpeed(d);
  }
}

// The sizes drawn follow the distribution from half a millimetre: their mean is its, and none is
// past the largest
TEST(Rain, TheSizesFollowTheDistribution)
{
  const float rate = 5.0f;
  double      sum  = 0.0;
  const int   n    = 20000;
  for(int i = 0; i < n; ++i)
  {
    const float d = rainDropDiameter(rate, (float(i) + 0.5f) / float(n));
    EXPECT_GE(d, RAIN_MIN_DIAMETER);
    EXPECT_LE(d, RAIN_MAX_DIAMETER);
    sum += d;
  }
  EXPECT_NEAR(sum / n, RAIN_MIN_DIAMETER + 1.0f / rainSlope(rate), 0.02);
}

// A drop stays in its cell's sheared prism, falls through it, and is where it was for the same cell
// and time (the rain is the world's)
TEST(Rain, ADropFallsThroughItsCell)
{
  const float  cell  = 1.0f;
  const float2 shear = rainShear(float2(3.0f, -2.0f));
  int          present = 0;
  for(uint i = 0; i < 400; ++i)
  {
    const int3     c = int3(4, -2, 7);
    const RainDrop a = rainDropAt(c, i, cell, shear, 10.0f, 0.5f, 1u, 5.0f);
    const RainDrop b = rainDropAt(c, i, cell, shear, 10.0f, 0.5f, 1u, 5.0f);
    EXPECT_EQ(a.position, b.position);
    present += a.present;
    EXPECT_GE(a.position.y, float(c.y) * cell);
    EXPECT_LT(a.position.y, float(c.y + 1) * cell);
    const float hx = a.position.x - shear.x * a.position.y;
    const float hz = a.position.z - shear.y * a.position.y;
    EXPECT_GE(hx, float(c.x) * cell - 1e-4f);
    EXPECT_LT(hx, float(c.x + 1) * cell + 1e-4f);
    EXPECT_GE(hz, float(c.z) * cell - 1e-4f);
    EXPECT_LT(hz, float(c.z + 1) * cell + 1e-4f);
    // a hundredth of a second later it is lower by its speed's worth, or came round to the top
    const RainDrop later = rainDropAt(c, i, cell, shear, 10.01f, 0.5f, 1u, 5.0f);
    const float    fell  = a.position.y - later.position.y;
    if(fell > 0.0f)
      EXPECT_NEAR(fell, a.speed * 0.01f, 2e-3f);
    else
      EXPECT_NEAR(fell, a.speed * 0.01f - cell, 2e-3f);
  }
  EXPECT_NEAR(present / 400.0f, 0.5f, 0.08f);
}

// The shelter map's texel under a point: its middle under the camera, off the map past its edge, and
// along the slant of the rain
TEST(Rain, TheShelterMapLiesAlongTheFall)
{
  const float2 centre(10.0f, -5.0f);
  const float2 shear(0.2f, 0.0f);
  const int    middle = (RAIN_SHELTER_SIZE / 2) * RAIN_SHELTER_SIZE + RAIN_SHELTER_SIZE / 2;
  EXPECT_EQ(rainShelterTexel(float3(10.0f, 0.0f, -5.0f), shear, centre), middle);
  // ten metres up, a drop over the same ground is two metres downwind
  EXPECT_EQ(rainShelterTexel(float3(12.0f, 10.0f, -5.0f), shear, centre), middle);
  EXPECT_EQ(rainShelterTexel(float3(10.0f + RAIN_SHELTER_EXTENT, 0.0f, -5.0f), shear, centre), -1);
}

// Where it lands: a drizzle wets, harder rain soaks and pools; up-facing surfaces take all of it, a
// wall half (more into the wind), an underside none, and nothing under a roof, though a wall isn't
// sheltered by its own top
TEST(Rain, ItWetsWhatItReaches)
{
  EXPECT_EQ(rainWetness(0.0f), 0.0f);
  EXPECT_NEAR(rainWetness(1.0f), 0.63f, 0.01f);
  EXPECT_GT(rainWetness(2.5f), 0.9f);
  EXPECT_EQ(rainPuddleShare(1.0f), 0.0f);
  EXPECT_EQ(rainPuddleShare(10.0f), 1.0f);

  const float2 still(0.0f);
  const float3 here(0.0f, 0.0f, 0.0f);
  EXPECT_EQ(rainExposureAt(here, float3(0.0f, 1.0f, 0.0f), still, float2(0.0f), nullptr), 1.0f);
  EXPECT_NEAR(rainExposureAt(here, float3(1.0f, 0.0f, 0.0f), still, float2(0.0f), nullptr), 0.5f, 1e-5f);
  EXPECT_EQ(rainExposureAt(here, float3(0.0f, -1.0f, 0.0f), still, float2(0.0f), nullptr), 0.0f);
  // a wind toward +x: a wall facing -x, into it, takes more than one facing away
  const float2 windy = rainShear(float2(5.0f, 0.0f));
  EXPECT_GT(rainExposureAt(here, float3(-1.0f, 0.0f, 0.0f), windy, float2(0.0f), nullptr),
            rainExposureAt(here, float3(1.0f, 0.0f, 0.0f), windy, float2(0.0f), nullptr));

  // a roof 3 m up over a 2 m square round the middle; a wall's face at the square's edge
  std::vector<float> shelter(RAIN_SHELTER_SIZE * RAIN_SHELTER_SIZE, RAIN_SHELTER_OPEN);
  for(int i = 0; i < RAIN_SHELTER_SIZE * RAIN_SHELTER_SIZE; ++i)
  {
    const float x = (float(i % RAIN_SHELTER_SIZE) + 0.5f) / RAIN_SHELTER_SIZE * RAIN_SHELTER_EXTENT - 0.5f * RAIN_SHELTER_EXTENT;
    const float z = (float(i / RAIN_SHELTER_SIZE) + 0.5f) / RAIN_SHELTER_SIZE * RAIN_SHELTER_EXTENT - 0.5f * RAIN_SHELTER_EXTENT;
    if(std::abs(x) < 1.0f && std::abs(z) < 1.0f)
      shelter[i] = 3.0f;
  }
  EXPECT_EQ(rainExposureAt(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), still, float2(0.0f), shelter.data()), 0.0f);
  EXPECT_EQ(rainExposureAt(float3(3.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), still, float2(0.0f), shelter.data()), 1.0f);
  // the roof's own top is in the rain, and the house's wall below it on the outside
  EXPECT_EQ(rainExposureAt(float3(0.0f, 3.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), still, float2(0.0f), shelter.data()), 1.0f);
  EXPECT_NEAR(rainExposureAt(float3(1.0f, 1.5f, 0.0f), float3(1.0f, 0.0f, 0.0f), still, float2(0.0f), shelter.data()), 0.5f, 1e-5f);
}

// A water sphere's phase without its diffraction peak integrates to 1 and is brightest toward the light
TEST(Rain, TheSunsPhaseIsNormalised)
{
  double    sum = 0.0;
  const int n   = 4000;
  for(int i = 0; i < n; ++i)
  {
    const float mu = -1.0f + 2.0f * (float(i) + 0.5f) / float(n);
    sum += rainSunPhase(mu) * 2.0 * 3.14159265 * (2.0 / n);
  }
  EXPECT_NEAR(sum, 1.0, 0.01);
  EXPECT_GT(rainSunPhase(1.0f), 50.0f * rainSunPhase(-1.0f));
}

// A frame draws the cells round the camera within its reach and in front of it, the nearest when
// there are more than it takes
TEST(Rain, AFrameDrawsTheCellsItSees)
{
  const glm::vec3 eye(0.0f, 1.6f, 0.0f);
  const glm::mat4 view = glm::lookAt(eye, eye + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(50.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
  const glm::vec2 shear(0.0f);
  const auto      cells = rain_cells::visible(proj * view, eye, shear, 1.0f, 12.0f, 100000);
  ASSERT_GT(cells.size(), 100u);
  for(const glm::ivec4& c : cells)
  {
    // in front: no cell wholly behind the eye, none past the reach
    EXPECT_LE(float(c.z), eye.z + 1.0f);
    const glm::vec3 middle = glm::vec3(c) + 0.5f;
    EXPECT_LT(glm::length(middle - eye), 12.0f + 1.0f);
  }
  const auto few = rain_cells::visible(proj * view, eye, shear, 1.0f, 12.0f, 50);
  ASSERT_EQ(few.size(), 50u);
  for(const glm::ivec4& c : few)
    EXPECT_LT(glm::length(glm::vec3(c) + 0.5f - eye), 6.0f);
}
