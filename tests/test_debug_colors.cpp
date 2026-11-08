// The Display views' colours (shaders/debug_colors.h.slang): an id's colour tells it from its
// neighbours, viridis brightens all the way, and the Density view's ramp is a tenth for each
// subdivision level. The shaders run this header.

#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <nvshaders/slang_types.h>

#include "../shaders/debug_colors.h.slang"

using shaderio::idColor;
using shaderio::triangleSizeRamp;
using shaderio::viridis;

namespace {

float distance(glm::vec3 a, glm::vec3 b)
{
  return glm::length(a - b);
}

float luminance(glm::vec3 c)
{
  return glm::dot(c, glm::vec3(0.2126F, 0.7152F, 0.0722F));
}

}  // namespace

// A figure's surfaces and meshes have neighbouring indices: each is far from the next, and from the
// one after; every colour is in range and none is black.
TEST(DebugColors, AnIdsColourIsFarFromItsNeighbours)
{
  float nearest = 1e9F;
  for(uint32_t id = 0; id < 4096; id++)
  {
    const glm::vec3 c = idColor(id);
    for(int k = 0; k < 3; k++)
    {
      EXPECT_GE(c[k], 0.0F);
      EXPECT_LE(c[k], 1.0F);
    }
    EXPECT_GT(luminance(c), 0.08F) << id;
    nearest = std::min({nearest, distance(c, idColor(id + 1)), distance(c, idColor(id + 2))});
  }
  EXPECT_GT(nearest, 0.25F);
  // a prop's parts (renderNodeOwner: its prop times 1024 plus its part) the same way
  EXPECT_GT(distance(idColor(3u * 1024u + 1u), idColor(3u * 1024u + 2u)), 0.25F);
}

// Viridis runs from dark violet to yellow, brighter at every step, and holds at its ends
TEST(DebugColors, ViridisBrightensAllTheWay)
{
  EXPECT_LT(distance(viridis(0.0F), glm::vec3(0.267F, 0.004F, 0.329F)), 1e-6F);
  EXPECT_LT(distance(viridis(1.0F), glm::vec3(0.992F, 0.906F, 0.145F)), 1e-6F);
  EXPECT_LT(distance(viridis(0.5F), glm::vec3(0.129F, 0.569F, 0.549F)), 1e-6F);
  EXPECT_EQ(viridis(-1.0F), viridis(0.0F));
  EXPECT_EQ(viridis(2.0F), viridis(1.0F));
  float last = -1.0F;
  for(int i = 0; i <= 100; i++)
  {
    const float l = luminance(viridis(float(i) / 100.0F));
    EXPECT_GT(l, last) << i;
    last = l;
  }
}

// Half a millimetre to half a metre over the ramp, a subdivision level (half the side) a tenth of it
TEST(DebugColors, TheDensityRampIsATenthALevel)
{
  EXPECT_NEAR(triangleSizeRamp(std::exp2(-11.0F)), 0.0F, 1e-6F);
  EXPECT_NEAR(triangleSizeRamp(0.5F), 1.0F, 1e-6F);
  EXPECT_NEAR(triangleSizeRamp(0.004F) - triangleSizeRamp(0.002F), 0.1F, 1e-5F);
  EXPECT_EQ(triangleSizeRamp(1e-6F), 0.0F);
  EXPECT_EQ(triangleSizeRamp(10.0F), 1.0F);
  EXPECT_EQ(triangleSizeRamp(0.0F), 0.0F);
}
