// The light groups' mix (shaders/light_mix.h.slang): a picture is the sum of what each light group gives
// it, so the mix of the whole picture and each group's share is each group's own light times its factor;
// and how a still's samples take turns noting one group's share each.

// GLM as test_bsdf_lobes.cpp, another file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/shaderio.h"
#include "../shaders/light_mix.h.slang"

namespace {

using Groups = std::array<float3, LIGHT_GROUP_COUNT>;
constexpr uint kAll = (1u << LIGHT_GROUP_COUNT) - 1u;

// The mix as the pass makes it (light_mix.slang), from the whole picture and each group's image
float3 mixed(float3 whole, const Groups& images, const Groups& mix, uint used = kAll)
{
  float3 floor(1e30f);
  for(int g = 0; g < LIGHT_GROUP_COUNT; g++)
    floor = lightMixFloor(floor, mix[g], ((used >> g) & 1u) != 0u);
  if(floor.x > 1e29f)
    floor = float3(1.0f);
  float3 terms(0.0f);
  for(int g = 0; g < LIGHT_GROUP_COUNT; g++)
    if((used >> g) & 1u)
      terms += lightMixTerm(images[g], mix[g], floor);
  return lightMixed(whole, floor, terms);
}

Groups randomLight(std::mt19937& rng)
{
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  Groups                                light;
  for(float3& l : light)
    l = u(rng) < 0.3f ? float3(0.0f) : float3(u(rng), u(rng), u(rng)) * std::exp2(8.0f * u(rng) - 4.0f);
  return light;
}

float3 sum(const Groups& light)
{
  float3 whole(0.0f);
  for(const float3& l : light)
    whole += l;
  return whole;
}

}  // namespace

// With every group at 1 the mix is the whole picture, bit for bit, whatever the groups' images hold
// (they are noisier than the whole: each is the mean of its own turns)
TEST(LightMix, NeutralIsTheWholePicture)
{
  std::mt19937 rng(3);
  Groups       neutral;
  neutral.fill(float3(1.0f));
  for(int i = 0; i < 1000; i++)
  {
    const float3 whole = sum(randomLight(rng));
    const float3 got   = mixed(whole, randomLight(rng), neutral);  // images of anything at all
    EXPECT_EQ(got.x, whole.x);
    EXPECT_EQ(got.y, whole.y);
    EXPECT_EQ(got.z, whole.z);
  }
}

// ...and under any mix it is each group's light times its own factor, where the images are the groups'
// light exactly
TEST(LightMix, EachGroupTakesItsOwnFactor)
{
  std::mt19937                          rng(5);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(int i = 0; i < 1000; i++)
  {
    const Groups light = randomLight(rng);
    Groups       mix;
    for(float3& m : mix)
      m = u(rng) < 0.2f ? float3(0.0f) : float3(u(rng), u(rng), u(rng)) * std::exp2(6.0f * u(rng) - 3.0f);
    float3 want(0.0f), size(0.0f);
    for(int g = 0; g < LIGHT_GROUP_COUNT; g++)
    {
      want += mix[g] * light[g];
      size += (mix[g] + float3(1.0f)) * light[g];
    }
    const float3 got = mixed(sum(light), light, mix);
    for(int c = 0; c < 3; c++)
      EXPECT_NEAR(got[c], want[c], 1e-4f * size[c] + 1e-6f) << i << " channel " << c;
  }
}

// The images are noisy (a group's is the mean of a K-th of the samples) and the whole is not: the mix
// is right on average, and never under zero, which a mix that took the sky as what the others leave of
// the whole was not (the sky alone came out a twentieth too bright, measured)
TEST(LightMix, NoisyImagesGiveTheRightMixOnAverageAndNeverLessThanNothing)
{
  std::mt19937                          rng(9);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const Groups                          light = {float3(0.05f), float3(0.3f), float3(0.0f), float3(1.0f), float3(0.4f), float3(0.0f)};
  const float3                          whole = sum(light);
  const std::array<Groups, 4>           mixes = {{
      {float3(1.0f), float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)},  // the sky alone
      {float3(1.0f), float3(1.0f), float3(1.0f), float3(2.0f), float3(1.0f), float3(1.0f)},  // the lamps a stop up
      {float3(0.5f), float3(1.0f), float3(1.0f), float3(1.0f), float3(0.0f), float3(1.0f)},  // the sky down, a group off
      {float3(1.0f), float3(1.0f, 0.5f, 0.2f), float3(1.0f), float3(1.0f), float3(1.0f), float3(1.0f)},  // the sun tinted
  }};
  for(const Groups& mix : mixes)
  {
    float3 want(0.0f);
    for(int g = 0; g < LIGHT_GROUP_COUNT; g++)
      want += mix[g] * light[g];
    dvec3     mean(0.0);
    const int n = 200000;
    for(int i = 0; i < n; i++)
    {
      // each image its light times a noise of mean 1, as wide as its light
      Groups images;
      for(int g = 0; g < LIGHT_GROUP_COUNT; g++)
        images[g] = light[g] * (2.0f * u(rng));
      const float3 got = mixed(whole, images, mix);
      ASSERT_GE(std::min(got.x, std::min(got.y, got.z)), 0.0f);
      mean += dvec3(got);
    }
    mean /= double(n);
    for(int c = 0; c < 3; c++)
      EXPECT_NEAR(mean[c], want[c], 0.01 * want[c] + 1e-4);
  }
}

// A group the scene has no light in does not hold the mix down: turned off, the picture is as it was
TEST(LightMix, AGroupWithNoLightDoesNotCount)
{
  std::mt19937 rng(11);
  const uint   used = (1u << LIGHT_GROUP_SKY) | (1u << LIGHT_GROUP_LAMPS);
  Groups       mix;
  mix.fill(float3(1.0f));
  mix[LIGHT_GROUP_GLOW]      = float3(0.0f);  // no glowing surface in the scene
  mix[LIGHT_GROUP_LAMPS + 2] = float3(4.0f);
  const float3 whole = float3(0.7f, 0.5f, 0.3f);
  const float3 got   = mixed(whole, randomLight(rng), mix, used);
  EXPECT_EQ(got.x, whole.x);
  EXPECT_EQ(got.y, whole.y);
  EXPECT_EQ(got.z, whole.z);
  // and with no group at all, the picture as it is
  const float3 none = mixed(whole, randomLight(rng), mix, 0u);
  EXPECT_EQ(none.x, whole.x);
}

// A sample carries one group's share: the groups the scene has light in take turns, evenly, from a
// start of the pixel's own, and a group's image is the mean of its own turns
TEST(LightMix, TheGroupsTakeTurns)
{
  // the sky, the sun, the lamps and the third lamp group have light
  const uint used = (1u << LIGHT_GROUP_SKY) | (1u << LIGHT_GROUP_SUN) | (1u << LIGHT_GROUP_LAMPS) | (1u << (LIGHT_GROUP_LAMPS + 2));
  EXPECT_EQ(lightGroupsInTurn(used), 4u);
  for(uint start : {0u, 1u, 7u, 0xdeadbeefu})
  {
    std::array<uint, LIGHT_GROUP_COUNT> turns{};
    for(uint sample = 0; sample < 400; sample++)
    {
      const uint group = lightGroupOfSample(used, sample, start);
      ASSERT_LT(group, uint(LIGHT_GROUP_COUNT));
      ASSERT_NE((used >> group) & 1u, 0u) << group;
      // the mean so far is of exactly the earlier samples of this group
      EXPECT_EQ(lightGroupTurnsBefore(used, sample), turns[group]) << sample;
      turns[group]++;
    }
    EXPECT_EQ(turns[LIGHT_GROUP_SKY], 100u);
    EXPECT_EQ(turns[LIGHT_GROUP_SUN], 100u);
    EXPECT_EQ(turns[LIGHT_GROUP_LAMPS], 100u);
    EXPECT_EQ(turns[LIGHT_GROUP_LAMPS + 2], 100u);
    EXPECT_EQ(turns[LIGHT_GROUP_GLOW] + turns[LIGHT_GROUP_LAMPS + 1], 0u);
  }
  // neighbours start at different turns, so a frame's samples are spread over the groups
  EXPECT_NE(lightGroupOfSample(used, 0, 0), lightGroupOfSample(used, 0, 1));
  // one group has every turn; no group, none
  EXPECT_EQ(lightGroupOfSample(1u << LIGHT_GROUP_GLOW, 17, 5), uint(LIGHT_GROUP_GLOW));
  EXPECT_EQ(lightGroupTurnsBefore(1u << LIGHT_GROUP_GLOW, 17), 17u);
  EXPECT_EQ(lightGroupOfSample(0u, 3, 9), uint(LIGHT_GROUP_COUNT));
}

// A pixel's start may be any number: three groups, which 2^32 is no multiple of, keep their turns where
// the start plus the sample would wrap
TEST(LightMix, TheTurnsGoRoundPastTheLargestStart)
{
  const uint used = (1u << LIGHT_GROUP_SKY) | (1u << LIGHT_GROUP_SUN) | (1u << LIGHT_GROUP_LAMPS);
  ASSERT_EQ(lightGroupsInTurn(used), 3u);
  for(uint start : {0xFFFFFFF0u, 0xFFFFFFFFu})
  {
    std::array<uint, LIGHT_GROUP_COUNT> turns{};
    for(uint sample = 0; sample < 300; sample++)
    {
      const uint group = lightGroupOfSample(used, sample, start);
      ASSERT_LT(group, uint(LIGHT_GROUP_COUNT));
      EXPECT_EQ(lightGroupTurnsBefore(used, sample), turns[group]) << sample << " from " << start;
      turns[group]++;
    }
    EXPECT_EQ(turns[LIGHT_GROUP_SKY], 100u);
    EXPECT_EQ(turns[LIGHT_GROUP_SUN], 100u);
    EXPECT_EQ(turns[LIGHT_GROUP_LAMPS], 100u);
  }
}
