// Auto white balance (src/white_balance.hpp): the white balance found for a light makes it grey through the
// tone mapper's own matrix, and the light found in a frame is the one it was lit by, whatever coloured
// surfaces and emitters the frame also holds.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "../src/white_balance.hpp"

#include <nvshaders/tonemap_functions.h.slang>  // getColorCorrectionMatrix

namespace {

using white_balance::estimateLight;
using white_balance::forLight;

// The light the tone mapper's white balance (kelvin, tint) takes to grey: the matrix's inverse on grey,
// in the C++ shim's c * M convention
glm::dvec3 lightOf(double kelvin, double tint)
{
  const glm::mat3 m = shaderio::getColorCorrectionMatrix(1.0F, float(kelvin), float(tint));
  const glm::dvec3 c = glm::dvec3(glm::vec3(1.0F) * glm::inverse(m));
  return c / (0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z);
}

double mireds(double kelvin)
{
  return 1e6 / kelvin;
}

// A frame of `count` pixels lit by `light`: surfaces of a mostly grey scene's reflectances, a share
// `red` of them a saturated red curtain, a share `neon` a blue sign a hundred times brighter than the
// rest; brightness varying as a lit room's does
std::vector<float> frame(const glm::dvec3& light, double red, double neon, uint32_t width, uint32_t height, uint32_t seed = 7)
{
  std::mt19937                           rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<float>                     img(size_t(width) * height * 4);
  for(size_t i = 0; i < size_t(width) * height; i++)
  {
    const double pick = u(rng);
    glm::dvec3   c;
    if(pick < neon)
      c = glm::dvec3(0.1, 0.3, 4.0) * 100.0;  // an emitter: its own colour, not the light's
    else
    {
      glm::dvec3 refl;
      if(pick < neon + red)
        refl = glm::dvec3(0.6, 0.05, 0.04);
      else
      {
        // near-grey surfaces with a little colour each way, averaging grey
        const double g = 0.1 + 0.7 * u(rng);
        refl           = g * glm::dvec3(1.0 + 0.15 * (u(rng) - 0.5), 1.0 + 0.15 * (u(rng) - 0.5), 1.0 + 0.15 * (u(rng) - 0.5));
      }
      const double lit = std::exp2(4.0 * (u(rng) - 0.5));  // two stops each way
      c                = refl * light * lit;
    }
    img[i * 4 + 0] = float(c.x);
    img[i * 4 + 1] = float(c.y);
    img[i * 4 + 2] = float(c.z);
    img[i * 4 + 3] = 1.0F;
  }
  return img;
}

}  // namespace

TEST(WhiteBalance, TheLightsWhiteBalanceIsTheOneItWasMadeFrom)
{
  for(double kelvin : {2200.0, 2700.0, 3200.0, 4000.0, 5500.0, 6500.0, 9000.0, 14000.0})
    for(double tint : {-0.02, 0.0, 0.00326, 0.02})
    {
      // a very warm magenta light is outside Rec. 709 (its blue below 0): no frame can hold it
      const glm::dvec3 light = lightOf(kelvin, tint);
      if(std::min(light.x, std::min(light.y, light.z)) <= 0.0)
        continue;
      const auto s = forLight(light);
      ASSERT_TRUE(s) << kelvin << " " << tint;
      EXPECT_NEAR(mireds(s->temperature), mireds(kelvin), 0.5) << kelvin << " " << tint;
      EXPECT_NEAR(s->tint, tint, 2e-4) << kelvin << " " << tint;
      EXPECT_FALSE(s->clamped);
    }
}

TEST(WhiteBalance, TheAnswerMakesTheLightGrey)
{
  // whatever the light, not only one on the locus: a sodium lamp's, a green fluorescent's
  for(const glm::dvec3 light : {glm::dvec3(1.0, 0.85, 0.5), glm::dvec3(0.9, 1.0, 0.85), glm::dvec3(0.8, 0.95, 1.3)})
  {
    const auto s = forLight(light);
    ASSERT_TRUE(s);
    ASSERT_FALSE(s->clamped);
    const glm::vec3 c = glm::vec3(light) * shaderio::getColorCorrectionMatrix(1.0F, float(s->temperature), float(s->tint));
    EXPECT_NEAR(c.x / c.y, 1.0, 1e-4);
    EXPECT_NEAR(c.z / c.y, 1.0, 1e-4);
  }
}

TEST(WhiteBalance, ALightPastTheRangeTakesItsEdge)
{
  // a candle's 1500 K is warmer than the renderer's 2000 K
  const auto candle = forLight(lightOf(2000.0, 0.0) * glm::dvec3(1.0, 0.8, 0.45));
  ASSERT_TRUE(candle);
  EXPECT_TRUE(candle->clamped);
  EXPECT_EQ(candle->temperature, white_balance::kMinKelvin);
  // a pure primary has no white balance
  EXPECT_FALSE(forLight(glm::dvec3(1.0, 0.0, 0.0)));
  EXPECT_FALSE(forLight(glm::dvec3(NAN, 1.0, 1.0)));
}

TEST(WhiteBalance, AFramesLightIsTheOneItWasLitBy)
{
  for(double kelvin : {2700.0, 4000.0, 6500.0, 10000.0})
  {
    const glm::dvec3   light = lightOf(kelvin, 0.0);
    std::vector<float> img   = frame(light, 0.0, 0.0, 640, 360);
    const auto         found = estimateLight(img.data(), 640, 360);
    ASSERT_TRUE(found);
    const auto s = forLight(*found);
    ASSERT_TRUE(s);
    EXPECT_NEAR(mireds(s->temperature), mireds(kelvin), 4.0) << kelvin;
    EXPECT_NEAR(s->tint, 0.0, 0.002) << kelvin;
  }
}

TEST(WhiteBalance, ARedCurtainAndANeonSignDontSteerIt)
{
  // under a warm 3000 K light, two fifths of the frame a red curtain and one percent a bright blue sign
  const glm::dvec3   light = lightOf(3000.0, 0.0);
  std::vector<float> img   = frame(light, 0.4, 0.01, 640, 360);
  const auto         s     = forLight(*estimateLight(img.data(), 640, 360));
  ASSERT_TRUE(s);
  EXPECT_NEAR(mireds(s->temperature), mireds(3000.0), 15.0);  // within about 140 K
  EXPECT_NEAR(s->tint, 0.0, 0.004);
  // a plain average would be far off: the curtain alone pulls it hundreds of kelvin warmer
  glm::dvec3 sum(0.0);
  for(size_t i = 0; i < img.size(); i += 4)
    if(img[i + 2] < 50.0F)
      sum += glm::dvec3(img[i], img[i + 1], img[i + 2]);
  const auto plain = forLight(sum);
  ASSERT_TRUE(plain);
  EXPECT_GT(std::abs(mireds(plain->temperature) - mireds(3000.0)), 40.0);
}

TEST(WhiteBalance, AnUnusableFrameHasNoLight)
{
  std::vector<float> black(64 * 64 * 4, 0.0F);
  EXPECT_FALSE(estimateLight(black.data(), 64, 64));
  std::vector<float> nan(64 * 64 * 4, NAN);
  EXPECT_FALSE(estimateLight(nan.data(), 64, 64));
  EXPECT_FALSE(estimateLight(nullptr, 0, 0));
}
