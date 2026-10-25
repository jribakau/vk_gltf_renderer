// The path tracer's aperture sampler (shaders/aperture.h.slang) compiled as C++: a round aperture
// samples as samplePixel always did, and a bladed one stays inside its polygon and covers it evenly.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cmath>
#include <random>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/aperture.h.slang"

namespace {

constexpr float kPi = 3.14159265358979f;

}  // namespace

// Below 3 blades it is the disc of the same two random numbers.
TEST(Aperture, ARoundApertureIsTheDiscItWas)
{
  std::mt19937                          rng(3);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(int i = 0; i < 1000; i++)
  {
    const float  u1 = u(rng), u2 = u(rng);
    const float  phi = u1 * 2.0f * kPi;
    const float2 p   = sampleAperture(u1, u2, 0, 0.7f);
    EXPECT_NEAR(p.x, std::cos(phi) * std::sqrt(u2), 1e-6f);
    EXPECT_NEAR(p.y, std::sin(phi) * std::sqrt(u2), 1e-6f);
  }
}

// A corner sits at the rotation, on the unit circle.
TEST(Aperture, TheFirstCornerIsAtTheRotation)
{
  const float2 p = sampleAperture(0.0f, 1.0f, 6, 0.4f);
  EXPECT_NEAR(p.x, std::cos(0.4f), 1e-6f);
  EXPECT_NEAR(p.y, std::sin(0.4f), 1e-6f);
}

// Every sample is inside the polygon, and as many fall inside its inscribed circle as its share of
// the area says: uniform over the polygon, not crowded toward its centre or its corners.
TEST(Aperture, BladesCoverTheirPolygonEvenly)
{
  std::mt19937                          rng(5);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for(int blades : {3, 5, 6, 9})
  {
    const float rotation = 0.3f;
    const float apothem  = std::cos(kPi / blades);
    const float area     = 0.5f * blades * std::sin(2.0f * kPi / blades);
    const int   N        = 200000;
    int         inside   = 0;
    for(int i = 0; i < N; i++)
    {
      const float2 p = sampleAperture(u(rng), u(rng), blades, rotation);
      for(int k = 0; k < blades; k++)
      {
        // the edge's outward normal points between its two corners
        const float mid = rotation + (k + 0.5f) * 2.0f * kPi / blades;
        ASSERT_LE(p.x * std::cos(mid) + p.y * std::sin(mid), apothem + 1e-5f) << blades;
      }
      inside += dot(p, p) <= apothem * apothem;
    }
    const float expected = kPi * apothem * apothem / area;
    EXPECT_NEAR(float(inside) / N, expected, 0.005f) << blades;
  }
}
