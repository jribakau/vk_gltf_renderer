// Iray Uber's metallic flakes (shaders/flake_noise.h.slang) compiled as C++: base.mdl's two lattice
// noises agree where they should, the flakes' weights and tilts spread as base.mdl's do, and a flake
// shows on as many cells as its weight asks.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <tuple>

using namespace glm;
#include "../shaders/shaderio.h"
using namespace shaderio;
#include "../shaders/flake_noise.h.slang"

TEST(FlakeNoise, TheCellNoiseIsTheSmoothNoiseAtTheCellsMiddle)
{
  // base.mdl's integer mi_noise is its float one half a cell back: the same spline, at its knot
  for(int x = -3; x < 3; x++)
    for(int y = -2; y < 2; y++)
      for(int z = -1; z < 1; z++)
      {
        const FlakeNoise cell   = flakeCellNoise(int3(x * 37, y * 11, z * 5));
        const FlakeNoise smooth = flakeSmoothNoise(float3(x * 37, y * 11, z * 5) - 0.5f);
        EXPECT_NEAR(cell.val, smooth.val, 1e-5f);
        EXPECT_NEAR(cell.grad.x, smooth.grad.x, 1e-5f);
        EXPECT_NEAR(cell.grad.y, smooth.grad.y, 1e-5f);
        EXPECT_NEAR(cell.grad.z, smooth.grad.z, 1e-5f);
      }
}

TEST(FlakeNoise, TheSmoothNoisesGradientIsItsSlope)
{
  const float h = 1e-3f;
  for(const float3 p : {float3(0.3f, 1.7f, 0.0f), float3(-4.2f, 9.1f, 0.6f), float3(12.5f, -3.3f, 2.2f)})
  {
    const FlakeNoise n = flakeSmoothNoise(p);
    EXPECT_NEAR(n.grad.x, (flakeSmoothNoise(p + float3(h, 0, 0)).val - flakeSmoothNoise(p - float3(h, 0, 0)).val) / (2 * h), 2e-3f);
    EXPECT_NEAR(n.grad.y, (flakeSmoothNoise(p + float3(0, h, 0)).val - flakeSmoothNoise(p - float3(0, h, 0)).val) / (2 * h), 2e-3f);
    EXPECT_NEAR(n.grad.z, (flakeSmoothNoise(p + float3(0, 0, h)).val - flakeSmoothNoise(p - float3(0, 0, h)).val) / (2 * h), 2e-3f);
  }
}

TEST(FlakeNoise, FlakesWeighAboutHalfAndTiltAFewDegrees)
{
  // a weighted mean of 27 bytes: near 0.5, never past 0..1; at strength 1 a tilt of about ten degrees
  double sum = 0, tilt = 0;
  int    n   = 0;
  for(int x = 0; x < 100; x++)
    for(int y = 0; y < 100; y++)
    {
      const Flake f = flakeAt(float2(x * 0.0137f, y * 0.0191f), 0.001f, 1.0f);
      EXPECT_GE(f.reflectivity, 0.0f);
      EXPECT_LT(f.reflectivity, 1.0f);
      sum += f.reflectivity;
      tilt += std::acos(std::min(1.0f, flakeNormal(f, float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), 1.0f).z));
      n++;
    }
  EXPECT_NEAR(sum / n, 0.5, 0.02);
  const double degrees = glm::degrees(tilt / n);
  EXPECT_GT(degrees, 3.0);
  EXPECT_LT(degrees, 25.0);
}

TEST(FlakeNoise, DensityThinsTheWeightAndStrengthTheTilt)
{
  const Flake full = flakeAt(float2(0.41f, 0.73f), 0.01f, 1.0f);
  const Flake thin = flakeAt(float2(0.41f, 0.73f), 0.01f, 0.25f);
  EXPECT_EQ(full.cell, thin.cell);
  EXPECT_NEAR(thin.reflectivity, std::pow(full.reflectivity, 4.0f), 1e-5f);
  const float3 N(0, 0, 1);
  const float  strong = flakeNormal(full, float3(1, 0, 0), float3(0, 1, 0), N, 1.0f).z;
  const float  weak   = flakeNormal(full, float3(1, 0, 0), float3(0, 1, 0), N, 0.1f).z;
  EXPECT_GT(weak, strong);
  EXPECT_NEAR(flakeNormal(full, float3(1, 0, 0), float3(0, 1, 0), N, 1e-6f).z, 1.0f, 1e-4f);
}

TEST(FlakeNoise, AFlakeIsOneCellOfARaggedGrid)
{
  // cells about `size` across: over a UV square of 50 sizes, on the order of 2500 of them (more, as
  // the push also moves a point across the cells' third axis, as it does in base.mdl)
  std::set<std::tuple<int, int, int>> cells;
  for(int x = 0; x < 200; x++)
    for(int y = 0; y < 200; y++)
    {
      const Flake f = flakeAt(float2(x, y) * (0.05f / 200.0f), 0.001f, 1.0f);
      cells.insert({f.cell.x, f.cell.y, f.cell.z});
    }
  EXPECT_GT(cells.size(), 1500u);
  EXPECT_LT(cells.size(), 8000u);
  // the same point is the same flake
  EXPECT_EQ(flakeAt(float2(0.25f, 0.5f), 0.001f, 1.0f).cell, flakeAt(float2(0.25f, 0.5f), 0.001f, 1.0f).cell);
}

TEST(FlakeNoise, AFlakeShowsOnAsManyCellsAsItsWeight)
{
  int shown = 0, all = 0, none = 0;
  const int cells = 20000;
  for(int i = 0; i < cells; i++)
  {
    Flake f;
    f.cell = int3(i % 141, i / 141, 0);
    shown += flakeShows(f, 0.3f);
    all += flakeShows(f, 1.0f);
    none += flakeShows(f, 0.0f);
  }
  EXPECT_NEAR(double(shown) / cells, 0.3, 0.015);
  EXPECT_EQ(all, cells);
  EXPECT_EQ(none, 0);
}
