// Russian roulette on a path's next surface bounce (shaders/path_roulette.h.slang): where a still's
// and a DLSS frame's start, and that starting it at the first bounce keeps the light a path brings
// while ending the paths through dark surfaces sooner.

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <nvshaders/slang_types.h>

#include "../shaders/path_roulette.h.slang"

TEST(PathRoulette, APathGoesOnWithTheLightItCarries)
{
  EXPECT_FLOAT_EQ(pathContinueChance(0.15F, 1.0F), 0.151F);
  EXPECT_FLOAT_EQ(pathContinueChance(0.9F, 1.0F), 0.901F);
  EXPECT_FLOAT_EQ(pathContinueChance(2.0F, 1.0F), 1.0F);   // one that gained light is never ended
  EXPECT_FLOAT_EQ(pathContinueChance(0.9F, 0.85F), 0.85F);  // a cap (a clear fibre's)
  EXPECT_GT(pathContinueChance(0.0F, 1.0F), 0.0F);          // a path carrying nothing may still go on, rarely
  // the still as it always was; a DLSS frame's cut-out surfaces from the first bounce
  EXPECT_EQ(PATH_ROULETTE_FROM_STILL, 3);
  EXPECT_EQ(PATH_ROULETTE_FROM_CUTOUT_DLSS, 1);
}

namespace {

struct Estimate
{
  double mean     = 0.0;
  double error    = 0.0;  // standard error of the mean
  double vertices = 0.0;  // surface vertices a path reached, on average
};

// A path through a scene where every surface reflects `albedo` of the light and shows 1 of its own,
// to `maxDepth` vertices, with the path tracer's roulette from `from`: the light it brings is
// sum(albedo^k) for k below maxDepth, whatever the roulette.
Estimate walk(float albedo, int from, int maxDepth, int paths, uint32_t seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> u(0.0F, 1.0F);
  double                                sum = 0.0, sumSq = 0.0, vertices = 0.0;
  for(int p = 0; p < paths; p++)
  {
    float  throughput = 1.0F;
    double light      = 0.0;
    for(int depth = 0; depth < maxDepth; depth++)
    {
      vertices += 1.0;
      light += throughput;
      throughput *= albedo;
      if(depth >= from && depth + 1 < maxDepth)
      {
        const float chance = pathContinueChance(throughput, 1.0F);
        if(chance < 1.0F)
        {
          if(u(rng) >= chance)
            break;
          throughput /= chance;
        }
      }
    }
    sum += light;
    sumSq += light * light;
  }
  const double mean = sum / paths;
  return {mean, std::sqrt(std::max(sumSq / paths - mean * mean, 0.0) / paths), vertices / paths};
}

}  // namespace

// A fir's needles reflect about a sixth of the light: from the first bounce, a path through a canopy
// ends after two vertices on average instead of four, and brings the same light
TEST(PathRoulette, FromTheFirstBounceTheSameLightWithShorterPaths)
{
  for(const float albedo : {0.15F, 0.5F, 0.9F})
  {
    double exact = 0.0;
    for(int k = 0; k < 8; k++)
      exact += std::pow(double(albedo), k);
    const Estimate still  = walk(albedo, PATH_ROULETTE_FROM_STILL, 8, 400000, 1);
    const Estimate moving = walk(albedo, PATH_ROULETTE_FROM_CUTOUT_DLSS, 8, 400000, 2);
    EXPECT_NEAR(still.mean, exact, 4.0 * still.error + 1e-9) << albedo;
    EXPECT_NEAR(moving.mean, exact, 4.0 * moving.error + 1e-9) << albedo;
    EXPECT_LE(moving.vertices, still.vertices) << albedo;
    if(albedo < 0.2F)
    {
      EXPECT_LT(still.vertices, 4.1);
      EXPECT_GT(still.vertices, 3.9);
      EXPECT_LT(moving.vertices, 2.3);
    }
    // a bright surface (a white wall, skin under a sun) loses few paths to it
    if(albedo > 0.8F)
      EXPECT_GT(moving.vertices, 0.75 * still.vertices);
  }
}
