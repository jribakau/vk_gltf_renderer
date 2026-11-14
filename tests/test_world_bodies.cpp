//
// Other bodies round a planet (shaders/world_bodies.h.slang): where a ray meets one, what one leaves of
// the sun's disk, and how its surface is lit. The header is Slang and plain C++.
//

#include <cmath>

#include <glm/glm.hpp>
#include <gtest/gtest.h>

namespace world_bodies_test {

using namespace glm;
using float3 = glm::vec3;
using std::lerp;
#include "../shaders/world_bodies.h.slang"

constexpr float kSun = 0.004675F;  // the sun's disk, radians across the half

TEST(WorldBodies, ARayMeetsTheNearSideOfAFarSphere)
{
  // a moon 150 km off, 3.5 km in radius, straight ahead
  const float3 centre(0.0F, 0.0F, -150000.0F);
  EXPECT_NEAR(worldBodyHit(float3(0.0F), float3(0.0F, 0.0F, -1.0F), centre, 3500.0F), 146500.0F, 1.0F);
  // past its edge by a hundredth of its radius: a miss; inside it by as much: a hit
  for(const float edge : {0.99F, 1.01F})
  {
    const float3 dir = normalize(float3(3500.0F * edge, 0.0F, -sqrt(150000.0F * 150000.0F - 3500.0F * 3500.0F)));
    EXPECT_EQ(worldBodyHit(float3(0.0F), dir, centre, 3500.0F) > 0.0F, edge < 1.0F) << edge;
  }
  // behind the eye, and from inside it: nothing
  EXPECT_LT(worldBodyHit(float3(0.0F), float3(0.0F, 0.0F, 1.0F), centre, 3500.0F), 0.0F);
  EXPECT_LT(worldBodyHit(centre + float3(100.0F, 0.0F, 0.0F), float3(0.0F, 1.0F, 0.0F), centre, 3500.0F), 0.0F);
  // a planet ten thousand kilometres off: the edge is still where it is
  const float3 far(0.0F, 0.0F, -1.0e7F);
  const float  r = 6.0e5F;
  for(const float edge : {0.995F, 1.005F})
  {
    const float3 dir = normalize(float3(r * edge, 0.0F, -sqrt(1.0e14F - r * r)));
    EXPECT_EQ(worldBodyHit(float3(0.0F), dir, far, r) > 0.0F, edge < 1.0F) << edge;
  }
}

TEST(WorldBodies, ADiskHidesWhatOfAnotherItCovers)
{
  EXPECT_FLOAT_EQ(worldDiskCovered(1.0F, 1.0F, 2.5F), 0.0F);   // apart
  EXPECT_FLOAT_EQ(worldDiskCovered(1.0F, 2.0F, 0.5F), 1.0F);   // under a larger one
  EXPECT_FLOAT_EQ(worldDiskCovered(1.0F, 0.5F, 0.2F), 0.25F);  // a smaller one inside it: their areas
  // two alike, a radius apart: the lens, 2 pi / 3 - sqrt(3) / 2 of pi
  EXPECT_NEAR(worldDiskCovered(1.0F, 1.0F, 1.0F), (2.0F * 3.14159265F / 3.0F - 0.8660254F) / 3.14159265F, 1e-4F);
  // more as they close, without a jump at either end
  float was = 0.0F;
  for(int i = 0; i <= 200; i++)
  {
    const float s       = 2.2F - 2.2F * float(i) / 200.0F;
    const float covered = worldDiskCovered(1.0F, 1.2F, s);
    EXPECT_GE(covered, was - 1e-5F) << s;
    EXPECT_LT(covered - was, 0.03F) << s;
    was = covered;
  }
  EXPECT_FLOAT_EQ(was, 1.0F);
}

TEST(WorldBodies, ASphereTakesTheSunItStandsBefore)
{
  const float3 sun(0.0F, 0.0F, -1.0F);
  // a moon between the point and the sun, larger than the sun's disk: its full shadow
  EXPECT_FLOAT_EQ(worldSunPast(float3(0.0F), float3(0.0F, 0.0F, -150000.0F), 3500.0F, sun, kSun), 0.0F);
  // the same moon behind the point, and well to the side: all of the sun
  EXPECT_FLOAT_EQ(worldSunPast(float3(0.0F), float3(0.0F, 0.0F, 150000.0F), 3500.0F, sun, kSun), 1.0F);
  EXPECT_FLOAT_EQ(worldSunPast(float3(0.0F), float3(20000.0F, 0.0F, -150000.0F), 3500.0F, sun, kSun), 1.0F);
  // its edge on the sun's middle: about half, and more of the sun the further it moves off
  const float half = worldSunPast(float3(0.0F), float3(3500.0F, 0.0F, -150000.0F), 3500.0F, sun, kSun);
  EXPECT_NEAR(half, 0.5F, 0.05F);
  float was = 0.0F;
  for(int i = 0; i <= 100; i++)
  {
    const float x    = 2500.0F + 2500.0F * float(i) / 100.0F;
    const float left = worldSunPast(float3(0.0F), float3(x, 0.0F, -150000.0F), 3500.0F, sun, kSun);
    EXPECT_GE(left, was - 1e-4F) << x;
    EXPECT_LT(left - was, 0.06F) << x;  // a penumbra, not a step
    was = left;
  }
  EXPECT_FLOAT_EQ(was, 1.0F);
  // a moon smaller than the sun's disk leaves a ring of it
  const float ring = worldSunPast(float3(0.0F), float3(0.0F, 0.0F, -150000.0F), 350.0F, sun, kSun);
  EXPECT_NEAR(ring, 1.0F - (350.0F / 150000.0F) * (350.0F / 150000.0F) / (kSun * kSun), 0.01F);
}

TEST(WorldBodies, ABodyHasItsPhaseFromTheSunAndTheEye)
{
  const float3 eye(0.0F, 0.0F, 1.0F);
  // full: lit across, the limb no darker than half the middle
  const float middle = worldBodyShade(float3(0.0F, 0.0F, 1.0F), eye, float3(0.0F, 0.0F, 1.0F));
  const float limb   = worldBodyShade(normalize(float3(0.98F, 0.0F, 0.2F)), eye, float3(0.0F, 0.0F, 1.0F));
  EXPECT_GT(middle, 0.2F);
  EXPECT_GT(limb, 0.5F * middle);
  // half: the side away from the sun is dark, the terminator has no step
  const float3 sun(1.0F, 0.0F, 0.0F);
  EXPECT_FLOAT_EQ(worldBodyShade(normalize(float3(-0.5F, 0.0F, 0.87F)), eye, sun), 0.0F);
  EXPECT_LT(worldBodyShade(normalize(float3(0.01F, 0.0F, 1.0F)), eye, sun), 0.02F);
  EXPECT_GT(worldBodyShade(normalize(float3(0.7F, 0.0F, 0.7F)), eye, sun), 0.1F);
}

TEST(WorldBodies, MarkingsAverageAboutOneAndTellBodiesApart)
{
  for(const int kind : {WORLD_BODY_ROCK, WORLD_BODY_GAS})
  {
    double sum = 0.0, apart = 0.0;
    float  least = 10.0F, most = 0.0F;
    int    n = 0;
    for(int i = 0; i < 40; i++)
      for(int j = 0; j < 80; j++)
      {
        const float  y = -1.0F + 2.0F * (float(i) + 0.5F) / 40.0F;
        const float  a = 6.2831853F * (float(j) + 0.5F) / 80.0F;
        const float  s = sqrt(1.0F - y * y);
        const float3 dir(s * cos(a), y, s * sin(a));
        const float  m = worldBodyMarkings(dir, kind, 1.0F);
        sum += m;
        apart += std::abs(m - worldBodyMarkings(dir, kind, 2.0F));
        least = std::min(least, m);
        most  = std::max(most, m);
        n++;
      }
    EXPECT_NEAR(sum / n, 0.95, 0.2) << kind;
    EXPECT_GT(least, 0.3F) << kind;
    EXPECT_LT(most, 1.6F) << kind;
    EXPECT_GT(most - least, 0.3F) << kind;  // there is something to see
    if(kind == WORLD_BODY_ROCK)
      EXPECT_GT(apart / n, 0.05) << kind;   // another seed, another face
  }
}

}  // namespace world_bodies_test
