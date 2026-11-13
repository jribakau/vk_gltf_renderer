// The physical sky's moon (src/moon.hpp): its light against the sun's by phase, and how its face is
// lit, checked against what is measured of the real one.

#include <gtest/gtest.h>

#include <cmath>

#include "moon.hpp"

namespace M = fv::moon;

TEST(Moon, ThePhaseAngleIsTheSkysAngleBetweenThemFromTheOtherSide)
{
  const glm::vec3 east(1.0F, 0.0F, 0.0F), west(-1.0F, 0.0F, 0.0F), up(0.0F, 1.0F, 0.0F);
  EXPECT_NEAR(M::phaseAngle(east, west), 0.0, 1e-4);   // opposite the sun: full
  EXPECT_NEAR(M::phaseAngle(east, east), 180.0, 1e-4);  // beside it: new
  EXPECT_NEAR(M::phaseAngle(east, up), 90.0, 1e-4);     // a quarter
}

TEST(Moon, ItsLightFallsWithThePhaseAsMeasured)
{
  // a full moon, 0.32 lux above the air, against the sun's 127,000
  EXPECT_DOUBLE_EQ(M::lightShare(0.0), M::kFullShare);
  EXPECT_NEAR(M::kFullShare * 1.27e5, 0.32, 0.01);
  // a quarter is about a tenth of a full moon's light, not half: the opposition surge
  const double quarter = M::lightShare(90.0) / M::lightShare(0.0);
  EXPECT_GT(quarter, 0.07);
  EXPECT_LT(quarter, 0.12);
  // a thin crescent, under a hundredth; new, none
  EXPECT_LT(M::lightShare(150.0) / M::lightShare(0.0), 0.01);
  EXPECT_EQ(M::lightShare(180.0), 0.0);
  for(double a = 0.0; a < 175.0; a += 5.0)
    EXPECT_GT(M::lightShare(a), M::lightShare(a + 5.0)) << a;
}

TEST(Moon, AFullMoonIsAsBrightAtItsEdgeAsInItsMiddle)
{
  // Lommel-Seeliger: head-on, the sun behind the eye, every point of the disk is cosI / 2cosI
  EXPECT_DOUBLE_EQ(M::faceShade(0.9, 0.9), 0.5);
  EXPECT_DOUBLE_EQ(M::faceShade(0.1, 0.1), 0.5);
  EXPECT_EQ(M::faceShade(-0.2, 0.5), 0.0);
  EXPECT_NEAR(M::meanFaceShade(0.0), 0.5, 2e-3);
  // half of it dark at a quarter, all at new; the mean falls in between
  EXPECT_LT(M::meanFaceShade(90.0), 0.25);
  EXPECT_GT(M::meanFaceShade(90.0), 0.1);
  EXPECT_NEAR(M::meanFaceShade(180.0), 0.0, 1e-6);
  for(double a = 0.0; a < 170.0; a += 10.0)
    EXPECT_GT(M::meanFaceShade(a), M::meanFaceShade(a + 10.0)) << a;
}

TEST(Moon, EarthshineIsACrescentsAndFaint)
{
  EXPECT_DOUBLE_EQ(M::earthshineShare(0.0), 0.0);
  EXPECT_NEAR(M::earthshineShare(180.0), M::kEarthshine, 1e-12);
  EXPECT_GT(M::earthshineShare(150.0), M::earthshineShare(90.0));
  // a crescent's dark side is thousands of times fainter than a full moon
  EXPECT_LT(M::kEarthshine, 1e-3);
}
