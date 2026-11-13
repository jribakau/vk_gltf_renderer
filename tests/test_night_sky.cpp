// The physical sky's night (shaders/night_sky.h.slang, src/night_stars.cpp, fv_lighting.cpp
// celestialAxes): the catalogue's stars where the sky puts them, their colours and their grid, the
// sky turned by the clock, the Milky Way on the galaxy's plane, and the air's glow.

#include <gtest/gtest.h>

#include <cmath>

#include "fv_lighting.hpp"
#include "night_stars.hpp"

using namespace glm;
using namespace shaderio;
#include "shaders/night_sky.h.slang"

namespace N = fv::night;

namespace {
constexpr float kDeg = 3.14159265F / 180.0F;
}

TEST(NightSky, TheCatalogueHoldsTheEyesStarsBrightestFirst)
{
  const auto stars = N::catalog();
  EXPECT_GT(stars.size(), 8000U);
  // Sirius, -1.46, in Canis Major
  EXPECT_NEAR(stars[0].v, -1.46F, 1e-3F);
  EXPECT_NEAR(stars[0].ra, 101.287F, 1e-2F);
  EXPECT_NEAR(stars[0].dec, -16.716F, 1e-2F);
  for(const N::CatalogStar& s : stars)
    EXPECT_LE(s.v, 6.5F);
  // a zero-magnitude star gives some 2.5e-6 lux, against the sun's 1.27e5
  EXPECT_NEAR(starLightShare(0.0F) / 2.0e-11F, 1.0F, 0.02F);
}

TEST(NightSky, AStarsColourIsItsTemperaturesAndTheSunsIsWhite)
{
  const auto lum = [](vec3 c) { return 0.2126F * c.x + 0.7152F * c.y + 0.0722F * c.z; };
  const vec3 sunlike = N::starColor(0.65F);
  EXPECT_NEAR(sunlike.x, 1.0F, 0.03F);
  EXPECT_NEAR(sunlike.z, 1.0F, 0.03F);
  const vec3 blue = N::starColor(-0.2F);  // Rigel's kind
  const vec3 red  = N::starColor(1.6F);   // Betelgeuse's
  EXPECT_GT(blue.z, blue.x);
  EXPECT_GT(red.x, 1.5F * red.z);
  for(float bv : {-0.3F, 0.0F, 0.65F, 1.2F, 2.0F})
    EXPECT_NEAR(lum(N::starColor(bv)), 1.0F, 1e-4F) << bv;
}

TEST(NightSky, EveryStarIsInTheCellARayToItLooksIn)
{
  const N::StarTables t = N::buildStarTables(N::catalog());
  ASSERT_EQ(t.stars.size(), N::catalog().size());
  ASSERT_EQ(t.cells.size(), size_t(NIGHT_STAR_CELLS_RA) * NIGHT_STAR_CELLS_DEC);
  size_t counted = 0;
  for(int row = 0; row < NIGHT_STAR_CELLS_DEC; ++row)
    for(int column = 0; column < NIGHT_STAR_CELLS_RA; ++column)
    {
      const shaderio::NightStarCell c = t.cells[size_t(row) * NIGHT_STAR_CELLS_RA + column];
      counted += c.count;
      for(uint32_t k = 0; k < c.count; ++k)
      {
        const vec3 d = t.stars[c.first + k].direction;
        EXPECT_EQ(nightStarRow(d), row);
        EXPECT_EQ(nightStarColumn(d), column);
      }
    }
  EXPECT_EQ(counted, t.stars.size());
}

TEST(NightSky, TheClockTurnsTheSkyAboutThePole)
{
  fv::lighting::Night n;
  n.latitude = 45.0;
  n.turn     = 30.0;
  for(double lst : {0.0, 75.0, 200.0})
  {
    n.siderealTime = lst;
    const auto a   = fv::lighting::celestialAxes(n);
    // a rotation: unit and square to one another
    for(int i = 0; i < 3; ++i)
    {
      EXPECT_NEAR(length(a[i]), 1.0F, 1e-5F);
      EXPECT_NEAR(dot(a[i], a[(i + 1) % 3]), 0.0F, 1e-5F);
    }
    // the pole stands at the latitude's height, opposite the meridian's south, whatever the hour
    EXPECT_NEAR(std::asin(a[2].y), 45.0F * kDeg, 1e-4F);
    EXPECT_NEAR(dot(normalize(vec3(a[2].x, 0.0F, a[2].z)), fv::lighting::sunDirection(30.0 + 180.0, 0.0)), 1.0F, 1e-4F);
    // the equator's point on the meridian (right ascension = sidereal time) stands due south at 90 - 45
    const float r  = float(lst) * kDeg;
    const vec3  on = a[0] * std::cos(r) + a[1] * std::sin(r);
    EXPECT_NEAR(std::asin(on.y), 45.0F * kDeg, 1e-4F);
    EXPECT_NEAR(dot(normalize(vec3(on.x, 0.0F, on.z)), fv::lighting::sunDirection(30.0, 0.0)), 1.0F, 1e-4F);
  }
}

TEST(NightSky, TheMilkyWayLiesOnTheGalaxysPlane)
{
  const vec2 centre = galactic(galacticCentre());
  EXPECT_NEAR(centre.x, 0.0F, 1e-4F);
  EXPECT_NEAR(centre.y, 0.0F, 1e-4F);
  EXPECT_NEAR(galactic(galacticNorth()).y, 90.0F * kDeg, 1e-3F);  // asin's float, at its top
  // brightest at the centre, a quarter of it at the anticentre's plane, nothing far off it
  EXPECT_NEAR(milkyWayBand(0.0F, 0.0F), 2.2F, 1e-4F);
  EXPECT_NEAR(milkyWayBand(3.14159F, 0.0F), 0.25F, 1e-3F);
  EXPECT_LT(milkyWayBand(1.0F, 40.0F * kDeg), 1e-3F);
  EXPECT_GT(length(milkyWay(galacticCentre())), length(milkyWay(-galacticCentre())));
  EXPECT_EQ(length(milkyWay(galacticNorth())), 0.0F);
}

TEST(NightSky, TheGlowIsThickerTowardTheHorizon)
{
  const float R = 6360.0F;
  EXPECT_FLOAT_EQ(nightGlowPath(1.0F, R), 1.0F);
  EXPECT_NEAR(nightGlowPath(0.0F, R), 6.0F, 0.1F);  // van Rhijn's at the horizon, the layer 90 km up
  EXPECT_GT(nightGlowPath(0.3F, R), nightGlowPath(0.6F, R));
  const vec3 c = nightGlowColor();
  EXPECT_NEAR(0.2126F * c.x + 0.7152F * c.y + 0.0722F * c.z, 1.0F, 1e-3F);
}
