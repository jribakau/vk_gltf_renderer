// The physical sky over a small planet (src/sky_world.hpp): its air scaled with it, as deep straight up
// as the atmosphere it was made from.

#include <cmath>

#include <glm/glm.hpp>

#include <gtest/gtest.h>

#include "../shaders/shaderio.h"

#include "../src/sky_world.hpp"

namespace {

// Earth's, as GltfRenderer::buildAtmosphereParameters lays it out: an exponential gas in layer 1, and
// an ozone tent over two layers
shaderio::SkyAtmosphereParameters earth()
{
  shaderio::SkyAtmosphereParameters p{};
  p.bottomRadius                             = 6360.0F;
  p.topRadius                                = 6420.0F;
  p.rayleighScattering                       = {0.0058F, 0.0135F, 0.0331F};
  p.mieScattering                            = glm::vec3(0.004F);
  p.mieExtinction                            = glm::vec3(0.0044F);
  p.absorptionExtinction                     = {0.00065F, 0.0019F, 0.000085F};
  p.rayleighDensity.layers[1].expTerm        = 1.0F;
  p.rayleighDensity.layers[1].expScale       = -1.0F / 8.0F;
  p.mieDensity.layers[1].expTerm             = 1.0F;
  p.mieDensity.layers[1].expScale            = -1.0F / 1.2F;
  p.absorptionDensity.layers[0].width        = 25.0F;
  p.absorptionDensity.layers[0].linearTerm   = 1.0F / 15.0F;
  p.absorptionDensity.layers[0].constantTerm = -2.0F / 3.0F;
  p.absorptionDensity.layers[1].linearTerm   = -1.0F / 15.0F;
  p.absorptionDensity.layers[1].constantTerm = 8.0F / 3.0F;
  p.muSMin                                   = -0.5F;
  return p;
}

// How much of a gas a ray straight up goes through, to the top of the air: its density integrated
float column(const shaderio::DensityProfile& profile, float depth)
{
  float       sum   = 0.0F;
  const int   steps = 4000;
  const float dh    = depth / float(steps);
  for(int i = 0; i < steps; i++)
  {
    const float                          h = (float(i) + 0.5F) * dh;
    const shaderio::DensityProfileLayer& l = h < profile.layers[0].width ? profile.layers[0] : profile.layers[1];
    sum += std::clamp(l.expTerm * std::exp(l.expScale * h) + l.linearTerm * h + l.constantTerm, 0.0F, 1.0F) * dh;
  }
  return sum;
}

}  // namespace

TEST(SkyWorld, APlanetsAirIsAQuarterOfItsRadiusDeep)
{
  EXPECT_NEAR(sky_world::airScale(20.0F), 5.0F / 60.0F, 1e-6F);
  EXPECT_NEAR(sky_world::airScale(2.0F), 0.5F / 60.0F, 1e-6F);
  // and never deeper than the atmosphere's own
  EXPECT_EQ(sky_world::airScale(6360.0F), 1.0F);
}

TEST(SkyWorld, TheScaledAirIsAsDeepStraightUp)
{
  const shaderio::SkyAtmosphereParameters was = earth();
  shaderio::SkyAtmosphereParameters       p   = was;
  const float                             s   = sky_world::airScale(20.0F);
  sky_world::apply(p, 20.0F, s);
  EXPECT_EQ(p.bottomRadius, 20.0F);
  EXPECT_NEAR(p.topRadius - p.bottomRadius, 5.0F, 1e-4F);
  // each gas: its density at the ground times how much of it is overhead, as it was
  const float depthWas = was.topRadius - was.bottomRadius, depth = p.topRadius - p.bottomRadius;
  EXPECT_NEAR(p.rayleighScattering.y * column(p.rayleighDensity, depth), was.rayleighScattering.y * column(was.rayleighDensity, depthWas), 1e-4F);
  EXPECT_NEAR(p.mieExtinction.y * column(p.mieDensity, depth), was.mieExtinction.y * column(was.mieDensity, depthWas), 1e-5F);
  EXPECT_NEAR(p.absorptionExtinction.y * column(p.absorptionDensity, depth), was.absorptionExtinction.y * column(was.absorptionDensity, depthWas), 1e-4F);
  // the aerosol's albedo is as it was
  EXPECT_NEAR(p.mieScattering.y / p.mieExtinction.y, was.mieScattering.y / was.mieExtinction.y, 1e-5F);
}

TEST(SkyWorld, TheTablesCoverTheSunThatGrazesASmallPlanet)
{
  // seen from the top of the air the ground's horizon is far under the level: the tables' lowest sun
  // is past it, where Earth's -0.5 would cut the dusk off
  shaderio::SkyAtmosphereParameters p = earth();
  sky_world::apply(p, 20.0F, sky_world::airScale(20.0F));
  const float horizon = -std::sqrt(1.0F - (p.bottomRadius / p.topRadius) * (p.bottomRadius / p.topRadius));
  EXPECT_LT(p.muSMin, horizon);
  EXPECT_GE(p.muSMin, -0.98F);
  // Earth's own is left as it is
  EXPECT_EQ(sky_world::lowestSun(6360.0F, 6420.0F), -0.5F);
}
