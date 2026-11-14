#pragma once

//-------------------------------------------------------------------------------------------------------
// The physical sky over a small planet (the backdrop's `planet`, fv_world.hpp).
//
// The atmosphere's settings describe air over a planet thousands of kilometres across: 60 km deep,
// thinning by 1/e every 8. Over a planet 20 km in radius that much air is three times the planet. So
// the air is scaled with the planet: every height in it `airScale` times what it was, and every
// density 1/airScale times, which leaves the optical depth straight up as it was. The zenith is then
// the colour and brightness the settings describe; what changes is what a small planet changes: the
// horizon is nearer and the air there thinner, and the sky goes dark a few kilometres up.
//
// Pure, and free of the shaders' headers (the atmosphere's struct is a template's, so the backdrop's
// state can read the scale without them); tested in tests/test_sky_world.cpp.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>

namespace sky_world {

// How deep a planet's air is against the 60 km the atmosphere's settings are for: a quarter of its
// radius deep. Never deeper than they say.
inline float airScale(float radiusKm, float thicknessKm = 60.0F)
{
  return std::clamp(0.25F * radiusKm / std::max(thicknessKm, 1e-3F), 1e-3F, 1.0F);
}

// The lowest sun the tables cover, as the cosine of its angle from straight up: the sun that grazes
// the ground as the top of the air sees it, and a little past. Earth's -0.5 where that is enough,
// which it is for any planet much larger than its air is deep.
inline float lowestSun(float bottomRadius, float topRadius)
{
  const float ratio = bottomRadius / std::max(topRadius, 1e-6F);
  return std::clamp(std::min(-0.5F, -std::sqrt(std::max(1.0F - ratio * ratio, 0.0F)) - 0.05F), -0.98F, -0.5F);
}

// `p` (shaderio::SkyAtmosphereParameters), built for the atmosphere's own planet, over one `radiusKm`
// in radius with its air `scale` times as deep
template <class Atmosphere>
void apply(Atmosphere& p, float radiusKm, float scale)
{
  const float s     = std::clamp(scale, 1e-3F, 1.0F);
  const float depth = (p.topRadius - p.bottomRadius) * s;
  p.bottomRadius    = radiusKm;
  p.topRadius       = radiusKm + depth;
  p.rayleighScattering /= s;
  p.mieScattering /= s;
  p.mieExtinction /= s;
  p.absorptionExtinction /= s;
  for(auto* profile : {&p.rayleighDensity, &p.mieDensity, &p.absorptionDensity})
    for(auto& layer : profile->layers)
    {
      layer.width *= s;
      layer.expScale /= s;
      layer.linearTerm /= s;
    }
  p.muSMin = lowestSun(p.bottomRadius, p.topRadius);
}

}  // namespace sky_world
