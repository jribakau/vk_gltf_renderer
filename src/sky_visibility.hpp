#pragma once

//-------------------------------------------------------------------------------------------------------
// How far one sees through the physical sky's air, as a setting (atmoVisibility, the lighting's
// env.skyVisibility): the aerosol scaled until a dark object at that distance keeps 2% of its
// contrast against the horizon, Koschmieder's meteorological range, V = 3.912 / extinction. The
// extinction is the air's at the ground in the green, where the eye weighs most: Rayleigh's
// scattering and the aerosol's extinction (the ozone is kilometres up).
//
// The Earth preset's own air is about 49 km by this: a clear day. A summer haze is 10 to 20 km, and
// under 5 it is mist. The sky and the air between the camera and the land (aerial_perspective.h.slang)
// are one model, so a hazier air whitens the horizon, widens the glow round the sun, dims the sun
// itself and veils the far hills together.
//
// Vulkan-free; km and 1/km.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>

namespace sky_visibility {

inline constexpr float kKoschmieder = 3.912F;  // -ln(0.02)
// The aerosol is never scaled past these: below, Rayleigh alone already limits the range; above, the
// tables' march is no longer of a thin medium
inline constexpr float kLeast = 0.05F;
inline constexpr float kMost  = 200.0F;

// The visibility of an air with these at the ground, km
inline float of(float rayleighGreen, float aerosolExtinctionGreen)
{
  return kKoschmieder / std::max(rayleighGreen + aerosolExtinctionGreen, 1e-6F);
}

// A mist's droplets are water, which scatters what it meets and absorbs next to nothing: the share of
// the haze added past the atmosphere's own aerosol that is scattered (applyAerosolScale)
inline constexpr float kAddedHazeAlbedo = 0.99F;

// What to multiply the aerosol by for `visibilityKm`; 1 for 0 or less, which leaves the atmosphere as
// it is
inline float aerosolScale(float visibilityKm, float rayleighGreen, float aerosolExtinctionGreen)
{
  if(!(visibilityKm > 0.0F) || !(aerosolExtinctionGreen > 0.0F))
    return 1.0F;
  const float wanted = kKoschmieder / visibilityKm - rayleighGreen;
  return std::clamp(wanted / aerosolExtinctionGreen, kLeast, kMost);
}

// The aerosol for a scale (aerosolScale): clearer air is less of the same aerosol; hazier air is the
// atmosphere's own and the rest water haze (kAddedHazeAlbedo). Scaling Earth's own aerosol, dust that
// absorbs a tenth of what it scatters, past a summer haze turned kilometres of it brown and dark where
// a mist is white.
template <class Vec3>
inline void applyAerosolScale(float scale, Vec3& scattering, Vec3& extinction)
{
  if(scale <= 1.0F)
  {
    scattering *= scale;
    extinction *= scale;
    return;
  }
  const Vec3 added = extinction * (scale - 1.0F);
  scattering += added * kAddedHazeAlbedo;
  extinction += added;
}

}  // namespace sky_visibility
