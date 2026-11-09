// The one meaning a hair colour has: the colour a dense lock of the fibres shows, by Chiang, Bitterli,
// Tappan and Burley 2016's fit, colour = exp(-sqrt(sigma) fit(b)) at azimuthal roughness b. OmniHair's
// dyes become absorption through it (as Iray's OmniHairBase.mdl does), a hair surface's swatch is its
// absorption read back through it, and a recolour moves the absorption so its colour reaches the
// target. The hair shader means the same (hair_bsdf.h.slang: hairColorFit, hairColorToAbsorption,
// hairLockColor; tests/test_hair_bsdf.cpp holds the two to each other).
//
// Until 2026-11-09 the swatch and a recolour read the fit with its exponent scaled by 0.45, a stand-in
// for the light a raster ribbon's fibres passed on to one another, so a colour picked through them
// meant 4.9 times the absorption a DAZ dye of that colour did.

#pragma once

#include <algorithm>
#include <cmath>

namespace fv::mat {

// Chiang et al.'s fit, by azimuthal roughness (clamped to 0..1, as the shader's)
inline float hairColorFit(float azimuthal)
{
  const float b = std::clamp(azimuthal, 0.f, 1.f);
  return 5.969f - 0.215f * b + 2.532f * b * b - 10.73f * b * b * b + 5.574f * b * b * b * b + 0.245f * b * b * b * b * b;
}

// The colour a lock of fibres of absorption `sigma` shows
inline float hairLockColor(float sigma, float azimuthal)
{
  return std::exp(-std::sqrt(std::max(sigma, 0.f)) * hairColorFit(azimuthal));
}

// The absorption whose lock shows `colour`: white absorbs nothing, and black is held at 1e-3
inline float hairAbsorption(float colour, float azimuthal)
{
  const float l = std::log(std::clamp(colour, 1e-3f, 1.f)) / hairColorFit(azimuthal);
  return l * l;
}

}  // namespace fv::mat
