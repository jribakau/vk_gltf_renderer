#pragma once

//-------------------------------------------------------------------------------------------------------
// The adaptive stop's relax table (shaders/adaptive_stop.h.slang, ePtAdaptiveStop): how much more noise
// a pixel of the still may keep where the tone mapper flattens its brightness.
//
// The stop's own rule, a standard error at most a threshold times the root of the luminance, keeps the
// noise even as the eye sees it where the display is about linear in the scene's light, which is the
// case for the midtones. A tone mapper's shoulder flattens highlights: a lamp, a sunlit wall, anything
// past white shows its noise many times smaller, or not at all where the curve clips, and the plain rule
// made those pixels sample to the cap. The table holds, a stop apart from mid grey up, the displayed
// noise of mid grey over that of the brightness there, for the same rule: the curve's slope at mid grey
// times the root of mid grey, over the same at the brightness. The curve is the host's own, grading
// included (applyTonemap on grey), so mid grey's widening is 1 and the shadows are never relaxed.
//
// Not with auto exposure, whose exposure only the GPU knows: no table, the plain rule.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>

#include <glm/glm.hpp>

// The curves call fma on vectors, which glm declares but, with the standard library's fma, defines for
// scalars only (nvpro's own host code never runs a curve)
namespace glm {
template <>
inline vec3 fma<vec3>(vec3 const& a, vec3 const& b, vec3 const& c)
{
  return a * b + c;
}
}  // namespace glm

#include <nvshaders/tonemap_functions.h.slang>

#include "../shaders/adaptive_stop.h.slang"

namespace adaptive_relax {

// The widening where the curve is flat, or clips: noise there does not show at all
inline constexpr float kMaxRelax = 1.0e4f;

struct Table
{
  float                                      base = 0.0f;  // mid grey's luminance in the scene; 0: no table
  std::array<float, ADAPTIVE_RELAX_STOPS>    relax{};      // the widening at base * 2^i
};

// The table for the tone mapper `tm` as the host sets it
inline Table build(shaderio::TonemapperData tm)
{
  Table table;
  if(tm.autoExposure != 0 || tm.isActive == 0 || !(tm.exposure > 0.0f))
    return table;
  // the curve on grey, at the image's centre and without the dither's noise
  tm.dither      = 0;
  tm.vignette    = 0.0f;
  tm.inputMatrix = shaderio::getColorCorrectionMatrix(tm.exposure, tm.temperature, tm.tint);
  const auto shown = [&](float luminance) {
    return shaderio::bt709Luminance(shaderio::applyTonemap(tm, glm::vec3(luminance), glm::vec2(0.5f), glm::vec2(1.0f)));
  };
  const auto slope = [&](float luminance) {
    constexpr float k = 1.05f;
    return (shown(luminance * k) - shown(luminance / k)) / (luminance * (k - 1.0f / k));
  };
  const float base = 0.18f / tm.exposure;
  const float mid  = std::sqrt(base) * slope(base);
  if(!(mid > 0.0f))
    return table;  // a curve flat at mid grey: nothing to measure against
  table.base = base;
  for(int i = 0; i < ADAPTIVE_RELAX_STOPS; i++)
  {
    const float luminance = base * std::exp2(float(i));
    const float shownHere = std::sqrt(luminance) * slope(luminance);
    table.relax[i]        = shownHere > mid / kMaxRelax ? std::clamp(mid / shownHere, 1.0f, kMaxRelax) : kMaxRelax;
  }
  return table;
}

}  // namespace adaptive_relax
