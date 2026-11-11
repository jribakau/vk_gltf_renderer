#pragma once

//-------------------------------------------------------------------------------------------------------
// Auto white balance (the app's Camera › Image › White balance › Auto, the host's render.whiteBalance):
// the colour of the light a frame was lit by, and the white balance that takes it to grey.
//
// estimateLight looks at the image the tone mapper takes, before its white balance, as a camera's auto
// white balance looks at the sensor's: an iterative grey-pixel average. Most surfaces are near grey, so
// the light's colour is what the frame's surfaces share, weighted toward the ones that look grey once that
// light is taken out. Strongly coloured surfaces (a red curtain, a blue sofa) and coloured emitters (neon,
// a sign) count little, and the brightest pixels (lamps, the sun, clipped highlights) not at all.
//
// forLight then finds the temperature and Duv tint whose correction, nvshaders' getColorCorrectionMatrix
// (the one the tone mapper applies), makes that light grey: solved on the matrix itself, so the slider's
// value and the picture agree whatever the matrix's colour science.
//
// Vulkan-free; the renderer reads the pixels back (GltfRenderer::readLight).
//-------------------------------------------------------------------------------------------------------

#include <cstdint>
#include <optional>

#include <glm/glm.hpp>

namespace white_balance {

// The white balance as the lighting topic's tonemap.temperature (Kelvin) and tonemap.tint (Duv) take it
struct Setting
{
  double temperature = 6500.0;
  double tint        = 0.0;
  bool   clamped     = false;  // the light was beyond the range below: this is its nearest edge
};

// The range forLight answers in: the renderer's tmTemperature and tmTint
inline constexpr double kMinKelvin = 2000.0;
inline constexpr double kMaxKelvin = 15000.0;
inline constexpr double kMaxTint   = 0.03;

// The light's colour in linear Rec. 709, luminance 1, from `rgba` (width x height pixels of four floats,
// rows packed): nothing when the image has too few usable pixels (black, or not finite)
std::optional<glm::dvec3> estimateLight(const float* rgba, uint32_t width, uint32_t height);

// The white balance whose correction takes light of colour `light` (linear Rec. 709, any brightness) to
// grey; nothing for a light with a channel at or under 0
std::optional<Setting> forLight(const glm::dvec3& light);

}  // namespace white_balance
