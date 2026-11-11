#include "white_balance.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <nvshaders/tonemap_functions.h.slang>  // getColorCorrectionMatrix

namespace white_balance {

namespace {

double luminance(const glm::dvec3& c)
{
  return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// About this many pixels are looked at, spread evenly over the frame: plenty for an average, and a
// 4K frame's 8 million would only take longer
constexpr uint64_t kSamples = 512 * 512;
// The brightest share of the frame is left out: lamps, the sun, neon, clipped highlights, whose colour is
// their own (or the sensor's) and not the light's on the surfaces
constexpr double kBrightestShare = 0.02;
// Under this share of the median luminance a pixel is too dark to say much but its noise
constexpr double kDarkest = 1e-3;
// How hard a coloured pixel is discounted, once the light is taken out: (1 - saturation)^this
constexpr int kGreyPower = 4;
constexpr int kRounds    = 4;

// How coloured `c` is: 0 grey, 1 a pure primary or secondary
double saturation(const glm::dvec3& c)
{
  const double hi = std::max(c.x, std::max(c.y, c.z));
  const double lo = std::min(c.x, std::min(c.y, c.z));
  return hi > 0.0 ? (hi - lo) / hi : 0.0;
}

// What the tone mapper makes of `light` at this white balance, as applyTonemap's mul(inputMatrix, c),
// which the C++ shim writes c * M (renderer_selection.cpp)
glm::dvec3 corrected(const glm::dvec3& light, double kelvin, double tint)
{
  return glm::dvec3(glm::vec3(light) * shaderio::getColorCorrectionMatrix(1.0F, float(kelvin), float(tint)));
}

// How far from grey: the red and blue channels against green, in log
glm::dvec2 offGrey(const glm::dvec3& c)
{
  return {std::log(c.x / c.y), std::log(c.z / c.y)};
}

}  // namespace

std::optional<glm::dvec3> estimateLight(const float* rgba, uint32_t width, uint32_t height)
{
  if(!rgba || width == 0 || height == 0)
    return std::nullopt;
  const uint64_t pixels = uint64_t(width) * height;
  const uint32_t step   = uint32_t(std::max<double>(1.0, std::ceil(std::sqrt(double(pixels) / double(kSamples)))));

  std::vector<glm::dvec3> colours;
  std::vector<double>     lum;
  colours.reserve(size_t(pixels / (uint64_t(step) * step) + width / step + height / step + 1));
  for(uint32_t y = step / 2; y < height; y += step)
    for(uint32_t x = step / 2; x < width; x += step)
    {
      const float*     p = rgba + (size_t(y) * width + x) * 4;
      const glm::dvec3 c(p[0], p[1], p[2]);
      if(!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z) || c.x < 0.0 || c.y < 0.0 || c.z < 0.0)
        continue;
      const double l = luminance(c);
      if(!(l > 0.0))
        continue;
      colours.push_back(c);
      lum.push_back(l);
    }
  if(colours.size() < 64)
    return std::nullopt;

  // the brightest few percent out, and the near-black
  std::vector<double> sorted = lum;
  const size_t        top    = std::min(sorted.size() - 1, size_t(double(sorted.size()) * (1.0 - kBrightestShare)));
  std::nth_element(sorted.begin(), sorted.begin() + top, sorted.end());
  const double bright = sorted[top];
  std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
  const double dark = sorted[sorted.size() / 2] * kDarkest;
  std::vector<glm::dvec3> kept;
  kept.reserve(colours.size());
  for(size_t i = 0; i < colours.size(); i++)
    if(lum[i] <= bright && lum[i] >= dark)
      kept.push_back(colours[i]);
  if(kept.size() < 64)
    return std::nullopt;

  // From a white light: the pixels weighed by how grey they look with the light so far taken out, their
  // sum the next light. A surface the light makes look coloured counts little at first, and fully once
  // the light is known; what is coloured of itself stays discounted.
  glm::dvec3 light(1.0);
  for(int round = 0; round < kRounds; round++)
  {
    glm::dvec3 sum(0.0);
    double     weights = 0.0;
    for(const glm::dvec3& c : kept)
    {
      const double w = std::pow(1.0 - saturation(c / light), kGreyPower);
      sum += w * c;
      weights += w;
    }
    // every pixel a pure colour: the plain average
    if(!(weights > 1e-6 * double(kept.size())))
    {
      sum = glm::dvec3(0.0);
      for(const glm::dvec3& c : kept)
        sum += c;
    }
    const double l = luminance(sum);
    if(!(l > 0.0))
      return std::nullopt;
    light = sum / l;
  }
  return light;
}

std::optional<Setting> forLight(const glm::dvec3& light)
{
  if(!(light.x > 0.0 && light.y > 0.0 && light.z > 0.0) || !std::isfinite(light.x + light.y + light.z))
    return std::nullopt;
  // Gauss-Newton on the temperature in mireds (about even in how the colour moves) and the tint, from
  // the tone mapper's neutral: the red and the blue against the green of the light corrected, to 0. The
  // solve runs wider than the answer's range, so a light past it is clamped to the nearest edge rather
  // than to wherever the solve stalled.
  const double mMin = 1e6 / 25000.0, mMax = 1e6 / 1200.0, dMax = 0.06;
  double       m = 1e6 / 6500.0, d = 0.0;
  const double hm = 0.5, hd = 2e-4;  // the derivative's steps: the matrix is single precision
  for(int i = 0; i < 40; i++)
  {
    const glm::dvec2 f = offGrey(corrected(light, 1e6 / m, d));
    if(std::abs(f.x) < 1e-6 && std::abs(f.y) < 1e-6)
      break;
    const glm::dvec2 fm  = (offGrey(corrected(light, 1e6 / (m + hm), d)) - offGrey(corrected(light, 1e6 / (m - hm), d))) / (2.0 * hm);
    const glm::dvec2 fd  = (offGrey(corrected(light, 1e6 / m, d + hd)) - offGrey(corrected(light, 1e6 / m, d - hd))) / (2.0 * hd);
    const double     det = fm.x * fd.y - fd.x * fm.y;
    if(!(std::abs(det) > 0.0))
      break;
    // the step that zeroes f on the tangent plane, no larger than 40 mireds or 0.01 Duv a round
    double dm = -(f.x * fd.y - fd.x * f.y) / det;
    double dd = -(fm.x * f.y - f.x * fm.y) / det;
    const double scale = std::min({1.0, 40.0 / std::max(std::abs(dm), 1e-12), 0.01 / std::max(std::abs(dd), 1e-12)});
    dm *= scale;
    dd *= scale;
    m = std::clamp(m + dm, mMin, mMax);
    d = std::clamp(d + dd, -dMax, dMax);
  }
  Setting s;
  const double kelvin = 1e6 / m;
  s.temperature       = std::clamp(kelvin, kMinKelvin, kMaxKelvin);
  s.tint              = std::clamp(d, -kMaxTint, kMaxTint);
  s.clamped           = s.temperature != kelvin || s.tint != d;
  return s;
}

}  // namespace white_balance
