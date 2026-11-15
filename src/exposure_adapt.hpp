#pragma once

//-------------------------------------------------------------------------------------------------------
// Exposure that adapts to the view (the Camera tab's Adapt to the view, tmAdapt): the frame metered,
// and the stops the exposure the host sets is opened up or closed down for it, eased over time as an
// eye takes to a dark room. The arithmetic, Vulkan-free (tests/test_exposure_adapt.cpp); the meter
// and the frame's part are ExposureAdapt (src/exposure_adapt_vk.hpp).
//
// - The meter: the tone mapper's histogram of the image it takes (nvshaders Tonemapper::cmdHistogram),
//   its luminance in EV100 bins from kBelow stops under mid grey to kAbove over, mid grey as the
//   host's exposure makes it (meterRange), counted more at the frame's middle. The reading is the
//   mean, in stops from mid grey, of the middle half of what is lit (meteredStops): a patch of sun or
//   a window at either end of the frame's tones moves it no more than a black corner does.
// - The adaptation (adaptationFor): nothing within kKnee stops of mid grey, so a view the exposure
//   already suits keeps it; past that kShare of each stop is made up, at most kMostUp stops opened
//   and kMostDown closed. A share, not the whole: an eye in a barn on a sunny day still sees it dark.
// - Over time (eased): toward what the reading asks for, opening up over kOpenSeconds and closing
//   down faster, over kCloseSeconds, as an eye does; the same at any frame rate.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace exposure_adapt {

inline constexpr int   kBins         = 256;    // the tone mapper's (EXPOSURE_HISTOGRAM_SIZE)
inline constexpr float kMidGrey      = 0.18F;  // what the exposure puts a view's tones about
inline constexpr float kBelow        = 16.0F;  // the meter's range, stops under mid grey
inline constexpr float kAbove        = 8.0F;   // and over it
inline constexpr float kTrimLow      = 0.25F;  // the share of what is lit, from the darkest, the reading starts at
inline constexpr float kTrimHigh     = 0.75F;  // and ends at
inline constexpr float kKnee         = 1.0F;   // stops either side of mid grey that move nothing
inline constexpr float kSoft         = 0.5F;   // the knee's half-width: the adaptation bends in over twice this
inline constexpr float kShare        = 0.8F;   // of each stop past the knee, the share made up
inline constexpr float kMostUp       = 8.0F;   // stops opened at most
inline constexpr float kMostDown     = 3.0F;   // and closed
inline constexpr float kOpenSeconds  = 0.9F;   // the time constant of opening up (into the dark)
inline constexpr float kCloseSeconds = 0.35F;  // and of closing down (into the light)

// The histogram's range, EV100 as nvshaders' luminanceEv100 counts it, for an image the tone mapper
// exposes by `exposure`
struct Range
{
  float evMin = 0.0F;
  float evMax = 0.0F;
};
inline Range meterRange(float exposure)
{
  const float mid = std::log2(kMidGrey / std::max(exposure, 1e-30F) * (100.0F / 12.5F));
  return {mid - kBelow, mid + kAbove};
}

// Stops from mid grey at a bin's middle: bins 1 to kBins - 2 split the range evenly (the first also
// holds what is under it), and the last holds what is over it
inline float binStops(int bin)
{
  const float n = bin >= kBins - 1 ? 1.0F : (float(std::max(bin, 1)) - 0.5F) / float(kBins - 2);
  return n * (kBelow + kAbove) - kBelow;
}

// The view's reading, stops from mid grey: the mean of the middle half of what is lit, each bin as
// many times as the histogram counted it (bin 0, the black, is left out). None where nothing is lit.
inline std::optional<float> meteredStops(std::span<const uint32_t> bins)
{
  double total = 0.0;
  for(size_t b = 1; b < bins.size(); b++)
    total += double(bins[b]);
  if(total <= 0.0)
    return std::nullopt;
  const double from = double(kTrimLow) * total, to = double(kTrimHigh) * total;
  double       below = 0.0, sum = 0.0, weight = 0.0;
  for(size_t b = 1; b < bins.size(); b++)
  {
    const double count = double(bins[b]);
    const double lo = std::max(below, from), hi = std::min(below + count, to);
    if(hi > lo)
    {
      sum += (hi - lo) * double(binStops(int(b)));
      weight += hi - lo;
    }
    below += count;
  }
  if(weight <= 0.0)
    return std::nullopt;
  return float(sum / weight);
}

// The stops the exposure is opened (closed, negative) for a view that reads `stops` from mid grey
inline float adaptationFor(float stops)
{
  const float off = std::abs(stops);
  float       past;  // stops past the knee, bent in smoothly
  if(off <= kKnee - kSoft)
    past = 0.0F;
  else if(off >= kKnee + kSoft)
    past = off - kKnee;
  else
  {
    const float x = off - (kKnee - kSoft);
    past          = x * x / (4.0F * kSoft);
  }
  const float made = kShare * past;
  return stops < 0.0F ? std::min(made, kMostUp) : -std::min(made, kMostDown);
}

// The adaptation `seconds` on from `current`, on its way to `target`
inline float eased(float current, float target, float seconds)
{
  const float tau = target > current ? kOpenSeconds : kCloseSeconds;
  return target + (current - target) * std::exp(-std::max(seconds, 0.0F) / tau);
}

}  // namespace exposure_adapt
