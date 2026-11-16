#pragma once

//-------------------------------------------------------------------------------------------------------
// Exposure that adapts to the view (the Camera tab's Adapt to the view, tmAdapt; and play mode's eye):
// the frame metered, and the stops the exposure the host sets is opened up or closed down for it,
// eased over time as an eye takes to a dark room. The arithmetic, Vulkan-free
// (tests/test_exposure_adapt.cpp); the meter and the frame's part are ExposureAdapt
// (src/exposure_adapt_vk.hpp). How far and how it adapts is a Tuning: a camera's (kCamera) or an
// eye's (kEye).
//
// - The meter: the tone mapper's histogram of the image it takes (nvshaders Tonemapper::cmdHistogram),
//   its luminance in EV100 bins from `below` stops under mid grey to `above` over, mid grey as the
//   host's exposure makes it (meterRange), counted more at the frame's middle (`spread`). The reading
//   is the mean, in stops from mid grey, of a band of what is lit (meteredStops): the camera's the
//   middle half, so a patch of sun or a window at either end of the frame's tones moves it no more
//   than a black corner does.
// - The adaptation (adaptationFor): nothing within `knee` stops of mid grey, so a view the exposure
//   already suits keeps it; past that `share` of each stop is made up, at most `mostUp` stops opened
//   and `mostDown` closed. The camera's a share, not the whole: an eye in a barn on a sunny day still
//   sees it dark.
// - Over time (eased): toward what the reading asks for, opening up over `openSeconds` and closing
//   down faster, over `closeSeconds`, as an eye does; the same at any frame rate.
//
// Play mode's eye (kEye) is the one who walks there: it reads what it looks at (a spot about as wide
// as the torch's beam, the brighter part of it, so the torch's pool is what it takes to and not the
// dark round it), takes to it wholly, and has an eye's range: from a day down to a moonlit night,
// which it still leaves a night's dark.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace exposure_adapt {

inline constexpr int   kBins    = 256;    // the tone mapper's (EXPOSURE_HISTOGRAM_SIZE)
inline constexpr float kMidGrey = 0.18F;  // what the exposure puts a view's tones about

struct Tuning
{
  float below        = 16.0F;  // the meter's range, stops under mid grey
  float above        = 8.0F;   // and over it
  float spread       = 0.75F;  // how far from the frame's middle a pixel counts at all, of its width
  float trimLow      = 0.25F;  // the share of what is lit, from the darkest, the reading starts at
  float trimHigh     = 0.75F;  // and ends at
  float knee         = 1.0F;   // stops either side of mid grey that move nothing
  float soft         = 0.5F;   // the knee's half-width: the adaptation bends in over twice this
  float share        = 0.8F;   // of each stop past the knee, the share made up
  float mostUp       = 8.0F;   // stops opened at most
  float mostDown     = 3.0F;   // and closed
  float openSeconds  = 0.9F;   // the time constant of opening up (into the dark)
  float closeSeconds = 0.35F;  // and of closing down (into the light)
};

// A camera's: the Camera tab's Adapt to the view
inline constexpr Tuning kCamera{};

// An eye's: play mode's. A moonlit night is some 20 stops under a day's exposure (0.3 lux against
// 1e5), a torch's pool a few metres off some 8 (11,000 cd); the eye opens to the one and leaves it
// three stops under mid grey, and to the other wholly; a moonless night (29) it leaves dark.
inline constexpr Tuning kEye{.below        = 32.0F,
                             .above        = 12.0F,
                             .spread       = 0.25F,
                             .trimLow      = 0.5F,
                             .trimHigh     = 0.95F,
                             .share        = 1.0F,
                             .mostUp       = 17.0F,
                             .mostDown     = 12.0F,
                             .openSeconds  = 1.2F,
                             .closeSeconds = 0.4F};

// The histogram's range, EV100 as nvshaders' luminanceEv100 counts it, for an image the tone mapper
// exposes by `exposure`
struct Range
{
  float evMin = 0.0F;
  float evMax = 0.0F;
};
inline Range meterRange(float exposure, const Tuning& t = kCamera)
{
  const float mid = std::log2(kMidGrey / std::max(exposure, 1e-30F) * (100.0F / 12.5F));
  return {mid - t.below, mid + t.above};
}

// Stops from mid grey at a bin's middle: bins 1 to kBins - 2 split the range evenly (the first also
// holds what is under it), and the last holds what is over it
inline float binStops(int bin, const Tuning& t = kCamera)
{
  const float n = bin >= kBins - 1 ? 1.0F : (float(std::max(bin, 1)) - 0.5F) / float(kBins - 2);
  return n * (t.below + t.above) - t.below;
}

// The view's reading, stops from mid grey: the mean of the tuning's band of what is lit, each bin as
// many times as the histogram counted it (bin 0, the black, is left out). None where nothing is lit.
inline std::optional<float> meteredStops(std::span<const uint32_t> bins, const Tuning& t = kCamera)
{
  double total = 0.0;
  for(size_t b = 1; b < bins.size(); b++)
    total += double(bins[b]);
  if(total <= 0.0)
    return std::nullopt;
  const double from = double(t.trimLow) * total, to = double(t.trimHigh) * total;
  double       below = 0.0, sum = 0.0, weight = 0.0;
  for(size_t b = 1; b < bins.size(); b++)
  {
    const double count = double(bins[b]);
    const double lo = std::max(below, from), hi = std::min(below + count, to);
    if(hi > lo)
    {
      sum += (hi - lo) * double(binStops(int(b), t));
      weight += hi - lo;
    }
    below += count;
  }
  if(weight <= 0.0)
    return std::nullopt;
  return float(sum / weight);
}

// The stops the exposure is opened (closed, negative) for a view that reads `stops` from mid grey
inline float adaptationFor(float stops, const Tuning& t = kCamera)
{
  const float off = std::abs(stops);
  float       past;  // stops past the knee, bent in smoothly
  if(off <= t.knee - t.soft)
    past = 0.0F;
  else if(off >= t.knee + t.soft)
    past = off - t.knee;
  else
  {
    const float x = off - (t.knee - t.soft);
    past          = x * x / (4.0F * t.soft);
  }
  const float made = t.share * past;
  return stops < 0.0F ? std::min(made, t.mostUp) : -std::min(made, t.mostDown);
}

// The adaptation `seconds` on from `current`, on its way to `target`
inline float eased(float current, float target, float seconds, const Tuning& t = kCamera)
{
  const float tau = target > current ? t.openSeconds : t.closeSeconds;
  return target + (current - target) * std::exp(-std::max(seconds, 0.0F) / tau);
}

}  // namespace exposure_adapt
