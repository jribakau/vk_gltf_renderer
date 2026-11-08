#pragma once

// The Time view's sums (shaders/pixel_time.h.slang) as the host reads them back: the frame's time by
// what its pixels' camera rays met first, the scale the next frames colour the image by, and the
// breakdown by share that the view's topic publishes. Pure: tests/test_pixel_time.cpp.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pixel_time {

// The uints a slot takes in the sums: the time's low and high halves, the samples, one unused
inline constexpr uint32_t kSlotWords = 4;

struct Slot
{
  uint32_t slot    = 0;  // kPixelTimeSky ... kPixelTimeInstance + an instance
  uint64_t ticks   = 0;  // in the clock's ticks
  uint64_t samples = 0;
};

struct Reading
{
  uint64_t          ticks   = 0;  // every slot's
  uint64_t          samples = 0;
  std::vector<Slot> slots;  // those that took a sample, in slot order
  // One over the average ticks a pixel's sample took (SceneFrameInfo::pixelTimeScale); 0 with nothing summed
  float scale() const;
};

// The sums as a frame leaves them, kSlotWords a slot
Reading read(std::span<const uint32_t> sums);

// What the breakdown adds up: one thing the pixels show, by a key the caller gives it
struct Group
{
  std::string key;
  uint64_t    ticks   = 0;
  uint64_t    samples = 0;
};

// One line of it: its share of the frame's time, and its samples' cost against the frame's average
// sample (2: twice as slow)
struct Share
{
  std::string key;
  double      share   = 0.0;
  double      cost    = 0.0;
  uint64_t    samples = 0;
};

// The groups by share, largest first, the first `top` of them; the rest summed into one more line
// keyed `rest` (none when nothing is left over). Groups with the same key count as one.
std::vector<Share> shares(const std::vector<Group>& groups, size_t top, const std::string& rest);

}  // namespace pixel_time
