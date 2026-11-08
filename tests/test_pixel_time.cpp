// The Time view's sums as the host reads them back (src/pixel_time.cpp): 64-bit times from the
// shader's two halves, the scale the next frames colour by, and the breakdown by share.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <nvshaders/slang_types.h>

#include "../shaders/pixel_time.h.slang"
#include "../src/pixel_time.hpp"

using pixel_time::Group;
using pixel_time::Reading;
using pixel_time::Share;

namespace {

// a frame's sums with `slots` slots, each {time in units, samples}
std::vector<uint32_t> sums(const std::vector<std::pair<uint64_t, uint32_t>>& slots)
{
  std::vector<uint32_t> out;
  for(const auto& [units, samples] : slots)
  {
    out.push_back(uint32_t(units & 0xFFFFFFFFu));
    out.push_back(uint32_t(units >> 32));
    out.push_back(samples);
    out.push_back(0u);
  }
  return out;
}

}  // namespace

// The halves make one time, in the clock's ticks; a slot nothing took a sample in is left out
TEST(PixelTime, TheHalvesMakeOneTimeInTheClocksTicks)
{
  const uint64_t big = (uint64_t(3) << 32) + 17;  // past what one uint holds: the shader's carry
  const auto     raw = sums({{1000, 10}, {0, 0}, {0, 0}, {0, 0}, {big, 40}});
  const Reading  r   = pixel_time::read(raw);
  ASSERT_EQ(r.slots.size(), 2u);
  EXPECT_EQ(r.slots[0].slot, shaderio::kPixelTimeSky);
  EXPECT_EQ(r.slots[0].ticks, 1000u << shaderio::kPixelTimeUnitShift);
  EXPECT_EQ(r.slots[1].slot, shaderio::kPixelTimeInstance);
  EXPECT_EQ(r.slots[1].ticks, big << shaderio::kPixelTimeUnitShift);
  EXPECT_EQ(r.slots[1].samples, 40u);
  EXPECT_EQ(r.ticks, (1000 + big) << shaderio::kPixelTimeUnitShift);
  EXPECT_EQ(r.samples, 50u);
}

// The scale is one over the average sample's ticks; nothing summed has none
TEST(PixelTime, TheScaleIsOneOverTheAverageSample)
{
  const Reading r = pixel_time::read(sums({{100, 4}, {300, 4}}));  // 400 units over 8 samples
  EXPECT_FLOAT_EQ(r.scale(), 8.0F / float(400u << shaderio::kPixelTimeUnitShift));
  EXPECT_EQ(pixel_time::read({}).scale(), 0.0F);
  EXPECT_EQ(pixel_time::read(sums({{0, 0}})).scale(), 0.0F);
}

// Largest first, the same key counted once, the tail folded into one line; the cost against the
// frame's average sample
TEST(PixelTime, SharesRankByTimeAndFoldTheTail)
{
  const std::vector<Group> groups = {
      {.key = "sky", .ticks = 100, .samples = 100},  // cheap: 1 a sample
      {.key = "hair", .ticks = 500, .samples = 50},  // dear: 10 a sample
      {.key = "body", .ticks = 300, .samples = 100},
      {.key = "hair", .ticks = 100, .samples = 10},  // a second instance of the hair
      {.key = "eyes", .ticks = 50, .samples = 20},
  };
  const std::vector<Share> s = pixel_time::shares(groups, 2, "rest");
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(s[0].key, "hair");
  EXPECT_DOUBLE_EQ(s[0].share, 600.0 / 1050.0);
  EXPECT_EQ(s[0].samples, 60u);
  EXPECT_DOUBLE_EQ(s[0].cost, (600.0 / 60.0) / (1050.0 / 280.0));
  EXPECT_EQ(s[1].key, "body");
  EXPECT_EQ(s[2].key, "rest");  // the sky and the eyes
  EXPECT_DOUBLE_EQ(s[2].share, 150.0 / 1050.0);
  EXPECT_EQ(s[2].samples, 120u);
  double total = 0.0;
  for(const Share& line : s)
    total += line.share;
  EXPECT_NEAR(total, 1.0, 1e-12);
  // nothing left over: no rest line
  EXPECT_EQ(pixel_time::shares(groups, 8, "rest").size(), 4u);
  EXPECT_TRUE(pixel_time::shares({}, 4, "rest").empty());
}
