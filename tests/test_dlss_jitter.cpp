// The DLSS frames' sub-pixel jitter (shaders/dlss_util.h): the Halton sequence in bases 2 and 3, about
// the pixel's centre.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <nvshaders/slang_types.h>

#include "../shaders/dlss_util.h"

// The sequence's first points, each base by its own digits. One loop for both bases once ended with
// the base-3 digits and gave x = 0, 0, 0.5, 0.25 at indices 2, 4, 5, 6.
TEST(DlssJitter, TheSequenceIsHaltonInBasesTwoAndThree)
{
  const float x[] = {1 / 2.F, 1 / 4.F, 3 / 4.F, 1 / 8.F, 5 / 8.F, 3 / 8.F, 7 / 8.F, 1 / 16.F, 9 / 16.F};
  const float y[] = {1 / 3.F, 2 / 3.F, 1 / 9.F, 4 / 9.F, 7 / 9.F, 2 / 9.F, 5 / 9.F, 8 / 9.F, 1 / 27.F};
  for(uint32_t i = 0; i < 9; i++)
  {
    const auto p = shaderio::halton(i + 1);
    EXPECT_FLOAT_EQ(p.x, x[i]) << "index " << i + 1;
    EXPECT_FLOAT_EQ(p.y, y[i]) << "index " << i + 1;
  }
  EXPECT_EQ(shaderio::radicalInverse(0u, 2u), 0.0F);
  EXPECT_FLOAT_EQ(shaderio::radicalInverse(5u, 5u), 1 / 25.F);
}

// Any 2^a * 3^b frames in a row put one sample in each cell of a 2^a by 3^b grid over the pixel,
// wherever the run starts: what lets the sequence run on without a cycle.
TEST(DlssJitter, ARunOfFramesFillsThePixelEvenlyWhereverItStarts)
{
  const int grids[][2] = {{2, 1}, {4, 3}, {8, 9}, {16, 27}};
  for(const auto& grid : grids)
    for(const uint32_t start : {1u, 2u, 17u, 1000u, 123457u})
    {
      std::vector<int> count(size_t(grid[0]) * grid[1], 0);
      for(uint32_t i = 0; i < uint32_t(grid[0] * grid[1]); i++)
      {
        const auto j = shaderio::dlssJitter(start + i);
        ASSERT_GE(j.x, -0.5F);
        ASSERT_LT(j.x, 0.5F);
        ASSERT_GE(j.y, -0.5F);
        ASSERT_LT(j.y, 0.5F);
        const int cx = int((j.x + 0.5F) * float(grid[0]));
        const int cy = int((j.y + 0.5F) * float(grid[1]) * 1.000001F);  // a third's rounding, not a cell's edge
        count[size_t(cy) * grid[0] + cx]++;
      }
      for(const int c : count)
        EXPECT_EQ(c, 1) << grid[0] << "x" << grid[1] << " from frame " << start;
    }
}

// No frame samples the pixel's edge, where a half-pixel offset belongs to two pixels alike
TEST(DlssJitter, NoFrameSitsOnThePixelsEdge)
{
  for(uint32_t i = 1; i <= 4096; i++)
  {
    const auto j = shaderio::dlssJitter(i);
    EXPECT_GT(j.x, -0.5F) << i;
    EXPECT_GT(j.y, -0.5F) << i;
  }
}
