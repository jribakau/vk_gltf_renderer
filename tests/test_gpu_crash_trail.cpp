/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// gpu_trail without a device: the frames it keeps for a lost device's log, and what each says

#include <gtest/gtest.h>

#include "gpu_crash_trail.hpp"

TEST(GpuCrashTrail, TheLastFramesAreKeptOldestFirst)
{
  for(uint64_t f = 1; f <= 700; f++)
    gpu_trail::record({.frame = f});
  const std::vector<gpu_trail::Frame> last = gpu_trail::recent(3);
  ASSERT_EQ(last.size(), 3u);
  EXPECT_EQ(last[0].frame, 698u);
  EXPECT_EQ(last[2].frame, 700u);
  EXPECT_EQ(gpu_trail::recent(100000).size(), 600u) << "the last 600";
  EXPECT_GE(last[2].seconds, last[0].seconds);
}

TEST(GpuCrashTrail, AFrameSaysItsMemoryAndWhatItsSyncDid)
{
  const std::string text = gpu_trail::describe({.frame       = 9123,
                                                .seconds     = 672.34,
                                                .hostMs      = 4319.7,
                                                .budget      = 14683ull << 20,
                                                .usage       = 13869ull << 20,
                                                .allocated   = 11194ull << 20,
                                                .spilled     = 608ull << 20,
                                                .stagingHeld = 128ull << 20,
                                                .stagingLive = 12ull << 20,
                                                .decoded     = 0,
                                                .copied      = 2,
                                                .refused     = 0,
                                                .queued      = 3,
                                                .sync        = "changed, textures moved"});
  EXPECT_EQ(text, "frame 9123 at 672.3 s, 4320 ms after the last: GPU 13869 of 14683 MB (allocator 11194), 608 MB in system memory; "
                  "staging 128 MB held, 12 live; images 0 decoded, 2 copied, 0 refused, 3 waiting; sync: changed, textures moved");
}

TEST(GpuCrashTrail, AFaultsAddressIsNamedByTheBufferItWasIn)
{
  const auto handle = [](uint64_t h) { return reinterpret_cast<VkBuffer>(h); };
  gpu_trail::frameStart(VK_NULL_HANDLE, 40);
  gpu_trail::buffer(handle(0x1001), 0x7000'0000, 1 << 20, true);
  gpu_trail::named(VK_OBJECT_TYPE_BUFFER, 0x1001, "previous positions");
  gpu_trail::named(VK_OBJECT_TYPE_IMAGE, 0x1001, "an image's name is not a buffer's");
  gpu_trail::buffer(handle(0x1002), 0x7010'0000, 1 << 20, true);  // the next MB, with no name
  gpu_trail::frameStart(VK_NULL_HANDLE, 45);
  gpu_trail::buffer(handle(0x1001), 0x7000'0000, 1 << 20, false);

  EXPECT_EQ(gpu_trail::whereIs(0x7000'5000, 0x1000),
            "in 'previous positions' (1.0 MB from 0x70000000), made in frame 40 and destroyed in frame 45");
  EXPECT_EQ(gpu_trail::whereIs(0x7010'0000, 0x1000), "in 'a buffer with no name' (1.0 MB from 0x70100000), made in frame 40, there still");
  // a page reaching into the next buffer is in both
  EXPECT_EQ(gpu_trail::whereIs(0x700F'F800, 0x1000),
            "in 'a buffer with no name' (1.0 MB from 0x70100000), made in frame 40, there still; before that in 'previous positions' "
            "(1.0 MB from 0x70000000), made in frame 40 and destroyed in frame 45");
  EXPECT_EQ(gpu_trail::whereIs(0x9000'0000, 0x1000).rfind("in none of the ", 0), 0u);

  // the range made again: what is there now, then what was
  gpu_trail::frameStart(VK_NULL_HANDLE, 50);
  gpu_trail::buffer(handle(0x1003), 0x7000'0000, 1 << 19, true);
  gpu_trail::named(VK_OBJECT_TYPE_BUFFER, 0x1003, "scene");
  EXPECT_EQ(gpu_trail::whereIs(0x7000'0000, 0x1000), "in 'scene' (0.5 MB from 0x70000000), made in frame 50, there still; before that in "
                                                     "'previous positions' (1.0 MB from 0x70000000), made in frame 40 and destroyed in frame 45");
  for(uint64_t h : {0x1002, 0x1003})
    gpu_trail::buffer(handle(h), 0, 0, false);
}

TEST(GpuCrashTrail, AFaultsAddressIsNamedByTheImageItWasIn)
{
  const auto image = [](uint64_t h) { return reinterpret_cast<VkImage>(h); };
  gpu_trail::frameStart(VK_NULL_HANDLE, 60);
  gpu_trail::image(image(0x2001), 0x8000'0000, 4 << 20, false, true);
  gpu_trail::named(VK_OBJECT_TYPE_IMAGE, 0x2001, "skin_D_1001.jpg");
  gpu_trail::named(VK_OBJECT_TYPE_BUFFER, 0x2001, "a buffer's name is not an image's");
  gpu_trail::image(image(0x2002), 0x8040'0000, 4 << 20, true, true);  // spilled to system memory
  gpu_trail::image(image(0x2003), 0, 64 << 20, false, true);          // dedicated memory: no address
  gpu_trail::frameStart(VK_NULL_HANDLE, 66);
  gpu_trail::image(image(0x2001), 0, 0, false, false);

  EXPECT_EQ(gpu_trail::whereIs(0x8000'1000, 0x1000),
            "in the image 'skin_D_1001.jpg' (4.0 MB from 0x80000000), made in frame 60 and destroyed in frame 66");
  EXPECT_EQ(gpu_trail::whereIs(0x8040'1000, 0x1000),
            "in the image 'an image with no name' (4.0 MB from 0x80400000, in system memory), made in frame 60, there still");
  // the one with no address is counted where nothing is found, never matched (not at 0 either)
  EXPECT_NE(gpu_trail::whereIs(0x0000'1000, 0x1000).find("one of the 1 images in memory of their own"), std::string::npos);
  for(uint64_t h : {0x2002, 0x2003})
    gpu_trail::image(image(h), 0, 0, false, false);
}

TEST(GpuCrashTrail, WithoutTheExtensionsNothingIsMarked)
{
  // no device: init never ran, so checkpoints are off and a section or frame marks nothing
  EXPECT_FALSE(gpu_trail::checkpointsOn());
  gpu_trail::frameStart(VK_NULL_HANDLE, 1);
  gpu_trail::section(VK_NULL_HANDLE, "Path trace");
  gpu_trail::section(VK_NULL_HANDLE, nullptr);
}
