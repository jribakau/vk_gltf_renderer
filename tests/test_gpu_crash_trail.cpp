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

TEST(GpuCrashTrail, WithoutTheExtensionsNothingIsMarked)
{
  // no device: init never ran, so checkpoints are off and a section or frame marks nothing
  EXPECT_FALSE(gpu_trail::checkpointsOn());
  gpu_trail::frameStart(VK_NULL_HANDLE, 1);
  gpu_trail::section(VK_NULL_HANDLE, "Path trace");
  gpu_trail::section(VK_NULL_HANDLE, nullptr);
}
