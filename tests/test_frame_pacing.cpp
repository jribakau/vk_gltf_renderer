/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Frame pacing of a --stream run: paced to the stream's rate, except a still being refined, which
// runs unpaced and hands the stream a frame once per period.

#include <gtest/gtest.h>

#include "frame_pacing.hpp"

using namespace std::chrono_literals;
using Clock = FramePacer::Clock;

namespace {
const Clock::time_point t0 = Clock::time_point{} + 1000s;
}

TEST(FramePacing, PacedFramesStartOnePeriodApart)
{
  FramePacer p(16ms);
  EXPECT_EQ(p.startTime(t0, false), t0);  // the first frame starts at once
  EXPECT_TRUE(p.begin(t0));
  EXPECT_EQ(p.startTime(t0 + 3ms, false), t0 + 16ms);  // early: wait
  EXPECT_EQ(p.startTime(t0 + 40ms, false), t0 + 40ms);  // late: no catching up
}

TEST(FramePacing, PacedFramesAreAllCopied)
{
  FramePacer p(16ms);
  auto       t = t0;
  for(int i = 0; i < 10; i++)
  {
    t = p.startTime(t, false);
    EXPECT_TRUE(p.begin(t)) << i;
  }
}

TEST(FramePacing, RefiningFramesRunUnpaced)
{
  FramePacer p(16ms);
  p.begin(t0);
  EXPECT_EQ(p.startTime(t0 + 1ms, true), t0 + 1ms);
  EXPECT_EQ(p.startTime(t0 + 2ms, true), t0 + 2ms);
}

TEST(FramePacing, RefiningFramesFeedTheStreamOncePerPeriod)
{
  FramePacer p(16ms);
  int        frames = 0, copies = 0;
  // one second of a 1 ms frame: 1000 frames, 60 per second to the stream (a period is 16 ms)
  for(auto t = t0; t < t0 + 1000ms; t += 1ms)
  {
    frames++;
    copies += p.begin(p.startTime(t, true)) ? 1 : 0;
  }
  EXPECT_EQ(frames, 1000);
  EXPECT_GE(copies, 60);
  EXPECT_LE(copies, 63);
}

TEST(FramePacing, RefiningFramesFeedTheStreamAtTheCopyPeriod)
{
  FramePacer p(16ms);
  p.setCopyPeriod(50ms);
  int copies = 0;
  for(auto t = t0; t < t0 + 1000ms; t += 1ms)
    copies += p.begin(p.startTime(t, true), true) ? 1 : 0;
  EXPECT_GE(copies, 20);
  EXPECT_LE(copies, 21);
  // paced frames are not held to the copy period: one period apart, every one is copied
  const auto t = p.startTime(t0 + 1000ms, false);
  EXPECT_TRUE(p.begin(t, false));
  const auto next = p.startTime(t + 1ms, false);
  EXPECT_EQ(next, t + 16ms);
  EXPECT_TRUE(p.begin(next, false));
}

TEST(FramePacing, ASlowRefiningFrameIsAlwaysCopied)
{
  FramePacer p(16ms);
  EXPECT_TRUE(p.begin(t0));
  EXPECT_TRUE(p.begin(t0 + 50ms));  // a frame longer than the period: every frame is due
  EXPECT_TRUE(p.begin(t0 + 100ms));
}

TEST(FramePacing, PacingResumesWhenTheStillHasConverged)
{
  FramePacer p(16ms);
  p.begin(t0);
  p.begin(p.startTime(t0 + 5ms, true));
  const auto last = t0 + 5ms;
  // converged: the next frame waits a period after the last one, and its copy is due
  const auto next = p.startTime(last + 1ms, false);
  EXPECT_EQ(next, last + 16ms);
  EXPECT_TRUE(p.begin(next));
}
