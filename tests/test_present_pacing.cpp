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

// Native presentation's pacing: a frame that fits in a refresh waits for the swapchain before its
// input, one that doesn't waits before its present, and a band between the two keeps a frame near
// the boundary from flipping it.

#include <cmath>

#include <gtest/gtest.h>

#include "present_pacing.hpp"

namespace {
constexpr double k60Hz  = 1000.0 / 60.0;
constexpr double k144Hz = 1000.0 / 144.0;

// `frames` frames of the same cost
void run(PresentPacing& p, double costMs, double refreshMs, int frames = 40)
{
  for(int i = 0; i < frames; i++)
    p.note(costMs, refreshMs);
}
}  // namespace

TEST(PresentPacing, StartsWaitingBeforeTheInput)
{
  const PresentPacing p;
  EXPECT_FALSE(p.waitsLate());
}

TEST(PresentPacing, AFrameThatFitsARefreshWaitsBeforeItsInput)
{
  PresentPacing p;
  run(p, 8.0, k60Hz);
  EXPECT_FALSE(p.waitsLate());
}

TEST(PresentPacing, AFrameLongerThanARefreshWaitsBeforeItsPresent)
{
  // the moving view of a figure at 4K: 19 ms of GPU and a few of recording, at 60 Hz
  PresentPacing p;
  run(p, 22.0, k60Hz);
  EXPECT_TRUE(p.waitsLate());
}

TEST(PresentPacing, TheRefreshDecides)
{
  // 10 ms fits a 60 Hz refresh, not a 144 Hz one
  PresentPacing at60;
  run(at60, 10.0, k60Hz);
  EXPECT_FALSE(at60.waitsLate());
  PresentPacing at144;
  run(at144, 10.0, k144Hz);
  EXPECT_TRUE(at144.waitsLate());
}

TEST(PresentPacing, AnUnknownRefreshIsTakenAs60Hz)
{
  PresentPacing p;
  run(p, 10.0, 0.0);
  EXPECT_FALSE(p.waitsLate());
  run(p, 22.0, 0.0);
  EXPECT_TRUE(p.waitsLate());
}

TEST(PresentPacing, OneSlowFrameDoesNotSwitch)
{
  // a shader variant swapped in, a BLAS rebuilt: one long frame among short ones
  PresentPacing p;
  run(p, 6.0, k60Hz);
  p.note(40.0, k60Hz);
  EXPECT_FALSE(p.waitsLate());
  run(p, 6.0, k60Hz, 3);
  EXPECT_FALSE(p.waitsLate());
}

TEST(PresentPacing, TheBandKeepsAFrameNearTheBoundaryWhereItIs)
{
  // between the two thresholds the mode stays as it was, whichever it was
  const double between = 0.7 * k60Hz;
  PresentPacing early;
  run(early, between, k60Hz);
  EXPECT_FALSE(early.waitsLate());

  PresentPacing late;
  run(late, 22.0, k60Hz);
  ASSERT_TRUE(late.waitsLate());
  run(late, between, k60Hz);
  EXPECT_TRUE(late.waitsLate());
}

TEST(PresentPacing, FramesThatGetShortGoBackToTheEarlyWait)
{
  // the view stops moving: the still's idle frames tonemap and present, a millisecond or two
  PresentPacing p;
  run(p, 22.0, k60Hz);
  ASSERT_TRUE(p.waitsLate());
  run(p, 2.0, k60Hz);
  EXPECT_FALSE(p.waitsLate());
}

TEST(PresentPacing, AMeaninglessCostIsIgnored)
{
  PresentPacing p;
  run(p, 8.0, k60Hz);
  p.note(std::nan(""), k60Hz);
  p.note(-5.0, k60Hz);
  EXPECT_DOUBLE_EQ(p.costMs(), 8.0);
  EXPECT_FALSE(p.waitsLate());
}

TEST(PresentPacing, ResetForgetsTheCost)
{
  PresentPacing p;
  run(p, 22.0, k60Hz);
  ASSERT_TRUE(p.waitsLate());
  p.reset();
  EXPECT_FALSE(p.waitsLate());
  // the first frame after a reset is taken as it is, not smoothed against the old swapchain's
  p.note(5.0, k60Hz);
  EXPECT_DOUBLE_EQ(p.costMs(), 5.0);
}
