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

// The host's drag timing (DialTiming): the once-a-second dial line, and the drag frames' second line of
// the application thread's loop, the host update by phase, the recording and the frames the GPU waited for.

#include <gtest/gtest.h>

#include "fv_host_impl.hpp"

TEST(DialTiming, DragFramesLineSumsTheirPhasesRecordingAndWaits)
{
  DialTiming t;
  const auto start = std::chrono::steady_clock::now();
  t.add(DialTiming::eSetMany, 4.0);
  // a phase named twice in a frame (two figures) counts once for the frame, summed
  t.addHost(7.0, {{"frame: input and morphs", 4.0}, {"frame: figure pump", 0.5}, {"frame: figure pump", 0.5}});
  t.addFrame(19.0, 2.0, false);
  t.add(DialTiming::eSetMany, 4.0);
  t.addHost(9.0, {{"frame: input and morphs", 6.0}, {"frame: figure pump", 1.0}});
  t.addFrame(21.0, 1.0, true);
  EXPECT_TRUE(t.report(start).empty());  // not a second yet

  const std::string out = t.report(start + std::chrono::seconds(2));
  const size_t      nl  = out.find('\n');
  ASSERT_NE(nl, std::string::npos) << out;
  // the first line as scripts/baseline-report.py reads it
  EXPECT_EQ(out.rfind("[fv-time] dial: 2 steps, setMany 4.0 ms mean, 4.0 max;", 0), 0u) << out;
  EXPECT_EQ(out.substr(nl + 1),
            "[fv-time] drag frames: 2, loop 20.0 ms mean, 21.0 max; host update 8.0 ms mean, 9.0 max (input and "
            "morphs 5.0, figure pump 1.0); recording 1.5 ms mean, 2.0 max; the GPU waited on 1 of 2");
  // the frames' own sums don't count as steps landing, and the count outlives the report
  EXPECT_EQ(t.landed, 2u);
  EXPECT_TRUE(t.report(start + std::chrono::seconds(4)).empty());
}

TEST(DialTiming, NoDragFramesNoSecondLine)
{
  DialTiming t;
  const auto start = std::chrono::steady_clock::now();
  t.add(DialTiming::ePoseSet, 1.0);
  const std::string out = t.report(start + std::chrono::seconds(2));
  EXPECT_FALSE(out.empty());
  EXPECT_EQ(out.find('\n'), std::string::npos) << out;
}
