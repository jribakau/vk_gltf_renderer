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

// Frame pacing of a --stream run (frame_stream.cpp), kept apart from Vulkan so it can be tested.
//
// A headless run has no v-sync and would spin as fast as the GPU allows, so frames are paced to
// the stream's rate. A still being refined is the exception: the path tracer's image improves with
// every frame it accumulates, so the viewer waits on frames, not on the clock. Those run unpaced
// and only the frames the stream is due are copied out.

#pragma once

#include <algorithm>
#include <chrono>

class FramePacer
{
public:
  using Clock = std::chrono::steady_clock;

  explicit FramePacer(Clock::duration period)
      : m_period(period)
  {
  }

  void setPeriod(Clock::duration period) { m_period = period; }

  // When the next frame may start. `refining`: the renderer is accumulating a still.
  Clock::time_point startTime(Clock::time_point now, bool refining) const
  {
    if(refining || m_lastFrame == Clock::time_point{})
      return now;
    return std::max(now, m_lastFrame + m_period);
  }

  // A frame starts at `at`. Returns whether its image goes to the stream: always at the paced rate,
  // and when refining, once per period.
  bool begin(Clock::time_point at)
  {
    m_lastFrame   = at;
    const bool due = m_lastCopy == Clock::time_point{} || at >= m_lastCopy + m_period;
    if(due)
      m_lastCopy = at;
    return due;
  }

private:
  Clock::duration   m_period;
  Clock::time_point m_lastFrame{};
  Clock::time_point m_lastCopy{};
};
