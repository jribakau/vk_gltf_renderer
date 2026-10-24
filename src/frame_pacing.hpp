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
//
// The other exception is nothing happening: when the renderer has drawn nothing (a converged still,
// a static rasterized view) and no request has come in for a while, the loop slows to the idle
// period. Each idle frame still copies the image out and the encoder still compares it with the
// last one sent, so a change that reached the image without a draw or a request shows within an
// idle period; a request ends the idle at once (FrameStream wakes the sleep).

#pragma once

#include <algorithm>
#include <chrono>

class FramePacer
{
public:
  using Clock = std::chrono::steady_clock;

  explicit FramePacer(Clock::duration period)
      : m_period(period)
      , m_copyPeriod(period)
  {
  }

  void setPeriod(Clock::duration period) { m_period = period; }

  // How often a refining still's frames go to the stream. A converging image needs fewer than the
  // paced rate, and a stream fed at the paced rate with full-chroma frames keeps the encoder busy
  // just when the next drag's first frames arrive.
  void setCopyPeriod(Clock::duration period) { m_copyPeriod = period; }

  // The period while idle (see idle()), and how long nothing must have happened before.
  void setIdlePeriod(Clock::duration period) { m_idlePeriod = period; }
  void setIdleAfter(Clock::duration after) { m_idleAfter = after; }

  // The renderer drew a frame (its drawn-frame count moved): the image is changing.
  void noteDrawn(Clock::time_point at) { m_lastDrawn = at; }
  // A request came in (a command, input, a client, a resize): something is about to change.
  void noteActivity(Clock::time_point at) { m_lastActivity = at; }

  // Nothing drew and nothing came in for the idle delay. Never while refining: there every frame
  // draws. Nor before the first draw or request: idle is something having stopped.
  bool idle(Clock::time_point now, bool refining = false) const
  {
    if(refining)
      return false;
    const Clock::time_point last = std::max(m_lastDrawn, m_lastActivity);
    return last != Clock::time_point{} && now - last > m_idleAfter;
  }

  // When the next frame may start. `refining`: the renderer is accumulating a still.
  Clock::time_point startTime(Clock::time_point now, bool refining) const
  {
    if(refining || m_lastFrame == Clock::time_point{})
      return now;
    const Clock::duration period = idle(now) ? std::max(m_period, m_idlePeriod) : m_period;
    return std::max(now, m_lastFrame + period);
  }

  // A frame starts at `at`. Returns whether its image goes to the stream: every paced frame, and
  // when refining, one per copy period.
  bool begin(Clock::time_point at, bool refining = false)
  {
    m_lastFrame       = at;
    const auto period = refining ? m_copyPeriod : m_period;
    const bool due    = m_lastCopy == Clock::time_point{} || at >= m_lastCopy + period;
    if(due)
      m_lastCopy = at;
    return due;
  }

private:
  Clock::duration   m_period;
  Clock::duration   m_copyPeriod;
  Clock::duration   m_idlePeriod{std::chrono::milliseconds(100)};
  Clock::duration   m_idleAfter{std::chrono::milliseconds(500)};
  Clock::time_point m_lastFrame{};
  Clock::time_point m_lastCopy{};
  Clock::time_point m_lastDrawn{};
  Clock::time_point m_lastActivity{};
};
