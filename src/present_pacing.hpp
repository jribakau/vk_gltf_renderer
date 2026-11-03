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

// Where native presentation (frame_present.cpp) waits for the swapchain, kept apart from D3D so it
// can be tested.
//
// The swapchain takes one frame at a time (frame latency 1): its waitable is signalled once the
// frame before has been composed, at a refresh. Waiting for it before the host takes its input has
// the frame drawn next shown at the next refresh, the least latency there is, as long as the frame
// is recorded and rendered by then. One that takes longer misses that refresh, and the loop starts
// the next only at the refresh after: the GPU idles while the display waits, and a 20 ms frame on a
// 60 Hz display runs at 30 fps.
//
// Such frames wait before their present instead, after their submit: the next frame is recorded
// while the one before is still on its way to the screen, so the GPU always has one queued and the
// display shows every frame it makes (at 20 ms, 50 fps), at one refresh more of latency. Which of
// the two follows the frame's cost against the refresh period, smoothed, with a band between
// switching to the late wait and back so that a frame near the boundary doesn't flip it every
// frame.

#pragma once

#include <algorithm>

class PresentPacing
{
public:
  // The share of a refresh period a frame may cost and still be waited for before its input, and
  // the share below which a late wait goes back to that. The first leaves room for the compositor,
  // which takes a frame some way ahead of the refresh, and for the copy into the swapchain.
  static constexpr double kLateAbove  = 0.8;
  static constexpr double kEarlyBelow = 0.6;
  // How much each frame moves the smoothed cost, and the most a frame counts for, in refreshes: a
  // spike (a shader swap, a rebuild) moves it little, a frame rate that changed within a few frames
  static constexpr double kSmoothing     = 0.2;
  static constexpr double kMostRefreshes = 2.0;
  // The refresh period taken when the display's is unknown: 60 Hz
  static constexpr double kDefaultRefreshMs = 1000.0 / 60.0;

  // A frame presented with the loop paced: its cost (its CPU time from its input to its submit,
  // plus its GPU time) and the display's refresh period, both in milliseconds (0: unknown).
  void note(double costMs, double refreshMs)
  {
    if(!(costMs >= 0.0))  // negative, or NaN
      return;
    const double refresh = refreshMs > 0.0 ? refreshMs : kDefaultRefreshMs;
    const double cost    = std::min(costMs, kMostRefreshes * refresh);
    m_costMs             = m_noted ? m_costMs + kSmoothing * (cost - m_costMs) : cost;
    m_noted              = true;

    if(!m_late && m_costMs > kLateAbove * refresh)
      m_late = true;
    else if(m_late && m_costMs < kEarlyBelow * refresh)
      m_late = false;
  }

  // Whether the next paced frame waits for the swapchain after its submit, before its present
  // (true), or before the host takes its input (false).
  bool waitsLate() const { return m_late; }

  double costMs() const { return m_costMs; }

  // A new swapchain: nothing is known of what its frames cost
  void reset() { *this = PresentPacing(); }

private:
  double m_costMs{0.0};
  bool   m_noted{false};
  bool   m_late{false};
};
