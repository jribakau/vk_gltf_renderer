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

#pragma once

//
// When textures follow GPU memory as other programs take it and give it back: the policy, Vulkan-free
// so its tests reach it. The renderer reads the memory once a second and does what this says.
//
// A reading that dips isn't acted on, and what comes back is left alone for a while, so a budget that
// wanders doesn't turn into textures going up and down. A change of texture restarts a still, so one
// being refined, or an export, is left to finish; only memory past the driver's budget, where the
// frame would crawl in system memory, is worth interrupting them for.
//

#include <algorithm>
#include <cstdint>
#include <optional>

namespace rs {

class MemoryWatch
{
public:
  static constexpr double  kPeriod             = 1;   // seconds between readings
  static constexpr double  kLowerAfter         = 2;   // no headroom for this long
  static constexpr double  kRaiseAfter         = 5;   // room for this long
  static constexpr double  kRaiseAfterLowering = 30;  // whatever took the memory may want more
  // left free by a lowering and by a raising, so that the one doesn't call for the other
  static constexpr int64_t kMargin = int64_t(256) << 20;
  // decoded at a time: the frame waits for it
  static constexpr int64_t kRaiseBatch = int64_t(256) << 20;

  enum class Act
  {
    eNone,
    eLower,
    eRaise,
  };
  struct Reading
  {
    double  now        = 0;      // in seconds
    int64_t headroom   = 0;      // what is free beyond what is kept back (GltfRenderer::sceneHeadroom)
    bool    pastBudget = false;  // more in use than the driver's budget
    bool    canLower   = false;  // something can still come down
    bool    lowered    = false;  // something is down
    bool    refining   = false;  // a still is being refined
    bool    held       = false;  // an export is running
  };

  bool due(double now) const { return now >= m_next; }

  // What to do now. A raising that has begun carries on through the still it restarted.
  Act read(const Reading& r)
  {
    m_next = r.now + kPeriod;
    if(r.headroom < 0)
    {
      m_roomSince.reset();
      m_raising = false;
      if(!m_shortSince)
        m_shortSince = r.now;
      if(!r.canLower || r.now - *m_shortSince < kLowerAfter || (!r.pastBudget && (r.refining || r.held)))
        return Act::eNone;
      m_shortSince = r.now;  // what it frees isn't back until the frames that use it are done
      m_lowered    = r.now;
      return Act::eLower;
    }
    m_shortSince.reset();
    if(!r.lowered || r.headroom <= kMargin)
    {
      m_roomSince.reset();
      m_raising = false;
      return Act::eNone;
    }
    if(!m_roomSince)
      m_roomSince = r.now;
    if(r.held)
      return Act::eNone;
    const bool settled = r.now - *m_roomSince >= kRaiseAfter && (!m_lowered || r.now - *m_lowered >= kRaiseAfterLowering);
    if(!m_raising && (r.refining || !settled))
      return Act::eNone;
    m_raising = true;
    return Act::eRaise;
  }

  // Nothing fitted what eRaise had room for: it is asked again once kRaiseAfter has passed
  void raiseEnded(double now)
  {
    m_raising   = false;
    m_roomSince = now;
  }

  // what a raising can take of `headroom`
  static uint64_t raiseRoom(int64_t headroom) { return uint64_t(std::clamp<int64_t>(headroom - kMargin, 0, kRaiseBatch)); }

private:
  double                m_next = 0;
  std::optional<double> m_shortSince, m_roomSince, m_lowered;
  bool                  m_raising = false;
};

}  // namespace rs
