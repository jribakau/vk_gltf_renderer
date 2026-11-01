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
// When system memory is short, the policy alone: the host gives back what it can keep without (what it
// holds only for an undo or for later) and says so, once for each time it gets short. Past physical memory
// Windows pages and everything stalls with no signal to anyone, so the reading is taken beforehand.
//

#include <algorithm>
#include <cstdint>
#include <span>

namespace rs {

class RamWatch
{
public:
  struct Reading
  {
    uint64_t totalPhysical = 0, availablePhysical = 0, availableCommit = 0;
  };
  struct Step
  {
    bool short_   = false;  // short of it now
    bool justNow  = false;  // and wasn't at the last reading: time to say so
  };

  // Short: under a sixteenth of physical memory or 2 GiB, whichever is more, of it or of what can still
  // be committed. No longer: at twice that, so that what is given back doesn't end it.
  Step step(const Reading& r)
  {
    const uint64_t low   = std::max<uint64_t>(r.totalPhysical / 16, uint64_t(2) << 30);
    const uint64_t least = std::min(r.availablePhysical, r.availableCommit);
    const bool     was   = m_short;
    if(!m_short && least < low)
      m_short = true;
    else if(m_short && least >= low * 2)
      m_short = false;
    return {m_short, m_short && !was};
  }

private:
  bool m_short = false;
};

// Of what is kept for later, oldest first, how many to let go of for the rest to fit `budget`.
inline size_t oldestToLetGo(std::span<const uint64_t> bytes, uint64_t budget)
{
  uint64_t total = 0;
  for(uint64_t b : bytes)
    total += b;
  size_t gone = 0;
  while(gone < bytes.size() && total > budget)
    total -= bytes[gone++];
  return gone;
}

}  // namespace rs
