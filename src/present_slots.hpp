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

// The shared images of native presentation (frame_present.cpp), kept apart from Vulkan and D3D so
// it can be tested.
//
// Vulkan copies each frame into one of a few images it shares with D3D11 and signals the frame's
// value on a shared fence; D3D waits for that value on its own queue, copies the image into the
// swapchain, presents, and signals the same value on a fence of its own (`completed`). Vulkan
// never waits on D3D on the GPU: a D3D side that stalled or lost its device would hang the
// renderer's queue, and the stream with it. Instead a slot is reused only once the CPU sees D3D
// is done with it, and a frame that finds every slot busy is not presented (counted as dropped).
//
// The images only grow: a smaller frame uses their top left, which D3D copies into the swapchain.
// The swapchain itself is always the frame's size: a composition swapchain stretches a source
// region (SetSourceSize) over its whole buffer rather than showing it at its own size.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

class PresentSlots
{
public:
  explicit PresentSlots(int count = 3)
      : m_lastUse(size_t(std::max(1, count)), 0)
  {
  }

  int count() const { return int(m_lastUse.size()); }

  // A slot the consumer has finished with (its fence reached `completed`), the least recently used
  // of those; -1 when every slot still holds a frame being presented.
  int acquire(uint64_t completed) const
  {
    int best = -1;
    for(int i = 0; i < count(); i++)
      if(m_lastUse[i] <= completed && (best < 0 || m_lastUse[i] < m_lastUse[best]))
        best = i;
    return best;
  }

  // The slot takes the next frame; returns that frame's fence value.
  uint64_t use(int slot)
  {
    m_lastUse[size_t(slot)] = ++m_value;
    return m_value;
  }

  // The last value handed out: once the consumer reaches it, no slot is in use.
  uint64_t value() const { return m_value; }

  // The images were made again (both sides idle): every slot is free. The fence values go on
  // from where they were, as a timeline semaphore's must.
  void reset() { std::fill(m_lastUse.begin(), m_lastUse.end(), 0); }

private:
  std::vector<uint64_t> m_lastUse;  // the fence value of each slot's last frame
  uint64_t              m_value{0};
};

struct PresentSize
{
  uint32_t width{0};
  uint32_t height{0};
  bool     operator==(const PresentSize&) const = default;
};

// The images' size for a frame of `want`: as they are when it fits, else grown to hold it.
inline PresentSize growCapacity(PresentSize capacity, PresentSize want)
{
  return {std::max(capacity.width, want.width), std::max(capacity.height, want.height)};
}

inline bool fits(PresentSize capacity, PresentSize want)
{
  return want.width <= capacity.width && want.height <= capacity.height;
}
