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
// The Fit: what gives way when GPU memory is short, Vulkan-free so its tests reach it. Everything resident
// has steps down, each freeing bytes at some loss, and the plan takes the cheapest loss first: what nothing
// looks at before what something does, the largest texture before a smaller one, and what is marked Keep
// full last of all. It only says what: rs::MemoryWatch says when, and the renderer asks for a step at a time,
// reading the memory again between them (docs/scene-memory-plan.md).
//

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include "image_levels.hpp"

namespace rs {

// Textures come down this far before a worn item loses any detail: past it they show, as half a hair's
// strands hardly do on what the eye isn't on
inline constexpr uint32_t kDetailAfterExtent = 2048;

// What can give way, in the order it does
struct FitRungs
{
  uint64_t                  idleBytes = 0;  // images nothing samples: nothing is lost
  std::span<const uint64_t> hidden;         // items nothing shows, whole: nothing is lost until shown, then they're made again
  std::span<const uint64_t> settle;         // items, in the order they go: the picture stays, and the next edit waits a moment
  std::span<ImageFit>       images;         // top levels, down to `leastExtent`; the plan leaves them out in place
  uint32_t                  leastExtent = 0;
  // worn items a step down in detail, in the order they go, by what each frees: after textures have come
  // down to `detailAfter`, and the last `detailLast` of them (marked Keep full) after every texture
  std::span<const uint64_t> detail;
  size_t                    detailLast  = 0;
  uint32_t                  detailAfter = 0;
};

struct FitPlan
{
  uint64_t            idle = 0;    // of the idle images' bytes
  std::vector<size_t> hidden;      // of FitRungs::hidden, these let go of
  uint64_t            hiddenBytes = 0;
  std::vector<size_t> settled;     // of FitRungs::settle, these settle
  uint64_t            settleBytes = 0;
  uint64_t            images      = 0;  // what the levels left out free
  std::vector<size_t> detail;           // of FitRungs::detail, these come down a step
  uint64_t            detailBytes = 0;
  uint64_t            freed() const { return idle + hiddenBytes + settleBytes + images + detailBytes; }
};

// What to give up for `wanted` bytes, less when there isn't that much to give up.
inline FitPlan planToFit(uint64_t wanted, const FitRungs& rungs)
{
  FitPlan  out;
  uint64_t left = wanted;
  const auto took = [&](uint64_t bytes) { left -= std::min(left, bytes); };

  out.idle = std::min(left, rungs.idleBytes);
  took(out.idle);

  // the largest first, so that as few as possible have to be made again
  const auto largestFirst = [&](std::span<const uint64_t> bytes, std::vector<size_t>& chosen, uint64_t& sum) {
    std::vector<size_t> order(bytes.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return bytes[a] > bytes[b]; });
    for(size_t i : order)
    {
      if(left == 0)
        break;
      chosen.push_back(i);
      sum += bytes[i];
      took(bytes[i]);
    }
  };
  largestFirst(rungs.hidden, out.hidden, out.hiddenBytes);
  for(size_t i = 0; i < rungs.settle.size() && left > 0; i++)
  {
    out.settled.push_back(i);
    out.settleBytes += rungs.settle[i];
    took(rungs.settle[i]);
  }

  const auto lowerImages = [&](uint32_t least, bool pinnedToo) {
    if(left == 0)
      return;
    const uint64_t freed = lowerToFit(rungs.images, left, least, pinnedToo);
    out.images += freed;
    took(freed);
  };
  const auto giveDetail = [&](size_t from, size_t to) {
    for(size_t i = from; i < to && left > 0; i++)
    {
      out.detail.push_back(i);
      out.detailBytes += rungs.detail[i];
      took(rungs.detail[i]);
    }
  };
  const size_t notLast = rungs.detail.size() - std::min(rungs.detailLast, rungs.detail.size());
  lowerImages(std::max(rungs.leastExtent, rungs.detailAfter), false);
  giveDetail(0, notLast);
  lowerImages(rungs.leastExtent, false);
  lowerImages(rungs.leastExtent, true);
  giveDetail(notLast, rungs.detail.size());
  return out;
}

}  // namespace rs
