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

// What can give way, in the order it does
struct FitRungs
{
  uint64_t                  idleBytes = 0;  // images nothing samples: nothing is lost
  std::span<const uint64_t> hidden;         // items nothing shows, whole: nothing is lost until shown, then they're made again
  std::span<ImageFit>       images;         // top levels, down to `leastExtent`; the plan leaves them out in place
  uint32_t                  leastExtent = 0;
};

struct FitPlan
{
  uint64_t            idle = 0;    // of the idle images' bytes
  std::vector<size_t> hidden;      // of FitRungs::hidden, these let go of
  uint64_t            hiddenBytes = 0;
  uint64_t            images      = 0;  // what the levels left out free
  uint64_t            freed() const { return idle + hiddenBytes + images; }
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
  std::vector<size_t> order(rungs.hidden.size());
  std::iota(order.begin(), order.end(), size_t(0));
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rungs.hidden[a] > rungs.hidden[b]; });
  for(size_t i : order)
  {
    if(left == 0)
      break;
    out.hidden.push_back(i);
    out.hiddenBytes += rungs.hidden[i];
    took(rungs.hidden[i]);
  }

  if(left > 0)
  {
    const uint64_t freed = lowerToFit(rungs.images, left, rungs.leastExtent);
    out.images += freed;
    took(freed);
  }
  if(left > 0)
    out.images += lowerToFit(rungs.images, left, rungs.leastExtent, true);
  return out;
}

}  // namespace rs
