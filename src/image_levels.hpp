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
// Which of an image's stored mip levels to keep: Max texture size's limit, and lowering images to fit
// GPU memory. Vulkan-free, so the residency's tests reach it.
//

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rs {

inline uint32_t levelExtent(uint32_t base, uint32_t level)
{
  return std::max(base >> level, 1u);
}

// A level can start an image only when its sides are whole 4x4 blocks (the cooker's read_levels has
// the same rule): a BC7 image's lower levels would otherwise not be the sizes its chain stores.
inline bool canStartAt(uint32_t width, uint32_t height, uint32_t level)
{
  return levelExtent(width, level) % 4 == 0 && levelExtent(height, level) % 4 == 0;
}

// The levels to leave out at the top so the long edge is at most `maxExtent`, or as near as the rule
// above allows; 0: no limit.
inline uint32_t levelsToSkip(uint32_t width, uint32_t height, uint32_t levels, uint32_t maxExtent)
{
  if(maxExtent == 0)
    return 0;
  uint32_t skip = 0;
  while(skip + 1 < levels && std::max(levelExtent(width, skip), levelExtent(height, skip)) > maxExtent
        && canStartAt(width, height, skip + 1))
    skip++;
  return skip;
}

// An image's stored mip chain, read without decoding it
struct ImageChain
{
  uint32_t              width = 0, height = 0;
  std::vector<uint64_t> levelBytes;
};

// An image being fitted: its chain, and the levels left out so far
struct ImageFit
{
  ImageChain chain;
  uint32_t   skip = 0;

  uint64_t bytes() const
  {
    uint64_t sum = 0;
    for(size_t i = skip; i < chain.levelBytes.size(); i++)
      sum += chain.levelBytes[i];
    return sum;
  }
  uint32_t extent() const { return std::max(levelExtent(chain.width, skip), levelExtent(chain.height, skip)); }
  // whether its top level can go, leaving a long edge of at least `minExtent`
  bool canLower(uint32_t minExtent) const
  {
    return skip + 1 < chain.levelBytes.size() && canStartAt(chain.width, chain.height, skip + 1)
           && std::max(levelExtent(chain.width, skip + 1), levelExtent(chain.height, skip + 1)) >= minExtent;
  }
};

// Leaves out top levels until `bytes` are freed, whichever image's top level is largest first (the
// earlier of a tie), so the largest come down to the others before any falls below them. The bytes
// freed: less than asked when every image is down to `minExtent`.
inline uint64_t lowerToFit(std::span<ImageFit> images, uint64_t bytes, uint32_t minExtent)
{
  uint64_t freed = 0;
  while(freed < bytes)
  {
    ImageFit* largest = nullptr;
    for(ImageFit& image : images)
      if(image.canLower(minExtent) && (!largest || image.chain.levelBytes[image.skip] > largest->chain.levelBytes[largest->skip]))
        largest = &image;
    if(!largest)
      break;
    freed += largest->chain.levelBytes[largest->skip++];
  }
  return freed;
}

}  // namespace rs
