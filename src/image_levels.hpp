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
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rs {

// What first hits want of a material, by how large a long edge they could show: 256 << bin (shaders/hit_signal.h.slang)
inline constexpr uint32_t kHitBins = 6;
// of the scene's first hits: a level that fewer would miss is as good as one that none would
inline constexpr double kLeastHitLoss = 1.0 / 1024;

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
  bool       favoured = false;  // what the eye goes to first: it stays at least a level above the others
  bool       pinned   = false;  // marked Keep full: it comes down only when nothing else can
  bool       wholeOnly = false;  // no stored levels to read: it can come down, and comes back all at once
  // what the scene's first hits want of it, each bin a share of them all; none: not known yet
  std::optional<std::array<float, kHitBins>> wanting;

  // the share of the scene's first hits that would show less were level `level` gone
  double lossWithout(uint32_t level) const
  {
    const uint32_t kept = std::max(levelExtent(chain.width, level + 1), levelExtent(chain.height, level + 1));
    double         loss = 0;
    for(uint32_t bin = 0; bin < kHitBins; bin++)
      if((256u << bin) > kept)
        loss += (*wanting)[bin];
    return loss;
  }
  // what a level counts as against the others', bytes freed for what is lost: a favoured image's as the level
  // under it, and with hits known, by how few want it. What none want goes before all that anything does.
  double weight(uint32_t level) const
  {
    const double bytes = double(chain.levelBytes[level]) / (favoured ? 4.0 : 1.0);
    return wanting ? bytes / std::max(lossWithout(level), kLeastHitLoss) : bytes;
  }
  // whether this image's level `level` goes before `other`'s `otherLevel` when images come down:
  // the heavier, and of equals the one not favoured
  bool goesBefore(uint32_t level, const ImageFit& other, uint32_t otherLevel) const
  {
    const double mine = weight(level), theirs = other.weight(otherLevel);
    return mine > theirs || (mine == theirs && !favoured && other.favoured);
  }
  uint64_t bytes() const
  {
    uint64_t sum = 0;
    for(size_t i = skip; i < chain.levelBytes.size(); i++)
      sum += chain.levelBytes[i];
    return sum;
  }
  uint64_t wholeBytes() const
  {
    uint64_t sum = 0;
    for(uint64_t level : chain.levelBytes)
      sum += level;
    return sum;
  }
  uint32_t extent() const { return std::max(levelExtent(chain.width, skip), levelExtent(chain.height, skip)); }
  // whether its top level can go, leaving a long edge of at least `minExtent`
  bool canLower(uint32_t minExtent) const
  {
    return skip + 1 < chain.levelBytes.size() && canStartAt(chain.width, chain.height, skip + 1)
           && std::max(levelExtent(chain.width, skip + 1), levelExtent(chain.height, skip + 1)) >= minExtent;
  }
  // whether a level can go back on top, with `wanted` the most it may leave out
  bool canRaise(uint32_t wanted) const { return skip > wanted && (wholeOnly || skip == 1 || canStartAt(chain.width, chain.height, skip - 1)); }
};

// Leaves out top levels until `bytes` are freed, whichever image's top level weighs most first (the
// earlier of a tie), so the largest come down to the others before any falls below them. The bytes
// freed: less than asked when every image is down to `minExtent`. What is pinned stays, unless `pinnedToo`.
inline uint64_t lowerToFit(std::span<ImageFit> images, uint64_t bytes, uint32_t minExtent, bool pinnedToo = false)
{
  uint64_t freed = 0;
  while(freed < bytes)
  {
    ImageFit* largest = nullptr;
    for(ImageFit& image : images)
      if((pinnedToo || !image.pinned) && image.canLower(minExtent) && (!largest || image.goesBefore(image.skip, *largest, largest->skip)))
        largest = &image;
    if(!largest)
      break;
    freed += largest->chain.levelBytes[largest->skip++];
  }
  return freed;
}

// lowerToFit undone: top levels put back, what is pinned and then the lightest first (the later of a
// tie), down to each image's `wanted` levels left out, while what they take fits `room`. An image is
// decoded again whole beside the one it replaces, so its first level back counts its whole chain. The
// bytes they take; stops at the first that doesn't fit, so none gets ahead of the others.
inline uint64_t raiseToFit(std::span<ImageFit> images, std::span<const uint32_t> wanted, uint64_t room)
{
  std::vector<uint32_t> before;
  for(const ImageFit& image : images)
    before.push_back(image.skip);
  uint64_t taken = 0;
  while(true)
  {
    size_t next = images.size();
    for(size_t i = 0; i < images.size(); i++)
      if(images[i].canRaise(wanted[i])
         && (next == images.size() || (images[i].pinned && !images[next].pinned)
             || (images[i].pinned == images[next].pinned && !images[i].goesBefore(images[i].skip - 1, images[next], images[next].skip - 1))))
        next = i;
    if(next == images.size())
      break;
    ImageFit& image = images[next];
    if(image.wholeOnly)
    {
      const uint64_t whole = image.wholeBytes();
      if(taken + whole > room)
        break;
      taken += whole;
      image.skip = 0;
      continue;
    }
    const uint64_t cost = image.chain.levelBytes[image.skip - 1] + (image.skip == before[next] ? image.bytes() : 0);
    if(taken + cost > room)
      break;
    taken += cost;
    image.skip--;
  }
  return taken;
}

}  // namespace rs
