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

// What the path tracer threw away as not finite, summed over an accumulation from the frames' blocks
// (shaders/invalid_samples.h.slang), and the line native.log gets for it. Vulkan-free, for the tests.

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "shaders/invalid_samples.h.slang"

namespace invalid_samples {

struct Pixel
{
  uint32_t x    = 0;
  uint32_t y    = 0;
  uint32_t kind = 0;  // INVALID_SAMPLE_*
};

struct Tally
{
  std::array<uint64_t, INVALID_SAMPLE_KINDS> counts{};
  std::vector<Pixel>                         first;  // the first few, in the order the frames gave them
  uint64_t                                   frames = 0;

  static constexpr size_t kKept = 8;  // pixels a log line names

  bool     empty() const { return total() == 0; }
  uint64_t total() const { return counts[0] + counts[1] + counts[2]; }

  // One frame's block, INVALID_SAMPLES_WORDS uints
  void add(const uint32_t* block)
  {
    frames++;
    for(uint32_t k = 0; k < INVALID_SAMPLE_KINDS; k++)
      counts[k] += block[k];
    const uint32_t records = std::min<uint32_t>(block[INVALID_SAMPLES_OFFERED], INVALID_SAMPLES_KEPT);
    for(uint32_t i = 0; i < records && first.size() < kKept; i++)
    {
      const uint32_t r = block[INVALID_SAMPLES_FIRST + i];
      first.push_back({shaderio::invalidSampleX(r), shaderio::invalidSampleY(r), shaderio::invalidSampleKind(r)});
    }
  }
};

inline const char* kindName(uint32_t kind)
{
  return kind == INVALID_SAMPLE_COLOR ? "colour" : kind == INVALID_SAMPLE_DLSS_GUIDE ? "DLSS guide" : "still guide";
}

// "[invalid] 3 samples dropped, 0 DLSS guide pixels and 1 still guide pixel replaced over 40 frames; first at (12, 7) colour, ..."
inline std::string describe(const Tally& t)
{
  auto plural = [](uint64_t n, const char* one, const char* many) { return fmt::format("{} {}", n, n == 1 ? one : many); };
  std::string line = fmt::format("[invalid] {} dropped, {} and {} replaced over {} frame{}", plural(t.counts[INVALID_SAMPLE_COLOR], "sample", "samples"),
                                 plural(t.counts[INVALID_SAMPLE_DLSS_GUIDE], "DLSS guide pixel", "DLSS guide pixels"),
                                 plural(t.counts[INVALID_SAMPLE_STILL_GUIDE], "still guide pixel", "still guide pixels"),
                                 t.frames, t.frames == 1 ? "" : "s");
  for(size_t i = 0; i < t.first.size(); i++)
    line += fmt::format("{}({}, {}) {}", i == 0 ? "; first at " : ", ", t.first[i].x, t.first[i].y, kindName(t.first[i].kind));
  return line;
}

}  // namespace invalid_samples
