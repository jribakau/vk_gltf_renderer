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
// Textures decoded away from the render thread, which never waits for one: a 4096 JPEG is a few hundred
// milliseconds of stb_image, and a lowering's textures come back eight at a time. A few threads below normal
// priority, so the frame and the host come first; what one decodes waits here, by its ticket, until the
// render thread takes it to make into an image in its frame. Any thread.
//

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "gltf_image_loader.hpp"
#include "scene_gpu_backend.hpp"

namespace rs {

class ImageDecodePool
{
public:
  // One decode, on a pool thread: the image's levels, false where it can't be read; `prefetched` when a
  // host had decoded it ahead. What it throws counts as unreadable.
  using Decode = std::function<bool(const ImageSource& source, nvvkgltf::LoadedImageData& out, bool& prefetched)>;

  ImageDecodePool(Decode decode, uint32_t threads);
  // What waits is let go of and what runs finishes: no thread outlives the pool
  ~ImageDecodePool();
  ImageDecodePool(const ImageDecodePool&)            = delete;
  ImageDecodePool& operator=(const ImageDecodePool&) = delete;

  // Its ticket, never 0
  uint64_t start(HeldImageSource source);
  // Not wanted after all: let go of now when it waits or is done, when it finishes when it runs
  void cancel(uint64_t ticket);
  struct Done
  {
    uint64_t                  ticket = 0;
    HeldImageSource           source;
    nvvkgltf::LoadedImageData loaded;
    bool                      ok         = false;
    bool                      prefetched = false;
    uint64_t                  bytes      = 0;  // of its levels
  };
  // The decodes done, by ticket, until `budget` bytes of levels (the first whatever its size)
  std::vector<Done> take(uint64_t budget);
  bool              ready() const;
  // Begun and not yet taken or cancelled
  size_t pending() const;
  // Until nothing waits or runs (tests)
  void waitIdle();

  // The threads a pool of this machine's has: half its cores, at most eight
  static uint32_t defaultThreads();

private:
  struct Job
  {
    uint64_t        ticket = 0;
    HeldImageSource source;
  };
  void run();

  Decode                     m_decode;
  mutable std::mutex         m_mutex;
  std::condition_variable    m_wake, m_idle;
  std::deque<Job>            m_waiting;    // under m_mutex
  std::set<uint64_t>         m_running;    // under m_mutex
  std::set<uint64_t>         m_cancelled;  // of m_running, under m_mutex
  std::map<uint64_t, Done>   m_done;       // under m_mutex
  uint64_t                   m_nextTicket = 1;
  bool                       m_stopping   = false;
  std::vector<std::thread>   m_threads;
};

}  // namespace rs
