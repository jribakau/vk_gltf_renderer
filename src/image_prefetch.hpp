/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>

#include "gltf_image_loader.hpp"

namespace nvvkgltf {

// Images a host decodes ahead on its worker, for the render thread's upload to take instead of
// decoding them there. A file is told apart by its path, time, size and colour space. Any thread. It is
// only ahead of time, so past a budget nothing more is decoded (the upload decodes what isn't here), and
// what nobody takes for a while goes: a load cancelled, a wear that never came.
class ImagePrefetch
{
public:
  void prefetch(const std::filesystem::path& diskPath, bool srgb, uint32_t maxExtent = 0);
  // What was decoded ahead of the file, taken; none when nothing was
  std::optional<LoadedImageData> take(const std::filesystem::path& diskPath, bool srgb);
  // What a scene replace did not take would never be taken
  void drop();
  // What has waited longer than `seconds` for its upload goes
  void dropOlderThan(double seconds);
  // How much may wait. 0: nothing is decoded ahead.
  void setBudget(uint64_t bytes);
  uint64_t bytes() const;

  static constexpr uint64_t kDefaultBudget = uint64_t(1) << 30;

private:
  using Key = std::tuple<std::filesystem::path, std::filesystem::file_time_type, std::uintmax_t, bool>;
  static std::optional<Key> keyOf(const std::filesystem::path& diskPath, bool srgb);
  struct Waiting
  {
    LoadedImageData                       image;
    uint64_t                              bytes = 0;
    std::chrono::steady_clock::time_point at;
  };

  mutable std::mutex     m_mutex;
  std::map<Key, Waiting> m_images;  // under m_mutex
  uint64_t               m_bytes  = 0;
  uint64_t               m_budget = kDefaultBudget;
};

}  // namespace nvvkgltf
