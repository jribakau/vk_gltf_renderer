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

#include "image_prefetch.hpp"

#include <nvutils/file_mapping.hpp>

namespace nvvkgltf {

std::optional<ImagePrefetch::Key> ImagePrefetch::keyOf(const std::filesystem::path& diskPath, bool srgb)
{
  if(diskPath.empty())
    return std::nullopt;
  std::error_code             ec;
  const std::filesystem::path absolute = std::filesystem::absolute(diskPath, ec);
  if(ec)
    return std::nullopt;
  const std::filesystem::directory_entry entry(absolute, ec);
  if(ec)
    return std::nullopt;
  const std::uintmax_t size = entry.file_size(ec);
  if(ec)
    return std::nullopt;
  const std::filesystem::file_time_type mtime = entry.last_write_time(ec);
  if(ec)
    return std::nullopt;
  return Key{absolute, mtime, size, srgb};
}

void ImagePrefetch::prefetch(const std::filesystem::path& diskPath, bool srgb, uint32_t maxExtent)
{
  const std::optional<Key> key = keyOf(diskPath, srgb);
  if(!key)
    return;
  {
    std::lock_guard lock(m_mutex);
    if(m_images.count(*key) || m_bytes >= m_budget)
      return;
  }
  nvutils::FileReadMapping fileMapping;
  if(!fileMapping.open(diskPath))
    return;
  LoadedImageData loaded;
  if(!loadFromMemory(loaded, fileMapping.data(), fileMapping.size(), srgb, 0, maxExtent))
    return;
  uint64_t bytes = 0;
  for(const std::vector<char>& level : loaded.mipData)
    bytes += level.size();
  std::lock_guard lock(m_mutex);
  // two asked for it at once: one is kept
  if(const auto [it, made] = m_images.try_emplace(*key); made)
  {
    it->second = {std::move(loaded), bytes, std::chrono::steady_clock::now()};
    m_bytes += bytes;
  }
}

std::optional<LoadedImageData> ImagePrefetch::take(const std::filesystem::path& diskPath, bool srgb)
{
  const std::optional<Key> key = keyOf(diskPath, srgb);
  if(!key)
    return std::nullopt;
  std::lock_guard lock(m_mutex);
  auto            it = m_images.find(*key);
  if(it == m_images.end())
    return std::nullopt;
  std::optional<LoadedImageData> taken = std::move(it->second.image);
  m_bytes -= it->second.bytes;
  m_images.erase(it);
  return taken;
}

void ImagePrefetch::drop()
{
  std::lock_guard lock(m_mutex);
  m_images.clear();
  m_bytes = 0;
}

void ImagePrefetch::dropOlderThan(double seconds)
{
  const auto      before = std::chrono::steady_clock::now() - std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
  std::lock_guard lock(m_mutex);
  std::erase_if(m_images, [&](const auto& entry) {
    if(entry.second.at > before)
      return false;
    m_bytes -= entry.second.bytes;
    return true;
  });
}

void ImagePrefetch::setBudget(uint64_t bytes)
{
  std::lock_guard lock(m_mutex);
  m_budget = bytes;
}

uint64_t ImagePrefetch::bytes() const
{
  std::lock_guard lock(m_mutex);
  return m_bytes;
}

}  // namespace nvvkgltf
