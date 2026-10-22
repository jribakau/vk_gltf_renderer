/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <vulkan/vulkan_core.h>

//--------------------------------------------------------------------------------------------------
// Pipeline Cache Manager
//
// Utility class for managing Vulkan pipeline cache with file persistence.
// This class simplifies the creation, loading, and saving of Vulkan pipeline caches.
// Pipeline caches can significantly speed up pipeline creation on subsequent runs.
//--------------------------------------------------------------------------------------------------

namespace nvvk {

// 64-bit FNV-1a over `size` bytes, continuing from `hash`.
constexpr uint64_t kFnv1aOffset = 14695981039346656037ULL;
uint64_t           fnv1a64(const void* data, size_t size, uint64_t hash = kFnv1aOffset);

// Writes `data` to a temporary file named after this process and a serial number, then replaces
// `path` with it, so a reader or a process killed mid-write never sees a truncated file. Creates
// the parent folder. On failure `error` (if given) says why and no temporary file is left.
bool writeFileAtomic(const std::filesystem::path& path, const void* data, size_t size, std::string* error = nullptr);

// Removes the temporary files writeFileAtomic left in `dir` when its process was killed: names
// starting with `prefix` and ending in ".tmp" whose last write is older than `minAge`, so another
// process's write in progress survives.
void removeStaleTempFiles(const std::filesystem::path& dir, std::chrono::seconds minAge, std::string_view prefix = {});

class PipelineCacheManager
{
public:
  PipelineCacheManager() = default;
  ~PipelineCacheManager() { deinit(); }

  // Non-copyable
  PipelineCacheManager(const PipelineCacheManager&)            = delete;
  PipelineCacheManager& operator=(const PipelineCacheManager&) = delete;

  // Create a pipeline cache, optionally loading from a file.
  // Returns VK_SUCCESS on success, error code otherwise.
  VkResult init(VkDevice device, const std::filesystem::path& cacheFilePath = "pipeline_cache.bin");

  // Save the cache to disk and destroy the cache object
  void deinit();

  // Save the current cache to disk (writes a temporary file, then replaces the old one).
  // Returns true on success, false on failure. Skips the write when the cache bytes (size and
  // hash) are the ones last saved or loaded.
  bool save();

  // Copy the cache and write it on a background thread. The caller does not wait.
  // A request made while a write is in flight is remembered and written when that one finishes,
  // so a killed process does not lose the later pipeline. The thread handle is only touched
  // while m_saveMutex is held. VkPipelineCache is internally synchronized (this one is not
  // created with EXTERNALLY_SYNCHRONIZED).
  void saveAsync();

  // Get the VkPipelineCache handle.
  // Returns the pipeline cache handle, or VK_NULL_HANDLE if not initialized.
  VkPipelineCache getCache() const { return m_cache; }

  // Implicit conversion to VkPipelineCache for convenience
  operator VkPipelineCache() const { return m_cache; }

private:
  // Join the writer without holding m_saveMutex across the wait.
  void joinSaveThread();
  void saveThreadMain();
  // vkGetPipelineCacheData, retried while the cache grows (VK_INCOMPLETE).
  bool copyCacheData(std::vector<char>& out) const;
  // writeFileAtomic, logged as a pipeline cache write.
  bool writeBytes(const std::filesystem::path& path, const std::vector<char>& data) const;

  VkDevice              m_device{VK_NULL_HANDLE};
  VkPipelineCache       m_cache{VK_NULL_HANDLE};
  std::filesystem::path m_cacheFilePath;
  size_t                m_initialSize{0};    // Size of cache data loaded from disk at init()
  size_t                m_lastSavedSize{0};  // Size last written, or the loaded size when nothing new was saved
  uint64_t              m_lastSavedHash{0};  // fnv1a64 of those bytes; equal size alone does not mean equal data
  std::mutex            m_saveMutex;  // Guards the thread handle, m_saveInFlight, m_saveAgain, m_lastSavedSize and m_lastSavedHash
  std::thread           m_saveThread;
  bool                  m_saveInFlight{false};
  bool                  m_saveAgain{false};  // A save was requested while the writer was busy
};

}  // namespace nvvk
