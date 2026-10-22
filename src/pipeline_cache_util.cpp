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

//
// Vulkan pipeline cache manager. Loads a previously saved pipeline cache
// from disk at startup, creates a VkPipelineCache from it, and writes
// the updated cache back to disk on shutdown. Reduces pipeline compilation
// time across application runs.
//

#include "pipeline_cache_util.hpp"
#include <atomic>
#include <fstream>
#include <vector>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace nvvk {

uint64_t fnv1a64(const void* data, size_t size, uint64_t hash)
{
  constexpr uint64_t kPrime = 0x100000001b3ULL;
  const auto*        bytes  = static_cast<const uint8_t*>(data);
  for(size_t i = 0; i < size; ++i)
  {
    hash ^= bytes[i];
    hash *= kPrime;
  }
  return hash;
}

bool writeFileAtomic(const std::filesystem::path& path, const void* data, size_t size, std::string* error)
{
  auto fail = [&](std::string message) {
    if(error)
      *error = std::move(message);
    return false;
  };
  std::error_code ec;
  if(path.has_parent_path())
    std::filesystem::create_directories(path.parent_path(), ec);

  // A shared ".tmp" name lets two processes truncate each other's write.
  static std::atomic<uint64_t> writeSerial{0};
#ifdef _WIN32
  const unsigned long pid = GetCurrentProcessId();
#else
  const unsigned long pid = static_cast<unsigned long>(::getpid());
#endif
  std::filesystem::path tmp = path;
  tmp += "." + std::to_string(pid) + "." + std::to_string(writeSerial.fetch_add(1)) + ".tmp";
  {
    std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
    if(!file.is_open())
      return fail("cannot open " + nvutils::utf8FromPath(tmp));
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    file.flush();
    const bool writeOk = !file.fail();
    file.close();
    if(!writeOk)
    {
      std::filesystem::remove(tmp, ec);
      return fail("cannot write " + nvutils::utf8FromPath(tmp));
    }
  }

#ifdef _WIN32
  // std::filesystem::rename fails on Windows when the destination already exists.
  if(!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
  {
    const unsigned long lastError = GetLastError();
    std::filesystem::remove(tmp, ec);
    return fail("cannot replace " + nvutils::utf8FromPath(path) + " (error " + std::to_string(lastError) + ")");
  }
#else
  std::filesystem::rename(tmp, path, ec);
  if(ec)
  {
    const std::string reason = ec.message();
    std::filesystem::remove(tmp, ec);
    return fail("cannot replace " + nvutils::utf8FromPath(path) + " (" + reason + ")");
  }
#endif
  return true;
}

void removeStaleTempFiles(const std::filesystem::path& dir, std::chrono::seconds minAge, std::string_view prefix)
{
  std::error_code ec;
  const auto      cutoff = std::filesystem::file_time_type::clock::now() - minAge;
  for(std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
  {
    std::error_code entryEc;
    if(it->path().extension() != ".tmp" || !nvutils::utf8FromPath(it->path().filename()).starts_with(prefix)
       || !it->is_regular_file(entryEc))
      continue;
    const auto written = it->last_write_time(entryEc);
    if(!entryEc && written < cutoff)
      std::filesystem::remove(it->path(), entryEc);
  }
}

VkResult PipelineCacheManager::init(VkDevice device, const std::filesystem::path& cacheFilePath)
{
  m_device        = device;
  m_cacheFilePath = cacheFilePath;

  std::vector<char> cacheData;

  // Try to load existing cache data from disk
  if(std::filesystem::exists(m_cacheFilePath))
  {
    try
    {
      std::ifstream file(m_cacheFilePath, std::ios::binary | std::ios::ate);
      if(file.is_open())
      {
        size_t           fileSize                  = static_cast<size_t>(file.tellg());
        constexpr size_t kMaxPipelineCacheFileSize = 256ULL * 1024 * 1024;  // 256 MB cap to avoid unbounded allocation
        if(fileSize > 0 && fileSize <= kMaxPipelineCacheFileSize)
        {
          cacheData.resize(fileSize);
          file.seekg(0);
          file.read(cacheData.data(), fileSize);
          file.close();
          m_initialSize   = fileSize;
          m_lastSavedSize = fileSize;
          m_lastSavedHash = fnv1a64(cacheData.data(), cacheData.size());
          LOGI("Loaded pipeline cache from %s (%zu bytes)\n", m_cacheFilePath.string().c_str(), fileSize);
        }
        else if(fileSize > kMaxPipelineCacheFileSize)
        {
          LOGW("Pipeline cache file too large (%zu bytes), ignoring\n", fileSize);
        }
      }
    }
    catch(const std::exception& e)
    {
      LOGW("Failed to load pipeline cache: %s\n", e.what());
      cacheData.clear();
    }
  }

  // Create the pipeline cache
  VkPipelineCacheCreateInfo cacheInfo{
      .sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .initialDataSize = cacheData.size(),
      .pInitialData    = cacheData.empty() ? nullptr : cacheData.data(),
  };

  VkResult result = vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_cache);
  if(result == VK_SUCCESS)
  {
    NVVK_DBG_NAME(m_cache);
    if(cacheData.empty())
    {
      LOGI("Created new pipeline cache\n");
    }
    else
    {
      LOGI("Created pipeline cache with existing data\n");
    }
  }
  else
  {
    LOGE("Failed to create pipeline cache: %d\n", result);
  }

  return result;
}

void PipelineCacheManager::joinSaveThread()
{
  std::thread writer;
  {
    std::lock_guard<std::mutex> lock(m_saveMutex);
    if(m_saveThread.joinable())
      writer = std::move(m_saveThread);
  }
  if(writer.joinable())
    writer.join();
}

void PipelineCacheManager::deinit()
{
  // The writer only touches the copied bytes, but it renames the same file save() writes.
  joinSaveThread();
  if(m_cache != VK_NULL_HANDLE)
  {
    save();
    vkDestroyPipelineCache(m_device, m_cache, nullptr);
    m_cache  = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
  }
}

bool PipelineCacheManager::copyCacheData(std::vector<char>& out) const
{
  size_t cacheSize = 0;
  if(vkGetPipelineCacheData(m_device, m_cache, &cacheSize, nullptr) != VK_SUCCESS)
    return false;
  if(cacheSize == 0)
  {
    out.clear();
    return true;
  }

  // The cache can grow between the size query and the copy. VK_INCOMPLETE updates cacheSize.
  for(int attempt = 0; attempt < 4; ++attempt)
  {
    out.resize(cacheSize);
    const VkResult result = vkGetPipelineCacheData(m_device, m_cache, &cacheSize, out.data());
    if(result == VK_SUCCESS)
    {
      out.resize(cacheSize);
      return true;
    }
    if(result != VK_INCOMPLETE)
      return false;
  }
  return false;
}

bool PipelineCacheManager::writeBytes(const std::filesystem::path& path, const std::vector<char>& data) const
{
  std::string error;
  try
  {
    if(writeFileAtomic(path, data.data(), data.size(), &error))
      return true;
  }
  catch(const std::exception& e)
  {
    error = e.what();
  }
  LOGW("Failed to save pipeline cache: %s\n", error.c_str());
  return false;
}

bool PipelineCacheManager::save()
{
  joinSaveThread();
  if(m_cache == VK_NULL_HANDLE)
    return false;

  std::vector<char> cacheData;
  if(!copyCacheData(cacheData))
  {
    LOGW("Failed to retrieve pipeline cache data\n");
    return false;
  }
  if(cacheData.empty())
  {
    LOGI("Pipeline cache is empty, not saving\n");
    return true;
  }
  const uint64_t hash = fnv1a64(cacheData.data(), cacheData.size());
  if(cacheData.size() == m_lastSavedSize && hash == m_lastSavedHash)
    return true;

  if(!writeBytes(m_cacheFilePath, cacheData))
    return false;

  m_lastSavedSize       = cacheData.size();
  m_lastSavedHash       = hash;
  const ptrdiff_t delta = static_cast<ptrdiff_t>(cacheData.size()) - static_cast<ptrdiff_t>(m_initialSize);
  LOGI("Saved pipeline cache to %s (%zu bytes, %+td vs load)\n", m_cacheFilePath.string().c_str(), cacheData.size(), delta);
  return true;
}

void PipelineCacheManager::saveThreadMain()
{
  for(;;)
  {
    std::vector<char> cacheData;
    const bool        copied = copyCacheData(cacheData);
    // Hashed outside the lock: tens of MB take a while.
    const uint64_t hash = copied ? fnv1a64(cacheData.data(), cacheData.size()) : 0;

    std::filesystem::path path;
    size_t                initialSize = 0;
    {
      std::lock_guard<std::mutex> lock(m_saveMutex);
      // A request arrived while this copy was running, so the bytes may be stale.
      if(m_saveAgain && m_cache != VK_NULL_HANDLE)
      {
        m_saveAgain = false;
        continue;
      }
      if(!copied || m_cache == VK_NULL_HANDLE)
      {
        if(!copied)
          LOGW("Failed to retrieve pipeline cache data\n");
        m_saveInFlight = false;
        return;
      }
      path        = m_cacheFilePath;
      initialSize = m_initialSize;
      if(cacheData.empty() || (cacheData.size() == m_lastSavedSize && hash == m_lastSavedHash))
      {
        m_saveInFlight = false;
        return;
      }
    }

    const bool ok = writeBytes(path, cacheData);
    std::lock_guard<std::mutex> lock(m_saveMutex);
    if(ok)
    {
      m_lastSavedSize       = cacheData.size();
      m_lastSavedHash       = hash;
      const ptrdiff_t delta = static_cast<ptrdiff_t>(cacheData.size()) - static_cast<ptrdiff_t>(initialSize);
      LOGI("Saved pipeline cache to %s (%zu bytes, %+td vs load)\n", path.string().c_str(), cacheData.size(), delta);
    }
    if(m_saveAgain && m_cache != VK_NULL_HANDLE)
    {
      m_saveAgain = false;
      continue;
    }
    m_saveInFlight = false;
    return;
  }
}

void PipelineCacheManager::saveAsync()
{
  std::lock_guard<std::mutex> lock(m_saveMutex);
  if(m_cache == VK_NULL_HANDLE)
    return;
  if(m_saveInFlight)
  {
    m_saveAgain = true;
    return;
  }
  // The previous writer has left the lock and will not take it again.
  if(m_saveThread.joinable())
    m_saveThread.join();
  m_saveAgain    = false;
  m_saveInFlight = true;
  m_saveThread   = std::thread([this] { saveThreadMain(); });
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
static void usage_PipelineCacheManager()
{
  VkDevice device = nullptr;  // EX: get the device from the app (m_app->getDevice())

  nvvk::PipelineCacheManager pipelineCacheManager;

  // Initialize the pipeline cache, loading from file if it exists
  pipelineCacheManager.init(device, "pipeline_cache.bin");

  // Use the cache when creating pipelines
  VkPipelineCache cache = pipelineCacheManager.getCache();

  VkGraphicsPipelineCreateInfo pipelineInfo = {};  // EX: fill in pipeline create info
  VkPipeline                   pipeline     = VK_NULL_HANDLE;
  vkCreateGraphicsPipelines(device, cache, 1, &pipelineInfo, nullptr, &pipeline);

  // The cache can also be used implicitly via conversion operator
  vkCreateGraphicsPipelines(device, pipelineCacheManager, 1, &pipelineInfo, nullptr, &pipeline);

  // Save the cache manually (optional, as it's saved automatically on deinit)
  pipelineCacheManager.save();

  // Cleanup - this will save the cache to disk and destroy it
  pipelineCacheManager.deinit();
}

}  // namespace nvvk
