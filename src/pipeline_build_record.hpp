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

//-------------------------------------------------------------------------------------------------------
// The pipelines built before, so a full driver shader cache can be told apart from a new shader.
//
// A pipeline's binary lives in the driver's own disk cache (NVIDIA: %LOCALAPPDATA%\NVIDIA\GLCache);
// pipeline_cache.bin holds little more than a key to it. Once that cache is at its cap the driver keeps
// nothing new, and a pipeline built on the last run compiles again on every start: 15 s for the still's
// on 2026-10-29. A new shader is slow once; a pipeline in this record that is slow again was dropped.
//
// The record is a file of keys (hex, one a line) beside the pipeline cache. A key is the caller's: the
// code, its specialization and flags, and the driver version (an update empties the driver's cache).
//-------------------------------------------------------------------------------------------------------

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>

class PipelineBuildRecord
{
public:
  // Slower than this, a pipeline was compiled: a cached one takes well under a second (0.07-0.7 s
  // measured), a compile of the path tracer 10-25 s
  static constexpr double kCompiledMs = 3000.0;

  // The keys of `path`, which note() then adds to. A missing or unreadable file is an empty record.
  void load(const std::filesystem::path& path)
  {
    std::lock_guard lock(m_mutex);
    m_path = path;
    m_keys.clear();
    std::ifstream in(path);
    for(std::string line; std::getline(in, line);)
      if(!line.empty())
        try
        {
          m_keys.insert(std::stoull(line, nullptr, 16));
        }
        catch(...)  // a line cut short by a killed write
        {
        }
  }

  // A pipeline of `key` built in `ms`. True when it was built before and compiled again: the driver
  // did not keep it. Any thread.
  bool note(uint64_t key, double ms)
  {
    std::lock_guard lock(m_mutex);
    if(m_keys.insert(key).second)
    {
      if(!m_path.empty())
        if(std::ofstream out(m_path, std::ios::app); out)
          out << std::hex << key << '\n';
      return false;
    }
    if(ms < kCompiledMs)
      return false;
    m_dropped = true;
    return true;
  }

  // A note() since load() said the driver dropped a pipeline
  bool driverDropped() const
  {
    std::lock_guard lock(m_mutex);
    return m_dropped;
  }

private:
  mutable std::mutex           m_mutex;
  std::filesystem::path        m_path;
  std::unordered_set<uint64_t> m_keys;
  bool                         m_dropped = false;
};
