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
// Binary payloads a host uploads ahead of a live edit. A whole figure's vertices are megabytes;
// as base64 inside a vk_gltf_edit call they cross the host's IPC, the MCP layer's JSON-RPC parse
// and the tool's own parse, which took ~500 ms a dial step. Posted raw to the frame stream's
// /blob instead, they are stored here and the edit names them, {"blob": id}, in place of the
// base64 string (mcp_scene.cpp). Each is taken once. Uploads nobody takes (a failed call) are
// dropped oldest first past the cap (kMaxBytes).
//

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

class BlobStore
{
public:
  static constexpr size_t kMaxBytes = size_t(512) << 20;

  explicit BlobStore(size_t maxBytes = kMaxBytes)
      : m_maxBytes(maxBytes)
  {
  }

  // Any thread.
  uint64_t put(std::string bytes)
  {
    std::lock_guard lock(m_mutex);
    const uint64_t id = m_next++;
    m_bytes += bytes.size();
    m_blobs.emplace_back(id, std::move(bytes));
    while(m_bytes > m_maxBytes && m_blobs.size() > 1)
    {
      m_bytes -= m_blobs.front().second.size();
      m_blobs.pop_front();
    }
    return id;
  }

  // Any thread. Removes the blob; nullopt when it was never stored, already taken or dropped.
  std::optional<std::string> take(uint64_t id)
  {
    std::lock_guard lock(m_mutex);
    for(auto it = m_blobs.begin(); it != m_blobs.end(); ++it)
    {
      if(it->first != id)
        continue;
      std::string bytes = std::move(it->second);
      m_bytes -= bytes.size();
      m_blobs.erase(it);
      return bytes;
    }
    return std::nullopt;
  }

private:
  size_t                                       m_maxBytes;
  std::mutex                                   m_mutex;
  uint64_t                                     m_next{1};
  size_t                                       m_bytes{0};
  std::deque<std::pair<uint64_t, std::string>> m_blobs;
};
