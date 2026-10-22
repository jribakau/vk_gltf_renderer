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
// Latency trace of a live edit (temporary diagnostics). A host stamps the stages on its side and
// passes them as vk_gltf_edit's "trace"; the MCP handler, the renderer and the frame stream add
// theirs, and the stream logs one `[trace]` line when the first frame showing the edit is encoded.
// Stamps are epoch milliseconds, the clock a browser's performance.timeOrigin + now() reads.
//

#include <chrono>
#include <string>
#include <utility>
#include <vector>

using LatencyTrace = std::vector<std::pair<std::string, double>>;

inline double traceNowMs()
{
  return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// "[trace] input +0.0 | morphed +12.3 (12.3) | ..." : each stage since the first, and since the one before.
inline std::string formatTrace(const LatencyTrace& t)
{
  std::string out = "[trace]";
  if(t.empty())
    return out;
  char buf[96];
  for(size_t i = 0; i < t.size(); i++)
  {
    const double total = t[i].second - t[0].second;
    const double step  = i ? t[i].second - t[i - 1].second : 0.0;
    snprintf(buf, sizeof(buf), "%s %s +%.1f (%.1f)", i ? " |" : "", t[i].first.c_str(), total, step);
    out += buf;
  }
  return out;
}
