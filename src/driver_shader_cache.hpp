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
// A shader cache of the renderer's own for NVIDIA's driver, kept apart from Vulkan so it can be tested.
//
// The driver keeps the binaries of the pipelines it compiles in a disk cache of its own (on Windows
// %LOCALAPPDATA%\NVIDIA\GLCache); pipeline_cache.bin holds little more than keys into it. At its cap
// it keeps nothing new and evicts nothing, so once the binaries of earlier builds had filled it, every
// start compiled the path tracer's two pipelines again: 25 and 28 s for a cold build of each
// (2026-11-03, with an empty cache), 71 and 145 ms from a warm one.
//
// So the renderer hands the driver a folder beside the pipeline cache, with a cap of its own, through
// the variables NVIDIA's driver reads (__GL_SHADER_DISK_CACHE_PATH and _SIZE; its Windows driver honours
// both, measured 2026-11-03), and empties it before the driver opens it: when it is near that cap, or
// when the last run found the driver had dropped a pipeline (PipelineBuildRecord), whatever the reason.
// An emptied cache costs one cold start; a full one cost every start. A path already set in the
// environment is left alone.
//-------------------------------------------------------------------------------------------------------

#include <cstdint>
#include <filesystem>
#include <system_error>

#include "pipeline_build_record.hpp"

namespace driver_shader_cache {

inline constexpr const char* kPathVariable = "__GL_SHADER_DISK_CACHE_PATH";
inline constexpr const char* kSizeVariable = "__GL_SHADER_DISK_CACHE_SIZE";
// A build of the path tracer leaves 17 MB or more there, and the builds of a busy day of shader work
// about 400 MB: room for a week or two of that, and nothing a user who doesn't rebuild comes near
inline constexpr uint64_t kCapBytes = uint64_t(4) << 30;
// Emptied at this share of the cap, before the driver stops keeping what it compiles
inline constexpr double kEmptyAtShare = 0.9;

enum class Emptied
{
  eNo,
  eDropped,  // the last run found a pipeline built before compiled again
  eNearCap,  // the folder was near its cap
};

struct Setup
{
  std::filesystem::path folder;              // the driver's cache: give it the driver
  Emptied               emptied = Emptied::eNo;
  uint64_t              bytes   = 0;         // what it held before (and still holds, unless emptied)
};

// The bytes of the files under `dir`; 0 for a folder that isn't there
inline uint64_t folderBytes(const std::filesystem::path& dir)
{
  uint64_t        bytes = 0;
  std::error_code ec;
  for(std::filesystem::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
  {
    std::error_code sizeEc;
    if(it->is_regular_file(sizeEc))
      if(const uint64_t size = it->file_size(sizeEc); !sizeEc)
        bytes += size;
  }
  return bytes;
}

// The driver's cache under `cacheRoot` (the pipeline cache's folder), emptied first when it should be.
// Before Vulkan starts: the driver opens its cache with the device, and holds its files from then on.
inline Setup prepare(const std::filesystem::path& cacheRoot, uint64_t capBytes = kCapBytes)
{
  Setup s;
  s.folder = cacheRoot / "driver";
  s.bytes  = folderBytes(s.folder);
  std::error_code existsEc;
  const bool      fresh = !std::filesystem::exists(s.folder, existsEc);

  const std::filesystem::path recordPath = cacheRoot / PipelineBuildRecord::kFileName;
  PipelineBuildRecord         record;
  record.load(recordPath);
  if(record.droppedOnRecord())
    s.emptied = Emptied::eDropped;
  else if(double(s.bytes) >= kEmptyAtShare * double(capBytes))
    s.emptied = Emptied::eNearCap;

  if(s.emptied != Emptied::eNo)
  {
    std::error_code ec;
    std::filesystem::remove_all(s.folder, ec);
  }
  // The keys were of what the driver held: built again into an emptied cache, or a new one (the first
  // start with a folder of its own, whose keys were of the driver's shared cache), they are no drop
  if(s.emptied != Emptied::eNo || fresh)
    PipelineBuildRecord::forgetKeys(recordPath);
  std::error_code ec;
  std::filesystem::create_directories(s.folder, ec);
  return s;
}

}  // namespace driver_shader_cache
