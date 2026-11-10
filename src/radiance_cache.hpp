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

#include <functional>

#include <glm/glm.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/resources.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"

//-------------------------------------------------------------------------------------------------------
// RadianceCache: the moving view's bounce cache (shaders/radiance_cache.h.slang), the host's part. A
// table of cells on the device, which the frames DLSS draws read and feed through its address in the
// frame's info, and a pass after each such frame that folds the frame's samples into the cells' means
// (shaders/radiance_cache_resolve.slang).
//
//   finfo.radianceCache... = cache.prepare(resources, dlss);   // before the frame's info is written
//   cache.cmdBeforeTrace(cmd, resources);                      // each traced frame
//   cache.cmdResolve(cmd, profiler);                           // after the trace of a frame DLSS draws
//
// The table is made when the setting (ptRadianceCache) first has a frame to serve, and kept until the
// renderer goes: with the setting off no frame is given it, and it starts empty when the setting comes
// back. Its light lives across stills: a view that moves again finds the room's where it left it, cut to
// a few samples' worth if something other than the camera changed meanwhile (cmdBeforeTrace), and
// emptied when the scene is replaced (clear).
//-------------------------------------------------------------------------------------------------------
class RadianceCache
{
public:
  // What a frame's info says of the cache (SceneFrameInfo::radianceCache and its fields)
  struct Frame
  {
    uint32_t* table = nullptr;  // null: no cache this frame
    int       slots = 0;
    float     cell  = 0.04F;
    int       frame = 0;
    bool      still = false;  // an accumulation's frames read it too (ptRadianceCacheStill)
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  void registerParameters(SettingsRegistry* settings, std::function<void()> reset);

  // Whether the frames DLSS draws (`dlssFrame`), or an accumulation's, read the cache: the first with the
  // setting on, the second only with the debug setting beside it
  bool reads(bool dlssFrame) const { return m_on && (dlssFrame || m_still); }

  // Before the frame's info is written: the table made at the setting's first frame (`dlss`: DLSS is on,
  // so some frames will read it), and where the frame finds it (nowhere with the setting off)
  Frame prepare(Resources& resources, bool dlss);
  // Each traced frame, still or moving, before its trace: a new table is cleared, and a change to the
  // scene since the last frame (Resources::sceneChanges) is noted for the next resolve
  void cmdBeforeTrace(VkCommandBuffer cmd, const Resources& resources);
  // The scene was replaced: nothing the cells hold is of the next one
  void clear() { m_fresh = true; }
  // After the trace of a frame that read the cache: the resolve. Nothing on a frame not given the table.
  void cmdResolve(VkCommandBuffer cmd, nvvk::ProfilerGpuTimer* profiler);

private:
  void freeTable(Resources& resources);

  bool  m_on   = false;
  float m_cell = 0.04F;
  bool  m_still = false;  // ptRadianceCacheStill (debug)

  nvvk::Buffer m_table;
  bool         m_fresh   = false;  // to be cleared before the next frame that is given it
  bool         m_served  = false;  // this frame was given it (prepare)
  bool         m_changed = false;  // the scene changed since the last resolve
  int          m_frame   = 0;
  uint64_t     m_sceneChanges = 0;  // Resources::sceneChanges at the last traced frame

  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  VkPipelineLayout            m_pipelineLayout{};
  VkPipeline                  m_pipeline{};
};
