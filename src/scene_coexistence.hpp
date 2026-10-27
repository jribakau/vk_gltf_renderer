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
// How the retained scene stands beside the legacy one until the host speaks handles (docs:
// daz-viewer docs/retained-scene-plan.md, "Coexistence"): which backend a scene replace takes, and
// the one-way bridge that carries the host's edits to the legacy CPU scene into the retained scene
// imported from it. GPU-free.
//

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <tinygltf/tiny_gltf.h>

#include "scene_import_gltf.hpp"

namespace nvvkgltf {
class Scene;
}

namespace rs {

enum class SceneBackend
{
  eLegacy,
  eRetained,
};

// The sceneBackend setting
enum class SceneBackendSetting
{
  eLegacy,
  eRetained,
  eAuto,
};

// The setting's spelling: "auto", "legacy", "retained"; none for anything else
std::optional<SceneBackendSetting> parseSceneBackendSetting(std::string_view text);
const char*                        sceneBackendSettingName(SceneBackendSetting setting);

struct SceneBackendChoice
{
  SceneBackend backend = SceneBackend::eLegacy;
  std::string  reason;  // for the log line
};

// What the retained scene can't draw yet in `model`, once each, naming the top-level node (a
// composed scene's prop) it was found under: skins, morph targets, animations (P3b), opacity
// micromaps (P3a2), KHR_interactivity, EXT_lights_ies, and glTF 2.1 external assets (`externalAssets`:
// the legacy scene resolved some)
std::vector<std::string> retainedCannotTake(const tinygltf::Model& model, bool externalAssets);

// The backend a scene replace takes. Windowed sessions stay legacy; `auto` takes retained unless
// there is something it can't draw; `retained` takes it regardless (a test setting), saying what
// will be missing.
SceneBackendChoice chooseSceneBackend(SceneBackendSetting setting, bool streamHost, const std::vector<std::string>& cannotTake);

// The host's edits to the legacy CPU scene, carried into the retained scene a document-order import
// made of it, once a frame before the legacy dirty flags are cleared: node transforms and
// visibility, materials, lights. What it can't carry (a re-parse, a texture or sampler change, a
// material the import never saw) asks the caller to import the legacy model again.
class LegacyDirtyBridge
{
public:
  enum class Result
  {
    eNothing,
    eCarried,
    eReimport,
  };

  LegacyDirtyBridge() = default;
  explicit LegacyDirtyBridge(ImportResult imported)
      : m_map(std::move(imported))
  {
  }

  Result              carry(const nvvkgltf::Scene& legacy, RetainedScene& retained) const;
  const ImportResult& map() const { return m_map; }

private:
  ImportResult m_map;
};

// The legacy-accessor debug check (developer.md, "The legacy scene and `SceneGpuView`"): whether a
// retained session may read the legacy CPU scene from `function` in `file` (a std::source_location's,
// any compiler's spelling). Allowed: the scene replace, the host's edits and addressing that the
// bridge carries across, and the scene's sky and sun lights. Anything else reads a scene the retained
// one doesn't keep in step, or a legacy GPU path it bypasses.
bool legacySceneUseAllowed(std::string_view file, std::string_view function);

// The check's memory: true the first time a call site outside the allow-list is seen, so it logs
// once. Any thread.
class LegacyUseCheck
{
public:
  bool firstUnlisted(std::string_view file, uint32_t line, std::string_view function);

private:
  std::mutex                   m_mutex;
  std::unordered_set<uint64_t> m_seen;  // call sites decided, either way
};

}  // namespace rs
