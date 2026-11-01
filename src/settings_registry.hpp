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

/*
 * One declaration per user-settable value.
 *
 * A setting has to reach the command line, the app's parameter messages, the optional MCP endpoint
 * (docs/mcp.md), and the ImGui.ini that remembers it between runs. The first three read
 * nvutils::ParameterRegistry; the last reads nvgui::SettingsHandler.
 *
 * Those used to be two hand-maintained lists, registered at two different points in the lifecycle
 * -- parameters in the constructor, settings in onAttach -- so no single place showed both and they
 * drifted badly (the tonemapper persisted its method but not its exposure; ptTechnique persisted
 * but ptUseSER did not). Declaring through this registry makes that impossible: one call feeds all
 * of them, and Persist is a decision you have to state rather than forget.
 *
 * Because the settings handler only exists later, persisted declarations are recorded here and
 * replayed by applyPersistence() once it does.
 *
 * ImGui.ini restore writes directly into the storage pointer through nvgui::SettingsHandler and
 * does NOT fire the parameter's callbackSuccess -- unlike the other paths, which do.
 * Most callbacks are "one-shot" side effects that should NOT re-run on
 * restore (e.g. --ptSamples disables adaptive sampling, but restoring ptSamples from a session
 * where the user had adaptive sampling on must not clobber the equally-restored ptAdaptiveSampling).
 * For the few cases where a persisted value drives derived state that must be recomputed after
 * restore (e.g. sunAzimuth/Elevation drive Resources::sunDirection), callers explicitly opt in
 * with addPostRestoreHook() and we replay those hooks in runPostRestoreHooks() once ImGui has
 * finished loading the ini (see GltfRenderer::onUIRender).
 */

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <nvgui/settings_handler.hpp>
#include <nvutils/parameter_registry.hpp>

// Whether a value is remembered in ImGui.ini between runs.
//
// eNo is for anything consumed once at start-up (window size, device index, headless frame count)
// and for actions -- remembering "load this file" or "take a screenshot" would replay it on the
// next launch.
enum class Persist
{
  eNo,
  eYes
};

class SettingsRegistry
{
public:
  explicit SettingsRegistry(nvutils::ParameterRegistry* registry)
      : m_registry(registry)
  {
  }

  // Scalars, paths, and enums-as-int. Trailing arguments (min/max) forward to the registry.
  template <class T, class... Rest>
  void add(const nvutils::ParameterBase::Info& info, T* storage, Persist persist, Rest&&... rest)
  {
    m_registry->add(info, storage, std::forward<Rest>(rest)...);
    if(persist == Persist::eYes)
      remember(info.name, storage);
  }

  // glm vectors (colors, sizes).
  template <class GLMVEC>
  void addVector(const nvutils::ParameterBase::Info& info, GLMVEC* storage, Persist persist)
  {
    m_registry->addVector(info, storage);
    if(persist == Persist::eYes)
      remember(info.name, storage);
  }

  // Replay the persisted declarations. Called once the handler exists (onAttach); declarations
  // happen earlier, in the constructor.
  void applyPersistence(nvgui::SettingsHandler& handler) const
  {
    for(size_t i = 0; i < m_persisted.size(); i++)
      m_persisted[i](handler);
  }

  // Register a hook that should run once after ImGui has finished loading the ini. Use this for
  // derived state that a persisted value drives but that ini restore does not itself refresh
  // (e.g. recompute Resources::sunDirection from the restored sunAzimuth/Elevation). Prefer
  // this over reusing ParameterBase::callbackSuccess, whose semantics are CLI/UI edits, not
  // full-session restore.
  //
  // CLI precedence contract: hooks must be *derived* and *idempotent*. They read persisted
  // storage after nvgui::SettingsHandler's loadFilter has already skipped keys marked
  // ParameterParser::wasParsed() (see GltfRenderer::onAttach), so the values a hook sees are
  // already "CLI wins per key, ini fills the rest". Never write back to a persisted storage
  // pointer from a hook -- that would silently overwrite a CLI-set value.
  void addPostRestoreHook(std::function<void()> hook) { m_postRestoreHooks.push_back(std::move(hook)); }

  // Fire all registered post-restore hooks. Call once after ImGui has loaded the ini
  // (i.e. after Application::run()'s ini reload, e.g. on the first frame).
  void runPostRestoreHooks() const
  {
    for(size_t i = 0; i < m_postRestoreHooks.size(); i++)
      m_postRestoreHooks[i]();
  }

private:
  template <class T>
  void remember(const std::string& name, T* storage)
  {
    m_persisted.push_back([name, storage](nvgui::SettingsHandler& handler) { handler.setSetting(name, storage); });
  }

  nvutils::ParameterRegistry*                               m_registry{};
  std::vector<std::function<void(nvgui::SettingsHandler&)>> m_persisted;
  std::vector<std::function<void()>>                        m_postRestoreHooks;
};
