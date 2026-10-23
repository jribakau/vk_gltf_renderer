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
// The Figure Viewer's materials, beside the figure (fv_host.hpp): the surfaces DAZ describes, the
// Material panel's edits over them, and the glTF materials the renderer shows (figure/materials).
//
// An edit is lowered again (mat::lower) and written over the scene's glTF material. Edits that
// change a texture, not only a factor (a bump strength baked into the normal map, a colour baked
// under makeup, the skin switching between Real-time and RTX), ask the texture server for the new
// image on a worker thread and import it when it arrives; the material is written then.
//
// Commands (application thread):
//   material.set    {material: "m:3" | ["m:3", ...], values: {color, roughness, metalness, opacity,
//                    alphaMode, doubleSided, normalStrength, bumpStrength, specular, clearcoat,
//                    clearcoatRoughness, emissive, emissiveIntensity}}   null resets a value
//   material.reset  {material}                    drops the edits of a material (or of all, if none)
//   Colours are "#rrggbb" in sRGB, as the colour pickers hold them. alphaMode: opaque, cutout, blend.
//
// Topic "materials": {figure, items: {"m:3": {name, kind, shader, meshes, textures, values, edited}}}
// as a snapshot when a figure loads, then merge patches; `values` are what the renderer was given
// (edits included), `edited` lists the values that are edits, `meshes` the scene meshes using it.
//

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <figure/gltf/gltf_build.hpp>
#include <figure/materials/lower.hpp>
#include <figure/scene/graph.hpp>

#include "fv_events.hpp"

class GltfRenderer;

class FvMaterials
{
public:
  struct Hooks
  {
    GltfRenderer* renderer = nullptr;
    fv::EventHub* events   = nullptr;
    // runs a function on the application thread, later
    std::function<void(std::function<void()>)> post;
    std::function<void(const std::string&, const char*)> notice;
    // a "jobs" entry for a bake in flight: a new id, and its state
    std::function<uint64_t()>                      nextJob;
    std::function<void(uint64_t, const fv::Json&)> job;
    // a texture for a request, cooked if need be: blocks, so only worker threads call it. `flipped`
    // says whether the figure's UVs expect rows bottom-up (the build's texturesFlipped).
    std::function<std::optional<fv::build::TextureImage>(const fv::build::TextureRequest&, bool flipped)> texture;
  };

  explicit FvMaterials(Hooks hooks);
  ~FvMaterials();

  // A figure was loaded: `scene` is its scene graph (which outlives the attachment), `built` what
  // the build made of its materials. Publishes the topic. Application thread.
  void attach(const std::string& figureId, fv::SceneGraph* scene, const fv::build::BuildResult& built, bool texturesFlipped,
              fv::mat::SkinMode skin);
  void detach();

  fv::Json set(const fv::Json& args);
  fv::Json reset(const fv::Json& args);

  // Surfaces that were parsed again (a material preset): lowers them and publishes.
  void surfacesChanged(const std::set<int>& materials);
  // Real-time and RTX shade skin differently; lowers the materials that differ.
  void setSkinMode(fv::mat::SkinMode skin);
  fv::mat::SkinMode skinMode() const { return m_skin; }

private:
  void refresh(const std::set<int>& materials);
  void write(int index, const fv::mat::Lowered& lowered);
  void bake(std::vector<fv::build::TextureRequest> requests);
  void baked(std::vector<std::pair<fv::build::TextureRequest, std::optional<fv::build::TextureImage>>> results, uint64_t generation);
  void publish(const std::set<int>& materials, bool all);
  std::string stateOf(const fv::mat::Lowered& lowered) const;
  fv::Json item(int index) const;
  std::vector<int> targets(const fv::Json& args) const;

  Hooks m_hooks;

  // application thread
  std::string                                         m_figure;
  fv::SceneGraph*                                     m_scene = nullptr;
  std::vector<fv::mat::LowerOptions>                  m_options;
  std::map<std::string, fv::mat::ResolvedTexture>     m_textures;  // request key -> glTF texture
  std::set<std::string>                               m_failed;    // keys a cook could not make
  std::set<std::string>                               m_baking;    // keys in flight
  std::set<int>                                       m_waiting;   // materials that need them
  std::vector<std::string>                            m_applied;   // per material, the glTF last written (as JSON text)
  std::vector<std::vector<int>>                       m_meshesOf;  // per material, the scene meshes using it
  std::vector<bool>                                   m_forceDouble;  // strand ribbons are seen from both sides
  bool                                                m_flipped = false;
  fv::mat::SkinMode                                   m_skin    = fv::mat::SkinMode::eRaster;
  uint64_t                                            m_generation = 0;

  std::shared_ptr<std::atomic<bool>> m_alive;  // false once destroyed, for work still queued
  std::mutex               m_threadsMutex;
  std::vector<std::thread> m_threads;
};
