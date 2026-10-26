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
//                    clearcoatRoughness, emissive, emissiveIntensity, irisColor, irisStrength,
//                    scleraColor, recolor, recolorStrength, tone, toneStrength, makeupColor,
//                    makeupWeight, scatterScale}}   null resets a value
//   material.reset  {material}                    drops the edits of a material (or of all, if none)
//   Colours are "#rrggbb" in sRGB, as the colour pickers hold them. alphaMode: opaque, cutout, blend.
//   irisColor and irisStrength (0..1) recolour an eye's iris in its map (the cooker's ir=), and are
//   refused on a surface that isn't one, or a Genesis 8 iris whose UVs gave no discs; scleraColor
//   multiplies the white: baked into a Genesis 9 eye's map, a factor on a Genesis 8 "Sclera".
//   recolor and recolorStrength move a surface's mean colour, its detail kept (a hair's; the
//   cooker's rc=). tone and toneStrength do it for skin against the figure's tone reference
//   (dson::toneReference), so every skin surface moves by the same shift. Both are refused on a
//   model. makeupColor and makeupWeight (0..1) change a PBRSkin makeup layer, refused on a surface
//   without one. scatterScale (0.1..4) scales how far light spreads under measured skin.
//
// One per prop with materials (a figure, a model). Its materials sit in the renderer's scene at
// `Placement::materials` on (fv_compose.hpp), and "m:3" is the prop's own third material.
//
// Topic "materials/<prop>": {figure, items: {"m:3": {name, kind, shader, meshes, textures, values,
// edited, eye?, makeup?}}, eyes?, skin?} as a snapshot when the prop is attached, then merge patches; `values` are
// what the renderer was given (edits included), `edited` lists the values that are edits, `meshes`
// the prop's meshes using it. An eye's surface has `eye: {iris, sclera, baked}` (the colours it
// takes, and whether changing them cooks its map) and always has irisColor/irisStrength or
// scleraColor in `values`, a colour null when unset. A surface with a makeup layer has
// `makeup: {mapped}` (whether a map shapes it) and makeupColor (null when unset) and makeupWeight
// in `values`; every surface has recolor and tone (null when unset) with their strengths; flesh
// under measured skin (SkinMode::eMeasured) has scatterScale. A figure's
// `eyes` is what the host set (setEyes, fv_host.hpp material.eyes): {rel, preset, own, ready,
// surfaces}; its `skin` the same for skin and makeup (setSkin, fv_host.hpp material.skin). The
// topic is dropped on detach.
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
#include "fv_workers.hpp"

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
    // a texture a bake made and the renderer took: kept with the figure, so the scene built again
    // after a wardrobe change has it too (application thread)
    std::function<void(const fv::build::TextureRequest&, const fv::build::TextureImage&)> remember;
  };

  // Where a prop's own arrays start in the renderer's scene.
  struct Placement
  {
    int materials = 0, textures = 0, samplers = 0, meshes = 0;
  };

  explicit FvMaterials(Hooks hooks);
  ~FvMaterials();

  // A figure was built into the scene: `scene` is its scene graph (which outlives the attachment),
  // `built` what the build made of its materials. Publishes the topic. Application thread.
  void attach(const std::string& figureId, fv::SceneGraph* scene, const fv::build::BuildResult& built, bool texturesFlipped,
              fv::mat::SkinMode skin, Placement at);
  // A model read from a glTF or OBJ file, `count` materials and `meshes` meshes of it at `at`: its
  // materials as they are, with the same edits laid over them directly (there is no Surface to
  // lower). Attached again (the scene was composed again), it keeps its edits and writes them.
  void attachModel(const std::string& id, Placement at, int count, int meshes);
  // The topic goes too.
  void detach();

  fv::Json set(const fv::Json& args);
  fv::Json reset(const fv::Json& args);

  // The edits of every material that has any, by "<name>" or "<name>#n" (the n-th material of that
  // name), as material.set values: for a scene file.
  fv::Json editState() const;
  // Puts such edits back; returns what could not be: materials that are not there any more.
  std::vector<std::string> restoreEdits(const fv::Json& state);

  // Surfaces that were parsed again (a material preset): lowers them and publishes.
  void surfacesChanged(const std::set<int>& materials);
  // The figure's eyes for the topic (fv_host.cpp eyesJson); kept across attachments, dropped on detach.
  void setEyes(fv::Json eyes);
  // The same for its skin and makeup (fv_host.cpp publishSkin).
  void setSkin(fv::Json skin);
  // Real-time and RTX shade skin differently; lowers the materials that differ.
  void setSkinMode(fv::mat::SkinMode skin);
  fv::mat::SkinMode skinMode() const { return m_skin; }

private:
  void refresh(const std::set<int>& materials);
  void write(int index, const fv::mat::Lowered& lowered);
  void writeModel(int index);
  fv::Json modelItem(int index) const;
  void bake(std::vector<fv::build::TextureRequest> requests);
  void baked(std::vector<std::pair<fv::build::TextureRequest, std::optional<fv::build::TextureImage>>> results, uint64_t generation);
  void publish(const std::set<int>& materials, bool all);
  std::string stateOf(const fv::mat::Lowered& lowered) const;
  fv::Json item(int index) const;
  std::vector<int> targets(const fv::Json& args) const;

  Hooks m_hooks;

  // application thread
  std::string                                         m_figure;
  Placement                                           m_at;
  fv::SceneGraph*                                     m_scene = nullptr;
  // the materials attached: a wear appends to the scene graph before the scene is built again, and
  // what it appended has no lowering here until the next attach
  size_t                                              m_count = 0;
  // an imported model: its materials stand in a scene graph of their own (names and edits), and the
  // glTF materials the edits are laid over are kept as they loaded
  bool                                                m_model = false;
  fv::SceneGraph                                      m_modelScene;
  std::vector<tinygltf::Material>                     m_originals;
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
  fv::Json                                            m_eyes;       // the topic's `eyes`, null for none
  fv::Json                                            m_skinState;  // the topic's `skin`, null for none

  std::shared_ptr<std::atomic<bool>> m_alive;  // false once destroyed, for work still queued
  FvWorkers m_threads;  // texture bakes
};
