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
// An edit is lowered again (mat::lower) and the host writes it into the retained scene: a figure's
// through its feed, with the options this lowered it with; a model's as its file's material with the
// edits laid over. Edits that change a texture, not only a factor (a bump strength baked into the
// normal map, a colour baked under makeup, the skin switching between Real-time and RTX), ask the
// texture server for the new image on a worker thread; the host keeps it with the figure
// (`remember`) and the material is written once it arrives.
//
// Commands (application thread):
//   material.set    {material: "m:3" | ["m:3", ...], values: {color, roughness, metalness, opacity,
//                    alphaMode, doubleSided, normalStrength, bumpStrength, specular, clearcoat,
//                    clearcoatRoughness, emissive, emissiveIntensity, irisColor, irisStrength,
//                    scleraColor, recolor, recolorStrength, tone, toneStrength, makeupColor,
//                    makeupWeight, scatterScale, baseMap, opacityMap, normalMap, bumpMap,
//                    roughnessMap, uvScaleU, uvScaleV, uvOffsetU, uvOffsetV, uvRotation, fabric,
//                    fabricScale, fabricStrength, sheenColor, sheenRoughness}}   null resets a value
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
//   The Clothes tab's keys (figure/materials/edits_json.hpp holds the ranges): baseMap ... roughnessMap
//   put another library-relative image ("/Runtime/Textures/...") in a slot, or "" none; a disk path is
//   refused (the app copies a picked file into the library first). uvScale*, uvOffset* and uvRotation
//   (degrees) tile, move and turn every map on the surface's UVs over its authored tiling
//   (mat::surfaceUv), KHR_texture_transform. fabric (fabric.hpp's keys, "" none) lays a built-in
//   weave over the surface as its detail layer, fabricScale its size and fabricStrength its normal's
//   strength. These are refused on a model. sheenColor and sheenRoughness (0..1) are
//   KHR_materials_sheen, on figures and models alike. A map chosen again that could not be made
//   before is tried again.
//
// One per prop with materials (a figure, a model); "m:3" is the prop's own third material.
//
// Topic "materials/<prop>": {figure, items: {"m:3": {name, kind, shader, meshes, textures, values,
// edited, eye?, makeup?}}, eyes?, skin?} as a snapshot when the prop is attached, then merge patches; `values` are
// what the renderer was given (edits included), `edited` lists the values that are edits, `meshes`
// the prop's meshes using it. `colorBaked`: its colour is in a bake, so the panel applies it on release.
// An eye's surface has `eye: {iris, sclera, baked}` (the colours it
// takes, and whether changing them cooks its map) and always has irisColor/irisStrength or
// scleraColor in `values`, a colour null when unset. A surface with a makeup layer has
// `makeup: {mapped}` (whether a map shapes it) and makeupColor (null when unset) and makeupWeight
// in `values`; every surface has recolor and tone (null when unset) with their strengths; flesh
// under measured skin (SkinMode::eMeasured) has scatterScale. normalStrength and bumpStrength are
// null when the surface has no such map. A figure surface has every Clothes key (mat::clothesValues:
// the maps, fabric and sheenColor null when unset, the numbers their defaults), and a model's has
// sheenColor and sheenRoughness. `textures` lists the maps: a figure surface's as lowering uses them
// (mat::mapSlots), {slot, label, file, rel, edited}, rel "" for one an edit took away; a model's
// {slot, label, file, edited: false}, with no rel (its images are its file's). A figure's
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
#include <tinygltf/tiny_gltf.h>

#include "fv_events.hpp"
#include "fv_workers.hpp"

class HostRenderer;

class FvMaterials
{
public:
  struct Hooks
  {
    HostRenderer* renderer = nullptr;
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
    // a texture a bake made: kept with the figure, where its feed finds it, so the scene fed again
    // after a wardrobe change has it too (application thread)
    std::function<void(const fv::build::TextureRequest&, const fv::build::TextureImage&)> remember;
    // whether the figure has the image for a request (its feed would find one)
    std::function<bool(const fv::build::TextureRequest&)> resolves;
    // the figure's materials lowered again with these options: written into the scene (application thread)
    std::function<void(std::vector<std::pair<int, fv::mat::LowerOptions>>)> writeFigure;
    // a model's material `index`, its file's with the edits laid over: written into the scene
    std::function<void(int index, fv::mat::Material material)> writeModel;
  };

  explicit FvMaterials(Hooks hooks);
  ~FvMaterials();

  // A figure was fed into the scene: `scene` is its scene graph (which outlives the attachment),
  // `lowering` the options its feed lowered each material with. Publishes the topic. Application
  // thread.
  void attach(const std::string&                 figureId,
              fv::SceneGraph*                    scene,
              std::vector<fv::mat::LowerOptions> lowering,
              bool                               texturesFlipped,
              fv::mat::SkinMode                  skin,
              const std::set<int>&               kept = {});
  // The same figure fed again after a wear, a take-off or a graft (its scene graph only grows): the
  // materials it appended are attached, and the edits, the bakes in flight and what could not be made
  // stay. `lowering` as attach takes it; `kept` (FigureFeed::takeKept), the materials its feed left as
  // they were for want of a map: asked for here, and written when they come.
  void extend(std::vector<fv::mat::LowerOptions> lowering, const std::set<int>& kept = {});
  // A model read from a glTF or OBJ file, `source` as the file has it: its materials as they are,
  // with the same edits laid over them directly (there is no Surface to lower), named as the file
  // names them or "Material-<i>" from their place in it, so a scene file's edits find them whatever
  // else is in the scene. Attached again (the scene was composed again), it keeps its edits and
  // writes them.
  void attachModel(const std::string& id, const tinygltf::Model& source);
  // The topic goes too.
  void detach();
  // The edits of a model's material `index`, to lay over its file's own (fv::model::editedMaterial)
  // for an export; null for a figure or a material it does not hold.
  const fv::mat::Edits* modelEdits(int index) const;

  fv::Json set(const fv::Json& args);
  fv::Json reset(const fv::Json& args);

  // The edits of every material that has any, by "<name>" or "<name>#n" (the n-th material of that
  // name), as material.set values: for a scene file.
  fv::Json editState() const;
  // Puts such edits back; returns what could not be: materials that are not there any more.
  std::vector<std::string> restoreEdits(const fv::Json& state);
  // One material's edits as material.set values (empty when it has none), its name, and how many
  // materials there are
  fv::Json    editsOf(int index) const;
  std::string nameOf(int index) const;
  size_t      count() const { return m_count; }

  // Surfaces that were parsed again (a material preset): lowers them and publishes.
  void surfacesChanged(const std::set<int>& materials);

  // While held, a change is only noted: a composition that follows builds the surfaces as they are
  // and attach() takes them from there, so lowering, writing and baking each change into the scene
  // that is about to go would be work thrown away (a restore's presets, skin, eyes and edits).
  // release() ends it; `refresh` lowers what was noted, for when no composition came.
  void hold();
  void release(bool refresh);
  // The figure's eyes for the topic (fv_host_looks.cpp publishEyes); kept across attachments, dropped on detach.
  void setEyes(fv::Json eyes);
  // The same for its skin and makeup (fv_host_looks.cpp publishSkin).
  void setSkin(fv::Json skin);
  // Real-time and RTX shade skin differently; lowers the materials that differ.
  void setSkinMode(fv::mat::SkinMode skin);
  fv::mat::SkinMode skinMode() const { return m_skin; }
  // A texture they sample changed under them (a cook that landed): every attached material written
  // again with its options and edits, the feed setting only those that differ. Not for a model.
  void rewrite();

private:
  void refresh(const std::set<int>& materials);
  // per material, the scene meshes using it
  void     mapMeshes();
  void     write(const std::set<int>& materials);
  void writeModel(int index);
  fv::Json modelItem(int index) const;
  // A texture being made, and whether any material still wants it
  struct BakeItem
  {
    fv::build::TextureRequest          request;
    std::shared_ptr<std::atomic<bool>> unwanted;
  };
  struct BakeResult
  {
    fv::build::TextureRequest              request;
    std::optional<fv::build::TextureImage> image;
    bool                                   skipped = false;  // nobody wanted it by its turn: not made
  };
  // Asks for `requests`. One bake runs at a time per figure, since a cook can't be stopped: what's asked
  // meanwhile waits, and what a material has moved on from by its turn is dropped unmade, so a drag
  // makes the bake it started with and the one it ended on, not one per frame.
  void bake(std::vector<fv::build::TextureRequest> requests);
  void startBake();
  // marks what no material wants now, and what it wants again
  void markUnwanted();
  void forgetBakes();
  void baked(std::vector<BakeResult> results, uint64_t generation);
  void publish(const std::set<int>& materials, bool all);
  std::string stateOf(const fv::mat::Lowered& lowered) const;
  fv::Json item(int index) const;
  std::vector<int> targets(const fv::Json& args) const;

  Hooks m_hooks;

  // application thread
  std::string                                         m_figure;
  fv::SceneGraph*                                     m_scene = nullptr;
  // the materials attached: a wear appends to the scene graph before the scene is built again, and
  // what it appended has no lowering here until the next attach
  size_t                                              m_count = 0;
  // an imported model: its materials stand in a scene graph of their own (names and edits), and the
  // glTF materials the edits are laid over are kept as they loaded
  bool                                                m_model = false;
  fv::SceneGraph                                      m_modelScene;
  std::vector<fv::mat::Material>                      m_originals;
  std::vector<std::string>                            m_modelImages;  // per texture of its file, the image's name
  std::vector<fv::mat::LowerOptions>                  m_options;
  std::set<std::string>                               m_failed;    // keys a cook could not make
  std::map<std::string, std::shared_ptr<std::atomic<bool>>> m_baking;  // keys queued or in flight, each's BakeItem::unwanted
  std::vector<BakeItem>                               m_bakeQueue;
  bool                                                m_bakeRunning = false;
  std::set<int>                                       m_waiting;   // materials that need them
  std::map<int, std::set<std::string>>                m_wants;     // each of those, the keys it needs
  bool                                                m_held = false;
  std::set<int>                                       m_heldChanges;  // noted while held
  std::vector<std::string>                            m_applied;   // per material, the glTF last written (as JSON text)
  std::vector<std::vector<int>>                       m_meshesOf;  // per material, the scene meshes using it
  bool                                                m_flipped = false;
  fv::mat::SkinMode                                   m_skin    = fv::mat::SkinMode::eRaster;
  uint64_t                                            m_generation = 0;
  fv::Json                                            m_eyes;       // the topic's `eyes`, null for none
  fv::Json                                            m_skinState;  // the topic's `skin`, null for none

  std::shared_ptr<std::atomic<bool>> m_alive;  // false once destroyed, for work still queued
  FvWorkers m_threads;  // texture bakes
};
