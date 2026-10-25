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

// The native viewport's camera, display and selection, beside the scene of props (fv_host.hpp).
//
// Commands (application thread; the host runs them there):
//   select        {prop, node?} or {prop: null}   a prop (its root), or a node of its "nodes/<prop>" topic
//   camera.view   {name}                     front, back, left, right, top, bottom, threeQuarter
//   camera.frame  {prop?, selected?}         the whole scene, a prop, or the selection
//   camera.set    {focalLength, dof, fStop, focusMode, focusDistance}   the lens of the view in use
//   camera.get    {}                         -> {position, target, up, focalLength}
//   camera.goto   {position, target, up?, focalLength?}
//   view.set      {display, grid, gizmo, pick, transform, helpers}  display: shaded, clay, wireframe, normals;
//                                               pick: "object" (the default: a click selects the prop),
//                                               "mesh" (the node hit) or "bone" (the posed bone nearest);
//                                               transform: the gizmo's "move", "rotate" or "scale";
//                                               helpers: whether camera and light markers show
//
// Topics: "selection" ({prop, node, name, mesh, kind, bone, surface}: `node` is null when the prop itself
// is selected; `bone`, the pose bone id, when a posable bone is; `surface`, after a click on the prop,
// the surface it hit as its "materials/<prop>" topic names it, "m:<i>"), "camera" ({position, target, up,
// focalLength, view, free?}, at most every 33 ms while it moves: `view` is the camera prop looked
// through or null, and `free` where the free view was left meanwhile). The host publishes "scene" and "nodes/<prop>"
// with what describeFigure and describeModel make.
// A click that didn't drag is a pick. Pointer events go to the gizmo first.

#include <chrono>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "fv_events.hpp"

class GltfRenderer;
namespace tinygltf {
class Model;
}

namespace fv {
class SceneGraph;
}

class FvView
{
public:
  // `onApp` runs a function on the application thread and returns what it returned.
  FvView(GltfRenderer* renderer, fv::EventHub& events, fv::CommandRegistry& commands, std::function<fv::Json(std::function<fv::Json()>)> onApp);

  // A figure's posable bones: topic node -> pose bone id, and the bone node nearest a world point
  // (-1 for none).
  struct Bones
  {
    std::unordered_map<int, std::string>  idOf;
    std::function<int(const glm::vec3&)> nearest;
  };

  // A prop as the viewport sees it: its root and its nodes in the renderer's scene.
  struct NodeRec
  {
    std::string name, kind;
    int         mesh = -1;
  };
  struct PropNodes
  {
    std::string          id;
    int                  root = -1;  // glTF node of the prop root
    std::vector<int>     gltfOf;     // topic node -> glTF node
    std::vector<NodeRec> nodes;      // per topic node
    Bones                bones;
    // its glTF materials, [materials, materials + materialCount): "m:<i>" of its materials topic
    int materials = 0, materialCount = 0;
  };

  // A figure's "nodes/<prop>" topic ({nodes: {"<i>": {name, parent, kind, order, depth}}, stats})
  // from its scene graph, `nodeOf` its scene nodes in the renderer's scene; fills `out`.
  static fv::Json describeFigure(PropNodes& out, const fv::SceneGraph& scene, int root, const std::vector<int>& nodeOf, int morphs);
  // The same for a model read from a file, nodes [firstNode, lastNode) of the composed `model`
  // under `root`: topic node i is glTF node firstNode + i, and a node's mesh is the model's own
  // (the composed one less `meshBase`), as the materials topic counts them.
  static fv::Json describeModel(PropNodes& out, const tinygltf::Model& model, int root, int firstNode, int lastNode, int meshBase);

  // The props now in the scene, after a composition; republishes the selection.
  void setProps(std::vector<PropNodes> props);
  // The prop the selection is in, or "".
  const std::string& selectedProp() const { return m_selectedProp; }
  // the selection, if it is in `prop`, cleared and published (a prop leaving the scene)
  void deselect(const std::string& prop);
  // Whether a glTF node is a prop root, and which prop's it is.
  const PropNodes* propOfRoot(int gltf) const;

  // The pick mode and the gizmo's handles in it, for the host's gizmo hooks.
  const std::string& pickMode() const { return m_pick; }
  const std::string& transformMode() const { return m_transform; }
  // Whether the host draws its camera and light markers, and which prop's marker is at a pixel
  // ("" for none), asked before a click's ray pick.
  bool helpersShown() const { return m_helpers; }
  void onHelperPick(std::function<std::string(const glm::vec2& px)> f) { m_helperPick = std::move(f); }

  // The lens of the view in use: the free view's, or the camera prop's it looks through.
  struct Lens
  {
    double      focalLength   = 40;
    bool        dof           = false;
    double      fStop         = 2.8;
    std::string focusMode     = "target";
    double      focusDistance = 3;
  };
  const Lens& lens() const { return m_lens; }
  void        setLens(const Lens& lens);
  // camera.set changed the lens: the host keeps it with the view in use.
  void onLensChange(std::function<void(const Lens&)> f) { m_onLens = std::move(f); }
  // More for the "camera" topic: the camera prop looked through ("view") and where the free view
  // was left ("free"), merged into the next snapshot.
  void setCameraExtra(fv::Json extra);

  // Pointer events, in device pixels. The gizmo takes its own (a drag on a handle); a left click
  // that barely moved picks. Returns the events the gizmo didn't take, for the camera.
  std::vector<fv::Json> onInput(const std::vector<fv::Json>& events);
  // Publishes selection and camera changes and keeps the lens matched to the viewport's aspect.
  void onFrame();

  static fv::Json lensJson(const Lens& lens);
  static Lens     lensOf(const fv::Json& args, Lens base);

private:
  void applyLens();
  void applyDof();
  void publishSelection(bool force);
  void publishCamera();
  // the prop and topic node of a glTF node (node -1: the prop root), or {-1, -1}
  std::pair<int, int> locate(int gltf) const;
  // A click hit `renderNode` (-1: nothing): the surface it hit, for the next selection published
  void notePick(int renderNode);

  GltfRenderer*                                      m_renderer = nullptr;
  fv::EventHub*                                      m_events   = nullptr;
  std::function<fv::Json(std::function<fv::Json()>)> m_onApp;

  std::vector<PropNodes>                        m_props;
  std::unordered_map<int, std::pair<int, int>> m_where;  // glTF node -> {prop, topic node}
  std::string                                   m_selectedProp;
  int                                           m_lastGltf = -2;
  std::string                                   m_surfaceProp, m_surface;  // what the last click hit
  std::string                                   m_pick      = "object";
  std::string                                   m_transform = "move";
  bool                                          m_helpers   = true;
  std::function<std::string(const glm::vec2&)>  m_helperPick;

  fv::Json                              m_camera;
  fv::Json                              m_cameraExtra = fv::Json::object();
  std::chrono::steady_clock::time_point m_cameraAt{};

  Lens                             m_lens;
  std::function<void(const Lens&)> m_onLens;
  double                           m_aspect = -1;

  bool   m_click = false;
  bool   m_lmb   = false;
  double m_x0 = 0, m_y0 = 0;
};
