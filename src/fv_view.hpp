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

// The native viewport's camera, display and selection, beside the figure (fv_host.hpp).
//
// Commands (application thread; the host runs them there):
//   select        {node} or {node: null}     a scene-graph node of the loaded figure
//   camera.view   {name}                     front, back, left, right, top, bottom, threeQuarter
//   camera.frame  {selected?}                the whole figure, or the selection
//   camera.set    {focalLength, dof, fStop, focusMode, focusDistance}
//   camera.get    {}                         -> {position, target, focalLength}
//   camera.goto   {position, target, focalLength}
//   view.set      {display, grid, gizmo, pick}  display: shaded, clay, wireframe, normals;
//                                               pick: "mesh" (the default) or "bone"
//
// Topics: "scene" (hierarchy and stats, published when a figure loads), "selection" (with `bone`,
// the pose bone id, when a posable bone is selected), "camera" ({position, target, up,
// focalLength}, at most every 33 ms while it moves).
// A click that didn't drag is a pick. In bone mode it selects the bone nearest the hit and shows
// the transform gizmo on it; pointer events go to the gizmo first.

#include <chrono>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

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

  // The figure's posable bones: scene node -> pose bone id, and the bone node nearest a world
  // point (-1 for none).
  struct Bones
  {
    std::unordered_map<int, std::string>  idOf;
    std::function<int(const glm::vec3&)> nearest;
  };

  // Publishes the scene topic and remembers how glTF nodes map back to the figure. `morphs` is the
  // dial count the panel shows.
  void setFigure(const std::string& id, const std::string& name, const fv::SceneGraph& scene, int root,
                 const std::vector<int>& nodeOf, int morphs, Bones posable);
  // The same for a model the renderer loaded from a glTF or OBJ file: node i of the topic is glTF
  // node i, there are no bones, and the stats come from the model.
  void setModel(const std::string& id, const std::string& name, const tinygltf::Model& model);
  void clearFigure();

  // Pointer events, in device pixels. The gizmo takes its own (a drag on a handle); a left click
  // that barely moved picks. Returns the events the gizmo didn't take, for the camera.
  std::vector<fv::Json> onInput(const std::vector<fv::Json>& events);
  // Publishes selection and camera changes and keeps the lens matched to the viewport's aspect.
  void onFrame();

private:
  void applyLens();
  void applyDof();
  void publishSelection(bool force);
  void publishCamera();

  GltfRenderer*                                      m_renderer = nullptr;
  fv::EventHub*                                      m_events   = nullptr;
  std::function<fv::Json(std::function<fv::Json()>)> m_onApp;

  std::string                        m_id;
  std::vector<int>                   m_gltfOf;
  std::unordered_map<int, int>       m_figureOf;
  struct NodeRec
  {
    std::string name, kind;
    int         mesh = -1;
  };
  std::vector<NodeRec> m_nodes;
  Bones                m_bones;
  int                  m_lastGltf = -2;
  bool                 m_pickBones = false;

  fv::Json                              m_camera;
  std::chrono::steady_clock::time_point m_cameraAt{};

  double      m_focal      = 40;
  double      m_aspect     = -1;
  bool        m_dof        = false;
  double      m_fStop      = 2.8;
  std::string m_focus      = "target";
  double      m_focusDistance = 3;

  bool   m_click = false;
  bool   m_lmb   = false;
  double m_x0 = 0, m_y0 = 0;
};
