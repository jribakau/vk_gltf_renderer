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
//   select        {prop, node?} or {prop: null}   a prop (its root), or a node of its "nodes/<prop>" topic;
//                 {props: [id]}                   several, the first the primary (the gizmo's, the panels');
//                 {prop, add: true} or {prop, toggle: true}   one more, or one in or out (a Ctrl+click
//                                               in the viewport toggles too)
//   camera.view   {name}                     front, back, left, right, top, bottom, threeQuarter
//   camera.frame  {prop?, selected?}         the whole scene, a prop, or the selection
//   camera.set    {focalLength, dof, fStop, focusMode, focusDistance, blades, bladeRotation}   the lens of the view in use
//   camera.get    {}                         -> {position, target, up, focalLength}
//   camera.goto   {position, target, up?, focalLength?}
//   view.set      {display, grid, gizmo, pick, select, transform, space, snap, helpers, frameAspect, solo}  display: shaded, clay, wireframe, normals;
//                                               pick: "object" (the default: a click selects the prop),
//                                               "mesh" (the node hit), "bone" (the posed bone nearest) or
//                                               "pull" (a drag on a figure in Physics mode pulls it, the
//                                               view still orbiting from anywhere else);
//                                               select: whether a click picks at all (true, the default;
//                                               off, the selection stays as it is and a double-click
//                                               still centres the view);
//                                               transform: the gizmo's "move", "rotate" or "scale";
//                                               space: its handles along the world's axes (what the prop
//                                               rides on) or the prop's own ("world", "local");
//                                               snap: {move (m), rotate (degrees), scale}, a drag going in
//                                               whole steps from where it started, or null for none;
//                                               helpers: whether camera and light markers show;
//                                               frameAspect: a frame's width over height the view is
//                                               widened to hold, with a margin (0: none; fovForFrame);
//                                               solo (Isolate): a prop or a list of them that alone show,
//                                               with what rides on them and what carries them, null for
//                                               all; a view of the scene, not an edit: the props' own
//                                               visibility and the history stay. A mesh hidden this way
//                                               long may be let go under memory pressure, as any hidden one
//
// Topics: "selection" ({prop, props, node, name, mesh, kind, bone, surface, state}: `props` every prop
// selected, `prop` first; `node` is null when the prop itself
// is selected; `bone`, the pose bone id, when a posable bone is; `state` for another node the Scene
// tab may change ({visible, t, r, e, s, edited, moves}, metres and degrees, the host's node.set); `surface`, after a click on the prop,
// the surface it hit as its "materials/<prop>" topic names it, "m:<i>"), "camera" ({position, target, up,
// focalLength, view, free?}, at most every 33 ms while it moves: `view` is the camera prop looked
// through or null, and `free` where the free view was left meanwhile). The host publishes "scene" and "nodes/<prop>"
// with what describeFigure and describeModel make.
// A click that didn't drag is a pick. Pointer events go to the gizmo first.

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "fv_events.hpp"
#include "fv_scene_map.hpp"
#include "host_renderer.hpp"  // HostRenderer::PixelReadout


// How far a frame guide leaves the frame from the view's edges, as a share of the frame where it is
// tightest. The app's FRAME_MARGIN (src/lib/frame.ts) is the same number, which it draws the frame with.
constexpr double kFrameMargin = 0.12;

// The vertical field of view, in degrees, of a lens of `mm` whose 35 mm spans the long side of a view
// of `viewAspect` (width over height). With a `frameAspect`, the field is widened so that a frame of that
// shape, its long side at 35 mm, fits in the view with kFrameMargin to spare.
double fovForFrame(double mm, double viewAspect, double frameAspect = 0);
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
  FvView(HostRenderer* renderer, fv::EventHub& events, fv::CommandRegistry& commands, std::function<fv::Json(std::function<fv::Json()>)> onApp);

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
    // What a pick at it is called (the pixel readout, the Time view's list): a figure's mesh by the
    // node that owns it, as a mesh node takes its geometry's name, often just "geometry"
    std::string label;
  };
  // (root, entityOf and its materials are its fv::PropAddress, what fv::SceneMap reads)
  struct PropNodes : fv::PropAddress
  {
    std::string          id;
    std::vector<NodeRec> nodes;  // per topic node
    Bones                bones;
  };

  // A figure's "nodes/<prop>" topic ({nodes: {"<i>": {name, parent, kind, order, depth}}, stats})
  // from its scene graph, `entityOf` its scene nodes' entities; fills `out`. Taken-off items are
  // left out. (fv_view_describe.cpp, as is describeModel.)
  static fv::Json describeFigure(PropNodes& out, const fv::SceneGraph& scene, int root, std::vector<rs::EntityId> entityOf, int morphs);
  // The same for a model read from a file, from the file's own `model`: topic node i is its node i,
  // imported as entityOf[i] (empty: the import failed), and a node's mesh is the model's own, as the
  // materials topic counts them. Unnamed nodes are named from their place in the file.
  static fv::Json describeModel(PropNodes& out, const tinygltf::Model& model, std::vector<rs::EntityId> entityOf);

  // The props now in the scene, after a composition; republishes the selection.
  void setProps(std::vector<PropNodes> props);
  // The prop the selection is in, or "".
  const std::string& selectedProp() const { return m_selectedProp; }
  // every prop selected, the primary (selectedProp) first: select's props, add and toggle, and a
  // Ctrl+click in the viewport, gather several; any other selection is one again
  const std::vector<std::string>& selectedProps() const { return m_selectedProps; }
  // the selection, if it is in `prop`, cleared and published (a prop leaving the scene)
  void deselect(const std::string& prop);
  // Whether an entity is a prop root, and which prop's it is.
  const PropNodes* propOfRoot(rs::EntityId entity) const;

  // The pick mode and the gizmo's handles in it, for the host's gizmo hooks.
  const std::string& pickMode() const { return m_pick; }
  const std::string& transformMode() const { return m_transform; }
  // Whether the host draws its camera and light markers, and which prop's marker is at a pixel
  // ("" for none), asked before a click's ray pick.
  bool helpersShown() const { return m_helpers; }
  // The frame the view is composed for (view.set's frameAspect), 0 for none: camera markers take its shape.
  double frameAspect() const { return m_frameAspect; }
  void onHelperPick(std::function<std::string(const glm::vec2& px)> f) { m_helperPick = std::move(f); }
  // The Pull tool's drag, in render pixels: "down" (true when it took hold of a figure: the drag is
  // then its, not the camera's), "move", "up".
  void onPull(std::function<bool(const std::string& phase, const glm::vec2& px)> f) { m_pull = std::move(f); }
  // view.set's solo: the props Isolate shows (empty: all), which the host applies
  void onSolo(std::function<void(const std::vector<std::string>&)> f) { m_solo = std::move(f); }
  // A selected node's place and whether it shows ({visible, t, e, s, edited}), for the selection topic's
  // `state`; null for one the host doesn't let be edited (a bone, a worn item's)
  void onNodeState(std::function<fv::Json(const std::string& prop, int node)> f) { m_nodeState = std::move(f); }
  // the selection topic again, as something about the selected node changed
  void refreshSelection() { publishSelection(true); }

  // The lens of the view in use: the free view's, or the camera prop's it looks through.
  struct Lens
  {
    double      focalLength   = 40;
    bool        dof           = false;
    double      fStop         = 2.8;
    std::string focusMode     = "target";
    double      focusDistance = 3;
    int         blades        = 0;  // aperture blades: 0 a round aperture, else 3 to 16 straight blades
    double      bladeRotation = 0;  // degrees, the first blade's corner from the camera's right toward its up
  };
  const Lens& lens() const { return m_lens; }
  void        setLens(const Lens& lens);
  // camera.set changed the lens: the host keeps it with the view in use.
  void onLensChange(std::function<void(const Lens&)> f) { m_onLens = std::move(f); }
  // More for the "camera" topic: the camera prop looked through ("view") and where the free view
  // was left ("free"), merged into the next snapshot.
  void setCameraExtra(fv::Json extra);

  // Pointer events, in device pixels. The gizmo takes its own (a drag on a handle); a left click
  // that barely moved picks, and a double-click turns the view to the surface under it, the point
  // the view then orbits (on nothing, the whole scene is framed). Returns the events the gizmo
  // didn't take, for the camera.
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
  // A click hit `at` (prop -1: nothing): the surface it hit, for the next selection published
  void notePick(const fv::HostAddress& at);
  // The pixel readout's topic, `inspect`: what a reading found at m_inspectAt (view.inspect)
  void publishInspect(const HostRenderer::PixelReadout& readout);
  // The Time view's topic, `time`: the frame's time by what its pixels show, the largest kPixelTimeLines
  // and the rest ({parts: [{what, prop, node, name, share, cost}], samples}; null once the view is off)
  void                    publishPixelTimes(const HostRenderer::PixelTimes& times);
  static constexpr size_t kPixelTimeLines = 6;

  HostRenderer*                                      m_renderer = nullptr;
  fv::EventHub*                                      m_events   = nullptr;
  std::function<fv::Json(std::function<fv::Json()>)> m_onApp;

  std::vector<PropNodes>                        m_props;
  fv::SceneMap                                  m_map;  // entities and materials -> m_props
  std::string                                   m_selectedProp;
  rs::EntityId                                  m_lastEntity;
  bool                                          m_publishAgain = true;
  std::string                                   m_surfaceProp, m_surface;  // what the last click hit
  std::optional<glm::dvec2>                     m_inspectAt;  // the pixel read out (view.inspect), render pixels; none: off
  bool                                          m_pixelTime = false;  // the display is the Time view
  std::string                                   m_pick      = "object";
  bool                                          m_select    = true;  // a click picks (view.set select)
  std::string                                   m_transform = "move";
  bool                                          m_helpers   = true;
  double                                        m_frameAspect = 0;
  std::function<std::string(const glm::vec2&)>  m_helperPick;
  std::function<bool(const std::string&, const glm::vec2&)> m_pull;
  std::function<void(const std::vector<std::string>&)>      m_solo;
  std::function<fv::Json(const std::string&, int)>          m_nodeState;
  // the props selected, the primary first (m_selectedProp), and whether the click being picked held Ctrl
  std::vector<std::string> m_selectedProps;
  bool                     m_clickCtrl = false;
  HostRenderer::GizmoOptions m_gizmo;  // view.set's space and snap
  // a prop in or out of the selection, the primary moving to what is left
  void toggleMember(const std::string& prop);
  const PropNodes* propNodes(const std::string& prop) const;
  bool                                          m_pulling = false;  // a Pull drag holds a figure

  fv::Json                              m_camera;
  fv::Json                              m_cameraExtra = fv::Json::object();
  std::chrono::steady_clock::time_point m_cameraAt{};

  Lens                             m_lens;
  std::function<void(const Lens&)> m_onLens;
  double                           m_aspect = -1;

  bool   m_click = false;
  bool   m_lmb   = false;
  double m_x0 = 0, m_y0 = 0;
  // the last click that picked, for the second click of a double-click
  std::chrono::steady_clock::time_point m_lastClick{};
  double                                m_lastX = 0, m_lastY = 0;
  bool                                  m_gizmoPress = false;  // the gizmo took the last press
};
