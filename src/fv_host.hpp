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
// The Figure Viewer's scene, hosted in the renderer: a scene of props, any number of DAZ figures,
// models read from glTF, GLB or OBJ files, cameras and lights. Figures load here (figure/) and are
// posed, morphed and dressed here; every prop sits under a prop root node of its own in the
// renderer's one glTF (fv_compose.hpp), which carries where it stands. The UI only sends commands
// and shows the streamed frames (fv_protocol.hpp). One element per renderer.
//
// Threads: a load runs on a worker; everything that touches the renderer's scene or a figure runs
// on the application thread, in onUpdate, before the frame renders. Commands that need the scene
// wait for that (a frame at most); the rest answer on the HTTP thread. A change to the scene's
// structure (a prop added or removed, something worn, a new parent) composes the scene again, once
// a frame however many asked; moving, showing and hiding a prop, dials, poses and material and light
// edits apply in place.
//
// A command for a prop names it as {prop: "p:3"} ({figure} is read the same way); without one it is
// for the selected prop, and a figure command then for the only figure, if there is one.
//
// Commands (args):
//   settings.apply  {libraryRoots: [dir], rootId, texServer, maxTextureSize, subdivision}
//   load            {path} or {rel}, prop?, as?   -> {job}; a figure or model becomes a new prop (the job's
//                                                end state has its {prop}); a wearable goes on the figure
//                                                (`prop`, or the selected or only one) unless `as` is "prop";
//                                                a pose, shape or material preset applies to that figure
//   prop.add        {path|rel} as load; or {kind: "camera"|"light", type?: point|spot|directional,
//                   name?, camera?, transform?}  -> {prop}; a camera stands where the view is
//   prop.set        {prop, name?, visible?, transform?: {t, e (degrees XYZ) | r (quaternion xyzw), s},
//                   parent?: prop id | null, order?, camera?: {focalLength, dof, fStop, focusMode,
//                   focusDistance, locked, targetDistance}, light?: {type, color, intensity, radius, inner, outer}}
//   prop.remove     {prop}                       what rode on it stays where it was in the world
//   prop.duplicate  {prop}                       -> {prop}, or {job} for a figure (loaded again, its state put back)
//   scene.clear     {}                           every prop goes
//   scene.order     {order: [prop]}              the props in this order (the ones it leaves out after them), one composition
//   pose.set        {prop, values: {"bone/kind/axis": value}}
//   pose.reset      {prop}
//   pose.eyes       {prop}                       aim the eyeballs at the camera
//   morph.set       {prop, values: {name: value}}   coalesced, one setMany per frame
//   morph.reset     {prop}
//   render.mode     {mode: "raster" | "rtx", skin?: "volume" | "thin"}
//   material.set    {prop, material, values}     edit a surface (fv_materials.hpp); material.reset {prop, material?}
//   wear            {prop, rel | path}           put a wearable on a figure -> {job}
//   wardrobe.remove {prop, item}, wardrobe.visible {prop, item, visible}   hidden in place; a removed item
//                                                leaves the renderer's scene at the next composition, and a geograft composes
//   wardrobe.collision {prop, on}                 worn conforming meshes are pushed out of the posed body
//                                                (figure/wardrobe/collider.hpp), on by default
//   wardrobe.cloth  {prop, item, on?: bool | null, strength?}   dForce on one item: true drapes it (at the
//                                                defaults when its file has no simulation settings), false
//                                                holds it, null leaves it to the file; `strength` scales
//                                                the weights (figure.hpp, clothSettings)
//   dforce.enable   {prop, on}; dforce.resimulate {prop}   cloth drapes over the body (figure/dforce)
//   dforce.mode     {prop, mode: "settle" | "live" | "gpu"}   one settle once the pose stops, a step every
//                                                frame, or that on the GPU solver (FV_DFORCE_GPU builds; the
//                                                "session" topic's features.dforceGpu says, and a scene file's
//                                                "gpu" restores as live without it)
//   scene.state / scene.restore {prop}           a scene file's part of a prop: source, worn, dials, pose,
//                                                material edits, collision and dForce on/off, dForce's mode
//   anim.select|play|pause|stop|seek|speed|loop  the clips in the scene (glTF and OBJ models')
//   export.glb      {prop, path} or {scene: true, path}   a GLB of a figure -> {job}; of a model, or of the
//                                                whole scene, written at once -> {ok}
//   lighting.set    a merge patch of the Light tab (fv_lighting.hpp) -> the whole state
//   params.get      {names?}                     registered renderer parameters, as nvmcp reads them
//   params.set      {values: {name: value}}      validated as a batch, then written; callbacks fire
//   session.shutdown {}                          a clean exit, before the host ends the process
//
// Lighting: the environment, tone mapping, the raster shadow map and the ground are renderer
// parameters; the sun is a directional light the host adds to every scene it composes, marked as the
// sky's sun. Light props are lights of their own beside it. An HDR decodes on a worker (a "jobs"
// entry) and uploads on the application thread. An environment preset .duf sets the HDR from its
// library and turns the sun off.
//
// Camera, selection and display commands live in fv_view.hpp.
//
// Topics: "session", "jobs", "notices", "render", "lighting", "animation", "selection", "camera",
// "scene" ({props: {id: {kind, name, parent, visible, order, transform: {t, r, e, s}, stats?, source?,
// format?, generation?, posable?, camera?, light?}}, order: [id]}), and per prop "nodes/<prop>"
// (its hierarchy), "materials/<prop>" (fv_materials.hpp), "pose/<prop>", "morphs/<prop>",
// "wardrobe/<prop>" ({figure, items: {id: {label, kind, rel, visible, own, channels, collision,
// cloth: {able, native, on, strength}}}, collision: {enabled, busy, corrected}}), "dforce/<prop>"
// ({figure, enabled, mode, active, busy, draped}); a prop's topics
// are dropped when it leaves the scene. "lighting" is the lighting.set shape, plus read-only
// env.name, env.size, env.loading. "pose/<prop>" carries `eyes` (whether the figure can look at
// the camera). "morphs/<prop>" is the visible channels ({label, group, value, min, max, weight,
// order}); a new channel list replaces it, and value or weight changes arrive as a merge patch. A
// figure's busy() line is a "jobs" entry, and takeWarnings() goes to "notices". "jobs" and
// "notices" keep only the latest finished jobs and notices (kKeptJobs, kKeptNotices); older ones
// leave with a null in a later patch.
//

#include <memory>
#include <string>

#include <nvapp/application.hpp>

class GltfRenderer;
class FvProtocol;
namespace nvutils {
class ParameterRegistry;
}

class FvHost : public nvapp::IAppElement
{
public:
  // `registry` holds the renderer's parameters, which lighting and params.* write.
  FvHost(GltfRenderer* renderer, std::shared_ptr<FvProtocol> protocol, nvutils::ParameterRegistry* registry);
  ~FvHost() override;

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onUpdate() override;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};
