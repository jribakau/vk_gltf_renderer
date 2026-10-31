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
//   settings.apply  {libraryRoots: [dir], rootId, texServer, maxTextureSize, subdivision, originalTextures}
//   load            {path} or {rel}, prop?, as?   -> {job}; a figure or model becomes a new prop (the job's
//                                                end state has its {prop}); a wearable goes on the figure
//                                                (`prop`, or the selected or only one) unless `as` is "prop";
//                                                a pose, shape or material preset applies to that figure;
//                                                `prefetch` [rel] names files a figure will wear soon (a
//                                                scene file's worn list), read from the start of its load
//   prop.add        {path|rel} as load; or {kind: "camera"|"light", type?: point|spot|directional,
//                   name?, camera?, transform?}  -> {prop}; a camera stands where the view is
//   prop.set        {prop, name?, visible?, transform?: {t, e (degrees XYZ) | r (quaternion xyzw), s},
//                   parent?: prop id | null, bone?: a posable bone of the parent figure (id or name) |
//                   null (with a new parent, none unless given; alone, another bone of the same figure):
//                   it rides on the bone, posed with it, its transform relative to the bone,
//                   order?, camera?: {focalLength, dof, fStop, focusMode,
//                   focusDistance, locked, targetDistance}, light?: {type, color, intensity, radius, inner, outer}}
//   prop.remove     {prop}                       what rode on it stays where it was in the world
//   prop.duplicate  {prop}                       -> {prop}, or {job} for a figure (loaded again, its state put back)
//   scene.clear     {}                           every prop goes
//   scene.order     {order: [prop]}              the props in this order (the ones it leaves out after them), one composition
//   pose.set        {prop, values: {"bone/kind/axis": value}, gesture?}
//   pose.reset      {prop}
//   pose.eyes       {prop}                       aim the eyeballs at the camera
//   morph.set       {prop, values: {name: value}, gesture?}   coalesced, one setMany per frame
//   morph.reset     {prop}                                    every dial back to where its file rests it
//   render.mode     {mode: "raster" | "rtx", skin?: "measured" | "volume" | "thin"}
//   render.status   {}  -> {refining, still, converged, complete, rtx, width, height}   where the still is, and the
//                                                frame's size (a /size lands before the next frame)
//   render.capture  {path}  -> {samples, converged}   the frame as a linear .hdr (path tracer), answered once saved
//   render.budget   {}  -> {known, budgetBytes, usageBytes, bytesPerPixel, width, height}   GPU memory: the
//                                                driver's budget and what a pixel of the frame costs, so an
//                                                export size is judged before /size refuses it (507)
//   material.set    {prop, material, values, gesture?}   edit a surface (fv_materials.hpp); material.reset {prop, material?}
//                                                `gesture` {id, end?}: a slider's drag, one step of the history
//                                                however long it takes, until the message that ends it
//   material.eyes   {prop, rel: string | null}   -> {job}: a figure's eyes become the eye preset `rel` (null: the
//                                                character's own, its file's and its add-ons' presets). Each
//                                                choice starts from the character's own, so eyes don't stack,
//                                                takes only the preset's eye surfaces (kind eye, corneas but
//                                                the tear film) and drops the panel's edits of them. Read on a
//                                                worker, applied once the add-ons are on. The job ends {state:
//                                                "done", eyes: rel | null, changed: n}, or "failed" with the
//                                                error for a prop that is no figure, a figure without eyes, or
//                                                a preset none of whose surfaces are its eyes. A material
//                                                preset opened with `load` that is only eyes goes this way too;
//                                                one that also sets other surfaces is recorded as the eyes'
//                                                `preset`. Material presets wait for the add-ons as well.
//                                                "materials/<prop>".eyes = {rel, preset, own: [rel] (the add-on
//                                                presets), ready (the add-ons are on), surfaces: ["m:N"]}
//   material.skin   {prop, rel: string | null}   -> {job}: the same for the character's skin: its surfaces
//                                                but the eyes and those of worn clothing, props and hair (a
//                                                hair's brows and lashes are the character's; skin_materials.hpp).
//                                                Drops the panel's colours on them (base colour, tone, recolour,
//                                                makeup) and the makeup chosen. The job ends {state: "done",
//                                                skin, changed}. A material preset that reaches the skin is
//                                                recorded as the skin's `preset`
//   material.makeup {prop, rel: string | null}   -> {job}: makeup laid over the skin on now: the channels the
//                                                last makeup set go back (null: only that), then the preset's
//                                                part on the skin; drops the panel's makeup colours. Ends {state:
//                                                "done", makeup, changed}. "materials/<prop>".skin = {rel, preset,
//                                                makeup, ready, surfaces: ["m:N"]}
//   wear            {prop, rel | path}           put a wearable on a figure -> {job}
//   wardrobe.remove {prop, item}, wardrobe.visible {prop, item, visible}   hidden in place; a removed item
//                                                leaves the renderer's scene at the next composition, and a geograft composes
//   wardrobe.color  {prop, item | [item...], rel: string | null}   -> {job}: the item's colour becomes the
//                                                material preset `rel` (null: the item as it was worn). Each
//                                                choice starts from the item as it was worn, so colours don't
//                                                stack, takes only the preset's part on the item's surfaces and
//                                                drops the panel's edits of them but the tiling, fabric and
//                                                sheen (uv*, fabric*, sheen*). Several items go as one (a hair
//                                                worn as three pieces). The job ends {state: "done", color, changed},
//                                                or "failed" when nothing of the preset lands on the items. Kept
//                                                in the scene file (worn[].color); a preset opened later that
//                                                reaches the item becomes its colour
//   wardrobe.collision {prop, on}                 worn conforming meshes are pushed out of the posed body
//                                                (figure/wardrobe/collider.hpp), on by default
//   wardrobe.cloth  {prop, item, on?: bool | null, strength?}   dForce on one item: true drapes it (at the
//                                                defaults when its file has no simulation settings), false
//                                                holds it, null leaves it to the file; `strength` scales
//                                                the weights (figure.hpp, clothSettings)
//   wardrobe.tight  {prop, item, amount}          how skin tight one item is, 0 as made to 1 on the skin;
//                                                the collider applies it (figure/wardrobe/collider.hpp)
//   dforce.enable   {prop, on}; dforce.resimulate {prop}   cloth drapes over the body (figure/dforce)
//   dforce.mode     {prop, mode: "settle" | "live" | "gpu"}   one settle once the pose stops (on the GPU solver
//                                                where there is one), a step every frame, or that on the GPU
//                                                solver. The app offers only settle: real-time cloth is paused
//                                                (FV_DFORCE_GPU builds; the
//                                                "session" topic's features.dforceGpu says, and a scene file's
//                                                "gpu" restores as live without it)
//   scene.state / scene.restore {prop}           a scene file's part of a prop: source, worn, dials, pose,
//                                                material presets, skin, eyes and makeup (rel | null, once
//                                                chosen), material edits, collision and dForce on/off, dForce's
//                                                mode; restored after the add-ons: presets (eye-only ones as
//                                                eyes), skin, eyes, makeup, worn items' colours, edits. A worn
//                                                item's `id` is the "i:N" it goes back on under; the edits name
//                                                the surfaces in use, so a taken-off item's count for nothing. A prop's id is
//                                                never given out twice, scene.clear or not
//   anim.select|play|pause|stop|seek|speed|loop  the clips in the scene (glTF and OBJ models'), by the
//                                                number the "animation" topic gives each: it stays the
//                                                clip's while its prop is in the scene, never reused
//   export.glb      {prop, path} or {scene: true, path}   a GLB of a figure -> {job}; of a model, or of the
//                                                whole scene, written at once -> {ok}
//   lighting.set    a merge patch of the Light tab (fv_lighting.hpp) -> the whole state
//   params.get      {names?}                     registered renderer parameters, as nvmcp reads them
//   params.set      {values: {name: value}}      validated as a batch, then written; callbacks fire
//   history.undo / history.redo {}  -> {label, problems?}, or {label, job} while a prop it puts back loads:
//                                                the step undone or redone, its label null when there is
//                                                none; another waits for the job ("history".busy). A step is
//                                                a prop added (load, prop.add, prop.duplicate), removed
//                                                (prop.remove, scene.clear: it comes back under its id, its
//                                                objects kept or, past the few kept, loaded again from its
//                                                scene-file part), an item put on (wear, load) or taken off
//                                                (wardrobe.remove: worn again under its id, as it was), a
//                                                preset or a look chosen (load of a pose, shape, property,
//                                                material or environment preset, material.skin/eyes/makeup,
//                                                wardrobe.color: the figure's look kept, surfaces and all), or a
//                                                panel edit (prop.set, morph.*,
//                                                pose.*, material.set/reset, wardrobe.visible/collision/cloth,
//                                                dforce.enable/mode, lighting.set, scene.order, view.use, and
//                                                camera.set looking through a camera prop), captured as it
//                                                applies (fv_host_journal.cpp); a slider's edits merge into one.
//                                                A gizmo drag of a prop or a bone, and a drag of the view
//                                                looking through a camera prop, are one step from press to release
//   history.begin {label} / history.end {}      what is recorded between them is one step under `label` (a
//                                                scene file opening), its undo the other way round; undo
//                                                waits for the end ("history".busy)
//   history.clear {}                             the history goes (the app's settings applied at the start)
//   recovery.file {path | null}                  where the scene is kept for a crash from now on: the ops
//                                                that put every prop back under its id (records, as
//                                                scene.clear's undo has them), the order, the lighting and
//                                                the view, rewritten on a worker a second after the changes
//                                                pause, whole then renamed into place. null: kept no more,
//                                                and the copy goes (the scene closed as meant)
//   recovery.peek {path}                         -> {props, at}: what a kept scene holds (props 0: none, or
//                                                unreadable), `at` when it was kept (ms since 1970)
//   scene.recover {path}                         a kept scene put back into an empty one, as an undo runs its
//                                                ops -> {label, problems?}, or {label, job} while its props load
//   session.shutdown {}                          a clean exit, before the host ends the process
//
// The scene: the host feeds the renderer's retained scene itself (HostRenderer::hostScene): a prop
// root per prop, a figure through its feed (fv_figure_feed.hpp), a model imported from its file, a
// camera or a light as a record, and the sun. A composition feeds the props' changes as deltas
// (fv_prop_feed.hpp); what didn't change keeps its handles.
//
// Lighting: the environment, tone mapping, the raster shadow map and the ground are renderer
// parameters; the sun is a directional light the host adds to every scene it composes, marked as the
// sky's sun. Light props are lights of their own beside it. An HDR decodes on a worker (a "jobs"
// entry) and uploads on the application thread. An environment preset .duf sets the HDR from its
// library and turns the sun off.
//
// Camera, selection and display commands live in fv_view.hpp.
//
// Topics: "session", "jobs", "notices", "render", "lighting", "history" ({undo: [label], redo: [label], busy},
// the next to undo or redo first), "animation" ({clips: {number: {name,
// duration, order, prop}}, current: number | null, playing, time, speed, loop}), "selection", "camera",
// "scene" ({props: {id: {kind, name, parent, bone, visible, order, transform: {t, r, e, s}, stats?, source?,
// format?, generation?, posable?, camera?, light?}}, order: [id]}), and per prop "nodes/<prop>"
// (its hierarchy), "materials/<prop>" (fv_materials.hpp), "pose/<prop>", "morphs/<prop>",
// "wardrobe/<prop>" ({figure, items: {id: {label, kind, rel, visible, own, channels, collision,
// cloth: {able, native, on, strength}, meshes: [scene mesh], color?: rel | null}}, collision: {enabled,
// busy, corrected}}; `color` is the colour preset on the item, from wardrobe.color or the last preset that
// reached it, null for its own; absent, it is as it was worn), "dforce/<prop>"
// ({figure, enabled, mode, active, busy, draped}); a prop's topics
// are dropped when it leaves the scene. "lighting" is the lighting.set shape, plus read-only
// env.name, env.size, env.loading. "pose/<prop>" carries `eyes` (whether the figure can look at
// the camera). "morphs/<prop>" is the visible channels ({label, group, value, min, max, weight,
// order}, and what a file says of its dial when that has been read: path, region, kind, helper, rest
// when it isn't 0, inert when it holds nothing; reach {cm, deg, morphs, bones, scales} a little after
// the load, and never on a dial no file holds, which counts as unknown; items, the worn items it moves
// through something of their own);
// a new channel list replaces it, and value or weight changes arrive as a merge patch. A
// figure's busy() line is a "jobs" entry, and takeWarnings() goes to "notices". "jobs" and
// "notices" keep only the latest finished jobs and notices (kKeptJobs, kKeptNotices); older ones
// leave with a null in a later patch.
//

#include <memory>
#include <string>

#include <nvapp/application.hpp>

class HostRenderer;
class FvProtocol;
namespace nvutils {
class ParameterRegistry;
}

class FvHost : public nvapp::IAppElement
{
public:
  // `registry` holds the renderer's parameters, which lighting and params.* write.
  FvHost(HostRenderer* renderer, std::shared_ptr<FvProtocol> protocol, nvutils::ParameterRegistry* registry);
  ~FvHost() override;

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onUpdate() override;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};
