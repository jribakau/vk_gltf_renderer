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
// The Figure Viewer's scene, hosted in the renderer: DAZ figures load here (figure/), are built
// into the renderer's glTF scene, and are posed, morphed and dressed here, while the UI only sends
// commands and shows the streamed frames (fv_protocol.hpp). One element per renderer.
//
// Threads: a load runs on a worker; everything that touches the renderer's scene or the loaded
// figure runs on the application thread, in onUpdate, before the frame renders. Commands that need
// the scene wait for that (a frame at most); the rest answer on the HTTP thread.
//
// Commands (args):
//   settings.apply  {libraryRoots: [dir], rootId, texServer, maxTextureSize, subdivision}
//   load            {path} or {rel}              -> {job}; the figure replaces the scene when done
//   pose.set        {figure, values: {"bone/kind/axis": value}}
//   pose.reset      {figure}
//   pose.eyes       {figure}                     aim the eyeballs at the camera
//   morph.set       {figure, values: {name: value}}   coalesced, one setMany per frame
//   morph.reset     {figure}
//   render.mode     {mode: "raster" | "rtx"}
//   material.set    {material, values}           edit a surface (fv_materials.hpp); material.reset {material?}
//   lighting.set    a merge patch of the Light tab (fv_lighting.hpp) -> the whole state
//   params.get      {names?}                     registered renderer parameters, as nvmcp reads them
//   params.set      {values: {name: value}}      validated as a batch, then written; callbacks fire
//   session.shutdown {}                          a clean exit, before the host ends the process
//
// Lighting: the environment, tone mapping, the raster shadow map and the ground are renderer
// parameters; the sun is a directional light the host adds to every model it loads, marked as the
// sky's sun. An HDR decodes on a worker (a "jobs" entry) and uploads on the application thread. An
// environment preset .duf sets the HDR from its library and turns the sun off.
//
// Camera, selection and display commands live in fv_view.hpp. A .duf that is a pose preset is
// applied by `load` to the figure already loaded; other presets are reported and left alone.
//
// Topics: "session", "jobs", "notices", "scene", "pose/<figure>", "morphs/<figure>", "render",
// "selection", "materials" (fv_materials.hpp), "lighting" (the lighting.set shape, plus read-only env.name, env.size, env.loading). "pose/<figure>" carries `eyes` (whether the figure can look at the camera).
// "morphs/<figure>" is the visible channels ({label, group, value, min, max, weight, order}); a new
// channel list replaces it, and value or weight changes arrive as a merge patch. A figure's busy()
// line is a "jobs" entry, and takeWarnings() goes to "notices".
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
