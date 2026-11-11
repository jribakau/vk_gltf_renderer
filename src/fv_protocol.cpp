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

// httplib pulls in winsock2; it has to come before anything that includes windows.h.
#include <httplib.h>

#include "fv_protocol.hpp"

#include <chrono>
#include <string>

using namespace std::chrono_literals;

namespace {
// 2: the scene of props (fv_host.hpp). An app checks it against what it was built for.
// 3: the upstream merge of 2026-10-24 renamed ptMaxFrames to ptMaxSamples and optixAutoDenoiseInterval
// to optixAutoDenoiseSpp, which the app sends.
// 5: dforce.mode. 6: wardrobe.collision, and the wardrobe topic's collision state. 7: render.status's
// width and height. 8: the Clothes tab's material values (maps, UV transform, fabric, sheen) and the
// texture slots in the materials topic (fv_materials.hpp). 9: the animation topic's clips keep their
// numbers while their prop is in the scene (never reused), and `current` is null without clips. 10:
// the edit history, history.undo and history.redo and the "history" topic (fv_journal.hpp). 11:
// `gesture` {id, end?} on morph.set, pose.set and material.set: a slider's drag is one step. 12:
// history.begin, history.end (a scene file opening as one step) and history.clear. 13: the scene
// kept for a crash, recovery.file, recovery.peek and scene.recover. 14: a prop rides on a figure's
// bone (prop.set's `bone`, the scene topic's). 16: view.set's frameAspect, render.save {path}
// (an image of the frame, answered once it is on disk), and the event stream's "synced" after the
// snapshots it starts or starts over with. 17: the path tracer only: render.status has no `rtx`,
// the lighting has no sun.shadows or shadow section (the raster view's shadow map; still taken from
// a kept scene). 18: render.mode refuses "raster", and a skin it already has changes nothing. 19: a
// DAZ animation played on a figure: load's `as` "animation", the animation topic's clips with a
// `kind` ("clip" or "figure"), and anim.remove. 20: body contact: figure.contact, the contact/<prop>
// topic, and the scene state's `contact`. 21: Physics mode: physics.set, the physics/<prop> topic,
// the scene state's `physics`, and view.set's pick "pull". 22: water on the ground, the lighting's
// ground.water and ground.ripples. 23: a Hold per body group, physics.set's and the physics topic's
// and scene state's `parts`. 24: the chest's soft bones: physics.secondary, the secondary/<prop> topic
// and the scene state's `secondary`. 25: a body of water over the ground, the lighting's ground.mode
// "water" with ground.depth and ground.waves.
// 26: pins and two ragdolls: physics.pin, the physics topic's and scene state's `pins`.
// 27: soft flesh: physics.soft, the soft/<prop> topic and the scene state's `soft`.
// 28: VaM Timeline clips: load's `as` "vam", and the animation topic's clip `vam`.
// 29: self-contact: figure.contact's, the contact/<prop> topic's and the scene state's `self`.
// 30: light cloth: dforce.light, the dforce/<prop> topic's and the scene state's `dforceLight`.
// 31: clouds over the physical sky, the lighting's clouds section (coverage, thickness, seed).
// 32: the kind of cloud, the lighting's clouds.type (cumulus, stratocumulus, stratus).
// 33: clouds.type "altocumulus".
// 34: the clouds in three tiers: clouds.mid (altocumulus) and clouds.high (cirrus); clouds.type is
// the low tier's, and "altocumulus" is no longer one of them.
// 35: the exposure follows the clouds: tonemap.followClouds and tonemap.skyShare, and the read-only
// tonemap.cloudLight, what the clouds leave of the light.
// 36: haze in the air, the lighting's haze section (enabled, visibility, glow).
// 37: ptSamplesDlss, the samples a DLSS frame traces (an older renderer refuses the app's whole
// batch of render parameters for it).
// 38: uneven haze, the lighting's haze.height and haze.patches.
// 39: looks, a grade over the Image sliders: the lighting's tonemap.look and tonemap.lookStrength,
// and lighting.preview, a look shown while the app's pointer is on its chip.
// 40: bloom, the lighting's tonemap.bloom and tonemap.star (a glow round bright light, and a star
// across the aperture's blades).
// 41: local exposure, the lighting's tonemap.localShadows and tonemap.localHighlights (dark regions
// lifted and bright ones held toward mid grey, region by region).
// 42: a named metal, material.set's `metal` and the materials topic's values.metal (an older
// renderer refuses the whole material.set for it, as the looks send it).
// 43: the Shaped tone curve, the lighting's tonemap.method "shaped" and its tonemap.toe and
// tonemap.shoulder.
// 44: the pixel readout, view.inspect and the inspect topic (what is at a pixel, and how bright).
// 45: props by kind: the scene topic's kind is character, creature, vehicle, set, prop, group, model,
// camera or light (a DAZ file is never "figure"), with `morphs` and `body` beside `posable`; load's and
// prop.add's `kind` (the app's reading of the library), prop.set's `kind` (treat a DAZ object as another),
// scene.state's `kind`, prop.add's kind "group"; wear, collision, dForce, contact and Physics mode name a
// body (a character or a creature).
// 46: placing: prop.drop (rest on what is under it, else the ground), prop.aim (look at a prop or a
// point), view.set's solo (Isolate) and the scene topic's `solo`, and prop.set's `label`.
// 47: a node inside a prop: node.set (hidden, moved, or back), the nodes topic's `hidden`, the
// selection's `state` for a node, and scene.state's `nodes`.
// 48: several selected: select's props, add and toggle, the selection topic's `props`, a Ctrl+click (since 60 a Ctrl+Shift+click)
// toggling, the gizmo moving them together; prop.setMany; prop.add's group `props` and prop.ungroup.
// 49: the gizmo's space ("world", "local") and snap ({move, rotate, scale} | null) in view.set.
// 50: prop.place, the selection put where the pointer's ray meets the scene.
// 51: figure.rain and the rain/<prop> topic: rain's drops on a person's skin, run along the pose; the
// scene file's `rain` block.
// 52: view.set's display "material", "object", "density" and "facing" (the Display views).
// 53: view.set's display "time" (each pixel timed by the GPU's clock) and its `time` topic, the frame's
// time by what the pixels show.
// 55: the pose topic's bones name their counterpart on the other side (`mirror`, read by the
// skeleton spec), which the Pose tab's mirror swaps by.
// 57: a look's cheap camera: tonemap.look's bend and fringe (its lens: barrel distortion and colour
// fringing) and clip and noise (its sensor: highlights clipped a channel at a time, noise in the
// shadows). An older renderer refuses the lighting for a look that has them.
// 58: the lens's flash: camera.set's and a camera prop's `flash`, a light on the camera (0 none, else
// its strength as a spot light prop's), in the camera topic's and a camera prop's lens.
// 59: the hover: view.set's `hover` and the input event {t: "hover", x, y} (without x and y, the
// pointer left): what is under a resting pointer is outlined, a character whole, a set by its nodes.
// 60: a Ctrl+click picks whatever view.set's `select` says (view.set's `selectCtrl` turns that off),
// and it is a Ctrl+Shift+click that puts a prop in the selection or takes it out (it was a Ctrl+click).
// 61: an object pick selects what the hover outlines: of a set the node hit (the selection topic's
// `node`), where the gizmo then moves that part as node.set does; any other prop whole, and the hover
// outlines a prop, a vehicle and a model whole too (they were by their nodes).
// 62: an editor's scheme for the view (Unity, Unreal): Alt+left orbits, the middle pans, Alt+right
// dollies, the right alone looks around and flies, the left alone moves nothing; a drag turns about
// the selection's middle, else the point under the pointer as it starts, a double-click's point until
// the selection changes. A Ctrl+click puts a prop in the selection or takes it out (it was
// Ctrl+Shift), an Alt+click picks nothing.
// 63: tonemap.highlightsToWhite (how fast Neutral's and Shaped's compressed colours go white; an older
// renderer refuses the lighting that has it), and render.whiteBalance (the white balance that makes the
// frame's light grey, for the app's Auto).
// 64: the backdrop: the lighting's `backdrop` section (kind, seed, size, density, relief; read-only
// state and plants), ground and plants from a kit the app fills (settings.apply's backdropKit and
// backdropKitChanged). An older renderer refuses the lighting that has the section.
// 65: backdrop.air (how many metres of air a metre of the backdrop's land counts for: the haze of its
// distance) and env.skyVisibility (km one sees through the physical sky's air; 0 the atmosphere's
// own). An older renderer refuses the lighting that has either.
constexpr int kProtocolVersion = 65;
// An idle event stream sends a comment this often, which keeps it open and finds dead clients.
constexpr auto kHeartbeat = 15s;
}  // namespace

FvProtocol::FvProtocol()
{
  m_commands.add("hello", [this](const fv::Json&) {
    return fv::Json{{"protocol", kProtocolVersion}, {"commands", m_commands.names()}, {"features", m_features}};
  });
  m_events.snapshot("session", {{"protocol", kProtocolVersion}, {"features", m_features}});
}

void FvProtocol::setFeatures(fv::Json features)
{
  m_features = std::move(features);
  m_events.snapshot("session", {{"protocol", kProtocolVersion}, {"features", m_features}});
}

void FvProtocol::registerRoutes(httplib::Server& http, const std::atomic<bool>& running)
{
  http.Post("/cmd", [this](const httplib::Request& req, httplib::Response& res) {
    fv::Json request = fv::Json::parse(req.body, nullptr, /*allow_exceptions=*/false);
    fv::Json answer  = request.is_discarded() ? fv::Json{{"ok", false}, {"error", "the body is not JSON"}} : m_commands.dispatch(request);
    // the rev after the command: events up to it include whatever the command changed
    answer["rev"] = m_events.rev();
    res.set_content(answer.dump(-1, ' ', false, fv::Json::error_handler_t::replace), "application/json");
  });

  http.Post("/input", [this](const httplib::Request& req, httplib::Response& res) {
    fv::Json events = fv::Json::parse(req.body, nullptr, /*allow_exceptions=*/false);
    if(!events.is_array())
    {
      res.status = 400;
      res.set_content("an array of input events", "text/plain");
      return;
    }
    input(events);
    res.status = 204;
  });

  http.Get("/events", [this, &running](const httplib::Request&, httplib::Response& res) {
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider("text/event-stream", [this, &running, rev = uint64_t(0), started = false,
                                                           idle = std::chrono::steady_clock::now()](size_t, httplib::DataSink& sink) mutable {
      if(!running || m_events.closed())
        return false;
      std::string out;
      if(!started)
      {
        started = true;
        for(const fv::Event& e : m_events.subscribe(&rev))
          out += fv::EventHub::sse(e);
      }
      else
      {
        for(const fv::Event& e : m_events.next(rev, 500ms))
          out += fv::EventHub::sse(e);
      }
      const auto now = std::chrono::steady_clock::now();
      if(out.empty() && now - idle >= kHeartbeat)
        out = ": ping\n\n";
      if(out.empty())
        return sink.is_writable();
      idle = now;
      return sink.write(out.data(), out.size());
    });
  });
}
