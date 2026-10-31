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
// bone (prop.set's `bone`, the scene topic's).
constexpr int kProtocolVersion = 15;
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
    res.set_content(answer.dump(), "application/json");
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
        for(const fv::Event& e : m_events.snapshots(&rev))
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
