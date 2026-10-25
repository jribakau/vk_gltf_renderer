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
// Figure Viewer host protocol over HTTP, served by the --stream server (frame_stream.hpp) next to
// the frames, so the UI talks to the renderer directly instead of through the host process:
//
//   POST /cmd      {id, cmd, args} -> {id, ok, rev, result | error}       (fv_events.hpp)
//   GET  /events   server-sent events: every topic's snapshot, then each change
//   POST /input    [{type, ...}, ...] pointer, wheel and key events for the viewport
//
// The UI sends its JSON as text/plain, which a browser posts without a CORS preflight.
//

#include <atomic>
#include <functional>
#include <memory>

#include "fv_events.hpp"

namespace httplib {
class Server;
}

class FvProtocol
{
public:
  FvProtocol();

  fv::CommandRegistry& commands() { return m_commands; }
  fv::EventHub&        events() { return m_events; }

  // Receives each /input batch (a JSON array), on an HTTP thread. Unset: input is dropped.
  void setInputSink(std::function<void(const fv::Json& events)> sink) { m_input = std::move(sink); }

  // What this build can do beyond the protocol's commands ({dforceGpu: bool}, ...), carried by
  // `hello` and the "session" topic, so an app shows only what the renderer has.
  void setFeatures(fv::Json features);

  // Called by stop(): the server is about to join its threads, so whoever holds a request waiting on
  // the application thread must let it go. Set and called on the application thread.
  void setStopHook(std::function<void()> hook) { m_stop = std::move(hook); }
  void stop()
  {
    if(m_stop)
      m_stop();
  }

  // Adds the routes. `running` turns false when the server stops; open event streams end then.
  void registerRoutes(httplib::Server& http, const std::atomic<bool>& running);

private:
  fv::CommandRegistry                         m_commands;
  fv::EventHub                                m_events;
  std::function<void(const fv::Json& events)> m_input;
  std::function<void()>                       m_stop;
  fv::Json                                    m_features = fv::Json::object();
};
