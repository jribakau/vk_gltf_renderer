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
// Figure Viewer host protocol, the part without HTTP (fv_protocol.cpp serves it).
//
// Commands: the UI posts {id, cmd, args}; a handler registered under `cmd` answers with a result,
// or throws, and the UI gets {id, ok:true, rev, result} or {id, ok:false, error}.
//
// Events: the state the panels show is kept as one snapshot per topic ("scene", "pose/fig:1",
// ...). A subscriber first gets every snapshot, then each change as it happens. A change is either
// a whole new snapshot or a patch: a JSON merge patch (RFC 7386) on the topic's snapshot, which the
// hub also applies to the snapshot it keeps, so a later subscriber starts from the current state.
// Merge patches replace arrays whole, so snapshots are objects keyed by id, not lists.
//
// A topic can also be dropped (a prop that left the scene takes its panels' topics with it): a
// "drop" event, after which the topic has no state and later subscribers don't see it.
//
// Every event carries a sequence number (rev) increasing across topics. A command's answer carries
// the rev current when it finished, so the UI can tell which events already include its effect.
//

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <tinygltf/json.hpp>  // nlohmann::json

namespace fv {

using Json = nlohmann::json;

class CommandRegistry
{
public:
  using Handler = std::function<Json(const Json& args)>;

  void add(const std::string& name, Handler handler);
  // {id, ok:true, result} or {id, ok:false, error}. An unknown command, a request that isn't an
  // object with a string `cmd`, or a handler that throws, all answer ok:false.
  Json dispatch(const Json& request) const;
  std::vector<std::string> names() const;

private:
  mutable std::mutex             m_mutex;
  std::map<std::string, Handler> m_handlers;
};

struct Event
{
  uint64_t    rev{};
  std::string topic;
  std::string kind;  // "snapshot", "patch", "drop", or "reset" before a lagging subscriber's resync
  Json        data;
};

class EventHub
{
public:
  explicit EventHub(size_t keep = 1024);

  // The topic's whole state. Returns the event's rev.
  uint64_t snapshot(const std::string& topic, Json data);
  // A merge patch on the topic's state (an empty snapshot if it has none yet). Returns the rev.
  uint64_t patch(const std::string& topic, const Json& data);
  // The topic goes, state and all. Returns the rev, or 0 when there was no such topic.
  uint64_t drop(const std::string& topic);

  // The latest rev published (0 before any).
  uint64_t rev() const;
  // Every topic's current state as snapshot events, and the rev they are current up to.
  std::vector<Event> snapshots(uint64_t* upTo) const;
  // The events after `rev`, waiting up to `timeout` for one. A subscriber so far behind that the
  // events it missed are gone gets a "reset" event (no topic: forget every topic) and every
  // snapshot instead, and `rev` moves to their rev. Empty
  // when nothing happened in time, or once closed.
  std::vector<Event> next(uint64_t& rev, std::chrono::milliseconds timeout);
  // Wakes every waiting subscriber and makes `next` return at once from now on.
  void close();
  bool closed() const;

  // One event as a server-sent-events message.
  static std::string sse(const Event& event);

private:
  uint64_t publish(const std::string& topic, const char* kind, Json data);

  mutable std::mutex          m_mutex;
  std::condition_variable     m_cv;
  std::map<std::string, Json> m_state;
  std::deque<Event>           m_events;
  size_t                      m_keep;
  uint64_t                    m_rev{0};
  bool                        m_closed{false};
};

}  // namespace fv
