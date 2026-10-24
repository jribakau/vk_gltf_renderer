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

#include "fv_events.hpp"

#include <exception>

namespace fv {

void CommandRegistry::add(const std::string& name, Handler handler)
{
  std::lock_guard lock(m_mutex);
  m_handlers[name] = std::move(handler);
}

std::vector<std::string> CommandRegistry::names() const
{
  std::lock_guard          lock(m_mutex);
  std::vector<std::string> out;
  for(const auto& [name, handler] : m_handlers)
    out.push_back(name);
  return out;
}

Json CommandRegistry::dispatch(const Json& request) const
{
  Json out = Json::object();
  if(request.is_object() && request.contains("id"))
    out["id"] = request["id"];
  auto fail = [&out](const std::string& error) {
    out["ok"]    = false;
    out["error"] = error;
    return out;
  };
  if(!request.is_object() || !request.contains("cmd") || !request["cmd"].is_string())
    return fail("a command is an object with a string 'cmd'");
  const std::string name = request["cmd"].get<std::string>();
  Handler           handler;
  {
    std::lock_guard lock(m_mutex);
    auto            it = m_handlers.find(name);
    if(it == m_handlers.end())
      return fail("unknown command '" + name + "'");
    handler = it->second;
  }
  const Json args = request.contains("args") ? request["args"] : Json::object();
  try
  {
    out["result"] = handler(args);
    out["ok"]     = true;
  }
  catch(const std::exception& e)
  {
    return fail(e.what());
  }
  return out;
}

//--------------------------------------------------------------------------------------------------

EventHub::EventHub(size_t keep)
    : m_keep(keep < 1 ? 1 : keep)
{
}

uint64_t EventHub::snapshot(const std::string& topic, Json data)
{
  return publish(topic, "snapshot", std::move(data));
}

uint64_t EventHub::patch(const std::string& topic, const Json& data)
{
  return publish(topic, "patch", data);
}

uint64_t EventHub::drop(const std::string& topic)
{
  uint64_t rev = 0;
  {
    std::lock_guard lock(m_mutex);
    if(!m_state.erase(topic))
      return 0;
    rev = ++m_rev;
    m_events.push_back({rev, topic, "drop", nullptr});
    while(m_events.size() > m_keep)
      m_events.pop_front();
  }
  m_cv.notify_all();
  return rev;
}

uint64_t EventHub::publish(const std::string& topic, const char* kind, Json data)
{
  uint64_t rev = 0;
  {
    std::lock_guard lock(m_mutex);
    if(kind[0] == 's')
      m_state[topic] = data;
    else
    {
      Json& state = m_state[topic];
      if(!state.is_object())
        state = Json::object();
      state.merge_patch(data);
    }
    rev = ++m_rev;
    m_events.push_back({rev, topic, kind, std::move(data)});
    while(m_events.size() > m_keep)
      m_events.pop_front();
  }
  m_cv.notify_all();
  return rev;
}

uint64_t EventHub::rev() const
{
  std::lock_guard lock(m_mutex);
  return m_rev;
}

std::vector<Event> EventHub::snapshots(uint64_t* upTo) const
{
  std::lock_guard    lock(m_mutex);
  std::vector<Event> out;
  for(const auto& [topic, data] : m_state)
    out.push_back({m_rev, topic, "snapshot", data});
  if(upTo)
    *upTo = m_rev;
  return out;
}

std::vector<Event> EventHub::next(uint64_t& rev, std::chrono::milliseconds timeout)
{
  std::unique_lock lock(m_mutex);
  m_cv.wait_for(lock, timeout, [&] { return m_closed || m_rev > rev; });
  std::vector<Event> out;
  if(m_closed || m_rev <= rev)
    return out;
  // the oldest event kept is m_events.front(); the subscriber needs rev + 1
  if(m_events.empty() || m_events.front().rev > rev + 1)
  {
    // what it holds may include topics dropped since: it starts over
    out.push_back({m_rev, "", "reset", nullptr});
    for(const auto& [topic, data] : m_state)
      out.push_back({m_rev, topic, "snapshot", data});
    rev = m_rev;
    return out;
  }
  for(const Event& e : m_events)
    if(e.rev > rev)
      out.push_back(e);
  rev = m_rev;
  return out;
}

void EventHub::close()
{
  {
    std::lock_guard lock(m_mutex);
    m_closed = true;
  }
  m_cv.notify_all();
}

bool EventHub::closed() const
{
  std::lock_guard lock(m_mutex);
  return m_closed;
}

std::string EventHub::sse(const Event& event)
{
  const Json body = {{"rev", event.rev}, {"topic", event.topic}, {"kind", event.kind}, {"data", event.data}};
  // dump() escapes control characters, so the payload is one line as SSE requires
  return "id: " + std::to_string(event.rev) + "\ndata: " + body.dump() + "\n\n";
}

}  // namespace fv
