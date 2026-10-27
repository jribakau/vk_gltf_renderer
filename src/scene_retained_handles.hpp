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
// The retained scene's store handles (docs: daz-viewer docs/retained-scene-plan.md, "Identity"): an
// index into a slot map and the generation the slot had when the handle was made. Freeing a slot
// bumps its generation, so a handle kept past its object's life names nothing (valid() is false)
// rather than whatever took the slot next, and an index never comes back with a generation it had
// before. Handles never reach the wire (the protocol keeps its prop-local ids) nor the GPU (that is
// the residency layer's slots).
//

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace rs {

template <class Tag>
struct Handle
{
  static constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

  uint32_t index = kNone;
  uint32_t gen   = 0;

  bool isNone() const { return index == kNone; }
  bool operator==(const Handle&) const = default;
  auto operator<=>(const Handle&) const = default;
};

// Objects of one kind under generational handles. Erase frees the slot for a later insert, under a
// new generation; iteration is in slot order, which is not insertion order once slots are reused.
template <class T, class Tag>
class SlotMap
{
public:
  using Id = Handle<Tag>;

  Id insert(T value)
  {
    uint32_t index;
    if(!m_free.empty())
    {
      index = m_free.back();
      m_free.pop_back();
    }
    else
    {
      index = static_cast<uint32_t>(m_slots.size());
      m_slots.emplace_back();
    }
    Slot& s = m_slots[index];
    s.value = std::move(value);
    m_live++;
    return {index, s.gen};
  }

  bool erase(Id id)
  {
    if(!contains(id))
      return false;
    Slot& s = m_slots[id.index];
    s.value.reset();
    s.gen++;
    m_free.push_back(id.index);
    m_live--;
    return true;
  }

  bool contains(Id id) const { return id.index < m_slots.size() && m_slots[id.index].gen == id.gen && m_slots[id.index].value.has_value(); }

  T* get(Id id) { return contains(id) ? &*m_slots[id.index].value : nullptr; }
  const T* get(Id id) const { return contains(id) ? &*m_slots[id.index].value : nullptr; }

  T& operator[](Id id)
  {
    assert(contains(id) && "stale or empty handle");
    return *m_slots[id.index].value;
  }
  const T& operator[](Id id) const
  {
    assert(contains(id) && "stale or empty handle");
    return *m_slots[id.index].value;
  }

  size_t size() const { return m_live; }
  bool   empty() const { return m_live == 0; }
  // One past the highest slot ever used: what a dense side array indexed by slot needs.
  size_t capacity() const { return m_slots.size(); }

  template <class F>
  void forEach(F&& f)
  {
    for(uint32_t i = 0; i < m_slots.size(); i++)
      if(m_slots[i].value)
        f(Id{i, m_slots[i].gen}, *m_slots[i].value);
  }
  template <class F>
  void forEach(F&& f) const
  {
    for(uint32_t i = 0; i < m_slots.size(); i++)
      if(m_slots[i].value)
        f(Id{i, m_slots[i].gen}, *m_slots[i].value);
  }

private:
  struct Slot
  {
    std::optional<T> value;
    uint32_t         gen = 0;
  };
  std::vector<Slot>     m_slots;
  std::vector<uint32_t> m_free;
  size_t                m_live = 0;
};

}  // namespace rs

template <class Tag>
struct std::hash<rs::Handle<Tag>>
{
  size_t operator()(const rs::Handle<Tag>& h) const noexcept { return (size_t(h.gen) << 32) ^ size_t(h.index); }
};
