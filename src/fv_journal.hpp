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
// The renderer's edit history (daz-viewer docs/command-journal-plan.md, C1): each step holds the ops
// that redo it and the ops that undo it, both captured from the host's own state as the edit
// applied, so undoing puts back exactly what was there. An op is a protocol command as the wire has
// it, {cmd, args}. Pure: the host applies the ops (fv_host_journal.cpp), this keeps the steps.
//
// A step merges into the one before it when their keys match within kMerge and nothing was undone
// or redone between them: the redo ops become the later step's, the undo ops stay the first's (a
// key names what the step touches, so both read the same things). A merge that ends where it began
// leaves no step. Recording a step drops what could be redone.
//
// A gesture (a viewport drag from press to release, a slider's drag) is one step however long it
// takes: its steps carry its id in their key and merge whatever the time between, and its last one
// closes it.
//
// A group (a scene file opening: a clear, loads, restores, places) is one step under its own label,
// whatever its steps touch: from begin to end their redo ops add up in the order they applied, and
// their undo ops the other way round.
//
// Undoing a step reads, as each op applies, what it is about to change: that is the step's redo from
// then on (rewrite), and redoing it reads its undo the same way. A structural step needs that: a
// load's redo is the prop its undo took away, as it was then.
//

#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>

#include <tinygltf/json.hpp>  // nlohmann::json

namespace fv {

struct JournalStep
{
  nlohmann::json redo = nlohmann::json::array();  // ops, in order
  nlohmann::json undo = nlohmann::json::array();  // ops, in order
  std::string    label;
  std::string    key;  // empty: never merges
  std::chrono::steady_clock::time_point at;
  bool gesture = false;  // merges with its key whatever the time between
  bool closes  = false;  // a gesture's last: nothing merges into it after
  uint64_t serial = 0;   // given by record()
};

class Journal
{
public:
  static constexpr size_t                    kLimit = 200;
  static constexpr std::chrono::milliseconds kMerge{800};

  void record(JournalStep step);
  // The step to undo (redo), moved to the other side; none when there is nothing to.
  std::optional<JournalStep> takeUndo();
  std::optional<JournalStep> takeRedo();
  void                       clear();
  // The steps recorded from now until end() are one, labelled `label`
  void begin(std::string label);
  void end();
  bool grouping() const { return m_group.has_value(); }
  // The step `serial`'s ops one way (`undo`, else redo), as an undo or redo read them; nothing when
  // the step is gone meanwhile.
  void rewrite(uint64_t serial, bool undo, nlohmann::json ops);
  // Every op of every step, both ways
  template <class F>
  void forEachOp(F&& f) const
  {
    for(const auto* side : {&m_undo, &m_redo})
      for(const JournalStep& s : *side)
        for(const auto* ops : {&s.undo, &s.redo})
          for(const nlohmann::json& o : *ops)
            f(o);
  }

  size_t undoCount() const { return m_undo.size(); }
  size_t redoCount() const { return m_redo.size(); }
  // Topic "history": {undo: [label], redo: [label]}, the next one to undo (redo) first.
  nlohmann::json topic() const;

private:
  std::deque<JournalStep> m_undo, m_redo;
  bool                    m_sealed = false;  // an undo, a redo or a gesture's end since the last record: no merge
  uint64_t                m_serial = 0;
  struct Group
  {
    std::string label;
    uint64_t    serial = 0;  // its step, once one is recorded
  };
  std::optional<Group> m_group;
};

}  // namespace fv
