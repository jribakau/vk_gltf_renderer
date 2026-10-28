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

  size_t undoCount() const { return m_undo.size(); }
  size_t redoCount() const { return m_redo.size(); }
  // Topic "history": {undo: [label], redo: [label]}, the next one to undo (redo) first.
  nlohmann::json topic() const;

private:
  std::deque<JournalStep> m_undo, m_redo;
  bool                    m_sealed = false;  // an undo, a redo or a gesture's end since the last record: no merge
};

}  // namespace fv
