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

// fv::Journal's steps (daz-viewer docs/command-journal-plan.md, C1): undo and redo move them across,
// a step of the same key soon after merges, and the history keeps kLimit steps.

#include <gtest/gtest.h>

#include "fv_journal.hpp"

using nlohmann::json;
using namespace std::chrono_literals;

namespace {

const auto t0 = std::chrono::steady_clock::time_point{} + 1h;

// a dial's step: from `from` to `to`
fv::JournalStep dial(double from, double to, std::chrono::steady_clock::duration at, const std::string& key = "Taper")
{
  const auto set = [](double v) { return json::array({{{"cmd", "morph.set"}, {"args", {{"values", {{"Taper", v}}}}}}}); };
  return {.redo = set(to), .undo = set(from), .label = "Morph Taper", .key = key, .at = t0 + at};
}

double valueOf(const json& ops)
{
  return ops[0]["args"]["values"]["Taper"].get<double>();
}

}  // namespace

TEST(FvJournal, UndoAndRedoMoveTheStepAcross)
{
  fv::Journal j;
  j.record(dial(0, 0.5, 0s, ""));
  j.record(dial(0.5, 1, 1s, ""));
  EXPECT_EQ(j.topic(), json({{"undo", {"Morph Taper", "Morph Taper"}}, {"redo", json::array()}}));

  auto step = j.takeUndo();
  ASSERT_TRUE(step);
  EXPECT_EQ(valueOf(step->undo), 0.5);
  EXPECT_EQ(j.undoCount(), 1u);
  EXPECT_EQ(j.redoCount(), 1u);
  step = j.takeRedo();
  ASSERT_TRUE(step);
  EXPECT_EQ(valueOf(step->redo), 1);
  EXPECT_FALSE(j.takeRedo());

  // a new step drops what could be redone
  j.takeUndo();
  j.record(dial(0.5, 0.2, 2s, ""));
  EXPECT_EQ(j.redoCount(), 0u);
  EXPECT_EQ(j.undoCount(), 2u);
}

TEST(FvJournal, AStepOfTheSameKeySoonAfterMerges)
{
  fv::Journal j;
  j.record(dial(0, 0.2, 0ms));
  j.record(dial(0.2, 0.4, 500ms));
  j.record(dial(0.4, 0.6, 1000ms));  // within kMerge of the last, not of the first
  ASSERT_EQ(j.undoCount(), 1u);
  auto step = j.takeUndo();
  EXPECT_EQ(valueOf(step->undo), 0);
  EXPECT_EQ(valueOf(step->redo), 0.6);
}

TEST(FvJournal, ADifferentKeyOrAPauseOrAnUndoBetweenKeepsStepsApart)
{
  fv::Journal j;
  j.record(dial(0, 0.2, 0ms));
  j.record(dial(0.2, 0.4, 100ms, "Stretch"));
  j.record(dial(0.4, 0.6, 100ms + fv::Journal::kMerge + 1ms, "Stretch"));
  EXPECT_EQ(j.undoCount(), 3u);

  // undone and redone: the next edit is a step of its own
  j.takeUndo();
  j.takeRedo();
  j.record(dial(0.6, 0.8, 100ms + fv::Journal::kMerge + 2ms, "Stretch"));
  EXPECT_EQ(j.undoCount(), 4u);
}

TEST(FvJournal, AMergeThatEndsWhereItBeganLeavesNoStep)
{
  fv::Journal j;
  j.record(dial(0, 0.5, 0ms, ""));
  j.record(dial(0.5, 0.7, 1s));
  j.record(dial(0.7, 0.5, 1100ms));
  EXPECT_EQ(j.undoCount(), 1u);
}

TEST(FvJournal, KeepsTheLatestStepsUpToItsLimit)
{
  fv::Journal j;
  for(size_t i = 0; i < fv::Journal::kLimit + 5; i++)
    j.record(dial(double(i), double(i + 1), std::chrono::seconds(i), ""));
  EXPECT_EQ(j.undoCount(), fv::Journal::kLimit);
  std::optional<fv::JournalStep> last;
  while(auto s = j.takeUndo())
    last = s;
  EXPECT_EQ(valueOf(last->undo), 5);
}

TEST(FvJournal, AGestureIsOneStepWhateverTheTimeUntilItsEnd)
{
  fv::Journal j;
  fv::JournalStep a = dial(0, 0.2, 0s, "g1"), b = dial(0.2, 0.4, 5s, "g1"), c = dial(0.4, 0.6, 9s, "g1"), d = dial(0.6, 0.8, 9100ms, "g1");
  a.gesture = b.gesture = c.gesture = d.gesture = true;
  c.closes  = true;
  j.record(a);
  j.record(b);
  j.record(c);
  ASSERT_EQ(j.undoCount(), 1u);
  // closed: the same key starts a step of its own
  j.record(d);
  ASSERT_EQ(j.undoCount(), 2u);
  EXPECT_EQ(valueOf(j.takeUndo()->undo), 0.6);
  const auto first = j.takeUndo();
  EXPECT_EQ(valueOf(first->undo), 0);
  EXPECT_EQ(valueOf(first->redo), 0.6);
}

TEST(FvJournal, AGroupIsOneStepOfEverythingRecordedInIt)
{
  fv::Journal j;
  j.record(dial(0, 0.2, 0s));
  const auto op = [](const char* cmd) { return json::array({{{"cmd", cmd}, {"args", json::object()}}}); };
  const auto step = [&](const char* redo, const char* undo, std::chrono::steady_clock::duration at) {
    return fv::JournalStep{.redo = op(redo), .undo = op(undo), .label = redo, .key = redo, .at = t0 + at};
  };
  j.begin("Open scene");
  ASSERT_TRUE(j.grouping());
  j.record(step("a", "-a", 1s));
  j.record(step("b", "-b", 1s));
  j.record(step("b", "-b2", 10s));  // the same key: still added, not merged
  j.end();
  j.record(dial(0.2, 0.4, 10s));  // after the end: a step of its own, its key or not
  EXPECT_EQ(j.topic()["undo"], json({"Morph Taper", "Open scene", "Morph Taper"}));

  j.takeUndo();
  const auto group = j.takeUndo();
  ASSERT_TRUE(group);
  const auto cmds = [](const json& ops) {
    std::vector<std::string> out;
    for(const json& o : ops)
      out.push_back(o["cmd"].get<std::string>());
    return out;
  };
  EXPECT_EQ(cmds(group->redo), (std::vector<std::string>{"a", "b", "b"}));
  EXPECT_EQ(cmds(group->undo), (std::vector<std::string>{"-b2", "-b", "-a"}));
  // it doesn't merge into the step before it
  EXPECT_EQ(j.undoCount(), 1u);
}
