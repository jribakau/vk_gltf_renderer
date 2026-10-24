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

// Figure Viewer host protocol without the server: commands answer by name, and a subscriber sees
// every topic's state first, then each change, and catches up from snapshots when it falls behind.

#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>
#include <thread>

#include "fv_events.hpp"

using namespace std::chrono_literals;
using fv::Json;

TEST(FvCommands, DispatchesByName)
{
  fv::CommandRegistry commands;
  commands.add("double", [](const Json& args) { return Json(args.at("x").get<int>() * 2); });
  const Json answer = commands.dispatch({{"id", 7}, {"cmd", "double"}, {"args", {{"x", 21}}}});
  EXPECT_EQ(answer["id"], 7);
  EXPECT_TRUE(answer["ok"].get<bool>());
  EXPECT_EQ(answer["result"], 42);
}

TEST(FvCommands, FailuresAnswerWithAnError)
{
  fv::CommandRegistry commands;
  commands.add("throws", [](const Json&) -> Json { throw std::runtime_error("no figure loaded"); });

  const Json unknown = commands.dispatch({{"id", 1}, {"cmd", "missing"}});
  EXPECT_FALSE(unknown["ok"].get<bool>());
  EXPECT_EQ(unknown["id"], 1);

  const Json thrown = commands.dispatch({{"id", 2}, {"cmd", "throws"}});
  EXPECT_FALSE(thrown["ok"].get<bool>());
  EXPECT_EQ(thrown["error"], "no figure loaded");

  EXPECT_FALSE(commands.dispatch(Json::array())["ok"].get<bool>());
  EXPECT_FALSE(commands.dispatch({{"cmd", 3}})["ok"].get<bool>());
}

TEST(FvEvents, PatchesMergeIntoTheSnapshot)
{
  fv::EventHub hub;
  hub.snapshot("pose/fig:1", {{"hip/rx", 0.0}, {"hip/ry", 0.0}});
  hub.patch("pose/fig:1", {{"hip/rx", 12.5}});
  // a key set to null in a merge patch is removed
  hub.patch("pose/fig:1", {{"hip/ry", nullptr}});

  uint64_t upTo = 0;
  auto     all  = hub.snapshots(&upTo);
  ASSERT_EQ(all.size(), 1u);
  EXPECT_EQ(upTo, 3u);
  EXPECT_EQ(all[0].data, (Json{{"hip/rx", 12.5}}));
}

TEST(FvEvents, SubscriberGetsChangesAfterItsRev)
{
  fv::EventHub hub;
  hub.snapshot("scene", Json::object());
  uint64_t rev = hub.rev();
  hub.patch("scene", {{"fig:1", {{"label", "Genesis 9"}}}});
  hub.patch("selection", {{"id", "fig:1"}});

  auto events = hub.next(rev, 0ms);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].topic, "scene");
  EXPECT_EQ(events[0].kind, "patch");
  EXPECT_EQ(events[1].topic, "selection");
  EXPECT_EQ(rev, hub.rev());
  EXPECT_TRUE(hub.next(rev, 0ms).empty());
}

TEST(FvEvents, WaitsForTheNextEvent)
{
  fv::EventHub hub;
  uint64_t     rev = hub.rev();
  std::thread  producer([&] {
    std::this_thread::sleep_for(20ms);
    hub.patch("stats", {{"fps", 60}});
  });
  auto events = hub.next(rev, 2s);
  producer.join();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].data["fps"], 60);
}

TEST(FvEvents, LaggingSubscriberResyncsFromSnapshots)
{
  fv::EventHub hub(/*keep=*/4);
  hub.snapshot("morphs/fig:1", Json::object());
  uint64_t rev = hub.rev();
  for(int i = 0; i < 10; i++)
    hub.patch("morphs/fig:1", {{"fig:1/morph/FBMHeavy", i}});

  auto events = hub.next(rev, 0ms);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].kind, "reset");
  EXPECT_EQ(events[1].kind, "snapshot");
  EXPECT_EQ(events[1].data["fig:1/morph/FBMHeavy"], 9);
  EXPECT_EQ(rev, hub.rev());
}

// A prop that leaves the scene takes its topics along: subscribers get a drop, later ones never see it.
TEST(FvEvents, DroppedTopicsLeave)
{
  fv::EventHub hub;
  hub.snapshot("morphs/p:1", {{"channels", Json::object()}});
  hub.snapshot("scene", Json::object());
  uint64_t rev = hub.rev();
  EXPECT_GT(hub.drop("morphs/p:1"), 0u);
  EXPECT_EQ(hub.drop("morphs/p:1"), 0u);  // already gone

  auto events = hub.next(rev, 0ms);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, "drop");
  EXPECT_EQ(events[0].topic, "morphs/p:1");

  uint64_t upTo = 0;
  auto     all  = hub.snapshots(&upTo);
  ASSERT_EQ(all.size(), 1u);
  EXPECT_EQ(all[0].topic, "scene");
}

TEST(FvEvents, CloseWakesWaiters)
{
  fv::EventHub hub;
  uint64_t     rev = hub.rev();
  std::thread  closer([&] {
    std::this_thread::sleep_for(20ms);
    hub.close();
  });
  const auto start  = std::chrono::steady_clock::now();
  auto       events = hub.next(rev, 10s);
  closer.join();
  EXPECT_TRUE(events.empty());
  EXPECT_LT(std::chrono::steady_clock::now() - start, 5s);
}

TEST(FvEvents, SseMessageIsOneDataLine)
{
  const std::string msg = fv::EventHub::sse({5, "notices", "patch", {{"text", "line one\nline two"}}});
  EXPECT_EQ(msg.rfind("id: 5\ndata: ", 0), 0u);
  // the payload's newline is escaped, so only the framing has raw newlines
  EXPECT_EQ(msg.find('\n', std::string("id: 5\ndata: ").size()), msg.size() - 2);
  EXPECT_EQ(msg.substr(msg.size() - 2), "\n\n");
}
