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

// Who watches a --stream run: without a local viewer a run behaves as it always did; with one it
// runs at the stream's rate and copies frames out only for a client or a waiting /frame.

#include <gtest/gtest.h>

#include "present_policy.hpp"

using present_policy::Viewers;

TEST(PresentPolicy, WithoutALocalViewerNothingChanges)
{
  const Viewers nobody{};
  const Viewers client{.clients = 1};
  EXPECT_EQ(present_policy::pacedFps(60, nobody), 10);
  EXPECT_EQ(present_policy::pacedFps(60, client), 60);
  EXPECT_FALSE(present_policy::refining(true, nobody));
  EXPECT_TRUE(present_policy::refining(true, client));
  // every due frame is copied out, watched or not
  EXPECT_TRUE(present_policy::copyOut(true, nobody));
  EXPECT_TRUE(present_policy::copyOut(true, client));
  EXPECT_FALSE(present_policy::copyOut(false, client));
  EXPECT_FALSE(present_policy::frameWaits(nobody));
}

TEST(PresentPolicy, ALocalViewerRunsAtTheStreamRate)
{
  const Viewers local{.localViewer = true};
  EXPECT_EQ(present_policy::pacedFps(60, local), 60);
  EXPECT_EQ(present_policy::pacedFps(5, local), 5);  // a cap below 10 stays
  EXPECT_TRUE(present_policy::refining(true, local));
  EXPECT_FALSE(present_policy::refining(false, local));
}

TEST(PresentPolicy, ALocalViewerAloneCopiesNothing)
{
  const Viewers local{.localViewer = true};
  EXPECT_FALSE(present_policy::copyOut(true, local));
  EXPECT_TRUE(present_policy::frameWaits(local));
  // a /frame waiting for one, or a stream client, has frames copied again
  EXPECT_TRUE(present_policy::copyOut(true, {.localViewer = true, .frameWanted = true}));
  EXPECT_TRUE(present_policy::copyOut(true, {.clients = 1, .localViewer = true}));
  EXPECT_FALSE(present_policy::copyOut(false, {.localViewer = true, .frameWanted = true}));
  EXPECT_FALSE(present_policy::frameWaits({.clients = 1, .localViewer = true}));
}

TEST(PresentPolicy, ZeroCapIsOneFrameASecond)
{
  EXPECT_EQ(present_policy::pacedFps(0, {}), 1);
  EXPECT_EQ(present_policy::pacedFps(0, {.localViewer = true}), 1);
}
