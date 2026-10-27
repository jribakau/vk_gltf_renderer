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

// Native presentation's shared images: a slot is reused only once the consumer's fence shows it
// done, frames that find none free are dropped, fence values only rise, and the images only grow.

#include <gtest/gtest.h>

#include "present_slots.hpp"

TEST(PresentSlots, FreshSlotsAreAllFree)
{
  PresentSlots s(3);
  EXPECT_EQ(s.count(), 3);
  EXPECT_GE(s.acquire(0), 0);
}

TEST(PresentSlots, ASlotIsReusedOnlyOnceTheConsumerIsDone)
{
  PresentSlots s(3);
  const int    a = s.acquire(0);
  EXPECT_EQ(s.use(a), 1u);
  const int b = s.acquire(0);
  EXPECT_NE(b, a);
  EXPECT_EQ(s.use(b), 2u);
  const int c = s.acquire(0);
  EXPECT_NE(c, a);
  EXPECT_NE(c, b);
  EXPECT_EQ(s.use(c), 3u);
  // all three hold frames the consumer has not finished: the next frame is dropped
  EXPECT_EQ(s.acquire(0), -1);
  // the consumer finished the first frame: its slot comes back, and only it
  EXPECT_EQ(s.acquire(1), a);
  EXPECT_EQ(s.use(a), 4u);
  EXPECT_EQ(s.acquire(1), -1);
}

TEST(PresentSlots, TheLeastRecentlyUsedFreeSlotGoesFirst)
{
  PresentSlots s(3);
  const int    a = s.acquire(0);
  s.use(a);
  const int b = s.acquire(1);
  s.use(b);
  const int c = s.acquire(2);
  s.use(c);
  // everything done: the oldest, a, is next
  EXPECT_EQ(s.acquire(3), a);
}

TEST(PresentSlots, ValuesRiseAcrossAReset)
{
  PresentSlots s(2);
  s.use(s.acquire(0));
  s.use(s.acquire(0));
  EXPECT_EQ(s.acquire(0), -1);
  s.reset();  // the images were made again, both sides idle
  EXPECT_GE(s.acquire(0), 0);
  EXPECT_EQ(s.use(s.acquire(0)), 3u);
  EXPECT_EQ(s.value(), 3u);
}

TEST(PresentSlots, CapacityOnlyGrows)
{
  const PresentSize cap{1920, 1080};
  EXPECT_TRUE(fits(cap, {1920, 1080}));
  EXPECT_TRUE(fits(cap, {800, 600}));
  EXPECT_FALSE(fits(cap, {1921, 600}));
  EXPECT_EQ(growCapacity(cap, {800, 600}), cap);
  EXPECT_EQ(growCapacity(cap, {2560, 900}), (PresentSize{2560, 1080}));
  EXPECT_EQ(growCapacity({}, {640, 480}), (PresentSize{640, 480}));
}
