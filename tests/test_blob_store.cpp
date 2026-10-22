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

// Binary payloads a host posts for a live edit: taken once by id, the oldest dropped past the cap.

#include <gtest/gtest.h>

#include "blob_store.hpp"

TEST(BlobStore, TakenOnceById)
{
  BlobStore      store;
  const uint64_t a = store.put("first");
  const uint64_t b = store.put("second");
  EXPECT_NE(a, b);
  EXPECT_EQ(store.take(b).value_or(""), "second");
  EXPECT_FALSE(store.take(b).has_value());
  EXPECT_EQ(store.take(a).value_or(""), "first");
}

TEST(BlobStore, UnknownIdIsEmpty)
{
  BlobStore store;
  EXPECT_FALSE(store.take(42).has_value());
}

TEST(BlobStore, OldestDroppedPastTheCap)
{
  BlobStore         store(100);
  const std::string half(51, 'x');
  const uint64_t    a = store.put(half);
  const uint64_t    b = store.put(half);
  EXPECT_FALSE(store.take(a).has_value());
  EXPECT_TRUE(store.take(b).has_value());
}

TEST(BlobStore, OneOversizedBlobIsKept)
{
  BlobStore      store(100);
  const uint64_t a = store.put(std::string(101, 'x'));
  EXPECT_TRUE(store.take(a).has_value());
}
