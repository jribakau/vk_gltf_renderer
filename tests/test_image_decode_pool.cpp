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

// Textures decoded away from the render thread (rs::ImageDecodePool): what a take hands over, and
// what a cancel lets go of, whether its decode waited, ran or was done.

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "image_decode_pool.hpp"

using namespace rs;

namespace {

// A decode whose levels are as many bytes as its source's width says; one named "boom" throws as a
// system memory run out would, and "unreadable" can't be read. One named "gate" waits until opened.
struct Decoder
{
  std::promise<void>       open;
  std::shared_future<void> opened = open.get_future().share();
  std::promise<void>       started;  // the "gate" one is running
  std::mutex               mutex;
  std::map<std::string, int> calls;  // by name
  std::vector<unsigned char> bytesSeen;

  ImageDecodePool::Decode decode()
  {
    return [this](const ImageSource& source, nvvkgltf::LoadedImageData& out, bool& prefetched) {
      {
        std::lock_guard lock(mutex);
        calls[source.name]++;
        bytesSeen.assign(source.bytes.begin(), source.bytes.end());
      }
      if(source.name == "gate")
      {
        started.set_value();
        opened.wait();
      }
      if(source.name == "boom")
        throw std::bad_alloc();
      if(source.name == "unreadable")
        return false;
      prefetched = source.name == "prefetched";
      out.mipData.assign(1, std::vector<char>(size_t(std::max(source.width, 0))));
      return true;
    };
  }
  int callsOf(const std::string& name)
  {
    std::lock_guard lock(mutex);
    return calls[name];
  }
};

HeldImageSource source(const std::string& name, int bytes = 16)
{
  return HeldImageSource({.uri = name, .width = bytes, .height = 1, .name = name});
}

}  // namespace

TEST(ImageDecodePool, ADecodeDoneIsHandedOverOnceWithItsLevels)
{
  Decoder         d;
  ImageDecodePool pool(d.decode(), 2);
  const uint64_t  a = pool.start(source("a", 100));
  const uint64_t  b = pool.start(source("prefetched", 50));
  EXPECT_NE(a, 0u);
  EXPECT_NE(a, b);
  pool.waitIdle();
  EXPECT_TRUE(pool.ready());
  EXPECT_EQ(pool.pending(), 2u);
  std::vector<ImageDecodePool::Done> done = pool.take(1 << 20);
  ASSERT_EQ(done.size(), 2u);
  EXPECT_EQ(done[0].ticket, a);  // by ticket
  EXPECT_TRUE(done[0].ok);
  EXPECT_EQ(done[0].bytes, 100u);
  EXPECT_EQ(done[0].source.source.name, "a");
  EXPECT_FALSE(done[0].prefetched);
  EXPECT_TRUE(done[1].prefetched);
  EXPECT_FALSE(pool.ready());
  EXPECT_EQ(pool.pending(), 0u);
  EXPECT_TRUE(pool.take(1 << 20).empty());
}

// as a sync stages so much and no more (ResidencyOptions::decodeBatchBytes)
TEST(ImageDecodePool, ATakeStopsAtItsBudgetTheFirstWhateverItsSize)
{
  Decoder         d;
  ImageDecodePool pool(d.decode(), 1);
  pool.start(source("a", 100));
  pool.start(source("b", 100));
  pool.start(source("c", 100));
  pool.waitIdle();
  EXPECT_EQ(pool.take(150).size(), 1u);
  EXPECT_EQ(pool.take(1).size(), 1u);  // more than the budget: the first comes all the same
  EXPECT_EQ(pool.take(1000).size(), 1u);
}

TEST(ImageDecodePool, OneCancelledWhileItWaitsIsNeverDecoded)
{
  Decoder         d;
  ImageDecodePool pool(d.decode(), 1);
  pool.start(source("gate"));
  const uint64_t waiting = pool.start(source("b"));
  d.started.get_future().wait();  // the one thread is busy: b waits
  pool.cancel(waiting);
  d.open.set_value();
  pool.waitIdle();
  EXPECT_EQ(d.callsOf("b"), 0);
  const std::vector<ImageDecodePool::Done> done = pool.take(1 << 20);
  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].source.source.name, "gate");
}

TEST(ImageDecodePool, OneCancelledWhileItRunsFinishesAndIsLetGoOf)
{
  Decoder         d;
  ImageDecodePool pool(d.decode(), 1);
  const uint64_t  running = pool.start(source("gate"));
  d.started.get_future().wait();
  pool.cancel(running);
  EXPECT_EQ(pool.pending(), 0u);
  d.open.set_value();
  pool.waitIdle();
  EXPECT_FALSE(pool.ready());
  EXPECT_TRUE(pool.take(1 << 20).empty());
}

TEST(ImageDecodePool, OneCancelledOnceDoneIsLetGoOf)
{
  Decoder         d;
  ImageDecodePool pool(d.decode(), 1);
  const uint64_t  a = pool.start(source("a"));
  pool.waitIdle();
  ASSERT_TRUE(pool.ready());
  pool.cancel(a);
  EXPECT_FALSE(pool.ready());
  EXPECT_EQ(pool.pending(), 0u);
  EXPECT_TRUE(pool.take(1 << 20).empty());
  pool.cancel(a);  // twice, or one never begun: nothing
  pool.cancel(12345);
}

TEST(ImageDecodePool, OneThatCantBeReadOrThrowsIsHandedOverWithNothing)
{
  Decoder         d;
  ImageDecodePool pool(d.decode(), 2);
  pool.start(source("unreadable"));
  pool.start(source("boom"));
  pool.waitIdle();
  const std::vector<ImageDecodePool::Done> done = pool.take(1 << 20);
  ASSERT_EQ(done.size(), 2u);
  for(const ImageDecodePool::Done& one : done)
  {
    EXPECT_FALSE(one.ok) << one.source.source.name;
    EXPECT_TRUE(one.loaded.mipData.empty());
    EXPECT_EQ(one.bytes, 0u);
  }
}

// what the record held may go once the decode is begun: a scene replace, an item taken off
TEST(ImageDecodePool, ASourceHoldsItsOwnBytes)
{
  Decoder d;
  std::vector<unsigned char> record = {1, 2, 3, 4};
  ImageSource                from{.bytes = record, .width = 4, .height = 1, .name = "embedded"};
  HeldImageSource            held(from);
  record.assign(4, 9);
  ImageDecodePool pool(d.decode(), 1);
  pool.start(std::move(held));
  pool.waitIdle();
  EXPECT_EQ(d.bytesSeen, (std::vector<unsigned char>{1, 2, 3, 4}));
}

TEST(ImageDecodePool, ItsEndLetsWhatWaitsGoAndJoinsWhatRuns)
{
  Decoder     d;
  std::thread opener;
  {
    ImageDecodePool pool(d.decode(), 1);
    pool.start(source("gate"));
    pool.start(source("b"));
    d.started.get_future().wait();
    opener = std::thread([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      d.open.set_value();
    });
  }  // waits for the gate, never decodes b
  opener.join();
  EXPECT_EQ(d.callsOf("gate"), 1);
  EXPECT_EQ(d.callsOf("b"), 0);
}
