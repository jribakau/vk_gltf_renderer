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

// VariantJobs (D15) with a fake compiler: a job is a snapshot, built on a worker, taken once

#include <gtest/gtest.h>

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "variant_compile_job.hpp"

namespace {

// What the renderer's live state would be: settings and the scene's features, changed and torn down
// while a variant builds
struct FakeResources
{
  bool                     wireframe = false;
  std::vector<std::string> features;
};

// The snapshot: values only
struct FakeJob
{
  int                                              key = 0;
  std::vector<std::pair<std::string, std::string>> macros;
};

FakeJob snapshot(const FakeResources& resources, int key)
{
  FakeJob job{.key = key};
  job.macros.push_back({"WIREFRAME", resources.wireframe ? "1" : "0"});
  for(const std::string& f : resources.features)
    job.macros.push_back({f, "1"});
  return job;
}

struct FakeBuilt
{
  std::string     source;  // "compiled" from the job's macros
  std::thread::id thread;
};

}  // namespace

// The build runs on a worker from the snapshot alone: the live state it was taken from changes and
// goes before the build runs, and the result is still the snapshot's
TEST(VariantCompileJob, ABuildReadsItsSnapshotOnly)
{
  auto resources = std::make_unique<FakeResources>(FakeResources{.wireframe = false, .features = {"GLTF_USE_TRANSMISSION"}});
  VariantJobs<FakeJob, FakeBuilt> jobs;
  std::promise<void>              release;
  std::shared_future<void>        released = release.get_future().share();
  ASSERT_TRUE(jobs.start(snapshot(*resources, 7), [released](const FakeJob& job) {
    released.wait();
    FakeBuilt built{.thread = std::this_thread::get_id()};
    for(const auto& [name, value] : job.macros)
      built.source += name + "=" + value + ";";
    return built;
  }));
  ASSERT_NE(jobs.job(), nullptr);
  EXPECT_EQ(jobs.job()->key, 7);

  resources->wireframe = true;
  resources->features.clear();
  resources.reset();
  EXPECT_FALSE(jobs.take());                                                                // still building
  EXPECT_FALSE(jobs.start(FakeJob{.key = 8}, [](const FakeJob&) { return FakeBuilt{}; }));  // one at a time

  release.set_value();
  jobs.join();
  const auto done = jobs.take();
  ASSERT_TRUE(done);
  EXPECT_EQ(done->first.key, 7);
  EXPECT_EQ(done->second.source, "WIREFRAME=0;GLTF_USE_TRANSMISSION=1;");
  EXPECT_NE(done->second.thread, std::this_thread::get_id());
  EXPECT_FALSE(jobs.take());  // taken once
  EXPECT_EQ(jobs.job(), nullptr);
}

// Once taken, the next job starts; a job never taken is waited for at teardown
TEST(VariantCompileJob, TheNextJobStartsOnceTheLastIsTaken)
{
  std::vector<int> built;
  {
    VariantJobs<FakeJob, int> jobs;
    ASSERT_TRUE(jobs.start(FakeJob{.key = 1}, [](const FakeJob& job) { return job.key * 10; }));
    jobs.join();
    const auto first = jobs.take();
    ASSERT_TRUE(first);
    built.push_back(first->second);
    ASSERT_TRUE(jobs.start(FakeJob{.key = 2}, [](const FakeJob& job) { return job.key * 10; }));
  }  // destroyed with job 2 untaken: joined, not crashed
  EXPECT_EQ(built, std::vector<int>{10});
}

// done() says a result waits, so the renderer draws a frame to install it even with the still
// converged: not before the build returns, and not after the result is taken
TEST(VariantCompileJob, DoneMeansAResultWaitsToBeTaken)
{
  VariantJobs<FakeJob, int> jobs;
  EXPECT_FALSE(jobs.done());
  std::promise<void> release;
  std::shared_future<void> go = release.get_future().share();
  ASSERT_TRUE(jobs.start(FakeJob{.key = 3}, [go](const FakeJob& job) {
    go.wait();
    return job.key;
  }));
  EXPECT_FALSE(jobs.done());  // still building
  release.set_value();
  jobs.join();
  EXPECT_TRUE(jobs.done());
  ASSERT_TRUE(jobs.take());
  EXPECT_FALSE(jobs.done());
}
