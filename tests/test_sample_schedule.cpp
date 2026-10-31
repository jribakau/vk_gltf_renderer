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

// sample_schedule: how many samples per pixel the path tracer traces in a frame.

#include <vector>

#include <gtest/gtest.h>

#include "sample_schedule.hpp"

namespace {

// The accumulated count after each frame of `perFrame`, up to `maxSamples`
std::vector<int> accumulate(int perFrame, int maxSamples, const std::vector<int>& stops)
{
  std::vector<int> counts;
  for(int samples = 0; samples < maxSamples;)
  {
    samples += sample_schedule::frameSamples(samples, perFrame, maxSamples, stops);
    counts.push_back(samples);
  }
  return counts;
}

}  // namespace

TEST(SampleSchedule, TheFinalImageKeepsAllItsDetailWhereverItStopped)
{
  // the same image stopped at 200: the cap the user set must not smooth it
  EXPECT_FLOAT_EQ(sample_schedule::denoiseBlend(0.4f, 200, 500, true), 0.4f);
  EXPECT_FLOAT_EQ(sample_schedule::denoiseBlend(0.4f, 200, 1000, true), 0.4f);
  // while it refines, it grows with the samples
  EXPECT_FLOAT_EQ(sample_schedule::denoiseBlend(0.4f, 250, 500, false), 0.2f);
  EXPECT_FLOAT_EQ(sample_schedule::denoiseBlend(0.4f, 9000, 500, false), 0.4f);
  EXPECT_EQ(sample_schedule::denoiseBlend(0.f, 200, 500, true), 0.f);
  EXPECT_EQ(sample_schedule::denoiseBlend(0.4f, 200, 0, true), 0.f);
}

TEST(SampleSchedule, EndsExactlyOnTheCap)
{
  EXPECT_EQ(accumulate(3, 10, {}), (std::vector<int>{3, 6, 9, 10}));
  EXPECT_EQ(accumulate(16, 16, {}), (std::vector<int>{16}));
  EXPECT_EQ(accumulate(1, 3, {}), (std::vector<int>{1, 2, 3}));
}

TEST(SampleSchedule, EndsExactlyOnEachStop)
{
  // --outputSamples 1,2,4,8 with 5 spp per frame: every count is an image of its own
  EXPECT_EQ(accumulate(5, 20, {1, 2, 4, 8}), (std::vector<int>{1, 2, 4, 8, 13, 18, 20}));
  // Stops above the cap change nothing past it
  EXPECT_EQ(accumulate(4, 8, {6, 100}), (std::vector<int>{4, 6, 8}));
}

TEST(SampleSchedule, AlwaysTracesAtLeastOneSample)
{
  // Past the cap (it was lowered mid-accumulation) the renderer stops, but a frame never traces 0
  EXPECT_EQ(sample_schedule::frameSamples(12, 4, 10, {}), 1);
  EXPECT_EQ(sample_schedule::frameSamples(0, 0, 10, {}), 1);
}

// A still that accumulates ends on its samples, however many frames came first (a DLSS hand-off frame
// adds none: 128 frames once ended it at 127 samples); a view with no samples ends on its frames
TEST(SampleSchedule, AccumulationEndsOnItsSamplesNotItsFrames)
{
  EXPECT_FALSE(sample_schedule::accumulationComplete(128, 127, 128));  // one hand-off frame before
  EXPECT_TRUE(sample_schedule::accumulationComplete(129, 128, 128));
  EXPECT_TRUE(sample_schedule::accumulationComplete(40, 128, 128));    // several samples a frame
  EXPECT_FALSE(sample_schedule::accumulationComplete(127, 0, 128));    // DLSS frames alone
  EXPECT_TRUE(sample_schedule::accumulationComplete(128, 0, 128));
  EXPECT_FALSE(sample_schedule::accumulationComplete(-1, 128, 128));   // just reset
}

TEST(SampleSchedule, TheAdaptiveStopEndsAnAccumulationBeforeItsCap)
{
  EXPECT_TRUE(sample_schedule::accumulationComplete(200, 200, 500, true));
  EXPECT_FALSE(sample_schedule::accumulationComplete(200, 200, 500, false));
  EXPECT_FALSE(sample_schedule::accumulationComplete(-1, 200, 500, true));  // just reset
  EXPECT_FALSE(sample_schedule::accumulationComplete(3, 0, 500, true));     // DLSS frames, no samples yet
}

TEST(SampleSchedule, DlssFramesTakeTheLowerBounceCap)
{
  EXPECT_EQ(sample_schedule::frameMaxDepth(8, 4, true), 4);
  EXPECT_EQ(sample_schedule::frameMaxDepth(8, 4, false), 8);  // the still keeps its own
  EXPECT_EQ(sample_schedule::frameMaxDepth(8, 0, true), 8);   // 0: the still's
  EXPECT_EQ(sample_schedule::frameMaxDepth(3, 6, true), 3);   // never above the still's
}

namespace {

// Drive a ThroughputSearch with a GPU time model: `fixedMs` per frame plus `perSampleMs(samples)` per
// sample. Returns the count it settles on.
template <class PerSample>
int searchThroughput(double fixedMs, PerSample perSampleMs, double maxFrameMs = 100.0, int maxSamples = 1024)
{
  sample_schedule::ThroughputSearch search;
  int                               samples = 1;
  for(int frame = 0; frame < 1000 && !search.settled(); frame++)
    samples = search.update(samples, fixedMs + samples * perSampleMs(samples), maxFrameMs, maxSamples);
  EXPECT_TRUE(search.settled());
  return samples;
}

}  // namespace

TEST(SampleSchedule, ThroughputClimbsWhileSamplesGetCheaper)
{
  // 1 ms per frame of fixed work, 1 ms per sample: the time per sample is 1 + 1/spp, and every step of
  // x4 pays off. It stops at 64: a frame of 256 would take 257 ms, past the 100 ms ceiling
  EXPECT_EQ(searchThroughput(1.0, [](int) { return 1.0; }), 64);
  // Cheaper samples gain more per step; a 50 ms ceiling stops it at 64 spp (33 ms; 256 would be 129)
  EXPECT_EQ(searchThroughput(1.0, [](int) { return 0.5; }, 50.0), 64);
}

TEST(SampleSchedule, ThroughputStaysAtOneWhenMoreCostsMore)
{
  // A scene where a sample costs more in a longer frame (measured on complex scenes)
  EXPECT_EQ(searchThroughput(0.0, [](int samples) { return 10.0 * (1.0 + 0.05 * (samples - 1)); }), 1);
}

TEST(SampleSchedule, ThroughputIgnoresGainsWithinTheNoise)
{
  // 0.5 % fixed work: going from 1 to 4 saves 0.4 % per sample, less than the noise threshold
  EXPECT_EQ(searchThroughput(0.05, [](int) { return 10.0; }), 1);
}

TEST(SampleSchedule, ThroughputLooksPastAStepThatGainsLittle)
{
  // 4 spp gains under the noise threshold over 1, but 16 gains clearly: the search does not stop at 4
  EXPECT_EQ(searchThroughput(0.0, [](int samples) { return samples == 1 ? 1.0 : samples == 4 ? 0.995 : 0.9; }), 16);
}

TEST(SampleSchedule, ThroughputRespectsTheCeiling)
{
  EXPECT_EQ(searchThroughput(1.0, [](int) { return 0.01; }, 100.0, 256), 256);
  // The next step would pass the samples ceiling: it stays at the last count that fits
  EXPECT_EQ(searchThroughput(1.0, [](int) { return 0.01; }, 100.0, 1000), 256);
}

namespace {

// Drive a FrameRateControl from one sample per frame for `frames` frames. The GPU time of a frame comes
// `lag` frames late, as the profiler reports it. Returns the count of each frame.
template <class GpuMs>
std::vector<int> climb(GpuMs gpuMs, double targetMs, int frames, int lag = 0)
{
  sample_schedule::FrameRateControl control;
  std::vector<int>                  counts{1};
  for(int frame = 1; frame < frames; frame++)
  {
    const int timed = counts[std::max(int(counts.size()) - 1 - lag, 0)];
    counts.push_back(control.update(counts.back(), gpuMs(timed), targetMs, 1, 1024));
  }
  return counts;
}

}  // namespace

TEST(SampleSchedule, FrameRateClimbsByOneUntilTheTarget)
{
  // 1 ms a sample, 30 ms target: one more each frame until 24 (24 ms is 0.8 x target), then held
  const std::vector<int> counts = climb([](int samples) { return 1.0 * samples; }, 30.0, 60);
  for(int frame = 0; frame < 24; frame++)
    EXPECT_EQ(counts[frame], frame + 1);
  for(size_t frame = 24; frame < counts.size(); frame++)
    EXPECT_EQ(counts[frame], 24);
}

TEST(SampleSchedule, FrameRateStepsBackAfterTheLagThenHolds)
{
  // The time comes 4 frames late: the count overshoots, steps back under 1.1 x target, and stays there
  // -- it does not climb again, which is what made it fluctuate
  const std::vector<int> counts = climb([](int samples) { return 2.0 * samples; }, 30.0, 80, 4);
  EXPECT_EQ(counts.back(), 16);  // 32 ms, under 33
  for(size_t frame = 40; frame < counts.size(); frame++)
    EXPECT_EQ(counts[frame], 16);
}

TEST(SampleSchedule, FrameRateHoldsThroughTimingNoise)
{
  sample_schedule::FrameRateControl control;
  int                               samples = 20;
  samples                                   = control.update(samples, 25.0, 30.0, 1, 1024);  // reached: 0.83 x target
  ASSERT_TRUE(control.held());
  for(double ms : {19.0, 22.0, 30.0, 18.0, 32.0})  // a held count does not climb on a fast frame
    samples = control.update(samples, ms, 30.0, 1, 1024);
  EXPECT_EQ(samples, 20);
}

TEST(SampleSchedule, FrameRateStepsDownUnderLoad)
{
  sample_schedule::FrameRateControl control;
  int                               samples = 20;
  samples                                   = control.update(samples, 25.0, 30.0, 1, 1024);
  samples = control.update(samples, 40.0, 30.0, 1, 1024);  // another application took the GPU
  EXPECT_EQ(samples, 19);
  samples = control.update(samples, 31.0, 30.0, 1, 1024);
  EXPECT_EQ(samples, 19);
}

TEST(SampleSchedule, FrameRateStaysInRange)
{
  EXPECT_EQ(climb([](int) { return 0.01; }, 30.0, 40).back(), 40);  // still climbing, one a frame
  EXPECT_EQ(climb([](int) { return 100.0; }, 30.0, 10).back(), 1);  // never under one
}
