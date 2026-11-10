/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// gpu_trail without a device: the frames it keeps for a lost device's log, and what each says

#include <gtest/gtest.h>

#include "gpu_crash_trail.hpp"

TEST(GpuCrashTrail, TheLastFramesAreKeptOldestFirst)
{
  for(uint64_t f = 1; f <= 700; f++)
    gpu_trail::record({.frame = f});
  const std::vector<gpu_trail::Frame> last = gpu_trail::recent(3);
  ASSERT_EQ(last.size(), 3u);
  EXPECT_EQ(last[0].frame, 698u);
  EXPECT_EQ(last[2].frame, 700u);
  EXPECT_EQ(gpu_trail::recent(100000).size(), 600u) << "the last 600";
  EXPECT_GE(last[2].seconds, last[0].seconds);
}

TEST(GpuCrashTrail, AFrameSaysItsMemoryAndWhatItsSyncDid)
{
  const std::string text = gpu_trail::describe({.frame       = 9123,
                                                .seconds     = 672.34,
                                                .hostMs      = 4319.7,
                                                .budget      = 14683ull << 20,
                                                .usage       = 13869ull << 20,
                                                .allocated   = 11194ull << 20,
                                                .spilled     = 608ull << 20,
                                                .stagingHeld = 128ull << 20,
                                                .stagingLive = 12ull << 20,
                                                .decoded     = 0,
                                                .copied      = 2,
                                                .refused     = 0,
                                                .queued      = 3,
                                                .sync        = "changed, textures moved"});
  EXPECT_EQ(text, "frame 9123 at 672.3 s, 4320 ms after the last: GPU 13869 of 14683 MB (allocator 11194), 608 MB in system memory; "
                  "staging 128 MB held, 12 live; images 0 decoded, 2 copied, 0 refused, 3 waiting; sync: changed, textures moved");
}

TEST(GpuCrashTrail, AFaultsAddressIsNamedByTheBufferItWasIn)
{
  const auto handle = [](uint64_t h) { return reinterpret_cast<VkBuffer>(h); };
  gpu_trail::frameStart(VK_NULL_HANDLE, 40);
  gpu_trail::buffer(handle(0x1001), 0x7000'0000, 1 << 20, true);
  gpu_trail::named(VK_OBJECT_TYPE_BUFFER, 0x1001, "previous positions");
  gpu_trail::named(VK_OBJECT_TYPE_IMAGE, 0x1001, "an image's name is not a buffer's");
  gpu_trail::buffer(handle(0x1002), 0x7010'0000, 1 << 20, true);  // the next MB, with no name
  gpu_trail::frameStart(VK_NULL_HANDLE, 45);
  gpu_trail::buffer(handle(0x1001), 0x7000'0000, 1 << 20, false);

  EXPECT_EQ(gpu_trail::whereIs(0x7000'5000, 0x1000),
            "in 'previous positions' (1.0 MB from 0x70000000), made in frame 40 and destroyed in frame 45");
  EXPECT_EQ(gpu_trail::whereIs(0x7010'0000, 0x1000), "in 'a buffer with no name' (1.0 MB from 0x70100000), made in frame 40, there still");
  // a page reaching into the next buffer is in both
  EXPECT_EQ(gpu_trail::whereIs(0x700F'F800, 0x1000),
            "in 'a buffer with no name' (1.0 MB from 0x70100000), made in frame 40, there still; before that in 'previous positions' "
            "(1.0 MB from 0x70000000), made in frame 40 and destroyed in frame 45");
  EXPECT_EQ(gpu_trail::whereIs(0x9000'0000, 0x1000).rfind("in none of the ", 0), 0u);

  // the range made again: what is there now, then what was
  gpu_trail::frameStart(VK_NULL_HANDLE, 50);
  gpu_trail::buffer(handle(0x1003), 0x7000'0000, 1 << 19, true);
  gpu_trail::named(VK_OBJECT_TYPE_BUFFER, 0x1003, "scene");
  EXPECT_EQ(gpu_trail::whereIs(0x7000'0000, 0x1000), "in 'scene' (0.5 MB from 0x70000000), made in frame 50, there still; before that in "
                                                     "'previous positions' (1.0 MB from 0x70000000), made in frame 40 and destroyed in frame 45");
  for(uint64_t h : {0x1002, 0x1003})
    gpu_trail::buffer(handle(h), 0, 0, false);
}

TEST(GpuCrashTrail, AFaultsAddressIsNamedByTheImageItWasIn)
{
  const auto image = [](uint64_t h) { return reinterpret_cast<VkImage>(h); };
  gpu_trail::frameStart(VK_NULL_HANDLE, 60);
  gpu_trail::image(image(0x2001), 0x8000'0000, 4 << 20, false, true);
  gpu_trail::named(VK_OBJECT_TYPE_IMAGE, 0x2001, "skin_D_1001.jpg");
  gpu_trail::named(VK_OBJECT_TYPE_BUFFER, 0x2001, "a buffer's name is not an image's");
  gpu_trail::image(image(0x2002), 0x8040'0000, 4 << 20, true, true);  // spilled to system memory
  gpu_trail::image(image(0x2003), 0, 64 << 20, false, true);          // dedicated memory: no address
  gpu_trail::frameStart(VK_NULL_HANDLE, 66);
  gpu_trail::image(image(0x2001), 0, 0, false, false);

  EXPECT_EQ(gpu_trail::whereIs(0x8000'1000, 0x1000),
            "in the image 'skin_D_1001.jpg' (4.0 MB from 0x80000000), made in frame 60 and destroyed in frame 66");
  EXPECT_EQ(gpu_trail::whereIs(0x8040'1000, 0x1000),
            "in the image 'an image with no name' (4.0 MB from 0x80400000, in system memory), made in frame 60, there still");
  // the one with no address is counted where nothing is found, never matched (not at 0 either)
  EXPECT_NE(gpu_trail::whereIs(0x0000'1000, 0x1000).find("one of the 1 images in memory of their own"), std::string::npos);
  for(uint64_t h : {0x2002, 0x2003})
    gpu_trail::image(image(h), 0, 0, false, false);
  gpu_trail::forgetJournals();
}

// The driver's report names whatever it bound at an address: an image in system memory's type, which no buffer may
// bind to and so the allocator's hooks can't place, a memory block, the driver's own
TEST(GpuCrashTrail, AFaultsAddressIsNamedByWhatTheDriverBoundThere)
{
  gpu_trail::forgetJournals();
  gpu_trail::frameStart(VK_NULL_HANDLE, 70);
  gpu_trail::binding(VK_OBJECT_TYPE_DEVICE_MEMORY, 0x3001, 0x9000'0000, 64 << 20, true, false);
  gpu_trail::binding(VK_OBJECT_TYPE_IMAGE, 0x3002, 0x9000'0000, 16 << 20, true, false);
  gpu_trail::named(VK_OBJECT_TYPE_IMAGE, 0x3002, "wall_N.jpg");
  gpu_trail::named(VK_OBJECT_TYPE_BUFFER, 0x3003, "named before it was bound");
  gpu_trail::binding(VK_OBJECT_TYPE_BUFFER, 0x3003, 0x9400'0000, 1 << 20, true, false);
  gpu_trail::binding(VK_OBJECT_TYPE_DEVICE_MEMORY, 0x3004, 0x9500'0000, 1 << 20, true, true);
  gpu_trail::frameStart(VK_NULL_HANDLE, 75);
  gpu_trail::binding(VK_OBJECT_TYPE_IMAGE, 0x3002, 0x9000'0000, 16 << 20, false, false);
  gpu_trail::binding(VK_OBJECT_TYPE_DEVICE_MEMORY, 0x3001, 0x9000'0000, 64 << 20, false, false);

  EXPECT_EQ(gpu_trail::whereIs(0x9000'1000, 0x1000),
            "in the memory block 'with no name' (64.0 MB from 0x90000000), made in frame 70 and destroyed in frame 75; before that in "
            "the image 'wall_N.jpg' (16.0 MB from 0x90000000), made in frame 70 and destroyed in frame 75");
  EXPECT_EQ(gpu_trail::whereIs(0x9400'0000, 0x1000),
            "in the buffer 'named before it was bound' (1.0 MB from 0x94000000), made in frame 70, there still");
  EXPECT_EQ(gpu_trail::whereIs(0x9500'0800, 0x100),
            "in the memory block 'with no name' (1.0 MB from 0x95000000, the driver's own), made in frame 70, there still");
  EXPECT_EQ(gpu_trail::whereIs(0x1'0000'0000, 0x1000).rfind("in nothing the driver gave that address", 0), 0u);
  gpu_trail::forgetJournals();
}

// What the driver's ray-tracing validation says (FV_RTV) is kept for the lost device's log: each different message
// once, with the frames it was said in, a warning apart from an error, and only the last few
TEST(GpuCrashTrail, WhatTheDriversValidationSaidIsKept)
{
  gpu_trail::forgetJournals();
  gpu_trail::frameStart(VK_NULL_HANDLE, 41);
  const auto say = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, const char* id, const char* text) {
    const VkDebugUtilsMessengerCallbackDataEXT data{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT, .pMessageIdName = id, .pMessage = text};
    EXPECT_EQ(gpu_trail::validationMessage(severity, VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, &data, nullptr), VkBool32(VK_FALSE));
  };
  say(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, "RTV", "a BLAS update read a NaN position");
  say(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, "RTV", "nothing wrong");  // not kept
  gpu_trail::frameStart(VK_NULL_HANDLE, 42);
  say(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, nullptr, "a TLAS names a BLAS built after it");
  for(uint64_t frame = 43; frame <= 45; frame++)
  {
    gpu_trail::frameStart(VK_NULL_HANDLE, frame);
    say(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, "RTV", "a BLAS update read a NaN position");
  }
  EXPECT_EQ(gpu_trail::messages(), (std::vector<std::string>{"frames 41 to 45, 4 times, error: [RTV] a BLAS update read a NaN position",
                                                             "frame 42, warning: a TLAS names a BLAS built after it"}));

  for(int i = 0; i < 40; i++)
    gpu_trail::message(true, "RTV", ("message " + std::to_string(i)).c_str(), false);
  const std::vector<std::string> kept = gpu_trail::messages();
  ASSERT_EQ(kept.size(), 16u);
  EXPECT_EQ(kept.back(), "frame 45, error: [RTV] message 39");
  EXPECT_EQ(kept.front(), "frame 45, error: [RTV] message 24");
  gpu_trail::forgetJournals();
  EXPECT_TRUE(gpu_trail::messages().empty());

  // with the objects a message is about, by the names they were given; the instance says nothing
  gpu_trail::frameStart(VK_NULL_HANDLE, 50);
  const VkDebugUtilsObjectNameInfoEXT objects[] = {
      {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, .objectType = VK_OBJECT_TYPE_INSTANCE, .objectHandle = 0x2000, .pObjectName = "instance"},
      {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, .objectType = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, .objectHandle = 0x2001, .pObjectName = "BLAS hair"},
      {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, .objectType = VK_OBJECT_TYPE_BUFFER, .objectHandle = 0x2002},
  };
  const VkDebugUtilsMessengerCallbackDataEXT about{.sType          = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
                                                   .pMessageIdName = "NVIDIA",
                                                   .pMessage       = "INVALID_VERTEX_COORDINATE\n    index: 3165\n",
                                                   .objectCount    = 3,
                                                   .pObjects       = objects};
  for(int i = 0; i < 2; i++)
    gpu_trail::validationMessage(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, &about, nullptr);
  EXPECT_EQ(gpu_trail::messages(), (std::vector<std::string>{"frames 50 to 50, 2 times, error: [NVIDIA] INVALID_VERTEX_COORDINATE\n    index: 3165\n"
                                                             "    about acceleration structure 0x2001 'BLAS hair'\n    about buffer 0x2002"}));
  gpu_trail::forgetJournals();
}

TEST(GpuCrashTrail, WithoutTheExtensionsNothingIsMarked)
{
  // no device: init never ran, so checkpoints are off and a section or frame marks nothing
  EXPECT_FALSE(gpu_trail::checkpointsOn());
  gpu_trail::frameStart(VK_NULL_HANDLE, 1);
  gpu_trail::section(VK_NULL_HANDLE, "Path trace");
  gpu_trail::section(VK_NULL_HANDLE, nullptr);
}
