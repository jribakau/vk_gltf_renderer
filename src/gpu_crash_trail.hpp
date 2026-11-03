/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

// What a lost device leaves in the log, where "the GPU stopped responding" otherwise names nothing: where its queue had
// got to (NVIDIA's diagnostic checkpoints at each frame's start and each profiled section's start and end), what
// VK_EXT_device_fault reports (an address the GPU could not read or write, or none: it hung), and the last frames'
// memory, one of which taking a second or more is logged as it happens; and what each address it could not reach was
// in, from a journal of the buffers made and destroyed (a buffer's own, or one let go of too soon, or none: an image or
// the driver's). Each part only where the device has its extension; the frames always.
namespace gpu_trail {

struct Frame
{
  uint64_t    frame   = 0;
  double      seconds = 0;  // since the trail began (record() sets it)
  double      hostMs  = 0;  // since the frame before was recorded (record() sets it)
  uint64_t    budget = 0, usage = 0;  // the driver's, over the device-local heaps
  uint64_t    allocated   = 0;        // the allocator's blocks there
  uint64_t    spilled     = 0;        // of the scene, what lies in system memory
  uint64_t    stagingHeld = 0, stagingLive = 0;
  uint32_t    decoded = 0, copied = 0, refused = 0;  // the sync's images
  uint32_t    queued  = 0;                           // images still waiting to be decoded
  std::string sync;                                  // what the sync did, by name
};

// After the device is made: the queue the frames go to, and which of the two extensions it has enabled
void init(VkDevice device, VkQueue queue, bool checkpoints, bool deviceFault);
bool checkpointsOn();
// a frame's first checkpoint, numbered; the sections after it, and the buffers made and destroyed, are of that frame
void frameStart(VkCommandBuffer cmd, uint64_t frame);
// a profiled section's start (its name) or end (nullptr), as the GPU profiler calls it
void section(VkCommandBuffer cmd, const char* name);
// a buffer's GPU address range made or about to be destroyed (nvvk::ResourceAllocator::bufferHook), and the name it is
// given (nvvk::DebugUtil::nameHook: only a buffer's is kept)
void buffer(VkBuffer buffer, VkDeviceAddress address, VkDeviceSize size, bool made);
void named(VkObjectType type, uint64_t handle, const char* name);
// what the `precision` bytes from `address` were in: a buffer still there, ones destroyed (and in which frames), or none
std::string whereIs(VkDeviceAddress address, VkDeviceSize precision);
// a frame's memory: the last few hundred are kept
void record(Frame frame);
// the last `count` frames recorded, oldest first
std::vector<Frame> recent(size_t count);
std::string        describe(const Frame& frame);
// all of it into the log, once; the Vulkan error callback calls it for VK_ERROR_DEVICE_LOST
void deviceLost();

}  // namespace gpu_trail
