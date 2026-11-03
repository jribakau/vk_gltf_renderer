/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_crash_trail.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <volk.h>

#include <nvutils/logger.hpp>

namespace gpu_trail {
namespace {

// What a checkpoint points to: its text, at an address that stays for the program's life
struct Marker
{
  std::string text;
};
struct FrameMarker
{
  uint64_t frame = 0;
};

constexpr size_t kFramesKept    = 600;
constexpr size_t kFramesDumped  = 120;
constexpr size_t kFrameMarkers  = 16;  // more than frames in flight: a lost queue's marker still names its frame
constexpr double kSlowFrameMs   = 1000;

struct State
{
  std::recursive_mutex mutex;  // deviceLost can come from inside what holds it
  VkDevice             device      = VK_NULL_HANDLE;
  VkQueue              queue       = VK_NULL_HANDLE;
  bool                 checkpoints = false;
  bool                 deviceFault = false;
  bool                 lost        = false;
  std::deque<Marker>   markers;
  std::unordered_map<std::string, std::pair<const Marker*, const Marker*>> named;  // a section's start and end
  std::unordered_map<VkCommandBuffer, std::vector<const Marker*>>         open;   // the ends of the sections open
  std::array<FrameMarker, kFrameMarkers> frameMarkers{};
  std::array<Frame, kFramesKept>         frames{};
  size_t                                 frameCount = 0;
  std::chrono::steady_clock::time_point  start      = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point  last       = start;
};

State& state()
{
  static State s;
  return s;
}

std::string markerText(const State& s, const void* marker)
{
  const auto* frame = static_cast<const FrameMarker*>(marker);
  if(frame >= s.frameMarkers.data() && frame < s.frameMarkers.data() + kFrameMarkers)
    return fmt::format("frame {} begins", frame->frame);
  for(const Marker& m : s.markers)
    if(&m == marker)
      return m.text;
  return "(no checkpoint)";
}

const char* stageName(VkPipelineStageFlagBits stage)
{
  switch(stage)
  {
    case VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT:
      return "started";
    case VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT:
      return "finished";
    default:
      return "reached";
  }
}

const char* addressTypeName(VkDeviceFaultAddressTypeEXT type)
{
  switch(type)
  {
    case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT:
      return "a read of an address with nothing there";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT:
      return "a write to an address with nothing there";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT:
      return "an instruction fetched from an address with nothing there";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT:
      return "the shader's instruction pointer (unknown)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT:
      return "the shader's instruction pointer (invalid)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:
      return "the shader's instruction pointer (at the fault)";
    default:
      return "an address";
  }
}

void logCheckpoints(const State& s)
{
  if(!s.checkpoints || !vkGetQueueCheckpointDataNV)
  {
    LOGE("[gpu]   no checkpoints (VK_NV_device_diagnostic_checkpoints is not enabled)\n");
    return;
  }
  uint32_t count = 0;
  vkGetQueueCheckpointDataNV(s.queue, &count, nullptr);
  std::vector<VkCheckpointDataNV> data(count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
  vkGetQueueCheckpointDataNV(s.queue, &count, data.data());
  if(count == 0)
    LOGE("[gpu]   the queue reports no checkpoint\n");
  for(uint32_t i = 0; i < count; i++)
    LOGE("[gpu]   the queue's last checkpoint %s: %s\n", stageName(data[i].stage), markerText(s, data[i].pCheckpointMarker).c_str());
}

void logFault(const State& s)
{
  if(!s.deviceFault || !vkGetDeviceFaultInfoEXT)
  {
    LOGE("[gpu]   no fault report (VK_EXT_device_fault is not enabled)\n");
    return;
  }
  VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
  if(vkGetDeviceFaultInfoEXT(s.device, &counts, nullptr) < 0)
  {
    LOGE("[gpu]   the driver gave no fault report\n");
    return;
  }
  std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
  std::vector<VkDeviceFaultVendorInfoEXT>  vendor(counts.vendorInfoCount);
  std::vector<uint8_t>                     binary(counts.vendorBinarySize);
  VkDeviceFaultInfoEXT                     info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
  info.pAddressInfos     = addresses.empty() ? nullptr : addresses.data();
  info.pVendorInfos      = vendor.empty() ? nullptr : vendor.data();
  info.pVendorBinaryData = binary.empty() ? nullptr : binary.data();
  if(vkGetDeviceFaultInfoEXT(s.device, &counts, &info) < 0)
  {
    LOGE("[gpu]   the driver gave no fault report\n");
    return;
  }
  LOGE("[gpu]   the driver's report: %s\n", info.description);
  for(const VkDeviceFaultAddressInfoEXT& a : addresses)
    LOGE("[gpu]   %s: 0x%llx, to within 0x%llx\n", addressTypeName(a.addressType), (unsigned long long)a.reportedAddress,
         (unsigned long long)a.addressPrecision);
  for(const VkDeviceFaultVendorInfoEXT& v : vendor)
    LOGE("[gpu]   %s (code 0x%llx, data 0x%llx)\n", v.description, (unsigned long long)v.vendorFaultCode, (unsigned long long)v.vendorFaultData);
  if(addresses.empty())
    LOGE("[gpu]   no address it could not reach: it hung (the driver's timeout), it did not read freed memory\n");
  if(!binary.empty())
    LOGE("[gpu]   (and %zu bytes of the vendor's own dump, not kept)\n", binary.size());
}

}  // namespace

void init(VkDevice device, VkQueue queue, bool checkpoints, bool deviceFault)
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  s.device      = device;
  s.queue       = queue;
  s.checkpoints = checkpoints && vkCmdSetCheckpointNV && vkGetQueueCheckpointDataNV;
  s.deviceFault = deviceFault && vkGetDeviceFaultInfoEXT;
  LOGI("GPU crash trail: checkpoints %s, fault report %s\n", s.checkpoints ? "on" : "off", s.deviceFault ? "on" : "off");
}

bool checkpointsOn()
{
  return state().checkpoints;
}

void frameStart(VkCommandBuffer cmd, uint64_t frame)
{
  State& s = state();
  if(!s.checkpoints)
    return;
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  FrameMarker& marker = s.frameMarkers[frame % kFrameMarkers];
  marker.frame        = frame;
  vkCmdSetCheckpointNV(cmd, &marker);
}

void section(VkCommandBuffer cmd, const char* name)
{
  State& s = state();
  if(!s.checkpoints)
    return;
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if(name)
  {
    auto it = s.named.find(name);
    if(it == s.named.end())
    {
      const Marker* begin = &s.markers.emplace_back(Marker{fmt::format("{} starts", name)});
      const Marker* end   = &s.markers.emplace_back(Marker{fmt::format("{} ends", name)});
      it                  = s.named.emplace(name, std::make_pair(begin, end)).first;
    }
    vkCmdSetCheckpointNV(cmd, it->second.first);
    s.open[cmd].push_back(it->second.second);
    return;
  }
  const auto open = s.open.find(cmd);
  if(open == s.open.end() || open->second.empty())
    return;
  vkCmdSetCheckpointNV(cmd, open->second.back());
  open->second.pop_back();
}

void record(Frame frame)
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  const auto                            now = std::chrono::steady_clock::now();
  frame.seconds = std::chrono::duration<double>(now - s.start).count();
  frame.hostMs  = std::chrono::duration<double, std::milli>(now - s.last).count();
  s.last        = now;
  s.frames[s.frameCount % kFramesKept] = frame;
  s.frameCount++;
  if(frame.hostMs >= kSlowFrameMs && s.frameCount > 1)
    LOGW("[gpu] a slow frame: %s\n", describe(frame).c_str());
}

std::vector<Frame> recent(size_t count)
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  count = std::min({count, s.frameCount, kFramesKept});
  std::vector<Frame> out;
  out.reserve(count);
  for(size_t i = s.frameCount - count; i < s.frameCount; i++)
    out.push_back(s.frames[i % kFramesKept]);
  return out;
}

std::string describe(const Frame& f)
{
  const auto mb = [](uint64_t bytes) { return bytes >> 20; };
  return fmt::format("frame {} at {:.1f} s, {:.0f} ms after the last: GPU {} of {} MB (allocator {}), {} MB in system memory; "
                     "staging {} MB held, {} live; images {} decoded, {} copied, {} refused, {} waiting{}{}",
                     f.frame, f.seconds, f.hostMs, mb(f.usage), mb(f.budget), mb(f.allocated), mb(f.spilled), mb(f.stagingHeld),
                     mb(f.stagingLive), f.decoded, f.copied, f.refused, f.queued, f.sync.empty() ? "" : "; sync: ", f.sync);
}

void deviceLost()
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if(std::exchange(s.lost, true))
    return;
  LOGE("[gpu] The device is lost. What it was doing:\n");
  logCheckpoints(s);
  logFault(s);
  const std::vector<Frame> frames = recent(kFramesDumped);
  LOGE("[gpu]   the last %zu frames, oldest first:\n", frames.size());
  for(const Frame& f : frames)
    LOGE("[gpu]   %s\n", describe(f).c_str());
}

}  // namespace gpu_trail
