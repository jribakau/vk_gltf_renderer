/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_crash_trail.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <volk.h>

#include <nvutils/logger.hpp>

namespace gpu_trail {
namespace {

constexpr size_t kFramesKept   = 600;
constexpr size_t kFramesDumped = 120;
constexpr size_t kFrameMarkers = 16;  // more than frames in flight: a lost queue's marker still names its frame
constexpr double kSlowFrameMs  = 1000;
constexpr size_t kFreedKept    = 50000;  // buffers, images and memory destroyed, the last of them
constexpr size_t kWhereShown   = 3;      // of those an address was in, the latest

// What a checkpoint points to, at an address that stays for the program's life: its text, and the frame slot it was set
// in (FrameMarker), so that it names its frame
struct Marker
{
  std::string text;
  size_t      slot = 0;
};
struct FrameMarker
{
  uint64_t frame = 0;
};
using SectionMarkers = std::array<std::pair<const Marker*, const Marker*>, kFrameMarkers>;  // a start and an end a slot

// A buffer's or an image's memory, by GPU address
struct BufferRange
{
  VkDeviceAddress address = 0;  // an image's: 0 when its memory is dedicated to it, which tells no address
  VkDeviceSize    size    = 0;
  std::string     name;
  uint64_t        made = 0, destroyed = 0;  // frames
  bool            image        = false;
  bool            systemMemory = false;    // an image made in system memory, the device's being full
  const char*     kind         = nullptr;  // what the driver said it bound there (bindingMessage): "image", "memory block", ...
  bool            internal     = false;    // memory the driver bound for itself
};

struct State
{
  std::recursive_mutex mutex;  // deviceLost can come from inside what holds it
  VkDevice             device      = VK_NULL_HANDLE;
  VkQueue              queue       = VK_NULL_HANDLE;
  bool                 checkpoints = false;
  bool                 deviceFault = false;
  bool                 lost        = false;
  uint64_t             frame       = 0;  // the one recording (frameStart)
  std::deque<Marker>   markers;
  std::unordered_map<std::string, SectionMarkers>                 named;  // a section's, by name
  std::unordered_map<VkCommandBuffer, std::vector<const Marker*>> open;   // the ends of the sections open
  std::array<FrameMarker, kFrameMarkers> frameMarkers{};
  std::unordered_map<VkBuffer, BufferRange> buffers;  // those there now
  std::unordered_map<VkImage, BufferRange>  images;   // those there now
  // what the driver reports binding (VK_EXT_device_address_binding_report), by object and address: every object,
  // in every memory type, the driver's own too. When it reports, the two above stay empty
  std::map<std::pair<uint64_t, VkDeviceAddress>, BufferRange> bound;
  std::unordered_map<uint64_t, std::string>                   names;  // given before an object was bound
  bool                                                        reported = false;
  std::deque<BufferRange>                   freed;    // the last destroyed, oldest first
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
      return fmt::format("frame {}: {}", s.frameMarkers[m.slot].frame, m.text);
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
  {
    LOGE("[gpu]   %s: 0x%llx, to within 0x%llx\n", addressTypeName(a.addressType), (unsigned long long)a.reportedAddress,
         (unsigned long long)a.addressPrecision);
    if(a.addressType == VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT || a.addressType == VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT
       || a.addressType == VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT)
      LOGE("[gpu]     it was %s\n", whereIs(a.reportedAddress, a.addressPrecision).c_str());
  }
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
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  s.frame = frame;
  if(!s.checkpoints)
    return;
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
    const size_t slot  = s.frame % kFrameMarkers;
    auto& [begin, end] = s.named[name][slot];
    if(!begin)
    {
      begin = &s.markers.emplace_back(Marker{fmt::format("{} starts", name), slot});
      end   = &s.markers.emplace_back(Marker{fmt::format("{} ends", name), slot});
    }
    vkCmdSetCheckpointNV(cmd, begin);
    s.open[cmd].push_back(end);
    return;
  }
  const auto open = s.open.find(cmd);
  if(open == s.open.end() || open->second.empty())
    return;
  vkCmdSetCheckpointNV(cmd, open->second.back());
  open->second.pop_back();
}

namespace {

// A buffer or an image made (`made` true: into `live`) or destroyed (out of it, into the freed, with the frame)
template <class Handle>
void journal(std::unordered_map<Handle, BufferRange>& live, Handle handle, BufferRange range, bool made)
{
  State& s = state();
  if(made)
  {
    range.made   = s.frame;
    live[handle] = std::move(range);
    return;
  }
  const auto it = live.find(handle);
  if(it != live.end())
  {
    range = std::move(it->second);
    live.erase(it);
  }
  range.destroyed = s.frame;
  s.freed.push_back(std::move(range));
  if(s.freed.size() > kFreedKept)
    s.freed.pop_front();
}

}  // namespace

void buffer(VkBuffer buffer, VkDeviceAddress address, VkDeviceSize size, bool made)
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  journal(s.buffers, buffer, {.address = address, .size = size}, made);
}

void image(VkImage image, VkDeviceAddress address, VkDeviceSize size, bool systemMemory, bool made)
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  journal(s.images, image, {.address = address, .size = size, .image = true, .systemMemory = systemMemory}, made);
}

namespace {

const char* kindOf(VkObjectType type)
{
  switch(type)
  {
    case VK_OBJECT_TYPE_BUFFER:
      return "buffer";
    case VK_OBJECT_TYPE_IMAGE:
      return "image";
    case VK_OBJECT_TYPE_DEVICE_MEMORY:
      return "memory block";
    case VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR:
      return "acceleration structure";
    case VK_OBJECT_TYPE_MICROMAP_EXT:
      return "micromap";
    default:
      return "object";
  }
}

}  // namespace

void binding(VkObjectType type, uint64_t handle, VkDeviceAddress address, VkDeviceSize size, bool bound, bool internal)
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if(!std::exchange(s.reported, true))
    LOGI("GPU crash trail: the driver reports the addresses it binds (the first: a %s, %.1f MB)\n", kindOf(type), double(size) / double(1 << 20));
  const auto       key = std::pair{handle, address};
  const auto       name = s.names.find(handle);
  if(bound)
  {
    s.bound[key] = {.address  = address,
                    .size     = size,
                    .name     = name != s.names.end() ? name->second : std::string{},
                    .made     = s.frame,
                    .kind     = kindOf(type),
                    .internal = internal};
    return;
  }
  BufferRange range{.address = address, .size = size, .kind = kindOf(type), .internal = internal};
  if(const auto it = s.bound.find(key); it != s.bound.end())
  {
    range = std::move(it->second);
    s.bound.erase(it);
  }
  if(range.name.empty() && name != s.names.end())
    range.name = name->second;
  if(name != s.names.end())
    s.names.erase(name);
  range.destroyed = s.frame;
  s.freed.push_back(std::move(range));
  if(s.freed.size() > kFreedKept)
    s.freed.pop_front();
}

VKAPI_ATTR VkBool32 VKAPI_CALL bindingMessage(VkDebugUtilsMessageSeverityFlagBitsEXT /*severity*/, VkDebugUtilsMessageTypeFlagsEXT type,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user*/)
{
  if(!(type & VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT) || !data || data->objectCount == 0 || !data->pObjects)
    return VK_FALSE;
  for(auto* p = static_cast<const VkBaseInStructure*>(data->pNext); p; p = p->pNext)
    if(p->sType == VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT)
    {
      const auto* b = reinterpret_cast<const VkDeviceAddressBindingCallbackDataEXT*>(p);
      binding(data->pObjects[0].objectType, data->pObjects[0].objectHandle, b->baseAddress, b->size,
              b->bindingType == VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT, (b->flags & VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT) != 0);
    }
  return VK_FALSE;
}

void named(VkObjectType type, uint64_t handle, const char* name)
{
  if(!name)
    return;
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if(s.reported)
  {
    // what the driver bound for this object, named; kept for a binding still to come
    bool any = false;
    for(auto it = s.bound.lower_bound({handle, 0}); it != s.bound.end() && it->first.first == handle; ++it, any = true)
      it->second.name = name;
    if(!any)
      s.names[handle] = name;
  }
  if(type == VK_OBJECT_TYPE_BUFFER)
  {
    if(const auto it = s.buffers.find(reinterpret_cast<VkBuffer>(handle)); it != s.buffers.end())
      it->second.name = name;
  }
  else if(const auto it = s.images.find(reinterpret_cast<VkImage>(handle)); it != s.images.end())
    it->second.name = name;
}

void forgetJournals()
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  s.buffers.clear();
  s.images.clear();
  s.bound.clear();
  s.names.clear();
  s.freed.clear();
  s.reported = false;
}

std::string whereIs(VkDeviceAddress address, VkDeviceSize precision)
{
  State&                                s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  const VkDeviceAddress end    = address + std::max<VkDeviceSize>(precision, 1);
  const auto            within = [&](const BufferRange& b) {
    return b.size > 0 && !(b.image && b.address == 0) && b.address < end && address < b.address + b.size;
  };
  const auto            what   = [](const BufferRange& b) {
    if(b.kind)  // the driver's report
      return fmt::format("the {} '{}' ({:.1f} MB from 0x{:x}{}), made in frame {}", b.kind, b.name.empty() ? "with no name" : b.name,
                         double(b.size) / double(1 << 20), b.address, b.internal ? ", the driver's own" : "", b.made);
    return fmt::format("{}'{}' ({:.1f} MB from 0x{:x}{}), made in frame {}", b.image ? "the image " : "",
                       b.name.empty() ? (b.image ? "an image with no name" : "a buffer with no name") : b.name,
                       double(b.size) / double(1 << 20), b.address, b.systemMemory ? ", in system memory" : "", b.made);
  };
  std::vector<std::string> said;
  const auto               stillThere = [&](const auto& live) {
    for(const auto& [handle, b] : live)
      if(within(b))
        said.push_back("in " + what(b) + ", there still");
  };
  stillThere(s.buffers);
  stillThere(s.images);
  stillThere(s.bound);
  size_t gone = 0;
  for(auto it = s.freed.rbegin(); it != s.freed.rend() && gone < kWhereShown; ++it)
    if(within(*it))
    {
      said.push_back(fmt::format("in {} and destroyed in frame {}", what(*it), it->destroyed));
      gone++;
    }
  if(said.empty() && s.reported)
    return fmt::format("in nothing the driver gave that address: none of the {} bindings there or the last {} let go of", s.bound.size(),
                       s.freed.size());
  if(said.empty())
  {
    size_t unplaced = 0;
    for(const auto& [handle, b] : s.images)
      unplaced += b.address == 0;
    return fmt::format("in none of the {} buffers and {} images there or the last {} destroyed: one of the {} images in memory of their own, "
                       "whose address is not known, or the driver's own",
                       s.buffers.size(), s.images.size() - unplaced, s.freed.size(), unplaced);
  }
  std::string out = said[0];
  for(size_t i = 1; i < said.size(); i++)
    out += "; before that " + said[i];
  return out;
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
