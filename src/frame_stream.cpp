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

// httplib pulls in winsock2; it has to come before anything that includes windows.h.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <httplib.h>
#ifdef _WIN32
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

#include <volk.h>

#include "frame_stream.hpp"
#include "frame_pacing.hpp"
#include "present_policy.hpp"
#include "blob_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nvutils/logger.hpp>
#include <turbojpeg.h>

using namespace std::chrono_literals;

namespace {

// A host-visible buffer a frame is copied into, read by the encoder in place: no copy of the
// pixels is made on the application thread. Its states, every change under Impl::rawMutex:
//   eFree      available for the next copy
//   eCopying   the copy is recorded; it is in once the application's frame timeline reaches `signal`
//              (the next onUpdate waits for it while the loop would otherwise sleep)
//   eWaiting   handed to the encoder, not yet taken. One at a time: a newer frame that arrives while
//              the encoder is busy takes the place of the one waiting, so the newest is encoded
//   eEncoding  the encoder reads its pixels
//   eLast      the last frame encoded, kept so the next one can be told a repeat; freed by the next
struct Readback
{
  enum class State
  {
    eFree,
    eCopying,
    eWaiting,
    eEncoding,
    eLast
  };
  VkBuffer       buffer{};
  VkDeviceMemory memory{};
  void*          mapped{};
  VkDeviceSize   size{};
  VkExtent2D     extent{};
  bool           coherent{true};
  State          state{State::eFree};
  uint64_t       signal{};  // the frame's timeline value: the copy is in once the semaphore reached it
  uint64_t       drawn{};   // the renderer's drawn-frame count when the frame was copied (0 without the callback)
  LatencyTrace   trace;     // A traced edit this frame shows (latency_trace.hpp)
  int64_t        inputNs{};  // The first /input this frame shows arrived then (steady clock), or 0
};

// Now on the steady clock, in nanoseconds: the input latency's timestamps
int64_t steadyNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits, VkMemoryPropertyFlags wanted, bool* found)
{
  VkPhysicalDeviceMemoryProperties props{};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
  for(uint32_t i = 0; i < props.memoryTypeCount; i++)
  {
    if((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & wanted) == wanted)
    {
      *found = true;
      return i;
    }
  }
  *found = false;
  return 0;
}

}  // namespace

struct FrameStream::Impl
{
  FrameStreamCreateInfo info;
  nvapp::Application*   app{};
  VkDevice              device{};
  VkPhysicalDevice      physicalDevice{};
  // Frames in flight, plus one waiting for the encoder, one being encoded, one kept as the last
  // encoded (see Readback): enough for every frame to find a free buffer. Rebuilt on resize.
  std::vector<Readback> ring;

  // The ring's states, the frame waiting for the encoder and the last one it encoded, and the
  // trace riding on the waiting frame: application thread <-> encoder thread, under rawMutex.
  // rawCv wakes the encoder for a waiting frame, and the application thread when an encode ends.
  std::mutex              rawMutex;
  std::condition_variable rawCv;
  Readback*               waiting{nullptr};
  Readback*               lastEncoded{nullptr};
  LatencyTrace            rawTrace;
  int64_t                 rawInputNs{0};  // the input riding on the waiting frame, as rawTrace
  // a traced edit waiting for the next frame copied out (application thread)
  LatencyTrace pendingTrace;

  // Input latency. inputNs: the oldest /input not yet taken by a frame (HTTP threads -> application
  // thread). frameInputNs: the input the frames from now on show, until one is copied out
  // (application thread). The averages: encoder thread -> /stats.
  std::atomic<int64_t> inputNs{0};
  int64_t              frameInputNs{0};
  std::atomic<double>  inputMs{0.0};
  std::atomic<double>  inputMsLast{0.0};
  std::atomic<uint64_t> inputSamples{0};

  // The host's profile (info.profile), refreshed on the application thread for /stats
  std::mutex                            profileMutex;
  std::string                           profileJson{"null"};
  std::chrono::steady_clock::time_point profileAt{};

  // encoded frames: encoder thread -> HTTP client threads
  std::mutex                         jpgMutex;
  std::condition_variable            jpgCv;
  std::shared_ptr<const std::string> jpg;
  uint64_t                           jpgSeq{0};
  // frames the encoder finished with, encoded or found a repeat of the last: what a waiting /frame watches
  uint64_t jpgHandled{0};
  VkExtent2D                         jpgExtent{};

  std::atomic<bool>     running{false};
  std::atomic<int>      clients{0};
  std::atomic<uint64_t> framesRendered{0};
  std::atomic<uint64_t> framesEncoded{0};
  std::atomic<uint64_t> framesSent{0};
  std::atomic<uint64_t> framesReplaced{0};  // copied out, then replaced by a newer frame before the encoder took them
  std::atomic<uint64_t> framesSkipped{0};   // not copied: every buffer of the ring was in use
  std::atomic<double>   renderFps{0.0};
  std::atomic<double>   encodeMs{0.0};
  // live encoder settings, read by the encoder thread each frame: the host's for a moving view,
  // the still's while the renderer accumulates or holds one (onUpdate)
  std::atomic<int>  quality{92};
  std::atomic<bool> fullChroma{false};
  std::atomic<int>  hostQuality{92};
  std::atomic<bool> hostFullChroma{false};

  std::thread     encoder;
  std::thread     server;
  httplib::Server http;

  std::chrono::steady_clock::time_point lastFrame{};
  FramePacer                            pacer{std::chrono::milliseconds(16)};
  // this frame's image goes to the stream (set in onUpdate, read in onRender)
  bool copyFrame{true};
  // the application's frame timeline (set on the first copy) and the drawn-frame count last seen
  VkSemaphore timeline{};
  uint64_t    lastDrawn{~uint64_t(0)};
  // a request came in since the last frame: it ends an idle, and cuts the pacing sleep short
  std::atomic<bool>       activity{false};
  std::mutex              wakeMutex;
  std::condition_variable wakeCv;
  std::atomic<bool>       idle{false};
  // the app presents the frames itself (present_policy.hpp), and a /frame waits for one to be copied
  std::atomic<bool> localViewer{false};
  std::atomic<bool> frameWanted{false};

  present_policy::Viewers viewers() const { return {clients.load(), localViewer.load(), frameWanted.load()}; }
  // The encoder is done with a frame (encoder thread, holding no lock): wakes a waiting /frame.
  void handled();

  void createRing(const VkExtent2D& extent);
  void destroyRing();
  bool waitSlot(const Readback& rb, uint64_t timeoutNs) const;
  // Hands the encoder the copies that are in: those of frames up to `waitedSignal` however long they
  // take (the application waits for that frame before it records anyway), the others until `deadline`.
  void collect(std::chrono::steady_clock::time_point deadline, uint64_t waitedSignal);
  void harvest(Readback& rb);
  Readback* freeBuffer();  // a buffer for the next copy, or null when every one is in use
  void wake();
  void encodeLoop();
  void serve();
};

//--------------------------------------------------------------------------------------------------

FrameStream::FrameStream(FrameStreamCreateInfo info)
    : m_impl(std::make_unique<Impl>())
{
  m_impl->info = std::move(info);
  m_impl->quality.store(m_impl->info.quality);
  m_impl->hostQuality.store(m_impl->info.quality);
}

FrameStream::~FrameStream() = default;

void FrameStream::setQuality(int quality)
{
  m_impl->hostQuality.store(std::clamp(quality, 1, 100));
  if(!m_impl->info.still)
    m_impl->quality.store(m_impl->hostQuality.load());
}

void FrameStream::setFullChroma(bool fullChroma)
{
  m_impl->hostFullChroma.store(fullChroma);
  if(!m_impl->info.still)
    m_impl->fullChroma.store(fullChroma);
}

void FrameStream::setLocalViewer(bool local)
{
  if(m_impl->localViewer.exchange(local) != local)
    m_impl->wake();
}

void FrameStream::traceNextFrame(LatencyTrace trace)
{
  m_impl->pendingTrace = std::move(trace);
}

void FrameStream::onAttach(nvapp::Application* app)
{
  Impl& s          = *m_impl;
  s.app            = app;
  s.device         = app->getDevice();
  s.physicalDevice = app->getPhysicalDevice();
  if(!app->isHeadless())
  {
    LOGW("--stream is only available in a headless run; frames are not served.\n");
    return;
  }
  s.running = true;
#ifdef _WIN32
  // the pacing sleeps are a frame long; at the default 15.6 ms timer they would each overshoot by one
  timeBeginPeriod(1);
#endif
  s.encoder = std::thread([&s] { s.encodeLoop(); });
  s.server  = std::thread([&s] { s.serve(); });
}

void FrameStream::onDetach()
{
  Impl& s = *m_impl;
  if(s.running.exchange(false))
  {
    if(s.info.stopping)
      s.info.stopping();
    s.http.stop();
    // the encoder tests `running` under rawMutex: taking the lock here means it has either seen the
    // change or is blocked in its wait, where the notify reaches it (no lost wakeup)
    {
      std::lock_guard lock(s.rawMutex);
    }
    s.rawCv.notify_all();
    s.jpgCv.notify_all();
    s.wakeCv.notify_all();
    if(s.encoder.joinable())
      s.encoder.join();
    if(s.server.joinable())
      s.server.join();
#ifdef _WIN32
    timeEndPeriod(1);
#endif
  }
  vkDeviceWaitIdle(s.device);
  s.destroyRing();
}

// Called by the application with the device idle (start-up, and headless resizes).
void FrameStream::onResize(VkCommandBuffer /*cmd*/, const VkExtent2D& size)
{
  Impl& s = *m_impl;
  s.destroyRing();
  s.createRing(size);
}

// Frame pacing, the GPU wait and the readbacks. A headless run has no v-sync and would otherwise
// spin as fast as the GPU allows; the cap also keeps the renderer from starving the host while
// nobody is watching. The wait for the frame this one's ring slot last held happens here rather
// than in the application, so the host element behind this one takes its input after it; the
// copies the GPU has finished are handed to the encoder while the loop would otherwise sleep.
void FrameStream::onUpdate()
{
  Impl& s = *m_impl;
  if(!s.running)
    return;
  using Clock = std::chrono::steady_clock;
  auto now    = Clock::now();
  if(s.info.drawn)
  {
    const uint64_t drawn = s.info.drawn();
    if(drawn != s.lastDrawn)
    {
      s.lastDrawn = drawn;
      s.pacer.noteDrawn(now);
    }
  }
  if(s.activity.exchange(false))
    s.pacer.noteActivity(now);
  const present_policy::Viewers viewers = s.viewers();
  const int                     fps     = present_policy::pacedFps(s.info.maxFps, viewers);
  const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));
  // Nobody watching: keep to the cap, the host needs the machine more than a still needs to converge.
  const bool refining = present_policy::refining(s.info.refining && s.info.refining(), viewers);
  // A still's frames at full chroma and the still quality; a moving view's as the host set them
  if(s.info.still)
  {
    const bool still = s.info.still();
    s.quality.store(still ? std::max(s.hostQuality.load(), s.info.stillQuality) : s.hostQuality.load());
    s.fullChroma.store(still || s.hostFullChroma.load());
  }
  s.pacer.setPeriod(period);
  s.pacer.setCopyPeriod(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / std::clamp(s.info.refineFps, 1, fps))));
  s.pacer.setIdlePeriod(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / std::max(1, s.info.idleFps))));
  Clock::time_point start = s.pacer.startTime(now, refining);
  s.idle                  = s.pacer.idle(now, refining);

  // The copies that are in, oldest first. The frame this one's ring slot last held is waited for
  // however long it takes: the application waits for it right after onUpdate anyway, and doing it
  // here puts the host element's input after that wait. The others wait until the frame's start time.
  if(!s.ring.empty())
    s.collect(start, s.app->getFrameSignalSemaphore().value);

  // The rest of the wait is a sleep a request cuts short: an idle loop wakes at once, a paced one
  // keeps its start time.
  for(now = Clock::now(); now < start; now = Clock::now())
  {
    {
      std::unique_lock lock(s.wakeMutex);
      s.wakeCv.wait_until(lock, start, [&] { return s.activity.load() || !s.running; });
    }
    if(s.activity.exchange(false))
    {
      now = Clock::now();
      s.pacer.noteActivity(now);
      start  = s.pacer.startTime(now, refining);
      s.idle = false;
    }
  }
  if(s.lastFrame.time_since_epoch().count() != 0)
  {
    const double dt = std::chrono::duration<double>(now - s.lastFrame).count();
    if(dt > 0.0)
      s.renderFps = s.renderFps * 0.9 + (1.0 / dt) * 0.1;
  }
  s.lastFrame = now;
  // a local viewer alone needs no copy (present_policy.hpp); a /frame that came in by now does
  s.copyFrame = present_policy::copyOut(s.pacer.begin(now, refining), s.viewers());

  // Input that came in by now is the host element's this frame (it runs after this one): the frame
  // drawn next shows it. One arriving between here and the host's update counts a frame short.
  if(const int64_t t = s.inputNs.exchange(0); t != 0 && s.frameInputNs == 0)
    s.frameInputNs = t;
  if(s.info.profile && now - s.profileAt >= std::chrono::milliseconds(250))
  {
    s.profileAt         = now;
    std::string profile = s.info.profile();
    std::lock_guard lock(s.profileMutex);
    s.profileJson = profile.empty() ? "null" : std::move(profile);
  }
}

void FrameStream::onRender(VkCommandBuffer cmd)
{
  Impl& s = *m_impl;
  if(!s.running || s.ring.empty() || !s.info.image || !s.info.imageSize)
    return;
  // an unpaced frame the stream is not due: nothing to copy out
  if(!s.copyFrame)
    return;

  const VkImage    image  = s.info.image();
  const VkExtent2D extent = s.info.imageSize();
  if(image == VK_NULL_HANDLE || extent.width == 0 || extent.height == 0)
    return;
  // Copies that completed since onUpdate free their buffers (no waiting); the first free one takes this frame
  s.collect(std::chrono::steady_clock::now(), 0);
  Readback* rbp = s.freeBuffer();
  if(rbp == nullptr)
  {
    s.framesSkipped++;  // the ring is sized so this does not happen in a steady state
    return;
  }
  Readback& rb = *rbp;
  if(VkDeviceSize(extent.width) * extent.height * 4 > rb.size)
    return;  // a resize is on its way; the ring is rebuilt in onResize

  // Finish writing the image, copy it out, then hand it back to whoever uses it next.
  VkImageMemoryBarrier2 toTransfer{
      .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask    = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dstAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT,
      .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
      .image            = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toTransfer};
  vkCmdPipelineBarrier2(cmd, &dep);

  const VkBufferImageCopy2 region{
      .sType            = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent      = {extent.width, extent.height, 1},
  };
  const VkCopyImageToBufferInfo2 copy{
      .sType          = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
      .srcImage       = image,
      .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .dstBuffer      = rb.buffer,
      .regionCount    = 1,
      .pRegions       = &region,
  };
  vkCmdCopyImageToBuffer2(cmd, &copy);

  VkImageMemoryBarrier2 fromTransfer = toTransfer;
  fromTransfer.srcStageMask          = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  fromTransfer.srcAccessMask         = VK_ACCESS_2_TRANSFER_READ_BIT;
  fromTransfer.dstStageMask          = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  fromTransfer.dstAccessMask         = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  const VkBufferMemoryBarrier2 toHost{
      .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
      .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
      .buffer        = rb.buffer,
      .size          = VK_WHOLE_SIZE,
  };
  dep.bufferMemoryBarrierCount = 1;
  dep.pBufferMemoryBarriers    = &toHost;
  dep.pImageMemoryBarriers     = &fromTransfer;
  vkCmdPipelineBarrier2(cmd, &dep);

  const nvvk::SemaphoreInfo signal = s.app->getFrameSignalSemaphore();
  s.timeline                       = signal.semaphore;
  rb.signal                        = signal.value;
  rb.extent                        = extent;
  rb.drawn                         = s.info.drawn ? s.info.drawn() : 0;
  rb.inputNs                       = s.frameInputNs;
  s.frameInputNs                   = 0;
  s.frameWanted                    = false;
  {
    std::lock_guard lock(s.rawMutex);
    rb.state = Readback::State::eCopying;
  }
  if(!s.pendingTrace.empty())
  {
    rb.trace = std::move(s.pendingTrace);
    rb.trace.emplace_back("copied", traceNowMs());
    s.pendingTrace.clear();
  }
  s.framesRendered++;
}

//--------------------------------------------------------------------------------------------------

void FrameStream::Impl::createRing(const VkExtent2D& extent)
{
  if(extent.width == 0 || extent.height == 0)
    return;
  const VkDeviceSize size = VkDeviceSize(extent.width) * extent.height * 4;
  // Before frame N is recorded the application has waited for frame N - K (K frames in flight), so
  // at most K copies are on the GPU, plus one waiting, one being encoded and the last encoded; one
  // more for a copy that completed after the frame's poll for it.
  ring.resize(std::max(1u, app->getFrameCycleSize()) + 4);
  for(Readback& rb : ring)
  {
    const VkBufferCreateInfo bufferInfo{
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if(vkCreateBuffer(device, &bufferInfo, nullptr, &rb.buffer) != VK_SUCCESS)
    {
      LOGE("frame stream: could not create the readback buffer\n");
      destroyRing();
      return;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, rb.buffer, &req);
    // Cached host memory: the encoder reads every byte, and reads from write-combined memory
    // are an order of magnitude slower.
    bool     found = false;
    uint32_t type  = findMemoryType(physicalDevice, req.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &found);
    if(found)
    {
      VkPhysicalDeviceMemoryProperties props{};
      vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
      rb.coherent = (props.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
    else
    {
      type = findMemoryType(physicalDevice, req.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &found);
      rb.coherent = true;
    }
    if(!found)
    {
      LOGE("frame stream: no host-visible memory for the readback buffer\n");
      destroyRing();
      return;
    }
    const VkMemoryAllocateInfo alloc{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = type};
    if(vkAllocateMemory(device, &alloc, nullptr, &rb.memory) != VK_SUCCESS
       || vkBindBufferMemory(device, rb.buffer, rb.memory, 0) != VK_SUCCESS
       || vkMapMemory(device, rb.memory, 0, VK_WHOLE_SIZE, 0, &rb.mapped) != VK_SUCCESS)
    {
      LOGE("frame stream: could not allocate the readback buffer\n");
      destroyRing();
      return;
    }
    rb.size   = size;
    rb.extent = extent;
    rb.state  = Readback::State::eFree;
  }
}

// Application thread, with the device idle. The encoder reads the ring in place, so a frame it is
// encoding is waited for, and the frames waiting for it or kept as the last encoded are dropped.
void FrameStream::Impl::destroyRing()
{
  {
    std::unique_lock lock(rawMutex);
    rawCv.wait(lock, [&] {
      return std::none_of(ring.begin(), ring.end(), [](const Readback& rb) { return rb.state == Readback::State::eEncoding; });
    });
    waiting     = nullptr;
    lastEncoded = nullptr;
    rawTrace.clear();
  }
  for(Readback& rb : ring)
  {
    if(rb.mapped)
      vkUnmapMemory(device, rb.memory);
    if(rb.buffer)
      vkDestroyBuffer(device, rb.buffer, nullptr);
    if(rb.memory)
      vkFreeMemory(device, rb.memory, nullptr);
  }
  ring.clear();
}

// Whether the frame that recorded the slot's copy has completed, waiting up to `timeoutNs` for it.
bool FrameStream::Impl::waitSlot(const Readback& rb, uint64_t timeoutNs) const
{
  if(timeline == VK_NULL_HANDLE)
    return true;
  const VkSemaphoreWaitInfo wait{
      .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores    = &timeline,
      .pValues        = &rb.signal,
  };
  return vkWaitSemaphores(device, &wait, timeoutNs) == VK_SUCCESS;
}

// Application thread: hands the encoder the copies whose frames the GPU has finished, oldest first
// (a newer frame replaces an older one still waiting for the encoder, so the order matters). A copy
// of a frame up to `waitedSignal` is waited for however long it takes, the others until `deadline`;
// frames complete in order, so the first one not in ends the pass.
void FrameStream::Impl::collect(std::chrono::steady_clock::time_point deadline, uint64_t waitedSignal)
{
  std::vector<Readback*> copying;
  {
    std::lock_guard lock(rawMutex);
    for(Readback& rb : ring)
      if(rb.state == Readback::State::eCopying)
        copying.push_back(&rb);
  }
  std::sort(copying.begin(), copying.end(), [](const Readback* a, const Readback* b) { return a->signal < b->signal; });
  for(Readback* rb : copying)
  {
    uint64_t timeout = std::numeric_limits<uint64_t>::max();
    if(rb->signal > waitedSignal)
    {
      const auto left = deadline - std::chrono::steady_clock::now();
      timeout         = left.count() > 0 ? uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(left).count()) : 0;
    }
    if(!waitSlot(*rb, timeout))
      break;
    harvest(*rb);
  }
}

// Application thread: the first free buffer of the ring, or null.
Readback* FrameStream::Impl::freeBuffer()
{
  std::lock_guard lock(rawMutex);
  for(Readback& rb : ring)
    if(rb.state == Readback::State::eFree)
      return &rb;
  return nullptr;
}

// Any thread: a request came in. Ends an idle and cuts the pacing sleep short (onUpdate).
void FrameStream::Impl::wake()
{
  {
    std::lock_guard lock(wakeMutex);
    activity = true;
  }
  wakeCv.notify_all();
}

// Application thread: the buffer's copy has completed. Hand it to the encoder, which reads it in
// place. If the encoder is still busy with an earlier frame and one is already waiting, this one
// takes its place: at 4K an encode takes longer than a frame, and the frame that waits should be
// the newest, not the stalest.
void FrameStream::Impl::harvest(Readback& rb)
{
  if(!rb.coherent)
  {
    const VkMappedMemoryRange range{.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = rb.memory, .size = VK_WHOLE_SIZE};
    vkInvalidateMappedMemoryRanges(device, 1, &range);
  }
  std::unique_lock lock(rawMutex);
  if(waiting != nullptr)
  {
    waiting->state = Readback::State::eFree;
    framesReplaced++;
    // the replaced frame's trace rides on this one, which shows its edit too
    if(rb.trace.empty())
      rb.trace = std::move(rawTrace);
    rawTrace.clear();
    if(rb.inputNs == 0)
      rb.inputNs = rawInputNs;
  }
  rawInputNs = rb.inputNs;
  rb.inputNs = 0;
  if(!rb.trace.empty())
  {
    rb.trace.emplace_back("harvested", traceNowMs());
    rawTrace = std::move(rb.trace);
    rb.trace.clear();
  }
  rb.state = Readback::State::eWaiting;
  waiting  = &rb;
  lock.unlock();
  rawCv.notify_all();
}

void FrameStream::Impl::handled()
{
  {
    std::lock_guard lock(jpgMutex);
    jpgHandled++;
  }
  jpgCv.notify_all();
}

void FrameStream::Impl::encodeLoop()
{
  tjhandle tj = tj3Init(TJINIT_COMPRESS);
  if(!tj)
  {
    LOGE("frame stream: could not create the JPEG encoder\n");
    return;
  }
  int appliedQuality = -1;
  // The fast integer DCT: a tenth to a fifth off the encode, no difference a viewer can see at
  // these qualities.
  tj3Set(tj, TJPARAM_FASTDCT, 1);

  bool lastFullChroma = false;
  int  lastQuality    = -1;
  // Release a buffer to the ring, waking a destroyRing() that waits for the encode to end
  auto release = [&](Readback* rb, Readback::State state) {
    {
      std::lock_guard lock(rawMutex);
      rb->state = state;
    }
    rawCv.notify_all();
  };
  while(running)
  {
    Readback*    cur  = nullptr;
    Readback*    last = nullptr;
    LatencyTrace trace;
    int64_t      inputAt = 0;
    {
      std::unique_lock lock(rawMutex);
      rawCv.wait(lock, [&] { return waiting != nullptr || !running; });
      if(!running)
        break;
      cur        = waiting;
      waiting    = nullptr;
      cur->state = Readback::State::eEncoding;
      last       = lastEncoded;  // stays allocated while cur is encoding: destroyRing() waits for that
      trace.swap(rawTrace);
      inputAt    = rawInputNs;
      rawInputNs = 0;
    }
    const VkExtent2D extent      = cur->extent;
    const size_t     bytes       = size_t(extent.width) * extent.height * 4;
    const int        wantQuality = quality.load();
    const bool       wantFull    = fullChroma.load();
    // A converged path trace repeats the same image frame after frame; don't re-send it, except
    // once with full chroma, or when the quality rose above the one it went out at, since the frame
    // that was sent was encoded for speed. A quality lowered while it stands (the first touch after
    // a still) is no reason to encode the image already on screen again. A frame the renderer drew
    // is never a repeat, so the compare runs only while the drawn count stands still (a still, an
    // idle loop), where a tonemapper or denoiser change can still alter the image without a draw.
    const bool sameDrawn = !info.drawn || (last != nullptr && cur->drawn == last->drawn);
    const bool repeat    = last != nullptr && sameDrawn && extent.width == last->extent.width
                        && extent.height == last->extent.height && std::memcmp(cur->mapped, last->mapped, bytes) == 0;
    if(repeat && lastFullChroma && wantQuality <= lastQuality)
    {
      if(!trace.empty())
        LOGI("%s (frame repeated, not re-sent)\n", formatTrace(trace).c_str());
      release(cur, Readback::State::eFree);
      handled();
      continue;
    }

    const auto     start = std::chrono::steady_clock::now();
    unsigned char* jpeg  = nullptr;
    size_t         size  = 0;
    if(wantQuality != appliedQuality)
    {
      tj3Set(tj, TJPARAM_QUALITY, wantQuality);
      appliedQuality = wantQuality;
    }
    const bool full = repeat || wantFull;
    tj3Set(tj, TJPARAM_SUBSAMP, full ? TJSAMP_444 : TJSAMP_420);
    if(tj3Compress8(tj, static_cast<const unsigned char*>(cur->mapped), int(extent.width), 0, int(extent.height), TJPF_RGBX,
                    &jpeg, &size)
       != 0)
    {
      LOGW("frame stream: JPEG encoding failed: %s\n", tj3GetErrorStr(tj));
      tj3Free(jpeg);
      release(cur, Readback::State::eFree);
      handled();
      continue;
    }
    auto out = std::make_shared<std::string>(reinterpret_cast<const char*>(jpeg), size);
    tj3Free(jpeg);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    encodeMs        = encodeMs == 0.0 ? ms : encodeMs * 0.9 + ms * 0.1;

    // This frame is now the last encoded; the previous one goes back to the ring
    {
      std::lock_guard lock(rawMutex);
      if(lastEncoded != nullptr)
        lastEncoded->state = Readback::State::eFree;
      lastEncoded = cur;
      cur->state  = Readback::State::eLast;
    }
    rawCv.notify_all();
    lastFullChroma = full;
    lastQuality    = wantQuality;
    {
      std::lock_guard lock(jpgMutex);
      jpg       = std::move(out);
      jpgExtent = extent;
      jpgSeq++;
      jpgHandled++;
    }
    framesEncoded++;
    jpgCv.notify_all();
    if(inputAt != 0)
    {
      const double inMs = double(steadyNs() - inputAt) * 1e-6;
      inputMsLast       = inMs;
      inputMs           = inputSamples++ == 0 ? inMs : inputMs * 0.8 + inMs * 0.2;
    }
    if(!trace.empty())
    {
      trace.emplace_back("encoded", traceNowMs());
      LOGI("%s\n", formatTrace(trace).c_str());
    }
  }
  tj3Destroy(tj);
}

void FrameStream::Impl::serve()
{
  http.set_default_headers({{"Access-Control-Allow-Origin", "*"}, {"Cache-Control", "no-store"}});
  http.set_payload_max_length(BlobStore::kMaxBytes);
  // a frame is one write; Nagle would hold its tail for the next one
  http.set_tcp_nodelay(true);
  // Every request but a CORS preflight carries the token when one is set. A request that changes
  // something (a command, input, a blob, a resize, a client arriving) also ends an idle loop.
  http.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
    if(req.method == "OPTIONS")
      return httplib::Server::HandlerResponse::Unhandled;
    if(!info.token.empty() && req.get_param_value("t") != info.token)
    {
      res.status = 403;
      return httplib::Server::HandlerResponse::Handled;
    }
    if(req.method == "POST" || req.path == "/size" || req.path == "/stream")
      wake();
    if(req.path == "/input")
    {
      int64_t none = 0;  // the oldest input not yet shown is the one the latency counts from
      inputNs.compare_exchange_strong(none, steadyNs());
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  http.Get("/stream", [this](const httplib::Request&, httplib::Response& res) {
    clients++;
    uint64_t seen = 0;
    res.set_content_provider(
        "multipart/x-mixed-replace; boundary=frame",
        [this, seen, first = true](size_t, httplib::DataSink& sink) mutable -> bool {
          std::shared_ptr<const std::string> frame;
          {
            std::unique_lock lock(jpgMutex);
            if(!jpgCv.wait_for(lock, 500ms, [&] { return jpgSeq > seen || !running; }))
              return sink.is_writable();  // nothing new yet; keep the connection
            if(!running)
              return false;
            seen  = jpgSeq;
            frame = jpg;
          }
          // The boundary that ends a part follows the frame's bytes at once, so a browser that
          // waits for it shows the frame now rather than when the next one arrives; the opening
          // boundary goes once, on connect. Head, frame and tail go as one write.
          std::string part = first ? "--frame\r\n" : "";
          first            = false;
          part += "Content-Type: image/jpeg\r\nContent-Length: " + std::to_string(frame->size()) + "\r\n\r\n";
          part += *frame;
          part += "\r\n--frame\r\n";
          if(!sink.write(part.data(), part.size()))
            return false;
          framesSent++;
          return true;
        },
        [this](bool) { clients--; });
  });

  http.Get("/frame", [this](const httplib::Request&, httplib::Response& res) {
    std::shared_ptr<const std::string> frame;
    // Where frames are not copied out on their own (a local viewer and no client), ask for one and
    // wait for the encoder to finish it; a frame that repeats the last one sent leaves that one.
    const bool wait = present_policy::frameWaits(viewers());
    uint64_t   seen = 0;
    if(wait)
    {
      {
        std::lock_guard lock(jpgMutex);
        seen = jpgHandled;
      }
      frameWanted = true;
      wake();
    }
    {
      std::unique_lock lock(jpgMutex);
      if(wait)
        jpgCv.wait_for(lock, 2s, [&] { return jpgHandled > seen || !running; });
      frame = jpg;
    }
    if(!frame)
    {
      res.status = 204;
      return;
    }
    res.set_content(*frame, "image/jpeg");
  });

  http.Get("/size", [this](const httplib::Request& req, httplib::Response& res) {
    const int w = req.has_param("w") ? std::atoi(req.get_param_value("w").c_str()) : 0;
    const int h = req.has_param("h") ? std::atoi(req.get_param_value("h").c_str()) : 0;
    if(w < 16 || h < 16 || w > 16384 || h > 16384)
    {
      res.status = 400;
      res.set_content("w and h must be 16..16384", "text/plain");
      return;
    }
    // a frame the GPU memory cannot hold would end the process at its first allocation
    if(info.refuseSize)
      if(const std::string why = info.refuseSize({uint32_t(w), uint32_t(h)}); !why.empty())
      {
        res.status = 507;
        res.set_content(why, "text/plain");
        return;
      }
    app->requestHeadlessResize({uint32_t(w), uint32_t(h)});
    res.set_content("ok", "text/plain");
  });

  // A host's binary payload for a live edit (blob_store.hpp): stored, and its id returned, so the
  // edit can name it instead of carrying megabytes of base64 through JSON.
  http.Options("/blob", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Methods", "POST");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
  });
  http.Post("/blob", [this](const httplib::Request& req, httplib::Response& res) {
    if(!info.upload)
    {
      res.status = 404;
      return;
    }
    const uint64_t id = info.upload(req.body);
    res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
  });

  http.Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
    VkExtent2D extent{};
    {
      std::lock_guard lock(jpgMutex);
      extent = jpgExtent;
    }
    std::string profile;
    {
      std::lock_guard lock(profileMutex);
      profile = profileJson;
    }
    const std::string body = "{\"rendered\":" + std::to_string(framesRendered.load()) + ",\"encoded\":" + std::to_string(framesEncoded.load())
                             + ",\"sent\":" + std::to_string(framesSent.load()) + ",\"replaced\":" + std::to_string(framesReplaced.load())
                             + ",\"skipped\":" + std::to_string(framesSkipped.load())
                             + ",\"width\":" + std::to_string(extent.width)
                             + ",\"height\":" + std::to_string(extent.height) + ",\"fps\":" + std::to_string(renderFps.load())
                             + ",\"encodeMs\":" + std::to_string(encodeMs.load()) + ",\"clients\":" + std::to_string(clients.load())
                             + ",\"idle\":" + (idle.load() ? "true" : "false")
                             + ",\"inputMs\":" + std::to_string(inputMs.load()) + ",\"inputMsLast\":" + std::to_string(inputMsLast.load())
                             + ",\"inputSamples\":" + std::to_string(inputSamples.load()) + ",\"profile\":" + profile + "}";
    res.set_content(body, "application/json");
  });

  if(info.routes)
    info.routes(http, running);

  if(!http.bind_to_port("127.0.0.1", info.port))
  {
    LOGE("frame stream: could not bind 127.0.0.1:%u\n", unsigned(info.port));
    // nothing reaches a renderer without its port: end the run, so the host sees the exit and the log
    app->close();
    return;
  }
  LOGI("frame stream: http://127.0.0.1:%u/stream\n", unsigned(info.port));
  http.listen_after_bind();
}
