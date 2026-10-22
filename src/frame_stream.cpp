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
#include <httplib.h>

#include <volk.h>

#include "frame_stream.hpp"
#include "frame_pacing.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nvutils/logger.hpp>
#include <turbojpeg.h>

using namespace std::chrono_literals;

namespace {

// One host-visible buffer per frame in flight. The copy recorded in frame N is read back when the
// ring slot comes around again: the application waits for that frame to complete before it
// re-records the slot's command buffer, so by then the data is there.
struct Readback
{
  VkBuffer       buffer{};
  VkDeviceMemory memory{};
  void*          mapped{};
  VkDeviceSize   size{};
  VkExtent2D     extent{};
  bool           coherent{true};
  bool           pending{false};
};

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
  std::vector<Readback> ring;

  // raw frames: application thread -> encoder thread (one slot; a frame is dropped if the
  // encoder is still busy with the previous one)
  std::mutex              rawMutex;
  std::condition_variable rawCv;
  std::vector<uint8_t>    rawPending;
  VkExtent2D              rawExtent{};
  bool                    rawReady{false};

  // encoded frames: encoder thread -> HTTP client threads
  std::mutex                         jpgMutex;
  std::condition_variable            jpgCv;
  std::shared_ptr<const std::string> jpg;
  uint64_t                           jpgSeq{0};
  VkExtent2D                         jpgExtent{};

  std::atomic<bool>     running{false};
  std::atomic<int>      clients{0};
  std::atomic<uint64_t> framesRendered{0};
  std::atomic<uint64_t> framesEncoded{0};
  std::atomic<uint64_t> framesSent{0};
  std::atomic<double>   renderFps{0.0};
  std::atomic<double>   encodeMs{0.0};

  std::thread     encoder;
  std::thread     server;
  httplib::Server http;

  std::chrono::steady_clock::time_point lastFrame{};
  FramePacer                            pacer{std::chrono::milliseconds(16)};
  // this frame's image goes to the stream (set in onUpdate, read in onRender)
  bool copyFrame{true};

  void createRing(const VkExtent2D& extent);
  void destroyRing();
  void harvest(Readback& rb);
  void encodeLoop();
  void serve();
};

//--------------------------------------------------------------------------------------------------

FrameStream::FrameStream(FrameStreamCreateInfo info)
    : m_impl(std::make_unique<Impl>())
{
  m_impl->info = std::move(info);
}

FrameStream::~FrameStream() = default;

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
  s.encoder = std::thread([&s] { s.encodeLoop(); });
  s.server  = std::thread([&s] { s.serve(); });
}

void FrameStream::onDetach()
{
  Impl& s = *m_impl;
  if(s.running.exchange(false))
  {
    s.http.stop();
    s.rawCv.notify_all();
    s.jpgCv.notify_all();
    if(s.encoder.joinable())
      s.encoder.join();
    if(s.server.joinable())
      s.server.join();
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

// Frame pacing. A headless run has no v-sync and would otherwise spin as fast as the GPU allows;
// the cap also keeps the renderer from starving the host while nobody is watching.
void FrameStream::onUpdate()
{
  Impl& s = *m_impl;
  if(!s.running)
    return;
  const int  fps    = s.clients > 0 ? std::max(1, s.info.maxFps) : std::min(std::max(1, s.info.maxFps), 10);
  const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / fps));
  // Nobody watching: keep to the cap, the host needs the machine more than a still needs to converge.
  const bool refining = s.clients > 0 && s.info.refining && s.info.refining();
  s.pacer.setPeriod(period);
  auto       now   = std::chrono::steady_clock::now();
  const auto start = s.pacer.startTime(now, refining);
  if(now < start)
  {
    std::this_thread::sleep_until(start);
    now = std::chrono::steady_clock::now();
  }
  if(s.lastFrame.time_since_epoch().count() != 0)
  {
    const double dt = std::chrono::duration<double>(now - s.lastFrame).count();
    if(dt > 0.0)
      s.renderFps = s.renderFps * 0.9 + (1.0 / dt) * 0.1;
  }
  s.lastFrame = now;
  s.copyFrame = s.pacer.begin(now);
}

void FrameStream::onRender(VkCommandBuffer cmd)
{
  Impl& s = *m_impl;
  if(!s.running || s.ring.empty() || !s.info.image || !s.info.imageSize)
    return;
  const uint32_t slot = s.app->getFrameCycleIndex();
  if(slot >= s.ring.size())
    return;
  Readback& rb = s.ring[slot];
  if(rb.pending)
  {
    rb.pending = false;
    s.harvest(rb);
  }
  // an unpaced frame the stream is not due: nothing to copy out
  if(!s.copyFrame)
    return;

  const VkImage    image  = s.info.image();
  const VkExtent2D extent = s.info.imageSize();
  if(image == VK_NULL_HANDLE || extent.width == 0 || extent.height == 0)
    return;
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

  rb.extent  = extent;
  rb.pending = true;
  s.framesRendered++;
}

//--------------------------------------------------------------------------------------------------

void FrameStream::Impl::createRing(const VkExtent2D& extent)
{
  if(extent.width == 0 || extent.height == 0)
    return;
  const VkDeviceSize size = VkDeviceSize(extent.width) * extent.height * 4;
  ring.resize(std::max(1u, app->getFrameCycleSize()));
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
    rb.size    = size;
    rb.extent  = extent;
    rb.pending = false;
  }
}

void FrameStream::Impl::destroyRing()
{
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

// Application thread: the slot's copy has completed. Hand the pixels to the encoder unless it is
// still busy with the previous frame, in which case this one is dropped (the next will do).
void FrameStream::Impl::harvest(Readback& rb)
{
  std::unique_lock lock(rawMutex);
  if(rawReady)
    return;
  const size_t bytes = size_t(rb.extent.width) * rb.extent.height * 4;
  if(!rb.coherent)
  {
    const VkMappedMemoryRange range{.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = rb.memory, .size = VK_WHOLE_SIZE};
    vkInvalidateMappedMemoryRanges(device, 1, &range);
  }
  rawPending.resize(bytes);
  std::memcpy(rawPending.data(), rb.mapped, bytes);
  rawExtent = rb.extent;
  rawReady  = true;
  lock.unlock();
  rawCv.notify_one();
}

void FrameStream::Impl::encodeLoop()
{
  tjhandle tj = tj3Init(TJINIT_COMPRESS);
  if(!tj)
  {
    LOGE("frame stream: could not create the JPEG encoder\n");
    return;
  }
  tj3Set(tj, TJPARAM_QUALITY, info.quality);

  std::vector<uint8_t> raw;
  std::vector<uint8_t> last;
  VkExtent2D           lastExtent{};
  bool                 lastFullChroma = false;
  while(running)
  {
    VkExtent2D extent{};
    {
      std::unique_lock lock(rawMutex);
      rawCv.wait(lock, [&] { return rawReady || !running; });
      if(!running)
        break;
      raw.swap(rawPending);
      extent   = rawExtent;
      rawReady = false;
    }
    // A converged path trace repeats the same image frame after frame; don't re-send it, except
    // once with full chroma, since the frame that was sent was encoded for speed.
    const bool repeat = extent.width == lastExtent.width && extent.height == lastExtent.height
                        && raw.size() == last.size() && std::memcmp(raw.data(), last.data(), raw.size()) == 0;
    if(repeat && lastFullChroma)
      continue;

    const auto     start = std::chrono::steady_clock::now();
    unsigned char* jpeg  = nullptr;
    size_t         size  = 0;
    tj3Set(tj, TJPARAM_SUBSAMP, repeat ? TJSAMP_444 : TJSAMP_420);
    if(tj3Compress8(tj, raw.data(), int(extent.width), 0, int(extent.height), TJPF_RGBX, &jpeg, &size) != 0)
    {
      LOGW("frame stream: JPEG encoding failed: %s\n", tj3GetErrorStr(tj));
      tj3Free(jpeg);
      continue;
    }
    auto out = std::make_shared<std::string>(reinterpret_cast<const char*>(jpeg), size);
    tj3Free(jpeg);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    encodeMs        = encodeMs == 0.0 ? ms : encodeMs * 0.9 + ms * 0.1;

    last.swap(raw);
    lastExtent     = extent;
    lastFullChroma = repeat;
    {
      std::lock_guard lock(jpgMutex);
      jpg       = std::move(out);
      jpgExtent = extent;
      jpgSeq++;
    }
    framesEncoded++;
    jpgCv.notify_all();
  }
  tj3Destroy(tj);
}

void FrameStream::Impl::serve()
{
  http.set_default_headers({{"Access-Control-Allow-Origin", "*"}, {"Cache-Control", "no-store"}});

  http.Get("/stream", [this](const httplib::Request&, httplib::Response& res) {
    clients++;
    uint64_t seen = 0;
    res.set_content_provider(
        "multipart/x-mixed-replace; boundary=frame",
        [this, seen](size_t, httplib::DataSink& sink) mutable -> bool {
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
          const std::string head = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + std::to_string(frame->size()) + "\r\n\r\n";
          if(!sink.write(head.data(), head.size()) || !sink.write(frame->data(), frame->size()) || !sink.write("\r\n", 2))
            return false;
          framesSent++;
          return true;
        },
        [this](bool) { clients--; });
  });

  http.Get("/frame", [this](const httplib::Request&, httplib::Response& res) {
    std::shared_ptr<const std::string> frame;
    {
      std::lock_guard lock(jpgMutex);
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
    app->requestHeadlessResize({uint32_t(w), uint32_t(h)});
    res.set_content("ok", "text/plain");
  });

  http.Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
    VkExtent2D extent{};
    {
      std::lock_guard lock(jpgMutex);
      extent = jpgExtent;
    }
    const std::string body = "{\"rendered\":" + std::to_string(framesRendered.load()) + ",\"encoded\":" + std::to_string(framesEncoded.load())
                             + ",\"sent\":" + std::to_string(framesSent.load()) + ",\"width\":" + std::to_string(extent.width)
                             + ",\"height\":" + std::to_string(extent.height) + ",\"fps\":" + std::to_string(renderFps.load())
                             + ",\"encodeMs\":" + std::to_string(encodeMs.load()) + ",\"clients\":" + std::to_string(clients.load()) + "}";
    res.set_content(body, "application/json");
  });

  if(!http.bind_to_port("127.0.0.1", info.port))
  {
    LOGE("frame stream: could not bind 127.0.0.1:%u\n", unsigned(info.port));
    return;
  }
  LOGI("frame stream: http://127.0.0.1:%u/stream\n", unsigned(info.port));
  http.listen_after_bind();
}
