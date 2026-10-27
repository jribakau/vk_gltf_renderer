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

#include <volk.h>

#include "frame_present.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>
#include <vector>

#include <nvutils/logger.hpp>

using namespace std::chrono_literals;

namespace {

// A string as a JSON string literal
std::string jsonString(const std::string& s)
{
  std::string out = "\"";
  for(const char c : s)
  {
    if(c == '"' || c == '\\')
      out += '\\', out += c;
    else if(static_cast<unsigned char>(c) < 0x20)
    {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(c));
      out += buf;
    }
    else
      out += c;
  }
  return out + "\"";
}

}  // namespace

struct FramePresent::Impl
{
  using Clock = std::chrono::steady_clock;

  FramePresentCreateInfo info;
  nvapp::Application*    app{};

  // Application thread only.
  std::unique_ptr<PresentTarget> target;
  PresentSlots                   slots;
  bool                           visible{true};
  // the frame copied in onRender, presented in onPostRender
  int               pendingSlot{-1};
  uint64_t          pendingValue{0};
  PresentSize       pendingSize;
  PresentSize       lastSize;
  uint64_t          lastDrawn{~uint64_t(0)};
  Clock::time_point lastPresent{};

  // Work for the application thread, from the HTTP threads (onUpdate runs it).
  std::mutex                         mailMutex;
  std::vector<std::function<void()>> mail;
  std::atomic<bool>                  running{false};

  // What GET /present reports: application thread -> HTTP threads.
  std::mutex            statusMutex;
  std::string           state{"detached"};
  std::string           error;
  PresentSize           size;
  std::atomic<uint64_t> presented{0};
  std::atomic<uint64_t> dropped{0};
  std::atomic<uint64_t> skipped{0};

  // HTTP thread: runs `work` on the application thread and returns what it returns.
  std::string run(std::function<std::string()> work);
  std::string attach(uintptr_t surface);
  // Back to the stream alone; `why` is the error that ended presenting, if one did.
  void        detach(const std::string& why);
  void        setVisible(bool on);
  void        setStatus(const std::string& st, const std::string& err);
  std::string statusJson();
};

FramePresent::FramePresent(FramePresentCreateInfo info)
    : m_impl(std::make_unique<Impl>())
{
  m_impl->info = std::move(info);
}

FramePresent::~FramePresent() = default;

void FramePresent::onAttach(nvapp::Application* app)
{
  m_impl->app     = app;
  m_impl->running = true;
}

void FramePresent::onDetach()
{
  Impl& s   = *m_impl;
  s.running = false;
  // anything still waiting in the mailbox is answered (run() sees `running` false and stops waiting)
  {
    std::lock_guard lock(s.mailMutex);
    s.mail.clear();
  }
  if(s.target)
  {
    vkDeviceWaitIdle(s.app->getDevice());
    s.target.reset();
  }
}

// Called with the device idle (start-up, headless resizes): a frame larger than the shared images
// grows them, which needs both sides idle.
void FramePresent::onResize(VkCommandBuffer /*cmd*/, const VkExtent2D& extent)
{
  Impl& s = *m_impl;
  if(!s.target)
    return;
  const PresentSize want{extent.width, extent.height};
  if(fits(s.target->capacity(), want))
    return;
  if(const std::string err = s.target->grow(growCapacity(s.target->capacity(), want)); !err.empty())
  {
    s.detach(err);
    return;
  }
  s.slots.reset();
}

void FramePresent::onUpdate()
{
  Impl&                              s = *m_impl;
  std::vector<std::function<void()>> work;
  {
    std::lock_guard lock(s.mailMutex);
    work.swap(s.mail);
  }
  for(auto& w : work)
    w();
}

void FramePresent::onRender(VkCommandBuffer cmd)
{
  Impl& s       = *m_impl;
  s.pendingSlot = -1;
  if(!s.target || !s.visible || !s.info.image || !s.info.imageSize)
    return;
  const VkImage     image  = s.info.image();
  const VkExtent2D  extent = s.info.imageSize();
  const PresentSize size{extent.width, extent.height};
  if(image == VK_NULL_HANDLE || size.width == 0 || size.height == 0 || !fits(s.target->capacity(), size))
    return;  // a resize on its way grows the images (onResize)

  // An unchanged image is presented again only now and then: a change that reached it without a
  // draw (a tonemapper setting) still shows within a tenth of a second.
  const uint64_t drawn = s.info.drawn ? s.info.drawn() : s.lastDrawn + 1;
  const auto     now   = Impl::Clock::now();
  if(drawn == s.lastDrawn && size == s.lastSize && now - s.lastPresent < 100ms)
    return;

  const int slot = s.slots.acquire(s.target->completed());
  if(slot < 0)
  {
    s.dropped++;
    return;
  }
  // The loop is paced by the stream (or not at all while a still refines); a frame the swapchain
  // has no room for is skipped, and the next draw, or the re-present above, goes instead.
  if(!s.target->presentable())
  {
    s.skipped++;
    return;
  }
  const VkImage  dst    = s.target->slotImage(slot);
  const uint32_t family = s.app->getQueue(0).familyIndex;

  // The renderer's image: finish writing it, copy, hand it back as FrameStream does.
  VkImageMemoryBarrier2 srcToTransfer{
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
  // The shared image: its old contents are not needed, so no acquire from D3D's side
  VkImageMemoryBarrier2 dstToTransfer{
      .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask     = VK_PIPELINE_STAGE_2_NONE,
      .srcAccessMask    = VK_ACCESS_2_NONE,
      .dstStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .image            = dst,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  VkImageMemoryBarrier2 before[] = {srcToTransfer, dstToTransfer};
  VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 2, .pImageMemoryBarriers = before};
  vkCmdPipelineBarrier2(cmd, &dep);

  const VkImageCopy2 region{
      .sType          = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .extent         = {size.width, size.height, 1},
  };
  const VkCopyImageInfo2 copy{
      .sType          = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
      .srcImage       = image,
      .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .dstImage       = dst,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount    = 1,
      .pRegions       = &region,
  };
  vkCmdCopyImage2(cmd, &copy);

  VkImageMemoryBarrier2 srcBack = srcToTransfer;
  srcBack.srcStageMask          = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  srcBack.srcAccessMask         = VK_ACCESS_2_TRANSFER_READ_BIT;
  srcBack.dstStageMask          = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  srcBack.dstAccessMask         = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  // ... and the shared image goes to D3D's side
  VkImageMemoryBarrier2 dstRelease{
      .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask        = VK_PIPELINE_STAGE_2_NONE,
      .dstAccessMask       = VK_ACCESS_2_NONE,
      .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = family,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL,
      .image               = dst,
      .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  VkImageMemoryBarrier2 after[] = {srcBack, dstRelease};
  dep.pImageMemoryBarriers      = after;
  vkCmdPipelineBarrier2(cmd, &dep);

  const uint64_t value = s.slots.use(slot);
  s.app->addSignalSemaphore({
      .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = s.target->ready(),
      .value     = value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  });
  s.pendingSlot  = slot;
  s.pendingValue = value;
  s.pendingSize  = size;
  s.lastDrawn    = drawn;
  s.lastSize     = size;
  s.lastPresent  = now;
}

// After the frame's submit: D3D waits for the copy on its own queue and presents.
void FramePresent::onPostRender()
{
  Impl& s = *m_impl;
  if(s.pendingSlot < 0 || !s.target)
    return;
  const int slot = s.pendingSlot;
  s.pendingSlot  = -1;
  if(const std::string err = s.target->present(slot, s.pendingValue, s.pendingSize); !err.empty())
  {
    s.detach(err);
    return;
  }
  s.presented++;
  std::lock_guard lock(s.statusMutex);
  s.size = s.pendingSize;
}

//--------------------------------------------------------------------------------------------------

std::string FramePresent::Impl::run(std::function<std::string()> work)
{
  if(!running)
    return "the renderer is stopping";
  auto promise = std::make_shared<std::promise<std::string>>();
  auto result  = promise->get_future();
  {
    std::lock_guard lock(mailMutex);
    mail.push_back([promise, work = std::move(work)] { promise->set_value(work()); });
  }
  // the application thread takes it at its next update; a loading scene can hold that up a while
  if(result.wait_for(10s) != std::future_status::ready)
    return "the renderer did not take the request in time";
  return result.get();
}

std::string FramePresent::Impl::attach(uintptr_t surface)
{
  if(target)
    detach("");
  const VkExtent2D  extent = info.imageSize ? info.imageSize() : VkExtent2D{};
  PresentTargetInfo ti{
      .instance       = app->getInstance(),
      .physicalDevice = app->getPhysicalDevice(),
      .device         = app->getDevice(),
      .surface        = surface,
      .capacity       = {std::max(extent.width, 16u), std::max(extent.height, 16u)},
  };
  std::string err;
  target = createPresentTarget(ti, err);
  if(!target)
  {
    LOGW("native presentation: %s\n", err.c_str());
    setStatus("detached", err);
    return err;
  }
  slots     = PresentSlots(target->slotCount());
  lastDrawn = ~uint64_t(0);
  LOGI("native presentation: attached, %ux%u\n", ti.capacity.width, ti.capacity.height);
  setStatus(visible ? "attached" : "hidden", "");
  if(info.localViewer)
    info.localViewer(visible);
  return "";
}

void FramePresent::Impl::detach(const std::string& why)
{
  if(target)
  {
    // the frames in flight may still use the shared images
    vkDeviceWaitIdle(app->getDevice());
    target.reset();
  }
  pendingSlot = -1;
  if(info.localViewer)
    info.localViewer(false);
  if(!why.empty())
    LOGW("native presentation stopped: %s\n", why.c_str());
  setStatus("detached", why);
}

void FramePresent::Impl::setVisible(bool on)
{
  visible = on;
  if(!target)
    return;
  setStatus(on ? "attached" : "hidden", "");
  // hidden, nobody watches: the stream may cap the loop again
  if(info.localViewer)
    info.localViewer(on);
}

void FramePresent::Impl::setStatus(const std::string& st, const std::string& err)
{
  std::lock_guard lock(statusMutex);
  state = st;
  error = err;
}

std::string FramePresent::Impl::statusJson()
{
  std::lock_guard lock(statusMutex);
  return "{\"state\":" + jsonString(state) + ",\"width\":" + std::to_string(size.width)
         + ",\"height\":" + std::to_string(size.height) + ",\"presented\":" + std::to_string(presented.load())
         + ",\"dropped\":" + std::to_string(dropped.load()) + ",\"skipped\":" + std::to_string(skipped.load())
         + ",\"error\":" + (error.empty() ? std::string("null") : jsonString(error)) + "}";
}

void FramePresent::registerRoutes(httplib::Server& http)
{
  Impl& s = *m_impl;
  http.Post("/present/attach", [&s](const httplib::Request& req, httplib::Response& res) {
    uintptr_t surface = 0;
    try
    {
      surface = uintptr_t(std::stoull(req.get_param_value("surface"), nullptr, 0));
    }
    catch(const std::exception&)
    {
    }
    if(surface == 0)
    {
      res.status = 400;
      res.set_content("surface=<handle> is required", "text/plain");
      return;
    }
    const std::string err = s.run([&s, surface] { return s.attach(surface); });
    res.set_content("{\"ok\":" + std::string(err.empty() ? "true" : "false") + ",\"status\":" + s.statusJson() + "}", "application/json");
  });
  http.Post("/present/detach", [&s](const httplib::Request&, httplib::Response& res) {
    s.run([&s] {
      s.detach("");
      return std::string();
    });
    res.set_content(s.statusJson(), "application/json");
  });
  http.Post("/present/visible", [&s](const httplib::Request& req, httplib::Response& res) {
    const bool on = req.get_param_value("on") != "0";
    s.run([&s, on] {
      s.setVisible(on);
      return std::string();
    });
    res.set_content(s.statusJson(), "application/json");
  });
  http.Get("/present",
           [&s](const httplib::Request&, httplib::Response& res) { res.set_content(s.statusJson(), "application/json"); });
}

#ifndef _WIN32
std::unique_ptr<PresentTarget> createPresentTarget(const PresentTargetInfo&, std::string& error)
{
  error = "native presentation is only implemented on Windows";
  return nullptr;
}
#endif
