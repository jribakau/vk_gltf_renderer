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

#pragma once

//
// Native presentation (roadmap item 1): the frames of a --stream run go straight into the host
// app's window instead of through JPEG. The app keeps a DirectComposition visual under its
// transparent webview and hands this process a composition surface handle; the renderer makes a
// D3D11 swapchain on that surface and presents the tonemapped image into it every frame, through
// images it shares with D3D11 (present_slots.hpp has the hand-off). Nothing is read back to the CPU.
//
// The routes, on the frame stream's server (with its token):
//   POST /present/attach?surface=<handle>   the surface handle, duplicated into this process.
//                                           Answers {"ok":true,...} once the swapchain presents,
//                                           or {"ok":false,"error":...}; the stream goes on as before
//   POST /present/detach                    back to the stream alone
//   POST /present/visible?on=0|1            the window is minimized or hidden: skip presenting
//   GET  /present                           JSON: state, size, frames presented, dropped (every
//                                           shared image busy) and skipped (the swapchain had no
//                                           room: the next draw goes instead), the last error
//
// While attached and visible the stream is told it has a local viewer (FrameStream::setLocalViewer):
// it stops copying frames out for nobody, and the loop runs at the stream's rate. A failure while
// presenting (a lost device) detaches and records the error; the app falls back to the stream.
//
// Windows only (frame_present_win32.cpp); elsewhere attach answers that it is not supported.
//

#include <functional>
#include <memory>
#include <string>

#include <nvapp/application.hpp>

#include "present_slots.hpp"

namespace httplib {
class Server;
}

// The platform's side: the swapchain and the images Vulkan shares with it.
class PresentTarget
{
public:
  virtual ~PresentTarget() = default;

  virtual int         slotCount() const         = 0;
  virtual VkImage     slotImage(int slot) const = 0;
  virtual PresentSize capacity() const          = 0;  // the shared images' size: at least the frame's
  virtual PresentSize frameSize() const         = 0;  // the swapchain's size: the frame's, exactly
  // Vulkan signals each frame's value on it once the frame's copy into its slot is done
  virtual VkSemaphore ready() const = 0;
  // the last frame value the consumer has finished with (its slot is free again)
  virtual uint64_t completed() = 0;
  // A new frame size: the swapchain takes it exactly, the images grow to hold it. Both sides must
  // be idle: Vulkan's device and every frame handed to present(). Returns an error, or empty.
  virtual std::string resize(PresentSize frame) = 0;
  // After the frame's submit: wait for `value`, show the slot's top left `size`. Returns an error, or empty.
  virtual std::string present(int slot, uint64_t value, PresentSize size) = 0;
  // The swapchain takes another frame now (its frame-latency waitable is signalled). Never blocks:
  // a refining still runs unpaced, and a minimized window takes no frames at all. A true answer
  // claims that frame, so it must be followed by present().
  virtual bool presentable() = 0;
};

struct PresentTargetInfo
{
  VkInstance       instance{};
  VkPhysicalDevice physicalDevice{};
  VkDevice         device{};
  uintptr_t        surface{};  // the composition surface handle, this process's
  PresentSize      frame;      // the frame's size
};

// Null with `error` set when the platform or the GPU can't present natively.
std::unique_ptr<PresentTarget> createPresentTarget(const PresentTargetInfo& info, std::string& error);

struct FramePresentCreateInfo
{
  // The image to present and its size: the renderer's tonemapped output, in
  // VK_IMAGE_LAYOUT_GENERAL after the elements before this one. Called on the application thread.
  std::function<VkImage()>    image;
  std::function<VkExtent2D()> imageSize;
  // Frames the renderer has drawn so far: an unchanged count presents again only now and then.
  std::function<uint64_t()> drawn;
  // The stream has (or no longer has) a local viewer (FrameStream::setLocalViewer). Optional.
  std::function<void(bool)> localViewer;
};

class FramePresent : public nvapp::IAppElement
{
public:
  explicit FramePresent(FramePresentCreateInfo info);
  ~FramePresent() override;

  FramePresent(const FramePresent&)            = delete;
  FramePresent& operator=(const FramePresent&) = delete;

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onUpdate() override;
  void onRender(VkCommandBuffer cmd) override;
  void onPostRender() override;

  // The /present routes, on the frame stream's server (FrameStreamCreateInfo::routes).
  void registerRoutes(httplib::Server& http);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};
