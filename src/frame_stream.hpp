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
// Headless frame server (--stream): a host application embeds this renderer as a render process.
// Every frame the tonemapped output is copied to the CPU, JPEG-encoded (libjpeg-turbo) on a worker
// thread and pushed to HTTP clients as a multipart/x-mixed-replace (MJPEG) stream, which a browser
// <img> displays as is. The host drives everything else (camera, scene, settings) over --mcp.
//
// Changing frames are encoded with 4:2:0 chroma, which halves the encode time. Once the image
// stops changing (a converged path trace), that frame is sent once more with full 4:4:4 chroma.
// The host can also ask for full chroma and a higher quality on every frame (`streamFullChroma`,
// `streamQuality` over --mcp) while a still view refines, so the frames it watches converge
// are not chroma-subsampled.
//
//   GET /stream         MJPEG stream of every new frame (unchanged frames are not re-sent)
//   GET /frame          the latest frame as one image/jpeg
//   GET /size?w=&h=     resize the offscreen viewport (applied before the next frame)
//   GET /stats          JSON: frames rendered/encoded/sent, size, render fps, average encode
//                       time in ms, connected clients
//
// Headless only: there is no window, and the viewport is whatever --size and /size say.
//

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <nvapp/application.hpp>

#include "latency_trace.hpp"

struct FrameStreamCreateInfo
{
  uint16_t port{7672};
  int      quality{92};  // JPEG quality 1-100
  int      maxFps{60};   // frame-rate cap while a client is connected (10 fps without one)

  // The image to stream and its size: the renderer's tonemapped output, in
  // VK_IMAGE_LAYOUT_GENERAL after the renderer's onRender. Called on the application thread.
  std::function<VkImage()>    image;
  std::function<VkExtent2D()> imageSize;

  // True while the renderer accumulates a still, where every frame improves the image. With a
  // client connected such frames run unpaced; the stream still gets maxFps. Optional.
  std::function<bool()> refining;

  // Stores a binary payload a host posts to /blob for a later live edit and returns its id
  // (blob_store.hpp). Called on an HTTP thread. Optional: without it /blob answers 404.
  std::function<uint64_t(std::string)> upload;
};

class FrameStream : public nvapp::IAppElement
{
public:
  explicit FrameStream(FrameStreamCreateInfo info);
  ~FrameStream() override;

  FrameStream(const FrameStream&)            = delete;
  FrameStream& operator=(const FrameStream&) = delete;

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onUpdate() override;
  void onRender(VkCommandBuffer cmd) override;

  // Live encoder settings (the `streamQuality` and `streamFullChroma` parameters): a host lowers
  // them while the image moves and raises them while a still view converges. Thread-safe; the
  // next frame encoded picks them up, and a converged still is re-sent once when they rise.
  void setQuality(int quality);
  void setFullChroma(bool fullChroma);
  // The next frame copied out carries this latency trace and logs it once encoded (application thread).
  void traceNextFrame(LatencyTrace trace);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};
