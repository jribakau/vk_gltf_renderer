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
// Every frame the tonemapped output is copied into one of a ring of host-visible buffers, which a
// worker thread JPEG-encodes in place (libjpeg-turbo; no copy of the pixels on the application
// thread) and pushes to HTTP clients as a multipart/x-mixed-replace (MJPEG) stream, which a browser
// <img> displays as is. The host drives everything else (camera, scene, settings) over --mcp.
//
// Changing frames are encoded with 4:2:0 chroma, which halves the encode time. Once the image
// stops changing (a converged path trace), that frame is sent once more with full 4:4:4 chroma.
// While the renderer accumulates or holds a still (`still`), frames go out at full chroma and at
// least `stillQuality`, so the frames a viewer watches converge are not chroma-subsampled; the
// host's `streamQuality` and `streamFullChroma` are what moving frames get.
//
//   GET /stream         MJPEG stream of every new frame (unchanged frames are not re-sent)
//   GET /frame          the latest frame as one image/jpeg
//   GET /size?w=&h=     resize the offscreen viewport (applied before the next frame)
//   GET /stats          JSON: frames rendered/encoded/sent, size, render fps, average encode
//                       time in ms, connected clients, whether the loop is idle
//
// The element also paces the headless loop (frame_pacing.hpp) and waits for the GPU there, before
// the host element behind it takes its input; copies that completed are handed to the encoder
// during that wait rather than when their ring slot comes round again. When nothing has drawn and
// no request has come in for a while the loop slows to `idleFps`; any request wakes it.
//
// A host can add routes of its own to the same server (`routes`, e.g. fv_protocol.hpp), and
// require a token: with `token` set, every request but a CORS preflight must carry ?t=<token>.
//
// Headless only: there is no window, and the viewport is whatever --size and /size say.
//

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <nvapp/application.hpp>

#include "latency_trace.hpp"

namespace httplib {
class Server;
}

struct FrameStreamCreateInfo
{
  uint16_t port{7672};
  int      quality{92};  // JPEG quality 1-100
  int      maxFps{60};   // frame-rate cap while a client is connected (10 fps without one)
  // frames a second the stream gets while a still refines (its frames run unpaced): fewer than
  // maxFps, so the encoder is idle when the next drag's frames arrive
  int refineFps{20};
  // the loop's rate once nothing has drawn and nothing has come in for half a second
  int idleFps{10};
  // JPEG quality while `still` says so (at least; the host's quality when higher), at full chroma
  int stillQuality{95};

  // The image to stream and its size: the renderer's tonemapped output, in
  // VK_IMAGE_LAYOUT_GENERAL after the renderer's onRender. Called on the application thread.
  std::function<VkImage()>    image;
  std::function<VkExtent2D()> imageSize;

  // True while the renderer accumulates a still, where every frame improves the image. With a
  // client connected such frames run unpaced; the stream still gets maxFps. Optional.
  std::function<bool()> refining;

  // Frames the renderer has drawn so far: while the count stands still and no request comes in,
  // the loop slows to idleFps. Optional: without it the loop never idles.
  std::function<uint64_t()> drawn;

  // True while the renderer accumulates or holds a still: its frames are encoded at full chroma
  // and stillQuality. Optional: without it the host's settings apply to every frame.
  std::function<bool()> still;

  // Stores a binary payload a host posts to /blob for a later live edit and returns its id
  // (blob_store.hpp). Called on an HTTP thread. Optional: without it /blob answers 404.
  std::function<uint64_t(std::string)> upload;

  // Required on every request (?t=) when not empty: the server is on loopback, but any local
  // process or web page could reach it otherwise.
  std::string token;

  // Adds a host's own routes before the server starts. `running` turns false when it stops, which
  // is when long-lived responses (event streams) must end. Optional.
  std::function<void(httplib::Server&, const std::atomic<bool>& running)> routes;

  // Called first when the stream stops, before the server joins its threads: a request a host holds
  // waiting on the application thread (a command) must end now, not at its own timeout. Optional.
  std::function<void()> stopping;

  // Why a /size must be refused (GPU memory the frame would not have), or empty. Called on an HTTP
  // thread, so it may read only what the renderer keeps for any thread. Optional.
  std::function<std::string(VkExtent2D)> refuseSize;
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

  // Live encoder settings (the `streamQuality` and `streamFullChroma` parameters) for the frames
  // of a moving view; a still's are stillQuality at full chroma. Thread-safe; the next frame
  // encoded picks them up, and a converged still is re-sent once when they rise.
  void setQuality(int quality);
  void setFullChroma(bool fullChroma);
  // The next frame copied out carries this latency trace and logs it once encoded (application thread).
  void traceNextFrame(LatencyTrace trace);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};
