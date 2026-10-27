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

// Who is watching a --stream run, and what that asks of FrameStream (frame_stream.cpp), kept apart
// from Vulkan so it can be tested.
//
// A run serves its frames to MJPEG clients (/stream), to one-off /frame requests, and, once the
// app presents frames natively (roadmap item 1), to a local viewer that is no client of the stream
// at all. Without a local viewer everything is as it always was: frames are copied out and encoded
// whether or not anyone is connected, and nobody connected caps the loop at 10 fps. With one, the
// loop runs at the stream's rate and refines as it would for a client, but the image is copied out
// and encoded only for a client or a /frame waiting for one: 33 MB a frame at 4K for nobody.

#pragma once

#include <algorithm>

namespace present_policy {

struct Viewers
{
  int  clients{0};         // /stream connections
  bool localViewer{false};  // the app presents the frames itself
  bool frameWanted{false};  // a /frame request waits for a frame
};

// The loop's frame-rate cap: the stream's while someone watches, 10 fps while nobody does.
inline int pacedFps(int maxFps, const Viewers& v)
{
  const int fps = std::max(1, maxFps);
  return v.clients > 0 || v.localViewer ? fps : std::min(fps, 10);
}

// A refining still runs unpaced only while someone watches it converge.
inline bool refining(bool renderer, const Viewers& v)
{
  return renderer && (v.clients > 0 || v.localViewer);
}

// Whether a frame the pacer made due is copied out for the encoder.
inline bool copyOut(bool due, const Viewers& v)
{
  return due && (!v.localViewer || v.clients > 0 || v.frameWanted);
}

// Whether a /frame must wait for a frame to be copied and encoded rather than take the last one:
// only where frames are not copied out on their own.
inline bool frameWaits(const Viewers& v)
{
  return v.localViewer && v.clients == 0;
}

}  // namespace present_policy
