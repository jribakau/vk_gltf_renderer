/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// AnimationControl: playback state and its domain logic (play/pause, clip selection, scrub, reset,
// speed), and whether a scene has anything to play.
//
// State owner: Resources::animationControl. The renderer's update path
// (GltfRenderer::updateAnimation) reads the queries (doAnimation, deltaTime,
// isReset) and calls clearStates() after consuming a one-shot step/reset.
// GltfRenderer::cleanupScene resets the struct to defaults on scene load, then
// sets `play` from GltfRenderer::m_animPlay -- the user's last explicit
// Play/Pause choice, persisted as --animPlay.

#include <string>



namespace nvvkgltf {
class Scene;
}

//-----------------------------------------------------------------------------
// Playback state + domain logic
//-----------------------------------------------------------------------------

struct AnimationControl
{
  // State
  bool  play             = true;   // Continuous playback
  bool  runOnce          = false;  // One-shot step (timeline scrub or frame advance)
  bool  reset            = false;  // Jump back to start on next update
  float speed            = 1.0F;   // Playback multiplier
  int   currentAnimation = 0;      // Index into AnimationSystem clips
  bool  loop             = true;   // Wrap at the clip end; false stops playback there
  bool  seek             = false;  // One-shot: evaluate the clip at its current time without advancing it

  void togglePlay() { play = !play; }
  void stepOne()
  {
    runOnce = true;
    play    = false;
  }
  void resetToStart() { reset = true; }

  // Move the current clip to `time`, pause playback, and request a one-shot
  // update so the scene reflects the new pose on the next frame.
  // No-op if the scene has no animation or the selection is out of range.
  void scrubTo(float time, nvvkgltf::Scene* scene);

  // Queries consumed by the animation update path
  [[nodiscard]] bool  doAnimation() const { return play || runOnce || reset || seek; }
  [[nodiscard]] float deltaTime() const;
  [[nodiscard]] bool  isReset() const { return reset; }

  // Called by the animation update path after consuming a one-shot step/reset.
  void clearStates() { runOnce = reset = seek = false; }

  // Format a time value in seconds as mm:ss.ms (used by the time readout).
  static std::string formatTime(float seconds);
};

namespace ui::animation {

// A scene has "playable animation" when at least one clip declares a positive duration.
bool hasPlayableAnimation(nvvkgltf::Scene* scene);

}  // namespace ui::animation
