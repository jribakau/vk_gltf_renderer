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

//
// Animation UI for the glTF renderer: AnimationControl (playback state + domain logic --
// play/pause, clip selection, scrub, frame step, reset, speed; no ImGui, so other call sites can
// read it without pulling in the UI layer) and ui::animation::renderStripWidgets, the transport
// row overlaid at the bottom of the viewport. State owner: Resources::animationControl;
// GltfRenderer::updateAnimation drives it every frame.
//

#include "animation_control.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <imgui/imgui.h>

#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"

//=============================================================================
// AnimationControl -- state + domain logic
//=============================================================================

// The 1/60 second step for single-frame advancement matches the nominal rate
// used by glTF viewers and keeps frame-step behavior independent of display FPS.
static constexpr float kStepDeltaSeconds = 1.0F / 60.0F;

float AnimationControl::deltaTime() const
{
  // runOnce produces a fixed-size step; continuous playback scales real delta time.
  return runOnce ? speed * kStepDeltaSeconds : ImGui::GetIO().DeltaTime * speed;
}

void AnimationControl::scrubTo(float time, nvvkgltf::Scene* scene)
{
  if(!scene || !scene->animation().hasAnimation())
    return;

  const int nAnim = scene->animation().getNumAnimations();
  if(currentAnimation < 0 || currentAnimation >= nAnim)
    return;

  nvvkgltf::AnimationInfo& info = scene->animation().getAnimationInfo(currentAnimation);
  info.currentTime              = std::clamp(time, info.start, info.end);
  runOnce                       = true;   // Request a single evaluation on next update
  play                          = false;  // Scrubbing implies pause
}

std::string AnimationControl::formatTime(float seconds)
{
  if(!std::isfinite(seconds) || seconds < 0.0F)
    seconds = 0.0F;
  const int total   = static_cast<int>(seconds);
  const int minutes = total / 60;
  const int secs    = total % 60;
  const int ms      = static_cast<int>((seconds - static_cast<float>(total)) * 1000.0F);
  char      buffer[16];
  std::snprintf(buffer, sizeof(buffer), "%02d:%02d.%03d", minutes, secs, ms);
  return buffer;
}

//=============================================================================
// Animation Strip widget
//=============================================================================

namespace ui::animation {

bool hasPlayableAnimation(nvvkgltf::Scene* scene)
{
  if(!scene || !scene->animation().hasAnimation())
    return false;

  // A clip is "playable" only when its timeline spans a meaningful positive
  // duration. Require the span to exceed a microsecond: well below any real
  // animation frame (~16 ms at 60 fps), well above numerical noise. This
  // defends against malformed files with all keyframes at the same time,
  // NaN-poisoned timelines (all comparisons return false), and any future
  // regression of the default sentinel in AnimationInfo.
  constexpr float            kMinPlayableDurationSeconds = 1e-6F;
  nvvkgltf::AnimationSystem& system                      = scene->animation();
  for(int i = 0; i < system.getNumAnimations(); ++i)
  {
    const nvvkgltf::AnimationInfo& info = system.getAnimationInfo(i);
    if(info.end - info.start > kMinPlayableDurationSeconds)
      return true;
  }
  return false;
}

}  // namespace ui::animation
