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
// Playback state: play/pause, clip selection, scrub, reset, speed. State owner:
// Resources::animationControl; GltfRenderer::advanceHostClip drives it every frame.
//

#include "animation_control.hpp"

#include <imgui/imgui.h>


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


