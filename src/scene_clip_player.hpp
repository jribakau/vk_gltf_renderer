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
// A clip of the retained scene played (daz-viewer docs/retained-scene-plan.md, "Imported clips"):
// nvvkgltf::AnimationSystem's evaluation ported, with its arithmetic and its edge cases, writing
// entity TRS (setLocal) and morph weights (setMorphWeights), so takeChanges reports what a frame of
// playback moved like any other edit. One player per clip; its time is its own, advanced as
// nvvkgltf::AnimationInfo advances (wrapping, or held at the end).
//
// Where the legacy evaluation leaves a channel alone, so does this: before its first key, after its
// last, and with fewer than two keys; the entity keeps what it had. Exactly on the last key a STEP
// channel gives the next-to-last key's value, as there. One difference on purpose: a STEP rotation
// key is read as the quaternion it is, where AnimationSystem's glm::quat(vec4) makes a rotation
// from the key's x, y and z taken as Euler angles.
//

#include "scene_retained.hpp"

namespace rs {

class ClipPlayer
{
public:
  ClipPlayer() = default;
  // at the clip's first key
  ClipPlayer(const RetainedScene& scene, ClipId clip);

  ClipId clip() const { return m_clip; }
  float  start() const { return m_start; }
  float  end() const { return m_end; }
  float  time() const { return m_time; }
  void   setTime(float time) { m_time = time; }
  float  reset() { return m_time = m_start; }
  // Moves the time on by `deltaTime`: wrapped into the clip's range, or held at its end
  // (AnimationInfo::incrementTime). A clip with no length stays at its start.
  float advance(float deltaTime, bool loop = true);

  // Writes the clip at time() into its entities; false when no channel wrote anything (the clip is
  // gone, its entities are, or every channel is outside its keys).
  bool apply(RetainedScene& scene) const;

private:
  ClipId m_clip;
  float  m_start = 0.f;
  float  m_end   = 0.f;
  float  m_time  = 0.f;
};

}  // namespace rs
