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

// What a viewport drag does to the camera. FvHost feeds mouseMove() the buttons and modifiers the app
// reports; the manipulator does the motion itself.

#include <nvutils/camera_manipulator.hpp>

namespace camera_controls {

using Manipulator = nvutils::CameraManipulator;

// The motion a drag asks for, as a game engine's editor has it (Unity, Unreal): Alt with the left
// button orbits (Fly and Walk look around instead), the middle button pans with or without Alt, Alt
// with the right button dollies. The left button alone is a click's, which selects, and the right
// alone the host's look around (FvHost::Impl::applyInput), so neither moves the camera here. The left
// button comes first.
inline Manipulator::Actions dragMotion(const Manipulator::Inputs& inputs, Manipulator::Modes mode)
{
  if(inputs.lmb)
  {
    if(!inputs.alt)
      return Manipulator::Actions::NoAction;
    return mode == Manipulator::Examine ? Manipulator::Actions::Orbit : Manipulator::Actions::LookAround;
  }
  if(inputs.mmb)
    return Manipulator::Actions::Pan;
  if(inputs.rmb && inputs.alt)
    return Manipulator::Actions::Dolly;
  return Manipulator::Actions::NoAction;
}

// Whether a drag with these buttons moves the camera (a locked camera refuses it)
inline bool dragMovesCamera(const Manipulator::Inputs& inputs)
{
  return (inputs.lmb && inputs.alt) || inputs.mmb || inputs.rmb;
}

}  // namespace camera_controls

class DragCameraManipulator : public nvutils::CameraManipulator
{
public:
  Actions mouseMove(glm::dvec2 screenPos, const Inputs& inputs) override;
};
