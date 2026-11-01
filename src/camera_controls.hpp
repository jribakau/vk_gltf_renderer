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

// The motion a drag asks for, the left button first. Fly and Walk swap the left button's orbit and
// look around, and Alt swaps them back.
inline Manipulator::Actions dragMotion(const Manipulator::Inputs& inputs, Manipulator::Modes mode)
{
  const bool examine = mode == Manipulator::Examine;
  if(inputs.lmb)
  {
    if(inputs.alt)
      return examine ? Manipulator::Actions::LookAround : Manipulator::Actions::Orbit;
    if(inputs.ctrl)
      return Manipulator::Actions::Pan;
    if(inputs.shift)
      return Manipulator::Actions::Orbit;
    return examine ? Manipulator::Actions::Orbit : Manipulator::Actions::LookAround;
  }
  if(inputs.mmb)
    return Manipulator::Actions::Pan;
  if(inputs.rmb)
    return Manipulator::Actions::Dolly;
  return Manipulator::Actions::NoAction;
}

}  // namespace camera_controls

class DragCameraManipulator : public nvutils::CameraManipulator
{
public:
  Actions mouseMove(glm::dvec2 screenPos, const Inputs& inputs) override;
};
