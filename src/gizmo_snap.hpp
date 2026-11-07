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

// The gizmo's snapping: a drag moves by whole steps from where it started, so a prop at 0.37 m moves
// to 0.62, 0.87..., not onto the grid's 0.25 and 0.5. Header-only, so the host's tests check it
// without the gizmo's Vulkan side.

#include <cmath>

#include <glm/glm.hpp>

namespace gizmo {

// `now`, moved from `start` by a whole number of `step`s; as it is without a step
inline float snapFrom(float start, float now, float step)
{
  return step > 0.f ? start + std::round((now - start) / step) * step : now;
}

inline glm::vec3 snapFrom(const glm::vec3& start, const glm::vec3& now, float step)
{
  return {snapFrom(start.x, now.x, step), snapFrom(start.y, now.y, step), snapFrom(start.z, now.z, step)};
}

// a change, in whole steps
inline float snapped(float delta, float step)
{
  return step > 0.f ? std::round(delta / step) * step : delta;
}

}  // namespace gizmo
