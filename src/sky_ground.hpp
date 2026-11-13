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

// The planet's ground the physical sky is built over: what its tables, its ground below the horizon
// and the clouds' ground bounce take the land to reflect (docs/sky.md, the planet's ground under the horizon).

#include <glm/glm.hpp>

namespace sky_ground {

// The sky's ground albedo: the land a backdrop's kit gives (`land`, 0 0 0 for none), else the sea
// when the ground is a body of water (`waterDepth` > 0 over a floor of `floor`; groundWaterAlbedo),
// else `ground`, the atmosphere's own setting
glm::vec3 albedo(const glm::vec3& land, float waterDepth, const glm::vec3& floor, const glm::vec3& ground);

}  // namespace sky_ground
