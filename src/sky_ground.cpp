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

#include "sky_ground.hpp"

#include <algorithm>
#include <cmath>

using namespace glm;
using float3 = glm::vec3;
#include "../shaders/water_optics.h.slang"

namespace sky_ground {

glm::vec3 albedo(const glm::vec3& land, float waterDepth, const glm::vec3& floor, const glm::vec3& ground)
{
  if(glm::any(glm::greaterThan(land, glm::vec3(0.0F))))
    return land;
  if(waterDepth > 0.0F)
    return groundWaterAlbedo(glm::clamp(floor, glm::vec3(0.0F), glm::vec3(1.0F)), waterDepth);
  return ground;
}

}  // namespace sky_ground
