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
// The night sky's stars as the path tracer reads them (shaders/night_sky.h.slang, SceneFrameInfo::
// nightStars): the Yale Bright Star Catalogue's, to the eye's limit (src/star_catalog.inc, made by
// utils/make_star_catalog.py), each a direction, its light against the sun's and its colour, binned
// by right ascension and declination. Pure: tested without a device (tests/test_night_sky.cpp).
//

#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "shaders/shaderio.h"

namespace fv::night {

// A star as the catalogue gives it: J2000 right ascension and declination in degrees, V magnitude, B-V
struct CatalogStar
{
  float ra;
  float dec;
  float v;
  float bv;
};

std::span<const CatalogStar> catalog();

// A star's colour by its B-V, linear Rec. 709 at luminance 1: the black body at the temperature
// Ballesteros' formula gives it (2012), so a B-V of 0.65, the sun's, is white
glm::vec3 starColor(float bv);

struct StarTables
{
  std::vector<shaderio::NightStar>     stars;  // grouped by cell
  std::vector<shaderio::NightStarCell> cells;  // NIGHT_STAR_CELLS_RA x NIGHT_STAR_CELLS_DEC, rows of right ascension
};

StarTables buildStarTables(std::span<const CatalogStar> stars);

}  // namespace fv::night
