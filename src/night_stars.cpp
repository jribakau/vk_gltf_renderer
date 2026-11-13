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

#include "night_stars.hpp"

#include <algorithm>
#include <cmath>

using namespace glm;
using namespace shaderio;
#include "shaders/night_sky.h.slang"

namespace fv::night {

namespace {

const CatalogStar kCatalog[] = {
#include "star_catalog.inc"
};

// The Planck spectrum at `kelvin` through the CIE 1931 colour matching functions (Wyman, Sloan and
// Shirley's fits, 2013), to linear Rec. 709
glm::vec3 blackBody(double kelvin)
{
  const auto g = [](double x, double mu, double s1, double s2) {
    const double t = (x - mu) / (x < mu ? s1 : s2);
    return std::exp(-0.5 * t * t);
  };
  double X = 0, Y = 0, Z = 0;
  for(double nm = 380.0; nm <= 780.0; nm += 5.0)
  {
    const double l     = nm * 1e-9;
    const double power = 1.0 / (l * l * l * l * l * (std::exp(1.4388e-2 / (l * kelvin)) - 1.0));
    X += power * (1.056 * g(nm, 599.8, 37.9, 31.0) + 0.362 * g(nm, 442.0, 16.0, 26.7) - 0.065 * g(nm, 501.1, 20.4, 26.2));
    Y += power * (0.821 * g(nm, 568.8, 46.9, 40.5) + 0.286 * g(nm, 530.9, 16.3, 31.1));
    Z += power * (1.217 * g(nm, 437.0, 11.8, 36.0) + 0.681 * g(nm, 459.0, 26.0, 13.8));
  }
  const glm::dvec3 rgb(3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.2040 * Y + 1.0570 * Z);
  return glm::vec3(glm::max(rgb, glm::dvec3(0.0)) / Y);
}

}  // namespace

std::span<const CatalogStar> catalog()
{
  return kCatalog;
}

glm::vec3 starColor(float bv)
{
  const double b = std::clamp(double(bv), -0.4, 2.0);
  const double t = 4600.0 * (1.0 / (0.92 * b + 1.7) + 1.0 / (0.92 * b + 0.62));
  // against the sun's own (B-V 0.65): a sunlike star white, as the eye under the sun takes it
  const glm::vec3 c = blackBody(t) / blackBody(5778.0);
  return c / std::max(0.2126F * c.x + 0.7152F * c.y + 0.0722F * c.z, 1e-6F);
}

StarTables buildStarTables(std::span<const CatalogStar> stars)
{
  StarTables out;
  out.cells.resize(size_t(NIGHT_STAR_CELLS_RA) * NIGHT_STAR_CELLS_DEC);
  std::vector<std::pair<uint32_t, shaderio::NightStar>> binned;
  binned.reserve(stars.size());
  for(const CatalogStar& s : stars)
  {
    shaderio::NightStar n{};
    n.direction          = celestialUnit(s.ra, s.dec);
    n.light              = starColor(s.bv) * starLightShare(s.v);
    const uint32_t cell = uint32_t(nightStarRow(n.direction) * NIGHT_STAR_CELLS_RA + nightStarColumn(n.direction));
    binned.push_back({cell, n});
  }
  std::stable_sort(binned.begin(), binned.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  out.stars.reserve(binned.size());
  for(const auto& [cell, star] : binned)
  {
    shaderio::NightStarCell& c = out.cells[cell];
    if(c.count == 0)
      c.first = uint32_t(out.stars.size());
    c.count++;
    out.stars.push_back(star);
  }
  return out;
}

}  // namespace fv::night
