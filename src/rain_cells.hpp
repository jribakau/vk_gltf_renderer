/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The rain's cells a frame draws (src/rain.hpp, shaders/rain.h.slang): those round the camera within
// its reach that its frustum sees. Plain C++, for the tests.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

namespace rain_cells {

// The cells of `cellSize` metres round `camera` within `reach` that the frustum of `viewProj` (a
// Vulkan projection, depth 0 to 1, times the view) sees, in the drops' sheared coordinates: x and z
// along the ground sheared by `shear` (a drop at height y is at x + shear.x y, z + shear.y y), y up.
// At most `limit`, the nearest kept.
inline std::vector<glm::ivec4> visible(const glm::mat4& viewProj, const glm::vec3& camera, const glm::vec2& shear, float cellSize,
                                       float reach, size_t limit)
{
  std::vector<glm::ivec4> cells;
  if(cellSize <= 0.0F || reach <= 0.0F || limit == 0)
    return cells;
  // the frustum's planes, inward (Gribb and Hartmann), depth 0 to 1
  const glm::mat4           m = glm::transpose(viewProj);  // rows as columns
  std::array<glm::vec4, 5> planes{m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[2]};
  const auto                outside = [&](const glm::vec3& lo, const glm::vec3& hi) {
    for(const glm::vec4& p : planes)
    {
      // the box's corner farthest along the plane's normal
      const glm::vec3 c(p.x >= 0.0F ? hi.x : lo.x, p.y >= 0.0F ? hi.y : lo.y, p.z >= 0.0F ? hi.z : lo.z);
      if(glm::dot(glm::vec3(p), c) + p.w < 0.0F)
        return true;
    }
    return false;
  };

  const int y0 = int(std::floor((camera.y - reach) / cellSize));
  const int y1 = int(std::floor((camera.y + reach) / cellSize));
  struct Found
  {
    glm::ivec4 cell;
    float      distance;
  };
  std::vector<Found> found;
  for(int j = y0; j <= y1; ++j)
  {
    const float lowY = float(j) * cellSize, highY = lowY + cellSize;
    // the sheared ground's square under the camera's sphere at these heights
    const float hx = camera.x - shear.x * camera.y, hz = camera.z - shear.y * camera.y;
    const float slantX = std::abs(shear.x) * (reach + cellSize), slantZ = std::abs(shear.y) * (reach + cellSize);
    const int   i0 = int(std::floor((hx - reach - slantX) / cellSize)), i1 = int(std::floor((hx + reach + slantX) / cellSize));
    const int   l0 = int(std::floor((hz - reach - slantZ) / cellSize)), l1 = int(std::floor((hz + reach + slantZ) / cellSize));
    for(int i = i0; i <= i1; ++i)
      for(int l = l0; l <= l1; ++l)
      {
        // the cell's box in the world: its square sheared over its height
        const float     ax = float(i) * cellSize, az = float(l) * cellSize;
        const glm::vec3 lo(ax + std::min(shear.x * lowY, shear.x * highY), lowY, az + std::min(shear.y * lowY, shear.y * highY));
        const glm::vec3 hi(ax + cellSize + std::max(shear.x * lowY, shear.x * highY), highY,
                           az + cellSize + std::max(shear.y * lowY, shear.y * highY));
        const glm::vec3 nearest = glm::clamp(camera, lo, hi);
        const float     d       = glm::length(nearest - camera);
        if(d > reach || outside(lo, hi))
          continue;
        found.push_back({glm::ivec4(i, j, l, 0), d});
      }
  }
  if(found.size() > limit)
  {
    std::nth_element(found.begin(), found.begin() + std::ptrdiff_t(limit), found.end(),
                     [](const Found& a, const Found& b) { return a.distance < b.distance; });
    found.resize(limit);
  }
  cells.reserve(found.size());
  for(const Found& f : found)
    cells.push_back(f.cell);
  return cells;
}

}  // namespace rain_cells
