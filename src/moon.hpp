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
// The moon as the physical sky draws it and lights by it: how bright it is against the sun at a
// phase, and how its face is lit, so the drawn disk averages to the light it gives. Pure, tested
// without a device (tests/test_moon.cpp). Where it stands is the app's (sunPath.ts moonAt): on the
// sun's path through the year, as far from the sun as its phase says, at the clock's hour.
//
// The phase angle is the angle at the moon between the sun and the Earth: 0 full, 90 a quarter,
// 180 new. With the sun far off it is 180 degrees less the angle in the sky between the two.

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

namespace fv::moon {

// The full moon's light at the Earth against the sun's, both above the air: 0.32 lux against
// 1.27e5 (the solar illuminance constant).
inline constexpr double kFullShare = 2.5e-6;

// The light of the moon's dark part, lit by the Earth (earthshine), against the full moon's mean
// over its disk: the most, a crescent's, when the Earth seen from the moon is full.
inline constexpr double kEarthshine = 2.0e-4;

// The phase angle, degrees, of a moon seen along `toMoon` with the sun along `toSun` (unit vectors).
inline double phaseAngle(const glm::vec3& toSun, const glm::vec3& toMoon)
{
  const double c = std::clamp(double(glm::dot(toSun, toMoon)), -1.0, 1.0);
  return 180.0 - std::acos(c) * 180.0 / 3.14159265358979323846;
}

// The moon's light against the sun's at a phase angle, degrees: kFullShare at full, falling as
// Allen's lunar magnitude does, 0.026 |a| + 4e-9 a^4 (a quarter about a tenth of a full moon, a thin
// crescent under a hundredth). 0 at new.
inline double lightShare(double phaseAngleDeg)
{
  const double a = std::clamp(std::abs(phaseAngleDeg), 0.0, 180.0);
  if(a >= 179.0)
    return 0.0;
  return kFullShare * std::pow(10.0, -0.4 * (0.026 * a + 4e-9 * a * a * a * a));
}

// How the face is lit at a point of it, by its normal's cosines to the sun (cosI) and to the eye
// (cosE): Lommel-Seeliger, cosI / (cosI + cosE), the law of a dusty surface that scatters once,
// which is why a full moon is as bright at its edge as at its middle. 0 on the dark side.
inline double faceShade(double cosI, double cosE)
{
  return cosI > 0.0 && cosE > 0.0 ? cosI / (cosI + cosE) : 0.0;
}

// The mean of faceShade over the disk as seen, at a phase angle, degrees: what the drawn disk's
// shading is divided by so its mean radiance is lightShare's (both are the same solid angle as the
// sun's). 0.5 at full, 0 at new. Numerical, on an n x n grid over the disk.
inline double meanFaceShade(double phaseAngleDeg, int n = 96)
{
  const double a   = std::clamp(phaseAngleDeg, 0.0, 180.0) * 3.14159265358979323846 / 180.0;
  const double sx  = std::sin(a), sz = std::cos(a);  // the sun, the eye along +z
  double       sum = 0.0;
  int          in  = 0;
  for(int j = 0; j < n; ++j)
    for(int i = 0; i < n; ++i)
    {
      const double x  = (2.0 * (i + 0.5)) / n - 1.0;
      const double y  = (2.0 * (j + 0.5)) / n - 1.0;
      const double rr = x * x + y * y;
      if(rr >= 1.0)
        continue;
      const double z = std::sqrt(1.0 - rr);
      sum += faceShade(x * sx + z * sz, z);
      ++in;
    }
  return in > 0 ? sum / in : 0.0;
}

// Earthshine's radiance against the full moon's mean, at a phase angle, degrees: the Earth as the
// moon sees it is as full as the moon is new.
inline double earthshineShare(double phaseAngleDeg)
{
  const double a = std::clamp(phaseAngleDeg, 0.0, 180.0) * 3.14159265358979323846 / 180.0;
  return kEarthshine * 0.5 * (1.0 - std::cos(a));
}

}  // namespace fv::moon
