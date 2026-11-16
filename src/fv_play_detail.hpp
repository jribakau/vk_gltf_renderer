/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Small helpers play mode's controllers share (fv_play.cpp, fv_play_car.cpp)

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

#include "fv_play.hpp"

namespace fv::play::detail {

inline double ease(double t)
{
  t = std::clamp(t, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// The share of what is left that a step of `dt` takes, easing at time constant `tau`: the same over
// a second however the second is cut into frames
inline double share(double dt, double tau)
{
  return tau > 0.0 ? 1.0 - std::exp(-dt / tau) : 1.0;
}

inline glm::dvec3 level(const glm::dvec3& v, const glm::dvec3& up)
{
  return v - up * glm::dot(v, up);
}

// A unit direction square to `up`, the nearest to `v`; any level one for `v` along up
inline glm::dvec3 levelUnit(const glm::dvec3& v, const glm::dvec3& up)
{
  for(const glm::dvec3& d : {v, glm::dvec3(0.0, 0.0, -1.0), glm::dvec3(1.0, 0.0, 0.0)})
  {
    const glm::dvec3 l = level(d, up);
    if(glm::length(l) > 1e-6)
      return glm::normalize(l);
  }
  return {1.0, 0.0, 0.0};
}

inline double key(uint32_t keys, Key k)
{
  return (keys & k) ? 1.0 : 0.0;
}

// `value` moved toward `target` by at most `rate * dt`
inline double toward(double value, double target, double rate, double dt)
{
  return value < target ? std::min(target, value + rate * dt) : std::max(target, value - rate * dt);
}

}  // namespace fv::play::detail
