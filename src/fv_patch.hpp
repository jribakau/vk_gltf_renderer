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
// What the fields of a merge patch are checked with: the panels' states the host keeps (lighting.set's
// sections in fv_lighting.cpp, backdrop.set in fv_backdrop_state.cpp). A value of the wrong type
// throws std::invalid_argument naming its key, and a number is clamped to its range.
//

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "fv_events.hpp"

namespace fv::patch {

struct Checks
{
  const char* what;  // whose patch it is, the start of every message: "lighting"

  [[noreturn]] void reject(const std::string& why) const { throw std::invalid_argument(std::string(what) + ": " + why); }

  double number(const Json& v, const std::string& key, double lo, double hi) const
  {
    if(!v.is_number())
      reject(key + " must be a number");
    const double x = v.get<double>();
    if(!std::isfinite(x))
      reject(key + " must be finite");
    return std::clamp(x, lo, hi);
  }

  bool boolean(const Json& v, const std::string& key) const
  {
    if(!v.is_boolean())
      reject(key + " must be true or false");
    return v.get<bool>();
  }

  std::string string(const Json& v, const std::string& key) const
  {
    if(!v.is_string())
      reject(key + " must be a string");
    return v.get<std::string>();
  }

  std::string oneOf(const Json& v, const std::string& key, std::initializer_list<const char*> values) const
  {
    const std::string s = string(v, key);
    for(const char* allowed : values)
      if(s == allowed)
        return s;
    std::string list;
    for(const char* allowed : values)
      list += (list.empty() ? "" : ", ") + std::string(allowed);
    reject(key + " is one of " + list + ", not \"" + s + "\"");
  }
};

// Each field: null resets it, anything else is checked and clamped. `T` is the field's type.
template <class T, class Parse>
void field(T& dst, const T& def, const Json& v, Parse parse)
{
  dst = v.is_null() ? def : T(parse(v));
}

// Degrees wrapped to [-180, 180).
inline double wrapDegrees(double degrees)
{
  const double x = std::fmod(degrees + 180.0, 360.0);
  return (x < 0.0 ? x + 360.0 : x) - 180.0;
}

// The merge patch that turns `from` into `to` (objects recurse, anything else is replaced): what a
// topic's change is published as.
inline Json mergeDiff(const Json& from, const Json& to)
{
  if(!from.is_object() || !to.is_object())
    return to;
  Json out = Json::object();
  for(auto it = to.begin(); it != to.end(); ++it)
  {
    const auto old = from.find(it.key());
    if(old == from.end())
      out[it.key()] = it.value();
    else if(*old != it.value())
      out[it.key()] = (old->is_object() && it.value().is_object()) ? mergeDiff(*old, it.value()) : it.value();
  }
  for(auto it = from.begin(); it != from.end(); ++it)
    if(!to.contains(it.key()))
      out[it.key()] = nullptr;
  return out;
}

}  // namespace fv::patch
