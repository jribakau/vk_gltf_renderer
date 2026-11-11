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

// A decoded environment image's texels made fit to light by (loadEnvImage, for EXR: an RGBE .hdr can
// hold none of these). One NaN or infinite texel makes nvvk::HdrIbl's importance integral, every
// texel's pdf and its average non-finite, and the path tracer then drops every environment light
// sample: the dome lights nothing. The sun finder (fv::lighting::environmentSun) finds none either. A
// negative texel gives negative importance. Header-only, for the tests.

#include <cmath>
#include <cstdint>
#include <span>

// What +Inf becomes: half float's largest, the most a path tracer sample keeps (a DLSS input is half),
// so the texel stays the brightest of the image without overflowing the shading that multiplies it
inline constexpr float kEnvTexelBrightest = 65504.0F;

// The channel values a repair replaced, by what they were
struct EnvTexelRepairs
{
  uint64_t nan      = 0;  // to 0
  uint64_t infinite = 0;  // +Inf, to kEnvTexelBrightest
  uint64_t negative = 0;  // below 0, -Inf too, to 0

  uint64_t total() const { return nan + infinite + negative; }
};

// RGBA texels' colours made finite and non-negative; alpha is left alone (HdrIbl writes the pdf there)
inline EnvTexelRepairs repairEnvTexels(std::span<float> rgba)
{
  EnvTexelRepairs repairs;
  for(size_t i = 0; i + 4 <= rgba.size(); i += 4)
    for(size_t c = 0; c < 3; c++)
    {
      float& v = rgba[i + c];
      if(std::isnan(v))
      {
        v = 0.0F;
        repairs.nan++;
      }
      else if(std::isinf(v) && v > 0.0F)
      {
        v = kEnvTexelBrightest;
        repairs.infinite++;
      }
      else if(v < 0.0F)
      {
        v = 0.0F;
        repairs.negative++;
      }
    }
  return repairs;
}
