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

// The display a native presentation's frames go to, as far as HDR is concerned, kept apart from D3D
// and Vulkan so it can be tested. The presenter (frame_present_win32.cpp) reads it from Windows; the
// renderer's HDR output (hdr_output.hpp) turns it into the levels its tone curve takes.
//
// An HDR display shows scRGB, linear light where 1.0 is 80 nits. SDR content, the app's panels among
// it, is shown at the display's SDR white level (Windows' "SDR content brightness"), so the SDR image's
// white goes there too and the 3D view's midtones match the panels around it; the highlights the SDR
// image had to squeeze under white rise towards the display's peak.

#pragma once

#include <algorithm>

struct DisplayHdr
{
  bool  hdr          = false;  // Windows shows this display in HDR (its colour space is BT.2100 PQ)
  float sdrWhiteNits = 0.0F;   // Its SDR white level; 0: unknown
  float peakNits     = 0.0F;   // Its peak luminance, as its EDID has it; 0: unknown
  bool operator==(const DisplayHdr&) const = default;
};

struct HdrLevels
{
  float paperWhite = 1.0F;  // SDR white in scRGB
  float peak       = 1.0F;  // The display's peak in SDR whites, at least 1
};

namespace present_hdr {

inline constexpr float kScRgbNits = 80.0F;  // scRGB's 1.0
// Taken when Windows doesn't say: its own default SDR white level on an HDR display is about this
inline constexpr float kUnknownSdrWhiteNits = 200.0F;
// Taken when the display's EDID gives no peak: an HDR10 display's usual mastering peak
inline constexpr float kUnknownPeakNits = 1000.0F;

// The levels for `display`; `peakOverrideNits` above 0 stands for the display's peak (presentHdrPeak)
inline HdrLevels levels(const DisplayHdr& display, float peakOverrideNits = 0.0F)
{
  const float white = display.sdrWhiteNits > 0.0F ? display.sdrWhiteNits : kUnknownSdrWhiteNits;
  const float peak  = peakOverrideNits > 0.0F ? peakOverrideNits : display.peakNits > 0.0F ? display.peakNits : kUnknownPeakNits;
  return {white / kScRgbNits, std::max(1.0F, peak / white)};
}

// Windows' SDR white level (DISPLAYCONFIG_SDR_WHITE_LEVEL::SDRWhiteLevel) in nits: it counts in
// thousandths of scRGB's 80 nits
inline float sdrWhiteNits(unsigned long sdrWhiteLevel)
{
  return float(sdrWhiteLevel) / 1000.0F * kScRgbNits;
}

}  // namespace present_hdr
