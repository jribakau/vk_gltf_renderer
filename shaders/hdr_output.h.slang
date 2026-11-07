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

// The tone curve of HDR display output (src/hdr_output.hpp), shared with the unit tests.
//
// Every look the app offers ends in an SDR curve: ACES, AgX, filmic, Shaped and Khronos PBR Neutral each
// squeeze the scene's highlights under SDR white. On an HDR display each pixel of the SDR image, graded
// as it is, is shown brighter by a gain: the curve's value with the display's headroom over its SDR
// value. Below a knee that gain is 1, so the look is unchanged where the SDR curve doesn't compress.
//   - Khronos PBR Neutral has a peak of its own to move: hdrNeutral() is nvpro's tonemapKhronosPBR
//     compressing to the display's peak instead of SDR white (at a peak of 1 the same curve).
//   - The other curves have none: their shoulder is opened again (hdrOpenShoulder), so that what the SDR
//     curve brought close to white rises towards the peak instead, reaching it where the SDR curve
//     reaches white. The opening is the one rational curve that meets the SDR value at the knee with the
//     same slope and ends at the peak.
//
// Values are linear and relative to SDR white (1.0 = the white of the SDR image, which the display
// shows at its SDR white level); `peak` is the display's peak in those units, at least 1.

#ifndef HDR_OUTPUT_H_SLANG
#define HDR_OUTPUT_H_SLANG

// The tone mappers (TONEMAP_METHOD in nvshaders/tonemap_io.h.slang) by their numbers, so that this
// header stands alone for the tests
static const int HDR_METHOD_CLIP    = 2;
static const int HDR_METHOD_NEUTRAL = 5;

// Where a tone mapper's SDR result starts to be opened: Khronos PBR Neutral's own start of compression
// (0.8 - 0.04, below which it is the identity), and for the curves that compress all the way up, half
// of SDR white. Clip has no shoulder to open: its highlights stay at SDR white.
inline float hdrKnee(int method)
{
  if(method == HDR_METHOD_NEUTRAL)
    return 0.76F;
  if(method == HDR_METHOD_CLIP)
    return 1.0F;
  return 0.5F;
}

// An SDR value `s` (linear, at most 1) opened above `knee` towards `peak`: s itself up to the knee,
// then rising with the same slope there and reaching `peak` where s reaches 1.
inline float hdrOpenShoulder(float s, float knee, float peak)
{
  if(s <= knee || knee >= 1.0F || peak <= 1.0F)
    return s;
  const float d = 1.0F - knee;              // the SDR shoulder's height
  const float D = peak - knee;              // the HDR one's
  const float w = 1.0F - min(s, 1.0F);      // how far below SDR white
  return peak - D * D * w / (d * d + (D - d) * w);
}

// Khronos PBR Neutral (nvshaders' tonemapKhronosPBR, before its sRGB encoding) compressing towards
// `peak` instead of SDR white: the identity up to its start of compression, as in SDR
inline float3 hdrNeutral(float3 color, float peak)
{
  const float startCompression = 0.8F - 0.04F;
  const float desaturation     = 0.15F;
  const float x                = min(color.x, min(color.y, color.z));
  const float top              = max(color.x, max(color.y, color.z));
  const float offset           = x < 0.08F ? x * (-6.25F * x + 1.F) : 0.04F;
  color -= offset;
  if(top >= startCompression)
  {
    const float d       = peak - startCompression;
    const float newPeak = peak - d * d / (top + d - startCompression);
    color *= newPeak / top;
    const float g = 1.F - 1.F / (desaturation * (top - newPeak) + 1.F);
    color         = lerp(color, float3(newPeak), float3(g));
  }
  return color;
}

// The gain on a pixel of the SDR image for the HDR display, by brightest channels: the curve's value
// with the display's headroom over its SDR value. `curveInput` is the colour the curve takes (exposure
// and white balance applied), `sdrCurve` the curve's SDR result for it, linear and before any grading.
// 1 where the SDR curve doesn't compress, so the SDR look stays there; hue and saturation stay everywhere.
inline float hdrGain(int method, float3 curveInput, float3 sdrCurve, float peak)
{
  const float sdr = max(sdrCurve.x, max(sdrCurve.y, sdrCurve.z));
  if(!(sdr > 0.0F) || peak <= 1.0F)
    return 1.0F;
  float hdr;
  if(method == HDR_METHOD_NEUTRAL)
  {
    const float3 c = hdrNeutral(curveInput, peak);
    hdr            = max(c.x, max(c.y, c.z));
  }
  else
    hdr = hdrOpenShoulder(sdr, hdrKnee(method), peak);
  return max(hdr / sdr, 1.0F);
}

#endif  // HDR_OUTPUT_H_SLANG
