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

// HDR display output's tone curve (shaders/hdr_output.h.slang) compiled as C++: an SDR look below its
// knee is left as it is, the shoulder above opens smoothly to the display's peak, and Khronos PBR Neutral
// is nvpro's curve with its peak moved to the display's.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <gtest/gtest.h>

// GLM declares fma for any type but defines it for scalars only; nvpro's toSrgb() calls it on a float3
namespace glm {
template <>
vec3 fma(vec3 const& a, vec3 const& b, vec3 const& c)
{
  return a * b + c;
}
}  // namespace glm

using namespace glm;
#include <nvshaders/slang_types.h>
#include <nvshaders/tonemap_functions.h.slang>
using namespace shaderio;
#include "../shaders/hdr_output.h.slang"

TEST(HdrOutput, BelowTheKneeTheSdrValueStays)
{
  for(float s : {0.0F, 0.1F, 0.18F, 0.4F, 0.5F})
    EXPECT_EQ(hdrOpenShoulder(s, 0.5F, 5.0F), s);
  EXPECT_EQ(hdrOpenShoulder(0.7F, 0.76F, 5.0F), 0.7F);
}

TEST(HdrOutput, SdrWhiteReachesThePeak)
{
  EXPECT_FLOAT_EQ(hdrOpenShoulder(1.0F, 0.5F, 5.0F), 5.0F);
  EXPECT_FLOAT_EQ(hdrOpenShoulder(1.0F, 0.76F, 12.5F), 12.5F);
  // a value past SDR white (a grade that overshot) stays at the peak
  EXPECT_FLOAT_EQ(hdrOpenShoulder(1.2F, 0.5F, 5.0F), 5.0F);
}

// No step and no kink where the opening starts: a gradient through the knee stays smooth
TEST(HdrOutput, TheOpeningMeetsTheKneeWithItsSlope)
{
  const float knee = 0.5F, peak = 6.0F, h = 1e-3F;
  EXPECT_NEAR(hdrOpenShoulder(knee + 1e-6F, knee, peak), knee, 1e-5F);
  const float slope = (hdrOpenShoulder(knee + h, knee, peak) - hdrOpenShoulder(knee, knee, peak)) / h;
  EXPECT_NEAR(slope, 1.0F, 0.02F);
}

TEST(HdrOutput, TheOpeningRisesThroughout)
{
  float last = 0.0F;
  for(int i = 0; i <= 1000; i++)
  {
    const float v = hdrOpenShoulder(float(i) / 1000.0F, 0.5F, 8.0F);
    EXPECT_GE(v, last) << i;
    last = v;
  }
}

// A display no brighter than SDR white, or a tone mapper without a shoulder, shows the SDR image
TEST(HdrOutput, NoHeadroomOrNoShoulderOpensNothing)
{
  EXPECT_EQ(hdrOpenShoulder(0.9F, 0.5F, 1.0F), 0.9F);
  EXPECT_EQ(hdrOpenShoulder(0.9F, 0.5F, 0.5F), 0.9F);
  EXPECT_EQ(hdrOpenShoulder(0.9F, hdrKnee(HDR_METHOD_CLIP), 5.0F), 0.9F);
}

// The knees are the curves': Khronos PBR Neutral is the identity up to its start of compression
TEST(HdrOutput, TheKneesFollowTheToneMapper)
{
  EXPECT_FLOAT_EQ(hdrKnee(int(ToneMapMethod::eKhronosPBR)), 0.76F);
  EXPECT_FLOAT_EQ(hdrKnee(int(ToneMapMethod::eClip)), 1.0F);
  EXPECT_FLOAT_EQ(hdrKnee(int(ToneMapMethod::eACES)), 0.5F);
  EXPECT_FLOAT_EQ(hdrKnee(int(ToneMapMethod::eAgX)), 0.5F);
  EXPECT_FLOAT_EQ(hdrKnee(int(ToneMapMethod::eFilmic)), 0.5F);
}

// At a peak of SDR white it is nvpro's Khronos PBR Neutral; the identity below its start of compression
// whatever the peak; and it compresses towards the peak above
TEST(HdrOutput, NeutralIsNvprosCurveWithItsPeakMoved)
{
  for(const float3 c : {float3(0.05F), float3(0.3F, 0.2F, 0.1F), float3(0.9F), float3(2.0F, 1.0F, 0.5F), float3(8.0F), float3(40.0F, 3.0F, 1.0F)})
  {
    const float3 nvpro = toLinear(tonemapKhronosPBR(c));
    const float3 ours  = hdrNeutral(c, 1.0F);
    EXPECT_NEAR(ours.x, nvpro.x, 1e-5F);
    EXPECT_NEAR(ours.y, nvpro.y, 1e-5F);
    EXPECT_NEAR(ours.z, nvpro.z, 1e-5F);
  }
  const float3 dim(0.5F, 0.4F, 0.3F);
  EXPECT_EQ(hdrNeutral(dim, 5.0F), hdrNeutral(dim, 1.0F));
  EXPECT_GT(hdrNeutral(float3(4.0F), 5.0F).x, 2.5F);
  EXPECT_LT(hdrNeutral(float3(400.0F), 5.0F).x, 5.0F);
  EXPECT_GT(hdrNeutral(float3(400.0F), 5.0F).x, 4.9F);
}

// The gain on the SDR image: 1 where its curve doesn't compress, so mid grey and skin stay as in SDR,
// growing with the highlight, and for Neutral exactly the HDR curve over the SDR one
TEST(HdrOutput, TheGainLeavesTheMidtonesAndRaisesTheHighlights)
{
  const float peak = 5.0F;
  for(int method : {int(ToneMapMethod::eACES), int(ToneMapMethod::eAgX), int(ToneMapMethod::eFilmic), int(ToneMapMethod::eKhronosPBR)})
  {
    const auto sdr = [&](float x) {
      const float3 c = float3(x);
      if(method == int(ToneMapMethod::eKhronosPBR))
        return hdrNeutral(c, 1.0F);
      const float3 e = method == int(ToneMapMethod::eACES) ? tonemapACES(c) : method == int(ToneMapMethod::eAgX) ? tonemapAgX(c) : tonemapFilmic(c);
      return toLinear(clamp(e, float3(0.0F), float3(1.0F)));
    };
    const auto gain = [&](float x) { return hdrGain(method, float3(x), sdr(x), peak); };
    EXPECT_EQ(gain(0.18F), 1.0F) << "mid grey, method " << method;
    // AgX is the gentlest: 8 times mid grey's light is some 1.4 times SDR white at this peak
    EXPECT_GT(gain(8.0F) * sdr(8.0F).x, 1.2F) << "a highlight past SDR white, method " << method;
    EXPECT_GT(gain(8.0F) * sdr(8.0F).x, gain(2.0F) * sdr(2.0F).x) << method;
    EXPECT_LE(gain(1000.0F) * sdr(1000.0F).x, peak * 1.0001F) << "never past the peak, method " << method;
  }
  EXPECT_NEAR(hdrGain(int(ToneMapMethod::eKhronosPBR), float3(3.0F), hdrNeutral(float3(3.0F), 1.0F), peak) * hdrNeutral(float3(3.0F), 1.0F).x,
              hdrNeutral(float3(3.0F), peak).x, 1e-5F);
  // a display with no headroom, or a black pixel, keeps the SDR image
  EXPECT_EQ(hdrGain(int(ToneMapMethod::eACES), float3(8.0F), float3(0.95F), 1.0F), 1.0F);
  EXPECT_EQ(hdrGain(int(ToneMapMethod::eACES), float3(0.0F), float3(0.0F), peak), 1.0F);
}
