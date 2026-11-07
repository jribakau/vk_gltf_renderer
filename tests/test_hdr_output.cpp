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

#include <algorithm>
#include <array>
#include <cmath>

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

// The SDR grading's Saturation (nvshaders/tonemap_functions.h.slang) takes sRGB-encoded values and
// scales them away from the gray of the same luminance in linear light, so 0 is the colour's own
// luminance as a gray: BT.709's weights on the encoded values made pure red 0.21 instead of 0.50
TEST(TonemapGrading, SaturationOneLeavesTheColour)
{
  for(const float3 c : {float3(0.9F, 0.2F, 0.1F), float3(0.0F), float3(1.0F), float3(0.3F, 0.6F, 0.95F)})
  {
    const float3 out = adjustSaturation(c, 1.0F);
    EXPECT_EQ(out.x, c.x);
    EXPECT_EQ(out.y, c.y);
    EXPECT_EQ(out.z, c.z);
  }
}

TEST(TonemapGrading, ZeroIsTheGrayOfTheLinearLuminance)
{
  // pure red, green and blue: their linear luminance, encoded
  EXPECT_NEAR(adjustSaturation(float3(1.0F, 0.0F, 0.0F), 0.0F).x, toSrgb(float3(0.2126F)).x, 1e-5F);
  EXPECT_NEAR(adjustSaturation(float3(0.0F, 1.0F, 0.0F), 0.0F).x, toSrgb(float3(0.7152F)).x, 1e-5F);
  EXPECT_NEAR(adjustSaturation(float3(0.0F, 0.0F, 1.0F), 0.0F).x, toSrgb(float3(0.0722F)).x, 1e-5F);
  EXPECT_NEAR(adjustSaturation(float3(1.0F, 0.0F, 0.0F), 0.0F).x, 0.4985F, 1e-3F);  // not 0.2126
  for(const float3 c : {float3(0.9F, 0.2F, 0.1F), float3(0.3F, 0.6F, 0.95F), float3(0.55F, 0.42F, 0.38F)})
  {
    const float3 g = adjustSaturation(c, 0.0F);
    EXPECT_NEAR(g.x, g.y, 1e-6F);
    EXPECT_NEAR(g.y, g.z, 1e-6F);
    EXPECT_NEAR(bt709Luminance(toLinear(g)), bt709Luminance(toLinear(c)), 1e-5F);
  }
}

TEST(TonemapGrading, EverySaturationKeepsTheLuminance)
{
  // a skin tone and a muted blue stay in gamut up to twice the saturation
  for(const float3 c : {float3(0.80F, 0.62F, 0.55F), float3(0.40F, 0.48F, 0.60F)})
    for(const float sat : {0.25F, 0.5F, 1.5F, 2.0F})
    {
      const float3 out = adjustSaturation(c, sat);
      EXPECT_NEAR(bt709Luminance(toLinear(out)), bt709Luminance(toLinear(c)), 1e-5F) << sat;
    }
  // more saturation moves the channels apart, less brings them together
  const float3 skin = float3(0.80F, 0.62F, 0.55F);
  EXPECT_GT(adjustSaturation(skin, 1.5F).x - adjustSaturation(skin, 1.5F).z, skin.x - skin.z);
  EXPECT_LT(adjustSaturation(skin, 0.5F).x - adjustSaturation(skin, 0.5F).z, skin.x - skin.z);
}

TEST(TonemapGrading, GraysStayGrayAndOutOfGamutClampsAtBlack)
{
  for(const float v : {0.0F, 0.18F, 0.5F, 1.0F})
    for(const float sat : {0.0F, 0.5F, 2.0F})
    {
      const float3 out = adjustSaturation(float3(v), sat);
      EXPECT_NEAR(out.x, v, 1e-5F);
      EXPECT_NEAR(out.z, v, 1e-5F);
    }
  // twice the saturation of pure red would take green and blue below 0: they stop at 0, no NaN
  const float3 red = adjustSaturation(float3(1.0F, 0.0F, 0.0F), 2.0F);
  EXPECT_EQ(red.y, 0.0F);
  EXPECT_EQ(red.z, 0.0F);
  EXPECT_FALSE(std::isnan(red.x));
}

// The whole grade at saturation 0 on a saturated colour: a gray as bright as the colour
TEST(TonemapGrading, TheGradeAtZeroSaturationIsTheColoursLuminance)
{
  TonemapperData tm;
  tm.method     = int(ToneMapMethod::eClip);
  tm.saturation = 0.0F;
  tm.dither     = 0;
  tm.vignette   = 0.0F;
  tm.inputMatrix = float3x3(1.0F);  // no exposure or white balance
  const float3 c   = float3(0.6F, 0.05F, 0.02F);  // linear, a deep red
  const float3 out = applyTonemap(tm, c, float2(0.0F), float2(1.0F));
  EXPECT_NEAR(out.x, out.y, 1e-5F);
  EXPECT_NEAR(toLinear(out).x, bt709Luminance(c), 1e-5F);
}

// Looks (nvshaders/tonemap_functions.h.slang): each tool leaves the image at its default, and does
// what it says at its ends
namespace {
void expectNear3(const float3& got, const float3& want)
{
  EXPECT_NEAR(got.x, want.x, 1e-6F);
  EXPECT_NEAR(got.y, want.y, 1e-6F);
  EXPECT_NEAR(got.z, want.z, 1e-6F);
}

TonemapperData plainGrade()
{
  TonemapperData tm;
  tm.method      = int(ToneMapMethod::eClip);
  tm.dither      = 0;
  tm.inputMatrix = float3x3(1.0F);
  return tm;
}
}  // namespace

TEST(TonemapLooks, TheirDefaultsLeaveTheImage)
{
  const TonemapperData tm = plainGrade();
  for(const float3 c : {float3(0.6F, 0.05F, 0.02F), float3(0.18F), float3(0.02F, 0.3F, 0.7F), float3(0.0F), float3(1.0F)})
  {
    const float3 out  = applyTonemap(tm, c, float2(7.0F, 3.0F), float2(1920.0F, 1080.0F));
    expectNear3(out, clamp(toSrgb(c), float3(0.0F), float3(1.0F)));
  }
}

TEST(TonemapLooks, LiftRaisesBlackGainScalesGammaBendsTheMiddle)
{
  const float3 lift = float3(0.1F, 0.0F, 0.05F);
  expectNear3(liftGammaGain(float3(0.0F), lift, float3(1.0F), float3(1.0F)), lift);
  EXPECT_EQ(liftGammaGain(float3(1.0F), lift, float3(1.0F), float3(1.0F)), float3(1.0F));  // white stays
  EXPECT_NEAR(liftGammaGain(float3(0.5F), float3(0.0F), float3(1.0F), float3(0.8F)).x, 0.4F, 1e-6F);
  EXPECT_EQ(liftGammaGain(float3(0.0F), float3(0.0F), float3(1.0F), float3(0.8F)), float3(0.0F));  // black stays
  // gamma 2 lifts a quarter to a half, and leaves black and white
  EXPECT_NEAR(liftGammaGain(float3(0.25F), float3(0.0F), float3(2.0F), float3(1.0F)).x, 0.5F, 1e-6F);
  EXPECT_EQ(liftGammaGain(float3(1.0F), float3(0.0F), float3(2.0F), float3(1.0F)), float3(1.0F));
  // a negative lift crushes the blacks to 0, no NaN from the power
  const float3 crushed = liftGammaGain(float3(0.02F), float3(-0.1F), float3(1.2F), float3(1.0F));
  EXPECT_EQ(crushed, float3(0.0F));
}

TEST(TonemapLooks, BlackAndWhiteThroughARedFilter)
{
  const float3 red = float3(0.85F, 0.25F, -0.1F);  // a red filter's weights: sum 1
  // red comes out light, blue dark, grey as it was
  EXPECT_GT(monoAdjust(float3(1.0F, 0.0F, 0.0F), red, 1.0F).x, 0.9F);
  expectNear3(monoAdjust(float3(0.0F, 0.0F, 1.0F), red, 1.0F), float3(0.0F));
  const float3 grey = monoAdjust(float3(0.5F), red, 1.0F);
  EXPECT_NEAR(grey.x, 0.5F, 1e-5F);
  EXPECT_NEAR(grey.z, 0.5F, 1e-5F);
  // the plain weights are the colour's luminance, and half the amount is halfway there
  const float3 skin = float3(0.80F, 0.62F, 0.55F);
  const float3 bw   = monoAdjust(skin, float3(0.2126F, 0.7152F, 0.0722F), 1.0F);
  EXPECT_NEAR(bt709Luminance(toLinear(bw)), bt709Luminance(toLinear(skin)), 1e-5F);
  EXPECT_NEAR(monoAdjust(skin, red, 0.5F).x, (skin.x + monoAdjust(skin, red, 1.0F).x) * 0.5F, 1e-6F);
}

TEST(TonemapLooks, PosterisingSnapsEachChannelToItsLevels)
{
  expectNear3(posterizeAdjust(float3(0.1F, 0.4F, 0.8F), 2.0F, 1.0F), float3(0.0F, 0.0F, 1.0F));
  expectNear3(posterizeAdjust(float3(0.1F, 0.4F, 0.8F), 3.0F, 1.0F), float3(0.0F, 0.5F, 1.0F));
  expectNear3(posterizeAdjust(float3(0.1F, 0.4F, 0.8F), 3.0F, 0.0F), float3(0.1F, 0.4F, 0.8F));
  // in the grade, fewer than 2 levels is none
  TonemapperData tm = plainGrade();
  tm.posterize      = 1.0F;
  const float3 c    = float3(0.3F);
  EXPECT_NEAR(applyTonemap(tm, c, float2(0.0F), float2(1.0F)).x, toSrgb(c).x, 1e-6F);
}

TEST(TonemapLooks, GrainAveragesOutMovesWithTheSeedAndSparesBlack)
{
  const float2 size = float2(1920.0F, 1080.0F);
  const auto   stats = [&](float v, uint32_t seed) {
    double sum = 0.0, sq = 0.0, most = 0.0;
    int    n   = 0;
    for(int y = 0; y < 64; y++)
      for(int x = 0; x < 64; x++, n++)
      {
        const double d = filmGrain(float3(v), 1.0F, float2(float(x), float(y)), size, seed).x - v;
        sum += d;
        sq += d * d;
        most = std::max(most, std::abs(d));
      }
    const double mean = sum / n;
    return std::array<double, 3>{mean, std::sqrt(sq / n - mean * mean), most};
  };
  const auto mid = stats(0.5F, 1);
  EXPECT_LT(std::abs(mid[0]), 0.01);  // no brighter or darker on average
  EXPECT_GT(mid[1], 0.03);            // but visible
  EXPECT_LE(mid[2], 0.2 + 1e-6);      // and within its bound
  EXPECT_LT(stats(0.02F, 1)[1], mid[1] * 0.6);  // less in the shadows
  // the same seed is the same grain; another seed another
  const float3 a = filmGrain(float3(0.5F), 1.0F, float2(10.0F, 20.0F), size, 1);
  EXPECT_EQ(a, filmGrain(float3(0.5F), 1.0F, float2(10.0F, 20.0F), size, 1));
  int moved = 0;
  for(int x = 0; x < 32; x++)
    moved += filmGrain(float3(0.5F), 1.0F, float2(float(x), 5.0F), size, 1) != filmGrain(float3(0.5F), 1.0F, float2(float(x), 5.0F), size, 2);
  EXPECT_GT(moved, 28);
  // the same in every channel
  EXPECT_EQ(a.x, a.y);
  EXPECT_EQ(a.y, a.z);
}

TEST(TonemapLooks, TheExposureViewShowsNoGrain)
{
  TonemapperData tm = plainGrade();
  tm.falseColor     = 1;
  const float3 none = applyTonemap(tm, float3(0.18F), float2(3.0F), float2(1920.0F, 1080.0F));
  tm.grain          = 1.0F;
  EXPECT_EQ(applyTonemap(tm, float3(0.18F), float2(3.0F), float2(1920.0F, 1080.0F)), none);
}

// The exposure view: whole stops around middle grey, a band each, centred on its stop; what the
// image shows clipped or crushed in colours of their own, whatever the stop
TEST(HdrOutput, TheExposureViewBandsStopsAroundMiddleGrey)
{
  EXPECT_EQ(falseColorBand(0.18F), 4);  // the middle band
  EXPECT_EQ(falseColorBand(0.18F * std::exp2(0.49F)), 4);
  EXPECT_EQ(falseColorBand(0.18F * std::exp2(0.51F)), 5);
  EXPECT_EQ(falseColorBand(0.36F), 5);
  EXPECT_EQ(falseColorBand(0.18F * std::exp2(-1.0F)), 3);
  EXPECT_EQ(falseColorBand(0.18F * std::exp2(-4.0F)), 0);
  EXPECT_EQ(falseColorBand(0.0F), 0);  // and below
  EXPECT_EQ(falseColorBand(0.18F * std::exp2(10.0F)), 8);
  const float3 mid = float3(188.F, 55.F, 84.F) / 255.F;
  EXPECT_EQ(falseColorExposure(0.18F, float3(0.46F)), mid);
  const float3 clipped = falseColorExposure(0.18F, float3(1.0F, 0.5F, 0.5F));
  const float3 crushed = falseColorExposure(0.18F, float3(0.0F, 0.0F, 0.001F));
  EXPECT_NE(clipped, mid);
  EXPECT_NE(crushed, mid);
  EXPECT_NE(clipped, crushed);
  // nine distinct bands
  for(int a = 0; a < 9; a++)
    for(int b = a + 1; b < 9; b++)
      EXPECT_NE(falseColorExposure(0.18F * std::exp2(float(a - 4)), float3(0.5F)), falseColorExposure(0.18F * std::exp2(float(b - 4)), float3(0.5F)))
          << a << " " << b;
}

// The tone mapper paints it in place of the image, from the exposure the image uses: a grey card
// at the metered exposure lands in the middle band, two stops over it two bands up
TEST(HdrOutput, TheToneMapperPaintsTheExposureViewAtItsOwnExposure)
{
  TonemapperData tm;
  tm.inputMatrix = float3x3(4.0F);  // two stops of exposure
  tm.falseColor  = 1;
  EXPECT_EQ(applyTonemap(tm, float3(0.045F), float2(0.0F), float2(1.0F)), float3(188.F, 55.F, 84.F) / 255.F);
  EXPECT_EQ(applyTonemap(tm, float3(0.18F), float2(0.0F), float2(1.0F)), falseColorExposure(0.72F, float3(0.5F)));
  tm.falseColor = 0;
  EXPECT_NE(applyTonemap(tm, float3(0.045F), float2(0.0F), float2(1.0F)), float3(188.F, 55.F, 84.F) / 255.F);
}
