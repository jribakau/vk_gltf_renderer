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

// Local exposure's grid and curve (shaders/local_exposure.h.slang) compiled as C++, and the whole
// pass, run here on the CPU as shaders/local_exposure.slang runs it on the GPU: the curve leaves the
// mid-tones alone and never lets one tone overtake another, a flat image gets the curve's gain, an
// edge between a dark region and a bright one gets no halo, and a region's texture keeps its contrast.

// GLM as test_bsdf_lobes.cpp, the other file of this executable, sees it
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/local_exposure.h.slang"

namespace {

// An image of exposed luminances: the grid reads nothing else
struct Lum
{
  int                w = 0, h = 0;
  std::vector<float> px;
  Lum(int width, int height, float value)
      : w(width)
      , h(height)
      , px(size_t(width) * size_t(height), value)
  {
  }
  float&       at(int x, int y) { return px[size_t(y) * size_t(w) + size_t(x)]; }
  const float& at(int x, int y) const { return px[size_t(y) * size_t(w) + size_t(x)]; }
};

// A luminance `stops` from mid grey
float level(float stops)
{
  return LEXP_MID_GREY * std::exp2(stops);
}

// Each pixel's gain in stops, as lexpSplat, lexpBlur and lexpApply give it at exposure 1
std::vector<float> gains(const Lum& image, float shadows, float highlights)
{
  const int cell   = lexpCellSize(image.h);
  const int cellsX = lexpCells(image.w, cell), cellsY = lexpCells(image.h, cell);
  const size_t        n = size_t(cellsX) * size_t(cellsY) * LEXP_BINS;
  std::vector<int>    sum(n, 0);
  std::vector<int>    count(n, 0);
  const int           half = cell / 2;
  for(int by = 0; by * 2 < image.h; by++)
    for(int bx = 0; bx * 2 < image.w; bx++)
    {
      // the block's mean, as the bilinear tap at its middle gives it, the edge clamped
      float mean = 0.0f;
      for(int dy = 0; dy < 2; dy++)
        for(int dx = 0; dx < 2; dx++)
          mean += 0.25f * image.at(std::min(bx * 2 + dx, image.w - 1), std::min(by * 2 + dy, image.h - 1));
      const float s = lexpStops(mean, 1.0f);
      const int   i = lexpIndex(bx / half, by / half, lexpBin(s), cellsX);
      sum[i] += int(std::round(s * LEXP_FIXED));
      count[i] += 1;
    }
  std::vector<vec2> grid(n);
  for(size_t i = 0; i < n; i++)
    grid[i] = vec2(float(sum[i]) / LEXP_FIXED, float(count[i]));

  std::vector<vec2> blurred(n, vec2(0.0f));
  for(int cy = 0; cy < cellsY; cy++)
    for(int cx = 0; cx < cellsX; cx++)
      for(int bin = 0; bin < LEXP_BINS; bin++)
      {
        vec2 acc(0.0f);
        for(int dy = -1; dy <= 1; dy++)
          for(int dx = -1; dx <= 1; dx++)
            for(int dz = -1; dz <= 1; dz++)
            {
              const int x = std::clamp(cx + dx, 0, cellsX - 1), y = std::clamp(cy + dy, 0, cellsY - 1);
              const int z = std::clamp(bin + dz, 0, LEXP_BINS - 1);
              acc += lexpBlurWeight(dx) * lexpBlurWeight(dy) * lexpBlurWeight(dz) * grid[lexpIndex(x, y, z, cellsX)];
            }
        blurred[lexpIndex(cx, cy, bin, cellsX)] = acc;
      }

  std::vector<float> out(image.px.size());
  for(int py = 0; py < image.h; py++)
    for(int px = 0; px < image.w; px++)
    {
      const float s  = lexpStops(image.at(px, py), 1.0f);
      const vec2  g  = (vec2(float(px), float(py)) + 0.5f) / float(cell) - 0.5f;
      const ivec2 g0 = ivec2(std::floor(g.x), std::floor(g.y));
      const vec2  t  = g - vec2(g0);
      const float z  = lexpBinCoord(s);
      const int   z0 = int(std::floor(z));
      const float tz = z - float(z0);
      vec2        at(0.0f);
      for(int dy = 0; dy <= 1; dy++)
        for(int dx = 0; dx <= 1; dx++)
          for(int dz = 0; dz <= 1; dz++)
          {
            const int   x = std::clamp(g0.x + dx, 0, cellsX - 1), y = std::clamp(g0.y + dy, 0, cellsY - 1);
            const int   b = std::min(z0 + dz, LEXP_BINS - 1);
            const float w = (dx != 0 ? t.x : 1.0f - t.x) * (dy != 0 ? t.y : 1.0f - t.y) * (dz != 0 ? tz : 1.0f - tz);
            at += w * blurred[lexpIndex(x, y, b, cellsX)];
          }
      const float base = at.y > 1e-3f ? at.x / at.y : s;
      out[size_t(py) * size_t(image.w) + size_t(px)] = lexpGain(base, shadows, highlights);
    }
  return out;
}

}  // namespace

// Within the knee nothing moves; past it shadows only rise and highlights only fall, by at most
// LEXP_MAX_GAIN, smoothly; and a region's new level never falls as its old one rises
TEST(LocalExposure, TheCurveLeavesMidTonesAloneAndKeepsTonesInOrder)
{
  for(float v = -(LEXP_KNEE - LEXP_SOFT); v <= LEXP_KNEE - LEXP_SOFT; v += 0.01f)
    EXPECT_EQ(lexpGain(v, 1.0f, 1.0f), 0.0f) << v;
  for(float shadows : {0.0f, 0.3f, 0.7f, 1.0f})
    for(float highlights : {0.0f, 0.3f, 0.7f, 1.0f})
    {
      float before = -24.0f + lexpGain(-24.0f, shadows, highlights);
      float gain   = lexpGain(-24.0f, shadows, highlights);
      for(float v = -24.0f; v <= 24.0f; v += 0.01f)
      {
        const float g = lexpGain(v, shadows, highlights);
        EXPECT_LE(std::abs(g), LEXP_MAX_GAIN);
        if(highlights == 0.0f)
          EXPECT_GE(g, 0.0f) << "lift alone darkened " << v;
        if(shadows == 0.0f)
          EXPECT_LE(g, 0.0f) << "hold alone brightened " << v;
        EXPECT_GE(v + g, before - 1e-5f) << "tones crossed at " << v << " (" << shadows << ", " << highlights << ")";
        EXPECT_LE(std::abs(g - gain), LEXP_SLOPE * 0.01f + 1e-5f) << "a jump at " << v;  // no corner, no step
        before = v + g;
        gain   = g;
      }
    }
  // far out the gain is the cap; at an amount, its share of each stop past the knee
  EXPECT_FLOAT_EQ(lexpGain(-10.0f, 1.0f, 0.0f), LEXP_MAX_GAIN);
  EXPECT_FLOAT_EQ(lexpGain(10.0f, 0.0f, 1.0f), -LEXP_MAX_GAIN);
  EXPECT_NEAR(lexpGain(-4.0f, 0.5f, 0.0f), 0.5f * LEXP_SLOPE * (4.0f - LEXP_KNEE), 1e-5f);
}

// The grid follows the image's height, its bins cover the stops it is built for
TEST(LocalExposure, TheGridsSizes)
{
  EXPECT_EQ(lexpCellSize(1395), 36);  // 35, made even for the splat's 2x2 blocks
  EXPECT_EQ(lexpCellSize(2160), 54);
  EXPECT_EQ(lexpCellSize(200), LEXP_MIN_CELL);
  EXPECT_EQ(lexpCells(2160, 36), 60);
  EXPECT_EQ(lexpBin(LEXP_LOW), 0);
  EXPECT_EQ(lexpBin(-24.0f), 0);
  EXPECT_EQ(lexpBin(24.0f), LEXP_BINS - 1);
  const float binStops = (LEXP_HIGH - LEXP_LOW) / float(LEXP_BINS);
  for(int b = 0; b < LEXP_BINS; b++)
  {
    const float centre = LEXP_LOW + (float(b) + 0.5f) * binStops;
    EXPECT_EQ(lexpBin(centre), b);
    EXPECT_NEAR(lexpBinCoord(centre), float(b), 1e-4f);
  }
  EXPECT_NEAR(lexpStops(level(-3.0f), 1.0f), -3.0f, 1e-5f);
  EXPECT_NEAR(lexpStops(level(-3.0f), 2.0f), -2.0f, 1e-5f);
  EXPECT_EQ(lexpStops(0.0f, 1.0f), -24.0f);
}

// A flat image is one region: every pixel gets the curve's gain at its level
TEST(LocalExposure, AFlatImageGetsTheCurvesGain)
{
  for(float stops : {-6.0f, -4.0f, -1.5f, 0.0f, 2.0f, 5.0f})
  {
    const Lum                image(320, 200, level(stops));
    const std::vector<float> g = gains(image, 0.8f, 0.6f);
    const float              want = lexpGain(stops, 0.8f, 0.6f);
    for(float gain : g)
      ASSERT_NEAR(gain, want, 0.01f) << stops;
  }
}

// A dark region against a bright one: each side gets its own region's gain up to the edge, where a
// plain blur would mix them over a cell or two either way
TEST(LocalExposure, AnEdgeGetsNoHalo)
{
  for(const vec2 sides : {vec2(-5.0f, 0.0f), vec2(-5.0f, 4.0f), vec2(-2.5f, 3.5f)})
  {
    Lum image(320, 200, 0.0f);
    for(int y = 0; y < image.h; y++)
      for(int x = 0; x < image.w; x++)
        image.at(x, y) = level(x < 157 ? sides.x : sides.y);  // the edge inside a cell
    const std::vector<float> g    = gains(image, 1.0f, 1.0f);
    const float              dark = lexpGain(sides.x, 1.0f, 1.0f), bright = lexpGain(sides.y, 1.0f, 1.0f);
    // the splat's 2x2 block across the edge (pixels 156 and 157) is a sample between the sides' levels,
    // a bin or so below the bright one past the knee: within 8 pixels of the edge on that side, a
    // fringe of at most 0.07 stops (0.061 at -5 against +4); everywhere else, each side's own gain
    for(int y = 0; y < image.h; y++)
      for(int x = 0; x < image.w; x++)
        ASSERT_NEAR(g[size_t(y) * size_t(image.w) + size_t(x)], x < 157 ? dark : bright, (x >= 157 && x < 165) ? 0.07f : 0.02f)
            << "(" << x << ", " << y << ") between " << sides.x << " and " << sides.y;
  }
}

// Fine texture inside a dark region keeps most of its contrast: the region is lifted as one, where a
// curve on each pixel alone would take back LEXP_SLOPE of it
TEST(LocalExposure, ARegionsTextureKeepsItsContrast)
{
  Lum image(320, 200, 0.0f);
  for(int y = 0; y < image.h; y++)
    for(int x = 0; x < image.w; x++)
      image.at(x, y) = level(-5.0f + (((x + y) & 1) != 0 ? 0.25f : -0.25f));  // a checker, half a stop
  const std::vector<float> g = gains(image, 1.0f, 0.0f);
  double                   before = 0.0, after = 0.0;
  int                      pairs  = 0;
  for(int y = 40; y < 160; y++)
    for(int x = 40; x < 280; x += 2)
    {
      const size_t i  = size_t(y) * size_t(image.w) + size_t(x);
      const float  s0 = lexpStops(image.at(x, y), 1.0f), s1 = lexpStops(image.at(x + 1, y), 1.0f);
      before += std::abs(s1 - s0);
      after += std::abs((s1 + g[i + 1]) - (s0 + g[i]));
      pairs++;
    }
  before /= pairs;
  after /= pairs;
  EXPECT_NEAR(before, 0.5, 1e-3);
  EXPECT_GT(after, 0.8 * before) << "the texture flattened";
  EXPECT_LE(after, before + 1e-3) << "the texture sharpened";
  // and the region rose by about the curve's gain at its level
  EXPECT_NEAR(g[size_t(100) * size_t(image.w) + 100], lexpGain(-5.0f, 1.0f, 0.0f), 0.15f);
}
