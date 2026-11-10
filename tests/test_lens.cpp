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

// The lens pass's mapping (shaders/lens.h.slang) compiled as C++: where in the image as rendered a
// point of the picture takes its light from. The corners and the centre stay, the middle is drawn
// larger and the edges squeezed, nothing outside the rendered frame is read for the bend, and the
// fringes put red outside green and blue inside it, by nothing at the centre.

#include <glm/glm.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace glm;
#include <nvshaders/slang_types.h>
using namespace shaderio;
#include "../shaders/lens.h.slang"

namespace {

constexpr float kAspect = 16.0F / 9.0F;

void expectNear2(vec2 a, vec2 b, float tolerance = 1e-5F)
{
  EXPECT_NEAR(a.x, b.x, tolerance);
  EXPECT_NEAR(a.y, b.y, tolerance);
}

}  // namespace

TEST(Lens, WithoutBendOrFringeThePictureIsTheImage)
{
  for(const vec2 uv : {vec2(0.0F), vec2(0.5F), vec2(1.0F), vec2(0.13F, 0.87F), vec2(0.9F, 0.2F)})
    for(float channel : {-1.0F, 0.0F, 1.0F})
      expectNear2(lensSource(uv, kAspect, 0.0F, 0.0F, channel), uv, 1e-6F);
}

TEST(Lens, TheCentreAndTheCornersStay)
{
  for(float bend : {0.25F, 0.55F, 1.0F})
  {
    expectNear2(lensSource(vec2(0.5F), kAspect, bend, 0.0F, 0.0F), vec2(0.5F));
    for(const vec2 corner : {vec2(0.0F, 0.0F), vec2(1.0F, 0.0F), vec2(0.0F, 1.0F), vec2(1.0F, 1.0F)})
      expectNear2(lensSource(corner, kAspect, bend, 0.0F, 0.0F), corner);
  }
  // a square image's too, and an upright one's
  expectNear2(lensSource(vec2(1.0F, 0.0F), 1.0F, 1.0F, 0.0F, 0.0F), vec2(1.0F, 0.0F));
  expectNear2(lensSource(vec2(0.0F, 1.0F), 9.0F / 16.0F, 1.0F, 0.0F, 0.0F), vec2(0.0F, 1.0F));
}

// The bend reads nearer the centre everywhere inside the corners' circle: the whole picture is filled
// from inside the rendered frame
TEST(Lens, TheBendReadsNothingOutsideTheFrame)
{
  for(float bend : {0.1F, 0.55F, 1.0F})
    for(int y = 0; y <= 20; y++)
      for(int x = 0; x <= 20; x++)
      {
        const vec2 uv = vec2(float(x), float(y)) / 20.0F;
        const vec2 s  = lensSource(uv, kAspect, bend, 0.0F, 0.0F);
        EXPECT_GE(s.x, -1e-6F);
        EXPECT_LE(s.x, 1.0F + 1e-6F);
        EXPECT_GE(s.y, -1e-6F);
        EXPECT_LE(s.y, 1.0F + 1e-6F);
        // toward the centre, never past it
        EXPECT_LE(length(s - vec2(0.5F)), length(uv - vec2(0.5F)) + 1e-6F);
        EXPECT_GE(dot(s - vec2(0.5F), uv - vec2(0.5F)), 0.0F);
      }
}

// Barrel distortion: the middle of the image is drawn 1 + k times larger, the corners smaller than
// they were, and what lies further out in the picture lies further out in the image (no fold)
TEST(Lens, TheMiddleIsLargerAndTheEdgesSqueezed)
{
  const float bend = 0.5F;
  const float k    = bend * LENS_MAX_BEND;
  const float step = 1e-3F;
  // a small step at the centre of the picture covers 1 / (1 + k) of it in the image
  const vec2 c = lensSource(vec2(0.5F + step, 0.5F), kAspect, bend, 0.0F, 0.0F) - lensSource(vec2(0.5F), kAspect, bend, 0.0F, 0.0F);
  EXPECT_NEAR(c.x / step, 1.0F / (1.0F + k), 1e-3F);
  // and along the diagonal at the corner (1 + 3k) / (1 + k) of it: more image in less picture
  const vec2 d = vec2(step) * 0.5F;
  const vec2 e = lensSource(vec2(1.0F), kAspect, bend, 0.0F, 0.0F) - lensSource(vec2(1.0F) - d, kAspect, bend, 0.0F, 0.0F);
  EXPECT_NEAR(e.x / d.x, (1.0F + 3.0F * k) / (1.0F + k), 5e-3F);
  // outward in the picture is outward in the image, along a diagonal and along the middle row
  float before = -1.0F;
  for(int i = 0; i <= 50; i++)
  {
    const float t = 0.5F + 0.5F * float(i) / 50.0F;
    const float r = length(lensFromUv(lensSource(vec2(t), kAspect, 1.0F, 0.0F, 0.0F), kAspect));
    EXPECT_GT(r, before);
    before = r;
  }
  // a straight line off the centre bows outward: its middle is read nearer the centre than its ends'
  // row, so the image's row there is drawn further out in the picture
  const float rowEnd = lensSource(vec2(0.0F, 0.1F), kAspect, bend, 0.0F, 0.0F).y;
  const float rowMid = lensSource(vec2(0.5F, 0.1F), kAspect, bend, 0.0F, 0.0F).y;
  EXPECT_GT(rowMid, rowEnd);
}

TEST(Lens, TheFringesSplitRedOutwardAndBlueInwardAwayFromTheCentre)
{
  // none at the centre
  for(float channel : {-1.0F, 1.0F})
    expectNear2(lensSource(vec2(0.5F), kAspect, 0.0F, 1.0F, channel), vec2(0.5F), 1e-7F);
  // at a corner red is read nearer the centre (drawn larger), blue further out, by LENS_MAX_FRINGE of
  // the half diagonal each
  const vec2  corner = vec2(1.0F, 1.0F);
  const float green  = length(lensFromUv(lensSource(corner, kAspect, 0.0F, 1.0F, 0.0F), kAspect));
  const float red    = length(lensFromUv(lensSource(corner, kAspect, 0.0F, 1.0F, -1.0F), kAspect));
  const float blue   = length(lensFromUv(lensSource(corner, kAspect, 0.0F, 1.0F, 1.0F), kAspect));
  EXPECT_NEAR(green, 1.0F, 1e-5F);
  EXPECT_NEAR(red, 1.0F - LENS_MAX_FRINGE, 1e-5F);
  EXPECT_NEAR(blue, 1.0F + LENS_MAX_FRINGE, 1e-5F);
  // growing with the square of the distance: a quarter of it halfway out
  const vec2  half     = vec2(0.75F, 0.75F);
  const float redHalf  = length(lensFromUv(lensSource(half, kAspect, 0.0F, 1.0F, -1.0F), kAspect));
  EXPECT_NEAR(0.5F - redHalf, 0.5F * LENS_MAX_FRINGE * 0.25F, 1e-5F);
  // at half the strength, half of it
  const float redWeak = length(lensFromUv(lensSource(corner, kAspect, 0.0F, 0.5F, -1.0F), kAspect));
  EXPECT_NEAR(1.0F - redWeak, 0.5F * LENS_MAX_FRINGE, 1e-5F);
}

TEST(Lens, UvAndLensPositionsAreEachOthersInverse)
{
  for(const vec2 uv : {vec2(0.0F), vec2(0.3F, 0.9F), vec2(1.0F, 0.25F)})
    for(float aspect : {kAspect, 1.0F, 0.5F})
      expectNear2(lensToUv(lensFromUv(uv, aspect), aspect), uv, 1e-6F);
  // the corners are one half diagonal out
  EXPECT_NEAR(length(lensFromUv(vec2(1.0F, 0.0F), kAspect)), 1.0F, 1e-6F);
}
