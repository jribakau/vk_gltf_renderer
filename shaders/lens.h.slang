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

// A cheap wide lens's defects (src/lens.hpp), shared with the unit tests: where in the image as
// rendered (through a perfect lens) a pixel of the picture takes its light from.
//
//   - Barrel distortion: the picture is the rendered image drawn in toward its corners, most at the
//     centre, so the middle is larger, the edges are squeezed and straight lines bow outward. The
//     corners stay where they are: nothing from outside the rendered frame is needed, and all of it
//     along the diagonals is in the picture.
//   - Fringing (lateral chromatic aberration): red is drawn a little larger than green and blue a
//     little smaller, by nothing at the centre and most at the corners, so edges away from the centre
//     split into colours.
//
// Positions are uv (0 to 1 across the image and down it). `aspect` is the image's width over its height.

#ifndef LENS_H_SLANG
#define LENS_H_SLANG

// The distortion at bend 1, as the k of r' = r (1 + k r^2) / (1 + k) on the half diagonal: the
// centre 1.6 times larger, about a fisheye action camera's
static const float LENS_MAX_BEND = 0.6F;
// The fringing at 1: how much larger red is drawn at the corners, and blue smaller (some 6 pixels
// each on a 1080-line image)
static const float LENS_MAX_FRINGE = 0.006F;

// A position from the image's centre, in half diagonals: the corners are at length 1
inline float2 lensFromUv(float2 uv, float aspect)
{
  const float2 half = float2(aspect, 1.0F) / sqrt(aspect * aspect + 1.0F);
  return (uv * 2.0F - float2(1.0F)) * half;
}

inline float2 lensToUv(float2 q, float aspect)
{
  const float2 half = float2(aspect, 1.0F) / sqrt(aspect * aspect + 1.0F);
  return (q / half) * 0.5F + float2(0.5F);
}

// How far from the centre the rendered image is read for a picture's point at squared distance `r2`,
// over that point's own distance: under 1 inside the corners' circle, 1 on it
inline float lensBarrel(float r2, float bend)
{
  const float k = bend * LENS_MAX_BEND;
  return (1.0F + k * r2) / (1.0F + k);
}

// Where the rendered image is read for the picture at `uv`: green's place, and with `channel` -1
// red's, +1 blue's
inline float2 lensSource(float2 uv, float aspect, float bend, float fringe, float channel)
{
  const float2 q  = lensFromUv(uv, aspect);
  const float  r2 = dot(q, q);
  // red drawn larger is red read nearer the centre
  const float colour = 1.0F + channel * fringe * LENS_MAX_FRINGE * r2;
  return lensToUv(q * (lensBarrel(r2, bend) * colour), aspect);
}

#endif  // LENS_H_SLANG
