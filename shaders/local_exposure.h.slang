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

#ifndef FV_LOCAL_EXPOSURE_H
#define FV_LOCAL_EXPOSURE_H

// Local exposure's grid and curve (src/local_exposure.hpp, shaders/local_exposure.slang), Slang and
// C++ (tests/test_local_exposure.cpp), so the tests run the functions the shader runs.
//
// A bilateral grid (Chen, Paris and Durand 2007; Unreal's local exposure): the image's log luminance,
// in stops from mid grey after the tone mapper's exposure, averaged over cells of the image by bins of
// that luminance, blurred a cell and a bin each way, and read back at each pixel's own place and
// luminance. That is the brightness of the region the pixel belongs to, a region ending where the
// luminance jumps by more than a bin or two, so a dark face against a bright wall is a region of its
// own and a halo can't form round it. The region's level goes through a curve (lexpGain): within
// LEXP_KNEE stops of mid grey nothing moves, beyond it the region is brought toward mid grey by
// LEXP_SLOPE times the slider's amount of each stop further out (shadows lifted, highlights held),
// at most LEXP_MAX_GAIN stops; every pixel of the region gets the region's gain, so its detail and
// texture keep their contrast exactly.

static const int   LEXP_GRID_ROWS = 40;      // cells across the image's height
static const int   LEXP_MIN_CELL  = 8;       // a cell's side at least this, in pixels
static const int   LEXP_BINS      = 32;      // luminance bins of a cell
static const float LEXP_LOW       = -16.0F;  // the lowest bin's lower edge, stops from mid grey
static const float LEXP_HIGH      = 8.0F;    // the highest bin's upper edge (a bin is 0.75 stops)
static const float LEXP_MID_GREY  = 0.18F;   // the level the curve pivots on, after the exposure
static const float LEXP_KNEE      = 1.0F;    // stops each side of mid grey the curve leaves alone
static const float LEXP_SOFT      = 0.5F;    // the knee's half-width, stops: the curve bends over twice this
static const float LEXP_SLOPE     = 0.6F;    // at an amount of 1, the share of each stop past the knee taken back
static const float LEXP_MAX_GAIN  = 3.0F;    // stops at most, either way
static const float LEXP_FIXED     = 256.0F;  // the splat's fixed point: stops in 1/256ths, summed in integers

// A cell's side in pixels for an image this tall: LEXP_GRID_ROWS of them across it, so the regions
// are the same share of the frame at every window size; even, as the splat reads 2x2 blocks
int lexpCellSize(int height)
{
  const int cell = max(LEXP_MIN_CELL, (height + LEXP_GRID_ROWS - 1) / LEXP_GRID_ROWS);
  return cell + (cell & 1);
}

// The cells across a side of `side` pixels
int lexpCells(int side, int cell)
{
  return (side + cell - 1) / cell;
}

// A pixel's luminance as stops from mid grey after `exposure`, held to [-24, 24] (black is very low,
// not minus infinity)
float lexpStops(float luminance, float exposure)
{
  return clamp(log2(max(luminance * exposure, 1e-12F) / LEXP_MID_GREY), -24.0F, 24.0F);
}

// Its bin's coordinate in the grid, continuous, bin k's centre at k: where the slice interpolates
float lexpBinCoord(float stops)
{
  return clamp((stops - LEXP_LOW) / (LEXP_HIGH - LEXP_LOW) * float(LEXP_BINS) - 0.5F, 0.0F, float(LEXP_BINS - 1));
}

// The bin a pixel splats into
int lexpBin(float stops)
{
  return clamp(int(floor((stops - LEXP_LOW) / (LEXP_HIGH - LEXP_LOW) * float(LEXP_BINS))), 0, LEXP_BINS - 1);
}

// The grid's index of cell (x, y)'s bin
int lexpIndex(int x, int y, int bin, int cellsX)
{
  return (y * cellsX + x) * LEXP_BINS + bin;
}

// The blur's weight `d` cells or bins off the centre (-1, 0, 1): 1 2 1 / 4
float lexpBlurWeight(int d)
{
  return d == 0 ? 0.5F : 0.25F;
}

// u past the knee, bent over [-LEXP_SOFT, LEXP_SOFT] so the curve has no corner: 0 below, u above
float lexpSoftKnee(float u)
{
  if(u <= -LEXP_SOFT)
    return 0.0F;
  if(u >= LEXP_SOFT)
    return u;
  return (u + LEXP_SOFT) * (u + LEXP_SOFT) / (4.0F * LEXP_SOFT);
}

// The gain in stops for a region `base` stops from mid grey: lifted by `shadows` below the knee, held
// by `highlights` above it (each 0 to 1). The region's new level, base + gain, never falls as base
// rises (the slope is at least 1 - LEXP_SLOPE), so no tone overtakes another.
float lexpGain(float base, float shadows, float highlights)
{
  const float lift = min(LEXP_SLOPE * shadows * lexpSoftKnee(-base - LEXP_KNEE), LEXP_MAX_GAIN);
  const float hold = min(LEXP_SLOPE * highlights * lexpSoftKnee(base - LEXP_KNEE), LEXP_MAX_GAIN);
  return lift - hold;
}

#endif  // FV_LOCAL_EXPOSURE_H
