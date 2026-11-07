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

//
// Host/device interface of local exposure (src/local_exposure.hpp, shaders/local_exposure.slang):
// shadows lifted and highlights held, region by region, on the image the tone mapper takes.
//

#ifndef LOCAL_EXPOSURE_IO_H
#define LOCAL_EXPOSURE_IO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define LEXP_WORKGROUP_SIZE 16  // the splat's and the apply's: 16 x 16
#define LEXP_BLUR_GROUP 64      // the blur's: one thread a cell's bin

// Push descriptors, the image in VK_IMAGE_LAYOUT_GENERAL
enum LocalExposureBindings
{
  eLexpImage   = 0,  // Sampler2D: the image (the tone mapper's input), linear, clamped to the edge
  eLexpGrid    = 1,  // RWStructuredBuffer<float2>: the splat's (stops summed, 2x2 blocks), a cell's bins in a row
  eLexpBlurred = 2,  // RWStructuredBuffer<float2>: the grid blurred, as the apply reads it
  eLexpOut     = 3,  // RWTexture2D: the image with its gain (RGBA16F)
};

struct LocalExposurePushConstant
{
  int2  size;        // the image's
  int2  cells;       // the grid's cells across and down
  int   cell;        // a cell's side, pixels (even)
  float exposure;    // the tone mapper's: the stops are counted from mid grey after it
  float shadows;     // lift, 0 to 1
  float highlights;  // hold, 0 to 1
};

NAMESPACE_SHADERIO_END()

#endif  // LOCAL_EXPOSURE_IO_H
