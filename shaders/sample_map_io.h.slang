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
// Host/device interface of the Samples view (src/sample_map.hpp, shaders/sample_map.slang): each pixel
// of a still coloured by the samples it took before the adaptive stop let it go.
//

#ifndef SAMPLE_MAP_IO_H
#define SAMPLE_MAP_IO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define SAMPLE_MAP_WORKGROUP_SIZE 16

// Push descriptors, in VK_IMAGE_LAYOUT_GENERAL
enum SampleMapBindings
{
  eSampleMapState = 0,  // RWTexture2D: the path tracer's convergence image (OutputImage::eConvergence, RG32F)
  eSampleMapOut   = 1,  // RWTexture2D: the image shown, in place of the tone mapper's (RGBA8)
};

struct SampleMapPushConstant
{
  int2 outSize;     // the image's size: the dispatch
  int  minSamples;  // the stop's minimum (ptAdaptiveMin): the scale's low end
  int  cap;         // the still's cap (maxSamples): its high end
};

NAMESPACE_SHADERIO_END()

#endif  // SAMPLE_MAP_IO_H
