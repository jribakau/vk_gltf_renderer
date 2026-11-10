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
// Host/device interface of the lens pass (src/lens.hpp, shaders/lens.slang): a cheap wide lens's
// distortion and colour fringing on the image the tone mapper takes.
//

#ifndef LENS_IO_H
#define LENS_IO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define LENS_WORKGROUP_SIZE 16

// Push descriptors, in VK_IMAGE_LAYOUT_GENERAL
enum LensBindings
{
  eLensIn  = 0,  // Sampler2D: the image as rendered, linear and clamped to its edge
  eLensOut = 1,  // RWTexture2D: the picture through the lens (RGBA16F)
};

struct LensPushConstant
{
  int2  outSize;  // the image's size: the dispatch
  float bend;     // barrel distortion, 0 none to 1 (LENS_MAX_BEND)
  float fringe;   // colour fringing, 0 none to 1 (LENS_MAX_FRINGE)
};

NAMESPACE_SHADERIO_END()

#endif  // LENS_IO_H
