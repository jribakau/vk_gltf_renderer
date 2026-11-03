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
// Host/device interface of HDR display output (src/hdr_output.hpp, shaders/hdr_output.slang): the
// image an HDR display is shown, in scRGB half floats, beside the SDR image everything else takes.
//

#ifndef HDR_OUTPUT_IO_H
#define HDR_OUTPUT_IO_H 1

#include "nvshaders/slang_types.h"
#include "nvshaders/tonemap_io.h.slang"

NAMESPACE_SHADERIO_BEGIN()

// Push descriptors, in VK_IMAGE_LAYOUT_GENERAL
enum HdrOutputBindings
{
  eHdrInput   = 0,  // Sampler2D: the tonemapper's input (the rendered or denoised image)
  eHdrSdr     = 1,  // RWTexture2D: the SDR image (RGBA8, sRGB values), after the overlays
  eHdrSdrKept = 2,  // RWTexture2D: the SDR image as the tonemapper left it, before the overlays
  eHdrOut     = 3,  // RWTexture2D: the HDR image (RGBA16F, scRGB: linear, 1.0 = 80 nits), or an EXR export's (RGBA32F)
};

// The display's levels first, then the tonemapper's settings as the SDR image was made with
struct HdrOutputPushConstant
{
  float          paperWhite;  // SDR white in scRGB: the display's SDR white level over 80 nits
  float          peak;        // The display's peak in SDR whites, at least 1
  int            whole;       // composeHdr: every pixel from the SDR image (nothing tone mapped for HDR)
  int            _pad;
  TonemapperData tm;          // As the SDR tonemapper took it, inputMatrix filled
};

NAMESPACE_SHADERIO_END()

#endif  // HDR_OUTPUT_IO_H
