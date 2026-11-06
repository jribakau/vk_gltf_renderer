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

// What emissive_texels.comp.slang averages: a textured emitter's triangles in its texture's UV space
// (texture transform applied), and their mean colour there, which weighs each one as a light
// (nvvkgltf::Emitter::triangleLuminance). After Falcor's EmissiveIntegrator (BSD-3).

#ifndef EMISSIVE_TEXELS_IO_H
#define EMISSIVE_TEXELS_IO_H

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// One triangle to average: its corners' UVs and the bindless slots of the texture and its sampler
struct TexelTriangle
{
  float2 uv0;
  float2 uv1;
  float2 uv2;
  uint   texture;  // eTextures slot
  uint   sampler;  // eSamplers slot
};

struct EmissiveTexelsPushConstant
{
  TexelTriangle* triangles;
  float4*        means;  // per triangle: its mean linear colour (rgb), a
  uint           count;
};

// points a triangle is averaged over, stratified over its area at a level of detail that spreads them
// over its texels
static const uint EMISSIVE_TEXELS_SAMPLES   = 64;
static const uint EMISSIVE_TEXELS_WORKGROUP = 64;

NAMESPACE_SHADERIO_END()

#endif
