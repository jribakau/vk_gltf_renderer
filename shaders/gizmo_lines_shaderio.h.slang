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

// Shader interface for world-space helper lines (camera frustums, light markers).
// Shared between C++ and gizmo_lines.slang.

#pragma once

#ifdef __cplusplus
#include <glm/glm.hpp>
#include "nvshaders/slang_types.h"
namespace shaderio::lines {
#endif

// One end of a line: every two make one segment.
struct LineVertex
{
  float3 position;  // world space
  uint   color;     // RGBA8, R in the low byte
};

struct LinePushConstants
{
  float4x4    viewProj;       // what the lines are drawn with (a wide depth range)
  float4x4    sceneViewProj;  // the scene camera's: the depth test reprojects into it
  float2      viewportSize;
  float2      depthBufferSize;
  int         isPerspective;
  float       occludedAlpha;  // how much of a line behind geometry still shows
  int         padding0;
  int         padding1;
  LineVertex* vertices;       // buffer device address
};

#ifdef __cplusplus
}  // namespace shaderio::lines
#endif
