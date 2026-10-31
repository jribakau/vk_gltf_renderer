/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan_core.h>

namespace nvvkgltf {

// The skinning and morph compute pipelines. Each uses only push constants (shaders/animation_io.h.slang):
// all buffer addresses are passed via BDA pointers.
struct DeformPipelines
{
  VkPipeline       skin{}, morph{}, stencil{}, normals{}, tangents{};
  VkPipelineLayout skinLayout{}, morphLayout{}, stencilLayout{}, normalsLayout{}, tangentsLayout{};
  void             create(VkDevice device);
  void             destroy(VkDevice device);
};

}  // namespace nvvkgltf
