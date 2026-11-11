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

#pragma once

#include <nvvk/descriptors.hpp>

#include "resources.hpp"
#include "shaders/sample_map_io.h.slang"

//-------------------------------------------------------------------------------------------------------
// SampleMap
//
// The Samples view: each pixel of a still in viridis by the samples it took before the adaptive stop
// let it go (shaders/sample_map.slang), from the stop's minimum (violet) to the still's cap (yellow), on
// a log scale. Where a finished still's leftover noise was accepted, and where it ran to the cap.
//
//   map.cmdDraw(cmd, counts, out, size, cap)  // in runTonemapPass, in place of the tone mapper
//
// It reads the path tracer's convergence image (PathTracer::sampleCounts), which holds counts only on
// a still with the stop on: the moving view and an exact still show the picture instead. Nothing is
// allocated; it writes the image the tone mapper would.
//-------------------------------------------------------------------------------------------------------
class SampleMap
{
public:
  void init(Resources& resources);
  void deinit(Resources& resources);

  // `counts`: the convergence image (storage); `out`: the tone mapper's output (storage), `size` both
  void cmdDraw(VkCommandBuffer cmd, const VkDescriptorImageInfo& counts, const VkDescriptorImageInfo& out,
               VkExtent2D size, int minSamples, int cap);

private:
  VkDevice                 m_device = VK_NULL_HANDLE;
  nvvk::DescriptorBindings m_bindings;
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipelineLayout         m_pipelineLayout{};
  VkPipeline               m_pipeline{};
};
