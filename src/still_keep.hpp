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

#include "shaders/shaderio.h"

//-------------------------------------------------------------------------------------------------------
// StillKeep: the still's per-pixel Keep detail (shaders/still_keep.slang). Over the still's denoised
// image, each pixel whose accumulation has converged takes back its raw value, by the share
// stillKeepShare gives its noise (shaders/still_keep.h.slang); the denoised image stays where the
// accumulation is still noisy. Storage images in VK_IMAGE_LAYOUT_GENERAL; the denoised one is written
// in place. The caller orders the writes before it and the reads after it with barriers.
//-------------------------------------------------------------------------------------------------------
class StillKeep
{
public:
  void init(VkDevice device);
  void deinit(VkDevice device);
  void cmdKeep(VkCommandBuffer                     cmd,
               const VkDescriptorImageInfo&        denoised,
               const VkDescriptorImageInfo&        raw,
               const VkDescriptorImageInfo&        moments,
               VkExtent2D                          size,
               const shaderio::StillKeepPushConstant& pushConstant);

private:
  nvvk::DescriptorBindings m_bindings;
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipelineLayout         m_pipelineLayout{};
  VkPipeline               m_pipeline{};
};
