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

#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/uploader_interface.hpp>

namespace nvvkgltf {

// A sampled image from decoded mip levels: created, its levels uploaded through `staging` (in row
// strips under the uploader's block size), the rest of its chain generated when there is one level,
// `generateMipmaps` is set and the format can be blitted, and left in SHADER_READ_ONLY_OPTIMAL.
// False, with nothing created, for an empty image, a format the device can't sample, or GPU memory
// that can't hold it (`outOfMemory` set then).
bool createSampledImage(VkCommandBuffer                       cmd,
                        nvvk::CmdUploaderInterface&           staging,
                        nvvk::ResourceAllocator&              alloc,
                        VkPhysicalDevice                      physicalDevice,
                        bool                                  generateMipmaps,
                        VkFormat                              format,
                        VkExtent2D                            size,
                        const std::vector<std::vector<char>>& mipData,
                        const VkComponentMapping&             componentMapping,
                        const std::string&                    name,
                        nvvk::Image&                          out,
                        bool*                                 outOfMemory = nullptr);

}  // namespace nvvkgltf
