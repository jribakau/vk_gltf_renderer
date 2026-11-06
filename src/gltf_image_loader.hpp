/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "image_levels.hpp"

namespace nvvkgltf {

// CPU-side result of decoding an image from raw bytes (disk, buffer, or embedded).
// What ImageLoader decodes; createSampledImage makes the Vulkan image of it.
struct LoadedImageData
{
  VkFormat                       format{VK_FORMAT_UNDEFINED};
  VkExtent2D                     size{0, 0};
  std::vector<std::vector<char>> mipData{};
  VkComponentMapping             componentMapping{};
  // one value everywhere: a map that held it over its whole chain, kept as the one block it was made
  // of (a 4x4 image of one level), whose every texel is the map's mean
  bool constant = false;
};

// Decodes raw image bytes into LoadedImageData. Dispatches by magic bytes:
// DDS, KTX (1/2), WebP, or falls back to stb_image (PNG, JPEG, etc.).
// srgb: when true, format may be forced to an sRGB variant where applicable.
// imageIDForLog: used only for log messages (e.g. "image 3").
// maxExtent: the levels above this long edge are left out (rs::levelsToSkip), where the image stores
// levels; a KTX file doesn't inflate them. 0: no limit.
// downscale: 2, 4 or 8 decodes a JPEG at that fraction of its size, in the decode itself (libjpeg-turbo's
// scaled IDCT), which is quicker than the whole; other formats, and other values, decode whole.
// Returns true if decoding succeeded and out is filled; false on failure or unsupported format.
[[nodiscard]] bool loadFromMemory(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog = 0, uint32_t maxExtent = 0,
                                  uint32_t downscale = 1);

// The levels above a long edge of `maxExtent` left out of an image already decoded (a prefetch)
void trimLevels(LoadedImageData& image, uint32_t maxExtent);

// A KTX file's mip chain, from its header alone; none for anything else
std::optional<rs::ImageChain> readImageChain(const void* data, size_t byteLength);

// The mean luminance (linear, Rec. 709) of an 8-bit image's texels, from a grid of at most 64 x 64 of
// them; negative for a format it does not read (compressed, float). An emissive texture's weight as a
// light (GltfShadeMaterial::emissiveLightScale).
float imageMeanLuminance(VkFormat format, VkExtent2D size, const std::vector<char>& texels);

// The same of a decoded image as a sampler reads it. An 8-bit one as above, from its first level. A
// BC7 or BC4 one from its smallest level when that is 1 x 1, as every texture the app cooks has it
// (texcache.rs `cook`): the image averaged in linear light, so its one texel is the mean, and through
// the image's swizzle. Negative where unknown (other formats, a chain that stops short of 1 x 1).
float imageMeanLuminance(const LoadedImageData& image);

// The top-left texel of a BC7 block, RGBA: a 1 x 1 level's only one. Texel 0 is subset 0's anchor
// under every partition, so this needs the mode table and the weights only, not the partitions.
std::array<uint8_t, 4> bc7FirstTexel(const uint8_t block[16]);
// The same of a BC4 block, its one channel
uint8_t bc4FirstTexel(const uint8_t block[8]);

// Number of color components a sampled VkFormat actually carries. Vulkan synthesizes the missing
// ones (0 for G/B, 1 for A), so a consumer that needs a component the format does not store must
// know. Only the formats the image loaders can produce are listed; anything else counts 4, which
// disables the special-casing rather than guessing wrong.
uint32_t formatColorComponentCount(VkFormat format);

}  // namespace nvvkgltf
