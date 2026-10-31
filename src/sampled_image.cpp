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

#include "sampled_image.hpp"

#include <algorithm>
#include <cstring>
#include <span>

#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/mipmaps.hpp>

#include "staging_config.hpp"

namespace {

// Returns the Y-axis dimension of the format's texel block (in texels): 1 for
// uncompressed formats, 4 for BC/ETC/EAC block-compressed formats, and the ASTC
// block height for VK_FORMAT_ASTC_*x*_*_BLOCK (4, 5, 6, 8, 10 or 12). Used to
// keep row-strip uploads on block boundaries.
uint32_t getFormatBlockHeight(VkFormat format)
{
  switch(format)
  {
    // BC1..BC7 -- all 4x4.
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
    // ETC2 / EAC -- all 4x4.
    case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
    case VK_FORMAT_EAC_R11_UNORM_BLOCK:
    case VK_FORMAT_EAC_R11_SNORM_BLOCK:
    case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
    case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
    // ASTC with block height 4.
    case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
    case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
    case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
    case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
      return 4;
    // ASTC with block height 5.
    case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
    case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
    case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
    case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
      return 5;
    // ASTC with block height 6.
    case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
    case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
    case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
      return 6;
    // ASTC with block height 8.
    case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
    case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
      return 8;
    // ASTC with block height 10.
    case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
    case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
    case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
      return 10;
    // ASTC with block height 12.
    case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
    case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
      return 12;
    default:
      return 1;
  }
}

}  // namespace

bool nvvkgltf::createSampledImage(VkCommandBuffer                       cmd,
                                  nvvk::CmdUploaderInterface&           staging,
                                  nvvk::ResourceAllocator&              alloc,
                                  VkPhysicalDevice                      physicalDevice,
                                  bool                                  generateMipmaps,
                                  VkFormat                              format,
                                  VkExtent2D                            imgSize,
                                  const std::vector<std::vector<char>>& mipData,
                                  const VkComponentMapping&             componentMapping,
                                  const std::string&                    name,
                                  nvvk::Image&                          out,
                                  bool*                                 outOfMemory)
{
  if(imgSize.width == 0 || imgSize.height == 0 || mipData.empty())
    return false;

  // Check if we can generate mipmap with the the incoming image
  bool               canGenerateMipmaps = false;
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);

  // Skip formats the device cannot sample (e.g. ASTC from EXT_texture_astc on desktop GPUs).
  // Returning false lets the caller substitute a default image instead of crashing in vkCreateImage.
  if((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0)
  {
    LOGW("Image '%s' uses a format (%d) not supported by this device; skipping (using default image).\n", name.c_str(),
         static_cast<int>(format));
    return false;
  }

  if((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == VK_FORMAT_FEATURE_BLIT_DST_BIT)
  {
    canGenerateMipmaps = true;
  }
  VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
  imageCreateInfo.extent            = VkExtent3D{imgSize.width, imgSize.height, 1};
  imageCreateInfo.format            = format;
  imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  // Mip-mapping images were defined (.ktx, .dds), use the number of levels defined
  if(mipData.size() > 1)
  {
    imageCreateInfo.mipLevels = static_cast<uint32_t>(mipData.size());
  }
  else if(canGenerateMipmaps && generateMipmaps)
  {
    // Compute the number of mipmaps levels
    imageCreateInfo.mipLevels = nvvk::mipLevels(imgSize);
  }

  // Use custom view info with component mapping (e.g. for grayscale textures)
  VkImageViewCreateInfo imageViewCreateInfo = DEFAULT_VkImageViewCreateInfo;
  imageViewCreateInfo.components            = componentMapping;

  nvvk::Image resultImage;
  if(const VkResult result = alloc.createImage(resultImage, imageCreateInfo, imageViewCreateInfo); result != VK_SUCCESS)
  {
    LOGW("Image '%s' (%ux%u): no GPU memory for it (VkResult %d); using the default image.\n", name.c_str(), imgSize.width,
         imgSize.height, int(result));
    if(outOfMemory)
      *outOfMemory = true;
    return false;
  }
  NVVK_DBG_NAME(resultImage.image);
  NVVK_DBG_NAME(resultImage.descriptor.imageView);

  // Set the initial layout to TRANSFER_DST_OPTIMAL
  resultImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;  // Setting this, tells the appendImage that the image is in this layout (no need to transfer)
  nvvk::cmdImageMemoryBarrier(cmd, {resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED, resultImage.descriptor.imageLayout});

  // Upload one mip level as a full-image append when it fits, or split it into
  // horizontal row-strips when it exceeds the staging block-size limit
  // (nvvk::BufferCircularAllocator asserts size <= blockSize per subAllocate).
  // Row-strip chunking keeps the pattern used by SceneVk (linear per-row layout)
  // and works for arbitrarily large mips (e.g. 8K/16K, 16-bpc, HDR32F).
  auto uploadMipLevel = [&](uint32_t mip, const VkExtent3D& mipExtent, std::span<const char> mipBytes) {
    VkImageSubresourceLayers subresource{};
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.layerCount = 1;
    subresource.mipLevel   = mip;

    // Per-append cap matches the FrameUploader block size configured in renderer.cpp
    // (`kFrameUploaderBlockSize` in staging_config.hpp). Sharing the single symbol
    // keeps this chunking in lock-step with the app-owned `blockSize`, so lowering
    // that value cannot silently reintroduce the assertion at
    // buffer_circular_allocator.cpp:236 (`size <= blockSize`).
    constexpr size_t kAppendCap = static_cast<size_t>(kFrameUploaderBlockSize);

    const size_t   mipSize     = mipBytes.size();
    const uint32_t height      = std::max(1u, mipExtent.height);
    const size_t   bytesPerRow = mipSize / height;

    // If the mip fits in one shot, or we can't derive a clean per-row size
    // (e.g. compressed formats with padding), fall back to a single append.
    // Compressed KTX/DDS payloads today are far smaller than kAppendCap, so this
    // is a safe fall-back for the current asset formats.
    if(mipSize <= kAppendCap || bytesPerRow == 0 || bytesPerRow * height != mipSize)
    {
      NVVK_CHECK(staging.appendImageSub(resultImage, {0, 0, 0}, mipExtent, subresource, mipSize, mipBytes.data(),
                                        resultImage.descriptor.imageLayout));
      return;
    }

    // Rows per strip = as many full rows as fit under the cap, aligned down to the
    // format's texel-block height so block-compressed formats stay on block
    // boundaries: 1 for uncompressed, 4 for BC/ETC/EAC, and 4/5/6/8/10/12 for the
    // various ASTC variants (see getFormatBlockHeight()). Cropping a strip mid-block
    // would send partial blocks to vkCmdCopyBufferToImage and corrupt the upload.
    const uint32_t blockH       = getFormatBlockHeight(format);
    uint32_t       rowsPerStrip = static_cast<uint32_t>(kAppendCap / bytesPerRow);
    rowsPerStrip -= rowsPerStrip % blockH;  // align down to a whole number of block rows
    if(rowsPerStrip == 0)
    {
      // The cap can't hold even a single block row. Fall back to a single append;
      // if the mip still exceeds the cap the underlying allocator will assert, matching
      // the pre-existing behaviour for pathologically wide compressed mips.
      NVVK_CHECK(staging.appendImageSub(resultImage, {0, 0, 0}, mipExtent, subresource, mipSize, mipBytes.data(),
                                        resultImage.descriptor.imageLayout));
      return;
    }
    rowsPerStrip = std::min(rowsPerStrip, height);

    for(uint32_t y = 0; y < height; y += rowsPerStrip)
    {
      const uint32_t   stripH = std::min(rowsPerStrip, height - y);
      const VkOffset3D offset{0, static_cast<int32_t>(y), 0};
      const VkExtent3D extent{mipExtent.width, stripH, 1};
      const size_t     stripBytes = static_cast<size_t>(stripH) * bytesPerRow;
      NVVK_CHECK(staging.appendImageSub(resultImage, offset, extent, subresource, stripBytes,
                                        mipBytes.data() + static_cast<size_t>(y) * bytesPerRow, resultImage.descriptor.imageLayout));
    }
  };

  uploadMipLevel(0, imageCreateInfo.extent, std::span<const char>(mipData[0]));
  staging.cmdUploadAppended(cmd);  // Upload the first mip level

  // The image require to generate the mipmaps
  if(mipData.size() == 1 && (canGenerateMipmaps && generateMipmaps))
  {
    nvvk::cmdGenerateMipmaps(cmd, resultImage.image, imgSize, imageCreateInfo.mipLevels, 1, resultImage.descriptor.imageLayout);
  }
  else
  {
    for(uint32_t mip = 1; mip < (uint32_t)imageCreateInfo.mipLevels; mip++)
    {
      const VkExtent3D mipExtent{std::max(1u, imgSize.width >> mip), std::max(1u, imgSize.height >> mip), 1};

      if(mipExtent.width > 0 && mipExtent.height > 0)
      {
        uploadMipLevel(mip, mipExtent, std::span<const char>(mipData[mip]));
      }
    }
    // Upload all the mip levels
    staging.cmdUploadAppended(cmd);
  }
  // Barrier to change the layout to SHADER_READ_ONLY_OPTIMAL
  resultImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  nvvk::cmdImageMemoryBarrier(cmd, {resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, resultImage.descriptor.imageLayout});

  if(!name.empty())
  {
    nvvk::DebugUtil::getInstance().setObjectName(resultImage.image, name);
  }
  else
  {
    NVVK_DBG_NAME(resultImage.image);
  }
  out = resultImage;
  return true;
}
