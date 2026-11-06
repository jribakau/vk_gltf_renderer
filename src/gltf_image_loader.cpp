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

//
// Loads glTF image data from various container formats (KTX2, DDS, WebP,
// and standard formats via stb_image) into Vulkan-compatible image
// descriptors. Handles format detection, swizzle mapping, mip chains,
// and conversion to VkFormat-based ImageData structures.
//

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <stb/stb_image.h>
#include <turbojpeg.h>
#include <webp/decode.h>
#include "nvimageformats/nv_dds.h"
#include "nvimageformats/nv_ktx.h"
#include "nvimageformats/texture_formats.h"
#include "nvutils/logger.hpp"
#include "gltf_image_loader.hpp"

namespace nvvkgltf {

namespace {

VkComponentSwizzle ktxSwizzleToVk(nv_ktx::Swizzle swizzle)
{
  switch(swizzle)
  {
    case nv_ktx::Swizzle::ZERO:
      return VK_COMPONENT_SWIZZLE_ZERO;
    case nv_ktx::Swizzle::ONE:
      return VK_COMPONENT_SWIZZLE_ONE;
    case nv_ktx::Swizzle::R:
      return VK_COMPONENT_SWIZZLE_R;
    case nv_ktx::Swizzle::G:
      return VK_COMPONENT_SWIZZLE_G;
    case nv_ktx::Swizzle::B:
      return VK_COMPONENT_SWIZZLE_B;
    case nv_ktx::Swizzle::A:
      return VK_COMPONENT_SWIZZLE_A;
    default:
      return VK_COMPONENT_SWIZZLE_IDENTITY;
  }
}

VkComponentMapping ktxSwizzleToVkComponentMapping(const std::array<nv_ktx::Swizzle, 4>& swizzle)
{
  return {ktxSwizzleToVk(swizzle[0]), ktxSwizzleToVk(swizzle[1]), ktxSwizzleToVk(swizzle[2]), ktxSwizzleToVk(swizzle[3])};
}

bool loadDds(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  const char ddsIdentifier[4] = {'D', 'D', 'S', ' '};
  if(byteLength < sizeof(ddsIdentifier) || memcmp(data, ddsIdentifier, sizeof(ddsIdentifier)) != 0)
    return false;

  nv_dds::Image        ddsImage{};
  nv_dds::ReadSettings settings{};
  const nv_dds::ErrorWithText readResult = ddsImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, settings);
  if(readResult.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_dds: %s\n", imageIDForLog, readResult.value().c_str());
    return false;
  }

  out.size.width  = ddsImage.getWidth(0);
  out.size.height = ddsImage.getHeight(0);
  if(ddsImage.getDepth(0) > 1)
  {
    LOGW("This DDS image had a depth of %u, but loadFromMemory() cannot handle volume textures.\n", ddsImage.getDepth(0));
    return false;
  }
  if(ddsImage.getNumFaces() > 1)
  {
    LOGW("This DDS image had %u faces, but loadFromMemory() cannot handle cubemaps.\n", ddsImage.getNumFaces());
    return false;
  }
  if(ddsImage.getNumLayers() > 1)
  {
    LOGW("This DDS image had %u array elements, but loadFromMemory() cannot handle array textures.\n", ddsImage.getNumLayers());
    return false;
  }
  out.format = texture_formats::dxgiToVulkan(ddsImage.dxgiFormat);
  out.format = texture_formats::tryForceVkFormatTransferFunction(out.format, srgb);
  if(out.format == VK_FORMAT_UNDEFINED)
  {
    LOGW("Could not determine a VkFormat for DXGI format %u (%s).\n", ddsImage.dxgiFormat,
         texture_formats::getDXGIFormatName(ddsImage.dxgiFormat));
    return false;
  }

  for(uint32_t i = 0; i < ddsImage.getNumMips(); i++)
  {
    std::vector<char>& mip = ddsImage.subresource(i, 0, 0).data;
    out.mipData.push_back(std::move(mip));
  }
  return true;
}

bool loadKtx(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog, uint32_t maxExtent)
{
  const uint8_t ktxIdentifier[5] = {0xAB, 0x4B, 0x54, 0x58, 0x20};  // Common for KTX1 + KTX2
  if(byteLength < sizeof(ktxIdentifier) || memcmp(data, ktxIdentifier, sizeof(ktxIdentifier)) != 0)
    return false;

  nv_ktx::Image              ktxImage;
  const nv_ktx::ReadSettings ktxReadSettings;
  const char*                reading = reinterpret_cast<const char*>(data);
  const nv_ktx::ErrorWithText maybeError = ktxImage.readHeaderFromMemory(reading, byteLength, ktxReadSettings);
  if(maybeError.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_ktx: %s\n", imageIDForLog, maybeError->c_str());
    return false;
  }

  if(ktxImage.mip0Depth > 1)
  {
    LOGW("KTX image %" PRIu64 " had a depth of %u, but loadFromMemory() cannot handle volume textures.\n",
         imageIDForLog, ktxImage.mip0Depth);
    return false;
  }
  if(ktxImage.numFaces > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u faces, but loadFromMemory() cannot handle cubemaps.\n", imageIDForLog, ktxImage.numFaces);
    return false;
  }
  if(ktxImage.numLayersPossibly0 > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u array elements, but loadFromMemory() cannot handle array textures.\n",
         imageIDForLog, ktxImage.numLayersPossibly0);
    return false;
  }

  // only the levels kept are read, and inflated
  const uint32_t skip = rs::levelsToSkip(ktxImage.mip0Width, ktxImage.mip0Height, ktxImage.numMips, maxExtent);
  out.mipData.resize(ktxImage.numMips - skip);
  std::vector<nv_ktx::SubresourceTarget> targets(out.mipData.size());
  for(uint32_t i = 0; i < targets.size(); i++)
  {
    out.mipData[i].resize(ktxImage.getSubresourceLayout(skip + i, 0, 0).uncompressedByteSize);
    targets[i] = {.data = out.mipData[i].data(), .capacityInBytes = out.mipData[i].size()};
  }
  const nv_ktx::SubresourceRange range{.firstMip = skip, .numMips = uint32_t(targets.size())};
  const nv_ktx::ErrorWithText    readError = ktxImage.readSubresourcesFromMemory(reading, byteLength, range, targets.data());
  if(readError.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_ktx: %s\n", imageIDForLog, readError->c_str());
    out.mipData.clear();
    return false;
  }

  out.size.width       = rs::levelExtent(ktxImage.mip0Width, skip);
  out.size.height      = rs::levelExtent(ktxImage.mip0Height, skip);
  out.format           = texture_formats::tryForceVkFormatTransferFunction(ktxImage.format, srgb);
  out.componentMapping = ktxSwizzleToVkComponentMapping(ktxImage.swizzle);
  return true;
}

bool loadWebp(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  // VP8/VP8L bitstreams store each dimension in 14 bits. Only the VP8X canvas can declare more, and
  // WebPGetFeatures returns it unchecked for animated or truncated files, so a tiny file could otherwise
  // request a multi-GiB allocation.
  constexpr int kMaxWebpDimension = 1 << 14;

  const uint8_t* dataU8 = reinterpret_cast<const uint8_t*>(data);

  WebPBitstreamFeatures features{};
  if(WebPGetFeatures(dataU8, byteLength, &features) != VP8_STATUS_OK)
    return false;
  if(features.has_animation)
  {
    LOGW("WebP image %" PRIu64 " is animated, which is not supported.\n", imageIDForLog);
    return false;
  }
  const int width  = features.width;
  const int height = features.height;
  if(width <= 0 || height <= 0 || width > kMaxWebpDimension || height > kMaxWebpDimension)
  {
    LOGW("WebP image %" PRIu64 " has invalid dimensions %dx%d.\n", imageIDForLog, width, height);
    return false;
  }

  std::vector<char> decompressed(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
  if(!WebPDecodeRGBAInto(dataU8, byteLength, reinterpret_cast<uint8_t*>(decompressed.data()), decompressed.size(), width * 4))
  {
    LOGW("Failed to decode WebP image %" PRIu64 "\n", imageIDForLog);
    return false;
  }

  out.format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
  out.size   = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  out.mipData.clear();
  out.mipData.push_back(std::move(decompressed));
  return true;
}

// A JPEG at 1/downscale of its size, as libjpeg-turbo's IDCT makes it: RGBA, or one channel for a
// grey one, as loadStb gives them. False for anything else, a CMYK one included (stb reads it).
bool loadJpegScaled(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint32_t downscale, uint64_t imageIDForLog)
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  if((downscale != 2 && downscale != 4 && downscale != 8) || byteLength < 3 || bytes[0] != 0xFF || bytes[1] != 0xD8 || bytes[2] != 0xFF)
    return false;
  const tjhandle tj = tj3Init(TJINIT_DECOMPRESS);
  if(!tj)
    return false;
  struct Destroy
  {
    tjhandle tj;
    ~Destroy() { tj3Destroy(tj); }
  } destroy{tj};
  if(tj3DecompressHeader(tj, bytes, byteLength) != 0)
    return false;
  const int colorspace = tj3Get(tj, TJPARAM_COLORSPACE);
  if(colorspace == TJCS_CMYK || colorspace == TJCS_YCCK)
    return false;
  const tjscalingfactor factor{1, int(downscale)};
  if(tj3SetScalingFactor(tj, factor) != 0)
    return false;
  const bool grey     = colorspace == TJCS_GRAY;
  const int  channels = grey ? 1 : 4;
  const int  w = TJSCALED(tj3Get(tj, TJPARAM_JPEGWIDTH), factor), h = TJSCALED(tj3Get(tj, TJPARAM_JPEGHEIGHT), factor);
  if(w <= 0 || h <= 0)
    return false;
  std::vector<char> texels(size_t(w) * size_t(h) * size_t(channels));
  // a warning (a damaged stream it read past) still leaves the image
  if(tj3Decompress8(tj, bytes, byteLength, reinterpret_cast<unsigned char*>(texels.data()), w * channels, grey ? TJPF_GRAY : TJPF_RGBA) != 0
     && tj3GetErrorCode(tj) != TJERR_WARNING)
  {
    LOGW("Image %" PRIu64 ": %s\n", imageIDForLog, tj3GetErrorStr(tj));
    return false;
  }
  out.format = grey ? VK_FORMAT_R8_UNORM : srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
  if(grey)
    out.componentMapping = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
  out.size = VkExtent2D{uint32_t(w), uint32_t(h)};
  out.mipData.clear();
  out.mipData.push_back(std::move(texels));
  return true;
}

bool loadStb(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  if(byteLength > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    LOGW("File for image %" PRIu64 " was too large (%zu bytes) for stb_image to read.\n", imageIDForLog, byteLength);
    return false;
  }

  const stbi_uc* dataStb   = reinterpret_cast<const stbi_uc*>(data);
  const int      lengthStb = static_cast<int>(byteLength);

  int w = 0, h = 0, comp = 0;
  if(!stbi_info_from_memory(dataStb, lengthStb, &w, &h, &comp))
  {
    LOGW("Failed to get info using stb_image for image %" PRIu64 "\n", imageIDForLog);
    return false;
  }

  const bool is16Bit = stbi_is_16_bit_from_memory(dataStb, lengthStb);

  stbi_uc* decompressed = nullptr;
  size_t   bytesPerPixel{0};
  int      requiredComponents = comp == 1 ? 1 : 4;
  if(is16Bit)
  {
    stbi_us* decompressed16 = stbi_load_16_from_memory(dataStb, lengthStb, &w, &h, &comp, requiredComponents);
    bytesPerPixel           = sizeof(*decompressed16) * requiredComponents;
    decompressed            = reinterpret_cast<stbi_uc*>(decompressed16);
  }
  else
  {
    decompressed  = stbi_load_from_memory(dataStb, lengthStb, &w, &h, &comp, requiredComponents);
    bytesPerPixel = sizeof(*decompressed) * requiredComponents;
  }

  switch(requiredComponents)
  {
    case 1:
      out.format = is16Bit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
      out.componentMapping = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
      break;
    case 4:
      out.format = is16Bit ? VK_FORMAT_R16G16B16A16_UNORM : srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
      break;
    default:
      stbi_image_free(decompressed);
      return false;
  }

  if(decompressed && w > 0 && h > 0 && out.format != VK_FORMAT_UNDEFINED)
  {
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(w) * h * bytesPerPixel;
    out.size                = VkExtent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    out.mipData.clear();
    out.mipData.emplace_back(decompressed, decompressed + bufferSize);
  }

  stbi_image_free(decompressed);
  return out.size.width > 0 && out.size.height > 0;
}

}  // namespace

void trimLevels(LoadedImageData& image, uint32_t maxExtent)
{
  const uint32_t skip = rs::levelsToSkip(image.size.width, image.size.height, uint32_t(image.mipData.size()), maxExtent);
  if(skip == 0)
    return;
  image.mipData.erase(image.mipData.begin(), image.mipData.begin() + skip);
  image.size = {rs::levelExtent(image.size.width, skip), rs::levelExtent(image.size.height, skip)};
}

std::optional<rs::ImageChain> readImageChain(const void* data, size_t byteLength)
{
  const uint8_t ktxIdentifier[5] = {0xAB, 0x4B, 0x54, 0x58, 0x20};
  if(data == nullptr || byteLength < sizeof(ktxIdentifier) || memcmp(data, ktxIdentifier, sizeof(ktxIdentifier)) != 0)
    return std::nullopt;
  nv_ktx::Image ktxImage;
  if(ktxImage.readHeaderFromMemory(reinterpret_cast<const char*>(data), byteLength, nv_ktx::ReadSettings{}).has_value())
    return std::nullopt;
  rs::ImageChain chain{.width = ktxImage.mip0Width, .height = ktxImage.mip0Height};
  for(uint32_t i = 0; i < ktxImage.numMips; i++)
    chain.levelBytes.push_back(ktxImage.getSubresourceLayout(i, 0, 0).uncompressedByteSize);
  return chain;
}

bool loadFromMemory(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog, uint32_t maxExtent, uint32_t downscale)
{
  out = LoadedImageData{};

  if(data == nullptr || byteLength == 0)
    return false;

  if(loadDds(out, data, byteLength, srgb, imageIDForLog))
  {
    trimLevels(out, maxExtent);
    return true;
  }
  if(loadKtx(out, data, byteLength, srgb, imageIDForLog, maxExtent))
    return true;
  if(loadWebp(out, data, byteLength, srgb, imageIDForLog))
    return true;
  if(downscale > 1 && loadJpegScaled(out, data, byteLength, srgb, downscale, imageIDForLog))
    return true;
  out = LoadedImageData{};
  if(loadStb(out, data, byteLength, srgb, imageIDForLog))
    return true;

  return false;
}

float imageMeanLuminance(VkFormat format, VkExtent2D size, const std::vector<char>& texels)
{
  int  channels = 0;
  bool srgb = false, bgr = false;
  switch(format)
  {
    case VK_FORMAT_R8G8B8A8_SRGB: srgb = true; [[fallthrough]];
    case VK_FORMAT_R8G8B8A8_UNORM: channels = 4; break;
    case VK_FORMAT_B8G8R8A8_SRGB: srgb = true; [[fallthrough]];
    case VK_FORMAT_B8G8R8A8_UNORM: channels = 4; bgr = true; break;
    case VK_FORMAT_R8G8B8_SRGB: srgb = true; [[fallthrough]];
    case VK_FORMAT_R8G8B8_UNORM: channels = 3; break;
    case VK_FORMAT_R8_SRGB: srgb = true; [[fallthrough]];
    case VK_FORMAT_R8_UNORM: channels = 1; break;
    default: return -1.0f;
  }
  if(size.width == 0 || size.height == 0 || texels.size() < size_t(size.width) * size.height * channels)
    return -1.0f;
  const auto linear = [srgb](uint8_t v) {
    const float c = float(v) / 255.0f;
    return srgb ? (c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f)) : c;
  };
  const uint32_t nx = std::min(size.width, 64u), ny = std::min(size.height, 64u);
  double         sum = 0.0;
  for(uint32_t j = 0; j < ny; ++j)
    for(uint32_t i = 0; i < nx; ++i)
    {
      const size_t   x = (size_t(i) * 2 + 1) * size.width / (2 * nx), y = (size_t(j) * 2 + 1) * size.height / (2 * ny);
      const uint8_t* p = reinterpret_cast<const uint8_t*>(texels.data()) + (y * size.width + x) * channels;
      if(channels == 1)
        sum += linear(p[0]);
      else
        sum += 0.2126f * linear(p[bgr ? 2 : 0]) + 0.7152f * linear(p[1]) + 0.0722f * linear(p[bgr ? 0 : 2]);
    }
  return float(sum / (double(nx) * ny));
}

std::array<uint8_t, 4> bc7FirstTexel(const uint8_t block[16])
{
  // the mode's fields (the BC7 format's mode table): subsets, partition bits, rotation bits, index
  // selector bits, colour and alpha endpoint bits, p-bits per endpoint or per subset, index bits
  struct Mode
  {
    uint8_t subsets, partitionBits, rotationBits, selectorBits, colorBits, alphaBits, endpointPBit, sharedPBit, indexBits, index2Bits;
  };
  static constexpr Mode kModes[8] = {{3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0}, {3, 6, 0, 0, 5, 0, 0, 0, 2, 0},
                                     {2, 6, 0, 0, 7, 0, 1, 0, 2, 0}, {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
                                     {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0}};
  static constexpr uint8_t kWeights2[4]  = {0, 21, 43, 64};
  static constexpr uint8_t kWeights3[8]  = {0, 9, 18, 27, 37, 46, 55, 64};
  static constexpr uint8_t kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
  uint32_t pos  = 0;  // the bits from the block's low end
  const auto take = [&](uint32_t n) {
    uint32_t v = 0;
    for(uint32_t i = 0; i < n; i++, pos++)
      v |= uint32_t((block[pos >> 3] >> (pos & 7)) & 1u) << i;
    return v;
  };
  uint32_t mode = 0;
  while(mode < 8 && take(1) == 0)
    mode++;
  if(mode == 8)
    return {0, 0, 0, 0};  // a reserved mode decodes as transparent black
  const Mode& m = kModes[mode];
  take(m.partitionBits);  // texel 0 is in subset 0 under every partition
  const uint32_t rotation = take(m.rotationBits), selector = take(m.selectorBits);
  const uint32_t ends     = 2u * m.subsets;  // the endpoints of all subsets, channel by channel
  uint32_t       e[4][2]  = {};
  for(uint32_t c = 0; c < 3; c++)
    for(uint32_t i = 0; i < ends; i++)
      if(const uint32_t v = take(m.colorBits); i < 2)
        e[c][i] = v;
  for(uint32_t i = 0; m.alphaBits && i < ends; i++)
    if(const uint32_t v = take(m.alphaBits); i < 2)
      e[3][i] = v;
  uint32_t p[2] = {};
  if(m.endpointPBit)
    for(uint32_t i = 0; i < ends; i++)
    {
      if(const uint32_t v = take(1); i < 2)
        p[i] = v;
    }
  else if(m.sharedPBit)
    for(uint32_t s = 0; s < m.subsets; s++)
      if(const uint32_t v = take(1); s == 0)
        p[0] = p[1] = v;
  const uint32_t pBit   = m.endpointPBit | m.sharedPBit;
  const auto     expand = [](uint32_t v, uint32_t bits) {
    v <<= 8 - bits;  // the high bits repeated below
    return v | (v >> bits);
  };
  uint32_t ep[4][2];
  for(uint32_t c = 0; c < 4; c++)
    for(uint32_t i = 0; i < 2; i++)
    {
      const uint32_t bits = (c < 3 ? m.colorBits : m.alphaBits) + pBit;
      ep[c][i]            = (c == 3 && !m.alphaBits) ? 255u : expand(pBit ? (e[c][i] << 1) | p[i] : e[c][i], bits);
    }
  // texel 0 is subset 0's anchor: one index bit fewer, first in each index set
  const uint32_t index  = take(m.indexBits - 1);
  uint32_t       index2 = 0;
  if(m.index2Bits)
  {
    take(15u * m.indexBits);  // the other texels' primary indices
    index2 = take(m.index2Bits - 1);
  }
  const auto weight = [&](uint32_t idx, uint32_t bits) {
    return bits == 2 ? kWeights2[idx] : bits == 3 ? kWeights3[idx] : kWeights4[idx];
  };
  uint32_t colorIndex = index, colorBits = m.indexBits, alphaIndex = index, alphaBits = m.indexBits;
  if(m.index2Bits)
  {
    alphaIndex = index2;
    alphaBits  = m.index2Bits;
    if(selector)
    {
      std::swap(colorIndex, alphaIndex);
      std::swap(colorBits, alphaBits);
    }
  }
  std::array<uint8_t, 4> out;
  for(uint32_t c = 0; c < 4; c++)
  {
    const uint32_t w = c < 3 ? weight(colorIndex, colorBits) : weight(alphaIndex, alphaBits);
    out[c]           = uint8_t(((64u - w) * ep[c][0] + w * ep[c][1] + 32u) >> 6);
  }
  if(rotation)
    std::swap(out[3], out[rotation - 1]);
  return out;
}

uint8_t bc4FirstTexel(const uint8_t block[8])
{
  const uint32_t r0 = block[0], r1 = block[1], index = block[2] & 7u;
  if(index < 2)
    return uint8_t(index == 0 ? r0 : r1);
  if(r0 > r1)
    return uint8_t(((8u - index) * r0 + (index - 1u) * r1) / 7u);
  return index == 6 ? 0 : index == 7 ? 255 : uint8_t(((6u - index) * r0 + (index - 1u) * r1) / 5u);
}

float imageMeanLuminance(const LoadedImageData& image)
{
  if(image.mipData.empty())
    return -1.0f;
  const bool bc7 = image.format == VK_FORMAT_BC7_UNORM_BLOCK || image.format == VK_FORMAT_BC7_SRGB_BLOCK;
  if(!bc7 && image.format != VK_FORMAT_BC4_UNORM_BLOCK)
    return imageMeanLuminance(image.format, image.size, image.mipData[0]);
  // the smallest level, when it is 1 x 1: the image averaged in linear light (texcache.rs `cook`)
  const uint32_t last = uint32_t(image.mipData.size() - 1);
  const std::vector<char>& level = image.mipData.back();
  if(rs::levelExtent(image.size.width, last) != 1 || rs::levelExtent(image.size.height, last) != 1 || level.size() < (bc7 ? 16u : 8u))
    return -1.0f;
  const auto* block = reinterpret_cast<const uint8_t*>(level.data());
  float       texel[4];  // what the image's own channels hold, 0-1
  if(bc7)
  {
    const std::array<uint8_t, 4> t = bc7FirstTexel(block);
    for(int c = 0; c < 4; c++)
      texel[c] = t[c] / 255.0f;
  }
  else
  {
    texel[0] = bc4FirstTexel(block) / 255.0f;
    texel[1] = texel[2] = 0.0f;
    texel[3] = 1.0f;
  }
  // the channels a sampler reads through the image's swizzle
  const auto read = [&](VkComponentSwizzle s, int own) {
    switch(s)
    {
      case VK_COMPONENT_SWIZZLE_ZERO: return 0.0f;
      case VK_COMPONENT_SWIZZLE_ONE: return 1.0f;
      case VK_COMPONENT_SWIZZLE_R: return texel[0];
      case VK_COMPONENT_SWIZZLE_G: return texel[1];
      case VK_COMPONENT_SWIZZLE_B: return texel[2];
      case VK_COMPONENT_SWIZZLE_A: return texel[3];
      default: return texel[own];
    }
  };
  const VkComponentMapping& m    = image.componentMapping;
  const bool                srgb = image.format == VK_FORMAT_BC7_SRGB_BLOCK;
  const auto                lin  = [srgb](float c) { return srgb ? (c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f)) : c; };
  return 0.2126f * lin(read(m.r, 0)) + 0.7152f * lin(read(m.g, 1)) + 0.0722f * lin(read(m.b, 2));
}

uint32_t formatColorComponentCount(VkFormat format)
{
  switch(format)
  {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
    case VK_FORMAT_EAC_R11_UNORM_BLOCK:
    case VK_FORMAT_EAC_R11_SNORM_BLOCK:
      return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
    case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
      return 2;
    default:
      return 4;
  }
}

}  // namespace nvvkgltf
