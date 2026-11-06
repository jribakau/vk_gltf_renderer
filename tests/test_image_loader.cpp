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

// CPU-only unit test for nvvkgltf::loadFromMemory's WebP and KTX2 paths: the decoder shared by glTF
// loading and the material inspector's image import.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>
#include <turbojpeg.h>
#include <webp/encode.h>

#include <cstdlib>
#include <map>
#include <sstream>
#include <string>

#include "gltf_image_loader.hpp"
#include "image_levels.hpp"
#include "image_prefetch.hpp"
#include "nvimageformats/nv_ktx.h"

namespace {

constexpr int kWidth  = 3;
constexpr int kHeight = 2;

// Distinct RGBA texel values so a row/channel swap would be caught.
std::vector<uint8_t> makePixels()
{
  std::vector<uint8_t> rgba(kWidth * kHeight * 4);
  for(size_t i = 0; i < rgba.size(); ++i)
    rgba[i] = static_cast<uint8_t>(i * 11 + 7);
  return rgba;
}

// Lossless, so the decoded bytes must match the source exactly.
std::vector<uint8_t> encodeLossless(const std::vector<uint8_t>& rgba)
{
  uint8_t*             out  = nullptr;
  const size_t         size = WebPEncodeLosslessRGBA(rgba.data(), kWidth, kHeight, kWidth * 4, &out);
  std::vector<uint8_t> file(out, out + size);
  WebPFree(out);
  return file;
}

// A RIFF/WEBP file holding only a VP8X chunk that declares a 65536 x 65535 canvas. libwebp reports
// that canvas without a bitstream to check it against, so it must be rejected before allocating.
std::vector<uint8_t> makeHugeCanvasHeader(bool animated)
{
  std::vector<uint8_t> file = {'R', 'I', 'F', 'F', 22, 0, 0, 0, 'W', 'E', 'B', 'P', 'V', 'P', '8', 'X', 10, 0, 0, 0};
  const uint8_t vp8x[10] = {static_cast<uint8_t>(animated ? 0x02 : 0x00), 0, 0, 0, 0xFF, 0xFF, 0x00, 0xFE, 0xFF, 0x00};
  file.insert(file.end(), vp8x, vp8x + sizeof(vp8x));
  return file;
}

}  // namespace

TEST(ImageLoader, WebpDecodesLosslessPixels)
{
  const std::vector<uint8_t> pixels = makePixels();
  const std::vector<uint8_t> file   = encodeLossless(pixels);
  ASSERT_FALSE(file.empty());

  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
  EXPECT_EQ(out.size.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(out.size.height, static_cast<uint32_t>(kHeight));
  ASSERT_EQ(out.mipData.size(), 1u);
  ASSERT_EQ(out.mipData[0].size(), pixels.size());
  EXPECT_EQ(std::memcmp(out.mipData[0].data(), pixels.data(), pixels.size()), 0);
}

TEST(ImageLoader, WebpFormatFollowsSrgbFlag)
{
  const std::vector<uint8_t> file = encodeLossless(makePixels());

  nvvkgltf::LoadedImageData linear;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(linear, file.data(), file.size(), /*srgb*/ false));
  EXPECT_EQ(linear.format, VK_FORMAT_R8G8B8A8_UNORM);

  nvvkgltf::LoadedImageData srgb;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(srgb, file.data(), file.size(), /*srgb*/ true));
  EXPECT_EQ(srgb.format, VK_FORMAT_R8G8B8A8_SRGB);
}

TEST(ImageLoader, WebpRejectsUnbackedHugeCanvas)
{
  for(bool animated : {true, false})
  {
    const std::vector<uint8_t> file = makeHugeCanvasHeader(animated);
    nvvkgltf::LoadedImageData  out;
    EXPECT_FALSE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false)) << "animated=" << animated;
    EXPECT_TRUE(out.mipData.empty());
  }
}

// The texture cooker (daz-viewer src-tauri/src/texcache.rs) writes each level zstd-supercompressed:
// a 32x16 sRGB map it cooked reads back as BC7 with every level its full size.
// Max texture size: the levels above the limit are left out, and what is kept is what the whole read
// has at those levels, byte for byte. A 32x16 map's 16x8 level is whole blocks, its 8x4 one too, and
// its 4x2 one isn't.
TEST(ImageLoader, KtxLeavesOutTheLevelsOverItsLimit)
{
  std::ifstream           in(std::filesystem::path(FV_TEST_DATA) / "texcache" / "cooked_zstd.ktx2", std::ios::binary);
  const std::vector<char> file{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  ASSERT_FALSE(file.empty());
  nvvkgltf::LoadedImageData whole;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(whole, file.data(), file.size(), /*srgb*/ true));

  const auto read = [&](uint32_t maxExtent) {
    nvvkgltf::LoadedImageData out;
    EXPECT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ true, 0, maxExtent));
    return out;
  };
  const nvvkgltf::LoadedImageData half = read(16);
  EXPECT_EQ(half.size.width, 16u);
  EXPECT_EQ(half.size.height, 8u);
  EXPECT_EQ(half.format, VK_FORMAT_BC7_SRGB_BLOCK);
  ASSERT_EQ(half.mipData.size(), 5u);
  for(uint32_t i = 0; i < half.mipData.size(); i++)
    EXPECT_EQ(half.mipData[i], whole.mipData[i + 1]) << "level " << i;

  EXPECT_EQ(read(20).size.width, 16u);  // the largest that fits
  EXPECT_EQ(read(32).size.width, 32u);
  EXPECT_EQ(read(0).size.width, 32u);
  EXPECT_EQ(read(1).size.width, 8u);  // as near as whole blocks go

  // an image already decoded comes to the same
  nvvkgltf::LoadedImageData trimmed = whole;
  nvvkgltf::trimLevels(trimmed, 16);
  EXPECT_EQ(trimmed.size.width, half.size.width);
  EXPECT_EQ(trimmed.size.height, half.size.height);
  EXPECT_EQ(trimmed.mipData, half.mipData);

  // and its chain reads from the header alone
  const std::optional<rs::ImageChain> chain = nvvkgltf::readImageChain(file.data(), file.size());
  ASSERT_TRUE(chain);
  EXPECT_EQ(chain->width, 32u);
  EXPECT_EQ(chain->height, 16u);
  ASSERT_EQ(chain->levelBytes.size(), whole.mipData.size());
  for(size_t i = 0; i < chain->levelBytes.size(); i++)
    EXPECT_EQ(chain->levelBytes[i], whole.mipData[i].size()) << "level " << i;
  const char png[] = "\x89PNG";
  EXPECT_FALSE(nvvkgltf::readImageChain(png, sizeof(png)));
}

TEST(ImageLoader, KtxReadsTheCookersZstdLevels)
{
  std::ifstream     in(std::filesystem::path(FV_TEST_DATA) / "texcache" / "cooked_zstd.ktx2", std::ios::binary);
  const std::vector<char> file{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  ASSERT_FALSE(file.empty());

  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ true));
  EXPECT_EQ(out.format, VK_FORMAT_BC7_SRGB_BLOCK);
  EXPECT_EQ(out.size.width, 32u);
  EXPECT_EQ(out.size.height, 16u);
  ASSERT_EQ(out.mipData.size(), 6u);
  for(uint32_t i = 0; i < out.mipData.size(); i++)
  {
    const uint32_t w = std::max(32u >> i, 1u), h = std::max(16u >> i, 1u);
    EXPECT_EQ(out.mipData[i].size(), size_t((w + 3) / 4) * ((h + 3) / 4) * 16) << "level " << i;
  }
}

// A map of one grey as the cooker writes it (texcache.rs GreyMap): BC4, and a swizzle saying where the grey goes
TEST(ImageLoader, KtxKeepsAGreyMapsSwizzle)
{
  const auto word = [](std::vector<char>& out, uint32_t v) {
    for(int i = 0; i < 4; i++)
      out.push_back(char(v >> (8 * i)));
  };
  std::vector<char> kvd;
  const std::string key = "KTXswizzle", value = "rr11";
  word(kvd, uint32_t(key.size() + 1 + value.size() + 1));
  kvd.insert(kvd.end(), key.begin(), key.end());
  kvd.push_back(0);
  kvd.insert(kvd.end(), value.begin(), value.end());
  kvd.push_back(0);
  while(kvd.size() % 4)
    kvd.push_back(0);
  std::vector<char> dfd;
  for(const uint32_t v : {uint32_t(44), uint32_t(0), uint32_t(2 | (40 << 16)), uint32_t(131 | (1 << 8) | (1 << 16)), uint32_t(3 | (3 << 8)), uint32_t(8), uint32_t(0), uint32_t(63 << 16), uint32_t(0), uint32_t(0), UINT32_MAX})
    word(dfd, v);

  const std::vector<char> blocks(4 * 8, char(0x55));  // two by two blocks: 8x8
  const uint8_t           id[12] = {0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<char>       file(id, id + 12);
  for(const uint32_t v : {uint32_t(VK_FORMAT_BC4_UNORM_BLOCK), uint32_t(1), uint32_t(8), uint32_t(8), uint32_t(0), uint32_t(0), uint32_t(1), uint32_t(1), uint32_t(0)})
    word(file, v);
  const uint32_t dfdAt = 80 + 24, kvdAt = dfdAt + uint32_t(dfd.size()), dataAt = (kvdAt + uint32_t(kvd.size()) + 15) / 16 * 16;
  for(const uint32_t v : {dfdAt, uint32_t(dfd.size()), kvdAt, uint32_t(kvd.size())})
    word(file, v);
  for(int i = 0; i < 4; i++)  // no supercompression data
    word(file, 0);
  for(const uint64_t v : {uint64_t(dataAt), uint64_t(blocks.size()), uint64_t(blocks.size())})
  {
    word(file, uint32_t(v));
    word(file, uint32_t(v >> 32));
  }
  file.insert(file.end(), dfd.begin(), dfd.end());
  file.insert(file.end(), kvd.begin(), kvd.end());
  file.resize(dataAt, 0);
  file.insert(file.end(), blocks.begin(), blocks.end());

  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
  EXPECT_EQ(out.format, VK_FORMAT_BC4_UNORM_BLOCK);
  EXPECT_EQ(out.size.width, 8u);
  ASSERT_EQ(out.mipData.size(), 1u);
  EXPECT_EQ(out.mipData[0].size(), blocks.size());
  EXPECT_EQ(out.componentMapping.r, VK_COMPONENT_SWIZZLE_R);
  EXPECT_EQ(out.componentMapping.g, VK_COMPONENT_SWIZZLE_R);
  EXPECT_EQ(out.componentMapping.b, VK_COMPONENT_SWIZZLE_ONE);
  EXPECT_EQ(out.componentMapping.a, VK_COMPONENT_SWIZZLE_ONE);
}

namespace {

// A block-compressed KTX2 map of `size` square with every level, written as the cooker writes them
// (texcache.rs: zstd per level), each block `blockOf(level, index)`
template <class BlockOf>
std::vector<char> makeKtx(VkFormat format, uint32_t size, uint32_t blockBytes, BlockOf blockOf, bool zstd = true)
{
  nv_ktx::Image ktx;
  uint32_t      levels = 1;
  while((size >> (levels - 1)) > 1)
    levels++;
  EXPECT_FALSE(ktx.allocate(levels, 0, 1).has_value());
  ktx.mip0Width  = size;
  ktx.mip0Height = size;
  ktx.format     = format;
  for(uint32_t level = 0; level < levels; level++)
  {
    const uint32_t     across = (std::max(size >> level, 1u) + 3) / 4;
    std::vector<char>& data   = ktx.subresource(level);
    for(uint32_t i = 0; i < across * across; i++)
    {
      const std::vector<char> block = blockOf(level, i);
      EXPECT_EQ(block.size(), blockBytes);
      data.insert(data.end(), block.begin(), block.end());
    }
  }
  nv_ktx::WriteSettings settings;
  settings.supercompression = zstd ? nv_ktx::SupercompressionScheme::eZstd : nv_ktx::SupercompressionScheme::eNone;
  std::ostringstream out;
  EXPECT_FALSE(ktx.writeKTX2Stream(out, settings).has_value());
  const std::string s = out.str();
  return std::vector<char>(s.begin(), s.end());
}

std::vector<char> filled(uint32_t bytes, char value)
{
  return std::vector<char>(bytes, value);
}

}  // namespace

// A map of one value everywhere keeps the one block it is made of, as a 4x4 image of one level: nothing
// visible lost, 1.4 MB less for this 1024 square. The planner's chain says the same, so it has nothing to
// lower or raise, and the mean a light weighs it by is still read.
TEST(ImageLoader, AConstantMapKeepsOneBlock)
{
  const std::vector<char> block = {char(0x20), 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, char(0x88), char(0x99), 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  const std::vector<char> file  = makeKtx(VK_FORMAT_BC7_SRGB_BLOCK, 1024, 16, [&](uint32_t, uint32_t) { return block; });
  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ true));
  EXPECT_TRUE(out.constant);
  EXPECT_EQ(out.format, VK_FORMAT_BC7_SRGB_BLOCK);
  EXPECT_EQ(out.size.width, 4u);
  EXPECT_EQ(out.size.height, 4u);
  ASSERT_EQ(out.mipData.size(), 1u);
  EXPECT_EQ(out.mipData[0], block);

  const std::optional<rs::ImageChain> chain = nvvkgltf::readImageChain(file.data(), file.size());
  ASSERT_TRUE(chain);
  EXPECT_EQ(chain->width, 4u);
  ASSERT_EQ(chain->levelBytes.size(), 1u);
  EXPECT_EQ(chain->levelBytes[0], 16u);
  rs::ImageFit fit{.chain = *chain};
  EXPECT_FALSE(fit.canLower(1));
  EXPECT_FALSE(fit.canRaise(0));
  std::vector<rs::ImageFit> fits{fit};
  EXPECT_EQ(rs::lowerToFit(fits, 1 << 20, 1), 0u);

  // the mean: what the map's own 1 x 1 level held, the same block
  nvvkgltf::LoadedImageData oneTexel{.format = VK_FORMAT_BC7_SRGB_BLOCK, .size = {1, 1}, .mipData = {block}, .componentMapping = out.componentMapping};
  EXPECT_GE(nvvkgltf::imageMeanLuminance(out), 0.0f);
  EXPECT_EQ(nvvkgltf::imageMeanLuminance(out), nvvkgltf::imageMeanLuminance(oneTexel));

  // a flat BC5 normal map too
  const std::vector<char> flat = makeKtx(VK_FORMAT_BC5_UNORM_BLOCK, 512, 16, [](uint32_t, uint32_t) { return filled(16, char(0x7f)); });
  nvvkgltf::LoadedImageData normal;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(normal, flat.data(), flat.size(), false));
  EXPECT_TRUE(normal.constant);
  EXPECT_EQ(normal.size.width, 4u);
}

// What only looks like one value stays whole: a pattern repeating every 4 texels (one block repeated,
// another by level 2, where it has averaged out), a map with one block of its own, one too small to be
// worth it, and one written without supercompression (whose level index can't say it is one value)
TEST(ImageLoader, WhatOnlyLooksConstantStaysWhole)
{
  const auto whole = [](const std::vector<char>& file, uint32_t size) {
    nvvkgltf::LoadedImageData out;
    EXPECT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), false));
    EXPECT_FALSE(out.constant);
    EXPECT_EQ(out.size.width, size);
    const std::optional<rs::ImageChain> chain = nvvkgltf::readImageChain(file.data(), file.size());
    EXPECT_TRUE(chain && chain->width == size && chain->levelBytes.size() == out.mipData.size());
  };
  // the checker's block at the top, its average below
  whole(makeKtx(VK_FORMAT_BC7_UNORM_BLOCK, 1024, 16, [](uint32_t level, uint32_t) { return filled(16, char(level == 0 ? 0x5a : 0x33)); }), 1024);
  whole(makeKtx(VK_FORMAT_BC7_UNORM_BLOCK, 1024, 16, [](uint32_t level, uint32_t i) { return filled(16, char(level == 0 && i == 777 ? 0x01 : 0x33)); }), 1024);
  whole(makeKtx(VK_FORMAT_BC4_UNORM_BLOCK, 32, 8, [](uint32_t, uint32_t) { return filled(8, char(0x40)); }), 32);
  whole(makeKtx(VK_FORMAT_BC7_UNORM_BLOCK, 1024, 16, [](uint32_t, uint32_t) { return filled(16, char(0x33)); }, /*zstd*/ false), 1024);
}

// The cooked texture cache in FV_TEXCACHE_SCAN, read and counted: how many maps are one value, and
// what keeping them as one block saves. Opt in (it reads the user's cache), with
// --gtest_also_run_disabled_tests --gtest_filter=ImageLoader.DISABLED_CountConstantMapsInACache
TEST(ImageLoader, DISABLED_CountConstantMapsInACache)
{
  const char* dir = std::getenv("FV_TEXCACHE_SCAN");
  if(!dir)
    GTEST_SKIP() << "set FV_TEXCACHE_SCAN to a cache folder";
  int      files = 0, constant = 0;
  uint64_t saved = 0;
  std::map<uint32_t, int> bySize;
  for(const auto& e : std::filesystem::directory_iterator(dir))
  {
    if(e.path().extension() != ".ktx2")
      continue;
    std::ifstream           in(e.path(), std::ios::binary);
    const std::vector<char> file{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    files++;
    const std::optional<rs::ImageChain> chain = nvvkgltf::readImageChain(file.data(), file.size());
    if(!chain || chain->width != 4 || chain->levelBytes.size() != 1)
      continue;
    nv_ktx::Image header;
    ASSERT_FALSE(header.readHeaderFromMemory(file.data(), file.size(), nv_ktx::ReadSettings{}).has_value());
    nvvkgltf::LoadedImageData out;
    ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), false));
    EXPECT_TRUE(out.constant) << e.path().string();
    constant++;
    bySize[header.mip0Width]++;
    for(uint32_t i = 0; i < header.numMips; i++)
      saved += header.getSubresourceLayout(i, 0, 0).uncompressedByteSize;
    saved -= 16;
  }
  std::string sizes;
  for(const auto& [size, n] : bySize)
    sizes += " " + std::to_string(n) + " at " + std::to_string(size);
  printf("%d of %d cooked maps are one value (%s), %.0f MiB less when resident\n", constant, files, sizes.c_str(), double(saved) / (1 << 20));
}

namespace {

// A w x h JPEG of a smooth ramp: x in red, y in blue, green mid; or grey
std::vector<uint8_t> encodeJpeg(int w, int h, bool grey)
{
  const int            channels = grey ? 1 : 3;
  std::vector<uint8_t> px(size_t(w) * h * channels);
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
      for(int c = 0; c < channels; c++)
        px[(size_t(y) * w + x) * channels + c] = uint8_t(c == 1 ? 128 : c == 0 ? x * 255 / (w - 1) : y * 255 / (h - 1));
  tjhandle tj = tj3Init(TJINIT_COMPRESS);
  tj3Set(tj, TJPARAM_QUALITY, 95);
  tj3Set(tj, TJPARAM_SUBSAMP, grey ? TJSAMP_GRAY : TJSAMP_444);
  unsigned char* jpeg = nullptr;
  size_t         size = 0;
  EXPECT_EQ(tj3Compress8(tj, px.data(), w, 0, h, grey ? TJPF_GRAY : TJPF_RGB, &jpeg, &size), 0) << tj3GetErrorStr(tj);
  std::vector<uint8_t> file(jpeg, jpeg + size);
  tj3Free(jpeg);
  tj3Destroy(tj);
  return file;
}

}  // namespace

// A stand-in JPEG decodes at a fraction of its size in the IDCT itself, as RGBA in the colour space asked
// for, each texel the ramp where its four were; whole when not asked
TEST(ImageLoader, JpegDecodesAtAFractionOfItsSize)
{
  const std::vector<uint8_t> jpeg = encodeJpeg(64, 48, false);
  nvvkgltf::LoadedImageData  out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, jpeg.data(), jpeg.size(), true, 0, 0, 2));
  EXPECT_EQ(out.size.width, 32u);
  EXPECT_EQ(out.size.height, 24u);
  EXPECT_EQ(out.format, VK_FORMAT_R8G8B8A8_SRGB);
  ASSERT_EQ(out.mipData.size(), 1u);
  ASSERT_EQ(out.mipData[0].size(), size_t(32) * 24 * 4);
  const auto texel = [&](int x, int y, int c) { return int(uint8_t(out.mipData[0][(size_t(y) * 32 + x) * 4 + c])); };
  for(const auto& [x, y] : {std::pair{0, 0}, std::pair{16, 12}, std::pair{31, 23}})
  {
    EXPECT_NEAR(texel(x, y, 0), (2 * x + 0.5) * 255 / 63, 6) << x << "," << y;
    EXPECT_NEAR(texel(x, y, 1), 128, 6);
    EXPECT_NEAR(texel(x, y, 2), (2 * y + 0.5) * 255 / 47, 6) << x << "," << y;
    EXPECT_EQ(texel(x, y, 3), 255);
  }
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, jpeg.data(), jpeg.size(), false, 0, 0, 4));
  EXPECT_EQ(out.size.width, 16u);
  EXPECT_EQ(out.format, VK_FORMAT_R8G8B8A8_UNORM);
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, jpeg.data(), jpeg.size(), true));
  EXPECT_EQ(out.size.width, 64u);
  EXPECT_EQ(out.size.height, 48u);
}

// A grey JPEG comes as one channel read as grey, as stb gives it; a format that can't decode smaller
// (WebP, for now) comes whole
TEST(ImageLoader, AGreyJpegStaysOneChannelAndOtherFormatsComeWhole)
{
  const std::vector<uint8_t> jpeg = encodeJpeg(64, 48, true);
  nvvkgltf::LoadedImageData  out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, jpeg.data(), jpeg.size(), false, 0, 0, 2));
  EXPECT_EQ(out.size.width, 32u);
  EXPECT_EQ(out.format, VK_FORMAT_R8_UNORM);
  ASSERT_EQ(out.mipData[0].size(), size_t(32) * 24);
  EXPECT_EQ(out.componentMapping.g, VK_COMPONENT_SWIZZLE_R);
  EXPECT_EQ(out.componentMapping.a, VK_COMPONENT_SWIZZLE_ONE);

  const std::vector<uint8_t> webp = encodeLossless(makePixels());
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, webp.data(), webp.size(), false, 0, 0, 2));
  EXPECT_EQ(out.size.width, uint32_t(kWidth));
  EXPECT_EQ(out.size.height, uint32_t(kHeight));
}

// Decoding ahead is only ahead of time: past its budget nothing more is decoded, and what nobody takes goes
TEST(ImagePrefetch, KeepsToItsBudgetAndWhatNobodyTakesGoes)
{
  const std::filesystem::path dir = std::filesystem::path(FV_TEST_DATA) / "daz-library" / "Runtime" / "Textures" / "Test";
  const std::filesystem::path a = dir / "BoxDetail_NM.png", b = dir / "BoxMakeup.png";
  nvvkgltf::ImagePrefetch     prefetch;
  prefetch.prefetch(a, false);
  const uint64_t one = prefetch.bytes();
  ASSERT_GT(one, 0u);
  prefetch.prefetch(a, false);  // the same file again
  EXPECT_EQ(prefetch.bytes(), one);

  prefetch.setBudget(one);  // full: not decoded, and the upload decodes it
  prefetch.prefetch(b, false);
  EXPECT_EQ(prefetch.bytes(), one);
  EXPECT_FALSE(prefetch.take(b, false));

  EXPECT_TRUE(prefetch.take(a, false));
  EXPECT_EQ(prefetch.bytes(), 0u);
  EXPECT_FALSE(prefetch.take(a, false));

  prefetch.setBudget(nvvkgltf::ImagePrefetch::kDefaultBudget);
  prefetch.prefetch(b, false);
  EXPECT_GT(prefetch.bytes(), 0u);
  prefetch.dropOlderThan(60);
  EXPECT_GT(prefetch.bytes(), 0u);  // not yet
  prefetch.dropOlderThan(0);
  EXPECT_EQ(prefetch.bytes(), 0u);
  EXPECT_FALSE(prefetch.take(b, false));

  prefetch.setBudget(0);
  prefetch.prefetch(a, false);
  EXPECT_EQ(prefetch.bytes(), 0u);
}
