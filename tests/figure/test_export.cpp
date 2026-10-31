// The GLB export composes each texture a material asks for as the texture cooker does: green
// inverted for a normal map, the cutout folded into alpha, rows bottom-up.

#include <gtest/gtest.h>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <filesystem>
#include <map>

#include <figure/gltf/export_glb.hpp>

namespace {

namespace fs = std::filesystem;

struct Dir
{
  fs::path path = fs::temp_directory_path() / "fv_test_export";
  Dir() { fs::create_directories(path); }
  ~Dir() { std::error_code ec; fs::remove_all(path, ec); }
  // a 2 x 2 image, each pixel an RGBA
  std::string write(const std::string& name, const std::vector<uint8_t>& rgba) const
  {
    const std::string file = (path / name).generic_string();
    stbi_write_png(file.c_str(), 2, 2, 4, rgba.data(), 8);
    return name;
  }
};

std::vector<uint8_t> decoded(const std::vector<uint8_t>& png, int* channels = nullptr)
{
  int w, h, n;
  unsigned char* d = stbi_load_from_memory(png.data(), int(png.size()), &w, &h, &n, 4);
  EXPECT_NE(d, nullptr);
  EXPECT_EQ(w, 2);
  std::vector<uint8_t> out(d, d + 16);
  stbi_image_free(d);
  if(channels)
    *channels = n;
  return out;
}

TEST(Export, ANormalMapGetsItsGreenInvertedAndItsRowsFlipped)
{
  Dir  dir;
  // top row: (10, 20, 30), (40, 50, 60); bottom row: (70, 80, 90), (100, 110, 120)
  const auto rel = dir.write("n.png", {10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255});
  fv::build::ExportOptions options;
  options.find = [&](const std::string& r) -> std::optional<fs::path> { return dir.path / r; };
  fv::mat::TextureRequest req;
  req.rel      = rel;
  req.srgb     = false;
  req.channels = "ng";
  const auto png = fv::build::composeImage(req, options);
  ASSERT_TRUE(png);
  const auto px = decoded(*png);
  // rows swapped: the first pixel is the old bottom-left, green inverted
  EXPECT_EQ(px[0], 70);
  EXPECT_EQ(px[1], 255 - 80);
  EXPECT_EQ(px[2], 90);
  EXPECT_EQ(px[8], 10);
  EXPECT_EQ(px[9], 255 - 20);
}

// The same numbers as the cooker's lays_layers_by_opacity_blend_and_mask (texcache.rs)
TEST(Export, ALayerLiesOverByItsOpacityBlendAndMask)
{
  Dir        dir;
  const auto pixels = [](uint8_t r, uint8_t g, uint8_t b) { return std::vector<uint8_t>{r, g, b, 255, r, g, b, 255, r, g, b, 255, r, g, b, 255}; };
  const auto base   = dir.write("b.png", pixels(200, 100, 40));
  const auto top    = dir.write("t.png", pixels(100, 200, 50));
  // white on the left, black on the right
  const auto mask = dir.write("m.png", {255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255});
  fv::build::ExportOptions options;
  options.find = [&](const std::string& r) -> std::optional<fs::path> { return dir.path / r; };
  const auto lay = [&](const fv::mat::MapLayer& layer) {
    fv::mat::TextureRequest req;
    req.rel    = base;
    req.layers = {fv::mat::layerText(layer)};
    const auto png = fv::build::composeImage(req, options);
    EXPECT_TRUE(png);
    return decoded(*png);
  };
  const auto near = [](const std::vector<uint8_t>& px, int i, std::array<int, 3> want) {
    for(int c = 0; c < 3; c++)
      EXPECT_NEAR(int(px[size_t(i) * 4 + size_t(c)]), want[size_t(c)], 1) << "pixel " << i << " channel " << c;
  };

  fv::mat::MapLayer half;
  half.file    = top;
  half.opacity = 0.5f;
  near(lay(half), 0, {150, 150, 45});

  fv::mat::MapLayer multiply;
  multiply.file  = top;
  multiply.blend = "multiply";
  near(lay(multiply), 0, {78, 78, 8});

  fv::mat::MapLayer masked;
  masked.file = top;
  masked.mask = mask;
  const auto maskedPx = lay(masked);
  near(maskedPx, 0, {100, 200, 50});
  near(maskedPx, 1, {200, 100, 40});

  // a lipstick: no file, its colour where the mask says, by overlay
  fv::mat::MapLayer colour;
  colour.mask  = mask;
  colour.color = {0.5f, 0.25f, 1.f};
  colour.blend = "overlay";
  const auto colourPx = lay(colour);
  near(colourPx, 0, {200, 50, 80});
  near(colourPx, 1, {200, 100, 40});
}

TEST(Export, ACutoutMapBecomesTheAlphaOfTheBaseColour)
{
  Dir  dir;
  const auto base  = dir.write("b.png", {200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255});
  const auto alpha = dir.write("a.png", {0, 0, 0, 255, 0, 128, 0, 255, 0, 255, 0, 255, 0, 64, 0, 255});
  fv::build::ExportOptions options;
  options.find = [&](const std::string& r) -> std::optional<fs::path> { return dir.path / r; };
  fv::mat::TextureRequest req;
  req.rel      = base;
  req.alphaRel = alpha;
  const auto png = fv::build::composeImage(req, options);
  ASSERT_TRUE(png);
  const auto px = decoded(*png);
  // rows flipped: the bottom row of the cutout comes first; alpha is its green channel
  EXPECT_EQ(px[3], 255);
  EXPECT_EQ(px[7], 64);
  EXPECT_EQ(px[11], 0);
  EXPECT_EQ(px[15], 128);
  EXPECT_EQ(px[0], 200);
}

TEST(Export, ACutoutReadFromItsAlphaTakesTheMapsAlpha)
{
  // DAZ's grayscale_mode "alpha": a logo white throughout, cut by its own alpha (the cooker's twin)
  Dir  dir;
  const auto base  = dir.write("b.png", {200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255});
  const auto alpha = dir.write("a.png", {255, 255, 255, 0, 255, 255, 255, 128, 255, 255, 255, 255, 255, 255, 255, 64});
  fv::build::ExportOptions options;
  options.find = [&](const std::string& r) -> std::optional<fs::path> { return dir.path / r; };
  fv::mat::TextureRequest req;
  req.rel            = base;
  req.alphaRel       = alpha;
  req.alphaFromAlpha = true;
  const auto png = fv::build::composeImage(req, options);
  ASSERT_TRUE(png);
  const auto px = decoded(*png);
  // rows flipped: the bottom row first
  EXPECT_EQ(px[3], 255);
  EXPECT_EQ(px[7], 64);
  EXPECT_EQ(px[11], 0);
  EXPECT_EQ(px[15], 128);
  EXPECT_EQ(px[0], 200);
}

TEST(Export, AMissingSourceSaysSo)
{
  fv::build::ExportOptions options;
  options.find = [](const std::string&) -> std::optional<fs::path> { return std::nullopt; };
  fv::mat::TextureRequest req;
  req.rel = "nowhere.png";
  std::string why;
  EXPECT_FALSE(fv::build::composeImage(req, options, &why));
  EXPECT_NE(why.find("nowhere.png"), std::string::npos);
}

TEST(Export, AMetalMapFillsTheBlueOfMetallicRoughness)
{
  // the cooker's twin: a metal mask's grey (the mean of R, G, B) in blue, the roughness kept in green
  Dir        dir;
  const auto rough = dir.write("r.png", {1, 60, 7, 255, 2, 120, 7, 255, 3, 180, 7, 255, 4, 240, 7, 255});
  const auto metal = dir.write("m.png", {0, 0, 0, 255, 255, 255, 255, 255, 30, 60, 90, 255, 100, 100, 100, 255});
  fv::build::ExportOptions options;
  options.find = [&](const std::string& r) -> std::optional<fs::path> { return dir.path / r; };
  fv::mat::TextureRequest req;
  req.rel      = rough;
  req.srgb     = false;
  req.channels = "mr";
  req.metalRel = metal;
  const auto png = fv::build::composeImage(req, options);
  ASSERT_TRUE(png);
  const auto px = decoded(*png);
  // rows flipped: the bottom row first
  EXPECT_EQ(px[0], 3);
  EXPECT_EQ(px[1], 180);
  EXPECT_EQ(px[2], 60);
  EXPECT_EQ(px[3], 255);
  EXPECT_EQ(px[5], 240);
  EXPECT_EQ(px[6], 100);
  EXPECT_EQ(px[9], 60);
  EXPECT_EQ(px[10], 0);
  EXPECT_EQ(px[13], 120);
  EXPECT_EQ(px[14], 255);

  // without a metal map, full metalness: the factor alone
  req.metalRel.clear();
  const auto plain = decoded(*fv::build::composeImage(req, options));
  EXPECT_EQ(plain[1], 180);
  EXPECT_EQ(plain[2], 255);

  // the mask alone: its grey in blue, roughness white for the factor to set
  fv::mat::TextureRequest alone;
  alone.rel      = metal;
  alone.srgb     = false;
  alone.channels = "mt";
  const auto mt  = decoded(*fv::build::composeImage(alone, options));
  EXPECT_EQ(mt[0], 255);
  EXPECT_EQ(mt[1], 255);
  EXPECT_EQ(mt[2], 60);
  EXPECT_EQ(mt[10], 0);
  EXPECT_EQ(mt[14], 255);
}

TEST(Export, AWeightMapIsItsGreyInEveryChannelTimesTheMapsMultipliedIn)
{
  // the cooker's twin ("wa", mul=): specularTexture's alpha and clearcoatTexture's red read the same
  // weight, the mean of R, G and B (DAZ's grayscale_mode "average"); Skin_SLW is a flat 153
  Dir        dir;
  const auto slw  = dir.write("slw.png", {153, 153, 153, 255, 200, 100, 0, 255, 153, 153, 153, 255, 153, 153, 153, 255});
  const auto half = dir.write("half.png", {128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255});
  fv::build::ExportOptions options;
  options.find = [&](const std::string& r) -> std::optional<fs::path> { return dir.path / r; };
  fv::mat::TextureRequest req;
  req.rel      = slw;
  req.srgb     = false;
  req.channels = "wa";
  const auto px = decoded(*fv::build::composeImage(req, options));
  // rows flipped: the coloured texel (top right) is last
  for(int c = 0; c < 4; c++)
  {
    EXPECT_EQ(px[c], 153);
    EXPECT_EQ(px[12 + c], 100);
  }
  // times a map of 128: (153 x 128 + 127) / 255 = 77
  req.mulRels = {half};
  const auto product = decoded(*fv::build::composeImage(req, options));
  for(int c = 0; c < 4; c++)
    EXPECT_EQ(product[c], 77);
  req.mulRels = {"nowhere.png"};
  EXPECT_FALSE(fv::build::composeImage(req, options));
}

}  // namespace
