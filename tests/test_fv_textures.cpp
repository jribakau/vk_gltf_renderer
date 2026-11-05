/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Where a DAZ texture comes from outside the cooked set, with and without Original textures.

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>

#include "fv_textures.hpp"

using fv::textures::Fallback;
using fv::textures::fallbackOf;

TEST(FvTextures, Readable)
{
  EXPECT_TRUE(fv::textures::readable("a/b/Skin.JPG"));
  EXPECT_TRUE(fv::textures::readable("x.webp"));
  EXPECT_FALSE(fv::textures::readable("x.tif"));
  EXPECT_FALSE(fv::textures::readable("x.bmp"));
}

TEST(FvTextures, CookerFoldsCutouts)
{
  // plain sources as they are, cutouts and other laid textures cooked, roughness one texel until its
  // cook lands, laid or not
  EXPECT_EQ(fallbackOf("d.jpg", false, false, true, false), Fallback::eSource);
  EXPECT_EQ(fallbackOf("d.jpg", false, true, true, false), Fallback::eCook);
  EXPECT_EQ(fallbackOf("r.jpg", true, false, true, false), Fallback::eFlat);
  EXPECT_EQ(fallbackOf("r.jpg", true, true, true, false), Fallback::eFlat);
  EXPECT_EQ(fallbackOf("d.tif", false, false, true, false), Fallback::eCook);
}

TEST(FvTextures, OriginalCooksOnlyUnreadable)
{
  EXPECT_EQ(fallbackOf("d.jpg", false, false, true, true), Fallback::eSource);
  EXPECT_EQ(fallbackOf("d.jpg", false, true, true, true), Fallback::eCompose);
  EXPECT_EQ(fallbackOf("r.jpg", true, false, true, true), Fallback::eCompose);
  // the renderer can't read these, so they still go to the cooker
  EXPECT_EQ(fallbackOf("d.tif", false, true, true, true), Fallback::eCook);
  EXPECT_EQ(fallbackOf("r.tif", true, false, true, true), Fallback::eCook);
}

TEST(FvTextures, NoCooker)
{
  // the same whether or not Original textures is on
  for(bool original : {false, true})
  {
    EXPECT_EQ(fallbackOf("d.jpg", false, true, false, original), Fallback::eCompose);
    EXPECT_EQ(fallbackOf("d.tif", false, true, false, original), Fallback::eSource);
    EXPECT_EQ(fallbackOf("r.tif", true, false, false, original), Fallback::eNone);
  }
}

// A metal map packed into glTF's metallic-roughness is derived like a roughness map: as it is, its
// grey would be read as roughness and full metal everywhere.
TEST(FvTextures, MetalChannelsAreDerived)
{
  EXPECT_TRUE(fv::textures::derivedChannels("mt"));
  EXPECT_TRUE(fv::textures::derivedChannels("mr"));
  EXPECT_TRUE(fv::textures::derivedChannels("bn"));
  EXPECT_FALSE(fv::textures::derivedChannels("ng"));
  EXPECT_FALSE(fv::textures::derivedChannels(""));
  EXPECT_EQ(fallbackOf("m.jpg", fv::textures::derivedChannels("mt"), false, true, true), Fallback::eCompose);
  EXPECT_EQ(fallbackOf("m.jpg", fv::textures::derivedChannels("mt"), false, true, false), Fallback::eFlat);
}

// A weight map's grey in every channel: the source as it is would put full specular in its alpha
TEST(FvTextures, WeightChannelsAreDerived)
{
  EXPECT_TRUE(fv::textures::derivedChannels("wa"));
  EXPECT_EQ(fallbackOf("Skin_SLW.jpg", true, false, false, false), Fallback::eCompose);
  EXPECT_EQ(fallbackOf("slw.tif", true, false, false, false), Fallback::eNone);
}

// The composed fallbacks' folder comes down to its cap oldest first; a stand-in kept by its name, and
// what is under the cap, stay
TEST(FvTextures, TrimFolderTakesTheOldestFirst)
{
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "fv-trim-folder-test";
  fs::remove_all(dir);
  fs::create_directories(dir / "below");
  const auto now  = fs::file_time_type::clock::now();
  const auto make = [&](const char* name, size_t bytes, int hoursAgo) {
    std::ofstream(dir / name, std::ios::binary) << std::string(bytes, 'x');
    fs::last_write_time(dir / name, now - std::chrono::hours(hoursAgo));
  };
  make("old.png", 400, 30);
  make("flat-white.png", 400, 40);
  make("middle.png", 300, 20);
  make("new.png", 300, 1);
  std::ofstream(dir / "below" / "deep.png") << std::string(1000, 'x');
  // 1000 bytes count (the kept stand-in and the folder below don't): the two oldest must go to come to 350
  EXPECT_EQ(fv::textures::trimFolder(dir, 350, L"flat-"), 700u);
  EXPECT_FALSE(fs::exists(dir / "old.png"));
  EXPECT_FALSE(fs::exists(dir / "middle.png"));
  EXPECT_TRUE(fs::exists(dir / "new.png"));
  EXPECT_TRUE(fs::exists(dir / "flat-white.png"));
  EXPECT_TRUE(fs::exists(dir / "below" / "deep.png"));
  // under the cap nothing goes, and a folder that isn't there is no error
  EXPECT_EQ(fv::textures::trimFolder(dir, 350, L"flat-"), 0u);
  EXPECT_EQ(fv::textures::trimFolder(dir / "missing", 0), 0u);
  fs::remove_all(dir);
}
