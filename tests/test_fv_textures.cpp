/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Where a DAZ texture comes from outside the cooked set, with and without Original textures.

#include <gtest/gtest.h>

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
  // plain sources as they are, cutouts and other laid textures cooked, roughness composed here
  EXPECT_EQ(fallbackOf("d.jpg", false, false, true, false), Fallback::eSource);
  EXPECT_EQ(fallbackOf("d.jpg", false, true, true, false), Fallback::eCook);
  EXPECT_EQ(fallbackOf("r.jpg", true, false, true, false), Fallback::eCompose);
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
