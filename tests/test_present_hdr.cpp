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

// An HDR display's levels: the SDR image's white at Windows' SDR white level, the peak in SDR whites,
// and what is taken where Windows or the display's EDID says nothing (src/present_hdr.hpp).

#include <gtest/gtest.h>

#include "present_hdr.hpp"

TEST(PresentHdr, SdrWhiteIsTheDisplaysSdrWhiteLevel)
{
  const HdrLevels l = present_hdr::levels({.hdr = true, .sdrWhiteNits = 240.0F, .peakNits = 1200.0F});
  EXPECT_FLOAT_EQ(l.paperWhite, 3.0F);  // 240 nits in scRGB's 80
  EXPECT_FLOAT_EQ(l.peak, 5.0F);        // 1200 nits is five SDR whites
}

TEST(PresentHdr, ADisplayNoBrighterThanSdrWhiteHasNoHeadroom)
{
  EXPECT_FLOAT_EQ(present_hdr::levels({.hdr = true, .sdrWhiteNits = 400.0F, .peakNits = 350.0F}).peak, 1.0F);
}

TEST(PresentHdr, WhatIsUnknownIsTakenAsUsual)
{
  const HdrLevels l = present_hdr::levels({.hdr = true});
  EXPECT_FLOAT_EQ(l.paperWhite, present_hdr::kUnknownSdrWhiteNits / 80.0F);
  EXPECT_FLOAT_EQ(l.peak, present_hdr::kUnknownPeakNits / present_hdr::kUnknownSdrWhiteNits);
}

TEST(PresentHdr, AGivenPeakStandsForTheDisplays)
{
  const HdrLevels l = present_hdr::levels({.hdr = true, .sdrWhiteNits = 200.0F, .peakNits = 1000.0F}, 600.0F);
  EXPECT_FLOAT_EQ(l.peak, 3.0F);
}

// Windows counts the SDR white level in thousandths of 80 nits: 1000 is 80 nits, its default 2500 is 200
TEST(PresentHdr, WindowsSdrWhiteLevelInNits)
{
  EXPECT_FLOAT_EQ(present_hdr::sdrWhiteNits(1000), 80.0F);
  EXPECT_FLOAT_EQ(present_hdr::sdrWhiteNits(2500), 200.0F);
}
