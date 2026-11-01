// What first hits want of a material's textures (shaders/hit_signal.h.slang) compiled as C++, and what the Fit makes of the count.

#include <gtest/gtest.h>

#include "../shaders/hit_signal.h.slang"
#include "../src/memory_plan.hpp"

static_assert(HIT_SIGNAL_BINS == rs::kHitBins);

namespace {

rs::ImageFit fitOf(uint32_t extent, std::array<float, rs::kHitBins> wanting, bool pinned = false)
{
  rs::ImageFit fit;
  fit.chain.width = fit.chain.height = extent;
  for(uint32_t e = extent;; e /= 2)
  {
    fit.chain.levelBytes.push_back(uint64_t(e) * e * 4);
    if(e == 1)
      break;
  }
  fit.pinned  = pinned;
  fit.wanting = wanting;
  return fit;
}

}  // namespace

// A footprint wants the long edge that gives a pixel two texels
TEST(HitSignal, ABinIsTheLongEdgeAFootprintCanShow)
{
  EXPECT_EQ(hitSignalBin(0.f), 0) << "it samples no texture";
  EXPECT_EQ(hitSignalBin(1.f), 0);
  EXPECT_EQ(hitSignalBin(1.f / 128), 0);
  EXPECT_EQ(hitSignalBin(1.f / 150), 1);
  EXPECT_EQ(hitSignalBin(1.f / 256), 1);
  EXPECT_EQ(hitSignalBin(1.f / 512), 2);
  EXPECT_EQ(hitSignalBin(1.f / 1024), 3);
  EXPECT_EQ(hitSignalBin(1.f / 2048), 4);
  EXPECT_EQ(hitSignalBin(1.f / 4096), 5);
  EXPECT_EQ(hitSignalBin(1e-7f), 5) << "the last holds the rest";
}

// One pixel in sixteen each frame, and over a window of them more than the same few
TEST(HitSignal, OnePixelInSixteenCounts)
{
  for(int frame : {1, 2, 3, 8})
  {
    int counted = 0;
    for(int y = 0; y < 64; y++)
      for(int x = 0; x < 64; x++)
        counted += hitSignalPixel(x, y, frame) ? 1 : 0;
    EXPECT_EQ(counted, 64 * 64 / 16);
  }
  int everCounted = 0;
  for(int y = 0; y < 4; y++)
    for(int x = 0; x < 4; x++)
    {
      bool any = false;
      for(int frame = 1; frame <= 8; frame++)
        any = any || hitSignalPixel(x, y, frame);
      everCounted += any ? 1 : 0;
    }
  EXPECT_GE(everCounted, 8);
}

// A level goes by what it frees for the hits it would cost: what none want before what any do, however tiny
TEST(HitSignal, WhatHitsDontWantGoesFirst)
{
  const uint64_t top = 1024ull * 1024 * 4;
  // all of the first's hits could show 256; half of the scene's hits want the second's whole 1024
  const std::array<float, rs::kHitBins> distant{0.01f}, closeUp{0, 0, 0.5f};
  std::vector<rs::ImageFit> images{fitOf(2048, distant), fitOf(1024, closeUp)};
  EXPECT_DOUBLE_EQ(images[0].lossWithout(0), 0.0);
  EXPECT_DOUBLE_EQ(images[1].lossWithout(0), 0.5);
  rs::lowerToFit(images, 1, 256);
  EXPECT_EQ(images[0].skip, 1u);
  rs::lowerToFit(images, top * 5 / 4, 256);
  EXPECT_EQ(images[0].extent(), 256u) << "all the way, before anything wanted";
  EXPECT_EQ(images[1].skip, 0u);

  // what isn't known counts for its bytes alone, as before there were hits: after what none want
  rs::ImageFit unknown = fitOf(4096, distant);
  unknown.wanting.reset();
  std::vector<rs::ImageFit> mixed{std::move(unknown), fitOf(2048, distant)};
  rs::lowerToFit(mixed, 1, 256);
  EXPECT_EQ(mixed[1].skip, 1u);
  EXPECT_EQ(mixed[0].skip, 0u);

  // the plan's rungs: what no ray reaches goes before any texture, what they reach after textures reach 2K
  std::vector<rs::ImageFit> plain{fitOf(4096, closeUp)};
  const std::vector<uint64_t> detail{100, 200};
  rs::FitPlan plan = rs::planToFit(50, {.images = plain, .leastExtent = 256, .detail = detail, .detailAfter = 2048, .detailFirst = 1});
  EXPECT_EQ(plan.detail, std::vector<size_t>{0});
  EXPECT_EQ(plain[0].skip, 0u);
}

// Memory full: what hits want back makes room against what they want less, by at least twice over, and no more
TEST(HitSignal, AnExchangeNeedsTwiceTheHitsForTheBytes)
{
  // the hero's 2K is down at 1K, and half the hits want it back; a far figure's 2K has hits that could show 256
  const std::array<float, rs::kHitBins> hero{0, 0, 0, 0.5f}, distant{0.001f};
  rs::ImageFit                          down = fitOf(2048, hero);
  down.skip                                  = 1;
  std::vector<rs::ImageFit>   images{down, fitOf(4096, distant)};
  const std::vector<uint32_t> wanted{0, 0};
  const uint64_t              level = 2048ull * 2048 * 4;
  const uint64_t              peak  = level + down.bytes();  // it comes whole beside the one it replaces
  EXPECT_EQ(rs::lowerForExchange(images, wanted, 256, peak), peak);
  EXPECT_EQ(images[0].skip, 1u) << "it comes up as anything lowered does, when the room shows";
  EXPECT_EQ(images[1].skip, 1u);

  // short of the room to begin with: the first raising pays for that too
  std::vector<rs::ImageFit> shortOf{down, fitOf(4096, distant)};
  const uint64_t            lacking = 4096ull * 4096 * 4 - peak + 1;  // its top level just doesn't do
  EXPECT_EQ(rs::lowerForExchange(shortOf, wanted, 256, peak, lacking), peak + lacking);
  EXPECT_EQ(shortOf[1].skip, 2u);

  // nothing that gains too little: as wanted by one as by the other
  rs::ImageFit same = fitOf(2048, hero);
  same.skip         = 1;
  std::vector<rs::ImageFit> level2{same, fitOf(2048, hero)};
  EXPECT_EQ(rs::lowerForExchange(level2, wanted, 256, level), 0u);
  EXPECT_EQ(level2[1].skip, 0u);

  // nor for what no hit wants: a level none can show is no gain, however small it is
  rs::ImageFit nobody = fitOf(2048, {0.5f});
  nobody.skip         = 1;
  std::vector<rs::ImageFit> unwanted{nobody, fitOf(4096, distant)};
  EXPECT_EQ(rs::lowerForExchange(unwanted, wanted, 256, level), 0u);
  EXPECT_EQ(unwanted[1].skip, 0u);

  // nor at the cost of what is Keep full, or anything not yet known
  rs::ImageFit wants = fitOf(2048, hero);
  wants.skip         = 1;
  rs::ImageFit unknown = fitOf(2048, distant);
  unknown.wanting.reset();
  std::vector<rs::ImageFit> kept{wants, fitOf(2048, distant, true), std::move(unknown)};
  const std::vector<uint32_t> three{0, 0, 0};
  EXPECT_EQ(rs::lowerForExchange(kept, three, 256, level), 0u);
  EXPECT_EQ(kept[1].skip, 0u);
  EXPECT_EQ(kept[2].skip, 0u);

  // what doesn't cover it is put back as it was
  rs::ImageFit big = fitOf(2048, hero);
  big.skip         = 1;
  std::vector<rs::ImageFit> tiny{big, fitOf(64, distant)};
  EXPECT_EQ(rs::lowerForExchange(tiny, wanted, 8, level), 0u);
  EXPECT_EQ(tiny[1].skip, 0u);
}
