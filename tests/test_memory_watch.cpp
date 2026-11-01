#include <gtest/gtest.h>

#include "memory_plan.hpp"
#include "memory_watch.hpp"
#include "ram_watch.hpp"

namespace {

using rs::MemoryWatch;
using Act = MemoryWatch::Act;

constexpr int64_t kRoom = int64_t(1) << 30;

MemoryWatch::Reading shortOfMemory(double now)
{
  return {.now = now, .headroom = -1, .canLower = true};
}

MemoryWatch::Reading roomFor(double now)
{
  return {.now = now, .headroom = kRoom, .canLower = true, .lowered = true};
}

// A reading that dips isn't acted on; one that stays short is, and then waits for what it freed
TEST(MemoryWatch, LowersOnlyWhenTheHeadroomStaysGone)
{
  MemoryWatch watch;
  EXPECT_EQ(watch.read(shortOfMemory(0)), Act::eNone);
  EXPECT_EQ(watch.read(roomFor(1)), Act::eNone);
  EXPECT_EQ(watch.read(shortOfMemory(2)), Act::eNone);
  EXPECT_EQ(watch.read(shortOfMemory(3)), Act::eNone);
  EXPECT_EQ(watch.read(shortOfMemory(2 + MemoryWatch::kLowerAfter)), Act::eLower);
  EXPECT_EQ(watch.read(shortOfMemory(3 + MemoryWatch::kLowerAfter)), Act::eNone);
  EXPECT_EQ(watch.read(shortOfMemory(2 + 2 * MemoryWatch::kLowerAfter)), Act::eLower);
}

// Nothing left to lower: nothing is asked, however long it stays short
TEST(MemoryWatch, DoesNotAskWhatCannotComeDown)
{
  MemoryWatch          watch;
  MemoryWatch::Reading r = shortOfMemory(0);
  r.canLower             = false;
  EXPECT_EQ(watch.read(r), Act::eNone);
  r.now = 60;
  EXPECT_EQ(watch.read(r), Act::eNone);
}

// A lowering restarts a still and changes an export: only past the budget is it worth that
TEST(MemoryWatch, LowersDuringARenderOnlyPastTheBudget)
{
  for(const bool held : {false, true})
  {
    MemoryWatch          watch;
    MemoryWatch::Reading r = shortOfMemory(0);
    r.refining             = !held;
    r.held                 = held;
    EXPECT_EQ(watch.read(r), Act::eNone);
    r.now = MemoryWatch::kLowerAfter;
    EXPECT_EQ(watch.read(r), Act::eNone);
    r.pastBudget = true;
    EXPECT_EQ(watch.read(r), Act::eLower);
  }
}

// Memory that comes back has to stay, and a lowering is left alone for a while
TEST(MemoryWatch, RaisesOnceTheRoomHasSettled)
{
  MemoryWatch watch;
  EXPECT_EQ(watch.read(roomFor(0)), Act::eNone);
  EXPECT_EQ(watch.read(roomFor(MemoryWatch::kRaiseAfter - 1)), Act::eNone);
  EXPECT_EQ(watch.read(roomFor(MemoryWatch::kRaiseAfter)), Act::eRaise);

  MemoryWatch lowered;
  EXPECT_EQ(lowered.read(shortOfMemory(0)), Act::eNone);
  EXPECT_EQ(lowered.read(shortOfMemory(MemoryWatch::kLowerAfter)), Act::eLower);
  const double after = MemoryWatch::kLowerAfter + MemoryWatch::kRaiseAfterLowering;
  EXPECT_EQ(lowered.read(roomFor(after - 1)), Act::eNone);  // the room has been there too short
  EXPECT_EQ(lowered.read(roomFor(after - 1 + MemoryWatch::kRaiseAfter)), Act::eRaise);

  MemoryWatch soon;
  EXPECT_EQ(soon.read(shortOfMemory(0)), Act::eNone);
  EXPECT_EQ(soon.read(shortOfMemory(MemoryWatch::kLowerAfter)), Act::eLower);
  EXPECT_EQ(soon.read(roomFor(MemoryWatch::kLowerAfter + 1)), Act::eNone);
  EXPECT_EQ(soon.read(roomFor(MemoryWatch::kLowerAfter + 1 + MemoryWatch::kRaiseAfter)), Act::eNone);  // settled, but too soon
}

// Room only for what stays free, or nothing to bring back: nothing to do
TEST(MemoryWatch, DoesNotRaiseIntoItsMarginOrWithNothingLowered)
{
  MemoryWatch          watch;
  MemoryWatch::Reading r = roomFor(0);
  r.headroom             = MemoryWatch::kMargin;
  EXPECT_EQ(watch.read(r), Act::eNone);
  r.now = 60;
  EXPECT_EQ(watch.read(r), Act::eNone);

  MemoryWatch          clean;
  MemoryWatch::Reading whole = roomFor(0);
  whole.lowered              = false;
  EXPECT_EQ(clean.read(whole), Act::eNone);
  whole.now = 60;
  EXPECT_EQ(clean.read(whole), Act::eNone);
}

// A still being refined waits for its end; a raising that began carries on through the still it
// restarted, and an export stops it until the export is over
TEST(MemoryWatch, ARaisingWaitsForARenderAndCarriesOnThroughItsOwnRestart)
{
  MemoryWatch          watch;
  MemoryWatch::Reading r = roomFor(0);
  r.refining             = true;
  EXPECT_EQ(watch.read(r), Act::eNone);
  r.now = 60;
  EXPECT_EQ(watch.read(r), Act::eNone);
  r.refining = false;
  r.now      = 61;
  EXPECT_EQ(watch.read(r), Act::eRaise);
  r.refining = true;
  r.now      = 62;
  EXPECT_EQ(watch.read(r), Act::eRaise);

  r.held = true;
  r.now  = 63;
  EXPECT_EQ(watch.read(r), Act::eNone);
  r.held = false;
  r.now  = 64;
  EXPECT_EQ(watch.read(r), Act::eRaise);
}

// Nothing fitted the room: it is left a while before the chains are read again
TEST(MemoryWatch, ARaisingThatFittedNothingWaits)
{
  MemoryWatch watch;
  EXPECT_EQ(watch.read(roomFor(0)), Act::eNone);
  EXPECT_EQ(watch.read(roomFor(MemoryWatch::kRaiseAfter)), Act::eRaise);
  watch.raiseEnded(MemoryWatch::kRaiseAfter);
  EXPECT_EQ(watch.read(roomFor(MemoryWatch::kRaiseAfter + 1)), Act::eNone);
  EXPECT_EQ(watch.read(roomFor(2 * MemoryWatch::kRaiseAfter)), Act::eRaise);
}

// Read once a period
TEST(MemoryWatch, IsDueOnceAPeriod)
{
  MemoryWatch watch;
  EXPECT_TRUE(watch.due(0));
  watch.read(roomFor(0));
  EXPECT_FALSE(watch.due(MemoryWatch::kPeriod / 2));
  EXPECT_TRUE(watch.due(MemoryWatch::kPeriod));
}

// What a raising can take: nothing of the margin, and a batch at most
TEST(MemoryWatch, ARaisingTakesABatchAtMostAndLeavesTheMargin)
{
  EXPECT_EQ(MemoryWatch::raiseRoom(MemoryWatch::kMargin), 0u);
  EXPECT_EQ(MemoryWatch::raiseRoom(-1), 0u);
  EXPECT_EQ(MemoryWatch::raiseRoom(MemoryWatch::kMargin + 5), 5u);
  EXPECT_EQ(MemoryWatch::raiseRoom(kRoom * 8), uint64_t(MemoryWatch::kRaiseBatch));
}

}  // namespace

// System memory: short under a sixteenth of it or 2 GiB, whichever is more, of it or of what can still be
// committed; said once; no longer short at twice that, so that what the host gives back doesn't flip it.
TEST(RamWatch, ShortIsSaidOnceAndEndsWellAboveWhereItStarted)
{
  constexpr uint64_t kGiB = uint64_t(1) << 30;
  rs::RamWatch       watch;
  const auto         at = [&](uint64_t physical, uint64_t commit = 90) { return watch.step({64 * kGiB, physical * kGiB, commit * kGiB}); };
  EXPECT_FALSE(at(30).short_);
  EXPECT_FALSE(at(4).short_);  // a sixteenth of 64 GiB
  auto step = at(3);
  EXPECT_TRUE(step.short_);
  EXPECT_TRUE(step.justNow);
  step = at(2);
  EXPECT_TRUE(step.short_);
  EXPECT_FALSE(step.justNow);
  EXPECT_TRUE(at(7).short_);
  EXPECT_FALSE(at(8).short_);
  EXPECT_FALSE(at(5).short_);
  // nothing left to commit is short too, whatever is free
  step = at(30, 1);
  EXPECT_TRUE(step.short_ && step.justNow);
  // a small machine: 2 GiB
  rs::RamWatch small;
  EXPECT_FALSE(small.step({8 * kGiB, 3 * kGiB, 90 * kGiB}).short_);
  EXPECT_TRUE(small.step({8 * kGiB, kGiB, 90 * kGiB}).short_);
}

TEST(RamWatch, WhatIsKeptForLaterGoesOldestFirst)
{
  const std::vector<uint64_t> bytes = {100, 200, 300};
  EXPECT_EQ(rs::oldestToLetGo(bytes, 600), 0u);
  EXPECT_EQ(rs::oldestToLetGo(bytes, 599), 1u);
  EXPECT_EQ(rs::oldestToLetGo(bytes, 300), 2u);
  EXPECT_EQ(rs::oldestToLetGo(bytes, 299), 3u);
  EXPECT_EQ(rs::oldestToLetGo({}, 0), 0u);
}

namespace {
rs::ImageFit fitOf(uint32_t extent, bool pinned = false)
{
  rs::ImageFit fit;
  fit.chain.width = fit.chain.height = extent;
  for(uint32_t e = extent;; e /= 2)
  {
    fit.chain.levelBytes.push_back(uint64_t(e) * e * 4);
    if(e == 1)
      break;
  }
  fit.pinned = pinned;
  return fit;
}
}  // namespace

// What gives way goes in order: what nothing looks at, then the largest texture; what is Keep full last
TEST(MemoryPlan, TheCheapestLossGoesFirst)
{
  const uint64_t              top = 1024ull * 1024 * 4;
  std::vector<rs::ImageFit>   images{fitOf(1024), fitOf(1024, true), fitOf(256)};
  const std::vector<uint64_t> hidden{100, 300, 200};
  const rs::FitRungs          rungs{.idleBytes = 50, .hidden = hidden, .images = images, .leastExtent = 256};

  rs::FitPlan plan = rs::planToFit(30, rungs);
  EXPECT_EQ(plan.idle, 30u);
  EXPECT_TRUE(plan.hidden.empty());
  EXPECT_EQ(plan.images, 0u);

  plan = rs::planToFit(50 + 250, rungs);  // the largest hidden item covers it
  EXPECT_EQ(plan.idle, 50u);
  EXPECT_EQ(plan.hidden, (std::vector<size_t>{1}));
  EXPECT_EQ(plan.hiddenBytes, 300u);
  EXPECT_EQ(images[0].skip, 0u);

  plan = rs::planToFit(650 + 1, rungs);  // and all of them, then a level
  EXPECT_EQ(plan.hidden, (std::vector<size_t>{1, 2, 0}));
  EXPECT_EQ(images[0].skip, 1u);
  EXPECT_EQ(images[1].skip, 0u) << "what is pinned waits";
  EXPECT_EQ(plan.images, top);

  // everything else down to the least, and only then what is pinned
  images                   = {fitOf(1024), fitOf(1024, true), fitOf(256)};
  const uint64_t theRest   = top + top / 4;  // the first image, 1024 and 512
  plan                     = rs::planToFit(650 + theRest + 1, rungs);
  EXPECT_EQ(images[0].skip, 2u);
  EXPECT_EQ(images[1].skip, 1u);
  EXPECT_EQ(images[2].skip, 0u);
  EXPECT_EQ(plan.freed(), 650 + theRest + top);

  // and less than asked, when there is no more to give up
  images = {fitOf(1024), fitOf(1024, true), fitOf(256)};
  plan   = rs::planToFit(UINT64_MAX / 2, rungs);
  EXPECT_EQ(plan.freed(), 650 + 2 * theRest);
  EXPECT_EQ(images[0].extent(), 256u);
  EXPECT_EQ(images[1].extent(), 256u);
}

// Asked for nothing, it does nothing; asked again for what it has done, it goes on from there; asked for
// more it does what it would have done for less, and more
TEST(MemoryPlan, IsMonotonicAndGoesOnFromWhereItIs)
{
  const auto levels = [](uint64_t wanted) {
    std::vector<rs::ImageFit> images{fitOf(2048), fitOf(1024), fitOf(1024, true), fitOf(512)};
    (void)rs::planToFit(wanted, {.images = images, .leastExtent = 64});
    std::vector<uint32_t> skips;
    for(const rs::ImageFit& image : images)
      skips.push_back(image.skip);
    return skips;
  };
  std::vector<uint32_t> before(4, 0);
  for(uint64_t wanted = 0; wanted < 40ull << 20; wanted += 1 << 20)
  {
    const std::vector<uint32_t> now = levels(wanted);
    for(size_t i = 0; i < now.size(); i++)
      EXPECT_GE(now[i], before[i]) << wanted << " image " << i;
    before = now;
  }

  // nothing wanted: nothing moves
  std::vector<rs::ImageFit> images{fitOf(2048), fitOf(1024)};
  EXPECT_EQ(rs::planToFit(0, {.idleBytes = 10, .images = images, .leastExtent = 64}).freed(), 0u);
  EXPECT_EQ(images[0].skip, 0u);
  // in two steps as in one: the closed loop's steps come to what one plan would have
  std::vector<rs::ImageFit> once = images, twice = images;
  const uint64_t            step = 20ull << 20;
  (void)rs::planToFit(2 * step, {.images = once, .leastExtent = 64});
  (void)rs::planToFit(step, {.images = twice, .leastExtent = 64});
  (void)rs::planToFit(2 * step - (twice[0].skip ? twice[0].chain.levelBytes[0] : 0), {.images = twice, .leastExtent = 64});
  EXPECT_GE(twice[0].skip, once[0].skip);
}

// What is Keep full comes back first, and one with no stored levels comes back all at once
TEST(MemoryPlan, WhatComesBackFirst)
{
  std::vector<rs::ImageFit> images{fitOf(1024), fitOf(1024, true)};
  images[0].skip = images[1].skip = 1;
  const std::vector<uint32_t> wanted{0, 0};
  const uint64_t              onlyOne = images[1].chain.levelBytes[0] + images[1].bytes();
  EXPECT_EQ(rs::raiseToFit(images, wanted, onlyOne), onlyOne);
  EXPECT_EQ(images[1].skip, 0u);
  EXPECT_EQ(images[0].skip, 1u);

  std::vector<rs::ImageFit> whole{fitOf(1024)};
  whole[0].wholeOnly = true;
  whole[0].skip      = 3;
  const std::vector<uint32_t> none{0};
  EXPECT_EQ(rs::raiseToFit(whole, none, 1000), 0u);
  EXPECT_EQ(whole[0].skip, 3u);
  uint64_t all = 0;
  for(uint64_t level : whole[0].chain.levelBytes)
    all += level;
  EXPECT_EQ(rs::raiseToFit(whole, none, all), all);
  EXPECT_EQ(whole[0].skip, 0u);
}
