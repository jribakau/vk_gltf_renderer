#include <gtest/gtest.h>

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
