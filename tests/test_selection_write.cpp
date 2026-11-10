// SelectionWrite (src/selection_write.hpp): when the path tracer traces the selection image's object IDs.

#include <gtest/gtest.h>

#include "../src/selection_write.hpp"

TEST(SelectionWrite, AStillWritesItOnItsFirstFrameOnly)
{
  SelectionWrite s;
  EXPECT_TRUE(s.due(true, false, false));
  EXPECT_FALSE(s.due(false, false, false));
  EXPECT_FALSE(s.due(false, false, true));  // the frame 0 image is current: a selection reads that one
}

TEST(SelectionWrite, ADlssFrameWritesItOnlyWhileItIsRead)
{
  SelectionWrite s;
  EXPECT_FALSE(s.due(true, true, false));  // orbiting with nothing selected
  EXPECT_FALSE(s.due(true, true, false));
  EXPECT_TRUE(s.due(true, true, true));  // something selected: every moving frame
  EXPECT_TRUE(s.due(true, true, true));
}

// The hover's outline reads the image only once the view rests: a moving DLSS frame traces no IDs
// for it, the first frame after does, once
TEST(SelectionWrite, TheHoverReadsItOnceTheViewRests)
{
  EXPECT_FALSE(SelectionWrite::read(false, false, true, true));
  EXPECT_FALSE(SelectionWrite::read(false, true, true, true));  // orbiting with the hover on: no selection ray
  EXPECT_TRUE(SelectionWrite::read(true, true, true, true));    // a selection reads every frame
  EXPECT_TRUE(SelectionWrite::read(false, true, false, true));  // resting on DLSS frames
  EXPECT_TRUE(SelectionWrite::read(false, true, true, false));  // a still's frame 0
  SelectionWrite s;
  // orbit, then rest: written on the first resting frame, then not again
  EXPECT_FALSE(s.due(true, true, SelectionWrite::read(false, true, true, true)));
  EXPECT_FALSE(s.due(true, true, SelectionWrite::read(false, true, true, true)));
  EXPECT_TRUE(s.stale);
  EXPECT_TRUE(s.due(false, true, SelectionWrite::read(false, true, false, true)));
  EXPECT_FALSE(s.stale);
  EXPECT_FALSE(s.due(false, true, SelectionWrite::read(false, true, false, true)));
  // moving again leaves it stale, so nothing is outlined from it
  EXPECT_FALSE(s.due(true, true, SelectionWrite::read(false, true, true, true)));
  EXPECT_TRUE(s.stale);
}

TEST(SelectionWrite, AReaderAfterASkippedFrameGetsItOnce)
{
  // the view rests on DLSS frames (their count climbs) when the selection is made
  SelectionWrite s;
  EXPECT_FALSE(s.due(true, true, false));
  EXPECT_FALSE(s.due(false, true, false));
  EXPECT_TRUE(s.due(false, true, true));
  EXPECT_FALSE(s.due(false, true, true));  // written; the view has not moved
  // a still's frame 0 makes it current again
  EXPECT_FALSE(s.due(true, true, false));
  EXPECT_TRUE(s.due(true, false, false));
  EXPECT_FALSE(s.due(false, false, true));
}
