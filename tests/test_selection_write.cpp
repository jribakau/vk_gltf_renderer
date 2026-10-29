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
