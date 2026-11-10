#include <gtest/gtest.h>

#include "camera_controls.hpp"

using camera_controls::dragMotion;
using camera_controls::Manipulator;
using Actions = Manipulator::Actions;
using Inputs  = Manipulator::Inputs;

// An editor's scheme (Unity, Unreal): Alt+left orbits, the left alone is a click's
TEST(CameraDrag, AltAndTheLeftButtonOrbitAndTheLeftAloneMovesNothing)
{
  EXPECT_EQ(dragMotion({.lmb = true, .alt = true}, Manipulator::Examine), Actions::Orbit);
  EXPECT_EQ(dragMotion({.lmb = true, .shift = true, .alt = true}, Manipulator::Examine), Actions::Orbit);
  EXPECT_EQ(dragMotion({.lmb = true}, Manipulator::Examine), Actions::NoAction);
  EXPECT_EQ(dragMotion({.lmb = true, .ctrl = true}, Manipulator::Examine), Actions::NoAction);
  EXPECT_EQ(dragMotion({.lmb = true, .shift = true}, Manipulator::Examine), Actions::NoAction);
  for(Manipulator::Modes mode : {Manipulator::Fly, Manipulator::Walk})
    EXPECT_EQ(dragMotion({.lmb = true, .alt = true}, mode), Actions::LookAround);
}

TEST(CameraDrag, TheMiddlePansAndAltWithTheRightDollies)
{
  for(Manipulator::Modes mode : {Manipulator::Examine, Manipulator::Fly, Manipulator::Walk})
  {
    EXPECT_EQ(dragMotion({.mmb = true}, mode), Actions::Pan);
    EXPECT_EQ(dragMotion({.mmb = true, .shift = true, .alt = true}, mode), Actions::Pan);
    EXPECT_EQ(dragMotion({.rmb = true, .alt = true}, mode), Actions::Dolly);
    // the right alone looks around, which the host does
    EXPECT_EQ(dragMotion({.rmb = true}, mode), Actions::NoAction);
    EXPECT_EQ(dragMotion({.lmb = true, .mmb = true, .alt = true}, Manipulator::Examine), Actions::Orbit);
    EXPECT_EQ(dragMotion(Inputs{}, mode), Actions::NoAction);
  }
}

TEST(CameraDrag, OnlyACameraDragMovesTheCamera)
{
  EXPECT_TRUE(camera_controls::dragMovesCamera({.lmb = true, .alt = true}));
  EXPECT_TRUE(camera_controls::dragMovesCamera({.mmb = true}));
  EXPECT_TRUE(camera_controls::dragMovesCamera({.rmb = true}));
  EXPECT_FALSE(camera_controls::dragMovesCamera({.lmb = true}));
  EXPECT_FALSE(camera_controls::dragMovesCamera({.lmb = true, .ctrl = true}));
}
