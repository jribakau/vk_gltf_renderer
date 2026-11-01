#include <gtest/gtest.h>

#include "camera_controls.hpp"

using camera_controls::dragMotion;
using camera_controls::Manipulator;
using Actions = Manipulator::Actions;
using Inputs  = Manipulator::Inputs;

TEST(CameraDrag, TheLeftButtonOrbitsInExamine)
{
  EXPECT_EQ(dragMotion({.lmb = true}, Manipulator::Examine), Actions::Orbit);
  EXPECT_EQ(dragMotion({.lmb = true, .shift = true}, Manipulator::Examine), Actions::Orbit);
  EXPECT_EQ(dragMotion({.lmb = true, .ctrl = true}, Manipulator::Examine), Actions::Pan);
  EXPECT_EQ(dragMotion({.lmb = true, .shift = true, .ctrl = true}, Manipulator::Examine), Actions::Pan);
  EXPECT_EQ(dragMotion({.lmb = true, .alt = true}, Manipulator::Examine), Actions::LookAround);
  EXPECT_EQ(dragMotion({.lmb = true, .shift = true, .alt = true}, Manipulator::Examine), Actions::LookAround);
  EXPECT_EQ(dragMotion({.lmb = true, .ctrl = true, .alt = true}, Manipulator::Examine), Actions::LookAround);
}

TEST(CameraDrag, TheLeftButtonLooksAroundInFlyAndWalk)
{
  for(Manipulator::Modes mode : {Manipulator::Fly, Manipulator::Walk})
  {
    EXPECT_EQ(dragMotion({.lmb = true}, mode), Actions::LookAround);
    EXPECT_EQ(dragMotion({.lmb = true, .shift = true}, mode), Actions::Orbit);
    EXPECT_EQ(dragMotion({.lmb = true, .ctrl = true}, mode), Actions::Pan);
    EXPECT_EQ(dragMotion({.lmb = true, .alt = true}, mode), Actions::Orbit);
  }
}

TEST(CameraDrag, TheOtherButtonsIgnoreModifiersAndTheLeftComesFirst)
{
  for(Manipulator::Modes mode : {Manipulator::Examine, Manipulator::Fly, Manipulator::Walk})
  {
    EXPECT_EQ(dragMotion({.mmb = true}, mode), Actions::Pan);
    EXPECT_EQ(dragMotion({.mmb = true, .shift = true, .alt = true}, mode), Actions::Pan);
    EXPECT_EQ(dragMotion({.rmb = true}, mode), Actions::Dolly);
    EXPECT_EQ(dragMotion({.rmb = true, .shift = true}, mode), Actions::Dolly);
    EXPECT_EQ(dragMotion({.lmb = true, .mmb = true, .ctrl = true}, mode), Actions::Pan);
    EXPECT_EQ(dragMotion(Inputs{}, mode), Actions::NoAction);
  }
}
