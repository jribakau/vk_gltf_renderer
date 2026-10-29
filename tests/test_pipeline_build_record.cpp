// A pipeline built before that compiles again says the driver's shader cache dropped it; a new one
// compiling, or one built before coming back fast, says nothing (src/pipeline_build_record.hpp).

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "pipeline_build_record.hpp"

namespace {

std::filesystem::path recordPath()
{
  return std::filesystem::temp_directory_path() / "fv_pipeline_build_record_test.txt";
}

}  // namespace

TEST(PipelineBuildRecord, ARebuiltPipelineThatCompilesAgainWasDropped)
{
  PipelineBuildRecord record;
  EXPECT_FALSE(record.note(1, 15000.0));  // new: a compile is expected
  EXPECT_FALSE(record.note(1, 70.0));     // again, from the driver's cache
  EXPECT_FALSE(record.driverDropped());
  EXPECT_TRUE(record.note(1, 15000.0));   // again, and compiled
  EXPECT_TRUE(record.driverDropped());
}

TEST(PipelineBuildRecord, TheKeysOutliveTheProcess)
{
  std::filesystem::remove(recordPath());
  {
    PipelineBuildRecord first;
    first.load(recordPath());
    EXPECT_FALSE(first.note(0xfb2f4ba214aea8ddull, 15000.0));
    EXPECT_FALSE(first.note(7, 15000.0));
  }
  // a line a killed write cut short is skipped
  std::ofstream(recordPath(), std::ios::app) << "zz\n";
  PipelineBuildRecord next;
  next.load(recordPath());
  EXPECT_FALSE(next.note(0xfb2f4ba214aea8ddull, 700.0));
  EXPECT_FALSE(next.driverDropped());
  EXPECT_TRUE(next.note(7, 15000.0));
  EXPECT_FALSE(next.note(8, 15000.0));
  // a missing file is an empty record
  std::filesystem::remove(recordPath());
  PipelineBuildRecord empty;
  empty.load(recordPath());
  EXPECT_FALSE(empty.note(7, 15000.0));
  std::filesystem::remove(recordPath());
}
