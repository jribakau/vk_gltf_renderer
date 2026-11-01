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

// Compiled twice in a run isn't the driver's doing: a warm-up and the pipeline it warms, both compiling
// at once, said "the cache is full" of a cache just emptied
TEST(PipelineBuildRecord, TwoCompilesInOneRunSayNothingOfTheDriver)
{
  std::filesystem::remove(recordPath());
  PipelineBuildRecord record;
  record.load(recordPath());
  EXPECT_FALSE(record.note(1, 15000.0));  // new: a compile is expected
  EXPECT_FALSE(record.note(1, 70.0));     // again, from the driver's cache
  EXPECT_FALSE(record.note(1, 15000.0));  // again, compiled
  EXPECT_FALSE(record.driverDropped());
  std::filesystem::remove(recordPath());
}

// What the badge has for how long a wait will be: the last run's longest, and not the longest ever, since
// what a shader takes changes with the shader and the driver
TEST(PipelineBuildRecord, TheLastRunsLongestCompileIsKept)
{
  std::filesystem::remove(recordPath());
  {
    PipelineBuildRecord first;
    first.load(recordPath());
    EXPECT_EQ(first.lastCompiledMs(), 0.0) << "none on record";
    first.note(1, 20000.0);
    first.note(2, 70.0);  // from the cache: no compile
    first.note(3, 23000.0);
    first.note(4, 5000.0);
    EXPECT_EQ(first.lastCompiledMs(), 0.0) << "what this run takes is for the next";
  }
  {
    PipelineBuildRecord second;
    second.load(recordPath());
    EXPECT_EQ(second.lastCompiledMs(), 23000.0);
    EXPECT_FALSE(second.note(4, 70.0)) << "the keys still read among the times";
    second.note(5, 9000.0);
  }
  {
    PipelineBuildRecord third;
    third.load(recordPath());
    EXPECT_EQ(third.lastCompiledMs(), 9000.0);
    EXPECT_TRUE(third.note(1, 15000.0));
  }
  std::filesystem::remove(recordPath());
  {
    PipelineBuildRecord fromCache;
    fromCache.load(recordPath());
    fromCache.note(1, 70.0);
  }
  PipelineBuildRecord none;
  none.load(recordPath());
  EXPECT_EQ(none.lastCompiledMs(), 0.0) << "a run that only read the cache says nothing of a compile";
  std::filesystem::remove(recordPath());
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
