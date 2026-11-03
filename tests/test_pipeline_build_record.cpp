// A pipeline built before that compiles again says the driver's shader cache dropped it; a new one
// compiling, or one built before coming back fast, says nothing (src/pipeline_build_record.hpp).
// A drop is marked for the next start, which empties the renderer's own driver cache, as it does
// one near its cap (src/driver_shader_cache.hpp).

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "driver_shader_cache.hpp"
#include "pipeline_build_record.hpp"

namespace {

std::filesystem::path recordPath()
{
  return std::filesystem::temp_directory_path() / "fv_pipeline_build_record_test.txt";
}

std::string readAll(const std::filesystem::path& path)
{
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// A pipeline cache's folder of its own for a test, empty
std::filesystem::path freshCacheRoot(const char* name)
{
  const std::filesystem::path root = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  return root;
}

void writeBytes(const std::filesystem::path& path, size_t bytes)
{
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << std::string(bytes, 'x');
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

// A drop is marked in the record once, for the next start, which empties the driver's cache
TEST(PipelineBuildRecord, ADropIsMarkedForTheNextStart)
{
  std::filesystem::remove(recordPath());
  {
    PipelineBuildRecord first;
    first.load(recordPath());
    EXPECT_FALSE(first.droppedOnRecord());
    first.note(1, 15000.0);
    first.note(2, 15000.0);
  }
  {
    PipelineBuildRecord second;
    second.load(recordPath());
    EXPECT_FALSE(second.droppedOnRecord()) << "new pipelines compiling say nothing";
    EXPECT_TRUE(second.note(1, 15000.0));
    EXPECT_TRUE(second.note(2, 15000.0));
  }
  const std::string text = readAll(recordPath());
  EXPECT_EQ(text.find("dropped"), text.rfind("dropped")) << "once a run";
  PipelineBuildRecord third;
  third.load(recordPath());
  EXPECT_TRUE(third.droppedOnRecord());
  EXPECT_FALSE(third.driverDropped()) << "this run has dropped nothing yet";
  // the mark is no key: "dropped" begins with a hex digit
  EXPECT_FALSE(third.note(0xd, 15000.0));
  std::filesystem::remove(recordPath());
}

// Once the driver's cache is emptied the keys say nothing of it: built again, they are no drop
TEST(PipelineBuildRecord, ForgettingTheKeysKeepsTheTimes)
{
  std::filesystem::remove(recordPath());
  {
    PipelineBuildRecord first;
    first.load(recordPath());
    first.note(1, 20000.0);
  }
  {
    PipelineBuildRecord second;
    second.load(recordPath());
    EXPECT_TRUE(second.note(1, 25000.0));
  }
  PipelineBuildRecord::forgetKeys(recordPath());
  PipelineBuildRecord third;
  third.load(recordPath());
  EXPECT_FALSE(third.droppedOnRecord());
  EXPECT_EQ(third.lastCompiledMs(), 25000.0) << "the badge's estimate stays";
  EXPECT_FALSE(third.note(1, 25000.0));
  EXPECT_FALSE(third.driverDropped());
  // a record that isn't there is left so
  std::filesystem::remove(recordPath());
  PipelineBuildRecord::forgetKeys(recordPath());
  EXPECT_FALSE(std::filesystem::exists(recordPath()));
}

TEST(DriverShaderCache, AFolderIsMadeBesideThePipelineCache)
{
  const std::filesystem::path root = freshCacheRoot("fv_driver_cache_fresh");
  const driver_shader_cache::Setup s = driver_shader_cache::prepare(root);
  EXPECT_EQ(s.folder, root / "driver");
  EXPECT_TRUE(std::filesystem::is_directory(s.folder));
  EXPECT_EQ(s.emptied, driver_shader_cache::Emptied::eNo);
  EXPECT_EQ(s.bytes, 0u);
  std::filesystem::remove_all(root);
}

TEST(DriverShaderCache, WhatItHoldsBelowTheCapIsKept)
{
  const std::filesystem::path root = freshCacheRoot("fv_driver_cache_below");
  writeBytes(root / "driver" / "GLCache" / "a" / "b.bin", 500);
  const driver_shader_cache::Setup s = driver_shader_cache::prepare(root, 1000);
  EXPECT_EQ(s.emptied, driver_shader_cache::Emptied::eNo);
  EXPECT_EQ(s.bytes, 500u);
  EXPECT_TRUE(std::filesystem::exists(root / "driver" / "GLCache" / "a" / "b.bin"));
  std::filesystem::remove_all(root);
}

TEST(DriverShaderCache, NearItsCapItIsEmptied)
{
  const std::filesystem::path root = freshCacheRoot("fv_driver_cache_near");
  writeBytes(root / "driver" / "GLCache" / "a" / "b.bin", 600);
  writeBytes(root / "driver" / "GLCache" / "a" / "c.bin", 350);
  {
    PipelineBuildRecord record;
    record.load(root / PipelineBuildRecord::kFileName);
    record.note(1, 20000.0);
  }
  const driver_shader_cache::Setup s = driver_shader_cache::prepare(root, 1000);
  EXPECT_EQ(s.emptied, driver_shader_cache::Emptied::eNearCap);
  EXPECT_EQ(s.bytes, 950u);
  EXPECT_TRUE(std::filesystem::is_directory(s.folder));
  EXPECT_EQ(driver_shader_cache::folderBytes(s.folder), 0u);
  PipelineBuildRecord after;
  after.load(root / PipelineBuildRecord::kFileName);
  EXPECT_FALSE(after.note(1, 20000.0)) << "built again into the emptied cache, it is no drop";
  std::filesystem::remove_all(root);
}

TEST(DriverShaderCache, AfterADropItIsEmptiedOnce)
{
  const std::filesystem::path root = freshCacheRoot("fv_driver_cache_drop");
  writeBytes(root / "driver" / "GLCache" / "a" / "b.bin", 100);
  const std::filesystem::path recordFile = root / PipelineBuildRecord::kFileName;
  {
    PipelineBuildRecord first;
    first.load(recordFile);
    first.note(1, 20000.0);
  }
  {
    PipelineBuildRecord second;
    second.load(recordFile);
    ASSERT_TRUE(second.note(1, 20000.0));
  }
  const driver_shader_cache::Setup s = driver_shader_cache::prepare(root, 1000);
  EXPECT_EQ(s.emptied, driver_shader_cache::Emptied::eDropped);
  EXPECT_EQ(driver_shader_cache::folderBytes(s.folder), 0u);
  // the start after it compiles into the empty cache; the one after that keeps what it holds
  {
    PipelineBuildRecord third;
    third.load(recordFile);
    EXPECT_FALSE(third.droppedOnRecord());
    EXPECT_FALSE(third.note(1, 20000.0));
  }
  writeBytes(root / "driver" / "GLCache" / "a" / "b.bin", 100);
  EXPECT_EQ(driver_shader_cache::prepare(root, 1000).emptied, driver_shader_cache::Emptied::eNo);
  EXPECT_TRUE(std::filesystem::exists(root / "driver" / "GLCache" / "a" / "b.bin"));
  std::filesystem::remove_all(root);
}

// The first start with a folder of its own: the record's keys were of the driver's shared cache, so the
// pipelines they name compile again into the new one, and that is no drop
TEST(DriverShaderCache, ANewFolderForgetsTheKeysOfTheOldCache)
{
  const std::filesystem::path root       = freshCacheRoot("fv_driver_cache_new");
  const std::filesystem::path recordFile = root / PipelineBuildRecord::kFileName;
  {
    PipelineBuildRecord before;
    before.load(recordFile);
    before.note(1, 20000.0);
  }
  EXPECT_EQ(driver_shader_cache::prepare(root, 1000).emptied, driver_shader_cache::Emptied::eNo);
  PipelineBuildRecord first;
  first.load(recordFile);
  EXPECT_EQ(first.lastCompiledMs(), 20000.0);
  EXPECT_FALSE(first.note(1, 20000.0)) << "compiled into the new folder, not dropped by it";
  // the folder there now, its keys stand
  driver_shader_cache::prepare(root, 1000);
  PipelineBuildRecord second;
  second.load(recordFile);
  EXPECT_TRUE(second.note(1, 20000.0));
  std::filesystem::remove_all(root);
}
