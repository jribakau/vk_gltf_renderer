// What the path tracer throws away as not finite (shaders/invalid_samples.h.slang): a frame's block as
// the shader writes it, and the accumulation's tally and log line the host makes of the blocks
// (src/invalid_samples.hpp).

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "invalid_samples.hpp"

using namespace shaderio;

namespace {

// A frame's block as countInvalidSample in gltf_pathtrace.slang fills it
struct Block
{
  std::vector<uint32_t> words = std::vector<uint32_t>(INVALID_SAMPLES_WORDS, 0u);

  void count(uint32_t x, uint32_t y, uint32_t kind)
  {
    words[kind]++;
    const uint32_t at = words[INVALID_SAMPLES_OFFERED]++;
    if(at < INVALID_SAMPLES_KEPT)
      words[INVALID_SAMPLES_FIRST + at] = invalidSampleRecord(x, y, kind);
  }
};

}  // namespace

TEST(InvalidSamples, ARecordKeepsItsPixelAndKind)
{
  const uint32_t r = invalidSampleRecord(3839, 2159, INVALID_SAMPLE_STILL_GUIDE);
  EXPECT_EQ(invalidSampleX(r), 3839u);
  EXPECT_EQ(invalidSampleY(r), 2159u);
  EXPECT_EQ(invalidSampleKind(r), INVALID_SAMPLE_STILL_GUIDE);
  // the largest pixel the record holds
  const uint32_t edge = invalidSampleRecord(16383, 16383, INVALID_SAMPLE_COLOR);
  EXPECT_EQ(invalidSampleX(edge), 16383u);
  EXPECT_EQ(invalidSampleY(edge), 16383u);
  EXPECT_EQ(invalidSampleKind(edge), INVALID_SAMPLE_COLOR);
}

TEST(InvalidSamples, ACleanFrameTalliesNothing)
{
  invalid_samples::Tally tally;
  Block                  frame;
  tally.add(frame.words.data());
  EXPECT_TRUE(tally.empty());
  EXPECT_EQ(tally.frames, 1u);
  EXPECT_TRUE(tally.first.empty());
}

TEST(InvalidSamples, FramesSumByKindAndKeepTheFirstPixels)
{
  invalid_samples::Tally tally;
  Block                  a;
  a.count(10, 20, INVALID_SAMPLE_COLOR);
  a.count(11, 20, INVALID_SAMPLE_DLSS_GUIDE);
  Block b;
  b.count(400, 300, INVALID_SAMPLE_COLOR);
  tally.add(a.words.data());
  tally.add(b.words.data());
  EXPECT_EQ(tally.counts[INVALID_SAMPLE_COLOR], 2u);
  EXPECT_EQ(tally.counts[INVALID_SAMPLE_DLSS_GUIDE], 1u);
  EXPECT_EQ(tally.counts[INVALID_SAMPLE_STILL_GUIDE], 0u);
  EXPECT_EQ(tally.total(), 3u);
  ASSERT_EQ(tally.first.size(), 3u);
  EXPECT_EQ(tally.first[0].x, 10u);
  EXPECT_EQ(tally.first[1].kind, INVALID_SAMPLE_DLSS_GUIDE);
  EXPECT_EQ(tally.first[2].y, 300u);
}

TEST(InvalidSamples, AFrameWithMoreThanItKeepsStillCountsThemAll)
{
  invalid_samples::Tally tally;
  Block                  frame;
  for(uint32_t i = 0; i < INVALID_SAMPLES_KEPT + 50; i++)
    frame.count(i, 0, INVALID_SAMPLE_COLOR);
  tally.add(frame.words.data());
  EXPECT_EQ(tally.counts[INVALID_SAMPLE_COLOR], INVALID_SAMPLES_KEPT + 50);
  EXPECT_EQ(tally.first.size(), invalid_samples::Tally::kKept);
}

TEST(InvalidSamples, TheLogLineSaysHowManyAndWhere)
{
  invalid_samples::Tally tally;
  Block                  frame;
  frame.count(12, 7, INVALID_SAMPLE_COLOR);
  frame.count(13, 7, INVALID_SAMPLE_STILL_GUIDE);
  tally.add(frame.words.data());
  EXPECT_EQ(invalid_samples::describe(tally),
            "[invalid] 1 sample dropped, 0 DLSS guide pixels and 1 still guide pixel replaced over 1 frame; first at (12, 7) "
            "colour, (13, 7) still guide");
}
