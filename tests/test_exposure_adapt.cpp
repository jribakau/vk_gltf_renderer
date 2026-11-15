// The exposure adapting to the view (src/exposure_adapt.hpp): the meter's range puts mid grey where the
// tone mapper's histogram counts it, a view is read from the middle half of its tones, a view the
// exposure suits moves nothing while a dark one opens it up by a share of how dark it is, and the
// adaptation eases at the same pace whatever the frame rate, faster into the light than into the dark.

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "../src/exposure_adapt.hpp"

namespace {

namespace A = exposure_adapt;

using Bins = std::array<uint32_t, A::kBins>;

// The bin the tone mapper puts a luminance in (nvshaders tonemapper.slang inputToHistogramBucket)
int binOf(float luminance, const A::Range& r)
{
  if(luminance < 0.00001F)
    return 0;
  const float ev = std::log2(luminance * (100.0F / 12.5F));
  const float n  = std::clamp((ev - r.evMin) / (r.evMax - r.evMin), 0.0F, 1.0F);
  return int(n * float(A::kBins - 2) + 1.0F);
}

// A view of `count` pixels at each of these luminances (after an exposure of `exposure`'s, raw)
Bins view(std::initializer_list<std::pair<float, uint32_t>> tones, float exposure)
{
  Bins       bins{};
  const auto r = A::meterRange(exposure);
  for(const auto& [luminance, count] : tones)
    bins[size_t(binOf(luminance, r))] += count;
  return bins;
}

}  // namespace

TEST(ExposureAdapt, MidGreyIsWhereTheMeterSaysItIs)
{
  for(const float exposure : {0.01F, 1.0F, 37.0F, 1.0e6F})
  {
    // a flat view at mid grey after the exposure reads 0 stops; four stops under, -4
    const float grey = A::kMidGrey / exposure;
    EXPECT_NEAR(*A::meteredStops(view({{grey, 1000}}, exposure)), 0.0F, 0.1F) << exposure;
    EXPECT_NEAR(*A::meteredStops(view({{grey / 16.0F, 1000}}, exposure)), -4.0F, 0.1F) << exposure;
    EXPECT_NEAR(*A::meteredStops(view({{grey * 8.0F, 1000}}, exposure)), 3.0F, 0.1F) << exposure;
  }
  // the range's ends: what is past them reads as them
  EXPECT_NEAR(A::binStops(1), -A::kBelow, 0.1F);
  EXPECT_FLOAT_EQ(A::binStops(A::kBins - 1), A::kAbove);
}

TEST(ExposureAdapt, AViewIsReadFromTheMiddleOfItsTones)
{
  const float grey = A::kMidGrey;
  // nothing lit: no reading, and black pixels are no part of one
  EXPECT_FALSE(A::meteredStops(Bins{}).has_value());
  EXPECT_FALSE(A::meteredStops(view({{0.0F, 5000}}, 1.0F)).has_value());
  EXPECT_NEAR(*A::meteredStops(view({{0.0F, 5000}, {grey / 4.0F, 100}}, 1.0F)), -2.0F, 0.1F);
  // a barn: most of it eight stops under, a fifth of it sunlit patches and sky two stops over. The
  // patches are past the middle half: the reading is the dark
  EXPECT_NEAR(*A::meteredStops(view({{grey / 256.0F, 800}, {grey * 4.0F, 200}}, 1.0F)), -8.0F, 0.1F);
  // half and half: the middle half is a quarter of each, read between them
  EXPECT_NEAR(*A::meteredStops(view({{grey / 256.0F, 500}, {grey * 4.0F, 500}}, 1.0F)), -3.0F, 0.1F);
}

TEST(ExposureAdapt, AViewTheExposureSuitsMovesNothingADarkOneOpensItUp)
{
  // within the knee nothing; past it, smoothly, a share of each stop
  EXPECT_FLOAT_EQ(A::adaptationFor(0.0F), 0.0F);
  EXPECT_FLOAT_EQ(A::adaptationFor(-0.5F), 0.0F);
  EXPECT_FLOAT_EQ(A::adaptationFor(0.5F), 0.0F);
  EXPECT_NEAR(A::adaptationFor(-4.0F), A::kShare * 3.0F, 1e-5F);
  EXPECT_NEAR(A::adaptationFor(2.5F), -A::kShare * 1.5F, 1e-5F);
  // a dark view opened up, a bright one closed down: never past mid grey, never past the most
  for(float stops = -16.0F; stops <= 8.0F; stops += 0.05F)
  {
    const float a = A::adaptationFor(stops);
    EXPECT_GE(a * -stops, 0.0F) << stops;
    EXPECT_LE(std::abs(a), std::abs(stops) + 1e-5F) << stops;
    EXPECT_LE(a, A::kMostUp);
    EXPECT_GE(a, -A::kMostDown);
    // the darker the view, the more it opens, without a step (the bend's slope is at most kShare)
    EXPECT_GE(A::adaptationFor(stops - 0.05F), a - 1e-5F) << stops;
    EXPECT_LE(std::abs(A::adaptationFor(stops - 0.05F) - a), A::kShare * 0.05F + 1e-4F) << stops;
  }
  EXPECT_FLOAT_EQ(A::adaptationFor(-16.0F), A::kMostUp);
  EXPECT_FLOAT_EQ(A::adaptationFor(8.0F), -A::kMostDown);
  // the barn of 2026-11-15, nine stops under mid grey, is shown about two and a half under
  EXPECT_NEAR(-9.0F + A::adaptationFor(-9.0F), -2.6F, 0.05F);
}

TEST(ExposureAdapt, ItEasesAtTheSamePaceAtAnyFrameRateFasterIntoTheLight)
{
  // a second at 10, 60 or 240 frames a second ends in the same place
  for(const int fps : {10, 60, 240})
  {
    float stops = 0.0F;
    for(int i = 0; i < fps; i++)
      stops = A::eased(stops, 6.0F, 1.0F / float(fps));
    EXPECT_NEAR(stops, 6.0F * (1.0F - std::exp(-1.0F / A::kOpenSeconds)), 1e-3F) << fps;
  }
  // opening up takes longer than closing down by as much
  const float up   = A::eased(0.0F, 4.0F, 0.3F);
  const float down = 4.0F - A::eased(4.0F, 0.0F, 0.3F);
  EXPECT_LT(up, down);
  EXPECT_GT(up, 0.0F);
  // there, it stays; no time, no move
  EXPECT_FLOAT_EQ(A::eased(3.0F, 3.0F, 0.5F), 3.0F);
  EXPECT_FLOAT_EQ(A::eased(1.0F, 3.0F, 0.0F), 1.0F);
  EXPECT_FLOAT_EQ(A::eased(1.0F, 3.0F, -1.0F), 1.0F);
}
