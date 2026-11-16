// The exposure adapting to the view (src/exposure_adapt.hpp): the meter's range puts mid grey where the
// tone mapper's histogram counts it, a view is read from the middle half of its tones, a view the
// exposure suits moves nothing while a dark one opens it up by a share of how dark it is, and the
// adaptation eases at the same pace whatever the frame rate, faster into the light than into the dark.
// Play mode's eye: a moonlit night seen as a night, a moonless one dark, a torch's pool by its light
// whatever the dark round it, a day as it is.

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "../src/exposure_adapt.hpp"

namespace {

namespace A = exposure_adapt;

using Bins = std::array<uint32_t, A::kBins>;

// The bin the tone mapper puts a luminance in (nvshaders tonemapper.slang inputToHistogramBucket):
// black under 1e-5 and under the range's bottom
int binOf(float luminance, const A::Range& r)
{
  const float ev = luminance > 0.0F ? std::log2(luminance * (100.0F / 12.5F)) : -1e30F;
  if(luminance <= 0.0F || (luminance < 0.00001F && ev < r.evMin))
    return 0;
  const float n  = std::clamp((ev - r.evMin) / (r.evMax - r.evMin), 0.0F, 1.0F);
  return int(n * float(A::kBins - 2) + 1.0F);
}

// A view of `count` pixels at each of these luminances (after an exposure of `exposure`'s, raw)
Bins view(std::initializer_list<std::pair<float, uint32_t>> tones, float exposure, const A::Tuning& t = A::kCamera)
{
  Bins       bins{};
  const auto r = A::meterRange(exposure, t);
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
  EXPECT_NEAR(A::binStops(1), -A::kCamera.below, 0.1F);
  EXPECT_FLOAT_EQ(A::binStops(A::kBins - 1), A::kCamera.above);
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
  EXPECT_NEAR(A::adaptationFor(-4.0F), A::kCamera.share * 3.0F, 1e-5F);
  EXPECT_NEAR(A::adaptationFor(2.5F), -A::kCamera.share * 1.5F, 1e-5F);
  // a dark view opened up, a bright one closed down: never past mid grey, never past the most
  for(float stops = -16.0F; stops <= 8.0F; stops += 0.05F)
  {
    const float a = A::adaptationFor(stops);
    EXPECT_GE(a * -stops, 0.0F) << stops;
    EXPECT_LE(std::abs(a), std::abs(stops) + 1e-5F) << stops;
    EXPECT_LE(a, A::kCamera.mostUp);
    EXPECT_GE(a, -A::kCamera.mostDown);
    // the darker the view, the more it opens, without a step (the bend's slope is at most kShare)
    EXPECT_GE(A::adaptationFor(stops - 0.05F), a - 1e-5F) << stops;
    EXPECT_LE(std::abs(A::adaptationFor(stops - 0.05F) - a), A::kCamera.share * 0.05F + 1e-4F) << stops;
  }
  EXPECT_FLOAT_EQ(A::adaptationFor(-16.0F), A::kCamera.mostUp);
  EXPECT_FLOAT_EQ(A::adaptationFor(8.0F), -A::kCamera.mostDown);
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
    EXPECT_NEAR(stops, 6.0F * (1.0F - std::exp(-1.0F / A::kCamera.openSeconds)), 1e-3F) << fps;
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

// Play mode's eye, at a day's exposure (the user's 0.65): the stops it opens (closed: negative) for a
// view of `tones` (stops from mid grey, and how many pixels the meter counts at each) once it has
// taken to it
float eyeFor(std::initializer_list<std::pair<float, uint32_t>> tones, float exposure = 0.65F)
{
  const float grey = A::kMidGrey / exposure;
  Bins        bins{};
  const auto  r = A::meterRange(exposure, A::kEye);
  for(const auto& [stops, count] : tones)
    bins[size_t(binOf(grey * std::exp2(stops), r))] += count;
  const std::optional<float> reading = A::meteredStops(bins, A::kEye);
  return reading ? A::adaptationFor(*reading, A::kEye) : 0.0F;
}

TEST(ExposureAdapt, AnEyeSeesAMoonlitNightAsANightAndATorchsPoolByItsLight)
{
  // a moonlit field, some 20 stops under the day (0.3 lux against 1e5), its grass a stop either way:
  // seen, as a night is, two to four stops under mid grey. The camera's meter doesn't reach it.
  const float night = -20.0F + eyeFor({{-21.0F, 300}, {-20.0F, 400}, {-19.0F, 300}});
  EXPECT_GT(night, -4.0F);
  EXPECT_LT(night, -2.0F);
  EXPECT_FALSE(A::meteredStops(view({{A::kMidGrey / 0.65F * std::exp2(-20.0F), 1000}}, 0.65F)).has_value());
  // a moonless one (29 stops, the sky's glow) darker still
  EXPECT_LT(-29.0F + eyeFor({{-30.0F, 300}, {-29.0F, 400}, {-28.0F, 300}}), night - 4.0F);
  // the torch's pool, 8 stops under the day a few metres off and brighter at its middle, over half the
  // spot the eye reads: taken to wholly, within a stop of mid grey, whatever the moon does round it,
  // which goes dark beside it
  for(const float around : {-20.0F, -29.0F})
  {
    const float open = eyeFor({{around, 450}, {-9.0F, 200}, {-8.0F, 250}, {-7.0F, 100}});
    EXPECT_GT(-8.0F + open, -1.6F) << around;
    EXPECT_LT(-8.0F + open, 0.5F) << around;
    EXPECT_LT(around + open, night - 6.0F) << around;
  }
  // and over a third of it, the rest moonlit: brighter than mid grey, a few stops at most
  EXPECT_LT(-8.0F + eyeFor({{-20.0F, 650}, {-8.0F, 350}}), 3.5F);
  // a day is as the exposure has it: a view within a stop of mid grey moves nothing, one with a bright
  // sky in the spot a little
  EXPECT_FLOAT_EQ(eyeFor({{-1.5F, 300}, {-0.5F, 400}, {0.5F, 300}}), 0.0F);
  const float sky = eyeFor({{-1.0F, 500}, {1.5F, 500}});
  EXPECT_LT(sky, 0.0F);
  EXPECT_GT(sky, -1.0F);
  // the eye takes as long to open into the dark as a game's does, and closes faster
  EXPECT_GT(A::kEye.openSeconds, A::kEye.closeSeconds);
}
