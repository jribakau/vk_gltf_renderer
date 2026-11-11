// DLSS-RR's network presets as the settings number them (src/rr_presets.hpp): dlssPreset for the
// moving view and rrStillPreset for the still share one numbering, and the still defaults to E.

#include <gtest/gtest.h>

#include "../src/rr_presets.hpp"

TEST(RrPresets, ASettingsIndexIsItsNgxHint)
{
  // NVSDK_NGX_RayReconstruction_Hint_Render_Preset: Default 0, D 4, E 5, F 6 (dlss.cpp asserts the same)
  EXPECT_EQ(rr_presets::hint(0), 0u);
  EXPECT_EQ(rr_presets::hint(1), 4u);
  EXPECT_EQ(rr_presets::hint(2), 5u);
  EXPECT_EQ(rr_presets::hint(3), 6u);
  // past the table, its last (a settings file from a build with more)
  EXPECT_EQ(rr_presets::hint(9), 6u);
}

// F smooths away a refined still's resolved detail; the still's denoiser runs E unless told otherwise
TEST(RrPresets, TheStillRunsTheLatestTransformer)
{
  EXPECT_EQ(rr_presets::hint(rr_presets::kStillDefault), 5u);
}
