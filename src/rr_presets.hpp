#pragma once

// DLSS Ray Reconstruction's network presets as the settings number them: `dlssPreset` for the moving
// view, `rrStillPreset` for the still's denoiser (RrStillDenoiser). Each index stands for an NGX hint
// (NVSDK_NGX_RayReconstruction_Hint_Render_Preset); dlss.cpp checks these values against NGX's enum, so
// this header needs no NGX include and the tests can read it.
//
// The still's default is E, not F: on a refined still F (RR2, NGX's own default) smooths away detail a
// 2048-sample reference has (a fir's needles, grass blades), where E keeps it (docs/denoising.md, the
// still's model).

#include <cstdint>

namespace rr_presets {

inline constexpr uint32_t kHints[] = {
    0,  // Default: NGX chooses (F with the SDK of 2026-11)
    4,  // D: transformer
    5,  // E: latest transformer
    6,  // F: RR2
};
inline constexpr uint32_t kCount        = sizeof(kHints) / sizeof(kHints[0]);
inline constexpr uint32_t kStillDefault = 2;  // E

// The NGX hint of a setting's index; past the table, its last
inline constexpr uint32_t hint(uint32_t index)
{
  return kHints[index < kCount ? index : kCount - 1];
}

}  // namespace rr_presets
