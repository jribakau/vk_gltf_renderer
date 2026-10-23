#pragma once

// A stand-in for the smoothing and collision modifiers the loader does not read: a garment vertex
// that ends just under the skin is moved out along the skin's normal. A port of
// src/lib/wardrobe/fit/collide.ts.

#include <span>
#include <vector>

#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace fv::wardrobe {

// Pushes `positions` out of `body` to `margin` centimetres. Only a vertex within `depth` of the
// skin, and behind the face it is closest to rather than off to one side of its edge, counts as
// inside: a heel's stem below the sole is outside, though it lies behind the sole's plane. The push
// of one vertex is capped, so a bad binding cannot throw it across the figure.
std::vector<float> pushOut(std::span<const float> positions, const Surface& body, const SurfaceBinding& binding, float margin, float depth = 3);

// Two Laplacian steps on the correction (`next` - `prev`), leaving the garment's cut in place.
std::vector<float> smoothCorrection(std::span<const float> prev, std::span<const float> next, std::span<const uint32_t> tris);

}  // namespace fv::wardrobe
