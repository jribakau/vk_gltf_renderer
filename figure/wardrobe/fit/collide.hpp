#pragma once

// A stand-in for the smoothing and collision modifiers the loader does not read: a garment vertex
// that ends just under the skin is moved out along the skin's normal. A port of
// src/lib/wardrobe/fit/collide.ts.

#include <span>
#include <vector>

#include <figure/wardrobe/bind.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace fv::wardrobe {

// The unit normal of a triangle of `p` (xyz per vertex); up when it is degenerate.
V3 triNormal(std::span<const float> p, size_t a, size_t b, size_t c);

// Pushes `positions` out of `body` to `margin` centimetres. Only a vertex within `depth` of the
// skin, and behind the face it is closest to rather than off to one side of its edge, counts as
// inside: a heel's stem below the sole is outside, though it lies behind the sole's plane. The push
// of one vertex is capped, so a bad binding cannot throw it across the figure.
std::vector<float> pushOut(std::span<const float> positions, const Surface& body, const SurfaceBinding& binding, float margin, float depth = 3);

// Pulls `positions` toward `margin` centimetres over `body`, by `amount` of the way (0 to 1): a garment
// made skin tight. Only what lies within `reach` of the skin, fading from half of it, so a hem hanging
// free stays free; and only what is over a face rather than off past its edge, as in pushOut.
std::vector<float> pullIn(std::span<const float> positions, const Surface& body, const SurfaceBinding& binding, float margin, float amount, float reach = 6);

// The polygons of a body (`polyStart`, `polyVerts` over `body`) that a garment covers: each corner lies
// behind the garment's face, within `reach` of it, and at least `hemClear` along the garment from any
// open edge, so a pose that slides the fabric doesn't bare what isn't drawn. Skin that isn't drawn
// can't show through the fabric, however the two meet, so this is the answer to what the collider's
// clearance can't promise at a sharp joint.
std::vector<int32_t> coveredPolys(std::span<const float>    body,
                                  std::span<const int32_t>  polyStart,
                                  std::span<const int32_t>  polyVerts,
                                  std::span<const float>    item,
                                  std::span<const uint32_t> itemTris,
                                  float                     reach,
                                  float                     hemClear);

// Laplacian steps on the correction (`next` - `prev`), leaving the garment's cut in place.
std::vector<float> smoothCorrection(std::span<const float> prev, std::span<const float> next, std::span<const uint32_t> tris, int passes = 2);

}  // namespace fv::wardrobe
