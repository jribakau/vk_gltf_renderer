#pragma once

// A geograft on a body of another topology (docs/wardrobe-plan.md, step 4d): the polygons it hides
// number the mesh it was made for, so they are carried over through the two bodies' maps. Each vertex
// of the source body takes its signed distance from the hidden region's edge (inside positive); the
// reference body's vertices take that where they land on the source, the figure's where they land on
// the reference; and a figure polygon is hidden when its corners' mean is inside: its middle lies in
// the region. Not grown past that: the graft's seam lies where the region's edge does, so a polygon
// more would open a hole beside it, where one fewer only tucks the skin under its edge.

#include <cstdint>
#include <span>
#include <vector>

namespace fv::bodymap {
struct TypeMap;
}
namespace fv::wardrobe {
struct BaseCage;
}

namespace fv::garment {

// The figure's polygons (its cage's loops, `polyStart` and `polyVerts`) under the source polygons
// `hidden`, which number `source` (its triPoly). `sourceMap` is the source body's map, `targetMap` the
// figure's. Empty when the source has no polygons to read or a map doesn't fit its body.
std::vector<int32_t> hiddenThrough(const wardrobe::BaseCage& source, std::span<const int32_t> hidden, const bodymap::TypeMap& sourceMap,
                                   const bodymap::TypeMap& targetMap, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts);

}  // namespace fv::garment
