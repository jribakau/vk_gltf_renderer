#pragma once

// Strand hair through the common layer (docs/wardrobe-plan.md, step 4c): each root goes where its
// address puts it on the figure's skin, at the height it had over its own body's, and its strand is
// carried with it, the move fading out along the strand. The rest of the strand keeps the fit it had
// (the bone transfer and the shape correction, which already keep strands whole), so only the roots,
// which that fit leaves on the push's 1.5 mm off the scalp, are moved.

#include <cstddef>
#include <span>
#include <vector>

namespace fv::bodymap {
struct TypeMap;
}
namespace fv::garment {
struct Target;
}

namespace fv::garment {

struct CarriedRoots
{
  std::vector<float> positions;  // the strands, xyz per cage vertex
  size_t             moved = 0;  // roots placed by their addresses
};

// `made`: the strands as made, over the source body (`sourcePositions`, `sourceTris`, the body
// `sourceMap` maps); `fitted`: as the fit left them; lines as mesh::BuiltMesh has them (cageLineStart,
// cageLineVerts). `onto` is what decode reads, and `figure` the figure's cage, whose vertices its places
// name. A root its address holds whole (within 2 cm of the source's skin) is moved; its strand's points
// follow by how far along it they lie, all of it at the root and none past `fade` cm.
CarriedRoots carryRoots(std::span<const float> made, std::span<const float> fitted, std::span<const int32_t> lineStart, std::span<const int32_t> lineVerts,
                        std::span<const float> sourcePositions, std::span<const uint32_t> sourceTris, const bodymap::TypeMap& sourceMap, const Target& onto,
                        std::span<const float> figure, float fade = 2);

}  // namespace fv::garment
