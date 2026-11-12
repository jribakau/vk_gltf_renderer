#pragma once

// DAZ HD morph detail on a subdivided mesh: a .dhdm file's edits (figure/dson/dhdm.hpp), placed on
// a Refinement's vertices, and the reference that applies them.
//
// Each subdivision level of the cage may carry edits. An edit names a cage face, a face of that
// level among the face's descendants, and a corner of it, which is one vertex of that level; its
// offset is measured in the frame of the cage face corner the descendant lies in, and added once the
// level is refined, before the next level is made from it. Edits are linear, so several morphs'
// offsets add, weighted.
//
// The frame of cage face corner i (corners v0 .. vn-1): x along the edge to the corner before,
// normalize(v[i-1] - v[i]); n the face's normal, normalize(cross(v1 - v0, v[n-1] - v0)); y =
// normalize(cross(n, x)). An offset (a, b, c) moves the vertex by a x + b n - c y. The format is
// undocumented; the layout and this basis follow edolstra's reading of it
// (https://github.com/edolstra/dhdm, src/subdivide.cc), and the basis was checked on Genesis 9's FACS
// Details: it gives the smoothest field of the bases tried (tests/figure/test_hd_edits.cpp).

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <figure/subdiv/catmark.hpp>

namespace fv::subdiv {

// One edit as the file has it.
struct FaceEdit
{
  uint32_t face   = 0;  // cage face
  uint32_t sub    = 0;  // the face's descendant at the edit's level, in OpenSubdiv's order
  uint8_t  corner = 0;  // the descendant's corner: the vertex edited
  float    offset[3]{};
};

// A file's edits, level by level (levels[0] edits level 1).
using LevelEdits = std::vector<std::vector<FaceEdit>>;

// Edits placed on a refinement, for the levels it has.
struct HdEdits
{
  struct Level
  {
    std::vector<uint32_t> vertex;      // the level's vertex (vertices keep their index at the next level)
    std::vector<uint32_t> faceCorner;  // the cage face corner whose frame the offset is in (faceStart[f] + i)
    std::vector<float>    offset;      // three per edit, in that frame
  };
  std::vector<Level> levels;  // levels[k] edits level k + 1
  // edits at levels the refinement doesn't make, or naming a face or corner it doesn't have
  size_t dropped = 0;

  bool empty() const;
};

// Places `edits` on `r`, made from a cage whose faces start at `faceStart`.
HdEdits placeEdits(const Refinement& r, std::span<const int32_t> faceStart, const LevelEdits& edits);

// The frame of cage face corner `faceCorner`: columns x, n, -y as above, so frame * offset is the move.
std::array<float, 9> cornerFrame(std::span<const float> cagePositions, std::span<const int32_t> faceStart,
                                 std::span<const int32_t> faceVerts, std::span<const int32_t> cornerFace, uint32_t faceCorner);

// The face each corner belongs to (faceStart's inverse), for cornerFrame.
std::vector<int32_t> cornerFaces(std::span<const int32_t> faceStart);

// The reference: `cagePositions` refined to the finest level with the weighted edits added level by
// level, frames from those cage positions.
std::vector<float> refineWithEdits(const Refinement& r, std::span<const float> cagePositions, std::span<const int32_t> faceStart,
                                   std::span<const int32_t> faceVerts, std::span<const std::pair<float, const HdEdits*>> edits);

}  // namespace fv::subdiv
