#pragma once

// A garment's hard pieces (docs/wardrobe-plan.md, step 7): buttons, buckles, studs, plates. A fit moves
// every vertex by itself, which suits cloth and bends a buckle; DAZ marks such parts with rigidity
// groups for its morphs only, and a Skyrim mesh marks nothing. A piece is found by what it is: cloth
// joined to nothing else that is small, or whose every vertex has the same weights (a plate on one
// bone). It is put back in the shape it was made in, moved and turned as a whole to where the fit put
// it. The smallest, and those on one set of weights, are hard for a pose too: their vertices share one
// set of weights, so it carries them without bending them. A part a few centimetres long may as well be
// a strip of lace, which should bend with the skin under it (nearly half of one Skyrim dress is such
// parts), so it keeps its own weights.
//
// Not found this way: a plate joined to the cloth under it, or weighted across two bones.

#include <span>
#include <vector>

#include <figure/mesh/mesh_build.hpp>

namespace fv::garment {

struct PieceOptions
{
  float small   = 8;      // cm: a part no longer than this corner to corner is a piece
  float stud    = 4;      // cm: one no longer than this is hard for a pose too
  float plate   = 45;     // cm: a uniformly weighted part up to this long is one (a skirt all on the hip isn't)
  float sameWithin = 0.02f;  // weights this close on every vertex are the same weights
  int   least   = 4;      // vertices: fewer is a sliver, left alone
};

struct Pieces
{
  std::vector<int32_t> of;  // per cage vertex: its piece, or -1
  int                  count = 0;
  // per piece: a pose carries it whole (PieceOptions::stud, or one set of weights)
  std::vector<uint8_t> hard;
};

// The hard pieces of a cage: its connected parts (by `tris`) that are small, or uniformly weighted.
// The largest part of a cage with several is never one by its size alone (a small garment's body).
Pieces hardPieces(std::span<const float> positions, std::span<const uint32_t> tris, const mesh::SkinInput& skin, const PieceOptions& options = {});

// `fitted` with each piece back in its made shape: `made`'s, turned and moved as a whole the way that
// best matches where the fit put its vertices (Horn's fit), and sized as a whole by how much the fit
// sized it, within 0.8 to 1.25.
std::vector<float> keepPieces(std::span<const float> made, std::span<const float> fitted, const Pieces& pieces);

// `skin` with each hard piece's vertices on one set of weights, the mean of theirs (and of their dual
// quaternion shares).
mesh::SkinInput rigidSkin(const mesh::SkinInput& skin, const Pieces& pieces);

}  // namespace fv::garment
