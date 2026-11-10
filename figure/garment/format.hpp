#pragma once

// A garment as addresses on the reference body (docs/wardrobe-plan.md, step 4a; "The garment format"):
// where each vertex of its cage sits over the skin of the body it was made on, said in a way any mapped
// body can answer. The address is a place on the reference body, Genesis 9's base (three of its vertices
// and their weights); the offset is how far off the skin there the vertex lies along the skin's smooth
// normal, and what is left of it across the skin. Nothing in it names the body the garment came from or
// the one it goes onto.
//
// Not here yet, of the plan's format: the item's facts (rig family, kinds, cloth settings), weights on
// the spec's joints (step 5), strands, rigid pieces and grafts (4c-4e), and the copy kept on disk.

#include <vector>

#include <figure/wardrobe/bind.hpp>

namespace fv::garment {

struct Addresses
{
  // per cage vertex: three reference vertices and their weights (a SurfaceBinding on the reference, so
  // on Genesis 9 it is the binding the worn item follows the body by)
  wardrobe::SurfaceBinding at;
  // per cage vertex, cm off the skin along its smooth normal there; under the skin is negative
  std::vector<float> height;
  // xyz per cage vertex: the rest of its offset, across the skin, turned as the reference's base lies
  // where the address is (a map's frame turn)
  std::vector<float> across;
  // per cage vertex, 0 to 1: how much of it is its address's. 1 near the skin, fading with distance to
  // 0, where it has none: cloth that hangs free is carried by the cloth round it (settle)
  std::vector<float> hold;
};

}  // namespace fv::garment
