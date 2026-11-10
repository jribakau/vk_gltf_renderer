#pragma once

// A fitted garment's skin (docs/wardrobe-plan.md, step 5; rule 6): cloth lying on the skin takes the
// body's own weights where it lies, so it bends as the skin under it bends, at every joint the body has
// and with its dual quaternion share. Cloth hanging off the body keeps the weights it was made with,
// on its own bones, which follow the figure's; in between, a share of each.
//
// Not here yet, of the rule: a chain's weights transferred where the figure has a joint the garment's
// rig doesn't, and a loose vertex kept to one side's limb.

#include <span>
#include <vector>

#include <figure/mesh/mesh_build.hpp>
#include <figure/wardrobe/bind.hpp>

namespace fv::garment {

// How much of each cage vertex is the skin's under it, 0 to 1, by how far off the body it lies where
// `at` binds it (three body vertices and weights per vertex): all within `inner` cm, none beyond
// `outer`. A vertex bound to nothing holds nothing.
std::vector<float> holdByDistance(std::span<const float> item, std::span<const float> body, const wardrobe::SurfaceBinding& at, float inner = 2, float outer = 5);

// The garment's skin with the body's worked in. `item` is its skin as made, over `itemBones` bones of
// its own; `body` the body's, whose bone b becomes bone `itemBones + b` of the result; `at` each cage
// vertex's place on the body and `hold` its share of the body's weights there (the three corners'
// influences by their weights). The dual quaternion share goes over the same way. A bone's influences
// are summed, and those under a thousandth dropped.
mesh::SkinInput withBodyWeights(const mesh::SkinInput& item, size_t itemBones, const mesh::SkinInput& body, const wardrobe::SurfaceBinding& at,
                                std::span<const float> hold);

}  // namespace fv::garment
