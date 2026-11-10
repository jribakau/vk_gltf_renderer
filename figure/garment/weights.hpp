#pragma once

// A fitted garment's skin (docs/wardrobe-plan.md, step 5; rule 6): cloth lying on the skin takes the
// body's own weights where it lies, so it bends as the skin under it bends, at every joint the body has
// and with its dual quaternion share. Cloth hanging off the body keeps the weights it was made with,
// on its own bones, which follow the figure's; in between, a share of each.
//
// A chain's weights are transferred where the figure has a joint the garment's rig doesn't
// (transferChains). Not here yet, of the rule: a loose vertex kept to one side's limb.

#include <array>
#include <map>
#include <span>
#include <string>
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

// A joint of a skeleton as transferChains reads it: its name, its parent (an index into the same list,
// -1 for none) and its rest centre.
struct Joint
{
  std::string           name;
  int                   parent = -1;
  std::array<double, 3> center{};
};

// The rule's chain transfer: an item bone follows its leader on the figure (`follow`, item bone name
// -> figure bone name), and its child the child's; where the figure has joints between the two that
// no item bone follows (Skyrim's three spine bones on Genesis 9's four, its one neck on two), the
// item bone's cloth past such a joint's centre, along the chain from its leader to its child's, is
// weighted to that joint, over a band `band` cm either side. `itemBones` are the skin's bones, in its
// order; a figure joint j becomes bone `itemBones.size() + bodyBone[j]` of the result (the body's
// bones after the item's, as withBodyWeights numbers them), and one the body's skin hasn't (-1) is
// left out. `positions`: the cage, as the skeletons' centres are (at rest). `moved`, when given,
// counts the vertices that took any weight over.
mesh::SkinInput transferChains(const mesh::SkinInput& item, const std::vector<Joint>& itemBones, const std::map<std::string, std::string>& follow,
                               const std::vector<Joint>& figure, const std::vector<int>& bodyBone, std::span<const float> positions, double band = 1.5,
                               size_t* moved = nullptr);

}  // namespace fv::garment
