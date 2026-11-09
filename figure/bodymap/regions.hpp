#pragma once

// A body's regions, which a body map matches within (docs/wardrobe-plan.md, step 3). Each vertex's
// come from its own skin weights: its strongest bone's region by the skeleton spec (a bone the spec
// doesn't know takes its nearest known parent's, so the face's bones are the head's), and on a border
// any other bone's that holds a good share of it, so a vertex there may match either side of the
// border. Left and right are regions of their own, and so is each finger and toe. Also which
// surfaces are outer skin: a map addresses only those, not the eyes, lashes, mouth, teeth or tongue.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <figure/mesh/mesh_build.hpp>
#include <figure/rig/spec.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace fv::bodymap {

// A set of regions, a bit each.
using RegionMask = uint64_t;

// A region's bit: the spec region on that side ('l', 'r', or 0 for a region with no side). None for
// rig::Region::eNone.
RegionMask regionBit(rig::Region region, char side = 0);

// Each vertex's regions from `skin` (`bone` indexes `bones`, which name their parents): its strongest
// bone's region, and that of any other bone holding at least `border` of its weight. A vertex no bone
// with a region holds has none.
std::vector<RegionMask> vertexRegions(const mesh::SkinInput& skin, const std::vector<wardrobe::SkelBone>& bones, size_t vertices, float border = 0.25f);

// The regions a mask holds, by name, the right side marked: "thigh pelvis", "r thigh", "none".
std::string regionNames(RegionMask mask);

// Whether a surface (a material group: "Torso", "Lips", "Teeth") is outer skin.
bool outerSkin(std::string_view surface);

}  // namespace fv::bodymap
