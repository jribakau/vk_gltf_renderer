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

// The region a region hangs from, by the spec's joints: the forearm's is the upper arm, a toe's the
// foot, the breast's the chest; none for the pelvis, the root.
rig::Region regionAbove(rig::Region region);
// The regions a region runs on into, by the spec's joints: the one its joint aims down the limb at
// (the upper arm's forearm, the hand's middle finger, the chest's neck), else every region hanging
// from it (the foot's toes); none at a limb's end, or for the pelvis, the root.
std::vector<rig::Region> regionsBelow(rig::Region region);
// Whether a region comes as a left and a right one (its joints have sides).
bool regionSided(rig::Region region);
// Whether the band where a region meets the one above (its joint: the forearm's is the elbow) is a
// landmark two bodies share, by the spec's flag on the region's joint.
rig::Landmark regionLandmark(rig::Region region);
// A mask with the regions next to each it holds: the one above and those below, on its side. The left
// thigh's are the pelvis and the left shin, never the right thigh.
RegionMask withNeighbours(RegionMask mask);

// The regions a mask holds, by name, the right side marked: "thigh pelvis", "r thigh", "none".
std::string regionNames(RegionMask mask);

// Whether a surface (a material group: "Torso", "Lips", "Teeth") is outer skin.
bool outerSkin(std::string_view surface);

}  // namespace fv::bodymap
