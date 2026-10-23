#pragma once

// What part of the body a worn mesh sits on, and which limb each vertex belongs to, so a binding
// keeps a left shoe off the right foot and a sleeve off the chest. A port of
// src/lib/wardrobe/fit/regions.ts.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <figure/mesh/mesh_build.hpp>
#include <figure/wardrobe/bind.hpp>

namespace fv::wardrobe {

enum class FitRegion
{
  eBody,
  eShoe,
  eBoot,
  eBootThigh,
  eGlove,
  eHead,
};

// Limb ids, small enough to pack into a bitmask. 0 means the vertex has no limb.
namespace limb {
constexpr int16_t kHead = 1, kLArm = 3, kRArm = 4, kLThigh = 5, kRThigh = 6, kLShin = 7, kRShin = 8, kLFoot = 9, kRFoot = 10, kLHand = 11, kRHand = 12;
}

// The limb a bone belongs to, or 0 when it is not a limb we fit against.
int16_t limbOfBone(const std::string& name);

// The region a wearable occupies, from its path and, failing that, the bones it skins to.
FitRegion regionOf(const std::string& path, const std::vector<std::string>& boneNames);

// Per cage vertex: the limb of its strongest bone, or 0. `boneLimb` is per bone of the skin.
std::vector<int16_t> vertexLimbs(size_t count, const mesh::SkinInput* skin, const std::vector<int16_t>& boneLimb);

// Binds a vertex with a limb only to faces of that limb (or of no limb). None when the body has none.
std::optional<BindFilter> limbFilter(const std::vector<int16_t>& bodyLimbs, const std::vector<int16_t>& itemLimbs);

}  // namespace fv::wardrobe
