#pragma once

// The foot has to adopt the shoe. A heel's file poses the figure's feet (DAZ applies its scene
// animations to the figure it is put on): foot and toe bones turned, or on Genesis 9 the figure's
// own foot dials set (body_ctrl_lFootHeeledShoe, whose formulas point the foot and bend the toes
// back). The bones go through the cross-generation alias. A port of src/lib/wardrobe/fit/heel.ts.

#include <array>
#include <map>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>

namespace fv::wardrobe {

struct FootChannel
{
  std::string bone, axis;  // a bone's rotation about an axis, in degrees
  double      value = 0;
  std::string dial;  // or, when set, the figure's dial of that name, at `value`
};

// Foot and toe rotations, and the figure's foot and toe dials, a wearable's scene animations set.
std::vector<FootChannel> footChannels(const dson::Json& doc);

// A bone's frame as the pose reads it (rig::boneTurn): its orientation's angles in degrees, its rotation
// order and its rotation limits.
struct BoneFrame
{
  std::array<double, 3> orientation{};
  std::string           order = "XYZ";
  // the limits it holds its angles to (none, unclamped): a file's angles past them were never the pose
  std::array<double, 3> min{-kNoLimit, -kNoLimit, -kNoLimit}, max{kNoLimit, kNoLimit, kNoLimit};
  static constexpr double kNoLimit = 1e300;
};

// Those channels addressed to the figure's bones (a dial keeps its name). Empty when every value is rest.
// A bone's channels go over as the turn they give it at rest, from the frame of the bone the file was
// made on (`from`, by the file's bone name: the item's own skeleton) to the frame of the figure's bone
// (`to`, by its name), and the figure's bone takes all three of its angles: G8F's and G9's feet differ
// by about 8 degrees about z, so a pitch copied axis by axis would also roll the foot. A bone either map
// lacks, or whose frames agree, is copied axis by axis, as a heel on the figure it was made for is.
std::vector<FootChannel> retargetFootPose(const std::vector<FootChannel>& channels, const std::vector<std::string>& targets,
                                          const std::map<std::string, BoneFrame>& from = {}, const std::map<std::string, BoneFrame>& to = {});

}  // namespace fv::wardrobe
