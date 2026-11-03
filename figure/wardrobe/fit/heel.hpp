#pragma once

// The foot has to adopt the shoe. A heel's file poses the figure's feet (DAZ applies its scene
// animations to the figure it is put on): foot and toe bones turned, or on Genesis 9 the figure's
// own foot dials set (body_ctrl_lFootHeeledShoe, whose formulas point the foot and bend the toes
// back). The bones go through the cross-generation alias. A port of src/lib/wardrobe/fit/heel.ts.

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

// Those channels addressed to the figure's bones (a dial keeps its name). Empty when every value is rest.
std::vector<FootChannel> retargetFootPose(const std::vector<FootChannel>& channels, const std::vector<std::string>& targets);

}  // namespace fv::wardrobe
