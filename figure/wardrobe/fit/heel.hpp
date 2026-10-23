#pragma once

// The foot has to adopt the shoe. A heel's file poses the figure's feet (DAZ applies its scene
// animations to the figure it is put on); the bones go through the cross-generation alias. A port
// of src/lib/wardrobe/fit/heel.ts.

#include <string>
#include <vector>

#include <figure/dson/store.hpp>

namespace fv::wardrobe {

struct FootChannel
{
  std::string bone, axis;
  double      value = 0;
};

// Foot and toe rotations named by a wearable's scene animations.
std::vector<FootChannel> footChannels(const dson::Json& doc);

// Those channels addressed to the figure's bones. Empty when every value is rest.
std::vector<FootChannel> retargetFootPose(const std::vector<FootChannel>& channels, const std::vector<std::string>& targets);

}  // namespace fv::wardrobe
