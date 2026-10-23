#pragma once

// Genesis 9 figures leave their eyes, mouth, eyelashes and tear to a post-load script: DAZ Studio's
// Character Addon Loader reads a list from the scene file, loads each add-on onto the figure and
// applies its material preset. A port of src/lib/loaders/dson/addons.ts.

#include <string>
#include <vector>

#include <figure/dson/store.hpp>

namespace fv::dson {

struct PostLoadAddon
{
  // the add-on's figure node, e.g. Genesis9Eyes: a figure that already has it doesn't load it again
  std::string name;
  // library-relative path of the .duf that loads it
  std::string file;
  // library-relative paths of the presets applied once it's on
  std::vector<std::string> presets;
};

// The add-ons a scene file's Character Addon Loader script lists, in its order.
std::vector<PostLoadAddon> postLoadAddons(const Json& doc);

}  // namespace fv::dson
