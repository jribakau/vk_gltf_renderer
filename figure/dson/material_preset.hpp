#pragma once

// A DAZ material preset (.duf `scene.animations` addressing `#materials/<surface>:?<channel>/value`)
// applied to a scene's materials: each surface of the name gets the preset's channels laid over its
// own and is parsed again. A port of the materials half of Wardrobe.applyPreset
// (src/lib/wardrobe/Wardrobe.ts). The URL's node ("Genesis9Eyelashes#materials/...") narrows the
// surfaces to those under that node when the scene has it; otherwise every surface of the name
// takes it. A name the scene lacks falls back to the surfaces it was split into ("Eyelashes" for
// "Eyelashes Upper" and "Eyelashes Lower"), then to the name without a product prefix ("FE Base"
// for "Base").

#include <set>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dson {

struct MaterialPresetResult
{
  // scene materials whose surface changed
  std::set<int> changed;
  // channel values and images the preset set
  int edits = 0;
  // surfaces the preset names that the scene doesn't have, and those it found
  std::set<std::string> missing;
  std::set<std::string> matched;
};

MaterialPresetResult applyMaterialPreset(const Json& doc, SceneGraph& scene);

// The scene materials a preset's (node, surface) lands on, by the rules above.
std::vector<int> presetTargets(const SceneGraph& scene, const std::string& node, const std::string& surface);

}  // namespace fv::dson
