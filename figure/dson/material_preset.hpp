#pragma once

// A DAZ material preset (.duf `scene.animations` addressing `#materials/<surface>:?<channel>/value`)
// applied to a scene's materials: each surface of the name gets the preset's channels laid over its
// own and is parsed again. A port of the materials half of Wardrobe.applyPreset
// (src/lib/wardrobe/Wardrobe.ts). A scene has one figure here, so the preset's node chain is not
// used to pick a target: every surface of the name takes it.

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
  // surfaces the preset names that the scene doesn't have
  std::set<std::string> missing;
};

MaterialPresetResult applyMaterialPreset(const Json& doc, SceneGraph& scene);

}  // namespace fv::dson
