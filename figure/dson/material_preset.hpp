#pragma once

// A DAZ material preset (.duf `scene.animations` addressing `#materials/<surface>:?<channel>/value`)
// applied to a scene's materials: each surface of the name gets the preset's channels laid over its
// own and is parsed again. A port of the materials half of Wardrobe.applyPreset
// (src/lib/wardrobe/Wardrobe.ts). The URL's node ("Genesis9Eyelashes#materials/...", or a chain
// "name://@selection/Genesis9:/Genesis9Eyelashes:#materials/...", outermost first) narrows the
// surfaces to those whose owning node ends the longest part of the chain the scene has, ending at
// its last node; failing that, to those under the innermost node of the chain the scene has;
// otherwise every surface of the name takes it. A name the scene lacks falls back to the surfaces it was split into ("Eyelashes" for
// "Eyelashes Upper" and "Eyelashes Lower"), then to the name without a product prefix ("FE Base"
// for "Base"). A preset made for an item the scene lacks sets its values there but keeps the maps it
// would remove.
//
// A whole preset (kFullIrayPreset values or more on a surface) whose scene material names a shader
// of another family than the surface's (PBRSkin, Iray Uber, or a 3Delight-era one) starts the
// surface from that shader, as DAZ Studio replaces the material: the definition its url names
// (MaterialDefinitions), the entry's own channels, the old material's values and maps where the new
// shader has a channel of the same id, then the preset's values. The old shader's other channels go.
// A partial preset (a skin shine, a lip colour) is laid over the surface as it is.

#include <map>
#include <set>
#include <string>
#include <vector>

#include <figure/dson/channels.hpp>
#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dson {

// The shader definitions a preset's scene materials name in another file, by url as file#id
// (".../PBRSkin.dsf#PBRSkin"): their channels, as the character-load path lays them under a scene
// material. Iray Uber's is inside DAZ Studio, so it has none.
using MaterialDefinitions = std::map<std::string, Channels>;
// Those of `doc`, from the store; one that isn't installed is left out.
MaterialDefinitions definitionsIn(const Json& doc, Store& store);

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

// `only`: the materials that may take it (the rest of the preset is left out, and its surfaces count
// as missing when none of theirs is among them). `definitions`: definitionsIn(doc); without them a
// shader switch starts from the entry's own channels.
MaterialPresetResult applyMaterialPreset(const Json& doc, SceneGraph& scene, const std::set<int>* only = nullptr,
                                         const MaterialDefinitions& definitions = {});

// The scene materials a material preset would change, without changing them.
std::set<int> materialPresetTargets(const Json& doc, const SceneGraph& scene);

// The scene's channels as they are now become its materials' baseline: a wearable with the presets
// of the add-on it was worn as.
void keepBaseline(SceneGraph& scene);
// The materials `which` back to their baseline channels, parsed again; returns those that differed.
// Their edits stay.
std::set<int> restoreBaseline(SceneGraph& scene, const std::set<int>& which);

struct PresetChoice
{
  std::set<int>        changed;  // surfaces parsed again
  std::set<int>        owned;    // what this choice changed, for the next one
  MaterialPresetResult preset;
};

// A preset chosen for some surfaces (a pair of eyes, a worn item's colour) that replaces the last
// choice instead of stacking on it: `surfaces` and `previous` (what the last choice changed) back to
// their baseline, then the part of `doc` on `surfaces` over them; null `doc` leaves them as they
// came. Edits stay.
PresetChoice choosePreset(SceneGraph& scene, const std::set<int>& surfaces, const Json* doc, const std::set<int>& previous = {},
                          const MaterialDefinitions& definitions = {});

// Which of an item's installed Iray material presets it takes when worn, so a 3Delight-era item
// looks as its Iray version does (roadmap item 5); -1 for none. `scene` is the item alone. A
// candidate qualifies only when it keeps the item's look:
// - it is Iray's (it sets an Iray Uber or PBRSkin channel);
// - it is a whole material, at least kFullIrayPreset channels on every surface it reaches, not an
//   option (a glossiness, a wet look, a bump close-up);
// - it covers at least half the item's legacy surfaces;
// - any base colour or normal map it sets is the one the surface already has (the same colour of
//   Basic Wear Boxers in Iray, not another).
// Of those, the one that matches most maps, then sets most channels, then comes first.
constexpr int kFullIrayPreset = 40;
int           chooseIrayPreset(const SceneGraph& scene, const std::vector<const Json*>& candidates);

// Which scene materials are in use: those a node that wasn't taken off draws with, and those no node
// uses (the scene's own). Taken-off items' surfaces are not.
std::vector<bool> liveMaterials(const SceneGraph& scene);

// The scene materials a preset's (node chain, surface) lands on, by the rules above. An empty chain
// (`@selection` alone) means every surface of the name. `stray`: set when the scene lacks the item the
// chain ends at, so the surfaces are another item's; a preset there keeps the maps it would remove.
std::vector<int> presetTargets(const SceneGraph& scene, const std::vector<std::string>& chain, const std::string& surface, bool* stray = nullptr);
// The same for a single node.
std::vector<int> presetTargets(const SceneGraph& scene, const std::string& node, const std::string& surface);

}  // namespace fv::dson
