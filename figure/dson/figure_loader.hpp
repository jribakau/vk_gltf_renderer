#pragma once

// Native DAZ loader: a .duf scene/character file (or a bare .dsf asset) from the content library to
// a SceneGraph with UVs, smooth seam-free normals, skeletons, skin weights and materials. A port of
// src/lib/loaders/dson/figure.ts, so far the parts that build what a scene shows at rest: nodes,
// worn items and their skeletons, geometry (subdivided when SubD), skin and surfaces. Morphs,
// fitting and dForce join it as they are ported.
//
// The scene is in centimetres, as DSON is; the host scales it to metres.

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <figure/dson/morphs.hpp>
#include <figure/dson/rig.hpp>
#include <figure/dson/modifiers.hpp>
#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/wardrobe/fit/heel.hpp>
#include <figure/wardrobe/fit/regions.hpp>
#include <figure/wardrobe/follow.hpp>

namespace fv::wardrobe {
struct BaseCage;
}

namespace fv::dson {

// A worn item: one wearable's nodes, which go on and come off together.
struct Item
{
  std::string key, label;
  // top-level objects, each with the figure bone it hangs from ("" for the figure itself)
  std::vector<std::pair<int, std::string>> objects;
  // DSON names of its nodes, which hierarchical presets address
  std::set<std::string> nodeNames;
  // "file#id" of the node definitions it was built from
  std::set<std::string> sources;
  // a URL in the file that names the figure it was authored on, and the foot poses it sets (a heel)
  std::string                          conformTarget;
  std::vector<wardrobe::FootChannel>   footPose;
  // the figure it says it was made for: its figure node's preferred base ("/Genesis 8/Female"), or
  // Skyrim for a .nif. Its bones can overrule it, its path can't (wardrobe::identifyGeneration)
  std::optional<wardrobe::Generation> madeFor;
  // what the file says it is, over what its path and bones suggest: a .nif's biped slots
  std::optional<wardrobe::FitRegion> region;
  // its skeletons (indices into LoadResult::rigs), which follow the figure's pose
  std::vector<int> rigs;
  // its meshes (indices into SceneGraph::meshes)
  std::vector<int> meshes;
};

// Loading a wearable (clothing, hair, a prop) for a figure that is already loaded: its figure nodes
// are items that follow that figure's skeleton, bound at its joint centres. The result is a scene
// of its own, which Figure::wear merges into the figure's.
struct AttachTarget
{
  const SceneGraph* scene = nullptr;  // the figure's scene, for its joint centres
  const Rig*        rig   = nullptr;  // the figure's skeleton
  std::string       key;              // names the items ("w1", "w1.1")
};

struct LoadOptions
{
  // SubD geometry drawn subdivided this many times (0-3)
  int subdivision = 0;
  // the share of a strand hair's strands drawn (mesh::Lines::keep), each wider by as much, so that
  // the hair covers what it did
  float strandKeep = 1;
  // the pieces each of its segments is drawn in (mesh::Lines::pieces)
  int strandPieces = 2;
  std::optional<AttachTarget> attach;
  // morphs move only the cages of the meshes whose drawn vertices are made from them
  // (mesh::drawnFromCage): the GPU makes those
  bool cageMorphs = false;
};

// A morph dial a scene file sets, and the node or geometry instance it sits on: a character's shape
// dials sit on the figure, a prop's on that one copy of it.
struct SceneDial
{
  // the modifier: library file and id
  std::string file, id;
  std::string owner;
  double      value = 0;
};

// What the morph engine (figure/figure.hpp) needs from a load.
struct MorphContext
{
  // the meshes morphs can target
  std::vector<MeshRef> meshes;
  // the main figure's meshes (indices into `meshes`)
  std::vector<size_t> mainMeshes;
  // the scene node of the main figure; "" without one
  std::string mainNode;
  // the figure's own node and meshes' owner ids: dials on these are the main figure's
  std::set<std::string> mainOwners;
  // every non-zero morph dial the scene sets
  std::vector<SceneDial> dials;
  // scene node or geometry instance id -> the worn item it belongs to (index into LoadResult::items)
  std::map<std::string, int> itemOf;
  // "<figure folder>/Morphs", library-relative; empty without a figure
  std::string morphDir;
  // the scene recorded where its shape put the main figure's joints
  bool previewCenters = false;
};

// One mesh of a worn item, for fitting it to the body.
struct ItemPart
{
  int item = -1;
  int mesh = -1;  // into SceneGraph::meshes
  // skinned to a skeleton of its own: it conforms, so it follows the body's shape too
  bool                           conforms = false;
  std::optional<wardrobe::Graft>    graft;
  std::optional<wardrobe::Rigidity> rigidity;
  // DAZ Studio's smoothing modifier on the item's node, when the file puts one there
  std::optional<Smoothing> smoothing;
  // the library file of its node definition, whose Morphs folder holds the item's own morphs
  std::string defFile;
};

struct LoadResult
{
  SceneGraph  scene;
  int         root = -1;
  std::string name;
  std::vector<Rig> rigs;
  // the main figure's rig (index into rigs) and its scene node group; -1 without a figure
  int               mainRig   = -1;
  int               mainGroup = -1;
  std::vector<Item>     items;
  std::vector<ItemPart> parts;
  MorphContext             morphs;
  std::vector<std::string> warnings;
  // library files this load referenced and aren't installed
  std::vector<std::string> missing;
  // the body an outfit from outside DAZ was built on (a Skyrim mesh's CBBE or UNP body), which the fit
  // measures it against in place of a Genesis base from the library; null for a DSON load
  std::shared_ptr<const wardrobe::BaseCage> sourceBase;
};

// Loads `rel` (library-relative). Throws when nothing could be built, with the reason (a missing
// base figure names the product to install).
LoadResult loadDson(Store& store, const std::string& rel, const LoadOptions& options = {});

// What a notice says of the library files `fileName` needs and aren't installed (at least one).
std::string missingMessage(const std::string& fileName, const std::vector<std::string>& missing);

// Starts reading `rel` and the library files its scene names (nodes, geometries, materials, their
// UV sets) on the pool and returns at once, so a wear's files are on their way while the figure is
// still busy. loadDson then finds them in `store`. Errors are left for loadDson to report.
// With `subdivision`, each geometry is also built at that level with its default UV set, so a
// strand hair's meshes (half a second) are ready before the figure it goes on has loaded; a
// loadDson of `rel` on the same store takes a build whose geometry, UV set and level are the ones
// it would use, and builds the rest itself. The future is for tests; callers need not wait on it.
std::shared_future<void> prefetchScene(std::shared_ptr<Store> store, const std::string& rel, std::optional<int> subdivision = std::nullopt);
// Meshes loads took from a prefetch since start-up, for tests.
size_t earlyMeshesTaken();

}  // namespace fv::dson
