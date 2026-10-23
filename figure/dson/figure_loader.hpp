#pragma once

// Native DAZ loader: a .duf scene/character file (or a bare .dsf asset) from the content library to
// a SceneGraph with UVs, smooth seam-free normals, skeletons, skin weights and materials. A port of
// src/lib/loaders/dson/figure.ts, so far the parts that build what a scene shows at rest: nodes,
// worn items and their skeletons, geometry (subdivided when SubD), skin and surfaces. Morphs,
// fitting and dForce join it as they are ported.
//
// The scene is in centimetres, as DSON is; the host scales it to metres.

#include <map>
#include <set>
#include <string>
#include <vector>

#include <figure/dson/morphs.hpp>
#include <figure/dson/rig.hpp>
#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>

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
  // its skeletons (indices into LoadResult::rigs), which follow the figure's pose
  std::vector<int> rigs;
  // its meshes (indices into SceneGraph::meshes)
  std::vector<int> meshes;
};

struct LoadOptions
{
  // SubD geometry drawn subdivided this many times (0-3)
  int subdivision = 0;
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

struct LoadResult
{
  SceneGraph  scene;
  int         root = -1;
  std::string name;
  std::vector<Rig> rigs;
  // the main figure's rig (index into rigs) and its scene node group; -1 without a figure
  int               mainRig   = -1;
  int               mainGroup = -1;
  std::vector<Item> items;
  MorphContext             morphs;
  std::vector<std::string> warnings;
  // library files referenced but not installed
  std::vector<std::string> missing;
};

// Loads `rel` (library-relative). Throws when nothing could be built, with the reason (a missing
// base figure names the product to install).
LoadResult loadDson(Store& store, const std::string& rel, const LoadOptions& options = {});

}  // namespace fv::dson
