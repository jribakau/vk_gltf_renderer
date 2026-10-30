#pragma once

// DSON skeletons and skin bindings. Genesis 3/8/9 use general (node) weights; each joint lists
// [vertex, weight] pairs. Bones are placed at their center_point in the figure's rest pose, so the
// bind pose matches the mesh and the figure appears exactly as authored. A port of
// src/lib/loaders/dson/skin.ts.

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dson {

struct Rig
{
  // the root bone's node
  int root = -1;
  // bone id -> node, in node_library order
  std::vector<std::pair<std::string, int>> bones;

  int byId(const std::string& id) const;
  // a bone by id or name
  int find(const SceneGraph& scene, const std::string& name) const;
};

// Builds bones from every bone/figure node in a node_library, as nodes of `scene` (the root left
// unparented). `centers` overrides joint centres by bone id: a character's scene file records where
// its shape morphs moved the joints.
std::optional<Rig> buildRig(SceneGraph& scene, const Json& nodeLibrary, const std::map<std::string, std::array<double, 3>>* centers);

struct SkinBinding
{
  // bones in skin-index order (nodes)
  std::vector<int> order;
  mesh::SkinInput  input;
};

// A DSON skin modifier's influences per cage vertex (its binding's node weights, or the strongest
// TriAx axis weight for Genesis 2), with the bones in the order they are met, and its skin settings
// (extra "skin_settings": general_map_mode DualQuat is all dual quaternion, and blend_mode
// BlendLinearDualQuat blends it with linear by the blend_weights map, read as the dual quaternion
// share; a uniform map is kept as one value). None without joints.
std::optional<SkinBinding> skinBinding(const Doc& doc, const Json& modifier, const Rig& rig, int vertexCount);

// Strand hair whose skin has no joints: DAZ Studio moves it with a dForce Hair Follow modifier, which
// rides on the figure's surface. Each strand goes wholly on the bone the skin's selection map names for
// its polygon group, or the head where it names none. None for anything but strands without joints.
std::optional<SkinBinding> strandBinding(const Doc& doc, const Json& modifier, const Json& geo, const Rig& rig, int vertexCount);

}  // namespace fv::dson
