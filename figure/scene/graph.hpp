#pragma once

// The figure scene as the loaders build it: a tree of groups, bones and meshes with three.js-style
// local transforms (position, quaternion, scale), meshes with their built geometry, skin and
// materials. It is what the TypeScript viewer kept as three.js objects; gltf/gltf_build.cpp turns
// it into the renderer's glTF model, and posing, morphs and fitting edit it.

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/mesh/mesh_build.hpp>

namespace fv {

// One posable DSON channel of a bone (rotation in degrees, translation in cm, scale as factor).
struct JointChannel
{
  std::string axis;  // "x", "y", "z" or "general"
  std::string label;
  double      min = -10000, max = 10000;
  bool        clamped = false, locked = false;
  double      value = 0;
};

// What posing needs to know about a bone (src/lib/loaders/dson/skin.ts JointData).
struct JointData
{
  std::string id, label;
  // joint axes as an Euler in degrees, in rotationOrder; the order's first axis is the twist axis
  std::array<double, 3> orientation{};
  std::string           rotationOrder = "XYZ";
  // joint centre in the figure's rest pose (cm), after any character shape moved it
  std::array<double, 3> center{};
  // the centre in the bone's own file, before a worn item took the figure's centres
  std::array<double, 3> centerAuthored{};
  std::array<double, 3> endPoint{};
  std::vector<JointChannel> rotation, translation, scale;
};

enum class NodeKind
{
  eGroup,
  eBone,
  eMesh,
};

struct Node
{
  std::string      name;
  NodeKind         kind   = NodeKind::eGroup;
  int              parent = -1;
  std::vector<int> children;
  Vec3             position;
  Quat             quaternion;
  Vec3             scale{1, 1, 1};
  bool             visible = true;
  // the DSON node it was made from (presets address nodes by it)
  std::string dsonNode;
  // bones: their joint data
  std::shared_ptr<JointData> joint;
  // meshes: index into SceneGraph::meshes
  int mesh = -1;
};

// A surface's look, for now its base colour and cutout (the full Surface comes with the material port).
struct Material
{
  std::string name;
  // linear RGB, from the sRGB the DSON colour pickers hold
  std::array<float, 3> baseColor{1, 1, 1};
  // library-relative image paths, empty when none
  std::string baseColorMap;
  float       opacity = 1;
  std::string opacityMap;
  float       roughness = 0.6f;
  // the merged DSON channels, by normalised name: value as JSON text, and image path
  std::map<std::string, std::pair<std::string, std::string>> channels;
};

struct Mesh
{
  std::string                            name;
  std::shared_ptr<mesh::BuiltMesh>       geo;
  std::vector<std::string>               materialGroups;
  std::vector<int>                       materials;  // per material group, into SceneGraph::materials
  // skinned: the bones in skin-index order (node ids), the weights, and the bind as three's
  // SkinnedMesh.bind makes it (bindMatrix = the mesh's world, boneInverses = inverse bone worlds)
  std::vector<int>       bones;
  mesh::SkinOutput       skin;
  Mat4                   bindMatrix;
  std::vector<Mat4>      boneInverses;
  bool                   skinned() const { return !bones.empty(); }
};

class SceneGraph
{
public:
  std::vector<Node>     nodes;
  std::vector<Mesh>     meshes;
  std::vector<Material> materials;

  // Adds a node under `parent` (-1: a root), at `index` among its children (-1: last).
  int add(Node node, int parent = -1, int index = -1);
  // Moves a node under another parent (-1: a root), at `index` among its children (-1: last).
  void attach(int node, int parent, int index = -1);
  std::vector<int> roots() const;

  Mat4 local(int node) const;
  Mat4 world(int node) const;
  // every node's world matrix, parents before children
  std::vector<Mat4> worlds() const;
};

}  // namespace fv
