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

#include <figure/dson/channels.hpp>
#include <figure/dson/dforce.hpp>
#include <figure/materials/surface.hpp>
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
  // the centre before formulas moved it (set when the first one does)
  std::optional<std::array<double, 3>> centerBase;
  std::array<double, 3> endPoint{};
  std::vector<JointChannel> rotation, translation, scale;
  // whether its parent's own scale reaches it (what the parent inherited always does)
  bool inheritsScale = true;
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
  // taken off (a worn item): the node and what is under it are left out of the glTF
  bool removed = false;
  // the DSON node it was made from (presets address nodes by it)
  std::string dsonNode;
  // bones: their joint data
  std::shared_ptr<JointData> joint;
  // meshes: index into SceneGraph::meshes
  int mesh = -1;
};

// A surface: what DAZ describes (materials/surface.hpp), the Material panel's edits laid over it,
// and the merged DSON channels it was parsed from, for what reads them directly and for presets
// that override some of them.
struct Material
{
  std::string                           name;
  mat::Surface                          surface;
  mat::Edits                            edits;
  // what its mesh belongs to, which its surface is read with whenever its channels change
  mat::Context                          context = mat::Context::eBody;
  std::shared_ptr<const dson::Channels> channels;
  // the channels as the character came: its file's own and the presets its add-ons were worn with.
  // Choosing other eyes, or the character's own, puts these back; presets opened later leave it.
  std::shared_ptr<const dson::Channels> baseline;
  // How much surface one UV unit covers, in metres (the square root of area over UV area), on the
  // unmorphed mesh as it loaded: what turns a bump map's height into slopes. 0: not measured.
  float metresPerUv = 0;
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
  // the influences per cage vertex the skin was made from (bone indices into `bones`), for fitting
  mesh::SkinInput        cageSkin;
  // dForce cloth settings, when the scene made this a simulation object (per cage vertex)
  std::shared_ptr<const dson::DforceData> dforce;
  Mat4                   bindMatrix;
  std::vector<Mat4>      boneInverses;
  bool                   skinned() const { return !bones.empty(); }
};

class SceneGraph;
// Measures Material::metresPerUv for every material still at 0, from the meshes as they are (DSON
// centimetres), to two significant figures.
void measureUvScales(SceneGraph& scene);
// Whether a node was taken off: it or a node above it is removed (Node::removed). Hidden is not.
bool takenOff(const SceneGraph& scene, int node);
// Per mesh, whether it is still in use: false only when every node that draws it was taken off. A
// mesh no node draws counts as in use, as liveMaterials counts a material no node uses.
std::vector<bool> liveMeshes(const SceneGraph& scene);

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
