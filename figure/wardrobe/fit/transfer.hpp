#pragma once

// Rest-pose placement of a garment made for one figure onto another, in two steps.
//
// Bones: each source bone gets a transform onto the bone that leads it. Its centre moves to the
// target joint, and it turns and stretches so the line to the next joint down the limb (thigh to
// knee, foot to the ball) matches the target's. Genesis 8 stands with its feet 40 cm apart and
// Genesis 9 with them 22 cm apart, so the shin leans differently: a translation alone leaves a shoe
// tilted off the foot.
//
// Shape: the source base mesh goes through the same transforms, and each garment vertex near the
// skin moves by how far the target body lies from that transferred base (a G8 bust onto a flatter
// G9 chest). The move is a translation, smoothed over the garment, so neighbouring vertices cannot
// tear apart. A port of src/lib/wardrobe/fit/transfer.ts.

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <figure/mesh/mesh_build.hpp>
#include <figure/wardrobe/bind.hpp>

namespace fv::wardrobe {

using V3 = std::array<double, 3>;

struct SkelBone
{
  std::string                name;
  std::optional<std::string> parent;
  V3                         center{};  // rest joint centre (cm), as authored in the bone's file
};

// p' = R (p - from) + to, with R row-major: a rotation, stretched along the bone.
struct Rigid
{
  std::array<double, 9> r{1, 0, 0, 0, 1, 0, 0, 0, 1};
  V3                    from{}, to{};
};

// A cage: positions (xyz) and triangles.
struct Surface
{
  std::span<const float>    positions;
  std::span<const uint32_t> tris;
};

const SkelBone* findSlot(const std::vector<SkelBone>& bones, char side, const std::string& slot);

// A transform for every source bone. A bone whose leader is the same as its parent's (a twist, or a
// metatarsal on a figure without one) moves with that parent. A bone with no leader moves with its
// nearest led ancestor, and stays put when it has none. A pectoral moves with its parent: its joint
// is in a different place in each generation.
std::map<std::string, Rigid> rigidPlan(const std::vector<SkelBone>& source, const std::vector<SkelBone>& target);

// Linear blend of the bones' rigid transforms, by skin weight. Unweighted vertices stay. `bones`
// is per bone of the skin (null: no transform).
std::vector<float> rigidPositions(std::span<const float> positions, const mesh::SkinInput& skin, const std::vector<const Rigid*>& bones);

struct ShapeOptions
{
  float near = 1.5f, far = 5, cap = 6;
  int   smooth = 8;
};

struct ShapeInputs
{
  bool                      sameTopology = false;
  const BindFilter*         baseFilter   = nullptr;
  const BindFilter*         itemFilter   = nullptr;
  ShapeOptions              shape;
  // the nearest-point binding of the base onto the target when the caller already has it
  const SurfaceBinding*     baseToTarget = nullptr;
};

// How far each garment vertex moves so its distance from the body is the distance it had from the
// source base. `base` is the source base already through the bone transfer; `target` is the body it
// now dresses.
std::vector<float> shapeCorrection(std::span<const float> item, std::span<const uint32_t> itemTris, const Surface& base, const Surface& target,
                                   const ShapeInputs& in);

// `passes` steps of half-way averaging with the edge neighbours, over a per-vertex vec3 field.
std::vector<float> laplacian(std::vector<float> field, std::span<const uint32_t> tris, size_t n, int passes);

struct LimbBox
{
  V3                                    origin{};
  std::array<V3, 3>                     axes{};  // orthonormal frame, as three vectors
  std::array<std::array<double, 4>, 3>  range{};  // per axis: source low/high, target low/high
};

// Maps the source base's limb onto the target body's limb axis by axis, in a frame along the target
// limb: length, width and height each take the target's. None when either side has too few
// vertices on the limb.
std::optional<LimbBox> limbBox(const V3& frameDir, const V3& origin, std::span<const float> source, const std::function<bool(size_t)>& sourceOn,
                               std::span<const float> target, const std::function<bool(size_t)>& targetOn);

// `positions` through `box`, moved by `weight` of the way. Scale per axis stays within 0.7-1.4.
std::vector<float> applyLimbBox(const LimbBox& box, std::span<const float> positions, const std::function<double(size_t)>& weight);

}  // namespace fv::wardrobe
