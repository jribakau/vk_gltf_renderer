#pragma once

// DAZ Studio's rigid follow nodes (studio/node/rigid_follow): a button, a buckle or an earring that is
// a node of its own, carried as a whole by a patch of the mesh it sits on (its Follow Target). The
// node names that patch by its reference vertices (its rigidity group's); where they go, turned and
// shifted as one, the node goes: with the pose, the shape and the drape of what it follows. Every one
// in the library turns in full and keeps its size (rotation_mode "full", scale_modes "none").

#include <cstdint>
#include <span>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/scene/graph.hpp>

namespace fv::wardrobe {

// The world the follow node takes now, from the one it had at rest in the followed mesh's frame
// (`restInMesh`: inverse(mesh world) · node world, as loaded): the best turn and shift taking its
// reference vertices from `rest` to `now` (xyz each, in the mesh's own space), applied in that frame.
// `turns` false shifts it alone (rotation_mode "none").
Mat4 followWorld(const Mat4& meshWorld, const Mat4& restInMesh, std::span<const float> rest, std::span<const float> now, bool turns);

// A node's position, quaternion and scale taken from `m` (a matrix with no shear).
void setTransform(Node& node, const Mat4& m);

}  // namespace fv::wardrobe
