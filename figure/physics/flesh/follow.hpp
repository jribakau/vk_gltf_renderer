#pragma once

// What is worn over the soft flesh moves with it (physics-plan step 9). A conforming item is bound
// once to the body's cage (wardrobe::SurfaceBinding: three body vertices and their weights per item
// vertex), and its morphs follow the body's through that binding (wardrobe::projectTarget); the
// flesh's offsets from the skin follow it the same way, frame by frame: each item vertex as the body
// point it is bound to, and a rigid group (a button, a buckle) as one, by its reference vertices'
// mean. The renderer adds them where it adds the body's (RetainedScene::setMeshDisplacement).

#include <cstdint>
#include <span>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/wardrobe/follow.hpp>

namespace fv::physics {

// Whether the binding reads any of `bodyVertices` (the cage vertices the flesh moves): an item bound
// nowhere near them (shoes, a hat) never follows it.
bool bindsTo(const wardrobe::SurfaceBinding& binding, std::span<const uint32_t> bodyVertices);

// The item's offsets, xyz per item cage vertex, into `out`, from the body's (`body`, xyz per body cage
// vertex, in the body mesh's posed space). `toItem` turns an offset from that space into the item
// mesh's (its linear part is read), or null when the two meshes share a space. False, and `out` all
// zeros, when nothing it is bound to has moved.
bool followFlesh(const wardrobe::SurfaceBinding& binding, const wardrobe::Rigidity* rigidity, std::span<const float> body, const Mat4* toItem,
                 std::vector<float>& out);

// Offsets per drawn vertex of a mesh drawn as it is, from its cage's: each drawn vertex takes its cage
// vertex's (`drawnFrom`, mesh::BuiltMesh::renderIndex, which is the cage vertex there).
void drawnOffsets(std::span<const float> cage, std::span<const uint32_t> drawnFrom, std::vector<float>& out);

}  // namespace fv::physics
