#pragma once

// Fitting worn items to the figure's shape, DAZ Studio's "auto-follow". Every item vertex is bound
// once to the closest point on the figure's unmorphed body; a body morph then moves the item by the
// same amount as that point. Items that ship their own version of a morph (DAZ clothing has fitted
// copies of the base body shapes, same id) use it instead; the MorphSet asks `own` first.
//
// Geograft seam vertices copy the body vertex they are welded to, so the seam stays closed.
// Rigidity groups (eyes, teeth) move as one block with the average of their reference vertices, so
// eyeballs follow the eye socket instead of being squashed by it. A port of
// src/lib/loaders/dson/follow.ts.

#include <cstdint>
#include <optional>
#include <vector>

#include <figure/dson/morphs.hpp>
#include <figure/dson/store.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/wardrobe/bind.hpp>

namespace fv::wardrobe {

struct Rigidity
{
  // per item cage vertex, 0 (follows the surface) to 1 (moves with its group)
  std::vector<float> weight;
  struct Group
  {
    std::vector<uint32_t> refs, mask;
  };
  std::vector<Group> groups;
};

// A geometry's `rigidity` block; none when it has none.
std::optional<Rigidity> parseRigidity(const dson::Doc& doc, const dson::Json& geo, int vertexCount);

// Geograft seam: (graft vertex, body vertex) pairs, flattened, and the body polygons it replaces.
struct Graft
{
  std::vector<int32_t> pairs;
  std::vector<int32_t> hidden;
  // the vertex and polygon counts of the figure mesh it was made for, -1 when the file doesn't say
  int vertexCount = -1, polyCount = -1;
};
std::optional<Graft> parseGraft(const dson::Doc& doc, const dson::Json& geo);
// Whether its seam pairs and hidden polygons, which number the vertices and polygons of the mesh it was
// made for, mean the same on `body`: a Genesis 3 graft's don't on Genesis 8, whose mesh is another.
bool graftFits(const Graft& graft, const mesh::BuiltMesh& body);

// The item's version of one body morph target, or null when it moves nothing on the item. Both
// meshes work through their cages' vertices; `scratch` holds body-cage-sized deltas between calls.
dson::TargetPtr projectTarget(const SurfaceBinding& binding, const Rigidity* rigidity, const mesh::BuiltMesh& body, int itemVertices,
                              const mesh::RenderTarget& target, std::vector<float>& scratch);

}  // namespace fv::wardrobe
