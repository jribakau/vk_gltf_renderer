#pragma once

// A triangle mesh's quads found again (docs/wardrobe-plan.md, step 7). A game mesh is all triangles,
// most of them quads cut in two. The cloth solver holds every polygon edge at its length and softens
// only a quad's diagonals (its shear), so a quad cut in two has one diagonal held as stiffly as its
// sides: the cloth can't shear along it and hangs like card. Two triangles are a quad again when they
// share an edge, lie nearly flat together and make a shape near a rectangle; the best pairs are taken
// first, so a grid of right triangles pairs across its long edges. Triangles that make no such pair
// (a fan, a mesh of even triangles) stay triangles.
//
// For the cloth's polygons only: what is drawn keeps its triangles, so no diagonal turns.

#include <cstdint>
#include <span>
#include <vector>

namespace fv::garment {

struct QuadOptions
{
  float flat   = 30;  // degrees: two triangles' normals no further apart than this are one face
  float square = 25;  // degrees: each corner of the quad within this of a right angle
};

struct Polys
{
  std::vector<int32_t> start{0}, verts;  // polygon i is verts[start[i] .. start[i + 1])
  size_t               paired = 0;       // triangles that went into quads
  size_t               triangles = 0;    // triangles there were
};

// The polygons of `polyStart`/`polyVerts` (as mesh::BuiltMesh's cage has them) with their triangles
// paired into quads over `positions` (xyz per vertex). A polygon that isn't a triangle is kept as it is,
// and every polygon keeps the way it winds.
Polys pairQuads(std::span<const float> positions, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts, const QuadOptions& options = {});

}  // namespace fv::garment
