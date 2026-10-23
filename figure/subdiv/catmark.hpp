#pragma once

// Catmull-Clark subdivision as DAZ Studio applies it to SubD figures: OpenSubdiv's Catmark scheme
// with boundary edges kept sharp, refined uniformly (refined points, not the limit surface), and
// UVs refined as face-varying data that stays linear only at UV corners. The result matches
// OpenSubdiv's Far::TopologyRefiner vertex for vertex: the same child ordering (children of
// vertices, then of faces, then of edges, at every level), so data addressed by refined vertex
// index, such as HD morphs, lines up.
//
// The refinement is kept as one sparse stencil per level (child vertex = weighted sum of parent
// vertices). Subdivision is linear, so the same stencils refine positions, morph deltas and skin
// weights alike.
//
// A port of src/lib/subdiv/catmark.ts. Arithmetic follows it: sums in double, stored as float
// where the TypeScript stores into a Float32Array, so both produce the same numbers.

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace fv::subdiv {

enum class BoundaryInterpolation
{
  eEdges,
  eEdgesAndCorners,
};

struct CageMesh
{
  int vertexCount = 0;
  // Start of each face's corners in faceVerts, plus the end.
  std::span<const int32_t> faceStart;
  std::span<const int32_t> faceVerts;
  // UV value index per face corner, laid out as faceVerts; with `values`, optional (empty).
  std::span<const int32_t> faceValues;
  // Two floats per UV value.
  std::span<const float> values;
};

// One level's refinement: child vertex r is the sum over k in [rowStart[r], rowStart[r + 1]) of
// weight[k] * parent[col[k]].
struct Stencils
{
  std::vector<int32_t> rowStart;
  std::vector<int32_t> col;
  std::vector<float>   weight;
  int                  parentCount = 0;

  int rows() const { return int(rowStart.size()) - 1; }

  // Parent vertex -> the child rows that read it (built on first sparse use).
  struct Transpose
  {
    std::vector<int32_t> start;
    std::vector<int32_t> row;
  };
  const Transpose& transpose() const;

private:
  mutable std::shared_ptr<Transpose> m_transpose;
};

struct Refinement
{
  std::vector<Stencils> levels;
  int                   vertexCount = 0;
  // The finest level's faces: quads, four vertices each.
  std::vector<int32_t> quads;
  // The cage face each quad came from.
  std::vector<int32_t> quadParent;
  // UV value per quad corner, and the values (two floats each); empty without UVs.
  std::vector<int32_t> quadValues;
  std::vector<float>   values;
};

// Refines `cage` `levels` times.
Refinement refine(const CageMesh& cage, int levels, BoundaryInterpolation boundary = BoundaryInterpolation::eEdges);

// Refines per-vertex data (`width` floats per cage vertex) to the finest level.
std::vector<float> refineDense(const Refinement& r, std::span<const float> data, int width);

struct Sparse
{
  std::vector<uint32_t> indices;
  std::vector<float>    data;
};

// Refines sparse per-vertex data (`width` floats per listed cage vertex), such as a morph's deltas.
// Only the refined vertices that read a listed vertex are computed; the result lists those, in order.
Sparse refineSparse(const Refinement& r, std::span<const uint32_t> indices, std::span<const float> data, int width);

}  // namespace fv::subdiv
