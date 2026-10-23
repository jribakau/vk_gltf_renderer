#pragma once

// Smooth vertex normals for DSON meshes, at build and after CPU morphs move vertices. Each render
// vertex sums the area-weighted normals of its triangles in double precision; all its splits (drawn
// vertices along UV seams) take the result. A port of src/lib/workers/normals.ts.

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace fv::mesh {

// Normals welded over render vertices (smooth across UV seams), area-weighted, into `out` (xyz per
// drawn vertex). `position` is xyz per drawn vertex; renderIndex maps drawn -> render vertex.
void weldNormals(std::span<const float> position, std::span<const uint32_t> renderIndex, std::span<const int32_t> firstSplit,
                 std::span<const uint32_t> renderTris, int renderCount, std::span<float> out);

// Keeps a mesh's welded normals up to date as CPU morphs move its vertices: all of them, or only
// around the vertices that moved, which is what a slider step or a joint corrective needs.
class NormalWelder
{
public:
  // The spans must outlive the welder (they are the mesh's).
  NormalWelder(std::span<const uint32_t> renderIndex, std::span<const int32_t> firstSplit, std::span<const int32_t> nextSplit,
               std::span<const uint32_t> renderTris, int renderCount);

  void full(std::span<const float> position, std::span<float> out);

  // Recomputes the normals the moved drawn vertices `moved` affect: every vertex of a triangle that
  // touches one. Returns the range of drawn vertices written (inclusive), or none when nothing was.
  // Falls back to `full` when most of the mesh moved.
  std::optional<std::pair<uint32_t, uint32_t>> partial(std::span<const float> position, std::span<float> out, std::span<const uint32_t> moved);

private:
  void adjacency();

  std::span<const uint32_t> m_renderIndex;
  std::span<const int32_t>  m_firstSplit;
  std::span<const int32_t>  m_nextSplit;
  std::span<const uint32_t> m_renderTris;
  int                       m_renderCount;

  // render vertex -> its triangles, built on first partial use
  std::vector<int32_t> m_adjStart, m_adjTris;
  // partial updates' working space: which vertices and triangles this pass met, and the sums
  std::vector<int32_t> m_vertexPass, m_triPass, m_affected;
  std::vector<double>  m_sum;
  int32_t              m_pass = 0;
};

}  // namespace fv::mesh
