#pragma once

// A collider made of triangles, in its own space: a prop's mesh, indexed once and placed per pass by
// the matrix into the body's space, so a sofa of a few hundred thousand triangles is not moved or
// indexed again when it or the figure moves. What is inside it follows the contact design
// (docs/physics-plan.md, step 1): behind the face of the closest point, not off past its edge, and
// within a depth cap; on a closed piece of the mesh, where every edge is shared by two faces, the
// crossings of a ray say it instead.

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/physics/collide/surface_index.hpp>

namespace fv::physics {

// An axis-aligned box; empty until a point is added.
struct Box
{
  std::array<double, 3> min{INFINITY, INFINITY, INFINITY}, max{-INFINITY, -INFINITY, -INFINITY};

  bool empty() const { return min[0] > max[0]; }
  void add(double x, double y, double z);
  void add(const Box& b);
  Box  grown(double by) const;
  bool overlaps(const Box& b) const;
  bool contains(double x, double y, double z) const;
};
// The box around a box's corners carried by a matrix.
Box transformed(const Box& b, const Mat4& m);
// The box around points, xyz per point.
Box boundsOf(std::span<const float> points);

class TriangleSet
{
public:
  // xyz per vertex and three vertices per triangle, in the set's own space.
  TriangleSet(std::vector<float> positions, std::vector<uint32_t> tris);
  TriangleSet(const TriangleSet&)            = delete;
  TriangleSet& operator=(const TriangleSet&) = delete;

  std::span<const float>    positions() const { return m_positions; }
  std::span<const uint32_t> tris() const { return m_tris; }
  const Box&                bounds() const { return m_bounds; }
  const SurfaceIndex&       index() const { return m_index; }
  // The connected piece a triangle is on (through shared edges, vertices welded by position), and
  // whether that piece is closed: each of its edges shared by exactly two of its triangles.
  uint32_t pieceOf(size_t tri) const { return m_piece[tri]; }
  bool     closed(size_t tri) const { return m_closed[m_piece[tri]] != 0; }

private:
  std::vector<float>    m_positions;
  std::vector<uint32_t> m_tris;
  Box                   m_bounds;
  SurfaceIndex          m_index;
  std::vector<uint32_t> m_piece;
  std::vector<uint8_t>  m_closed;  // per piece
};

// A triangle set made the first time a pass needs it, on whichever thread asks first, and kept: a
// prop's mesh is read and indexed on the pool, not on the application thread. Its bounds are known
// before (a glTF position accessor's min and max).
class SharedTriangles
{
public:
  SharedTriangles(std::function<std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>()> read, Box bounds);
  explicit SharedTriangles(std::shared_ptr<const TriangleSet> made);

  const TriangleSet& get() const;
  const Box&         bounds() const { return m_bounds; }
  // Made already: get() costs nothing. What asks every frame on the application thread (the soft
  // flesh) leaves one that isn't to a pass on the pool.
  bool made() const { return m_made.load(std::memory_order_acquire); }

private:
  std::function<std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>()> m_read;
  Box                                                                                m_bounds;
  mutable std::once_flag                                                             m_once;
  mutable std::shared_ptr<const TriangleSet>                                         m_set;
  mutable std::atomic<bool>                                                          m_made{false};
};

// What a pass meets: triangles in their own space and the matrix that places them in the body's root
// space (centimetres).
struct ColliderRef
{
  std::shared_ptr<const SharedTriangles> set;
  Mat4                                   toBody;
  // this body's share of an overlap: 1 against a prop, which gives nothing
  double share = 1;
};

// Where a point inside a set leaves it.
struct Inside
{
  bool                  inside = false;
  std::array<double, 3> exit{};    // the closest point of the surface
  std::array<double, 3> normal{};  // out of the set there, unit
  double                depth = 0;
};

// Whether point p (in the set's space) is inside the set. Past `depth` from the surface it is left
// alone, as the far side of a thin prop or of a limb would answer; `closedDepth` is the same on a
// closed piece, where the crossings make inside certain.
Inside insideOf(const TriangleSet& set, const std::array<double, 3>& p, double depth, double closedDepth, Stamps& stamps);

// Where the ray from p along `dir` (unit) first meets the set's surface within `reach`: the point, and
// the face's normal turned along the ray (out of the set, for a p inside). `inside` false when it
// meets nothing.
Inside exitAlong(const TriangleSet& set, const std::array<double, 3>& p, const std::array<double, 3>& dir, double reach, Stamps& stamps);

}  // namespace fv::physics
