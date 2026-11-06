#pragma once

// A uniform grid over a surface's triangles: the exact closest-triangle search the wardrobe binds
// with (wardrobe/bind.cpp), and the ray crossings the contact pass counts to tell inside from
// outside on a closed prop. It reads the positions and triangles it is given and keeps no copy: the
// caller keeps them alive (a TriangleSet owns its own).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace fv::physics {

// A search marks the triangles it has looked at with its id. Each thread keeps one buffer for every
// search it runs, and ids only grow, so a small search allocates nothing and no mark left by an
// earlier one can match.
struct Stamps
{
  std::vector<int32_t> stamp;
  int32_t              last = 0;

  int32_t next(size_t triangles);
};
// This thread's.
Stamps& threadStamps();

class SurfaceIndex
{
public:
  SurfaceIndex() = default;
  SurfaceIndex(std::span<const float> positions, std::span<const uint32_t> tris);

  bool   empty() const { return nt == 0; }
  size_t triangles() const { return nt; }

  struct Hit
  {
    int64_t tri = -1;
    double  u = 1, v = 0, w = 0;
    // squared distance to the point found
    double d2 = INFINITY;
  };

  // The closest acceptable triangle to p, if one lies within sqrt(maxDist2). `stamp` (one entry per
  // triangle, owned by the caller's thread) and `id` (unique per search on that thread) mark the
  // triangles this search has seen. With `limbs` (a limb id per vertex, wardrobe::BindFilter's) and a
  // `mask`, only a triangle with a vertex of an allowed limb, or of none, is acceptable.
  Hit search(double px, double py, double pz, int32_t mask, std::span<const int16_t> limbs, double maxDist2, std::vector<int32_t>& stamp, int32_t id) const;
  // The same with `accept` (triangle index -> bool) saying which triangles are acceptable.
  template <class Accept>
  Hit nearest(double px, double py, double pz, Accept&& accept, double maxDist2, std::vector<int32_t>& stamp, int32_t id) const;

  // How many of the triangles `accept` takes (triangle index -> bool) the ray from o along d crosses
  // ahead of o. Walks the cells the ray passes through, so a big surface costs its cells, not its
  // triangles.
  template <class Accept>
  int crossings(const double o[3], const double d[3], Accept&& accept, std::vector<int32_t>& stamp, int32_t id) const;

  // The nearest triangle the ray from o along d meets ahead of o within maxT (in units of d's
  // length): its index, or -1, and where along the ray.
  struct RayHit
  {
    int64_t tri = -1;
    double  t   = INFINITY;
  };
  RayHit firstHit(const double o[3], const double d[3], double maxT, std::vector<int32_t>& stamp, int32_t id) const;

private:
  // shaved off the cell-wall distance so float error cannot hide a strictly closer triangle on the face
  static constexpr double kWallSlack = 1e-4;

  int  cellOf(double v, double min, int n) const { return std::min(n - 1, std::max(0, int(std::floor((v - min) / cell)))); }
  bool triOk(size_t t, int32_t mask, std::span<const int16_t> limbs) const;
  // closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5), as barycentric u, v, w
  double closest(double px, double py, double pz, size_t a, size_t b, size_t c, double& bu, double& bv, double& bw) const;
  // whether the ray o + t d meets triangle t ahead of o (Möller-Trumbore), and where along it
  bool hits(const double o[3], const double d[3], size_t t, double* at = nullptr) const;
  // The cells the ray from o along d passes through inside the grid, in order: visit(cell index, the
  // ray's parameter where it leaves the cell) until it returns false or the ray leaves the grid.
  template <class Visit>
  void walk(const double o[3], const double d[3], Visit&& visit) const;

  std::span<const float>    P;
  std::span<const uint32_t> T;
  size_t                    nt   = 0;
  float                     minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0;
  double                    cell = 1;
  int                       nx = 1, ny = 1, nz = 1, maxR = 1;
  std::vector<uint32_t>     start, tris;
};

template <class Visit>
void SurfaceIndex::walk(const double o[3], const double d[3], Visit&& visit) const
{
  if(!nt)
    return;
  // where the ray is inside the grid's box: slabs
  const double lo[3] = {minX, minY, minZ};
  const double hi[3] = {minX + nx * cell, minY + ny * cell, minZ + nz * cell};
  double       t0 = 0, t1 = INFINITY;
  for(int a = 0; a < 3; a++)
  {
    if(std::fabs(d[a]) < 1e-300)
    {
      if(o[a] < lo[a] || o[a] > hi[a])
        return;
      continue;
    }
    double ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a];
    if(ta > tb)
      std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
  }
  if(t0 > t1)
    return;
  // the cells it passes through, in order (Amanatides and Woo)
  const int n[3] = {nx, ny, nz};
  int       c[3], step[3];
  double    next[3], delta[3];
  for(int a = 0; a < 3; a++)
  {
    const double at   = o[a] + d[a] * t0;
    c[a]              = cellOf(at, lo[a], n[a]);
    step[a]           = d[a] > 0 ? 1 : d[a] < 0 ? -1 : 0;
    delta[a]          = step[a] ? cell / std::fabs(d[a]) : INFINITY;
    const double wall = lo[a] + (c[a] + (step[a] > 0 ? 1 : 0)) * cell;
    next[a]           = step[a] ? t0 + (wall - at) / d[a] : INFINITY;
  }
  while(true)
  {
    const int a = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : (next[1] < next[2] ? 1 : 2);
    if(!visit((size_t(c[2]) * ny + c[1]) * nx + c[0], std::min(next[a], t1)))
      return;
    if(next[a] > t1)
      return;
    c[a] += step[a];
    if(c[a] < 0 || c[a] >= n[a])
      return;
    next[a] += delta[a];
  }
}

template <class Accept>
SurfaceIndex::Hit SurfaceIndex::nearest(double px, double py, double pz, Accept&& accept, double maxDist2, std::vector<int32_t>& stamp, int32_t id) const
{
  Hit hit;
  if(!nt)
    return hit;
  if(maxDist2 < INFINITY)
  {
    // further than that from the grid's box: nothing can be within reach
    const double bx = std::max({0.0, minX - px, px - maxX}), by = std::max({0.0, minY - py, py - maxY}), bz = std::max({0.0, minZ - pz, pz - maxZ});
    if(bx * bx + by * by + bz * bz > maxDist2)
      return hit;
  }
  const int ci = cellOf(px, minX, nx), cj = cellOf(py, minY, ny), ck = cellOf(pz, minZ, nz);
  // clamped in from outside the grid: the cell does not contain the point, so its walls are not a bound
  const bool contained = int(std::floor((px - minX) / cell)) == ci && int(std::floor((py - minY) / cell)) == cj && int(std::floor((pz - minZ) / cell)) == ck;
  double     best      = INFINITY;
  double     bu = 0, bv = 0, bw = 0;
  for(int r = 0; r <= maxR; r++)
  {
    // cells r rings out are at least (r - 1) cells away
    if(r > 1)
    {
      const double ring = std::pow((r - 1) * cell, 2);
      if((hit.tri >= 0 && best <= ring) || ring > maxDist2)
        break;
    }
    for(int k = ck - r; k <= ck + r; k++)
    {
      if(k < 0 || k >= nz)
        continue;
      for(int j = cj - r; j <= cj + r; j++)
      {
        if(j < 0 || j >= ny)
          continue;
        const bool shell = std::abs(k - ck) == r || std::abs(j - cj) == r;
        for(int i = ci - r; i <= ci + r; i += shell ? 1 : std::max(1, 2 * r))
        {
          if(i < 0 || i >= nx)
            continue;
          const size_t c = (size_t(k) * ny + j) * nx + i;
          // A cell whose box lies further off than the nearest found so far, or than maxDist2, holds no
          // nearer triangle: a triangle's closest point lies in a cell it is filed in. A point near a
          // dense surface (a body's 33k triangles in 3 cm cells) otherwise tried every face of the 27
          // cells about it
          if(r > 0)
          {
            const double gx  = std::max({0.0, minX + i * cell - px, px - (minX + (i + 1) * cell)});
            const double gy  = std::max({0.0, minY + j * cell - py, py - (minY + (j + 1) * cell)});
            const double gz  = std::max({0.0, minZ + k * cell - pz, pz - (minZ + (k + 1) * cell)});
            const double gap = std::sqrt(gx * gx + gy * gy + gz * gz) - kWallSlack;
            if(gap > 0 && gap * gap > std::min(best, maxDist2))
              continue;
          }
          for(uint32_t s = start[c], e = start[c + 1]; s < e; s++)
          {
            const uint32_t t = tris[s];
            if(stamp[t] == id)
              continue;
            stamp[t] = id;
            if(!accept(t))
              continue;
            const double d = closest(px, py, pz, T[t * 3] * 3ull, T[t * 3 + 1] * 3ull, T[t * 3 + 2] * 3ull, bu, bv, bw);
            if(d < best)
            {
              best    = d;
              hit.tri = t;
              hit.u = bu, hit.v = bv, hit.w = bw;
            }
          }
        }
      }
    }
    // ring 0 holds every triangle that meets the cell: a point inside it, closer to its hit than
    // to the nearest wall, has no strictly closer triangle outside it
    if(r == 0 && contained && hit.tri >= 0)
    {
      const double dx   = std::min(px - (minX + ci * cell), minX + (ci + 1) * cell - px);
      const double dy   = std::min(py - (minY + cj * cell), minY + (cj + 1) * cell - py);
      const double dz   = std::min(pz - (minZ + ck * cell), minZ + (ck + 1) * cell - pz);
      const double wall = std::min({dx, dy, dz}) - kWallSlack;
      if(wall > 0 && best <= wall * wall)
        break;
    }
  }
  if(best > maxDist2)
    hit.tri = -1;
  else
    hit.d2 = best;
  return hit;
}

template <class Accept>
int SurfaceIndex::crossings(const double o[3], const double d[3], Accept&& accept, std::vector<int32_t>& stamp, int32_t id) const
{
  int count = 0;
  walk(o, d, [&](size_t cellIndex, double) {
    for(uint32_t s = start[cellIndex], e = start[cellIndex + 1]; s < e; s++)
    {
      const uint32_t t = tris[s];
      if(stamp[t] == id)
        continue;
      stamp[t] = id;
      if(accept(t) && hits(o, d, t))
        count++;
    }
    return true;
  });
  return count;
}

}  // namespace fv::physics
