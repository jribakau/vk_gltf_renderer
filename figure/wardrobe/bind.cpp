#include <figure/wardrobe/bind.hpp>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace fv::wardrobe {

namespace {

// A uniform grid over the body's triangles and the exact closest-triangle search over it.
class SurfaceIndex
{
public:
  SurfaceIndex(std::span<const float> P, std::span<const uint32_t> T, const BindFilter* filter)
      : P(P)
      , T(T)
      , filter(filter)
      , nt(T.size() / 3)
  {
    if(!nt)
      return;
    minX = minY = minZ = INFINITY;
    maxX = maxY = maxZ = -INFINITY;
    for(uint32_t i : T)
    {
      const size_t o = size_t(i) * 3;
      minX           = std::min(minX, P[o]);
      minY           = std::min(minY, P[o + 1]);
      minZ           = std::min(minZ, P[o + 2]);
      maxX           = std::max(maxX, P[o]);
      maxY           = std::max(maxY, P[o + 1]);
      maxZ           = std::max(maxZ, P[o + 2]);
    }
    // a uniform grid of triangles: about one per cell on average, capped at a few million cells
    const double ex = std::max(double(maxX - minX), 1e-3), ey = std::max(double(maxY - minY), 1e-3), ez = std::max(double(maxZ - minZ), 1e-3);
    cell            = std::cbrt(ex * ey * ez / double(nt)) * 1.5;
    cell            = std::max(cell, std::cbrt(ex * ey * ez / 4e6));
    nx              = std::max(1, int(std::ceil(ex / cell)));
    ny              = std::max(1, int(std::ceil(ey / cell)));
    nz              = std::max(1, int(std::ceil(ez / cell)));
    maxR            = std::max({nx, ny, nz});
    // triangles per cell, as compressed rows: count, prefix-sum, fill
    start.assign(size_t(nx) * ny * nz + 1, 0);
    std::vector<int> range(nt * 6);
    for(size_t t = 0; t < nt; t++)
    {
      const size_t a = T[t * 3] * 3ull, b = T[t * 3 + 1] * 3ull, c = T[t * 3 + 2] * 3ull;
      int*         r = &range[t * 6];
      r[0]           = cellOf(std::min({P[a], P[b], P[c]}), minX, nx);
      r[1]           = cellOf(std::max({P[a], P[b], P[c]}), minX, nx);
      r[2]           = cellOf(std::min({P[a + 1], P[b + 1], P[c + 1]}), minY, ny);
      r[3]           = cellOf(std::max({P[a + 1], P[b + 1], P[c + 1]}), minY, ny);
      r[4]           = cellOf(std::min({P[a + 2], P[b + 2], P[c + 2]}), minZ, nz);
      r[5]           = cellOf(std::max({P[a + 2], P[b + 2], P[c + 2]}), minZ, nz);
      for(int k = r[4]; k <= r[5]; k++)
        for(int j = r[2]; j <= r[3]; j++)
          for(int i = r[0]; i <= r[1]; i++)
            start[(size_t(k) * ny + j) * nx + i + 1]++;
    }
    for(size_t i = 1; i < start.size(); i++)
      start[i] += start[i - 1];
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    tris.resize(start.back());
    for(size_t t = 0; t < nt; t++)
    {
      const int* r = &range[t * 6];
      for(int k = r[4]; k <= r[5]; k++)
        for(int j = r[2]; j <= r[3]; j++)
          for(int i = r[0]; i <= r[1]; i++)
            tris[fill[(size_t(k) * ny + j) * nx + i]++] = uint32_t(t);
    }
  }

  bool empty() const { return nt == 0; }
  size_t triangles() const { return nt; }

  struct Hit
  {
    int64_t tri = -1;
    double  u = 1, v = 0, w = 0;
  };

  // The closest acceptable triangle to p, if one lies within sqrt(maxDist2). `stamp` (one entry per
  // triangle, owned by the caller's thread) and `id` (unique per search on that thread) mark the
  // triangles this search has seen.
  Hit search(double px, double py, double pz, int32_t mask, double maxDist2, std::vector<int32_t>& stamp, int32_t id) const
  {
    Hit hit;
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
            for(uint32_t s = start[c], e = start[c + 1]; s < e; s++)
            {
              const uint32_t t = tris[s];
              if(stamp[t] == id)
                continue;
              stamp[t] = id;
              if(!triOk(t, mask))
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
    return hit;
  }

private:
  // shaved off the cell-wall distance so float error cannot hide a strictly closer triangle on the face
  static constexpr double kWallSlack = 1e-4;

  int cellOf(double v, double min, int n) const { return std::min(n - 1, std::max(0, int(std::floor((v - min) / cell)))); }

  bool triOk(size_t t, int32_t mask) const
  {
    if(!filter || !mask)
      return true;
    for(int k = 0; k < 3; k++)
    {
      const uint32_t v    = T[t * 3 + k];
      const int      limb = v < filter->bodyLimb.size() ? filter->bodyLimb[v] : 0;
      if(!limb || (mask & (1 << limb)))
        return true;
    }
    return false;
  }

  // closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5), as barycentric u, v, w
  double closest(double px, double py, double pz, size_t a, size_t b, size_t c, double& bu, double& bv, double& bw) const
  {
    const double ax = P[a], ay = P[a + 1], az = P[a + 2];
    const double abx = P[b] - ax, aby = P[b + 1] - ay, abz = P[b + 2] - az;
    const double acx = P[c] - ax, acy = P[c + 1] - ay, acz = P[c + 2] - az;
    const double apx = px - ax, apy = py - ay, apz = pz - az;
    const double d1 = abx * apx + aby * apy + abz * apz;
    const double d2 = acx * apx + acy * apy + acz * apz;
    const double bpx = px - P[b], bpy = py - P[b + 1], bpz = pz - P[b + 2];
    const double d3 = abx * bpx + aby * bpy + abz * bpz;
    const double d4 = acx * bpx + acy * bpy + acz * bpz;
    const double cpx = px - P[c], cpy = py - P[c + 1], cpz = pz - P[c + 2];
    const double d5 = abx * cpx + aby * cpy + abz * cpz;
    const double d6 = acx * cpx + acy * cpy + acz * cpz;
    const double vc = d1 * d4 - d3 * d2;
    const double vb = d5 * d2 - d1 * d6;
    const double va = d3 * d6 - d5 * d4;
    // weights of b and c; a gets the rest
    double v, w;
    if(d1 <= 0 && d2 <= 0)
      v = 0, w = 0;
    else if(d3 >= 0 && d4 <= d3)
      v = 1, w = 0;
    else if(vc <= 0 && d1 >= 0 && d3 <= 0)
      v = d1 / (d1 - d3), w = 0;
    else if(d6 >= 0 && d5 <= d6)
      v = 0, w = 1;
    else if(vb <= 0 && d2 >= 0 && d6 <= 0)
      v = 0, w = d2 / (d2 - d6);
    else if(va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
      w = (d4 - d3) / (d4 - d3 + (d5 - d6)), v = 1 - w;
    else
    {
      const double denom = 1 / (va + vb + vc);
      v                  = vb * denom;
      w                  = vc * denom;
    }
    bu = 1 - v - w;
    bv = v;
    bw = w;
    const double qx = ax + abx * v + acx * w - px;
    const double qy = ay + aby * v + acy * w - py;
    const double qz = az + abz * v + acz * w - pz;
    return qx * qx + qy * qy + qz * qz;
  }

  std::span<const float>    P;
  std::span<const uint32_t> T;
  const BindFilter*         filter;
  size_t                    nt;
  float                     minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0;
  double                    cell = 1;
  int                       nx = 1, ny = 1, nz = 1, maxR = 1;
  std::vector<uint32_t>     start, tris;
};

void write(SurfaceBinding& out, size_t v, std::span<const uint32_t> T, const SurfaceIndex::Hit& hit)
{
  out.verts[v * 3]       = T[size_t(hit.tri) * 3];
  out.verts[v * 3 + 1]   = T[size_t(hit.tri) * 3 + 1];
  out.verts[v * 3 + 2]   = T[size_t(hit.tri) * 3 + 2];
  out.weights[v * 3]     = float(hit.u);
  out.weights[v * 3 + 1] = float(hit.v);
  out.weights[v * 3 + 2] = float(hit.w);
}

// A geograft's seam pairs bound outright, then the body vertices any binding reads.
void finish(SurfaceBinding& out, std::span<const int32_t> pairs, size_t nvDst, size_t nvSrc)
{
  for(size_t k = 0; k + 1 < pairs.size(); k += 2)
  {
    const int32_t g = pairs[k], b = pairs[k + 1];
    if(g < 0 || size_t(g) >= nvDst || b < 0 || size_t(b) >= nvSrc)
      continue;
    out.verts[size_t(g) * 3] = out.verts[size_t(g) * 3 + 1] = out.verts[size_t(g) * 3 + 2] = uint32_t(b);
    out.weights[size_t(g) * 3]                                                          = 1;
    out.weights[size_t(g) * 3 + 1] = out.weights[size_t(g) * 3 + 2] = 0;
  }
  for(size_t i = 0; i < out.verts.size(); i++)
    if(out.weights[i] != 0)
      out.used[out.verts[i]] = 1;
}

SurfaceBinding blank(size_t nvDst, size_t nvSrc)
{
  SurfaceBinding out;
  out.verts.assign(nvDst * 3, 0);
  out.weights.assign(nvDst * 3, 0.f);
  out.used.assign(nvSrc, 0);
  return out;
}

}  // namespace

SurfaceBinding bindToSurface(std::span<const float>    P,
                             std::span<const uint32_t> T,
                             std::span<const float>    D,
                             std::span<const int32_t>  pairs,
                             const BindFilter*         filter)
{
  const size_t   nvDst = D.size() / 3;
  SurfaceBinding out   = blank(nvDst, P.size() / 3);
  const auto     tGrid = std::chrono::steady_clock::now();
  const SurfaceIndex index(P, T, filter);
  if(index.empty())
    return out;
  timing::report("grid", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tGrid).count(), 5);
  // each vertex is found on its own, so the vertices are split among threads
  const auto tSearch = std::chrono::steady_clock::now();
  jobs::parallelFor(nvDst, 4096, [&](size_t begin, size_t end) {
    std::vector<int32_t> stamp(index.triangles(), -1);
    for(size_t v = begin; v < end; v++)
    {
      const int32_t mask = filter && v < filter->itemMask.size() ? filter->itemMask[v] : 0;
      const auto    hit  = index.search(D[v * 3], D[v * 3 + 1], D[v * 3 + 2], mask, INFINITY, stamp, int32_t(v));
      if(hit.tri >= 0)
        write(out, v, T, hit);
    }
  });
  timing::report("search", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tSearch).count(), 5);
  finish(out, pairs, nvDst, P.size() / 3);
  return out;
}

SurfaceBinding bindStrands(std::span<const float>    P,
                           std::span<const uint32_t> T,
                           std::span<const float>    D,
                           std::span<const int32_t>  lineStart,
                           std::span<const int32_t>  lineVerts,
                           float                     reach,
                           const BindFilter*         filter)
{
  const size_t   nvDst = D.size() / 3;
  SurfaceBinding out   = blank(nvDst, P.size() / 3);
  const SurfaceIndex index(P, T, filter);
  if(index.empty())
    return out;
  const auto        tSearch = std::chrono::steady_clock::now();
  const double      reach2  = double(reach) * reach;
  const size_t      lines   = lineStart.empty() ? 0 : lineStart.size() - 1;
  std::vector<char> onLine(nvDst, 0);
  for(int32_t v : lineVerts)
    if(v >= 0 && size_t(v) < nvDst)
      onLine[size_t(v)] = 1;
  jobs::parallelFor(lines, 256, [&](size_t begin, size_t end) {
    std::vector<int32_t> stamp(index.triangles(), -1);
    int32_t              id = 0;
    for(size_t l = begin; l < end; l++)
    {
      SurfaceIndex::Hit above;
      for(int32_t c = lineStart[l]; c < lineStart[l + 1]; c++)
      {
        const int32_t v = lineVerts[size_t(c)];
        if(v < 0 || size_t(v) >= nvDst)
          continue;
        const int32_t mask = filter && size_t(v) < filter->itemMask.size() ? filter->itemMask[size_t(v)] : 0;
        // the root always binds; the rest bind where the body is within reach, and further out
        // ride with the point above them
        const bool root = above.tri < 0;
        const auto hit  = index.search(D[size_t(v) * 3], D[size_t(v) * 3 + 1], D[size_t(v) * 3 + 2], mask, root ? INFINITY : reach2, stamp, ++id);
        if(hit.tri >= 0)
          above = hit;
        if(above.tri >= 0)
          write(out, size_t(v), T, above);
      }
    }
  });
  // points on no line bind on their own
  jobs::parallelFor(nvDst, 4096, [&](size_t begin, size_t end) {
    std::vector<int32_t> stamp(index.triangles(), -1);
    for(size_t v = begin; v < end; v++)
    {
      if(onLine[v])
        continue;
      const int32_t mask = filter && v < filter->itemMask.size() ? filter->itemMask[v] : 0;
      const auto    hit  = index.search(D[v * 3], D[v * 3 + 1], D[v * 3 + 2], mask, INFINITY, stamp, int32_t(v));
      if(hit.tri >= 0)
        write(out, v, T, hit);
    }
  });
  timing::report("strand search", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tSearch).count(), 5);
  finish(out, {}, nvDst, P.size() / 3);
  return out;
}

}  // namespace fv::wardrobe
