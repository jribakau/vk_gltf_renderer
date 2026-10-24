#include <figure/wardrobe/bind.hpp>

#include <figure/jobs/pool.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace fv::wardrobe {

SurfaceBinding bindToSurface(std::span<const float>    P,
                             std::span<const uint32_t> T,
                             std::span<const float>    D,
                             std::span<const int32_t>  pairs,
                             const BindFilter*         filter)
{
  const size_t nt     = T.size() / 3;
  const size_t nvDst  = D.size() / 3;
  const size_t nvSrc  = P.size() / 3;
  SurfaceBinding out;
  out.verts.assign(nvDst * 3, 0);
  out.weights.assign(nvDst * 3, 0.f);
  out.used.assign(nvSrc, 0);
  if(!nt)
    return out;

  float minX = INFINITY, minY = INFINITY, minZ = INFINITY, maxX = -INFINITY, maxY = -INFINITY, maxZ = -INFINITY;
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
  double       cell = std::cbrt(ex * ey * ez / double(nt)) * 1.5;
  cell              = std::max(cell, std::cbrt(ex * ey * ez / 4e6));
  const int nx = std::max(1, int(std::ceil(ex / cell))), ny = std::max(1, int(std::ceil(ey / cell))), nz = std::max(1, int(std::ceil(ez / cell)));
  const auto cellOf = [&](double v, double min, int n) { return std::min(n - 1, std::max(0, int(std::floor((v - min) / cell)))); };

  // triangles per cell, as compressed rows: count, prefix-sum, fill
  std::vector<uint32_t> start(size_t(nx) * ny * nz + 1, 0);
  std::vector<int>      range(nt * 6);
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
  std::vector<uint32_t> tris(start.back());
  for(size_t t = 0; t < nt; t++)
  {
    const int* r = &range[t * 6];
    for(int k = r[4]; k <= r[5]; k++)
      for(int j = r[2]; j <= r[3]; j++)
        for(int i = r[0]; i <= r[1]; i++)
          tris[fill[(size_t(k) * ny + j) * nx + i]++] = uint32_t(t);
  }

  // shaved off the cell-wall distance so float error cannot hide a strictly closer triangle on the face
  constexpr double kWallSlack = 1e-4;
  const int        maxR       = std::max({nx, ny, nz});
  const auto       triOk      = [&](size_t t, int32_t mask) {
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
  };
  // closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5), as barycentric u, v, w
  const auto closest = [&](double px, double py, double pz, size_t a, size_t b, size_t c, double& bu, double& bv, double& bw) {
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
  };

  // each vertex is found on its own, so the vertices are split among threads (strand hair has
  // hundreds of thousands, many far from the body, where the ring search goes wide); `stamp` marks
  // the triangles one vertex's search has seen, so each chunk has its own
  const auto bindRange = [&](size_t begin, size_t end) {
    std::vector<int32_t> stamp(nt, -1);
    double               bu = 0, bv = 0, bw = 0;
    for(size_t v = begin; v < end; v++)
    {
      const double  px = D[v * 3], py = D[v * 3 + 1], pz = D[v * 3 + 2];
      const int32_t mask = filter && v < filter->itemMask.size() ? filter->itemMask[v] : 0;
      const int     ci = cellOf(px, minX, nx), cj = cellOf(py, minY, ny), ck = cellOf(pz, minZ, nz);
      // clamped in from outside the grid: the cell does not contain the point, so its walls are not a bound
      const bool contained = int(std::floor((px - minX) / cell)) == ci && int(std::floor((py - minY) / cell)) == cj && int(std::floor((pz - minZ) / cell)) == ck;
      double     best      = INFINITY;
      int64_t    bestTri   = -1;
      double     wu = 1, wv = 0, ww = 0;
      for(int r = 0; r <= maxR; r++)
      {
        // cells r rings out are at least (r - 1) cells away
        if(bestTri >= 0 && r > 1 && best <= std::pow((r - 1) * cell, 2))
          break;
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
                if(stamp[t] == int32_t(v))
                  continue;
                stamp[t] = int32_t(v);
                if(!triOk(t, mask))
                  continue;
                const double d = closest(px, py, pz, T[t * 3] * 3ull, T[t * 3 + 1] * 3ull, T[t * 3 + 2] * 3ull, bu, bv, bw);
                if(d < best)
                {
                  best    = d;
                  bestTri = t;
                  wu = bu, wv = bv, ww = bw;
                }
              }
            }
          }
        }
        // ring 0 holds every triangle that meets the cell: a point inside it, closer to its hit than
        // to the nearest wall, has no strictly closer triangle outside it
        if(r == 0 && contained && bestTri >= 0)
        {
          const double dx   = std::min(px - (minX + ci * cell), minX + (ci + 1) * cell - px);
          const double dy   = std::min(py - (minY + cj * cell), minY + (cj + 1) * cell - py);
          const double dz   = std::min(pz - (minZ + ck * cell), minZ + (ck + 1) * cell - pz);
          const double wall = std::min({dx, dy, dz}) - kWallSlack;
          if(wall > 0 && best <= wall * wall)
            break;
        }
      }
      if(bestTri < 0)
        continue;
      out.verts[v * 3]       = T[bestTri * 3];
      out.verts[v * 3 + 1]   = T[bestTri * 3 + 1];
      out.verts[v * 3 + 2]   = T[bestTri * 3 + 2];
      out.weights[v * 3]     = float(wu);
      out.weights[v * 3 + 1] = float(wv);
      out.weights[v * 3 + 2] = float(ww);
    }
  };
  jobs::parallelFor(nvDst, 4096, bindRange);
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
  return out;
}

}  // namespace fv::wardrobe
