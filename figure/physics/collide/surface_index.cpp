#include <figure/physics/collide/surface_index.hpp>

#include <limits>

namespace fv::physics {

int32_t Stamps::next(size_t triangles)
{
  if(stamp.size() < triangles || last == std::numeric_limits<int32_t>::max())
  {
    stamp.assign(std::max(stamp.size(), triangles), -1);
    last = 0;
  }
  return ++last;
}

Stamps& threadStamps()
{
  thread_local Stamps stamps;
  return stamps;
}

SurfaceIndex::SurfaceIndex(std::span<const float> P, std::span<const uint32_t> T)
    : P(P)
    , T(T)
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

SurfaceIndex::Hit SurfaceIndex::search(double px, double py, double pz, int32_t mask, std::span<const int16_t> limbs, double maxDist2, std::vector<int32_t>& stamp, int32_t id) const
{
  return nearest(px, py, pz, [&](uint32_t t) { return triOk(t, mask, limbs); }, maxDist2, stamp, id);
}

bool SurfaceIndex::triOk(size_t t, int32_t mask, std::span<const int16_t> limbs) const
{
  if(limbs.empty() || !mask)
    return true;
  for(int k = 0; k < 3; k++)
  {
    const uint32_t v    = T[t * 3 + k];
    const int      limb = v < limbs.size() ? limbs[v] : 0;
    if(!limb || (mask & (1 << limb)))
      return true;
  }
  return false;
}

double SurfaceIndex::closest(double px, double py, double pz, size_t a, size_t b, size_t c, double& bu, double& bv, double& bw) const
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

bool SurfaceIndex::hits(const double o[3], const double d[3], size_t t, double* at) const
{
  const size_t a = T[t * 3] * 3ull, b = T[t * 3 + 1] * 3ull, c = T[t * 3 + 2] * 3ull;
  const double e1[3] = {P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]};
  const double e2[3] = {P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]};
  const double p[3]  = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
  const double det   = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if(std::fabs(det) < 1e-18)
    return false;
  const double inv  = 1 / det;
  const double s[3] = {o[0] - P[a], o[1] - P[a + 1], o[2] - P[a + 2]};
  const double u    = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
  if(u < 0 || u > 1)
    return false;
  const double q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
  const double v    = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
  if(v < 0 || u + v > 1)
    return false;
  const double along = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
  if(at)
    *at = along;
  return along > 1e-9;
}

SurfaceIndex::RayHit SurfaceIndex::firstHit(const double o[3], const double d[3], double maxT, std::vector<int32_t>& stamp, int32_t id) const
{
  RayHit best;
  walk(o, d, [&](size_t cellIndex, double leaves) {
    for(uint32_t s = start[cellIndex], e = start[cellIndex + 1]; s < e; s++)
    {
      const uint32_t t = tris[s];
      if(stamp[t] == id)
        continue;
      stamp[t] = id;
      double at = INFINITY;
      if(hits(o, d, t, &at) && at <= maxT && at < best.t)
        best = {int64_t(t), at};
    }
    // a hit no further than where the ray leaves this cell: no later cell holds a nearer one
    return best.t > leaves && leaves <= maxT;
  });
  return best;
}

}  // namespace fv::physics
