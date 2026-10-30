#include <figure/wardrobe/fit/collide.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <queue>
#include <unordered_map>
#include <utility>

namespace fv::wardrobe {

V3 triNormal(std::span<const float> p, size_t a, size_t b, size_t c)
{
  const double ax = p[b * 3] - p[a * 3], ay = p[b * 3 + 1] - p[a * 3 + 1], az = p[b * 3 + 2] - p[a * 3 + 2];
  const double bx = p[c * 3] - p[a * 3], by = p[c * 3 + 1] - p[a * 3 + 1], bz = p[c * 3 + 2] - p[a * 3 + 2];
  const double x = ay * bz - az * by, y = az * bx - ax * bz, z = ax * by - ay * bx;
  const double len = std::hypot(x, y, z);
  if(len < 1e-8)
    return {0, 1, 0};
  return {x / len, y / len, z / len};
}

std::vector<float> pushOut(std::span<const float> positions, const Surface& body, const SurfaceBinding& binding, float margin, float depth)
{
  std::vector<float> out(positions.begin(), positions.end());
  const auto         p   = body.positions;
  const size_t       n   = positions.size() / 3;
  const double       cap = 2;
  for(size_t v = 0; v < n; v++)
  {
    const size_t k = v * 3;
    const double w = double(binding.weights[k]) + binding.weights[k + 1] + binding.weights[k + 2];
    if(w <= 0)
      continue;
    const size_t ia = binding.verts[k], ib = binding.verts[k + 1], ic = binding.verts[k + 2];
    const double wa = binding.weights[k], wb = binding.weights[k + 1], wc = binding.weights[k + 2];
    const double sx = p[ia * 3] * wa + p[ib * 3] * wb + p[ic * 3] * wc;
    const double sy = p[ia * 3 + 1] * wa + p[ib * 3 + 1] * wb + p[ic * 3 + 1] * wc;
    const double sz = p[ia * 3 + 2] * wa + p[ib * 3 + 2] * wb + p[ic * 3 + 2] * wc;
    const V3     nrm = triNormal(p, ia, ib, ic);
    const double dx = positions[k] - sx, dy = positions[k + 1] - sy, dz = positions[k + 2] - sz;
    const double signedDist = dx * nrm[0] + dy * nrm[1] + dz * nrm[2];
    if(signedDist >= margin)
      continue;
    const double dist = std::hypot(dx, dy, dz);
    if(dist > depth)
      continue;
    // mostly sideways from the closest point: past an edge, not under the face
    if(signedDist < 0 && -signedDist < 0.6 * dist)
      continue;
    const double push = std::min(margin - signedDist, cap);
    out[k]     = float(out[k] + nrm[0] * push);
    out[k + 1] = float(out[k + 1] + nrm[1] * push);
    out[k + 2] = float(out[k + 2] + nrm[2] * push);
  }
  return out;
}

std::vector<float> pullIn(std::span<const float> positions, const Surface& body, const SurfaceBinding& binding, float margin, float amount, float reach)
{
  std::vector<float> out(positions.begin(), positions.end());
  const auto         p = body.positions;
  const size_t       n = positions.size() / 3;
  for(size_t v = 0; v < n; v++)
  {
    const size_t k = v * 3;
    const double w = double(binding.weights[k]) + binding.weights[k + 1] + binding.weights[k + 2];
    if(w <= 0)
      continue;
    const size_t ia = binding.verts[k], ib = binding.verts[k + 1], ic = binding.verts[k + 2];
    const double wa = binding.weights[k], wb = binding.weights[k + 1], wc = binding.weights[k + 2];
    const double sx = p[ia * 3] * wa + p[ib * 3] * wb + p[ic * 3] * wc;
    const double sy = p[ia * 3 + 1] * wa + p[ib * 3 + 1] * wb + p[ic * 3 + 1] * wc;
    const double sz = p[ia * 3 + 2] * wa + p[ib * 3 + 2] * wb + p[ic * 3 + 2] * wc;
    const V3     nrm = triNormal(p, ia, ib, ic);
    const double dx = positions[k] - sx, dy = positions[k + 1] - sy, dz = positions[k + 2] - sz;
    const double signedDist = dx * nrm[0] + dy * nrm[1] + dz * nrm[2];
    const double dist       = std::hypot(dx, dy, dz);
    // at the clearance or under it is pushOut's; mostly sideways is past an edge, not over the face
    if(signedDist <= margin || dist > reach || signedDist < 0.6 * dist)
      continue;
    const double fade = std::clamp((reach - dist) / (0.5 * reach), 0.0, 1.0);
    const double move = amount * fade * fade * (3 - 2 * fade);
    out[k]     = float(positions[k] + (sx + nrm[0] * margin - positions[k]) * move);
    out[k + 1] = float(positions[k + 1] + (sy + nrm[1] * margin - positions[k + 1]) * move);
    out[k + 2] = float(positions[k + 2] + (sz + nrm[2] * margin - positions[k + 2]) * move);
  }
  return out;
}

std::vector<int32_t> coveredPolys(std::span<const float>    body,
                                  std::span<const int32_t>  polyStart,
                                  std::span<const int32_t>  polyVerts,
                                  std::span<const float>    item,
                                  std::span<const uint32_t> itemTris,
                                  float                     reach,
                                  float                     hemClear)
{
  const size_t items = item.size() / 3;
  std::vector<int32_t> out;
  if(!items || itemTris.empty())
    return out;
  // how far along the garment each of its vertices is from an open edge, as far as hemClear
  std::map<std::pair<uint32_t, uint32_t>, int>     edges;
  for(size_t t = 0; t + 2 < itemTris.size(); t += 3)
    for(size_t e = 0; e < 3; e++)
    {
      const uint32_t a = itemTris[t + e], b = itemTris[t + (e + 1) % 3];
      edges[{std::min(a, b), std::max(a, b)}]++;
    }
  std::vector<std::vector<std::pair<uint32_t, float>>> next(items);
  using Step = std::pair<float, uint32_t>;
  std::priority_queue<Step, std::vector<Step>, std::greater<Step>> open;
  std::vector<float> fromHem(items, hemClear);
  for(const auto& [edge, count] : edges)
  {
    const auto [a, b]  = edge;
    const float length = float(std::hypot(item[a * 3] - item[b * 3], item[a * 3 + 1] - item[b * 3 + 1], item[a * 3 + 2] - item[b * 3 + 2]));
    next[a].push_back({b, length});
    next[b].push_back({a, length});
    if(count == 1)
      for(const uint32_t v : {a, b})
        if(fromHem[v] > 0)
        {
          fromHem[v] = 0;
          open.push({0.f, v});
        }
  }
  while(!open.empty())
  {
    const auto [d, v] = open.top();
    open.pop();
    if(d > fromHem[v])
      continue;
    for(const auto& [u, length] : next[v])
      if(d + length < fromHem[u])
      {
        fromHem[u] = d + length;
        open.push({fromHem[u], u});
      }
  }

  // the garment's vertices by cell, and its smooth normals (by face area): a body vertex looks only at
  // what lies within reach, so skin far from the fabric costs nothing
  std::vector<float> normal(item.size(), 0.f);
  for(size_t t = 0; t + 2 < itemTris.size(); t += 3)
  {
    const size_t a = itemTris[t], b = itemTris[t + 1], c = itemTris[t + 2];
    const float  ux = item[b * 3] - item[a * 3], uy = item[b * 3 + 1] - item[a * 3 + 1], uz = item[b * 3 + 2] - item[a * 3 + 2];
    const float  wx = item[c * 3] - item[a * 3], wy = item[c * 3 + 1] - item[a * 3 + 1], wz = item[c * 3 + 2] - item[a * 3 + 2];
    const float  nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    for(const size_t v : {a, b, c})
    {
      normal[v * 3] += nx;
      normal[v * 3 + 1] += ny;
      normal[v * 3 + 2] += nz;
    }
  }
  const auto cell = [reach](float x) { return int32_t(std::floor(x / reach)); };
  const auto key  = [](int32_t x, int32_t y, int32_t z) { return (int64_t(x & 0x1fffff) << 42) | (int64_t(y & 0x1fffff) << 21) | int64_t(z & 0x1fffff); };
  std::unordered_map<int64_t, std::vector<uint32_t>> grid;
  for(size_t v = 0; v < items; v++)
    grid[key(cell(item[v * 3]), cell(item[v * 3 + 1]), cell(item[v * 3 + 2]))].push_back(uint32_t(v));

  const size_t         bodies = body.size() / 3;
  std::vector<uint8_t> under(bodies, 0);
  for(size_t v = 0; v < bodies; v++)
  {
    const float  x = body[v * 3], y = body[v * 3 + 1], z = body[v * 3 + 2];
    const int32_t cx = cell(x), cy = cell(y), cz = cell(z);
    float        best = reach * reach;
    int64_t      near = -1;
    for(int32_t i = cx - 1; i <= cx + 1; i++)
      for(int32_t j = cy - 1; j <= cy + 1; j++)
        for(int32_t k = cz - 1; k <= cz + 1; k++)
        {
          const auto found = grid.find(key(i, j, k));
          if(found == grid.end())
            continue;
          for(const uint32_t u : found->second)
          {
            const float dx = x - item[u * 3], dy = y - item[u * 3 + 1], dz = z - item[u * 3 + 2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            if(d2 <= best)
            {
              best = d2;
              near = u;
            }
          }
        }
    // the closest of the fabric is clear of its hems, and the skin is behind it, or just out through it
    if(near < 0 || fromHem[size_t(near)] < hemClear)
      continue;
    const size_t u   = size_t(near);
    const double len = std::hypot(normal[u * 3], normal[u * 3 + 1], normal[u * 3 + 2]);
    if(len <= 0)
      continue;
    const double behind = -((x - item[u * 3]) * normal[u * 3] + (y - item[u * 3 + 1]) * normal[u * 3 + 1] + (z - item[u * 3 + 2]) * normal[u * 3 + 2]) / len;
    if(behind > -0.2)
      under[v] = 1;
  }
  for(size_t p = 0; p + 1 < polyStart.size(); p++)
  {
    bool all = polyStart[p + 1] > polyStart[p];
    for(int32_t c = polyStart[p]; c < polyStart[p + 1] && all; c++)
      all = under[size_t(polyVerts[size_t(c)])] != 0;
    if(all)
      out.push_back(int32_t(p));
  }
  return out;
}

std::vector<float> smoothCorrection(std::span<const float> prev, std::span<const float> next, std::span<const uint32_t> tris, int passes)
{
  std::vector<float> corr(prev.size());
  for(size_t i = 0; i < corr.size(); i++)
    corr[i] = next[i] - prev[i];
  const std::vector<float> smooth = laplacian(std::move(corr), tris, prev.size() / 3, passes);
  std::vector<float>       out(prev.size());
  for(size_t i = 0; i < out.size(); i++)
    out[i] = prev[i] + smooth[i];
  return out;
}

}  // namespace fv::wardrobe
