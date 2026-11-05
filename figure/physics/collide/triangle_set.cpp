#include <figure/physics/collide/triangle_set.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace fv::physics {

void Box::add(double x, double y, double z)
{
  min = {std::min(min[0], x), std::min(min[1], y), std::min(min[2], z)};
  max = {std::max(max[0], x), std::max(max[1], y), std::max(max[2], z)};
}

void Box::add(const Box& b)
{
  if(b.empty())
    return;
  add(b.min[0], b.min[1], b.min[2]);
  add(b.max[0], b.max[1], b.max[2]);
}

Box Box::grown(double by) const
{
  if(empty())
    return *this;
  Box out = *this;
  for(int a = 0; a < 3; a++)
  {
    out.min[a] -= by;
    out.max[a] += by;
  }
  return out;
}

bool Box::overlaps(const Box& b) const
{
  if(empty() || b.empty())
    return false;
  for(int a = 0; a < 3; a++)
    if(max[a] < b.min[a] || b.max[a] < min[a])
      return false;
  return true;
}

bool Box::contains(double x, double y, double z) const
{
  return x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1] && z >= min[2] && z <= max[2];
}

Box transformed(const Box& b, const Mat4& m)
{
  Box out;
  if(b.empty())
    return out;
  for(int c = 0; c < 8; c++)
  {
    const Vec3 p = m.transformPoint({c & 1 ? b.max[0] : b.min[0], c & 2 ? b.max[1] : b.min[1], c & 4 ? b.max[2] : b.min[2]});
    out.add(p.x, p.y, p.z);
  }
  return out;
}

Box boundsOf(std::span<const float> points)
{
  Box out;
  for(size_t i = 0; i + 2 < points.size(); i += 3)
    out.add(points[i], points[i + 1], points[i + 2]);
  return out;
}

namespace {

uint32_t find(std::vector<uint32_t>& parent, uint32_t a)
{
  while(parent[a] != a)
  {
    parent[a] = parent[parent[a]];
    a         = parent[a];
  }
  return a;
}

}  // namespace

TriangleSet::TriangleSet(std::vector<float> positions, std::vector<uint32_t> tris)
    : m_positions(std::move(positions))
    , m_tris(std::move(tris))
{
  const size_t nv = m_positions.size() / 3;
  // a triangle naming a vertex the set lacks is left out
  std::vector<uint32_t> kept;
  kept.reserve(m_tris.size());
  for(size_t t = 0; t + 2 < m_tris.size(); t += 3)
    if(m_tris[t] < nv && m_tris[t + 1] < nv && m_tris[t + 2] < nv)
      kept.insert(kept.end(), {m_tris[t], m_tris[t + 1], m_tris[t + 2]});
  m_tris   = std::move(kept);
  m_bounds = boundsOf(m_positions);
  m_index  = SurfaceIndex(m_positions, m_tris);

  // vertices welded by position, as a file splits them at a UV seam or a hard edge: a closed box
  // stays closed
  const size_t nt   = m_tris.size() / 3;
  const double size = m_bounds.empty() ? 1 : std::max({m_bounds.max[0] - m_bounds.min[0], m_bounds.max[1] - m_bounds.min[1], m_bounds.max[2] - m_bounds.min[2], 1e-6});
  const double q    = size * 1e-6;
  std::vector<uint32_t>                  weld(nv);
  std::unordered_map<uint64_t, uint32_t> at;
  at.reserve(nv);
  for(size_t v = 0; v < nv; v++)
  {
    const auto     key = [&](int a) { return uint64_t(int64_t(std::llround((m_positions[v * 3 + a] - m_bounds.min[a]) / q)) & 0x1fffff); };
    const uint64_t k   = (key(0) << 42) | (key(1) << 21) | key(2);
    weld[v]            = at.try_emplace(k, uint32_t(v)).first->second;
  }
  // pieces through shared edges, and how many triangles share each edge
  std::vector<uint32_t> parent(nt);
  std::iota(parent.begin(), parent.end(), 0u);
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> edges;  // edge -> (first triangle, count)
  edges.reserve(nt * 2);
  for(size_t t = 0; t < nt; t++)
    for(int e = 0; e < 3; e++)
    {
      const uint32_t a = weld[m_tris[t * 3 + e]], b = weld[m_tris[t * 3 + (e + 1) % 3]];
      if(a == b)
        continue;
      const uint64_t key = (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
      auto [it, made]    = edges.try_emplace(key, uint32_t(t), 0u);
      it->second.second++;
      if(!made)
        parent[find(parent, uint32_t(t))] = find(parent, it->second.first);
    }
  m_piece.resize(nt);
  std::unordered_map<uint32_t, uint32_t> pieceOfRoot;
  for(size_t t = 0; t < nt; t++)
    m_piece[t] = pieceOfRoot.try_emplace(find(parent, uint32_t(t)), uint32_t(pieceOfRoot.size())).first->second;
  m_closed.assign(pieceOfRoot.size(), 1);
  for(const auto& [key, use] : edges)
    if(use.second != 2)
      m_closed[m_piece[use.first]] = 0;
}

SharedTriangles::SharedTriangles(std::function<std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>()> read, Box bounds)
    : m_read(std::move(read))
    , m_bounds(bounds)
{
}

SharedTriangles::SharedTriangles(std::shared_ptr<const TriangleSet> made)
    : m_bounds(made ? made->bounds() : Box{})
    , m_set(std::move(made))
{
  std::call_once(m_once, [] {});
  m_made.store(true, std::memory_order_release);
}

const TriangleSet& SharedTriangles::get() const
{
  std::call_once(m_once, [this] {
    std::vector<float>    positions;
    std::vector<uint32_t> tris;
    if(m_read)
      for(auto& [p, t] : m_read())
      {
        const uint32_t offset = uint32_t(positions.size() / 3);
        positions.insert(positions.end(), p.begin(), p.end());
        for(uint32_t i : t)
          tris.push_back(i + offset);
      }
    m_set = std::make_shared<const TriangleSet>(std::move(positions), std::move(tris));
    m_made.store(true, std::memory_order_release);
  });
  return *m_set;
}

Inside insideOf(const TriangleSet& set, const std::array<double, 3>& p, double depth, double closedDepth, Stamps& stamps)
{
  Inside             out;
  const SurfaceIndex& index = set.index();
  if(index.empty())
    return out;
  const double reach = std::max(depth, closedDepth);
  const auto   hit   = index.search(p[0], p[1], p[2], 0, {}, reach * reach, stamps.stamp, stamps.next(index.triangles()));
  if(hit.tri < 0)
    return out;
  const auto     P = set.positions();
  const auto     T = set.tris();
  const size_t   t = size_t(hit.tri);
  const size_t   a = T[t * 3] * 3ull, b = T[t * 3 + 1] * 3ull, c = T[t * 3 + 2] * 3ull;
  std::array<double, 3> q, n;
  for(int k = 0; k < 3; k++)
    q[size_t(k)] = P[a + k] * hit.u + P[b + k] * hit.v + P[c + k] * hit.w;
  {
    const double ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const double vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    n                = {uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx};
    const double len = std::hypot(n[0], n[1], n[2]);
    if(len < 1e-20)
      return out;
    for(double& x : n)
      x /= len;
  }
  const std::array<double, 3> d = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  const double                dist = std::hypot(d[0], d[1], d[2]);
  if(set.closed(t))
  {
    if(dist > closedDepth)
      return out;
    // two rays, either way along a direction no axis-aligned prop lines up with: odd on both is inside
    const uint32_t piece   = set.pieceOf(t);
    const auto     onPiece = [&](uint32_t tri) { return set.pieceOf(tri) == piece; };
    const double   dir[3]  = {0.57735, 0.57746, 0.57724};
    const double   back[3] = {-dir[0], -dir[1], -dir[2]};
    const double   o[3]    = {p[0], p[1], p[2]};
    const int      ahead   = index.crossings(o, dir, onPiece, stamps.stamp, stamps.next(index.triangles()));
    const int      behind  = index.crossings(o, back, onPiece, stamps.stamp, stamps.next(index.triangles()));
    if(ahead % 2 == 0 || behind % 2 == 0)
      return out;
    // out through the closest point: away from p, whichever way the file winds its faces
    if(dist > 1e-9)
      n = {-d[0] / dist, -d[1] / dist, -d[2] / dist};
    out = {true, q, n, dist};
    return out;
  }
  // open: behind the face, and not mostly sideways from it (past its edge), as pushOut has it
  const double signedDist = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
  if(dist > depth || signedDist >= 0 || -signedDist < 0.6 * dist)
    return out;
  out = {true, q, n, dist};
  return out;
}

Inside exitAlong(const TriangleSet& set, const std::array<double, 3>& p, const std::array<double, 3>& dir, double reach, Stamps& stamps)
{
  Inside             out;
  const SurfaceIndex& index = set.index();
  const double        o[3]  = {p[0], p[1], p[2]};
  const double        d[3]  = {dir[0], dir[1], dir[2]};
  const auto          hit   = index.firstHit(o, d, reach, stamps.stamp, stamps.next(index.triangles()));
  if(hit.tri < 0)
    return out;
  const auto   P = set.positions();
  const auto   T = set.tris();
  const size_t t = size_t(hit.tri);
  const size_t a = T[t * 3] * 3ull, b = T[t * 3 + 1] * 3ull, c = T[t * 3 + 2] * 3ull;
  const double ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
  const double vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
  std::array<double, 3> n   = {uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx};
  const double          len = std::hypot(n[0], n[1], n[2]);
  if(len < 1e-20)
    return out;
  const double s = (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) < 0 ? -1 / len : 1 / len;
  for(double& x : n)
    x *= s;
  out = {true, {p[0] + d[0] * hit.t, p[1] + d[1] * hit.t, p[2] + d[2] * hit.t}, n, hit.t};
  return out;
}

}  // namespace fv::physics
