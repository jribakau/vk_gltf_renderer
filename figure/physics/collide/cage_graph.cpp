#include <figure/physics/collide/cage_graph.hpp>

#include <algorithm>

namespace fv::physics {

CageGraph::CageGraph(std::span<const float> P, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts, std::span<const uint32_t> tris)
{
  const size_t n = P.size() / 3;
  // each edge once, from the loops the file has, else from the triangles
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  const auto addEdge = [&](int64_t a, int64_t b) {
    if(a != b && a >= 0 && b >= 0 && size_t(a) < n && size_t(b) < n)
      edges.emplace_back(uint32_t(std::min(a, b)), uint32_t(std::max(a, b)));
  };
  if(polyStart.size() > 1 && !polyVerts.empty())
    for(size_t p = 0; p + 1 < polyStart.size(); p++)
    {
      const int32_t s = polyStart[p], e = polyStart[p + 1];
      for(int32_t c = s; c < e; c++)
        addEdge(polyVerts[size_t(c)], polyVerts[size_t(c + 1 < e ? c + 1 : s)]);
    }
  else
    for(size_t t = 0; t + 2 < tris.size(); t += 3)
      for(int k = 0; k < 3; k++)
        addEdge(tris[t + size_t(k)], tris[t + size_t((k + 1) % 3)]);
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  m_start.assign(n + 1, 0);
  for(const auto& [a, b] : edges)
  {
    m_start[a + 1]++;
    m_start[b + 1]++;
  }
  for(size_t v = 0; v < n; v++)
    m_start[v + 1] += m_start[v];
  m_next.resize(m_start[n]);
  m_length.resize(m_start[n]);
  std::vector<uint32_t> fill(m_start.begin(), m_start.end() - 1);
  for(const auto& [a, b] : edges)
  {
    const float l = float(std::hypot(P[a * 3] - P[b * 3], P[a * 3 + 1] - P[b * 3 + 1], P[a * 3 + 2] - P[b * 3 + 2]));
    m_next[fill[a]]     = b;
    m_length[fill[a]++] = l;
    m_next[fill[b]]     = a;
    m_length[fill[b]++] = l;
  }

  m_area.assign(n, 0.0);
  m_normal.assign(n * 3, 0.0);
  for(size_t t = 0; t + 2 < tris.size(); t += 3)
  {
    const size_t a = tris[t], b = tris[t + 1], c = tris[t + 2];
    if(a >= n || b >= n || c >= n)
      continue;
    const double ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
    const double vx = P[c * 3] - P[a * 3], vy = P[c * 3 + 1] - P[a * 3 + 1], vz = P[c * 3 + 2] - P[a * 3 + 2];
    // the cross product is twice the area along the face's normal: area-weighted as it is
    const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const double area = 0.5 * std::hypot(nx, ny, nz);
    for(const size_t v : {a, b, c})
    {
      m_area[v] += area / 3;
      m_normal[v * 3] += nx;
      m_normal[v * 3 + 1] += ny;
      m_normal[v * 3 + 2] += nz;
    }
  }
  for(size_t v = 0; v < n; v++)
  {
    double*      nv  = &m_normal[v * 3];
    const double len = std::hypot(nv[0], nv[1], nv[2]);
    if(len < 1e-20)
    {
      nv[0] = 0, nv[1] = 1, nv[2] = 0;
      continue;
    }
    nv[0] /= len, nv[1] /= len, nv[2] /= len;
  }
}

}  // namespace fv::physics
