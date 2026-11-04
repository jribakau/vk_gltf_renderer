#pragma once

// A cage as a graph of its edges, with each vertex's area and normal: what the contact bulge spreads
// over, by distance along the surface rather than through the air, so a dent on one thigh doesn't
// bulge the other across the gap between them. Flesh and cloth builds will share it.

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace fv::physics {

class CageGraph
{
public:
  // `positions` xyz per vertex; edges from the polygon loops (`polyStart`, `polyVerts`) when given,
  // else from the triangles, which add the diagonals of quads. Areas and normals come from `tris`.
  CageGraph(std::span<const float> positions, std::span<const int32_t> polyStart, std::span<const int32_t> polyVerts, std::span<const uint32_t> tris);

  size_t size() const { return m_area.size(); }
  // A third of the area of the triangles around the vertex.
  double area(size_t v) const { return m_area[v]; }
  // The area-weighted normal of those triangles, unit; up for a vertex on none.
  const double* normal(size_t v) const { return &m_normal[v * 3]; }
  // The vertices next to v, and how far each is.
  std::span<const uint32_t> next(size_t v) const { return {m_next.data() + m_start[v], m_start[v + 1] - m_start[v]}; }
  std::span<const float>    length(size_t v) const { return {m_length.data() + m_start[v], m_start[v + 1] - m_start[v]}; }

  // Each vertex `through` accepts within `reach` of the sources along the edges, and its distance,
  // nearest first; the sources themselves at 0. A source is reached whatever `through` says.
  template <class Through>
  std::vector<std::pair<uint32_t, float>> within(std::span<const uint32_t> sources, float reach, Through&& through) const;

private:
  std::vector<uint32_t> m_start, m_next;
  std::vector<float>    m_length;
  std::vector<double>   m_area, m_normal;
  // Dijkstra's distances, kept between calls: only what a call reached is put back
  mutable std::vector<float> m_dist;
};

}  // namespace fv::physics

#include <algorithm>
#include <cmath>
#include <queue>

namespace fv::physics {

template <class Through>
std::vector<std::pair<uint32_t, float>> CageGraph::within(std::span<const uint32_t> sources, float reach, Through&& through) const
{
  if(m_dist.size() != size())
    m_dist.assign(size(), INFINITY);
  using Step = std::pair<float, uint32_t>;
  std::priority_queue<Step, std::vector<Step>, std::greater<Step>> open;
  std::vector<std::pair<uint32_t, float>>                          out;
  for(uint32_t s : sources)
    if(s < size() && m_dist[s] > 0)
    {
      m_dist[s] = 0;
      open.push({0.f, s});
    }
  while(!open.empty())
  {
    const auto [d, v] = open.top();
    open.pop();
    if(d > m_dist[v])
      continue;
    out.emplace_back(v, d);
    const auto n = next(v);
    const auto l = length(v);
    for(size_t k = 0; k < n.size(); k++)
    {
      const uint32_t u  = n[k];
      const float    du = d + l[k];
      if(du <= reach && du < m_dist[u] && through(u))
      {
        m_dist[u] = du;
        open.push({du, u});
      }
    }
  }
  // each vertex pushed is settled once (a push needs a strictly shorter way); put back what was touched
  for(const auto& [v, d] : out)
    m_dist[v] = INFINITY;
  return out;
}

}  // namespace fv::physics
