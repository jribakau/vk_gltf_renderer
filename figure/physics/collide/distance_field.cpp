#include <figure/physics/collide/distance_field.hpp>

#include <cmath>
#include <cstdint>

#include <figure/jobs/pool.hpp>

namespace fv::physics {

namespace {

struct Face
{
  P3 a, b, c, n;
};

P3 closestOn(const Face& f, P3 p)
{
  // Ericson, Real-Time Collision Detection 5.1.5
  const P3    ab = f.b - f.a, ac = f.c - f.a, ap = p - f.a;
  const float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if(d1 <= 0 && d2 <= 0)
    return f.a;
  const P3    bp = p - f.b;
  const float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if(d3 >= 0 && d4 <= d3)
    return f.b;
  const float vc = d1 * d4 - d3 * d2;
  if(vc <= 0 && d1 >= 0 && d3 <= 0)
    return f.a + ab * (d1 / (d1 - d3));
  const P3    cp = p - f.c;
  const float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if(d6 >= 0 && d5 <= d6)
    return f.c;
  const float vb = d5 * d2 - d1 * d6;
  if(vb <= 0 && d2 >= 0 && d6 <= 0)
    return f.a + ac * (d2 / (d2 - d6));
  const float va = d3 * d6 - d5 * d4;
  if(va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    return f.b + (f.c - f.b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  const float denom = 1 / (va + vb + vc);
  return f.a + ab * (vb * denom) + ac * (vc * denom);
}

}  // namespace

DistanceField::DistanceField(const std::vector<cloth::Surface>& colliders, float offset, P3 from, P3 to)
{
  const float span = std::max({to.x - from.x, to.y - from.y, to.z - from.z, 0.f});
  // and coarser for an offset that large, which only meets broad shapes, as the band's cost goes
  // with its cube
  m_cell           = std::max({kCell, span / 192.f, offset / 4.f});
  // a point `offset` off a face or kDeepest behind it must find all eight of its corners known
  const float band  = std::max(offset, kDeepest) + m_cell * 1.74f;
  const float band2 = band * band;
  from              = from - P3{band, band, band};
  to                = to + P3{band, band, band};

  const auto lowest  = [](const Face& f) { return P3{std::min({f.a.x, f.b.x, f.c.x}), std::min({f.a.y, f.b.y, f.c.y}), std::min({f.a.z, f.b.z, f.c.z})}; };
  const auto highest = [](const Face& f) { return P3{std::max({f.a.x, f.b.x, f.c.x}), std::max({f.a.y, f.b.y, f.c.y}), std::max({f.a.z, f.b.z, f.c.z})}; };
  std::vector<Face> faces;
  P3                lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
  for(const cloth::Surface& m : colliders)
    for(size_t t = 0; t + 2 < m.tris.size(); t += 3)
    {
      if(!m.collide.empty() && (!m.collide[m.tris[t]] || !m.collide[m.tris[t + 1]] || !m.collide[m.tris[t + 2]]))
        continue;
      const P3 a = pointAt(m.posed, int(m.tris[t])), b = pointAt(m.posed, int(m.tris[t + 1])), c = pointAt(m.posed, int(m.tris[t + 2]));
      const P3 n = cross(b - a, c - a);
      if(len(n) < 1e-8f)
        continue;
      const Face f{a, b, c, unit(n)};
      const P3   l = lowest(f), h = highest(f);
      if(h.x + band < from.x || h.y + band < from.y || h.z + band < from.z || l.x - band > to.x || l.y - band > to.y || l.z - band > to.z)
        continue;
      faces.push_back(f);
      lo = {std::min(lo.x, l.x), std::min(lo.y, l.y), std::min(lo.z, l.z)};
      hi = {std::max(hi.x, h.x), std::max(hi.y, h.y), std::max(hi.z, h.z)};
    }
  if(faces.empty())
    return;
  m_lo = {std::max(from.x, lo.x - band), std::max(from.y, lo.y - band), std::max(from.z, lo.z - band)};
  hi   = {std::min(to.x, hi.x + band), std::min(to.y, hi.y + band), std::min(to.z, hi.z + band)};
  m_nx = int((hi.x - m_lo.x) / m_cell) + 2;
  m_ny = int((hi.y - m_lo.y) / m_cell) + 2;
  m_nz = int((hi.z - m_lo.z) / m_cell) + 2;
  m_d.assign(size_t(m_nx) * m_ny * m_nz, kUnset);

  // A layer of nodes at a time, each with the faces in reach of it, so threads share no node.
  const auto firstNode = [&](float v, float lowEnd, int n) { return std::clamp(int(std::ceil((v - band - lowEnd) / m_cell)), 0, n); };
  const auto lastNode  = [&](float v, float lowEnd, int n) { return std::clamp(int(std::floor((v + band - lowEnd) / m_cell)), -1, n - 1); };
  std::vector<std::vector<uint32_t>> layers(static_cast<size_t>(m_nz));
  for(uint32_t i = 0; i < faces.size(); i++)
    for(int z = firstNode(lowest(faces[i]).z, m_lo.z, m_nz); z <= lastNode(highest(faces[i]).z, m_lo.z, m_nz); z++)
      layers[size_t(z)].push_back(i);
  const size_t stride = size_t(m_nx) * m_ny;
  jobs::parallelFor(size_t(m_nz), 1, [&](size_t begin, size_t end) {
    // Which side of a shared edge or corner a node is on goes by the face turned most squarely to it:
    // the others can have it behind them from outside.
    std::vector<float> square;
    for(size_t z = begin; z < end; z++)
    {
      square.assign(stride, 0.f);
      float* layer = m_d.data() + stride * z;
      for(const uint32_t i : layers[z])
      {
        const Face& f = faces[i];
        const P3    l = lowest(f), h = highest(f);
        const int   y1 = lastNode(h.y, m_lo.y, m_ny), x0 = firstNode(l.x, m_lo.x, m_nx), x1 = lastNode(h.x, m_lo.x, m_nx);
        for(int y = firstNode(l.y, m_lo.y, m_ny); y <= y1; y++)
          for(int x = x0; x <= x1; x++)
          {
            const P3    node{m_lo.x + float(x) * m_cell, m_lo.y + float(y) * m_cell, m_lo.z + float(z) * m_cell};
            const P3    d  = node - closestOn(f, node);
            const float dd = dot(d, d);
            if(dd >= band2)
              continue;
            const size_t k    = size_t(x) + size_t(y) * size_t(m_nx);
            const float  seen = layer[k] * layer[k];
            const float  dist = std::sqrt(dd);
            const float  side = dot(d, f.n);
            const float  face = dist > 1e-5f ? std::fabs(side) / dist : 1.f;
            if(dd < seen * 0.999f || (dd <= seen * 1.001f && face > square[k]))
            {
              layer[k]  = side < 0 ? -dist : dist;
              square[k] = face;
            }
          }
      }
    }
  });
}

}  // namespace fv::physics
