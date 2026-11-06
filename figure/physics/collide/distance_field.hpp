#pragma once

// A signed distance to a set of surfaces, baked once on a grid (physics-plan step 14): a shape any island
// can meet whose colliders hold still through its run. A look costs eight numbers however many faces are
// near, where hundreds of thousands of strand points a strand's worth of steps would look. Details under
// a cell go: hair rests on a shoulder, not an eyelash. Strand hair settles against one (physics/hair).

#include <algorithm>
#include <vector>

#include <figure/physics/cloth/surface.hpp>
#include <figure/physics/p3.hpp>

namespace fv::physics {

class DistanceField
{
public:
  // How deep behind a surface a point is still brought back from, cm
  static constexpr float kDeepest = 3.f;
  // The grid's spacing, cm, at its finest: finer than the hair's spans, and coarse enough to bake in a moment
  static constexpr float kCell = 1.5f;
  // In a node of the grid: too far from any face to say
  static constexpr float kUnset = 1e9f;

  DistanceField() = default;
  // The colliding faces of `surfaces` at their `posed` shape, for points kept up to `offset` off them in
  // the box from `from` to `to`.
  DistanceField(const std::vector<cloth::Surface>& surfaces, float offset, P3 from, P3 to);

  bool empty() const { return m_d.empty(); }

  // Keeps `p` `gap` off the surfaces. True on a contact, with `normal` the way it was pushed.
  bool pushOut(P3& p, float gap, P3& normal) const
  {
    if(m_d.empty())
      return false;
    const float fx = (p.x - m_lo.x) / m_cell, fy = (p.y - m_lo.y) / m_cell, fz = (p.z - m_lo.z) / m_cell;
    if(fx < 0 || fy < 0 || fz < 0 || fx >= float(m_nx - 1) || fy >= float(m_ny - 1) || fz >= float(m_nz - 1))
      return false;
    const int    ix = int(fx), iy = int(fy), iz = int(fz);
    const size_t sy = size_t(m_nx), sz = size_t(m_nx) * m_ny;
    const float* d  = m_d.data() + size_t(ix) + sy * iy + sz * iz;
    const float  c000 = d[0], c100 = d[1], c010 = d[sy], c110 = d[sy + 1];
    const float  c001 = d[sz], c101 = d[sz + 1], c011 = d[sz + sy], c111 = d[sz + sy + 1];
    if(std::max({c000, c100, c010, c110, c001, c101, c011, c111}) >= kUnset)
      return false;
    const float tx = fx - float(ix), ty = fy - float(iy), tz = fz - float(iz);
    const float x00 = mix(c000, c100, tx), x10 = mix(c010, c110, tx), x01 = mix(c001, c101, tx), x11 = mix(c011, c111, tx);
    const float y0 = mix(x00, x10, ty), y1 = mix(x01, x11, ty);
    const float dist = mix(y0, y1, tz);
    if(dist >= gap || dist <= -kDeepest)
      return false;
    const P3 slope{mix(mix(c100 - c000, c110 - c010, ty), mix(c101 - c001, c111 - c011, ty), tz), mix(x10 - x00, x11 - x01, tz), y1 - y0};
    if(len(slope) < 1e-5f)
      return false;
    normal = unit(slope);
    p      = p + normal * (gap - dist);
    return true;
  }

private:
  std::vector<float> m_d;
  P3                 m_lo;
  float              m_cell = kCell;
  int                m_nx = 0, m_ny = 0, m_nz = 0;
};

}  // namespace fv::physics
