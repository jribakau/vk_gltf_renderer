#include <figure/garment/rigid.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

#include <figure/bodymap/clone.hpp>

namespace fv::garment {

Mat4 RigidMove::matrix() const
{
  // to + turn (p - from): the shift to `to`, the turn, the shift back from `from`
  const Vec3 back = to - rotate(from, turn);
  return compose(back, turn, {1, 1, 1});
}

std::optional<RigidMove> bestMove(const std::vector<Vec3>& from, const std::vector<Vec3>& to)
{
  const size_t n = std::min(from.size(), to.size());
  if(n == 0)
    return std::nullopt;
  RigidMove m;
  for(size_t i = 0; i < n; i++)
  {
    m.from = m.from + from[i];
    m.to   = m.to + to[i];
  }
  m.from = m.from * (1.0 / double(n));
  m.to   = m.to * (1.0 / double(n));
  if(n < 4)
    return m;
  std::vector<Vec3> a(n), b(n);
  for(size_t i = 0; i < n; i++)
  {
    a[i] = from[i] - m.from;
    b[i] = to[i] - m.to;
  }
  const Quat   q       = bodymap::bestTurn(a, b);
  const double degrees = 2 * std::acos(std::clamp(std::abs(q.w), 0.0, 1.0)) * 180 / std::numbers::pi;
  if(degrees <= kRigidMaxTurn)
    m.turn = q;
  return m;
}

Near nearVertices(std::span<const float> surface, std::span<const float> part, double reach, size_t samples)
{
  Near         out;
  const size_t parts = part.size() / 3, verts = surface.size() / 3;
  if(!parts || !verts)
    return out;
  const size_t step = std::max<size_t>(1, parts / std::max<size_t>(samples, 1));
  // the part's box grown by the reach: a vertex outside it is never near
  float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
  for(size_t p = 0; p < parts; p++)
    for(size_t k = 0; k < 3; k++)
      lo[k] = std::min(lo[k], part[p * 3 + k]), hi[k] = std::max(hi[k], part[p * 3 + k]);
  double least2 = -1;
  for(size_t v = 0; v < verts; v++)
  {
    const float* s = &surface[v * 3];
    if(s[0] < lo[0] - reach || s[0] > hi[0] + reach || s[1] < lo[1] - reach || s[1] > hi[1] + reach || s[2] < lo[2] - reach || s[2] > hi[2] + reach)
      continue;
    double best = 1e30;
    for(size_t p = 0; p < parts; p += step)
    {
      const double d2 = double(s[0] - part[p * 3]) * (s[0] - part[p * 3]) + double(s[1] - part[p * 3 + 1]) * (s[1] - part[p * 3 + 1])
                        + double(s[2] - part[p * 3 + 2]) * (s[2] - part[p * 3 + 2]);
      best = std::min(best, d2);
    }
    if(best <= reach * reach)
      out.vertices.push_back(v);
    least2 = least2 < 0 ? best : std::min(least2, best);
  }
  out.least = least2 < 0 ? -1 : std::sqrt(least2);
  return out;
}

std::optional<Vec3> onFigure(const bodymap::TypeMap& source, const bodymap::TypeMap& target, std::span<const float> figure, size_t v)
{
  const auto& to   = source.map.toReference;
  const auto& back = target.map.fromReference;
  if(v >= to.mapped.size() || !to.mapped[v] || to.at.verts.size() < (v + 1) * 3)
    return std::nullopt;
  Vec3   p;
  double sum = 0;
  for(size_t i = 0; i < 3; i++)
  {
    const uint32_t r = to.at.verts[v * 3 + i];
    const double   w = to.at.weights[v * 3 + i];
    if(w <= 0 || r >= back.mapped.size() || !back.mapped[r] || back.at.verts.size() < (size_t(r) + 1) * 3)
      continue;
    for(size_t j = 0; j < 3; j++)
    {
      const uint32_t c = back.at.verts[size_t(r) * 3 + j];
      const double   u = back.at.weights[size_t(r) * 3 + j];
      if(u <= 0 || size_t(c) * 3 + 2 >= figure.size())
        continue;
      p = p + Vec3{figure[size_t(c) * 3], figure[size_t(c) * 3 + 1], figure[size_t(c) * 3 + 2]} * (w * u);
      sum += w * u;
    }
  }
  if(sum <= 0)
    return std::nullopt;
  return p * (1.0 / sum);
}

}  // namespace fv::garment
