#include <figure/wardrobe/fit/collide.hpp>

#include <algorithm>
#include <cmath>

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
