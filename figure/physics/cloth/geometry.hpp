#pragma once

// The cloth island's own small geometry, shared by its build (the start's separation) and its step.
// Internal to physics/cloth: what the core shares lives in physics/collide.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <figure/math/xform.hpp>

namespace fv::physics::cloth {

// A vertex further behind a face than this is left where it is (it came from the other side).
constexpr float kDepth = 2.f;

using V3 = std::array<float, 3>;

// A 3x4 float matrix (column-major, 3 rows), for skinning in bulk.
struct Mat34
{
  float m[12];
  static Mat34 of(const Mat4& a)
  {
    Mat34 o;
    for(int c = 0; c < 4; c++)
      for(int r = 0; r < 3; r++)
        o.m[c * 3 + r] = float(a.e[c * 4 + r]);
    return o;
  }
};

inline bool triNormal(const std::vector<float>& p, int a, int b, int c, V3& n)
{
  const float ax = p[b * 3] - p[a * 3], ay = p[b * 3 + 1] - p[a * 3 + 1], az = p[b * 3 + 2] - p[a * 3 + 2];
  const float bx = p[c * 3] - p[a * 3], by = p[c * 3 + 1] - p[a * 3 + 1], bz = p[c * 3 + 2] - p[a * 3 + 2];
  const float x = ay * bz - az * by, y = az * bx - ax * bz, z = ax * by - ay * bx;
  const float len = std::sqrt(x * x + y * y + z * z);
  if(len < 1e-8f)
    return false;
  n = {x / len, y / len, z / len};
  return true;
}

struct Closest
{
  float signedDistance, dist;
  // the vertex projects onto the face: it rests on the face, and a face holding it keeps it while
  // this holds (a vertex crossing an edge is held by one face or the next all the way)
  bool over;
};

// Closest point of `q` on the triangle in `tri`, and the signed distance along `n`.
inline Closest closest(float qx, float qy, float qz, const std::vector<float>& tri, int a, int b, int c, const V3& n)
{
  const float abx = tri[b * 3] - tri[a * 3], aby = tri[b * 3 + 1] - tri[a * 3 + 1], abz = tri[b * 3 + 2] - tri[a * 3 + 2];
  const float acx = tri[c * 3] - tri[a * 3], acy = tri[c * 3 + 1] - tri[a * 3 + 1], acz = tri[c * 3 + 2] - tri[a * 3 + 2];
  const float aqx = qx - tri[a * 3], aqy = qy - tri[a * 3 + 1], aqz = qz - tri[a * 3 + 2];
  const float d00 = abx * abx + aby * aby + abz * abz;
  const float d01 = abx * acx + aby * acy + abz * acz;
  const float d11 = acx * acx + acy * acy + acz * acz;
  const float d20 = abx * aqx + aby * aqy + abz * aqz;
  const float d21 = acx * aqx + acy * aqy + acz * aqz;
  const float den = d00 * d11 - d01 * d01;
  float       bv = 0, cv = 0;
  bool        over = false;
  if(den > 1e-12f)
  {
    bv       = (d11 * d20 - d01 * d21) / den;
    cv       = (d00 * d21 - d01 * d20) / den;
    over     = bv >= -1e-3f && cv >= -1e-3f && bv + cv <= 1 + 1e-3f;
    bv       = std::min(1.f, std::max(0.f, bv));
    cv       = std::min(1.f, std::max(0.f, cv));
    if(bv + cv > 1)
    {
      const float s = bv + cv;
      bv /= s;
      cv /= s;
    }
  }
  const float cx = tri[a * 3] + abx * bv + acx * cv;
  const float cy = tri[a * 3 + 1] + aby * bv + acy * cv;
  const float cz = tri[a * 3 + 2] + abz * bv + acz * cv;
  const float dx = qx - cx, dy = qy - cy, dz = qz - cz;
  return {dx * n[0] + dy * n[1] + dz * n[2], std::sqrt(dx * dx + dy * dy + dz * dz), over};
}

}  // namespace fv::physics::cloth
