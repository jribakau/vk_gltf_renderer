#pragma once

// A point or vector of three floats, and what strand hair's step and the distance field do with them
// (physics-plan step 14). Positions elsewhere in the core are xyz floats in one array: pointAt reads one.

#include <cmath>
#include <vector>

namespace fv::physics {

struct P3
{
  float x = 0, y = 0, z = 0;
};
inline P3    operator+(P3 a, P3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline P3    operator-(P3 a, P3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline P3    operator*(P3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(P3 a, P3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline P3    cross(P3 a, P3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float len(P3 a) { return std::sqrt(dot(a, a)); }
inline P3    unit(P3 a, P3 fallback = {0, -1, 0})
{
  const float l = len(a);
  return l > 1e-6f ? a * (1 / l) : fallback;
}
inline float mix(float a, float b, float t) { return a + (b - a) * t; }
// `v` turned by the shortest rotation that takes unit `from` to unit `to`
inline P3 turned(P3 v, P3 from, P3 to)
{
  const float c = dot(from, to);
  if(c > 0.9999f)
    return v;
  if(c < -0.9999f)
    return v * -1;
  const P3    axis = cross(from, to);
  const float k    = 1 / (1 + c);
  return v * c + cross(axis, v) + axis * (dot(axis, v) * k);
}
// Point i of xyz floats
inline P3 pointAt(const std::vector<float>& a, int i)
{
  return {a[size_t(i) * 3], a[size_t(i) * 3 + 1], a[size_t(i) * 3 + 2]};
}

}  // namespace fv::physics
