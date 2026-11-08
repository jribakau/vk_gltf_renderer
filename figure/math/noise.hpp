#pragma once

// Value noise over space: a hashed lattice smoothly blended, so it has no seams on a surface whatever its
// UVs. The procedural layers break their fields up with it (materials/procedural.cpp), and rain's runs
// wander by it (physics/rain/rain_sim.cpp).

#include <cmath>
#include <cstdint>

namespace fv::noise {

inline double fade(double t)
{
  return t * t * (3 - 2 * t);
}

// A lattice point's value in [0, 1)
inline double lattice(int x, int y, int z)
{
  uint32_t h = uint32_t(x) * 0x8da6b343u ^ uint32_t(y) * 0xd8163841u ^ uint32_t(z) * 0xcb1ab31fu;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return double(h >> 8) / double(1u << 24);
}

// In [0, 1], one lattice cell a unit across
inline double noise3(double x, double y, double z)
{
  const double fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
  const int    ix = int(fx), iy = int(fy), iz = int(fz);
  const double sx = fade(x - fx), sy = fade(y - fy), sz = fade(z - fz);
  const auto   at  = [&](int dx, int dy, int dz) { return lattice(ix + dx, iy + dy, iz + dz); };
  const auto   mix = [](double a, double b, double t) { return a + (b - a) * t; };
  return mix(mix(mix(at(0, 0, 0), at(1, 0, 0), sx), mix(at(0, 1, 0), at(1, 1, 0), sx), sy),
             mix(mix(at(0, 0, 1), at(1, 0, 1), sx), mix(at(0, 1, 1), at(1, 1, 1), sx), sy), sz);
}

}  // namespace fv::noise
