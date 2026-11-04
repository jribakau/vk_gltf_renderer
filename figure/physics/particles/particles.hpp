#pragma once

// The particle core (docs/physics-plan.md step 7): what cloth is solved with, and strand hair from
// step 14. Positions are xyz floats in one array, inverse masses one float each (0: held by its
// owner), and each row's lambda is a double, so a long settle's sums don't drift.
//
// A sweep runs blocks of the particle range side by side. The block count is fixed, never the
// machine's, so a drape comes out the same on any number of cores.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include <figure/jobs/pool.hpp>

namespace fv::physics::particles {

// One XPBD distance row between particles a and b, toward `rest`, as compliant as `alphaAt(length)`
// says (a bend strut softens once it buckles). `pullOnly`: a row that only ever pulls (a reach link):
// what it has given back it can't give again.
template <class AlphaAt>
inline void projectDistance(float* x, int a, int b, float wa, float wb, float rest, const AlphaAt& alphaAt, float atScale, double& lambda, bool pullOnly)
{
  const float w   = wa + wb;
  const float dx  = x[b * 3] - x[a * 3], dy = x[b * 3 + 1] - x[a * 3 + 1], dz = x[b * 3 + 2] - x[a * 3 + 2];
  const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
  if(len < 1e-8f)
    return;
  const float alpha = alphaAt(len);
  const float C     = len - rest;
  const float nx = dx / len, ny = dy / len, nz = dz / len;
  const float at = alpha * atScale;
  float       dl = float((-C - at * lambda) / (w + at));
  if(pullOnly)
    dl = float(std::min(lambda + double(dl), 0.0) - lambda);
  lambda += dl;
  if(wa > 0)
  {
    x[a * 3] -= nx * wa * dl;
    x[a * 3 + 1] -= ny * wa * dl;
    x[a * 3 + 2] -= nz * wa * dl;
  }
  if(wb > 0)
  {
    x[b * 3] += nx * wb * dl;
    x[b * 3 + 1] += ny * wb * dl;
    x[b * 3 + 2] += nz * wb * dl;
  }
}

// One XPBD attachment of particle i (inverse mass `w`) to the point `to` (xyz), at distance 0: a tether.
inline void projectAttach(float* x, int i, const float* to, float w, float alpha, float atScale, double& lambda)
{
  const float dx = x[i * 3] - to[0], dy = x[i * 3 + 1] - to[1], dz = x[i * 3 + 2] - to[2];
  const float C  = std::sqrt(dx * dx + dy * dy + dz * dz);
  if(C < 1e-8f)
    return;
  const float at = alpha * atScale;
  const float dl = float((-C - at * lambda) / (w + at));
  lambda += dl;
  const float s = w * dl / C;
  x[i * 3] += dx * s;
  x[i * 3 + 1] += dy * s;
  x[i * 3 + 2] += dz * s;
}

// A free particle's step: its velocity (xyz) with gravity's, damped by `damped` (exp(-rate dt)),
// capped at `maxSpeed`, then its position.
inline void integrate(float* x, float* v, const float* g, float damped, float dt, float maxSpeed)
{
  for(int k = 0; k < 3; k++)
    v[k] = (v[k] + g[k] * dt) * damped;
  const float sp = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if(sp > maxSpeed)
    for(int k = 0; k < 3; k++)
      v[k] *= maxSpeed / sp;
  for(int k = 0; k < 3; k++)
    x[k] += v[k] * dt;
}

// Rows in blocks of the particle range: a row goes to the block of its free particles (one with a
// held end to its free end's), and one between two blocks to `crossing`, swept after them on one
// thread. Each list starts with its leading rows (`prefix` of them), so a sweep over those alone is a
// prefix. DSON vertex order is spatially coherent, so few rows cross.
struct Blocks
{
  std::vector<std::vector<int>> rows;
  std::vector<int>              prefix;
  std::vector<int>              crossing;
  int                           crossingPrefix = 0;
};

// How many blocks every sweep is cut into, whatever the machine and the model, so a sweep's order (and
// with it a drape: the count moves a hem by about 2%) is the same everywhere. 32 is what a 16-core
// machine used before the count was fixed (two per pool thread), so its drapes stay as they were; any
// count is as right as another, the solver converging alike.
constexpr int kBlocks = 32;

// `ends(row)`: a row's two particles; `held(p)`: a particle no row moves; `leads(row)`: the rows
// that go first in each list.
Blocks partitionRows(int particles, size_t rows, const std::function<std::pair<int, int>(size_t)>& ends, const std::function<bool(int)>& held,
                     const std::function<bool(size_t)>& leads);

// Sweeps the blocks side by side on the pool, then the rows that cross them; with `leadingOnly` the
// leading rows alone. In order, or backwards: sweeps that alternate carry a correction to both ends of
// a chain in two of them.
template <class Project>
void sweepBlocks(const Blocks& blocks, bool leadingOnly, bool forward, const Project& project)
{
  const auto sweep = [&](const std::vector<int>& list, int count) {
    const size_t n = leadingOnly ? size_t(count) : list.size();
    if(forward)
      for(size_t k = 0; k < n; k++)
        project(list[k]);
    else
      for(size_t k = n; k-- > 0;)
        project(list[k]);
  };
  jobs::parallelFor(blocks.rows.size(), 1, [&](size_t begin, size_t end) {
    for(size_t b = begin; b < end; b++)
      sweep(blocks.rows[b], blocks.prefix[b]);
  });
  sweep(blocks.crossing, blocks.crossingPrefix);
}

}  // namespace fv::physics::particles
