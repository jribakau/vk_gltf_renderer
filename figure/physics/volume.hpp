#pragma once

// One constraint on how much volume a set of surface particles sweeps out from where they started:
// C = sum_i A_i n_i . (x_i - s_i) - target, with A_i a vertex's area, n_i its normal and s_i its start.
// Projected at no compliance it moves each particle along its normal by w_i A_i C / sum_j w_j A_j^2 (a
// particle of weight 0 counts but stays put). The contact pass uses it for the bulge, with
// w_i = f(r_i) / A_i (f the falloff with distance from the dent): one projection from C = -beta V puts
// back h_i = beta V f(r_i) / sum_j A_j f(r_j), so sum_i A_i h_i = beta V, the design's bulge exactly.
// Flesh (step 8) will use it per lobe.

#include <cstdint>
#include <span>
#include <vector>

namespace fv::physics {

struct DisplacementVolume
{
  std::vector<uint32_t> particles;
  std::vector<double>   weight;  // per particle; 0 holds it where it is
  std::vector<double>   an;      // per particle, xyz: its area times its unit normal
  std::vector<double>   start;   // per particle, xyz
  double                target = 0;
};

// C for positions `x` (xyz per particle of the whole set, indexed by `particles`).
double volumeError(const DisplacementVolume& c, std::span<const double> x);
// One projection at no compliance. False when nothing could move.
bool projectVolume(const DisplacementVolume& c, std::span<double> x);

}  // namespace fv::physics
