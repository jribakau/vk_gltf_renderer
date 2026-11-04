#include <figure/physics/volume.hpp>

namespace fv::physics {

double volumeError(const DisplacementVolume& c, std::span<const double> x)
{
  double sum = 0;
  for(size_t k = 0; k < c.particles.size(); k++)
  {
    const size_t i = size_t(c.particles[k]) * 3;
    for(int a = 0; a < 3; a++)
      sum += c.an[k * 3 + a] * (x[i + a] - c.start[k * 3 + a]);
  }
  return sum - c.target;
}

bool projectVolume(const DisplacementVolume& c, std::span<double> x)
{
  double norm = 0;
  for(size_t k = 0; k < c.particles.size(); k++)
    norm += c.weight[k] * (c.an[k * 3] * c.an[k * 3] + c.an[k * 3 + 1] * c.an[k * 3 + 1] + c.an[k * 3 + 2] * c.an[k * 3 + 2]);
  if(norm <= 0)
    return false;
  const double lambda = -volumeError(c, x) / norm;
  for(size_t k = 0; k < c.particles.size(); k++)
  {
    if(c.weight[k] == 0)
      continue;
    const size_t i = size_t(c.particles[k]) * 3;
    for(int a = 0; a < 3; a++)
      x[i + a] += c.weight[k] * c.an[k * 3 + a] * lambda;
  }
  return true;
}

}  // namespace fv::physics
