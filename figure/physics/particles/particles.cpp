#include <figure/physics/particles/particles.hpp>

namespace fv::physics::particles {

Blocks partitionRows(int particles, size_t rows, const std::function<std::pair<int, int>(size_t)>& ends, const std::function<bool(int)>& held,
                     const std::function<bool(size_t)>& leads)
{
  Blocks     out;
  const int  blocks  = kBlocks;
  const auto blockOf = [&](int v) { return int(size_t(v) * size_t(blocks) / size_t(std::max(1, particles))); };
  out.rows.assign(size_t(blocks), {});
  for(size_t r = 0; r < rows; r++)
  {
    const auto [a, b] = ends(r);
    const bool fa = !held(a), fb = !held(b);
    if(!fa && !fb)
      continue;
    const int ba = blockOf(a), bb = blockOf(b);
    if(fa && fb && ba != bb)
      out.crossing.push_back(int(r));
    else
      out.rows[size_t(fa ? ba : bb)].push_back(int(r));
  }
  // the leading rows first in every list, in their order
  const auto leadFirst = [&](std::vector<int>& list) {
    const auto mid = std::stable_partition(list.begin(), list.end(), [&](int r) { return leads(size_t(r)); });
    return int(mid - list.begin());
  };
  for(auto& list : out.rows)
    out.prefix.push_back(leadFirst(list));
  out.crossingPrefix = leadFirst(out.crossing);
  return out;
}

}  // namespace fv::physics::particles
