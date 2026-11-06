#pragma once

// dForce strand hair's settle under its old names: the hair island is physics/hair/strands.hpp since
// physics-plan step 14, on the core's follow-the-leader rows and distance field. The names stay for the
// driver, the parity harness and the DforceStrands tests until step 15 retires this directory.

#include <vector>

#include <figure/dforce/solver.hpp>
#include <figure/physics/hair/strands.hpp>

namespace fv::dforce {

using StrandSet     = physics::hair::StrandSet;
using StrandOptions = physics::hair::StrandOptions;

inline std::vector<float> settleStrands(const StrandSet& strands, const std::vector<SettleMesh>& colliders, const StrandOptions& options = {})
{
  return physics::hair::settle(strands, colliders, options);
}

}  // namespace fv::dforce
