#pragma once

// A cloth job on disk: the surfaces of a settle and how it steps, so a run can be replayed away from
// the scene that made it. fv_parity's dforce step writes one ("job"), and the GPU cloth bench reads it
// (docs/physics-plan.md step 13a), so the CPU island and the GPU solvers drape exactly the same input.

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <figure/physics/cloth/island.hpp>

namespace fv::physics::cloth {

struct Job
{
  std::vector<Surface> surfaces;
  // all of it but `cancelled`
  Options options;
  // the surfaces that are drapes, by slot, with their names; the body is slot 0
  std::vector<std::pair<int, std::string>> drapes;
};

bool               writeJob(const std::filesystem::path& file, const Job& job);
std::optional<Job> readJob(const std::filesystem::path& file);

}  // namespace fv::physics::cloth
