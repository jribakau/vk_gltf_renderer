#pragma once

// What the operating system says of its memory.

#include <cstdint>

namespace fv::io {

struct SystemMemory
{
  uint64_t totalPhysical     = 0;
  uint64_t availablePhysical = 0;
  // what can still be committed, the page file's included: past it an allocation fails
  uint64_t availableCommit = 0;
  bool     known           = false;
};

SystemMemory systemMemory();

}  // namespace fv::io
