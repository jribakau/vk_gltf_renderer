#pragma once

// Coarse phase timers for loads and wears: `fv::timing::Scope s("prepareWear");` reports how long
// its scope took to the process-wide sink, indented by how many scopes the thread has open. With no
// sink set nothing is measured or written, so the timers cost nothing in the unit tests.

#include <chrono>
#include <functional>
#include <string>

namespace fv::timing {

// Receives one finished line, "[fv-time] <indent><name> <ms> ms". Set once at start-up (the host's
// log, the parity tool's stderr); empty turns the timers off.
void setSink(std::function<void(const std::string&)> sink);
bool enabled();

// Reports its scope's time when it ends; one shorter than `minMs` is left out, so per-mesh scopes
// only show the meshes that cost something.
class Scope
{
public:
  explicit Scope(std::string name, double minMs = 0);
  ~Scope();
  Scope(const Scope&)            = delete;
  Scope& operator=(const Scope&) = delete;

private:
  std::string                           m_name;
  std::chrono::steady_clock::time_point m_start;
  double                                m_minMs;
  bool                                  m_on;
};

}  // namespace fv::timing
