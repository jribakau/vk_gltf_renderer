#pragma once

// Coarse phase timers for loads and wears: `fv::timing::Scope s("prepareWear");` reports how long
// its scope took to the process-wide sink, indented by how many scopes the thread has open. With no
// sink set nothing is measured or written, so the timers cost nothing in the unit tests.

#include <chrono>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

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

// Phases that run many times a second (a dial step's formula graph, its deltas, its normals): one
// line each would flood the log, so `Accum a("driver.run");` adds its scope's time under the name,
// and takeSummary() returns one "[fv-time] <name>: <n> calls, <mean> ms mean, <max> max" line per
// name seen since the last call (the host logs them once a second), and clears. Off without a sink.
void                     accumulate(std::string_view name, double ms);
std::vector<std::string> takeSummary();

class Accum
{
public:
  explicit Accum(const char* name);
  ~Accum();
  Accum(const Accum&)            = delete;
  Accum& operator=(const Accum&) = delete;

private:
  const char*                           m_name;
  std::chrono::steady_clock::time_point m_start;
  bool                                  m_on;
};

}  // namespace fv::timing
