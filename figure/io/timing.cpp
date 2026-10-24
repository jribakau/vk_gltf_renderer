#include <figure/io/timing.hpp>

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>

namespace fv::timing {

namespace {
std::mutex                                                 g_mutex;
std::shared_ptr<const std::function<void(const std::string&)>> g_sink;
std::atomic<bool>                                          g_on{false};
thread_local int                                           t_depth = 0;
}  // namespace

void setSink(std::function<void(const std::string&)> sink)
{
  std::lock_guard lock(g_mutex);
  g_on   = bool(sink);
  g_sink = sink ? std::make_shared<const std::function<void(const std::string&)>>(std::move(sink)) : nullptr;
}

bool enabled()
{
  return g_on.load(std::memory_order_relaxed);
}

Scope::Scope(std::string name, double minMs)
    : m_minMs(minMs)
    , m_on(enabled())
{
  if(!m_on)
    return;
  m_name  = std::move(name);
  m_start = std::chrono::steady_clock::now();
  t_depth++;
}

Scope::~Scope()
{
  if(!m_on)
    return;
  t_depth--;
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_start).count();
  if(ms < m_minMs)
    return;
  std::shared_ptr<const std::function<void(const std::string&)>> sink;
  {
    std::lock_guard lock(g_mutex);
    sink = g_sink;
  }
  if(!sink)
    return;
  char tail[32];
  std::snprintf(tail, sizeof tail, " %.1f ms", ms);
  (*sink)("[fv-time] " + std::string(size_t(t_depth) * 2, ' ') + m_name + tail);
}

}  // namespace fv::timing
