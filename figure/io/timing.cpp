#include <figure/io/timing.hpp>

#include <atomic>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>

namespace fv::timing {

namespace {
std::mutex                                                 g_mutex;
std::shared_ptr<const std::function<void(const std::string&)>> g_sink;
std::atomic<bool>                                          g_on{false};
thread_local int                                           t_depth = 0;
struct Sum
{
  int    n     = 0;
  double ms    = 0;
  double maxMs = 0;
};
std::map<std::string, Sum, std::less<>> g_accum;  // under g_mutex
}  // namespace

void accumulate(std::string_view name, double ms)
{
  if(!enabled())
    return;
  std::lock_guard lock(g_mutex);
  auto            it = g_accum.find(name);
  if(it == g_accum.end())
    it = g_accum.emplace(std::string(name), Sum{}).first;
  Sum& s = it->second;
  s.n++;
  s.ms += ms;
  if(ms > s.maxMs)
    s.maxMs = ms;
}

std::vector<std::string> takeSummary()
{
  std::map<std::string, Sum, std::less<>> taken;
  {
    std::lock_guard lock(g_mutex);
    taken.swap(g_accum);
  }
  std::vector<std::string> lines;
  for(const auto& [name, s] : taken)
  {
    char tail[96];
    std::snprintf(tail, sizeof tail, ": %d calls, %.2f ms mean, %.2f max", s.n, s.n ? s.ms / s.n : 0.0, s.maxMs);
    lines.push_back("[fv-time] " + name + tail);
  }
  return lines;
}

Accum::Accum(const char* name)
    : m_name(name)
    , m_on(enabled())
{
  if(m_on)
    m_start = std::chrono::steady_clock::now();
}

Accum::~Accum()
{
  if(!m_on)
    return;
  accumulate(m_name, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_start).count());
}

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

double Scope::elapsedMs() const
{
  return m_on ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_start).count() : 0.0;
}

Scope::~Scope()
{
  if(!m_on)
    return;
  t_depth--;
  report(m_name, elapsedMs(), m_minMs);
}

void line(std::string_view text)
{
  if(!enabled())
    return;
  std::shared_ptr<const std::function<void(const std::string&)>> sink;
  {
    std::lock_guard lock(g_mutex);
    sink = g_sink;
  }
  if(sink)
    (*sink)("[fv-time] " + std::string(size_t(t_depth) * 2, ' ') + std::string(text));
}

void report(std::string_view name, double ms, double minMs)
{
  if(!enabled() || ms < minMs)
    return;
  char tail[32];
  std::snprintf(tail, sizeof tail, " %.1f ms", ms);
  line(std::string(name) + tail);
}

}  // namespace fv::timing
