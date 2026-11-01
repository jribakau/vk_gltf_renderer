#pragma once

// Large builds in system memory, a few at a time: several cold strand hairs side by side each want
// gigabytes, more than there may be. One that would not fit waits for those running; one alone always
// goes, so nothing waits for ever, and nothing small comes here.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>

namespace fv::jobs {

class MemoryGate
{
public:
  // `available`: what can be taken now, asked each time a build looks
  explicit MemoryGate(std::function<uint64_t()> available)
      : m_available(std::move(available))
  {
  }

  class Ticket
  {
  public:
    Ticket() = default;
    Ticket(MemoryGate* gate, uint64_t bytes)
        : m_gate(gate)
        , m_bytes(bytes)
    {
    }
    Ticket(Ticket&& o) noexcept
        : m_gate(o.m_gate)
        , m_bytes(o.m_bytes)
    {
      o.m_gate = nullptr;
    }
    Ticket& operator=(Ticket&& o) noexcept
    {
      if(this != &o)
      {
        release();
        m_gate   = o.m_gate;
        m_bytes  = o.m_bytes;
        o.m_gate = nullptr;
      }
      return *this;
    }
    ~Ticket() { release(); }
    void release()
    {
      if(m_gate)
        m_gate->leave(m_bytes);
      m_gate = nullptr;
    }

  private:
    MemoryGate* m_gate  = nullptr;
    uint64_t    m_bytes = 0;
  };

  // Waits until `bytes` fits beside what is running, or nothing is. Held while the build runs.
  Ticket enter(uint64_t bytes)
  {
    std::unique_lock lock(m_mutex);
    m_cv.wait(lock, [&] { return m_inside == 0 || m_inside + bytes <= m_available(); });
    m_inside += bytes;
    return {this, bytes};
  }

  uint64_t inside() const
  {
    std::lock_guard lock(m_mutex);
    return m_inside;
  }

private:
  void leave(uint64_t bytes)
  {
    {
      std::lock_guard lock(m_mutex);
      m_inside -= bytes;
    }
    m_cv.notify_all();
  }

  std::function<uint64_t()> m_available;
  mutable std::mutex        m_mutex;
  std::condition_variable   m_cv;
  uint64_t                  m_inside = 0;
};

}  // namespace fv::jobs
