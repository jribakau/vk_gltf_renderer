#pragma once

// The lifecycle of a pass that runs once the scene stops changing for a moment: a change schedules
// it after a debounce, it runs on the pool over a snapshot taken on the owner's thread, and its
// result is written back there. An edit that comes while it runs lets the result land (a drag shows
// where it got to) and runs it again on what the edit left. The wardrobe collider and dForce each
// wrote their own copy of this; the contact pass is the first on the shared one.
//
// One thread (the owner) calls everything.

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>

namespace fv::physics {

template <class Result>
class StillPass
{
public:
  explicit StillPass(std::chrono::milliseconds debounce)
      : m_debounce(debounce)
  {
  }
  ~StillPass()
  {
    if(m_job.valid())
      m_job.wait();
  }

  // An upstream pass this one waits for: while it says true, nothing starts here.
  std::function<bool()> waitWhile;

  // Due or running.
  bool busy() const { return m_scheduled || m_job.valid(); }
  // Bumps whenever busy() changes, or the owner says something it shows did.
  int  version() const { return m_version; }
  void bump() { m_version++; }

  // Something it reads changed: a pass once the changes stop for the debounce.
  void poke()
  {
    m_gen++;
    if(!m_scheduled)
      m_version++;
    m_scheduled = true;
    m_due       = std::chrono::steady_clock::now() + m_debounce;
  }
  // The same while an edit is being dragged: a pass as soon as the one in hand ends, not after a pause.
  void pokeNow()
  {
    m_gen++;
    // an earlier due time stands: a drag doesn't put it off
    if(!m_scheduled)
    {
      m_version++;
      m_due = std::chrono::steady_clock::now();
    }
    m_scheduled = true;
  }
  // Nothing is due, and the pass in hand is dropped when it lands.
  void cancel()
  {
    if(m_scheduled || m_job.valid())
      m_version++;
    m_scheduled = false;
    m_drop      = m_job.valid();
    m_gen++;
  }

  // Lands a pass that finished, then starts one that is due. `start()` takes the snapshot and returns
  // the job, or an invalid future when there is nothing to do; `finish(result)` writes what a job
  // returned, `finish(nullptr)` what nothing to do means. Both return whether they wrote. Call every
  // frame.
  template <class Start, class Finish>
  bool pump(Start&& start, Finish&& finish)
  {
    bool wrote = false;
    if(m_job.valid() && m_job.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
    {
      Result result = m_job.get();
      if(!m_drop)
        wrote = finish(&result);
      // edits that arrived while it ran: once more, on the shape they left
      if(!m_drop && m_jobGen != m_gen)
      {
        m_scheduled = true;
        m_due       = std::chrono::steady_clock::now();
      }
      m_drop = false;
      if(!m_scheduled)
        m_version++;
    }
    if(!m_scheduled || m_job.valid() || std::chrono::steady_clock::now() < m_due || (waitWhile && waitWhile()))
      return wrote;
    m_scheduled = false;
    m_jobGen    = m_gen;
    m_job       = start();
    if(!m_job.valid())
    {
      wrote = finish(static_cast<const Result*>(nullptr)) || wrote;
      m_version++;  // no longer busy
    }
    return wrote;
  }

private:
  std::chrono::milliseconds             m_debounce;
  bool                                  m_scheduled = false;
  bool                                  m_drop      = false;
  std::chrono::steady_clock::time_point m_due;
  uint64_t                              m_gen    = 0;
  uint64_t                              m_jobGen = 0;
  std::future<Result>                   m_job;
  int                                   m_version = 0;
};

}  // namespace fv::physics
