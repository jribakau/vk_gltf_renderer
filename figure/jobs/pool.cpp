#include <figure/jobs/pool.hpp>

#include <algorithm>
#include <atomic>
#include <exception>

namespace fv::jobs {

Pool::Pool(unsigned threads)
{
  for(unsigned i = 0; i < std::max(1u, threads); i++)
    m_threads.emplace_back([this] { run(); });
}

Pool::~Pool()
{
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stop = true;
  }
  m_wake.notify_all();
  for(std::thread& t : m_threads)
    t.join();
}

void Pool::run()
{
  for(;;)
  {
    std::function<void()> job;
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_wake.wait(lock, [this] { return m_stop || !m_queue.empty(); });
      if(m_queue.empty())
        return;
      job = std::move(m_queue.front());
      m_queue.pop_front();
    }
    job();
  }
}

Pool& pool()
{
  const unsigned cores = std::thread::hardware_concurrency();
  static Pool    instance(std::max(2u, cores > 1 ? cores - 1 : 2u));
  return instance;
}

void parallelFor(size_t count, size_t chunk, const std::function<void(size_t, size_t)>& fn, size_t maxHelpers)
{
  chunk               = std::max<size_t>(chunk, 1);
  const size_t chunks = (count + chunk - 1) / chunk;
  if(chunks <= 1)
  {
    if(count)
      fn(0, count);
    return;
  }
  // shared with the helpers, which may start after the work is gone (and then do nothing)
  struct State
  {
    std::atomic<size_t>                         next{0}, done{0};
    size_t                                      chunks = 0, chunk = 0, count = 0;
    const std::function<void(size_t, size_t)>* fn = nullptr;
    std::mutex                                  mutex;
    std::condition_variable                     finished;
    std::exception_ptr                          error;
  };
  auto state    = std::make_shared<State>();
  state->chunks = chunks;
  state->chunk  = chunk;
  state->count  = count;
  state->fn     = &fn;
  // takes chunks until none are left; fn is only reached through a claimed chunk, so never after
  // the caller has returned
  const auto work = [](State& s) {
    for(size_t c; (c = s.next.fetch_add(1)) < s.chunks;)
    {
      try
      {
        (*s.fn)(c * s.chunk, std::min(s.count, (c + 1) * s.chunk));
      }
      catch(...)
      {
        std::lock_guard lock(s.mutex);
        if(!s.error)
          s.error = std::current_exception();
      }
      if(s.done.fetch_add(1) + 1 == s.chunks)
      {
        std::lock_guard lock(s.mutex);
        s.finished.notify_all();
      }
    }
  };
  Pool&        p       = pool();
  const size_t helpers = std::min({p.size(), chunks - 1, maxHelpers});
  for(size_t i = 0; i < helpers; i++)
    p.submit([state, work] { work(*state); });
  work(*state);
  std::unique_lock lock(state->mutex);
  state->finished.wait(lock, [&] { return state->done.load() == state->chunks; });
  if(state->error)
    std::rethrow_exception(state->error);
}

}  // namespace fv::jobs
