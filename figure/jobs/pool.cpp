#include <figure/jobs/pool.hpp>

#include <algorithm>

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

}  // namespace fv::jobs
