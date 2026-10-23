#pragma once

// A small thread pool for the work a figure does off the application thread: reading and parsing
// library files and building morph targets. Results come back as futures, which the figure's owner
// collects on its own thread, so figure objects stay single-threaded.

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace fv::jobs {

class Pool
{
public:
  explicit Pool(unsigned threads);
  ~Pool();
  Pool(const Pool&)            = delete;
  Pool& operator=(const Pool&) = delete;

  template <class F>
  auto submit(F&& f) -> std::future<std::invoke_result_t<F>>
  {
    using R   = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut  = task->get_future();
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_queue.emplace_back([task] { (*task)(); });
    }
    m_wake.notify_one();
    return fut;
  }

private:
  void run();

  std::mutex                        m_mutex;
  std::condition_variable           m_wake;
  std::deque<std::function<void()>> m_queue;
  std::vector<std::thread>          m_threads;
  bool                              m_stop = false;
};

// The process-wide pool, sized to the machine (leaving the application thread its core).
Pool& pool();

}  // namespace fv::jobs
