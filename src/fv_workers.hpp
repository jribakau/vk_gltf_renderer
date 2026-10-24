/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// The Figure Viewer host's worker threads (loads, wears, exports, texture bakes): each runs to the
// end on its own. Finished ones are joined when the next one starts, so a long session doesn't
// keep a stack per finished job; the rest are joined at shutdown.
//

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

class FvWorkers
{
public:
  FvWorkers() = default;
  FvWorkers(const FvWorkers&)            = delete;
  FvWorkers& operator=(const FvWorkers&) = delete;
  ~FvWorkers() { joinAll(); }

  template <class F>
  void spawn(F&& f)
  {
    auto            done = std::make_shared<std::atomic<bool>>(false);
    std::lock_guard lock(m_mutex);
    reap();
    m_workers.push_back({std::thread([f = std::forward<F>(f), done]() mutable {
                           struct Done
                           {
                             std::atomic<bool>& flag;
                             ~Done() { flag = true; }
                           } mark{*done};
                           f();
                         }),
                         done});
  }

  // Waits for every worker, including ones the running ones start meanwhile. Not under the lock,
  // so a worker that starts another doesn't wait on it.
  void joinAll()
  {
    for(;;)
    {
      std::vector<Worker> all;
      {
        std::lock_guard lock(m_mutex);
        all.swap(m_workers);
      }
      if(all.empty())
        return;
      for(Worker& w : all)
        if(w.thread.joinable())
          w.thread.join();
    }
  }

private:
  struct Worker
  {
    std::thread                        thread;
    std::shared_ptr<std::atomic<bool>> done;
  };

  void reap()
  {
    for(auto it = m_workers.begin(); it != m_workers.end();)
    {
      if(*it->done)
      {
        if(it->thread.joinable())
          it->thread.join();
        it = m_workers.erase(it);
      }
      else
        ++it;
    }
  }

  std::mutex          m_mutex;
  std::vector<Worker> m_workers;
};
