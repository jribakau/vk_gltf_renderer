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
// A shader variant built off the application thread (decision D15; daz-viewer
// docs/retained-scene-plan.md, "Shader variants under incremental edits"). The application thread
// snapshots what to build into a Job, a value that names none of the renderer's live state; a
// worker builds it with nothing but the job and what the build function captured, destroying
// nothing; the application thread takes the result at frame top and installs it. One job at a
// time: the next starts once the last one's result is taken.
//

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

template <class Job, class Built>
class VariantJobs
{
public:
  using Build = std::function<Built(const Job&)>;

  VariantJobs() = default;
  ~VariantJobs() { join(); }
  VariantJobs(const VariantJobs&)            = delete;
  VariantJobs& operator=(const VariantJobs&) = delete;

  // Builds `job` on a worker; false, starting nothing, while the last job's result is not taken
  bool start(Job job, Build build)
  {
    if(m_state)
      return false;
    join();  // the last worker, its result taken
    m_state      = std::make_shared<State>();
    m_state->job = std::move(job);
    m_worker     = std::thread([state = m_state, build = std::move(build)] {
      Built                       built = build(state->job);
      std::lock_guard<std::mutex> lock(state->mutex);
      state->built = std::move(built);
    });
    return true;
  }

  // The job building or built and not taken; none
  const Job* job() const { return m_state ? &m_state->job : nullptr; }

  // Whether a build has returned and its result waits for take()
  bool done() const
  {
    if(!m_state)
      return false;
    std::lock_guard<std::mutex> lock(m_state->mutex);
    return m_state->built.has_value();
  }

  // Once the build has returned: the job and what it built, taken
  std::optional<std::pair<Job, Built>> take()
  {
    if(!m_state)
      return std::nullopt;
    {
      std::lock_guard<std::mutex> lock(m_state->mutex);
      if(!m_state->built)
        return std::nullopt;
    }
    join();
    std::pair<Job, Built> out(std::move(m_state->job), std::move(*m_state->built));
    m_state.reset();
    return out;
  }

  // Waits for the worker: at teardown, or before what its build reads goes; take() then has the result
  void join()
  {
    if(m_worker.joinable())
      m_worker.join();
  }

private:
  struct State
  {
    Job                  job;
    std::mutex           mutex;
    std::optional<Built> built;
  };
  std::shared_ptr<State> m_state;
  std::thread            m_worker;
};
