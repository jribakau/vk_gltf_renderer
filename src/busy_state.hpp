/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Tracks a background operation: start() on the worker, setReason() to update the message,
 * stop() when finished. The main thread polls isBusy() and isDone()/consumeDone().
 */

#include <string>
#include <atomic>
#include <mutex>

class BusyState
{
public:
  void start(const std::string& reason)
  {
    {
      std::lock_guard<std::mutex> lock(m_reasonMutex);
      m_reason = reason;
    }
    m_busy = true;
  }

  // Update the progress message while the busy window is running. Safe to call from
  // the worker thread; the UI thread picks up the new text on its next show() call.
  void setReason(const std::string& reason)
  {
    std::lock_guard<std::mutex> lock(m_reasonMutex);
    m_reason = reason;
  }

  void stop()
  {
    m_busy = false;
    m_done = true;
  }
  void consumeDone() { m_done = false; }
  bool isBusy() const { return m_busy; }
  bool isDone() const { return m_done; }

  // Display a modal window when loading assets or other long operation on separated thread
private:
  std::atomic<bool> m_busy{false};
  std::atomic<bool> m_done{false};
  std::mutex        m_reasonMutex;
  std::string       m_reason;
};
