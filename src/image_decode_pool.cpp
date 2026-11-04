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

#include "image_decode_pool.hpp"

#include <algorithm>
#include <exception>
#include <optional>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rs {

ImageDecodePool::ImageDecodePool(Decode decode, uint32_t threads)
    : m_decode(std::move(decode))
{
  for(uint32_t i = 0; i < std::max(threads, 1u); i++)
    m_threads.emplace_back([this] { run(); });
}

ImageDecodePool::~ImageDecodePool()
{
  std::deque<Job> waiting;
  {
    std::lock_guard lock(m_mutex);
    m_stopping = true;
    waiting.swap(m_waiting);
  }
  m_wake.notify_all();
  for(std::thread& t : m_threads)
    t.join();
}

uint32_t ImageDecodePool::defaultThreads()
{
  return std::clamp(std::thread::hardware_concurrency() / 2, 1u, 8u);
}

uint64_t ImageDecodePool::start(HeldImageSource source)
{
  uint64_t ticket = 0;
  {
    std::lock_guard lock(m_mutex);
    ticket = m_nextTicket++;
    m_waiting.push_back({ticket, std::move(source)});
  }
  m_wake.notify_one();
  return ticket;
}

void ImageDecodePool::cancel(uint64_t ticket)
{
  // what it held is let go of outside the lock: a decoded 4096 is 64 MB to free
  std::optional<Job>  waiting;
  std::optional<Done> done;
  {
    std::lock_guard lock(m_mutex);
    if(auto it = std::find_if(m_waiting.begin(), m_waiting.end(), [&](const Job& j) { return j.ticket == ticket; }); it != m_waiting.end())
    {
      waiting.emplace(std::move(*it));
      m_waiting.erase(it);
    }
    else if(m_running.count(ticket))
      m_cancelled.insert(ticket);
    else if(auto d = m_done.find(ticket); d != m_done.end())
    {
      done.emplace(std::move(d->second));
      m_done.erase(d);
    }
  }
  m_idle.notify_all();
}

std::vector<ImageDecodePool::Done> ImageDecodePool::take(uint64_t budget)
{
  std::vector<Done> out;
  uint64_t          bytes = 0;
  std::lock_guard   lock(m_mutex);
  for(auto it = m_done.begin(); it != m_done.end();)
  {
    if(!out.empty() && bytes + it->second.bytes > budget)
      break;
    bytes += it->second.bytes;
    out.push_back(std::move(it->second));
    it = m_done.erase(it);
  }
  return out;
}

bool ImageDecodePool::ready() const
{
  std::lock_guard lock(m_mutex);
  return !m_done.empty();
}

size_t ImageDecodePool::pending() const
{
  std::lock_guard lock(m_mutex);
  return m_waiting.size() + m_running.size() - m_cancelled.size() + m_done.size();
}

void ImageDecodePool::waitIdle()
{
  std::unique_lock lock(m_mutex);
  m_idle.wait(lock, [&] { return m_waiting.empty() && m_running.empty(); });
}

void ImageDecodePool::run()
{
#ifdef _WIN32
  // the frame and the host first: a decode is only ever ahead of what the view needs now
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
  SetThreadDescription(GetCurrentThread(), L"texture decode");
#endif
  for(;;)
  {
    std::optional<Job> job;
    {
      std::unique_lock lock(m_mutex);
      m_wake.wait(lock, [&] { return m_stopping || !m_waiting.empty(); });
      if(m_stopping)
        return;
      job.emplace(std::move(m_waiting.front()));
      m_waiting.pop_front();
      m_running.insert(job->ticket);
    }
    Done done{.ticket = job->ticket, .source = std::move(job->source)};
    job.reset();
    try
    {
      done.ok = m_decode(done.source.source, done.loaded, done.prefetched);
    }
    catch(const std::exception&)  // system memory ran out, most likely: as one that can't be read
    {
      done.ok     = false;
      done.loaded = {};
    }
    if(!done.ok)
      done.loaded = {};
    for(const std::vector<char>& level : done.loaded.mipData)
      done.bytes += level.size();
    std::optional<Done> dropped;
    {
      std::lock_guard lock(m_mutex);
      m_running.erase(done.ticket);
      if(m_cancelled.erase(done.ticket))
        dropped.emplace(std::move(done));
      else
        m_done.emplace(done.ticket, std::move(done));
    }
    m_idle.notify_all();
  }
}

}  // namespace rs
