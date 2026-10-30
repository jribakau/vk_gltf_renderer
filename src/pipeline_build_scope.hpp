/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

// Whether shaders or pipelines are being built on the application thread, which answers no command
// meanwhile. The stream's /stats says so from its own thread, so the app's watchdog can tell a cold
// pipeline from a hang, and the viewport can say what the wait is.
inline std::atomic<int>& pipelineBuildsRunning()
{
  static std::atomic<int> running{0};
  return running;
}

// What /stats reports. eCompile and ePipeline hold the view until they end; eWarm is the warm-up's
// threads, building ahead of need while everything goes on.
enum class ShaderWork : int
{
  eNone,
  eCompile,
  ePipeline,
  eWarm,
};

inline const char* shaderWorkName(ShaderWork work)
{
  switch(work)
  {
    case ShaderWork::eCompile:
      return "compile";
    case ShaderWork::ePipeline:
      return "pipeline";
    case ShaderWork::eWarm:
      return "warm";
    default:
      return "";
  }
}

namespace shader_work_detail {
inline int64_t nowMs()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline std::atomic<int>& stage()
{
  static std::atomic<int> value{int(ShaderWork::eNone)};
  return value;
}
inline std::atomic<int64_t>& buildSince()
{
  static std::atomic<int64_t> value{0};
  return value;
}
inline std::atomic<int>& warms()
{
  static std::atomic<int> value{0};
  return value;
}
inline std::atomic<int64_t>& warmSince()
{
  static std::atomic<int64_t> value{0};
  return value;
}
}  // namespace shader_work_detail

struct PipelineBuildScope
{
  // the wait is counted from here, across the compile and the pipeline after it
  PipelineBuildScope()
  {
    if(pipelineBuildsRunning().fetch_add(1) == 0)
      shader_work_detail::buildSince() = shader_work_detail::nowMs();
  }
  ~PipelineBuildScope() { pipelineBuildsRunning().fetch_sub(1); }
  PipelineBuildScope(const PipelineBuildScope&)            = delete;
  PipelineBuildScope& operator=(const PipelineBuildScope&) = delete;
};

// A build that holds the view: its stage, for as long as this lives
struct ShaderWorkScope
{
  explicit ShaderWorkScope(ShaderWork work) { shader_work_detail::stage() = int(work); }
  ~ShaderWorkScope() { shader_work_detail::stage() = int(ShaderWork::eNone); }
  ShaderWorkScope(const ShaderWorkScope&)            = delete;
  ShaderWorkScope& operator=(const ShaderWorkScope&) = delete;
};

// One of the warm-up's threads
struct ShaderWarmScope
{
  ShaderWarmScope()
  {
    if(shader_work_detail::warms().fetch_add(1) == 0)
      shader_work_detail::warmSince() = shader_work_detail::nowMs();
  }
  ~ShaderWarmScope() { shader_work_detail::warms().fetch_sub(1); }
  ShaderWarmScope(const ShaderWarmScope&)            = delete;
  ShaderWarmScope& operator=(const ShaderWarmScope&) = delete;
};

struct ShaderWorkNow
{
  ShaderWork work = ShaderWork::eNone;
  int64_t    ms   = 0;  // how long it has been going
};

// What is being built now. A build that holds the view comes first: that is the wait to explain.
inline ShaderWorkNow shaderWorkNow()
{
  using namespace shader_work_detail;
  const auto held = ShaderWork(stage().load());
  if(held != ShaderWork::eNone)
    return {held, nowMs() - buildSince().load()};
  if(warms().load() > 0)
    return {ShaderWork::eWarm, nowMs() - warmSince().load()};
  return {};
}
