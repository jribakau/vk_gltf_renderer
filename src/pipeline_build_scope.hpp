/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>

// Whether shaders or pipelines are being built on the application thread, which answers no command
// meanwhile. The stream's /stats says so from its own thread, so the app's watchdog can tell a cold
// pipeline from a hang.
inline std::atomic<int>& pipelineBuildsRunning()
{
  static std::atomic<int> running{0};
  return running;
}

struct PipelineBuildScope
{
  PipelineBuildScope() { pipelineBuildsRunning().fetch_add(1); }
  ~PipelineBuildScope() { pipelineBuildsRunning().fetch_sub(1); }
  PipelineBuildScope(const PipelineBuildScope&)            = delete;
  PipelineBuildScope& operator=(const PipelineBuildScope&) = delete;
};
