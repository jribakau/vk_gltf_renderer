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

//
// The whole MCP surface for shader-timing work: the GltfRenderer automation members and the
// three tools that expose them. Loading a scene or an HDR is a parameter, not a tool -- see
// --scenefile / --hdrfile in main.cpp.
//

#ifdef USE_NVMCP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <thread>

#include <nvmcp/element_mcp.hpp>
#include <nvutils/logger.hpp>

#include "mcp_timing.hpp"
#include "renderer.hpp"

using Json = nlohmann::json;

namespace {

// The timer almost always wanted, so `measure` has a useful default.
constexpr const char* kDefaultTimer = "Path Trace (RTX)";

// Upper bound on `warmup` / `frames`, matching vk_gltf_measure's advertised inputSchema. Each
// sample waits for a rendered frame, so an unbounded count would park the request worker for
// hours and reserve the matching amount of memory.
constexpr int64_t kMaxFrames = 10000;

// Reported when the profiler timeline is absent: either the renderer is not attached yet, or it
// detached mid-call (elements detach in registration order, before the MCP server stops).
constexpr const char* kNoTimeline = "the profiler timeline is not available (renderer not attached, or shutting down)";

double median(std::vector<double> values)
{
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const double high = values[mid];
  if(values.size() % 2 != 0)
    return high;
  return (high + *std::max_element(values.begin(), values.begin() + mid)) * 0.5;
}

// The distribution of a timer's samples, in milliseconds.
TimerMeasurement summarize(const std::string& name, const std::vector<double>& samples)
{
  const double mean     = std::accumulate(samples.begin(), samples.end(), 0.0) / double(samples.size());
  const double variance = std::accumulate(samples.begin(), samples.end(), 0.0,
                                          [mean](double acc, double v) { return acc + (v - mean) * (v - mean); })
                          / double(samples.size());
  const auto [low, high] = std::minmax_element(samples.begin(), samples.end());

  return {.timer             = name,
          .samples           = samples.size(),
          .mean              = mean,
          .median            = median(samples),
          .standardDeviation = std::sqrt(variance),
          .minimum           = *low,
          .maximum           = *high};
}

// Upper bound on the timers one vk_gltf_measure call samples together.
constexpr size_t kMaxTimers = 64;

}  // namespace

//--------------------------------------------------------------------------------------------------
// Renderer-side automation. Defined here rather than in renderer.cpp so the whole optional feature
// is one translation unit: nothing outside it needs to know these exist.

// The profiler rebuilds its snapshot every frame from the sections that actually ran, so a timer
// is only listed while it is being recorded. An idle renderer (accumulation complete, ptMaxSamples
// reached) reports nothing, which looks like a missing timer rather than a paused one -- so give
// rendering a moment to resume before believing an empty list.
std::vector<std::string> GltfRenderer::profilerTimerNames() const
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  do
  {
    nvutils::ProfilerTimeline::Snapshot snapshot;
    {
      // The timeline only exists between onAttach() and onDetach(); this runs off the application
      // thread, so the pointer is read and used under the lock that teardown also takes.
      const std::lock_guard<std::mutex> lock(m_profilerTimelineMutex);
      if(!m_profilerTimeline)
        return {};
      m_profilerTimeline->getFrameSnapshot(snapshot);
    }
    if(!snapshot.timerNames.empty())
      return snapshot.timerNames;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  } while(std::chrono::steady_clock::now() < deadline);

  return {};
}

// Blocking: samples one GPU timer once per rendered frame. Must not run on the application thread,
// which is the thread producing the frames being waited on.
//
// The window is collected here rather than read from the profiler's running average, which spans
// the whole session and so cannot answer "is this shader edit faster". The profiler's per-timer
// ring index marks a new frame, so no renderer-side frame counter is needed.
TimerMeasurement GltfRenderer::measureTimer(const std::string& name, int warmup, int frames) const
{
  if(frames < 1)
    return TimerMeasurement::failure("'frames' must be at least 1");
  if(const std::string notReady = waitUntilMeasurable(); !notReady.empty())
    return TimerMeasurement::failure(notReady);

  uint32_t lastIndex    = 0;
  bool     haveIndex    = false;
  bool     timelineGone = false;  // set when teardown released the timeline mid-measurement

  const auto nextSample = [&](double& outMicroseconds) {
    // Generous per-frame budget: a heavy scene at a high sample count can take a while, and
    // giving up is better than hanging the caller forever on a stopped renderer.
    //
    // The first sample gets far longer. --scenefile now runs synchronously and returns with the
    // Vulkan resources installed, but BLAS/TLAS builds and the initial data uploads are still
    // draining out of m_loadPipeline (one submission per frame in interactive mode; a full drain
    // per frame in headless/benchmark modes), and the target section does not run -- and its
    // timer is not recorded -- until the pipeline it needs is live. The extended first-sample
    // budget is what lets a caller measure straight after a load without a manual sleep.
    const auto budget   = haveIndex ? std::chrono::seconds(5) : std::chrono::seconds(60);
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while(std::chrono::steady_clock::now() < deadline)
    {
      nvutils::ProfilerTimeline::TimerInfo info{};
      std::string                          apiName;
      bool                                 haveInfo = false;
      {
        // Same lock as destroyResources(): the pointer cannot be released while it is being read.
        const std::lock_guard<std::mutex> lock(m_profilerTimelineMutex);
        if(!m_profilerTimeline)
        {
          timelineGone = true;
          return false;
        }
        haveInfo = m_profilerTimeline->getFrameTimerInfo(name, info, apiName);
      }
      if(haveInfo && info.numAveraged > 0)
      {
        if(!haveIndex)
        {
          lastIndex = info.gpu.index;
          haveIndex = true;
        }
        else if(info.gpu.index != lastIndex)
        {
          lastIndex       = info.gpu.index;
          outMicroseconds = info.gpu.last;
          return true;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  };

  double sample = 0.0;
  for(int i = 0; i < std::max(0, warmup); i++)
  {
    if(!nextSample(sample))
      return TimerMeasurement::failure(timelineGone ? std::string(kNoTimeline) :
                                                      "timer '" + name
                                      + "' produced no GPU samples. Either the name is wrong (see "
                                        "vk_gltf_list_timers) or the renderer is idle -- accumulation "
                                        "stops at ptMaxSamples, and a stopped renderer records no timers.");
  }

  std::vector<double> samples;
  samples.reserve(size_t(frames));
  for(int i = 0; i < frames; i++)
  {
    if(!nextSample(sample))
      return TimerMeasurement::failure(timelineGone ? std::string(kNoTimeline) :
                                                      std::string("renderer stopped producing frames during the measurement"));
    samples.push_back(sample * 0.001);  // the profiler reports microseconds
  }

  return summarize(name, samples);
}

// The timeline is there and no UI-initiated load is running; otherwise why not, for the caller to
// return as its error.
std::string GltfRenderer::waitUntilMeasurable() const
{
  {
    const std::lock_guard<std::mutex> lock(m_profilerTimelineMutex);
    if(!m_profilerTimeline)
      return kNoTimeline;
  }

  // Wait out any UI-initiated load (drag-and-drop / File menu) that set m_busy: those still run
  // on a worker thread with a "Loading" overlay, and frames drawn during that window would time a
  // partial scene. The scenefile/hdrfile parameter path is now synchronous and does NOT set
  // m_busy, so this wait is a no-op for MCP-driven loads -- the safeguard for those is the
  // per-sample budget of the measurement, which refuses to return until the requested timer produces a
  // non-zero sample (the target section does not run until the pipeline it needs is live).
  // Measured on Sponza, sampling into an un-warmed scene understates cost by ~4x.
  const auto busyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while(m_busy.isBusy() && std::chrono::steady_clock::now() < busyDeadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  // Still loading when the wait ran out: sampling now would time a partial scene and report that
  // understatement as a result. Fail instead -- a timeout the caller can retry beats a wrong number.
  if(m_busy.isBusy())
    return "the renderer was still loading after 120 s; measuring now would time a "
           "partially loaded scene. Retry once the load completes.";
  return {};
}

// Blocking, off the application thread like measureTimer: samples several timers on the same
// rendered frames, so a short window (a drag, an orbit) times all of them while it lasts instead
// of one after another, the later ones after it ended.
//
// A frame records only the sections that ran, so some timers never tick in a given window (the
// raster's blend sort with nothing to sort, DLSS under accumulation). Such a timer is reported
// with an error once every timer that did tick has its samples, instead of holding the call for
// the first-sample budget. One that stops ticking for the per-sample budget (the motion ended)
// keeps the samples it has; `samples` says how many.
std::vector<TimerMeasurement> GltfRenderer::measureTimers(const std::vector<std::string>& names, int warmup, int frames) const
{
  std::vector<TimerMeasurement> results(names.size());
  if(frames < 1)
  {
    std::fill(results.begin(), results.end(), TimerMeasurement::failure("'frames' must be at least 1"));
    return results;
  }
  if(const std::string notReady = waitUntilMeasurable(); !notReady.empty())
  {
    std::fill(results.begin(), results.end(), TimerMeasurement::failure(notReady));
    return results;
  }

  struct Series
  {
    bool                                  seen      = false;  // a first frame index was read
    bool                                  finished  = false;  // all its samples, or it stopped ticking
    uint32_t                              lastIndex = 0;
    int                                   skipped   = 0;  // warmup frames passed over
    std::vector<double>                   samples;
    std::chrono::steady_clock::time_point lastTick;
  };
  std::vector<Series> series(names.size());

  // The same budgets as measureTimer: the first tick of any timer may take a while after a load,
  // a timer that ticked and then waits this long between frames has stopped.
  const auto start        = std::chrono::steady_clock::now();
  const auto firstBudget  = std::chrono::seconds(60);
  const auto sampleBudget = std::chrono::seconds(5);
  bool       timelineGone = false;
  while(true)
  {
    const auto now = std::chrono::steady_clock::now();
    for(size_t i = 0; i < names.size() && !timelineGone; i++)
    {
      Series& s = series[i];
      if(s.finished)
        continue;
      nvutils::ProfilerTimeline::TimerInfo info{};
      std::string                          apiName;
      bool                                 haveInfo = false;
      {
        // One profiler call per lock, as measureTimer does
        const std::lock_guard<std::mutex> lock(m_profilerTimelineMutex);
        if(!m_profilerTimeline)
        {
          timelineGone = true;
          break;
        }
        haveInfo = m_profilerTimeline->getFrameTimerInfo(names[i], info, apiName);
      }
      if(!haveInfo || info.numAveraged == 0)
        continue;
      if(!s.seen)
      {
        s.seen      = true;
        s.lastIndex = info.gpu.index;
        s.lastTick  = now;
        continue;
      }
      if(info.gpu.index == s.lastIndex)
        continue;
      s.lastIndex = info.gpu.index;
      s.lastTick  = now;
      if(s.skipped < warmup)
        s.skipped++;
      else
        s.samples.push_back(info.gpu.last * 0.001);  // the profiler reports microseconds
      s.finished = int(s.samples.size()) >= frames;
    }
    if(timelineGone)
      break;

    bool anySeen = false, allSeenFinished = true;
    for(Series& s : series)
    {
      if(s.seen && !s.finished && now - s.lastTick > sampleBudget)
        s.finished = true;
      anySeen |= s.seen;
      allSeenFinished &= !s.seen || s.finished;
    }
    // Done when every timer that ticked has finished: the ones that never did are not running in
    // these frames. Nothing ticking at all is a renderer that is idle or still warming up.
    if(anySeen && allSeenFinished)
      break;
    if(!anySeen && now - start > firstBudget)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  for(size_t i = 0; i < names.size(); i++)
  {
    if(timelineGone)
      results[i] = TimerMeasurement::failure(kNoTimeline);
    else if(!series[i].samples.empty())
      results[i] = summarize(names[i], series[i].samples);
    else
      results[i] = TimerMeasurement::failure("timer '" + names[i]
                                             + "' produced no GPU samples in these frames: the section did not run, the "
                                               "name is wrong (see vk_gltf_list_timers), or the renderer is idle");
    results[i].timer = names[i];
  }
  return results;
}


//--------------------------------------------------------------------------------------------------
// The MCP endpoint.

std::shared_ptr<nvmcp::Element> createTimingMcpServer(const McpTimingCreateInfo& info, const std::shared_ptr<GltfRenderer>& renderer)
{
  if(!renderer)
    return nullptr;

  auto mcp = std::make_shared<nvmcp::Element>(nvmcp::ElementCreateInfo{
      .serverName   = "vk_gltf_renderer",
      .instructions = "Measure the cost of a shader or render-configuration change in this live vk_gltf_renderer "
                      "instance. Load a scene or environment by setting scenefile/hdrfile. Typical loop: set the "
                      "configuration with nvpro_set_parameters, call "
                      "vk_gltf_measure, edit shaders/gltf_pathtrace.slang, call vk_gltf_reload_shaders, measure "
                      "again. Compare the MEDIAN, and read nvpro_get_logs for compiler diagnostics. "
                      "Every command-line parameter is exposed, but many are only read at start-up and writing "
                      "them changes nothing. The ones that take effect live, and that a timing run should pin, "
                      "are: ptAdaptiveSampling (false, for a fixed-cost frame), ptSamples, ptTechnique, ptUseSER, "
                      "ptOptimalShader, dlssEnable, ptMaxDepth, ptMaxSamples, and envSystem (not None -- that "
                      "removes all environment light). For live scene editing (materials, nodes, lights, "
                      "animation, morph weights, vertices) read vk_gltf_scene_info and batch edits with "
                      "vk_gltf_edit.",
      .port         = info.port,
  });

  // Weak so a call arriving during teardown errors out instead of resurrecting the renderer.
  const std::weak_ptr<GltfRenderer> weak = renderer;

  mcp->registerTool({
      .name        = "vk_gltf_reload_shaders",
      .description = "Recompile the active renderer's Slang shaders from source, swap in the new "
                     "pipelines and reset accumulation. Returns the wall-clock compile time. A compile "
                     "error is reported as an error and does NOT keep the previous shader running: the "
                     "renderer falls back to the build-time embedded SPIR-V, so any measurement taken "
                     "afterwards times that, not the edited source. Read nvpro_get_logs for the "
                     "diagnostics, fix the shader and reload again.",
      .handler =
          [weak](std::string_view) {
            auto renderer = weak.lock();
            if(!renderer)
              return nvmcp::ToolResult::error(R"({"error":"renderer is shutting down"})");
            const auto start    = std::chrono::steady_clock::now();
            const bool compiled = renderer->reloadShaders();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if(!compiled)
              return nvmcp::ToolResult::error(Json{
                  {"error",
                   "shader compilation failed; the renderer is now running the build-time embedded SPIR-V, "
                   "not the edited source. Read nvpro_get_logs for the compiler diagnostics."},
                  {"compileMilliseconds", ms}}.dump());
            return nvmcp::ToolResult::success(Json{{"compileMilliseconds", ms}}.dump());
          },
      // Destroys and recreates pipelines and drains the queue: application thread only.
      .runOnApplicationThread = true,
  });

  mcp->registerTool({
      .name        = "vk_gltf_list_timers",
      .description = "Return the GPU profiler timer names that vk_gltf_measure can time, such as the "
                     "path-tracer dispatch or the denoiser pass. The names depend on the active "
                     "renderer and technique. Only timers recorded in the frames being drawn right "
                     "now are listed, so an idle renderer returns an empty list.",
      .handler =
          [weak](std::string_view) {
            auto renderer = weak.lock();
            if(!renderer)
              return nvmcp::ToolResult::error(R"({"error":"renderer is shutting down"})");
            return nvmcp::ToolResult::success(Json{{"timers", renderer->profilerTimerNames()}}.dump());
          },
      .annotations = {.readOnlyHint = true, .destructiveHint = false, .idempotentHint = true},
  });

  mcp->registerTool({
      .name        = "vk_gltf_measure",
      .description = "Time one GPU profiler timer over a controlled number of frames and return the "
                     "distribution in milliseconds. Discards `warmup` frames, then samples one value "
                     "per rendered frame, so the result describes only what is running now. Compare "
                     "runs on MEDIAN: it reproduces to about 1%, while mean and standardDeviation are "
                     "inflated by occasional scheduling outliers and will hide a real change. Pass "
                     "`timers` (a list) instead of `timer` to sample several on the same frames: the "
                     "result is {timers: [...]}, one entry per name, and a timer that does not run in "
                     "those frames gets an `error` once the others are done instead of a timeout.",
      .inputSchema = R"({"type":"object","properties":{"timer":{"type":"string"},)"
                     R"("timers":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":64},)"
                     R"("frames":{"type":"integer","minimum":1,"maximum":10000},)"
                     R"("warmup":{"type":"integer","minimum":0,"maximum":10000}},"additionalProperties":false})",
      .handler =
          [weak](std::string_view payload) {
            auto renderer = weak.lock();
            if(!renderer)
              return nvmcp::ToolResult::error(R"({"error":"renderer is shutting down"})");

            // nvmcp does not enforce inputSchema, so the handler validates what it reads: both the
            // types (a wrong type would throw out of Json::value) and the advertised bounds (an
            // unbounded frame count reserves that many samples and waits seconds for each one).
            const Json input = Json::parse(payload, nullptr, /*allow_exceptions=*/false);
            if(input.is_discarded() || !input.is_object())
              return nvmcp::ToolResult::error(R"({"error":"arguments must be a JSON object"})");

            std::string timer = kDefaultTimer;
            if(const auto it = input.find("timer"); it != input.end())
            {
              if(!it->is_string())
                return nvmcp::ToolResult::error(R"({"error":"'timer' must be a string"})");
              timer = it->get<std::string>();
            }
            std::vector<std::string> timers;
            if(const auto it = input.find("timers"); it != input.end())
            {
              if(input.contains("timer"))
                return nvmcp::ToolResult::error(R"({"error":"pass 'timer' or 'timers', not both"})");
              if(!it->is_array() || it->empty() || it->size() > kMaxTimers
                 || !std::all_of(it->begin(), it->end(), [](const Json& name) { return name.is_string(); }))
                return nvmcp::ToolResult::error(
                    Json{{"error", "'timers' must be a list of 1 to " + std::to_string(kMaxTimers) + " names"}}.dump());
              timers = it->get<std::vector<std::string>>();
            }

            const auto readCount = [&input](const char* key, int64_t defaultValue, int64_t low, int64_t& out) -> bool {
              out           = defaultValue;
              const auto it = input.find(key);
              if(it == input.end())
                return true;
              if(!it->is_number_integer())
                return false;
              // Read unsigned values as such: a JSON integer above INT64_MAX would narrow
              // implementation-defined through get<int64_t>() and could sneak past the range check.
              if(it->is_number_unsigned() && it->get<uint64_t>() > uint64_t(kMaxFrames))
                return false;
              out = it->get<int64_t>();
              return out >= low && out <= kMaxFrames;
            };

            int64_t warmup = 0;
            int64_t frames = 0;
            if(!readCount("warmup", 20, 0, warmup))
              return nvmcp::ToolResult::error(
                  Json{{"error", "'warmup' must be an integer between 0 and " + std::to_string(kMaxFrames)}}.dump());
            if(!readCount("frames", 100, 1, frames))
              return nvmcp::ToolResult::error(
                  Json{{"error", "'frames' must be an integer between 1 and " + std::to_string(kMaxFrames)}}.dump());

            const auto toJson = [](const TimerMeasurement& result) {
              return Json{{"timer", result.timer},
                          {"samples", result.samples},
                          {"median", result.median},
                          {"mean", result.mean},
                          {"standardDeviation", result.standardDeviation},
                          {"minimum", result.minimum},
                          {"maximum", result.maximum},
                          {"unit", "milliseconds"}};
            };

            if(!timers.empty())
            {
              // One entry per name: a timer that did not run is an entry with its error, not a
              // failed call, since the others were measured.
              Json list = Json::array();
              for(const TimerMeasurement& result : renderer->measureTimers(timers, int(warmup), int(frames)))
                list.push_back(result.error.empty() ? toJson(result) : Json{{"timer", result.timer}, {"error", result.error}});
              return nvmcp::ToolResult::success(Json{{"timers", list}}.dump());
            }

            const TimerMeasurement result = renderer->measureTimer(timer, int(warmup), int(frames));
            if(!result.error.empty())
              return nvmcp::ToolResult::error(Json{{"error", result.error}}.dump());
            return nvmcp::ToolResult::success(toJson(result).dump());
          },
      .annotations = {.readOnlyHint = true, .destructiveHint = false},
      // Blocks on rendered frames, so it must stay off the thread producing them.
      .runOnApplicationThread = false,
  });

  LOGI("MCP endpoint configured on port %u\n", unsigned(info.port));
  return mcp;
}

#endif  // USE_NVMCP
