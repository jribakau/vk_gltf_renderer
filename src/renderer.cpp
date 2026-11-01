/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*
    GLTF Renderer with Ray Tracing and Rasterization Support

    This renderer demonstrates advanced real-time rendering of GLTF scenes 
    using both ray tracing and rasterization pipelines. Key features include:
    
    - Dual rendering modes: path tracing and traditional rasterization
    - PBR (Physically Based Rendering) material system
    - HDR environment mapping with prefiltered importance sampling
    - Procedural sky simulation
    - Animation support with skeletal and keyframe animations
    - Progressive rendering for path tracing
    - GLTF 2.0 specification compliance with extensions
    - Interactive ray picking for scene manipulation
    - UI-driven scene editing capabilities
    
    The implementation uses Vulkan with ray tracing extensions and
    employs a modular architecture to handle the full rendering pipeline
    from scene loading to final display, with careful memory management
    and asynchronous command processing for optimal performance.
*/
//////////////////////////////////////////////////////////////////////////

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }
#define IMGUI_DEFINE_MATH_OPERATORS

#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>  // glm::rotation
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vulkan/vulkan_core.h>
#include <stb/stb_image.h>

#include "GLFW/glfw3.h"
#undef APIENTRY

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

// Pre-compiled shaders
#include "_autogen/tonemapper.slang.h"
#include "_autogen/hdr_dome.slang.h"
#include "_autogen/hdr_integrate_brdf.slang.h"
#include "_autogen/hdr_prefilter_diffuse.slang.h"
#include "_autogen/hdr_prefilter_glossy.slang.h"
#include "_autogen/env_write_pdf.slang.h"
#include "_autogen/sky_bruneton_precompute.slang.h"

//
#include <backends/imgui_impl_vulkan.h>
#include <nvaftermath/aftermath.hpp>
#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/mipmaps.hpp>

#include <glm/gtc/quaternion.hpp>

#include "renderer.hpp"
#include "fv_compose.hpp"
#include "scene_import_gltf.hpp"

#include <figure/io/timing.hpp>
#include "env_image_loader.hpp"
#include "gltf_environment_sky.hpp"
#include "sun_position.hpp"
#include "sky_preset.hpp"
#include "shadow_fit.hpp"
#include "staging_config.hpp"
#include "tinygltf_utils.hpp"
#include "utils.hpp"
#include "animation_control.hpp"

extern nvutils::ProfilerManager g_profilerManager;  // #PROFILER

namespace {
// Background clear color used when no scene is loaded or to show DLSS render resolution borders
constexpr VkClearColorValue kBackgroundClearColor = {{0.17f, 0.21f, 0.25f, 1.f}};

// Angular radius of the disk the gradient sky draws. The extension has no field for it, so this is
// the sun's true radius as seen from Earth (~0.27 deg) -- enough for the disk to read as a sun.
// Authored decoration rather than a measurement, unlike the physical sky's, which comes from the
// atmosphere. One constant because three places need the same number: the drawn disk, the light's
// angular size, and what a save writes.
constexpr float kGradientSunAngularRadius = 0.00465f;

// How much brighter the gradient sky's sun is than the dome it sits under, as illuminance on a
// surface facing each.
//
// A number this file has to invent, because the gradient sky never measured anything: its colours
// are authored in arbitrary linear units where the dome sits near 1, and nothing in them says how
// bright a sun should be. So it is fitted to the real thing instead, against the clear-day
// direct:diffuse band a measured clear day gives -- which makes it checkable rather than merely
// plausible.
//
// Fitted across the whole band rather than at one sun angle, because this sky cannot follow its
// shape. A real sky's diffuse half dims as the sun drops, so direct:diffuse climbs faster than the
// sun's own cosine; the gradient's dome does not move with the sun at all (its bake excludes the
// sun entirely), so the modelled ratio is exactly proportional to sin(elevation). That leaves one
// free scale, and only 5.2-8.0 keeps every reference elevation from 15 to 80 degrees inside the
// band. 6.5 sits in the middle of that window: measured 1.8 / 3.4 / 4.8 / 5.8 / 6.6 against
// 1.0-2.2 / 2.2-4.0 / 3.5-6.5 / 4.5-8.0 / 5.0-9.0.
constexpr float kGradientSunToSkyRatio = 6.5f;

// Host-side twin of skyGradientBandWeight() in shaders/sky_gradient.h.slang. The two must agree:
// the sun's illuminance is scaled against the dome's, and the dome is this curve.
float gradientBandWeight(float c, float curve)
{
  const float safeCurve = std::max(curve, 1e-3f);
  return glm::clamp(1.0f - std::pow(1.0f - glm::clamp(c, 0.0f, 1.0f), 1.0f / safeCurve), 0.0f, 1.0f);
}

}  // namespace

namespace {

uint64_t sumTrackerCurrentBytes(const nvvkgltf::GpuMemoryTracker& tracker)
{
  return tracker.getTotalStats().currentBytes;
}

// Two timestamps around work that runs once (a load's BLAS or TLAS build), read when the GPU has
// finished it: the profiler's frame sections see only the work of the frames being drawn. Held by
// the completion callback, so its query pool goes with the callback, or with the pipeline's clear()
// when the work is dropped (after a queue wait, so never while in use).
class OneShotGpuTimer
{
public:
  OneShotGpuTimer(VkDevice device, VkPhysicalDevice physicalDevice)
      : m_device(device)
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    m_msPerTick = double(props.limits.timestampPeriod) * 1e-6;
    const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
    NVVK_CHECK(vkCreateQueryPool(device, &info, nullptr, &m_pool));
  }
  ~OneShotGpuTimer() { vkDestroyQueryPool(m_device, m_pool, nullptr); }
  OneShotGpuTimer(const OneShotGpuTimer&)            = delete;
  OneShotGpuTimer& operator=(const OneShotGpuTimer&) = delete;

  void begin(VkCommandBuffer cmd) const
  {
    vkCmdResetQueryPool(cmd, m_pool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_pool, 0);
  }
  void end(VkCommandBuffer cmd) const { vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pool, 1); }

  // Once the command buffer has completed; 0 if the results are not there.
  double ms() const
  {
    uint64_t stamps[2]{};
    if(vkGetQueryPoolResults(m_device, m_pool, 0, 2, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
      return 0.0;
    return double(stamps[1] - stamps[0]) * m_msPerTick;
  }

  // An [fv-time] line with three decimals, since a TLAS build of a few hundred instances is
  // hundredths of a millisecond (fv::timing::report rounds to one).
  static void report(const char* name, double ms)
  {
    char text[96];
    std::snprintf(text, sizeof(text), "%s %.3f ms", name, ms);
    fv::timing::line(text);
  }

private:
  VkDevice    m_device{};
  VkQueryPool m_pool{};
  double      m_msPerTick{};
};

}  // namespace

// The constructor registers the parameters that can be set from the command line
GltfRenderer::GltfRenderer(nvutils::ParameterRegistry* paramReg, const nvutils::ParameterParser* paramParser)
    : m_parameterParser(paramParser)
    , m_settings(paramReg)
{
  // Every user-settable value is declared once here (see settings_registry.hpp): the command line,
  // MCP, the app's parameter messages and ImGui.ini all follow from this one call. Persist says whether the
  // value is remembered between runs.
  m_settings.add({.name            = "envSystem",
                  .help            = "Environment: [Sky:0, HDR:1, None:2, Plain:3, Gradient:4]",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { onEnvironmentChanged(); }},
                 (int*)&m_resources.settings.envSystem, Persist::eYes, 0, 4);
  m_settings.add({.name            = "renderSystem",
                  .help            = "Renderer [Path tracer:0, Rasterizer:1]",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { onRenderSystemChanged(); }},
                 (int*)&m_resources.settings.renderSystem, Persist::eYes, 0, 1);
  // what the path tracer reads of the environment: a change starts the still again, or it blends in
  const auto restartStill = [this](const nvutils::ParameterBase* const) { resetFrame(); };
  m_settings.add({.name = "hdrIntensity", .help = "HDR Environment Intensity", .callbackSuccess = restartStill},
                 &m_resources.settings.hdrEnvIntensity, Persist::eYes, 0.0F, 100.0F);
  // One orientation for every environment -- OMI_environment_sky's `rotation`, in its own form. It
  // is also where north is, so like the Time of Day settings, an edit re-places
  // the sun after the ini restore (see m_todDrivesSun).
  m_settings.addVector({.name = "envRotation",
                        .help = "Environment rotation: unit quaternion x y z w (glTF order); turns the HDR, "
                                "every sky, and the compass the sun is placed with",
                        .callbackSuccess =
                            [this](const nvutils::ParameterBase* const) {
                              if(skyHasSun(m_resources.settings.envSystem))
                                m_todDrivesSun = true;
                              onEnvironmentRotated();
                            }},
                       &m_resources.settings.envRotation, Persist::eYes);
  m_settings.add({.name = "hdrBlur", .help = "HDR Environment Blur", .callbackSuccess = restartStill},
                 &m_resources.settings.hdrBlur, Persist::eYes, 0.0F, 1.0F);
  m_settings.addVector({"silhouetteColor", "Color of the silhouette"}, &m_resources.settings.silhouetteColor, Persist::eYes);
  m_settings.add({"dbgVisualization", "Visualization Mode"}, (int*)&m_resources.settings.visualization, Persist::eYes);
  m_settings.add({"dbgWireframe", "Enable wireframe overlay"}, &m_resources.settings.wireframe, Persist::eYes);
  m_settings.add({"ptOptimalShader",
                  "Compile gltf_pathtrace.slang with GLTF_USE_* gates specialized per scene "
                  "(no runtime MAT_EXT_* changes; triggers shader recompile on scene/material change). Default off."},
                 &m_resources.settings.optimalShader, Persist::eYes);
  // A value rather than a trigger, so a host can turn it off again over MCP (triggers aren't exposed there).
  m_settings.add({.name = "useSolidBackground", .help = "Use solid color background", .callbackSuccess = restartStill},
                 &m_resources.settings.useSolidBackground, Persist::eYes);
  m_settings.addVector({.name = "solidBackgroundColor", .help = "Solid Background Color", .callbackSuccess = restartStill},
                       &m_resources.settings.solidBackgroundColor, Persist::eYes);
  m_settings.add({"ptMaxSamples", "Path tracer: stop accumulating at this many samples per pixel"},
                 &m_resources.settings.maxSamples, Persist::eYes, 1, 1000000);
  // Persisted. Headless and scripted runs neither read nor write the ini, so they can't change what
  // the next session plays.
  m_settings.add(
      {.name = "animPlay",
       .help = "Play animations [Off:0, On:1]. Default: on (last Play/Pause choice is remembered), off in headless runs. "
               "A scene with a behavior graph never autoplays",
       .callbackSuccess = [this](const nvutils::ParameterBase* const) { m_resources.animationControl.play = m_animPlay; }},
      &m_animPlay, Persist::eYes);
  // The ini restore writes m_animPlay directly, after a start-up scene may already have set
  // animationControl.play from the default. Only ever turn playback off here: a scene with a
  // behavior graph has already forced it off and must stay so.
  m_settings.addPostRestoreHook([this]() { m_resources.animationControl.play &= m_animPlay; });
  // Headless-only output path: consumed at start-up, so remembering it would be misleading.
  m_settings.add({"output", "Output image file path for headless mode"}, &m_resources.headlessOutputPath, Persist::eNo);
  m_settings.add({"outputSamples",
                  "Path tracer: also save the output image each time the displayed image reaches one of these "
                  "samples per pixel, e.g. \"1,2,4,8\". Written as <output>_<spp>spp<ext> (see --output); the "
                  "count is that of the image shown, denoised or not"},
                 &m_outputSamples, Persist::eNo);

  // Tonemapper. tmBrightness and tmVignette were once tmGamma and
  // tmWhitePoint -- names for fields TonemapperData does not have.
  m_settings.add({"tmMethod", "Tonemapper: Method [Filmic:0, Uncharted:1, Clip:2, ACES:3, AgX:4, KhronosPBR:5]"},
                 &m_resources.tonemapperData.method, Persist::eYes, 0, 5);
  m_settings.add({"tmActive", "Tonemapper: Enable tone mapping [Off:0, On:1]"}, &m_resources.tonemapperData.isActive,
                 Persist::eYes, 0, 1);
  // Floor below 0.1, which is not enough headroom for a bright environment with
  // auto-exposure off -- what a reproducible capture wants. Out-of-range values are clamped rather
  // than refused, so too high a floor is silent: the run succeeds and the image is simply not the
  // one that was asked for.
  m_settings.add({"tmExposure", "Tonemapper: Exposure multiplier"}, &m_resources.tonemapperData.exposure, Persist::eYes,
                 0.01F, 200.0F);
  m_settings.add({"tmContrast", "Tonemapper: Contrast"}, &m_resources.tonemapperData.contrast, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmBrightness", "Tonemapper: Brightness"}, &m_resources.tonemapperData.brightness, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmSaturation", "Tonemapper: Saturation"}, &m_resources.tonemapperData.saturation, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmVignette", "Tonemapper: Vignette (0 = none)"}, &m_resources.tonemapperData.vignette, Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmDither", "Tonemapper: Dither [Off:0, On:1]"}, &m_resources.tonemapperData.dither, Persist::eYes, 0, 1);
  // White balance
  m_settings.add({"tmTemperature", "Tonemapper: White balance temperature (Kelvin)"},
                 &m_resources.tonemapperData.temperature, Persist::eYes, 2000.0F, 15000.0F);
  m_settings.add({"tmTint", "Tonemapper: White balance tint (ANSI C78.377 Duv)"}, &m_resources.tonemapperData.tint,
                 Persist::eYes, -0.03F, 0.03F);
  // Colour grading
  m_settings.add({"tmVibrance", "Tonemapper: Vibrance (boosts muted colors only)"},
                 &m_resources.tonemapperData.vibrance, Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmShadowBias", "Tonemapper: Shadow tone bias"}, &m_resources.tonemapperData.shadowBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmMidtoneBias", "Tonemapper: Midtone brightness bias"}, &m_resources.tonemapperData.midtoneBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmHighlightBias", "Tonemapper: Highlight tone bias"}, &m_resources.tonemapperData.highlightBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.addVector({"tmCoolColor", "Tonemapper: Split-toning tint for shadows"},
                       &m_resources.tonemapperData.coolColor, Persist::eYes);
  m_settings.addVector({"tmWarmColor", "Tonemapper: Split-toning tint for highlights"},
                       &m_resources.tonemapperData.warmColor, Persist::eYes);
  m_settings.add({"tmSplitBalance", "Tonemapper: Split-toning cool/warm balance"},
                 &m_resources.tonemapperData.splitBalance, Persist::eYes, -0.5F, 0.5F);
  // Auto exposure (turn off for reproducible headless captures)
  m_settings.add({"tmAutoExposure", "Tonemapper: Auto-exposure [Off:0, On:1]"},
                 &m_resources.tonemapperData.autoExposure, Persist::eYes, 0, 1);
  m_settings.add({"tmAutoExposureSpeed", "Tonemapper: Auto-exposure adaptation speed"},
                 &m_resources.tonemapperData.autoExposureSpeed, Persist::eYes, 0.0F, 100.0F);
  m_settings.add({"tmEvMin", "Tonemapper: Auto-exposure minimum (EV100)"}, &m_resources.tonemapperData.evMinValue,
                 Persist::eYes, -24.0F, 24.0F);
  m_settings.add({"tmEvMax", "Tonemapper: Auto-exposure maximum (EV100)"}, &m_resources.tonemapperData.evMaxValue,
                 Persist::eYes, -24.0F, 24.0F);

  // Panel visibility: persisted before, but not settable. Exposed now so a scripted capture can
  // hide the chrome.

  // The sun. One direction is shared by every sky type that has a sun, which is why these are not
  // named sky*: Resources::sunDirection is the single source of truth and the angles drive it
  // through the callback. Everything else below is direct.
  //
  // SkySun is initialized here, not in onAttach: the command line is parsed between construction
  // and onAttach, and these callbacks fire during that parse.
  m_skySun.init(&m_resources,
                {
                    .resetFrame         = [this] { resetFrame(); },
                    .environmentPreview = [this] { requestEnvironmentPreview(); },
                });
  // SkySun::aim() reports the consequences: a re-bake for the physical sky (the only one whose image
  // depends on the sun) and an accumulation reset.
  const auto applySunAngles = [this](const nvutils::ParameterBase* const) {
    m_skySun.setAngles(m_resources.settings.sunAzimuth, m_resources.settings.sunElevation);
  };
  m_settings.add({.name = "sunAzimuth",
                  .help = "Sun azimuth as a compass bearing (degrees, 0 = north, 90 = east); shared by every sky type",
                  .callbackSuccess = applySunAngles},
                 &m_resources.settings.sunAzimuth, Persist::eYes, 0.0F, 360.0F);
  m_settings.add({.name = "sunElevation", .help = "Sun elevation (degrees); shared by every sky type", .callbackSuccess = applySunAngles},
                 &m_resources.settings.sunElevation, Persist::eYes, -90.0F, 90.0F);
  // ImGui.ini restore writes sunAzimuth/Elevation directly and never runs applySunAngles, so
  // Resources::sunDirection -- the value rendering actually uses -- would stay at
  // its default and the remembered sun position would silently not be restored. Re-run the
  // conversion once, after Application::run() has reloaded the ini (see onUIRender's one-shot).
  //
  // CLI precedence: the loadFilter installed in onAttach already skips ini writes for any key
  // ParameterParser::wasParsed() marked, so a CLI-set azimuth or elevation survives the ini
  // reload. This hook then recomputes sunDirection from the resulting per-key mix (CLI value
  // where the user overrode it, ini value otherwise), which is exactly the intended
  // CLI-wins-per-key behavior.
  m_settings.addPostRestoreHook([this, applySunAngles]() { applySunAngles(nullptr); });

  // Time of Day: the same sun, said differently. These do not add a second source of truth -- each
  // one recomputes sunAzimuth/sunElevation through m_skySun.applyTimeOfDay(), so the sun still has exactly
  // one owner and the two angles remain what is persisted, scripted and written to a marked light.
  //
  // Persisted so the widget reopens where it was left; the callback is what makes a command-line
  // --todHour actually move the sun. ini restore deliberately does not fire it: the sun's direction
  // is restored from its own persisted angles, and re-deriving it here would overwrite a sun that
  // had since been moved by hand.
  // Open on today and on this machine's clock, rather than on an arbitrary epoch somebody has to
  // correct before the widget says anything true. Read here, before the declarations below, so it
  // is the value SettingsRegistry captures as the default -- and therefore what "Reset All to
  // Default" comes back to.
  if(const sun_position::LocalClock now = sun_position::systemLocalClock(); now.valid)
  {
    m_resources.settings.todDate      = now.date;
    m_resources.settings.todHour      = now.hour;
    m_resources.settings.todUtcOffset = now.utcOffsetHours;
  }
  const auto applyTod = [this](const nvutils::ParameterBase* const) {
    m_todDrivesSun = true;
    m_skySun.applyTimeOfDay();
  };
  m_settings.add({.name = "todHour", .help = "Time of day, in local clock hours (0..24) at --todUtcOffset", .callbackSuccess = applyTod},
                 &m_resources.settings.todHour, Persist::eYes, 0.0F, 24.0F);
  m_settings.add({.name = "todDate", .help = "Date the sun is computed for, yyyy-mm-dd", .callbackSuccess = applyTod},
                 &m_resources.settings.todDate, Persist::eYes);
  m_settings.add({.name = "todLatitude", .help = "Observer latitude in degrees, positive north", .callbackSuccess = applyTod},
                 &m_resources.settings.todLatitude, Persist::eYes, -90.0F, 90.0F);
  m_settings.add({.name = "todLongitude", .help = "Observer longitude in degrees, positive east", .callbackSuccess = applyTod},
                 &m_resources.settings.todLongitude, Persist::eYes, -180.0F, 180.0F);
  m_settings.add({.name = "todUtcOffset", .help = "Hours the local clock runs ahead of UTC", .callbackSuccess = applyTod},
                 &m_resources.settings.todUtcOffset, Persist::eYes, -12.0F, 14.0F);

  // A city is an action, not a mode: it writes latitude, longitude and the offset, and nothing
  // reads it back -- the same shape as atmoPreset, and not persisted for the same reason.
  //
  // Named rather than indexed, unlike atmoPreset. Twenty-odd entries make an index unreadable in a
  // script, and worse, inserting a city would silently move every script and saved sequence that
  // used one. A name survives the list growing.
  m_settings.add({.name = "todCity",
                  .help = "Set latitude, longitude and UTC offset from a named city (see the Environment panel "
                          "for the list); standard time, no daylight saving",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        const int index = sun_position::findCity(m_resources.settings.todCity);
                        if(index < 0)
                        {
                          LOGW("--todCity: '%s' is not one of the listed cities; the location is unchanged.\n",
                               m_resources.settings.todCity.c_str());
                          return;
                        }
                        const sun_position::City& city    = sun_position::kCities[index];
                        m_resources.settings.todLatitude  = city.latitudeDeg;
                        m_resources.settings.todLongitude = city.longitudeDeg;
                        m_resources.settings.todUtcOffset = city.utcOffsetHours;
                        m_todDrivesSun                    = true;
                        m_skySun.applyTimeOfDay();
                      }},
                 &m_resources.settings.todCity, Persist::eNo);

  // Precedence, and the one place it has to be stated.
  //
  // The command line is parsed before the ini is read, so `--todHour 20` computes a sun and then
  // the sunAzimuth/sunElevation hook above puts the *remembered* sun back over it -- those two keys
  // were not on the command line, so the ini is allowed to supply them, and the hook that re-derives
  // the direction from them cannot tell it is undoing an explicit instruction. Silently: the run
  // simply renders yesterday's sun, which looks exactly like the widget not working.
  //
  // So if anything asked for a time of day, it asks again here, after the
  // restore. Registered after that hook, and hooks run in order, so it wins. With no such request
  // the flag is false and the remembered angles stand, which is what reopening the app should do.
  m_settings.addPostRestoreHook([this]() {
    if(m_todDrivesSun)
      m_skySun.applyTimeOfDay();
  });
  // Authored sky types (glTF OMI_environment_sky). Every one of these invalidates the baked
  // lighting environment, so they share one callback that marks it for a re-bake; the bake itself
  // runs on the application thread at the top of the next frame, never inside a parameter write.
  const auto invalidateBake = [this](const nvutils::ParameterBase* const) { onEnvironmentChanged(); };
  m_settings.add({.name = "envSaveToGltf", .help = "Write OMI_environment_sky into the glTF when saving the scene"},
                 &m_resources.settings.envSaveToGltf, Persist::eYes);
  m_settings.addVector({.name = "plainColor", .help = "Plain sky: solid color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.plainColor, Persist::eYes);
  m_settings.addVector({.name = "gradientBottomColor", .help = "Gradient sky: nadir color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientBottomColor, Persist::eYes);
  m_settings.addVector({.name = "gradientHorizonColor", .help = "Gradient sky: horizon color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientHorizonColor, Persist::eYes);
  m_settings.addVector({.name = "gradientTopColor", .help = "Gradient sky: zenith color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientTopColor, Persist::eYes);
  m_settings.addVector({.name = "gradientSunColor", .help = "Gradient sky: sun color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientSunColor, Persist::eYes);
  m_settings.add({.name = "gradientBottomCurve", .help = "Gradient sky: horizon-to-nadir falloff", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientBottomCurve, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "gradientTopCurve", .help = "Gradient sky: horizon-to-zenith falloff", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientTopCurve, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "gradientSunAngleMax", .help = "Gradient sky: angular extent of the sun glow (radians)", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientSunAngleMax, Persist::eYes, 0.0F, 3.1416F);
  m_settings.add({.name = "gradientSunCurve", .help = "Gradient sky: sun-to-sky falloff across the glow", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientSunCurve, Persist::eYes, 0.0F, 1.0F);

  // The physical sky's atmosphere.
  //
  // Every one of these goes through the same preview-then-settle path a sky colour does. Most of
  // them change nothing but the scattering tables, so the drag rebuilds those at preview quality
  // and the commit rebuilds them at full -- see SkyBruneton::Quality. Ground albedo, sun angular
  // radius and altitude are the exceptions that the bake also reads directly, so they move even
  // before the tables catch up.
  const auto invalidateAtmosphere = [this](const nvutils::ParameterBase* const) { requestEnvironmentPreview(); };
  m_settings.add({.name = "atmoSunAngularRadius", .help = "Physical sky: angular radius of the sun disk (radians)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoSunAngularRadius, Persist::eYes, 0.0F, 0.2F);
  m_settings.addVector({.name = "atmoGroundAlbedo", .help = "Physical sky: ground reflectance, linear RGB", .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoGroundAlbedo, Persist::eYes);
  // Altitude changes no LUT -- it is a bake input only -- so it is the cheap one of the three.
  m_settings.add({.name            = "atmoObserverAltitude",
                  .help            = "Physical sky: observer height above the ground (metres)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoObserverAltitude, Persist::eYes, 0.0F, 60000.0F);

  // Aerial perspective. Both shading paths read it straight out of SceneFrameInfo, so unlike every
  // other atmosphere setting it needs no re-bake and no LUT rebuild: it changes what a frame does
  // with the tables, not the tables. Hence resetFrame() alone.
  m_settings.add({.name            = "atmoAerialPerspectiveScale",
                  .help            = "Aerial perspective: metres of air per scene unit (0 disables)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.atmoAerialPerspectiveScale, Persist::eYes, 0.0F, 1000.0F);

  m_settings.addVector({.name            = "atmoSolarIrradiance",
                        .help            = "Physical sky: solar irradiance at the top of the atmosphere (W/m^2)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoSolarIrradiance, Persist::eYes);

  // Rayleigh -- the air itself.
  m_settings.addVector({.name            = "atmoRayleighScattering",
                        .help            = "Physical sky: Rayleigh scattering at the surface (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoRayleighScattering, Persist::eYes);
  m_settings.add({.name            = "atmoRayleighScaleHeight",
                  .help            = "Physical sky: height over which the air thins by 1/e (km)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoRayleighScaleHeight, Persist::eYes, 0.1F, 100.0F);

  // Mie -- the aerosol.
  m_settings.addVector({.name            = "atmoMieScattering",
                        .help            = "Physical sky: Mie scattering at the surface (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoMieScattering, Persist::eYes);
  m_settings.add({.name            = "atmoMieScaleHeight",
                  .help            = "Physical sky: height over which the aerosol thins by 1/e (km)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieScaleHeight, Persist::eYes, 0.1F, 100.0F);
  m_settings.add({.name            = "atmoMieAnisotropy",
                  .help            = "Physical sky: Cornette-Shanks asymmetry; > 0 scatters forward",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieAnisotropy, Persist::eYes, -0.99F, 0.99F);
  m_settings.add({.name            = "atmoMieAlbedo",
                  .help            = "Physical sky: aerosol single-scattering albedo; 1 scatters all it removes",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieAlbedo, Persist::eYes, 0.05F, 1.0F);

  // Ozone -- an absorbing layer well above the ground, and what keeps a clear zenith blue rather
  // than grey at low sun.
  m_settings.addVector({.name            = "atmoOzoneExtinction",
                        .help            = "Physical sky: absorption at the peak of the ozone layer (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoOzoneExtinction, Persist::eYes);
  m_settings.add({.name = "atmoOzoneCenter", .help = "Physical sky: altitude of the ozone peak (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoOzoneCenter, Persist::eYes, 0.0F, 200.0F);
  m_settings.add({.name = "atmoOzoneWidth", .help = "Physical sky: full thickness of the ozone layer (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoOzoneWidth, Persist::eYes, 0.1F, 400.0F);

  // The planet.
  m_settings.add({.name = "atmoBottomRadius", .help = "Physical sky: planet radius (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoBottomRadius, Persist::eYes, 100.0F, 20000.0F);
  m_settings.add({.name = "atmoThickness", .help = "Physical sky: atmosphere depth above the surface (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoThickness, Persist::eYes, 1.0F, 500.0F);

  // A preset is an action, not a mode: its callback overwrites every field above and nothing reads
  // it back. That is also why it does not persist -- restoring it after the individual values
  // would undo them.
  m_settings.add({.name = "atmoPreset",
                  .help = "Physical sky: apply an atmosphere preset [Earth:0, Mars:1, Alien:2]",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        applyAtmospherePreset(m_resources.settings, m_resources.settings.atmoPreset);
                        requestEnvironmentPreview();
                      }},
                 &m_resources.settings.atmoPreset, Persist::eNo, 0, eAtmospherePresetCount - 1);

  // The MDL sky's eleven knobs (skyMultiplier, skyHaze, skyRedBlueShift, skySaturation,
  // skyHorizonHeight, skyHorizonBlur, skyGroundColor, skyNightColor, skySunDiskScale,
  // skySunDiskIntensity, skySunGlowIntensity) were registered here until that sky was deleted.
  // They are deliberately not re-registered under new names: the Bruneton atmosphere is not a
  // renaming of those parameters, it is a different model. A persisted .ini still carrying them
  // is handled by SettingsRegistry, which ignores keys it does not know.

  // Live: the retained scene's next batch of BLAS builds regrows its scratch when it needs more.
  m_settings.add({.name = "blasScratchBudgetMB",
                  .help = "Retained scene: the scratch one batch of BLAS builds shares, in MB",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        m_resources.settings.blasScratchBudgetMB = std::max(m_resources.settings.blasScratchBudgetMB, 1u);
                        if(m_resources.retainedGpu)
                          m_resources.retainedGpu->setBlasScratchBudget(VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20);
                      }},
                 &m_resources.settings.blasScratchBudgetMB, Persist::eNo);
  // Start-up only, so not exposed to a host (fv_parameter_filter.hpp): read at each retained scene
  // replace, where a smaller capacity than the scene needs only grows by doubling.
  // Live: a smaller budget trims the grace list at once.
  m_settings.add({.name = "textureGraceMB",
                  .help = "Retained scene: decoded images nothing samples, kept for the next scene or edit, in MB",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        for(rs::RetainedSceneVk* retained : {m_resources.retainedGpu.get(), m_resources.retainedSpare.get()})
                          if(retained)
                            retained->setTextureGrace(VkDeviceSize(m_resources.settings.textureGraceMB) << 20);
                      }},
                 &m_resources.settings.textureGraceMB, Persist::eNo);
  m_settings.add({.name = "sceneBudgetMB",
                  .help = "Retained scene: the GPU memory the scene is fitted to, where less than the device's budget, in MB (0: that budget)"},
                 &m_resources.settings.sceneBudgetMB, Persist::eNo);
  m_settings.add({.name = "residencyMinInstances", .help = "Retained scene: the least instance capacity (start-up only)"},
                 &m_resources.settings.residencyMinInstances, Persist::eNo);
  // Live: from the next refit on (P7, measured on Path Trace after a pose far from the load pose).
  m_settings.add({.name = "blasRefitsBeforeRebuild",
                  .help = "Retained scene: a BLAS refitted this many times is built again (0: never)",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        for(rs::RetainedSceneVk* retained : {m_resources.retainedGpu.get(), m_resources.retainedSpare.get()})
                          if(retained)
                            retained->setRefitsBeforeRebuild(m_resources.settings.blasRefitsBeforeRebuild);
                      }},
                 &m_resources.settings.blasRefitsBeforeRebuild, Persist::eNo);
  // Live: for the BLASes built from then on (P7, B56: measured on the scene's BLAS memory).
  m_settings.add({.name = "blasCompactStatic",
                  .help = "Retained scene: a static BLAS is compacted once its size is known",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        for(rs::RetainedSceneVk* retained : {m_resources.retainedGpu.get(), m_resources.retainedSpare.get()})
                          if(retained)
                            retained->setCompactStatic(m_resources.settings.blasCompactStatic);
                      }},
                 &m_resources.settings.blasCompactStatic, Persist::eNo);
  // Debug, live (read at the next scene replace): every composition then retires the old scene's
  // slots and reallocates, so holes, slot reuse, capacity growth and deferred frees run on the GPU
  // before structural edits make them ordinary (daz-viewer docs/retained-scene-plan.md, P4).
  m_settings.add({.name = "residencyRetireOnReplace",
                  .help = "Retained scene (debug): a scene replace retires the old scene's slots and reallocates instead of starting over"},
                 &m_resources.settings.residencyRetireOnReplace, Persist::eNo);

  // Gizmo, grid and snap: previously persisted but unreachable from the command line or MCP.
  m_settings.add({"uiGridUnit", "Grid base unit (world units)"}, &m_resources.settings.gridUnit, Persist::eYes);
  m_settings.add({"uiSnapEnabled", "Snap gizmo transforms to grid increments"}, &m_resources.settings.snapEnabled, Persist::eYes);
  m_settings.add({"uiSnapRotation", "Rotation snap increment (degrees)"}, &m_resources.settings.snapRotation, Persist::eYes);
  m_settings.add({"uiSnapScale", "Scale snap increment"}, &m_resources.settings.snapScale, Persist::eYes);

  // The ground: the path tracer's infinite plane and the rasterizer's floor pass draw the same one.
  const auto groundChanged = [this](const nvutils::ParameterBase* const) { resetFrame(); };
  m_settings.add({.name = "useInfinitePlane", .help = "Ground: draw an infinite plane under the scene", .callbackSuccess = groundChanged},
                 &m_resources.settings.useInfinitePlane, Persist::eYes);
  m_settings.add({.name = "isShadowCatcher", .help = "Ground: the plane only darkens the background where shadows fall", .callbackSuccess = groundChanged},
                 &m_resources.settings.isShadowCatcher, Persist::eYes);
  m_settings.add({.name = "infinitePlaneDistance", .help = "Ground: height of the plane (world units)", .callbackSuccess = groundChanged},
                 &m_resources.settings.infinitePlaneDistance, Persist::eNo);
  m_settings.addVector({.name = "infinitePlaneBaseColor", .help = "Ground: plane color (linear)", .callbackSuccess = groundChanged},
                       &m_resources.settings.infinitePlaneBaseColor, Persist::eYes);
  m_settings.add({.name = "infinitePlaneMetallic", .help = "Ground: plane metalness", .callbackSuccess = groundChanged},
                 &m_resources.settings.infinitePlaneMetallic, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "infinitePlaneRoughness", .help = "Ground: plane roughness", .callbackSuccess = groundChanged},
                 &m_resources.settings.infinitePlaneRoughness, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "shadowCatcherDarkness", .help = "Ground: extra darkening of caught shadows (0..1)", .callbackSuccess = groundChanged},
                 &m_resources.settings.shadowCatcherDarkness, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "groundFade", .help = "Ground: fade the plane out over a disk of this many scene radii (0: no fade)", .callbackSuccess = groundChanged},
                 &m_resources.settings.groundFade, Persist::eYes, 0.0F, 100.0F);
  m_settings.add({.name = "skySunScale", .help = "Sky: multiplier on the light the sky's sun gives (0: no sun light)", .callbackSuccess = groundChanged},
                 &m_resources.settings.skySunScale, Persist::eNo, 0.0F, 100.0F);

  // The rasterizer's sun shadow map.
  const auto shadowChanged = [this](const nvutils::ParameterBase* const) { resetFrame(); };
  m_settings.add({.name = "rasterShadows", .help = "Rasterizer: sun shadow map [Off:0, On:1]", .callbackSuccess = shadowChanged},
                 &m_resources.settings.rasterShadows, Persist::eYes);
  m_settings.add({.name = "shadowLight",
                  .help = "Rasterizer: scene light index that casts the shadow map, or -1 for the sun (the marked sun light, "
                          "else the sky's sun, else the brightest directional light)",
                  .callbackSuccess = shadowChanged},
                 &m_resources.settings.shadowLight, Persist::eNo, -1, 1 << 20);
  m_settings.add({.name = "shadowMapSize", .help = "Rasterizer: shadow map size in texels (1024..8192)", .callbackSuccess = shadowChanged},
                 &m_resources.settings.shadowMapSize, Persist::eYes, 1024, 8192);
  m_settings.add({.name = "shadowContactHardening",
                  .help = "Rasterizer: penumbrae widen with the distance to the occluder, from the light's angular size [Off:0, On:1]",
                  .callbackSuccess = shadowChanged},
                 &m_resources.settings.shadowContactHardening, Persist::eYes);
  m_settings.add({.name = "shadowBias", .help = "Rasterizer: shadow depth bias, in shadow-map texels", .callbackSuccess = shadowChanged},
                 &m_resources.settings.shadowBias, Persist::eYes, 0.0F, 50.0F);
  m_settings.add({.name = "shadowNormalBias", .help = "Rasterizer: shadow normal offset, in shadow-map texels", .callbackSuccess = shadowChanged},
                 &m_resources.settings.shadowNormalBias, Persist::eYes, 0.0F, 50.0F);

  // Register PathTracer-specific command line parameters
  m_pathTracer.registerParameters(&m_settings, [this]() { resetFrame(); });
  m_rasterizer.registerParameters(&m_settings);

  // Saves the viewport without its overlays (scripts/render-scene.py). The next frame is drawn that
  // way and saved once submitted (onPostRender).
  paramReg->add({.name = "screenshot",
                 .help = "Save the viewport to this image, without its overlays",
                 .callbackSuccess =
                     [this](const nvutils::ParameterBase* const) {
                       if(m_app && !m_screenshotParam.empty())
                         m_screenshotPending = std::exchange(m_screenshotParam, {});
                     }},
                {".png", ".jpg", ".jpeg"}, &m_screenshotParam);

  // Initialize camera manipulator
  m_cameraManip = std::make_shared<DragCameraManipulator>();
  m_resources.cameraManip = m_cameraManip;  // Share with resources
}

//--------------------------------------------------------------------------------------------------
// The onAttach method is called when the application is attached to the renderer
void GltfRenderer::onAttach(nvapp::Application* app)
{
  SCOPED_TIMER("GltfRenderer::onAttach");

  m_app                = app;
  m_device             = app->getDevice();
  m_resources.instance = app->getInstance();
  m_resources.app      = app;

  // VK_EXT_memory_budget is asked for as an optional extension (main.cpp), so a device that has it
  // has it enabled: deviceMemoryBudget() may chain its query then
  {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(app->getPhysicalDevice(), nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(app->getPhysicalDevice(), nullptr, &count, extensions.data());
    m_memoryBudgetExtension = std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& e) {
      return std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    });
  }

  // ===== Settings Handler (ImGui persistent) =====
  // The list lives with the declarations in the constructor; this replays the persisted ones now
  // that the handler exists. Nothing to keep in sync by hand.
  m_settingsHandler.setHandlerName("GltfRenderer");
  m_settings.applyPersistence(m_settingsHandler);
  m_settingsHandler.setLoadFilter([this](const std::string& key) {
    // Skip loading settings that were explicitly set via the command line
    return !(m_parameterParser && m_parameterParser->wasParsed(key));
  });
  m_settingsHandler.addImGuiHandler();

  // ===== Memory Allocation & Buffer Management =====
  m_resources.allocator.init({
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = app->getPhysicalDevice(),
      .device           = app->getDevice(),
      .instance         = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });  // Allocator

  // If there is a leak (nvvkAllocID -> ID)
  // m_resources.allocator.setLeakID(155);


  m_transientCmdPool = nvvk::createTransientCommandPool(m_device, app->getQueue(0).familyIndex);
  NVVK_DBG_NAME(m_transientCmdPool);

  m_loadPipeline.init(m_device, app->getQueue(0).queue, m_transientCmdPool);

  // FrameUploader records copies into the caller's command buffer and retires staging
  // once the frame timeline semaphore (or a load-pipeline wait) has signaled.
  // blockSize is the max size of a single append (circular allocator constraint) and
  // is larger than the library default so scene-create can stage large meshes/textures.
  NVVK_CHECK(m_resources.staging.init({
      .allocator = &m_resources.allocator,
      .blockSize = kFrameUploaderBlockSize,  // shared with createSampledImage's row-strip chunking
      .debugName = "frameUploads",
  }));

  m_resources.commandPool = app->getCommandPool();


  // ===== Texture & Image Resources =====
  m_resources.samplerPool.init(m_device);
  VkSampler linearSampler{};
  NVVK_CHECK(m_resources.samplerPool.acquireSampler(linearSampler));
  NVVK_DBG_NAME(linearSampler);

  // IBL environment map
  m_resources.hdrIbl.init(&m_resources.allocator, &m_resources.samplerPool);
  // Required before HdrIbl::updateFromGpuImage: the file-load path writes the per-texel PDF into
  // the image alpha on the CPU, but a GPU-baked environment is never read back, so the PDF write
  // is a compute pass that has to be created up front. Without it EnvBaker::commit dispatches a
  // null pipeline -- caught only by an assert, so a release build simply crashes.
  m_resources.hdrIbl.initGpuWriter(std::span(env_write_pdf_slang));
  m_resources.hdrDome.init(&m_resources.allocator, &m_resources.samplerPool, m_app->getQueue(0));

  // Application-level memory tracker (frame targets, DLSS, OptiX images)
  m_resources.appMemoryTracker.init(&m_resources.allocator);

  // G-Buffer
  NVVK_CHECK(m_resources.gBuffers.init({.device = m_device,
                                        .alloc  = &m_resources.allocator,
                                        .colorFormats =
                                            {
                                                VK_FORMAT_R8G8B8A8_UNORM,       // Tonemapped (eImgTonemapped)
                                                VK_FORMAT_R32G32B32A32_SFLOAT,  // Rendered image (eImgRendered)
                                                VK_FORMAT_R32_SFLOAT,  // ObjectID for selection/silhouette (eImgSelection), .r = render node ID+1
                                            },
                                        .depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice()),
                                        .debugName   = "GBuffers"}));
  m_resources.linearSampler = linearSampler;
  {
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    NVVK_CHECK(m_resources.gBuffers.update(cmd, {100, 100}));
    m_resources.gBuffers.cmdClear(cmd);
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
    m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  }
  m_resources.tonemappedUi.init(m_resources.gBuffers.getUiImageView(Resources::eImgTonemapped));

  // ===== Rendering Utilities =====

  // Ray picker
  m_rayPicker.init(&m_resources.allocator);

  // Tonemapper
  m_resources.tonemapper.init(&m_resources.allocator, tonemapper_slang);

  // Silhouette renderer
  m_silhouette.init(m_resources);


  // Producer for analytic skies. Its lat-long image feeds the same HdrIbl / HdrEnvDome pair the
  // HDR file path uses, so both render paths consume one environment regardless of sky type.
  // Allocated here but only dispatched once a baked sky type is selected.

  // Bruneton atmosphere LUTs. Built once here from the Earth defaults; they depend only on the
  // atmosphere, never on the sun or the camera, so moving the sun later costs nothing.
  m_skyBruneton.init(m_resources, SkyBruneton::Shaders{.precompute = std::span(sky_bruneton_precompute_slang)});
  // Seed from the settings, not from the Earth defaults init() installs. Otherwise the first
  // refreshBakedEnvironment() finds the parameters changed and rebuilds the whole set immediately
  // -- two full precomputes, every startup, even when nothing was customised.
  m_skyBruneton.setParameters(buildAtmosphereParameters());
  m_skyBruneton.precompute(m_resources);

  // After SkyBruneton: EnvBaker takes its descriptor-set layout to build a pipeline layout that
  // covers both producers.
  m_envBaker.init(m_resources, m_skyBruneton.getDescriptorSetLayout(), m_skyBruneton.getDescriptorSet());
  m_envBaker.setResolution(m_resources, kEnvBakeSize);

  // Publish the same set to the two renderers, which read the tables directly for aerial
  // perspective. Here rather than in each of them: they build their pipeline layouts from
  // Resources, and neither needs to know which class owns the LUTs.
  m_resources.skyLutDescriptorSetLayout = m_skyBruneton.getDescriptorSetLayout();
  m_resources.skyLutDescriptorSet       = m_skyBruneton.getDescriptorSet();

  // ===== Profiling & Performance =====
  {
    SCOPED_TIMER("Profiler");
    m_profilerTimeline = g_profilerManager.createTimeline({.name = "Primary Timeline"});
    m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(),
                            int32_t(m_app->getQueue(0).familyIndex), false);
  }


  // ===== Shader Compilation =====
  {
    SCOPED_TIMER("Shader Slang");
    using namespace slang;
    m_resources.slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_resources.slangCompiler.defaultTarget();
    m_resources.slangCompiler.defaultOptions();

    // Specific options for this sample. Standard debug info (NonSemantic.Shader.DebugInfo, for
    // Nsight) only on NVIDIA: AMD's Vulkan driver crashes compiling SPIR-V that carries it, while
    // minimal debug info (OpLine) is accepted everywhere. Embedded shaders follow
    // SHADER_DEBUG_LEVEL in CMake instead.
    VkPhysicalDeviceProperties gpuProps{};
    vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &gpuProps);
    constexpr uint32_t kNvidiaVendorId = 0x10DE;
    const int debugInfoLevel = (gpuProps.vendorID == kNvidiaVendorId) ? SLANG_DEBUG_INFO_LEVEL_STANDARD : SLANG_DEBUG_INFO_LEVEL_MINIMAL;
    m_resources.slangCompiler.addOption({CompilerOptionName::DebugInformation, {CompilerOptionValueKind::Int, debugInfoLevel}});
    m_resources.slangCompiler.addOption(
        {CompilerOptionName::Optimization, {CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_NONE}});

    // Enable specific capabilities for better performance and features.
    m_resources.slangCompiler.addCapability("spvShaderInvocationReorderEXT");  // SER via VK_EXT_ray_tracing_invocation_reorder
    m_resources.slangCompiler.addCapability("spvInt64Atomics");                // # 64-bit atomic operations
    m_resources.slangCompiler.addCapability("spvShaderClockKHR");              // # Shader clock for profiling
    m_resources.slangCompiler.addCapability("spvRayTracingMotionBlurNV");  // # Motion blur for ray tracing
    m_resources.slangCompiler.addCapability("spvRayQueryKHR");             // # Ray query operations
    m_resources.slangCompiler.addCapability("spvGroupNonUniformBallot");  // # Ballot operations for subgroup functionality
    m_resources.slangCompiler.addCapability("spvGroupNonUniformArithmetic");  // # Arithmetic operations across subgroups
    m_resources.slangCompiler.addCapability("nonuniformqualifier");  // # NonUniformResourceIndex on the bindless arrays (also in CMake)

#if defined(USE_DLSS)
    m_resources.slangCompiler.addMacro({"HAS_DLSS_MOTION", "1"});
#endif

#if defined(AFTERMATH_AVAILABLE)
    // This aftermath callback is used to report the shader hash (Spirv) to the Aftermath library.
    m_resources.slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
      std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
      AftermathCrashTracker::getInstance().addShaderBinary(data);
    });
#endif
  }

  // ===== Renderer Initialization =====

  // Create resources
  createDescriptorSets();
  createHDR("");  // Dummy HDR
  createResourceBuffers();

  // Initialize the renderers
  m_pathTracer.setCacheDir(m_cacheDir);
  m_pathTracer.onAttach(m_resources, &m_profilerGpuTimer);
  m_pathTracer.setProfilerTimeline(m_profilerTimeline);
  m_pathTracer.setBusyState(&m_busy);
  m_rasterizer.onAttach(m_resources, &m_profilerGpuTimer);

  m_pathTracer.createPipeline(m_resources);
  m_rasterizer.createPipeline(m_resources);
  // the RTX pipeline the first path-traced frame needs, built on a thread while the app starts
  m_pathTracer.startPipelineWarm(m_resources);

  // ===== Visual Helpers (Grid + Transform Gizmo) =====
  {
    VkFormat depthFormat = m_resources.gBuffers.getDepthFormat();
    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Matches eImgTonemapped

    VisualHelpers::Resources helperRes{
        .app           = m_app,
        .alloc         = &m_resources.allocator,
        .uploader      = &m_resources.staging,
        .device        = m_device,
        .sampler       = linearSampler,
        .slangCompiler = &m_resources.slangCompiler,
        .colorFormat   = colorFormat,
        .depthFormat   = depthFormat,
    };
    m_visualHelpers.init(helperRes);

    m_visualHelpers.transform.setOnTransformChange([this]() {
      if(!m_gizmoEntity.isNone() && m_gizmoHooks.edit)
        m_gizmoHooks.edit(m_gizmoEntity, m_gizmoPosition, glm::quat(glm::radians(m_gizmoRotation)), m_gizmoScale, false);
    });

    m_visualHelpers.transform.setOnTransformEnd([this]() {
      if(!m_gizmoEntity.isNone() && m_gizmoHooks.edit)
        m_gizmoHooks.edit(m_gizmoEntity, m_gizmoPosition, glm::quat(glm::radians(m_gizmoRotation)), m_gizmoScale, true);
    });
  }

}


//--------------------------------------------------------------------------------------------------
// Detach the renderers and destroy the resources
void GltfRenderer::onDetach()
{
  // SYNC NOTE: Full device wait during shutdown is the standard Vulkan teardown pattern.
  vkDeviceWaitIdle(m_device);
  m_visualHelpers.deinit();
  m_pathTracer.onDetach(m_resources);
  m_rasterizer.onDetach(m_resources);
  destroyResources();
}

//--------------------------------------------------------------------------------------------------
// Resize the G-Buffer and the renderers
void GltfRenderer::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  NVVK_CHECK(m_resources.gBuffers.update(cmd, size));
  m_resources.gBuffers.cmdClear(cmd);
  m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  m_pathTracer.onResize(cmd, size, m_resources);
  m_rasterizer.onResize(cmd, size, m_resources);
  m_resources.hdrDome.setOutImage(m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgRendered));
  m_resources.tonemappedUi.update(m_resources.gBuffers.getUiImageView(Resources::eImgTonemapped));

  // Resize visual helpers (depth buffer + scene depth descriptor set)
  m_visualHelpers.onResize(cmd, size, m_resources.gBuffers.getDepthImage(), m_resources.gBuffers.getDepthImageView(),
                           m_resources.linearSampler);

  // Camera (was handled by ElementCamera, now owned by the renderer)
  m_cameraManip->setWindowSize({size.width, size.height});
  m_cameraManip->adjustOrthographicAspect();

  resetFrame();  // Reset frame to restart the rendering
}

//--------------------------------------------------------------------------------------------------
bool GltfRenderer::isHeadlessMode() const
{
  return m_app != nullptr && m_app->isHeadless();
}

bool GltfRenderer::isAutomatedRun() const
{
  return isHeadlessMode();
}

void GltfRenderer::alignMaxSamplesForHeadless(uint32_t headlessFrames)
{
  // An explicit --ptMaxSamples is the point of the run (e.g. --frames 2000 --ptMaxSamples 1024)
  if(m_parameterParser && m_parameterParser->wasParsed("ptMaxSamples"))
    return;
  const int samplesPerFrame =
      m_pathTracer.m_adaptiveSampling ? PathTracer::MAX_ADAPTIVE_SAMPLES_PER_PIXEL : m_pathTracer.m_pushConst.numSamples;
  const int64_t needed = std::min<int64_t>(int64_t(headlessFrames) * std::max(samplesPerFrame, 1), INT_MAX);
  if(m_resources.settings.maxSamples < needed)
  {
    LOGW("ptMaxSamples (%d) is less than headless --frames (%u) x %d spp per frame; setting it to %lld\n",
         m_resources.settings.maxSamples, headlessFrames, samplesPerFrame, static_cast<long long>(needed));
    m_resources.settings.maxSamples = static_cast<int>(needed);
  }
}

void GltfRenderer::applyHeadlessDefaults()
{
  if(!(m_parameterParser && m_parameterParser->wasParsed("animPlay")))
    m_animPlay = false;
  m_resources.animationControl.play = m_animPlay;
  // Nobody watches a headless frame: Auto SPP aims at throughput rather than a frame rate
  if(!(m_parameterParser && m_parameterParser->wasParsed("ptPerformanceTarget")))
    m_pathTracer.m_performanceTarget = PathTracer::PerformanceTarget::eThroughput;
}

//--------------------------------------------------------------------------------------------------
// End of a headless run: the cost of the AI denoisers that ran. OIDN is GPU time inside the frame;
// OptiX runs outside it (queue drain, copies, CUDA), so its breakdown is host wall time.
void GltfRenderer::logDenoiserTimings() const
{
#if defined(USE_OIDN_DENOISER)
  if(const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser(); oidn && oidn->hasValidDenoisedOutput() && oidn->getGpuTimeMs() > 0.0)
    LOGI("DENOISER_TIMING oidn gpu_ms=%.3f\n", oidn->getGpuTimeMs());
#endif
#if defined(USE_OPTIX_DENOISER)
  if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser(); optix && optix->hasValidDenoisedOutput())
  {
    const OptiXDenoiser::Timings t = optix->getTimings();
    LOGI("DENOISER_TIMING optix total_ms=%.3f drain_ms=%.3f copy_in_ms=%.3f invoke_ms=%.3f invoke_gpu_ms=%.3f copy_out_ms=%.3f\n",
         t.totalMs, t.drainMs, t.copyInMs, t.invokeMs, t.invokeGpuMs, t.copyOutMs);
  }
#endif
}

std::filesystem::path GltfRenderer::headlessOutputPathOrDefault() const
{
  return m_resources.headlessOutputPath.empty() ? nvutils::getExecutablePath().replace_extension(".jpg") : m_resources.headlessOutputPath;
}

//--------------------------------------------------------------------------------------------------
// --outputSamples: the samples per pixel of the image the tonemapper shows. A denoised image counts
// the samples it was denoised from (OptiX runs one frame behind, OIDN in the same frame). Returns
// -1 for an image that is not a path-traced accumulation (the DLSS hand-off's blend included).
int GltfRenderer::displayedImageSamples(DisplayBuffer& source) const
{
  source = m_resources.settings.displayBuffer;
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_pathTracer.handoffImage(m_resources))
    return -1;
#if defined(USE_OPTIX_DENOISER)
  if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser();
     source == DisplayBuffer::eOptixDenoised && optix && optix->hasValidDenoisedOutput())
    return optix->denoisedSamples();
#endif
#if defined(USE_OIDN_DENOISER)
  if(const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser();
     source == DisplayBuffer::eOidnDenoised && oidn && oidn->hasValidDenoisedOutput())
    return oidn->denoisedSamples();
#endif
  source = DisplayBuffer::eRendered;  // what runTonemapPass falls back to
  return m_resources.accumulatedSamples;
}

//--------------------------------------------------------------------------------------------------
// --outputSamples: parse the counts when the option changed. They are also stops for the path tracer
// (Resources::sampleStops): a frame ends exactly on each, so none is stepped over, whatever the samples
// per frame. Called before the frame renders.
void GltfRenderer::refreshSampleTargets()
{
  SampleCapture& capture = m_sampleCapture;
  if(capture.parsedFrom == m_outputSamples)
    return;
  capture          = {.parsedFrom = m_outputSamples};
  std::string list = m_outputSamples;
  std::replace(list.begin(), list.end(), ',', ' ');
  std::istringstream stream(list);
  for(int value; stream >> value;)
    if(value > 0)
      capture.targets.push_back(value);
  std::sort(capture.targets.begin(), capture.targets.end());
  capture.targets.erase(std::unique(capture.targets.begin(), capture.targets.end()), capture.targets.end());
  m_resources.sampleStops = capture.targets;
}

//--------------------------------------------------------------------------------------------------
// --outputSamples: one run saves a whole convergence series instead of one run per sample count.
// When the accumulation restarts, the counts from the current one on are captured again, so the saved
// files always come from the last accumulation. When another image is shown (e.g. the first denoised
// one), the current count is captured again from it, and the counts no image reached yet still are.
// A count skipped over (several samples per frame, a denoiser interval) is saved at the first image
// past it, named by its actual count.
void GltfRenderer::updateSampleCaptures()
{
  SampleCapture& capture = m_sampleCapture;
  refreshSampleTargets();
  if(capture.targets.empty() || m_resources.settings.renderSystem != RenderingMode::ePathtracer)
    return;

  // A .hdr capture saves the rendered image, never a denoiser's (saveViewportImage): name it after
  // that image's samples, not after the denoised one on screen, which OptiX keeps a frame behind.
  const bool    linear = nvutils::extensionMatches(headlessOutputPathOrDefault(), ".hdr");
  DisplayBuffer source{};
  const int     samples = linear ? m_resources.accumulatedSamples : displayedImageSamples(source);
  if(linear)
    source = DisplayBuffer::eRendered;
  if(samples <= 0)
    return;
  const size_t current = std::lower_bound(capture.targets.begin(), capture.targets.end(), samples) - capture.targets.begin();
  if(samples < capture.lastSamples)  // the accumulation restarted
    capture.next = current;
  else if(int(source) != capture.lastSource)  // another image: keep the counts not saved yet
    capture.next = std::min(capture.next, current);
  capture.lastSamples = samples;
  capture.lastSource  = int(source);

  if(capture.next < capture.targets.size() && capture.targets[capture.next] <= samples)
  {
    while(capture.next < capture.targets.size() && capture.targets[capture.next] <= samples)
      capture.next++;
    const std::filesystem::path output = headlessOutputPathOrDefault();
    capture.pending                    = output.parent_path()
                      / (output.stem().string() + "_" + std::to_string(samples) + "spp" + output.extension().string());
  }
}

//--------------------------------------------------------------------------------------------------
// The frame is submitted: save the image updateSampleCaptures() asked for. saveImageToFile records
// its copy on the same queue and waits for it, so it sees this frame's tonemapped image.
void GltfRenderer::onPostRender()
{
  if(!m_app)
    return;
  // A still's frames run one at a time. Queued, they gain a sample nothing can see, and an edit arriving
  // now would wait for all of them before its first moving frame, freezing and then jumping the image
  // at the start of every action. (A moving view's frames queue, for their rate; a window presents after this.)
  if(m_streamHost && m_resources.settings.renderSystem == RenderingMode::ePathtracer && !m_pathTracer.dlssFrame())
  {
    const nvvk::SemaphoreInfo frame = m_app->getFrameSignalSemaphore();
    const VkSemaphoreWaitInfo wait{.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .semaphoreCount = 1,
                                   .pSemaphores    = &frame.semaphore,
                                   .pValues        = &frame.value};
    NVVK_CHECK(vkWaitSemaphores(m_device, &wait, std::numeric_limits<uint64_t>::max()));
  }
  if(!m_screenshotPending.empty())
  {
    saveViewportImage(m_screenshotPending);
    m_screenshotPending.clear();
  }
  if(m_sampleCapture.pending.empty())
    return;
  saveViewportImage(m_sampleCapture.pending);
  m_sampleCapture.pending.clear();
}

// One rule for every way the viewport is saved (--output, --outputSamples, the screenshot
// parameter): a .hdr file gets the linear rendered image, so it can be measured; any
// other extension gets what the screen shows.
void GltfRenderer::saveViewportImage(const std::filesystem::path& filename)
{
  const Resources::ImageType imageType =
      nvutils::extensionMatches(filename, ".hdr") ? Resources::eImgRendered : Resources::eImgTonemapped;
  m_app->saveImageToFile(m_resources.gBuffers.getColorImage(imageType), m_resources.gBuffers.getSize(), filename);
}

void GltfRenderer::saveHeadlessOutputImage()
{
  std::string                 outputPath = headlessOutputPathOrDefault().string();
  const std::filesystem::path parentDir  = std::filesystem::path(outputPath).parent_path();
  if(!parentDir.empty() && !std::filesystem::exists(parentDir))
  {
    LOGW("GltfRenderer::saveHeadlessOutputImage(): output directory does not exist for path %s\n", outputPath.c_str());
  }
  if(!m_app)
  {
    LOGE("GltfRenderer::saveHeadlessOutputImage(): application is not set; cannot save image to %s\n", outputPath.c_str());
    return;
  }
  saveViewportImage(outputPath);
}

void GltfRenderer::onUIRender()
{
  // Run any post-restore hooks now that Application::run() has reloaded the ini (which happens
  // between onAttach and the first frame, in both windowed and headless paths). ImGui writes
  // restored values straight into storage without firing the parameter's callbackSuccess, so
  // derived state (e.g. Resources::sunDirection from the restored sunAzimuth/Elevation) would
  // otherwise stay stale until the user touched a UI slider.
  if(m_pendingRestoreCallbacks)
  {
    m_pendingRestoreCallbacks = false;
    m_settings.runPostRestoreHooks();
  }

  applyPendingSkyPresets();          // frame-top: start-up presets
  applyPendingEnvironmentRefresh();  // frame-top: re-bake / rebind the environment after a sky change (see method)

  // The viewport's per-frame camera step: camera transitions and camera.frame need it.
  m_cameraManip->updateAnim();
}

//--------------------------------------------------------------------------------------------------
// Render the scene
// What a pixel of the frame holds in GPU memory, for refuseFrameSize. Measured at 4K on 2026-10-25:
// the rasterizer's G-buffers (RGBA8, RGBA32F, R32F, depth: 28 B), and under RTX those plus the DLSS
// guides and output (72 B), OptiX's three images and four CUDA-shared float4 buffers (112 B) and
// the hand-off's two targets (24 B), about 244 B of buffers the renderer owns, plus the upscaler's
// own history, which the driver holds. Estimates: a refusal keeps the process alive, nothing more.
constexpr uint32_t kRasterBytesPerPixel     = 48;
constexpr uint32_t kPathTracerBytesPerPixel = 300;

GltfRenderer::MemoryBudget GltfRenderer::deviceMemoryBudget() const
{
  MemoryBudget b;
  if(!m_memoryBudgetExtension || !m_app)
    return b;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &budget};
  vkGetPhysicalDeviceMemoryProperties2(m_app->getPhysicalDevice(), &props);
  for(uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; i++)
    if(props.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
    {
      b.budget += budget.heapBudget[i];
      b.usage += budget.heapUsage[i];
    }
  b.known = true;
  return b;
}

std::string GltfRenderer::frameProfileJson() const
{
  std::string out = "{\"sections\":[";
  if(m_profilerTimeline)
  {
    nvutils::ProfilerTimeline::Snapshot snapshot;
    m_profilerTimeline->getFrameSnapshot(snapshot);
    bool first = true;
    for(size_t i = 0; i < snapshot.timerInfos.size(); i++)
    {
      const auto& info = snapshot.timerInfos[i];
      if(info.numAveraged == 0)
        continue;
      std::string name;
      for(char c : snapshot.timerNames[i])
        if(c == '"' || c == '\\')
          name += {'\\', c};
        else if(uint8_t(c) >= 0x20)
          name += c;
      char buf[96];
      // the profiler keeps microseconds
      std::snprintf(buf, sizeof(buf), "\",\"level\":%u,\"cpu\":%.3f,\"gpu\":%.3f}", info.level, info.cpu.average * 1e-3,
                    info.gpu.average * 1e-3);
      out += (first ? "{\"name\":\"" : ",{\"name\":\"") + name + buf;
      first = false;
    }
  }
  out += "]";
  const MemoryBudget b = deviceMemoryBudget();
  if(b.known)
  {
    char buf[96];
    std::snprintf(buf, sizeof(buf), ",\"gpuMemoryMB\":%.0f,\"gpuBudgetMB\":%.0f,\"gpuAllocatedMB\":%.0f", double(b.usage) / (1 << 20),
                  double(b.budget) / (1 << 20), double(allocatedMemory().allocations) / (1 << 20));
    out += buf;
  }
  return out + "}";
}

double GltfRenderer::frameGpuMs() const
{
  nvutils::ProfilerTimeline::TimerInfo info;
  std::string                          api;
  // the profiler keeps microseconds
  return m_profilerTimeline && m_profilerTimeline->getFrameTimerInfo("Frame", info, api) ? info.gpu.last * 1e-3 : 0.0;
}

std::string GltfRenderer::refuseFrameSize(VkExtent2D size, double extraBytesPerPixel) const
{
  const MemoryBudget b = deviceMemoryBudget();
  if(!b.known)
    return {};
  const uint64_t packed = m_frameSizePacked.load();
  const int64_t  pixelsNow  = int64_t(packed >> 32) * int64_t(packed & 0xffffffffu);
  const int64_t  pixelsMore = int64_t(size.width) * size.height - pixelsNow;
  if(pixelsMore <= 0)
    return {};
  const double need = double(pixelsMore) * (frameBytesPerPixel() + extraBytesPerPixel);
  const double free = double(b.budget) - double(b.usage);
  // a tenth of what is free stays for the scene's own growth (a BLAS refit, a texture)
  if(need <= free * 0.9)
    return {};
  return fmt::format("a {}x{} frame needs about {:.1f} GB more of GPU memory and {:.1f} GB is free ({:.1f} of {:.1f} GB in use)",
                     size.width, size.height, need / 1e9, std::max(free, 0.0) / 1e9, b.usage / 1e9, b.budget / 1e9);
}

// What the allocator holds on the device and has handed out: the driver's usage counts whole blocks,
// so what was freed inside one still reads as used, though a new image can go there
GltfRenderer::AllocatedMemory GltfRenderer::allocatedMemory() const
{
  AllocatedMemory out;
  if(!m_app)
    return out;
  VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
  vmaGetHeapBudgets(m_resources.allocator, budgets);
  VkPhysicalDeviceMemoryProperties props{};
  vkGetPhysicalDeviceMemoryProperties(m_app->getPhysicalDevice(), &props);
  for(uint32_t i = 0; i < props.memoryHeapCount; i++)
    if(props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
    {
      out.blocks += budgets[i].statistics.blockBytes;
      out.allocations += budgets[i].statistics.allocationBytes;
    }
  return out;
}

// A tenth of the budget stays free for what grows unasked (a BLAS refit, a denoiser's targets), and so
// does what the path tracer's frame would add to the rasterizer's, for a switch of mode to find.
std::optional<int64_t> GltfRenderer::sceneHeadroom() const
{
  const MemoryBudget b = deviceMemoryBudget();
  if(!b.known)
    return std::nullopt;
  const uint64_t packed = m_frameSizePacked.load();
  const uint64_t pixels = (packed >> 32) * (packed & 0xffffffffu);
  const uint32_t perPixel = frameBytesPerPixel();
  const uint64_t frameMore = perPixel < kPathTracerBytesPerPixel ? pixels * (kPathTracerBytesPerPixel - perPixel) : 0;
  uint64_t       budget = b.budget;
  if(const uint32_t limit = m_resources.settings.sceneBudgetMB; limit > 0)
    budget = std::min<uint64_t>(budget, uint64_t(limit) << 20);
  const AllocatedMemory allocated = allocatedMemory();
  const uint64_t        used      = b.usage - std::min<uint64_t>(b.usage, allocated.blocks - allocated.allocations);
  return int64_t(budget) - int64_t(budget / 10 + frameMore + used);
}

// A load or wear that can't fit even with every texture lowered is refused here, before it is in the
// scene: past the budget it would crawl, in system memory, or end the process if none is left.
std::string GltfRenderer::refuseGeometry(uint64_t bytes)
{
  const std::optional<int64_t> headroom = sceneHeadroom();
  if(!headroom)
    return {};
  const int64_t lowerable = m_resources.retainedGpu ? int64_t(m_resources.retainedGpu->lowerableImageBytes()) : 0;
  const int64_t room      = *headroom - int64_t(m_pendingGeometry) + lowerable;
  if(int64_t(bytes) <= room)
  {
    m_pendingGeometry += bytes;
    return {};
  }
  const MemoryBudget b = deviceMemoryBudget();
  const auto size = [](double n) { return n < 1e9 ? fmt::format("{:.0f} MB", n / 1e6) : fmt::format("{:.1f} GB", n / 1e9); };
  return fmt::format("it needs about {} of GPU memory and {} can be freed ({:.1f} of {:.1f} GB in use). "
                     "Remove something from the scene or lower Subdivision.",
                     size(double(bytes)), size(double(std::max<int64_t>(room, 0))), double(b.usage) / 1e9, double(b.budget) / 1e9);
}

void GltfRenderer::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  m_profilerTimeline->frameAdvance();
  // for refuseFrameSize on the stream's thread
  m_frameBytesPerPixel.store(m_resources.settings.renderSystem == RenderingMode::ePathtracer ? kPathTracerBytesPerPixel : kRasterBytesPerPixel);
  {
    const VkExtent2D s = renderSize();
    m_frameSizePacked.store(uint64_t(s.width) << 32 | s.height);
  }
  // Before anything reads the sun: if the scene marks a light as the sky's sun, that light owns
  // its direction.
  m_skySun.syncFromMarkedLight();
  // Don't do anything if the busy window is open
  if(m_busy.isBusy())
  {
    return;
  }

  // Consume the done signal from the busy state.
  if(m_busy.isDone())
  {
    m_busy.consumeDone();
    resetFrame();
  }

  // Loading pipeline: submit queued work, poll completion, run callbacks
  if(isAutomatedRun())
  {
    // The GPU work a load or rebuild enqueued (uploads, BLAS and TLAS builds), which the compose
    // that enqueued it never waits for: timed here, as its own depth-0 scope.
    std::optional<fv::timing::Scope> wait;
    if(fv::timing::enabled() && !m_loadPipeline.idle())
      wait.emplace("drain load pipeline");
    m_loadPipeline.drain();
  }
  else if(m_loadPipeline.poll())
    return;  // Still loading

  // Preview re-bake for a control that is being dragged. Unlike the commit path this records into
  // the frame's own command buffer and waits on nothing, so it belongs here rather than in
  // onUIRender: it is per-frame work, not a frame-boundary rebuild.
  if(m_envPreviewPending)
  {
    m_envPreviewPending = false;
    if(isBakedEnvironment(m_resources.settings.envSystem))
    {
      // Only the rasterizer reads the prefiltered cubemaps, so the path tracer skips the prefilter
      // and a drag costs it little more than the bake itself.
      const bool refreshDome = m_resources.settings.renderSystem == RenderingMode::eRasterizer;

      // Nothing waits here, so order explicitly: the LUTs, the lat-long image and the sun-disk
      // buffer are still being read by frames in flight, and are read again by this frame's
      // ray-tracing and graphics passes after the rewrite.
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

      // Most of the atmosphere controls -- Rayleigh, Mie, ozone, the planet radii -- change
      // nothing except the LUTs, so without a preview rebuild they look like dead sliders until
      // the mouse comes up. Preview quality is what makes that affordable: a sixteenth of the
      // directions and two bounces instead of four (see SkyBruneton::Quality), and recorded into this
      // frame rather than submitted and waited on. The commit that follows restores full quality.
      if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky)
      {
        const shaderio::SkyAtmosphereParameters wanted = buildAtmosphereParameters();
        if(!m_skyBruneton.lutsSatisfy(wanted, SkyBruneton::kQualityPreview))
        {
          m_skyBruneton.setParameters(wanted);
          m_skyBruneton.recordPrecompute(cmd, SkyBruneton::kQualityPreview);
        }
      }

      // The baker holds the parameters from the last commit; refresh them or the preview re-bakes
      // the image it already has. Invisible for the authored skies -- their background is drawn
      // per-ray from settings, so only the lighting lagged -- but the Bruneton sky *is* its baked
      // image, so without this a drag showed nothing until the mouse came up.
      updateBakerParameters();
      m_envBaker.preview(cmd, m_resources, refreshDome);
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      resetFrame();
    }
  }
  else if(m_envCommitOwed)
  {
    // First frame since the edits stopped: pay for the full rebuild the previews skipped.
    m_envCommitOwed = false;
    onEnvironmentChanged();
  }

  // Recycle staging from completed frames, then tag new uploads with this frame's signal.
  m_resources.staging.releaseCompletedAllocations();
  m_resources.staging.updateFrameSemaphoreState(nvvk::SemaphoreState::makeFixed(m_app->getFrameSignalSemaphore()));

  // Empty scene, clear the G-Buffer
  if(!m_resources.gpuView().hasContent())
  {
    clearGbuffer(cmd);
    return;
  }

  // Keep runtime denoiser guide usage synchronized with the optimal-shader feature set.
  // PathTracer::ensureShadersAndPipelines() watches currentFeatureSet and recompiles when this flips.
  m_resources.currentFeatureSet.set(nvvkgltf::SceneFeatureSet::eDlssGuide, dlssGuideRequired());


  // Start the profiler section for the GPU timer
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  updateClickPickState();   // Poll the previous click's async ray-pick, apply it once ready (see m_pendingClickPick)
  // Check for changes
  bool changed{false};
  changed |= m_pathTracer.keepOneAccumulationDenoiser(m_resources);  // OptiX or OIDN, not both
  changed |= syncRetainedScene(cmd);  // the clip, and the residency's sync of the host's edits

  // #DLSS motion: the retained scene keeps both tables current every sync (a row at rest is no motion)
  const bool dlss = m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_pathTracer.isDlssEnabled();
  m_resources.dlssInstanceMotionActive = dlss && m_resources.gpuView().prevObjectToWorldTable() != 0;
  m_resources.dlssDeformMotionActive   = dlss && m_resources.gpuView().prevPositionTable() != 0;

  if(changed)
  {
    resetFrame();
  }
  refreshSampleTargets();                    // --outputSamples: the stops this frame may end on
  bool frameChanged = updateFrameCounter();  // Check if the frame counter has changed
  // A shader variant built off the frame goes in only when a frame draws: one draws for it even with
  // the still converged, or the edits it holds back would wait for an unrelated change
  const bool variantWaiting = m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_pathTracer.variantBuilt();

  if(changed || frameChanged || variantWaiting)
  {
    if(m_resources.frameCount == 0)
    {
      m_cpuTimer.reset();
      m_cpuTimePrinted = false;  // Reset print flag when rendering starts
    }

    // Update the scene frame information uniform buffer
    const glm::mat4          viewProj = m_cameraManip->getPerspectiveMatrix() * m_cameraManip->getViewMatrix();
    const VkExtent2D         gbufSize = m_resources.gBuffers.getSize();
    const glm::quat          envRot   = m_resources.settings.envRotationQuat();  // normalised
    shaderio::SceneFrameInfo finfo{
        .viewMatrix     = m_cameraManip->getViewMatrix(),
        .projInv        = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
        .viewInv        = glm::inverse(m_cameraManip->getViewMatrix()),
        .viewProjMatrix = viewProj,
        .prevMVP        = m_prevMVP,
        .jitter         = {0.0f, 0.0f},
        .imageSize      = {float(gbufSize.width), float(gbufSize.height)},
        .flags =
            ((m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? shaderio::eSceneIsOrthographic : 0)
            | (m_resources.settings.useSolidBackground ? shaderio::eSceneUseSolidBackground : 0)
            // A baked analytic sky IS an environment image by the time the shaders see it:
            // same lat-long texture, same alias table, same prefiltered cubes. Flagging it as
            // such is what keeps sampling, MIS and raster IBL free of per-sky-type branches.
            | ((m_resources.settings.envSystem == shaderio::EnvSystem::eHdr || isBakedEnvironment(m_resources.settings.envSystem)) ?
                   shaderio::eSceneUseHdrEnvironment :
                   0)
            | ((m_resources.settings.envSystem == shaderio::EnvSystem::eNone) ? shaderio::eSceneUseNoEnvironment : 0)
            | (m_resources.settings.useInfinitePlane ? shaderio::eSceneUseInfinitePlane : 0)
            | ((m_resources.settings.useInfinitePlane && m_resources.settings.isShadowCatcher) ? shaderio::eSceneInfinitePlaneShadowCatcher : 0),
        .envRotation               = {envRot.x, envRot.y, envRot.z, envRot.w},
        .envBlur                   = m_resources.settings.hdrBlur,
        .envIntensity              = m_resources.settings.hdrEnvIntensity,
        .backgroundColor           = m_resources.settings.solidBackgroundColor,
        .visualization             = m_resources.settings.visualization,
        .infinitePlaneDistance     = m_resources.settings.infinitePlaneDistance,
        .infinitePlaneBaseColor    = m_resources.settings.infinitePlaneBaseColor,
        .infinitePlaneMetallic     = m_resources.settings.infinitePlaneMetallic,
        .infinitePlaneRoughness    = m_resources.settings.infinitePlaneRoughness,
        .shadowCatcherDarkenAmount = std::max(m_resources.settings.shadowCatcherDarkness, 0.0f),
        // Only the physical sky has a disk to draw; a null pointer is how every other environment
        // tells the background passes there is nothing to add. Declaration order matters here --
        // these sit before envSkyType in SceneFrameInfo.
        .sunDisk      = (m_resources.settings.envSystem == shaderio::EnvSystem::eSky) ?
                            reinterpret_cast<shaderio::SkySunDisk*>(m_envBaker.sunDiskAddress()) :
                            nullptr,
        .sunDirection = m_resources.sunDirection,
        // The sky's own sun, supplied only while the scene has no marked one. Selecting an
        // environment must not edit the user's scene, so nothing is added to the graph until a
        // save asks for it; from then on the marked light is the sun and this stays off.
        .rendererSunLight = (skyHasSun(m_resources.settings.envSystem) && m_skySun.lightIndex() < 0) ? 1 : 0,
        .sunLightIndex    = skyHasSun(m_resources.settings.envSystem) ? m_skySun.lightIndex() : -1,
        .envSkyType       = authoredSkyType(m_resources.settings.envSystem),
        .skyOmi           = buildSkyOmiParameters(),
        // Aerial perspective. Physical sky only (the sunDisk gate above says which), and the
        // altitude is the same fixed observer the bake uses -- metres in the settings because that
        // is what a person types, kilometres here because that is what the atmosphere works in.
        .aerialPerspectiveScale = m_resources.settings.atmoAerialPerspectiveScale,
        .observerAltitudeKm     = aerialPerspectiveEyeAltitudeKm(),
        // The gradient sky's sun. Zero for every other environment: the physical sky's sun comes
        // from `sunDisk` above, and no other sky has one.
        .skySunIlluminance = (m_resources.settings.envSystem == shaderio::EnvSystem::eGradient) ? gradientSunIlluminance() :
                                                                                                  glm::vec3(0.0F),
        .envSamplingGridWidth  = m_resources.hdrIbl.getSamplingGrid().width,
        .envSamplingGridHeight = m_resources.hdrIbl.getSamplingGrid().height,
        .skySunScale           = m_resources.settings.skySunScale,
        .envPickEstimate       = 0.25F * m_resources.hdrIbl.getIntegral(),
    };
    fillShadowAndGround(finfo);
    // Update the camera information
    m_prevMVP = finfo.viewProjMatrix;

    vkCmdUpdateBuffer(cmd, m_resources.bFrameInfo.buffer, 0, sizeof(shaderio::SceneFrameInfo), &finfo);
    // Update the sky
    m_resources.sunYIsUp = m_cameraManip->getUp().y > 0.5f;
    // Make sure buffer is ready to be used
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Switch between renderers based on the current mode
    ++m_resources.drawnFrameCount;
    switch(m_resources.settings.renderSystem)
    {
      case RenderingMode::ePathtracer:
        m_pathTracer.onRender(cmd, m_resources);
        break;
      case RenderingMode::eRasterizer:
        m_rasterizer.onRender(cmd, m_resources);
        break;
    }
  }
  else
  {
    // Print CPU time only once after render completes
    if(!m_cpuTimePrinted)
    {
      LOGI("Rendering finished: %f ms\n", m_cpuTimer.getMilliseconds());
      m_cpuTimePrinted = true;
    }
    // Auto denoise the final image of the accumulation
    if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_resources.accumulationComplete())
    {
      m_pathTracer.onAccumulationComplete(cmd, m_resources);
      // A headless run given --ptMaxSamples is done: this frame shows the final image (and its
      // denoise, which OptiX makes on this first idle frame), and --frames is only an upper bound.
      if(isHeadlessMode() && m_parameterParser && m_parameterParser->wasParsed("ptMaxSamples") && m_app)
        m_app->close();
    }
  }

  // Apply the post-processing effects
  const bool nrAllowed = tonemap(cmd);
#if defined(USE_DLSS) && defined(USE_DLSSNR)
  // DLSS-NR: enhance eImgTonemapped (LDR) after tonemapping; NR expects display-range input.
  // Skip on guide buffers, visualization debug modes, and bypassed-tonemapper paths.
  if(nrAllowed)
    if(Dlss* dlss = activeDlss())
      if(dlss->isNrActive())
      {
        auto nrSection = m_profilerGpuTimer.cmdFrameSection(cmd, "DLSS-NR");
        dlss->evaluateNr(cmd);
      }
#endif
  // The overlays go into the tonemapped image; a frame a screenshot takes is drawn without them
  // (the stream shows that one frame bare, the next has them again)
  const bool overlays = m_screenshotPending.empty();
  if(overlays)
    silhouette(cmd);
  updateSampleCaptures();  // --outputSamples: the image is final for this frame
  // Async click ray-pick: record the pick dispatch on this frame's own command buffer - the TLAS is
  // already proven valid here (the path tracer/rasterizer just used it above) - then hand off a
  // frame-semaphore to poll instead of blocking. See m_pendingClickPick's comment (renderer.hpp).
  if(m_pendingClickPick)
  {
    m_rayPicker.run(cmd, {.modelViewInv   = m_pendingClickPick->modelViewInv,
                          .perspectiveInv = m_pendingClickPick->perspectiveInv,
                          .isOrthographic = m_pendingClickPick->isOrthographic,
                          .pickPos        = m_pendingClickPick->pickPos,
                          .tlas           = m_resources.gpuView().tlas()});
    m_pendingClickResult =
        PendingClickResult{.semaphoreState = nvvk::SemaphoreState::makeFixed(m_app->getFrameSignalSemaphore()),
                           .isDoubleClick  = m_pendingClickPick->isDoubleClick};
    m_pendingClickPick.reset();
  }
  // The grid and transform gizmo are drawn into the frame: a headless --stream is the native
  // viewport, so the grid shows there.
  if(overlays)
  {
    renderVisualHelpers(cmd);
  }
}

//--------------------------------------------------------------------------------------------------
// Called with headless rendering, to save the final image
void GltfRenderer::onLastHeadlessFrame()
{
  logDenoiserTimings();
  if(m_sampleCapture.next < m_sampleCapture.targets.size())
    LOGW("--outputSamples: %d spp and above were not reached (the image has %d spp); raise --frames\n",
         m_sampleCapture.targets[m_sampleCapture.next], m_sampleCapture.lastSamples);
  saveHeadlessOutputImage();
}

#if defined(USE_DLSS)
//--------------------------------------------------------------------------------------------------
// Returns the Dlss instance owned by whichever renderer is currently active.
Dlss* GltfRenderer::activeDlss()
{
  return (m_resources.settings.renderSystem == RenderingMode::ePathtracer) ? m_pathTracer.getDlss() : m_rasterizer.getDlss();
}

const Dlss* GltfRenderer::activeDlss() const
{
  return (m_resources.settings.renderSystem == RenderingMode::ePathtracer) ? m_pathTracer.getDlss() : m_rasterizer.getDlss();
}
#endif

//--------------------------------------------------------------------------------------------------
// Runtime gate for path-tracer guide-buffer shader code.
bool GltfRenderer::dlssGuideRequired() const
{
  if(m_resources.settings.renderSystem != RenderingMode::ePathtracer)
    return false;

#if defined(USE_DLSS) || defined(USE_OPTIX_DENOISER) || defined(USE_OIDN_DENOISER)
  // The build with guides is the one in the exe, warmed at start, and it writes none unless a denoiser
  // asks (ePtDenoiserGuides). Without guides there's nothing prepared: turning the denoiser off froze
  // the renderer for 27 s the first time, to compile and build the pipeline of a shader that differs
  // only by code that doesn't run. The optimal shader is built per scene anyway, and leaves it out.
  if(!m_resources.settings.optimalShader)
    return true;
#endif

#if defined(USE_DLSS)
  if(m_pathTracer.isDlssEnabled())
    return true;
#endif

#if defined(USE_OPTIX_DENOISER)
  if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser(); optix && optix->isEnabled())
    return true;
#endif

#if defined(USE_OIDN_DENOISER)
  if(const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser(); oidn && oidn->isEnabled())
    return true;
#endif

  return false;
}

//--------------------------------------------------------------------------------------------------
// Push an authored sky into the live settings. See the declaration for what `source` decides.
//
void GltfRenderer::applySkyDescriptor(const SkyDescriptor& sky, SkySource source, const std::filesystem::path& uriBase)
{
  Settings& settings = m_resources.settings;

  // An explicit `--envSystem` is a statement about this run, and it outranks the file: the same
  // rule the .ini already follows, extended to the place it should have reached first. Without
  // this a script that forces an environment silently gets whatever the scene authored instead,
  // and the measurement it was taking is of something else.
  //
  // Only the *selection* is suppressed. The sky's parameters are still applied, so forcing the
  // mode a scene authored still picks up its colours, and `m_environmentSky` keeps the descriptor
  // either way so a later save preserves it.
  //
  // A preset is the exception, and obviously so: loading one by hand at runtime *is* the statement,
  // and a flag from start-up has no business overruling it.
  // OMI's `rotation` is the renderer's own, quaternion for quaternion, tilt included.
  //
  // Only applied when the file actually states one. The rotation is viewer state -- the user, the
  // command line or MCP may have set it -- and loadScene() leaves viewer state alone on purpose; a
  // scene that says nothing about orientation must not silently reset it. A scene that does state
  // one is making a statement about its own sky, and that wins, like every other authored sky value
  // here. The sun is left where it is: a file's sun is its marked light and a preset's is its
  // sunRotation, both already in world space.
  if(sky.rotationAuthored)
    settings.envRotation = {sky.rotation.x, sky.rotation.y, sky.rotation.z, sky.rotation.w};

  const bool forcedOnCommandLine = source == SkySource::eScene && m_parameterParser && m_parameterParser->wasParsed("envSystem");
  const auto selectEnvironment = [&](shaderio::EnvSystem env) {
    if(forcedOnCommandLine)
    {
      if(settings.envSystem != env)
      {
        LOGI("OMI_environment_sky: scene authors a different environment; --envSystem wins for this run.\n");
      }
      return;
    }
    settings.envSystem = env;
  };

  switch(sky.type)
  {
    case SkyDescriptor::Type::ePlain:
      settings.plainColor = sky.plain.color;
      selectEnvironment(shaderio::EnvSystem::ePlain);
      break;
    case SkyDescriptor::Type::eGradient:
      settings.gradientBottomColor  = sky.gradient.bottomColor;
      settings.gradientHorizonColor = sky.gradient.horizonColor;
      settings.gradientTopColor     = sky.gradient.topColor;
      settings.gradientBottomCurve  = sky.gradient.bottomCurve;
      settings.gradientTopCurve     = sky.gradient.topCurve;
      settings.gradientSunAngleMax  = sky.gradient.sunAngleMax;
      settings.gradientSunCurve     = sky.gradient.sunCurve;
      selectEnvironment(shaderio::EnvSystem::eGradient);
      break;
    case SkyDescriptor::Type::ePanorama: {
      // The image is an external asset, so the URI resolves against the file that named it -- the
      // glTF or the preset, the same rule every other external asset follows
      // (docs/external_assets.md), and NOT against the resources directories, which would silently
      // load someone else's environment when two scenes name the same file.
      if(sky.panorama.uri.empty())
      {
        LOGW("OMI_environment_sky: panorama sky has no NV_environment_sky_panorama.uri%s; keeping the current environment.\n",
             sky.panorama.equirectangular >= 0 ? " (its `equirectangular` texture index is not supported)" : "");
        break;
      }
      const std::filesystem::path resolved =
          uriBase.empty() ? std::filesystem::path(sky.panorama.uri) : uriBase / std::filesystem::path(sky.panorama.uri);
      if(!std::filesystem::exists(resolved))
      {
        LOGW("OMI_environment_sky: panorama image not found: %s\n", nvutils::utf8FromPath(resolved).c_str());
        break;
      }
      createHDR(resolved);
      selectEnvironment(shaderio::EnvSystem::eHdr);
      break;
    }
    case SkyDescriptor::Type::ePhysical:
      // The inverse of the split on save: colour times coefficient, m^-1 back to km^-1.
      settings.atmoRayleighScattering = sky.physical.rayleighColor * (sky.physical.rayleighCoefficient * 1000.0F);
      settings.atmoMieScattering      = sky.physical.mieColor * (sky.physical.mieCoefficient * 1000.0F);
      settings.atmoMieAnisotropy      = sky.physical.mieAnisotropy;
      settings.atmoGroundAlbedo       = sky.physical.groundColor;

      // Then the NV overlay, for everything OMI cannot say. Only when the file carried one: a
      // scene with just the OMI block has not said what its planet is, so leaving the current
      // values alone is more honest than forcing Earth on it.
      if(sky.atmosphere.present)
      {
        const SkyDescriptor::Atmosphere& a = sky.atmosphere;
        settings.atmoSolarIrradiance       = a.solarIrradiance;
        settings.atmoSunAngularRadius      = a.sunAngularRadius;
        settings.atmoMieAlbedo             = a.mieAlbedo;
        // Metres in the file, kilometres in the model. Divided rather than multiplied by 0.001:
        // the reciprocal is not representable in binary, so scaling by it loses a few ulps on
        // every load, and a scene saved and reopened repeatedly would creep. Dividing by an exact
        // power-of-ten float keeps radii like 3389.5 km landing back on themselves.
        settings.atmoOzoneExtinction     = a.ozoneExtinction * 1000.0F;
        settings.atmoRayleighScaleHeight = a.rayleighScaleHeight / 1000.0F;
        settings.atmoMieScaleHeight      = a.mieScaleHeight / 1000.0F;
        settings.atmoOzoneCenter         = a.ozoneCenter / 1000.0F;
        settings.atmoOzoneWidth          = a.ozoneWidth / 1000.0F;
        settings.atmoBottomRadius        = a.planetRadius / 1000.0F;
        settings.atmoThickness           = a.atmosphereThickness / 1000.0F;
      }
      selectEnvironment(shaderio::EnvSystem::eSky);
      break;
    default:
      assert(false && "unhandled SkyDescriptor::Type");
      break;
  }

  // No calibrateFireflyClamp() here: hdrIbl still describes the previous environment until the
  // refresh at the next frame top, which calibrates when the environment kind changed.
  onEnvironmentChanged();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Capture the current sky as a `.sky.json`, and apply one.
//
// The sun angle travels with it, because a sky and the angle of its sun are one look. On the way
// out it is read from the direction the renderer is actually using, whichever of the four things
// set it; on the way in it is applied through m_skySun.aim(), so a scene whose sun is a marked light
// records the move on its undo stack exactly as a slider drag would.
//
// Note what is *not* undoable: everything else a preset changes is viewer state -- the atmosphere,
// the ambient terms, the environment type -- and no environment control in this renderer is on the
// undo stack. Putting presets there alone would mean a settings-snapshot command unlike anything
// else in undo_redo, for a consistency the panel does not otherwise have.
//
bool GltfRenderer::saveSkyPreset(const std::filesystem::path& path) const
{
  const EnvironmentState sky = skyDescriptorFromSettings(path);
  if(!sky.has_value())
  {
    LOGW("Sky preset: the current environment is not a sky that can be written (%s).\n", nvutils::utf8FromPath(path).c_str());
    return false;
  }

  sky_preset::Preset preset{.sky = *sky};
  if(skyHasSun(m_resources.settings.envSystem))
  {
    // glTF aims a light down -Z, so the rotation that takes +Z to the sun is what a node would
    // store -- the same convention SkySun::setMarkedDirection writes.
    preset.sunRotation = glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), glm::normalize(m_resources.sunDirection));
  }
  return sky_preset::save(path, preset);
}

bool GltfRenderer::loadSkyPreset(const std::filesystem::path& path)
{
  const std::optional<sky_preset::Preset> preset = sky_preset::load(path);
  if(!preset.has_value())
    return false;

  // A preset loaded after a scene supersedes that scene's sky, even one still waiting for the frame
  // top -- otherwise the scene's would land a frame later and silently undo this.
  m_environmentSkyPending = false;
  applySkyDescriptor(preset->sky, SkySource::ePreset, path.parent_path());

  // Absent means "leave the sun alone", which is why this is not an else.
  if(preset->sunRotation.has_value())
  {
    m_skySun.aim(*preset->sunRotation * glm::vec3(0.0F, 0.0F, 1.0F));
    m_skySun.syncAngles();
  }

  // The preset is now what a save would write, so a scene saved after loading one carries it.
  m_environmentSky = preset->sky;
  return true;
}

void GltfRenderer::queueStartupSkyPresets(const std::filesystem::path& load, const std::filesystem::path& save)
{
  m_pendingLoadSkyPreset = load;
  m_pendingSaveSkyPreset = save;
}

void GltfRenderer::applyPendingSkyPresets()
{
  if(!m_pendingLoadSkyPreset.empty())
    loadSkyPreset(std::exchange(m_pendingLoadSkyPreset, {}));
  if(!m_pendingSaveSkyPreset.empty())
    saveSkyPreset(std::exchange(m_pendingSaveSkyPreset, {}));
}

//--------------------------------------------------------------------------------------------------
// The URI to record for an environment image, relative to the glTF being written when that is
// possible and absolute when it is not.
//
// Relative is what makes a scene and its environment movable as a pair, which is the common case.
// It stops being the right answer once the image lives outside the glTF's tree -- on another
// drive, or in a shared asset library -- where a relative path would be a long chain of "..\"
// that breaks the moment either end moves. Absolute is honest there.
//
std::string GltfRenderer::relativeUriForSave(const std::filesystem::path& asset, const std::filesystem::path& destination) const
{
  const std::filesystem::path base = destination.parent_path();
  if(base.empty())
    return nvutils::utf8FromPath(asset);

  std::error_code             ec;
  const std::filesystem::path rel = std::filesystem::relative(asset, base, ec);
  if(ec || rel.empty() || *rel.begin() == "..")
    return nvutils::utf8FromPath(asset);

  // glTF URIs use forward slashes regardless of platform.
  std::string uri = nvutils::utf8FromPath(rel);
  std::replace(uri.begin(), uri.end(), '\\', '/');
  return uri;
}

//--------------------------------------------------------------------------------------------------
// Write the current sky into the model being serialized. Runs from Scene::save, on the copy that
// is actually written to disk.
//
// With the toggle off we strip rather than skip: a scene loaded with a sky and saved with the
// toggle off should not keep an extension that no longer matches what the user is looking at.
//
// Reconcile the scene's marked sun light with the sky being written.
//
// OMI_environment_sky describes a medium and says to add suns with KHR_lights_punctual, so a sky
// with a sun is only half-written without one: another renderer would build the atmosphere, find
// nothing to light it, and show an unlit sky. That failure is invisible here, because this
// renderer supplies its own sun when the scene has none -- which is exactly why it has to be
// fixed at the point of writing.
//
// The light is marked (gltf_environment_sky.hpp) so a reload knows which light is the sky's
// rather than guessing, and so this function knows which light is its own to manage. Any other
// light in the file is untouched.
//
// Intensity is the solar illuminance *above* the atmosphere, not the attenuated value this
// renderer shows. The atmosphere travels in the same file; a reader that applies it to an already
// attenuated sun would attenuate twice.
//
void GltfRenderer::writeSkySunLight(tinygltf::Model& outModel) const
{
  const int  existingNode = gltf_environment_sky::findSkySunNode(outModel);
  const bool wantSun      = m_resources.settings.envSaveToGltf && skyHasSun(m_resources.settings.envSystem);

  if(!wantSun)
  {
    // The sky being written has no sun, so nothing should still claim to be one.
    if(existingNode >= 0)
    {
      tinygltf::Node& node = outModel.nodes[existingNode];
      const bool      ours = gltf_environment_sky::isRendererOwnedSun(node);

      tinygltf::Value::Object extras = node.extras.Get<tinygltf::Value::Object>();
      extras.erase(gltf_environment_sky::kSkySunMarkerKey);
      node.extras = tinygltf::Value(std::move(extras));

      if(ours)
      {
        // A sun this renderer invented for an earlier save. Withdrawing it is what keeps the file
        // describing what is on screen.
        //
        // The node is emptied rather than erased: deleting it would renumber every node index in
        // the file. The light entry is left too, for the same reason -- compaction collects both
        // if the user asks for it.
        LOGI("OMI_environment_sky: withdrawing the sun light this renderer added; the sky being saved has none.\n");
        node.light = -1;
        node.extensions.erase("KHR_lights_punctual");
      }
      else
      {
        // Someone else's light that was chosen as the sun. It stops being the sun and stays
        // exactly as it was -- unpicking it is not the same as deleting it.
        LOGI("OMI_environment_sky: '%s' is no longer the sky's sun; the light itself is unchanged.\n", node.name.c_str());
      }
    }
    return;
  }

  // Colour and illuminance of the sun this sky implies, in the units the light carries.
  const shaderio::SkyAtmosphereParameters atmosphere = buildAtmosphereParameters();
  glm::vec3                               illuminance(0.0F);
  float                                   angularRadius = atmosphere.sunAngularRadius;
  if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky)
  {
    // Above the atmosphere on purpose -- see the note below on double attenuation. Shared with the
    // firefly clamp, which wants the same number for its own reasons.
    illuminance = skySunIlluminanceAboveAtmosphere();
  }
  else
  {
    // The gradient sky's sun is drawn rather than measured, so there is no disk radiance to
    // convert; the illuminance is scaled against the dome instead. Same number the renderer lights
    // with, so the file describes what is on screen.
    angularRadius = kGradientSunAngularRadius;
    illuminance   = gradientSunIlluminance() * m_resources.settings.hdrEnvIntensity;
  }

  const float     peak  = std::max({illuminance.x, illuminance.y, illuminance.z});
  const glm::vec3 color = (peak > 0.0F) ? illuminance / peak : glm::vec3(1.0F);

  // Rotation taking +Z to the sun: glTF aims a light down -Z, and the renderer's sunDirection
  // points at it. A world rotation; turned into the node's local one below.
  const glm::vec3 toSun = glm::normalize(m_resources.sunDirection);
  const glm::quat rot   = glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), toSun);

  int nodeIdx = existingNode;
  if(nodeIdx < 0)
  {
    tinygltf::Node node;
    node.name = "Sun";
    gltf_environment_sky::setSkySunMarker(node, gltf_environment_sky::kSkySunOwnerRenderer);
    outModel.nodes.push_back(node);
    nodeIdx = static_cast<int>(outModel.nodes.size()) - 1;

    const int sceneIndex = outModel.defaultScene >= 0 ? outModel.defaultScene : 0;
    if(sceneIndex < static_cast<int>(outModel.scenes.size()))
      outModel.scenes[sceneIndex].nodes.push_back(nodeIdx);
  }

  // A sun the user picked may sit under a rotated parent; the one created above is a root.
  const glm::quat local = gltf_environment_sky::localRotationForWorld(outModel, nodeIdx, rot);
  tinygltf::Node& node  = outModel.nodes[nodeIdx];
  if(!node.matrix.empty())
  {
    // A matrix would win over the TRS set below, so keep its translation and scale as TRS.
    glm::vec3 t, s;
    glm::quat r;
    tinygltf::utils::getNodeTRS(node, t, r, s);
    node.translation = {t.x, t.y, t.z};
    node.scale       = {s.x, s.y, s.z};
    node.matrix.clear();
  }
  node.rotation = {local.x, local.y, local.z, local.w};

  // Reuse the light this node already points at, so a re-save does not add one per save.
  //
  // `node.light` rather than the extensions map: that is the canonical reference in this codebase
  // -- tinygltf mirrors the extension into it on load and serialises from it on save, which is why
  // SceneEditor::addLightNode writes only this field.
  int        lightIdx    = node.light;
  const bool createdHere = (lightIdx < 0 || lightIdx >= static_cast<int>(outModel.lights.size()));
  if(createdHere)
  {
    outModel.lights.push_back({});
    lightIdx = static_cast<int>(outModel.lights.size()) - 1;
  }
  node.light = lightIdx;

  tinygltf::Light& light = outModel.lights[lightIdx];
  // Name only what we create. Adopting someone's "Directional Light" and calling it "Sun" is
  // their scene being edited by a viewer, which is the thing this whole design avoids.
  if(createdHere)
    light.name = "Sun";
  light.type      = "directional";
  light.color     = {color.x, color.y, color.z};
  light.intensity = peak;

  tinygltf::utils::syncExtensionsUsed(outModel);
}

void GltfRenderer::writeEnvironmentSky(tinygltf::Model& outModel, const std::filesystem::path& destination) const
{
  if(!m_resources.settings.envSaveToGltf)
  {
    gltf_environment_sky::strip(outModel);
    writeSkySunLight(outModel);  // drops the sun too: the environment is not part of this file
    return;
  }

  const EnvironmentState state = skyDescriptorFromSettings(destination);
  if(!state.has_value())
  {
    gltf_environment_sky::strip(outModel);
    writeSkySunLight(outModel);  // no sky, so no sun either: withdraw the marker it would leave behind
    return;
  }

  const int sceneIndex = outModel.defaultScene >= 0 ? outModel.defaultScene : 0;
  gltf_environment_sky::write(outModel, sceneIndex, *state);
  writeSkySunLight(outModel);
}

//--------------------------------------------------------------------------------------------------
// The live settings as an authored sky. See the declaration for what nullopt means.
//
EnvironmentState GltfRenderer::skyDescriptorFromSettings(const std::filesystem::path& destination) const
{
  const Settings& settings = m_resources.settings;

  // Start from the descriptor the scene was loaded with, so unmodelled keys and any sky type this
  // build does not render survive the round trip. A scene that authored nothing starts fresh.
  SkyDescriptor sky = m_environmentSky.value_or(SkyDescriptor{});

  // The environment's orientation is the sky's, the same quaternion, so it is written exactly.
  sky.rotation         = settings.envRotationQuat();
  sky.rotationAuthored = true;

  switch(settings.envSystem)
  {
    case shaderio::EnvSystem::ePlain:
      // A sky type this build does not recognize loads as this plain fallback (see
      // SkyDescriptor::unknownType). While the fallback is still on screen untouched, the file's
      // own sky is the truer thing to save; every branch that authors a sky of its own drops it.
      if(settings.plainColor != sky.plain.color)
        sky.unknownType.clear();
      sky.type        = SkyDescriptor::Type::ePlain;
      sky.plain.color = settings.plainColor;
      break;
    case shaderio::EnvSystem::eGradient:
      sky.unknownType.clear();
      sky.type                  = SkyDescriptor::Type::eGradient;
      sky.gradient.bottomColor  = settings.gradientBottomColor;
      sky.gradient.horizonColor = settings.gradientHorizonColor;
      sky.gradient.topColor     = settings.gradientTopColor;
      sky.gradient.bottomCurve  = settings.gradientBottomCurve;
      sky.gradient.topCurve     = settings.gradientTopCurve;
      sky.gradient.sunAngleMax  = settings.gradientSunAngleMax;
      sky.gradient.sunCurve     = settings.gradientSunCurve;
      break;
    case shaderio::EnvSystem::eSky: {
      // OMI describes the atmosphere as a magnitude plus a colour, and in m^-1; the model keeps a
      // per-channel coefficient in km^-1. Splitting on the largest channel is what makes the pair
      // reproduce the extension's own defaults in shape -- its rayleighColor is (0.3, 0.5, 1.0),
      // blue-dominant and normalised to 1 -- and it is exactly invertible, which is the property
      // the round trip depends on.
      const auto split = [](const glm::vec3& perChannelPerKm, float& outCoefficient, glm::vec3& outColor) {
        const float peak = std::max({perChannelPerKm.x, perChannelPerKm.y, perChannelPerKm.z});
        // Divided, not scaled by 0.001: the reciprocal is not representable in binary, so scaling
        // by it costs a few ulps and the value no longer lands back on itself when the file is
        // reopened and saved again.
        outCoefficient = peak / 1000.0F;  // km^-1 -> m^-1, as the extension specifies
        outColor       = (peak > 0.0F) ? perChannelPerKm / peak : glm::vec3(1.0F);
      };

      sky.unknownType.clear();
      sky.type = SkyDescriptor::Type::ePhysical;
      split(settings.atmoRayleighScattering, sky.physical.rayleighCoefficient, sky.physical.rayleighColor);
      split(settings.atmoMieScattering, sky.physical.mieCoefficient, sky.physical.mieColor);
      sky.physical.mieAnisotropy = settings.atmoMieAnisotropy;
      sky.physical.groundColor   = settings.atmoGroundAlbedo;

      // Everything OMI has no field for, in the sibling NV block. Kilometres in the model, metres
      // in the file.
      sky.atmosphere.present             = true;
      sky.atmosphere.solarIrradiance     = settings.atmoSolarIrradiance;
      sky.atmosphere.sunAngularRadius    = settings.atmoSunAngularRadius;
      sky.atmosphere.mieAlbedo           = settings.atmoMieAlbedo;
      sky.atmosphere.ozoneExtinction     = settings.atmoOzoneExtinction / 1000.0F;
      sky.atmosphere.rayleighScaleHeight = settings.atmoRayleighScaleHeight * 1000.0F;
      sky.atmosphere.mieScaleHeight      = settings.atmoMieScaleHeight * 1000.0F;
      sky.atmosphere.ozoneCenter         = settings.atmoOzoneCenter * 1000.0F;
      sky.atmosphere.ozoneWidth          = settings.atmoOzoneWidth * 1000.0F;
      sky.atmosphere.planetRadius        = settings.atmoBottomRadius * 1000.0F;
      sky.atmosphere.atmosphereThickness = settings.atmoThickness * 1000.0F;
      break;
    }
    case shaderio::EnvSystem::eHdr: {
      // eHdr *is* OMI's `panorama`; the enumerator kept its name and value because both the .ini
      // and the glTF store it as an integer.
      if(m_lastHdrFile.empty())
      {
        // Nothing loaded (the start-up dummy). Writing `panorama` with no source would describe an
        // environment that does not exist, so fall through to the preserve-or-strip rule below.
        if(!m_environmentSky.has_value())
          return std::nullopt;
        break;
      }
      sky.unknownType.clear();
      sky.type = SkyDescriptor::Type::ePanorama;
      // Relative to the glTF being written, so the pair moves together. An image outside that
      // tree keeps an absolute path rather than a chain of "..\" that only works from here.
      sky.panorama.uri = relativeUriForSave(m_lastHdrFile, destination);
      break;
    }
    default:
      // The active mode is not an authored OMI type (None). If the scene came with a sky,
      // re-emit it unchanged rather than inventing one or silently discarding the author's data;
      // if it did not, there is nothing to write.
      if(!m_environmentSky.has_value())
        return std::nullopt;
      break;
  }

  return sky;
}

//--------------------------------------------------------------------------------------------------
// Frame-top hook for a pending environment change, paired with onEnvironmentChanged().
//
// Runs from onUIRender rather than onRender:
// rebuilding the environment waits on the device and rewrites descriptors that in-flight frames
// are reading, which is not safe in the middle of recording a frame's command buffer. onUIRender
// is the frame-top point this renderer already uses for deferred GPU rebuilds, and it runs in the
// headless and benchmark paths too.
//
void GltfRenderer::applyPendingEnvironmentRefresh()
{
  if(!m_envBakeDirty)
    return;

  if(isBakedEnvironment(m_resources.settings.envSystem))
  {
    refreshBakedEnvironment();
    return;
  }

  m_envBakeDirty = false;

  if(m_envOwnedByBaker && m_resources.settings.envSystem == shaderio::EnvSystem::eHdr)
  {
    // Leaving a baked sky for the HDR path: hdrIbl still points at EnvBaker's image and holds an
    // alias table built over it, so the file has to be reloaded to take ownership back.
    if(m_lastHdrFile.empty() && m_lastHdrImage)
    {
      EnvImage copy = *m_lastHdrImage;
      uploadHdrEnvironment(std::move(copy), {});
    }
    else
      createHDR(m_lastHdrFile);
  }
  // eNone samples neither hdrIbl nor the prefiltered cubes, so a stale binding is harmless -- and
  // re-selecting a baked sky re-commits before anything reads it.

  // Same rule as refreshBakedEnvironment: calibrate on the first refresh after the kind changes.
  if(m_fireflyClampCalibratedFor != int(m_resources.settings.envSystem))
    calibrateFireflyClamp();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Assemble the authored sky parameters the shaders read.
//
// Built in one place because two consumers need the same numbers for different purposes: the bake
// evaluates them with the sun excluded (that image is the lighting field, and the sun is a
// directional light NEE samples), while the per-ray background evaluates them with the sun in.
// Splitting the fill would let the visible sky and the sky that lights the scene drift apart.
//
shaderio::SkyOmiParameters GltfRenderer::buildSkyOmiParameters() const
{
  const Settings&            settings = m_resources.settings;
  shaderio::SkyOmiParameters params{};

  params.plain.color = settings.plainColor;

  params.gradient.bottomColor  = settings.gradientBottomColor;
  params.gradient.horizonColor = settings.gradientHorizonColor;
  params.gradient.topColor     = settings.gradientTopColor;
  params.gradient.sunColor     = settings.gradientSunColor;
  params.gradient.bottomCurve  = settings.gradientBottomCurve;
  params.gradient.topCurve     = settings.gradientTopCurve;
  params.gradient.sunAngleMax  = settings.gradientSunAngleMax;
  params.gradient.sunCurve     = settings.gradientSunCurve;

  // Resources::sunDirection is the one sun the whole renderer shares -- the sky UI, the physical
  // sky and next-event estimation all read it -- so the visible disk lands exactly where the light
  // that casts the shadows is. The gradient is evaluated in the environment's frame, so the sun
  // is handed over in that frame too.
  params.gradient.sunDirection = sunInEnvironmentFrame();

  params.gradient.sunAngularRadius = kGradientSunAngularRadius;

  return params;
}

glm::vec3 GltfRenderer::sunInEnvironmentFrame() const
{
  // The same conversion the shaders apply before an environment lookup (quatRotateInverse).
  return glm::inverse(m_resources.settings.envRotationQuat()) * m_resources.sunDirection;
}

//--------------------------------------------------------------------------------------------------
// Settings::envRotation was edited. The rotation is where north is, so a sky's sun is placed again
// from Time of Day on the turned compass -- what North Offset used to do. That goes through aim(),
// which also re-bakes the physical sky and reaches a marked sun light. An HDR has no sun to place.
//
void GltfRenderer::onEnvironmentRotated()
{
  if(skyHasSun(m_resources.settings.envSystem))
    m_skySun.applyTimeOfDay();
  else
    m_skySun.syncAngles();  // the sun stays put, but its bearing is on the turned compass now
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Illuminance of the gradient sky's sun, before `envIntensity`.
//
// The physical sky measures its sun: the bake reports a disk radiance, and radiance times the solid
// angle the disk subtends is an illuminance. The gradient sky measures nothing. `sunColor` is the
// colour the disk is *drawn* in, and running it through that same product is what left this sun
// visible but dark -- a half-degree disk subtends 6.8e-5 sr, so a `sunColor` of 1 became 7e-5 lux
// under a dome delivering about 1.5. Forty thousand times too dim, and silently so: the disk was
// still there to look at.
//
// So the number is scaled against the sky instead of derived from the disk. The dome's own
// illuminance is the one quantity the gradient sky does define -- it is those authored colours over
// a hemisphere -- and `kGradientSunToSkyRatio` puts the sun where a clear day puts it relative to
// that. `sunColor` keeps its job as the sun's colour *and* its relative brightness, so brightening
// it in the UI now brightens the light as well as the disk.
//
// The shadow map's light, and where its box goes.
//
// The light is the scene's marked sun when there is one -- whatever the environment, since a host
// that supplies its own sun marks it -- then the sky's own sun, then the brightest directional
// light. `shadowLight` names a scene light instead. A light that gives no light casts nothing.
void GltfRenderer::fillShadowAndGround(shaderio::SceneFrameInfo& finfo)
{
  finfo.shadowLightIndex       = -1;
  finfo.groundFadeRadius       = 0.0F;
  m_resources.shadowPassActive = false;
  const SceneGpuView& view     = m_resources.gpuView();
  if(!view.hasContent() || view.renderNodeCapacity() == 0)
    return;

  const Settings&     settings = m_resources.settings;
  const nvutils::Bbox bounds   = view.bounds();
  const glm::vec3     center   = bounds.center();
  const float         radius   = std::max(bounds.radius(), 1e-3F);

  float groundReach = 0.0F;
  if(settings.useInfinitePlane)
  {
    if(settings.groundFade > 0.0F)
    {
      finfo.groundFadeCenterX = center.x;
      finfo.groundFadeCenterZ = center.z;
      finfo.groundFadeRadius  = shadow_fit::groundReach(radius, settings.groundFade);
      groundReach             = finfo.groundFadeRadius;
    }
    else
      groundReach = radius * shadow_fit::kMaxFarRadii;
  }

  if(!settings.rasterShadows || settings.renderSystem != RenderingMode::eRasterizer)
    return;

  const std::vector<SceneGpuView::LightInfo> rlights = view.lights();
  // Toward the light, for a directional render light that shines; nullopt for anything else.
  const auto directional = [&](int i) -> std::optional<glm::vec3> {
    if(i < 0 || i >= int(rlights.size()) || !rlights[i].directional)
      return std::nullopt;
    const bool skyDriven = skyHasSun(settings.envSystem) && i == m_skySun.lightIndex();
    if(!skyDriven && rlights[i].intensity <= 0.0)
      return std::nullopt;
    const glm::vec3 z   = glm::vec3(rlights[i].worldMatrix[2]);
    const float     len = glm::length(z);
    if(len < 1e-6F)
      return std::nullopt;
    return z / len;
  };

  int                      index = -1;
  std::optional<glm::vec3> toLight;
  if(settings.shadowLight >= 0)
  {
    index   = settings.shadowLight;
    toLight = directional(index);
  }
  else if(m_skySun.lightIndex() >= 0)
  {
    index   = m_skySun.lightIndex();
    toLight = directional(index);
  }
  else if(finfo.rendererSunLight != 0)
  {
    index   = int(rlights.size());
    toLight = m_resources.sunDirection;
  }
  else
  {
    double brightest = 0.0;
    for(int i = 0; i < int(rlights.size()); ++i)
    {
      if(!directional(i))
        continue;
      if(rlights[i].intensity * rlights[i].maxColor > brightest)
      {
        brightest = rlights[i].intensity * rlights[i].maxColor;
        index     = i;
      }
    }
    toLight = directional(index);
  }
  if(!toLight)
    return;

  const shadow_fit::Result fit = shadow_fit::fit({.center        = center,
                                                  .radius        = radius,
                                                  .toLight       = *toLight,
                                                  .mapSize       = settings.shadowMapSize,
                                                  .groundY       = settings.infinitePlaneDistance,
                                                  .groundCenterX = center.x,
                                                  .groundCenterZ = center.z,
                                                  .groundReach   = groundReach});
  if(!fit.valid)
    return;

  finfo.shadowLightIndex       = index;
  finfo.shadowViewProj         = fit.viewProj;
  finfo.shadowTexelWorld       = fit.texelWorld;
  finfo.shadowDepthRange       = fit.depthRange;
  finfo.shadowBias             = settings.shadowBias * fit.texelWorld;
  finfo.shadowNormalBias       = settings.shadowNormalBias * fit.texelWorld;
  finfo.shadowMinRadius        = fit.texelWorld;
  finfo.shadowContactHardening = settings.shadowContactHardening ? 1 : 0;
  finfo.shadowMapSize          = settings.shadowMapSize;
  m_resources.shadowPassActive = true;
}

// The sky half only: the ground half of the gradient lights nothing that faces up, and including it
// would make the sun follow the ground colour.
//
glm::vec3 GltfRenderer::gradientSunIlluminance() const
{
  const Settings& settings = m_resources.settings;

  // Illuminance the dome alone puts on an up-facing surface:
  //   E = 2*PI * integral(0..PI/2) L(v) cos(v) sin(v) dv
  // with L the same horizon -> top band the shader evaluates, and v measured from the zenith.
  // Numeric because the band shape is a free exponent; 256 steps settle this to well under a
  // percent, and it is evaluated once per frame on the CPU.
  constexpr int kSteps = 256;
  glm::vec3     dome(0.0f);
  for(int i = 0; i < kSteps; ++i)
  {
    const float     v = (static_cast<float>(i) + 0.5f) / static_cast<float>(kSteps) * glm::half_pi<float>();
    const float     c = 1.0f - v / glm::half_pi<float>();
    const glm::vec3 L =
        glm::mix(settings.gradientHorizonColor, settings.gradientTopColor, gradientBandWeight(c, settings.gradientTopCurve));
    dome += L * std::cos(v) * std::sin(v);
  }
  dome *= glm::two_pi<float>() * glm::half_pi<float>() / static_cast<float>(kSteps);

  // One scalar for the dome: the ratio is a brightness, and letting it run per channel would tint
  // the sun by the inverse of the sky it stands against.
  const float domeIlluminance = (dome.x + dome.y + dome.z) / 3.0f;
  return settings.gradientSunColor * (kGradientSunToSkyRatio * domeIlluminance);
}

//--------------------------------------------------------------------------------------------------
// The atmosphere the physical sky is built from: Earth, with the panel's settings laid over it.
//
shaderio::SkyAtmosphereParameters GltfRenderer::buildAtmosphereParameters() const
{
  const Settings& s = m_resources.settings;

  // Starts from the Earth defaults for the two fields no setting covers: muSMin, which is LUT
  // coverage rather than physics, and the zero padding layer every density profile needs.
  shaderio::SkyAtmosphereParameters p = skyBrunetonEarthDefaults();

  p.solarIrradiance    = s.atmoSolarIrradiance;
  p.sunAngularRadius   = s.atmoSunAngularRadius;
  p.groundAlbedo       = s.atmoGroundAlbedo;
  p.rayleighScattering = s.atmoRayleighScattering;
  p.mieScattering      = s.atmoMieScattering;
  p.miePhaseFunctionG  = s.atmoMieAnisotropy;

  // Extinction is what the aerosol removes; scattering is the part it gives back. The albedo is
  // the ratio, and it is what the setting exposes -- OMI has no absorption field, so a scene that
  // sets mieScattering keeps the ratio rather than inheriting an unrelated extinction.
  p.mieExtinction = s.atmoMieScattering / std::max(s.atmoMieAlbedo, 1e-3F);

  p.absorptionExtinction = s.atmoOzoneExtinction;
  p.bottomRadius         = s.atmoBottomRadius;
  p.topRadius            = s.atmoBottomRadius + s.atmoThickness;

  // Both gases fall off exponentially, so a scale height is the whole profile. Layer 0 stays the
  // zero padding upstream inserts -- see skyBrunetonEarthDefaults().
  p.rayleighDensity.layers[1].expTerm  = 1.0F;
  p.rayleighDensity.layers[1].expScale = -1.0F / std::max(s.atmoRayleighScaleHeight, 1e-3F);
  p.mieDensity.layers[1].expTerm       = 1.0F;
  p.mieDensity.layers[1].expScale      = -1.0F / std::max(s.atmoMieScaleHeight, 1e-3F);

  // Ozone is a tent: zero at centre-width/2, one at the centre, zero again at centre+width/2.
  // Layer 0 covers the rise and layer 1 the fall, so the split is at the centre.
  const float halfWidth                      = std::max(s.atmoOzoneWidth, 1e-3F) * 0.5F;
  const float slope                          = 1.0F / halfWidth;
  p.absorptionDensity.layers[0].width        = s.atmoOzoneCenter;
  p.absorptionDensity.layers[0].expTerm      = 0.0F;
  p.absorptionDensity.layers[0].expScale     = 0.0F;
  p.absorptionDensity.layers[0].linearTerm   = slope;
  p.absorptionDensity.layers[0].constantTerm = 1.0F - s.atmoOzoneCenter * slope;
  p.absorptionDensity.layers[1].expTerm      = 0.0F;
  p.absorptionDensity.layers[1].expScale     = 0.0F;
  p.absorptionDensity.layers[1].linearTerm   = -slope;
  p.absorptionDensity.layers[1].constantTerm = 1.0F + s.atmoOzoneCenter * slope;

  return p;
}

//--------------------------------------------------------------------------------------------------
// Push the current settings into EnvBaker -- the parameters both the preview and the commit bake
// from.
//
// Separate from refreshBakedEnvironment because the preview path needs it too, and needs only
// this: these are member assignments, uploaded by the bake through the command buffer, so they
// are safe in the middle of recording a frame. setResolution is deliberately *not* here -- it
// reallocates the image, which is a frame-boundary operation and belongs to the commit alone.
//
// The atmosphere block is not here either: the bake reads it for ground albedo and the sun's
// angular size, but any change to it also fails SkyBruneton::lutsSatisfy, and the rebuild that
// follows uploads it.
//
void GltfRenderer::updateBakerParameters()
{
  // Metres in the UI, kilometres in the model.
  m_envBaker.setObserverAltitude(m_resources.settings.atmoObserverAltitude * 0.001F);

  m_envBaker.skyParams() = buildSkyOmiParameters();
  m_envBaker.setSkyType(shaderio::SkyType(bakeSkyType(m_resources.settings.envSystem)));
  m_envBaker.setSunDirection(sunInEnvironmentFrame());
}

//--------------------------------------------------------------------------------------------------
// Rebuild the baked lighting environment from the current settings.
//
// This is the single point where an analytic sky becomes the environment every renderer path
// samples. After it runs, hdrIbl and hdrDome hold a lat-long image, an alias table and prefiltered
// cubemaps built from EnvBaker's output instead of from an HDR file -- which is why the path
// tracer and the rasterizer need no per-sky-type branch to light a scene.
//
// Application thread only: EnvBaker::commit submits and waits.
//
void GltfRenderer::refreshBakedEnvironment()
{
  m_envBakeDirty = false;
  // A full commit supersedes any preview raised before it, and the commit that preview would owe.
  m_envPreviewPending = false;
  m_envCommitOwed     = false;
  if(!isBakedEnvironment(m_resources.settings.envSystem))
    return;

  // Both the bake and the descriptor rewrite below touch resources that in-flight frames are
  // still reading: the bake overwrites the lat-long image, and updateHdrImages() rewrites set 0
  // while a previously submitted frame may still be sampling it. createHDR() opens with
  // the same wait for the same reason -- without it this crashes, not merely trips validation.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  const Settings& settings = m_resources.settings;

  // Rebuild the LUTs unless they already hold this atmosphere at full quality. Compared by value
  // rather than tracked with a dirty flag so that an .ini restore, a benchmark script and an MCP
  // write are all covered without each having to remember to raise one -- and so that a settle
  // that changed nothing does not pay for a full precompute.
  //
  // Quality is part of the test, not just the parameters: a drag leaves the LUTs holding the right
  // atmosphere at preview quality, and this is the moment that owes the upgrade.
  if(settings.envSystem == shaderio::EnvSystem::eSky)
  {
    const shaderio::SkyAtmosphereParameters wanted = buildAtmosphereParameters();
    if(!m_skyBruneton.lutsSatisfy(wanted, SkyBruneton::kQualityFinal))
    {
      m_skyBruneton.setParameters(wanted);
      m_skyBruneton.precompute(m_resources);
    }
  }

  updateBakerParameters();

  // Only the rasterizer reads the prefiltered cubemaps; skipping them on the path-tracing path
  // saves ~28 ms of prefilter work per commit for an image nothing would sample.
  const bool refreshDome = m_resources.settings.renderSystem == RenderingMode::eRasterizer;
  m_envBaker.commit(m_resources, refreshDome);
  m_envOwnedByBaker = true;

  // Rebind the environment into the renderer's own descriptor set. HdrIbl::updateFromGpuImage
  // repoints HdrIbl's internal set, which is what the path tracer binds for the alias table --
  // but `texturesHdr[HDR_IMAGE_INDEX]` lives in set 0 here and still names whichever image was
  // bound last (at start-up, the dummy from createHDR("")). Without this the scene is lit and
  // backed by that stale image and the authored sky is silently ignored.
  //
  // Only that one element is rewritten. The BRDF LUT and the prefiltered cubes in the same
  // binding are owned by HdrEnvDome and keep their identity across an update -- rewriting them
  // per commit is the descriptor churn the design explicitly warns against.
  {
    nvvk::WriteSetContainer    write{};
    const VkWriteDescriptorSet hdrTexture =
        m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr, m_resources.descriptorSet,
                                                     HDR_IMAGE_INDEX, 1U);
    write.append(hdrTexture, m_resources.hdrIbl.getHdrImage());
    vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
  }

  // A baked environment has its own luminance integral, exactly like a loaded HDR -- but this runs
  // on every commit, and a commit happens every time the sun moves or an atmosphere slider settles.
  //
  // So calibrate once, on the first commit after the environment changes kind, and then leave it
  // alone. Two reasons, and the second is the important one:
  //
  //   - The value would otherwise swing with the sun, because the sky's integral does.
  //   - It is a *starting point*, not a derived quantity. The clamp has to accommodate every light
  //     in the scene -- emissive materials, punctual lights, a bright HDR backplate -- and the
  //     environment integral knows about none of them. Once the user has adjusted it, recomputing
  //     from the environment alone throws that away, and does so on a gesture as ordinary as
  //     dragging the sun.
  //
  // This is where the calibration belongs rather than at the switch itself: hdrIbl's integral does
  // not describe the new environment until the bake above has run.
  if(m_fireflyClampCalibratedFor != int(settings.envSystem))
  {
    calibrateFireflyClamp();
  }
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Returns the scene-appropriate firefly clamp threshold:
//   - HDR or baked analytic environment → use its luminance integral (already calibrated to the
//     environment's range; a baked sky populates the same integral through updateFromGpuImage)
//   - Otherwise → a fixed baseline high enough to preserve legitimately bright highlights
void GltfRenderer::calibrateFireflyClamp()
{
  m_fireflyClampCalibratedFor = int(m_resources.settings.envSystem);
  // a host owns the clamp (ptFireflyClamp, from its exposure): an environment's kind changing must not take it
  if(!m_resources.hostFed || m_resources.sceneFromFile)
    m_pathTracer.m_pushConst.fireflyClampThreshold = defaultFireflyClamp();
}

//--------------------------------------------------------------------------------------------------
// Where aerial perspective stands the eye. See the declaration for why this is not the sky's
// observer altitude.
//
float GltfRenderer::aerialPerspectiveEyeAltitudeKm() const
{
  const float liftKm = m_resources.settings.atmoObserverAltitude * 0.001F;

  if(!m_resources.gpuView().hasContent() || m_cameraManip == nullptr)
    return liftKm;

  const nvutils::Bbox bounds = m_resources.gpuView().bounds();
  if(!bounds.isEmpty())
  {
    // The scene's floor goes `liftKm` above the planet's surface, so nothing in it is ever
    // underground however the camera moves; the eye then stands wherever it stands above that
    // floor. A camera below the floor -- inside the terrain, or under a ground plane -- would
    // otherwise push itself under the surface instead, so the lift is the floor for that too.
    const float kmPerSceneUnit = 0.001F * m_resources.settings.atmoAerialPerspectiveScale;
    // The camera manipulator works in doubles; the difference is a scene-scale number, so it fits a
    // float long before it is turned into kilometres.
    const float aboveFloor = static_cast<float>(m_cameraManip->getEye().y - bounds.min().y) * kmPerSceneUnit;
    return liftKm + std::max(aboveFloor, 0.0F);
  }
  return liftKm;
}

glm::vec3 GltfRenderer::skySunIlluminanceAboveAtmosphere() const
{
  // The constant is written in shader types; float3 is an alias that lives in `shaderio`.
  using shaderio::float3;
  // The sun's factor set, not the sky's -- this is the sun. See SUN_SPECTRAL_TO_LUMINANCE.
  const glm::vec3 radianceToLuminance = SUN_RADIANCE_TO_LUMINANCE;
  return buildAtmosphereParameters().solarIrradiance * radianceToLuminance * m_resources.settings.hdrEnvIntensity;
}

float GltfRenderer::defaultFireflyClamp() const
{
  const shaderio::EnvSystem env = m_resources.settings.envSystem;
  if(env != shaderio::EnvSystem::eHdr && !isBakedEnvironment(env))
    return 30.0f;

  // The environment's own integral: the sum of max(R,G,B) over solid angle.
  float clamp = m_resources.hdrIbl.getIntegral();

  // The physical sky keeps its sun *out* of that image -- the sun is a directional light so that
  // its disk stays sharp and next-event estimation can sample it -- so the integral measures the
  // sky alone, and for a daylit scene the sky is the smaller half by a wide margin. An .hdr file
  // has its sun baked in and needs no correction, which is why this only surfaced once the
  // analytic skies grew a sun of their own.
  //
  // Added in the integral's own units: a disk of radiance L covering solid angle W contributes
  // max(L) * W to that sum, and radiance times solid angle is illuminance. So the term is the
  // largest channel of the sun's illuminance.
  //
  // Which sun, though -- the one above the atmosphere, not the one currently shining. The
  // attenuated sun is the physically honest answer to "how much light is there right now", and it
  // is the wrong answer here, because it goes to zero at dusk and takes the clamp with it: at 2
  // degrees below the horizon this threshold was 0.34, at 5 degrees 0.03, and at 10 degrees it
  // reached exactly 0 -- which the shader reads as "disabled", so the clamp flipped from crushing
  // every sample to not clamping at all. A clamp is a safety net against outliers, and a safety
  // net that scales itself to nothing at night is worse than a fixed one.
  //
  // The unattenuated value is a property of the star rather than of the hour, so it is stable
  // across a whole time-of-day scrub, and it still follows the atmosphere: Mars' sun is dimmer
  // than Earth's, and editing the solar irradiance moves it.
  //
  // The gradient sky keeps its sun out of the bake for the same reason and needs the same
  // correction; there is just no atmosphere to ask, so the term is the sun this sky implies.
  glm::vec3 sun(0.0f);
  if(env == shaderio::EnvSystem::eSky)
    sun = skySunIlluminanceAboveAtmosphere();
  else if(env == shaderio::EnvSystem::eGradient)
    sun = gradientSunIlluminance() * m_resources.settings.hdrEnvIntensity;
  clamp += std::max({sun.x, sun.y, sun.z});
  return clamp;
}

//--------------------------------------------------------------------------------------------------
// Apply the tonemapper on the rendered image
bool GltfRenderer::tonemap(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);
  return runTonemapPass(cmd);
}

bool GltfRenderer::runTonemapPass(VkCommandBuffer cmd)
{
  // Select which buffer to tonemap based on user selection
  VkDescriptorImageInfo inputBuffer =
      m_resources.gBuffers.getColorSampleDescriptorImageInfo(Resources::eImgRendered, m_resources.linearSampler);
  VkExtent2D gbufSize         = m_resources.gBuffers.getSize();
  bool       usingGuideBuffer = false;
  bool       bypassTonemapper = false;


  // OptiX / OIDN denoised output (path-tracer only). Routed via the global DisplayBuffer toggle the
  // denoisers write when they denoise or the user clicks their thumbnail.
#if defined(USE_OPTIX_DENOISER)
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_resources.settings.displayBuffer == DisplayBuffer::eOptixDenoised)
  {
    const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser();
    if(optix && optix->hasValidDenoisedOutput())
    {
      inputBuffer         = optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferDenoised);
      inputBuffer.sampler = m_resources.linearSampler;
    }
  }
#endif
#if defined(USE_OIDN_DENOISER)
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_resources.settings.displayBuffer == DisplayBuffer::eOidnDenoised)
  {
    const OidnDenoiser* oidn = m_pathTracer.getOidnDenoiser();
    if(oidn && oidn->hasValidDenoisedOutput())
    {
      inputBuffer         = oidn->getDescriptorImageInfo(OidnDenoiser::eGBufferDenoised);
      inputBuffer.sampler = m_resources.linearSampler;
    }
  }
#endif
  // The DLSS hand-off, shown in place of the denoiser's view while OptiX or OIDN takes over from DLSS
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
  {
    if(const VkDescriptorImageInfo* handoff = m_pathTracer.handoffImage(m_resources))
    {
      inputBuffer         = *handoff;
      inputBuffer.sampler = m_resources.linearSampler;
    }
  }

#if defined(USE_DLSS)
  // Displaying or not DLSS guide buffer.
  if(Dlss* dlss = activeDlss())
  {
    if(auto guide = dlss->activeGuideImage())
    {
      inputBuffer         = guide->image;
      inputBuffer.sampler = m_resources.linearSampler;
      gbufSize            = guide->extent;
      usingGuideBuffer    = true;

      // Clear the output image to a distinct color so the DLSS render-resolution borders are
      // visible when the guide buffer is smaller than the display.
      VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
      vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                           &kBackgroundClearColor, 1, &range);
      // Barrier: clear must complete before tonemapper compute shader runs
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    }
  }
#endif

  // Disable tonemapping for debug buffers or guide buffers (display raw values)
  shaderio::TonemapperData tonemapperData = m_resources.tonemapperData;
  const bool               vizNormal      = (m_resources.settings.visualization == shaderio::Visualization::eRendered
                          || m_resources.settings.visualization == shaderio::Visualization::eClay);
  if(!vizNormal || usingGuideBuffer)
    tonemapperData.isActive = 0;
  if(bypassTonemapper)
    tonemapperData.isActive = 0;

  m_resources.tonemapper.runCompute(cmd, gbufSize, tonemapperData, inputBuffer,
                                    m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgTonemapped));

  // Barrier: tonemapper compute writes must be visible to both the DLSS-NR compute pass (which
  // reads eImgTonemapped immediately after tonemap()) and the silhouette fragment shader.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

  // NR post-process is only valid on the main rendered path (not guide buffers, viz modes, or bypassed tonemapper).
  return vizNormal && !usingGuideBuffer && !bypassTonemapper;
}

//--------------------------------------------------------------------------------------------------
// Render the silhouette of the selected object(s)
//
// Selection visualization uses an ObjectID buffer (R32_UINT, filled on frame 0 by the path tracer
// with the first-hit render node ID per pixel) and a selection bitmask (one bit per
// render node). The silhouette compute shader reads ObjectID per pixel, checks the bitmask, and
// applies a Sobel edge filter where the bit is set, then composites the outline onto the image.
void GltfRenderer::silhouette(VkCommandBuffer cmd)
{
  // Run the silhouette pass only when something is selected (one primitive or a node and its branch).
  // We rebuild the CPU bitmask from selectedRenderNodes, upload it, then dispatch the silhouette compute.
  if(!m_resources.selectedRenderNodes.empty())
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

    // Rebuild selection bitmask and ensure GPU buffer exists and is up to date.
    int numRenderNodes = static_cast<int>(m_resources.gpuView().renderNodeCapacity());
    m_resources.updateSelectionBitMask(numRenderNodes);
    const VkDeviceSize bitmaskBytes = m_resources.selectionBitMask.size() * sizeof(uint32_t);
    if(bitmaskBytes > 0)
    {
      // Build or update the selection bitmask buffer, a power of two of the bytes it needs (render
      // node capacity grows in powers of two); a smaller one goes once no frame in flight reads it
      if(m_resources.bSelectionBitMask.buffer == VK_NULL_HANDLE || m_resources.bSelectionBitMask.bufferSize < bitmaskBytes)
      {
        if(m_resources.bSelectionBitMask.buffer != VK_NULL_HANDLE)
          m_app->submitResourceFree([alloc = &m_resources.allocator, buffer = m_resources.bSelectionBitMask]() mutable {
            alloc->destroyBuffer(buffer);
          });
        m_resources.bSelectionBitMask = {};
        NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bSelectionBitMask, std::bit_ceil(bitmaskBytes),
                                                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                      VMA_MEMORY_USAGE_CPU_TO_GPU));
        NVVK_DBG_NAME(m_resources.bSelectionBitMask.buffer);
      }

      // In case the bitmask is too large (> 524,288 render nodes), we need to update it in chunks
      constexpr VkDeviceSize kMaxCmdUpdateSize = 65536;
      for(VkDeviceSize offset = 0; offset < bitmaskBytes; offset += kMaxCmdUpdateSize)
      {
        const VkDeviceSize chunkSize = std::min(kMaxCmdUpdateSize, bitmaskBytes - offset);
        vkCmdUpdateBuffer(cmd, m_resources.bSelectionBitMask.buffer, offset, chunkSize,
                          reinterpret_cast<const uint8_t*>(m_resources.selectionBitMask.data()) + offset);
      }
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    std::vector<VkDescriptorImageInfo> imageInfos = {
        m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgSelection),
        m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgTonemapped),
    };
    VkDescriptorBufferInfo bitmaskBufferInfo = {m_resources.bSelectionBitMask.buffer, 0, bitmaskBytes};
    m_silhouette.dispatch(cmd, m_resources.gBuffers.getSize(), imageInfos, bitmaskBufferInfo,
                          static_cast<uint32_t>(m_resources.selectionBitMask.size()));
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
  }
}

//--------------------------------------------------------------------------------------------------
// Render visual helpers (grid + transform gizmo) onto the tonemapped image
void GltfRenderer::renderVisualHelpers(VkCommandBuffer cmd)
{
  // Sync grid and snap settings to visual helpers
  m_visualHelpers.grid.setVisible(m_resources.settings.showGrid);
  m_visualHelpers.grid.style().baseUnit = m_resources.settings.gridUnit;
  m_visualHelpers.transform.setSnapEnabled(m_resources.settings.snapEnabled);
  m_visualHelpers.transform.setSnapValues(m_resources.settings.gridUnit, m_resources.settings.snapRotation,
                                          m_resources.settings.snapScale);

  // Sync gizmo attachment from selection state
  updateGizmoAttachment();

  if(!m_visualHelpers.shouldRender())
    return;

  NVVK_DBG_SCOPE(cmd);

  const VkExtent2D size = m_resources.gBuffers.getSize();
  glm::vec2        viewportSize(static_cast<float>(size.width), static_cast<float>(size.height));
  glm::vec2        depthBufferSize = viewportSize;
#if defined(USE_DLSS)
  // The path tracer writes depth at the DLSS render size into the corner of the G-buffer depth; the
  // rasterizer blits DLSS's inner depth up to full size. isActive() is strict (false during async
  // init), so we never pick up a 0x0 inner extent.
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
    if(const Dlss* dlss = activeDlss(); dlss && dlss->isActive())
    {
      const VkExtent2D rs = dlss->getRenderSize();
      depthBufferSize     = {static_cast<float>(rs.width), static_cast<float>(rs.height)};
    }
#endif

  m_visualHelpers.render(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                         m_resources.gBuffers.getColorAttachmentView(Resources::eImgTonemapped), m_resources.descriptorSet,
                         m_cameraManip->getViewMatrix(), m_cameraManip->getPerspectiveMatrix(), viewportSize, depthBufferSize);
}

//--------------------------------------------------------------------------------------------------
// Update gizmo attachment based on current node selection
void GltfRenderer::updateGizmoAttachment()
{
  if(!m_resources.settings.showGizmo || !m_resources.gpuView().hasContent())
  {
    if(!m_gizmoEntity.isNone())
    {
      m_visualHelpers.transform.clearAttachment();
      m_gizmoEntity = {};
    }
    return;
  }
  updateEntityGizmo();
}

// The gizmo on a stream host's selected entity: its local TRS and its parent's world matrix from the
// store, its handles from the host; a drag goes to the host (GizmoHooks::edit), which writes the store.
void GltfRenderer::updateEntityGizmo()
{
  rs::RetainedScene* store  = m_resources.retainedScene.get();
  const rs::Entity*  entity = store && !m_selectedEntity.isNone() ? store->entity(m_selectedEntity) : nullptr;
  const uint32_t     flags  = entity && m_gizmoHooks.attach ? m_gizmoHooks.attach(m_selectedEntity) : 0;
  if(flags == 0)
  {
    if(!m_gizmoEntity.isNone())
    {
      m_visualHelpers.transform.clearAttachment();
      m_gizmoEntity = {};
    }
    return;
  }
  const auto read = [&] {
    glm::quat rotation;
    if(entity->local.matrix)
    {
      glm::vec3 skew;
      glm::vec4 perspective;
      glm::decompose(*entity->local.matrix, m_gizmoScale, rotation, m_gizmoPosition, skew, perspective);
    }
    else
    {
      m_gizmoPosition = entity->local.translation;
      rotation        = entity->local.rotation;
      m_gizmoScale    = entity->local.scale;
    }
    m_gizmoRotation          = glm::degrees(glm::eulerAngles(rotation));
    m_gizmoParentWorldMatrix = entity->parent.isNone() ? glm::mat4(1.f) : store->world(entity->parent);
  };
  if(m_selectedEntity != m_gizmoEntity || (flags != m_gizmoFlags && !m_visualHelpers.transform.isDragging()))
  {
    m_gizmoEntity = m_selectedEntity;
    m_gizmoFlags  = flags;
    read();
    m_visualHelpers.transform.attachTransform(&m_gizmoPosition, &m_gizmoRotation, &m_gizmoScale, flags);
    m_visualHelpers.transform.setParentWorldMatrix(&m_gizmoParentWorldMatrix);
  }
  else if(!m_visualHelpers.transform.isDragging())
    read();  // the host's edits and what moves above it show at once
}

//--------------------------------------------------------------------------------------------------
// Set DLSS hardware/extension availability per NGX feature (Ray Reconstruction / Super Resolution).
// This should be called early, before any DLSS initialization occurs.
void GltfRenderer::setDlssHardwareAvailability(bool rrAvailable, bool srAvailable)
{
  m_resources.settings.dlssRrHardwareAvailable = rrAvailable;
  m_resources.settings.dlssSrHardwareAvailable = srAvailable;
}

//--------------------------------------------------------------------------------------------------
// Set Shader Execution Reordering (VK_EXT_ray_tracing_invocation_reorder) availability. Call
// before the path tracer is attached: it decides whether SER is compiled into the shader.
void GltfRenderer::setSerAvailable(bool available)
{
  m_resources.settings.serHardwareAvailable = available;
}

//--------------------------------------------------------------------------------------------------
// Names of the extensions enabled on the device. Call before the path tracer is attached: the
// vk_oidn denoiser picks its optional features (tensor cores, ...) from them.
void GltfRenderer::setDeviceExtensions(std::vector<std::string> extensions)
{
  m_resources.deviceExtensions = std::move(extensions);
}

//--------------------------------------------------------------------------------------------------
// Set Opacity Micromap (VK_KHR_opacity_micromap) availability: whether the device supports it, and
// whether the user enabled it (--useOpacityMicromap). When unavailable, the
// EXT_mesh_opacity_micromap glTF extension is ignored. Call early, before scene creation.
void GltfRenderer::setOpacityMicromapAvailable(bool hardware, bool enabled)
{
  m_resources.settings.opacityMicromapMissing   = enabled && !hardware;
  m_resources.settings.opacityMicromapSupported = hardware && enabled;
}

//--------------------------------------------------------------------------------------------------
// Load the scene
bool GltfRenderer::createScene(const std::filesystem::path& sceneFilename)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  m_resources.selectedRenderNodes.clear();

  if(sceneFilename.empty())
  {
    return false;
  }

  std::filesystem::path filename = nvutils::findFile(sceneFilename, nvsamples::getResourcesDirs(), false);
  if(!filename.has_filename())
  {
    LOGW("Cannot find file: %s\n", nvutils::utf8FromPath(sceneFilename).c_str());
    return false;
  }

  // The file goes the way a prop's does: read, imported into the store, uploaded
  LOGI("Loading scene: %s\n", nvutils::utf8FromPath(filename).c_str());
  tinygltf::Model model;
  std::string     error;
  if(!fv::composer::readModelFile(filename, model, error))
  {
    LOGW("Error loading scene: %s: %s\n", nvutils::utf8FromPath(filename).c_str(), error.c_str());
    return false;
  }
  m_resources.retainedScene = std::make_unique<rs::RetainedScene>();
  m_resources.hostFed       = true;
  m_resources.sceneFromFile = true;
  rs::importGltf(*m_resources.retainedScene, model, {.prop = 1, .imageSearchPaths = {filename.parent_path()}});
  replaceHostScene();
  placeCameraFromFile(model);
  return true;
}

// The file's first camera, or a fit of the scene when it has none
void GltfRenderer::placeCameraFromFile(const tinygltf::Model& model)
{
  const nvutils::Bbox bounds = m_resources.gpuView().bounds();
  bool                placed = false;
  const auto          place  = [&](int nodeId, const glm::mat4& world) {
    const tinygltf::Node&   node = model.nodes[size_t(nodeId)];
    const tinygltf::Camera& from = model.cameras[size_t(node.camera)];
    const bool              orthographic = from.type != "perspective";

    nvutils::CameraManipulator::Camera camera;
    camera.projectionType = orthographic ? nvutils::CameraManipulator::ProjectionType::Orthographic :
                                           nvutils::CameraManipulator::ProjectionType::Perspective;
    double zNear = orthographic ? from.orthographic.znear : from.perspective.znear;
    double zFar  = orthographic ? from.orthographic.zfar : from.perspective.zfar;
    if(zFar <= zNear)
      zFar = std::max(zNear * 2.0, 4.0 * double(bounds.radius()));
    camera.nearFar = {zNear, zFar};
    if(orthographic)
    {
      camera.orthMag = {from.orthographic.xmag, from.orthographic.ymag};
      camera.fov     = 45.0;
    }
    else
      camera.fov = glm::degrees(from.perspective.yfov);

    // looking down the node's -Z, as far as the scene's centre
    const glm::dmat4 m(world);
    const glm::dvec3 eye     = glm::dvec3(m[3]);
    const glm::dvec3 forward = -glm::dmat3(m) * glm::dvec3(0.0, 0.0, 1.0);
    camera.eye               = eye;
    camera.ctr               = eye + std::abs(glm::dot(glm::dvec3(bounds.center()) - eye, forward)) * forward;
    camera.up                = glm::dvec3(0.0, 1.0, 0.0);
    if(node.extras.IsObject())
    {
      tinygltf::utils::getArrayValue(node.extras, "camera::eye", camera.eye);
      tinygltf::utils::getArrayValue(node.extras, "camera::center", camera.ctr);
      tinygltf::utils::getArrayValue(node.extras, "camera::up", camera.up);
    }
    m_cameraManip->setCamera(camera);
    placed = true;
    return true;
  };
  if(!model.scenes.empty())
    for(int root : model.scenes[size_t(std::max(model.defaultScene, 0))].nodes)
      if(!placed)
        tinygltf::utils::traverseSceneGraph(model, root, glm::mat4(1), place);
  if(!placed && !bounds.isEmpty())
  {
    m_cameraManip->fit(bounds.min(), bounds.max(), true);
    m_cameraManip->setClipPlanes(glm::vec2(0.001F * bounds.radius(), 100.0F * bounds.radius()));
  }
}

//--------------------------------------------------------------------------------------------------
// Helper function to cleanup the current scene
//
void GltfRenderer::cleanupScene(bool keepImagesForReuse)
{
  // Drop any renderer-side state tied to the outgoing Scene BEFORE the unique_ptr is reset.
  // The heap allocator is free to hand the same address back to the next Scene instance, so
  // any Scene-pointer-based invalidation inside the render loop would be unreliable -- this is
  // the authoritative invalidation point.
  m_pathTracer.onSceneInvalidated(m_resources);
  m_rasterizer.onSceneInvalidated(m_resources);
  // The retained scene is kept, emptied, when images are kept for reuse (retainedSpare); otherwise
  // destroyed now, its backend waiting for the device and running what it retired.
  if(m_resources.retainedGpu && keepImagesForReuse)
  {
    m_resources.retainedGpu->clear();
    m_resources.retainedSpare = std::move(m_resources.retainedGpu);
  }
  m_resources.retainedGpu.reset();
  if(!keepImagesForReuse)
    m_resources.retainedSpare.reset();
  if(!m_resources.hostFed)  // a stream host's store is the session's
    m_resources.retainedScene.reset();
  m_resources.selectedRenderNodes.clear();
  m_resources.animationControl      = AnimationControl{};
  m_resources.animationControl.play = m_animPlay;

  // Drop async pick state tied to the outgoing scene - a pending click ray-pick would otherwise
  // be consumed by the newly loaded scene once its semaphore settles, applying a pick result to the wrong node indices.
  m_pendingClickPick.reset();
  m_pendingClickResult.reset();

}

//--------------------------------------------------------------------------------------------------
// The retained scene's backend over the renderer's allocator, uploader, sampler pool and the scene's
// texture and sampler arrays
//
rs::VulkanGpuBackend::Info GltfRenderer::retainedBackendInfo()
{
  nvapp::Application* app = m_app;
  return {.device             = m_device,
          .physicalDevice     = m_app->getPhysicalDevice(),
          .alloc              = &m_resources.allocator,
          .staging            = &m_resources.staging,
          .samplerPool        = &m_resources.samplerPool,
          .submitResourceFree = [app](std::function<void()>&& fn) { app->submitResourceFree(std::move(fn)); },
          .bindings           = &m_resources.descriptorBinding[0],
          .descriptorSet      = m_resources.descriptorSet,
          .textureBinding     = shaderio::BindingPoints::eTextures,
          .samplerBinding     = shaderio::BindingPoints::eSamplers,
          .maxTextures        = m_maxTextures,
          .maxSamplers        = m_maxSamplers,
          .blasScratchBudget  = VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20,
          .opacityMicromaps   = m_resources.settings.opacityMicromapSupported,
          .takePrefetched = [this](const std::filesystem::path& path, bool srgb) { return m_resources.imagePrefetch.take(path, srgb); },
          .maxImageExtent = [this] { return m_maxTextureExtent.load(); },
          .imageHeadroom  = [this] { return sceneHeadroom(); }};
}

//--------------------------------------------------------------------------------------------------
// A stream host's own retained scene (renderer.hpp, hostScene)
//
rs::RetainedScene& GltfRenderer::hostScene()
{
  if(!m_resources.hostFed)
  {
    m_resources.retainedScene = std::make_unique<rs::RetainedScene>();
    m_resources.hostFed       = true;
  }
  return *m_resources.retainedScene;
}

void GltfRenderer::replaceHostScene()
{
  rs::RetainedScene& store = hostScene();
  fv::timing::Scope  timeScene("retained scene");
  {
    fv::timing::Scope t("wait idle + cleanup");
    // the teardown loadModel runs, for the same reasons; the host's store and selection stay
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    m_loadPipeline.clear();
    const rs::EntityId selected = m_selectedEntity;
    // residencyRetireOnReplace: the backend stays, out of the cleanup's way, and syncs the new scene
    // over the old one below
    std::unique_ptr<rs::RetainedSceneVk> kept;
    if(m_resources.settings.residencyRetireOnReplace)
      kept = std::move(m_resources.retainedGpu);
    cleanupScene(/*keepImagesForReuse=*/true);
    m_resources.retainedGpu = std::move(kept);
    m_selectedEntity        = selected;  // where its entity is still in the scene
    if(!store.clip(m_hostClip.clip()))
      m_hostClip = {};
  }
  // the previous scene's backend, emptied, when the cleanup kept it: its decoded images are this one's to take
  const bool retire = m_resources.retainedGpu != nullptr;
  if(retire)
    m_resources.retainedSpare.reset();
  else if(m_resources.retainedSpare)
  {
    m_resources.retainedGpu = std::move(m_resources.retainedSpare);
    m_resources.retainedGpu->setBlasScratchBudget(VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20);
  }
  else
    m_resources.retainedGpu = std::make_unique<rs::RetainedSceneVk>(
        retainedBackendInfo(),
        rs::ResidencyOptions{.minInstances        = std::max(m_resources.settings.residencyMinInstances, 1u),
                             .blasScratchBudget   = VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20,
                             .textureGraceBytes   = VkDeviceSize(m_resources.settings.textureGraceMB) << 20,
                             .refitsBeforeRebuild = m_resources.settings.blasRefitsBeforeRebuild,
                             .compactStatic       = m_resources.settings.blasCompactStatic});
  // the queue is idle: what the old scene let go of isn't in the new one's way when it is fitted
  m_resources.retainedGpu->runRetired();
  {
    fv::timing::Scope timeUpload("retained upload");
    VkCommandBuffer   cmd = m_app->createTempCmdBuffer();
    if(fv::timing::enabled())
      m_resources.retainedGpu->timeBuilds();
    m_resources.retainedGpu->setShaderFeatures(drawableFeatures());
    if(retire)
    {
      m_resources.retainedGpu->beginFrame();
      m_resources.retainedGpu->sync(cmd, store);
    }
    else
      m_resources.retainedGpu->replace(cmd, store);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    logRetainedNotices();
    m_resources.imagePrefetch.drop();  // what this scene did not take would never be taken
    m_pendingGeometry = 0;
    if(const std::optional<rs::BuildTimes> times = m_resources.retainedGpu->buildTimes())
    {
      OneShotGpuTimer::report("GPU BLAS build", times->blasMs);
      OneShotGpuTimer::report("GPU TLAS build", times->tlasMs);
    }
  }
  const rs::Residency&             residency = m_resources.retainedGpu->residency();
  const rs::Residency::ImageCounts images    = residency.imageCounts();
  LOGI("Scene (retained, host): %u instances, %u prims, %u materials, %u lights; images %u decoded (%u prefetched), %u reused; GPU memory %.1f MB\n",
       residency.instanceHighWater(), residency.primHighWater(), residency.materialHighWater(), residency.lightCount(),
       images.decoded, m_resources.retainedGpu->takePrefetchCount(), images.reused,
       m_resources.retainedGpu->memoryStats().usedBytes / (1024.0 * 1024.0));

  // the camera stays
  m_resources.currentFeatureSet = {};
  updateRetainedFeatures(false);
  calibrateFireflyClamp();
  if(const nvutils::Bbox bounds = m_resources.gpuView().bounds(); !bounds.isEmpty())
    m_resources.settings.infinitePlaneDistance = bounds.min().y;
  refreshEntitySelection();
  resetFrame();
}

void GltfRenderer::selectEntity(rs::EntityId entity)
{
  // the same again (a click on nothing with nothing selected) changes no picture: the still keeps its samples
  const bool same  = entity == m_selectedEntity;
  m_selectedEntity = entity;
  refreshEntitySelection();
  if(!same)
    resetFrame();
}

void GltfRenderer::refreshEntitySelection()
{
  if(!m_resources.hostFed)
    return;
  m_resources.selectedRenderNodes.clear();
  rs::RetainedScene& store = *m_resources.retainedScene;
  if(!m_selectedEntity.isNone() && !store.entity(m_selectedEntity))
    m_selectedEntity = {};  // gone with what it was in
  if(m_selectedEntity.isNone() || !m_resources.retainedGpu)
    return;
  const rs::Residency& residency = m_resources.retainedGpu->residency();
  for(uint32_t slot = 0; slot < residency.instanceHighWater(); slot++)
    if(const rs::Residency::Instance* inst = residency.instance(slot))
      if(store.isWithin(inst->entity, m_selectedEntity))
        m_resources.selectedRenderNodes.insert(int(slot));
}

void GltfRenderer::setHostClip(rs::ClipId clip, float at)
{
  m_hostClip = !clip.isNone() && m_resources.retainedScene && m_resources.retainedScene->clip(clip) ?
                   rs::ClipPlayer(*m_resources.retainedScene, clip) :
                   rs::ClipPlayer{};
  if(!m_hostClip.clip().isNone())
    m_hostClip.setTime(std::clamp(m_hostClip.start() + at, m_hostClip.start(), std::max(m_hostClip.start(), m_hostClip.end())));
  m_resources.animationControl.seek = true;  // shown where it is at the next frame
}

void GltfRenderer::seekHostClip(float time)
{
  if(m_hostClip.clip().isNone())
    return;
  m_hostClip.setTime(std::clamp(m_hostClip.start() + time, m_hostClip.start(), std::max(m_hostClip.start(), m_hostClip.end())));
  m_resources.animationControl.play = false;
  m_resources.animationControl.seek = true;
}

// The clip playing moved on by this frame's time and written into the store's entities.
bool GltfRenderer::advanceHostClip()
{
  AnimationControl&  ctrl  = m_resources.animationControl;
  rs::RetainedScene& store = *m_resources.retainedScene;
  if(m_hostClip.clip().isNone() || !store.clip(m_hostClip.clip()) || !ctrl.doAnimation())
    return false;
  if(ctrl.isReset())
    m_hostClip.reset();
  else if(!ctrl.seek)
  {
    m_hostClip.advance(ctrl.deltaTime(), ctrl.loop);
    if(!ctrl.loop && m_hostClip.time() >= m_hostClip.end())
      ctrl.play = false;
  }
  const bool wrote = m_hostClip.apply(store);
  if(wrote)
    ctrl.clearStates();
  return wrote;
}

//--------------------------------------------------------------------------------------------------
// What the retained scene's last syncs could not do as asked (a repack, a material drawn without its
// textures), in the log
//
void GltfRenderer::logRetainedNotices()
{
  for(const std::string& notice : m_resources.retainedGpu->takeNotices())
    LOGW("Retained scene: %s\n", notice.c_str());
  for(const std::string& notice : m_resources.retainedGpu->takeUserNotices())
    if(m_noticeSink)
      m_noticeSink(notice);
}

//--------------------------------------------------------------------------------------------------
// The retained scene's frame: the clip moved on, then the
// residency's sync of what the host fed since the last frame. True when the frame changed.
//
bool GltfRenderer::syncRetainedScene(VkCommandBuffer cmd)
{
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Retained sync");
  advanceHostClip();
  m_resources.retainedGpu->beginFrame();
  m_resources.retainedGpu->setShaderFeatures(drawableFeatures());
  const rs::SyncResult result = m_resources.retainedGpu->sync(cmd, *m_resources.retainedScene);
  if(result.drawListChanged)
    m_pendingGeometry = 0;  // what was accepted is resident, and counted as used
  logRetainedNotices();
  if(const auto compaction = m_resources.retainedGpu->takeCompaction(); compaction.blases && fv::timing::enabled())
  {
    constexpr double kMB = 1024.0 * 1024.0;
    fv::timing::line(fmt::format("retained BLAS compaction: {} BLASes, {:.1f} MB to {:.1f} MB; every BLAS {:.1f} MB",
                                 compaction.blases, compaction.before / kMB, compaction.after / kMB, compaction.blasBytes / kMB));
  }
  // the variant only grows (D15), or a removal earlier in the session would compile a smaller one
  // now, with a queue wait
  if(result.featuresChanged)
    updateRetainedFeatures(true);
  return afterRetainedSync(result);
}

// What a retained sync's result asks of the renderers; true when the frame changed
bool GltfRenderer::afterRetainedSync(const rs::SyncResult& result)
{
  // The rasterizer's recorded draws name instances and buffers; any change records them again
  if(result.changed)
    m_rasterizer.freeRecordCommandBuffer(m_resources);
  // instances came or went: the selection's outline follows
  if(result.drawListChanged)
    refreshEntitySelection();
  if(result.dlssReset)
  {
    if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
      m_pathTracer.notifyDlssContentReset(m_resources);
    else
      m_rasterizer.notifyDlssContentReset(m_resources);
  }
  return result.changed;
}

void GltfRenderer::updateRetainedFeatures(bool grow)
{
  nvvkgltf::SceneFeatureSet features = m_resources.retainedScene->features();
  if(grow)  // a removal never recompiles mid-session: more compiled in still draws the rest
  {
    nvvkgltf::SceneFeatureSet had = m_resources.currentFeatureSet;
    had.set(nvvkgltf::SceneFeatureSet::eDlssGuide, false);
    features |= had;
  }
  features.set(nvvkgltf::SceneFeatureSet::eDlssGuide, dlssGuideRequired());
  m_resources.currentFeatureSet = features;
}

std::optional<nvvkgltf::SceneFeatureSet> GltfRenderer::drawableFeatures()
{
  if(m_resources.settings.renderSystem != RenderingMode::ePathtracer)
    return std::nullopt;
  return m_pathTracer.liveShaderFeatures();
}

//--------------------------------------------------------------------------------------------------
// Clear the G-Buffer
void GltfRenderer::clearGbuffer(VkCommandBuffer cmd)
{
  VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
  vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                       &kBackgroundClearColor, 1, &range);

  // Ensure the clear operation completes before any subsequent reads from this image
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
}

//--------------------------------------------------------------------------------------------------
// Create the uniform buffer for frame-specific data: per-frame camera matrices, environment
// settings and debug information, updated each frame.
//
void GltfRenderer::createResourceBuffers()
{
  // Create the buffer of the current camera transformation, changing at each frame
  NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bFrameInfo, sizeof(shaderio::SceneFrameInfo),
                                                VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                VMA_MEMORY_USAGE_CPU_TO_GPU));
  NVVK_DBG_NAME(m_resources.bFrameInfo.buffer);
}

//--------------------------------------------------------------------------------------------------
// Create the descriptor set and the pipelines
// There are two descriptor: one for the textures (set) and one (push) for the top level acceleration structure and the default output image
// There are two pipelines: one for the PathTracer and one for the Rasterizer
// The descriptor set is shared between the two pipelines
void GltfRenderer::createDescriptorSets()
{
  // Reserve texture descriptors (m_maxTextures from renderer.hpp, clamped to device limits)
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &deviceProperties);
  m_maxTextures = std::min(m_maxTextures, deviceProperties.limits.maxDescriptorSetSampledImages - 1);  // Set limits of sample textures (defaut: 100 000)
  m_maxSamplers = std::min(m_maxTextures, deviceProperties.limits.maxDescriptorSetSamplers - 1);

  // 0: Descriptor SET: scene material images (SAMPLED_IMAGE) and samplers (SAMPLER) are now separate
  // bindless arrays instead of one combined-image-sampler array (see GltfTextureInfo.index/samplerIndex).
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTextures, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                              m_maxTextures, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eSamplers, VK_DESCRIPTOR_TYPE_SAMPLER,
                                              m_maxSamplers, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  // 2D IBL/transmission textures (see shaderio.h HDR_* indices):
  //   [0] HDR_IMAGE_INDEX   lat-long HDR environment image
  //   [1] HDR_LUT_INDEX     GGX split-sum BRDF LUT
  //   [2] HDR_SHEEN_INDEX   Charlie sheen directional-albedo LUT (raster only)
  //   [3] HDR_OPAQUE_INDEX  opaque-pass color capture for screen-space transmission (raster only)
  // PARTIALLY_BOUND lets the path tracer leave slots [2]/[3] unbound.
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTexturesHdr, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              HDR_TEXTURE_COUNT, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  // Prefiltered HDR cubemaps: [0] diffuse Lambertian, [1] GGX glossy.
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTexturesCube,
                                              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  NVVK_CHECK(m_resources.descriptorBinding[0].createDescriptorSetLayout(
      m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, &m_resources.descriptorSetLayout[0]));
  NVVK_DBG_NAME(m_resources.descriptorSetLayout[0]);

  // Pool sizes are derived from descriptorBinding[0] (now SAMPLED_IMAGE + SAMPLER for the scene, plus
  // COMBINED_IMAGE_SAMPLER for the HDR/cube arrays). Each nvapp::ImTexture (viewport: 1, DLSS guides: 8,
  // OptiX denoised: 1 + margin) additionally uses one VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE descriptor.
  constexpr uint32_t                kImTextureMaxSets = 15;
  std::vector<VkDescriptorPoolSize> poolSize          = m_resources.descriptorBinding[0].calculatePoolSizes();
  poolSize.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kImTextureMaxSets});
  VkDescriptorPoolCreateInfo dpoolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  // allows descriptor sets to be updated after they have been bound to a command buffer
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // individual descriptor sets can be freed from the descriptor pool
      .maxSets       = kImTextureMaxSets + 1,                      // ImTexture sets + the scene texture set
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &dpoolInfo, nullptr, &m_resources.descriptorPool));
  NVVK_DBG_NAME(m_resources.descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_resources.descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_resources.descriptorSetLayout[0],
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_resources.descriptorSet));
  NVVK_DBG_NAME(m_resources.descriptorSet);


  // 1: Descriptor PUSH: top level acceleration structure and the output image
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eTlas,
                                              VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eOutImages, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10,
                                              VK_SHADER_STAGE_ALL);
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eOutDepth, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                              VK_SHADER_STAGE_ALL);

  NVVK_CHECK(m_resources.descriptorBinding[1].createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                                        &m_resources.descriptorSetLayout[1]));
  NVVK_DBG_NAME(m_resources.descriptorSetLayout[1]);
}

//--------------------------------------------------------------------------------------------------
// Load an environment map / scene by path.
//
// These back the --hdrfile / --scenefile parameters, which carry a callbackSuccess so that setting
// them does the load -- from the command line or nvpro_set_parameters alike
// (nvmcp invokes the callback on the application thread). Before that, writing either was accepted
// and did nothing, because both were only read once during start-up.
//
// Path resolution is delegated to createHDR / createScene: both call nvutils::findFile against
// nvsamples::getResourcesDirs() so bare resource-relative inputs (e.g. `--hdrfile studio.hdr`
// where studio.hdr ships under the sample's resources dir) work exactly like a cwd-relative or
// absolute path. Reporting success/failure is likewise the loaders' job: HdrIbl::isValid() flips
// off with a warning log when its input cannot be loaded -- so we no longer need a pre-flight
// std::filesystem::exists() gate here that would reject resource-relative inputs the loader would otherwise have found.
bool GltfRenderer::loadHdrEnvironment(const std::filesystem::path& filename)
{
  // Not attached yet: this is the start-up command-line parse, and main() performs that load
  // itself once the device exists.
  if(!m_app)
    return false;

  createHDR(filename);
  if(!m_resources.hdrIbl.isValid())
    return false;

  // Loading an environment the renderer is not told to sample would be another silent no-op.
  m_resources.settings.envSystem = shaderio::EnvSystem::eHdr;
  // A new HDR changes hdrIbl.getIntegral(), which defaultFireflyClamp() uses when envSystem is
  // eHdr. Refresh the path-tracer clamp in lock-step -- so a --hdrfile change from an MCP write
  // doesn't keep clamping against the previous HDR's integral.
  calibrateFireflyClamp();
  resetFrame();
  return true;
}

bool GltfRenderer::loadHdrEnvironment(EnvImage&& image, const std::filesystem::path& name)
{
  if(!m_app || !image.valid())
    return false;
  LOGI("HDR environment: %s (%ux%u)\n", name.empty() ? "(generated)" : nvutils::utf8FromPath(name).c_str(),
       image.size.width, image.size.height);
  uploadHdrEnvironment(std::move(image), name);
  if(!m_resources.hdrIbl.isValid())
    return false;
  m_resources.settings.envSystem = shaderio::EnvSystem::eHdr;
  calibrateFireflyClamp();
  resetFrame();
  return true;
}

nvutils::Bbox GltfRenderer::sceneBounds() const
{
  const SceneGpuView& view = m_resources.gpuView();
  if(!view.hasContent() || view.renderNodeCapacity() == 0)
    return {};
  return view.bounds();
}

bool GltfRenderer::loadSceneFile(const std::filesystem::path& filename)
{
  if(!m_app)
    return false;

  // Synchronous: a client needs the scene on the GPU before this returns. A failed load leaves an
  // empty scene rather than the previous one, and the caller in main.cpp names the file.
  vkQueueWaitIdle(m_app->getQueue(0).queue);
  m_loadPipeline.clear();
  cleanupScene(/*keepImagesForReuse=*/true);
  return createScene(filename);
}

void GltfRenderer::prefetchImage(const std::filesystem::path& diskPath, bool srgb)
{
  m_resources.imagePrefetch.prefetch(diskPath, srgb, m_maxTextureExtent.load());
}

void GltfRenderer::setRenderSystem(RenderingMode mode)
{
  if(m_resources.settings.renderSystem == mode)
    return;
  m_resources.settings.renderSystem = mode;
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Recompile the active renderer's shaders and show the result.
// SYNC NOTE: the recompile destroys live pipelines, so the queue must be drained first.
bool GltfRenderer::reloadShaders()
{
  vkQueueWaitIdle(m_app->getQueue(0).queue);
  const bool compiled = compileShaders();
  resetFrame();
  return compiled;
}

//--------------------------------------------------------------------------------------------------
// Recompile the shaders of the current renderer.
bool GltfRenderer::compileShaders()
{
  nvutils::ScopedTimer st(__FUNCTION__);
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
  {
    return m_pathTracer.reloadShader(m_resources);
  }
  return m_rasterizer.compileShader(m_resources, true);
}

//--------------------------------------------------------------------------------------------------
// Update the HDR images : add the 2D images to allTextures and the cube images to allTexturesCube
//
void GltfRenderer::updateHdrImages()
{
  const std::vector<nvvk::Image>& hdrPreconvolutedTextures = m_resources.hdrDome.getTextures();
  nvvk::WriteSetContainer         write{};
  VkWriteDescriptorSet hdrTextures = m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr,
                                                                                  m_resources.descriptorSet, HDR_IMAGE_INDEX, 1U);
  // Adding the HDR image (RGBA32F)
  write.append(hdrTextures, m_resources.hdrIbl.getHdrImage());
  // Add pre-integrated LUT BRDF
  hdrTextures.dstArrayElement = HDR_LUT_INDEX;
  write.append(hdrTextures, hdrPreconvolutedTextures[2]);

  // Adding cube images: diffuse, glossy
  VkWriteDescriptorSet hdrTexturesCube =
      m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesCube, m_resources.descriptorSet, 0, 2U);
  write.append(hdrTexturesCube, m_resources.hdrDome.getTextures().data());

  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Reset the frame counter
void GltfRenderer::resetFrame()
{
  m_resources.frameCount = -1;
}

//--------------------------------------------------------------------------------------------------
// Update the frame counter
// This is called every frame to update the frame counter or to reset it if the camera has changed
// The frame counter is used to limit the number of frames rendered
// If the frame counter is greater than the maximum number of frames, the rendering stops
// Returns true if the frame counter is less than the maximum number of frames
bool GltfRenderer::updateFrameCounter()
{

  static nvutils::CameraManipulator::Camera ref_camera{};

  const auto currentCamera = m_cameraManip->getCamera();

  if(ref_camera != currentCamera)
  {
    resetFrame();
    ref_camera = currentCamera;
  }

  if(m_resources.accumulationComplete())
  {
    return false;
  }
  // The rasterizer's image is complete after one frame: drawing it again changes nothing, and a
  // drawn-frame count that stands still is what lets the frame stream idle (frame_stream.cpp). A
  // dirty scene still draws (the caller ORs this with its dirty flags). Not with DLSS on, whose
  // super-resolution accumulates its jittered frames.
  if(m_resources.settings.renderSystem == RenderingMode::eRasterizer && m_resources.frameCount >= 1)
  {
#if defined(USE_DLSS)
    const bool dlssActive = m_rasterizer.getDlss() && m_rasterizer.getDlss()->isActive();
#else
    const bool dlssActive = false;
#endif
    if(!dlssActive)
      return false;
  }
  // The DLSS frames of a hand-off delay are no part of the accumulation: the count waits at 1 (frame
  // 0 was DLSS's) so the frame bound cannot end the still before it begins. They draw all the same.
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_resources.frameCount >= 1 && m_pathTracer.dlssHandoffDelayPending())
    return true;
  m_resources.frameCount++;
  return true;
}

//--------------------------------------------------------------------------------------------------
// Create or load the HDR environment map
// If the filename is empty, a default environment map (empty) is created, which allow the descriptor set to be updated
void GltfRenderer::createHDR(const std::filesystem::path& hdrFilename)
{
  // Load an HDR and create the important sampling acceleration structure
  std::filesystem::path filename;
  if(!hdrFilename.empty())
  {
    if(std::filesystem::exists(hdrFilename))
      filename = hdrFilename;
    else
      filename = nvutils::findFile(hdrFilename, nvsamples::getResourcesDirs(), false);
  }
  // Decode here rather than through HdrIbl's path overload, so .hdr and .exr take one route --
  // see env_image_loader.hpp for why EXR cannot live inside nvvk. An empty or failed decode hands
  // over an empty span, which is what produces the dummy environment at start-up.
  EnvImage envImage = loadEnvImage(filename);

  // Say which file this is, or that there is none. A request that does not resolve, or an image
  // that does not decode, silently becomes the dummy environment above -- and a dummy is easy to
  // mistake for a real one, because plenty of studio HDRs genuinely are flat and achromatic.
  // Without a line here the only way to tell them apart is to know what the file looks like.
  if(hdrFilename.empty())
  {
    LOGI("HDR environment: none (dummy)\n");
  }
  else if(envImage.pixels.empty())
  {
    LOGW("HDR environment: %s could not be %s; using the dummy environment\n",
         nvutils::utf8FromPath(hdrFilename).c_str(), filename.empty() ? "found" : "decoded");
  }
  else
  {
    LOGI("HDR environment: %s (%ux%u)\n", nvutils::utf8FromPath(filename).c_str(), envImage.size.width, envImage.size.height);
  }

  const std::filesystem::path source = envImage.pixels.empty() ? std::filesystem::path{} : filename;
  uploadHdrEnvironment(std::move(envImage), source);
}

//--------------------------------------------------------------------------------------------------
// Install an already-decoded lat-long as the HDR environment. The decode is the slow half of an
// HDR load; split out so a caller can run it on a worker and only pay for the upload here.
void GltfRenderer::uploadHdrEnvironment(EnvImage&& envImage, const std::filesystem::path& sourceFile)
{
  // An HDRI reload can happen mid-frame. Wait until
  // in-flight work finishes before tearing down env images the rasterizer/path
  // tracer may still be sampling.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
  nvvk::FrameUploader uploader;
  NVVK_CHECK(uploader.init({
      .allocator = &m_resources.allocator,
      .blockSize = 1024ull * 1024 * 1024,  // 8K RGBA32F HDR is 1 GiB
      .debugName = "hdrUpload",
  }));

  // An image with no file behind it (a generated studio) is kept, so leaving a baked sky for the
  // HDR path can put it back -- the reload there has nothing else to read. HdrIbl writes into the
  // pixels it is given, hence a copy taken before the upload.
  if(sourceFile.empty() && envImage.valid())
    m_lastHdrImage = std::make_shared<const EnvImage>(envImage);
  else
    m_lastHdrImage.reset();

  m_resources.hdrIbl.destroyEnvironment();
  m_resources.hdrIbl.loadEnvironment(cmd, uploader, std::span<float>(envImage.pixels), envImage.size, true);
  // Remembered so that switching to a baked sky and back to eHdr can rebind hdrIbl to this file:
  // EnvBaker::commit repoints hdrIbl at its own image, and only a reload undoes that. The resolved
  // path, not the request: a bare name found through the resources directories would otherwise be
  // saved as a panorama URI that resolves nowhere. Empty when nothing decoded -- the dummy is not a
  // panorama, and saving it as one would describe an environment that does not exist.
  m_lastHdrFile     = sourceFile;
  m_envOwnedByBaker = false;

  uploader.cmdUploadAppended(cmd);

  // Generate mipmaps for the HDR image
  VkExtent2D hdrSize = m_resources.hdrIbl.getHdrImageSize();
  if(hdrSize.width > 1 && hdrSize.height > 1)
  {
    nvvk::cmdGenerateMipmaps(cmd, m_resources.hdrIbl.getHdrImage().image, hdrSize, nvvk::mipLevels(hdrSize));
  }

  nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
  uploader.deinit();

  // Create the diffuse and glossy cube maps for the HDR image (raster)
  m_resources.hdrDome.create(m_resources.hdrIbl.getDescriptorSet(), m_resources.hdrIbl.getDescriptorSetLayout(),
                             std::span(hdr_prefilter_diffuse_slang), std::span(hdr_prefilter_glossy_slang),
                             std::span(hdr_integrate_brdf_slang), std::span(hdr_dome_slang));

  updateHdrImages();
  m_resources.hdrDome.setOutImage(m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgRendered));
  m_rasterizer.resetHdr();
  // addToRecentFiles(hdrFilename);
}

//--------------------------------------------------------------------------------------------------
// Destroy the resources
// Resource cleanup follows a specific order to prevent validation errors:
// 1. First flush any pending command buffers to ensure GPU work is complete
// 2. Then destroy higher-level objects before their dependencies
// 3. Finally clean up allocator after all resources using it are destroyed
// This ensures proper synchronization and prevents use-after-free errors
void GltfRenderer::destroyResources()
{

  m_loadPipeline.destroy();
  m_resources.retainedGpu.reset();  // while the allocator, the uploader and the sampler pool live
  m_resources.retainedSpare.reset();

  m_resources.allocator.destroyBuffer(m_resources.bFrameInfo);
  if(m_resources.bSelectionBitMask.buffer != VK_NULL_HANDLE)
    m_resources.allocator.destroyBuffer(m_resources.bSelectionBitMask);

  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[0], nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[1], nullptr);
  vkDestroyDescriptorPool(m_device, m_resources.descriptorPool, nullptr);
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);

  m_profilerGpuTimer.deinit();
  {
#ifdef USE_NVMCP
    // Automation (MCP) reads the timeline from a worker thread under this same lock, so take it
    // here: the release only happens once no read is in flight, and every later read sees nullptr.
    const std::lock_guard<std::mutex> lock(m_profilerTimelineMutex);
#endif
    g_profilerManager.destroyTimeline(m_profilerTimeline);
    m_profilerTimeline = nullptr;
  }
  m_silhouette.deinit(m_resources);
  m_skyBruneton.deinit(m_resources);
  m_envBaker.deinit(m_resources);

  m_resources.tonemapper.deinit();
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  m_resources.tonemappedUi.deinit();
  m_resources.gBuffers.deinit();
  m_resources.hdrIbl.deinit();
  m_resources.hdrDome.deinit();
  m_resources.samplerPool.deinit();
  m_resources.staging.deinit();
  m_rayPicker.deinit();
  m_resources.allocator.deinit();
}


//--------------------------------------------------------------------------------------------------
// Non-blocking poll of the async click ray-pick (see m_pendingClickPick's comment in renderer.hpp).
// Once the frame that recorded the pick has signaled, nvvk::RayPicker::getResult() is a plain
// host-mapped-buffer read (confirmed in nvvk/ray_picker.cpp - no internal wait of its own), so this
// is a cheap poll-then-memcpy.
//--------------------------------------------------------------------------------------------------
void GltfRenderer::updateClickPickState()
{
  if(!m_pendingClickResult || !m_pendingClickResult->semaphoreState.testSignaled(m_device))
    return;

  const bool isDoubleClick = m_pendingClickResult->isDoubleClick;
  m_pendingClickResult.reset();
  applyClickPickResult(m_rayPicker.getResult(), isDoubleClick);
}


