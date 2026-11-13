/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <vector>

#include <glm/glm.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/resource_allocator.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"
#include "shaders/rain_io.h.slang"

//-------------------------------------------------------------------------------------------------------
// Rain
//
// The drops falling in the air (shaders/rain.slang; the physics in shaders/rain.h.slang): at a rate
// in mm/h, Marshall and Palmer's sizes falling at their own speeds, slanted by the wind, each drawn
// as the streak its fall makes over the camera's shutter, lit by the environment round it and the
// sun behind it. Drawn over the image the tone mapper takes, after DLSS and the denoisers, so a still
// keeps them sharp and the moving view does not smear them; before the lens and the bloom.
//
//   rain.cmdApply(cmd, resources, input, size, view)  // in runTonemapPass, first
//
// Each frame it traces a shelter map from above (shaders/rain_shelter.slang), so no rain falls under a
// roof, a tree or a figure, and hides a drop behind the scene by the path tracer's depth. Drops from
// minDistance to `reach` metres are drawn; past that their streaks are under a pixel, the air's haze.
// While `falling`, its clock runs and every frame is presented anew (version()).
//-------------------------------------------------------------------------------------------------------
class Rain
{
public:
  struct Settings
  {
    float rate     = 0.0f;         // mm/h, 0 none
    float wind     = 0.0f;         // m/s at the ground
    float windFrom = 0.0f;         // degrees, an azimuth as the sun's (+z at 0, +x at 90)
    float shutter  = 1.0f / 60.0f;  // seconds the camera's shutter is open
    bool  falling  = true;         // the drops fall; off, they hold where they are
    float reach    = 15.0f;        // metres out to which drops are drawn
  };

  // What the pass needs of the frame: the scene camera's matrices and the depth the path tracer wrote
  struct View
  {
    glm::mat4  view{1.0f};
    glm::mat4  proj{1.0f};
    VkExtent2D depthExtent{};
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  // `wetChanged`: a setting the path tracer sees changed (the rate, the wind: where it rains on what)
  void registerParameters(SettingsRegistry* settings, std::function<void()> wetChanged);
  void onResize(VkCommandBuffer cmd, VkExtent2D size);

  bool active() const { return m_settings.rate > 0.0f; }
  // Advanced on every frame the drops fall, so the presenter shows each one
  uint64_t version() const { return m_version; }

  // Before the path tracer, every frame while it rains: the frame planned (the clock, the cells the
  // camera sees) and the shelter map traced, which the path tracer reads for where it rains on what
  // (frame info's rainShelter) and the drops for where they fall
  void cmdPrepare(VkCommandBuffer cmd, Resources& resources, VkExtent2D size, const View& view);
  // This frame's, once prepared: how wet it leaves an open surface (0 to 1: a drizzle wets, harder
  // rain soaks), the share of flat ground under puddles, the drops' slant, and the shelter map
  bool      prepared() const { return m_prepared; }
  float     rate() const { return active() ? m_settings.rate : 0.0f; }
  float     time() const { return m_time; }
  float     wetness() const;
  float     puddles() const;
  glm::vec2 shear() const { return m_planShear; }
  glm::vec2 shelterCentre() const { return m_planCentre; }
  VkDeviceAddress shelterAddress() const { return m_prepared ? m_shelter.address : 0; }

  // The tone mapper's input with the rain drawn over it, sampled linearly; `input` itself when off.
  // `timer`: each pass in a section of its own (the wait for the image, the shelter map, the light
  // table, the copy, the drops)
  VkDescriptorImageInfo cmdApply(VkCommandBuffer cmd, Resources& resources, const VkDescriptorImageInfo& input, VkExtent2D size,
                                 const View& view, nvvk::ProfilerGpuTimer& timer);

  static constexpr float    kCellSize    = 1.0f;     // metres
  static constexpr float    kMinDistance = 0.3f;     // metres: nearer drops are out of focus and would fill the frame
  static constexpr size_t   kMaxCells    = 16384;    // cells a frame at most
  static constexpr uint32_t kMaxSlots    = 4096;     // drops a cell at most
  static constexpr float    kLongestFrame = 0.1f;    // seconds a frame's fall counts for at most
  static constexpr float    kClockWrap    = 600.0f;  // the clock goes round, so a float holds it finely
  static constexpr float    kMinCoverage  = 0.04f;   // a fainter drop is drawn at a chance (rainCull)
  static constexpr float    kSplashReach  = 8.0f;    // metres out to which splashes are drawn
  static constexpr uint32_t kMaxSplashSlots = 64;     // splashes a cell at once at most

private:
  bool ensureTarget(VkCommandBuffer cmd, VkExtent2D size);
  void releaseTarget();
  void createPipelines(Resources& resources);

  Settings m_settings{};
  float    m_time    = 0.0f;
  // this frame's plan (cmdPrepare)
  bool            m_prepared     = false;
  VkDeviceAddress m_frameAddress = 0;
  uint32_t        m_slots        = 0;
  uint32_t        m_cellCount    = 0;
  glm::vec2       m_planShear{0.0f};
  glm::vec2       m_planCentre{0.0f};
  uint64_t m_version = 0;
  std::chrono::steady_clock::time_point m_lastFrame{};

  VkDevice                    m_device           = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*    m_alloc            = nullptr;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvapp::Application*         m_app              = nullptr;
  VkExtent2D                  m_size{};
  nvvk::RenderTarget          m_out;

  // the frame's RainFrame and its cells, one buffer a frame in flight (host-written)
  std::vector<nvvk::Buffer> m_frames;
  nvvk::Buffer              m_light;    // the light table (GPU-written)
  nvvk::Buffer              m_shelter;  // the shelter map (GPU-written)
  nvvk::Buffer              m_streaks;   // the drops drawn (rainCull's)
  nvvk::Buffer              m_drawArgs;  // their draw's VkDrawIndirectCommand
  VkPipeline                m_cullPipeline{};
  VkPipeline                m_splashPipeline{};
  uint32_t                  m_splashSlots = 0;

  nvvk::DescriptorBindings    m_bindings;
  VkDescriptorSetLayout       m_setLayout{};
  VkPipelineLayout            m_pipelineLayout{};
  VkPipeline                  m_lightPipeline{};
  VkPipeline                  m_copyPipeline{};
  VkPipeline                  m_shelterPipeline{};
  VkShaderEXT                 m_vertexShader   = VK_NULL_HANDLE;
  VkShaderEXT                 m_fragmentShader = VK_NULL_HANDLE;
  nvvk::GraphicsPipelineState m_drawState;
};
