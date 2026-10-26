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

#pragma once

#include "settings_registry.hpp"
#include <memory>

#include <nvapp/application.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/resources.hpp>
#include <nvshaders_host/sky.hpp>

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device


#include "resources.hpp"
#include "renderer_base.hpp"

#if defined(USE_DLSS)
class Dlss;  // forward decl, defined in dlss.hpp; only declared under USE_DLSS so the unique_ptr below has a complete type when needed
#endif

class Rasterizer : public BaseRenderer
{
public:
  Rasterizer();
  ~Rasterizer() override;

  void onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler) override;
  void onDetach(Resources& resources) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources) override;
  bool onUIRender(Resources& resources) override;
  void onRender(VkCommandBuffer cmd, Resources& resources) override;
  void onSceneInvalidated(Resources& resources) override;
  void notifyDlssContentReset(Resources& resources) override;

  bool compileShader(Resources& resources, bool fromFile = true) override;
  void createPipeline(Resources& resources) override;
  void freeRecordCommandBuffer(Resources& resources);

  // Register command line parameters
  /// Declare this subsystem's settings (command line + benchmark + MCP + persistence).
  void registerParameters(SettingsRegistry* settings);

  void resetHdr() { m_lastHdrDomeView = {}; }

private:
  void renderNodes(VkCommandBuffer cmd, Resources& resources, const std::vector<uint32_t>& nodeIDs);
  void recordRasterScene(Resources& resources, VkExtent2D renderExtent);
  void renderRasterScene(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent);
  void renderOpaqueOnly(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent);
  void renderBlendOnly(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent);
  void createRecordCommandBuffer();

  // Allocate / destroy the opaque-pass color capture used for KHR_materials_transmission.
  void createOpaqueColorImage(Resources& resources);
  void destroyOpaqueColorImage(Resources& resources);
  void captureAndMipOpaqueColor(VkCommandBuffer cmd, Resources& resources);

  // Allocate and bake the sheen directional-albedo LUT
  void createSheenLut(Resources& resources);
  void destroySheenLut(Resources& resources);

  // Sort blend (transparent) nodes back-to-front against the current view matrix. Returns true
  // if the order changed since the last call (used to invalidate recorded command buffers).
  bool updateSortedBlendNodes(VkCommandBuffer cmd, Resources& resources);


  VkDevice         m_device{};                 // Vulkan device
  VkCommandBuffer  m_recordedSceneCmd{};       // Command buffer for recording the scene
  VkCommandPool    m_commandPool{};            // Command pool for recording the scene
  VkPipelineLayout m_graphicPipelineLayout{};  // The pipeline layout use with graphics pipeline

  nvvk::GraphicsPipelineState m_dynamicPipeline;  // Graphics pipeline state

  shaderio::RasterPushConstant m_pushConst{};  // Reusing the same push constant structure for now


  VkShaderEXT m_vertexShader{};     // Vertex shader
  VkShaderEXT m_fragmentShader{};   // Fragment shader
  VkShaderEXT m_wireframeShader{};  // Wireframe shader

  // Sun shadow map: a depth-only pass along the shadow light (SceneFrameInfo::shadowViewProj),
  // read by the light loop through set 2 -- raster-only, so the layout the path tracer shares with
  // set 0 is untouched. Alpha-tested entry points, so cutout hair and lashes cast cutout shadows.
  VkShaderEXT          m_shadowVertexShader{};
  VkShaderEXT          m_shadowFragmentShader{};
  nvvk::Image          m_shadowMap{};
  int                  m_shadowMapSize = 0;
  VkSampler            m_shadowCompareSampler{};
  VkSampler            m_shadowPointSampler{};
  nvvk::DescriptorPack m_shadowDescPack;
  void                 initShadowResources(Resources& resources);
  void                 deinitShadowResources(Resources& resources);
  // (Re)allocate the map at `size` texels, leaving it in SHADER_READ_ONLY. Records the transition into `cmd`.
  void createShadowMap(VkCommandBuffer cmd, Resources& resources, int size);
  void renderShadowMap(VkCommandBuffer cmd, Resources& resources);

  // Ground: a fullscreen pass that intersects each pixel's ray with the plane, drawn after the opaque
  // buckets. A floor shades and fades; a shadow catcher only darkens what is behind it.
  VkShaderEXT m_groundVertexShader{};
  VkShaderEXT m_groundFragmentShader{};
  bool        m_lastGroundOn = false;  // Re-records the scene cmd when the ground comes or goes
  void        renderGround(VkCommandBuffer cmd, Resources& resources);


  // Background pass for authored (OMI_environment_sky) sky types. Not nvshaders::SkyBase: that
  // template pushes its whole parameter block, and SkyOmiParameters plus a matrix overruns the
  // 128-byte push-constant guarantee. The parameters travel in SceneFrameInfo instead.
  VkPipeline               m_skyBackgroundPipeline{};
  VkPipelineLayout         m_skyBackgroundLayout{};
  VkDescriptorSetLayout    m_skyBackgroundDescLayout{};
  nvvk::DescriptorBindings m_skyBackgroundBindings;

  void initSkyBackground(Resources& resources);
  void deinitSkyBackground(Resources& resources);
  // Fills `targets.colorView` with the analytic background for the active authored sky.
  void drawSkyBackground(VkCommandBuffer             cmd,
                         shaderio::SkyBackgroundMode mode,
                         Resources&                  resources,
                         const VkExtent2D&           size,
                         VkImageView                 colorView,
                         const glm::mat4&            viewMatrix,
                         const glm::mat4&            projMatrix);

  std::vector<uint32_t> m_sortedBlendNodes;     // Transparent (blend) draw ordering. Sorted back-to-front
  glm::mat4             m_lastViewMatrix{0.f};  // Last view matrix used to sort blend nodes.
  uint64_t m_lastShadedNodesRevision = 0;  // Last scene graph revision at which we updated m_sortedBlendNodes. Used to detect when we need to re-sort.
  uint64_t m_lastSceneGraphRevision = 0;  // Last scene graph revision we recorded commands against. Used to detect when we need to re-record.

  // Opaque-pass color capture for screen-space transmission (KHR_materials_transmission).
  // Allocated lazily on first use; size matches OPAQUE_COLOR_SIZE in shaderio.h.
  nvvk::Image m_opaqueColorImage{};
  bool        m_sceneHasTransmission = false;  // computed each frame from scene materials

  // Baked sheen directional-albedo LUT (R16G16_SFLOAT, 64x64). Independent of the HDR env, so
  // this is created once at attach time.
  nvvk::Image m_sheenLut{};

  // UI
  bool m_useRecordedCmd = true;   // Use recorded command buffer for rendering
  bool m_lastWireframe  = false;  // Tracks settings.wireframe to invalidate recorded commands
  // Latched by registerParameters()'s CLI/benchmark callbacks (rasterUseRecordedCmd, dlssQuality)
  // and consumed at the top of onRender() to invalidate m_recordedSceneCmd. The UI paths still
  // call freeRecordCommandBuffer() inline; this flag closes the gap for non-UI setters.
  bool m_recordedSceneCmdDirty = false;
  // Extent the recorded cmd baked its viewport/scissor from. Compared in onRender() so any change
  // of active extent re-records, including the ones no callback sees: ImGui.ini restore and
  // Windows > Reset All to Default write setting storage directly (see settings_registry.hpp) and
  // can flip DLSS-SR between its inner and the outer extent under a live recording.
  VkExtent2D m_recordedSceneExtent{};

  // Set 3: the scene's TLAS, pushed with the draws, for the ray-traced shadows of the lights the
  // shadow map does not cover. The recorded commands bake the handle and the ray count, so a change
  // of either re-records (onRender).
  nvvk::DescriptorBindings   m_tlasBindings;
  VkDescriptorSetLayout      m_tlasLayout{};
  int                        m_rtShadowRays = 4;  // rasterShadowRays: rays per light, 0 none
  int                        m_aoRays       = 4;  // rasterAoRays: ambient occlusion rays per pixel, 0 none
  VkAccelerationStructureKHR m_recordedTlas{};
  int                        m_recordedRtShadowRays = -1;
  int                        m_recordedAoRays       = -1;

  // ---- DLSS-SR (DLAA / Quality / Balanced / Performance / Ultra) ----
#if defined(USE_DLSS)
public:
  Dlss*       getDlss() { return m_dlss.get(); }
  const Dlss* getDlss() const { return m_dlss.get(); }

private:
  std::unique_ptr<Dlss> m_dlss;
#endif
  VkImageView m_lastHdrDomeView = VK_NULL_HANDLE;  // Last image view we wrote into HdrEnvDome::setOutImage
};
