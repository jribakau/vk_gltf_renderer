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

#include <algorithm>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <nvapp/application.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/resource_allocator.hpp>

#include "shaders/gizmo_lines_shaderio.h.slang"

//-----------------------------------------------------------------------------
// LineHelperVk - world-space line segments over the image: what a host draws for things that
// have no mesh, a camera's frustum, a light's marker.
//-----------------------------------------------------------------------------
// - The host hands over the segments (two vertices each) whenever they change; each frame copies
//   them into a host-visible buffer of its own (one per frame in flight), which the vertex shader
//   reads by vertex index through its device address.
// - Shader objects and dynamic state, as GridHelperVk; alpha-over blending, no depth writes.
// - Depth-tested against the scene depth in the shader: behind geometry a line is faint, not gone.
//-----------------------------------------------------------------------------

class LineHelperVk
{
public:
  using Vertex = shaderio::lines::LineVertex;

  LineHelperVk() = default;
  ~LineHelperVk() { deinit(); }

  LineHelperVk(const LineHelperVk&)            = delete;
  LineHelperVk& operator=(const LineHelperVk&) = delete;

  struct Resources
  {
    nvapp::Application*      app                       = nullptr;
    nvvk::ResourceAllocator* alloc                     = nullptr;
    VkDevice                 device                    = VK_NULL_HANDLE;
    VkDescriptorSetLayout    helperDescriptorSetLayout = VK_NULL_HANDLE;  // Set 0: scene depth
  };

  void init(const Resources& res);
  void deinit();

  // The segments to draw, two vertices each; empty draws nothing.
  void setLines(std::vector<Vertex> vertices) { m_vertices = std::move(vertices); }
  bool hasLines() const { return !m_vertices.empty(); }

  // An RGBA colour, 0..1, as a vertex holds it.
  static uint32_t pack(const glm::vec4& c)
  {
    const auto byte = [](float v) { return uint32_t(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); };
    return byte(c.x) | byte(c.y) << 8 | byte(c.z) << 16 | byte(c.w) << 24;
  }

  void renderRaster(VkCommandBuffer  cmd,
                    VkDescriptorSet  helperDescriptorSet,
                    const glm::mat4& viewProj,
                    const glm::mat4& sceneViewProj,
                    bool             isPerspective,
                    const glm::vec2& viewportSize,
                    const glm::vec2& depthBufferSize);

private:
  void createPipeline();

  nvapp::Application*      m_app    = nullptr;
  nvvk::ResourceAllocator* m_alloc  = nullptr;
  VkDevice                 m_device = VK_NULL_HANDLE;

  VkPipelineLayout      m_pipelineLayout            = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_helperDescriptorSetLayout = VK_NULL_HANDLE;
  VkShaderEXT           m_vertexShader              = VK_NULL_HANDLE;
  VkShaderEXT           m_fragmentShader            = VK_NULL_HANDLE;

  nvvk::GraphicsPipelineState m_dynamicPipeline;

  std::vector<Vertex>       m_vertices;
  std::vector<nvvk::Buffer> m_buffers;   // one per slot of the app's frame ring
  std::vector<size_t>       m_capacity;  // vertices each holds
  // a buffer that grew: the old one is freed once every frame that could read it has finished
  struct Retired
  {
    nvvk::Buffer buffer;
    uint64_t     frame = 0;
  };
  std::vector<Retired> m_retired;
  uint64_t             m_frames = 0;  // renders so far
  size_t               m_frame  = 0;
  bool                      m_initialized = false;
};
