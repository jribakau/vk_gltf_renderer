/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include <vector>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/uploader_interface.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_vk.hpp"
#include "gpu_memory_tracker.hpp"
#include "shaders/animation_io.h.slang"

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::AnimationVk

>  GPU-accelerated skeletal skinning and morph target blending via compute shaders.

Manages two compute pipelines (skinning + morph) and the GPU SSBOs that feed them.
Static data (base geometry, skin weights, joint indices, IBMs, morph deltas) is uploaded
once at scene load via `createGpuBuffers()`. Each frame, `dispatchAnimation()` computes
joint matrices on CPU, uploads the small per-frame data (joint matrices, morph weights),
and dispatches compute shaders that write transformed vertices directly into SceneVk's
existing vertex buffers via buffer device addresses (BDA push constants).

The CPU fallback path in `SceneVk::uploadPrimitives()` is preserved and selectable via
`SceneGpu::useComputeAnimation`.

 -------------------------------------------------------------------------------------------------*/
namespace nvvkgltf {

// The skinning and morph compute pipelines (push constants only, shaders/animation_io.h.slang), for
// AnimationVk and the retained scene's Vulkan backend
struct DeformPipelines
{
  VkPipeline       skin{}, morph{};
  VkPipelineLayout skinLayout{}, morphLayout{};
  void             create(VkDevice device);
  void             destroy(VkDevice device);
};

class AnimationVk
{
public:
  AnimationVk() = default;
  ~AnimationVk() { assert(!m_alloc); }

  // --- Lifecycle ---
  void init(nvvk::ResourceAllocator* alloc);  // Create compute pipelines
  void deinit();                              // Destroy everything

  // --- Scene data (call once per scene load/rebuild) ---
  void createGpuBuffers(nvvk::CmdUploaderInterface& staging, const Scene& scn);  // Upload static SSBOs
  void destroyGpuBuffers();                                                      // Release static + workspace SSBOs

  // --- Live edits ---
  // Replace the static base geometry a skinned or morphed primitive is computed from (its SceneVk
  // vertex buffers are compute outputs, rewritten from these on every dispatch). Sizes must match
  // the buffers createGpuBuffers() made. Returns false when the primitive is neither skinned nor
  // morphed; the caller then uploads into the SceneVk buffers directly.
  // `runs`: the [first, count) vertex runs that changed, staged alone; empty means the whole buffer
  bool updateBaseGeometry(nvvk::CmdUploaderInterface&                    staging,
                          const Scene&                                   scn,
                          int                                            renderPrimID,
                          std::span<const glm::vec3>                     positions,
                          std::span<const glm::vec3>                     normals,
                          std::span<const std::pair<uint32_t, uint32_t>> runs = {});

  // --- Per-frame animation (call each frame when animation is active) ---
  void dispatchAnimation(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, Scene& scn, const SceneVk& scnVk);

  [[nodiscard]] bool isInitialized() const { return m_alloc != nullptr && m_pipelines.skin != VK_NULL_HANDLE; }

  // --- What the last dispatch wrote ---
  // A task whose joint matrices (or morph weights) and base geometry are what they were when it last
  // ran is skipped: its vertex buffers already hold the result. This lists the render primitives the
  // last dispatchAnimation() wrote, every member of each set it skinned or morphed, sorted and
  // unique: the BLASes to refit.
  const std::vector<uint32_t>& primitivesWritten() const { return m_written; }
  // Every task runs again on the next dispatch: after something else rewrote the vertex buffers.
  void invalidateOutputs();

  const GpuMemoryTracker& getMemoryTracker() const { return m_memoryTracker; }
  GpuMemoryTracker&       getMemoryTracker() { return m_memoryTracker; }

private:
  nvvk::ResourceAllocator* m_alloc = nullptr;

  DeformPipelines m_pipelines;

  // Per-SkinTask GPU buffers (static, uploaded once)
  struct SkinGpuData
  {
    nvvk::Buffer basePositions;
    nvvk::Buffer baseNormals;
    nvvk::Buffer baseTangents;
    nvvk::Buffer weights;
    nvvk::Buffer joints;
    nvvk::Buffer inverseBindMatrices;
    size_t       jointMatOffset  = 0;  // Byte offset into packed m_jointMatricesBuffer
    size_t       normalMatOffset = 0;  // Byte offset into packed m_normalMatricesBuffer
  };
  std::vector<SkinGpuData> m_skinGpuData;

  // Per-MorphResult GPU buffers (static, uploaded once)
  struct MorphGpuData
  {
    nvvk::Buffer basePositions;
    nvvk::Buffer baseNormals;
    nvvk::Buffer baseTangents;
    nvvk::Buffer positionDeltas;  // all targets packed: [numTargets * vertexCount]
    nvvk::Buffer normalDeltas;
    nvvk::Buffer tangentDeltas;
    uint32_t     numTargets    = 0;
    size_t       weightsOffset = 0;  // Byte offset into packed m_morphWeightsBuffer
  };
  std::vector<MorphGpuData> m_morphGpuData;

  // Packed per-frame buffers: all tasks' matrices/weights in one buffer, uploaded in a single transfer.
  // Each task's data is at its stored byte offset. Allocated once at createGpuBuffers() to the total size.
  nvvk::Buffer m_jointMatricesBuffer;
  nvvk::Buffer m_normalMatricesBuffer;
  nvvk::Buffer m_morphWeightsBuffer;

  GpuMemoryTracker m_memoryTracker;

  // The inputs each task last ran with (empty: it must run), and what the last dispatch wrote
  std::vector<std::vector<glm::mat4>> m_lastJointMatrices;  // per skin task
  std::vector<std::vector<float>>     m_lastMorphWeights;   // per morph task
  std::vector<uint32_t>               m_written;


  template <typename T>
  nvvk::Buffer createBufferFromSpan(nvvk::CmdUploaderInterface& staging, std::span<const T> data, const char* debugName = nullptr);

  nvvk::Buffer createBufferFromData(nvvk::CmdUploaderInterface& staging, const void* data, size_t byteSize, const char* debugName = nullptr);
};

}  // namespace nvvkgltf
