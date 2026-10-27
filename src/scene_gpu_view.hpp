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

//
// What the renderers read of the scene on the GPU (docs: daz-viewer docs/retained-scene-plan.md,
// "Coexistence"): the path tracer, the rasterizer, the silhouette, the pickers and the frame's
// shadow and ground fit read this, never nvvkgltf::Scene, SceneVk or SceneRtx, so the retained
// backend (P3a) can stand behind it. LegacySceneGpuView is today's objects behind it.
//
// Per-slot side tables: every GPU array indexed by a render node, render primitive or material
// slot. A new one belongs here, so the retained residency layer sizes, grows, fills and retires it
// with the rest (capacity and holes; see "Slots"):
//   render node   GltfRenderNode[] (SceneVk m_bRenderNode); RenderNodeGpuMapping
//                 (TransformComputeVk m_bRenderNodeMappings); gpuInstLocalMatrices
//                 (m_bGpuInstLocalMatrices); prevRenderNodeObjectToWorld (m_bPrevRenderNodeO2W);
//                 TLAS instance rows (SceneRtx m_tlasInstances, m_instancesBuffer) and their flags
//                 (m_instanceFlagsCache); the selection bitmask (Resources bSelectionBitMask)
//   render prim   GltfRenderPrimitive[] (SceneVk m_bRenderPrim); vertex streams and the index
//                 buffer (m_vertexBuffers, m_bIndices); its BLAS (SceneRtx m_blasAccel,
//                 m_blasBuildData); OMM index buffers (SceneOmm m_indexBuffers); previous
//                 positions and their address table (PrevPositionsVk m_prev, m_table; retained:
//                 prevPositionTable())
//   material      GltfShadeMaterial[] (SceneVk m_bMaterial) and its span of GltfTextureInfo
//                 (m_bTextureInfos)
//   scene-wide    lights (m_bLights) and IES profiles (m_bIesProfiles), dense; the emissive
//                 triangle table (m_bEmissiveTriangles), derived from render node and prim slots;
//                 joint, normal-matrix and morph-weight packs (AnimationVk)
//

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include <nvutils/bounding_box.hpp>

namespace nvvkgltf {
class Scene;
class SceneVk;
class SceneRtx;
class AnimationVk;
class TransformComputeVk;
}  // namespace nvvkgltf

class SceneGpuView
{
public:
  // The raster draw lists, by how the material draws
  enum class DrawList
  {
    eSolid,
    eSolidDoubleSided,
    eBlend,
    eAll,
  };

  // What one render node draws. The offsets are 0 while every stream has its own buffer; a draw
  // binds them all the same, so sub-allocated streams change no raster code.
  struct DrawRecord
  {
    VkBuffer     vertexBuffer = VK_NULL_HANDLE;  // positions
    VkDeviceSize vertexOffset = 0;
    VkBuffer     indexBuffer  = VK_NULL_HANDLE;  // uint32
    VkDeviceSize indexOffset  = 0;
    uint32_t     indexCount   = 0;
    int32_t      materialID   = -1;
    int32_t      renderPrimID = -1;
    bool         visible      = false;
  };

  // A light as the shadow fit picks one
  struct LightInfo
  {
    bool      directional = false;
    double    intensity   = 0.0;
    double    maxColor    = 1.0;  // the largest colour channel
    glm::mat4 worldMatrix{1.0f};
  };

  struct MemoryStats
  {
    uint64_t usedBytes = 0;
    uint64_t peakBytes = 0;
  };

  virtual ~SceneGpuView() = default;

  // A scene is loaded and valid; it may still draw nothing (see renderNodeCapacity)
  virtual bool hasContent() const = 0;
  // GltfScene's device address, what every shader reads the scene through; 0 without a scene
  virtual VkDeviceAddress sceneDesc() const = 0;
  // The TLAS; a new handle means it was recreated
  virtual VkAccelerationStructureKHR tlas() const = 0;
  // Render node ids are below this; 0 without a scene
  virtual uint32_t renderNodeCapacity() const = 0;

  // The render nodes of a raster list; its membership changes with drawListsRevision()
  virtual const std::vector<uint32_t>& drawList(DrawList list) const = 0;
  virtual uint64_t                     drawListsRevision() const     = 0;
  // Bumped when a world matrix may have moved or the scene graph changed
  virtual uint64_t   transformsRevision() const            = 0;
  virtual DrawRecord drawRecord(uint32_t renderNode) const = 0;
  // A render node's primitive centre in world space, for sorting blended draws; none for an id
  // that names no primitive
  virtual std::optional<glm::vec3> worldCentre(uint32_t renderNode) const = 0;

  // Material features
  virtual bool hasTransmission() const      = 0;
  virtual bool hasOpacityMicromaps() const  = 0;
  virtual bool hasPlayableAnimation() const = 0;
  // Whether a variant that only adds scene features may be built off the frame (D15): the view
  // holds the material edits that need them and draws nothing the live variant lacks (retained);
  // never on legacy, whose edits draw at once
  virtual bool variantMayBuildOffFrame() const = 0;
  // The render lights, in the light buffer's order
  virtual std::vector<LightInfo> lights() const = 0;

  // The bounds of what draws; empty without a scene. (The legacy scene gives a unit box around the
  // origin when nothing in it has bounds, so a caller that must skip an empty scene checks
  // renderNodeCapacity() too.)
  virtual nvutils::Bbox bounds() const = 0;
  // The scene's GPU memory: geometry, materials, textures, acceleration structures, animation
  virtual MemoryStats memoryStats() const = 0;

  // #DLSS motion the view keeps itself: per render primitive the device address of its positions as
  // the last frame drew them (0: none), and per render node the objectToWorld it was last drawn
  // with. 0 when the view keeps none: the legacy scene's are PrevPositionsVk and TransformComputeVk,
  // which the renderer drives frame by frame.
  virtual VkDeviceAddress prevPositionTable() const { return 0; }
  virtual VkDeviceAddress prevObjectToWorldTable() const { return 0; }
};

// Today's objects behind the view: nvvkgltf::Scene on the CPU, SceneVk, SceneRtx, AnimationVk and
// TransformComputeVk on the GPU.
class LegacySceneGpuView : public SceneGpuView
{
public:
  LegacySceneGpuView(const std::unique_ptr<nvvkgltf::Scene>& scene,
                     const nvvkgltf::SceneVk&                sceneVk,
                     const nvvkgltf::SceneRtx&               sceneRtx,
                     const nvvkgltf::AnimationVk&            animationVk,
                     const nvvkgltf::TransformComputeVk&     transformCompute)
      : m_scene(scene)
      , m_sceneVk(sceneVk)
      , m_sceneRtx(sceneRtx)
      , m_animationVk(animationVk)
      , m_transformCompute(transformCompute)
  {
  }

  bool                         hasContent() const override;
  VkDeviceAddress              sceneDesc() const override;
  VkAccelerationStructureKHR   tlas() const override;
  uint32_t                     renderNodeCapacity() const override;
  const std::vector<uint32_t>& drawList(DrawList list) const override;
  uint64_t                     drawListsRevision() const override;
  uint64_t                     transformsRevision() const override;
  DrawRecord                   drawRecord(uint32_t renderNode) const override;
  std::optional<glm::vec3>     worldCentre(uint32_t renderNode) const override;
  bool                         hasTransmission() const override;
  bool                         hasOpacityMicromaps() const override;
  bool                         variantMayBuildOffFrame() const override { return false; }
  bool                         hasPlayableAnimation() const override;
  std::vector<LightInfo>       lights() const override;
  nvutils::Bbox                bounds() const override;
  MemoryStats                  memoryStats() const override;

private:
  const nvvkgltf::Scene* valid() const;

  const std::unique_ptr<nvvkgltf::Scene>& m_scene;
  const nvvkgltf::SceneVk&                m_sceneVk;
  const nvvkgltf::SceneRtx&               m_sceneRtx;
  const nvvkgltf::AnimationVk&            m_animationVk;
  const nvvkgltf::TransformComputeVk&     m_transformCompute;
};
