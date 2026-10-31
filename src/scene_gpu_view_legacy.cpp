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

// SceneGpuView over nvvkgltf::Scene, SceneVk and SceneRtx (scene_gpu_view.hpp).

#include "scene_gpu_view.hpp"

#include <algorithm>

#include "gltf_scene.hpp"
#include "gltf_scene_animation_vk.hpp"
#include "gltf_scene_rtx.hpp"
#include "gltf_scene_transform_vk.hpp"
#include "gltf_scene_vk.hpp"
#include "animation_control.hpp"

namespace {

nvvkgltf::Scene::PipelineType pipelineOf(SceneGpuView::DrawList list)
{
  switch(list)
  {
    case SceneGpuView::DrawList::eSolid:
      return nvvkgltf::Scene::eRasterSolid;
    case SceneGpuView::DrawList::eSolidDoubleSided:
      return nvvkgltf::Scene::eRasterSolidDoubleSided;
    case SceneGpuView::DrawList::eBlend:
      return nvvkgltf::Scene::eRasterBlend;
    case SceneGpuView::DrawList::eAll:
      break;
  }
  return nvvkgltf::Scene::eRasterAll;
}

}  // namespace

const nvvkgltf::Scene* LegacySceneGpuView::valid() const
{
  return m_scene && m_scene->valid() ? m_scene.get() : nullptr;
}

bool LegacySceneGpuView::hasContent() const
{
  return valid() != nullptr;
}

VkDeviceAddress LegacySceneGpuView::sceneDesc() const
{
  return m_sceneVk.sceneDesc().address;
}

VkAccelerationStructureKHR LegacySceneGpuView::tlas() const
{
  return m_sceneRtx.topLevelAS();
}

uint32_t LegacySceneGpuView::renderNodeCapacity() const
{
  const nvvkgltf::Scene* scene = valid();
  return scene ? uint32_t(scene->getRenderNodes().size()) : 0;
}

const std::vector<uint32_t>& LegacySceneGpuView::drawList(DrawList list) const
{
  static const std::vector<uint32_t> none;
  // the lists of a scene that is still loading are read too, as before (getShadedNodes needs no valid())
  return m_scene ? m_scene->getShadedNodes(pipelineOf(list)) : none;
}

uint64_t LegacySceneGpuView::drawListsRevision() const
{
  return m_scene ? m_scene->getShadedNodesRevision() : 0;
}

uint64_t LegacySceneGpuView::transformsRevision() const
{
  return m_scene ? m_scene->getSceneGraphRevision() : 0;
}

SceneGpuView::DrawRecord LegacySceneGpuView::drawRecord(uint32_t renderNode) const
{
  if(!m_scene || renderNode >= m_scene->getRenderNodes().size())
    return {};
  const nvvkgltf::RenderNode& rn   = m_scene->getRenderNodes()[renderNode];
  const size_t                prim = size_t(rn.renderPrimID);
  if(rn.renderPrimID < 0 || prim >= m_scene->getRenderPrimitives().size() || prim >= m_sceneVk.vertexBuffers().size()
     || prim >= m_sceneVk.indices().size())
    return {};
  return {.vertexBuffer = m_sceneVk.vertexBuffers()[prim].position.buffer,
          .indexBuffer  = m_sceneVk.indices()[prim].buffer,
          .indexCount   = uint32_t(m_scene->getRenderPrimitives()[prim].indexCount),
          .materialID   = rn.materialID,
          .renderPrimID = rn.renderPrimID,
          .visible      = rn.visible};
}

std::optional<glm::vec3> LegacySceneGpuView::worldCentre(uint32_t renderNode) const
{
  if(!m_scene || renderNode >= m_scene->getRenderNodes().size())
    return std::nullopt;
  const nvvkgltf::RenderNode&   rn      = m_scene->getRenderNodes()[renderNode];
  const std::vector<glm::vec3>& centres = m_scene->getRenderPrimCenterObj();
  if(rn.renderPrimID < 0 || size_t(rn.renderPrimID) >= centres.size())
    return std::nullopt;
  return glm::vec3(rn.worldMatrix * glm::vec4(centres[size_t(rn.renderPrimID)], 1.f));
}

bool LegacySceneGpuView::hasTransmission() const
{
  return m_scene && m_scene->hasTransmissionMaterials();
}

bool LegacySceneGpuView::hasOpacityMicromaps() const
{
  return m_sceneVk.opacityMicromap().hasAny();
}

bool LegacySceneGpuView::hasPlayableAnimation() const
{
  return ui::animation::hasPlayableAnimation(m_scene.get());
}

std::vector<SceneGpuView::LightInfo> LegacySceneGpuView::lights() const
{
  std::vector<LightInfo> out;
  const nvvkgltf::Scene* scene = valid();
  if(!scene)
    return out;
  const tinygltf::Model& model = scene->getModel();
  for(const nvvkgltf::RenderLight& rl : scene->getRenderLights())
  {
    LightInfo info;
    info.worldMatrix = rl.worldMatrix;
    if(rl.light >= 0 && rl.light < int(model.lights.size()))
    {
      const tinygltf::Light& l = model.lights[size_t(rl.light)];
      info.directional         = l.type == "directional";
      info.intensity           = l.intensity;
      if(l.color.size() == 3)
        info.maxColor = std::max({l.color[0], l.color[1], l.color[2]});
    }
    out.push_back(info);
  }
  return out;
}

nvutils::Bbox LegacySceneGpuView::bounds() const
{
  const nvvkgltf::Scene* scene = valid();
  return scene ? scene->getSceneBounds() : nvutils::Bbox{};
}

SceneGpuView::MemoryStats LegacySceneGpuView::memoryStats() const
{
  MemoryStats out;
  for(const nvvkgltf::GpuMemoryTracker* tracker : {&m_sceneVk.getMemoryTracker(), &m_sceneRtx.getMemoryTracker(),
                                                   &m_transformCompute.getMemoryTracker(), &m_animationVk.getMemoryTracker()})
  {
    out.usedBytes += tracker->getTotalStats().currentBytes;
    out.peakBytes += tracker->getTotalStats().peakBytes;
  }
  return out;
}
