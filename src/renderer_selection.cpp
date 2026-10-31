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

//
// What the host calls on the renderer for picks, selection, framing and the gizmo, and what a
// completed click pick reaches: moved out of ui_renderer.cpp, which belongs to the frozen windowed
// editor, so the Figure Viewer path does not run through it (daz-viewer docs/fork-boundary.md, D1).
//

#include <nvutils/bounding_box.hpp>
#include <nvutils/logger.hpp>

#include "renderer.hpp"
#include "scene_selection_render_nodes.hpp"
#include "tinygltf_utils.hpp"

// Applies a completed async ray-pick result (see m_pendingClickPick's comment in renderer.hpp for
// the full pipeline) - selection, and on a double click, camera recenter. Exactly the logic that
// used to run inline in mouseClickedInViewport() right after its now-removed blocking GPU wait.
void GltfRenderer::applyClickPickResult(const nvvk::RayPicker::PickResult& pickResult, bool isDoubleClick)
{
  // Environment was picked (no hit)
  const bool hasHit = pickResult.instanceID >= 0;
  // KHR_interactivity event/onSelect's selectionPoint/selectionRayOrigin - real ray-pick hit
  // point/origin (global space) when this click actually hit something, so a bound onSelect
  // handler sees finite values instead of the "no ray info" NaN fallback (KHR_node_selectability
  // README ~110 spec-sanctions NaN when there's no ray, but this IS a ray, so it must be used).
  glm::vec3 worldPos = hasHit ? pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT : glm::vec3(0);

  // Set or de-select the selected object
  // pickResult.instanceID is the TLAS instance index: a render node, or a stream host's instance slot
  if(!isDoubleClick && m_resources.hostFed)
  {
    std::optional<InstancePick> pick;
    if(hasHit && m_resources.retainedGpu)
      if(const rs::Residency::Instance* inst = m_resources.retainedGpu->residency().instance(uint32_t(pickResult.instanceID)))
      {
        pick = InstancePick{inst->entity, inst->material};
        // a prim that draws merged groups (Submesh::mergedInto): the material of the group hit
        const rs::RetainedScene& scene  = hostScene();
        const rs::MeshRecord*    record = scene.mesh(inst->mesh);
        const rs::Entity*        entity = scene.entity(inst->entity);
        if(record && entity && entity->mesh && inst->submesh < record->submeshes.size())
        {
          const uint32_t hit = record->submeshAt(inst->submesh, uint32_t(std::max(pickResult.primitiveID, 0)));
          for(const rs::MeshPart& part : entity->mesh->parts)
            if(part.mesh == inst->mesh && hit != inst->submesh && record->submeshes[hit].material < part.materials.size())
              pick->material = part.materials[record->submeshes[hit].material];
        }
      }
    // the entity hit, or nothing; the one selected, hit again, is let go
    if(!(m_pickObserver && m_pickObserver(pick, worldPos)))
      selectEntity(pick && pick->entity != m_selectedEntity ? pick->entity : rs::EntityId{});
  }
  else if(!isDoubleClick)
  {
    updateSelectionFromPick(pickResult.instanceID, worldPos, pickResult.worldRayOrigin);
  }

  if(!hasHit)
  {
    // a stream host's double-click on nothing frames the whole scene
    if(isDoubleClick && m_resources.hostFed)
      fitCamera(true);
    return;
  }
  if(isDoubleClick)
  {
    // Set the camera CENTER to the hit position
    glm::dvec3 eye, center, up;
    m_cameraManip->getLookat(eye, center, up);
    m_cameraManip->setLookat(eye, worldPos, up, false);  // Nice with CameraManip.updateAnim();
    m_cameraManip->setSpeed(pickResult.hitT);            // Re-adjust speed based on the new distance
  }

  if(nvvkgltf::Scene* scene = m_resources.getScene())
  {
    const int   renderNodeIdx = pickResult.instanceID;
    const auto& renderNodes   = scene->getRenderNodes();
    if(renderNodeIdx >= 0 && renderNodeIdx < static_cast<int>(renderNodes.size()))
    {
      const nvvkgltf::RenderNode& renderNode = renderNodes[renderNodeIdx];
      const tinygltf::Node&       node       = scene->getModel().nodes[renderNode.refNodeID];
      LOGI("Node Name: %s\n", node.name.c_str());
      LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pickResult.primitiveID);
      LOGI(" - Render: renderNode: %d, RenderPrim: %d\n", renderNodeIdx, pickResult.instanceCustomIndex);
      LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", worldPos.x, worldPos.y, worldPos.z, pickResult.hitT);
    }
  }
}

// SceneSelection's events into Resources, for the silhouette and Fit, and into KHR_interactivity's
// onSelect. Installed once in onAttach, so a selection fills selectedRenderNodes whether or not the
// UI pass runs (a stream host skips it).
void GltfRenderer::onSceneSelectionEvent(const SceneSelection::Event& event)
{
  switch(event.type)
  {
    case SceneSelection::EventType::CameraApply:
      applyGltfCamera(event.data);
      break;
    case SceneSelection::EventType::CameraSetFromView:
      setGltfCameraFromView(event.data);
      break;
    case SceneSelection::EventType::NodeSelected: {
      m_resources.selectedRenderNodes.clear();
      nvvkgltf::Scene* scn = m_resources.getScene();
      if(scn && scn->valid())
        m_resources.selectedRenderNodes = renderNodesOfNode(*scn, event.data);
      break;
    }
    case SceneSelection::EventType::PrimitiveSelected: {
      m_resources.selectedRenderNodes = {event.renderNodeIndex};
      break;
    }
    case SceneSelection::EventType::MaterialSelected:
      // Material selection event
      break;
  }
}

// The Scene Browser scrolls to the selection in windowed sessions; elsewhere there is no browser to show it.
void GltfRenderer::focusSelectionInBrowser()
{
  if(m_focusSelectionInBrowser)
    m_focusSelectionInBrowser();
}

// Update the current selection from a ray-pick result. `renderNodeIdx` is the TLAS instance index
// (== render node index), or < 0 when the environment/nothing was hit. `selectionPoint`/
// `selectionRayOrigin` (global space) are threaded through to SceneSelection::selectPrimitive() for
// KHR_interactivity event/onSelect; default NaN for callers with no real ray (e.g. script-driven tests).
void GltfRenderer::updateSelectionFromPick(int renderNodeIdx, const glm::vec3& selectionPoint, const glm::vec3& selectionRayOrigin)
{
  const auto clearSelection = [&]() {
    m_resources.selectedRenderNodes.clear();
    m_sceneSelection.clearSelection();
  };

  nvvkgltf::Scene* scene       = m_resources.getScene();
  const auto&      renderNodes = scene->getRenderNodes();

  // Nothing hit, or invalid instance: clear the selection.
  if(renderNodeIdx < 0 || renderNodeIdx >= static_cast<int>(renderNodes.size()))
  {
    clearSelection();
    return;
  }

  // Clicking the currently selected primitive again de-selects it
  if(m_resources.selectedRenderNodes.size() == 1 && m_resources.selectedRenderNodes.count(renderNodeIdx))
  {
    clearSelection();
    return;
  }

  // KHR_node_selectability: honor author intent. If the picked node opts out of selection (itself or
  // via an ancestor with `selectable:false`), redirect to the nearest selectable ancestor, or clear.
  if(!scene->isNodeSelectable(renderNodes[renderNodeIdx].refNodeID))
  {
    int selectableNodeID = scene->nearestSelectableAncestor(renderNodes[renderNodeIdx].refNodeID);
    if(selectableNodeID >= 0)
    {
      m_sceneSelection.selectNode(selectableNodeID, selectionPoint, selectionRayOrigin);
      focusSelectionInBrowser();
    }
    else
    {
      clearSelection();
    }
    return;
  }

  // The picked node is read-only (e.g. part of a referenced external asset). Select the first
  // editable ancestor instead, so the user can still change its transformation.
  if(scene->isNodeReadOnly(renderNodes[renderNodeIdx].refNodeID))
  {
    int editableNodeID = renderNodes[renderNodeIdx].refNodeID;
    while(editableNodeID >= 0 && scene->isNodeReadOnly(editableNodeID))
      editableNodeID = scene->editor().getNodeParent(editableNodeID);

    if(editableNodeID >= 0)
    {
      m_sceneSelection.selectNode(editableNodeID, selectionPoint, selectionRayOrigin);
      focusSelectionInBrowser();
    }
    else
    {
      clearSelection();
    }
    return;
  }

  // An instance of a placed external asset drawn without a node of its own: select its placing node.
  if(scene->isInstanceSetRenderNode(renderNodeIdx))
  {
    m_sceneSelection.selectNode(renderNodes[renderNodeIdx].refNodeID, selectionPoint, selectionRayOrigin);
    focusSelectionInBrowser();
    return;
  }

  // Regular case: select the picked primitive.
  const auto& renderNode = renderNodes[renderNodeIdx];
  int         nodeID     = renderNode.refNodeID;
  int         primIndex  = scene->getPrimitiveIndexForRenderNode(renderNodeIdx);
  int         meshID     = scene->getModel().nodes[nodeID].mesh;

  m_resources.selectedRenderNodes = {renderNodeIdx};
  m_sceneSelection.selectPrimitive(renderNodeIdx, nodeID, primIndex, meshID, selectionPoint, selectionRayOrigin);
  focusSelectionInBrowser();
}

nvutils::Bbox GltfRenderer::instanceWorldBounds(uint32_t slot) const
{
  const rs::Residency&           residency = m_resources.retainedGpu->residency();
  const rs::Residency::Instance* inst      = residency.instance(slot);
  const rs::Residency::Prim*     prim      = inst ? residency.prim(inst->prim) : nullptr;
  if(!prim || !prim->indexCount)
    return {};
  return nvutils::Bbox(prim->boundsMin, prim->boundsMax).transform(inst->world);
}

nvutils::Bbox GltfRenderer::getRenderNodeBbox(int renderNodeIndex)
{
  // the retained scene's instance slot (a render node until a stream host feeds it)
  if(m_resources.retainedGpu)
    return renderNodeIndex < 0 ? nvutils::Bbox{} : instanceWorldBounds(uint32_t(renderNodeIndex));
  // Empty when the render node is invalid or its primitive has no readable bounds, so a union
  // over a selection is never widened by a made-up box.
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(renderNodeIndex < 0 || !scene || !scene->valid())
    return {};

  const auto& renderNodes = scene->getRenderNodes();
  if(renderNodeIndex >= static_cast<int>(renderNodes.size()))
    return {};

  const nvvkgltf::RenderNode&      renderNode      = renderNodes[renderNodeIndex];
  const nvvkgltf::RenderPrimitive& renderPrimitive = scene->getRenderPrimitive(renderNode.renderPrimID);

  glm::vec3 minValues, maxValues;
  if(!tinygltf::utils::getPrimitivePositionBounds(scene->getModel(), *renderPrimitive.pPrimitive, minValues, maxValues))
    return {};
  nvutils::Bbox objBbox(minValues, maxValues);

  glm::mat4 nodeWorld = scene->computeNodeWorldMatrix(renderNode.refNodeID);

  // Nodes drawn more than once (EXT_mesh_gpu_instancing, instance sets): one local matrix per render node.
  const auto& gpuInstMap = scene->getRenderNodeLocalMatrices();
  auto        instIt     = gpuInstMap.find(renderNode.refNodeID);
  if(instIt != gpuInstMap.end() && !instIt->second.empty())
  {
    const auto& rnList = scene->getRenderNodeRegistry().getRenderNodesForNode(renderNode.refNodeID);
    for(size_t k = 0; k < rnList.size(); ++k)
    {
      if(rnList[k] == renderNodeIndex)
      {
        if(k < instIt->second.size())
          nodeWorld = nodeWorld * instIt->second[k];
        break;
      }
    }
  }
  return objBbox.transform(nodeWorld);
}

nvutils::Bbox GltfRenderer::getRenderNodesBbox(const std::unordered_set<int>& renderNodeIndices)
{
  nvutils::Bbox unionBbox;
  for(int idx : renderNodeIndices)
    unionBbox.insert(getRenderNodeBbox(idx));
  if(unionBbox.isEmpty())
    return {};
  return unionBbox;
}

bool GltfRenderer::hasValidScene()
{
  return m_resources.gpuView().hasContent();
}

int GltfRenderer::selectedNodeIndex()
{
  if(!hasValidScene() || !m_sceneSelection.hasSelection())
    return -1;
  const auto sel = m_sceneSelection.getSelection();
  if(sel.type == SceneSelection::SelectionType::eNode || sel.type == SceneSelection::SelectionType::ePrimitive)
    return sel.nodeIndex;
  return -1;
}

// Frame the whole scene, or the selected render nodes. A selection with no readable bounds leaves the camera alone.
nvutils::Bbox GltfRenderer::visibleSceneBounds()
{
  // the retained scene's shown instances, where they are now
  if(m_resources.retainedGpu)
    return m_resources.gpuView().bounds();
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || !scene->valid())
    return {};
  nvutils::Bbox box;
  const auto&   nodes = scene->getRenderNodes();
  for(int i = 0; i < int(nodes.size()); i++)
    if(nodes[size_t(i)].visible)
      box.insert(getRenderNodeBbox(i));
  return box.isEmpty() ? scene->getSceneBounds() : box;
}

void GltfRenderer::fitCamera(bool wholeScene)
{
  const nvutils::Bbox bbox = wholeScene ? visibleSceneBounds() : GltfRenderer::getRenderNodesBbox(m_resources.selectedRenderNodes);
  if(!bbox.isEmpty())
    m_cameraManip->fit(bbox.min(), bbox.max(), false, true, m_cameraManip->getAspectRatio());
}

void GltfRenderer::pickAt(glm::vec2 uv, bool center)
{
  if(!m_cameraManip || !hasValidScene())
    return;
  m_pendingClickPick = PendingClickPick{
      .modelViewInv   = glm::inverse(m_cameraManip->getViewMatrix()),
      .perspectiveInv = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
      .isOrthographic = (m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? 1 : 0,
      .pickPos        = uv,
      .isDoubleClick  = center,
  };
}

void GltfRenderer::setVisualization(shaderio::Visualization mode)
{
  if(m_resources.settings.visualization == mode)
    return;
  m_resources.settings.visualization = mode;
  resetFrame();
}

void GltfRenderer::setWireframe(bool on)
{
  if(m_resources.settings.wireframe == on)
    return;
  m_resources.settings.wireframe = on;
  resetFrame();
}

void GltfRenderer::setShowGrid(bool on)
{
  m_resources.settings.showGrid = on;
  resetFrame();
}

void GltfRenderer::setShowAxis(bool on)
{
  m_resources.settings.showAxis = on;
}

// a host names the gizmo's handles without its header
static_assert(HostRenderer::eGizmoTranslation == TransformHelperVk::ShowTranslation && HostRenderer::eGizmoRotation == TransformHelperVk::ShowRotation
              && HostRenderer::eGizmoScale == TransformHelperVk::ShowScale);

void GltfRenderer::setGizmoHooks(GizmoHooks hooks)
{
  m_gizmoHooks = std::move(hooks);
  // Reattach on the next frame so the handles come from the new attach hook
  m_visualHelpers.transform.clearAttachment();
  m_gizmoNodeIndex = -1;
  m_gizmoEntity    = {};
}

void GltfRenderer::setShowGizmo(bool on)
{
  m_resources.settings.showGizmo = on;
}

void GltfRenderer::setHelperLines(std::vector<shaderio::lines::LineVertex> vertices)
{
  m_visualHelpers.lines.setLines(std::move(vertices));
}

bool GltfRenderer::gizmoInput(glm::vec2 px, bool down, bool pressed, bool released)
{
  const glm::vec2 delta = pressed ? glm::vec2(0.f) : px - m_gizmoInputLast;
  m_gizmoInputLast      = px;
  if(!m_cameraManip || !m_resources.settings.showGizmo || !m_visualHelpers.transform.isAttached())
    return false;
  const VkExtent2D size     = renderSize();
  const bool       dragging = m_visualHelpers.transform.isDragging();
  const bool took = m_visualHelpers.transform.processInput(px, delta, down, pressed, released, m_cameraManip->getViewMatrix(),
                                                           m_cameraManip->getPerspectiveMatrix(),
                                                           glm::vec2(float(size.width), float(size.height)));
  // processInput reports the release that ends a drag as not taken; it still belongs to the drag
  return took || dragging;
}

void GltfRenderer::setPickObserver(std::function<bool(const std::optional<InstancePick>&, const glm::vec3& hit)> observer)
{
  m_pickObserver = std::move(observer);
}

void GltfRenderer::setDepthOfField(float focalDistance, float aperture, bool autoFocus, int blades, float bladeRotation)
{
  m_pathTracer.setDepthOfField(focalDistance, aperture, autoFocus, blades, bladeRotation);
  resetFrame();
}
