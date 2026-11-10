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

#include <nvshaders/tonemap_functions.h.slang>  // getColorCorrectionMatrix

#include "gizmo_pivot.hpp"
#include "renderer.hpp"
#include "tinygltf_utils.hpp"

// Applies a completed async ray-pick result (see m_pendingClickPick's comment in renderer.hpp for
// the full pipeline) - selection, and on a double click, camera recenter.
void GltfRenderer::applyClickPickResult(const nvvk::RayPicker::PickResult& pickResult, bool isDoubleClick)
{
  // Environment was picked (no hit)
  const bool hasHit = pickResult.instanceID >= 0;
  glm::vec3 worldPos = hasHit ? pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT : glm::vec3(0);

  // Set or de-select the selected object
  // pickResult.instanceID is the TLAS instance index, the retained scene's instance slot
  if(!isDoubleClick)
  {
    std::optional<InstancePick> pick;
    if(hasHit)
      pick = instancePickAt(pickResult.instanceID, pickResult.primitiveID);
    // the entity hit, or nothing; the one selected, hit again, is let go
    if(!(m_pickObserver && m_pickObserver(pick, worldPos)))
      selectEntity(pick && pick->entity != m_selectedEntity ? pick->entity : rs::EntityId{});
  }

  if(!hasHit)
  {
    // a double-click on nothing frames the whole scene
    if(isDoubleClick)
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
  // the retained scene's instance slot
  if(!m_resources.retainedGpu || renderNodeIndex < 0)
    return {};
  return instanceWorldBounds(uint32_t(renderNodeIndex));
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

// Frame the whole scene, or the selected render nodes. A selection with no readable bounds leaves the camera alone.
nvutils::Bbox GltfRenderer::visibleSceneBounds()
{
  // the retained scene's shown instances, where they are now
  return m_resources.gpuView().bounds();
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
      .pickPos        = m_lens.rendered(uv),  // the picture's point, in the image the rays are traced for
      .isDoubleClick  = center,
  };
}

void GltfRenderer::setHover(bool on)
{
  if(on == m_hover.on)
    return;
  m_hover            = {.on = on, .version = m_hover.version + 1};
  m_resources.hoverOn = on;
  m_resources.hoveredRenderNodes.clear();
  // a finished still traces no frame that could take the object IDs: begin again, once
  if(on && m_pathTracer.selectionImageStale())
    resetFrame();
}

void GltfRenderer::hoverAt(std::optional<glm::vec2> uv)
{
  if(!m_hover.on || uv == m_hover.uv)
    return;
  m_hover.uv  = uv;
  m_hover.due = uv.has_value();
  if(!uv)
    hoverEntity({});
}

void GltfRenderer::hoverEntity(rs::EntityId entity)
{
  if(entity == m_hover.entity)
    return;
  m_hover.entity = entity;
  refreshEntityHover();
  m_hover.version++;
}

void GltfRenderer::refreshEntityHover()
{
  m_resources.hoveredRenderNodes.clear();
  if(!m_resources.hostFed || !m_resources.retainedScene)
    return;
  rs::RetainedScene& store = *m_resources.retainedScene;
  if(!m_hover.entity.isNone() && !store.entity(m_hover.entity))
    m_hover.entity = {};  // gone with what it was in
  if(m_hover.entity.isNone() || !m_resources.retainedGpu)
    return;
  const rs::Residency& residency = m_resources.retainedGpu->residency();
  for(uint32_t slot = 0; slot < residency.instanceHighWater(); slot++)
    if(const rs::Residency::Instance* inst = residency.instance(slot))
      if(store.isWithin(inst->entity, m_hover.entity))
        m_resources.hoveredRenderNodes.insert(int(slot));
}

bool GltfRenderer::hoverDraws() const
{
  return m_hover.on && !m_lens.bent() && !m_pathTracer.selectionImageStale();
}

bool GltfRenderer::hoverPickDue()
{
  if(!m_hover.on || !m_hover.uv || !m_cameraManip || !hasValidScene())
    return false;
  // the scene moved under the pointer (the wheel, a camera looked through, an animation's camera)
  const glm::mat4 camera = m_cameraManip->getPerspectiveMatrix() * m_cameraManip->getViewMatrix();
  if(camera != m_hover.pickedWith)
    m_hover.due = true;
  if(!m_hover.due || !hoverDraws())
    return false;
  m_hover.due        = false;
  m_hover.pickedWith = camera;
  m_pendingClickPick = PendingClickPick{
      .modelViewInv   = glm::inverse(m_cameraManip->getViewMatrix()),
      .perspectiveInv = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
      .isOrthographic = (m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? 1 : 0,
      .pickPos        = *m_hover.uv,
      .isDoubleClick  = false,
      .hover          = true,
  };
  return true;
}

void GltfRenderer::applyHoverPickResult(const nvvk::RayPicker::PickResult& pickResult)
{
  if(!m_hover.on || !m_hover.uv)
    return;  // the pointer left meanwhile
  std::optional<InstancePick> pick;
  if(pickResult.instanceID >= 0)
    pick = instancePickAt(pickResult.instanceID, pickResult.primitiveID);
  if(m_hoverObserver)
    m_hoverObserver(pick);
  else
    hoverEntity(pick ? pick->entity : rs::EntityId{});
}

void GltfRenderer::inspectAt(std::optional<glm::vec2> uv)
{
  m_inspect.uv  = uv;
  m_inspect.due = uv.has_value();
}

std::optional<HostRenderer::InstancePick> GltfRenderer::instancePickAt(int instanceID, int primitiveID)
{
  if(instanceID < 0 || !m_resources.retainedGpu)
    return std::nullopt;
  const rs::Residency::Instance* inst = m_resources.retainedGpu->residency().instance(uint32_t(instanceID));
  if(!inst)
    return std::nullopt;
  InstancePick pick{inst->entity, inst->material};
  // a prim that draws merged groups (Submesh::mergedInto): the material of the group hit
  const rs::RetainedScene& scene  = hostScene();
  const rs::MeshRecord*    record = scene.mesh(inst->mesh);
  const rs::Entity*        entity = scene.entity(inst->entity);
  if(record && entity && entity->mesh && inst->submesh < record->submeshes.size())
  {
    const uint32_t hit = record->submeshAt(inst->submesh, uint32_t(std::max(primitiveID, 0)));
    for(const rs::MeshPart& part : entity->mesh->parts)
      if(part.mesh == inst->mesh && hit != inst->submesh && record->submeshes[hit].material < part.materials.size())
        pick.material = part.materials[record->submeshes[hit].material];
  }
  return pick;
}

// A reading at the frame's end: at once for a new place, then every kInspectEvery while frames are
// traced (the still refining, or the view moving), and once more after the last of them, so the
// finished image is what it settles on. One in flight at a time.
void GltfRenderer::cmdInspect(VkCommandBuffer cmd, bool traced)
{
  m_inspect.traced = m_inspect.traced || traced;
  if(!m_inspect.uv || m_inspect.inFlight || !m_app)
    return;
  const auto now = std::chrono::steady_clock::now();
  if(!m_inspect.due && !(m_inspect.traced && now - m_inspect.last >= kInspectEvery))
    return;
  if(!m_pathTracer.cmdInspectPixel(cmd, m_resources, m_lens.rendered(*m_inspect.uv)))
    return;
  m_inspect.due      = false;
  m_inspect.traced   = false;
  m_inspect.last     = now;
  m_inspect.inFlight = nvvk::SemaphoreState::makeFixed(m_app->getFrameSignalSemaphore());
}

void GltfRenderer::updateInspectState()
{
  if(!m_inspect.inFlight || !m_inspect.inFlight->testSignaled(m_device))
    return;
  m_inspect.inFlight.reset();
  if(!m_inspect.uv || !m_inspectObserver)
    return;
  PixelReadout readout{.data = m_pathTracer.inspectResult()};
  readout.pick = instancePickAt(readout.data.ids.x, readout.data.ids.y);
  // as applyTonemap takes it: mul(inputMatrix, color), which the C++ shim writes v * M
  const shaderio::TonemapperData& tm   = m_tonemapInput.data;
  const glm::vec3                 seen = glm::vec3(readout.data.radiance) * shaderio::getColorCorrectionMatrix(tm.exposure, tm.temperature, tm.tint);
  readout.exposed                      = 0.2126F * seen.x + 0.7152F * seen.y + 0.0722F * seen.z;
  m_inspectObserver(readout);
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

void GltfRenderer::setWipe(std::optional<float> at)
{
  m_wipeAt = at ? std::clamp(*at, 0.0F, 1.0F) : -1.0F;
  if(!at)
  {
    m_wipeKeep = false;
    dropWipeFrame();
  }
}

// The kept frame goes once the frames in flight that read it are done
void GltfRenderer::dropWipeFrame()
{
  if(!m_wipeFrame.image)
    return;
  if(m_app)
    m_app->submitResourceFree([alloc = &m_resources.allocator, image = m_wipeFrame]() mutable { alloc->destroyImage(image); });
  else
    m_resources.allocator.destroyImage(m_wipeFrame);
  m_wipeFrame  = {};
  m_wipeExtent = {};
}

void GltfRenderer::setShowGrid(bool on)
{
  // view.set sends the grid with the other display options: the same one again keeps a still refining
  if(m_resources.settings.showGrid == on)
    return;
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
  m_gizmoEntity    = {};
}

void GltfRenderer::setShowGizmo(bool on)
{
  m_resources.settings.showGizmo = on;
}

// the registered settings (renderer_settings.cpp), which renderVisualHelpers hands the gizmo each frame
void GltfRenderer::setGizmoOptions(const GizmoOptions& options)
{
  m_resources.settings.gizmoLocal  = options.local;
  m_resources.settings.snapEnabled = options.snap;
  if(options.move > 0)
    m_resources.settings.snapMove = options.move;
  if(options.rotate > 0)
    m_resources.settings.snapRotation = options.rotate;
  if(options.scale > 0)
    m_resources.settings.snapScale = options.scale;
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
  m_bloom.setAperture(blades, bladeRotation);  // the star's spikes: across the blades' edges
  resetFrame();
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
    // on its pivot (a figure's root bone), turned as the drag will turn it back
    const std::optional<glm::vec3> pivot = m_gizmoHooks.pivot ? m_gizmoHooks.pivot(m_selectedEntity) : std::nullopt;
    m_gizmoPivot                         = pivot.value_or(glm::vec3(0.f));
    m_gizmoPosition = pivotPosition(m_gizmoPosition, glm::quat(glm::radians(m_gizmoRotation)), m_gizmoScale, m_gizmoPivot);
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
  const bool hover         = m_pendingClickResult->hover;
  m_pendingClickResult.reset();
  if(hover)
    applyHoverPickResult(m_rayPicker.getResult());
  else
    applyClickPickResult(m_rayPicker.getResult(), isDoubleClick);
}
