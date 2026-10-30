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

// What a stream host (FvHost, FvView, FvMaterials) asks of the renderer: its retained scene, the
// view, selection, the gizmo, the host's clip, the environment and a few states for the app.
// GltfRenderer implements it; a test implements it over a RetainedScene without a GPU, so the
// host's commands run in gtest (daz-viewer docs/command-journal-plan.md, C0). Application thread
// only, except where a method says otherwise.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vulkan/vulkan_core.h>

#include <nvutils/bounding_box.hpp>
#include <nvutils/camera_manipulator.hpp>

#include "env_image_loader.hpp"
#include "rendering_mode.hpp"
#include "scene_retained.hpp"
#include "shaders/gizmo_lines_shaderio.h.slang"
#include "shaders/shaderio.h"
#include "ui_animation.hpp"

namespace nvvk {
class ResourceAllocator;
}
namespace tinygltf {
class Model;
}

class HostRenderer
{
public:
  virtual ~HostRenderer() = default;

  // ---- The host's retained scene (daz-viewer docs/retained-scene-plan.md, P4)
  virtual rs::RetainedScene& hostScene()            = 0;
  virtual bool               feedsHostScene() const = 0;
  // the retained scene draws (no CPU glTF scene beside it)
  virtual bool drawsRetainedScene() const = 0;
  // a composition: makes the store resident whole
  virtual void replaceHostScene() = 0;
  // Decodes an image now, on the calling thread (a load or wear worker), for the next replace
  virtual void prefetchImage(const std::filesystem::path& diskPath, bool srgb) = 0;
  // what is shown, where it is now (world space)
  virtual nvutils::Bbox visibleSceneBounds() = 0;

  // ---- Selection, picking and the gizmo
  virtual void         selectEntity(rs::EntityId entity) = 0;
  virtual rs::EntityId selectedEntity() const            = 0;
  // a click at a normalized viewport position (origin top-left); `center` recenters the camera
  virtual void pickAt(glm::vec2 uv, bool center) = 0;
  struct InstancePick
  {
    rs::EntityId   entity;
    rs::MaterialId material;
  };
  // sees every click pick first; true keeps the renderer from changing the selection
  virtual void setPickObserver(std::function<bool(const std::optional<InstancePick>& pick, const glm::vec3& hit)> observer) = 0;
  // the handles the gizmo shows on an entity (TransformHelperVk's flags), and its edits: the
  // entity's new local translation, rotation and scale, `done` on the release that ends the drag
  enum GizmoHandles : uint32_t
  {
    eGizmoTranslation = 1 << 0,
    eGizmoRotation    = 1 << 1,
    eGizmoScale       = 1 << 2,
  };
  struct GizmoHooks
  {
    std::function<uint32_t(rs::EntityId entity)> attach;
    std::function<void(rs::EntityId entity, const glm::vec3& t, const glm::quat& r, const glm::vec3& s, bool done)> edit;
  };
  virtual void setGizmoHooks(GizmoHooks hooks) = 0;
  virtual void setShowGizmo(bool on)           = 0;
  // the gizmo's pointer input in render pixels; true when the gizmo took the event
  virtual bool gizmoInput(glm::vec2 px, bool down, bool pressed, bool released) = 0;
  // world-space segments drawn over the image (a camera's frustum, a light's marker)
  virtual void setHelperLines(std::vector<shaderio::lines::LineVertex> vertices) = 0;

  // ---- The view
  virtual std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() = 0;
  virtual VkExtent2D                                  renderSize() const     = 0;
  virtual void                                        fitCamera(bool wholeScene) = 0;
  virtual void setVisualization(shaderio::Visualization mode)                    = 0;
  virtual void setWireframe(bool on)                                             = 0;
  virtual void setShowGrid(bool on)                                              = 0;
  virtual void setShowAxis(bool on)                                              = 0;
  // path-traced depth of field: `aperture` is the lens radius squared in metres (0: pinhole)
  virtual void setDepthOfField(float focalDistance, float aperture, bool autoFocus, int blades = 0, float bladeRotation = 0) = 0;

  // ---- Rendering, and its states the app shows
  virtual void          setRenderSystem(RenderingMode mode) = 0;
  virtual RenderingMode renderSystem() const               = 0;
  virtual bool          isStill() const                    = 0;
  virtual bool          isRefiningStill() const            = 0;
  virtual bool          isStillConverged() const           = 0;
  virtual bool          accumulationComplete() const       = 0;
  // the samples per pixel the still has accumulated, and the cap it refines to
  virtual int           accumulatedSamples() const         = 0;
  virtual int           sampleCap() const                  = 0;
  // saves the frame once the next one is drawn (a .hdr: the linear render), and whether that is still to come
  virtual void          saveImage(const std::filesystem::path& path) = 0;
  virtual bool          imageSavePending() const           = 0;
  // a pipeline built on an earlier run compiled again: the driver's shader cache is full
  virtual bool          driverCacheFull() const            = 0;
  // the VRAM budget (VK_EXT_memory_budget); `known` false without the extension. Any thread.
  struct MemoryBudget
  {
    VkDeviceSize budget = 0;
    VkDeviceSize usage  = 0;
    bool         known  = false;
  };
  virtual MemoryBudget deviceMemoryBudget() const = 0;
  virtual uint32_t     frameBytesPerPixel() const = 0;
  // the device allocator a GPU simulation allocates from (dForce); none without a device
  virtual nvvk::ResourceAllocator* deviceAllocator() = 0;

  // ---- The host's clip
  virtual void              setHostClip(rs::ClipId clip, float at = 0.f) = 0;
  virtual float             hostClipTime() const                         = 0;
  virtual float             hostClipDuration() const                     = 0;
  virtual void              seekHostClip(float time)                     = 0;
  virtual AnimationControl& editAnimationControl()                       = 0;

  // ---- The environment
  // installs a lat-long decoded elsewhere; `name` is its file, or empty for a generated one
  [[nodiscard]] virtual bool loadHdrEnvironment(EnvImage&& image, const std::filesystem::path& name) = 0;
  // the current sky into a model an export writes
  virtual void writeEnvironmentSky(tinygltf::Model& outModel, const std::filesystem::path& destination) const = 0;
};
