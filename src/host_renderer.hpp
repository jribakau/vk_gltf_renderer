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
#include <set>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vulkan/vulkan_core.h>

#include <nvutils/bounding_box.hpp>
#include <nvutils/camera_manipulator.hpp>

#include "env_image_loader.hpp"
#include "scene_price.hpp"
#include "scene_retained.hpp"
#include "shaders/gizmo_lines_shaderio.h.slang"
#include "shaders/shaderio.h"
#include "animation_control.hpp"

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
  // What is being edited, ready for it: it doesn't settle under memory pressure, and wakes if it has
  virtual void setLiveMeshes(std::set<rs::MeshId> meshes) = 0;
  // Lets go of what was decoded ahead, and whether to go on doing so: system memory is short
  virtual void dropDecodedAhead() = 0;
  virtual void decodeAhead(bool allowed) = 0;
  // Why a mesh of about `bytes` on the GPU won't fit, empty when it will. It may come all the same: what
  // doesn't fit falls to system memory and draws slower. `keepTextureDetail`: for what can come in smaller
  // instead, so textures count as coming down only as far as hardly shows. Application thread.
  virtual std::string geometryShortfall(uint64_t bytes, bool keepTextureDetail = false) = 0;
  // It is coming: counted as taken until it is resident. Application thread.
  virtual void admitGeometry(uint64_t bytes) = 0;
  // What a BLAS of this many triangles would hold, as the driver says, before any is built: one built to
  // be refitted, or one that is compacted (rs::kCompactedBlasShare). Application thread.
  virtual uint64_t blasBytes(uint64_t triangles, uint64_t vertices, bool refittable) const = 0;
  // What the retained scene holds on the GPU now; empty before it has one. Application thread.
  virtual rs::MemoryLedger memoryLedger() const = 0;
  // The GPU memory free beyond what is kept back, negative when that is in use too; none without a
  // known budget. Application thread.
  virtual std::optional<int64_t> sceneHeadroom() const = 0;
  // Max texture size: the largest long edge an image decoded from now on keeps; 0: no limit. Any thread.
  virtual void setMaxTextureExtent(uint32_t extent) = 0;
  // Where what the renderer has to tell the user goes, from the application thread (a texture lowered
  // to fit GPU memory); empty: nowhere
  virtual void setNoticeSink(std::function<void(const std::string&)> sink) = 0;
  // What the scene is drawn without, or slower for, for want of GPU memory
  struct MemoryReport
  {
    uint32_t lowered = 0, largest = 0;  // textures lowered to fit, and the long edge of the largest
    uint32_t parked       = 0;          // meshes that found no memory anywhere, and aren't drawn
    uint64_t spilledBytes = 0;          // what lies in system memory, in steps worth mentioning
    uint32_t letGoHidden  = 0;          // items let go of while hidden: they come back when shown
    uint32_t settled      = 0;          // items holding only what is drawn: the next edit waits a moment
    // the bar: all of it in steps worth mentioning. `budget` is what the driver offers, `used` what it counts
    uint64_t budget = 0, used = 0, geometry = 0, textures = 0;
    std::string nextToGo;               // what gives way next, in a few words
    bool        operator==(const MemoryReport&) const = default;
  };
  // Where that is reported as it changes, from the application thread; empty: nowhere
  virtual void setMemorySink(std::function<void(const MemoryReport&)> sink) = 0;
  // What can move of the worn items, as often as it changes or a reading comes (rs::ItemDetail), and where
  // the Fit's asks go: an item (ItemDetail's own), a step up or down. Application thread.
  virtual void setItemDetail(rs::ItemDetail detail) = 0;
  virtual void setDetailSink(std::function<void(uint64_t item, bool up)> sink) = 0;
  // An export is running: textures don't follow GPU memory until it ends, unless it runs past the
  // driver's budget (rs::MemoryWatch). Application thread.
  virtual void holdTextures(bool held) = 0;
  // Textures the scene took are still being cooked, to land one by one (FvHost::Impl::cookInPlace): a
  // still doesn't begin until they are in, DLSS drawing the view meanwhile, so it isn't restarted as
  // each lands. Application thread, every frame.
  virtual void setTexturesArriving(bool arriving) = 0;
  // A drape is on its way on the GPU's other queue (physics-plan step 13b): the still stops refining a
  // few frames after the last change, so the cloth has the GPU, and begins when the drape lands. A move of
  // the view draws as ever. Application thread, every frame.
  virtual void setStillPaused(bool paused) = 0;
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
  // The pixel readout: what is at a normalized viewport position (origin top-left) and how bright it
  // is, read a frame or two later and again while the still refines (a few times a second), so it
  // settles as the image does; none: stop reading. Every reading goes to the observer.
  struct PixelReadout
  {
    shaderio::PixelInspect      data{};  // the hit, its material, the image's radiance
    std::optional<InstancePick> pick;    // the mesh and material hit, when it is one
    // that radiance's luminance after the last frame's exposure and white balance: what the tone curve
    // takes, and what the false-colour exposure view bands in stops from middle grey (0.18)
    float exposed = 0.0F;
  };
  virtual void inspectAt(std::optional<glm::vec2> uv)                                        = 0;
  virtual void setInspectObserver(std::function<void(const PixelReadout& readout)> observer) = 0;
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
    // Where the gizmo sits on the entity, and what it turns and scales the entity about, in the
    // entity's own space (under its local transform); none: its origin. `edit` still gets the
    // entity's own transform (gizmo_pivot.hpp).
    std::function<std::optional<glm::vec3>(rs::EntityId entity)> pivot;
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
  // The false-colour exposure view in place of the image (TonemapperData::falseColor): the tone
  // mapper's alone, so a still keeps its samples
  virtual void setFalseColor(bool on) = 0;
  // The wipe against a kept frame: keep the image as the next frame shows it (without the overlays),
  // and show it left of `at` (a share of the view's width from its left edge), the live image right
  // of it. None: no wipe, and the kept frame goes. A resize drops the kept frame too.
  virtual void keepWipeFrame()                  = 0;
  virtual void setWipe(std::optional<float> at) = 0;
  virtual void setShowGrid(bool on)                                              = 0;
  virtual void setShowAxis(bool on)                                              = 0;
  // path-traced depth of field: `aperture` is the lens radius squared in metres (0: pinhole)
  virtual void setDepthOfField(float focalDistance, float aperture, bool autoFocus, int blades = 0, float bladeRotation = 0) = 0;

  // ---- Rendering, and its states the app shows
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
    // of the usage, what the allocator holds on the device and what of that it has handed out
    VkDeviceSize allocatorBlocks = 0, allocatorAllocations = 0;
  };
  virtual MemoryBudget deviceMemoryBudget() const = 0;
  virtual uint32_t     frameBytesPerPixel() const = 0;
  // The last frame the application thread recorded, for the host's drag timing at its next update: its
  // recording's time (onRender), and whether the frame before it was already done on the GPU when it
  // was submitted, so the GPU had sat idle waiting for the CPU. Measured only with a timing sink.
  struct FrameCpu
  {
    double recordMs        = 0;
    bool   gpuIdleAtSubmit = false;
  };
  virtual FrameCpu lastFrameCpu() const { return {}; }
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
  // What the physical sky's clouds leave of its light, as its last full bake measured it, for the
  // camera's meter: `sky`, the sky's own light on a figure with the clouds against without; `sun`, the
  // share of the sun that gets past them. 1 and 1 under a clear sky. Application thread.
  struct SkyCloudLight
  {
    float sky = 1.0F;
    float sun = 1.0F;
    bool  operator==(const SkyCloudLight&) const = default;
  };
  virtual SkyCloudLight skyCloudLight() const { return {}; }
};
