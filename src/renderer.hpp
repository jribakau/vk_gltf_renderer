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

#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>

#include <nvapp/application.hpp>
#include <nvapp/imgui_texture.hpp>
#include <nvgui/camera.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/ray_picker.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/semaphore.hpp>
#include "settings_registry.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_rtx.hpp"
#include "gltf_scene_vk.hpp"
#include <nvvk/profiler_vk.hpp>
#include "nvutils/parameter_parser.hpp"
#include "nvutils/parameter_registry.hpp"
#include "nvutils/parameter_sequencer.hpp"

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device
#include "shaders/gizmo_lines_shaderio.h.slang"

#include "renderer_pathtracer.hpp"
#include "renderer_rasterizer.hpp"
#include "resources.hpp"
#include "renderer_silhouette.hpp"
#include "env_baker.hpp"
#include "env_image_loader.hpp"
#include "sky_bruneton.hpp"
#include "sky_sun.hpp"
#include "gltf_environment_sky.hpp"
#include "camera_controls.hpp"
#include "busy_state.hpp"
#include "scene_selection.hpp"
#include "gizmo_visuals_vk.hpp"
#include "timeline_pipeline.hpp"
#include "scene_clip_player.hpp"
#include "host_renderer.hpp"

#ifdef USE_NVMCP
/// A clean timing series for one GPU profiler timer, gathered over a controlled window.
/// Times are milliseconds. Compare runs on `median`; see src/mcp_timing.cpp.
struct TimerMeasurement
{
  std::string timer{};
  size_t      samples{0};
  double      mean{0.0};
  double      median{0.0};
  double      standardDeviation{0.0};
  double      minimum{0.0};
  double      maximum{0.0};
  std::string error{};

  [[nodiscard]] static TimerMeasurement failure(std::string message) { return {.error = std::move(message)}; }
};
#endif

class GltfRenderer : public nvapp::IAppElement, public HostRenderer
{
public:
  GltfRenderer(nvutils::ParameterRegistry* parameterReg, const nvutils::ParameterParser* parameterParser);
  ~GltfRenderer() override = default;


  /// Load a scene from disk into `m_resources.scene`. Returns true only if a new scene was
  /// actually installed; returns false on any failure (empty path, findFile miss, OBJ parse
  /// error, glTF load error) and leaves the current scene pointer in place. Does *not* tear
  /// down previously-loaded derived state (sceneRtx BLAS/TLAS, sceneGpu buffers, undo stack,
  /// thumbnails, rasterizer recorded cmd, ...): callers that replace an existing scene at
  /// runtime must tear down first, see `loadSceneFile` and `onFileDrop`. Startup in `main.cpp`
  /// runs before any scene exists and can call this directly.
  bool createScene(const std::filesystem::path& sceneFilename);
  // Ensure an editable Scene exists (wired to the UI, no GPU build) so add/import can run from nothing.
  void ensureEmptyScene();
  void placeCameraFromFile(const tinygltf::Model& model);
  void createSceneFromDescriptor(const std::filesystem::path& descriptorPath);
  void createHDR(const std::filesystem::path& hdrFilename);
  // The upload half of createHDR: `envImage` is consumed. `sourceFile` is empty for a generated image.
  void uploadHdrEnvironment(EnvImage&& envImage, const std::filesystem::path& sourceFile);
  void onMergeScene(const std::filesystem::path& filename);
  void onReferenceScene(const std::filesystem::path& filename);
  // Shared worker for merge (embed) and reference (glTF 2.1 external asset) imports.
  void addSceneFromFile(const std::filesystem::path& filename, bool asReference);
  /// Full GPU rebuild (textures + geometry + AS). Only uploads GPU data; CPU side must already be
  /// parsed (`mergeScene`, `parseScene`, etc., depending on how the model was changed).
  void                                        rebuildVulkanSceneFull();
  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() override { return m_cameraManip; }
  void                                        setDlssHardwareAvailability(bool rrAvailable, bool srAvailable);
  void                                        setSerAvailable(bool available);
  void                                        setDeviceExtensions(std::vector<std::string> extensions);
  void                                        setOpacityMicromapAvailable(bool hardware, bool enabled);
  /// Unless --ptMaxSamples was given, raises the cap so the accumulation covers the full headless run (--frames).
  void alignMaxSamplesForHeadless(uint32_t headlessFrames);
  // Headless runs render still images: unless --animPlay was given, animations don't play, since
  // advancing them resets the path tracer's accumulation every frame.
  void applyHeadlessDefaults();

  /// Presentation mode (F11, View > Presentation Mode, --presentationMode): see ui_presentation_mode.hpp.
  /// Registers a panel owned by another application element, so presentation mode hides it too.
  /// Leave presentation mode now, restoring the panels and the window. main() calls this before
  /// Application::deinit(), which persists the window geometry and the ImGui.ini.

  /// Drain the queue, recompile the active renderer's shaders, and reset accumulation.
  /// Application thread only: it destroys and recreates live pipelines.
  bool reloadShaders();  // false: the from-file compile failed and the embedded SPIR-V is now running

  /// Mirror Resources::sunDirection back into sunAzimuth/sunElevation after the sky UI moves it.

  /// Drive the shared sun from azimuth/elevation in degrees -- the inverse of syncSunAngles().

  /// Find the light the scene marks as the sky's sun and adopt its direction, or report that
  /// there is none. Runs at the frame top, so moving that light -- gizmo, inspector, animation --
  /// carries the sky with it.
  ///
  /// Only a *marked* light qualifies. An ordinary directional light is an ordinary light: the
  /// specification's model is that the sky is a medium and suns are lights, but nothing in a bare
  /// directional light says which one a sky should scatter.

  /// Where an authored sky arrived from. It decides exactly one thing: whether an explicit
  /// `--envSystem` outranks the sky's own type.
  ///
  /// It does for a scene, because the flag is a statement about the run and the scene arrives as
  /// part of it. It does not for a preset the user loads by hand at runtime, because that *is* the
  /// statement, and a flag from start-up has no business overruling it.
  enum class SkySource
  {
    eScene,
    ePreset
  };

  /// Push an authored sky into the live settings, and mark the environment for a re-bake.
  /// `uriBase` is the directory of the file that carried the sky -- the glTF or the `.sky.json` --
  /// which is what a panorama's relative URI was written against.
  void applySkyDescriptor(const SkyDescriptor& sky, SkySource source, const std::filesystem::path& uriBase);

  /// The live settings as an authored sky, ready to serialize.
  ///
  /// nullopt when there is nothing to write: the active environment is not an OMI type (the
  /// physical Sky is, None is not) *and* no scene sky was loaded to re-emit. `destination` is only
  /// used to make a panorama's URI relative to where the file is going.
  [[nodiscard]] EnvironmentState skyDescriptorFromSettings(const std::filesystem::path& destination) const;

  /// Write the current sky to a `.sky.json`, and read one back. False / no-op on failure, having
  /// logged it.
  ///
  /// The sun angle travels with the sky -- see sky_preset.hpp -- and a loaded one is aimed through
  /// SkySun::aim(), so a scene whose sun is a marked light records the move on the undo stack like
  /// any other transform.
  bool saveSkyPreset(const std::filesystem::path& path) const;
  bool loadSkyPreset(const std::filesystem::path& path);

  /// The start-up `--loadSkyPreset` / `--saveSkyPreset`, applied at the first frame top rather than
  /// now: the ini restore and the scene's authored sky both land there, and either would otherwise
  /// overwrite a preset applied before them. Load runs before save, so a run can do both.
  void queueStartupSkyPresets(const std::filesystem::path& load, const std::filesystem::path& save);

  /// Report an environment edit: preview it this frame, commit once the edits stop.
  ///
  /// The preview skips the alias rebuild, the PDF write and (on the path-tracing path) the
  /// prefilter refresh. Skipping the PDF write leaves the image's PDF at zero, so while a preview is
  /// showing the path tracer lights the environment by BSDF sampling alone -- see
  /// EnvBaker::preview.
  ///
  /// The commit is owed rather than immediate, and onRender pays it on the first frame that asks
  /// for no preview. "The edits stopped" is deliberately not "ImGui has no active item": the sun
  /// also moves from the transform gizmo on its light, from an animation and from an MCP
  /// parameter write, none of which ImGui knows about.
  ///
  /// A no-op while a full rebuild is already pending: that rebuild reads the current settings, so
  /// a preview would only add a second commit behind it.
  void requestEnvironmentPreview()
  {
    if(m_envBakeDirty)
      return;
    m_envPreviewPending = true;
    m_envCommitOwed     = true;
  }

  /// Set the firefly clamp from the current environment, and record which environment that was.
  ///
  /// Every site that wants to recalibrate goes through here, so the "what was it calibrated for"
  /// bookkeeping cannot drift from the value itself -- switching sky -> HDR -> sky has to
  /// recalibrate, and it only does if each hop updates both.
  void calibrateFireflyClamp();

  /// Illuminance of the sun *above* the atmosphere, in the units the renderer works in.
  ///
  /// Deliberately unattenuated. Two callers want exactly that and for the same reason: it is the
  /// property of the star rather than of this moment, so it does not change as the sun sets.
  [[nodiscard]] glm::vec3 skySunIlluminanceAboveAtmosphere() const;

  /// Height above the planet's surface, in km, that aerial perspective puts the eye at.
  ///
  /// Not the sky's observer altitude, and deliberately so. That one is fixed because a moving
  /// observer would invalidate the baked image every frame; this one is evaluated per pixel and has
  /// no image to invalidate, so it can afford to be true. It has to be: aerial perspective places
  /// the scene *relative* to the eye, so an eye 300 m up with a camera two kilometres above the
  /// terrain puts the whole landscape under the ground, where the lookup degenerates and the haze
  /// stops at the horizon line.
  ///
  /// So the scene's lowest point is placed at the observer altitude and the eye goes wherever the
  /// camera stands above it. Y-up, like everything else the sky does.
  [[nodiscard]] float aerialPerspectiveEyeAltitudeKm() const;

  /// Mark the baked lighting environment stale. Cheap and safe to call from a parameter callback
  /// or an ImGui handler: the bake itself runs on the application thread at the top of the next
  /// frame, because it submits and waits on the queue.
  void onEnvironmentChanged() { m_envBakeDirty = true; }

  /// The active renderer changed. A baked sky committed under the path tracer skipped the prefilter,
  /// so the rasterizer arriving would light from cubemaps of an older environment.
  void onRenderSystemChanged()
  {
    if(m_resources.settings.renderSystem == RenderingMode::eRasterizer && isBakedEnvironment(m_resources.settings.envSystem))
      onEnvironmentChanged();
    resetFrame();
  }

  /// Copy the current sky settings into EnvBaker. Cheap and safe mid-frame, so both the preview
  /// and the commit call it; the resolution is not part of it, since changing that reallocates.
  void updateBakerParameters();

  /// Re-run EnvBaker against the current settings and rebind the shared environment. Application
  /// thread only. No-op unless the active envSystem is a baked one.
  void refreshBakedEnvironment();

  /// The physical sky's atmosphere: the Earth defaults with the panel's settings laid over them.
  /// One builder, for the same reason buildSkyOmiParameters() is one: the LUT
  /// precompute and the bake must never disagree about which atmosphere they are describing.
  shaderio::SkyAtmosphereParameters buildAtmosphereParameters() const;

  /// The authored sky parameters as the shaders see them, built from the current settings plus the
  /// scene's sun. One builder feeds both consumers -- the bake (lighting field, sun excluded) and
  /// the per-ray background (sun included) -- so the two can never disagree about the same sky.
  shaderio::SkyOmiParameters buildSkyOmiParameters() const;

  /// Resources::sunDirection, which is world space, expressed in the environment's own frame --
  /// the frame the lat-long image and the analytic sky are evaluated in, turned by envRotation.
  /// Everything that places the sun *in the sky* (the bake, the gradient's glow and disk) wants
  /// this; everything that lights the scene with it wants the world-space direction.
  [[nodiscard]] glm::vec3 sunInEnvironmentFrame() const;

  /// Settings::envRotation was edited: north moved, so a sky's sun is placed again from Time of Day.
  void onEnvironmentRotated();

  /// Illuminance of the gradient sky's sun, before `envIntensity`.
  ///
  /// Unlike the physical sky's, it is not measured: `sunColor` says what the disk is drawn in, not
  /// how bright it is, so this scales the sun against the dome it sits under. Three callers need
  /// the same answer -- the frame the renderer lights, the file a save writes, and the firefly
  /// clamp -- which is why it is one function.
  [[nodiscard]] glm::vec3 gradientSunIlluminance() const;

  /// The rasterizer's shadow-map light and box, and the ground's fade disk, into the frame info.
  /// `rendererSunLight` is the frame's own flag: whether the sky's sun is a light past the scene's.
  void fillShadowAndGround(shaderio::SceneFrameInfo& finfo);

  /// Frame-top consumer of onEnvironmentChanged(). Called from onUIRender, not onRender: it waits
  /// on the device and rewrites descriptors, which is unsafe mid-command-buffer.
  void applyPendingEnvironmentRefresh();

  /// Parse the sky a freshly loaded scene authors, if it has one, and queue it for the next frame
  /// top. Does not touch settings directly -- see applyPendingAuthoredSky for why.
  void applyEnvironmentSkyFromScene();

  /// Apply the queued authored sky to the live settings. Runs after the ini restore so that glTF
  /// wins over ini; scenes without the extension leave the settings untouched.
  void applyPendingAuthoredSky();

  /// Apply the presets queued by queueStartupSkyPresets(). Runs right after applyPendingAuthoredSky.
  void applyPendingSkyPresets();

  /// URI to record for an environment image on save: relative to the glTF when the image lives
  /// under it, absolute otherwise.
  std::string relativeUriForSave(const std::filesystem::path& asset, const std::filesystem::path& destination) const;

  /// Pre-save hook body: write the current sky into the model being serialized, or strip any
  /// stale extension when the save toggle is off.
  void writeEnvironmentSky(tinygltf::Model& outModel, const std::filesystem::path& destination) const override;

  /// Write, update, or remove the directional light that carries the sky's sun, so a saved sky is
  /// complete for a renderer that only reads the extension. Marked, so a reload identifies it and
  /// a later save manages the same one instead of adding another.
  void writeSkySunLight(tinygltf::Model& outModel) const;

  /// Load an environment map / scene by path; these back --hdrfile / --scenefile, which load on
  /// change. False if the file does not exist or the renderer is not attached yet (the start-up
  /// parse, where main() does the load itself). Application thread only.
  [[nodiscard]] bool loadHdrEnvironment(const std::filesystem::path& filename);
  /// Install a lat-long decoded elsewhere -- loadEnvImage() on a worker, or one generated in memory
  /// -- and switch to the HDR environment, as the path overload does after its own decode. `name`
  /// is the file it came from, or empty for a generated image (kept in memory then, so leaving a
  /// baked sky can restore it). False for an invalid image. Application thread only.
  [[nodiscard]] bool loadHdrEnvironment(EnvImage&& image, const std::filesystem::path& name) override;
  /// Scene bounds as loaded (world space), or an empty box with no scene.
  [[nodiscard]] nvutils::Bbox sceneBounds() const;
  [[nodiscard]] bool loadSceneFile(const std::filesystem::path& filename);
  // Decodes an image file now, on the calling thread, for the next scene replace to take ready
  // (SceneVk::prefetchImage): a host's load and wear workers call it for the textures they resolved.
  void prefetchImage(const std::filesystem::path& diskPath, bool srgb) override;
  // Max texture size (HostRenderer): the images decoded from now on. Any thread.
  void setMaxTextureExtent(uint32_t extent) override { m_maxTextureExtent.store(extent); }
  void setNoticeSink(std::function<void(const std::string&)> sink) override { m_noticeSink = std::move(sink); }

  /// The path tracer or the rasterizer, as the renderSystem parameter switches them.
  void          setRenderSystem(RenderingMode mode) override;
  RenderingMode renderSystem() const override { return m_resources.settings.renderSystem; }

  // The tonemapped output and its size, for the headless frame stream (src/frame_stream.hpp). The
  // image is in VK_IMAGE_LAYOUT_GENERAL once onRender() has recorded the frame.
  VkImage    tonemappedImage() const { return m_resources.gBuffers.getColorImage(Resources::eImgTonemapped); }
  // The allocator, for a host element that owns GPU work of its own (the dForce GPU solver)
  nvvk::ResourceAllocator& allocator() { return m_resources.allocator; }
  nvvk::ResourceAllocator* deviceAllocator() override { return &m_resources.allocator; }
  VkExtent2D renderSize() const override { return m_resources.gBuffers.getSize(); }

  // The device's VRAM budget (VK_EXT_memory_budget): the bytes the driver allows this process over
  // its device-local heaps and the bytes in use; `known` is false on a device without the
  // extension. A physical-device query: any thread.
  MemoryBudget deviceMemoryBudget() const override;
  // What a pixel of the frame costs in the mode drawn, an estimate of the per-pixel buffers (the
  // G-buffers; under RTX the DLSS guides and output, OptiX's images and shared buffers, the hand-off
  // and the upscaler's own history). Updated each frame; any thread.
  uint32_t frameBytesPerPixel() const override { return m_frameBytesPerPixel.load(); }
  // Why a frame of this size must not be asked for (the budget cannot hold what it adds over the
  // frame drawn now), or empty. Any allocation that fails ends the process (NVVK_CHECK), so the
  // stream's /size asks here first. Any thread.
  // `extraBytesPerPixel`: what else a pixel of the frame costs outside the renderer (native presentation)
  std::string refuseFrameSize(VkExtent2D size, double extraBytesPerPixel = 0.0) const;
  // The bytes of new textures that fit in GPU memory now, negative when it is over; none where the
  // budget is unknown
  std::optional<int64_t> sceneHeadroom() const;
  std::string            refuseGeometry(uint64_t bytes) override;
  struct AllocatedMemory
  {
    uint64_t blocks = 0, allocations = 0;
  };
  AllocatedMemory allocatedMemory() const;
  // Frames that ran a renderer so far (Resources::drawnFrameCount): a count that stands still is a
  // view with nothing to draw, which the frame stream slows the loop for.
  uint64_t drawnFrameCount() const { return m_resources.drawnFrameCount; }
  // The frame's profile as a JSON object for the viewport's stats (the frame stream's /stats):
  // `sections`, the profiler's frame sections in the order they ran (name, nesting level, CPU and
  // GPU ms averaged over its recent frames), and the device-local memory in use and its budget, in
  // MB. Application thread.
  std::string frameProfileJson() const;
  // The profiler's GPU time of the last frame it measured, ms (0 before the first)
  double frameGpuMs() const;

  /// The path tracer is accumulating a still: the camera and scene have held for a frame, and
  /// the image does not yet hold maxSamples samples per pixel. Every further frame improves it.
  bool isRefiningStill() const override { return isStill() && !m_resources.accumulationComplete(); }
  // The path tracer is accumulating a still, or holds a finished one: every frame since the view
  // stopped added samples to one image. Not while DLSS renders (every frame without an accumulation
  // denoiser, or the hand-off delay's frames with one): its frames rebuild the image each time.
  bool isStill() const override
  {
    return m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_resources.frameCount >= 1
           && !m_pathTracer.dlssFrame();
  }
  bool isStillConverged() const override { return isStill() && m_resources.accumulationComplete(); }
  void saveImage(const std::filesystem::path& path) override
  {
    if(m_app)
      m_screenshotPending = path;
  }
  bool imageSavePending() const override { return !m_screenshotPending.empty(); }
  // The sample cap is reached, whatever renders (a DLSS view counts its frames too)
  bool accumulationComplete() const override { return m_resources.accumulationComplete(); }
  int  accumulatedSamples() const override { return m_resources.accumulatedSamples; }
  int  sampleCap() const override { return m_resources.settings.maxSamples; }
  bool driverCacheFull() const override { return m_pathTracer.driverCacheFull(); }

  // Live-edit surface for a host application (src/mcp_scene.cpp). Application thread only.
  nvvkgltf::Scene*  editScene() { return m_resources.getScene(); }
  // The retained scene draws (a stream host feeds it, below): there is no CPU scene above, and
  // mcp_scene's edit tools are refused.
  bool drawsRetainedScene() const override { return m_resources.retainedGpu != nullptr; }

  // ---- A stream host's own retained scene (daz-viewer docs/retained-scene-plan.md, P4): the host
  // feeds the store, the renderer keeps it resident and draws it, with no CPU glTF scene beside it.
  // The store lives for the session, so its handles stay unique. Application thread only.
  rs::RetainedScene& hostScene() override;
  bool               feedsHostScene() const override { return m_resources.hostFed; }
  // A scene replace of what the host fed (a composition): waits for the device, drops the pending
  // picks, and makes the store resident whole. The selection stays where its entity still is.
  void replaceHostScene() override;
  // The host's selection: an entity, drawn outlined with every instance under it; none clears it
  void         selectEntity(rs::EntityId entity) override;
  rs::EntityId selectedEntity() const override { return m_selectedEntity; }
  // The clip that plays (none: none), `at` seconds from its start; AnimationControl plays, pauses,
  // loops and speeds it as it does a legacy clip
  void       setHostClip(rs::ClipId clip, float at = 0.f) override;
  rs::ClipId hostClip() const { return m_hostClip.clip(); }
  // its time and length, from its start
  float hostClipTime() const override { return m_hostClip.time() - m_hostClip.start(); }
  float hostClipDuration() const override { return std::max(0.f, m_hostClip.end() - m_hostClip.start()); }
  // to `time` from its start, paused, shown at once
  void              seekHostClip(float time) override;
  AnimationControl& editAnimationControl() override { return m_resources.animationControl; }
  bool              isBlasUpdatable(uint32_t renderPrimID) const { return m_resources.sceneRtx.isBlasUpdatable(renderPrimID); }
  // --editable (implied by --stream): every BLAS keeps ALLOW_UPDATE and is never compacted, so
  // geometry edits can refit in place. Must be set before the scene loads.
  void setEditable(bool editable) { m_editable = editable; }
  // A --stream host (the Figure Viewer): its frames go out through the frame stream, so the ImGui
  // panels nobody sees are not built. Before the first frame.
  void setStreamHost(bool streamHost) { m_streamHost = streamHost; }
  // --cacheDir. Empty keeps the pipeline cache and compiled SPIR-V next to the executable.
  // Copied into the path tracer before onAttach.
  void setCacheDir(const std::filesystem::path& dir) { m_cacheDir = dir; }
  bool isEditable() const { return m_editable; }
  // Bumped on every scene load, so a host's cached ids can tell scenes apart.
  uint64_t sceneGeneration() const { return m_sceneGeneration; }
  /// Write the current scene to `filename`; backs --savefile, so it is reachable from the command
  /// line, a benchmark sequence and nvpro_set_parameters through the same single registration the
  /// load parameters use. Application thread only: it reads scene and model state. The view
  /// becomes the scene's first camera, as File > Save has it. (A host's scene export builds its own
  /// model instead: FvHost's export.glb.)
  bool save(const std::filesystem::path& filename, bool selfContained = false);

  /// Ray-pick at a normalized viewport position (0..1, origin top-left): the viewport click path
  /// without ImGui. `center` recenters the camera on the hit instead of selecting. The result is
  /// applied a frame or two later, once the pick's frame has finished (see m_pendingClickPick).
  void pickAt(glm::vec2 uv, bool center) override;
  /// Shaded, clay, normals and the other debug views (shaderio::Visualization).
  void setVisualization(shaderio::Visualization mode) override;
  void setWireframe(bool on) override;
  void setShowGrid(bool on) override;
  void setShowAxis(bool on) override;
  /// Path-tracer depth of field. `aperture` is the lens radius squared in metres (0: pinhole).
  /// `autoFocus` keeps the orbit target sharp and ignores `focalDistance`.
  void setDepthOfField(float focalDistance, float aperture, bool autoFocus, int blades = 0, float bladeRotation = 0) override;
  int  selectedNodeIndex();  // The node of a node or primitive selection, or -1
  void fitCamera(bool wholeScene) override;
  /// What is shown, where it is now: the union of the visible render nodes' boxes through their
  /// current node transforms (getSceneBounds is the scene as it loaded).
  nvutils::Bbox visibleSceneBounds() override;
  void selectSceneNodeFromScript(int nodeIndex);  // Script-driven selection (Inspector/Scene Browser)

  /// Lets a stream host own what the transform gizmo edits, on its scene's entities. `attach` gives
  /// the handles the selected entity shows (TransformHelperVk::ShowRotation, ShowTranslation, ...),
  /// 0 for no gizmo on it; asked every frame, and new handles attach at once. `edit` gets the
  /// entity's new local translation, rotation and scale, which the host writes; `done` on the release
  /// that ends the drag. (A windowed session edits the glTF node selected, with undo.)
  void setGizmoHooks(GizmoHooks hooks) override;
  void setShowGizmo(bool on) override;
  /// World-space line segments a host draws over the image (two vertices each), for what has no
  /// mesh: a camera's frustum, a light's marker. Drawn in both renderers, never in the path-traced
  /// image itself. Application thread only.
  void setHelperLines(std::vector<shaderio::lines::LineVertex> vertices) override;
  /// The transform gizmo's pointer input without ImGui (a headless stream): `px` in render pixels,
  /// origin top-left. True when the gizmo took the event (a press on a handle, a drag, its release),
  /// so the camera and click selection should not see it.
  bool gizmoInput(glm::vec2 px, bool down, bool pressed, bool released) override;
  /// Sees every click pick of a stream host's scene first: what was hit (none for nothing) and the
  /// hit point in world space. Returning true keeps the renderer from changing the selection;
  /// otherwise the entity hit is selected, or the selection cleared (a hit on it again clears it too).
  void setPickObserver(std::function<bool(const std::optional<InstancePick>& pick, const glm::vec3& hit)> observer) override;

#ifdef USE_NVMCP
  /// Automation surface for the optional MCP endpoint (src/mcp_timing.cpp), which is where both
  /// of these are defined. Thread-safe: they only read the profiler's mutex-guarded snapshots.
  [[nodiscard]] std::vector<std::string> profilerTimerNames() const;
  /// Blocks for `warmup + frames` rendered frames, so it must be called off the application
  /// thread -- that is the thread producing the frames it waits on.
  [[nodiscard]] TimerMeasurement measureTimer(const std::string& name, int warmup, int frames) const;
  /// Several timers over the same rendered frames, one result per name in order. A timer that does
  /// not tick in those frames gets an error once the others are done, not after a timeout.
  [[nodiscard]] std::vector<TimerMeasurement> measureTimers(const std::vector<std::string>& names, int warmup, int frames) const;
#endif

private:
  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onLastHeadlessFrame() override;
  void onRender(VkCommandBuffer cmd) override;
  void onPostRender() override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onUIRender() override;

  bool updateFrameCounter();

  void clearGbuffer(VkCommandBuffer cmd);
  // Drops the current scene. keepImagesForReuse retains file-backed GPU images (scenefile replace)
  // so the following create() can adopt unchanged files; every other caller frees them.
  void cleanupScene(bool keepImagesForReuse = false);
  void rebuildSceneFromModel();  // Rebuild Vulkan scene after modifying the glTF model in-place (preserves textures); clears undo
  void rebuildSceneGeometry();  // Geometry-only rebuild (preserves textures); does NOT clear undo (used by undoable geometry edits)
  void reconcileGeometryIfNeeded();  // Rebuild geometry when GPU buffers are behind the render-primitive count (e.g. added primitive)
  void applyPendingTextureRebuild();  // Consume DirtyFlags::texturesChanged at frame top: full texture rebuild + cache refresh
  void applyPendingTextureTailSync();  // Consume DirtyFlags::texturesTailChanged at frame top: incremental append/remove of tail textures
  void applyPendingSamplerUpdate();  // Consume DirtyFlags::samplers at frame top: in-place VkSampler update, no image touch
  void refreshCpuSceneGraphFromModel();
  void rebuildVulkanSceneInternal(nvvkgltf::SceneGpu::RebuildMode mode);  // GPU upload + AS; CPU scene must already be parsed
  bool compileShaders();
  void createDescriptorSets();
  void createResourceBuffers();
  void createVulkanScene();
  // The retained scene's per-frame half (replaceHostScene() is its scene replace), in place of the
  // legacy per-frame GPU paths
  bool                       syncRetainedScene(VkCommandBuffer cmd);
  // a stream host's scene: the clip moved on and written into the store; true when it wrote anything
  bool advanceHostClip();
  bool afterRetainedSync(const rs::SyncResult& result);
  // the instances under the selected entity, outlined; the selection cleared when its entity went
  void refreshEntitySelection();
  // the gizmo on a stream host's scene (updateGizmoAttachment's half for it)
  void updateEntityGizmo();
  // the instance slots of a stream host's selection, and an instance's box in the world
  nvutils::Bbox              instanceWorldBounds(uint32_t slot) const;
  void                       logRetainedNotices();
  // The variant's scene features from the retained store (the materials live instances draw with),
  // grown only, when `grow`, until the next scene replace (D15)
  void updateRetainedFeatures(bool grow);
  // What the residency may draw: the live path-tracer variant's features, or every one
  std::optional<nvvkgltf::SceneFeatureSet> drawableFeatures();
  rs::VulkanGpuBackend::Info retainedBackendInfo();
  void finalizeSceneSetup(const std::filesystem::path& filename);  // Shared GPU build + UI wiring after a load
  void buildAccelerationStructures();                              // Helper for BLAS/TLAS building
  void destroyResources();
  void resetFrame();
  // Script-driven *ray-pick* selection (`pickrendernode`): selects the given glTF node's
  // first render primitive the same way a real 3D-viewport click would - unlike
  // selectSceneNodeFromScript() above, this supplies finite (camera-eye/node-position) selectionPoint/
  // selectionRayOrigin values, so KHR_interactivity event/onSelect's ray-info output sockets can be
  // exercised end-to-end without an actual mouse click.
  void pickSceneNodeFromScript(int nodeIndex);
  void silhouette(VkCommandBuffer cmd);
  bool tonemap(VkCommandBuffer cmd);
  bool runTonemapPass(VkCommandBuffer cmd);
  void renderVisualHelpers(VkCommandBuffer cmd);
#if defined(USE_DLSS)
  Dlss*       activeDlss();
  const Dlss* activeDlss() const;
#endif

  bool dlssGuideRequired() const;  // True when the path tracer currently needs DLSS/OptiX/OIDN guide-buffer capture code.
  float defaultFireflyClamp() const;  // Scene-appropriate firefly clamp: HDR luminance integral, else a fixed baseline.
  void  updateGizmoAttachment();
  bool  updateTextures();
  // Write a contiguous range of scene texture / sampler descriptors (eTextures / eSamplers). updateTextures()
  // writes the whole set; applyPendingTextureTailSync() writes only the newly appended slots.
  bool writeTextureDescriptorRange(uint32_t firstTexture, uint32_t textureCount, uint32_t firstSampler, uint32_t samplerCount);
  void updateHdrImages();

  bool updateSceneChanges(VkCommandBuffer cmd);
  bool updateAnimation(VkCommandBuffer cmd);
  // Shared downstream GPU reconciliation after CPU-side animation-channel evaluation dirtied the
  // model (world matrices, GPU sync, morph/skin compute, BLAS/TLAS update, dirty-flag clear) - the
  // tail of updateAnimation().
  void reconcileAnimationGpuState(VkCommandBuffer cmd);
  // #DLSS deformation motion: keep the deforming primitives' positions from before this frame's
  // writers (see PrevPositionsVk). Runs before any of them.
  void snapshotPrevPositions(VkCommandBuffer cmd);
  bool                      evaluateClip(nvvkgltf::Scene& scn, VkCommandBuffer cmd);
  bool                      m_editable{false};
  bool                      m_streamHost{false};  // see setStreamHost()
  bool m_pendingAnimReconcile{false};             // skin and morph again (a tangent upload put the base geometry back)
  bool                      m_verticesWrittenLastFrame{false};  // See snapshotPrevPositions()
  bool                      m_prevPositionsStale{false};        // copies skipped while DLSS was off
  uint64_t                  m_sceneGeneration{0};
  // Polls m_pendingClickResult's frame semaphore and, once signaled, calls m_rayPicker.getResult()
  // and applies it via applyClickPickResult() - called once per frame.
  void updateClickPickState();
  // Same for the camera anchor pick: answers m_cameraManip->resolveAnchor() once its frame has signaled.
  void updateAnchorPickState();
  // AnchorCameraManipulator::AnchorRequest - queues m_pendingAnchorPick for the next onRender().
  bool requestAnchorPick(const glm::dvec2& screenPos);
  // Applies a completed ray-pick result: selection (and, on a double click, camera recenter) -
  // the exact logic mouseClickedInViewport() used to run inline, now one to two frames removed
  // from the click itself (see m_pendingClickPick's comment for why).
  void applyClickPickResult(const nvvk::RayPicker::PickResult& pickResult, bool isDoubleClick);

  // Headless / scripted benchmark (shared automation paths)
  [[nodiscard]] bool                             isHeadlessMode() const;
  [[nodiscard]] bool                             isAutomatedRun() const;
  void                                           saveHeadlessOutputImage();
  void saveViewportImage(const std::filesystem::path& filename);  // .hdr: linear render, else tonemapped
  std::filesystem::path headlessOutputPathOrDefault() const;
  void                  logDenoiserTimings() const;  // DENOISER_TIMING lines (headless)
  // --outputSamples: the samples per pixel of the image shown, and which one it is (DisplayBuffer)
  int displayedImageSamples(DisplayBuffer& source) const;
  void refreshSampleTargets();  // Before the frame renders: parses the counts, and hands them to the path tracer as stops
  void updateSampleCaptures();  // After the frame is recorded: schedules a save when a requested count is reached

  // updateSceneChanges phase helpers (keep main function readable)
  void updateSceneChanges_BlasRebuild(const nvvkgltf::Scene::DirtyFlags& df);
  void updateSceneChanges_NodeTransforms(VkCommandBuffer cmd, nvvkgltf::Scene* scene, const nvvkgltf::Scene::DirtyFlags& df);
  uint32_t updateSceneChanges_SyncGpuBuffers(VkCommandBuffer cmd, nvvkgltf::Scene* scene);
  void     updateSceneChanges_TlasUpdate(VkCommandBuffer cmd, nvvkgltf::Scene* scene);
  void     markGpuTransformStaleAfterCpuSync(const nvvkgltf::Scene& scene);
  void     updateSceneChanges_RasterizerInvalidate(bool renderNodeOrNodeDirty);
  void     updateSceneChanges_TangentUpload(VkCommandBuffer cmd, nvvkgltf::Scene* scene, bool& changed);
  void     updateSceneChanges_Finalize(VkCommandBuffer cmd, bool changed, bool stagingFlushed, nvvkgltf::Scene* scene);

  // Selection (renderer_selection.cpp)
  void onSceneSelectionEvent(const SceneSelection::Event& event);

  // UI
  bool hasValidScene();
  // Windows > Reset UI Layout / Reset All to Default. Both are deferred to the top of the next UI
  // bar is mid-submission, and so the panels pick the new state up before they are submitted.
  // F11 / Esc / --presentationMode: bring PresentationMode in line with the request. Top of the UI pass.
  // `selectionPoint`/`selectionRayOrigin` (global space) are the real ray-pick hit point/origin -
  // default NaN for callers with no ray (e.g. script-driven test commands), matching the
  // spec-sanctioned "no ray info" fallback (see SceneSelection::Event's doc comment).
  void          updateSelectionFromPick(int              renderNodeIdx,
                                        const glm::vec3& selectionPoint = glm::vec3(std::numeric_limits<float>::quiet_NaN()),
                                        const glm::vec3& selectionRayOrigin = glm::vec3(std::numeric_limits<float>::quiet_NaN()));
  nvutils::Bbox getRenderNodeBbox(int renderNodeIndex);
  nvutils::Bbox getRenderNodesBbox(const std::unordered_set<int>& renderNodeIndices);
  /// The `.sky.json` pickers behind the Environment panel's two preset buttons.

  // Opens an image file dialog for the inspector's "Load from file" texture action. Returns the chosen
  // path, or an empty path if cancelled.

  // Resolve a glTF texture / image index to a bounded ImGui thumbnail (0 if not resident). Handed to
  // the inspector and scene browser so their panels can show thumbnails without knowing about the GPU.

  // Push a transient notification to the on-screen toast overlay (isError = red). Wired into the
  // inspector / scene browser so their edit actions can surface failures beyond the log.
  void notify(const std::string& message, bool isError);

  // Recent files management

  // File dialog directories
  std::filesystem::path m_lastSceneDirectory;
  std::filesystem::path m_lastHdrDirectory;
  std::filesystem::path m_lastSkyPresetDirectory;  // where the last preset was read or written
  std::filesystem::path m_lastImageDirectory;
  // Default filename offered by the Save Image / Save Screen Image dialogs: the loaded scene's
  // name (.jpg) until the user saves under a different name, reset on New Scene / new scene load.
  std::filesystem::path m_imageSaveFilename;

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*        m_app{};               // Application pointer
  VkDevice                   m_device{};            // Convenient
  nvvk::RayPicker            m_rayPicker{};         // Ray picker
  nvvk::RayPicker            m_anchorPicker{};      // Ray picker for the camera anchor (see m_pendingAnchorPick)
  nvutils::ProfilerTimeline* m_profilerTimeline{};  // Timeline profiler
#ifdef USE_NVMCP
  // Guards the *lifetime* of m_profilerTimeline (not its contents, which are already mutex-guarded
  // inside the profiler). Elements detach in registration order, so this renderer destroys the
  // timeline while the MCP element is still serving: a worker thread can be inside a profiler read
  // at that moment. Held for one profiler call at a time, never across a wait for a frame.
  mutable std::mutex m_profilerTimelineMutex;
  // Waits out a UI-initiated load before a measurement; the reason it cannot measure, or empty.
  [[nodiscard]] std::string waitUntilMeasurable() const;
#endif
  nvvk::ProfilerGpuTimer                   m_profilerGpuTimer{};  // GPU profiler
  std::shared_ptr<AnchorCameraManipulator> m_cameraManip;         // Camera manipulator

  nvutils::PerformanceTimer m_cpuTimer;               // CPU performance timer (path-tracer accumulation window)
  bool                      m_cpuTimePrinted{false};  // Track if CPU time has been printed

#ifndef NDEBUG
  bool m_validateGpuSync{true};
  bool m_skipGpuSyncValidation{false};  // GPU transform path skipped uploadRenderNodes / CPU TLAS sync
#endif

  uint32_t m_maxTextures{100'000U};  // Maximum number of material images (eTextures SAMPLED_IMAGE array)
  uint32_t m_maxSamplers{0U};        // Maximum number of samplers (eSamplers SAMPLER array); set from device limits

  Resources             m_resources;
  std::filesystem::path m_cacheDir;   // --cacheDir, applied to the path tracer before onAttach
  PathTracer            m_pathTracer;  // Path tracer renderer
  Rasterizer m_rasterizer;  // Rasterizer renderer

  // New Scene Browser system (parallel implementation)
  SceneSelection m_sceneSelection;  // Shared selection state
  SkySun         m_skySun;          // Who the sun is, and where it points

  BusyState    m_busy;
  Silhouette    m_silhouette;     // Silhouette renderer
  VisualHelpers m_visualHelpers;  // Grid + transform gizmo overlay
  EnvBaker      m_envBaker;       // Bakes analytic skies into the shared lighting environment
  SkyBruneton   m_skyBruneton;    // Precomputed atmospheric scattering LUTs
  // Which environment the firefly clamp was last calibrated for, as an int so it can hold a
  // "none yet" sentinel. The clamp is a starting point the user is free to override, so it is
  // recalibrated when the environment changes *kind* and not on every re-bake.
  int  m_fireflyClampCalibratedFor{-1};
  bool m_envBakeDirty{true};  // Baked environment needs rebuilding before the next frame
  // Set when a Time of Day value arrives from outside the UI -- the command line, a benchmark
  // sequence or MCP. It decides a precedence question the ini restore would otherwise settle
  // wrongly: see the post-restore hook beside the tod* declarations.
  bool m_todDrivesSun{false};
  // --animPlay: whether animations play. Re-applied to animationControl.play on every scene load,
  // which resets the animation state. The UI's Play/Pause writes it, so a pause carries over to
  // the next scene and, persisted, to the next session. Off by default in headless runs
  // (applyHeadlessDefaults).
  bool m_animPlay{true};
  // --outputSamples: save the output image when the displayed image reaches these samples per pixel
  std::string m_outputSamples;
  struct SampleCapture
  {
    std::string           parsedFrom;       // m_outputSamples the targets were parsed from
    std::vector<int>      targets;          // Sorted, unique
    size_t                next{0};          // First target not saved yet
    int                   lastSamples{-1};  // Displayed image of the previous frame
    int                   lastSource{-1};   // DisplayBuffer of the previous frame
    std::filesystem::path pending;          // Saved in onPostRender, once the frame is submitted
  } m_sampleCapture;
  // `screenshot`: saved in onPostRender from a frame drawn without the selection silhouette, the
  // grid, the gizmo and the helper lines, which otherwise go into the tonemapped image
  std::filesystem::path m_screenshotPending;
  std::filesystem::path m_screenshotParam;  // --screenshot, taken up by m_screenshotPending
  // for refuseFrameSize on another thread: the frame's per-pixel cost and size, written each frame
  std::atomic<uint32_t> m_frameBytesPerPixel{0};
  std::atomic<uint64_t> m_frameSizePacked{0};  // width << 32 | height
  bool                  m_memoryBudgetExtension = false;  // the device has VK_EXT_memory_budget (onAttach)
  std::atomic<uint32_t> m_maxTextureExtent{0};
  uint64_t              m_pendingGeometry = 0;  // accepted by refuseGeometry and not resident yet
  std::function<void(const std::string&)> m_noticeSink;
  bool                  m_envPreviewPending{false};  // Record a colour-only re-bake into this frame
  bool                  m_envCommitOwed{false};      // Previewed at least once; owes a commit once the edits stop
  bool                  m_envOwnedByBaker{false};    // hdrIbl currently points at EnvBaker's image, not a loaded file
  std::filesystem::path m_lastHdrFile;               // Last HDR handed to createHDR, replayed when returning to eHdr
  std::shared_ptr<const EnvImage> m_lastHdrImage;    // The last HDR when no file backs it (generated), replayed likewise
  EnvironmentState      m_environmentSky;            // Sky the loaded scene authors, preserved verbatim for save
  bool                  m_environmentSkyPending{false};  // Authored sky parsed but not yet pushed to settings
  std::filesystem::path m_pendingLoadSkyPreset;          // Start-up --loadSkyPreset, applied at the frame top
  std::filesystem::path m_pendingSaveSkyPreset;          // Start-up --saveSkyPreset, written after the load

  // Undo/Redo

  // Application-level delete confirmation (shared with scene browser via pointers)

  // Gizmo TRS local storage (pointers passed to TransformHelperVk)
  glm::vec3 m_gizmoPosition{0.f};
  glm::vec3 m_gizmoRotation{0.f};  // Euler angles in degrees
  glm::vec3 m_gizmoScale{1.f};
  glm::mat4 m_gizmoParentWorldMatrix{1.f};
  int       m_gizmoNodeIndex = -1;
  uint32_t  m_gizmoFlags     = 0;
  GizmoHooks m_gizmoHooks;
  glm::vec2  m_gizmoInputLast{0.f};
  std::function<bool(const std::optional<InstancePick>&, const glm::vec3& hit)> m_pickObserver;
  // a stream host's scene: the entity selected and the one the gizmo is on, and the clip playing
  rs::EntityId   m_selectedEntity;
  rs::EntityId   m_gizmoEntity;
  rs::ClipPlayer m_hostClip;
  // Set in windowed sessions only: the Scene Browser scrolls to a new selection

  // Gizmo TRS snapshot for undo (captured at drag start)
  glm::vec3 m_gizmoSnapshotT{0.f};
  glm::quat m_gizmoSnapshotR{1, 0, 0, 0};
  glm::vec3 m_gizmoSnapshotS{1.f};

  // Async click ray-pick (avoids the ~300ms stall of the old synchronous submitAndWaitTempCmdBuffer
  // path). A click captured during the UI phase
  // (mouseClickedInViewport(), ui_renderer.cpp) becomes a PendingClickPick with everything the pick
  // needs baked in *at click time* (camera state, normalized cursor position, single/double-click
  // classification) - no GPU work yet. onRender(cmd) then records m_rayPicker.run(cmd, ...) into the
  // frame's own command buffer (the TLAS is already proven
  // valid there) and moves it into a PendingClickResult carrying just the frame semaphore to poll.
  // updateClickPickState() polls that semaphore non-blocking and,
  // once signaled, reads m_rayPicker.getResult() and applies it via applyClickPickResult(). A newer
  // click simply overwrites whichever of these is pending - nvvk::RayPicker has one internal result
  // buffer, so only the most recent run() before a getResult() call is ever meaningful anyway.
  struct PendingClickPick
  {
    glm::mat4 modelViewInv;
    glm::mat4 perspectiveInv;
    int       isOrthographic;
    glm::vec2 pickPos;
    bool      isDoubleClick;
  };
  std::optional<PendingClickPick> m_pendingClickPick;
  struct PendingClickResult
  {
    nvvk::SemaphoreState semaphoreState;
    bool                 isDoubleClick;
  };
  std::optional<PendingClickResult> m_pendingClickResult;

  // Async ray-pick of the camera anchor (Shift+drag, see camera_controls.hpp), on the same pattern as
  // the click pick above but with its own picker, since a Shift+click also queues a selection pick.
  // The answer goes to AnchorCameraManipulator::resolveAnchor(), and a dropped pick must answer too,
  // or the drag waiting on it stays frozen. A new request drops the answer of one still in flight.
  std::optional<nvvk::RayPicker::PickInfo> m_pendingAnchorPick;
  std::optional<nvvk::SemaphoreState>      m_pendingAnchorResult;

  // Non-blocking GPU loading pipeline (see timeline_pipeline.hpp for details).
  // Worker threads enqueue command buffers; the main thread calls poll() each frame.
  TimelinePipeline m_loadPipeline;

  glm::mat4 m_prevMVP{1.f};  // Previous MVP matrix for motion vectors

  VkCommandPool m_transientCmdPool{};  // Command pool for transient command buffers

  SettingsRegistry                m_settings;           // Single declaration point for every setting
  nvgui::SettingsHandler          m_settingsHandler;    // Settings handler for ImGui.ini
  const nvutils::ParameterParser* m_parameterParser{};  // CLI parameter parser, for INI load filtering (see wasParsed)
  // ImGui.ini restore runs in Application::run() *after* onAttach and writes storage directly, so
  // it bypasses the per-setting callbackSuccess. Run any opted-in post-restore hooks once on the
  // first frame to refresh derived state (e.g. Resources::sunDirection from the restored
  // sunAzimuth/Elevation).
  bool m_pendingRestoreCallbacks{true};

  // Requested from the Windows menu, consumed by applyPendingResets() at the top of the UI pass.

  // Presentation mode. The registered parameter (--presentationMode, MCP) is only the request;
  // updatePresentationMode() applies it and writes the actual state back, so every path -- F11,
  // Esc, the View menu, the command line, MCP -- goes through the same enter/leave.



};
