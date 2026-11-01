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

// GltfRenderer's scene: a file or the host's retained scene going in, its sync each frame, its clip.

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>

#include "renderer.hpp"
#include "fv_compose.hpp"
#include "scene_import_gltf.hpp"

#include <figure/io/timing.hpp>
#include "gltf_environment_sky.hpp"
#include "tinygltf_utils.hpp"
#include "utils.hpp"
#include "animation_control.hpp"

extern nvutils::ProfilerManager g_profilerManager;  // #PROFILER

namespace {

// Two timestamps around work that runs once (a load's BLAS or TLAS build), read when the GPU has
// finished it: the profiler's frame sections see only the work of the frames being drawn. Held by
// the completion callback, so its query pool goes with the callback, or with the pipeline's clear()
// when the work is dropped (after a queue wait, so never while in use).
class OneShotGpuTimer
{
public:
  OneShotGpuTimer(VkDevice device, VkPhysicalDevice physicalDevice)
      : m_device(device)
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    m_msPerTick = double(props.limits.timestampPeriod) * 1e-6;
    const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
    NVVK_CHECK(vkCreateQueryPool(device, &info, nullptr, &m_pool));
  }
  ~OneShotGpuTimer() { vkDestroyQueryPool(m_device, m_pool, nullptr); }
  OneShotGpuTimer(const OneShotGpuTimer&)            = delete;
  OneShotGpuTimer& operator=(const OneShotGpuTimer&) = delete;

  void begin(VkCommandBuffer cmd) const
  {
    vkCmdResetQueryPool(cmd, m_pool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_pool, 0);
  }
  void end(VkCommandBuffer cmd) const { vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pool, 1); }

  // Once the command buffer has completed; 0 if the results are not there.
  double ms() const
  {
    uint64_t stamps[2]{};
    if(vkGetQueryPoolResults(m_device, m_pool, 0, 2, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
      return 0.0;
    return double(stamps[1] - stamps[0]) * m_msPerTick;
  }

  // An [fv-time] line with three decimals, since a TLAS build of a few hundred instances is
  // hundredths of a millisecond (fv::timing::report rounds to one).
  static void report(const char* name, double ms)
  {
    char text[96];
    std::snprintf(text, sizeof(text), "%s %.3f ms", name, ms);
    fv::timing::line(text);
  }

private:
  VkDevice    m_device{};
  VkQueryPool m_pool{};
  double      m_msPerTick{};
};

}  // namespace

//--------------------------------------------------------------------------------------------------
// Load the scene
bool GltfRenderer::createScene(const std::filesystem::path& sceneFilename)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  m_resources.selectedRenderNodes.clear();

  if(sceneFilename.empty())
  {
    return false;
  }

  std::filesystem::path filename = nvutils::findFile(sceneFilename, nvsamples::getResourcesDirs(), false);
  if(!filename.has_filename())
  {
    LOGW("Cannot find file: %s\n", nvutils::utf8FromPath(sceneFilename).c_str());
    return false;
  }

  // The file goes the way a prop's does: read, imported into the store, uploaded
  LOGI("Loading scene: %s\n", nvutils::utf8FromPath(filename).c_str());
  tinygltf::Model model;
  std::string     error;
  if(!fv::composer::readModelFile(filename, model, error))
  {
    LOGW("Error loading scene: %s: %s\n", nvutils::utf8FromPath(filename).c_str(), error.c_str());
    return false;
  }
  m_resources.retainedScene = std::make_unique<rs::RetainedScene>();
  m_resources.hostFed       = true;
  m_resources.sceneFromFile = true;
  rs::importGltf(*m_resources.retainedScene, model, {.prop = 1, .imageSearchPaths = {filename.parent_path()}});
  replaceHostScene();
  placeCameraFromFile(model);
  return true;
}

// The file's first camera, or a fit of the scene when it has none
void GltfRenderer::placeCameraFromFile(const tinygltf::Model& model)
{
  const nvutils::Bbox bounds = m_resources.gpuView().bounds();
  bool                placed = false;
  const auto          place  = [&](int nodeId, const glm::mat4& world) {
    const tinygltf::Node&   node = model.nodes[size_t(nodeId)];
    const tinygltf::Camera& from = model.cameras[size_t(node.camera)];
    const bool              orthographic = from.type != "perspective";

    nvutils::CameraManipulator::Camera camera;
    camera.projectionType = orthographic ? nvutils::CameraManipulator::ProjectionType::Orthographic :
                                           nvutils::CameraManipulator::ProjectionType::Perspective;
    double zNear = orthographic ? from.orthographic.znear : from.perspective.znear;
    double zFar  = orthographic ? from.orthographic.zfar : from.perspective.zfar;
    if(zFar <= zNear)
      zFar = std::max(zNear * 2.0, 4.0 * double(bounds.radius()));
    camera.nearFar = {zNear, zFar};
    if(orthographic)
    {
      camera.orthMag = {from.orthographic.xmag, from.orthographic.ymag};
      camera.fov     = 45.0;
    }
    else
      camera.fov = glm::degrees(from.perspective.yfov);

    // looking down the node's -Z, as far as the scene's centre
    const glm::dmat4 m(world);
    const glm::dvec3 eye     = glm::dvec3(m[3]);
    const glm::dvec3 forward = -glm::dmat3(m) * glm::dvec3(0.0, 0.0, 1.0);
    camera.eye               = eye;
    camera.ctr               = eye + std::abs(glm::dot(glm::dvec3(bounds.center()) - eye, forward)) * forward;
    camera.up                = glm::dvec3(0.0, 1.0, 0.0);
    if(node.extras.IsObject())
    {
      tinygltf::utils::getArrayValue(node.extras, "camera::eye", camera.eye);
      tinygltf::utils::getArrayValue(node.extras, "camera::center", camera.ctr);
      tinygltf::utils::getArrayValue(node.extras, "camera::up", camera.up);
    }
    m_cameraManip->setCamera(camera);
    placed = true;
    return true;
  };
  if(!model.scenes.empty())
    for(int root : model.scenes[size_t(std::max(model.defaultScene, 0))].nodes)
      if(!placed)
        tinygltf::utils::traverseSceneGraph(model, root, glm::mat4(1), place);
  if(!placed && !bounds.isEmpty())
  {
    m_cameraManip->fit(bounds.min(), bounds.max(), true);
    m_cameraManip->setClipPlanes(glm::vec2(0.001F * bounds.radius(), 100.0F * bounds.radius()));
  }
}

//--------------------------------------------------------------------------------------------------
// Helper function to cleanup the current scene
//
void GltfRenderer::cleanupScene(bool keepImagesForReuse)
{
  // Drop any renderer-side state tied to the outgoing Scene BEFORE the unique_ptr is reset.
  // The heap allocator is free to hand the same address back to the next Scene instance, so
  // any Scene-pointer-based invalidation inside the render loop would be unreliable -- this is
  // the authoritative invalidation point.
  m_pathTracer.onSceneInvalidated(m_resources);
  m_rasterizer.onSceneInvalidated(m_resources);
  // The retained scene is kept, emptied, when images are kept for reuse (retainedSpare); otherwise
  // destroyed now, its backend waiting for the device and running what it retired.
  if(m_resources.retainedGpu && keepImagesForReuse)
  {
    m_resources.retainedGpu->clear();
    m_resources.retainedSpare = std::move(m_resources.retainedGpu);
  }
  m_resources.retainedGpu.reset();
  if(!keepImagesForReuse)
    m_resources.retainedSpare.reset();
  if(!m_resources.hostFed)  // a stream host's store is the session's
    m_resources.retainedScene.reset();
  m_resources.selectedRenderNodes.clear();
  m_resources.animationControl      = AnimationControl{};
  m_resources.animationControl.play = m_animPlay;

  // Drop async pick state tied to the outgoing scene - a pending click ray-pick would otherwise
  // be consumed by the newly loaded scene once its semaphore settles, applying a pick result to the wrong node indices.
  m_pendingClickPick.reset();
  m_pendingClickResult.reset();

}

//--------------------------------------------------------------------------------------------------
// The retained scene's backend over the renderer's allocator, uploader, sampler pool and the scene's
// texture and sampler arrays
//
rs::VulkanGpuBackend::Info GltfRenderer::retainedBackendInfo()
{
  nvapp::Application* app = m_app;
  return {.device             = m_device,
          .physicalDevice     = m_app->getPhysicalDevice(),
          .alloc              = &m_resources.allocator,
          .staging            = &m_resources.staging,
          .samplerPool        = &m_resources.samplerPool,
          .submitResourceFree = [app](std::function<void()>&& fn) { app->submitResourceFree(std::move(fn)); },
          .bindings           = &m_resources.descriptorBinding[0],
          .descriptorSet      = m_resources.descriptorSet,
          .textureBinding     = shaderio::BindingPoints::eTextures,
          .samplerBinding     = shaderio::BindingPoints::eSamplers,
          .maxTextures        = m_maxTextures,
          .maxSamplers        = m_maxSamplers,
          .blasScratchBudget  = VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20,
          .opacityMicromaps   = m_resources.settings.opacityMicromapSupported,
          .takePrefetched = [this](const std::filesystem::path& path, bool srgb) { return m_resources.imagePrefetch.take(path, srgb); },
          .maxImageExtent = [this] { return m_maxTextureExtent.load(); },
          .imageHeadroom  = [this] { return sceneHeadroom(); }};
}

//--------------------------------------------------------------------------------------------------
// A stream host's own retained scene (renderer.hpp, hostScene)
//
rs::RetainedScene& GltfRenderer::hostScene()
{
  if(!m_resources.hostFed)
  {
    m_resources.retainedScene = std::make_unique<rs::RetainedScene>();
    m_resources.hostFed       = true;
  }
  return *m_resources.retainedScene;
}

void GltfRenderer::replaceHostScene()
{
  rs::RetainedScene& store = hostScene();
  fv::timing::Scope  timeScene("retained scene");
  {
    fv::timing::Scope t("wait idle + cleanup");
    // the teardown loadModel runs, for the same reasons; the host's store and selection stay
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    m_loadPipeline.clear();
    const rs::EntityId selected = m_selectedEntity;
    // residencyRetireOnReplace: the backend stays, out of the cleanup's way, and syncs the new scene
    // over the old one below
    std::unique_ptr<rs::RetainedSceneVk> kept;
    if(m_resources.settings.residencyRetireOnReplace)
      kept = std::move(m_resources.retainedGpu);
    cleanupScene(/*keepImagesForReuse=*/true);
    m_resources.retainedGpu = std::move(kept);
    m_selectedEntity        = selected;  // where its entity is still in the scene
    if(!store.clip(m_hostClip.clip()))
      m_hostClip = {};
  }
  // the previous scene's backend, emptied, when the cleanup kept it: its decoded images are this one's to take
  const bool retire = m_resources.retainedGpu != nullptr;
  if(retire)
    m_resources.retainedSpare.reset();
  else if(m_resources.retainedSpare)
  {
    m_resources.retainedGpu = std::move(m_resources.retainedSpare);
    m_resources.retainedGpu->setBlasScratchBudget(VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20);
  }
  else
    m_resources.retainedGpu = std::make_unique<rs::RetainedSceneVk>(
        retainedBackendInfo(),
        rs::ResidencyOptions{.minInstances        = std::max(m_resources.settings.residencyMinInstances, 1u),
                             .blasScratchBudget   = VkDeviceSize(m_resources.settings.blasScratchBudgetMB) << 20,
                             .textureGraceBytes   = VkDeviceSize(m_resources.settings.textureGraceMB) << 20,
                             .refitsBeforeRebuild = m_resources.settings.blasRefitsBeforeRebuild,
                             .compactStatic       = m_resources.settings.blasCompactStatic});
  // the queue is idle: what the old scene let go of isn't in the new one's way when it is fitted
  m_resources.retainedGpu->runRetired();
  {
    fv::timing::Scope timeUpload("retained upload");
    VkCommandBuffer   cmd = m_app->createTempCmdBuffer();
    if(fv::timing::enabled())
      m_resources.retainedGpu->timeBuilds();
    m_resources.retainedGpu->setShaderFeatures(drawableFeatures());
    if(retire)
    {
      m_resources.retainedGpu->beginFrame();
      m_resources.retainedGpu->sync(cmd, store);
    }
    else
      m_resources.retainedGpu->replace(cmd, store);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    logRetainedNotices();
    m_resources.imagePrefetch.drop();  // what this scene did not take would never be taken
    m_pendingGeometry = 0;
    if(const std::optional<rs::BuildTimes> times = m_resources.retainedGpu->buildTimes())
    {
      OneShotGpuTimer::report("GPU BLAS build", times->blasMs);
      OneShotGpuTimer::report("GPU TLAS build", times->tlasMs);
    }
  }
  const rs::Residency&             residency = m_resources.retainedGpu->residency();
  const rs::Residency::ImageCounts images    = residency.imageCounts();
  LOGI("Scene (retained, host): %u instances, %u prims, %u materials, %u lights; images %u decoded (%u prefetched), %u reused; GPU memory %.1f MB\n",
       residency.instanceHighWater(), residency.primHighWater(), residency.materialHighWater(), residency.lightCount(),
       images.decoded, m_resources.retainedGpu->takePrefetchCount(), images.reused,
       m_resources.retainedGpu->memoryStats().usedBytes / (1024.0 * 1024.0));

  // the camera stays
  m_resources.currentFeatureSet = {};
  updateRetainedFeatures(false);
  calibrateFireflyClamp();
  if(const nvutils::Bbox bounds = m_resources.gpuView().bounds(); !bounds.isEmpty())
    m_resources.settings.infinitePlaneDistance = bounds.min().y;
  refreshEntitySelection();
  resetFrame();
}

void GltfRenderer::setHostClip(rs::ClipId clip, float at)
{
  m_hostClip = !clip.isNone() && m_resources.retainedScene && m_resources.retainedScene->clip(clip) ?
                   rs::ClipPlayer(*m_resources.retainedScene, clip) :
                   rs::ClipPlayer{};
  if(!m_hostClip.clip().isNone())
    m_hostClip.setTime(std::clamp(m_hostClip.start() + at, m_hostClip.start(), std::max(m_hostClip.start(), m_hostClip.end())));
  m_resources.animationControl.seek = true;  // shown where it is at the next frame
}

void GltfRenderer::seekHostClip(float time)
{
  if(m_hostClip.clip().isNone())
    return;
  m_hostClip.setTime(std::clamp(m_hostClip.start() + time, m_hostClip.start(), std::max(m_hostClip.start(), m_hostClip.end())));
  m_resources.animationControl.play = false;
  m_resources.animationControl.seek = true;
}

// The clip playing moved on by this frame's time and written into the store's entities.
bool GltfRenderer::advanceHostClip()
{
  AnimationControl&  ctrl  = m_resources.animationControl;
  rs::RetainedScene& store = *m_resources.retainedScene;
  if(m_hostClip.clip().isNone() || !store.clip(m_hostClip.clip()) || !ctrl.doAnimation())
    return false;
  if(ctrl.isReset())
    m_hostClip.reset();
  else if(!ctrl.seek)
  {
    m_hostClip.advance(ctrl.deltaTime(), ctrl.loop);
    if(!ctrl.loop && m_hostClip.time() >= m_hostClip.end())
      ctrl.play = false;
  }
  const bool wrote = m_hostClip.apply(store);
  if(wrote)
    ctrl.clearStates();
  return wrote;
}

//--------------------------------------------------------------------------------------------------
// What the retained scene's last syncs could not do as asked (a repack, a material drawn without its
// textures), in the log
//
void GltfRenderer::logRetainedNotices()
{
  for(const std::string& notice : m_resources.retainedGpu->takeNotices())
    LOGW("Retained scene: %s\n", notice.c_str());
  for(const std::string& notice : m_resources.retainedGpu->takeUserNotices())
    if(m_noticeSink)
      m_noticeSink(notice);
}

//--------------------------------------------------------------------------------------------------
// The retained scene's frame: the clip moved on, then the
// residency's sync of what the host fed since the last frame. True when the frame changed.
//
bool GltfRenderer::syncRetainedScene(VkCommandBuffer cmd)
{
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Retained sync");
  advanceHostClip();
  m_resources.retainedGpu->beginFrame();
  m_resources.retainedGpu->setShaderFeatures(drawableFeatures());
  const rs::SyncResult result = m_resources.retainedGpu->sync(cmd, *m_resources.retainedScene);
  if(result.drawListChanged)
    m_pendingGeometry = 0;  // what was accepted is resident, and counted as used
  logRetainedNotices();
  if(const auto compaction = m_resources.retainedGpu->takeCompaction(); compaction.blases && fv::timing::enabled())
  {
    constexpr double kMB = 1024.0 * 1024.0;
    fv::timing::line(fmt::format("retained BLAS compaction: {} BLASes, {:.1f} MB to {:.1f} MB; every BLAS {:.1f} MB",
                                 compaction.blases, compaction.before / kMB, compaction.after / kMB, compaction.blasBytes / kMB));
  }
  // the variant only grows (D15), or a removal earlier in the session would compile a smaller one
  // now, with a queue wait
  if(result.featuresChanged)
    updateRetainedFeatures(true);
  return afterRetainedSync(result);
}

// What a retained sync's result asks of the renderers; true when the frame changed
bool GltfRenderer::afterRetainedSync(const rs::SyncResult& result)
{
  // The rasterizer's recorded draws name instances and buffers; any change records them again
  if(result.changed)
    m_rasterizer.freeRecordCommandBuffer(m_resources);
  // instances came or went: the selection's outline follows
  if(result.drawListChanged)
    refreshEntitySelection();
  if(result.dlssReset)
  {
    if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
      m_pathTracer.notifyDlssContentReset(m_resources);
    else
      m_rasterizer.notifyDlssContentReset(m_resources);
  }
  return result.changed;
}

void GltfRenderer::updateRetainedFeatures(bool grow)
{
  nvvkgltf::SceneFeatureSet features = m_resources.retainedScene->features();
  if(grow)  // a removal never recompiles mid-session: more compiled in still draws the rest
  {
    nvvkgltf::SceneFeatureSet had = m_resources.currentFeatureSet;
    had.set(nvvkgltf::SceneFeatureSet::eDlssGuide, false);
    features |= had;
  }
  features.set(nvvkgltf::SceneFeatureSet::eDlssGuide, dlssGuideRequired());
  m_resources.currentFeatureSet = features;
}

std::optional<nvvkgltf::SceneFeatureSet> GltfRenderer::drawableFeatures()
{
  if(m_resources.settings.renderSystem != RenderingMode::ePathtracer)
    return std::nullopt;
  return m_pathTracer.liveShaderFeatures();
}

nvutils::Bbox GltfRenderer::sceneBounds() const
{
  const SceneGpuView& view = m_resources.gpuView();
  if(!view.hasContent() || view.renderNodeCapacity() == 0)
    return {};
  return view.bounds();
}

bool GltfRenderer::loadSceneFile(const std::filesystem::path& filename)
{
  if(!m_app)
    return false;

  // Synchronous: a client needs the scene on the GPU before this returns. A failed load leaves an
  // empty scene rather than the previous one, and the caller in main.cpp names the file.
  vkQueueWaitIdle(m_app->getQueue(0).queue);
  m_loadPipeline.clear();
  cleanupScene(/*keepImagesForReuse=*/true);
  return createScene(filename);
}

void GltfRenderer::prefetchImage(const std::filesystem::path& diskPath, bool srgb)
{
  m_resources.imagePrefetch.prefetch(diskPath, srgb, m_maxTextureExtent.load());
}
