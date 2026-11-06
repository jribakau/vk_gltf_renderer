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
#include <chrono>
#include <cmath>
#include <optional>
#include <utility>

#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>

#include "memory_plan.hpp"
#include "renderer.hpp"
#include "gpu_crash_trail.hpp"
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
          .descriptorSetLayout = m_resources.descriptorSetLayout[0],
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

void GltfRenderer::reportMemory(const MemoryReport& report)
{
  if(report == m_memoryReported)
    return;
  m_memoryReported = report;
  if(m_memorySink)
    m_memorySink(report);
}

// A line in the log when what the scene holds has moved, so that a real scene's bytes can be read back
void GltfRenderer::logMemoryLedger()
{
  constexpr uint64_t kWorthSaying = 32ull << 20;
  const rs::MemoryLedger ledger   = memoryLedger();
  const uint64_t         total    = ledger.total.total();
  if((total > m_ledgerLogged ? total - m_ledgerLogged : m_ledgerLogged - total) < kWorthSaying)
    return;
  m_ledgerLogged = total;
  const MemoryBudget budget = deviceMemoryBudget();
  const auto         mb     = [](uint64_t bytes) { return unsigned(bytes >> 20); };
  LOGI("[memory] scene %u MB: drawn %u, BVH %u, remake %u, working %u, unread %u, images %u (idle %u), scene %u; retiring %u; in system memory %u; allocated %u in blocks of %u; driver %u of %u MB\n",
       mb(total), mb(ledger.total.drawn), mb(ledger.total.bvh), mb(ledger.total.remake), mb(ledger.total.working), mb(ledger.total.unread),
       mb(ledger.total.images), mb(ledger.idleImages), mb(ledger.total.scene), mb(ledger.retiring), mb(ledger.spilled),
       mb(budget.allocatorAllocations), mb(budget.allocatorBlocks), mb(budget.usage), mb(budget.budget));
}

// memoryFillMB: device memory nothing else uses, touched every frame so the driver has to keep it where it
// is, for a measure of what the device does near and past its budget (scripts/memory-limit-check.py)
void GltfRenderer::applyMemoryFill(VkCommandBuffer cmd)
{
  constexpr VkDeviceSize kBlock  = VkDeviceSize(256) << 20;
  const size_t           blocks  = size_t(m_resources.settings.memoryFillMB) * (VkDeviceSize(1) << 20) / kBlock;
  while(m_memoryFill.size() > blocks)
  {
    m_app->submitResourceFree([alloc = &m_resources.allocator, buffer = m_memoryFill.back()]() mutable { alloc->destroyBuffer(buffer); });
    m_memoryFill.pop_back();
  }
  while(m_memoryFill.size() < blocks)
  {
    const auto   start = std::chrono::steady_clock::now();
    nvvk::Buffer buffer;
    const VkResult result = m_resources.allocator.createBuffer(buffer, kBlock, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    const double   ms     = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    VmaAllocationInfo info{};
    if(result == VK_SUCCESS)
      vmaGetAllocationInfo(m_resources.allocator, buffer.allocation, &info);
    LOGI("[memory fill] block %zu: result %d in %.2f ms, memory type %u\n", m_memoryFill.size(), int(result), ms, info.memoryType);
    if(result != VK_SUCCESS)
    {
      m_resources.settings.memoryFillMB = uint32_t(m_memoryFill.size() * (kBlock >> 20));
      break;
    }
    m_memoryFill.push_back(buffer);
  }
  for(const nvvk::Buffer& buffer : m_memoryFill)
    vkCmdFillBuffer(cmd, buffer.buffer, 0, 4, uint32_t(m_resources.frameCount));
  // memoryFillWriteMB: blocks written whole each frame, in turn, so one the driver moved to system memory
  // shows in the frame's time
  if(!m_memoryFill.empty())
    for(uint32_t i = 0; i < m_resources.settings.memoryFillWriteMB / (kBlock >> 20); i++)
      vkCmdFillBuffer(cmd, m_memoryFill[m_memoryFillNext++ % m_memoryFill.size()].buffer, 0, VK_WHOLE_SIZE, uint32_t(m_resources.frameCount));
}

void GltfRenderer::setLiveMeshes(std::set<rs::MeshId> meshes)
{
  if(m_resources.retainedGpu)
    m_resources.retainedGpu->setLive(std::move(meshes));
}

void GltfRenderer::setItemDetail(rs::ItemDetail detail)
{
  if(m_resources.retainedGpu)
    m_resources.retainedGpu->setItemDetail(std::move(detail));
}

void GltfRenderer::watchMemory()
{
  m_resources.imagePrefetch.dropOlderThan(120);
  const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  // No host to tell: a render from the command line keeps to what it loaded with, so that two of them
  // compare. isAutomatedRun() can't say so: the app's renderer runs headless too.
  if(!m_memorySink || !m_memoryBudgetExtension || !m_memoryWatch.due(now))
    return;
  logMemoryLedger();
  rs::RetainedSceneVk&               gpu     = *m_resources.retainedGpu;
  const rs::Residency::LoweredImages lowered = gpu.residency().loweredImages();
  constexpr uint64_t                 kSpillStep = 64ull << 20;  // a figure that wanders doesn't tell the app
  const uint64_t                     spilled    = gpu.backend().spilledBytes();
  const std::string                  next       = gpu.residency().nextToGo(*m_resources.retainedScene);
  const rs::MemoryLedger             ledger     = gpu.residency().ledger();
  const MemoryBudget                 budget     = deviceMemoryBudget();
  const auto                         step       = [](uint64_t bytes) { return bytes / kSpillStep * kSpillStep; };
  reportMemory({.lowered      = lowered.count,
                .largest      = lowered.largest,
                .parked       = uint32_t(gpu.residency().parkedForMemory()),
                .spilledBytes = step(spilled),
                .letGoHidden  = uint32_t(gpu.residency().letGoHidden().size()),
                .settled      = uint32_t(gpu.residency().settledCount()),
                .budget       = step(budget.budget),
                .used         = step(budget.usage),
                .geometry     = step(ledger.total.total() - ledger.total.images),
                .textures     = step(ledger.total.images),
                .nextToGo     = next});
  const std::optional<int64_t> headroom = sceneHeadroom();
  if(!headroom)
    return;
  // what found no memory, tried again now and then once there may be some
  if(gpu.residency().parkedForMemory() > 0 && *headroom > int64_t(rs::MemoryWatch::kMargin) && now >= m_retryParkedAt)
  {
    constexpr double kRetryEvery = 30;  // what still finds none makes the frame wait for the driver to say so
    m_retryParkedAt = now + kRetryEvery;
    gpu.retryParked();
  }
  const int64_t free = *headroom - int64_t(m_pendingGeometry);
  const rs::MemoryWatch::Act act = m_memoryWatch.read({.now        = now,
                             .headroom   = free,
                             .pastBudget = *headroom < -int64_t(sceneBudget() / 10),  // the tenth kept back is gone too
                             .canLower   = gpu.lowerableImageBytes(rs::kLeastImageExtent) > 0 || !next.empty(),
                             .lowered    = lowered.count > 0 || spilled > 0 || gpu.residency().detailBelowFull(),
                             .refining   = isRefiningStill(),
                             .held       = m_texturesHeld,
                             .decoding   = gpu.residency().queuedDecodes() > 0});
  // textures have stopped moving: a still takes its samples from them alone, once
  if(act == rs::MemoryWatch::Act::eNone && !m_texturesHeld && std::exchange(m_texturesMoved, false))
    resetFrame();
  switch(act)
  {
    case rs::MemoryWatch::Act::eLower:
      gpu.lowerImages(uint64_t(rs::MemoryWatch::kMargin));
      break;
    case rs::MemoryWatch::Act::eRaise:
    {
      // what lies in system memory first: it costs the frame more than a texture a size down
      if(spilled > 0 && gpu.bringBack(*m_resources.retainedScene, uint64_t(free - rs::MemoryWatch::kMargin)) > 0)
        break;
      const auto itemUp = [&](bool unseenToo) {
        const std::optional<rs::ItemDetail::Up> up = gpu.nextDetailUp(uint64_t(free - rs::MemoryWatch::kMargin), unseenToo);
        if(!up || !m_detailSink)
          return false;
        admitGeometry(up->bytes);
        m_detailSink(up->item, true);
        return true;
      };
      // the way it came down: textures to where detail began to go, then detail, then the rest of them, and
      // the detail of what no ray reaches once what they reach is back
      if(gpu.raiseImages(*m_resources.retainedScene, rs::MemoryWatch::raiseRoom(free), rs::kDetailAfterExtent) > 0 || itemUp(false)
         || gpu.raiseImages(*m_resources.retainedScene, rs::MemoryWatch::raiseRoom(free)) > 0 || itemUp(true))
        break;
      m_memoryWatch.raiseEnded(now);
      break;
    }
    case rs::MemoryWatch::Act::eNone:
      break;
  }
  // Near full, what rays reach starts to decide what gives way; a scene with room never counts
  const bool nearFull = free < int64_t(sceneBudget() / 4) || lowered.count > 0 || spilled > 0 || gpu.residency().settledCount() > 0 || gpu.residency().detailBelowFull();
  m_pathTracer.setCountHits(nearFull);
  // what they want back doesn't wait as a raising does: the view just asked for it (short of memory, the watch goes first)
  if(act == rs::MemoryWatch::Act::eNone && !m_texturesHeld && free >= 0)
    if(const std::optional<rs::ItemDetail::Up> up = gpu.followHits(*m_resources.retainedScene, free - rs::MemoryWatch::kMargin); up && m_detailSink)
    {
      admitGeometry(up->bytes);
      m_detailSink(up->item, true);
    }
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
  applyMemoryFill(cmd);
  watchMemory();
  const rs::SyncResult result = m_resources.retainedGpu->sync(cmd, *m_resources.retainedScene);
  {
    // the frame's memory and what its sync did, for what a lost device leaves in the log
    const MemoryBudget                          budget  = deviceMemoryBudget();
    const nvvk::BufferCircularAllocator::Report staging = m_resources.staging.getStagingReport();
    const rs::Residency&                        res     = m_resources.retainedGpu->residency();
    const rs::Residency::ImageCounts            images  = res.imageCounts();
    std::string                                 did;
    for(const auto& [on, name] : {std::pair{result.changed, "changed"}, std::pair{result.imagesMoved, "textures moved"},
                                  std::pair{result.drawListChanged, "instances"}, std::pair{result.buffersChanged, "buffers"},
                                  std::pair{result.tlasHandleChanged, "TLAS made"}, std::pair{result.verticesWritten, "vertices"}})
      if(on)
        did += did.empty() ? name : std::string(", ") + name;
    gpu_trail::record({.frame       = m_trailFrames,
                       .budget      = budget.budget,
                       .usage       = budget.usage,
                       .allocated   = budget.allocatorBlocks,
                       .spilled     = m_resources.retainedGpu->backend().spilledBytes(),
                       .stagingHeld = staging.allocatedSize,
                       .stagingLive = staging.requestedSize,
                       .decoded     = images.decoded,
                       .copied      = images.copied,
                       .refused     = images.refused,
                       .queued      = uint32_t(res.queuedDecodes()),
                       .sync        = std::move(did)});
  }
  if(result.drawListChanged)
    m_pendingGeometry = 0;  // what was accepted is resident, and counted as used
  // an export keeps the detail it began with
  for(uint64_t item : m_resources.retainedGpu->takeDetailAsks())
    if(m_detailSink && !m_texturesHeld)
      m_detailSink(item, false);
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
  // instances came or went: the selection's outline follows
  if(result.drawListChanged)
    refreshEntitySelection();
  if(result.dlssReset)
    m_pathTracer.notifyDlssContentReset(m_resources);
  m_texturesMoved = m_texturesMoved || result.imagesMoved;
  return result.changed && !result.samePicture;
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
