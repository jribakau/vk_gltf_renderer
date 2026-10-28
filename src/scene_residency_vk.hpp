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
// The retained scene on Vulkan (docs: daz-viewer docs/retained-scene-plan.md, "Testing without a
// GPU", "Coexistence"). VulkanGpuBackend is rs::GpuBackend over the renderer's allocator, frame
// uploader, sampler pool and scene descriptor set; RetainedSceneVk owns one and an rs::Residency
// over it, and is what the renderers read of a RetainedScene (SceneGpuView). What these decide
// apart from the device is in scene_residency_vk_util.hpp, which the tests cover.
//
// VulkanGpuBackend records into the command buffer record() names, in call order: an upload is
// appended to the frame uploader and recorded ahead of the next command; BLAS builds and refits
// queue until the next other command and go out in batches over one scratch buffer (planScratch,
// I8). Retired closures run through the application's deferred-free queue, and whatever is still
// pending when the backend goes runs after its idle wait. Application thread only.
//

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

#include <nvvk/acceleration_structures.hpp>
#include <nvvk/resources.hpp>

#include "gltf_image_loader.hpp"
#include "gltf_scene_animation_vk.hpp"
#include "gpu_memory_tracker.hpp"
#include "scene_gpu_view.hpp"
#include "scene_residency.hpp"
#include "scene_residency_vk_util.hpp"

namespace nvvk {
class ResourceAllocator;
class CmdUploaderInterface;
class SamplerPool;
class DescriptorBindings;
}  // namespace nvvk

namespace rs {

class VulkanGpuBackend : public GpuBackend
{
public:
  struct Info
  {
    VkDevice                    device         = VK_NULL_HANDLE;
    VkPhysicalDevice            physicalDevice = VK_NULL_HANDLE;
    nvvk::ResourceAllocator*    alloc          = nullptr;
    nvvk::CmdUploaderInterface* staging        = nullptr;  // the frame uploader
    nvvk::SamplerPool*          samplerPool    = nullptr;
    // runs a closure once no frame in flight can use what it frees (Application::submitResourceFree)
    std::function<void(std::function<void()>&&)> submitResourceFree;
    // the scene's texture (SAMPLED_IMAGE) and sampler (SAMPLER) arrays
    const nvvk::DescriptorBindings* bindings          = nullptr;
    VkDescriptorSet                 descriptorSet     = VK_NULL_HANDLE;
    uint32_t                        textureBinding    = 0;
    uint32_t                        samplerBinding    = 0;
    uint32_t                        maxTextures       = 0;
    uint32_t                        maxSamplers       = 0;
    VkDeviceSize                    blasScratchBudget = 512ull << 20;  // blasScratchBudgetMB
    bool                            generateMipmaps   = true;
    // VK_KHR_opacity_micromap available and on (Settings::opacityMicromapSupported)
    bool opacityMicromaps = false;
    // a file a host decoded ahead (SceneVk::takePrefetchedFile); any thread. Unset: none.
    std::function<std::optional<nvvkgltf::LoadedImageData>(const std::filesystem::path&, bool srgb)> takePrefetched;
  };

  // The first record() makes the default images and writes texture slot 0 (white) and sampler slot 0
  // (glTF's default sampler)
  explicit VulkanGpuBackend(const Info& info);
  // Waits for the device, runs the retired closures still pending, then frees what is left
  ~VulkanGpuBackend() override;
  VulkanGpuBackend(const VulkanGpuBackend&)            = delete;
  VulkanGpuBackend& operator=(const VulkanGpuBackend&) = delete;

  // The command buffer this frame's sync records into, until finish() records what is still queued
  void record(VkCommandBuffer cmd);
  void finish();
  void setBlasScratchBudget(VkDeviceSize bytes) { m_info.blasScratchBudget = bytes; }
  // GPU time of the AS builds recorded from here to finish(): timestamps around each flush of BLAS
  // builds and refits, and around each TLAS build with its instance upload (as legacy's `GPU BLAS
  // build` / `GPU TLAS build` lines count them), up to kTimedSpans. buildTimes() reads them once the
  // command buffer has completed: none before that, or when nothing was timed.
  void                      timeBuilds() { m_timeBuilds = true; }
  std::optional<BuildTimes> buildTimes();

  // What the renderers bind; null for none or a destroyed handle
  VkBuffer                          vkBuffer(BufferHandle buffer) const;
  VkAccelerationStructureKHR        vkAccel(AccelHandle accel) const;
  const nvvkgltf::GpuMemoryTracker& memory() const { return m_memory; }

  // GpuBackend
  void         beginFrame() override { m_frame++; }
  uint64_t     frame() const override { return m_frame; }
  void         retire(std::function<void()> closure) override;
  void         waitIdle() override;
  BufferHandle createBuffer(const BufferDesc& desc) override;
  void         destroyBuffer(BufferHandle buffer) override;
  uint64_t     bufferAddress(BufferHandle buffer) const override;
  uint64_t     bufferSize(BufferHandle buffer) const override;
  void         upload(BufferHandle dst, uint64_t offset, std::span<const std::byte> bytes) override;
  void copyBuffer(BufferHandle src, uint64_t srcOffset, BufferHandle dst, uint64_t dstOffset, uint64_t size) override;
  // decoded on this thread (a file, encoded bytes, or decoded pixels of 8 or 16 bits), its mips made
  ImageHandle createImage(const ImageSource& source, ImageInfo& info) override;
  // decoded side by side on the worker pool (a file a host prefetched taken instead), then made in order
  void createImages(std::span<const ImageSource> sources, std::span<ImageHandle> images, std::span<ImageInfo> infos) override;
  // the images createImages took from a host's prefetches, since the last call
  uint32_t    takePrefetchCount() { return std::exchange(m_prefetchTaken, 0u); }
  // the BLASes compacted since the last call, their sizes before and after, and every BLAS's now
  struct Compaction
  {
    uint32_t blases = 0;
    uint64_t before = 0, after = 0, blasBytes = 0;
  };
  Compaction  takeCompaction();
  void        destroyImage(ImageHandle image) override;
  // none: the white default in slot 0, the magenta "missing" image elsewhere (as SceneVk shows one)
  void        writeTextureDescriptor(uint32_t slot, ImageHandle image) override;
  void        writeSamplerDescriptor(uint32_t slot, const GpuSamplerDesc& desc) override;
  bool        opacityMicromaps() const override { return m_info.opacityMicromaps; }
  // as SceneOmm::create builds one; its inputs and scratch freed a frame cycle after the build
  AccelHandle buildMicromap(const MicromapInput& input) override;
  AccelHandle buildBlas(const BlasInput& input, AccelBuildFlags flags) override;
  void        refitBlas(AccelHandle blas, const BlasInput& input) override;
  // not yet: compaction needs the built size read back a frame later (P7); returns none
  AccelHandle compactBlas(AccelHandle blas) override;
  AccelHandle createTlas(uint32_t capacity) override;
  void        buildTlas(AccelHandle tlas, std::span<const TlasRow> rows, TlasMode mode) override;
  void        destroyAccel(AccelHandle accel) override;
  // shaders/morph.comp.slang or skinning.comp.slang over the dispatch's ranges (the pipelines are
  // made at the first)
  void dispatchDeform(const DeformDispatch& dispatch) override;
  void barrier(GpuBarrier barrier) override;

private:
  struct BufferEntry
  {
    nvvk::Buffer buffer;
    uint64_t     size = 0;  // as asked (the allocation is at least 16 bytes)
  };
  struct BlasEntry
  {
    nvvk::AccelerationStructure          accel;
    nvvk::AccelerationStructureBuildData data;
    // the opacity micromap link its geometry's triangles.pNext points at, where a move of the
    // entry leaves it
    std::unique_ptr<VkAccelerationStructureTrianglesOpacityMicromapKHR> micromap;
    // built with eAccelAllowCompaction: the query its compacted size is written to (-1: none), and
    // whether the frame that wrote it has retired, so the size can be read
    int32_t               compactQuery = -1;
    std::shared_ptr<bool> sizeWritten;
  };
  struct MicromapEntry
  {
    VkAccelerationStructureKHR      micromap = VK_NULL_HANDLE;
    nvvk::Buffer                    storage;
    nvvk::Buffer                    data, triangles, scratch;  // until the build is recorded and retired
    uint32_t                        triangleStride = 8;
    std::vector<VkMicromapUsageKHR> usages;
  };
  struct TlasEntry
  {
    nvvk::AccelerationStructure          accel;
    nvvk::AccelerationStructureBuildData data;
    nvvk::Buffer                         instances, scratch;
    uint32_t                             capacity = 0;
  };
  struct QueuedBuild
  {
    uint64_t blas   = 0;
    bool     update = false;
  };
  // the closures retire() handed on, by id; shared with the deferred-free queue's thunks, which do
  // nothing once the backend is gone (it ran them itself)
  struct RetireQueue
  {
    std::map<uint64_t, std::function<void()>> pending;
    uint64_t                                  next = 0;
  };

  bool                                    recording(const char* op) const;
  void                                    flushUploads();
  void                                    flushBuilds();
  void                                    recordMicromaps();
  void                                    writeCompactedSizes(std::span<const uint64_t> blases);
  void                                    releaseCompactQuery(BlasEntry& blas);
  bool                                    inLiveBuffer(VkDeviceAddress address, VkDeviceSize size) const;
  void                                    checkInput(const BlasInput& input, const char* op) const;
  nvvk::AccelerationStructureGeometryInfo geometryOf(const BlasInput& input, const void* micromap) const;
  void                                    writeImageDescriptor(uint32_t slot, const nvvk::Image& image);
  ImageHandle createFromDecoded(const ImageSource& source, nvvkgltf::LoadedImageData& loaded, ImageInfo& info);
  void                                    ensureDefaults();
  bool                                    beginSpan();
  void                                    endSpan(BuildSpan span);

  static constexpr uint32_t kTimedSpans     = 64;
  static constexpr uint32_t kCompactQueries = 4096;  // BLASes waiting for their compacted size

  Info                                      m_info;
  VkDeviceSize                              m_scratchAlignment = 256;
  uint64_t                                  m_frame            = 0;
  uint64_t                                  m_nextHandle       = 1;
  VkCommandBuffer                           m_cmd              = VK_NULL_HANDLE;
  bool                                      m_uploadsPending   = false;
  std::unordered_map<uint64_t, BufferEntry> m_buffers;
  std::map<VkDeviceAddress, VkDeviceSize>   m_ranges;  // live buffers by address, to check BLAS inputs
  std::unordered_map<uint64_t, nvvk::Image> m_images;
  std::unordered_map<uint64_t, BlasEntry>   m_blases;
  std::unordered_map<uint64_t, MicromapEntry> m_micromaps;
  std::vector<uint64_t>                       m_queuedMicromaps;  // built ahead of the queued BLASes
  std::unordered_map<uint64_t, TlasEntry>   m_tlases;
  std::vector<QueuedBuild>                  m_queued;
  nvvk::Buffer                              m_blasScratch;
  std::unordered_map<uint32_t, VkSampler>   m_samplers;  // by sampler slot
  nvvk::Image                               m_white, m_missing;
  bool                                      m_defaultsUploaded = false;
  std::shared_ptr<RetireQueue>              m_retired          = std::make_shared<RetireQueue>();
  nvvkgltf::GpuMemoryTracker                m_memory;
  bool                                      m_timeBuilds = false;  // until finish()
  VkQueryPool                               m_timestamps = VK_NULL_HANDLE;
  double                                    m_msPerTick  = 0.0;
  std::vector<BuildSpan>                    m_spans;  // recorded, not read yet
  VkQueryPool                                 m_compactQueries = VK_NULL_HANDLE;
  std::vector<uint32_t>                       m_freeCompactQueries;
  Compaction                                  m_compaction;
  uint32_t                                  m_prefetchTaken = 0;
  nvvkgltf::DeformPipelines                 m_deformPipelines;
};

// The retained scene as the renderers read it: a VulkanGpuBackend and an rs::Residency over a
// RetainedScene, brought up to date once per frame. Not reachable yet: the sceneBackend setting
// (P3a1) creates it in place of the legacy scene.
class RetainedSceneVk : public SceneGpuView
{
public:
  explicit RetainedSceneVk(const VulkanGpuBackend::Info& info, ResidencyOptions options = {});

  // A scene replace, recorded into `cmd`: what was resident is retired and `scene` made resident
  SyncResult replace(VkCommandBuffer cmd, RetainedScene& scene);
  // This frame's sync, recorded into `cmd` ahead of the frame's passes
  SyncResult sync(VkCommandBuffer cmd, RetainedScene& scene);
  // Starts the backend's next frame; once per rendered frame, before sync
  void beginFrame() { m_gpu.beginFrame(); }
  // blasScratchBudgetMB: the next batch regrows the scratch when it needs more
  void setBlasScratchBudget(VkDeviceSize bytes) { m_gpu.setBlasScratchBudget(bytes); }
  // textureGraceMB: the decoded images nothing samples that stay for the next replace or edit
  void setTextureGrace(VkDeviceSize bytes) { m_residency.setTextureGrace(bytes); }
  // blasRefitsBeforeRebuild: a BLAS refitted this many times is built again (0: never)
  void setRefitsBeforeRebuild(uint32_t refits) { m_residency.setRefitsBeforeRebuild(refits); }
  // blasCompactStatic: static BLASes built from now on are compacted
  void setCompactStatic(bool compact) { m_residency.setCompactStatic(compact); }
  // what the residency's last syncs could not do as asked, once each
  std::vector<std::string> takeNotices() { return m_residency.takeNotices(); }
  // the features the live shader variant has compiled in; none: every one (Residency::setShaderFeatures)
  void setShaderFeatures(std::optional<nvvkgltf::SceneFeatureSet> features) { m_residency.setShaderFeatures(features); }
  // the images taken from a host's prefetches since the last call
  uint32_t takePrefetchCount() { return m_gpu.takePrefetchCount(); }
  // blasCompactStatic: what the copies since the last call saved
  VulkanGpuBackend::Compaction takeCompaction() { return m_gpu.takeCompaction(); }
  // Nothing resident (what was is retired), the decoded images kept in grace: a host's scene
  // between a cleanup and the next retained scene replace, which takes what it shares
  void clear();
  // The next replace's or sync's AS builds timed on the GPU, read once its command buffer completed
  void                      timeBuilds() { m_gpu.timeBuilds(); }
  std::optional<BuildTimes> buildTimes() { return m_gpu.buildTimes(); }

  const Residency&        residency() const { return m_residency; }
  const VulkanGpuBackend& backend() const { return m_gpu; }

  // SceneGpuView
  bool                         hasContent() const override { return m_hasScene; }
  VkDeviceAddress              sceneDesc() const override;
  VkAccelerationStructureKHR   tlas() const override;
  uint32_t                     renderNodeCapacity() const override;
  const std::vector<uint32_t>& drawList(DrawList list) const override;
  uint64_t                     drawListsRevision() const override { return m_residency.drawListsRevision(); }
  uint64_t                     transformsRevision() const override { return m_transformsRevision; }
  DrawRecord                   drawRecord(uint32_t renderNode) const override;
  std::optional<glm::vec3>     worldCentre(uint32_t renderNode) const override;
  bool                         hasTransmission() const override { return m_residency.hasTransmission(); }
  bool                         hasOpacityMicromaps() const override { return m_residency.hasOpacityMicromaps(); }
  bool variantMayBuildOffFrame() const override { return !m_residency.drawsBeyondShaderFeatures(); }
  bool                         hasPlayableAnimation() const override { return m_playableAnimation; }
  // whether the scene it was imported from has a clip that plays (its clips are evaluated there)
  void setPlayableAnimation(bool playable) { m_playableAnimation = playable; }
  std::vector<LightInfo>       lights() const override { return m_lights; }
  nvutils::Bbox                bounds() const override { return m_bounds; }
  MemoryStats                  memoryStats() const override;
  VkDeviceAddress prevPositionTable() const override { return m_gpu.bufferAddress(m_residency.prevPositionTable()); }
  VkDeviceAddress prevObjectToWorldTable() const override { return m_gpu.bufferAddress(m_residency.prevWorldTable()); }

private:
  // what the view keeps of a sync: the lights and the bounds, which the store derives on demand
  void afterSync(RetainedScene& scene, const SyncResult& result);

  // the residency goes first, retiring what it holds into the backend, which runs it all after its
  // idle wait
  VulkanGpuBackend       m_gpu;
  Residency              m_residency;
  bool                   m_hasScene           = false;
  bool                   m_playableAnimation  = false;
  uint64_t               m_transformsRevision = 0;
  std::vector<LightInfo> m_lights;
  nvutils::Bbox          m_bounds;
};

}  // namespace rs
