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

// A HostRenderer without a GPU: the host's retained scene, its selection and clip, and a camera,
// so FvHost runs its commands in gtest (daz-viewer docs/command-journal-plan.md, C0). What only
// draws (the gizmo, helper lines, the debug views) is kept or dropped; nothing is rendered.

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>

#include "camera_controls.hpp"
#include "fake_gpu_backend.hpp"
#include "host_renderer.hpp"

class FakeHostRenderer : public HostRenderer
{
public:
  FakeHostRenderer()
  {
    // a view as the app starts with one: a camera prop added stands here
    m_camera->setWindowSize({1280, 720});
    m_camera->setLookat({0.0, 1.0, 3.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, true);
  }

  rs::RetainedScene& hostScene() override { return m_scene; }
  bool               feedsHostScene() const override { return true; }
  bool               drawsRetainedScene() const override { return replaces > 0; }
  void               replaceHostScene() override { replaces++; }
  void               prefetchImage(const std::filesystem::path&, bool) override { prefetches++; }
  void               setLiveMeshes(std::set<rs::MeshId> meshes) override { liveMeshes = std::move(meshes); }
  std::set<rs::MeshId> liveMeshes;
  void               dropDecodedAhead() override { decodedAheadDropped++; }
  void               decodeAhead(bool allowed) override { decodingAhead = allowed; }
  std::atomic<int>   prefetches{0};
  std::atomic<int>   decodedAheadDropped{0};
  std::atomic<bool>  decodingAhead{true};
  std::string        geometryShortfall(uint64_t bytes, bool keepTextureDetail = false) override
  {
    askedGeometry += bytes;
    lastGeometry      = bytes;
    lastKeptTextures  = keepTextureDetail;
    return bytes > geometryLimit ? std::string("no room") : geometryRefusal;
  }
  uint64_t blasBytes(uint64_t triangles, uint64_t, bool refittable) const override
  {
    return triangles * (refittable ? rs::FakeGpuBackend::kBlasBytesPerTriangle : rs::FakeGpuBackend::kCompactedBlasBytesPerTriangle);
  }
  rs::MemoryLedger   memoryLedger() const override { return {}; }
  void               admitGeometry(uint64_t bytes) override { admittedGeometry += bytes; }
  uint64_t           admittedGeometry = 0;
  std::string        geometryRefusal;  // what geometryShortfall answers
  uint64_t           geometryLimit = UINT64_MAX;  // past this it answers "no room"
  uint64_t           askedGeometry = 0, lastGeometry = 0;
  bool               lastKeptTextures = false;
  void               setMaxTextureExtent(uint32_t) override {}
  void               setNoticeSink(std::function<void(const std::string&)>) override {}
  void               setItemDetail(rs::ItemDetail) override {}
  void               setDetailSink(std::function<void(uint64_t, bool)>) override {}
  std::optional<int64_t> sceneHeadroom() const override { return headroom; }
  std::optional<int64_t> headroom;  // none: no budget known
  void               setMemorySink(std::function<void(const MemoryReport&)> sink) override { memorySink = std::move(sink); }
  std::function<void(const MemoryReport&)> memorySink;
  void               holdTextures(bool held) override { texturesHeld = held; }
  bool               texturesHeld = false;
  void               setTexturesArriving(bool arriving) override { texturesArriving = arriving; }
  bool               texturesArriving = false;
  void               setStillPaused(bool paused) override { stillPaused = paused; }
  bool               stillPaused = false;
  nvutils::Bbox      visibleSceneBounds() override
  {
    const auto b = m_scene.bounds();
    return b[0].x > b[1].x ? nvutils::Bbox() : nvutils::Bbox(b[0], b[1]);
  }

  void         selectEntity(rs::EntityId entity) override { m_selected = entity; }
  rs::EntityId selectedEntity() const override { return m_selected; }
  void         pickAt(glm::vec2 uv, bool center) override { picks.push_back({uv, center}); }
  void         pivotAt(glm::vec2 uv) override { pivots.push_back(uv); }
  void setPivotObserver(std::function<void(const std::optional<glm::vec3>&, bool)> observer) override { pivotObserver = std::move(observer); }
  void setPickObserver(std::function<bool(const std::optional<InstancePick>&, const glm::vec3&)> observer) override
  {
    pickObserver = std::move(observer);
  }
  void inspectAt(std::optional<glm::vec2> uv) override { inspects.push_back(uv); }
  void setInspectObserver(std::function<void(const PixelReadout&)> observer) override { inspectObserver = std::move(observer); }
  void setGizmoHooks(GizmoHooks hooks) override { gizmo = std::move(hooks); }
  void setShowGizmo(bool) override {}
  void setGizmoOptions(const GizmoOptions& options) override { gizmoOptions = options; }
  GizmoOptions gizmoOptions;
  bool gizmoInput(glm::vec2, bool, bool, bool) override { return false; }
  void setHelperLines(std::vector<shaderio::lines::LineVertex> vertices) override { helperLines = std::move(vertices); }

  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() override { return m_camera; }
  VkExtent2D                                  renderSize() const override { return {1280, 720}; }
  void                                        fitCamera(bool) override {}
  void setVisualization(shaderio::Visualization) override {}
  void setWireframe(bool) override {}
  void setFalseColor(bool on) override { falseColor = on; }
  bool falseColor = false;
  bool pixelTimeAvailable() const override { return clockAvailable; }
  void setPixelTime(bool on) override { pixelTime = on; }
  void setPixelTimeObserver(std::function<void(const PixelTimes&)> observer) override { pixelTimeObserver = std::move(observer); }
  bool                                   clockAvailable = true;
  bool                                   pixelTime      = false;
  std::function<void(const PixelTimes&)> pixelTimeObserver;
  void keepWipeFrame() override { wipeKept++; }
  void setWipe(std::optional<float> at) override { wipeAt = at; }
  int                  wipeKept = 0;
  std::optional<float> wipeAt;
  void setShowGrid(bool) override {}
  void setShowAxis(bool) override {}
  void setDepthOfField(float, float, bool, int, float) override {}

  bool          isStill() const override { return false; }
  bool          isRefiningStill() const override { return false; }
  bool          isStillConverged() const override { return false; }
  bool          accumulationComplete() const override { return false; }
  bool          driverCacheFull() const override { return false; }
  int           accumulatedSamples() const override { return stillSamples; }
  int           sampleCap() const override { return stillSampleCap; }
  // saved as soon as asked, an empty file standing for the image
  void saveImage(const std::filesystem::path& path) override
  {
    savedImage = path;
    std::ofstream{path};
  }
  bool imageSavePending() const override { return false; }
  MemoryBudget  deviceMemoryBudget() const override { return {}; }
  uint32_t      frameBytesPerPixel() const override { return 4; }
  nvvk::ResourceAllocator* deviceAllocator() override { return nullptr; }

  void              setHostClip(rs::ClipId clip, float at = 0.f) override { m_clip = clip, m_clipTime = at; }
  float             hostClipTime() const override { return m_clipTime; }
  float             hostClipDuration() const override { return 0.f; }
  void              seekHostClip(float time) override { m_clipTime = time; }
  AnimationControl& editAnimationControl() override { return m_animation; }

  bool loadHdrEnvironment(EnvImage&&, const std::filesystem::path&) override { return true; }
  void writeEnvironmentSky(tinygltf::Model&, const std::filesystem::path&) const override {}

  // what the host handed over, for a test to read
  // how many times the host made its scene resident; resident (drawsRetainedScene) from the first
  int                                                                           replaces = 0;
  GizmoHooks                                                                    gizmo;
  std::function<bool(const std::optional<InstancePick>&, const glm::vec3&)>     pickObserver;
  std::vector<shaderio::lines::LineVertex>                                      helperLines;
  std::vector<std::pair<glm::vec2, bool>>                                       picks;  // pickAt(uv, center), in order
  std::vector<glm::vec2>                                                        pivots;  // pivotAt(uv), in order
  std::function<void(const std::optional<glm::vec3>&, bool)>                    pivotObserver;
  std::vector<std::optional<glm::vec2>>                                         inspects;  // inspectAt(uv), in order
  std::function<void(const PixelReadout&)>                                      inspectObserver;
  int stillSamples   = 0;  // what accumulatedSamples() and sampleCap() say
  int stillSampleCap = 0;
  std::filesystem::path savedImage;  // the last saveImage's path
  rs::ClipId            hostClip() const { return m_clip; }  // the model's clip setHostClip last gave

private:
  rs::RetainedScene                           m_scene;
  rs::EntityId                                m_selected;
  std::shared_ptr<nvutils::CameraManipulator> m_camera = std::make_shared<DragCameraManipulator>();  // the renderer's
  rs::ClipId                                  m_clip;
  float                                       m_clipTime = 0.f;
  AnimationControl                            m_animation;
};
