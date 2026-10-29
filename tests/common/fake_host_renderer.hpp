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

#include <memory>

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
  bool               drawsRetainedScene() const override { return true; }
  void               replaceHostScene() override { replaces++; }
  void               prefetchImage(const std::filesystem::path&, bool) override {}
  nvutils::Bbox      visibleSceneBounds() override
  {
    const auto b = m_scene.bounds();
    return b[0].x > b[1].x ? nvutils::Bbox() : nvutils::Bbox(b[0], b[1]);
  }

  void         selectEntity(rs::EntityId entity) override { m_selected = entity; }
  rs::EntityId selectedEntity() const override { return m_selected; }
  void         pickAt(glm::vec2, bool) override {}
  void setPickObserver(std::function<bool(const std::optional<InstancePick>&, const glm::vec3&)> observer) override
  {
    pickObserver = std::move(observer);
  }
  void setGizmoHooks(GizmoHooks hooks) override { gizmo = std::move(hooks); }
  void setShowGizmo(bool) override {}
  bool gizmoInput(glm::vec2, bool, bool, bool) override { return false; }
  void setHelperLines(std::vector<shaderio::lines::LineVertex> vertices) override { helperLines = std::move(vertices); }

  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() override { return m_camera; }
  VkExtent2D                                  renderSize() const override { return {1280, 720}; }
  void                                        fitCamera(bool) override {}
  void setVisualization(shaderio::Visualization) override {}
  void setWireframe(bool) override {}
  void setShowGrid(bool) override {}
  void setShowAxis(bool) override {}
  void setDepthOfField(float, float, bool, int, float) override {}

  void          setRenderSystem(RenderingMode mode) override { m_mode = mode; }
  RenderingMode renderSystem() const override { return m_mode; }
  bool          isStill() const override { return false; }
  bool          isRefiningStill() const override { return false; }
  bool          isStillConverged() const override { return false; }
  bool          accumulationComplete() const override { return false; }
  bool          driverCacheFull() const override { return false; }
  int           accumulatedSamples() const override { return stillSamples; }
  int           sampleCap() const override { return stillSampleCap; }
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
  int                                                                           replaces = 0;
  GizmoHooks                                                                    gizmo;
  std::function<bool(const std::optional<InstancePick>&, const glm::vec3&)>     pickObserver;
  std::vector<shaderio::lines::LineVertex>                                      helperLines;
  int stillSamples   = 0;  // what accumulatedSamples() and sampleCap() say
  int stillSampleCap = 0;

private:
  rs::RetainedScene                           m_scene;
  rs::EntityId                                m_selected;
  std::shared_ptr<nvutils::CameraManipulator> m_camera = std::make_shared<nvutils::CameraManipulator>();
  RenderingMode                               m_mode   = RenderingMode::eRasterizer;
  rs::ClipId                                  m_clip;
  float                                       m_clipTime = 0.f;
  AnimationControl                            m_animation;
};
