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

// The retained scene beside the legacy one (src/scene_coexistence.hpp): which backend a scene
// takes, and the bridge oracle: after each editor op on the legacy scene, the bridged retained scene
// draws what a fresh document-order import of the legacy model draws.

#include <gtest/gtest.h>

#include <functional>

#include <glm/gtc/matrix_transform.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "scene_coexistence.hpp"
#include "tinygltf_utils.hpp"

using namespace rs;

namespace {

// Two props drawing one quad with their own materials, the first with a lamp below it
tinygltf::Model fixture()
{
  tinygltf::Model m;
  m.buffers.emplace_back();
  const std::vector<glm::vec3> pos = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const std::vector<uint32_t>  idx = {0, 1, 2, 0, 2, 3};
  const int p = tinygltf::utils::appendAccessor(m, 0, pos.data(), pos.size() * 12, TINYGLTF_COMPONENT_TYPE_FLOAT,
                                                TINYGLTF_TYPE_VEC3, 4);
  const int i = tinygltf::utils::appendAccessor(m, 0, idx.data(), idx.size() * 4, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,
                                                TINYGLTF_TYPE_SCALAR, 6);
  for(int k = 0; k < 2; k++)
  {
    tinygltf::Mesh      mesh;
    tinygltf::Primitive prim;
    prim.attributes = {{"POSITION", p}};
    prim.indices    = i;
    prim.material   = k;
    mesh.primitives = {prim};
    m.meshes.push_back(mesh);
    tinygltf::Material material;
    material.name = k ? "blue" : "red";
    m.materials.push_back(material);
  }
  tinygltf::Light lamp;
  lamp.type      = "point";
  lamp.intensity = 20.0;
  m.lights       = {lamp};

  tinygltf::Node a, light, b;
  a.name            = "propA";
  a.mesh            = 0;
  a.children        = {1};
  light.name        = "lamp";
  light.light       = 0;
  light.translation = {0.0, 2.0, 0.0};
  b.name            = "propB";
  b.mesh            = 1;
  b.translation     = {3.0, 0.0, 0.0};
  m.nodes           = {a, light, b};
  m.scenes.push_back({});
  m.scenes[0].nodes = {0, 2};
  m.defaultScene    = 0;
  return m;
}

// What the two retained scenes draw and light, item for item
void expectSameDrawing(RetainedScene& bridged, RetainedScene& fresh, const char* step)
{
  const std::vector<DrawItem> a = bridged.drawItems(), b = fresh.drawItems();
  ASSERT_EQ(a.size(), b.size()) << step;
  for(size_t k = 0; k < a.size(); k++)
  {
    for(int c = 0; c < 4; c++)
      for(int r = 0; r < 4; r++)
        EXPECT_NEAR(a[k].world[c][r], b[k].world[c][r], 1e-5f) << step << ": item " << k;
    EXPECT_EQ(a[k].shown, b[k].shown) << step << ": item " << k;
    EXPECT_EQ(bridged.material(a[k].material)->payload, fresh.material(b[k].material)->payload) << step << ": item " << k;
  }
  const std::vector<LightItem> la = bridged.lightItems(), lb = fresh.lightItems();
  ASSERT_EQ(la.size(), lb.size()) << step;
  for(size_t k = 0; k < la.size(); k++)
  {
    EXPECT_EQ(la[k].world, lb[k].world) << step;
    EXPECT_EQ(la[k].shown, lb[k].shown) << step;
    EXPECT_EQ(bridged.light(la[k].light)->payload, fresh.light(lb[k].light)->payload) << step;
  }
}

}  // namespace

TEST(SceneCoexistence, WhatRetainedCannotTakeIsNamedOnceUnderItsProp)
{
  tinygltf::Model plain = fixture();
  EXPECT_TRUE(retainedCannotTake(plain, false).empty());

  tinygltf::Model m = fixture();
  m.skins.emplace_back();
  m.nodes[1].skin = 0;  // under propA
  m.nodes[2].skin = 0;
  m.meshes[1].primitives[0].targets.push_back({{"POSITION", 0}});
  m.animations.emplace_back();
  m.animations[0].channels.push_back({});
  m.animations[0].channels[0].target_node = 2;
  m.extensionsUsed                        = {KHR_INTERACTIVITY_EXTENSION_NAME};
  const std::vector<std::string> what     = retainedCannotTake(m, true);
  ASSERT_EQ(what.size(), 5u);
  EXPECT_EQ(what[0], "skinning (under 'propA')");  // once, though propB has one too
  EXPECT_EQ(what[1], "morph targets (under 'propB')");
  EXPECT_EQ(what[2], "animation (under 'propB')");
  EXPECT_EQ(what[3], "KHR_interactivity");
  EXPECT_EQ(what[4], "glTF 2.1 external assets");
}

TEST(SceneCoexistence, TheBackendASceneTakes)
{
  for(SceneBackendSetting s : {SceneBackendSetting::eAuto, SceneBackendSetting::eLegacy, SceneBackendSetting::eRetained})
    EXPECT_EQ(parseSceneBackendSetting(sceneBackendSettingName(s)), s);
  EXPECT_FALSE(parseSceneBackendSetting("Retained").has_value());

  const std::vector<std::string> none, skin = {"skinning (under 'figure')"};
  EXPECT_EQ(chooseSceneBackend(SceneBackendSetting::eAuto, true, none).backend, SceneBackend::eRetained);
  const SceneBackendChoice fallback = chooseSceneBackend(SceneBackendSetting::eAuto, true, skin);
  EXPECT_EQ(fallback.backend, SceneBackend::eLegacy);
  EXPECT_NE(fallback.reason.find("skinning (under 'figure')"), std::string::npos);  // the log names the prop
  EXPECT_EQ(chooseSceneBackend(SceneBackendSetting::eLegacy, true, none).backend, SceneBackend::eLegacy);
  const SceneBackendChoice forced = chooseSceneBackend(SceneBackendSetting::eRetained, true, skin);
  EXPECT_EQ(forced.backend, SceneBackend::eRetained);
  EXPECT_NE(forced.reason.find("not drawn"), std::string::npos);
  // a windowed session stays legacy whatever the setting
  for(SceneBackendSetting s : {SceneBackendSetting::eAuto, SceneBackendSetting::eRetained})
    EXPECT_EQ(chooseSceneBackend(s, false, none).backend, SceneBackend::eLegacy);
}

// The bridge oracle: each editor op the host makes on the legacy scene, carried, leaves the
// retained scene drawing what a fresh import of the edited legacy model draws
TEST(SceneCoexistence, TheBridgeCarriesTheHostsEdits)
{
  nvvkgltf::Scene legacy;
  legacy.takeModel(fixture());
  ASSERT_TRUE(legacy.valid());
  RetainedScene           retained;
  const LegacyDirtyBridge bridge(importGltf(retained, legacy.getModel()));
  legacy.clearDirtyFlags();

  const auto step = [&](const char* what, const std::function<void()>& op) {
    op();
    EXPECT_EQ(bridge.carry(legacy, retained), LegacyDirtyBridge::Result::eCarried) << what;
    legacy.clearDirtyFlags();
    RetainedScene fresh;
    importGltf(fresh, legacy.getModel());
    expectSameDrawing(retained, fresh, what);
  };
  EXPECT_EQ(bridge.carry(legacy, retained), LegacyDirtyBridge::Result::eNothing);

  step("move",
       [&] { legacy.editor().setNodeTRS(2, {5.f, 1.f, 0.f}, glm::angleAxis(0.5f, glm::vec3(0, 1, 0)), glm::vec3(2.f)); });
  step("move the parent of a light",
       [&] { legacy.editor().setNodeTRS(0, {0.f, 0.f, -1.f}, glm::quat(1, 0, 0, 0), glm::vec3(1.f)); });
  step("hide", [&] {
    tinygltf::utils::setNodeVisibility(legacy.editor().getNodeForEdit(0), {false});
    legacy.editor().updateVisibility(0);
  });
  step("show", [&] {
    tinygltf::utils::setNodeVisibility(legacy.editor().getNodeForEdit(0), {true});
    legacy.editor().updateVisibility(0);
  });
  step("material", [&] {
    legacy.editor().getMaterialForEdit(1).pbrMetallicRoughness.baseColorFactor = {0.0, 0.0, 1.0, 1.0};
    legacy.markMaterialDirty(1);
  });
  step("light off", [&] {
    legacy.editor().getLightForEdit(0).intensity = 0.0;
    legacy.markLightDirty(0);
  });
}

TEST(SceneCoexistence, WhatTheBridgeCantCarryIsAReimport)
{
  nvvkgltf::Scene legacy;
  legacy.takeModel(fixture());
  RetainedScene           retained;
  const LegacyDirtyBridge bridge(importGltf(retained, legacy.getModel()));
  for(auto set : std::vector<std::function<void(nvvkgltf::Scene::DirtyFlags&)>>{
          [](auto& df) { df.texturesChanged = true; }, [](auto& df) { df.texturesTailChanged = true; },
          [](auto& df) { df.samplers.insert(0); }, [](auto& df) { df.allRenderNodesDirty = true; },
          [](auto& df) { df.primitivesChanged = true; }})
  {
    legacy.clearDirtyFlags();
    set(legacy.getDirtyFlags());
    EXPECT_EQ(bridge.carry(legacy, retained), LegacyDirtyBridge::Result::eReimport);
  }
}
