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

#include <filesystem>
#include <functional>
#include <source_location>

#include <glm/gtc/matrix_transform.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_editor.hpp"
#include "scene_coexistence.hpp"
#include "tinygltf_utils.hpp"

using namespace rs;

namespace {

// A stand-in for the renderer's class, so the check sees this compiler's own spelling of a member
// and of a lambda inside one
struct GltfRenderer
{
  std::source_location onAttach() const
  {
    return [] { return std::source_location::current(); }();
  }
  std::source_location setNodeTransforms() const { return std::source_location::current(); }
  std::source_location updateAnimation() const { return std::source_location::current(); }
};

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

// The file a material's base colour samples through its texture table, and the sampler's wrap; "" for none
std::string baseMap(RetainedScene& scene, MaterialId id)
{
  const MaterialRecord* m = scene.material(id);
  const int             i = m ? m->payload.pbrMetallicRoughness.baseColorTexture.index : -1;
  if(i < 0 || !m->textures || i >= int(m->textures->size()))
    return "";
  const TextureRef&    ref = (*m->textures)[size_t(i)];
  const TextureRecord* t   = scene.texture(ref.texture);
  return (t ? t->uri : std::string("?")) + "|" + std::to_string(ref.sampler.wrapS);
}

// A bake as the host's FvMaterials::baked makes one: an image and a texture appended (the editor's
// flags: the first textures of a model with none are a full change), a material sampling it
void bake(nvvkgltf::Scene& legacy, const char* uri, int material)
{
  tinygltf::Model& model = legacy.getModel();
  const bool       first = model.images.empty() || model.textures.empty();
  tinygltf::Image  image;
  image.uri = uri;
  model.images.push_back(image);
  tinygltf::Texture texture;
  texture.source = int(model.images.size()) - 1;
  model.textures.push_back(texture);
  (first ? legacy.getDirtyFlags().texturesChanged : legacy.getDirtyFlags().texturesTailChanged) = true;
  legacy.editor().getMaterialForEdit(material).pbrMetallicRoughness.baseColorTexture.index = int(model.textures.size()) - 1;
  legacy.markMaterialDirty(material);
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
    EXPECT_EQ(baseMap(bridged, a[k].material), baseMap(fresh, b[k].material)) << step << ": item " << k;
  }
  const std::vector<LightItem> la = bridged.lightItems(), lb = fresh.lightItems();
  ASSERT_EQ(la.size(), lb.size()) << step;
  for(size_t k = 0; k < la.size(); k++)
  {
    EXPECT_EQ(la[k].world, lb[k].world) << step;
    EXPECT_EQ(la[k].shown, lb[k].shown) << step;
    EXPECT_EQ(bridged.light(la[k].light)->payload, fresh.light(lb[k].light)->payload) << step;
    EXPECT_EQ(bridged.light(la[k].light)->ies, fresh.light(lb[k].light)->ies) << step;
  }
}

// The fixture's lamp an EXT_lights_ies light too, over a profile in the buffer
tinygltf::Model withIesLamp(tinygltf::Model m)
{
  const std::string    file = "IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 3 1 1 2 0 0 0\n1 1 100\n0 90 180\n0\n100 50 0\n";
  tinygltf::Buffer&    b    = m.buffers[0];
  tinygltf::BufferView view;
  view.buffer     = 0;
  view.byteOffset = b.data.size();
  view.byteLength = file.size();
  b.data.insert(b.data.end(), file.begin(), file.end());
  m.bufferViews.push_back(view);
  tinygltf::Value::Object profile, root, ref;
  profile["bufferView"]                       = tinygltf::Value(int(m.bufferViews.size()) - 1);
  root["lights"]                              = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(profile)});
  m.extensions[EXT_LIGHTS_IES_EXTENSION_NAME] = tinygltf::Value(root);
  m.extensionsUsed.push_back(EXT_LIGHTS_IES_EXTENSION_NAME);
  ref["light"]                                         = tinygltf::Value(0);
  ref["multiplier"]                                    = tinygltf::Value(2.0);
  m.nodes[1].extensions[EXT_LIGHTS_IES_EXTENSION_NAME] = tinygltf::Value(ref);
  return m;
}

}  // namespace

TEST(SceneCoexistence, WhatRetainedCannotTakeIsNamedOnceUnderItsProp)
{
  tinygltf::Model plain = fixture();
  EXPECT_TRUE(retainedCannotTake(plain, false).empty());

  // skins, morph targets and a clip's node channels are drawn
  tinygltf::Model m = fixture();
  m.skins.emplace_back();
  m.nodes[1].skin = 0;
  m.meshes[1].primitives[0].targets.push_back({{"POSITION", 0}});
  m.animations.emplace_back();
  m.animations[0].channels.push_back({});
  m.animations[0].channels[0].target_node = 2;
  m.animations[0].channels[0].target_path = "rotation";
  m.extensionsUsed                        = {EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME};  // micromaps are drawn
  EXPECT_TRUE(retainedCannotTake(m, false).empty());

  // a pointer channel, named under its prop, once though two channels have one
  for(int node : {2, 1})
  {
    tinygltf::AnimationChannel c;
    c.target_node                              = node;
    c.target_path                              = "pointer";
    c.target_extensions[KHR_ANIMATION_POINTER] = tinygltf::Value(tinygltf::Value::Object{});
    m.animations[0].channels.push_back(c);
  }
  m.extensionsUsed.push_back(KHR_INTERACTIVITY_EXTENSION_NAME);
  const std::vector<std::string> what = retainedCannotTake(m, true);
  ASSERT_EQ(what.size(), 3u);
  EXPECT_EQ(what[0], "KHR_animation_pointer (under 'propB')");
  EXPECT_EQ(what[1], "KHR_interactivity");
  EXPECT_EQ(what[2], "glTF 2.1 external assets");
}

TEST(SceneCoexistence, TheBackendASceneTakes)
{
  for(SceneBackendSetting s : {SceneBackendSetting::eAuto, SceneBackendSetting::eLegacy, SceneBackendSetting::eRetained})
    EXPECT_EQ(parseSceneBackendSetting(sceneBackendSettingName(s)), s);
  EXPECT_FALSE(parseSceneBackendSetting("Retained").has_value());

  const std::vector<std::string> none, skin = {"KHR_animation_pointer (under 'figure')"};
  EXPECT_EQ(chooseSceneBackend(SceneBackendSetting::eAuto, true, none).backend, SceneBackend::eRetained);
  const SceneBackendChoice fallback = chooseSceneBackend(SceneBackendSetting::eAuto, true, skin);
  EXPECT_EQ(fallback.backend, SceneBackend::eLegacy);
  EXPECT_NE(fallback.reason.find("KHR_animation_pointer (under 'figure')"), std::string::npos);  // the log names the prop
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
  LegacyDirtyBridge       bridge(importGltf(retained, legacy.getModel()));
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

// An IES light's multiplier and colour are its node's: an edit to them is carried with the node
TEST(SceneCoexistence, TheBridgeCarriesAnIesLightsTint)
{
  nvvkgltf::Scene legacy;
  legacy.takeModel(withIesLamp(fixture()));
  ASSERT_TRUE(legacy.valid());
  RetainedScene           retained;
  LegacyDirtyBridge       bridge(importGltf(retained, legacy.getModel()));
  legacy.clearDirtyFlags();
  ASSERT_TRUE(retained.light(retained.entity(bridge.map().entities[1])->light)->ies);

  tinygltf::utils::setNodeIesLight(legacy.editor().getNodeForEdit(1), {.light = 0, .multiplier = 5.f, .color = {0.f, 1.f, 0.f}});
  legacy.markNodeDirty(1);
  EXPECT_EQ(bridge.carry(legacy, retained), LegacyDirtyBridge::Result::eCarried);
  legacy.clearDirtyFlags();
  RetainedScene fresh;
  importGltf(fresh, legacy.getModel());
  expectSameDrawing(retained, fresh, "an IES tint");
  EXPECT_EQ(retained.light(retained.entity(bridge.map().entities[1])->light)->ies->multiplier, 5.f);
}

// The legacy-accessor check's allow-list, in MSVC's and clang's spellings of a function, a lambda
// inside an allowed member, and a member whose name an allowed one prefixes
TEST(SceneCoexistence, TheLegacyUsesARetainedSessionAllows)
{
  EXPECT_TRUE(legacySceneUseAllowed("F:\\x\\src\\fv_host.cpp", "void __cdecl fv::FvHost::Impl::compose(void)"));
  EXPECT_TRUE(legacySceneUseAllowed("/x/src/sky_sun.cpp", "anything"));
  EXPECT_TRUE(legacySceneUseAllowed("src/renderer.hpp", "struct nvvkgltf::Scene *__cdecl GltfRenderer::editScene(void)"));
  EXPECT_TRUE(legacySceneUseAllowed("src/renderer.cpp", "void GltfRenderer::setNodeTransforms(std::span<const int>)"));
  EXPECT_TRUE(legacySceneUseAllowed("src/renderer.cpp", "auto __cdecl GltfRenderer::onAttach::<lambda_3>::operator ()(int) const"));
  EXPECT_TRUE(legacySceneUseAllowed("src/renderer.cpp",
                                    "GltfRenderer::onAttach(nvapp::Application *)::(anonymous class)::operator()() const"));

  EXPECT_FALSE(legacySceneUseAllowed("src/renderer.cpp", "void __cdecl GltfRenderer::updateSceneChanges(struct VkCommandBuffer_T *)"));
  EXPECT_FALSE(legacySceneUseAllowed("src/renderer.cpp", "void __cdecl GltfRenderer::onAttachSomething(void)"));
  EXPECT_FALSE(legacySceneUseAllowed("src/rasterizer.cpp", "void __cdecl Rasterizer::onRender(void)"));
  EXPECT_FALSE(legacySceneUseAllowed("src/not_fv_host.cpp", "f"));
  EXPECT_TRUE(legacySceneUseAllowed("src/renderer.cpp", "GltfRenderer::setNodeTransforms"));  // no signature

  // as this compiler names them
  const GltfRenderer stand;
  EXPECT_TRUE(legacySceneUseAllowed("renderer.cpp", stand.onAttach().function_name())) << stand.onAttach().function_name();
  EXPECT_TRUE(legacySceneUseAllowed("renderer.cpp", stand.setNodeTransforms().function_name()))
      << stand.setNodeTransforms().function_name();
  EXPECT_FALSE(legacySceneUseAllowed("renderer.cpp", stand.updateAnimation().function_name()))
      << stand.updateAnimation().function_name();

  // once per call site, and never for an allowed one
  LegacyUseCheck check;
  EXPECT_TRUE(check.firstUnlisted("src/renderer.cpp", 10, "void GltfRenderer::updateAnimation(void)"));
  EXPECT_FALSE(check.firstUnlisted("src/renderer.cpp", 10, "void GltfRenderer::updateAnimation(void)"));
  EXPECT_TRUE(check.firstUnlisted("src/renderer.cpp", 11, "void GltfRenderer::updateAnimation(void)"));
  EXPECT_FALSE(check.firstUnlisted("src/renderer.cpp", 12, "void GltfRenderer::cleanupScene(void)"));
}

// Over a textured model: an image replaced (a full texture change), a texture removed from the end,
// a sampler edited, a re-parse
TEST(SceneCoexistence, WhatTheBridgeCantCarryIsAReimport)
{
  nvvkgltf::Scene legacy;
  legacy.takeModel(fixture());
  bake(legacy, "maps/first.png", 0);
  legacy.clearDirtyFlags();
  for(auto set :
      std::vector<std::function<void(nvvkgltf::Scene&)>>{[](auto& s) { s.getDirtyFlags().texturesChanged = true; },
                                                         [](auto& s) {
                                                           s.getModel().textures.pop_back();
                                                           s.getDirtyFlags().texturesTailChanged = true;
                                                         },
                                                         [](auto& s) { s.getDirtyFlags().samplers.insert(0); },
                                                         [](auto& s) { s.getDirtyFlags().allRenderNodesDirty = true; },
                                                         [](auto& s) { s.getDirtyFlags().primitivesChanged = true; }})
  {
    nvvkgltf::Scene copy;
    copy.takeModel(tinygltf::Model(legacy.getModel()));
    RetainedScene     retained;
    LegacyDirtyBridge bridge(importGltf(retained, copy.getModel()));
    copy.clearDirtyFlags();
    set(copy);
    EXPECT_EQ(bridge.carry(copy, retained), LegacyDirtyBridge::Result::eReimport);
  }
}

// Bakes are carried, not re-imported: the first onto a model with no textures, another after it, and
// a texture over an image the model has; the materials take the table with them appended, and draw
// as a fresh import of the legacy model does
TEST(SceneCoexistence, TheBridgeCarriesABake)
{
  nvvkgltf::Scene legacy;
  legacy.takeModel(fixture());
  RetainedScene     retained;
  LegacyDirtyBridge bridge(importGltf(retained, legacy.getModel()));
  legacy.clearDirtyFlags();
  const auto step = [&](const char* what, const std::function<void()>& op) {
    op();
    EXPECT_EQ(bridge.carry(legacy, retained), LegacyDirtyBridge::Result::eCarried) << what;
    legacy.clearDirtyFlags();
    RetainedScene fresh;
    importGltf(fresh, legacy.getModel());
    expectSameDrawing(retained, fresh, what);
  };
  step("onto none", [&] { bake(legacy, "maps/red.png", 0); });
  step("another", [&] { bake(legacy, "maps/blue.png", 1); });
  step("a texture over the first image, clamped", [&] {
    tinygltf::Model&  model = legacy.getModel();
    tinygltf::Sampler clamp;
    clamp.wrapS = clamp.wrapT = TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE;
    model.samplers.push_back(clamp);
    tinygltf::Texture texture;
    texture.source  = 0;
    texture.sampler = int(model.samplers.size()) - 1;
    model.textures.push_back(texture);
    legacy.getDirtyFlags().texturesTailChanged                                        = true;
    legacy.editor().getMaterialForEdit(1).pbrMetallicRoughness.baseColorTexture.index = int(model.textures.size()) - 1;
    legacy.markMaterialDirty(1);
  });
  EXPECT_NE(baseMap(retained, retained.drawItems()[1].material).find("|33071"), std::string::npos);  // clamped
}

// A dial on the legacy scene (a host's vertex upload) carried onto the import's record: positions
// and normals whole as the live copy holds them, and tangents made again where the moved vertices
// reach, the same bits a full computeTangents makes; a rebind's inverse binds and a node's morph
// weights carried as well
TEST(SceneCoexistence, TheBridgeCarriesDialsRebindsAndWeights)
{
  tinygltf::Model model = fixture();
  // a normal map on the first prop's material, so the import made it tangents; a skin on its node
  model.materials[0].normalTexture.index = 0;
  model.images.push_back({});
  model.images.back().uri = "normal.png";
  model.textures.push_back({});
  model.textures.back().source = 0;
  model.skins.push_back({});
  model.skins.back().joints = {1};
  model.nodes[0].skin       = 0;
  model.nodes[0].weights    = {0.0};
  nvvkgltf::Scene legacy;
  legacy.takeModel(std::move(model));
  ASSERT_TRUE(legacy.valid());
  RetainedScene     retained;
  LegacyDirtyBridge bridge(importGltf(retained, legacy.getModel()));
  legacy.clearDirtyFlags();
  const MeshId mesh = bridge.map().meshes[0].at(0);
  ASSERT_FALSE(retained.mesh(mesh)->tangents.empty());
  const int prim = legacy.getMeshRenderPrimitive(0, 0);
  ASSERT_GE(prim, 0);

  const auto tangentsOf = [](const MeshRecord& r) {
    std::vector<glm::vec4> t(r.positions.size(), glm::vec4(0.f));
    tinygltf::utils::computeTangents(r.positions, r.normals, r.uv0, r.indices, t);
    return t;
  };
  // whole
  std::vector<glm::vec3> positions = retained.mesh(mesh)->positions;
  for(glm::vec3& p : positions)
    p.y *= 2.f;
  // (the quad has no normals: normals for it are refused)
  EXPECT_FALSE(bridge.carryVertices(legacy, retained, prim, positions, std::vector<glm::vec3>(positions.size()), {}));
  ASSERT_TRUE(bridge.carryVertices(legacy, retained, prim, positions, {}, {}));
  EXPECT_EQ(retained.mesh(mesh)->positions, positions);
  EXPECT_EQ(retained.mesh(mesh)->tangents, tangentsOf(*retained.mesh(mesh)));
  // one vertex moved: the one-ring made again, the rest kept, the same bits as a full remake
  positions[2] = {1.5f, 2.5f, 0.25f};
  const std::vector<uint32_t> moved{2};
  ASSERT_TRUE(bridge.carryVertices(legacy, retained, prim, positions, {}, moved));
  EXPECT_EQ(retained.mesh(mesh)->positions, positions);
  EXPECT_EQ(retained.mesh(mesh)->tangents, tangentsOf(*retained.mesh(mesh)));
  // the wrong size is refused, and nothing changes
  EXPECT_FALSE(bridge.carryVertices(legacy, retained, prim, std::vector<glm::vec3>(3), {}, {}));
  EXPECT_EQ(retained.mesh(mesh)->positions, positions);

  // a rebind
  const std::vector<glm::mat4> ibm{glm::translate(glm::mat4(1.f), glm::vec3(0, -2, 0))};
  ASSERT_TRUE(bridge.carryInverseBinds(retained, 0, ibm));
  const Entity* propA = retained.entity(bridge.map().entities[0]);
  EXPECT_EQ(retained.skin(propA->mesh->skin)->inverseBind, ibm);
  EXPECT_FALSE(bridge.carryInverseBinds(retained, 1, ibm));

  // a node's morph weights
  legacy.editor().getNodeForEdit(0).weights = {0.75};
  legacy.markNodeDirty(0);
  EXPECT_EQ(bridge.carry(legacy, retained), LegacyDirtyBridge::Result::eCarried);
  EXPECT_EQ(retained.entity(bridge.map().entities[0])->mesh->morphWeights, std::vector<float>{0.75f});
}

// A clip played on the legacy scene in a retained session (GltfRenderer::evaluateClip): what its
// channels wrote, nodes and morph weights, carried frame by frame, leaves the retained scene as a
// fresh import of the posed model
TEST(SceneCoexistence, TheBridgeCarriesAClipsFrames)
{
  namespace fs = std::filesystem;
  const fs::path models = fs::path(GLTF_SAMPLE_ASSETS_PATH) / "Models";
  int            tested = 0;
  for(const char* file : {"RiggedSimple/glTF/RiggedSimple.gltf", "AnimatedMorphCube/glTF/AnimatedMorphCube.gltf",
                          "CesiumMan/glTF/CesiumMan.gltf"})
  {
    std::error_code ec;
    if(!fs::exists(models / file, ec))
      continue;
    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        err, warn;
    ASSERT_TRUE(loader.LoadASCIIFromFile(&model, &err, &warn, (models / file).string())) << file;
    nvvkgltf::Scene legacy;
    legacy.takeModel(std::move(model));
    ASSERT_TRUE(legacy.valid()) << file;
    RetainedScene     retained;
    LegacyDirtyBridge bridge(importGltf(retained, legacy.getModel()));
    legacy.clearDirtyFlags();
    nvvkgltf::AnimationInfo& info = legacy.animation().getAnimationInfo(0);
    for(int frame = 0; frame < 5; frame++)
    {
      info.incrementTime(std::max(info.end - info.start, 0.1f) / 4.f, true);
      ASSERT_TRUE(legacy.animation().updateAnimation(0)) << file;
      EXPECT_EQ(bridge.carry(legacy, retained), LegacyDirtyBridge::Result::eCarried) << file;
      legacy.clearDirtyFlags();
      RetainedScene fresh;
      importGltf(fresh, legacy.getModel());
      const std::vector<EntityId> a = retained.entitiesInOrder(), b = fresh.entitiesInOrder();
      ASSERT_EQ(a.size(), b.size()) << file;
      for(size_t i = 0; i < a.size(); i++)
      {
        const Entity &x = *retained.entity(a[i]), &y = *fresh.entity(b[i]);
        EXPECT_TRUE(x.local == y.local) << file << " frame " << frame << ", " << x.name;
        if(x.mesh && y.mesh)
          EXPECT_EQ(x.mesh->morphWeights, y.mesh->morphWeights) << file << " frame " << frame << ", " << x.name;
      }
      expectSameDrawing(retained, fresh, file);
    }
    tested++;
  }
  if(tested == 0)
    GTEST_SKIP() << "no sample assets";
}
