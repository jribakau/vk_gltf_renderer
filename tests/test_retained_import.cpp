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

// glTF into the retained scene (oracle 3 of daz-viewer docs/retained-scene-plan.md): what the
// importer draws equals what nvvkgltf::Scene draws of the same model, item for render node, in the
// same order: world matrix, visibility, material, positions, indices and generated tangents bit for
// bit, and the lights in the same order.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nvutils/file_operations.hpp>

#include "gltf_scene.hpp"
#include "ies_profile.hpp"
#include "scene_import_gltf.hpp"
#include "tinygltf_utils.hpp"

namespace {

int add(tinygltf::Model& m, const void* data, size_t bytes, int component, int type, size_t count)
{
  return tinygltf::utils::appendAccessor(m, 0, data, bytes, component, type, count);
}

// A model with what the importer has to get right: a mesh of two material groups over one vertex
// set (one normal-mapped, so it gets tangents from its own triangles only), a non-indexed mesh, an
// undrawable primitive, a hierarchy of TRS and matrix nodes, a hidden node, a node drawing a mesh a
// second time, GPU instancing, a light and a camera.
tinygltf::Model fixture()
{
  tinygltf::Model m;
  m.buffers.emplace_back();
  const std::vector<glm::vec3> pos = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const std::vector<glm::vec3> nrm(4, glm::vec3(0, 0, 1));
  const std::vector<glm::vec2> uv  = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  const std::vector<uint32_t>  i0 = {0, 1, 2}, i1 = {0, 2, 3};
  const int p  = add(m, pos.data(), pos.size() * 12, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 4);
  const int n  = add(m, nrm.data(), nrm.size() * 12, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 4);
  const int t  = add(m, uv.data(), uv.size() * 8, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, 4);
  const int a0 = add(m, i0.data(), 12, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, 3);
  const int a1 = add(m, i1.data(), 12, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, 3);
  const std::vector<glm::vec3> tri = {{0, 0, 2}, {2, 0, 2}, {0, 2, 2}};
  const int p2 = add(m, tri.data(), tri.size() * 12, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 3);
  const int empty = add(m, pos.data(), 0, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, 0);

  tinygltf::Mesh groups;
  groups.name = "groups";
  tinygltf::Primitive g0;
  g0.attributes = {{"POSITION", p}, {"NORMAL", n}, {"TEXCOORD_0", t}};
  g0.indices    = a0;
  g0.material   = 0;
  tinygltf::Primitive g1 = g0;
  g1.indices             = a1;
  g1.material            = 1;
  tinygltf::Primitive bad = g0;  // no triangles: neither scene draws it
  bad.indices             = empty;
  groups.primitives       = {g0, bad, g1};
  tinygltf::Mesh other;
  other.name = "other";
  tinygltf::Primitive o;
  o.attributes = {{"POSITION", p2}};
  o.material   = 1;
  other.primitives = {o};
  m.meshes = {groups, other};

  tinygltf::Material normalMapped;
  normalMapped.name                = "normal mapped";
  normalMapped.normalTexture.index = 0;
  tinygltf::Material plain;
  plain.name                                 = "plain";
  plain.pbrMetallicRoughness.baseColorFactor = {1, 0, 0, 1};
  m.materials = {normalMapped, plain};
  tinygltf::Image image;
  image.uri = "normal.png";
  m.images  = {image};
  tinygltf::Texture texture;
  texture.source = 0;
  m.textures     = {texture};

  // instancing: two copies along x
  const std::vector<glm::vec3> offsets = {{0, 0, 0}, {3, 0, 0}};
  const int                    inst    = add(m, offsets.data(), 24, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 2);

  tinygltf::Light lamp;
  lamp.type      = "point";
  lamp.intensity = 5;
  m.lights       = {lamp};
  m.extensionsUsed.push_back("KHR_lights_punctual");
  tinygltf::Camera cam;
  cam.type             = "perspective";
  cam.perspective.yfov = 0.6;
  cam.perspective.znear = 0.1;
  m.cameras = {cam};

  tinygltf::Node root;
  root.name        = "root";
  root.translation = {1, 2, 3};
  root.rotation    = {0, 0.7071067811865476, 0, 0.7071067811865476};
  root.children    = {1, 2, 3, 4, 5, 6};
  tinygltf::Node a;
  a.name  = "a";
  a.mesh  = 0;
  a.scale = {2, 2, 2};
  tinygltf::Node b;
  b.name   = "b";
  b.mesh   = 1;
  b.matrix = std::vector<double>(glm::value_ptr(glm::translate(glm::dmat4(1.0), {0, 5, 0})), glm::value_ptr(glm::translate(glm::dmat4(1.0), {0, 5, 0})) + 16);
  tinygltf::Node light;
  light.name  = "lamp";
  light.light = 0;
  light.translation = {0, 3, 0};
  tinygltf::Node hidden;
  hidden.name = "hidden";
  hidden.mesh = 0;
  tinygltf::Value::Object vis;
  vis["visible"]                          = tinygltf::Value(false);
  hidden.extensions["KHR_node_visibility"] = tinygltf::Value(vis);
  tinygltf::Node copies;
  copies.name = "copies";
  copies.mesh = 1;
  tinygltf::Value::Object attrs;
  attrs["TRANSLATION"] = tinygltf::Value(inst);
  tinygltf::Value::Object instancing;
  instancing["attributes"]                     = tinygltf::Value(attrs);
  copies.extensions["EXT_mesh_gpu_instancing"] = tinygltf::Value(instancing);
  tinygltf::Node camera;
  camera.name   = "camera";
  camera.camera = 0;
  m.nodes = {root, a, b, light, hidden, copies, camera};
  m.scenes.emplace_back().nodes = {0};
  m.defaultScene                = 0;
  return m;
}

bool nearly(const glm::mat4& x, const glm::mat4& y)
{
  for(int c = 0; c < 4; c++)
    for(int r = 0; r < 4; r++)
      if(std::abs(x[c][r] - y[c][r]) > 1e-5f)
        return false;
  return true;
}

template <class T>
std::vector<T> read(const tinygltf::Model& m, int accessor)
{
  std::vector<T>     storage;
  std::span<const T> d = tinygltf::utils::getAccessorData(m, m.accessors[size_t(accessor)], &storage);
  return {d.begin(), d.end()};
}

// The two scenes of one model, compared draw item by render node.
void expectParity(const tinygltf::Model& model, const std::string& what)
{
  nvvkgltf::Scene legacy;
  legacy.takeModel(tinygltf::Model(model));
  if(!legacy.valid())
  {
    ADD_FAILURE() << what << ": the legacy scene did not load";
    return;
  }
  const tinygltf::Model& lm = legacy.getModel();

  rs::RetainedScene      retained;
  const rs::ImportResult imported = rs::importGltf(retained, model, {.prop = 7});
  const std::vector<rs::DrawItem> items = retained.drawItems();
  const auto&                     nodes = legacy.getRenderNodes();
  ASSERT_EQ(items.size(), nodes.size()) << what;
  for(size_t i = 0; i < items.size(); i++)
  {
    SCOPED_TRACE(what + ", render node " + std::to_string(i));
    const nvvkgltf::RenderNode& rn   = nodes[i];
    const rs::DrawItem&         item = items[i];
    EXPECT_TRUE(nearly(item.world, rn.worldMatrix));
    EXPECT_EQ(item.shown, rn.visible);
    EXPECT_EQ(retained.entity(item.entity)->owner.local, rn.refNodeID);
    const rs::MaterialRecord* mat = retained.material(item.material);
    ASSERT_NE(mat, nullptr);
    EXPECT_EQ(mat->owner.local, rn.materialID);

    const tinygltf::Primitive& prim = *legacy.getRenderPrimitive(size_t(rn.renderPrimID)).pPrimitive;
    const rs::MeshRecord*      mesh = retained.mesh(item.mesh);
    ASSERT_NE(mesh, nullptr);
    const std::vector<glm::vec3> pos = read<glm::vec3>(lm, prim.attributes.at("POSITION"));
    ASSERT_EQ(mesh->positions.size(), pos.size());
    EXPECT_EQ(std::memcmp(mesh->positions.data(), pos.data(), pos.size() * sizeof(glm::vec3)), 0);
    const rs::Submesh&    run = mesh->submeshes[item.submesh];
    std::vector<uint32_t> idx;
    if(prim.indices >= 0)
      idx = read<uint32_t>(lm, prim.indices);
    else
      for(uint32_t v = 0; v < pos.size(); v++)
        idx.push_back(v);
    ASSERT_EQ(run.indexCount, idx.size());
    EXPECT_TRUE(std::equal(idx.begin(), idx.end(), mesh->indices.begin() + run.firstIndex));
    // tangents where the legacy scene made or read them, bit for bit
    if(auto it = prim.attributes.find("TANGENT"); it != prim.attributes.end())
    {
      const std::vector<glm::vec4> tan = read<glm::vec4>(lm, it->second);
      ASSERT_EQ(mesh->tangents.size(), tan.size());
      EXPECT_EQ(std::memcmp(mesh->tangents.data(), tan.data(), tan.size() * sizeof(glm::vec4)), 0);
    }
    else if(!mesh->tangents.empty())
    {
      // the legacy scene gives tangents only to the primitives that need them, which splits a
      // vertex set (the TANGENT attribute changes their key); here the set keeps one vertex set,
      // with the tangents a sibling submesh needed
      const auto& parts = retained.entity(item.entity)->mesh->parts;
      const auto  part  = std::find_if(parts.begin(), parts.end(), [&](const rs::MeshPart& p) { return p.mesh == item.mesh; });
      ASSERT_NE(part, parts.end());
      const bool siblingNeeds = std::any_of(mesh->submeshes.begin(), mesh->submeshes.end(), [&](const rs::Submesh& s) {
        const rs::MaterialRecord* m = retained.material(part->materials[s.material]);
        return m && rs::materialNeedsTangents(m->payload);
      });
      EXPECT_TRUE(siblingNeeds);
    }
  }

  const std::vector<rs::LightItem> lights = retained.lightItems();
  ASSERT_EQ(lights.size(), legacy.getRenderLights().size()) << what;
  for(size_t i = 0; i < lights.size(); i++)
  {
    EXPECT_TRUE(nearly(lights[i].world, legacy.getRenderLights()[i].worldMatrix)) << what << ", light " << i;
    EXPECT_EQ(retained.light(lights[i].light)->owner.local, legacy.getRenderLights()[i].light) << what << ", light " << i;
  }
}

}  // namespace

TEST(RetainedImport, DrawsWhatTheLegacySceneDraws)
{
  expectParity(fixture(), "fixture");
}

TEST(RetainedImport, KeepsWhatItDrawsAndLetsTheRestGo)
{
  rs::RetainedScene      scene;
  const rs::ImportResult r = rs::importGltf(scene, fixture(), {.prop = 3});
  EXPECT_EQ(r.roots.size(), 1u);
  EXPECT_EQ(r.entities.size(), 7u);
  EXPECT_EQ(scene.entityCount(), 7u);
  EXPECT_EQ(scene.entity(r.entities[0])->owner, (rs::OwnerTag{3, 0}));
  // the two groups share one vertex set; the other mesh is its own
  ASSERT_EQ(r.meshes[0].size(), 1u);
  EXPECT_EQ(scene.mesh(r.meshes[0][0])->submeshes.size(), 2u);
  EXPECT_EQ(scene.meshCount(), 2u);
  EXPECT_EQ(scene.materialCount(), 2u);
  EXPECT_EQ(scene.textureCount(), 1u);  // the normal map, held by its material's table
  // the camera and the light are on their entities
  EXPECT_FALSE(scene.entity(r.entities[6])->camera.isNone());
  EXPECT_FLOAT_EQ(float(scene.camera(scene.entity(r.entities[6])->camera)->payload.perspective.yfov), 0.6f);
  // everything goes with the entities
  scene.destroyEntity(r.roots[0]);
  EXPECT_EQ(scene.meshCount() + scene.materialCount() + scene.textureCount(), 0u);
}

// An image's URI is found in the search paths and kept as its absolute path; one not found, or with
// no paths given, stays as the file wrote it.
TEST(RetainedImport, AnImageIsFoundWhereTheFileLives)
{
  namespace fs       = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "retained_import_images";
  fs::create_directories(dir / "maps");
  {
    std::ofstream(dir / "maps" / "normal map.png") << "png";
  }
  tinygltf::Model model = fixture();
  model.images[0].uri   = "maps/normal%20map.png";

  rs::RetainedScene        found;
  const rs::ImportResult   r = rs::importGltf(found, model, {.imageSearchPaths = {dir / "elsewhere", dir}});
  const rs::TextureRecord* t = found.texture((*found.material(r.materials[0])->textures)[0].texture);
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(fs::path(nvutils::pathFromUtf8(t->uri)), fs::absolute(dir / "maps" / "normal map.png"));

  rs::RetainedScene missing, unresolved;
  model.images[0].uri      = "gone.png";
  const rs::ImportResult m = rs::importGltf(missing, model, {.imageSearchPaths = {dir}});
  EXPECT_EQ(missing.texture((*missing.material(m.materials[0])->textures)[0].texture)->uri, "gone.png");
  const rs::ImportResult u = rs::importGltf(unresolved, fixture());
  EXPECT_EQ(unresolved.texture((*unresolved.material(u.materials[0])->textures)[0].texture)->uri, "normal.png");

  // an absolute URI, %-encoded as fv_compose writes it, needs no search path
  rs::RetainedScene absolute;
  model.images[0].uri = tinygltf::utils::encodePathAsUri(nvutils::utf8FromPath(fs::absolute(dir / "maps" / "normal map.png")));
  const rs::ImportResult a = rs::importGltf(absolute, model);
  EXPECT_EQ(fs::path(nvutils::pathFromUtf8(absolute.texture((*absolute.material(a.materials[0])->textures)[0].texture)->uri)),
            fs::absolute(dir / "maps" / "normal map.png"));
  fs::remove_all(dir);
}

// A small LM-63 profile: 100, 50 and 0 cd at 0, 90 and 180 degrees, so its samples are 1 at the
// nadir, 0.5 across and 0 straight up
const std::string kIesFile = "IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 3 1 1 2 0 0 0\n1 1 100\n0 90 180\n0\n100 50 0\n";

// EXT_lights_ies: the profiles read and sampled into the store; a node's IES light stands alone with
// its multiplier and colour, even over a KHR light on the same node; a profile that can't be read
// stays neutral, and a reference to none is left out, each with a notice
TEST(RetainedImport, IesLightsStandAloneWithTheirProfiles)
{
  tinygltf::Model m;
  m.buffers.emplace_back();
  m.buffers[0].data.assign(kIesFile.begin(), kIesFile.end());
  tinygltf::BufferView view;
  view.buffer     = 0;
  view.byteLength = kIesFile.size();
  m.bufferViews   = {view};
  tinygltf::Value::Object embedded, missing, root;
  embedded["bufferView"] = tinygltf::Value(0);
  embedded["mimeType"]   = tinygltf::Value(std::string("application/x-ies-lm-63"));
  embedded["name"]       = tinygltf::Value(std::string("downlight"));
  missing["uri"]         = tinygltf::Value(std::string("missing.ies"));
  root["lights"]         = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(embedded), tinygltf::Value(missing)});
  m.extensions[EXT_LIGHTS_IES_EXTENSION_NAME] = tinygltf::Value(root);
  m.extensionsUsed                            = {EXT_LIGHTS_IES_EXTENSION_NAME};
  tinygltf::Light lamp;
  lamp.type      = "spot";
  lamp.intensity = 30.0;
  m.lights       = {lamp};

  const auto ies = [](int profile, double multiplier) {
    tinygltf::Value::Object ref;
    ref["light"]      = tinygltf::Value(profile);
    ref["multiplier"] = tinygltf::Value(multiplier);
    ref["color"] = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(1.0), tinygltf::Value(0.5), tinygltf::Value(0.25)});
    return tinygltf::Value(ref);
  };
  tinygltf::Node both, alone, unreadable, dangling;
  both.light                                           = 0;
  both.extensions[EXT_LIGHTS_IES_EXTENSION_NAME]       = ies(0, 2.0);
  alone.extensions[EXT_LIGHTS_IES_EXTENSION_NAME]      = ies(0, 3.0);
  unreadable.extensions[EXT_LIGHTS_IES_EXTENSION_NAME] = ies(1, 1.0);
  dangling.extensions[EXT_LIGHTS_IES_EXTENSION_NAME]   = ies(7, 1.0);
  m.nodes                                              = {both, alone, unreadable, dangling};
  m.scenes.push_back({});
  m.scenes[0].nodes = {0, 1, 2, 3};
  m.defaultScene    = 0;

  rs::RetainedScene      scene;
  const rs::ImportResult r = rs::importGltf(scene, m);
  ASSERT_EQ(scene.iesProfiles().size(), 2u);
  const rs::IesProfileRecord& downlight = scene.iesProfiles()[0];
  EXPECT_EQ(downlight.name, "downlight");
  EXPECT_EQ(std::string(downlight.bytes.begin(), downlight.bytes.end()), kIesFile);
  ASSERT_EQ(downlight.samples.size(), size_t(nvvkgltf::kIesProfileSampleCount));
  EXPECT_FLOAT_EQ(downlight.samples[0], 1.f);
  EXPECT_FLOAT_EQ(downlight.samples[90], 0.5f);
  EXPECT_FLOAT_EQ(downlight.samples[180], 0.f);
  EXPECT_TRUE(scene.iesProfiles()[1].bytes.empty());
  EXPECT_EQ(scene.iesProfiles()[1].samples, std::vector<float>(size_t(nvvkgltf::kIesProfileSampleCount), 1.f));

  const auto lightOf = [&](int node) { return scene.light(scene.entity(r.entities[size_t(node)])->light); };
  ASSERT_NE(lightOf(0), nullptr);
  ASSERT_TRUE(lightOf(0)->ies);
  EXPECT_EQ(*lightOf(0)->ies, (rs::IesLight{.profile = 0, .multiplier = 2.f, .color = {1.f, 0.5f, 0.25f}}));
  EXPECT_EQ(lightOf(0)->payload.type, "spot");  // kept, though not drawn
  ASSERT_NE(lightOf(1), nullptr);
  EXPECT_EQ(lightOf(1)->ies->multiplier, 3.f);
  EXPECT_EQ(lightOf(2)->ies->profile, 1u);
  EXPECT_EQ(lightOf(3), nullptr);
  const auto noticed = [&](const char* text) {
    return std::any_of(r.notices.begin(), r.notices.end(),
                       [&](const std::string& n) { return n.find(text) != std::string::npos; });
  };
  EXPECT_TRUE(noticed("missing.ies"));
  EXPECT_TRUE(noticed("naming no profile"));
}

// The generated glTF 2.1 fixtures and, when present, the Khronos sample assets: every one the legacy
// scene loads on its own (no external assets) draws the same here.
TEST(RetainedImport, DrawsTheFixturesAsTheLegacySceneDoes)
{
  namespace fs = std::filesystem;
  int tested   = 0;
  for(const fs::path dir : {fs::path(GLTF_2_1_FIXTURES_DIR), fs::path(GLTF_SAMPLE_ASSETS_PATH)})
  {
    std::error_code ec;
    if(dir.empty() || !fs::exists(dir, ec))
      continue;
    for(const auto& entry : fs::recursive_directory_iterator(dir, ec))
    {
      const fs::path file = entry.path();
      if(file.extension() != ".gltf" && file.extension() != ".glb")
        continue;
      tinygltf::TinyGLTF loader;
      tinygltf::Model    model;
      std::string        err, warn;
      const bool ok = file.extension() == ".glb" ? loader.LoadBinaryFromFile(&model, &err, &warn, file.string()) :
                                                   loader.LoadASCIIFromFile(&model, &err, &warn, file.string());
      if(!ok || model.nodes.empty() || model.scenes.empty() || model.extensionsUsed.end() != std::find_if(model.extensionsUsed.begin(), model.extensionsUsed.end(), [](const std::string& e) {
           return e == "KHR_materials_variants" || e == "KHR_interactivity" || e.rfind("EXT_external", 0) == 0;
         }))
        continue;
      if(std::any_of(model.meshes.begin(), model.meshes.end(), [](const tinygltf::Mesh& mesh) {
           return std::any_of(mesh.primitives.begin(), mesh.primitives.end(), [](const tinygltf::Primitive& p) { return p.material < 0; });
         }))
        continue;  // the one difference on purpose: no material is the glTF default here
      expectParity(model, file.filename().string());
      tested++;
    }
  }
  if(tested == 0)
    GTEST_SKIP() << "no fixtures generated and no sample assets";
  std::printf("compared %d files\n", tested);
}
