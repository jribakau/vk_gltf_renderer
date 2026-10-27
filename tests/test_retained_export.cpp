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

// The retained scene out as glTF (oracle 9 of daz-viewer docs/retained-scene-plan.md): an export
// imported again draws the same scene, and the export leaves the scene as it was.

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <sstream>

#include "scene_export_gltf.hpp"
#include "scene_import_gltf.hpp"
#include "tinygltf_utils.hpp"

namespace {

bool nearly(const glm::mat4& x, const glm::mat4& y)
{
  for(int c = 0; c < 4; c++)
    for(int r = 0; r < 4; r++)
      if(std::abs(x[c][r] - y[c][r]) > 1e-4f)
        return false;
  return true;
}

// The image a material's base colour or normal map samples, by its file
std::string mapOf(rs::RetainedScene& s, rs::MaterialId id, bool normal)
{
  const rs::MaterialRecord* m = s.material(id);
  const int i = normal ? m->payload.normalTexture.index : m->payload.pbrMetallicRoughness.baseColorTexture.index;
  if(i < 0 || !m->textures || i >= int(m->textures->size()))
    return "";
  const rs::TextureRecord* t = s.texture((*m->textures)[size_t(i)].texture);
  // the file, or held bytes, or decoded pixels (an encoded image re-encoded by a writer differs in its
  // bytes, so only that there are some, and the pixels' size)
  return t ? t->uri + "|" + (t->bytes.empty() ? "" : "bytes") + "|" + std::to_string(t->pixels.size()) : "?";
}

void expectSameScene(rs::RetainedScene& a, rs::RetainedScene& b, const std::string& what)
{
  const auto ia = a.drawItems(), ib = b.drawItems();
  ASSERT_EQ(ia.size(), ib.size()) << what;
  for(size_t i = 0; i < ia.size(); i++)
  {
    SCOPED_TRACE(what + ", item " + std::to_string(i));
    EXPECT_TRUE(nearly(ia[i].world, ib[i].world));
    EXPECT_EQ(ia[i].shown, ib[i].shown);
    EXPECT_EQ(a.entity(ia[i].entity)->name, b.entity(ib[i].entity)->name);
    const rs::MeshRecord &ma = *a.mesh(ia[i].mesh), &mb = *b.mesh(ib[i].mesh);
    ASSERT_EQ(ma.positions.size(), mb.positions.size());
    EXPECT_EQ(std::memcmp(ma.positions.data(), mb.positions.data(), ma.positions.size() * 12), 0);
    EXPECT_EQ(ma.normals.size(), mb.normals.size());
    EXPECT_EQ(ma.tangents.size(), mb.tangents.size());
    EXPECT_EQ(ma.joints.size(), mb.joints.size());
    EXPECT_EQ(ma.targets.size(), mb.targets.size());
    const rs::Submesh &sa = ma.submeshes[ia[i].submesh], &sb = mb.submeshes[ib[i].submesh];
    ASSERT_EQ(sa.indexCount, sb.indexCount);
    EXPECT_TRUE(std::equal(ma.indices.begin() + sa.firstIndex, ma.indices.begin() + sa.firstIndex + sa.indexCount, mb.indices.begin() + sb.firstIndex));
    const rs::MaterialRecord &pa = *a.material(ia[i].material), &pb = *b.material(ib[i].material);
    EXPECT_EQ(pa.payload.pbrMetallicRoughness.baseColorFactor, pb.payload.pbrMetallicRoughness.baseColorFactor);
    EXPECT_EQ(pa.payload.alphaMode, pb.payload.alphaMode);
    EXPECT_EQ(pa.payload.extensions.size(), pb.payload.extensions.size());
    EXPECT_EQ(mapOf(a, ia[i].material, false), mapOf(b, ib[i].material, false));
    EXPECT_EQ(mapOf(a, ia[i].material, true), mapOf(b, ib[i].material, true));
  }
  const auto la = a.lightItems(), lb = b.lightItems();
  ASSERT_EQ(la.size(), lb.size()) << what;
  for(size_t i = 0; i < la.size(); i++)
  {
    EXPECT_TRUE(nearly(la[i].world, lb[i].world)) << what;
    EXPECT_EQ(la[i].shown, lb[i].shown) << what;
    EXPECT_EQ(a.light(la[i].light)->payload.intensity, b.light(lb[i].light)->payload.intensity) << what;
  }
}

tinygltf::Model load(const std::filesystem::path& file, bool& ok)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model    model;
  std::string        err, warn;
  ok = file.extension() == ".glb" ? loader.LoadBinaryFromFile(&model, &err, &warn, file.string()) :
                                    loader.LoadASCIIFromFile(&model, &err, &warn, file.string());
  return model;
}

}  // namespace

TEST(RetainedExport, AnExportImportedAgainIsTheSameScene)
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
      bool                  ok    = false;
      const tinygltf::Model model = load(file, ok);
      if(!ok || model.nodes.empty() || model.scenes.empty())
        continue;
      rs::RetainedScene a;
      rs::importGltf(a, model, {.prop = 1});
      const size_t entities = a.entityCount(), meshes = a.meshCount();
      tinygltf::Model exported = rs::exportGltf(a);
      // the export changed nothing
      EXPECT_EQ(a.entityCount(), entities);
      EXPECT_EQ(a.meshCount(), meshes);
      // written as a GLB and read back, so it is a file other tools can open, not only a Model
      std::ostringstream glb;
      tinygltf::TinyGLTF io;
      ASSERT_TRUE(io.WriteGltfSceneToStream(&exported, glb, false, true)) << file;
      const std::string bytes = glb.str();
      tinygltf::Model   reread;
      std::string       err, warn;
      ASSERT_TRUE(io.LoadBinaryFromMemory(&reread, &err, &warn, reinterpret_cast<const unsigned char*>(bytes.data()), unsigned(bytes.size())))
          << file << ": " << err;
      rs::RetainedScene b;
      rs::importGltf(b, reread, {.prop = 1});
      expectSameScene(a, b, file.filename().string());
      tested++;
    }
  }
  if(tested == 0)
    GTEST_SKIP() << "no fixtures generated and no sample assets";
  std::printf("round-tripped %d files\n", tested);
}

TEST(RetainedExport, ASubtreeAndAHiddenLight)
{
  rs::RetainedScene  scene;
  const rs::EntityId keep = scene.createEntity("keep", {});
  const rs::EntityId skip = scene.createEntity("skip", {});
  const rs::EntityId lamp = scene.createEntity("lamp", {}, keep);
  tinygltf::Light    light;
  light.type      = "point";
  light.intensity = 40;
  const rs::LightId l = scene.addLight({.payload = light});
  scene.setEntityLight(lamp, l);
  scene.release(l);
  scene.setLightShown(l, false);  // a hidden lamp keeps its real intensity
  scene.createEntity("under skip", {}, skip);

  const tinygltf::Model out = rs::exportGltf(scene, {.roots = {keep}});
  ASSERT_EQ(out.nodes.size(), 2u);
  EXPECT_EQ(out.nodes[0].name, "keep");
  EXPECT_EQ(out.nodes[0].children, (std::vector<int>{1}));
  ASSERT_EQ(out.lights.size(), 1u);
  EXPECT_EQ(out.lights[0].intensity, 40);
  EXPECT_FALSE(tinygltf::utils::getNodeVisibility(out.nodes[1]).visible);
  EXPECT_EQ(out.scenes[0].nodes, (std::vector<int>{0}));
}
