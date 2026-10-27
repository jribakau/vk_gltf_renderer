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
// imported again draws the same scene and plays the same clips, and the export leaves the scene as
// it was.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
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

// Clips in their owners' order: the same names, each channel on the entity at the same place in
// document order, along the same path, with the same keys
std::vector<rs::ClipId> clipsInOrder(const rs::RetainedScene& s)
{
  std::vector<rs::ClipId> clips = s.clips();
  std::stable_sort(clips.begin(), clips.end(), [&](rs::ClipId x, rs::ClipId y) { return s.clip(x)->owner < s.clip(y)->owner; });
  return clips;
}

void expectSameClips(const rs::RetainedScene& a, const rs::RetainedScene& b, const std::string& what)
{
  const auto place = [](const rs::RetainedScene& s) {
    std::map<rs::EntityId, size_t> at;
    for(rs::EntityId id : s.entitiesInOrder())
      at.emplace(id, at.size());
    return at;
  };
  const std::map<rs::EntityId, size_t> pa = place(a), pb = place(b);
  const std::vector<rs::ClipId>        ca = clipsInOrder(a), cb = clipsInOrder(b);
  ASSERT_EQ(ca.size(), cb.size()) << what;
  for(size_t i = 0; i < ca.size(); i++)
  {
    SCOPED_TRACE(what + ", clip " + std::to_string(i));
    const rs::ClipRecord &x = *a.clip(ca[i]), &y = *b.clip(cb[i]);
    EXPECT_EQ(x.name, y.name);
    ASSERT_EQ(x.channels.size(), y.channels.size());
    for(size_t c = 0; c < x.channels.size(); c++)
    {
      const rs::ClipChannel &cx = x.channels[c], &cy = y.channels[c];
      ASSERT_TRUE(pa.count(cx.target) && pb.count(cy.target));
      EXPECT_EQ(pa.at(cx.target), pb.at(cy.target));
      EXPECT_EQ(cx.path, cy.path);
      EXPECT_TRUE(x.samplers[cx.sampler] == y.samplers[cy.sampler]) << "channel " << c;
    }
  }
}

// written as a GLB and read back, so it is a file other tools can open, not only a Model
tinygltf::Model throughGlb(tinygltf::Model exported)
{
  std::ostringstream glb;
  tinygltf::TinyGLTF io;
  EXPECT_TRUE(io.WriteGltfSceneToStream(&exported, glb, false, true));
  const std::string bytes = glb.str();
  tinygltf::Model   reread;
  std::string       err, warn;
  EXPECT_TRUE(io.LoadBinaryFromMemory(&reread, &err, &warn, reinterpret_cast<const unsigned char*>(bytes.data()), unsigned(bytes.size())))
      << err;
  return reread;
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

// A small LM-63 profile: 100, 50 and 0 cd at 0, 90 and 180 degrees, so its samples are 1 at the
// nadir, 0.5 across and 0 straight up
const std::string kIesFile = "IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 3 1 1 2 0 0 0\n1 1 100\n0 90 180\n0\n100 50 0\n";

// An EXT_lights_ies light goes out as its node's reference to its profile, the file in the buffer,
// and comes back the same; a profile no exported light uses is not written
TEST(RetainedExport, IesLightsGoOutWithTheirProfiles)
{
  rs::RetainedScene a;
  const uint32_t    unused = a.addIesProfile({.name = "unused", .samples = std::vector<float>(181, 1.f)});
  const uint32_t    profile =
      a.addIesProfile({.name = "downlight", .bytes = std::vector<uint8_t>(kIesFile.begin(), kIesFile.end())});
  rs::LightRecord light;
  light.payload.type    = "point";
  light.ies             = rs::IesLight{.profile = profile, .multiplier = 4.f, .color = {0.5f, 1.f, 1.f}};
  const rs::LightId  id = a.addLight(light);
  const rs::EntityId e  = a.createEntity("lamp", {});
  a.setEntityLight(e, id);
  a.release(id);
  (void)unused;

  const tinygltf::Model out = rs::exportGltf(a);
  EXPECT_TRUE(out.lights.empty());  // it stands alone
  const tinygltf::Model  reread = throughGlb(out);
  rs::RetainedScene      b;
  const rs::ImportResult r = rs::importGltf(b, reread);
  ASSERT_EQ(b.iesProfiles().size(), 1u);
  EXPECT_EQ(b.iesProfiles()[0].name, "downlight");
  EXPECT_EQ(std::string(b.iesProfiles()[0].bytes.begin(), b.iesProfiles()[0].bytes.end()), kIesFile);
  EXPECT_FLOAT_EQ(b.iesProfiles()[0].samples[90], 0.5f);
  const rs::LightRecord* back = b.light(b.entity(r.entities[0])->light);
  ASSERT_TRUE(back && back->ies);
  EXPECT_EQ(*back->ies, (rs::IesLight{.profile = 0, .multiplier = 4.f, .color = {0.5f, 1.f, 1.f}}));
}

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
      const size_t entities = a.entityCount(), meshes = a.meshCount(), clips = a.clipCount();
      const tinygltf::Model reread = throughGlb(rs::exportGltf(a));
      // the export changed nothing
      EXPECT_EQ(a.entityCount(), entities);
      EXPECT_EQ(a.meshCount(), meshes);
      EXPECT_EQ(a.clipCount(), clips);
      rs::RetainedScene b;
      rs::importGltf(b, reread, {.prop = 1});
      expectSameScene(a, b, file.filename().string());
      expectSameClips(a, b, file.filename().string());
      tested++;
    }
  }
  if(tested == 0)
    GTEST_SKIP() << "no fixtures generated and no sample assets";
  std::printf("round-tripped %d files\n", tested);
}

// Clips of every path and interpolation go out as animations and come back the same; exporting a
// subtree keeps the channels on what it writes, and a clip with none left is not written.
TEST(RetainedExport, ClipsGoOutAndComeBack)
{
  rs::RetainedScene  a;
  const rs::EntityId root  = a.createEntity("root", {});
  const rs::EntityId arm   = a.createEntity("arm", {}, root);
  const rs::EntityId other = a.createEntity("other", {});
  rs::MeshRecord     tri;
  tri.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  tri.indices   = {0, 1, 2};
  tri.submeshes = {{.indexCount = 3}};
  tri.targets   = {{.positions = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}}}};
  const rs::MeshId     mesh = a.addMesh(tri);
  const rs::MaterialId mat  = a.addMaterial({});
  a.setMeshInstance(arm, rs::MeshInstance{.parts = {{mesh, {mat}}}});
  a.release(mesh);
  a.release(mat);

  rs::ClipRecord wave{.name = "wave", .owner = {1, 0}};
  wave.samplers = {{rs::ClipInterpolation::eLinear, {0, 1}, {0, 0, 0, 1, 2, 3}},
                   {rs::ClipInterpolation::eCubicSpline, {0, 2}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}},
                   {rs::ClipInterpolation::eStep, {0, 1, 2}, {0, 1, 0.5f}},
                   {rs::ClipInterpolation::eLinear, {0, 4}, {1, 1, 1, 2, 2, 2}}};
  wave.channels = {{arm, rs::ClipPath::eTranslation, 0},
                   {root, rs::ClipPath::eRotation, 1},
                   {arm, rs::ClipPath::eWeights, 2},
                   {other, rs::ClipPath::eScale, 3},
                   {arm, rs::ClipPath::eScale, 3}};
  a.addClip(wave);
  a.addClip({.name = "elsewhere", .owner = {1, 1}, .samplers = {wave.samplers[0]}, .channels = {{other, rs::ClipPath::eTranslation, 0}}});

  rs::RetainedScene b;
  rs::importGltf(b, throughGlb(rs::exportGltf(a)), {.prop = 1});
  expectSameClips(a, b, "the whole scene");
  EXPECT_EQ(a.clipCount(), 2u);

  const tinygltf::Model part = rs::exportGltf(a, {.roots = {root}});
  ASSERT_EQ(part.animations.size(), 1u);
  EXPECT_EQ(part.animations[0].name, "wave");
  EXPECT_EQ(part.animations[0].channels.size(), 4u);  // not the one on "other"
  EXPECT_EQ(part.animations[0].samplers.size(), 4u);
  rs::RetainedScene      c;
  const rs::ImportResult r = rs::importGltf(c, throughGlb(part), {.prop = 1});
  ASSERT_EQ(r.clips.size(), 1u);
  const rs::ClipRecord* back = c.clip(r.clips[0]);
  ASSERT_NE(back, nullptr);
  EXPECT_EQ(back->channels.size(), 4u);
  EXPECT_TRUE(back->samplers[back->channels[1].sampler] == wave.samplers[1]);
  EXPECT_EQ(back->range(), (std::array<float, 2>{0, 4}));
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
