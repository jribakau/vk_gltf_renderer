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

// The Figure Viewer's scene of props composed into one glTF (src/fv_compose.hpp): prop roots stay
// at the props' indices, each prop's arrays start where `base` says, ids are prefixed per prop, and
// cameras, lights, visibility and parents come out as the props have them.

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

#include <gtest/gtest.h>
#include <meshoptimizer/src/meshoptimizer.h>

#include "fv_compose.hpp"
#include "tinygltf_utils.hpp"

namespace {

using fv::composer::Part;
using fv::composer::Placed;

// A figure-like model: a root with one mesh node under it, a material and a mesh, fvIds as gltf_build writes them.
tinygltf::Model smallModel(const std::string& name, int meshes = 1)
{
  tinygltf::Model m;
  m.scenes.emplace_back();
  m.scenes[0].nodes = {0};
  tinygltf::Node root;
  root.name = name;
  tinygltf::Value::Object rootExtras;
  rootExtras["fvId"] = tinygltf::Value(std::string("n:0"));
  root.extras        = tinygltf::Value(rootExtras);
  m.nodes.push_back(root);
  for(int i = 0; i < meshes; i++)
  {
    tinygltf::Material mat;
    mat.name = name + " surface " + std::to_string(i);
    tinygltf::Value::Object o;
    o["fvId"]  = tinygltf::Value("m:" + std::to_string(i));
    mat.extras = tinygltf::Value(o);
    m.materials.push_back(mat);
    tinygltf::Mesh mesh;
    tinygltf::Primitive prim;
    prim.material = i;
    mesh.primitives.push_back(prim);
    m.meshes.push_back(mesh);
    tinygltf::Node n;
    n.name = name + " mesh " + std::to_string(i);
    n.mesh = i;
    tinygltf::Value::Object ne;
    ne["fvId"] = tinygltf::Value("n:" + std::to_string(i + 1));
    n.extras   = tinygltf::Value(ne);
    m.nodes[0].children.push_back(int(m.nodes.size()));
    m.nodes.push_back(n);
  }
  return m;
}

std::string fvIdOf(const tinygltf::Value& extras)
{
  return extras.IsObject() && extras.Has("fvId") ? extras.Get("fvId").Get<std::string>() : "";
}

}  // namespace

TEST(FvCompose, PropRootsComeFirstAndEachPropsArraysFollow)
{
  const tinygltf::Model a = smallModel("A", 2), b = smallModel("B", 1);
  std::vector<Part>     parts(4);
  parts[0].id    = "p:1";
  parts[0].name  = "Figure A";
  parts[0].model = &a;
  parts[1].id    = "p:2";
  parts[1].name  = "Camera";
  parts[1].camera = fv::composer::perspectiveCamera(50);
  parts[2].id    = "p:3";
  parts[2].name  = "Figure B";
  parts[2].model = &b;
  parts[2].transform.translation = {1.f, 0.f, -2.f};
  parts[3].id    = "p:4";
  parts[3].name  = "Lamp";
  parts[3].light = tinygltf::Light{};
  parts[3].light->type = "point";

  std::vector<Placed>   placed;
  const tinygltf::Model out = fv::composer::compose(parts, placed);
  ASSERT_EQ(placed.size(), 4u);
  for(int i = 0; i < 4; i++)
  {
    EXPECT_EQ(placed[size_t(i)].root, i);
    EXPECT_EQ(fvIdOf(out.nodes[size_t(i)].extras), parts[size_t(i)].id);
  }
  // A's 3 nodes after the 4 roots, then B's 2
  EXPECT_EQ(placed[0].base.nodes, 4);
  EXPECT_EQ(placed[0].firstNode, 4);
  EXPECT_EQ(placed[0].lastNode, 7);
  EXPECT_EQ(placed[2].base.nodes, 7);
  EXPECT_EQ(placed[2].base.meshes, 2);
  EXPECT_EQ(placed[2].base.materials, 2);
  EXPECT_EQ(out.nodes.size(), 9u);
  EXPECT_EQ(out.meshes.size(), 3u);
  // B's mesh node points at B's mesh, which points at B's material
  const tinygltf::Node& bMesh = out.nodes[size_t(placed[2].base.nodes + 1)];
  EXPECT_EQ(bMesh.mesh, placed[2].base.meshes);
  EXPECT_EQ(out.meshes[size_t(bMesh.mesh)].primitives[0].material, placed[2].base.materials);
  // each model hangs from its prop root
  ASSERT_EQ(out.nodes[0].children.size(), 1u);
  EXPECT_EQ(out.nodes[0].children[0], placed[0].firstNode);
  EXPECT_EQ(out.nodes[2].children[0], placed[2].firstNode);
  // the prop's place is its root's
  EXPECT_FLOAT_EQ(float(out.nodes[2].translation[0]), 1.f);
  EXPECT_FLOAT_EQ(float(out.nodes[2].translation[2]), -2.f);
  // the camera and the light are on their roots
  EXPECT_EQ(out.nodes[1].camera, 0);
  EXPECT_EQ(placed[1].camera, 0);
  EXPECT_EQ(out.nodes[3].light, 0);
  EXPECT_EQ(placed[3].light, 0);
  EXPECT_NE(std::find(out.extensionsUsed.begin(), out.extensionsUsed.end(), "KHR_lights_punctual"), out.extensionsUsed.end());
  EXPECT_EQ(out.scenes[0].nodes, (std::vector<int>{0, 1, 2, 3}));
}

TEST(FvCompose, IdsArePrefixedSoTwoFiguresNeverCollide)
{
  const tinygltf::Model a = smallModel("A"), b = smallModel("B");
  std::vector<Part>     parts(2);
  parts[0].id    = "p:1";
  parts[0].model = &a;
  parts[1].id    = "p:7";
  parts[1].model = &b;
  std::vector<Placed>   placed;
  const tinygltf::Model out = fv::composer::compose(parts, placed);
  std::set<std::string> ids;
  for(const tinygltf::Node& n : out.nodes)
    EXPECT_TRUE(ids.insert(fvIdOf(n.extras)).second) << fvIdOf(n.extras);
  for(const tinygltf::Material& m : out.materials)
    EXPECT_TRUE(ids.insert(fvIdOf(m.extras)).second) << fvIdOf(m.extras);
  EXPECT_TRUE(ids.count("p:7/n:1"));
  EXPECT_TRUE(ids.count("p:1/m:0"));
  EXPECT_TRUE(ids.count("p:7/m:0"));
}

TEST(FvCompose, HiddenPropsAndParents)
{
  const tinygltf::Model a = smallModel("A");
  std::vector<Part>     parts(3);
  parts[0].id      = "p:1";
  parts[0].model   = &a;
  parts[1].id      = "p:2";
  parts[1].parent  = 0;  // rides on p:1
  parts[1].visible = false;
  parts[2].id      = "p:3";
  parts[2].parent  = 2;  // itself: stays a root
  std::vector<Placed>   placed;
  const tinygltf::Model out = fv::composer::compose(parts, placed);
  EXPECT_EQ(out.scenes[0].nodes, (std::vector<int>{0, 2}));
  EXPECT_NE(std::find(out.nodes[0].children.begin(), out.nodes[0].children.end(), 1), out.nodes[0].children.end());
  EXPECT_TRUE(out.nodes[1].extensions.count(KHR_NODE_VISIBILITY_EXTENSION_NAME));
  EXPECT_FALSE(out.nodes[0].extensions.count(KHR_NODE_VISIBILITY_EXTENSION_NAME));
  // a cycle is broken at the part that would close it
  parts[0].parent = 1;
  const tinygltf::Model loop = fv::composer::compose(parts, placed);
  EXPECT_FALSE(loop.scenes[0].nodes.empty());
}

TEST(FvCompose, PartOfNode)
{
  const tinygltf::Model a = smallModel("A", 2), b = smallModel("B", 1);
  std::vector<Part>     parts(2);
  parts[0].id    = "p:1";
  parts[0].model = &a;
  parts[1].id    = "p:2";
  parts[1].model = &b;
  std::vector<Placed>   placed;
  const tinygltf::Model out = fv::composer::compose(parts, placed);
  EXPECT_EQ(fv::composer::partOfNode(placed, 0), 0);
  EXPECT_EQ(fv::composer::partOfNode(placed, 1), 1);
  EXPECT_EQ(fv::composer::partOfNode(placed, placed[0].firstNode + 2), 0);
  EXPECT_EQ(fv::composer::partOfNode(placed, placed[1].firstNode), 1);
  EXPECT_EQ(fv::composer::partOfNode(placed, int(out.nodes.size())), -1);
  EXPECT_EQ(fv::composer::partOfNode(placed, -1), -1);
}

// A removed prop is left out of the next composition; the props after it move up, and their
// own indices, which the host keeps per prop, still hold once `base` is added.
TEST(FvCompose, RecomposingWithoutAPropKeepsTheOthersLocalIndices)
{
  const tinygltf::Model a = smallModel("A", 2), b = smallModel("B", 3), c = smallModel("C", 1);
  std::vector<Part>     parts(3);
  parts[0].id    = "p:1";
  parts[0].model = &a;
  parts[1].id    = "p:2";
  parts[1].model = &b;
  parts[2].id    = "p:3";
  parts[2].model = &c;
  std::vector<Placed> placed;
  fv::composer::compose(parts, placed);
  parts.erase(parts.begin() + 1);
  const tinygltf::Model out = fv::composer::compose(parts, placed);
  // C's mesh 0 is its material 0, local indices plus base
  const tinygltf::Mesh& mesh = out.meshes[size_t(placed[1].base.meshes)];
  EXPECT_EQ(mesh.primitives[0].material, placed[1].base.materials);
  EXPECT_EQ(out.materials[size_t(placed[1].base.materials)].name, "C surface 0");
  EXPECT_EQ(fvIdOf(out.nodes[1].extras), "p:3");
}

TEST(FvCompose, ReadModelFileMakesImagesAbsolute)
{
  const auto dir = std::filesystem::temp_directory_path() / "fv_compose_test";
  std::filesystem::create_directories(dir / "tex");
  const auto file = dir / "box.gltf";
  std::ofstream(dir / "tex" / "a b.png") << "not decoded here";
  {
    std::ofstream out(file);
    out << R"({"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{"name":"box"}],
              "images":[{"uri":"tex/a%20b.png"}],"textures":[{"source":0}]})";
  }
  tinygltf::Model model;
  std::string     error;
  ASSERT_TRUE(fv::composer::readModelFile(file, model, error)) << error;
  ASSERT_EQ(model.images.size(), 1u);
  const std::string decoded = tinygltf::utils::decodePathFromUri(model.images[0].uri);
  const std::filesystem::path path(std::u8string(decoded.begin(), decoded.end()));
  EXPECT_TRUE(path.is_absolute()) << decoded;
  EXPECT_EQ(path.filename().string(), "a b.png");

  tinygltf::Model none;
  EXPECT_FALSE(fv::composer::readModelFile(dir / "missing.gltf", none, error));
  std::filesystem::remove_all(dir);
}

// A .glb compressed by gltfpack: its views live in a fallback buffer with no bytes of its own, and only
// the compressed one is in the file. They read back as the mesh that was encoded.
TEST(FvCompose, ReadModelFileUnpacksMeshoptCompression)
{
  const std::array<float, 9>    positions = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const std::array<uint32_t, 3> indices   = {0, 1, 2};

  std::vector<unsigned char> vertexBytes(meshopt_encodeVertexBufferBound(3, 12));
  vertexBytes.resize(meshopt_encodeVertexBuffer(vertexBytes.data(), vertexBytes.size(), positions.data(), 3, 12));
  std::vector<unsigned char> indexBytes(meshopt_encodeIndexBufferBound(3, 3));
  indexBytes.resize(meshopt_encodeIndexBuffer(indexBytes.data(), indexBytes.size(), indices.data(), 3));

  std::vector<unsigned char> bin = vertexBytes;
  while(bin.size() % 4)
    bin.push_back(0);
  const size_t indexOffset = bin.size();
  bin.insert(bin.end(), indexBytes.begin(), indexBytes.end());
  while(bin.size() % 4)
    bin.push_back(0);

  std::string json = R"({"asset":{"version":"2.0"},"extensionsUsed":["EXT_meshopt_compression"],
    "extensionsRequired":["EXT_meshopt_compression"],"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
    "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
    "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
                 {"bufferView":1,"componentType":5125,"count":3,"type":"SCALAR"}],
    "bufferViews":[{"buffer":1,"byteOffset":0,"byteLength":36,"byteStride":12,"extensions":{"EXT_meshopt_compression":
                     {"buffer":0,"byteOffset":0,"byteLength":)"
                     + std::to_string(vertexBytes.size()) + R"(,"byteStride":12,"count":3,"mode":"ATTRIBUTES"}}},
                   {"buffer":1,"byteOffset":36,"byteLength":12,"extensions":{"EXT_meshopt_compression":
                     {"buffer":0,"byteOffset":)"
                     + std::to_string(indexOffset) + R"(,"byteLength":)" + std::to_string(indexBytes.size())
                     + R"(,"byteStride":4,"count":3,"mode":"TRIANGLES"}}}],
    "buffers":[{"byteLength":)"
                     + std::to_string(bin.size()) + R"(},
               {"byteLength":48,"extensions":{"EXT_meshopt_compression":{"fallback":true}}}]})";
  while(json.size() % 4)
    json.push_back(' ');

  const auto dir = std::filesystem::temp_directory_path() / "fv_compose_meshopt_test";
  std::filesystem::create_directories(dir);
  const auto file = dir / "tri.glb";
  {
    const auto    word = [](std::ofstream& out, uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    std::ofstream out(file, std::ios::binary);
    out.write("glTF", 4);
    word(out, 2);
    word(out, uint32_t(12 + 8 + json.size() + 8 + bin.size()));
    word(out, uint32_t(json.size()));
    out.write("JSON", 4);
    out.write(json.data(), std::streamsize(json.size()));
    word(out, uint32_t(bin.size()));
    out.write("BIN\0", 4);
    out.write(reinterpret_cast<const char*>(bin.data()), std::streamsize(bin.size()));
  }

  tinygltf::Model model;
  std::string     error;
  ASSERT_TRUE(fv::composer::readModelFile(file, model, error)) << error;
  std::filesystem::remove_all(dir);

  ASSERT_EQ(model.bufferViews.size(), 2u);
  for(const tinygltf::BufferView& view : model.bufferViews)
    EXPECT_TRUE(view.extensions.empty());
  const auto bytesOf = [&](int view) {
    const tinygltf::BufferView& v = model.bufferViews[size_t(view)];
    return model.buffers[size_t(v.buffer)].data.data() + v.byteOffset;
  };
  EXPECT_EQ(std::memcmp(bytesOf(0), positions.data(), sizeof(positions)), 0);
  EXPECT_EQ(std::memcmp(bytesOf(1), indices.data(), sizeof(indices)), 0);
  // the compressed buffer has no view left, so it goes
  EXPECT_EQ(model.buffers.size(), 1u);
}

// A model prop's own lights, after a lamp prop: the composed model holds the lamp's light and then the
// model's, and the model's node points at its own light, not the lamp's.
TEST(FvCompose, AModelsLightsFollowALampsLight)
{
  const auto dir  = std::filesystem::temp_directory_path() / "fv_compose_lights";
  const auto file = dir / "lit.gltf";
  std::filesystem::create_directories(dir);
  {
    std::ofstream out(file);
    out << R"({"asset":{"version":"2.0"},"extensionsUsed":["KHR_lights_punctual"],
              "extensions":{"KHR_lights_punctual":{"lights":[{"type":"spot","intensity":7,"spot":{}}]}},
              "scenes":[{"nodes":[0]}],"nodes":[{"name":"lit","extensions":{"KHR_lights_punctual":{"light":0}}}]})";
  }
  tinygltf::Model model;
  std::string     error;
  ASSERT_TRUE(fv::composer::readModelFile(file, model, error)) << error;
  ASSERT_EQ(model.lights.size(), 1u);

  std::vector<Part> parts(2);
  parts[0].id          = "p:1";
  parts[0].light       = tinygltf::Light{};
  parts[0].light->type = "point";
  parts[1].id          = "p:2";
  parts[1].model       = &model;
  std::vector<Placed>   placed;
  const tinygltf::Model out = fv::composer::compose(parts, placed);
  ASSERT_EQ(out.lights.size(), 2u);
  EXPECT_EQ(out.lights[0].type, "point");
  EXPECT_EQ(out.lights[1].type, "spot");
  EXPECT_EQ(placed[1].base.lights, 1);
  const tinygltf::Node& lit = out.nodes[size_t(placed[1].firstNode)];
  EXPECT_EQ(lit.light, 1);
  // a save writes the extension's array, so it lists both, in the same order
  ASSERT_TRUE(out.extensions.count("KHR_lights_punctual"));
  const tinygltf::Value& lights = out.extensions.at("KHR_lights_punctual").Get("lights");
  ASSERT_EQ(lights.ArrayLen(), 2u);
  EXPECT_EQ(lights.Get(1).Get("type").Get<std::string>(), "spot");
  std::filesystem::remove_all(dir);
}

// A scene export strips what SceneMerger does not rebase (B58, D23): the pointer channels, with the
// samplers only they read and an animation left with none; interactivity; material variants.
TEST(FvCompose, AnExportStripsWhatTheComposerCannotRebase)
{
  tinygltf::Model m = smallModel("lamp");
  m.animations.resize(2);
  tinygltf::Animation& mixed = m.animations[0];
  mixed.samplers.resize(3);
  mixed.samplers[2].input = 7;  // read by the node channel: it survives, renumbered
  tinygltf::AnimationChannel node;
  node.sampler     = 2;
  node.target_node = 1;
  node.target_path = "rotation";
  tinygltf::AnimationChannel pointer;
  pointer.sampler     = 0;
  pointer.target_path = "pointer";
  tinygltf::Value::Object ptr;
  ptr["pointer"]                                = tinygltf::Value(std::string("/materials/0/emissiveFactor"));
  pointer.target_extensions["KHR_animation_pointer"] = tinygltf::Value(ptr);
  mixed.channels = {pointer, node};
  m.animations[1].samplers.resize(1);
  m.animations[1].channels = {pointer};

  m.extensions["KHR_interactivity"]                        = tinygltf::Value(tinygltf::Value::Object{});
  m.extensions["KHR_materials_variants"]                   = tinygltf::Value(tinygltf::Value::Object{});
  m.meshes[0].primitives[0].extensions["KHR_materials_variants"] = tinygltf::Value(tinygltf::Value::Object{});
  m.extensionsUsed     = {"KHR_lights_punctual", "KHR_animation_pointer", "KHR_interactivity", "KHR_materials_variants"};
  m.extensionsRequired = {"KHR_interactivity"};

  const std::vector<std::string> notes = fv::composer::stripForFile(m);
  EXPECT_EQ(notes.size(), 3u);
  ASSERT_EQ(m.animations.size(), 1u);
  ASSERT_EQ(m.animations[0].channels.size(), 1u);
  EXPECT_EQ(m.animations[0].channels[0].target_path, "rotation");
  ASSERT_EQ(m.animations[0].samplers.size(), 1u);
  EXPECT_EQ(m.animations[0].channels[0].sampler, 0);
  EXPECT_EQ(m.animations[0].samplers[0].input, 7);
  EXPECT_TRUE(m.extensions.empty());
  EXPECT_TRUE(m.meshes[0].primitives[0].extensions.empty());
  EXPECT_EQ(m.extensionsUsed, std::vector<std::string>{"KHR_lights_punctual"});
  EXPECT_TRUE(m.extensionsRequired.empty());
}

TEST(FvCompose, AnExportOfAPlainSceneStripsNothing)
{
  std::vector<Part> parts(1);
  const tinygltf::Model model = smallModel("chair");
  parts[0].id                 = "p:0";
  parts[0].model              = &model;
  std::vector<Placed>   placed;
  tinygltf::Model       scene  = fv::composer::compose(parts, placed);
  const tinygltf::Model before = scene;
  EXPECT_TRUE(fv::composer::stripForFile(scene).empty());
  EXPECT_TRUE(scene == before);
}
