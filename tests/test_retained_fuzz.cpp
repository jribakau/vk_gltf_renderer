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

// Incremental equals fresh, at the store (oracle 2a of daz-viewer docs/retained-scene-plan.md, the
// headline test's first stage). Seeded random op sequences drive a retained scene as the host will:
// props imported and removed, the Box Figure fed while things are worn, taken off, grafted, hidden,
// dialled and posed, entities hidden, moved, reparented (cycles refused), reordered and duplicated,
// materials edited and their textures swapped, vertices edited, lights added, edited and removed,
// clips added and played, the scene cleared.
//
// After every op the store is held to what is worked out from scratch: the references each record
// has, the tree, world matrices and visibility, the shader features (counted, and found in the
// export), checkFeed on the figure, and the fuzz's own account of the props (a shadow kept under the
// store's handles, so a handle that survives keeps its record); and the next takeChanges() to what
// the op touched. Every few ops, and at the end of each sequence, a fresh store is built in one go
// from the same state (the shadow replayed, the figure fed again from its scene graph) and the two
// are compared as multisets keyed by owner tag: slot numbering may differ, content may not.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <span>

#include <glm/gtc/matrix_transform.hpp>

#include <figure/figure.hpp>
#include <figure/gltf/gltf_build.hpp>
#include <figure/mesh/mesh_build.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/pose_rig.hpp>

#include "fv_figure_feed.hpp"
#include "fv_figure_feed_check.hpp"
#include "scene_clip_player.hpp"
#include "scene_export_gltf.hpp"
#include "scene_feature_detection.hpp"
#include "scene_import_gltf.hpp"
#include "tinygltf_utils.hpp"

namespace {

namespace fs = std::filesystem;
using nvvkgltf::SceneFeatureSet;

const fs::path kLibrary = fs::path(FV_TEST_DATA) / "daz-library";

// The figure's owner tags; every prop, light and clip the fuzz adds takes the next number
constexpr uint32_t kFigureProp = 1;
// the host's centimetres
constexpr double kRootScale = 0.01;

// Seeds past the default run's that a sequence once failed on, run with it (none yet: the
// default run's own seeds come first, and a failing one among them stays there)
const std::vector<uint32_t> kRegressionSeeds = {};
// The default run: seeds 1 to kSequences, each a sequence of kSteps ops, compared with a fresh store
// every kCompareEvery ops and at the end.
constexpr uint32_t kSequences    = 20;
constexpr int      kSteps        = 500;
constexpr int      kCompareEvery = 20;

//--------------------------------------------------------------------------------------------------
// Random content, the same for a seed on every platform (no std:: distributions)

bool chance(std::mt19937& rng, uint32_t n)
{
  return rng() % n == 0;
}

float uniform(std::mt19937& rng, float a, float b)
{
  return a + (b - a) * float(rng() >> 8) * (1.f / 16777216.f);
}

template <class T>
const T& pick(std::mt19937& rng, const std::vector<T>& v)
{
  return v[rng() % v.size()];
}

glm::quat randomRotation(std::mt19937& rng)
{
  const glm::quat q(uniform(rng, -1, 1), uniform(rng, -1, 1), uniform(rng, -1, 1), uniform(rng, -1, 1));
  return glm::length(q) > 0.01f ? glm::normalize(q) : glm::quat(1.f, 0.f, 0.f, 0.f);
}

glm::vec3 randomVec3(std::mt19937& rng, float a, float b)
{
  return {uniform(rng, a, b), uniform(rng, a, b), uniform(rng, a, b)};
}

rs::Transform randomTransform(std::mt19937& rng)
{
  rs::Transform t;
  t.translation = randomVec3(rng, -2, 2);
  if(chance(rng, 2))
    t.rotation = randomRotation(rng);
  if(chance(rng, 4))
    t.scale = randomVec3(rng, 0.5f, 2);
  if(chance(rng, 5))
    t.matrix = t.local();
  return t;
}

tinygltf::Light randomLight(std::mt19937& rng)
{
  static const char* types[] = {"point", "spot", "directional"};
  tinygltf::Light    l;
  l.type      = types[rng() % 3];
  l.color     = {uniform(rng, 0, 1), uniform(rng, 0, 1), uniform(rng, 0, 1)};
  l.intensity = uniform(rng, 0.5f, 50);
  if(l.type == "spot")
    l.spot.outerConeAngle = uniform(rng, 0.2f, 0.8f);
  return l;
}

// A payload over a table of `textures`: base colour, a few extensions the detector knows, now and
// then a normal map (tangents) and a texture transform
tinygltf::Material randomPayload(std::mt19937& rng, size_t textures)
{
  static const char* extensions[] = {KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME,    KHR_MATERIALS_SHEEN_EXTENSION_NAME,
                                     KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME, KHR_MATERIALS_VOLUME_EXTENSION_NAME,
                                     KHR_MATERIALS_IOR_EXTENSION_NAME,          KHR_MATERIALS_SPECULAR_EXTENSION_NAME,
                                     KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME,  KHR_MATERIALS_UNLIT_EXTENSION_NAME};
  tinygltf::Material m;
  m.name                                 = "edited";
  m.pbrMetallicRoughness.baseColorFactor = {uniform(rng, 0, 1), uniform(rng, 0, 1), uniform(rng, 0, 1), 1};
  m.pbrMetallicRoughness.roughnessFactor = uniform(rng, 0, 1);
  for(const char* e : extensions)
    if(chance(rng, 6))
      m.extensions[e] = tinygltf::Value(tinygltf::Value::Object{});
  if(textures && chance(rng, 3))
    m.normalTexture.index = int(rng() % textures);
  if(textures && chance(rng, 2))
  {
    m.pbrMetallicRoughness.baseColorTexture.index = int(rng() % textures);
    if(chance(rng, 3))
      m.pbrMetallicRoughness.baseColorTexture.extensions[KHR_TEXTURE_TRANSFORM_EXTENSION_NAME] =
          tinygltf::Value(tinygltf::Value::Object{});
  }
  return m;
}

const char* pathName(rs::ClipPath path)
{
  static const char* names[] = {"translation", "rotation", "scale", "weights"};
  return names[int(path)];
}

// Keys for `path`: two to four times in order, the values (`weights` per key for morph weights),
// a cubic spline key with its tangents around its value
rs::ClipSampler randomSampler(std::mt19937& rng, rs::ClipPath path, uint32_t weights)
{
  rs::ClipSampler s;
  s.interpolation     = rs::ClipInterpolation(rng() % 3);
  const uint32_t keys = 2 + rng() % 3;
  float          t    = uniform(rng, 0, 0.5f);
  for(uint32_t k = 0; k < keys; k++, t += uniform(rng, 0.1f, 1))
    s.inputs.push_back(t);
  const uint32_t parts = s.interpolation == rs::ClipInterpolation::eCubicSpline ? 3 : 1;
  for(uint32_t k = 0; k < keys; k++)
    for(uint32_t p = 0; p < parts; p++)
    {
      const bool value = parts == 1 || p == 1;
      if(path == rs::ClipPath::eRotation && value)
      {
        const glm::quat q = randomRotation(rng);
        s.outputs.insert(s.outputs.end(), {q.x, q.y, q.z, q.w});
        continue;
      }
      const uint32_t count = path == rs::ClipPath::eRotation ? 4 : path == rs::ClipPath::eWeights ? weights : 3;
      for(uint32_t c = 0; c < count; c++)
        s.outputs.push_back(!value                         ? uniform(rng, -1, 1) :
                            path == rs::ClipPath::eScale   ? uniform(rng, 0.5f, 1.5f) :
                            path == rs::ClipPath::eWeights ? uniform(rng, 0, 1) :
                                                             uniform(rng, -1, 1));
    }
  return s;
}

template <class T>
int accessor(tinygltf::Model& m, const std::vector<T>& v, int component, int type)
{
  const size_t count =
      v.size() * sizeof(T)
      / (tinygltf::GetComponentSizeInBytes(uint32_t(component)) * size_t(tinygltf::GetNumComponentsInType(uint32_t(type))));
  return tinygltf::utils::appendAccessor(m, 0, v.data(), v.size() * sizeof(T), component, type, count);
}

// A prop of its own for each add: a random tree of TRS and matrix nodes (some hidden), a mesh of
// one to three material groups over one vertex set (now and then one with no triangles, which no
// scene draws, or with no material), a mesh over two vertex sets (one not indexed), a skinned mesh
// with a morph target, now and then instancing copies, a light, a camera and an animation of its
// nodes. A normal-mapped material gives its groups tangents; extensions give features.
tinygltf::Model propModel(std::mt19937& rng)
{
  constexpr int   kFloat = TINYGLTF_COMPONENT_TYPE_FLOAT;
  tinygltf::Model m;
  m.buffers.emplace_back();

  // a 3 x 3 grid of vertices and its eight triangles
  std::vector<glm::vec3> gridPositions, gridNormals;
  std::vector<glm::vec2> gridUvs;
  std::vector<uint32_t>  gridIndices;
  for(int y = 0; y < 3; y++)
    for(int x = 0; x < 3; x++)
    {
      gridPositions.push_back({float(x), float(y), uniform(rng, -0.2f, 0.2f)});
      gridNormals.push_back({0, 0, 1});
      gridUvs.push_back({x * 0.5f, y * 0.5f});
    }
  for(uint32_t y = 0; y < 2; y++)
    for(uint32_t x = 0; x < 2; x++)
    {
      const uint32_t a = y * 3 + x;
      gridIndices.insert(gridIndices.end(), {a, a + 1, a + 4, a, a + 4, a + 3});
    }
  const std::map<std::string, int> grid = {{"POSITION", accessor(m, gridPositions, kFloat, TINYGLTF_TYPE_VEC3)},
                                           {"NORMAL", accessor(m, gridNormals, kFloat, TINYGLTF_TYPE_VEC3)},
                                           {"TEXCOORD_0", accessor(m, gridUvs, kFloat, TINYGLTF_TYPE_VEC2)}};
  // a strip of six vertices skinned to two joints, with one morph target
  const std::vector<glm::vec3>    stripPositions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 2, 0}, {1, 2, 0}};
  const std::vector<glm::vec3>    stripNormals(6, glm::vec3(0, 0, 1));
  const std::vector<glm::u16vec4> stripJoints  = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0},
                                                  {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
  const std::vector<glm::vec4>    stripWeights = {{1, 0, 0, 0}, {1, 0, 0, 0}, {0.5f, 0.5f, 0, 0},
                                                  {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
  std::vector<glm::vec3>          lift;
  for(int v = 0; v < 6; v++)
    lift.push_back(randomVec3(rng, -0.3f, 0.3f));
  const std::vector<uint32_t> stripIndices = {0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4};
  // a lone triangle, positions only
  const std::vector<glm::vec3> triangle = {{0, 0, 2}, {2, 0, 2}, {0, 2, 2}};

  // three materials: plain, normal-mapped, and with a few extensions
  for(const char* uri : {"prop-a.png", "prop-b.png"})
  {
    tinygltf::Image image;
    image.uri = uri;
    m.images.push_back(image);
  }
  tinygltf::Sampler clamp;
  clamp.wrapS = TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE;
  m.samplers  = {clamp};
  tinygltf::Texture a, b;
  a.source   = 0;
  b.source   = 1;
  b.sampler  = 0;
  m.textures = {a, b};
  tinygltf::Material plain;
  plain.name                                 = "plain";
  plain.pbrMetallicRoughness.baseColorFactor = {uniform(rng, 0, 1), uniform(rng, 0, 1), uniform(rng, 0, 1), 1};
  tinygltf::Material mapped;
  mapped.name                                        = "mapped";
  mapped.normalTexture.index                         = 0;
  mapped.pbrMetallicRoughness.baseColorTexture.index = 1;
  tinygltf::Material extended                        = randomPayload(rng, 2);
  extended.name                                      = "extended";
  m.materials                                        = {plain, mapped, extended};
  const auto material                                = [&] { return chance(rng, 6) ? -1 : int(rng() % 3); };

  // a run of whole triangles of a vertex set, now and then none
  const auto indices = [&](const std::vector<uint32_t>& all) {
    if(chance(rng, 8))
      return accessor(m, std::vector<uint32_t>{}, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR);
    const size_t triangles = all.size() / 3;
    const size_t first     = rng() % triangles;
    const size_t count     = 1 + rng() % (triangles - first);
    return accessor(m, std::vector<uint32_t>(all.begin() + first * 3, all.begin() + (first + count) * 3),
                    TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR);
  };
  tinygltf::Mesh groups;
  groups.name = "groups";
  for(uint32_t p = 1 + rng() % 3; p > 0; p--)
  {
    tinygltf::Primitive prim;
    prim.attributes = grid;
    prim.indices    = indices(gridIndices);
    prim.material   = material();
    groups.primitives.push_back(prim);
  }
  tinygltf::Mesh parts;
  parts.name = "parts";
  {
    tinygltf::Primitive onGrid;
    onGrid.attributes = grid;
    onGrid.indices    = indices(gridIndices);
    onGrid.material   = material();
    tinygltf::Primitive lone;
    lone.attributes  = {{"POSITION", accessor(m, triangle, kFloat, TINYGLTF_TYPE_VEC3)}};
    lone.material    = material();
    parts.primitives = {onGrid, lone};
  }
  tinygltf::Mesh skinned;
  skinned.name = "skinned";
  {
    tinygltf::Primitive prim;
    prim.attributes = {{"POSITION", accessor(m, stripPositions, kFloat, TINYGLTF_TYPE_VEC3)},
                       {"NORMAL", accessor(m, stripNormals, kFloat, TINYGLTF_TYPE_VEC3)},
                       {"JOINTS_0", accessor(m, stripJoints, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC4)},
                       {"WEIGHTS_0", accessor(m, stripWeights, kFloat, TINYGLTF_TYPE_VEC4)}};
    prim.targets       = {{{"POSITION", accessor(m, lift, kFloat, TINYGLTF_TYPE_VEC3)}}};
    prim.indices       = accessor(m, stripIndices, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR);
    prim.material      = material();
    skinned.primitives = {prim};
    skinned.weights    = {uniform(rng, 0, 1)};
  }
  m.meshes = {groups, parts, skinned};

  // the tree
  const int count = 4 + int(rng() % 5);
  m.nodes.resize(size_t(count));
  for(int i = 0; i < count; i++)
  {
    tinygltf::Node& n = m.nodes[size_t(i)];
    n.name            = "n" + std::to_string(i);
    if(i > 0)
      m.nodes[rng() % size_t(i)].children.push_back(i);
    const rs::Transform t = randomTransform(rng);
    if(t.matrix)
      n.matrix.assign(&(*t.matrix)[0][0], &(*t.matrix)[0][0] + 16);
    else
    {
      n.translation = {t.translation.x, t.translation.y, t.translation.z};
      n.rotation    = {t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};
      n.scale       = {t.scale.x, t.scale.y, t.scale.z};
    }
    if(chance(rng, 7))
      tinygltf::utils::setNodeVisibility(n, {false});
    if(chance(rng, 2))
    {
      n.mesh = int(rng() % 2);
      if(chance(rng, 8))
      {
        const std::vector<glm::vec3> offsets = {{0, 0, 0}, randomVec3(rng, -3, 3)};
        tinygltf::Value::Object      attributes;
        attributes["TRANSLATION"] = tinygltf::Value(accessor(m, offsets, kFloat, TINYGLTF_TYPE_VEC3));
        tinygltf::Value::Object instancing;
        instancing["attributes"]                             = tinygltf::Value(attributes);
        n.extensions[EXT_MESH_GPU_INSTANCING_EXTENSION_NAME] = tinygltf::Value(instancing);
      }
    }
  }
  // the skinned mesh on one node, over two of the prop's nodes
  const int skinnedNode             = int(rng() % size_t(count));
  m.nodes[size_t(skinnedNode)].mesh = 2;
  m.nodes[size_t(skinnedNode)].skin = 0;
  m.nodes[size_t(skinnedNode)].extensions.erase(EXT_MESH_GPU_INSTANCING_EXTENSION_NAME);
  tinygltf::Skin skin;
  skin.joints                               = {int(rng() % size_t(count)), int(rng() % size_t(count))};
  skin.skeleton                             = 0;
  const std::vector<glm::mat4> inverseBinds = {glm::translate(glm::mat4(1.f), randomVec3(rng, -1, 1)),
                                               glm::translate(glm::mat4(1.f), randomVec3(rng, -1, 1))};
  skin.inverseBindMatrices                  = accessor(m, inverseBinds, kFloat, TINYGLTF_TYPE_MAT4);
  m.skins                                   = {skin};

  if(chance(rng, 2))
  {
    m.lights                             = {randomLight(rng)};
    m.nodes[rng() % size_t(count)].light = 0;
  }
  if(chance(rng, 3))
  {
    tinygltf::Camera camera;
    camera.type                           = "perspective";
    camera.perspective.yfov               = uniform(rng, 0.3f, 1.2f);
    camera.perspective.znear              = 0.1;
    m.cameras                             = {camera};
    m.nodes[rng() % size_t(count)].camera = 0;
  }
  // an animation of a few nodes, and of the skinned node's morph weight
  if(chance(rng, 2))
  {
    tinygltf::Animation animation;
    animation.name = "take";
    for(uint32_t c = 1 + rng() % 3; c > 0; c--)
    {
      const bool                 weights = chance(rng, 4);
      const rs::ClipPath         path    = weights ? rs::ClipPath::eWeights : rs::ClipPath(rng() % 3);
      const rs::ClipSampler      keys    = randomSampler(rng, path, 1);
      tinygltf::AnimationSampler s;
      s.input         = accessor(m, keys.inputs, kFloat, TINYGLTF_TYPE_SCALAR);
      s.output        = accessor(m, keys.outputs, kFloat,
                                 path == rs::ClipPath::eRotation ? TINYGLTF_TYPE_VEC4 :
                                 path == rs::ClipPath::eWeights  ? TINYGLTF_TYPE_SCALAR :
                                                                   TINYGLTF_TYPE_VEC3);
      s.interpolation = keys.interpolation == rs::ClipInterpolation::eStep        ? "STEP" :
                        keys.interpolation == rs::ClipInterpolation::eCubicSpline ? "CUBICSPLINE" :
                                                                                    "LINEAR";
      tinygltf::AnimationChannel channel;
      channel.sampler     = int(animation.samplers.size());
      channel.target_node = weights ? skinnedNode : int(rng() % size_t(count));
      channel.target_path = pathName(path);
      animation.samplers.push_back(s);
      animation.channels.push_back(channel);
    }
    m.animations = {animation};
  }
  m.scenes.emplace_back().nodes = {0};
  m.defaultScene                = 0;
  return m;
}

// An item's cage: n x n quads of 10 cm, UVs over the square
fv::mesh::MeshInput itemGrid(int n, float z)
{
  fv::mesh::MeshInput m;
  for(int y = 0; y <= n; y++)
    for(int x = 0; x <= n; x++)
    {
      m.positions.insert(m.positions.end(), {x * 10.f, y * 10.f + 100.f, z});
      m.uvs.insert(m.uvs.end(), {float(x) / float(n), float(y) / float(n)});
    }
  m.polyStart.push_back(0);
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++)
    {
      const int a = y * (n + 1) + x;
      m.polyVerts.insert(m.polyVerts.end(), {a, a + 1, a + n + 2, a + n + 1});
      m.polyStart.push_back(int(m.polyVerts.size()));
      m.polyMaterial.push_back(0);
    }
  return m;
}

// Every texture resolves to its source file, uncooked (the UVs flip)
fv::build::BuildOptions textured()
{
  fv::build::BuildOptions o;
  o.texture = [](const fv::build::TextureRequest& req) -> std::optional<fv::build::TextureImage> {
    return fv::build::TextureImage{.uri = (kLibrary / req.rel).generic_string(), .mimeType = "image/png"};
  };
  return o;
}

SceneFeatureSet materialBits(SceneFeatureSet set)
{
  set.set(SceneFeatureSet::eDlssGuide, false);
  return set;
}

std::string tag(rs::OwnerTag o)
{
  return std::to_string(o.prop) + ":" + std::to_string(o.local);
}

template <class Tag>
std::string handle(rs::Handle<Tag> id)
{
  return std::to_string(id.index) + "." + std::to_string(id.gen);
}

// The first stream two mesh records differ in; empty when they hold the same (contentRev aside)
std::string meshDifference(const rs::MeshRecord& a, const rs::MeshRecord& b)
{
  if(a.name != b.name || a.owner != b.owner)
    return "name or owner";
  if(a.positions != b.positions)
    return "positions";
  if(a.normals != b.normals)
    return "normals";
  if(a.tangents != b.tangents)
    return "tangents";
  if(a.uv0 != b.uv0 || a.uv1 != b.uv1)
    return "UVs";
  if(a.colors != b.colors)
    return "colours";
  if(a.joints != b.joints || a.weights != b.weights)
    return "joints or weights";
  if(a.indices != b.indices)
    return "indices";
  if(a.submeshes != b.submeshes)
    return "submeshes";
  if(a.targets != b.targets || a.targetWeights != b.targetWeights)
    return "morph targets";
  return {};
}

bool sameTexture(const rs::TextureRecord& a, const rs::TextureRecord& b)
{
  return a.name == b.name && a.uri == b.uri && a.bytes == b.bytes && a.mimeType == b.mimeType && a.width == b.width
         && a.height == b.height && a.pixels == b.pixels && a.component == b.component && a.bits == b.bits
         && a.sourceExtension == b.sourceExtension;
}

bool sameSkinRecord(const rs::SkinRecord& a, const rs::SkinRecord& b)
{
  return a.name == b.name && a.joints == b.joints && a.inverseBind == b.inverseBind && a.skeleton == b.skeleton;
}

bool sameInstance(const rs::MeshInstance& a, const rs::MeshInstance& b)
{
  if(a.skin != b.skin || a.morphWeights != b.morphWeights || a.copies != b.copies || a.parts.size() != b.parts.size())
    return false;
  for(size_t p = 0; p < a.parts.size(); p++)
    if(a.parts[p].mesh != b.parts[p].mesh || a.parts[p].materials != b.parts[p].materials)
      return false;
  return true;
}

template <class Id>
bool live(const rs::RetainedScene& s, Id id)
{
  if constexpr(std::is_same_v<Id, rs::EntityId>)
    return s.entity(id) != nullptr;
  else if constexpr(std::is_same_v<Id, rs::MeshId>)
    return s.mesh(id) != nullptr;
  else if constexpr(std::is_same_v<Id, rs::MaterialId>)
    return s.material(id) != nullptr;
  else if constexpr(std::is_same_v<Id, rs::LightId>)
    return s.light(id) != nullptr;
  else
    return s.skin(id) != nullptr;
}

// Pairs each value of `a` under a key with an equal one of `b` under the same key; what is left over
// on either side is a difference
template <class K, class V, class Eq, class Describe>
void expectSameMultiset(const std::multimap<K, V>& a, const std::multimap<K, V>& b, Eq eq, Describe describe, const char* what)
{
  EXPECT_EQ(a.size(), b.size()) << what;
  for(auto it = a.begin(); it != a.end();)
  {
    const auto [ab, ae] = a.equal_range(it->first);
    const auto [bb, be] = b.equal_range(it->first);
    std::vector<bool> used(size_t(std::distance(bb, be)), false);
    for(auto x = ab; x != ae; ++x)
    {
      bool   found = false;
      size_t k     = 0;
      for(auto y = bb; y != be && !found; ++y, ++k)
        if(!used[k] && eq(x->second, y->second))
          used[k] = found = true;
      if(!found)
        ADD_FAILURE() << what << " " << describe(it->first) << ": the incremental store's has no match in the fresh one";
    }
    it = ae;
  }
}

//--------------------------------------------------------------------------------------------------
// What an op has to make the next takeChanges() report: everything in `must`, nothing outside
// `must` and `may` (where the store may say more than it has to: a new entity's visibility, a
// reorder's transforms, a dial's skins)

template <class Id>
struct Want
{
  std::set<Id> must, may;
  void         both(Id id)
  {
    must.insert(id);
    may.insert(id);
  }
};

struct Expected
{
  bool                 structure    = false;
  bool                 structureMay = false;  // either way
  Want<rs::EntityId>   transforms, visibility, morphWeights;
  Want<rs::MeshId>     meshes;
  Want<rs::MaterialId> materials;
  Want<rs::LightId>    lights;
  Want<rs::SkinId>     skins;
};

//--------------------------------------------------------------------------------------------------
// The fuzz's own account of the props, kept beside the store under the store's handles: what each
// entity is and what it draws or carries, every record by value. The store is held to it after
// every op, and a fresh store is built from it.

struct Shadow
{
  struct EntityRecord
  {
    std::string                     name;
    rs::OwnerTag                    owner;
    rs::EntityId                    parent;
    rs::Transform                   local;
    bool                            visible = true;
    std::optional<rs::MeshInstance> mesh;
    rs::LightId                     light;
    rs::CameraId                    camera;
  };
  std::map<rs::EntityId, EntityRecord>         entities;
  std::map<rs::MeshId, rs::MeshRecord>         meshes;
  std::map<rs::MaterialId, rs::MaterialRecord> materials;
  std::map<rs::TextureId, rs::TextureRecord>   textures;
  std::map<rs::SkinId, rs::SkinRecord>         skins;
  std::map<rs::LightId, rs::LightRecord>       lights;
  std::map<rs::CameraId, rs::CameraRecord>     cameras;
  std::map<rs::ClipId, rs::ClipRecord>         clips;

  // An entity the store just made (an import's, a duplicate's), with the records it names that
  // the shadow has not met
  void adopt(const rs::RetainedScene& s, rs::EntityId id)
  {
    const rs::Entity& e = *s.entity(id);
    entities[id]        = {e.name, e.owner, e.parent, e.local, e.visible, e.mesh, e.light, e.camera};
    if(e.mesh)
    {
      for(const rs::MeshPart& part : e.mesh->parts)
      {
        meshes.try_emplace(part.mesh, *s.mesh(part.mesh));
        for(rs::MaterialId m : part.materials)
          adoptMaterial(s, m);
      }
      if(!e.mesh->skin.isNone())
        skins.try_emplace(e.mesh->skin, *s.skin(e.mesh->skin));
    }
    if(!e.light.isNone())
      lights.try_emplace(e.light, *s.light(e.light));
    if(!e.camera.isNone())
      cameras.try_emplace(e.camera, *s.camera(e.camera));
  }
  void adoptMaterial(const rs::RetainedScene& s, rs::MaterialId id)
  {
    if(materials.try_emplace(id, *s.material(id)).second)
      adoptTextures(s, materials[id].textures);
  }
  void adoptTextures(const rs::RetainedScene& s, const std::shared_ptr<const rs::TextureTable>& table)
  {
    if(table)
      for(const rs::TextureRef& ref : *table)
        if(const rs::TextureRecord* t = s.texture(ref.texture))
          textures.try_emplace(ref.texture, *t);
  }
  // `root` and everything under it, by the shadow's own links
  void drop(rs::EntityId root)
  {
    std::set<rs::EntityId> gone{root};
    for(bool grew = true; grew;)
    {
      grew = false;
      for(const auto& [id, e] : entities)
        if(!gone.count(id) && gone.count(e.parent))
        {
          gone.insert(id);
          grew = true;
        }
    }
    for(rs::EntityId id : gone)
      entities.erase(id);
  }
};

// What the fuzz added, and lets go of together: imported nodes, a duplicate, a lamp; and the clips
// held for it (an import hands them over, and so does a clip add)
struct Prop
{
  uint32_t                  id = 0;
  std::vector<rs::EntityId> roots;
  std::vector<rs::ClipId>   clips;
};

//--------------------------------------------------------------------------------------------------

class Fuzz
{
public:
  Fuzz(uint32_t seed)
      : m_seed(seed)
      , m_rng(seed)
  {
    fv::dson::LoadOptions options;
    options.subdivision = 1;
    // odd seeds dress the PBRSkin variant, whose detail layer asks for no tangents
    const std::string rel = seed % 2 ? "People/Test/Box Figure PBRSkin.duf" : "People/Test/Box Figure.duf";
    m_figure              = fv::Figure::load(
        std::make_shared<fv::dson::Store>(std::make_shared<fv::dson::Library>(std::vector<fs::path>{kLibrary})), rel, options);
    m_figure->settle();
    m_figure->drain();
    m_bounds = m_scene.bounds();
  }

  // `steps` ops, each checked; compared with a fresh store every `compareEvery` and at the end.
  // Stops at the first failing step: the trace names the seed, the step and the op.
  void run(int steps, int compareEvery)
  {
    for(int i = 0; i < steps; i++)
    {
      SCOPED_TRACE("seed " + std::to_string(m_seed) + ", step " + std::to_string(i));
      Expected          want;
      const std::string op = step(want);
      SCOPED_TRACE(op);
      checkStore();
      expectChanges(want);
      if((i + 1) % compareEvery == 0 || i + 1 == steps)
        compareWithFresh();
      if(::testing::Test::HasFailure())
        return;
    }
  }

private:
  bool  chance(uint32_t n) { return ::chance(m_rng, n); }
  float uniform(float a, float b) { return ::uniform(m_rng, a, b); }
  template <class T>
  const T& pick(const std::vector<T>& v)
  {
    return ::pick(m_rng, v);
  }

  fv::SceneGraph&       graph() { return m_figure->loaded.scene; }
  const fv::SceneGraph& graph() const { return m_figure->loaded.scene; }
  int                   root() const { return m_figure->loaded.root; }

  //------------------------------------------------------------------------------------------------
  // Ops

  std::string step(Expected& want)
  {
    struct Op
    {
      int weight;
      std::string (Fuzz::*run)(Expected&);
    };
    // weighted so that each runs a few hundred times in the default run (a clear, a few dozen)
    static const Op ops[] = {
        {12, &Fuzz::addProp},     {6, &Fuzz::removeProp},    {8, &Fuzz::duplicate},      {12, &Fuzz::hide},
        {10, &Fuzz::move},        {14, &Fuzz::reparent},     {8, &Fuzz::reorder},        {10, &Fuzz::editMaterial},
        {8, &Fuzz::swapTextures}, {10, &Fuzz::editVertices}, {6, &Fuzz::addLight},       {8, &Fuzz::editLight},
        {4, &Fuzz::removeLight},  {6, &Fuzz::addClip},       {10, &Fuzz::playClip},      {8, &Fuzz::wear},
        {6, &Fuzz::takeOff},      {8, &Fuzz::graft},         {8, &Fuzz::hideFigureNode}, {6, &Fuzz::editFigureMaterial},
        {6, &Fuzz::dial},         {6, &Fuzz::pose},          {2, &Fuzz::toggleTextures}, {1, &Fuzz::clear},
    };
    int total = 0;
    for(const Op& op : ops)
      total += op.weight;
    int r = int(m_rng() % uint32_t(total));
    for(const Op& op : ops)
    {
      if(r < op.weight)
        return (this->*op.run)(want);
      r -= op.weight;
    }
    return "none";
  }

  // the props' entities, in handle order
  std::vector<rs::EntityId> propEntities() const
  {
    std::vector<rs::EntityId> out;
    for(const auto& [id, e] : m_shadow.entities)
      out.push_back(id);
    return out;
  }

  // an entity and everything under it, as the store has them
  std::vector<rs::EntityId> subtree(rs::EntityId id) const
  {
    std::vector<rs::EntityId> out;
    for(std::vector<rs::EntityId> stack{id}; !stack.empty();)
    {
      const rs::EntityId e = stack.back();
      stack.pop_back();
      if(const rs::Entity* x = m_scene.entity(e))
      {
        out.push_back(e);
        stack.insert(stack.end(), x->children.begin(), x->children.end());
      }
    }
    return out;
  }

  const std::vector<rs::EntityId>& siblings(rs::EntityId parent) const
  {
    return parent.isNone() ? m_scene.roots() : m_scene.entity(parent)->children;
  }

  // where setParent(id, parent, index) puts an entity among `before`, its new siblings without it
  static std::vector<rs::EntityId> placed(std::vector<rs::EntityId> before, rs::EntityId id, int index)
  {
    before.insert(index < 0 || index >= int(before.size()) ? before.end() : before.begin() + index, id);
    return before;
  }

  // the materials and meshes the props draw with, the lights they carry
  std::vector<rs::MaterialId> propMaterials() const
  {
    std::set<rs::MaterialId> out;
    for(const auto& [id, e] : m_shadow.entities)
      if(e.mesh)
        for(const rs::MeshPart& part : e.mesh->parts)
          out.insert(part.materials.begin(), part.materials.end());
    return {out.begin(), out.end()};
  }
  std::vector<rs::MeshId> propMeshes() const
  {
    std::set<rs::MeshId> out;
    for(const auto& [id, e] : m_shadow.entities)
      if(e.mesh)
        for(const rs::MeshPart& part : e.mesh->parts)
          out.insert(part.mesh);
    return {out.begin(), out.end()};
  }
  std::vector<rs::LightId> propLights() const
  {
    std::set<rs::LightId> out;
    for(const auto& [id, e] : m_shadow.entities)
      if(!e.light.isNone())
        out.insert(e.light);
    return {out.begin(), out.end()};
  }

  std::string addProp(Expected& want)
  {
    Prop                            p{m_nextProp++};
    const std::vector<rs::EntityId> entities = propEntities();
    const rs::EntityId              parent   = !entities.empty() && chance(3) ? pick(entities) : rs::EntityId{};
    const rs::ImportResult          r = rs::importGltf(m_scene, propModel(m_rng), {.prop = p.id, .parent = parent});
    p.roots                           = r.roots;
    for(rs::ClipId c : r.clips)
      if(!c.isNone())
      {
        p.clips.push_back(c);
        m_shadow.clips.emplace(c, *m_scene.clip(c));
      }
    for(rs::EntityId e : r.entities)
      if(!e.isNone())
      {
        m_shadow.adopt(m_scene, e);
        want.transforms.both(e);
        want.visibility.may.insert(e);
        if(const rs::Entity* x = m_scene.entity(e); x->mesh && !x->mesh->skin.isNone())
          want.skins.may.insert(x->mesh->skin);
      }
    want.structure = true;
    m_props.push_back(std::move(p));
    return "add prop " + std::to_string(m_props.back().id) + (parent.isNone() ? "" : " under " + handle(parent));
  }

  std::string removeProp(Expected& want)
  {
    if(m_props.empty())
      return "remove prop: none";
    const size_t i = m_rng() % m_props.size();
    const Prop   p = m_props[i];
    m_props.erase(m_props.begin() + ptrdiff_t(i));
    for(rs::EntityId r : p.roots)
      if(m_scene.entity(r))
      {
        m_shadow.drop(r);
        m_scene.destroyEntity(r);
        want.structure = true;
      }
    for(rs::ClipId c : p.clips)
    {
      m_scene.release(c);
      m_shadow.clips.erase(c);
    }
    return "remove prop " + std::to_string(p.id);
  }

  // A copy of an entity and everything under it beside it, a prop of its own, drawing with the same
  // records and carrying the same lights and cameras
  std::string duplicate(Expected& want)
  {
    const std::vector<rs::EntityId> entities = propEntities();
    if(entities.empty())
      return "duplicate: none";
    const rs::EntityId src = pick(entities);
    Prop               p{m_nextProp++};
    int32_t            local = 0;
    const std::function<rs::EntityId(rs::EntityId, rs::EntityId, int)> copy = [&](rs::EntityId from, rs::EntityId parent, int index) {
      const rs::Entity   e  = *m_scene.entity(from);  // a copy: creating entities may move the store's
      const rs::EntityId id = m_scene.createEntity(e.name, {p.id, local++}, parent, index);
      m_scene.setLocal(id, e.local);
      m_scene.setVisible(id, e.visible);
      if(e.mesh)
        m_scene.setMeshInstance(id, e.mesh);
      if(!e.light.isNone())
        m_scene.setEntityLight(id, e.light);
      if(!e.camera.isNone())
        m_scene.setEntityCamera(id, e.camera);
      m_shadow.adopt(m_scene, id);
      want.transforms.both(id);
      want.visibility.may.insert(id);
      for(rs::EntityId c : e.children)
        copy(c, id, -1);
      return id;
    };
    const rs::EntityId              parent = m_scene.entity(src)->parent;
    const std::vector<rs::EntityId> before = siblings(parent);
    const int                       index  = int(m_rng() % (before.size() + 2)) - 1;
    const rs::EntityId              top    = copy(src, parent, index);
    EXPECT_EQ(siblings(parent), placed(before, top, index)) << "the copy's place among its siblings";
    p.roots        = {top};
    want.structure = true;
    m_props.push_back(std::move(p));
    return "duplicate " + handle(src);
  }

  std::string hide(Expected& want)
  {
    const std::vector<rs::EntityId> entities = propEntities();
    if(entities.empty())
      return "hide: none";
    const rs::EntityId id      = pick(entities);
    const bool         visible = !m_shadow.entities[id].visible;
    m_scene.setVisible(id, visible);
    m_shadow.entities[id].visible = visible;
    for(rs::EntityId e : subtree(id))
      want.visibility.both(e);
    return (visible ? "show " : "hide ") + handle(id);
  }

  std::string move(Expected& want)
  {
    const std::vector<rs::EntityId> entities = propEntities();
    if(entities.empty())
      return "move: none";
    const rs::EntityId  id    = pick(entities);
    const rs::Transform local = randomTransform(m_rng);
    m_scene.setLocal(id, local);
    m_shadow.entities[id].local = local;
    for(rs::EntityId e : subtree(id))
      want.transforms.both(e);
    return "move " + handle(id);
  }

  // Under another prop entity or to the roots, keeping its world or not; a quarter of the time
  // under itself or something under it, which is refused and changes nothing
  std::string reparent(Expected& want)
  {
    const std::vector<rs::EntityId> entities = propEntities();
    if(entities.empty())
      return "reparent: none";
    const rs::EntityId              id    = pick(entities);
    const std::vector<rs::EntityId> under = subtree(id);
    const bool                      cycle = chance(4);
    rs::EntityId                    parent;
    if(cycle)
      parent = pick(under);
    else
    {
      std::vector<rs::EntityId> others;
      for(rs::EntityId e : entities)
        if(std::find(under.begin(), under.end(), e) == under.end())
          others.push_back(e);
      if(!others.empty() && !chance(4))
        parent = pick(others);
    }
    const bool                keepWorld = chance(2);
    std::vector<rs::EntityId> before    = siblings(parent);
    std::erase(before, id);
    const int       index       = int(m_rng() % (before.size() + 2)) - 1;
    const glm::mat4 worldBefore = m_scene.world(id);
    const glm::mat4 parentWorld = parent.isNone() ? glm::mat4(1.f) : m_scene.world(parent);
    const bool      moved       = m_scene.setParent(id, parent, index, keepWorld);
    const std::string what = std::string(cycle ? "reparent into itself " : "reparent ") + handle(id) + " under "
                             + (parent.isNone() ? "the roots" : handle(parent)) + (keepWorld ? ", keeping its world" : "");
    EXPECT_EQ(moved, !cycle) << what;
    if(!moved)
      return what;
    EXPECT_EQ(siblings(parent), placed(before, id, index)) << "its place among its new siblings";
    Shadow::EntityRecord& e = m_shadow.entities[id];
    e.parent                = parent;
    if(keepWorld)
    {
      e.local = rs::Transform{.matrix = glm::inverse(parentWorld) * worldBefore};
      // where it was, give or take the rounding of an inverse
      const glm::mat4 after = m_scene.world(id);
      for(int c = 0; c < 4; c++)
        for(int r = 0; r < 4; r++)
          EXPECT_NEAR(after[c][r], worldBefore[c][r], 1e-3f * (1.f + std::abs(worldBefore[c][r]))) << "kept its world";
    }
    for(rs::EntityId x : subtree(id))
    {
      want.transforms.both(x);
      want.visibility.both(x);
    }
    want.structure = true;
    return what;
  }

  std::string reorder(Expected& want)
  {
    const std::vector<rs::EntityId> entities = propEntities();
    if(entities.empty())
      return "reorder: none";
    const rs::EntityId        id     = pick(entities);
    const rs::EntityId        parent = m_shadow.entities[id].parent;
    std::vector<rs::EntityId> before = siblings(parent);
    std::erase(before, id);
    const int index = int(m_rng() % (before.size() + 1));
    EXPECT_TRUE(m_scene.setParent(id, parent, index));
    EXPECT_EQ(siblings(parent), placed(before, id, index)) << "its place among its siblings";
    for(rs::EntityId x : subtree(id))
    {
      want.transforms.may.insert(x);
      want.visibility.may.insert(x);
    }
    want.structure = true;
    return "reorder " + handle(id) + " to " + std::to_string(index);
  }

  std::string editMaterial(Expected& want)
  {
    const std::vector<rs::MaterialId> materials = propMaterials();
    if(materials.empty())
      return "material edit: none";
    const rs::MaterialId id   = pick(materials);
    rs::MaterialRecord&  r    = m_shadow.materials.at(id);
    tinygltf::Material   edit = randomPayload(m_rng, r.textures ? r.textures->size() : 0);
    m_scene.setMaterial(id, edit);
    r.payload = std::move(edit);
    want.materials.both(id);
    return "material edit " + handle(id);
  }

  // The material's payload over a new table: textures added for it (some held as bytes) and ones
  // other tables name
  std::string swapTextures(Expected& want)
  {
    const std::vector<rs::MaterialId> materials = propMaterials();
    if(materials.empty())
      return "texture swap: none";
    const rs::MaterialId       id = pick(materials);
    std::vector<rs::TextureId> known;
    for(const auto& [t, record] : m_shadow.textures)
      if(m_scene.texture(t))
        known.push_back(t);
    auto                       table = std::make_shared<rs::TextureTable>();
    std::vector<rs::TextureId> added;
    for(uint32_t n = 1 + m_rng() % 3; n > 0; n--)
    {
      rs::TextureId t;
      if(!known.empty() && chance(2))
        t = pick(known);
      else
      {
        rs::TextureRecord record{.name = "swap " + std::to_string(m_nextTexture++), .mimeType = "image/png"};
        if(chance(3))
          record.bytes = {uint8_t(m_rng()), uint8_t(m_rng()), uint8_t(m_rng()), uint8_t(m_rng())};
        else
          record.uri = record.name + ".png";
        t = m_scene.addTexture(record);
        m_shadow.textures.emplace(t, record);
        added.push_back(t);
      }
      table->push_back({t, rs::SamplerDesc{.wrapS = chance(2) ? TINYGLTF_TEXTURE_WRAP_REPEAT : TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT}});
    }
    tinygltf::Material payload = randomPayload(m_rng, table->size());
    m_scene.setMaterial(id, payload, table);
    for(rs::TextureId t : added)
      m_scene.release(t);
    rs::MaterialRecord& r = m_shadow.materials.at(id);
    r.payload             = std::move(payload);
    r.textures            = table;
    want.materials.both(id);
    return "texture swap " + handle(id);
  }

  // Positions moved, normals and tangents now and then given; a tenth of the time a stream of the
  // wrong length, which is refused and changes nothing
  std::string editVertices(Expected& want)
  {
    const std::vector<rs::MeshId> meshes = propMeshes();
    if(meshes.empty())
      return "vertex edit: none";
    const rs::MeshId       id  = pick(meshes);
    rs::MeshRecord&        had = m_shadow.meshes.at(id);
    std::vector<glm::vec3> positions, normals;
    std::vector<glm::vec4> tangents;
    for(const glm::vec3& p : had.positions)
      positions.push_back(p + randomVec3(m_rng, -0.1f, 0.1f));
    if(!had.normals.empty() && chance(2))
      for(size_t v = 0; v < had.normals.size(); v++)
        normals.push_back(glm::normalize(glm::vec3(0, 0, 1) + randomVec3(m_rng, -0.3f, 0.3f)));
    if(chance(3))
      for(size_t v = 0; v < had.positions.size(); v++)
        tangents.push_back(glm::vec4(1, 0, 0, chance(2) ? 1 : -1));
    const bool wrong = chance(10);
    if(wrong)
      positions.push_back(glm::vec3(0.f));
    m_scene.setMeshVertices(id, positions, normals, tangents);
    if(wrong)
      return "vertex edit of the wrong length " + handle(id);
    had.positions = std::move(positions);
    if(!normals.empty())
      had.normals = std::move(normals);
    if(!tangents.empty())
      had.tangents = std::move(tangents);
    want.meshes.both(id);
    return "vertex edit " + handle(id);
  }

  // A lamp of its own, under a prop entity or a root; a third of the time carrying a light another
  // entity does (one record, two places)
  std::string addLight(Expected& want)
  {
    const std::vector<rs::EntityId> entities = propEntities();
    const std::vector<rs::LightId>  lights   = propLights();
    Prop                            p{m_nextProp++};
    const rs::EntityId              parent = !entities.empty() && chance(2) ? pick(entities) : rs::EntityId{};
    const rs::EntityId              id     = m_scene.createEntity("lamp", {p.id, 0}, parent);
    m_scene.setLocal(id, randomTransform(m_rng));
    const bool shared = !lights.empty() && chance(3);
    if(shared)
      m_scene.setEntityLight(id, pick(lights));
    else
    {
      const rs::LightId light = m_scene.addLight({.owner = {p.id, 0}, .payload = randomLight(m_rng), .shown = !chance(5)});
      m_scene.setEntityLight(id, light);
      m_scene.release(light);
    }
    m_shadow.adopt(m_scene, id);
    p.roots = {id};
    m_props.push_back(std::move(p));
    want.transforms.both(id);
    want.visibility.may.insert(id);
    want.structure = true;
    return shared ? "add a lamp sharing a light" : "add a lamp";
  }

  std::string editLight(Expected& want)
  {
    const std::vector<rs::LightId> lights = propLights();
    if(lights.empty())
      return "light edit: none";
    const rs::LightId id = pick(lights);
    rs::LightRecord&  r  = m_shadow.lights.at(id);
    want.lights.both(id);
    switch(m_rng() % 3)
    {
      case 0:
        r.payload = randomLight(m_rng);
        m_scene.setLight(id, r.payload);
        return "light edit " + handle(id);
      case 1:
        r.shown = !r.shown;
        m_scene.setLightShown(id, r.shown);
        return "light shown " + handle(id);
      default:
        r.skySun = !r.skySun;
        m_scene.setSkySun(id, r.skySun);
        return "sky sun " + handle(id);
    }
  }

  std::string removeLight(Expected& want)
  {
    std::vector<rs::EntityId> carrying;
    for(const auto& [id, e] : m_shadow.entities)
      if(!e.light.isNone())
        carrying.push_back(id);
    if(carrying.empty())
      return "light remove: none";
    const rs::EntityId id = pick(carrying);
    m_scene.setEntityLight(id, {});
    m_shadow.entities[id].light = {};
    want.structure              = true;
    return "light remove " + handle(id);
  }

  // A clip over prop entities (morph weights on one whose meshes have targets), held with a prop
  std::string addClip(Expected& want)
  {
    const std::vector<rs::EntityId> entities = propEntities();
    if(entities.empty() || m_props.empty())
      return "clip add: none";
    rs::ClipRecord clip{.name = "fuzz", .owner = {m_nextProp++, 0}};
    for(uint32_t c = 1 + m_rng() % 3; c > 0; c--)
    {
      const rs::EntityId          target  = pick(entities);
      const Shadow::EntityRecord& e       = m_shadow.entities.at(target);
      uint32_t                    targets = 0;
      if(e.mesh && !e.mesh->parts.empty())
        targets = uint32_t(m_shadow.meshes.at(e.mesh->parts[0].mesh).targets.size());
      const rs::ClipPath path = targets && chance(3) ? rs::ClipPath::eWeights : rs::ClipPath(m_rng() % 3);
      clip.channels.push_back({.target = target, .path = path, .sampler = uint32_t(clip.samplers.size())});
      clip.samplers.push_back(randomSampler(m_rng, path, targets));
    }
    const rs::ClipId id = m_scene.addClip(clip);
    m_shadow.clips.emplace(id, std::move(clip));
    m_props[m_rng() % m_props.size()].clips.push_back(id);
    (void)want;
    return "clip add " + handle(id);
  }

  // A held clip played to a time inside its keys or a little either side
  std::string playClip(Expected& want)
  {
    std::vector<rs::ClipId> clips;
    for(const Prop& p : m_props)
      clips.insert(clips.end(), p.clips.begin(), p.clips.end());
    if(clips.empty())
      return "play: none";
    const rs::ClipId id = pick(clips);
    rs::ClipPlayer   player(m_scene, id);
    const float      time = uniform(player.start() - 0.25f, player.end() + 0.25f);
    player.setTime(time);
    player.apply(m_scene);
    // what each channel's entity has now: those that changed were moved, and every target may say so
    for(const rs::ClipChannel& c : m_shadow.clips.at(id).channels)
    {
      const rs::Entity* e = m_scene.entity(c.target);
      if(!e || !m_shadow.entities.count(c.target))
        continue;
      Shadow::EntityRecord& s = m_shadow.entities[c.target];
      if(c.path == rs::ClipPath::eWeights)
      {
        if(!e->mesh)
          continue;
        want.morphWeights.may.insert(c.target);
        if(e->mesh->morphWeights != s.mesh->morphWeights)
          want.morphWeights.must.insert(c.target);
        s.mesh->morphWeights = e->mesh->morphWeights;
        continue;
      }
      const bool changed = !(e->local == s.local);
      for(rs::EntityId x : subtree(c.target))
      {
        want.transforms.may.insert(x);
        if(changed)
          want.transforms.must.insert(x);
      }
      s.local = e->local;
    }
    return "play " + handle(id) + " at " + std::to_string(time);
  }

  // Everything goes: every prop entity (a removed prop's that had moved elsewhere too), the clips,
  // the figure's feed; the store is left empty
  std::string clear(Expected& want)
  {
    want.structure = m_scene.entityCount() > 0;
    for(rs::EntityId r : std::vector<rs::EntityId>(m_scene.roots()))
      if(m_shadow.entities.count(r))
        m_scene.destroyEntity(r);
    for(const Prop& p : m_props)
      for(rs::ClipId c : p.clips)
        m_scene.release(c);
    m_props.clear();
    m_shadow = {};
    m_feed.reset();
    m_fedEntities.clear();
    m_fedMaterials.clear();
    EXPECT_EQ(m_scene.entityCount(), 0u);
    EXPECT_TRUE(m_scene.roots().empty());
    EXPECT_EQ(m_scene.meshCount() + m_scene.materialCount() + m_scene.textureCount() + m_scene.skinCount()
                  + m_scene.lightCount() + m_scene.cameraCount() + m_scene.clipCount(),
              0u);
    EXPECT_TRUE(m_scene.features() == SceneFeatureSet{}) << m_scene.features().toString();
    return "clear";
  }

  //------------------------------------------------------------------------------------------------
  // Figure ops: the Box Figure's scene graph changed as the host changes it, then fed. Before the
  // first one, and after a clear, the figure is fed from scratch.

  fv::build::BuildOptions buildOptions() const { return m_textured ? textured() : fv::build::BuildOptions{}; }
  fv::FigureFeedOptions   feedOptions() const
  {
    return {.build = buildOptions(), .rootScale = kRootScale, .prop = kFigureProp};
  }

  // the nodes traversal reaches from the root, taken-off subtrees left out
  std::vector<bool> reached() const
  {
    std::vector<bool> out(graph().nodes.size(), false);
    for(std::vector<int> stack{root()}; !stack.empty();)
    {
      const int i = stack.back();
      stack.pop_back();
      if(graph().nodes[size_t(i)].removed)
        continue;
      out[size_t(i)] = true;
      stack.insert(stack.end(), graph().nodes[size_t(i)].children.begin(), graph().nodes[size_t(i)].children.end());
    }
    return out;
  }

  // a node's local transform as the feed sets it
  rs::Transform fedLocal(int i) const
  {
    const fv::Node& n     = graph().nodes[size_t(i)];
    const double    scale = i == root() ? kRootScale : 1;
    rs::Transform   t;
    t.translation = glm::vec3(float(n.position.x), float(n.position.y), float(n.position.z));
    t.rotation = glm::quat(float(n.quaternion.w), float(n.quaternion.x), float(n.quaternion.y), float(n.quaternion.z));
    t.scale    = glm::vec3(float(n.scale.x * scale), float(n.scale.y * scale), float(n.scale.z * scale));
    return t;
  }

  // what the figure's nodes were fed as before an op
  struct Fed
  {
    std::vector<rs::EntityId>  entity;
    std::vector<rs::Transform> local;
    std::vector<bool>          visible;
  };
  Fed fed() const
  {
    Fed out;
    for(size_t i = 0; i < graph().nodes.size(); i++)
    {
      out.entity.push_back(m_feed->entityOf(int(i)));
      out.local.push_back(fedLocal(int(i)));
      out.visible.push_back(graph().nodes[i].visible);
    }
    return out;
  }

  // Feeds the figure. A new node's entity is new; a node whose transform or visibility changed
  // moved or was hidden with what is under it; the meshes the figure moved had their vertices set.
  void sync(Expected& want, const Fed& before, std::span<const int> moved = {})
  {
    m_feed->sync(graph(), root(), moved);
    const std::vector<bool> on = reached();
    std::set<rs::EntityId>  made;
    for(size_t i = 0; i < on.size(); i++)
      if(on[i] && (i >= before.entity.size() || before.entity[i].isNone()))
      {
        const rs::EntityId e = m_feed->entityOf(int(i));
        made.insert(e);
        want.transforms.both(e);
        want.visibility.may.insert(e);
      }
    for(size_t i = 0; i < on.size() && i < before.entity.size(); i++)
    {
      if(!on[i] || before.entity[i].isNone())
        continue;
      const rs::EntityId e = m_feed->entityOf(int(i));
      if(!(fedLocal(int(i)) == before.local[i]))
        for(rs::EntityId x : subtree(e))
          made.count(x) ? (void)want.transforms.may.insert(x) : want.transforms.both(x);
      if(graph().nodes[i].visible != before.visible[i])
        for(rs::EntityId x : subtree(e))
          made.count(x) ? (void)want.visibility.may.insert(x) : want.visibility.both(x);
    }
    for(int mi : moved)
      if(!m_feed->meshOf(mi).isNone())
        want.meshes.both(m_feed->meshOf(mi));
  }

  std::string feedFigure(Expected& want)
  {
    m_feed = std::make_unique<fv::FigureFeed>(m_scene, feedOptions());
    m_fedEntities.clear();
    m_fedMaterials.clear();
    sync(want, {});
    want.structure = true;
    return "feed the figure";
  }

  // Puts on an item: a group under the root or a bone with a mesh node of its own, a material of its
  // own (half of them normal-mapped, so the mesh has tangents), half of them skinned to two bones; a
  // quarter of the time a node drawing a mesh an item brought instead (a DAZ instance)
  std::string wear(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    const Fed               before = fed();
    fv::SceneGraph&         g      = graph();
    const std::vector<bool> on     = reached();
    std::vector<int>        bones;
    for(size_t i = 0; i < g.nodes.size(); i++)
      if(on[i] && g.nodes[i].kind == fv::NodeKind::eBone)
        bones.push_back(int(i));
    const int   parent = bones.empty() || chance(2) ? root() : pick(bones);
    const int   index  = int(m_rng() % (g.nodes[size_t(parent)].children.size() + 2)) - 1;
    std::string what;
    if(!m_itemMeshes.empty() && chance(4))
    {
      fv::Node n{.name = "Instance " + std::to_string(g.nodes.size()), .kind = fv::NodeKind::eMesh, .mesh = pick(m_itemMeshes)};
      n.position.x = uniform(-50, 50);
      m_items.push_back(g.add(n, parent, index));
      what = "wear an instance of mesh " + std::to_string(n.mesh);
    }
    else
    {
      const int    material = int(g.materials.size());
      fv::Material mat;
      mat.name = "Item " + std::to_string(material);
      if(chance(2))
        mat.surface.normalMap = "/Runtime/Textures/Test/BoxDetail_NM.png";
      g.materials.push_back(mat);
      fv::Mesh m;
      m.name = "item " + std::to_string(g.meshes.size());
      m.geo  = std::make_shared<fv::mesh::BuiltMesh>(fv::mesh::build(itemGrid(1 + int(m_rng() % 3), uniform(-20, 20))));
      m.materials = {material};
      if(bones.size() >= 2 && chance(2))
      {
        const int b0 = pick(bones), b1 = pick(bones);
        m.bones = {b0, b1};
        for(size_t v = 0; v < m.geo->position.size() / 3; v++)
        {
          m.skin.skinIndex.insert(m.skin.skinIndex.end(), {uint16_t(v % 2), 0, 0, 0});
          m.skin.skinWeight.insert(m.skin.skinWeight.end(), {1.f, 0.f, 0.f, 0.f});
        }
        m.boneInverses = {fv::inverse(g.world(b0)), fv::inverse(g.world(b1))};
      }
      const int mesh = int(g.meshes.size());
      g.meshes.push_back(std::move(m));
      const int item = g.add(fv::Node{.name = "Item " + std::to_string(mesh)}, parent, index);
      g.add(fv::Node{.name = "Item mesh " + std::to_string(mesh), .kind = fv::NodeKind::eMesh, .mesh = mesh}, item);
      m_items.push_back(item);
      m_itemMeshes.push_back(mesh);
      what = "wear mesh " + std::to_string(mesh) + (g.meshes.back().skinned() ? ", skinned" : "");
    }
    want.structure = true;
    sync(want, before);
    return what;
  }

  std::string takeOff(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    if(m_items.empty())
      return "take off: none";
    const Fed    before = fed();
    const size_t i      = m_rng() % m_items.size();
    const int    node   = m_items[i];
    m_items.erase(m_items.begin() + ptrdiff_t(i));
    graph().nodes[size_t(node)].removed = true;
    want.structure                      = true;
    sync(want, before);
    return "take off node " + std::to_string(node);
  }

  // The body's hidden polygons: none, the whole of its first group (a graft hiding a surface), or
  // a handful
  std::string graft(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    const Fed                                   before = fed();
    fv::mesh::BuiltMesh&                        geo    = *graph().meshes[0].geo;
    std::map<std::string, std::vector<int32_t>> hidden;
    const uint32_t                              kind = m_rng() % 3;
    if(kind == 1)
    {
      std::set<int32_t>      polys;
      const fv::mesh::Group& first = geo.groups[0];
      for(uint32_t t = first.start / 3; t < (first.start + first.count) / 3; t++)
        polys.insert(int32_t(geo.triPoly[t]));
      hidden["w1"].assign(polys.begin(), polys.end());
    }
    else if(kind == 2)
    {
      std::set<int32_t> polys;
      for(int n = 0; n < 6; n++)
        polys.insert(int32_t(geo.triPoly[m_rng() % geo.triPoly.size()]));
      hidden["w2"].assign(polys.begin(), polys.end());
    }
    if(hidden == geo.hiddenPolys)
      return "graft: as it was";
    geo.hiddenPolys = std::move(hidden);
    want.structure  = true;
    sync(want, before);
    return std::string("graft: ") + (kind == 0 ? "none" : kind == 1 ? "a group whole" : "a handful");
  }

  std::string hideFigureNode(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    const Fed               before = fed();
    const std::vector<bool> on     = reached();
    std::vector<int>        nodes;
    for(size_t i = 0; i < on.size(); i++)
      if(on[i])
        nodes.push_back(int(i));
    const int i                      = pick(nodes);
    graph().nodes[size_t(i)].visible = !graph().nodes[size_t(i)].visible;
    sync(want, before);
    return (graph().nodes[size_t(i)].visible ? "show node " : "hide node ") + std::to_string(i);
  }

  std::string editFigureMaterial(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    const Fed    before = fed();
    const size_t i      = m_rng() % graph().materials.size();
    // a value the lowered payload does not have yet (an edit to the value it has changes nothing)
    const double had = m_scene.material(m_feed->materialOf(int(i)))->payload.pbrMetallicRoughness.roughnessFactor;
    std::optional<float>& roughness = graph().materials[i].edits.roughness;
    float                 value     = 0.05f * float(1 + m_rng() % 19);
    if(value == float(had))
      value += 0.025f;
    roughness = value;
    sync(want, before);
    want.materials.both(m_feed->materialOf(int(i)));
    return "figure material edit " + std::to_string(i);
  }

  // A body dial, which moves the body's vertices and the joint centres (the skin binds again)
  std::string dial(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    const Fed         before = fed();
    const std::string name   = chance(2) ? "Stretch" : "Taper";
    const double      value  = 0.25 * double(m_rng() % 5);
    m_figure->morphs->set(name, value);
    m_figure->morphs->settled();
    return figureMoved(want, before, name + " " + std::to_string(value));
  }

  std::string pose(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    const Fed                              before = fed();
    const std::vector<fv::pose::PoseBone>& bones  = m_figure->pose->bones();
    const fv::pose::PoseBone&              bone   = bones[m_rng() % bones.size()];
    static const char*                     axes[] = {"x", "y", "z"};
    const double                           value  = double(int(m_rng() % 81) - 40);
    m_figure->pose->setMany({{bone.name, fv::pose::ChannelKind::eRotation, axes[m_rng() % 3], value}});
    return figureMoved(want, before, "pose " + bone.name);
  }

  // fed after the figure moved: the meshes it moved, and every skin may have been bound again
  std::string figureMoved(Expected& want, const Fed& before, const std::string& what)
  {
    std::vector<int> moved;
    for(const auto& c : m_figure->drain().meshes)
      moved.push_back(c.mesh);
    sync(want, before, moved);
    for(size_t mi = 0; mi < graph().meshes.size(); mi++)
      if(!m_feed->skinOf(int(mi)).isNone())
        want.skins.may.insert(m_feed->skinOf(int(mi)));
    return what;
  }

  // Textures on or off (a cook that finished, a texture setting): any material may be set again,
  // and a mesh whose normal map came or went has tangents made or dropped (a new record)
  std::string toggleTextures(Expected& want)
  {
    if(!m_feed)
      return feedFigure(want);
    const Fed before = fed();
    m_textured       = !m_textured;
    m_feed->setOptions(buildOptions());
    sync(want, before);
    for(size_t i = 0; i < graph().materials.size(); i++)
      want.materials.may.insert(m_feed->materialOf(int(i)));
    want.structureMay = true;
    return m_textured ? "textures on" : "textures off";
  }

  //------------------------------------------------------------------------------------------------
  // Checks after every op

  void checkStore()
  {
    checkTree();
    checkReferences();
    checkFeatures();
    checkShadow();
    if(m_feed)
      checkFigure();
    checkBounds();
  }

  // Bounds that moved come with a new boundsRevision
  void checkBounds()
  {
    const std::array<glm::vec3, 2> bounds = m_scene.bounds();
    if(bounds != m_bounds)
      EXPECT_NE(m_scene.boundsRevision(), m_boundsRevision) << "the bounds moved under the same revision";
    m_bounds         = bounds;
    m_boundsRevision = m_scene.boundsRevision();
  }

  // Every entity once, under the parent it names, from the roots (so no cycle); its world matrix
  // and visibility those a walk from the roots makes
  void checkTree()
  {
    std::set<rs::EntityId>                                                        seen;
    const std::function<void(rs::EntityId, rs::EntityId, const glm::mat4&, bool)> walk =
        [&](rs::EntityId id, rs::EntityId parent, const glm::mat4& parentWorld, bool parentShown) {
          const rs::Entity* e = m_scene.entity(id);
          if(!e)
          {
            ADD_FAILURE() << "a gone entity " << handle(id) << " in the tree";
            return;
          }
          if(!seen.insert(id).second)
          {
            ADD_FAILURE() << "entity " << e->name << " reached twice";
            return;
          }
          EXPECT_EQ(e->parent, parent) << e->name;
          const glm::mat4 world = parentWorld * e->local.local();
          const bool      shown = parentShown && e->visible;
          EXPECT_EQ(m_scene.world(id), world) << e->name;
          EXPECT_EQ(m_scene.shown(id), shown) << e->name;
          for(rs::EntityId c : e->children)
            walk(c, id, world, shown);
        };
    for(rs::EntityId r : m_scene.roots())
      walk(r, {}, glm::mat4(1.f), true);
    EXPECT_EQ(seen.size(), m_scene.entityCount()) << "entities the roots do not reach";
  }

  // Each record's references counted from scratch: the instances, lights and cameras of every
  // entity, each material's texture table, the feed's own holds and the clips the props hold. A
  // count for a record that is gone is a dangling reference; a record nothing counts, a leak.
  void checkReferences()
  {
    std::map<rs::MeshId, uint32_t>     meshes;
    std::map<rs::MaterialId, uint32_t> materials;
    std::map<rs::TextureId, uint32_t>  textures;
    std::map<rs::SkinId, uint32_t>     skins;
    std::map<rs::LightId, uint32_t>    lights;
    std::map<rs::CameraId, uint32_t>   cameras;
    std::map<rs::ClipId, uint32_t>     clips;
    for(rs::EntityId id : m_scene.entitiesInOrder())
    {
      const rs::Entity& e = *m_scene.entity(id);
      if(e.mesh)
      {
        for(const rs::MeshPart& part : e.mesh->parts)
        {
          meshes[part.mesh]++;
          for(rs::MaterialId m : part.materials)
            materials[m]++;
        }
        if(!e.mesh->skin.isNone())
          skins[e.mesh->skin]++;
      }
      if(!e.light.isNone())
        lights[e.light]++;
      if(!e.camera.isNone())
        cameras[e.camera]++;
    }
    if(m_feed)
    {
      for(size_t mi = 0; mi < graph().meshes.size(); mi++)
      {
        if(!m_feed->meshOf(int(mi)).isNone())
          meshes[m_feed->meshOf(int(mi))]++;
        if(!m_feed->skinOf(int(mi)).isNone())
          skins[m_feed->skinOf(int(mi))]++;
      }
      for(size_t i = 0; i < graph().materials.size(); i++)
        if(!m_feed->materialOf(int(i)).isNone())
          materials[m_feed->materialOf(int(i))]++;
      for(const rs::TextureRef& ref : *m_feed->textures())
        textures[ref.texture]++;
    }
    for(const Prop& p : m_props)
      for(rs::ClipId c : p.clips)
        clips[c]++;
    for(const auto& [id, n] : materials)
      if(const rs::MaterialRecord* r = m_scene.material(id); r && r->textures)
        for(const rs::TextureRef& ref : *r->textures)
          if(!ref.texture.isNone())
            textures[ref.texture]++;

    const auto expectRefs = [&](const auto& counts, size_t live, const char* what) {
      for(const auto& [id, n] : counts)
        EXPECT_EQ(m_scene.refs(id), n) << what << " " << handle(id) << (m_scene.refs(id) ? "" : " is gone");
      EXPECT_EQ(live, counts.size()) << what << " records nothing references";
    };
    expectRefs(meshes, m_scene.meshCount(), "mesh");
    expectRefs(materials, m_scene.materialCount(), "material");
    expectRefs(textures, m_scene.textureCount(), "texture");
    expectRefs(skins, m_scene.skinCount(), "skin");
    expectRefs(lights, m_scene.lightCount(), "light");
    expectRefs(cameras, m_scene.cameraCount(), "camera");
    expectRefs(clips, m_scene.clipCount(), "clip");
  }

  // features() is what the materials submeshes draw with need, and what the detector finds in the
  // export
  void checkFeatures()
  {
    SceneFeatureSet counted;
    for(rs::EntityId id : m_scene.entitiesInOrder())
      if(const rs::Entity& e = *m_scene.entity(id); e.mesh)
        for(const rs::MeshPart& part : e.mesh->parts)
          if(const rs::MeshRecord* r = m_scene.mesh(part.mesh))
            for(const rs::Submesh& s : r->submeshes)
              if(s.material < part.materials.size())
                if(const rs::MaterialRecord* m = m_scene.material(part.materials[s.material]))
                  counted |= nvvkgltf::detectMaterialFeatures(m->payload);
    const SceneFeatureSet kept = materialBits(m_scene.features());
    EXPECT_TRUE(kept == materialBits(counted)) << "kept {" << kept.toString() << "}, counted {" << counted.toString() << "}";
    const SceneFeatureSet exported = materialBits(nvvkgltf::detectSceneFeatures(rs::exportGltf(m_scene)));
    EXPECT_TRUE(kept == exported) << "kept {" << kept.toString() << "}, export {" << exported.toString() << "}";
  }

  // The props' entities and records are what the ops made them, under the handles they were made
  // with; every other entity is the figure's
  void checkShadow()
  {
    size_t figure = 0;
    for(rs::EntityId id : m_scene.entitiesInOrder())
    {
      const rs::Entity& e  = *m_scene.entity(id);
      const auto        it = m_shadow.entities.find(id);
      if(it == m_shadow.entities.end())
      {
        if(m_feed && e.owner.prop == kFigureProp && m_feed->entityOf(e.owner.local) == id)
          figure++;
        else
          ADD_FAILURE() << "entity " << e.name << " (" << tag(e.owner) << ") is no prop's and not the figure's";
        continue;
      }
      const Shadow::EntityRecord& s = it->second;
      EXPECT_TRUE(e.name == s.name && e.owner == s.owner) << e.name;
      EXPECT_EQ(e.parent, s.parent) << e.name;
      EXPECT_TRUE(e.local == s.local) << e.name;
      EXPECT_EQ(e.visible, s.visible) << e.name;
      EXPECT_EQ(e.light, s.light) << e.name;
      EXPECT_EQ(e.camera, s.camera) << e.name;
      EXPECT_EQ(e.mesh.has_value(), s.mesh.has_value()) << e.name;
      if(!e.mesh || !s.mesh)
        continue;
      EXPECT_TRUE(sameInstance(*e.mesh, *s.mesh)) << e.name;
      for(const rs::MeshPart& part : e.mesh->parts)
      {
        const rs::MeshRecord* r = m_scene.mesh(part.mesh);
        if(r && m_shadow.meshes.count(part.mesh))
          EXPECT_EQ(meshDifference(*r, m_shadow.meshes.at(part.mesh)), "") << e.name;
        for(rs::MaterialId m : part.materials)
          if(const rs::MaterialRecord* mr = m_scene.material(m); mr && m_shadow.materials.count(m))
          {
            const rs::MaterialRecord& sm = m_shadow.materials.at(m);
            EXPECT_TRUE(mr->owner == sm.owner && mr->payload == sm.payload && mr->textures == sm.textures)
                << e.name << " material " << handle(m);
            if(mr->textures)
              for(const rs::TextureRef& ref : *mr->textures)
                if(const rs::TextureRecord* t = m_scene.texture(ref.texture); t && m_shadow.textures.count(ref.texture))
                  EXPECT_TRUE(sameTexture(*t, m_shadow.textures.at(ref.texture))) << e.name << " texture " << handle(ref.texture);
          }
      }
      if(const rs::SkinRecord* skin = m_scene.skin(e.mesh->skin); skin && m_shadow.skins.count(e.mesh->skin))
        EXPECT_TRUE(sameSkinRecord(*skin, m_shadow.skins.at(e.mesh->skin))) << e.name;
    }
    EXPECT_EQ(m_scene.entityCount(), m_shadow.entities.size() + figure) << "a prop entity the store lost";
    for(const auto& [id, light] : m_shadow.lights)
      if(const rs::LightRecord* l = m_scene.light(id))
        EXPECT_TRUE(l->owner == light.owner && l->payload == light.payload && l->shown == light.shown
                    && l->skySun == light.skySun)
            << "light " << handle(id);
    for(const auto& [id, camera] : m_shadow.cameras)
      if(const rs::CameraRecord* c = m_scene.camera(id))
        EXPECT_TRUE(c->owner == camera.owner && c->payload == camera.payload) << "camera " << handle(id);
    for(const auto& [id, clip] : m_shadow.clips)
    {
      const rs::ClipRecord* c = m_scene.clip(id);
      EXPECT_TRUE(c && c->name == clip.name && c->owner == clip.owner && c->samplers == clip.samplers && c->channels == clip.channels)
          << "clip " << handle(id);
    }
  }

  // checkFeed is clean; each reached node's entity has the node's transform and visibility and is
  // the one it had (a material too), every other node has none
  void checkFigure()
  {
    EXPECT_EQ(fv::checkFeed(graph(), root(), *m_feed, m_scene, buildOptions()), std::vector<std::string>{});
    const std::vector<bool> on = reached();
    m_fedEntities.resize(on.size());
    for(size_t i = 0; i < on.size(); i++)
    {
      const rs::EntityId e = m_feed->entityOf(int(i));
      if(!on[i])
      {
        EXPECT_TRUE(e.isNone()) << "taken-off node " << i;
        m_fedEntities[i] = {};
        continue;
      }
      const rs::Entity* x = m_scene.entity(e);
      if(!x)
      {
        ADD_FAILURE() << "node " << i << " has no entity";
        continue;
      }
      EXPECT_TRUE(x->local == fedLocal(int(i))) << "node " << i;
      EXPECT_EQ(x->visible, graph().nodes[i].visible) << "node " << i;
      if(!m_fedEntities[i].isNone())
        EXPECT_EQ(e, m_fedEntities[i]) << "node " << i << "'s entity changed";
      m_fedEntities[i] = e;
    }
    for(size_t i = 0; i < graph().materials.size(); i++)
    {
      const rs::MaterialId id = m_feed->materialOf(int(i));
      if(i < m_fedMaterials.size())
        EXPECT_EQ(id, m_fedMaterials[i]) << "material " << i << " changed record";
      else
        m_fedMaterials.push_back(id);
    }
  }

  // The next takeChanges() reports what the op touched (dead handles aside), and the features
  // flag says whether features() moved since the last one
  void expectChanges(const Expected& want)
  {
    const rs::SceneChanges got = m_scene.takeChanges();
    if(!want.structureMay)
      EXPECT_EQ(got.structure, want.structure) << "structure";
    const auto expectSet = [&](const auto& set, const auto& w, const char* what) {
      for(const auto& id : w.must)
        if(live(m_scene, id) && !set.count(id))
          ADD_FAILURE() << what << " misses " << handle(id);
      for(const auto& id : set)
        if(!w.must.count(id) && !w.may.count(id))
          ADD_FAILURE() << what << " names " << handle(id) << ", which the op did not touch";
    };
    expectSet(got.transforms, want.transforms, "transforms");
    expectSet(got.visibility, want.visibility, "visibility");
    expectSet(got.morphWeights, want.morphWeights, "morph weights");
    expectSet(got.meshes, want.meshes, "meshes");
    expectSet(got.materials, want.materials, "materials");
    expectSet(got.lights, want.lights, "lights");
    expectSet(got.skins, want.skins, "skins");
    const SceneFeatureSet now = m_scene.features();
    EXPECT_EQ(got.features, now != m_featuresTaken) << "features";
    m_featuresTaken = now;
  }

  //------------------------------------------------------------------------------------------------
  // Incremental equals fresh

  // The props replayed into `out` from the shadow, in one go: records made as an entity first
  // names them, entities parents first, skins and clips over the new entities
  void replay(rs::RetainedScene& out) const
  {
    std::map<rs::EntityId, rs::EntityId>                                       entities;
    std::map<rs::MeshId, rs::MeshId>                                           meshes;
    std::map<rs::MaterialId, rs::MaterialId>                                   materials;
    std::map<rs::TextureId, rs::TextureId>                                     textures;
    std::map<rs::SkinId, rs::SkinId>                                           skins;
    std::map<rs::LightId, rs::LightId>                                         lights;
    std::map<rs::CameraId, rs::CameraId>                                       cameras;
    std::map<const rs::TextureTable*, std::shared_ptr<const rs::TextureTable>> tables;

    const auto entityOf = [&](rs::EntityId id) {
      const auto it = entities.find(id);
      return it == entities.end() ? rs::EntityId{} : it->second;
    };
    const auto texture = [&](rs::TextureId id) {
      const auto s = m_shadow.textures.find(id);
      if(s == m_shadow.textures.end())
        return rs::TextureId{};
      auto [it, fresh] = textures.try_emplace(id);
      if(fresh)
        it->second = out.addTexture(s->second);
      return it->second;
    };
    const auto table = [&](const std::shared_ptr<const rs::TextureTable>& t) -> std::shared_ptr<const rs::TextureTable> {
      if(!t)
        return nullptr;
      auto [it, fresh] = tables.try_emplace(t.get());
      if(fresh)
      {
        auto made = std::make_shared<rs::TextureTable>();
        for(const rs::TextureRef& ref : *t)
          made->push_back({texture(ref.texture), ref.sampler});
        it->second = std::move(made);
      }
      return it->second;
    };
    const auto material = [&](rs::MaterialId id) {
      auto [it, fresh] = materials.try_emplace(id);
      if(fresh)
      {
        const rs::MaterialRecord& r = m_shadow.materials.at(id);
        it->second = out.addMaterial({.owner = r.owner, .payload = r.payload, .textures = table(r.textures)});
      }
      return it->second;
    };
    const auto mesh = [&](rs::MeshId id) {
      auto [it, fresh] = meshes.try_emplace(id);
      if(fresh)
        it->second = out.addMesh(m_shadow.meshes.at(id));
      return it->second;
    };
    const auto skin = [&](rs::SkinId id) {
      if(id.isNone())
        return rs::SkinId{};
      auto [it, fresh] = skins.try_emplace(id);
      if(fresh)
      {
        rs::SkinRecord r = m_shadow.skins.at(id);
        for(rs::EntityId& j : r.joints)
          j = entityOf(j);
        r.skeleton = entityOf(r.skeleton);
        it->second = out.addSkin(std::move(r));
      }
      return it->second;
    };

    const std::function<rs::EntityId(rs::EntityId)> entity = [&](rs::EntityId id) {
      if(const auto it = entities.find(id); it != entities.end())
        return it->second;
      const Shadow::EntityRecord& e      = m_shadow.entities.at(id);
      const rs::EntityId          parent = e.parent.isNone() ? rs::EntityId{} : entity(e.parent);
      const rs::EntityId          made   = out.createEntity(e.name, e.owner, parent);
      out.setLocal(made, e.local);
      out.setVisible(made, e.visible);
      return entities[id] = made;
    };
    for(const auto& [id, e] : m_shadow.entities)
      entity(id);
    for(const auto& [id, e] : m_shadow.entities)
    {
      const rs::EntityId made = entities.at(id);
      if(e.mesh)
      {
        rs::MeshInstance instance = *e.mesh;
        for(rs::MeshPart& part : instance.parts)
        {
          part.mesh = mesh(part.mesh);
          for(rs::MaterialId& m : part.materials)
            m = material(m);
        }
        instance.skin = skin(instance.skin);
        out.setMeshInstance(made, std::move(instance));
      }
      if(!e.light.isNone())
      {
        auto [it, fresh] = lights.try_emplace(e.light);
        if(fresh)
          it->second = out.addLight(m_shadow.lights.at(e.light));
        out.setEntityLight(made, it->second);
      }
      if(!e.camera.isNone())
      {
        auto [it, fresh] = cameras.try_emplace(e.camera);
        if(fresh)
          it->second = out.addCamera(m_shadow.cameras.at(e.camera));
        out.setEntityCamera(made, it->second);
      }
    }
    // the clips are held for as long as `out` lives
    for(const Prop& p : m_props)
      for(rs::ClipId c : p.clips)
      {
        rs::ClipRecord clip = m_shadow.clips.at(c);
        for(rs::ClipChannel& channel : clip.channels)
          channel.target = entityOf(channel.target);
        out.addClip(std::move(clip));
      }
    // what the entities hold stays; the replay's own references go
    for(const auto& [from, to] : meshes)
      out.release(to);
    for(const auto& [from, to] : materials)
      out.release(to);
    for(const auto& [from, to] : textures)
      out.release(to);
    for(const auto& [from, to] : skins)
      out.release(to);
    for(const auto& [from, to] : lights)
      out.release(to);
    for(const auto& [from, to] : cameras)
      out.release(to);
  }

  // A texture by what it holds and how it is sampled: the same id in both stores for the same content
  int textureId(const rs::RetainedScene& s, const rs::MaterialRecord& m, int index)
  {
    if(index < 0 || !m.textures || index >= int(m.textures->size()))
      return -1;
    const rs::TextureRef&    ref = (*m.textures)[size_t(index)];
    const rs::TextureRecord* t   = s.texture(ref.texture);
    std::string              key = t ? t->name + '|' + t->uri + '|' + t->mimeType + '|' + t->sourceExtension + '|'
                                           + std::string(t->bytes.begin(), t->bytes.end()) + '|'
                                           + std::string(t->pixels.begin(), t->pixels.end()) :
                                       "none";
    key += '|' + std::to_string(ref.sampler.magFilter) + ',' + std::to_string(ref.sampler.minFilter) + ','
           + std::to_string(ref.sampler.wrapS) + ',' + std::to_string(ref.sampler.wrapT);
    return m_textureNames.try_emplace(key, int(m_textureNames.size())).first->second;
  }

  // A payload with its texture indices as the ids of what they name, so stores whose tables differ
  // compare
  tinygltf::Material canonical(const rs::RetainedScene& s, const rs::MaterialRecord& m)
  {
    tinygltf::Material out = m.payload;
    for(tinygltf::TextureInfo* t :
        {&out.pbrMetallicRoughness.baseColorTexture, &out.pbrMetallicRoughness.metallicRoughnessTexture, &out.emissiveTexture})
      if(t->index >= 0)
        t->index = textureId(s, m, t->index);
    if(out.normalTexture.index >= 0)
      out.normalTexture.index = textureId(s, m, out.normalTexture.index);
    if(out.occlusionTexture.index >= 0)
      out.occlusionTexture.index = textureId(s, m, out.occlusionTexture.index);
    const std::function<void(tinygltf::Value&)> walk = [&](tinygltf::Value& v) {
      if(v.IsObject())
        for(auto& [key, child] : v.Get<tinygltf::Value::Object>())
          if(key == "index" && child.IsInt())
            child = tinygltf::Value(textureId(s, m, child.Get<int>()));
          else
            walk(child);
      else if(v.IsArray())
        for(auto& child : v.Get<tinygltf::Value::Array>())
          walk(child);
    };
    for(auto& [name, ext] : out.extensions)
      walk(ext);
    return out;
  }

  std::string ownerOf(const rs::RetainedScene& s, rs::EntityId id) const
  {
    const rs::Entity* e = s.entity(id);
    return e ? tag(e->owner) : "none";
  }

  // A material by its owner and its canonical payload
  bool sameMaterial(const rs::RetainedScene& a, rs::MaterialId ia, const rs::RetainedScene& b, rs::MaterialId ib)
  {
    const rs::MaterialRecord* x = a.material(ia);
    const rs::MaterialRecord* y = b.material(ib);
    if(!x || !y)
      return !x && !y;
    return x->owner == y->owner && canonical(a, *x) == canonical(b, *y);
  }

  // A skin by its inverse binds and its joints' owner tags
  bool sameSkin(const rs::RetainedScene& a, rs::SkinId ia, const rs::RetainedScene& b, rs::SkinId ib) const
  {
    const rs::SkinRecord* x = a.skin(ia);
    const rs::SkinRecord* y = b.skin(ib);
    if(!x || !y)
      return !x && !y;
    if(x->name != y->name || x->inverseBind != y->inverseBind || x->joints.size() != y->joints.size()
       || ownerOf(a, x->skeleton) != ownerOf(b, y->skeleton))
      return false;
    for(size_t j = 0; j < x->joints.size(); j++)
      if(ownerOf(a, x->joints[j]) != ownerOf(b, y->joints[j]))
        return false;
    return true;
  }

  // The two stores hold the same scene: their entities, what each draws and carries, the draw and
  // light items, the clips, as multisets keyed by owner tag with records compared by content. Every
  // entity in the fuzz has an owner tag of its own, so the entities pair one to one.
  void expectSameScene(rs::RetainedScene& a, rs::RetainedScene& b)
  {
    const auto byOwner = [](const rs::RetainedScene& s) {
      std::map<rs::OwnerTag, rs::EntityId> out;
      for(rs::EntityId id : s.entitiesInOrder())
        if(!out.emplace(s.entity(id)->owner, id).second)
          ADD_FAILURE() << "owner tag " << tag(s.entity(id)->owner) << " held twice";
      return out;
    };
    const std::map<rs::OwnerTag, rs::EntityId> ea = byOwner(a), eb = byOwner(b);
    EXPECT_EQ(ea.size(), eb.size()) << "entities";
    for(const auto& [owner, ia] : ea)
    {
      const auto it = eb.find(owner);
      if(it == eb.end())
      {
        ADD_FAILURE() << "entity " << tag(owner) << " is not in the fresh store";
        continue;
      }
      const rs::EntityId ib   = it->second;
      const rs::Entity&  x    = *a.entity(ia);
      const rs::Entity&  y    = *b.entity(ib);
      const std::string  what = "entity " + x.name + " (" + tag(owner) + ")";
      EXPECT_EQ(x.name, y.name) << what;
      EXPECT_TRUE(x.local == y.local) << what;
      EXPECT_EQ(x.visible, y.visible) << what;
      EXPECT_EQ(a.world(ia), b.world(ib)) << what;
      EXPECT_EQ(a.shown(ia), b.shown(ib)) << what;
      EXPECT_EQ(ownerOf(a, x.parent), ownerOf(b, y.parent)) << what;
      EXPECT_EQ(x.mesh.has_value(), y.mesh.has_value()) << what;
      if(x.mesh && y.mesh)
      {
        EXPECT_EQ(x.mesh->morphWeights, y.mesh->morphWeights) << what;
        EXPECT_EQ(x.mesh->copies, y.mesh->copies) << what;
        EXPECT_TRUE(sameSkin(a, x.mesh->skin, b, y.mesh->skin)) << what << " skin";
        EXPECT_EQ(a.jointMatrices(ia), b.jointMatrices(ib)) << what;
        EXPECT_EQ(x.mesh->parts.size(), y.mesh->parts.size()) << what;
        for(size_t p = 0; p < x.mesh->parts.size() && p < y.mesh->parts.size(); p++)
        {
          const rs::MeshPart& px = x.mesh->parts[p];
          const rs::MeshPart& py = y.mesh->parts[p];
          EXPECT_EQ(meshDifference(*a.mesh(px.mesh), *b.mesh(py.mesh)), "") << what << " part " << p;
          EXPECT_EQ(px.materials.size(), py.materials.size()) << what << " part " << p;
          for(size_t m = 0; m < px.materials.size() && m < py.materials.size(); m++)
            EXPECT_TRUE(sameMaterial(a, px.materials[m], b, py.materials[m])) << what << " part " << p << " material " << m;
        }
      }
      const rs::LightRecord* lx = a.light(x.light);
      const rs::LightRecord* ly = b.light(y.light);
      EXPECT_EQ(lx != nullptr, ly != nullptr) << what << " light";
      if(lx && ly)
        EXPECT_TRUE(lx->owner == ly->owner && lx->payload == ly->payload && lx->shown == ly->shown && lx->skySun == ly->skySun)
            << what << " light";
      const rs::CameraRecord* cx = a.camera(x.camera);
      const rs::CameraRecord* cy = b.camera(y.camera);
      EXPECT_EQ(cx != nullptr, cy != nullptr) << what << " camera";
      if(cx && cy)
        EXPECT_TRUE(cx->owner == cy->owner && cx->payload == cy->payload) << what << " camera";
    }

    // draw items: (entity, mesh, submesh, copy) -> world, shown, material
    using ItemKey = std::tuple<rs::OwnerTag, rs::OwnerTag, int32_t, uint32_t>;
    struct ItemValue
    {
      glm::mat4   world;
      bool        shown;
      std::string material;
    };
    const auto items = [&](rs::RetainedScene& s) {
      std::multimap<ItemKey, ItemValue> out;
      for(const rs::DrawItem& d : s.drawItems())
      {
        const rs::MeshRecord&     m = *s.mesh(d.mesh);
        const rs::MaterialRecord* r = s.material(d.material);
        out.emplace(ItemKey{s.entity(d.entity)->owner, m.owner, m.submeshes[d.submesh].order, d.copy},
                    ItemValue{d.world, d.shown, r ? tag(r->owner) : "none"});
      }
      return out;
    };
    expectSameMultiset(
        items(a), items(b),
        [](const ItemValue& x, const ItemValue& y) {
          return x.world == y.world && x.shown == y.shown && x.material == y.material;
        },
        [](const ItemKey& k) {
          return tag(std::get<0>(k)) + " mesh " + tag(std::get<1>(k)) + " submesh " + std::to_string(std::get<2>(k))
                 + " copy " + std::to_string(std::get<3>(k));
        },
        "draw item");

    // light items: entity -> world, shown, the light's owner
    struct LightValue
    {
      glm::mat4    world;
      bool         shown;
      rs::OwnerTag light;
    };
    const auto lightItems = [&](rs::RetainedScene& s) {
      std::multimap<rs::OwnerTag, LightValue> out;
      for(const rs::LightItem& l : s.lightItems())
        out.emplace(s.entity(l.entity)->owner, LightValue{l.world, l.shown, s.light(l.light)->owner});
      return out;
    };
    expectSameMultiset(
        lightItems(a), lightItems(b),
        [](const LightValue& x, const LightValue& y) {
          return x.world == y.world && x.shown == y.shown && x.light == y.light;
        },
        [](const rs::OwnerTag& k) { return tag(k); }, "light item");

    // clips: owner -> name, keys, channels on their entities' owner tags
    const auto clips = [&](const rs::RetainedScene& s) {
      std::multimap<rs::OwnerTag, std::pair<const rs::RetainedScene*, rs::ClipId>> out;
      for(rs::ClipId c : s.clips())
        out.emplace(s.clip(c)->owner, std::pair{&s, c});
      return out;
    };
    expectSameMultiset(
        clips(a), clips(b),
        [&](const auto& x, const auto& y) {
          const rs::ClipRecord& cx = *x.first->clip(x.second);
          const rs::ClipRecord& cy = *y.first->clip(y.second);
          if(cx.name != cy.name || cx.samplers != cy.samplers || cx.channels.size() != cy.channels.size())
            return false;
          for(size_t c = 0; c < cx.channels.size(); c++)
            if(cx.channels[c].path != cy.channels[c].path || cx.channels[c].sampler != cy.channels[c].sampler
               || ownerOf(*x.first, cx.channels[c].target) != ownerOf(*y.first, cy.channels[c].target))
              return false;
          return true;
        },
        [](const rs::OwnerTag& k) { return tag(k); }, "clip");

    // the same records; the fresh feed's texture list holds only what its materials ask for now
    EXPECT_EQ(a.meshCount(), b.meshCount());
    EXPECT_EQ(a.materialCount(), b.materialCount());
    EXPECT_GE(a.textureCount(), b.textureCount());
    EXPECT_EQ(a.skinCount(), b.skinCount());
    EXPECT_EQ(a.lightCount(), b.lightCount());
    EXPECT_EQ(a.cameraCount(), b.cameraCount());
    EXPECT_EQ(a.clipCount(), b.clipCount());
    EXPECT_TRUE(materialBits(a.features()) == materialBits(b.features()));
    EXPECT_EQ(a.bounds(), b.bounds());
  }

  void compareWithFresh()
  {
    SCOPED_TRACE("against a fresh store");
    rs::RetainedScene               fresh;
    std::unique_ptr<fv::FigureFeed> feed;
    if(m_feed)
    {
      feed = std::make_unique<fv::FigureFeed>(fresh, feedOptions());
      feed->sync(graph(), root());
    }
    replay(fresh);
    expectSameScene(m_scene, fresh);
  }

  uint32_t                        m_seed;
  std::mt19937                    m_rng;
  std::unique_ptr<fv::Figure>     m_figure;
  rs::RetainedScene               m_scene;
  bool                            m_textured = true;
  std::unique_ptr<fv::FigureFeed> m_feed;
  std::vector<rs::EntityId>       m_fedEntities;   // per figure node: its entity at the last check
  std::vector<rs::MaterialId>     m_fedMaterials;  // per figure material
  std::vector<int>                m_items;         // the item nodes worn, not taken off
  std::vector<int>                m_itemMeshes;    // the meshes items brought
  std::vector<Prop>               m_props;
  uint32_t                        m_nextProp    = kFigureProp + 1;
  uint32_t                        m_nextTexture = 0;
  Shadow                          m_shadow;
  SceneFeatureSet                 m_featuresTaken;
  std::array<glm::vec3, 2>        m_bounds;
  uint64_t                        m_boundsRevision = 0;
  std::map<std::string, int>      m_textureNames;  // texture content -> id, across both stores
};

}  // namespace

// The headline at the store: 10^4 random ops in the default run, every one checked, the store
// compared with a fresh one every few ops. A seed that fails goes into kRegressionSeeds.
TEST(RetainedFuzz, IncrementalEqualsFresh)
{
  std::vector<uint32_t> seeds;
  for(uint32_t s = 1; s <= kSequences; s++)
    seeds.push_back(s);
  seeds.insert(seeds.end(), kRegressionSeeds.begin(), kRegressionSeeds.end());
  const auto start = std::chrono::steady_clock::now();
  for(uint32_t seed : seeds)
  {
    Fuzz fuzz(seed);
    fuzz.run(kSteps, kCompareEvery);
    if(HasFailure())
      return;
  }
  std::printf("[ fuzz     ] %zu sequences of %d ops in %.1f s\n", seeds.size(), kSteps,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}
