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

// Clips and skinning on the retained scene (oracle 6 of daz-viewer docs/retained-scene-plan.md): a
// model imported into the store and loaded into nvvkgltf::Scene, each clip played by a ClipPlayer
// and by AnimationSystem::updateAnimation at the same times (on keys, between them, before the
// first, after the last, wrapped and held at the end), gives every node the same local TRS, world
// matrix and morph weights, and every skinned mesh the same skinned positions and normals, bit for
// bit: the arithmetic is the same, in the same order.
//
// computeSkinning keeps its joint matrices to itself and exposes the skinned vertices, so the joint
// matrices are held to it through those: RetainedScene::jointMatrices put through computeSkinning's
// own vertex loop. Both use inverse(world[mesh node]) * world[joint] * IBM; there is no conversion.

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include "scene_clip_player.hpp"
#include "scene_import_gltf.hpp"
#include "tinygltf_utils.hpp"

namespace {

int add(tinygltf::Model& m, const void* data, size_t bytes, int type, size_t count, int component = TINYGLTF_COMPONENT_TYPE_FLOAT)
{
  return tinygltf::utils::appendAccessor(m, 0, data, bytes, component, type, count);
}

template <class T>
int add(tinygltf::Model& m, const std::vector<T>& v, int type)
{
  return add(m, v.data(), v.size() * sizeof(T), type, v.size() * sizeof(T) / sizeof(float) / size_t(tinygltf::GetNumComponentsInType(uint32_t(type))));
}

struct AnimationBuilder
{
  tinygltf::Model&    m;
  tinygltf::Animation animation;

  void channel(int node, const char* path, const char* interpolation, const std::vector<float>& times, const std::vector<float>& values)
  {
    static const std::map<std::string, int> types = {{"translation", TINYGLTF_TYPE_VEC3}, {"scale", TINYGLTF_TYPE_VEC3},
                                                     {"rotation", TINYGLTF_TYPE_VEC4}, {"weights", TINYGLTF_TYPE_SCALAR}};
    tinygltf::AnimationSampler s;
    s.input = add(m, times, TINYGLTF_TYPE_SCALAR);
    m.accessors[size_t(s.input)].minValues = {*std::min_element(times.begin(), times.end())};
    m.accessors[size_t(s.input)].maxValues = {*std::max_element(times.begin(), times.end())};
    s.output        = add(m, values, types.at(path));
    s.interpolation = interpolation;
    tinygltf::AnimationChannel c;
    c.sampler     = int(animation.samplers.size());
    c.target_node = node;
    c.target_path = path;
    animation.samplers.push_back(s);
    animation.channels.push_back(c);
  }
};

// A model with every path under every interpolation: a hierarchy (a node animated with a static
// child), a morphed mesh drawn three times with its weights animated three ways, a skinned mesh
// under an offset node with an animated skeleton under its own parent, a channel with one key,
// one with a zero-length interval, channels whose keys cover part of the clip, a rotation that
// crosses the quaternion sign, and a second animation.
tinygltf::Model fixture()
{
  tinygltf::Model m;
  m.buffers.emplace_back();
  const float h = 0.7071067811865476f;

  // a quad with two morph targets, and a strip of six vertices skinned to two joints
  const std::vector<glm::vec3> quad    = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const std::vector<glm::vec3> up      = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  const std::vector<glm::vec3> lift    = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0.5f}, {0, 0, 0.5f}};
  const std::vector<glm::vec3> spread  = {{-0.2f, 0, 0}, {0.2f, 0, 0}, {0.2f, 0, 0}, {-0.2f, 0, 0}};
  const std::vector<glm::vec3> tilt    = {{0, 0.1f, 0}, {0, 0.1f, 0}, {0, -0.1f, 0}, {0, -0.1f, 0}};
  const std::vector<uint32_t>  quadIdx = {0, 1, 2, 0, 2, 3};
  tinygltf::Mesh               morph;
  morph.name = "morph";
  tinygltf::Primitive mp;
  mp.attributes = {{"POSITION", add(m, quad, TINYGLTF_TYPE_VEC3)}, {"NORMAL", add(m, up, TINYGLTF_TYPE_VEC3)}};
  mp.indices    = add(m, quadIdx.data(), quadIdx.size() * 4, TINYGLTF_TYPE_SCALAR, quadIdx.size(), TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
  mp.targets    = {{{"POSITION", add(m, lift, TINYGLTF_TYPE_VEC3)}, {"NORMAL", add(m, tilt, TINYGLTF_TYPE_VEC3)}},
                   {{"POSITION", add(m, spread, TINYGLTF_TYPE_VEC3)}}};
  mp.material   = 0;
  morph.primitives = {mp};
  morph.weights    = {0.25, 0};

  const std::vector<glm::vec3>  strip     = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 2, 0}, {1, 2, 0}};
  const std::vector<glm::vec3>  stripN    = {{0, 0, 1}, {0, 0, 1}, {0.6f, 0, 0.8f}, {0, 0.6f, 0.8f}, {0, 0, 1}, {0, 0, 1}};
  const std::vector<glm::u16vec4> joints  = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 2, 0, 0}};
  const std::vector<glm::vec4>  weights   = {{1, 0, 0, 0}, {1, 0, 0, 0}, {0.5f, 0.5f, 0, 0}, {0.75f, 0.25f, 0, 0}, {1, 0, 0, 0}, {0.6f, 0.4f, 0, 0}};
  const std::vector<uint32_t>   stripIdx  = {0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4};
  tinygltf::Mesh                skinned;
  skinned.name = "skinned";
  tinygltf::Primitive sp;
  sp.attributes = {{"POSITION", add(m, strip, TINYGLTF_TYPE_VEC3)},
                   {"NORMAL", add(m, stripN, TINYGLTF_TYPE_VEC3)},
                   {"JOINTS_0", add(m, joints.data(), joints.size() * 8, TINYGLTF_TYPE_VEC4, joints.size(), TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)},
                   {"WEIGHTS_0", add(m, weights, TINYGLTF_TYPE_VEC4)}};
  sp.indices  = add(m, stripIdx.data(), stripIdx.size() * 4, TINYGLTF_TYPE_SCALAR, stripIdx.size(), TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
  sp.material = 0;
  skinned.primitives = {sp};
  m.meshes           = {morph, skinned};
  m.materials.emplace_back().name = "plain";

  enum
  {
    kRoot,
    kLinear,
    kLeaf,
    kStep,
    kCubic,
    kMorphLinear,
    kMorphStep,
    kMorphCubic,
    kSkinned,
    kArmature,
    kBone0,
    kBone1,
    kShort,
    kCount
  };
  m.nodes.resize(kCount);
  const char* names[] = {"root", "linear", "leaf", "step", "cubic", "morph linear", "morph step", "morph cubic", "skinned", "armature", "bone0", "bone1", "short"};
  for(int i = 0; i < kCount; i++)
    m.nodes[size_t(i)].name = names[i];
  m.nodes[kRoot].translation = {0, 1, 0};
  m.nodes[kRoot].children    = {kLinear, kStep, kCubic, kMorphLinear, kMorphStep, kMorphCubic, kSkinned, kArmature, kShort};
  m.nodes[kLinear].children  = {kLeaf};
  m.nodes[kLeaf].translation = {1, 0, 0};
  m.nodes[kLeaf].scale       = {0.5, 0.5, 0.5};
  for(int n : {kMorphLinear, kMorphStep, kMorphCubic})
    m.nodes[size_t(n)].mesh = 0;
  m.nodes[kSkinned].mesh        = 1;
  m.nodes[kSkinned].skin        = 0;
  m.nodes[kSkinned].translation = {0, 0, 2};
  m.nodes[kSkinned].rotation    = {0, 0, h, h};
  m.nodes[kArmature].translation = {0.5, 0, 0};
  m.nodes[kArmature].scale       = {1, 2, 1};
  m.nodes[kArmature].children    = {kBone0};
  m.nodes[kBone0].children       = {kBone1};
  m.nodes[kBone1].translation    = {0, 1, 0};
  m.scenes.emplace_back().nodes = {kRoot};
  m.defaultScene                = 0;

  tinygltf::Skin skin;
  skin.joints   = {kBone0, kBone1, kLeaf};
  skin.skeleton = kArmature;
  const std::vector<glm::mat4> ibm = {glm::mat4(1.f), glm::translate(glm::mat4(1.f), glm::vec3(0, -1, 0)),
                                      glm::scale(glm::translate(glm::mat4(1.f), glm::vec3(-1, -1, 0)), glm::vec3(2.f))};
  skin.inverseBindMatrices = add(m, ibm, TINYGLTF_TYPE_MAT4);
  m.skins                  = {skin};

  AnimationBuilder all{m};
  all.animation.name = "all";
  all.channel(kLinear, "translation", "LINEAR", {0, 1, 2}, {0, 0, 0, 1, 0.5f, 0, 2, -1, 0.25f});
  // from (0, 0, sin 40°, cos 40°) to a rotation whose quaternion has the other sign: slerp takes the short way
  all.channel(kLinear, "rotation", "LINEAR", {0, 1, 2}, {0, 0, 0.3420201f, 0.9396926f, 0, -0.5f, -0.5f, -h, h, 0, 0, h});
  all.channel(kLinear, "scale", "LINEAR", {0.5f, 1.5f}, {1, 1, 1, 2, 0.5f, 1});
  all.channel(kStep, "translation", "STEP", {0, 0.5f, 1.5f}, {0, 0, 0, 1, 0, 0, 1, 1, 0});
  all.channel(kStep, "rotation", "STEP", {0, 1, 2}, {0, 0, 0, 1, 0, h, 0, h, h, 0, 0, h});
  all.channel(kStep, "scale", "STEP", {0, 1, 2}, {1, 1, 1, 2, 2, 2, 0.5f, 0.5f, 0.5f});
  // cubic splines: (in-tangent, value, out-tangent) per key
  all.channel(kCubic, "translation", "CUBICSPLINE", {0, 1, 2.5f},
              {0, 0, 0, 0, 0, 0, 1, 2, 0,  /**/ -1, 0, 1, 1, 1, 1, 0.5f, 0, -1,  /**/ 0, 3, 0, 2, 0, 2, 0, 0, 0});
  all.channel(kCubic, "rotation", "CUBICSPLINE", {0, 1, 2.5f},
              {0, 0, 0, 0, 0, 0, 0, 1, 0.2f, 0, 0, 0,  /**/ 0.1f, 0.3f, 0, 0, 0, h, 0, h, 0, 0, 0.4f, 0,  /**/ 0, 0, 0, 0, h, 0, 0, h, 0, 0, 0, 0});
  all.channel(kCubic, "scale", "CUBICSPLINE", {0, 1, 2.5f},
              {0, 0, 0, 1, 1, 1, 0.5f, 0, 0,  /**/ 0, 0.5f, 0, 2, 1, 1, 0, 0, 0.5f,  /**/ 0, 0, 0, 1, 1, 2, 0, 0, 0});
  all.channel(kMorphLinear, "weights", "LINEAR", {0, 1, 2}, {0, 0, 1, 0.5f, 0.25f, 1});
  all.channel(kMorphStep, "weights", "STEP", {0, 1, 2}, {0, 0, 1, 0.5f, 0.25f, 1});
  all.channel(kMorphCubic, "weights", "CUBICSPLINE", {0, 1, 2},
              {0, 0, 0, 0, 1, -1,  /**/ 0.5f, 0, 1, 0.5f, 0, 0,  /**/ 0, 0, 0.25f, 1, 0, 0});
  all.channel(kBone0, "rotation", "LINEAR", {0, 2}, {0, 0, 0, 1, 0, 0, h, h});
  all.channel(kBone1, "translation", "CUBICSPLINE", {0, 1, 2},
              {0, 0, 0, 0, 1, 0, 1, 0, 0,  /**/ 0, 1, 0, 0.5f, 1.5f, 0, 0, 0, 0,  /**/ 0, 0, 0, 0, 1, 0, 0, 0, 0});
  all.channel(kBone1, "scale", "STEP", {0, 1}, {1, 1, 1, 1.5f, 1, 1});
  // one key: never applied; a zero-length last interval: its first key
  all.channel(kShort, "translation", "LINEAR", {1}, {5, 5, 5});
  all.channel(kShort, "scale", "LINEAR", {0, 1, 2, 2}, {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4});

  AnimationBuilder second{m};
  second.animation.name = "second";
  second.channel(kBone0, "translation", "LINEAR", {0, 3}, {0, 0, 0, 1, 0, 0});
  second.channel(kRoot, "rotation", "LINEAR", {0.5f, 3}, {0, 0, 0, 1, 0, 1, 0, 0});
  m.animations = {all.animation, second.animation};
  return m;
}

glm::quat legacyRotation(const tinygltf::Node& n)
{
  return n.rotation.size() == 4 ? glm::quat(float(n.rotation[3]), float(n.rotation[0]), float(n.rotation[1]), float(n.rotation[2])) :
                                  glm::quat(1.f, 0.f, 0.f, 0.f);
}
glm::vec3 legacyVec3(const std::vector<double>& v, float fallback)
{
  return v.size() == 3 ? glm::vec3(float(v[0]), float(v[1]), float(v[2])) : glm::vec3(fallback);
}

// computeSkinning's vertex loop over a retained mesh and joint matrices
void skin(const rs::MeshRecord& m, const std::vector<glm::mat4>& jointMatrices, std::vector<glm::vec3>& positions, std::vector<glm::vec3>& normals)
{
  std::vector<glm::mat3> normalMats;
  for(const glm::mat4& j : jointMatrices)
    normalMats.push_back(glm::transpose(glm::inverse(glm::mat3(j))));
  const size_t numJoints  = jointMatrices.size();
  const bool   hasNormals = !m.normals.empty();
  positions.assign(m.weights.size(), glm::vec3(0.f));
  normals.assign(hasNormals ? m.weights.size() : 0, glm::vec3(0.f));
  for(size_t v = 0; v < m.weights.size(); v++)
  {
    const glm::vec4&  w = m.weights[v];
    const glm::ivec4  j = glm::ivec4(m.joints[v]);
    glm::vec3         skinnedPos(0.0f);
    glm::vec3         skinnedNrm(0.0f);
    for(int i = 0; i < 4; ++i)
    {
      const float jointWeight = w[i];
      if(jointWeight > 0.0f)
      {
        const int        jointIndex = j[i];
        const bool       validJoint = jointIndex >= 0 && static_cast<size_t>(jointIndex) < numJoints;
        const glm::mat4& jMat       = validJoint ? jointMatrices[jointIndex] : glm::identity<glm::mat4>();
        const glm::mat3& nMat       = validJoint ? normalMats[jointIndex] : glm::identity<glm::mat3>();
        skinnedPos += jointWeight * glm::vec3(jMat * glm::vec4(m.positions[v], 1.0f));
        if(hasNormals)
          skinnedNrm += jointWeight * (nMat * m.normals[v]);
      }
    }
    positions[v] = skinnedPos;
    if(hasNormals)
      normals[v] = glm::normalize(skinnedNrm);
  }
}

template <class T>
bool sameBits(const std::vector<T>& a, const std::vector<T>& b)
{
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// The times a clip is sampled at: before its first key, on every key, a third and a half of the way
// between keys, and after its last
std::vector<float> sampleTimes(const rs::ClipRecord& clip)
{
  std::vector<float> keys;
  for(const rs::ClipSampler& s : clip.samplers)
    keys.insert(keys.end(), s.inputs.begin(), s.inputs.end());
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  std::vector<float> times{keys.front() - 1.f};
  for(size_t k = 0; k < keys.size(); k++)
  {
    times.push_back(keys[k]);
    if(k + 1 < keys.size())
    {
      times.push_back(keys[k] + (keys[k + 1] - keys[k]) / 3.f);
      times.push_back(keys[k] + (keys[k + 1] - keys[k]) * 0.5f);
    }
  }
  times.push_back(keys.back() + 1.f);
  return times;
}

struct Counts
{
  int times = 0, nodes = 0, skinned = 0;
};

// A model's clips played on both scenes, compared after every step; the first difference fails the
// model and stops it
void expectAnimationParity(const tinygltf::Model& model, const std::string& what, Counts& counts)
{
  nvvkgltf::Scene legacy;
  legacy.takeModel(tinygltf::Model(model));
  if(!legacy.valid())
  {
    ADD_FAILURE() << what << ": the legacy scene did not load";
    return;
  }
  nvvkgltf::AnimationSystem& anim = legacy.animation();
  const tinygltf::Model&     lm   = legacy.getModel();
  rs::RetainedScene          retained;
  const rs::ImportResult     imported = rs::importGltf(retained, model, {.prop = 1});
  ASSERT_EQ(anim.getNumAnimations(), int(model.animations.size())) << what;
  ASSERT_EQ(imported.clips.size(), model.animations.size()) << what;

  // each skin task's retained vertex set: the one of its glTF mesh with its base positions
  std::vector<const rs::MeshRecord*> skinMeshes;
  for(const nvvkgltf::SkinTask& task : anim.getSkinTasks())
  {
    const rs::MeshRecord* found = nullptr;
    for(rs::MeshId id : imported.meshes[size_t(legacy.getRenderPrimitive(size_t(task.renderPrimID)).meshID)])
      if(sameBits(retained.mesh(id)->positions, task.basePositions))
        found = retained.mesh(id);
    ASSERT_NE(found, nullptr) << what << ": no vertex set for a skin task";
    skinMeshes.push_back(found);
  }

  for(size_t a = 0; a < model.animations.size(); a++)
  {
    const rs::ClipRecord* clip = retained.clip(imported.clips[a]);
    ASSERT_NE(clip, nullptr) << what << ", animation " << a;
    rs::ClipPlayer           player(retained, imported.clips[a]);
    nvvkgltf::AnimationInfo& info = anim.getAnimationInfo(int(a));
    EXPECT_EQ(player.start(), info.start) << what;
    EXPECT_EQ(player.end(), info.end) << what;

    const auto compare = [&](const std::string& when) -> bool {
      SCOPED_TRACE(what + ", animation " + std::to_string(a) + ", " + when);
      counts.times++;
      for(rs::EntityId id : retained.entitiesInOrder())
      {
        const int             n    = retained.entity(id)->owner.local;
        const tinygltf::Node& node = lm.nodes[size_t(n)];
        const rs::Entity&     e    = *retained.entity(id);
        counts.nodes++;
        if(node.matrix.size() == 16)
        {
          EXPECT_TRUE(e.local.matrix && *e.local.matrix == glm::mat4(glm::make_mat4(node.matrix.data()))) << node.name;
        }
        else
        {
          EXPECT_EQ(e.local.translation, legacyVec3(node.translation, 0.f)) << node.name;
          EXPECT_EQ(e.local.scale, legacyVec3(node.scale, 1.f)) << node.name;
          const glm::quat r = legacyRotation(node);
          EXPECT_EQ(r, e.local.rotation) << node.name << ": rotation (" << r.x << ", " << r.y << ", " << r.z << ", " << r.w
                                         << ") against (" << e.local.rotation.x << ", " << e.local.rotation.y << ", "
                                         << e.local.rotation.z << ", " << e.local.rotation.w << ")";
        }
        if(retained.world(id) != legacy.getNodesWorldMatrices()[size_t(n)])
        {
          ADD_FAILURE() << node.name << ": world matrix";
          return false;
        }
        if(e.mesh)
        {
          std::vector<float> lw;
          for(double w : node.weights)
            lw.push_back(float(w));
          if(lw != e.mesh->morphWeights)
          {
            ADD_FAILURE() << node.name << ": morph weights";
            return false;
          }
        }
        if(::testing::Test::HasFailure())
          return false;
      }
      for(size_t t = 0; t < anim.getSkinTasks().size(); t++)
      {
        const nvvkgltf::SkinTask& task = anim.getSkinTasks()[t];
        const tinygltf::Skin&     s    = lm.skins[size_t(task.skinID)];
        const std::vector<glm::mat4> joints = retained.jointMatrices(imported.entities[size_t(task.refNodeID)]);
        EXPECT_EQ(joints.size(), s.joints.size());
        std::vector<glm::vec3> positions, normals;
        skin(*skinMeshes[t], joints, positions, normals);
        const nvvkgltf::SkinningResult& result = anim.getSkinningResult(t);
        EXPECT_TRUE(sameBits(positions, result.positions)) << "skinned positions, task " << t;
        EXPECT_TRUE(sameBits(normals, result.normals)) << "skinned normals, task " << t;
        counts.skinned++;
      }
      return !::testing::Test::HasFailure();
    };
    const auto step = [&]() {
      (void)anim.updateAnimation(uint32_t(a));
      legacy.updateNodeWorldMatrices();
      if(anim.hasSkinning())
        anim.computeSkinning();
      player.apply(retained);
    };

    for(float t : sampleTimes(*clip))
    {
      info.currentTime = t;
      player.setTime(t);
      step();
      if(!compare("t = " + std::to_string(t)))
        return;
    }
    // wrapped: steps that are not a whole fraction of the clip, from its start
    EXPECT_EQ(player.reset(), info.reset());
    const float dt = std::max(info.end - info.start, 0.1f) * 0.37f;
    for(int i = 0; i < 12; i++)
    {
      info.incrementTime(dt, true);
      player.advance(dt, true);
      ASSERT_EQ(player.time(), info.currentTime) << what;
      step();
      if(!compare("looped step " + std::to_string(i)))
        return;
    }
    // and held at the end
    for(int i = 0; i < 4; i++)
    {
      info.incrementTime(dt, false);
      player.advance(dt, false);
      ASSERT_EQ(player.time(), info.currentTime) << what;
      step();
      if(!compare("held step " + std::to_string(i)))
        return;
    }
  }
}

}  // namespace

TEST(RetainedAnimation, ClipsPlayAsTheLegacySceneDoes)
{
  Counts counts;
  expectAnimationParity(fixture(), "fixture", counts);
  EXPECT_GT(counts.skinned, 0);
}

// What glTF says a STEP channel holds, on both scenes: each key's value up to the next key, the last
// key's value exactly on it, and a rotation key read as the quaternion it is (AnimationSystem once
// gave the next-to-last key there, and took a rotation key's x, y and z as Euler angles)
TEST(RetainedAnimation, StepHoldsEachKeyAndTheLastOnIt)
{
  const tinygltf::Model model = fixture();
  const float           h     = 0.7071067811865476f;
  int                   step = -1, morphStep = -1;
  for(size_t n = 0; n < model.nodes.size(); n++)
  {
    if(model.nodes[n].name == "step")
      step = int(n);
    if(model.nodes[n].name == "morph step")
      morphStep = int(n);
  }
  ASSERT_GE(step, 0);
  ASSERT_GE(morphStep, 0);

  nvvkgltf::Scene legacy;
  legacy.takeModel(tinygltf::Model(model));
  ASSERT_TRUE(legacy.valid());
  nvvkgltf::AnimationSystem& anim = legacy.animation();
  const tinygltf::Model&     lm   = legacy.getModel();
  rs::RetainedScene          retained;
  const rs::ImportResult     imported = rs::importGltf(retained, model, {.prop = 1});
  rs::ClipPlayer             player(retained, imported.clips[0]);  // "all"
  const rs::Entity&          stepEntity      = *retained.entity(imported.entities[size_t(step)]);
  const rs::Entity&          morphStepEntity = *retained.entity(imported.entities[size_t(morphStep)]);

  struct Expected
  {
    float              time;
    glm::quat          rotation;     // keys at 0, 1, 2
    glm::vec3          translation;  // keys at 0, 0.5, 1.5
    std::vector<float> weights;      // keys at 0, 1, 2
  };
  for(const Expected& x : {Expected{0.5f, glm::quat(1, 0, 0, 0), {1, 0, 0}, {0, 0}},
                           Expected{1.f, glm::quat(h, 0, h, 0), {1, 0, 0}, {1, 0.5f}},
                           Expected{1.5f, glm::quat(h, 0, h, 0), {1, 1, 0}, {1, 0.5f}},
                           Expected{2.f, glm::quat(h, h, 0, 0), {1, 1, 0}, {0.25f, 1}}})
  {
    SCOPED_TRACE("t = " + std::to_string(x.time));
    anim.getAnimationInfo(0).currentTime = x.time;
    (void)anim.updateAnimation(0);
    player.setTime(x.time);
    player.apply(retained);

    EXPECT_EQ(legacyRotation(lm.nodes[size_t(step)]), x.rotation);
    EXPECT_EQ(stepEntity.local.rotation, x.rotation);
    if(x.time <= 1.5f)  // the translation's last key; past it the channel leaves the node alone
    {
      EXPECT_EQ(legacyVec3(lm.nodes[size_t(step)].translation, 0.f), x.translation);
      EXPECT_EQ(stepEntity.local.translation, x.translation);
    }
    std::vector<float> lw;
    for(double w : lm.nodes[size_t(morphStep)].weights)
      lw.push_back(float(w));
    EXPECT_EQ(lw, x.weights);
    EXPECT_EQ(morphStepEntity.mesh->morphWeights, x.weights);
  }
}

// The Khronos samples with animations, skins and morphs, when present.
TEST(RetainedAnimation, SampleAssetsPlayAsTheLegacySceneDoes)
{
  namespace fs = std::filesystem;
  const fs::path models = fs::path(GLTF_SAMPLE_ASSETS_PATH) / "Models";
  int            tested = 0;
  Counts         counts;
  for(const char* file : {"AnimatedCube/glTF/AnimatedCube.gltf", "BoxAnimated/glTF/BoxAnimated.gltf",
                          "AnimatedMorphCube/glTF/AnimatedMorphCube.gltf", "AnimatedMorphCube/glTF-Quantized/AnimatedMorphCube.gltf",
                          "InterpolationTest/glTF/InterpolationTest.gltf", "RiggedSimple/glTF/RiggedSimple.gltf",
                          "RiggedFigure/glTF/RiggedFigure.gltf", "CesiumMan/glTF/CesiumMan.gltf", "Fox/glTF/Fox.gltf",
                          "SimpleSkin/glTF/SimpleSkin.gltf", "SimpleMorph/glTF/SimpleMorph.gltf",
                          "AnimatedTriangle/glTF/AnimatedTriangle.gltf", "RecursiveSkeletons/glTF/RecursiveSkeletons.gltf"})
  {
    std::error_code ec;
    const fs::path  path = models / file;
    if(!fs::exists(path, ec))
      continue;
    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        err, warn;
    ASSERT_TRUE(loader.LoadASCIIFromFile(&model, &err, &warn, path.string())) << file << ": " << err;
    expectAnimationParity(model, file, counts);
    tested++;
  }
  if(tested == 0)
    GTEST_SKIP() << "no sample assets";
  std::printf("played %d files: %d times, %d node comparisons, %d skinned meshes\n", tested, counts.times, counts.nodes,
              counts.skinned);
}

// Playing writes through the store's edits, so takeChanges reports it: the entities moved (and what
// hangs under them) and the morph weights, never a mesh's vertices.
TEST(RetainedAnimation, PlayingReportsItsEdits)
{
  rs::RetainedScene      scene;
  const rs::ImportResult r = rs::importGltf(scene, fixture());
  (void)scene.takeChanges();
  rs::ClipPlayer player(scene, r.clips[0]);
  player.setTime(0.75f);
  EXPECT_TRUE(player.apply(scene));
  const rs::SceneChanges changes = scene.takeChanges();
  EXPECT_FALSE(changes.structure);
  EXPECT_TRUE(changes.meshes.empty());
  EXPECT_TRUE(changes.transforms.count(r.entities[1]));  // linear
  EXPECT_TRUE(changes.transforms.count(r.entities[2]));  // under it
  EXPECT_TRUE(changes.transforms.count(r.entities[11]));  // bone1
  EXPECT_EQ(changes.morphWeights, (std::set<rs::EntityId>{r.entities[5], r.entities[6], r.entities[7]}));
  EXPECT_FALSE(changes.transforms.count(r.entities[0]));  // the root is not in this clip
  // before every key nothing is written, and nothing is reported
  player.setTime(-1.f);
  EXPECT_FALSE(player.apply(scene));
  EXPECT_TRUE(scene.takeChanges().empty());
}

// A channel whose entity was destroyed is skipped; the others still play. A clip released for the
// last time is gone, and its player writes nothing.
TEST(RetainedAnimation, AGoneEntityOrClipIsSkipped)
{
  rs::RetainedScene      scene;
  const rs::ImportResult r = rs::importGltf(scene, fixture());
  scene.destroyEntity(r.entities[1]);  // linear and its child
  scene.destroyEntity(r.entities[5]);  // a morphed one
  rs::ClipPlayer player(scene, r.clips[0]);
  player.setTime(1.25f);
  EXPECT_TRUE(player.apply(scene));
  EXPECT_NE(scene.entity(r.entities[4])->local, rs::Transform{});  // cubic still moves

  EXPECT_EQ(scene.clipCount(), 2u);
  scene.retain(r.clips[0]);
  scene.release(r.clips[0]);
  EXPECT_NE(scene.clip(r.clips[0]), nullptr);  // the import's reference holds it
  scene.release(r.clips[0]);
  EXPECT_EQ(scene.clip(r.clips[0]), nullptr);
  EXPECT_EQ(scene.clipCount(), 1u);
  EXPECT_FALSE(player.apply(scene));
  // a new clip in the freed slot is not the old one
  const rs::ClipId again = scene.addClip({.name = "again"});
  EXPECT_NE(again, r.clips[0]);
  EXPECT_FALSE(player.apply(scene));
}

// KHR_animation_pointer channels are not clips (D23): they are left out with a notice, an animation
// of nothing else is no clip, and the rest of the import is as it was.
TEST(RetainedAnimation, PointerChannelsAreLeftOut)
{
  tinygltf::Model m = fixture();
  AnimationBuilder pointer{m};
  pointer.channel(1, "translation", "LINEAR", {0, 1}, {0, 0, 0, 1, 1, 1});
  pointer.channel(1, "rotation", "LINEAR", {0, 1}, {0, 0, 0, 1, 0, 0, 0, 1});
  pointer.animation.channels[1].target_path = "pointer";
  pointer.animation.channels[1].target_node = -1;
  AnimationBuilder only{m};
  only.channel(1, "rotation", "LINEAR", {0, 1}, {0, 0, 0, 1, 0, 0, 0, 1});
  only.animation.channels[0].target_path = "pointer";
  m.animations.push_back(pointer.animation);
  m.animations.push_back(only.animation);
  m.extensionsUsed.push_back(KHR_ANIMATION_POINTER);

  rs::RetainedScene      scene;
  const rs::ImportResult r = rs::importGltf(scene, m);
  ASSERT_EQ(r.clips.size(), 4u);
  const rs::ClipRecord* kept = scene.clip(r.clips[2]);
  ASSERT_NE(kept, nullptr);
  ASSERT_EQ(kept->channels.size(), 1u);
  EXPECT_EQ(kept->channels[0].path, rs::ClipPath::eTranslation);
  EXPECT_EQ(kept->samplers.size(), 1u);  // the pointer's sampler is not kept
  EXPECT_TRUE(r.clips[3].isNone());
  EXPECT_EQ(scene.clipCount(), 3u);
  EXPECT_FALSE(r.notices.empty());
  EXPECT_EQ(scene.entityCount(), m.nodes.size());
  EXPECT_EQ(scene.clip(r.clips[0])->channels.size(), m.animations[0].channels.size());
}

// The joint matrices: the mesh entity's inverse world, the joint's world and its inverse bind
// matrix; identity for a joint that is gone, none for an entity without a skin.
TEST(RetainedAnimation, JointMatrices)
{
  rs::RetainedScene      scene;
  const rs::ImportResult r       = rs::importGltf(scene, fixture());
  const rs::EntityId     skinned = r.entities[8];
  const rs::SkinRecord*  s       = scene.skin(scene.entity(skinned)->mesh->skin);
  ASSERT_NE(s, nullptr);
  std::vector<glm::mat4> joints = scene.jointMatrices(skinned);
  ASSERT_EQ(joints.size(), 3u);
  for(size_t j = 0; j < joints.size(); j++)
    EXPECT_EQ(joints[j], glm::inverse(scene.world(skinned)) * scene.world(s->joints[j]) * s->inverseBind[j]);
  EXPECT_TRUE(scene.jointMatrices(r.entities[5]).empty());  // morphed, not skinned
  EXPECT_TRUE(scene.jointMatrices(r.entities[0]).empty());  // draws nothing
  scene.destroyEntity(r.entities[1]);  // and with it the leaf, the third joint
  joints = scene.jointMatrices(skinned);
  ASSERT_EQ(joints.size(), 3u);
  EXPECT_EQ(joints[2], glm::mat4(1.f));
}
