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

#ifdef USE_NVMCP

#include "mcp_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <tinygltf/json.hpp>  // nlohmann::json
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

#include "gltf_scene_animation.hpp"
#include "gltf_scene_editor.hpp"
#include "renderer.hpp"
#include "tinygltf_utils.hpp"
#include "ui_animation.hpp"

using Json = nlohmann::json;

namespace {

//--------------------------------------------------------------------------------------------------
// Id maps: host ids (extras.fvId) -> indices, rebuilt per scene load / structural change.

struct PrimRef
{
  int mesh         = -1;
  int primitive    = -1;
  int renderPrimID = -1;
};

struct RestPose
{
  struct NodeTrs
  {
    std::vector<double> translation, rotation, scale, matrix;
  };
  std::vector<NodeTrs>             nodes;
  std::vector<std::vector<double>> meshWeights;
  std::unordered_set<int>          animatedNodes;  // targets of animation channels: what "stop" restores
};

struct SceneIds
{
  const void* scene      = nullptr;
  uint64_t    generation = ~0ull;
  size_t      nodeCount = 0, materialCount = 0, primCount = 0;

  std::unordered_map<std::string, int>                  nodes;
  std::unordered_map<std::string, int>                  materials;
  std::unordered_map<std::string, std::vector<PrimRef>> prims;
  std::vector<std::vector<int>>                         renderPrimsOfMesh;  // mesh -> render prim ids
  std::unordered_set<int>                               joints;
  RestPose                                              rest;
};

std::string fvIdOf(const tinygltf::Value& extras)
{
  if(!extras.IsObject() || !extras.Has("fvId"))
    return {};
  const tinygltf::Value& v = extras.Get("fvId");
  return v.IsString() ? v.Get<std::string>() : std::string{};
}

void rebuildIds(SceneIds& ids, nvvkgltf::Scene& scene, uint64_t generation)
{
  const tinygltf::Model& model = scene.getModel();
  const bool             newLoad = ids.scene != &scene || ids.generation != generation;
  ids.scene                      = &scene;
  ids.generation                 = generation;
  ids.nodeCount                  = model.nodes.size();
  ids.materialCount              = model.materials.size();
  ids.primCount                  = scene.getRenderPrimitives().size();
  ids.nodes.clear();
  ids.materials.clear();
  ids.prims.clear();
  ids.joints.clear();
  ids.renderPrimsOfMesh.assign(model.meshes.size(), {});

  for(size_t i = 0; i < model.nodes.size(); i++)
    if(std::string id = fvIdOf(model.nodes[i].extras); !id.empty())
      ids.nodes.emplace(std::move(id), int(i));
  for(size_t i = 0; i < model.materials.size(); i++)
    if(std::string id = fvIdOf(model.materials[i].extras); !id.empty())
      ids.materials.emplace(std::move(id), int(i));

  const auto& renderPrims = scene.getRenderPrimitives();
  for(size_t rp = 0; rp < renderPrims.size(); rp++)
  {
    const auto& prim = renderPrims[rp];
    if(prim.meshID < 0 || prim.meshID >= int(model.meshes.size()) || !prim.pPrimitive)
      continue;
    ids.renderPrimsOfMesh[prim.meshID].push_back(int(rp));
    const auto& mesh = model.meshes[prim.meshID];
    int         p    = -1;
    for(size_t k = 0; k < mesh.primitives.size(); k++)
      if(&mesh.primitives[k] == prim.pPrimitive)
        p = int(k);
    if(std::string id = fvIdOf(prim.pPrimitive->extras); !id.empty())
      ids.prims[id].push_back({prim.meshID, p, int(rp)});
  }
  for(const auto& skin : model.skins)
    ids.joints.insert(skin.joints.begin(), skin.joints.end());

  // The rest pose is what the file says, captured before anything here or an animation changed it.
  // Nodes and meshes a merge appended (vk_gltf_add_scene) are captured as they arrive.
  if(!newLoad)
  {
    for(size_t i = ids.rest.nodes.size(); i < model.nodes.size(); i++)
    {
      const auto& n = model.nodes[i];
      ids.rest.nodes.push_back({n.translation, n.rotation, n.scale, n.matrix});
    }
    for(size_t m = ids.rest.meshWeights.size(); m < model.meshes.size(); m++)
      ids.rest.meshWeights.push_back(model.meshes[m].weights);
  }
  else
  {
    ids.rest.nodes.resize(model.nodes.size());
    for(size_t i = 0; i < model.nodes.size(); i++)
    {
      const auto& n      = model.nodes[i];
      ids.rest.nodes[i]  = {n.translation, n.rotation, n.scale, n.matrix};
    }
    ids.rest.meshWeights.resize(model.meshes.size());
    for(size_t m = 0; m < model.meshes.size(); m++)
      ids.rest.meshWeights[m] = model.meshes[m].weights;
    ids.rest.animatedNodes.clear();
    for(const auto& a : model.animations)
      for(const auto& ch : a.channels)
        if(ch.target_node >= 0)
          ids.rest.animatedNodes.insert(ch.target_node);
  }
}

//--------------------------------------------------------------------------------------------------
// Small JSON helpers. nvmcp does not enforce input schemas, so every read is checked.

struct OpError
{
  std::string message;
};

[[noreturn]] void fail(std::string message)
{
  throw OpError{std::move(message)};
}

template <size_t N>
std::vector<double> numbers(const Json& j, const char* key)
{
  if(!j.is_array() || j.size() != N)
    fail(std::string("'") + key + "' must be an array of " + std::to_string(N) + " numbers");
  std::vector<double> out(N);
  for(size_t i = 0; i < N; i++)
  {
    if(!j[i].is_number())
      fail(std::string("'") + key + "' must contain numbers");
    out[i] = j[i].get<double>();
  }
  return out;
}

double number(const Json& j, const char* key)
{
  if(!j.is_number())
    fail(std::string("'") + key + "' must be a number");
  return j.get<double>();
}

bool boolean(const Json& j, const char* key)
{
  if(!j.is_boolean())
    fail(std::string("'") + key + "' must be a boolean");
  return j.get<bool>();
}

int resolveIndex(const Json& op, const std::unordered_map<std::string, int>& byId, size_t count, const char* what)
{
  if(auto it = op.find("id"); it != op.end())
  {
    if(!it->is_string())
      fail("'id' must be a string");
    auto found = byId.find(it->get<std::string>());
    if(found == byId.end())
      fail(std::string("no ") + what + " with id '" + it->get<std::string>() + "'");
    return found->second;
  }
  if(auto it = op.find("index"); it != op.end())
  {
    if(!it->is_number_integer() || it->get<int64_t>() < 0 || it->get<int64_t>() >= int64_t(count))
      fail(std::string("'index' is not a valid ") + what + " index");
    return int(it->get<int64_t>());
  }
  fail("give 'id' or 'index'");
}

// Primitives addressed by {id} (a primitive id shared by all primitives of one host geometry) or
// {mesh, primitive?}.
std::vector<PrimRef> resolvePrims(const Json& op, const SceneIds& ids, const tinygltf::Model& model)
{
  if(auto it = op.find("id"); it != op.end())
  {
    if(!it->is_string())
      fail("'id' must be a string");
    auto found = ids.prims.find(it->get<std::string>());
    if(found == ids.prims.end())
      fail("no primitive with id '" + it->get<std::string>() + "'");
    return found->second;
  }
  auto it = op.find("mesh");
  if(it == op.end() || !it->is_number_integer() || it->get<int64_t>() < 0 || it->get<int64_t>() >= int64_t(model.meshes.size()))
    fail("give 'id' or a valid 'mesh' index");
  const int            mesh = int(it->get<int64_t>());
  int                  only = -1;
  std::vector<PrimRef> out;
  if(auto p = op.find("primitive"); p != op.end())
  {
    if(!p->is_number_integer())
      fail("'primitive' must be an integer");
    only = int(p->get<int64_t>());
  }
  for(int rp : ids.renderPrimsOfMesh[mesh])
    out.push_back({mesh, -1, rp});
  if(only >= 0)
  {
    // render primitives of a mesh are listed in primitive order
    if(only >= int(out.size()))
      fail("'primitive' is out of range");
    out = {out[only]};
  }
  return out;
}

//--------------------------------------------------------------------------------------------------
// base64 (RFC 4648, padding optional) into little-endian float32 xyz triples.

std::vector<uint8_t> decodeBase64(const Json& j, const char* key)
{
  if(!j.is_string())
    fail(std::string("'") + key + "' must be a base64 string");
  const std::string& s = j.get_ref<const std::string&>();
  static const auto  table = [] {
    std::array<int8_t, 256> t{};
    t.fill(-1);
    const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for(int i = 0; i < 64; i++)
      t[uint8_t(abc[i])] = int8_t(i);
    return t;
  }();
  std::vector<uint8_t> bytes;
  bytes.reserve(s.size() * 3 / 4);
  uint32_t acc  = 0;
  int      bits = 0;
  for(char c : s)
  {
    if(c == '=' || c == '\n' || c == '\r')
      continue;
    const int8_t v = table[uint8_t(c)];
    if(v < 0)
      fail(std::string("'") + key + "' is not valid base64");
    acc = (acc << 6) | uint32_t(v);
    bits += 6;
    if(bits >= 8)
    {
      bits -= 8;
      bytes.push_back(uint8_t((acc >> bits) & 0xFF));
    }
  }
  return bytes;
}

std::vector<glm::vec3> decodeVec3(const std::vector<uint8_t>& bytes, const char* key)
{
  if(bytes.size() % sizeof(glm::vec3) != 0)
    fail(std::string("'") + key + "' is not a whole number of float32 xyz triples");
  std::vector<glm::vec3> out(bytes.size() / sizeof(glm::vec3));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

std::vector<uint32_t> decodeU32(const std::vector<uint8_t>& bytes, const char* key)
{
  if(bytes.size() % sizeof(uint32_t) != 0)
    fail(std::string("'") + key + "' is not a whole number of uint32 values");
  std::vector<uint32_t> out(bytes.size() / sizeof(uint32_t));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

//--------------------------------------------------------------------------------------------------
// Ops. Each returns a JSON result and throws OpError on bad input.

struct Ctx
{
  GltfRenderer&    renderer;
  nvvkgltf::Scene& scene;
  SceneIds&        ids;
};

// An array field's bytes: a base64 string, or {"blob": id} naming a payload the host posted raw to
// the frame stream's /blob (blob_store.hpp), which skips base64 and the JSON parses of megabytes.
std::vector<uint8_t> bytesOf(Ctx& c, const Json& j, const char* key)
{
  if(!j.is_object())
    return decodeBase64(j, key);
  auto id = j.find("blob");
  if(id == j.end() || !id->is_number_unsigned())
    fail(std::string("'") + key + "' must be a base64 string or {\"blob\": id}");
  std::optional<std::string> blob = c.renderer.blobs().take(id->get<uint64_t>());
  if(!blob)
    fail(std::string("'") + key + "' names blob " + std::to_string(id->get<uint64_t>()) + ", which was not uploaded or was already used");
  return std::vector<uint8_t>(blob->begin(), blob->end());
}

void mergeTextureInfo(tinygltf::TextureInfo& info, const Json& j, size_t textureCount)
{
  if(!j.is_object())
    fail("texture info must be an object");
  if(auto it = j.find("index"); it != j.end())
  {
    if(!it->is_number_integer() || it->get<int64_t>() < -1 || it->get<int64_t>() >= int64_t(textureCount))
      fail("texture 'index' must be an existing texture or -1");
    info.index = int(it->get<int64_t>());
  }
  if(auto it = j.find("texCoord"); it != j.end())
    info.texCoord = int(number(*it, "texCoord"));
}

// Instance-flag inputs (SceneRtx::getInstanceFlag): a change needs the TLAS instance refreshed.
struct InstanceInputs
{
  std::string alphaMode;
  bool        doubleSided, transmission, volume, diffuseTransmission;
  bool        operator==(const InstanceInputs&) const = default;
};

InstanceInputs instanceInputs(const tinygltf::Material& m)
{
  return {m.alphaMode, m.doubleSided, tinygltf::utils::getTransmission(m).factor > 0.f,
          tinygltf::utils::getVolume(m).thicknessFactor > 0.f, tinygltf::utils::getDiffuseTransmission(m).diffuseTransmissionFactor > 0.f};
}

Json opSetMaterial(Ctx& c, const Json& op)
{
  tinygltf::Model& model = c.scene.getModel();
  const int        idx   = resolveIndex(op, c.ids.materials, model.materials.size(), "material");
  auto             it    = op.find("material");
  if(it == op.end() || !it->is_object())
    fail("'material' must be an object");
  const Json&         in      = *it;
  tinygltf::Material& mat     = c.scene.editor().getMaterialForEdit(idx);
  const auto          before  = instanceInputs(mat);
  const size_t        texCount = model.textures.size();

  if(auto pbr = in.find("pbrMetallicRoughness"); pbr != in.end())
  {
    if(!pbr->is_object())
      fail("'pbrMetallicRoughness' must be an object");
    auto& p = mat.pbrMetallicRoughness;
    if(auto v = pbr->find("baseColorFactor"); v != pbr->end())
      p.baseColorFactor = numbers<4>(*v, "baseColorFactor");
    if(auto v = pbr->find("metallicFactor"); v != pbr->end())
      p.metallicFactor = number(*v, "metallicFactor");
    if(auto v = pbr->find("roughnessFactor"); v != pbr->end())
      p.roughnessFactor = number(*v, "roughnessFactor");
    if(auto v = pbr->find("baseColorTexture"); v != pbr->end())
      mergeTextureInfo(p.baseColorTexture, *v, texCount);
    if(auto v = pbr->find("metallicRoughnessTexture"); v != pbr->end())
      mergeTextureInfo(p.metallicRoughnessTexture, *v, texCount);
  }
  if(auto v = in.find("emissiveFactor"); v != in.end())
    mat.emissiveFactor = numbers<3>(*v, "emissiveFactor");
  if(auto v = in.find("alphaMode"); v != in.end())
  {
    if(!v->is_string() || (*v != "OPAQUE" && *v != "MASK" && *v != "BLEND"))
      fail("'alphaMode' must be OPAQUE, MASK or BLEND");
    mat.alphaMode = v->get<std::string>();
  }
  if(auto v = in.find("alphaCutoff"); v != in.end())
    mat.alphaCutoff = number(*v, "alphaCutoff");
  if(auto v = in.find("doubleSided"); v != in.end())
    mat.doubleSided = boolean(*v, "doubleSided");
  if(auto v = in.find("normalTexture"); v != in.end())
  {
    if(!v->is_object())
      fail("'normalTexture' must be an object");
    if(auto s = v->find("scale"); s != v->end())
      mat.normalTexture.scale = number(*s, "scale");
    if(auto i = v->find("index"); i != v->end())
    {
      if(!i->is_number_integer() || i->get<int64_t>() < -1 || i->get<int64_t>() >= int64_t(texCount))
        fail("texture 'index' must be an existing texture or -1");
      mat.normalTexture.index = int(i->get<int64_t>());
    }
  }
  if(auto v = in.find("occlusionTexture"); v != in.end())
  {
    if(!v->is_object())
      fail("'occlusionTexture' must be an object");
    if(auto s = v->find("strength"); s != v->end())
      mat.occlusionTexture.strength = number(*s, "strength");
  }
  if(auto v = in.find("emissiveTexture"); v != in.end())
    mergeTextureInfo(mat.emissiveTexture, *v, texCount);

  // Extensions: per extension, a JSON merge patch (RFC 7386) over the current value, so a
  // factors-only update keeps the texture entries; null removes the extension.
  if(auto ext = in.find("extensions"); ext != in.end())
  {
    if(!ext->is_object())
      fail("'extensions' must be an object");
    for(auto e = ext->begin(); e != ext->end(); ++e)
    {
      if(e.value().is_null())
      {
        mat.extensions.erase(e.key());
        continue;
      }
      if(!e.value().is_object())
        fail("extension '" + e.key() + "' must be an object or null");
      auto found = mat.extensions.find(e.key());
      Json cur   = found != mat.extensions.end() ? tinygltf::utils::valueToJson(found->second) : Json::object();
      if(!cur.is_object())
        cur = Json::object();
      cur.merge_patch(e.value());
      mat.extensions[e.key()] = tinygltf::utils::valueFromJson(cur);
      if(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), e.key()) == model.extensionsUsed.end())
        model.extensionsUsed.push_back(e.key());
    }
  }

  c.scene.markMaterialDirty(idx);
  if(!(instanceInputs(mat) == before))
    c.scene.markRenderNodeRtxDirtyForMaterials({idx});
  return {{"material", idx}};
}

// The render nodes drawn with `skin`: their skinned geometry moves, so the TLAS needs their
// instances refit to the BLAS's new bounds.
void markSkinRenderNodes(Ctx& c, int skin)
{
  const auto& renderNodes = c.scene.getRenderNodes();
  for(size_t rn = 0; rn < renderNodes.size(); rn++)
    if(renderNodes[rn].skinID == skin)
      c.scene.markRenderNodeDirty(int(rn), /*forVk=*/false, /*forRtx=*/true);
}

// A transform on `node` moves every joint at or below it: the skins using any of them deform.
// Moving a bone marks no render node by itself (bones have no mesh), so without this the TLAS
// would keep the rest-pose boxes and rays would miss whatever was posed out of them.
void markSkinsBelow(Ctx& c, int node)
{
  const tinygltf::Model& model = c.scene.getModel();
  std::vector<bool>      below(model.nodes.size(), false);
  std::vector<int>       stack{node};
  while(!stack.empty())
  {
    const int n = stack.back();
    stack.pop_back();
    if(n < 0 || n >= int(below.size()) || below[n])
      continue;
    below[n] = true;
    stack.insert(stack.end(), model.nodes[n].children.begin(), model.nodes[n].children.end());
  }
  for(size_t s = 0; s < model.skins.size(); s++)
    for(int joint : model.skins[s].joints)
      if(joint >= 0 && joint < int(below.size()) && below[joint])
      {
        markSkinRenderNodes(c, int(s));
        break;
      }
}

Json opSetNode(Ctx& c, const Json& op)
{
  tinygltf::Model& model = c.scene.getModel();
  const int        idx   = resolveIndex(op, c.ids.nodes, model.nodes.size(), "node");
  auto&            editor = c.scene.editor();

  const bool hasT = op.contains("translation"), hasR = op.contains("rotation"), hasS = op.contains("scale");
  if(hasT || hasR || hasS)
  {
    glm::vec3 t{0.f}, s{1.f};
    glm::quat r{1.f, 0.f, 0.f, 0.f};
    tinygltf::utils::getNodeTRS(model.nodes[idx], t, r, s);
    if(hasT)
    {
      auto v = numbers<3>(op["translation"], "translation");
      t      = {float(v[0]), float(v[1]), float(v[2])};
    }
    if(hasR)
    {
      auto v = numbers<4>(op["rotation"], "rotation");  // glTF order: x, y, z, w
      r      = glm::normalize(glm::quat(float(v[3]), float(v[0]), float(v[1]), float(v[2])));
    }
    if(hasS)
    {
      auto v = numbers<3>(op["scale"], "scale");
      s      = {float(v[0]), float(v[1]), float(v[2])};
    }
    editor.setNodeTRS(idx, t, r, s);
    // Joints (or anything above them) moved: skinning and the skinned BLAS must be recomputed.
    if(c.scene.animation().hasSkinning())
    {
      markSkinsBelow(c, idx);
      c.renderer.requestAnimationReconcile();
    }
  }
  if(auto v = op.find("visible"); v != op.end())
  {
    tinygltf::utils::setNodeVisibility(model.nodes[idx], {boolean(*v, "visible")});
    editor.updateVisibility(idx);
  }
  return {{"node", idx}};
}

Json opSetLight(Ctx& c, const Json& op)
{
  tinygltf::Model& model = c.scene.getModel();
  int              light = -1;
  if(op.contains("id"))
  {
    const int node = resolveIndex(op, c.ids.nodes, model.nodes.size(), "node");
    light          = model.nodes[node].light;
    if(light < 0)
      fail("node has no light");
  }
  else
    light = resolveIndex(op, {}, model.lights.size(), "light");
  tinygltf::Light& l = c.scene.editor().getLightForEdit(light);
  if(auto v = op.find("color"); v != op.end())
    l.color = numbers<3>(*v, "color");
  if(auto v = op.find("intensity"); v != op.end())
    l.intensity = std::max(0.0, number(*v, "intensity"));
  // `extras.radius`: a directional light's disc radius (km, at the sun's distance), which gives it
  // an angular size and soft shadows; the upload reads it from the light each time it is dirty.
  if(auto v = op.find("radius"); v != op.end())
  {
    tinygltf::Value::Object extras = l.extras.IsObject() ? l.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
    extras["radius"]                = tinygltf::Value(std::max(0.0, number(*v, "radius")));
    l.extras                        = tinygltf::Value(extras);
  }
  c.scene.markLightDirty(light);
  return {{"light", light}};
}

Json animationState(Ctx& c)
{
  AnimationControl& ctrl = c.renderer.editAnimationControl();
  auto&             anim = c.scene.animation();
  Json              out  = {{"clips", anim.getNumAnimations()}, {"clip", ctrl.currentAnimation}, {"playing", ctrl.play},
                            {"speed", ctrl.speed},           {"loop", ctrl.loop}};
  if(ctrl.currentAnimation >= 0 && ctrl.currentAnimation < anim.getNumAnimations())
  {
    const auto& info = anim.getAnimationInfo(ctrl.currentAnimation);
    out["name"]      = info.name;
    out["time"]      = info.currentTime;
    out["start"]     = info.start;
    out["end"]       = info.end;
  }
  return out;
}

void restorePose(Ctx& c)
{
  tinygltf::Model& model = c.scene.getModel();
  const RestPose&  rest  = c.ids.rest;
  for(size_t i = 0; i < model.nodes.size() && i < rest.nodes.size(); i++)
  {
    // only what animations move; transforms the host edited stay
    if(!rest.animatedNodes.count(int(i)))
      continue;
    auto&       n = model.nodes[i];
    const auto& r = rest.nodes[i];
    if(n.translation != r.translation || n.rotation != r.rotation || n.scale != r.scale || n.matrix != r.matrix)
    {
      n.translation = r.translation;
      n.rotation    = r.rotation;
      n.scale       = r.scale;
      n.matrix      = r.matrix;
      c.scene.markNodeDirty(int(i));
    }
  }
  for(size_t m = 0; m < model.meshes.size() && m < rest.meshWeights.size(); m++)
  {
    if(model.meshes[m].weights == rest.meshWeights[m])
      continue;
    model.meshes[m].weights = rest.meshWeights[m];
    const auto& renderNodes = c.scene.getRenderNodes();
    for(int rp : c.ids.renderPrimsOfMesh[m])
      for(size_t rn = 0; rn < renderNodes.size(); rn++)
        if(renderNodes[rn].renderPrimID == rp)
          c.scene.markRenderNodeDirty(int(rn), false, true);
  }
}

Json opAnimation(Ctx& c, const Json& op)
{
  AnimationControl& ctrl  = c.renderer.editAnimationControl();
  auto&             anim  = c.scene.animation();
  const int         clips = anim.getNumAnimations();

  if(auto v = op.find("stop"); v != op.end() && boolean(*v, "stop"))
  {
    ctrl.play    = false;
    ctrl.runOnce = ctrl.reset = ctrl.seek = false;
    for(int i = 0; i < clips; i++)
      anim.getAnimationInfo(i).reset();
    restorePose(c);
    c.renderer.requestAnimationReconcile();
    return animationState(c);
  }

  const bool wantsClip = op.contains("clip") || op.contains("time") || op.contains("play") || op.contains("reset");
  if(wantsClip && clips == 0)
    fail("the scene has no animations");

  if(auto v = op.find("clip"); v != op.end())
  {
    int clip = -1;
    if(v->is_number_integer())
      clip = int(v->get<int64_t>());
    else if(v->is_string())
      for(int i = 0; i < clips; i++)
        if(anim.getAnimationInfo(i).name == v->get<std::string>())
          clip = i;
    if(clip < 0 || clip >= clips)
      fail("'clip' is not a valid animation index or name");
    if(clip != ctrl.currentAnimation)
    {
      ctrl.currentAnimation = clip;
      anim.getAnimationInfo(clip).reset();
      ctrl.seek = true;  // show the new clip's first frame even while paused
    }
  }
  if(auto v = op.find("speed"); v != op.end())
    ctrl.speed = float(number(*v, "speed"));
  if(auto v = op.find("loop"); v != op.end())
    ctrl.loop = boolean(*v, "loop");
  if(auto v = op.find("reset"); v != op.end() && boolean(*v, "reset"))
    ctrl.reset = true;
  if(auto v = op.find("time"); v != op.end())
  {
    if(ctrl.currentAnimation < 0 || ctrl.currentAnimation >= clips)
      ctrl.currentAnimation = 0;
    auto& info       = anim.getAnimationInfo(ctrl.currentAnimation);
    info.currentTime = std::clamp(float(number(*v, "time")), info.start, info.end);
    ctrl.seek        = true;  // evaluate at exactly this time on the next frame
  }
  if(auto v = op.find("play"); v != op.end())
    ctrl.play = boolean(*v, "play");
  return animationState(c);
}

void markMeshRenderNodes(Ctx& c, int mesh)
{
  const auto& renderNodes = c.scene.getRenderNodes();
  for(int rp : c.ids.renderPrimsOfMesh[mesh])
    for(size_t rn = 0; rn < renderNodes.size(); rn++)
      if(renderNodes[rn].renderPrimID == rp)
        c.scene.markRenderNodeDirty(int(rn), false, true);
}

Json opSetMorphWeights(Ctx& c, const Json& op)
{
  tinygltf::Model&        model = c.scene.getModel();
  std::unordered_set<int> meshes;
  for(const PrimRef& p : resolvePrims(op, c.ids, model))
    meshes.insert(p.mesh);
  auto w = op.find("weights");
  if(w == op.end() || !w->is_array())
    fail("'weights' must be an array of numbers");
  for(int m : meshes)
  {
    auto& weights = model.meshes[m].weights;
    if(weights.empty())
      fail("mesh " + std::to_string(m) + " has no morph weights; it was loaded without morph targets");
    if(w->size() != weights.size())
      fail("mesh " + std::to_string(m) + " has " + std::to_string(weights.size()) + " morph targets, got "
           + std::to_string(w->size()) + " weights");
    for(size_t i = 0; i < weights.size(); i++)
    {
      if(!(*w)[i].is_number())
        fail("'weights' must contain numbers");
      weights[i] = (*w)[i].get<double>();
    }
    markMeshRenderNodes(c, m);
  }
  c.renderer.requestAnimationReconcile();
  return {{"meshes", meshes.size()}};
}

Json opSetVertices(Ctx& c, const Json& op)
{
  if(!c.renderer.isEditable())
    fail("vertex edits need an --editable (or --stream) session");
  tinygltf::Model&             model = c.scene.getModel();
  const std::vector<PrimRef>   prims = resolvePrims(op, c.ids, model);
  auto                         pos   = op.find("positions");
  if(pos == op.end())
    fail("'positions' is required");
  std::vector<glm::vec3> positions = decodeVec3(bytesOf(c, *pos, "positions"), "positions");
  std::vector<glm::vec3> normals;
  if(auto n = op.find("normals"); n != op.end() && !n->is_null())
  {
    normals = decodeVec3(bytesOf(c, *n, "normals"), "normals");
    if(normals.size() != positions.size())
      fail("'normals' and 'positions' differ in length");
  }
  // sparse: only the listed vertices, positions[k] (and normals[k]) for vertex indices[k]
  std::vector<uint32_t> indices;
  if(auto ix = op.find("indices"); ix != op.end() && !ix->is_null())
  {
    indices = decodeU32(bytesOf(c, *ix, "indices"), "indices");
    if(indices.size() != positions.size())
      fail("'indices' and 'positions' differ in length");
  }
  const auto& renderPrims = c.scene.getRenderPrimitives();
  for(const PrimRef& p : prims)
  {
    const int count = renderPrims[p.renderPrimID].vertexCount;
    if(indices.empty() && count != int(positions.size()))
      fail("primitive has " + std::to_string(count) + " vertices, got " + std::to_string(positions.size()));
    for(uint32_t v : indices)
      if(v >= uint32_t(count))
        fail("vertex index " + std::to_string(v) + " is out of range: the primitive has " + std::to_string(count) + " vertices");
  }
  for(size_t i = 0; i < prims.size(); i++)
  {
    GltfRenderer::VertexUpload u;
    u.renderPrimID = prims[i].renderPrimID;
    if(i + 1 == prims.size())
    {
      u.positions = std::move(positions);
      u.normals   = std::move(normals);
      u.indices   = std::move(indices);
    }
    else
    {
      u.positions = positions;
      u.normals   = normals;
      u.indices   = indices;
    }
    c.renderer.queueVertexUpload(std::move(u));
  }
  return {{"primitives", prims.size()}};
}

// New inverse bind matrices for the skin of node {id|index}: 16 float32 per joint, column-major as
// glTF stores them, base64. A host that moves joint centres sends these with the bones' new
// transforms instead of exporting the scene again.
Json opSetSkin(Ctx& c, const Json& op)
{
  tinygltf::Model& model = c.scene.getModel();
  const int        idx   = resolveIndex(op, c.ids.nodes, model.nodes.size(), "node");
  const int        skin  = model.nodes[idx].skin;
  if(skin < 0)
    fail("node has no skin");
  auto it = op.find("inverseBindMatrices");
  if(it == op.end())
    fail("'inverseBindMatrices' is required");
  const std::vector<uint8_t> bytes = bytesOf(c, *it, "inverseBindMatrices");
  if(bytes.size() % sizeof(glm::mat4) != 0)
    fail("'inverseBindMatrices' is not a whole number of float32 4x4 matrices");
  std::vector<glm::mat4> ibm(bytes.size() / sizeof(glm::mat4));
  std::memcpy(ibm.data(), bytes.data(), bytes.size());
  if(!c.scene.animation().setInverseBindMatrices(skin, ibm))
    fail("got " + std::to_string(ibm.size()) + " matrices, the skin has " + std::to_string(model.skins[skin].joints.size()) + " joints");
  markSkinRenderNodes(c, skin);
  c.renderer.requestAnimationReconcile();
  return {{"skin", skin}, {"joints", ibm.size()}};
}

Json applyOp(Ctx& c, const std::string& name, const Json& op)
{
  if(name == "set_material")
    return opSetMaterial(c, op);
  if(name == "set_node")
    return opSetNode(c, op);
  if(name == "set_light")
    return opSetLight(c, op);
  if(name == "animation")
    return opAnimation(c, op);
  if(name == "set_morph_weights")
    return opSetMorphWeights(c, op);
  if(name == "set_vertices")
    return opSetVertices(c, op);
  if(name == "set_skin")
    return opSetSkin(c, op);
  fail("unknown op '" + name + "'");
}

Json sceneInfo(Ctx& c)
{
  const tinygltf::Model& model = c.scene.getModel();
  const auto&            anim  = c.scene.animation();
  Json                   nodes = Json::array();
  for(size_t i = 0; i < model.nodes.size(); i++)
  {
    const auto& n  = model.nodes[i];
    std::string id = fvIdOf(n.extras);
    nodes.push_back({{"index", i},
                     {"id", id.empty() ? Json(nullptr) : Json(id)},
                     {"name", n.name},
                     {"mesh", n.mesh},
                     {"skin", n.skin},
                     {"light", n.light},
                     {"visible", tinygltf::utils::getNodeVisibility(n).visible}});
  }
  Json materials = Json::array();
  for(size_t i = 0; i < model.materials.size(); i++)
  {
    const auto& m  = model.materials[i];
    std::string id = fvIdOf(m.extras);
    materials.push_back({{"index", i}, {"id", id.empty() ? Json(nullptr) : Json(id)}, {"name", m.name}, {"alphaMode", m.alphaMode}});
  }
  std::unordered_set<int> skinned, morphed;
  for(const auto& t : anim.getSkinTasks())
    skinned.insert(t.renderPrimID);
  for(uint32_t p : anim.getMorphPrimitives())
    morphed.insert(int(p));
  Json        meshes      = Json::array();
  const auto& renderPrims = c.scene.getRenderPrimitives();
  for(size_t m = 0; m < model.meshes.size(); m++)
  {
    Json prims = Json::array();
    for(int rp : c.ids.renderPrimsOfMesh[m])
    {
      const auto& prim = renderPrims[rp];
      std::string id   = prim.pPrimitive ? fvIdOf(prim.pPrimitive->extras) : std::string{};
      prims.push_back({{"renderPrim", rp},
                       {"id", id.empty() ? Json(nullptr) : Json(id)},
                       {"material", prim.pPrimitive ? prim.pPrimitive->material : -1},
                       {"vertexCount", prim.vertexCount},
                       {"targets", prim.pPrimitive ? prim.pPrimitive->targets.size() : 0},
                       {"skinned", skinned.count(rp) != 0},
                       {"morphed", morphed.count(rp) != 0},
                       {"blasUpdatable", c.renderer.isBlasUpdatable(uint32_t(rp))}});
    }
    meshes.push_back({{"index", m}, {"name", model.meshes[m].name}, {"weights", model.meshes[m].weights.size()}, {"primitives", prims}});
  }
  Json animations = Json::array();
  for(int i = 0; i < anim.getNumAnimations(); i++)
  {
    const auto& info = anim.getAnimationInfo(i);
    animations.push_back({{"index", i}, {"name", info.name}, {"start", info.start}, {"end", info.end}});
  }
  Json lights = Json::array();
  for(size_t i = 0; i < model.lights.size(); i++)
    lights.push_back({{"index", i}, {"type", model.lights[i].type}, {"color", model.lights[i].color}, {"intensity", model.lights[i].intensity}});
  return {{"editable", c.renderer.isEditable()},
          {"nodes", nodes},
          {"materials", materials},
          {"meshes", meshes},
          {"animations", animations},
          {"lights", lights},
          {"animation", animationState(c)}};
}

}  // namespace

//--------------------------------------------------------------------------------------------------

void registerSceneMcpTools(nvmcp::Element& mcp, const std::shared_ptr<GltfRenderer>& renderer)
{
  const std::weak_ptr<GltfRenderer> weak = renderer;
  auto                              ids  = std::make_shared<SceneIds>();

  // Runs `body` with a resolved context, turning bad input into a tool error.
  auto withScene = [weak, ids](std::string_view payload, auto&& body) -> nvmcp::ToolResult {
    auto r = weak.lock();
    if(!r)
      return nvmcp::ToolResult::error(R"({"error":"renderer is shutting down"})");
    nvvkgltf::Scene* scene = r->editScene();
    if(!scene || !scene->valid())
      return nvmcp::ToolResult::error(R"({"error":"no scene is loaded"})");
    const Json input = payload.empty() ? Json::object() : Json::parse(payload, nullptr, /*allow_exceptions=*/false);
    if(input.is_discarded() || !input.is_object())
      return nvmcp::ToolResult::error(R"({"error":"arguments must be a JSON object"})");
    const tinygltf::Model& model = scene->getModel();
    if(ids->scene != scene || ids->generation != r->sceneGeneration() || ids->nodeCount != model.nodes.size()
       || ids->materialCount != model.materials.size() || ids->primCount != scene->getRenderPrimitives().size())
      rebuildIds(*ids, *scene, r->sceneGeneration());
    Ctx c{*r, *scene, *ids};
    try
    {
      return nvmcp::ToolResult::success(body(c, input).dump());
    }
    catch(const OpError& e)
    {
      return nvmcp::ToolResult::error(Json{{"error", e.message}}.dump());
    }
    catch(const std::exception& e)
    {
      return nvmcp::ToolResult::error(Json{{"error", e.what()}}.dump());
    }
  };

  mcp.registerTool({
      .name        = "vk_gltf_scene_info",
      .description = "Describe the loaded scene for live editing: nodes, materials, meshes with their primitives "
                     "(render primitive id, host id from extras.fvId, vertex count, skinned, morphed, whether the "
                     "BLAS can be refit), animations, lights and the animation state.",
      .handler     = [withScene](std::string_view p) { return withScene(p, [](Ctx& c, const Json&) { return sceneInfo(c); }); },
      .annotations = {.readOnlyHint = true, .destructiveHint = false, .idempotentHint = true, .openWorldHint = false},
      .runOnApplicationThread = true,
  });

  mcp.registerTool({
      .name        = "vk_gltf_edit",
      .description = "Apply several live edits in order, all visible in the same frame. Each op is {\"op\": "
                     "\"set_material\"|\"set_node\"|\"set_light\"|\"animation\"|\"set_morph_weights\"|\"set_vertices\"|\"set_skin\", "
                     "...the same fields as the single tool}. A failing op is reported and the rest still apply.",
      .inputSchema = R"({"type":"object","properties":{"ops":{"type":"array","items":{"type":"object"}}},"required":["ops"]})",
      .handler =
          [withScene](std::string_view p) {
            const double handled = traceNowMs();
            return withScene(p, [handled](Ctx& c, const Json& in) {
              const double parsed = traceNowMs();
              auto ops = in.find("ops");
              if(ops == in.end() || !ops->is_array())
                fail("'ops' must be an array");
              Json   results = Json::array();
              size_t applied = 0;
              for(const Json& op : *ops)
              {
                try
                {
                  if(!op.is_object() || !op.contains("op") || !op["op"].is_string())
                    fail("each op needs an \"op\" name");
                  Json r = applyOp(c, op["op"].get<std::string>(), op);
                  r["ok"] = true;
                  results.push_back(std::move(r));
                  applied++;
                }
                catch(const OpError& e)
                {
                  results.push_back({{"ok", false}, {"error", e.message}});
                }
              }
              // A host's latency trace: its stamps, then this call's; the renderer and the
              // stream add the rest (latency_trace.hpp).
              if(auto tr = in.find("trace"); tr != in.end() && tr->is_object())
              {
                LatencyTrace t;
                for(const char* stage : {"input", "morphed", "flushed", "built", "sent"})
                  if(auto s = tr->find(stage); s != tr->end() && s->is_number())
                    t.emplace_back(stage, s->get<double>());
                t.emplace_back("handled", handled);
                t.emplace_back("parsed", parsed);
                t.emplace_back("queued", traceNowMs());
                c.renderer.traceEdit(std::move(t));
              }
              return Json{{"applied", applied}, {"results", results}};
            });
          },
      .annotations            = {.readOnlyHint = false, .destructiveHint = false, .idempotentHint = true, .openWorldHint = false},
      .runOnApplicationThread = true,
  });

  mcp.registerTool({
      .name = "vk_gltf_add_scene",
      .description = "Merge a glTF/GLB file into the loaded scene without reloading it: each root node of the file goes "
                     "under the node whose extras.fvId equals the root's extras.fvParent (a scene root when it has "
                     "none). Textures already loaded are kept; only the file's own images are read. Fails, changing "
                     "nothing, when a named parent is not in the scene.",
      .inputSchema = R"({"type":"object","properties":{"file":{"type":"string"}},"required":["file"]})",
      .handler =
          [withScene](std::string_view p) {
            return withScene(p, [](Ctx& c, const Json& in) {
              if(!c.renderer.isEditable())
                fail("adding to the scene needs an --editable (or --stream) session");
              auto file = in.find("file");
              if(file == in.end() || !file->is_string() || file->get_ref<const std::string&>().empty())
                fail("'file' must be a path");
              // parents resolve against the ids before the merge; an unknown one refuses it
              std::vector<std::string> missing;
              const auto&              byId     = c.ids.nodes;
              const auto               parentOf = [&](const tinygltf::Node& root) -> int {
                std::string parent;
                if(root.extras.IsObject() && root.extras.Has("fvParent") && root.extras.Get("fvParent").IsString())
                  parent = root.extras.Get("fvParent").Get<std::string>();
                if(parent.empty())
                  return -1;
                if(auto it = byId.find(parent); it != byId.end())
                  return it->second;
                missing.push_back(parent);
                return std::numeric_limits<int>::max();  // out of range: mergeUnder refuses
              };
              const int roots = c.renderer.appendSceneFile(nvutils::pathFromUtf8(file->get<std::string>()), parentOf);
              if(!missing.empty())
                fail("no node with id '" + missing.front() + "' to attach to");
              if(roots < 0)
                fail("could not merge '" + file->get<std::string>() + "' (see the log)");
              return Json{{"roots", roots},
                          {"nodes", c.scene.getModel().nodes.size()},
                          {"primitives", c.scene.getRenderPrimitives().size()}};
            });
          },
      .annotations = {.readOnlyHint = false, .destructiveHint = false, .idempotentHint = false, .openWorldHint = false},
      .runOnApplicationThread = true,
  });

  struct Single
  {
    const char* tool;
    const char* op;
    const char* description;
  };
  const Single singles[] = {
      {"vk_gltf_set_material", "set_material",
       "Merge a partial glTF material into material {id|index}: pbrMetallicRoughness factors, emissiveFactor, "
       "alphaMode, alphaCutoff, doubleSided, normalTexture.scale, and extensions (merge patch per extension; null "
       "removes it). Texture indices must already exist in the scene."},
      {"vk_gltf_set_node", "set_node", "Set a node's visible flag and/or local translation, rotation (x,y,z,w) and scale."},
      {"vk_gltf_set_light", "set_light", "Set a punctual light's color, intensity and/or radius (extras.radius: a directional light's angular size), by light index or by the id of its node."},
      {"vk_gltf_animation", "animation",
       "Control playback: clip (index or name), time (seconds, evaluated exactly), play, speed, loop, reset, stop "
       "(stop restores the pose the scene was loaded with). Returns the state."},
      {"vk_gltf_set_morph_weights", "set_morph_weights", "Set all morph target weights of the mesh owning primitive {id} or of {mesh}."},
      {"vk_gltf_set_vertices", "set_vertices",
       "Replace vertex positions (and normals) of primitive {id} (all primitives sharing it) or {mesh, primitive?}: "
       "base64 little-endian float32 xyz, one per vertex. With indices (base64 little-endian uint32), only those "
       "vertices change: positions[k] (and normals[k]) belong to vertex indices[k]. Skinned or morphed primitives get "
       "new base geometry. Any of the arrays may instead be {\"blob\": id}, the raw bytes posted to the frame "
       "stream's /blob."},
      {"vk_gltf_set_skin", "set_skin",
       "Replace the inverse bind matrices of the skin of node {id|index}: inverseBindMatrices is base64 little-endian "
       "float32, 16 per joint, column-major as glTF stores them. Send it with the joints' new node transforms when a "
       "host moves joint centres."},
  };
  for(const Single& s : singles)
  {
    const std::string opName = s.op;
    mcp.registerTool({
        .name        = s.tool,
        .description = s.description,
        .inputSchema = R"({"type":"object"})",
        .handler = [withScene, opName](std::string_view p) { return withScene(p, [&](Ctx& c, const Json& in) { return applyOp(c, opName, in); }); },
        .annotations            = {.readOnlyHint = false, .destructiveHint = false, .idempotentHint = true, .openWorldHint = false},
        .runOnApplicationThread = true,
    });
  }
}

#endif  // USE_NVMCP
