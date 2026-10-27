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

#include "scene_export_gltf.hpp"

#include <map>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "gltf_scene_merger.hpp"
#include "tinygltf_utils.hpp"

namespace rs {

namespace {

class Writer
{
public:
  explicit Writer(tinygltf::Model& m)
      : m_model(m)
  {
    m.buffers.emplace_back();
  }

  template <class T>
  int accessor(const std::vector<T>& data, int componentType, int type, int target = 0)
  {
    if(data.empty())
      return -1;
    return tinygltf::utils::appendAccessor(m_model, 0, data.data(), data.size() * sizeof(T), componentType, type, data.size(), target);
  }

  int vec3(const std::vector<glm::vec3>& d, bool bounds = false)
  {
    const int a = accessor(d, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, TINYGLTF_TARGET_ARRAY_BUFFER);
    if(a >= 0 && bounds)
    {
      glm::vec3 lo(std::numeric_limits<float>::max()), hi(std::numeric_limits<float>::lowest());
      for(const glm::vec3& p : d)
      {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
      }
      m_model.accessors[size_t(a)].minValues = {lo.x, lo.y, lo.z};
      m_model.accessors[size_t(a)].maxValues = {hi.x, hi.y, hi.z};
    }
    return a;
  }

  int bufferView(const std::vector<unsigned char>& bytes)
  {
    tinygltf::Buffer& b = m_model.buffers[0];
    while(b.data.size() % 4)
      b.data.push_back(0);
    tinygltf::BufferView view;
    view.buffer     = 0;
    view.byteOffset = b.data.size();
    view.byteLength = bytes.size();
    b.data.insert(b.data.end(), bytes.begin(), bytes.end());
    m_model.bufferViews.push_back(view);
    return int(m_model.bufferViews.size()) - 1;
  }

private:
  tinygltf::Model& m_model;
};

std::string base64(const std::vector<unsigned char>& bytes)
{
  static const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string        out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for(size_t i = 0; i < bytes.size(); i += 3)
  {
    const uint32_t v = uint32_t(bytes[i]) << 16 | (i + 1 < bytes.size() ? uint32_t(bytes[i + 1]) << 8 : 0) | (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    out += chars[(v >> 18) & 63];
    out += chars[(v >> 12) & 63];
    out += i + 1 < bytes.size() ? chars[(v >> 6) & 63] : '=';
    out += i + 2 < bytes.size() ? chars[v & 63] : '=';
  }
  return out;
}

}  // namespace

tinygltf::Model exportGltf(const RetainedScene& scene, const ExportOptions& options)
{
  tinygltf::Model out;
  out.asset.version   = "2.0";
  out.asset.generator = "Figure Viewer";
  Writer w(out);

  // the entities written, depth first, each a node
  std::vector<EntityId>                 order;
  std::map<EntityId, int>               nodeOf;
  const std::vector<EntityId>&          roots = options.roots.empty() ? scene.roots() : options.roots;
  std::vector<EntityId>                 stack(roots.rbegin(), roots.rend());
  while(!stack.empty())
  {
    const EntityId id = stack.back();
    stack.pop_back();
    const Entity* e = scene.entity(id);
    if(!e || nodeOf.count(id))
      continue;
    nodeOf[id] = int(order.size());
    order.push_back(id);
    stack.insert(stack.end(), e->children.rbegin(), e->children.rend());
  }

  // images, samplers and textures: each texture table appended once
  std::map<TextureId, int>                         imageOf;
  std::vector<SamplerDesc>                         samplers;
  std::map<const TextureTable*, int>               tableBase;
  const auto image = [&](TextureId id) -> int {
    if(auto it = imageOf.find(id); it != imageOf.end())
      return it->second;
    const TextureRecord* t = scene.texture(id);
    if(!t)
      return imageOf[id] = -1;
    tinygltf::Image img;
    img.name     = t->name;
    img.mimeType = t->mimeType;
    img.width    = t->width;
    img.height   = t->height;
    if(!t->uri.empty())
      img.uri = t->uri;
    else if(!t->bytes.empty() && options.binary)
      img.bufferView = w.bufferView(t->bytes);
    else if(!t->bytes.empty())
      img.uri = "data:" + (t->mimeType.empty() ? std::string("application/octet-stream") : t->mimeType) + ";base64," + base64(t->bytes);
    else if(!t->pixels.empty())
    {
      // decoded pixels: tinygltf's writer encodes them (a PNG inside a GLB)
      img.image     = t->pixels;
      img.component = t->component;
      img.bits      = t->bits;
      img.pixel_type = t->bits == 16 ? TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT : TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
      img.mimeType  = "image/png";
    }
    out.images.push_back(std::move(img));
    return imageOf[id] = int(out.images.size()) - 1;
  };
  const auto sampler = [&](const SamplerDesc& d) -> int {
    for(size_t i = 0; i < samplers.size(); i++)
      if(samplers[i] == d)
        return int(i);
    samplers.push_back(d);
    tinygltf::Sampler s;
    s.magFilter = d.magFilter;
    s.minFilter = d.minFilter;
    s.wrapS     = d.wrapS;
    s.wrapT     = d.wrapT;
    out.samplers.push_back(s);
    return int(samplers.size()) - 1;
  };
  const auto textures = [&](const std::shared_ptr<const TextureTable>& table) -> int {
    if(!table)
      return 0;
    if(auto it = tableBase.find(table.get()); it != tableBase.end())
      return it->second;
    const int base = int(out.textures.size());
    for(const TextureRef& ref : *table)
    {
      tinygltf::Texture t;
      t.source  = image(ref.texture);
      t.sampler = sampler(ref.sampler);
      if(const TextureRecord* r = scene.texture(ref.texture); r && !r->sourceExtension.empty())
      {
        tinygltf::Value::Object source;
        source["source"]                   = tinygltf::Value(t.source);
        t.extensions[r->sourceExtension]    = tinygltf::Value(source);
        t.source                           = -1;
      }
      out.textures.push_back(std::move(t));
    }
    return tableBase[table.get()] = base;
  };

  std::map<MaterialId, int> materialOf;
  const auto                material = [&](MaterialId id) -> int {
    if(auto it = materialOf.find(id); it != materialOf.end())
      return it->second;
    const MaterialRecord* m = scene.material(id);
    if(!m)
      return materialOf[id] = -1;
    tinygltf::Material g = m->payload;
    nvvkgltf::SceneMerger::rebaseMaterialTextures(g, textures(m->textures));
    out.materials.push_back(std::move(g));
    return materialOf[id] = int(out.materials.size()) - 1;
  };

  // vertex sets: one set of accessors each, shared by the primitives of its submeshes
  struct Streams
  {
    std::map<std::string, int>              attributes;
    std::vector<std::map<std::string, int>> targets;
    std::vector<int>                        indices;  // per submesh
  };
  std::map<MeshId, Streams> streamsOf;
  const auto                streams = [&](MeshId id) -> const Streams& {
    if(auto it = streamsOf.find(id); it != streamsOf.end())
      return it->second;
    const MeshRecord& r = *scene.mesh(id);
    Streams           s;
    const auto        put = [&](const char* name, int accessor) {
      if(accessor >= 0)
        s.attributes[name] = accessor;
    };
    put("POSITION", w.vec3(r.positions, true));
    put("NORMAL", w.vec3(r.normals));
    put("TANGENT", w.accessor(r.tangents, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, TINYGLTF_TARGET_ARRAY_BUFFER));
    put("TEXCOORD_0", w.accessor(r.uv0, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, TINYGLTF_TARGET_ARRAY_BUFFER));
    put("TEXCOORD_1", w.accessor(r.uv1, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, TINYGLTF_TARGET_ARRAY_BUFFER));
    put("COLOR_0", w.accessor(r.colors, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, TINYGLTF_TARGET_ARRAY_BUFFER));
    if(!r.joints.empty())
    {
      std::vector<glm::u16vec4> joints(r.joints.begin(), r.joints.end());
      put("JOINTS_0", w.accessor(joints, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC4, TINYGLTF_TARGET_ARRAY_BUFFER));
    }
    put("WEIGHTS_0", w.accessor(r.weights, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, TINYGLTF_TARGET_ARRAY_BUFFER));
    for(const MorphTarget& t : r.targets)
    {
      std::map<std::string, int> a;
      if(const int p = w.vec3(t.positions, true); p >= 0)
        a["POSITION"] = p;
      if(const int n = w.vec3(t.normals); n >= 0)
        a["NORMAL"] = n;
      if(const int tg = w.vec3(t.tangents); tg >= 0)
        a["TANGENT"] = tg;
      s.targets.push_back(std::move(a));
    }
    for(const Submesh& sub : r.submeshes)
    {
      const std::vector<uint32_t> idx(r.indices.begin() + sub.firstIndex, r.indices.begin() + sub.firstIndex + sub.indexCount);
      s.indices.push_back(w.accessor(idx, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER));
    }
    return streamsOf[id] = std::move(s);
  };

  // a glTF mesh per distinct instance: the same parts and materials share one
  std::map<std::vector<std::pair<MeshId, std::vector<MaterialId>>>, int> meshOf;
  const auto                                                          meshFor = [&](const MeshInstance& instance) -> int {
    std::vector<std::pair<MeshId, std::vector<MaterialId>>> key;
    for(const MeshPart& p : instance.parts)
      key.push_back({p.mesh, p.materials});
    if(auto it = meshOf.find(key); it != meshOf.end())
      return it->second;
    struct Run
    {
      int                 order;
      tinygltf::Primitive primitive;
    };
    std::vector<Run> runs;
    std::string      name;
    std::vector<double> weights;
    for(const MeshPart& part : instance.parts)
    {
      const MeshRecord* r = scene.mesh(part.mesh);
      if(!r)
        continue;
      if(name.empty())
        name = r->name;
      if(weights.empty())
        weights.assign(r->targetWeights.begin(), r->targetWeights.end());
      const Streams& s = streams(part.mesh);
      for(size_t i = 0; i < r->submeshes.size(); i++)
      {
        tinygltf::Primitive p;
        p.mode       = TINYGLTF_MODE_TRIANGLES;
        p.attributes = s.attributes;
        p.targets    = s.targets;
        p.indices    = s.indices[i];
        const uint32_t slot = r->submeshes[i].material;
        p.material          = slot < part.materials.size() ? material(part.materials[slot]) : -1;
        runs.push_back({r->submeshes[i].order, std::move(p)});
      }
    }
    std::stable_sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.order < b.order; });
    tinygltf::Mesh mesh;
    mesh.name    = name;
    mesh.weights = weights;
    for(Run& r : runs)
      mesh.primitives.push_back(std::move(r.primitive));
    out.meshes.push_back(std::move(mesh));
    return meshOf[key] = int(out.meshes.size()) - 1;
  };

  // nodes
  out.nodes.resize(order.size());
  std::map<SkinId, int> skinOf;
  for(size_t i = 0; i < order.size(); i++)
  {
    const Entity&   e = *scene.entity(order[i]);
    tinygltf::Node& n = out.nodes[i];
    n.name            = e.name;
    if(e.local.matrix)
      n.matrix.assign(glm::value_ptr(*e.local.matrix), glm::value_ptr(*e.local.matrix) + 16);
    else
    {
      if(e.local.translation != glm::vec3(0.f))
        n.translation = {e.local.translation.x, e.local.translation.y, e.local.translation.z};
      if(e.local.rotation != glm::quat(1.f, 0.f, 0.f, 0.f))
        n.rotation = {e.local.rotation.x, e.local.rotation.y, e.local.rotation.z, e.local.rotation.w};
      if(e.local.scale != glm::vec3(1.f))
        n.scale = {e.local.scale.x, e.local.scale.y, e.local.scale.z};
    }
    for(EntityId c : e.children)
      if(auto it = nodeOf.find(c); it != nodeOf.end())
        n.children.push_back(it->second);
    const LightRecord* light = scene.light(e.light);
    if(!e.visible || (light && !light->shown))
      tinygltf::utils::setNodeVisibility(n, {false});
    if(e.mesh && !e.mesh->parts.empty())
    {
      n.mesh = meshFor(*e.mesh);
      n.weights.assign(e.mesh->morphWeights.begin(), e.mesh->morphWeights.end());
      if(!e.mesh->skin.isNone())
        skinOf.try_emplace(e.mesh->skin, -1);
      if(!e.mesh->copies.empty())
      {
        std::vector<glm::vec3> t, s;
        std::vector<glm::quat> r;
        for(const glm::mat4& m : e.mesh->copies)
        {
          glm::vec3 tr, sc, skew;
          glm::quat ro;
          glm::vec4 persp;
          glm::decompose(m, sc, ro, tr, skew, persp);
          t.push_back(tr);
          r.push_back(ro);
          s.push_back(sc);
        }
        tinygltf::Value::Object attributes;
        attributes["TRANSLATION"] = tinygltf::Value(w.accessor(t, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3));
        attributes["ROTATION"]    = tinygltf::Value(w.accessor(r, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4));
        attributes["SCALE"]       = tinygltf::Value(w.accessor(s, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3));
        tinygltf::Value::Object ext;
        ext["attributes"]                                = tinygltf::Value(attributes);
        n.extensions[EXT_MESH_GPU_INSTANCING_EXTENSION_NAME] = tinygltf::Value(ext);
      }
    }
    if(light)
    {
      n.light = int(out.lights.size());
      out.lights.push_back(light->payload);
    }
    if(const CameraRecord* camera = scene.camera(e.camera))
    {
      n.camera = int(out.cameras.size());
      out.cameras.push_back(camera->payload);
    }
  }
  // skins: the joints written, their inverse binds
  for(auto& [id, index] : skinOf)
  {
    const SkinRecord* s = scene.skin(id);
    if(!s)
      continue;
    tinygltf::Skin skin;
    skin.name = s->name;
    std::vector<glm::mat4> ibm;
    for(size_t j = 0; j < s->joints.size(); j++)
      if(auto it = nodeOf.find(s->joints[j]); it != nodeOf.end())
      {
        skin.joints.push_back(it->second);
        ibm.push_back(j < s->inverseBind.size() ? s->inverseBind[j] : glm::mat4(1.f));
      }
    skin.inverseBindMatrices = w.accessor(ibm, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_MAT4);
    if(auto it = nodeOf.find(s->skeleton); it != nodeOf.end())
      skin.skeleton = it->second;
    index = int(out.skins.size());
    out.skins.push_back(std::move(skin));
  }
  for(size_t i = 0; i < order.size(); i++)
  {
    const Entity& e = *scene.entity(order[i]);
    if(e.mesh && !e.mesh->skin.isNone())
      out.nodes[i].skin = skinOf[e.mesh->skin];
  }

  tinygltf::Scene s;
  for(EntityId r : roots)
    if(auto it = nodeOf.find(r); it != nodeOf.end())
      s.nodes.push_back(it->second);
  out.scenes.push_back(std::move(s));
  out.defaultScene = 0;
  if(out.buffers[0].data.empty())
    out.buffers.clear();
  tinygltf::utils::syncExtensionsUsed(out);
  return out;
}

}  // namespace rs
