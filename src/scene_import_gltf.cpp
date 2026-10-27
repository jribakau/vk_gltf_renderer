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

#include "scene_import_gltf.hpp"

#include <map>
#include <numeric>
#include <set>
#include <span>

#include <glm/gtc/type_ptr.hpp>

#include "tinygltf_utils.hpp"

namespace rs {

namespace {

// The attribute accessors (and morph targets) a primitive reads: primitives with the same key share
// one vertex set (as nvvkgltf::Scene::attributeKey).
std::string attributeKey(const tinygltf::Primitive& p)
{
  std::string key;
  for(const auto& [name, accessor] : p.attributes)
    key += name + ':' + std::to_string(accessor) + ';';
  for(const auto& target : p.targets)
  {
    key += '|';
    for(const auto& [name, accessor] : target)
      key += name + ':' + std::to_string(accessor) + ';';
  }
  return key;
}

template <class T>
std::vector<T> attribute(const tinygltf::Model& model, const std::map<std::string, int>& attributes, const char* name)
{
  const auto it = attributes.find(name);
  if(it == attributes.end() || it->second < 0 || it->second >= int(model.accessors.size()))
    return {};
  const tinygltf::Accessor& accessor = model.accessors[size_t(it->second)];
  if(!tinygltf::utils::canReadAccessor<T>(accessor))
    return {};
  std::vector<T>     storage;
  std::span<const T> data = tinygltf::utils::getAccessorData(model, accessor, &storage);
  return std::vector<T>(data.begin(), data.end());
}

std::vector<uint32_t> primitiveIndices(const tinygltf::Model& model, const tinygltf::Primitive& p)
{
  if(p.indices >= 0)
  {
    std::vector<uint32_t>     storage;
    std::span<const uint32_t> data = tinygltf::utils::getAccessorData(model, model.accessors[size_t(p.indices)], &storage);
    return std::vector<uint32_t>(data.begin(), data.end());
  }
  std::vector<uint32_t> out(tinygltf::utils::getVertexCount(model, p));
  std::iota(out.begin(), out.end(), 0u);
  return out;
}

Transform transformOf(const tinygltf::Node& node)
{
  Transform t;
  if(node.matrix.size() == 16)
  {
    t.matrix = glm::make_mat4(node.matrix.data());
    return t;
  }
  if(node.translation.size() == 3)
    t.translation = glm::vec3(glm::make_vec3(node.translation.data()));
  if(node.rotation.size() == 4)
    t.rotation = glm::quat(float(node.rotation[3]), float(node.rotation[0]), float(node.rotation[1]), float(node.rotation[2]));
  if(node.scale.size() == 3)
    t.scale = glm::vec3(glm::make_vec3(node.scale.data()));
  return t;
}

SamplerDesc samplerOf(const tinygltf::Model& model, int sampler)
{
  SamplerDesc d;
  if(sampler >= 0 && sampler < int(model.samplers.size()))
  {
    const tinygltf::Sampler& s = model.samplers[size_t(sampler)];
    d.magFilter                = s.magFilter;
    d.minFilter                = s.minFilter;
    d.wrapS                    = s.wrapS;
    d.wrapT                    = s.wrapT;
  }
  return d;
}

}  // namespace

bool materialNeedsTangents(const tinygltf::Material& material)
{
  return material.normalTexture.index >= 0 || material.extensions.count(KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME);
}

ImportResult importGltf(RetainedScene& scene, const tinygltf::Model& model, const ImportOptions& options)
{
  ImportResult          out;
  std::set<std::string> noticed;
  const auto            notice = [&](const std::string& text) {
    if(noticed.insert(text).second)
      out.notices.push_back(text);
  };
  for(const char* ext : {KHR_INTERACTIVITY_EXTENSION_NAME, KHR_ANIMATION_POINTER, EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME})
    if(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), ext) != model.extensionsUsed.end())
      notice(std::string(ext) + " is not drawn yet; left out");

  // ---- images and the texture table every material of this file shares
  std::vector<TextureId> images(model.images.size());
  for(size_t i = 0; i < model.images.size(); i++)
  {
    const tinygltf::Image& img = model.images[i];
    TextureRecord          t;
    t.name     = img.name;
    t.uri      = img.uri;
    t.mimeType = img.mimeType;
    t.width    = img.width;
    t.height   = img.height;
    if(img.uri.empty() && img.bufferView >= 0 && img.bufferView < int(model.bufferViews.size()))
    {
      const tinygltf::BufferView& view = model.bufferViews[size_t(img.bufferView)];
      if(view.buffer >= 0 && view.buffer < int(model.buffers.size())
         && view.byteOffset + view.byteLength <= model.buffers[size_t(view.buffer)].data.size())
      {
        const auto begin = model.buffers[size_t(view.buffer)].data.begin() + std::ptrdiff_t(view.byteOffset);
        t.bytes.assign(begin, begin + std::ptrdiff_t(view.byteLength));
      }
    }
    images[i] = scene.addTexture(std::move(t));
  }
  auto table = std::make_shared<TextureTable>();
  for(const tinygltf::Texture& texture : model.textures)
  {
    const int source = tinygltf::utils::getTextureImageIndex(texture);
    table->push_back({source >= 0 && source < int(images.size()) ? images[size_t(source)] : TextureId{}, samplerOf(model, texture.sampler)});
  }

  // ---- materials, made when a primitive draws with one
  out.materials.assign(model.materials.size(), {});
  MaterialId defaultMaterial;
  const auto materialOf = [&](int index) -> MaterialId {
    if(index < 0 || index >= int(model.materials.size()))
    {
      if(defaultMaterial.isNone())
        defaultMaterial = scene.addMaterial({.owner = {options.prop, -1}, .textures = table});
      return defaultMaterial;
    }
    if(out.materials[size_t(index)].isNone())
      out.materials[size_t(index)] = scene.addMaterial({.owner = {options.prop, index}, .payload = model.materials[size_t(index)], .textures = table});
    return out.materials[size_t(index)];
  };

  // ---- tangents for the vertex sets whose materials need them, made as nvvkgltf::Scene makes
  // them: one set of tangents per attribute key, over the triangles of every primitive that needs
  // them (across meshes that share the accessors), the material chosen as it chooses it
  std::map<std::string, std::vector<uint32_t>> tangentIndices;
  for(const tinygltf::Mesh& mesh : model.meshes)
    for(const tinygltf::Primitive& p : mesh.primitives)
    {
      const int matId = p.material >= 0 && p.material < int(model.materials.size()) ? p.material : 0;
      if(matId >= int(model.materials.size()) || !materialNeedsTangents(model.materials[size_t(matId)]))
        continue;
      if(p.attributes.count("TANGENT") || !tinygltf::utils::primitiveGeometryError(model, p).empty())
        continue;
      const std::vector<uint32_t> own = primitiveIndices(model, p);
      auto&                       all = tangentIndices[attributeKey(p)];
      all.insert(all.end(), own.begin(), own.end());
    }

  // ---- meshes: each glTF mesh's drawable primitives, grouped into vertex sets by attribute key
  struct PartPlan
  {
    MeshId                mesh;
    std::vector<int>      primitives;  // the glTF primitive of each submesh
  };
  std::vector<std::vector<PartPlan>> plans(model.meshes.size());
  out.meshes.assign(model.meshes.size(), {});
  for(size_t mi = 0; mi < model.meshes.size(); mi++)
  {
    const tinygltf::Mesh&         mesh = model.meshes[mi];
    std::map<std::string, size_t> partOfKey;
    std::vector<MeshRecord>       records;
    for(size_t pi = 0; pi < mesh.primitives.size(); pi++)
    {
      const tinygltf::Primitive& p = mesh.primitives[pi];
      // (the mode is not read: nvvkgltf::Scene draws every primitive as triangles, and so does this)
      if(!tinygltf::utils::primitiveGeometryError(model, p).empty())
        continue;
      const std::string key = attributeKey(p);
      auto [it, fresh]      = partOfKey.try_emplace(key, records.size());
      if(fresh)
      {
        MeshRecord r;
        r.name      = mesh.name;
        r.owner     = {options.prop, int32_t(mi)};
        r.positions = attribute<glm::vec3>(model, p.attributes, "POSITION");
        r.normals   = attribute<glm::vec3>(model, p.attributes, "NORMAL");
        r.uv0       = attribute<glm::vec2>(model, p.attributes, "TEXCOORD_0");
        r.uv1       = attribute<glm::vec2>(model, p.attributes, "TEXCOORD_1");
        r.colors    = attribute<glm::vec4>(model, p.attributes, "COLOR_0");
        if(r.colors.empty())
          for(const glm::vec3& c : attribute<glm::vec3>(model, p.attributes, "COLOR_0"))
            r.colors.push_back(glm::vec4(c, 1.f));
        r.joints  = attribute<glm::uvec4>(model, p.attributes, "JOINTS_0");
        r.weights = attribute<glm::vec4>(model, p.attributes, "WEIGHTS_0");
        r.tangents = attribute<glm::vec4>(model, p.attributes, "TANGENT");
        if(r.tangents.empty())
          if(auto t = tangentIndices.find(key); t != tangentIndices.end())
          {
            r.tangents.assign(r.positions.size(), glm::vec4(0.f));
            tinygltf::utils::computeTangents(r.positions, r.normals, r.uv0, t->second, r.tangents);
          }
        for(const auto& target : p.targets)
          r.targets.push_back({attribute<glm::vec3>(model, target, "POSITION"), attribute<glm::vec3>(model, target, "NORMAL"),
                               attribute<glm::vec3>(model, target, "TANGENT")});
        for(double w : mesh.weights)
          r.targetWeights.push_back(float(w));
        records.push_back(std::move(r));
        plans[mi].push_back({});
      }
      MeshRecord&                 r   = records[it->second];
      const std::vector<uint32_t> idx = primitiveIndices(model, p);
      r.submeshes.push_back({.firstIndex = uint32_t(r.indices.size()), .indexCount = uint32_t(idx.size()),
                             .material = uint32_t(r.submeshes.size()), .order = int32_t(pi)});
      r.indices.insert(r.indices.end(), idx.begin(), idx.end());
      plans[mi][it->second].primitives.push_back(int(pi));
      if(p.extensions.count(KHR_MATERIALS_VARIANTS_EXTENSION_NAME))
        notice("KHR_materials_variants: the default material is drawn");
    }
    for(size_t r = 0; r < records.size(); r++)
    {
      plans[mi][r].mesh = scene.addMesh(std::move(records[r]));
      out.meshes[mi].push_back(plans[mi][r].mesh);
    }
  }

  // ---- skins, their joints filled in once every node is an entity
  std::vector<SkinId> skins(model.skins.size());

  // ---- the scene's nodes, depth first
  out.entities.assign(model.nodes.size(), {});
  std::vector<LightId>  lights;
  std::vector<CameraId> cameras;
  const int sceneIndex = options.scene >= 0                                                  ? options.scene :
                         model.defaultScene >= 0 && model.defaultScene < int(model.scenes.size()) ? model.defaultScene :
                                                                                               0;
  std::vector<std::pair<int, EntityId>> stack;  // (node, parent entity)
  if(sceneIndex < int(model.scenes.size()))
    for(auto it = model.scenes[size_t(sceneIndex)].nodes.rbegin(); it != model.scenes[size_t(sceneIndex)].nodes.rend(); ++it)
      stack.push_back({*it, options.parent});
  while(!stack.empty())
  {
    const auto [ni, parent] = stack.back();
    stack.pop_back();
    if(ni < 0 || ni >= int(model.nodes.size()) || !out.entities[size_t(ni)].isNone())
      continue;  // out of range, or reached twice (a node has one parent)
    const tinygltf::Node& node = model.nodes[size_t(ni)];
    const EntityId        e    = scene.createEntity(node.name, {options.prop, ni}, parent);
    out.entities[size_t(ni)]   = e;
    if(parent == options.parent)
      out.roots.push_back(e);
    scene.setLocal(e, transformOf(node));
    scene.setVisible(e, tinygltf::utils::getNodeVisibility(node).visible);
    if(node.mesh >= 0 && node.mesh < int(model.meshes.size()) && !plans[size_t(node.mesh)].empty())
    {
      MeshInstance instance;
      for(const PartPlan& plan : plans[size_t(node.mesh)])
      {
        MeshPart part{.mesh = plan.mesh};
        for(int pi : plan.primitives)
          part.materials.push_back(materialOf(model.meshes[size_t(node.mesh)].primitives[size_t(pi)].material));
        instance.parts.push_back(std::move(part));
      }
      for(double w : node.weights)
        instance.morphWeights.push_back(float(w));
      if(tinygltf::utils::hasElementName(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME))
        instance.copies = tinygltf::utils::gpuInstancingMatrices(
            model, tinygltf::utils::getElementValue(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME).Get("attributes"));
      if(node.skin >= 0 && node.skin < int(model.skins.size()))
      {
        if(skins[size_t(node.skin)].isNone())
          skins[size_t(node.skin)] = scene.addSkin({.name = model.skins[size_t(node.skin)].name});
        instance.skin = skins[size_t(node.skin)];
      }
      scene.setMeshInstance(e, std::move(instance));
    }
    if(node.light >= 0 && node.light < int(model.lights.size()))
    {
      lights.push_back(scene.addLight({.owner = {options.prop, node.light}, .payload = model.lights[size_t(node.light)]}));
      scene.setEntityLight(e, lights.back());
    }
    if(node.camera >= 0 && node.camera < int(model.cameras.size()))
    {
      cameras.push_back(scene.addCamera({.owner = {options.prop, node.camera}, .payload = model.cameras[size_t(node.camera)]}));
      scene.setEntityCamera(e, cameras.back());
    }
    for(auto it = node.children.rbegin(); it != node.children.rend(); ++it)
      stack.push_back({*it, e});
  }

  // skins: joints as entities, inverse binds from their accessor (identity when there is none)
  for(size_t si = 0; si < skins.size(); si++)
  {
    if(skins[si].isNone())
      continue;
    const tinygltf::Skin& skin = model.skins[si];
    SkinRecord            record{.name = skin.name};
    for(int j : skin.joints)
      record.joints.push_back(j >= 0 && j < int(out.entities.size()) ? out.entities[size_t(j)] : EntityId{});
    if(skin.inverseBindMatrices >= 0 && skin.inverseBindMatrices < int(model.accessors.size()))
    {
      std::vector<glm::mat4>     storage;
      std::span<const glm::mat4> ibm = tinygltf::utils::getAccessorData(model, model.accessors[size_t(skin.inverseBindMatrices)], &storage);
      record.inverseBind.assign(ibm.begin(), ibm.end());
    }
    record.inverseBind.resize(record.joints.size(), glm::mat4(1.f));
    if(skin.skeleton >= 0 && skin.skeleton < int(out.entities.size()))
      record.skeleton = out.entities[size_t(skin.skeleton)];
    scene.setSkin(skins[si], std::move(record));
  }

  // the import lets go of what it made: the entities hold what they draw, the rest goes
  for(const auto& meshPlans : plans)
    for(const PartPlan& plan : meshPlans)
      scene.release(plan.mesh);
  for(MaterialId m : out.materials)
    if(!m.isNone())
      scene.release(m);
  if(!defaultMaterial.isNone())
    scene.release(defaultMaterial);
  for(TextureId t : images)
    scene.release(t);
  for(SkinId s : skins)
    if(!s.isNone())
      scene.release(s);
  for(LightId l : lights)
    scene.release(l);
  for(CameraId c : cameras)
    scene.release(c);
  // what nothing kept is gone: say so in the result
  for(MaterialId& m : out.materials)
    if(!scene.material(m))
      m = {};
  for(auto& meshes : out.meshes)
    std::erase_if(meshes, [&](MeshId id) { return !scene.mesh(id); });
  return out;
}

}  // namespace rs
