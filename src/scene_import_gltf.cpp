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

#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>

#include <glm/gtc/type_ptr.hpp>
#include <nvutils/file_operations.hpp>

#include "ies_profile.hpp"
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

using Notice = std::function<void(const std::string&)>;

// A bufferView's bytes; none when it or its buffer is out of range (an extension's indices are
// untrusted)
std::optional<std::span<const unsigned char>> bufferViewBytes(const tinygltf::Model& model, int view)
{
  if(view < 0 || size_t(view) >= model.bufferViews.size())
    return std::nullopt;
  const tinygltf::BufferView& bv = model.bufferViews[size_t(view)];
  if(bv.buffer < 0 || size_t(bv.buffer) >= model.buffers.size()
     || bv.byteOffset + bv.byteLength > model.buffers[size_t(bv.buffer)].data.size())
    return std::nullopt;
  return std::span<const unsigned char>(model.buffers[size_t(bv.buffer)].data).subspan(bv.byteOffset, bv.byteLength);
}

// EXT_mesh_opacity_micromap's root micromaps, checked as SceneOmm::create checks them, and for at
// least one triangle record at a stride Vulkan takes (a multiple of 4, at least 8); null for one
// left out
std::vector<std::shared_ptr<const OpacityMicromap>> opacityMicromaps(const tinygltf::Model& model, const Notice& notice)
{
  std::vector<std::shared_ptr<const OpacityMicromap>> out;
  const tinygltf::Value* root = tinygltf::utils::findExtension(model.extensions, EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME);
  if(!root || !root->Has("micromaps") || !root->Get("micromaps").IsArray())
    return out;
  const tinygltf::Value& list = root->Get("micromaps");
  out.resize(list.ArrayLen());
  for(size_t i = 0; i < out.size(); i++)
  {
    const tinygltf::Value& def = list.Get(int(i));
    const std::string at = std::string(EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME) + ": micromaps[" + std::to_string(i) + "] ";
    if(!def.Has("data") || !def.Has("triangles") || !def.Has("usageCounts") || !def.Has("usageLevels") || !def.Has("usageFormats"))
    {
      notice(at + "misses a required field; left out");
      continue;
    }
    const int  trianglesView = def.Get("triangles").GetNumberAsInt();
    const auto data          = bufferViewBytes(model, def.Get("data").GetNumberAsInt());
    const auto triangles     = bufferViewBytes(model, trianglesView);
    if(!data || !triangles)
    {
      notice(at + "names a data or triangles bufferView out of range; left out");
      continue;
    }
    const tinygltf::Value& counts  = def.Get("usageCounts");
    const tinygltf::Value& levels  = def.Get("usageLevels");
    const tinygltf::Value& formats = def.Get("usageFormats");
    if(!counts.IsArray() || !levels.IsArray() || !formats.IsArray() || counts.ArrayLen() != levels.ArrayLen()
       || counts.ArrayLen() != formats.ArrayLen())
    {
      notice(at + "has usage arrays of different lengths; left out");
      continue;
    }
    const size_t stride = model.bufferViews[size_t(trianglesView)].byteStride;
    auto         mm     = std::make_shared<OpacityMicromap>();
    mm->triangleStride  = stride ? uint32_t(stride) : 8u;
    if(mm->triangleStride < 8 || mm->triangleStride % 4 || triangles->size() < mm->triangleStride)
    {
      notice(at + "has no triangle records at a stride Vulkan takes; left out");
      continue;
    }
    mm->data.assign(data->begin(), data->end());
    mm->triangles.assign(triangles->begin(), triangles->end());
    for(size_t u = 0; u < counts.ArrayLen(); u++)
      mm->usages.push_back({.count            = uint32_t(counts.Get(int(u)).GetNumberAsInt()),
                            .subdivisionLevel = uint32_t(levels.Get(int(u)).GetNumberAsInt()),
                            .format           = uint32_t(formats.Get(int(u)).GetNumberAsInt())});
    out[i] = std::move(mm);
  }
  return out;
}

// A primitive's EXT_mesh_opacity_micromap link over its `triangles` triangles, checked as
// SceneOmm::create checks it, and for an index per triangle that names a triangle of the micromap
// (a special index aside); null for none, or one left out
std::shared_ptr<const MicromapUse> micromapUseOf(const tinygltf::Model&                                     model,
                                                 const tinygltf::Primitive&                                 p,
                                                 size_t                                                     triangles,
                                                 const std::vector<std::shared_ptr<const OpacityMicromap>>& micromaps,
                                                 const Notice&                                              notice)
{
  const tinygltf::Value* ext = tinygltf::utils::findExtension(p.extensions, EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME);
  if(!ext || !ext->Has("micromap"))
    return nullptr;
  const std::string at    = std::string(EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME) + ": ";
  const int         index = ext->Get("micromap").GetNumberAsInt();
  if(index < 0 || size_t(index) >= micromaps.size())
  {
    notice(at + "a primitive names a micromap out of range; drawn without it");
    return nullptr;
  }
  if(!micromaps[size_t(index)])
    return nullptr;  // the micromap's own notice says why
  const int base = ext->Has("micromapBaseTriangle") ? ext->Get("micromapBaseTriangle").GetNumberAsInt() : 0;
  if(base < 0)
  {
    notice(at + "a primitive has a negative micromapBaseTriangle; drawn without its micromap");
    return nullptr;
  }
  auto use               = std::make_shared<MicromapUse>();
  use->micromap          = micromaps[size_t(index)];
  use->baseTriangle      = uint32_t(base);
  const uint64_t records = use->micromap->triangles.size() / use->micromap->triangleStride;
  if(!ext->Has("micromapIndices"))
  {
    if(uint64_t(base) + triangles > records)
    {
      notice(at + "a primitive's triangles run past its micromap's; drawn without it");
      return nullptr;
    }
    return use;
  }
  const int accessorIndex = ext->Get("micromapIndices").GetNumberAsInt();
  if(accessorIndex < 0 || size_t(accessorIndex) >= model.accessors.size())
  {
    notice(at + "a primitive names a micromapIndices accessor out of range; drawn without its micromap");
    return nullptr;
  }
  const tinygltf::Accessor& accessor = model.accessors[size_t(accessorIndex)];
  const int                 bytes    = tinygltf::GetComponentSizeInBytes(uint32_t(accessor.componentType));
  if(bytes != 1 && bytes != 2 && bytes != 4)
  {
    notice(at + "a primitive's micromapIndices are not 8, 16 or 32-bit integers; drawn without its micromap");
    return nullptr;
  }
  const size_t size   = size_t(bytes);
  const auto   view   = bufferViewBytes(model, accessor.bufferView);
  const size_t stride = view && model.bufferViews[size_t(accessor.bufferView)].byteStride ?
                            size_t(model.bufferViews[size_t(accessor.bufferView)].byteStride) :
                            size;
  if(!view || accessor.count < triangles
     || (accessor.count > 0 && accessor.byteOffset + (accessor.count - 1) * stride + size > view->size()))
  {
    notice(at + "a primitive's micromapIndices are fewer than its triangles, or run past their bufferView; drawn without its micromap");
    return nullptr;
  }
  use->indexComponentType = accessor.componentType;
  use->indices.resize(accessor.count * size);
  for(size_t e = 0; e < accessor.count; e++)  // tightly packed
    std::memcpy(&use->indices[e * size], view->data() + accessor.byteOffset + e * stride, size);
  const uint64_t special = (uint64_t(1) << (8 * size)) - 4;  // -4 to -1 in the type's bits
  for(size_t t = 0; t < triangles; t++)
  {
    uint32_t v = 0;
    std::memcpy(&v, &use->indices[t * size], size);  // little-endian
    if(v < special && uint64_t(base) + v >= records)
    {
      notice(at + "a primitive's micromapIndices name triangles past its micromap's; drawn without it");
      return nullptr;
    }
  }
  return use;
}

}  // namespace

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

TextureRecord textureRecordOf(const tinygltf::Model& model, size_t image, std::span<const std::filesystem::path> imageSearchPaths)
{
  const tinygltf::Image& img = model.images[image];
  TextureRecord          t;
  // an image a texture names through an extension (a KTX2 under KHR_texture_basisu, a WebP) keeps
  // that extension, so an export names it the same way
  for(const tinygltf::Texture& texture : model.textures)
    for(const char* ext : {KHR_TEXTURE_BASISU_EXTENSION_NAME, EXT_TEXTURE_WEBP_EXTENSION_NAME, "MSFT_texture_dds"})
      if(auto it = texture.extensions.find(ext); it != texture.extensions.end() && it->second.Has("source"))
        if(it->second.Get("source").GetNumberAsInt() == int(image))
          t.sourceExtension = ext;
  t.name     = img.name;
  t.uri      = img.uri;
  t.mimeType = img.mimeType;
  t.width    = img.width;
  t.height   = img.height;
  // a data URI's bytes or pixels are in img.image (below), never the URI itself
  if(img.uri.rfind("data:", 0) == 0)
    t.uri.clear();
  else if(!t.uri.empty())
  {
    // a URI is %-encoded (fv_compose writes absolute ones so); the record holds the file's path
    const std::filesystem::path path = nvutils::pathFromUtf8(tinygltf::utils::decodePathFromUri(t.uri));
    std::error_code             ec;
    std::filesystem::path       found;
    if(path.is_absolute() && std::filesystem::is_regular_file(path, ec))
      found = path;
    else if(!imageSearchPaths.empty())
      found = nvutils::findFile(path, std::vector<std::filesystem::path>(imageSearchPaths.begin(), imageSearchPaths.end()), false);
    if(!found.empty())
      t.uri = nvutils::utf8FromPath(std::filesystem::absolute(found));
  }
  if(t.uri.empty() && img.bufferView < 0 && !img.image.empty())
  {
    // the renderer's loaders keep the encoded file (as_is, or no size decoded); tinygltf's own
    // decodes it to pixels
    const size_t decoded =
        img.width > 0 && img.height > 0 && img.component > 0 ?
            size_t(img.width) * size_t(img.height) * size_t(img.component) * size_t(std::max(img.bits, 8) / 8) :
            0;
    if(!img.as_is && decoded == img.image.size())
    {
      t.pixels    = img.image;
      t.component = img.component;
      t.bits      = img.bits;
    }
    else
      t.bytes = img.image;
  }
  if(t.uri.empty() && img.bufferView >= 0 && img.bufferView < int(model.bufferViews.size()))
  {
    const tinygltf::BufferView& view = model.bufferViews[size_t(img.bufferView)];
    if(view.buffer >= 0 && view.buffer < int(model.buffers.size())
       && view.byteOffset + view.byteLength <= model.buffers[size_t(view.buffer)].data.size())
    {
      const auto begin = model.buffers[size_t(view.buffer)].data.begin() + std::ptrdiff_t(view.byteOffset);
      t.bytes.assign(begin, begin + std::ptrdiff_t(view.byteLength));
    }
  }
  return t;
}

namespace {

// A sampler's keys as AnimationSystem::parseSamplersAndChannels reads them (an interpolation it
// does not name is linear); nothing when its accessors can't be read.
std::optional<ClipSampler> clipSampler(const tinygltf::Model& model, const tinygltf::AnimationSampler& s)
{
  if(s.input < 0 || s.input >= int(model.accessors.size()) || s.output < 0 || s.output >= int(model.accessors.size()))
    return {};
  ClipSampler out;
  out.interpolation = s.interpolation == "STEP"        ? ClipInterpolation::eStep :
                      s.interpolation == "CUBICSPLINE" ? ClipInterpolation::eCubicSpline :
                                                         ClipInterpolation::eLinear;
  if(!tinygltf::utils::copyAccessorData(model, model.accessors[size_t(s.input)], out.inputs))
    return {};
  const tinygltf::Accessor& output = model.accessors[size_t(s.output)];
  const auto                flat   = [&]<class T>(T) {
    std::vector<T> values;
    if(!tinygltf::utils::copyAccessorData(model, output, values))
      return false;
    const float* f = reinterpret_cast<const float*>(values.data());
    out.outputs.assign(f, f + values.size() * sizeof(T) / sizeof(float));
    return true;
  };
  switch(output.type)
  {
    case TINYGLTF_TYPE_VEC3:
      return flat(glm::vec3()) ? std::optional(std::move(out)) : std::nullopt;
    case TINYGLTF_TYPE_VEC4:
      return flat(glm::vec4()) ? std::optional(std::move(out)) : std::nullopt;
    case TINYGLTF_TYPE_SCALAR:
      return flat(float()) ? std::optional(std::move(out)) : std::nullopt;
  }
  return {};
}

}  // namespace

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
  for(const char* ext : {KHR_INTERACTIVITY_EXTENSION_NAME, KHR_ANIMATION_POINTER})
    if(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), ext) != model.extensionsUsed.end())
      notice(std::string(ext) + " is not drawn yet; left out");

  // ---- EXT_lights_ies profiles, read and sampled here as the legacy scene reads them; by index
  std::vector<uint32_t> iesProfiles;
  {
    std::vector<std::string> warnings;
    for(nvvkgltf::IesProfileData& profile : nvvkgltf::loadIesProfiles(model, options.imageSearchPaths, warnings))
      iesProfiles.push_back(scene.addIesProfile(
          {.name = std::move(profile.name), .bytes = std::move(profile.bytes), .samples = std::move(profile.samples)}));
    for(const std::string& warning : warnings)
      notice(warning);
  }

  // ---- images and the texture table every material of this file shares
  std::vector<TextureId> images(model.images.size());
  for(size_t i = 0; i < model.images.size(); i++)
    images[i] = scene.addTexture(textureRecordOf(model, i, options.imageSearchPaths));
  auto table = std::make_shared<TextureTable>();
  for(const tinygltf::Texture& texture : model.textures)
  {
    const int source = tinygltf::utils::getTextureImageIndex(texture);
    table->push_back({source >= 0 && source < int(images.size()) ? images[size_t(source)] : TextureId{}, samplerOf(model, texture.sampler)});
  }
  out.images   = images;
  out.textures = table;

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

  // ---- EXT_mesh_opacity_micromap's micromaps, shared by the submeshes that index them
  const std::vector<std::shared_ptr<const OpacityMicromap>> micromaps = opacityMicromaps(model, notice);

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
      r.submeshes.push_back({.firstIndex = uint32_t(r.indices.size()),
                             .indexCount = uint32_t(idx.size()),
                             .material   = uint32_t(r.submeshes.size()),
                             .order      = int32_t(pi),
                             .micromap   = micromapUseOf(model, p, idx.size() / 3, micromaps, notice)});
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
    // a light per node: its KHR light, or EXT_lights_ies, which stands alone when both are there
    const EXT_lights_ies_ref ies      = tinygltf::utils::getNodeIesLight(node);
    const bool               hasIes   = ies.light >= 0 && ies.light < int(iesProfiles.size());
    const bool               hasLight = node.light >= 0 && node.light < int(model.lights.size());
    if(ies.light >= 0 && !hasIes)
      notice("EXT_lights_ies lights naming no profile");
    if(hasIes || hasLight)
    {
      LightRecord light{.owner   = {options.prop, node.light},
                        .payload = hasLight ? model.lights[size_t(node.light)] : tinygltf::Light{}};
      if(hasIes)
        light.ies = IesLight{.profile = iesProfiles[size_t(ies.light)], .multiplier = ies.multiplier, .color = ies.color};
      lights.push_back(scene.addLight(std::move(light)));
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

  // ---- clips: each animation's channels on the scene's nodes, with the samplers they use. A
  // KHR_animation_pointer channel is left out (D23, noticed above), as is one whose node is not in
  // the scene (it moves nothing drawn) or whose values are not its path's type (which
  // AnimationSystem leaves alone too)
  out.clips.assign(model.animations.size(), {});
  for(size_t ai = 0; ai < model.animations.size(); ai++)
  {
    const tinygltf::Animation& animation = model.animations[ai];
    ClipRecord                 clip{.name = animation.name, .owner = {options.prop, int32_t(ai)}};
    std::map<int, uint32_t>    samplerOf;  // the animation's sampler: the clip's
    for(const tinygltf::AnimationChannel& c : animation.channels)
    {
      static const std::map<std::string, std::pair<ClipPath, int>> paths = {
          {"translation", {ClipPath::eTranslation, TINYGLTF_TYPE_VEC3}},
          {"rotation", {ClipPath::eRotation, TINYGLTF_TYPE_VEC4}},
          {"scale", {ClipPath::eScale, TINYGLTF_TYPE_VEC3}},
          {"weights", {ClipPath::eWeights, TINYGLTF_TYPE_SCALAR}},
      };
      const auto path = paths.find(c.target_path);
      if(path == paths.end() || c.target_node < 0 || c.target_node >= int(out.entities.size())
         || out.entities[size_t(c.target_node)].isNone() || c.sampler < 0 || c.sampler >= int(animation.samplers.size()))
        continue;
      const int output = animation.samplers[size_t(c.sampler)].output;
      if(output < 0 || output >= int(model.accessors.size()) || model.accessors[size_t(output)].type != path->second.second)
        continue;
      auto it = samplerOf.find(c.sampler);
      if(it == samplerOf.end())
      {
        std::optional<ClipSampler> sampler = clipSampler(model, animation.samplers[size_t(c.sampler)]);
        if(!sampler)
          continue;
        it = samplerOf.emplace(c.sampler, uint32_t(clip.samplers.size())).first;
        clip.samplers.push_back(std::move(*sampler));
      }
      clip.channels.push_back({.target = out.entities[size_t(c.target_node)], .path = path->second.first, .sampler = it->second});
    }
    if(!clip.channels.empty())
      out.clips[ai] = scene.addClip(std::move(clip));
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
