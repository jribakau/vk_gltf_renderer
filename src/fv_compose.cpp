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

#include "fv_compose.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

#include <tinyobjloader/tiny_obj_loader.h>

#include "gltf_scene.hpp"
#include "tinygltf_converter.hpp"
#include "tinygltf_utils.hpp"

namespace fv::composer {

namespace {

void useExtension(tinygltf::Model& m, const char* name)
{
  if(std::find(m.extensionsUsed.begin(), m.extensionsUsed.end(), name) == m.extensionsUsed.end())
    m.extensionsUsed.push_back(name);
}

void setFvId(tinygltf::Value& extras, const std::string& id)
{
  tinygltf::Value::Object o = extras.IsObject() ? extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
  o["fvId"]                 = tinygltf::Value(id);
  extras                    = tinygltf::Value(std::move(o));
}

// "n:5" becomes "p:3/n:5"
void prefixFvId(tinygltf::Value& extras, const std::string& prefix)
{
  if(!extras.IsObject() || !extras.Has("fvId") || !extras.Get("fvId").IsString())
    return;
  setFvId(extras, prefix + "/" + extras.Get("fvId").Get<std::string>());
}

// Whether making `parent` the parent of `part` would close a loop.
bool cycles(const std::vector<Part>& parts, int part, int parent)
{
  for(int p = parent, steps = 0; p >= 0 && p < int(parts.size()) && steps <= int(parts.size()); p = parts[size_t(p)].parent, steps++)
    if(p == part)
      return true;
  return false;
}

// An image URI that is a path, relative to `dir`, made absolute; data URIs and embedded images stay.
void absoluteImages(tinygltf::Model& model, const std::filesystem::path& dir)
{
  for(tinygltf::Image& img : model.images)
  {
    if(img.uri.empty() || img.uri.rfind("data:", 0) == 0)
      continue;
    const std::string     decoded = tinygltf::utils::decodePathFromUri(img.uri);
    std::filesystem::path path(std::u8string(decoded.begin(), decoded.end()));
    if(path.is_absolute())
      continue;
    std::error_code ec;
    path                  = std::filesystem::weakly_canonical(dir / path, ec);
    const std::u8string s = path.generic_u8string();
    img.uri               = tinygltf::utils::encodePathAsUri(std::string(s.begin(), s.end()));
  }
}

}  // namespace

tinygltf::Model compose(const std::vector<Part>& parts, std::vector<Placed>& placed)
{
  tinygltf::Model out;
  out.asset.version   = "2.0";
  out.asset.generator = "Figure Viewer";
  out.scenes.emplace_back();
  out.defaultScene = 0;
  placed.assign(parts.size(), {});

  // the prop roots first, so their indices are the parts'
  for(size_t i = 0; i < parts.size(); i++)
  {
    const Part&    part = parts[i];
    tinygltf::Node node;
    node.name        = part.name;
    node.translation = {part.transform.translation.x, part.transform.translation.y, part.transform.translation.z};
    node.rotation    = {part.transform.rotation.x, part.transform.rotation.y, part.transform.rotation.z, part.transform.rotation.w};
    node.scale       = {part.transform.scale.x, part.transform.scale.y, part.transform.scale.z};
    setFvId(node.extras, part.id);
    if(!part.visible)
    {
      tinygltf::utils::setNodeVisibility(node, {false});
      useExtension(out, KHR_NODE_VISIBILITY_EXTENSION_NAME);
    }
    out.nodes.push_back(std::move(node));
    placed[i].root = int(i);
  }
  for(size_t i = 0; i < parts.size(); i++)
  {
    const int parent = parts[i].parent;
    if(parent >= 0 && parent < int(parts.size()) && parent != int(i) && !cycles(parts, int(i), parent))
      out.nodes[size_t(parent)].children.push_back(int(i));
    else
      out.scenes[0].nodes.push_back(int(i));
  }

  for(size_t i = 0; i < parts.size(); i++)
  {
    const Part& part = parts[i];
    Placed&     at   = placed[i];
    at.base          = nvvkgltf::SceneMerger::offsetsOf(out);
    if(part.model && !part.model->nodes.empty() && !part.model->scenes.empty())
    {
      const nvvkgltf::MergeResult r = nvvkgltf::SceneMerger::mergeUnder(out, *part.model, [&](const tinygltf::Node&) { return int(i); }, 0);
      if(r.valid())
      {
        at.firstNode = r.firstNode;
        at.lastNode  = r.lastNode;
        for(int n = r.firstNode; n < r.lastNode; n++)
          prefixFvId(out.nodes[size_t(n)].extras, part.id);
        for(size_t m = size_t(at.base.materials); m < out.materials.size(); m++)
          prefixFvId(out.materials[m].extras, part.id);
        for(size_t m = size_t(at.base.meshes); m < out.meshes.size(); m++)
          prefixFvId(out.meshes[m].extras, part.id);
      }
    }
    else if(part.camera)
    {
      at.camera = int(out.cameras.size());
      out.cameras.push_back(*part.camera);
      out.nodes[i].camera = at.camera;
    }
    else if(part.light)
    {
      at.light = int(out.lights.size());
      out.lights.push_back(*part.light);
      out.nodes[i].light = at.light;
      useExtension(out, "KHR_lights_punctual");
    }
  }
  return out;
}

int partOfNode(const std::vector<Placed>& placed, int node)
{
  if(node < 0)
    return -1;
  for(size_t i = 0; i < placed.size(); i++)
    if(placed[i].root == node || (node >= placed[i].firstNode && node < placed[i].lastNode))
      return int(i);
  return -1;
}

bool readModelFile(const std::filesystem::path& file, tinygltf::Model& model, std::string& error)
{
  std::string ext = file.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  const std::u8string u8   = file.u8string();
  const std::string   utf8(u8.begin(), u8.end());
  model = {};
  if(ext == ".obj")
  {
    tinyobj::ObjReaderConfig config;
    const std::u8string      dir = file.parent_path().u8string();
    config.mtl_search_path       = std::string(dir.begin(), dir.end());
    tinyobj::ObjReader reader;
    if(!reader.ParseFromFile(utf8, config))
    {
      error = reader.Error().empty() ? "not an OBJ file" : reader.Error();
      return false;
    }
    TinyConverter converter;
    converter.convert(model, reader);
  }
  else
  {
    tinygltf::TinyGLTF loader;
    std::string        warn;
    loader.SetMaxExternalFileSize(-1);
    // the renderer decodes images itself, from their files; embedded ones keep their bytes
    loader.SetImageLoader(
        [](tinygltf::Image* image, const int, std::string*, std::string*, int, int, const unsigned char* bytes, int size, void*) {
          if(bytes && size > 0)
            image->image.assign(bytes, bytes + size);
          return true;
        },
        nullptr);
    const bool ok = ext == ".glb" ? loader.LoadBinaryFromFile(&model, &error, &warn, utf8) : loader.LoadASCIIFromFile(&model, &error, &warn, utf8);
    if(!ok)
    {
      if(error.empty())
        error = "not a glTF file";
      return false;
    }
    // meshopt-compressed buffer views only the renderer's own loader unpacks (a merge skips them)
    const auto uses = [&](const char* name) {
      return std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), name) != model.extensionsUsed.end();
    };
    if(uses("EXT_meshopt_compression") || uses("KHR_meshopt_compression"))
    {
      nvvkgltf::Scene scene;
      if(!scene.load(file))
      {
        error = "its compressed meshes could not be read";
        return false;
      }
      model = scene.getModel();
    }
  }
  // a primitive with no material gets the model's own default one, as an OBJ does: the renderer's
  // fallback, the scene's material 0, is another prop's in a composed scene
  int fallback = -1;
  for(tinygltf::Mesh& mesh : model.meshes)
    for(tinygltf::Primitive& prim : mesh.primitives)
      if(prim.material < 0 || prim.material >= int(model.materials.size()))
      {
        if(fallback < 0)
        {
          fallback = int(model.materials.size());
          model.materials.emplace_back();
          model.materials.back().name = "Default";
        }
        prim.material = fallback;
      }
  if(model.scenes.empty() && !model.nodes.empty())
  {
    // roots are the nodes no other node holds
    std::vector<bool> child(model.nodes.size(), false);
    for(const tinygltf::Node& n : model.nodes)
      for(int c : n.children)
        if(c >= 0 && c < int(child.size()))
          child[size_t(c)] = true;
    model.scenes.emplace_back();
    for(size_t i = 0; i < child.size(); i++)
      if(!child[i])
        model.scenes[0].nodes.push_back(int(i));
  }
  absoluteImages(model, std::filesystem::absolute(file).parent_path());
  return true;
}

bool packForGlb(tinygltf::Model& model, std::string& error)
{
  tinygltf::Buffer packed;
  const auto       align = [&] {
    while(packed.data.size() % 4)
      packed.data.push_back(0);
  };
  std::vector<size_t> offsetOf;
  for(const tinygltf::Buffer& b : model.buffers)
  {
    align();
    offsetOf.push_back(packed.data.size());
    packed.data.insert(packed.data.end(), b.data.begin(), b.data.end());
  }
  for(tinygltf::BufferView& v : model.bufferViews)
    if(v.buffer >= 0 && v.buffer < int(offsetOf.size()))
    {
      v.byteOffset += offsetOf[size_t(v.buffer)];
      v.buffer = 0;
    }
  for(tinygltf::Image& img : model.images)
  {
    if(img.bufferView >= 0 || img.uri.empty())
    {
      img.image.clear();
      continue;
    }
    if(img.uri.rfind("data:", 0) == 0)
      continue;
    const std::string           decoded = tinygltf::utils::decodePathFromUri(img.uri);
    const std::filesystem::path path(std::u8string(decoded.begin(), decoded.end()));
    std::ifstream               in(path, std::ios::binary);
    if(!in)
    {
      error = "could not read " + decoded;
      return false;
    }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    align();
    tinygltf::BufferView view;
    view.buffer     = 0;
    view.byteOffset = packed.data.size();
    view.byteLength = bytes.size();
    packed.data.insert(packed.data.end(), bytes.begin(), bytes.end());
    img.bufferView = int(model.bufferViews.size());
    model.bufferViews.push_back(view);
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    img.mimeType = ext == ".png" ? "image/png" : ext == ".ktx2" ? "image/ktx2" : ext == ".webp" ? "image/webp" : "image/jpeg";
    img.uri.clear();
    img.image.clear();
  }
  model.buffers.assign(1, std::move(packed));
  return true;
}

tinygltf::Camera perspectiveCamera(double focalLengthMm, double aspect)
{
  aspect             = std::max(aspect, 1e-3);
  const double film  = 35.0 / std::max(aspect, 1.0);  // the frame's height on 35 mm film, as FvView's lens
  tinygltf::Camera c;
  c.type                    = "perspective";
  c.perspective.yfov        = 2.0 * std::atan(film / 2.0 / std::max(focalLengthMm, 1.0));
  c.perspective.aspectRatio = aspect;
  c.perspective.znear       = 0.01;
  c.perspective.zfar        = 1000.0;
  return c;
}

}  // namespace fv::composer
