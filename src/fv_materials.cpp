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

#include "fv_materials.hpp"
#include "fv_model_material.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>

#include <nvutils/logger.hpp>

#include <figure/dson/material_preset.hpp>
#include <figure/dson/skin_materials.hpp>
#include <figure/materials/edits_json.hpp>
#include <figure/materials/lobes.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "host_renderer.hpp"
#include "tinygltf_utils.hpp"

using fv::Json;
using fv::mat::Rgb;

namespace {

const char* kindName(fv::mat::Kind k)
{
  using K = fv::mat::Kind;
  switch(k)
  {
    case K::eSkin: return "skin";
    case K::eCornea: return "cornea";
    case K::eEye: return "eye";
    case K::eLashes: return "lashes";
    case K::eNails: return "nails";
    case K::eTeeth: return "teeth";
    case K::eMouth: return "mouth";
    case K::eHair: return "hair";
    case K::eOther: break;
  }
  return "other";
}

const char* shaderName(fv::mat::Surface::Shader s)
{
  switch(s)
  {
    case fv::mat::Surface::Shader::eIray: return "iray";
    case fv::mat::Surface::Shader::ePbrSkin: return "pbrskin";
    case fv::mat::Surface::Shader::eDazDefault: return "dazdefault";
    case fv::mat::Surface::Shader::eUberSurface: return "ubersurface";
    case fv::mat::Surface::Shader::eAoa: return "aoa";
    case fv::mat::Surface::Shader::eLegacy: break;
  }
  return "legacy";
}

using fv::mat::applyValue;
using fv::mat::editedKeys;
using fv::mat::hexOf;

std::string baseName(const std::string& rel)
{
  const size_t slash = rel.find_last_of("/\\");
  return slash == std::string::npos ? rel : rel.substr(slash + 1);
}

// Which of an eye's colours a surface takes: an iris colour where the bake knows where the iris is
// (a Genesis 8 iris map needs its discs; a flat iris takes it as its colour), a sclera colour on a
// white of its own or one shared with the iris.
struct EyeRoles
{
  bool iris = false, sclera = false;
};

EyeRoles eyeRoles(const fv::mat::Surface& s, const fv::mat::LowerOptions* o)
{
  using L            = fv::mat::EyeLayout;
  const L    layout  = fv::mat::eyeLayout(s);
  const bool located = layout == L::eShared || (layout == L::eIris && (s.base.map.empty() || (o && !o->irisDiscs.empty())));
  return {located, layout == L::eShared || layout == L::eSclera};
}

std::filesystem::path pathOfUri(const std::string& uri)
{
  const std::string utf8 = fv::dson::decodeUri(uri);
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

}  // namespace

FvMaterials::FvMaterials(Hooks hooks)
    : m_hooks(std::move(hooks))
    , m_alive(std::make_shared<std::atomic<bool>>(true))
{
}

FvMaterials::~FvMaterials()
{
  *m_alive = false;
  m_threads.joinAll();
}

void FvMaterials::attach(const std::string&                 figureId,
                         fv::SceneGraph*                    scene,
                         std::vector<fv::mat::LowerOptions> lowering,
                         bool                               texturesFlipped,
                         fv::mat::SkinMode                  skin,
                         const std::set<int>&               kept)
{
  m_generation++;
  // the feed lowered every surface as it is now
  m_heldChanges.clear();
  // bakes still running were for the scene before this one, whose result is dropped: their materials ask again
  std::set<std::string> inflight;
  if(m_figure == figureId && !m_model)
    for(const auto& [key, unwanted] : m_baking)
      inflight.insert(key);
  m_model                              = false;
  m_figure                             = figureId;
  m_scene                              = scene;
  m_options                            = std::move(lowering);
  m_options.resize(scene->materials.size());
  m_failed.clear();
  forgetBakes();
  m_waiting.clear();
  m_flipped = texturesFlipped;
  m_skin    = skin;
  m_applied.assign(scene->materials.size(), {});
  m_count = scene->materials.size();
  std::set<int> again;
  for(size_t mi = 0; mi < scene->materials.size(); mi++)
  {
    const fv::mat::Lowered lowered = fv::mat::lower(scene->materials[mi].surface, scene->materials[mi].edits, m_options[mi]);
    m_applied[mi]                  = stateOf(lowered);
    for(const fv::mat::TextureRequest& r : lowered.requests)
    {
      if(inflight.count(fv::mat::requestKey(r)) && !(m_hooks.resolves && m_hooks.resolves(r)))
      {
        m_applied[mi].clear();
        again.insert(int(mi));
      }
    }
  }
  for(int mi : kept)
    if(mi >= 0 && mi < int(m_count))
    {
      m_applied[size_t(mi)].clear();
      again.insert(mi);
    }
  mapMeshes();
  publish({}, true);
  if(!again.empty())
    refresh(again);
}

void FvMaterials::extend(std::vector<fv::mat::LowerOptions> lowering, const std::set<int>& kept)
{
  if(m_model || !m_scene)
    return;
  // what the feed wrote for every material, the new ones at the end: the edits, the bakes in flight
  // and what could not be made stay
  m_options = std::move(lowering);
  m_options.resize(m_scene->materials.size());
  m_applied.resize(m_scene->materials.size());
  for(size_t mi = 0; mi < m_scene->materials.size(); mi++)
    if(!m_waiting.count(int(mi)))
      m_applied[mi] = stateOf(fv::mat::lower(m_scene->materials[mi].surface, m_scene->materials[mi].edits, m_options[mi]));
  m_count = m_scene->materials.size();
  // what the feed left as it was isn't what was just lowered
  std::set<int> again;
  for(int mi : kept)
    if(mi >= 0 && mi < int(m_count) && !m_waiting.count(mi))
    {
      m_applied[size_t(mi)].clear();
      again.insert(mi);
    }
  mapMeshes();
  publish({}, true);
  if(!again.empty())
    refresh(again);
}

void FvMaterials::mapMeshes()
{
  m_meshesOf.assign(m_scene->materials.size(), {});
  for(size_t i = 0; i < m_scene->meshes.size(); i++)
  {
    const fv::Mesh& m = m_scene->meshes[i];
    for(int mi : m.materials)
      if(mi >= 0 && mi < int(m_meshesOf.size()))
        if(m_meshesOf[mi].empty() || m_meshesOf[mi].back() != int(i))
          m_meshesOf[mi].push_back(int(i));
  }
}

void FvMaterials::attachModel(const std::string& id, const tinygltf::Model& source)
{
  const int count = int(source.materials.size()), meshes = int(source.meshes.size());
  m_generation++;
  // the same model composed again keeps the edits made to it
  const bool again = m_model && m_figure == id;
  m_model          = true;
  m_figure         = id;
  m_options.clear();
  m_waiting.clear();
  forgetBakes();
  std::vector<fv::mat::Edits> edits;
  if(again)
    for(const fv::Material& m : m_modelScene.materials)
      edits.push_back(m.edits);
  m_modelScene = {};
  m_originals.clear();
  for(int i = 0; i < count; i++)
  {
    const tinygltf::Material& g = source.materials[size_t(i)];
    fv::Material              m;
    m.name = g.name.empty() ? "Material-" + std::to_string(i) : g.name;
    if(size_t(i) < edits.size())
      m.edits = edits[size_t(i)];
    m_modelScene.materials.push_back(std::move(m));
    m_originals.push_back(fv::mat::fromGltf(g));
  }
  m_modelImages.clear();
  for(const tinygltf::Texture& t : source.textures)
  {
    const tinygltf::Image* img = t.source >= 0 && t.source < int(source.images.size()) ? &source.images[size_t(t.source)] : nullptr;
    m_modelImages.push_back(img ? baseName(img->uri.empty() ? img->name : img->uri) : std::string());
  }
  m_scene = &m_modelScene;
  m_meshesOf.assign(m_modelScene.materials.size(), {});
  m_count = m_modelScene.materials.size();
  for(int i = 0; i < meshes; i++)
    for(const tinygltf::Primitive& p : source.meshes[size_t(i)].primitives)
    {
      const int mi = p.material;
      if(mi >= 0 && size_t(mi) < m_meshesOf.size() && (m_meshesOf[size_t(mi)].empty() || m_meshesOf[size_t(mi)].back() != i))
        m_meshesOf[size_t(mi)].push_back(i);
    }
  for(size_t i = 0; i < m_count; i++)
    if(!m_modelScene.materials[i].edits.empty())
      writeModel(int(i));
  publish({}, true);
}

// The original material with the edits laid over it.
void FvMaterials::writeModel(int index)
{
  if(index < 0 || index >= int(m_originals.size()) || !m_hooks.writeModel)
    return;
  m_hooks.writeModel(index, fv::model::editedMaterial(m_originals[size_t(index)], m_scene->materials[size_t(index)].edits));
}

const fv::mat::Edits* FvMaterials::modelEdits(int index) const
{
  if(!m_model || !m_scene || index < 0 || index >= int(m_scene->materials.size()))
    return nullptr;
  return &m_scene->materials[size_t(index)].edits;
}

Json FvMaterials::modelItem(int index) const
{
  const fv::Material&       m = m_scene->materials[size_t(index)];
  const fv::mat::Material   g = fv::model::editedMaterial(m_originals[size_t(index)], m.edits);  // as the scene has it
  Json                      values = {{"color", hexOf({g.baseColor[0], g.baseColor[1], g.baseColor[2]})},
                                      {"roughness", g.roughness},
                                      {"metalness", g.metallic},
                                      {"opacity", g.baseColor[3]},
                                      {"alphaMode", g.alphaMode == fv::mat::AlphaMode::eMask ? "cutout" : g.alphaMode == fv::mat::AlphaMode::eBlend ? "blend" : "opaque"},
                                      {"doubleSided", g.doubleSided},
                                      {"specular", g.specular ? g.specular->factor : 1.f},
                                      {"clearcoat", g.clearcoat ? g.clearcoat->factor : 0.f},
                                      {"clearcoatRoughness", g.clearcoat ? g.clearcoat->roughness : 0.f},
                                      {"emissive", hexOf(g.emissive)},
                                      {"emissiveIntensity", g.emissiveStrength.value_or(1.f)}};
  if(g.normalMap.has())
    values["normalStrength"] = g.normalMap.scale.value_or(1.f);
  // sheen: the edit as it was given (black included), else the model's own; null for none
  const std::optional<Rgb> sheen = m.edits.sheenColor ? m.edits.sheenColor : g.sheen ? std::optional<Rgb>(g.sheen->color) : std::nullopt;
  values["sheenColor"] = sheen && (m.edits.sheenColor || (*sheen)[0] > 0 || (*sheen)[1] > 0 || (*sheen)[2] > 0) ? Json(hexOf(*sheen)) : Json(nullptr);
  values["sheenRoughness"] = m.edits.sheenRoughness ? *m.edits.sheenRoughness : g.sheen ? g.sheen->roughness : 0.5f;
  // a model's images are its file's, not the library's: no rel, and none of them can be replaced
  Json textures = Json::array();
  const auto add      = [&](const char* slot, const char* label, int texture) {
    if(texture >= 0 && texture < int(m_modelImages.size()) && !m_modelImages[size_t(texture)].empty())
      textures.push_back({{"slot", slot}, {"label", label}, {"file", m_modelImages[size_t(texture)]}, {"edited", false}});
  };
  add("base", "Base colour", g.baseColorMap.index);
  add("roughness", "Metallic roughness", g.metallicRoughnessMap.index);
  add("normal", "Normal", g.normalMap.index);
  add("emission", "Emissive", g.emissiveMap.index);
  add("occlusion", "Occlusion", g.occlusionMap.index);
  // the values that multiply one of its maps
  Json       mapped = Json::array();
  const auto multiplies = [&](bool map, const char* key) {
    if(map)
      mapped.push_back(key);
  };
  multiplies(g.baseColorMap.has(), "color");
  multiplies(g.metallicRoughnessMap.has(), "roughness");
  multiplies(g.metallicRoughnessMap.has(), "metalness");
  multiplies(g.specular && g.specular->map.has(), "specular");
  multiplies(g.clearcoat && g.clearcoat->map.has(), "clearcoat");
  multiplies(g.clearcoat && g.clearcoat->roughnessMap.has(), "clearcoatRoughness");
  multiplies(g.emissiveMap.has(), "emissive");
  return {{"name", m.name}, {"kind", "other"}, {"shader", "gltf"}, {"meshes", m_meshesOf[size_t(index)]}, {"textures", textures}, {"values", values},
          {"edited", editedKeys(m.edits)}, {"mapped", mapped}};
}

void FvMaterials::detach()
{
  m_model = false;
  m_originals.clear();
  m_modelScene = {};
  m_generation++;
  m_scene = nullptr;
  m_count = 0;
  m_options.clear();
  m_modelImages.clear();
  m_applied.clear();
  m_meshesOf.clear();
  m_waiting.clear();
  forgetBakes();
  m_eyes = nullptr;
  m_skinState = nullptr;
  if(!m_figure.empty())
    m_hooks.events->drop("materials/" + m_figure);
  m_figure.clear();
}

std::string FvMaterials::stateOf(const fv::mat::Lowered& lowered) const
{
  // the material, and the textures its slots ask for (the JSON holds only their numbers)
  std::string text = fv::mat::materialJson(lowered.material, "slot").dump();
  for(const fv::mat::TextureRequest& r : lowered.requests)
    text += "\n" + fv::mat::requestKey(r);
  return text;
}

std::vector<int> FvMaterials::targets(const Json& args) const
{
  std::vector<int> out;
  const auto       one = [&](const Json& v) {
    int index = -1;
    if(v.is_number_integer())
      index = v.get<int>();
    else if(v.is_string() && v.get<std::string>().rfind("m:", 0) == 0)
      index = std::atoi(v.get<std::string>().c_str() + 2);
    if(!m_scene || index < 0 || index >= int(m_count))
      throw std::runtime_error("no such material " + v.dump());
    out.push_back(index);
  };
  const auto it = args.find("material");
  if(it == args.end())
    return out;
  if(it->is_array())
    for(const Json& v : *it)
      one(v);
  else
    one(*it);
  return out;
}

Json FvMaterials::set(const Json& args)
{
  if(!m_scene)
    throw std::runtime_error("no figure loaded");
  const std::vector<int> index = targets(args);
  if(index.empty())
    throw std::runtime_error("material.set wants a material");
  const auto values = args.find("values");
  if(values == args.end() || !values->is_object())
    throw std::runtime_error("material.set wants values: {name: value}");
  // check every value before any is applied
  {
    fv::mat::Edits scratch;
    for(auto it = values->begin(); it != values->end(); ++it)
      applyValue(scratch, it.key(), it.value());
    // an eye's colours only where the surface holds that part of the eye
    for(int mi : index)
    {
      const fv::Material& m     = m_scene->materials[mi];
      const EyeRoles      roles = m_model ? EyeRoles{} : eyeRoles(m.surface, mi < int(m_options.size()) ? &m_options[mi] : nullptr);
      if((scratch.irisColor || scratch.irisStrength) && !roles.iris)
        throw std::runtime_error(fv::mat::eyeLayout(m.surface) == fv::mat::EyeLayout::eIris ?
                                     "Where the irises of " + m.name + " are in its texture isn't known, so their colour can't be changed" :
                                     "irisColor is for an eye's iris, and " + m.name + " is not one");
      if(scratch.scleraColor && !roles.sclera)
        throw std::runtime_error("scleraColor is for the white of an eye, and " + m.name + " is not one");
      // a model's materials take their edits as they are, with no image to bake the colour into
      if(m_model && (scratch.recolor || scratch.recolorStrength || scratch.tone || scratch.toneStrength))
        throw std::runtime_error(std::string(scratch.tone || scratch.toneStrength ? "tone" : "recolor") + " is for a figure's surfaces, and " + m.name
                                 + " belongs to a model");
      if((scratch.makeupColor || scratch.makeupWeight) && (m_model || !m.surface.makeup))
        throw std::runtime_error("makeupColor is for a surface with a makeup layer, and " + m.name + " has none");
      // a model has no Surface to lower: its images, their tiling and a detail layer stay its file's
      if(m_model)
        for(auto it = values->begin(); it != values->end(); ++it)
          if(fv::mat::figureOnlyKey(it.key()))
            throw std::runtime_error(it.key() + " is for a figure's surfaces, and " + m.name + " belongs to a model");
    }
  }
  bool images = false;
  for(auto it = values->begin(); it != values->end(); ++it)
    images = images || fv::mat::mapKey(it.key()) || it.key() == "fabric";
  std::set<int> changed;
  for(int mi : index)
  {
    for(auto it = values->begin(); it != values->end(); ++it)
      applyValue(m_scene->materials[mi].edits, it.key(), it.value());
    changed.insert(mi);
    // an image chosen again is tried again, though it could not be made before
    if(images && !m_model && mi < int(m_options.size()))
      for(const fv::mat::TextureRequest& r : fv::mat::lower(m_scene->materials[mi].surface, m_scene->materials[mi].edits, m_options[mi]).requests)
        m_failed.erase(fv::mat::requestKey(r));
  }
  refresh(changed);
  return {{"ok", true}, {"materials", index.size()}};
}

Json FvMaterials::reset(const Json& args)
{
  if(!m_scene)
    throw std::runtime_error("no figure loaded");
  std::vector<int> index = targets(args);
  if(index.empty())
    for(size_t i = 0; i < m_count; i++)
      if(!m_scene->materials[i].edits.empty())
        index.push_back(int(i));
  std::set<int> changed;
  for(int mi : index)
    if(!m_scene->materials[mi].edits.empty())
    {
      m_scene->materials[mi].edits = {};
      changed.insert(mi);
    }
  refresh(changed);
  return {{"ok", true}, {"materials", changed.size()}};
}

namespace {

// "<name>", or "<name>#n" for the n-th material of that name in scene order, among those in use: a
// taken-off item's surfaces stay in the scene, but count for nothing (an empty key), so a file
// written after a take-off names its materials as the figure loaded fresh does.
std::vector<std::string> materialKeys(const fv::SceneGraph& scene)
{
  const std::vector<bool>    live = fv::dson::liveMaterials(scene);
  std::map<std::string, int> counts;
  std::vector<std::string>   keys;
  for(size_t i = 0; i < scene.materials.size(); i++)
  {
    const fv::Material& m = scene.materials[i];
    if(i < live.size() && !live[i])
    {
      keys.emplace_back();
      continue;
    }
    const int n = counts[m.name]++;
    keys.push_back(n > 0 ? m.name + "#" + std::to_string(n) : m.name);
  }
  return keys;
}

}  // namespace

Json FvMaterials::editState() const
{
  Json out = Json::object();
  if(!m_scene)
    return out;
  const auto keys = materialKeys(*m_scene);
  for(size_t i = 0; i < m_count; i++)
  {
    if(!keys[i].empty() && !m_scene->materials[i].edits.empty())
      out[keys[i]] = editsOf(int(i));
  }
  return out;
}

Json FvMaterials::editsOf(int index) const
{
  Json edited = Json::object();
  if(!m_scene || index < 0 || index >= int(m_count) || m_scene->materials[size_t(index)].edits.empty())
    return edited;
  return fv::mat::editedValues(m_scene->materials[size_t(index)].edits);
}

std::string FvMaterials::nameOf(int index) const
{
  return m_scene && index >= 0 && index < int(m_count) ? m_scene->materials[size_t(index)].name : std::string();
}

std::vector<std::string> FvMaterials::restoreEdits(const Json& state)
{
  std::vector<std::string> problems;
  if(!m_scene || !state.is_object())
    return problems;
  const auto keys = materialKeys(*m_scene);
  for(auto it = state.begin(); it != state.end(); ++it)
  {
    const auto at = std::find(keys.begin(), keys.end(), it.key());
    if(at == keys.end())
    {
      problems.push_back("material " + it.key() + " not found");
      continue;
    }
    try
    {
      set({{"material", int(at - keys.begin())}, {"values", it.value()}});
    }
    catch(const std::exception& e)
    {
      problems.push_back("material " + it.key() + ": " + e.what());
    }
  }
  return problems;
}

void FvMaterials::setEyes(Json eyes)
{
  m_eyes = std::move(eyes);
  if(m_scene && !m_model && !m_figure.empty())
    m_hooks.events->patch("materials/" + m_figure, {{"eyes", m_eyes}});
}

void FvMaterials::setSkin(Json skin)
{
  m_skinState = std::move(skin);
  if(m_scene && !m_model && !m_figure.empty())
    m_hooks.events->patch("materials/" + m_figure, {{"skin", m_skinState}});
}

void FvMaterials::surfacesChanged(const std::set<int>& materials)
{
  if(m_scene && !materials.empty())
    refresh(materials);
}

void FvMaterials::setSkinMode(fv::mat::SkinMode skin)
{
  if(skin == m_skin)
    return;
  m_skin = skin;
  if(!m_scene || m_model)
    return;
  // the ones that differ: refresh() skips what it would write unchanged
  std::set<int> all;
  for(size_t i = 0; i < m_count; i++)
    all.insert(int(i));
  refresh(all);
}

void FvMaterials::rewrite()
{
  if(!m_scene || m_model)
    return;
  std::set<int> all;
  for(size_t i = 0; i < m_count; i++)
    if(!m_waiting.count(int(i)))
      all.insert(int(i));
  write(all);
}

void FvMaterials::hold()
{
  m_held = true;
}

void FvMaterials::release(bool refreshNow)
{
  m_held = false;
  std::set<int> noted;
  noted.swap(m_heldChanges);
  if(refreshNow && m_scene && !noted.empty())
    refresh(noted);
}

void FvMaterials::refresh(const std::set<int>& materials)
{
  if(m_held)
  {
    m_heldChanges.insert(materials.begin(), materials.end());
    return;
  }
  if(m_model)
  {
    for(int mi : materials)
      if(mi >= 0 && mi < int(m_count))
        writeModel(mi);
    publish(materials, false);
    return;
  }
  std::vector<fv::build::TextureRequest> missing;
  std::set<std::string>                  asked;
  std::set<int>                          shown, written;
  // every skin surface tones against the same map, whichever of them changed
  const std::string toneReference = fv::dson::toneReference(*m_scene);
  for(int mi : materials)
  {
    if(mi < 0 || mi >= int(m_count))
      continue;
    m_options[mi].skin          = m_skin;
    m_options[mi].toneReference = toneReference;
    const fv::Material& m = m_scene->materials[mi];
    const fv::mat::Lowered     lowered = fv::mat::lower(m.surface, m.edits, m_options[mi]);
    std::set<std::string>      wants;
    for(const fv::mat::TextureRequest& r : lowered.requests)
    {
      const std::string key = fv::mat::requestKey(r);
      if((m_hooks.resolves && m_hooks.resolves(r)) || m_failed.count(key))
        continue;
      wants.insert(key);
      if(!m_baking.count(key) && asked.insert(key).second)
        missing.push_back(r);
    }
    shown.insert(mi);
    if(!wants.empty())
    {
      m_waiting.insert(mi);
      m_wants[mi] = std::move(wants);
      continue;
    }
    m_waiting.erase(mi);
    m_wants.erase(mi);
    const std::string state = stateOf(lowered);
    if(state != m_applied[mi])
    {
      written.insert(mi);
      m_applied[mi] = state;
    }
  }
  write(written);
  markUnwanted();
  if(!missing.empty())
    bake(std::move(missing));
  publish(shown, false);
}

// The figure's feed lowers them again, with the options they were lowered with here
void FvMaterials::write(const std::set<int>& materials)
{
  if(materials.empty() || !m_hooks.writeFigure)
    return;
  std::vector<std::pair<int, fv::mat::LowerOptions>> out;
  for(int mi : materials)
    out.emplace_back(mi, m_options[size_t(mi)]);
  m_hooks.writeFigure(std::move(out));
}

void FvMaterials::markUnwanted()
{
  std::set<std::string> wanted;
  for(const auto& [mi, keys] : m_wants)
    wanted.insert(keys.begin(), keys.end());
  for(const auto& [key, unwanted] : m_baking)
    *unwanted = !wanted.count(key);
}

// The scene they were for is going: what runs still makes no more than it has to
void FvMaterials::forgetBakes()
{
  for(const auto& [key, unwanted] : m_baking)
    *unwanted = true;
  m_baking.clear();
  m_bakeQueue.clear();
  m_wants.clear();
}

void FvMaterials::bake(std::vector<fv::build::TextureRequest> requests)
{
  for(fv::build::TextureRequest& r : requests)
  {
    auto unwanted = std::make_shared<std::atomic<bool>>(false);
    m_baking[fv::mat::requestKey(r)] = unwanted;
    m_bakeQueue.push_back({std::move(r), std::move(unwanted)});
  }
  startBake();
}

void FvMaterials::startBake()
{
  if(m_bakeRunning)
    return;
  std::vector<BakeItem> items;
  for(BakeItem& item : m_bakeQueue)
    if(*item.unwanted)
      m_baking.erase(fv::mat::requestKey(item.request));
    else
      items.push_back(std::move(item));
  m_bakeQueue.clear();
  if(items.empty())
    return;
  m_bakeRunning = true;
  const uint64_t generation = m_generation;
  const uint64_t jobId      = m_hooks.nextJob ? m_hooks.nextJob() : 0;
  if(m_hooks.job)
    m_hooks.job(jobId, {{"label", "Baking " + std::to_string(items.size()) + (items.size() == 1 ? " texture" : " textures")}, {"state", "running"}});
  const bool flipped = m_flipped;
  auto       alive   = m_alive;
  m_threads.spawn([this, items = std::move(items), generation, jobId, flipped, alive] {
    std::vector<BakeResult> results;
    for(const BakeItem& item : items)
    {
      if(!*alive)
        return;
      const fv::build::TextureRequest& r = item.request;
      if(*item.unwanted)
      {
        results.push_back({r, std::nullopt, true});
        continue;
      }
      std::optional<fv::build::TextureImage> image;
      try
      {
        image = m_hooks.texture(r, flipped);
      }
      catch(const std::exception& e)
      {
        LOGW("texture %s: %s\n", r.rel.c_str(), e.what());
      }
      // decoded here, off the app thread: the scene that takes the texture claims it, if one still will
      if(image && m_hooks.renderer && !*item.unwanted)
        m_hooks.renderer->prefetchImage(pathOfUri(image->uri), r.srgb);
      results.push_back({r, std::move(image)});
    }
    if(!*alive)
      return;
    m_hooks.post([this, results = std::move(results), generation, jobId, alive]() mutable {
      if(!*alive)
        return;
      if(m_hooks.job)
        m_hooks.job(jobId, {{"state", "done"}});
      m_bakeRunning = false;
      baked(std::move(results), generation);
      startBake();
    });
  });
}

void FvMaterials::baked(std::vector<BakeResult> results, uint64_t generation)
{
  if(generation != m_generation || !m_scene)
    return;
  for(auto& [req, image, skipped] : results)
  {
    const std::string key = fv::mat::requestKey(req);
    m_baking.erase(key);
    if(skipped)
      continue;
    // kept with the figure, where its feed takes it as the figure's sampler repeats it (Genesis
    // UVs use one 0..1 tile per surface group)
    if(image && m_hooks.remember)
    {
      m_hooks.remember(req, *image);
      continue;
    }
    m_failed.insert(key);
    if(m_hooks.notice)
      m_hooks.notice("The texture " + baseName(req.rel) + " could not be made.", "warning");
  }
  std::set<int> again;
  again.swap(m_waiting);
  refresh(again);
}

Json FvMaterials::item(int index) const
{
  if(m_model)
    return modelItem(index);
  const fv::Material& m = m_scene->materials[index];
  const fv::mat::Surface&    s = m.surface;
  const fv::mat::Lowered     lowered = fv::mat::lower(s, m.edits, m_options[index]);
  const fv::mat::Effective&  v = lowered.effective;
  Json                       values = {{"color", hexOf(v.color)},
                                       {"roughness", v.roughness},
                                       {"metalness", v.metalness},
                                       {"opacity", v.opacity},
                                       {"alphaMode", v.alphaMode},
                                       {"doubleSided", v.doubleSided},
                                       {"specular", v.specular},
                                       {"clearcoat", v.clearcoat},
                                       {"clearcoatRoughness", v.clearcoatRoughness},
                                       {"emissive", hexOf(v.emissive)},
                                       {"emissiveIntensity", v.emissiveIntensity}};
  // null without such a map, so a merge patch takes the strength away when an edit removes the map
  values["normalStrength"] = v.normalStrength ? Json(*v.normalStrength) : Json(nullptr);
  values["bumpStrength"]   = v.bumpStrength ? Json(*v.bumpStrength) : Json(nullptr);
  fv::mat::clothesValues(m.edits, values);
  // the maps as lowering uses them, the edits' included; one an edit took away has rel ""
  Json textures = Json::array();
  for(const fv::mat::MapSlot& t : fv::mat::mapSlots(s, m.edits))
    textures.push_back({{"slot", t.slot}, {"label", t.label}, {"file", baseName(t.rel)}, {"rel", t.rel}, {"edited", t.edited}});
  Json out = {{"name", m.name},
              {"kind", kindName(s.kind)},
              {"shader", shaderName(s.shader)},
              {"meshes", m_meshesOf[index]},
              {"textures", textures},
              {"values", values},
              {"edited", editedKeys(m.edits)},
              {"colorBaked", v.colorBaked},
              {"colorUnused", v.colorUnused},
              {"fibre", v.fibre},
              {"ribbon", m_options[index].ribbon},
              {"coatRole", v.coatRole},
              {"mapped", v.mapped}};
  // an eye: which of its colours it takes, and whether a change cooks its map (else it is live). The
  // colours go out as null when unset, so a merge patch takes an old one away.
  if(const EyeRoles roles = eyeRoles(s, &m_options[index]); roles.iris || roles.sclera)
  {
    // a Genesis 9 eye's white is in the iris's map, so it bakes too; a sclera of its own is a factor
    out["eye"]     = {{"iris", roles.iris}, {"sclera", roles.sclera}, {"baked", roles.iris && !s.base.map.empty()}};
    const auto hex = [](const std::optional<Rgb>& c) { return c ? Json(hexOf(*c)) : Json(nullptr); };
    if(roles.iris)
    {
      out["values"]["irisColor"]    = hex(m.edits.irisColor);
      out["values"]["irisStrength"] = m.edits.irisStrength.value_or(1.f);
    }
    if(roles.sclera)
      out["values"]["scleraColor"] = hex(m.edits.scleraColor);
  }
  // the colour a hair (or any surface) is moved to, null when unset so a merge patch takes it away
  out["values"]["recolor"]         = m.edits.recolor ? Json(hexOf(*m.edits.recolor)) : Json(nullptr);
  out["values"]["recolorStrength"] = m.edits.recolorStrength.value_or(1.f);
  out["values"]["tone"]            = m.edits.tone ? Json(hexOf(*m.edits.tone)) : Json(nullptr);
  out["values"]["toneStrength"]    = m.edits.toneStrength.value_or(1.f);
  // PBRSkin makeup: its colour once changed (null: as the preset made it) and its weight
  if(s.makeup)
  {
    out["makeup"]                  = {{"mapped", !s.makeup->weightMap.empty() || !s.makeup->map.empty()}};
    out["values"]["makeupColor"]   = m.edits.makeupColor ? Json(hexOf(*m.edits.makeupColor)) : Json(nullptr);
    out["values"]["makeupWeight"]  = std::clamp(m.edits.makeupWeight.value_or(s.makeup->weight), 0.f, 1.f);
  }
  // measured skin's scatter distance, times human skin's: only where it applies (not on a
  // thin-walled PBRSkin, which light does not enter)
  if(m_options[index].skin == fv::mat::SkinMode::eMeasured && fv::mat::skinRole(s) == fv::mat::SkinRole::eFlesh
     && !(s.translucency && s.translucency->thinWalled))
    out["values"]["scatterScale"] = m.edits.scatterScale.value_or(1.f);
  return out;
}

void FvMaterials::publish(const std::set<int>& materials, bool all)
{
  if(!m_scene)
    return;
  Json items = Json::object();
  if(all)
    for(size_t i = 0; i < m_count; i++)
      items["m:" + std::to_string(i)] = item(int(i));
  else
    for(int i : materials)
      if(i >= 0 && i < int(m_count))
      {
        Json it = item(i);
        // a merge patch: an edit list that emptied replaces the old one, whole
        items["m:" + std::to_string(i)] = std::move(it);
      }
  if(all)
  {
    Json topic = {{"figure", m_figure}, {"items", items}};
    if(!m_eyes.is_null() && !m_model)
      topic["eyes"] = m_eyes;
    if(!m_skinState.is_null() && !m_model)
      topic["skin"] = m_skinState;
    m_hooks.events->snapshot("materials/" + m_figure, topic);
  }
  else if(!items.empty())
    m_hooks.events->patch("materials/" + m_figure, {{"items", items}});
}
