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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>

#include <nvutils/logger.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "renderer.hpp"
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
    case fv::mat::Surface::Shader::eLegacy: break;
  }
  return "legacy";
}

// "#rrggbb" of a linear colour, in sRGB as the colour pickers hold it
std::string hexOf(const Rgb& linear)
{
  char buf[8];
  int  c[3];
  for(int i = 0; i < 3; i++)
    c[i] = int(std::lround(std::clamp(fv::mat::linearToSrgb(std::clamp(linear[i], 0.f, 1.f)), 0.f, 1.f) * 255.f));
  std::snprintf(buf, sizeof buf, "#%02x%02x%02x", c[0], c[1], c[2]);
  return buf;
}

std::optional<Rgb> parseHex(const Json& v)
{
  if(!v.is_string())
    return std::nullopt;
  const std::string s = v.get<std::string>();
  if(s.size() != 7 || s[0] != '#')
    return std::nullopt;
  Rgb out;
  for(int i = 0; i < 3; i++)
  {
    char*      end = nullptr;
    const auto n   = std::strtol(s.substr(1 + i * 2, 2).c_str(), &end, 16);
    if(*end)
      return std::nullopt;
    out[i] = fv::mat::srgbToLinear(n / 255.0);
  }
  return out;
}

std::string baseName(const std::string& rel)
{
  const size_t slash = rel.find_last_of("/\\");
  return slash == std::string::npos ? rel : rel.substr(slash + 1);
}

// What decides a material's place in the renderer's instance flags and raster buckets.
struct InstanceInputs
{
  std::string alphaMode;
  bool        doubleSided, transmission, volume, diffuseTransmission;
  bool        operator==(const InstanceInputs&) const = default;
};

InstanceInputs instanceInputs(const tinygltf::Material& m)
{
  return {m.alphaMode, m.doubleSided, tinygltf::utils::getTransmission(m).factor > 0.f, tinygltf::utils::getVolume(m).thicknessFactor > 0.f,
          tinygltf::utils::getDiffuseTransmission(m).diffuseTransmissionFactor > 0.f};
}

float numberOf(const Json& v, const char* key, float lo, float hi)
{
  if(!v.is_number())
    throw std::runtime_error(std::string(key) + " must be a number");
  return std::clamp(v.get<float>(), lo, hi);
}

// One value of material.set onto the edits; null drops the edit.
void applyValue(fv::mat::Edits& e, const std::string& key, const Json& v)
{
  const bool drop = v.is_null();
  if(key == "color" || key == "emissive")
  {
    std::optional<Rgb>& slot = key == "color" ? e.color : e.emissive;
    if(drop)
      slot.reset();
    else if(const auto c = parseHex(v))
      slot = *c;
    else
      throw std::runtime_error(key + " must be a #rrggbb colour");
  }
  else if(key == "alphaMode")
  {
    if(drop)
      e.alphaMode.reset();
    else if(v == "opaque" || v == "cutout" || v == "blend")
      e.alphaMode = v.get<std::string>();
    else
      throw std::runtime_error("alphaMode is opaque, cutout or blend");
  }
  else if(key == "doubleSided")
  {
    if(drop)
      e.doubleSided.reset();
    else if(v.is_boolean())
      e.doubleSided = v.get<bool>();
    else
      throw std::runtime_error("doubleSided must be a boolean");
  }
  else
  {
    struct Field
    {
      const char*                       name;
      std::optional<float> fv::mat::Edits::*member;
      float                             lo, hi;
    };
    static const Field fields[] = {
        {"roughness", &fv::mat::Edits::roughness, 0, 1},
        {"metalness", &fv::mat::Edits::metalness, 0, 1},
        {"opacity", &fv::mat::Edits::opacity, 0, 1},
        {"normalStrength", &fv::mat::Edits::normalStrength, 0, 3},
        {"bumpStrength", &fv::mat::Edits::bumpStrength, 0, 2},
        {"specular", &fv::mat::Edits::specular, 0, 1},
        {"clearcoat", &fv::mat::Edits::clearcoat, 0, 1},
        {"clearcoatRoughness", &fv::mat::Edits::clearcoatRoughness, 0, 1},
        {"emissiveIntensity", &fv::mat::Edits::emissiveIntensity, 0, 1000},
    };
    for(const Field& f : fields)
      if(key == f.name)
      {
        if(drop)
          (e.*f.member).reset();
        else
          e.*f.member = numberOf(v, f.name, f.lo, f.hi);
        return;
      }
    throw std::runtime_error("unknown material value " + key);
  }
}

Json editedKeys(const fv::mat::Edits& e)
{
  Json out = Json::array();
  const auto add = [&](bool set, const char* name) {
    if(set)
      out.push_back(name);
  };
  add(e.color.has_value(), "color");
  add(e.roughness.has_value(), "roughness");
  add(e.metalness.has_value(), "metalness");
  add(e.opacity.has_value(), "opacity");
  add(e.alphaMode.has_value(), "alphaMode");
  add(e.doubleSided.has_value(), "doubleSided");
  add(e.normalStrength.has_value(), "normalStrength");
  add(e.bumpStrength.has_value(), "bumpStrength");
  add(e.specular.has_value(), "specular");
  add(e.clearcoat.has_value(), "clearcoat");
  add(e.clearcoatRoughness.has_value(), "clearcoatRoughness");
  add(e.emissive.has_value(), "emissive");
  add(e.emissiveIntensity.has_value(), "emissiveIntensity");
  return out;
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
  std::lock_guard lock(m_threadsMutex);
  for(std::thread& t : m_threads)
    if(t.joinable())
      t.join();
}

void FvMaterials::attach(const std::string& figureId, fv::SceneGraph* scene, const fv::build::BuildResult& built, bool texturesFlipped, fv::mat::SkinMode skin)
{
  m_generation++;
  m_figure   = figureId;
  m_scene    = scene;
  m_options  = built.lowering;
  m_textures = built.textureOf;
  m_failed.clear();
  m_baking.clear();
  m_waiting.clear();
  m_flipped = texturesFlipped;
  m_skin    = skin;
  m_applied.assign(scene->materials.size(), {});
  m_meshesOf.assign(scene->materials.size(), {});
  m_forceDouble.assign(scene->materials.size(), false);
  for(size_t mi = 0; mi < scene->materials.size(); mi++)
    m_applied[mi] = stateOf(fv::mat::lower(scene->materials[mi].surface, scene->materials[mi].edits, m_options[mi]));
  for(size_t i = 0; i < scene->meshes.size(); i++)
  {
    const fv::Mesh& m = scene->meshes[i];
    for(int mi : m.materials)
      if(mi >= 0 && mi < int(m_meshesOf.size()))
      {
        if(m_meshesOf[mi].empty() || m_meshesOf[mi].back() != int(i))
          m_meshesOf[mi].push_back(int(i));
        // strand hair ribbons face one way, so their surfaces are seen from both sides
        if(m.geo && m.geo->ribbon)
          m_forceDouble[mi] = true;
      }
  }
  publish({}, true);
}

void FvMaterials::detach()
{
  m_generation++;
  m_scene = nullptr;
  m_options.clear();
  m_textures.clear();
  m_applied.clear();
  m_meshesOf.clear();
  m_forceDouble.clear();
  m_waiting.clear();
  m_baking.clear();
  m_hooks.events->snapshot("materials", {{"figure", nullptr}, {"items", Json::object()}});
}

std::string FvMaterials::stateOf(const fv::mat::Lowered& lowered) const
{
  // the material, and the textures its slots ask for (the JSON holds only their numbers)
  std::string text = lowered.material.dump();
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
    if(!m_scene || index < 0 || index >= int(m_scene->materials.size()))
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
  }
  std::set<int> changed;
  for(int mi : index)
  {
    for(auto it = values->begin(); it != values->end(); ++it)
      applyValue(m_scene->materials[mi].edits, it.key(), it.value());
    changed.insert(mi);
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
    for(size_t i = 0; i < m_scene->materials.size(); i++)
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

// "<name>", or "<name>#n" for the n-th material of that name in scene order.
std::vector<std::string> materialKeys(const fv::SceneGraph& scene)
{
  std::map<std::string, int> counts;
  std::vector<std::string>   keys;
  for(const fv::Material& m : scene.materials)
  {
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
  for(size_t i = 0; i < m_scene->materials.size(); i++)
  {
    if(m_scene->materials[i].edits.empty())
      continue;
    const Json shown = item(int(i));
    Json       edited = Json::object();
    for(const Json& k : shown["edited"])
      edited[k.get<std::string>()] = shown["values"][k.get<std::string>()];
    out[keys[i]] = std::move(edited);
  }
  return out;
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
  if(!m_scene)
    return;
  // the ones that differ: refresh() skips what it would write unchanged
  std::set<int> all;
  for(size_t i = 0; i < m_scene->materials.size(); i++)
    all.insert(int(i));
  refresh(all);
}

void FvMaterials::refresh(const std::set<int>& materials)
{
  std::vector<fv::build::TextureRequest> missing;
  std::set<std::string>                  asked;
  std::set<int>                          shown;
  for(int mi : materials)
  {
    if(mi < 0 || mi >= int(m_scene->materials.size()))
      continue;
    m_options[mi].skin = m_skin;
    const fv::Material& m = m_scene->materials[mi];
    const fv::mat::Lowered     lowered = fv::mat::lower(m.surface, m.edits, m_options[mi]);
    bool                       wait    = false;
    for(const fv::mat::TextureRequest& r : lowered.requests)
    {
      const std::string key = fv::mat::requestKey(r);
      const auto        it  = m_textures.find(key);
      if((it != m_textures.end() && it->second.index >= 0) || m_failed.count(key))
        continue;
      wait = true;
      if(!m_baking.count(key) && asked.insert(key).second)
        missing.push_back(r);
    }
    shown.insert(mi);
    if(wait)
    {
      m_waiting.insert(mi);
      continue;
    }
    m_waiting.erase(mi);
    if(stateOf(lowered) != m_applied[mi])
      write(mi, lowered);
  }
  if(!missing.empty())
    bake(std::move(missing));
  publish(shown, false);
}

void FvMaterials::write(int index, const fv::mat::Lowered& lowered)
{
  nvvkgltf::Scene* scn = m_hooks.renderer->editScene();
  if(!scn || !scn->valid() || index >= int(scn->getModel().materials.size()))
    return;
  tinygltf::Model&    model  = scn->getModel();
  tinygltf::Material& mat    = scn->editor().getMaterialForEdit(index);
  const auto          before = instanceInputs(mat);
  tinygltf::Material  next   = fv::mat::toGltf(
      lowered,
      [&](const fv::mat::TextureRequest& r) {
        const auto it = m_textures.find(fv::mat::requestKey(r));
        return it == m_textures.end() ? fv::mat::ResolvedTexture{} : it->second;
      },
      model.extensionsUsed);
  next.name   = mat.name;
  next.extras = mat.extras;
  if(m_forceDouble[index])
    next.doubleSided = true;
  mat = std::move(next);
  scn->markMaterialDirty(index);
  if(!(instanceInputs(mat) == before))
    scn->markRenderNodeRtxDirtyForMaterials({index});
  m_applied[index] = stateOf(lowered);
}

void FvMaterials::bake(std::vector<fv::build::TextureRequest> requests)
{
  for(const auto& r : requests)
    m_baking.insert(fv::mat::requestKey(r));
  const uint64_t generation = m_generation;
  const uint64_t jobId      = m_hooks.nextJob ? m_hooks.nextJob() : 0;
  if(m_hooks.job)
    m_hooks.job(jobId, {{"label", "Baking " + std::to_string(requests.size()) + (requests.size() == 1 ? " texture" : " textures")}, {"state", "running"}});
  const bool flipped = m_flipped;
  auto       alive   = m_alive;
  std::lock_guard lock(m_threadsMutex);
  m_threads.emplace_back([this, requests = std::move(requests), generation, jobId, flipped, alive] {
    std::vector<std::pair<fv::build::TextureRequest, std::optional<fv::build::TextureImage>>> results;
    for(const auto& r : requests)
    {
      if(!*alive)
        return;
      std::optional<fv::build::TextureImage> image;
      try
      {
        image = m_hooks.texture(r, flipped);
      }
      catch(const std::exception& e)
      {
        LOGW("texture %s: %s\n", r.rel.c_str(), e.what());
      }
      results.emplace_back(r, std::move(image));
    }
    if(!*alive)
      return;
    m_hooks.post([this, results = std::move(results), generation, jobId, alive]() mutable {
      if(!*alive)
        return;
      if(m_hooks.job)
        m_hooks.job(jobId, {{"state", "done"}});
      baked(std::move(results), generation);
    });
  });
}

void FvMaterials::baked(std::vector<std::pair<fv::build::TextureRequest, std::optional<fv::build::TextureImage>>> results, uint64_t generation)
{
  if(generation != m_generation || !m_scene)
    return;
  nvvkgltf::Scene* scn = m_hooks.renderer->editScene();
  if(!scn || !scn->valid())
    return;
  for(auto& [req, image] : results)
  {
    const std::string key = fv::mat::requestKey(req);
    m_baking.erase(key);
    int         texture = -1;
    std::string error;
    if(image)
    {
      texture = scn->editor().importImageAsTexture(pathOfUri(image->uri), &error);
      if(texture >= 0)
      {
        tinygltf::Model& model = scn->getModel();
        // Genesis UVs use one 0..1 tile per surface group; the figure's sampler repeats
        if(!model.samplers.empty())
          model.textures[texture].sampler = 0;
        m_textures[key] = {texture, image->flipGreen};
        continue;
      }
    }
    m_failed.insert(key);
    if(m_hooks.notice)
      m_hooks.notice("The texture " + baseName(req.rel) + " could not be made" + (error.empty() ? "." : ": " + error), "warning");
  }
  std::set<int> again;
  again.swap(m_waiting);
  refresh(again);
}

Json FvMaterials::item(int index) const
{
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
  if(v.normalStrength)
    values["normalStrength"] = *v.normalStrength;
  if(v.bumpStrength)
    values["bumpStrength"] = *v.bumpStrength;
  Json textures = Json::array();
  const auto add = [&](const char* label, const std::string& rel) {
    if(!rel.empty())
      textures.push_back({{"label", label}, {"file", baseName(rel)}});
  };
  add("Base colour", s.base.map);
  add("Cutout", s.opacity.map);
  add("Normal", s.normalMap);
  add("Bump", s.bumpMap);
  add("Roughness", s.glossy.roughness.map);
  if(s.translucency)
    add("Translucency", s.translucency->map);
  if(s.emission)
    add("Emission", s.emission->color.map);
  if(s.detail)
    add("Detail", s.detail->map);
  if(s.makeup)
    add("Makeup", s.makeup->map);
  Json out = {{"name", m.name},
              {"kind", kindName(s.kind)},
              {"shader", shaderName(s.shader)},
              {"meshes", m_meshesOf[index]},
              {"textures", textures},
              {"values", values},
              {"edited", editedKeys(m.edits)}};
  if(s.translucency && s.translucency->weight > 0)
    out["translucency"] = s.translucency->weight;
  return out;
}

void FvMaterials::publish(const std::set<int>& materials, bool all)
{
  if(!m_scene)
    return;
  Json items = Json::object();
  if(all)
    for(size_t i = 0; i < m_scene->materials.size(); i++)
      items["m:" + std::to_string(i)] = item(int(i));
  else
    for(int i : materials)
      if(i >= 0 && i < int(m_scene->materials.size()))
      {
        Json it = item(i);
        // a merge patch: an edit list that emptied replaces the old one, whole
        items["m:" + std::to_string(i)] = std::move(it);
      }
  if(all)
    m_hooks.events->snapshot("materials", {{"figure", m_figure}, {"items", items}});
  else if(!items.empty())
    m_hooks.events->patch("materials", {{"items", items}});
}
