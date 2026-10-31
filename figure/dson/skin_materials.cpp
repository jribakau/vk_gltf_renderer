#include <figure/dson/skin_materials.hpp>

#include <algorithm>
#include <cctype>
#include <vector>

#include <figure/dson/eye_materials.hpp>
#include <figure/dson/materials.hpp>
#include <figure/materials/texture_request.hpp>

namespace fv::dson {

std::set<int> skinSurfaces(const SceneGraph& scene, const std::set<int>& leaveOut)
{
  const std::vector<bool> live = liveMaterials(scene);
  std::set<int>           out;
  for(size_t i = 0; i < scene.materials.size(); i++)
    if(live[i] && !leaveOut.count(int(i)) && !isEyeSurface(scene.materials[i]))
      out.insert(int(i));
  return out;
}

LayerChoice chooseLayer(SceneGraph& scene, const std::set<int>& surfaces, const Json* doc, const PresetLayer& previous,
                        const MaterialDefinitions& definitions)
{
  LayerChoice r;
  // the last layer off, channel by channel
  for(const auto& [i, layer] : previous)
  {
    if(i < 0 || i >= int(scene.materials.size()) || !scene.materials[size_t(i)].channels)
      continue;
    Material& m    = scene.materials[size_t(i)];
    Channels  next = *m.channels;
    bool      any  = false;
    for(const auto& [key, c] : layer.channels)
    {
      if(next.shared(key) != c.after)
        continue;
      if(c.before)
        next.set(key, c.before);
      else
        next.erase(key);
      any = true;
    }
    if(layer.shader && next.shader() == layer.shader->second)
    {
      next.setShader(layer.shader->first);
      any = true;
    }
    if(!any)
      continue;
    m.surface  = parseSurface(m.name, next, m.context);
    m.channels = std::make_shared<const Channels>(std::move(next));
    r.changed.insert(i);
  }
  if(!doc)
    return r;
  // the next one on, keeping what each channel was
  std::map<int, std::shared_ptr<const Channels>> before;
  for(int i : surfaces)
    if(i >= 0 && i < int(scene.materials.size()))
      before[i] = scene.materials[size_t(i)].channels;
  r.preset = applyMaterialPreset(*doc, scene, &surfaces, definitions);
  for(int i : r.preset.changed)
  {
    const auto& now = scene.materials[size_t(i)].channels;
    const auto& was = before[i];
    if(!now)
      continue;
    LayerMaterial& layer = r.layer[i];
    for(const std::string& key : now->order())
    {
      std::shared_ptr<Channel> a = now->shared(key), b = was ? was->shared(key) : nullptr;
      if(a != b)
        layer.channels[key] = {b, a};
    }
    // a preset of another shader took the old one's channels away: they come back with it
    if(was)
    {
      for(const std::string& key : was->order())
        if(!now->shared(key))
          layer.channels[key] = {was->shared(key), nullptr};
      if(was->shader() != now->shader())
        layer.shader = std::pair{was->shader(), now->shader()};
    }
    r.changed.insert(i);
  }
  return r;
}

Json fadedLayers(const Json& doc, float opacity)
{
  Json faded = doc;
  if(!faded.contains("image_library") || !faded["image_library"].is_array())
    return faded;
  for(Json& image : faded["image_library"])
  {
    if(!image.is_object() || !image.contains("map") || !image["map"].is_array())
      continue;
    Json& map = image["map"];
    // A whole image at the bottom, once under full opacity, would read as a placeholder and be left
    // out (collectImages): a placeholder under it keeps it, faded, over the map there.
    const auto bottom = std::find_if(map.begin(), map.end(), [](const Json& l) { return l.is_object() && l.value("active", true); });
    const bool whole  = bottom != map.end() && !bottom->value("url", std::string()).empty() && bottom->value("transparency", 1.0) >= 1;
    for(Json& layer : map)
      if(layer.is_object())
        layer["transparency"] = layer.value("transparency", 1.0) * opacity;
    if(whole)
      map.insert(map.begin(), Json{{"color", Json::array({0, 0, 0})}, {"transparency", 0}});
  }
  return faded;
}

LayerChoice chooseLayers(SceneGraph&                                            scene,
                         const std::set<int>&                                   surfaces,
                         const std::vector<LaidPreset>&                         presets,
                         const PresetLayer&                                     previous,
                         const std::function<MaterialDefinitions(const Json&)>& definitions)
{
  LayerChoice r = chooseLayer(scene, surfaces, nullptr, previous);
  for(const LaidPreset& p : presets)
  {
    if(!p.doc)
      continue;
    Json        faded;
    const Json* doc = p.doc;
    if(p.opacity < 1)
    {
      faded = fadedLayers(*p.doc, p.opacity);
      doc   = &faded;
    }
    LayerChoice one = chooseLayer(scene, surfaces, doc, {}, definitions ? definitions(*p.doc) : MaterialDefinitions{});
    r.changed.insert(one.changed.begin(), one.changed.end());
    // taken off together, a channel goes back to what it was before the first that set it
    for(auto& [i, layer] : one.layer)
    {
      LayerMaterial& into = r.layer[i];
      for(const auto& [key, c] : layer.channels)
        if(const auto at = into.channels.find(key); at != into.channels.end())
          at->second.after = c.after;
        else
          into.channels[key] = c;
      if(layer.shader)
        into.shader = into.shader ? std::pair{into.shader->first, layer.shader->second} : *layer.shader;
    }
  }
  return r;
}

std::string toneReference(const SceneGraph& scene, const std::set<int>& leaveOut)
{
  const std::vector<bool> live  = liveMaterials(scene);
  const auto              lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
  };
  std::string first;
  for(size_t i = 0; i < scene.materials.size(); i++)
  {
    const Material& m = scene.materials[i];
    if(!live[i] || leaveOut.count(int(i)) || m.surface.kind != mat::Kind::eSkin || m.surface.base.map.empty())
      continue;
    const std::string map = mat::layerFile(mat::mapLayers(m.surface.base.map).front());
    if(const std::string name = lower(m.name); name == "body" || name == "torso")
      return map;
    if(first.empty())
      first = map;
  }
  return first;
}

}  // namespace fv::dson
