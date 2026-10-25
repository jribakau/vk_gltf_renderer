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

LayerChoice chooseLayer(SceneGraph& scene, const std::set<int>& surfaces, const Json* doc, const PresetLayer& previous)
{
  LayerChoice r;
  // the last layer off, channel by channel
  for(const auto& [i, channels] : previous)
  {
    if(i < 0 || i >= int(scene.materials.size()) || !scene.materials[size_t(i)].channels)
      continue;
    Material& m    = scene.materials[size_t(i)];
    Channels  next = *m.channels;
    bool      any  = false;
    for(const auto& [key, c] : channels)
    {
      if(next.shared(key) != c.after)
        continue;
      if(c.before)
        next.set(key, c.before);
      else
        next.erase(key);
      any = true;
    }
    if(!any)
      continue;
    m.surface  = parseSurface(m.name, next);
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
  r.preset = applyMaterialPreset(*doc, scene, &surfaces);
  for(int i : r.preset.changed)
  {
    const auto& now = scene.materials[size_t(i)].channels;
    const auto& was = before[i];
    if(!now)
      continue;
    for(const std::string& key : now->order())
    {
      std::shared_ptr<Channel> a = now->shared(key), b = was ? was->shared(key) : nullptr;
      if(a != b)
        r.layer[i][key] = {b, a};
    }
    r.changed.insert(i);
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
    const std::string map = mat::mapLayers(m.surface.base.map).front();
    if(const std::string name = lower(m.name); name == "body" || name == "torso")
      return map;
    if(first.empty())
      first = map;
  }
  return first;
}

}  // namespace fv::dson
