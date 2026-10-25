#include <figure/dson/eye_materials.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace fv::dson {

bool isEyeSurface(const Material& m)
{
  if(m.surface.kind == mat::Kind::eEye)
    return true;
  if(m.surface.kind != mat::Kind::eCornea)
    return false;
  std::string name = m.name;
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return name.find("tear") == std::string::npos;
}

std::set<int> eyeSurfaces(const SceneGraph& scene)
{
  // a node is gone when it or a node above it was taken off
  const auto isGone = [&](int n) {
    for(; n >= 0 && n < int(scene.nodes.size()); n = scene.nodes[size_t(n)].parent)
      if(scene.nodes[size_t(n)].removed)
        return true;
    return false;
  };
  // materials no node uses count as the scene's own
  std::vector<int> users(scene.materials.size(), 0), live(scene.materials.size(), 0);
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    const int mesh = scene.nodes[i].mesh;
    if(mesh < 0 || mesh >= int(scene.meshes.size()))
      continue;
    const bool out = isGone(int(i));
    for(int m : scene.meshes[size_t(mesh)].materials)
      if(m >= 0 && m < int(scene.materials.size()))
      {
        users[size_t(m)]++;
        live[size_t(m)] += out ? 0 : 1;
      }
  }
  std::set<int> out;
  for(size_t i = 0; i < scene.materials.size(); i++)
    if((users[i] == 0 || live[i] > 0) && isEyeSurface(scene.materials[i]))
      out.insert(int(i));
  return out;
}

bool isEyePreset(const Json& doc, const SceneGraph& scene)
{
  const std::set<int> targets = materialPresetTargets(doc, scene);
  if(targets.empty())
    return false;
  const std::set<int> eyes = eyeSurfaces(scene);
  return std::includes(eyes.begin(), eyes.end(), targets.begin(), targets.end());
}

EyesResult chooseEyes(SceneGraph& scene, const Json* doc, const std::set<int>& previous)
{
  EyesResult    r;
  std::set<int> eyes = eyeSurfaces(scene);
  std::set<int> back = eyes;
  back.insert(previous.begin(), previous.end());
  r.changed = restoreBaseline(scene, back);
  if(doc)
  {
    // only its eye part: lashes or a face it also sets would wipe what was chosen for them since
    r.preset = applyMaterialPreset(*doc, scene, &eyes);
    r.owned  = r.preset.changed;
    r.changed.insert(r.preset.changed.begin(), r.preset.changed.end());
  }
  return r;
}

}  // namespace fv::dson
