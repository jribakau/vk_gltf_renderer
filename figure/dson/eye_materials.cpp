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
  const std::vector<bool> live = liveMaterials(scene);
  std::set<int>           out;
  for(size_t i = 0; i < scene.materials.size(); i++)
    if(live[i] && isEyeSurface(scene.materials[i]))
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

EyesResult chooseEyes(SceneGraph& scene, const Json* doc, const std::set<int>& previous, const MaterialDefinitions& definitions)
{
  return choosePreset(scene, eyeSurfaces(scene), doc, previous, definitions);
}

}  // namespace fv::dson
