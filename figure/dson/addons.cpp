#include <figure/dson/addons.hpp>

#include <algorithm>

namespace fv::dson {

namespace {

std::string rel(std::string p)
{
  std::replace(p.begin(), p.end(), '\\', '/');
  const size_t at = p.find_first_not_of('/');
  return at == std::string::npos ? std::string() : p.substr(at);
}

}  // namespace

std::vector<PostLoadAddon> postLoadAddons(const Json& doc)
{
  std::vector<PostLoadAddon> out;
  const Json*                extra = get(get(&doc, "scene"), "extra");
  if(!extra || !extra->is_array())
    return out;
  for(const Json& e : *extra)
  {
    if(getString(&e, "type") != "scene_post_load_script")
      continue;
    const Json* addons = get(get(get(&e, "settings"), "PostLoadAddons"), "value");
    if(!addons || !addons->is_object())
      continue;
    for(const Json& a : *addons)
    {
      const Json* v = get(&a, "value");
      if(!v || !v->is_object() || !get(v, "AssetFile") || !get(v, "AssetFile")->is_string())
        continue;
      PostLoadAddon addon;
      addon.name = getString(v, "AssetName");
      addon.file = rel(getString(v, "AssetFile"));
      if(const Json* presets = get(get(v, "Presets"), "value"); presets && presets->is_object())
        for(const Json& p : *presets)
        {
          const std::string file = getString(get(&p, "value"), "PresetFile");
          if(!file.empty())
            addon.presets.push_back(rel(file));
        }
      out.push_back(std::move(addon));
    }
  }
  return out;
}

}  // namespace fv::dson
