#include <figure/dson/presets.hpp>

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <vector>

namespace fv::dson {

namespace {

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

const char* kindOf(const std::string& type)
{
  if(type == "preset_pose" || type == "preset_hierarchical_pose")
    return "pose preset";
  if(type == "preset_material" || type == "preset_hierarchical_material")
    return "material preset";
  if(type == "preset_shader")
    return "shader preset";
  if(type == "preset_shape")
    return "shape preset";
  if(type == "preset_render_settings")
    return "render settings preset";
  if(type == "preset_camera")
    return "camera preset";
  if(type == "preset_light")
    return "light preset";
  if(type == "preset_puppeteer")
    return "puppeteer preset";
  if(type == "preset_dform")
    return "d-former preset";
  if(type == "preset_layered_image")
    return "layered image preset";
  if(type == "preset_properties")
    return "property preset";
  if(type == "preset_simulation_settings")
    return "simulation settings preset";
  if(type == "preset_deformer")
    return "deformer preset";
  return nullptr;
}

std::string nodeType(const Json& n)
{
  std::vector<std::string> extra;
  if(const Json* ex = get(&n, "extra"); ex && ex->is_array())
    for(const Json& e : *ex)
      extra.push_back(getString(&e, "type"));
  for(const std::string& t : extra)
    if(t.find("group_node") != std::string::npos)
      return "group";
  for(const std::string& t : extra)
    if(t.find("/light") != std::string::npos)
      return "light";
  std::string url = getString(&n, "url");
  if(url.empty())
    url = getString(&n, "id");
  static const std::regex camera("camera", std::regex::icase);
  if(std::regex_search(url, camera))
    return "camera";
  return getString(&n, "type", "node");
}

void walkChannels(const Json& el, std::map<std::string, const Json*>& channels)
{
  if(const Json* list = get(&el, "channels"); list && list->is_array())
    for(const Json& c : *list)
      if(const Json* ch = get(&c, "channel"); ch && ch->is_object())
        if(const Json* id = get(ch, "id"); id && id->is_string())
          channels[lower(id->get<std::string>())] = ch;
  if(const Json* children = get(&el, "children"); children && children->is_array())
    for(const Json& child : *children)
      walkChannels(child, channels);
}

std::optional<EnvironmentPreset> environmentFromDoc(const Json& doc)
{
  std::map<std::string, const Json*> channels;
  if(const Json* extra = get(get(&doc, "scene"), "extra"); extra && extra->is_array())
    for(const Json& ex : *extra)
    {
      if(getString(&ex, "type") != "studio_render_settings")
        continue;
      const Json* options = get(&ex, "render_options");
      const Json* elements = options ? get(options, "render_elements") : nullptr;
      if(elements && elements->is_array())
        for(const Json& el : *elements)
          walkChannels(el, channels);
    }
  auto it = channels.find("environment map");
  if(it == channels.end())
    return std::nullopt;
  const Json* image = get(it->second, "image_file");
  if(!image || !image->is_string() || image->get<std::string>().empty())
    return std::nullopt;
  auto num = [&](const char* id) -> std::optional<double> {
    auto c = channels.find(id);
    if(c == channels.end())
      return std::nullopt;
    const Json* v = get(c->second, "current_value");
    if(!v || v->is_null())
      v = get(c->second, "value");
    return v && v->is_number() ? std::optional<double>(v->get<double>()) : std::nullopt;
  };
  EnvironmentPreset env;
  env.image     = image->get<std::string>();
  env.intensity = num("environment intensity");
  env.rotation  = num("dome rotation");
  if(auto dome = channels.find("draw dome"); dome != channels.end())
    if(const Json* v = get(dome->second, "current_value"); v && v->is_boolean())
      env.drawDome = v->get<bool>();
  return env;
}

}  // namespace

std::optional<PresetInfo> describePreset(const Json& doc)
{
  const Json* scene = get(&doc, "scene");
  const Json* nodes = get(scene, "nodes");
  const bool  nodesEmpty = !nodes || !nodes->is_array() || nodes->empty();
  bool        hasGeometry = false;
  if(nodes && nodes->is_array())
    for(const Json& n : *nodes)
      if(const Json* g = get(&n, "geometries"); g && g->is_array() && !g->empty())
        hasGeometry = true;
  if(nodesEmpty)
    if(const Json* geos = get(&doc, "geometry_library"); geos && geos->is_array() && !geos->empty())
      hasGeometry = true;

  const std::string type = getString(get(&doc, "asset_info"), "type");
  const char*       named = kindOf(type);
  // hierarchical presets list the nodes they apply to, geometry included: the type decides
  if(hasGeometry && !named)
    return std::nullopt;

  PresetInfo info;
  info.environment = environmentFromDoc(doc);
  if(named)
    info.kind = named;
  else if(info.environment)
    info.kind = "lighting preset";
  else if(!nodesEmpty)
  {
    bool cameras = true, lights = true;
    for(const Json& n : *nodes)
    {
      const std::string t = nodeType(n);
      cameras &= t == "camera" || t == "group";
      lights &= t == "light" || t == "group";
    }
    if(cameras)
      info.kind = "camera preset";
    else if(lights)
      info.kind = "light preset";
    else
      info.kind = "preset with no geometry";
  }
  else
    info.kind = "preset with no geometry";
  return info;
}

}  // namespace fv::dson
