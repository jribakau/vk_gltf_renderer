#include <figure/dson/material_preset.hpp>

#include <algorithm>
#include <map>
#include <regex>
#include <set>

#include <figure/dson/materials.hpp>

namespace fv::dson {

namespace {

std::string dec(const std::string& s)
{
  try
  {
    return decodeUri(s);
  }
  catch(const std::exception&)
  {
    return s;
  }
}

// The value of an animation's last key: [time, value].
const Json* lastValue(const Json& anim)
{
  const Json* keys = get(&anim, "keys");
  if(!keys || !keys->is_array() || keys->empty())
    return nullptr;
  const Json& last = keys->back();
  return last.is_array() && last.size() > 1 ? &last[1] : nullptr;
}

// The id an image_library reference names: "#id", percent-decoded.
std::string imageId(const std::string& value)
{
  return dec(!value.empty() && value[0] == '#' ? value.substr(1) : value);
}

std::vector<std::string> split(const std::string& s, char sep)
{
  std::vector<std::string> out;
  size_t                   from = 0;
  for(;;)
  {
    const size_t at = s.find(sep, from);
    out.push_back(dec(s.substr(from, at == std::string::npos ? std::string::npos : at - from)));
    if(at == std::string::npos)
      return out;
    from = at + 1;
  }
}

// The DSON node each material belongs to: the nearest figure or prop over every mesh using it (a
// worn item hangs under the body's group, which is not its owner).
std::vector<std::set<std::string>> ownersOf(const SceneGraph& scene)
{
  std::vector<std::set<std::string>> owners(scene.materials.size());
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    const Node& n = scene.nodes[i];
    if(n.mesh < 0 || n.mesh >= int(scene.meshes.size()))
      continue;
    int p = int(i);
    while(p >= 0 && scene.nodes[size_t(p)].dsonNode.empty())
      p = scene.nodes[size_t(p)].parent;
    if(p < 0)
      continue;
    for(int m : scene.meshes[size_t(n.mesh)].materials)
      if(m >= 0 && m < int(owners.size()))
        owners[size_t(m)].insert(scene.nodes[size_t(p)].dsonNode);
  }
  return owners;
}

}  // namespace

std::vector<int> presetTargets(const SceneGraph& scene, const std::string& node, const std::string& surface)
{
  const auto owners    = ownersOf(scene);
  const bool nodeKnown = !node.empty() && std::any_of(owners.begin(), owners.end(), [&](const auto& o) { return o.count(node) > 0; });
  const auto pick      = [&](const auto& matches) {
    std::vector<int> out;
    for(size_t i = 0; i < scene.materials.size(); i++)
      if((!nodeKnown || owners[i].count(node)) && matches(scene.materials[i].name))
        out.push_back(int(i));
    return out;
  };
  if(auto exact = pick([&](const std::string& name) { return name == surface; }); !exact.empty())
    return exact;
  // "Eyelashes" for the "Eyelashes Upper" and "Eyelashes Lower" a geometry splits it into
  const std::string stem = surface + " ";
  if(auto parts = pick([&](const std::string& name) { return name.rfind(stem, 0) == 0; }); !parts.empty())
    return parts;
  // "FE Base" for "Base": a product prefix the surfaces of another version of the item don't carry
  const size_t space = surface.find(' ');
  if(space == std::string::npos || space + 1 >= surface.size())
    return {};
  const std::string bare = surface.substr(space + 1);
  return pick([&](const std::string& name) { return name == bare; });
}

MaterialPresetResult applyMaterialPreset(const Json& doc, SceneGraph& scene)
{
  MaterialPresetResult result;
  const Json*          animations = get(get(&doc, "scene"), "animations");
  if(!animations || !animations->is_array())
    return result;
  std::unordered_map<std::string, std::string> images;
  collectImages(doc, images);

  // (node, surface) -> channel -> override, in the order the preset lists them
  std::map<std::pair<std::string, std::string>, std::vector<ChannelOverride>> bySurface;
  static const std::regex url(R"(^(.*?)#materials/([^:]+):\?(.+)$)");
  for(const Json& a : *animations)
  {
    const std::string u = getString(&a, "url");
    std::smatch       m;
    if(!std::regex_match(u, m, url))
      continue;
    const Json* value = lastValue(a);
    if(!value)
      continue;
    const std::vector<std::string> segs = split(m[3].str(), '/');
    // "extra/<shader>/channels/<id>/<field>" or "<id>/<field>"
    const bool        extra   = !segs.empty() && segs[0] == "extra";
    const std::string channel = extra ? (segs.size() > 3 ? segs[3] : "") : segs[0];
    const std::string field   = extra ? (segs.size() > 4 ? segs[4] : "") : (segs.size() > 1 ? segs[1] : "");
    const size_t      rest    = extra ? 5 : 2;
    if(channel.empty() || segs.size() > rest || (field != "value" && field != "image_file" && field != "image"))
      continue;
    const std::string surface = dec(m[2].str());
    // the node: "Genesis9Eyelashes", or the last segment of a "name://@selection/<node>" chain
    std::string node = m[1].str();
    if(const size_t slash = node.rfind('/'); slash != std::string::npos)
      node = node.substr(slash + 1);
    if(!node.empty() && node.back() == ':')
      node.pop_back();
    auto& list = bySurface[{dec(node), surface}];
    auto              it      = std::find_if(list.begin(), list.end(), [&](const ChannelOverride& o) { return o.channel == channel; });
    if(it == list.end())
      it = list.insert(list.end(), ChannelOverride{channel, std::nullopt, std::nullopt});
    if(field == "value")
      it->value = *value;
    else if(field == "image_file")
      it->image = value->is_string() ? value->get<std::string>() : std::string();
    else
    {
      // an image by image_library id, or null to remove the map
      std::string file;
      if(value->is_string())
      {
        const auto found = images.find(imageId(value->get<std::string>()));
        file             = found != images.end() ? found->second : value->get<std::string>();
      }
      it->image = file;
    }
    result.edits++;
  }

  for(const auto& [key, overrides] : bySurface)
  {
    const auto& [node, surface] = key;
    const std::vector<int> targets = presetTargets(scene, node, surface);
    if(targets.empty())
      result.missing.insert(surface);
    else
      result.matched.insert(surface);
    for(int i : targets)
      if(overrideChannels(scene.materials[size_t(i)], overrides))
        result.changed.insert(i);
  }
  return result;
}

}  // namespace fv::dson
