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

// The DSON nodes each material belongs to, each as its chain of DSON nodes from the outermost down:
// the nearest figure or prop over every mesh using it (a worn item hangs under the body's group,
// which is not its owner), then the DSON nodes above that.
std::vector<std::set<std::vector<std::string>>> ownersOf(const SceneGraph& scene)
{
  std::vector<std::set<std::vector<std::string>>> owners(scene.materials.size());
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    const Node& n = scene.nodes[i];
    if(n.mesh < 0 || n.mesh >= int(scene.meshes.size()))
      continue;
    std::vector<std::string> chain;
    for(int p = int(i); p >= 0; p = scene.nodes[size_t(p)].parent)
      if(!scene.nodes[size_t(p)].dsonNode.empty())
        chain.push_back(scene.nodes[size_t(p)].dsonNode);
    if(chain.empty())
      continue;
    std::reverse(chain.begin(), chain.end());
    for(int m : scene.meshes[size_t(n.mesh)].materials)
      if(m >= 0 && m < int(owners.size()))
        owners[size_t(m)].insert(chain);
  }
  return owners;
}

bool endsWith(const std::vector<std::string>& chain, const std::vector<std::string>& tail)
{
  return tail.size() <= chain.size() && std::equal(tail.rbegin(), tail.rend(), chain.rbegin());
}

// The node chain of a preset URL's node part: "Genesis9Eyelashes", "name://@selection" (none), or
// "name://@selection/Genesis9:/Genesis9Eyelashes:".
std::vector<std::string> nodeChain(std::string node)
{
  if(node.rfind("name://", 0) == 0)
    node = node.substr(7);
  std::vector<std::string> chain;
  for(std::string seg : split(node, '/'))
  {
    while(!seg.empty() && seg.back() == ':')
      seg.pop_back();
    if(!seg.empty() && seg != "@selection")
      chain.push_back(seg);
  }
  return chain;
}

}  // namespace

std::vector<int> presetTargets(const SceneGraph& scene, const std::vector<std::string>& chain, const std::string& surface, bool* stray)
{
  const auto        owners = ownersOf(scene);
  std::vector<bool> inScope(scene.materials.size(), true);
  const auto        narrow = [&](const auto& owned) {
    std::vector<bool> scope(scene.materials.size(), false);
    bool              any = false;
    for(size_t i = 0; i < scene.materials.size(); i++)
      for(const auto& c : owners[i])
        if(owned(c))
        {
          scope[i] = any = true;
          break;
        }
    if(any)
      inScope = std::move(scope);
    return any;
  };
  bool found = false;
  // the longest end of the chain some owner ends with
  for(size_t len = chain.size(); len > 0 && !found; len--)
  {
    const std::vector<std::string> tail(chain.end() - std::ptrdiff_t(len), chain.end());
    found = narrow([&](const std::vector<std::string>& c) { return endsWith(c, tail); });
  }
  if(stray)
    *stray = !chain.empty() && !found;
  // else under the innermost node of the chain the scene has
  for(size_t k = chain.size(); k > 0 && !found; k--)
    found = narrow([&](const std::vector<std::string>& c) { return std::find(c.begin(), c.end(), chain[k - 1]) != c.end(); });
  const auto pick = [&](const auto& matches) {
    std::vector<int> out;
    for(size_t i = 0; i < scene.materials.size(); i++)
      if(inScope[i] && matches(scene.materials[i].name))
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

std::vector<int> presetTargets(const SceneGraph& scene, const std::string& node, const std::string& surface)
{
  return presetTargets(scene, node.empty() ? std::vector<std::string>{} : std::vector<std::string>{node}, surface);
}

namespace {

// (node chain, surface) -> channel -> override, in the order the preset lists them
using Overrides = std::map<std::pair<std::vector<std::string>, std::string>, std::vector<ChannelOverride>>;

// The channel values and images a material preset sets, by the surface it addresses; `edits` counts them.
Overrides parseOverrides(const Json& doc, int* edits)
{
  Overrides   bySurface;
  const Json* animations = get(get(&doc, "scene"), "animations");
  if(!animations || !animations->is_array())
    return bySurface;
  std::unordered_map<std::string, std::string> images;
  collectImages(doc, images);
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
    auto& list = bySurface[{nodeChain(m[1].str()), surface}];
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
    if(edits)
      (*edits)++;
  }
  return bySurface;
}

}  // namespace

std::vector<bool> liveMaterials(const SceneGraph& scene)
{
  std::vector<int> users(scene.materials.size(), 0), live(scene.materials.size(), 0);
  for(size_t i = 0; i < scene.nodes.size(); i++)
  {
    const int mesh = scene.nodes[i].mesh;
    if(mesh < 0 || mesh >= int(scene.meshes.size()))
      continue;
    const bool out = takenOff(scene, int(i));
    for(int m : scene.meshes[size_t(mesh)].materials)
      if(m >= 0 && m < int(scene.materials.size()))
      {
        users[size_t(m)]++;
        live[size_t(m)] += out ? 0 : 1;
      }
  }
  std::vector<bool> out(scene.materials.size());
  for(size_t i = 0; i < out.size(); i++)
    out[i] = users[i] == 0 || live[i] > 0;
  return out;
}

MaterialPresetResult applyMaterialPreset(const Json& doc, SceneGraph& scene, const std::set<int>* only)
{
  MaterialPresetResult result;
  for(const auto& [key, overrides] : parseOverrides(doc, &result.edits))
  {
    const auto& [chain, surface] = key;
    bool             stray   = false;
    std::vector<int> targets = presetTargets(scene, chain, surface, &stray);
    if(only)
      std::erase_if(targets, [&](int i) { return !only->count(i); });
    if(targets.empty())
      result.missing.insert(surface);
    else
      result.matched.insert(surface);
    // made for an item the scene doesn't have (fiber brows' colour on card brows): its values, but
    // not the maps it removes, which that item never had and this one is drawn by
    std::vector<ChannelOverride> kept = overrides;
    if(stray)
      for(ChannelOverride& o : kept)
        if(o.image && o.image->empty())
          o.image.reset();
    for(int i : targets)
      if(overrideChannels(scene.materials[size_t(i)], kept))
        result.changed.insert(i);
  }
  return result;
}

std::set<int> materialPresetTargets(const Json& doc, const SceneGraph& scene)
{
  std::set<int> out;
  for(const auto& [key, overrides] : parseOverrides(doc, nullptr))
    for(int i : presetTargets(scene, key.first, key.second))
      out.insert(i);
  return out;
}

void keepBaseline(SceneGraph& scene)
{
  for(Material& m : scene.materials)
    m.baseline = m.channels;
}

std::set<int> restoreBaseline(SceneGraph& scene, const std::set<int>& which)
{
  std::set<int> changed;
  for(int i : which)
  {
    if(i < 0 || i >= int(scene.materials.size()))
      continue;
    Material& m = scene.materials[size_t(i)];
    if(!m.baseline || m.channels == m.baseline)
      continue;
    m.channels = m.baseline;
    m.surface  = parseSurface(m.name, *m.baseline);
    changed.insert(i);
  }
  return changed;
}

PresetChoice choosePreset(SceneGraph& scene, const std::set<int>& surfaces, const Json* doc, const std::set<int>& previous)
{
  PresetChoice  r;
  std::set<int> back = surfaces;
  back.insert(previous.begin(), previous.end());
  r.changed = restoreBaseline(scene, back);
  if(doc)
  {
    // only its part on these: what else it sets (lashes with eyes, the body with a garment) keeps
    // what was chosen for it since
    r.preset = applyMaterialPreset(*doc, scene, &surfaces);
    r.owned  = r.preset.changed;
    r.changed.insert(r.preset.changed.begin(), r.preset.changed.end());
  }
  return r;
}

}  // namespace fv::dson
