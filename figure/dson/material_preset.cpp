#include <figure/dson/material_preset.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
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
  // "#materials/<surface>:?…", or "#materials:?…" for the surfaces selected when it is applied
  static const std::regex url(R"(^(.*?)#materials(?:/([^:]+))?:\?(.+)$)");
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
    // "extra/<shader>/channels/<id>/<field>" or "<id>/<field>", the field perhaps
    // "image_modification/<setting>"
    const bool        extra   = !segs.empty() && segs[0] == "extra";
    const std::string channel = extra ? (segs.size() > 3 ? segs[3] : "") : segs[0];
    const std::string field   = extra ? (segs.size() > 4 ? segs[4] : "") : (segs.size() > 1 ? segs[1] : "");
    const size_t      rest    = extra ? 5 : 2;
    // a map's texture instance settings, by the names DAZ Studio writes (Smaroula Hair's
    // misspelled "SDcale" and "graySDcale_mode" are no setting)
    static const std::set<std::string> kSettings = {"scale",          "offset",         "invert",           "grayscale_mode", "horizontal_tiles",
                                                    "vertical_tiles", "horizontal_tiling_offset", "vertical_tiling_offset"};
    const bool        setting = field == "image_modification" && segs.size() == rest + 1 && kSettings.count(segs[rest]);
    if(channel.empty() || (!setting && (segs.size() > rest || (field != "value" && field != "image_file" && field != "image"))))
      continue;
    const std::string surface = dec(m[2].str());
    auto& list = bySurface[{nodeChain(m[1].str()), surface}];
    auto              it      = std::find_if(list.begin(), list.end(), [&](const ChannelOverride& o) { return o.channel == channel; });
    if(it == list.end())
      it = list.insert(list.end(), ChannelOverride{channel, std::nullopt, std::nullopt, Json()});
    // a setting is how the map is read, not a value of its own: not counted
    if(setting)
    {
      it->imageModification[segs[rest]] = *value;
      continue;
    }
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

namespace {

// The definition a scene material's url names in another file, as file#id; empty for a local one
// ("#Legs") and for none.
std::string definitionUrl(const Json& entry)
{
  const auto ref = parseRef(getString(&entry, "url"), "");
  return ref && !ref->file.empty() && !ref->id.empty() ? ref->file + "#" + ref->id : std::string();
}

// The preset's scene material for a surface: the entry whose groups name it, else whose id is it.
const Json* sceneMaterial(const Json& doc, const std::string& surface)
{
  const Json* materials = get(get(&doc, "scene"), "materials");
  if(!materials || !materials->is_array())
    return nullptr;
  for(const Json& m : *materials)
    if(const Json* groups = get(&m, "groups"); groups && groups->is_array()
       && std::any_of(groups->begin(), groups->end(), [&](const Json& g) { return g.is_string() && g.get<std::string>() == surface; }))
      return &m;
  for(const Json& m : *materials)
    if(getString(&m, "id") == surface)
      return &m;
  return nullptr;
}

// PBRSkin, Iray Uber, or one of the 3Delight-era shaders: the families a preset switches between.
// The legacy shaders are told apart by marker channels a preset of the same one may leave out.
int shaderFamily(mat::Surface::Shader s)
{
  using S = mat::Surface::Shader;
  return s == S::ePbrSkin ? 1 : s == S::eIray ? 2 : 0;
}

// `m` as a whole preset makes it when its scene material `entry` names a shader of another family:
// built in DAZ's order, the definition, the entry's own channels, what carries over from the old
// material, then the preset's values. False, leaving `m`, when the family stays.
bool presetMaterial(Material& m, const Json& entry, const std::vector<ChannelOverride>& kept, const MaterialDefinitions& definitions,
                    const std::unordered_map<std::string, std::string>& images)
{
  Channels fresh;
  if(const auto def = definitions.find(definitionUrl(entry)); def != definitions.end())
    fresh = def->second;
  collectChannels(&entry, fresh, images, {});
  // DAZ keeps the old material's values where the new shader has a property of the same name (Iray
  // Optimized Genesis 2 Female MAT sets values only and keeps the skin's maps); the old shader's
  // other channels go. By id only: Uber's Horizontal Tiles is not PBRSkin's Horizontal Tiles2,
  // though that one is labelled so.
  std::set<std::string> ids;
  for(const std::string& k : fresh.order())
    if(const Channel* c = fresh.get(k); !c->id.empty())
      ids.insert(c->id);
  for(const ChannelOverride& o : kept)
    ids.insert(norm(o.channel));
  Json                     carried = Json::array();
  std::set<const Channel*> seen;
  for(const std::string& k : m.channels->order())
  {
    const Channel* c = m.channels->get(k);
    if(c->id.empty() || !ids.count(c->id) || !seen.insert(c).second)
      continue;
    Json ch = {{"id", c->id}};
    if(!c->value.is_null())
      ch["current_value"] = c->value;
    if(c->image)
      ch["image_file"] = *c->image;
    if(c->imageModification.is_object())
      ch["image_modification"] = c->imageModification;
    carried.push_back({{"channel", ch}});
  }
  const Json carry = {{"extra", Json::array({{{"channels", carried}}})}};
  collectChannels(&carry, fresh, {}, {});
  // a copy keeps the edits, the baseline and the measured UV scale
  Material next = m;
  next.channels = std::make_shared<const Channels>(std::move(fresh));
  next.surface  = parseSurface(next.name, *next.channels);
  overrideChannels(next, kept);
  if(shaderFamily(next.surface.shader) == shaderFamily(m.surface.shader))
    return false;
  m = std::move(next);
  return true;
}

}  // namespace

MaterialDefinitions definitionsIn(const Json& doc, Store& store)
{
  MaterialDefinitions out;
  const Json*         materials = get(get(&doc, "scene"), "materials");
  if(!materials || !materials->is_array())
    return out;
  for(const Json& entry : *materials)
  {
    const std::string url = definitionUrl(entry);
    if(url.empty() || out.count(url))
      continue;
    try
    {
      if(auto r = store.resolve(*parseRef(getString(&entry, "url"), ""), "material_library"))
      {
        std::unordered_map<std::string, std::string> images;
        collectImages(r->doc->json, images);
        Channels ch;
        collectChannels(r->entry, ch, images, r->file);
        out.emplace(url, std::move(ch));
      }
    }
    catch(const std::exception&)
    {
    }
  }
  return out;
}

namespace {

// The material a preset applies to whatever surfaces are selected names no group. DAZ Studio puts
// it on the selection; here that is every surface whose shader is the one it names (FeSoul's hair
// colours: the OmniHair strands, not the scalp cap). Without a shader to go by, none.
std::vector<int> selectionTargets(const Json& doc, const SceneGraph& scene)
{
  const auto lowered = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
  };
  std::string shader;
  if(const Json* materials = get(get(&doc, "scene"), "materials"); materials && materials->is_array())
    for(const Json& m : *materials)
      if(!m.contains("groups"))
        if(const auto ref = parseRef(getString(&m, "url"), ""); ref && !ref->id.empty())
        {
          shader = lowered(ref->file + "#" + ref->id);
          break;
        }
  std::vector<int> out;
  if(shader.empty())
    return out;
  for(size_t i = 0; i < scene.materials.size(); i++)
    if(lowered(scene.materials[i].surface.dazShader.url) == shader)
      out.push_back(int(i));
  return out;
}

// The surfaces one entry of a preset goes on: those it names, or for an entry made for the
// selection, those with its shader.
std::vector<int> entryTargets(const Json& doc, const SceneGraph& scene, const std::vector<std::string>& chain, const std::string& surface, bool* stray = nullptr)
{
  return surface.empty() ? selectionTargets(doc, scene) : presetTargets(scene, chain, surface, stray);
}

}  // namespace

MaterialPresetResult applyMaterialPreset(const Json& doc, SceneGraph& scene, const std::set<int>* only, const MaterialDefinitions& definitions)
{
  MaterialPresetResult                         result;
  std::unordered_map<std::string, std::string> images;
  collectImages(doc, images);
  for(const auto& [key, overrides] : parseOverrides(doc, &result.edits))
  {
    const auto& [chain, surface] = key;
    bool             stray   = false;
    std::vector<int> targets = entryTargets(doc, scene, chain, surface, &stray);
    if(only)
      std::erase_if(targets, [&](int i) { return !only->count(i); });
    const std::string label = surface.empty() ? kSelectionSurfaces : surface;
    if(targets.empty())
      result.missing.insert(label);
    else
      result.matched.insert(label);
    // made for an item the scene doesn't have (fiber brows' colour on card brows): its values, but
    // not the maps it removes, which that item never had and this one is drawn by
    std::vector<ChannelOverride> kept = overrides;
    if(stray)
      for(ChannelOverride& o : kept)
        if(o.image && o.image->empty())
          o.image.reset();
    // A whole material may name another shader. A partial one is an option made for the shader the
    // surface has (a CJ skin shine, a lip colour): swapped in, the new shader's defaults would
    // switch off what it leaves out, translucency and the dual lobe among them, so it is laid over.
    // DAZ keys the swap on the url alone; that is inferred, its preset code isn't available.
    const int   set   = int(std::count_if(kept.begin(), kept.end(), [](const ChannelOverride& o) { return o.value || (o.image && !o.image->empty()); }));
    const Json* entry = !stray && set >= kFullIrayPreset ? sceneMaterial(doc, surface) : nullptr;
    for(int i : targets)
    {
      Material& m = scene.materials[size_t(i)];
      if((entry && m.channels && presetMaterial(m, *entry, kept, definitions, images)) || overrideChannels(m, kept))
        result.changed.insert(i);
    }
  }
  return result;
}

namespace {

bool legacySurface(mat::Surface::Shader s)
{
  using S = mat::Surface::Shader;
  return s == S::eDazDefault || s == S::eUberSurface || s == S::eAoa || s == S::eLegacy;
}

// A channel only an Iray Uber or PBRSkin material has
bool irayChannel(const std::string& channel)
{
  static const std::set<std::string> ids = {"glossyroughness", "glossylayeredweight", "translucencyweight", "metallicweight",
                                            "duallobespecularweight", "specularlobe1roughness"};
  return ids.count(norm(channel)) > 0;
}

// An image path as two files name the same image: percent-decoded, one kind of slash, any case
std::string imageKey(const std::string& path)
{
  std::string s = dec(path);
  std::replace(s.begin(), s.end(), '\\', '/');
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

}  // namespace

int chooseIrayPreset(const SceneGraph& scene, const std::vector<const Json*>& candidates)
{
  std::set<int> legacy;
  for(size_t i = 0; i < scene.materials.size(); i++)
    if(legacySurface(scene.materials[i].surface.shader))
      legacy.insert(int(i));
  if(legacy.empty())
    return -1;
  int     best      = -1;
  int64_t bestScore = -1;
  for(size_t k = 0; k < candidates.size(); k++)
  {
    if(!candidates[k])
      continue;
    bool          iray = false, whole = true, keepsLook = true;
    std::set<int> covered;
    int           maps = 0, channels = 0;
    for(const auto& [key, overrides] : parseOverrides(*candidates[k], nullptr))
    {
      const std::vector<int> targets = entryTargets(*candidates[k], scene, key.first, key.second);
      if(targets.empty())
        continue;
      // what it sets: a value, or an image (a null image removes a map, which says nothing of a look)
      int set = 0;
      for(const ChannelOverride& o : overrides)
      {
        if(o.value || (o.image && !o.image->empty()))
          set++;
        // a texture instance setting alone sets nothing of Iray's
        iray = iray || ((o.value || o.image) && irayChannel(o.channel));
      }
      whole = whole && set >= kFullIrayPreset;
      channels += set;
      for(int t : targets)
      {
        if(legacy.count(t))
          covered.insert(t);
        const mat::Surface& s = scene.materials[size_t(t)].surface;
        for(const ChannelOverride& o : overrides)
        {
          if(!o.image || o.image->empty())
            continue;
          const std::string n   = norm(o.channel);
          const std::string* own = n == "diffuse" || n == "basecolor" ? &s.base.map : n == "normalmap" || n == "normal" ? &s.normalMap : nullptr;
          if(!own)
            continue;
          if(imageKey(*o.image) == imageKey(*own))
            maps++;
          else
            keepsLook = false;
        }
      }
    }
    if(!iray || !whole || !keepsLook || covered.size() * 2 < legacy.size())
      continue;
    const int64_t score = int64_t(maps) * 1000000 + channels;
    if(score > bestScore)
    {
      best      = int(k);
      bestScore = score;
    }
  }
  return best;
}

std::set<int> materialPresetTargets(const Json& doc, const SceneGraph& scene)
{
  std::set<int> out;
  for(const auto& [key, overrides] : parseOverrides(doc, nullptr))
    for(int i : entryTargets(doc, scene, key.first, key.second))
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

PresetChoice choosePreset(SceneGraph& scene, const std::set<int>& surfaces, const Json* doc, const std::set<int>& previous,
                          const MaterialDefinitions& definitions)
{
  PresetChoice  r;
  std::set<int> back = surfaces;
  back.insert(previous.begin(), previous.end());
  r.changed = restoreBaseline(scene, back);
  if(doc)
  {
    // only its part on these: what else it sets (lashes with eyes, the body with a garment) keeps
    // what was chosen for it since
    r.preset = applyMaterialPreset(*doc, scene, &surfaces, definitions);
    r.owned  = r.preset.changed;
    r.changed.insert(r.preset.changed.begin(), r.preset.changed.end());
  }
  return r;
}

}  // namespace fv::dson
