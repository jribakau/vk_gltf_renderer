#include <figure/dson/modifiers.hpp>

#include <cctype>

namespace fv::dson {

namespace {

const char* const kSmoothingType = "studio/modifier/smoothing";

std::optional<std::pair<const Json*, const Doc*>> lookupDefinition(const Doc& scene, const Doc* geo, const std::string& url)
{
  if(url.empty())
    return std::nullopt;
  const auto ref = parseRef(url, "");
  if(!ref || ref->id.empty())
    return std::nullopt;
  for(const Doc* doc : {&scene, geo})
  {
    if(!doc)
      continue;
    const Json* lib = get(&doc->json, "modifier_library");
    if(!lib || !lib->is_array())
      continue;
    for(const Json& m : *lib)
      if(getString(&m, "id") == ref->id)
        return std::make_pair(&m, doc);
  }
  return std::nullopt;
}

}  // namespace

std::string modifierParent(const Json& mod)
{
  std::string p = getString(&mod, "parent");
  if(p.empty() || p[0] != '#')
    return p;
  std::string out;
  for(size_t i = 1; i < p.size(); i++)
  {
    if(p[i] == '%' && i + 2 < p.size() && std::isxdigit((unsigned char)p[i + 1]) && std::isxdigit((unsigned char)p[i + 2]))
    {
      out += char(std::stoi(p.substr(i + 1, 2), nullptr, 16));
      i += 2;
    }
    else
      out += p[i];
  }
  return out;
}

bool hasExtraType(const Json& mod, const char* type)
{
  const Json* extra = get(&mod, "extra");
  if(!extra || !extra->is_array())
    return false;
  for(const Json& e : *extra)
    if(getString(&e, "type") == type)
      return true;
  return false;
}

ChannelValues modifierChannels(const Json& mod)
{
  ChannelValues out;
  const auto    put = [&](const Json& ch) {
    const Json* c  = get(&ch, "channel");
    const Json& cc = c && c->is_object() ? *c : ch;
    const Json* id = get(&cc, "id");
    if(!id || !id->is_string())
      return;
    const Json* v = get(&cc, "current_value");
    if(!v || v->is_null())
      v = get(&cc, "value");
    if(v && (v->is_number() || v->is_boolean() || v->is_string()))
      out[id->get<std::string>()] = *v;
  };
  if(const Json* extra = get(&mod, "extra"); extra && extra->is_array())
    for(const Json& ex : *extra)
      if(const Json* chans = get(&ex, "channels"); chans && chans->is_array())
        for(const Json& c : *chans)
          put(c);
  if(const Json* chans = get(&mod, "studio_modifier_channels"); chans && chans->is_array())
    for(const Json& c : *chans)
      put(c);
  if(get(&mod, "channel"))
    put(mod);
  return out;
}

ChannelValues mergedModifierChannels(const Json& instance, const Doc& scene, const Doc* geo)
{
  ChannelValues out;
  if(const auto def = lookupDefinition(scene, geo, getString(&instance, "url")))
    out = modifierChannels(*def->first);
  for(auto& [k, v] : modifierChannels(instance))
    out[k] = v;
  return out;
}

double numberOf(const ChannelValues& ch, const char* id, double fallback)
{
  const auto it = ch.find(id);
  return it != ch.end() && it->second.is_number() ? it->second.get<double>() : fallback;
}

bool flagOf(const ChannelValues& ch, const char* id, bool fallback)
{
  const auto it = ch.find(id);
  if(it == ch.end())
    return fallback;
  if(it->second.is_boolean())
    return it->second.get<bool>();
  if(it->second.is_number())
    return it->second.get<double>() != 0;
  return fallback;
}

std::optional<Smoothing> parseSmoothing(const Doc& scene, const Doc* geo, const std::string& nodeId)
{
  const Json* mods = get(get(&scene.json, "scene"), "modifiers");
  if(!mods || !mods->is_array())
    return std::nullopt;
  for(const Json& m : *mods)
  {
    if(!hasExtraType(m, kSmoothingType) || modifierParent(m) != nodeId)
      continue;
    const ChannelValues ch = mergedModifierChannels(m, scene, geo);
    Smoothing           s;
    s.enabled   = flagOf(ch, "Enable Smoothing", true);
    s.smoothing = int(numberOf(ch, "Smoothing Iterations", s.smoothing));
    s.collision = int(numberOf(ch, "Collision Iterations", s.collision));
    return s;
  }
  return std::nullopt;
}

}  // namespace fv::dson
