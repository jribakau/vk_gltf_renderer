#include <figure/dson/material_preset.hpp>

#include <map>
#include <regex>

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

}  // namespace

MaterialPresetResult applyMaterialPreset(const Json& doc, SceneGraph& scene)
{
  MaterialPresetResult result;
  const Json*          animations = get(get(&doc, "scene"), "animations");
  if(!animations || !animations->is_array())
    return result;
  std::unordered_map<std::string, std::string> images;
  collectImages(doc, images);

  // surface -> channel -> override, in the order the preset lists them
  std::map<std::string, std::vector<ChannelOverride>> bySurface;
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
    auto&             list    = bySurface[surface];
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

  for(const auto& [surface, overrides] : bySurface)
  {
    bool hit = false;
    for(size_t i = 0; i < scene.materials.size(); i++)
      if(scene.materials[i].name == surface)
      {
        hit = true;
        if(overrideChannels(scene.materials[i], overrides))
          result.changed.insert(int(i));
      }
    if(!hit)
      result.missing.insert(surface);
  }
  return result;
}

}  // namespace fv::dson
