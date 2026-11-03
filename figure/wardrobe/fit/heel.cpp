#include <figure/wardrobe/fit/heel.hpp>

#include <cmath>
#include <regex>

#include <figure/wardrobe/fit/bones.hpp>

namespace fv::wardrobe {

std::vector<FootChannel> footChannels(const dson::Json& doc)
{
  std::vector<FootChannel> out;
  const dson::Json*        animations = dson::get(dson::get(&doc, "scene"), "animations");
  if(!animations || !animations->is_array())
    return out;
  static const std::regex selection(R"(^name://@selection(?:/([^:#]*))?(?:#([^:]*))?:\?(.*)$)");
  static const std::regex hierarchical(R"(^[^:]*:[^#]*#([^?]+)\?(.*)$)");
  static const std::regex rotation(R"(^rotation/([xyz])$)");
  static const std::regex footish("foot|toe|heel|metatarsal", std::regex::icase);
  for(const dson::Json& a : *animations)
  {
    const dson::Json* keys = dson::get(&a, "keys");
    if(!keys || !keys->is_array() || keys->empty())
      continue;
    const dson::Json& key = keys->back();
    if(!key.is_array() || key.size() < 2 || !key[1].is_number())
      continue;
    const double      value = key[1].get<double>();
    const std::string url   = dson::getString(&a, "url");
    std::smatch       m;
    std::string       bone, dial, prop;
    if(std::regex_match(url, m, selection) && (m[1].length() > 0 || m[2].length() > 0))
    {
      bone = dson::decodeUri(m[1].str());
      dial = dson::decodeUri(m[2].str());
      prop = m[3].str();
    }
    else if(std::regex_match(url, m, hierarchical))
    {
      bone = dson::decodeUri(m[1].str());
      prop = m[2].str();
    }
    else
      continue;
    if(prop.size() >= 6 && prop.compare(prop.size() - 6, 6, "/value") == 0)
      prop.resize(prop.size() - 6);
    // the figure's own dial (a bone's modifier is some other node's)
    if(!dial.empty())
    {
      if(bone.empty() && prop == "value" && std::regex_search(dial, footish))
        out.push_back({"", "", value, dial});
      continue;
    }
    std::smatch pm;
    if(!std::regex_match(prop, pm, rotation) || !std::regex_search(bone, footish))
      continue;
    out.push_back({bone, pm[1].str(), value});
  }
  return out;
}

std::vector<FootChannel> retargetFootPose(const std::vector<FootChannel>& channels, const std::vector<std::string>& targets)
{
  // a degree's half, a dial's hundredth: what a file writes as rest is not a pose
  bool any = false;
  for(const FootChannel& c : channels)
    any = any || std::abs(c.value) > (c.dial.empty() ? 0.5 : 0.01);
  std::vector<FootChannel> out;
  if(!any)
    return out;
  for(const FootChannel& c : channels)
    if(!c.dial.empty())
      out.push_back(c);
    else if(auto bone = matchBone(c.bone, targets))
      out.push_back({*bone, c.axis, c.value});
  return out;
}

}  // namespace fv::wardrobe
