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
  static const std::regex selection(R"(^name://@selection(?:/([^:#]*))?(?:#[^:]*)?:\?(.*)$)");
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
    std::string       bone, prop;
    if(std::regex_match(url, m, selection) && m[1].matched && m[1].length() > 0)
    {
      bone = dson::decodeUri(m[1].str());
      prop = m[2].str();
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
    std::smatch pm;
    if(!std::regex_match(prop, pm, rotation) || !std::regex_search(bone, footish))
      continue;
    out.push_back({bone, pm[1].str(), value});
  }
  return out;
}

std::vector<FootChannel> retargetFootPose(const std::vector<FootChannel>& channels, const std::vector<std::string>& targets)
{
  bool any = false;
  for(const FootChannel& c : channels)
    any = any || std::abs(c.value) > 0.5;
  std::vector<FootChannel> out;
  if(!any)
    return out;
  for(const FootChannel& c : channels)
    if(auto bone = matchBone(c.bone, targets))
      out.push_back({*bone, c.axis, c.value});
  return out;
}

}  // namespace fv::wardrobe
