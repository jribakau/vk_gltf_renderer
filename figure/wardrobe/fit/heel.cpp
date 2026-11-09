#include <figure/wardrobe/fit/heel.hpp>

#include <algorithm>
#include <cmath>
#include <regex>
#include <set>

#include <figure/rig/spec.hpp>
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

std::vector<FootChannel> retargetFootPose(const std::vector<FootChannel>& channels, const std::vector<std::string>& targets,
                                          const std::map<std::string, BoneFrame>& from, const std::map<std::string, BoneFrame>& to)
{
  // a degree's half, a dial's hundredth: what a file writes as rest is not a pose
  bool any = false;
  for(const FootChannel& c : channels)
    any = any || std::abs(c.value) > (c.dial.empty() ? 0.5 : 0.01);
  std::vector<FootChannel> out;
  if(!any)
    return out;
  const auto same = [](const BoneFrame& a, const BoneFrame& b) {
    for(size_t i = 0; i < 3; i++)
      if(std::abs(a.orientation[i] - b.orientation[i]) > 1e-9)
        return false;
    return a.order == b.order;
  };
  std::set<std::string> turned;
  for(const FootChannel& c : channels)
  {
    if(!c.dial.empty())
    {
      out.push_back(c);
      continue;
    }
    const auto bone = matchBone(c.bone, targets);
    if(!bone)
      continue;
    const auto f = from.find(c.bone);
    const auto t = to.find(*bone);
    if(f == from.end() || t == to.end() || same(f->second, t->second))
    {
      out.push_back({*bone, c.axis, c.value});
      continue;
    }
    if(!turned.insert(c.bone).second)
      continue;
    // the bone's angles together, those the file leaves out at rest, held to the limits the bone it was
    // made on keeps
    std::array<double, 3> angles{};
    for(const FootChannel& o : channels)
      if(o.dial.empty() && o.bone == c.bone && o.axis.size() == 1 && o.axis[0] >= 'x' && o.axis[0] <= 'z')
      {
        const size_t i = size_t(o.axis[0] - 'x');
        angles[i]      = std::clamp(o.value, f->second.min[i], std::max(f->second.min[i], f->second.max[i]));
      }
    const std::array<double, 3> values = rig::boneValues(rig::boneTurn(angles, f->second.orientation, f->second.order), t->second.orientation, t->second.order);
    for(size_t i = 0; i < 3; i++)
      out.push_back({*bone, std::string(1, char('x' + i)), std::abs(values[i]) < 1e-9 ? 0 : values[i]});
  }
  return out;
}

}  // namespace fv::wardrobe
