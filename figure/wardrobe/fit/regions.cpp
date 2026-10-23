#include <figure/wardrobe/fit/regions.hpp>

#include <algorithm>
#include <regex>
#include <set>

#include <figure/wardrobe/fit/bones.hpp>

namespace fv::wardrobe {

int16_t limbOfBone(const std::string& name)
{
  static const std::set<std::string> foot{"foot", "metatarsal", "toes"};
  static const std::set<std::string> hand{"hand", "indexmc", "midmc", "ringmc", "pinkymc"};
  static const std::set<std::string> arm{"collar", "upperarm", "forearm"};
  static const std::set<std::string> head{"head", "neck1", "neck2"};
  static const std::regex            toeN(R"(toe\d$)");
  static const std::regex            fingerN(R"(^(thumb|index|mid|ring|pinky)\d$)");
  const ParsedBone                   p    = parseBone(name);
  const bool                         left = p.side != 'r';
  if(foot.count(p.slot) || std::regex_search(p.slot, toeN))
    return left ? limb::kLFoot : limb::kRFoot;
  if(p.slot == "shin")
    return left ? limb::kLShin : limb::kRShin;
  if(p.slot == "thigh")
    return left ? limb::kLThigh : limb::kRThigh;
  if(hand.count(p.slot) || std::regex_search(p.slot, fingerN))
    return left ? limb::kLHand : limb::kRHand;
  if(arm.count(p.slot))
    return left ? limb::kLArm : limb::kRArm;
  if(head.count(p.slot))
    return limb::kHead;
  return 0;
}

FitRegion regionOf(const std::string& path, const std::vector<std::string>& boneNames)
{
  std::string p = path;
  std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  for(size_t at = p.find("%20"); at != std::string::npos; at = p.find("%20", at + 1))
    p.replace(at, 3, " ");
  static const std::regex thighHigh(R"(thigh\s*-?\s*high|over\s*-?\s*knee)");
  static const std::regex boot(R"(/boots?\b|\bboots?\b)");
  static const std::regex shoe(R"(/shoes?\b|\bfootwear\b|\bheels?\b|highheels?\b|\bsneakers?\b|\bpumps?\b|\bsandals?\b)");
  static const std::regex glove(R"(/gloves?\b|\bgloves?\b)");
  static const std::regex head(R"(/hair\b|\bhelmet|\bglasses\b|\beyewear\b)");
  if(std::regex_search(p, thighHigh))
    return FitRegion::eBootThigh;
  if(std::regex_search(p, boot))
    return FitRegion::eBoot;
  if(std::regex_search(p, shoe))
    return FitRegion::eShoe;
  if(std::regex_search(p, glove))
    return FitRegion::eGlove;
  if(std::regex_search(p, head))
    return FitRegion::eHead;
  int foot = 0, hand = 0, headN = 0, other = 0;
  for(const std::string& name : boneNames)
  {
    const int16_t l = limbOfBone(name);
    if(l == limb::kLFoot || l == limb::kRFoot)
      foot++;
    else if(l == limb::kLHand || l == limb::kRHand)
      hand++;
    else if(l == limb::kHead)
      headN++;
    else if(l)
      other++;
  }
  if(foot && foot > other)
    return FitRegion::eShoe;
  if(hand && hand > other)
    return FitRegion::eGlove;
  if(headN && !other && !foot)
    return FitRegion::eHead;
  return FitRegion::eBody;
}

std::vector<int16_t> vertexLimbs(size_t count, const mesh::SkinInput* skin, const std::vector<int16_t>& boneLimb)
{
  std::vector<int16_t> out(count, 0);
  if(!skin)
    return out;
  for(size_t v = 0; v < count && v + 1 < skin->start.size(); v++)
  {
    float   best = 0;
    int16_t l    = 0;
    for(int32_t i = skin->start[v]; i < skin->start[v + 1]; i++)
    {
      const float w = skin->weight[size_t(i)];
      if(w > best)
      {
        best = w;
        const size_t b = size_t(skin->bone[size_t(i)]);
        l              = b < boneLimb.size() ? boneLimb[b] : int16_t(0);
      }
    }
    out[v] = l;
  }
  return out;
}

std::optional<BindFilter> limbFilter(const std::vector<int16_t>& bodyLimbs, const std::vector<int16_t>& itemLimbs)
{
  if(std::none_of(bodyLimbs.begin(), bodyLimbs.end(), [](int16_t l) { return l != 0; }))
    return std::nullopt;
  BindFilter f;
  f.bodyLimb = bodyLimbs;
  f.itemMask.resize(itemLimbs.size());
  for(size_t v = 0; v < itemLimbs.size(); v++)
    f.itemMask[v] = itemLimbs[v] ? (1 << itemLimbs[v]) : 0;
  return f;
}

}  // namespace fv::wardrobe
