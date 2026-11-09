#include <figure/rig/spec.hpp>

#include <algorithm>
#include <cctype>
#include <regex>
#include <unordered_map>

#include <figure/rig/aliases/aliases.hpp>

namespace fv::rig {

namespace {

using L  = Limb;
using R  = Region;
using G  = Group;
using P  = Part;
using LM = Landmark;

// Parents before children. Columns: id, parent, sided, fallback, aims, limb, region, group, part, landmark.
constexpr Joint kJoints[] = {
    {"hip", "", false, "", {}, L::eNone, R::ePelvis, G::eTorso, P::eOwn, LM::eNone},
    {"pelvis", "hip", false, "", {}, L::eNone, R::ePelvis, G::eTorso, P::eMerged, LM::eNone},
    {"spine1", "hip", false, "", {"spine2", "spine3"}, L::eNone, R::eAbdomen, G::eTorso, P::eOwn, LM::eNone},
    {"spine2", "spine1", false, "spine1", {"spine3"}, L::eNone, R::eAbdomen, G::eTorso, P::eMerged, LM::eNone},
    {"spine3", "spine2", false, "", {"spine4", "neck1"}, L::eNone, R::eChest, G::eTorso, P::eOwn, LM::eNone},
    {"spine4", "spine3", false, "spine3", {"neck1"}, L::eNone, R::eChest, G::eTorso, P::eMerged, LM::eNone},
    {"neck1", "spine4", false, "", {"neck2", "head"}, L::eHead, R::eNeck, G::eHeadNeck, P::eOwn, LM::eNone},
    {"neck2", "neck1", false, "neck1", {"head"}, L::eHead, R::eNeck, G::eHeadNeck, P::eMerged, LM::eNone},
    {"head", "neck2", false, "", {}, L::eHead, R::eHead, G::eHeadNeck, P::eOwn, LM::eNone},
    {"pectoral", "spine4", true, "", {}, L::eNone, R::eBreast, G::eTorso, P::eNone, LM::eNone},
    // the arm
    {"collar", "spine4", true, "", {"upperarm"}, L::eArm, R::eCollar, G::eArm, P::eOwn, LM::eNone},
    {"upperarm", "collar", true, "", {"forearm"}, L::eArm, R::eUpperArm, G::eArm, P::eOwn, LM::eFull},
    {"forearm", "upperarm", true, "", {"hand"}, L::eArm, R::eForearm, G::eArm, P::eOwn, LM::eFull},
    {"hand", "forearm", true, "", {"mid1", "midmc"}, L::eHand, R::eHand, G::eArm, P::eOwn, LM::eFull},
    {"thumb1", "hand", true, "hand", {}, L::eHand, R::eThumb, G::eArm, P::eNone, LM::eNone},
    {"thumb2", "thumb1", true, "", {}, L::eHand, R::eThumb, G::eArm, P::eNone, LM::eNone},
    {"thumb3", "thumb2", true, "", {}, L::eHand, R::eThumb, G::eArm, P::eNone, LM::eNone},
    {"indexmc", "hand", true, "hand", {}, L::eHand, R::eHand, G::eArm, P::eNone, LM::eNone},
    {"index1", "indexmc", true, "indexmc", {}, L::eHand, R::eIndex, G::eArm, P::eNone, LM::eNone},
    {"index2", "index1", true, "", {}, L::eHand, R::eIndex, G::eArm, P::eNone, LM::eNone},
    {"index3", "index2", true, "", {}, L::eHand, R::eIndex, G::eArm, P::eNone, LM::eNone},
    {"midmc", "hand", true, "hand", {}, L::eHand, R::eHand, G::eArm, P::eNone, LM::eNone},
    {"mid1", "midmc", true, "midmc", {}, L::eHand, R::eMid, G::eArm, P::eNone, LM::eNone},
    {"mid2", "mid1", true, "", {}, L::eHand, R::eMid, G::eArm, P::eNone, LM::eNone},
    {"mid3", "mid2", true, "", {}, L::eHand, R::eMid, G::eArm, P::eNone, LM::eNone},
    {"ringmc", "hand", true, "hand", {}, L::eHand, R::eHand, G::eArm, P::eNone, LM::eNone},
    {"ring1", "ringmc", true, "ringmc", {}, L::eHand, R::eRing, G::eArm, P::eNone, LM::eNone},
    {"ring2", "ring1", true, "", {}, L::eHand, R::eRing, G::eArm, P::eNone, LM::eNone},
    {"ring3", "ring2", true, "", {}, L::eHand, R::eRing, G::eArm, P::eNone, LM::eNone},
    {"pinkymc", "hand", true, "hand", {}, L::eHand, R::eHand, G::eArm, P::eNone, LM::eNone},
    {"pinky1", "pinkymc", true, "pinkymc", {}, L::eHand, R::ePinky, G::eArm, P::eNone, LM::eNone},
    {"pinky2", "pinky1", true, "", {}, L::eHand, R::ePinky, G::eArm, P::eNone, LM::eNone},
    {"pinky3", "pinky2", true, "", {}, L::eHand, R::ePinky, G::eArm, P::eNone, LM::eNone},
    // the leg
    {"thigh", "pelvis", true, "", {"shin"}, L::eThigh, R::eThigh, G::eLeg, P::eOwn, LM::eFull},
    {"shin", "thigh", true, "", {"foot"}, L::eShin, R::eShin, G::eLeg, P::eOwn, LM::eFull},
    {"foot", "shin", true, "", {"toes"}, L::eFoot, R::eFoot, G::eLeg, P::eOwn, LM::eAlongLength},
    {"metatarsal", "foot", true, "foot", {}, L::eFoot, R::eFoot, G::eLeg, P::eNone, LM::eNone},
    {"toes", "metatarsal", true, "foot", {}, L::eFoot, R::eFoot, G::eLeg, P::eNone, LM::eAlongLength},
    {"bigtoe1", "toes", true, "toes", {}, L::eFoot, R::eBigToe, G::eLeg, P::eNone, LM::eNone},
    {"bigtoe2", "bigtoe1", true, "bigtoe1", {}, L::eFoot, R::eBigToe, G::eLeg, P::eNone, LM::eNone},
    {"indextoe1", "toes", true, "toes", {}, L::eFoot, R::eIndexToe, G::eLeg, P::eNone, LM::eNone},
    {"indextoe2", "indextoe1", true, "indextoe1", {}, L::eFoot, R::eIndexToe, G::eLeg, P::eNone, LM::eNone},
    {"midtoe1", "toes", true, "toes", {}, L::eFoot, R::eMidToe, G::eLeg, P::eNone, LM::eNone},
    {"midtoe2", "midtoe1", true, "midtoe1", {}, L::eFoot, R::eMidToe, G::eLeg, P::eNone, LM::eNone},
    {"ringtoe1", "toes", true, "toes", {}, L::eFoot, R::eRingToe, G::eLeg, P::eNone, LM::eNone},
    {"ringtoe2", "ringtoe1", true, "ringtoe1", {}, L::eFoot, R::eRingToe, G::eLeg, P::eNone, LM::eNone},
    {"pinkytoe1", "toes", true, "toes", {}, L::eFoot, R::ePinkyToe, G::eLeg, P::eNone, LM::eNone},
    {"pinkytoe2", "pinkytoe1", true, "pinkytoe1", {}, L::eFoot, R::ePinkyToe, G::eLeg, P::eNone, LM::eNone},
};

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

bool upperOrUnderscore(char c)
{
  return std::isupper(static_cast<unsigned char>(c)) || c == '_';
}

// lowercase, without separators
std::string tokenOf(const std::string& s)
{
  std::string out;
  for(char c : lower(s))
    if(c != '_' && c != '-' && !std::isspace(static_cast<unsigned char>(c)))
      out += c;
  return out;
}

}  // namespace

std::span<const Joint> joints()
{
  return kJoints;
}

const Joint* joint(std::string_view id)
{
  static const std::unordered_map<std::string_view, const Joint*> byId = [] {
    std::unordered_map<std::string_view, const Joint*> m;
    for(const Joint& j : kJoints)
      m.emplace(j.id, &j);
    return m;
  }();
  const auto it = byId.find(id);
  return it == byId.end() ? nullptr : it->second;
}

TwistPlacement twistPlacement(Family family)
{
  switch(family)
  {
    case Family::eGenesis3:
    case Family::eGenesis8:
      return TwistPlacement::eInChain;
    case Family::eGenesis9:
    case Family::eSkyrim:
      return TwistPlacement::eHelpers;
    case Family::eGenesis2:
      break;
  }
  return TwistPlacement::eNone;
}

Name parse(const std::string& name)
{
  Name                    out;
  static const std::regex twist(R"(twist\d?$)");
  // Skyrim's: "NPC L Thigh [LThg]", "NPC Spine1 [Spn1]", "NPC Head [Head]"
  if(name.size() > 4 && name.compare(0, 4, "NPC ") == 0)
  {
    std::string rest = name.substr(4);
    if(const size_t tag = rest.find(" ["); tag != std::string::npos)
      rest.resize(tag);
    if(rest.size() > 2 && (rest[0] == 'L' || rest[0] == 'R') && rest[1] == ' ')
    {
      out.side = rest[0] == 'L' ? 'l' : 'r';
      rest     = rest.substr(2);
    }
    out.token     = tokenOf(rest);
    const auto it = aliases::skyrim().find(out.token);
    out.slot      = it != aliases::skyrim().end() ? it->second : out.token;
    out.twist     = std::regex_search(out.token, twist);
    out.skyrim    = true;
    out.joint     = joint(out.slot);
    return out;
  }
  std::string rest     = name;
  const auto  prefixed = [&](const char* word, size_t at) {
    // `left` / `right` in any case, followed by a letter or underscore
    const size_t n = std::char_traits<char>::length(word);
    return name.size() > n && lower(name.substr(at, n)) == word && (std::isalpha(static_cast<unsigned char>(name[n])) || name[n] == '_');
  };
  if(name.size() > 1 && name[0] == 'l' && upperOrUnderscore(name[1]))
  {
    out.side = 'l';
    rest     = name.substr(name.size() > 2 && name[1] == '_' ? 2 : 1);
  }
  else if(prefixed("left", 0))
  {
    out.side = 'l';
    rest     = name.substr(4 + (name.size() > 4 && name[4] == '_' ? 1 : 0));
  }
  else if(name.size() > 1 && name[0] == 'r' && upperOrUnderscore(name[1]))
  {
    out.side = 'r';
    rest     = name.substr(name.size() > 2 && name[1] == '_' ? 2 : 1);
  }
  else if(prefixed("right", 0))
  {
    out.side = 'r';
    rest     = name.substr(5 + (name.size() > 5 && name[5] == '_' ? 1 : 0));
  }
  out.token     = tokenOf(rest);
  const auto it = aliases::genesis().find(out.token);
  out.slot      = it != aliases::genesis().end() ? it->second : out.token;
  out.twist     = std::regex_search(out.token, twist);
  out.joint     = joint(out.slot);
  return out;
}

const Joint* genesisJoint(const std::string& token)
{
  const auto it = aliases::genesis().find(token);
  return joint(it != aliases::genesis().end() ? std::string_view(it->second) : std::string_view(token));
}

}  // namespace fv::rig
