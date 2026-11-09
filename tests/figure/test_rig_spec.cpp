// The skeleton spec (wardrobe plan step 2a): the joint table every rig maps into, and the readers that
// became shims over it answering byte for byte as the tables they replaced. The old readers are kept
// here, frozen, as the golden answers: on names made from every alias in every spelling a rig uses,
// and, with FV_RIG_NAMES naming a file of every installed rig's bones (one "R\t<file>\t<figure>" line a
// rig, then "B\t<name>\t<parent>" a bone, made by a script that saves it outside the repository), on
// the library's.

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <cmath>

#include <figure/dson/rig.hpp>
#include <figure/math/xform.hpp>
#include <figure/pose/pose_rig.hpp>
#include <figure/scene/graph.hpp>
#include <figure/rig/aliases/aliases.hpp>
#include <figure/rig/spec.hpp>
#include <figure/wardrobe/fit/bones.hpp>
#include <figure/wardrobe/fit/generation.hpp>
#include <figure/wardrobe/fit/regions.hpp>
#include <figure/wardrobe/fit/transfer.hpp>

namespace {

// ---- the readers as they were before the spec (renderer 33709c0), frozen
namespace old {

const std::unordered_map<std::string, std::string>& slots()
{
  static const std::unordered_map<std::string, std::string> table{
      {"thighbend", "thigh"},        {"thightwist", "thigh"},       {"thightwist1", "thigh"},      {"thightwist2", "thigh"},
      {"metatarsals", "metatarsal"}, {"toe", "toes"},               {"heel", "foot"},              {"bigtoe", "bigtoe1"},
      {"smalltoe1", "indextoe1"},    {"smalltoe12", "indextoe2"},   {"smalltoe2", "midtoe1"},      {"smalltoe22", "midtoe2"},
      {"smalltoe3", "ringtoe1"},     {"smalltoe32", "ringtoe2"},    {"smalltoe4", "pinkytoe1"},    {"smalltoe42", "pinkytoe2"},
      {"abdomen", "spine1"},         {"abdomenlower", "spine1"},    {"abdomen2", "spine2"},        {"abdomenupper", "spine2"},
      {"chest", "spine3"},           {"chestlower", "spine3"},      {"chestupper", "spine4"},      {"neck", "neck1"},
      {"necklower", "neck1"},        {"neckupper", "neck2"},
      {"shoulder", "collar"},        {"shldr", "upperarm"},         {"shldrbend", "upperarm"},     {"shldrtwist", "upperarm"},
      {"upperarmtwist1", "upperarm"}, {"upperarmtwist2", "upperarm"}, {"forearmbend", "forearm"},   {"forearmtwist", "forearm"},
      {"forearmtwist1", "forearm"},  {"forearmtwist2", "forearm"},  {"handanchor", "hand"},        {"carpal1", "indexmc"},
      {"carpal2", "midmc"},          {"carpal3", "ringmc"},         {"carpal4", "pinkymc"},        {"indexmetacarpal", "indexmc"},
      {"midmetacarpal", "midmc"},    {"ringmetacarpal", "ringmc"},  {"pinkymetacarpal", "pinkymc"},
  };
  return table;
}

const std::unordered_map<std::string, std::string>& fallbacks()
{
  static const std::unordered_map<std::string, std::string> table{
      {"metatarsal", "foot"}, {"toes", "foot"},       {"bigtoe1", "toes"},    {"bigtoe2", "bigtoe1"}, {"indextoe1", "toes"},
      {"indextoe2", "indextoe1"}, {"midtoe1", "toes"}, {"midtoe2", "midtoe1"}, {"ringtoe1", "toes"},  {"ringtoe2", "ringtoe1"},
      {"pinkytoe1", "toes"},  {"pinkytoe2", "pinkytoe1"}, {"spine2", "spine1"}, {"spine4", "spine3"}, {"neck2", "neck1"},
      {"indexmc", "hand"},    {"midmc", "hand"},      {"ringmc", "hand"},     {"pinkymc", "hand"},    {"thumb1", "hand"},
      {"index1", "indexmc"},  {"mid1", "midmc"},      {"ring1", "ringmc"},    {"pinky1", "pinkymc"},
  };
  return table;
}

const std::unordered_map<std::string, std::string>& skyrimSlots()
{
  static const std::unordered_map<std::string, std::string> table{
      {"com", "hip"},           {"spine", "spine1"},     {"spine1", "spine2"},          {"spine2", "spine3"},
      {"neck", "neck1"},        {"clavicle", "collar"},  {"upperarmtwist1", "upperarm"}, {"upperarmtwist2", "upperarm"},
      {"forearmtwist1", "forearm"}, {"forearmtwist2", "forearm"}, {"calf", "shin"},     {"toe0", "toes"},
      {"breast", "pectoral"},   {"finger00", "thumb1"},  {"finger01", "thumb2"},        {"finger02", "thumb3"},
      {"finger10", "index1"},   {"finger11", "index2"},  {"finger12", "index3"},        {"finger20", "mid1"},
      {"finger21", "mid2"},     {"finger22", "mid3"},    {"finger30", "ring1"},         {"finger31", "ring2"},
      {"finger32", "ring3"},    {"finger40", "pinky1"},  {"finger41", "pinky2"},        {"finger42", "pinky3"},
  };
  return table;
}

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

bool upperOrUnderscore(char c)
{
  return std::isupper(static_cast<unsigned char>(c)) || c == '_';
}

fv::wardrobe::ParsedBone parseBone(const std::string& name)
{
  fv::wardrobe::ParsedBone out;
  static const std::regex  twist(R"(twist\d?$)");
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
    for(char c : lower(rest))
      if(c != '_' && c != '-' && !std::isspace(static_cast<unsigned char>(c)))
        out.token += c;
    const auto it = skyrimSlots().find(out.token);
    out.slot      = it != skyrimSlots().end() ? it->second : out.token;
    out.twist     = std::regex_search(out.token, twist);
    return out;
  }
  std::string rest     = name;
  const auto  prefixed = [&](const char* word, size_t at) {
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
  for(char c : lower(rest))
    if(c != '_' && c != '-' && !std::isspace(static_cast<unsigned char>(c)))
      out.token += c;
  const auto it = slots().find(out.token);
  out.slot      = it != slots().end() ? it->second : out.token;
  out.twist     = std::regex_search(out.token, twist);
  return out;
}

std::optional<std::string> matchBone(const std::string& name, const std::vector<std::string>& targets)
{
  if(std::find(targets.begin(), targets.end(), name) != targets.end())
    return name;
  const fv::wardrobe::ParsedBone                                  parsed = parseBone(name);
  std::vector<std::pair<std::string, fv::wardrobe::ParsedBone>> parsedTargets;
  for(const std::string& t : targets)
    parsedTargets.emplace_back(t, parseBone(t));
  std::set<std::string> seen;
  std::string           slot = parsed.slot;
  for(;;)
  {
    if(!seen.insert(slot).second)
      break;
    std::optional<std::string> twist;
    for(const auto& [t, p] : parsedTargets)
    {
      if(p.side != parsed.side || p.slot != slot)
        continue;
      if(!p.twist)
        return t;
      if(!twist)
        twist = t;
    }
    if(twist)
      return twist;
    const auto next = fallbacks().find(slot);
    if(next == fallbacks().end())
      break;
    slot = next->second;
  }
  return std::nullopt;
}

fv::wardrobe::LeaderPlan planLeaders(const std::vector<fv::wardrobe::BoneRef>& bones, const std::vector<std::string>& targets)
{
  std::map<std::string, const fv::wardrobe::BoneRef*> byName;
  for(const fv::wardrobe::BoneRef& b : bones)
    byName[b.name] = &b;
  std::vector<std::pair<std::string, fv::wardrobe::ParsedBone>> posableTwists;
  for(const std::string& t : targets)
    if(const fv::wardrobe::ParsedBone p = parseBone(t); p.twist && !p.token.empty() && !std::isdigit(static_cast<unsigned char>(p.token.back())))
      posableTwists.emplace_back(t, p);
  std::map<std::string, std::string> direct;
  for(const fv::wardrobe::BoneRef& b : bones)
    if(auto leader = matchBone(b.name, targets))
    {
      direct[b.name] = *leader;
      if(const fv::wardrobe::ParsedBone p = parseBone(b.name); p.twist)
        for(const auto& [t, tp] : posableTwists)
          if(tp.side == p.side && tp.slot == p.slot)
          {
            direct[b.name] = t;
            break;
          }
    }
  fv::wardrobe::LeaderPlan plan;
  const auto               ancestorLeader = [&](const std::string& name) -> std::optional<std::string> {
    auto                       it     = byName.find(name);
    std::optional<std::string> parent = it != byName.end() ? it->second->parent : std::nullopt;
    std::set<std::string>      seen;
    while(parent && seen.insert(*parent).second)
    {
      if(auto d = direct.find(*parent); d != direct.end())
        return d->second;
      auto p = byName.find(*parent);
      parent = p != byName.end() ? p->second->parent : std::nullopt;
    }
    return std::nullopt;
  };
  for(const fv::wardrobe::BoneRef& b : bones)
  {
    const auto leader = direct.find(b.name);
    if(leader == direct.end())
    {
      if(ancestorLeader(b.name))
        plan.inherit.insert(b.name);
      continue;
    }
    if(ancestorLeader(b.name) == std::optional<std::string>(leader->second))
      plan.inherit.insert(b.name);
    else
      plan.follow[b.name] = leader->second;
  }
  return plan;
}

int16_t limbOfBone(const std::string& name)
{
  namespace limb = fv::wardrobe::limb;
  static const std::set<std::string> foot{"foot", "metatarsal", "toes"};
  static const std::set<std::string> hand{"hand", "indexmc", "midmc", "ringmc", "pinkymc"};
  static const std::set<std::string> arm{"collar", "upperarm", "forearm"};
  static const std::set<std::string> head{"head", "neck1", "neck2"};
  static const std::regex            toeN(R"(toe\d$)");
  static const std::regex            fingerN(R"(^(thumb|index|mid|ring|pinky)\d$)");
  const fv::wardrobe::ParsedBone     p    = parseBone(name);
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

std::string groupOf(const std::string& name, bool isRoot)
{
  if(isRoot)
    return "Figure";
  const fv::wardrobe::ParsedBone p = parseBone(name);
  const std::string&             n = p.token;
  static const std::regex        face(
      R"(^(eye|ear)|brow|lid|lip|cheek|nose|nostril|jaw|tongue|teeth|mouth|chin|squint|nasolabial|face|forehead|temple|upperfacerig|lowerfacerig|belowjaw|center|lash)");
  static const std::regex head(R"(^(head|neck|necklower|neckupper))");
  static const std::regex torso(R"(^(hip|pelvis|abdomen|chest|spine|pectoral))");
  static const std::regex arm(R"((collar|shldr|shoulder|upperarm|forearm|hand|carpal|thumb|index|mid|ring|pinky))");
  static const std::regex leg(R"((thigh|shin|foot|toe|metatarsal|heel))");
  if(std::regex_search(n, face))
    return "Face";
  if(std::regex_search(n, head))
    return "Head & neck";
  if(std::regex_search(n, torso))
    return "Torso";
  const std::string side = p.side == 'l' ? "Left" : p.side == 'r' ? "Right" : "";
  if(!side.empty() && std::regex_search(n, leg))
    return side + " leg";
  if(!side.empty() && std::regex_search(n, arm))
    return side + " arm";
  return "Other";
}

const std::map<std::string, std::vector<std::string>>& nextSlots()
{
  static const std::map<std::string, std::vector<std::string>> table{
      {"thigh", {"shin"}},       {"shin", {"foot"}},          {"foot", {"toes"}},           {"collar", {"upperarm"}},
      {"upperarm", {"forearm"}}, {"forearm", {"hand"}},       {"hand", {"mid1", "midmc"}},  {"spine1", {"spine2", "spine3"}},
      {"spine2", {"spine3"}},    {"spine3", {"spine4", "neck1"}}, {"spine4", {"neck1"}},    {"neck1", {"neck2", "head"}},
      {"neck2", {"head"}},
  };
  return table;
}

const std::set<std::string> kPartSlots{"hip", "spine1", "spine3", "neck1", "head", "collar", "upperarm", "forearm", "hand", "thigh", "shin", "foot"};
const std::set<std::string> kMergedSlots{"pelvis", "spine2", "spine4", "neck2"};

std::optional<fv::wardrobe::Generation> generationFromBones(const std::vector<std::string>& names)
{
  using fv::wardrobe::Generation;
  static const std::regex underscored(R"(^[lr]_(thigh|shin|foot|upperarm|forearm|hand))");
  static const std::regex heelRe(R"(^[lr]heel$)");
  static const std::regex bendRe(R"(^[lr](thigh|shldr|forearm)(bend|twist)$)");
  static const std::regex trunkRe(R"(^(abdomen|chest|neck)(lower|upper)$)");
  static const std::regex plainRe(R"(^[lr]thigh$)");
  bool                    under = false, heel = false, g3or8 = false, plain = false;
  for(const std::string& name : names)
  {
    const std::string n = lower(name);
    if(n.rfind("npc ", 0) == 0)
      return Generation::eSkyrim;
    under = under || std::regex_search(n, underscored);
    heel  = heel || std::regex_search(n, heelRe);
    g3or8 = g3or8 || std::regex_search(n, bendRe) || std::regex_search(n, trunkRe);
    plain = plain || std::regex_search(n, plainRe);
  }
  if(under)
    return Generation::eG9;
  if(heel)
    return Generation::eG3;
  if(g3or8)
    return Generation::eG8;
  if(plain)
    return Generation::eG2;
  return std::nullopt;
}

}  // namespace old

// What each reader says of one name, old and new alike, as text: a difference prints itself.
std::string answers(const std::string& name, bool fresh)
{
  const fv::wardrobe::ParsedBone p = fresh ? fv::wardrobe::parseBone(name) : old::parseBone(name);
  return std::string(1, p.side ? p.side : '-') + "|" + p.token + "|" + p.slot + "|" + (p.twist ? "t" : "-") + "|"
         + std::to_string(fresh ? fv::wardrobe::limbOfBone(name) : old::limbOfBone(name)) + "|"
         + (fresh ? fv::pose::groupOf(name, false) : old::groupOf(name, false));
}

std::string leaders(const fv::wardrobe::LeaderPlan& plan)
{
  std::string out;
  for(const auto& [a, b] : plan.follow)
    out += a + ">" + b + ";";
  out += "|";
  for(const std::string& i : plan.inherit)
    out += i + ";";
  return out;
}

// Names in every spelling the readers take: each alias and joint id, Genesis 8's camel case and 9's
// underscores on both sides, Skyrim's "NPC L ... [tag]", and words no table knows (a face rig's, a
// skirt's).
std::vector<std::string> spellings()
{
  std::set<std::string> tokens;
  for(const auto& [t, s] : fv::rig::aliases::genesis())
    tokens.insert(t);
  for(const auto& [t, s] : fv::rig::aliases::skyrim())
    tokens.insert(t);
  for(const fv::rig::Joint& j : fv::rig::joints())
    tokens.insert(std::string(j.id));
  for(const char* w : {"eye", "ear", "lowerjaw", "tongue01", "browinner", "lipcorner", "cheekupper", "nosetip", "lash", "center", "skirtfront",
                       "skirt01", "hair1", "breast01", "breast03", "butt", "belly", "toe5", "pinky4", "bigtoe3", "thumb4", "heel", "upperarmtwist3",
                       "thightwist3", "foretwist", "calf", "finger03", "magic", "weapon", "shield", "", "l", "r"})
    tokens.insert(w);
  std::vector<std::string> out;
  for(const std::string& t : tokens)
  {
    std::string cap = t;
    if(!cap.empty())
      cap[0] = char(std::toupper(static_cast<unsigned char>(cap[0])));
    for(const std::string& n : {t, cap, "l" + cap, "r" + cap, "l_" + t, "r_" + t, "left" + cap, "Right" + cap, "Left_" + t, "lt" + t, "rr" + cap,
                                "NPC " + cap + " [Tag]", "NPC L " + cap + " [LTag]", "NPC R " + cap, "NPC " + cap, "npc " + t, cap + "_2", "l" + cap + "Twist",
                                "r" + cap + "Bend", "l_" + t + "_twist1", t + "-end", "L " + cap})
      out.push_back(n);
  }
  return out;
}

}  // namespace

// The table hangs together: ids are unique, every parent, fallback and aim is a joint, a parent comes
// before its children, and a fallback is a joint up the chain or the foot.
TEST(RigSpec, EveryJointNamesOnlyJointsAndParentsComeFirst)
{
  std::set<std::string_view> seen;
  for(const fv::rig::Joint& j : fv::rig::joints())
  {
    EXPECT_TRUE(seen.insert(j.id).second) << j.id;
    EXPECT_EQ(fv::rig::joint(j.id), &j) << j.id;
    if(!j.parent.empty())
      EXPECT_TRUE(seen.count(j.parent)) << j.id << " before its parent " << j.parent;
    if(!j.fallback.empty())
      EXPECT_NE(fv::rig::joint(j.fallback), nullptr) << j.id;
    for(std::string_view a : j.aims)
      if(!a.empty())
        EXPECT_NE(fv::rig::joint(a), nullptr) << j.id << " aims at " << a;
  }
  for(const auto& [token, id] : fv::rig::aliases::genesis())
    EXPECT_NE(fv::rig::joint(id), nullptr) << token;
  for(const auto& [token, id] : fv::rig::aliases::skyrim())
    EXPECT_NE(fv::rig::joint(id), nullptr) << token;
  EXPECT_EQ(fv::rig::joint("lip"), nullptr);
  // the pectoral is never a landmark, the foot only along its length
  EXPECT_EQ(fv::rig::joint("pectoral")->landmark, fv::rig::Landmark::eNone);
  EXPECT_EQ(fv::rig::joint("foot")->landmark, fv::rig::Landmark::eAlongLength);
  EXPECT_EQ(fv::rig::twistPlacement(fv::rig::Family::eGenesis8), fv::rig::TwistPlacement::eInChain);
  EXPECT_EQ(fv::rig::twistPlacement(fv::rig::Family::eGenesis9), fv::rig::TwistPlacement::eHelpers);
  EXPECT_EQ(fv::rig::twistPlacement(fv::rig::Family::eGenesis2), fv::rig::TwistPlacement::eNone);
}

// The conventions the spec writes down (rig/spec.hpp), as the rig poses a bone: for a rotation order
// "ABC" the turn is O R_C(c) R_B(b) R_A(a) O^-1, the pose's first axis innermost, with O = R_A R_B R_C
// of the orientation's angles, its first axis outermost; translation is in the parent's frame, not
// turned by O; scale is along the rest's axes, before the turn. Built from axis-angle turns, not from
// the code's Euler helpers, so a change to either shows.
TEST(RigSpec, ABonesTurnTranslationAndScaleAreTheSpecsConventions)
{
  const std::string channels = R"([{"id":"x","min":-180,"max":180},{"id":"y","min":-180,"max":180},{"id":"z","min":-180,"max":180}])";
  const auto        rig      = [&](const std::string& order) {
    return R"([{"id":"hip","type":"figure","name":"hip","center_point":[{"id":"y","value":100}]},)"
           R"({"id":"lForearm","type":"bone","name":"lForearm","parent":"#hip","center_point":[{"id":"x","value":30},{"id":"y","value":140}],)"
           R"("rotation_order":")" + order + R"(","orientation":[{"id":"x","value":12},{"id":"y","value":30},{"id":"z","value":-25}],)"
           R"("rotation":)" + channels + R"(,"translation":)" + channels + R"(,"scale":)" + channels + "}]";
  };
  const auto turn = [](char axis, double degrees) {
    const double h = degrees * fv::kDeg / 2;
    return fv::Quat{axis == 'X' ? std::sin(h) : 0, axis == 'Y' ? std::sin(h) : 0, axis == 'Z' ? std::sin(h) : 0, std::cos(h)};
  };
  const std::map<char, double> orient{{'X', 12}, {'Y', 30}, {'Z', -25}}, angles{{'X', 25}, {'Y', -35}, {'Z', 15}};
  for(const std::string order : {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"})
  {
    fv::SceneGraph scene;
    const auto     built = fv::dson::buildRig(scene, fv::dson::Json::parse(rig(order)), nullptr);
    ASSERT_TRUE(built);
    fv::pose::PoseRig pose(scene, *built, {});
    const int         node = built->byId("lForearm");
    const fv::Vec3    rest = scene.nodes[size_t(node)].position;
    pose.setMany({{"lForearm", fv::pose::ChannelKind::eRotation, "x", angles.at('X')},
                  {"lForearm", fv::pose::ChannelKind::eRotation, "y", angles.at('Y')},
                  {"lForearm", fv::pose::ChannelKind::eRotation, "z", angles.at('Z')},
                  {"lForearm", fv::pose::ChannelKind::eTranslation, "x", 3},
                  {"lForearm", fv::pose::ChannelKind::eScale, "y", 2}});
    const char     a = order[0], b = order[1], c = order[2];
    const fv::Quat o = turn(a, orient.at(a)) * turn(b, orient.at(b)) * turn(c, orient.at(c));
    const fv::Quat q = o * (turn(c, angles.at(c)) * turn(b, angles.at(b)) * turn(a, angles.at(a))) * o.inverse();
    const fv::Quat got = scene.nodes[size_t(node)].quaternion;
    EXPECT_NEAR(std::abs(got.x * q.x + got.y * q.y + got.z * q.z + got.w * q.w), 1, 1e-12) << order;
    // the spec's own reading of it, and back to the angles
    const std::array<double, 3> angle{angles.at('X'), angles.at('Y'), angles.at('Z')}, frame{orient.at('X'), orient.at('Y'), orient.at('Z')};
    const fv::Quat              by = fv::rig::boneTurn(angle, frame, order);
    EXPECT_NEAR(std::abs(by.x * q.x + by.y * q.y + by.z * q.z + by.w * q.w), 1, 1e-12) << order;
    const std::array<double, 3> back = fv::rig::boneValues(q, frame, order);
    for(size_t i = 0; i < 3; i++)
      EXPECT_NEAR(back[i], angle[i], 1e-9) << order << i;
    const fv::Vec3 at = scene.nodes[size_t(node)].position, scale = scene.nodes[size_t(node)].scale;
    EXPECT_NEAR(at.x - rest.x, 3, 1e-12) << order;
    EXPECT_NEAR(at.y - rest.y, 0, 1e-12) << order;
    EXPECT_NEAR(at.z - rest.z, 0, 1e-12) << order;
    EXPECT_EQ(scale.x, 1) << order;
    EXPECT_EQ(scale.y, 2) << order;
    EXPECT_EQ(scale.z, 1) << order;
  }
}

// A name read by the spec: its side, its joint, Skyrim's by its own table.
TEST(RigSpec, ANameIsReadIntoASideAndAJoint)
{
  const fv::rig::Name g8 = fv::rig::parse("lShldrBend"), g9 = fv::rig::parse("l_upperarm"), sk = fv::rig::parse("NPC R Calf [RClf]");
  ASSERT_NE(g8.joint, nullptr);
  EXPECT_EQ(g8.joint, g9.joint);
  EXPECT_EQ(g8.side, 'l');
  EXPECT_EQ(g8.joint->id, "upperarm");
  ASSERT_NE(sk.joint, nullptr);
  EXPECT_EQ(sk.joint->id, "shin");
  EXPECT_EQ(sk.side, 'r');
  EXPECT_TRUE(sk.skyrim);
  EXPECT_EQ(fv::rig::parse("NPC Spine1 [Spn1]").slot, "spine2");
  EXPECT_EQ(fv::rig::parse("l_shoulder").joint->id, "collar");
  EXPECT_EQ(fv::rig::parse("lip").joint, nullptr);
  EXPECT_EQ(fv::rig::parse("lip").side, 0);
}

// The readers that became shims answer as the tables they replaced, on every spelling, and so do the
// tables derived from the spec (the fitter's aims, the ragdoll's parts) and the family rule.
TEST(RigSpec, TheShimsAnswerAsTheTablesTheyReplaced)
{
  const std::vector<std::string> names = spellings();
  ASSERT_GT(names.size(), 3000u);
  size_t differ = 0;
  for(const std::string& n : names)
    if(answers(n, true) != answers(n, false))
    {
      if(++differ < 20)
        ADD_FAILURE() << n << ": " << answers(n, true) << " was " << answers(n, false);
    }
  EXPECT_EQ(differ, 0u);
  // a figure of each family's spellings, and every name matched against each
  const std::vector<std::vector<std::string>> figures{
      {"hip", "pelvis", "lThighBend", "lThighTwist", "lShin", "lFoot", "lMetatarsals", "lToe", "lBigToe", "lBigToe_2", "lSmallToe1", "lSmallToe1_2",
       "rThighBend", "rThighTwist", "rShin", "rFoot", "abdomenLower", "abdomenUpper", "chestLower", "chestUpper", "neckLower", "neckUpper", "head",
       "lCollar", "lShldrBend", "lShldrTwist", "lForearmBend", "lForearmTwist", "lHand", "lCarpal1", "lIndex1", "lIndex2", "lThumb1", "lPectoral",
       "rCollar", "rShldrBend", "rShldrTwist", "rForearmBend", "rForearmTwist", "rHand", "lEye", "lowerJaw"},
      {"hip", "pelvis", "l_thigh", "l_thightwist1", "l_thightwist2", "l_shin", "l_foot", "l_metatarsal", "l_toes", "l_bigtoe1", "l_bigtoe2",
       "l_indextoe1", "spine1", "spine2", "spine3", "spine4", "neck1", "neck2", "head", "l_shoulder", "l_upperarm", "l_upperarmtwist1",
       "l_upperarmtwist2", "l_forearm", "l_forearmtwist1", "l_forearmtwist2", "l_hand", "l_indexmetacarpal", "l_index1", "l_thumb1", "l_pectoral",
       "r_shoulder", "r_upperarm", "r_forearm", "r_hand", "r_thigh", "r_shin", "l_eye"},
      {"hip", "abdomen", "abdomen2", "chest", "neck", "head", "lCollar", "lShldr", "lForeArm", "lHand", "lThigh", "lShin", "lFoot", "lToe", "rThigh",
       "rShin"},
  };
  differ = 0;
  for(const auto& figure : figures)
    for(const std::string& n : names)
      if(fv::wardrobe::matchBone(n, figure) != old::matchBone(n, figure))
      {
        if(++differ < 20)
          ADD_FAILURE() << n << " -> " << fv::wardrobe::matchBone(n, figure).value_or("none") << " was " << old::matchBone(n, figure).value_or("none");
      }
  EXPECT_EQ(differ, 0u);

  std::map<std::string, std::vector<std::string>> aims;
  for(const fv::rig::Joint& j : fv::rig::joints())
    for(std::string_view a : j.aims)
      if(!a.empty())
        aims[std::string(j.id)].push_back(std::string(a));
  EXPECT_EQ(aims, old::nextSlots());
  std::set<std::string> own, merged;
  for(const fv::rig::Joint& j : fv::rig::joints())
    if(j.part == fv::rig::Part::eOwn)
      own.insert(std::string(j.id));
    else if(j.part == fv::rig::Part::eMerged)
      merged.insert(std::string(j.id));
  EXPECT_EQ(own, old::kPartSlots);
  EXPECT_EQ(merged, old::kMergedSlots);
  for(const auto& figure : figures)
    EXPECT_EQ(fv::wardrobe::generationFromBones(figure), old::generationFromBones(figure));
}

// Every installed rig's every bone, through the old readers and the shims: the gate of step 2a. The
// names come from FV_RIG_NAMES (see the top of this file); without it there is nothing to read.
TEST(RigSpec, EveryInstalledRigsBonesAnswerAsBefore)
{
  const char* path = std::getenv("FV_RIG_NAMES");
  if(!path || !*path)
    GTEST_SKIP() << "FV_RIG_NAMES not set";
  std::ifstream in(path);
  ASSERT_TRUE(in) << path;
  struct Rig
  {
    std::string                         file;
    bool                                figure = false;
    std::vector<fv::wardrobe::BoneRef> bones;
  };
  std::vector<Rig> rigs;
  for(std::string line; std::getline(in, line);)
  {
    std::vector<std::string> cols;
    for(size_t at = 0;;)
    {
      const size_t tab = line.find('\t', at);
      cols.push_back(line.substr(at, tab == std::string::npos ? std::string::npos : tab - at));
      if(tab == std::string::npos)
        break;
      at = tab + 1;
    }
    if(cols[0] == "R" && cols.size() >= 3)
      rigs.push_back({cols[1], cols[2] == "1", {}});
    else if(cols[0] == "B" && cols.size() >= 2 && !rigs.empty())
      rigs.back().bones.push_back({cols[1], cols.size() > 2 && !cols[2].empty() ? std::optional<std::string>(cols[2]) : std::nullopt});
  }
  ASSERT_FALSE(rigs.empty());
  // the base figures items are worn on: a Genesis figure file with a whole skeleton
  std::vector<std::vector<std::string>> figures;
  for(const Rig& r : rigs)
    if(r.figure && r.bones.size() >= 100)
    {
      figures.emplace_back();
      for(const auto& b : r.bones)
        figures.back().push_back(b.name);
    }
  ASSERT_FALSE(figures.empty());
  std::set<std::string> names;
  size_t                bones = 0, plans = 0, differ = 0;
  for(const Rig& r : rigs)
    for(const auto& b : r.bones)
      names.insert(b.name), bones++;
  const auto report = [&](const std::string& what) {
    if(++differ < 30)
      ADD_FAILURE() << what;
  };
  for(const std::string& n : names)
  {
    if(answers(n, true) != answers(n, false))
      report(n + ": " + answers(n, true) + " was " + answers(n, false));
    for(const auto& figure : figures)
      if(fv::wardrobe::matchBone(n, figure) != old::matchBone(n, figure))
        report(n + " matched " + fv::wardrobe::matchBone(n, figure).value_or("none") + " was " + old::matchBone(n, figure).value_or("none"));
  }
  for(const Rig& r : rigs)
  {
    std::vector<std::string> own;
    for(const auto& b : r.bones)
      own.push_back(b.name);
    if(fv::wardrobe::generationFromBones(own) != old::generationFromBones(own))
      report(r.file + ": its generation");
    for(const auto& figure : figures)
    {
      plans++;
      if(leaders(fv::wardrobe::planLeaders(r.bones, figure)) != leaders(old::planLeaders(r.bones, figure)))
        report(r.file + ": its leaders");
    }
  }
  std::printf("[rig spec] %zu rigs, %zu bones, %zu names, %zu figures, %zu leader plans: %zu differ\n", rigs.size(), bones, names.size(), figures.size(),
              plans, differ);
  EXPECT_EQ(differ, 0u);
}
