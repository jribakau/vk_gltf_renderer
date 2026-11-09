#include <figure/wardrobe/fit/bones.hpp>

#include <algorithm>
#include <cctype>
#include <regex>
#include <unordered_map>

namespace fv::wardrobe {

namespace {

// Name token -> slot, for Genesis 2, 3, 8, 8.1 and 9.
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
      // Genesis 9 calls the collar its shoulder
      {"shoulder", "collar"},        {"shldr", "upperarm"},         {"shldrbend", "upperarm"},     {"shldrtwist", "upperarm"},
      {"upperarmtwist1", "upperarm"}, {"upperarmtwist2", "upperarm"}, {"forearmbend", "forearm"},   {"forearmtwist", "forearm"},
      {"forearmtwist1", "forearm"},  {"forearmtwist2", "forearm"},  {"handanchor", "hand"},        {"carpal1", "indexmc"},
      {"carpal2", "midmc"},          {"carpal3", "ringmc"},         {"carpal4", "pinkymc"},        {"indexmetacarpal", "indexmc"},
      {"midmetacarpal", "midmc"},    {"ringmetacarpal", "ringmc"},  {"pinkymetacarpal", "pinkymc"},
  };
  return table;
}

// The slot to try next when the figure has nothing in this one.
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

// Skyrim's skeleton, token (its name without "NPC ", the side and the [tag]) -> slot. Its own table:
// Skyrim's Spine1 is Genesis 9's spine2. Three spine bones to Genesis' four, the chest the third;
// Finger0x is the thumb, 1x to 4x index to pinky; 3BA's Breast is the pectoral.
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

}  // namespace

ParsedBone parseBone(const std::string& name)
{
  ParsedBone out;
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
    for(char c : lower(rest))
      if(c != '_' && c != '-' && !std::isspace(static_cast<unsigned char>(c)))
        out.token += c;
    const auto it = skyrimSlots().find(out.token);
    out.slot      = it != skyrimSlots().end() ? it->second : out.token;
    out.twist     = std::regex_search(out.token, twist);
    return out;
  }
  std::string rest = name;
  const auto  prefixed = [&](const char* word, size_t at) {
    // `left` / `right` in any case, followed by an upper-case letter or underscore
    const size_t n = std::char_traits<char>::length(word);
    // (the regex is case-insensitive, so [A-Z] accepts lower-case letters here)
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
  const ParsedBone                                  parsed = parseBone(name);
  std::vector<std::pair<std::string, ParsedBone>> parsedTargets;
  parsedTargets.reserve(targets.size());
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

LeaderPlan planLeaders(const std::vector<BoneRef>& bones, const std::vector<std::string>& targets)
{
  std::map<std::string, const BoneRef*> byName;
  for(const BoneRef& b : bones)
    byName[b.name] = &b;
  // a twist bone on a figure whose twist is a bone of its own to pose (Genesis 3 and 8's lForearmTwist,
  // not Genesis 9's numbered helpers, which its formulas turn): it follows that bone, not the bend
  // matchBone prefers, or a Genesis 9 or Skyrim sleeve wouldn't twist on Genesis 8
  std::vector<std::pair<std::string, ParsedBone>> posableTwists;
  for(const std::string& t : targets)
    if(const ParsedBone p = parseBone(t); p.twist && !p.token.empty() && !std::isdigit(static_cast<unsigned char>(p.token.back())))
      posableTwists.emplace_back(t, p);
  std::map<std::string, std::string> direct;
  for(const BoneRef& b : bones)
    if(auto leader = matchBone(b.name, targets))
    {
      direct[b.name] = *leader;
      if(const ParsedBone p = parseBone(b.name); p.twist)
        for(const auto& [t, tp] : posableTwists)
          if(tp.side == p.side && tp.slot == p.slot)
          {
            direct[b.name] = t;
            break;
          }
    }
  LeaderPlan plan;
  const auto ancestorLeader = [&](const std::string& name) -> std::optional<std::string> {
    auto                  it = byName.find(name);
    std::optional<std::string> parent = it != byName.end() ? it->second->parent : std::nullopt;
    std::set<std::string> seen;
    while(parent && seen.insert(*parent).second)
    {
      if(auto d = direct.find(*parent); d != direct.end())
        return d->second;
      auto p = byName.find(*parent);
      parent = p != byName.end() ? p->second->parent : std::nullopt;
    }
    return std::nullopt;
  };
  for(const BoneRef& b : bones)
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

}  // namespace fv::wardrobe
