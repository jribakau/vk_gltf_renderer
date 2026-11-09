#include <figure/rig/spec.hpp>

#include <algorithm>
#include <cctype>
#include <regex>

namespace fv::rig {

Evidence identify(const std::vector<std::string>& boneNames)
{
  static const std::regex underscored(R"(^[lr]_(thigh|shin|foot|upperarm|forearm|hand))");
  static const std::regex heelRe(R"(^[lr]heel$)");
  static const std::regex bendRe(R"(^[lr](thigh|shldr|forearm)(bend|twist)$)");
  static const std::regex trunkRe(R"(^(abdomen|chest|neck)(lower|upper)$)");
  static const std::regex plainRe(R"(^[lr]thigh$)");
  bool                    under = false, heel = false, g3or8 = false, plain = false;
  for(const std::string& name : boneNames)
  {
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    // Skyrim's bones ("NPC Head [Head]"), which no DAZ figure's are named like: a glove or a helmet has
    // only its hand's or head's
    if(n.rfind("npc ", 0) == 0)
      return {Family::eSkyrim, false};
    under = under || std::regex_search(n, underscored);
    heel  = heel || std::regex_search(n, heelRe);
    g3or8 = g3or8 || std::regex_search(n, bendRe) || std::regex_search(n, trunkRe);
    plain = plain || std::regex_search(n, plainRe);
  }
  // underscored side prefixes (`l_thigh`) are Genesis 9, whose twist helpers would otherwise read as Genesis 8
  if(under)
    return {Family::eGenesis9, false};
  if(heel)
    return {Family::eGenesis3, false};
  if(g3or8)
    return {std::nullopt, true};
  if(plain)
    return {Family::eGenesis2, false};
  return {};
}

}  // namespace fv::rig
