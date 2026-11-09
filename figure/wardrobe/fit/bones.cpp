#include <figure/wardrobe/fit/bones.hpp>

#include <algorithm>
#include <cctype>

#include <figure/rig/spec.hpp>

namespace fv::wardrobe {

ParsedBone parseBone(const std::string& name)
{
  rig::Name  n = rig::parse(name);
  ParsedBone out;
  out.side  = n.side;
  out.token = std::move(n.token);
  out.slot  = std::move(n.slot);
  out.twist = n.twist;
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
    const rig::Joint* j = rig::joint(slot);
    if(!j || j->fallback.empty())
      break;
    slot = std::string(j->fallback);
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
