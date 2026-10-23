#pragma once

// A clothing bone and the figure bone that should lead it, across Genesis generations. Both names
// reduce to an anatomical slot (G8 `lShldrBend` and G9 `l_upperarm` are both the upper arm; G9's
// `l_shoulder` is the collar). Exact names win, so a same-generation outfit never takes the alias.
// Left never matches right. A port of src/lib/wardrobe/fit/bones.ts.

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace fv::wardrobe {

struct BoneRef
{
  std::string                name;
  std::optional<std::string> parent;
};

struct LeaderPlan
{
  // item bone name -> figure bone name: the item bone copies that leader's rotation
  std::map<std::string, std::string> follow;
  // item bones that move because an ancestor follows, and must not also copy the leader
  std::set<std::string> inherit;
};

struct ParsedBone
{
  char        side = 0;  // 0, 'l' or 'r'
  std::string token;     // lowercase, side and separators removed: `lBigToe_2` -> `bigtoe2`
  std::string slot;      // the anatomical slot, or the token itself when the table does not know it
  bool        twist = false;
};

// `lThighBend` -> left thigh, `l_toes` -> left toes, `lip` stays a whole token with no side.
ParsedBone parseBone(const std::string& name);

// The figure bone `name` should follow, or none when nothing on the figure corresponds.
std::optional<std::string> matchBone(const std::string& name, const std::vector<std::string>& targets);

// Which item bones copy a figure bone, and which inherit that motion from their parent. A twist
// bone parented to the bend bone that already follows the same leader is inherited: copying the
// leader again would apply the rotation twice.
LeaderPlan planLeaders(const std::vector<BoneRef>& bones, const std::vector<std::string>& targets);

}  // namespace fv::wardrobe
