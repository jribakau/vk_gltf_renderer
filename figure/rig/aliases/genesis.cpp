#include <figure/rig/aliases/aliases.hpp>

namespace fv::rig::aliases {

// Genesis 2, 3, 8, 8.1 and 9: Genesis 8's Bend/Twist limbs, a lower and upper abdomen, chest and neck,
// numbered toes; Genesis 9's underscored names, its collar called a shoulder, its numbered twist
// helpers and metacarpals.
const std::unordered_map<std::string, std::string>& genesis()
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

}  // namespace fv::rig::aliases
