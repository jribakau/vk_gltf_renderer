#include <figure/rig/aliases/aliases.hpp>

namespace fv::rig::aliases {

// Skyrim's skeleton: Skyrim's Spine1 is Genesis 9's spine2. Three spine bones to Genesis' four, the
// chest the third; Finger0x is the thumb, 1x to 4x index to pinky; a Breast (CBBE's "NPC L Breast",
// 3BA's "L Breast01" to "03") is the pectoral.
const std::unordered_map<std::string, std::string>& skyrim()
{
  static const std::unordered_map<std::string, std::string> table{
      {"com", "hip"},           {"spine", "spine1"},     {"spine1", "spine2"},          {"spine2", "spine3"},
      {"neck", "neck1"},        {"clavicle", "collar"},  {"upperarmtwist1", "upperarm"}, {"upperarmtwist2", "upperarm"},
      {"forearmtwist1", "forearm"}, {"forearmtwist2", "forearm"}, {"calf", "shin"},     {"toe0", "toes"},
      {"breast", "pectoral"},   {"breast01", "pectoral"}, {"breast02", "pectoral"},   {"breast03", "pectoral"},
      {"finger00", "thumb1"},  {"finger01", "thumb2"},        {"finger02", "thumb3"},
      {"finger10", "index1"},   {"finger11", "index2"},  {"finger12", "index3"},        {"finger20", "mid1"},
      {"finger21", "mid2"},     {"finger22", "mid3"},    {"finger30", "ring1"},         {"finger31", "ring2"},
      {"finger32", "ring3"},    {"finger40", "pinky1"},  {"finger41", "pinky2"},        {"finger42", "pinky3"},
  };
  return table;
}

}  // namespace fv::rig::aliases
