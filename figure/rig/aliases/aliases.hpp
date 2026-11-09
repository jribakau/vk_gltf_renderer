#pragma once

// One alias table per rig family: a bone name's token (lowercase, the side and separators taken off)
// to the spec joint it is (rig/spec.hpp). A token a table doesn't list is taken as a joint id itself
// when the spec has one (Genesis 9's "upperarm", Skyrim's "Thigh"), else as no joint.

#include <string>
#include <unordered_map>

namespace fv::rig::aliases {

// Genesis 2, 3, 8, 8.1 and 9 (their names don't collide).
const std::unordered_map<std::string, std::string>& genesis();
// Skyrim's human skeleton, its name without "NPC ", the side and the [tag].
const std::unordered_map<std::string, std::string>& skyrim();

}  // namespace fv::rig::aliases
