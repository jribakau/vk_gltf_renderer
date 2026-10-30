#pragma once

// Scene modifiers a .duf puts on a node: the shared reading of their channels, and the ones the
// figure acts on besides dForce (dson/dforce.hpp): DAZ Studio's smoothing modifier, whose collision
// settings the wardrobe's collider (wardrobe/collider.hpp) takes.

#include <map>
#include <optional>
#include <string>

#include <figure/dson/store.hpp>

namespace fv::dson {

// channel id -> current value (number, flag or text)
using ChannelValues = std::map<std::string, Json>;

// The node a scene modifier hangs from, as its id ("#Dress" -> "Dress", URI-decoded).
std::string modifierParent(const Json& modifier);
// Whether a modifier's `extra` lists this type.
bool hasExtraType(const Json& modifier, const char* type);
// A modifier's channels, from its studio_modifier_channels and its own channel.
ChannelValues modifierChannels(const Json& modifier);
// The definition's channels (found in the scene's or the geometry's modifier_library by the
// instance's url), then the instance's over any it repeats.
ChannelValues mergedModifierChannels(const Json& instance, const Doc& scene, const Doc* geo);

double numberOf(const ChannelValues& ch, const char* id, double fallback);
bool   flagOf(const ChannelValues& ch, const char* id, bool fallback);

// DAZ Studio's smoothing modifier on a worn item: a few iterations of mesh smoothing and of pushing
// the mesh out of the figure it collides with.
struct Smoothing
{
  bool enabled   = true;
  int  smoothing = 2;  // "Smoothing Iterations"
  int  collision = 3;  // "Collision Iterations"
  // no file's: how far the user pulled the item onto the body, 0 to 1 (WornItem::tight)
  float tight = 0;
};

// The smoothing modifier `scene` puts on `nodeId`, or none.
std::optional<Smoothing> parseSmoothing(const Doc& scene, const Doc* geo, const std::string& nodeId);

}  // namespace fv::dson
