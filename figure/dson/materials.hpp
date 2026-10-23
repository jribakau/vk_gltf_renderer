#pragma once

// DSON material definitions: collecting a surface's channels across the files that define and
// override it. An Iray Uber material's parameters live in
// `extra[{ type: "studio/material/uber_iray", channels: [...] }]`, PBRSkin's in a daz_brick extra,
// older shaders in the top-level DSON channels (diffuse, bump, transparency...). A port of
// src/lib/loaders/dson/materials.ts; channels are looked up by id and label, case- and
// space-insensitively (norm), because both spellings appear in the wild.

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dson {

struct Channel
{
  Json                       value;  // null when unset
  std::optional<std::string> image;  // library path
  // every name it was stored under, so an override finds them
  std::vector<std::string> keys;
};

// Channels by normalised name, in the order first stored (as a Map iterates).
class Channels
{
public:
  const Channel* get(const std::string& normName) const;
  // the first of `names` (not normalised) present
  const Channel* find(std::initializer_list<const char*> names) const;
  void           set(const std::string& key, std::shared_ptr<Channel> c);
  std::shared_ptr<Channel> shared(const std::string& key) const;
  const std::vector<std::string>& order() const { return m_order; }

private:
  std::vector<std::string>                                  m_order;
  std::unordered_map<std::string, std::shared_ptr<Channel>> m_map;
};

// Lower case, letters and digits only.
std::string norm(std::string_view s);

// Collects channels from a DSON material definition into `into`. Later calls override.
void collectChannels(const Json* def, Channels& into, const std::unordered_map<std::string, std::string>& images, const std::string& file);

// image_library entries map an id to image layers; the first active layer with a file.
void collectImages(const Json& doc, std::unordered_map<std::string, std::string>& into);

// Identifies the material a surface name and its channels make, so equal ones share.
std::string materialKey(const std::string& name, const Channels& ch);

// The surface's look from its channels: base colour, its map, cutout opacity.
Material buildMaterial(const std::string& name, const Channels& ch);

// three's SRGBToLinear.
float srgbToLinear(double c);

}  // namespace fv::dson
