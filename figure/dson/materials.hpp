#pragma once

// DSON material definitions: collecting a surface's channels across the files that define and
// override it. A material's extras name its shader and hold its parameters, as two entries:
// `extra[{ type: "studio/material/uber_iray", version }, { type: "studio_material_channels",
// channels: [...] }]`. PBRSkin's shader is a daz_brick, a 3Delight one a daz_shader with its .dse
// definition; older shaders also keep top-level DSON channels (diffuse, bump, transparency...). A port of
// src/lib/loaders/dson/materials.ts; channels are looked up by id and label, case- and
// space-insensitively (norm), because both spellings appear in the wild.

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/dson/channels.hpp>
#include <figure/materials/surface.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dson {

// Collects channels from a DSON material definition into `into`. Later calls override. It also keeps
// the shader the layers name (Channels::shader); a brick's url is resolved against `file`.
void collectChannels(const Json* def, Channels& into, const std::unordered_map<std::string, std::string>& images, const std::string& file);

// image_library entries map an id to image layers; the first active layer with a file.
void collectImages(const Json& doc, std::unordered_map<std::string, std::string>& into);

// Identifies the material a surface name and its channels make, so equal ones share.
std::string materialKey(const std::string& name, const Channels& ch);

// The surface a material's channels describe: an Iray Uber, PBRSkin or legacy material (a port of
// materials/parse/dson.ts parseDsonSurface).
mat::Surface parseSurface(const std::string& name, const Channels& ch);

// OmniHair's colour, linear, or none for any other shader (omniHairColor in parse/dson.ts).
std::optional<mat::Rgb> omniHairColor(const Channels& ch);

// A channel a preset sets: its value, and/or its image (an empty image removes the map), and/or
// fields of its map's texture instance settings (Channel::imageModification; null: none).
struct ChannelOverride
{
  std::string                channel;  // id or label
  std::optional<Json>        value;
  std::optional<std::string> image;
  Json                       imageModification;
};

// Lays channel overrides over a material's own and parses its surface again. False when nothing
// changed. Material edits (m.edits) and the baseline stay. Texture instance settings alone, for a
// channel the material lacks, are not laid: they would make a channel with no value.
bool overrideChannels(Material& m, const std::vector<ChannelOverride>& overrides);

// The material a surface name and its channels make: the surface, and the channels kept for what
// reads them directly (strand widths), which are also its baseline.
Material buildMaterial(const std::string& name, const Channels& ch);

}  // namespace fv::dson
