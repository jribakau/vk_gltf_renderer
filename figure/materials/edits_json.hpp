#pragma once

// The Material panel's edits as the protocol carries them: material.set's flat values onto Edits,
// and back into a surface's `values` and `edited` (src/fv_materials.hpp lists the keys). What a
// surface publishes for a key is what material.set takes back.

#include <optional>
#include <string>
#include <string_view>

#include <tinygltf/json.hpp>

#include <figure/materials/surface.hpp>

namespace fv::mat {

using EditsJson = nlohmann::json;  // fv::Json

// "#rrggbb" in sRGB, as the colour pickers hold it, of a linear colour; and back (none when it is not one).
std::string        hexOf(const Rgb& linear);
std::optional<Rgb> parseHex(const EditsJson& v);

// One value of material.set onto the edits; null drops the edit. Throws std::runtime_error for an
// unknown key or a value it can't take. A map is a library-relative image: backslashes become
// slashes and it gets its leading slash; a disk path (a drive, a share, "..") is refused.
void applyValue(Edits& e, const std::string& key, const EditsJson& v);

// The keys of the values that are edits.
EditsJson editedKeys(const Edits& e);

// Keys only a figure's surfaces take (a model's materials have no Surface to lower): the maps, the
// UV transform and the fabric.
bool figureOnlyKey(std::string_view key);
// The map slots' keys (baseMap, ...).
bool mapKey(std::string_view key);

// The Clothes tab's values of a figure surface: every map, UV, fabric and sheen key, null or its
// default when unset (a merge patch keeps what it is not sent).
void clothesValues(const Edits& e, EditsJson& values);

}  // namespace fv::mat
