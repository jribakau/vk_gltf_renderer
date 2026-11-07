#pragma once

// DAZ Studio's script material presets (.dsa, "// DAZ Studio version ... filetype Materials
// Preset"), read as data: DAZ's preset saver wrote every one from the same template, a switch on the
// surface's name whose cases set each property through DzPresetHelper with literal values
//
//   case "gauges":
//     g_oPresetHelper.setColorProperty( "Emission Color", [ 205, 247, 255 ], "/Runtime/Textures/.../x.png" );
//     g_oPresetHelper.setNumericPropertyWithAttributes( "Glossy Roughness", true, 0, 1, [ 0.3 ] );
//     g_oPresetHelper.setNumericProperty( "Luminance Units", [ "lm" ] );
//     break;
//
// and the file becomes the .duf material preset it stands for: an animation per property,
// `name://@selection#materials/<surface>:?<property>/value`, and `/image_file` for its map. No code
// of the script runs, and a call whose arguments aren't all literals is left out. As DAZ Studio with
// its default map setting (Replace) does: a colour is 0-255, an option written as its text is the
// index of that text among the option's values, and a property the script sets without a map loses
// the one it had. The script's properties are its shader's (Iray Uber, in every one in the library),
// named as the .duf files' channel ids are, with "Diffuse Color" for "diffuse"; a surface of another
// shader takes them over its own channels. Options whose values are a scene's (a UV set) are left out.

#include <optional>
#include <string_view>

#include <figure/dson/store.hpp>

namespace fv::dson {

// Whether `text` (its start is enough) is a script material preset.
bool isScriptMaterialsPreset(std::string_view text);

// The .duf material preset `text` stands for, {"asset_info": {"type": "preset_material"}, "scene":
// {"animations": [...]}}; none when it isn't a script material preset or sets nothing.
std::optional<Json> scriptMaterialsPreset(std::string_view text);

}  // namespace fv::dson
