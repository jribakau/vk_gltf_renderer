#pragma once

// The named metals a surface can be made of (Edits::metal): what each reflects head-on and its edge
// tint, OpenPBR's F82 tint, from measured optical constants (utils/metal_edge_tints.py prints the
// table and says where the constants are from). DAZ and glTF author neither, so a metal is chosen in
// the panel: its tint goes on the lobe (Material::edgeTint, nvshaders f82Fresnel), and its colour
// stands in for the surface's own where the colour is neither edited nor a map's.

#include <optional>
#include <span>
#include <string_view>

#include <figure/materials/surface.hpp>

namespace fv::mat {

struct Metal
{
  const char* key;  // Edits::metal, and the app's (src/lib/materialLooks.ts METALS)
  Rgb         f0;   // linear sRGB reflectance head-on, clamped to 1
  // reflectance at cos 1/7 (about 82 degrees) over Schlick's from f0 there, clamped to 1: below 1
  // where the metal dips under Schlick's toward grazing, by up to a quarter for chrome and iron
  Rgb edgeTint;
};

// Every named metal.
std::span<const Metal> metals();
// The metal of that key, none when there is no such metal.
const Metal* findMetal(std::string_view key);

// The metal the edits name, none for none or one this table doesn't have.
const Metal* metalOf(const Edits& e);

// The base colour a surface shows with these edits: the edited colour, else the named metal's where
// the colour is no map's factor (`mapped`), else its own.
Rgb metalColor(const Edits& e, const Rgb& own, bool mapped);

}  // namespace fv::mat
