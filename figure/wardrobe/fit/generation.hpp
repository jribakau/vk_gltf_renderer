#pragma once

// Which figure an asset was built for. Bones that only one generation has decide; else what the item
// says it was made for (its figure node's preferred base, a .nif's Skyrim), then a conform target,
// then the library path: the first of them its bones allow. A path alone can mislead: a product for
// Genesis 9 may keep its Genesis 8 pieces in its own folder.

#include <optional>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>

namespace fv::wardrobe {

enum class Generation
{
  eG2,
  eG3,
  eG8,
  eG81,
  eG9,
  // Skyrim's human skeleton (NPC Pelvis [Pelv]...): CBBE, UNP and their kin, from a .nif
  eSkyrim,
};

const char* label(Generation g);

// Genesis 3, 8 and 8.1 share a polygon topology, so a vertex index on one is the same point on the others.
bool sameTopology(Generation a, Generation b);
// Genesis 8 and 8.1 share a skeleton and a mesh: an outfit for one needs no fitting on the other.
bool sameRig(Generation a, Generation b);

std::optional<Generation> generationFromPath(const std::string& path);
// The bones' best guess: Genesis 3 or 8 bones without a heel read as Genesis 8.
std::optional<Generation> generationFromBones(const std::vector<std::string>& names);
// `stated`: what the item says it was made for (dson::Item::madeFor).
std::optional<Generation> identifyGeneration(const std::string& path, const std::vector<std::string>& bones, const std::string& conformTarget,
                                             std::optional<Generation> stated = std::nullopt);

// A URL in the wearable that names the figure it was authored on, or empty.
std::string conformTargetOf(const dson::Json& doc);

// "female" or "male", when the path says; empty otherwise.
std::string genderHint(const std::string& path);

}  // namespace fv::wardrobe
