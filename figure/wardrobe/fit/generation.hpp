#pragma once

// Which Genesis figure an asset was built for. The library path is the reliable signal; bone names
// cover a file whose path never says, and a conform-target URL covers a wearable that points at the
// figure it was fitted to. A port of src/lib/wardrobe/fit/generation.ts.

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
};

const char* label(Generation g);

// Genesis 3, 8 and 8.1 share a polygon topology, so a vertex index on one is the same point on the others.
bool sameTopology(Generation a, Generation b);
// Genesis 8 and 8.1 share a skeleton and a mesh: an outfit for one needs no fitting on the other.
bool sameRig(Generation a, Generation b);

std::optional<Generation> generationFromPath(const std::string& path);
std::optional<Generation> generationFromBones(const std::vector<std::string>& names);
// Path, then the conform-target URL, then the skeleton.
std::optional<Generation> identifyGeneration(const std::string& path, const std::vector<std::string>& bones, const std::string& conformTarget);

// A URL in the wearable that names the figure it was authored on, or empty.
std::string conformTargetOf(const dson::Json& doc);

// "female" or "male", when the path says; empty otherwise.
std::string genderHint(const std::string& path);

}  // namespace fv::wardrobe
