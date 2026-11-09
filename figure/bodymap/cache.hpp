#pragma once

// Body maps kept on disk (docs/wardrobe-plan.md, step 3: "Maps are cached under their key"). A map is a
// pure function of what it is made from (a clone map: the type's base, Genesis 9's base, and Genesis 9's
// shaped by the clone; a fitted map: the source body, Genesis 9's base, the reference and the fit's
// options) and of the map code, so it is stored under a hash of exactly those and of that code's source
// (figure/CMakeLists.txt), and made again only when one of them changes. A file that
// doesn't read back whole is a miss. The folder and its budget are the maps' own, apart from the bind
// cache's, and like it they live in the user's cache only (the plan's rule 7).

#include <cstdint>
#include <filesystem>

#include <figure/bodymap/clone.hpp>
#include <figure/bodymap/fit.hpp>

namespace fv::bodymap {

// Where the files go (created on first write); empty, the default, turns the cache off.
void                  setMapCacheDir(const std::filesystem::path& dir);
std::filesystem::path mapCacheDir();

// makeCloneMap and makeFittedMap through the cache.
TypeMap cachedCloneMap(const CloneInputs& inputs);
TypeMap cachedFittedMap(const FitInputs& inputs, const FitOptions& options = {});

// Reads and writes since start-up, for tests and the timing log.
struct MapCacheCounts
{
  uint64_t hits = 0, misses = 0;
};
MapCacheCounts mapCacheCounts();

}  // namespace fv::bodymap
