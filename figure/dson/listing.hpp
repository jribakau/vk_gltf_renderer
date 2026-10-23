#pragma once

// Folder listings for a figure load: the figure's Morphs folder and each worn item's own. A folder
// of a Genesis figure holds thousands of files and the walk is on disk, and a session swaps
// characters on the same figure over and over. A listing is kept for a few minutes, so content
// installed by another program shows up soon, and dropped when the app installs or removes content
// (clearListings). A port of src/lib/loaders/dson/listing.ts.

#include <string>
#include <vector>

#include <figure/dson/store.hpp>

namespace fv::dson {

// The library-relative paths (with '/') of the files under `rel` in every content folder whose
// extension is `ext` (e.g. ".dsf"), case-insensitively; each path once. Empty when no folder has it.
std::vector<std::string> listFiles(const Library& library, const std::string& rel, const std::string& ext);

void clearListings();

}  // namespace fv::dson
