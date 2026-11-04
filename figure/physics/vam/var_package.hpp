#pragma once

// A VaM package (.var): a zip of what a scene needs, its scenes as .json under Saves/scene/. Only
// what playing a scene's clips takes is read: the names in it, and one file whole.

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace fv::physics::vam {

// The files in a zip, by their names in it, zip64's sizes read where an entry has them (VaM's
// packages do). Throws when it isn't a zip, or is damaged.
std::vector<std::string> zipEntries(const std::filesystem::path& zip);

// One file of a zip, inflated, its checksum checked. Throws when it isn't there, is encrypted or
// compressed some other way than deflate, is damaged, or is longer than `maxBytes`.
std::string zipRead(const std::filesystem::path& zip, const std::string& name, size_t maxBytes = size_t(512) << 20);

// The scenes in a package (Saves/scene/*.json), in the order they are in it.
std::vector<std::string> varScenes(const std::filesystem::path& var);

}  // namespace fv::physics::vam
