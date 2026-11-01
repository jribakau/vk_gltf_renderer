#pragma once

// A folder of cache files on a budget: least recently written or read first to go.

#include <cstdint>
#include <filesystem>
#include <string_view>

namespace fv::io {

// Over `budget`, takes the folder's files with `extension` down to three quarters of it, oldest first.
// Files another process holds stay.
void trimCacheFolder(const std::filesystem::path& dir, std::string_view extension, uintmax_t budget);

// A file was read: the newest to keep when its folder is trimmed.
void touchCacheFile(const std::filesystem::path& file);

}  // namespace fv::io
