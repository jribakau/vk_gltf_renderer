#pragma once

// DSON files (.duf, .dsf) are JSON, stored either plain or gzip-compressed; DAZ writes both.

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace fv {

// True when the bytes start with the gzip magic number.
bool isGzip(std::span<const uint8_t> bytes);

// The bytes inflated when they are gzip (or zlib), else returned as they are. Throws
// std::runtime_error on a corrupt stream.
std::string inflateIfCompressed(std::span<const uint8_t> bytes);

// A whole file, inflated if it is compressed. Throws std::runtime_error when it can't be read.
std::string readMaybeCompressed(const std::filesystem::path& path);

// The first `bytes` of a file's text, inflated if it is compressed, reading no more of it than that
// takes. A damaged stream gives what came before the damage. Throws when it can't be read.
std::string readHead(const std::filesystem::path& path, size_t bytes);

}  // namespace fv
