#pragma once

// What the disk caches share (the bind cache, the body map cache): a 128-bit key over exactly the
// bytes a result is made from, the file that key names, writing a file whole or not at all, and a
// folder of such files on a budget, least recently written or read first to go.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <memory>
#include <span>
#include <string_view>

namespace fv::io {

struct CacheKey
{
  uint64_t a = 0, b = 0;
  bool     operator==(const CacheKey&) const = default;
};

// Two XXH64 streams with different seeds over the same bytes: a 128-bit key.
class CacheHasher
{
public:
  CacheHasher();
  ~CacheHasher();
  CacheHasher(const CacheHasher&)            = delete;
  CacheHasher& operator=(const CacheHasher&) = delete;

  void bytes(const void* data, size_t size);
  // an array as (tag, element size, count, elements), so no two inputs can run into each other
  template <class T>
  void array(uint32_t tag, std::span<const T> values)
  {
    const uint32_t elem  = sizeof(T);
    const uint64_t count = values.size();
    bytes(&tag, sizeof tag);
    bytes(&elem, sizeof elem);
    bytes(&count, sizeof count);
    bytes(values.data(), values.size_bytes());
  }
  // a string as its length and its characters
  void     text(std::string_view s);
  CacheKey key() const;

private:
  struct State;
  std::unique_ptr<State> m_state;
};

// The file `key` names in `dir`: its 32 hex digits and `extension` (".fvb").
std::filesystem::path cacheFileOf(const std::filesystem::path& dir, const CacheKey& key, std::string_view extension);

// Writes `file` through a temporary beside it renamed into place, so a reader never sees half of it
// and two writers (two app instances) each write their own; `write` returns whether it wrote it all.
// False, with nothing left behind, when it didn't or another writer was first. Creates the folder.
bool writeCacheFile(const std::filesystem::path& file, const std::function<bool(std::ostream&)>& write);

// Over `budget`, takes the folder's files with `extension` down to three quarters of it, oldest first.
// Files another process holds stay.
void trimCacheFolder(const std::filesystem::path& dir, std::string_view extension, uintmax_t budget);

// A file was read: the newest to keep when its folder is trimmed.
void touchCacheFile(const std::filesystem::path& file);

}  // namespace fv::io
