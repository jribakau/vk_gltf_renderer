#include <figure/io/cache_folder.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// xxHash as zstd ships it (XXH64 only), inlined here so it shares no symbols with zstd's copy
#define XXH_INLINE_ALL
#include <xxhash.h>

namespace fv::io {

struct CacheHasher::State
{
  XXH64_state_t a, b;
};

CacheHasher::CacheHasher()
    : m_state(std::make_unique<State>())
{
  XXH64_reset(&m_state->a, 0x9E3779B97F4A7C15ull);
  XXH64_reset(&m_state->b, 0xC2B2AE3D27D4EB4Full);
}

CacheHasher::~CacheHasher() = default;

void CacheHasher::bytes(const void* data, size_t size)
{
  XXH64_update(&m_state->a, data, size);
  XXH64_update(&m_state->b, data, size);
}

void CacheHasher::text(std::string_view s)
{
  const uint64_t n = s.size();
  bytes(&n, sizeof n);
  bytes(s.data(), s.size());
}

CacheKey CacheHasher::key() const
{
  return {XXH64_digest(&m_state->a), XXH64_digest(&m_state->b)};
}

std::filesystem::path cacheFileOf(const std::filesystem::path& dir, const CacheKey& key, std::string_view extension)
{
  char name[40];
  std::snprintf(name, sizeof name, "%016llx%016llx", static_cast<unsigned long long>(key.a), static_cast<unsigned long long>(key.b));
  return dir / (std::string(name) + std::string(extension));
}

bool writeCacheFile(const std::filesystem::path& file, const std::function<bool(std::ostream&)>& write)
{
  std::error_code ec;
  std::filesystem::create_directories(file.parent_path(), ec);
  static std::atomic<unsigned> counter{0};
  std::filesystem::path        tmp = file;
  tmp += ".tmp" + std::to_string(counter++) + "-" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff) + "-"
         + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() & 0xffffff);
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if(!out)
      return false;
    if(!write(out) || !out)
    {
      out.close();
      std::filesystem::remove(tmp, ec);
      return false;
    }
  }
  std::filesystem::rename(tmp, file, ec);
  if(ec)
  {
    // another writer put the same file there first, or holds it open
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

void trimCacheFolder(const std::filesystem::path& dir, std::string_view extension, uintmax_t budget)
{
  std::error_code ec;
  struct Entry
  {
    std::filesystem::path           path;
    uintmax_t                       size;
    std::filesystem::file_time_type time;
  };
  std::vector<Entry> files;
  uintmax_t          total = 0;
  for(const auto& e : std::filesystem::directory_iterator(dir, ec))
  {
    if(e.path().extension() != extension)
      continue;
    std::error_code fe;
    const uintmax_t size = e.file_size(fe);
    const auto      time = e.last_write_time(fe);
    if(fe)
      continue;
    files.push_back({e.path(), size, time});
    total += size;
  }
  if(total <= budget)
    return;
  std::sort(files.begin(), files.end(), [](const Entry& a, const Entry& b) { return a.time < b.time; });
  for(const Entry& f : files)
  {
    if(total <= budget / 4 * 3)
      break;
    std::error_code re;
    if(std::filesystem::remove(f.path, re))
      total -= f.size;
  }
}

void touchCacheFile(const std::filesystem::path& file)
{
  std::error_code ec;
  std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(), ec);
}

}  // namespace fv::io
