#include <figure/wardrobe/bind_cache.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

#include <figure/io/timing.hpp>

// xxHash as zstd ships it (XXH64 only), inlined here so it shares no symbols with zstd's copy
#define XXH_INLINE_ALL
#include <xxhash.h>

// bind.cpp's SHA-256, set by CMake (figure/CMakeLists.txt): a change to the binding code is another key
#ifndef FV_BIND_SOURCE_SHA
#define FV_BIND_SOURCE_SHA "unknown"
#endif

namespace fv::wardrobe {

namespace {

std::mutex            g_dirMutex;
std::filesystem::path g_dir;
std::atomic<uint64_t> g_hits{0}, g_misses{0};

constexpr char     kMagic[4] = {'F', 'V', 'B', '1'};
constexpr uint32_t kVersion  = 1;
// the files kept, oldest written or read first to go
constexpr uintmax_t kBudget = 1ull << 30;

struct Key
{
  uint64_t a = 0, b = 0;
};

struct Header
{
  char     magic[4];
  uint32_t version;
  uint64_t keyA, keyB;
  uint64_t verts, weights, used;
};

// Two XXH64 streams with different seeds over the same bytes: a 128-bit key.
class Hasher
{
public:
  Hasher()
  {
    XXH64_reset(&m_a, 0x9E3779B97F4A7C15ull);
    XXH64_reset(&m_b, 0xC2B2AE3D27D4EB4Full);
  }
  void bytes(const void* data, size_t size)
  {
    XXH64_update(&m_a, data, size);
    XXH64_update(&m_b, data, size);
  }
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
  Key key() { return {XXH64_digest(&m_a), XXH64_digest(&m_b)}; }

private:
  XXH64_state_t m_a, m_b;
};

Key keyOf(uint32_t kind,
          std::span<const float>    bodyPositions,
          std::span<const uint32_t> bodyTris,
          std::span<const float>    itemPositions,
          std::span<const int32_t>  pairs,
          std::span<const int32_t>  lineStart,
          std::span<const int32_t>  lineVerts,
          float                     reach,
          const BindFilter*         filter)
{
  Hasher           h;
  const std::string_view source = FV_BIND_SOURCE_SHA;
  h.bytes(kMagic, sizeof kMagic);
  h.bytes(&kVersion, sizeof kVersion);
  h.bytes(source.data(), source.size());
  h.bytes(&kind, sizeof kind);
  h.array<float>(1, bodyPositions);
  h.array<uint32_t>(2, bodyTris);
  h.array<float>(3, itemPositions);
  h.array<int32_t>(4, pairs);
  h.array<int32_t>(5, lineStart);
  h.array<int32_t>(6, lineVerts);
  h.bytes(&reach, sizeof reach);
  const uint8_t filtered = filter ? 1 : 0;
  h.bytes(&filtered, 1);
  if(filter)
  {
    h.array<int16_t>(7, filter->bodyLimb);
    h.array<int32_t>(8, filter->itemMask);
  }
  return h.key();
}

std::filesystem::path fileOf(const std::filesystem::path& dir, const Key& key)
{
  char name[40];
  std::snprintf(name, sizeof name, "%016llx%016llx.fvb", static_cast<unsigned long long>(key.a), static_cast<unsigned long long>(key.b));
  return dir / name;
}

template <class T>
bool readRaw(std::istream& in, std::vector<T>& out, uint64_t count)
{
  out.resize(size_t(count));
  return count == 0 || bool(in.read(reinterpret_cast<char*>(out.data()), std::streamsize(count * sizeof(T))));
}

template <class T>
void writeRaw(std::ostream& out, const std::vector<T>& data)
{
  if(!data.empty())
    out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(T)));
}

// A binding of `itemVerts` item and `bodyVerts` body vertices read back, if the file is whole and
// is this key's.
std::optional<SurfaceBinding> readBinding(const std::filesystem::path& file, const Key& key, size_t itemVerts, size_t bodyVerts)
{
  std::error_code ec;
  const uintmax_t size = std::filesystem::file_size(file, ec);
  if(ec || size < sizeof(Header))
    return std::nullopt;
  std::ifstream in(file, std::ios::binary);
  Header        h{};
  if(!in.read(reinterpret_cast<char*>(&h), sizeof h) || std::memcmp(h.magic, kMagic, 4) != 0 || h.version != kVersion || h.keyA != key.a
     || h.keyB != key.b || h.verts != itemVerts * 3 || h.weights != itemVerts * 3 || h.used != bodyVerts
     || size != sizeof(Header) + h.verts * sizeof(uint32_t) + h.weights * sizeof(float) + h.used)
    return std::nullopt;
  SurfaceBinding b;
  if(!readRaw(in, b.verts, h.verts) || !readRaw(in, b.weights, h.weights) || !readRaw(in, b.used, h.used))
    return std::nullopt;
  // read: the newest to keep when the folder is trimmed
  std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(), ec);
  return b;
}

// Keeps the folder under its budget, the least recently written or read first to go. Files another
// process holds stay.
void trim(const std::filesystem::path& dir)
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
    if(e.path().extension() != ".fvb")
      continue;
    std::error_code fe;
    const uintmax_t size = e.file_size(fe);
    const auto      time = e.last_write_time(fe);
    if(fe)
      continue;
    files.push_back({e.path(), size, time});
    total += size;
  }
  if(total <= kBudget)
    return;
  std::sort(files.begin(), files.end(), [](const Entry& a, const Entry& b) { return a.time < b.time; });
  for(const Entry& f : files)
  {
    if(total <= kBudget * 3 / 4)
      break;
    std::error_code re;
    if(std::filesystem::remove(f.path, re))
      total -= f.size;
  }
}

void writeBinding(const std::filesystem::path& file, const Key& key, const SurfaceBinding& b)
{
  std::error_code ec;
  std::filesystem::create_directories(file.parent_path(), ec);
  Header h{};
  std::memcpy(h.magic, kMagic, 4);
  h.version = kVersion;
  h.keyA    = key.a;
  h.keyB    = key.b;
  h.verts   = b.verts.size();
  h.weights = b.weights.size();
  h.used    = b.used.size();
  // through a temporary beside it, renamed into place: a reader never sees a half-written file, and
  // two writers (two app instances) each write their own
  static std::atomic<unsigned> counter{0};
  const std::filesystem::path  tmp = file.string() + ".tmp" + std::to_string(counter++) + "-"
                                    + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff) + "-"
                                    + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() & 0xffffff);
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if(!out)
      return;
    out.write(reinterpret_cast<const char*>(&h), sizeof h);
    writeRaw(out, b.verts);
    writeRaw(out, b.weights);
    writeRaw(out, b.used);
    if(!out)
    {
      out.close();
      std::filesystem::remove(tmp, ec);
      return;
    }
  }
  std::filesystem::rename(tmp, file, ec);
  if(ec)
  {
    // another instance put the same binding there first, or holds it open
    std::filesystem::remove(tmp, ec);
    return;
  }
  trim(file.parent_path());
}

SurfaceBinding cached(const Key& key, size_t itemVerts, size_t bodyVerts, const std::function<SurfaceBinding()>& bind)
{
  const std::filesystem::path dir = bindCacheDir();
  if(dir.empty())
    return bind();
  const std::filesystem::path file = fileOf(dir, key);
  const auto                  t0   = std::chrono::steady_clock::now();
  if(auto b = readBinding(file, key, itemVerts, bodyVerts))
  {
    g_hits++;
    timing::report("bind cache hit", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), 1);
    return std::move(*b);
  }
  g_misses++;
  SurfaceBinding b = bind();
  writeBinding(file, key, b);
  return b;
}

}  // namespace

void setBindCacheDir(const std::filesystem::path& dir)
{
  std::lock_guard lock(g_dirMutex);
  g_dir = dir;
}

std::filesystem::path bindCacheDir()
{
  std::lock_guard lock(g_dirMutex);
  return g_dir;
}

SurfaceBinding cachedBindToSurface(std::span<const float>    bodyPositions,
                                   std::span<const uint32_t> bodyTris,
                                   std::span<const float>    itemPositions,
                                   std::span<const int32_t>  pairs,
                                   const BindFilter*         filter)
{
  const Key key = keyOf(1, bodyPositions, bodyTris, itemPositions, pairs, {}, {}, 0.f, filter);
  return cached(key, itemPositions.size() / 3, bodyPositions.size() / 3,
                [&] { return bindToSurface(bodyPositions, bodyTris, itemPositions, pairs, filter); });
}

SurfaceBinding cachedBindStrands(std::span<const float>    bodyPositions,
                                 std::span<const uint32_t> bodyTris,
                                 std::span<const float>    itemPositions,
                                 std::span<const int32_t>  lineStart,
                                 std::span<const int32_t>  lineVerts,
                                 float                     reach,
                                 const BindFilter*         filter)
{
  const Key key = keyOf(2, bodyPositions, bodyTris, itemPositions, {}, lineStart, lineVerts, reach, filter);
  return cached(key, itemPositions.size() / 3, bodyPositions.size() / 3,
                [&] { return bindStrands(bodyPositions, bodyTris, itemPositions, lineStart, lineVerts, reach, filter); });
}

BindCacheCounts bindCacheCounts()
{
  return {g_hits.load(), g_misses.load()};
}

}  // namespace fv::wardrobe
