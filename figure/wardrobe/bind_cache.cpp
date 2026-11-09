#include <figure/wardrobe/bind_cache.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include <figure/io/cache_folder.hpp>
#include <figure/io/timing.hpp>

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

using Key = io::CacheKey;

struct Header
{
  char     magic[4];
  uint32_t version;
  uint64_t keyA, keyB;
  uint64_t verts, weights, used;
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
  io::CacheHasher        h;
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
  io::touchCacheFile(file);
  return b;
}

void writeBinding(const std::filesystem::path& file, const Key& key, const SurfaceBinding& b)
{
  Header h{};
  std::memcpy(h.magic, kMagic, 4);
  h.version = kVersion;
  h.keyA    = key.a;
  h.keyB    = key.b;
  h.verts   = b.verts.size();
  h.weights = b.weights.size();
  h.used    = b.used.size();
  const bool written = io::writeCacheFile(file, [&](std::ostream& out) {
    out.write(reinterpret_cast<const char*>(&h), sizeof h);
    writeRaw(out, b.verts);
    writeRaw(out, b.weights);
    writeRaw(out, b.used);
    return bool(out);
  });
  if(written)
    io::trimCacheFolder(file.parent_path(), ".fvb", kBudget);
}

SurfaceBinding cached(const Key& key, size_t itemVerts, size_t bodyVerts, const std::function<SurfaceBinding()>& bind)
{
  const std::filesystem::path dir = bindCacheDir();
  if(dir.empty())
    return bind();
  const std::filesystem::path file = io::cacheFileOf(dir, key, ".fvb");
  const auto                  t0   = std::chrono::steady_clock::now();
  if(auto b = readBinding(file, key, itemVerts, bodyVerts))
  {
    g_hits++;
    timing::report("bind cache hit", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), 1);
    return std::move(*b);
  }
  g_misses++;
  SurfaceBinding b = bind();
  // the cache is optional: failing to keep what was bound mustn't fail the wear
  try
  {
    writeBinding(file, key, b);
  }
  catch(const std::exception&)
  {
  }
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
