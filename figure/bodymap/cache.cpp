#include <figure/bodymap/cache.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <figure/io/cache_folder.hpp>
#include <figure/io/timing.hpp>
#include <figure/wardrobe/fit/bases.hpp>

// The map code's SHA-256, set by CMake (figure/CMakeLists.txt): a change to it is another key
#ifndef FV_BODYMAP_SOURCE_SHA
#define FV_BODYMAP_SOURCE_SHA "unknown"
#endif

namespace fv::bodymap {

namespace {

std::mutex            g_dirMutex;
std::filesystem::path g_dir;
std::atomic<uint64_t> g_hits{0}, g_misses{0};

constexpr char     kMagic[4] = {'F', 'V', 'M', '1'};
constexpr uint32_t kVersion  = 1;
// the files kept, oldest written or read first to go: a Genesis map is about 4 MB
constexpr uintmax_t kBudget = 256ull << 20;

struct Header
{
  char     magic[4];
  uint32_t version;
  uint64_t keyA, keyB, bytes;
};

void hashCage(io::CacheHasher& h, uint32_t tag, const wardrobe::BaseCage& c)
{
  h.array<float>(tag, c.positions);
  h.array<uint32_t>(tag + 1, c.tris);
  h.array<uint16_t>(tag + 2, c.triSurface);
  h.array<int32_t>(tag + 3, c.skin.start);
  h.array<int32_t>(tag + 4, c.skin.bone);
  h.array<float>(tag + 5, c.skin.weight);
  const uint64_t surfaces = c.surfaces.size(), bones = c.skeleton.size();
  h.bytes(&surfaces, sizeof surfaces);
  for(const std::string& s : c.surfaces)
    h.text(s);
  h.bytes(&bones, sizeof bones);
  for(const wardrobe::SkelBone& b : c.skeleton)
  {
    h.text(b.name);
    const uint8_t parented = b.parent ? 1 : 0;
    h.bytes(&parented, 1);
    h.text(b.parent ? *b.parent : std::string());
    h.bytes(b.center.data(), sizeof b.center);
  }
}

io::CacheKey keyOf(const CloneInputs& in)
{
  io::CacheHasher        h;
  const std::string_view source = FV_BODYMAP_SOURCE_SHA;
  h.bytes(kMagic, sizeof kMagic);
  h.bytes(&kVersion, sizeof kVersion);
  h.text(source);
  hashCage(h, 10, *in.base);
  hashCage(h, 20, *in.g9);
  h.array<float>(30, in.reference.positions);
  h.array<uint32_t>(31, in.reference.tris);
  h.array<uint8_t>(32, in.reference.outer);
  h.array<RegionMask>(33, in.reference.regions);
  return h.key();
}

// Every array of a map, in the file's order (`M` const to write, not to read).
template <class M, class F>
void eachArray(M& m, F&& f)
{
  for(auto* b : {&m.source, &m.reference})
  {
    f(b->positions);
    f(b->tris);
    f(b->outer);
    f(b->regions);
  }
  for(auto* p : {&m.map.toReference, &m.map.fromReference})
  {
    f(p->at.verts);
    f(p->at.weights);
    f(p->at.used);
    f(p->mapped);
  }
  f(m.map.frameTurn);
}

template <class T>
void put(std::ostream& out, const T& value)
{
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char*>(&value), sizeof value);
}

template <class T>
void put(std::ostream& out, const std::vector<T>& values)
{
  static_assert(std::is_trivially_copyable_v<T>);
  put(out, uint64_t(values.size()));
  if(!values.empty())
    out.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
}

void putText(std::ostream& out, const std::string& s)
{
  put(out, uint64_t(s.size()));
  out.write(s.data(), std::streamsize(s.size()));
}

// Reads a file's bytes in order, each read held to what is left.
struct Cursor
{
  const char* at;
  const char* end;
  bool        ok = true;

  template <class T>
  void get(T& value)
  {
    static_assert(std::is_trivially_copyable_v<T>);
    if(!ok || size_t(end - at) < sizeof value)
    {
      ok = false;
      return;
    }
    std::memcpy(&value, at, sizeof value);
    at += sizeof value;
  }
  template <class T>
  void get(std::vector<T>& values)
  {
    uint64_t n = 0;
    get(n);
    if(!ok || n > uint64_t(end - at) / sizeof(T))
    {
      ok = false;
      return;
    }
    values.resize(size_t(n));
    if(n)
      std::memcpy(values.data(), at, size_t(n) * sizeof(T));
    at += n * sizeof(T);
  }
  void getText(std::string& s)
  {
    uint64_t n = 0;
    get(n);
    if(!ok || n > uint64_t(end - at))
    {
      ok = false;
      return;
    }
    s.assign(at, size_t(n));
    at += n;
  }
};

void writeMap(const std::filesystem::path& file, const io::CacheKey& key, const CloneMap& m)
{
  std::ostringstream body(std::ios::binary);
  eachArray(m, [&](const auto& values) { put(body, values); });
  put(body, uint64_t(m.turns.size()));
  for(const LimbTurn& t : m.turns)
  {
    putText(body, t.joint);
    put(body, t.side);
    put(body, t.degrees);
  }
  const std::string bytes = body.str();
  Header            h{};
  std::memcpy(h.magic, kMagic, 4);
  h.version = kVersion;
  h.keyA    = key.a;
  h.keyB    = key.b;
  h.bytes   = bytes.size();
  if(io::writeCacheFile(file, [&](std::ostream& out) {
       put(out, h);
       out.write(bytes.data(), std::streamsize(bytes.size()));
       return bool(out);
     }))
    io::trimCacheFolder(file.parent_path(), ".fvm", kBudget);
}

// Whether every corner a projection names is a vertex of the body it lands on.
bool cornersWithin(const Projection& p, size_t from, size_t onto)
{
  if(p.at.verts.size() != from * 3 || p.at.weights.size() != from * 3 || p.mapped.size() != from || p.at.used.size() != onto)
    return false;
  for(const uint32_t c : p.at.verts)
    if(c >= onto)
      return false;
  return true;
}

// The map read back, if the file is whole, is this key's and fits these inputs.
std::optional<CloneMap> readMap(const std::filesystem::path& file, const io::CacheKey& key, const CloneInputs& in)
{
  std::error_code ec;
  const uintmax_t size = std::filesystem::file_size(file, ec);
  if(ec || size < sizeof(Header))
    return std::nullopt;
  std::vector<char> bytes(static_cast<size_t>(size));
  {
    std::ifstream f(file, std::ios::binary);
    if(!f.read(bytes.data(), std::streamsize(bytes.size())))
      return std::nullopt;
  }
  Header h{};
  std::memcpy(&h, bytes.data(), sizeof h);
  if(std::memcmp(h.magic, kMagic, 4) != 0 || h.version != kVersion || h.keyA != key.a || h.keyB != key.b || h.bytes != size - sizeof h)
    return std::nullopt;
  Cursor   c{bytes.data() + sizeof h, bytes.data() + bytes.size()};
  CloneMap m;
  eachArray(m, [&](auto& values) { c.get(values); });
  uint64_t turns = 0;
  c.get(turns);
  for(uint64_t i = 0; c.ok && i < turns; i++)
  {
    LimbTurn t;
    c.getText(t.joint);
    c.get(t.side);
    c.get(t.degrees);
    m.turns.push_back(std::move(t));
  }
  const size_t ns = in.base->positions.size() / 3, nr = in.g9->positions.size() / 3;
  if(!c.ok || c.at != c.end || m.source.positions.size() != ns * 3 || m.reference.positions.size() != nr * 3
     || !cornersWithin(m.map.toReference, ns, nr) || !cornersWithin(m.map.fromReference, nr, ns) || m.map.frameTurn.size() != ns)
    return std::nullopt;
  io::touchCacheFile(file);
  return m;
}

}  // namespace

void setMapCacheDir(const std::filesystem::path& dir)
{
  std::lock_guard lock(g_dirMutex);
  g_dir = dir;
}

std::filesystem::path mapCacheDir()
{
  std::lock_guard lock(g_dirMutex);
  return g_dir;
}

CloneMap cachedCloneMap(const CloneInputs& in)
{
  const std::filesystem::path dir = mapCacheDir();
  if(dir.empty())
    return makeCloneMap(in);
  const io::CacheKey          key  = keyOf(in);
  const std::filesystem::path file = io::cacheFileOf(dir, key, ".fvm");
  const auto                  t0   = std::chrono::steady_clock::now();
  if(auto m = readMap(file, key, in))
  {
    g_hits++;
    timing::report("map cache hit", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), 1);
    return std::move(*m);
  }
  g_misses++;
  CloneMap m = makeCloneMap(in);
  // the cache is optional: failing to keep a map mustn't fail what wanted it
  try
  {
    writeMap(file, key, m);
  }
  catch(const std::exception&)
  {
  }
  return m;
}

MapCacheCounts mapCacheCounts()
{
  return {g_hits.load(), g_misses.load()};
}

}  // namespace fv::bodymap
