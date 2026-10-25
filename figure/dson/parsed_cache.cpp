// The on-disk cache of parsed DSON files: the tree as CBOR (ordered_json keeps its key order and
// number types through it) and the packed tables raw, stamped with the source file's path hash,
// size and modification time. A 76 MB strand-hair geometry parses in 0.8 s and reads back in tens
// of milliseconds.

#include <figure/dson/store.hpp>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>

namespace fv::dson {

namespace {

std::mutex            g_dirMutex;
std::filesystem::path g_dir;

constexpr char     kMagic[4] = {'F', 'V', 'D', '1'};
constexpr uint32_t kVersion  = 1;

struct Header
{
  char     magic[4];
  uint32_t version;
  uint64_t pathHash, size;
  int64_t  mtime;
  uint64_t bytes;  // Doc::bytes
  uint64_t cborBytes;
  uint32_t tables, reserved;
};

template <class T>
bool readRaw(std::istream& in, T* out, size_t count)
{
  return count == 0 || bool(in.read(reinterpret_cast<char*>(out), std::streamsize(count * sizeof(T))));
}

template <class T>
void writeRaw(std::ostream& out, const T* data, size_t count)
{
  if(count)
    out.write(reinterpret_cast<const char*>(data), std::streamsize(count * sizeof(T)));
}

uint64_t fnv1a64(std::string_view s)
{
  uint64_t h = 14695981039346656037ull;
  for(const unsigned char c : s)
  {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

}  // namespace

void setParsedCacheDir(const std::filesystem::path& dir)
{
  std::lock_guard lock(g_dirMutex);
  g_dir = dir;
}

std::filesystem::path parsedCacheDir()
{
  std::lock_guard lock(g_dirMutex);
  return g_dir;
}

ParsedStamp stampOf(const std::filesystem::path& file)
{
  ParsedStamp     s;
  std::error_code ec;
  const auto      abs = std::filesystem::absolute(file, ec);
  const auto      u8  = (ec ? file : abs).generic_u8string();
  s.pathHash          = fnv1a64(std::string_view(reinterpret_cast<const char*>(u8.data()), u8.size()));
  const uintmax_t size = std::filesystem::file_size(file, ec);
  s.size               = ec ? 0 : size;
  const auto t         = std::filesystem::last_write_time(file, ec);
  s.mtime              = ec ? 0 : int64_t(t.time_since_epoch().count());
  return s;
}

std::filesystem::path parsedCacheFile(const std::filesystem::path& dir, const ParsedStamp& stamp)
{
  char name[32];
  std::snprintf(name, sizeof name, "%016llx.fvd", static_cast<unsigned long long>(stamp.pathHash));
  return dir / name;
}

bool writeParsed(const std::filesystem::path& cacheFile, const Doc& doc, const ParsedStamp& stamp)
{
  std::error_code ec;
  std::filesystem::create_directories(cacheFile.parent_path(), ec);
  const std::vector<uint8_t> cbor = Json::to_cbor(doc.json);
  Header                     h{};
  std::memcpy(h.magic, kMagic, 4);
  h.version   = kVersion;
  h.pathHash  = stamp.pathHash;
  h.size      = stamp.size;
  h.mtime     = stamp.mtime;
  h.bytes     = doc.bytes;
  h.cborBytes = cbor.size();
  h.tables    = uint32_t(doc.tables.size());
  // through a temporary beside it, renamed into place: a reader never sees a half-written file
  static std::atomic<unsigned> counter{0};
  const std::filesystem::path  tmp = cacheFile.string() + ".tmp" + std::to_string(counter++) + "-"
                                    + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff);
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if(!out)
      return false;
    writeRaw(out, &h, 1);
    writeRaw(out, cbor.data(), cbor.size());
    for(const Rows& t : doc.tables)
    {
      const int32_t  count      = t.count;
      const uint32_t startCount = uint32_t(t.start.size());
      const uint64_t dataCount  = t.data.size();
      writeRaw(out, &count, 1);
      writeRaw(out, &startCount, 1);
      writeRaw(out, &dataCount, 1);
      writeRaw(out, t.start.data(), t.start.size());
      writeRaw(out, t.data.data(), t.data.size());
    }
    if(!out)
    {
      out.close();
      std::filesystem::remove(tmp, ec);
      return false;
    }
  }
  std::filesystem::rename(tmp, cacheFile, ec);
  if(ec)
  {
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

std::shared_ptr<Doc> readParsed(const std::filesystem::path& cacheFile, const ParsedStamp& stamp)
{
  std::ifstream in(cacheFile, std::ios::binary);
  if(!in)
    return nullptr;
  Header h{};
  if(!readRaw(in, &h, 1) || std::memcmp(h.magic, kMagic, 4) != 0 || h.version != kVersion || h.pathHash != stamp.pathHash
     || h.size != stamp.size || h.mtime != stamp.mtime)
    return nullptr;
  std::error_code ec;
  const uintmax_t fileSize = std::filesystem::file_size(cacheFile, ec);
  if(ec || h.cborBytes > fileSize)
    return nullptr;
  std::vector<uint8_t> cbor(h.cborBytes);
  if(!readRaw(in, cbor.data(), cbor.size()))
    return nullptr;
  auto doc   = std::make_shared<Doc>();
  doc->bytes = h.bytes;
  try
  {
    doc->json = Json::from_cbor(cbor);
  }
  catch(const std::exception&)
  {
    return nullptr;
  }
  doc->tables.resize(h.tables);
  for(Rows& t : doc->tables)
  {
    int32_t  count      = 0;
    uint32_t startCount = 0;
    uint64_t dataCount  = 0;
    if(!readRaw(in, &count, 1) || !readRaw(in, &startCount, 1) || !readRaw(in, &dataCount, 1))
      return nullptr;
    if(count < 0 || startCount != uint32_t(count) + 1 || dataCount > fileSize / sizeof(double))
      return nullptr;
    t.count = count;
    t.start.resize(startCount);
    t.data.resize(dataCount);
    if(!readRaw(in, t.start.data(), startCount) || !readRaw(in, t.data.data(), dataCount))
      return nullptr;
  }
  return doc;
}

}  // namespace fv::dson
