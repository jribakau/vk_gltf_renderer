// The on-disk cache of parsed DSON files: the tree as CBOR (ordered_json keeps its key order and
// number types through it) and the packed tables raw (their 4-byte cells, as Rows holds them),
// stamped with the source file's path hash, size and modification time. A 76 MB strand-hair geometry parses in 0.8 s and reads back in tens
// of milliseconds. The folder keeps to a budget; a damaged file is a miss and goes.

#include <figure/dson/store.hpp>

#include <figure/io/cache_folder.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <new>
#include <string_view>
#include <thread>

namespace fv::dson {

namespace {

std::mutex            g_dirMutex;
std::filesystem::path g_dir;

constexpr char     kMagic[4] = {'F', 'V', 'D', '1'};
// 2: a table's cells are 4 bytes typed by column (Rows::whole), and rows of one length keep it, not
// their starts; a version 1 file is a miss, and the parse after it writes the file again
constexpr uint32_t kVersion  = 2;
// the files kept, oldest written or read first to go: a hair's is tens of MB
constexpr uintmax_t kBudget = 5ull << 30;

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

// The rest of a file whose stamp matched: null when its bytes don't add up to a document.
std::shared_ptr<Doc> readBody(std::istream& in, const Header& h, uintmax_t fileSize)
{
  // count, width, starts, columns, cells
  constexpr uint64_t kTableHead = 2 * sizeof(int32_t) + 2 * sizeof(uint32_t) + sizeof(uint64_t);
  if(fileSize < sizeof(Header) || h.cborBytes > fileSize - sizeof(Header))
    return nullptr;
  uint64_t left = fileSize - sizeof(Header) - h.cborBytes;
  if(h.tables > left / kTableHead)
    return nullptr;
  std::vector<uint8_t> cbor(h.cborBytes);
  if(!readRaw(in, cbor.data(), cbor.size()))
    return nullptr;
  auto doc   = std::make_shared<Doc>();
  doc->bytes = h.bytes;
  doc->json  = Json::from_cbor(cbor);
  doc->tables.resize(h.tables);
  for(Rows& t : doc->tables)
  {
    int32_t  count = 0, width = 0;
    uint32_t startCount = 0, columns = 0;
    uint64_t cellCount = 0;
    if(left < kTableHead || !readRaw(in, &count, 1) || !readRaw(in, &width, 1) || !readRaw(in, &startCount, 1)
       || !readRaw(in, &columns, 1) || !readRaw(in, &cellCount, 1))
      return nullptr;
    left -= kTableHead;
    if(count < 0 || width < 0)
      return nullptr;
    // rows of one length: no starts, a type for each place in them; otherwise the starts of all
    if(width > 0 ? startCount != 0 || columns != uint32_t(width) || cellCount != uint64_t(count) * uint64_t(width)
                 : startCount != uint32_t(count) + 1)
      return nullptr;
    if(columns > left)
      return nullptr;
    left -= columns;
    if(startCount > left / sizeof(int32_t))
      return nullptr;
    left -= uint64_t(startCount) * sizeof(int32_t);
    if(cellCount > left / sizeof(uint32_t))
      return nullptr;
    left -= cellCount * sizeof(uint32_t);
    t.count = count;
    t.width = width;
    t.whole.resize(columns);
    t.start.resize(startCount);
    t.cells.resize(cellCount);
    if(!readRaw(in, t.whole.data(), columns) || !readRaw(in, t.start.data(), startCount) || !readRaw(in, t.cells.data(), cellCount))
      return nullptr;
    if(std::any_of(t.whole.begin(), t.whole.end(), [](uint8_t w) { return w > 1; }))
      return nullptr;
    // every row within the cells, in order, none longer than the columns typed
    if(width == 0)
    {
      if(t.start.front() != 0 || uint64_t(t.start.back()) != cellCount || !std::is_sorted(t.start.begin(), t.start.end()))
        return nullptr;
      for(int i = 0; i < count; i++)
        if(uint32_t(t.start[i + 1] - t.start[i]) > columns)
          return nullptr;
    }
  }
  return left == 0 ? doc : nullptr;
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
  std::filesystem::path        tmp = cacheFile;
  tmp += ".tmp" + std::to_string(counter++) + "-" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff);
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if(!out)
      return false;
    writeRaw(out, &h, 1);
    writeRaw(out, cbor.data(), cbor.size());
    for(const Rows& t : doc.tables)
    {
      const int32_t  count = t.count, width = t.width;
      const uint32_t startCount = uint32_t(t.start.size()), columns = uint32_t(t.whole.size());
      const uint64_t cellCount = t.cells.size();
      writeRaw(out, &count, 1);
      writeRaw(out, &width, 1);
      writeRaw(out, &startCount, 1);
      writeRaw(out, &columns, 1);
      writeRaw(out, &cellCount, 1);
      writeRaw(out, t.whole.data(), t.whole.size());
      writeRaw(out, t.start.data(), t.start.size());
      writeRaw(out, t.cells.data(), t.cells.size());
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
  io::trimCacheFolder(cacheFile.parent_path(), ".fvd", kBudget);
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
  if(ec)
    return nullptr;
  std::shared_ptr<Doc> doc;
  try
  {
    doc = readBody(in, h, fileSize);
  }
  catch(const std::bad_alloc&)
  {
    return nullptr;  // nothing wrong with the file
  }
  catch(const std::exception&)
  {
  }
  in.close();
  if(doc)
    io::touchCacheFile(cacheFile);
  else
    std::filesystem::remove(cacheFile, ec);  // the next parse writes it again
  return doc;
}

}  // namespace fv::dson
