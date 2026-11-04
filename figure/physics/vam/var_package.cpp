#include <figure/physics/vam/var_package.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include <zlib.h>

namespace fv::physics::vam {

namespace {

struct Entry
{
  std::string name;
  uint16_t    flags = 0, method = 0;
  uint32_t    crc    = 0;
  uint64_t    packed = 0, size = 0, local = 0;
};

uint16_t u16(const std::string& b, size_t at)
{
  return uint16_t(uint8_t(b[at]) | uint8_t(b[at + 1]) << 8);
}

uint32_t u32(const std::string& b, size_t at)
{
  return uint32_t(uint8_t(b[at])) | uint32_t(uint8_t(b[at + 1])) << 8 | uint32_t(uint8_t(b[at + 2])) << 16 | uint32_t(uint8_t(b[at + 3])) << 24;
}

uint64_t u64(const std::string& b, size_t at)
{
  return uint64_t(u32(b, at)) | uint64_t(u32(b, at + 4)) << 32;
}

std::string readAt(std::ifstream& in, uint64_t at, size_t bytes)
{
  std::string out(bytes, '\0');
  in.seekg(std::streamoff(at));
  if(bytes && !in.read(out.data(), std::streamsize(bytes)))
    throw std::runtime_error("the package ends early");
  return out;
}

std::string label(const std::filesystem::path& p)
{
  const std::u8string s = p.filename().u8string();
  return std::string(s.begin(), s.end());
}

// The central directory: every entry's name, how it is stored, and where its local header is
std::vector<Entry> directory(std::ifstream& in, const std::filesystem::path& path)
{
  in.seekg(0, std::ios::end);
  const uint64_t size = uint64_t(in.tellg());
  // the end record is the last 22 bytes before a comment of at most 64 KiB
  const size_t      tail = size_t(std::min<uint64_t>(size, 22 + 0xFFFF));
  const std::string end  = readAt(in, size - tail, tail);
  size_t            at   = std::string::npos;
  for(size_t i = tail >= 22 ? tail - 22 + 1 : 0; i-- > 0;)
    if(u32(end, i) == 0x06054b50)
    {
      at = i;
      break;
    }
  if(at == std::string::npos)
    throw std::runtime_error(label(path) + " isn't a zip, as a VaM package is");
  uint64_t count    = u16(end, at + 10);
  uint64_t cdSize   = u32(end, at + 12);
  uint64_t cdOffset = u32(end, at + 16);
  // zip64: the locator just before the end record points at the record with the full sizes
  if(count == 0xFFFF || cdOffset == 0xFFFFFFFF || cdSize == 0xFFFFFFFF)
  {
    const uint64_t endAt = size - tail + at;
    if(endAt < 20)
      throw std::runtime_error(label(path) + " is damaged (its zip64 locator is missing)");
    const std::string locator = readAt(in, endAt - 20, 20);
    if(u32(locator, 0) != 0x07064b50)
      throw std::runtime_error(label(path) + " is damaged (its zip64 locator is missing)");
    const uint64_t recordAt = u64(locator, 8);
    if(recordAt + 56 > size)
      throw std::runtime_error(label(path) + " is damaged (its zip64 record is past its end)");
    const std::string record = readAt(in, recordAt, 56);
    if(u32(record, 0) != 0x06064b50)
      throw std::runtime_error(label(path) + " is damaged (its zip64 record doesn't read)");
    count    = u64(record, 32);
    cdSize   = u64(record, 40);
    cdOffset = u64(record, 48);
  }
  if(cdOffset + cdSize > size || cdSize > (uint64_t(1) << 31))
    throw std::runtime_error(label(path) + " is damaged (its directory is past its end)");
  const std::string cd = readAt(in, cdOffset, size_t(cdSize));
  std::vector<Entry> out;
  size_t             p = 0;
  for(uint64_t i = 0; i < count; i++)
  {
    if(p + 46 > cd.size() || u32(cd, p) != 0x02014b50)
      throw std::runtime_error(label(path) + " is damaged (its directory doesn't read)");
    Entry e;
    e.flags                = u16(cd, p + 8);
    e.method               = u16(cd, p + 10);
    e.crc                  = u32(cd, p + 16);
    e.packed               = u32(cd, p + 20);
    e.size                 = u32(cd, p + 24);
    const size_t nameBytes = u16(cd, p + 28), extra = u16(cd, p + 30), comment = u16(cd, p + 32);
    e.local                = u32(cd, p + 42);
    if(p + 46 + nameBytes + extra + comment > cd.size())
      throw std::runtime_error(label(path) + " is damaged (an entry runs past its directory)");
    e.name = cd.substr(p + 46, nameBytes);
    std::replace(e.name.begin(), e.name.end(), '\\', '/');
    // zip64's sizes and offset, in its extra field, for each the entry gives as 0xFFFFFFFF, in this order
    for(size_t x = p + 46 + nameBytes, xEnd = x + extra; x + 4 <= xEnd;)
    {
      const size_t id = u16(cd, x), bytes = u16(cd, x + 2);
      if(x + 4 + bytes > xEnd)
        break;
      if(id == 0x0001)
      {
        size_t f = x + 4;
        for(uint64_t* field : {e.size == 0xFFFFFFFF ? &e.size : nullptr, e.packed == 0xFFFFFFFF ? &e.packed : nullptr, e.local == 0xFFFFFFFF ? &e.local : nullptr})
          if(field && f + 8 <= x + 4 + bytes)
          {
            *field = u64(cd, f);
            f += 8;
          }
      }
      x += 4 + bytes;
    }
    p += 46 + nameBytes + extra + comment;
    out.push_back(std::move(e));
  }
  return out;
}

bool endsWithNoCase(const std::string& s, const std::string& end)
{
  if(s.size() < end.size())
    return false;
  for(size_t i = 0; i < end.size(); i++)
    if(std::tolower(uint8_t(s[s.size() - end.size() + i])) != std::tolower(uint8_t(end[i])))
      return false;
  return true;
}

bool startsWithNoCase(const std::string& s, const std::string& start)
{
  if(s.size() < start.size())
    return false;
  for(size_t i = 0; i < start.size(); i++)
    if(std::tolower(uint8_t(s[i])) != std::tolower(uint8_t(start[i])))
      return false;
  return true;
}

}  // namespace

std::vector<std::string> zipEntries(const std::filesystem::path& zip)
{
  std::ifstream in(zip, std::ios::binary);
  if(!in)
    throw std::runtime_error("can't open " + label(zip));
  std::vector<std::string> out;
  for(const Entry& e : directory(in, zip))
    out.push_back(e.name);
  return out;
}

std::string zipRead(const std::filesystem::path& zip, const std::string& name, size_t maxBytes)
{
  std::ifstream in(zip, std::ios::binary);
  if(!in)
    throw std::runtime_error("can't open " + label(zip));
  in.seekg(0, std::ios::end);
  const uint64_t           size    = uint64_t(in.tellg());
  const std::vector<Entry> entries = directory(in, zip);
  const auto it = std::find_if(entries.begin(), entries.end(), [&](const Entry& e) { return e.name == name; });
  if(it == entries.end())
    throw std::runtime_error(label(zip) + " has no " + name);
  const Entry& e = *it;
  if(e.flags & 1)
    throw std::runtime_error(name + " is encrypted in " + label(zip));
  if(e.method != 0 && e.method != 8)
    throw std::runtime_error(name + " is compressed in a way this doesn't read (method " + std::to_string(e.method) + ")");
  if(e.size > maxBytes)
    throw std::runtime_error(name + " is too large to read (" + std::to_string(e.size >> 20) + " MiB)");
  if(e.local + 30 > size)
    throw std::runtime_error(label(zip) + " is damaged (an entry is past its end)");
  const std::string local = readAt(in, e.local, 30);
  if(u32(local, 0) != 0x04034b50)
    throw std::runtime_error(label(zip) + " is damaged (an entry's header doesn't read)");
  const uint64_t data = e.local + 30 + u16(local, 26) + u16(local, 28);
  if(data + e.packed > size || e.packed > maxBytes)
    throw std::runtime_error(label(zip) + " is damaged (an entry runs past its end)");
  std::string packed = readAt(in, data, size_t(e.packed));
  std::string out;
  if(e.method == 0)
  {
    if(e.packed != e.size)
      throw std::runtime_error(label(zip) + " is damaged (a stored entry's sizes differ)");
    out = std::move(packed);
  }
  else
  {
    out.assign(size_t(e.size), '\0');
    z_stream s{};
    if(inflateInit2(&s, -15) != Z_OK)
      throw std::runtime_error("inflate: could not initialise zlib");
    s.next_in   = reinterpret_cast<Bytef*>(packed.data());
    s.avail_in  = uInt(packed.size());
    s.next_out  = reinterpret_cast<Bytef*>(out.data());
    s.avail_out = uInt(out.size());
    const int ret = inflate(&s, Z_FINISH);
    const uLong n = s.total_out;
    inflateEnd(&s);
    if(ret != Z_STREAM_END || n != e.size)
      throw std::runtime_error(name + " is damaged in " + label(zip));
  }
  if(uint32_t(crc32(0L, reinterpret_cast<const Bytef*>(out.data()), uInt(out.size()))) != e.crc)
    throw std::runtime_error(name + " is damaged in " + label(zip) + " (its checksum differs)");
  return out;
}

std::vector<std::string> varScenes(const std::filesystem::path& var)
{
  std::vector<std::string> out;
  for(const std::string& name : zipEntries(var))
    if(startsWithNoCase(name, "Saves/scene/") && endsWithNoCase(name, ".json"))
      out.push_back(name);
  return out;
}

}  // namespace fv::physics::vam
