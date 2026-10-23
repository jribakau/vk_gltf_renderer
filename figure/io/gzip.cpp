#include <figure/io/gzip.hpp>

#include <fstream>
#include <stdexcept>
#include <vector>

#include <zlib.h>

namespace fv {

bool isGzip(std::span<const uint8_t> bytes)
{
  return bytes.size() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b;
}

// zlib streams start with a CMF byte whose low nibble is 8 (deflate) and a header that is a
// multiple of 31; plain JSON starts with '{' or whitespace, which never passes both checks.
static bool isZlib(std::span<const uint8_t> bytes)
{
  return bytes.size() >= 2 && (bytes[0] & 0x0f) == 8 && ((bytes[0] << 8) | bytes[1]) % 31 == 0;
}

std::string inflateIfCompressed(std::span<const uint8_t> bytes)
{
  if(!isGzip(bytes) && !isZlib(bytes))
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());

  z_stream s{};
  // 15 window bits + 32: detect gzip or zlib from the header
  if(inflateInit2(&s, 15 + 32) != Z_OK)
    throw std::runtime_error("inflate: could not initialise zlib");
  s.next_in  = const_cast<Bytef*>(bytes.data());
  s.avail_in = uInt(bytes.size());

  std::string out;
  out.resize(bytes.size() * 4 + 4096);
  int ret = Z_OK;
  while(true)
  {
    s.next_out  = reinterpret_cast<Bytef*>(out.data() + s.total_out);
    s.avail_out = uInt(out.size() - s.total_out);
    ret         = inflate(&s, Z_NO_FLUSH);
    if(ret == Z_STREAM_END)
      break;
    if(ret != Z_OK && ret != Z_BUF_ERROR)
      break;
    if(s.avail_out == 0)
      out.resize(out.size() * 2);
    else if(ret == Z_BUF_ERROR)
      break;  // no progress with room left: the input ended early
  }
  const uLong total = s.total_out;
  inflateEnd(&s);
  if(ret != Z_STREAM_END)
    throw std::runtime_error("inflate: corrupt or truncated stream");
  out.resize(total);
  return out;
}

std::string readMaybeCompressed(const std::filesystem::path& path)
{
  std::ifstream f(path, std::ios::binary);
  if(!f)
    throw std::runtime_error("cannot read " + path.string());
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return inflateIfCompressed(bytes);
}

}  // namespace fv
