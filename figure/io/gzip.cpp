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

// Bytes to skip to reach the deflate stream of a gzip member. 0 when it isn't gzip.
static size_t gzipPayload(std::span<const uint8_t> bytes)
{
  if(!isGzip(bytes) || bytes.size() < 10 || bytes[2] != 8)
    return 0;
  const uint8_t flags = bytes[3];
  size_t        i     = 10;
  if(flags & 4)
  {
    if(i + 2 > bytes.size())
      return 0;
    i += 2u + (size_t(bytes[i]) | size_t(bytes[i + 1]) << 8);
  }
  if(flags & 8)
    while(i < bytes.size() && bytes[i++])
      ;
  if(flags & 16)
    while(i < bytes.size() && bytes[i++])
      ;
  if(flags & 2)
    i += 2;
  return i < bytes.size() ? i : 0;
}

std::string inflateIfCompressed(std::span<const uint8_t> bytes)
{
  const size_t gzipAt = gzipPayload(bytes);
  if(!gzipAt && isGzip(bytes))
    throw std::runtime_error("inflate: truncated or unsupported gzip header");
  const bool zlib = !gzipAt && isZlib(bytes);
  if(!gzipAt && !zlib)
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());

  z_stream s{};
  // Raw deflate for gzip (the header is parsed above). Window 47, which asks zlib to detect gzip
  // itself, is rejected by the zlib libjpeg ships, and the renderer links that one too.
  if(inflateInit2(&s, gzipAt ? -15 : 15) != Z_OK)
    throw std::runtime_error("inflate: could not initialise zlib");
  s.next_in  = const_cast<Bytef*>(bytes.data() + gzipAt);
  s.avail_in = uInt(bytes.size() - gzipAt);

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
