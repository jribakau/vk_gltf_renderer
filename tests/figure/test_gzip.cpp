// DSON files come plain or gzip-compressed; both read the same.

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

#include <figure/io/gzip.hpp>

namespace {

std::vector<uint8_t> gzip(const std::string& text)
{
  z_stream s{};
  // 15 window bits + 16: write a gzip header and trailer
  EXPECT_EQ(deflateInit2(&s, Z_BEST_SPEED, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY), Z_OK);
  std::vector<uint8_t> out(deflateBound(&s, uLong(text.size())) + 32);
  s.next_in   = reinterpret_cast<Bytef*>(const_cast<char*>(text.data()));
  s.avail_in  = uInt(text.size());
  s.next_out  = out.data();
  s.avail_out = uInt(out.size());
  EXPECT_EQ(deflate(&s, Z_FINISH), Z_STREAM_END);
  out.resize(s.total_out);
  deflateEnd(&s);
  return out;
}

std::vector<uint8_t> bytesOf(const std::string& text)
{
  return {text.begin(), text.end()};
}

}  // namespace

TEST(FigureGzip, PlainJsonPassesThrough)
{
  const std::string json = "{\"file_version\":\"0.6.0.0\"}";
  EXPECT_FALSE(fv::isGzip(bytesOf(json)));
  EXPECT_EQ(fv::inflateIfCompressed(bytesOf(json)), json);
}

TEST(FigureGzip, InflatesGzip)
{
  // larger than the first output guess (4x the input), so the buffer has to grow
  std::string json = "{\"a\":[";
  for(int i = 0; i < 20000; i++)
    json += "0,";
  json += "0]}";
  const auto packed = gzip(json);
  ASSERT_TRUE(fv::isGzip(packed));
  EXPECT_LT(packed.size() * 4, json.size());
  EXPECT_EQ(fv::inflateIfCompressed(packed), json);
}

TEST(FigureGzip, TruncatedStreamThrows)
{
  auto packed = gzip("{\"truncated\":true}");
  packed.resize(packed.size() / 2);
  EXPECT_THROW(fv::inflateIfCompressed(packed), std::runtime_error);
}
