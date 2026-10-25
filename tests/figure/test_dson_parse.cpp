// The streaming DSON parse (parse_sax.cpp) gives the same document as the tree parse: the same
// JSON with every table packed, the same rows bit for bit. Over the test library's files, over the
// cases that make it fall back, and, with DAZ_LIBRARY set, over the real files a wear reads. The
// parsed cache (parsed_cache.cpp) gives the document back as it was and misses when the file changed.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/io/gzip.hpp>

namespace {

namespace fs = std::filesystem;
using fv::dson::Doc;
using fv::dson::Json;

std::string readFile(const fs::path& p)
{
  return fv::readMaybeCompressed(p);  // inflated when the file is gzip
}

// The tree with every packed table written back as rows of doubles, so two documents compare by
// content whatever order their tables were numbered in.
Json expanded(const Json& node, const Doc& doc)
{
  if(node.is_object())
  {
    if(node.contains("$table") && node.size() == 1)
    {
      const auto& rows = doc.tables.at(node["$table"].get<size_t>());
      Json        out  = Json::array();
      for(int i = 0; i < rows.count; i++)
      {
        Json row = Json::array();
        for(int32_t k = rows.start[i]; k < rows.start[i + 1]; k++)
          row.push_back(rows.data[k]);
        out.push_back(std::move(row));
      }
      return out;
    }
    Json out = Json::object();
    for(const auto& [k, v] : node.items())
      out[k] = expanded(v, doc);
    return out;
  }
  if(node.is_array())
  {
    Json out = Json::array();
    for(const Json& v : node)
      out.push_back(expanded(v, doc));
    return out;
  }
  return node;
}

void expectSame(const std::string& text, const std::string& what)
{
  const auto a = fv::dson::parseDoc(text);
  const auto b = fv::dson::parseDocDom(text);
  ASSERT_TRUE(a && b) << what;
  EXPECT_EQ(a->tables.size(), b->tables.size()) << what;
  EXPECT_EQ(expanded(a->json, *a).dump(), expanded(b->json, *b).dump()) << what;
  // the tables themselves, bit for bit (the same count of them, in some order: match by content)
  size_t matched = 0;
  for(const auto& ta : a->tables)
    for(const auto& tb : b->tables)
      if(ta.count == tb.count && ta.start == tb.start && ta.data.size() == tb.data.size()
         && (ta.data.empty() || std::memcmp(ta.data.data(), tb.data.data(), ta.data.size() * sizeof(double)) == 0))
      {
        matched++;
        break;
      }
  EXPECT_EQ(matched, a->tables.size()) << what;
  EXPECT_EQ(a->bytes, b->bytes) << what;
}

// Written to the cache and read back: the same tree, the same tables bit for bit, the same size.
void expectCacheRoundTrip(const Doc& doc, const std::string& what)
{
  const fs::path dir = fs::temp_directory_path() / "fv-parsed-cache-test";
  fs::create_directories(dir);
  const fv::dson::ParsedStamp stamp{.pathHash = 42, .size = 1234, .mtime = 5678};
  const fs::path              file = fv::dson::parsedCacheFile(dir, stamp);
  ASSERT_TRUE(fv::dson::writeParsed(file, doc, stamp)) << what;
  const auto back = fv::dson::readParsed(file, stamp);
  ASSERT_TRUE(back) << what;
  EXPECT_EQ(back->json.dump(), doc.json.dump()) << what;
  EXPECT_EQ(back->bytes, doc.bytes) << what;
  ASSERT_EQ(back->tables.size(), doc.tables.size()) << what;
  for(size_t i = 0; i < doc.tables.size(); i++)
  {
    EXPECT_EQ(back->tables[i].count, doc.tables[i].count) << what;
    EXPECT_EQ(back->tables[i].start, doc.tables[i].start) << what;
    ASSERT_EQ(back->tables[i].data.size(), doc.tables[i].data.size()) << what;
    EXPECT_EQ(std::memcmp(back->tables[i].data.data(), doc.tables[i].data.data(), doc.tables[i].data.size() * sizeof(double)), 0) << what;
  }
  fs::remove_all(dir);
}

}  // namespace

TEST(DsonParse, StreamedTablesMatchTheTreeParseOverTheTestLibrary)
{
  const fs::path root = fs::path(FV_TEST_DATA) / "daz-library";
  ASSERT_TRUE(fs::exists(root));
  size_t files = 0, tables = 0;
  for(const auto& entry : fs::recursive_directory_iterator(root))
  {
    const std::string ext = entry.path().extension().string();
    if(!entry.is_regular_file() || (ext != ".dsf" && ext != ".duf"))
      continue;
    const std::string text = readFile(entry.path());
    expectSame(text, entry.path().string());
    files++;
    tables += fv::dson::parseDoc(text)->tables.size();
  }
  EXPECT_GE(files, 8u);
  EXPECT_GE(tables, 4u);  // vertices, polylist, deltas, UVs, skin weights among the box figure's files
}

TEST(DsonParse, ATableThatIsNotRowsOfNumbersFallsBackToTheTree)
{
  // a string in a row, a nested array, a bare number: each leaves the table as it was, in both
  for(const char* values : {R"([[0, "x"], [1, 2]])", R"([[0, [1]], [1, 2]])", R"([1, 2, 3])", R"([[1, 2], 3])", R"([{"a": 1}])"})
  {
    const std::string text = std::string(R"({"geometry_library": [{"id": "g", "vertices": {"count": 2, "values": )") + values + "}}]}";
    const auto        a    = fv::dson::parseDoc(text);
    const auto        b    = fv::dson::parseDocDom(text);
    EXPECT_EQ(a->json.dump(), b->json.dump()) << values;
    EXPECT_TRUE(a->tables.empty()) << values;
    EXPECT_EQ(a->json["geometry_library"][0]["vertices"]["values"].dump(), Json::parse(values).dump()) << values;
  }
}

TEST(DsonParse, EdgeCasesMatch)
{
  // an empty table, integer and float scalars outside tables, a repeated key (the last one wins in
  // place, as Json::parse does), tables in every library, a table-shaped array where no table is
  expectSame(R"({"geometry_library": [{"vertices": {"values": []}}]})", "empty table");
  expectSame(R"({"a": 1, "b": 1.0, "c": -0.0, "d": 1e3, "geometry_library": [{"polylist": {"values": [[0, 1, 2, 3, 4]]}}]})", "scalars");
  expectSame(R"({"a": 1, "a": 2, "geometry_library": [{"vertices": {"values": [[1, 2, 3]], "values": [[4, 5, 6]]}}]})", "repeated key");
  expectSame(R"({"geometry_library": [{"vertices": {"values": [[1, 2, 3]], "values": 5}, "polylist": {"values": [[0, 1, 2]]}}]})", "table replaced by a scalar");
  expectSame(R"({"uv_set_library": [{"uvs": {"values": [[0.5, 0.25]]}, "polygon_vertex_indices": [[0, 1, 2]]}],)"
             R"( "modifier_library": [{"morph": {"deltas": {"values": [[0, 0.1, 0.2, 0.3]]}}},)"
             R"( {"skin": {"joints": [{"node_weights": {"values": [[0, 1.0]]}, "scale_weights": {"values": [[1, 0.5]]},)"
             R"( "local_weights": {"x": {"values": [[2, 0.25]]}, "y": {"values": []}, "z": {"values": [[3, 1]]}}}]}}],)"
             R"( "geometry_library": [{"graft": {"vertex_pairs": {"values": [[1, 2]]}}, "rigidity": {"weights": {"values": [[0, 0.5]]}}}]})",
             "every library");
  expectSame(R"({"node_library": [{"values": [[1, 2]]}], "geometry_library": [{"other": {"values": [[1, 2]]}}], "values": [[1]]})", "no table");
  expectSame(R"({"geometry_library": {"vertices": {"values": [[1, 2]]}}})", "library not an array");
  expectSame(R"({"geometry_library": [{"vertices": [{"values": [[1, 2]]}]}]})", "holder not an object");
}

TEST(DsonParse, RealFilesMatchWhenTheLibraryIsAvailable)
{
  const char* lib = std::getenv("DAZ_LIBRARY");
  if(!lib || !*lib)
    GTEST_SKIP() << "DAZ_LIBRARY not set";
  const fs::path root(lib);
  for(const char* rel : {"data/Daz 3D/Genesis 9/Base/Genesis9.dsf",
                         "data/outoftouch/Summer Style Wavy Long Bob Hair/Summer Style Wavy Long Bob Hair Strands/SummerStyleWaveLongBob_426558.dsf",
                         "data/outoftouch/Summer Style Wavy Long Bob Hair/Summer Style Wavy Long Bob Hair Strands/UV Sets/outoftouch/Base/default.dsf"})
  {
    const fs::path p = root / rel;
    if(!fs::exists(p))
    {
      ADD_FAILURE() << "missing " << p;
      continue;
    }
    const std::string text = readFile(p);
    expectSame(text, rel);
    expectCacheRoundTrip(*fv::dson::parseDoc(text), rel);
  }
}

TEST(DsonParse, TheParsedCacheGivesTheDocumentBackAndMissesWhenTheFileChanged)
{
  const std::string text = R"({"a": 1, "b": 1.5, "c": "x", "d": [true, null, -0.0, 1e3, 2.5e-7, 18446744073709551615, -9007199254740993],)"
                           R"( "geometry_library": [{"id": "g", "vertices": {"count": 2, "values": [[0, 1, 2], [3.5, 4, 5]]},)"
                           R"( "polylist": {"values": []}, "polyline_list": {"values": [[0, 1, 2, 3, 4]]}}]})";
  const auto doc = fv::dson::parseDoc(text);
  ASSERT_EQ(doc->tables.size(), 3u);
  expectCacheRoundTrip(*doc, "fixture");
  // another size, another time, another path: a miss; so is a file that is not there
  const fs::path dir = fs::temp_directory_path() / "fv-parsed-cache-test-stale";
  fs::create_directories(dir);
  const fv::dson::ParsedStamp stamp{.pathHash = 7, .size = text.size(), .mtime = 99};
  const fs::path              file = fv::dson::parsedCacheFile(dir, stamp);
  ASSERT_TRUE(fv::dson::writeParsed(file, *doc, stamp));
  EXPECT_TRUE(fv::dson::readParsed(file, stamp));
  EXPECT_FALSE(fv::dson::readParsed(file, fv::dson::ParsedStamp{.pathHash = 7, .size = text.size() + 1, .mtime = 99}));
  EXPECT_FALSE(fv::dson::readParsed(file, fv::dson::ParsedStamp{.pathHash = 7, .size = text.size(), .mtime = 100}));
  EXPECT_FALSE(fv::dson::readParsed(file, fv::dson::ParsedStamp{.pathHash = 8, .size = text.size(), .mtime = 99}));
  EXPECT_FALSE(fv::dson::readParsed(dir / "missing.fvd", stamp));
  // a damaged file is a miss, not a crash
  {
    std::ofstream cut(file, std::ios::binary | std::ios::trunc);
    cut << "FVD1 but nothing after the magic";
  }
  EXPECT_FALSE(fv::dson::readParsed(file, stamp));
  fs::remove_all(dir);
  // the stamp of a real file follows its size and time
  const fs::path src = dir.parent_path() / "fv-parsed-cache-src.json";
  {
    std::ofstream out(src, std::ios::binary | std::ios::trunc);
    out << text;
  }
  const auto s1 = fv::dson::stampOf(src);
  EXPECT_EQ(s1.size, text.size());
  EXPECT_NE(s1.pathHash, 0u);
  {
    std::ofstream out(src, std::ios::binary | std::ios::trunc);
    out << text << " ";
  }
  EXPECT_NE(fv::dson::stampOf(src).size, s1.size);
  EXPECT_EQ(fv::dson::stampOf(src).pathHash, s1.pathHash);
  fs::remove(src);
}
