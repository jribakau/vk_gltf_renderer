// The streaming DSON parse (parse_sax.cpp) gives the same document as the tree parse: the same
// JSON with every table packed, the same rows bit for bit. Over the test library's files, over the
// cases that make it fall back, and, with DAZ_LIBRARY set, over the real files a wear reads. The
// parsed cache (parsed_cache.cpp) gives the document back as it was and misses when the file changed.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <figure/dson/listing.hpp>
#include <figure/dson/store.hpp>
#include <figure/io/cache_folder.hpp>
#include <figure/io/gzip.hpp>

namespace {

namespace fs = std::filesystem;
using fv::dson::Doc;
using fv::dson::Json;

std::string readFile(const fs::path& p)
{
  return fv::readMaybeCompressed(p);  // inflated when the file is gzip
}

// The tree with every packed table written back as rows of numbers (as the cells hold them), so two
// documents compare by content whatever order their tables were numbered in.
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
        for(int k = 0; k < rows.size(i); k++)
          row.push_back(rows.cell(i, k));
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
      if(ta == tb)
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
    EXPECT_TRUE(back->tables[i] == doc.tables[i]) << what << ", table " << i;
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

// A store lets go of the files only it holds, between loads: the next load reads them again. A failed
// read stays, so a missing file is looked for once.
TEST(DsonParse, AStoreLetsGoOfWhatOnlyItHolds)
{
  const auto       library = std::make_shared<fv::dson::Library>(std::vector<fs::path>{fs::path(FV_TEST_DATA) / "daz-library"});
  fv::dson::Store  store(library);
  const std::string kept = "People/Test/Box Figure.duf", loose = "People/Test/Clothing/Box Shirt.duf";
  const auto        held = store.load(kept);
  std::weak_ptr<const fv::dson::Doc> gone = store.load(loose);
  EXPECT_THROW(store.load("People/Test/not there.duf"), fv::dson::MissingFileError);
  ASSERT_FALSE(gone.expired());
  {
    const fv::dson::Store::Hold hold(store);
    EXPECT_EQ(store.trim(), 0u);
    EXPECT_FALSE(gone.expired());
  }
  EXPECT_EQ(store.trim(), 1u);
  EXPECT_TRUE(gone.expired());
  EXPECT_EQ(store.load(kept), held);
  EXPECT_TRUE(store.load(loose));
  EXPECT_THROW(store.load("People/Test/not there.duf"), fv::dson::MissingFileError);
}

// A table's cells are 4 bytes typed by column: a column of whole numbers keeps them exactly as int32,
// past float's 2^24 too, and any other column is float; rows of one length keep that length.
TEST(DsonParse, ATablesWholeColumnsStayExactAndTheRestAreFloat)
{
  const std::string text = R"({"modifier_library": [{"morph": {"deltas": {"values": [[16777217, 0.1, 2, -3], [20032121, -2.5e-7, 4, 5.5]]}}}],)"
                           R"( "geometry_library": [{"polylist": {"values": [[0, 1, 2, 3, 4], [0, 1, 5, 6, 7, 8]]}}]})";
  for(const auto& doc : {fv::dson::parseDoc(text), fv::dson::parseDocDom(text)})
  {
    ASSERT_EQ(doc->tables.size(), 2u);
    const auto            deltasAt = doc->rows(&doc->json["modifier_library"][0]["morph"]["deltas"]["values"]);
    const auto            polysAt  = doc->rows(&doc->json["geometry_library"][0]["polylist"]["values"]);
    const fv::dson::Rows& deltas   = *deltasAt;
    EXPECT_EQ(deltas.count, 2);
    EXPECT_EQ(deltas.width, 4);
    EXPECT_TRUE(deltas.start.empty());
    EXPECT_EQ(deltas.whole, (std::vector<uint8_t>{1, 0, 1, 0}));
    EXPECT_EQ(deltas.cell(0, 0), 16777217.0);  // float would make it 16777216
    EXPECT_EQ(deltas.cell(1, 0), 20032121.0);
    EXPECT_EQ(deltas.cell(0, 1), double(0.1f));
    EXPECT_EQ(deltas.cell(1, 1), double(-2.5e-7f));
    EXPECT_EQ(deltas.cell(1, 2), 4.0);
    EXPECT_EQ(deltas.cell(0, 3), -3.0);  // a whole number in a column that is not
    EXPECT_EQ(deltas.cell(1, 3), 5.5);
    EXPECT_TRUE(std::isnan(deltas.cell(0, 4)));
    EXPECT_EQ(deltas.bytes(), 8 * sizeof(uint32_t) + 4);
    // rows of several lengths keep their starts
    const fv::dson::Rows& polys = *polysAt;
    EXPECT_EQ(polys.width, 0);
    EXPECT_EQ(polys.start, (std::vector<int32_t>{0, 5, 11}));
    EXPECT_EQ(polys.whole, (std::vector<uint8_t>(6, 1)));
    EXPECT_EQ(polys.size(1), 6);
    EXPECT_EQ(polys.cell(1, 5), 8.0);
    EXPECT_TRUE(std::isnan(polys.cell(0, 5)));
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
  // a file of the format before (doubles, every table's starts) is a miss
  {
    std::fstream old(file, std::ios::binary | std::ios::in | std::ios::out);
    const uint32_t version = 1;
    old.seekp(4);  // Header::version
    old.write(reinterpret_cast<const char*>(&version), sizeof version);
  }
  EXPECT_FALSE(fv::dson::readParsed(file, stamp));
  // a damaged file is a miss, not a crash
  {
    std::ofstream cut(file, std::ios::binary | std::ios::trunc);
    cut << "FVD1 but nothing after the magic";
  }
  EXPECT_FALSE(fv::dson::readParsed(file, stamp));
  // and one whose stamp matches is deleted, so the next parse writes it again: cut short, a byte changed
  // at each of its places, or claiming more tables than it could hold
  const std::vector<char> whole = [&] {
    fv::dson::writeParsed(file, *doc, stamp);
    std::ifstream in(file, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(in), {});
  }();
  const auto written = [&](const std::vector<char>& bytes) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
  };
  for(const size_t keep : {whole.size() - 1, whole.size() / 2, size_t(60)})
  {
    written({whole.begin(), whole.begin() + std::ptrdiff_t(keep)});
    EXPECT_FALSE(fv::dson::readParsed(file, stamp)) << keep;
    EXPECT_FALSE(fs::exists(file)) << keep;
  }
  {
    std::vector<char> huge = whole;
    const uint32_t    tables = 0xfffffff0u;
    std::memcpy(huge.data() + 48, &tables, sizeof(tables));  // Header::tables
    written(huge);
    EXPECT_FALSE(fv::dson::readParsed(file, stamp));
    EXPECT_FALSE(fs::exists(file));
  }
  // anything else: nothing thrown, whatever each byte of the tables becomes, and never a table that
  // reads outside itself
  for(size_t at = 56; at < whole.size(); at += 3)
  {
    std::vector<char> bad = whole;
    bad[at]               = char(bad[at] ^ 0xa5);
    written(bad);
    if(const auto back = fv::dson::readParsed(file, stamp))
      for(const fv::dson::Rows& t : back->tables)
      {
        if(t.width)
          EXPECT_TRUE(t.start.empty() && t.cells.size() == size_t(t.count) * size_t(t.width) && t.whole.size() == size_t(t.width));
        else
        {
          ASSERT_EQ(t.start.size(), size_t(t.count) + 1);
          EXPECT_EQ(size_t(t.start.back()), t.cells.size());
          for(int i = 0; i < t.count; i++)
            EXPECT_LE(size_t(t.size(i)), t.whole.size());
        }
      }
  }
  fs::remove_all(dir);
  // the same file read again keeps its place in the folder, and the folder keeps to its budget
  {
    const fs::path trimmed = fs::temp_directory_path() / "fv-cache-folder-test";
    fs::remove_all(trimmed);
    fs::create_directories(trimmed);
    const auto put = [&](const char* name, int age) {
      std::ofstream(trimmed / name, std::ios::binary) << std::string(1000, 'x');
      fs::last_write_time(trimmed / name, fs::file_time_type::clock::now() - std::chrono::hours(age));
    };
    put("old.fvd", 3);
    put("older.fvd", 4);
    put("new.fvd", 2);
    put("other.fvb", 9);  // not this cache's
    fv::io::trimCacheFolder(trimmed, ".fvd", 3000);
    EXPECT_TRUE(fs::exists(trimmed / "older.fvd"));  // within it
    fv::io::trimCacheFolder(trimmed, ".fvd", 2500);  // to three quarters of it: 1875
    EXPECT_FALSE(fs::exists(trimmed / "older.fvd"));
    EXPECT_FALSE(fs::exists(trimmed / "old.fvd"));
    EXPECT_TRUE(fs::exists(trimmed / "new.fvd"));
    EXPECT_TRUE(fs::exists(trimmed / "other.fvb"));
    put("first.fvd", 5);
    fv::io::touchCacheFile(trimmed / "first.fvd");
    fv::io::trimCacheFolder(trimmed, ".fvd", 1500);  // to 1125
    EXPECT_TRUE(fs::exists(trimmed / "first.fvd"));
    EXPECT_FALSE(fs::exists(trimmed / "new.fvd"));
    fs::remove_all(trimmed);
  }
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

// A folder listing takes any file name, in UTF-8 as library paths are: a narrow path's conversion
// throws on Windows for a name outside the system's code page, which stopped the item loading
TEST(DsonParse, AFolderListingTakesAnyName)
{
  namespace fs       = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "fv_listing_test";
  fs::remove_all(dir);
  fs::create_directories(dir / "Morphs");
  const std::u8string name = u8"Reina_Xiāo.dsf";
  std::ofstream(dir / "Morphs" / fs::path(name)).put('x');
  std::ofstream(dir / "Morphs" / "plain.dsf").put('x');

  const fv::dson::Library library({dir});
  fv::dson::clearListings();
  const std::vector<std::string> files = fv::dson::listFiles(library, "Morphs", ".dsf");
  ASSERT_EQ(files.size(), 2u);
  const std::string expected = "Morphs/" + std::string(name.begin(), name.end());
  EXPECT_NE(std::find(files.begin(), files.end(), expected), files.end());
  // and the library finds it by the name the listing gave
  EXPECT_TRUE(library.find(expected).has_value());

  fv::dson::clearListings();
  fs::remove_all(dir);
}

TEST(Listing, AFolderOutsideTheSystemsCodePageIsListed)
{
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / L"fv \x30e9\x30a4\x30d6\x30e9\x30ea";
  fs::remove_all(dir);
  fs::create_directories(dir / "Morphs");
  std::ofstream(dir / "Morphs" / "one.dsf").put('x');
  const fv::dson::Library library({dir});
  fv::dson::clearListings();
  EXPECT_EQ(fv::dson::listFiles(library, "Morphs", ".dsf").size(), 1u);
  fv::dson::clearListings();
  fs::remove_all(dir);
}
