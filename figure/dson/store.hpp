#pragma once

// DSON (DAZ Scene Object Notation) file access. .duf and .dsf files are JSON, optionally
// gzip-compressed, and reference each other with URIs like
//   /data/DAZ%203D/Genesis%209/Base/Genesis9.dsf#Genesis9
// which resolve against the content library root. "#id" alone refers to the same file.
//
// DSON keeps its bulk data as tables of numeric rows: vertices [[x, y, z], ...], polygons
// [[group, material, v0, v1, ...], ...], morph deltas, skin weights, UVs. A Genesis figure has
// hundreds of thousands of these little arrays, so a parsed document has them taken out and packed
// (Rows: one flat array and the rows' starts), each replaced by {"$table": n}; read them through
// Doc::rows(). A port of src/lib/loaders/dson/parse.ts and rows.ts.
//
// Documents are kept as nlohmann::ordered_json: keys stay in file order, which decides how
// material channels override each other.

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <tinygltf/json.hpp>

namespace fv::dson {

using Json = nlohmann::ordered_json;

// A table of numeric rows, flat: row i is data[start[i]] up to data[start[i + 1]].
struct Rows
{
  int                  count = 0;
  std::vector<int32_t> start{0};
  std::vector<double>  data;

  int size(int i) const { return start[i + 1] - start[i]; }
  // Row i's entry k, or NaN past the end of the row (as reading a short array gives undefined).
  double cell(int i, int k) const
  {
    const int at = start[i] + k;
    return at < start[i + 1] ? data[at] : NAN;
  }
};

// A parsed document: its JSON, with the bulk tables packed.
struct Doc
{
  Json              json;
  std::vector<Rows> tables;
  size_t            bytes = 0;  // roughly what it holds in memory, for the shared cache

  // The table at `node`: packed as it came, or packed now from plain arrays; empty when missing or
  // not a table of numbers.
  std::shared_ptr<const Rows> rows(const Json* node) const;
  // Whether the table at `node` is present with at least one row.
  bool hasRows(const Json* node) const;
};

// Parses DSON text and packs its tables: the bulk tables stream into Rows as the text is read
// (parse_sax.cpp), the rest becomes the tree. parseDocDom builds the whole tree first and packs
// afterwards; parseDoc falls back to it for a table that holds anything but rows of numbers, and
// the tests hold the two to the same result.
std::shared_ptr<Doc> parseDoc(std::string text);
std::shared_ptr<Doc> parseDocDom(std::string text);

// An on-disk cache of parsed files (parsed_cache.cpp), keyed by the file's path, size and
// modification time: the tree as CBOR (ordered_json keeps key order and number types through it),
// the tables raw. A 76 MB strand-hair geometry parses in 0.8 s and reads back in tens of ms. Set at
// start-up; empty (the default) is off. Files of 2 MB and more go through it, written on a pool
// thread after the parse.
void                  setParsedCacheDir(const std::filesystem::path& dir);
std::filesystem::path parsedCacheDir();
struct ParsedStamp
{
  uint64_t pathHash = 0, size = 0;
  int64_t  mtime = 0;
};
ParsedStamp           stampOf(const std::filesystem::path& file);
std::filesystem::path parsedCacheFile(const std::filesystem::path& dir, const ParsedStamp& stamp);
// Writes the document through a temporary beside the file; reads it back when the stamp matches,
// null otherwise (a missing, stale or damaged file is a miss).
bool                 writeParsed(const std::filesystem::path& cacheFile, const Doc& doc, const ParsedStamp& stamp);
std::shared_ptr<Doc> readParsed(const std::filesystem::path& cacheFile, const ParsedStamp& stamp);

struct Ref
{
  // Library-relative file path without leading slash, e.g. "data/DAZ 3D/Genesis 9/Base/Genesis9.dsf".
  std::string file;
  // Fragment id, decoded. Empty when the URI names a whole file.
  std::string id;
};

// Parses a DSON asset URI, resolving "#id" against `currentFile`. None for scene-local schemes.
std::optional<Ref> parseRef(std::string_view uri, std::string_view currentFile);

// Percent-decodes a URI component; the input unchanged when it is malformed.
std::string decodeUri(std::string_view s);

// A referenced library file that isn't installed in any content folder.
class MissingFileError : public std::runtime_error
{
public:
  explicit MissingFileError(std::string file)
      : std::runtime_error(file + " is not installed")
      , m_file(std::move(file))
  {
  }
  const std::string& file() const { return m_file; }

private:
  std::string m_file;
};

// Content folders searched in order for a library-relative path (the app's assets folder, then the
// DAZ library).
class Library
{
public:
  explicit Library(std::vector<std::filesystem::path> roots)
      : m_roots(std::move(roots))
  {
  }
  const std::vector<std::filesystem::path>& roots() const { return m_roots; }
  // The file on disk, or none.
  std::optional<std::filesystem::path> find(std::string_view rel) const;

private:
  std::vector<std::filesystem::path> m_roots;
};

class Store
{
public:
  explicit Store(std::shared_ptr<const Library> library);

  const Library& library() const { return *m_library; }

  // Loads and parses a library file, decompressing gzip transparently. Cached per path; .dsf files
  // are also shared with other stores (bounded, least recently used out first). Throws
  // MissingFileError when no content folder has it.
  std::shared_ptr<const Doc> load(const std::string& file);

  struct Resolved
  {
    std::shared_ptr<const Doc> doc;
    const Json*                entry = nullptr;
    std::string                file;
  };
  // Finds the library entry a reference points at, in `preferLibrary` (e.g. "node_library") or
  // else every *_library array in the file. The first entry with the id wins.
  std::optional<Resolved> resolve(const Ref& ref, const char* preferLibrary = nullptr);

  // Library files that were referenced but not found, in the order first asked for.
  std::vector<std::string> missing() const;

  // Forgets every shared parsed file: the library's contents changed.
  static void clearShared();

private:
  std::shared_ptr<const Doc> fetch(const std::string& file);
  std::shared_ptr<const Doc> loadShared(const std::string& file);

  std::shared_ptr<const Library>                                       m_library;
  mutable std::mutex                                                   m_mutex;
  std::map<std::string, std::shared_future<std::shared_ptr<const Doc>>> m_cache;
  std::vector<std::string>                                             m_missing;
  // per document and library array, entries by id
  std::map<std::pair<const Doc*, std::string>, std::map<std::string, const Json*>> m_index;
};

// Reads a DSON float channel, preferring the scene's current value.
double channelValue(const Json* ch, double fallback = 0);
// DSON transform channels are arrays of {id: "x"|"y"|"z", value}.
std::array<double, 3> vec3Channel(const Json* arr, std::array<double, 3> fallback = {0, 0, 0});

// Small accessors: the member, or null when absent (or not an object).
const Json* get(const Json* obj, std::string_view key);
std::string getString(const Json* obj, std::string_view key, std::string fallback = {});
// A JSON number as JavaScript's ToInt32 would store it into an Int32Array (NaN and infinities: 0).
int32_t toInt32(double v);

}  // namespace fv::dson
