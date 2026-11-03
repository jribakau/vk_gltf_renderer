#pragma once

// DSON (DAZ Scene Object Notation) file access. .duf and .dsf files are JSON, optionally
// gzip-compressed, and reference each other with URIs like
//   /data/DAZ%203D/Genesis%209/Base/Genesis9.dsf#Genesis9
// which resolve against the content library root. "#id" alone refers to the same file.
//
// DSON keeps its bulk data as tables of numeric rows: vertices [[x, y, z], ...], polygons
// [[group, material, v0, v1, ...], ...], morph deltas, skin weights, UVs. A Genesis figure has
// hundreds of thousands of these little arrays, so a parsed document has them taken out and packed
// (Rows: one flat array of 4-byte cells, typed by column), each replaced by {"$table": n}; read them
// through Doc::rows(). A port of src/lib/loaders/dson/parse.ts and rows.ts.
//
// Documents are kept as nlohmann::ordered_json: keys stay in file order, which decides how
// material channels override each other.

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
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

#include <atomic>

namespace fv::dson {

using Json = nlohmann::ordered_json;

// A table of numeric rows, flat: row i is cells begin(i) up to begin(i) + size(i). A cell takes 4
// bytes: a column (an entry's place in its row) whose numbers are all whole and within int32 keeps
// them as int32, exactly (vertex and polygon indices, past float's 2^24 too); any other column as
// float, which is what every reader makes of DSON's decimals (written to about seven digits). Rows of
// one length (vertices, UVs, weights, deltas: nearly all) keep that length, not their starts.
struct Rows
{
  int                   count = 0;
  int                   width = 0;  // every row's length when they share one above 0; 0: `start` has them
  std::vector<int32_t>  start;      // when width is 0: row i's first cell, and the end after the last
  std::vector<uint8_t>  whole;      // per column: 1 when its cells hold int32, 0 when they hold float
  std::vector<uint32_t> cells;

  size_t begin(int i) const { return width ? size_t(i) * size_t(width) : size_t(start[i]); }
  int    size(int i) const { return width ? width : start[i + 1] - start[i]; }
  // Row i's entry k, or NaN past the end of the row (as reading a short array gives undefined).
  double cell(int i, int k) const
  {
    if(k >= size(i))
      return NAN;
    const uint32_t c = cells[begin(i) + size_t(k)];
    return whole[size_t(k)] ? double(std::bit_cast<int32_t>(c)) : double(std::bit_cast<float>(c));
  }
  // what it holds in memory
  size_t bytes() const { return cells.size() * sizeof(uint32_t) + start.size() * sizeof(int32_t) + whole.size(); }
  bool   operator==(const Rows&) const = default;
};

// Rows from numbers given a row at a time, as the parsers read them: each column's type is known
// only once the table is complete.
class RowsBuilder
{
public:
  void add(double v) { m_values.push_back(v); }
  void endRow() { m_start.push_back(int32_t(m_values.size())); }
  // The table given so far; the builder is empty again after it.
  Rows finish();

private:
  std::vector<double>  m_values;
  std::vector<int32_t> m_start{0};
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
// thread after the parse. The folder keeps to 5 GB, least recently used out first.
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
// null otherwise (a missing, stale or damaged file is a miss, and a damaged one is deleted).
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

  // Forgets every shared parsed file: the library's contents changed.
  static void clearShared();

  // Lets go of the parsed files nothing but this store holds any more, which the shared cache has let
  // go of too: the next load reads them again. Not while a Hold stands. A failed read stays, so a missing
  // file is looked for once. Returns how many went.
  size_t trim();
  // A loader's raw pointers into its parsed files are good while this stands
  class Hold
  {
  public:
    explicit Hold(Store& store)
        : m_store(store)
    {
      m_store.m_holds++;
    }
    ~Hold() { m_store.m_holds--; }
    Hold(const Hold&)            = delete;
    Hold& operator=(const Hold&) = delete;

  private:
    Store& m_store;
  };

  // State a loader keeps with the store, which goes with it (the loader's meshes built ahead):
  // made by `make` the first time it is asked for.
  std::shared_ptr<void> attachment(const std::function<std::shared_ptr<void>()>& make);

private:
  std::shared_ptr<const Doc> fetch(const std::string& file);
  std::shared_ptr<const Doc> loadShared(const std::string& file);

  std::shared_ptr<const Library>                                       m_library;
  mutable std::mutex                                                   m_mutex;
  std::map<std::string, std::shared_future<std::shared_ptr<const Doc>>> m_cache;
  std::shared_ptr<void>                                                m_attachment;
  std::atomic<int>                                                     m_holds{0};
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
