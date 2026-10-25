// The DSON parse that streams the bulk tables: nlohmann's SAX events build the document tree as
// Json::parse would (the same lexer, the same number conversions, the same handling of a repeated
// key), except at the places eachTable() packs, where the rows' numbers go straight into a Rows and
// the tree gets the {"$table": n} placeholder. A 76 MB strand-hair geometry is mostly those numbers,
// and building a tree node for each of them was most of its parse.

#include <figure/dson/store.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

static_assert(NLOHMANN_JSON_VERSION_MAJOR == 3 && NLOHMANN_JSON_VERSION_MINOR == 12,
              "parse_sax.cpp uses nlohmann::detail::json_sax_dom_parser; after a version bump rerun the parse equality tests");

namespace fv::dson {

namespace {

// Where in a DSON document the parser is, as far as the table paths go. Every other place is `none`.
enum class At : uint8_t
{
  none,
  root,          // the document object
  geoArray,      // geometry_library
  geo,           // one geometry
  geoTableHolder,  // vertices | polylist | polyline_list: its "values" is a table
  graft,         // graft
  graftPairs,    // graft/vertex_pairs
  rigidity,      // rigidity
  rigidityWeights,  // rigidity/weights
  modArray,      // modifier_library
  mod,           // one modifier
  morph,         // morph
  deltas,        // morph/deltas
  skin,          // skin
  joints,        // skin/joints
  joint,         // one joint
  jointWeights,  // node_weights | scale_weights
  local,         // local_weights
  localAxis,     // local_weights/x|y|z
  uvArray,       // uv_set_library
  uv,            // one UV set
  uvs,           // uvs
  table,         // the value here is a table when it is an array
};

At childOfObject(At at, std::string_view key)
{
  switch(at)
  {
    case At::root:
      return key == "geometry_library" ? At::geoArray : key == "modifier_library" ? At::modArray : key == "uv_set_library" ? At::uvArray : At::none;
    case At::geo:
      if(key == "vertices" || key == "polylist" || key == "polyline_list")
        return At::geoTableHolder;
      return key == "graft" ? At::graft : key == "rigidity" ? At::rigidity : At::none;
    case At::geoTableHolder:
    case At::graftPairs:
    case At::rigidityWeights:
    case At::deltas:
    case At::jointWeights:
    case At::localAxis:
    case At::uvs:
      return key == "values" ? At::table : At::none;
    case At::graft:
      return key == "vertex_pairs" ? At::graftPairs : At::none;
    case At::rigidity:
      return key == "weights" ? At::rigidityWeights : At::none;
    case At::mod:
      return key == "morph" ? At::morph : key == "skin" ? At::skin : At::none;
    case At::morph:
      return key == "deltas" ? At::deltas : At::none;
    case At::skin:
      return key == "joints" ? At::joints : At::none;
    case At::joint:
      if(key == "node_weights" || key == "scale_weights")
        return At::jointWeights;
      return key == "local_weights" ? At::local : At::none;
    case At::local:
      return (key == "x" || key == "y" || key == "z") ? At::localAxis : At::none;
    case At::uv:
      return key == "uvs" ? At::uvs : key == "polygon_vertex_indices" ? At::table : At::none;
    default:
      return At::none;
  }
}

At elementOfArray(At at)
{
  switch(at)
  {
    case At::geoArray:
      return At::geo;
    case At::modArray:
      return At::mod;
    case At::joints:
      return At::joint;
    case At::uvArray:
      return At::uv;
    default:
      return At::none;
  }
}

// The SAX handler: every event goes to the tree builder, except inside a table.
template <class Adapter>
class TableSax
{
public:
  using Delegate          = nlohmann::detail::json_sax_dom_parser<Json, Adapter>;
  using number_integer_t  = Json::number_integer_t;
  using number_unsigned_t = Json::number_unsigned_t;
  using number_float_t    = Json::number_float_t;
  using string_t          = Json::string_t;
  using binary_t          = Json::binary_t;

  explicit TableSax(Json& root)
      : m_dom(root, true)
  {
  }

  std::vector<Rows> tables;
  bool              fallback = false;  // a table held something other than rows of numbers: parse the tree way
  // a table's key came twice in one object: the tree keeps the last value, as Json::parse does, and
  // the first table is left unreferenced; parseDoc() then drops the tables the tree does not name
  bool duplicateTable = false;

  bool null()
  {
    return inTable() ? giveUp() : m_dom.null();
  }
  bool boolean(bool v)
  {
    return inTable() ? giveUp() : m_dom.boolean(v);
  }
  bool number_integer(number_integer_t v)
  {
    return inTable() ? cell(double(v)) : m_dom.number_integer(v);
  }
  bool number_unsigned(number_unsigned_t v)
  {
    return inTable() ? cell(double(v)) : m_dom.number_unsigned(v);
  }
  bool number_float(number_float_t v, const string_t& s)
  {
    return inTable() ? cell(v) : m_dom.number_float(v, s);
  }
  bool string(string_t& v)
  {
    return inTable() ? giveUp() : m_dom.string(v);
  }
  bool binary(binary_t& v)
  {
    return inTable() ? giveUp() : m_dom.binary(v);
  }

  bool start_object(std::size_t n)
  {
    if(inTable())
      return giveUp();
    m_frames.push_back({true, childAt(), At::none});
    return m_dom.start_object(n);
  }
  bool key(string_t& k)
  {
    Frame& f = m_frames.back();
    f.child  = childOfObject(f.at, k);
    if(f.hadTable && k == f.tableKey)
      duplicateTable = true;
    if(f.child == At::table)
      m_tableKey = k;
    return m_dom.key(k);
  }
  bool end_object()
  {
    m_frames.pop_back();
    return m_dom.end_object();
  }
  bool start_array(std::size_t n)
  {
    if(inTable())
    {
      if(m_rowDepth != 0)
        return giveUp();  // an array inside a row
      m_rowDepth = 1;
      return true;
    }
    const At at = childAt();
    if(at == At::table)
    {
      Frame& f   = m_frames.back();  // a table is always the value of a key, so under an object frame
      f.hadTable = true;
      f.tableKey = m_tableKey;
      m_table    = Rows{};
      m_inTable  = true;
      m_rowDepth = 0;
      return true;
    }
    m_frames.push_back({false, at, At::none});
    return m_dom.start_array(n);
  }
  bool end_array()
  {
    if(inTable())
    {
      if(m_rowDepth == 1)
      {
        m_rowDepth = 0;
        m_table.start.push_back(int32_t(m_table.data.size()));
        m_table.count++;
        return true;
      }
      // the table is complete: the tree gets its placeholder
      m_inTable = false;
      const std::size_t n = tables.size();
      tables.push_back(std::move(m_table));
      m_table = Rows{};
      string_t key = "$table";
      return m_dom.start_object(1) && m_dom.key(key) && m_dom.number_unsigned(n) && m_dom.end_object();
    }
    m_frames.pop_back();
    return m_dom.end_array();
  }
  bool parse_error(std::size_t position, const std::string& last, const nlohmann::detail::exception& ex)
  {
    return m_dom.parse_error(position, last, ex);
  }

private:
  struct Frame
  {
    bool     object;
    At       at;     // where this container is
    At       child;  // where the value of the key last seen goes (objects only)
    bool     hadTable = false;  // a table was started under this object, by the key tableKey
    string_t tableKey;
  };

  bool inTable() const { return m_inTable; }
  At   childAt() const
  {
    if(m_frames.empty())
      return At::root;
    const Frame& f = m_frames.back();
    return f.object ? f.child : elementOfArray(f.at);
  }
  bool cell(double v)
  {
    if(m_rowDepth != 1)
      return giveUp();  // a number directly in the table, not in a row
    m_table.data.push_back(v);
    return true;
  }
  bool giveUp()
  {
    fallback = true;
    return false;
  }

  Delegate           m_dom;
  std::vector<Frame> m_frames;
  string_t           m_tableKey;  // the key whose value is the table about to start
  bool               m_inTable  = false;
  int                m_rowDepth = 0;
  Rows               m_table;
};

// Keeps the tables the tree names, numbered in tree order, and renumbers the placeholders.
void compactTables(Json& node, std::vector<Rows>& from, std::vector<Rows>& to)
{
  if(node.is_object())
  {
    if(node.size() == 1 && node.contains("$table"))
    {
      Json& ref = node["$table"];
      to.push_back(std::move(from.at(ref.get<std::size_t>())));
      ref = to.size() - 1;
      return;
    }
    for(auto& [k, v] : node.items())
      compactTables(v, from, to);
  }
  else if(node.is_array())
    for(Json& v : node)
      compactTables(v, from, to);
}

}  // namespace

std::shared_ptr<Doc> parseDoc(std::string text)
{
  auto doc   = std::make_shared<Doc>();
  doc->bytes = text.size();
  using Adapter = decltype(nlohmann::detail::input_adapter(text));
  TableSax<Adapter> sax(doc->json);
  const bool ok = Json::sax_parse(text, &sax);
  if(!ok || sax.fallback)
    return parseDocDom(std::move(text));
  if(sax.duplicateTable)
    compactTables(doc->json, sax.tables, doc->tables);
  else
    doc->tables = std::move(sax.tables);
  for(const Rows& t : doc->tables)
    doc->bytes += t.data.size() * sizeof(double) + t.start.size() * sizeof(int32_t);
  return doc;
}

}  // namespace fv::dson
