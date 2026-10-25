#include <figure/dson/store.hpp>

#include <algorithm>
#include <cctype>
#include <list>
#include <unordered_map>

#include <figure/io/gzip.hpp>
#include <figure/io/timing.hpp>

namespace fv::dson {

namespace {

std::string lower(std::string_view s)
{
  std::string out(s);
  for(char& c : out)
    c = char(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

bool endsWith(std::string_view s, std::string_view suffix)
{
  return s.size() >= suffix.size() && lower(s.substr(s.size() - suffix.size())) == suffix;
}

// Packs an array of number arrays; none when it is anything else, which is left as it is.
std::optional<Rows> pack(const Json& table)
{
  if(!table.is_array())
    return std::nullopt;
  Rows   out;
  size_t n = 0;
  for(const Json& row : table)
  {
    if(!row.is_array())
      return std::nullopt;
    for(const Json& x : row)
      if(!x.is_number())
        return std::nullopt;
    n += row.size();
  }
  out.count = int(table.size());
  out.start.assign(table.size() + 1, 0);
  out.data.reserve(n);
  int i = 0;
  for(const Json& row : table)
  {
    for(const Json& x : row)
      out.data.push_back(x.get<double>());
    out.start[++i] = int32_t(out.data.size());
  }
  return out;
}

// Calls `visit` for every place a DSON document keeps a bulk table: geometry, morphs, skins, UV
// sets, graft seams and rigidity weights. Only these known places, so tables other code reads as
// plain arrays (an animation's keys, a colour) are left alone.
template <class Visit>
void eachTable(Json& doc, Visit visit)
{
  auto at = [&](Json* holder, const char* key) {
    if(holder && holder->is_object() && holder->contains(key))
      visit((*holder)[key]);
  };
  auto member = [](Json& j, const char* key) -> Json* {
    if(!j.is_object() || !j.contains(key))
      return nullptr;
    return &j[key];
  };
  if(Json* geos = member(doc, "geometry_library"); geos && geos->is_array())
    for(Json& geo : *geos)
    {
      at(member(geo, "vertices"), "values");
      at(member(geo, "polylist"), "values");
      at(member(geo, "polyline_list"), "values");
      if(Json* graft = member(geo, "graft"))
        at(member(*graft, "vertex_pairs"), "values");
      if(Json* rigidity = member(geo, "rigidity"))
        at(member(*rigidity, "weights"), "values");
    }
  if(Json* mods = member(doc, "modifier_library"); mods && mods->is_array())
    for(Json& mod : *mods)
    {
      if(Json* morph = member(mod, "morph"))
        at(member(*morph, "deltas"), "values");
      if(Json* skin = member(mod, "skin"))
        if(Json* joints = member(*skin, "joints"); joints && joints->is_array())
          for(Json& j : *joints)
          {
            at(member(j, "node_weights"), "values");
            at(member(j, "scale_weights"), "values");
            if(Json* local = member(j, "local_weights"))
              for(const char* axis : {"x", "y", "z"})
                at(member(*local, axis), "values");
          }
    }
  if(Json* uvs = member(doc, "uv_set_library"); uvs && uvs->is_array())
    for(Json& uv : *uvs)
    {
      at(member(uv, "uvs"), "values");
      at(&uv, "polygon_vertex_indices");
    }
}

int hexValue(char c)
{
  if(c >= '0' && c <= '9')
    return c - '0';
  if(c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if(c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

// The process-wide cache of parsed .dsf files (src/lib/loaders/dson/parse.ts: `shared`).
struct SharedCache
{
  static constexpr size_t kLimit = size_t(256) << 20;

  std::mutex                                                                     mutex;
  std::list<std::string>                                                         lru;  // most recently used last
  std::unordered_map<std::string, std::pair<std::shared_ptr<const Doc>, std::list<std::string>::iterator>> docs;
  std::unordered_map<std::string, std::shared_future<std::shared_ptr<const Doc>>> pending;
  size_t                                                                         bytes = 0;
  uint64_t                                                                       epoch = 0;

  void trim()
  {
    while(bytes > kLimit && !lru.empty())
    {
      auto it = docs.find(lru.front());
      bytes -= it->second.first->bytes;
      docs.erase(it);
      lru.pop_front();
    }
  }
};

SharedCache& shared()
{
  static SharedCache cache;
  return cache;
}

}  // namespace

//--------------------------------------------------------------------------------------------------

std::shared_ptr<const Rows> Doc::rows(const Json* node) const
{
  static const auto empty = std::make_shared<const Rows>();
  if(!node)
    return empty;
  if(node->is_object() && node->contains("$table"))
  {
    const size_t n = (*node)["$table"].get<size_t>();
    if(n < tables.size())
      return std::shared_ptr<const Rows>(std::shared_ptr<const Rows>{}, &tables[n]);
    return empty;
  }
  auto packed = pack(*node);
  return packed ? std::make_shared<const Rows>(std::move(*packed)) : empty;
}

bool Doc::hasRows(const Json* node) const
{
  if(!node)
    return false;
  if(node->is_object() && node->contains("$table"))
  {
    const size_t n = (*node)["$table"].get<size_t>();
    return n < tables.size() && tables[n].count > 0;
  }
  return node->is_array() && !node->empty();
}

std::shared_ptr<Doc> parseDocDom(std::string text)
{
  auto doc   = std::make_shared<Doc>();
  doc->bytes = text.size();
  doc->json  = Json::parse(text);
  eachTable(doc->json, [&](Json& table) {
    auto packed = pack(table);
    if(!packed)
      return;
    doc->bytes += packed->data.size() * sizeof(double) + packed->start.size() * sizeof(int32_t);
    table = Json{{"$table", doc->tables.size()}};
    doc->tables.push_back(std::move(*packed));
  });
  return doc;
}

std::string decodeUri(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for(size_t i = 0; i < s.size(); i++)
  {
    if(s[i] != '%')
    {
      out += s[i];
      continue;
    }
    if(i + 2 >= s.size())
      return std::string(s);
    const int hi = hexValue(s[i + 1]), lo = hexValue(s[i + 2]);
    if(hi < 0 || lo < 0)
      return std::string(s);
    out += char(hi * 16 + lo);
    i += 2;
  }
  return out;
}

std::optional<Ref> parseRef(std::string_view uri, std::string_view currentFile)
{
  if(uri.empty())
    return std::nullopt;
  // name://, file:// and other schemes are scene-local
  if(uri[0] != '/')
  {
    size_t i = 0;
    while(i < uri.size() && std::isalpha(static_cast<unsigned char>(uri[i])))
      i++;
    if(i > 0 && i < uri.size() && uri[i] == ':')
      return std::nullopt;
  }
  const size_t     hash     = uri.find('#');
  std::string_view pathPart = hash == std::string_view::npos ? uri : uri.substr(0, hash);
  std::string_view idPart   = hash == std::string_view::npos ? std::string_view{} : uri.substr(hash + 1);
  Ref              ref;
  if(!pathPart.empty())
  {
    ref.file      = decodeUri(pathPart);
    const auto at = ref.file.find_first_not_of('/');
    ref.file      = at == std::string::npos ? std::string{} : ref.file.substr(at);
  }
  else
    ref.file = std::string(currentFile);
  // ids can carry a property path after '?', e.g. "#Genesis9:?value"
  ref.id = decodeUri(idPart);
  ref.id = ref.id.substr(0, ref.id.find('?'));
  return ref;
}

std::optional<std::filesystem::path> Library::find(std::string_view rel) const
{
  std::error_code ec;
  // library paths are UTF-8 with forward slashes
  const std::filesystem::path relPath(std::u8string(reinterpret_cast<const char8_t*>(rel.data()), rel.size()));
  for(const auto& root : m_roots)
  {
    const auto p = root / relPath;
    if(std::filesystem::is_regular_file(p, ec))
      return p;
  }
  return std::nullopt;
}

//--------------------------------------------------------------------------------------------------

Store::Store(std::shared_ptr<const Library> library)
    : m_library(std::move(library))
{
}

std::shared_ptr<const Doc> Store::fetch(const std::string& file)
{
  const auto path = m_library->find(file);
  if(!path)
  {
    {
      std::lock_guard lock(m_mutex);
      if(std::find(m_missing.begin(), m_missing.end(), file) == m_missing.end())
        m_missing.push_back(file);
    }
    throw MissingFileError(file);
  }
  // a large file (strand hair runs to tens of MB of JSON) is timed: reading and parsing apart
  std::error_code ec;
  if(!timing::enabled() || std::filesystem::file_size(*path, ec) < (2u << 20) || ec)
    return parseDoc(readMaybeCompressed(*path));
  std::string text;
  {
    timing::Scope t("read " + file);
    text = readMaybeCompressed(*path);
  }
  timing::Scope t("parse " + std::to_string(text.size() >> 20) + " MB");
  return parseDoc(std::move(text));
}

std::shared_ptr<const Doc> Store::loadShared(const std::string& file)
{
  const auto path = m_library->find(file);
  if(!path)
    return fetch(file);  // reports it missing
  // keyed by where it is on disk: another library with the same relative path is another file
  const std::u8string where = path->generic_u8string();
  const std::string   key   = lower(std::string(where.begin(), where.end()));
  SharedCache&        s     = shared();
  std::optional<std::promise<std::shared_ptr<const Doc>>> promise;
  std::shared_future<std::shared_ptr<const Doc>>          future;
  uint64_t                                                epoch = 0;
  {
    std::lock_guard lock(s.mutex);
    if(auto hit = s.docs.find(key); hit != s.docs.end())
    {
      // most recently used last
      s.lru.splice(s.lru.end(), s.lru, hit->second.second);
      return hit->second.first;
    }
    if(auto p = s.pending.find(key); p != s.pending.end())
      future = p->second;
    else
    {
      promise.emplace();
      future         = promise->get_future().share();
      s.pending[key] = future;
      epoch          = s.epoch;
    }
  }
  if(promise)
  {
    std::shared_ptr<const Doc> doc;
    try
    {
      doc = fetch(file);
      promise->set_value(doc);
    }
    catch(...)
    {
      promise->set_exception(std::current_exception());
    }
    std::lock_guard lock(s.mutex);
    s.pending.erase(key);
    // not if the library changed while it loaded, or the budget has no room for it
    if(doc && epoch == s.epoch && doc->bytes <= SharedCache::kLimit && !s.docs.count(key))
    {
      s.lru.push_back(key);
      s.docs[key] = {doc, std::prev(s.lru.end())};
      s.bytes += doc->bytes;
      s.trim();
    }
  }
  return future.get();
}

std::shared_ptr<const Doc> Store::load(const std::string& file)
{
  const std::string                                       key = lower(file);
  std::optional<std::promise<std::shared_ptr<const Doc>>> promise;
  std::shared_future<std::shared_ptr<const Doc>>          future;
  {
    std::lock_guard lock(m_mutex);
    if(auto it = m_cache.find(key); it != m_cache.end())
      future = it->second;
    else
    {
      promise.emplace();
      future       = promise->get_future().share();
      m_cache[key] = future;
    }
  }
  // the first caller reads the file; others wait for it. A failure stays cached, as a rejected
  // promise does, so a missing file is looked for once.
  if(promise)
  {
    try
    {
      promise->set_value(endsWith(file, ".dsf") ? loadShared(file) : fetch(file));
    }
    catch(...)
    {
      promise->set_exception(std::current_exception());
    }
  }
  return future.get();
}

std::optional<Store::Resolved> Store::resolve(const Ref& ref, const char* preferLibrary)
{
  auto doc = load(ref.file);
  std::vector<std::string> libs;
  if(preferLibrary)
    libs.emplace_back(preferLibrary);
  else
    for(auto it = doc->json.begin(); it != doc->json.end(); ++it)
      if(endsWith(it.key(), "_library"))
        libs.push_back(it.key());
  for(const std::string& lib : libs)
  {
    const Json* entry = nullptr;
    {
      std::lock_guard lock(m_mutex);
      auto [it, made] = m_index.try_emplace({doc.get(), lib});
      auto& byId      = it->second;
      if(made)
      {
        const Json* list = get(&doc->json, lib);
        if(list && list->is_array())
          for(const Json& e : *list)
          {
            const Json* id = get(&e, "id");
            if(id && id->is_string())
              byId.try_emplace(id->get<std::string>(), &e);
          }
      }
      if(auto found = byId.find(ref.id); found != byId.end())
        entry = found->second;
    }
    if(entry)
      return Resolved{doc, entry, ref.file};
  }
  if(preferLibrary)
    return resolve(ref, nullptr);
  return std::nullopt;
}

std::vector<std::string> Store::missing() const
{
  std::lock_guard lock(m_mutex);
  return m_missing;
}

void Store::clearShared()
{
  SharedCache&    s = shared();
  std::lock_guard lock(s.mutex);
  s.docs.clear();
  s.lru.clear();
  s.pending.clear();
  s.bytes = 0;
  s.epoch++;
}

//--------------------------------------------------------------------------------------------------

const Json* get(const Json* obj, std::string_view key)
{
  if(!obj || !obj->is_object())
    return nullptr;
  auto it = obj->find(key);
  return it == obj->end() ? nullptr : &*it;
}

std::string getString(const Json* obj, std::string_view key, std::string fallback)
{
  const Json* v = get(obj, key);
  return v && v->is_string() ? v->get<std::string>() : fallback;
}

double channelValue(const Json* ch, double fallback)
{
  if(!ch || ch->is_null())
    return fallback;
  const Json* inner = get(ch, "channel");
  const Json* c     = inner && !inner->is_null() ? inner : ch;
  const Json* v     = get(c, "current_value");
  if(!v || v->is_null())
    v = get(c, "value");
  return v && v->is_number() ? v->get<double>() : fallback;
}

std::array<double, 3> vec3Channel(const Json* arr, std::array<double, 3> fallback)
{
  if(!arr || !arr->is_array())
    return fallback;
  std::array<double, 3> out = fallback;
  const char*           axes[3] = {"x", "y", "z"};
  for(int i = 0; i < 3; i++)
  {
    for(const Json& e : *arr)
    {
      const Json* id = get(&e, "id");
      if(!id || !id->is_string() || id->get<std::string>() != axes[i])
        continue;
      const Json* v = get(&e, "current_value");
      if(!v || v->is_null())
        v = get(&e, "value");
      out[i] = v && v->is_number() ? v->get<double>() : fallback[i];
      break;
    }
  }
  return out;
}

int32_t toInt32(double v)
{
  if(!std::isfinite(v))
    return 0;
  const double t = std::trunc(v);
  const double m = std::fmod(t, 4294967296.0);
  const double u = m < 0 ? m + 4294967296.0 : m;
  return int32_t(uint32_t(u));
}

}  // namespace fv::dson
