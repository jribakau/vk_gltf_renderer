#include <figure/dson/morphs.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <regex>
#include <utility>

#include <figure/io/gzip.hpp>

namespace fv::dson {

namespace {

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

bool endsWith(const std::string& s, const char* suffix)
{
  const size_t n = std::char_traits<char>::length(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// "Genesis8Female:/data/...#x?value" -> "/data/...#x?value"
std::string stripNode(const std::string& url)
{
  const size_t colon = url.find(':');
  if(colon != std::string::npos && colon + 1 < url.size() && (url[colon + 1] == '/' || url[colon + 1] == '#'))
    return url.substr(colon + 1);
  return url;
}

// `prefer`: the meshes of the scene node the dial sits on, tried first at each step
const MeshRef* targetFor(const Json& mod, const std::string& file, const std::vector<MeshRef>& meshes, const std::vector<size_t>* prefer)
{
  const Json*       morph  = get(&mod, "morph");
  const std::string parent = getString(&mod, "parent");
  const size_t      hash   = parent.find('#');
  const std::string pid    = hash != std::string::npos ? decodeUri(parent.substr(hash + 1)) : "";
  std::string       pfile  = file;
  if(hash != std::string::npos && hash > 0)
  {
    pfile = decodeUri(parent.substr(0, hash));
    pfile.erase(0, std::min(pfile.find_first_not_of('/'), pfile.size()));
  }
  const std::string key = lower(pfile + "#" + pid);
  auto              byKey = [&](const std::vector<size_t>* list) -> const MeshRef* {
    if(list)
    {
      for(size_t i : *list)
        if(meshes[i].geometryKey == key)
          return &meshes[i];
      return nullptr;
    }
    for(const MeshRef& m : meshes)
      if(m.geometryKey == key)
        return &m;
    return nullptr;
  };
  if(const MeshRef* found = (prefer ? byKey(prefer) : nullptr))
    return found;
  if(const MeshRef* found = byKey(nullptr))
    return found;
  // parent is often the figure node, not its geometry: fall back to vertex count
  const Json* count = get(morph, "vertex_count");
  auto        byCount = [&](const std::vector<size_t>* list) -> const MeshRef* {
    const int n = count->get<int>();
    if(list)
    {
      for(size_t i : *list)
        if(meshes[i].geo->dsonVertexCount == n)
          return &meshes[i];
      return nullptr;
    }
    for(const MeshRef& m : meshes)
      if(m.geo->dsonVertexCount == n)
        return &m;
    return nullptr;
  };
  // a stated vertex count that matches nothing is a morph for a different mesh, not this one.
  // HD morphs (a character's full-body shape among them) state -1: that says nothing.
  if(count && count->is_number() && count->get<double>() > 0)
  {
    if(const MeshRef* m = (prefer ? byCount(prefer) : nullptr))
      return m;
    return byCount(nullptr);
  }
  if(prefer && !prefer->empty())
    return &meshes[(*prefer)[0]];
  return meshes.empty() ? nullptr : &meshes[0];
}

}  // namespace

bool hiddenMorph(const std::string& name)
{
  static const std::regex re(R"(^(pjcm|pbmc|pcmc|jcm|mcm|pmcm|facs_jnt|facs_cbs|body_cbs|head_cbs|ctrl_?jcm)|_hd\d?$|HDLv\d)", std::regex::icase);
  return std::regex_search(name, re);
}

namespace {

// Reads the first modifier's fields as they go by, and stops at what comes after them.
class MorphInfoSax : public nlohmann::json_sax<Json>
{
public:
  MorphInfo info;
  bool      found = false;

  bool null() override { return true; }
  bool boolean(bool v) override
  {
    if(in("channel") && m_key == "visible" && !v)
      info.helper = true;
    return true;
  }
  bool number_integer(number_integer_t v) override { return number(double(v)); }
  bool number_unsigned(number_unsigned_t v) override { return number(double(v)); }
  bool number_float(number_float_t v, const string_t&) override { return number(v); }
  bool string(string_t& v) override
  {
    if(inModifier())
    {
      if(m_key == "id")
        info.id = v;
      else if(m_key == "region")
        info.region = v;
      else if(m_key == "group")
        info.path = v;
    }
    else if(in("presentation") && m_key == "type")
      info.kind = v;
    else if(in("channel"))
    {
      if(m_key == "label")
        info.label = v;
      else if(m_key == "type" && v == "alias")
        info.alias = true;
    }
    return true;
  }
  bool binary(binary_t&) override { return true; }
  bool key(string_t& v) override
  {
    m_key = v;
    // the rest of the modifier is what makes these files large
    return !(inModifier() && (v == "formulas" || v == "morph"));
  }
  bool start_object(std::size_t) override { return push(); }
  bool end_object() override { return pop(); }
  bool start_array(std::size_t) override { return push(); }
  bool end_array() override { return pop(); }
  bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override { return false; }

private:
  // the keys down to here: "", "modifier_library", "" is inside a modifier
  std::vector<std::string> m_stack;
  std::string              m_key;

  bool inModifier() const { return m_stack.size() == 3 && m_stack[1] == "modifier_library"; }
  bool in(const char* field) const { return m_stack.size() == 4 && m_stack[1] == "modifier_library" && m_stack[3] == field; }
  bool number(double v)
  {
    if(in("channel") && m_key == "min")
      info.min = v;
    else if(in("channel") && m_key == "max")
      info.max = v;
    return true;
  }
  bool push()
  {
    m_stack.push_back(std::exchange(m_key, {}));
    if(inModifier())
      found = true;
    return true;
  }
  bool pop()
  {
    const bool first = inModifier();
    m_stack.pop_back();
    // one modifier to a morph file
    return !first;
  }
};

}  // namespace

std::optional<MorphInfo> morphInfoFromText(std::string_view head)
{
  MorphInfoSax sax;
  Json::sax_parse(head.begin(), head.end(), &sax, Json::input_format_t::json, false);
  if(!sax.found)
    return std::nullopt;
  if(sax.info.alias || sax.info.kind == "Modifier/Corrective" || sax.info.path.rfind("/Hidden", 0) == 0)
    sax.info.helper = true;
  return sax.info;
}

std::optional<MorphInfo> morphInfo(const std::filesystem::path& file)
{
  // a morph's fields end within a kilobyte of the modifier's start, which follows a short asset_info
  constexpr size_t kHead = 8192;
  try
  {
    return morphInfoFromText(readHead(file, kHead));
  }
  catch(const std::exception&)
  {
    return std::nullopt;
  }
}

std::string groupFromPath(const std::string& file)
{
  std::vector<std::string> parts;
  size_t                   start = 0;
  for(;;)
  {
    const size_t slash = file.find('/', start);
    parts.push_back(file.substr(start, slash == std::string::npos ? std::string::npos : slash - start));
    if(slash == std::string::npos)
      break;
    start = slash + 1;
  }
  for(size_t i = 0; i < parts.size(); i++)
    if(lower(parts[i]) == "morphs")
      return parts.size() > i + 2 ? parts[i + 2] : "Morphs";
  return "Morphs";
}

std::vector<MorphDef> morphsFromDoc(const Doc& doc, const std::string& file, const std::vector<MeshRef>& meshes, const std::vector<size_t>* prefer)
{
  struct Found
  {
    const Json*    mod;
    const MeshRef* fm;
    mesh::CageDeltas deltas;
  };
  std::vector<Found> found;
  const Json*        mods = get(&doc.json, "modifier_library");
  if(!mods || !mods->is_array())
    return {};
  for(const Json& mod : *mods)
  {
    const Json* values = get(get(get(&mod, "morph"), "deltas"), "values");
    const auto  deltas = doc.rows(values);
    if(!deltas || !deltas->count)
      continue;
    const MeshRef* fm = targetFor(mod, file, meshes, prefer);
    if(!fm)
      continue;
    Found f{&mod, fm, {}};
    f.deltas.verts.resize(size_t(deltas->count));
    f.deltas.deltas.resize(size_t(deltas->count) * 3);
    for(int i = 0; i < deltas->count; i++)
    {
      f.deltas.verts[size_t(i)]          = toInt32(deltas->cell(i, 0));
      f.deltas.deltas[size_t(i) * 3]     = float(deltas->cell(i, 1));
      f.deltas.deltas[size_t(i) * 3 + 1] = float(deltas->cell(i, 2));
      f.deltas.deltas[size_t(i) * 3 + 2] = float(deltas->cell(i, 3));
    }
    found.push_back(std::move(f));
  }
  std::vector<MorphDef> out;
  for(const Found& f : found)
  {
    auto target = std::make_shared<mesh::RenderTarget>(mesh::target(*f.fm->geo, f.deltas));
    if(!target->moves())
      continue;
    const Json* ch = get(f.mod, "channel");
    MorphDef    def;
    def.id = getString(f.mod, "id");
    MorphPart part;
    part.mesh   = f.fm->mesh;
    part.target = std::move(target);
    // the channel's label, else the modifier's label, else its name
    for(const Json* v : {get(ch, "label"), get(f.mod, "label"), get(f.mod, "name")})
      if(v && v->is_string())
      {
        part.label    = v->get<std::string>();
        part.hasLabel = true;
        break;
      }
    const Json* clamped = get(ch, "clamped");
    const bool  limited = !(clamped && clamped->is_boolean() && !clamped->get<bool>());
    const Json* lo      = get(ch, "min");
    const Json* hi      = get(ch, "max");
    part.min = lo && lo->is_number() && limited ? std::max(-1.0, lo->get<double>()) : 0;
    part.max = hi && hi->is_number() && limited ? std::min(2.0, hi->get<double>()) : 1;
    // the mesh's shells take it too, remade on their own polygons
    def.parts.push_back(part);
    for(const MeshRef::Shell& s : f.fm->shells)
    {
      part.mesh   = s.mesh;
      part.target = std::make_shared<mesh::RenderTarget>(mesh::target(*s.geo, f.deltas));
      def.parts.push_back(part);
    }
    out.push_back(std::move(def));
  }
  return out;
}

namespace {

TargetPtr scaled(const TargetPtr& t, double k)
{
  if(k == 1)
    return t;
  auto out         = std::make_shared<mesh::RenderTarget>();
  out->cageIndices = t->cageIndices;
  out->cageDeltas  = t->cageDeltas;
  for(float& v : out->cageDeltas)
    v = float(v * k);
  // a target made by hand has its drawn deltas alone
  if(const mesh::RenderTarget::Drawn* d = t->drawnIfMade(); d && t->cageDeltas.empty())
  {
    mesh::RenderTarget::Drawn s = *d;
    for(float& v : s.deltas)
      v = float(v * k);
    out->setDrawn(std::move(s));
  }
  return out;
}

// the morph whose value a formula writes, or none for other outputs (bone centres, rotations...)
std::optional<Ref> formulaTarget(const Json* output, const std::string& currentFile)
{
  if(!output || !output->is_string() || !endsWith(output->get<std::string>(), "?value"))
    return std::nullopt;
  auto ref = parseRef(stripNode(output->get<std::string>()), currentFile);
  if(!ref || ref->id.empty())
    return std::nullopt;
  return ref;
}

// k when the operations compute `<id>.value * k` (or just `<id>.value`), otherwise none
std::optional<double> linearFactor(const Json* ops, const std::string& id)
{
  if(!ops || !ops->is_array())
    return std::nullopt;
  auto isInput = [&](const Json& o) {
    if(getString(&o, "op") != "push")
      return false;
    const Json* url = get(&o, "url");
    if(!url || !url->is_string() || !endsWith(url->get<std::string>(), "?value"))
      return false;
    auto ref = parseRef(stripNode(url->get<std::string>()), "");
    return ref && ref->id == id;
  };
  auto isConst = [](const Json& o) {
    const Json* v = get(&o, "val");
    return getString(&o, "op") == "push" && v && v->is_number();
  };
  if(ops->size() == 1 && isInput((*ops)[0]))
    return 1.0;
  if(ops->size() == 3 && getString(&(*ops)[2], "op") == "mult")
  {
    if(isInput((*ops)[0]) && isConst((*ops)[1]))
      return (*ops)[1]["val"].get<double>();
    if(isConst((*ops)[0]) && isInput((*ops)[1]))
      return (*ops)[0]["val"].get<double>();
  }
  return std::nullopt;
}

}  // namespace

std::vector<DrivenMorph> drivenMorphs(Store& store, const std::string& file, const std::string& id, double weight, int depth)
{
  auto        doc  = store.load(file);
  const Json* mods = get(&doc->json, "modifier_library");
  if(!mods || !mods->is_array())
    return {};
  const Json* mod = nullptr;
  for(const Json& m : *mods)
    if(getString(&m, "id") == id)
    {
      mod = &m;
      break;
    }
  if(!mod)
    return {};
  std::vector<DrivenMorph> out;
  if(doc->hasRows(get(get(get(mod, "morph"), "deltas"), "values")))
    out.push_back({file, id, weight});
  if(depth < 4)
  {
    // the driver's file usually holds its formulas, on the driver itself or on a sibling modifier
    for(const Json& m : *mods)
    {
      const Json* formulas = get(&m, "formulas");
      if(!formulas || !formulas->is_array())
        continue;
      for(const Json& f : *formulas)
      {
        const Json* stage = get(&f, "stage");
        if(stage && stage->is_string() && !stage->get<std::string>().empty() && stage->get<std::string>() != "sum")
          continue;
        const auto target = formulaTarget(get(&f, "output"), file);
        const auto k      = linearFactor(get(&f, "operations"), id);
        if(!target || !k || *k == 0 || (target->file == file && target->id == id))
          continue;
        // a driven morph that isn't installed: the store records it and the loader reports it
        try
        {
          for(DrivenMorph& d : drivenMorphs(store, target->file, target->id, weight * *k, depth + 1))
            out.push_back(std::move(d));
        }
        catch(const std::exception&)
        {
        }
      }
    }
  }
  std::vector<DrivenMorph>             merged;
  std::map<std::string, size_t>        at;
  for(DrivenMorph& d : out)
  {
    const std::string key = lower(d.file + "#" + d.id);
    if(auto it = at.find(key); it != at.end())
      merged[it->second].weight += d.weight;
    else
    {
      at[key] = merged.size();
      merged.push_back(std::move(d));
    }
  }
  return merged;
}

std::vector<MorphPart> loadMorphFile(Store& store, const std::string& file, const std::vector<MeshRef>& meshes)
{
  auto                   doc = store.load(file);
  std::vector<MorphPart> own;
  for(MorphDef& m : morphsFromDoc(*doc, file, meshes))
    for(MorphPart& p : m.parts)
      own.push_back(std::move(p));
  if(!own.empty())
    return own;
  std::vector<MorphPart> out;
  const Json*            mods = get(&doc->json, "modifier_library");
  if(!mods || !mods->is_array())
    return out;
  for(const Json& mod : *mods)
  {
    std::string label = getString(get(&mod, "channel"), "label");
    if(label.empty())
      label = getString(&mod, "label");
    for(const DrivenMorph& d : drivenMorphs(store, file, getString(&mod, "id")))
    {
      auto ddoc = store.load(d.file);
      for(MorphDef& m : morphsFromDoc(*ddoc, d.file, meshes))
      {
        if(m.id != d.id)
          continue;
        for(MorphPart& p : m.parts)
        {
          MorphPart q = p;
          if(!label.empty())
          {
            q.label    = label;
            q.hasLabel = true;
          }
          q.target = scaled(p.target, d.weight);
          out.push_back(std::move(q));
        }
      }
    }
  }
  return out;
}

}  // namespace fv::dson
