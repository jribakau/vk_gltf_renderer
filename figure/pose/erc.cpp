#include <figure/pose/erc.hpp>

#include <algorithm>
#include <cmath>

namespace fv::pose {

using dson::Json;

namespace {

// "lShldr:/data/...#lShldr?rotation/z" -> "/data/...#lShldr?rotation/z": the node prefix ERC urls carry
std::string stripNode(const std::string& url)
{
  const size_t colon = url.find(':');
  if(colon != std::string::npos && colon + 1 < url.size() && (url[colon + 1] == '/' || url[colon + 1] == '#'))
    return url.substr(colon + 1);
  return url;
}

bool boneProp(const std::string& prop, std::string& name, std::string& axis)
{
  static const char* props[] = {"rotation", "translation", "scale", "center_point", "end_point", "orientation"};
  const size_t       slash   = prop.find('/');
  if(slash == std::string::npos)
    return false;
  name = prop.substr(0, slash);
  axis = prop.substr(slash + 1);
  if(std::none_of(std::begin(props), std::end(props), [&](const char* p) { return name == p; }))
    return false;
  return axis == "x" || axis == "y" || axis == "z" || axis == "general";
}

KeyInfo parseKey(const std::string& key)
{
  KeyInfo k;
  if(key.rfind("m:", 0) == 0)
  {
    k.morph = true;
    k.id    = key.substr(2);
    return k;
  }
  // b:<bone>:<prop>/<axis>
  const size_t at = key.rfind(':');
  if(key.rfind("b:", 0) == 0 && at != std::string::npos && at >= 2)
  {
    k.id = key.substr(2, at - 2);
    const std::string rest  = key.substr(at + 1);
    const size_t      slash = rest.find('/');
    k.prop                  = rest.substr(0, slash);
    k.axis                  = slash == std::string::npos ? "" : rest.substr(slash + 1);
  }
  return k;
}

}  // namespace

std::optional<UrlKey> keyOf(const std::string& url, const std::string& file)
{
  const size_t q = url.find('?');
  if(q == std::string::npos)
    return std::nullopt;
  std::string prop = url.substr(q + 1);
  if(prop.size() >= 6 && prop.compare(prop.size() - 6, 6, "/value") == 0)
    prop.resize(prop.size() - 6);
  const std::string target = stripNode(url.substr(0, q));
  const auto        ref    = dson::parseRef(target, file);
  if(!ref || ref->id.empty())
    return std::nullopt;
  UrlKey out;
  if(!target.empty() && target[0] == '/')
    out.file = ref->file;
  if(prop == "value")
  {
    out.key = "m:" + ref->id;
    return out;
  }
  std::string name, axis;
  if(!boneProp(prop, name, axis))
    return std::nullopt;
  out.key = "b:" + ref->id + ":" + prop;
  return out;
}

KeyId ErcGraph::intern(const std::string& key)
{
  auto [it, added] = m_ids.try_emplace(key, KeyId(m_keys.size()));
  if(added)
  {
    Key k;
    k.name = key;
    k.info = parseKey(key);
    m_keys.push_back(std::move(k));
  }
  return it->second;
}

KeyId ErcGraph::find(const std::string& key) const
{
  auto it = m_ids.find(key);
  return it == m_ids.end() ? -1 : it->second;
}

const ChannelMeta* ErcGraph::meta(const std::string& id) const
{
  auto it = m_metas.find(id);
  return it == m_metas.end() ? nullptr : &it->second;
}

const std::string* ErcGraph::fileOf(const std::string& id) const
{
  auto it = m_files.find(id);
  return it == m_files.end() ? nullptr : &it->second;
}

void ErcGraph::addDoc(const Json& doc, const std::string& file)
{
  std::string fileKey = file;
  std::transform(fileKey.begin(), fileKey.end(), fileKey.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if(!m_docs.insert(fileKey).second)
    return;
  const Json* mods = dson::get(&doc, "modifier_library");
  if(mods && mods->is_array())
    for(const Json& mod : *mods)
    {
      const Json* idj = dson::get(&mod, "id");
      if(!idj || !idj->is_string())
        continue;
      const std::string id = idj->get<std::string>();
      m_files[id]          = file;
      if(const Json* ch = dson::get(&mod, "channel"); ch && ch->is_object())
      {
        const Json* raw = dson::get(ch, "current_value");
        if(!raw || raw->is_null())
          raw = dson::get(ch, "value");
        ChannelMeta meta;
        if(raw && raw->is_boolean())
          meta.base = raw->get<bool>() ? 1 : 0;
        else if(raw && raw->is_number())
          meta.base = raw->get<double>();
        const Json* clamped = dson::get(ch, "clamped");
        const Json* lo      = dson::get(ch, "min");
        const Json* hi      = dson::get(ch, "max");
        // truthy: true, or any non-zero number
        const bool truthy = clamped && ((clamped->is_boolean() && clamped->get<bool>()) || (clamped->is_number() && clamped->get<double>() != 0));
        if(truthy && lo && lo->is_number() && hi && hi->is_number())
        {
          meta.clamped = true;
          meta.lo      = lo->get<double>();
          meta.hi      = hi->get<double>();
        }
        m_metas[id] = meta;
      }
      if(const Json* formulas = dson::get(&mod, "formulas"); formulas && formulas->is_array())
        for(const Json& f : *formulas)
          addFormula(f, file);
    }
  version++;
}

void ErcGraph::addFormula(const Json& f, const std::string& file)
{
  const Json* ops = dson::get(&f, "operations");
  if(!ops || !ops->is_array())
    return;
  const Json* stageJ = dson::get(&f, "stage");
  const bool  hasStage = stageJ && !stageJ->is_null() && !(stageJ->is_string() && stageJ->get<std::string>().empty());
  const std::string stage = stageJ && stageJ->is_string() ? stageJ->get<std::string>() : "";
  bool              mult  = false;
  if(stage == "mult")
    mult = true;
  else if(hasStage && stage != "sum")
    return;
  const Json* outputJ = dson::get(&f, "output");
  if(!outputJ || !outputJ->is_string())
    return;
  const auto out = keyOf(outputJ->get<std::string>(), file);
  if(!out)
    return;
  if(out->file && out->key.rfind("m:", 0) == 0)
    m_files[out->key.substr(2)] = *out->file;

  Compiled c;
  c.mult = mult;
  std::unordered_map<std::string, int> inputIndex;
  c.ops = compileOps(*ops, [&](const std::string& url) {
    auto [it, added] = inputIndex.try_emplace(url, int(c.inputs.size()));
    if(added)
    {
      const auto k = keyOf(url, file);
      c.inputs.push_back(k ? intern(k->key) : -1);
      if(k && k->file && k->key.rfind("m:", 0) == 0)
        m_files[k->key.substr(2)] = *k->file;
    }
    return it->second;
  });
  c.output = intern(out->key);
  Key& key = m_keys[size_t(c.output)];
  if(key.formulas.empty())
    m_outputs.push_back(c.output);
  key.formulas.push_back(std::move(c));
}

std::optional<double> ErcGraph::Evaluator::operator()(KeyId key)
{
  if(size_t(key) >= m_state.size())
  {
    m_state.resize(m_graph.m_keys.size(), eNone);
    m_value.resize(m_graph.m_keys.size(), 0);
  }
  if(m_state[size_t(key)] == eDone)
    return m_value[size_t(key)];
  if(m_state[size_t(key)] == eUnknown)
    return std::nullopt;
  const Key&         k       = m_graph.m_keys[size_t(key)];
  const ChannelMeta* meta    = k.info.morph ? m_graph.meta(k.info.id) : nullptr;
  std::optional<double> v    = m_base(k.info);
  if(!v && meta)
    v = meta->base;
  if(!v && k.formulas.empty())
  {
    m_state[size_t(key)] = eUnknown;
    return std::nullopt;
  }
  // a cycle reads the property as its base
  if(m_state[size_t(key)] == eBusy)
    return v.value_or(0);
  m_state[size_t(key)] = eBusy;
  double                out  = v.value_or(0);
  std::optional<double> mult;
  // reading inputs may intern keys and grow m_keys: formulas are read through the index
  const size_t count = k.formulas.size();
  for(size_t i = 0; i < count; i++)
  {
    const Compiled& f = m_graph.m_keys[size_t(key)].formulas[i];
    const auto      r = evaluate(f.ops, [&](int input) -> std::optional<double> {
      const KeyId in = m_graph.m_keys[size_t(key)].formulas[i].inputs[size_t(input)];
      return in >= 0 ? (*this)(in) : std::nullopt;
    });
    if(!r || !std::isfinite(*r))
      continue;
    if(!f.mult)
      out += *r;
    else
      mult = mult.value_or(1) * *r;
  }
  if(mult)
    out *= *mult;
  if(meta && meta->clamped)
    out = std::min(meta->hi, std::max(meta->lo, out));
  m_state[size_t(key)] = eDone;
  m_value[size_t(key)] = out;
  return out;
}

}  // namespace fv::pose
