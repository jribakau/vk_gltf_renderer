#include <figure/morph/morph_set.hpp>

#include <algorithm>
#include <cmath>
#include <regex>

#include <figure/jobs/pool.hpp>

namespace fv::morph {

namespace {

// Incremental flushes before the buffer is rebuilt from the base positions, so rounding can't build up.
constexpr int kRebuildEvery = 400;
// How long a lazy file waits for others before the driver runs for it.
constexpr auto kDriverDelay = std::chrono::milliseconds(100);

std::string prettyLabel(const std::string& name)
{
  static const std::regex prefix(R"(^(blendShape\d*\.|morph_|Genesis\d*(Female|Male)?__?))", std::regex::icase);
  static const std::regex underscores("_+");
  static const std::regex camel("([a-z])([A-Z])");
  std::string             s = std::regex_replace(name, prefix, "");
  s                         = std::regex_replace(s, underscores, " ");
  s                         = std::regex_replace(s, camel, "$1 $2");
  const size_t a            = s.find_first_not_of(" \t");
  const size_t b            = s.find_last_not_of(" \t");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

std::string groupOf(const std::string& name)
{
  static const std::regex correctives(R"(^(phm|pjcm|pbm|pcm|jcm|mcm)|correct)", std::regex::icase);
  static const std::regex eyes("eye|brow|lid|lash");
  static const std::regex mouth("mouth|lip|jaw|tongue|teeth|smile|frown|viseme|^aa$|^ee$|^oo$");
  static const std::regex head("head|face|nose|cheek|chin|ear|forehead|expression|emotion");
  static const std::regex body("body|breast|glute|waist|hip|thigh|arm|leg|shoulder|chest|belly|weight|muscle|height");
  std::string             n = name;
  std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if(std::regex_search(name, correctives))
    return "Correctives";
  if(std::regex_search(n, eyes))
    return "Eyes";
  if(std::regex_search(n, mouth))
    return "Mouth";
  if(std::regex_search(n, head))
    return "Head";
  if(std::regex_search(n, body))
    return "Body";
  return "Other";
}

}  // namespace

MorphSet::MorphSet(std::function<mesh::BuiltMesh*(int)> geo)
    : m_geo(std::move(geo))
{
}

int MorphSet::size() const
{
  return int(std::count_if(m_channels.begin(), m_channels.end(), [](const Channel& c) { return !c.hidden; }));
}

void MorphSet::reindex()
{
  m_index.clear();
  for(size_t i = 0; i < m_channels.size(); i++)
    m_index[m_channels[i].name] = i;
}

const Channel* MorphSet::channel(const std::string& name) const
{
  auto it = m_index.find(name);
  return it == m_index.end() ? nullptr : &m_channels[it->second];
}

Channel* MorphSet::channelMut(const std::string& name)
{
  auto it = m_index.find(name);
  return it == m_index.end() ? nullptr : &m_channels[it->second];
}

double MorphSet::get(const std::string& name) const
{
  const Channel* c = channel(name);
  return c ? c->value : 0;
}

double MorphSet::weightOf(const std::string& name) const
{
  auto it = m_weights.find(name);
  return it != m_weights.end() ? it->second : get(name);
}

bool MorphSet::has(const std::string& name) const
{
  return m_bindings.count(name) || m_lazy.count(name);
}

MorphSet::MeshState& MorphSet::ensureCpu(int mesh)
{
  auto it = m_cpu.find(mesh);
  if(it != m_cpu.end())
    return it->second;
  mesh::BuiltMesh& geo = *m_geo(mesh);
  MeshState&       st  = m_cpu[mesh];
  st.basePos           = geo.position;
  st.baseNrm           = geo.normal;
  st.welder            = std::make_unique<mesh::NormalWelder>(geo.renderIndex, geo.firstSplit, geo.nextSplit, geo.renderTris, geo.renderVertexCount);
  return st;
}

void MorphSet::bind(const std::string& name, Binding b, double value, const ChannelOptions& opts)
{
  auto it = m_bindings.find(name);
  if(it == m_bindings.end())
  {
    it = m_bindings.emplace(name, std::vector<Binding>{}).first;
    Channel c;
    c.name   = name;
    c.label  = opts.label.value_or(prettyLabel(name));
    c.group  = opts.group.value_or(groupOf(name));
    c.value  = value;
    c.min    = opts.min.value_or(0);
    c.max    = opts.max.value_or(1);
    c.hidden = opts.hidden;
    m_index[name] = m_channels.size();
    m_channels.push_back(std::move(c));
  }
  it->second.push_back(std::move(b));
}

void MorphSet::addTarget(const std::string& name, int mesh, dson::TargetPtr target, const ChannelOptions& opts)
{
  ensureCpu(mesh);
  bind(name, {mesh, target}, opts.value.value_or(0), opts);
  const double v = weightOf(name);
  if(v != 0)
  {
    MeshState& st            = m_cpu.at(mesh);
    st.active[target.get()] = {target, v};
    st.dirty                 = true;
  }
}

void MorphSet::addLazy(const std::string& name, Loader loader, const ChannelOptions& opts)
{
  if(m_bindings.count(name) || m_lazy.count(name))
    return;
  m_lazy[name] = std::move(loader);
  Channel c;
  c.name        = name;
  c.label       = opts.label.value_or(prettyLabel(name));
  c.group       = opts.group.value_or(groupOf(name));
  c.min         = opts.min.value_or(0);
  c.max         = opts.max.value_or(1);
  c.lazy        = true;
  c.hidden      = opts.hidden;
  m_index[name] = m_channels.size();
  m_channels.push_back(std::move(c));
}

void MorphSet::ensureLoaded(const std::string& name)
{
  auto it = m_lazy.find(name);
  if(it == m_lazy.end() || m_loading.count(name))
    return;
  m_loading.emplace(name, jobs::pool().submit(it->second));
}

void MorphSet::finishLoad(const std::string& name, std::future<Finish>& fut)
{
  try
  {
    const Finish     finish = fut.get();
    const LazyResult result = finish();
    m_lazy.erase(name);
    // removed while it loaded (the item it belonged to was taken off)
    Channel* ch = channelMut(name);
    if(!ch)
      return;
    if(result.parts.empty() && !result.keep)
    {
      m_channels.erase(m_channels.begin() + std::ptrdiff_t(ch - m_channels.data()));
      reindex();
    }
    else
    {
      ch->lazy = false;
      const std::string* label = result.label ? &*result.label : (!result.parts.empty() && result.parts[0].hasLabel ? &result.parts[0].label : nullptr);
      if(label && !label->empty())
        ch->label = *label;
      if(result.min)
        ch->min = *result.min;
      else if(!result.parts.empty())
        ch->min = result.parts[0].min;
      if(result.max)
        ch->max = *result.max;
      else if(!result.parts.empty())
        ch->max = result.parts[0].max;
      auto& list = m_bindings[name];
      for(const dson::MorphPart& part : result.parts)
      {
        ensureCpu(part.mesh);
        list.push_back({part.mesh, part.target});
      }
      const double w = weightOf(name);
      if(w != 0)
        applyWeights({{name, w}});
    }
    // the file may have brought formulas: the driver re-evaluates with them
    if(!m_driverDirty)
      m_driverSince = std::chrono::steady_clock::now();
    m_driverDirty = true;
    m_listVersion++;
  }
  catch(const dson::MissingFileError&)
  {
    m_lazy.erase(name);
    if(auto* ch = channelMut(name))
    {
      m_channels.erase(m_channels.begin() + std::ptrdiff_t(ch - m_channels.data()));
      reindex();
    }
    m_listVersion++;
  }
  catch(const std::exception& e)
  {
    m_lazy.erase(name);
    if(auto* ch = channelMut(name))
    {
      m_channels.erase(m_channels.begin() + std::ptrdiff_t(ch - m_channels.data()));
      reindex();
    }
    m_errors.push_back("Morph " + name + ": " + e.what());
    m_listVersion++;
  }
}

void MorphSet::runDriver()
{
  if(!m_driverDirty)
    return;
  m_driverDirty = false;
  if(driver)
    driver->update();
}

void MorphSet::pump()
{
  bool any = false;
  for(auto it = m_loading.begin(); it != m_loading.end();)
  {
    if(it->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
      ++it;
      continue;
    }
    // finishing can start more loads (and so grow m_loading): work from a moved-out future
    const std::string name = it->first;
    std::future<Finish> fut = std::move(it->second);
    it                      = m_loading.erase(it);
    finishLoad(name, fut);
    any = true;
  }
  if(m_driverDirty && (m_loading.empty() || std::chrono::steady_clock::now() - m_driverSince >= kDriverDelay))
    runDriver();
  (void)any;
}

void MorphSet::settled()
{
  for(;;)
  {
    while(!m_loading.empty())
    {
      m_loading.begin()->second.wait();
      pump();
    }
    if(!m_driverDirty)
      return;
    runDriver();
  }
}

void MorphSet::set(const std::string& name, double value)
{
  setMany({{name, value}});
}

void MorphSet::setMany(const std::vector<std::pair<std::string, double>>& values)
{
  std::vector<std::pair<std::string, double>> changed;
  for(const auto& [name, value] : values)
  {
    Channel* ch = channelMut(name);
    if(!ch)
      continue;
    // the buffers still hold the old weight until the driver says otherwise
    if(driver && !m_weights.count(name))
      m_weights[name] = ch->value;
    ch->value = value;
    changed.emplace_back(name, value);
  }
  if(changed.empty())
    return;
  if(driver)
    driver->update();
  else
    applyWeights(changed);
}

bool MorphSet::apply(const std::string& name, double value)
{
  const Channel* ch = channel(name);
  if(!ch)
    return false;
  if(value == ch->value)
    m_weights.erase(name);
  else
    m_weights[name] = value;
  if(m_lazy.count(name))
  {
    if(value != 0)
      ensureLoaded(name);
    return true;
  }
  auto it = m_bindings.find(name);
  if(it == m_bindings.end())
    return true;
  for(const Binding& b : it->second)
  {
    MeshState& st = m_cpu.at(b.mesh);
    if(value == 0)
      st.active.erase(b.target.get());
    else
      st.active[b.target.get()] = {b.target, value};
    st.dirty = true;
  }
  return true;
}

void MorphSet::applyWeights(const std::vector<std::pair<std::string, double>>& values)
{
  for(const auto& [name, weight] : values)
    apply(name, weight);
  flush();
}

void MorphSet::resetAll()
{
  std::vector<std::pair<std::string, double>> zero;
  for(const Channel& c : m_channels)
    if(c.value != 0)
      zero.emplace_back(c.name, 0.0);
  setMany(zero);
}

void MorphSet::markRebuild(int mesh)
{
  auto it = m_cpu.find(mesh);
  if(it == m_cpu.end())
    return;
  it->second.sinceRebuild = kRebuildEvery;
  it->second.dirty        = true;
}

void MorphSet::sort()
{
  static const std::vector<std::string> order{"Head", "Eyes", "Mouth", "Body", "Other", "Correctives"};
  auto                                  rank = [&](const std::string& g) {
    auto i = std::find(order.begin(), order.end(), g);
    return size_t(i - order.begin());
  };
  // active channels first so a loaded character's shape is visible at the top
  std::stable_sort(m_channels.begin(), m_channels.end(), [&](const Channel& a, const Channel& b) {
    if((b.value != 0) != (a.value != 0))
      return (b.value != 0) < (a.value != 0);
    if(rank(a.group) != rank(b.group))
      return rank(a.group) < rank(b.group);
    if(a.group != b.group)
      return a.group < b.group;
    return a.label < b.label;
  });
  reindex();
}

std::vector<std::string> MorphSet::takeErrors()
{
  return std::exchange(m_errors, {});
}

void MorphSet::change(int mesh, bool all, const std::vector<uint32_t>& moved, const std::vector<uint32_t>& written)
{
  MeshChange& c = m_changes[mesh];
  c.mesh        = mesh;
  if(all)
    c.all = true;
  if(c.all)
  {
    c.indices.clear();
    return;
  }
  c.indices.insert(c.indices.end(), moved.begin(), moved.end());
  c.indices.insert(c.indices.end(), written.begin(), written.end());
}

std::vector<MeshChange> MorphSet::takeChanges()
{
  std::vector<MeshChange> out;
  for(auto& [mesh, c] : m_changes)
  {
    std::sort(c.indices.begin(), c.indices.end());
    c.indices.erase(std::unique(c.indices.begin(), c.indices.end()), c.indices.end());
    out.push_back(std::move(c));
  }
  m_changes.clear();
  return out;
}

void MorphSet::flush()
{
  for(auto& [mesh, st] : m_cpu)
  {
    if(!st.dirty)
      continue;
    st.dirty = false;
    if(++st.sinceRebuild > kRebuildEvery)
      rebuild(mesh, st);
    else
      update(mesh, st);
    st.applied = st.active;
  }
}

// Adds each changed target's weight difference; re-welds normals around the moved vertices.
void MorphSet::update(int mesh, MeshState& st)
{
  mesh::BuiltMesh& geo = *m_geo(mesh);
  float*           pos = geo.position.data();
  const uint32_t   n   = uint32_t(geo.position.size() / 3);
  if(st.mark.size() != n)
    st.mark.assign(n, 0);
  std::vector<uint32_t> moved;
  auto                  add = [&](const mesh::RenderTarget& t, double dw) {
    const size_t count = t.indices.size();
    for(size_t i = 0; i < count; i++)
    {
      const uint32_t v = t.indices[i];
      const size_t   o = size_t(v) * 3, d = i * 3;
      pos[o]     = float(pos[o] + t.deltas[d] * dw);
      pos[o + 1] = float(pos[o + 1] + t.deltas[d + 1] * dw);
      pos[o + 2] = float(pos[o + 2] + t.deltas[d + 2] * dw);
      if(!st.mark[v])
      {
        st.mark[v] = 1;
        moved.push_back(v);
      }
    }
  };
  for(const auto& [key, a] : st.active)
  {
    auto         it   = st.applied.find(key);
    const double prev = it == st.applied.end() ? 0 : it->second.weight;
    if(a.weight != prev)
      add(*a.target, a.weight - prev);
  }
  for(const auto& [key, prev] : st.applied)
    if(!st.active.count(key) && prev.weight != 0)
      add(*prev.target, -prev.weight);
  for(uint32_t v : moved)
    st.mark[v] = 0;
  if(moved.empty())
    return;
  std::vector<uint32_t> written;
  const auto            range = st.welder->partial(geo.position, geo.normal, moved, &written);
  change(mesh, range && written.empty(), moved, written);
}

// Rebuilds the buffers from the base positions and normals plus every active target.
void MorphSet::rebuild(int mesh, MeshState& st)
{
  st.sinceRebuild      = 0;
  mesh::BuiltMesh& geo = *m_geo(mesh);
  geo.position         = st.basePos;
  geo.normal           = st.baseNrm;
  float* pos           = geo.position.data();
  for(const auto& [key, a] : st.active)
  {
    const mesh::RenderTarget& t = *a.target;
    for(size_t i = 0, n = t.indices.size(); i < n; i++)
    {
      const size_t o = size_t(t.indices[i]) * 3, d = i * 3;
      pos[o]         = float(pos[o] + t.deltas[d] * a.weight);
      pos[o + 1]     = float(pos[o + 1] + t.deltas[d + 1] * a.weight);
      pos[o + 2]     = float(pos[o + 2] + t.deltas[d + 2] * a.weight);
    }
  }
  st.welder->full(geo.position, geo.normal);
  change(mesh, true, {}, {});
}

}  // namespace fv::morph
