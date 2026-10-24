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
    // a lazy channel of that name (a figure morph not read yet, which a worn item ships a fitted
    // copy of) takes the binding: one dial, at its own value
    if(m_index.count(name))
    {
      it->second.push_back(std::move(b));
      return;
    }
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

std::vector<std::pair<dson::TargetPtr, double>> MorphSet::activeTargets(int mesh, const mesh::RenderTarget* except) const
{
  std::vector<std::pair<dson::TargetPtr, double>> out;
  const auto it = m_cpu.find(mesh);
  if(it == m_cpu.end())
    return out;
  for(const auto& [key, a] : it->second.active)
    if(key != except)
      out.emplace_back(a.target, a.weight);
  return out;
}

std::set<int> MorphSet::meshesOf(const std::string& name) const
{
  std::set<int> out;
  if(const auto it = m_bindings.find(name); it != m_bindings.end())
    for(const Binding& b : it->second)
      out.insert(b.mesh);
  return out;
}

dson::TargetPtr MorphSet::cpuTarget(const std::string& name, int mesh) const
{
  const auto it = m_bindings.find(name);
  if(it == m_bindings.end())
    return nullptr;
  for(const Binding& b : it->second)
    if(b.mesh == mesh)
      return b.target;
  return nullptr;
}

void MorphSet::replaceCpuTarget(const std::string& name, int mesh, dson::TargetPtr target, const ChannelOptions& opts)
{
  MeshState& st = ensureCpu(mesh);
  auto       it = m_bindings.find(name);
  if(it == m_bindings.end())
  {
    addTarget(name, mesh, std::move(target), opts);
    m_listVersion++;
    flush();
    return;
  }
  for(auto b = it->second.begin(); b != it->second.end();)
  {
    if(b->mesh == mesh)
    {
      st.active.erase(b->target.get());
      b = it->second.erase(b);
    }
    else
      ++b;
  }
  it->second.push_back({mesh, target});
  const double v = weightOf(name);
  if(v != 0)
    st.active[target.get()] = {target, v};
  st.dirty = true;
  flush();
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
    // nothing in the file, and nothing else bound to it (a worn item's copy of the morph)
    if(result.parts.empty() && !result.keep && !m_bindings.count(name))
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
  // followers whose own fitted targets arrived replace their projections
  for(auto& st : m_followers)
    for(auto& [name, own] : st->own)
    {
      if(own.state != FollowerState::Own::State::ePending || own.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        continue;
      try
      {
        own.targets = own.future.get();
      }
      catch(const std::exception&)
      {
        own.targets.clear();
      }
      own.state = own.targets.empty() ? FollowerState::Own::State::eNone : FollowerState::Own::State::eReady;
      if(own.state == FollowerState::Own::State::eReady)
      {
        const double v = weightOf(name);
        if(v != 0 && follow(*st, name, v))
          any = true;
      }
    }
  if(any)
    flush();
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
  for(auto& f : m_followers)
    follow(*f, name, value);
  return true;
}

void MorphSet::addFollower(Follower f)
{
  ensureCpu(f.mesh);
  auto st = std::make_unique<FollowerState>();
  st->f   = std::move(f);
  FollowerState& ref = *st;
  m_followers.push_back(std::move(st));
  bool moved = false;
  for(const Channel& ch : m_channels)
  {
    const double w = weightOf(ch.name);
    if(w != 0 && follow(ref, ch.name, w))
      moved = true;
  }
  if(moved)
    flush();
}

// Applies channel `name` at `value` to one follower; returns whether its shape may have changed. A
// follower with its own binding for the channel (a scene file set the item's copy) is left to it.
bool MorphSet::follow(FollowerState& st, const std::string& name, double value)
{
  auto cpu = m_cpu.find(st.f.mesh);
  if(cpu == m_cpu.end())
    return false;
  const auto               list = m_bindings.find(name);
  std::vector<Binding>     sources;
  bool                     own = false;
  if(list != m_bindings.end())
    for(const Binding& b : list->second)
    {
      if(b.mesh == st.f.source)
        sources.push_back(b);
      if(b.mesh == st.f.mesh)
        own = true;
    }
  auto prev = st.applied.find(name);
  if(sources.empty() && prev == st.applied.end())
    return false;
  if(prev != st.applied.end())
  {
    for(const auto& t : prev->second)
      cpu->second.active.erase(t.get());
    st.applied.erase(prev);
  }
  cpu->second.dirty = true;
  if(value == 0 || sources.empty() || own)
    return true;
  auto targets = followerTargets(st, name, sources);
  for(const auto& t : targets)
    cpu->second.active[t.get()] = {t, value};
  st.applied[name] = std::move(targets);
  return true;
}

std::vector<dson::TargetPtr> MorphSet::followerTargets(FollowerState& st, const std::string& name, const std::vector<Binding>& sources)
{
  auto own = st.own.find(name);
  if(own == st.own.end())
  {
    FollowerState::Own o;
    if(st.f.own)
    {
      o.future = st.f.own(name);
      if(o.future.valid())
        o.state = FollowerState::Own::State::ePending;
    }
    own = st.own.emplace(name, std::move(o)).first;
  }
  if(own->second.state == FollowerState::Own::State::eReady)
    return own->second.targets;
  // projected meanwhile; the item's own shape replaces it when it has loaded (pump)
  std::vector<dson::TargetPtr> out;
  for(const Binding& b : sources)
  {
    auto it = st.projected.find(b.target.get());
    if(it == st.projected.end())
      it = st.projected.emplace(b.target.get(), std::make_pair(b.target, st.f.project ? st.f.project(*b.target) : nullptr)).first;
    if(it->second.second)
      out.push_back(it->second.second);
  }
  return out;
}

void MorphSet::removeMeshes(const std::set<int>& gone)
{
  m_followers.erase(std::remove_if(m_followers.begin(), m_followers.end(),
                                   [&](const auto& st) { return gone.count(st->f.mesh) || gone.count(st->f.source); }),
                    m_followers.end());
  for(int m : gone)
  {
    m_cpu.erase(m);
    m_changes.erase(m);
  }
  for(auto it = m_bindings.begin(); it != m_bindings.end();)
  {
    auto& list = it->second;
    const size_t before = list.size();
    list.erase(std::remove_if(list.begin(), list.end(), [&](const Binding& b) { return gone.count(b.mesh) > 0; }), list.end());
    if(list.size() != before && list.empty())
    {
      const std::string name = it->first;
      it                     = m_bindings.erase(it);
      if(!m_lazy.count(name))
      {
        m_channels.erase(std::remove_if(m_channels.begin(), m_channels.end(), [&](const Channel& c) { return c.name == name; }), m_channels.end());
        m_weights.erase(name);
      }
    }
    else
      ++it;
  }
  reindex();
  m_listVersion++;
}

void MorphSet::removeChannels(const std::vector<std::string>& names)
{
  const std::set<std::string> gone(names.begin(), names.end());
  for(const std::string& n : gone)
  {
    // what the channel applied comes off the meshes: its own targets and its followers' versions
    if(auto it = m_bindings.find(n); it != m_bindings.end())
      for(const Binding& b : it->second)
        if(auto cpu = m_cpu.find(b.mesh); cpu != m_cpu.end())
        {
          cpu->second.active.erase(b.target.get());
          cpu->second.dirty = true;
        }
    for(auto& f : m_followers)
      if(auto prev = f->applied.find(n); prev != f->applied.end())
      {
        if(auto cpu = m_cpu.find(f->f.mesh); cpu != m_cpu.end())
        {
          for(const auto& t : prev->second)
            cpu->second.active.erase(t.get());
          cpu->second.dirty = true;
        }
        f->applied.erase(prev);
      }
    m_bindings.erase(n);
    m_lazy.erase(n);
    m_weights.erase(n);
    m_loading.erase(n);
  }
  m_channels.erase(std::remove_if(m_channels.begin(), m_channels.end(), [&](const Channel& c) { return gone.count(c.name) > 0; }), m_channels.end());
  reindex();
  m_listVersion++;
  flush();
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
