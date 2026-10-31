#include <figure/morph/morph_set.hpp>
#include <figure/io/timing.hpp>

#include <algorithm>
#include <cmath>
#include <regex>

#include <figure/jobs/pool.hpp>

namespace fv::morph {

namespace {

// Incremental flushes before the buffer is rebuilt from the base positions, so rounding can't build up.
constexpr int kRebuildEvery = 400;
// a target with at least this many vertices adds its deltas in parallel chunks of this size
constexpr size_t kParallelDeltas = 16384;
// How long a lazy file waits for others before the driver runs for it.
constexpr auto kDriverDelay = std::chrono::milliseconds(100);

// made when a mesh's drawn vertices first move (a cageOnly mesh's may never)
std::unique_ptr<mesh::NormalWelder> welderOf(const mesh::BuiltMesh& geo)
{
  return std::make_unique<mesh::NormalWelder>(geo.renderIndex, geo.firstSplit, geo.nextSplit, geo.renderTris, geo.renderVertexCount);
}

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
    // a value given is one set: a scene's, for an item or another figure
    c.dialled = opts.value.has_value();
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

std::vector<std::string> MorphSet::itemsMoving(const std::string& name) const
{
  std::vector<std::string> out;
  const Channel*           ch = channel(name);
  for(const auto& st : m_followers)
  {
    const Follower& f = st->f;
    // an item of several meshes follows once for each
    if(std::find(out.begin(), out.end(), f.label) != out.end())
      continue;
    const auto holds = [&](const std::string& dial) { return meshesOf(dial).count(f.mesh) || (f.hasOwn && f.hasOwn(dial)); };
    if(holds(name) || (ch && ch->reach && std::any_of(ch->reach->drives.begin(), ch->reach->drives.end(), holds)))
      out.push_back(f.label);
  }
  return out;
}

std::vector<float> MorphSet::morphedCage(int mesh, const mesh::RenderTarget* except) const
{
  const mesh::BuiltMesh* geo = m_geo(mesh);
  if(!geo)
    return {};
  // a cageOnly mesh's kept cage, when no change waits for a flush
  if(const auto it = m_cpu.find(mesh); !except && it != m_cpu.end() && !it->second.dirty && !it->second.cage.empty())
    return it->second.cage;
  std::vector<float> out = geo->cagePositions;
  for(const auto& [target, weight] : activeTargets(mesh, except))
    for(size_t i = 0; i < target->cageIndices.size(); i++)
    {
      const size_t v = target->cageIndices[i];
      if(v * 3 + 2 >= out.size())
        continue;
      out[v * 3] += float(target->cageDeltas[i * 3] * weight);
      out[v * 3 + 1] += float(target->cageDeltas[i * 3 + 1] * weight);
      out[v * 3 + 2] += float(target->cageDeltas[i * 3 + 2] * weight);
    }
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
  m_listVersion++;
}

bool MorphSet::describe(const std::string& name, const dson::MorphInfo& info)
{
  Channel* ch = channelMut(name);
  if(!ch)
    return false;
  if(!info.label.empty())
    ch->label = info.label;
  ch->path   = info.path;
  ch->region = info.region;
  ch->kind   = info.kind;
  ch->helper = info.helper;
  if(ch->lazy)
  {
    ch->min = info.min.value_or(ch->min);
    ch->max = info.max.value_or(ch->max);
  }
  // one that loaded at its rest still does, whoever has dialled it since: a scene file holds a 0 set
  // before the rest was known
  if(ch->loaded == ch->rest)
    ch->loaded = info.rest;
  ch->rest = info.rest;
  // loaded before its rest was known: a switch isn't inert, whatever its file holds
  ch->inert = ch->inert && info.rest == 0;
  m_listVersion++;
  if(ch->value == info.rest)
    return false;
  if(ch->dialled)
  {
    // set away from a rest it didn't know of, which no change of value said (a 0 on a dial that was 0)
    if(!m_lazy.count(name) || m_loading.count(name))
      return false;
    ensureLoaded(name);
    return true;
  }
  // the buffers still hold the old weight until the driver says otherwise; a file not read has none,
  // and takes the weight when something does read it
  if(driver && !m_lazy.count(name) && !m_weights.count(name))
    m_weights[name] = ch->value;
  ch->value = info.rest;
  m_version++;
  if(driver)
  {
    if(!m_driverDirty)
      m_driverSince = std::chrono::steady_clock::now();
    m_driverDirty = true;
  }
  else
    applyWeights({{name, info.rest}});
  return true;
}

void MorphSet::setReach(const std::string& name, const Reach& reach)
{
  Channel* ch = channelMut(name);
  if(!ch || ch->reach == reach)
    return;
  ch->reach = reach;
  m_listVersion++;
}

bool MorphSet::load(const std::string& name)
{
  if(!m_lazy.count(name) || m_loading.count(name))
    return false;
  ensureLoaded(name);
  return true;
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
    // Nothing in the file. A worn item may hold the deltas, so the dial stays: its slider used to go on
    // the first drag. A hidden one has no slider to keep, unless an item's copy is bound to it. A switch
    // that rests elsewhere has nothing in its file either, but its readers go by what it's set to.
    const bool inert = result.parts.empty() && !result.keep && ch->rest == 0;
    if(inert && ch->hidden && !m_bindings.count(name))
    {
      m_channels.erase(m_channels.begin() + std::ptrdiff_t(ch - m_channels.data()));
      reindex();
    }
    else
    {
      ch->lazy  = false;
      ch->inert = inert;
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
  // meshes no longer cageOnly (the GPU stopped refining them) are drawn as their morphs have them
  catchUp([this](int mesh) { return !cageOnly || !cageOnly(mesh); });
}

void MorphSet::settled()
{
  for(;;)
  {
    while(!m_loading.empty())
    {
      {
        fv::timing::Accum t("settled: wait for a load");
        m_loading.begin()->second.wait();
      }
      fv::timing::Accum t("settled: pump");
      pump();
    }
    if(!m_driverDirty)
      break;
    fv::timing::Accum t("settled: driver");
    runDriver();
  }
  catchUp([this](int mesh) { return !cageOnly || !cageOnly(mesh); });
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
    ch->dialled = true;
    if(ch->value == value)
      continue;
    // the buffers still hold the old weight until the driver says otherwise
    if(driver && !m_weights.count(name))
      m_weights[name] = ch->value;
    ch->value = value;
    changed.emplace_back(name, value);
  }
  if(changed.empty())
    return;
  m_version++;
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
  {
    const auto   old    = m_weights.find(name);
    const double before = old == m_weights.end() ? ch->value : old->second;
    if(value != before)
      m_version++;
  }
  if(value == ch->value)
    m_weights.erase(name);
  else
    m_weights[name] = value;
  if(m_lazy.count(name))
  {
    // what a formula reads at its rest is the channel's own value: only leaving it wants the file
    if(value != ch->rest)
      ensureLoaded(name);
    return true;
  }
  for(auto& f : m_followers)
    follow(*f, name, value);
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

void MorphSet::addFollower(Follower f)
{
  ensureCpu(f.mesh);
  auto st = std::make_unique<FollowerState>();
  st->f   = std::move(f);
  for(auto& [source, projection] : st->f.preprojected)
    st->projected.emplace(source.get(), std::make_pair(source, projection));
  st->f.preprojected.clear();
  FollowerState& ref = *st;
  m_followers.push_back(std::move(st));
  // what the dials move is part of the list
  m_listVersion++;
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
// follower with its own binding for the channel (a scene file set the item's copy) is left to it. One
// with a copy of its own moves though the source has nothing for it, pump() applying the copy once it
// has loaded. A dial still waiting for its file waits with it: finishLoad() applies both.
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
  auto       prev   = st.applied.find(name);
  const bool moving = !sources.empty() || (!m_lazy.count(name) && st.f.hasOwn && st.f.hasOwn(name));
  if(!moving && prev == st.applied.end())
    return false;
  if(prev != st.applied.end())
  {
    for(const auto& t : prev->second)
      cpu->second.active.erase(t.get());
    st.applied.erase(prev);
  }
  cpu->second.dirty = true;
  if(value == 0 || !moving || own)
    return true;
  auto targets = followerTargets(st, name, sources);
  for(const auto& t : targets)
    cpu->second.active[t.get()] = {t, value};
  st.applied[name] = std::move(targets);
  return true;
}

// A pose drives dozens of correctives at once, and each worn item needs its own version of every
// body target they move: projected one at a time that was most of a restore's pose (0.5 s on a
// dressed figure). Each projection is a pure function of one follower's binding and one source
// target, so they are made side by side here, into the cache followerTargets() reads, and what
// follow() then applies is what it would have made itself. The conditions are follow()'s: a
// channel that is loaded and bound, a follower with no binding of its own for it and no shapes of
// its own that arrived.
void MorphSet::projectAhead(const std::vector<std::pair<std::string, double>>& values)
{
  struct Work
  {
    FollowerState*  st;
    dson::TargetPtr source, out;
  };
  std::vector<Work>                                                    work;
  std::set<std::pair<const FollowerState*, const mesh::RenderTarget*>> seen;
  for(const auto& [name, value] : values)
  {
    if(value == 0 || m_lazy.count(name))
      continue;
    const auto list = m_bindings.find(name);
    if(list == m_bindings.end())
      continue;
    for(auto& f : m_followers)
    {
      FollowerState& st = *f;
      if(!st.f.project || !m_cpu.count(st.f.mesh))
        continue;
      if(const auto own = st.own.find(name); own != st.own.end() && own->second.state == FollowerState::Own::State::eReady)
        continue;
      if(std::any_of(list->second.begin(), list->second.end(), [&](const Binding& b) { return b.mesh == st.f.mesh; }))
        continue;
      for(const Binding& b : list->second)
        if(b.mesh == st.f.source && !st.projected.count(b.target.get()) && seen.insert({&st, b.target.get()}).second)
          work.push_back({&st, b.target, nullptr});
    }
  }
  if(work.size() < 2)
    return;  // one is made where it is asked for
  fv::timing::Accum t("morph.project");
  jobs::parallelFor(work.size(), 1, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
      work[i].out = work[i].st->f.project(*work[i].source);
  });
  for(Work& w : work)
    w.st->projected.emplace(w.source.get(), std::make_pair(w.source, std::move(w.out)));
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
      // the figure's dial, which that item's copy was bound to, is there for the next one
      const auto kept = std::find_if(m_channels.begin(), m_channels.end(), [&](const Channel& c) { return c.name == name && c.inert; });
      if(kept != m_channels.end())
      {
        ++it;
        continue;
      }
      it = m_bindings.erase(it);
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
  projectAhead(values);
  for(const auto& [name, weight] : values)
    apply(name, weight);
  flush();
}

void MorphSet::markLoaded(const std::set<std::string>& except)
{
  for(Channel& c : m_channels)
    if(!except.count(c.name))
      c.loaded = c.value;
}

void MorphSet::resetAll()
{
  std::vector<std::pair<std::string, double>> back;
  // a hidden channel is no dial: a drape's or a follower's stays as it is
  for(const Channel& c : m_channels)
    if(c.value != c.rest && !c.hidden)
      back.emplace_back(c.name, c.rest);
  setMany(back);
  // nothing has dialled them now: what a file says later still applies
  for(Channel& c : m_channels)
    c.dialled = false;
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
    if((b.value != b.rest) != (a.value != a.rest))
      return (b.value != b.rest) < (a.value != a.rest);
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

void MorphSet::change(int mesh, bool all, const std::vector<uint32_t>& moved, const std::vector<uint32_t>& written, bool cage, bool caughtUp)
{
  // a cage change, or a catch-up, stays one only while nothing else is reported for the mesh
  const bool  fresh = !m_changes.count(mesh);
  MeshChange& c     = m_changes[mesh];
  c.mesh            = mesh;
  c.cage            = cage && (fresh || c.cage);
  c.caughtUp        = caughtUp && (fresh || c.caughtUp);
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
  std::vector<std::pair<int, MeshState*>> dirty;
  for(auto& [mesh, st] : m_cpu)
    if(st.dirty)
      dirty.emplace_back(mesh, &st);
  if(dirty.empty())
    return;
  // Each mesh's buffers, welder and state are its own, so the meshes update side by side; the
  // changes are recorded afterwards, in mesh order, as one thread's work would have.
  std::vector<MeshUpdate> updates(dirty.size());
  std::vector<uint8_t>    cages(dirty.size());
  if(cageOnly)
    for(size_t i = 0; i < dirty.size(); i++)
      cages[i] = cageOnly(dirty[i].first);
  jobs::parallelFor(dirty.size(), 1, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      auto& [mesh, st] = dirty[i];
      st->dirty        = false;
      if(cages[i])
      {
        // the drawn buffers stay at `applied` until catchUp()
        updateCage(mesh, *st);
        st->behind = true;
        continue;
      }
      updates[i]  = ++st->sinceRebuild > kRebuildEvery ? rebuild(mesh, *st) : update(mesh, *st);
      st->applied = st->active;
      st->behind  = false;
      st->cage.clear();
    }
  });
  for(size_t i = 0; i < dirty.size(); i++)
    if(cages[i])
      change(dirty[i].first, true, {}, {}, true);
    else if(updates[i].changed)
      change(updates[i].mesh, updates[i].all, updates[i].moved, updates[i].written);
}

void MorphSet::catchUp()
{
  catchUp([](int) { return true; });
}

void MorphSet::catchUp(const std::function<bool(int mesh)>& which)
{
  std::vector<std::pair<int, MeshState*>> behind;
  for(auto& [mesh, st] : m_cpu)
    if(st.behind && !st.dirty && which(mesh))
      behind.emplace_back(mesh, &st);
  if(behind.empty())
    return;
  fv::timing::Accum       t("morph.catch up");
  std::vector<MeshUpdate> updates(behind.size());
  jobs::parallelFor(behind.size(), 1, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      auto& [mesh, st] = behind[i];
      updates[i]       = ++st->sinceRebuild > kRebuildEvery ? rebuild(mesh, *st) : update(mesh, *st);
      st->applied      = st->active;
      st->behind       = false;
    }
  });
  for(const MeshUpdate& u : updates)
    if(u.changed)
      change(u.mesh, u.all, u.moved, u.written, false, true);
}

void MorphSet::updateCage(int mesh, MeshState& st)
{
  const mesh::BuiltMesh& geo = *m_geo(mesh);
  const auto             add = [&](const mesh::RenderTarget& t, double dw) {
    for(size_t i = 0; i < t.cageIndices.size(); i++)
    {
      const size_t v = t.cageIndices[i];
      if(v * 3 + 2 >= st.cage.size())
        continue;
      st.cage[v * 3]     = float(st.cage[v * 3] + t.cageDeltas[i * 3] * dw);
      st.cage[v * 3 + 1] = float(st.cage[v * 3 + 1] + t.cageDeltas[i * 3 + 1] * dw);
      st.cage[v * 3 + 2] = float(st.cage[v * 3 + 2] + t.cageDeltas[i * 3 + 2] * dw);
    }
  };
  if(st.cage.size() != geo.cagePositions.size() || ++st.sinceCageRebuild > kRebuildEvery)
  {
    st.sinceCageRebuild = 0;
    st.cage             = geo.cagePositions;
    for(const auto& [key, a] : st.active)
      add(*a.target, a.weight);
  }
  else
  {
    for(const auto& [key, a] : st.active)
    {
      const auto   it   = st.cageApplied.find(key);
      const double prev = it == st.cageApplied.end() ? 0 : it->second.weight;
      if(a.weight != prev)
        add(*a.target, a.weight - prev);
    }
    for(const auto& [key, prev] : st.cageApplied)
      if(!st.active.count(key) && prev.weight != 0)
        add(*prev.target, -prev.weight);
  }
  st.cageApplied = st.active;
}

// Adds each changed target's weight difference; re-welds normals around the moved vertices.
MorphSet::MeshUpdate MorphSet::update(int mesh, MeshState& st)
{
  MeshUpdate       out;
  out.mesh = mesh;
  mesh::BuiltMesh& geo = *m_geo(mesh);
  float*           pos = geo.position.data();
  const uint32_t   n   = uint32_t(geo.position.size() / 3);
  if(st.mark.size() != n)
    st.mark.assign(n, 0);
  std::vector<uint32_t> moved;
  const auto            deltasStart = std::chrono::steady_clock::now();
  auto                  add = [&](const mesh::RenderTarget& target, double dw) {
    const mesh::RenderTarget::Drawn& t     = target.drawn(geo);
    const size_t                     count = t.indices.size();
    const auto   run   = [&](size_t begin, size_t end, std::vector<uint32_t>& found) {
      for(size_t i = begin; i < end; i++)
      {
        const uint32_t v = t.indices[i];
        const size_t   o = size_t(v) * 3, d = i * 3;
        pos[o]     = float(pos[o] + t.deltas[d] * dw);
        pos[o + 1] = float(pos[o + 1] + t.deltas[d + 1] * dw);
        pos[o + 2] = float(pos[o + 2] + t.deltas[d + 2] * dw);
        if(!st.mark[v])
        {
          st.mark[v] = 1;
          found.push_back(v);
        }
      }
    };
    if(count < kParallelDeltas)
    {
      run(0, count, moved);
      return;
    }
    // A target lists each vertex once, so its chunks write disjoint positions and marks; the
    // vertices each chunk found moved go onto the list in chunk order, which is the order one
    // thread would have found them in.
    std::vector<std::vector<uint32_t>> found((count + kParallelDeltas - 1) / kParallelDeltas);
    jobs::parallelFor(count, kParallelDeltas, [&](size_t begin, size_t end) { run(begin, end, found[begin / kParallelDeltas]); });
    for(const auto& f : found)
      moved.insert(moved.end(), f.begin(), f.end());
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
  fv::timing::accumulate("morph.deltas", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - deltasStart).count());
  if(moved.empty())
    return out;
  std::vector<uint32_t> written;
  const auto            weldStart = std::chrono::steady_clock::now();
  if(!st.welder)
    st.welder = welderOf(geo);
  const auto            range     = st.welder->partial(geo.position, geo.normal, moved, &written);
  // a weld of the whole mesh (more than a quarter of it moved) makes the change `all`
  const bool full = range && written.empty();
  fv::timing::accumulate(full ? "morph.weld.full" : "morph.weld.partial", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - weldStart).count());
  out.changed = true;
  out.all     = full;
  out.moved   = std::move(moved);
  out.written = std::move(written);
  return out;
}

// Rebuilds the buffers from the base positions and normals plus every active target.
MorphSet::MeshUpdate MorphSet::rebuild(int mesh, MeshState& st)
{
  fv::timing::Accum rebuildTime("morph.rebuild");
  st.sinceRebuild      = 0;
  mesh::BuiltMesh& geo = *m_geo(mesh);
  geo.position         = st.basePos;
  geo.normal           = st.baseNrm;
  float* pos           = geo.position.data();
  for(const auto& [key, a] : st.active)
  {
    const mesh::RenderTarget::Drawn& t = a.target->drawn(geo);
    for(size_t i = 0, n = t.indices.size(); i < n; i++)
    {
      const size_t o = size_t(t.indices[i]) * 3, d = i * 3;
      pos[o]         = float(pos[o] + t.deltas[d] * a.weight);
      pos[o + 1]     = float(pos[o + 1] + t.deltas[d + 1] * a.weight);
      pos[o + 2]     = float(pos[o + 2] + t.deltas[d + 2] * a.weight);
    }
  }
  if(!st.welder)
    st.welder = welderOf(geo);
  st.welder->full(geo.position, geo.normal);
  MeshUpdate out;
  out.mesh    = mesh;
  out.changed = true;
  out.all     = true;
  return out;
}

}  // namespace fv::morph
