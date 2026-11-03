#include <figure/pose/figure_driver.hpp>
#include <figure/io/timing.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace fv::pose {

using dson::Json;

namespace {
constexpr double kEps = 1e-6;
}  // namespace

void MorphFiles::add(const std::string& file, const std::string& stem, const dson::MorphInfo& info)
{
  if(info.id.empty())
    return;
  // an alias's value is another channel's
  if(!info.alias)
    dials[info.id] = {stem, info.rest};
  for(const std::string& dial : info.reads)
    readers[dial].push_back({file, stem, info.id});
}

FigureDriver::FigureDriver(morph::MorphSet& morphs, std::shared_ptr<FigureSource> source)
    : m_morphs(morphs)
    , m_source(std::move(source))
{
}

std::string FigureDriver::nameOf(const std::string& id) const
{
  auto it = m_alias.find(id);
  return it == m_alias.end() ? id : it->second;
}

morph::Loader FigureDriver::loader(const std::string& file, const std::string& name)
{
  return [this, source = m_source, file, name]() -> morph::Finish {
    // on a pool thread: the file and its targets
    std::shared_ptr<const dson::Doc> doc     = source->doc(file);
    std::vector<dson::MorphDef>      targets = source->targets(*doc, file);
    // on the owner's thread: formulas into the graph, the driven morphs, the channel
    return [this, file, name, doc, targets = std::move(targets)]() mutable -> morph::LazyResult {
      graph.addDoc(doc->json, file);
      const Json* mods = dson::get(&doc->json, "modifier_library");
      // the channel is the modifier of its name; a file of one modifier under another name is that one
      const Json* mod = nullptr;
      if(mods && mods->is_array())
      {
        for(const Json& m : *mods)
          if(dson::getString(&m, "id") == name)
          {
            mod = &m;
            break;
          }
        if(!mod && mods->size() == 1)
          mod = &(*mods)[0];
      }
      const std::string modId = mod ? dson::getString(mod, "id") : std::string();
      if(mod && modId != name)
      {
        m_alias[modId] = name;
        m_idOf[name]   = modId;
        m_aliasVersion++;
      }
      for(dson::MorphDef& t : targets)
      {
        if((mod && t.id == modId) || m_morphs.has(t.id))
          continue;
        // a file's other morphs are driven by its formulas, not dials of their own
        for(const dson::MorphPart& p : t.parts)
        {
          morph::ChannelOptions opts;
          if(p.hasLabel)
            opts.label = p.label;
          opts.group  = "Driven";
          opts.hidden = true;
          m_morphs.addTarget(t.id, p.mesh, p.target, opts);
        }
      }
      morph::LazyResult out;
      if(mod)
        for(dson::MorphDef& t : targets)
          if(t.id == modId)
          {
            out.parts = std::move(t.parts);
            break;
          }
      if(mods && mods->is_array())
        for(const Json& m : *mods)
          if(const Json* f = dson::get(&m, "formulas"); f && f->is_array() && !f->empty())
          {
            out.keep = true;
            break;
          }
      const Json* ch = dson::get(mod, "channel");
      // slider limits as morphsFromDoc sets them, for a controller that has no deltas to say
      const Json* clamped = dson::get(ch, "clamped");
      const bool  limited = !(clamped && clamped->is_boolean() && !clamped->get<bool>());
      const Json* lo      = dson::get(ch, "min");
      const Json* hi      = dson::get(ch, "max");
      out.min = lo && lo->is_number() && limited ? dson::sliderMin(lo->get<double>()) : (out.parts.empty() ? 0.0 : out.parts[0].min);
      out.max = hi && hi->is_number() && limited ? dson::sliderMax(hi->get<double>()) : (out.parts.empty() ? 1.0 : out.parts[0].max);
      for(const Json* l : {dson::get(ch, "label"), dson::get(mod, "label")})
        if(l && l->is_string())
        {
          out.label = l->get<std::string>();
          break;
        }
      return out;
    };
  };
}

const morph::Channel* FigureDriver::dialOf(const std::string& id) const
{
  if(const morph::Channel* c = m_morphs.channel(nameOf(id)))
    return c;
  const auto it = m_files.dials.find(id);
  return it == m_files.dials.end() ? nullptr : m_morphs.channel(it->second.stem);
}

FigureDriver::Stamp FigureDriver::stamp() const
{
  return {m_morphs.layoutVersion(), m_aliasVersion, m_filesVersion, rig};
}

void FigureDriver::refresh()
{
  if(const Stamp now = stamp(); !(now == m_resolvedFor))
  {
    m_resolvedFor = now;
    m_resolved.clear();
    m_channelKeys.clear();
  }
}

const FigureDriver::Resolved& FigureDriver::resolved(KeyId key)
{
  refresh();
  if(size_t(key) >= m_resolved.size())
    m_resolved.resize(graph.size());
  Resolved& r = m_resolved[size_t(key)];
  if(r.base != Resolved::Base::eUnset)
    return r;
  const KeyInfo& info = graph.info(key);
  r.base              = Resolved::Base::eNone;
  if(info.morph)
  {
    // the dial a modifier reads (dialOf), else where its file says it rests
    const std::string name = nameOf(info.id);
    int               dial = m_morphs.indexOf(name);
    const auto        file = m_files.dials.find(info.id);
    if(dial < 0 && file != m_files.dials.end())
      dial = m_morphs.indexOf(file->second.stem);
    if(dial >= 0)
    {
      r.base  = Resolved::Base::eDial;
      r.index = dial;
    }
    else if(file != m_files.dials.end())
    {
      r.base = Resolved::Base::eRest;
      r.rest = file->second.rest;
    }
    r.outKind = Resolved::Out::eMorph;
    r.out     = m_morphs.indexOf(name);
    return r;
  }
  if(info.id.empty() || !rig)
    return r;
  pose::ChannelKind kind = pose::ChannelKind::eRotation;
  const bool        posed = pose::parseKind(info.prop, kind);
  r.kind                  = kind;
  r.axis                  = int8_t(pose::axisIndex(info.axis));
  // a posed channel's base by the bone's id first; a joint centre and the rest are 0 for any bone
  if(const int bone = rig->indexOf(info.id, posed); bone >= 0)
  {
    r.base  = posed ? Resolved::Base::eBone : Resolved::Base::eJoint;
    r.index = bone;
  }
  // what a formula writes goes to the bone by name first
  if(const int bone = rig->indexOf(info.id); bone >= 0)
  {
    if(posed)
      r.outKind = Resolved::Out::ePose;
    else if(info.prop == "center_point" && r.axis >= 0 && r.axis < 3)
      r.outKind = Resolved::Out::eCenter;
    r.out = bone;
  }
  return r;
}

KeyId FigureDriver::keyOfChannel(size_t channel)
{
  refresh();
  if(channel >= m_channelKeys.size())
    m_channelKeys.resize(m_morphs.channels().size(), -1);
  if(m_channelKeys[channel] < 0)
  {
    const std::string& name = m_morphs.channels()[channel].name;
    const auto         id   = m_idOf.find(name);
    const KeyId        key  = graph.intern("m:" + (id != m_idOf.end() ? id->second : name));
    m_channelKeys[channel]  = key;
  }
  return m_channelKeys[channel];
}

// A property's value before formulas: a dial, a posed bone channel, 0 for a joint centre's offset.
std::optional<double> FigureDriver::base(KeyId key)
{
  const Resolved& r = resolved(key);
  switch(r.base)
  {
    case Resolved::Base::eDial:
      return m_morphs.channels()[size_t(r.index)].value;
    case Resolved::Base::eRest:
      return r.rest;
    case Resolved::Base::eBone:
      return rig->bones()[size_t(r.index)].posed(r.kind, r.axis);
    case Resolved::Base::eJoint:
      return 0.0;
    default:
      return std::nullopt;
  }
}

// A property as the figure loaded: its dials where they stood then, its bones unposed.
std::optional<double> FigureDriver::baseAtLoad(KeyId key)
{
  const Resolved& r = resolved(key);
  switch(r.base)
  {
    case Resolved::Base::eDial:
      return m_morphs.channels()[size_t(r.index)].loaded;
    case Resolved::Base::eRest:
      return r.rest;
    case Resolved::Base::eBone:
      return r.kind == pose::ChannelKind::eScale ? 1.0 : 0.0;
    case Resolved::Base::eJoint:
      return 0.0;
    default:
      return std::nullopt;
  }
}

void FigureDriver::noteFiles(MorphFiles files)
{
  m_files = std::move(files);
  m_filesVersion++;
  if(m_baselineFormulas >= 0)
    captureBaseline();
  update();
}

bool FigureDriver::loadReaders()
{
  if(m_files.readers.empty())
    return false;
  fv::timing::Accum whole("driver.readers");
  bool              started = false;
  for(auto it = m_files.readers.begin(); it != m_files.readers.end();)
  {
    const morph::Channel* dial = dialOf(it->first);
    if(!dial || (dial->value == dial->rest && m_morphs.weightOf(dial->name) == dial->rest))
    {
      ++it;
      continue;
    }
    for(const MorphFiles::Reader& r : it->second)
    {
      // the channel it has by its id, as a scene's dial has, else the listing's by its file's name
      const std::string byId = nameOf(r.id);
      if(!m_morphs.has(byId) && m_morphs.has(r.stem))
      {
        started = m_morphs.load(r.stem) || started;
        continue;
      }
      // a listing leaves out a hidden name: such a morph joins as run() adds one a formula moves
      morph::ChannelOptions opts;
      opts.group  = "Driven";
      opts.hidden = true;
      m_morphs.addLazy(byId, loader(r.file, byId), opts);
      started = m_morphs.load(byId) || started;
    }
    it = m_files.readers.erase(it);
  }
  return started;
}

void FigureDriver::update()
{
  if(m_running)
  {
    m_again = true;
    return;
  }
  m_running = true;
  try
  {
    do
    {
      m_again = false;
      run();
    } while(m_again);
  }
  catch(...)
  {
    m_running = false;
    throw;
  }
  m_running = false;
}

void FigureDriver::run()
{
  fv::timing::Accum whole("driver.run");
  if(m_baselineFormulas >= 0 && m_baselineFormulas != graph.version)
    captureBaseline();
  auto        value    = graph.evaluatorById([this](KeyId k) { return base(k); });
  const auto& channels = m_morphs.channels();
  // channels that are dialled or applied, in order, then the ones formulas move (indices into
  // `channels`, which a run only adds to)
  m_names.clear();
  m_named.assign(channels.size(), 0);
  const auto addName = [&](int c) {
    if(c < 0)
      return;
    if(size_t(c) >= m_named.size())
      m_named.resize(channels.size(), 0);
    if(!m_named[size_t(c)])
    {
      m_named[size_t(c)] = 1;
      m_names.push_back(size_t(c));
    }
  };
  for(size_t i = 0; i < channels.size(); i++)
    if(channels[i].value != 0 || m_morphs.weightAt(i) != 0)
      addName(int(i));
  m_driven.clear();
  m_centers.assign(rig ? rig->bones().size() : 0, Vec3{});
  {
    fv::timing::Accum outputsTime("driver.outputs");
    // what the formulas write, as it stood when the run began
    const size_t outputs = graph.outputs().size();
    for(size_t o = 0; o < outputs; o++)
    {
      const KeyId    key = graph.outputs()[o];
      const Resolved r   = resolved(key);
      if(r.outKind == Resolved::Out::eMorph)
      {
        if(r.out >= 0)
        {
          addName(r.out);
          continue;
        }
        const std::string& id   = graph.info(key).id;
        const std::string  name = nameOf(id);
        if(!m_morphs.has(name))
        {
          // a morph a formula moves that isn't loaded: worth loading once it moves
          const std::string*         known = graph.fileOf(id);
          std::optional<std::string> found;
          if(!known)
            found = m_source->find(id);
          const std::string* file = known ? known : (found ? &*found : nullptr);
          if(!file)
            continue;
          const auto v = value(key);
          if(!v || *v == 0)
            continue;
          morph::ChannelOptions opts;
          opts.group  = "Driven";
          opts.hidden = true;
          m_morphs.addLazy(name, loader(*file, name), opts);
        }
        addName(m_morphs.indexOf(name));
        continue;
      }
      if(r.outKind == Resolved::Out::eNone)
        continue;
      const auto v = value(key);
      if(!v)
        continue;
      if(r.outKind == Resolved::Out::ePose)
      {
        const double d = *v - rig->bones()[size_t(r.out)].posed(r.kind, r.axis);
        if(std::abs(d) > kEps)
          m_driven.push_back({r.out, r.kind, r.axis, d});
        continue;
      }
      const double d = *v - (size_t(key) < m_baseline.size() ? m_baseline[size_t(key)] : 0.0);
      if(std::abs(d) <= kEps)
        continue;
      Vec3& c = m_centers[size_t(r.out)];
      (r.axis == 0 ? c.x : r.axis == 1 ? c.y : c.z) = d;
    }
  }
  m_weights.clear();
  {
    fv::timing::Accum weightsTime("driver.weights");
    for(const size_t c : m_names)
    {
      const auto   w      = value(keyOfChannel(c));
      const double weight = w ? *w : channels[c].value;
      if(std::abs(weight - m_morphs.weightAt(c)) > kEps)
        m_weights.emplace_back(channels[c].name, weight);
    }
  }
  if(!m_weights.empty())
  {
    fv::timing::Accum applyTime("driver.applyWeights");
    m_morphs.applyWeights(m_weights);
  }
  if(rig)
  {
    fv::timing::Accum rigTime("driver.rig");
    rig->setDriven(m_driven);
    // a bone's entry holds an axis over kEps, so a centre that stays at 0 has none
    m_centerList.clear();
    for(size_t b = 0; b < m_centers.size(); b++)
      if(m_centers[b].x != 0 || m_centers[b].y != 0 || m_centers[b].z != 0)
        m_centerList.push_back({int(b), m_centers[b]});
    rig->setCenterOffsets(m_centerList);
    // controllers that bend joints want the correctives as much as posing does
    if(!m_driven.empty() && onDriven)
      onDriven();
  }
}

void FigureDriver::captureBaseline()
{
  auto value         = graph.evaluatorById([this](KeyId k) { return baseAtLoad(k); });
  m_baselineFormulas = graph.version;
  m_baseline.assign(graph.size(), 0.0);
  const size_t outputs = graph.outputs().size();
  for(size_t o = 0; o < outputs; o++)
  {
    const KeyId key = graph.outputs()[o];
    if(graph.info(key).prop != "center_point")
      continue;
    if(const auto v = value(key); v && *v != 0)
      m_baseline[size_t(key)] = *v;
  }
}

}  // namespace fv::pose
