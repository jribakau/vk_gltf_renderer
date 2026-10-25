#include <figure/pose/figure_driver.hpp>
#include <figure/io/timing.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace fv::pose {

using dson::Json;

namespace {
constexpr double kEps = 1e-6;

bool poseProp(const std::string& p)
{
  return p == "rotation" || p == "translation" || p == "scale";
}
}  // namespace

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
      out.min = lo && lo->is_number() && limited ? std::max(-1.0, lo->get<double>()) : (out.parts.empty() ? 0.0 : out.parts[0].min);
      out.max = hi && hi->is_number() && limited ? std::min(2.0, hi->get<double>()) : (out.parts.empty() ? 1.0 : out.parts[0].max);
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

// A property's value before formulas: a dial, a posed bone channel, 0 for a joint centre's offset.
std::optional<double> FigureDriver::base(const KeyInfo& key)
{
  if(key.morph)
  {
    const morph::Channel* c = m_morphs.channel(nameOf(key.id));
    return c ? std::optional<double>(c->value) : std::nullopt;
  }
  if(key.id.empty())
    return std::nullopt;
  if(poseProp(key.prop))
    return rig ? rig->posed(key.id, key.prop, key.axis) : std::nullopt;
  return rig && rig->hasBone(key.id) ? std::optional<double>(0) : std::nullopt;
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
  auto              value = graph.evaluator([this](const KeyInfo& k) { return base(k); });
  // channels that are dialled or applied, in order, then the ones formulas move
  std::vector<std::string>        names;
  std::unordered_set<std::string> seen;
  auto                            addName = [&](const std::string& n) {
    if(seen.insert(n).second)
      names.push_back(n);
  };
  for(const morph::Channel& ch : m_morphs.channels())
    if(ch.value != 0 || m_morphs.weightOf(ch.name) != 0)
      addName(ch.name);
  std::map<std::string, double> driven;
  std::map<std::string, Vec3>   centers;
  const std::vector<KeyId>      outputs = graph.outputs();
  {
  fv::timing::Accum outputsTime("driver.outputs");
  for(KeyId key : outputs)
  {
    const KeyInfo info = graph.info(key);
    if(info.morph)
    {
      const std::string id   = info.id;
      const std::string name = nameOf(id);
      if(!m_morphs.has(name))
      {
        // a morph a formula moves that isn't loaded: worth loading once it moves
        const std::string* known = graph.fileOf(id);
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
      addName(name);
      continue;
    }
    const PoseBone* pb = info.id.empty() || !rig ? nullptr : rig->get(info.id);
    if(!pb)
      continue;
    const auto v = value(key);
    if(!v)
      continue;
    if(poseProp(info.prop))
    {
      const double d = *v - rig->posed(pb->id, info.prop, info.axis).value_or(0);
      if(std::abs(d) > kEps)
        driven[pb->id + "|" + info.prop + "|" + info.axis] = d;
    }
    else if(info.prop == "center_point" && (info.axis == "x" || info.axis == "y" || info.axis == "z"))
    {
      auto           bit = m_baseline.find(graph.name(key));
      const double   d   = *v - (bit == m_baseline.end() ? 0.0 : bit->second);
      if(std::abs(d) <= kEps)
        continue;
      Vec3& c = centers[pb->id];
      (info.axis == "x" ? c.x : info.axis == "y" ? c.y : c.z) = d;
    }
  }
  }
  std::vector<std::pair<std::string, double>> weights;
  {
  fv::timing::Accum weightsTime("driver.weights");
  for(const std::string& name : names)
  {
    auto              it = m_idOf.find(name);
    const std::string id = it != m_idOf.end() ? it->second : name;
    const auto        w  = value("m:" + id);
    const double      weight = w ? *w : m_morphs.get(name);
    if(std::abs(weight - m_morphs.weightOf(name)) > kEps)
      weights.emplace_back(name, weight);
  }
  }
  if(!weights.empty())
  {
    fv::timing::Accum applyTime("driver.applyWeights");
    m_morphs.applyWeights(weights);
  }
  if(rig)
  {
    fv::timing::Accum rigTime("driver.rig");
    rig->setDriven(driven);
    rig->setCenterOffsets(centers);
    // controllers that bend joints want the correctives as much as posing does
    if(!driven.empty() && onDriven)
      onDriven();
  }
}

void FigureDriver::captureBaseline()
{
  auto value = graph.evaluator([this](const KeyInfo& k) { return base(k); });
  m_baseline.clear();
  const std::vector<KeyId> outputs = graph.outputs();
  for(KeyId key : outputs)
  {
    if(graph.info(key).prop != "center_point")
      continue;
    const auto v = value(key);
    if(v && *v != 0)
      m_baseline[graph.name(key)] = *v;
  }
}

}  // namespace fv::pose
