#include <figure/pose/morph_reach.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <set>

#include <figure/io/timing.hpp>
#include <figure/pose/erc.hpp>

namespace fv::pose {

namespace {

// Who reads a property: the formula's output, and whether by a mult stage.
struct Forward
{
  KeyId to   = -1;
  bool  mult = false;
};
using Readers = std::unordered_map<KeyId, std::vector<Forward>>;

// A bone unposed and a joint centre's offset, as the driver's base has them with no pose. A morph has
// no answer here, so the graph reads the channel's rest.
std::optional<double> atRest(const KeyInfo& key)
{
  if(key.morph || key.id.empty())
    return std::nullopt;
  return key.prop == "scale" ? 1.0 : 0.0;
}

// What moving a property can move, itself included, and whether by adding all the way: past a mult
// stage it only scales what else moved it.
std::unordered_map<KeyId, bool> walk(KeyId from, const Readers& readers)
{
  std::unordered_map<KeyId, bool>     out{{from, true}};
  std::vector<std::pair<KeyId, bool>> todo{{from, true}};
  while(!todo.empty())
  {
    const auto [key, added] = todo.back();
    todo.pop_back();
    const auto it = readers.find(key);
    if(it == readers.end())
      continue;
    for(const Forward& f : it->second)
    {
      const bool adds        = added && !f.mult;
      const auto [at, first] = out.try_emplace(f.to, adds);
      if(!first)
      {
        // seen as well as this, or better: its readers have been told
        if(at->second || !adds)
          continue;
        at->second = true;
      }
      todo.emplace_back(f.to, adds);
    }
  }
  return out;
}

}  // namespace

DialReach auditReach(const std::vector<MorphFact>& facts)
{
  timing::Scope whole("morph reach: formulas", 1);
  ErcGraph      graph;
  // a modifier's key -> its size. Two copies of a file are one modifier, whose formulas count once.
  std::unordered_map<KeyId, double> sizes;
  for(const MorphFact& f : facts)
  {
    const dson::MorphInfo& info = f.info;
    if(info.id.empty())
      continue;
    const auto [known, first] = sizes.try_emplace(graph.intern("m:" + info.id), f.size);
    if(!first)
    {
      known->second = std::max(known->second, f.size);
      continue;
    }
    ChannelMeta meta;
    meta.base    = info.rest;
    meta.clamped = info.clamped;
    meta.lo      = info.lo;
    meta.hi      = info.hi;
    // an alias's value is another channel's, as the driver has it
    graph.addModifier(info.id, f.file, info.alias ? nullptr : &meta, info.formulas.get());
  }
  Readers readers;
  for(const KeyId key : graph.outputs())
    for(const ErcGraph::Input& in : graph.inputsOf(key))
      readers[in.key].push_back({key, in.mult});

  auto       rest  = graph.evaluator(atRest);
  const auto audit = [&](const dson::MorphInfo& info) {
    const auto dialAt = [&](double at) {
      return graph.evaluator([&id = info.id, at](const KeyInfo& k) -> std::optional<double> {
        if(k.morph && k.id == id)
          return at;
        return atRest(k);
      });
    };
    // as a channel's range before its file says otherwise (MorphSet::addLazy)
    std::array<ErcGraph::Evaluator, 2> ends = {dialAt(info.min.value_or(0)), dialAt(info.max.value_or(1))};
    morph::Reach                       out;
    // at each end: the furthest a morph moves, and each bone's centre, squared
    std::array<double, 2>                                  skin{};
    std::unordered_map<std::string, std::array<double, 2>> centres;
    std::set<std::string>                                  moved, scaled;
    for(const auto& [key, added] : walk(graph.intern("m:" + info.id), readers))
    {
      const KeyInfo& k    = graph.info(key);
      const double   from = rest(key).value_or(0);
      for(size_t e = 0; e < ends.size(); e++)
      {
        const double d = std::abs(ends[e](key).value_or(0) - from);
        if(k.morph)
        {
          if(const auto size = sizes.find(key); size != sizes.end())
            skin[e] = std::max(skin[e], d * size->second);
        }
        else if(k.prop == "center_point")
          centres[k.id][e] += d * d;
        else if(k.prop == "rotation")
          out.deg = std::max(out.deg, d);
      }
      if(k.morph)
      {
        if(const auto size = sizes.find(key); size != sizes.end() && size->second > 0)
          (added ? out.morphs : out.scales)++;
      }
      else if(k.prop == "rotation" || k.prop == "translation" || k.prop == "scale")
        (added ? moved : scaled).insert(k.id);
    }
    out.bones = int(moved.size());
    out.scales += int(std::count_if(scaled.begin(), scaled.end(), [&](const std::string& bone) { return !moved.count(bone); }));
    for(size_t e = 0; e < ends.size(); e++)
    {
      double centre = 0;
      for(const auto& [bone, squared] : centres)
        centre = std::max(centre, std::sqrt(squared[e]));
      out.cm = std::max(out.cm, skin[e] + centre);
    }
    return out;
  };

  DialReach                                     out;
  std::unordered_map<std::string, morph::Reach> byId;
  for(const MorphFact& f : facts)
  {
    const dson::MorphInfo& info = f.info;
    // not a dial anyone sees; and a file read short can't say what it holds, so it stays unknown
    if(info.id.empty() || info.alias || info.helper || !info.complete || dson::hiddenMorph(f.stem))
      continue;
    auto it = byId.find(info.id);
    if(it == byId.end())
      it = byId.emplace(info.id, audit(info)).first;
    // the names MorphNotes gives its files' dials
    out.emplace(info.id, it->second);
    out[f.stem] = it->second;
  }
  return out;
}

}  // namespace fv::pose
