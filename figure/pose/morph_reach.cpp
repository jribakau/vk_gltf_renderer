#include <figure/pose/morph_reach.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <set>
#include <unordered_set>

#include <figure/io/timing.hpp>
#include <figure/pose/erc.hpp>

namespace fv::pose {

namespace {

// A dial moving less than this at its ends is looked at on the way too, in this many steps: its formulas
// may rise and be back at 0 by the end of its range, as Neck Swallow's splines are. One moving more has
// answered what the panel asks, and each look runs all it reaches.
constexpr double kLookCloserCm = 1;
constexpr int    kRangeSteps   = 8;

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
  // the modifiers a formula moves by the name of a dial on the list, which a worn item's copy can follow
  std::unordered_set<KeyId> listed;
  for(const MorphFact& f : facts)
  {
    const dson::MorphInfo& info = f.info;
    // An alias has its dial's id and nothing of its dial, whose value is another channel's as the driver
    // has it. Its file usually lists before the real one, which is the one that counts.
    if(info.id.empty() || info.alias)
      continue;
    const KeyId key = graph.intern("m:" + info.id);
    if(f.stem == info.id && !dson::hiddenMorph(f.stem))
      listed.insert(key);
    const auto [known, first] = sizes.try_emplace(key, f.size);
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
    graph.addModifier(info.id, f.file, &meta, info.formulas.get());
  }
  Readers readers;
  for(const KeyId key : graph.outputs())
    for(const ErcGraph::Input& in : graph.inputsOf(key))
      readers[in.key].push_back({key, in.mult});

  auto       rest  = graph.evaluator(atRest);
  const auto audit = [&](const dson::MorphInfo& info) {
    const KeyId           dial    = graph.intern("m:" + info.id);
    const auto            reached = walk(dial, readers);
    morph::Reach          out;
    std::set<std::string> moved, scaled, drives;
    for(const auto& [key, added] : reached)
    {
      const KeyInfo& k = graph.info(key);
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

    // the furthest a morph moves with the dial here, and each bone's centre, squared
    const auto look = [&](double at) {
      auto value = graph.evaluator([&id = info.id, at](const KeyInfo& k) -> std::optional<double> {
        if(k.morph && k.id == id)
          return at;
        return atRest(k);
      });
      double                                  skin = 0;
      std::unordered_map<std::string, double> centres;
      for(const auto& [key, added] : reached)
      {
        const KeyInfo& k = graph.info(key);
        const double   d = std::abs(value(key).value_or(0) - rest(key).value_or(0));
        if(k.morph)
        {
          const auto size = sizes.find(key);
          if(size == sizes.end())
            continue;
          skin = std::max(skin, d * size->second);
          if(d > 0 && size->second == 0 && key != dial && listed.count(key))
            drives.insert(k.id);
        }
        else if(k.prop == "center_point")
          centres[k.id] += d * d;
        else if(k.prop == "rotation")
          out.deg = std::max(out.deg, d);
      }
      double centre = 0;
      for(const auto& [bone, squared] : centres)
        centre = std::max(centre, std::sqrt(squared));
      out.cm = std::max(out.cm, skin + centre);
    };
    // as a channel's range before its file says otherwise (MorphSet::addLazy)
    const double lo = info.min.value_or(0), hi = info.max.value_or(1);
    look(lo);
    look(hi);
    if(out.cm < kLookCloserCm && out.deg == 0 && reached.size() > 1)
      for(int step = 1; step < kRangeSteps; step++)
        look(lo + (hi - lo) * step / kRangeSteps);
    out.drives.assign(drives.begin(), drives.end());
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
