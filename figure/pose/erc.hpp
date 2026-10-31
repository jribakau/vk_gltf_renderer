#pragma once

// DAZ's ERC (Enhanced Remote Control): formulas that set one property from others. A figure's
// controllers, joint correctives, expressions and hand poses are all formulas: a controller dial
// drives shape morphs, a pose control drives finger rotations, a bone rotation drives a corrective
// morph, and a body shape moves the joint centres. One graph per figure holds every formula of the
// files loaded so far and evaluates them together, in dependency order. A port of
// src/lib/pose/erc.ts.
//
// A property's value is (base + the sum-stage formulas) x the mult-stage formulas, clamped when its
// channel is: the base is a morph's dial, a bone channel's posed value, or 0 for a joint centre,
// whose formulas give an offset. Properties are keyed:
//   m:<modifier id>                     a modifier's value (morph, controller, switch)
//   b:<bone id>:<property>/<axis>       rotation, translation or scale (x, y, z, general), and
//                                       center_point, end_point or orientation (x, y, z)

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/pose/formulas.hpp>

namespace fv::pose {

using KeyId = int;

// A property key taken apart.
struct KeyInfo
{
  bool        morph = false;
  // the modifier id, or the bone id
  std::string id;
  // bones: "rotation", "center_point", ... and the axis
  std::string prop, axis;
};

// The property a formula url names, and the file it names when it gives one explicitly
// ("/data/...#FBMHeavy?value"); a bare "#FBMHeavy?value" means a modifier of the same figure,
// wherever it is.
struct UrlKey
{
  std::string                key;
  std::optional<std::string> file;
};
std::optional<UrlKey> keyOf(const std::string& url, const std::string& file);

struct ChannelMeta
{
  // value with no dial: a switch such as "JCMs On" is 1
  double base    = 0;
  bool   clamped = false;
  double lo = 0, hi = 0;
};

// Looks a property's base up: a dial, a posed bone value; nullopt for a property nothing knows.
using BaseOf = std::function<std::optional<double>(const KeyInfo&)>;

class ErcGraph
{
public:
  // Registers every modifier of a DSON document: its channel and its formulas. Once per file.
  void addDoc(const dson::Json& doc, const std::string& file);
  // One modifier by its parts, as a skim holds them: its channel and its array of formulas, each null
  // for none.
  void addModifier(const std::string& id, const std::string& file, const ChannelMeta* meta, const dson::Json* formulas);

  // The id of a property key, made on first use.
  KeyId intern(const std::string& key);
  // The key with no side effect, or -1.
  KeyId find(const std::string& key) const;
  const KeyInfo&     info(KeyId k) const { return m_keys[size_t(k)].info; }
  const std::string& name(KeyId k) const { return m_keys[size_t(k)].name; }

  // Every property some formula writes, in the order the formulas were added.
  const std::vector<KeyId>& outputs() const { return m_outputs; }
  bool                      has(KeyId k) const { return k >= 0 && !m_keys[size_t(k)].formulas.empty(); }

  const ChannelMeta* meta(const std::string& id) const;
  // The file a modifier lives in, when a registered file or an explicit url said so.
  const std::string* fileOf(const std::string& id) const;

  // What a property's formulas read, and whether a mult stage does, which only scales what else moved it.
  struct Input
  {
    KeyId key  = -1;
    bool  mult = false;
  };
  std::vector<Input> inputsOf(KeyId k) const;

  // Evaluates properties on demand against `base`, each once (memoised for this evaluation).
  class Evaluator
  {
  public:
    Evaluator(ErcGraph& graph, BaseOf base)
        : m_graph(graph)
        , m_base(std::move(base))
    {
    }
    // nullopt for a property nothing knows: no base, no channel and no formula
    std::optional<double> operator()(KeyId key);
    std::optional<double> operator()(const std::string& key) { return (*this)(m_graph.intern(key)); }

  private:
    enum : uint8_t
    {
      eNone,
      eBusy,
      eDone,
      eUnknown,
    };
    ErcGraph&             m_graph;
    BaseOf                m_base;
    std::vector<uint8_t>  m_state;
    std::vector<double>   m_value;
  };
  Evaluator evaluator(BaseOf base) { return Evaluator(*this, std::move(base)); }

  // Bumped whenever formulas are added, so cached results can tell.
  int version = 0;

private:
  struct Compiled
  {
    KeyId           output = -1;
    bool            mult   = false;
    std::vector<Op> ops;
    // the properties its pushes read, by input index; -1 for an input this graph can't read
    std::vector<KeyId> inputs;
  };
  struct Key
  {
    std::string           name;
    KeyInfo               info;
    std::vector<Compiled> formulas;
  };

  void addFormula(const dson::Json& f, const std::string& file);

  std::vector<Key>                                 m_keys;
  std::unordered_map<std::string, KeyId>           m_ids;
  std::vector<KeyId>                               m_outputs;
  std::unordered_map<std::string, ChannelMeta>     m_metas;
  std::unordered_map<std::string, std::string>     m_files;
  std::unordered_set<std::string>                  m_docs;
};

}  // namespace fv::pose
