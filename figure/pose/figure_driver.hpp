#pragma once

// Runs a figure's ERC formulas (erc.hpp) against its dials and pose, and applies what they say:
// morph weights to the MorphSet, bone channel offsets and joint centre offsets to the PoseRig. A
// port of src/lib/pose/FigureDriver.ts.
//
// The graph grows as files load. A dial's file is read when the dial is first set; what its
// formulas write and isn't a channel yet becomes a hidden one, loaded from the file a url names or
// from the Morphs folder, and so on down the chain until nothing new is needed. A morph can also say
// in its own file what drives it, which nothing about that dial says: MorphFiles holds those, and
// they're read when the dial moves. The joint correctives join on the first pose change.

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

#include <figure/dson/morphs.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/pose/erc.hpp>
#include <figure/pose/pose_rig.hpp>

namespace fv::pose {

// Where a figure's modifier files come from and how their deltas become targets. doc() and
// targets() run on pool threads.
class FigureSource
{
public:
  virtual ~FigureSource() = default;
  virtual std::shared_ptr<const dson::Doc> doc(const std::string& file) = 0;
  // the morph targets of a document's modifiers that carry deltas
  virtual std::vector<dson::MorphDef> targets(const dson::Doc& doc, const std::string& file) = 0;
  // an installed modifier file by modifier id, when a Morphs folder lists one
  virtual std::optional<std::string> find(const std::string& id) = 0;
};

// What the skims of a Morphs folder's files say (dson::MorphInfo) that formulas need before the files
// are read: formulas go by modifier ids, and the listing by file names.
struct MorphFiles
{
  struct Reader
  {
    std::string file, stem, id;
  };
  // a dial's id -> the files that read it
  std::unordered_map<std::string, std::vector<Reader>> readers;
  struct Dial
  {
    // the channel a listing would have for it, unless its name is hidden
    std::string stem;
    double      rest = 0;
  };
  // a modifier's id -> the dial its file holds
  std::unordered_map<std::string, Dial> dials;

  void add(const std::string& file, const std::string& stem, const dson::MorphInfo& info);
};

class FigureDriver : public morph::Driver
{
public:
  FigureDriver(morph::MorphSet& morphs, std::shared_ptr<FigureSource> source);

  ErcGraph graph;
  PoseRig* rig = nullptr;
  // Called when formulas bend joints: the figure loads its joint correctives then.
  std::function<void()> onDriven;

  // A loader for a channel whose modifier lives in `file`: its deltas, and its formulas into the graph.
  morph::Loader loader(const std::string& file, const std::string& name);

  // Re-evaluates every formula and applies what changed. Re-entrant calls run once more at the end.
  void update() override;

  // Takes what the files say, and runs the formulas with it: a character's morph multiplied by a dial
  // whose file isn't read stands at 0 now, not at whatever the formula says without it.
  void noteFiles(MorphFiles files);
  // Starts reading the files that read a dial standing away from its rest, and forgets those: they
  // only have to be read once. A dial at its rest brings none, whatever that rest is: a file's default
  // is no move. True when a load started.
  bool loadReaders();

  // Takes the joint centres the formulas give at the dials the figure loaded with as where its
  // skeleton already is: a character's scene file stores its bones at the centres its shape moved them
  // to. Formulas that join later are counted the same way, as the file's centres include them too.
  void captureBaseline();

  // The channel name for a modifier id, where a file's modifier and the channel made for it differ.
  std::string nameOf(const std::string& id) const;

private:
  // What a key of the graph reads and writes here, worked out the first time it is asked for and kept
  // while the channels, the aliases, the files' dials and the rig stay as they were (Stamp): a run
  // reads thousands of keys, each once a name looked up in several tables.
  struct Resolved
  {
    enum class Base : uint8_t
    {
      eUnset,
      eNone,   // nothing knows it
      eDial,   // a channel's value (`index`)
      eRest,   // a dial no channel holds, where its file says it rests (`rest`)
      eBone,   // a bone channel's posed value (`index`, `kind`, `axis`)
      eJoint,  // a joint centre or another bone property: 0
    };
    enum class Out : uint8_t
    {
      eNone,
      eMorph,   // a morph a formula moves: the channel `out` (-1: not one yet)
      ePose,    // a bone channel: the bone `out`, by name first
      eCenter,  // a joint centre's axis of the bone `out`
    };
    Base               base  = Base::eUnset;
    Out                outKind = Out::eNone;
    pose::ChannelKind  kind  = pose::ChannelKind::eRotation;
    int8_t             axis  = -1;  // pose::axisIndex
    int                index = -1;
    int                out   = -1;
    double             rest  = 0;
  };
  struct Stamp
  {
    int            layout = -1, aliases = -1, files = -1;
    const PoseRig* rig    = nullptr;
    bool           operator==(const Stamp&) const = default;
  };
  Stamp           stamp() const;
  // forgets what was resolved when the stamp moved
  void            refresh();
  const Resolved& resolved(KeyId key);
  // the graph key a channel's weight is read from (m:<its modifier id>)
  KeyId keyOfChannel(size_t channel);

  // the channel that holds a modifier's dial: by its id, or the listing's before its file has said so
  const morph::Channel* dialOf(const std::string& id) const;
  std::optional<double> base(KeyId key);
  std::optional<double> baseAtLoad(KeyId key);
  void                  run();

  morph::MorphSet&                      m_morphs;
  std::shared_ptr<FigureSource>         m_source;
  // modifier id -> channel name, and back
  std::unordered_map<std::string, std::string> m_alias, m_idOf;
  int                                          m_aliasVersion = 0;
  MorphFiles                                   m_files;
  int                                          m_filesVersion = 0;
  // joint centre values the figure loaded with (its scene stored the moved centres), by key: offsets
  // count from here
  std::vector<double>           m_baseline;
  // the graph's version the baseline was taken at, -1 for a figure with none
  int                           m_baselineFormulas = -1;
  bool                          m_running = false;
  bool                          m_again   = false;

  Stamp                 m_resolvedFor;
  std::vector<Resolved> m_resolved;
  std::vector<KeyId>    m_channelKeys;
  // a run's working space
  std::vector<size_t>                         m_names;
  std::vector<uint8_t>                        m_named;
  std::vector<DrivenValue>                    m_driven;
  std::vector<Vec3>                           m_centers;  // by bone
  std::vector<CenterOffset>                   m_centerList;
  std::vector<std::pair<std::string, double>> m_weights;
};

}  // namespace fv::pose
