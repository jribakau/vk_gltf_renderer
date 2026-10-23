#pragma once

// Runs a figure's ERC formulas (erc.hpp) against its dials and pose, and applies what they say:
// morph weights to the MorphSet, bone channel offsets and joint centre offsets to the PoseRig. A
// port of src/lib/pose/FigureDriver.ts.
//
// The graph grows as files load. A dial's file is read when the dial is first set; what its
// formulas write and isn't a channel yet becomes a hidden one, loaded from the file a url names or
// from the Morphs folder, and so on down the chain until nothing new is needed. The joint
// correctives join on the first pose change.

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

  // Takes the joint centres the formulas give now as where the figure's skeleton already is: a
  // character's scene file stores its bones at the centres its shape moved them to.
  void captureBaseline();

  // The channel name for a modifier id, where a file's modifier and the channel made for it differ.
  std::string nameOf(const std::string& id) const;

private:
  std::optional<double> base(const KeyInfo& key);
  void                  run();

  morph::MorphSet&                      m_morphs;
  std::shared_ptr<FigureSource>         m_source;
  // modifier id -> channel name, and back
  std::unordered_map<std::string, std::string> m_alias, m_idOf;
  // joint centre values the figure loaded with (its scene stored the moved centres): offsets count from here
  std::map<std::string, double> m_baseline;
  bool                          m_running = false;
  bool                          m_again   = false;
};

}  // namespace fv::pose
