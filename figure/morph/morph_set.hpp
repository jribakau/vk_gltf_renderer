#pragma once

// Morphs: sparse deltas applied into the drawn vertices of a figure's meshes. Cost is proportional
// to the vertices actually moved by non-zero morphs. A port of src/lib/morphs/MorphSet.ts, the CPU
// backend (DSON morphs are always CPU: a Genesis figure carries thousands, which no GPU morph target
// set could hold).
//
// A dial (channel) is a name with a value, and bindings to the targets it moves on meshes. Files of
// morphs are not read until their dial is first set: a lazy channel's loader runs on the thread
// pool and its result is taken on the owner's thread by pump(). A driver (pose/figure_driver.hpp)
// turns dial values into the weights targets are applied at, running the figure's formulas.
//
// One thread owns a MorphSet; only the loaders (see Loader) run elsewhere.

#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <figure/dson/morphs.hpp>
#include <figure/mesh/normals.hpp>

namespace fv::morph {

struct Channel
{
  std::string name, label, group;
  double      value = 0, min = 0, max = 1;
  // not parsed yet: deltas load on first use (DSON morph files)
  bool lazy = false;
  // driven by something else (joint correctives follow bone rotations); not shown as a dial
  bool hidden = false;
};

struct ChannelOptions
{
  std::optional<std::string> label, group;
  std::optional<double>      value, min, max;
  bool                       hidden = false;
};

// What a lazy channel's file gave: the targets it moves. A channel whose file has none is removed,
// unless `keep` (a controller: a dial that drives other morphs or bones through formulas).
struct LazyResult
{
  std::vector<dson::MorphPart> parts;
  bool                         keep = false;
  std::optional<std::string>   label;
  std::optional<double>        min, max;
};

// Loading a channel is two steps: `Loader` runs on a pool thread, and returns what to do with its
// result on the owner's thread (`Finish`).
using Finish = std::function<LazyResult()>;
using Loader = std::function<Finish()>;

// Turns dial values into the weights the morphs are applied at: DAZ's formulas, by which controllers
// drive morphs and bones drive correctives. A MorphSet with a driver asks it after every dial
// change; the driver answers through applyWeights().
class Driver
{
public:
  virtual ~Driver() = default;
  virtual void update() = 0;
};

// Vertices of one mesh whose position or normal changed since the last takeChanges().
struct MeshChange
{
  int                   mesh = -1;
  bool                  all  = false;
  std::vector<uint32_t> indices;  // drawn vertices, sorted; meaningless when `all`
};

class MorphSet
{
public:
  // `geo` finds the built geometry of a mesh index; it must stay valid for the set's life.
  explicit MorphSet(std::function<mesh::BuiltMesh*(int)> geo);

  const std::vector<Channel>& channels() const { return m_channels; }
  Driver*                     driver = nullptr;

  // Number of user-facing morphs (hidden, formula-driven ones excluded).
  int size() const;

  // Adds a morph target to a channel, making the channel when it is new.
  void addTarget(const std::string& name, int mesh, dson::TargetPtr target, const ChannelOptions& opts = {});
  // Registers a morph whose deltas are only read the first time it is set to a non-zero value.
  void addLazy(const std::string& name, Loader loader, const ChannelOptions& opts = {});

  const Channel* channel(const std::string& name) const;
  // A channel's dial value: what its slider shows.
  double get(const std::string& name) const;
  // The weight a channel is applied at: its dial, or what the driver made of it.
  double weightOf(const std::string& name) const;
  // Whether a channel with this name exists, loaded or not.
  bool has(const std::string& name) const;

  void set(const std::string& name, double value);
  // Sets several dials and reports them as one change.
  void setMany(const std::vector<std::pair<std::string, double>>& values);
  // Applies channels at the given weights; the driver calls this.
  void applyWeights(const std::vector<std::pair<std::string, double>>& values);
  void resetAll();

  // Takes the lazy loads that finished, runs the driver for them (once they have all arrived, or
  // after a short wait if more keep coming) and flushes what moved. Call every frame.
  void pump();
  // Waits for every load, including the ones those start, and the driver after them.
  void settled();
  bool loading() const { return !m_loading.empty(); }

  // Applies pending morph changes to the drawn vertices of the meshes.
  void flush();

  // The vertices changed since last asked; clears them.
  std::vector<MeshChange> takeChanges();

  // Active channels first, then by group and label.
  void sort();

  // Counts up when channels were added, removed or relabelled.
  int listVersion() const { return m_listVersion; }
  // Loads that failed, worth telling the user about (missing files are reported separately).
  std::vector<std::string> takeErrors();

  // Rebuilds the vertex buffers from the base the next time they flush (a fit changed the base).
  void markRebuild(int mesh);

private:
  struct Binding
  {
    int             mesh;
    dson::TargetPtr target;
  };
  struct Active
  {
    dson::TargetPtr target;
    double          weight = 0;
  };
  struct MeshState
  {
    std::vector<float> basePos, baseNrm;
    std::unordered_map<const mesh::RenderTarget*, Active> active;
    // the weights the position buffer holds now; a flush adds the difference
    std::unordered_map<const mesh::RenderTarget*, Active> applied;
    int                                                   sinceRebuild = 0;
    bool                                                  dirty        = false;
    std::unique_ptr<mesh::NormalWelder>                   welder;
    std::vector<uint8_t>                                  mark;
  };

  Channel* channelMut(const std::string& name);
  void     reindex();
  void     bind(const std::string& name, Binding b, double value, const ChannelOptions& opts);
  MeshState& ensureCpu(int mesh);
  // writes a channel's weight into its bindings; returns false for an unknown channel
  bool apply(const std::string& name, double value);
  void ensureLoaded(const std::string& name);
  void finishLoad(const std::string& name, std::future<Finish>& fut);
  void runDriver();
  void rebuild(int mesh, MeshState& st);
  void update(int mesh, MeshState& st);
  void change(int mesh, bool all, const std::vector<uint32_t>& moved, const std::vector<uint32_t>& written);

  std::function<mesh::BuiltMesh*(int)>                  m_geo;
  std::vector<Channel>                                  m_channels;
  std::unordered_map<std::string, size_t>               m_index;
  // the weight each channel is applied at, when a driver set one other than its dial
  std::unordered_map<std::string, double>               m_weights;
  std::unordered_map<std::string, std::vector<Binding>> m_bindings;
  std::unordered_map<int, MeshState>                    m_cpu;
  std::unordered_map<std::string, Loader>               m_lazy;
  std::map<std::string, std::future<Finish>>            m_loading;
  bool                                                  m_driverDirty = false;
  std::chrono::steady_clock::time_point                 m_driverSince;
  bool                                                  m_running = false;
  int                                                   m_listVersion = 0;
  std::vector<std::string>                              m_errors;
  std::map<int, MeshChange>                             m_changes;
};

}  // namespace fv::morph
