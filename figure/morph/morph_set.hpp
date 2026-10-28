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

// A mesh that takes on the morphs of another, such as clothing worn over a figure's body: a channel
// that moves `source` moves `mesh` too, with the follower's version of each target.
struct Follower
{
  int mesh = -1, source = -1;
  // The follower's version of one of the source's targets, or null when it moves nothing. Called
  // from the owner's thread and, several at once, from pool threads (the surface binding is made
  // before).
  std::function<dson::TargetPtr(const mesh::RenderTarget&)> project;
  // The follower's own targets for a channel, used instead of projecting (an item ships fitted
  // copies of the body shapes); an invalid future when it has none. Read on a pool thread.
  std::function<std::shared_future<std::vector<dson::TargetPtr>>(const std::string&)> own;
  // Projections made ahead, off the owner's thread, of the source's targets dialled in when the
  // wear began: (source target, its projection, null when it moves nothing). addFollower() takes
  // them instead of calling project() for those; any other target is projected as usual.
  std::vector<std::pair<dson::TargetPtr, dson::TargetPtr>> preprojected;
};

// Vertices of one mesh whose position or normal changed since the last takeChanges().
struct MeshChange
{
  int                   mesh = -1;
  bool                  all  = false;
  std::vector<uint32_t> indices;  // drawn vertices, sorted; meaningless when `all`
  // only its morphed cage moved (MorphSet::cageOnly): its drawn vertices are as they were
  bool cage = false;
  // only its drawn vertices caught up with a cage change already reported (MorphSet::catchUp):
  // nothing moved that a reader of shapes (dForce, the collider) has not been told of
  bool caughtUp = false;
};

class MorphSet
{
public:
  // `geo` finds the built geometry of a mesh index; it must stay valid for the set's life.
  explicit MorphSet(std::function<mesh::BuiltMesh*(int)> geo);

  const std::vector<Channel>& channels() const { return m_channels; }
  Driver*                     driver = nullptr;
  // Meshes whose drawn vertices something else makes from their cage (the GPU refines them:
  // figureGpuRefine, daz-viewer docs/gpu-figures-plan.md). A flush keeps their morphed cage and
  // reports a cage change; their drawn positions and normals stay as they were until catchUp(), or
  // until the mesh is no longer cageOnly (the next pump). Asked on the owner's thread.
  std::function<bool(int mesh)> cageOnly;

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

  // Applies pending morph changes to the drawn vertices of the meshes (to the cages of cageOnly ones).
  void flush();
  // Brings the drawn vertices of cageOnly meshes up to their morphs (an export reads them); reported
  // as changes.
  void catchUp();

  // The vertices changed since last asked; clears them.
  std::vector<MeshChange> takeChanges();

  // Active channels first, then by group and label.
  void sort();

  // Counts up when channels were added, removed or relabelled.
  int listVersion() const { return m_listVersion; }
  // Bumped when a dial's value or a channel's applied weight changes (a driver's correctives
  // included), so a reader republishes the values only then. The list's own changes are listVersion().
  int version() const { return m_version; }
  // Loads that failed, worth telling the user about (missing files are reported separately).
  std::vector<std::string> takeErrors();

  // Rebuilds the vertex buffers from the base the next time they flush (a fit changed the base).
  void markRebuild(int mesh);

  // Registers a follower and fits it to every morph already dialled in.
  void addFollower(Follower f);
  // Forgets meshes that left the scene: their followers, CPU state and bindings. Channels left with
  // nothing to move and nothing to load are removed.
  void removeMeshes(const std::set<int>& meshes);
  // Removes channels outright, loaded or not.
  void removeChannels(const std::vector<std::string>& names);

  // The targets applied to a mesh now and the weight of each, except `except` (a mesh's own drape
  // is left out when a simulation reads the shape underneath it).
  std::vector<std::pair<dson::TargetPtr, double>> activeTargets(int mesh, const mesh::RenderTarget* except = nullptr) const;
  // The meshes a channel's targets move (none while it is not loaded).
  std::set<int> meshesOf(const std::string& name) const;
  // The target a channel binds on a mesh, or null.
  dson::TargetPtr cpuTarget(const std::string& name, int mesh) const;
  // The mesh's cage with every active target's cage deltas applied, except `except` (a mesh's own
  // drape or collision correction is left out when a simulation reads the shape underneath it).
  std::vector<float> morphedCage(int mesh, const mesh::RenderTarget* except = nullptr) const;
  // Binds `target` as the channel's target on a mesh, making the channel when it is new and
  // dropping the target it had there. The channel's weight applies to it at once.
  void replaceCpuTarget(const std::string& name, int mesh, dson::TargetPtr target, const ChannelOptions& opts = {});

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
    // a cageOnly mesh's morphed cage (empty: not kept), the weights it holds, and whether the drawn
    // buffers are behind `active`
    std::vector<float>                                    cage;
    std::unordered_map<const mesh::RenderTarget*, Active> cageApplied;
    int                                                   sinceCageRebuild = 0;
    bool                                                  behind           = false;
  };

  struct FollowerState
  {
    Follower f;
    // the source target (by identity) -> its projection, null for one that moves nothing
    // (the source is kept with it: a freed target's address could come back for another)
    std::unordered_map<const mesh::RenderTarget*, std::pair<dson::TargetPtr, dson::TargetPtr>> projected;
    struct Own
    {
      enum class State { ePending, eNone, eReady } state = State::eNone;
      std::shared_future<std::vector<dson::TargetPtr>> future;
      std::vector<dson::TargetPtr>                      targets;
    };
    std::unordered_map<std::string, Own>                           own;
    // the targets applied to the follower, per channel
    std::unordered_map<std::string, std::vector<dson::TargetPtr>> applied;
  };
  bool follow(FollowerState& st, const std::string& name, double value);
  // makes, side by side, the projections that applying `values` will ask the followers for
  void projectAhead(const std::vector<std::pair<std::string, double>>& values);
  std::vector<dson::TargetPtr> followerTargets(FollowerState& st, const std::string& name, const std::vector<Binding>& sources);

  Channel* channelMut(const std::string& name);
  void     reindex();
  void     bind(const std::string& name, Binding b, double value, const ChannelOptions& opts);
  MeshState& ensureCpu(int mesh);
  // writes a channel's weight into its bindings; returns false for an unknown channel
  bool apply(const std::string& name, double value);
  void ensureLoaded(const std::string& name);
  void finishLoad(const std::string& name, std::future<Finish>& fut);
  void runDriver();
  // What a flush did to one mesh's buffers: the whole mesh, or the vertices moved and re-welded.
  // Computed per mesh on the pool (a dial moves a body and everything that follows it: a dozen
  // meshes, each a delta sum and a normal weld), then recorded on the owner's thread.
  struct MeshUpdate
  {
    int                   mesh    = -1;
    bool                  changed = false;
    bool                  all     = false;
    std::vector<uint32_t> moved, written;
  };
  MeshUpdate rebuild(int mesh, MeshState& st);
  MeshUpdate update(int mesh, MeshState& st);
  // a cageOnly mesh's cage brought to `active`: each changed target's cage deltas, or every target
  // over the base cage when it is not kept yet or is due a rebuild
  void       updateCage(int mesh, MeshState& st);
  // the drawn vertices of the meshes behind their morphs that `which` names brought up to them
  void       catchUp(const std::function<bool(int mesh)>& which);
  void       change(int mesh, bool all, const std::vector<uint32_t>& moved, const std::vector<uint32_t>& written, bool cage = false,
                    bool caughtUp = false);

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
  int                                                   m_version     = 0;
  std::vector<std::string>                              m_errors;
  std::map<int, MeshChange>                             m_changes;
  std::vector<std::unique_ptr<FollowerState>>           m_followers;
};

}  // namespace fv::morph
