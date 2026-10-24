#pragma once

// dForce on the figure: drapes the worn dynamic surfaces and writes the drape back as a hidden
// morph. Two ways of running:
//
// - Settle (the default): one settle for the pose, scheduled once the pose or shape stops changing.
//   A newer pose, morph or wear bumps the generation, and a finished job applies only if it is still
//   that generation.
// - Live: the simulation keeps its state and takes the figure's bones every frame, stepping once on
//   the pool and writing the drape the frame after, so the cloth follows a drag or a playing clip.
//   It starts with a settle, and builds itself again (after a short wait) when a dial changes a rest
//   shape or something is worn or taken off.
//
// The drape is a hidden channel per mesh ("dforce:<mesh>", weight 1) holding the cage deltas whose
// skinning gives the simulated shape, so the renderer sees it as any other morph, and turning
// dForce off sets the channels to 0.
//
// One thread (the figure's owner) calls everything; only the solver runs elsewhere, on a snapshot
// of what it needs (a live simulation is touched by the owner only between its steps).

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dforce {

class ClothSim;

// The hidden channel a mesh's drape is stored as.
std::string channelOf(int mesh);

enum class Mode
{
  eSettle,
  eLive,
};

class Dforce
{
public:
  // `bodyMesh` is the figure's body (always a collider, under every cloth layer); -1 without one.
  // `root` is the figure's root node: the settle runs in its space (centimetres, the figure upright).
  Dforce(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root);
  ~Dforce();

  // True while a clip is playing: in settle mode the last drape stays until playback stops.
  std::function<bool()> playing;

  // Whether anything worn would be draped.
  bool active() const;
  bool enabled() const { return m_enabled; }
  Mode mode() const { return m_mode; }
  // A settle is waiting out its debounce or running (in live mode: the simulation is being built).
  bool busy() const;

  // Schedules a settle once the pose or shape stops changing; drops one that is running. In live
  // mode, rebuilds the simulation once things stop changing.
  void poke();
  // Something the figure shows changed: nodes (bones) moved, or these meshes' vertices did. Pokes
  // (live: a moved bone is read next frame, a changed mesh rebuilds).
  // The host reports what it flushed before pump() writes a drape, so a drape is never its own cause.
  void changed(const std::vector<int>& meshes, bool nodesMoved);
  // Runs now, still dropping the result if a newer edit arrives while it runs.
  void resimulate();
  // Weight 1 drapes, weight 0 restores the conforming meshes. Turning it on settles again.
  void setEnabled(bool on);
  // Settle or live; switching to live starts with a settle.
  void setMode(Mode mode);
  // Drops a pending or running settle (live: the simulation).
  void cancel();

  // Starts a settle that is due and writes one that finished; in live mode, writes the last step
  // and starts the next. Call every frame. True when it wrote drapes (the morph set has vertices
  // to flush).
  bool pump();

  // What happened since last asked: bumps whenever busy() or the drape changes, for a status topic.
  // A live step does not bump it.
  int version() const { return m_version; }
  // Meshes holding a drape now.
  int draped() const { return int(m_draped.size()); }

private:
  struct Job;
  std::vector<int>     dynamics() const;
  std::shared_ptr<Job> prepare(uint64_t generation) const;
  bool                 apply(const Job& job, const std::vector<std::vector<float>>& positions);
  // The bones of a job's skinned meshes as the figure has them now, per slot (empty: unskinned).
  std::vector<std::vector<Mat4>> currentBones(const Job& job) const;
  bool                           pumpLive();

  SceneGraph&      m_scene;
  morph::MorphSet& m_morphs;
  int              m_bodyMesh;
  int              m_root;

  bool                                  m_enabled   = true;
  Mode                                  m_mode      = Mode::eSettle;
  bool                                  m_scheduled = false;
  std::chrono::steady_clock::time_point m_due;
  uint64_t                              m_gen = 0;
  // the settle on the pool: its snapshot and the positions it made
  std::shared_ptr<Job>                           m_snapshot;
  std::future<std::vector<std::vector<float>>>   m_job;
  uint64_t                                       m_jobGen = 0;
  // live: the simulation, what it was built from, the step in flight and the bones it was given
  std::shared_ptr<ClothSim>                      m_live;
  std::shared_ptr<Job>                           m_liveJob;
  std::future<std::vector<std::vector<float>>>   m_liveStep;
  std::vector<std::vector<Mat4>>                 m_liveBones;
  bool                                           m_liveSettling = false;
  bool                                           m_liveMoving   = true;
  uint64_t                                       m_liveGen      = 0;
  std::set<int> m_draped;
  int           m_version = 0;
};

}  // namespace fv::dforce
