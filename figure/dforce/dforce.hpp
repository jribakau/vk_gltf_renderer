#pragma once

// dForce on the figure: drapes the worn dynamic surfaces and writes the drape back as a hidden
// morph. Two ways of running:
//
// - Settle (the default): one settle for the pose, scheduled once the pose or shape stops changing.
//   A newer pose, morph or wear bumps the generation, and a finished job applies only if it is still
//   that generation.
//   With `makeGpuSim` it runs on the renderer's compute solver instead, a slice a frame in pump(), so
//   the app stays usable and an edit drops it at once; if that solver fails, the job goes to the pool.
// - Live: the simulation keeps its state and takes the figure's bones every frame, stepping once on
//   the pool and writing the drape the frame after, so the cloth follows a drag or a playing clip.
//   It starts with a settle, and builds itself again (after a short wait) when a dial changes a rest
//   shape or something is worn or taken off.
// - Live on the GPU: the same, on a simulation the host provides through `makeGpuSim` (the
//   renderer's compute solver). One that must stay on its thread is stepped in pump() itself, so
//   its drape is written the same frame.
//
// The drape is a hidden channel per mesh ("dforce:<mesh>", weight 1) holding the cage deltas whose
// skinning gives the simulated shape, so the renderer sees it as any other morph, and turning
// dForce off sets the channels to 0.
//
// One thread (the figure's owner) calls everything; only the solver runs elsewhere, on a snapshot
// of what it needs (a live simulation is touched by the owner only between its steps).

#include <array>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <figure/math/xform.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dforce {

class ClothBackend;
struct SettleMesh;
struct SettleOptions;

// The hidden channel a mesh's drape is stored as.
std::string channelOf(int mesh);

enum class Mode
{
  eSettle,
  eLive,
  eLiveGpu,
};

// Makes a simulation for live mode on the GPU; null when the build has none (live then runs on the CPU).
using MakeSim = std::function<std::shared_ptr<ClothBackend>(std::vector<SettleMesh>, const SettleOptions&)>;

class Dforce
{
public:
  // `bodyMesh` is the figure's body (always a collider, under every cloth layer); -1 without one.
  // `root` is the figure's root node: the settle runs in its space (centimetres, the figure upright).
  Dforce(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root);
  ~Dforce();

  // True while a clip is playing: in settle mode the last drape stays until playback stops.
  std::function<bool()> playing;
  // The time live steps are measured by; unset: the steady clock. For a test.
  std::function<std::chrono::steady_clock::time_point()> clock;
  // The GPU simulation, when the renderer offers one.
  MakeSim makeGpuSim;
  // Whether a shown mesh with no simulation modifier is a collider the cloth meets (a belt, a bra,
  // shoes): DAZ collides with every visible object. Unset: every one.
  std::function<bool(int mesh)> collidesWith;
  // DAZ's skinning (eight influences, the file's dual quaternion share), as the figure's drawn meshes
  // are skinned (figureDazSkinning, gpu-figures-plan.md F3); off: four, linear
  bool dazSkinning = false;

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
  // Where the figure is placed in the world on top of its scene graph (the host's entity), so the
  // simulation knows which way is down. A change that matters settles again, or wakes a live one.
  void setPlacement(const Mat4& placed);
  // Weight 1 drapes, weight 0 restores the conforming meshes. Turning it on settles again.
  void setEnabled(bool on);
  // Settle or live; switching to live starts with a settle. Live on the GPU without `makeGpuSim`
  // runs on the CPU.
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

  // What a settle would be built from now, for a tool that runs the solver itself: the meshes in
  // the job's space, slot by slot, and which slots are drapes (mesh, slot). None when nothing drapes.
  struct Prepared
  {
    std::vector<SettleMesh>          bodies;
    std::vector<int>                 slotMesh;
    std::vector<std::pair<int, int>> drapes;
  };
  std::optional<Prepared> prepared() const;

private:
  struct Job;
  std::vector<int>     dynamics() const;
  std::shared_ptr<Job> prepare(uint64_t generation) const;
  bool                 apply(const Job& job, const std::vector<std::vector<float>>& positions);
  // The bones of a job's skinned meshes as the figure has them now, per slot (empty: unskinned).
  std::vector<std::vector<Mat4>> currentBones(const Job& job) const;
  // World down, in cm/s^2, in the job's space.
  std::array<float, 3> currentGravity() const;
  bool                           pumpLive();

  SceneGraph&      m_scene;
  morph::MorphSet& m_morphs;
  int              m_bodyMesh;
  int              m_root;
  Mat4             m_placed;
  // the gravity the last settle was asked for
  std::array<float, 3> m_settleGravity{0, 0, 0};

  bool                                  m_enabled   = true;
  Mode                                  m_mode      = Mode::eSettle;
  bool                                  m_scheduled = false;
  std::chrono::steady_clock::time_point m_due;
  uint64_t                              m_gen = 0;
  // the settle on the pool: its snapshot and the positions it made
  std::shared_ptr<Job>                           m_snapshot;
  std::future<std::vector<std::vector<float>>>   m_job;
  // a settle on the GPU, a slice a frame on this thread; m_snapshot is its job
  std::shared_ptr<ClothBackend>                  m_settling;
  uint64_t                                       m_jobGen = 0;
  // live: the simulation, what it was built from, the step in flight and the bones it was given
  std::shared_ptr<ClothBackend>                  m_live;
  std::shared_ptr<Job>                           m_liveJob;
  std::future<std::vector<std::vector<float>>>   m_liveStep;
  std::vector<std::vector<Mat4>>                 m_liveBones;
  bool                                           m_liveSettling = false;
  bool                                           m_liveMoving   = true;
  std::array<float, 3>                           m_liveGravity{0, 0, 0};
  // live: when the last step started, and the simulated seconds since the pose last moved
  std::chrono::steady_clock::time_point          m_liveLast;
  float                                          m_liveStill = 0;
  uint64_t                                       m_liveGen      = 0;
  std::set<int> m_draped;
  int           m_version = 0;
};

}  // namespace fv::dforce
