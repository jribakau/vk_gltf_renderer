#pragma once

// dForce on the figure: drapes the worn dynamic surfaces and writes the drape back as a hidden
// morph. Two ways of running:
//
// - Settle (the default): one settle for the pose, scheduled once the pose or shape stops changing.
//   A newer pose, morph or wear bumps the generation, and a finished job applies only if it is still
//   that generation.
//   With `makeGpuSim` it runs on the renderer's compute solver instead: on the pool too when that solver has a queue of its own,
//   else made on the pool and then a slice a frame in pump(), so the app stays usable and an edit drops it at once; if that
//   solver fails, the job goes to the CPU on the pool.
// - Live: the simulation keeps its state and takes the figure's bones every frame, stepping once on
//   the pool and writing the drape the frame after, so the cloth follows a drag or a playing clip.
//   It starts with a settle, and builds itself again (after a short wait) when a dial changes a cloth's
//   rest shape or something is worn or taken off. A rest shape the pose moves (its correctives, every
//   frame of a clip) or a collider's (a dent, a dial on the body) is taken in place by the next step.
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
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <figure/dforce/strands.hpp>
#include <figure/physics/cloth/coarse.hpp>
#include <figure/math/xform.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics::cloth {
struct Surface;
struct Options;
}  // namespace fv::physics::cloth

namespace fv::dforce {

class ClothBackend;
using SettleMesh    = physics::cloth::Surface;
using SettleOptions = physics::cloth::Options;

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
  // An upstream pass a settle waits for (the body's contact dents): while it says true, no settle
  // starts, so the cloth drapes the body as it will be.
  std::function<bool()> waitWhile;
  // The time live steps are measured by; unset: the steady clock. For a test.
  std::function<std::chrono::steady_clock::time_point()> clock;
  // The GPU simulation, when the renderer offers one.
  MakeSim makeGpuSim;
  // Whether a shown mesh with no simulation modifier is a collider the cloth meets (a belt, a bra,
  // shoes): DAZ collides with every visible object. Unset: every one.
  std::function<bool(int mesh)> collidesWith;
  // A collider much finer than the cloth (a necklace's links, lace) is met as a coarse copy of it, as
  // fine as the cloth (physics/cloth/coarse.hpp); off, as it is, for an A/B.
  bool coarseColliders = true;
  // Light cloth: a cloth much finer than this simulates as a coarse copy of itself with faces this wide
  // (cm), its own vertices riding on it (physics/cloth/coarse.hpp's coarseCloth). Off: as it is.
  bool  lightCloth     = false;
  float lightClothEdge = 2.f;

  // Whether anything worn would be draped.
  bool active() const;
  bool enabled() const { return m_enabled; }
  Mode mode() const { return m_mode; }
  // A settle is waiting out its debounce or running (in live mode: the simulation is being built).
  // A settle runs on the GPU solver: the still has little to gain from the GPU meanwhile.
  bool settlingOnGpu() const;
  bool busy() const;

  // Schedules a settle once the pose or shape stops changing; drops one that is running. In live
  // mode, rebuilds the simulation once things stop changing.
  void poke();
  // Something the figure shows changed: nodes (bones) moved, or these meshes' vertices did. Pokes
  // (live: a moved bone is read next frame; a mesh changed with them, the pose's correctives, or a
  // collider's, is read in place by the next step; a cloth's own changed with nothing moving rebuilds).
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
    // strand hair that falls: its mesh and its strands, against `bodies`
    std::vector<std::pair<int, StrandSet>> hairs;
    // per drape: its light copy, which is in `bodies` in its place (null: the cloth itself)
    std::vector<std::shared_ptr<const physics::cloth::CoarseCloth>> light;
  };
  std::optional<Prepared> prepared() const;

private:
  struct Job;
  std::vector<int>     dynamics() const;
  std::shared_ptr<Job> prepare(uint64_t generation) const;
  bool                 apply(const Job& job, const std::vector<std::vector<float>>& positions);
  // A job's drapes from its positions: each one's cage deltas, in job.drapes' order. Any thread.
  static std::vector<std::vector<float>> deltasOf(const Job& job, const std::vector<std::vector<float>>& positions);
  // Writes them, but not on an item taken off meanwhile. True when it wrote.
  bool writeDeltas(const Job& job, const std::vector<std::vector<float>>& deltas);
  // The bones of a job's skinned meshes as the figure has them now, per slot (empty: unskinned).
  std::vector<std::vector<Mat4>> currentBones(const Job& job) const;
  // Holds this drape on a mesh: cage deltas, xyz per cage vertex. True when written.
  void dropStale();
  bool writeDrape(int mesh, const std::vector<float>& delta);
  // World down, in cm/s^2, in the job's space.
  std::array<float, 3> currentGravity() const;
  bool                           pumpLive();
  // Live: a mesh's rest cage as the simulation skins it, in its own space: a drape's without its drape.
  std::vector<float>             liveRest(int mesh, bool drape) const;

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
  // the GPU solver of the last settle, kept for the next (ClothBackend::restart): for the same garments
  // and shapes it keeps its model and buffers. It goes when dForce is turned off or the mode changes
  std::shared_ptr<ClothBackend>                  m_gpuSim;
  bool                                           m_skippedPlaying = false;  // an edit came while a clip played
  std::shared_ptr<std::atomic<bool>>             m_jobStale;                // m_job's: an edit has made it out of date
  bool                                           m_jobOnGpu = false;        // m_job runs on the GPU solver
  uint64_t                                       m_jobGen = 0;
  // strand hair's settle, on the pool beside the cloth's: the positions per hair of its snapshot
  std::shared_ptr<Job>                           m_hairSnapshot;
  std::future<std::vector<std::vector<float>>>   m_hairJob;
  std::shared_ptr<std::atomic<bool>>             m_hairStale;
  uint64_t                                       m_hairGen = 0;
  // live: the simulation, what it was built from, the step in flight (its drapes' deltas, made on the
  // pool) and the bones it was given
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
  // live: meshes whose rest shapes moved since the last step (correctives, a dent), and whether the next
  // step has a moved rest shape to take though no bone moved
  std::set<int>                                  m_liveReshaped;
  bool                                           m_liveRestMoved = false;
  std::set<int> m_draped;
  int           m_version = 0;
  // the coarse copies of colliders the last job made, by the shape each was made from (its rest cage,
  // faces, and the cloth's edge): a rebuild for the same shapes doesn't simplify them again (188 ms on
  // a necklace, underwear and boots). None: one that isn't copied
  mutable std::map<uint64_t, std::optional<physics::cloth::Coarse>> m_coarse;
  // and the light copies of cloth, the same way (null: one that isn't copied)
  mutable std::map<uint64_t, std::shared_ptr<const physics::cloth::CoarseCloth>> m_light;
};

}  // namespace fv::dforce
