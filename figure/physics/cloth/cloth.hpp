#pragma once

// Cloth on the figure (physics-plan step 15: what was dForce's driver): drapes the worn surfaces DAZ's
// dForce settings make dynamic, on the physics core's cloth island or the renderer's GPU solver, and
// writes the drape back as a hidden morph. The wire, the scene file and the channels keep the name
// "dforce": they name DAZ's content, not this solver. Two ways of running:
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
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <figure/physics/hair/strands.hpp>
#include <figure/physics/cloth/coarse.hpp>
#include <figure/physics/cloth/leash.hpp>
#include <figure/physics/cloth/ride.hpp>
#include <figure/math/xform.hpp>
#include <figure/morph/morph_set.hpp>
#include <figure/scene/graph.hpp>

namespace fv::physics::cloth {
struct Surface;
struct Options;
class Backend;
}  // namespace fv::physics::cloth

namespace fv::physics {

struct SkinPose;

// The hidden channel a mesh's drape is stored as.
std::string clothChannel(int mesh);

enum class ClothMode
{
  eSettle,
  eLive,
  eLiveGpu,
};

// Makes a simulation for live mode on the GPU; null when the build has none (live then runs on the CPU).
using MakeClothSim = std::function<std::shared_ptr<cloth::Backend>(std::vector<cloth::Surface>, const cloth::Options&)>;

class Cloth
{
public:
  // `bodyMesh` is the figure's body (always a collider, under every cloth layer); -1 without one.
  // `root` is the figure's root node: the settle runs in its space (centimetres, the figure upright).
  Cloth(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh, int root);
  ~Cloth();

  // True while a clip is playing: in settle mode the last drape stays until playback stops.
  std::function<bool()> playing;
  // An upstream pass a settle waits for (the body's contact dents): while it says true, no settle
  // starts, so the cloth drapes the body as it will be.
  std::function<bool()> waitWhile;
  // The time live steps are measured by; unset: the steady clock. For a test.
  std::function<std::chrono::steady_clock::time_point()> clock;
  // Called on the pool as a live step's drapes are about to be made from it, after its simulation is free
  // for the next; unset: nothing. For a test, to hold them.
  std::function<void()> beforeLiveDrapes;
  // The GPU simulation, when the renderer offers one.
  MakeClothSim makeGpuSim;
  // Whether a shown mesh with no simulation modifier is a collider the cloth meets (a belt, a bra,
  // shoes): DAZ collides with every visible object. Unset: every one.
  std::function<bool(int mesh)> collidesWith;
  // A collider much finer than the cloth (a necklace's links, lace) is met as a coarse copy of it, as
  // fine as the cloth (physics/cloth/coarse.hpp); off, as it is, for an A/B.
  bool coarseColliders = true;
  // Light cloth: a cloth much finer than this simulates as a coarse copy of itself with faces this wide
  // (cm), its own vertices riding on it (physics/cloth/coarse.hpp's coarseCloth). Off: as it is. Read by
  // the next drape; setLight() also makes one (a tool sets the field alone)
  bool  lightCloth     = false;
  float lightClothEdge = 2.f;
  // Light cloth on or off, and the drape made again that way if dForce is on (live: the simulation).
  void setLight(bool on);
  bool light() const { return lightCloth; }

  // Whether anything worn would be draped.
  bool active() const;
  bool enabled() const { return m_enabled; }
  ClothMode mode() const { return m_mode; }
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
  void setMode(ClothMode mode);
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
    std::vector<cloth::Surface>          bodies;
    std::vector<int>                 slotMesh;
    std::vector<std::pair<int, int>> drapes;
    // strand hair that falls: its mesh and its strands, against `bodies`
    std::vector<std::pair<int, hair::StrandSet>> hairs;
    // per drape: its light copy, which is in `bodies` in its place (null: the cloth itself)
    std::vector<std::shared_ptr<const physics::cloth::CoarseCloth>> light;
  };
  std::optional<Prepared> prepared() const;

private:
  struct Job;
  std::vector<int>     dynamics() const;
  std::shared_ptr<Job> prepare(uint64_t generation) const;
  bool                 apply(const Job& job, const std::vector<std::vector<float>>& positions);
  // A job's drapes from its positions: each one's cage deltas, in job.drapes' order. Any thread. `rests` and
  // `poses`, drape by drape, in place of the job's: what a live step's deltas are taken against and read
  // back through, as they were when it began (the job's change as the next one does). `rides`: each drape
  // bound where it lies to the body where the job left it (physics/cloth/ride.hpp), for Live to carry it
  // between steps; with `hints`, the last rides, each vertex is found again on the face it had.
  // With `leashes`, each drape's rides as leashes (physics/cloth/leash.hpp): what is drawn until the next
  // step lands is held out of the body by them.
  static std::vector<std::vector<float>> deltasOf(const Job& job, const std::vector<std::vector<float>>& positions,
                                                  const std::vector<std::vector<float>>* rests = nullptr,
                                                  const std::vector<SkinPose>*           poses = nullptr,
                                                  std::vector<cloth::Ride>*              rides = nullptr,
                                                  const std::vector<cloth::Ride>*        hints = nullptr,
                                                  std::vector<cloth::Leashes>*           leashes = nullptr);
  // Live: the cloth tied to the body where a step (or the settle that began live) left it, `positions` as
  // the simulation has them, for the next step (physics/cloth/leash.hpp, cloth::tieTo). `rides`: the last
  // ties' rides, found again on their faces unless `fresh`; given back as these. Any thread, the simulation
  // free. How many vertices it tied.
  static size_t tieLive(const Job& job, cloth::Backend& sim, const std::vector<std::vector<float>>& positions, std::vector<cloth::Ride>& rides,
                        bool fresh);
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
  // Live: `advance` (a step, or the settle that begins live) run on `sim` on the pool, then its drapes made
  // from where it left the cloth, as a run at the back of m_liveRuns.
  void                           launchLive(std::shared_ptr<cloth::Backend> sim, std::function<void(cloth::Backend&)> advance, bool settle);
  // Live: the drapes a frame further from what was shown when the last step landed toward that step's.
  // True when it wrote.
  bool                           easeLive(std::chrono::steady_clock::time_point now);
  // Live: a mesh's rest cage as the simulation skins it, in its own space: a drape's without its drape.
  std::vector<float>             liveRest(int mesh, bool drape) const;
  // Live: nothing shown any more (the simulation let go or made again): no ease, no ride.
  void                           forgetShown();
  // Live: the body's bones now, as its slot is skinned (empty: no body, or none skinned).
  std::vector<Mat4>              liveBodyBones() const;
  // Live: what each drape shows, the body at `bones`: `eased` (the deltas the ease gives, `a` of the way
  // from what was shown as the last step landed to that step's), its riding vertices carried by their
  // faces from as far along between those.
  std::vector<std::vector<float>> rideLive(const std::vector<std::vector<float>>& eased, float a, const std::vector<Mat4>& bones) const;
  // Live: the offsets on each ride's faces, the body at `bones`, of where `shown` (deltas) puts its
  // riding vertices: where the ease sets out from.
  void                           rideFrom(const std::vector<std::vector<float>>& shown, const std::vector<Mat4>& bones);
  // Live: what each drape shows (`shown`, deltas), the body at `bones`, held out of the body by the last
  // landed step's leashes (m_leashes): within its reach of where it lay, out of the backstop under its face.
  void                           holdLive(std::vector<std::vector<float>>& shown, const std::vector<Mat4>& bones) const;

  SceneGraph&      m_scene;
  morph::MorphSet& m_morphs;
  int              m_bodyMesh;
  int              m_root;
  Mat4             m_placed;
  // the gravity the last settle was asked for
  std::array<float, 3> m_settleGravity{0, 0, 0};

  bool                                  m_enabled   = true;
  ClothMode                                  m_mode      = ClothMode::eSettle;
  bool                                  m_scheduled = false;
  std::chrono::steady_clock::time_point m_due;
  uint64_t                              m_gen = 0;
  // the settle on the pool: its snapshot and the positions it made
  std::shared_ptr<Job>                           m_snapshot;
  std::future<std::vector<std::vector<float>>>   m_job;
  // a settle on the GPU, a slice a frame on this thread; m_snapshot is its job
  std::shared_ptr<cloth::Backend>                  m_settling;
  // the GPU solver of the last settle, kept for the next (cloth::Backend::restart): for the same garments
  // and shapes it keeps its model and buffers. It goes when dForce is turned off or the mode changes
  std::shared_ptr<cloth::Backend>                  m_gpuSim;
  bool                                           m_skippedPlaying = false;  // an edit came while a clip played
  std::shared_ptr<std::atomic<bool>>             m_jobStale;                // m_job's: an edit has made it out of date
  bool                                           m_jobOnGpu = false;        // m_job runs on the GPU solver
  uint64_t                                       m_jobGen = 0;
  // strand hair's settle, on the pool beside the cloth's: the positions per hair of its snapshot
  std::shared_ptr<Job>                           m_hairSnapshot;
  std::future<std::vector<std::vector<float>>>   m_hairJob;
  std::shared_ptr<std::atomic<bool>>             m_hairStale;
  uint64_t                                       m_hairGen = 0;
  // live: the simulation, what it was built from, and the bones it was given
  std::shared_ptr<cloth::Backend>                  m_live;
  std::shared_ptr<Job>                           m_liveJob;
  // Live: the steps in flight, in the order they began, each the step and then its drapes' deltas on the
  // pool: `stepped` is ready once the simulation is free for the next (its positions taken), `drapes` once
  // the drapes are made from them. The next step begins while the last one's drapes are made, so a light
  // copy, whose drapes take as long as its step, steps every frame rather than every other. At most two are
  // in flight: one stepping, one making its drapes. A run of a job since made again lands and is dropped
  // A run's drapes: each one's deltas, and where it rides the body (physics/cloth/ride.hpp)
  struct LiveDrapes
  {
    std::vector<std::vector<float>> deltas;
    std::vector<cloth::Ride>        rides;
    // and its rides as leashes (deltasOf's)
    std::vector<cloth::Leashes>     leashes;
  };
  struct LiveRun
  {
    std::future<void>                                      stepped;
    std::future<LiveDrapes>                                drapes;
    std::shared_ptr<Job>                                   job;
    // what its deltas are taken against: each drape's rest as the step began (the job's moves on with
    // the next step's), so the drapes are shifted by as much as the rest moved since
    std::shared_ptr<const std::vector<std::vector<float>>> rests;
    // the body's bones it was simulated at, which its rides were bound at
    std::vector<Mat4>                                      bodyBones;
    std::chrono::steady_clock::time_point                  began;
    // the settle that began live: written at once, and the drape no longer busy
    bool                                                   settle = false;
  };
  std::deque<LiveRun>                            m_liveRuns;
  std::vector<std::vector<Mat4>>                 m_liveBones;
  bool                                           m_liveSettling = false;
  bool                                           m_liveMoving   = true;
  std::array<float, 3>                           m_liveGravity{0, 0, 0};
  // live: when the last step started, and the simulated seconds since the pose last moved
  std::chrono::steady_clock::time_point          m_liveLast;
  // live: when the last run's drapes landed; the next is eased in over the time since then, or since it
  // began if that is later (as long as it took, when one waits for the other)
  std::chrono::steady_clock::time_point          m_liveLanded;
  float                                          m_liveStill = 0;
  uint64_t                                       m_liveGen      = 0;
  // live: meshes whose rest shapes moved since the last step (correctives, a dent), and whether the next
  // step has a moved rest shape to take though no bone moved
  std::set<int>                                  m_liveReshaped;
  bool                                           m_liveRestMoved = false;
  // live: a step's drapes aren't written at once when it lands but eased in from what was shown, over the
  // time since the last landed (see m_liveLanded), so the next lands as this one is in (beside the path tracer a step lands
  // every two to four frames, and written at once the cloth jumped as often). Its deltas, those shown
  // when it landed, those shown now; when it landed, and over how long (0: nothing to ease)
  std::vector<std::vector<float>>                m_easeTo, m_easeFrom, m_shown;
  std::chrono::steady_clock::time_point          m_easeStart;
  float                                          m_easeFor = 0;
  // live: what is shown near the body rides it until the next step lands (physics/cloth/ride.hpp). Each
  // drape's ride from the last landed run, and the offsets on its faces of what was shown as that landed
  // (where the ease sets out); the body's bones that run met, then, and as last written. The rides are
  // shared with the next step's bind on the pool, which finds each vertex again on its face, and every
  // kLiveRebind-th step looks afresh (m_liveBinds counts them)
  std::shared_ptr<const std::vector<cloth::Ride>> m_rides;
  std::vector<std::vector<float>>                m_rideFrom;
  std::vector<Mat4>                              m_rideBones, m_rideLanded, m_rideWritten;
  int                                            m_liveBinds = 0;
  // live: each drape's leashes from the last landed run, what is drawn is held out of the body by (holdLive)
  std::shared_ptr<const std::vector<cloth::Leashes>> m_leashes;
  // live: the simulation's last ties' rides, per drape (tieLive's), shared with the runs on the pool, which
  // tie one after another: each step's from where the one before left the cloth
  std::shared_ptr<std::vector<cloth::Ride>>          m_tieRides;
  // live: the faces across each side of each of the body's faces (cloth::facesAcross), for the job they are of
  mutable std::shared_ptr<const std::vector<int>>   m_bodyAcross;
  mutable const Job*                                 m_bodyAcrossOf = nullptr;
  std::set<int> m_draped;
  int           m_version = 0;
  // the coarse copies of colliders the last job made, by the shape each was made from (its rest cage,
  // faces, and the cloth's edge): a rebuild for the same shapes doesn't simplify them again (188 ms on
  // a necklace, underwear and boots). None: one that isn't copied
  mutable std::map<uint64_t, std::optional<physics::cloth::Coarse>> m_coarse;
  // and the light copies of cloth, the same way (null: one that isn't copied)
  mutable std::map<uint64_t, std::shared_ptr<const physics::cloth::CoarseCloth>> m_light;
};

}  // namespace fv::physics
