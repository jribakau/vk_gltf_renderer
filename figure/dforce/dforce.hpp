#pragma once

// dForce on the figure: schedules one settle for the worn dynamic surfaces and writes it back as a
// hidden morph. A newer pose, morph or wear bumps the generation, and a finished job applies only
// if it is still that generation. The settle runs on the thread pool; nothing here runs on a frame
// that has nothing to do. A port of src/lib/dforce/Dforce.ts.
//
// The drape is a hidden channel per mesh ("dforce:<mesh>", weight 1) holding the cage deltas whose
// skinning gives the simulated shape, so the renderer sees it as any other morph, and turning
// dForce off sets the channels to 0.
//
// One thread (the figure's owner) calls everything; only the settle itself runs elsewhere, on a
// snapshot of what it needs.

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <figure/morph/morph_set.hpp>
#include <figure/scene/graph.hpp>

namespace fv::dforce {

// The hidden channel a mesh's drape is stored as.
std::string channelOf(int mesh);

class Dforce
{
public:
  // `bodyMesh` is the figure's body (always a collider, under every cloth layer); -1 without one.
  Dforce(SceneGraph& scene, morph::MorphSet& morphs, int bodyMesh);
  ~Dforce();

  // True while a clip is playing: the last drape stays until playback stops.
  std::function<bool()> playing;

  // Whether anything worn would be draped.
  bool active() const;
  bool enabled() const { return m_enabled; }
  // A settle is waiting out its debounce or running.
  bool busy() const { return m_scheduled || m_job.valid(); }

  // Schedules a settle once the pose or shape stops changing; drops one that is running.
  void poke();
  // Something the figure shows changed: nodes (bones) moved, or these meshes' vertices did. Pokes,
  // except for the echo of a drape just written.
  void changed(const std::vector<int>& meshes, bool nodesMoved);
  // Runs now, still dropping the result if a newer edit arrives while it runs.
  void resimulate();
  // Weight 1 drapes, weight 0 restores the conforming meshes. Turning it on settles again.
  void setEnabled(bool on);
  // Drops a pending or running settle.
  void cancel();

  // Starts a settle that is due and writes one that finished. Call every frame. True when it
  // wrote drapes (the morph set has vertices to flush).
  bool pump();

  // What happened since last asked: bumps whenever busy() or the drape changes, for a status topic.
  int version() const { return m_version; }
  // Meshes holding a drape now.
  int draped() const { return int(m_draped.size()); }

private:
  struct Job;
  std::vector<int> dynamics() const;
  std::shared_ptr<Job> prepare(uint64_t generation) const;
  bool apply(const Job& job, const std::vector<std::vector<float>>& positions);

  SceneGraph&      m_scene;
  morph::MorphSet& m_morphs;
  int              m_bodyMesh;

  bool                                  m_enabled   = true;
  bool                                  m_scheduled = false;
  std::chrono::steady_clock::time_point m_due;
  uint64_t                              m_gen = 0;
  // the settle on the pool: its snapshot and the positions it made
  std::shared_ptr<Job>                           m_snapshot;
  std::future<std::vector<std::vector<float>>>   m_job;
  uint64_t                                       m_jobGen = 0;
  // meshes whose drape was just written: their change is not a reason to settle again
  std::set<int> m_echo;
  std::set<int> m_draped;
  int           m_version = 0;
};

}  // namespace fv::dforce
